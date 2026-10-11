"""Texas Hold'em on Discord (FEAT-33 parts C1a, C1b): `/game start game:holdem`, lobby (Sit, Leave, Rebuy, Fill seats, Assist, Deal, Close table),
table thread, idle timer, restart recovery, and the hand: pinned board, one turn message (Fold, Check / Call, Raise, My cards), action
timer, character seats (a model's choice from the engine menu, else the fallback policy), showdown post, the pause between hands.

Same shape as ``games_doubt_discord``: one lock per table that covers store work only; a handler decides under the lock and builds an
``_Out`` (a private reply, the lobby text and buttons, a sequence number); the messages go out afterwards, one table at a time, and an
edit older than the last one sent is dropped. Timers re-check the stored state under the lock, so a stale timer does nothing.
Money moves only inside the store (buy-in, rebuy, cash-out).
"""
from __future__ import annotations

import asyncio
import functools
import logging
import re
import time
from dataclasses import dataclass, field

import discord

from . import game_chooser
from .admin_store import ConflictError, GameError
from .games import holdem
from .games.base import IllegalMove, Seat
from .games_discord import NO_CHARACTER_SEATS, NO_MENTIONS, NOT_SEATED, NOT_YOUR_TURN, OFF_HERE, SKIPPED, STALE, GameTables
from .game_talk import TalkBudget
from .table_runtime import TableRuntime

PREFIX = "llmcord:holdem"
NAME = "Texas Hold'em"
HOLDEM_OFF = f"{NAME} is off on this server."
BET_NOTE = f"{NAME} uses the server's blinds; you choose your buy-in when you sit."
NO_TABLE = "Use /game hand in your table's thread."
NO_SEAT = "You do not have a seat in this round."
CASHED = " Any chips were cashed out."
THREAD_FORBIDDEN = ("I couldn't open the table thread. I need the Create Public Threads and Send Messages in Threads "
                    "permissions in this channel. The table was closed.")
THREAD_FAILED = "Couldn't open the table thread; the table is closed."
NO_HAND = "No hand is being played right now."


def buy_in_range(rules) -> tuple[int, int]:
    """(min, max) buy-in in currency for a table's rules."""
    bb = 2 * rules["small_blind"]
    return rules["min_buy_in_bb"] * bb, rules["max_buy_in_bb"] * bb


def default_buy_in(rules) -> int:
    """100 big blinds, kept inside the allowed range."""
    low, high = buy_in_range(rules)
    return max(low, min(200 * rules["small_blind"], high))


class HoldemButton(discord.ui.DynamicItem[discord.ui.Button], template=PREFIX + r":(?P<action>[a-z_]+):(?P<table>\d+):(?P<round>\d+):(?P<moves>\d+)"):
    """One button; its custom_id carries the table, the round and a number the action may use. Never a seat."""

    def __init__(self, action: str, table_id: int, round_id: int, moves: int, label: str = "", style=discord.ButtonStyle.secondary, disabled: bool = False):
        super().__init__(discord.ui.Button(label=label, style=style, disabled=disabled, custom_id=f"{PREFIX}:{action}:{table_id}:{round_id}:{moves}"))
        self.action, self.table_id, self.round_id, self.moves = action, table_id, round_id, moves

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["action"], int(match["table"]), int(match["round"]), int(match["moves"]), item.label or "", item.style, item.disabled)

    async def callback(self, interaction: discord.Interaction):
        await interaction.client.holdem.press(interaction, self.action, self.table_id, self.round_id, self.moves)


class AmountModal(discord.ui.Modal):
    """One number to type: the buy-in when sitting, the amount of a rebuy or an exact raise. Bound to the server, table and round it was opened for."""

    def __init__(self, kind: str, guild_id: int, table_id: int, round_id: int, title: str, label: str, placeholder: str, default: str = "", moves: int = 0):
        super().__init__(title=title[:45], timeout=600)
        self.kind, self.guild_id, self.table_id, self.round_id, self.moves = kind, guild_id, table_id, round_id, moves
        self.field = discord.ui.TextInput(label=label[:45], placeholder=placeholder[:100], default=default or None, max_length=12, required=True)
        self.add_item(self.field)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.client.holdem.submit(interaction, self)

    async def on_error(self, interaction: discord.Interaction, error: Exception):
        logging.error("Texas Hold'em %s modal failed", self.kind, exc_info=error)
        await interaction.client.holdem._say(interaction, "Something went wrong; try again.")


class _RaisePicker(discord.ui.View):
    """The private raise picker: one button per raise of the engine's menu (each carries its total bet) and Exact amount."""

    def __init__(self, guild_id: int, table_id: int, round_id: int, moves: int, amounts: list, top: int):
        super().__init__(timeout=600)
        self.guild_id, self.table_id, self.round_id, self.moves = guild_id, table_id, round_id, moves
        for amount in amounts:
            button = discord.ui.Button(label=f"All-in {amount:,}" if amount == top else f"Raise to {amount:,}", style=discord.ButtonStyle.danger if amount == top else discord.ButtonStyle.primary)
            button.callback = functools.partial(self.pick, amount=amount)
            self.add_item(button)
        exact = discord.ui.Button(label="Exact amount", style=discord.ButtonStyle.secondary)
        exact.callback = self.exact
        self.add_item(exact)

    async def pick(self, interaction: discord.Interaction, amount: int):
        await interaction.client.holdem.raise_to(interaction, self, amount)

    async def exact(self, interaction: discord.Interaction):
        await interaction.client.holdem.ask_exact(interaction, self)


@dataclass(eq=False)
class _Table:
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    issued: int = 0  # numbers the outgoing edits, assigned under the table lock
    sent: dict = field(default_factory=dict)  # newest delivered number per message kind
    deadlines: dict = field(default_factory=dict)  # lobby / turn / next -> unix time


@dataclass
class _Out:
    """What a locked step decided: a private reply and the messages to send after the lock is released."""
    say: str | None = None
    key: tuple | None = None
    lobby: tuple | None = None  # (content, view or None)
    closing: str | None = None  # the table was closed from outside: this line goes to the thread
    closed: bool = False  # the table is over: the lobby message is repainted even though the table is closed
    seq: int = 0
    turn: tuple | None = None  # (content, view or None): the turn message, or the pause between hands
    start: str | None = None  # the board text of the table's first hand: post and pin it, then post the turn message
    board: bool = False  # the pinned board is out of date
    board_text: str | None = None  # the pinned board's last text (table over)
    result: tuple | None = None  # (text, view or None): the hand's result post
    summary: str | None = None  # the end-of-table post
    talk: str | None = None  # a public event the seated characters react to (fire and forget, after the posts)


class HoldemTables:
    LOBBY_SECONDS = 300
    NEXT_HAND_SECONDS = 15
    CHARACTER_SECONDS = 2
    ARCHIVE_SECONDS = 600
    BOARD_DEBOUNCE_SECONDS = 1
    RETRY_SECONDS = 5  # read once at construction (TableRuntime.retry_seconds); tests set runtime.retry_seconds
    SWEEP_SECONDS = 30
    MODEL_SECONDS = 20  # bounds one character decision by the model
    EVENT_COOLDOWN_SECONDS = 20

    def __init__(self, bot, clock=time.time):
        self.bot, self.clock = bot, clock
        self.runtime = TableRuntime(label=NAME, retry_seconds=self.RETRY_SECONDS)
        self.tables: dict[tuple[int, int], _Table] = {}
        self.threads: dict[int, tuple[int, int]] = {}  # thread id -> (guild, table) of the tables being played
        self.banter: dict[int, tuple[int, int]] = {}  # thread id -> (guild, table) of closed tables, until the archive timer
        self.event_at: dict[tuple[int, int], float] = {}  # table -> clock time of its last event turn
        self.event_tasks: set[asyncio.Task] = set()
        self.ended: set[tuple[int, int]] = set()  # tables this object closed itself: the sweeper leaves them alone
        self.dirty: set[tuple[int, int]] = set()
        self.board_tasks: dict[tuple[int, int], asyncio.Task] = {}
        self.assists: dict[tuple, str] = {}  # (guild, round, seat, street) -> the odds text, so a street's numbers never change
        self.shown: set[tuple[int, int]] = set()  # (guild, round) whose winner already showed their cards
        self.talk = getattr(bot, 'talk_budget', None) or TalkBudget(lambda: bot.store)  # failed attempts, shared by every table game

    @property
    def talk_failed(self):
        return self.talk.failed

    @property
    def store(self):
        return self.bot.store

    @property
    def closed(self) -> bool:
        return self.runtime.closed

    def table(self, key) -> _Table:
        table = self.tables.get(key)
        if table is None:
            table = self.tables[key] = _Table()
        return table

    # --- timers -----------------------------------------------------------------------------

    def arm(self, key, kind: str, delay: float, *args) -> None:
        if self.closed:
            return
        hook = {"lobby": self.on_lobby_timer, "turn": self.on_turn_timer, "character": self.on_character_timer, "next": self.on_next_timer,
                "archive": self.on_archive_timer}[kind]
        self.runtime.arm(key, kind, delay, self._run, kind, hook, key, args, name=f"holdem-{kind}-{key[1]}")

    async def _run(self, kind, hook, key, args) -> None:
        await hook(*key, *args)
        if kind != "archive":
            await self._reap(key)  # the hook may have found the table closed from outside

    async def close(self) -> None:
        tasks = [*self.board_tasks.values(), *self.event_tasks]
        self.board_tasks.clear()
        for task in tasks:
            task.cancel()
        await self.runtime.close()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _release(self, key) -> None:
        """Stops the table's play timers and drops its thread. The archive timer is armed separately."""
        for kind in ("lobby", "turn", "character", "next"):
            self.runtime.disarm(key, kind)
        self.threads = {thread: k for thread, k in self.threads.items() if k != key}
        self.dirty.discard(key)

    def _forget(self, key) -> None:
        """This object closed the table (in the store): release it, and keep the sweeper off it until its archive timer.
        Its thread keeps its (restricted) table talk until then."""
        self.ended.add(key)
        if (row := self._row(key)) is not None and row["thread_id"] is not None:
            self.banter[row["thread_id"]] = key
        self._release(key)

    def _drop_banter(self, key) -> None:
        self.banter = {thread: k for thread, k in self.banter.items() if k != key}
        self.event_at.pop(key, None)

    def _touch_lobby(self, key, round_id) -> None:
        """A seat taken, left or topped up restarts the idle clock of the lobby."""
        self.table(key).deadlines["lobby"] = self.clock() + self.LOBBY_SECONDS
        self.arm(key, "lobby", self.LOBBY_SECONDS, round_id)

    # --- tables closed from outside (dashboard, /admin games channel) -------------------------

    def _ensure_sweeper(self) -> None:
        if self.tables:
            self.runtime.ensure_sweeper(interval=lambda: self.SWEEP_SECONDS, held=lambda: bool(self._known()), sweep=self.sweep, name="holdem-sweep")

    def _known(self) -> set:
        return {*self.tables, *self.threads.values(), *(k[0] for k in self.runtime.timers if k[1] != "archive")}

    async def sweep(self) -> None:
        """Cleans up tables closed in the store by someone else (dashboard switch, channel removal) while this object still holds them."""
        for key in self._known():
            await self._reap(key)

    def _open_here(self, row) -> bool:
        return row is not None and row["game"] == "holdem" and row["status"] == "open"

    async def _reap(self, key) -> None:
        """If the table is closed in the store and this object did not close it, say so (lobby, thread), archive and forget it."""
        if key in self.ended:
            self._release(key)
            return
        if key not in self.tables and key not in self.threads.values() and not any(k[0] == key for k in self.runtime.timers):
            return
        if self._open_here(self._row(key)):
            return
        async with self.runtime.lock(key):
            out = self._closed_out(key)
        await self._publish(key, out)

    def _closed_out(self, key) -> _Out | None:
        """Under the table lock: the closing messages for a table that is closed in the store, or None (still open, or already handled)."""
        row = self._row(key)
        if key in self.ended or self._open_here(row):
            return None
        if row is None or row["game"] != "holdem":
            self._release(key)
            self.tables.pop(key, None)
            return None
        guild_id = key[0]
        if not self.store.game_enabled(guild_id, "holdem"):
            reason = f"{NAME} was turned off on this server."
        elif row["channel_id"] not in self.store.game_channels(guild_id):
            reason = "games were turned off here."
        else:
            reason = "the table was closed."
        self._forget(key)
        self.arm(key, "archive", self.ARCHIVE_SECONDS)
        text = f"Table closed: {reason}{CASHED}"
        return self._seal(key, _Out(lobby=(text, None), closing=text, closed=True))

    async def close_for_channel(self, guild_id, channel_id, table_id) -> int | None:
        """Turns games off in a channel whose open table is this game's: the store step under the table lock, then the closing messages.
        Returns the id of the table the store closed (None if it had closed already)."""
        key = (guild_id, table_id)
        async with self.runtime.lock(key):
            closed = self.store.set_game_channel(guild_id, channel_id, False)["closed_table"]
            out = self._closed_out(key) if closed == table_id else None
        await self._publish(key, out)
        if closed != table_id:
            await self._reap(key)
            if closed is not None:
                await self.sweep()
        return closed

    # --- messages ---------------------------------------------------------------------------

    async def _channel(self, channel_id):
        return self.bot.get_channel(channel_id) or await self.bot.fetch_channel(channel_id)

    async def _edit(self, channel_id, message_id, **kwargs) -> bool | None:
        """Best-effort edit. False: the message is gone. None: some other failure (logged)."""
        if channel_id is None or message_id is None:
            return None
        try:
            channel = await self._channel(channel_id)
            await channel.get_partial_message(message_id).edit(allowed_mentions=NO_MENTIONS, **kwargs)
            return True
        except discord.NotFound:
            return False
        except discord.HTTPException as error:
            logging.warning("Could not edit Texas Hold'em message %s in channel %s: %s", message_id, channel_id, type(error).__name__)
        except Exception:
            logging.exception("Could not edit Texas Hold'em message %s in channel %s", message_id, channel_id)
        return None

    async def _post(self, channel_id, content, view=None):
        if channel_id is None:
            return None
        try:
            channel = await self._channel(channel_id)
            return await channel.send(content, view=view, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning("Could not post to Texas Hold'em thread %s: %s", channel_id, type(error).__name__)
        except Exception:
            logging.exception("Could not post to Texas Hold'em thread %s", channel_id)
        return None

    async def _say(self, interaction, text: str) -> None:
        """An ephemeral reply: the first response, or a followup once the interaction was acknowledged."""
        try:
            if interaction.response.is_done():
                await interaction.followup.send(text, ephemeral=True, allowed_mentions=NO_MENTIONS)
            else:
                await interaction.response.send_message(text, ephemeral=True, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning("Texas Hold'em reply failed: %s", type(error).__name__)

    async def _ack(self, interaction) -> None:
        try:
            await interaction.response.defer()
        except discord.HTTPException as error:
            logging.warning("Texas Hold'em could not acknowledge the press: %s", type(error).__name__)

    async def _archive(self, thread_id) -> None:
        if thread_id is None:
            return
        try:
            thread = await self._channel(thread_id)
            await thread.edit(archived=True, locked=True)
        except Exception as error:
            logging.warning("Could not archive the Texas Hold'em thread (%s)", type(error).__name__)

    def _row(self, key):
        return self.store.one("SELECT * FROM game_tables WHERE id=? AND guild_id=?", (key[1], key[0]))

    # --- rendering --------------------------------------------------------------------------

    def render_lobby(self, key, snap, row) -> tuple[str, discord.ui.View]:
        rules = snap["rules"] or holdem.normalize_rules(None)
        name, (low, high), sb = self.store.currency_name(key[0]), buy_in_range(rules), rules["small_blind"]
        lines = [f"**{NAME}** · table {key[1]}", f"Blinds {sb}/{2 * sb} · Buy-in {low:,} to {high:,} {name} · {rules['action_timeout']} s per move",
                 f"Host: <@{row['opened_by']}>"]
        if rules["assist"]:
            lines.append("Assist: on")
        lines.append("")
        for seat in snap["seats"]:
            tag = " (house)" if seat["kind"] == "character" and not seat["brought_by"] else ""
            lines.append(f"{seat['index'] + 1}. {GameTables._who(seat)}{tag} — {seat['stack']:,} {name}")
        stamp = f"<t:{int(d)}:R>" if (d := self.table(key).deadlines.get("lobby")) else ""
        lines.append("")
        lines.append(f"Sit with the button. The host deals when {holdem.MIN_SEATS} to {holdem.MAX_SEATS} seats are taken"
                     + (f"; the table closes {stamp} if no hand starts." if stamp else "."))
        view, tid, rid = discord.ui.View(timeout=None), key[1], snap["round_id"]
        spec = [("sit", "Sit", discord.ButtonStyle.primary, 0, False), ("leave", "Leave", discord.ButtonStyle.secondary, 0, False),
                ("rebuy", "Rebuy", discord.ButtonStyle.secondary, 0, False)]
        if self.store.binding(row["channel_id"]) is not None:
            spec.append(("fill", "Fill seats", discord.ButtonStyle.secondary, 0, False))
        if rules["assist"] or self.store.game_rules(key[0], "holdem")["assist"]:
            spec.append(("assist", "Assist off" if rules["assist"] else "Assist on", discord.ButtonStyle.secondary, 0 if rules["assist"] else 1, False))
        spec += [("deal", "Deal", discord.ButtonStyle.success, 0, len(snap["seats"]) < holdem.MIN_SEATS or row["thread_id"] is None),
                 ("close", "Close table", discord.ButtonStyle.danger, 0, False)]
        for action, label, style, moves, disabled in spec:
            view.add_item(HoldemButton(action, tid, rid, moves, label, style, disabled))
        return "\n".join(lines), view

    # --- the common step machinery ----------------------------------------------------------

    def _seal(self, key, out: _Out) -> _Out:
        table = self.table(key)
        table.issued += 1
        out.key, out.seq = key, table.issued
        return out

    def _load(self, key, round_id):
        """((table row, snapshot), None) for an open Hold'em table and one of its rounds, else (None, refusal)."""
        row = self._row(key)
        if not self._open_here(row):
            return None, _Out(STALE)
        try:
            snap = self.store.round_snapshot(key[0], round_id)
        except GameError:
            return None, _Out(STALE)
        if snap["table_id"] != key[1]:
            return None, _Out(STALE)
        return (row, snap), None

    async def _publish(self, key, out: _Out | None) -> None:
        """After the table lock is released: the thread posts, then the edits (an edit older than the last one sent is dropped)."""
        if out is None:
            return
        table = self.table(key)
        async with table.send_lock:
            row = self._row(key)
            if row is None:
                return
            if row["status"] != "open" and not out.closed:
                return  # a step sealed before the table was closed must not repaint the closed table
            thread_id, channel_id = row["thread_id"], row["channel_id"]
            if out.start is not None and not await self._open_play(key, out, row):
                return
            if out.closing:
                await self._edit(thread_id, row["turn_message_id"], content="Table closed.", view=None)
                await self._post(thread_id, out.closing)
            if out.result:
                await self._post(thread_id, *out.result)
            if out.summary:
                await self._post(thread_id, out.summary)
            self._talk_event(thread_id, out)
            if out.turn and out.start is None and out.seq > table.sent.get("turn", 0):
                table.sent["turn"] = out.seq
                await self._turn_message(key, out, row)
            if out.lobby and out.seq > table.sent.get("lobby", 0):
                table.sent["lobby"] = out.seq
                await self._edit(channel_id, row["message_id"], content=out.lobby[0], view=out.lobby[1])
            if out.board_text is not None:
                self.dirty.discard(key)
                await self._edit(thread_id, row["board_message_id"], content=out.board_text)
            elif out.board:
                self._mark_board(key)

    def _talk_event(self, thread_id, out) -> None:
        """Fire and forget: the characters at the table react to a public event. A move never waits for it."""
        if out.talk is None or thread_id is None or self.closed:
            return
        text, out.talk = out.talk, None

        async def run():
            try:
                await self.bot.run_table_event(await self._channel(thread_id), text)
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("Texas Hold'em event turn failed (thread %s)", thread_id)

        task = asyncio.create_task(run(), name=f"holdem-event-{thread_id}")
        self.event_tasks.add(task)
        task.add_done_callback(self.event_tasks.discard)

    def _event(self, key, out: _Out, text: str) -> None:
        """Under the table lock: at most one event turn per table per cooldown, and only while the characters may talk."""
        if self.clock() - self.event_at.get(key, float("-inf")) >= self.EVENT_COOLDOWN_SECONDS and self.talk_allowed(key[0]):
            self.event_at[key] = self.clock()
            out.talk = text

    async def _turn_message(self, key, out, row) -> None:
        thread_id, old = row["thread_id"], row["turn_message_id"]
        content, view = out.turn
        if old is not None and await self._edit(thread_id, old, content=content, view=view) is not False:
            return
        message = await self._post(thread_id, content, view)
        if message is not None:
            self.store.set_turn_message(key[0], key[1], message.id)

    async def _open_play(self, key, out, row) -> bool:
        """Posts the board (pinned) and the turn message in the thread. False when the thread cannot be written to; the table is then closed."""
        thread_id = row["thread_id"]
        board = await self._post(thread_id, out.start)
        turn = await self._post(thread_id, *out.turn) if board is not None else None
        if board is None or turn is None:
            async with self.runtime.lock(key):
                if (current := self._row(key)) is not None and current["status"] == "open":
                    self.store.close_table(*key)
                self._forget(key)
            await self._edit(row["channel_id"], row["message_id"], content="Table closed: I couldn't post in the table thread." + CASHED, view=None)
            self.arm(key, "archive", self.ARCHIVE_SECONDS)
            return False
        try:
            await board.pin()
        except discord.HTTPException as error:
            logging.info("Could not pin the Texas Hold'em board (%s)", type(error).__name__)
        self.store.set_board_message(key[0], key[1], board.id)
        self.store.set_turn_message(key[0], key[1], turn.id)
        self.table(key).sent["turn"] = out.seq
        return True

    # --- the board (debounced) --------------------------------------------------------------

    def _mark_board(self, key) -> None:
        self.dirty.add(key)
        if key not in self.board_tasks and not self.closed:
            task = self.board_tasks[key] = asyncio.create_task(self._debounce(key), name=f"holdem-board-{key[1]}")
            task.add_done_callback(lambda done: self.board_tasks.get(key) is done and self.board_tasks.pop(key, None))

    async def _debounce(self, key) -> None:
        await asyncio.sleep(self.BOARD_DEBOUNCE_SECONDS)
        if key in self.dirty:
            self.dirty.discard(key)
            await self._edit_board(key)

    async def _edit_board(self, key) -> None:
        row = self._row(key)
        if not self._open_here(row) or row["board_message_id"] is None:
            return
        try:
            snap = self.store.latest_round(*key)
        except GameError:
            return
        await self._edit(row["thread_id"], row["board_message_id"], content=self.render_board(key, snap))

    async def flush(self) -> None:
        """Sends the pending board edits now."""
        keys = list(self.dirty)
        self.dirty.clear()
        for key in keys:
            await self._edit_board(key)

    # --- hand rendering ---------------------------------------------------------------------

    @staticmethod
    def _cards(cards) -> str:
        return " ".join(f"{c[0]}{c[1]}" for c in cards)

    @staticmethod
    def _move_text(snap, seat, move) -> str:
        who = GameTables._who(snap["seats"][seat])
        if move == "fold":
            return f"{who} folds."
        if move == "check":
            return f"{who} checks."
        if move == "call":
            return f"{who} calls."
        return f"{who} raises to {int(move[6:]):,}."

    def render_board(self, key, snap) -> str:
        rules, view = snap["rules"] or holdem.normalize_rules(None), snap["view"]
        sb = rules["small_blind"]
        lines = [f"**{NAME}** · table {key[1]} · hand {snap['number']}",
                 f"Blinds {sb}/{2 * sb} · {rules['action_timeout']} s per move" + (" · Assist: on" if rules["assist"] else "")]
        if view:
            lines.append(f"{view['street'].capitalize()} · Board: {self._cards(view['board']) or '—'} · Pot {view['pot']:,}")
        else:
            lines.append("Between hands.")
        name = self.store.currency_name(key[0])
        for seat in snap["seats"]:
            i, flags = seat["index"], ""
            if view:
                flags = (" (dealer)" if view["button"] == i else "") + (" · folded" if view["folded"][i] else " · all-in" if view["all_in"][i] else "") \
                    + (" · on turn" if view["turn"] == i and not view["finished"] else "")
            stack = view["stacks"][i] if view else seat["stack"]
            lines.append(f"{i + 1}. {GameTables._who(seat)} — {stack:,} {name}{flags}")
        lines.append(f"-# Seed: {snap['seed']} (hash {snap['seed_hash'][:12]})" if snap["seed"] else f"-# Seed hash: {snap['seed_hash'][:12]}")
        return "\n".join(lines)

    def render_turn(self, key, snap) -> tuple[str, discord.ui.View]:
        view, seats, table = snap["view"], snap["seats"], self.table(key)
        turn = view["turn"]
        lines = [f"**{view['street'].capitalize()}** · hand {snap['number']} · pot {view['pot']:,}", f"Board: {self._cards(view['board']) or '—'}"]
        lines.append(self._move_text(snap, *view["history"][-1]) if view["history"] else "Blinds are posted.")
        parts = []
        for seat in seats:
            i = seat["index"]
            parts.append(f"{GameTables._who(seat)} {view['stacks'][i]:,}" + (f" (bet {view['bets'][i]:,})" if view["bets"][i] else "")
                         + (" folded" if view["folded"][i] else " all-in" if view["all_in"][i] else ""))
        lines.append("Stacks: " + " · ".join(parts))
        buttons = discord.ui.View(timeout=None)
        if turn is not None:
            who, stamp = GameTables._who(seats[turn]), table.deadlines.get("turn")
            lines.append(f"On turn: {who}" + (f" — <t:{int(stamp)}:R>" if stamp and seats[turn]["kind"] == "member" else ", thinking…" if seats[turn]["kind"] == "character" else ""))
            legal = holdem.legal_moves_for_view(holdem.View(**{**view, "seat": turn}))
            tid, rid, moves = key[1], snap["round_id"], snap["moves"]
            buttons.add_item(HoldemButton("fold", tid, rid, moves, "Fold", discord.ButtonStyle.danger))
            buttons.add_item(HoldemButton("check", tid, rid, moves, "Check", discord.ButtonStyle.success) if "check" in legal
                             else HoldemButton("call", tid, rid, moves, f"Call {view['to_call']:,}", discord.ButtonStyle.success))
            buttons.add_item(HoldemButton("raise", tid, rid, moves, "Raise", discord.ButtonStyle.primary, not any(m.startswith("raise:") for m in legal)))
            buttons.add_item(HoldemButton("cards", tid, rid, moves, "My cards", discord.ButtonStyle.secondary))
        lines.append(f"-# Seed hash: {snap['seed_hash'][:12]}")
        return "\n".join(lines), buttons

    def render_pause(self, key, snap, row, waiting: bool = False) -> tuple[str, discord.ui.View]:
        """The turn message between hands: the seats with their stacks and Sit, Leave, Rebuy, Deal now (the lobby round's buttons)."""
        name, table = self.store.currency_name(key[0]), self.table(key)
        if waiting:
            stamp = f"<t:{int(d)}:R>" if (d := table.deadlines.get("lobby")) else ""
            lines = [f"**Waiting for players.** The next hand needs {holdem.MIN_SEATS} seats; sit to continue" + (f". The table closes {stamp} if no hand starts." if stamp else ".")]
        else:
            stamp = f"<t:{int(d)}:R>" if (d := table.deadlines.get("next")) else "soon"
            lines = [f"**Next hand starts {stamp}.** Sit, leave or top up now; the host can deal at once."]
        for seat in snap["seats"]:
            lines.append(f"{seat['index'] + 1}. {GameTables._who(seat)} — {seat['stack']:,} {name}")
        lines.append(f"-# Next seed hash: {snap['seed_hash'][:12]}")
        view, tid, rid = discord.ui.View(timeout=None), key[1], snap["round_id"]
        for action, label, style in (("sit", "Sit", discord.ButtonStyle.primary), ("leave", "Leave", discord.ButtonStyle.secondary),
                                     ("rebuy", "Rebuy", discord.ButtonStyle.secondary), ("deal", "Deal now", discord.ButtonStyle.success)):
            view.add_item(HoldemButton(action, tid, rid, 0, label, style))
        return "\n".join(lines), view

    def _result_text(self, key, snap, state, res) -> str:
        who = lambda i: GameTables._who(snap["seats"][i])
        lines = [f"**Hand {snap['number']}** · table {key[1]}"]
        if res.board:
            lines.append(f"Board: {self._cards(res.board)}")
        for i in res.shown:
            lines.append(f"{who(i)} shows {self._cards(state.holes[i])} — {holdem.hand_name(holdem.evaluate(state.holes[i] + state.board_full))}")
        for k, pot in enumerate(res.pots):
            label = "Pot" if len(res.pots) == 1 else "Main pot" if k == 0 else "Side pot" if len(res.pots) == 2 else f"Side pot {k}"
            winners = ", ".join(who(w) for w in pot.winners)
            lines.append(f"{label} {pot.amount:,}: " + (f"{winners} win" if len(pot.winners) > 1 else f"{winners} wins") + (f" with {holdem.hand_name(pot.hand)}" if pot.hand else ""))
        for seat in res.seats:
            lines.append(f"• {who(seat.seat)}: {seat.start:,} → {seat.end:,} ({seat.net:+,})")
        if snap["seed"]:
            lines.append(f"-# Seed: {snap['seed']} (hash {snap['seed_hash'][:12]})")
        return "\n".join(lines)

    def _final_state(self, key, snap):
        """The finished hand's engine state, replayed from the revealed seed (settled rounds only)."""
        seats = [Seat(s["index"], s["kind"], s["ref_id"], s["stack"]) for s in snap["seats"]]
        moves = [(r["seat_index"], r["move"]) for r in self.store.all("SELECT seat_index,move FROM game_moves WHERE guild_id=? AND round_id=? ORDER BY id", (key[0], snap["round_id"]))]
        return holdem.replay(snap["seed"], seats, moves, snap["rules"], snap["rules"]["button"])

    def _summary(self, key) -> str:
        lines = ["**Table closed.** Cash-outs:"]
        for row in self.store.table_cash_summary(*key)["cashouts"]:
            who = f"<@{row['ref_id']}>" if row["kind"] == "member" else f"**{discord.utils.escape_markdown(discord.utils.escape_mentions(row['name']))}**"
            lines.append(f"• {who}: bought in {row['bought_in']:,}, cashed out {row['cashed_out']:,} ({row['net']:+,})")
        return "\n".join(lines)

    def _timeout_streak(self, guild_id, table_id, user_id) -> int:
        """In how many of the table's latest hands in a row the member's turn ran out."""
        rows = self.store.all("SELECT (SELECT COUNT(*) FROM game_moves m JOIN game_seats s ON s.guild_id=m.guild_id AND s.round_id=m.round_id AND s.seat_index=m.seat_index "
                              "WHERE m.guild_id=r.guild_id AND m.round_id=r.id AND m.actor='timeout' AND s.kind='member' AND s.ref_id=?) AS n "
                              "FROM game_rounds r WHERE r.guild_id=? AND r.table_id=? AND r.status='settled' ORDER BY r.number DESC", (user_id, guild_id, table_id))
        streak = 0
        for row in rows:
            if not row["n"]:
                break
            streak += 1
        return streak

    # --- moving the hand on -----------------------------------------------------------------

    def _after(self, key, snap) -> None:
        """Arms the timer the new state needs. Call under the table lock."""
        table, view = self.table(key), snap["view"]
        self.runtime.disarm(key, "lobby")
        self.runtime.disarm(key, "next")
        turn = view["turn"] if view else None
        if snap["status"] != "playing" or turn is None:
            table.deadlines.pop("turn", None)
            self.runtime.disarm(key, "turn")
            self.runtime.disarm(key, "character")
            return
        rid, moves = snap["round_id"], snap["moves"]
        if snap["seats"][turn]["kind"] == "member":
            delay = (snap["rules"] or {}).get("action_timeout", 30)
            table.deadlines["turn"] = self.clock() + delay
            self.runtime.disarm(key, "character")
            self.arm(key, "turn", delay, rid, moves)
        else:
            table.deadlines.pop("turn", None)
            self.runtime.disarm(key, "turn")
            self.arm(key, "character", self.CHARACTER_SECONDS, rid, moves)

    def _progress(self, key, snap, out: _Out) -> None:
        """Under the table lock: the new snapshot of a dealt or moved hand becomes the timers and messages it needs."""
        if snap["status"] == "playing":
            self._after(key, snap)
            out.turn = self.render_turn(key, snap)
            out.board = True
        else:
            self._finish(key, snap, out)

    def _finish(self, key, snap, out: _Out) -> None:
        """A settled hand: the result post, the next lobby (busted seats leave), seats that timed out twice in a row cashed out; fewer than two seats left closes the table."""
        guild_id, table = key[0], self.table(key)
        self._after(key, snap)
        state = self._final_state(key, snap)
        res = holdem.result(state)
        who = lambda seat: GameTables._who(seat)
        notes = [f"{who(seat)} is out of chips" + (" and can sit again to rejoin." if seat["kind"] == "member" else " and leaves the table.") for seat in snap["seats"] if seat["busted"]]
        lobby = self.store.next_round(*key)
        over = len(lobby["seats"]) < holdem.MIN_SEATS
        if not over:
            for seat in list(lobby["seats"]):
                if seat["kind"] == "member" and self._timeout_streak(guild_id, key[1], seat["ref_id"]) >= 2:
                    lobby = self.store.leave_round(guild_id, lobby["round_id"], seat["ref_id"])
                    notes.append(f"{who(seat)} ran out of time in two hands in a row and was cashed out with {seat['stack']:,}.")
        view = None
        if not res.shown:
            view = discord.ui.View(timeout=None)
            view.add_item(HoldemButton("show", key[1], snap["round_id"], 0, "Show", discord.ButtonStyle.secondary))
        out.result = ("\n".join([self._result_text(key, snap, state, res), *notes]), view)
        busted = [i for i, seat in enumerate(snap["seats"]) if seat["busted"]]
        if len(res.shown) > 1 or busted:
            names, events = self._talk_names(guild_id, snap), []
            for pot in res.pots[:1]:
                events.append(f"Showdown in hand {snap['number']}: " + " and ".join(names[w] for w in pot.winners)
                              + f" {'win' if len(pot.winners) > 1 else 'wins'} the pot of {pot.amount:,}" + (f" with {holdem.hand_name(pot.hand).lower()}." if pot.hand else "."))
            events += [f"{names[i]} is out of chips." for i in busted]
            self._event(key, out, " ".join(events))
        row = self._row(key)
        if over:
            self.store.close_table(*key)
            self._forget(key)
            self.arm(key, "archive", self.ARCHIVE_SECONDS)
            out.summary = self._summary(key)
            out.turn = ("Table closed.", None)
            out.lobby = (f"Table closed: fewer than {holdem.MIN_SEATS} players still had chips." + CASHED, None)
            out.board_text = f"**{NAME}** · table {key[1]} — closed after hand {snap['number']}."
            out.closed = True
            return
        table.deadlines["next"] = self.clock() + self.NEXT_HAND_SECONDS
        self.arm(key, "next", self.NEXT_HAND_SECONDS, lobby["round_id"])
        if len(lobby["seats"]) < holdem.MIN_SEATS:
            self._touch_lobby(key, lobby["round_id"])
        out.turn = self.render_pause(key, lobby, row)
        out.board = True

    def _apply(self, key, snap, seat, move, actor) -> _Out:
        """Makes one move under the table lock; raises ConflictError / GameError from the store."""
        new = self.store.play(key[0], snap["round_id"], seat, move, actor, snap["moves"])
        out = _Out()
        self._progress(key, new, out)
        if new["status"] == "playing" and new["view"]["all_in"][seat] and not snap["view"]["all_in"][seat]:
            self._event(key, out, f"Big pot: {self._talk_names(key[0], new)[seat]} goes all-in. The pot is {new['view']['pot']:,}.")
        return self._seal(key, out)

    def _timer_snap(self, key, round_id, moves):
        """Under the table lock: the playing snapshot a timer may act on, or None (the table moved on or closed)."""
        row = self._row(key)
        if not self._open_here(row):
            self._release(key)
            return None
        try:
            snap = self.store.round_snapshot(key[0], round_id)
        except GameError:
            return None
        if snap["table_id"] != key[1] or snap["status"] != "playing" or snap["moves"] != moves or not snap["view"]:
            return None
        return snap

    def _member_move(self, key, user_id, round_id, moves, move) -> _Out:
        loaded, refusal = self._load(key, round_id)
        if refusal is not None:
            return refusal
        _, snap = loaded
        seat = self.store.member_seat_index(key[0], round_id, user_id)
        if seat is None:
            return _Out(NOT_SEATED)
        if snap["status"] != "playing" or snap["moves"] != moves:
            return _Out(STALE)
        if snap["view"]["turn"] != seat:
            return _Out(NOT_YOUR_TURN)
        if move.startswith("raise:") and (problem := self._raise_problem(key, snap, seat, int(move[6:]))):
            return _Out(problem)
        try:
            return self._apply(key, snap, seat, move, "member")
        except ConflictError:
            return _Out(STALE)
        except GameError as error:
            return _Out(str(error))

    def _raise_problem(self, key, snap, seat, amount) -> str | None:
        v = self.store.seat_view(key[0], snap["round_id"], seat)
        if not any(m.startswith("raise:") for m in holdem.legal_moves_for_view(v)):
            return "You can't raise now."
        top = v.bets[seat] + v.stacks[seat]
        if amount > top:
            return f"Your whole stack makes a total bet of {top:,}; you can't raise to more."
        if amount != top and amount < v.min_raise_to:
            return f"You can only go all-in, to {top:,}." if top < v.min_raise_to else f"The smallest raise is to {v.min_raise_to:,} in total."
        return None

    # --- /game start ------------------------------------------------------------------------

    async def start_command(self, interaction: discord.Interaction, favorites: bool = False) -> None:
        guild_id = interaction.guild_id
        channel_id = getattr(interaction, "channel_id", None) or interaction.channel.id
        user = interaction.user
        if user.bot:
            await self._say(interaction, f"Bots cannot play {NAME}.")
            return
        if not self.store.game_enabled(guild_id, "holdem"):
            await self._say(interaction, HOLDEM_OFF)
            return
        if channel_id not in self.store.game_channels(guild_id):
            await self._say(interaction, OFF_HERE)
            return
        _, binding = self.bot.location(interaction.channel)
        space_id, note = (binding["space_id"] if binding else None), ""
        if favorites and not binding:
            note = NO_CHARACTER_SEATS
        try:
            opened = self.store.open_table(guild_id, channel_id, "holdem", user.id)
        except GameError as error:
            await self._say(interaction, str(error))
            return
        key, round_id = (guild_id, opened["table_id"]), opened["round_id"]
        self.table(key)
        self._ensure_sweeper()
        async with self.runtime.lock(key):
            try:
                snap = opened
                if favorites:
                    stake = default_buy_in(opened["rules"] or holdem.normalize_rules(None))
                    if space_id is not None:
                        result = self.store.join_with_favorites(guild_id, round_id, user.id, stake, space_id)
                        snap, note = result["snapshot"], self.bot.games._favorites_note(result, note)
                    else:
                        snap = self.store.join_round(guild_id, round_id, user.id, stake)
            except GameError as error:
                self.store.close_table(*key)
                self._release(key)
                self.tables.pop(key, None)
                refusal = str(error)
            else:
                refusal = None
                self.table(key).deadlines["lobby"] = self.clock() + self.LOBBY_SECONDS
                self.arm(key, "lobby", self.LOBBY_SECONDS, round_id)
                content, view = self.render_lobby(key, snap, self._row(key))
                out = self._seal(key, _Out())
        if refusal is not None:
            await self._say(interaction, refusal)
            return
        table = self.table(key)
        async with table.send_lock:
            try:
                await interaction.response.send_message(content, view=view, allowed_mentions=NO_MENTIONS)
                message = await interaction.original_response()
                self.store.set_table_message(*key, message.id)
            except Exception:
                async with self.runtime.lock(key):
                    self.store.close_table(*key)
                    self._release(key)
                self.tables.pop(key, None)
                raise
            table.sent["lobby"] = max(table.sent.get("lobby", 0), out.seq)
        fresh = None
        async with self.runtime.lock(key):
            if table.issued > out.seq and (row := self._row(key))["status"] == "open":  # someone sat before the message existed
                fresh = self._seal(key, _Out(lobby=self.render_lobby(key, self.store.latest_round(*key), row)))
        await self._publish(key, fresh)
        try:
            partial = await self._channel(channel_id)
            thread = await partial.get_partial_message(message.id).create_thread(name=f"{NAME} · table {key[1]}", auto_archive_duration=1440)
        except discord.HTTPException as error:
            logging.warning("Could not open the Texas Hold'em thread (%s)", type(error).__name__)
            await self._thread_failed(interaction, key, isinstance(error, discord.Forbidden))
            return
        ready = None
        async with self.runtime.lock(key):  # Deal waits for the thread
            row = self._row(key)
            if row["status"] == "open":
                self.store.set_table_thread(*key, thread.id)
                self.threads[thread.id] = key
                ready = self._seal(key, _Out(lobby=self.render_lobby(key, self.store.latest_round(*key), row)))
        if ready is None:  # closed while the thread was being made: nothing else to say there, just archive it
            try:
                await thread.edit(archived=True, locked=True)
            except Exception as error:
                logging.warning("Could not archive the unused Texas Hold'em thread (%s)", type(error).__name__)
        await self._publish(key, ready)
        if note:
            await self._say(interaction, note)

    async def _thread_failed(self, interaction, key, forbidden: bool) -> None:
        async with self.runtime.lock(key):
            row = self._row(key)
            if row is not None and row["status"] == "open":
                self.store.close_table(*key)
            self._forget(key)
            out = self._seal(key, _Out(lobby=("Table closed: I couldn't open a thread here." + CASHED, None), closed=True))
        await self._publish(key, out)
        self.tables.pop(key, None)
        self.ended.discard(key)
        await self._say(interaction, (THREAD_FORBIDDEN if forbidden else THREAD_FAILED) + CASHED)

    # --- buttons ----------------------------------------------------------------------------

    async def press(self, interaction: discord.Interaction, action: str, table_id: int, round_id: int, moves: int) -> None:
        guild_id = interaction.guild_id
        if guild_id is None or interaction.user.bot:
            await self._say(interaction, STALE)
            return
        key = (guild_id, table_id)
        if action in ("sit", "rebuy"):
            async with self.runtime.lock(key):
                modal, refusal = self._modal(key, round_id, action, interaction.user.id)
            if modal is not None:
                try:
                    await interaction.response.send_modal(modal)
                except discord.HTTPException as error:
                    logging.warning("Texas Hold'em could not open the %s modal: %s", action, type(error).__name__)
                return
            await self._say(interaction, refusal)
            if refusal == STALE:
                await self._reap(key)
            return
        if action == "cards":
            await self._show_cards(interaction, key, round_id)
            return
        if action == "raise":
            await self._open_picker(interaction, key, round_id, moves)
            return
        if action == "show":
            await self._show_won_hand(interaction, key, round_id)
            return
        await self._ack(interaction)
        async with self.runtime.lock(key):
            out = self._press(interaction.user.id, key, action, round_id, moves)
        await self._publish(key, out)
        if out.say == STALE:
            await self._reap(key)  # the table may have been closed from outside
        if out.say:
            await self._say(interaction, out.say)

    def _press(self, user_id, key, action, round_id, moves) -> _Out:
        loaded, refusal = self._load(key, round_id)
        if refusal is not None:
            return refusal
        row, snap = loaded
        try:
            if action == "leave":
                return self._leave(key, row, snap, user_id)
            if action == "fill":
                return self._fill(key, row, snap, user_id)
            if action == "assist":
                return self._assist(key, row, snap, user_id, bool(moves))
            if action == "deal":
                return self._deal(key, row, snap, user_id)
            if action == "close":
                return self._close(key, row, snap, user_id)
            if action in ("fold", "check", "call"):
                return self._member_move(key, user_id, round_id, moves, action)
            return _Out(STALE)
        except ConflictError:
            return _Out(STALE)
        except GameError as error:
            return _Out(str(error))

    def _modal(self, key, round_id, action, user_id):
        """(the modal to open, None) or (None, the private refusal). Nothing is charged until the member submits."""
        loaded, refusal = self._load(key, round_id)
        if refusal is not None:
            return None, refusal.say
        row, snap = loaded
        if snap["status"] != "joining":
            return None, "This round has already started."
        rules, name = snap["rules"] or holdem.normalize_rules(None), self.store.currency_name(key[0])
        low, high = buy_in_range(rules)
        mine = next((s for s in snap["seats"] if s["kind"] == "member" and s["ref_id"] == user_id), None)
        if action == "rebuy":
            if mine is None:
                return None, NO_SEAT
            room = max(high - mine["stack"], 0)
            return AmountModal("rebuy", key[0], key[1], snap["round_id"], "Rebuy", f"Chips to add (up to {room:,})", f"Up to {room:,} {name}"), None
        if mine is not None:
            return None, "You already have a seat in this round."
        if len(snap["seats"]) >= holdem.MAX_SEATS:
            return None, "The table is full."
        default = default_buy_in(rules)
        return AmountModal("sit", key[0], key[1], snap["round_id"], "Sit at the table", f"Buy-in, {low:,} to {high:,}", f"{low:,} to {high:,} {name}", str(default)), None

    async def submit(self, interaction: discord.Interaction, modal: AmountModal) -> None:
        guild_id = interaction.guild_id
        if guild_id is None or interaction.user.bot or guild_id != modal.guild_id:
            await self._say(interaction, STALE)
            return
        key, text = (guild_id, modal.table_id), (modal.field.value or "").strip()
        if not re.fullmatch(r"-?[0-9]{1,12}", text):
            await self._say(interaction, {"sit": "The buy-in must be a whole number.", "raise": "The raise must be a whole number."}.get(modal.kind, "The rebuy must be a whole number above zero."))
            return
        if modal.kind == "raise":
            async with self.runtime.lock(key):
                out = self._member_move(key, interaction.user.id, modal.round_id, modal.moves, f"raise:{int(text)}")
            await self._say(interaction, out.say or f"You raise to {int(text):,}.")
            await self._publish(key, out)
            if out.say == STALE:
                await self._reap(key)
            return
        async with self.runtime.lock(key):
            out = self._buy(key, modal.kind, modal.round_id, interaction.user.id, int(text))
        await self._say(interaction, out.say)  # answer first, so a slow edit cannot make it miss Discord's 3 seconds
        await self._publish(key, out)
        if out.say == STALE:
            await self._reap(key)

    def _buy(self, key, kind, round_id, user_id, amount) -> _Out:
        loaded, refusal = self._load(key, round_id)
        if refusal is not None:
            return refusal
        row, _ = loaded
        name = self.store.currency_name(key[0])
        try:
            if kind == "sit":
                snap = self.store.join_round(key[0], round_id, user_id, amount)
                say = f"You sit down with {amount:,} {name}."
            else:
                snap = self.store.rebuy(key[0], round_id, user_id, amount)
                stack = next(s["stack"] for s in snap["seats"] if s["kind"] == "member" and s["ref_id"] == user_id)
                say = f"You added {amount:,} {name}; your stack is {stack:,}."
        except GameError as error:
            return _Out(str(error))
        self._touch_lobby(key, round_id)
        return self._paint(key, row, snap, self._seal(key, _Out(say)))

    def _leave(self, key, row, snap, user_id) -> _Out:
        mine = next((s for s in snap["seats"] if s["kind"] == "member" and s["ref_id"] == user_id), None)
        snap = self.store.leave_round(key[0], snap["round_id"], user_id)
        self._touch_lobby(key, snap["round_id"])  # an empty table stays open; the idle timer closes it
        name = self.store.currency_name(key[0])
        return self._paint(key, row, snap, self._seal(key, _Out(f"You left the table and cashed out {mine['stack']:,} {name}.")))

    def _fill(self, key, row, snap, user_id) -> _Out:
        if user_id != row["opened_by"]:
            return _Out("Only the host can fill seats.")
        guild_id, binding = key[0], self.store.binding(row["channel_id"])
        if binding is None:
            return _Out(NO_CHARACTER_SEATS)
        taken = {s["ref_id"] for s in snap["seats"] if s["kind"] == "character"}
        eligible = {r["id"] for r in self.store.eligible_characters(guild_id, binding["space_id"])}
        pool = [c for c in self.store.get_cast(row["channel_id"], None) if c in eligible and c not in taken]
        need, seated, skipped = (holdem.MIN_SEATS - len(snap["seats"])) if len(snap["seats"]) < holdem.MIN_SEATS else 1, [], []
        while need > 0 and pool:
            chunk, pool = pool[:need], pool[need:]
            result = self.store.fill_seats(guild_id, snap["round_id"], user_id, chunk, binding["space_id"])
            snap, need = result["snapshot"], need - len(result["seated"])
            seated += result["seated"]
            skipped += result["skipped"]
        name = self.store.currency_name(guild_id)
        why = lambda reason: f"not enough {name}" if reason == "broke" else SKIPPED.get(reason, reason)
        parts = (["Seated: " + ", ".join(s["name"] for s in seated) + "."] if seated else []) + \
                (["Not seated: " + ", ".join(f"{s['name']} ({why(s['reason'])})" for s in skipped) + "."] if skipped else [])
        if not parts:
            return _Out("No one is left in the cast who can take a seat.")
        if not seated:
            return _Out(" ".join(parts))
        self._touch_lobby(key, snap["round_id"])
        return self._paint(key, row, snap, self._seal(key, _Out(" ".join(parts))))

    def _assist(self, key, row, snap, user_id, on) -> _Out:
        snap = self.store.set_table_assist(key[0], snap["round_id"], user_id, on)
        return self._paint(key, row, snap, self._seal(key, _Out(f"Assist is {'on' if on else 'off'}.")))

    def _paint(self, key, row, snap, out: _Out) -> _Out:
        """Shows the seats where the table is waiting: the lobby message before the first hand, the turn message between hands."""
        if row["turn_message_id"] is not None:
            out.turn = self.render_pause(key, snap, row)
        else:
            out.lobby = self.render_lobby(key, snap, row)
        return out

    def _deal(self, key, row, snap, user_id) -> _Out:
        if user_id != row["opened_by"]:
            return _Out("Only the host can deal.")
        if row["thread_id"] is None:
            return _Out("The table thread isn't ready yet; try again in a moment.")
        if len(snap["seats"]) < holdem.MIN_SEATS:
            return _Out(f"{NAME} needs at least {holdem.MIN_SEATS} seats to deal.")
        snap = self.store.deal(key[0], snap["round_id"])
        out = _Out()
        if row["turn_message_id"] is None:  # the table's first hand: the board and the turn message are posted
            out.start = self.render_board(key, snap)
            out.lobby = (f"**{NAME}** · table {key[1]}\nThe hands are played in the thread.", None)
        self._progress(key, snap, out)
        return self._seal(key, out)

    def _close(self, key, row, snap, user_id) -> _Out:
        if user_id != row["opened_by"]:
            return _Out("Only the host can close the table.")
        self.store.close_table(*key)
        self._forget(key)
        self.arm(key, "archive", self.ARCHIVE_SECONDS)
        out = _Out("Table closed." + CASHED, lobby=("Table closed by the host." + CASHED, None), closed=True)
        self._ended_after_hands(key, row, out)
        return self._seal(key, out)

    def _ended_after_hands(self, key, row, out: _Out) -> None:
        """A table that played hands ends with its cash-out summary in the thread and a turn message without buttons."""
        if row["turn_message_id"] is not None:
            out.turn, out.summary = ("Table closed.", None), self._summary(key)

    # --- /game hand and /game rules ---------------------------------------------------------

    async def hand_command(self, interaction: discord.Interaction) -> None:
        guild_id = interaction.guild_id
        table = self.store.table_for_thread(guild_id, interaction.channel.id) if guild_id and interaction.channel else None
        if table is None or table["game"] != "holdem":
            await self._say(interaction, NO_TABLE)
            return
        await self._show_cards(interaction, (guild_id, table["id"]), None)

    async def _show_cards(self, interaction, key, round_id) -> None:
        """My cards: the presser's own hole cards (and, with Assist on, the made hand and the odds), privately."""
        async with self.runtime.lock(key):
            text, plan = self._cards_of(key, round_id, interaction.user.id)
        if plan is not None:
            text += "\n" + await self._assist_line(plan)
        await self._say(interaction, text)
        if text == STALE:
            await self._reap(key)

    def _cards_of(self, key, round_id, user_id):
        """Under the table lock: (private text, assist plan or None)."""
        if round_id is None:
            if not self._open_here(self._row(key)):
                return STALE, None
            snap = self.store.latest_round(*key)
        else:
            loaded, refusal = self._load(key, round_id)
            if refusal is not None:
                return STALE, None
            snap = loaded[1]
        seat = self.store.member_seat_index(key[0], snap["round_id"], user_id)
        if seat is None:
            return NOT_SEATED, None
        if snap["status"] == "joining":
            return "The hand hasn't started yet.", None
        if snap["status"] != "playing":
            return NO_HAND, None
        v = self.store.seat_view(key[0], snap["round_id"], seat)
        text = f"**Your cards:** {self._cards(v.hole)}"
        if not (snap["rules"] and snap["rules"]["assist"] and self.store.game_rules(key[0], "holdem")["assist"]):
            return text, None
        opponents = max(1, sum(1 for i, folded in enumerate(v.folded) if not folded and i != seat))
        return text, (key[0], snap["round_id"], seat, v.street, tuple(v.hole), tuple(v.board), opponents, self.store.policy_seed(key[0], snap["round_id"], f"assist/{seat}"))

    async def _assist_line(self, plan) -> str:
        """The odds text, computed off the event loop once per hand, seat and street."""
        guild_id, round_id, seat, street, hole, board, opponents, seed = plan
        cached = self.assists.get((guild_id, round_id, seat, street))
        if cached is None:
            a = await asyncio.to_thread(holdem.assist, hole, board, opponents, seed)
            cached = f"Assist — you have: {a.current}. Likely final hands: " + ", ".join(f"{name} {p:.0%}" for name, p in a.chances) + f". Chance to win against random hands: {a.win:.0%}."
            if len(self.assists) > 512:
                self.assists.clear()
            self.assists[(guild_id, round_id, seat, street)] = cached
        return cached

    # --- raising and showing ----------------------------------------------------------------

    async def _open_picker(self, interaction, key, round_id, moves) -> None:
        async with self.runtime.lock(key):
            picker, refusal = self._picker(key, round_id, moves, interaction.user.id)
        if picker is None:
            await self._say(interaction, refusal)
            if refusal == STALE:
                await self._reap(key)
            return
        try:
            await interaction.response.send_message("Choose your raise (the total bet this round).", view=picker, ephemeral=True, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning("Texas Hold'em could not open the raise picker: %s", type(error).__name__)

    def _picker(self, key, round_id, moves, user_id):
        loaded, refusal = self._load(key, round_id)
        if refusal is not None:
            return None, STALE
        _, snap = loaded
        seat = self.store.member_seat_index(key[0], round_id, user_id)
        if seat is None:
            return None, NOT_SEATED
        if snap["status"] != "playing" or snap["moves"] != moves:
            return None, STALE
        if snap["view"]["turn"] != seat:
            return None, NOT_YOUR_TURN
        amounts = [int(m[6:]) for m in holdem.legal_moves_for_view(self.store.seat_view(key[0], round_id, seat)) if m.startswith("raise:")]
        if not amounts:
            return None, "You can't raise now."
        return _RaisePicker(key[0], key[1], round_id, moves, amounts, amounts[-1]), None

    async def _answer(self, interaction, text: str) -> None:
        """Replaces the private picker with the answer (or replies when the interaction was already answered)."""
        if interaction.response.is_done():
            await self._say(interaction, text)
            return
        try:
            await interaction.response.edit_message(content=text, view=None)
        except discord.HTTPException as error:
            logging.warning("Texas Hold'em picker answer failed: %s", type(error).__name__)

    async def raise_to(self, interaction: discord.Interaction, picker: _RaisePicker, amount: int) -> None:
        if interaction.guild_id != picker.guild_id or interaction.user.bot:
            await self._answer(interaction, STALE)
            return
        key = (picker.guild_id, picker.table_id)
        async with self.runtime.lock(key):
            out = self._member_move(key, interaction.user.id, picker.round_id, picker.moves, f"raise:{amount}")
        await self._answer(interaction, out.say or f"You raise to {amount:,}.")
        await self._publish(key, out)
        if out.say == STALE:
            await self._reap(key)

    async def ask_exact(self, interaction: discord.Interaction, picker: _RaisePicker) -> None:
        if interaction.guild_id != picker.guild_id or interaction.user.bot:
            await self._say(interaction, STALE)
            return
        modal = AmountModal("raise", picker.guild_id, picker.table_id, picker.round_id, "Raise", "Total bet this round", "Your total bet, e.g. 35", moves=picker.moves)
        try:
            await interaction.response.send_modal(modal)
        except discord.HTTPException as error:
            logging.warning("Texas Hold'em could not open the raise modal: %s", type(error).__name__)

    async def _show_won_hand(self, interaction, key, round_id) -> None:
        """Show: the winner of a hand everyone else folded posts their two cards, once."""
        async with self.runtime.lock(key):
            text, say = self._show(key, round_id, interaction.user.id)
        if text is None:
            await self._say(interaction, say)
            if say == STALE:
                await self._reap(key)
            return
        view = discord.ui.View(timeout=None)
        view.add_item(HoldemButton("show", key[1], round_id, 0, "Show", disabled=True))
        try:
            await interaction.response.edit_message(view=view)
        except discord.HTTPException as error:
            logging.warning("Texas Hold'em could not close the Show button: %s", type(error).__name__)
        row = self._row(key)
        await self._post(row["thread_id"] if row else None, text)

    def _show(self, key, round_id, user_id):
        """Under the table lock: (public text, None) or (None, private refusal)."""
        loaded, refusal = self._load(key, round_id)
        if refusal is not None:
            return None, STALE
        _, snap = loaded
        mine = next((s for s in snap["seats"] if s["kind"] == "member" and s["ref_id"] == user_id), None)
        if mine is None:
            return None, NOT_SEATED
        if snap["status"] != "settled":
            return None, STALE
        if mine["outcome"] != "win":
            return None, "Only the winner can show their cards."
        state = self._final_state(key, snap)
        if holdem.result(state).shown:
            return None, "The cards were already shown at the showdown."
        if (key[0], round_id) in self.shown:
            return None, "You already showed your cards."
        self.shown.add((key[0], round_id))
        return f"{GameTables._who(mine)} shows {self._cards(state.holes[mine['index']])}.", None

    def rules_text(self, guild_id) -> str:
        rules = self.store.game_rules(guild_id, "holdem")
        return holdem.how_to_play(rules) + "\n" + holdem.rules_text(rules)

    # --- timers' work -----------------------------------------------------------------------

    async def on_lobby_timer(self, guild_id, table_id, round_id) -> None:
        """Nobody dealt a hand in time: the table closes and every stack is cashed out."""
        key, out = (guild_id, table_id), None
        async with self.runtime.lock(key):
            row = self._row(key)
            if not self._open_here(row):
                self._release(key)
                return
            try:
                snap = self.store.round_snapshot(guild_id, round_id)
            except GameError:
                return
            if snap["status"] != "joining":
                return
            self.store.close_table(*key)
            self._forget(key)
            self.arm(key, "archive", self.ARCHIVE_SECONDS)
            out = _Out(lobby=("Table closed: nobody started the game." + CASHED, None), closed=True)
            self._ended_after_hands(key, row, out)
            out = self._seal(key, out)
        await self._publish(key, out)

    async def on_turn_timer(self, guild_id, table_id, round_id, moves) -> None:
        """A member's action time ran out: check if that is free, else fold."""
        key, out = (guild_id, table_id), None
        async with self.runtime.lock(key):
            snap = self._timer_snap(key, round_id, moves)
            if snap is None:
                return
            seat = snap["view"]["turn"]
            if seat is None or snap["seats"][seat]["kind"] != "member":
                return
            try:
                out = self._apply(key, snap, seat, holdem.timeout_move_for_view(self.store.seat_view(guild_id, round_id, seat)), "timeout")
            except (ConflictError, GameError, IllegalMove) as error:
                logging.info("Texas Hold'em turn timeout refused (%s)", type(error).__name__)
                return
        await self._publish(key, out)

    def _talk_budget(self, guild_id) -> int:
        """How many model calls the table-talk daily limit still allows (0 when character table talk is off). Failed attempts count."""
        if not self.store.game_character_talk(guild_id):
            return 0
        return self.talk.remaining(guild_id)

    def _note_failure(self, guild_id) -> None:
        self.talk.note_failure(guild_id)  # one counter for every table game

    def talk_allowed(self, guild_id) -> bool:
        """Table talk is on and the daily limit (model calls plus failed attempts) is not reached."""
        return self._talk_budget(guild_id) > 0

    # --- table talk -------------------------------------------------------------------------

    def _talk_key(self, guild_id, thread_id):
        key = self.threads.get(thread_id) or self.banter.get(thread_id)
        return key if key is not None and key[0] == guild_id else None

    def _talk_names(self, guild_id, snap) -> list:
        """One prompt-safe name per seat for the table block and the event texts: characters by name, members by their guild display
        name (Player N when unknown); no mentions, no ids, no markdown."""
        guild, names, players = self.bot.get_guild(guild_id), [], 0
        for seat in snap["seats"]:
            if seat["kind"] == "character":
                name = seat["name"]
            else:
                players += 1
                member = guild.get_member(seat["ref_id"]) if guild is not None else None
                name = getattr(member, "display_name", None) or f"Player {players}"
            names.append(re.sub(r"[*_~`|>@<\\]", "", name).strip() or f"Player {len(names) + 1}")
        return names

    def table_talk_context(self, guild_id, thread_id) -> dict | None:
        """The table whose thread this is (lobby, hands, the pauses between them, and the banter after it closed until the archive timer), else None."""
        key = self._talk_key(guild_id, thread_id)
        if key is None:
            return None
        try:
            snap = self.store.latest_round(*key)
        except GameError:
            return None
        return {"table_id": key[1], "round_id": snap["round_id"], "status": snap["status"],
                "seated_character_ids": [s["ref_id"] for s in snap["seats"] if s["kind"] == "character"]}

    def table_block(self, guild_id, thread_id, character_id) -> str | None:
        """What one seated character may know of the table: the public state and, while a hand is played, its own two hole cards.
        Never another seat's cards, the Assist odds or the seed."""
        key = self._talk_key(guild_id, thread_id)
        if key is None:
            return None
        try:
            snap = self.store.latest_round(*key)
        except GameError:
            return None
        mine = next((s for s in snap["seats"] if s["kind"] == "character" and s["ref_id"] == character_id), None)
        if mine is None:
            return None
        names, view, me = self._talk_names(key[0], snap), snap["view"], mine["index"]
        who = lambda i: names[i] + (" (you)" if i == me else "")
        lines = [f"You are sitting at a table of {NAME} (no-limit poker: each seat holds two secret hole cards, five shared cards come on the board, the best five-card hand wins the pot)."]
        row = self._row(key)
        if row is None or row["status"] != "open":
            lines.append("The table is closed; the chips were cashed out.")
        if snap["status"] == "playing" and view:
            lines.append(f"Hand {snap['number']}, {view['street']}. Board: {self._cards(view['board']) or '(no cards yet)'}. Pot: {view['pot']:,}.")
            for i in range(len(snap["seats"])):
                flags = [f"bet {view['bets'][i]:,}"] if view["bets"][i] else []
                flags += ["dealer"] if view["button"] == i else []
                flags += ["folded"] if view["folded"][i] else ["all-in"] if view["all_in"][i] else []
                flags += ["on turn"] if view["turn"] == i and not view["finished"] else []
                lines.append(f"- {who(i)}: stack {view['stacks'][i]:,}" + (f" ({', '.join(flags)})" if flags else ""))
            recent = [self._talk_move(names, i, move) for i, move in view["history"][-8:]]
            if recent:
                lines.append("Recent play: " + "; ".join(recent) + ".")
            try:
                hole = self.store.seat_view(key[0], snap["round_id"], me).hole or ()
            except GameError:
                hole = ()
            if hole:
                lines.append("Your hole cards (keep them secret unless you choose to bluff about them): " + self._cards(hole))
            return "\n".join(lines)
        if snap["status"] == "joining":
            lines.append("No hand has been dealt yet." if snap["number"] == 1 else "Between hands; the next hand is dealt soon.")
        for seat in snap["seats"]:
            lines.append(f"- {who(seat['index'])}: stack {(seat['end'] if snap['status'] == 'settled' else seat['stack']):,}")
        if (last := self._last_hand_text(key, snap)) is not None:
            lines.append(last)
        return "\n".join(lines)

    @staticmethod
    def _talk_move(names, seat, move) -> str:
        return f"{names[seat]} raised to {int(move[6:]):,}" if move.startswith("raise:") else f"{names[seat]} {'checked' if move == 'check' else 'called' if move == 'call' else 'folded' if move == 'fold' else move}"

    def _last_hand_text(self, key, snap) -> str | None:
        """The table's last finished hand as posted in the thread: the board that was dealt out and who won or lost how much (no hole cards)."""
        row = self.store.one("SELECT id FROM game_rounds WHERE guild_id=? AND table_id=? AND status='settled' ORDER BY number DESC LIMIT 1", key)
        if row is None:
            return None
        last = snap if row["id"] == snap["round_id"] else self.store.round_snapshot(key[0], row["id"])
        names = self._talk_names(key[0], last)
        nets = [f"{names[s['index']]} {s['end'] - s['stack']:+,}" for s in last["seats"] if s["end"] is not None and s["end"] != s["stack"]]
        board = self._cards(last["view"]["board"]) if last["view"] else ""
        return f"Last hand ({last['number']}): board {board or '(none)'}" + (f"; {', '.join(nets)}" if nets else "") + "."

    async def on_character_timer(self, guild_id, table_id, round_id, moves) -> None:
        """A character seat on turn plays. With character table talk on, the daily limit not reached and Character moves not
        "style only", the model chooses from the engine menu (asked outside the table lock, applied only if the table is still at
        the same move count); any other answer or failure falls back to the engine policy with the saved Play style."""
        key, out, plan = (guild_id, table_id), None, None
        async with self.runtime.lock(key):
            snap = self._timer_snap(key, round_id, moves)
            if snap is None:
                return
            seat = snap["view"]["turn"]
            if seat is None or snap["seats"][seat]["kind"] != "character":
                return
            character = self.store.character_by_id(snap["seats"][seat]["ref_id"])
            if character is not None and character["guild_id"] == guild_id and self.store.game_character_moves(guild_id) != "style" and self._talk_budget(guild_id) > 0:
                seen = self.store.seat_view(guild_id, round_id, seat)
                plan = (character, seen, game_chooser.holdem_names(snap), holdem.legal_moves_for_view(seen), self.store.character_play_style(guild_id, character["id"]))
            else:
                out = self._character_move(key, snap, seat)
        if plan is not None:
            character, seen, names, legal, style = plan
            move = None
            try:
                result = await asyncio.wait_for(game_chooser.ask_holdem(self.bot, guild_id, snap["channel_id"], character, seen, names, legal, style), self.MODEL_SECONDS)
                move = game_chooser.holdem_move(result, legal)
            except asyncio.CancelledError:
                self._note_failure(guild_id)  # an attempt that was cut short still counts toward the limit
                raise
            except Exception as error:
                logging.warning("Texas Hold'em character move fell back to the policy (%s)", type(error).__name__)
            if move is None:
                self._note_failure(guild_id)
            async with self.runtime.lock(key):
                fresh = self._timer_snap(key, round_id, moves)  # None: the table moved on or closed while the model thought
                out = None if fresh is None else self._character_move(key, fresh, seat, move)
        await self._publish(key, out)

    def _character_move(self, key, snap, seat, chosen=None) -> _Out | None:
        """Under the table lock: the model's ``chosen`` move if given, else (or if refused) the policy's, then the free timeout move."""
        guild_id, round_id = key[0], snap["round_id"]
        seen = self.store.seat_view(guild_id, round_id, seat)
        seed = self.store.policy_seed(guild_id, round_id, f"{snap['moves']}/{seat}/character")
        tries = [(chosen, "character")] if chosen is not None else []
        try:
            tries.append((holdem.policy_action(seen, self.store.character_play_style(guild_id, snap["seats"][seat]["ref_id"]), seed), "character"))
        except (IllegalMove, ValueError) as error:
            logging.warning("Texas Hold'em character policy failed (%s)", type(error).__name__)
        tries.append((holdem.timeout_move_for_view(seen), "timeout"))
        for move, actor in tries:
            try:
                return self._apply(key, snap, seat, move, actor)
            except ConflictError:
                return None
            except (GameError, IllegalMove) as error:
                logging.warning("Texas Hold'em character move refused (%s)", type(error).__name__)
        return None

    async def on_next_timer(self, guild_id, table_id, round_id) -> None:
        """The pause between hands is over: deal the next hand when two or more seats have chips, else wait for players (the idle timer closes the table)."""
        key, out = (guild_id, table_id), None
        async with self.runtime.lock(key):
            row = self._row(key)
            if not self._open_here(row):
                self._release(key)
                return
            try:
                snap = self.store.round_snapshot(guild_id, round_id)
            except GameError:
                return
            if snap["status"] != "joining" or snap["table_id"] != table_id or self.store.latest_round(*key)["round_id"] != round_id:
                return
            if len(snap["seats"]) >= holdem.MIN_SEATS and row["thread_id"] is not None:
                try:
                    out = self._deal(key, row, snap, row["opened_by"])
                except GameError as error:
                    logging.warning("Texas Hold'em could not deal the next hand (%s)", type(error).__name__)
                    return
            else:
                self.runtime.disarm(key, "next")
                self._touch_lobby(key, round_id)
                out = self._seal(key, _Out(turn=self.render_pause(key, snap, row, waiting=True)))
        await self._publish(key, out)

    async def on_archive_timer(self, guild_id, table_id) -> None:
        self.ended.discard((guild_id, table_id))
        self.tables.pop((guild_id, table_id), None)
        self._drop_banter((guild_id, table_id))
        row = self._row((guild_id, table_id))
        if row is None or row["status"] == "open":
            return
        await self._archive(row["thread_id"])

    # --- restart ----------------------------------------------------------------------------

    async def recover(self) -> None:
        """Closes every Hold'em table left open by a previous run (the store voids an unfinished hand and cashes out every stack), then tells
        the lobby and the thread and archives it (best effort)."""
        closed = failed = 0
        for t in self.store.open_cash_tables():
            guild_id, table_id = t["guild_id"], t["table_id"]
            try:
                self.store.close_table(guild_id, table_id)
                closed += 1
            except Exception:
                failed += 1
                logging.exception("Could not close Texas Hold'em table %s on restart", table_id)
                continue
            self._release((guild_id, table_id))
            self._drop_banter((guild_id, table_id))
            self.tables.pop((guild_id, table_id), None)
            text = "This table closed when the bot restarted." + CASHED
            await self._edit(t["channel_id"], t["message_id"], content=text, view=None)
            if t["thread_id"] is not None:
                await self._post(t["thread_id"], text)
                await self._archive(t["thread_id"])
        if closed or failed:
            logging.info("Texas Hold'em recovery: %d tables closed, %d failures", closed, failed)
