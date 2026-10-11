"""I Doubt It on Discord (FEAT-21 part B1): `/game`, lobby, table thread, turns, doubt window, timers, settlement, recovery.

Rules, money and persistence live in ``games.doubt`` and the store. This module turns presses and timers into store calls
and store snapshots into messages. Every table runs under its own lock (never the channel skit lock) that covers store
work only: a handler decides under the lock and builds an ``_Out`` (texts, buttons, a sequence number); the messages go
out afterwards, one table at a time (``_Table.send_lock``), and an edit older than the last one sent is dropped.
Timers re-check the stored state under the lock, so a stale timer does nothing. Characters play by the deterministic
fallback policy (their decisions never read the clock); the model-chosen moves come later (B2).
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import discord
from discord import app_commands

from . import game_chooser
from .admin_store import ConflictError, GameError
from .games import blackjack, doubt
from .games_discord import MAX_BET, NO_CHARACTER_SEATS, NO_MENTIONS, NOT_SEATED, NOT_YOUR_TURN, OFF_HERE, STALE, GameTables, _sign
from .games_holdem_discord import BET_NOTE
from .table_runtime import TableRuntime

PREFIX = "llmcord:doubt"
DOUBT_OFF = "I Doubt It is off on this server."
ANTE_NOTE = "I Doubt It uses the server's ante; no bet needed."
BET_NEEDED = "Blackjack needs a bet: /game start game:Blackjack bet:<amount>."
MOVED_ON = "The table moved on — open My hand again."
DOUBTED_FIRST = "Someone doubted first."
NO_TABLE = "Use /game hand in your table's thread."
THREAD_FORBIDDEN = ("I couldn't open the table thread. I need the Create Public Threads and Send Messages in Threads "
                    "permissions in this channel. The table was closed.")
THREAD_FAILED = "Couldn't open the table thread; the table is closed."
THREAD_NOT_READY = "The table thread isn't ready yet; try again in a moment."
_STALE = object()  # the table moved on while a decision was made: drop it
WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}


class DoubtButton(discord.ui.DynamicItem[discord.ui.Button], template=PREFIX + r":(?P<action>[a-z_]+):(?P<table>\d+):(?P<round>\d+):(?P<moves>\d+)"):
    """One button; its custom_id carries the table, the round and the move count the message showed. Never a seat."""

    def __init__(self, action: str, table_id: int, round_id: int, moves: int, label: str = "", style=discord.ButtonStyle.secondary, disabled: bool = False):
        super().__init__(discord.ui.Button(label=label, style=style, disabled=disabled, custom_id=f"{PREFIX}:{action}:{table_id}:{round_id}:{moves}"))
        self.action, self.table_id, self.round_id, self.moves = action, table_id, round_id, moves

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["action"], int(match["table"]), int(match["round"]), int(match["moves"]), item.label or "", item.style, item.disabled)

    async def callback(self, interaction: discord.Interaction):
        await interaction.client.doubt.press(interaction, self.action, self.table_id, self.round_id, self.moves)


class _PickSelect(discord.ui.Select):
    def __init__(self, slot: int, options: list):
        super().__init__(placeholder="Cards to play (1 to 4 in all)", min_values=0, max_values=min(len(options), doubt.MAX_PLAY), options=options)
        self.slot = slot

    async def callback(self, interaction: discord.Interaction):
        self.view.picks[self.slot] = list(self.values)
        await interaction.response.defer()


class _Picker(discord.ui.View):
    """The private hand picker: the options are "n x rank" for each rank held (at most 25 per select); Play submits the choice."""

    def __init__(self, table_id: int, round_id: int, moves: int, held: dict):
        super().__init__(timeout=600)
        self.table_id, self.round_id, self.moves, self.picks = table_id, round_id, moves, {}
        options = [[]]
        for rank in doubt.RANKS:
            if rank not in held:
                continue
            mine = [discord.SelectOption(label=f"{n} × {rank}", value=f"{rank}:{n}") for n in range(1, min(held[rank], doubt.MAX_PLAY) + 1)]
            if len(options[-1]) + len(mine) > 25:
                options.append([])
            options[-1].extend(mine)
        for slot, chunk in enumerate(options):
            self.add_item(_PickSelect(slot, chunk))
        play = discord.ui.Button(label="Play", style=discord.ButtonStyle.primary)
        play.callback = self.submit
        self.add_item(play)

    async def submit(self, interaction: discord.Interaction):
        counts, twice = {}, False
        for values in self.picks.values():
            for value in values:
                rank, _, n = value.partition(":")
                twice = twice or rank in counts
                counts[rank] = int(n)
        if twice:
            await interaction.client.doubt.answer(interaction, "Pick one amount for each rank.")
            return
        await interaction.client.doubt.play_from_picker(interaction, self.table_id, self.round_id, self.moves, counts)


@dataclass(eq=False)
class _Table:
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    issued: int = 0  # numbers the outgoing edits, assigned under the table lock
    sent: dict = field(default_factory=dict)  # newest delivered number per message kind
    talk: int = 0  # member messages in the thread since the turn message was posted
    deadlines: dict = field(default_factory=dict)  # lobby / turn / window -> unix time
    one_card: bool = False  # a seat already reached its last card (that event is sent once)


@dataclass
class _Out:
    """What a locked step decided: a private reply and the messages to send after the lock is released."""
    say: str | None = None
    key: tuple | None = None
    ok: bool = False  # the reply confirms a played move
    events: list = field(default_factory=list)  # public lines for the thread, in order
    lobby: tuple | None = None  # (content, view or None)
    turn: tuple | None = None  # (content, view)
    board: bool = False
    start: str | None = None  # the board text: post the board and the turn message, pin the board
    final: tuple | None = None  # (result text, lobby text, turn text)
    advanced: bool = False  # a move was made: the turn message may be moved to the bottom
    seq: int = 0
    talk: tuple | None = None  # (public text, is the game's end): the event turn for the characters at the table
    closing: str | None = None  # the table was closed from outside: this line goes to the thread and the turn message loses its buttons
    closed: bool = False  # the table is over: forget it once this is delivered


def _claim(count: int, rank: str) -> str:
    return f"{WORDS.get(count, count)} {rank}{'s' if count != 1 else ''}"


class DoubtTables:
    LOBBY_SECONDS = 300
    WINDOW_SECONDS = 15
    CHARACTER_SECONDS = 2
    ARCHIVE_SECONDS = 600
    BOARD_DEBOUNCE_SECONDS = 1
    RETRY_SECONDS = 5  # read once at construction (TableRuntime.retry_seconds); tests set runtime.retry_seconds
    BUMP_AFTER = 8
    MODEL_SECONDS = 20
    EVENT_COOLDOWN_SECONDS = 20
    SWEEP_SECONDS = 30

    def __init__(self, bot, clock=time.time):
        self.bot, self.clock = bot, clock
        self.runtime = TableRuntime(label="I Doubt It", retry_seconds=self.RETRY_SECONDS)
        self.tables: dict[tuple[int, int], _Table] = {}
        self.threads: dict[int, tuple[int, int]] = {}  # thread id -> (guild, table) of the tables being played
        self.banter: dict[int, tuple[int, int]] = {}  # thread id -> (guild, table) of finished games, until the archive timer
        self.event_at: dict[tuple[int, int], float] = {}  # table -> clock time of its last event turn
        self.event_tasks: set[asyncio.Task] = set()
        self.dirty: set[tuple[int, int]] = set()
        self.board_tasks: dict[tuple[int, int], asyncio.Task] = {}
        self.ended: set[tuple[int, int]] = set()  # tables this object closed itself (settled, timed out, left): the sweeper leaves them alone
        self.talk_failed: dict[tuple[int, object], int] = {}  # model attempts without a usage row, per (guild, server day)

    @property
    def store(self):
        return self.bot.store

    @property
    def closed(self) -> bool:
        return self.runtime.closed

    @property
    def sweeper(self) -> asyncio.Task | None:
        return self.runtime.sweeper

    def table(self, key) -> _Table:
        table = self.tables.get(key)
        if table is None:
            table = self.tables[key] = _Table()
        return table

    # --- timers -----------------------------------------------------------------------------

    def arm(self, key, kind: str, delay: float, *args) -> None:
        if self.closed:
            return
        hook = {"lobby": self.on_lobby_timer, "window": self.on_window_timer, "turn": self.on_turn_timer,
                "character": self.on_character_timer, "archive": self.on_archive_timer}[kind]
        self.runtime.arm(key, kind, delay, self._run, kind, hook, key, args, name=f"doubt-{kind}-{key[1]}")

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
        for kind in ("lobby", "window", "turn", "character"):
            self.runtime.disarm(key, kind)
        self.threads = {thread: k for thread, k in self.threads.items() if k != key}
        self.event_at.pop(key, None)
        self.dirty.discard(key)

    def _forget(self, key) -> None:
        """This object closed the table (in the store): release it, and keep the sweeper off it until its archive timer."""
        self.ended.add(key)
        self._release(key)

    # --- tables closed from outside (dashboard, /admin games channel) -------------------------

    def _ensure_sweeper(self) -> None:
        if self.tables:
            self.runtime.ensure_sweeper(interval=lambda: self.SWEEP_SECONDS, held=lambda: bool(self._known()), sweep=self.sweep, name="doubt-sweep")

    def _known(self) -> set:
        return {*self.tables, *self.threads.values(), *(k[0] for k in self.runtime.timers if k[1] != "archive")}

    async def sweep(self) -> None:
        """Cleans up tables closed in the store by someone else (dashboard switch, channel removal) while this object still holds them."""
        for key in self._known():
            await self._reap(key)

    async def _reap(self, key) -> None:
        """If the table is closed in the store and this object did not close it, say so (lobby, turn message, thread), archive and forget it."""
        if key in self.ended:
            self._release(key)  # idempotent: a thread registered while the table closed must not keep it known
            return
        if key not in self.tables and key not in self.threads.values() and not any(k[0] == key for k in self.runtime.timers):
            return
        row = self._row(key)
        if row is not None and row["game"] == "doubt" and row["status"] == "open":
            return
        async with self.runtime.lock(key):
            out = self._closed_out(key)
        await self._publish(key, out)

    def _closed_out(self, key) -> _Out | None:
        """Under the table lock: the closing messages for a table that is closed in the store, or None (still open, or already handled)."""
        row = self._row(key)
        if key in self.ended or (row is not None and row["game"] == "doubt" and row["status"] == "open"):
            return None
        if row is None or row["game"] != "doubt":
            self._release(key)
            self.tables.pop(key, None)
            return None
        guild_id = key[0]
        if not self.store.game_enabled(guild_id, "doubt"):
            reason = "I Doubt It was turned off on this server."
        elif row["channel_id"] not in self.store.game_channels(guild_id):
            reason = "games were turned off here."
        else:
            reason = "the table was closed."
        self._forget(key)
        self.arm(key, "archive", self.ARCHIVE_SECONDS)
        text = f"Table closed: {reason}" + self._refunded(key)
        return self._seal(key, _Out(lobby=(text, None), closing=text, closed=True))

    async def close_for_channel(self, guild_id, channel_id, table_id) -> int | None:
        """Turns games off in a channel whose open table is this game's: the store step under the table lock, then the closing messages in order.
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

    def _after(self, key, snap) -> None:
        """Arms the timers the new state needs. Call under the table lock."""
        table, view = self.table(key), snap["view"]
        self.runtime.disarm(key, "lobby")
        if snap["status"] != "playing" or not view:
            for kind in ("window", "turn", "character"):
                self.runtime.disarm(key, kind)
            return
        rid, moves, seats = snap["round_id"], snap["moves"], snap["seats"]
        window, turn = view["window"], view["turn"]
        if window:
            table.deadlines["window"] = self.clock() + self.WINDOW_SECONDS
            self.arm(key, "window", self.WINDOW_SECONDS, rid, moves)
        else:
            table.deadlines.pop("window", None)
            self.runtime.disarm(key, "window")
        can_play = turn is not None and not window  # a play is only waited for once the doubt window is closed
        if can_play and seats[turn]["kind"] == "member":
            delay = (snap["rules"] or {}).get("turn_timeout", 60)
            table.deadlines["turn"] = self.clock() + delay
            self.arm(key, "turn", delay, rid, moves)
        else:
            table.deadlines.pop("turn", None)
            self.runtime.disarm(key, "turn")
        doubters = window and any(s["kind"] == "character" and view["active"][s["index"]] and s["index"] != window["claimer"] for s in seats)
        if (can_play and seats[turn]["kind"] == "character") or doubters:
            self.arm(key, "character", self.CHARACTER_SECONDS, rid, moves)
        else:
            self.runtime.disarm(key, "character")

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
            logging.warning("Could not edit I Doubt It message %s in channel %s: %s", message_id, channel_id, type(error).__name__)
        except Exception:
            logging.exception("Could not edit I Doubt It message %s in channel %s", message_id, channel_id)
        return None

    async def _delete(self, channel_id, message_id) -> bool:
        if channel_id is None or message_id is None:
            return False
        try:
            channel = await self._channel(channel_id)
            await channel.get_partial_message(message_id).delete()
            return True
        except discord.HTTPException as error:
            logging.info("Could not delete I Doubt It message %s (%s)", message_id, type(error).__name__)
        except Exception:
            logging.exception("Could not delete I Doubt It message %s", message_id)
        return False

    async def _post(self, channel_id, content, view=None):
        if channel_id is None:
            return None
        try:
            channel = await self._channel(channel_id)
            return await channel.send(content, view=view, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning("Could not post to I Doubt It thread %s: %s", channel_id, type(error).__name__)
        except Exception:
            logging.exception("Could not post to I Doubt It thread %s", channel_id)
        return None

    async def _say(self, interaction, text: str) -> None:
        """An ephemeral reply: the first response, or a followup once the interaction was acknowledged."""
        try:
            if interaction.response.is_done():
                await interaction.followup.send(text, ephemeral=True, allowed_mentions=NO_MENTIONS)
            else:
                await interaction.response.send_message(text, ephemeral=True, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning("I Doubt It reply failed: %s", type(error).__name__)

    async def answer(self, interaction, text: str, done: bool = False) -> None:
        """A reply to the picker: ``done`` replaces the picker message, anything else is a new private message (the picker stays)."""
        if done and not interaction.response.is_done() and getattr(interaction, "message", None) is not None:
            try:
                await interaction.response.edit_message(content=text, view=None, allowed_mentions=NO_MENTIONS)
                return
            except discord.HTTPException as error:
                logging.warning("I Doubt It picker edit failed: %s", type(error).__name__)
        await self._say(interaction, text)

    async def _ack(self, interaction) -> None:
        try:
            await interaction.response.defer()
        except discord.HTTPException as error:
            logging.warning("I Doubt It could not acknowledge the press: %s", type(error).__name__)

    def _refunded(self, key, past: bool = False) -> str:
        """" The antes are refunded." (``past``: "were") when the table's round had an ante, else nothing."""
        try:
            ante = (self.store.latest_round(*key)["rules"] or {}).get("ante", 0)
        except GameError:
            ante = 0
        return (" The antes were refunded." if past else " The antes are refunded.") if ante else ""

    def _host(self, row, snap) -> int:
        """The table's host: the opener while seated, else the first seated member."""
        members = [s["ref_id"] for s in snap["seats"] if s["kind"] == "member"]
        return row["opened_by"] if row["opened_by"] in members or not members else members[0]

    def _is_host(self, row, snap, user_id) -> bool:
        """The opener while seated; once the opener left, any seated member."""
        members = {s["ref_id"] for s in snap["seats"] if s["kind"] == "member"}
        return user_id == row["opened_by"] if row["opened_by"] in members else user_id in members

    def _row(self, key):
        return self.store.one("SELECT * FROM game_tables WHERE id=? AND guild_id=?", (key[1], key[0]))

    # --- rendering --------------------------------------------------------------------------

    @staticmethod
    def _who(snap, index) -> str:
        return GameTables._who({**snap["seats"][index], "brought_by": None})

    def _stamp(self, key, kind) -> str:
        deadline = self.table(key).deadlines.get(kind)
        return f"<t:{int(deadline)}:R>" if deadline else ""

    def render_lobby(self, key, snap, row) -> tuple[str, discord.ui.View]:
        rules = snap["rules"] or doubt.normalize_rules(None)
        name, pot, ante = self.store.currency_name(key[0]), sum(s["stake"] for s in snap["seats"]), rules["ante"]
        lines = [f"**I Doubt It** · table {key[1]}", f"Ante: {ante:,} {name} · Pot: {pot:,} {name}" if ante else f"No ante · Pot: {pot:,} {name}",
                 f"Host: <@{self._host(row, snap)}>", ""]
        for seat in snap["seats"]:
            tag = " (house)" if seat["kind"] == "character" and not seat["brought_by"] else ""
            lines.append(f"{seat['index'] + 1}. {GameTables._who(seat)}{tag}")
        stamp = self._stamp(key, "lobby")
        lines.append("")
        lines.append(f"Join with the button. The host (if the host has left, any player) starts the game with {doubt.MIN_SEATS} to {doubt.MAX_SEATS} seats"
                     + (f"; the table closes {stamp} if it hasn't started." if stamp else "."))
        lines.append("-# " + doubt.rules_text(rules))
        view, tid, rid = discord.ui.View(timeout=None), key[1], snap["round_id"]
        spec = [("join", "Join", discord.ButtonStyle.primary, False), ("leave", "Leave", discord.ButtonStyle.secondary, False)]
        if self.store.binding(row["channel_id"]) is not None:
            spec.append(("fill", "Fill seats", discord.ButtonStyle.secondary, False))
        spec += [("start", "Start", discord.ButtonStyle.success, len(snap["seats"]) < doubt.MIN_SEATS or row["thread_id"] is None), ("help", "How to play", discord.ButtonStyle.secondary, False)]
        for action, label, style, disabled in spec:
            view.add_item(DoubtButton(action, tid, rid, 0, label, style, disabled))
        return "\n".join(lines), view

    def _in_play_view(self, key, snap) -> discord.ui.View:
        view = discord.ui.View(timeout=None)
        view.add_item(DoubtButton("help", key[1], snap["round_id"], 0, "How to play"))
        return view

    def render_board(self, key, snap) -> str:
        view, rules, name = snap["view"], snap["rules"] or {}, self.store.currency_name(key[0])
        pot = sum(s["stake"] for s in snap["seats"])
        lines = [f"**I Doubt It** · table {key[1]}" + (f" · pot {pot:,} {name} (ante {rules.get('ante', 0):,})" if pot else "")]
        for seat in snap["seats"]:
            i = seat["index"]
            if not view["active"][i]:
                state = "forfeited"
            elif view["finished"] and view["winner"] == i:
                state = f"{doubt.cards_text(view['counts'][i])} · winner"
            else:
                state = doubt.cards_text(view["counts"][i]) + (" · on turn" if view["turn"] == i else "")
            lines.append(f"{i + 1}. {GameTables._who(seat)} — {state}")
        if view["finished"]:
            lines.append("Game over.")
        else:
            lines.append(f"Pile: {doubt.cards_text(view['pile'])} · Rank to claim: {view['rank']}")
        lines.append("-# " + doubt.rules_text(snap["rules"]))
        lines.append(f"-# Seed: {snap['seed']} (hash {snap['seed_hash'][:12]})" if snap["seed"] else f"-# Seed hash: {snap['seed_hash'][:12]}")
        return "\n".join(lines)

    def render_turn(self, key, snap) -> tuple[str, discord.ui.View]:
        view, seats = snap["view"], snap["seats"]
        last = view["history"][-1] if view["history"] else None
        if last is None:
            lines = ["The game has started."]
        elif last["kind"] == "play":
            lines = [f"{self._who(snap, last['seat'])} claims {_claim(last['count'], last['rank'])}."]
        else:
            taker = last["claimer"] if last["bluff"] else last["doubter"]
            lines = [f"{self._who(snap, taker)} took the pile after a doubt."]
        lines.append(f"Pile: {doubt.cards_text(view['pile'])}.")
        turn = view["turn"]
        if turn is not None:
            stamp = self._stamp(key, "turn")
            who = self._who(snap, turn)
            lines.append(f"Next: {who}, claim {view['rank']}" + (f" — {stamp}." if stamp and seats[turn]["kind"] == "member" else "."))
        if view["window"]:
            window_stamp = self._stamp(key, "window")
            lines.append("Doubt window open" + (f" {window_stamp}" if window_stamp else "") + ".")
        buttons = discord.ui.View(timeout=None)
        buttons.add_item(DoubtButton("doubt", key[1], snap["round_id"], snap["moves"], "Doubt", discord.ButtonStyle.danger, view["window"] is None))
        buttons.add_item(DoubtButton("hand", key[1], snap["round_id"], snap["moves"], "My hand", discord.ButtonStyle.primary))
        return "\n".join(lines), buttons

    def _reveal_text(self, snap, pile_before: int, names=None) -> str:
        rev = snap["view"]["history"][-1]
        cards = ", ".join(f"{rank}{suit}" for rank, suit in rev["cards"])
        who = (lambda s, i: names[i]) if names else self._who
        claimer, doubter = who(snap, rev["claimer"]), who(snap, rev["doubter"])
        head = f"Doubt! {doubter} doubted {claimer}'s {_claim(len(rev['cards']), rev['rank'])}. They were {cards}"
        if not rev["bluff"] and snap["view"]["finished"]:
            return head + f" — the claim was true, and {claimer} played their last cards."
        taker = who(snap, rev["claimer"] if rev["bluff"] else rev["doubter"])
        return head + f" — {taker} takes the pile ({doubt.cards_text(pile_before)})."

    def _result(self, key, snap) -> tuple[str, str, str]:
        """(thread result, one-line lobby text, final turn text) for a settled round."""
        name, seats, win = self.store.currency_name(key[0]), snap["seats"], snap["view"]["winner"]
        pot, who = sum(s["stake"] for s in seats), self._who(snap, win)
        head = f"**{who} wins the game" + (f" and takes the pot of {pot:,} {name}.**" if pot else ".**")
        lines = [head]
        for seat in seats:
            net = (seat["payout"] or 0) - seat["stake"]
            lines.append(f"• {GameTables._who(seat)}: {_sign(net) or 'even'}" + (" (forfeited)" if seat["outcome"] == "forfeit" else ""))
        if snap["seed"]:
            lines.append(f"-# Seed: {snap['seed']} (hash {snap['seed_hash'][:12]})")
        lobby = f"{who} won I Doubt It" + (f" and took the pot of {pot:,} {name}" if pot else "") + ". Table closed."
        return "\n".join(lines), lobby, f"Game over — {who} won."

    # --- the common step machinery ----------------------------------------------------------

    def _seal(self, key, out: _Out) -> _Out:
        table = self.table(key)
        table.issued += 1
        out.key, out.seq = key, table.issued
        return out

    def _load(self, key, round_id):
        """((table row, snapshot), None) for an open I Doubt It table and one of its rounds, else (None, refusal)."""
        row = self._row(key)
        if row is None or row["game"] != "doubt" or row["status"] != "open":
            return None, _Out(STALE)
        try:
            snap = self.store.round_snapshot(key[0], round_id)
        except GameError:
            return None, _Out(STALE)
        if snap["table_id"] != key[1]:
            return None, _Out(STALE)
        return (row, snap), None

    def _timer_snap(self, key, round_id, moves):
        """Under the table lock: the playing snapshot a timer may act on, or None (the table moved on or closed)."""
        row = self._row(key)
        if row is None or row["status"] != "open" or row["game"] != "doubt":
            self._release(key)
            return None
        try:
            snap = self.store.round_snapshot(*key[:1], round_id)
        except GameError:
            return None
        if snap["table_id"] != key[1] or snap["status"] != "playing" or snap["moves"] != moves or not snap["view"]:
            return None
        return snap

    def _apply(self, key, snap, seat, move, actor, note=None) -> _Out:
        """Makes one move under the table lock; raises ConflictError / GameError from the store."""
        before, new = snap, self.store.play(key[0], snap["round_id"], seat, move, actor, snap["moves"])
        out = _Out(advanced=True)
        if note:
            out.events.append(note)
        if move == "doubt":
            out.events.append(self._reveal_text(new, before["view"]["pile"]))
        if move == "forfeit":
            out.events.append(f"{self._who(before, seat)} forfeits and is out of the game.")
        table = self.table(key)
        if new["status"] == "settled":
            out.final = self._result(key, new)
            if self.talk_allowed(key[0]):
                out.talk = (f"Game over — {self._talk_names(key[0], new)[new['view']['winner']]} won the game.", True)
            self.store.close_table(*key)
            self._forget(key)
            if (row := self._row(key)) is not None and row["thread_id"] is not None:
                self.banter[row["thread_id"]] = key
            self.arm(key, "archive", self.ARCHIVE_SECONDS)
        else:
            self._after(key, new)
            out.turn = self.render_turn(key, new)
            out.board = True
            low = [i for i, n in enumerate(new["view"]["counts"]) if n == 1 and new["view"]["active"][i]]
            if (move == "doubt" or (low and not table.one_card)) and self.clock() - self.event_at.get(key, float("-inf")) >= self.EVENT_COOLDOWN_SECONDS \
                    and self.talk_allowed(key[0]):
                names = self._talk_names(key[0], new)
                text = self._reveal_text(new, before["view"]["pile"], names) if move == "doubt" else f"{names[low[0]]} is down to one card."
                self.event_at[key] = self.clock()
                table.one_card = table.one_card or move != "doubt"
                out.talk = (text, False)
        return self._seal(key, out)

    async def _publish(self, key, out: _Out | None) -> None:
        """After the table lock is released: the thread lines, then the edits (an edit older than the last one sent is dropped)."""
        if out is None:
            return
        await self._send(key, out)

    async def _send(self, key, out: _Out) -> None:
        table = self.table(key)
        async with table.send_lock:
            row = self._row(key)
            if row is None:
                return
            if row["status"] != "open" and not (out.final or out.closed or out.closing):
                return  # a step sealed before the table was closed: it must not repaint the closed table
            thread_id, channel_id = row["thread_id"], row["channel_id"]
            if out.start:
                if not await self._open_play(key, out, row):
                    return
            for text in out.events:
                await self._post(thread_id, text)
            if out.final:
                result, lobby, turn = out.final
                await self._post(thread_id, result)
                self._talk_event(thread_id, out)
                await self._edit(thread_id, row["turn_message_id"], content=turn, view=None)
                self.dirty.discard(key)
                await self._edit_board(key)
                await self._edit(channel_id, row["message_id"], content=lobby, view=None)
                return
            self._talk_event(thread_id, out)
            if out.closing:
                if out.seq > table.sent.get("turn", 0):
                    table.sent["turn"] = out.seq
                    await self._edit(thread_id, row["turn_message_id"], content="Table closed.", view=None)
                await self._post(thread_id, out.closing)
            if out.lobby and out.seq > table.sent.get("lobby", 0):
                table.sent["lobby"] = out.seq
                await self._edit(channel_id, row["message_id"], content=out.lobby[0], view=out.lobby[1])
            if out.turn and not out.start and out.seq > table.sent.get("turn", 0):
                table.sent["turn"] = out.seq
                await self._turn_message(key, out, row)
            if out.board:
                self._mark_board(key)

    def _talk_event(self, thread_id, out) -> None:
        """Fire and forget: the characters at the table react to a public event. A move never waits for it."""
        if out.talk is None or thread_id is None or self.closed:
            return
        text = out.talk[0]
        out.talk = None

        async def run():
            try:
                await self.bot.run_table_event(await self._channel(thread_id), text)
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("I Doubt It event turn failed (thread %s)", thread_id)

        task = asyncio.create_task(run(), name=f"doubt-event-{thread_id}")
        self.event_tasks.add(task)
        task.add_done_callback(self.event_tasks.discard)

    async def _turn_message(self, key, out, row) -> None:
        table, thread_id, old = self.table(key), row["thread_id"], row["turn_message_id"]
        content, view = out.turn
        if not (out.advanced and table.talk >= self.BUMP_AFTER and old is not None):
            if old is not None and await self._edit(thread_id, old, content=content, view=view) is not False:
                return
        message = await self._post(thread_id, content, view)
        if message is None:
            return
        self.store.set_turn_message(key[0], key[1], message.id)
        table.talk = 0
        if old is not None:
            await self._delete(thread_id, old)

    async def _open_play(self, key, out, row) -> bool:
        """Posts the board (pinned) and the turn message in the thread. False when the thread cannot be written to; the table is then closed."""
        thread_id = row["thread_id"]
        board = await self._post(thread_id, out.start)
        turn = await self._post(thread_id, *out.turn) if board is not None else None
        if board is None or turn is None:
            async with self.runtime.lock(key):
                refunded = self._refunded(key)
                if (current := self._row(key)) is not None and current["status"] == "open":
                    self.store.close_table(*key)
                self._forget(key)
            await self._edit(row["channel_id"], row["message_id"], content="Table closed: I couldn't post in the table thread." + refunded, view=None)
            self.arm(key, "archive", self.ARCHIVE_SECONDS)
            return False
        try:
            await board.pin()
        except discord.HTTPException as error:
            logging.info("Could not pin the board (%s)", type(error).__name__)
        self.store.set_board_message(key[0], key[1], board.id)
        self.store.set_turn_message(key[0], key[1], turn.id)
        self.table(key).talk = 0
        self.table(key).sent["turn"] = out.seq
        return True

    # --- the board (debounced) --------------------------------------------------------------

    def _mark_board(self, key) -> None:
        self.dirty.add(key)
        if key not in self.board_tasks and not self.closed:
            task = self.board_tasks[key] = asyncio.create_task(self._debounce(key), name=f"doubt-board-{key[1]}")
            task.add_done_callback(lambda done: self.board_tasks.get(key) is done and self.board_tasks.pop(key, None))

    async def _debounce(self, key) -> None:
        await asyncio.sleep(self.BOARD_DEBOUNCE_SECONDS)
        if key in self.dirty:
            self.dirty.discard(key)
            await self._edit_board(key)

    async def _edit_board(self, key) -> None:
        try:
            snap = self.store.latest_round(*key)
        except GameError:
            return
        if snap["view"] and snap["board_message_id"] is not None:
            await self._edit(snap["thread_id"], snap["board_message_id"], content=self.render_board(key, snap))

    async def flush(self) -> None:
        """Sends the pending board edits now."""
        keys = list(self.dirty)
        self.dirty.clear()
        for key in keys:
            await self._edit_board(key)

    # --- /game start ------------------------------------------------------------------------

    async def start_command(self, interaction: discord.Interaction, favorites: bool = False) -> None:
        guild_id = interaction.guild_id
        channel_id = getattr(interaction, "channel_id", None) or interaction.channel.id
        user = interaction.user
        if user.bot:
            await self._say(interaction, "Bots cannot play I Doubt It.")
            return
        if not self.store.game_enabled(guild_id, "doubt"):
            await self._say(interaction, DOUBT_OFF)
            return
        if channel_id not in self.store.game_channels(guild_id):
            await self._say(interaction, OFF_HERE)
            return
        _, binding = self.bot.location(interaction.channel)
        space_id, note = (binding["space_id"] if binding else None), ""
        if favorites and not binding:
            note = NO_CHARACTER_SEATS
        try:
            opened = self.store.open_table(guild_id, channel_id, "doubt", user.id)
        except GameError as error:
            await self._say(interaction, str(error))
            return
        key, round_id = (guild_id, opened["table_id"]), opened["round_id"]
        self.table(key)
        self._ensure_sweeper()
        async with self.runtime.lock(key):
            try:
                if favorites and space_id is not None:
                    result = self.store.join_with_favorites(guild_id, round_id, user.id, 0, space_id)
                    snap, note = result["snapshot"], self.bot.games._favorites_note(result, note)
                else:
                    snap = self.store.join_round(guild_id, round_id, user.id, 0)
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
            if table.issued > out.seq and (row := self._row(key))["status"] == "open":  # someone joined before the message existed
                fresh = self._seal(key, _Out(lobby=self.render_lobby(key, self.store.latest_round(*key), row)))
        await self._publish(key, fresh)
        try:
            partial = await self._channel(channel_id)
            thread = await partial.get_partial_message(message.id).create_thread(name=f"I Doubt It · table {key[1]}", auto_archive_duration=1440)
        except discord.HTTPException as error:
            logging.warning("Could not open the I Doubt It thread (%s)", type(error).__name__)
            await self._thread_failed(interaction, key, isinstance(error, discord.Forbidden))
            return
        ready = None
        async with self.runtime.lock(key):  # Start waits for the thread
            row = self._row(key)
            if row["status"] == "open":
                self.store.set_table_thread(*key, thread.id)
                self.threads[thread.id] = key
                ready = self._seal(key, _Out(lobby=self.render_lobby(key, self.store.latest_round(*key), row)))
        if ready is None:  # closed while the thread was being made: nothing else to say there, just archive it
            try:
                await thread.edit(archived=True, locked=True)
            except Exception as error:
                logging.warning("Could not archive the unused I Doubt It thread (%s)", type(error).__name__)
        await self._publish(key, ready)
        if note:
            await self._say(interaction, note)

    async def _thread_failed(self, interaction, key, forbidden: bool) -> None:
        async with self.runtime.lock(key):
            refunded = self._refunded(key)
            row = self._row(key)
            if row is not None and row["status"] == "open":
                self.store.close_table(*key)
            self._forget(key)
            out = self._seal(key, _Out(lobby=("Table closed: I couldn't open a thread here." + refunded, None), closed=True))
        await self._publish(key, out)
        self.tables.pop(key, None)
        self.ended.discard(key)
        await self._say(interaction, (THREAD_FORBIDDEN if forbidden else THREAD_FAILED) + refunded)

    # --- buttons ----------------------------------------------------------------------------

    def _help_text(self, guild_id, table_id, round_id) -> str:
        rules = None
        if guild_id is not None:
            try:
                snap = self.store.round_snapshot(guild_id, round_id)
                rules = snap["rules"] if snap["table_id"] == table_id else None
            except GameError:
                pass
            rules = rules or self.store.game_rules(guild_id, "doubt")
        return doubt.how_to_play(rules)

    async def press(self, interaction: discord.Interaction, action: str, table_id: int, round_id: int, moves: int) -> None:
        if action == "help":  # a guide, not a move: no state, no seat needed, works on old messages
            await self._say(interaction, self._help_text(interaction.guild_id, table_id, round_id))
            return
        guild_id = interaction.guild_id
        if guild_id is None or interaction.user.bot:
            await self._say(interaction, STALE)
            return
        key = (guild_id, table_id)
        if action == "hand":
            await self._show_hand(interaction, key, round_id)
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
            if action == "join":
                return self._join(key, row, snap, user_id)
            if action == "leave":
                return self._leave(key, row, snap, user_id)
            if action == "fill":
                return self._fill(key, row, snap, user_id)
            if action == "start":
                return self._start(key, row, snap, user_id)
            if action == "doubt":
                return self._doubt(key, snap, user_id, moves)
            return _Out(STALE)
        except ConflictError:
            return _Out(DOUBTED_FIRST if action == "doubt" else MOVED_ON)
        except GameError as error:
            return _Out(str(error))

    def _touch_lobby(self, key, round_id) -> None:
        """A join restarts the idle clock of the lobby."""
        self.table(key).deadlines["lobby"] = self.clock() + self.LOBBY_SECONDS
        self.arm(key, "lobby", self.LOBBY_SECONDS, round_id)

    def _join(self, key, row, snap, user_id) -> _Out:
        snap = self.store.join_round(key[0], snap["round_id"], user_id, 0)
        self._touch_lobby(key, snap["round_id"])
        ante, name = (snap["rules"] or {}).get("ante", 0), self.store.currency_name(key[0])
        say = "You joined the table." + (f" Your ante of {ante:,} {name} is in the pot." if ante else "")
        return self._seal(key, _Out(say, lobby=self.render_lobby(key, snap, row)))

    def _leave(self, key, row, snap, user_id) -> _Out:
        snap = self.store.leave_round(key[0], snap["round_id"], user_id)
        if not snap["seats"]:
            self.store.close_table(*key)
            self._forget(key)
            self.arm(key, "archive", self.ARCHIVE_SECONDS)
            return self._seal(key, _Out("You left the table. It is closed.", lobby=("Table closed: everyone left.", None), closed=True))
        return self._seal(key, _Out("You left the table. Your ante is refunded.", lobby=self.render_lobby(key, snap, row)))

    def _fill(self, key, row, snap, user_id) -> _Out:
        if not self._is_host(row, snap, user_id):
            return _Out("Only the host can fill seats.")
        guild_id, binding = key[0], self.store.binding(row["channel_id"])
        if binding is None:
            return _Out("Characters can't sit at tables in this channel.")
        taken = {s["ref_id"] for s in snap["seats"] if s["kind"] == "character"}
        eligible = {r["id"] for r in self.store.eligible_characters(guild_id, binding["space_id"])}
        pool = [c for c in self.store.get_cast(row["channel_id"], None) if c in eligible and c not in taken]
        need, seated, skipped = (doubt.MIN_SEATS - len(snap["seats"])) if len(snap["seats"]) < doubt.MIN_SEATS else 1, [], []
        while need > 0 and pool:
            chunk, pool = pool[:need], pool[need:]
            result = self.store.fill_seats(guild_id, snap["round_id"], user_id, chunk, binding["space_id"])
            snap, need = result["snapshot"], need - len(result["seated"])
            seated += result["seated"]
            skipped += result["skipped"]
        if not seated:
            return _Out("No one is left in the cast who can take a seat.")
        self._touch_lobby(key, snap["round_id"])
        say = "Seated: " + ", ".join(s["name"] for s in seated) + "."
        if skipped:
            say += " Not seated: " + ", ".join(s["name"] for s in skipped) + "."
        return self._seal(key, _Out(say, lobby=self.render_lobby(key, snap, row)))

    def _start(self, key, row, snap, user_id) -> _Out:
        if not self._is_host(row, snap, user_id):
            return _Out("Only the host can start the game.")
        if row["thread_id"] is None:
            return _Out(THREAD_NOT_READY)
        if len(snap["seats"]) < doubt.MIN_SEATS:
            return _Out(f"I Doubt It needs at least {doubt.MIN_SEATS} seats to start.")
        snap = self.store.deal(key[0], snap["round_id"])
        self._after(key, snap)
        out = _Out(start=self.render_board(key, snap), turn=self.render_turn(key, snap))
        out.lobby = ("In play — see the thread.", self._in_play_view(key, snap))
        return self._seal(key, out)

    def _doubt(self, key, snap, user_id, moves) -> _Out:
        seat = self.store.member_seat_index(key[0], snap["round_id"], user_id)
        if seat is None:
            return _Out(NOT_SEATED)
        if snap["status"] != "playing":
            return _Out(STALE)
        view = snap["view"]
        if snap["moves"] != moves:
            last = view["history"][-1] if view["history"] else None
            return _Out(DOUBTED_FIRST if last and last["kind"] == "reveal" else MOVED_ON)
        if view["window"] is None:
            return _Out("There's no claim to doubt right now.")
        if view["window"]["claimer"] == seat:
            return _Out("You can't doubt your own claim.")
        return self._apply(key, snap, seat, "doubt", "member")

    # --- the hand and the picker ------------------------------------------------------------

    async def hand_command(self, interaction: discord.Interaction) -> None:
        guild_id = interaction.guild_id
        table = self.store.table_for_thread(guild_id, interaction.channel.id) if guild_id and interaction.channel else None
        if table is None:
            await self._say(interaction, NO_TABLE)
            return
        await self._show_hand(interaction, (guild_id, table["id"]), None)

    async def _show_hand(self, interaction, key, round_id) -> None:
        async with self.runtime.lock(key):
            text, picker = self._hand(key, round_id, interaction.user.id)
        kwargs = {"view": _Picker(*picker)} if picker else {}
        try:
            if interaction.response.is_done():
                await interaction.followup.send(text, ephemeral=True, allowed_mentions=NO_MENTIONS, **kwargs)
            else:
                await interaction.response.send_message(text, ephemeral=True, allowed_mentions=NO_MENTIONS, **kwargs)
        except discord.HTTPException as error:
            logging.warning("I Doubt It hand reply failed: %s", type(error).__name__)

    def _hand(self, key, round_id, user_id):
        """(private text, picker arguments or None): the presser's own hand only; the picker only on their turn."""
        guild_id = key[0]
        try:
            snap = self.store.round_snapshot(guild_id, round_id) if round_id is not None else self.store.latest_round(*key)
        except GameError:
            return STALE, None
        if snap["table_id"] != key[1]:
            return STALE, None
        seat = self.store.member_seat_index(guild_id, snap["round_id"], user_id)
        if seat is None:
            return NOT_SEATED, None
        if snap["status"] == "joining":
            return "The game hasn't started yet.", None
        if snap["status"] != "playing":
            return "This game is over.", None
        view = self.store.seat_view(guild_id, snap["round_id"], seat)
        held = doubt.held_counts(view.hand)
        lines = [f"**Your hand** ({doubt.cards_text(len(view.hand))})"]
        lines += [f"{rank}: " + " ".join(f"{r}{s}" for r, s in view.hand if r == rank) for rank in doubt.RANKS if rank in held]
        window = view.window
        playable = view.turn == seat and view.hand and not (window and view.counts[window.claimer] == 0)
        if playable:
            lines.append(f"\nYour turn: claim {view.rank}. Choose what to play (1 to {doubt.MAX_PLAY} cards), then press Play.")
        elif view.turn is not None:
            lines.append(f"\nOn turn: {self._who(snap, view.turn)}, claiming {view.rank}.")
        return "\n".join(lines), ((key[1], snap["round_id"], snap["moves"], held) if playable else None)

    async def play_from_picker(self, interaction: discord.Interaction, table_id: int, round_id: int, moves: int, counts: dict) -> None:
        guild_id = interaction.guild_id
        if guild_id is None or interaction.user.bot:
            await self.answer(interaction, STALE)
            return
        key = (guild_id, table_id)
        async with self.runtime.lock(key):
            out = self._picker_play(key, interaction.user.id, round_id, moves, counts)
        await self.answer(interaction, out.say, done=out.ok)  # answer first, so a slow edit cannot make it miss Discord's 3 seconds
        await self._publish(key, out)
        if out.say == STALE:
            await self._reap(key)

    def _picker_play(self, key, user_id, round_id, moves, counts) -> _Out:
        loaded, refusal = self._load(key, round_id)
        if refusal is not None:
            return refusal
        _, snap = loaded
        seat = self.store.member_seat_index(key[0], round_id, user_id)
        if seat is None:
            return _Out(NOT_SEATED)
        if snap["status"] != "playing" or snap["moves"] != moves:
            return _Out(MOVED_ON)
        if snap["view"]["turn"] != seat:
            return _Out(NOT_YOUR_TURN)
        parts, held = [], doubt.held_counts(self.store.seat_view(key[0], round_id, seat).hand)
        if not isinstance(counts, dict) or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts.values()):
            return _Out(f"Pick 1 to {doubt.MAX_PLAY} cards to play.")
        for rank in doubt.RANKS:
            n = counts.get(rank, 0)
            if n > held.get(rank, 0):
                return _Out(f"You don't hold {_claim(n, rank)}.")
            if n:
                parts.append((rank, n))
        if any(rank not in doubt.RANKS for rank in counts):
            return _Out(f"Pick 1 to {doubt.MAX_PLAY} cards to play.")
        total = sum(n for _, n in parts)
        if not 1 <= total <= doubt.MAX_PLAY:
            return _Out(f"Pick 1 to {doubt.MAX_PLAY} cards to play.")
        move = "play:" + ",".join(f"{rank}x{n}" for rank, n in parts)
        try:
            out = self._apply(key, snap, seat, move, "member")
        except ConflictError:
            return _Out(MOVED_ON)
        except GameError as error:
            return _Out(str(error))
        out.say, out.ok = f"You play {doubt.cards_text(total)} and claim {_claim(total, snap['view']['rank'])}.", True
        return out

    # --- timers' work -----------------------------------------------------------------------

    async def on_lobby_timer(self, guild_id, table_id, round_id) -> None:
        key, out = (guild_id, table_id), None
        async with self.runtime.lock(key):
            row = self._row(key)
            if row is None or row["status"] != "open":
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
            out = self._seal(key, _Out(lobby=("Table closed: nobody started the game." + self._refunded(key), None), closed=True))
        await self._publish(key, out)

    async def on_window_timer(self, guild_id, table_id, round_id, moves) -> None:
        """The doubt window ran out: the claimer's claim stands."""
        key, out = (guild_id, table_id), None
        async with self.runtime.lock(key):
            snap = self._timer_snap(key, round_id, moves)
            if snap is None or snap["view"]["window"] is None:
                return
            try:
                out = self._apply(key, snap, snap["view"]["window"]["claimer"], "close", "timeout")
            except (ConflictError, GameError) as error:
                logging.info("I Doubt It window close refused (%s)", type(error).__name__)
                return
        await self._publish(key, out)

    def _missed_turns(self, guild_id, round_id, seat) -> int:
        """Timed-out plays in a row at the end of the seat's own plays (a play of its own resets it)."""
        rows = self.store.all("SELECT actor FROM game_moves WHERE guild_id=? AND round_id=? AND seat_index=? AND move LIKE 'play:%' ORDER BY id DESC", (guild_id, round_id, seat))
        count = 0
        for row in rows:
            if row["actor"] != "timeout":
                break
            count += 1
        return count

    async def on_turn_timer(self, guild_id, table_id, round_id, moves) -> None:
        """A member's turn ran out: the engine's timeout move; the second missed turn forfeits the seat."""
        key, out = (guild_id, table_id), None
        async with self.runtime.lock(key):
            snap = self._timer_snap(key, round_id, moves)
            if snap is None:
                return
            view, seat = snap["view"], snap["view"]["turn"]
            if seat is None or snap["seats"][seat]["kind"] != "member":
                return
            if view["window"]:
                return  # the turn clock only runs once the doubt window is closed
            try:
                if self._missed_turns(guild_id, round_id, seat) >= 1:  # the second timed-out play in a row
                    out = self._apply(key, snap, seat, "forfeit", "timeout")
                else:
                    move = doubt.timeout_move_for_view(self.store.seat_view(guild_id, round_id, seat), self.store.policy_seed(guild_id, round_id, f"{moves}/{seat}/timeout"))
                    out = self._apply(key, snap, seat, move, "timeout", note=f"{self._who(snap, seat)} ran out of time; a card was played for them.")
            except (ConflictError, GameError) as error:
                logging.info("I Doubt It turn timeout refused (%s)", type(error).__name__)
                return
        await self._publish(key, out)

    def _talk_budget(self, guild_id) -> int:
        """How many model calls the table-talk daily limit still allows (0 when character table talk is off). Failed attempts count like blackjack's."""
        if not self.store.game_character_talk(guild_id):
            return 0
        marker = (guild_id, self.store._server_day(guild_id, time.time())[1])
        if self.talk_failed.keys() - {marker}:
            self.talk_failed = {m: n for m, n in self.talk_failed.items() if m == marker}
        used = self.store.game_talk_calls_today(guild_id) + self.talk_failed.get(marker, 0)
        return max(self.store.game_talk_daily_limit(guild_id) - used, 0)

    def _count_failure(self, guild_id) -> None:
        marker = (guild_id, self.store._server_day(guild_id, time.time())[1])
        self.talk_failed[marker] = self.talk_failed.get(marker, 0) + 1

    def _note_failure(self, guild_id) -> None:
        """Route through the bot so a failure counts once toward the limit every table game shares."""
        note = getattr(self.bot, '_note_talk_failure', None)
        note(guild_id) if note else self._count_failure(guild_id)

    @staticmethod
    def _names(snap) -> list:
        """Display names by seat for a prompt: characters by name, members as Player 1, 2, ... in seat order (never member names or ids)."""
        names, players = [], 0
        for seat in snap["seats"]:
            if seat["kind"] == "character":
                names.append(seat["name"])
            else:
                players += 1
                names.append(f"Player {players}")
        return names

    def _apply_chain(self, key, snap, tried):
        """Under the table lock: the first move the store accepts (a refusal falls back to the next; the last resort frees the table).
        None when the state moved on."""
        for seat, move in tried:
            try:
                return self._apply(key, snap, seat, move, "character" if move != "forfeit" else "timeout")
            except ConflictError:
                return _STALE
            except GameError as error:
                logging.warning("I Doubt It character move refused (%s)", type(error).__name__)
        return None

    async def _apply_fresh(self, key, round_id, moves, tried):
        """Applies only if the table is still at the move count the decision was made for; else _STALE (dropped silently)."""
        async with self.runtime.lock(key):
            snap = self._timer_snap(key, round_id, moves)
            return _STALE if snap is None else self._apply_chain(key, snap, tried)

    def _character(self, guild_id, seat):
        character = self.store.character_by_id(seat["ref_id"])
        return character if character is not None and character["guild_id"] == guild_id else None

    async def on_character_timer(self, guild_id, table_id, round_id, moves) -> None:
        """Character seats first doubt an open claim, else the one on turn plays. With character table talk on and the daily
        limit not reached, those decisions are model calls on the seat's own view, made outside the table lock and applied
        only if the table is still at the same move count; any failure falls back to the deterministic policy (no clock)."""
        key, out = (guild_id, table_id), None
        async with self.runtime.lock(key):
            snap = self._timer_snap(key, round_id, moves)
            if snap is None:
                return
            view, seats = snap["view"], snap["seats"]
            window, turn = view["window"], view["turn"]
            budget = 0 if self.store.game_character_moves(guild_id) == "style" else self._talk_budget(guild_id)  # style only: the policy plays; table talk keeps its own budget
            if budget <= 0:
                out = self._policy_step(key, snap)
            elif window:
                plan = self._doubt_plan(key, snap)
            elif turn is not None and seats[turn]["kind"] == "character":
                plan = self._play_plan(key, snap)
            else:
                return
        if budget > 0:
            out = await (self._decide_doubts(key, snap, plan, budget) if window else self._decide_play(key, snap, plan))
        await self._publish(key, None if out is _STALE else out)

    def _tendencies(self, guild_id, character_id) -> tuple:
        """(bluff, suspicion) in 0..1: the saved play style's bluffing and suspicion in either Character moves mode, else the id-derived pair when none is saved."""
        style = self.store.character_play_style(guild_id, character_id)
        return doubt.tendencies(character_id) if style is None else (style.bluffing / 100, style.suspicion / 100)

    def _fallback_moves(self, guild_id, snap, seat, seen) -> list:
        """The seat's policy play, then the timeout play, then forfeit: the order the store is asked to accept them in."""
        seed = lambda kind: self.store.policy_seed(guild_id, snap["round_id"], f"{snap['moves']}/{seat}/{kind}")
        return [(seat, doubt.policy_play(seen, self._tendencies(guild_id, snap["seats"][seat]["ref_id"]), seed("play"))),
                (seat, doubt.timeout_move_for_view(seen, seed("fallback"))), (seat, "forfeit")]

    def _policy_step(self, key, snap):
        """The fallback policy's step. Call under the table lock."""
        guild_id, round_id, moves = key[0], snap["round_id"], snap["moves"]
        view, seats = snap["view"], snap["seats"]
        window, turn, n = view["window"], view["turn"], len(seats)
        seed = lambda seat, kind: self.store.policy_seed(guild_id, round_id, f"{moves}/{seat}/{kind}")
        tried = []
        if window:
            for k in range(1, n):
                i = (window["claimer"] + k) % n
                if seats[i]["kind"] == "character" and view["active"][i]:
                    seen = self.store.seat_view(guild_id, round_id, i)
                    if doubt.policy_doubt(seen, self._tendencies(guild_id, seats[i]["ref_id"]), seed(i, "doubt")):
                        tried = [(i, "doubt")]
                        break
        playable = turn is not None and seats[turn]["kind"] == "character" and not window
        if not tried and playable:
            seen = self.store.seat_view(guild_id, round_id, turn)
            tried = self._fallback_moves(guild_id, snap, turn, seen)
        return self._apply_chain(key, snap, tried)

    def _play_plan(self, key, snap):
        """Under the lock: what the play decision needs, taken before the model call (the seat's own view only)."""
        turn = snap["view"]["turn"]
        return turn, self._character(key[0], snap["seats"][turn]), self.store.seat_view(key[0], snap["round_id"], turn), self._names(snap)

    def _doubt_plan(self, key, snap):
        """Under the lock: (seats that see an impossible claim, [(seat, character, own view, tendencies)] triage picked, names)."""
        guild_id, view, seats = key[0], snap["view"], snap["seats"]
        claimer, n, scored, seen_by = view["window"]["claimer"], len(seats), [], {}
        for k in range(1, n):
            i = (claimer + k) % n
            if seats[i]["kind"] == "character" and view["active"][i]:
                seen = seen_by[i] = self.store.seat_view(guild_id, snap["round_id"], i)
                scored.append((i, doubt.suspicion(seen, self._tendencies(guild_id, seats[i]["ref_id"])), doubt.impossible(seen)))
        sure = [i for i, _, imp in scored if imp]
        asked = [(i, self._character(guild_id, seats[i]), seen_by[i], self._tendencies(guild_id, seats[i]["ref_id"])) for i in doubt.triage(scored, k=2) if i not in sure]
        return sure, asked, self._names(snap)

    @staticmethod
    def _model_play(result, seen):
        """The canonical play move for a valid model answer (every held rank, counts within what is held, 1 to 4 cards in all), else None."""
        held = doubt.held_counts(seen.hand)
        if not isinstance(result, dict) or set(result) != set(held):
            return None
        picks, total = [], 0
        for rank in doubt.RANKS:
            if rank in held:
                n = result[rank]
                if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= held[rank]:
                    return None
                total += n
                if n:
                    picks.append(f"{rank}x{n}")
        return "play:" + ",".join(picks) if 1 <= total <= doubt.MAX_PLAY else None

    async def _decide_play(self, key, snap, plan):
        guild_id, round_id, moves = key[0], snap["round_id"], snap["moves"]
        turn, character, seen, names = plan
        tried = []
        if character is not None:
            try:
                result = await asyncio.wait_for(game_chooser.ask_doubt_play(self.bot, guild_id, snap["channel_id"], character, seen, names, self.store.character_play_style(guild_id, character["id"])), self.MODEL_SECONDS)
                move = self._model_play(result, seen)
                if move is not None:
                    tried.append((turn, move))
            except asyncio.CancelledError:
                self._note_failure(guild_id)  # an attempt that was cut short still counts toward the limit
                raise
            except Exception as error:
                move = None
                logging.warning("I Doubt It character play fell back to the policy (%s)", type(error).__name__)
            if move is None:
                self._note_failure(guild_id)
        tried += self._fallback_moves(guild_id, snap, turn, seen)
        return await self._apply_fresh(key, round_id, moves, tried)

    async def _doubt_one(self, key, snap, seat, character, seen, tend, names, bound) -> tuple:
        """(seat, doubts?): the model's answer, else the policy's."""
        guild_id = key[0]
        try:
            return seat, await asyncio.wait_for(game_chooser.ask_doubt_doubt(self.bot, guild_id, snap["channel_id"], character, seen, names, self.store.character_play_style(guild_id, character["id"])), bound)
        except asyncio.CancelledError:
            self._note_failure(guild_id)  # a call dropped because another seat doubted first still counts toward the limit
            raise
        except Exception as error:
            logging.warning("I Doubt It character doubt fell back to the policy (%s)", type(error).__name__)
            self._note_failure(guild_id)
        return seat, doubt.policy_doubt(seen, tend, self.store.policy_seed(guild_id, snap["round_id"], f"{snap['moves']}/{seat}/doubt"))

    async def _decide_doubts(self, key, snap, plan, budget):
        sure, asked, names = plan
        round_id, moves = snap["round_id"], snap["moves"]
        for seat in sure:  # a claim that cannot be true from the seat's own hand: doubt at once, no model call
            out = await self._apply_fresh(key, round_id, moves, [(seat, "doubt")])
            if out is _STALE:
                return None
            if out is not None:
                return out
        bound = min(self.MODEL_SECONDS, self.WINDOW_SECONDS - 3)
        # Design: triaged seats without a character row, or beyond the remaining daily budget, simply pass (as do unselected seats).
        asked = [item for item in asked if item[1] is not None]
        tasks = {asyncio.create_task(self._doubt_one(key, snap, seat, character, seen, tend, names, bound)) for seat, character, seen, tend in asked[:budget]}
        try:
            while tasks:
                done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for seat, yes in sorted(task.result() for task in done):
                    if yes:  # the first yes wins; a later one finds the table moved on
                        return await self._apply_fresh(key, round_id, moves, [(seat, "doubt")])
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return None

    async def on_archive_timer(self, guild_id, table_id) -> None:
        self.ended.discard((guild_id, table_id))
        self.tables.pop((guild_id, table_id), None)
        self.banter = {thread: k for thread, k in self.banter.items() if k != (guild_id, table_id)}
        row = self._row((guild_id, table_id))
        if row is None or row["status"] == "open" or row["thread_id"] is None:
            return
        try:
            thread = await self._channel(row["thread_id"])
            await thread.edit(archived=True, locked=True)
        except Exception as error:
            logging.warning("Could not archive the I Doubt It thread (%s)", type(error).__name__)

    # --- table talk (FEAT-21 C) --------------------------------------------------------------

    def _talk_key(self, guild_id, thread_id):
        key = self.threads.get(thread_id) or self.banter.get(thread_id)
        return key if key is not None and key[0] == guild_id else None

    def _talk_names(self, guild_id, snap) -> list:
        """One prompt-safe name per seat, shared by the table block and the event texts: characters by name, members by their guild
        display name (Player N when unknown); no mentions, no ids, no markdown."""
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

    def talk_allowed(self, guild_id) -> bool:
        """Table talk is on and the daily limit (model calls plus failed move attempts) is not reached."""
        return self._talk_budget(guild_id) > 0

    def table_talk_context(self, guild_id, thread_id) -> dict | None:
        """The table whose thread this is (while it is joining or played, and during the banter after the game until it is archived), else None."""
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
        """What one seated character may know of the table: the public state and its own hand, never another seat's cards."""
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
        names, view = self._talk_names(key[0], snap), snap["view"]
        lines = ["You are sitting at a table of I Doubt It (a card game of bluffing: players claim the rank to play, add cards face down, and anyone may doubt)."]
        if not view:
            lines.append("The game has not started yet. Seated: " + ", ".join(names) + ".")
            return "\n".join(lines)

        def who(i):
            return names[i] + (" (you)" if i == mine["index"] else "")
        if view["finished"]:
            lines.append(f"The game is over. {who(view['winner'])} won.")
        else:
            lines.append(f"Rank to claim now: {view['rank']}. Pile: {doubt.cards_text(view['pile'])}.")
        for i, n in enumerate(view["counts"]):
            lines.append(f"- {who(i)}: {doubt.cards_text(n)}" + ("" if view["active"][i] else " (forfeited)") + (" (on turn)" if view["turn"] == i and not view["finished"] else ""))
        if view["window"]:
            w = view["window"]
            lines.append(f"Open claim: {names[w['claimer']]} claimed {_claim(w['count'], w['rank'])}; it can still be doubted.")
        recent = []
        for item in view["history"][-6:]:
            if item["kind"] == "reveal":
                cards = ", ".join(f"{r}{su}" for r, su in item["cards"])
                recent.append(f"{names[item['doubter']]} doubted {names[item['claimer']]}'s {_claim(len(item['cards']), item['rank'])}; they were {cards} ({'a bluff' if item['bluff'] else 'true'})")
            else:
                recent.append(f"{names[item['seat']]} claimed {_claim(item['count'], item['rank'])}")
        if recent:
            lines.append("Recent play: " + "; ".join(recent) + ".")
        if snap["status"] == "playing" and view["active"][mine["index"]]:
            try:
                hand = self.store.seat_view(key[0], snap["round_id"], mine["index"]).hand or ()
            except GameError:
                hand = ()
            lines.append("Keep your hand secret unless you choose to bluff about it.")
            lines.append("Your hand: " + (", ".join(f"{r}{su}" for r, su in hand) or "(no cards)"))
        return "\n".join(lines)

    # --- thread chat ------------------------------------------------------------------------

    async def note_thread_message(self, message) -> None:
        """Counts a member's message in a table thread; enough of them bury the turn message, which then moves to the bottom."""
        if message.author.bot or not isinstance(message.channel, discord.Thread):
            return
        key = self.threads.get(message.channel.id)
        if key is not None and message.guild is not None and message.guild.id == key[0]:
            self.table(key).talk += 1

    # --- restart ----------------------------------------------------------------------------

    async def recover(self) -> None:
        """Refunds and closes every I Doubt It table left open by a previous run, then tells the lobby and the thread (best effort)."""
        store, refunded, closed, failed = self.store, 0, 0, 0
        rounds = [r for r in store.unfinished_rounds() if r.get("game") == "doubt"]
        tables = {(r["guild_id"], r["table_id"]): (r["channel_id"], r["message_id"], r["thread_id"]) for r in rounds}
        for row in store.all("SELECT id,guild_id,channel_id,message_id,thread_id FROM game_tables WHERE status='open' AND game='doubt'"):
            tables.setdefault((row["guild_id"], row["id"]), (row["channel_id"], row["message_id"], row["thread_id"]))
        notes = {(r["guild_id"], r["table_id"]): self._refunded((r["guild_id"], r["table_id"]), past=True) for r in rounds}
        for r in rounds:
            try:
                store.cancel_round(r["guild_id"], r["round_id"], "bot restarted")
                refunded += 1
            except Exception:
                failed += 1
                logging.exception("Could not refund I Doubt It round %s on restart", r["round_id"])
        for (guild_id, table_id), (channel_id, message_id, thread_id) in tables.items():
            try:
                store.close_table(guild_id, table_id)
                closed += 1
            except Exception:
                failed += 1
                logging.exception("Could not close I Doubt It table %s on restart", table_id)
                continue
            self._release((guild_id, table_id))
            self.tables.pop((guild_id, table_id), None)
            self.banter = {thread: k for thread, k in self.banter.items() if k != (guild_id, table_id)}
            text = "This table closed when the bot restarted." + notes.get((guild_id, table_id), "")
            await self._edit(channel_id, message_id, content=text, view=None)
            await self._post(thread_id, text)
        if refunded or closed or failed:
            logging.info("I Doubt It recovery: %d rounds refunded, %d tables closed, %d failures", refunded, closed, failed)


def register_doubt_commands(bot, ctx: SimpleNamespace) -> None:
    require_guild = ctx.require_guild
    group = app_commands.Group(name="game", description="Open a table and play")
    choices = [app_commands.Choice(name="Blackjack", value="blackjack"), app_commands.Choice(name="I Doubt It", value="doubt"), app_commands.Choice(name="Texas Hold'em", value="holdem")]

    @group.command(name="start", description="Open a table in this channel")
    @app_commands.describe(game="Which game", bet="Blackjack only: how much to bet", favorites="Bring your favorite characters to the table")
    @app_commands.choices(game=choices)
    async def game_start(interaction: discord.Interaction, game: str, bet: app_commands.Range[int, 1, MAX_BET] | None = None, favorites: bool = False):
        require_guild(interaction)
        if game == "blackjack":
            if bet is None:
                await bot.doubt._say(interaction, BET_NEEDED)
                return
            await bot.games.play_command(interaction, bet, favorites)
            return
        if game == "holdem":
            if bet is not None:
                await bot.holdem._say(interaction, BET_NOTE)
                return
            await bot.holdem.start_command(interaction, favorites)
            return
        if bet is not None:
            await bot.doubt._say(interaction, ANTE_NOTE)
            return
        await bot.doubt.start_command(interaction, favorites)

    @group.command(name="hand", description="See your cards at this table (only you can see them)")
    async def game_hand(interaction: discord.Interaction):
        require_guild(interaction)
        table = bot.store.table_for_thread(interaction.guild_id, interaction.channel.id) if interaction.channel else None
        await (bot.holdem if table and table["game"] == "holdem" else bot.doubt).hand_command(interaction)

    @group.command(name="rules", description="How a game is played and the rules this server uses")
    @app_commands.describe(game="Which game")
    @app_commands.choices(game=choices)
    async def game_rules(interaction: discord.Interaction, game: str):
        require_guild(interaction)
        if game == "doubt":
            rules = bot.store.game_rules(interaction.guild_id, "doubt")
            text = doubt.how_to_play(rules) + "\n" + doubt.rules_text(rules)
        elif game == "holdem":
            text = bot.holdem.rules_text(interaction.guild_id)
        else:
            rules = bot.store.blackjack_rules(interaction.guild_id)
            text = bot.games.how_to_play(rules) + "\n" + blackjack.rules_text(rules)
        await bot.doubt._say(interaction, text)

    bot.tree.add_command(group)
