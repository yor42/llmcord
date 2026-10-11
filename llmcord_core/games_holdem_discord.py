"""Texas Hold'em on Discord (FEAT-33 part C1a): `/game start game:holdem`, lobby (Sit, Leave, Rebuy, Fill seats, Assist, Deal, Close table),
table thread, idle timer, restart recovery. The hand itself (deal, turn message, betting, showdown) is C1b.

Same shape as ``games_doubt_discord``: one lock per table that covers store work only; a handler decides under the lock and builds an
``_Out`` (a private reply, the lobby text and buttons, a sequence number); the messages go out afterwards, one table at a time, and an
edit older than the last one sent is dropped. Timers re-check the stored state under the lock, so a stale timer does nothing.
Money moves only inside the store (buy-in, rebuy, cash-out).
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

import discord

from .admin_store import GameError
from .games import holdem
from .games_discord import NO_CHARACTER_SEATS, NO_MENTIONS, NOT_SEATED, OFF_HERE, SKIPPED, STALE, GameTables
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
DEAL_LATER = "Dealing isn't available yet."


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
    """One number to type: the buy-in when sitting or the amount of a rebuy. Bound to the server, table and round it was opened for."""

    def __init__(self, kind: str, guild_id: int, table_id: int, round_id: int, title: str, label: str, placeholder: str, default: str = ""):
        super().__init__(title=title[:45], timeout=600)
        self.kind, self.guild_id, self.table_id, self.round_id = kind, guild_id, table_id, round_id
        self.field = discord.ui.TextInput(label=label[:45], placeholder=placeholder[:100], default=default or None, max_length=12, required=True)
        self.add_item(self.field)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.client.holdem.submit(interaction, self)

    async def on_error(self, interaction: discord.Interaction, error: Exception):
        logging.error("Texas Hold'em %s modal failed", self.kind, exc_info=error)
        await interaction.client.holdem._say(interaction, "Something went wrong; try again.")


@dataclass(eq=False)
class _Table:
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    issued: int = 0  # numbers the outgoing edits, assigned under the table lock
    sent: dict = field(default_factory=dict)  # newest delivered number per message kind
    deadlines: dict = field(default_factory=dict)  # lobby -> unix time


@dataclass
class _Out:
    """What a locked step decided: a private reply and the messages to send after the lock is released."""
    say: str | None = None
    key: tuple | None = None
    lobby: tuple | None = None  # (content, view or None)
    closing: str | None = None  # the table was closed from outside: this line goes to the thread
    closed: bool = False  # the table is over: the lobby message is repainted even though the table is closed
    seq: int = 0


class HoldemTables:
    LOBBY_SECONDS = 300
    ARCHIVE_SECONDS = 600
    BOARD_DEBOUNCE_SECONDS = 1
    RETRY_SECONDS = 5  # read once at construction (TableRuntime.retry_seconds); tests set runtime.retry_seconds
    SWEEP_SECONDS = 30

    def __init__(self, bot, clock=time.time):
        self.bot, self.clock = bot, clock
        self.runtime = TableRuntime(label=NAME, retry_seconds=self.RETRY_SECONDS)
        self.tables: dict[tuple[int, int], _Table] = {}
        self.threads: dict[int, tuple[int, int]] = {}  # thread id -> (guild, table) of the tables being played
        self.ended: set[tuple[int, int]] = set()  # tables this object closed itself: the sweeper leaves them alone

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
        hook = {"lobby": self.on_lobby_timer, "archive": self.on_archive_timer}[kind]
        self.runtime.arm(key, kind, delay, self._run, kind, hook, key, args, name=f"holdem-{kind}-{key[1]}")

    async def _run(self, kind, hook, key, args) -> None:
        await hook(*key, *args)
        if kind != "archive":
            await self._reap(key)  # the hook may have found the table closed from outside

    async def close(self) -> None:
        await self.runtime.close()

    def _release(self, key) -> None:
        """Stops the table's play timers and drops its thread. The archive timer is armed separately."""
        self.runtime.disarm(key, "lobby")
        self.threads = {thread: k for thread, k in self.threads.items() if k != key}

    def _forget(self, key) -> None:
        """This object closed the table (in the store): release it, and keep the sweeper off it until its archive timer."""
        self.ended.add(key)
        self._release(key)

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
        """After the table lock is released: the edits (an edit older than the last one sent is dropped)."""
        if out is None:
            return
        table = self.table(key)
        async with table.send_lock:
            row = self._row(key)
            if row is None:
                return
            if row["status"] != "open" and not out.closed:
                return  # a step sealed before the table was closed must not repaint the closed table
            if out.closing:
                await self._post(row["thread_id"], out.closing)
            if out.lobby and out.seq > table.sent.get("lobby", 0):
                table.sent["lobby"] = out.seq
                await self._edit(row["channel_id"], row["message_id"], content=out.lobby[0], view=out.lobby[1])

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
            await self._say(interaction, "The buy-in must be a whole number." if modal.kind == "sit" else "The rebuy must be a whole number above zero.")
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
        return self._seal(key, _Out(say, lobby=self.render_lobby(key, snap, row)))

    def _leave(self, key, row, snap, user_id) -> _Out:
        mine = next((s for s in snap["seats"] if s["kind"] == "member" and s["ref_id"] == user_id), None)
        snap = self.store.leave_round(key[0], snap["round_id"], user_id)
        self._touch_lobby(key, snap["round_id"])  # an empty table stays open; the idle timer closes it
        name = self.store.currency_name(key[0])
        return self._seal(key, _Out(f"You left the table and cashed out {mine['stack']:,} {name}.", lobby=self.render_lobby(key, snap, row)))

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
        return self._seal(key, _Out(" ".join(parts), lobby=self.render_lobby(key, snap, row)))

    def _assist(self, key, row, snap, user_id, on) -> _Out:
        snap = self.store.set_table_assist(key[0], snap["round_id"], user_id, on)
        return self._seal(key, _Out(f"Assist is {'on' if on else 'off'}.", lobby=self.render_lobby(key, snap, row)))

    def _deal(self, key, row, snap, user_id) -> _Out:
        if user_id != row["opened_by"]:
            return _Out("Only the host can deal.")
        if row["thread_id"] is None:
            return _Out("The table thread isn't ready yet; try again in a moment.")
        if len(snap["seats"]) < holdem.MIN_SEATS:
            return _Out(f"{NAME} needs at least {holdem.MIN_SEATS} seats to deal.")
        return _Out(DEAL_LATER)  # C1b: deal, post the board and the turn message

    def _close(self, key, row, snap, user_id) -> _Out:
        if user_id != row["opened_by"]:
            return _Out("Only the host can close the table.")
        self.store.close_table(*key)
        self._forget(key)
        self.arm(key, "archive", self.ARCHIVE_SECONDS)
        return self._seal(key, _Out("Table closed." + CASHED, lobby=("Table closed by the host." + CASHED, None), closed=True))

    # --- /game hand and /game rules ---------------------------------------------------------

    async def hand_command(self, interaction: discord.Interaction) -> None:
        guild_id = interaction.guild_id
        table = self.store.table_for_thread(guild_id, interaction.channel.id) if guild_id and interaction.channel else None
        if table is None or table["game"] != "holdem":
            await self._say(interaction, NO_TABLE)
            return
        try:
            snap = self.store.latest_round(guild_id, table["id"])
        except GameError:
            await self._say(interaction, STALE)
            return
        if snap["status"] == "joining":
            await self._say(interaction, "The hand hasn't started yet.")
        elif self.store.member_seat_index(guild_id, snap["round_id"], interaction.user.id) is None:
            await self._say(interaction, NOT_SEATED)
        else:
            await self._say(interaction, DEAL_LATER)  # C1b: My cards

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
            out = self._seal(key, _Out(lobby=("Table closed: nobody started the game." + CASHED, None), closed=True))
        await self._publish(key, out)

    async def on_archive_timer(self, guild_id, table_id) -> None:
        self.ended.discard((guild_id, table_id))
        self.tables.pop((guild_id, table_id), None)
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
            self.tables.pop((guild_id, table_id), None)
            text = "This table closed when the bot restarted." + CASHED
            await self._edit(t["channel_id"], t["message_id"], content=text, view=None)
            if t["thread_id"] is not None:
                await self._post(t["thread_id"], text)
                await self._archive(t["thread_id"])
        if closed or failed:
            logging.info("Texas Hold'em recovery: %d tables closed, %d failures", closed, failed)
