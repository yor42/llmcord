"""Blackjack on Discord (FEAT-19 part C): commands, buttons, timers and restart recovery.

Rules, money and persistence live in ``games`` and the store (part A and B). This module only turns
presses into store calls and store snapshots into messages. Every table runs under its own lock (never
the channel skit lock); timers re-check the stored state under that lock, so a stale timer does nothing.

No Discord HTTP happens while a table lock is held (MNT-39): a handler does its store work under the lock
and builds a ``Paint`` (the new message text and buttons, numbered by a per-table sequence); the paint and
any reply are sent after the lock is released. Button presses are acknowledged (``defer()``) before the lock
is taken. Paints of one table are sent one at a time, and a paint older than the last one sent is dropped,
so a slow older edit can never overwrite a newer one.
"""
from __future__ import annotations

import asyncio
import logging
import time
import weakref
from dataclasses import dataclass, field
from types import SimpleNamespace

import discord
from discord import app_commands

from .admin_store import ConflictError, GameError
from . import game_chooser
from .games import DefaultPolicy, blackjack

STALE = "That table has moved on — use the latest buttons."
OFF_HERE = "Games are off in this channel."
BUSY = "A round is being played; join the next one with /blackjack when it ends."
LEAVE_BUSY = "You can leave when this round ends."
NOT_SEATED = "You're not seated at this table."
NOT_YOUR_TURN = "It's not your turn."
NO_CHARACTER_SEATS = "Characters can't sit at tables in this channel."
SKIPPED = {"seated": "already at the table", "full": "table full", "archived": "archived"}
CLOSED_OFF = "Table closed: games were turned off here."
CLOSED = "Table closed."
MAX_BET = 100_000
PREFIX = "llmcord:bj"
NO_MENTIONS = discord.AllowedMentions.none()


@dataclass(eq=False)
class _Sender:
    """Orders the message edits of one table: ``issued`` numbers paints under the table lock, ``sent`` is the newest delivered."""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    issued: int = 0
    sent: int = 0


@dataclass
class Paint:
    """A table message computed under the table lock and sent after it."""
    key: tuple
    sender: _Sender
    seq: int
    content: str
    view: discord.ui.View | None = None
    closing: bool = False


@dataclass
class _Out:
    """What a locked step decided: an ephemeral reply, table repaints, and whether the table message still has to be posted."""
    say: str | None = None
    paints: list = field(default_factory=list)
    opened: bool = False
    joined: bool = False
    note: str = ""  # the favorites part of the join reply


@dataclass
class _Table:
    bets: dict = field(default_factory=dict)  # member id -> the bet they joined with
    present: set = field(default_factory=set)  # members still at the table between rounds
    deadline: float | None = None
    join_round: int | None = None  # round whose join timer is armed
    timeouts: set = field(default_factory=set)  # (round id, seat index)
    sender: _Sender | None = None


class GameButton(discord.ui.DynamicItem[discord.ui.Button], template=PREFIX + r":(?P<action>[a-z]+):(?P<table>\d+):(?P<round>\d+):(?P<moves>\d+)"):
    """One button; its custom_id carries the table, the round and the move count the message showed."""

    def __init__(self, action: str, table_id: int, round_id: int, moves: int, label: str = "", style=discord.ButtonStyle.secondary):
        super().__init__(discord.ui.Button(label=label, style=style, custom_id=f"{PREFIX}:{action}:{table_id}:{round_id}:{moves}"))
        self.action, self.table_id, self.round_id, self.moves = action, table_id, round_id, moves

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["action"], int(match["table"]), int(match["round"]), int(match["moves"]), item.label or "", item.style)

    async def callback(self, interaction: discord.Interaction):
        await interaction.client.games.press(interaction, self.action, self.table_id, self.round_id, self.moves)


def _sign(net: int) -> str:
    return f"+{net:,}" if net > 0 else f"−{-net:,}" if net < 0 else ""


def _outcome(seat: dict) -> str:
    outcome, net = seat["outcome"], (seat["payout"] or 0) - seat["stake"]
    if outcome == "push":
        return "push"
    if outcome == "refund":
        return "refunded"
    return f"{outcome} {_sign(net)}".strip()


class GameTables:
    JOIN_SECONDS = 30
    TURN_SECONDS = 60
    IDLE_SECONDS = 120
    CHARACTER_SECONDS = 2
    MODEL_SECONDS = 20
    POST_SECONDS = 10
    RETRY_SECONDS = 5
    SWEEP_SECONDS = 30

    def __init__(self, bot, clock=time.time):
        self.bot, self.clock = bot, clock
        self.locks: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self.senders: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self.timers: dict[tuple[int, int], asyncio.Task] = {}
        self.tables: dict[tuple[int, int], _Table] = {}
        self.sweeper: asyncio.Task | None = None
        self.closed = False

    @property
    def store(self):
        return self.bot.store

    def lock(self, key) -> asyncio.Lock:
        lock = self.locks.get(key)
        if lock is None:
            lock = self.locks[key] = asyncio.Lock()
        return lock

    def sender(self, key) -> _Sender:
        sender = self.senders.get(key)
        if sender is None:
            sender = self.senders[key] = _Sender()
        return sender

    def table(self, key) -> _Table:
        table = self.tables.get(key)
        if table is None:
            table = self.tables[key] = _Table(sender=self.sender(key))
            self._ensure_sweeper()
        return table

    # --- timers -----------------------------------------------------------------------------

    def arm(self, key, kind: str, round_id: int, moves: int = 0) -> None:
        if self.closed:
            return
        delay = {"join": self.JOIN_SECONDS, "turn": self.TURN_SECONDS, "idle": self.IDLE_SECONDS, "character": self.CHARACTER_SECONDS}[kind]
        self.disarm(key)
        self.table(key).deadline = self.clock() + delay
        task = self.timers[key] = asyncio.create_task(self._timer(key, kind, round_id, moves, delay), name=f"blackjack-{kind}-{key[1]}")
        task.add_done_callback(lambda done: self.timers.get(key) is done and self.timers.pop(key, None))

    def disarm(self, key) -> None:
        old = self.timers.pop(key, None)
        if old is not None and old is not asyncio.current_task():
            old.cancel()

    def _current(self, key) -> bool:
        return not self.closed and self.timers.get(key) is asyncio.current_task()

    async def _timer(self, key, kind, round_id, moves, delay) -> None:
        await asyncio.sleep(delay)
        for attempt in (1, 2):  # one retry, so held bets do not wait for a restart after a transient failure
            try:
                if kind == "join":
                    await self.on_join_timer(*key, round_id)
                elif kind == "turn":
                    await self.on_turn_timer(*key, round_id, moves)
                elif kind == "character":
                    await self.on_character_timer(*key, round_id, moves)
                else:
                    await self.on_idle_timer(*key, round_id)
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("Blackjack %s timer failed (table %s, attempt %d)", kind, key[1], attempt)
                if attempt == 1:
                    if not self._current(key):  # replaced, disarmed or closed: the retry would be untracked
                        return
                    await asyncio.sleep(self.RETRY_SECONDS)  # still the table's timer, so disarm, _forget and close() cancel it
                    if not self._current(key):
                        return

    async def close(self) -> None:
        self.closed = True
        tasks = list(self.timers.values())
        self.timers.clear()
        if self.sweeper is not None:
            tasks.append(self.sweeper)
            self.sweeper = None
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _after(self, key, snap) -> None:
        """Arms the timer the new state needs. Call under the table lock."""
        table, status = self.table(key), snap["status"]
        if status == "joining":
            if snap["seats"] and table.join_round != snap["round_id"]:
                table.join_round = snap["round_id"]
                self.arm(key, "join", snap["round_id"])
        elif status == "playing":
            turn = snap["view"]["turn"] if snap["view"] else None
            character = turn is not None and snap["seats"][turn]["kind"] == "character"
            self.arm(key, "character" if character else "turn", snap["round_id"], snap["moves"])
        elif status == "settled":
            table.present = {seat["ref_id"] for seat in snap["seats"] if seat["kind"] == "member"}
            self.arm(key, "idle", snap["round_id"])
        else:
            self.disarm(key)

    def _forget(self, key) -> None:
        self.disarm(key)
        self.tables.pop(key, None)

    # --- dashboard closes (no IPC: the dashboard closes tables in the shared database) -------

    def _ensure_sweeper(self) -> None:
        if self.closed or not self.tables or (self.sweeper is not None and not self.sweeper.done()):
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self.sweeper = asyncio.create_task(self._sweep_loop(), name="blackjack-sweep")

    async def _sweep_loop(self) -> None:
        while True:
            await asyncio.sleep(self.SWEEP_SECONDS)
            if not self.tables:  # nothing in memory: stay idle until a table appears
                self.sweeper = None
                return
            try:
                await self.sweep()
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("Blackjack table sweep failed")

    async def sweep(self) -> None:
        """Cleans up tables the dashboard closed in the store while the bot still holds them in memory."""
        for key in list(self.tables):
            if self._is_open(key):
                continue
            async with self.lock(key):
                _, paint = self._closed(key)
            if paint is not None:
                await self._deliver(paint)

    def _table_row(self, key):
        return self.store.one("SELECT status,channel_id,message_id FROM game_tables WHERE id=? AND guild_id=?", (key[1], key[0]))

    def _is_open(self, key) -> bool:
        row = self._table_row(key)
        return row is not None and row["status"] == "open"

    def _closed(self, key, repaint=False) -> tuple[bool, Paint | None]:
        """Under the table lock: is the table closed in the store? If the bot still holds it, forget it and return the paint that closes its message.
        ``repaint`` (a button press) also returns a one-off closing paint for a table the bot no longer holds, e.g. one closed while it was down."""
        if self._is_open(key):
            return False, None
        if key not in self.tables:
            if not repaint:
                return True, None
            row = self._table_row(key)
            if row is None:
                return True, None
            return True, self._closing(key, CLOSED if row["channel_id"] in self.store.game_channels(key[0]) else CLOSED_OFF)
        paint = self._closing(key, CLOSED_OFF)
        self._forget(key)
        return True, paint

    def _if_closed(self, key, text: str | None = None, repaint=False) -> _Out | None:
        """A refusal for a closed table. Without ``text``: "Games are off here" if the channel no longer allows games, else "Table closed."."""
        closed, paint = self._closed(key, repaint)
        if not closed:
            return None
        if text is None:
            row = self._table_row(key)
            text = STALE if row is None else CLOSED if row["channel_id"] in self.store.game_channels(key[0]) else OFF_HERE
        return _Out(text, [paint] if paint else [])

    # --- painting ---------------------------------------------------------------------------

    def _paint(self, key, snap) -> Paint:
        """Call under the table lock: the content is computed from the state the lock protects and numbered."""
        sender = self.table(key).sender
        sender.issued += 1
        return Paint(key, sender, sender.issued, self.render(snap), self.buttons(snap))

    def _closing(self, key, text: str) -> Paint:
        sender = self.sender(key)
        sender.issued += 1
        return Paint(key, sender, sender.issued, text, None, True)

    async def _deliver(self, paint: Paint, via=None) -> None:
        """Sends a paint (one at a time per table). A paint older than the last one sent is dropped; a table closed in the store takes only its closing paint."""
        sender = paint.sender
        async with sender.lock:
            if paint.seq <= sender.sent:
                return
            row = self._table_row(paint.key)
            if row is None or (row["status"] != "open" and not paint.closing):
                return
            sender.sent = paint.seq
            if via is not None:
                try:
                    await via.edit_original_response(content=paint.content, view=paint.view, allowed_mentions=NO_MENTIONS)
                    return
                except discord.HTTPException as error:
                    logging.warning("Blackjack press edit failed (%s); editing the table message", type(error).__name__)
                except Exception:
                    logging.exception("Blackjack press edit failed; editing the table message")
            await self.edit(row["channel_id"], row["message_id"], paint.content, paint.view)

    # --- rendering --------------------------------------------------------------------------

    @staticmethod
    def _who(seat) -> str:
        if seat["kind"] != "character":
            return f"<@{seat['ref_id']}>"
        who = f"**{discord.utils.escape_markdown(discord.utils.escape_mentions(seat['name']))}**"
        return who + (f" (with <@{seat['brought_by']}>)" if seat["brought_by"] else "")

    def render(self, snap) -> str:
        key = (snap["guild_id"], snap["table_id"])
        table, store, guild_id = self.table(key), self.store, snap["guild_id"]
        name, limits = store.currency_name(guild_id), store.game_settings(guild_id)
        status, view, deadline = snap["status"], snap["view"], table.deadline
        stamp = f"<t:{int(deadline)}:R>" if deadline else ""
        lines = ["**Blackjack**"]
        for seat in snap["seats"]:
            i = seat["index"]
            line = f"{i + 1}. {self._who(seat)} bet {(view['stakes'][i] if view else seat['stake']):,} {name}"
            if view:
                cards = view["hands"][i]
                line += f" — {blackjack.hand_str(cards)} ({blackjack.total_str(cards)})"
                if status == "playing":
                    word = {"stand": "stands", "bust": "bust", "blackjack": "blackjack"}.get(view["status"][i])
                    line += f" {word}" if word else ""
                    line += " (timed out)" if (snap["round_id"], i) in table.timeouts else ""
            if status == "settled":
                line += f" — {_outcome(seat)}"
                line += " (timed out)" if (snap["round_id"], i) in table.timeouts else ""
            lines.append(line)
        if view:
            dealer = view["dealer"]
            if view["dealer_hidden"]:
                lines.append(f"Dealer: {blackjack.hand_str(dealer)} ??")
            else:
                text = f"Dealer: {blackjack.hand_str(dealer)} ({blackjack.total_str(dealer)})"
                lines.append(text + (" bust" if blackjack.hand_total(dealer)[0] > 21 else ""))
        if status == "joining":
            lines.append(f"Join with /blackjack <bet> — dealing {stamp}." if stamp else "Join with /blackjack <bet>.")
            lines.append(f"Bets: {limits['min_bet']:,} to {limits['max_bet']:,} {name}.")
        elif status == "playing" and view and view["turn"] is not None:
            on_turn = snap["seats"][view["turn"]]
            if on_turn["kind"] == "character":
                lines.append(f"On turn: {self._who({**on_turn, 'brought_by': None})}.")
            else:
                lines.append(f"On turn: <@{on_turn['ref_id']}>" + (f" — decide {stamp}." if stamp else "."))
        elif status == "settled":
            lines.append(f"Play again or leave. The table closes {stamp} if nobody plays again." if stamp else "Play again or leave.")
        if status == "settled" and snap["seed"]:
            lines.append(f"-# Seed: {snap['seed']} (hash {snap['seed_hash'][:12]})")
        else:
            lines.append(f"-# Seed hash: {snap['seed_hash'][:12]}")
        return "\n".join(lines)

    def buttons(self, snap) -> discord.ui.View | None:
        status, table_id, round_id, moves = snap["status"], snap["table_id"], snap["round_id"], snap["moves"]
        spec = []
        if status == "joining":
            spec = [("deal", "Deal now", discord.ButtonStyle.primary), ("leave", "Leave", discord.ButtonStyle.secondary)]
        elif status == "playing":
            legal = next(iter(snap["legal"].values()), ())
            turn = snap["view"]["turn"] if snap["view"] else None
            if turn is not None and snap["seats"][turn]["kind"] == "character":
                legal = ()  # members cannot press for a character
            spec = [(m, m.capitalize(), discord.ButtonStyle.primary if m == "hit" else discord.ButtonStyle.secondary) for m in ("hit", "stand", "double") if m in legal]
        elif status == "settled":
            spec = [("again", "Play again (same bet)", discord.ButtonStyle.success), ("close", "Leave table", discord.ButtonStyle.secondary)]
        if not spec:
            return None
        view = discord.ui.View(timeout=None)
        for action, label, style in spec:
            view.add_item(GameButton(action, table_id, round_id, moves, label, style))
        return view

    async def _partial(self, channel_id, message_id):
        channel = self.bot.get_channel(channel_id) or await self.bot.fetch_channel(channel_id)
        return channel.get_partial_message(message_id)

    async def edit(self, channel_id, message_id, content, view=None) -> None:
        """Best-effort edit of the table message; a deleted message or a Discord error is logged, not raised."""
        if message_id is None:
            return
        try:
            partial = await self._partial(channel_id, message_id)
            await partial.edit(content=content, view=view, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning("Could not edit blackjack message %s in channel %s: %s", message_id, channel_id, type(error).__name__)
        except Exception:
            logging.exception("Could not edit blackjack message %s in channel %s", message_id, channel_id)

    async def _say(self, interaction, text: str) -> None:
        """An ephemeral reply: the first response, or a followup once the interaction was acknowledged."""
        try:
            if interaction.response.is_done():
                await interaction.followup.send(text, ephemeral=True, allowed_mentions=NO_MENTIONS)
            else:
                await interaction.response.send_message(text, ephemeral=True, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning("Blackjack reply failed: %s", type(error).__name__)

    async def _ack(self, interaction) -> None:
        try:
            await interaction.response.defer()  # a component defer updates the message in place, no "thinking" state
        except discord.HTTPException as error:
            logging.warning("Blackjack could not acknowledge the press: %s", type(error).__name__)

    async def _finish(self, interaction, out: _Out, via=None) -> None:
        """After the table lock is released. An unacknowledged interaction (/blackjack) is answered first, so a slow edit
        cannot make it miss Discord's 3 seconds; an acknowledged one (a press) repaints first, then replies."""
        say, first = out.say, not interaction.response.is_done()
        if first and say:
            if out.joined and not self._is_open(out.paints[0].key):
                say = OFF_HERE  # the table was closed (and the bet refunded) after the join
            await self._say(interaction, say)
            say = None
        for paint in out.paints:
            await self._deliver(paint, via)
        if say:
            await self._say(interaction, say)

    async def send_paint(self, paint: Paint) -> None:
        await self._deliver(paint)

    # --- table life -------------------------------------------------------------------------

    def _close_in_store(self, key) -> None:
        self.store.close_table(*key)
        self._forget(key)

    async def set_channel(self, guild_id, channel_id, enabled: bool) -> bool:
        """Turns games on or off in a channel. Returns True when an open table was closed (and refunded)."""
        closed, paint = await self.switch_channel(guild_id, channel_id, enabled)
        if paint is not None:
            await self._deliver(paint)
        return closed

    async def switch_channel(self, guild_id, channel_id, enabled: bool) -> tuple[bool, Paint | None]:
        """Like ``set_channel`` but returns the closing paint unsent (``send_paint``), so the caller can reply first."""
        if enabled:
            self.store.set_game_channel(guild_id, channel_id, True)
            return False, None
        paint = closed = None
        for attempt in range(3):
            table = self.store.open_table_for(guild_id, channel_id)
            if table is None:
                closed = self.store.set_game_channel(guild_id, channel_id, False)["closed_table"]
                break
            async with self.lock((guild_id, table["id"])):
                current = self.store.open_table_for(guild_id, channel_id)
                if current is not None and current["id"] != table["id"] and attempt < 2:
                    continue  # a newer table opened meanwhile: release this lock and close it under its own
                closed = self.store.set_game_channel(guild_id, channel_id, False)["closed_table"]
                if closed is not None:
                    paint = self._closing((guild_id, closed), CLOSED_OFF)
                    self._forget((guild_id, closed))
            break
        return closed is not None, paint

    # --- /blackjack -------------------------------------------------------------------------

    async def play_command(self, interaction: discord.Interaction, bet: int, favorites: bool = False) -> None:
        # Not deferred: the first opener's table message is the public response, which an ephemeral defer would
        # make impossible. The table lock is held for store work only (no HTTP), and `_finish` answers before any
        # table edit goes out, so the response does not wait for the edit queue or the channel HTTP.
        guild_id = interaction.guild_id
        channel_id = getattr(interaction, "channel_id", None) or interaction.channel.id
        if interaction.user.bot:
            await self._say(interaction, "Bots cannot play blackjack.")
            return
        if channel_id not in self.store.game_channels(guild_id):
            await self._say(interaction, OFF_HERE)
            return
        space_id, note = None, ""
        if favorites:
            _, binding = self.bot.location(interaction.channel)
            if binding:
                space_id = binding["space_id"]
            else:
                note = NO_CHARACTER_SEATS
        opened = None
        table = self.store.open_table_for(guild_id, channel_id)
        if table is None:
            try:
                opened = self.store.open_table(guild_id, channel_id, "blackjack", interaction.user.id)
            except GameError as error:
                await self._say(interaction, str(error))
                return
            table = {"id": opened["table_id"]}
        key = (guild_id, table["id"])
        async with self.lock(key):
            out = self._join(interaction.user.id, key, bet, opened, space_id=space_id, note=note)
        if out.opened:
            await self._open_message(interaction, key, out.paints[0])
            if out.note:
                await self._say(interaction, out.note)
        else:
            await self._finish(interaction, out)

    def _favorites_note(self, result, note) -> str:
        name, parts = self.store.currency_name(result["snapshot"]["guild_id"]), []
        if result["seated"]:
            parts.append("Your favorites joined too: " + ", ".join(f"{s['name']} ({s['stake']:,})" for s in result["seated"]) + ".")
        if result["skipped"]:
            why = lambda reason: f"not enough {name}" if reason == "broke" else SKIPPED[reason]
            parts.append("Not seated: " + ", ".join(f"{s['name']} ({why(s['reason'])})" for s in result["skipped"]) + ".")
        return " ".join([*parts, note] if note else parts)

    def _join(self, user_id, key, bet, opened, space_id=None, note="") -> _Out:
        guild_id = key[0]
        if opened is None and (out := self._if_closed(key, OFF_HERE)) is not None:
            return out
        try:
            snap = self.store.latest_round(*key)
            if snap["status"] == "playing":
                return _Out(BUSY)
            fresh = snap["status"] != "joining"
            if fresh:
                snap = self.store.next_round(*key)
            try:
                if space_id is not None:
                    result = self.store.join_with_favorites(guild_id, snap["round_id"], user_id, bet, space_id)
                    snap, note = result["snapshot"], self._favorites_note(result, note)
                else:
                    snap = self.store.join_round(guild_id, snap["round_id"], user_id, bet)
            except GameError:
                if fresh and opened is None:
                    self.store.cancel_round(guild_id, snap["round_id"], "nobody joined")
                raise
        except GameError as error:
            if opened is not None:
                self._close_in_store(key)
            return _Out(str(error))
        self.table(key).bets[user_id] = bet
        self.table(key).present.add(user_id)
        self._after(key, snap)
        paint = self._paint(key, snap)
        if opened is not None:
            return _Out(paints=[paint], opened=True, note=note)
        return _Out(f"You joined the table with {bet:,} {self.store.currency_name(guild_id)}." + (f" {note}" if note else ""), [paint], joined=True)

    async def _open_message(self, interaction, key, paint: Paint) -> None:
        """Posts the table message (the public response). Edits of this table wait behind it, so none is lost to a missing message id."""
        async with paint.sender.lock:
            try:
                await interaction.response.send_message(paint.content, view=paint.view, allowed_mentions=NO_MENTIONS)
                message = await interaction.original_response()
                self.store.set_table_message(*key, message.id)
            except Exception:
                async with self.lock(key):
                    self._close_in_store(key)  # refunds; no table without a message
                raise
            paint.sender.sent = max(paint.sender.sent, paint.seq)
        fresh = None
        async with self.lock(key):
            if paint.sender.issued > paint.seq and key in self.tables and self._is_open(key):
                try:
                    fresh = self._paint(key, self.store.latest_round(*key))  # someone joined before the message existed
                except GameError:
                    pass
        if fresh is not None:
            await self._deliver(fresh)

    # --- buttons ----------------------------------------------------------------------------

    async def press(self, interaction: discord.Interaction, action: str, table_id: int, round_id: int, moves: int) -> None:
        guild_id = interaction.guild_id
        if guild_id is None or interaction.user.bot:
            await self._say(interaction, STALE)
            return
        await self._ack(interaction)
        key = (guild_id, table_id)
        async with self.lock(key):
            out = self._press(interaction.user.id, key, action, round_id, moves)
        await self._finish(interaction, out, via=interaction)

    def _press(self, user_id, key, action, round_id, moves) -> _Out:
        """Store work for one button press, under the table lock. Returns what to send afterwards."""
        guild_id = key[0]
        if (out := self._if_closed(key, repaint=True)) is not None:
            return out
        try:
            snap = self.store.round_snapshot(guild_id, round_id)
        except GameError:
            return _Out(STALE)
        if snap["table_id"] != key[1]:
            return _Out(STALE)
        seat = next((s["index"] for s in snap["seats"] if s["kind"] == "member" and s["ref_id"] == user_id), None)
        if seat is None:
            return _Out(NOT_SEATED)
        try:
            if action in ("hit", "stand", "double"):
                return self._move(key, snap, seat, action, moves)
            if action == "deal":
                return self._deal(key, snap)
            if action == "leave":
                return self._leave(key, snap, user_id)
            if action == "again":
                return self._again(key, user_id)
            if action == "close":
                return self._leave_table(key, user_id)
            return _Out(STALE)
        except ConflictError:
            return _Out(STALE)
        except GameError as error:
            return _Out(str(error))

    def _move(self, key, snap, seat, move, moves) -> _Out:
        if snap["status"] != "playing" or snap["moves"] != moves:
            return _Out(STALE)
        if seat not in snap["legal"]:
            return _Out(NOT_YOUR_TURN)
        snap = self.store.play(key[0], snap["round_id"], seat, move, "member", moves)
        self._after(key, snap)
        return _Out(paints=[self._paint(key, snap)])

    def _deal(self, key, snap) -> _Out:
        snap = self.store.deal(key[0], snap["round_id"])
        self._after(key, snap)
        return _Out(paints=[self._paint(key, snap)])

    def _leave(self, key, snap, user_id) -> _Out:
        snap = self.store.leave_round(key[0], snap["round_id"], user_id)
        self.table(key).bets.pop(user_id, None)
        if not snap["seats"]:
            return self._close_empty(key)
        return _Out(paints=[self._paint(key, snap)])

    def _close_empty(self, key) -> _Out:
        self._close_in_store(key)
        return _Out(paints=[self._closing(key, CLOSED)])

    def _again(self, key, user_id) -> _Out:
        guild_id, table = key[0], self.table(key)
        snap = self.store.latest_round(*key)
        if snap["status"] == "playing":
            return _Out(BUSY)
        bet = table.bets.get(user_id)
        if bet is None:
            return _Out(STALE)
        fresh = snap["status"] != "joining"
        if fresh:
            snap = self.store.next_round(*key)
        try:
            snap = self.store.join_round(guild_id, snap["round_id"], user_id, bet)
        except GameError:
            if fresh:
                self.store.cancel_round(guild_id, snap["round_id"], "nobody joined")
            raise
        table.present.add(user_id)
        self._after(key, snap)
        return _Out(paints=[self._paint(key, snap)])

    def _leave_table(self, key, user_id) -> _Out:
        table = self.table(key)
        latest = self.store.latest_round(*key)
        seated = any(seat["kind"] == "member" and seat["ref_id"] == user_id for seat in latest["seats"])
        if latest["status"] == "playing" and seated:
            return _Out(LEAVE_BUSY)
        paints = []
        if latest["status"] == "joining" and seated:
            latest = self.store.leave_round(key[0], latest["round_id"], user_id)  # refunds the bet
            if not latest["seats"]:
                return self._close_empty(key)
            paints.append(self._paint(key, latest))
        table.present.discard(user_id)
        table.bets.pop(user_id, None)
        if table.present or latest["status"] in ("joining", "playing"):
            return _Out("You left the table.", paints)
        return self._close_empty(key)

    # --- timers' work -----------------------------------------------------------------------

    async def on_join_timer(self, guild_id, table_id, round_id) -> None:
        key, paint = (guild_id, table_id), None
        async with self.lock(key):
            closed, paint = self._closed(key)
            if not closed:
                snap = self.store.round_snapshot(guild_id, round_id)
                if snap["status"] != "joining" or not snap["seats"]:
                    return
                snap = self.store.deal(guild_id, round_id)
                self._after(key, snap)
                paint = self._paint(key, snap)
        if paint is not None:
            await self._deliver(paint)

    async def on_turn_timer(self, guild_id, table_id, round_id, moves) -> None:
        key, paint = (guild_id, table_id), None
        async with self.lock(key):
            closed, paint = self._closed(key)
            if not closed:
                snap = self.store.round_snapshot(guild_id, round_id)
                if snap["status"] != "playing" or snap["moves"] != moves or not snap["legal"]:
                    return
                seat, legal = next(iter(snap["legal"].items()))
                move = DefaultPolicy.pick(SimpleNamespace(total=blackjack.hand_total(snap["view"]["hands"][seat])[0]), legal)
                snap = self.store.play(guild_id, round_id, seat, move, "timeout", moves)
                self.table(key).timeouts.add((round_id, seat))
                self._after(key, snap)
                paint = self._paint(key, snap)
        if paint is not None:
            await self._deliver(paint)

    async def on_character_timer(self, guild_id, table_id, round_id, moves) -> None:
        """A character seat is on turn: decide outside the table lock (a model call may take seconds), then apply under it if nothing moved meanwhile."""
        key, paint = (guild_id, table_id), None
        async with self.lock(key):
            closed, paint = self._closed(key)
            if not closed:
                snap = self.store.round_snapshot(guild_id, round_id)
                if snap["status"] != "playing" or snap["moves"] != moves or not snap["legal"]:
                    return
                seat, legal = next(iter(snap["legal"].items()))
                seated = self.store.character_seat_context(guild_id, round_id, seat)
                if seated is None:
                    return
        if closed:
            if paint is not None:
                await self._deliver(paint)
            return
        move, line = await self._character_move(guild_id, snap, seat, legal, seated)
        async with self.lock(key):
            closed, paint = self._closed(key)
            if not closed:
                now = self.store.round_snapshot(guild_id, round_id)
                if now["status"] != "playing" or now["moves"] != moves or seat not in now["legal"]:
                    return
                legal = now["legal"][seat]
                if move not in legal:
                    move, line = self._house_move(now, seat, legal), ""
                try:
                    snap = self.store.play(guild_id, round_id, seat, move, "character", moves)
                except ConflictError:
                    return
                except GameError as error:
                    logging.warning("Blackjack character move refused (%s); playing the house rule", type(error).__name__)
                    move, line = self._house_move(now, seat, legal), ""
                    try:
                        snap = self.store.play(guild_id, round_id, seat, move, "character", moves)
                    except (ConflictError, GameError) as again:
                        logging.warning("Blackjack character house move refused (%s)", type(again).__name__)
                        return
                self._after(key, snap)
                paint = self._paint(key, snap)
        if paint is not None:
            await self._deliver(paint)
        if line and not closed:
            await self._post_line(snap["channel_id"], guild_id, seated, line)

    @staticmethod
    def _house_move(snap, seat, legal) -> str:
        return DefaultPolicy.pick(SimpleNamespace(total=blackjack.hand_total(snap["view"]["hands"][seat])[0]), legal)

    async def _character_move(self, guild_id, snap, seat, legal, seated) -> tuple[str, str]:
        """(move, line): the model's choice when table talk is on and the call works, else the house rule with no line."""
        house = self._house_move(snap, seat, legal)
        if not self.store.game_character_talk(guild_id):
            return house, ""
        try:
            character = self.store.character_by_id(seated["character_id"])
            if character is None or character["guild_id"] != guild_id:
                return house, ""
            result = await asyncio.wait_for(game_chooser.ask(self.bot, guild_id, snap["channel_id"], character, snap, seat, legal), self.MODEL_SECONDS)
        except Exception as error:
            logging.warning("Blackjack character decision fell back to the house rule (%s)", type(error).__name__)
            return house, ""
        move = result.get("move") if isinstance(result, dict) else None
        if not isinstance(move, str) or move not in legal:
            return house, ""
        return move, game_chooser.clean_line(result.get("line"))

    async def _post_line(self, channel_id, guild_id, seated, line) -> None:
        """Best effort: the character says its line in the table's channel; a failure is logged and never blocks the game. Nothing is saved to scene history."""
        try:
            character = self.store.character_by_id(seated["character_id"])
            if character is None or character["guild_id"] != guild_id:
                return
            channel = self.bot.get_channel(channel_id) or await self.bot.fetch_channel(channel_id)
            webhook = await self.bot._webhook(channel, character)
            slots = {row["slot_key"]: row for row in self.store.usable_avatars(guild_id, character["id"])}
            avatar = (await self.bot.resolve_avatar(slots["neutral"]))["url"] if "neutral" in slots else None
            thread = {"thread": channel} if isinstance(channel, discord.Thread) else {}
            await asyncio.wait_for(webhook.send(line, **thread, username=character["name"], avatar_url=avatar, allowed_mentions=NO_MENTIONS), self.POST_SECONDS)
        except Exception as error:
            logging.warning("Blackjack character line not posted (%s)", type(error).__name__)

    async def on_idle_timer(self, guild_id, table_id, round_id) -> None:
        key, paint = (guild_id, table_id), None
        async with self.lock(key):
            closed, paint = self._closed(key)
            if not closed:
                snap = self.store.latest_round(guild_id, table_id)
                open_table = self.store.open_table_for(guild_id, snap["channel_id"])
                if open_table is None or open_table["id"] != table_id or snap["status"] in ("joining", "playing"):
                    return
                self._close_in_store(key)
                paint = self._closing(key, CLOSED)
        if paint is not None:
            await self._deliver(paint)

    # --- restart ----------------------------------------------------------------------------

    async def recover(self) -> None:
        """Refunds and closes every table left open by a previous run, then tells the channel (best effort)."""
        store, refunded, closed, failed = self.store, 0, 0, 0
        rounds = store.unfinished_rounds()
        tables = {(r["guild_id"], r["table_id"]): (r["channel_id"], r["message_id"]) for r in rounds}
        for row in store.all("SELECT id,guild_id,channel_id,message_id FROM game_tables WHERE status='open'"):
            tables.setdefault((row["guild_id"], row["id"]), (row["channel_id"], row["message_id"]))
        for r in rounds:
            try:
                store.cancel_round(r["guild_id"], r["round_id"], "bot restarted")
                refunded += 1
            except Exception:
                failed += 1
                logging.exception("Could not refund blackjack round %s on restart", r["round_id"])
        for (guild_id, table_id), (channel_id, message_id) in tables.items():
            try:
                store.close_table(guild_id, table_id)
                closed += 1
            except Exception:
                failed += 1
                logging.exception("Could not close blackjack table %s on restart", table_id)
                continue
            await self.edit(channel_id, message_id, "This table closed when the bot restarted. Bets were refunded.", None)
        if refunded or closed or failed:
            logging.info("Blackjack recovery: %d rounds refunded, %d tables closed, %d failures", refunded, closed, failed)


def register_game_commands(bot, ctx: SimpleNamespace, admin_games: app_commands.Group) -> None:
    require_guild = ctx.require_guild

    async def reply(interaction: discord.Interaction, message: str) -> None:
        await interaction.response.send_message(message, ephemeral=True, allowed_mentions=NO_MENTIONS)

    @bot.tree.command(name="blackjack", description="Join or open a blackjack table in this channel")
    @app_commands.describe(bet="How much to bet", favorites="Bring your favorite characters to the table")
    async def blackjack_command(interaction: discord.Interaction, bet: app_commands.Range[int, 1, MAX_BET], favorites: bool = False):
        require_guild(interaction)
        await bot.games.play_command(interaction, bet, favorites)

    @admin_games.command(name="channel", description="Turn games on or off in a channel")
    @app_commands.describe(enabled="On or off", channel="The channel (default: this one)")
    @app_commands.checks.has_permissions(administrator=True)
    async def games_channel(interaction: discord.Interaction, enabled: bool, channel: discord.TextChannel | None = None):
        require_guild(interaction)
        channel_id = channel.id if channel else getattr(interaction, "channel_id", None) or interaction.channel.id
        where = channel.mention if channel else f"<#{channel_id}>"
        closed, paint = await bot.games.switch_channel(interaction.guild_id, channel_id, enabled)
        text = f"Games are now {'on' if enabled else 'off'} in {where}."
        try:
            await reply(interaction, text + (" The open table was closed and bets were refunded." if closed else ""))
        finally:
            if paint is not None:
                await bot.games.send_paint(paint)

    @admin_games.command(name="bets", description="Set the smallest and largest bet")
    @app_commands.describe(min="Smallest bet", max="Largest bet")
    @app_commands.checks.has_permissions(administrator=True)
    async def games_bets(interaction: discord.Interaction, min: app_commands.Range[int, 1, MAX_BET], max: app_commands.Range[int, 1, MAX_BET]):
        require_guild(interaction)
        try:
            saved = bot.store.set_game_settings(interaction.guild_id, min, max, expected=bot.store.game_settings(interaction.guild_id))
        except ConflictError:
            await reply(interaction, "The game settings were changed elsewhere. Run the command again.")
            return
        name = bot.store.currency_name(interaction.guild_id)
        await reply(interaction, f"Bets are now {saved['min_bet']:,} to {saved['max_bet']:,} {name}.")
