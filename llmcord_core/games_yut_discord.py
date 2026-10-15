"""Yut (윷놀이) on Discord (FEAT-58, D42): `/game start game:Yut`, lobby, table thread, board image, throws and moves, timers, settlement, recovery.

Humans only (character seats are FEAT-60). Rules, money and persistence live in ``games.yut`` and the store; the machinery mirrors
``games_onecard_discord``: every table runs under its own lock (store work only; no Discord call is made under it), a handler decides under
the lock and builds an ``_Out`` that is delivered afterwards one table at a time, and timers re-check the stored state under the lock.
Nothing is hidden in Yut, so the pieces of every seat are on the pinned board message: the PNG from ``games.yut_render`` (rendered off the
event loop, cached by piece positions) plus its English text fallback, which is always included.
A throw is one public line (``Gae (▮▮▯▯)``). The turn message carries Throw (throw phase) or Move (move phase) and Forfeit; Move and Forfeit
open private (ephemeral) views, so a misclick never moves a piece. A missed turn (``turn_timeout``) makes the throw, or the first legal move,
for the seat (actor ``timeout``); the second missed decision in a row forfeits the seat.
"""
from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass, field

import discord

from . import game_chooser
from .admin_store import ConflictError, GameError
from .game_talk import TalkBudget
from .games import yut, yut_render
from .games_discord import NOT_SEATED, NOT_YOUR_TURN, NO_MENTIONS, OFF_HERE, STALE, GameTables, _sign, words
from .games_doubt_discord import (GAME_OVER, JOINED, JOINED_ANTE, LEFT_CLOSED, LEFT_REFUNDED, NO_TABLE, NOT_STARTED, ONLY_HOST_START, THREAD_FAILED,
                                  THREAD_FORBIDDEN, THREAD_NOT_READY)
from .table_runtime import TableRuntime
from llmcord_core import i18n
from llmcord_core.i18n import N_, Phrase, discord_locale, terms

PREFIX = "llmcord:yut"
YUT_OFF = N_("Yut is off on this server.")
NO_YUT_CHARACTERS = N_("Characters can't sit at a Yut table yet.")
YUT_MOVED_ON = N_("The table moved on — use the buttons on the newest message.")
NOT_A_MOVE = N_("Pick one of the moves in the list.")
NOT_A_THROW = N_("There is nothing to throw right now.")
NOTHING_TO_MOVE = N_("There is nothing to move right now.")
OUT_OF_GAME = N_("You are out of this game.")
LEFT_TABLE = N_("You left the table.")
NEEDS_SEATS = N_("Yut needs at least {count} seats to start.")
YOU_MOVE = N_("You move piece {piece} with {throw}.")
YOU_FORFEIT = N_("You forfeit the game.")
FORFEIT_ASK = N_("Forfeit the game? Your ante stays in the pot and you are out.")
NO_HIDDEN = N_("Yut has no hidden cards: every piece is on the board.")
PIECE_COLORS = ("🔴", "🔵", "🟢", "🟡")
_STALE = object()  # the table moved on while a decision was made: drop it
MOVE_ACTIONS = ("move", "forfeit")  # private views, never an acknowledgement


def _sticks(sticks) -> str:
    return "".join("▮" if flat else "▯" for flat in sticks)


class YutButton(discord.ui.DynamicItem[discord.ui.Button], template=PREFIX + r":(?P<action>[a-z_]+):(?P<table>\d+):(?P<round>\d+):(?P<moves>\d+)"):
    """One button; its custom_id carries the table, the round and the move count the message showed. Never a seat."""

    def __init__(self, action: str, table_id: int, round_id: int, moves: int, label: str = "", style=discord.ButtonStyle.secondary, disabled: bool = False):
        super().__init__(discord.ui.Button(label=label, style=style, disabled=disabled, custom_id=f"{PREFIX}:{action}:{table_id}:{round_id}:{moves}"))
        self.action, self.table_id, self.round_id, self.moves = action, table_id, round_id, moves

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["action"], int(match["table"]), int(match["round"]), int(match["moves"]), item.label or "", item.style, item.disabled)

    async def callback(self, interaction: discord.Interaction):
        await interaction.client.yut.press(interaction, self.action, self.table_id, self.round_id, self.moves)


class _MoveSelect(discord.ui.Select):
    def __init__(self, options: list, *, t=None):
        t = t or i18n.english()
        super().__init__(placeholder=t.p("select.yut", "Move to make"), min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        picker = self.view
        await interaction.client.yut.play_from_picker(interaction, picker.table_id, picker.round_id, picker.moves, self.values[0])


class _Picker(discord.ui.View):
    """The private move picker: one option per distinct (piece, banked throw) move (25 to a select)."""

    def __init__(self, table_id: int, round_id: int, moves: int, options: list, *, t=None):
        super().__init__(timeout=600)
        self.table_id, self.round_id, self.moves = table_id, round_id, moves
        for start in range(0, len(options), 25):
            self.add_item(_MoveSelect(options[start:start + 25], t=t))


class _ForfeitView(discord.ui.View):
    """The private confirmation of a forfeit."""

    def __init__(self, table_id: int, round_id: int, *, t=None):
        super().__init__(timeout=600)
        t = t or i18n.english()
        self.table_id, self.round_id = table_id, round_id
        button = discord.ui.Button(label=t.p("button.yut", "Yes, forfeit"), style=discord.ButtonStyle.danger)
        button.callback = self.confirm
        self.add_item(button)

    async def confirm(self, interaction: discord.Interaction):
        await interaction.client.yut.play_from_picker(interaction, self.table_id, self.round_id, None, "forfeit")


@dataclass(eq=False)
class _Table:
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    issued: int = 0  # numbers the outgoing edits, assigned under the table lock
    sent: dict = field(default_factory=dict)  # newest delivered number per message kind
    talk: int = 0  # member messages in the thread since the turn message was posted
    deadlines: dict = field(default_factory=dict)  # lobby / turn -> unix time


@dataclass
class _Out:
    """What a locked step decided: a private reply and the messages to send after the lock is released."""
    say: str | None = None
    key: tuple | None = None
    ok: bool = False  # the reply confirms a made move
    events: list = field(default_factory=list)  # public lines for the thread, in order
    lobby: tuple | None = None  # (content, view or None)
    turn: tuple | None = None  # (content, view)
    board: bool = False
    start: bool = False  # post the board and the turn message, pin the board
    final: tuple | None = None  # (result text, lobby text, turn text)
    advanced: bool = False  # a move was made: the turn message may be moved to the bottom
    seq: int = 0
    talk: tuple | None = None  # (public text, is the game's end): the event turn for the cast watching the table
    closing: str | None = None  # the table was closed from outside: this line goes to the thread and the turn message loses its buttons
    closed: bool = False  # the table is over: forget it once this is delivered


class YutTables:
    LOBBY_SECONDS = 300
    ARCHIVE_SECONDS = 600
    BOARD_DEBOUNCE_SECONDS = 1
    RETRY_SECONDS = 5  # read once at construction (TableRuntime.retry_seconds); tests set runtime.retry_seconds
    BUMP_AFTER = 8
    EVENT_COOLDOWN_SECONDS = 20
    SWEEP_SECONDS = 30

    def __init__(self, bot, clock=time.time):
        self.bot, self.clock = bot, clock
        self.runtime = TableRuntime(label="Yut", retry_seconds=self.RETRY_SECONDS)
        self.tables: dict[tuple[int, int], _Table] = {}
        self.threads: dict[int, tuple[int, int]] = {}  # thread id -> (guild, table) of the tables being played
        self.banter: dict[int, tuple[int, int]] = {}  # thread id -> (guild, table) of finished games, until the archive timer
        self.event_at: dict[tuple[int, int], float] = {}  # table -> clock time of its last event turn
        self.event_tasks: set[asyncio.Task] = set()
        self.dirty: set[tuple[int, int]] = set()
        self.board_tasks: dict[tuple[int, int], asyncio.Task] = {}
        self.uploaded: dict[tuple[int, int], tuple] = {}  # table -> piece positions of the PNG on its board message
        self.ended: set[tuple[int, int]] = set()  # tables this object closed itself (settled, timed out, left): the sweeper leaves them alone
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
        hook = {"lobby": self.on_lobby_timer, "turn": self.on_turn_timer, "archive": self.on_archive_timer}[kind]
        self.runtime.arm(key, kind, delay, self._run, kind, hook, key, args, name=f"yut-{kind}-{key[1]}")

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
        for kind in ("lobby", "turn"):
            self.runtime.disarm(key, kind)
        self.threads = {thread: k for thread, k in self.threads.items() if k != key}
        self.event_at.pop(key, None)
        self.uploaded.pop(key, None)
        self.dirty.discard(key)

    def _forget(self, key) -> None:
        """This object closed the table (in the store): release it, and keep the sweeper off it until its archive timer."""
        self.ended.add(key)
        self._release(key)

    # --- tables closed from outside (dashboard, /admin games channel) -------------------------

    def _ensure_sweeper(self) -> None:
        if self.tables:
            self.runtime.ensure_sweeper(interval=lambda: self.SWEEP_SECONDS, held=lambda: bool(self._known()), sweep=self.sweep, name="yut-sweep")

    def _known(self) -> set:
        return {*self.tables, *self.threads.values(), *(k[0] for k in self.runtime.timers if k[1] != "archive")}

    async def sweep(self) -> None:
        """Cleans up tables closed in the store by someone else (dashboard switch, channel removal) while this object still holds them."""
        for key in self._known():
            await self._reap(key)

    def _open(self, row) -> bool:
        return row is not None and row["game"] == "yut" and row["status"] == "open"

    async def _reap(self, key) -> None:
        """If the table is closed in the store and this object did not close it, say so (lobby, turn message, thread), archive and forget it."""
        if key in self.ended:
            self._release(key)  # idempotent: a thread registered while the table closed must not keep it known
            return
        if key not in self.tables and key not in self.threads.values() and not any(k[0] == key for k in self.runtime.timers):
            return
        if self._open(self._row(key)):
            return
        async with self.runtime.lock(key):
            out = self._closed_out(key)
        await self._publish(key, out)

    def _closed_out(self, key) -> _Out | None:
        """Under the table lock: the closing messages for a table that is closed in the store, or None (still open, or already handled)."""
        row = self._row(key)
        if key in self.ended or self._open(row):
            return None
        if row is None or row["game"] != "yut":
            self._release(key)
            self.tables.pop(key, None)
            return None
        guild_id, t = key[0], self._t(key[0])
        if not self.store.game_enabled(guild_id, "yut"):
            reason = t("Table closed: {game} was turned off on this server.", game=terms.GAME_YUT)
        elif row["channel_id"] not in self.store.game_channels(guild_id):
            reason = t("Table closed: games were turned off here.")
        else:
            reason = t("Table closed: the table was closed.")
        self._forget(key)
        self.arm(key, "archive", self.ARCHIVE_SECONDS)
        text = reason + self._refunded(key, t=t)
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
        """Arms the timers the new state needs (a fresh turn timeout for every decision). Call under the table lock."""
        table, view = self.table(key), snap["view"]
        self.runtime.disarm(key, "lobby")
        if snap["status"] != "playing" or not view or view["turn"] is None:
            table.deadlines.pop("turn", None)
            self.runtime.disarm(key, "turn")
            return
        delay = (snap["rules"] or {}).get("turn_timeout", 60)
        table.deadlines["turn"] = self.clock() + delay
        self.arm(key, "turn", delay, snap["round_id"], snap["moves"])

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
            logging.warning("Could not edit Yut message %s in channel %s: %s", message_id, channel_id, type(error).__name__)
        except Exception:
            logging.exception("Could not edit Yut message %s in channel %s", message_id, channel_id)
        return None

    async def _delete(self, channel_id, message_id) -> bool:
        if channel_id is None or message_id is None:
            return False
        try:
            channel = await self._channel(channel_id)
            await channel.get_partial_message(message_id).delete()
            return True
        except discord.HTTPException as error:
            logging.info("Could not delete Yut message %s (%s)", message_id, type(error).__name__)
        except Exception:
            logging.exception("Could not delete Yut message %s", message_id)
        return False

    async def _post(self, channel_id, content, view=None, **kwargs):
        if channel_id is None:
            return None
        try:
            channel = await self._channel(channel_id)
            return await channel.send(content, view=view, allowed_mentions=NO_MENTIONS, **kwargs)
        except discord.HTTPException as error:
            logging.warning("Could not post to Yut thread %s: %s", channel_id, type(error).__name__)
        except Exception:
            logging.exception("Could not post to Yut thread %s", channel_id)
        return None

    async def _say(self, interaction, text, **kwargs) -> None:
        """An ephemeral reply: the first response, or a followup once the interaction was acknowledged."""
        text = words(discord_locale.invoker_t(interaction), text)
        try:
            if interaction.response.is_done():
                await interaction.followup.send(text, ephemeral=True, allowed_mentions=NO_MENTIONS, **kwargs)
            else:
                await interaction.response.send_message(text, ephemeral=True, allowed_mentions=NO_MENTIONS, **kwargs)
        except discord.HTTPException as error:
            logging.warning("Yut reply failed: %s", type(error).__name__)

    async def answer(self, interaction, text: str, done: bool = False) -> None:
        """A reply to a private view: ``done`` replaces its message, anything else is a new private message (the view stays)."""
        text = words(discord_locale.invoker_t(interaction), text)
        if done and not interaction.response.is_done() and getattr(interaction, "message", None) is not None:
            try:
                await interaction.response.edit_message(content=text, view=None, allowed_mentions=NO_MENTIONS)
                return
            except discord.HTTPException as error:
                logging.warning("Yut picker edit failed: %s", type(error).__name__)
        await self._say(interaction, text)

    async def _ack(self, interaction) -> None:
        try:
            await interaction.response.defer()
        except discord.HTTPException as error:
            logging.warning("Yut could not acknowledge the press: %s", type(error).__name__)

    def _refunded(self, key, past: bool = False, *, t=None) -> str:
        """" The antes are refunded." (``past``: "were") when the table's round had an ante, else nothing."""
        t = t or i18n.english()
        try:
            ante = (self.store.latest_round(*key)["rules"] or {}).get("ante", 0)
        except GameError:
            ante = 0
        return (" " + (t("The antes were refunded.") if past else t("The antes are refunded."))) if ante else ""

    def _host(self, row, snap) -> int:
        """The table's host: the opener while seated, else the first seated member."""
        members = [s["ref_id"] for s in snap["seats"] if s["kind"] == "member"]
        return row["opened_by"] if row["opened_by"] in members or not members else members[0]

    def _is_host(self, row, snap, user_id) -> bool:
        return user_id == self._host(row, snap)

    def _row(self, key):
        return self.store.one("SELECT * FROM game_tables WHERE id=? AND guild_id=?", (key[1], key[0]))

    # --- rendering --------------------------------------------------------------------------

    @staticmethod
    def _who(snap, index, *, t=None) -> str:
        return GameTables._who({**snap["seats"][index], "brought_by": None}, t=t)

    def _stamp(self, key, kind) -> str:
        deadline = self.table(key).deadlines.get(kind)
        return f"<t:{int(deadline)}:R>" if deadline else ""

    def _t(self, guild_id):
        """The channel's translator for a table message: resolved at each render, never kept."""
        get_guild = getattr(self.bot, "get_guild", None)
        return discord_locale.channel_t(get_guild(guild_id) if get_guild else None)

    def render_lobby(self, key, snap, row, *, t=None) -> tuple[str, discord.ui.View]:
        t = t or self._t(key[0])
        rules = snap["rules"] or yut.normalize_rules(None)
        name, pot, ante = self.store.currency_name(key[0]), sum(s["stake"] for s in snap["seats"]), rules["ante"]
        lines = [t("**{game}** · table {number}", game=terms.GAME_YUT, number=key[1]),
                 t("Ante: {ante:,} {currency} · Pot: {pot:,} {currency}", ante=ante, pot=pot, currency=name) if ante else t("No ante · Pot: {pot:,} {currency}", pot=pot, currency=name),
                 *([t("Host: {host}", host=f"<@{self._host(row, snap)}>")] if any(s["kind"] == "member" for s in snap["seats"]) else []), ""]
        for seat in snap["seats"]:
            lines.append(f"{PIECE_COLORS[seat['index'] % 4]} {seat['index'] + 1}. {GameTables._who(seat, t=t)}")
        stamp = self._stamp(key, "lobby")
        lines.append("")
        if stamp:
            lines.append(t("Join with the button. The host starts the game with {low} to {high} seats; the table closes {stamp} if it hasn't started.", low=yut.MIN_SEATS, high=yut.MAX_SEATS, stamp=stamp))
        else:
            lines.append(t("Join with the button. The host starts the game with {low} to {high} seats.", low=yut.MIN_SEATS, high=yut.MAX_SEATS))
        lines.append("-# " + " · ".join(t.render(p) for p in yut.rules_phrases(rules)))
        view, tid, rid = discord.ui.View(timeout=None), key[1], snap["round_id"]
        spec = [("join", t.p("button.yut", "Join"), discord.ButtonStyle.primary, False), ("leave", t.p("button.yut", "Leave"), discord.ButtonStyle.secondary, False),
                ("start", t.p("button.yut", "Start"), discord.ButtonStyle.success, len(snap["seats"]) < yut.MIN_SEATS or row["thread_id"] is None),
                ("help", t.p("button.yut", "How to play"), discord.ButtonStyle.secondary, False)]
        for action, label, style, disabled in spec:
            view.add_item(YutButton(action, tid, rid, 0, label, style, disabled))
        return "\n".join(lines), view

    def _in_play_view(self, key, snap, *, t=None) -> discord.ui.View:
        t = t or self._t(key[0])
        view = discord.ui.View(timeout=None)
        view.add_item(YutButton("help", key[1], snap["round_id"], 0, t.p("button.yut", "How to play")))
        return view

    def render_board(self, key, snap, *, t=None) -> str:
        """The board message text: the seats and the plain-text positions (the PNG is attached when the message is sent or edited)."""
        t = t or self._t(key[0])
        view, rules, name = snap["view"], snap["rules"] or {}, self.store.currency_name(key[0])
        pot = sum(s["stake"] for s in snap["seats"])
        lines = [t("**{game}** · table {number}", game=terms.GAME_YUT, number=key[1]) + (" · " + t("pot {pot:,} {currency} (ante {ante:,})", pot=pot, currency=name, ante=rules.get('ante', 0)) if pot else "")]
        for seat in snap["seats"]:
            i = seat["index"]
            if not view["active"][i]:
                state = t.p("seat state", "out")
            else:
                state = ""
                if view["finished"] and view["winner"] == i:
                    state = t.p("seat mark", "winner")
                elif view["turn"] == i:
                    state = t.p("seat mark", "on turn")
            lines.append(f"{PIECE_COLORS[i % 4]} {i + 1}. {GameTables._who(seat, t=t)}" + (f" — {state}" if state else ""))
        if view["finished"]:
            lines.append(t("Game over."))
        for seat in snap["seats"]:
            row = view["pieces"][seat["index"]]
            on = ", ".join(p for p in row if p not in (None, "done")) or "-"
            start, home = sum(p is None for p in row), sum(p == "done" for p in row)
            lines.append("-# " + t("{who}: on board {board} · at start {start} · home {home}", who=self._who(snap, seat["index"], t=t), board=on, start=start, home=home))
        lines.append("-# " + " · ".join(t.render(p) for p in yut.rules_phrases(snap["rules"])))
        lines.append("-# " + t("Seed: {seed} (hash {hash})", seed=snap['seed'], hash=snap['seed_hash'][:12]) if snap["seed"] else "-# " + t("Seed hash: {hash}", hash=snap['seed_hash'][:12]))
        return "\n".join(lines)

    def _event_lines(self, snap, event, *, t=None) -> list:
        """The public line(s) for one history event (a throw, a move with its captures, a lost 빽도, a forfeit); hands do not exist, names only."""
        t = t or i18n.english()
        who = self._who(snap, event["seat"], t=t)
        kind = event["kind"]
        if kind == "throw":
            return [t("{who} threw {throw} ({sticks}).", who=who, throw=terms.YUT_THROW[event["name"]], sticks=_sticks(event["sticks"]))]
        if kind == "move":
            pieces = "+".join(str(p + 1) for p in event["pieces"])
            line = (t("{who} brought piece {pieces} home with {throw}.", who=who, pieces=pieces, throw=terms.YUT_THROW[event["throw"]]) if event["end"] == "done"
                    else t("{who} moved piece {pieces} with {throw}.", who=who, pieces=pieces, throw=terms.YUT_THROW[event["throw"]]))
            victims = list(dict.fromkeys(seat for seat, _piece in event["captured"]))
            return [line, *(t("{who} captured {victim}!", who=who, victim=self._who(snap, seat, t=t)) for seat in victims)]
        if kind == "drop":
            return [t("{who}'s Back-do could not be used and is lost.", who=who)]
        if kind == "forfeit":
            return [t("{who} forfeits and is out of the game.", who=who)]
        return []

    def _happened(self, snap, *, t=None) -> str:
        """The line for the last thing that happened (the newest throw or move with what it did)."""
        t = t or i18n.english()
        history = [e for e in snap["view"]["history"] if e["kind"] != "win"]
        if not history:
            return t("The game has started.")
        last = history[-1]
        if last["kind"] in ("drop", "forfeit") and len(history) > 1 and history[-2]["kind"] in ("throw", "move"):
            return " ".join(self._event_lines(snap, history[-2], t=t) + self._event_lines(snap, last, t=t))
        return " ".join(self._event_lines(snap, last, t=t))

    def render_turn(self, key, snap, *, t=None) -> tuple[str, discord.ui.View]:
        t = t or self._t(key[0])
        view, seats = snap["view"], snap["seats"]
        lines = [self._happened(snap, t=t)]
        if view["bank"]:
            banked = ", ".join(t.render(terms.YUT_THROW[n]) for n in view["bank"])
            lines.append(t("Banked throws: {throws}.", throws=banked))
        turn = view["turn"]
        buttons = discord.ui.View(timeout=None)
        if turn is not None:
            stamp, who = self._stamp(key, "turn"), self._who(snap, turn, t=t)
            lines.append(t("Next: {who} — {stamp}.", who=who, stamp=stamp) if stamp and seats[turn]["kind"] == "member" else t("Next: {who}.", who=who))
            lines.append(t("Throw the sticks.") if view["phase"] == "throw" else t("Choose a banked throw and a piece to move."))
            if view["phase"] == "throw":
                buttons.add_item(YutButton("throw", key[1], snap["round_id"], snap["moves"], t.p("button.yut", "Throw"), discord.ButtonStyle.primary))
            else:
                buttons.add_item(YutButton("move", key[1], snap["round_id"], snap["moves"], t.p("button.yut", "Move"), discord.ButtonStyle.primary))
            buttons.add_item(YutButton("forfeit", key[1], snap["round_id"], snap["moves"], t.p("button.yut", "Forfeit"), discord.ButtonStyle.danger))
        return "\n".join(lines), buttons

    def _result(self, key, snap, *, t=None) -> tuple[str, str, str]:
        """(thread result, one-line lobby text, final turn text) for a settled round."""
        t = t or self._t(key[0])
        name, seats, win = self.store.currency_name(key[0]), snap["seats"], snap["view"]["winner"]
        pot, who = sum(s["stake"] for s in seats), self._who(snap, win, t=t)
        lines = [t("**{who} wins the game and takes the pot of {pot:,} {currency}.**", who=who, pot=pot, currency=name) if pot else t("**{who} wins the game.**", who=who)]
        even = t.p("round net", "even")
        for seat in seats:
            net = (seat["payout"] or 0) - seat["stake"]
            note = " " + t("(forfeited)") if seat["outcome"] == "forfeit" else ""
            lines.append(f"• {GameTables._who(seat, t=t)}: {_sign(net) or even}{note}")
        if snap["seed"]:
            lines.append("-# " + t("Seed: {seed} (hash {hash})", seed=snap['seed'], hash=snap['seed_hash'][:12]))
        lobby = t("{who} won {game} and took the pot of {pot:,} {currency}. Table closed.", who=who, game=terms.GAME_YUT, pot=pot, currency=name) if pot \
            else t("{who} won {game}. Table closed.", who=who, game=terms.GAME_YUT)
        return "\n".join(lines), lobby, t("Game over — {who} won.", who=who)

    # --- the common step machinery ----------------------------------------------------------

    def _seal(self, key, out: _Out) -> _Out:
        table = self.table(key)
        table.issued += 1
        out.key, out.seq = key, table.issued
        return out

    def _load(self, key, round_id):
        """((table row, snapshot), None) for an open Yut table and one of its rounds, else (None, refusal)."""
        row = self._row(key)
        if not self._open(row):
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
        if not self._open(row):
            self._release(key)
            return None
        try:
            snap = self.store.round_snapshot(key[0], round_id)
        except GameError:
            return None
        if snap["table_id"] != key[1] or snap["status"] != "playing" or snap["moves"] != moves or not snap["view"]:
            return None
        return snap

    def _apply(self, key, snap, seat, move, actor, note=None) -> _Out:
        """Makes one move under the table lock; raises ConflictError / GameError from the store."""
        before, new = snap, self.store.play(key[0], snap["round_id"], seat, move, actor, snap["moves"])
        t = self._t(key[0])
        out = _Out(advanced=True)
        if note:
            out.events.append(note)
        for event in new["view"]["history"][len(before["view"]["history"]):]:
            out.events.extend(self._event_lines(before, event, t=t))
        if new["status"] == "settled":
            out.final = self._result(key, new, t=t)
            if self.talk_allowed(key[0]):
                out.talk = (f"Game over — {self._talk_names(key[0], new)[new['view']['winner']]} won the game.", True)
            self.store.close_table(*key)
            self._forget(key)
            if (row := self._row(key)) is not None and row["thread_id"] is not None:
                self.banter[row["thread_id"]] = key
            self.arm(key, "archive", self.ARCHIVE_SECONDS)
        else:
            self._after(key, new)
            out.turn = self.render_turn(key, new, t=t)
            out.board = True
            if self.clock() - self.event_at.get(key, float("-inf")) >= self.EVENT_COOLDOWN_SECONDS and self.talk_allowed(key[0]):
                text = self._moment(key, before, new)
                if text is not None:
                    self.event_at[key] = self.clock()
                    out.talk = (text, False)
        return self._seal(key, out)

    def _moment(self, key, before, new) -> str | None:
        """The public line for a key moment of the move just made (a capture, a piece home), else None. Names only: no markdown, no mention."""
        names = self._talk_names(key[0], new)
        for event in new["view"]["history"][len(before["view"]["history"]):]:
            if event["kind"] != "move":
                continue
            if event["captured"]:
                return f"{names[event['seat']]} captured a piece of {names[event['captured'][0][0]]}."
            if event["end"] == "done":
                return f"{names[event['seat']]} brought a piece home."
        return None

    async def _publish(self, key, out: _Out | None) -> None:
        """After the table lock is released: the thread lines, then the edits (an edit older than the last one sent is dropped)."""
        if out is None:
            return
        await self._send(key, out)

    async def _send(self, key, out: _Out) -> None:
        table = self.tables.get(key)
        if table is None:
            return  # sealed outs register the table; a missing record means its archive already ran, so nothing may be said or re-registered
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
                self._talk_event(thread_id, out, key[0])
                await self._edit(thread_id, row["turn_message_id"], content=turn, view=None)
                self.dirty.discard(key)
                await self._edit_board(key)
                await self._edit(channel_id, row["message_id"], content=lobby, view=None)
                return
            self._talk_event(thread_id, out, key[0])
            if out.closing:
                if out.seq > table.sent.get("turn", 0):
                    table.sent["turn"] = out.seq
                    await self._edit(thread_id, row["turn_message_id"], content=self._t(key[0])("Table closed."), view=None)
                await self._post(thread_id, out.closing)
            if out.lobby and out.seq > table.sent.get("lobby", 0):
                table.sent["lobby"] = out.seq
                await self._edit(channel_id, row["message_id"], content=out.lobby[0], view=out.lobby[1])
            if out.turn and not out.start and out.seq > table.sent.get("turn", 0):
                table.sent["turn"] = out.seq
                await self._turn_message(key, out, row)
            if out.board:
                self._mark_board(key)

    def _talk_event(self, thread_id, out, guild_id=None) -> None:
        """Fire and forget: the cast watching the table reacts to a public event (FEAT-55). A move never waits for it."""
        if out.talk is None or thread_id is None or self.closed:
            return
        text = out.talk[0]
        out.talk = None

        async def run():
            try:
                thread = await self._channel(thread_id)
                await self.bot.run_table_event(thread, text)
                if guild_id is not None and self.bot.store.cast_watches_games(guild_id):
                    await self.bot.run_table_event(thread, text, spectators=True)  # after the seated turn, not under it
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("Yut event turn failed (thread %s)", thread_id)

        task = asyncio.create_task(run(), name=f"yut-event-{thread_id}")
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

    async def _board_file(self, view) -> discord.File | None:
        """The board PNG, rendered off the event loop (cached by piece positions); None when rendering fails (the text fallback still says everything)."""
        try:
            return discord.File(io.BytesIO(await asyncio.to_thread(yut_render.render_board_cached, view)), filename="yut-board.png")
        except Exception as error:
            logging.warning("Could not render the Yut board (%s)", type(error).__name__)
            return None

    @staticmethod
    def _positions(view) -> tuple:
        return tuple(tuple(row) for row in view["pieces"])

    async def _open_play(self, key, out, row) -> bool:
        """Posts the board (pinned) and the turn message in the thread. False when the thread cannot be written to; the table is then closed."""
        thread_id = row["thread_id"]
        snap = self.store.latest_round(*key)
        file = await self._board_file(snap["view"])
        if file:
            self.uploaded[key] = self._positions(snap["view"])
        board = await self._post(thread_id, self.render_board(key, snap), **({"file": file} if file else {}))
        turn = await self._post(thread_id, *out.turn) if board is not None else None
        if board is None or turn is None:
            t = self._t(key[0])
            async with self.runtime.lock(key):
                refunded = self._refunded(key, t=t)
                if (current := self._row(key)) is not None and current["status"] == "open":
                    self.store.close_table(*key)
                self._forget(key)
            await self._edit(row["channel_id"], row["message_id"], content=t("Table closed: I couldn't post in the table thread.") + refunded, view=None)
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
            task = self.board_tasks[key] = asyncio.create_task(self._debounce(key), name=f"yut-board-{key[1]}")
            task.add_done_callback(lambda done: self.board_tasks.get(key) is done and self.board_tasks.pop(key, None))

    async def _debounce(self, key) -> None:
        await asyncio.sleep(self.BOARD_DEBOUNCE_SECONDS)
        if key in self.dirty:
            self.dirty.discard(key)
            await self._edit_board_locked(key)

    async def _edit_board_locked(self, key) -> None:
        """A board edit under the table's send lock, reading the newest state inside it: an older edit can never land after the final one."""
        table = self.tables.get(key)
        if table is None:
            return
        async with table.send_lock:
            await self._edit_board(key)

    async def _edit_board(self, key) -> None:
        try:
            snap = self.store.latest_round(*key)
        except GameError:
            return
        if snap["view"] and snap["board_message_id"] is not None:
            positions, extra = self._positions(snap["view"]), {}
            if self.uploaded.get(key) != positions:  # the PNG only depends on the positions: same positions, edit the text alone
                file = await self._board_file(snap["view"])
                extra = {"attachments": [file] if file else []}
                if file:
                    self.uploaded[key] = positions
                else:
                    self.uploaded.pop(key, None)
            await self._edit(snap["thread_id"], snap["board_message_id"], content=self.render_board(key, snap), **extra)

    async def flush(self) -> None:
        """Sends the pending board edits now."""
        keys = list(self.dirty)
        self.dirty.clear()
        for key in keys:
            await self._edit_board_locked(key)

    # --- /game start ------------------------------------------------------------------------

    async def start_command(self, interaction: discord.Interaction, favorites: bool = False) -> None:
        guild_id = interaction.guild_id
        channel_id = getattr(interaction, "channel_id", None) or interaction.channel.id
        user = interaction.user
        if user.bot:
            await self._say(interaction, discord_locale.invoker_t(interaction)("Bots cannot play {game}.", game=terms.GAME_YUT))
            return
        if not self.store.game_enabled(guild_id, "yut"):
            await self._say(interaction, YUT_OFF)
            return
        if channel_id not in self.store.game_channels(guild_id):
            await self._say(interaction, OFF_HERE)
            return
        note = NO_YUT_CHARACTERS if favorites else ""
        try:
            opened = self.store.open_table(guild_id, channel_id, "yut", user.id)
        except GameError as error:
            await self._say(interaction, error)
            return
        key, round_id = (guild_id, opened["table_id"]), opened["round_id"]
        self.table(key)
        self._ensure_sweeper()
        async with self.runtime.lock(key):
            try:
                snap = self.store.join_round(guild_id, round_id, user.id, 0)
            except GameError as error:
                self.store.close_table(*key)
                self._release(key)
                self.tables.pop(key, None)
                refusal = discord_locale.invoker_t(interaction).error(error)
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
            thread = await partial.get_partial_message(message.id).create_thread(name=self._t(guild_id)("{game} · table {number}", game=terms.GAME_YUT, number=key[1]), auto_archive_duration=1440)
        except discord.HTTPException as error:
            logging.warning("Could not open the Yut thread (%s)", type(error).__name__)
            await self._thread_failed(interaction, key, isinstance(error, discord.Forbidden))
            return
        ready = None
        async with self.runtime.lock(key):  # Start waits for the thread
            row = self._row(key)
            if row["status"] == "open":
                self.store.set_table_thread(*key, thread.id)
                self.threads[thread.id] = key
                ready = self._seal(key, _Out(lobby=self.render_lobby(key, self.store.latest_round(*key), self._row(key))))
        if ready is None:  # closed while the thread was being made: nothing else to say there, just archive it
            try:
                await thread.edit(archived=True, locked=True)
            except Exception as error:
                logging.warning("Could not archive the unused Yut thread (%s)", type(error).__name__)
        await self._publish(key, ready)
        if note:
            await self._say(interaction, note)

    async def _thread_failed(self, interaction, key, forbidden: bool) -> None:
        t, ct = discord_locale.invoker_t(interaction), self._t(key[0])
        async with self.runtime.lock(key):
            refunded, refunded_say = self._refunded(key, t=ct), self._refunded(key, t=t)
            row = self._row(key)
            if row is not None and row["status"] == "open":
                self.store.close_table(*key)
            self._forget(key)
            out = self._seal(key, _Out(lobby=(ct("Table closed: I couldn't open a thread here.") + refunded, None), closed=True))
        await self._publish(key, out)
        self.tables.pop(key, None)
        self.ended.discard(key)
        await self._say(interaction, (t(THREAD_FORBIDDEN) if forbidden else t(THREAD_FAILED)) + refunded_say)

    # --- buttons ----------------------------------------------------------------------------

    def _help_text(self, guild_id, table_id, round_id, *, t=None) -> str:
        t = t or i18n.english()
        rules = None
        if guild_id is not None:
            row = self._row((guild_id, table_id))
            if row is not None and row["game"] != "yut":
                return STALE
            try:
                snap = self.store.round_snapshot(guild_id, round_id)
                rules = snap["rules"] if snap["table_id"] == table_id else None
            except GameError:
                pass
            rules = rules or self.store.game_rules(guild_id, "yut")
        return "\n".join(t.render(p) for p in yut.how_to_play_phrases(rules))

    async def press(self, interaction: discord.Interaction, action: str, table_id: int, round_id: int, moves: int) -> None:
        if action == "help":  # a guide, not a move: no state, no seat needed, works on old messages
            await self._say(interaction, self._help_text(interaction.guild_id, table_id, round_id, t=discord_locale.invoker_t(interaction)))
            return
        guild_id = interaction.guild_id
        if guild_id is None or interaction.user.bot:
            await self._say(interaction, STALE)
            return
        key = (guild_id, table_id)
        if action in MOVE_ACTIONS:
            await self._show_private(interaction, key, round_id, moves, action)
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
            if action == "start":
                return self._start(key, row, snap, user_id)
            if action == "throw":
                return self._play(key, user_id, snap, moves, "throw")
            return _Out(STALE)
        except ConflictError:
            return _Out(YUT_MOVED_ON)
        except GameError as error:
            return _Out(error)

    def _touch_lobby(self, key, round_id) -> None:
        """A join restarts the idle clock of the lobby."""
        self.table(key).deadlines["lobby"] = self.clock() + self.LOBBY_SECONDS
        self.arm(key, "lobby", self.LOBBY_SECONDS, round_id)

    def _join(self, key, row, snap, user_id) -> _Out:
        snap = self.store.join_round(key[0], snap["round_id"], user_id, 0)
        self._touch_lobby(key, snap["round_id"])
        ante, name = (snap["rules"] or {}).get("ante", 0), self.store.currency_name(key[0])
        say = Phrase(JOINED_ANTE, ante=ante, currency=name) if ante else JOINED
        return self._seal(key, _Out(say, lobby=self.render_lobby(key, snap, row)))

    def _leave(self, key, row, snap, user_id) -> _Out:
        snap = self.store.leave_round(key[0], snap["round_id"], user_id)
        if not snap["seats"]:
            self.store.close_table(*key)
            self._forget(key)
            self.arm(key, "archive", self.ARCHIVE_SECONDS)
            return self._seal(key, _Out(LEFT_CLOSED, lobby=(self._t(key[0])("Table closed: everyone left."), None), closed=True))
        ante = (snap["rules"] or {}).get("ante", 0)
        return self._seal(key, _Out(LEFT_REFUNDED if ante else LEFT_TABLE, lobby=self.render_lobby(key, snap, row)))

    def _start(self, key, row, snap, user_id) -> _Out:
        if not self._is_host(row, snap, user_id):
            return _Out(ONLY_HOST_START)
        if row["thread_id"] is None:
            return _Out(THREAD_NOT_READY)
        if len(snap["seats"]) < yut.MIN_SEATS:
            return _Out(Phrase(NEEDS_SEATS, count=yut.MIN_SEATS))
        snap = self.store.deal(key[0], snap["round_id"])
        self._after(key, snap)
        out = _Out(start=True, turn=self.render_turn(key, snap))
        out.lobby = (self._t(key[0])("In play — see the thread."), self._in_play_view(key, snap))
        return self._seal(key, out)

    # --- throws, moves and forfeits ---------------------------------------------------------

    def _play(self, key, user_id, snap, moves, move) -> _Out:
        """A member's throw or move from a button or the picker, under the table lock (``moves``: the count the message showed; None skips the check)."""
        seat = self.store.member_seat_index(key[0], snap["round_id"], user_id)
        if seat is None:
            return _Out(NOT_SEATED)
        if snap["status"] == "joining":
            return _Out(NOT_STARTED)
        if snap["status"] != "playing":
            return _Out(GAME_OVER)
        if not snap["view"]["active"][seat]:
            return _Out(OUT_OF_GAME)
        if move != "forfeit":
            if moves != snap["moves"]:
                return _Out(YUT_MOVED_ON)
            if snap["view"]["turn"] != seat:
                return _Out(NOT_YOUR_TURN)
            if move not in snap["legal"].get(seat, ()):
                return _Out(NOT_A_THROW if move == "throw" else NOT_A_MOVE)
        try:
            return self._apply(key, snap, seat, move, "member")
        except ConflictError:
            return _Out(YUT_MOVED_ON)
        except GameError as error:
            return _Out(error)

    async def _show_private(self, interaction, key, round_id, moves, action) -> None:
        """The Move picker or the Forfeit confirmation: a private message to the presser only."""
        async with self.runtime.lock(key):
            t = discord_locale.invoker_t(interaction)
            text, view = self._private(key, round_id, moves, action, interaction.user.id, t=t)
        kwargs = {"view": view} if view is not None else {}
        await self._say(interaction, text, **kwargs)

    def _private(self, key, round_id, moves, action, user_id, *, t):
        """(private text, view or None): the picker only for the seat on turn in the move phase at the message's move count."""
        loaded, refusal = self._load(key, round_id)
        if refusal is not None:
            return refusal.say, None
        _, snap = loaded
        seat = self.store.member_seat_index(key[0], round_id, user_id)
        if seat is None:
            return NOT_SEATED, None
        if snap["status"] == "joining":
            return NOT_STARTED, None
        if snap["status"] != "playing":
            return GAME_OVER, None
        view = snap["view"]
        if not view["active"][seat]:
            return OUT_OF_GAME, None
        if action == "forfeit":
            return FORFEIT_ASK, _ForfeitView(key[1], round_id, t=t)
        if snap["moves"] != moves:
            return YUT_MOVED_ON, None
        if view["turn"] != seat:
            return NOT_YOUR_TURN, None
        options = self._move_options(snap, seat, t)
        if not options:
            return NOTHING_TO_MOVE, None
        return t("Choose a banked throw and a piece to move."), _Picker(key[1], round_id, snap["moves"], options, t=t)

    @staticmethod
    def _move_options(snap, seat, t) -> list:
        """One option per distinct move: pieces on the same point with the same trail move alike, so only the first of them is offered."""
        view, seen, options = snap["view"], set(), []
        for move in snap["legal"].get(seat, ()):
            if not move.startswith("move:"):
                continue
            _, piece, name = move.split(":")
            piece = int(piece)
            same = (view["pieces"][seat][piece], view["trails"][seat][piece], name)
            if same in seen:
                continue
            seen.add(same)
            where = view["pieces"][seat][piece]
            options.append(discord.SelectOption(label=t("Piece {piece} · {throw}", piece=piece + 1, throw=terms.YUT_THROW[name]), value=move,
                                                description=t("At the start") if where is None else t("At {point}", point=where)))
        return options

    async def play_from_picker(self, interaction: discord.Interaction, table_id: int, round_id: int, moves: int | None, move: str) -> None:
        guild_id = interaction.guild_id
        if guild_id is None or interaction.user.bot:
            await self.answer(interaction, STALE)
            return
        key = (guild_id, table_id)
        async with self.runtime.lock(key):
            out = self._picker_play(key, interaction.user.id, round_id, moves, move)
        await self.answer(interaction, out.say, done=out.ok)  # answer first, so a slow edit cannot make it miss Discord's 3 seconds
        await self._publish(key, out)
        if out.say == STALE:
            await self._reap(key)

    def _picker_play(self, key, user_id, round_id, moves, move) -> _Out:
        loaded, refusal = self._load(key, round_id)
        if refusal is not None:
            return refusal
        _, snap = loaded
        if not isinstance(move, str) or not (move == "forfeit" or move.startswith("move:")):
            return _Out(NOT_A_MOVE)  # a throw comes from its button
        out = self._play(key, user_id, snap, moves, move)
        if out.key is None:  # refused: nothing was applied
            return out
        if move == "forfeit":
            out.say = YOU_FORFEIT
        else:
            _, piece, name = move.split(":")
            out.say = Phrase(YOU_MOVE, piece=int(piece) + 1, throw=terms.YUT_THROW[name])
        out.ok = True
        return out

    async def hand_command(self, interaction: discord.Interaction) -> None:
        """/game hand in a Yut thread: there is nothing hidden to show."""
        guild_id = interaction.guild_id
        table = self.store.table_for_thread(guild_id, interaction.channel.id) if guild_id and interaction.channel else None
        await self._say(interaction, NO_TABLE if table is None else NO_HIDDEN)

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
            t = self._t(guild_id)
            out = self._seal(key, _Out(lobby=(t("Table closed: nobody started the game.") + self._refunded(key, t=t), None), closed=True))
        await self._publish(key, out)

    def _missed_previous_turn(self, guild_id, round_id, seat) -> bool:
        """Did the seat's previous turn contain a timeout? A turn is a seat's run of consecutive moves (another seat's forfeit does not split it);
        the run in progress, if the seat is in the middle of one, is not the previous turn."""
        rows = self.store.all("SELECT seat_index,actor FROM game_moves WHERE guild_id=? AND round_id=? AND move<>'forfeit' ORDER BY id", (guild_id, round_id))
        runs: list[tuple[int, bool]] = []
        for row in rows:
            if runs and runs[-1][0] == row["seat_index"]:
                runs[-1] = (row["seat_index"], runs[-1][1] or row["actor"] == "timeout")
            else:
                runs.append((row["seat_index"], row["actor"] == "timeout"))
        mine = [missed for who, missed in runs if who == seat]
        if runs and runs[-1][0] == seat:
            mine = mine[:-1]  # the turn in progress
        return bool(mine) and mine[-1]

    async def on_turn_timer(self, guild_id, table_id, round_id, moves) -> None:
        """A member's decision ran out: the throw, or the first legal move, is made for them (every remaining decision of that turn too). A seat that
        times out in a turn right after missing its previous turn forfeits (two missed turns in a row); a turn without a timeout resets."""
        key, out = (guild_id, table_id), None
        async with self.runtime.lock(key):
            snap = self._timer_snap(key, round_id, moves)
            if snap is None:
                return
            seat = snap["view"]["turn"]
            if seat is None or snap["seats"][seat]["kind"] != "member":
                return
            t = self._t(guild_id)
            who = self._who(snap, seat, t=t)
            if self._missed_previous_turn(guild_id, round_id, seat):
                tried = [("forfeit", t("{who} missed two turns in a row.", who=who))]
            else:
                auto = next((m for m in snap["legal"].get(seat, ()) if m != "forfeit"), None)
                note = t("{who} ran out of time and a throw was made for them.", who=who) if auto == "throw" else t("{who} ran out of time and a move was made for them.", who=who)
                tried = [*([(auto, note)] if auto else []), ("forfeit", None)]  # a refused auto move falls back to forfeit, so the table is never stuck
            for move, note in tried:
                try:
                    out = self._apply(key, snap, seat, move, "timeout", note=note)
                    break
                except ConflictError:
                    return
                except GameError as error:
                    logging.info("Yut turn timeout refused (%s)", type(error).__name__)
            if out is None:
                return
        await self._publish(key, out)

    # --- table talk (the cast watching, FEAT-55) --------------------------------------------

    def _talk_budget(self, guild_id) -> int:
        """How many model calls the table-talk daily limit still allows (0 when character table talk is off). Failed attempts count."""
        if not self.store.game_character_talk(guild_id):
            return 0
        return self.talk.remaining(guild_id)

    def _talk_key(self, guild_id, thread_id):
        key = self.threads.get(thread_id) or self.banter.get(thread_id)
        return key if key is not None and key[0] == guild_id else None

    def _talk_names(self, guild_id, snap) -> list:
        """One prompt-safe name per seat (members by their guild display name, Player N when unknown); no mentions, no ids, no markdown."""
        guild, names, players = self.bot.get_guild(guild_id), [], 0
        for seat in snap["seats"]:
            players += 1
            member = guild.get_member(seat["ref_id"]) if guild is not None else None
            name = getattr(member, "display_name", None) or f"Player {players}"
            names.append(game_chooser.clean_name(name) or f"Player {len(names) + 1}")
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
        return {"table_id": key[1], "round_id": snap["round_id"], "status": snap["status"], "seated_character_ids": []}

    def table_block(self, guild_id, thread_id, character_id) -> str | None:
        """Characters do not sit at Yut tables yet (FEAT-60): nothing to tell a seated character."""
        return None

    def public_block(self, guild_id, key) -> str | None:
        """FEAT-55: the third-person public view of the table, for anyone not seated (everything is public in Yut)."""
        try:
            snap = self.store.latest_round(*key)
        except GameError:
            return None
        names, view = self._talk_names(key[0], snap), snap["view"]
        lines = ["A table of Yut (a Korean board game: players throw four sticks and move their pieces around the board; the first to bring every piece home wins)."]
        if not view:
            lines.append("The game has not started yet. Seated: " + ", ".join(names) + ".")
            return "\n".join(lines)
        if view["finished"]:
            lines.append(f"The game is over. {names[view['winner']]} won.")
        elif view["turn"] is not None:
            lines.append(f"On turn: {names[view['turn']]} ({'to throw' if view['phase'] == 'throw' else 'to move'}).")
        for i, row in enumerate(view["pieces"]):
            lines.append(f"- {names[i]}: {sum(p not in (None, 'done') for p in row)} on the board, {sum(p is None for p in row)} at the start, {sum(p == 'done' for p in row)} home"
                         + ("" if view["active"][i] else " (out of the game)"))
        recent = []
        for item in view["history"][-6:]:
            who = names[item["seat"]]
            if item["kind"] == "throw":
                recent.append(f"{who} threw {item['name']}")
            elif item["kind"] == "move":
                recent.append(f"{who} moved a piece with {item['throw']}" + (" and captured" if item["captured"] else "") + (" and brought it home" if item["end"] == "done" else ""))
            elif item["kind"] == "forfeit":
                recent.append(f"{who} left the game")
        if recent:
            lines.append("Recent play: " + "; ".join(recent) + ".")
        return "\n".join(lines)

    # --- archive ----------------------------------------------------------------------------

    async def on_archive_timer(self, guild_id, table_id) -> None:
        self.ended.discard((guild_id, table_id))
        self.tables.pop((guild_id, table_id), None)
        self.banter = {thread: k for thread, k in self.banter.items() if k != (guild_id, table_id)}
        row = self._row((guild_id, table_id))
        if row is None or row["status"] == "open":
            return
        await self._archive_thread(row["thread_id"])

    async def _archive_thread(self, thread_id) -> None:
        if thread_id is None:
            return
        try:
            thread = await self._channel(thread_id)
            await thread.edit(archived=True, locked=True)
        except Exception as error:
            logging.warning("Could not archive the Yut thread (%s)", type(error).__name__)

    # --- thread chat ------------------------------------------------------------------------

    async def note_thread_message(self, message) -> None:
        """Counts a member's message in a table thread; enough of them bury the turn message, which then moves to the bottom."""
        if message.author.bot or not isinstance(message.channel, discord.Thread):
            return
        key = self.threads.get(message.channel.id)
        if key is not None and message.guild is not None and message.guild.id == key[0]:
            self.table(key).talk += 1

    async def on_thread_message(self, message) -> None:
        """Nothing in a Yut thread is read as a move yet (moves are buttons); the hook exists so the bot treats every table game alike."""
        return None

    # --- restart ----------------------------------------------------------------------------

    async def recover(self) -> None:
        """Refunds and closes every Yut table left open by a previous run, then tells the lobby and the thread (best effort)."""
        store, refunded, closed, failed = self.store, 0, 0, 0
        rounds = [r for r in store.unfinished_rounds() if r.get("game") == "yut"]
        tables = {(r["guild_id"], r["table_id"]): (r["channel_id"], r["message_id"], r["thread_id"]) for r in rounds}
        for row in store.all("SELECT id,guild_id,channel_id,message_id,thread_id FROM game_tables WHERE status='open' AND game='yut'"):
            tables.setdefault((row["guild_id"], row["id"]), (row["channel_id"], row["message_id"], row["thread_id"]))
        # tables that closed just before the restart: their archive timer never fired (closed_at is updated_at; no schema change)
        earlier = [r["thread_id"] for r in store.all("SELECT thread_id FROM game_tables WHERE game='yut' AND status='closed' AND thread_id IS NOT NULL AND updated_at>=?", (time.time() - self.ARCHIVE_SECONDS,))]
        notes = {(r["guild_id"], r["table_id"]): self._refunded((r["guild_id"], r["table_id"]), past=True, t=self._t(r["guild_id"])) for r in rounds}
        for r in rounds:
            try:
                store.cancel_round(r["guild_id"], r["round_id"], "bot restarted")
                refunded += 1
            except Exception:
                failed += 1
                logging.exception("Could not refund Yut round %s on restart", r["round_id"])
        for (guild_id, table_id), (channel_id, message_id, thread_id) in tables.items():
            try:
                store.close_table(guild_id, table_id)
                closed += 1
            except Exception:
                failed += 1
                logging.exception("Could not close Yut table %s on restart", table_id)
                continue
            self._release((guild_id, table_id))
            self.tables.pop((guild_id, table_id), None)
            self.banter = {thread: k for thread, k in self.banter.items() if k != (guild_id, table_id)}
            text = self._t(guild_id)("This table closed when the bot restarted.") + notes.get((guild_id, table_id), "")
            await self._edit(channel_id, message_id, content=text, view=None)
            await self._post(thread_id, text)
            await self._archive_thread(thread_id)  # the old process's archive timer died with it
        for thread_id in earlier:
            await self._archive_thread(thread_id)
        if refunded or closed or failed:
            logging.info("Yut recovery: %d rounds refunded, %d tables closed, %d failures", refunded, closed, failed)
