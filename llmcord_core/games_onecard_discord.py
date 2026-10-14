"""One Card on Discord (FEAT-51, D40): `/game start game:One Card`, lobby, table thread, hand picker, turns, character seats, table talk, timers, settlement, recovery.

Rules, money and persistence live in ``games.onecard`` and the store. The machinery mirrors ``games_doubt_discord``: every table runs
under its own lock (store work only; no Discord call is made under it), a handler decides under the lock and builds an ``_Out`` that is
delivered afterwards one table at a time, and timers re-check the stored state under the lock so a stale timer does nothing.
Characters (favorites) play by the Play style policy or, with Character moves on "model", by a structured model call on their own view
made outside the lock; any failure falls back to the policy.
Chat declarations and call-outs (FEAT-52, D40): ``on_thread_message`` reads a seated member's message with ``spoken.onecard.read_onecard``
(always on while a round is playing; the game's own switch is the consent) and applies a sure reading through the normal step. A declaration
at two cards is held for the seat's next play (``DECLARE_WINDOW``). Characters declare after playing down to one card (they may forget) and
call out exposed members after ``CALLOUT_SECONDS``, speaking through ``bot.run_table_event``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import discord

from . import game_chooser
from .admin_store import ConflictError, GameError
from .game_talk import TalkBudget
from .games import onecard, playstyle
from .games_discord import NO_CHARACTER_SEATS, NO_MENTIONS, NOT_SEATED, NOT_YOUR_TURN, OFF_HERE, STALE, GameTables, _sign
from .games_doubt_discord import MOVED_ON, NO_TABLE, THREAD_FAILED, THREAD_FORBIDDEN, THREAD_NOT_READY
from .spoken.onecard import read_onecard
from .table_runtime import TableRuntime
from llmcord_core import i18n
from llmcord_core.i18n import discord_locale

PREFIX = "llmcord:onecard"
ONECARD_OFF = "One Card is off on this server."
NOT_A_MOVE = "Pick a card to play, or draw."
SUIT_NAMES = {"♠": "Spades", "♥": "Hearts", "♦": "Diamonds", "♣": "Clubs"}
_STALE = object()  # the table moved on while a decision was made: drop it
BIG_PILE = 5  # drawing an attack's pile of this many cards or more is a key moment
MAX_FORGET = 0.30  # the most a character ever forgets to declare (scaled down by its Play style)
REACTION = "✅"


def _cards(n: int) -> str:
    return f"{n} card" + ("" if n == 1 else "s")


def _suit_text(view) -> str:
    return view["suit"] or "any"


class OneCardButton(discord.ui.DynamicItem[discord.ui.Button], template=PREFIX + r":(?P<action>[a-z_]+):(?P<table>\d+):(?P<round>\d+):(?P<moves>\d+)"):
    """One button; its custom_id carries the table, the round and the move count the message showed. Never a seat."""

    def __init__(self, action: str, table_id: int, round_id: int, moves: int, label: str = "", style=discord.ButtonStyle.secondary, disabled: bool = False):
        super().__init__(discord.ui.Button(label=label, style=style, disabled=disabled, custom_id=f"{PREFIX}:{action}:{table_id}:{round_id}:{moves}"))
        self.action, self.table_id, self.round_id, self.moves = action, table_id, round_id, moves

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["action"], int(match["table"]), int(match["round"]), int(match["moves"]), item.label or "", item.style, item.disabled)

    async def callback(self, interaction: discord.Interaction):
        await interaction.client.onecard.press(interaction, self.action, self.table_id, self.round_id, self.moves)


class _CardSelect(discord.ui.Select):
    def __init__(self, options: list):
        super().__init__(placeholder="Card to play", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        picker, card = self.view, self.values[0]
        if card.startswith("7"):  # a 7 names the next suit: a second private step
            await interaction.client.onecard.ask_suit(interaction, _SuitView(picker.table_id, picker.round_id, picker.moves, card))
            return
        await interaction.client.onecard.play_from_picker(interaction, picker.table_id, picker.round_id, picker.moves, f"play:{card}")


class _Picker(discord.ui.View):
    """The private hand picker: one option per distinct playable card (25 to a select) and a Draw button."""

    def __init__(self, table_id: int, round_id: int, moves: int, cards: list, pending: int, *, t=None):
        super().__init__(timeout=600)
        t = t or i18n.english()
        self.table_id, self.round_id, self.moves = table_id, round_id, moves
        options = [discord.SelectOption(label=self._label(card), value=card) for card in cards]
        for start in range(0, len(options), 25):
            self.add_item(_CardSelect(options[start:start + 25]))
        draw = discord.ui.Button(label=t("Draw {count}", count=pending) if pending else t("Draw"), style=discord.ButtonStyle.danger if pending else discord.ButtonStyle.secondary)
        draw.callback = self.draw
        self.add_item(draw)

    @staticmethod
    def _label(card: str) -> str:
        return {"JKb": "Black joker (JKb)", "JKc": "Color joker (JKc)"}.get(card, card + (" (then name a suit)" if card.startswith("7") else ""))

    async def draw(self, interaction: discord.Interaction):
        await interaction.client.onecard.play_from_picker(interaction, self.table_id, self.round_id, self.moves, "draw")


class _SuitView(discord.ui.View):
    """The second step of a 7: the suit the next play must follow."""

    def __init__(self, table_id: int, round_id: int, moves: int, card: str):
        super().__init__(timeout=600)
        self.table_id, self.round_id, self.moves, self.card = table_id, round_id, moves, card
        for suit, name in SUIT_NAMES.items():
            item = discord.ui.Button(label=f"{suit} {name}", style=discord.ButtonStyle.primary)
            item.callback = self._pick(suit)
            self.add_item(item)

    def _pick(self, suit: str):
        async def callback(interaction: discord.Interaction):
            await interaction.client.onecard.play_from_picker(interaction, self.table_id, self.round_id, self.moves, f"play:{self.card}={suit}")
        return callback


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
    react: list = field(default_factory=list)  # member messages that got applied: one reaction each
    line: tuple | None = None  # (character, text): the model's table line, posted by the character's webhook after the move lines
    speak: tuple | None = None  # (character id, line): said through the table talk path as that character


class OneCardTables:
    LOBBY_SECONDS = 300
    CHARACTER_SECONDS = 2
    ARCHIVE_SECONDS = 600
    BOARD_DEBOUNCE_SECONDS = 1
    RETRY_SECONDS = 5  # read once at construction (TableRuntime.retry_seconds); tests set runtime.retry_seconds
    BUMP_AFTER = 8
    MODEL_SECONDS = 20
    POST_SECONDS = 10
    EVENT_COOLDOWN_SECONDS = 20
    SWEEP_SECONDS = 30
    CALLOUT_SECONDS = 3
    DECLARE_WINDOW = 10

    def __init__(self, bot, clock=time.time):
        self.bot, self.clock = bot, clock
        self.runtime = TableRuntime(label="One Card", retry_seconds=self.RETRY_SECONDS)
        self.tables: dict[tuple[int, int], _Table] = {}
        self.threads: dict[int, tuple[int, int]] = {}  # thread id -> (guild, table) of the tables being played
        self.banter: dict[int, tuple[int, int]] = {}  # thread id -> (guild, table) of finished games, until the archive timer
        self.event_at: dict[tuple[int, int], float] = {}  # table -> clock time of its last event turn
        self.event_tasks: set[asyncio.Task] = set()
        self.dirty: set[tuple[int, int]] = set()
        self.held: dict[tuple[int, int], dict] = {}  # table -> {seat: (clock time, round id, message)} of declarations made at two cards
        self.board_tasks: dict[tuple[int, int], asyncio.Task] = {}
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
        hook = {"lobby": self.on_lobby_timer, "turn": self.on_turn_timer, "character": self.on_character_timer, "callout": self.on_callout_timer, "archive": self.on_archive_timer}[kind]
        self.runtime.arm(key, kind, delay, self._run, kind, hook, key, args, name=f"onecard-{kind}-{key[1]}")

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
        for kind in ("lobby", "turn", "character", "callout"):
            self.runtime.disarm(key, kind)
        self.held.pop(key, None)
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
            self.runtime.ensure_sweeper(interval=lambda: self.SWEEP_SECONDS, held=lambda: bool(self._known()), sweep=self.sweep, name="onecard-sweep")

    def _known(self) -> set:
        return {*self.tables, *self.threads.values(), *(k[0] for k in self.runtime.timers if k[1] != "archive")}

    async def sweep(self) -> None:
        """Cleans up tables closed in the store by someone else (dashboard switch, channel removal) while this object still holds them."""
        for key in self._known():
            await self._reap(key)

    def _open(self, row) -> bool:
        return row is not None and row["game"] == "onecard" and row["status"] == "open"

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
        if row is None or row["game"] != "onecard":
            self._release(key)
            self.tables.pop(key, None)
            return None
        guild_id = key[0]
        if not self.store.game_enabled(guild_id, "onecard"):
            reason = "One Card was turned off on this server."
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

    def _after(self, key, snap, keep: bool = False) -> None:
        """Arms the timers the new state needs. Call under the table lock. ``keep``: the move did not change whose turn it is (a declaration or
        call-out), so the running turn or character timer continues with the time it had left instead of starting over."""
        table, view = self.table(key), snap["view"]
        self.runtime.disarm(key, "lobby")
        if snap["status"] != "playing" or not view or view["turn"] is None:
            for kind in ("turn", "character", "callout"):
                self.runtime.disarm(key, kind)
            return
        exposed = any(snap["seats"][i]["kind"] == "member" for i in view["exposed"])
        if exposed and any(s["kind"] == "character" and view["active"][s["index"]] for s in snap["seats"]):
            self.arm(key, "callout", self.CALLOUT_SECONDS, snap["round_id"], snap["moves"])
        else:
            self.runtime.disarm(key, "callout")
        if snap["seats"][view["turn"]]["kind"] == "member":
            delay = (snap["rules"] or {}).get("turn_timeout", 60)
            if keep and "turn" in table.deadlines:
                delay = max(0.0, table.deadlines["turn"] - self.clock())
            else:
                table.deadlines["turn"] = self.clock() + delay
            self.arm(key, "turn", delay, snap["round_id"], snap["moves"])
            self.runtime.disarm(key, "character")
        else:
            table.deadlines.pop("turn", None)
            self.runtime.disarm(key, "turn")
            delay = self.CHARACTER_SECONDS
            if keep and "character" in table.deadlines:
                delay = max(0.0, table.deadlines["character"] - self.clock())
            else:
                table.deadlines["character"] = self.clock() + delay
            self.arm(key, "character", delay, snap["round_id"], snap["moves"])

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
            logging.warning("Could not edit One Card message %s in channel %s: %s", message_id, channel_id, type(error).__name__)
        except Exception:
            logging.exception("Could not edit One Card message %s in channel %s", message_id, channel_id)
        return None

    async def _delete(self, channel_id, message_id) -> bool:
        if channel_id is None or message_id is None:
            return False
        try:
            channel = await self._channel(channel_id)
            await channel.get_partial_message(message_id).delete()
            return True
        except discord.HTTPException as error:
            logging.info("Could not delete One Card message %s (%s)", message_id, type(error).__name__)
        except Exception:
            logging.exception("Could not delete One Card message %s", message_id)
        return False

    async def _post(self, channel_id, content, view=None):
        if channel_id is None:
            return None
        try:
            channel = await self._channel(channel_id)
            return await channel.send(content, view=view, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning("Could not post to One Card thread %s: %s", channel_id, type(error).__name__)
        except Exception:
            logging.exception("Could not post to One Card thread %s", channel_id)
        return None

    async def _say(self, interaction, text: str, **kwargs) -> None:
        """An ephemeral reply: the first response, or a followup once the interaction was acknowledged."""
        try:
            if interaction.response.is_done():
                await interaction.followup.send(text, ephemeral=True, allowed_mentions=NO_MENTIONS, **kwargs)
            else:
                await interaction.response.send_message(text, ephemeral=True, allowed_mentions=NO_MENTIONS, **kwargs)
        except discord.HTTPException as error:
            logging.warning("One Card reply failed: %s", type(error).__name__)

    async def answer(self, interaction, text: str, done: bool = False) -> None:
        """A reply to the picker: ``done`` replaces the picker message, anything else is a new private message (the picker stays)."""
        if done and not interaction.response.is_done() and getattr(interaction, "message", None) is not None:
            try:
                await interaction.response.edit_message(content=text, view=None, allowed_mentions=NO_MENTIONS)
                return
            except discord.HTTPException as error:
                logging.warning("One Card picker edit failed: %s", type(error).__name__)
        await self._say(interaction, text)

    async def ask_suit(self, interaction, view) -> None:
        card = view.card
        await self._say(interaction, f"Name the suit for your {card}. The next play must follow it.", view=view)

    async def _ack(self, interaction) -> None:
        try:
            await interaction.response.defer()
        except discord.HTTPException as error:
            logging.warning("One Card could not acknowledge the press: %s", type(error).__name__)

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
        """The host shown in the lobby: the opener while seated, else the first seated member."""
        return user_id == self._host(row, snap)

    def _row(self, key):
        return self.store.one("SELECT * FROM game_tables WHERE id=? AND guild_id=?", (key[1], key[0]))

    # --- rendering --------------------------------------------------------------------------

    @staticmethod
    def _who(snap, index) -> str:
        return GameTables._who({**snap["seats"][index], "brought_by": None})

    def _stamp(self, key, kind) -> str:
        deadline = self.table(key).deadlines.get(kind)
        return f"<t:{int(deadline)}:R>" if deadline else ""

    def _t(self, guild_id):
        """The channel's translator for a table message: resolved at each render, never kept."""
        get_guild = getattr(self.bot, "get_guild", None)
        return discord_locale.channel_t(get_guild(guild_id) if get_guild else None)

    def render_lobby(self, key, snap, row, *, t=None) -> tuple[str, discord.ui.View]:
        t = t or self._t(key[0])
        rules = snap["rules"] or onecard.normalize_rules(None)
        name, pot, ante = self.store.currency_name(key[0]), sum(s["stake"] for s in snap["seats"]), rules["ante"]
        lines = [f"**One Card** · table {key[1]}", f"Ante: {ante:,} {name} · Pot: {pot:,} {name}" if ante else f"No ante · Pot: {pot:,} {name}",
                 *([f"Host: <@{self._host(row, snap)}>"] if any(s["kind"] == "member" for s in snap["seats"]) else []), ""]
        for seat in snap["seats"]:
            tag = " " + t("(house)") if seat["kind"] == "character" and not seat["brought_by"] else ""
            lines.append(f"{seat['index'] + 1}. {GameTables._who(seat)}{tag}")
        stamp = self._stamp(key, "lobby")
        lines.append("")
        lines.append(f"Join with the button. The host starts the game with {onecard.MIN_SEATS} to {onecard.MAX_SEATS} seats"
                     + (f"; the table closes {stamp} if it hasn't started." if stamp else "."))
        lines.append("-# " + onecard.rules_text(rules))
        view, tid, rid = discord.ui.View(timeout=None), key[1], snap["round_id"]
        spec = [("join", "Join", discord.ButtonStyle.primary, False), ("leave", "Leave", discord.ButtonStyle.secondary, False),
                ("start", "Start", discord.ButtonStyle.success, len(snap["seats"]) < onecard.MIN_SEATS or row["thread_id"] is None),
                ("help", "How to play", discord.ButtonStyle.secondary, False)]
        for action, label, style, disabled in spec:
            view.add_item(OneCardButton(action, tid, rid, 0, label, style, disabled))
        return "\n".join(lines), view

    def _in_play_view(self, key, snap) -> discord.ui.View:
        view = discord.ui.View(timeout=None)
        view.add_item(OneCardButton("help", key[1], snap["round_id"], 0, "How to play"))
        return view

    def render_board(self, key, snap) -> str:
        view, rules, name = snap["view"], snap["rules"] or {}, self.store.currency_name(key[0])
        pot = sum(s["stake"] for s in snap["seats"])
        lines = [f"**One Card** · table {key[1]}" + (f" · pot {pot:,} {name} (ante {rules.get('ante', 0):,})" if pot else "")]
        for seat in snap["seats"]:
            i = seat["index"]
            marks = ""
            if not view["active"][i]:
                state = "out"
            else:
                state = _cards(view["counts"][i])
                if view["finished"] and view["winner"] == i:
                    marks = " · winner"
                else:
                    marks = (" · on turn" if view["turn"] == i else "") + (" · exposed" if i in view["exposed"] else "") \
                        + (" · declared" if view["declared"][i] else "")
            lines.append(f"{i + 1}. {GameTables._who(seat)} — {state}{marks}")
        if view["finished"]:
            lines.append("Game over.")
        else:
            lines.append(f"Top card: {onecard.card_text(view['top'])} · Suit: {_suit_text(view)} · {'clockwise' if view['direction'] > 0 else 'counter-clockwise'}")
            if view["pending"]:
                lines.append(f"Pending attack: {view['pending']}")
            lines.append(f"-# Draw pile {view['draw']} · discard pile {view['discard']}")
        lines.append("-# " + onecard.rules_text(snap["rules"]))
        lines.append(f"-# Seed: {snap['seed']} (hash {snap['seed_hash'][:12]})" if snap["seed"] else f"-# Seed hash: {snap['seed_hash'][:12]}")
        return "\n".join(lines)

    def _happened(self, snap) -> str:
        """The line for the last thing that happened (a play, a draw, an exit); hands are never named."""
        view = snap["view"]
        last = next((e for e in reversed(view["history"]) if e["kind"] in ("play", "draw", "forfeit", "bankrupt")), None)
        if last is None:
            return "The game has started."
        if last["kind"] == "play":
            text = f"{self._who(snap, last['seat'])} played {onecard.card_text(last['card'])}"
            return text + (f" and named {last['suit']}." if last["suit"] else ".")
        if last["kind"] == "draw":
            count = last["count"]
            taken = "a card" if count == 1 else f"{count} cards" if count else "nothing"
            return f"{self._who(snap, last['seat'])} drew {taken}."
        if last["kind"] == "forfeit":
            return f"{self._who(snap, last['seat'])} forfeited and is out."
        return f"{self._who(snap, last['seat'])} went bankrupt and is out."

    def render_turn(self, key, snap) -> tuple[str, discord.ui.View]:
        view, seats = snap["view"], snap["seats"]
        lines = [self._happened(snap), f"Top card: {onecard.card_text(view['top'])} · Suit: {_suit_text(view)}."]
        if view["pending"]:
            lines.append(f"Attack pending: draw {view['pending']} or defend.")
        turn = view["turn"]
        if turn is not None:
            stamp = self._stamp(key, "turn")
            lines.append(f"Next: {self._who(snap, turn)}" + (f" — {stamp}." if stamp and seats[turn]["kind"] == "member" else "."))
        buttons = discord.ui.View(timeout=None)
        buttons.add_item(OneCardButton("hand", key[1], snap["round_id"], snap["moves"], "My hand", discord.ButtonStyle.primary))
        return "\n".join(lines), buttons

    def _result(self, key, snap) -> tuple[str, str, str]:
        """(thread result, one-line lobby text, final turn text) for a settled round."""
        name, seats, win = self.store.currency_name(key[0]), snap["seats"], snap["view"]["winner"]
        pot, who = sum(s["stake"] for s in seats), self._who(snap, win)
        head = f"**{who} wins the game" + (f" and takes the pot of {pot:,} {name}.**" if pot else ".**")
        lines = [head]
        broke = {e["seat"] for e in snap["view"]["history"] if e["kind"] == "bankrupt"}
        for index, seat in enumerate(seats):
            net = (seat["payout"] or 0) - seat["stake"]
            note = " (bankrupt)" if index in broke else " (forfeited)" if seat["outcome"] == "forfeit" else ""
            lines.append(f"• {GameTables._who(seat)}: {_sign(net) or 'even'}{note}")
        if snap["seed"]:
            lines.append(f"-# Seed: {snap['seed']} (hash {snap['seed_hash'][:12]})")
        lobby = f"{who} won One Card" + (f" and took the pot of {pot:,} {name}" if pot else "") + ". Table closed."
        return "\n".join(lines), lobby, f"Game over — {who} won."

    # --- the common step machinery ----------------------------------------------------------

    def _seal(self, key, out: _Out) -> _Out:
        table = self.table(key)
        table.issued += 1
        out.key, out.seq = key, table.issued
        return out

    def _load(self, key, round_id):
        """((table row, snapshot), None) for an open One Card table and one of its rounds, else (None, refusal)."""
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
        out = _Out(advanced=True)
        keep = move == "declare" or move.startswith("call_out:")
        new = self._declared_after(key, new, seat, move, actor, out)
        if note:
            out.events.append(note)
        for event in new["view"]["history"][len(before["view"]["history"]):]:
            if event["kind"] == "forfeit":
                out.events.append(f"{self._who(before, event['seat'])} forfeits and is out of the game.")
            elif event["kind"] == "bankrupt":
                out.events.append(f"{self._who(before, event['seat'])} went bankrupt and is out of the game.")
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
            keep = keep and before["view"]["turn"] == new["view"]["turn"]
            out.advanced = not keep
            self._after(key, new, keep)
            out.turn = self.render_turn(key, new)
            out.board = True
            if self.clock() - self.event_at.get(key, float("-inf")) >= self.EVENT_COOLDOWN_SECONDS and self.talk_allowed(key[0]):
                text = self._moment(key, before, new)
                if text is not None and not (out.speak and text.endswith("is down to one card.")):
                    self.event_at[key] = self.clock()
                    out.talk = (text, False)
        return self._seal(key, out)

    def _declared_after(self, key, new, seat, move, actor, out):
        """After a move: a member's held declaration (made at two cards) that this play fulfils, and a character's declaration (or slip) at one
        card. Returns the newest snapshot. Any other move drops the held declaration."""
        held = self.held.pop(key, None)
        view = new["view"]
        if not move.startswith("play:") or new["status"] != "playing" or not view or view["counts"][seat] != 1 or not view["active"][seat] or view["declared"][seat]:
            return new
        declare = None
        if actor == "member":
            hold = (held or {}).get(seat)
            if hold and hold[1] == new["round_id"] and self.clock() - hold[0] <= self.DECLARE_WINDOW:
                declare = "member"
        elif actor == "character" and seat in view["exposed"]:
            declare = "character" if self._remembers(key, new, seat) else None
        if declare is None:
            return new
        try:
            after = self.store.play(key[0], new["round_id"], seat, "declare", declare, new["moves"])
        except (ConflictError, GameError) as error:
            logging.info("One Card declaration refused (%s)", type(error).__name__)
            return new
        if declare == "member":
            out.react.append(hold[2])
        elif self.talk_allowed(key[0]):
            out.speak = (new["seats"][seat]["ref_id"], f"{self._talk_names(key[0], new)[seat]} is down to one card and declares it: One card!")
        return after

    def _remembers(self, key, snap, seat) -> bool:
        """A character at one card declares unless a seeded roll says it forgot; the chance is at most MAX_FORGET, higher for foolish, unpredictable styles."""
        try:
            style = self.store.resolved_play_style(key[0], snap["seats"][seat]["ref_id"])
            chance = MAX_FORGET * (0.1 + 0.9 * (style.unit("unpredictability") + 1 - style.unit("judgment")) / 2)
            return playstyle.roll(self.store.policy_seed(key[0], snap["round_id"], f"{snap['moves']}/{seat}/declare"), "declare") >= min(chance, MAX_FORGET)
        except Exception as error:
            logging.warning("One Card declaration roll failed (%s)", type(error).__name__)
            return True

    def _moment(self, key, before, new) -> str | None:
        """The public line for a key moment of the move just made (an attack, a skip or reverse, a big pile drawn, a seat down to one card), else None.
        Names and the card played only: no hand, no markdown, no mention."""
        old, view = before["view"], new["view"]
        fresh = view["history"][len(old["history"]):]
        names = self._talk_names(key[0], new)
        play = next((e for e in fresh if e["kind"] == "play"), None)
        drew = next((e for e in fresh if e["kind"] == "draw"), None)
        if play is not None:
            card, who = onecard.card_text(play["card"]), names[play["seat"]]
            if view["pending"] and play["card"][0] in ("2", "A", "JK"):
                return f"{who} played {card}: an attack of {view['pending']} cards."
            if play["card"][0] == "J":
                return f"{who} played {card} and skipped a player."
            if play["card"][0] == "Q":
                return f"{who} played {card} and reversed the direction."
        if drew is not None and drew["count"] >= BIG_PILE:
            return f"{names[drew['seat']]} drew {drew['count']} cards."
        low = [i for i, n in enumerate(view["counts"]) if n == 1 and view["active"][i] and old["counts"][i] != 1]
        return f"{names[low[0]]} is down to one card." if low else None

    async def _publish(self, key, out: _Out | None) -> None:
        """After the table lock is released: the thread lines, then the edits (an edit older than the last one sent is dropped)."""
        if out is None:
            return
        await self._send(key, out)
        for message in out.react:
            try:
                await message.add_reaction(REACTION)
            except Exception as error:
                logging.info("Could not react to a One Card declaration (%s)", type(error).__name__)
        if out.line is not None:
            await self._post_line(key, *out.line)
        if out.speak is not None:
            await self._speak(key, *out.speak)

    async def _post_line(self, key, character, line) -> None:
        """Best effort, outside the table lock: the character says its line in the table thread through its webhook. Nothing is saved to scene history."""
        try:
            row = self._row(key)
            if row is None or row["thread_id"] is None:
                return
            channel = await self._channel(row["thread_id"])
            webhook = await self.bot._webhook(channel, character)
            slots = {r["slot_key"]: r for r in self.store.usable_avatars(key[0], character["id"])}
            avatar = (await self.bot.resolve_avatar(slots["neutral"]))["url"] if "neutral" in slots else None
            thread = {"thread": channel} if isinstance(channel, discord.Thread) else {}
            await asyncio.wait_for(webhook.send(line, **thread, username=character["name"], avatar_url=avatar, allowed_mentions=NO_MENTIONS), self.POST_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logging.warning("One Card character line not posted (%s)", type(error).__name__)

    async def _speak(self, key, character_id, text) -> None:
        """Best effort, outside the table lock: one line from this character through the table talk path."""
        try:
            row = self._row(key)
            if row is None or row["thread_id"] is None:
                return
            await self.bot.run_table_event(await self._channel(row["thread_id"]), text, speaker_hint=character_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("One Card character line failed (table %s)", key[1])

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

    def _talk_event(self, thread_id, out, guild_id=None) -> None:
        """Fire and forget: the characters at the table react to a public event. A move never waits for it."""
        if out.talk is None or thread_id is None or self.closed:
            return
        text = out.talk[0]
        out.talk = None

        async def run():
            try:
                thread = await self._channel(thread_id)
                await self.bot.run_table_event(thread, text)
                if guild_id is not None and self.bot.store.cast_watches_games(guild_id):
                    await self.bot.run_table_event(thread, text, spectators=True)  # FEAT-55: after the seated turn, not under it
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("One Card event turn failed (thread %s)", thread_id)

        task = asyncio.create_task(run(), name=f"onecard-event-{thread_id}")
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
            task = self.board_tasks[key] = asyncio.create_task(self._debounce(key), name=f"onecard-board-{key[1]}")
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
            await self._say(interaction, "Bots cannot play One Card.")
            return
        if not self.store.game_enabled(guild_id, "onecard"):
            await self._say(interaction, ONECARD_OFF)
            return
        if channel_id not in self.store.game_channels(guild_id):
            await self._say(interaction, OFF_HERE)
            return
        _, binding = self.bot.location(interaction.channel)
        space_id, note = (binding["space_id"] if binding else None), ""
        if favorites and not binding:
            note = NO_CHARACTER_SEATS
        try:
            opened = self.store.open_table(guild_id, channel_id, "onecard", user.id)
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
            thread = await partial.get_partial_message(message.id).create_thread(name=f"One Card · table {key[1]}", auto_archive_duration=1440)
        except discord.HTTPException as error:
            logging.warning("Could not open the One Card thread (%s)", type(error).__name__)
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
                logging.warning("Could not archive the unused One Card thread (%s)", type(error).__name__)
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
            row = self._row((guild_id, table_id))
            if row is not None and row["game"] != "onecard":
                return STALE
            try:
                snap = self.store.round_snapshot(guild_id, round_id)
                rules = snap["rules"] if snap["table_id"] == table_id else None
            except GameError:
                pass
            rules = rules or self.store.game_rules(guild_id, "onecard")
        return onecard.how_to_play(rules)

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
            out = self._press(interaction.user.id, key, action, round_id)
        await self._publish(key, out)
        if out.say == STALE:
            await self._reap(key)  # the table may have been closed from outside
        if out.say:
            await self._say(interaction, out.say)

    def _press(self, user_id, key, action, round_id) -> _Out:
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
            return _Out(STALE)
        except ConflictError:
            return _Out(MOVED_ON)
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
        ante = (snap["rules"] or {}).get("ante", 0)
        return self._seal(key, _Out("You left the table." + (" Your ante is refunded." if ante else ""), lobby=self.render_lobby(key, snap, row)))

    def _start(self, key, row, snap, user_id) -> _Out:
        if not self._is_host(row, snap, user_id):
            return _Out("Only the host can start the game.")
        if row["thread_id"] is None:
            return _Out(THREAD_NOT_READY)
        if len(snap["seats"]) < onecard.MIN_SEATS:
            return _Out(f"One Card needs at least {onecard.MIN_SEATS} seats to start.")
        snap = self.store.deal(key[0], snap["round_id"])
        self._after(key, snap)
        out = _Out(start=self.render_board(key, snap), turn=self.render_turn(key, snap))
        out.lobby = ("In play — see the thread.", self._in_play_view(key, snap))
        return self._seal(key, out)

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
        kwargs = {"view": _Picker(*picker, t=discord_locale.invoker_t(interaction))} if picker else {}
        try:
            if interaction.response.is_done():
                await interaction.followup.send(text, ephemeral=True, allowed_mentions=NO_MENTIONS, **kwargs)
            else:
                await interaction.response.send_message(text, ephemeral=True, allowed_mentions=NO_MENTIONS, **kwargs)
        except discord.HTTPException as error:
            logging.warning("One Card hand reply failed: %s", type(error).__name__)

    def _hand(self, key, round_id, user_id):
        """(private text, picker arguments or None): the presser's own hand only; the picker only on their turn."""
        guild_id = key[0]
        row = self._row(key)
        if row is None or row["game"] != "onecard":
            return STALE, None
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
        if not view.active[seat]:
            return "You are out of this game.", None
        lines = [f"**Your hand** ({_cards(len(view.hand))})", " ".join(onecard.card_text(c) for c in view.hand),
                 f"Top card: {onecard.card_text(view.top)} · Suit: {view.suit or 'any'}"]
        playable = onecard.playable_cards(view, snap["rules"])
        if view.pending and view.turn == seat:
            lines.append(f"Attack pending: draw {view.pending}" + (" or defend." if playable else "."))
        if view.turn == seat:
            lines.append("\nYour turn: " + ("choose a card to play, or draw." if playable else "nothing can be played, so draw."))
        elif view.turn is not None:
            lines.append(f"\nOn turn: {self._who(snap, view.turn)}.")
        return "\n".join(lines), ((key[1], snap["round_id"], snap["moves"], list(playable), view.pending) if view.turn == seat else None)

    async def play_from_picker(self, interaction: discord.Interaction, table_id: int, round_id: int, moves: int, move: str) -> None:
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
        seat = self.store.member_seat_index(key[0], round_id, user_id)
        if seat is None:
            return _Out(NOT_SEATED)
        if snap["status"] != "playing" or snap["moves"] != moves:
            return _Out(MOVED_ON)
        if snap["view"]["turn"] != seat:
            return _Out(NOT_YOUR_TURN)
        if not isinstance(move, str) or not (move == "draw" or move.startswith("play:")):
            return _Out(NOT_A_MOVE)  # declare and call_out come by chat; timeout and forfeit are the timers'
        try:
            out = self._apply(key, snap, seat, move, "member")
        except ConflictError:
            return _Out(MOVED_ON)
        except GameError as error:
            return _Out(str(error))
        if move == "draw":
            out.say = "You draw." if not snap["view"]["pending"] else f"You draw {snap['view']['pending']} cards."
        else:
            card, _, named = move[5:].partition("=")
            out.say = f"You play {card}" + (f" and name {named}." if named else ".")
        out.ok = True
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

    def _missed_turns(self, guild_id, round_id, seat) -> int:
        """Timed-out turns in a row at the end of the seat's own moves (a move of its own resets it)."""
        rows = self.store.all("SELECT actor FROM game_moves WHERE guild_id=? AND round_id=? AND seat_index=? AND (move IN ('draw','timeout') OR move LIKE 'play:%') ORDER BY id DESC",
                              (guild_id, round_id, seat))
        count = 0
        for row in rows:
            if row["actor"] != "timeout":
                break
            count += 1
        return count

    async def on_turn_timer(self, guild_id, table_id, round_id, moves) -> None:
        """A member's turn ran out: the engine's timeout move (draw and pass); the second missed turn in a row forfeits the seat."""
        key, out = (guild_id, table_id), None
        async with self.runtime.lock(key):
            snap = self._timer_snap(key, round_id, moves)
            if snap is None:
                return
            seat = snap["view"]["turn"]
            if seat is None or snap["seats"][seat]["kind"] != "member":
                return
            who = self._who(snap, seat)
            try:
                if self._missed_turns(guild_id, round_id, seat) >= 1:
                    out = self._apply(key, snap, seat, "forfeit", "timeout", note=f"{who} missed two turns in a row.")
                else:
                    out = self._apply(key, snap, seat, "timeout", "timeout", note=f"{who} ran out of time and had to draw.")
            except (ConflictError, GameError) as error:
                logging.info("One Card turn timeout refused (%s)", type(error).__name__)
                return
        await self._publish(key, out)

    async def on_callout_timer(self, guild_id, table_id, round_id, moves) -> None:
        """No human called out the exposed member in time: the first active character seat does, and says so (table talk must be allowed)."""
        key, out = (guild_id, table_id), None
        async with self.runtime.lock(key):
            snap = self._timer_snap(key, round_id, moves)
            if snap is None or not self.talk_allowed(guild_id):
                return
            view, seats = snap["view"], snap["seats"]
            target = next((i for i in view["exposed"] if seats[i]["kind"] == "member" and view["active"][i]), None)
            caller = next((s for s in seats if s["kind"] == "character" and view["active"][s["index"]]), None)
            if target is None or caller is None:
                return
            names = self._talk_names(guild_id, snap)
            try:
                out = self._apply(key, snap, caller["index"], f"call_out:{target}", "character")
            except (ConflictError, GameError) as error:
                logging.info("One Card character call-out refused (%s)", type(error).__name__)
                return
            out.speak = (caller["ref_id"], f"{names[target]} had one card left and did not say One card. {names[caller['index']]} calls it out.")
        await self._publish(key, out)

    # --- characters -------------------------------------------------------------------------

    def _talk_budget(self, guild_id) -> int:
        """How many model calls the table-talk daily limit still allows (0 when character table talk is off). Failed attempts count."""
        if not self.store.game_character_talk(guild_id):
            return 0
        return self.talk.remaining(guild_id)

    def _note_failure(self, guild_id) -> None:
        self.talk.note_failure(guild_id)  # one counter for every table game

    def _character(self, guild_id, seat):
        character = self.store.character_by_id(seat["ref_id"])
        return character if character is not None and character["guild_id"] == guild_id else None

    def _fallback_moves(self, guild_id, snap, seat, seen) -> list:
        """The seat's policy move, then a plain draw, then forfeit: the order the store is asked to accept them in."""
        tried = []
        try:
            style = self.store.character_play_style(guild_id, snap["seats"][seat]["ref_id"])
            tried.append((seat, onecard.policy_move(seen, snap["rules"], style, self.store.policy_seed(guild_id, snap["round_id"], f"{snap['moves']}/{seat}/character"))))
        except Exception as error:
            logging.warning("One Card policy move failed (%s)", type(error).__name__)
        return [*tried, (seat, "draw"), (seat, "forfeit")]

    def _apply_chain(self, key, snap, tried):
        """Under the table lock: the first move the store accepts (a refusal falls back to the next; the last resort frees the table).
        _STALE when the state moved on, None when nothing was accepted."""
        for seat, move in tried:
            try:
                return self._apply(key, snap, seat, move, "character" if move != "forfeit" else "timeout")
            except ConflictError:
                return _STALE
            except GameError as error:
                logging.warning("One Card character move refused (%s)", type(error).__name__)
        return None

    async def _apply_fresh(self, key, round_id, moves, tried):
        """Applies only if the table is still at the move count the decision was made for; else _STALE (dropped silently)."""
        async with self.runtime.lock(key):
            snap = self._timer_snap(key, round_id, moves)
            return _STALE if snap is None else self._apply_chain(key, snap, tried)

    async def on_character_timer(self, guild_id, table_id, round_id, moves) -> None:
        """The character on turn plays. With Character moves "model", character table talk on and the daily limit not reached, the move is a model
        call on the seat's own view made outside the table lock and applied only if the table is still at the same move count; the policy plays
        otherwise and after any failure."""
        key, out, plan = (guild_id, table_id), None, None
        async with self.runtime.lock(key):
            snap = self._timer_snap(key, round_id, moves)
            if snap is None:
                return
            turn = snap["view"]["turn"]
            if turn is None or snap["seats"][turn]["kind"] != "character":
                return
            seen = self.store.seat_view(guild_id, round_id, turn)
            budget = 0 if self.store.game_character_moves(guild_id) == "style" else self._talk_budget(guild_id)  # style only: the policy plays; table talk keeps its own budget
            if budget <= 0:
                out = self._apply_chain(key, snap, self._fallback_moves(guild_id, snap, turn, seen))
            else:
                plan = turn, self._character(guild_id, snap["seats"][turn]), seen, game_chooser.onecard_names(snap), onecard.legal_moves_for_view(seen, snap["rules"])
        if plan is not None:
            out = await self._decide(key, snap, plan)
        await self._publish(key, None if out is _STALE else out)

    async def _decide(self, key, snap, plan):
        guild_id, round_id, moves = key[0], snap["round_id"], snap["moves"]
        turn, character, seen, names, legal = plan
        tried, line, asked = [], "", False
        if character is not None:
            style = self.store.character_play_style(guild_id, character["id"])
            try:
                result = await asyncio.wait_for(game_chooser.ask_onecard(self.bot, guild_id, snap["channel_id"], character, seen, names, legal, style), self.MODEL_SECONDS)
                move = game_chooser.holdem_move(result, legal)
                if move is not None:
                    asked = True
                    tried.append((turn, move))
                    line = game_chooser.clean_line(result.get("line"))
            except asyncio.CancelledError:
                self._note_failure(guild_id)  # an attempt that was cut short still counts toward the limit
                raise
            except Exception as error:
                move = None
                logging.warning("One Card character move fell back to the policy (%s)", type(error).__name__)
            if move is None:
                self._note_failure(guild_id)
        tried += self._fallback_moves(guild_id, snap, turn, seen)
        out = await self._apply_fresh(key, round_id, moves, tried)
        if out is _STALE and asked:
            self._note_failure(guild_id)  # a good answer dropped because the table moved on still counts toward the limit
        elif asked and line and out is not None and out is not _STALE:
            out.line = character, line
        return out

    # --- table talk -------------------------------------------------------------------------

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
        return {"table_id": key[1], "round_id": snap["round_id"], "status": snap["status"],
                "seated_character_ids": [s["ref_id"] for s in snap["seats"] if s["kind"] == "character"]}

    def public_block(self, guild_id, key) -> str | None:
        """FEAT-55: the third-person public view of the table (hand sizes only), for anyone not seated."""
        try:
            snap = self.store.latest_round(*key)
        except GameError:
            return None
        return "\n".join(["A table of One Card (a card game: play a card that matches the top card's suit or rank, or draw; the first to play their last card wins).",
                          *self._public_lines(key, snap, None)])

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
        view = snap["view"]
        lines = ["You are sitting at a table of One Card (a card game: play a card that matches the top card's suit or rank, or draw; the first to play their last card wins).",
                 *self._public_lines(key, snap, mine["index"])]
        if view and snap["status"] == "playing" and view["active"][mine["index"]]:
            try:
                hand = self.store.seat_view(key[0], snap["round_id"], mine["index"]).hand or ()
            except GameError:
                hand = ()
            lines.append("Keep your hand secret unless you choose to reveal it.")
            lines.append("Your hand: " + (", ".join(onecard.card_text(c) for c in hand) or "(no cards)"))
        return "\n".join(lines)

    def _public_lines(self, key, snap, me) -> list:
        names, view = self._talk_names(key[0], snap), snap["view"]
        if not view:
            return ["The game has not started yet. Seated: " + ", ".join(names) + "."]

        def who(i):
            return names[i] + (" (you)" if i == me else "")
        lines = []
        if view["finished"]:
            lines.append(f"The game is over. {who(view['winner'])} won.")
        else:
            lines.append(f"Top card: {onecard.card_text(view['top'])}. Suit: {_suit_text(view)}. Direction: {'clockwise' if view['direction'] > 0 else 'counter-clockwise'}.")
            if view["pending"]:
                lines.append(f"Pending attack: {view['pending']} cards to draw.")
        for i, n in enumerate(view["counts"]):
            lines.append(f"- {who(i)}: {_cards(n)}" + ("" if view["active"][i] else " (out of the game)") + (" (on turn)" if view["turn"] == i and not view["finished"] else ""))
        recent = []
        for item in view["history"][-6:]:
            if item["kind"] == "play":
                recent.append(f"{names[item['seat']]} played {onecard.card_text(item['card'])}" + (f" and named {item['suit']}" if item["suit"] else ""))
            elif item["kind"] == "draw":
                recent.append(f"{names[item['seat']]} drew " + ("a card" if item["count"] == 1 else f"{item['count']} cards"))
            elif item["kind"] in ("forfeit", "bankrupt"):
                recent.append(f"{names[item['seat']]} " + ("left the game" if item["kind"] == "forfeit" else "went bankrupt"))
        if recent:
            lines.append("Recent play: " + "; ".join(recent) + ".")
        return lines

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
            logging.warning("Could not archive the One Card thread (%s)", type(error).__name__)

    # --- thread chat ------------------------------------------------------------------------

    async def note_thread_message(self, message) -> None:
        """Counts a member's message in a table thread; enough of them bury the turn message, which then moves to the bottom."""
        if message.author.bot or not isinstance(message.channel, discord.Thread):
            return
        key = self.threads.get(message.channel.id)
        if key is not None and message.guild is not None and message.guild.id == key[0]:
            self.table(key).talk += 1

    def _reply_seat(self, message, snap, key):
        """The seat a reply points at: a member's message maps to that member, a character's webhook message to the character of that name;
        anything else (the table's own messages) is no target."""
        replied = getattr(getattr(message, "reference", None), "resolved", None)
        author = getattr(replied, "author", None)
        if author is None:
            return None
        if getattr(replied, "webhook_id", None):
            name = getattr(author, "name", None)
            return next((s["index"] for s in snap["seats"] if s["kind"] == "character" and s["name"] == name), None)
        if getattr(author, "bot", False):
            return None
        return self.store.member_seat_index(key[0], snap["round_id"], author.id)

    async def on_thread_message(self, message) -> None:
        """Reads a seated member's message in a table thread: "원카드" declares (held at two cards), a call-out makes the named (or only exposed)
        seat's slip count. Only a sure reading acts, silently apart from one reaction; a refused move is dropped."""
        try:
            await self._read_message(message)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("Reading a One Card thread message failed")

    async def _read_message(self, message) -> None:
        author = message.author
        if author.bot or getattr(message, "webhook_id", None) or message.guild is None or not isinstance(getattr(message, "content", None), str):
            return
        key = self.threads.get(message.channel.id)
        if key is None or key[0] != message.guild.id or self.closed:
            return
        out = None
        async with self.runtime.lock(key):
            if not self._open(self._row(key)) or not self.store.game_enabled(key[0], "onecard"):
                return
            snap = self.store.latest_round(*key)
            view = snap["view"]
            if snap["status"] != "playing" or not view:
                return
            seat = self.store.member_seat_index(key[0], snap["round_id"], author.id)
            if seat is None or not view["active"][seat]:
                return
            names = self._talk_names(key[0], snap)
            reading = read_onecard(message.content, [(i, [name]) for i, name in enumerate(names) if i != seat], self._reply_seat(message, snap, key),
                                   (snap["rules"] or {}).get("declare_phrases"))
            if reading.outcome != "sure":
                return
            try:
                if reading.kind == "declaration":
                    count = view["counts"][seat]
                    if count == 2 and not view["declared"][seat]:
                        self.held.setdefault(key, {})[seat] = (self.clock(), snap["round_id"], message)
                        return
                    if count != 1:
                        return
                    out = self._apply(key, snap, seat, "declare", "member")
                else:
                    target = reading.target_seat if reading.target_kind != "none" else (view["exposed"][0] if len(view["exposed"]) == 1 else None)
                    if target is None or target == seat:
                        return
                    out = self._apply(key, snap, seat, f"call_out:{target}", "member")
            except (ConflictError, GameError) as error:
                logging.info("One Card chat move refused (%s)", type(error).__name__)
                return
            out.react.append(message)
        await self._publish(key, out)

    # --- restart ----------------------------------------------------------------------------

    async def recover(self) -> None:
        """Refunds and closes every One Card table left open by a previous run, then tells the lobby and the thread (best effort)."""
        store, refunded, closed, failed = self.store, 0, 0, 0
        rounds = [r for r in store.unfinished_rounds() if r.get("game") == "onecard"]
        tables = {(r["guild_id"], r["table_id"]): (r["channel_id"], r["message_id"], r["thread_id"]) for r in rounds}
        for row in store.all("SELECT id,guild_id,channel_id,message_id,thread_id FROM game_tables WHERE status='open' AND game='onecard'"):
            tables.setdefault((row["guild_id"], row["id"]), (row["channel_id"], row["message_id"], row["thread_id"]))
        # tables that closed just before the restart: their archive timer never fired (closed_at is updated_at; no schema change)
        earlier = [r["thread_id"] for r in store.all("SELECT thread_id FROM game_tables WHERE game='onecard' AND status='closed' AND thread_id IS NOT NULL AND updated_at>=?", (time.time() - self.ARCHIVE_SECONDS,))]
        notes = {(r["guild_id"], r["table_id"]): self._refunded((r["guild_id"], r["table_id"]), past=True) for r in rounds}
        for r in rounds:
            try:
                store.cancel_round(r["guild_id"], r["round_id"], "bot restarted")
                refunded += 1
            except Exception:
                failed += 1
                logging.exception("Could not refund One Card round %s on restart", r["round_id"])
        for (guild_id, table_id), (channel_id, message_id, thread_id) in tables.items():
            try:
                store.close_table(guild_id, table_id)
                closed += 1
            except Exception:
                failed += 1
                logging.exception("Could not close One Card table %s on restart", table_id)
                continue
            self._release((guild_id, table_id))
            self.tables.pop((guild_id, table_id), None)
            self.banter = {thread: k for thread, k in self.banter.items() if k != (guild_id, table_id)}
            text = "This table closed when the bot restarted." + notes.get((guild_id, table_id), "")
            await self._edit(channel_id, message_id, content=text, view=None)
            await self._post(thread_id, text)
            await self._archive_thread(thread_id)  # the old process's archive timer died with it
        for thread_id in earlier:
            await self._archive_thread(thread_id)
        if refunded or closed or failed:
            logging.info("One Card recovery: %d rounds refunded, %d tables closed, %d failures", refunded, closed, failed)
