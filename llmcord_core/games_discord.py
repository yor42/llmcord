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
import re
import time
import weakref
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import NamedTuple

import discord
from discord import app_commands

from .admin_store import ConflictError, GameError
from . import game_chooser
from .games import DefaultPolicy, blackjack
from .spoken import veto_client
from .game_talk import TalkBudget
from .table_runtime import TableRuntime
from llmcord_core import i18n
from llmcord_core.i18n import N_, NP_, discord_locale, terms
from llmcord_core.i18n.discord_locale import C_

STALE = N_("That table has moved on — use the latest buttons.")
OFF_HERE = N_("Games are off in this channel.")
BLACKJACK_OFF = N_("Blackjack is off on this server.")
BUSY = N_("A round is being played; join the next one with /blackjack when it ends.")
LEAVE_BUSY = N_("You can leave when this round ends.")
LEFT = N_("You left the table.")
NOT_SEATED = N_("You're not seated at this table.")
NOT_YOUR_TURN = N_("It's not your turn.")
NO_CHARACTER_SEATS = N_("Characters can't sit at tables in this channel.")
_OPEN_ELSEWHERE = {"doubt": N_("An I Doubt It table is open in this channel."), "holdem": N_("A Texas Hold'em table is open in this channel."),
                   "onecard": N_("A One Card table is open in this channel."), "yut": N_("A Yut table is open in this channel.")}
SKIPPED = {"seated": N_("already at the table"), "full": N_("table full"), "archived": N_("archived")}
CLOSED_OFF = N_("Table closed: games were turned off here.")
CLOSED_BLACKJACK_OFF = N_("Table closed: blackjack was turned off on this server.")
CLOSED = N_("Table closed.")
MAX_BET = 100_000
HOW_TO_JOIN = N_("Join with /blackjack bet:<amount>; add favorites:True to bring your favorite characters.")
RESULT_REFUNDED = NP_("blackjack result", "refunded")
RESULT_WORDS = {"surrender": NP_("blackjack result", "surrendered"), "even_money": NP_("blackjack result", "even money")}
STATUS_WORDS = {"stand": NP_("blackjack status", "stands"), "bust": NP_("blackjack status", "bust"),
                "blackjack": NP_("blackjack status", "blackjack"), "surrender": NP_("blackjack status", "surrendered")}
VETO_ADMIN_ON = N_("The decision-model check for spoken moves is now on.")
VETO_ADMIN_OFF = N_("The decision-model check for spoken moves is now off.")
VETO_DISCLOSURE = N_("When on, a message that the local reader takes for a spoken move is sent to {provider}, a third-party service, to double-check it. "
                     "Only the message text and the legal moves are sent, only for members who turned on spoken moves, and only on their own turn. Off by default.")
VETO_NOT_CONFIGURED = N_("It has no effect until the operator configures it.")
PREFIX = "llmcord:bj"
NO_MENTIONS = discord.AllowedMentions.none()


class Turn(NamedTuple):
    """The member on turn at a playing table, kept in memory for spoken-move checks (FEAT-37 D)."""
    round_id: int
    moves: int
    member_id: int
    channel_id: int


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
    after: list = field(default_factory=list)  # later frames of a paced dealer play, sent in order after this one


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


class GameButton(discord.ui.DynamicItem[discord.ui.Button], template=PREFIX + r":(?P<action>[a-z_]+):(?P<table>\d+):(?P<round>\d+):(?P<moves>\d+)"):
    """One button; its custom_id carries the table, the round and the move count the message showed."""

    def __init__(self, action: str, table_id: int, round_id: int, moves: int, label: str = "", style=discord.ButtonStyle.secondary):
        super().__init__(discord.ui.Button(label=label, style=style, custom_id=f"{PREFIX}:{action}:{table_id}:{round_id}:{moves}"))
        self.action, self.table_id, self.round_id, self.moves = action, table_id, round_id, moves

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["action"], int(match["table"]), int(match["round"]), int(match["moves"]), item.label or "", item.style)

    async def callback(self, interaction: discord.Interaction):
        await interaction.client.games.press(interaction, self.action, self.table_id, self.round_id, self.moves)


def words(t, value) -> str:
    """An _Out.say or a refusal rendered for its audience: a coded error, a Msg, a Phrase or plain text."""
    return t.error(value) if isinstance(value, BaseException) else t.render(value)


def _sign(net: int) -> str:
    return f"+{net:,}" if net > 0 else f"−{-net:,}" if net < 0 else ""


def _outcome(seat: dict, natural: bool = False, *, t=None) -> str:
    """The result words and net of a settled seat; an insurance bet is shown apart from the main bet (``natural``: the dealer had blackjack)."""
    t = t or i18n.english()
    outcome, cost = seat["outcome"], seat.get("insurance") or 0
    net = (seat["payout"] or 0) - seat["stake"] - cost
    if outcome == "refund":
        return t.render(RESULT_REFUNDED)
    word = t.render(RESULT_WORDS.get(outcome) or terms.BJ_OUTCOMES.get(outcome, outcome))
    main = t("{outcome} {amount}", outcome=word, amount=_sign(net)).strip()
    if not cost:
        return main
    gain = 2 * cost if natural else -cost
    return t("{outcome} {amount}", outcome=word, amount=_sign(net - gain)).strip() + t(", insurance {amount}", amount=_sign(gain))


class GameTables:
    JOIN_SECONDS = 30
    TURN_SECONDS = 60
    IDLE_SECONDS = 120
    CHARACTER_SECONDS = 2
    MODEL_SECONDS = 20
    POST_SECONDS = 10
    RETRY_SECONDS = 5  # read once at construction (TableRuntime.retry_seconds); tests set runtime.retry_seconds
    SWEEP_SECONDS = 30
    DEALER_STEP_SECONDS = 1.5  # 0 turns the paced dealer play off

    def __init__(self, bot, clock=time.time):
        self.bot, self.clock = bot, clock
        self.runtime = TableRuntime(label="Blackjack", retry_seconds=self.RETRY_SECONDS)
        self.senders: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self.tables: dict[tuple[int, int], _Table] = {}
        self.turns: dict[tuple[int, int], Turn] = {}  # table key -> member on turn; the spoken-move check reads it without the store
        self.turn_channels: dict[tuple[int, int], tuple[int, int]] = {}  # (guild, channel) -> table key
        self.talk_capped: set[tuple[int, object]] = set()
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

    SLOT = "table"  # blackjack keeps one timer per table: arming any kind replaces the previous one

    def arm(self, key, kind: str, round_id: int, moves: int = 0) -> None:
        if self.closed:
            return
        delay = {"join": self.JOIN_SECONDS, "turn": self.TURN_SECONDS, "idle": self.IDLE_SECONDS, "character": self.CHARACTER_SECONDS}[kind]
        self.disarm(key)
        self.table(key).deadline = self.clock() + delay
        args = (*key, round_id) if kind in ("join", "idle") else (*key, round_id, moves)
        self.runtime.arm(key, self.SLOT, delay, self._fire, kind, *args, name=f"blackjack-{kind}-{key[1]}")

    async def _fire(self, kind, *args) -> None:
        await getattr(self, f"on_{kind}_timer")(*args)  # looked up when the timer fires

    def disarm(self, key) -> None:
        self.runtime.disarm(key, self.SLOT)

    async def close(self) -> None:
        await self.runtime.close()
        self.runtime.sweeper = None

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
        elif status in ("settled", "cancelled"):
            table.present = {seat["ref_id"] for seat in snap["seats"] if seat["kind"] == "member"}
            self.arm(key, "idle", snap["round_id"])
        else:
            self.disarm(key)
        self._note_turn(key, snap)

    def _note_turn(self, key, snap) -> None:
        """Records the member on turn (or clears it), then tells the spoken-move checker so a check for an earlier turn goes away."""
        seat = None
        if snap["status"] == "playing" and snap["view"] and snap["view"]["turn"] is not None:
            seat = snap["seats"][snap["view"]["turn"]]
        if seat is not None and seat["kind"] == "member":
            turn = Turn(snap["round_id"], snap["moves"], seat["ref_id"], snap["channel_id"])
            self.turns[key] = turn
            self.turn_channels[(key[0], turn.channel_id)] = key
        else:
            self._clear_turn(key)
        spoken = getattr(self.bot, "spoken", None)
        if spoken is not None:
            spoken.turn_changed(key)

    def _clear_turn(self, key) -> None:
        turn = self.turns.pop(key, None)
        if turn is not None and self.turn_channels.get((key[0], turn.channel_id)) == key:
            del self.turn_channels[(key[0], turn.channel_id)]

    def _forget(self, key) -> None:
        self.disarm(key)
        self.tables.pop(key, None)
        self._clear_turn(key)
        spoken = getattr(self.bot, "spoken", None)
        if spoken is not None:
            spoken.turn_changed(key)
            spoken.forget(key)

    # --- dashboard closes (no IPC: the dashboard closes tables in the shared database) -------

    def _ensure_sweeper(self) -> None:
        if self.tables:
            self.runtime.ensure_sweeper(interval=lambda: self.SWEEP_SECONDS, held=lambda: bool(self.tables), sweep=self.sweep, name="blackjack-sweep")

    async def sweep(self) -> None:
        """Cleans up tables the dashboard closed in the store while the bot still holds them in memory."""
        for key in list(self.tables):
            if self._is_open(key):
                continue
            async with self.runtime.lock(key):
                _, paint = self._closed(key)
            if paint is not None:
                await self._deliver(paint)

    def _table_row(self, key):
        return self.store.one("SELECT status,channel_id,message_id,game FROM game_tables WHERE id=? AND guild_id=?", (key[1], key[0]))

    def _foreign(self, key) -> bool:
        """A table of another game (I Doubt It has its own tables object): blackjack never touches it."""
        row = self._table_row(key)
        return row is not None and row["game"] != "blackjack"

    def _is_open(self, key) -> bool:
        row = self._table_row(key)
        return row is not None and row["status"] == "open" and self.store.blackjack_enabled(key[0])

    def _off_text(self, key) -> str:
        return CLOSED_OFF if self.store.blackjack_enabled(key[0]) else CLOSED_BLACKJACK_OFF

    def _on_here(self, key, row) -> bool:
        return self.store.blackjack_enabled(key[0]) and row["channel_id"] in self.store.game_channels(key[0])

    def _closed(self, key, repaint=False) -> tuple[bool, Paint | None]:
        """Under the table lock: is the table closed in the store? If the bot still holds it, forget it and return the paint that closes its message.
        ``repaint`` (a button press) also returns a one-off closing paint for a table the bot no longer holds, e.g. one closed while it was down."""
        row = self._table_row(key)
        if row is not None and row["status"] == "open" and not self.store.blackjack_enabled(key[0]):
            self.store.close_table(*key)  # left open by a flag flip: close it with refunds, never just forget it
        if self._is_open(key):
            return False, None
        if key not in self.tables:
            if not repaint:
                return True, None
            row = self._table_row(key)
            if row is None:
                return True, None
            return True, self._closing(key, CLOSED if self._on_here(key, row) else self._off_text(key))
        paint = self._closing(key, self._off_text(key))
        self._forget(key)
        return True, paint

    def _if_closed(self, key, text: str | None = None, repaint=False) -> _Out | None:
        """A refusal for a closed table. Without ``text``: "Games are off here" if the channel no longer allows games, else "Table closed."."""
        closed, paint = self._closed(key, repaint)
        if not closed:
            return None
        if text is None:
            row = self._table_row(key)
            text = STALE if row is None else CLOSED if self._on_here(key, row) else OFF_HERE if self.store.blackjack_enabled(key[0]) else BLACKJACK_OFF
        return _Out(text, [paint] if paint else [])

    # --- painting ---------------------------------------------------------------------------

    def _paint(self, key, snap) -> Paint:
        """Call under the table lock: the content is computed from the state the lock protects and numbered.
        A round that just settled gets the dealer's play as extra frames (``after``), numbered before the final paint, so any newer paint outranks them all."""
        sender, t = self.table(key).sender, self._t(key[0])
        paints = []
        for content in self._frames(snap, t=t):
            sender.issued += 1
            paints.append(Paint(key, sender, sender.issued, content, self._help_view(snap, t=t)))
        sender.issued += 1
        paints.append(Paint(key, sender, sender.issued, self.render(snap, t=t), self.buttons(snap, t=t)))
        paints[0].after = paints[1:]
        return paints[0]

    def _frames(self, snap, *, t=None) -> list[str]:
        if not self.DEALER_STEP_SECONDS or snap["status"] != "settled" or not snap["view"]:
            return []
        view = snap["view"]
        if len(view["dealer"]) == 2 and not blackjack.is_natural(view["dealer"]) and "stand" not in view["status"]:
            return []  # everyone busted or has a blackjack: the dealer neither draws nor decides anything
        return [self.render(snap, shown=n, t=t) for n in range(2, len(view["dealer"]) + 1)]

    @staticmethod
    def _help_view(snap, *, t=None) -> discord.ui.View:
        """The only button that is safe on a frame of the dealer's play, so a failed final edit never leaves a bare message."""
        t = t or i18n.english()
        view = discord.ui.View(timeout=None)
        view.add_item(GameButton("help", snap["table_id"], snap["round_id"], snap["moves"], t.p("button.blackjack", "How to play"), discord.ButtonStyle.secondary))
        return view

    def summary_rounds(self, guild_id) -> int:
        return self.store.game_summary_rounds(guild_id)

    def _closing(self, key, text: str) -> Paint:
        sender = self.sender(key)
        sender.issued += 1
        t = self._t(key[0])
        text = t.render(text)
        try:
            text = self._summary_text(text, self.store.game_table_summary(key[0], key[1], self.summary_rounds(key[0])), t=t)
        except Exception:
            logging.exception("Could not build the blackjack table summary (table %s)", key[1])
        return Paint(key, sender, sender.issued, text, None, True)

    def _summary_text(self, text: str, rounds: list, *, t=None) -> str:
        """The closing text plus the last rounds; the oldest rounds go first when it would not fit a message."""
        while rounds:
            lines = [text, *(self._round_line(r, t=t) for r in rounds)]
            if len(rounds) > 1:
                lines.append(self._net_line(rounds, t=t))
            if len("\n".join(lines)) <= 2000:
                return "\n".join(lines)
            rounds = rounds[1:]
        return text

    def _round_line(self, rnd, *, t=None) -> str:
        t = t or i18n.english()
        dealer, number = rnd.get("dealer_total"), rnd["number"]
        if dealer is None:
            head = t("Round {number}:", number=number)
        elif dealer > 21:
            head = t("Round {number}: dealer {total} bust —", number=number, total=dealer)
        else:
            head = t("Round {number}: dealer {total} —", number=number, total=dealer)
        even = t.p("round net", "even") if "winner" in rnd else t.p("round net", "push")  # an I Doubt It row carries winner/pot, not a dealer
        return head + " " + ", ".join(f"{self._who(s, t=t)} {_sign(s['net']) or even}" for s in rnd["seats"])

    def _net_line(self, rounds, *, t=None) -> str:
        t = t or i18n.english()
        totals: dict = {}
        for rnd in rounds:
            for s in rnd["seats"]:
                totals.setdefault((s["kind"], s["ref_id"]), [s, 0])[1] += s["net"]
        even = t.p("round net", "even")
        return t("Net: {entries}", entries=", ".join(f"{self._who(s, t=t)} {_sign(net) or even}" for s, net in totals.values()))

    async def _deliver(self, paint: Paint, via=None, pace=True) -> None:
        """Sends a paint, then (``pace``) the frames that follow it."""
        await self._send(paint, via)
        if pace:
            await self._pace(paint)

    async def _pace(self, paint: Paint) -> None:
        """Waits between the frames of the dealer's play. No lock is held while waiting; a frame older than a newer paint that was sent is dropped by ``_send``."""
        for nxt in paint.after:
            if nxt.seq <= paint.sender.sent:
                return
            await self._pause(self.DEALER_STEP_SECONDS)
            await self._send(nxt)

    @staticmethod
    async def _pause(seconds: float) -> None:
        await asyncio.sleep(seconds)

    async def _send(self, paint: Paint, via=None) -> None:
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

    def _t(self, guild_id):
        """The channel's translator for a table message: resolved at each render, never kept."""
        get_guild = getattr(self.bot, "get_guild", None)
        return discord_locale.channel_t(get_guild(guild_id) if get_guild else None)

    @staticmethod
    def _who(seat, *, t=None) -> str:
        t = t or i18n.english()
        if seat["kind"] != "character":
            return f"<@{seat['ref_id']}>"
        who = f"**{discord.utils.escape_markdown(discord.utils.escape_mentions(seat['name']))}**"
        return who + (" " + t("(with {member})", member=f"<@{seat['brought_by']}>") if seat["brought_by"] else "")

    def table_talk_context(self, guild_id, channel_id) -> dict | None:
        """FEAT-55: the open blackjack table of this channel in the Doubt/Hold'em shape (blackjack has no thread, so the key is the channel), else None."""
        row = self.store.open_table_for(guild_id, channel_id)
        if row is None or row["game"] != "blackjack":
            return None
        try:
            snap = self.store.latest_round(guild_id, row["id"])
        except GameError:
            return None
        return {"table_id": row["id"], "round_id": snap["round_id"], "status": snap["status"],
                "seated_character_ids": [s["ref_id"] for s in snap["seats"] if s["kind"] == "character"]}

    def _talk_names(self, guild_id, snap) -> list:
        """One prompt-safe name per seat: characters by name, members by guild display name (Player N when unknown); no mentions, ids or markdown."""
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

    def public_block(self, guild_id, key) -> str | None:
        """FEAT-55: the third-person public view of a table: bets, visible cards, the dealer's up card (the hole card stays face down), turn.
        Never the seed, its hash or a mention."""
        try:
            snap = self.store.latest_round(*key)
        except GameError:
            return None
        names, view, name = self._talk_names(key[0], snap), snap["view"], self.store.currency_name(key[0])
        lines = ["A table of Blackjack (players try to beat the dealer by getting closer to 21 without going over)."]
        status = snap["status"]
        sender = self.senders.get(key)
        # MNT-87: while the channel still animates the dealer's draws, the settled result is not public yet.
        dealing = status == "settled" and sender is not None and sender.sent < sender.issued and bool(self._frames(snap))
        if dealing:
            status, view = "dealing", {**view, "dealer": view["dealer"][:1], "dealer_hidden": True}
        for seat in snap["seats"]:
            i = seat["index"]
            line = f"- {names[i]}: bet {(view['stakes'][i] if view else seat['stake']):,} {name}"
            if view:
                cards = view["hands"][i]
                line += f", {blackjack.hand_str(cards)} ({blackjack.total_str(cards)})"
                word = {"stand": "stands", "bust": "bust", "blackjack": "blackjack", "surrender": "surrendered"}.get(view["status"][i])
                line += f", {word}" if word and status == "playing" else ""
                line += " (on turn)" if status == "playing" and view["turn"] == i else ""
            if status == "settled":
                line += f", {_outcome(seat, bool(view) and blackjack.is_natural(view['dealer']))}"
            lines.append(line)
        if view:
            dealer = view["dealer"]
            if dealing:
                lines.append(f"Dealer shows {blackjack.hand_str(dealer)}; the dealer is playing, and the results are not known yet.")
            elif view["dealer_hidden"]:
                lines.append(f"Dealer shows {blackjack.hand_str(dealer)}; the hole card is face down.")
            else:
                lines.append(f"Dealer: {blackjack.hand_str(dealer)} ({blackjack.total_str(dealer)})" + (" bust" if blackjack.hand_total(dealer)[0] > 21 else ""))
        elif status == "joining":
            lines.append("No cards have been dealt yet.")
        if status == "cancelled":
            lines.append("The round was cancelled; bets were refunded.")
        return "\n".join(lines)

    def render(self, snap, shown=None, *, t=None) -> str:
        """``shown``: a frame of the dealer's play with that many dealer cards, before the results."""
        t = t or i18n.english()
        if shown is not None:
            snap = {**snap, "status": "playing", "view": {**snap["view"], "dealer": snap["view"]["dealer"][:shown], "dealer_hidden": False, "turn": None}}
        key = (snap["guild_id"], snap["table_id"])
        table, store, guild_id = self.table(key), self.store, snap["guild_id"]
        name, limits = store.currency_name(guild_id), store.game_settings(guild_id)
        status, view, deadline = snap["status"], snap["view"], table.deadline
        stamp = f"<t:{int(deadline)}:R>" if deadline else ""
        lines = [f"**{t.render(terms.GAME_BLACKJACK)}**"]
        for seat in snap["seats"]:
            i = seat["index"]
            line = t("{number}. {who} bet {stake:,} {currency}", number=i + 1, who=self._who(seat, t=t), stake=(view['stakes'][i] if view else seat['stake']), currency=name)
            if view:
                cards = view["hands"][i]
                line += " — " + t("{hand} ({total})", hand=blackjack.hand_str(cards), total=blackjack.total_phrase(cards))
                if status == "playing":
                    word = STATUS_WORDS.get(view["status"][i])
                    line += f" {t.render(word)}" if word else ""
                    line += " " + t("(even money)") if view.get("even_money") and view["even_money"][i] else " " + t("(insured {amount:,})", amount=seat['insurance']) if seat.get("insurance") else ""
                    line += " " + t("(timed out)") if (snap["round_id"], i) in table.timeouts else ""
            if status == "settled":
                line += f" — {_outcome(seat, bool(view) and blackjack.is_natural(view['dealer']), t=t)}"
                line += " " + t("(timed out)") if (snap["round_id"], i) in table.timeouts else ""
            lines.append(line)
        if view:
            dealer = view["dealer"]
            if view["dealer_hidden"]:
                lines.append(t("Dealer: {hand} ??", hand=blackjack.hand_str(dealer)))
            elif blackjack.hand_total(dealer)[0] > 21:
                lines.append(t("Dealer: {hand} ({total}) bust", hand=blackjack.hand_str(dealer), total=blackjack.total_phrase(dealer)))
            else:
                lines.append(t("Dealer: {hand} ({total})", hand=blackjack.hand_str(dealer), total=blackjack.total_phrase(dealer)))
        if status == "joining":
            lines.append(t("Join with /blackjack bet:<amount> — dealing {stamp}.", stamp=stamp) if stamp else t("Join with /blackjack bet:<amount>."))
            lines.append(t("Bets: {low:,} to {high:,} {currency}.", low=limits['min_bet'], high=limits['max_bet'], currency=name))
        elif status == "playing" and view and view["turn"] is not None:
            on_turn = snap["seats"][view["turn"]]
            if snap.get("phase") == "insurance":
                lines.append(t("Dealer shows an ace. Insurance?"))
            if on_turn["kind"] == "character":
                lines.append(t("On turn: {who}.", who=self._who({**on_turn, 'brought_by': None}, t=t)))
            else:
                lines.append(t("On turn: {who} — decide {stamp}.", who=f"<@{on_turn['ref_id']}>", stamp=stamp) if stamp else t("On turn: {who}.", who=f"<@{on_turn['ref_id']}>"))
        elif status in ("settled", "cancelled"):
            if status == "cancelled":
                lines.append(t("Round cancelled. Bets are refunded."))
            lines.append(t("Play again or leave. The table closes {stamp} if nobody plays again.", stamp=stamp) if stamp else t("Play again or leave."))
        if shown is not None:
            lines.append(t("Dealer plays."))
        lines.append("-# " + " · ".join(t.render(p) for p in blackjack.rules_phrases(snap.get("rules"))))
        if status in ("settled", "cancelled") and snap["seed"]:
            lines.append("-# " + t("Seed: {seed} (hash {hash})", seed=snap['seed'], hash=snap['seed_hash'][:12]))
        else:
            lines.append("-# " + t("Seed hash: {hash}", hash=snap['seed_hash'][:12]))
        return "\n".join(lines)

    def buttons(self, snap, *, t=None) -> discord.ui.View | None:
        t = t or i18n.english()
        status, table_id, round_id, moves = snap["status"], snap["table_id"], snap["round_id"], snap["moves"]
        spec = []
        if status == "joining":
            spec = [("deal", t.p("button.blackjack", "Deal now"), discord.ButtonStyle.primary), ("leave", t.p("button.blackjack", "Leave"), discord.ButtonStyle.secondary)]
        elif status == "playing":
            legal = next(iter(snap["legal"].values()), ())
            turn = snap["view"]["turn"] if snap["view"] else None
            if turn is not None and snap["seats"][turn]["kind"] == "character":
                legal = ()  # members cannot press for a character
            cost = snap["view"]["stakes"][turn] // 2 if turn is not None else 0
            labels = {"hit": t.p("button.blackjack", "Hit"), "stand": t.p("button.blackjack", "Stand"), "double": t.p("button.blackjack", "Double"), "surrender": t.p("button.blackjack", "Surrender"),
                      "insure": t.p("button.blackjack", "Insure ({cost:,})", cost=cost), "even_money": t.p("button.blackjack", "Even money"), "no_insurance": t.p("button.blackjack", "No insurance")}
            spec = [(m, labels[m], discord.ButtonStyle.primary if m in ("hit", "insure") else discord.ButtonStyle.secondary) for m in labels if m in legal]
        elif status in ("settled", "cancelled"):
            spec = [("again", t.p("button.blackjack", "Play again (same bet)"), discord.ButtonStyle.success), ("close", t.p("button.blackjack", "Leave table"), discord.ButtonStyle.secondary)]
        if not spec and status not in ("joining", "playing", "settled", "cancelled"):
            return None
        spec.append(("help", t.p("button.blackjack", "How to play"), discord.ButtonStyle.secondary))
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

    async def _say(self, interaction, text) -> None:
        """An ephemeral reply: the first response, or a followup once the interaction was acknowledged."""
        text = words(discord_locale.invoker_t(interaction), text)
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
                say = OFF_HERE if self.store.blackjack_enabled(out.paints[0].key[0]) else BLACKJACK_OFF  # the table was closed (and the bet refunded) after the join
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
                if closed is not None:  # a table opened since the look: if it was an I Doubt It one, clean it up (blackjack's own sweep covers its tables)
                    await self.bot.doubt.sweep()
                    await self.bot.holdem.sweep()
                    await self.bot.onecard.sweep()
                    await self.bot.yut.sweep()
                break
            if table["game"] == "yut":
                closed = await self.bot.yut.close_for_channel(guild_id, channel_id, table["id"])
                break
            if table["game"] == "onecard":
                closed = await self.bot.onecard.close_for_channel(guild_id, channel_id, table["id"])
                break
            if table["game"] == "holdem":
                closed = await self.bot.holdem.close_for_channel(guild_id, channel_id, table["id"])
                break
            if table["game"] == "doubt":  # its own lock namespace: never hold a blackjack table lock for it
                closed = await self.bot.doubt.close_for_channel(guild_id, channel_id, table["id"])
                break
            async with self.runtime.lock((guild_id, table["id"])):
                current = self.store.open_table_for(guild_id, channel_id)
                if current is not None and current["id"] != table["id"] and attempt < 2:
                    continue  # a newer table opened meanwhile: release this lock and close it under its own
                closed = self.store.set_game_channel(guild_id, channel_id, False)["closed_table"]
                if closed is not None and (current or table)["game"] == "blackjack":
                    paint = self._closing((guild_id, closed), CLOSED_OFF)
                    self._forget((guild_id, closed))
            if closed is not None and paint is None:  # another game's table (opened meanwhile): its own sweep says so
                await self.bot.doubt.sweep()
                await self.bot.holdem.sweep()
                await self.bot.onecard.sweep()
                await self.bot.yut.sweep()
            break
        return closed is not None, paint

    # --- /blackjack -------------------------------------------------------------------------

    async def play_command(self, interaction: discord.Interaction, bet: int, favorites: bool = False) -> None:
        # Not deferred: the first opener's table message is the public response, which an ephemeral defer would
        # make impossible. The table lock is held for store work only (no HTTP), and `_finish` answers before any
        # table edit goes out, so the response does not wait for the edit queue or the channel HTTP.
        guild_id = interaction.guild_id
        t = discord_locale.invoker_t(interaction)
        channel_id = getattr(interaction, "channel_id", None) or interaction.channel.id
        if interaction.user.bot:
            await self._say(interaction, t("Bots cannot play blackjack."))
            return
        if not self.store.blackjack_enabled(guild_id):
            await self._say(interaction, BLACKJACK_OFF)
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
        if table is not None and table["game"] != "blackjack":
            await self._say(interaction, t(_OPEN_ELSEWHERE.get(table["game"], _OPEN_ELSEWHERE["doubt"])))
            return
        if table is None:
            try:
                opened = self.store.open_table(guild_id, channel_id, "blackjack", interaction.user.id)
            except GameError as error:
                await self._say(interaction, t.error(error))
                return
            table = {"id": opened["table_id"]}
        key = (guild_id, table["id"])
        async with self.runtime.lock(key):
            out = self._join(interaction.user.id, key, bet, opened, space_id=space_id, note=note, t=t)
        if out.opened:
            await self._open_message(interaction, key, out.paints[0])
            if out.note:
                await self._say(interaction, out.note)
        else:
            await self._finish(interaction, out)

    def _favorites_note(self, result, note, *, t=None) -> str:
        t = t or i18n.english()
        name, parts = self.store.currency_name(result["snapshot"]["guild_id"]), []
        if result["seated"]:
            seated_names = ", ".join(t("{name} ({stake:,})", name=s['name'], stake=s['stake']) for s in result["seated"])
            parts.append(t("Your favorites joined too: {names}.", names=seated_names))
        if result["skipped"]:
            why = lambda reason: t("not enough {currency}", currency=name) if reason == "broke" else t.render(SKIPPED[reason])
            skipped_names = ", ".join(t("{name} ({reason})", name=s['name'], reason=why(s['reason'])) for s in result["skipped"])
            parts.append(t("Not seated: {names}.", names=skipped_names))
        return " ".join([*parts, t.render(note)] if note else parts)

    def _join(self, user_id, key, bet, opened, space_id=None, note="", *, t=None) -> _Out:
        t = t or i18n.english()
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
                    snap, note = result["snapshot"], self._favorites_note(result, note, t=t)
                else:
                    snap = self.store.join_round(guild_id, snap["round_id"], user_id, bet)
            except GameError:
                if fresh and opened is None:
                    self.store.cancel_round(guild_id, snap["round_id"], "nobody joined")
                raise
        except GameError as error:
            if opened is not None:
                self._close_in_store(key)
            return _Out(error)
        self.table(key).bets[user_id] = bet
        self.table(key).present.add(user_id)
        self._after(key, snap)
        paint = self._paint(key, snap)
        if opened is not None:
            return _Out(paints=[paint], opened=True, note=note)
        return _Out(t("You joined the table with {bet:,} {currency}.", bet=bet, currency=self.store.currency_name(guild_id)) + (f" {t.render(note)}" if note else ""), [paint], joined=True)

    async def _open_message(self, interaction, key, paint: Paint) -> None:
        """Posts the table message (the public response). Edits of this table wait behind it, so none is lost to a missing message id."""
        async with paint.sender.lock:
            try:
                await interaction.response.send_message(paint.content, view=paint.view, allowed_mentions=NO_MENTIONS)
                message = await interaction.original_response()
                self.store.set_table_message(*key, message.id)
            except Exception:
                async with self.runtime.lock(key):
                    self._close_in_store(key)  # refunds; no table without a message
                raise
            paint.sender.sent = max(paint.sender.sent, paint.seq)
        fresh = None
        async with self.runtime.lock(key):
            if paint.sender.issued > paint.seq and key in self.tables and self._is_open(key):
                try:
                    fresh = self._paint(key, self.store.latest_round(*key))  # someone joined before the message existed
                except GameError:
                    pass
        if fresh is not None:
            await self._deliver(fresh)

    # --- buttons ----------------------------------------------------------------------------

    @staticmethod
    def how_to_play(rules=None, *, t=None) -> str:
        t = t or i18n.english()
        return "\n".join(t.render(p) for p in blackjack.how_to_play_phrases(rules)) + "\n" + t(HOW_TO_JOIN)

    def _help_text(self, guild_id, table_id, round_id, *, t=None) -> str:
        """The guide for the rules the table's round was dealt with; the server's current rules when the round is unknown or has none."""
        if guild_id is None:
            return self.how_to_play(t=t)
        rules = None
        try:
            snap = self.store.round_snapshot(guild_id, round_id)
            rules = snap["rules"] if snap["table_id"] == table_id else None
            return self.how_to_play(rules or self.store.blackjack_rules(guild_id), t=t)
        except GameError:
            return self.how_to_play(rules or self.store.blackjack_rules(guild_id), t=t)
        except Exception:
            logging.exception("Could not build the blackjack guide for table %s", table_id)
            return self.how_to_play(t=t)

    async def press(self, interaction: discord.Interaction, action: str, table_id: int, round_id: int, moves: int) -> None:
        if action == "help":  # a guide, not a move: no state, no seat needed, works on old messages
            await self._say(interaction, self._help_text(interaction.guild_id, table_id, round_id, t=discord_locale.invoker_t(interaction)))
            return
        guild_id = interaction.guild_id
        if guild_id is None or interaction.user.bot or self._foreign((guild_id, table_id)):
            await self._say(interaction, STALE)
            return
        await self._ack(interaction)
        key = (guild_id, table_id)
        async with self.runtime.lock(key):
            out = self._press(interaction.user.id, key, action, round_id, moves)
            spoken = getattr(self.bot, "spoken", None)
            claim = spoken.claim_table_move(key, round_id, moves, interaction.user.id, action) if spoken is not None and out.say is None and out.paints else None
        await self._finish(interaction, out, via=interaction)
        if claim is not None:
            spoken.keep_claimed(claim)  # FEAT-37 F: after the answer and the repaint, outside the lock

    async def play_spoken(self, interaction: discord.Interaction, table_id: int, round_id: int, moves: int, move: str) -> str | None:
        """A move chosen on a spoken-move check: the normal press path (same lock, same stale check), but the repaint goes to the
        table message, not to the interaction's own message. None when the move was played, else the refusal text."""
        guild_id = interaction.guild_id
        if guild_id is None or interaction.user.bot or self._foreign((guild_id, table_id)):
            await self._say(interaction, STALE)
            return discord_locale.invoker_t(interaction).render(STALE)
        await self._ack(interaction)
        key = (guild_id, table_id)
        async with self.runtime.lock(key):
            out = self._press(interaction.user.id, key, move, round_id, moves)
        await self._finish(interaction, out)
        return None if out.say is None and out.paints else words(discord_locale.invoker_t(interaction), out.say or STALE)

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
            if action in ("hit", "stand", "double", "surrender", "insure", "even_money", "no_insurance"):
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
            return _Out(error)

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
            return _Out(LEFT, paints)
        return self._close_empty(key)

    # --- timers' work -----------------------------------------------------------------------

    async def on_join_timer(self, guild_id, table_id, round_id) -> None:
        key, paint = (guild_id, table_id), None
        if self._foreign(key):
            return
        async with self.runtime.lock(key):
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
        if self._foreign(key):
            return
        async with self.runtime.lock(key):
            closed, paint = self._closed(key)
            if not closed:
                snap = self.store.round_snapshot(guild_id, round_id)
                if snap["status"] != "playing" or snap["moves"] != moves or not snap["legal"]:
                    return
                seat, legal = next(iter(snap["legal"].items()))
                move = DefaultPolicy.pick(SimpleNamespace(total=blackjack.hand_total(snap["view"]["hands"][seat])[0]), legal)
                asked_insurance = snap["phase"] == "insurance"
                snap = self.store.play(guild_id, round_id, seat, move, "timeout", moves)
                if not asked_insurance:  # declining insurance is not a missed turn
                    self.table(key).timeouts.add((round_id, seat))
                self._after(key, snap)
                paint = self._paint(key, snap)
        if paint is not None:
            await self._deliver(paint)

    async def on_character_timer(self, guild_id, table_id, round_id, moves) -> None:
        """A character seat is on turn: decide outside the table lock (a model call may take seconds), then apply under it if nothing moved meanwhile."""
        key, paint = (guild_id, table_id), None
        if self._foreign(key):
            return
        async with self.runtime.lock(key):
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
        async with self.runtime.lock(key):
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
                    except ConflictError:
                        return
                    except GameError as again:
                        logging.warning("Blackjack character house move refused (%s); cancelling the round", type(again).__name__)
                        snap = self.store.cancel_round(guild_id, round_id, "a character could not move")
                        line = ""
                self._after(key, snap)
                paint = self._paint(key, snap)
        if paint is not None:
            await self._deliver(paint, pace=False)
        if line and not closed:
            await self._post_line(snap["channel_id"], guild_id, seated, line)
        if paint is not None:
            await self._pace(paint)

    @staticmethod
    def _house_move(snap, seat, legal) -> str:
        return DefaultPolicy.pick(SimpleNamespace(total=blackjack.hand_total(snap["view"]["hands"][seat])[0]), legal)

    async def _character_move(self, guild_id, snap, seat, legal, seated) -> tuple[str, str]:
        """(move, line): the model's choice when table talk is on and the call works, else the house rule with no line."""
        house = self._house_move(snap, seat, legal)
        if not self.store.game_character_talk(guild_id):
            return house, ""
        if self.talk.capped():
            return house, ""  # spending hard cap: the house rule plays, nothing to log or count
        marker = (guild_id, self.store._server_day(guild_id, time.time())[1])
        if self.talk.remaining(guild_id) <= 0:
            limit = self.store.game_talk_daily_limit(guild_id)
            if marker not in self.talk_capped:
                self.talk_capped = {m for m in self.talk_capped if m[1] == marker[1]} | {marker}
                logging.info("Blackjack table talk reached its daily limit (%d) in guild %s; characters play by the house rule", limit, guild_id)
            return house, ""
        try:
            character = self.store.character_by_id(seated["character_id"])
            if character is None or character["guild_id"] != guild_id:
                return house, ""
            style_only = self.store.game_character_moves(guild_id) == "style"
            offered = (house,) if style_only else legal  # style only: the model may only add a line, the move stays the house rule
            result = await asyncio.wait_for(game_chooser.ask(self.bot, guild_id, snap["channel_id"], character, snap, seat, offered, self.store.character_play_style(guild_id, character["id"])), self.MODEL_SECONDS)
        except Exception as error:
            logging.warning("Blackjack character decision fell back to the house rule (%s)", type(error).__name__)
            self.talk.note_failure(guild_id)
            return house, ""
        move = result.get("move") if isinstance(result, dict) else None
        if not isinstance(move, str) or move not in offered:
            self.talk.note_failure(guild_id)
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
        if self._foreign(key):
            return
        async with self.runtime.lock(key):
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
        rounds = [r for r in store.unfinished_rounds() if r.get("game", "blackjack") == "blackjack"]
        tables = {(r["guild_id"], r["table_id"]): (r["channel_id"], r["message_id"]) for r in rounds}
        for row in store.all("SELECT id,guild_id,channel_id,message_id FROM game_tables WHERE status='open' AND game='blackjack'"):
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
            await self.edit(channel_id, message_id, self._t(guild_id)("This table closed when the bot restarted. Bets were refunded."), None)
        if refunded or closed or failed:
            logging.info("Blackjack recovery: %d rounds refunded, %d tables closed, %d failures", refunded, closed, failed)


def register_game_commands(bot, ctx: SimpleNamespace, admin_games: app_commands.Group) -> None:
    require_guild = ctx.require_guild

    async def reply(interaction: discord.Interaction, message: str) -> None:
        await interaction.response.send_message(message, ephemeral=True, allowed_mentions=NO_MENTIONS)

    @bot.tree.command(name="blackjack", description=C_("Join or open a blackjack table in this channel"))
    @app_commands.describe(bet=C_("How much to bet"), favorites=C_("Bring your favorite characters to the table"))
    async def blackjack_command(interaction: discord.Interaction, bet: app_commands.Range[int, 1, MAX_BET], favorites: bool = False):
        require_guild(interaction)
        await bot.games.play_command(interaction, bet, favorites)

    @admin_games.command(name="channel", description=C_("Turn games on or off in a channel"))
    @app_commands.describe(enabled=C_("On or off"), channel=C_("The channel (default: this one)"))
    @app_commands.checks.has_permissions(administrator=True)
    async def games_channel(interaction: discord.Interaction, enabled: bool, channel: discord.TextChannel | None = None):
        require_guild(interaction)
        channel_id = channel.id if channel else getattr(interaction, "channel_id", None) or interaction.channel.id
        where = channel.mention if channel else f"<#{channel_id}>"
        closed, paint = await bot.games.switch_channel(interaction.guild_id, channel_id, enabled)
        t = discord_locale.invoker_t(interaction)
        text = t("Games are now on in {where}.", where=where) if enabled else t("Games are now off in {where}.", where=where)
        try:
            await reply(interaction, text + (" " + t("The open table was closed and bets were refunded.") if closed else ""))
        finally:
            if paint is not None:
                await bot.games.send_paint(paint)

    @admin_games.command(name="spoken_veto", description=C_("Turn the decision-model check for spoken moves on or off"))
    @app_commands.describe(setting=C_("On or off"))
    @app_commands.choices(setting=[app_commands.Choice(name=C_("On"), value="on"), app_commands.Choice(name=C_("Off"), value="off")])
    @app_commands.checks.has_permissions(administrator=True)
    async def games_spoken_veto(interaction: discord.Interaction, setting: str):
        require_guild(interaction)
        on = setting == "on"
        bot.store.set_spoken_veto_enabled(interaction.guild_id, on)
        t = discord_locale.invoker_t(interaction)
        text = t(VETO_ADMIN_ON) if on else t(VETO_ADMIN_OFF)
        text += " " + t(VETO_DISCLOSURE, provider=veto_client.PROVIDER)
        if bot.spoken.veto is None:
            text += " " + t(VETO_NOT_CONFIGURED)
        await reply(interaction, text)

    @admin_games.command(name="bets", description=C_("Set the smallest and largest bet"))
    @app_commands.describe(min=C_("Smallest bet"), max=C_("Largest bet"))
    @app_commands.checks.has_permissions(administrator=True)
    async def games_bets(interaction: discord.Interaction, min: app_commands.Range[int, 1, MAX_BET], max: app_commands.Range[int, 1, MAX_BET]):
        require_guild(interaction)
        try:
            saved = bot.store.set_game_settings(interaction.guild_id, min, max, expected=bot.store.game_settings(interaction.guild_id))
        except ConflictError:
            await reply(interaction, discord_locale.invoker_t(interaction)("The game settings were changed elsewhere. Run the command again."))
            return
        name = bot.store.currency_name(interaction.guild_id)
        await reply(interaction, discord_locale.invoker_t(interaction)("Bets are now {low:,} to {high:,} {currency}.", low=saved['min_bet'], high=saved['max_bet'], currency=name))
