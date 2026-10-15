"""Spoken blackjack moves on Discord (FEAT-37 part D, D32/D34/D35): a move check offered when the member on turn writes in the table's channel.

Off unless the server switch is on. The in-memory turn index in ``GameTables`` (``turns``, ``turn_channels``) decides, without the store or the
model, whether a message is worth reading at all. The reading runs in a tracked background task; ``on_message`` never waits for it.
Buttons go through the normal press path (``GameTables.play_spoken``), so the table lock and the ``moves`` stale check apply as for the table's own buttons.
No Discord HTTP is done while a table lock is held. The turn timer is not paused.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field

import discord

from .admin_store import GameError
from .games_discord import BLACKJACK_OFF, CLOSED, NO_MENTIONS, OFF_HERE, STALE
from .spoken import blackjack as spoken_bj
from .spoken import head as spoken_head
from .spoken import interpret
from .spoken import veto as spoken_veto
from .spoken import veto_client, veto_stats
from .spoken.veto_limits import VetoLimiter
from llmcord_core.i18n import N_, discord_locale

MAX_CHARS = 200  # longer messages are chat, not a move: ignored before any inference
PREFIX = "llmcord:bjs"
NOT_YOURS = N_("This isn't your move check.")
TALK_KEPT = N_("Got it — left as table talk. Tired of these? Turn them off with /game spoken_moves.")
DISCORD_TOKEN = re.compile(r"<(?:@[!&]?|#|a?:\w+:)\d*")  # mention, channel and custom-emoji tokens never leave the process
VETO_NOTICE = N_("Heads up: when you type a move at a game table, the message may be sent to a third-party service to double-check it. Turn that off for yourself with /game spoken_veto off.")
RUNTIME_OFF ="Spoken moves are unavailable: the reading model is not ready."


def _names(t) -> dict:
    """The move names as the move buttons and the check question show them (one render, in the audience's words)."""
    return {"hit": t.p("button.blackjack", "Hit"), "stand": t.p("button.blackjack", "Stand"), "double": t.p("button.blackjack", "Double"), "surrender": t.p("button.blackjack", "Surrender"),
            "insure": t.p("button.blackjack", "Insure"), "even_money": t.p("button.blackjack", "Even money"), "no_insurance": t.p("button.blackjack", "No insurance")}


@dataclass
class _Check:
    """The one pending move check of a table."""
    message: object
    member_id: int
    round_id: int
    moves: int
    text: str
    offered: tuple
    band: str = "sure"
    scores: dict = field(default_factory=dict)
    number: int = 0
    dropped: tuple | None = None  # shadow mode: the moves the reader would have dropped (None: no shadow reading)


@dataclass
class _Reading:
    """What the newest phrase of a table's turn was read as (kept in memory only; silent readings too)."""
    turn: object
    text: str
    band: str
    offered: tuple
    scores: dict
    number: int = 0


class SpokenButton(discord.ui.DynamicItem[discord.ui.Button], template=PREFIX + r":(?P<action>move|talk):(?P<table>\d+):(?P<round>\d+):(?P<moves>\d+)(?::(?P<move>[a-z_]+))?"):
    def __init__(self, action: str, table_id: int, round_id: int, moves: int, move: str | None = None, label: str = "", style=discord.ButtonStyle.secondary):
        tail = f":{move}" if move else ""
        super().__init__(discord.ui.Button(label=label, style=style, custom_id=f"{PREFIX}:{action}:{table_id}:{round_id}:{moves}{tail}"))
        self.action, self.table_id, self.round_id, self.moves, self.move = action, table_id, round_id, moves, move

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["action"], int(match["table"]), int(match["round"]), int(match["moves"]), match["move"], item.label or "", item.style)

    async def callback(self, interaction: discord.Interaction):
        await interaction.client.spoken.press(interaction, self.action, self.table_id, self.round_id, self.moves, self.move)


class SpokenMoves:
    def __init__(self, bot, env=None):
        self.bot = bot
        self.runtime = None  # SpokenRuntime; fetched lazily on the first enabled message (tests set a stub)
        self.pending: dict[tuple, _Check] = {}
        self.latest: dict[tuple, int] = {}  # table key -> number of the newest phrase; older readings drop themselves
        self.readings: dict[tuple, _Reading] = {}  # table key -> reading of the turn now on
        self.ended: dict[tuple, _Reading] = {}  # table key -> reading of the turn that just ended (one slot), for the table-button capture
        self.example_locks: dict[int, asyncio.Lock] = {}
        self.examples: dict[int, tuple] = {}  # guild -> (revision, [(example id, vector, label)])
        self.tasks: set[asyncio.Task] = set()
        self.closed = False
        self._warned: set[str] = set()
        self.veto = veto_client.from_env(env if env is not None else os.environ)  # None unless the operator switch and a key are set
        self.veto_timeout = 1.5  # above the client's own 1.0 s timeout, so that one trips the breaker first
        self.veto_notice_pending: set[tuple] = set()  # (guild, member) whose notice post is in flight: no second notice, nothing sent meanwhile
        self.veto_notified: set[tuple] = set()  # (guild, member) told that a message may be double-checked by a third party; memory only, so a restart tells again
        self.limiter = VetoLimiter()
        self.shadow = str((env if env is not None else os.environ).get("LLMCORD_SPOKEN_VETO_SHADOW", "")).lower() in ("1", "true", "yes", "on")  # ask the reader and count, but post e5's own check
        self.stats = veto_stats.VetoStats(clock=time.monotonic)
        self._suspicious: set[tuple] = set()  # (guild, day) already warned about

    # --- entry points -------------------------------------------------------------------------

    def note_message(self, message) -> None:
        """Called from on_message. Cheap filters first; then (switch on) one background reading. Never raises into the caller's flow."""
        guild = message.guild
        if self.closed or guild is None or message.author.bot or message.webhook_id:
            return
        text = (message.content or "").strip()
        if not text or len(text) > MAX_CHARS:
            return
        games = self.bot.games
        key = games.turn_channels.get((guild.id, message.channel.id))
        turn = games.turns.get(key) if key else None
        if turn is None or turn.member_id != message.author.id:
            return
        if not self.bot.store.spoken_moves_enabled(guild.id):
            self.examples.pop(guild.id, None)
            return
        if not self.bot.store.spoken_member_on(guild.id, message.author.id):
            return
        self.latest[key] = number = self.latest.get(key, 0) + 1
        self._spawn(self._check(key, turn, number, message, text))

    def turn_changed(self, key) -> None:
        """Called under the table lock after every state change: a check for an earlier turn is deleted."""
        turn = self.bot.games.turns.get(key)
        reading = self.readings.get(key)
        if reading is not None and reading.turn != turn:
            del self.readings[key]
            if self.latest.get(key) == reading.number:  # a newer phrase still being read means the member had more to say
                self.ended[key] = reading
        if turn is None:
            self.latest.pop(key, None)
        check = self.pending.get(key)
        if check is None:
            return
        if turn is None or (turn.round_id, turn.moves) != (check.round_id, check.moves):
            del self.pending[key]
            self._unpressed(key, check)
            self._spawn(self._delete(check.message))

    def forget(self, key) -> None:
        """The table is gone: nothing of it stays in memory."""
        self.readings.pop(key, None)
        self.ended.pop(key, None)

    async def close(self) -> None:
        self.closed = True
        tasks = [t for t in self.tasks if not t.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.pending.clear()
        self.readings.clear()
        self.ended.clear()
        self.examples.clear()
        self.example_locks.clear()
        close = getattr(self.runtime, "close", None)
        if close is not None:
            close()
        for line in self.stats.lines():
            logging.info(line)
        veto, self.veto = self.veto, None
        if veto is not None:
            try:
                await veto.close()
            except Exception:
                logging.warning("Could not close the veto client")

    # --- the reading --------------------------------------------------------------------------

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro, name="spoken-move")
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def _runtime(self):
        if self.runtime is None:
            from .spoken import onnx_embedder  # not imported until a server has the switch on and a turn member writes
            self.runtime = onnx_embedder.get_runtime()
        return self.runtime

    def _warn_once(self, text) -> None:
        if text not in self._warned:
            self._warned.add(text)
            logging.warning(text)

    def _count(self, guild, name) -> None:
        """One counter, count-only. A day change logs the finished days first; a suspicious veto rate is warned about once per server and day."""
        for line in self.stats.rollover():
            logging.info(line)
        day = self.stats.today()
        self._suspicious = {w for w in self._suspicious if w[1] == day}
        self.stats.record(guild, name)
        if (guild, day) not in self._suspicious and self.stats.suspicious(guild, day):
            self._suspicious.add((guild, day))
            logging.warning("Spoken-move veto rate looks suspicious (guild=%s day=%s)", guild, day)

    def _unpressed(self, key, check) -> None:
        """A check of a shadow-vetoed reading that ended without a move press."""
        if check is not None and check.dropped:
            self._count(key[0], "unpressed_vetoed")

    def _current(self, key, turn, number) -> bool:
        return (not self.closed and self.latest.get(key) == number and self.bot.games.turns.get(key) == turn
                and self.bot.store.spoken_member_on(key[0], turn.member_id))  # MNT-62: a member who turned it Off mid-reading gets nothing posted or kept

    async def _check(self, key, turn, number, message, text) -> None:
        try:
            await self._read(key, turn, number, message, text)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("Spoken-move check failed (table %s)", key[1])

    async def _read(self, key, turn, number, message, text) -> None:
        from .spoken.onnx_embedder import ModelError
        try:
            head = spoken_head.blackjack_head()
        except spoken_head.HeadError as error:
            self._warn_once(f"Spoken moves are unavailable: {error}")
            return
        runtime = self._runtime()
        if not await runtime.ready():
            self._warn_once(RUNTIME_OFF)
            return
        if not self._current(key, turn, number):
            return  # superseded while the model loaded: no model work
        examples = await self._examples(key[0], runtime)
        if not self._current(key, turn, number):
            return
        try:
            vector = await runtime.embed(text)  # the runtime puts the e5 "query: " prefix on itself (head.prefix is the same string)
        except ModelError:
            return
        if not self._current(key, turn, number):
            return
        games, store = self.bot.games, self.bot.store
        try:
            snap = store.round_snapshot(key[0], turn.round_id)
        except GameError:
            return
        if snap["table_id"] != key[1] or snap["status"] != "playing" or snap["moves"] != turn.moves or not snap["legal"]:
            return
        seat, legal = next(iter(snap["legal"].items()))
        if snap["seats"][seat]["kind"] != "member" or snap["seats"][seat]["ref_id"] != turn.member_id:
            return
        decision = interpret.decide(vector, [m for m in legal if m in spoken_bj.MOVES], head, examples)
        f = decision.features
        scores = {k: round(getattr(f, k), 4) for k in ("p1", "margin", "talk_gap")} if f else {}
        self.readings[key] = _Reading(turn, text, decision.band, decision.moves, scores, number)
        if decision.band == "silent":
            return
        band, moves, dropped = await self._veto(key, turn, number, message, text, legal, decision)
        if band == "silent" or not self._current(key, turn, number):
            return
        cost = snap["view"]["stakes"][seat] // 2
        t = discord_locale.channel_t(message.guild)  # the check is posted in the channel
        all_names = _names(t)
        labels = {**all_names, "insure": t.p("button.blackjack", "Insure ({cost:,})", cost=cost)}
        names = [all_names[m] for m in moves]
        bold = [f"**{n}**" for n in names]
        question = t("Did you mean {first} or {second}?", first=bold[0], second=bold[1]) if len(bold) == 2 else t("Did you mean {moves}?", moves=" or ".join(bold))
        async with games.runtime.lock(key):  # no HTTP in here
            if not self._current(key, turn, number):
                return
            old = self.pending.pop(key, None)
        self._unpressed(key, old)
        if old is not None:
            self._spawn(self._delete(old.message))  # a newer phrase replaces the check
        view = discord.ui.View(timeout=None)
        for move in moves:
            view.add_item(SpokenButton("move", key[1], turn.round_id, turn.moves, move, labels[move], discord.ButtonStyle.primary))
        view.add_item(SpokenButton("talk", key[1], turn.round_id, turn.moves, None, t.p("button.spoken", "Just table talk"), discord.ButtonStyle.secondary))
        try:
            posted = await message.reply(question, view=view, mention_author=False, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning("Could not post a spoken-move check: %s", type(error).__name__)
            return
        if not self._current(key, turn, number):  # the turn moved on or a newer phrase arrived while posting
            await self._delete(posted)
            return
        self.pending[key] = _Check(posted, turn.member_id, turn.round_id, turn.moves, text, moves, band, scores, number, dropped)

    async def _veto(self, key, turn, number, message, text, legal, decision):
        """(band, moves, dropped) after the optional decision-model veto; e5's own reading whenever it is off, limited, slow or failing.
        In shadow mode the reader is asked and counted but e5's reading is returned, with ``dropped`` the moves the reader would have dropped (else None)."""
        veto, store, guild = self.veto, self.bot.store, key[0]
        if (veto is None or veto.breaker_open or not store.spoken_veto_enabled(guild) or not store.spoken_veto_member_on(guild, turn.member_id)
                or not store.spoken_moves_enabled(guild) or not store.spoken_member_on(guild, turn.member_id)
                or DISCORD_TOKEN.search(text) or any(getattr(message, a, None) for a in ("raw_mentions", "raw_role_mentions", "raw_channel_mentions"))
                or not self._current(key, turn, number)):
            return decision.band, decision.moves, None
        member = (guild, turn.member_id)
        if member not in self.veto_notified:  # nothing leaves before the member has been told, and this message is not sent
            if member not in self.veto_notice_pending:
                await self._notify(message, guild, turn.member_id)
            return decision.band, decision.moves, None
        if not self.limiter.allow(guild):
            return decision.band, decision.moves, None
        self._count(guild, "offered")
        started = time.monotonic()
        try:
            probs = await asyncio.wait_for(veto.ask(text, [m for m in legal if m in spoken_bj.MOVES]), self.veto_timeout)
            result = spoken_veto.apply_policy(decision.band, decision.moves, probs)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._count(guild, "errors")
            logging.warning("Spoken-move veto unavailable (%s) after %.0f ms", type(error).__name__, (time.monotonic() - started) * 1000)
            return decision.band, decision.moves, None
        logging.info("Spoken-move veto %s -> %s in %.0f ms", decision.band, result[0], (time.monotonic() - started) * 1000)
        dropped = tuple(m for m in decision.moves if m not in result[1])
        if dropped:
            self._count(guild, "vetoed")
            if result[0] == "silent":
                self._count(guild, "silenced")
        return (decision.band, decision.moves, dropped) if self.shadow else (*result, None)

    async def _notify(self, message, guild, member_id) -> None:
        """The one-line notice (channel language, rendered now), once per member and server per process; marked only once posted, a failed post is tried again with a later message."""
        member = (guild, member_id)
        self.veto_notice_pending.add(member)
        try:
            await message.reply(discord_locale.channel_t(message.guild)(VETO_NOTICE), mention_author=False, allowed_mentions=NO_MENTIONS)
            self.veto_notified.add(member)
        except Exception as error:
            logging.warning("Could not post the decision-model check notice: %s", type(error).__name__)
        finally:
            self.veto_notice_pending.discard(member)

    @staticmethod
    async def _delete(message) -> None:
        try:
            await message.delete()
        except discord.HTTPException as error:
            logging.warning("Could not delete a spoken-move check: %s", type(error).__name__)
        except Exception:
            logging.exception("Could not delete a spoken-move check")

    async def _examples(self, guild_id, runtime) -> list:
        """The server's example phrases as (vector, label), re-embedding only ids new since the cached revision."""
        store = self.bot.store
        cached = self.examples.get(guild_id)
        if cached is not None and cached[0] == store.spoken_examples_revision(guild_id):
            return [(v, l) for _, v, l in cached[1]]
        if cached is None and store.spoken_examples_revision(guild_id) == 0:
            return []
        from .spoken.onnx_embedder import ModelError
        async with self.example_locks.setdefault(guild_id, asyncio.Lock()):  # one refill per guild at a time
            revision = store.spoken_examples_revision(guild_id)
            cached = self.examples.get(guild_id)
            if cached is not None and cached[0] == revision:
                return [(v, l) for _, v, l in cached[1]]
            known = {i: v for i, v, _ in cached[1]} if cached is not None else {}
            rows, out, complete = store.spoken_examples(guild_id, "blackjack"), [], True
            for row in rows:
                vector = known.get(row["id"])
                if vector is None:
                    try:
                        vector = await runtime.embed(row["text"])
                    except ModelError:
                        complete = False
                        continue
                out.append((row["id"], vector, row["label"]))
            if self.closed:
                return [(v, l) for _, v, l in out]
            if not rows and revision == 0:
                self.examples.pop(guild_id, None)
            else:
                self.examples[guild_id] = (revision if complete else -1, out)  # -1: look again next time
            return [(v, l) for _, v, l in out]

    def _keep(self, guild_id, user_id, text, offered, band, scores, kind, label) -> None:
        """Part F capture. Call only after the move's transaction has committed and with no table lock work open; never raises."""
        try:
            self.bot.store.keep_spoken_phrase(guild_id, user_id, "blackjack", text, offered, band, scores, kind, label)
        except Exception:
            logging.exception("Could not keep a spoken-move phrase")

    def claim_table_move(self, key, round_id, moves_before, member_id, move):
        """Called by GameTables.press under the table lock after a member's own table button played a move. Memory only: pops the one-slot
        ``ended`` (filled by turn_changed, since the turn index is already cleared) and returns the args for ``keep_claimed``, or None."""
        if move not in spoken_bj.MOVES:
            return None
        reading = self.ended.pop(key, None)
        if reading is None:
            return None
        t = reading.turn
        if (t.round_id, t.moves, t.member_id) != (round_id, moves_before, member_id):
            return None
        if reading.band != "silent" and move in reading.offered:
            return None
        return (key[0], member_id, reading.text, reading.offered, reading.band, reading.scores, "false_negative", move)

    def keep_claimed(self, claim) -> None:
        """The store write for a claim. Call after the press was answered and the table lock is released."""
        if claim is not None:
            self._keep(*claim)

    # --- buttons ------------------------------------------------------------------------------

    @staticmethod
    async def _say(interaction, text) -> None:
        text = discord_locale.invoker_t(interaction).render(text)
        try:
            if interaction.response.is_done():
                await interaction.followup.send(text, ephemeral=True, allowed_mentions=NO_MENTIONS)
            else:
                await interaction.response.send_message(text, ephemeral=True, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning("Spoken-move reply failed: %s", type(error).__name__)

    async def press(self, interaction, action, table_id, round_id, moves, move) -> None:
        key = (interaction.guild_id, table_id)
        pressed = getattr(interaction, "message", None)
        found = self.pending.get(key)
        mine = found if found is not None and pressed is not None and found.message.id == pressed.id else None  # a lingering older check is not the table's current one
        if mine is not None and interaction.user.id != mine.member_id:
            await self._say(interaction, NOT_YOURS)
            return
        if mine is not None:
            del self.pending[key]  # claimed: the turn change this press causes must not delete the message we are about to edit
        if action == "talk":
            if mine is None:
                await self._say(interaction, STALE)
                if pressed is not None:
                    await self._delete(pressed)
                return
            reading = self.readings.get(key)
            if reading is not None and reading.number == mine.number:
                del self.readings[key]  # answered: a later table-button move must not also count as a miss for this phrase
            self._unpressed(key, mine)
            await self._say(interaction, TALK_KEPT)
            await self._delete(mine.message)
            self._keep(key[0], mine.member_id, mine.text, mine.offered, mine.band, mine.scores, "false_positive", "talk")
            return
        refused = await self.bot.games.play_spoken(interaction, table_id, round_id, moves, move)
        if mine is not None and refused is None and mine.dropped is not None:
            self._count(key[0], "pressed_vetoed" if move in mine.dropped else "pressed_ok")
        if mine is not None and refused is None and mine.band == "unsure" and move != mine.offered[0]:
            self._keep(key[0], mine.member_id, mine.text, mine.offered, mine.band, mine.scores, "false_positive", move)
        try:
            if refused is None:
                shown = discord_locale.channel_t(interaction.guild)  # the check is a channel message
                await interaction.edit_original_response(content=shown("✔ {move}", move=_names(shown)[move]), view=None)
            elif mine is not None or refused in (STALE, CLOSED, OFF_HERE, BLACKJACK_OFF):
                await interaction.delete_original_response()  # a bystander's refusal leaves the check alone
        except Exception as error:
            logging.warning("Could not update a spoken-move check: %s", type(error).__name__)
