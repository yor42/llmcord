"""Spoken blackjack moves on Discord (FEAT-37 part D, D32/D34/D35): a move check offered when the member on turn writes in the table's channel.

Off unless the server switch is on. The in-memory turn index in ``GameTables`` (``turns``, ``turn_channels``) decides, without the store or the
model, whether a message is worth reading at all. The reading runs in a tracked background task; ``on_message`` never waits for it.
Buttons go through the normal press path (``GameTables.play_spoken``), so the table lock and the ``moves`` stale check apply as for the table's own buttons.
No Discord HTTP is done while a table lock is held. The turn timer is not paused.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import discord

from .admin_store import GameError
from .games_discord import BLACKJACK_OFF, CLOSED, NO_MENTIONS, OFF_HERE, STALE
from .spoken import blackjack as spoken_bj
from .spoken import head as spoken_head
from .spoken import interpret

MAX_CHARS = 200  # longer messages are chat, not a move: ignored before any inference
PREFIX = "llmcord:bjs"
NOT_YOURS = "This isn't your move check."
TALK_KEPT = "Got it — left as table talk."
RUNTIME_OFF = "Spoken moves are unavailable: the reading model is not ready."


@dataclass
class _Check:
    """The one pending move check of a table."""
    message: object
    member_id: int
    round_id: int
    moves: int
    text: str
    offered: tuple


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
    def __init__(self, bot):
        self.bot = bot
        self.runtime = None  # SpokenRuntime; fetched lazily on the first enabled message (tests set a stub)
        self.pending: dict[tuple, _Check] = {}
        self.latest: dict[tuple, int] = {}  # table key -> number of the newest phrase; older readings drop themselves
        self.tasks: set[asyncio.Task] = set()
        self.closed = False
        self._warned: set[str] = set()

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
            return
        self.latest[key] = number = self.latest.get(key, 0) + 1
        self._spawn(self._check(key, turn, number, message, text))

    def turn_changed(self, key) -> None:
        """Called under the table lock after every state change: a check for an earlier turn is deleted."""
        turn = self.bot.games.turns.get(key)
        if turn is None:
            self.latest.pop(key, None)
        check = self.pending.get(key)
        if check is None:
            return
        if turn is None or (turn.round_id, turn.moves) != (check.round_id, check.moves):
            del self.pending[key]
            self._spawn(self._delete(check.message))

    async def close(self) -> None:
        self.closed = True
        tasks = [t for t in self.tasks if not t.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.pending.clear()
        close = getattr(self.runtime, "close", None)
        if close is not None:
            close()

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

    def _current(self, key, turn, number) -> bool:
        return not self.closed and self.latest.get(key) == number and self.bot.games.turns.get(key) == turn

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
        decision = interpret.decide(vector, [m for m in legal if m in spoken_bj.MOVES], head)
        if decision.band == "silent":
            return
        cost = snap["view"]["stakes"][seat] // 2
        labels = {**spoken_bj.LABELS, "insure": f"Insure ({cost:,})"}
        names = [spoken_bj.LABELS[m] for m in decision.moves]
        question = "Did you mean " + " or ".join(f"**{n}**" for n in names) + "?"
        async with games.runtime.lock(key):  # no HTTP in here
            if not self._current(key, turn, number):
                return
            old = self.pending.pop(key, None)
        if old is not None:
            self._spawn(self._delete(old.message))  # a newer phrase replaces the check
        view = discord.ui.View(timeout=None)
        for move in decision.moves:
            view.add_item(SpokenButton("move", key[1], turn.round_id, turn.moves, move, labels[move], discord.ButtonStyle.primary))
        view.add_item(SpokenButton("talk", key[1], turn.round_id, turn.moves, None, "Just table talk", discord.ButtonStyle.secondary))
        try:
            posted = await message.reply(question, view=view, mention_author=False, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning("Could not post a spoken-move check: %s", type(error).__name__)
            return
        if not self._current(key, turn, number):  # the turn moved on or a newer phrase arrived while posting
            await self._delete(posted)
            return
        self.pending[key] = _Check(posted, turn.member_id, turn.round_id, turn.moves, text, decision.moves)

    @staticmethod
    async def _delete(message) -> None:
        try:
            await message.delete()
        except discord.HTTPException as error:
            logging.warning("Could not delete a spoken-move check: %s", type(error).__name__)
        except Exception:
            logging.exception("Could not delete a spoken-move check")

    def _note_outcome(self, guild_id, text, offered, outcome) -> None:
        """Seam for false-positive / false-negative capture (part F). Nothing is stored yet."""

    # --- buttons ------------------------------------------------------------------------------

    @staticmethod
    async def _say(interaction, text) -> None:
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
            self._note_outcome(key[0], mine.text, mine.offered, "talk")
            await self._say(interaction, TALK_KEPT)
            await self._delete(mine.message)
            return
        refused = await self.bot.games.play_spoken(interaction, table_id, round_id, moves, move)
        if mine is not None:
            self._note_outcome(key[0], mine.text, mine.offered, move if refused is None else "refused")
        try:
            if refused is None:
                await interaction.edit_original_response(content=f"✔ {spoken_bj.LABELS[move]}", view=None)
            elif mine is not None or refused in (STALE, CLOSED, OFF_HERE, BLACKJACK_OFF):
                await interaction.delete_original_response()  # a bystander's refusal leaves the check alone
        except Exception as error:
            logging.warning("Could not update a spoken-move check: %s", type(error).__name__)
