"""Spoken doubts in I Doubt It table threads (FEAT-38 part B, D36): a doubt check offered when a seated member writes a doubt in the thread.

Off unless the server switch ``spoken_doubt`` is on (and the member's own ``/game spoken_moves`` switch, shared with blackjack). The in-memory
table state of ``DoubtTables`` (``threads``, the open ``window`` deadline) decides, without the store, whether a message is worth reading at all;
the reading itself is the pure word list in ``spoken.doubt``. The check's Doubt button goes through the table's own doubt path
(``DoubtTables.press_doubt``), so the table lock, the ``moves`` stale check and the refusal texts are the table button's. No Discord HTTP is done
while a table lock is held, and the window timer is neither paused nor extended by a pending check.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import discord

from .admin_store import GameError
from .games_discord import NO_MENTIONS
from .games_spoken_discord import MAX_CHARS, NOT_YOURS, TALK_KEPT
from .spoken.doubt import read_doubt

PREFIX = "llmcord:dts"
DOUBTED = "✔ Doubt it!"


@dataclass
class _Check:
    message: object
    round_id: int
    moves: int
    number: int


class SpokenDoubtButton(discord.ui.DynamicItem[discord.ui.Button], template=PREFIX + r":(?P<action>doubt|talk):(?P<table>\d+):(?P<round>\d+):(?P<moves>\d+):(?P<user>\d+)"):
    def __init__(self, action: str, table_id: int, round_id: int, moves: int, user_id: int, label: str = "", style=discord.ButtonStyle.secondary):
        super().__init__(discord.ui.Button(label=label, style=style, custom_id=f"{PREFIX}:{action}:{table_id}:{round_id}:{moves}:{user_id}"))
        self.action, self.table_id, self.round_id, self.moves, self.user_id = action, table_id, round_id, moves, user_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["action"], int(match["table"]), int(match["round"]), int(match["moves"]), int(match["user"]), item.label or "", item.style)

    async def callback(self, interaction: discord.Interaction):
        await interaction.client.spoken_doubt.press(interaction, self.action, self.table_id, self.round_id, self.moves, self.user_id)


class SpokenDoubts:
    def __init__(self, bot):
        self.bot = bot
        self.windows: dict[tuple, tuple] = {}  # table key -> (round, moves) of the open doubt window, set by DoubtTables._after
        self.pending: dict[tuple, _Check] = {}  # (table key, member) -> that member's one check
        self.latest: dict[tuple, int] = {}  # (table key, member) -> number of the newest phrase; older posts drop themselves
        self.tasks: set[asyncio.Task] = set()
        self.closed = False

    # --- entry points -------------------------------------------------------------------------

    def note_message(self, message) -> None:
        """Called from on_message. Cheap filters first (memory only); then the store; then one background post. Never raises into the caller's flow."""
        guild = message.guild
        if self.closed or guild is None or message.author.bot or message.webhook_id or not isinstance(message.channel, discord.Thread):
            return
        tables = self.bot.doubt
        key = tables.threads.get(message.channel.id)
        table = tables.tables.get(key) if key else None
        if table is None or key[0] != guild.id or "window" not in table.deadlines:
            return
        text = (message.content or "").strip()
        if not text or len(text) > MAX_CHARS or read_doubt(text) is None:
            return
        store, user = self.bot.store, message.author.id
        if not store.spoken_doubt_enabled(guild.id) or not store.spoken_member_on(guild.id, user):
            return
        try:
            snap = store.latest_round(*key)
        except GameError:
            return
        view = snap["view"]
        if snap["status"] != "playing" or not view or not view["window"]:
            return
        seat = store.member_seat_index(guild.id, snap["round_id"], user)
        if seat is None or not view["active"][seat] or view["window"]["claimer"] == seat:
            return
        slot = (key, user)
        self.latest[slot] = number = self.latest.get(slot, 0) + 1
        self._spawn(self._post(slot, number, message, snap["round_id"], snap["moves"]))

    def window_changed(self, key, snap) -> None:
        """Called under the table lock from DoubtTables._after (and when the table is released; ``snap`` None): a check for a window that is no longer open is dropped."""
        view = snap["view"] if snap else None
        now = (snap["round_id"], snap["moves"]) if snap and snap["status"] == "playing" and view and view["window"] else None
        if now is None:
            self.windows.pop(key, None)
        else:
            self.windows[key] = now
        for slot in [s for s, c in self.pending.items() if s[0] == key and (c.round_id, c.moves) != now]:
            self._spawn(self._delete(self.pending.pop(slot).message))
        if now is None:
            for slot in [s for s in self.latest if s[0] == key]:
                del self.latest[slot]

    async def close(self) -> None:
        self.closed = True
        tasks = [t for t in self.tasks if not t.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.pending.clear()
        self.windows.clear()
        self.latest.clear()

    # --- the check ----------------------------------------------------------------------------

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro, name="spoken-doubt")
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def _current(self, slot, number, round_id, moves) -> bool:
        return not self.closed and self.latest.get(slot) == number and self.windows.get(slot[0]) == (round_id, moves)

    async def _post(self, slot, number, message, round_id, moves) -> None:
        try:
            await self._offer(slot, number, message, round_id, moves)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("Spoken-doubt check failed (table %s)", slot[0][1])

    async def _offer(self, slot, number, message, round_id, moves) -> None:
        key, user = slot
        if not self._current(slot, number, round_id, moves):
            return
        old = self.pending.pop(slot, None)
        if old is not None:
            self._spawn(self._delete(old.message))  # a newer phrase replaces the member's check
        view = discord.ui.View(timeout=None)
        view.add_item(SpokenDoubtButton("doubt", key[1], round_id, moves, user, "Doubt it!", discord.ButtonStyle.danger))
        view.add_item(SpokenDoubtButton("talk", key[1], round_id, moves, user, "Just table talk", discord.ButtonStyle.secondary))
        try:
            posted = await message.reply("Did you mean **Doubt it!**?", view=view, mention_author=False, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning("Could not post a spoken-doubt check: %s", type(error).__name__)
            return
        if not self._current(slot, number, round_id, moves):  # the window closed or a newer phrase arrived while posting
            await self._delete(posted)
            return
        replaced = self.pending.pop(slot, None)
        if replaced is not None:
            self._spawn(self._delete(replaced.message))
        self.pending[slot] = _Check(posted, round_id, moves, number)

    @staticmethod
    async def _delete(message) -> None:
        try:
            await message.delete()
        except discord.HTTPException as error:
            logging.warning("Could not delete a spoken-doubt check: %s", type(error).__name__)
        except Exception:
            logging.exception("Could not delete a spoken-doubt check")

    # --- buttons ------------------------------------------------------------------------------

    @staticmethod
    async def _say(interaction, text) -> None:
        try:
            if interaction.response.is_done():
                await interaction.followup.send(text, ephemeral=True, allowed_mentions=NO_MENTIONS)
            else:
                await interaction.response.send_message(text, ephemeral=True, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning("Spoken-doubt reply failed: %s", type(error).__name__)

    async def press(self, interaction, action, table_id, round_id, moves, user_id) -> None:
        if interaction.user.id != user_id:
            await self._say(interaction, NOT_YOURS)
            return
        slot = ((interaction.guild_id, table_id), user_id)
        pressed = getattr(interaction, "message", None)
        found = self.pending.get(slot)
        if found is not None and pressed is not None and found.message.id == pressed.id:
            del self.pending[slot]  # claimed: the window change this press causes must not delete the message we are about to edit
        if action == "talk":
            await self._say(interaction, TALK_KEPT)
            if pressed is not None:
                await self._delete(pressed)
            return
        try:
            refused = await self.bot.doubt.press_doubt(interaction, table_id, round_id, moves)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("Spoken-doubt press failed (table %s)", table_id)
            if pressed is not None:
                await self._delete(pressed)
            return
        try:
            if refused is None:
                await interaction.edit_original_response(content=DOUBTED, view=None)
            else:
                await self._say(interaction, refused)
                await interaction.delete_original_response()  # the check is the member's own: a refused one has nothing left to offer
        except Exception as error:
            logging.warning("Could not update a spoken-doubt check: %s", type(error).__name__)
