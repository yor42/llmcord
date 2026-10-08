"""Per-server usage footer switch and self-cleaning "no character" hint (UX-08; decisions D2 and D12 in
docs/engineering/roadmap.md).

D2: ``guild_settings.usage_footer INTEGER NOT NULL DEFAULT 1`` (schema v4), ``Store.usage_footer_enabled(guild_id)``
(True when there is no row) and ``Store.set_usage_footer(guild_id, enabled)``. Off hides only the public ``-# ``
line; usage is still recorded in ``model_usage``.

D12: the public "No active character is available here" hint is deleted after ``NO_CHARACTER_NOTE_SECONDS``
(module constant in ``discord_bot``, patched small here) by a background task; a failed delete is logged, not raised.

Seams: ``Store`` (files for migrations), ``SkitBot.run_scene`` with ``helpers.FakeTextChannel`` and its real
``_webhook`` path, and a model fake that reports usage through the bot's real gateway sink.
"""
import asyncio
import logging
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from helpers import (FakeModels, FakeSentMessage, FakeTextChannel, drain_memory_tasks, make_settings, not_found)

from llmcord_core.discord_bot import SkitBot
from llmcord_core.engine import SceneContext
from llmcord_core.store import Store

LONG_MODEL = "m" * 120  # a long model label makes the footer long enough to shrink the chunk limit


class StoreSettingTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()

    def tearDown(self):
        self.store.close()

    def test_footer_defaults_on_without_a_row(self):
        """UX-08 (D2): existing servers see no change."""
        self.assertIs(self.store.usage_footer_enabled(1), True)

    def test_set_usage_footer_is_guild_scoped_and_reversible(self):
        """UX-08 (D2): turning the footer off for one guild leaves others on; turning it back on restores it."""
        self.store.set_usage_footer(1, False)
        self.assertIs(self.store.usage_footer_enabled(1), False)
        self.assertIs(self.store.usage_footer_enabled(2), True)
        self.store.set_usage_footer(1, True)
        self.assertIs(self.store.usage_footer_enabled(1), True)

    def test_set_usage_footer_keeps_preset_and_asset_settings(self):
        """UX-08 (D2): the upsert does not clobber preset_id/preset_revision/asset_channel_id."""
        self.store.execute("INSERT INTO guild_settings(guild_id,preset_id,preset_revision,asset_channel_id) VALUES(1,7,3,55)")
        self.store.set_usage_footer(1, False)
        row = self.store.one("SELECT * FROM guild_settings WHERE guild_id=1")
        self.assertEqual((row["preset_id"], row["preset_revision"], row["asset_channel_id"]), (7, 3, 55))
        self.assertIs(self.store.usage_footer_enabled(1), False)

    def test_preset_activation_does_not_reset_footer(self):
        """UX-08 (D2): activating a preset after the footer was turned off leaves the footer off."""
        self.store.set_usage_footer(1, False)
        self.store.execute("INSERT INTO guild_settings(guild_id,preset_id,preset_revision) VALUES(1,0,0) "
                           "ON CONFLICT(guild_id) DO UPDATE SET preset_id=excluded.preset_id")
        self.assertIs(self.store.usage_footer_enabled(1), False)


class SchemaV4Tests(unittest.TestCase):
    def v3_file(self, folder):
        """Create a real v3-era file: a current store with the v4 column removed and user_version=3."""
        path = Path(folder) / "old.sqlite3"
        store = Store(path)
        store.execute("INSERT INTO guild_settings(guild_id,preset_id,preset_revision,asset_channel_id) VALUES(1,7,3,55)")
        store.close()
        with closing(sqlite3.connect(path)) as connection, connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(guild_settings)")}
            if "usage_footer" in columns:
                connection.execute("ALTER TABLE guild_settings DROP COLUMN usage_footer")
            connection.execute("PRAGMA user_version=3")
        return path

    def test_fresh_database_is_v4(self):
        """UX-08 (D2): a new database is created at schema version 4 with the column."""
        store = Store()
        try:
            self.assertEqual(store.one("PRAGMA user_version")[0], 4)
            self.assertIn("usage_footer", {row[1] for row in store.all("PRAGMA table_info(guild_settings)")})
        finally:
            store.close()

    def test_v3_file_upgrades_with_backup_and_keeps_settings(self):
        """UX-08 (D2): a v3 file is backed up (.pre-v4-*), upgraded to v4, keeps its guild_settings rows and the
        footer defaults to on."""
        with tempfile.TemporaryDirectory() as folder:
            path = self.v3_file(folder)
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 4)
                row = store.one("SELECT * FROM guild_settings WHERE guild_id=1")
                self.assertEqual((row["preset_id"], row["preset_revision"], row["asset_channel_id"], row["usage_footer"]),
                                 (7, 3, 55, 1))
                self.assertIs(store.usage_footer_enabled(1), True)
            finally:
                store.close()
            backups = list(Path(folder).glob("*.pre-v4-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            with closing(sqlite3.connect(backups[0])) as backup:
                self.assertEqual(backup.execute("PRAGMA user_version").fetchone()[0], 3)
                self.assertNotIn("usage_footer", {row[1] for row in backup.execute("PRAGMA table_info(guild_settings)")})
            reopened = Store(path)
            reopened.close()
            self.assertEqual(len(list(Path(folder).glob("*.pre-v4-*.sqlite3"))), 1)

    def test_newer_database_is_still_rejected(self):
        """UX-08 (D2): opening a v5 database raises 'newer than this application'."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new.sqlite3"
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("PRAGMA user_version=5")
            with self.assertRaisesRegex(ValueError, "newer than this application"):
                Store(path)


class UsageModels(FakeModels):
    """Dialogue text of ``reply`` that reports usage through the bot's real gateway (so model_usage is written)."""

    def __init__(self, speakers, gateway, reply="Hello."):
        super().__init__(speakers)
        self.gateway, self.reply = gateway, reply

    async def stream_text(self, role, system, messages):
        yield "<emotion>neutral</emotion>\n" + self.reply
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=34)
        self.gateway._usage("test", "dialogue", usage)


class FooterTurnTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings(model=LONG_MODEL))
        self.store = self.bot.store
        self.gateway = self.bot.models
        self.worlds = {}
        self.channels = {}
        for guild, channel_id in ((1, 100), (2, 200)):
            world = self.store.create_space(guild, "World", "world")
            self.store.bind_channel(guild, channel_id, world)
            alice = self.store.add_character(guild, world, f"Alice{guild}", {"name": "Alice"}, None, [])
            self.store.set_cast(channel_id, None, [alice])
            self.worlds[guild] = (world, alice)
            self.channels[guild] = FakeTextChannel(channel_id)

    def tearDown(self):
        self.store.close()

    async def turn(self, guild, reply="Hello."):
        world, alice = self.worlds[guild]
        self.bot.engine.models = self.bot.models = UsageModels([alice], self.gateway, reply)
        channel = self.channels[guild]
        await self.bot.run_scene(SceneContext(guild, channel.id, None, world, 9, 1000 + guild, "Hi", None, [], []), channel)
        await drain_memory_tasks(self.bot)
        self.assertEqual(channel.errors, [])
        return [post.content for hook in channel.hooks for post in hook.posts]

    async def test_footer_off_hides_line_for_that_guild_only(self):
        """UX-08 (D2): with the footer off for guild 1 its delivered character message has no '-# ' line; guild 2
        (default) still gets it."""
        self.store.set_usage_footer(1, False)
        off = await self.turn(1)
        on = await self.turn(2)
        self.assertEqual(off, ["Hello."])
        self.assertTrue(on[0].startswith("Hello.\n\n-# "), on)
        self.assertIn("Reply input: 120", on[0])

    async def test_footer_default_on_is_unchanged(self):
        """UX-08 (D2): a guild with no setting keeps the existing footer text."""
        posts = await self.turn(1)
        self.assertIn(f"-# {LONG_MODEL} · Reply input: 120 · Output: 34", posts[0])

    async def test_footer_off_still_records_usage(self):
        """UX-08 (D2): only the display changes; model_usage still gets the dialogue row."""
        self.store.set_usage_footer(1, False)
        await self.turn(1)
        rows = self.store.all("SELECT * FROM model_usage WHERE role='dialogue'")
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["input_tokens"], rows[0]["output_tokens"]), (120, 34))

    async def test_footer_off_does_not_leak_into_stored_content(self):
        """UX-08 (D2): stored node content never has a footer, on or off."""
        self.store.set_usage_footer(1, False)
        await self.turn(1)
        for node in self.store.all("SELECT content FROM nodes WHERE character_id IS NOT NULL"):
            self.assertNotIn("-# ", node["content"])

    async def test_chunk_limit_uses_full_length_without_footer(self):
        """UX-08 (D2): a reply that needs two chunks to leave room for a long footer fits in one chunk when the footer
        is off; every posted message stays within Discord's 2000 characters either way."""
        reply = " ".join(["word"] * 370)  # about 1850 characters
        self.assertTrue(1800 < len(reply) <= 1900)
        self.store.set_usage_footer(1, False)
        off = await self.turn(1, reply)
        on = await self.turn(2, reply)
        self.assertEqual(off, [reply])
        self.assertGreater(len(on), 1)
        self.assertTrue(all(len(content) <= 2000 for content in on + off))


class DeletableMessage(FakeSentMessage):
    """A channel message whose delete signals the test and can raise."""

    def __init__(self, ident, content, error=None):
        super().__init__(ident, content)
        self.error, self.attempted = error, asyncio.Event()

    async def delete(self):
        self.attempted.set()
        if self.error:
            raise self.error
        self.deleted = True


class NoCharacterNoteTests(unittest.IsolatedAsyncioTestCase):
    HINT = "No active character is available here"

    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.world = self.store.create_space(1, "World", "world")
        self.store.bind_channel(1, 100, self.world)  # no cast: nobody can answer
        self.bot.engine.models = self.bot.models = FakeModels([])
        self.channel = FakeTextChannel(100)
        self.messages = []
        self.delete_error = None

        async def send(content=None, **kwargs):
            message = DeletableMessage(self.channel.next_id(), content, self.delete_error)
            self.channel.sent.append(message)
            self.messages.append(message)
            return message

        self.channel.send = send

    def tearDown(self):
        self.store.close()

    def scene(self, **kwargs):
        return SceneContext(1, 100, None, self.world, 9, 1000, "Hello?", None, [], [], **kwargs)

    def hints(self):
        return [m for m in self.messages if self.HINT in m.content]

    async def test_hint_is_posted_then_deleted_in_background(self):
        """UX-08 (D12): the hint is posted publicly, run_scene returns without waiting for the delete, and the hint
        is deleted after NO_CHARACTER_NOTE_SECONDS."""
        with patch("llmcord_core.discord_bot.NO_CHARACTER_NOTE_SECONDS", 0.05, create=True):
            await self.bot.run_scene(self.scene(), self.channel)
            hints = self.hints()
            self.assertEqual(len(hints), 1)
            self.assertFalse(hints[0].deleted, "run_scene must not block on the delayed delete")
            await asyncio.wait_for(hints[0].attempted.wait(), 1)
        self.assertTrue(hints[0].deleted)

    async def test_failed_delete_is_logged_not_raised(self):
        """UX-08 (D12): a hint already deleted by someone else (NotFound) is logged at warning level; nothing raises
        out of run_scene or the background task."""
        self.delete_error = not_found("Unknown Message")
        loop_errors = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: loop_errors.append(context))
        with patch("llmcord_core.discord_bot.NO_CHARACTER_NOTE_SECONDS", 0.01, create=True):
            with self.assertLogs(level="WARNING") as logs:
                await self.bot.run_scene(self.scene(), self.channel)
                hints = self.hints()
                self.assertEqual(len(hints), 1)
                await asyncio.wait_for(hints[0].attempted.wait(), 1)
                for _ in range(10):
                    await asyncio.sleep(0)
        self.assertTrue(any(record.levelno == logging.WARNING for record in logs.records))
        self.assertEqual(loop_errors, [])

    async def test_ambient_scene_posts_no_hint(self):
        """UX-08 (D12): ambient scenes with nobody to speak still post nothing."""
        with patch("llmcord_core.discord_bot.NO_CHARACTER_NOTE_SECONDS", 0.01, create=True):
            await self.bot.run_scene(self.scene(ambient=True), self.channel)
        self.assertEqual(self.hints(), [])
        self.assertEqual([m for m in self.messages if not m.deleted], [])


if __name__ == "__main__":
    unittest.main()
