"""D22 step 1: model usage is attributed to a channel and a feature (reply, ambient, summon, memory, catchup)."""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from helpers import FakeTextChannel, MemoryModels, drain_memory_tasks, make_settings
from llmcord_core.discord_bot import SkitBot
from llmcord_core.engine import SceneContext
from llmcord_core.store import Store
from llmcord_core.usage import ModelUsage, _capture, capture_usage
import test_catchup

CHANNEL = 100


class SpyModels(MemoryModels):
    """MemoryModels that notes the active usage attribution (guild, channel, feature) of every call."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen = []

    def note(self, kind):
        captured = _capture.get()
        self.seen.append((kind, captured[0], captured[2], captured[3]) if captured else (kind, None, None, None))

    async def text(self, role, system, messages, max_tokens=None):
        self.note('text')
        return await super().text(role, system, messages, max_tokens)

    async def structured(self, role, system, messages, schema_name, schema):
        self.note(schema_name)
        return await super().structured(role, system, messages, schema_name, schema)

    async def stream_text(self, role, system, messages):
        self.note('stream')
        async for piece in super().stream_text(role, system, messages):
            yield piece


class CaptureTests(unittest.TestCase):
    def test_values_inherit_override_and_default(self):
        """D22: context inheritance and override stack correctly."""
        with capture_usage(1):
            with capture_usage():
                pass
            with capture_usage(2, 50, 'reply'):
                with capture_usage():
                    self.assertEqual(_capture.get()[0:1] + _capture.get()[2:], (2, 50, 'reply'))
                with capture_usage(channel_id=60):
                    self.assertEqual(_capture.get()[2:], (60, 'reply'))
                with capture_usage(feature='memory'):
                    self.assertEqual(_capture.get()[2:], (50, 'memory'))
            self.assertEqual(_capture.get()[2:], (None, ''))
        self.assertIsNone(_capture.get())

    def test_collect_usage_fills_channel_and_feature(self):
        """D22: collect_usage records captured channel_id and feature; defaults to None and ''."""
        from types import SimpleNamespace
        from llmcord_core.config import ModelProfile
        from llmcord_core.usage import collect_usage
        profile = ModelProfile('compatible', 'm', 16000, True, billing_tier='free')
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=2)
        with capture_usage(1, 7, 'ambient') as records:
            collect_usage('p', profile, 'dialogue', usage)
        self.assertEqual((records[0].guild_id, records[0].channel_id, records[0].feature), (1, 7, 'ambient'))
        with capture_usage(1) as records:
            collect_usage('p', profile, 'dialogue', usage)
        self.assertEqual((records[0].channel_id, records[0].feature), (None, ''))

    def test_record_model_usage_persists_both(self):
        """D22: record_model_usage persists channel_id and feature to model_usage table."""
        store = Store()
        try:
            store.record_model_usage(ModelUsage(1, 'p', 'm', 'dialogue', 1, 2, 0, 0, 0.1, 'configured', 5.0, 77, 'summon'))
            store.record_model_usage(ModelUsage(1, 'p', 'm', 'dialogue', 1, 2, 0, 0, 0.1, 'configured', 6.0))
            rows = [tuple(r) for r in store.all('SELECT channel_id,feature FROM model_usage ORDER BY id')]
            self.assertEqual(rows, [(77, 'summon'), (None, '')])
        finally:
            store.close()


class MigrationTests(unittest.TestCase):
    def test_v7_file_upgrades_keeping_rows(self):
        """D22: v7 files upgrade to v8 with channel_id and feature columns added and preserved."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.sqlite3'
            store = Store(path)
            store.record_model_usage(ModelUsage(1, 'p', 'm', 'dialogue', 1, 2, 0, 0, 0.1, 'configured', 5.0))
            store.close()
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute('DROP INDEX model_usage_guild_time')
                connection.execute('ALTER TABLE model_usage DROP COLUMN channel_id')
                connection.execute('ALTER TABLE model_usage DROP COLUMN feature')
                connection.execute('PRAGMA user_version=7')
            store = Store(path)
            try:
                self.assertEqual(store.one('PRAGMA user_version')[0], 8)
                row = store.one('SELECT guild_id,input_tokens,channel_id,feature FROM model_usage')
                self.assertEqual(tuple(row), (1, 1, None, ''))
                self.assertTrue(store.one("SELECT 1 FROM sqlite_master WHERE name='model_usage_guild_time'"))
            finally:
                store.close()
            backups = list(Path(folder).glob('*.pre-v8-*.sqlite3'))
            self.assertEqual(len(backups), 1)
            with closing(sqlite3.connect(backups[0])) as backup:
                self.assertEqual(backup.execute('PRAGMA user_version').fetchone()[0], 7)
                self.assertNotIn('feature', [r[1] for r in backup.execute('PRAGMA table_info(model_usage)')])
            Store(path).close()
            self.assertEqual(len(list(Path(folder).glob('*.pre-*'))), 1)

    def test_v9_file_is_rejected(self):
        """D22: databases newer than v8 are rejected as too recent."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'new.sqlite3'
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute('PRAGMA user_version=9')
            with self.assertRaisesRegex(ValueError, 'newer than this application'):
                Store(path)


class AttributionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.world = self.store.create_space(1, 'World', 'world')
        self.store.bind_channel(1, CHANNEL, self.world)
        self.alice = self.store.add_character(1, self.world, 'Alice', {'name': 'Alice'}, None, [])
        self.store.set_cast(CHANNEL, None, [self.alice])
        self.channel = FakeTextChannel(CHANNEL)
        self.models = self.bot.engine.models = self.bot.models = SpyModels()
        self.models.chosen = [self.alice]

    def tearDown(self):
        self.store.close()

    async def run_scene(self, **kwargs):
        scene = SceneContext(1, CHANNEL, None, self.world, 9, 1001, 'Hi Alice', None, [], [], **kwargs)
        await self.bot.run_scene(scene, self.channel)
        await drain_memory_tasks(self.bot)

    def features(self):
        return {(kind, feature) for kind, guild, channel, feature in self.models.seen if (guild, channel) == (1, CHANNEL)}

    def assert_all_attributed(self, expected):
        self.assertTrue(self.models.seen)
        for kind, guild, channel, feature in self.models.seen:
            self.assertEqual((guild, channel), (1, CHANNEL), kind)
            self.assertEqual(feature, 'memory' if kind in ('text', 'extract_memory') else expected, kind)

    async def test_reply_turn(self):
        """D22: reply turns attribute all model calls correctly, including internal memory extraction."""
        await self.run_scene()
        self.assert_all_attributed('reply')
        self.assertIn(('extract_memory', 'memory'), self.features())

    async def test_ambient_turn(self):
        """D22: ambient turns attribute all model calls to 'ambient' feature (except memory)."""
        await self.run_scene(ambient=True)
        self.assert_all_attributed('ambient')

    async def test_summon_turn(self):
        """D22: summon turns attribute all model calls to 'summon' feature (except memory)."""
        await self.run_scene(forced_character_id=self.alice)
        self.assert_all_attributed('summon')


class CatchupAttributionTests(unittest.IsolatedAsyncioTestCase):
    async def test_catchup_is_attributed(self):
        """D22: /catchup command attributes its model calls to 'catchup' feature."""
        bot = SkitBot(make_settings())
        try:
            bot.store.bind_channel(1, 100, bot.store.create_space(1, 'Harbor', 'world'))
            seen = []

            class Models(test_catchup.RecordingModels):
                async def text(self, role, system, messages, max_tokens=None):
                    captured = _capture.get()
                    seen.append((captured[0], captured[2], captured[3]))
                    return await super().text(role, system, messages, max_tokens)

            bot.models = Models()
            channel = test_catchup.FakeChannel(100, [test_catchup.recent(2, 'Duel at dawn', uid=5)])
            it = test_catchup.granted(test_catchup.FakeInteraction(channel_id=100, channel=channel, user_id=test_catchup.ME))
            await test_catchup.invoke(bot, 'catchup', it)
            self.assertEqual(seen, [(it.guild_id, 100, 'catchup')])
        finally:
            bot.store.close()


if __name__ == '__main__':
    unittest.main()
