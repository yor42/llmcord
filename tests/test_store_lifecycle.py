"""Characterization tests for scene/state lifecycle in Store, Engine and the bot loop.

``test_known_defect_*`` tests are ``expectedFailure`` for confirmed defects (docs/engineering/audit.md).
"""
import asyncio
import unittest
from unittest.mock import patch

from helpers import FakeModels, make_settings

from llmcord_core.discord_bot import SkitBot
from llmcord_core.engine import Engine
from llmcord_core.store import Store


class StoreLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.world = self.store.create_space(1, "World", "world")
        self.store.bind_channel(1, 100, self.world)
        self.alice = self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])

    def tearDown(self):
        self.store.close()

    def chain(self, count, start=1000, created_at=None):
        parent = None
        for offset in range(count):
            ident = start + offset
            self.store.record_node(ident, 1, 100, parent, 9 if offset % 2 == 0 else None,
                                   None if offset % 2 == 0 else self.alice, f"line {offset}", created_at=created_at)
            parent = ident
        return parent

    def test_ambient_counter_and_cooldown_state(self):
        self.store.set_ambient(100, True)
        self.store.count_ambient_message(100)
        self.store.count_ambient_message(100)
        self.assertEqual(self.store.ambient_state(100), (2, 0.0))
        self.store.mark_ambient_response(100, now=50.0)
        self.assertEqual(self.store.ambient_state(100), (0, 50.0))
        self.store.set_ambient(100, False)
        self.assertEqual(self.store.ambient_state(100), (0, 0.0))

    def test_reset_scene_timestamp(self):
        self.assertEqual(self.store.scene_reset_at(100), 0.0)
        self.store.reset_scene(100, now=123.0)
        self.assertEqual(self.store.scene_reset_at(100), 123.0)

    def test_delete_scene_removes_whole_tree_including_other_branches(self):
        """Characterization (UX-09): any node's root deletes every branch under that root."""
        last = self.chain(3)
        self.store.record_node(2000, 1, 100, 1000, 10, None, "other user's branch")
        self.store.save_summary(last, "summary")
        self.store.save_trace(last, {"lore": []})
        root = self.store.node(2000)["root_id"]
        self.store.delete_scene(1, root)
        self.assertIsNone(self.store.node(1000))
        self.assertIsNone(self.store.node(2000))
        self.assertIsNone(self.store.summary(last))
        self.assertIsNone(self.store.trace(last))

    def test_expire_history_keeps_personal_facts_from_deleted_nodes(self):
        """Characterization (ARCH-05): durable memories outlive their source nodes by design or omission."""
        self.chain(2, created_at=1.0)
        self.store.set_consent(1, 9, True)
        self.store.add_personal(1, 9, self.alice, "Likes tea", 1000)
        removed = self.store.expire_history(1, now=10 * 86400)
        self.assertEqual(removed, 2)
        self.assertIsNone(self.store.node(1000))
        self.assertEqual([r["content"] for r in self.store.personal(1, 9)], ["Likes tea"])

    def test_forget_personal_is_scoped_to_owner(self):
        self.store.set_consent(1, 9, True)
        self.store.add_personal(1, 9, self.alice, "Likes tea", 1)
        memory_id = self.store.personal(1, 9)[0]["id"]
        self.store.forget_personal(1, 10, memory_id)
        self.store.forget_personal(2, 9, memory_id)
        self.assertEqual(len(self.store.personal(1, 9)), 1)
        self.store.forget_personal(1, 9, memory_id)
        self.assertEqual(self.store.personal(1, 9), [])

    def test_archive_character_prunes_casts_and_eligibility(self):
        self.store.set_cast(100, None, [self.alice], default=True)
        self.store.archive_character(1, self.alice, True)
        self.assertEqual(self.store.get_cast(100), [])
        self.assertEqual(self.store.eligible_characters(1, self.world), [])
        self.store.archive_character(1, self.alice, False)
        self.assertEqual([r["id"] for r in self.store.eligible_characters(1, self.world)], [self.alice])

    def test_archive_from_other_guild_does_not_archive(self):
        self.store.archive_character(2, self.alice, True)
        self.assertEqual(self.store.character_by_id(self.alice)["archived"], 0)

    @unittest.expectedFailure
    def test_known_defect_rerecording_node_keeps_its_summary(self):
        """BUG-04: record_node uses INSERT OR REPLACE; the replace cascades and drops the node's summary."""
        last = self.chain(2)
        self.store.save_summary(last, "kept")
        self.store.record_node(last, 1, 100, 1000, None, self.alice, "edited line")
        self.assertEqual(self.store.summary(last), "kept")


class SummaryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Store()
        world = self.store.create_space(1, "World", "world")
        self.store.bind_channel(1, 100, world)
        self.engine = Engine(self.store, FakeModels(), make_settings())

    def tearDown(self):
        self.store.close()

    def chain(self, count):
        parent = None
        for ident in range(1000, 1000 + count):
            self.store.record_node(ident, 1, 100, parent, 9, None, f"line {ident}")
            parent = ident
        return parent

    async def test_short_branch_is_summarized(self):
        last = self.chain(4)
        await self.engine.summarize_scene(last)
        self.assertEqual(self.store.summary(last), "Earlier scene summary")

    async def test_branch_over_30_unsummarized_nodes_is_never_summarized(self):
        """Characterization (BUG-03, probable defect): past 30 unsummarized nodes summarization stops for good."""
        last = self.chain(31)
        await self.engine.summarize_scene(last)
        self.assertIsNone(self.store.summary(last))
        self.store.record_node(5000, 1, 100, last, 9, None, "one more")
        await self.engine.summarize_scene(5000)
        self.assertIsNone(self.store.summary(5000))


class CleanupLoopTests(unittest.IsolatedAsyncioTestCase):
    @unittest.expectedFailure
    async def test_known_defect_cleanup_loop_survives_a_failed_pass(self):
        """BUG-02: an exception from expire_history ends the daily cleanup task permanently."""
        bot = SkitBot(make_settings())
        calls = []

        def failing(days):
            calls.append(days)
            raise RuntimeError("database is locked")

        async def no_sleep(_seconds):
            if len(calls) >= 2:
                raise asyncio.CancelledError
        try:
            with patch.object(bot.store, "expire_history", failing), patch("asyncio.sleep", no_sleep):
                try:
                    await bot._cleanup_loop()
                except asyncio.CancelledError:
                    pass
                except RuntimeError:
                    pass
            self.assertEqual(len(calls), 2, "loop should retry on the next cycle")
        finally:
            bot.store.close()


if __name__ == "__main__":
    unittest.main()
