"""Characterization tests for scene/state lifecycle in Store, Engine and the bot loop.

Confirmed defects follow the ``test_known_defect_*`` / ``expectedFailure`` convention (docs/engineering/audit.md).
"""
import asyncio
import unittest
from unittest.mock import patch

from discord.ext import commands

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

    def test_delete_subtree_keeps_ancestors_and_sibling_branches(self):
        """UX-09 (D5): deleting a node removes it and its descendants only, with their summaries and traces."""
        last = self.chain(3)
        self.store.record_node(2000, 1, 100, 1000, 10, None, "other user's branch")
        self.store.save_summary(last, "summary")
        self.store.save_trace(last, {"lore": []})
        self.assertEqual(self.store.delete_subtree(1, 1001), 2)
        self.assertIsNotNone(self.store.node(1000))
        self.assertIsNotNone(self.store.node(2000))
        self.assertIsNone(self.store.node(1001))
        self.assertIsNone(self.store.node(last))
        self.assertIsNone(self.store.summary(last))
        self.assertIsNone(self.store.trace(last))

    def test_delete_subtree_of_root_removes_whole_tree_including_other_branches(self):
        """UX-09 (D5): deleting the root removes every branch under it."""
        last = self.chain(3)
        self.store.record_node(2000, 1, 100, 1000, 10, None, "other user's branch")
        self.store.save_summary(last, "summary")
        self.store.save_trace(last, {"lore": []})
        self.assertEqual(self.store.delete_subtree(1, 1000), 4)
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
        """SEC-06: archiving through the wrong guild raises (like update_character) and archives nothing."""
        with self.assertRaisesRegex(ValueError, "Character not found in this server"):
            self.store.archive_character(2, self.alice, True)
        self.assertEqual(self.store.character_by_id(self.alice)["archived"], 0)

    def test_archive_from_other_guild_leaves_thread_casts_alone(self):
        """SEC-06: a wrong-guild archive must not strip the character from another server's thread cast."""
        self.store.set_cast(100, None, [self.alice], default=True)
        self.store.set_cast(101, 100, [self.alice])
        with self.assertRaises(ValueError):
            self.store.archive_character(2, self.alice, True)
        self.assertEqual(self.store.get_cast(101, 100), [self.alice])
        self.assertEqual(self.store.get_cast(100), [self.alice])

    def test_archive_from_other_guild_of_archived_character_leaves_thread_casts_alone(self):
        """SEC-06: the cast scan must not run for a wrong guild even when the character is already archived."""
        self.store.set_cast(100, None, [self.alice], default=True)
        self.store.set_cast(101, 100, [self.alice])
        self.store.db.execute("UPDATE characters SET archived=1 WHERE id=?", (self.alice,))
        with self.assertRaises(ValueError):
            self.store.archive_character(2, self.alice, True)
        self.assertEqual(self.store.get_cast(101, 100), [self.alice])

    def test_archive_unknown_character_raises(self):
        """SEC-06: an unknown character id raises the same not-found error."""
        with self.assertRaisesRegex(ValueError, "Character not found in this server"):
            self.store.archive_character(1, 99999, True)

    def test_archive_in_own_guild_removes_character_from_thread_casts(self):
        """SEC-06: the guild-scoped archive still prunes the character from thread casts."""
        self.store.set_cast(100, None, [self.alice], default=True)
        self.store.set_cast(101, 100, [self.alice])
        self.store.archive_character(1, self.alice, True)
        self.assertEqual(self.store.get_cast(101, 100), [])

    def test_rerecording_node_keeps_its_summary(self):
        """BUG-04 (fixed): re-recording an existing node updates it in place, so its summary row survives."""
        last = self.chain(2)
        self.store.save_summary(last, "kept")
        self.store.record_node(last, 1, 100, 1000, None, self.alice, "edited line")
        self.assertEqual(self.store.summary(last), "kept")

    def test_rerecording_node_last_write_wins_for_its_columns(self):
        """Re-recording a node overwrites its own columns (last write wins) and keeps its place in the chain.

        The trace is unaffected too (it has no FK to nodes, so it was never at risk from BUG-04).
        """
        last = self.chain(2)
        before = self.store.node(last)
        self.store.save_trace(last, {"speaker": "alice"})
        self.store.record_node(last, 1, 100, 1000, None, self.alice, "edited line")
        node = self.store.node(last)
        self.assertEqual(node["content"], "edited line")
        self.assertEqual(node["character_id"], self.alice)
        self.assertEqual(node["parent_id"], 1000)
        self.assertEqual(node["root_id"], 1000)
        self.assertEqual((node["parent_id"], node["root_id"]), (before["parent_id"], before["root_id"]))
        self.assertEqual(self.store.trace(last), {"speaker": "alice"})


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

    # BUG-03 (fixed): the over-30-node characterization moved to test_memory_cadence.SummaryWindowTests.


class CleanupLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_loop_survives_a_failed_pass(self):
        """BUG-02 (fixed): the cleanup loop logs a failed pass and retries next cycle."""
        bot = SkitBot(make_settings())
        calls = []

        def failing(days):
            calls.append(days)
            raise RuntimeError("database is locked")

        async def no_sleep(_seconds):
            if len(calls) >= 2:
                raise asyncio.CancelledError
        try:
            with patch.object(bot.store, "expire_history", failing), patch("asyncio.sleep", no_sleep), \
                    self.assertLogs(level="ERROR") as logs:
                with self.assertRaises(asyncio.CancelledError):
                    await bot._cleanup_loop()
            self.assertEqual(len(calls), 2, "loop should retry on the next cycle")
            failures = [record for record in logs.records if record.getMessage() == "History cleanup failed"]
            self.assertEqual(len(failures), 2, "each failed pass should be logged")
            self.assertTrue(all(record.exc_info and record.exc_info[0] is RuntimeError for record in failures))
        finally:
            bot.store.close()

    async def test_close_closes_store_when_models_close_raises(self):
        """REL-05 (fixed): if models.close() raises, close() still cancels cleanup, closes the store and the
        Discord client, then re-raises the model error."""
        bot = SkitBot(make_settings())
        real_store_close = bot.store.close
        store_closed = []
        parent_closed = []

        async def failing_models_close():
            raise RuntimeError("provider shutdown failed")

        async def parent_close(_self):
            parent_closed.append(True)
        bot.cleanup_task = asyncio.create_task(asyncio.sleep(3600))
        try:
            with patch.object(bot.models, "close", failing_models_close), \
                    patch.object(bot.store, "close", lambda: store_closed.append(True)), \
                    patch.object(commands.Bot, "close", parent_close):
                with self.assertRaises(RuntimeError):
                    await bot.close()
            await asyncio.sleep(0)
            self.assertTrue(bot.cleanup_task.cancelled(), "cleanup task should be cancelled")
            self.assertEqual(store_closed, [True], "store should be closed even if models.close() raises")
            self.assertEqual(parent_closed, [True], "discord client close should still run")
        finally:
            if not bot.cleanup_task.done():
                bot.cleanup_task.cancel()
            real_store_close()


if __name__ == "__main__":
    unittest.main()
