"""``/admin scene delete`` removes one stored node and its descendants, not the whole tree (UX-09, decision D5, R4 step 5c).

Tree used throughout (message ids)::

    1000 -> 1001 -> 1002 -> 1003      (the branch that gets deleted from 1002)
                 \\-> 2000            (sibling branch, kept)
"""
import unittest

from helpers import FakeInteraction, invoke, make_settings

from llmcord_core.discord_bot import SkitBot
from llmcord_core.store import Store


def seed(store, alice):
    store.record_node(1000, 1, 100, None, 9, None, "start", created_at=1.0)
    store.record_node(1001, 1, 100, 1000, None, alice, "reply", created_at=2.0)
    store.record_node(1002, 1, 100, 1001, 9, None, "branch a", created_at=3.0)
    store.record_node(1003, 1, 100, 1002, None, alice, "branch a reply", created_at=4.0)
    store.record_node(2000, 1, 100, 1001, 10, None, "sibling branch", created_at=5.0)
    for ident in (1000, 1001, 1002, 1003, 2000):
        store.save_summary(ident, f"summary {ident}")
        store.save_trace(ident, {"lore": [ident]})
        store.save_lore_activations(ident, [f"key{ident}"])
        store.add_candidate(1, "channel", 100, f"fact {ident}", ident, ident)


def evidence_sources(store):
    return sorted(r["source_message_id"] for r in store.db.execute("SELECT source_message_id FROM evidence"))


def activation_nodes(store):
    return sorted(r["node_id"] for r in store.db.execute("SELECT node_id FROM lore_activations"))


class DeleteSubtreeStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.world = self.store.create_space(1, "World", "world")
        self.store.bind_channel(1, 100, self.world)
        self.alice = self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])
        seed(self.store, self.alice)

    def tearDown(self):
        self.store.close()

    def test_mid_branch_delete_keeps_ancestors_and_sibling_branch(self):
        """UX-09 (D5): deleting a mid-branch node removes it and its descendants only, and returns that count."""
        self.assertEqual(self.store.delete_subtree(1, 1002), 2)
        for gone in (1002, 1003):
            self.assertIsNone(self.store.node(gone))
        for kept in (1000, 1001, 2000):
            self.assertIsNotNone(self.store.node(kept))

    def test_leaf_delete_returns_one(self):
        """UX-09 (D5): a leaf deletes just itself."""
        self.assertEqual(self.store.delete_subtree(1, 1003), 1)
        self.assertIsNotNone(self.store.node(1002))

    def test_root_delete_removes_whole_tree(self):
        """UX-09 (D5): deleting the root removes every branch (same result as the old whole-scene delete)."""
        self.assertEqual(self.store.delete_subtree(1, 1000), 5)
        for ident in (1000, 1001, 1002, 1003, 2000):
            self.assertIsNone(self.store.node(ident))
        self.assertEqual(evidence_sources(self.store), [])
        self.assertEqual(activation_nodes(self.store), [])

    def test_dependent_rows_follow_deleted_nodes_only(self):
        """UX-09 (D5): summaries, traces, lore activations and evidence of deleted nodes go; kept nodes' remain."""
        self.store.delete_subtree(1, 1002)
        for gone in (1002, 1003):
            self.assertIsNone(self.store.summary(gone))
            self.assertIsNone(self.store.trace(gone))
        for kept in (1000, 1001, 2000):
            self.assertEqual(self.store.summary(kept), f"summary {kept}")
            self.assertEqual(self.store.trace(kept), {"lore": [kept]})
        self.assertEqual(activation_nodes(self.store), [1000, 1001, 2000])
        self.assertEqual(evidence_sources(self.store), [1000, 1001, 2000])

    def test_unpromoted_candidates_without_evidence_are_pruned(self):
        """UX-09 (D5): candidates whose only evidence was deleted disappear, like delete_scene did."""
        self.store.delete_subtree(1, 1002)
        contents = sorted(r["content"] for r in self.store.db.execute("SELECT content FROM candidates"))
        self.assertEqual(contents, ["fact 1000", "fact 1001", "fact 2000"])

    def test_other_guild_node_is_untouched(self):
        """UX-09 (D5): guild scoping; a node in another guild is never deleted and the count is 0."""
        self.store.record_node(7000, 2, 500, None, 9, None, "other guild", created_at=1.0)
        self.store.save_summary(7000, "other summary")
        self.assertEqual(self.store.delete_subtree(1, 7000), 0)
        self.assertIsNotNone(self.store.node(7000))
        self.assertEqual(self.store.summary(7000), "other summary")
        self.assertEqual(self.store.delete_subtree(2, 1000), 0)
        self.assertIsNotNone(self.store.node(1000))

    def test_missing_node_returns_zero(self):
        """UX-09 (D5): an unknown id deletes nothing."""
        self.assertEqual(self.store.delete_subtree(1, 424242), 0)
        self.assertIsNotNone(self.store.node(1000))


class SceneDeleteCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.world = self.store.create_space(1, "World", "world")
        self.store.bind_channel(1, 100, self.world)
        self.alice = self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])
        seed(self.store, self.alice)

    def tearDown(self):
        self.store.close()

    async def test_deletes_given_node_and_descendants_with_count(self):
        """UX-09 (D5): the command deletes the chosen node's subtree, keeps the rest, and states the count."""
        interaction = FakeInteraction(admin=True)
        await invoke(self.bot, "admin scene delete", interaction, "1002")
        self.assertEqual(len(interaction.replies), 1)
        self.assertIn("2", interaction.replies[0])
        self.assertEqual(interaction.replies[0], "Deleted 2 stored messages from this scene.")
        self.assertIsNone(self.store.node(1002))
        self.assertIsNone(self.store.node(1003))
        for kept in (1000, 1001, 2000):
            self.assertIsNotNone(self.store.node(kept))

    async def test_root_id_deletes_whole_tree(self):
        """UX-09 (D5): giving the root still deletes everything in the scene."""
        interaction = FakeInteraction(admin=True)
        await invoke(self.bot, "admin scene delete", interaction, "1000")
        self.assertIn("5", interaction.replies[0])
        for ident in (1000, 1001, 1002, 1003, 2000):
            self.assertIsNone(self.store.node(ident))

    async def test_non_numeric_id_is_rejected(self):
        """UX-09 (D5): a non-numeric id gets the numeric-id error and deletes nothing."""
        interaction = FakeInteraction(admin=True)
        await invoke(self.bot, "admin scene delete", interaction, "abc")
        self.assertEqual(len(interaction.replies), 1)
        self.assertIn("numeric", interaction.replies[0])
        self.assertIn("message ID", interaction.replies[0])
        self.assertIsNotNone(self.store.node(1002))

    async def test_node_from_another_channel_or_guild_is_not_found(self):
        """UX-09 (D5): nodes outside this guild/channel report not-found and are untouched."""
        self.store.record_node(7000, 2, 500, None, 9, None, "other guild", created_at=1.0)
        self.store.record_node(7100, 1, 101, None, 9, None, "other channel", created_at=1.0)
        for ident in ("7000", "7100", "999999"):
            interaction = FakeInteraction(admin=True)
            await invoke(self.bot, "admin scene delete", interaction, ident)
            self.assertEqual(len(interaction.replies), 1)
            self.assertIn("not found", interaction.replies[0])
        self.assertIsNotNone(self.store.node(7000))
        self.assertIsNotNone(self.store.node(7100))
        self.assertIsNotNone(self.store.node(1002))

    async def test_requires_administrator(self):
        """UX-09 (D5): the admin check is unchanged."""
        interaction = FakeInteraction(admin=False)
        await invoke(self.bot, "admin scene delete", interaction, "1002")
        self.assertEqual(interaction.replies, ["Only server administrators can use that command."])
        self.assertIsNotNone(self.store.node(1002))


if __name__ == "__main__":
    unittest.main()
