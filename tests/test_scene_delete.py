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


def personal_sources(store, guild_id=1):
    return sorted(r["source_message_id"] for r in store.db.execute(
        "SELECT source_message_id FROM personal_memories WHERE guild_id=?", (guild_id,)))


def encounter_sources(store, guild_id=1):
    return sorted(r["source_message_id"] for r in store.db.execute(
        "SELECT source_message_id FROM encounters WHERE guild_id=?", (guild_id,)))


def count(store, table):
    return store.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


class DeleteSubtreeDerivedMemoryTests(unittest.TestCase):
    """ARCH-05 / D10: delete_subtree also forgets personal facts and encounters sourced from the deleted nodes."""

    def setUp(self):
        self.store = Store()
        self.world = self.store.create_space(1, "World", "world")
        self.store.bind_channel(1, 100, self.world)
        self.alice = self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])
        seed(self.store, self.alice)
        self.store.set_consent(1, 5, True)
        for ident in (1000, 1001, 1002, 1003, 2000):
            self.store.add_personal(1, 5, self.alice, f"likes {ident}", ident)
            self.store.add_encounter(1, self.alice, self.world, f"met {ident}", ident)

    def tearDown(self):
        self.store.close()

    def test_facts_and_encounters_of_deleted_nodes_are_removed_others_kept(self):
        """ARCH-05 (D10): facts sourced from the deleted subtree go; ancestor and sibling-branch facts stay."""
        self.store.delete_subtree(1, 1002)
        self.assertEqual(personal_sources(self.store), [1000, 1001, 2000])
        self.assertEqual(encounter_sources(self.store), [1000, 1001, 2000])

    def test_root_delete_removes_all_derived_facts(self):
        """ARCH-05 (D10): deleting the root forgets every fact sourced from the scene."""
        self.store.delete_subtree(1, 1000)
        self.assertEqual(personal_sources(self.store), [])
        self.assertEqual(encounter_sources(self.store), [])

    def test_other_guild_facts_are_kept(self):
        """ARCH-05 (D10): the new deletes are guild-scoped; another guild's rows are never touched."""
        self.store.db.execute("PRAGMA foreign_keys=OFF")
        self.store.db.execute("INSERT INTO personal_memories(guild_id,user_id,character_id,content,source_message_id) VALUES(2,6,?,?,1002)", (self.alice, "other guild"))
        self.store.db.execute("INSERT INTO encounters(guild_id,character_id,space_id,content,source_message_id) VALUES(2,?,?,?,1002)", (self.alice, self.world, "other guild"))
        self.store.db.commit()
        self.store.db.execute("PRAGMA foreign_keys=ON")
        self.store.delete_subtree(1, 1002)
        self.assertEqual(personal_sources(self.store, 2), [1002])
        self.assertEqual(encounter_sources(self.store, 2), [1002])

    def test_missing_node_deletes_no_facts(self):
        """ARCH-05 (D10): an unknown id removes no facts."""
        self.store.delete_subtree(1, 424242)
        self.assertEqual(len(personal_sources(self.store)), 5)
        self.assertEqual(len(encounter_sources(self.store)), 5)

    def test_promoted_lore_stays_after_its_evidence_is_deleted(self):
        """ARCH-05 (D10): lore promoted from candidates keeps existing when its source nodes are deleted."""
        self.store.add_candidate(1, "channel", 100, "shared fact", 1000, 1002)
        self.store.add_candidate(1, "channel", 100, "shared fact", 1001, 1003)
        before = [r["content"] for r in self.store.db.execute("SELECT content FROM lore WHERE content='shared fact'")]
        self.assertEqual(before, ["shared fact"])
        self.store.delete_subtree(1, 1002)
        after = [r["content"] for r in self.store.db.execute("SELECT content FROM lore WHERE content='shared fact'")]
        self.assertEqual(after, ["shared fact"])

    def test_expire_history_keeps_facts_and_encounters(self):
        """ARCH-05 (D10): history expiry still keeps derived facts and encounters."""
        self.assertEqual(self.store.expire_history(1, now=10 * 86400), 5)
        self.assertIsNone(self.store.node(1000))
        self.assertEqual(len(personal_sources(self.store)), 5)
        self.assertEqual(len(encounter_sources(self.store)), 5)


class DeletedSourceRaceTests(unittest.TestCase):
    """ARCH-05 / D10: a background memory task finishing after its turn was deleted writes nothing."""

    def setUp(self):
        self.store = Store()
        self.world = self.store.create_space(1, "World", "world")
        self.store.bind_channel(1, 100, self.world)
        self.alice = self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])
        seed(self.store, self.alice)
        self.store.set_consent(1, 5, True)
        self.store.delete_subtree(1, 1002)

    def tearDown(self):
        self.store.close()

    def test_add_personal_for_deleted_source_writes_nothing(self):
        self.store.add_personal(1, 5, self.alice, "late fact", 1002)
        self.assertEqual(count(self.store, "personal_memories"), 0)

    def test_add_personal_for_live_source_still_writes(self):
        self.store.add_personal(1, 5, self.alice, "live fact", 1001)
        self.assertEqual(personal_sources(self.store), [1001])

    def test_add_encounter_for_deleted_source_writes_nothing(self):
        self.store.add_encounter(1, self.alice, self.world, "late", 1003)
        self.assertEqual(count(self.store, "encounters"), 0)

    def test_add_encounter_for_live_source_still_writes(self):
        self.store.add_encounter(1, self.alice, self.world, "live", 2000)
        self.assertEqual(encounter_sources(self.store), [2000])

    def test_add_candidate_for_deleted_source_writes_nothing(self):
        """No candidate, evidence or promotion even when a second mention would have promoted it."""
        lore_before, ev_before, cand_before = count(self.store, "lore"), count(self.store, "evidence"), count(self.store, "candidates")
        self.store.add_candidate(1, "channel", 100, "fact 1000", 1000, 1002)
        self.store.add_candidate(1, "channel", 100, "late new", 1000, 1003)
        self.assertEqual(count(self.store, "lore"), lore_before)
        self.assertEqual(count(self.store, "evidence"), ev_before)
        self.assertEqual(count(self.store, "candidates"), cand_before)

    def test_add_candidate_for_live_source_still_writes(self):
        self.store.add_candidate(1, "channel", 100, "fact 1000", 2000, 2000)
        self.assertEqual(count(self.store, "lore"), 1)

    def test_save_summary_for_deleted_node_is_a_noop(self):
        self.store.save_summary(1002, "late summary")
        self.assertIsNone(self.store.summary(1002))

    def test_save_summary_for_live_node_still_writes(self):
        self.store.save_summary(1001, "updated")
        self.assertEqual(self.store.summary(1001), "updated")

    def test_source_existing_only_in_another_guild_is_not_live(self):
        """The existence check is guild-scoped: guild 2's node does not make a guild-1 write valid."""
        self.store.record_node(7000, 2, 500, None, 9, None, "other guild", created_at=1.0)
        self.store.add_personal(1, 5, self.alice, "cross guild", 7000)
        self.store.add_encounter(1, self.alice, self.world, "cross guild", 7000)
        self.store.add_candidate(1, "channel", 100, "cross guild", 7000, 7000)
        self.assertEqual(count(self.store, "personal_memories"), 0)
        self.assertEqual(count(self.store, "encounters"), 0)
        self.assertEqual(count(self.store, "evidence WHERE source_message_id=7000"), 0)

    def test_consent_still_required_for_live_source(self):
        self.store.add_personal(1, 6, self.alice, "no consent", 1001)
        self.assertEqual(count(self.store, "personal_memories"), 0)


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

    async def test_reply_mentions_forgotten_personal_facts_and_encounters(self):
        """ARCH-05 / D10: the reply appends how many personal facts and encounters were forgotten."""
        self.store.set_consent(1, 5, True)
        self.store.add_personal(1, 5, self.alice, "likes tea", 1002)
        self.store.add_encounter(1, self.alice, self.world, "met at the gate", 1003)
        interaction = FakeInteraction(admin=True)
        await invoke(self.bot, "admin scene delete", interaction, "1002")
        self.assertEqual(interaction.replies[0], "Deleted 2 stored messages from this scene. Also forgot 1 personal fact and 1 encounter.")

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
