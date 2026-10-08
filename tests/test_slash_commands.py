"""Characterization tests for slash-command callbacks (see docs/engineering/audit.md).

Tests named ``test_known_defect_*`` are ``expectedFailure``: they assert the intended behavior
for an objectively incorrect current behavior and will report "unexpected success" once fixed.
Command error mapping and definite /memory forget and /admin lore delete replies (UX-07, SEC-05 per D3) are pinned in
tests/test_error_mapping.py.
"""
import unittest

from helpers import FakeInteraction, invoke, make_settings

from llmcord_core.discord_bot import SkitBot
from llmcord_core.world_info import evaluate


class SlashCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.world = self.store.create_space(1, "Harbor", "world")
        self.store.bind_channel(1, 100, self.world)
        self.alice = self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])
        self.bob = self.store.add_character(1, self.world, "Bob", {"name": "Bob"}, None, [])

    def tearDown(self):
        self.store.close()

    async def test_admin_commands_reject_non_admins_with_clear_message(self):
        interaction = FakeInteraction(admin=False)
        await invoke(self.bot, "admin space create", interaction, "world", "Elsewhere")
        self.assertEqual(interaction.replies, ["Only server administrators can use that command."])
        self.assertIsNone(self.store.space(1, "Elsewhere"))

    async def test_space_create_and_list(self):
        interaction = FakeInteraction(admin=True)
        await invoke(self.bot, "admin space create", interaction, "hub", "Plaza")
        listing = FakeInteraction()
        await invoke(self.bot, "space list", listing)
        self.assertIn("hub: Plaza", listing.replies[0])

    async def test_rebinding_channel_keeps_cast_and_ambient(self):
        """UX-03: /admin space bind to the channel's current space keeps its casts and ambient (currently it silently
        clears them). The keep/prune contract for other spaces and hub unlinks is pinned in tests/test_rebind_cast.py."""
        self.store.set_cast(100, None, [self.alice], default=True)
        self.store.set_ambient(100, True)
        channel = type("Chan", (), {"id": 100, "mention": "<#100>"})()
        await invoke(self.bot, "admin space bind", FakeInteraction(admin=True), channel, "Harbor")
        row = self.store.channel(100)
        self.assertEqual((row["default_cast"], row["active_cast"], row["ambient"]), (f"[{self.alice}]", f"[{self.alice}]", 1))

    async def test_unbound_channel_message(self):
        interaction = FakeInteraction(channel_id=999)
        await invoke(self.bot, "cast show", interaction)
        self.assertEqual(interaction.replies, ["This channel is not bound to a world or hub"])

    async def test_cast_set_is_case_insensitive(self):
        """Regression (UX-04): /cast set resolves each comma-separated name case-insensitively (unique matches).
        The shared resolver's full rules are pinned in tests/test_name_resolution.py."""
        interaction = FakeInteraction()
        await invoke(self.bot, "cast set", interaction, "alice, BOB")
        self.assertEqual(self.store.get_cast(100), [self.alice, self.bob])

    async def test_cast_add_is_case_insensitive(self):
        """UX-04: /cast add resolves a unique case-insensitive name like /cast set does (previously it needed the exact name
        and replied "Character not found")."""
        interaction = FakeInteraction()
        await invoke(self.bot, "cast add", interaction, "alice")
        self.assertEqual(self.store.get_cast(100), [self.alice], interaction.replies)

    async def test_cast_remove_and_show(self):
        self.store.set_cast(100, None, [self.alice, self.bob])
        await invoke(self.bot, "cast remove", FakeInteraction(), "Alice")
        show = FakeInteraction()
        await invoke(self.bot, "cast show", show)
        self.assertEqual(show.replies, ["Bob"])

    async def test_cast_rejects_ineligible_character(self):
        """Regression (UX-04): a character from a world not bound here is refused and the cast is unchanged. The wording
        is not pinned: previously "Character is not eligible in this space"; once /cast add resolves among eligible
        characters only, a "No character named Carol" style reply is equally correct."""
        other = self.store.create_space(1, "Other", "world")
        self.store.add_character(1, other, "Carol", {"name": "Carol"}, None, [])
        interaction = FakeInteraction()
        await invoke(self.bot, "cast add", interaction, "Carol")
        self.assertEqual(len(interaction.replies), 1, interaction.replies)
        self.assertRegex(interaction.replies[0], r"(?i)carol|not eligible")
        self.assertNotIn("Added", interaction.replies[0])
        self.assertEqual(self.store.get_cast(100), [])

    async def test_memory_consent_lifecycle(self):
        user = 9
        await invoke(self.bot, "memory opt_in", FakeInteraction(user_id=user))
        self.assertTrue(self.store.has_consent(1, user))
        self.store.add_personal(1, user, self.alice, "Likes tea", 1)
        listing = FakeInteraction(user_id=user)
        await invoke(self.bot, "memory list", listing)
        self.assertIn("Alice: Likes tea", listing.replies[0])
        memory_id = self.store.personal(1, user)[0]["id"]
        await invoke(self.bot, "memory forget", FakeInteraction(user_id=77), memory_id)
        self.assertEqual(len(self.store.personal(1, user)), 1, "another user cannot forget it")
        await invoke(self.bot, "memory opt_out", FakeInteraction(user_id=user))
        self.assertEqual(self.store.personal(1, user), [])
        self.store.add_personal(1, user, self.alice, "Ignored without consent", 2)
        self.assertEqual(self.store.personal(1, user), [])

    async def test_lore_add_without_keys_is_constant(self):
        await invoke(self.bot, "admin lore add", FakeInteraction(admin=True), "The tide is high")
        row = self.store.list_lore(1, "channel", 100)[0]
        self.assertTrue(row["constant"])
        self.assertTrue(row["pinned"])

    async def test_lore_add_with_blank_keys_is_constant(self):
        """BUG-01 (fixed): keys that are only commas/whitespace parse to no keys, so the entry is constant and pinned."""
        await invoke(self.bot, "admin lore add", FakeInteraction(admin=True), "The tide is high", " , ")
        row = self.store.list_lore(1, "channel", 100)[0]
        self.assertTrue(row["constant"])
        self.assertTrue(row["pinned"])

    async def test_lore_add_with_keys_is_keyword_triggered(self):
        """BUG-01 (fixed): /admin lore add with keys creates an unpinned, non-constant entry gated by its keys."""
        await invoke(self.bot, "admin lore add", FakeInteraction(admin=True), "The lighthouse is haunted", "lighthouse")
        active = evaluate(self.store, 1, [("channel", 100)], "nothing relevant here", 1000)
        self.assertNotIn("The lighthouse is haunted", [item.content for item in active])
        active = evaluate(self.store, 1, [("channel", 100)], "we walk to the lighthouse", 1000)
        self.assertIn("The lighthouse is haunted", [item.content for item in active])

    async def test_scene_reset_records_timestamp(self):
        interaction = FakeInteraction()
        await invoke(self.bot, "scene reset", interaction)
        self.assertGreater(self.store.scene_reset_at(100), 0)
        self.assertIn("Scene reset", interaction.replies[0])

    async def test_scene_delete_requires_numeric_id(self):
        interaction = FakeInteraction(admin=True)
        await invoke(self.bot, "admin scene delete", interaction, "abc")
        self.assertEqual(interaction.replies, ["Give a numeric message ID"])

    async def test_context_without_trace(self):
        interaction = FakeInteraction()
        await invoke(self.bot, "context", interaction)
        self.assertEqual(interaction.replies, ["No saved context for that character line"])


if __name__ == "__main__":
    unittest.main()
