"""Characterization tests for slash-command callbacks (see docs/engineering/audit.md).

Tests named ``test_known_defect_*`` are ``expectedFailure``: they assert the intended behavior
for an objectively incorrect current behavior and will report "unexpected success" once fixed.
Command error mapping and definite /memory forget and /admin lore delete replies (UX-07, SEC-05 per D3) are pinned in
tests/test_error_mapping.py.
"""
import json
import unittest

import discord

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
        self.assertEqual(interaction.replies, ["This channel is not bound to a world or hub. Ask an administrator to bind it with /admin space bind."])

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
        self.assertEqual(show.replies, ["Active cast: Bob"])

    async def test_cast_show_empty_says_how_to_fill_it(self):
        """UI-26: an empty cast gets a full-sentence hint instead of a bare "Active cast: "."""
        interaction = FakeInteraction()
        await invoke(self.bot, "cast show", interaction)
        self.assertEqual(interaction.replies, ["The active cast is empty. Use /cast set to choose one."])

    async def test_cast_remove_drop_count_is_pluralised(self):
        """UI-26: "Also dropped N character(s) no longer available here." says "1 character" vs "2 characters"."""
        for extra, expected in ((1, "1 character no longer"), (2, "2 characters no longer")):
            other = self.store.create_space(1, f"Gone{extra}", "world")
            strays = [self.store.add_character(1, other, f"Stray{extra}{i}", {"name": "S"}, None, []) for i in range(extra)]
            self.store.execute("UPDATE channels SET active_cast=? WHERE channel_id=100", (json.dumps([self.alice, *strays]),))
            interaction = FakeInteraction()
            await invoke(self.bot, "cast remove", interaction, "Alice")
            self.assertIn(f"Also dropped {expected} available here.", interaction.replies[0], interaction.replies)

    async def test_ambient_on_off_status_replies_name_the_channel(self):
        """UI-26: ambient replies are full sentences naming the channel mention."""
        for command, state in (("admin ambient on", "on"), ("admin ambient off", "off")):
            interaction = FakeInteraction(admin=True)
            await invoke(self.bot, command, interaction)
            self.assertEqual(interaction.replies, [f"Ambient is {state} for <#100>."])
        status = FakeInteraction()
        await invoke(self.bot, "ambient status", status)
        self.assertEqual(status.replies, ["Ambient is off for <#100>."])

    async def test_lore_promote_reply_names_destination(self):
        """UI-26: promote says where the copy went (channel mention, or world/hub kind and name) and its new id."""
        source = self.store.add_lore(1, "channel", 100, "Salt air", [])
        interaction = FakeInteraction(admin=True)
        await invoke(self.bot, "admin lore promote", interaction, source, "space")
        new_id = self.store.list_lore(1, "space", self.world)[0]["id"]
        self.assertEqual(interaction.replies, [f"Promoted lore #{source} to world Harbor as lore #{new_id}."])
        interaction = FakeInteraction(admin=True)
        await invoke(self.bot, "admin lore promote", interaction, source, "channel")
        self.assertRegex(interaction.replies[0], rf"^Promoted lore #{source} to <#100> as lore #\d+\.$")

    async def test_lore_edit_and_promote_check_entry_exists_first(self):
        """UI-26: editing or promoting a missing lore entry replies with the not-found hint and changes nothing."""
        for command, args in (("admin lore edit", (4242, "New text")), ("admin lore promote", (4242, "channel"))):
            interaction = FakeInteraction(admin=True)
            await invoke(self.bot, command, interaction, *args)
            self.assertEqual(interaction.replies, ["Lore entry not found. Check the number with /lore list."])
        self.assertEqual(self.store.list_lore(1, "channel", 100), [])

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
        self.store.record_node(1, 1, 100, None, 9, None, 'src')
        self.store.record_node(2, 1, 100, None, 9, None, 'src')
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
        self.assertEqual(interaction.replies, ["Give a numeric message ID."])

    async def test_context_without_trace(self):
        interaction = FakeInteraction()
        await invoke(self.bot, "context", interaction)
        self.assertEqual(interaction.replies, ["No saved context for that character line. Leave the ID empty to use the latest one."])

    def _context_interaction(self, channels=None, admin=False):
        interaction = FakeInteraction(admin=admin)
        known = channels or {}
        interaction.guild.get_channel_or_thread = lambda ident: known.get(ident)
        return interaction

    def _item(self, kind, scope_id, lore_id, entry_key, reason="keyword: dragon", **extra):
        return {"id": lore_id, "scope": kind, "scope_id": scope_id, "entry_key": entry_key, "reason": reason, **extra}

    def _line(self, interaction, item):
        from llmcord_core.discord_bot import _context_lore_line
        return _context_lore_line(self.bot, interaction, item)

    async def _context_trace(self):
        self.store.record_node(5000, 1, 100, None, None, self.alice, "Hello")
        world_lore = self.store.add_lore(1, "space", self.world, "Dragons sleep under the hill", ["wyrm", "dragon"])
        chan_lore = self.store.add_lore(1, "channel", 100, "  The tide   is high today ", [])
        ghost_lore = self.store.add_lore(1, "channel", 4242, "Hidden room", ["secret"])
        self.store.save_trace(5000, {"lore": [
            self._item("space", self.world, world_lore, f"lore:{world_lore}"),
            self._item("channel", 100, chan_lore, f"lore:{chan_lore}", "always on"),
            self._item("guild", 1, 77, "lore:77", "always on"),
            self._item("channel", 4242, ghost_lore, f"lore:{ghost_lore}", "always on"),
        ]})

    async def test_context_names_lore_without_raw_ids(self):
        """Regression (UI-07): /context shows a member the matched keyword (or "an unnamed entry") and the owner in
        words: no other keys, no content excerpt, no raw ids, and the reply suppresses mentions."""
        from types import SimpleNamespace
        await self._context_trace()
        interaction = self._context_interaction({100: SimpleNamespace(id=100, name="harbor-chat")})
        await invoke(self.bot, "context", interaction)
        reply = interaction.replies[0]
        self.assertIn('"dragon" (world Harbor, keyword: dragon)', reply)
        self.assertIn("an unnamed entry (#harbor-chat, always on)", reply)
        self.assertIn("an unnamed entry (server-wide lore, always on)", reply)
        self.assertIn("an unnamed entry (#unknown-channel, always on)", reply)
        for hidden in ("wyrm", "tide", "secret", "Hidden", "lore:"):
            self.assertNotIn(hidden, reply, "members never see other keys, excerpts or entry keys")
        self.assertNotRegex(reply.split("\n")[0], r"#\d")
        self.assertNotIn("guild #", reply)
        self.assertNotIn("space #", reply)
        self.assertEqual(interaction.response.sent[0][1]["allowed_mentions"].to_dict(), discord.AllowedMentions.none().to_dict())

    async def test_context_shows_administrators_keys_and_excerpts(self):
        """Regression (UI-07): an administrator's /context labels an entry by its first key, else a 40-character excerpt."""
        from types import SimpleNamespace
        await self._context_trace()
        interaction = self._context_interaction({100: SimpleNamespace(id=100, name="harbor-chat")}, admin=True)
        await invoke(self.bot, "context", interaction)
        reply = interaction.replies[0]
        self.assertIn('"wyrm" (world Harbor, keyword: dragon)', reply)
        self.assertIn('"The tide is high today" (#harbor-chat, always on)', reply)
        self.assertIn("an unnamed entry (server-wide lore, always on)", reply)
        self.assertIn('"secret" (#unknown-channel, always on)', reply)
        self.assertNotRegex(reply.split("\n")[0], r"#\d")

    async def test_context_lore_line_member_label_is_only_the_matched_keyword(self):
        """Regression (UI-07): for a member a constant entry with keys is "an unnamed entry", and a keyword entry
        shows the keyword that matched, not keys[0]."""
        lore = self.store.add_lore(1, "space", self.world, "Body", ["first", "second"])
        member = self._context_interaction()
        self.assertEqual(self._line(member, self._item("space", self.world, lore, "named-key", "always on")),
                         "an unnamed entry (world Harbor, always on)")
        self.assertEqual(self._line(member, self._item("space", self.world, lore, "named-key", "keyword: second")),
                         '"second" (world Harbor, keyword: second)')
        self.assertEqual(self._line(member, self._item("space", self.world, lore, "named-key", "probability")),
                         "an unnamed entry (world Harbor, probability)")

    async def test_context_lore_line_owner_wording(self):
        """Regression (UI-07): _context_lore_line owner wording and admin label fallbacks for hub, character,
        lorebook, thread and unknown owners."""
        from types import SimpleNamespace
        hub = self.store.create_space(1, "Plaza", "hub")
        char_lore = self.store.add_lore(1, "character", self.alice, "Alice fears fire", [])
        interaction = self._context_interaction({300: SimpleNamespace(id=300, name="side-quest")}, admin=True)
        line = lambda item: self._line(interaction, item)  # noqa: E731

        self.assertEqual(line(self._item("space", hub, 999, "plaza-rule")), '"plaza-rule" (hub Plaza, keyword: dragon)')
        self.assertEqual(line(self._item("space", 9999, 999, "book:1:2")), "an unnamed entry (a world or hub, keyword: dragon)")
        self.assertEqual(line(self._item("character", self.alice, char_lore, "x")),
                         '"Alice fears fire" (character Alice, keyword: dragon)')
        self.assertEqual(line(self._item("lorebook", 5, 6, "book:5:1", book_name="Bestiary")),
                         "an unnamed entry (lorebook Bestiary, keyword: dragon)")
        self.assertEqual(line(self._item("lorebook", 5, 6, "troll", book_name=None)), '"troll" (a lorebook, keyword: dragon)')
        self.assertEqual(line(self._item("thread", 300, 8, "")), "an unnamed entry (#side-quest, keyword: dragon)")
        self.assertEqual(line(self._item("thread", 301, 8, "")), "an unnamed entry (a thread, keyword: dragon)")
        self.assertEqual(line(self._item("mystery", 1, 8, "")), "an unnamed entry (another owner, keyword: dragon)")
        long_text = self.store.add_lore(1, "space", self.world, "x" * 60, [])
        self.assertEqual(line(self._item("space", self.world, long_text, "lore:1")),
                         f'"{"x" * 40}" (world Harbor, keyword: dragon)')
        interaction.permissions = discord.Permissions(administrator=False)
        self.assertEqual(line(self._item("character", self.alice, char_lore, "x", "always on")),
                         "an unnamed entry (character Alice, always on)")

    async def test_context_lore_line_escapes_markdown_and_mentions(self):
        """Regression (UI-07): keys, names and reasons are escaped so lore text cannot ping or format the /context reply."""
        lore = self.store.add_lore(1, "space", self.world, "body", ["@everyone *x*"])
        reason = "keyword: @everyone *x*"
        for admin in (True, False):
            text = self._line(self._context_interaction(admin=admin), self._item("space", self.world, lore, "k", reason))
            self.assertNotIn("@everyone", text.replace("@\u200beveryone", ""), admin)
            self.assertIn("\\*x\\*", text, admin)
            self.assertEqual(text.count("\\*x\\*"), 2, "label and reason are both escaped")

    async def test_context_lore_line_other_guild_space_is_not_named(self):
        """Regression (UI-07): a space or character belonging to another server is not named in the reply."""
        foreign = self.store.create_space(2, "Secretland", "world")
        foreign_char = self.store.add_character(2, foreign, "Spy", {"name": "Spy"}, None, [])
        interaction = self._context_interaction(admin=True)
        self.assertEqual(self._line(interaction, self._item("space", foreign, 999, "k")), '"k" (a world or hub, keyword: dragon)')
        self.assertEqual(self._line(interaction, self._item("character", foreign_char, 999, "k")),
                         '"k" (a character, keyword: dragon)')

    async def test_context_lore_line_tolerates_malformed_keys_json(self):
        """Regression (UI-07): keys_json that is not a non-empty list of strings falls back to the excerpt for an
        administrator instead of raising."""
        interaction = self._context_interaction(admin=True)
        for raw in ("{}", "[1]", "not json", "[]"):
            lore = self.store.add_lore(1, "space", self.world, "Plain body", ["a"])
            self.store.execute("UPDATE lore SET keys_json=? WHERE id=?", (raw, lore))
            text = self._line(interaction, self._item("space", self.world, lore, f"lore:{lore}"))
            self.assertEqual(text, '"Plain body" (world Harbor, keyword: dragon)', raw)



if __name__ == "__main__":
    unittest.main()
