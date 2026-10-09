"""Where slash commands may run (SEC-05, R4 step 2a): guild-only synced payload plus a DM runtime safety net.
The ``/admin`` group surface (R4 step 2b, D11) is pinned in tests/test_admin_commands.py.
"""
import unittest

from helpers import FakeInteraction, FakeThread, command, invoke, leaf_commands, make_settings, reference_ids

from llmcord_core.discord_bot import SkitBot

# Discord InteractionContextType values (https://discord.com/developers/docs/interactions/application-commands).
GUILD, BOT_DM, PRIVATE_CHANNEL = 0, 1, 2
EXPECTED_TOP_LEVEL = {"space", "character", "cast", "ambient", "summon", "memory", "lore", "scene", "context"}


def dm_interaction(**kwargs) -> FakeInteraction:
    """A DM invocation: no guild, but Discord still supplies the DM channel."""
    return FakeInteraction(guild_id=None, channel_id=555, **kwargs)


class SyncedPayloadTests(unittest.TestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())

    def tearDown(self):
        self.bot.store.close()

    def test_registered_top_level_commands_include_the_known_surface(self):
        """Guard for the payload test below: it iterates the live tree, so it must see the known commands."""
        names = {cmd.name for cmd in self.bot.tree.get_commands(type=None)}
        self.assertTrue(EXPECTED_TOP_LEVEL <= names, EXPECTED_TOP_LEVEL - names)

    def test_every_synced_command_is_guild_only(self):
        """SEC-05: every top-level command/group's synced payload is limited to guilds (contexts == [GUILD], or no
        contexts and dm_permission False), so Discord never offers it in DMs or group DMs. Iterates the live tree,
        so a command added later without the flag fails here."""
        offenders = {}
        for cmd in self.bot.tree.get_commands(type=None):
            payload = cmd.to_dict(self.bot.tree)
            contexts = payload.get("contexts")
            guild_only = contexts == [GUILD] if contexts is not None else payload.get("dm_permission") is False
            if not guild_only:
                offenders[cmd.name] = (contexts, payload.get("dm_permission"))
        self.assertEqual(offenders, {})


class DirectMessageInvocationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.world = self.store.create_space(1, "Harbor", "world")
        self.store.bind_channel(1, 100, self.world)
        self.alice = self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])
        self.store.set_consent(1, 9, True)
        self.store.record_node(1, 1, 100, None, 9, None, 'src')
        self.store.add_personal(1, 9, self.alice, "Likes tea", 1)
        self.memory_id = self.store.personal(1, 9)[0]["id"]
        self.guild_ids_seen = []
        for name in ("set_consent", "personal", "forget_personal"):
            self._record_guild_ids(name)

    def tearDown(self):
        self.store.close()

    def _record_guild_ids(self, name):
        """Observe (not replace) a store method: record the guild_id each call receives, then call through."""
        original = getattr(self.store, name)

        def recording(guild_id, *args, **kwargs):
            self.guild_ids_seen.append((name, guild_id))
            return original(guild_id, *args, **kwargs)
        setattr(self.store, name, recording)

    def null_guild_rows(self) -> dict[str, int]:
        found = {}
        for (table,) in self.store.db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            columns = [row[1] for row in self.store.db.execute(f"PRAGMA table_info({table})").fetchall()]
            if "guild_id" in columns:
                count = self.store.db.execute(f"SELECT COUNT(*) FROM {table} WHERE guild_id IS NULL").fetchone()[0]
                if count:
                    found[table] = count
        return found

    def assert_server_only_reply(self, interaction):
        self.assertEqual(len(interaction.replies), 1, interaction.replies)
        reply = interaction.replies[0]
        self.assertIn("server", reply.lower())
        self.assertEqual(reference_ids(reply), [])
        for leak in ("sqlite", "constraint", "integrity", "NULL"):
            self.assertNotIn(leak.lower(), reply.lower())
        sent = interaction.response.sent + interaction.followup.sent
        self.assertTrue(sent[0][1].get("ephemeral"), sent)
        self.assertNotIn(None, [guild_id for _, guild_id in self.guild_ids_seen], self.guild_ids_seen)
        self.assertEqual(self.null_guild_rows(), {})

    async def test_dm_memory_opt_in_says_use_a_server(self):
        """SEC-05 (fixed): /memory opt_in from a DM never reaches the store with guild_id None and tells the person
        to use it in a server (before SEC-05: set_consent(None, ...) hit NOT NULL and the generic ref message was sent)."""
        interaction = dm_interaction()
        await invoke(self.bot, "memory opt_in", interaction)
        self.assert_server_only_reply(interaction)

    async def test_dm_memory_opt_out_says_use_a_server(self):
        """SEC-05: /memory opt_out from a DM never reaches the store with guild_id None, says to use a server, and
        leaves the person's guild memories alone."""
        interaction = dm_interaction()
        await invoke(self.bot, "memory opt_out", interaction)
        self.assert_server_only_reply(interaction)
        self.assertTrue(self.store.has_consent(1, 9))
        self.assertEqual(len(self.store.personal(1, 9)), 1)

    async def test_dm_memory_list_says_use_a_server(self):
        """SEC-05 (fixed): /memory list from a DM says to use a server instead of querying guild None (before SEC-05
        it replied "No personal memories saved.", which is misleading for someone with memories in a server)."""
        interaction = dm_interaction()
        await invoke(self.bot, "memory list", interaction)
        self.assert_server_only_reply(interaction)

    async def test_dm_memory_forget_says_use_a_server(self):
        """SEC-05 (fixed): /memory forget from a DM says to use a server and deletes nothing (before SEC-05 it queried
        guild None and replied "No memory #N of yours was found.")."""
        interaction = dm_interaction()
        await invoke(self.bot, "memory forget", interaction, self.memory_id)
        self.assert_server_only_reply(interaction)
        self.assertEqual([row["id"] for row in self.store.personal(1, 9)], [self.memory_id])

    async def test_dm_cast_show_keeps_its_server_channels_message(self):
        """Characterization (SEC-05): commands using ``binding_for`` already refuse DMs with a clear message."""
        interaction = dm_interaction()
        await invoke(self.bot, "cast show", interaction)
        self.assertEqual(interaction.replies, ["Use this command in a server channel."])
        self.assertTrue(interaction.response.sent[0][1].get("ephemeral"))
        self.assertEqual(self.null_guild_rows(), {})


class GuildChannelAndThreadTests(unittest.IsolatedAsyncioTestCase):
    """Regression guards (SEC-05): limiting commands to guilds must keep guild text channels and threads working."""

    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.world = self.store.create_space(1, "Harbor", "world")
        self.store.bind_channel(1, 100, self.world)
        self.alice = self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])

    def tearDown(self):
        self.store.close()

    async def test_memory_opt_in_in_guild_channel_records_consent(self):
        interaction = FakeInteraction()
        await invoke(self.bot, "memory opt_in", interaction)
        self.assertTrue(self.store.has_consent(1, 9))
        self.assertTrue(interaction.replies[0].startswith("Personal memory enabled."))

    async def test_memory_opt_in_in_thread_records_consent(self):
        interaction = FakeInteraction(channel=FakeThread(101, parent_id=100))
        await invoke(self.bot, "memory opt_in", interaction)
        self.assertTrue(self.store.has_consent(1, 9))

    async def test_cast_set_and_show_in_thread_use_the_parent_binding(self):
        thread = FakeThread(101, parent_id=100)
        await invoke(self.bot, "cast set", FakeInteraction(channel=thread), "Alice")
        shown = FakeInteraction(channel=thread)
        await invoke(self.bot, "cast show", shown)
        self.assertEqual(shown.replies, ["Active cast: Alice"])
        self.assertEqual(self.store.get_cast(101, 100), [self.alice])

    async def test_admin_lore_add_in_thread_scopes_to_the_thread(self):
        interaction = FakeInteraction(admin=True, channel=FakeThread(101, parent_id=100))
        await invoke(self.bot, "admin lore add", interaction, "The thread is foggy")
        self.assertEqual([row["content"] for row in self.store.list_lore(1, "thread", 101)], ["The thread is foggy"])

    async def test_admin_lore_add_scope_channel_in_a_channel_is_channel_lore(self):
        """UX-02: ``scope:channel`` (the default) in a plain channel adds channel lore and says so."""
        interaction = FakeInteraction(admin=True)
        await invoke(self.bot, "admin lore add", interaction, "The channel is quiet", scope="channel")
        self.assertEqual([row["content"] for row in self.store.list_lore(1, "channel", 100)], ["The channel is quiet"])
        self.assertEqual(self.store.list_lore(1, "thread", 100), [])
        self.assertRegex(interaction.replies[0], r"^Added channel lore #\d+\.$")

    async def test_admin_lore_add_scope_channel_in_a_thread_is_thread_lore(self):
        """UX-02: ``scope:channel`` run inside a thread owns the lore by the thread, not its parent channel."""
        interaction = FakeInteraction(admin=True, channel=FakeThread(101, parent_id=100))
        await invoke(self.bot, "admin lore add", interaction, "Fog in the thread", scope="channel")
        self.assertEqual([row["content"] for row in self.store.list_lore(1, "thread", 101)], ["Fog in the thread"])
        self.assertEqual(self.store.list_lore(1, "channel", 100), [])
        self.assertRegex(interaction.replies[0], r"^Added channel lore #\d+\.$")

    async def test_admin_lore_add_scope_space_is_owned_by_the_bound_world(self):
        """UX-02: ``scope:space`` adds lore to the channel's bound world and the reply names it a world."""
        interaction = FakeInteraction(admin=True)
        await invoke(self.bot, "admin lore add", interaction, "Harbor lore", scope="space")
        self.assertEqual([row["content"] for row in self.store.list_lore(1, "space", self.world)], ["Harbor lore"])
        self.assertEqual(self.store.list_lore(1, "channel", 100), [])
        self.assertRegex(interaction.replies[0], r"^Added world lore #\d+\.$")

    async def test_admin_lore_add_scope_choices_are_channel_and_space(self):
        """UX-02: the synced ``scope`` option offers exactly ``channel`` and ``space`` (no ``local``)."""
        scope = command(self.bot, "admin lore add").to_dict(self.bot.tree)["options"]
        scope = next(option for option in scope if option["name"] == "scope")
        self.assertEqual([choice["value"] for choice in scope["choices"]], ["channel", "space"])

    def test_link_world_commands_replace_allow_world_commands(self):
        """UX-02: the command tree has ``/admin space link_world`` and ``unlink_world`` and no allow/disallow names."""
        paths = set(leaf_commands(self.bot))
        self.assertTrue({"admin space link_world", "admin space unlink_world"} <= paths)
        self.assertFalse({"admin space allow_world", "admin space disallow_world"} & paths)

    async def test_admin_permission_check_still_runs_in_guild(self):
        """The runtime ``has_permissions(administrator=True)`` check stays alongside the guild-only flag and the
        ``/admin`` group's ``default_member_permissions`` (D11)."""
        cmd = command(self.bot, "admin lore add")
        self.assertTrue(cmd.checks)
        interaction = FakeInteraction(admin=False, channel=FakeThread(101, parent_id=100))
        await invoke(self.bot, "admin lore add", interaction, "Nope")
        self.assertEqual(interaction.replies, ["Only server administrators can use that command."])
        self.assertEqual(self.store.list_lore(1, "thread", 101), [])

    async def test_admin_command_from_a_dm_says_use_a_server_channel(self):
        """UI-06: a stale DM client invoking an admin command (has_permissions fails with no guild) is told to use a
        server channel, ephemerally, not that only administrators may; a guild non-admin still gets the admin reply."""
        interaction = dm_interaction()
        await invoke(self.bot, "admin space create", interaction, "world", "Elsewhere")
        self.assertEqual(interaction.replies, ["Use this command in a server channel."])
        self.assertIs(interaction.response.sent[0][1].get("ephemeral"), True)
        guild_user = FakeInteraction(admin=False)
        await invoke(self.bot, "admin space create", guild_user, "world", "Elsewhere")
        self.assertEqual(guild_user.replies, ["Only server administrators can use that command."])
        uncached = FakeInteraction(admin=False)
        uncached.guild = None  # guild not cached, but guild_id is set: still a real server non-admin
        await invoke(self.bot, "admin space create", uncached, "world", "Elsewhere")
        self.assertEqual(uncached.replies, ["Only server administrators can use that command."])


if __name__ == "__main__":
    unittest.main()
