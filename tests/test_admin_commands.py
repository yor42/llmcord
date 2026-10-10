"""Admin commands live under one hidden ``/admin`` group (SEC-05, UX-06; decision D11, R4 step 2b).

Also pins the admin commands' option names (UX-06, R4 step 3); member option names and the shared name resolver are
pinned in tests/test_name_resolution.py.
"""
import inspect
import unittest
from types import SimpleNamespace

import discord
from discord import app_commands
from helpers import FakeInteraction, command, invoke, leaf_commands, make_settings

from llmcord_core.discord_bot import SkitBot

# Discord InteractionContextType.guild (see test_command_contexts.py).
GUILD = 0
ADMIN_BIT = str(discord.Permissions(administrator=True).value)  # "8"

# D11: admin subgroup -> subcommands.
ADMIN_SURFACE = {
    "space": {"create", "bind", "link_world", "unlink_world", "tone"},
    "character": {"import"},
    "cast": {"default"},
    "ambient": {"on", "off"},
    "lore": {"add", "pin", "edit", "promote", "delete"},
    "scene": {"delete"},
    "currency": {"grant", "revoke", "name", "daily"},
    "games": {"channel", "bets"},
}
# D11: member group -> subcommands; ``None`` marks a top-level command without subcommands.
MEMBER_SURFACE = {
    "space": {"list"},
    "character": {"list", "info"},
    "cast": {"set", "add", "remove", "show"},
    "ambient": {"status"},
    "summon": None,
    "catchup": None,
    "memory": {"opt_in", "opt_out", "list", "forget"},
    "time": {"set", "show", "clear"},
    "favorites": {"add", "remove", "list", "mode", "clear"},
    "balance": None,
    "daily": None,
    "blackjack": None,
    "lore": {"list"},
    "scene": {"reset"},
}
# UI-49: top-level admin-only command outside the ``/admin`` group.
ADMIN_TOP_LEVEL = {"context"}
ADMIN_PATHS = {f"admin {group} {name}" for group, names in ADMIN_SURFACE.items() for name in names}
MEMBER_PATHS = {group if names is None else f"{group} {name}"
                for group, names in MEMBER_SURFACE.items() for name in (names or [None])}
# Synced option names of each admin command (UX-06, R4 step 3: ``space``/``hub``/``world``/``characters``).
ADMIN_OPTIONS = {
    "admin space create": ["kind", "name"],
    "admin space bind": ["channel", "space"],
    "admin space link_world": ["hub", "world"],
    "admin space unlink_world": ["hub", "world"],
    "admin space tone": ["hub", "tone"],
    "admin character import": ["world", "attachment"],
    "admin cast default": ["characters"],
    "admin ambient on": [],
    "admin ambient off": [],
    "admin lore add": ["content", "keys", "scope"],
    "admin lore pin": ["lore_id"],
    "admin lore edit": ["lore_id", "content"],
    "admin lore promote": ["lore_id", "destination", "space"],
    "admin lore delete": ["lore_id"],
    "admin scene delete": ["message_id"],
    "admin currency grant": ["member", "amount", "reason"],
    "admin currency revoke": ["member", "amount", "reason"],
    "admin currency name": ["name"],
    "admin currency daily": ["amount", "streak_bonus", "streak_days"],
    "admin games channel": ["enabled", "channel"],
    "admin games bets": ["min", "max"],
}
# Arguments for invoking each member command (DM test); everything else uses defaults.
MEMBER_ARGS = {
    "favorites add": ("Alice",),
    "favorites remove": ("Alice",),
    "favorites mode": ("lean",),
    "character info": ("Alice",),
    "cast set": ("Alice",),
    "cast add": ("Alice",),
    "cast remove": ("Alice",),
    "summon": ("Alice", "hello"),
    "memory forget": (1,),
    "time set": ("Asia/Seoul",),
    "blackjack": (10,),
}


def dump(store) -> list[str]:
    return list(store.db.iterdump())


class AdminSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())

    def tearDown(self):
        self.bot.store.close()

    def test_surface_table_matches_d11(self):
        """Guard for the tests below: the expected tables have the 21 admin and 27 member (plus the admin-only /context) commands D11 lists."""
        self.assertEqual(len(ADMIN_PATHS), 21)
        self.assertEqual(len(MEMBER_PATHS), 27)
        self.assertEqual(set(ADMIN_OPTIONS), ADMIN_PATHS)

    def test_admin_group_holds_exactly_the_admin_subcommands(self):
        """SEC-05/UX-06 (D11): a top-level ``admin`` group exists and its subgroups contain exactly the admin
        subcommands (before step 2b there is no ``admin`` group)."""
        admin = self.bot.tree.get_command("admin")
        self.assertIsInstance(admin, app_commands.Group)
        found = {sub.name: {cmd.name for cmd in sub.commands} for sub in admin.commands}
        self.assertEqual(found, ADMIN_SURFACE)

    def test_member_groups_still_offer_every_member_subcommand(self):
        """Regression (D11): member commands keep their paths."""
        missing = sorted(path for path in MEMBER_PATHS if path not in leaf_commands(self.bot))
        self.assertEqual(missing, [])

    def test_member_groups_hold_only_member_subcommands(self):
        """SEC-05/UX-06 (D11): outside ``/admin`` the tree has exactly the member commands, so no member group still
        contains an admin subcommand and the old admin paths (``/space create``, ``/lore add``...) are gone."""
        top = {cmd.name: cmd for cmd in self.bot.tree.get_commands(type=None) if cmd.name != "admin"}
        found = {name: ({sub.name for sub in cmd.commands} if isinstance(cmd, app_commands.Group) else None)
                 for name, cmd in top.items()}
        self.assertEqual(found, MEMBER_SURFACE | {name: None for name in ADMIN_TOP_LEVEL})

    def test_whole_tree_is_exactly_the_d11_surface(self):
        """SEC-05/UX-06 (D11): every invocable path is either a listed admin or a listed member command."""
        self.assertEqual(set(leaf_commands(self.bot)), ADMIN_PATHS | MEMBER_PATHS | ADMIN_TOP_LEVEL)

    def test_admin_commands_use_consistent_option_names(self):
        """UX-06: each admin command's synced option names and order are ``ADMIN_OPTIONS`` (previously ``space_name``,
        ``hub_name``, ``world_name`` and ``names``)."""
        found = {path: [option["name"] for option in command(self.bot, path).to_dict(self.bot.tree).get("options", [])]
                 for path in ADMIN_OPTIONS}
        self.assertEqual(found, ADMIN_OPTIONS)

    def test_admin_payload_is_hidden_and_guild_only(self):
        """SEC-05 (D11): the synced ``/admin`` payload sets ``default_member_permissions`` to the administrator bit
        (Discord hides it from non-admins), is guild-only, and carries the admin subgroups."""
        admin = self.bot.tree.get_command("admin")
        self.assertIsNotNone(admin, "no /admin group registered")
        payload = admin.to_dict(self.bot.tree)
        self.assertEqual(str(payload.get("default_member_permissions")), ADMIN_BIT)
        contexts = payload.get("contexts")
        self.assertTrue(contexts == [GUILD] if contexts is not None else payload.get("dm_permission") is False,
                        (contexts, payload.get("dm_permission")))
        options = {option["name"]: {sub["name"] for sub in option.get("options", [])} for option in payload["options"]}
        self.assertEqual(options, ADMIN_SURFACE)

    def test_member_payloads_have_no_default_member_permissions(self):
        """Regression (D11): member groups and commands stay visible to everyone (no ``default_member_permissions``)."""
        flagged = {}
        for name in MEMBER_SURFACE:
            payload = self.bot.tree.get_command(name).to_dict(self.bot.tree)
            if payload.get("default_member_permissions") is not None:
                flagged[name] = payload["default_member_permissions"]
        self.assertEqual(flagged, {})

    def test_every_admin_subcommand_keeps_the_runtime_admin_check(self):
        """SEC-05 (D11): ``default_member_permissions`` is only a client-side default (server owners can override it), so
        each ``/admin`` subcommand still carries ``has_permissions(administrator=True)``: its checks reject a
        non-admin with ``MissingPermissions`` and pass an admin."""
        unchecked = []
        for path in sorted(ADMIN_PATHS):
            cmd = command(self.bot, path)
            for check in cmd.checks:
                check(FakeInteraction(admin=True))
            try:
                for check in cmd.checks:
                    check(FakeInteraction(admin=False))
            except app_commands.MissingPermissions:
                continue
            unchecked.append(path)
        self.assertEqual(unchecked, [])

    def test_context_is_declared_administrator_only(self):
        """UI-49: /context is hidden from members (administrator default permission) and keeps the runtime admin check
        for stale clients: a non-admin gets ``MissingPermissions``, an admin passes."""
        cmd = self.bot.tree.get_command("context")
        payload = cmd.to_dict(self.bot.tree)
        self.assertEqual(str(payload.get("default_member_permissions")), ADMIN_BIT)
        self.assertTrue(cmd.checks)
        for check in cmd.checks:
            check(FakeInteraction(admin=True))
        with self.assertRaises(app_commands.MissingPermissions):
            for check in cmd.checks:
                check(FakeInteraction(admin=False))

    def test_member_commands_have_no_admin_check(self):
        """Regression (D11): member commands run for non-admins (their checks, if any, pass)."""
        for path in sorted(MEMBER_PATHS):
            for check in command(self.bot, path).checks:
                with self.subTest(path=path):
                    check(FakeInteraction(admin=False))


class AdminInvocationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.world = self.store.create_space(1, "Harbor", "world")
        self.hub = self.store.create_space(1, "Plaza", "hub")
        self.store.bind_channel(1, 100, self.world)
        self.store.bind_channel(1, 101, self.world)
        self.alice = self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])

    def tearDown(self):
        self.store.close()

    async def test_non_admin_gets_the_admin_message_from_every_admin_subcommand(self):
        """SEC-05 (D11): a non-admin calling any ``/admin`` subcommand gets the plain ephemeral admin-only message and
        nothing in the database changes. (Checks run before the callback, so no arguments are needed.)"""
        for path in sorted(ADMIN_PATHS):
            with self.subTest(path=path):
                before = dump(self.store)
                interaction = FakeInteraction(admin=False)
                await invoke(self.bot, path, interaction)
                self.assertEqual(interaction.replies, ["Only server administrators can use that command."])
                self.assertIs(interaction.response.sent[0][1].get("ephemeral"), True)
                self.assertEqual(dump(self.store), before)

    async def test_non_admin_gets_the_admin_message_from_context(self):
        """UI-49: a member (stale client) calling /context gets the admin-only message."""
        interaction = FakeInteraction(admin=False)
        await invoke(self.bot, "context", interaction)
        self.assertEqual(interaction.replies, ["Only server administrators can use that command."])
        self.assertIs(interaction.response.sent[0][1].get("ephemeral"), True)

    async def test_non_admin_admin_lore_add_writes_nothing(self):
        """SEC-05 (D11): ``/admin lore add`` from a non-admin is refused with the admin-only message, no lore row."""
        interaction = FakeInteraction(admin=False)
        await invoke(self.bot, "admin lore add", interaction, "Nope")
        self.assertEqual(interaction.replies, ["Only server administrators can use that command."])
        self.assertEqual(self.store.list_lore(1, "channel", 100), [])

    async def test_admin_space_bind_binds_the_channel(self):
        """UX-06 (D11): ``/admin space bind`` behaves like the old ``/space bind``."""
        channel = SimpleNamespace(id=300, mention="<#300>")
        interaction = FakeInteraction(admin=True)
        await invoke(self.bot, "admin space bind", interaction, channel, "Plaza")
        self.assertEqual(interaction.replies, ["Bound <#300> to hub Plaza."])
        self.assertEqual(self.store.channel(300)["space_id"], self.hub)

    async def test_admin_scene_delete_removes_the_node_and_descendants(self):
        """UX-09 (D5): ``/admin scene delete`` deletes the given node and its descendants in this channel (not the
        whole tree), and refuses a node from another channel."""
        self.store.record_node(500, 1, 100, None, 9, None, "start", created_at=1.0)
        self.store.record_node(501, 1, 100, 500, None, self.alice, "reply", created_at=2.0)
        elsewhere = FakeInteraction(admin=True, channel_id=101)
        await invoke(self.bot, "admin scene delete", elsewhere, "500")
        self.assertIn("not found", elsewhere.replies[0])
        self.assertIsNotNone(self.store.node(500))
        interaction = FakeInteraction(admin=True)
        await invoke(self.bot, "admin scene delete", interaction, "501")
        self.assertIn("1", interaction.replies[0])
        self.assertIsNone(self.store.node(501))
        self.assertIsNotNone(self.store.node(500))


class MemberDirectMessageTests(unittest.IsolatedAsyncioTestCase):
    """Regression (SEC-05): every member command refuses a DM with an ephemeral "use a server" reply and never reaches
    the store with ``guild_id`` None. (Admin commands are covered by the guild-only payload and admin check.)"""

    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.world = self.store.create_space(1, "Harbor", "world")
        self.store.bind_channel(1, 100, self.world)
        self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])
        self.guild_ids_seen = []
        for name, method in inspect.getmembers(self.store, inspect.ismethod):
            params = list(inspect.signature(method).parameters)
            if not name.startswith("_") and params[:1] == ["guild_id"]:
                setattr(self.store, name, self._recording(name, method))

    def tearDown(self):
        self.store.close()

    def _recording(self, name, method):
        """Observe (not replace) a store method: record the guild_id it receives, then call through."""
        def recording(guild_id, *args, **kwargs):
            self.guild_ids_seen.append((name, guild_id))
            return method(guild_id, *args, **kwargs)
        return recording

    async def test_every_member_command_refuses_dms(self):
        for path in sorted(MEMBER_PATHS):
            with self.subTest(path=path):
                self.guild_ids_seen.clear()
                interaction = FakeInteraction(guild_id=None, channel_id=555)
                await invoke(self.bot, path, interaction, *MEMBER_ARGS.get(path, ()))
                self.assertEqual(len(interaction.replies), 1, interaction.replies)
                self.assertIn("server", interaction.replies[0].lower())
                sent = interaction.response.sent + interaction.followup.sent
                self.assertIs(sent[0][1].get("ephemeral"), True)
                self.assertEqual([seen for seen in self.guild_ids_seen if seen[1] is None], [])

    async def test_context_refuses_dms(self):
        interaction = FakeInteraction(guild_id=None, channel_id=555, admin=True)
        await invoke(self.bot, "context", interaction)
        self.assertIn("server", interaction.replies[0].lower())

    async def test_context_refuses_dms_for_non_admins_with_the_server_message(self):
        """UI-49: a DM has no member permissions, so /context gives the use-in-a-server message, not admin-only."""
        from llmcord_core.discord_bot import USE_IN_SERVER_CHANNEL
        interaction = FakeInteraction(guild_id=None, channel_id=555, admin=False)
        await invoke(self.bot, "context", interaction)
        self.assertEqual(interaction.replies, [USE_IN_SERVER_CHANNEL])


if __name__ == "__main__":
    unittest.main()
