"""Store-level queries and character save replace raw SQL in the dashboard (ARCH-02; roadmap R5 step 1).

Target API: ``Store.update_character(..., expected_revision=None)`` raises ``ConflictError`` on a stale revision;
``Store.list_characters/list_channels/thread_lore_scopes/asset_channel_id`` replace ``store.all/one`` calls in
``dashboard.py`` and ``admin.py``. ``AdminService.owners`` output is pinned as a characterization, and a source
guard keeps raw SQL and private store helpers out of the two UI-facing modules.
"""
import json
import unittest
from pathlib import Path

import httpx

from helpers import discord_transport

from llmcord_core.admin_store import ConflictError
from llmcord_core.store import Store
from llmcord_core.web import create_app

CORE = Path(__file__).resolve().parent.parent / "llmcord_core"


def seed(store):
    """Guild 1: worlds Harbor (Alice, Bob) and Forest (Carol); hub Plaza links both. Guild 2: Faraway (Zara)."""
    s = store
    s.harbor = s.create_space(1, "Harbor", "world")
    s.forest = s.create_space(1, "Forest", "world")
    s.plaza = s.create_space(1, "Plaza", "hub")
    s.link_world(1, s.plaza, s.harbor)
    s.link_world(1, s.plaza, s.forest)
    add = lambda g, w, n: s.add_character(g, w, n, {"name": n, "description": "d"}, None, [])  # noqa: E731
    s.bob, s.alice = add(1, s.harbor, "Bob"), add(1, s.harbor, "Alice")
    s.carol = add(1, s.forest, "Carol")
    s.faraway = s.create_space(2, "Faraway", "world")
    s.zara = add(2, s.faraway, "Zara")


class UpdateCharacterTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        seed(self.store)

    def tearDown(self):
        self.store.close()

    def char(self, ident=None):
        return self.store.character_by_id(ident or self.store.alice)

    def rev(self):
        return self.store.owner_revision(1, "character", self.store.alice)

    def update(self, **kw):
        args = dict(guild_id=1, character_id=self.store.alice, world_id=self.store.harbor,
                    name="Alice", card={"name": "Alice", "description": "new"})
        args.update(kw)
        return self.store.update_character(**args)

    def test_matching_revision_succeeds_and_bumps(self):
        before = self.rev()
        self.update(name="Alicia", expected_revision=before)
        self.assertEqual(self.char()["name"], "Alicia")
        self.assertEqual(json.loads(self.char()["card"])["description"], "new")
        self.assertEqual(self.rev(), before + 1)

    def test_stale_revision_raises_and_changes_nothing(self):
        stale = self.rev()
        self.update(name="First")  # bumps
        with self.assertRaises(ConflictError):
            self.update(name="Second", world_id=self.store.forest, expected_revision=stale)
        row = self.char()
        self.assertEqual(row["name"], "First")
        self.assertEqual(row["world_id"], self.store.harbor)
        self.assertEqual(self.rev(), stale + 1)

    def test_no_expected_revision_still_works(self):
        self.update(name="Legacy")
        self.assertEqual(self.char()["name"], "Legacy")

    def test_other_guild_character_rejected_and_unchanged(self):
        with self.assertRaises(ValueError):
            self.store.update_character(1, self.store.zara, self.store.harbor, "Hacked", {})
        with self.assertRaises(ValueError):
            self.store.update_character(2, self.store.zara, self.store.harbor, "Hacked", {})
        self.assertEqual(self.char(self.store.zara)["name"], "Zara")
        self.assertEqual(self.char(self.store.zara)["world_id"], self.store.faraway)

    def test_foreign_or_non_world_target_rejected(self):
        for target in (self.store.faraway, self.store.plaza):
            with self.assertRaises(ValueError):
                self.update(world_id=target)
        self.assertEqual(self.char()["world_id"], self.store.harbor)

    def test_empty_name_rejected(self):
        for name in ("", "   "):
            with self.assertRaises(ValueError):
                self.update(name=name)
        self.assertEqual(self.char()["name"], "Alice")

    def test_name_is_stripped(self):
        self.update(name="  Padded  ")
        self.assertEqual(self.char()["name"], "Padded")

    def test_world_move_prunes_channel_and_thread_casts(self):
        s = self.store
        s.bind_channel(1, 100, s.harbor)   # Alice ineligible after moving to Forest
        s.bind_channel(1, 200, s.plaza)    # Alice stays eligible via the hub
        s.set_cast(100, None, [s.alice, s.bob], default=True)
        s.set_cast(200, None, [s.alice, s.carol], default=True)
        s.set_cast(555, 200, [s.alice, s.carol])  # thread under the hub channel
        self.update(world_id=s.forest)
        cast = lambda ch: (json.loads(s.channel(ch)["default_cast"]), json.loads(s.channel(ch)["active_cast"]))  # noqa: E731
        self.assertEqual(cast(100), ([s.bob], [s.bob]))
        self.assertEqual(cast(200), ([s.alice, s.carol], [s.alice, s.carol]))
        thread = s.one('SELECT "cast" FROM thread_casts WHERE thread_id=?', (555,))
        self.assertEqual(json.loads(thread["cast"]), [s.carol])

    def test_non_move_edit_keeps_casts(self):
        s = self.store
        s.bind_channel(1, 100, s.harbor)
        s.bind_channel(1, 200, s.plaza)
        s.set_cast(100, None, [s.alice, s.bob], default=True)
        s.set_cast(555, 200, [s.alice])
        self.update(name="Renamed")
        self.assertEqual(json.loads(s.channel(100)["default_cast"]), [s.alice, s.bob])
        thread = s.one('SELECT "cast" FROM thread_casts WHERE thread_id=?', (555,))
        self.assertEqual(json.loads(thread["cast"]), [s.alice])


class ReadHelperTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        seed(self.store)

    def tearDown(self):
        self.store.close()

    def test_list_characters_ordered_scoped_includes_archived(self):
        s = self.store
        self.assertEqual([r["name"] for r in s.list_characters(1)], ["Alice", "Bob", "Carol"])
        s.archive_character(1, s.bob, True)
        rows = s.list_characters(1)
        self.assertEqual([r["name"] for r in rows], ["Alice", "Bob", "Carol"])
        self.assertEqual([r["name"] for r in s.list_characters(2)], ["Zara"])
        self.assertEqual(s.list_characters(3), [])

    def test_list_channels_ordered_scoped(self):
        s = self.store
        self.assertEqual(s.list_channels(1), [])
        s.bind_channel(1, 300, s.harbor)
        s.bind_channel(1, 100, s.plaza)
        s.bind_channel(2, 200, s.faraway)
        self.assertEqual([r["channel_id"] for r in s.list_channels(1)], [100, 300])
        self.assertEqual([r["channel_id"] for r in s.list_channels(2)], [200])
        self.assertEqual(s.list_channels(3), [])

    def test_thread_lore_scopes_distinct_scoped(self):
        s = self.store
        self.assertEqual(list(s.thread_lore_scopes(1)), [])
        s.add_lore(1, "thread", 900, "a")
        s.add_lore(1, "thread", 900, "b")
        s.add_lore(1, "thread", 800, "c")
        s.add_lore(1, "channel", 100, "not a thread")
        s.add_lore(2, "thread", 700, "other guild")
        scopes = s.thread_lore_scopes(1)
        self.assertEqual(sorted(scopes), [800, 900])
        self.assertTrue(all(isinstance(x, int) for x in scopes))
        self.assertEqual(sorted(s.thread_lore_scopes(2)), [700])
        self.assertEqual(list(s.thread_lore_scopes(3)), [])

    def test_asset_channel_id(self):
        s = self.store
        self.assertIsNone(s.asset_channel_id(1))
        s.execute("INSERT INTO guild_settings(guild_id,asset_channel_id) VALUES(1,NULL)")
        self.assertIsNone(s.asset_channel_id(1))
        s.execute("UPDATE guild_settings SET asset_channel_id=555 WHERE guild_id=1")
        self.assertEqual(s.asset_channel_id(1), 555)
        self.assertIsNone(s.asset_channel_id(2))


class OwnersTests(unittest.TestCase):
    def test_owners_characterization(self):
        """Characterization (ARCH-02): AdminService.owners lists spaces, characters, channels, guild, books, threads."""
        transport, _ = discord_transport(admin_guilds=[1])
        app = create_app(":memory:", "https://pi.test", "client", "secret", "bot",
                         httpx.AsyncClient(transport=transport), enable_dashboard=False,
                         config_path="tests/nonexistent-config.yaml")
        store = app.state.store
        try:
            seed(store)
            store.bind_channel(1, 300, store.harbor)
            store.bind_channel(1, 100, store.plaza)
            book = store.create_lorebook(1, "Tales", "guild")
            store.add_lore(1, "thread", 900, "a")
            store.add_lore(1, "thread", 900, "b")
            store.bind_channel(2, 200, store.faraway)
            owners = app.state.admin.owners(1)
            by = lambda kind: [(o["id"], o["label"]) for o in owners if o["kind"] == kind]  # noqa: E731
            self.assertEqual(by("space"), [(store.plaza, "Hub: Plaza"), (store.forest, "World: Forest"),
                                           (store.harbor, "World: Harbor")])
            self.assertEqual(by("character"), [(store.alice, "Character: Alice"), (store.bob, "Character: Bob"),
                                               (store.carol, "Character: Carol")])
            self.assertEqual(by("channel"), [(100, "Channel: 100"), (300, "Channel: 300")])
            self.assertEqual(by("guild"), [(1, "Server: Server-wide lore")])
            self.assertEqual(by("book"), [(book, "Lorebook: Tales")])
            self.assertEqual(by("thread"), [(900, "Thread: 900")])
            kinds = [o["kind"] for o in owners]
            self.assertEqual(kinds, sorted(kinds, key=["space", "character", "channel", "guild", "book", "thread"].index))
        finally:
            store.close()


class OwnersFromRowsTests(unittest.TestCase):
    def test_owners_from_preloaded_rows_match_owners(self):
        """PERF-05: owners built from already-loaded rows equal AdminService.owners(guild_id).

        Seam under test:
        ``AdminService.owners_from(guild_id, spaces, characters, channels, lorebooks, thread_scopes)``.
        """
        transport, _ = discord_transport(admin_guilds=[1])
        app = create_app(":memory:", "https://pi.test", "client", "secret", "bot",
                         httpx.AsyncClient(transport=transport), enable_dashboard=False,
                         config_path="tests/nonexistent-config.yaml")
        store = app.state.store
        try:
            seed(store)
            store.bind_channel(1, 300, store.harbor)
            store.bind_channel(1, 100, store.plaza)
            store.create_lorebook(1, "Tales", "guild")
            store.add_lore(1, "thread", 900, "a")
            store.bind_channel(2, 200, store.faraway)
            admin = app.state.admin
            rows = (store.list_spaces(1), store.list_characters(1), store.list_channels(1),
                    store.list_lorebooks(1), store.thread_lore_scopes(1))
            self.assertEqual(admin.owners_from(1, *rows), admin.owners(1))
            self.assertTrue(any(o["kind"] == "book" for o in admin.owners(1)))
        finally:
            store.close()


class OwnersChannelNamesTests(unittest.TestCase):
    def test_owners_label_channels_by_name_when_known(self):
        """UX-01: owners/owners_from label known channels 'Channel: #name' and unknown ones 'Channel: <id>'."""
        transport, _ = discord_transport(admin_guilds=[1])
        app = create_app(":memory:", "https://pi.test", "client", "secret", "bot",
                         httpx.AsyncClient(transport=transport), enable_dashboard=False,
                         config_path="tests/nonexistent-config.yaml")
        store = app.state.store
        try:
            seed(store)
            store.bind_channel(1, 300, store.harbor)
            store.bind_channel(1, 100, store.plaza)
            admin = app.state.admin
            names = {100: "#scene", 999: "#other-guild-channel"}
            labels = lambda owners: [(o["id"], o["label"]) for o in owners if o["kind"] == "channel"]  # noqa: E731
            self.assertEqual(labels(admin.owners(1, channel_names=names)),
                             [(100, "Channel: #scene"), (300, "Channel: 300")])
            rows = (store.list_spaces(1), store.list_characters(1), store.list_channels(1),
                    store.list_lorebooks(1), store.thread_lore_scopes(1))
            self.assertEqual(admin.owners_from(1, *rows, channel_names=names), admin.owners(1, channel_names=names))
            # Names for ids that are not bound in this guild never appear as owners.
            self.assertFalse(any("other-guild" in o["label"] for o in admin.owners(1, channel_names=names)))
            # Without a mapping the labels are unchanged.
            self.assertEqual(labels(admin.owners(1)), [(100, "Channel: 100"), (300, "Channel: 300")])
            self.assertEqual(labels(admin.owners(1, channel_names={})), [(100, "Channel: 100"), (300, "Channel: 300")])
        finally:
            store.close()


class SourceGuardTests(unittest.TestCase):
    def test_no_raw_sql_or_private_store_access_in_ui_modules(self):
        """ARCH-02: dashboard.py and admin.py go through public Store methods only."""
        banned = ("store.all(", "store.one(", "store.db.", "store._", "self.store.all(", "self.store.one(")
        for name in ("dashboard.py", "admin.py"):
            text = (CORE / name).read_text(encoding="utf-8")
            for token in banned:
                self.assertNotIn(token, text, f"{name} contains {token!r}")


if __name__ == "__main__":
    unittest.main()
