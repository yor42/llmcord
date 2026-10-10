"""FEAT-23 part A: schema v15, member favorites, cast and favorites limits (store only; no Discord)."""
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from llmcord_core.admin_store import ConflictError
from llmcord_core.store import Store

G, H = 1, 2
ALICE, BOB = 5, 6


class FavoritesCase(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)
        s = self.store
        self.w1, self.w2, self.w3 = (s.create_space(G, n, "world") for n in ("One", "Two", "Three"))
        self.hub = s.create_space(G, "Hub", "hub")
        s.link_world(G, self.hub, self.w1)
        s.link_world(G, self.hub, self.w2)
        self.chars = {n: s.create_character(G, w, n) for n, w in (("Ann", self.w1), ("Bea", self.w1), ("Cy", self.w2), ("Di", self.w3))}
        self.other = s.create_space(H, "Other", "world")
        self.foreign = s.create_character(H, self.other, "Zed")

    def c(self, name):
        return self.chars[name]

    def ids(self, rows):
        return [r["character_id"] for r in rows]


class MigrationTests(unittest.TestCase):
    def v14_file(self, folder):
        path = Path(folder) / "old.sqlite3"
        store = Store(path)
        w = store.create_space(G, "W", "world")
        store.create_character(G, w, "Ann")
        store.set_game_settings(G, 2, 50)
        store.close()
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("DROP TABLE member_favorites")
            db.execute("DROP TABLE member_settings")
            for column in ("max_cast", "max_favorites"):
                db.execute(f"ALTER TABLE guild_settings DROP COLUMN {column}")
            db.execute("ALTER TABLE spaces DROP COLUMN hub_tone")
            db.execute("PRAGMA user_version=14")
        return path

    def test_v14_file_upgrades_with_a_backup_and_keeps_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.v14_file(folder)
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 15)
                self.assertEqual(store.game_settings(G), {"min_bet": 2, "max_bet": 50})
                self.assertEqual(store.cast_limits(G), {"max_cast": 5, "max_favorites": 5})
                self.assertEqual(store.one("SELECT hub_tone FROM spaces")[0], "in_character")
                self.assertEqual(store.favorites(G, ALICE), [])
                self.assertEqual(store.favorites_mode(G, ALICE), "lean")
                store.add_favorite(G, ALICE, store.character(G, "Ann")["id"])
            finally:
                store.close()
            backups = list(Path(folder).glob("old.sqlite3.pre-v15-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            with closing(sqlite3.connect(backups[0])) as db:
                self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 14)
                self.assertNotIn("member_favorites", {r[0] for r in db.execute("SELECT name FROM sqlite_master")})
            Store(path).close()
            self.assertEqual(len(list(Path(folder).glob("*.pre-*"))), 1)

    def test_cascade_delete_works_on_an_upgraded_file(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(self.v14_file(folder))
            try:
                ann = store.character(G, "Ann")["id"]
                store.add_favorite(G, ALICE, ann)
                store.delete_character(G, ann, store.owner_revision(G, "character", ann))
                self.assertEqual(store.favorites(G, ALICE), [])
                self.assertEqual(store.db.execute("SELECT COUNT(*) FROM member_favorites").fetchone()[0], 0)
            finally:
                store.close()

    def test_fresh_store_has_the_v15_schema(self):
        store = Store()
        try:
            names = {r[0] for r in store.db.execute("SELECT name FROM sqlite_master")}
            self.assertTrue({"member_favorites", "member_settings"} <= names)
            self.assertEqual(store.cast_limits(G), {"max_cast": 5, "max_favorites": 5})
            self.assertIn("hub_tone", {r[1] for r in store.db.execute("PRAGMA table_info(spaces)")})
        finally:
            store.close()


class LimitTests(FavoritesCase):
    def test_set_and_read_limits(self):
        self.assertEqual(self.store.set_cast_limits(G, 6, 3), {"max_cast": 6, "max_favorites": 3})
        self.assertEqual(self.store.cast_limits(G), {"max_cast": 6, "max_favorites": 3})
        self.assertEqual(self.store.cast_limits(H), {"max_cast": 5, "max_favorites": 5})

    def test_bad_values_are_refused(self):
        for bad in (0, 16, -1, 2.5, "5", None, True):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, r"whole numbers from 1 to 15\.$"):
                self.store.set_cast_limits(G, bad, 5)
            with self.assertRaises(ValueError):
                self.store.set_cast_limits(G, 5, bad)
        self.assertEqual(self.store.cast_limits(G), {"max_cast": 5, "max_favorites": 5})

    def test_stale_expected_is_a_conflict(self):
        old = self.store.cast_limits(G)
        self.store.set_cast_limits(G, 7, 5, expected=old)
        with self.assertRaises(ConflictError):
            self.store.set_cast_limits(G, 8, 5, expected=old)
        self.assertEqual(self.store.cast_limits(G)["max_cast"], 7)


class CastLimitTests(FavoritesCase):
    def setUp(self):
        super().setUp()
        s = self.store
        self.names = [f"N{i}" for i in range(8)]
        self.ids_ = [s.create_character(G, self.w1, n) for n in self.names]
        s.bind_channel(G, 100, self.w1)

    def test_default_limit_is_five_with_the_new_message(self):
        self.store.set_cast(100, None, self.ids_[:5])
        with self.assertRaisesRegex(ValueError, r"^A cast can have at most 5 characters\.$"):
            self.store.set_cast(100, None, self.ids_[:6])

    def test_raised_limit_allows_more(self):
        self.store.set_cast_limits(G, 6, 5)
        self.store.set_cast(100, None, self.ids_[:6])
        self.assertEqual(self.store.get_cast(100), self.ids_[:6])
        with self.assertRaisesRegex(ValueError, "at most 6 characters"):
            self.store.set_cast(100, None, self.ids_[:7])

    def test_lowering_keeps_a_longer_cast_which_can_shrink_but_not_grow(self):
        self.store.set_cast_limits(G, 6, 5)
        self.store.set_cast(100, None, self.ids_[:6])
        self.store.set_cast_limits(G, 3, 5)
        self.assertEqual(self.store.get_cast(100), self.ids_[:6])
        self.store.set_cast(100, None, self.ids_[:5])
        self.store.set_cast(100, None, self.ids_[1:6])
        with self.assertRaisesRegex(ValueError, "at most 3 characters"):
            self.store.set_cast(100, None, self.ids_[:6])
        self.store.set_cast(100, None, self.ids_[:3])
        with self.assertRaises(ValueError):
            self.store.set_cast(100, None, self.ids_[:4])

    def test_thread_without_its_own_cast_compares_with_the_parent_cast(self):
        self.store.set_cast_limits(G, 6, 5)
        self.store.set_cast(100, None, self.ids_[:6])
        self.store.set_cast_limits(G, 3, 5)
        self.store.set_cast(555, 100, self.ids_[:5])
        with self.assertRaisesRegex(ValueError, "at most 3 characters"):
            self.store.set_cast(556, 100, self.ids_[:7])

    def test_default_cast_after_lowering_can_shrink_not_grow(self):
        self.store.set_cast_limits(G, 6, 5)
        self.store.set_cast(100, None, self.ids_[:6], default=True)
        self.store.set_cast_limits(G, 3, 5)
        self.store.set_cast(100, None, self.ids_[:4], default=True)
        with self.assertRaisesRegex(ValueError, "at most 3 characters"):
            self.store.set_cast(100, None, self.ids_[:5], default=True)

    def test_limit_is_per_guild(self):
        self.store.set_cast_limits(H, 1, 1)
        self.store.set_cast(100, None, self.ids_[:5])


class FavoritesTests(FavoritesCase):
    def test_add_orders_and_lists(self):
        for n in ("Cy", "Ann", "Bea"):
            self.store.add_favorite(G, ALICE, self.c(n))
        rows = self.store.favorites(G, ALICE)
        self.assertEqual([(r["name"], r["position"]) for r in rows], [("Cy", 0), ("Ann", 1), ("Bea", 2)])
        self.assertEqual(rows[0], {"character_id": self.c("Cy"), "name": "Cy", "world_id": self.w2, "position": 0})

    def test_duplicate_is_refused(self):
        self.store.add_favorite(G, ALICE, self.c("Ann"))
        with self.assertRaisesRegex(ValueError, r"already.*\.$"):
            self.store.add_favorite(G, ALICE, self.c("Ann"))

    def test_limit_is_refused_with_advice(self):
        self.store.set_cast_limits(G, 5, 2)
        self.store.add_favorite(G, ALICE, self.c("Ann"))
        self.store.add_favorite(G, ALICE, self.c("Bea"))
        with self.assertRaisesRegex(ValueError, r"^You can have at most 2 favorites\. Remove one first\.$"):
            self.store.add_favorite(G, ALICE, self.c("Cy"))
        self.store.add_favorite(G, BOB, self.c("Cy"))

    def test_lowering_the_limit_keeps_the_list(self):
        for n in ("Ann", "Bea", "Cy"):
            self.store.add_favorite(G, ALICE, self.c(n))
        self.store.set_cast_limits(G, 5, 1)
        self.assertEqual(len(self.store.favorites(G, ALICE)), 3)
        with self.assertRaises(ValueError):
            self.store.add_favorite(G, ALICE, self.c("Di"))
        self.store.remove_favorite(G, ALICE, self.c("Ann"))
        with self.assertRaises(ValueError):
            self.store.add_favorite(G, ALICE, self.c("Di"))

    def test_missing_archived_and_other_guild_characters_are_refused(self):
        with self.assertRaisesRegex(ValueError, r"\.$"):
            self.store.add_favorite(G, ALICE, 99999)
        with self.assertRaises(ValueError):
            self.store.add_favorite(G, ALICE, self.foreign)
        with self.assertRaises(ValueError):
            self.store.add_favorite(H, ALICE, self.c("Ann"))
        self.store.archive_character(G, self.c("Ann"), True)
        with self.assertRaises(ValueError):
            self.store.add_favorite(G, ALICE, self.c("Ann"))
        self.assertEqual(self.store.favorites(G, ALICE), [])

    def test_remove_renumbers_and_clear_counts(self):
        for n in ("Ann", "Bea", "Cy"):
            self.store.add_favorite(G, ALICE, self.c(n))
        self.assertTrue(self.store.remove_favorite(G, ALICE, self.c("Ann")))
        self.assertFalse(self.store.remove_favorite(G, ALICE, self.c("Ann")))
        self.assertEqual([(r["name"], r["position"]) for r in self.store.favorites(G, ALICE)], [("Bea", 0), ("Cy", 1)])
        self.store.add_favorite(G, ALICE, self.c("Di"))
        self.assertEqual([r["position"] for r in self.store.favorites(G, ALICE)], [0, 1, 2])
        self.assertEqual(self.store.clear_favorites(G, ALICE), 3)
        self.assertEqual(self.store.clear_favorites(G, ALICE), 0)

    def test_members_and_guilds_are_independent(self):
        self.store.add_favorite(G, ALICE, self.c("Ann"))
        self.store.add_favorite(G, BOB, self.c("Cy"))
        self.assertEqual(self.ids(self.store.favorites(G, ALICE)), [self.c("Ann")])
        self.assertEqual(self.ids(self.store.favorites(G, BOB)), [self.c("Cy")])
        self.assertEqual(self.store.favorites(H, ALICE), [])
        self.assertFalse(self.store.remove_favorite(H, ALICE, self.c("Ann")))
        self.assertEqual(self.store.clear_favorites(H, ALICE), 0)
        self.assertEqual(self.store.clear_favorites(G, BOB), 1)
        self.assertEqual(self.ids(self.store.favorites(G, ALICE)), [self.c("Ann")])

    def test_archived_favorite_is_hidden_and_returns_when_restored(self):
        self.store.add_favorite(G, ALICE, self.c("Ann"))
        self.store.add_favorite(G, ALICE, self.c("Bea"))
        self.store.archive_character(G, self.c("Ann"), True)
        self.assertEqual(self.ids(self.store.favorites(G, ALICE)), [self.c("Bea")])
        self.store.archive_character(G, self.c("Ann"), False)
        self.assertEqual(self.ids(self.store.favorites(G, ALICE)), [self.c("Ann"), self.c("Bea")])

    def test_archived_favorites_do_not_count_toward_the_limit(self):
        self.store.set_cast_limits(G, 5, 2)
        self.store.add_favorite(G, ALICE, self.c("Ann"))
        self.store.add_favorite(G, ALICE, self.c("Bea"))
        self.store.archive_character(G, self.c("Ann"), True)
        self.store.add_favorite(G, ALICE, self.c("Cy"))
        self.store.archive_character(G, self.c("Ann"), False)
        self.assertEqual(len(self.store.favorites(G, ALICE)), 3)
        with self.assertRaises(ValueError):
            self.store.add_favorite(G, ALICE, self.c("Di"))

    def test_clear_counts_only_visible_but_deletes_all(self):
        self.store.add_favorite(G, ALICE, self.c("Ann"))
        self.store.add_favorite(G, ALICE, self.c("Bea"))
        self.store.archive_character(G, self.c("Ann"), True)
        self.assertEqual(self.store.clear_favorites(G, ALICE), 1)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM member_favorites").fetchone()[0], 0)

    def test_deleting_a_character_removes_its_favorites(self):
        self.store.add_favorite(G, ALICE, self.c("Ann"))
        self.store.add_favorite(G, BOB, self.c("Ann"))
        self.store.add_favorite(G, BOB, self.c("Bea"))
        rev = self.store.owner_revision(G, "character", self.c("Ann"))
        self.store.delete_character(G, self.c("Ann"), rev)
        self.assertEqual(self.store.favorites(G, ALICE), [])
        self.assertEqual(self.ids(self.store.favorites(G, BOB)), [self.c("Bea")])
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM member_favorites WHERE character_id=?", (self.c("Ann"),)).fetchone()[0], 0)


class ModeTests(FavoritesCase):
    def test_default_set_and_isolation(self):
        self.assertEqual(self.store.favorites_mode(G, ALICE), "lean")
        self.store.set_favorites_mode(G, ALICE, "step_in")
        self.assertEqual(self.store.favorites_mode(G, ALICE), "step_in")
        self.assertEqual(self.store.favorites_mode(G, BOB), "lean")
        self.assertEqual(self.store.favorites_mode(H, ALICE), "lean")
        self.store.set_favorites_mode(G, ALICE, "lean")
        self.assertEqual(self.store.favorites_mode(G, ALICE), "lean")

    def test_bad_mode_is_refused(self):
        for bad in ("", "Lean", "step in", None, 1):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, r"\.$"):
                self.store.set_favorites_mode(G, ALICE, bad)
        self.assertEqual(self.store.favorites_mode(G, ALICE), "lean")


class EligibleTests(FavoritesCase):
    def test_filters_by_world_and_hub_links(self):
        for n in ("Cy", "Ann", "Di", "Bea"):
            self.store.add_favorite(G, ALICE, self.c(n))
        names = lambda space: [r["name"] for r in self.store.eligible_favorites(G, ALICE, space)]  # noqa: E731
        self.assertEqual(names(self.w1), ["Ann", "Bea"])
        self.assertEqual(names(self.w3), ["Di"])
        self.assertEqual(names(self.hub), ["Cy", "Ann", "Bea"])
        self.store.unlink_world(G, self.hub, self.w2)
        self.assertEqual(names(self.hub), ["Ann", "Bea"])

    def test_other_guild_space_gives_nothing(self):
        self.store.add_favorite(G, ALICE, self.c("Ann"))
        self.assertEqual(self.store.eligible_favorites(H, ALICE, self.w1), [])
        self.assertEqual(self.store.eligible_favorites(G, ALICE, self.other), [])
        self.assertEqual(self.store.eligible_favorites(G, BOB, self.w1), [])


if __name__ == "__main__":
    unittest.main()
