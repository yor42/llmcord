"""FEAT-37 part F1: schema v21 and the store API for kept phrases, server examples and the member opt-out (store only; no Discord)."""
import sqlite3
import tempfile
import unittest
from unittest import mock
from contextlib import closing
from pathlib import Path

from llmcord_core.admin_store import ConflictError
from llmcord_core.store import Store

G, H = 1, 2
ALICE, BOB = 5, 6
SCORES = {"p1": 0.123456, "margin": 0.0123456, "talk_gap": 0.5}


def v20_file(folder):
    path = Path(folder) / "v20.sqlite3"
    store = Store(path)
    store.set_favorites_mode(G, ALICE, "step_in")
    store.close()
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("DROP TABLE spoken_phrases")
        db.execute("DROP TABLE spoken_examples")
        db.execute("ALTER TABLE member_settings DROP COLUMN spoken_moves")
        db.execute("PRAGMA user_version=20")
    return path


class UpgradeTests(unittest.TestCase):
    def test_v20_file_upgrades_in_place_with_a_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = v20_file(folder)
            with closing(sqlite3.connect(path)) as db:
                self.assertNotIn("spoken_moves", {r[1] for r in db.execute("PRAGMA table_info(member_settings)")})
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 22)
                self.assertEqual(store.favorites_mode(G, ALICE), "step_in")
                self.assertTrue(store.spoken_member_on(G, ALICE))
                self.assertEqual(store.spoken_phrases(G), [])
                self.assertEqual(store.spoken_examples(G, "blackjack"), [])
                with self.assertRaises(sqlite3.IntegrityError):
                    store.db.execute("UPDATE member_settings SET spoken_moves=2")
                store.db.rollback()
            finally:
                store.close()
            backups = list(Path(folder).glob("v20.sqlite3.pre-v21-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            with closing(sqlite3.connect(backups[0])) as old:
                self.assertEqual(old.execute("PRAGMA user_version").fetchone()[0], 20)
                self.assertNotIn("spoken_moves", {r[1] for r in old.execute("PRAGMA table_info(member_settings)")})

    def test_reopening_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            path = v20_file(folder)
            Store(path).close()
            Store(path).close()
            self.assertEqual(len(list(Path(folder).glob("*.pre-v21-*.sqlite3"))), 1)

    def test_fresh_database_has_the_tables(self):
        store = Store()
        self.addCleanup(store.close)
        tables = {r[0] for r in store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertLessEqual({"spoken_phrases", "spoken_examples"}, tables)
        self.assertEqual(store.one("PRAGMA user_version")[0], 22)
        self.assertIn("spoken_moves", {r[1] for r in store.db.execute("PRAGMA table_info(member_settings)")})


class SpokenCase(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)
        self.s = self.store

    def on(self, guild=G):
        self.s.set_spoken_moves_enabled(guild, True)
        self.s.set_spoken_keep_phrases(guild, True)

    def keep(self, text="hit me please", user=ALICE, guild=G, **kw):
        args = dict(offered=["hit", "stand"], band="unsure", scores=SCORES, kind="false_negative", label="hit")
        args.update(kw)
        return self.s.keep_spoken_phrase(guild, user, "blackjack", text, **args)


class KeepTests(SpokenCase):
    def test_keep_is_a_noop_when_keeping_is_off(self):
        self.s.set_spoken_moves_enabled(G, True)
        self.assertIsNone(self.keep())
        self.assertEqual(self.s.spoken_phrases(G), [])

    def test_keep_is_a_noop_when_the_main_switch_is_off(self):
        self.on()
        self.s.set_spoken_moves_enabled(G, False)
        self.assertIsNone(self.keep())
        self.assertEqual(self.s.spoken_phrases(G), [])

    def test_keep_is_a_noop_when_the_member_is_off(self):
        self.on()
        self.s.set_spoken_member_on(G, ALICE, False)
        self.assertIsNone(self.keep())
        self.assertIsNotNone(self.keep(user=BOB))

    def test_keep_stores_the_phrase_without_the_member(self):
        self.on()
        ident = self.keep("  hit me please  ")
        row = self.s.spoken_phrases(G)[0]
        self.assertEqual(row["id"], ident)
        self.assertEqual((row["text"], row["offered"], row["band"], row["kind"], row["label"]), ("hit me please", ["hit", "stand"], "unsure", "false_negative", "hit"))
        self.assertEqual(row["scores"], {"p1": 0.1235, "margin": 0.0123, "talk_gap": 0.5})
        self.assertNotIn("user_id", row)

    def test_silent_phrase_has_an_empty_offered_list(self):
        self.on()
        self.keep(offered=[], band="silent", label="stand")
        self.assertEqual(self.s.spoken_phrases(G)[0]["offered"], [])

    def test_keep_caps_the_guild_at_500_dropping_the_oldest(self):
        self.on()
        self.s.SPOKEN_PHRASES_MAX = 5
        ids = [self.keep(f"phrase {i}") for i in range(8)]
        self.assertEqual([r["id"] for r in self.s.spoken_phrases(G)], ids[::-1][:5])
        self.assertEqual(Store.SPOKEN_PHRASES_MAX, 500)

    def test_keep_validates_text_label_and_game(self):
        self.on()
        for kw in ({"text": ""}, {"text": "x" * 201}, {"label": "fold"}, {"band": "loud"}, {"kind": "other"}):
            with self.subTest(kw), self.assertRaises(ValueError):
                self.keep(**kw)
        with self.assertRaises(ValueError):
            self.s.keep_spoken_phrase(G, ALICE, "doubt", "hi", [], "silent", SCORES, "false_negative", "talk")
        self.assertIsNotNone(self.keep("x" * 200, label="talk"))

    def test_keep_rechecks_the_switches_under_the_write_lock(self):
        self.on()
        real = self.s.spoken_keep_phrases
        calls = []

        def racy(guild):
            calls.append(guild)
            if len(calls) == 1:
                return True
            return real(guild)
        self.s.set_spoken_keep_phrases(G, False)
        with mock.patch.object(self.s, "spoken_keep_phrases", racy):
            self.assertIsNone(self.keep())
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.s.spoken_phrases(G), [])

    def test_keep_rechecks_the_member_switch_under_the_write_lock(self):
        self.on()
        real, calls = self.s.spoken_member_on, []

        def racy(guild, user):
            calls.append(user)
            return True if len(calls) == 1 else real(guild, user)
        self.s.set_spoken_member_on(G, ALICE, False)
        with mock.patch.object(self.s, "spoken_member_on", racy):
            self.assertIsNone(self.keep())
        self.assertEqual(self.s.spoken_phrases(G), [])

    def test_keep_tolerates_missing_scores_and_odd_offered(self):
        self.on()
        self.keep("a", scores=None, offered=None)
        self.keep("b", offered=[1, None, "hit"])
        rows = {r["text"]: r for r in self.s.spoken_phrases(G)}
        self.assertEqual(rows["a"]["scores"], {"p1": 0.0, "margin": 0.0, "talk_gap": 0.0})
        self.assertEqual((rows["a"]["offered"], rows["b"]["offered"]), ([], ["1", "None", "hit"]))

    def test_keep_is_scoped_to_the_guild_switches(self):
        self.on()
        self.assertIsNone(self.keep(guild=H))


class PurgeTests(SpokenCase):
    def test_member_off_purges_only_that_member_in_that_guild(self):
        self.on(); self.on(H)
        self.keep("a1"); self.keep("a2"); self.keep("b1", user=BOB); self.keep("ah", guild=H)
        self.s.set_spoken_member_on(G, ALICE, False)
        self.assertEqual([r["text"] for r in self.s.spoken_phrases(G)], ["b1"])
        self.assertEqual([r["text"] for r in self.s.spoken_phrases(H)], ["ah"])
        self.assertFalse(self.s.spoken_member_on(G, ALICE))
        self.assertTrue(self.s.spoken_member_on(H, ALICE))

    def test_member_switch_keeps_favorites_mode(self):
        self.s.set_favorites_mode(G, ALICE, "step_in")
        self.s.set_spoken_member_on(G, ALICE, False)
        self.assertEqual(self.s.favorites_mode(G, ALICE), "step_in")
        self.s.set_favorites_mode(G, ALICE, "lean")
        self.assertFalse(self.s.spoken_member_on(G, ALICE))
        self.s.set_spoken_member_on(G, ALICE, True)
        self.assertTrue(self.s.spoken_member_on(G, ALICE))

    def test_keep_switch_off_forgets_phrases_but_not_examples(self):
        self.on(); self.on(H)
        pid = self.keep("a1"); self.keep("h1", guild=H)
        self.s.promote_spoken_phrase(G, self.keep("a2"), "hit", ALICE)
        self.s.set_spoken_keep_phrases(G, False)
        self.assertEqual(self.s.spoken_phrases(G), [])
        self.assertEqual(len(self.s.spoken_phrases(H)), 1)
        self.assertEqual(len(self.s.spoken_examples(G, "blackjack")), 1)
        self.assertIsNotNone(pid)

    def test_main_switch_off_keeps_examples(self):
        self.on()
        self.s.promote_spoken_phrase(G, self.keep("a2"), "hit", ALICE)
        self.s.set_spoken_moves_enabled(G, False)
        self.assertEqual(len(self.s.spoken_examples(G, "blackjack")), 1)

    def test_retention_purges_old_phrases_only(self):
        import time
        self.on()
        old, new = self.keep("old"), self.keep("new")
        self.s.db.execute("UPDATE spoken_phrases SET created_at=? WHERE id=?", (time.time() - 40 * 86400, old))
        self.s.db.commit()
        self.s.expire_history(30)
        self.assertEqual([r["text"] for r in self.s.spoken_phrases(G)], ["new"])
        self.assertIsNotNone(new)


class ExampleTests(SpokenCase):
    def test_promote_creates_the_example_and_forgets_the_phrase(self):
        self.on()
        pid = self.keep("hit me", label="stand")
        eid = self.s.promote_spoken_phrase(G, pid, "hit", BOB)
        self.assertEqual(self.s.spoken_phrases(G), [])
        rows = self.s.spoken_examples(G, "blackjack")
        self.assertEqual([(r["id"], r["text"], r["label"]) for r in rows], [(eid, "hit me", "hit")])

    def test_promote_with_talk_label_and_duplicate_text_updates_the_label(self):
        self.on()
        first = self.s.promote_spoken_phrase(G, self.keep("hit me"), "hit", ALICE)
        second = self.s.promote_spoken_phrase(G, self.keep("hit me"), "talk", ALICE)
        self.assertEqual(first, second)
        rows = self.s.spoken_examples(G, "blackjack")
        self.assertEqual([(r["text"], r["label"]) for r in rows], [("hit me", "talk")])

    def test_promote_rejects_a_bad_label_and_keeps_the_phrase(self):
        self.on()
        pid = self.keep("hit me")
        with self.assertRaises(ValueError):
            self.s.promote_spoken_phrase(G, pid, "fold", ALICE)
        self.assertEqual(len(self.s.spoken_phrases(G)), 1)

    def test_example_cap(self):
        self.on()
        self.s.SPOKEN_EXAMPLES_MAX = 3
        for i in range(3):
            self.s.promote_spoken_phrase(G, self.keep(f"p{i}"), "hit", ALICE)
        pid = self.keep("one more")
        with self.assertRaisesRegex(ValueError, "already has 3 example phrases. Remove one first."):
            self.s.promote_spoken_phrase(G, pid, "hit", ALICE)
        self.assertEqual(len(self.s.spoken_phrases(G)), 1)
        self.assertEqual(Store.SPOKEN_EXAMPLES_MAX, 200)
        eid = self.s.spoken_examples(G, "blackjack")[0]["id"]
        self.assertTrue(self.s.delete_spoken_example(G, eid))
        self.s.promote_spoken_phrase(G, pid, "hit", ALICE)

    def test_delete_example_and_phrase(self):
        self.on()
        pid = self.keep("a")
        self.assertTrue(self.s.delete_spoken_phrase(G, pid))
        self.assertFalse(self.s.delete_spoken_phrase(G, pid))
        eid = self.s.promote_spoken_phrase(G, self.keep("b"), "hit", ALICE)
        self.assertTrue(self.s.delete_spoken_example(G, eid))
        self.assertEqual(self.s.spoken_examples(G, "blackjack"), [])

    def test_stale_expected_revision_is_a_conflict(self):
        self.on()
        rev = self.s.spoken_examples_revision(G)
        self.s.promote_spoken_phrase(G, self.keep("a"), "hit", ALICE, expected=rev)
        pid = self.keep("b")
        with self.assertRaises(ConflictError):
            self.s.promote_spoken_phrase(G, pid, "hit", ALICE, expected=rev)
        eid = self.s.spoken_examples(G, "blackjack")[0]["id"]
        with self.assertRaises(ConflictError):
            self.s.delete_spoken_example(G, eid, expected=rev)
        self.assertEqual(len(self.s.spoken_examples(G, "blackjack")), 1)
        self.assertEqual(len(self.s.spoken_phrases(G)), 1)
        self.s.delete_spoken_example(G, eid, expected=self.s.spoken_examples_revision(G))
        self.assertEqual(self.s.spoken_examples(G, "blackjack"), [])


class IsolationTests(SpokenCase):
    def test_other_guild_cannot_read_delete_or_promote(self):
        self.on()
        pid = self.keep("secret")
        eid = self.s.promote_spoken_phrase(G, self.keep("shared"), "hit", ALICE)
        self.assertEqual(self.s.spoken_phrases(H), [])
        self.assertEqual(self.s.spoken_examples(H, "blackjack"), [])
        self.assertFalse(self.s.delete_spoken_phrase(H, pid))
        self.assertFalse(self.s.delete_spoken_example(H, eid))
        with self.assertRaises(ValueError):
            self.s.promote_spoken_phrase(H, pid, "hit", BOB)
        self.assertEqual([r["text"] for r in self.s.spoken_phrases(G)], ["secret"])
        self.assertEqual(len(self.s.spoken_examples(G, "blackjack")), 1)

    def test_same_text_in_two_guilds_is_two_examples(self):
        self.on(); self.on(H)
        self.s.promote_spoken_phrase(G, self.keep("same"), "hit", ALICE)
        self.s.promote_spoken_phrase(H, self.keep("same", guild=H), "stand", BOB)
        self.assertEqual(self.s.spoken_examples(G, "blackjack")[0]["label"], "hit")
        self.assertEqual(self.s.spoken_examples(H, "blackjack")[0]["label"], "stand")


if __name__ == "__main__":
    unittest.main()
