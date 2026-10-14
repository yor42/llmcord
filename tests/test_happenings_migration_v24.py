"""MNT-84: schema v24 (item_uses_channel_time index, append-only item_use_posts table backfilled for existing uses) upgrades a v23 file in place.

The v23 file is a current store with the v24 additions removed again and user_version set back to 23.
"""
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from llmcord_core.store import Store

G, CH = 1, 100
RESULT = '{"text": "Yes.", "fields": [], "footer_note": ""}'


def v23_file(folder, uses=3):
    path = Path(folder) / "v23.sqlite3"
    Store(path).close()
    with closing(sqlite3.connect(path)) as db, db:
        for n in range(uses):
            db.execute("INSERT INTO item_uses(guild_id,user_id,channel_id,item_key,seed,result,used_at) VALUES(?,?,?,?,?,?,?)",
                       (G, 7, CH, "8ball", f"s{n}", RESULT, 1000.0 + n))
        db.execute("DROP INDEX IF EXISTS item_uses_channel_time")
        db.execute("DROP TABLE IF EXISTS item_use_posts")
        db.execute("PRAGMA user_version=23")
    return path


def item_use_rows(path):
    with closing(sqlite3.connect(path)) as db:
        return db.execute("SELECT * FROM item_uses ORDER BY id").fetchall()


class V23UpgradeTests(unittest.TestCase):
    def test_upgrade_adds_index_and_table_with_a_backup_and_keeps_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = v23_file(folder)
            before = item_use_rows(path)
            self.assertEqual(len(before), 3)
            Store(path).close()
            self.assertEqual(item_use_rows(path), before)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 24)
                indexes = {row[1]: row for row in db.execute("PRAGMA index_list(item_uses)")}
                self.assertIn("item_uses_channel_time", indexes)
                self.assertEqual([c[2] for c in db.execute("PRAGMA index_info(item_uses_channel_time)")], ["guild_id", "channel_id", "used_at"])
                self.assertEqual([r[1] for r in db.execute("PRAGMA table_info(item_use_posts)")], ["use_id", "posted_at"])
            backups = list(Path(folder).glob("v23.sqlite3.pre-v24-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            with closing(sqlite3.connect(backups[0])) as old:
                self.assertEqual(old.execute("PRAGMA user_version").fetchone()[0], 23)
                tables = {r[0] for r in old.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertNotIn("item_use_posts", tables)

    def test_every_existing_use_is_backfilled_as_posted_and_stays_visible(self):
        from llmcord_core.happenings import recent_happenings
        with tempfile.TemporaryDirectory() as folder:
            path = v23_file(folder)
            ids = [row[0] for row in item_use_rows(path)]
            store = Store(path)
            try:
                self.assertEqual(sorted(r["use_id"] for r in store.all("SELECT use_id FROM item_use_posts")), ids)
                self.assertEqual(len(recent_happenings(store, G, CH, now=1010.0)), 3)
                # A use made after the upgrade is unposted until marked.
                new_id = store.db.execute("INSERT INTO item_uses(guild_id,user_id,channel_id,item_key,seed,result,used_at) VALUES(?,?,?,?,?,?,?)",
                                          (G, 7, CH, "8ball", "new", RESULT, 1005.0)).lastrowid
                store.db.commit()
                self.assertEqual(len(recent_happenings(store, G, CH, now=1010.0)), 3)
                store.mark_use_posted(G, new_id)
                self.assertEqual(len(recent_happenings(store, G, CH, now=1010.0)), 3)
                self.assertEqual(len(store.all("SELECT 1 FROM item_use_posts")), 4)
            finally:
                store.close()

    def test_post_rows_are_append_only_references_to_uses(self):
        """MNT-84: item_use_posts.use_id references item_uses (foreign keys on) and is the primary key."""
        store = Store()
        self.addCleanup(store.close)
        columns = {r[1]: r for r in store.all("PRAGMA table_info(item_use_posts)")}
        self.assertEqual(columns["use_id"][5], 1)  # primary key
        self.assertEqual(columns["posted_at"][3], 1)  # NOT NULL
        self.assertEqual([(r[2], r[3]) for r in store.all("PRAGMA foreign_key_list(item_use_posts)")], [("item_uses", "use_id")])

    def test_migrating_twice_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            path = v23_file(folder)
            Store(path).close()
            with closing(sqlite3.connect(path)) as db:
                first = db.execute("SELECT * FROM item_use_posts ORDER BY use_id").fetchall()
            Store(path).close()
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute("SELECT * FROM item_use_posts ORDER BY use_id").fetchall(), first)
            self.assertEqual(len(list(Path(folder).glob("*.pre-v24-*.sqlite3"))), 1)

    def test_fresh_store_is_v24_with_index_and_empty_post_table(self):
        store = Store()
        self.addCleanup(store.close)
        self.assertEqual(store.one("PRAGMA user_version")[0], 24)
        self.assertIn("item_uses_channel_time", {r[1] for r in store.all("PRAGMA index_list(item_uses)")})
        self.assertEqual(store.all("SELECT * FROM item_use_posts"), [])


if __name__ == "__main__":
    unittest.main()
