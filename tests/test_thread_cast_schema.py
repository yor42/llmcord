"""MNT-05 follow-up: schema v10 gives ``thread_casts`` nullable ``guild_id`` and ``parent_id``.

- A v9 file upgrades to 10 with a pre-v10 backup; rows are backfilled with the single guild of the characters their
  cast references (NULL when the cast is empty, references no existing character, or spans guilds - the last also logs
  one warning with the thread id). ``parent_id`` stays NULL for old rows. An interrupted v9 file (columns present,
  user_version 9) is backfilled on its next open; NULL-guild rows at v10 are left alone.
- ``set_cast`` for a thread stores the binding's guild and the parent channel; ``_remove_from_thread_casts`` only scans
  the caller's guild.
- Newer-than-app (v11) files are refused.
Seams: ``Store`` on temp files and in memory.
"""
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from llmcord_core.store import Store


def columns(db, table):
    return [row[1] for row in db.execute(f"PRAGMA table_info({table})")]


class Fixture:
    def build(self, store):
        self.store = store
        self.harbor = store.create_space(1, "Harbor", "world")
        self.far = store.create_space(2, "Far", "world")
        add = lambda guild, world, name: store.add_character(guild, world, name, {"name": name}, None, [])  # noqa: E731
        self.alice, self.bob = add(1, self.harbor, "Alice"), add(1, self.harbor, "Bob")
        self.zara = add(2, self.far, "Zara")


class MigrationTests(Fixture, unittest.TestCase):
    def v9_file(self, folder):
        path = Path(folder) / "old.sqlite3"
        store = Store(path)
        self.build(store)
        store.close()
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("ALTER TABLE thread_casts DROP COLUMN guild_id")
            connection.execute("ALTER TABLE thread_casts DROP COLUMN parent_id")
            rows = {1001: [self.alice], 1002: [], 1003: [999999], 1004: [self.alice, self.zara], 1005: [999999, self.zara]}
            for thread, cast in rows.items():
                connection.execute("INSERT INTO thread_casts(thread_id,cast) VALUES(?,?)", (thread, json.dumps(cast)))
            connection.execute("PRAGMA user_version=9")
        return path

    def test_v9_file_upgrades_backs_up_and_backfills(self):
        """MNT-05: v9 -> v10 adds both columns, backs up first, backfills guild_id from the cast's characters."""
        with tempfile.TemporaryDirectory() as folder:
            path = self.v9_file(folder)
            with self.assertLogs("llmcord_core.store", "WARNING") as logs:
                store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 13)
                self.assertEqual(columns(store.db, "thread_casts"), ["thread_id", "cast", "guild_id", "parent_id"])
                rows = {r["thread_id"]: (r["guild_id"], r["parent_id"]) for r in store.all("SELECT * FROM thread_casts")}
                self.assertEqual(rows, {1001: (1, None), 1002: (None, None), 1003: (None, None),
                                        1004: (None, None), 1005: (2, None)})
            finally:
                store.close()
            warned = [m for m in logs.output if "1004" in m]
            self.assertEqual(len(logs.output), 1, logs.output)
            self.assertEqual(len(warned), 1)
            backups = list(Path(folder).glob("*.pre-v10-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            with closing(sqlite3.connect(backups[0])) as backup:
                self.assertEqual(backup.execute("PRAGMA user_version").fetchone()[0], 9)
                self.assertNotIn("guild_id", columns(backup, "thread_casts"))

    def test_reopening_does_not_backfill_again_or_back_up(self):
        """MNT-05: the backfill runs once; a later NULL guild row stays NULL and no second backup is made."""
        with tempfile.TemporaryDirectory() as folder:
            path = self.v9_file(folder)
            with self.assertLogs("llmcord_core.store", "WARNING"):
                Store(path).close()
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("UPDATE thread_casts SET guild_id=NULL WHERE thread_id=1001")
            Store(path).close()
            with closing(sqlite3.connect(path)) as connection:
                self.assertIsNone(connection.execute("SELECT guild_id FROM thread_casts WHERE thread_id=1001").fetchone()[0])
            self.assertEqual(len(list(Path(folder).glob("*.pre-*"))), 1)

    def test_interrupted_v9_file_with_columns_is_backfilled_on_open(self):
        """MNT-05: a v9 file that already has both columns (an interrupted upgrade) still gets the backfill."""
        with tempfile.TemporaryDirectory() as folder:
            path = self.v9_file(folder)
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("ALTER TABLE thread_casts ADD COLUMN guild_id INTEGER")
                connection.execute("ALTER TABLE thread_casts ADD COLUMN parent_id INTEGER")
            with self.assertLogs("llmcord_core.store", "WARNING"):
                store = Store(path)
            try:
                self.assertEqual(store.one("SELECT guild_id FROM thread_casts WHERE thread_id=1001")[0], 1)
                self.assertEqual(store.one("PRAGMA user_version")[0], 13)
            finally:
                store.close()

    def test_v10_file_null_row_is_not_touched(self):
        """MNT-05: a v10-shaped file with a NULL-guild row, upgraded to the current version, leaves that row alone."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v10.sqlite3"
            store = Store(path)
            self.build(store)
            store.execute("INSERT INTO thread_casts(thread_id,cast) VALUES(7,?)", (json.dumps([self.alice]),))
            store.close()
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("PRAGMA user_version=10")
            Store(path).close()
            with closing(sqlite3.connect(path)) as connection:
                self.assertIsNone(connection.execute("SELECT guild_id FROM thread_casts WHERE thread_id=7").fetchone()[0])
            self.assertEqual(len(list(Path(folder).glob("*.pre-v11-*.sqlite3"))), 1)

    def test_fresh_store_has_columns(self):
        """MNT-05: a new database is version 13 with the columns from SCHEMA."""
        store = Store()
        try:
            self.assertEqual(store.one("PRAGMA user_version")[0], 13)
            self.assertEqual(columns(store.db, "thread_casts"), ["thread_id", "cast", "guild_id", "parent_id"])
        finally:
            store.close()

    def test_v14_file_is_refused(self):
        """MNT-05: a version 14 file raises 'newer than this application'."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new.sqlite3"
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("PRAGMA user_version=14")
            with self.assertRaisesRegex(ValueError, "newer than this application"):
                Store(path)


class WriteTests(Fixture, unittest.TestCase):
    def setUp(self):
        self.build(Store())
        self.store.bind_channel(1, 100, self.harbor)
        self.store.bind_channel(2, 400, self.far)

    def tearDown(self):
        self.store.close()

    def row(self, thread):
        return tuple(self.store.one('SELECT guild_id,parent_id,"cast" FROM thread_casts WHERE thread_id=?', (thread,)))

    def test_set_cast_stores_guild_and_parent_and_updates_on_conflict(self):
        """MNT-05: a thread cast records the binding guild and parent channel; a rewrite keeps them."""
        self.store.set_cast(101, 100, [self.alice])
        self.assertEqual(self.row(101), (1, 100, json.dumps([self.alice])))
        self.store.set_cast(101, 100, [])
        self.assertEqual(self.row(101), (1, 100, "[]"))
        self.assertEqual(self.store.get_cast(101, 100), [])
        self.store.execute("UPDATE thread_casts SET guild_id=NULL,parent_id=NULL WHERE thread_id=101")
        self.store.set_cast(101, 100, [self.bob])
        self.assertEqual(self.row(101), (1, 100, json.dumps([self.bob])))

    def test_remove_from_thread_casts_only_scans_own_guild(self):
        """MNT-05 (guild isolation): rows of another guild, and NULL-guild rows, are not touched."""
        self.store.set_cast(101, 100, [self.alice, self.bob])
        self.store.execute("INSERT INTO thread_casts(thread_id,cast,guild_id,parent_id) VALUES(900,?,2,400)", (json.dumps([self.alice]),))
        self.store.execute("INSERT INTO thread_casts(thread_id,cast) VALUES(901,?)", (json.dumps([self.alice]),))
        with self.store.db:
            self.store._remove_from_thread_casts(1, self.alice)
        self.assertEqual(self.store.get_cast(101, 100), [self.bob])
        self.assertEqual(json.loads(self.row(900)[2]), [self.alice])
        self.assertEqual(json.loads(self.row(901)[2]), [self.alice])

    def test_prune_thread_casts_only_drops_the_named_character(self):
        """MNT-05: with ``only``, just that character is dropped when ineligible; other ineligible ids stay."""
        self.store.execute("INSERT INTO thread_casts(thread_id,cast,guild_id,parent_id) VALUES(101,?,1,100)",
                           (json.dumps([self.alice, self.bob, self.zara]),))
        with self.store.db:
            self.store._prune_thread_casts(1, 100, {self.bob}, only=self.alice)
        self.assertEqual(json.loads(self.row(101)[2]), [self.bob, self.zara])


if __name__ == "__main__":
    unittest.main()
