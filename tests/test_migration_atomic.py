"""MNT-19: the whole schema upgrade (backup included) is one write-locked transaction.

- A failure at any step rolls back to the untouched old file (version, columns, sqlite_master, rows) with the one backup
  already taken; a retry succeeds.
- Two Stores opening the same legacy file at once both succeed with exactly one backup and no "duplicate column".
- An upgraded legacy file has the same schema as a fresh one; ``run_script`` keeps trigger bodies whole and never commits.
Seams: ``Store`` on temp files; patched ``_backfill_thread_cast_guilds`` / ``migrate_admin``.
"""
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from llmcord_core.admin_store import run_script
from llmcord_core.store import Store


def snapshot(path):
    with closing(sqlite3.connect(path)) as db:
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {
            "version": db.execute("PRAGMA user_version").fetchone()[0],
            "master": db.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name").fetchall(),
            "columns": {t: db.execute(f"PRAGMA table_info({t})").fetchall() for t in tables},
            "rows": {t: db.execute(f"SELECT * FROM {t}").fetchall() for t in tables},
        }


def legacy_file(folder, version=9):
    """A real older file: a current store with later-version columns removed and user_version set back."""
    path = Path(folder) / "old.sqlite3"
    store = Store(path)
    store.set_usage_footer(1, False)
    store.close()
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("ALTER TABLE thread_casts DROP COLUMN guild_id")
        db.execute("ALTER TABLE thread_casts DROP COLUMN parent_id")
        db.execute("ALTER TABLE guild_settings DROP COLUMN catchup_anywhere")
        db.execute("ALTER TABLE avatar_slots DROP COLUMN image_hash")
        db.execute("ALTER TABLE guild_settings DROP COLUMN currency_name")
        db.execute("DROP TABLE currency_balances")
        db.execute("DROP TABLE currency_ledger")
        db.execute("DROP TABLE currency_daily")
        for column in ("daily_amount", "daily_streak_bonus", "daily_streak_days"):
            db.execute(f"ALTER TABLE guild_settings DROP COLUMN {column}")
        db.execute("INSERT INTO thread_casts(thread_id,cast) VALUES(5,'[]')")
        db.execute(f"PRAGMA user_version={version}")
    return path


def backups(folder):
    return list(Path(folder).glob("*.pre-*.sqlite3"))


class AtomicUpgradeTests(unittest.TestCase):
    def test_late_failure_rolls_back_to_the_old_file_and_retry_succeeds(self):
        # Regression guard: also passes at HEAD (the failing step was already in its own rolled-back block).
        """MNT-19: a failing last data step leaves the file byte-for-byte schema-identical at the old version, with one backup; a retry finishes."""
        with tempfile.TemporaryDirectory() as folder:
            path = legacy_file(folder)
            before = snapshot(path)
            with mock.patch.object(Store, "_backfill_thread_cast_guilds", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    Store(path)
            self.assertEqual(snapshot(path), before)
            self.assertEqual(len(backups(folder)), 1)
            Store(path).close()
            after = snapshot(path)
            self.assertEqual(after["version"], 13)
            self.assertIn("catchup_anywhere", [c[1] for c in after["columns"]["guild_settings"]])
            self.assertEqual(len(backups(folder)), 2)

    def test_failure_after_admin_steps_rolls_the_admin_columns_back(self):
        """MNT-19: admin ALTERs/CREATEs done before a failure are undone with everything else."""
        with tempfile.TemporaryDirectory() as folder:
            path = legacy_file(folder, 6)
            before = snapshot(path)
            real = Store.migrate_admin

            def fail_after(store):
                real(store)
                raise RuntimeError("late")

            with mock.patch.object(Store, "migrate_admin", fail_after):
                with self.assertRaisesRegex(RuntimeError, "late"):
                    Store(path)
            self.assertEqual(snapshot(path), before)
            Store(path).close()
            self.assertEqual(snapshot(path)["version"], 13)

    def test_two_stores_upgrading_one_file_make_one_backup(self):
        """MNT-19: the second opener waits for the lock, sees version 13, and neither backs up nor re-adds a column."""
        with tempfile.TemporaryDirectory() as folder:
            path = legacy_file(folder)
            real = Store._backfill_thread_cast_guilds

            def slow(store):
                time.sleep(0.4)
                real(store)

            errors, stores = [], []
            barrier = threading.Barrier(2)

            def open_store():
                try:
                    barrier.wait()
                    stores.append(Store(path))
                except BaseException as exc:
                    errors.append(exc)

            with mock.patch.object(Store, "_backfill_thread_cast_guilds", slow):
                threads = [threading.Thread(target=open_store) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(60)
            for store in stores:
                store.close()
            self.assertEqual(errors, [])
            self.assertEqual(len(stores), 2)
            self.assertEqual(len(backups(folder)), 1)
            self.assertEqual(snapshot(path)["version"], 13)

    def test_busy_timeout_is_restored_after_upgrade(self):
        """MNT-19: the long upgrade wait does not stay on the connection."""
        with tempfile.TemporaryDirectory() as folder:
            store = Store(legacy_file(folder))
            try:
                self.assertEqual(store.one("PRAGMA busy_timeout")[0], 5000)
            finally:
                store.close()

    def test_upgraded_legacy_file_has_the_fresh_schema(self):
        """MNT-19: same tables, columns and objects as a new database (column order aside, since ALTER appends)."""
        with tempfile.TemporaryDirectory() as folder:
            path = legacy_file(folder, 5)
            Store(path).close()
            fresh = Path(folder) / "fresh.sqlite3"
            Store(fresh).close()
            old, new = snapshot(path), snapshot(fresh)
            self.assertEqual(old["version"], new["version"])
            self.assertEqual({(m[0], m[1], m[2]) for m in old["master"]}, {(m[0], m[1], m[2]) for m in new["master"]})
            self.assertEqual({t: sorted(tuple(c[1:]) for c in cols) for t, cols in old["columns"].items()},
                             {t: sorted(tuple(c[1:]) for c in cols) for t, cols in new["columns"].items()})

    def test_newer_file_is_still_refused_without_a_backup(self):
        """MNT-19: version 14 raises and writes nothing."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new.sqlite3"
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("PRAGMA user_version=14")
            with self.assertRaisesRegex(ValueError, "newer than this application"):
                Store(path)
            self.assertEqual(backups(folder), [])


class SchemaPinTests(unittest.TestCase):
    def test_schema_sql_matches_the_pin_generated_at_head(self):
        """MNT-19: sorted sqlite_master SQL for a new database and for an upgraded v5 file equals the pre-change output (tests/fixtures/schema_pin.json)."""
        pin = json.loads((Path(__file__).parent / "fixtures" / "schema_pin.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            fresh = Path(folder) / "fresh.sqlite3"
            Store(fresh).close()
            legacy = legacy_file(folder, 5)
            Store(legacy).close()
            sql = lambda p: sorted(r[0] for r in sqlite3.connect(p).execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"))  # noqa: E731
            self.assertEqual(sql(fresh), pin["fresh"])
            # The legacy file gets image_hash by ALTER (MNT-04), so its table text differs; compare avatar_slots by columns.
            not_slots = lambda rows: [r for r in rows if not r.startswith("CREATE TABLE avatar_slots")]  # noqa: E731
            self.assertEqual(not_slots(sql(legacy)), not_slots(pin["legacy_v5"]))
            cols = lambda p: sorted(r[1:] for r in sqlite3.connect(p).execute("PRAGMA table_info(avatar_slots)"))  # noqa: E731
            self.assertEqual(cols(legacy), cols(fresh))


class StandaloneMigrateAdminTests(unittest.TestCase):
    def test_standalone_migrate_admin_commits_its_own_transaction(self):
        """MNT-19: called outside a transaction it opens one and commits it."""
        store = Store()
        store.migrate_admin()
        self.assertFalse(store.db.in_transaction)
        store.close()


class RunScriptTests(unittest.TestCase):
    def test_trigger_body_stays_whole_and_nothing_commits(self):
        """MNT-19: statements split at real ends only, and a rollback undoes the whole script."""
        with closing(sqlite3.connect(":memory:")) as db:
            self.check_script(db)

    def check_script(self, db):
        db.execute("BEGIN IMMEDIATE")
        run_script(db, """CREATE TABLE a (x);
CREATE TABLE log (y);
CREATE TRIGGER t AFTER INSERT ON a BEGIN
 INSERT INTO log VALUES(1);
 INSERT INTO log VALUES(2);
END;
INSERT INTO a VALUES(1);
""")
        self.assertEqual(db.execute("SELECT COUNT(*) FROM log").fetchone()[0], 2)
        self.assertTrue(db.in_transaction)
        db.rollback()
        self.assertEqual(db.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
