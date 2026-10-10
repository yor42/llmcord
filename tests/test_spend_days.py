import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

from llmcord_core.store import Store

DAY1 = 1_700_000_000.0  # 2023-11-14 UTC
DAY2 = DAY1 + 86400


def usage(created_at, cost, guild_id=1):
    return SimpleNamespace(guild_id=guild_id, profile="p", model="m", role="dialogue", input_tokens=1, output_tokens=1,
                           cached_tokens=0, reasoning_tokens=0, cost_usd=cost, cost_basis="x", created_at=created_at,
                           channel_id=None, feature="")


def spend(store):
    return {r["day"]: (r["cost_usd"], r["unpriced_calls"]) for r in store.all("SELECT * FROM spend_days")}


class SpendDaysTests(unittest.TestCase):
    def v5_file(self, folder):
        path = Path(folder) / "old.sqlite3"
        store = Store(path)
        for u in (usage(DAY1, 0.5), usage(DAY1 + 10, None), usage(DAY2, 1.25), usage(DAY2 + 5, 0.25)):
            store.db.execute("INSERT INTO model_usage(guild_id,profile,model,role,input_tokens,output_tokens,cached_tokens,reasoning_tokens,cost_usd,cost_basis,created_at) VALUES(1,'p','m','d',1,1,0,0,?,'x',?)", (u.cost_usd, u.created_at))
        store.db.commit()
        store.close()
        with closing(sqlite3.connect(path)) as connection, connection:
            for table in ("bot_settings", "spend_days", "budget_notices"):
                connection.execute(f"DROP TABLE {table}")
            connection.execute("PRAGMA user_version=5")
        return path

    def test_fresh_store_is_v8_with_default_settings(self):
        """FEAT-16: a new database is the current version with one default bot_settings row and empty spend tables."""
        store = Store()
        try:
            self.assertEqual(store.one("PRAGMA user_version")[0], 16)
            row = store.one("SELECT * FROM bot_settings")
            self.assertEqual((row["id"], row["soft_cap_usd"], row["hard_cap_usd"], row["reset_day"], row["channel_notice"], row["revision"]), (1, None, None, 1, 0, 0))
            self.assertEqual(store.one("SELECT COUNT(*) FROM bot_settings")[0], 1)
            self.assertEqual(store.one("SELECT COUNT(*) FROM spend_days")[0], 0)
            self.assertEqual(store.one("SELECT COUNT(*) FROM budget_notices")[0], 0)
        finally:
            store.close()

    def test_v5_file_upgrades_with_one_backup_and_backfill(self):
        """FEAT-16: a v5 file gets one .pre-v6 backup (still v5), backfills spend_days per UTC day, and reopening neither backs up nor doubles."""
        with tempfile.TemporaryDirectory() as folder:
            path = self.v5_file(folder)
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 16)
                self.assertEqual(spend(store), {"2023-11-14": (0.5, 1), "2023-11-15": (1.5, 0)})
            finally:
                store.close()
            backups = list(Path(folder).glob("*.pre-v6-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(len(list(Path(folder).glob("*.pre-*"))), 1)
            with closing(sqlite3.connect(backups[0])) as backup:
                self.assertEqual(backup.execute("PRAGMA user_version").fetchone()[0], 5)
            store = Store(path)
            try:
                self.assertEqual(spend(store), {"2023-11-14": (0.5, 1), "2023-11-15": (1.5, 0)})
            finally:
                store.close()
            self.assertEqual(len(list(Path(folder).glob("*.pre-*"))), 1)

    def test_record_usage_updates_spend_in_same_transaction(self):
        """FEAT-16: record_model_usage upserts spend_days, also for guild-less calls, which write no model_usage row."""
        store = Store()
        try:
            store.record_model_usage(usage(DAY1, 0.5))
            store.record_model_usage(usage(DAY1 + 1, 0.25, guild_id=None))
            self.assertEqual(spend(store), {"2023-11-14": (0.75, 0)})
            self.assertEqual(store.one("SELECT COUNT(*) FROM model_usage")[0], 1)
            self.assertFalse(store.db.in_transaction)
        finally:
            store.close()

    def test_null_cost_counts_unpriced(self):
        """FEAT-16: a NULL cost adds no cost and one unpriced call."""
        store = Store()
        try:
            store.record_model_usage(usage(DAY1, None))
            store.record_model_usage(usage(DAY1, None, guild_id=None))
            store.record_model_usage(usage(DAY1, 1.0))
            self.assertEqual(spend(store), {"2023-11-14": (1.0, 2)})
        finally:
            store.close()

    def test_expire_history_keeps_spend_but_prunes_over_400_days(self):
        """FEAT-16 (D19): expire_history no longer touches model_usage, keeps spend_days, and prunes days older than 400."""
        now = DAY1 + 500 * 86400
        store = Store()
        try:
            store.record_model_usage(usage(now - 100 * 86400, 1.0))
            store.record_model_usage(usage(now - 450 * 86400, 2.0))
            store.expire_history(30, now=now)
            self.assertEqual(store.one("SELECT COUNT(*) FROM model_usage")[0], 2)
            self.assertEqual(list(spend(store).values()), [(1.0, 0)])
        finally:
            store.close()

    def test_prune_boundary_is_400_days(self):
        """FEAT-16: a spend day exactly 400 days old stays; 401 days old is removed."""
        now = DAY1 + 500 * 86400
        store = Store()
        try:
            store.record_model_usage(usage(now - 400 * 86400, 1.0))
            store.record_model_usage(usage(now - 401 * 86400, 2.0))
            store.expire_history(30, now=now)
            self.assertEqual(list(spend(store).values()), [(1.0, 0)])
        finally:
            store.close()

    def test_interrupted_upgrade_does_not_double_count(self):
        """FEAT-16: resetting user_version to 5 after an upgrade and reopening leaves spend_days unchanged."""
        with tempfile.TemporaryDirectory() as folder:
            path = self.v5_file(folder)
            Store(path).close()
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("PRAGMA user_version=5")
            store = Store(path)
            try:
                self.assertEqual(spend(store), {"2023-11-14": (0.5, 1), "2023-11-15": (1.5, 0)})
                self.assertEqual(store.one("PRAGMA user_version")[0], 16)
            finally:
                store.close()

    def test_reopen_at_v6_leaves_spend_days_untouched(self):
        """FEAT-16: a guild-less spend row in a v6 file survives reopening (no backfill)."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cur.sqlite3"
            store = Store(path)
            store.record_model_usage(usage(DAY1, 0.75, guild_id=None))
            store.close()
            store = Store(path)
            try:
                self.assertEqual(spend(store), {"2023-11-14": (0.75, 0)})
            finally:
                store.close()

    def test_bot_settings_checks(self):
        """FEAT-16: CHECK constraints reject a bad reset_day, negative caps, and a second row."""
        store = Store()
        try:
            for sql in ("UPDATE bot_settings SET reset_day=0", "UPDATE bot_settings SET reset_day=29",
                        "UPDATE bot_settings SET soft_cap_usd=-1", "UPDATE bot_settings SET hard_cap_usd=-0.5",
                        "UPDATE bot_settings SET channel_notice=2", "INSERT INTO bot_settings(id) VALUES(2)"):
                with self.assertRaises(sqlite3.IntegrityError, msg=sql):
                    store.db.execute(sql)
        finally:
            store.close()

    def test_v17_file_is_rejected(self):
        """FEAT-16: a version 17 file raises 'newer than this application'."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new.sqlite3"
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("PRAGMA user_version=17")
            with self.assertRaisesRegex(ValueError, "newer than this application"):
                Store(path)

    def test_catchup_accessors_default_roundtrip_and_isolation(self):
        """FEAT-15: catchup_anywhere is False without a row, round-trips, and is per guild."""
        store = Store()
        try:
            self.assertFalse(store.catchup_anywhere(1))
            store.set_catchup_anywhere(1, True)
            self.assertTrue(store.catchup_anywhere(1))
            self.assertFalse(store.catchup_anywhere(2))
            store.set_catchup_anywhere(2, True)
            store.set_catchup_anywhere(1, False)
            self.assertFalse(store.catchup_anywhere(1))
            self.assertTrue(store.catchup_anywhere(2))
        finally:
            store.close()

    def downgrade(self, path, version, drop_column=True):
        with closing(sqlite3.connect(path)) as connection, connection:
            if drop_column:
                connection.execute("ALTER TABLE guild_settings DROP COLUMN catchup_anywhere")
            connection.execute(f"PRAGMA user_version={version}")

    def test_v6_file_upgrades_to_v7_with_backup_and_column(self):
        """FEAT-15: a v6 file gets one .pre-v7 backup (still v6 without the column), the column defaults to 0 and keeps data, and reopening does not back up again."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.sqlite3"
            store = Store(path)
            store.set_usage_footer(1, False)
            store.close()
            self.downgrade(path, 6)
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 16)
                self.assertFalse(store.catchup_anywhere(1))
                self.assertFalse(store.usage_footer_enabled(1))
                self.assertEqual(store.one("SELECT catchup_anywhere FROM guild_settings WHERE guild_id=1")[0], 0)
            finally:
                store.close()
            backups = list(Path(folder).glob("*.pre-v7-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(len(list(Path(folder).glob("*.pre-*"))), 1)
            with closing(sqlite3.connect(backups[0])) as backup:
                self.assertEqual(backup.execute("PRAGMA user_version").fetchone()[0], 6)
                self.assertNotIn("catchup_anywhere", [r[1] for r in backup.execute("PRAGMA table_info(guild_settings)")])
            Store(path).close()
            self.assertEqual(len(list(Path(folder).glob("*.pre-*"))), 1)

    def test_v5_file_reaches_v8_with_backfill_and_column(self):
        """FEAT-15: a v5 file without the column still backfills spend_days and ends at v8 with the column."""
        with tempfile.TemporaryDirectory() as folder:
            path = self.v5_file(folder)
            self.downgrade(path, 5)
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 16)
                self.assertEqual(spend(store), {"2023-11-14": (0.5, 1), "2023-11-15": (1.5, 0)})
                self.assertFalse(store.catchup_anywhere(1))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
