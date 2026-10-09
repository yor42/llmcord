"""Timezone settings and schema v5 (FEAT-01; decisions D14, D15).

Seam: ``Store`` (AdminStore helpers) in memory, plus temp files for the v4 -> v5 migration.
"""
import sqlite3
import stat
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from llmcord_core.store import Store

NEW_GUILD_COLUMNS = {"timezone", "turn_log_enabled", "turn_log_days"}
TURN_LOG_COLUMNS = {"id", "guild_id", "channel_id", "message_id", "stage", "profile", "model", "status",
                    "reference_id", "error_detail", "request_text", "response_text", "input_tokens",
                    "output_tokens", "created_at"}


def columns(db, table):
    return {row[1]: row for row in db.execute(f"PRAGMA table_info({table})")}


class SchemaV5Tests(unittest.TestCase):
    def v4_file(self, folder):
        """A v4-era file: a current store with the v5 things removed and user_version=4."""
        path = Path(folder) / "old.sqlite3"
        store = Store(path)
        store.execute("INSERT INTO guild_settings(guild_id,preset_id,preset_revision,usage_footer) VALUES(1,7,3,0)")
        store.close()
        with closing(sqlite3.connect(path)) as connection, connection:
            have = columns(connection, "guild_settings")
            for name in NEW_GUILD_COLUMNS & set(have):
                connection.execute(f"ALTER TABLE guild_settings DROP COLUMN {name}")
            connection.execute("DROP TABLE IF EXISTS user_timezones")
            connection.execute("DROP TABLE IF EXISTS turn_log")
            connection.execute("PRAGMA user_version=4")
        return path

    def assert_new_schema(self, db):
        guild = columns(db, "guild_settings")
        self.assertTrue(NEW_GUILD_COLUMNS <= set(guild))
        self.assertEqual((guild["timezone"][2].upper(), guild["timezone"][3], guild["timezone"][4]), ("TEXT", 1, "''"))
        self.assertEqual((guild["turn_log_enabled"][3], guild["turn_log_enabled"][4]), (1, "0"))
        self.assertEqual((guild["turn_log_days"][3], guild["turn_log_days"][4]), (1, "14"))
        user = columns(db, "user_timezones")
        self.assertEqual(set(user), {"guild_id", "user_id", "timezone", "updated_at"})
        self.assertEqual(sorted((r[5], r[1]) for r in user.values() if r[5]), [(1, "guild_id"), (2, "user_id")])
        self.assertEqual(set(columns(db, "turn_log")), TURN_LOG_COLUMNS)

    def test_fresh_database_is_v7_with_new_schema(self):
        """FEAT-01 (D14): a new database is at version 6 with the timezone and turn_log schema."""
        store = Store()
        try:
            self.assertEqual(store.one("PRAGMA user_version")[0], 9)
            self.assert_new_schema(store.db)
        finally:
            store.close()

    def test_v4_file_upgrades_with_backup_and_keeps_rows(self):
        """FEAT-01 (D14): a v4 file gets one .pre-v5 backup (still v4, no new schema), upgrades, keeps its rows,
        and reopening makes no second backup."""
        with tempfile.TemporaryDirectory() as folder:
            path = self.v4_file(folder)
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 9)
                self.assert_new_schema(store.db)
                row = store.one("SELECT * FROM guild_settings WHERE guild_id=1")
                self.assertEqual((row["preset_id"], row["preset_revision"], row["usage_footer"]), (7, 3, 0))
                self.assertEqual((row["timezone"], row["turn_log_enabled"], row["turn_log_days"]), ("", 0, 14))
            finally:
                store.close()
            backups = list(Path(folder).glob("*.pre-v5-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(stat.S_IMODE(backups[0].stat().st_mode), 0o600)
            with closing(sqlite3.connect(backups[0])) as backup:
                self.assertEqual(backup.execute("PRAGMA user_version").fetchone()[0], 4)
                self.assertFalse(NEW_GUILD_COLUMNS & set(columns(backup, "guild_settings")))
                self.assertEqual(columns(backup, "user_timezones"), {})
                self.assertEqual(columns(backup, "turn_log"), {})
            Store(path).close()
            self.assertEqual(len(list(Path(folder).glob("*.pre-v5-*.sqlite3"))), 1)

    def test_v4_file_missing_identity_columns_backs_up_once_as_identities(self):
        """FEAT-01 (D14): a v4 file lacking the node identity columns gets only the .pre-identities backup, ends at
        v5 with both the identity columns and the new schema, and the backup stays untouched."""
        with tempfile.TemporaryDirectory() as folder:
            path = self.v4_file(folder)
            with closing(sqlite3.connect(path)) as connection, connection:
                for name in ("author_label", "mentions_json"):
                    connection.execute(f"ALTER TABLE nodes DROP COLUMN {name}")
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 9)
                self.assert_new_schema(store.db)
                self.assertTrue({"author_label", "mentions_json"} <= set(columns(store.db, "nodes")))
            finally:
                store.close()
            self.assertEqual(len(list(Path(folder).glob("*.pre-identities-*.sqlite3"))), 1)
            self.assertEqual(list(Path(folder).glob("*.pre-v5-*")), [])
            backup_path = next(Path(folder).glob("*.pre-identities-*.sqlite3"))
            with closing(sqlite3.connect(backup_path)) as backup:
                self.assertFalse({"author_label", "mentions_json"} & set(columns(backup, "nodes")))

    def test_v5_file_opens_without_backup(self):
        """FEAT-01 (D14): a current file opens without making any backup."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cur.sqlite3"
            Store(path).close()
            Store(path).close()
            self.assertEqual(list(Path(folder).glob("*.pre-v*")), [])

    def test_v10_file_is_rejected(self):
        """FEAT-01 (D14): a version 10 file raises 'newer than this application'."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new.sqlite3"
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("PRAGMA user_version=10")
            with self.assertRaisesRegex(ValueError, "newer than this application"):
                Store(path)


class TimezoneTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()

    def tearDown(self):
        self.store.close()

    def test_defaults(self):
        """FEAT-01 (D14): nothing set means '' per level and ('UTC', 'default') resolved."""
        self.assertEqual(self.store.guild_timezone(1), "")
        self.assertEqual(self.store.user_timezone(1, 5), "")
        self.assertEqual(self.store.resolve_timezone(1, 5), ("UTC", "default"))

    def test_server_zone_set_and_clear(self):
        """FEAT-01 (D14): a server zone resolves as 'server'; '' clears back to the UTC default."""
        self.store.set_guild_timezone(1, "Asia/Seoul")
        self.assertEqual(self.store.guild_timezone(1), "Asia/Seoul")
        self.assertEqual(self.store.resolve_timezone(1, 5), ("Asia/Seoul", "server"))
        self.store.set_guild_timezone(1, "")
        self.assertEqual(self.store.guild_timezone(1), "")
        self.assertEqual(self.store.resolve_timezone(1, 5), ("UTC", "default"))

    def test_member_zone_overrides_server(self):
        """FEAT-01 (D14): member beats server beats default."""
        self.store.set_guild_timezone(1, "Asia/Seoul")
        self.store.set_user_timezone(1, 5, "America/New_York")
        self.assertEqual(self.store.user_timezone(1, 5), "America/New_York")
        self.assertEqual(self.store.resolve_timezone(1, 5), ("America/New_York", "member"))
        self.assertEqual(self.store.resolve_timezone(1, 6), ("Asia/Seoul", "server"))

    def test_clear_user_timezone_reports_removal(self):
        """FEAT-01 (D14): clear returns True only when a row was removed."""
        self.assertIs(self.store.clear_user_timezone(1, 5), False)
        self.store.set_user_timezone(1, 5, "UTC")
        self.assertIs(self.store.clear_user_timezone(1, 5), True)
        self.assertEqual(self.store.user_timezone(1, 5), "")
        self.assertIs(self.store.clear_user_timezone(1, 5), False)

    def test_set_user_timezone_overwrites(self):
        """FEAT-01 (D14): setting again replaces the previous zone."""
        self.store.set_user_timezone(1, 5, "Asia/Seoul")
        self.store.set_user_timezone(1, 5, "UTC")
        self.assertEqual(self.store.user_timezone(1, 5), "UTC")

    def test_set_guild_timezone_keeps_other_columns(self):
        """FEAT-01 (D14): the upsert leaves usage_footer and preset columns alone."""
        self.store.execute("INSERT INTO guild_settings(guild_id,preset_id,preset_revision,usage_footer) VALUES(1,7,3,0)")
        self.store.set_guild_timezone(1, "Asia/Seoul")
        row = self.store.one("SELECT * FROM guild_settings WHERE guild_id=1")
        self.assertEqual((row["preset_id"], row["preset_revision"], row["usage_footer"], row["timezone"]),
                         (7, 3, 0, "Asia/Seoul"))
        self.assertIs(self.store.usage_footer_enabled(1), False)

    def test_exact_iana_names_are_accepted(self):
        """FEAT-01 (D14): exact IANA names validate."""
        for name in ("Asia/Seoul", "America/New_York", "UTC"):
            with self.subTest(name=name):
                self.store.set_guild_timezone(1, name)
                self.store.set_user_timezone(1, 5, name)
                self.assertEqual(self.store.guild_timezone(1), name)
                self.assertEqual(self.store.user_timezone(1, 5), name)

    def test_invalid_names_raise_value_error_and_write_nothing(self):
        """FEAT-01 (D14): bad, path-like or blank names raise ValueError only; nothing is stored."""
        for name in ("Mars/Olympus", "../etc/passwd", "/UTC", " ", "Asia/Seoul\x00", "Asia/Seoul "):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.store.set_user_timezone(1, 5, name)
                with self.assertRaises(ValueError):
                    self.store.set_guild_timezone(1, name)
        self.assertEqual(self.store.user_timezone(1, 5), "")
        self.assertEqual(self.store.guild_timezone(1), "")

    def test_invalid_guild_name_message_names_value_and_suggests_iana(self):
        """FEAT-01 (D14): the error names the bad value and suggests an IANA name."""
        with self.assertRaises(ValueError) as caught:
            self.store.set_guild_timezone(1, "Mars/Olympus")
        self.assertIn("Mars/Olympus", str(caught.exception))
        self.assertIn("Asia/Seoul", str(caught.exception))

    def test_empty_user_timezone_is_invalid(self):
        """FEAT-01 (D14): '' is not a member zone; use clear."""
        with self.assertRaises(ValueError):
            self.store.set_user_timezone(1, 5, "")

    def test_stored_unloadable_names_fall_through(self):
        """FEAT-01 (D14): a stale stored name falls through to the next level instead of raising."""
        self.store.execute("INSERT INTO user_timezones(guild_id,user_id,timezone,updated_at) VALUES(1,5,'Mars/Olympus',0)")
        self.store.set_guild_timezone(1, "Asia/Seoul")
        self.assertEqual(self.store.resolve_timezone(1, 5), ("Asia/Seoul", "server"))
        self.store.execute("UPDATE guild_settings SET timezone='Mars/Olympus' WHERE guild_id=1")
        self.assertEqual(self.store.resolve_timezone(1, 5), ("UTC", "default"))

    def test_member_zone_is_guild_scoped(self):
        """FEAT-01 (D14): a member zone in guild A is invisible in guild B, which falls back to B's server zone."""
        self.store.set_user_timezone(1, 5, "America/New_York")
        self.store.set_guild_timezone(2, "Asia/Seoul")
        self.assertEqual(self.store.user_timezone(2, 5), "")
        self.assertEqual(self.store.resolve_timezone(2, 5), ("Asia/Seoul", "server"))

    def test_clear_in_one_guild_leaves_the_other(self):
        """FEAT-01 (D14): clearing in guild A keeps guild B's row."""
        self.store.set_user_timezone(1, 5, "UTC")
        self.store.set_user_timezone(2, 5, "Asia/Seoul")
        self.assertIs(self.store.clear_user_timezone(1, 5), True)
        self.assertEqual(self.store.user_timezone(2, 5), "Asia/Seoul")

    def test_memory_opt_out_keeps_timezone(self):
        """FEAT-01 (D14): timezone is not a personal memory; opting out of memory does not delete it."""
        self.store.set_consent(1, 5, True)
        self.store.set_user_timezone(1, 5, "Asia/Seoul")
        self.store.set_consent(1, 5, False)
        self.assertEqual(self.store.user_timezone(1, 5), "Asia/Seoul")
        self.assertEqual(self.store.resolve_timezone(1, 5), ("Asia/Seoul", "member"))


if __name__ == "__main__":
    unittest.main()
