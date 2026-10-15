"""FEAT-49b1: schema v23 (member_settings.spoken_veto) and the store API for the veto switches (store only; no Discord).

Chosen store API (mirrors the neighbouring spoken switches):
    spoken_veto_enabled(guild_id) -> bool                              guild switch, game_settings pseudo row 'spoken_veto', default off
    set_spoken_veto_enabled(guild_id, on, expected=None) -> on         write_admin + ConflictError when `expected` is stale; non-bool -> UserError
    spoken_veto_member_on(guild_id, user_id) -> bool                   member switch, default on
    set_spoken_veto_member_on(guild_id, user_id, on) -> on             non-bool -> UserError
"""
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from llmcord_core.admin_store import ConflictError
from llmcord_core.i18n import UserError
from llmcord_core.store import Store

G, H = 1, 2
ALICE, BOB = 5, 6


def v22_file(folder):
    """A current store with the v23 column removed again and user_version set back to 22; member rows exist for both switches."""
    path = Path(folder) / "v22.sqlite3"
    store = Store(path)
    store.set_favorites_mode(G, ALICE, "step_in")
    store.set_spoken_member_on(G, BOB, False)
    store.set_spoken_moves_enabled(G, True)
    store.close()
    with closing(sqlite3.connect(path)) as db, db:
        if "spoken_veto" in {r[1] for r in db.execute("PRAGMA table_info(member_settings)")}:
            db.execute("ALTER TABLE member_settings DROP COLUMN spoken_veto")
        db.execute("PRAGMA user_version=22")
    return path


def member_columns(path):
    with closing(sqlite3.connect(path)) as db:
        return {r[1]: r for r in db.execute("PRAGMA table_info(member_settings)")}


def user_version(path):
    with closing(sqlite3.connect(path)) as db:
        return db.execute("PRAGMA user_version").fetchone()[0]


class UpgradeTests(unittest.TestCase):
    def test_v22_file_upgrades_with_a_backup_and_keeps_rows(self):
        """FEAT-49b1: a v22 file gets a .pre-v23 backup and the spoken_veto column; existing member rows keep their values and read veto-on."""
        with tempfile.TemporaryDirectory() as folder:
            path = v22_file(folder)
            self.assertNotIn("spoken_veto", member_columns(path))
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 26)
                self.assertEqual(store.favorites_mode(G, ALICE), "step_in")
                self.assertIs(store.spoken_member_on(G, BOB), False)
                self.assertIs(store.spoken_moves_enabled(G), True)
                self.assertIs(store.spoken_veto_member_on(G, ALICE), True)
                self.assertIs(store.spoken_veto_member_on(G, BOB), True)
                self.assertIs(store.spoken_veto_enabled(G), False)
                with self.assertRaises(sqlite3.IntegrityError):
                    store.db.execute("UPDATE member_settings SET spoken_veto=2")
                store.db.rollback()
            finally:
                store.close()
            column = member_columns(path)["spoken_veto"]
            self.assertEqual((column[2], column[3], column[4]), ("INTEGER", 1, "1"))
            backups = list(Path(folder).glob("v22.sqlite3.pre-v23-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(user_version(backups[0]), 22)
            self.assertNotIn("spoken_veto", member_columns(backups[0]))

    def test_reopening_is_idempotent(self):
        """FEAT-49b1: opening an upgraded file again changes nothing and makes no second backup."""
        with tempfile.TemporaryDirectory() as folder:
            path = v22_file(folder)
            Store(path).close()
            store = Store(path)
            try:
                store.set_spoken_veto_member_on(G, ALICE, False)
            finally:
                store.close()
            Store(path).close()
            self.assertEqual(user_version(path), 26)
            self.assertEqual(len(list(Path(folder).glob("*.pre-v23-*.sqlite3"))), 1)
            with closing(Store(path)) as store:
                self.assertIs(store.spoken_veto_member_on(G, ALICE), False)

    def test_fresh_database_is_v23_with_the_same_column_as_an_upgraded_one(self):
        """FEAT-49b1: a new file is v23 and its spoken_veto column matches the upgraded file's."""
        with tempfile.TemporaryDirectory() as folder:
            fresh = Path(folder) / "fresh.sqlite3"
            Store(fresh).close()
            upgraded = v22_file(folder)
            Store(upgraded).close()
            self.assertEqual(user_version(fresh), 26)
            self.assertEqual(member_columns(fresh)["spoken_veto"][1:], member_columns(upgraded)["spoken_veto"][1:])
            self.assertEqual(list(Path(folder).glob("fresh.sqlite3.pre-*")), [])


class VetoCase(unittest.TestCase):
    def setUp(self):
        self.s = Store()
        self.addCleanup(self.s.close)


class GuildSwitchTests(VetoCase):
    def test_default_off_and_round_trip(self):
        """FEAT-49b1: the guild veto switch is off until an admin turns it on."""
        self.assertIs(self.s.spoken_veto_enabled(G), False)
        self.assertIs(self.s.set_spoken_veto_enabled(G, True), True)
        self.assertIs(self.s.spoken_veto_enabled(G), True)
        self.assertIs(self.s.set_spoken_veto_enabled(G, False), False)
        self.assertIs(self.s.spoken_veto_enabled(G), False)

    def test_guild_isolation(self):
        """FEAT-49b1: guild G's switch does not touch guild H."""
        self.s.set_spoken_veto_enabled(G, True)
        self.assertIs(self.s.spoken_veto_enabled(H), False)

    def test_conflict_and_bad_value(self):
        """FEAT-49b1: a stale expected value raises ConflictError and writes nothing; a non-bool is refused."""
        with self.assertRaises(ConflictError):
            self.s.set_spoken_veto_enabled(G, True, expected=True)
        self.assertIs(self.s.spoken_veto_enabled(G), False)
        self.assertIs(self.s.set_spoken_veto_enabled(G, True, expected=False), True)
        with self.assertRaises(ConflictError):
            self.s.set_spoken_veto_enabled(G, False, expected=False)
        self.assertIs(self.s.spoken_veto_enabled(G), True)
        with self.assertRaises(UserError):
            self.s.set_spoken_veto_enabled(G, 1)

    def test_independent_of_the_other_spoken_switches(self):
        """FEAT-49b1: the veto switch is its own pseudo row; moves, doubt and keep-phrases are unaffected both ways."""
        self.s.set_spoken_veto_enabled(G, True)
        self.assertIs(self.s.spoken_moves_enabled(G), False)
        self.assertIs(self.s.spoken_doubt_enabled(G), False)
        self.assertIs(self.s.spoken_keep_setting(G), False)
        self.s.set_spoken_moves_enabled(G, True)
        self.s.set_spoken_doubt_enabled(G, True)
        self.s.set_spoken_moves_enabled(G, False)
        self.assertIs(self.s.spoken_veto_enabled(G), True)
        self.s.set_spoken_veto_enabled(G, False)
        self.assertIs(self.s.spoken_doubt_enabled(G), True)


class MemberSwitchTests(VetoCase):
    def test_default_on_and_round_trip(self):
        """FEAT-49b1: a member with no row is covered; an explicit opt-out stores off; opting back in stores on."""
        self.assertIs(self.s.spoken_veto_member_on(G, ALICE), True)
        self.assertIs(self.s.set_spoken_veto_member_on(G, ALICE, False), False)
        self.assertIs(self.s.spoken_veto_member_on(G, ALICE), False)
        self.assertIs(self.s.set_spoken_veto_member_on(G, ALICE, True), True)
        self.assertIs(self.s.spoken_veto_member_on(G, ALICE), True)

    def test_member_and_guild_isolation(self):
        """FEAT-49b1: an opt-out in guild G affects only that member in that guild."""
        self.s.set_spoken_veto_member_on(G, ALICE, False)
        self.assertIs(self.s.spoken_veto_member_on(H, ALICE), True)
        self.assertIs(self.s.spoken_veto_member_on(G, BOB), True)

    def test_guild_switch_does_not_change_member_choice(self):
        """FEAT-49b1: turning the guild switch on or off leaves a member's opt-out as it was."""
        self.s.set_spoken_veto_member_on(G, ALICE, False)
        self.s.set_spoken_veto_enabled(G, True)
        self.s.set_spoken_veto_enabled(G, False)
        self.assertIs(self.s.spoken_veto_member_on(G, ALICE), False)

    def test_bad_value_refused(self):
        """FEAT-49b1: a non-bool is a UserError and stores nothing."""
        with self.assertRaises(UserError):
            self.s.set_spoken_veto_member_on(G, ALICE, 0)
        self.assertIs(self.s.spoken_veto_member_on(G, ALICE), True)

    def test_veto_and_moves_member_switches_are_independent(self):
        """FEAT-49b1: opting out of the veto leaves spoken_moves on, and opting out of spoken moves leaves the veto on, in both orders."""
        self.s.set_spoken_veto_member_on(G, ALICE, False)
        self.assertIs(self.s.spoken_member_on(G, ALICE), True)
        self.s.set_spoken_member_on(G, ALICE, False)
        self.assertIs(self.s.spoken_veto_member_on(G, ALICE), False)
        self.s.set_spoken_veto_member_on(G, ALICE, True)
        self.assertIs(self.s.spoken_member_on(G, ALICE), False)
        self.s.set_spoken_member_on(G, ALICE, True)
        self.s.set_spoken_member_on(G, BOB, False)
        self.assertIs(self.s.spoken_veto_member_on(G, BOB), True)

    def test_shares_the_row_with_favorites_mode(self):
        """FEAT-49b1: the veto switch lives in the same member_settings row as favorites_mode and neither write disturbs the other."""
        self.s.set_favorites_mode(G, ALICE, "step_in")
        self.s.set_spoken_veto_member_on(G, ALICE, False)
        self.assertEqual(self.s.favorites_mode(G, ALICE), "step_in")
        self.s.set_favorites_mode(G, ALICE, "lean")
        self.assertIs(self.s.spoken_veto_member_on(G, ALICE), False)
        self.assertEqual(self.s.one("SELECT COUNT(*) FROM member_settings WHERE guild_id=? AND user_id=?", (G, ALICE))[0], 1)


if __name__ == "__main__":
    unittest.main()
