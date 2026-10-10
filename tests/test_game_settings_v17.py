"""FEAT-21 A2: schema v17 (game_settings, game_seats stake >= 0, table thread columns), the generic game settings API and the game registry."""
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from llmcord_core.admin_store import ConflictError, GameError
from llmcord_core.games import registry
from llmcord_core.store import Store

G, G2, CH = 1, 2, 100
RULES = {"insurance": True, "stand_on": 16}


class V16UpgradeTests(unittest.TestCase):
    def v16_file(self, folder):
        path = Path(folder) / "v16.sqlite3"
        store = Store(path)
        store.close()
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("DROP TABLE game_settings")
            for column in ("thread_id", "board_message_id", "turn_message_id"):
                db.execute(f"ALTER TABLE game_tables DROP COLUMN {column}")
            db.execute("DROP TABLE game_seats")
            db.execute("""CREATE TABLE game_seats (
 guild_id INTEGER NOT NULL, round_id INTEGER NOT NULL REFERENCES game_rounds(id), seat_index INTEGER NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('member','character')), ref_id INTEGER NOT NULL,
 stake INTEGER NOT NULL CHECK(stake>0), outcome TEXT, payout INTEGER, brought_by INTEGER, insurance INTEGER NOT NULL DEFAULT 0,
 PRIMARY KEY(round_id,seat_index), UNIQUE(round_id,kind,ref_id)
)""")
            db.execute("INSERT INTO guild_settings(guild_id,blackjack_enabled,blackjack_rules) VALUES(1,0,'{\"insurance\": true}')")
            db.execute("INSERT INTO guild_settings(guild_id,blackjack_enabled) VALUES(2,1)")
            db.execute("INSERT INTO game_tables(guild_id,channel_id,game,status,opened_by,created_at,updated_at) VALUES(1,100,'blackjack','open',1,0,0)")
            db.execute("INSERT INTO game_rounds(guild_id,table_id,number,seed,seed_hash,status,created_at) VALUES(1,1,1,'s','h','joining',0)")
            db.execute("INSERT INTO game_seats(guild_id,round_id,seat_index,kind,ref_id,stake) VALUES(1,1,0,'member',7,25)")
            db.execute("UPDATE game_seats SET brought_by=9,insurance=3,outcome='win',payout=50 WHERE round_id=1")
            db.execute("PRAGMA user_version=16")
        return path

    def test_v16_file_upgrades_once_with_backup_and_keeps_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.v16_file(folder)
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 17)
                self.assertEqual(len(list(Path(folder).glob("v16.sqlite3.pre-v17-*.sqlite3"))), 1)
                self.assertFalse(store.blackjack_enabled(1))
                self.assertTrue(store.blackjack_enabled(2))
                self.assertEqual(store.blackjack_rules(1)["insurance"], True)
                self.assertEqual(store.blackjack_rules(2)["insurance"], False)
                seat = store.one("SELECT * FROM game_seats WHERE round_id=1")
                self.assertEqual(tuple(seat), (1, 1, 0, "member", 7, 25, "win", 50, 9, 3))
                backup = next(Path(folder).glob("v16.sqlite3.pre-v17-*.sqlite3"))
                with closing(sqlite3.connect(backup)) as old:
                    self.assertIn("CHECK(stake>0)", old.execute("SELECT sql FROM sqlite_master WHERE name='game_seats'").fetchone()[0])
                store.db.execute("INSERT INTO game_seats(guild_id,round_id,seat_index,kind,ref_id,stake) VALUES(1,1,1,'member',8,0)")
                cols = {r[1] for r in store.db.execute("PRAGMA table_info(game_tables)")}
                self.assertLessEqual({"thread_id", "board_message_id", "turn_message_id"}, cols)
                store.db.commit()
            finally:
                store.close()
            Store(path).close()
            self.assertEqual(len(list(Path(folder).glob("*.pre-v17-*.sqlite3"))), 1)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM game_seats").fetchone()[0], 2)
                self.assertEqual(db.execute("SELECT COUNT(*) FROM game_settings").fetchone()[0], 2)


class GameSettingsTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)
        self.store.set_game_channel(G, CH, True)

    def test_defaults_and_unknown_game(self):
        self.assertTrue(self.store.game_enabled(G, "blackjack"))
        self.assertEqual(self.store.game_rules(G, "blackjack"), self.store.blackjack_rules(G))
        with self.assertRaises(GameError):
            self.store.game_enabled(G, "nope")
        with self.assertRaises(GameError):
            self.store.set_game_rules(G, "nope", {})

    def test_enabled_conflict_and_guild_isolation(self):
        s = self.store
        s.set_game_enabled(G, "blackjack", False, True)
        with self.assertRaisesRegex(ConflictError, "The blackjack setting was changed elsewhere. Reload the page and try again."):
            s.set_game_enabled(G, "blackjack", True, True)
        self.assertFalse(s.game_enabled(G, "blackjack"))
        self.assertTrue(s.game_enabled(G2, "blackjack"))
        with self.assertRaises(ValueError):
            s.set_game_enabled(G, "blackjack", 1)

    def test_rules_conflict_and_guild_isolation(self):
        s = self.store
        classic = s.game_rules(G, "blackjack")
        out = s.set_game_rules(G, "blackjack", RULES, classic)
        with self.assertRaisesRegex(ConflictError, "The blackjack rules were changed elsewhere. Reload the page and try again."):
            s.set_game_rules(G, "blackjack", {}, classic)
        self.assertEqual(s.game_rules(G, "blackjack"), out)
        self.assertEqual(s.game_rules(G2, "blackjack"), classic)
        self.assertTrue(s.game_enabled(G, "blackjack"))
        self.assertEqual(s.one("SELECT COUNT(*) FROM game_settings WHERE guild_id=?", (G2,))[0], 0)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)
        self.store.set_game_channel(G, CH, True)

    def test_registry_entries(self):
        self.assertEqual(registry.keys(), ("blackjack", "doubt"))
        game = registry.get("blackjack")
        self.assertEqual((game.name, game.ledger_label, game.max_seats, game.default_enabled), ("Blackjack", "Blackjack", 7, True))
        with self.assertRaises(KeyError):
            registry.get("nope")

    def test_open_table_rejects_unknown_game(self):
        with self.assertRaisesRegex(GameError, "That game is not available."):
            self.store.open_table(G, CH, "nope", 1)
        self.assertIsNone(self.store.open_table_for(G, CH))

    def test_blackjack_ledger_reasons_are_unchanged(self):
        s = self.store
        s.change_balance(G, 5, 100, "start", 1)
        snap = s.open_table(G, CH, "blackjack", 1)
        s.join_round(G, snap["round_id"], 5, 10)
        s.cancel_round(G, snap["round_id"], "test")
        label = f"table {snap['table_id']}, round 1"
        reasons = [r["reason"] for r in s.ledger(G, 5)][::-1]
        self.assertEqual(reasons[1:], [f"Blackjack bet ({label})", f"Blackjack refund (test), {label}"])

    def test_a_table_of_an_unknown_game_raises_game_error_not_key_error(self):
        s = self.store
        snap = s.open_table(G, CH, "blackjack", 1)
        with s.write_admin():
            s.db.execute("UPDATE game_tables SET game='nope' WHERE id=?", (snap["table_id"],))
        for call in (lambda: s.round_snapshot(G, snap["round_id"]), lambda: s.cancel_round(G, snap["round_id"], "x"),
                     lambda: s.join_round(G, snap["round_id"], 5, 10)):
            with self.assertRaisesRegex(GameError, "That game is not available."):
                call()

    def test_zero_stake_is_refused_by_the_join(self):
        s = self.store
        s.change_balance(G, 5, 100, "start", 1)
        snap = s.open_table(G, CH, "blackjack", 1)
        with self.assertRaises(GameError):
            s.join_round(G, snap["round_id"], 5, 0)


if __name__ == "__main__":
    unittest.main()
