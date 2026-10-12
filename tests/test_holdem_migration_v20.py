"""FEAT-33 part B: schema v20 (game_seats.stack, game_rounds.button, game_cashouts) upgrades a v19 database in place.

The v19 file is a current store with the v20 additions removed again and user_version set back to 19 (the removal is
conditional, so the helper builds a true v19 file whether or not this build already creates v20).
"""
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from llmcord_core.store import Store

G, CH_BJ, CH_DOUBT = 1, 100, 101
MONEY = ("currency_ledger", "currency_balances", "character_balances")


def v19_file(folder):
    path = Path(folder) / "v19.sqlite3"
    store = Store(path)
    store.set_daily_settings(G, 0, 0, 7)
    store.set_game_channels(G, [CH_BJ, CH_DOUBT])
    for user in (5, 6, 7):
        store.change_balance(G, user, 500, "start", 1)
    bj = store.open_table(G, CH_BJ, "blackjack", 5, seed="bj-seed")
    store.join_round(G, bj["round_id"], 5, 20)
    dealt = store.deal(G, bj["round_id"])
    if dealt["status"] == "playing":
        store.play(G, bj["round_id"], dealt["view"]["turn"], "stand", "member", 0)
    store.set_game_rules(G, "doubt", {"ante": 10})
    doubt = store.open_table(G, CH_DOUBT, "doubt", 5, seed="doubt-seed")
    for user in (5, 6, 7):
        store.join_round(G, doubt["round_id"], user, 1)
    store.deal(G, doubt["round_id"])
    store.close()
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("DROP TABLE IF EXISTS game_cashouts")
        for table, column in (("game_seats", "stack"), ("game_rounds", "button")):
            if column in {row[1] for row in db.execute(f"PRAGMA table_info({table})")}:
                db.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
        db.execute("PRAGMA user_version=19")
    return path


def rows(path, table, skip=()):
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        return [{k: v for k, v in dict(r).items() if k not in skip} for r in db.execute(f"SELECT * FROM {table} ORDER BY rowid")]


def columns(path, table):
    with closing(sqlite3.connect(path)) as db:
        return {r[1]: r for r in db.execute(f"PRAGMA table_info({table})")}


class V19UpgradeTests(unittest.TestCase):
    def test_upgrade_adds_columns_and_table_with_a_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = v19_file(folder)
            self.assertNotIn("stack", columns(path, "game_seats"))
            Store(path).close()
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 22)
            stack = columns(path, "game_seats")["stack"]
            self.assertEqual((stack[2], stack[3], stack[4]), ("INTEGER", 1, "0"))  # type, NOT NULL, default 0
            button = columns(path, "game_rounds")["button"]
            self.assertEqual((button[2], button[3]), ("INTEGER", 0))  # nullable
            self.assertEqual(list(columns(path, "game_cashouts")), ["id", "guild_id", "table_id", "kind", "ref_id", "bought_in", "cashed_out", "created_at"])
            with closing(sqlite3.connect(path)) as db:
                indexed = [[c[2] for c in db.execute(f"PRAGMA index_info({i[1]})")] for i in db.execute("PRAGMA index_list(game_cashouts)")]
            self.assertIn(["guild_id", "table_id"], indexed)
            backups = list(Path(folder).glob("v19.sqlite3.pre-v20-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            with closing(sqlite3.connect(backups[0])) as old:
                self.assertEqual(old.execute("PRAGMA user_version").fetchone()[0], 19)
                self.assertNotIn("stack", {r[1] for r in old.execute("PRAGMA table_info(game_seats)")})

    def test_existing_game_rows_survive_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            path = v19_file(folder)
            before = {t: rows(path, t) for t in ("game_tables", "game_moves", *MONEY)}
            before["game_rounds"] = rows(path, "game_rounds", skip=("button",))
            before["game_seats"] = rows(path, "game_seats", skip=("stack",))
            self.assertGreaterEqual(len(before["game_seats"]), 4)  # blackjack and doubt both have seats
            self.assertTrue(before["game_moves"] or before["game_rounds"])
            Store(path).close()
            for table in ("game_tables", "game_moves", *MONEY):
                self.assertEqual(rows(path, table), before[table], table)
            self.assertEqual(rows(path, "game_rounds", skip=("button",)), before["game_rounds"])
            self.assertEqual(rows(path, "game_seats", skip=("stack",)), before["game_seats"])
            self.assertEqual({r["stack"] for r in rows(path, "game_seats")}, {0})
            self.assertEqual({r["button"] for r in rows(path, "game_rounds")}, {None})
            self.assertEqual(rows(path, "game_cashouts"), [])

    def test_blackjack_and_doubt_tables_still_play_after_the_upgrade(self):
        with tempfile.TemporaryDirectory() as folder:
            path = v19_file(folder)
            store = Store(path)
            try:
                bj = store.latest_round(G, store.open_table_for(G, CH_BJ)["id"])
                self.assertEqual((bj["game"], [s["stake"] for s in bj["seats"]]), ("blackjack", [20]))
                doubt = store.latest_round(G, store.open_table_for(G, CH_DOUBT)["id"])
                self.assertEqual((doubt["game"], doubt["status"], [s["stake"] for s in doubt["seats"]]), ("doubt", "playing", [10, 10, 10]))
                store.cancel_round(G, doubt["round_id"], "test")
                self.assertEqual([store.balance(G, u) for u in (6, 7)], [500, 500])  # the antes came back
                nxt = store.next_round(G, doubt["table_id"])
                self.assertEqual(nxt["status"], "joining")
            finally:
                store.close()

    def test_migrating_twice_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            path = v19_file(folder)
            Store(path).close()
            first = {t: rows(path, t) for t in ("game_tables", "game_rounds", "game_seats", "game_moves", "game_cashouts", *MONEY)}
            Store(path).close()
            Store(path).close()
            self.assertEqual({t: rows(path, t) for t in first}, first)
            self.assertEqual(len(list(Path(folder).glob("*.pre-v20-*.sqlite3"))), 1)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 22)

    def test_a_newer_database_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new.sqlite3"
            Store(path).close()
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("PRAGMA user_version=23")
            with self.assertRaisesRegex(ValueError, "newer than this application"):
                Store(path)


class FreshDatabaseTests(unittest.TestCase):
    def test_fresh_database_is_v20_and_has_the_same_shape_as_an_upgraded_one(self):
        with tempfile.TemporaryDirectory() as folder:
            fresh = Path(folder) / "fresh.sqlite3"
            Store(fresh).close()
            upgraded = v19_file(folder)
            Store(upgraded).close()
            for path in (fresh, upgraded):
                with closing(sqlite3.connect(path)) as db:
                    self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 22)
            for table in ("game_seats", "game_rounds", "game_cashouts"):
                shape = lambda path: {n: c[2:5] for n, c in columns(path, table).items()}  # noqa: E731 (column order may differ: ALTER appends)
                self.assertEqual(shape(fresh), shape(upgraded), table)
            self.assertEqual(list(Path(folder).glob("fresh.sqlite3.pre-*")), [])  # a new file needs no backup

    def test_in_memory_store_is_v20_and_stack_cannot_go_negative(self):
        store = Store()
        try:
            self.assertEqual(store.one("PRAGMA user_version")[0], 22)
            store.db.execute("INSERT INTO game_tables(guild_id,channel_id,game,status,opened_by,created_at,updated_at) VALUES(1,100,'holdem','open',1,0,0)")
            store.db.execute("INSERT INTO game_rounds(guild_id,table_id,number,seed,seed_hash,status,created_at) VALUES(1,1,1,'s','h','joining',0)")
            store.db.execute("INSERT INTO game_seats(guild_id,round_id,seat_index,kind,ref_id,stake) VALUES(1,1,0,'member',5,100)")
            self.assertEqual(store.one("SELECT stack FROM game_seats")[0], 0)
            with self.assertRaises(sqlite3.IntegrityError):
                store.db.execute("INSERT INTO game_seats(guild_id,round_id,seat_index,kind,ref_id,stake,stack) VALUES(1,1,1,'member',6,100,-1)")
            store.db.rollback()
        finally:
            store.close()


class OldSchemaStackTests(unittest.TestCase):
    """Opus review (FEAT-33): the stack column added to an old file defaults to 0 on migrated rows and refuses a negative value."""

    def refuses_negative_stack(self, path):
        store = Store(path)
        try:
            store.db.execute("INSERT INTO game_tables(guild_id,channel_id,game,status,opened_by,created_at,updated_at) VALUES(1,555,'holdem','open',1,0,0)")
            table = store.one("SELECT id FROM game_tables WHERE channel_id=555")[0]
            rnd = store.db.execute("INSERT INTO game_rounds(guild_id,table_id,number,seed,seed_hash,status,created_at) VALUES(1,?,1,'s','h','joining',0)", (table,)).lastrowid
            store.db.execute("INSERT INTO game_seats(guild_id,round_id,seat_index,kind,ref_id,stake,stack) VALUES(1,?,0,'member',6,100,0)", (rnd,))
            with self.assertRaises(sqlite3.IntegrityError):
                store.db.execute("INSERT INTO game_seats(guild_id,round_id,seat_index,kind,ref_id,stake,stack) VALUES(1,?,1,'member',7,100,-1)", (rnd,))
            store.db.rollback()
        finally:
            store.close()

    def test_v19_file_upgrades_with_default_zero_stacks_and_refuses_a_negative_stack(self):
        with tempfile.TemporaryDirectory() as folder:
            path = v19_file(folder)
            Store(path).close()
            self.assertEqual({r["stack"] for r in rows(path, "game_seats")}, {0})
            self.refuses_negative_stack(path)

    def test_v16_file_with_a_stake_check_upgrades_to_a_stack_column_that_refuses_negatives(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v16.sqlite3"
            Store(path).close()
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("DROP TABLE game_cashouts")
                db.execute("DROP TABLE game_seats")
                db.execute("""CREATE TABLE game_seats (
 guild_id INTEGER NOT NULL, round_id INTEGER NOT NULL REFERENCES game_rounds(id), seat_index INTEGER NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('member','character')), ref_id INTEGER NOT NULL,
 stake INTEGER NOT NULL CHECK(stake>0), outcome TEXT, payout INTEGER, brought_by INTEGER, insurance INTEGER NOT NULL DEFAULT 0,
 PRIMARY KEY(round_id,seat_index), UNIQUE(round_id,kind,ref_id)
)""")
                db.execute("ALTER TABLE game_rounds DROP COLUMN button")
                db.execute("INSERT INTO game_tables(guild_id,channel_id,game,status,opened_by,created_at,updated_at) VALUES(1,100,'blackjack','open',1,0,0)")
                db.execute("INSERT INTO game_rounds(guild_id,table_id,number,seed,seed_hash,status,created_at) VALUES(1,1,1,'s','h','joining',0)")
                db.execute("INSERT INTO game_seats(guild_id,round_id,seat_index,kind,ref_id,stake) VALUES(1,1,0,'member',7,25)")
                db.execute("PRAGMA user_version=16")
            Store(path).close()
            self.assertIn("stack", columns(path, "game_seats"))
            self.assertEqual([(r["ref_id"], r["stake"], r["stack"]) for r in rows(path, "game_seats")], [(7, 25, 0)])
            self.refuses_negative_stack(path)


if __name__ == "__main__":
    unittest.main()
