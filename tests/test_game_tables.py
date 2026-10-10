"""FEAT-19 part B: game tables, rounds, seats and money (schema v14, store API; no Discord)."""
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from llmcord_core.admin_store import ConflictError, GameError
from llmcord_core.games import blackjack as bj
from llmcord_core.games import seed_hash
from llmcord_core.store import Store

G, CH = 1, 100
ALICE, BOB = 5, 6


def card(rank):
    return bj.RANKS.index(rank)


class TableCase(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.shoe = None
        patcher = mock.patch.object(bj, "new_shoe", side_effect=self.make_shoe)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.store.close)
        self.store.set_game_channel(G, CH, True)

    def make_shoe(self, seed):
        if self.shoe is not None:
            return self.shoe
        from llmcord_core.games.base import shuffled
        return tuple(shuffled(list(range(52)) * bj.DECKS, seed))

    def rig(self, cards, tail="2"):
        """Deal order: one card per seat, dealer up, one more per seat, dealer hole, then draws (all `tail`)."""
        self.shoe = tuple(card(r) for r in cards) + (card(tail),) * 100

    def fund(self, user, amount, guild=G):
        self.store.change_balance(guild, user, amount, "start", 1)

    def table(self, **kw):
        return self.store.open_table(G, CH, "blackjack", 1, **kw)

    def game_ledger(self, guild=G):
        return [dict(r) for r in self.store.db.execute(
            "SELECT * FROM currency_ledger WHERE guild_id=? AND source='game' ORDER BY id", (guild,))]

    def moves(self, guild=G):
        return self.store.db.execute("SELECT seat_index,move,actor FROM game_moves WHERE guild_id=? ORDER BY id", (guild,)).fetchall()

    def started(self, cards, stakes=((ALICE, 10),), tail="2", funds=100):
        """Open a table, seat the members, rig the shoe, deal. Returns the dealt snapshot."""
        self.rig(cards, tail)
        snap = self.table(seed="s-" + "".join(cards))
        for user, stake in stakes:
            if self.store.balance(G, user) < funds:
                self.fund(user, funds - self.store.balance(G, user))
            self.store.join_round(G, snap["round_id"], user, stake)
        return self.store.deal(G, snap["round_id"])


class MigrationTests(unittest.TestCase):
    def v13_file(self, folder):
        path = Path(folder) / "old.sqlite3"
        store = Store(path)
        store.change_balance(1, 5, 40, "start", 9)
        store.set_currency_name(1, "gold")
        store.close()
        with closing(sqlite3.connect(path)) as db, db:
            for table in ("game_channels", "game_tables", "game_rounds", "game_seats", "game_moves"):
                db.execute(f"DROP TABLE {table}")
            for column in ("game_min_bet", "game_max_bet"):
                db.execute(f"ALTER TABLE guild_settings DROP COLUMN {column}")
            db.execute("PRAGMA user_version=13")
        return path

    def test_v13_file_upgrades_with_a_backup_and_keeps_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.v13_file(folder)
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 15)
                self.assertEqual((store.currency_name(1), store.balance(1, 5)), ("gold", 40))
                self.assertEqual(store.game_settings(1), {"min_bet": 1, "max_bet": 1000})
                self.assertEqual(store.game_channels(1), set())
                store.set_game_channel(1, 7, True)
                self.assertEqual(store.game_channels(1), {7})
            finally:
                store.close()
            backups = list(Path(folder).glob("old.sqlite3.pre-v14-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            with closing(sqlite3.connect(backups[0])) as db:
                self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 13)
                self.assertNotIn("game_tables", {r[0] for r in db.execute("SELECT name FROM sqlite_master")})
            Store(path).close()
            self.assertEqual(len(list(Path(folder).glob("*.pre-*"))), 1)

    def test_fresh_store_has_the_game_tables_and_move_log_is_append_only(self):
        store = Store()
        try:
            names = {r[0] for r in store.db.execute("SELECT name FROM sqlite_master")}
            self.assertTrue({"game_channels", "game_tables", "game_rounds", "game_seats", "game_moves"} <= names)
            store.db.execute("INSERT INTO game_moves(guild_id,round_id,seat_index,move,actor,created_at) VALUES(1,1,0,'stand','member',0)")
            with self.assertRaises(sqlite3.DatabaseError):
                store.db.execute("UPDATE game_moves SET move='hit'")
            with self.assertRaises(sqlite3.DatabaseError):
                store.db.execute("DELETE FROM game_moves")
            store.db.rollback()
        finally:
            store.close()


class ChannelAndSettingsTests(TableCase):
    def test_channels_on_off_and_per_guild(self):
        self.assertEqual(self.store.game_channels(G), {CH})
        self.store.set_game_channel(G, 101, True)
        self.store.set_game_channel(G, 101, True)
        self.assertEqual(self.store.game_channels(G), {CH, 101})
        self.assertEqual(self.store.game_channels(2), set())
        self.store.set_game_channel(G, 101, False)
        self.store.set_game_channel(G, 101, False)
        self.assertEqual(self.store.game_channels(G), {CH})

    def test_open_table_needs_a_game_channel(self):
        with self.assertRaisesRegex(GameError, "not turned on"):
            self.store.open_table(G, 555, "blackjack", 1)
        with self.assertRaisesRegex(GameError, "not turned on"):
            self.store.open_table(2, CH, "blackjack", 1)
        with self.assertRaises(GameError):
            self.store.open_table(G, CH, "poker", 1)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM game_tables").fetchone()[0], 0)

    def test_one_open_table_per_channel(self):
        first = self.table()
        with self.assertRaisesRegex(GameError, "already"):
            self.table()
        self.assertEqual(self.store.open_table_for(G, CH)["id"], first["table_id"])
        self.assertIsNone(self.store.open_table_for(2, CH))
        self.store.close_table(G, first["table_id"])
        self.assertIsNone(self.store.open_table_for(G, CH))
        self.assertNotEqual(self.table()["table_id"], first["table_id"])

    def test_partial_unique_index_enforces_it_in_the_schema(self):
        self.table()
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.db.execute("INSERT INTO game_tables(guild_id,channel_id,game,status,opened_by,created_at,updated_at) VALUES(?,?,?,?,?,0,0)", (G, CH, "blackjack", "open", 1))
        self.store.db.rollback()

    def test_settings_defaults_validation_and_conflict(self):
        self.assertEqual(self.store.game_settings(G), {"min_bet": 1, "max_bet": 1000})
        self.assertEqual(self.store.set_game_settings(G, 5, 200), {"min_bet": 5, "max_bet": 200})
        self.assertEqual(self.store.game_settings(2), {"min_bet": 1, "max_bet": 1000})
        for low, high in ((0, 10), (11, 10), (1, 100_001), (-1, 5), (1.5, 5), (True, 5), ("1", 5), (1, None)):
            with self.subTest(low=low, high=high), self.assertRaises(GameError):
                self.store.set_game_settings(G, low, high)
        self.assertEqual(self.store.set_game_settings(G, 1, 100_000), {"min_bet": 1, "max_bet": 100_000})
        expected = self.store.game_settings(G)
        self.store.set_game_settings(G, 2, 50)
        with self.assertRaises(ConflictError):
            self.store.set_game_settings(G, 3, 60, expected=expected)
        self.assertEqual(self.store.game_settings(G), {"min_bet": 2, "max_bet": 50})
        self.store.set_game_settings(G, 3, 60, expected={"min_bet": 2, "max_bet": 50})

    def test_settings_survive_other_settings_writes(self):
        self.store.set_game_settings(G, 5, 200)
        self.store.set_usage_footer(G, False)
        self.store.set_currency_name(G, "gold")
        self.assertEqual(self.store.game_settings(G), {"min_bet": 5, "max_bet": 200})


class SetGameChannelsTests(TableCase):
    def test_adds_and_removes_in_one_call_and_returns_closed_tables(self):
        self.assertEqual(self.store.set_game_channels(G, {CH, 101, 102}), {"closed_tables": []})
        self.assertEqual(self.store.game_channels(G), {CH, 101, 102})
        self.assertEqual(self.store.set_game_channels(G, [101]), {"closed_tables": []})
        self.assertEqual(self.store.game_channels(G), {101})

    def test_removing_a_channel_closes_its_table_and_refunds(self):
        snap = self.started(["10", "9", "10", "8"])
        self.assertEqual(self.store.balance(G, ALICE), 90)
        self.store.set_game_channel(G, 101, True)
        out = self.store.set_game_channels(G, {101}, expected={CH, 101})
        self.assertEqual(out, {"closed_tables": [snap["table_id"]]})
        self.assertEqual(self.store.balance(G, ALICE), 100)
        self.assertIsNone(self.store.open_table_for(G, CH))
        self.assertEqual(self.store.game_channels(G), {101})
        self.assertEqual(sum(r["amount"] for r in self.game_ledger()), 0)

    def test_stale_expected_set_is_refused_and_writes_nothing(self):
        self.table()
        self.store.set_game_channel(G, 101, True)
        with self.assertRaisesRegex(ConflictError, "game channels were changed elsewhere"):
            self.store.set_game_channels(G, {555}, expected={CH})
        self.assertEqual(self.store.game_channels(G), {CH, 101})
        self.assertIsNotNone(self.store.open_table_for(G, CH))

    def test_bad_ids_are_refused(self):
        for bad in ({"x"}, {True}, {1.5}, {0}, {-3}, {2 ** 63}, {None}):
            with self.subTest(bad=bad), self.assertRaises(GameError):
                self.store.set_game_channels(G, bad)
        self.assertEqual(self.store.game_channels(G), {CH})

    def test_other_guilds_are_untouched(self):
        self.store.set_game_channel(2, CH, True)
        other = self.store.open_table(2, CH, "blackjack", 1)
        self.store.set_game_channels(G, set())
        self.assertEqual(self.store.game_channels(2), {CH})
        self.assertEqual(self.store.open_table_for(2, CH)["id"], other["table_id"])
        self.store.set_game_channels(2, {7}, expected={CH})
        self.assertEqual(self.store.game_channels(G), set())


class JoinLeaveTests(TableCase):
    def test_open_creates_round_one_joining_with_a_seed_hash(self):
        snap = self.table(seed="abc")
        self.assertEqual((snap["number"], snap["status"], snap["seats"], snap["moves"]), (1, "joining", [], 0))
        self.assertEqual(snap["seed_hash"], seed_hash("abc"))
        self.assertIsNone(snap["seed"])
        self.assertIsNone(snap["view"])
        self.assertEqual((snap["guild_id"], snap["channel_id"], snap["game"]), (G, CH, "blackjack"))
        self.assertEqual(self.store.open_table_for(G, CH)["id"], snap["table_id"])

    def test_default_seed_is_random_and_hash_matches(self):
        self.store.close_table(G, self.table()["table_id"])
        a = self.table()
        row = self.store.db.execute("SELECT seed,seed_hash FROM game_rounds WHERE id=?", (a["round_id"],)).fetchone()
        self.assertEqual(len(row["seed"]), 32)
        self.assertEqual(row["seed_hash"], seed_hash(row["seed"]))

    def test_join_takes_the_stake_through_the_ledger(self):
        snap = self.table()
        self.fund(ALICE, 100)
        out = self.store.join_round(G, snap["round_id"], ALICE, 30)
        self.assertEqual(self.store.balance(G, ALICE), 70)
        self.assertEqual([(s["index"], s["kind"], s["ref_id"], s["stake"]) for s in out["seats"]], [(0, "member", ALICE, 30)])
        row = self.game_ledger()[0]
        self.assertEqual((row["amount"], row["balance_after"], row["source"], row["user_id"]), (-30, 70, "game", ALICE))
        self.assertIn(f"table {snap['table_id']}, round 1", row["reason"])
        self.assertTrue(row["reason"].startswith("Blackjack bet"))

    def test_insufficient_balance_records_nothing(self):
        snap = self.table()
        self.fund(ALICE, 10)
        with self.assertRaisesRegex(GameError, "10"):
            self.store.join_round(G, snap["round_id"], ALICE, 11)
        self.assertEqual(self.store.balance(G, ALICE), 10)
        self.assertEqual(self.game_ledger(), [])
        self.assertEqual(self.store.round_snapshot(G, snap["round_id"])["seats"], [])
        with self.assertRaises(GameError):
            self.store.join_round(G, snap["round_id"], BOB, 5)

    def test_stake_validation_and_limits(self):
        snap = self.table()
        self.fund(ALICE, 1000)
        self.store.set_game_settings(G, 5, 50)
        for stake in (4, 51, 0, -5, 10.5, True, "10", None):
            with self.subTest(stake=stake), self.assertRaises(GameError):
                self.store.join_round(G, snap["round_id"], ALICE, stake)
        self.assertEqual(self.store.balance(G, ALICE), 1000)
        self.store.join_round(G, snap["round_id"], ALICE, 5)
        self.store.leave_round(G, snap["round_id"], ALICE)
        self.store.join_round(G, snap["round_id"], ALICE, 50)

    def test_double_join_refused(self):
        snap = self.table()
        self.fund(ALICE, 100)
        self.store.join_round(G, snap["round_id"], ALICE, 10)
        with self.assertRaisesRegex(GameError, "already"):
            self.store.join_round(G, snap["round_id"], ALICE, 10)
        self.assertEqual(self.store.balance(G, ALICE), 90)
        self.assertEqual(len(self.game_ledger()), 1)

    def test_eighth_seat_refused(self):
        snap = self.table()
        for user in range(10, 17):
            self.fund(user, 10)
            self.store.join_round(G, snap["round_id"], user, 5)
        self.fund(99, 10)
        with self.assertRaisesRegex(GameError, "full"):
            self.store.join_round(G, snap["round_id"], 99, 5)
        self.assertEqual(self.store.balance(G, 99), 10)
        self.assertEqual(len(self.store.round_snapshot(G, snap["round_id"])["seats"]), bj.MAX_SEATS)

    def test_character_seats_are_refused_for_now(self):
        snap = self.table()
        self.fund(ALICE, 100)
        with self.assertRaises(GameError):
            self.store.join_round(G, snap["round_id"], ALICE, 10, kind="character")
        self.assertEqual(self.store.balance(G, ALICE), 100)

    def test_leave_refunds_and_renumbers(self):
        snap = self.table()
        for user in (ALICE, BOB, 7):
            self.fund(user, 100)
            self.store.join_round(G, snap["round_id"], user, 20)
        out = self.store.leave_round(G, snap["round_id"], BOB)
        self.assertEqual(self.store.balance(G, BOB), 100)
        self.assertEqual([(s["index"], s["ref_id"]) for s in out["seats"]], [(0, ALICE), (1, 7)])
        refund = self.game_ledger()[-1]
        self.assertEqual((refund["amount"], refund["user_id"]), (20, BOB))
        self.assertTrue(refund["reason"].startswith("Blackjack refund"))
        with self.assertRaises(GameError):
            self.store.leave_round(G, snap["round_id"], BOB)
        dealt = self.store.deal(G, snap["round_id"])
        self.assertIn(dealt["status"], ("playing", "settled"))

    def test_join_and_leave_only_while_joining(self):
        self.started(["10", "9", "7", "8"])
        rid = self.store.open_table_for(G, CH)["id"]
        round_id = self.store.latest_round(G, rid)["round_id"]
        self.fund(BOB, 50)
        with self.assertRaises(GameError):
            self.store.join_round(G, round_id, BOB, 10)
        with self.assertRaises(GameError):
            self.store.leave_round(G, round_id, ALICE)
        self.assertEqual(self.store.balance(G, ALICE), 90)


class DealAndPlayTests(TableCase):
    def test_deal_needs_a_seat_and_a_joining_round(self):
        snap = self.table()
        with self.assertRaises(GameError):
            self.store.deal(G, snap["round_id"])
        self.assertEqual(self.store.round_snapshot(G, snap["round_id"])["status"], "joining")
        self.fund(ALICE, 100)
        self.rig(["10", "9", "7", "8"])
        self.store.join_round(G, snap["round_id"], ALICE, 10)
        dealt = self.store.deal(G, snap["round_id"])
        self.assertEqual(dealt["status"], "playing")
        self.assertEqual(dealt["legal"], {0: ("hit", "stand", "double")})
        with self.assertRaises(GameError):
            self.store.deal(G, snap["round_id"])

    def test_play_stand_win_pays_double_stake(self):
        snap = self.started(["10", "9", "10", "8"])  # player 20, dealer 17
        self.assertEqual(snap["status"], "playing")
        out = self.store.play(G, snap["round_id"], 0, "stand", "member", 0)
        self.assertEqual(out["status"], "settled")
        self.assertEqual((out["seats"][0]["outcome"], out["seats"][0]["payout"]), ("win", 20))
        self.assertEqual(self.store.balance(G, ALICE), 110)
        payout = self.game_ledger()[-1]
        self.assertEqual(payout["amount"], 20)
        self.assertIn("win", payout["reason"])
        self.assertTrue(payout["reason"].startswith("Blackjack payout"))

    def test_push_returns_the_stake_and_lose_pays_nothing(self):
        snap = self.started(["10", "10", "8", "8"])
        out = self.store.play(G, snap["round_id"], 0, "stand", "member", 0)
        self.assertEqual((out["seats"][0]["outcome"], out["seats"][0]["payout"]), ("push", 10))
        self.assertEqual(self.store.balance(G, ALICE), 100)
        self.assertIn("push", self.game_ledger()[-1]["reason"])

    def test_lose_and_bust(self):
        snap = self.started(["10", "10", "7", "9"])  # 17 vs 19
        out = self.store.play(G, snap["round_id"], 0, "stand", "member", 0)
        self.assertEqual((out["seats"][0]["outcome"], out["seats"][0]["payout"]), ("lose", 0))
        self.assertEqual(self.store.balance(G, ALICE), 90)
        self.assertEqual(len(self.game_ledger()), 1)
        self.store.next_round(G, out["table_id"])
        snap = self.started_next([(ALICE, 10)], ["10", "9", "6", "8"], tail="K")
        out = self.store.play(G, snap["round_id"], 0, "hit", "member", 0)
        self.assertEqual((out["status"], out["seats"][0]["outcome"], out["seats"][0]["payout"]), ("settled", "bust", 0))
        self.assertEqual(self.store.balance(G, ALICE), 80)

    def started_next(self, stakes, cards, tail="2"):
        self.rig(cards, tail)
        table = self.store.open_table_for(G, CH)
        round_id = self.store.latest_round(G, table["id"])["round_id"]
        for user, stake in stakes:
            self.store.join_round(G, round_id, user, stake)
        return self.store.deal(G, round_id)

    def test_blackjack_pays_three_to_two_rounded_down_at_deal(self):
        snap = self.started(["A", "9", "K", "8"], stakes=((ALICE, 5),))
        self.assertEqual(snap["status"], "settled")
        self.assertEqual((snap["seats"][0]["outcome"], snap["seats"][0]["payout"]), ("blackjack", 12))
        self.assertEqual(self.store.balance(G, ALICE), 107)
        self.assertIn("blackjack", self.game_ledger()[-1]["reason"])
        self.assertEqual(snap["seed"] is not None, True)

    def test_dealer_natural_settles_at_deal(self):
        snap = self.started(["10", "9", "A", "5", "6", "K"], stakes=((ALICE, 10), (BOB, 10)))
        self.assertEqual(snap["status"], "settled")
        self.assertEqual([s["outcome"] for s in snap["seats"]], ["lose", "lose"])
        self.assertEqual(self.moves(), [])
        self.assertEqual((self.store.balance(G, ALICE), self.store.balance(G, BOB)), (90, 90))

    def test_two_seats_play_in_order(self):
        snap = self.started(["10", "10", "9", "8", "7", "9"], stakes=((ALICE, 10), (BOB, 10)))
        # alice 10+8=18, bob 10+7=17, dealer 9+9=18
        self.assertEqual(snap["view"]["turn"], 0)
        with self.assertRaisesRegex(GameError, "turn"):
            self.store.play(G, snap["round_id"], 1, "stand", "member", 0)
        self.store.play(G, snap["round_id"], 0, "stand", "member", 0)
        out = self.store.play(G, snap["round_id"], 1, "stand", "member", 1)
        self.assertEqual([s["outcome"] for s in out["seats"]], ["push", "lose"])
        self.assertEqual(len(self.moves()), 2)

    def test_illegal_and_wrong_turn_record_nothing(self):
        snap = self.started(["10", "10", "9", "8", "7", "9"], stakes=((ALICE, 10), (BOB, 10)))
        for seat, move in ((1, "stand"), (0, "split"), (0, ""), (5, "stand"), (-1, "stand")):
            with self.subTest(seat=seat, move=move), self.assertRaises(GameError):
                self.store.play(G, snap["round_id"], seat, move, "member", 0)
        with self.assertRaises(GameError):
            self.store.play(G, snap["round_id"], 0, "stand", "robot", 0)
        self.assertEqual(self.moves(), [])
        self.store.play(G, snap["round_id"], 0, "stand", "timeout", 0)
        self.assertEqual([tuple(m) for m in self.moves()], [(0, "stand", "timeout")])

    def test_stale_expected_moves_conflict_and_apply_nothing(self):
        snap = self.started(["10", "10", "5", "8", "5", "9"], stakes=((ALICE, 10),))
        self.store.play(G, snap["round_id"], 0, "hit", "member", 0)  # 15 + 5 = 20, still playing
        for stale in (0, 2, -1):
            with self.subTest(stale=stale), self.assertRaises(ConflictError):
                self.store.play(G, snap["round_id"], 0, "stand", "member", stale)
        self.assertEqual(len(self.moves()), 1)
        self.assertEqual(self.store.round_snapshot(G, snap["round_id"])["moves"], 1)

    def test_play_on_unfinished_states_is_refused(self):
        snap = self.table()
        with self.assertRaises(GameError):
            self.store.play(G, snap["round_id"], 0, "stand", "member", 0)

    def test_play_after_settle_is_refused(self):
        snap = self.started(["10", "9", "10", "8"])
        self.store.play(G, snap["round_id"], 0, "stand", "member", 0)
        with self.assertRaises(GameError):
            self.store.play(G, snap["round_id"], 0, "stand", "member", 1)
        self.assertEqual(self.store.balance(G, ALICE), 110)
        self.assertEqual(len(self.game_ledger()), 2)

    def test_double_takes_the_extra_stake_and_pays_on_the_doubled_stake(self):
        snap = self.started(["5", "9", "6", "8"], tail="10")  # player 11, dealer 17; double draws a 10 -> 21
        out = self.store.play(G, snap["round_id"], 0, "double", "member", 0)
        self.assertEqual(out["status"], "settled")
        seat = out["seats"][0]
        self.assertEqual((seat["stake"], seat["outcome"], seat["payout"]), (20, "win", 40))
        self.assertEqual(self.store.balance(G, ALICE), 120)
        amounts = [r["amount"] for r in self.game_ledger()]
        self.assertEqual(amounts, [-10, -10, 40])
        self.assertEqual(sum(amounts), seat["payout"] - seat["stake"])

    def test_double_refused_when_unaffordable_and_not_offered(self):
        snap = self.started(["5", "9", "6", "8"], tail="10", funds=10)  # all 10 coins staked
        self.assertEqual(self.store.balance(G, ALICE), 0)
        self.assertEqual(snap["legal"], {0: ("hit", "stand")})
        with self.assertRaises(GameError):
            self.store.play(G, snap["round_id"], 0, "double", "member", 0)
        self.assertEqual(self.moves(), [])
        self.assertEqual(self.store.balance(G, ALICE), 0)
        self.assertEqual(len(self.game_ledger()), 1)

    def test_double_offered_when_exactly_affordable(self):
        snap = self.started(["5", "9", "6", "8"], tail="10", funds=20)
        self.assertEqual(snap["legal"], {0: ("hit", "stand", "double")})

    def test_doubled_bust_loses_both_stakes(self):
        snap = self.started(["10", "9", "6", "8"], tail="K")
        out = self.store.play(G, snap["round_id"], 0, "double", "member", 0)
        self.assertEqual((out["seats"][0]["outcome"], out["seats"][0]["payout"], out["seats"][0]["stake"]), ("bust", 0, 20))
        self.assertEqual(self.store.balance(G, ALICE), 80)

    def test_money_is_conserved_across_seats(self):
        self.rig(["10", "10", "9", "8", "7", "9"])
        snap = self.table(seed="x")
        for user, stake in ((ALICE, 10), (BOB, 25)):
            self.fund(user, 100)
            self.store.join_round(G, snap["round_id"], user, stake)
        self.store.deal(G, snap["round_id"])
        self.store.play(G, snap["round_id"], 0, "stand", "member", 0)
        out = self.store.play(G, snap["round_id"], 1, "stand", "member", 1)
        net = sum(r["amount"] for r in self.game_ledger())
        self.assertEqual(net, sum(s["payout"] for s in out["seats"]) - sum(s["stake"] for s in out["seats"]))

    def test_payout_clamps_at_the_balance_cap(self):
        top = self.store.MAX_CURRENCY_BALANCE
        self.store.db.execute("INSERT INTO currency_balances(guild_id,user_id,balance,updated_at) VALUES(?,?,?,0)", (G, ALICE, top))
        self.store.db.commit()
        self.rig(["10", "9", "10", "8"])
        snap = self.table(seed="cap")
        self.store.join_round(G, snap["round_id"], ALICE, 10)
        self.store.deal(G, snap["round_id"])
        out = self.store.play(G, snap["round_id"], 0, "stand", "member", 0)
        self.assertEqual(out["status"], "settled")
        self.assertEqual(out["seats"][0]["payout"], 10)  # 20 owed, only 10 fits
        self.assertEqual(self.store.balance(G, ALICE), top)
        self.assertEqual(self.game_ledger()[-1]["amount"], 10)

    def test_payout_at_the_cap_pays_nothing_and_still_settles(self):
        top = self.store.MAX_CURRENCY_BALANCE
        self.store.db.execute("INSERT INTO currency_balances(guild_id,user_id,balance,updated_at) VALUES(?,?,?,0)", (G, ALICE, top))
        self.store.db.commit()
        self.rig(["10", "9", "10", "8"])
        snap = self.table(seed="cap2")
        self.store.join_round(G, snap["round_id"], ALICE, 10)
        self.store.deal(G, snap["round_id"])
        # a concurrent admin grant fills the 10 coins the stake freed
        self.store.db.execute("UPDATE currency_balances SET balance=? WHERE guild_id=? AND user_id=?", (top, G, ALICE))
        self.store.db.commit()
        out = self.store.play(G, snap["round_id"], 0, "stand", "member", 0)
        self.assertEqual((out["status"], out["seats"][0]["payout"]), ("settled", 0))
        self.assertEqual(self.store.balance(G, ALICE), top)
        self.assertEqual(len(self.game_ledger()), 1)


class CancelAndTableTests(TableCase):
    def test_cancel_refunds_everything_including_doubles(self):
        snap = self.started(["5", "5", "9", "6", "6", "8"], stakes=((ALICE, 10), (BOB, 20)), tail="2")
        # alice 5+6=11 doubles, drawing a 2 -> 13, then bob is on turn
        self.store.play(G, snap["round_id"], 0, "double", "member", 0)
        self.assertEqual(self.store.balance(G, ALICE), 80)
        out = self.store.cancel_round(G, snap["round_id"], "restart")
        self.assertEqual(out["status"], "cancelled")
        self.assertEqual((self.store.balance(G, ALICE), self.store.balance(G, BOB)), (100, 100))
        self.assertEqual(sum(r["amount"] for r in self.game_ledger()), 0)
        self.assertIsNotNone(out["seed"])

    def test_cancel_is_idempotent_and_leaves_settled_alone(self):
        snap = self.started(["10", "9", "10", "8"])
        again = self.store.cancel_round(G, snap["round_id"], "x")
        self.assertEqual(again["status"], "cancelled")
        n = len(self.game_ledger())
        self.store.cancel_round(G, snap["round_id"], "x")
        self.assertEqual(len(self.game_ledger()), n)
        self.assertEqual(self.store.balance(G, ALICE), 100)
        # settled rounds are left alone
        self.store.next_round(G, snap["table_id"])
        s2 = self.started_again(["10", "9", "10", "8"])
        self.store.play(G, s2["round_id"], 0, "stand", "member", 0)
        before = (self.store.balance(G, ALICE), len(self.game_ledger()))
        out = self.store.cancel_round(G, s2["round_id"], "late")
        self.assertEqual(out["status"], "settled")
        self.assertEqual((self.store.balance(G, ALICE), len(self.game_ledger())), before)

    def started_again(self, cards):
        self.rig(cards)
        table = self.store.open_table_for(G, CH)
        round_id = self.store.latest_round(G, table["id"])["round_id"]
        self.store.join_round(G, round_id, ALICE, 10)
        return self.store.deal(G, round_id)

    def test_cancel_a_joining_round(self):
        snap = self.table()
        self.fund(ALICE, 50)
        self.store.join_round(G, snap["round_id"], ALICE, 50)
        out = self.store.cancel_round(G, snap["round_id"], "idle")
        self.assertEqual((out["status"], self.store.balance(G, ALICE)), ("cancelled", 50))

    def test_next_round_rules(self):
        snap = self.table()
        tid = snap["table_id"]
        with self.assertRaisesRegex(GameError, "unfinished|still"):
            self.store.next_round(G, tid)
        self.store.cancel_round(G, snap["round_id"], "x")
        nxt = self.store.next_round(G, tid)
        self.assertEqual((nxt["number"], nxt["status"], nxt["table_id"]), (2, "joining", tid))
        self.assertNotEqual(nxt["seed_hash"], snap["seed_hash"])
        self.assertEqual(self.store.latest_round(G, tid)["round_id"], nxt["round_id"])
        self.store.close_table(G, tid)
        with self.assertRaises(GameError):
            self.store.next_round(G, tid)
        with self.assertRaises(GameError):
            self.store.next_round(G, 999)

    def test_close_table_refunds_the_unfinished_round(self):
        snap = self.started(["10", "9", "10", "8"])
        self.store.close_table(G, snap["table_id"])
        self.assertEqual(self.store.balance(G, ALICE), 100)
        self.assertEqual(self.store.round_snapshot(G, snap["round_id"])["status"], "cancelled")
        self.assertIsNone(self.store.open_table_for(G, CH))
        self.store.close_table(G, snap["table_id"])  # idempotent

    def test_set_table_message(self):
        snap = self.table()
        self.store.set_table_message(G, snap["table_id"], 777)
        self.assertEqual(self.store.open_table_for(G, CH)["message_id"], 777)
        self.assertEqual(self.store.round_snapshot(G, snap["round_id"])["message_id"], 777)

    def test_unfinished_rounds_lists_joining_and_playing_across_guilds(self):
        self.store.set_game_channel(2, 200, True)
        self.store.set_game_channel(G, 101, True)
        joining = self.table()
        playing = self.started_in(2, 200, ["10", "9", "10", "8"], 8)
        done = self.store.open_table(G, 101, "blackjack", 1)
        self.store.cancel_round(G, done["round_id"], "x")
        found = {(r["guild_id"], r["round_id"]): r for r in self.store.unfinished_rounds()}
        self.assertEqual(set(found), {(G, joining["round_id"]), (2, playing["round_id"])})
        row = found[(2, playing["round_id"])]
        self.assertEqual((row["table_id"], row["channel_id"], row["status"]), (playing["table_id"], 200, "playing"))
        self.assertNotIn("seed", row)

    def started_in(self, guild, channel, cards, user):
        self.rig(cards)
        snap = self.store.open_table(guild, channel, "blackjack", 1, seed="z")
        self.fund(user, 50, guild)
        self.store.join_round(guild, snap["round_id"], user, 10)
        return self.store.deal(guild, snap["round_id"])


class SnapshotAndReplayTests(TableCase):
    def test_snapshot_hides_the_seed_until_the_round_ends(self):
        seed = "secret-seed-value"
        self.rig(["10", "9", "7", "8"])
        snap = self.table(seed=seed)
        self.fund(ALICE, 100)
        self.store.join_round(G, snap["round_id"], ALICE, 10)
        dealt = self.store.deal(G, snap["round_id"])
        self.assertIsNone(dealt["seed"])
        self.assertNotIn(seed, repr(dealt))
        self.assertEqual(dealt["view"]["dealer_hidden"], True)
        self.assertEqual(len(dealt["view"]["dealer"]), 1)
        self.assertEqual(dealt["seed_hash"], seed_hash(seed))
        self.assertEqual(dealt["view"]["seed_hash"], seed_hash(seed)[:12])
        out = self.store.play(G, snap["round_id"], 0, "stand", "member", 0)
        self.assertEqual(out["seed"], seed)
        self.assertEqual(self.store.round_snapshot(G, snap["round_id"])["seed"], seed)

    def test_seed_does_not_leak_through_errors(self):
        seed = "secret-seed-value"
        snap = self.table(seed=seed)
        self.fund(ALICE, 100)
        self.store.join_round(G, snap["round_id"], ALICE, 10)
        self.store.deal(G, snap["round_id"])
        for call in (lambda: self.store.play(G, snap["round_id"], 3, "stand", "member", 0),
                     lambda: self.store.play(G, snap["round_id"], 0, "nope", "member", 0),
                     lambda: self.store.play(G, snap["round_id"], 0, "stand", "member", 9),
                     lambda: self.store.join_round(G, snap["round_id"], BOB, 5),
                     lambda: self.store.deal(G, snap["round_id"])):
            with self.assertRaises(ValueError) as caught:
                call()
            self.assertNotIn(seed, str(caught.exception))

    def test_cancelled_snapshot_reveals_the_seed(self):
        snap = self.table(seed="abc123")
        self.assertIsNone(snap["seed"])
        self.assertEqual(self.store.cancel_round(G, snap["round_id"], "x")["seed"], "abc123")

    def test_replay_from_the_database_equals_the_live_state(self):
        snap = self.table(seed="replay-seed")
        for user in (ALICE, BOB):
            self.fund(user, 100)
            self.store.join_round(G, snap["round_id"], user, 10)
        out = self.store.deal(G, snap["round_id"])
        count = 0

        def rebuilt(out):
            seats = [bj.Seat(s["index"], s["kind"], s["ref_id"], s["stake"]) for s in out["seats"]]
            rows = self.store.db.execute("SELECT seat_index,move FROM game_moves ORDER BY id").fetchall()
            return seats, bj.replay("replay-seed", seats, [(r[0], r[1]) for r in rows])

        while out["status"] == "playing":
            seat = out["view"]["turn"]
            move = "hit" if out["view"]["totals"][seat][0] < 15 else "stand"
            out = self.store.play(G, snap["round_id"], seat, move, "member", count)
            count += 1
            state = rebuilt(out)[1]
            self.assertEqual(state.hands, tuple(tuple(h) for h in out["view"]["hands"]))
            self.assertEqual(state.finished, out["status"] == "settled")
        self.assertEqual(out["status"], "settled")
        for s, expected in zip(out["seats"], bj.result(rebuilt(out)[1]).seats):
            self.assertEqual((s["outcome"], s["payout"]), (expected.outcome, expected.returned))
        total = sum(r["amount"] for r in self.game_ledger())
        self.assertEqual(total, sum(s["payout"] for s in out["seats"]) - sum(s["stake"] for s in out["seats"]))

    def test_snapshot_legal_moves_only_for_the_seat_on_turn(self):
        snap = self.started(["10", "10", "9", "8", "7", "9"], stakes=((ALICE, 10), (BOB, 10)))
        self.assertEqual(snap["legal"], {0: ("hit", "stand", "double")})
        self.store.play(G, snap["round_id"], 0, "stand", "member", 0)
        self.assertEqual(self.store.round_snapshot(G, snap["round_id"])["legal"], {1: ("hit", "stand", "double")})

    def test_snapshot_legal_moves_respect_the_balance(self):
        snap = self.started(["10", "10", "9", "8", "7", "9"], stakes=((ALICE, 60),), funds=100)
        self.assertEqual(snap["legal"], {0: ("hit", "stand")})  # 40 left, a double needs 60


class HardeningTests(TableCase):
    def test_write_methods_build_their_snapshot_inside_the_transaction(self):
        seen = []
        real = self.store._snapshot_reads

        def spy(guild_id, round_id):
            seen.append(self.store.db.in_transaction)
            return real(guild_id, round_id)

        self.store._snapshot_reads = spy
        snap = self.started(["10", "9", "10", "8"])
        self.store.play(G, snap["round_id"], 0, "stand", "member", 0)
        self.store.next_round(G, snap["table_id"])
        self.assertTrue(seen and all(seen))
        seen.clear()
        self.store.round_snapshot(G, snap["round_id"])
        self.assertEqual(seen, [True])  # a read transaction of its own
        self.assertFalse(self.store.db.in_transaction)

    def test_snapshot_reads_share_one_transaction(self):
        snap = self.started(["10", "9", "10", "8"])
        states = []
        for name in ("_game_seats", "_game_moves"):
            real = getattr(self.store, name)

            def spy(*args, real=real):
                states.append(self.store.db.in_transaction)
                return real(*args)

            setattr(self.store, name, spy)
        self.store.round_snapshot(G, snap["round_id"])
        self.assertTrue(states and all(states))

    def test_damaged_move_log_does_not_block_cancel_or_close(self):
        for closer in ("cancel", "close"):
            with self.subTest(closer):
                store = Store()
                self.addCleanup(store.close)
                store.set_game_channel(G, CH, True)
                self.rig(["10", "9", "7", "8"])
                snap = store.open_table(G, CH, "blackjack", 1, seed="bad")
                store.change_balance(G, ALICE, 100, "start", 1)
                store.join_round(G, snap["round_id"], ALICE, 10)
                store.deal(G, snap["round_id"])
                for move in ("stand", "stand"):
                    store.db.execute("INSERT INTO game_moves(guild_id,round_id,seat_index,move,actor,created_at) VALUES(?,?,0,?,'member',0)", (G, snap["round_id"], move))
                store.db.commit()
                with self.assertRaises(GameError):
                    store.round_snapshot(G, snap["round_id"])
                if closer == "cancel":
                    out = store.cancel_round(G, snap["round_id"], "x")
                    self.assertEqual((out["status"], out["view"]), ("cancelled", None))
                else:
                    store.close_table(G, snap["table_id"])
                    out = store.round_snapshot(G, snap["round_id"])
                    self.assertEqual(out["status"], "cancelled")
                self.assertEqual(store.balance(G, ALICE), 100)

    def test_leave_near_the_balance_cap_still_refunds_what_fits(self):
        snap = self.table()
        self.fund(ALICE, 100)
        self.store.join_round(G, snap["round_id"], ALICE, 10)
        top = self.store.MAX_CURRENCY_BALANCE
        self.store.db.execute("UPDATE currency_balances SET balance=? WHERE guild_id=? AND user_id=?", (top - 4, G, ALICE))
        self.store.db.commit()
        out = self.store.leave_round(G, snap["round_id"], ALICE)
        self.assertEqual(out["seats"], [])
        self.assertEqual(self.store.balance(G, ALICE), top)

    def test_turning_a_channel_off_closes_its_table_and_refunds(self):
        snap = self.started(["10", "9", "10", "8"])
        self.store.set_game_channel(G, 101, True)
        self.assertEqual(self.store.set_game_channel(G, 101, False), {"closed_table": None})
        self.assertEqual(self.store.set_game_channel(G, CH, True), {"closed_table": None})
        self.assertEqual(self.store.set_game_channel(G, CH, False), {"closed_table": snap["table_id"]})
        self.assertIsNone(self.store.open_table_for(G, CH))
        self.assertEqual(self.store.round_snapshot(G, snap["round_id"])["status"], "cancelled")
        self.assertEqual(self.store.balance(G, ALICE), 100)
        self.assertEqual(self.store.set_game_channel(G, CH, False), {"closed_table": None})

    def test_illegal_move_messages_are_member_facing(self):
        snap = self.started(["10", "10", "9", "8", "7", "9"], stakes=((ALICE, 10), (BOB, 10)))
        for seat, move, text in ((1, "stand", "It's not your turn."), (0, "split", "That move isn't allowed right now."), (7, "stand", "It's not your turn.")):
            with self.assertRaises(GameError) as caught:
                self.store.play(G, snap["round_id"], seat, move, "member", 0)
            self.assertEqual(str(caught.exception), text)


class IsolationTests(TableCase):
    def test_a_round_id_from_another_guild_fails_everywhere(self):
        snap = self.started(["10", "9", "10", "8"])
        rid, tid = snap["round_id"], snap["table_id"]
        self.store.set_game_channel(2, CH, True)
        self.fund(ALICE, 100, 2)
        before = (self.store.balance(G, ALICE), self.store.balance(2, ALICE), len(self.game_ledger()), len(self.game_ledger(2)))
        calls = {
            "snapshot": lambda: self.store.round_snapshot(2, rid),
            "join": lambda: self.store.join_round(2, rid, BOB, 5),
            "leave": lambda: self.store.leave_round(2, rid, ALICE),
            "deal": lambda: self.store.deal(2, rid),
            "play": lambda: self.store.play(2, rid, 0, "stand", "member", 0),
            "cancel": lambda: self.store.cancel_round(2, rid, "x"),
            "next": lambda: self.store.next_round(2, tid),
            "close": lambda: self.store.close_table(2, tid),
            "message": lambda: self.store.set_table_message(2, tid, 1),
            "latest": lambda: self.store.latest_round(2, tid),
        }
        for name, call in calls.items():
            with self.subTest(name), self.assertRaises(GameError):
                call()
        after = (self.store.balance(G, ALICE), self.store.balance(2, ALICE), len(self.game_ledger()), len(self.game_ledger(2)))
        self.assertEqual(before, after)
        self.assertEqual(self.store.round_snapshot(G, rid)["status"], "playing")
        self.assertEqual(self.moves(), [])
        self.assertEqual(self.store.open_table_for(G, CH)["id"], tid)
        self.assertIsNone(self.store.open_table_for(2, CH))

    def test_same_channel_in_two_guilds_is_independent(self):
        self.store.set_game_channel(2, CH, True)
        a = self.table()
        b = self.store.open_table(2, CH, "blackjack", 1)
        self.assertNotEqual(a["table_id"], b["table_id"])
        self.assertEqual(self.store.game_settings(2), {"min_bet": 1, "max_bet": 1000})
        self.store.set_game_settings(G, 5, 10)
        self.fund(ALICE, 100, 2)
        self.store.join_round(2, b["round_id"], ALICE, 50)  # guild 2 keeps default limits


class ConcurrencyTests(unittest.TestCase):
    def test_two_connections_racing_one_move_apply_it_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "shared.sqlite3"
            first, second = Store(path), Store(path)
            try:
                first.set_game_channel(G, CH, True)
                snap = first.open_table(G, CH, "blackjack", 1, seed="race")
                first.change_balance(G, ALICE, 100, "start", 1)
                first.join_round(G, snap["round_id"], ALICE, 10)
                shoe = tuple(card(r) for r in ["5", "9", "6", "8"]) + (card("2"),) * 100
                results = []
                barrier = threading.Barrier(2)

                def press(store):
                    barrier.wait()
                    try:
                        store.play(G, snap["round_id"], 0, "hit", "member", 0)
                        results.append("ok")
                    except ConflictError:
                        results.append("conflict")
                    except Exception as exc:  # pragma: no cover
                        results.append(repr(exc))

                with mock.patch.object(bj, "new_shoe", return_value=shoe):
                    first.deal(G, snap["round_id"])
                    threads = [threading.Thread(target=press, args=(s,)) for s in (first, second)]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()
                self.assertEqual(sorted(results), ["conflict", "ok"])
                self.assertEqual(second.round_snapshot(G, snap["round_id"])["moves"], 1)
                self.assertEqual(second.db.execute("SELECT COUNT(*) FROM game_moves").fetchone()[0], 1)
            finally:
                first.close()
                second.close()


if __name__ == "__main__":
    unittest.main()
