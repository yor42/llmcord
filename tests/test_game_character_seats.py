"""FEAT-20 part A: character seats in blackjack (store only; schema v16 additions game_seats.brought_by, guild_settings.game_character_talk)."""
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from test_game_tables import ALICE, BOB, G, TableCase
from test_migration_atomic import legacy_file

from llmcord_core.admin_store import ConflictError, GameError
from llmcord_core.games import blackjack as bj
from llmcord_core.store import Store

MAX = Store.MAX_CURRENCY_BALANCE


class SeatCase(TableCase):
    def setUp(self):
        super().setUp()
        s = self.store
        self.w = s.create_space(G, "W", "world")
        self.ann, self.bo, self.cy = (s.create_character(G, self.w, n) for n in ("Ann", "Bo", "Cy"))
        s.set_daily_settings(G, 0, 0, 7)
        for cid in (self.ann, self.bo, self.cy):
            s.add_favorite(G, ALICE, cid)
        self.fund(ALICE, 1000)

    def wallet(self, cid, amount):
        self.store.change_character_balance(G, cid, amount, "start", 1)

    def bal(self, cid):
        return self.store.character_balance(G, cid)

    def join(self, stake=10, snap=None, user=ALICE, space=None):
        snap = snap or self.table(seed="s")
        return snap, self.store.join_with_favorites(G, snap["round_id"], user, stake, self.w if space is None else space)

    def seats(self, rid):
        return [(r["seat_index"], r["kind"], r["ref_id"], r["stake"], r["brought_by"]) for r in
                self.store.db.execute("SELECT * FROM game_seats WHERE round_id=? ORDER BY seat_index", (rid,))]


class JoinTests(SeatCase):
    def test_favorites_seat_in_order_after_the_member(self):
        for c in (self.ann, self.bo, self.cy):
            self.wallet(c, 100)
        snap, out = self.join(10)
        self.assertEqual(out["skipped"], [])
        self.assertEqual(out["seated"], [{"character_id": c, "name": n, "stake": 10} for c, n in ((self.ann, "Ann"), (self.bo, "Bo"), (self.cy, "Cy"))])
        self.assertEqual(self.seats(snap["round_id"]), [(0, "member", ALICE, 10, None), (1, "character", self.ann, 10, ALICE),
                                                       (2, "character", self.bo, 10, ALICE), (3, "character", self.cy, 10, ALICE)])
        self.assertEqual([(s["kind"], s["name"], s["brought_by"]) for s in out["snapshot"]["seats"]],
                         [("member", None, None), ("character", "Ann", ALICE), ("character", "Bo", ALICE), ("character", "Cy", ALICE)])
        self.assertEqual((self.store.balance(G, ALICE), self.bal(self.ann)), (990, 90))
        row = self.store.ledger(G, character_id=self.ann)[0]
        self.assertEqual((row["amount"], row["source"], row["holder_kind"], row["reason"]), (-10, "game", "character", "Blackjack bet (table 1, round 1)"))

    def test_seats_stop_at_the_table_limit(self):
        for c in (self.ann, self.bo, self.cy):
            self.wallet(c, 100)
        snap = self.table(seed="s")
        for user in range(10, 15):
            self.fund(user, 10)
            self.store.join_round(G, snap["round_id"], user, 5)
        _, out = self.join(10, snap)
        self.assertEqual(len(out["snapshot"]["seats"]), bj.MAX_SEATS)
        self.assertEqual([c["name"] for c in out["seated"]], ["Ann"])
        self.assertEqual(out["skipped"], [{"character_id": self.bo, "name": "Bo", "reason": "full"}, {"character_id": self.cy, "name": "Cy", "reason": "full"}])
        self.assertEqual(self.bal(self.bo), 100)

    def test_stake_is_lowered_to_the_wallet_and_the_max_bet(self):
        self.wallet(self.ann, 7)
        self.wallet(self.bo, 100)
        self.store.set_game_settings(G, 1, 20)
        _, out = self.join(20)
        self.assertEqual([(c["name"], c["stake"]) for c in out["seated"]], [("Ann", 7), ("Bo", 20)])
        self.assertEqual((self.bal(self.ann), self.bal(self.bo)), (0, 80))

    def test_broke_character_is_skipped_below_the_min_bet_without_writing(self):
        self.wallet(self.ann, 4)
        self.wallet(self.bo, 100)
        self.store.set_game_settings(G, 5, 1000)
        before = len(self.store.ledger(G, character_id=self.ann))
        _, out = self.join(10)
        self.assertEqual(out["skipped"], [{"character_id": self.ann, "name": "Ann", "reason": "broke"}, {"character_id": self.cy, "name": "Cy", "reason": "broke"}])
        self.assertEqual([c["name"] for c in out["seated"]], ["Bo"])
        self.assertEqual((self.bal(self.ann), len(self.store.ledger(G, character_id=self.ann))), (4, before))

    def test_refill_applies_to_the_wallet_for_the_bet(self):
        self.store.set_daily_settings(G, 100, 0, 7)
        _, out = self.join(30)
        self.assertEqual([c["stake"] for c in out["seated"]], [30, 30, 30])
        self.assertEqual(self.bal(self.ann), 70)
        self.assertEqual([r["amount"] for r in self.store.ledger(G, character_id=self.ann)], [-30, 100])

    def test_a_seated_character_is_skipped_not_doubled(self):
        self.wallet(self.ann, 100)
        snap, out = self.join(10)
        self.assertEqual(len(out["seated"]), 1)
        self.store.db.execute("DELETE FROM game_seats WHERE round_id=? AND kind='member'", (snap["round_id"],))
        self.store.db.execute("UPDATE game_seats SET seat_index=0")
        self.store.db.commit()
        out = self.store.join_with_favorites(G, snap["round_id"], ALICE, 10, self.w)
        self.assertEqual([s["reason"] for s in out["skipped"] if s["name"] == "Ann"], ["seated"])
        self.assertEqual(self.bal(self.ann), 90)

    def test_archived_favorite_never_seats_even_when_the_server_allows_them(self):
        self.wallet(self.ann, 100)
        self.wallet(self.bo, 100)
        self.store.set_archived_favorites(G, True)
        self.store.archive_character(G, self.ann, True)
        self.assertIn(self.ann, [f["character_id"] for f in self.store.eligible_favorites(G, ALICE, self.w)])
        _, out = self.join(10)
        self.assertEqual([c["name"] for c in out["seated"]], ["Bo"])
        self.assertIn({"character_id": self.ann, "name": "Ann", "reason": "archived"}, out["skipped"])
        self.assertEqual(self.bal(self.ann), 100)

    def test_unlinked_world_and_other_guild_favorites_never_seat(self):
        s = self.store
        w2 = s.create_space(G, "W2", "world")
        far = s.create_character(G, w2, "Far")
        s.add_favorite(G, ALICE, far)
        self.wallet(far, 100)
        hub = s.create_space(G, "Hub", "hub")
        s.link_world(G, hub, w2)
        for c in (self.ann, self.bo, self.cy):
            self.wallet(c, 100)
        snap, out = self.join(10)
        self.assertNotIn("Far", [c["name"] for c in out["seated"]])
        self.assertEqual(self.bal(far), 100)
        other = s.create_space(2, "O", "world")
        s.create_character(2, other, "Zed")
        self.assertEqual(self.store.eligible_favorites(2, ALICE, other), [])
        self.fund(BOB, 100)
        self.store.cancel_round(G, snap["round_id"], "x")
        _, again = self.join(10, self.store.next_round(G, snap["table_id"]), user=BOB)
        self.assertEqual(again["seated"], [])  # Bob has no favorites

    def test_refused_member_join_writes_nothing_not_even_a_refill(self):
        self.store.set_daily_settings(G, 100, 0, 7)
        snap = self.table(seed="s")
        for stake, why in ((5000, "between"), (10, "need")):
            if why == "need":
                self.store.db.execute("UPDATE currency_balances SET balance=0")
                self.store.db.commit()
            with self.assertRaisesRegex(GameError, why):
                self.store.join_with_favorites(G, snap["round_id"], ALICE, stake, self.w)
        self.assertEqual(self.seats(snap["round_id"]), [])
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM currency_ledger WHERE holder_kind='character'").fetchone()[0], 0)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM character_balances").fetchone()[0], 0)

    def test_no_space_seats_only_the_member(self):
        self.wallet(self.ann, 100)
        snap = self.table(seed="s")
        out = self.store.join_with_favorites(G, snap["round_id"], ALICE, 10, None)
        self.assertEqual((out["seated"], out["skipped"], len(out["snapshot"]["seats"])), ([], [], 1))
        self.assertEqual(self.bal(self.ann), 100)

    def test_join_round_is_unchanged_for_members(self):
        snap = self.table(seed="s")
        out = self.store.join_round(G, snap["round_id"], ALICE, 10)
        self.assertEqual(self.seats(snap["round_id"]), [(0, "member", ALICE, 10, None)])
        self.assertEqual(out["seats"][0]["name"], None)


class LeaveTests(SeatCase):
    def test_leaving_refunds_and_removes_the_brought_characters_and_keeps_indexes_contiguous(self):
        for c in (self.ann, self.bo):
            self.wallet(c, 100)
        self.fund(BOB, 100)
        snap = self.table(seed="s")
        self.store.join_round(G, snap["round_id"], BOB, 5)
        self.store.join_with_favorites(G, snap["round_id"], ALICE, 10, self.w)
        self.assertEqual(len(self.seats(snap["round_id"])), 4)
        out = self.store.leave_round(G, snap["round_id"], ALICE)
        self.assertEqual(self.seats(snap["round_id"]), [(0, "member", BOB, 5, None)])
        self.assertEqual([s["index"] for s in out["seats"]], [0])
        self.assertEqual((self.store.balance(G, ALICE), self.bal(self.ann), self.bal(self.bo)), (1000, 100, 100))
        row = self.store.ledger(G, character_id=self.ann)[0]
        self.assertEqual((row["amount"], row["reason"]), (10, "Blackjack refund (table 1, round 1)"))

    def test_other_members_character_seats_stay_and_reindex(self):
        self.wallet(self.ann, 100)
        self.wallet(self.bo, 100)
        self.fund(BOB, 100)
        self.store.add_favorite(G, BOB, self.bo)
        snap = self.table(seed="s")
        self.store.join_with_favorites(G, snap["round_id"], ALICE, 10, self.w)   # Alice, Ann, Bo(by Alice)
        out = self.store.join_with_favorites(G, snap["round_id"], BOB, 5, self.w)  # Bob; Bo already seated
        self.assertEqual([s["reason"] for s in out["skipped"] if s["name"] == "Bo"], ["seated"])
        self.store.leave_round(G, snap["round_id"], ALICE)
        self.assertEqual(self.seats(snap["round_id"]), [(0, "member", BOB, 5, None)])


class PlayTests(SeatCase):
    def setUp(self):
        super().setUp()
        self.store.remove_favorite(G, ALICE, self.bo)
        self.store.remove_favorite(G, ALICE, self.cy)

    def dealt(self, cards, stake=10, wallet=100, tail="2"):
        """Round with Alice (seat 0) and Ann (seat 1), rigged and dealt. Rounds after the first reuse the table."""
        if wallet:
            self.wallet(self.ann, wallet)
        self.rig(cards, tail)
        latest = self.store.db.execute("SELECT table_id FROM game_rounds ORDER BY id DESC LIMIT 1").fetchone()
        snap = self.store.next_round(G, latest[0]) if latest else self.table(seed="c")
        self.store.join_with_favorites(G, snap["round_id"], ALICE, stake, self.w)
        return self.store.deal(G, snap["round_id"])

    def finish(self, snap, *moves):
        for seat, move in moves:
            snap = self.store.play(G, snap["round_id"], seat, move, "character" if seat else "member", snap["moves"])
        return snap

    def test_win_loss_and_push_pay_the_character_wallet(self):
        out = self.finish(self.dealt(["10", "10", "9", "7", "8", "8"]), (0, "stand"), (1, "stand"))  # member 17 push, Ann 18 win vs 17
        self.assertEqual([(x["outcome"], x["payout"]) for x in out["seats"]], [("push", 10), ("win", 20)])
        self.assertEqual((self.bal(self.ann), self.store.balance(G, ALICE)), (110, 1000))
        top = self.store.ledger(G, character_id=self.ann)[0]
        self.assertEqual((top["amount"], top["reason"], top["source"]), (20, "Blackjack payout (win), table 1, round 1", "game"))
        self.assertEqual(self.moves()[1]["actor"], "character")
        out = self.finish(self.dealt(["7", "7", "9", "9", "9", "10"], wallet=0), (0, "stand"), (1, "stand"))  # 16 vs 19
        self.assertEqual([x["outcome"] for x in out["seats"]], ["lose", "lose"])
        self.assertEqual((self.bal(self.ann), self.store.ledger(G, character_id=self.ann)[0]["amount"]), (100, -10))
        out = self.finish(self.dealt(["10", "7", "9", "9", "10", "8"], wallet=0), (0, "stand"), (1, "stand"))  # member 19 wins, Ann 17 pushes the dealer's 17
        self.assertEqual([x["outcome"] for x in out["seats"]], ["win", "push"])
        self.assertEqual(self.bal(self.ann), 100)

    def test_natural_blackjack_pays_three_to_two(self):
        out = self.finish(self.dealt(["10", "A", "9", "9", "K", "8"]), (0, "stand"))
        self.assertEqual((out["seats"][1]["outcome"], out["seats"][1]["payout"]), ("blackjack", 25))
        self.assertEqual(self.bal(self.ann), 100 - 10 + 25)

    def test_double_is_not_offered_or_allowed_when_the_wallet_is_short(self):
        snap = self.dealt(["5", "5", "9", "6", "6", "8"], wallet=10, tail="10")  # Ann 11 but her wallet is empty after the bet
        snap = self.finish(snap, (0, "stand"))
        self.assertEqual(snap["legal"], {1: ("hit", "stand")})
        with self.assertRaises(GameError):
            self.store.play(G, snap["round_id"], 1, "double", "character", snap["moves"])
        self.assertEqual((self.bal(self.ann), len(self.moves())), (0, 1))

    def test_double_debits_the_wallet_and_pays_on_the_doubled_stake(self):
        snap = self.finish(self.dealt(["5", "5", "9", "6", "6", "8"], tail="10"), (0, "stand"))
        self.assertEqual(snap["legal"], {1: ("hit", "stand", "double")})
        out = self.finish(snap, (1, "double"))
        self.assertEqual((out["seats"][1]["stake"], out["seats"][1]["outcome"], out["seats"][1]["payout"]), (20, "win", 40))
        self.assertEqual([r["amount"] for r in self.store.ledger(G, character_id=self.ann)][:3], [40, -10, -10])
        self.assertEqual(self.store.ledger(G, character_id=self.ann)[2]["reason"], "Blackjack bet (table 1, round 1)")
        self.assertEqual(self.store.ledger(G, character_id=self.ann)[1]["reason"], "Blackjack double (table 1, round 1)")
        self.assertEqual(self.bal(self.ann), 120)

    def test_payout_is_clamped_at_the_maximum(self):
        self.store.db.execute("INSERT INTO character_balances(guild_id,character_id,balance,refill_day,updated_at) VALUES(?,?,?,'',0)", (G, self.ann, MAX - 5))
        self.store.db.commit()
        out = self.finish(self.dealt(["10", "10", "9", "10", "8", "8"], wallet=0), (0, "stand"), (1, "stand"))
        self.assertEqual((self.bal(self.ann), out["seats"][1]["payout"]), (MAX, 15))

    def test_nobody_moves_the_wrong_kind_of_seat(self):
        snap = self.dealt(["10", "10", "9", "10", "8", "8"])
        with self.assertRaisesRegex(GameError, "cannot move"):
            self.store.play(G, snap["round_id"], 0, "stand", "character", 0)
        self.store.play(G, snap["round_id"], 0, "stand", "member", 0)
        with self.assertRaisesRegex(GameError, "cannot move"):
            self.store.play(G, snap["round_id"], 1, "stand", "member", 1)
        with self.assertRaises(GameError):
            self.store.play(G, snap["round_id"], 1, "stand", "robot", 1)
        self.assertEqual(len(self.moves()), 1)
        self.store.play(G, snap["round_id"], 1, "stand", "timeout", 1)
        self.assertEqual(self.store.round_snapshot(G, snap["round_id"])["status"], "settled")

    def test_cancel_refunds_character_wallets_while_joining_and_while_playing(self):
        self.wallet(self.ann, 100)
        snap = self.table(seed="s")
        self.store.join_with_favorites(G, snap["round_id"], ALICE, 10, self.w)
        out = self.store.cancel_round(G, snap["round_id"], "nobody joined")
        self.assertEqual([(x["outcome"], x["payout"]) for x in out["seats"]], [("refund", 10), ("refund", 10)])
        self.assertEqual((self.bal(self.ann), self.store.balance(G, ALICE)), (100, 1000))
        snap = self.finish(self.dealt(["5", "5", "9", "6", "6", "8"], wallet=0, tail="10"), (0, "stand"))
        self.store.play(G, snap["round_id"], 1, "double", "character", snap["moves"])
        self.assertEqual(self.store.round_snapshot(G, snap["round_id"])["status"], "settled")
        snap = self.dealt(["5", "5", "9", "6", "6", "8"], tail="10")
        before = self.bal(self.ann)
        self.store.cancel_round(G, snap["round_id"], "restart")
        self.assertEqual(self.bal(self.ann), before + 10)
        self.assertEqual(self.store.round_snapshot(G, snap["round_id"])["status"], "cancelled")

    def test_deleted_character_does_not_break_settlement(self):
        snap = self.dealt(["10", "10", "9", "10", "8", "8"])
        self.store.db.execute("PRAGMA foreign_keys=OFF")
        self.store.db.execute("DELETE FROM characters WHERE id=?", (self.ann,))
        self.store.db.commit()
        out = self.finish(snap, (0, "stand"), (1, "stand"))
        self.assertEqual((out["status"], out["seats"][1]["payout"], out["seats"][1]["name"]), ("settled", 0, f"Deleted character (#{self.ann})"))

    def test_context_for_part_b(self):
        snap = self.dealt(["10", "10", "9", "10", "8", "8"])
        self.assertIsNone(self.store.character_seat_context(G, snap["round_id"], 0))
        self.assertIsNone(self.store.character_seat_context(G, snap["round_id"], 5))
        self.assertEqual(self.store.character_seat_context(G, snap["round_id"], 1),
                         {"seat_index": 1, "character_id": self.ann, "name": "Ann", "brought_by": ALICE, "stake": 10})
        with self.assertRaises(GameError):
            self.store.character_seat_context(2, snap["round_id"], 1)


class DeleteGuardTests(SeatCase):
    def test_delete_is_refused_mid_round_and_allowed_after_settle_or_cancel(self):
        self.wallet(self.ann, 100)
        snap = self.table(seed="s")
        self.store.join_with_favorites(G, snap["round_id"], ALICE, 10, self.w)
        rev = lambda: self.store.owner_revision(G, "character", self.ann)  # noqa: E731
        with self.assertRaisesRegex(ValueError, "That character is at a blackjack table. Try again when the round is over."):
            self.store.delete_character(G, self.ann, rev())
        self.store.deal(G, snap["round_id"])
        with self.assertRaisesRegex(ValueError, "blackjack table"):
            self.store.delete_character(G, self.ann, rev())
        with self.assertRaisesRegex(ValueError, "Move or delete"):
            self.store.delete_space(G, self.w, self.store.owner_revision(G, "space", self.w))
        self.store.cancel_round(G, snap["round_id"], "x")
        self.assertTrue(self.store.delete_character(G, self.ann, rev()))
        self.assertEqual(self.store.character_names(G, [self.ann]), {})

    def test_delete_is_allowed_after_settlement_and_for_unseated_characters(self):
        self.wallet(self.ann, 100)
        self.rig(["10", "10", "9", "10", "8", "8"])
        snap = self.table(seed="s")
        self.store.join_with_favorites(G, snap["round_id"], ALICE, 10, self.w)
        snap = self.store.deal(G, snap["round_id"])
        self.store.play(G, snap["round_id"], 0, "stand", "member", 0)
        self.store.play(G, snap["round_id"], 1, "stand", "character", 1)
        self.assertTrue(self.store.delete_character(G, self.ann, self.store.owner_revision(G, "character", self.ann)))
        self.assertTrue(self.store.delete_character(G, self.cy, self.store.owner_revision(G, "character", self.cy)))


class ConservationTests(SeatCase):
    def test_money_in_equals_money_out_plus_house_net(self):
        for c in (self.ann, self.bo, self.cy):
            self.wallet(c, 200)
        self.fund(BOB, 500)
        self.store.add_favorite(G, BOB, self.ann)
        wallets = lambda: sum(self.bal(c) for c in (self.ann, self.bo, self.cy)) + self.store.balance(G, ALICE) + self.store.balance(G, BOB)  # noqa: E731
        start = wallets()
        for n in range(6):
            self.shoe = None
            snap = self.table(seed=f"k{n}") if n == 0 else self.store.next_round(G, 1)
            self.store.join_round(G, snap["round_id"], BOB, 7)
            self.store.join_with_favorites(G, snap["round_id"], ALICE, 13, self.w)
            snap = self.store.deal(G, snap["round_id"])
            while snap["status"] == "playing":
                turn = next(iter(snap["legal"]))
                seat = snap["seats"][turn]
                move = "double" if "double" in snap["legal"][turn] and n % 2 else "stand"
                snap = self.store.play(G, snap["round_id"], turn, move, "character" if seat["kind"] == "character" else "member", snap["moves"])
        rows = self.store.db.execute("SELECT SUM(amount) FROM currency_ledger WHERE source='game'").fetchone()[0]
        seats = self.store.db.execute("SELECT SUM(payout-stake) FROM game_seats").fetchone()[0]
        self.assertEqual(rows, seats)
        self.assertEqual(wallets() - start, rows)

    def test_leave_and_cancel_net_to_zero(self):
        self.wallet(self.ann, 100)
        start = self.bal(self.ann) + self.store.balance(G, ALICE)
        snap = self.table(seed="s")
        self.store.join_with_favorites(G, snap["round_id"], ALICE, 10, self.w)
        self.store.leave_round(G, snap["round_id"], ALICE)
        self.store.join_with_favorites(G, snap["round_id"], ALICE, 10, self.w)
        self.store.cancel_round(G, snap["round_id"], "x")
        self.assertEqual(self.bal(self.ann) + self.store.balance(G, ALICE), start)


class TalkSettingTests(SeatCase):
    def test_default_on_set_off_on_and_conflict(self):
        s = self.store
        self.assertTrue(s.game_character_talk(G))
        self.assertTrue(s.game_character_talk(999))
        self.assertFalse(s.set_game_character_talk(G, False, True))
        self.assertFalse(s.game_character_talk(G))
        self.assertTrue(s.game_character_talk(2))
        with self.assertRaisesRegex(ConflictError, "The character table talk setting was changed elsewhere. Reload the page and try again."):
            s.set_game_character_talk(G, True, True)
        self.assertTrue(s.set_game_character_talk(G, True, False))
        for bad in (1, 0, "yes", None):
            with self.assertRaisesRegex(ValueError, "The setting must be on or off."):
                s.set_game_character_talk(G, bad)
        s.set_game_settings(G, 2, 50)
        self.assertTrue(s.game_character_talk(G))


class TalkLimitTests(SeatCase):
    def usage(self, guild, at, feature="game"):
        self.store.db.execute("INSERT INTO model_usage(guild_id,profile,model,role,cached_tokens,reasoning_tokens,cost_basis,created_at,feature) VALUES(?,?,?,?,0,0,?,?,?)", (guild, "p", "m", "director", "none", at, feature))
        self.store.db.commit()

    def test_default_set_validate_and_conflict(self):
        s = self.store
        self.assertEqual((s.game_talk_daily_limit(G), s.game_talk_daily_limit(999)), (100, 100))
        self.assertEqual(s.set_game_talk_daily_limit(G, 5, 100), 5)
        self.assertEqual((s.game_talk_daily_limit(G), s.game_talk_daily_limit(2)), (5, 100))
        with self.assertRaisesRegex(ConflictError, "The table talk daily limit was changed elsewhere. Reload the page and try again."):
            s.set_game_talk_daily_limit(G, 6, 100)
        for bad in (0, -1, 10001, 1.5, "7", None, True):
            with self.assertRaisesRegex(ValueError, "The daily limit must be a whole number from 1 to 10,000."):
                s.set_game_talk_daily_limit(G, bad)
        self.assertEqual(s.set_game_talk_daily_limit(G, 10000), 10000)
        s.set_game_character_talk(G, False)
        self.assertEqual(s.game_talk_daily_limit(G), 10000)

    def test_calls_today_counts_this_guilds_game_rows_since_the_server_day_start(self):
        s, now = self.store, 1_700_000_000  # 2023-11-14 22:13 UTC
        s.set_guild_timezone(G, "Asia/Seoul")  # local 2023-11-15 07:13; the day began 2023-11-14 15:00 UTC
        start = 1_699_974_000
        self.usage(G, start - 1)
        self.usage(G, start)
        self.usage(G, now)
        self.usage(G, now, "chat")
        self.usage(2, now)
        self.assertEqual(s.game_talk_calls_today(G, now), 2)
        self.assertEqual(s.game_talk_calls_today(G, start + 86400), 0)
        self.assertEqual(s.game_talk_calls_today(2, now), 1)
        self.assertEqual(s.game_talk_calls_today(3, now), 0)


class MigrationTests(unittest.TestCase):
    def test_a_v16_file_without_the_new_columns_gets_them_on_open(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "f.sqlite3"
            Store(path).close()
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("ALTER TABLE guild_settings DROP COLUMN game_character_talk")
                db.execute("ALTER TABLE guild_settings DROP COLUMN game_talk_daily_limit")
                db.execute("ALTER TABLE game_seats DROP COLUMN brought_by")
                db.execute("INSERT INTO guild_settings(guild_id) VALUES(1)")
            store = Store(path)
            self.assertTrue(store.game_character_talk(1))
            self.assertEqual(store.game_talk_daily_limit(1), 100)
            self.assertIn("brought_by", [r[1] for r in store.db.execute("PRAGMA table_info(game_seats)")])
            store.close()
            Store(path).close()

    def test_legacy_file_gets_the_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(legacy_file(folder, 5))
            self.assertTrue(store.game_character_talk(1))
            self.assertEqual(store.game_talk_daily_limit(1), 100)
            self.assertIn("brought_by", [r[1] for r in store.db.execute("PRAGMA table_info(game_seats)")])
            store.close()


if __name__ == "__main__":
    unittest.main()
