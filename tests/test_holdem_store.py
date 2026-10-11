"""FEAT-33 part B: the Texas Hold'em store API (cash table: buy-in, stacks, hands without wallet movement, cash-out, recovery).

Spec: "Hold'em store API (FEAT-33 part B, schema v20)" in docs/engineering/minigames-plan.md. Written before the implementation.
Defaults used throughout: small blind 5, big blind 10, buy-in 400..2000. Table 1 is the first table of a fresh store.

Spec gaps pinned here as assumptions (see the report): table_cash_summary returns {'cashouts': [...], 'seated': [...]};
closing a table whose newest hand is settled cashes out the END stacks; cancel_round closes a holdem table.
"""
import random
import tempfile
import unittest
from pathlib import Path

from test_game_tables import ALICE, BOB, CH, G, TableCase

from llmcord_core.admin_store import ConflictError, GameError
from llmcord_core.games import holdem, registry, seed_hash
from llmcord_core.games.base import Seat
from llmcord_core.store import Store

G2 = 2
CARL, DAN, ERIN = 7, 8, 9
BUY = "Texas Hold'em buy-in (table 1)"
REBUY = "Texas Hold'em rebuy (table 1)"
CASH = "Texas Hold'em cash-out (table 1)"
LEDGER_SQL = "SELECT COUNT(*) FROM currency_ledger WHERE guild_id=?"


class HoldemCase(TableCase):
    def setUp(self):
        super().setUp()
        s = self.store
        s.set_daily_settings(G, 0, 0, 7)
        self.w = s.create_space(G, "W", "world")
        self.ann, self.bo, self.cy = (s.create_character(G, self.w, n) for n in ("Ann", "Bo", "Cy"))

    # ---- builders
    def open(self, seed="h1", opener=ALICE, channel=CH, rules=None):
        if rules:
            self.store.set_game_rules(G, "holdem", rules)
        return self.store.open_table(G, channel, "holdem", opener, seed=seed)

    def wallet(self, user, amount=10000):
        """Tops the member's wallet up to `amount`."""
        have = self.store.balance(G, user)
        if have < amount:
            self.fund(user, amount - have)

    def set_balance(self, user, amount):
        """Writes the balance directly: change_balance caps one change at 1,000,000, below the balance cap."""
        self.store.db.execute(
            "INSERT INTO currency_balances(guild_id,user_id,balance,updated_at) VALUES(?,?,?,0) "
            "ON CONFLICT(guild_id,user_id) DO UPDATE SET balance=excluded.balance", (G, user, amount))
        self.store.db.commit()

    def cwallet(self, cid, amount):
        self.store.change_character_balance(G, cid, amount, "start", 1)

    def cbal(self, cid):
        return self.store.character_balance(G, cid)

    def sit(self, rid, users, buy_in=1000):
        for u in users:
            self.wallet(u)
            self.store.join_round(G, rid, u, buy_in)

    def lobby(self, users=(ALICE, BOB, CARL), seed="h1", buy_in=1000, rules=None):
        snap = self.open(seed, rules=rules)
        self.sit(snap["round_id"], users, buy_in)
        return snap["round_id"]

    def seat_rows(self, rid):
        return [dict(r) for r in self.store.db.execute("SELECT * FROM game_seats WHERE round_id=? ORDER BY seat_index", (rid,))]

    def ledger_count(self):
        return self.store.db.execute(LEDGER_SQL, (G,)).fetchone()[0]

    def reasons(self):
        return [r["reason"] for r in self.game_ledger()]

    def cashouts(self, table_id=1):
        return [dict(r) for r in self.store.db.execute("SELECT * FROM game_cashouts WHERE guild_id=? AND table_id=? ORDER BY id", (G, table_id))]

    def snap(self, rid):
        return self.store.round_snapshot(G, rid)

    def hand(self, rid, chooser):
        """Plays the hand to its end with chooser(menu, snapshot) -> move; only the store API is used."""
        for _ in range(500):
            s = self.snap(rid)
            if s["status"] != "playing":
                return s
            turn = s["view"]["turn"]
            menu = holdem.legal_moves_for_view(self.store.seat_view(G, rid, turn))
            actor = "character" if s["seats"][turn]["kind"] == "character" else "member"
            self.store.play(G, rid, turn, chooser(menu, s), actor, s["moves"])
        raise AssertionError("hand did not finish")

    @staticmethod
    def call_down(menu, s):
        return "check" if "check" in menu else "call"

    @staticmethod
    def shove(menu, s):
        raises = [m for m in menu if m.startswith("raise:")]
        return raises[-1] if raises else "call" if "call" in menu else "check"

    @staticmethod
    def fold_out(menu, s):
        return "fold" if "fold" in menu else "check"

    def total_money(self, users, chars):
        """Wallets plus the stacks sitting at open tables (a settled newest hand counts its end stacks)."""
        total = sum(self.store.balance(G, u) for u in users) + sum(self.cbal(c) for c in chars)
        for (stack,) in self.store.db.execute(
                "SELECT COALESCE(SUM(CASE r.status WHEN 'settled' THEN s.payout ELSE s.stack END),0) FROM game_tables t "
                "JOIN game_rounds r ON r.table_id=t.id AND r.number=(SELECT MAX(number) FROM game_rounds WHERE table_id=t.id) "
                "JOIN game_seats s ON s.round_id=r.id WHERE t.status='open' AND t.guild_id=?", (G,)):
            total += stack
        return total


class RegistryTests(unittest.TestCase):
    def test_holdem_entry(self):
        h = registry.get("holdem")
        self.assertIn("holdem", registry.keys())
        self.assertEqual((h.min_seats, h.max_seats, h.ledger_label), (2, 9, "Texas Hold'em"))
        self.assertIs(h.engine, holdem)
        self.assertIs(h.normalize_rules, holdem.normalize_rules)
        self.assertTrue(h.cash)

    def test_other_games_unchanged(self):
        b, d = registry.get("blackjack"), registry.get("doubt")
        self.assertEqual((b.min_seats, b.max_seats, b.ledger_label, b.ante), (1, 7, "Blackjack", False))
        self.assertEqual((d.min_seats, d.max_seats, d.ledger_label, d.ante), (3, 8, "I Doubt It", True))
        self.assertFalse(getattr(b, "cash", False))
        self.assertFalse(getattr(d, "cash", False))
        self.assertLessEqual({"blackjack", "doubt", "holdem"}, set(registry.keys()))


class BuyInTests(HoldemCase):
    def test_buy_in_debits_wallet_and_sets_stack_and_stake(self):
        snap = self.open()
        self.wallet(ALICE, 5000)
        out = self.store.join_round(G, snap["round_id"], ALICE, 1000)
        self.assertEqual(self.store.balance(G, ALICE), 4000)
        row = self.seat_rows(snap["round_id"])[0]
        self.assertEqual((row["kind"], row["ref_id"], row["stake"], row["stack"]), ("member", ALICE, 1000, 1000))
        self.assertEqual(self.reasons(), [BUY])
        entry = self.store.ledger(G, ALICE)[0]
        self.assertEqual((entry["amount"], entry["source"], entry["reason"]), (-1000, "game", BUY))
        self.assertEqual((out["seats"][0]["stake"], out["seats"][0]["stack"]), (1000, 1000))

    def test_first_lobby_has_no_assist_and_pins_full_rules(self):
        snap = self.open(rules={"assist": True, "small_blind": 2})
        self.assertEqual((snap["game"], snap["status"], snap["rules"]["small_blind"], snap["rules"]["assist"]), ("holdem", "joining", 2, False))
        self.assertEqual(snap["rules"]["min_buy_in_bb"], 40)

    def test_range_is_checked_at_both_edges(self):
        snap = self.open()
        self.wallet(ALICE, 5000)
        before = self.ledger_count()
        for bad in (399, 2001, 0, -5, True, "1000", 10.5):
            with self.assertRaises(GameError, msg=repr(bad)):
                self.store.join_round(G, snap["round_id"], ALICE, bad)
        self.assertEqual((self.ledger_count(), self.seat_rows(snap["round_id"])), (before, []))
        self.store.join_round(G, snap["round_id"], ALICE, 400)
        self.wallet(BOB, 5000)
        self.store.join_round(G, snap["round_id"], BOB, 2000)
        self.assertEqual([r["stack"] for r in self.seat_rows(snap["round_id"])], [400, 2000])

    def test_range_follows_the_pinned_rules(self):
        snap = self.open(rules={"small_blind": 1, "min_buy_in_bb": 20, "max_buy_in_bb": 50})  # bb 2: 40..100
        self.wallet(ALICE, 500)
        for bad in (39, 101):
            with self.assertRaises(GameError):
                self.store.join_round(G, snap["round_id"], ALICE, bad)
        self.store.join_round(G, snap["round_id"], ALICE, 40)
        self.store.set_game_rules(G, "holdem", {"small_blind": 5})  # the lobby keeps the rules it opened with
        self.wallet(BOB, 500)
        self.store.join_round(G, snap["round_id"], BOB, 100)
        self.assertEqual([r["stack"] for r in self.seat_rows(snap["round_id"])], [40, 100])

    def test_insufficient_wallet_writes_nothing(self):
        snap = self.open()
        self.fund(ALICE, 399)
        with self.assertRaises(GameError):
            self.store.join_round(G, snap["round_id"], ALICE, 400)
        self.assertEqual((self.store.balance(G, ALICE), self.ledger_count(), self.seat_rows(snap["round_id"])), (399, 1, []))

    def test_second_seat_for_the_same_member_is_refused(self):
        rid = self.lobby((ALICE,))
        with self.assertRaisesRegex(GameError, "already have a seat"):
            self.store.join_round(G, rid, ALICE, 500)
        self.assertEqual((len(self.seat_rows(rid)), self.store.balance(G, ALICE)), (1, 9000))

    def test_table_is_full_at_nine(self):
        snap = self.open()
        rid = snap["round_id"]
        self.sit(rid, range(20, 29), 400)
        self.assertEqual(len(self.seat_rows(rid)), 9)
        self.wallet(40)
        with self.assertRaisesRegex(GameError, "full"):
            self.store.join_round(G, rid, 40, 400)
        self.assertEqual(self.store.balance(G, 40), 10000)

    def test_joining_only(self):
        rid = self.lobby((ALICE, BOB))
        self.store.deal(G, rid)
        self.wallet(CARL)
        with self.assertRaisesRegex(GameError, "This round has already started."):
            self.store.join_round(G, rid, CARL, 500)
        self.assertEqual(self.store.balance(G, CARL), 10000)

    def test_guild_isolation(self):
        snap = self.open()
        self.fund(ALICE, 5000, guild=G2)
        with self.assertRaises(GameError):
            self.store.join_round(G2, snap["round_id"], ALICE, 500)
        self.assertEqual((self.store.balance(G2, ALICE), self.seat_rows(snap["round_id"])), (5000, []))
        with self.assertRaises(GameError):
            self.store.deal(G2, snap["round_id"])
        with self.assertRaises(GameError):
            self.store.next_round(G2, snap["table_id"])
        with self.assertRaises(GameError):
            self.store.close_table(G2, snap["table_id"])

    def test_buy_in_ledger_is_per_guild(self):
        rid = self.lobby((ALICE,))
        self.assertEqual(self.store.db.execute(LEDGER_SQL, (G2,)).fetchone()[0], 0)
        self.assertEqual(len(self.store.ledger(G, ALICE)), 2)  # the start grant and the buy-in
        self.assertTrue(rid)


class CharacterSeatTests(HoldemCase):
    def test_favorites_buy_in_from_their_own_wallets(self):
        s = self.store
        for c in (self.ann, self.bo, self.cy):
            s.add_favorite(G, ALICE, c)
        self.cwallet(self.ann, 5000)
        self.cwallet(self.bo, 600)
        self.cwallet(self.cy, 300)  # below the 400 minimum: broke
        snap = self.open()
        self.wallet(ALICE)
        out = s.join_with_favorites(G, snap["round_id"], ALICE, 1000, self.w)
        self.assertEqual(out["seated"], [{"character_id": self.ann, "name": "Ann", "stake": 1000}, {"character_id": self.bo, "name": "Bo", "stake": 600}])
        self.assertEqual(out["skipped"], [{"character_id": self.cy, "name": "Cy", "reason": "broke"}])
        rows = self.seat_rows(snap["round_id"])
        self.assertEqual([(r["kind"], r["stake"], r["stack"], r["brought_by"]) for r in rows],
                         [("member", 1000, 1000, None), ("character", 1000, 1000, ALICE), ("character", 600, 600, ALICE)])
        self.assertEqual((s.balance(G, ALICE), self.cbal(self.ann), self.cbal(self.bo), self.cbal(self.cy)), (9000, 4000, 0, 300))
        entry = s.ledger(G, character_id=self.ann)[0]
        self.assertEqual((entry["amount"], entry["source"], entry["reason"]), (-1000, "game", BUY))

    def test_favorites_member_buy_in_is_still_checked_and_writes_nothing(self):
        self.store.add_favorite(G, ALICE, self.ann)
        self.cwallet(self.ann, 5000)
        snap = self.open()
        self.wallet(ALICE)
        before = self.ledger_count()
        with self.assertRaises(GameError):
            self.store.join_with_favorites(G, snap["round_id"], ALICE, 100, self.w)
        self.assertEqual((self.ledger_count(), self.cbal(self.ann), self.seat_rows(snap["round_id"])), (before, 5000, []))

    def test_favorites_stop_at_nine_seats(self):
        s = self.store
        s.set_cast_limits(G, 12, 12)  # the default favorites limit (5) would stop the loop before the table does
        extra = [s.create_character(G, self.w, f"X{i}") for i in range(8)]
        for c in extra:
            s.add_favorite(G, ALICE, c)
            self.cwallet(c, 5000)
        snap = self.open()
        self.wallet(ALICE)
        s.add_favorite(G, ALICE, self.ann)
        self.cwallet(self.ann, 5000)
        out = s.join_with_favorites(G, snap["round_id"], ALICE, 500, self.w)
        self.assertEqual(len(self.seat_rows(snap["round_id"])), 9)
        self.assertEqual([x["reason"] for x in out["skipped"]], ["full"])

    def fill_rules(self):
        return {"small_blind": 1, "min_buy_in_bb": 20, "max_buy_in_bb": 50}  # bb 2: buy-in 40..100, 100 big blinds = 200

    def test_fill_seats_house_characters_buy_in_100_big_blinds_capped(self):
        self.cwallet(self.ann, 5000)  # 100 bb = 200, capped at the max buy-in 100
        self.cwallet(self.bo, 70)  # the wallet is the limit
        self.cwallet(self.cy, 30)  # below the 40 minimum: broke
        snap = self.open(rules=self.fill_rules())
        out = self.store.fill_seats(G, snap["round_id"], ALICE, [self.ann, self.bo, self.cy], self.w)
        self.assertEqual([(x["character_id"], x["stake"]) for x in out["seated"]], [(self.ann, 100), (self.bo, 70)])
        self.assertEqual(out["skipped"], [{"character_id": self.cy, "name": "Cy", "reason": "broke"}])
        rows = self.seat_rows(snap["round_id"])
        self.assertEqual([(r["kind"], r["stake"], r["stack"], r["brought_by"]) for r in rows], [("character", 100, 100, None), ("character", 70, 70, None)])
        self.assertEqual((self.cbal(self.ann), self.cbal(self.bo), self.cbal(self.cy)), (4900, 0, 30))
        self.assertEqual(self.store.ledger(G, character_id=self.ann)[0]["reason"], BUY)

    def test_fill_seats_default_rules_take_100_big_blinds(self):
        self.cwallet(self.ann, 5000)
        snap = self.open()
        self.store.fill_seats(G, snap["round_id"], ALICE, [self.ann], self.w)
        self.assertEqual(self.seat_rows(snap["round_id"])[0]["stack"], 1000)

    def test_fill_seats_is_host_only_and_joining_only(self):
        self.cwallet(self.ann, 5000)
        snap = self.open()
        with self.assertRaisesRegex(GameError, "opened the table"):
            self.store.fill_seats(G, snap["round_id"], BOB, [self.ann], self.w)
        self.assertEqual((self.seat_rows(snap["round_id"]), self.cbal(self.ann)), ([], 5000))
        self.sit(snap["round_id"], (ALICE, BOB))
        self.store.deal(G, snap["round_id"])
        with self.assertRaisesRegex(GameError, "already started"):
            self.store.fill_seats(G, snap["round_id"], ALICE, [self.ann], self.w)

    def test_fill_seats_guild_isolation(self):
        self.cwallet(self.ann, 5000)
        snap = self.open()
        with self.assertRaises(GameError):
            self.store.fill_seats(G2, snap["round_id"], ALICE, [self.ann], self.w)
        self.assertEqual(self.cbal(self.ann), 5000)

    def test_fill_seats_skips_the_seated_and_the_full(self):
        snap = self.open()
        self.cwallet(self.ann, 5000)
        self.cwallet(self.bo, 5000)
        self.store.fill_seats(G, snap["round_id"], ALICE, [self.ann], self.w)
        out = self.store.fill_seats(G, snap["round_id"], ALICE, [self.ann, self.bo], self.w)
        self.assertEqual(([x["character_id"] for x in out["seated"]], [x["reason"] for x in out["skipped"]]), ([self.bo], ["seated"]))


class RebuyTests(HoldemCase):
    def test_rebuy_grows_stack_and_stake_and_logs_the_ledger(self):
        rid = self.lobby((ALICE, BOB))
        self.store.rebuy(G, rid, ALICE, 500)
        row = self.seat_rows(rid)[0]
        self.assertEqual((row["stack"], row["stake"], self.store.balance(G, ALICE)), (1500, 1500, 8500))
        entry = self.store.ledger(G, ALICE)[0]
        self.assertEqual((entry["amount"], entry["source"], entry["reason"]), (-500, "game", REBUY))
        seat = self.snap(rid)["seats"][0]
        self.assertEqual((seat["stack"], seat["stake"]), (1500, 1500))

    def test_rebuy_is_capped_at_the_max_buy_in(self):
        rid = self.lobby((ALICE, BOB))
        before = self.ledger_count()
        with self.assertRaises(GameError):
            self.store.rebuy(G, rid, ALICE, 1001)
        self.assertEqual((self.ledger_count(), self.seat_rows(rid)[0]["stack"]), (before, 1000))
        self.store.rebuy(G, rid, ALICE, 1000)
        self.assertEqual(self.seat_rows(rid)[0]["stack"], 2000)
        with self.assertRaises(GameError):
            self.store.rebuy(G, rid, ALICE, 1)

    def test_rebuy_amount_and_wallet_are_checked(self):
        rid = self.lobby((ALICE, BOB), buy_in=500)
        for bad in (0, -10, True, "5", 2.5):
            with self.assertRaises(GameError, msg=repr(bad)):
                self.store.rebuy(G, rid, ALICE, bad)
        self.store.change_balance(G, ALICE, -(self.store.balance(G, ALICE) - 100), "drain", 1)
        with self.assertRaises(GameError):
            self.store.rebuy(G, rid, ALICE, 200)
        self.assertEqual((self.seat_rows(rid)[0]["stack"], self.store.balance(G, ALICE)), (500, 100))

    def test_rebuy_needs_a_seat_and_joining_and_the_right_guild(self):
        rid = self.lobby((ALICE, BOB))
        self.wallet(CARL)
        with self.assertRaises(GameError):
            self.store.rebuy(G, rid, CARL, 100)
        with self.assertRaises(GameError):
            self.store.rebuy(G2, rid, ALICE, 100)
        self.store.deal(G, rid)
        with self.assertRaisesRegex(GameError, "This round has already started."):
            self.store.rebuy(G, rid, ALICE, 100)
        self.assertEqual(self.store.balance(G, ALICE), 9000)

    def test_characters_never_rebuy(self):
        self.cwallet(self.ann, 5000)
        snap = self.open()
        self.store.fill_seats(G, snap["round_id"], ALICE, [self.ann], self.w)
        self.sit(snap["round_id"], (BOB,))
        self.wallet(self.ann, 5000)  # a member whose id equals the character's id has no seat of their own
        with self.assertRaises(GameError):
            self.store.rebuy(G, snap["round_id"], self.ann, 100)
        self.assertEqual((self.seat_rows(snap["round_id"])[0]["stack"], self.cbal(self.ann)), (1000, 4000))


class LeaveTests(HoldemCase):
    def test_leave_cashes_out_member_and_brought_characters(self):
        s = self.store
        s.add_favorite(G, ALICE, self.ann)
        self.cwallet(self.ann, 3000)
        snap = self.open()
        rid = snap["round_id"]
        self.wallet(ALICE)
        s.join_with_favorites(G, rid, ALICE, 1000, self.w)
        self.sit(rid, (BOB,), 700)
        s.leave_round(G, rid, ALICE)
        self.assertEqual((s.balance(G, ALICE), self.cbal(self.ann)), (10000, 3000))
        self.assertEqual(s.ledger(G, ALICE)[0]["reason"], CASH)
        self.assertEqual(s.ledger(G, character_id=self.ann)[0]["reason"], CASH)
        rows = self.cashouts()
        self.assertEqual([(r["kind"], r["ref_id"], r["bought_in"], r["cashed_out"]) for r in rows], [("member", ALICE, 1000, 1000), ("character", self.ann, 1000, 1000)])
        self.assertEqual(set(rows[0]), {"id", "guild_id", "table_id", "kind", "ref_id", "bought_in", "cashed_out", "created_at"})
        self.assertEqual([(r["seat_index"], r["ref_id"]) for r in self.seat_rows(rid)], [(0, BOB)])

    def test_leave_renumbers_seats(self):
        rid = self.lobby((ALICE, BOB, CARL, DAN))
        self.store.leave_round(G, rid, BOB)
        self.assertEqual([(r["seat_index"], r["ref_id"]) for r in self.seat_rows(rid)], [(0, ALICE), (1, CARL), (2, DAN)])

    def test_leave_needs_a_seat_joining_and_the_right_guild(self):
        rid = self.lobby((ALICE, BOB))
        with self.assertRaises(GameError):
            self.store.leave_round(G, rid, CARL)
        with self.assertRaises(GameError):
            self.store.leave_round(G2, rid, ALICE)
        self.store.deal(G, rid)
        before = self.ledger_count()
        with self.assertRaisesRegex(GameError, "This round has already started."):
            self.store.leave_round(G, rid, ALICE)
        self.assertEqual((self.ledger_count(), self.cashouts()), (before, []))

    def test_leave_after_a_hand_pays_the_current_stack(self):
        rid = self.lobby((ALICE, BOB))
        self.store.deal(G, rid)
        end = self.hand(rid, self.shove)
        nxt = self.store.next_round(G, 1)["round_id"]
        stacks = {r["ref_id"]: r["stack"] for r in self.seat_rows(nxt)}
        self.assertEqual(sum(stacks.values()), 2000)
        for user in tuple(stacks):
            self.store.leave_round(G, nxt, user)
        self.assertEqual({u: self.store.balance(G, u) for u in (ALICE, BOB)}, {u: 9000 + stacks.get(u, 0) for u in (ALICE, BOB)})
        self.assertEqual(end["status"], "settled")
        paid = {r["ref_id"]: (r["bought_in"], r["cashed_out"]) for r in self.cashouts()}
        for user in (ALICE, BOB):
            if user in paid:
                self.assertEqual(paid[user], (1000, stacks.get(user, 0)))

    def test_cash_out_is_clamped_at_the_balance_cap(self):
        rid = self.lobby((ALICE, BOB))
        cap = self.store.MAX_CURRENCY_BALANCE
        self.set_balance(ALICE, cap - 600)
        self.store.leave_round(G, rid, ALICE)
        self.assertEqual(self.store.balance(G, ALICE), cap)
        row = self.cashouts()[0]
        self.assertEqual((row["bought_in"], row["cashed_out"]), (1000, 600))
        self.assertEqual(self.store.ledger(G, ALICE)[0]["amount"], 600)


class AssistTests(HoldemCase):
    def allowed(self):
        return {"assist": True}

    def test_host_turns_assist_on_and_off(self):
        snap = self.open(rules=self.allowed())
        rid = snap["round_id"]
        self.assertFalse(snap["rules"]["assist"])
        self.store.set_table_assist(G, rid, ALICE, True)
        self.assertTrue(self.snap(rid)["rules"]["assist"])
        self.store.set_table_assist(G, rid, ALICE, False)
        self.assertFalse(self.snap(rid)["rules"]["assist"])

    def test_only_the_host_and_only_in_the_lobby(self):
        rid = self.lobby((ALICE, BOB), rules=self.allowed())
        with self.assertRaises(GameError):
            self.store.set_table_assist(G, rid, BOB, True)
        with self.assertRaises(GameError):
            self.store.set_table_assist(G2, rid, ALICE, True)
        self.assertFalse(self.snap(rid)["rules"]["assist"])
        self.store.deal(G, rid)
        with self.assertRaisesRegex(GameError, "This round has already started."):
            self.store.set_table_assist(G, rid, ALICE, True)

    def test_refused_when_the_server_does_not_allow_it(self):
        snap = self.open()
        with self.assertRaises(GameError):
            self.store.set_table_assist(G, snap["round_id"], ALICE, True)
        self.assertFalse(self.snap(snap["round_id"])["rules"]["assist"])

    def test_refused_when_the_server_turns_it_off_after_opening(self):
        snap = self.open(rules=self.allowed())
        self.store.set_game_rules(G, "holdem", {"assist": False})
        with self.assertRaises(GameError):
            self.store.set_table_assist(G, snap["round_id"], ALICE, True)

    def test_assist_carries_to_the_next_hand_while_allowed(self):
        rid = self.lobby((ALICE, BOB), rules=self.allowed())
        self.store.set_table_assist(G, rid, ALICE, True)
        self.store.deal(G, rid)
        self.hand(rid, self.call_down)
        nxt = self.store.next_round(G, 1)
        self.assertTrue(nxt["rules"]["assist"])
        self.store.deal(G, nxt["round_id"])
        self.hand(nxt["round_id"], self.fold_out)
        self.store.set_game_rules(G, "holdem", {"assist": False})
        self.assertFalse(self.store.next_round(G, 1)["rules"]["assist"])


class DealTests(HoldemCase):
    def test_deal_sets_button_from_the_seed_and_starts_the_engine(self):
        rid = self.lobby((ALICE, BOB, CARL), seed="deal-seed")
        before = self.ledger_count()
        snap = self.store.deal(G, rid)
        button = int(seed_hash("deal-seed")[:8], 16) % 3
        self.assertEqual((snap["status"], snap["rules"]["button"], snap["moves"]), ("playing", button, 0))
        self.assertEqual(self.store.db.execute("SELECT button FROM game_rounds WHERE id=?", (rid,)).fetchone()[0], button)
        engine = holdem.new("deal-seed", [Seat(i, "member", u, 1000) for i, u in enumerate((ALICE, BOB, CARL))], snap["rules"], button)
        pub = holdem.view(engine)
        self.assertEqual((snap["view"]["stacks"], snap["view"]["bets"], snap["view"]["pot"], snap["view"]["turn"]), (pub.stacks, pub.bets, pub.pot, pub.turn))
        self.assertEqual(snap["view"]["pot"], 15)
        self.assertEqual(self.ledger_count(), before)  # blinds live in the engine only
        self.assertEqual(self.store.balance(G, ALICE), 9000)
        self.assertEqual([r["stack"] for r in self.seat_rows(rid)], [1000, 1000, 1000])

    def test_deal_needs_two_seats_with_chips(self):
        snap = self.open()
        rid = snap["round_id"]
        with self.assertRaises(GameError):
            self.store.deal(G, rid)
        self.sit(rid, (ALICE,))
        with self.assertRaises(GameError):
            self.store.deal(G, rid)
        self.sit(rid, (BOB,))
        self.store.db.execute("UPDATE game_seats SET stack=0 WHERE round_id=? AND seat_index=1", (rid,))
        self.store.db.commit()
        with self.assertRaises(GameError):
            self.store.deal(G, rid)
        self.assertEqual(self.snap(rid)["status"], "joining")

    def test_deal_is_once_and_guild_scoped(self):
        rid = self.lobby((ALICE, BOB))
        with self.assertRaises(GameError):
            self.store.deal(G2, rid)
        self.store.deal(G, rid)
        with self.assertRaisesRegex(GameError, "This round has already started."):
            self.store.deal(G, rid)

    def test_button_rotates_to_the_next_seat_each_hand(self):
        rid = self.lobby((ALICE, BOB, CARL), seed="rot")
        first = self.store.deal(G, rid)["rules"]["button"]
        self.hand(rid, self.fold_out)
        buttons = [first]
        for _ in range(3):
            nxt = self.store.next_round(G, 1)["round_id"]
            buttons.append(self.store.deal(G, nxt)["rules"]["button"])
            self.hand(nxt, self.fold_out)
        self.assertEqual(buttons, [(first + k) % 3 for k in range(4)])

    def test_button_survives_seat_changes(self):
        users = [ALICE, BOB, CARL, DAN]
        rid = self.lobby(users, seed="rot2")
        first = self.store.deal(G, rid)["rules"]["button"]
        holder = users[first]
        self.hand(rid, self.fold_out)
        nxt = self.store.next_round(G, 1)["round_id"]
        self.store.leave_round(G, nxt, holder)  # the button holder leaves between hands
        self.sit(nxt, (ERIN,))
        left = [u for u in users if u != holder] + [ERIN]
        expected_user = users[(first + 1) % 4]  # the next seat clockwise who is still seated
        snap = self.store.deal(G, nxt)
        self.assertEqual(left[snap["rules"]["button"]], expected_user)

    def test_a_hand_finished_at_the_deal_settles_at_once(self):
        rid = self.lobby((ALICE, BOB))
        self.store.db.execute("UPDATE game_seats SET stack=5 WHERE round_id=? AND seat_index=0", (rid,))
        self.store.db.execute("UPDATE game_seats SET stack=10 WHERE round_id=? AND seat_index=1", (rid,))
        self.store.db.commit()
        before = self.ledger_count()
        snap = self.store.deal(G, rid)
        self.assertEqual(snap["status"], "settled")
        self.assertIsNotNone(snap["seed"])
        self.assertEqual(sum(r["payout"] for r in self.seat_rows(rid)), 15)
        self.assertEqual(self.ledger_count(), before)


class PlayTests(HoldemCase):
    def dealt(self, users=(ALICE, BOB, CARL), seed="play"):
        rid = self.lobby(users, seed=seed)
        return rid, self.store.deal(G, rid)

    def test_a_legal_move_is_applied_and_logged(self):
        rid, snap = self.dealt()
        turn = snap["view"]["turn"]
        self.assertIn("call", snap["legal"][turn])
        after = self.store.play(G, rid, turn, "call", "member", 0)
        self.assertEqual(after["moves"], 1)
        row = self.store.db.execute("SELECT seat_index,move,actor FROM game_moves WHERE round_id=?", (rid,)).fetchone()
        self.assertEqual(tuple(row), (turn, "call", "member"))
        self.assertNotEqual(after["view"]["turn"], turn)

    def test_a_stale_move_count_conflicts_and_applies_nothing(self):
        rid, snap = self.dealt()
        turn = snap["view"]["turn"]
        self.store.play(G, rid, turn, "call", "member", 0)
        with self.assertRaises(ConflictError):
            self.store.play(G, rid, turn, "call", "member", 0)
        self.assertEqual(self.snap(rid)["moves"], 1)

    def test_illegal_moves_are_game_errors(self):
        rid, snap = self.dealt()
        turn = snap["view"]["turn"]
        other = (turn + 1) % 3
        for seat, move in ((turn, "check"), (turn, "raise:1"), (turn, "jump"), (other, "call"), (7, "call"), (-1, "call")):
            with self.assertRaises(GameError, msg=(seat, move)):
                self.store.play(G, rid, seat, move, "member", 0)
        with self.assertRaises(GameError):
            self.store.play(G, rid, turn, "call", "robot", 0)
        self.assertEqual(self.snap(rid)["moves"], 0)

    def test_play_needs_a_hand_in_progress(self):
        rid = self.lobby((ALICE, BOB))
        with self.assertRaises(GameError):
            self.store.play(G, rid, 0, "call", "member", 0)
        self.store.deal(G, rid)
        self.hand(rid, self.fold_out)
        with self.assertRaises(GameError):
            self.store.play(G, rid, 0, "call", "member", self.snap(rid)["moves"])

    def test_actor_rules_for_character_and_member_seats(self):
        self.cwallet(self.ann, 5000)
        snap = self.open(seed="actors")
        rid = snap["round_id"]
        self.sit(rid, (ALICE,))
        self.store.fill_seats(G, rid, ALICE, [self.ann], self.w)
        dealt = self.store.deal(G, rid)
        turn = dealt["view"]["turn"]
        wrong = "character" if dealt["seats"][turn]["kind"] == "member" else "member"
        with self.assertRaisesRegex(GameError, "cannot move here"):
            self.store.play(G, rid, turn, "call", wrong, 0)
        self.assertEqual(self.snap(rid)["moves"], 0)
        right = "character" if wrong == "member" else "member"
        self.assertEqual(self.store.play(G, rid, turn, "call", right, 0)["moves"], 1)

    def test_timeout_may_move_any_seat(self):
        self.cwallet(self.ann, 5000)
        snap = self.open(seed="timeouts")
        self.sit(snap["round_id"], (ALICE,))
        self.store.fill_seats(G, snap["round_id"], ALICE, [self.ann], self.w)
        turn = self.store.deal(G, snap["round_id"])["view"]["turn"]
        self.assertEqual(self.store.play(G, snap["round_id"], turn, "fold", "timeout", 0)["moves"], 1)

    def test_settle_moves_chips_between_stacks_only(self):
        rid, _ = self.dealt()
        users = (ALICE, BOB, CARL)
        wallets = {u: self.store.balance(G, u) for u in users}
        before = self.ledger_count()
        done = self.hand(rid, self.call_down)
        self.assertEqual(done["status"], "settled")
        rows = self.seat_rows(rid)
        self.assertEqual(sum(r["payout"] for r in rows), 3000)  # chips are conserved across the seats
        self.assertEqual([r["stack"] for r in rows], [1000, 1000, 1000])
        self.assertTrue(all(r["outcome"] in ("win", "split", "lose", "fold") for r in rows))
        self.assertEqual(self.ledger_count(), before)
        self.assertEqual({u: self.store.balance(G, u) for u in users}, wallets)
        rnd = self.store.db.execute("SELECT * FROM game_rounds WHERE id=?", (rid,)).fetchone()
        moves = [(r["seat_index"], r["move"]) for r in self.store.db.execute("SELECT * FROM game_moves WHERE round_id=? ORDER BY id", (rid,))]
        res = holdem.result(holdem.replay(rnd["seed"], [Seat(i, "member", u, 1000) for i, u in enumerate(users)], moves, done["rules"], rnd["button"]))
        self.assertEqual([r["payout"] for r in rows], [x.end for x in res.seats])
        self.assertEqual([r["outcome"] for r in rows], [x.outcome for x in res.seats])
        self.assertEqual([(x["stack"], x["end"], x["busted"], x["stake"]) for x in done["seats"]], [(1000, r["payout"], r["payout"] == 0, 1000) for r in rows])

    def test_a_fold_around_hand_pays_the_blinds_to_the_big_blind(self):
        rid, snap = self.dealt()
        done = self.hand(rid, self.fold_out)
        ends = sorted(x["end"] for x in done["seats"])
        self.assertEqual(ends, [995, 1000, 1005])
        self.assertEqual(sum(ends), 3000)
        self.assertEqual(sorted(x["outcome"] for x in done["seats"]).count("fold"), 2)
        self.assertEqual(done["seed"], self.store.db.execute("SELECT seed FROM game_rounds WHERE id=?", (rid,)).fetchone()[0])
        self.assertIsNone(snap["seed"])

    def test_the_snapshot_never_holds_a_hole_card(self):
        rid, snap = self.dealt()
        holes = [self.store.seat_view(G, rid, i).hole for i in range(3)]
        self.assertTrue(all(holes))
        seen = [snap]
        for _ in range(60):
            cur = self.snap(rid)
            seen.append(cur)
            if cur["status"] != "playing":
                break
            turn = cur["view"]["turn"]
            menu = holdem.legal_moves_for_view(self.store.seat_view(G, rid, turn))
            self.store.play(G, rid, turn, self.call_down(menu, cur), "member", cur["moves"])
        seen.append(self.store.latest_round(G, 1))
        for shot in seen:
            text = repr(shot)
            for pair in holes:
                for held in pair:
                    self.assertNotIn(repr(held), text)
        self.assertTrue(all(s["seed"] is None for s in seen[:-3] if s["status"] == "playing"))
        self.assertIsNotNone(seen[-1]["seed"])

    def test_the_public_menu_is_for_the_player_on_turn(self):
        rid, snap = self.dealt()
        turn = snap["view"]["turn"]
        self.assertEqual(set(snap["legal"]), {turn})
        self.assertEqual(snap["legal"][turn], holdem.legal_moves_for_view(self.store.seat_view(G, rid, turn)))


class NextRoundTests(HoldemCase):
    def test_next_lobby_keeps_the_opening_rules_and_new_tables_get_the_new_ones(self):
        rid = self.lobby((ALICE, BOB), buy_in=100, rules={"small_blind": 2, "min_buy_in_bb": 20, "max_buy_in_bb": 100, "action_timeout": 60})
        self.store.deal(G, rid)
        self.hand(rid, self.fold_out)
        self.store.set_game_rules(G, "holdem", {"small_blind": 9, "min_buy_in_bb": 30, "max_buy_in_bb": 90, "action_timeout": 20})
        nxt = self.store.next_round(G, 1)
        keep = {k: nxt["rules"][k] for k in ("small_blind", "min_buy_in_bb", "max_buy_in_bb", "action_timeout")}
        self.assertEqual(keep, {"small_blind": 2, "min_buy_in_bb": 20, "max_buy_in_bb": 100, "action_timeout": 60})
        self.store.close_table(G, 1)
        fresh = self.open()
        self.assertEqual((fresh["rules"]["small_blind"], fresh["rules"]["action_timeout"]), (9, 20))

    def test_rebuy_bounds_use_the_pinned_rules_after_a_server_change(self):
        rid = self.lobby((ALICE, BOB), buy_in=50, rules={"small_blind": 1, "min_buy_in_bb": 20, "max_buy_in_bb": 50})  # bb 2: 40..100
        self.store.deal(G, rid)
        self.hand(rid, self.fold_out)
        self.store.set_game_rules(G, "holdem", {"small_blind": 50})  # a server max of 5000 would allow the rebuy below
        nxt = self.store.next_round(G, 1)["round_id"]
        stack = next(r["stack"] for r in self.seat_rows(nxt) if r["ref_id"] == ALICE)
        with self.assertRaises(GameError):
            self.store.rebuy(G, nxt, ALICE, 100 - stack + 1)
        self.store.rebuy(G, nxt, ALICE, 100 - stack)
        self.assertEqual(next(r["stack"] for r in self.seat_rows(nxt) if r["ref_id"] == ALICE), 100)

    def test_next_round_copies_seats_with_end_stacks(self):
        self.cwallet(self.ann, 5000)
        self.store.add_favorite(G, ALICE, self.ann)
        snap = self.open(seed="nr")
        self.wallet(ALICE)
        self.store.join_with_favorites(G, snap["round_id"], ALICE, 1000, self.w)
        self.sit(snap["round_id"], (BOB,), 1500)
        rid = snap["round_id"]
        self.store.deal(G, rid)
        done = self.hand(rid, self.call_down)
        ends = [x["end"] for x in done["seats"]]
        if 0 in ends:
            self.skipTest("a seat busted on this seed")
        nxt = self.store.next_round(G, 1)
        self.assertEqual((nxt["status"], nxt["number"]), ("joining", 2))
        self.assertNotEqual(nxt["seed_hash"], done["seed_hash"])
        old = self.seat_rows(rid)
        new = self.seat_rows(nxt["round_id"])
        self.assertEqual([(r["kind"], r["ref_id"], r["stake"], r["brought_by"]) for r in new], [(r["kind"], r["ref_id"], r["stake"], r["brought_by"]) for r in old])
        self.assertEqual([r["stack"] for r in new], [r["payout"] for r in old])
        self.assertTrue(all(r["outcome"] is None and r["payout"] is None for r in new))
        self.assertEqual(self.cashouts(), [])

    def test_busted_seats_are_dropped_and_logged_with_a_zero_cash_out(self):
        for n in range(30):
            rid = self.lobby((ALICE, BOB), seed=f"bust-{n}")
            self.store.deal(G, rid)
            done = self.hand(rid, self.shove)
            if any(x["busted"] for x in done["seats"]):
                break
            self.store.close_table(G, n + 1)
        else:
            self.fail("no seed produced a bust")
        tid = n + 1
        wallets = {u: self.store.balance(G, u) for u in (ALICE, BOB)}
        busted = [x for x in done["seats"] if x["busted"]]
        self.assertEqual(len(busted), 1)
        before = self.ledger_count()
        nxt = self.store.next_round(G, tid)
        rows = self.seat_rows(nxt["round_id"])
        survivor = [x for x in done["seats"] if not x["busted"]][0]
        self.assertEqual([(r["ref_id"], r["stack"], r["stake"]) for r in rows], [(survivor["ref_id"], 2000, 1000)])
        out = self.cashouts(tid)
        self.assertEqual([(r["kind"], r["ref_id"], r["bought_in"], r["cashed_out"]) for r in out], [("member", busted[0]["ref_id"], 1000, 0)])
        self.assertEqual(self.ledger_count(), before)
        self.assertEqual({u: self.store.balance(G, u) for u in (ALICE, BOB)}, wallets)
        with self.assertRaises(GameError):  # one player left: no hand
            self.store.deal(G, nxt["round_id"])

    def test_refused_while_unfinished_and_when_closed(self):
        rid = self.lobby((ALICE, BOB))
        with self.assertRaisesRegex(GameError, "unfinished"):
            self.store.next_round(G, 1)
        self.store.deal(G, rid)
        with self.assertRaisesRegex(GameError, "unfinished"):
            self.store.next_round(G, 1)
        self.hand(rid, self.fold_out)
        self.store.close_table(G, 1)
        with self.assertRaisesRegex(GameError, "closed"):
            self.store.next_round(G, 1)

    def test_a_voided_hand_copies_the_starting_stacks(self):
        """Characterization (FEAT-33): cancel_round closes a holdem table, so a voided hand on an open table is only reachable by writing the status."""
        rid = self.lobby((ALICE, BOB))
        self.store.deal(G, rid)
        self.store.db.execute("UPDATE game_rounds SET status='cancelled' WHERE id=?", (rid,))
        self.store.db.commit()
        nxt = self.store.next_round(G, 1)
        self.assertEqual([r["stack"] for r in self.seat_rows(nxt["round_id"])], [1000, 1000])

    def test_cancel_round_ends_the_table_so_next_round_is_refused(self):
        rid = self.lobby((ALICE, BOB))
        self.store.deal(G, rid)
        self.store.cancel_round(G, rid, "restart")
        with self.assertRaisesRegex(GameError, "closed"):
            self.store.next_round(G, 1)


class CloseTests(HoldemCase):
    def test_close_voids_the_hand_and_cashes_out_starting_stacks(self):
        s = self.store
        s.add_favorite(G, ALICE, self.ann)
        self.cwallet(self.ann, 4000)
        snap = self.open(seed="close")
        rid = snap["round_id"]
        self.wallet(ALICE)
        s.join_with_favorites(G, rid, ALICE, 1000, self.w)
        self.sit(rid, (BOB,), 700)
        users, chars = (ALICE, BOB), (self.ann,)
        s.deal(G, rid)
        turn = self.snap(rid)["view"]["turn"]
        s.play(G, rid, turn, "call", "character" if self.snap(rid)["seats"][turn]["kind"] == "character" else "member", 0)
        total = self.total_money(users, chars)
        s.close_table(G, 1)
        self.assertEqual((s.balance(G, ALICE), s.balance(G, BOB), self.cbal(self.ann)), (10000, 10000, 4000))
        self.assertEqual(self.snap(rid)["status"], "cancelled")
        self.assertEqual(s.db.execute("SELECT status FROM game_tables WHERE id=1").fetchone()[0], "closed")
        self.assertEqual(sorted(r["reason"] for r in self.game_ledger() if r["reason"] == CASH), [CASH] * 3)
        self.assertEqual(sorted((r["kind"], r["ref_id"], r["bought_in"], r["cashed_out"]) for r in self.cashouts()),
                         sorted([("member", ALICE, 1000, 1000), ("member", BOB, 700, 700), ("character", self.ann, 1000, 1000)]))
        self.assertEqual(sum(s.balance(G, u) for u in users) + sum(self.cbal(c) for c in chars), total)  # nothing stays at the closed table
        self.assertIsNotNone(self.snap(rid)["seed"])

    def test_double_close_is_harmless(self):
        rid = self.lobby((ALICE, BOB))
        self.store.deal(G, rid)
        self.store.close_table(G, 1)
        before = (self.ledger_count(), len(self.cashouts()), self.store.balance(G, ALICE))
        self.store.close_table(G, 1)
        self.assertEqual((self.ledger_count(), len(self.cashouts()), self.store.balance(G, ALICE)), before)
        self.assertEqual(self.store.cancel_round(G, rid, "again")["status"], "cancelled")
        self.assertEqual((self.ledger_count(), len(self.cashouts())), before[:2])

    def test_close_from_the_lobby_returns_every_buy_in(self):
        rid = self.lobby((ALICE, BOB, CARL), buy_in=600)
        self.store.close_table(G, 1)
        self.assertEqual([self.store.balance(G, u) for u in (ALICE, BOB, CARL)], [10000] * 3)
        self.assertEqual(len(self.cashouts()), 3)
        self.assertEqual(self.snap(rid)["status"], "cancelled")

    def test_close_after_a_settled_hand_keeps_money_whole(self):
        """Characterization (FEAT-33): the spec says seats are cashed out at their stack; after a settled hand the sensible reading is the end stack."""
        rid = self.lobby((ALICE, BOB, CARL))
        self.store.deal(G, rid)
        done = self.hand(rid, self.call_down)
        self.store.close_table(G, 1)
        self.assertEqual(sum(self.store.balance(G, u) for u in (ALICE, BOB, CARL)), 30000)
        for seat in done["seats"]:
            self.assertEqual(self.store.balance(G, seat["ref_id"]), 9000 + seat["end"])

    def test_cancel_round_ends_the_table_and_cashes_out(self):
        rid = self.lobby((ALICE, BOB))
        self.store.deal(G, rid)
        out = self.store.cancel_round(G, rid, "restart")
        self.assertEqual(out["status"], "cancelled")
        self.assertEqual(self.store.db.execute("SELECT status FROM game_tables WHERE id=1").fetchone()[0], "closed")
        self.assertEqual((self.store.balance(G, ALICE), self.store.balance(G, BOB)), (10000, 10000))
        self.assertEqual(len(self.cashouts()), 2)

    def test_a_deleted_character_wallet_pays_nothing(self):
        self.cwallet(self.ann, 5000)
        snap = self.open()
        self.store.fill_seats(G, snap["round_id"], ALICE, [self.ann], self.w)
        self.sit(snap["round_id"], (ALICE,))
        self.store.db.commit()
        self.store.db.execute("PRAGMA foreign_keys=OFF")  # delete_character refuses while seated; the row is removed directly
        self.store.db.execute("DELETE FROM characters WHERE id=?", (self.ann,))
        self.store.db.commit()
        self.store.db.execute("PRAGMA foreign_keys=ON")
        self.store.close_table(G, 1)
        self.assertEqual(self.store.balance(G, ALICE), 10000)
        row = [r for r in self.cashouts() if r["kind"] == "character"][0]
        self.assertEqual((row["bought_in"], row["cashed_out"]), (1000, 0))

    def test_close_is_clamped_at_the_balance_cap(self):
        rid = self.lobby((ALICE, BOB))
        cap = self.store.MAX_CURRENCY_BALANCE
        self.set_balance(ALICE, cap - 250)
        self.store.close_table(G, 1)
        self.assertEqual((self.store.balance(G, ALICE), self.store.balance(G, BOB)), (cap, 10000))
        self.assertEqual(self.snap(rid)["status"], "cancelled")

    def test_close_is_guild_scoped(self):
        self.lobby((ALICE, BOB))
        with self.assertRaises(GameError):
            self.store.close_table(G2, 1)
        self.assertEqual(self.cashouts(), [])


class SummaryTests(HoldemCase):
    def test_game_table_summary_has_pot_winners_and_net_per_seat(self):
        rid = self.lobby((ALICE, BOB, CARL), seed="sum")
        self.store.deal(G, rid)
        self.hand(rid, self.fold_out)
        nxt = self.store.next_round(G, 1)["round_id"]
        self.store.deal(G, nxt)
        done = self.hand(nxt, self.call_down)
        rows = self.store.game_table_summary(G, 1, 10)
        self.assertEqual([r["number"] for r in rows], [1, 2])
        first, second = rows
        # Uncalled excess is refunded, so it is not counted in the result pot.
        self.assertEqual(first["pot"], 10)
        self.assertEqual(len(first["winners"]), 1)
        self.assertIsNone(first["hand"])
        self.assertEqual(sum(s["net"] for s in first["seats"]), 0)
        self.assertEqual(sorted(s["net"] for s in first["seats"]), [-5, 0, 5])
        self.assertEqual(second["pot"], 30)
        self.assertTrue(second["winners"])
        self.assertIsInstance(second["hand"], str)
        self.assertEqual([s["net"] for s in second["seats"]], [x["end"] - x["stack"] for x in done["seats"]])
        self.assertEqual([s["index"] for s in second["seats"]], [0, 1, 2])
        self.assertEqual(self.store.game_table_summary(G, 1, 1)[0]["number"], 2)
        self.assertEqual(self.store.game_table_summary(G2, 1, 10), [])

    def test_summary_net_is_end_minus_start_not_minus_stake(self):
        rid = self.lobby((ALICE, BOB), seed="net")
        self.store.deal(G, rid)
        self.hand(rid, self.fold_out)
        self.store.rebuy(G, self.store.next_round(G, 1)["round_id"], ALICE, 300)  # stake grows, but the next hand's net is still end - start
        nxt = self.store.latest_round(G, 1)["round_id"]
        self.store.deal(G, nxt)
        done = self.hand(nxt, self.fold_out)
        last = self.store.game_table_summary(G, 1, 5)[-1]
        self.assertEqual([s["net"] for s in last["seats"]], [x["end"] - x["stack"] for x in done["seats"]])
        self.assertEqual(sum(s["net"] for s in last["seats"]), 0)

    def test_table_cash_summary_aggregates_a_seat_that_left_and_sat_again(self):
        rid = self.lobby((ALICE, BOB), buy_in=500)
        self.store.leave_round(G, rid, ALICE)
        self.store.join_round(G, rid, ALICE, 800)
        self.store.leave_round(G, rid, ALICE)
        self.wallet(CARL)
        self.store.join_round(G, rid, CARL, 600)
        out = self.store.table_cash_summary(G, 1)
        rows = [r for r in out["cashouts"] if r["ref_id"] == ALICE]
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["kind"], rows[0]["bought_in"], rows[0]["cashed_out"], rows[0]["net"]), ("member", 1300, 1300, 0))
        self.assertIn("name", rows[0])
        self.assertEqual(sorted((r["ref_id"], r["stack"], r["stake"]) for r in out["seated"]), [(BOB, 500, 500), (CARL, 600, 600)])
        self.assertEqual(self.store.table_cash_summary(G, 1)["cashouts"], out["cashouts"])
        with self.assertRaises(GameError):
            self.store.table_cash_summary(G2, 1)

    def test_table_cash_summary_after_close_lists_everyone_with_net(self):
        self.cwallet(self.ann, 5000)
        snap = self.open()
        self.sit(snap["round_id"], (ALICE, BOB))
        self.store.fill_seats(G, snap["round_id"], ALICE, [self.ann], self.w)
        self.store.deal(G, snap["round_id"])
        self.hand(snap["round_id"], self.shove)
        self.store.close_table(G, 1)
        out = self.store.table_cash_summary(G, 1)
        self.assertEqual(out["seated"], [])
        self.assertEqual(sum(r["net"] for r in out["cashouts"]), 0)
        self.assertEqual({(r["kind"], r["ref_id"]) for r in out["cashouts"]}, {("member", ALICE), ("member", BOB), ("character", self.ann)})
        self.assertEqual([r["name"] for r in out["cashouts"] if r["kind"] == "character"], ["Ann"])


class RecoveryTests(HoldemCase):
    def test_unfinished_rounds_lists_holdem_lobbies_and_hands(self):
        snap = self.open()
        self.assertEqual([(r["guild_id"], r["round_id"], r["status"], r["game"]) for r in self.store.unfinished_rounds()], [(G, snap["round_id"], "joining", "holdem")])
        self.sit(snap["round_id"], (ALICE, BOB))
        self.store.deal(G, snap["round_id"])
        got = self.store.unfinished_rounds()
        self.assertEqual([(r["status"], r["table_id"]) for r in got], [("playing", 1)])
        self.hand(snap["round_id"], self.fold_out)
        self.assertEqual(self.store.unfinished_rounds(), [])

    def test_a_reopened_store_restores_the_hand_from_seed_and_moves(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "holdem.sqlite3"
            store = Store(path)
            store.set_daily_settings(G, 0, 0, 7)
            store.set_game_channel(G, CH, True)
            snap = store.open_table(G, CH, "holdem", ALICE, seed="persist")
            rid = snap["round_id"]
            for u in (ALICE, BOB, CARL):
                store.change_balance(G, u, 5000, "start", 1)
                store.join_round(G, rid, u, 1000)
            store.deal(G, rid)
            for _ in range(3):
                cur = store.round_snapshot(G, rid)
                turn = cur["view"]["turn"]
                menu = holdem.legal_moves_for_view(store.seat_view(G, rid, turn))
                store.play(G, rid, turn, "call" if "call" in menu else "check", "member", cur["moves"])
            before = store.round_snapshot(G, rid)
            views = [store.seat_view(G, rid, i) for i in range(3)]
            self.assertEqual(before["status"], "playing")
            store.close()
            again = Store(path)
            try:
                self.assertEqual(again.round_snapshot(G, rid), before)
                self.assertEqual([again.seat_view(G, rid, i) for i in range(3)], views)
                self.assertEqual([(r["round_id"], r["status"]) for r in again.unfinished_rounds()], [(rid, "playing")])
                again.close_table(G, 1)  # what the Discord layer does at startup: void, then cash out
                self.assertEqual([again.balance(G, u) for u in (ALICE, BOB, CARL)], [5000] * 3)
                self.assertEqual(again.unfinished_rounds(), [])
                self.assertEqual(again.round_snapshot(G, rid)["status"], "cancelled")
            finally:
                again.close()

    def test_a_damaged_move_log_is_a_game_error(self):
        rid = self.lobby((ALICE, BOB))
        self.store.deal(G, rid)
        self.store.db.execute("INSERT INTO game_moves(guild_id,round_id,seat_index,move,actor,created_at) VALUES(?,?,?,?,?,?)", (G, rid, 0, "raise:999999", "member", 0))
        self.store.db.commit()
        with self.assertRaises(GameError):
            self.snap(rid)
        with self.assertRaises(GameError):
            self.store.seat_view(G, rid, 0)


class SweepTests(HoldemCase):
    def test_random_play_keeps_all_currency_constant(self):
        rng = random.Random(33)
        s = self.store
        users_by_table, chars_by_table, rounds = {}, {}, {}
        every_user, every_char = [], []
        for t in range(3):
            channel = CH + 1 + t
            s.set_game_channel(G, channel, True)
            users = [100 + t * 10 + i for i in range(5)]
            chars = [s.create_character(G, self.w, f"T{t}C{i}") for i in range(3)]
            for u in users:
                self.fund(u, 20000)
            for c in chars:
                self.cwallet(c, rng.choice([500, 1500, 6000]))
            users_by_table[t], chars_by_table[t] = users, chars
            every_user += users
            every_char += chars
            rounds[t] = s.open_table(G, channel, "holdem", users[0], seed=f"sweep-{t}")["round_id"]
        total = self.total_money(every_user, every_char)
        ops = hands = 0

        def check():
            nonlocal ops
            ops += 1
            self.assertEqual(self.total_money(every_user, every_char), total, f"after op {ops}")

        def seated(rid):
            return self.seat_rows(rid)

        for round_no in range(32):
            for t in range(3):
                rid, users, chars = rounds[t], users_by_table[t], chars_by_table[t]
                table_id = t + 1
                rows = seated(rid)
                members = {r["ref_id"] for r in rows if r["kind"] == "member"}
                for u in users:  # lobby churn
                    roll = rng.random()
                    if u not in members and roll < 0.5 and len(rows) < 6:
                        try:
                            s.join_round(G, rid, u, rng.randrange(400, 2001, 50))
                        except GameError:
                            pass
                        check()
                    elif u in members and roll < 0.15:
                        s.leave_round(G, rid, u)
                        check()
                    elif u in members and roll < 0.4:
                        try:
                            s.rebuy(G, rid, u, rng.randrange(50, 800, 50))
                        except GameError:
                            pass
                        check()
                    rows = seated(rid)
                    members = {r["ref_id"] for r in rows if r["kind"] == "member"}
                if rng.random() < 0.5 and len(rows) < 6:
                    s.fill_seats(G, rid, users[0], chars, self.w)
                    check()
                if len(seated(rid)) < 2:
                    for u in users[:2]:
                        if u not in {r["ref_id"] for r in seated(rid)}:
                            if s.balance(G, u) < 1000:  # a deliberate top-up from outside: the expected total moves with it
                                self.fund(u, 5000)
                                total += 5000
                            s.join_round(G, rid, u, 1000)
                            check()
                if len(seated(rid)) > 6:
                    for r in seated(rid)[6:]:
                        if r["kind"] == "member":
                            s.leave_round(G, rid, r["ref_id"])
                            check()
                self.assertTrue(2 <= len(seated(rid)) <= 9)
                s.deal(G, rid)
                check()
                for _ in range(400):
                    cur = s.round_snapshot(G, rid)
                    if cur["status"] != "playing":
                        break
                    turn = cur["view"]["turn"]
                    menu = holdem.legal_moves_for_view(s.seat_view(G, rid, turn))
                    weights = [6 if m in ("call", "check") else 1 if m == "fold" else 2 for m in menu]
                    move = rng.choices(menu, weights)[0]
                    kind = cur["seats"][turn]["kind"]
                    s.play(G, rid, turn, move, "character" if kind == "character" else "member", cur["moves"])
                    check()
                else:
                    self.fail("hand did not finish")
                done = s.round_snapshot(G, rid)
                self.assertEqual(sum(x["end"] for x in done["seats"]), sum(x["stack"] for x in done["seats"]))
                hands += 1
                rounds[t] = s.next_round(G, table_id)["round_id"]
                check()
        self.assertGreaterEqual(hands, 90)
        for t in range(3):
            s.close_table(G, t + 1)
        self.assertEqual(sum(s.balance(G, u) for u in every_user) + sum(self.cbal(c) for c in every_char), total)


class ReviewFindingTests(HoldemCase):
    """Opus review findings on the Hold'em store (FEAT-33 part B)."""

    def seated_character_table(self, seed="rv"):
        """A table with Alice, Bo's wallet-funded character Ann and a member, hand not yet dealt; returns the round id."""
        self.cwallet(self.ann, 5000)
        snap = self.open(seed=seed)
        rid = snap["round_id"]
        self.sit(rid, (ALICE, BOB))
        self.store.fill_seats(G, rid, ALICE, [self.ann], self.w)
        return rid

    def delete_ann(self):
        return self.store.delete_character(G, self.ann, self.store.owner_revision(G, "character", self.ann))

    def character_exists(self):
        return self.store.character_by_id(self.ann) is not None

    # ---- (1) delete_character while seated at an open cash table
    def test_delete_character_is_refused_in_a_lobby_and_mid_hand(self):
        """Regression (FEAT-33 review): a character seated in a joining or playing Hold'em round cannot be deleted."""
        rid = self.seated_character_table()
        with self.assertRaises(ValueError):
            self.delete_ann()
        self.store.deal(G, rid)
        with self.assertRaises(ValueError):
            self.delete_ann()
        self.assertTrue(self.character_exists())

    def test_known_defect_delete_character_is_refused_between_hands_of_an_open_table(self):
        """FEAT-33 review: a character seated in a settled newest hand of an open cash table still holds chips there, so deleting it is refused."""
        rid = self.seated_character_table()
        self.store.deal(G, rid)
        self.hand(rid, self.fold_out)
        self.assertEqual(self.snap(rid)["status"], "settled")
        self.assertEqual(self.store.db.execute("SELECT status FROM game_tables WHERE id=1").fetchone()[0], "open")
        with self.assertRaises(ValueError):
            self.delete_ann()
        self.assertTrue(self.character_exists())


    def test_delete_character_is_allowed_once_the_table_is_closed(self):
        """Regression (FEAT-33 review): after the table closes the character has been cashed out and may be deleted."""
        rid = self.seated_character_table()
        self.store.deal(G, rid)
        self.hand(rid, self.fold_out)
        self.store.close_table(G, 1)
        self.assertTrue(self.delete_ann())
        self.assertFalse(self.character_exists())

    # ---- (2) open_cash_tables
    def test_known_defect_open_cash_tables_lists_open_holdem_tables_with_newest_status(self):
        """FEAT-33 review: open_cash_tables() lists open cash tables (any newest-round status, settled included) for startup recovery, not closed, blackjack or doubt tables."""
        s = self.store
        for ch in (CH + 1, CH + 2, CH + 3, CH + 4):
            s.set_game_channel(G, ch, True)
        settled = self.lobby((ALICE, BOB), seed="oc1")  # table 1: newest hand settled
        s.deal(G, settled)
        self.hand(settled, self.fold_out)
        s.set_table_message(G, 1, 555)
        lobby = s.open_table(G, CH + 1, "holdem", ALICE, seed="oc2")  # table 2: joining
        closed = s.open_table(G, CH + 2, "holdem", ALICE, seed="oc3")  # table 3: closed below
        s.close_table(G, closed["table_id"])
        bj = s.open_table(G, CH + 3, "blackjack", ALICE)
        doubt = s.open_table(G, CH + 4, "doubt", ALICE)
        s.set_game_enabled(G2, "holdem", True)
        s.set_game_channel(G2, 900, True)
        other = s.open_table(G2, 900, "holdem", ALICE, seed="oc4")
        got = sorted(s.open_cash_tables(), key=lambda r: (r["guild_id"], r["table_id"]))
        self.assertEqual(got, [
            {"guild_id": G, "table_id": 1, "game": "holdem", "channel_id": CH, "thread_id": None, "message_id": 555, "status_of_newest_round": "settled"},
            {"guild_id": G, "table_id": lobby["table_id"], "game": "holdem", "channel_id": CH + 1, "thread_id": None, "message_id": None, "status_of_newest_round": "joining"},
            {"guild_id": G2, "table_id": other["table_id"], "game": "holdem", "channel_id": 900, "thread_id": None, "message_id": None, "status_of_newest_round": "joining"},
        ])
        self.assertNotIn(bj["table_id"], [r["table_id"] for r in got])
        self.assertNotIn(doubt["table_id"], [r["table_id"] for r in got])


    # ---- (3) pot excludes the uncalled excess
    def test_known_defect_summary_pot_excludes_an_uncalled_all_in_excess(self):
        """FEAT-33 review: the table summary's pot is the sum of result pots; shove 1000 into a 200 all-in call is a 400 pot."""
        rules = {"min_buy_in_bb": 20}
        snap = self.open(seed="pot", rules=rules)
        rid = snap["round_id"]
        self.wallet(ALICE)
        self.wallet(BOB)
        self.store.join_round(G, rid, ALICE, 1000)
        self.store.join_round(G, rid, BOB, 200)
        self.store.deal(G, rid)

        def chooser(menu, s):
            if s["seats"][s["view"]["turn"]]["ref_id"] == ALICE:
                raises = [m for m in menu if m.startswith("raise:")]
                return raises[-1] if raises else "call" if "call" in menu else "check"
            return "call" if "call" in menu else "check"

        done = self.hand(rid, chooser)
        self.assertEqual(done["status"], "settled")
        self.assertEqual(sum(x["end"] for x in done["seats"]), 1200)
        self.assertEqual(self.store.game_table_summary(G, 1, 5)[0]["pot"], 400)


    # ---- (4) buy-in wording
    def test_known_defect_non_integer_buy_in_message_says_buy_in(self):
        """FEAT-33 review: a non-whole or boolean buy-in at a Hold'em table is refused in terms of the buy-in, not a bet."""
        rid = self.open()["round_id"]
        self.wallet(ALICE)
        for bad in ("1000", 10.5, True):
            with self.assertRaises(GameError, msg=repr(bad)) as ctx:
                self.store.join_round(G, rid, ALICE, bad)
            self.assertIn("buy-in", str(ctx.exception), repr(bad))
            self.assertNotIn("bet", str(ctx.exception), repr(bad))


    def test_out_of_range_buy_in_message_says_buy_in(self):
        """Regression (FEAT-33 review): the range refusal already talks about the buy-in."""
        rid = self.open()["round_id"]
        self.wallet(ALICE)
        with self.assertRaisesRegex(GameError, "buy-in"):
            self.store.join_round(G, rid, ALICE, 5)

    # ---- (6) exactly one game_cashouts row per departure
    def three_hands_in(self):
        rid = self.lobby((ALICE, BOB, CARL), seed="co")
        self.store.deal(G, rid)
        self.hand(rid, self.fold_out)
        return rid

    def departures(self):
        return sorted((r["kind"], r["ref_id"]) for r in self.cashouts())

    def test_next_round_then_close_table_records_one_cash_out_per_seat(self):
        """Regression (FEAT-33 review): a hand, next_round, then close_table: one game_cashouts row per seat and money conserved."""
        self.three_hands_in()
        self.store.next_round(G, 1)
        self.assertEqual(self.cashouts(), [])
        self.store.close_table(G, 1)
        self.assertEqual(self.departures(), [("member", ALICE), ("member", BOB), ("member", CARL)])
        self.assertEqual(sum(self.store.balance(G, u) for u in (ALICE, BOB, CARL)), 30000)
        self.assertEqual(sum(r["bought_in"] for r in self.cashouts()), sum(r["cashed_out"] for r in self.cashouts()))

    def test_close_table_then_next_round_is_refused_and_adds_no_cash_out(self):
        """Regression (FEAT-33 review): close_table then next_round: refused (closed), still one row per seat."""
        self.three_hands_in()
        self.store.close_table(G, 1)
        rows = self.cashouts()
        with self.assertRaisesRegex(GameError, "closed"):
            self.store.next_round(G, 1)
        self.assertEqual(self.cashouts(), rows)
        self.assertEqual(self.departures(), [("member", ALICE), ("member", BOB), ("member", CARL)])
        self.assertEqual(sum(self.store.balance(G, u) for u in (ALICE, BOB, CARL)), 30000)

    def test_turning_holdem_off_cashes_out_an_open_table(self):
        """Characterization (FEAT-33 review): set_game_enabled('holdem', False) closes the table and cashes every seat out once, conserving money."""
        self.three_hands_in()
        total = 30000
        out = self.store.set_game_enabled(G, "holdem", False)
        self.assertEqual(out["closed_tables"], [1])
        self.assertEqual(self.departures(), [("member", ALICE), ("member", BOB), ("member", CARL)])
        self.assertEqual(sum(self.store.balance(G, u) for u in (ALICE, BOB, CARL)), total)
        self.assertEqual(self.total_money((ALICE, BOB, CARL), ()), total)

    def test_removing_the_game_channel_cashes_out_an_open_table(self):
        """Characterization (FEAT-33 review): set_game_channels without the table's channel closes it and cashes every seat out once."""
        rid = self.lobby((ALICE, BOB), seed="chan")
        self.store.deal(G, rid)
        out = self.store.set_game_channels(G, [])
        self.assertEqual(out["closed_tables"], [1])
        self.assertEqual(self.departures(), [("member", ALICE), ("member", BOB)])
        self.assertEqual((self.store.balance(G, ALICE), self.store.balance(G, BOB)), (10000, 10000))
        self.assertEqual(self.snap(rid)["status"], "cancelled")

    # ---- (7) guild isolation
    def test_cash_summary_and_assist_refuse_another_guild(self):
        """Regression (FEAT-33 review): table_cash_summary and set_table_assist never act on another server's table."""
        rid = self.open()["round_id"]
        self.store.set_game_rules(G, "holdem", {"assist": True})
        with self.assertRaises(GameError):
            self.store.table_cash_summary(G2, 1)
        with self.assertRaises(GameError):
            self.store.set_table_assist(G2, rid, ALICE, True)
        self.assertFalse(self.snap(rid)["rules"]["assist"])


if __name__ == "__main__":
    unittest.main()
