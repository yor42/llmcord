"""FEAT-21 part A3: the I Doubt It store API (table, ante, seats, moves, settle, refund, thread columns).

Assumed API (pinned here, implemented in admin_store.py / games/registry.py):
  fill_seats(guild_id, round_id, user_id, character_ids, space_id, now=None) -> {'snapshot','seated','skipped'}
  set_table_thread / set_board_message / set_turn_message(guild_id, table_id, value); table_for_thread(guild_id, thread_id)
  seat_view(guild_id, round_id, seat_index) -> doubt.View; member_seat_index(guild_id, round_id, user_id) -> int | None
  For a doubt table the join `stake` argument is ignored: the seat stakes the ante of the round's rules snapshot.
Blackjack ledger strings are pinned elsewhere too (test_game_tables.py, test_game_character_seats.py); the last class here is a guard.
"""
import json
import tempfile
import unittest
from pathlib import Path

from test_game_tables import ALICE, BOB, CH, G, TableCase

from llmcord_core.admin_store import ConflictError, GameError
from llmcord_core.games import blackjack as bj
from llmcord_core.games import doubt, registry
from llmcord_core.games.base import Seat
from llmcord_core.store import Store

CARL, DAN = 7, 8
G2 = 2
ANTE_REASON = "I Doubt It ante (table 1, round 1)"
PAYOUT_REASON = "I Doubt It payout (table 1, round 1)"


class DoubtCase(TableCase):
    def setUp(self):
        super().setUp()
        s = self.store
        s.set_daily_settings(G, 0, 0, 7)
        self.w = s.create_space(G, "W", "world")
        self.ann, self.bo, self.cy = (s.create_character(G, self.w, n) for n in ("Ann", "Bo", "Cy"))

    def open(self, ante=10, seed="a", opener=ALICE):
        self.store.set_game_rules(G, "doubt", {"ante": ante})
        return self.store.open_table(G, CH, "doubt", opener, seed=seed)

    def wallet(self, cid, amount):
        self.store.change_character_balance(G, cid, amount, "start", 1)

    def bal(self, cid):
        return self.store.character_balance(G, cid)

    def seat_up(self, rid, users, fund=100):
        for u in users:
            if fund and self.store.balance(G, u) < fund:
                self.fund(u, fund)
            self.store.join_round(G, rid, u, 1)

    def three(self, ante=10, seed="a", fund=100):
        snap = self.open(ante, seed)
        self.seat_up(snap["round_id"], (ALICE, BOB, CARL), fund)
        return snap["round_id"]

    def dealt(self, **kw):
        rid = self.three(**kw)
        self.store.deal(G, rid)
        return rid

    def seat_rows(self, rid):
        return [(r["seat_index"], r["kind"], r["ref_id"], r["stake"], r["brought_by"], r["outcome"], r["payout"]) for r in
                self.store.db.execute("SELECT * FROM game_seats WHERE round_id=? ORDER BY seat_index", (rid,))]

    def reasons(self, guild=G):
        return [r["reason"] for r in self.game_ledger(guild)]

    def engine_state(self, rid):
        rnd = self.store.db.execute("SELECT * FROM game_rounds WHERE id=?", (rid,)).fetchone()
        seats = [Seat(r["seat_index"], r["kind"], r["ref_id"], r["stake"]) for r in
                 self.store.db.execute("SELECT * FROM game_seats WHERE round_id=? ORDER BY seat_index", (rid,))]
        moves = [(r["seat_index"], r["move"]) for r in self.store.db.execute("SELECT * FROM game_moves WHERE round_id=? ORDER BY id", (rid,))]
        return doubt.replay(rnd["seed"], seats, moves, json.loads(rnd["rules_json"]))

    def count(self, rid):
        return self.store.db.execute("SELECT COUNT(*) FROM game_moves WHERE round_id=?", (rid,)).fetchone()[0]


def pick(state, seed, n):
    """A scripted player: the claimer closes an emptied hand, the player on turn doubts a bluff, otherwise plays the timeout card."""
    w = state.window
    if w is not None and not state.hands[w[0]]:
        return w[0], "close"
    if w is not None and any(c[0] != w[1] for c in w[2]):
        return state.turn, "doubt"
    return state.turn, doubt.timeout_move(state, state.turn, f"{seed}/{n}")


def pure_winner(seed, n_seats, limit=2000):
    state = doubt.new(seed, [Seat(i, "member", i, 0) for i in range(n_seats)])
    for n in range(limit):
        if state.finished:
            return state.winner
        seat, move = pick(state, seed, n)
        state = doubt.apply(state, seat, move)
    raise AssertionError("scripted game did not finish")


class RegistryTests(unittest.TestCase):
    def test_doubt_entry_and_min_seats(self):
        d, b = registry.get("doubt"), registry.get("blackjack")
        self.assertIn("doubt", registry.keys())
        self.assertEqual((d.name, d.engine, d.max_seats, d.min_seats, d.ledger_label), ("I Doubt It", doubt, 8, 3, "I Doubt It"))
        self.assertIs(d.normalize_rules, doubt.normalize_rules)
        self.assertEqual((b.min_seats, d.default_enabled), (1, b.default_enabled))


class OpenTests(DoubtCase):
    def test_opens_a_doubt_table_and_snapshots_the_rules(self):
        snap = self.open(ante=10)
        self.assertEqual((snap["game"], snap["status"], snap["rules"]), ("doubt", "joining", {"claim_rule": "sequence", "ante": 10, "turn_timeout": 60}))
        self.store.set_game_rules(G, "doubt", {"ante": 50})
        self.fund(ALICE, 100)
        self.store.join_round(G, snap["round_id"], ALICE, 1)
        self.assertEqual(self.store.balance(G, ALICE), 90)  # the round keeps its own ante

    def test_disabled_not_a_game_channel_and_one_table_per_channel(self):
        self.store.set_game_enabled(G, "doubt", False)
        with self.assertRaises(GameError):
            self.store.open_table(G, CH, "doubt", ALICE)
        self.store.set_game_enabled(G, "doubt", True)
        with self.assertRaises(GameError):
            self.store.open_table(G, CH + 1, "doubt", ALICE)
        self.open()
        with self.assertRaises(GameError):
            self.store.open_table(G, CH, "doubt", BOB)

    def test_blackjack_off_does_not_close_doubt(self):
        self.store.set_game_enabled(G, "blackjack", False)
        self.assertEqual(self.open()["game"], "doubt")


class JoinTests(DoubtCase):
    def test_ante_zero_joins_free_with_no_ledger_rows(self):
        snap = self.open(ante=0)
        self.store.join_round(G, snap["round_id"], ALICE, 5)  # no balance, no ledger
        self.assertEqual(self.seat_rows(snap["round_id"]), [(0, "member", ALICE, 0, None, None, None)])
        self.assertEqual(self.game_ledger(), [])

    def test_ante_debits_with_the_ledger_reason_and_ignores_the_stake_argument(self):
        snap = self.open(ante=10)
        self.store.set_game_settings(G, 50, 60)  # blackjack bet limits do not apply
        self.fund(ALICE, 100)
        self.fund(BOB, 100)
        self.store.join_round(G, snap["round_id"], ALICE, 999)
        self.store.join_round(G, snap["round_id"], BOB, 0)
        self.assertEqual([r[3] for r in self.seat_rows(snap["round_id"])], [10, 10])
        self.assertEqual((self.store.balance(G, ALICE), self.store.balance(G, BOB)), (90, 90))
        rows = self.game_ledger()
        self.assertEqual([(r["user_id"], r["amount"], r["reason"]) for r in rows], [(ALICE, -10, ANTE_REASON), (BOB, -10, ANTE_REASON)])

    def test_cannot_afford_the_ante(self):
        snap = self.open(ante=10)
        self.fund(ALICE, 9)
        with self.assertRaises(GameError):
            self.store.join_round(G, snap["round_id"], ALICE, 10)
        self.assertEqual((self.seat_rows(snap["round_id"]), self.game_ledger(), self.store.balance(G, ALICE)), ([], [], 9))

    def test_double_seat_and_seat_cap_of_eight(self):
        snap = self.open(ante=0)
        rid = snap["round_id"]
        self.store.join_round(G, rid, 50, 1)
        with self.assertRaises(GameError):
            self.store.join_round(G, rid, 50, 1)
        for u in range(51, 58):
            self.store.join_round(G, rid, u, 1)
        self.assertEqual(len(self.seat_rows(rid)), 8)
        with self.assertRaises(GameError):
            self.store.join_round(G, rid, 99, 1)
        self.assertEqual(len(self.seat_rows(rid)), 8)

    def test_member_seat_index(self):
        rid = self.three()
        self.assertEqual([self.store.member_seat_index(G, rid, u) for u in (ALICE, BOB, CARL, DAN)], [0, 1, 2, None])

    def test_favorites_ante_from_wallets_and_skip_broke(self):
        s = self.store
        for c in (self.ann, self.bo, self.cy):
            s.add_favorite(G, ALICE, c)
        self.wallet(self.ann, 100)
        self.wallet(self.bo, 5)
        self.wallet(self.cy, 10)
        snap = self.open(ante=10)
        self.fund(ALICE, 100)
        out = s.join_with_favorites(G, snap["round_id"], ALICE, 77, self.w)
        self.assertEqual(out["seated"], [{"character_id": self.ann, "name": "Ann", "stake": 10}, {"character_id": self.cy, "name": "Cy", "stake": 10}])
        self.assertEqual(out["skipped"], [{"character_id": self.bo, "name": "Bo", "reason": "broke"}])
        self.assertEqual([(r[1], r[3], r[4]) for r in self.seat_rows(snap["round_id"])],
                         [("member", 10, None), ("character", 10, ALICE), ("character", 10, ALICE)])
        self.assertEqual((s.balance(G, ALICE), self.bal(self.ann), self.bal(self.bo), self.bal(self.cy)), (90, 90, 5, 0))
        row = s.ledger(G, character_id=self.ann)[0]
        self.assertEqual((row["amount"], row["source"], row["reason"]), (-10, "game", ANTE_REASON))

    def test_favorites_seat_free_at_ante_zero(self):
        s = self.store
        s.add_favorite(G, ALICE, self.ann)
        snap = self.open(ante=0)
        out = s.join_with_favorites(G, snap["round_id"], ALICE, 1, self.w)
        self.assertEqual(out["seated"], [{"character_id": self.ann, "name": "Ann", "stake": 0}])
        self.assertEqual(self.game_ledger(), [])

    def test_joining_without_a_space_seats_only_the_member(self):
        self.store.add_favorite(G, ALICE, self.ann)
        self.wallet(self.ann, 100)
        snap = self.open(ante=10)
        self.fund(ALICE, 100)
        out = self.store.join_with_favorites(G, snap["round_id"], ALICE, 10, None)
        self.assertEqual((out["seated"], self.bal(self.ann)), ([], 100))
        self.assertEqual(len(self.seat_rows(snap["round_id"])), 1)

    def test_blackjack_still_needs_a_positive_bet(self):
        snap = self.table(seed="s")
        self.fund(ALICE, 100)
        with self.assertRaises(GameError):
            self.store.join_round(G, snap["round_id"], ALICE, 0)


class FillSeatsTests(DoubtCase):
    def setUp(self):
        super().setUp()
        for c in (self.ann, self.bo, self.cy):
            self.wallet(c, 100)
        self.snap = self.open(ante=10)
        self.rid = self.snap["round_id"]

    def fill(self, ids, user=ALICE, space=None, rid=None):
        return self.store.fill_seats(G, rid or self.rid, user, ids, self.w if space is None else space)

    def test_host_fills_house_seats_from_wallets(self):
        out = self.fill([self.ann, self.bo])
        self.assertEqual(out["seated"], [{"character_id": self.ann, "name": "Ann", "stake": 10}, {"character_id": self.bo, "name": "Bo", "stake": 10}])
        self.assertEqual(out["skipped"], [])
        self.assertEqual([s["name"] for s in out["snapshot"]["seats"]], ["Ann", "Bo"])
        self.assertEqual([(r[1], r[2], r[3], r[4]) for r in self.seat_rows(self.rid)], [("character", self.ann, 10, None), ("character", self.bo, 10, None)])
        self.assertEqual((self.bal(self.ann), self.bal(self.bo)), (90, 90))
        row = self.store.ledger(G, character_id=self.ann)[0]
        self.assertEqual((row["amount"], row["reason"]), (-10, ANTE_REASON))

    def test_only_the_host_may_fill(self):
        with self.assertRaises(GameError):
            self.fill([self.ann], user=BOB)
        self.assertEqual((self.seat_rows(self.rid), self.bal(self.ann)), ([], 100))

    def test_unbound_channel_is_refused(self):
        with self.assertRaises(GameError):
            self.store.fill_seats(G, self.rid, ALICE, [self.ann], None)
        self.assertEqual(self.seat_rows(self.rid), [])

    def test_ineligible_characters_are_refused_and_nothing_is_written(self):
        s = self.store
        w2 = s.create_space(G, "W2", "world")
        far = s.create_character(G, w2, "Far")
        other = s.create_space(G2, "O", "world")
        alien = s.create_character(G2, other, "Zed")
        s.archive_character(G, self.cy, True)
        for bad in (far, alien, self.cy, 9999):
            with self.subTest(character=bad), self.assertRaises(GameError):
                self.fill([self.ann, bad])
            self.assertEqual((self.seat_rows(self.rid), self.bal(self.ann)), ([], 100))

    def test_a_hub_reaches_its_linked_worlds(self):
        s = self.store
        w2 = s.create_space(G, "W2", "world")
        far = s.create_character(G, w2, "Far")
        self.wallet(far, 50)
        hub = s.create_space(G, "Hub", "hub")
        s.link_world(G, hub, self.w)
        s.link_world(G, hub, w2)
        out = self.fill([self.ann, far], space=hub)
        self.assertEqual([c["name"] for c in out["seated"]], ["Ann", "Far"])

    def test_skips_broke_seated_and_full(self):
        self.wallet(self.bo, -95)  # 5 left, below the ante
        out = self.fill([self.ann, self.bo])
        self.assertEqual([c["name"] for c in out["seated"]], ["Ann"])
        self.assertEqual(out["skipped"], [{"character_id": self.bo, "name": "Bo", "reason": "broke"}])
        out = self.fill([self.ann, self.cy])
        self.assertEqual(out["skipped"], [{"character_id": self.ann, "name": "Ann", "reason": "seated"}])
        self.assertEqual(self.bal(self.ann), 90)

    def test_full_table_skips_with_reason_full(self):
        s = self.store
        chars = [s.create_character(G, self.w, f"X{i}") for i in range(9)]
        for c in chars:
            self.wallet(c, 100)
        out = self.fill(chars)
        self.assertEqual(len(out["seated"]), 8)
        self.assertEqual(out["skipped"], [{"character_id": chars[8], "name": "X8", "reason": "full"}])
        self.assertEqual(self.bal(chars[8]), 100)

    def test_ante_zero_seats_free(self):
        s = self.store
        s.cancel_round(G, self.rid, "x")
        s.close_table(G, self.snap["table_id"])
        snap = self.open(ante=0)
        out = self.fill([self.ann], rid=snap["round_id"])
        self.assertEqual(out["seated"][0]["stake"], 0)
        self.assertEqual((self.bal(self.ann), self.store.ledger(G, character_id=self.ann)[0]["reason"]), (100, "start"))

    def test_only_while_joining_and_only_for_doubt(self):
        self.seat_up(self.rid, (ALICE, BOB), 100)
        self.fill([self.ann])
        self.store.deal(G, self.rid)
        with self.assertRaises(GameError):
            self.fill([self.bo])
        self.store.cancel_round(G, self.rid, "x")
        self.store.close_table(G, self.snap["table_id"])
        bj_snap = self.table(seed="s")  # opened by user 1
        with self.assertRaises(GameError):
            self.store.fill_seats(G, bj_snap["round_id"], 1, [self.ann], self.w)
        self.assertEqual(self.bal(self.ann), 100)


class LeaveCancelTests(DoubtCase):
    def test_leave_refunds_the_member_and_the_characters_they_brought(self):
        s = self.store
        s.add_favorite(G, ALICE, self.ann)
        self.wallet(self.ann, 100)
        snap = self.open(ante=10)
        rid = snap["round_id"]
        self.seat_up(rid, (BOB,))
        self.fund(ALICE, 100)
        s.join_with_favorites(G, rid, ALICE, 10, self.w)
        self.seat_up(rid, (CARL,))
        after = s.leave_round(G, rid, ALICE)
        self.assertEqual([(x["kind"], x["ref_id"]) for x in after["seats"]], [("member", BOB), ("member", CARL)])
        self.assertEqual([x["index"] for x in after["seats"]], [0, 1])
        self.assertEqual((s.balance(G, ALICE), self.bal(self.ann)), (100, 100))
        refunds = [r["reason"] for r in self.game_ledger() if r["amount"] > 0]
        self.assertEqual(refunds, ["I Doubt It refund (table 1, round 1)"] * 2)

    def test_leave_at_ante_zero_writes_no_ledger(self):
        snap = self.open(ante=0)
        self.store.join_round(G, snap["round_id"], ALICE, 1)
        self.store.leave_round(G, snap["round_id"], ALICE)
        self.assertEqual((self.game_ledger(), self.seat_rows(snap["round_id"])), ([], []))

    def test_cancel_refunds_every_seat_with_the_reason(self):
        s = self.store
        rid = self.three(ante=10)
        self.wallet(self.ann, 100)
        s.fill_seats(G, rid, ALICE, [self.ann], self.w)
        s.cancel_round(G, rid, "mod stop")
        self.assertEqual([s.balance(G, u) for u in (ALICE, BOB, CARL)], [100, 100, 100])
        self.assertEqual(self.bal(self.ann), 100)
        refunds = [r["reason"] for r in self.game_ledger() if r["amount"] > 0]
        self.assertEqual(refunds, ["I Doubt It refund (mod stop), table 1, round 1"] * 4)
        snap = s.round_snapshot(G, rid)
        self.assertEqual(snap["status"], "cancelled")
        self.assertEqual({(x["outcome"], x["payout"]) for x in snap["seats"]}, {("refund", 10)})

    def test_cancel_a_playing_round_refunds_and_ante_zero_writes_nothing(self):
        rid = self.dealt(ante=10)
        self.store.cancel_round(G, rid, "stopped")
        self.assertEqual([self.store.balance(G, u) for u in (ALICE, BOB, CARL)], [100, 100, 100])
        s = Store()
        self.addCleanup(s.close)
        s.set_game_channel(G, CH, True)
        snap = s.open_table(G, CH, "doubt", ALICE, seed="z")
        for u in (ALICE, BOB, CARL):
            s.join_round(G, snap["round_id"], u, 1)
        s.deal(G, snap["round_id"])
        s.cancel_round(G, snap["round_id"], "stopped")
        self.assertEqual(s.db.execute("SELECT COUNT(*) FROM currency_ledger").fetchone()[0], 0)


class DealPlayTests(DoubtCase):
    def test_deal_needs_three_players(self):
        snap = self.open(ante=10)
        rid = snap["round_id"]
        with self.assertRaises(GameError):
            self.store.deal(G, rid)
        self.seat_up(rid, (ALICE, BOB))
        with self.assertRaises(GameError) as cm:
            self.store.deal(G, rid)
        self.assertIn("3", str(cm.exception))
        self.assertIn("I Doubt It", str(cm.exception))
        self.assertEqual(self.store.round_snapshot(G, rid)["status"], "joining")
        self.seat_up(rid, (CARL,))
        out = self.store.deal(G, rid)
        self.assertEqual(out["status"], "playing")
        self.assertEqual(sum(out["view"]["counts"]), 52)
        self.assertIsNone(out["view"]["hand"])  # the public snapshot never carries a hand

    def test_seat_view_shows_only_the_seats_own_hand(self):
        rid = self.dealt()
        state = self.engine_state(rid)
        for i in range(3):
            v = self.store.seat_view(G, rid, i)
            self.assertEqual((v.seat, v.counts, v.hand), (i, tuple(len(h) for h in state.hands), doubt.view(state, i).hand))
        self.assertNotEqual(self.store.seat_view(G, rid, 0).hand, self.store.seat_view(G, rid, 1).hand)
        with self.assertRaises(GameError):
            self.store.seat_view(G, rid, 9)

    def test_seat_view_needs_a_playing_round(self):
        snap = self.open()
        with self.assertRaises(GameError):
            self.store.seat_view(G, snap["round_id"], 0)

    def test_off_turn_doubt_then_second_press_conflicts(self):
        rid = self.dealt()
        state = self.engine_state(rid)
        self.store.play(G, rid, 0, doubt.timeout_move(state, 0, "x"), "member", 0)
        self.store.play(G, rid, 2, "doubt", "member", 1)  # seat 2 is not on turn; any seat may doubt
        with self.assertRaises(ConflictError):
            self.store.play(G, rid, 1, "doubt", "member", 1)
        self.assertEqual([m[1] for m in self.moves()][1:], ["doubt"])
        self.assertEqual(self.count(rid), 2)

    def test_illegal_moves_are_friendly_game_errors(self):
        rid = self.dealt()
        state = self.engine_state(rid)
        move = doubt.timeout_move(state, 0, "x")
        with self.assertRaises(GameError):
            self.store.play(G, rid, 1, move, "member", 0)  # not on turn
        with self.assertRaises(GameError):
            self.store.play(G, rid, 0, move, "character", 0)  # wrong kind of actor
        self.store.play(G, rid, 0, move, "member", 0)
        with self.assertRaises(GameError) as cm:
            self.store.play(G, rid, 0, "doubt", "member", 1)  # the claimer cannot doubt their own claim
        self.assertNotIn("not your turn", str(cm.exception).lower())
        self.assertEqual(self.count(rid), 1)

    def test_timeout_actor_closes_the_window(self):
        rid = self.dealt()
        state = self.engine_state(rid)
        self.store.play(G, rid, 0, doubt.timeout_move(state, 0, "x"), "member", 0)
        self.store.play(G, rid, 0, "close", "timeout", 1)
        self.assertIsNone(self.engine_state(rid).window)
        self.assertIsNone(self.store.round_snapshot(G, rid)["view"]["window"])


class SettleTests(DoubtCase):
    def drive(self, rid, seed, limit=300):
        for n in range(limit):
            state = self.engine_state(rid)
            if state.finished:
                return n
            seat, move = pick(state, seed, n)
            kind = state.seats[seat].kind
            self.store.play(G, rid, seat, move, "character" if kind == "character" else "member", n)
        self.fail("scripted game did not finish")

    def test_full_game_pays_the_pot_to_the_winner(self):
        s = self.store
        for c in (self.ann,):
            self.wallet(c, 100)
        snap = self.open(ante=10, seed="a")
        rid = snap["round_id"]
        self.seat_up(rid, (ALICE, BOB, CARL))
        s.fill_seats(G, rid, ALICE, [self.ann], self.w)
        s.deal(G, rid)
        self.drive(rid, "a")
        winner = pure_winner("a", 4)
        out = s.round_snapshot(G, rid)
        self.assertEqual(out["status"], "settled")
        self.assertEqual([(x["outcome"], x["payout"]) for x in out["seats"]],
                         [("win", 40) if i == winner else ("lose", 0) for i in range(4)])
        holders = [(ALICE, False), (BOB, False), (CARL, False), (self.ann, True)]
        for i, (ref, is_char) in enumerate(holders):
            have = self.bal(ref) if is_char else s.balance(G, ref)
            self.assertEqual(have, 130 if i == winner else 90, i)
        payouts = [r for r in self.game_ledger() if r["amount"] > 0]
        self.assertEqual([(r["amount"], r["reason"]) for r in payouts], [(40, PAYOUT_REASON)])
        self.assertEqual(sorted(self.reasons()), sorted([ANTE_REASON] * 4 + [PAYOUT_REASON]))
        self.assertEqual(s.round_snapshot(G, rid)["seed"], "a")  # revealed once settled
        self.assertNotIn("forfeit", [m[1] for m in self.moves()])

    def test_ante_zero_game_settles_with_no_ledger_rows(self):
        rid = self.open(ante=0, seed="a")["round_id"]
        self.seat_up(rid, (ALICE, BOB, CARL, DAN), fund=0)
        self.store.deal(G, rid)
        self.drive(rid, "a")
        out = self.store.round_snapshot(G, rid)
        self.assertEqual(out["status"], "settled")
        self.assertEqual(sorted(x["outcome"] for x in out["seats"]), ["lose", "lose", "lose", "win"])
        self.assertEqual({x["payout"] for x in out["seats"]}, {0})
        self.assertEqual(self.game_ledger(), [])

    def test_forfeits_end_the_game_and_pay_the_last_seat(self):
        rid = self.dealt(ante=10)
        self.store.play(G, rid, 1, "forfeit", "timeout", 0)
        self.store.play(G, rid, 2, "forfeit", "timeout", 1)
        out = self.store.round_snapshot(G, rid)
        self.assertEqual((out["status"], [(x["outcome"], x["payout"]) for x in out["seats"]]),
                         ("settled", [("win", 30), ("forfeit", 0), ("forfeit", 0)]))
        self.assertEqual([self.store.balance(G, u) for u in (ALICE, BOB, CARL)], [120, 90, 90])
        self.assertEqual([r["reason"] for r in self.game_ledger()][-1], PAYOUT_REASON)
        with self.assertRaises(GameError):
            self.store.play(G, rid, 0, "close", "timeout", 2)

    def test_table_summary_reports_nets_without_blackjack_totals(self):
        rid = self.dealt(ante=10)
        self.store.play(G, rid, 1, "forfeit", "timeout", 0)
        self.store.play(G, rid, 2, "forfeit", "timeout", 1)
        rows = self.store.game_table_summary(G, self.store.round_snapshot(G, rid)["table_id"], 5)
        self.assertEqual([r["number"] for r in rows], [1])
        self.assertEqual([s["net"] for s in rows[0]["seats"]], [20, -10, -10])


class RecoveryTests(DoubtCase):
    def test_unfinished_rounds_and_replay_survive_a_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "d.sqlite3"
            store = Store(path)
            store.set_game_channel(G, CH, True)
            snap = store.open_table(G, CH, "doubt", ALICE, seed="r")
            for u in (ALICE, BOB, CARL):
                store.join_round(G, snap["round_id"], u, 1)
            store.deal(G, snap["round_id"])
            first = store.seat_view(G, snap["round_id"], 1)
            store.close()
            store = Store(path)
            try:
                found = [r for r in store.unfinished_rounds() if r["round_id"] == snap["round_id"]]
                self.assertEqual([(r["guild_id"], r["status"]) for r in found], [(G, "playing")])
                self.assertEqual(store.seat_view(G, snap["round_id"], 1), first)
                self.assertEqual(store.round_snapshot(G, snap["round_id"])["status"], "playing")
            finally:
                store.close()


class ThreadColumnTests(DoubtCase):
    def test_setters_and_lookup_are_guild_scoped(self):
        s = self.store
        tid = self.open()["table_id"]
        s.set_table_thread(G, tid, 555)
        s.set_board_message(G, tid, 556)
        s.set_turn_message(G, tid, 557)
        row = s.open_table_for(G, CH)
        self.assertEqual((row["thread_id"], row["board_message_id"], row["turn_message_id"]), (555, 556, 557))
        self.assertEqual(s.table_for_thread(G, 555)["id"], tid)
        self.assertIsNone(s.table_for_thread(G2, 555))
        self.assertIsNone(s.table_for_thread(G, 999))
        for setter in (s.set_table_thread, s.set_board_message, s.set_turn_message):
            with self.assertRaises(GameError):
                setter(G2, tid, 1)
        s.set_turn_message(G, tid, None)
        self.assertIsNone(s.open_table_for(G, CH)["turn_message_id"])
        self.assertEqual(s.open_table_for(G, CH)["thread_id"], 555)


class IsolationTests(DoubtCase):
    def test_another_guild_cannot_see_or_act_on_a_doubt_table(self):
        s = self.store
        rid = self.dealt()
        s.set_game_channel(G2, CH, True)
        s.set_daily_settings(G2, 0, 0, 7)
        for call in (lambda: s.join_round(G2, rid, 9, 1), lambda: s.deal(G2, rid), lambda: s.leave_round(G2, rid, ALICE),
                     lambda: s.play(G2, rid, 1, "forfeit", "timeout", 0), lambda: s.seat_view(G2, rid, 0),
                     lambda: s.cancel_round(G2, rid, "x"), lambda: s.round_snapshot(G2, rid),
                     lambda: s.fill_seats(G2, rid, ALICE, [self.ann], self.w)):
            with self.assertRaises(GameError):
                call()
        self.assertIsNone(s.member_seat_index(G2, rid, ALICE))
        self.assertEqual(s.round_snapshot(G, rid)["status"], "playing")
        self.assertEqual(self.count(rid), 0)


class BlackjackUnchangedTests(DoubtCase):
    def test_blackjack_round_and_ledger_strings(self):
        """Characterization (FEAT-21 A3): blackjack still bets, pays and refunds with its exact ledger strings, with doubt registered."""
        snap = self.started(["10", "9", "10", "8"], stakes=((ALICE, 10),))
        rid = snap["round_id"]
        self.store.play(G, rid, 0, "stand", "member", 0)
        self.assertEqual(self.store.round_snapshot(G, rid)["status"], "settled")
        self.assertEqual([r["reason"] for r in self.game_ledger()], ["Blackjack bet (table 1, round 1)", "Blackjack payout (win), table 1, round 1"])
        self.assertEqual(self.store.balance(G, ALICE), 110)
        nxt = self.store.next_round(G, snap["table_id"])
        self.store.join_round(G, nxt["round_id"], ALICE, 10)
        self.store.cancel_round(G, nxt["round_id"], "x")
        self.assertEqual(self.reasons()[-2:], ["Blackjack bet (table 1, round 2)", "Blackjack refund (x), table 1, round 2"])
        again = self.store.next_round(G, snap["table_id"])
        self.store.join_round(G, again["round_id"], ALICE, 10)
        self.store.leave_round(G, again["round_id"], ALICE)
        self.assertEqual(self.reasons()[-1], "Blackjack refund (table 1, round 3)")
        self.assertEqual(bj.MAX_SEATS, registry.get("blackjack").max_seats)


if __name__ == "__main__":
    unittest.main()
