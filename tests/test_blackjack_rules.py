"""FEAT-28 part A: blackjack rule options (engine rules dict, insurance, even money, surrender, 6:5, ties, dealer stand total) and their store side
(guild rules, per-round snapshot, insurance money, refunds, schema v16 columns)."""
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import test_games_discord as tgd
from test_game_character_seats import SeatCase
from test_game_tables import ALICE, CH, G, TableCase

from llmcord_core.admin_store import ConflictError, GameError
from llmcord_core.games import DefaultPolicy, IllegalMove, Seat
from llmcord_core.games import blackjack as bj
from llmcord_core.store import Store

CLASSIC = {"stand_on": 17, "hit_soft_17": False, "blackjack_pays": "3:2", "ties": "push", "insurance": False, "surrender": False}
INS = {"insurance": True}
SUR = {"surrender": True}


def c(rank, suit=0):
    return bj.RANKS.index(rank) + 13 * suit


def rigged(cards, stakes=(10,), rules=None):
    """Deal order: one card per seat, dealer up, one more per seat, dealer hole, then draws (all 2s)."""
    shoe = tuple(c(r) for r in cards) + tuple(c("2") for _ in range(50))
    seats = [Seat(i, "member", 100 + i, s) for i, s in enumerate(stakes)]
    with mock.patch.object(bj, "new_shoe", return_value=shoe):
        return bj.new("rigged", seats, rules)


def play(state, *moves):
    for seat, move in moves:
        state = bj.apply(state, seat, move)
    return state


def outcome(state, seat=0):
    r = bj.result(state).seats[seat]
    return r.outcome, r.returned


class NormalizeTests(unittest.TestCase):
    def test_empty_values_are_the_classic_table(self):
        self.assertEqual(bj.normalize_rules(None), CLASSIC)
        self.assertEqual(bj.normalize_rules({}), CLASSIC)
        self.assertEqual(bj.CLASSIC_RULES, CLASSIC)

    def test_partial_values_are_filled_in(self):
        self.assertEqual(bj.normalize_rules({"stand_on": 18, "surrender": True}), {**CLASSIC, "stand_on": 18, "surrender": True})

    def test_soft_17_hit_only_means_something_at_17(self):
        self.assertTrue(bj.normalize_rules({"hit_soft_17": True})["hit_soft_17"])
        self.assertFalse(bj.normalize_rules({"stand_on": 16, "hit_soft_17": True})["hit_soft_17"])
        self.assertFalse(bj.normalize_rules({"stand_on": 18, "hit_soft_17": True})["hit_soft_17"])

    def test_bad_values_raise_readable_errors(self):
        for bad, text in [({"stand_on": 15}, "stand on 16, 17 or 18"), ({"stand_on": True}, "stand on 16, 17 or 18"), ({"stand_on": "17"}, "stand on 16, 17 or 18"),
                          ({"hit_soft_17": 1}, "Dealer on soft 17 must be on or off"), ({"insurance": "yes"}, "Insurance must be on or off"),
                          ({"surrender": None}, "Late surrender must be on or off"), ({"blackjack_pays": "2:1"}, "3:2 or 6:5"),
                          ({"ties": "player"}, "push or go to the dealer"), ({"decks": 1}, "Unknown blackjack rule: decks"), ([], "set of options")]:
            with self.assertRaisesRegex(ValueError, text):
                bj.normalize_rules(bad)

    def test_text_covers_every_option(self):
        self.assertEqual(bj.rules_text(CLASSIC), bj.rules_text())
        self.assertIn("Dealer stands on 16", bj.rules_text({"stand_on": 16}))
        self.assertIn("Dealer stands on 18", bj.rules_text({"stand_on": 18}))
        self.assertIn("Dealer hits soft 17", bj.rules_text({"hit_soft_17": True}))
        self.assertIn("Dealer stands on all 17s", bj.rules_text({"stand_on": 17}))
        self.assertIn("Blackjack pays 6:5", bj.rules_text({"blackjack_pays": "6:5"}))
        self.assertIn("Dealer wins ties", bj.rules_text({"ties": "dealer"}))
        self.assertNotIn("Ties push", bj.rules_text({"ties": "dealer"}))
        full = bj.rules_text({"insurance": True, "surrender": True})
        self.assertIn(" · Insurance · Late surrender · ", full)
        for off in ("Insurance", "Late surrender"):
            self.assertNotIn(off, bj.rules_text())

    def test_how_to_play_follows_the_options(self):
        base = bj.how_to_play()
        self.assertNotIn("Insurance", base)
        self.assertNotIn("Surrender", base)
        self.assertIn("Tie: your bet comes back.", base)
        on = bj.how_to_play({"insurance": True, "surrender": True, "ties": "dealer", "stand_on": 18})
        self.assertIn("Insurance", on)
        self.assertIn("Surrender", on)
        self.assertIn("Tie: the dealer wins.", on)
        self.assertIn("draws until 18 or more", on)


class DisplayTests(unittest.TestCase):
    def test_soft_17_hit_is_ignored_in_the_text_unless_the_stand_total_is_17(self):
        self.assertNotIn("hits soft 17", bj.rules_text({"stand_on": 16, "hit_soft_17": True}))
        self.assertIn("draws until 18 or more.", bj.how_to_play({"stand_on": 18, "hit_soft_17": True}))
        self.assertNotIn("soft 17", bj.how_to_play({"stand_on": 18, "hit_soft_17": True}))
        self.assertIn("Dealer hits soft 17", bj.rules_text({"hit_soft_17": True}))

    def test_damaged_round_rules_read_as_none(self):
        from llmcord_core.admin_store import AdminStore
        self.assertIsNone(AdminStore._round_rules({"rules_json": "{nope"}))
        self.assertIsNone(AdminStore._round_rules({"rules_json": '{"stand_on": 3}'}))
        self.assertEqual(AdminStore._round_rules({"rules_json": ""}), CLASSIC)


class GoldenClassicTests(unittest.TestCase):
    SEATS = [Seat(0, "member", 1, 10), Seat(1, "member", 2, 6)]
    CASES = (
        ("g3", [(0, "hit"), (0, "stand"), (1, "stand")], [("win", 20), ("win", 12)]),
        ("g4", [(0, "stand"), (1, "stand")], [("win", 20), ("lose", 0)]),
        ("g5", [(0, "stand")], [("lose", 0), ("blackjack", 15)]),
        ("g15", [(0, "hit"), (0, "hit"), (0, "stand"), (1, "hit"), (1, "stand")], [("win", 20), ("push", 6)]),
        ("g14", [(0, "hit"), (0, "hit"), (1, "hit")], [("bust", 0), ("bust", 0)]),
    )

    def test_fixed_seeds_with_moves_match_under_none_and_classic(self):
        for seed, moves, golden in self.CASES:
            a = bj.replay(seed, self.SEATS, moves)
            b = bj.replay(seed, self.SEATS, moves, bj.CLASSIC_RULES)
            self.assertEqual(a, b)
            self.assertEqual(bj.result(a), bj.result(b))
            self.assertEqual([(r.outcome, r.returned) for r in bj.result(a).seats], golden, seed)


class DealerTests(unittest.TestCase):
    def test_stands_on_16_when_set(self):
        s = play(rigged(["10", "10", "8", "6"], rules={"stand_on": 16}), (0, "stand"))
        self.assertEqual((len(s.dealer), outcome(s)), (2, ("win", 20)))
        s = play(rigged(["10", "10", "8", "6"]), (0, "stand"))
        self.assertEqual((len(s.dealer), outcome(s)), (3, ("push", 10)))

    def test_stands_on_18_when_set(self):
        s = play(rigged(["10", "10", "9", "7"], rules={"stand_on": 18}), (0, "stand"))
        self.assertEqual((len(s.dealer), outcome(s)), (3, ("push", 10)))  # 17 draws a 2 -> 19
        s = play(rigged(["10", "10", "9", "8"], rules={"stand_on": 18}), (0, "stand"))
        self.assertEqual((len(s.dealer), outcome(s)), (2, ("win", 20)))
        s = play(rigged(["10", "10", "9", "7"]), (0, "stand"))
        self.assertEqual((len(s.dealer), outcome(s)), (2, ("win", 20)))

    def test_soft_17_stand_versus_hit(self):
        soft = ["10", "A", "10", "6"]
        s = play(rigged(soft), (0, "stand"))
        self.assertEqual((len(s.dealer), outcome(s)), (2, ("win", 20)))
        s = play(rigged(soft, rules={"hit_soft_17": True}), (0, "stand"))
        self.assertEqual((len(s.dealer), bj.hand_total(s.dealer), outcome(s)), (3, (19, True), ("win", 20)))

    def test_hard_17_always_stands_and_soft_17_is_not_hit_below_the_option(self):
        s = play(rigged(["10", "10", "10", "7"], rules={"hit_soft_17": True}), (0, "stand"))
        self.assertEqual(len(s.dealer), 2)
        s = play(rigged(["10", "A", "10", "6"], rules={"stand_on": 16, "hit_soft_17": True}), (0, "stand"))
        self.assertEqual(len(s.dealer), 2)

    def test_soft_17_draws_at_stand_on_18(self):
        s = play(rigged(["10", "A", "10", "6"], rules={"stand_on": 18}), (0, "stand"))
        self.assertEqual(bj.hand_total(s.dealer), (19, True))


class PayoutTests(unittest.TestCase):
    def test_blackjack_36_rounding(self):
        bjk = ["A", "6", "K", "5"]
        for stake, classic, six_five in [(10, 25, 22), (5, 12, 11), (3, 7, 6), (1, 2, 2)]:
            self.assertEqual(outcome(rigged(bjk, (stake,))), ("blackjack", classic))
            self.assertEqual(outcome(rigged(bjk, (stake,), {"blackjack_pays": "6:5"})), ("blackjack", six_five))

    def test_dealer_wins_ties(self):
        s = play(rigged(["10", "10", "8", "8"], rules={"ties": "dealer"}), (0, "stand"))
        self.assertEqual(outcome(s), ("lose", 0))
        self.assertEqual(outcome(play(rigged(["10", "10", "8", "8"]), (0, "stand"))), ("push", 10))

    def test_dealer_wins_blackjack_versus_blackjack(self):
        self.assertEqual(outcome(rigged(["A", "A", "K", "K"], rules={"ties": "dealer"})), ("lose", 0))
        self.assertEqual(outcome(rigged(["A", "A", "K", "K"])), ("push", 10))

    def test_ties_rule_does_not_touch_wins(self):
        s = play(rigged(["10", "10", "9", "8"], rules={"ties": "dealer"}), (0, "stand"))
        self.assertEqual(outcome(s), ("win", 20))


class InsuranceTests(unittest.TestCase):
    def test_phase_only_with_the_option_and_an_ace_up(self):
        s = rigged(["10", "A", "7", "K"], rules=INS)
        self.assertEqual((s.phase, s.turn, s.finished), ("insurance", 0, False))
        self.assertEqual(bj.legal_moves(s, 0), ("insure", "no_insurance"))
        v = bj.view(s, 0)
        self.assertEqual((v.phase, v.insured, v.legal, v.dealer_hidden), ("insurance", (0,), ("insure", "no_insurance"), True))
        off = rigged(["10", "A", "7", "K"])
        self.assertEqual((off.phase, off.finished, off.dealer_natural), ("play", True, True))
        not_ace = rigged(["10", "K", "7", "A"], rules=INS)
        self.assertEqual((not_ace.phase, not_ace.finished), ("play", True))
        no_natural = rigged(["10", "K", "7", "5"], rules=INS)
        self.assertEqual((no_natural.phase, no_natural.turn), ("play", 0))

    def test_insure_pays_two_to_one_on_a_dealer_natural(self):
        s = play(rigged(["10", "A", "7", "K"], rules=INS), (0, "insure"))
        self.assertTrue(s.finished)
        r = bj.result(s)
        self.assertEqual((r.seats[0].outcome, r.seats[0].returned, r.seats[0].insurance, r.seats[0].stake), ("lose", 15, 5, 10))
        self.assertIn("Seat 1: 10♠ 7♠ (17) lose, insurance paid 2:1", r.summary)

    def test_insurance_is_lost_without_a_natural_and_play_starts(self):
        s = play(rigged(["10", "A", "7", "5"], rules=INS), (0, "insure"))
        self.assertEqual((s.phase, s.turn, s.insured, s.finished), ("play", 0, (5,), False))
        self.assertEqual(bj.legal_moves(s, 0), ("hit", "stand", "double"))
        s = play(s, (0, "stand"))
        r = bj.result(s).seats[0]
        self.assertEqual((r.outcome, r.returned, r.insurance), ("lose", 0, 5))
        self.assertIn("insurance lost", bj.result(s).summary)

    def test_no_insurance_and_a_dealer_natural(self):
        s = play(rigged(["10", "A", "7", "K"], rules=INS), (0, "no_insurance"))
        r = bj.result(s).seats[0]
        self.assertEqual((s.finished, r.outcome, r.returned, r.insurance), (True, "lose", 0, 0))

    def test_even_money_for_a_blackjack_seat(self):
        s = rigged(["A", "A", "K", "6"], rules=INS)
        self.assertEqual(bj.legal_moves(s, 0), ("even_money", "no_insurance"))
        done = play(s, (0, "even_money"))
        self.assertEqual((done.finished, outcome(done)), (True, ("even_money", 20)))
        self.assertEqual(bj.result(done).seats[0].insurance, 0)
        self.assertEqual(outcome(play(s, (0, "no_insurance"))), ("blackjack", 25))

    def test_even_money_pays_one_to_one_even_when_the_dealer_has_a_natural(self):
        s = rigged(["A", "A", "K", "K"], rules=INS)
        self.assertEqual(outcome(play(s, (0, "even_money"))), ("even_money", 20))
        self.assertEqual(outcome(play(s, (0, "no_insurance"))), ("push", 10))
        s = rigged(["A", "A", "K", "K"], rules={**INS, "ties": "dealer"})
        self.assertEqual(outcome(play(s, (0, "no_insurance"))), ("lose", 0))
        self.assertEqual(outcome(play(s, (0, "even_money"))), ("even_money", 20))

    def test_even_money_uses_the_six_five_free_rule(self):
        s = rigged(["A", "A", "K", "6"], rules={**INS, "blackjack_pays": "6:5"})
        self.assertEqual(outcome(play(s, (0, "even_money"))), ("even_money", 20))
        self.assertEqual(outcome(play(s, (0, "no_insurance"))), ("blackjack", 22))

    def test_insure_needs_a_stake_of_two(self):
        s = rigged(["10", "A", "7", "K"], (1,), INS)
        self.assertEqual(bj.legal_moves(s, 0), ("no_insurance",))
        with self.assertRaises(IllegalMove):
            bj.apply(s, 0, "insure")
        s = play(rigged(["10", "A", "7", "K"], (3,), INS), (0, "insure"))
        self.assertEqual((s.insured, outcome(s)), ((1,), ("lose", 3)))  # cost 3//2 = 1, returns 3

    def test_can_insure_false_hides_the_move(self):
        s = rigged(["10", "A", "7", "K"], rules=INS)
        self.assertEqual(bj.legal_moves(s, 0, can_insure=False), ("no_insurance",))

    def test_seats_answer_in_order_before_anything_else(self):
        s = rigged(["10", "10", "A", "7", "8", "5"], (10, 10), INS)
        self.assertEqual(s.turn, 0)
        for seat, move in ((1, "no_insurance"), (0, "hit"), (0, "stand"), (0, "double")):
            with self.assertRaises(IllegalMove):
                bj.apply(s, seat, move)
        s = play(s, (0, "insure"))
        self.assertEqual((s.phase, s.turn, s.insured), ("insurance", 1, (5, 0)))
        s = play(s, (1, "no_insurance"))
        self.assertEqual((s.phase, s.turn), ("play", 0))
        with self.assertRaises(IllegalMove):
            bj.apply(s, 0, "insure")

    def test_natural_seats_are_asked_too_and_play_skips_them(self):
        s = rigged(["A", "10", "A", "K", "7", "5"], (10, 10), INS)
        s = play(s, (0, "no_insurance"), (1, "no_insurance"))
        self.assertEqual((s.phase, s.turn), ("play", 1))

    def test_surrender_is_not_offered_during_the_insurance_phase(self):
        s = rigged(["10", "A", "7", "5"], rules={**INS, **SUR})
        self.assertEqual(bj.legal_moves(s, 0), ("insure", "no_insurance"))
        with self.assertRaises(IllegalMove):
            bj.apply(s, 0, "surrender")
        self.assertIn("surrender", bj.legal_moves(play(s, (0, "no_insurance")), 0))

    def test_default_policy_declines_insurance_and_never_surrenders(self):
        view = SimpleNamespace(total=12)
        self.assertEqual(DefaultPolicy.pick(view, ("insure", "no_insurance")), "no_insurance")
        self.assertEqual(DefaultPolicy.pick(view, ("even_money", "no_insurance")), "no_insurance")
        self.assertEqual(DefaultPolicy.pick(view, ("hit", "stand", "double", "surrender")), "hit")
        self.assertEqual(DefaultPolicy.pick(SimpleNamespace(total=18), ("hit", "stand", "double", "surrender")), "stand")


class SurrenderTests(unittest.TestCase):
    def test_legal_on_the_first_two_cards_only(self):
        s = rigged(["10", "10", "6", "6"], rules=SUR)
        self.assertEqual(bj.legal_moves(s, 0), ("hit", "stand", "double", "surrender"))
        self.assertEqual(bj.legal_moves(s, 0, can_double=False), ("hit", "stand", "surrender"))
        after = play(s, (0, "hit"))
        self.assertEqual(bj.legal_moves(after, 0), ("hit", "stand"))
        with self.assertRaises(IllegalMove):
            bj.apply(after, 0, "surrender")

    def test_off_by_default(self):
        s = rigged(["10", "10", "6", "6"])
        self.assertEqual(bj.legal_moves(s, 0), ("hit", "stand", "double"))
        with self.assertRaises(IllegalMove):
            bj.apply(s, 0, "surrender")

    def test_surrender_returns_half_rounded_down_and_the_dealer_does_not_play(self):
        s = play(rigged(["10", "10", "6", "6"], rules=SUR), (0, "surrender"))
        self.assertEqual((s.finished, s.status, len(s.dealer), outcome(s)), (True, ("surrender",), 2, ("surrender", 5)))
        self.assertEqual(outcome(play(rigged(["10", "10", "6", "6"], (11,), SUR), (0, "surrender"))), ("surrender", 5))
        self.assertEqual(outcome(play(rigged(["10", "10", "6", "6"], (1,), SUR), (0, "surrender"))), ("surrender", 0))

    def test_never_with_a_dealer_natural(self):
        s = rigged(["10", "A", "6", "K"], rules=SUR)
        self.assertEqual((s.finished, bj.legal_moves(s, 0)), (True, ()))
        self.assertEqual(outcome(s), ("lose", 0))

    def test_other_seats_still_play(self):
        s = rigged(["10", "9", "6", "7", "8", "10"], (10, 10), SUR)
        s = play(s, (0, "surrender"), (1, "stand"))
        self.assertEqual(len(s.dealer), 3)  # 16 draws a 2 -> 18 for the seat that stood
        self.assertEqual([r.outcome for r in bj.result(s).seats], ["surrender", "lose"])

    def test_view_of_a_surrendered_seat(self):
        s = play(rigged(["10", "10", "6", "6"], (10, 10), SUR), (0, "surrender"))
        self.assertEqual(bj.view(s, 1).status, ("surrender", "playing"))


class ReplayTests(unittest.TestCase):
    SEATS = [Seat(0, "member", 1, 10), Seat(1, "member", 2, 6)]

    def test_classic_rules_replay_exactly_as_before(self):
        for seed in ("a", "golden", "k3"):
            a = bj.replay(seed, self.SEATS, [])
            self.assertEqual(a, bj.replay(seed, self.SEATS, [], bj.CLASSIC_RULES))
            self.assertEqual(a, bj.replay(seed, self.SEATS, [], {}))
        golden = bj.new("golden", self.SEATS)
        self.assertEqual(bj.hand_str(golden.dealer), "8♥ 8♠")
        self.assertEqual((golden.phase, golden.turn, golden.insured), ("play", 0, (0, 0)))

    def test_replay_with_rules_is_deterministic(self):
        rules = {"stand_on": 18, "surrender": True, "insurance": True, "ties": "dealer"}
        a = bj.replay("seed-x", self.SEATS, [], rules)
        b = bj.replay("seed-x", self.SEATS, [], bj.normalize_rules(rules))
        self.assertEqual(a, b)
        self.assertEqual(a.rules, bj.normalize_rules(rules))

    def test_replay_follows_insurance_and_surrender_moves(self):
        shoe = tuple(c(r) for r in ["10", "A", "6", "5"]) + tuple(c("2") for _ in range(50))
        with mock.patch.object(bj, "new_shoe", return_value=shoe):
            s = bj.replay("r", [Seat(0, "member", 1, 10)], [(0, "no_insurance"), (0, "surrender")], {**INS, **SUR})
            self.assertEqual((s.finished, outcome(s)), (True, ("surrender", 5)))
            with self.assertRaises(IllegalMove):
                bj.replay("r", [Seat(0, "member", 1, 10)], [(0, "surrender")], {**INS, **SUR})

    def test_bad_rules_fail_the_replay(self):
        with self.assertRaises(ValueError):
            bj.new("x", self.SEATS, {"stand_on": 20})


class RulesStoreTests(TableCase):
    def stored(self, guild=G):
        return self.store.one("SELECT blackjack_rules FROM guild_settings WHERE guild_id=?", (guild,))

    def test_default_set_get_and_conflict(self):
        s = self.store
        self.assertEqual(s.blackjack_rules(G), CLASSIC)
        out = s.set_blackjack_rules(G, {"insurance": True, "stand_on": 16}, CLASSIC)
        self.assertEqual(out, {**CLASSIC, "insurance": True, "stand_on": 16})
        self.assertEqual(s.blackjack_rules(G), out)
        self.assertEqual(s.blackjack_rules(2), CLASSIC)
        with self.assertRaisesRegex(ConflictError, "The blackjack rules were changed elsewhere. Reload the page and try again."):
            s.set_blackjack_rules(G, {}, CLASSIC)
        self.assertEqual(s.blackjack_rules(G), out)
        self.assertEqual(s.set_blackjack_rules(G, {}, {"insurance": True, "stand_on": 16}), CLASSIC)
        self.assertEqual(self.stored()[0], "")

    def test_invalid_rules_change_nothing(self):
        with self.assertRaisesRegex(ValueError, "16, 17 or 18"):
            self.store.set_blackjack_rules(G, {"stand_on": 19})
        with self.assertRaises(ValueError):
            self.store.set_blackjack_rules(G, {"ties": "x"}, {"stand_on": 3})
        self.assertIsNone(self.stored())

    def test_rules_do_not_close_tables_or_touch_the_switch(self):
        snap = self.table(seed="a")
        self.store.set_blackjack_rules(G, INS)
        self.assertEqual(self.store.open_table_for(G, CH)["id"], snap["table_id"])
        self.assertTrue(self.store.blackjack_enabled(G))

    def test_each_round_keeps_the_rules_it_was_created_with(self):
        self.store.set_blackjack_rules(G, INS)
        snap = self.table(seed="a")
        self.assertTrue(snap["rules"]["insurance"])
        self.store.set_blackjack_rules(G, {})
        self.assertTrue(self.store.round_snapshot(G, snap["round_id"])["rules"]["insurance"])
        self.rig(["10", "A", "7", "K"])
        self.fund(ALICE, 100)
        self.store.join_round(G, snap["round_id"], ALICE, 10)
        dealt = self.store.deal(G, snap["round_id"])
        self.assertEqual((dealt["status"], dealt["phase"], dealt["legal"]), ("playing", "insurance", {0: ("insure", "no_insurance")}))
        self.store.cancel_round(G, snap["round_id"], "x")
        nxt = self.store.next_round(G, snap["table_id"])
        self.assertEqual(nxt["rules"], CLASSIC)
        self.assertEqual(self.store.one("SELECT rules_json FROM game_rounds WHERE id=?", (nxt["round_id"],))[0], "")

    def test_default_round_stores_empty_rules(self):
        snap = self.table(seed="a")
        self.assertEqual(snap["rules"], CLASSIC)
        self.assertEqual(self.store.one("SELECT rules_json FROM game_rounds WHERE id=?", (snap["round_id"],))[0], "")

    def test_a_round_replays_with_its_own_rules_after_a_rules_change(self):
        self.store.set_blackjack_rules(G, {"ties": "dealer"})
        snap = self.started(["10", "10", "8", "8"])
        self.store.set_blackjack_rules(G, {})
        out = self.store.play(G, snap["round_id"], 0, "stand", "member", 0)
        self.assertEqual((out["seats"][0]["outcome"], self.store.balance(G, ALICE)), ("lose", 90))


class InsuranceMoneyTests(TableCase):
    def setUp(self):
        super().setUp()
        self.store.set_blackjack_rules(G, {"insurance": True, "surrender": True})

    def reasons(self):
        return [(r["amount"], r["reason"]) for r in self.game_ledger()]

    def test_insure_debits_half_and_loses_without_a_natural(self):
        snap = self.started(["10", "A", "7", "5"])
        self.assertEqual(snap["legal"], {0: ("insure", "no_insurance")})
        out = self.store.play(G, snap["round_id"], 0, "insure", "member", 0)
        self.assertEqual((self.store.balance(G, ALICE), out["phase"], out["seats"][0]["insurance"]), (85, "play", 5))
        self.assertEqual(self.reasons()[-1], (-5, "Blackjack insurance (table 1, round 1)"))
        out = self.store.play(G, snap["round_id"], 0, "stand", "member", 1)
        self.assertEqual((out["status"], out["seats"][0]["outcome"], out["seats"][0]["payout"]), ("settled", "lose", 0))
        self.assertEqual(self.store.balance(G, ALICE), 85)
        self.assertEqual(sum(a for a, _ in self.reasons()), -15)

    def test_insure_returns_the_stake_back_in_full_on_a_natural(self):
        snap = self.started(["10", "A", "7", "K"])
        out = self.store.play(G, snap["round_id"], 0, "insure", "member", 0)
        self.assertEqual((out["status"], out["seats"][0]["outcome"], out["seats"][0]["payout"], out["seats"][0]["insurance"]), ("settled", "lose", 15, 5))
        self.assertEqual(self.store.balance(G, ALICE), 100)
        self.assertEqual(self.reasons()[-1], (15, "Blackjack payout (lose), table 1, round 1, insurance paid"))
        self.assertEqual(self.store.game_table_summary(G, snap["table_id"], 5)[0]["seats"][0]["net"], 0)

    def test_declined_insurance_costs_nothing(self):
        snap = self.started(["10", "A", "7", "K"])
        out = self.store.play(G, snap["round_id"], 0, "no_insurance", "member", 0)
        self.assertEqual((out["status"], self.store.balance(G, ALICE)), ("settled", 90))
        self.assertEqual(len(self.game_ledger()), 1)  # the bet only

    def test_short_member_is_refused_and_cannot_see_the_button(self):
        snap = self.started(["10", "A", "7", "5"], funds=10)
        self.assertEqual(snap["legal"], {0: ("no_insurance",)})
        name = self.store.currency_name(G)
        with self.assertRaisesRegex(GameError, f"You need 5 {name} for insurance but have 0."):
            self.store.play(G, snap["round_id"], 0, "insure", "member", 0)
        self.assertEqual((self.store.balance(G, ALICE), len(self.moves())), (0, 0))

    def test_insure_when_not_offered_is_refused(self):
        self.store.set_blackjack_rules(G, {})
        snap = self.started(["10", "A", "7", "5"])
        with self.assertRaisesRegex(GameError, "isn't allowed"):
            self.store.play(G, snap["round_id"], 0, "insure", "member", 0)

    def test_insure_checks_legality_before_the_balance(self):
        snap = self.started(["10", "A", "7", "5"], funds=10)
        self.store.play(G, snap["round_id"], 0, "no_insurance", "member", 0)  # play phase now, balance 0
        with self.assertRaisesRegex(GameError, "isn't allowed"):
            self.store.play(G, snap["round_id"], 0, "insure", "member", 1)
        self.assertEqual(self.store.balance(G, ALICE), 0)

    def test_insurance_payout_clamps_at_the_balance_cap(self):
        snap = self.started(["10", "A", "7", "K"])
        top = self.store.MAX_CURRENCY_BALANCE
        self.store.db.execute("UPDATE currency_balances SET balance=? WHERE guild_id=? AND user_id=?", (top - 4, G, ALICE))
        self.store.db.commit()
        self.store.play(G, snap["round_id"], 0, "insure", "member", 0)  # costs 5 -> top - 9; 15 owed, 9 fits
        out = self.store.round_snapshot(G, snap["round_id"])
        self.assertEqual((out["seats"][0]["payout"], self.store.balance(G, ALICE)), (9, top))

    def test_even_money_debits_nothing_and_pays_one_to_one(self):
        snap = self.started(["A", "A", "K", "6"])
        self.assertEqual(snap["legal"], {0: ("even_money", "no_insurance")})
        out = self.store.play(G, snap["round_id"], 0, "even_money", "member", 0)
        self.assertEqual((out["seats"][0]["outcome"], out["seats"][0]["payout"], self.store.balance(G, ALICE)), ("even_money", 20, 110))
        self.assertEqual([a for a, _ in self.reasons()], [-10, 20])

    def test_surrender_returns_half(self):
        snap = self.started(["10", "10", "6", "6"])
        self.assertEqual(snap["legal"], {0: ("hit", "stand", "double", "surrender")})
        out = self.store.play(G, snap["round_id"], 0, "surrender", "member", 0)
        self.assertEqual((out["seats"][0]["outcome"], out["seats"][0]["payout"], self.store.balance(G, ALICE)), ("surrender", 5, 95))

    def test_cancel_refunds_stake_and_insurance(self):
        snap = self.started(["10", "A", "7", "5"])
        self.store.play(G, snap["round_id"], 0, "insure", "member", 0)
        self.assertEqual(self.store.balance(G, ALICE), 85)
        out = self.store.cancel_round(G, snap["round_id"], "restart")
        self.assertEqual((out["seats"][0]["payout"], self.store.balance(G, ALICE), sum(a for a, _ in self.reasons())), (15, 100, 0))

    def test_timeout_declines_insurance(self):
        snap = self.started(["10", "A", "7", "K"])
        out = self.store.play(G, snap["round_id"], 0, "no_insurance", "timeout", 0)
        self.assertEqual((out["status"], self.moves()[0]["actor"]), ("settled", "timeout"))

    def test_six_five_and_dealer_ties_through_the_store(self):
        self.store.set_blackjack_rules(G, {"blackjack_pays": "6:5", "ties": "dealer"})
        self.started(["A", "6", "K", "5"])
        self.assertEqual(self.store.balance(G, ALICE), 112)


class TimerTests(tgd.GameCase):
    async def test_turn_timer_in_the_insurance_phase_declines_and_debits_nothing(self):
        self.store.set_blackjack_rules(G, INS)
        self.rig("10", "A", "7", "5")
        await self.bet(tgd.ALICE, 10)
        await self.press(tgd.ALICE, "Deal now")
        snap = self.latest()
        self.assertEqual(snap["phase"], "insurance")
        before = self.balance(tgd.ALICE)
        await self.games.on_turn_timer(G, snap["table_id"], snap["round_id"], 0)
        rows = [tuple(r) for r in self.store.db.execute("SELECT seat_index,move,actor FROM game_moves")]
        self.assertEqual(rows, [(0, "no_insurance", "timeout")])
        self.assertEqual(self.balance(tgd.ALICE), before)
        self.assertEqual(self.latest()["phase"], "play")


class CharacterInsuranceTests(SeatCase):
    def setUp(self):
        super().setUp()
        self.store.set_blackjack_rules(G, INS)
        self.wallet(self.ann, 100)

    def deal(self, cards):
        self.rig(cards)
        snap, _ = self.join(10)
        return self.store.deal(G, snap["round_id"])

    def test_character_insures_from_its_wallet(self):
        snap = self.deal(["10", "10", "A", "7", "8", "K"])
        self.assertEqual(snap["legal"], {0: ("insure", "no_insurance")})
        snap = self.store.play(G, snap["round_id"], 0, "no_insurance", "member", 0)
        self.assertEqual(snap["legal"], {1: ("insure", "no_insurance")})
        with self.assertRaises(GameError):
            self.store.play(G, snap["round_id"], 1, "insure", "member", 1)
        snap = self.store.play(G, snap["round_id"], 1, "insure", "character", 1)
        self.assertEqual(snap["status"], "settled")
        row = [r for r in self.store.ledger(G, character_id=self.ann) if r["reason"].startswith("Blackjack insurance")][0]
        self.assertEqual((row["amount"], row["reason"]), (-5, "Blackjack insurance (table 1, round 1)"))
        self.assertEqual(self.bal(self.ann), 100 - 10 - 5 + 15)
        self.assertEqual(self.store.balance(G, ALICE), 1000 - 10)

    def test_broke_character_cannot_insure(self):
        self.store.change_character_balance(G, self.ann, -95, "spend", 1)
        snap = self.deal(["10", "10", "A", "7", "8", "K"])
        self.store.play(G, snap["round_id"], 0, "no_insurance", "member", 0)
        snap = self.store.round_snapshot(G, snap["round_id"])
        self.assertEqual((self.bal(self.ann), snap["legal"]), (0, {1: ("no_insurance",)}))
        with self.assertRaisesRegex(GameError, "for insurance but have 0"):
            self.store.play(G, snap["round_id"], 1, "insure", "character", 1)

    def test_cancel_refunds_a_character_its_insurance(self):
        snap = self.deal(["10", "10", "A", "7", "8", "5"])
        self.store.play(G, snap["round_id"], 0, "no_insurance", "member", 0)
        self.store.play(G, snap["round_id"], 1, "insure", "character", 1)
        self.assertEqual(self.bal(self.ann), 85)
        self.store.cancel_round(G, snap["round_id"], "x")
        self.assertEqual((self.bal(self.ann), self.store.balance(G, ALICE)), (100, 1000))


class ConservationTests(SeatCase):
    """Every ledger row is a debit or credit of a seat; wallets move by exactly the rows and the seat nets."""

    SCENARIOS = [
        ({"insurance": True}, ["10", "A", "A", "7", "K", "K"], ("even_money", "insure")),   # dealer natural: seat 0 insured, seat 1 even money
        ({"insurance": True, "blackjack_pays": "6:5"}, ["10", "A", "A", "5", "K", "6"], ("insure", "no_insurance", "stand")),
        ({"insurance": True, "surrender": True}, ["10", "9", "A", "6", "7", "5"], ("no_insurance", "surrender")),
        ({"surrender": True, "ties": "dealer", "stand_on": 18}, ["10", "10", "10", "8", "6", "8"], ("surrender", "stand")),
        ({"hit_soft_17": True, "blackjack_pays": "6:5"}, ["A", "10", "10", "K", "9", "6"], ("stand",)),
        ({"insurance": True, "ties": "dealer"}, ["A", "A", "K", "K", "K", "K"], ("no_insurance", "even_money")),
        ({"insurance": True}, ["5", "6", "A", "6", "5", "9"], ("insure", "double", "stand")),
    ]

    def test_money_in_equals_money_out_plus_house_net(self):
        self.wallet(self.ann, 300)
        wallets = lambda: self.bal(self.ann) + self.store.balance(G, ALICE)  # noqa: E731
        start, snap = wallets(), None
        for n, (rules, cards, prefer) in enumerate(self.SCENARIOS):
            self.store.set_blackjack_rules(G, rules)
            self.rig(cards)
            snap = self.table(seed=f"c{n}") if n == 0 else self.store.next_round(G, 1)
            self.store.join_with_favorites(G, snap["round_id"], ALICE, 13, self.w)
            snap = self.store.deal(G, snap["round_id"])
            while snap["status"] == "playing":
                turn, legal = next(iter(snap["legal"].items()))
                move = next((m for m in prefer if m in legal), legal[-1])
                kind = snap["seats"][turn]["kind"]
                snap = self.store.play(G, snap["round_id"], turn, move, "character" if kind == "character" else "member", snap["moves"])
            self.assertEqual(snap["status"], "settled", (n, snap["status"]))
        rows = self.store.db.execute("SELECT SUM(amount) FROM currency_ledger WHERE source='game'").fetchone()[0]
        seats = self.store.db.execute("SELECT SUM(payout-stake-insurance) FROM game_seats").fetchone()[0]
        self.assertEqual(rows, seats)
        self.assertEqual(wallets() - start, rows)
        self.assertGreater(self.store.db.execute("SELECT COUNT(*) FROM game_seats WHERE insurance>0").fetchone()[0], 0)
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM game_seats WHERE kind='character'").fetchone()[0], len(self.SCENARIOS))
        for r in self.store.db.execute("SELECT * FROM game_seats"):
            self.assertGreaterEqual(r["payout"], 0)
        self.assertGreaterEqual(min(self.bal(self.ann), self.store.balance(G, ALICE)), 0)


class SchemaColumnTests(unittest.TestCase):
    def test_existing_v16_file_gets_the_columns_and_replays(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v16.sqlite3"
            store = Store(path)
            store.set_game_channel(G, CH, True)
            store.close()
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("INSERT INTO game_tables(guild_id,channel_id,game,status,opened_by,created_at,updated_at) VALUES(1,100,'blackjack','open',1,0,0)")
                db.execute("ALTER TABLE game_rounds DROP COLUMN rules_json")
                db.execute("ALTER TABLE game_seats DROP COLUMN insurance")
                db.execute("ALTER TABLE guild_settings DROP COLUMN blackjack_rules")
                db.execute("INSERT INTO game_rounds(guild_id,table_id,number,seed,seed_hash,status,created_at) VALUES(1,1,1,'old','h','joining',0)")
            store = Store(path)
            try:
                cols = lambda t: {r[1] for r in store.db.execute(f"PRAGMA table_info({t})")}  # noqa: E731
                self.assertIn("rules_json", cols("game_rounds"))
                self.assertIn("insurance", cols("game_seats"))
                self.assertIn("blackjack_rules", cols("guild_settings"))
                self.assertEqual(store.blackjack_rules(G), CLASSIC)
                self.assertEqual(store.round_snapshot(G, 1)["rules"], CLASSIC)
                self.assertEqual(store.set_blackjack_rules(G, INS)["insurance"], True)
            finally:
                store.close()
            self.assertEqual(list(Path(folder).glob("*.pre-*.sqlite3")), [])
            Store(path).close()


if __name__ == "__main__":
    unittest.main()
