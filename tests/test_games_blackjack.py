"""Blackjack rules engine (llmcord_core/games/blackjack.py): every outcome path, view secrecy, replay."""
import unittest
from unittest import mock

from llmcord_core.games import DefaultPolicy, IllegalMove, Seat
from llmcord_core.games import blackjack as bj


def c(rank, suit=0):
    return bj.RANKS.index(rank) + 13 * suit


def rigged(cards, stakes=(10,)):
    """Start a round whose shoe begins with `cards` (rank strings) in deal order.

    Deal order: one card per seat, dealer up, one more per seat, dealer hole, then draws.
    """
    shoe = tuple(c(r) for r in cards) + tuple(c("2") for _ in range(50))
    seats = [Seat(i, "member", 100 + i, s) for i, s in enumerate(stakes)]
    with mock.patch.object(bj, "new_shoe", return_value=shoe):
        return bj.new("rigged", seats)


def play(state, *moves):
    for seat, move in moves:
        state = bj.apply(state, seat, move)
    return state


class ShoeTests(unittest.TestCase):
    def test_golden_first_cards(self):
        shoe = bj.new_shoe("golden")
        self.assertEqual([bj.card_str(x) for x in shoe[:8]], ["J♥", "Q♥", "8♥", "7♠", "5♣", "8♠", "K♦", "4♥"])

    def test_shoe_is_six_decks_permutation_differing_by_seed(self):
        shoe = bj.new_shoe("a")
        self.assertEqual(len(shoe), 312)
        self.assertEqual(sorted(shoe), sorted(list(range(52)) * 6))
        self.assertNotEqual(shoe, bj.new_shoe("b"))
        self.assertEqual(shoe, bj.new_shoe("a"))

    def test_deal_order_with_two_seats(self):
        s = rigged(["2", "3", "4", "5", "6", "7"], stakes=(1, 1))
        self.assertEqual([bj.hand_str(h) for h in s.hands], ["2♠ 5♠", "3♠ 6♠"])
        self.assertEqual(bj.hand_str(s.dealer), "4♠ 7♠")


class RenderTests(unittest.TestCase):
    def test_cards_and_totals(self):
        self.assertEqual(bj.hand_str([c("A"), c("10", 1)]), "A♠ 10♥")
        self.assertEqual(bj.total_str([c("A"), c("6")]), "soft 17")
        self.assertEqual(bj.total_str([c("A"), c("6"), c("10")]), "17")
        self.assertEqual(bj.total_str([c("A"), c("A"), c("9")]), "soft 21")
        self.assertEqual(bj.hand_total([c("A"), c("A"), c("A")]), (13, True))
        self.assertEqual(bj.hand_total([c("K"), c("Q"), c("5")]), (25, False))


class OutcomeTests(unittest.TestCase):
    def outcome(self, state):
        r = bj.result(state)
        return r.seats[0].outcome, r.seats[0].returned

    def test_player_natural_pays_three_to_two(self):
        s = rigged(["A", "9", "K", "8"], stakes=(10,))  # player A K, dealer 9 8
        self.assertTrue(s.finished)
        self.assertEqual(self.outcome(s), ("blackjack", 25))

    def test_natural_payout_rounds_down_for_odd_stakes(self):
        self.assertEqual(self.outcome(rigged(["A", "9", "K", "8"], stakes=(5,))), ("blackjack", 12))  # profit 7
        self.assertEqual(self.outcome(rigged(["A", "9", "K", "8"], stakes=(1,))), ("blackjack", 2))
        self.assertEqual(self.outcome(rigged(["A", "9", "K", "8"], stakes=(3,))), ("blackjack", 7))

    def test_dealer_natural_ends_round_at_once(self):
        s = rigged(["10", "A", "9", "K"])
        self.assertTrue(s.finished)
        self.assertIsNone(s.turn)
        self.assertEqual(self.outcome(s), ("lose", 0))
        self.assertEqual(s.pos, 4)  # nobody drew

    def test_both_naturals_push(self):
        s = rigged(["A", "A", "K", "K"])
        self.assertEqual(self.outcome(s), ("push", 10))

    def test_dealer_natural_mixed_table(self):
        s = rigged(["A", "5", "A", "K", "5", "K"], stakes=(10, 10))
        # seats: A K (natural), 5 5 ; dealer A K
        r = bj.result(s)
        self.assertEqual([(x.outcome, x.returned) for x in r.seats], [("push", 10), ("lose", 0)])

    def test_player_bust(self):
        s = rigged(["10", "9", "6", "8", "K"])  # 16 hits K -> 26
        s = play(s, (0, "hit"))
        self.assertTrue(s.finished)
        self.assertEqual(self.outcome(s), ("bust", 0))
        self.assertEqual(s.pos, 5)  # dealer did not draw against a busted table

    def test_dealer_bust(self):
        s = rigged(["10", "6", "8", "10", "K"])  # player 18, dealer 6+10=16 draws K
        s = play(s, (0, "stand"))
        self.assertEqual(bj.hand_total(s.dealer)[0], 26)
        self.assertEqual(self.outcome(s), ("win", 20))

    def test_dealer_stands_on_soft_17(self):
        s = rigged(["10", "A", "9", "6"])  # dealer soft 17
        s = play(s, (0, "stand"))
        self.assertEqual(len(s.dealer), 2)
        self.assertEqual(self.outcome(s), ("win", 20))

    def test_dealer_hits_soft_16(self):
        s = rigged(["10", "A", "10", "5", "3"])  # player 20, dealer soft 16 draws 3 -> soft 19
        s = play(s, (0, "stand"))
        self.assertEqual(len(s.dealer), 3)
        self.assertEqual(bj.hand_total(s.dealer), (19, True))
        self.assertEqual(self.outcome(s), ("win", 20))

    def test_soft_hand_counts_ace_as_one_when_needed(self):
        s = rigged(["A", "10", "6", "7", "9"])  # player soft 17, dealer 17
        s = play(s, (0, "hit"))  # + 9 -> 16 hard
        self.assertEqual(bj.hand_total(s.hands[0]), (16, False))
        self.assertFalse(s.finished)

    def test_win_lose_push_by_totals(self):
        s = play(rigged(["10", "10", "8", "7"]), (0, "stand"))
        self.assertEqual(self.outcome(s), ("win", 20))
        s = play(rigged(["10", "10", "7", "7"]), (0, "stand"))
        self.assertEqual(self.outcome(s), ("push", 10))
        s = play(rigged(["10", "10", "6", "7"]), (0, "stand"))
        self.assertEqual(self.outcome(s), ("lose", 0))

    def test_double_win_and_lose_use_final_stake(self):
        s = rigged(["5", "10", "6", "8", "10"], stakes=(10,))  # 11 doubles, draws 10 -> 21; dealer 18
        s = play(s, (0, "double"))
        self.assertTrue(s.finished)
        self.assertEqual(s.stakes, (20,))
        r = bj.result(s)
        self.assertEqual((r.seats[0].outcome, r.seats[0].stake, r.seats[0].returned), ("win", 20, 40))
        s = rigged(["5", "10", "6", "10", "2"], stakes=(10,))  # draws 2 -> 13 vs dealer 20
        s = play(s, (0, "double"))
        r = bj.result(s)
        self.assertEqual((r.seats[0].outcome, r.seats[0].stake, r.seats[0].returned), ("lose", 20, 0))

    def test_double_bust(self):
        s = play(rigged(["10", "10", "6", "8", "K"]), (0, "double"))
        r = bj.result(s)
        self.assertEqual((r.seats[0].outcome, r.seats[0].stake, r.seats[0].returned), ("bust", 20, 0))

    def test_double_win_lose_push_amounts(self):
        s = play(rigged(["5", "10", "6", "7", "7"]), (0, "double"))  # 5+6+7=18 vs dealer 17
        self.assertEqual(self.outcome(s), ("win", 40))
        s = play(rigged(["5", "10", "5", "8", "7"]), (0, "double"))  # 17 vs 18
        self.assertEqual(self.outcome(s), ("lose", 0))
        s = play(rigged(["5", "10", "5", "8", "8"]), (0, "double"))  # 18 vs 18
        self.assertEqual(self.outcome(s), ("push", 20))

    def test_natural_vs_non_natural_21_dealer(self):
        # dealer 7 4 hits 10 -> 21 (not a natural): player natural still pays 3:2
        s = rigged(["A", "7", "K", "4", "10"])
        self.assertEqual(self.outcome(play(s)), ("blackjack", 25))

    def test_three_card_21_is_not_natural(self):
        s = rigged(["5", "10", "6", "7", "K"])  # 5+6+K = 21 in three cards vs dealer 17
        s = play(s, (0, "hit"))
        self.assertEqual(self.outcome(play(s, (0, "stand"))), ("win", 20))

    def test_summary_text(self):
        s = play(rigged(["10", "10", "8", "7"]), (0, "stand"))
        text = bj.result(s).summary
        self.assertIn("Dealer: 10♠ 7♠ (17)", text)
        self.assertIn("Seat 1: 10♠ 8♠ (18) win", text)


class EdgeTests(unittest.TestCase):
    def test_dealer_natural_with_ace_hole_card(self):
        s = rigged(["9", "K", "9", "A"])
        self.assertTrue(s.finished)
        self.assertEqual(bj.result(s).seats[0].outcome, "lose")

    def test_mixed_table_dealer_still_draws(self):
        s = rigged(["10", "10", "6", "6", "9", "10", "K"], stakes=(10, 10))
        s = play(s, (0, "hit"), (1, "stand"))
        self.assertEqual(s.status, ("bust", "stand"))
        self.assertGreater(len(s.dealer), 2)
        r = bj.result(s)
        self.assertEqual([x.outcome for x in r.seats], ["bust", "win"])

    def test_dealer_multi_ace_draws_to_hard_17(self):
        s = play(rigged(["10", "A", "8", "A", "3", "10"]), (0, "stand"))
        self.assertEqual(bj.hand_total(s.dealer), (17, False))
        self.assertEqual(len(s.dealer), 5)
        self.assertEqual(bj.result(s).seats[0].outcome, "win")

    def test_seat_count_and_indexes_validated(self):
        with self.assertRaises(ValueError):
            bj.new("x", [Seat(i, "member", i, 1) for i in range(bj.MAX_SEATS + 1)])
        with self.assertRaises(ValueError):
            bj.new("x", [Seat(0, "member", 1, 1), Seat(2, "member", 2, 1)])
        bj.new("x", [Seat(i, "member", i, 1) for i in range(bj.MAX_SEATS)])

    def test_repr_does_not_leak_seed_shoe_or_hole_card(self):
        s = bj.new("super-secret-seed", [Seat(0, "member", 1, 5)])
        text = repr(s)
        self.assertNotIn("super-secret-seed", text)
        self.assertNotIn(repr(s.shoe), text)
        self.assertNotIn("shoe", text)
        self.assertNotIn("dealer=", text)

    def test_view_rejects_bad_seat(self):
        s = rigged(["10", "9", "10", "6"])
        for bad in (-1, 1, 5):
            with self.assertRaises(ValueError):
                bj.view(s, bad)
        bj.view(s, 0)
        bj.view(s, None)

    def test_can_double_false_hides_and_refuses_double(self):
        s = rigged(["5", "10", "6", "8", "10"])
        self.assertEqual(bj.legal_moves(s, 0, can_double=False), ("hit", "stand"))
        self.assertEqual(bj.legal_moves(s, 0), ("hit", "stand", "double"))
        with self.assertRaises(IllegalMove):
            bj.apply(s, 0, "double", can_double=False)
        self.assertEqual(bj.apply(s, 0, "double").stakes, (20,))


class TurnTests(unittest.TestCase):
    def test_seats_play_in_order_then_dealer(self):
        s = rigged(["10", "9", "10", "6", "7", "8"], stakes=(1, 1))  # p0 16, p1 16, dealer 10 8? up=10
        self.assertEqual(s.turn, 0)
        s = play(s, (0, "stand"))
        self.assertEqual(s.turn, 1)
        self.assertFalse(s.finished)
        s = play(s, (1, "stand"))
        self.assertTrue(s.finished)

    def test_natural_seat_is_skipped(self):
        s = rigged(["A", "9", "10", "K", "7", "7"], stakes=(1, 1))  # p0 A K natural, p1 9 7
        self.assertEqual(s.turn, 1)

    def test_all_naturals_dealer_does_not_draw(self):
        s = rigged(["A", "6", "K", "5", "9"])  # dealer 6 5 = 11 would draw
        self.assertTrue(s.finished)
        self.assertEqual(len(s.dealer), 2)

    def test_hit_to_21_stays_on_turn(self):
        s = play(rigged(["5", "10", "6", "7", "10"]), (0, "hit"))
        self.assertEqual(bj.hand_total(s.hands[0])[0], 21)
        self.assertEqual(s.turn, 0)
        self.assertEqual(bj.legal_moves(s, 0), ("hit", "stand"))


class IllegalMoveTests(unittest.TestCase):
    def test_double_after_hit(self):
        s = play(rigged(["2", "10", "3", "7", "2"]), (0, "hit"))
        self.assertEqual(bj.legal_moves(s, 0), ("hit", "stand"))
        with self.assertRaises(IllegalMove):
            bj.apply(s, 0, "double")

    def test_not_on_turn(self):
        s = rigged(["10", "9", "10", "6", "7", "8"], stakes=(1, 1))
        with self.assertRaises(IllegalMove):
            bj.apply(s, 1, "stand")
        self.assertEqual(bj.legal_moves(s, 1), ())

    def test_after_finish_and_unknown_move(self):
        s = rigged(["10", "9", "10", "6"])
        with self.assertRaises(IllegalMove):
            bj.apply(s, 0, "split")
        s = play(s, (0, "stand"))
        with self.assertRaises(IllegalMove):
            bj.apply(s, 0, "hit")
        self.assertEqual(bj.legal_moves(s, 0), ())

    def test_state_is_not_mutated(self):
        s = rigged(["10", "9", "10", "6"])
        before = (s.hands, s.pos, s.status)
        bj.apply(s, 0, "hit")
        self.assertEqual((s.hands, s.pos, s.status), before)

    def test_result_requires_finished(self):
        with self.assertRaises(IllegalMove):
            bj.result(rigged(["10", "9", "10", "6"]))

    def test_new_validates_seats(self):
        with self.assertRaises(ValueError):
            bj.new("x", [])
        with self.assertRaises(ValueError):
            bj.new("x", [Seat(1, "member", 1, 5)])


class ViewTests(unittest.TestCase):
    def test_hole_card_and_shoe_hidden_until_finished(self):
        s = rigged(["10", "9", "10", "6"])
        v = bj.view(s, 0)
        self.assertEqual(v.dealer, (c("9"),))
        self.assertTrue(v.dealer_hidden)
        self.assertIsNone(v.dealer_total)
        self.assertIsNone(v.seed)
        self.assertEqual(v.seed_hash, bj.short_seed_hash("rigged"))
        self.assertEqual(v.hands, s.hands)
        self.assertEqual((v.total, v.soft), (20, False))
        self.assertEqual(v.legal, ("hit", "stand", "double"))
        self.assertFalse(hasattr(v, "shoe"))
        hole = s.dealer[1]
        self.assertNotIn(hole, v.dealer)
        s = play(s, (0, "stand"))
        v = bj.view(s, 0)
        self.assertEqual(v.dealer, s.dealer)
        self.assertFalse(v.dealer_hidden)
        self.assertEqual(v.seed, "rigged")

    def test_spectator_and_other_seat_views(self):
        s = rigged(["10", "9", "10", "6", "7", "8"], stakes=(1, 1))
        spec = bj.view(s, None)
        self.assertIsNone(spec.total)
        self.assertEqual(spec.legal, ())
        self.assertEqual(spec.hands, s.hands)
        self.assertEqual(spec.turn, 0)
        self.assertEqual(bj.view(s, 1).legal, ())

    def test_dealer_natural_reveals_hand(self):
        v = bj.view(rigged(["10", "A", "9", "K"]), None)
        self.assertTrue(v.finished)
        self.assertEqual(len(v.dealer), 2)


class ReplayTests(unittest.TestCase):
    def test_replay_equals_step_by_step(self):
        seats = [Seat(0, "member", 1, 10), Seat(1, "member", 2, 5)]
        seed = "replay-seed"
        state = bj.new(seed, seats)
        log = []
        policy = DefaultPolicy()
        while not state.finished:
            seat = state.turn
            move = policy.pick(bj.view(state, seat), bj.legal_moves(state, seat))
            state = bj.apply(state, seat, move)
            log.append((seat, move))
        self.assertEqual(bj.replay(seed, seats, log), state)
        partial = bj.replay(seed, seats, log[:1])
        self.assertEqual(partial, bj.apply(bj.new(seed, seats), *log[0]))

    def test_replay_rejects_bad_log(self):
        seats = [Seat(0, "member", 1, 10)]
        with self.assertRaises(IllegalMove):
            bj.replay("seed", seats, [(0, "stand"), (0, "hit")])


class DefaultPolicyTests(unittest.TestCase):
    def test_only_legal_moves_across_many_seeds(self):
        seats = [Seat(0, "member", 1, 10), Seat(1, "character", 2, 10)]
        for n in range(200):
            state = bj.new(f"seed-{n}", seats)
            while not state.finished:
                seat = state.turn
                legal = bj.legal_moves(state, seat)
                move = DefaultPolicy.pick(bj.view(state, seat), legal)
                self.assertIn(move, legal)
                self.assertNotEqual(move, "double")
                state = bj.apply(state, seat, move)
            r = bj.result(state)
            for sr in r.seats:
                self.assertGreaterEqual(sr.returned, 0)


if __name__ == "__main__":
    unittest.main()
