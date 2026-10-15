"""FEAT-60 (D42): the pure Yut policy (``yut.policy_move``, ``yut.choices``, ``yut.preview``).

A policy is deterministic in (table, seat, Play style, seed), reads only what a ``View`` also has (Yut hides nothing), never forfeits,
never returns an illegal move, throws when it is time to throw, and scores moves: a capture or a piece home beats advancing, a bold style
takes shortcuts, a cautious one stacks. Offline; rigged states stand in for deals.
"""
import dataclasses
import unittest

from llmcord_core.games import IllegalMove, Seat, playstyle, yut

BOLD = playstyle.PlayStyle(aggression=100, risk=100, unpredictability=0)
CAUTIOUS = playstyle.PlayStyle(aggression=0, risk=0, unpredictability=0)
ERRATIC = playstyle.PlayStyle(unpredictability=100)


def seats(n):
    return [Seat(i, "member", 100 + i, 0) for i in range(n)]


def ring(point):
    """The trail of a piece that walked the ring to ``point`` (P1..P19 only)."""
    return tuple(f"P{i}" for i in range(1, int(point[1:]) + 1))


def rig(rows, bank, *, seed="policy", turn=0, pieces=3):
    """A state in the move phase: ``rows`` is each seat's piece positions (ring points, None or "done"), ``bank`` the banked throws."""
    state = yut.new(seed, seats(len(rows)), {"pieces": pieces})
    return dataclasses.replace(
        state, pieces=tuple(tuple(r) for r in rows), trails=tuple(tuple(ring(p) if p not in (None, "done") else () for p in r) for r in rows),
        turn=turn, bank=tuple(bank), phase="move", to_throw=0)


class ChoicesTests(unittest.TestCase):
    def test_a_throw_is_the_only_choice_in_the_throw_phase(self):
        self.assertEqual(yut.choices(yut.new("s", seats(2)), 0), ("throw",))

    def test_pieces_that_move_alike_are_offered_once_and_forfeit_never(self):
        state = rig([[None, None, "P3"], [None, None, None]], ["gae"])
        self.assertEqual(yut.choices(state, 0), ("move:0:gae", "move:2:gae"))
        self.assertIn("forfeit", yut.legal_moves(state, 0))

    def test_nothing_is_offered_off_turn_and_a_view_works_like_the_state(self):
        state = rig([[None, "P3", None], [None, None, None]], ["gae"])
        self.assertEqual(yut.choices(state, 1), ())
        self.assertEqual(yut.choices(yut.view(state, 0), 0), yut.choices(state, 0))
        self.assertEqual(yut.legal_moves(yut.view(state, 0), 0), yut.legal_moves(state, 0))


class PreviewTests(unittest.TestCase):
    def test_a_capture_a_stack_and_going_home_are_reported_without_moving(self):
        state = rig([["P1", "P18", "P18"], ["P3", None, None]], ["gae", "geol"])
        capture = yut.preview(state, 0, "move:0:gae")
        self.assertEqual((capture["start"], capture["end"], capture["captured"], capture["moving"], capture["home"]), ("P1", "P3", 1, 1, False))
        home = yut.preview(state, 0, "move:1:geol")
        self.assertEqual((home["end"], home["moving"], home["home"], home["captured"]), ("done", 2, True, 0))
        self.assertEqual(state.pieces[0], ("P1", "P18", "P18"))

    def test_a_back_do_from_the_first_step_goes_back_to_the_start(self):
        back = yut.preview(rig([["P1", None, None], [None, None, None]], ["backdo"]), 0, "move:0:backdo")
        self.assertIsNone(back["end"])


class PolicyTests(unittest.TestCase):
    def test_the_throw_is_made_when_it_is_time_to_throw(self):
        self.assertEqual(yut.policy_move(yut.new("s", seats(3)), 0, BOLD, "x"), "throw")

    def test_off_turn_is_an_illegal_move(self):
        with self.assertRaises(IllegalMove):
            yut.policy_move(yut.new("s", seats(2)), 1)

    def test_it_prefers_a_capture(self):
        state = rig([["P1", "P4", None], ["P3", None, None]], ["gae"])
        for style in (None, BOLD, CAUTIOUS):
            self.assertEqual(yut.policy_move(state, 0, style, "x"), "move:0:gae", style)

    def test_it_prefers_bringing_a_piece_home(self):
        state = rig([["P18", "P4", None], [None, None, None]], ["geol"])
        self.assertEqual(yut.policy_move(state, 0), "move:0:geol")

    def test_it_advances_the_front_piece_when_nothing_else_stands_out(self):
        state = rig([["P7", "P11", None], [None, None, None]], ["do"])
        self.assertEqual(yut.policy_move(state, 0), "move:1:do")

    def test_it_does_not_step_a_piece_back_when_it_can_advance(self):
        state = rig([["P7", "P12", None], [None, None, None]], ["backdo", "gae"])
        self.assertEqual(yut.policy_move(state, 0), "move:1:gae")

    def test_a_bold_style_takes_the_shortcut_and_a_cautious_one_stacks(self):
        state = rig([["P2", "P5", "P4"], [None, None, None]], ["gae"])
        self.assertEqual(yut.policy_move(state, 0, BOLD, "x"), "move:1:gae")
        self.assertEqual(yut.policy_move(state, 0, CAUTIOUS, "x"), "move:0:gae")

    def test_a_cautious_style_avoids_a_landing_an_opponent_can_hit(self):
        # Piece 0 P6 -> P8 would stand 2 steps ahead of the opponent's P6... the opponent at P4 hits P8 with a yut; piece 1 P12 -> P14 is out of reach.
        state = rig([["P6", "P12", None], ["P4", None, None]], ["gae"])
        self.assertEqual(yut.policy_move(state, 0, CAUTIOUS, "x"), "move:1:gae")
        self.assertGreater(yut._threat(state, 0, "P8"), 0)
        self.assertEqual(yut._threat(state, 0, "P14"), 0)

    def test_it_never_forfeits_and_always_returns_a_legal_move(self):
        state = rig([["P3", "P8", None], ["P5", "P6", "P12"]], ["do", "yut", "mo"])
        for seed in range(100):
            for style in (None, BOLD, CAUTIOUS, ERRATIC):
                move = yut.policy_move(state, 0, style, f"s{seed}")
                self.assertIn(move, yut.legal_moves(state, 0))
                self.assertNotEqual(move, "forfeit")

    def test_it_is_deterministic_and_reads_a_view_like_the_state(self):
        state = rig([["P3", "P8", None], ["P5", "P6", "P12"]], ["do", "yut"])
        for style in (None, BOLD, ERRATIC):
            first = yut.policy_move(state, 0, style, "seed-1")
            self.assertEqual(first, yut.policy_move(state, 0, style, "seed-1"))
            self.assertEqual(first, yut.policy_move(yut.view(state, 0), 0, style, "seed-1"))

    def test_an_erratic_style_sometimes_deviates_a_steady_one_never(self):
        state = rig([["P3", "P8", None], ["P5", "P6", "P12"]], ["do", "yut"])
        steady = {yut.policy_move(state, 0, BOLD, f"s{i}") for i in range(60)}
        erratic = {yut.policy_move(state, 0, ERRATIC, f"s{i}") for i in range(60)}
        self.assertEqual(len(steady), 1)
        self.assertGreater(len(erratic), 1)

    def test_without_a_seed_the_state_seed_and_history_make_one(self):
        state = rig([["P3", "P8", None], ["P5", "P6", "P12"]], ["do", "yut"])
        self.assertEqual(yut.policy_move(state, 0, ERRATIC), yut.policy_move(state, 0, ERRATIC))


class FullGameTests(unittest.TestCase):
    def play(self, seed, n, style=None):
        state, count = yut.new(seed, seats(n)), 0
        while not state.finished:
            seat = state.turn
            move = yut.policy_move(state, seat, style if seat % 2 == 0 else None, f"{seed}/{count}")
            self.assertIn(move, yut.legal_moves(state, seat))
            self.assertNotEqual(move, "forfeit")
            state = yut.apply(state, seat, move)
            count += 1
            self.assertLess(count, 5000, f"game {seed} does not end")
        return state

    def test_200_seeded_games_with_policy_seats_run_to_the_end(self):
        winners = set()
        for i in range(200):
            state = self.play(f"full-{i}", 2 + i % 3, BOLD if i % 2 else CAUTIOUS)
            self.assertTrue(all(p == "done" for p in state.pieces[state.winner]))
            winners.add(state.winner)
        self.assertGreater(len(winners), 1)

    def test_the_same_seed_plays_the_same_game(self):
        self.assertEqual(yut.to_json(self.play("again", 3)), yut.to_json(self.play("again", 3)))


if __name__ == "__main__":
    unittest.main()
