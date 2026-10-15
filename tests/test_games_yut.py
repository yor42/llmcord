"""Yut (윷놀이) rules engine (llmcord_core/games/yut.py), FEAT-56, decision D42.

These tests define the engine API. Contract the tests choose where the brief leaves a detail open:

Module: MIN_SEATS = 2, MAX_SEATS = 4, P_FLAT_UP, DEFAULT_RULES, THROWS = ("do", "gae", "geol", "yut", "mo", "backdo"),
  throw_result(seed, n, backdo=True) -> (name, sticks), normalize_rules, new(seed, seats, rules=None), legal_moves,
  apply, view, board_view, result, replay, to_json, from_json, and IllegalMove from games.base.
  ValueError (UserError) for bad rules, a seat count outside 2..4 and seat indexes that are not 0..n-1.
Point ids are strings: "P0".."P19" (ring, P0 = start/finish), "a1","a2" (diagonal A, P5 -> C), "b1","b2" (C -> P15),
  "c1","c2" (diagonal B, P10 -> C), "d1","d2" (C -> P0), "C" (center). A piece position is a point id, None (unstarted)
  or "done" (finished).
State (frozen dataclass; seed repr=False). Fields used by the tests:
  seed, seats, rules, pieces (tuple per seat of tuple per piece of position), trails (same shape: tuple of every
  single-step point id stepped through since the start; (), for an unstarted or finished piece; entering with k steps
  gives ("P1", ..., "Pk"); a piece standing on P0 has trail ending in "P0"), bank (tuple of banked throw names in the
  order thrown), phase ("throw" | "move"), to_throw (throws the turn seat still owes now; 1 at the start of a turn),
  throws_made (the n for the next throw; apply of "throw" uses it and increments it), turn (None when finished),
  active (tuple[bool]), last_throw (None or (name, sticks)), history (tuple; every apply that changes the game appends
  at least one event), finished, winner.
  A fresh state: phase "throw", to_throw 1, bank (), throws_made 0, turn 0, all pieces None.
  A capture in the move phase sets phase "throw" and to_throw 1 (unless the capturing throw is a yut or mo); after the
  extra throw is made the phase is "move" again when the bank is not empty. When the bank is empty and nothing is owed
  the turn passes: next active seat, phase "throw", to_throw 1, bank ().
  A backdo that no piece on the board can take is dropped from the bank as soon as it cannot be used.
legal_moves(state, seat): tuple of strings; "throw" (turn seat, phase throw), "move:<piece>:<throw>" for EVERY movable
  piece index (each member of a stack is listed, unstarted pieces too, finished pieces never) and each distinct banked
  throw name (turn seat, phase move), "forfeit" for every active seat while the game runs; () for an inactive seat or a
  finished game.
Stacks: pieces of one seat on the same point. Moving any member moves all members; trails extend together.
Forfeit: the seat becomes inactive, all its pieces become None (off the board), trails (); a forfeit by the turn seat
  passes the turn (phase "throw", to_throw 1, bank ()); a lone active seat wins (finished, winner, turn None).
board_view(state) -> dict with keys "pieces" (same shape as state.pieces), "stacks" (tuple of (point, seat, piece
  indexes) for every stack of 2+ on a board point), "bank", "to_throw", "phase", "turn", "last_throw" (None or
  {"name": str, "sticks": tuple[bool]}).
view(state, seat=None) -> object with seat, active, turn, phase, pieces, trails, bank, to_throw, last_throw, history,
  finished, winner; ValueError for a seat that does not exist.
result(state) -> Result(finished, winner, seats=(SeatResult(seat, outcome, payout)), pot).
"""
import dataclasses
import json
import math
import unittest
from collections import Counter

from llmcord_core.games import IllegalMove, Seat, playstyle
from llmcord_core.games import yut

OUTER = tuple(f"P{i}" for i in range(1, 20)) + ("P0",)
PATH_OUT = OUTER
PATH_A = tuple(f"P{i}" for i in range(1, 6)) + ("a1", "a2", "C", "b1", "b2", "P15", "P16", "P17", "P18", "P19", "P0")
PATH_B = tuple(f"P{i}" for i in range(1, 11)) + ("c1", "c2", "C", "d1", "d2", "P0")
STEPS = {"do": 1, "gae": 2, "geol": 3, "yut": 4, "mo": 5}


def seats(n, stake=0):
    return [Seat(i, "member", 100 + i, stake) for i in range(n)]


def mk(n=2, seed="s", stake=0, rules=None):
    return yut.new(seed, seats(n, stake), rules)


def default_path(pos):
    if pos in ("a1", "a2", "b1", "b2", "C"):
        return PATH_A
    if pos in ("c1", "c2", "d1", "d2"):
        return PATH_B
    return PATH_OUT


def at(pos, path=None):
    """A piece spec: standing on pos having walked path up to it."""
    return (pos, path or default_path(pos))


def stage(n=2, pieces=None, bank=(), phase="move", to_throw=0, turn=0, rules=None, stake=0, throws_made=0, seed="stage",
          active=None):
    """A state with chosen pieces (seat -> list of None | "done" | point | (point, path)); the rest are unstarted."""
    s = mk(n, seed, stake, rules)
    count = s.rules["pieces"]
    pos, trails = [], []
    for seat in range(n):
        specs = list((pieces or {}).get(seat, ()))
        specs += [None] * (count - len(specs))
        p, t = [], []
        for spec in specs:
            if spec is None or spec == "done":
                p.append(spec)
                t.append(())
                continue
            point, path = spec if isinstance(spec, tuple) else at(spec)
            p.append(point)
            t.append(path[:path.index(point) + 1])
        pos.append(tuple(p))
        trails.append(tuple(t))
    return dataclasses.replace(
        s, pieces=tuple(pos), trails=tuple(trails), bank=tuple(bank), phase=phase, to_throw=to_throw, turn=turn,
        throws_made=throws_made, active=tuple(active) if active is not None else (True,) * n)


def go(start, throw, seat=0, piece=0, **kw):
    """Move one piece from `start` by `throw` on an otherwise empty board; returns the new state."""
    s = stage(pieces={seat: [start]}, bank=(throw,), turn=seat, **kw)
    return yut.apply(s, seat, f"move:{piece}:{throw}")


def where(state, seat=0, piece=0):
    return state.pieces[seat][piece]


_SEEDS = {}


def find_seed(names, start=0, backdo=True):
    """A seed whose throws start, start+1, ... come out as `names`."""
    key = (tuple(names), start, backdo)
    if key not in _SEEDS:
        for k in range(400000):
            seed = f"find{k}"
            if all(yut.throw_result(seed, start + i, backdo)[0] == name for i, name in enumerate(names)):
                _SEEDS[key] = seed
                break
        else:
            raise AssertionError(f"no seed for {names}")
    return _SEEDS[key]


def throwing(names, **kw):
    """A fresh-turn state (phase throw, to_throw 1) whose next throws are `names`."""
    return stage(phase="throw", to_throw=1, seed=find_seed(names), **kw)


def legal(state, seat):
    return set(yut.legal_moves(state, seat))


class ThrowTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(yut.P_FLAT_UP, math.pi / (2 + math.pi))
        self.assertEqual((yut.MIN_SEATS, yut.MAX_SEATS), (2, 4))

    def test_sticks_and_names_follow_the_playstyle_roll(self):
        names = {0: "mo", 1: "do", 2: "gae", 3: "geol", 4: "yut"}
        for seed in ("a", "b", "c"):
            for n in range(40):
                name, sticks = yut.throw_result(seed, n)
                expected = tuple(playstyle.roll(f"{seed}/throw/{n}", f"stick{i}") < yut.P_FLAT_UP for i in range(4))
                self.assertEqual(sticks, expected)
                self.assertIsInstance(sticks, tuple)
                flat = sum(expected)
                if flat == 1 and expected[0]:
                    self.assertEqual(name, "backdo")
                else:
                    self.assertEqual(name, names[flat])

    def test_distribution(self):
        counts = Counter(yut.throw_result("dist", n)[0] for n in range(20000))
        want = {"mo": 2.3, "do": 10.8, "backdo": 3.6, "gae": 33.9, "geol": 35.5, "yut": 13.9}
        self.assertEqual(set(counts), set(want))
        for name, pct in want.items():
            self.assertAlmostEqual(counts[name] / 200, pct, delta=1.2, msg=name)

    def test_backdo_off_turns_it_into_do(self):
        found = 0
        for n in range(400):
            on = yut.throw_result("bd", n, True)
            off = yut.throw_result("bd", n, False)
            self.assertNotEqual(off[0], "backdo")
            self.assertEqual(on[1], off[1])
            if on[0] == "backdo":
                found += 1
                self.assertEqual(off[0], "do")
            else:
                self.assertEqual(on[0], off[0])
        self.assertGreater(found, 3)

    def test_determinism_and_dependence_on_seed_and_index(self):
        self.assertEqual(yut.throw_result("x", 3), yut.throw_result("x", 3))
        seq = [yut.throw_result("x", n)[1] for n in range(30)]
        self.assertGreater(len(set(seq)), 4)
        self.assertNotEqual(seq, [yut.throw_result("y", n)[1] for n in range(30)])


class RulesTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(yut.normalize_rules(), {"backdo": True, "pieces": 4, "ante": 0, "turn_timeout": 60})
        self.assertEqual(yut.normalize_rules({}), yut.normalize_rules(None))
        self.assertEqual(yut.normalize_rules(yut.DEFAULT_RULES), yut.normalize_rules())

    def test_partial_override_json_string_and_unknown_keys_dropped(self):
        got = yut.normalize_rules(json.dumps({"backdo": False, "pieces": 2, "bogus": 1}))
        self.assertEqual(got, {"backdo": False, "pieces": 2, "ante": 0, "turn_timeout": 60})

    def test_boundaries_are_accepted(self):
        for ok in ({"pieces": 2}, {"pieces": 5}, {"ante": 0}, {"ante": 10 ** 9}, {"turn_timeout": 15}, {"turn_timeout": 300}):
            yut.normalize_rules(ok)

    def test_bad_values_raise(self):
        for bad in ({"backdo": "yes"}, {"backdo": 1}, {"backdo": None}, {"pieces": 1}, {"pieces": 6}, {"pieces": True},
                    {"pieces": "4"}, {"pieces": 3.0}, {"ante": -1}, {"ante": 10 ** 9 + 1}, {"ante": True}, {"ante": 1.5},
                    {"turn_timeout": 14}, {"turn_timeout": 301}, {"turn_timeout": 60.5}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                yut.normalize_rules(bad)

    def test_non_object_values_raise(self):
        for bad in ([], 5, "not json", "[1]"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                yut.normalize_rules(bad)

    def test_new_snapshots_normalized_rules_and_rejects_bad_ones(self):
        self.assertEqual(mk(2, rules={"pieces": 3}).rules["pieces"], 3)
        with self.assertRaises(ValueError):
            mk(2, rules={"pieces": 9})


class NewTests(unittest.TestCase):
    def test_opening_state(self):
        s = mk(3)
        self.assertEqual(s.pieces, ((None,) * 4,) * 3)
        self.assertEqual((s.turn, s.phase, s.to_throw, s.bank, s.throws_made, s.finished, s.winner),
                         (0, "throw", 1, (), 0, False, None))
        self.assertEqual(s.active, (True,) * 3)
        self.assertEqual(yut.new("s", seats(2), {"pieces": 2}).pieces, ((None, None),) * 2)

    def test_seat_count_limits_and_indexes(self):
        for n in (0, 1, 5):
            with self.assertRaises(ValueError):
                yut.new("s", seats(n))
        for n in (2, 3, 4):
            yut.new("s", seats(n))
        with self.assertRaises(ValueError):
            yut.new("s", [Seat(0, "member", 1, 0), Seat(2, "member", 2, 0)])

    def test_repr_hides_the_seed(self):
        self.assertNotIn("very-secret-seed", repr(mk(2, "very-secret-seed")))

    def test_legal_moves_at_the_start(self):
        s = mk(3)
        self.assertEqual(legal(s, 0), {"throw", "forfeit"})
        self.assertEqual(legal(s, 1), {"forfeit"})
        with self.assertRaises(ValueError):
            yut.legal_moves(s, 9)


class ThrowPhaseTests(unittest.TestCase):
    def test_a_plain_throw_banks_and_moves_to_the_move_phase(self):
        s = throwing(["gae"])
        t = yut.apply(s, 0, "throw")
        self.assertEqual((t.bank, t.phase, t.to_throw, t.turn, t.throws_made), (("gae",), "move", 0, 0, 1))
        self.assertEqual(t.last_throw, ("gae", yut.throw_result(s.seed, 0)[1]))
        self.assertGreater(len(t.history), len(s.history))
        self.assertEqual(s.bank, ())  # the old state is untouched

    def test_yut_and_mo_bank_and_throw_again(self):
        for first in ("yut", "mo"):
            with self.subTest(first):
                s = throwing([first, "geol"])
                t = yut.apply(s, 0, "throw")
                self.assertEqual((t.bank, t.phase, t.to_throw, t.throws_made), ((first,), "throw", 1, 1))
                self.assertEqual(legal(t, 0), {"throw", "forfeit"})
                u = yut.apply(t, 0, "throw")
                self.assertEqual((u.bank, u.phase, u.to_throw, u.turn), ((first, "geol"), "move", 0, 0))

    def test_chain_of_extra_throws(self):
        s = throwing(["yut", "yut", "do"])
        for _ in range(3):
            s = yut.apply(s, 0, "throw")
        self.assertEqual((s.bank, s.phase, s.throws_made), (("yut", "yut", "do"), "move", 3))

    def test_throw_uses_the_throw_index(self):
        s = throwing(["do"])
        later = dataclasses.replace(s, throws_made=5)
        self.assertEqual(yut.apply(later, 0, "throw").last_throw[0], yut.throw_result(s.seed, 5)[0])

    def test_backdo_rule_off_gives_do(self):
        n = next(n for n in range(2000) if yut.throw_result("off", n)[0] == "backdo")
        s = stage(phase="throw", to_throw=1, seed="off", throws_made=n, rules={"backdo": False})
        self.assertEqual(yut.apply(s, 0, "throw").bank, ("do",))

    def test_backdo_with_no_piece_on_the_board_passes_the_turn(self):
        s = throwing(["backdo"])
        t = yut.apply(s, 0, "throw")
        self.assertEqual((t.turn, t.phase, t.to_throw, t.bank), (1, "throw", 1, ()))

    def test_backdo_with_only_finished_pieces_passes_the_turn(self):
        s = throwing(["backdo"], pieces={0: ["done", "done", None, None]})
        self.assertEqual(yut.apply(s, 0, "throw").turn, 1)

    def test_backdo_is_kept_when_a_piece_can_take_it(self):
        s = throwing(["backdo"], pieces={0: ["P3"]})
        t = yut.apply(s, 0, "throw")
        self.assertEqual((t.turn, t.phase, t.bank), (0, "move", ("backdo",)))

    def test_backdo_after_a_yut_with_nothing_on_the_board_is_dropped_but_the_yut_stays(self):
        s = throwing(["yut", "backdo"])
        t = yut.apply(yut.apply(s, 0, "throw"), 0, "throw")
        self.assertEqual((t.turn, t.phase, t.bank), (0, "move", ("yut",)))
        self.assertNotIn("move:0:backdo", legal(t, 0))


class RouteTests(unittest.TestCase):
    def check(self, table):
        for start, throw, want in table:
            with self.subTest(start=start, throw=throw):
                self.assertEqual(where(go(start, throw)), want)

    def test_entering_from_the_start(self):
        s = stage(bank=("gae",))
        t = yut.apply(s, 0, "move:2:gae")
        self.assertEqual(t.pieces[0], (None, None, "P2", None))
        self.assertEqual(t.trails[0][2], ("P1", "P2"))
        self.check([(None, "do", "P1"), (None, "geol", "P3"), (None, "yut", "P4"), (None, "mo", "P5")])

    def test_plain_ring_moves(self):
        self.check([("P1", "gae", "P3"), ("P11", "yut", "P15"), ("P16", "geol", "P19"), ("P12", "do", "P13")])

    def test_stopping_on_a_corner_then_turning_into_the_diagonal(self):
        self.check([("P3", "gae", "P5"), ("P8", "gae", "P10"), ("P13", "gae", "P15"),
                    ("P5", "do", "a1"), ("P5", "gae", "a2"), ("P5", "geol", "C"), ("P5", "yut", "b1"), ("P5", "mo", "b2"),
                    ("P10", "do", "c1"), ("P10", "gae", "c2"), ("P10", "geol", "C"), ("P10", "yut", "d1"), ("P10", "mo", "d2")])

    def test_passing_a_corner_goes_straight_on(self):
        self.check([("P4", "gae", "P6"), ("P3", "yut", "P7"), ("P9", "gae", "P11"), ("P8", "geol", "P11"),
                    ("P14", "gae", "P16"), ("P13", "geol", "P16"), ("P2", "mo", "P7")])

    def test_corner_p15_continues_on_the_ring(self):
        self.check([("P15", "do", "P16"), ("P15", "mo", "P0")])

    def test_arriving_on_a_corner_by_the_diagonal_does_not_turn_again(self):
        self.check([("b2", "do", "P15"), ("b2", "gae", "P16"), ("b1", "mo", "P18"),
                    ("a2", "mo", "P16"), ("a1", "mo", "P15")])

    def test_diagonal_a_passes_through_the_center_to_b_and_p15(self):
        self.check([("a1", "do", "a2"), ("a1", "gae", "C"), ("a1", "geol", "b1"), ("a1", "yut", "b2"),
                    ("a2", "do", "C"), ("a2", "gae", "b1"), ("a2", "geol", "b2"), ("a2", "yut", "P15")])

    def test_center_from_the_b_diagonal_goes_home_not_to_b(self):
        self.check([("c1", "do", "c2"), ("c1", "gae", "C"), ("c1", "geol", "d1"), ("c1", "yut", "d2"), ("c1", "mo", "P0"),
                    ("c2", "do", "C"), ("c2", "gae", "d1"), ("c2", "geol", "d2"), ("c2", "yut", "P0"), ("c2", "mo", "done")])

    def test_stopping_at_the_center_goes_home_whichever_diagonal_it_came_by(self):
        for path in (PATH_A, PATH_B):
            for throw, want in (("do", "d1"), ("gae", "d2"), ("geol", "P0"), ("yut", "done"), ("mo", "done")):
                with self.subTest(path=path[0] + path[-1], throw=throw):
                    s = stage(pieces={0: [at("C", path)]}, bank=(throw,))
                    self.assertEqual(where(yut.apply(s, 0, f"move:0:{throw}")), want)

    def test_home_stretch_d1_d2(self):
        self.check([("d1", "do", "d2"), ("d1", "gae", "P0"), ("d1", "geol", "done"), ("d2", "do", "P0"), ("d2", "gae", "done")])

    def test_stopping_exactly_on_p0_does_not_finish_and_the_next_move_does(self):
        t = go("P17", "geol")
        self.assertEqual(where(t), "P0")
        self.assertFalse(t.finished)
        self.assertEqual(t.trails[0][0][-1], "P0")
        s = dataclasses.replace(t, bank=("do",), phase="move", to_throw=0, turn=0)
        self.assertEqual(where(yut.apply(s, 0, "move:0:do")), "done")

    def test_overshooting_p0_finishes_without_an_exact_roll(self):
        self.check([("P18", "yut", "done"), ("P19", "gae", "done"), ("P16", "mo", "done"), ("P18", "gae", "P0"),
                    ("P19", "do", "P0"), ("P0", "do", "done"), ("P0", "mo", "done")])

    def test_trail_grows_one_point_per_step(self):
        s = stage(pieces={0: ["P3"]}, bank=("geol",))
        t = yut.apply(s, 0, "move:0:geol")
        self.assertEqual(t.trails[0][0], ("P1", "P2", "P3", "P4", "P5", "P6"))
        self.assertEqual(where(t), "P6")

    def test_a_move_through_the_center_records_every_point(self):
        t = go("a1", "yut")
        self.assertEqual(t.trails[0][0][-4:], ("a2", "C", "b1", "b2"))


class BackdoTests(unittest.TestCase):
    def back(self, start, count=1, **kw):
        s = stage(pieces={0: [start]}, bank=("backdo",) * count, **kw)
        for _ in range(count):
            s = yut.apply(s, 0, "move:0:backdo")
        return s

    def test_unstarted_pieces_cannot_take_a_backdo(self):
        s = stage(bank=("backdo", "gae"))
        self.assertFalse([m for m in legal(s, 0) if m.endswith(":backdo")])
        with self.assertRaises(IllegalMove):
            yut.apply(s, 0, "move:0:backdo")

    def test_finished_pieces_cannot_take_a_backdo(self):
        s = stage(pieces={0: ["done", "P3"]}, bank=("backdo",))
        self.assertEqual({m for m in legal(s, 0) if m.startswith("move:")}, {"move:1:backdo"})

    def test_only_the_pieces_on_the_board_are_listed(self):
        s = stage(pieces={0: ["P3", "P8"]}, bank=("backdo",))
        self.assertEqual({m for m in legal(s, 0) if m.startswith("move:")}, {"move:0:backdo", "move:1:backdo"})

    def test_none_on_board_leaves_a_move_phase_without_backdo_moves(self):
        s = stage(bank=("backdo",))
        with self.assertRaises(IllegalMove):
            yut.apply(s, 0, "move:0:backdo")

    def test_from_the_first_point_back_to_the_start_not_finished(self):
        t = self.back("P1")
        self.assertIsNone(where(t))
        self.assertEqual(t.trails[0][0], ())
        self.assertNotEqual(where(t), "done")
        self.assertFalse(t.finished)

    def test_a_piece_back_at_the_start_can_enter_again_and_cannot_back_up(self):
        t = self.back("P1")
        s = dataclasses.replace(t, bank=("backdo", "do"), turn=0, phase="move", to_throw=0)
        self.assertFalse([m for m in legal(s, 0) if m.endswith(":backdo")])
        self.assertEqual(where(yut.apply(s, 0, "move:0:do")), "P1")

    def test_ring_points_step_back_one(self):
        for start, want in (("P3", "P2"), ("P12", "P11"), ("P19", "P18"), ("P6", "P5"), ("P11", "P10")):
            with self.subTest(start):
                self.assertEqual(where(self.back(start)), want)

    def test_trail_loses_exactly_the_last_point(self):
        s = stage(pieces={0: ["P4"]}, bank=("backdo",))
        before = s.trails[0][0]
        self.assertEqual(yut.apply(s, 0, "move:0:backdo").trails[0][0], before[:-1])

    def test_corners_retrace_their_approach(self):
        for start, want in (("P5", "P4"), ("P10", "P9"), ("P15", "P14")):
            with self.subTest(start):
                self.assertEqual(where(self.back(start)), want)

    def test_back_from_the_first_diagonal_point_returns_to_the_corner(self):
        self.assertEqual(where(self.back("a1")), "P5")
        self.assertEqual(where(self.back("c1")), "P10")

    def test_diagonal_retrace_through_the_center(self):
        self.assertEqual(where(self.back("b1")), "C")
        self.assertEqual(where(self.back("d1")), "C")
        self.assertEqual(where(self.back("b1", 2)), "a2")
        self.assertEqual(where(self.back("d1", 2)), "c2")

    def test_center_goes_back_the_way_it_came(self):
        self.assertEqual(where(self.back("C")), "a2")  # arrived by diagonal A
        s = stage(pieces={0: [at("C", PATH_B)]}, bank=("backdo",))
        self.assertEqual(where(yut.apply(s, 0, "move:0:backdo")), "c2")

    def test_retrace_from_the_ring_after_the_diagonal(self):
        s = stage(pieces={0: [at("P16", PATH_A)]}, bank=("backdo",))
        t = yut.apply(s, 0, "move:0:backdo")
        self.assertEqual(where(t), "P15")
        t = dataclasses.replace(t, bank=("backdo",), phase="move", to_throw=0, turn=0)
        self.assertEqual(where(yut.apply(t, 0, "move:0:backdo")), "b2")

    def test_stopped_on_p0_goes_back_the_way_it_came(self):
        self.assertEqual(where(self.back(at("P0", PATH_OUT))), "P19")
        self.assertEqual(where(self.back(at("P0", PATH_B))), "d2")

    def test_repeated_backdo_walks_back_to_the_start(self):
        s = stage(pieces={0: ["P3"]}, bank=("backdo", "backdo"))
        s = yut.apply(s, 0, "move:0:backdo")
        self.assertEqual((where(s), s.bank, s.turn), ("P2", ("backdo",), 0))
        s = yut.apply(s, 0, "move:0:backdo")
        self.assertEqual((where(s), s.bank), ("P1", ()))
        self.assertEqual(s.turn, 1)

    def test_backdo_can_capture_and_earns_one_extra_throw(self):
        s = stage(pieces={0: ["P5"], 1: ["P4"]}, bank=("backdo",))
        t = yut.apply(s, 0, "move:0:backdo")
        self.assertEqual((where(t), where(t, 1)), ("P4", None))
        self.assertEqual((t.phase, t.to_throw, t.turn), ("throw", 1, 0))

    def test_unused_backdo_is_discarded_when_the_last_piece_is_gone(self):
        s = stage(pieces={0: ["P1"]}, bank=("backdo", "backdo"))
        s = yut.apply(s, 0, "move:0:backdo")
        self.assertEqual((s.turn, s.bank, s.phase), (1, (), "throw"))


class CaptureTests(unittest.TestCase):
    def test_landing_on_an_opponent_captures_and_earns_an_extra_throw(self):
        s = stage(pieces={0: ["P3"], 1: ["P5", "P8"]}, bank=("gae",))
        t = yut.apply(s, 0, "move:0:gae")
        self.assertEqual(where(t), "P5")
        self.assertEqual(t.pieces[1], (None, "P8", None, None))
        self.assertEqual(t.trails[1][0], ())
        self.assertEqual((t.phase, t.to_throw, t.turn, t.bank), ("throw", 1, 0, ()))
        self.assertEqual(legal(t, 0), {"throw", "forfeit"})

    def test_a_captured_stack_goes_home_and_earns_only_one_throw(self):
        s = stage(pieces={0: ["P3"], 1: ["P5", "P5", "P5"]}, bank=("gae",))
        t = yut.apply(s, 0, "move:0:gae")
        self.assertEqual(t.pieces[1], (None,) * 4)
        self.assertEqual(t.trails[1], ((),) * 4)
        self.assertEqual((t.phase, t.to_throw), ("throw", 1))

    def test_a_yut_or_mo_that_captures_earns_nothing_more(self):
        for start, throw in ((None, "mo"), ("P1", "yut")):
            with self.subTest(throw):
                s = stage(pieces={0: [start], 1: ["P5"]}, bank=(throw,))
                t = yut.apply(s, 0, f"move:0:{throw}")
                self.assertEqual((where(t), where(t, 1)), ("P5", None))
                self.assertEqual((t.turn, t.phase, t.to_throw, t.bank), (1, "throw", 1, ()))

    def test_a_yut_capture_with_more_banked_throws_continues_in_the_move_phase(self):
        s = stage(pieces={0: ["P1", "P12"], 1: ["P5"]}, bank=("yut", "do"))
        t = yut.apply(s, 0, "move:0:yut")
        self.assertEqual((t.phase, t.to_throw, t.bank, t.turn), ("move", 0, ("do",), 0))

    def test_a_gae_capture_with_more_banked_throws_throws_first(self):
        s = stage(pieces={0: ["P3", "P12"], 1: ["P5"]}, bank=("gae", "do"))
        t = yut.apply(s, 0, "move:0:gae")
        self.assertEqual((t.phase, t.to_throw, t.bank, t.turn), ("throw", 1, ("do",), 0))
        t = dataclasses.replace(t, seed=find_seed(["geol"], t.throws_made))
        u = yut.apply(t, 0, "throw")
        self.assertEqual((u.phase, u.bank, u.to_throw), ("move", ("do", "geol"), 0))

    def test_extra_throw_from_the_capture_can_be_a_yut_and_chain(self):
        s = stage(pieces={0: ["P3", "P12"], 1: ["P5"]}, bank=("gae",), seed="x")
        t = yut.apply(s, 0, "move:0:gae")
        t = dataclasses.replace(t, seed=find_seed(["yut", "do"], t.throws_made))
        t = yut.apply(t, 0, "throw")
        self.assertEqual((t.phase, t.to_throw, t.bank), ("throw", 1, ("yut",)))
        t = yut.apply(t, 0, "throw")
        self.assertEqual((t.phase, t.bank), ("move", ("yut", "do")))

    def test_capture_on_p0_after_stopping_there(self):
        s = stage(pieces={0: ["P17"], 1: [at("P0")]}, bank=("geol",))
        t = yut.apply(s, 0, "move:0:geol")
        self.assertEqual((where(t), where(t, 1), t.to_throw), ("P0", None, 1))

    def test_a_piece_stopped_on_p0_can_be_captured_by_a_later_arrival_and_it_is_not_finished(self):
        s = stage(pieces={0: [at("P0")], 1: ["P18"]}, bank=("gae",), turn=1)
        t = yut.apply(s, 1, "move:0:gae")
        self.assertEqual((where(t, 1), where(t, 0)), ("P0", None))

    def test_passing_over_an_opponent_does_not_capture(self):
        s = stage(pieces={0: ["P3"], 1: ["P5"]}, bank=("yut",))
        t = yut.apply(s, 0, "move:0:yut")
        self.assertEqual((where(t), where(t, 1)), ("P7", "P5"))
        self.assertEqual(t.turn, 1)

    def test_finishing_over_opponents_on_p0_does_not_capture(self):
        s = stage(pieces={0: ["P18"], 1: [at("P0")]}, bank=("yut",))
        t = yut.apply(s, 0, "move:0:yut")
        self.assertEqual((where(t), where(t, 1)), ("done", "P0"))

    def test_the_center_and_diagonal_points_capture_too(self):
        s = stage(pieces={0: ["a1"], 1: ["C"]}, bank=("gae",))
        t = yut.apply(s, 0, "move:0:gae")
        self.assertEqual((where(t), where(t, 1), t.to_throw), ("C", None, 1))

    def test_capture_in_a_four_seat_game_hits_only_the_landing_point(self):
        s = stage(4, pieces={0: ["P3"], 1: ["P5"], 2: ["P5"], 3: ["P6"]}, bank=("gae",))
        t = yut.apply(s, 0, "move:0:gae")
        self.assertEqual((where(t, 1), where(t, 2), where(t, 3)), (None, None, "P6"))

    def test_own_pieces_are_not_captured(self):
        s = stage(pieces={0: ["P3", "P5"]}, bank=("gae",))
        t = yut.apply(s, 0, "move:0:gae")
        self.assertEqual(t.pieces[0][:2], ("P5", "P5"))
        self.assertEqual((t.turn, t.to_throw), (1, 1))


class StackTests(unittest.TestCase):
    def test_landing_on_a_friend_stacks_without_an_extra_throw(self):
        s = stage(pieces={0: ["P1", "P3"]}, bank=("gae",))
        t = yut.apply(s, 0, "move:0:gae")
        self.assertEqual(t.pieces[0][:2], ("P3", "P3"))
        self.assertEqual(t.turn, 1)

    def test_moving_any_member_moves_the_whole_stack(self):
        for piece in (0, 1):
            with self.subTest(piece):
                s = stage(pieces={0: ["P3", "P3", "P9"]}, bank=("geol",))
                t = yut.apply(s, 0, f"move:{piece}:geol")
                self.assertEqual(t.pieces[0], ("P6", "P6", "P9", None))
                self.assertEqual(t.trails[0][0], t.trails[0][1])
                self.assertEqual(t.trails[0][0][-3:], ("P4", "P5", "P6"))

    def test_every_member_of_a_stack_is_listed_as_movable(self):
        s = stage(pieces={0: ["P3", "P3"]}, bank=("do",))
        self.assertTrue({"move:0:do", "move:1:do", "move:2:do", "move:3:do"} <= legal(s, 0))

    def test_a_stack_never_splits_on_the_board(self):
        s = stage(pieces={0: ["P3", "P3"]}, bank=("do", "gae"))
        t = yut.apply(s, 0, "move:1:do")
        self.assertEqual(t.pieces[0][:2], ("P4", "P4"))
        t = yut.apply(t, 0, "move:0:gae")
        self.assertEqual(t.pieces[0][:2], ("P6", "P6"))

    def test_backdo_on_a_stack_with_different_trails_follows_the_chosen_piece(self):
        via_a = at("C", PATH_A)
        via_b = at("C", PATH_B)
        s = stage(pieces={0: [via_a, via_b]}, bank=("backdo",))
        t = yut.apply(s, 0, "move:1:backdo")
        self.assertEqual(t.pieces[0][:2], ("c2", "c2"))
        self.assertEqual(t.trails[0][0], t.trails[0][1])
        self.assertEqual(t.trails[0][0][-1], "c2")
        u = yut.apply(s, 0, "move:0:backdo")
        self.assertEqual(u.pieces[0][:2], ("a2", "a2"))
        self.assertEqual(u.trails[0][0], u.trails[0][1])

    def test_history_records_a_capturing_stack_move_and_a_dropped_backdo(self):
        s = stage(pieces={0: ["P3", "P3"], 1: ["P5"]}, bank=("gae",))
        ev = yut.apply(s, 0, "move:0:gae").history[-1]
        self.assertIsInstance(ev, yut.Moved)
        self.assertEqual((ev.pieces, ev.start, ev.end, ev.captured), ((0, 1), "P3", "P5", ((1, 0),)))
        s = stage(pieces={0: ["P1"]}, bank=("backdo", "backdo"))
        t = yut.apply(s, 0, "move:0:backdo")
        self.assertIsInstance(t.history[-1], yut.Dropped)

    def test_a_stack_captures_and_stacks_grow(self):
        s = stage(pieces={0: ["P3", "P3", "P4"], 1: ["P5"]}, bank=("gae",))
        t = yut.apply(s, 0, "move:0:gae")
        self.assertEqual(t.pieces[0], ("P5", "P5", "P4", None))
        self.assertEqual((where(t, 1), t.to_throw), (None, 1))
        s = stage(pieces={0: ["P3", "P3", "P4"]}, bank=("do",))
        t = yut.apply(s, 0, "move:2:do")
        self.assertEqual(t.pieces[0], ("P3", "P3", "P5", None))

    def test_a_stack_finishes_together(self):
        s = stage(pieces={0: ["P18", "P18", "done", "done"]}, bank=("yut",))
        t = yut.apply(s, 0, "move:1:yut")
        self.assertEqual(t.pieces[0], ("done",) * 4)
        self.assertTrue(t.finished)

    def test_a_stack_moves_through_the_corner_shortcut_together(self):
        s = stage(pieces={0: ["P5", "P5"]}, bank=("gae",))
        t = yut.apply(s, 0, "move:0:gae")
        self.assertEqual(t.pieces[0][:2], ("a2", "a2"))

    def test_unstarted_pieces_do_not_form_a_stack(self):
        s = stage(bank=("gae",))
        t = yut.apply(s, 0, "move:0:gae")
        self.assertEqual(t.pieces[0], ("P2", None, None, None))

    def test_a_captured_stack_restarts_separately(self):
        s = stage(pieces={0: ["P3"], 1: ["P5", "P5"]}, bank=("gae",))
        t = yut.apply(s, 0, "move:0:gae")
        t = dataclasses.replace(t, turn=1, phase="move", to_throw=0, bank=("do",))
        u = yut.apply(t, 1, "move:0:do")
        self.assertEqual(u.pieces[1][:2], ("P1", None))

    def test_board_view_lists_stacks(self):
        s = stage(3, pieces={0: ["P3", "P3", "P9"], 1: ["P7", "C"]})
        self.assertEqual(yut.board_view(s)["stacks"], (("P3", 0, (0, 1)),))


class BankOrderTests(unittest.TestCase):
    def test_any_banked_throw_can_be_used_first(self):
        s = stage(pieces={0: ["P3"]}, bank=("gae", "geol"))
        self.assertTrue({"move:0:gae", "move:0:geol"} <= legal(s, 0))
        a = yut.apply(s, 0, "move:0:geol")
        self.assertEqual((where(a), a.bank, a.turn), ("P6", ("gae",), 0))
        b = yut.apply(s, 0, "move:0:gae")
        self.assertEqual((where(b), b.bank), ("P5", ("geol",)))

    def test_order_changes_the_route(self):
        first_gae = yut.apply(yut.apply(stage(pieces={0: ["P3"]}, bank=("gae", "do")), 0, "move:0:gae"), 0, "move:0:do")
        first_do = yut.apply(yut.apply(stage(pieces={0: ["P3"]}, bank=("gae", "do")), 0, "move:0:do"), 0, "move:0:gae")
        self.assertEqual(where(first_gae), "a1")  # stops on P5 with the gae, then turns
        self.assertEqual(where(first_do), "P6")   # passes P5 on the gae

    def test_different_pieces_may_use_different_throws(self):
        s = stage(pieces={0: ["P3", "P8"]}, bank=("do", "mo"))
        s = yut.apply(s, 0, "move:1:mo")
        s = yut.apply(s, 0, "move:0:do")
        self.assertEqual(s.pieces[0][:2], ("P4", "P13"))
        self.assertEqual(s.turn, 1)

    def test_a_duplicate_throw_is_used_up_once(self):
        s = stage(pieces={0: ["P3"]}, bank=("do", "do"))
        s = yut.apply(s, 0, "move:0:do")
        self.assertEqual(s.bank, ("do",))
        self.assertEqual({m for m in legal(s, 0) if m.startswith("move:0")}, {"move:0:do"})

    def test_a_throw_not_in_the_bank_is_illegal(self):
        s = stage(pieces={0: ["P3"]}, bank=("gae",))
        for bad in ("move:0:do", "move:0:mo", "move:0:bogus"):
            with self.subTest(bad), self.assertRaises(IllegalMove):
                yut.apply(s, 0, bad)

    def test_moves_listed_match_the_bank(self):
        s = stage(pieces={0: ["P3"]}, bank=("gae", "yut"))
        self.assertEqual({m for m in legal(s, 0) if m.startswith("move:0")}, {"move:0:gae", "move:0:yut"})


class TurnTests(unittest.TestCase):
    def test_the_turn_passes_when_the_bank_is_empty(self):
        s = stage(pieces={0: ["P3"]}, bank=("do",))
        t = yut.apply(s, 0, "move:0:do")
        self.assertEqual((t.turn, t.phase, t.to_throw, t.bank), (1, "throw", 1, ()))
        self.assertEqual(legal(t, 1), {"throw", "forfeit"})
        self.assertEqual(legal(t, 0), {"forfeit"})

    def test_the_turn_wraps_around_the_table(self):
        s = stage(3, pieces={2: ["P3"]}, bank=("do",), turn=2)
        self.assertEqual(yut.apply(s, 2, "move:0:do").turn, 0)

    def test_inactive_seats_are_skipped(self):
        s = stage(3, pieces={0: ["P3"]}, bank=("do",), active=(True, False, True))
        self.assertEqual(yut.apply(s, 0, "move:0:do").turn, 2)

    def test_a_turn_does_not_pass_while_banked_throws_remain(self):
        s = stage(pieces={0: ["P3"]}, bank=("do", "do"))
        self.assertEqual(yut.apply(s, 0, "move:0:do").turn, 0)

    def test_two_seat_turns_alternate_through_throws(self):
        s = throwing(["gae"])
        s = yut.apply(s, 0, "throw")
        s = yut.apply(s, 0, "move:0:gae")
        self.assertEqual((s.turn, s.phase), (1, "throw"))
        s = dataclasses.replace(s, seed=find_seed(["do"], s.throws_made))
        s = yut.apply(s, 1, "throw")
        s = yut.apply(s, 1, "move:0:do")
        self.assertEqual((s.turn, s.phase, where(s, 1)), (0, "throw", "P1"))

    def test_legal_moves_in_the_move_phase_have_no_throw(self):
        s = stage(pieces={0: ["P3"]}, bank=("do",))
        self.assertNotIn("throw", legal(s, 0))


class WinTests(unittest.TestCase):
    def test_finishing_the_last_piece_wins(self):
        s = stage(pieces={0: ["done", "done", "done", "P19"]}, bank=("gae", "do"), stake=5)
        t = yut.apply(s, 0, "move:3:gae")
        self.assertEqual((t.finished, t.winner, t.turn), (True, 0, None))
        self.assertEqual(t.pieces[0], ("done",) * 4)
        self.assertEqual(yut.legal_moves(t, 0), ())
        self.assertEqual(yut.legal_moves(t, 1), ())
        for seat, move in ((0, "throw"), (1, "throw"), (0, "move:0:do"), (1, "forfeit")):
            with self.subTest(move), self.assertRaises(IllegalMove):
                yut.apply(t, seat, move)
        r = yut.result(t)
        self.assertEqual((r.finished, r.winner, r.pot), (True, 0, 10))
        self.assertEqual([(x.seat, x.outcome, x.payout) for x in r.seats], [(0, "win", 10), (1, "lose", 0)])

    def test_stopping_on_p0_is_not_a_win_yet(self):
        s = stage(pieces={0: ["done", "done", "done", "P18"]}, bank=("gae",))
        t = yut.apply(s, 0, "move:3:gae")
        self.assertFalse(t.finished)
        self.assertEqual(where(t, 0, 3), "P0")

    def test_fewer_pieces_per_rules(self):
        s = stage(rules={"pieces": 2}, pieces={0: ["done", "P19"]}, bank=("do",))
        self.assertTrue(yut.apply(s, 0, "move:1:do").finished is False)
        s = stage(rules={"pieces": 2}, pieces={0: ["done", "P19"]}, bank=("gae",))
        self.assertTrue(yut.apply(s, 0, "move:1:gae").finished)

    def test_result_while_running(self):
        r = yut.result(mk(3, stake=5))
        self.assertEqual((r.finished, r.winner, r.pot), (False, None, 15))
        self.assertEqual([(x.outcome, x.payout) for x in r.seats], [(None, 0)] * 3)

    def test_a_win_in_a_three_seat_game_pays_the_whole_pot(self):
        s = stage(3, pieces={1: ["done", "done", "done", "P19"]}, bank=("do", "do"), turn=1, stake=4)
        s = yut.apply(s, 1, "move:3:do")
        self.assertFalse(s.finished)
        s = yut.apply(s, 1, "move:3:do")
        r = yut.result(s)
        self.assertEqual((s.winner, r.pot), (1, 12))
        self.assertEqual([(x.outcome, x.payout) for x in r.seats], [("lose", 0), ("win", 12), ("lose", 0)])


class ForfeitTests(unittest.TestCase):
    def test_every_active_seat_may_forfeit_any_time(self):
        s = stage(3, pieces={1: ["P3"]})
        for seat in range(3):
            self.assertIn("forfeit", legal(s, seat))

    def test_a_non_turn_seat_forfeits_and_its_pieces_leave_the_board(self):
        s = stage(3, pieces={1: ["P3", "P8"], 0: ["P5"]}, phase="throw", to_throw=1)
        t = yut.apply(s, 1, "forfeit")
        self.assertEqual(t.active, (True, False, True))
        self.assertEqual(t.pieces[1], (None,) * 4)
        self.assertEqual(t.trails[1], ((),) * 4)
        self.assertEqual((t.turn, t.phase, t.to_throw, t.finished), (0, "throw", 1, False))
        self.assertEqual(t.pieces[0][0], "P5")
        self.assertEqual(yut.legal_moves(t, 1), ())
        with self.assertRaises(IllegalMove):
            yut.apply(t, 1, "forfeit")

    def test_the_turn_seat_forfeiting_passes_the_turn(self):
        s = stage(3, pieces={0: ["P3"]}, bank=("gae", "do"), turn=0)
        t = yut.apply(s, 0, "forfeit")
        self.assertEqual((t.turn, t.phase, t.to_throw, t.bank), (1, "throw", 1, ()))
        self.assertEqual(t.active, (False, True, True))

    def test_a_forfeit_by_the_turn_seat_during_the_throw_phase(self):
        t = yut.apply(mk(3), 0, "forfeit")
        self.assertEqual((t.turn, t.phase), (1, "throw"))

    def test_the_last_seat_left_wins(self):
        s = stage(2, stake=7)
        t = yut.apply(s, 1, "forfeit")
        self.assertEqual((t.finished, t.winner, t.turn), (True, 0, None))
        r = yut.result(t)
        self.assertEqual(r.pot, 14)
        self.assertEqual([(x.outcome, x.payout) for x in r.seats], [("win", 14), ("forfeit", 0)])

    def test_forfeits_down_to_one_seat(self):
        s = stage(3, stake=2)
        s = yut.apply(s, 0, "forfeit")
        self.assertFalse(s.finished)
        s = yut.apply(s, 2, "forfeit")
        self.assertEqual((s.finished, s.winner), (True, 1))
        r = yut.result(s)
        self.assertEqual([(x.outcome, x.payout) for x in r.seats], [("forfeit", 0), ("win", 6), ("forfeit", 0)])

    def test_forfeited_seat_is_not_a_capture_target(self):
        s = stage(3, pieces={0: ["P3"], 1: ["P5"]}, bank=("gae",))
        s = yut.apply(s, 1, "forfeit")
        t = yut.apply(s, 0, "move:0:gae")
        self.assertEqual(where(t), "P5")
        self.assertEqual(t.turn, 2)  # no extra throw: nothing was captured


class ViewTests(unittest.TestCase):
    def test_view_is_public(self):
        s = stage(3, pieces={0: ["P3"]}, bank=("gae",), turn=0)
        v = yut.view(s)
        self.assertEqual((v.seat, v.turn, v.phase, v.pieces, v.bank, v.to_throw, v.finished, v.winner),
                         (None, 0, "move", s.pieces, ("gae",), 0, False, None))
        self.assertEqual(v.active, (True,) * 3)
        self.assertEqual(yut.view(s, 2).seat, 2)
        self.assertEqual(yut.view(s, 2).pieces, v.pieces)
        for bad in (3, -1):
            with self.assertRaises(ValueError):
                yut.view(s, bad)

    def test_board_view_structure(self):
        s = throwing(["geol"], pieces={0: ["P3", "done"], 1: ["C"]})
        s = yut.apply(s, 0, "throw")
        b = yut.board_view(s)
        self.assertEqual(b["pieces"], s.pieces)
        self.assertEqual(b["pieces"][0][:2], ("P3", "done"))
        self.assertEqual(b["pieces"][1][0], "C")
        self.assertEqual((b["bank"], b["to_throw"], b["phase"], b["turn"]), (("geol",), 0, "move", 0))
        self.assertEqual(b["last_throw"], {"name": "geol", "sticks": yut.throw_result(s.seed, 0)[1]})
        self.assertEqual(b["stacks"], ())

    def test_board_view_before_any_throw(self):
        b = yut.board_view(mk(2))
        self.assertIsNone(b["last_throw"])
        self.assertEqual((b["phase"], b["to_throw"], b["bank"]), ("throw", 1, ()))

    def test_board_view_is_plain_data(self):
        json.dumps(yut.board_view(stage(pieces={0: ["P3", "P3"]})), default=list)


def play_out(seed, n=2, rules=None, limit=3000):
    """Drive a game with a fixed policy (throw; else the first legal move); returns (final state, move log)."""
    rules = rules or {"pieces": 2}
    s = yut.new(seed, seats(n), rules)
    log = []
    for _ in range(limit):
        if s.finished:
            break
        moves = sorted(m for m in yut.legal_moves(s, s.turn) if m != "forfeit")
        move = "throw" if "throw" in moves else moves[0]
        log.append((s.turn, move))
        s = yut.apply(s, s.turn, move)
    return s, log


class SerializationTests(unittest.TestCase):
    def test_a_driven_game_is_deterministic(self):
        a, log_a = play_out("drive")
        b, log_b = play_out("drive")
        self.assertEqual((a, log_a), (b, log_b))
        if a.finished:
            self.assertEqual(sum(1 for p in a.pieces[a.winner] if p == "done"), 2)

    def test_replay_matches_the_driven_game(self):
        for n in (2, 3):
            with self.subTest(n):
                final, log = play_out(f"rp{n}", n)
                self.assertEqual(yut.replay(f"rp{n}", seats(n), log, {"pieces": 2}), final)

    def test_replay_of_a_prefix_matches_the_intermediate_state(self):
        final, log = play_out("prefix")
        mid = yut.replay("prefix", seats(2), log[:15], {"pieces": 2})
        s = yut.new("prefix", seats(2), {"pieces": 2})
        for seat, move in log[:15]:
            s = yut.apply(s, seat, move)
        self.assertEqual(mid, s)

    def test_replay_rejects_illegal_logs(self):
        with self.assertRaises(IllegalMove):
            yut.replay("rp", seats(2), [(1, "throw")])
        with self.assertRaises(IllegalMove):
            yut.replay("rp", seats(2), [(0, "move:0:do")])

    def test_json_round_trip_for_fresh_mid_and_finished_states(self):
        final, log = play_out("json")
        mid = yut.replay("json", seats(2), log[:20], {"pieces": 2})
        staged = stage(3, pieces={0: [at("C", PATH_B), "P3", "done"], 1: [at("P0")]}, bank=("backdo", "yut"), stake=3)
        won = yut.apply(stage(pieces={0: ["done"] * 3 + ["P19"]}, bank=("gae",)), 0, "move:3:gae")
        for s in (mk(3, stake=2), mid, final, staged, won):
            with self.subTest(turn=s.turn):
                text = yut.to_json(s)
                self.assertIsInstance(text, str)
                json.loads(text)
                back = yut.from_json(text)
                self.assertEqual(back, s)
                self.assertEqual(yut.to_json(back), text)
                self.assertEqual(yut.legal_moves(back, 0), yut.legal_moves(s, 0))

    def test_a_restored_state_keeps_playing_identically(self):
        _, log = play_out("resume")
        s = yut.replay("resume", seats(2), log[:12], {"pieces": 2})
        back = yut.from_json(yut.to_json(s))
        for seat, move in log[12:40]:
            s = yut.apply(s, seat, move)
            back = yut.apply(back, seat, move)
        self.assertEqual(s, back)


class IllegalMoveTests(unittest.TestCase):
    def test_wrong_seat(self):
        s = mk(3)
        for seat in (1, 2):
            with self.assertRaises(IllegalMove):
                yut.apply(s, seat, "throw")
        m = stage(pieces={0: ["P3"]}, bank=("do",))
        with self.assertRaises(IllegalMove):
            yut.apply(m, 1, "move:0:do")

    def test_wrong_phase(self):
        with self.assertRaises(IllegalMove):
            yut.apply(mk(2), 0, "move:0:do")
        with self.assertRaises(IllegalMove):
            yut.apply(stage(pieces={0: ["P3"]}, bank=("do",)), 0, "throw")

    def test_cannot_move_while_a_throw_is_owed(self):
        s = stage(pieces={0: ["P3"]}, bank=("yut",), phase="throw", to_throw=1)
        with self.assertRaises(IllegalMove):
            yut.apply(s, 0, "move:0:yut")

    def test_malformed_and_unknown_moves(self):
        s = stage(pieces={0: ["P3"]}, bank=("do",))
        for bad in ("", "move", "move:0", "move:x:do", "move:0:do:1", "move:-1:do", "move:4:do", "move:9:do", "pass", "THROW",
                    "move:0:", "draw"):
            with self.subTest(bad), self.assertRaises(IllegalMove):
                yut.apply(s, 0, bad)

    def test_finished_pieces_cannot_move(self):
        s = stage(pieces={0: ["done", "P3"]}, bank=("do",))
        with self.assertRaises(IllegalMove):
            yut.apply(s, 0, "move:0:do")
        self.assertNotIn("move:0:do", legal(s, 0))

    def test_another_seats_piece_index_is_not_a_target_only_own_indexes_exist(self):
        s = stage(3, pieces={0: ["P3"]}, bank=("do",))
        with self.assertRaises(IllegalMove):
            yut.apply(s, 0, "move:4:do")

    def test_a_move_for_a_nonexistent_seat_raises(self):
        with self.assertRaises(IllegalMove):
            yut.apply(mk(2), 5, "throw")

    def test_a_rejected_move_changes_nothing(self):
        s = stage(pieces={0: ["P3"]}, bank=("do",))
        snapshot = dataclasses.replace(s)
        with self.assertRaises(IllegalMove):
            yut.apply(s, 0, "move:0:mo")
        self.assertEqual(s, snapshot)

    def test_every_listed_legal_move_applies(self):
        s = stage(3, pieces={0: ["P3", "P3", "C", "P18"], 1: ["P5"]}, bank=("do", "gae", "backdo", "yut", "mo"))
        for move in yut.legal_moves(s, 0):
            with self.subTest(move):
                yut.apply(s, 0, move)
        for seat in range(3):
            with self.assertRaises(IllegalMove):
                yut.apply(s, seat, "throw")


if __name__ == "__main__":
    unittest.main()
