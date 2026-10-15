"""Yut board image (llmcord_core/games/yut_render.py), FEAT-57, decision D42.

Geometry and determinism are pinned, not pixels. Contract: POINTS (29 ids -> (x, y)), SIZE (w, h), render_board(view) -> PNG
bytes, piece_positions(view) -> [(seat, piece, (x, y))], text_fallback(view) -> English text, render_board_cached(view) (cache keyed
by piece positions only) and clear_cache().
"""
import dataclasses
import itertools
import math
import struct
import unittest
from unittest import mock

from llmcord_core.games import Seat
from llmcord_core.games import yut
from llmcord_core.games import yut_render as r

IDS = [f"P{i}" for i in range(20)] + ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2", "C"]


def mk(n=2):
    return yut.new("s", [Seat(i, "member", 100 + i, 0) for i in range(n)])


def view_of(pieces, **over):
    state = dataclasses.replace(mk(len(pieces)), pieces=tuple(tuple(p) for p in pieces))
    v = yut.board_view(state)
    v.update(over)
    return v


def png_size(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    return struct.unpack(">II", data[16:24])


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def off_line(p, a, b):
    """Perpendicular distance of p from the line through a and b."""
    (x, y), (x1, y1), (x2, y2) = p, a, b
    return abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / dist(a, b)


class PointsTests(unittest.TestCase):
    def test_all_29_points_present(self):
        self.assertEqual(set(r.POINTS), set(IDS))
        self.assertEqual(len(r.POINTS), 29)

    def test_points_inside_canvas_distinct_and_spaced(self):
        w, h = r.SIZE
        for k, (x, y) in r.POINTS.items():
            self.assertTrue(0 <= x <= w and 0 <= y <= h, k)
        for (a, pa), (b, pb) in itertools.combinations(r.POINTS.items(), 2):
            self.assertGreaterEqual(dist(pa, pb), 20, (a, b))

    def test_corners_form_a_square_with_center_in_the_middle(self):
        P = r.POINTS
        corners = [P["P0"], P["P5"], P["P10"], P["P15"]]
        sides = [dist(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        diags = [dist(corners[0], corners[2]), dist(corners[1], corners[3])]
        for s in sides:
            self.assertAlmostEqual(s, sides[0], delta=2)
        for d in diags:
            self.assertAlmostEqual(d, sides[0] * math.sqrt(2), delta=3)
        cx = sum(c[0] for c in corners) / 4
        cy = sum(c[1] for c in corners) / 4
        self.assertAlmostEqual(P["C"][0], cx, delta=2)
        self.assertAlmostEqual(P["C"][1], cy, delta=2)

    def test_diagonals_are_collinear_and_evenly_spaced(self):
        P = r.POINTS
        for chain in (["P5", "a1", "a2", "C", "b1", "b2", "P15"], ["P10", "c1", "c2", "C", "d1", "d2", "P0"]):
            ends = (P[chain[0]], P[chain[-1]])
            for k in chain:
                self.assertLessEqual(off_line(P[k], *ends), 2, k)
            gaps = [dist(P[chain[i]], P[chain[i + 1]]) for i in range(len(chain) - 1)]
            for g in gaps:
                self.assertAlmostEqual(g, gaps[0], delta=2)

    def test_ring_sides_are_evenly_spaced_and_collinear(self):
        P = r.POINTS
        for start in (0, 5, 10, 15):
            chain = [f"P{(start + i) % 20}" for i in range(6)]
            ends = (P[chain[0]], P[chain[-1]])
            gaps = [dist(P[chain[i]], P[chain[i + 1]]) for i in range(5)]
            for k, g in zip(chain, gaps + [gaps[0]]):
                self.assertLessEqual(off_line(P[k], *ends), 2, k)
                self.assertAlmostEqual(g, gaps[0], delta=2)


class RenderTests(unittest.TestCase):
    def test_png_with_declared_size(self):
        data = r.render_board(view_of([[None] * 5, [None] * 5]))
        self.assertEqual(png_size(data), tuple(r.SIZE))

    def test_deterministic(self):
        v = view_of([["P3", None, "done", "C", None], [None, "a1", None, None, None]])
        self.assertEqual(r.render_board(v), r.render_board(v))

    def test_moving_a_piece_changes_the_bytes_including_from_unstarted(self):
        base = r.render_board(view_of([[None] * 5, [None] * 5]))
        on_p5 = r.render_board(view_of([["P5", None, None, None, None], [None] * 5]))
        on_p6 = r.render_board(view_of([["P6", None, None, None, None], [None] * 5]))
        self.assertEqual(len({base, on_p5, on_p6}), 3)

    def test_other_seat_same_point_differs(self):
        a = r.render_board(view_of([["P5", None, None, None, None], [None] * 5]))
        b = r.render_board(view_of([[None] * 5, ["P5", None, None, None, None]]))
        self.assertNotEqual(a, b)

    def test_four_seats_five_pieces_and_stacks_do_not_crash(self):
        v = view_of([["P1", "P1", "P2", "done", None], ["P1", "a1", "a1", None, None], ["C", "C", "C", "C", "C"], ["d2"] * 5])
        self.assertEqual(png_size(r.render_board(v)), tuple(r.SIZE))

    def test_real_game_state(self):
        s = mk(3)
        s = yut.apply(s, 0, "throw")
        v = yut.board_view(s)
        self.assertEqual(png_size(r.render_board(v)), tuple(r.SIZE))


class PiecePositionTests(unittest.TestCase):
    def test_only_on_board_pieces_listed(self):
        v = view_of([["P5", None, "done", "C", None], [None, "a1", None, None, None]])
        got = r.piece_positions(v)
        self.assertEqual(sorted((s, i) for s, i, _ in got), [(0, 0), (0, 3), (1, 1)])

    def test_piece_is_near_its_point(self):
        v = view_of([["P5", None, None, None, None], [None] * 5])
        ((seat, idx, xy),) = r.piece_positions(v)
        self.assertEqual((seat, idx), (0, 0))
        self.assertLessEqual(dist(xy, r.POINTS["P5"]), 20)

    def test_stack_and_shared_point_sprites_do_not_overlap_exactly(self):
        v = view_of([["P5", "P5", "P5", None, None], ["P5", "P5", None, None, None]])
        got = r.piece_positions(v)
        self.assertEqual(len(got), 5)
        coords = [xy for _, _, xy in got]
        self.assertEqual(len(set(coords)), 5)
        for xy in coords:
            self.assertLessEqual(dist(xy, r.POINTS["P5"]), 30)

    def test_full_board_has_no_shared_coordinates(self):
        pts = ["P0", "P1", "a1", "C", "d2"]
        v = view_of([list(pts), list(pts), list(pts), list(pts)])
        coords = [xy for _, _, xy in r.piece_positions(v)]
        self.assertEqual(len(coords), 20)
        self.assertEqual(len(set(coords)), 20)

    def test_positions_are_deterministic(self):
        v = view_of([["P5", "P5", None, None, None], [None] * 5])
        self.assertEqual(r.piece_positions(v), r.piece_positions(v))


class TextFallbackTests(unittest.TestCase):
    def test_lists_points_and_counts(self):
        v = view_of([["P5", "C", None, None, "done"], [None, "a1", "a1", "done", "done"]])
        self.assertEqual(
            r.text_fallback(v).splitlines(),
            ["Seat 1: on board P5, C; at start 2; done 1", "Seat 2: on board a1, a1; at start 1; done 2"],
        )

    def test_one_line_per_seat_and_no_board_points_when_none(self):
        v = view_of([[None] * 5, [None] * 5, [None] * 5])
        lines = [ln for ln in r.text_fallback(v).splitlines() if ln.strip()]
        self.assertEqual(lines, ["Seat %d: on board -; at start 5; done 0" % n for n in (1, 2, 3)])

    def test_text_changes_when_a_piece_moves(self):
        a = r.text_fallback(view_of([["P5", None, None, None, None], [None] * 5]))
        b = r.text_fallback(view_of([["P6", None, None, None, None], [None] * 5]))
        self.assertNotEqual(a, b)
        self.assertIn("P6", b)
        self.assertNotIn("P5", b)


class CacheTests(unittest.TestCase):
    def setUp(self):
        r.clear_cache()
        self.addCleanup(r.clear_cache)

    def test_same_bytes_as_render_board(self):
        v = view_of([["P5", None, None, None, None], [None] * 5])
        self.assertEqual(r.render_board_cached(v), r.render_board(v))

    def test_repeat_call_returns_identical_object(self):
        v = view_of([["P5", None, None, None, None], [None] * 5])
        first = r.render_board_cached(v)
        self.assertIs(r.render_board_cached(view_of([["P5", None, None, None, None], [None] * 5])), first)

    def test_bank_throw_and_turn_do_not_affect_key_or_bytes(self):
        pieces = [["P5", "P5", None, None, None], [None, "C", None, None, None]]
        a = view_of(pieces)
        b = view_of(pieces, bank=("yut", "do"), turn=1, to_throw=2, phase="move",
                    last_throw={"name": "mo", "sticks": (False,) * 4})
        first = r.render_board_cached(a)
        self.assertIs(r.render_board_cached(b), first)
        self.assertEqual(r.render_board(a), r.render_board(b))

    def test_eviction_drops_least_recently_used(self):
        def v(p):
            return view_of([[p, None, None, None, None], [None] * 5])

        a, b, c = v("P1"), v("P2"), v("P3")
        with mock.patch.object(r, "CACHE_MAX", 2):
            ra = r.render_board_cached(a)
            rb = r.render_board_cached(b)
            self.assertIs(r.render_board_cached(a), ra)
            r.render_board_cached(c)
            self.assertEqual(len(r._cache), 2)
            self.assertIs(r.render_board_cached(a), ra)
            self.assertIsNot(r.render_board_cached(b), rb)
            self.assertEqual(len(r._cache), 2)

    def test_different_positions_not_shared_and_clear_cache_drops_entries(self):
        a = view_of([["P5", None, None, None, None], [None] * 5])
        b = view_of([["P6", None, None, None, None], [None] * 5])
        self.assertNotEqual(r.render_board_cached(a), r.render_board_cached(b))
        first = r.render_board_cached(a)
        r.clear_cache()
        again = r.render_board_cached(a)
        self.assertEqual(first, again)
        self.assertIsNot(first, again)


if __name__ == "__main__":
    unittest.main()
