"""I Doubt It rules engine and fallback policy (llmcord_core/games/doubt.py), FEAT-21 part A1.

Assumed public shapes (attribute access on the objects returned by ``view`` / ``result``):
view: counts, active, turn, rank, pile, window (None or claimer/count/rank), history (events with
kind 'play'/'reveal'; reveals carry claimer/doubter/cards/bluff), hand (own sorted cards, None for
the public view), finished, winner. result: finished, winner, seats[i].outcome / .payout.
Cards are (rank, suit) tuples. Hands are found through ``view`` rather than assuming deal order.
"""
import dataclasses
import itertools
import json
import unittest
from collections import Counter

from llmcord_core.games import IllegalMove, Seat, shuffled
from llmcord_core.games import doubt

RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")


def seats(n, stake=0):
    return [Seat(i, "member", 100 + i, stake) for i in range(n)]


def mk(n=3, seed="s", stake=0, rules=None):
    return doubt.new(seed, seats(n, stake), rules)


def hand(state, i):
    return list(doubt.view(state, i).hand)


def mv(cards):
    """Canonical move string for physical cards."""
    counts = Counter(c[0] for c in cards)
    return "play:" + ",".join(f"{r}x{counts[r]}" for r in RANKS if counts[r])


def plays(state, i):
    return {m for m in doubt.legal_moves(state, i) if m.startswith("play:")}


def expected_plays(h):
    held = Counter(c[0] for c in h)
    out = set()
    for combo in itertools.product(*(range(0, min(held[r], 4) + 1) for r in RANKS)):
        if 1 <= sum(combo) <= 4:
            out.add("play:" + ",".join(f"{r}x{n}" for r, n in zip(RANKS, combo) if n))
    return out


def seed_where(pred, limit=400):
    for k in range(limit):
        seed = f"seed-{k}"
        if pred(seed):
            return seed
    raise AssertionError("no seed found")


def forced_card(state, i):
    """A card of the forced rank in seat i's hand, or None."""
    rank = doubt.view(state).rank
    return next((c for c in hand(state, i) if c[0] == rank), None)


def non_forced_card(state, i):
    rank = doubt.view(state).rank
    return next(c for c in hand(state, i) if c[0] != rank)


def walk_cards(obj, skip=()):
    """Every (rank, suit)-looking tuple inside a view, except under the given attribute names."""
    found = []
    if isinstance(obj, tuple) and len(obj) == 2 and obj[0] in RANKS:
        return [obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        items = [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)]
    elif isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        items = [(None, x) for x in obj]
    elif hasattr(obj, "__dict__"):
        items = list(vars(obj).items())
    else:
        return found
    for name, value in items:
        if name in skip:
            continue
        found.extend(walk_cards(value))
    return found


def natural_turn(state):
    """Play one card (a forced-rank one if held, else the first): returns (state, truthful)."""
    v = doubt.view(state)
    seat = v.turn
    card = forced_card(state, seat) or hand(state, seat)[0]
    return doubt.apply(state, seat, mv([card])), card[0] == v.rank


def drive_to_empty(seed, n=3):
    """Everyone plays one card per turn, nobody doubts, until a claimer's hand is empty.

    Returns (state, last_play_truthful)."""
    state = mk(n, seed, stake=5)
    truthful = None
    for _ in range(200):
        state, truthful = natural_turn(state)
        w = doubt.view(state).window
        if w is not None and doubt.view(state).counts[w.claimer] == 0:
            return state, truthful
    raise AssertionError("never emptied a hand")


def reveals(state):
    return [e for e in doubt.view(state).history if getattr(e, "bluff", None) is not None]


class DealTests(unittest.TestCase):
    def test_counts_and_decks_per_seat_count(self):
        for n, total, decks in ((3, 52, 1), (5, 52, 1), (6, 104, 2), (8, 104, 2)):
            with self.subTest(seats=n):
                s = mk(n)
                counts = doubt.view(s).counts
                self.assertEqual(len(counts), n)
                self.assertEqual(sum(counts), total)
                self.assertEqual(max(counts) - min(counts), 1 if total % n else 0)
                self.assertEqual(list(counts), sorted(counts, reverse=True))  # round-robin from seat 0
                everything = [c for i in range(n) for c in hand(s, i)]
                self.assertEqual(Counter(c[0] for c in everything), {r: 4 * decks for r in RANKS})
                self.assertEqual(len(set(everything)), 52)  # suits distinguish the copies of one deck

    def test_deal_is_deterministic_and_seed_dependent(self):
        self.assertEqual(hand(mk(4, "a"), 0), hand(mk(4, "a"), 0))
        self.assertNotEqual([hand(mk(4, "a"), i) for i in range(4)], [hand(mk(4, "b"), i) for i in range(4)])

    def test_first_player_is_seat_zero_and_first_rank_is_ace(self):
        v = doubt.view(mk(4))
        self.assertEqual((v.turn, v.rank, v.pile, v.window, v.finished), (0, "A", 0, None, False))
        self.assertEqual(list(v.active), [True] * 4)

    def test_seat_count_limits_and_indexes(self):
        for n in (0, 1, 2, 9):
            with self.assertRaises(ValueError):
                doubt.new("s", seats(n))
        with self.assertRaises(ValueError):
            doubt.new("s", [Seat(0, "member", 1, 0), Seat(2, "member", 2, 0), Seat(1, "member", 3, 0)])

    def test_new_rejects_bad_rules(self):
        with self.assertRaises(ValueError):
            mk(3, rules={"claim_rule": "free"})


class ForcedRankTests(unittest.TestCase):
    def test_rank_follows_sequence_and_wraps_after_king(self):
        s = mk(3)
        seen = []
        for _ in range(15):
            seen.append(doubt.view(s).rank)
            s, _ = natural_turn(s)
        self.assertEqual(seen, list(RANKS) + ["A", "2"])

    def test_window_remembers_claimed_rank_while_next_rank_advances(self):
        s = mk(3)
        s = doubt.apply(s, 0, mv(hand(s, 0)[:2]))
        v = doubt.view(s)
        self.assertEqual((v.window.claimer, v.window.count, v.window.rank), (0, 2, "A"))
        self.assertEqual((v.rank, v.turn, v.pile), ("2", 1, 2))

    def test_rank_advances_even_when_a_doubt_resolves(self):
        s = mk(3)
        s = doubt.apply(s, 0, mv([non_forced_card(s, 0)]))
        s = doubt.apply(s, 1, "doubt")
        self.assertEqual(doubt.view(s).rank, "2")
        self.assertIsNone(doubt.view(s).window)


class PlayTests(unittest.TestCase):
    def test_legal_plays_are_all_canonical_one_to_four_card_combinations(self):
        s = mk(3, "enum")
        self.assertEqual(plays(s, 0), expected_plays(hand(s, 0)))

    def test_only_the_seat_on_turn_may_play(self):
        s = mk(3)
        self.assertEqual(plays(s, 1), set())
        self.assertEqual(plays(s, 2), set())
        with self.assertRaises(IllegalMove):
            doubt.apply(s, 1, mv(hand(s, 1)[:1]))

    def test_play_removes_the_named_ranks_from_the_hand_and_grows_the_pile(self):
        s = mk(3, "remove")
        h = hand(s, 0)
        chosen = h[:3]
        t = doubt.apply(s, 0, mv(chosen))
        self.assertEqual(Counter(c[0] for c in hand(t, 0)), Counter(c[0] for c in h) - Counter(c[0] for c in chosen))
        self.assertEqual((doubt.view(t).counts[0], doubt.view(t).pile), (len(h) - 3, 3))
        self.assertEqual(hand(t, 1), hand(s, 1))

    def test_multi_rank_move_parses_and_counts_as_one_claim(self):
        s = mk(3, "multi")
        h = hand(s, 0)
        two = next(r for r in RANKS if Counter(c[0] for c in h)[r] >= 2)
        other = next(r for r in RANKS if r != two and Counter(c[0] for c in h)[r] >= 1)
        move = "play:" + ",".join(f"{r}x{n}" for r, n in sorted(((two, 2), (other, 1)), key=lambda p: RANKS.index(p[0])))
        self.assertIn(move, plays(s, 0))
        t = doubt.apply(s, 0, move)
        self.assertEqual(doubt.view(t).window.count, 3)

    def test_illegal_play_strings(self):
        s = mk(3, "illegal")
        h = hand(s, 0)
        held = Counter(c[0] for c in h)
        absent = [r for r in RANKS if not held[r]]
        two_ranks = sorted({c[0] for c in h}, key=RANKS.index)[:2]
        bad = [
            "play:", "play", "", "pass", "play:Zx1", "play:7x0", "play:7", "play:x1", "play:7x-1", "play:7x1.5",
            f"play:{two_ranks[1]}x1,{two_ranks[0]}x1",  # not in RANKS order
            f"play:{two_ranks[0]}x1,{two_ranks[0]}x1",  # duplicate rank
            f"play:{two_ranks[0]}x1,", " play:" + two_ranks[0] + "x1",
        ]
        most = max(RANKS, key=lambda r: held[r])
        bad.append(f"play:{most}x{held[most] + 1}")  # more than held (or above 4)
        bad += [f"play:{r}x1" for r in absent[:1]]
        bad.append("play:" + ",".join(f"{r}x1" for r in sorted(held, key=RANKS.index)[:5]))  # five cards
        for move in bad:
            with self.subTest(move=move):
                with self.assertRaises(IllegalMove):
                    doubt.apply(s, 0, move)
        for seat in (-1, 3):
            with self.assertRaises((IllegalMove, ValueError)):
                doubt.apply(s, seat, mv(h[:1]))

    def test_malformed_counts_raise_illegal_move_not_value_error(self):
        s = mk(3, "illegal")
        for move in ("play:7x" + "9" * 5000, "play:7x05", "play:7x\u0661", "play:7x5", "play:7x 1"):
            with self.subTest(move=move[:20]):
                with self.assertRaises(IllegalMove):
                    doubt.apply(s, 0, move)

    def test_bad_seat_and_move_types(self):
        s = mk(3)
        for seat in (True, "0", None, 1.0):
            self.assertEqual(doubt.legal_moves(s, seat), ())
            with self.assertRaises(IllegalMove):
                doubt.apply(s, seat, "forfeit")
        self.assertEqual(doubt.legal_moves(s, -1), ())
        self.assertEqual(doubt.legal_moves(s, 3), ())
        for move in (None, 5, b"doubt", ["doubt"]):
            with self.assertRaises(IllegalMove):
                doubt.apply(s, 0, move)

    def test_apply_does_not_mutate_the_old_state(self):
        s = mk(3)
        before = doubt.view(s, 0)
        doubt.apply(s, 0, mv(hand(s, 0)[:1]))
        self.assertEqual(doubt.view(s, 0), before)


class WindowTests(unittest.TestCase):
    def test_who_may_do_what_while_the_window_is_open(self):
        s = mk(4, "window")
        s = doubt.apply(s, 0, mv(hand(s, 0)[:1]))
        self.assertNotIn("doubt", doubt.legal_moves(s, 0))
        self.assertIn("close", doubt.legal_moves(s, 0))
        for i in (1, 2, 3):
            self.assertIn("doubt", doubt.legal_moves(s, i))
            self.assertNotIn("close", doubt.legal_moves(s, i))
        self.assertEqual(plays(s, 1), expected_plays(hand(s, 1)))
        self.assertEqual(plays(s, 2), set())
        self.assertEqual(plays(s, 0), set())
        with self.assertRaises(IllegalMove):
            doubt.apply(s, 2, mv(hand(s, 2)[:1]))
        with self.assertRaises(IllegalMove):
            doubt.apply(s, 0, "doubt")
        with self.assertRaises(IllegalMove):
            doubt.apply(s, 1, "close")

    def test_no_doubt_or_close_without_a_window(self):
        s = mk(3)
        for seat in range(3):
            self.assertNotIn("doubt", doubt.legal_moves(s, seat))
            self.assertNotIn("close", doubt.legal_moves(s, seat))
            for move in ("doubt", "close"):
                with self.assertRaises(IllegalMove):
                    doubt.apply(s, seat, move)

    def test_close_by_the_claimer_closes_without_a_doubt(self):
        s = mk(3)
        s = doubt.apply(s, 0, mv(hand(s, 0)[:1]))
        t = doubt.apply(s, 0, "close")
        v = doubt.view(t)
        self.assertIsNone(v.window)
        self.assertEqual((v.turn, v.pile, v.finished), (1, 1, False))
        self.assertEqual(reveals(t), [])
        with self.assertRaises(IllegalMove):
            doubt.apply(t, 2, "doubt")

    def test_next_play_closes_the_window_and_leaves_the_pile(self):
        s = mk(3, "implicit")
        s = doubt.apply(s, 0, mv(hand(s, 0)[:2]))
        s = doubt.apply(s, 1, mv(hand(s, 1)[:1]))
        v = doubt.view(s)
        self.assertEqual((v.window.claimer, v.window.count, v.window.rank), (1, 1, "2"))
        self.assertEqual((v.pile, v.turn, v.rank), (3, 2, "3"))
        self.assertEqual(reveals(s), [])
        self.assertNotIn("doubt", doubt.legal_moves(s, 1))
        self.assertIn("doubt", doubt.legal_moves(s, 0))  # seat 0 may now doubt seat 1's claim only

    def test_doubt_after_the_next_play_only_reveals_the_latest_claim(self):
        s = mk(3, "latest")
        s = doubt.apply(s, 0, mv(hand(s, 0)[:2]))
        played = hand(s, 1)[:1]
        s = doubt.apply(s, 1, mv(played))
        s = doubt.apply(s, 2, "doubt")
        (r,) = reveals(s)
        self.assertEqual((r.claimer, r.doubter, len(r.cards)), (1, 2, 1))
        self.assertEqual(r.cards[0][0], played[0][0])


class DoubtTests(unittest.TestCase):
    def test_bluff_doubt_makes_the_claimer_take_the_pile(self):
        s = mk(3, "bluff")
        h = hand(s, 0)
        bluff = non_forced_card(s, 0)
        s = doubt.apply(s, 0, mv([bluff]))
        t = doubt.apply(s, 2, "doubt")
        v = doubt.view(t)
        self.assertEqual(v.counts[0], len(h))  # played 1, took the pile of 1 back
        self.assertEqual((v.pile, v.turn, v.finished, v.window), (0, 1, False, None))
        (r,) = reveals(t)
        self.assertEqual((r.claimer, r.doubter, r.bluff), (0, 2, True))
        self.assertEqual([c[0] for c in r.cards], [bluff[0]])
        self.assertIn(bluff[0], [c[0] for c in hand(t, 0)])

    def test_truthful_doubt_makes_the_doubter_take_the_pile(self):
        seed = seed_where(lambda sd: forced_card(mk(3, sd), 0) is not None)
        s = mk(3, seed)
        h = hand(s, 0)
        s = doubt.apply(s, 0, mv([forced_card(s, 0)]))
        t = doubt.apply(s, 2, "doubt")
        v = doubt.view(t)
        self.assertEqual((v.counts[0], v.counts[2], v.pile, v.turn), (len(h) - 1, 17 + 1, 0, 1))
        (r,) = reveals(t)
        self.assertEqual((r.claimer, r.doubter, r.bluff), (0, 2, False))
        self.assertEqual([c[0] for c in r.cards], ["A"])

    def test_a_single_wrong_card_among_true_ones_is_a_bluff(self):
        def mixed(sd):
            s = mk(3, sd)
            return forced_card(s, 0) is not None and any(c[0] != "A" for c in hand(s, 0))
        s = mk(3, seed_where(mixed))
        cards = [forced_card(s, 0), non_forced_card(s, 0)]
        t = doubt.apply(doubt.apply(s, 0, mv(cards)), 1, "doubt")
        (r,) = reveals(t)
        self.assertTrue(r.bluff)
        self.assertEqual(sorted(c[0] for c in r.cards), sorted(c[0] for c in cards))

    def test_pile_collects_several_plays_then_goes_to_the_loser_whole(self):
        s = mk(3, "bigpile")
        s, _ = natural_turn(s)   # seat 0
        s, _ = natural_turn(s)   # seat 1
        before = doubt.view(s)
        self.assertEqual(before.pile, 2)
        claimed = hand(s, 2)[:1]
        s = doubt.apply(s, 2, mv(claimed))
        t = doubt.apply(s, 0, "doubt")
        (r,) = reveals(t)
        loser = r.claimer if r.bluff else r.doubter
        self.assertEqual(doubt.view(t).counts[loser], doubt.view(s).counts[loser] + 3)
        self.assertEqual(doubt.view(t).pile, 0)

    def test_turn_after_a_doubt_skips_forfeited_seats(self):
        s = mk(4, "skip")
        s = doubt.apply(s, 3, "forfeit")
        s = doubt.apply(s, 0, mv(hand(s, 0)[:1]))
        s = doubt.apply(s, 2, "doubt")
        self.assertEqual(doubt.view(s).turn, 1)
        s = doubt.apply(s, 1, mv(hand(s, 1)[:1]))
        s = doubt.apply(s, 0, "doubt")
        self.assertEqual(doubt.view(s).turn, 2)
        s = doubt.apply(s, 2, mv(hand(s, 2)[:1]))
        s = doubt.apply(s, 1, "doubt")
        self.assertEqual(doubt.view(s).turn, 0)  # 3 is gone, wraps to 0

    def test_cannot_doubt_twice_or_after_the_window_closed(self):
        s = mk(3, "twice")
        s = doubt.apply(s, 0, mv(hand(s, 0)[:1]))
        s = doubt.apply(s, 1, "doubt")
        with self.assertRaises(IllegalMove):
            doubt.apply(s, 2, "doubt")


class WinTests(unittest.TestCase):
    def test_emptied_hand_allows_only_doubt_and_close(self):
        s, _ = drive_to_empty("win-a")
        v = doubt.view(s)
        c = v.window.claimer
        self.assertFalse(v.finished)
        self.assertEqual(plays(s, (c + 1) % 3), set())
        self.assertEqual(plays(s, (c + 2) % 3), set())
        self.assertIn("close", doubt.legal_moves(s, c))
        self.assertIn("doubt", doubt.legal_moves(s, (c + 1) % 3))
        with self.assertRaises(IllegalMove):
            doubt.apply(s, (c + 1) % 3, "play:Ax1")

    def test_close_on_an_emptied_hand_wins(self):
        s, _ = drive_to_empty("win-b")
        c = doubt.view(s).window.claimer
        t = doubt.apply(s, c, "close")
        r = doubt.result(t)
        self.assertTrue(doubt.view(t).finished)
        self.assertEqual(doubt.view(t).winner, c)
        self.assertEqual((r.finished, r.winner), (True, c))

    def test_truthful_doubt_on_the_last_play_wins_for_the_claimer(self):
        seed = seed_where(lambda sd: drive_to_empty(sd)[1])
        s, truthful = drive_to_empty(seed)
        self.assertTrue(truthful)
        c = doubt.view(s).window.claimer
        doubter = (c + 1) % 3
        t = doubt.apply(s, doubter, "doubt")
        self.assertTrue(doubt.view(t).finished)
        self.assertEqual(doubt.result(t).winner, c)
        self.assertFalse(reveals(t)[0].bluff)

    def test_bluff_doubt_on_the_last_play_hands_back_the_pile_and_play_continues(self):
        seed = seed_where(lambda sd: not drive_to_empty(sd)[1])
        s, truthful = drive_to_empty(seed)
        self.assertFalse(truthful)
        before = doubt.view(s)
        c = before.window.claimer
        t = doubt.apply(s, (c + 2) % 3, "doubt")
        v = doubt.view(t)
        self.assertFalse(v.finished)
        self.assertEqual((v.counts[c], v.pile, v.turn), (before.pile, 0, (c + 1) % 3))
        self.assertTrue(reveals(t)[0].bluff)
        self.assertTrue(plays(t, (c + 1) % 3))
        self.assertFalse(doubt.result(t).finished)

    def test_nothing_is_legal_after_the_game_ends(self):
        s, _ = drive_to_empty("win-c")
        c = doubt.view(s).window.claimer
        t = doubt.apply(s, c, "close")
        for seat in range(3):
            self.assertEqual(plays(t, seat), set())
            self.assertNotIn("doubt", doubt.legal_moves(t, seat))
            for move in ("doubt", "close", "forfeit", "play:Ax1"):
                with self.assertRaises(IllegalMove):
                    doubt.apply(t, seat, move)

    def test_pot_goes_to_the_winner_and_the_rest_lose(self):
        s, _ = drive_to_empty("pot")
        c = doubt.view(s).window.claimer
        r = doubt.result(doubt.apply(s, c, "close"))
        for i, seat in enumerate(r.seats):
            self.assertEqual((seat.outcome, seat.payout), ("win", 15) if i == c else ("lose", 0))


class ForfeitTests(unittest.TestCase):
    def test_forfeit_on_turn_moves_the_turn_and_discards_the_cards(self):
        s = mk(3, "ff1")
        t = doubt.apply(s, 0, "forfeit")
        v = doubt.view(t)
        self.assertEqual((v.turn, v.pile, v.finished), (1, 0, False))
        self.assertEqual(list(v.active), [False, True, True])
        self.assertEqual(sum(v.counts[1:]), 34)
        self.assertEqual(plays(t, 0), set())
        self.assertEqual(plays(t, 1), expected_plays(hand(t, 1)))
        with self.assertRaises(IllegalMove):
            doubt.apply(t, 0, "forfeit")

    def test_forfeit_off_turn_just_removes_the_seat_from_the_order(self):
        s = mk(4, "ff2")
        t = doubt.apply(s, 2, "forfeit")
        v = doubt.view(t)
        self.assertEqual((v.turn, list(v.active)), (0, [True, True, False, True]))
        t = doubt.apply(t, 0, mv(hand(t, 0)[:1]))
        t = doubt.apply(t, 1, mv(hand(t, 1)[:1]))
        self.assertEqual(doubt.view(t).turn, 3)
        with self.assertRaises(IllegalMove):
            doubt.apply(t, 2, "doubt")

    def test_claimer_forfeiting_closes_the_window_without_a_doubt(self):
        s = mk(4, "ff3")
        s = doubt.apply(s, 0, mv(hand(s, 0)[:2]))
        t = doubt.apply(s, 0, "forfeit")
        v = doubt.view(t)
        self.assertIsNone(v.window)
        self.assertEqual((v.turn, v.pile, v.finished), (1, 2, False))  # played cards stay in the pile
        self.assertEqual(reveals(t), [])
        for i in (1, 2, 3):
            self.assertNotIn("doubt", doubt.legal_moves(t, i))

    def test_turn_holder_forfeit_with_the_window_open_keeps_the_window(self):
        s = mk(4, "ff6")
        s = doubt.apply(s, 0, mv(hand(s, 0)[:1]))
        t = doubt.apply(s, 1, "forfeit")
        v = doubt.view(t)
        self.assertEqual((v.window.claimer, v.window.count, v.turn, v.pile), (0, 1, 2, 1))
        self.assertIn("doubt", doubt.legal_moves(t, 3))
        t = doubt.apply(t, 2, mv(hand(t, 2)[:1]))
        v = doubt.view(t)
        self.assertEqual((v.window.claimer, v.turn, v.pile), (2, 3, 2))
        self.assertEqual(reveals(t), [])

    def test_off_turn_non_claimer_forfeit_with_the_window_open(self):
        s = mk(4, "ff7")
        s = doubt.apply(s, 0, mv(hand(s, 0)[:1]))
        t = doubt.apply(s, 3, "forfeit")
        v = doubt.view(t)
        self.assertEqual((v.window.claimer, v.turn, v.pile, list(v.active)), (0, 1, 1, [True, True, True, False]))
        self.assertEqual(doubt.legal_moves(t, 3), ())
        self.assertIn("doubt", doubt.legal_moves(t, 2))

    def test_claimer_forfeiting_after_emptying_the_hand_does_not_win(self):
        s, _ = drive_to_empty("win-b")
        c = doubt.view(s).window.claimer
        t = doubt.apply(s, c, "forfeit")
        v = doubt.view(t)
        self.assertEqual((v.finished, v.winner, v.window), (False, None, None))
        self.assertEqual(v.turn, (c + 1) % 3)
        self.assertTrue(plays(t, (c + 1) % 3))
        self.assertEqual(doubt.result(t).seats[c].outcome, "forfeit")

    def test_last_remaining_seat_wins(self):
        s = mk(3, "ff4", stake=4)
        s = doubt.apply(s, 1, "forfeit")
        self.assertFalse(doubt.view(s).finished)
        s = doubt.apply(s, 0, "forfeit")
        v = doubt.view(s)
        self.assertEqual((v.finished, v.winner), (True, 2))
        r = doubt.result(s)
        self.assertEqual([x.outcome for x in r.seats], ["forfeit", "forfeit", "win"])
        self.assertEqual([x.payout for x in r.seats], [0, 0, 12])

    def test_forfeit_during_an_open_window_by_the_last_other_seat_ends_the_game(self):
        s = mk(3, "ff5")
        s = doubt.apply(s, 0, mv(hand(s, 0)[:1]))
        s = doubt.apply(s, 1, "forfeit")
        s = doubt.apply(s, 2, "forfeit")
        self.assertEqual(doubt.view(s).winner, 0)


class ViewTests(unittest.TestCase):
    def test_public_view_has_counts_but_no_cards(self):
        s = mk(4, "hide")
        s = doubt.apply(s, 0, mv(hand(s, 0)[:2]))
        v = doubt.view(s)
        self.assertIsNone(v.hand)
        self.assertEqual(walk_cards(v), [])
        self.assertEqual((v.counts[0], v.pile, v.window.count), (len(hand(s, 0)), 2, 2))

    def test_seat_view_shows_only_its_own_sorted_hand(self):
        s = mk(4, "hide2")
        s = doubt.apply(s, 0, mv(hand(s, 0)[:2]))
        for i in range(4):
            v = doubt.view(s, i)
            ranks = [RANKS.index(c[0]) for c in v.hand]
            self.assertEqual(ranks, sorted(ranks))
            self.assertEqual(len(v.hand), v.counts[i])
            self.assertEqual(walk_cards(v, skip=("hand",)), [])
        mine = set(hand(s, 1))
        for i in (0, 2, 3):
            self.assertTrue(mine.isdisjoint(hand(s, i)))

    def test_history_records_plays_publicly_and_reveals_only_the_doubted_cards(self):
        s = mk(3, "hist")
        played = hand(s, 0)[:2]
        s = doubt.apply(s, 0, mv(played))
        plays_seen = [e for e in doubt.view(s).history if getattr(e, "bluff", None) is None]
        self.assertEqual([(e.seat, e.count, e.rank) for e in plays_seen], [(0, 2, "A")])
        s = doubt.apply(s, 1, "doubt")
        v = doubt.view(s, 2)
        self.assertEqual(len(walk_cards(v, skip=("hand",))), 2)
        self.assertEqual(sorted(c[0] for c in reveals(s)[0].cards), sorted(c[0] for c in played))

    def test_view_of_an_unknown_seat_is_an_error(self):
        with self.assertRaises(ValueError):
            doubt.view(mk(3), 3)

    def test_finished_view_reports_the_winner(self):
        s = doubt.apply(doubt.apply(mk(3), 0, "forfeit"), 1, "forfeit")
        v = doubt.view(s)
        self.assertEqual((v.finished, v.winner), (True, 2))
        self.assertEqual(list(v.active), [False, False, True])


class ReplayTests(unittest.TestCase):
    def test_replay_equals_step_by_step(self):
        stakes = seats(4, 3)
        s = doubt.new("rp", stakes)
        moves = []
        for k in range(9):
            seat = doubt.view(s).turn
            move = mv(hand(s, seat)[: 1 + k % 3])
            moves.append((seat, move))
            s = doubt.apply(s, seat, move)
            if k % 3 == 1:
                moves.append(((seat + 2) % 4, "doubt"))
                s = doubt.apply(s, (seat + 2) % 4, "doubt")
        moves.append((1, "forfeit"))
        s = doubt.apply(s, 1, "forfeit")
        t = doubt.replay("rp", stakes, moves)
        for i in range(4):
            self.assertEqual(doubt.view(t, i), doubt.view(s, i))
        self.assertEqual(doubt.view(t), doubt.view(s))

    def test_replay_rejects_an_illegal_log(self):
        with self.assertRaises(IllegalMove):
            doubt.replay("rp", seats(3), [(1, "play:Ax1")])

    def test_replay_with_no_moves_is_the_deal(self):
        self.assertEqual(doubt.view(doubt.replay("x", seats(3), []), 0), doubt.view(mk(3, "x"), 0))


class RulesTests(unittest.TestCase):
    def test_defaults(self):
        want = {"claim_rule": "sequence", "ante": 0, "turn_timeout": 60}
        self.assertEqual(doubt.normalize_rules(), want)
        self.assertEqual(doubt.normalize_rules(None), want)
        self.assertEqual(doubt.normalize_rules({}), want)

    def test_dict_and_json_string_and_unknown_keys_dropped(self):
        got = doubt.normalize_rules({"ante": 25, "turn_timeout": 90, "bogus": 1})
        self.assertEqual(got, {"claim_rule": "sequence", "ante": 25, "turn_timeout": 90})
        self.assertEqual(doubt.normalize_rules(json.dumps({"ante": 7, "claim_rule": "sequence"})),
                         {"claim_rule": "sequence", "ante": 7, "turn_timeout": 60})

    def test_bounds(self):
        for key, lo, hi in (("ante", 0, 10 ** 9), ("turn_timeout", 15, 300)):
            self.assertEqual(doubt.normalize_rules({key: lo})[key], lo)
            self.assertEqual(doubt.normalize_rules({key: hi})[key], hi)
            for bad in (lo - 1, hi + 1, True, "10", 20.5, None):
                with self.subTest(key=key, bad=bad):
                    with self.assertRaises(ValueError):
                        doubt.normalize_rules({key: bad})

    def test_only_the_sequence_claim_rule_is_accepted(self):
        for bad in ("free", "", None, 1, "Sequence"):
            with self.assertRaises(ValueError):
                doubt.normalize_rules({"claim_rule": bad})

    def test_non_dict_values_are_rejected(self):
        for bad in ([], 5, "not json", "[1]", "null5"):
            with self.assertRaises(ValueError):
                doubt.normalize_rules(bad)

    def test_normalizing_is_idempotent(self):
        once = doubt.normalize_rules({"ante": 3})
        self.assertEqual(doubt.normalize_rules(once), once)

    def test_rules_text_and_how_to_play(self):
        rules = doubt.normalize_rules({"ante": 5})
        text = doubt.rules_text(rules)
        self.assertIsInstance(text, str)
        self.assertTrue(text.strip())
        self.assertNotIn("\n", text)
        guide = doubt.how_to_play(rules)
        self.assertGreater(guide.count("\n"), 2)
        for word in ("claim", "doubt", "pile"):
            self.assertIn(word, guide.lower())


class TimeoutMoveTests(unittest.TestCase):
    def test_holding_the_forced_rank_plays_one_truthfully(self):
        seed = seed_where(lambda sd: forced_card(mk(3, sd), 0) is not None)
        s = mk(3, seed)
        move = doubt.timeout_move(s, 0, "t")
        self.assertEqual(move, "play:Ax1")
        self.assertIn(move, doubt.legal_moves(s, 0))

    def test_without_the_forced_rank_plays_one_random_card_as_a_bluff(self):
        seed = seed_where(lambda sd: forced_card(mk(3, sd), 0) is None)
        s = mk(3, seed)
        owned = {c[0] for c in hand(s, 0)}
        choices = set()
        for k in range(30):
            move = doubt.timeout_move(s, 0, f"t{k}")
            self.assertEqual(move, doubt.timeout_move(s, 0, f"t{k}"))
            self.assertIn(move, doubt.legal_moves(s, 0))
            self.assertRegex(move, r"^play:[^,]+x1$")
            rank = move[len("play:"):-2]
            self.assertIn(rank, owned)
            self.assertNotEqual(rank, "A")
            choices.add(move)
        self.assertGreater(len(choices), 1)

    def test_only_valid_for_the_seat_on_turn(self):
        s = mk(3)
        with self.assertRaises(IllegalMove):
            doubt.timeout_move(s, 1, "t")
        s = doubt.apply(s, 0, mv(hand(s, 0)[:1]))
        with self.assertRaises(IllegalMove):
            doubt.timeout_move(s, 2, "t")
        doubt.apply(s, 1, doubt.timeout_move(s, 1, "t"))  # the next seat is on turn and the move applies


class PolicyTests(unittest.TestCase):
    def test_tendencies_are_deterministic_bounded_and_vary(self):
        seen = set()
        for cid in range(60):
            b, sus = doubt.tendencies(cid)
            self.assertEqual((b, sus), doubt.tendencies(cid))
            self.assertTrue(0.0 <= b <= 1.0 and 0.0 <= sus <= 1.0)
            seen.add((b, sus))
        self.assertGreater(len(seen), 30)
        self.assertEqual(doubt.tendencies(7), doubt.tendencies(7))

    def test_truthful_play_when_holding_the_forced_rank_and_never_bluffing(self):
        seed = seed_where(lambda sd: forced_card(mk(3, sd), 0) is not None)
        s = mk(3, seed)
        held = Counter(c[0] for c in hand(s, 0))["A"]
        move = doubt.policy_play(doubt.view(s, 0), (0.0, 0.5), "p")
        self.assertEqual(move, f"play:Ax{min(held, 4)}")

    def test_bluff_of_one_card_when_not_holding_the_rank_and_never_bluffing(self):
        seed = seed_where(lambda sd: forced_card(mk(3, sd), 0) is None)
        s = mk(3, seed)
        move = doubt.policy_play(doubt.view(s, 0), (0.0, 0.5), "p")
        self.assertIn(move, doubt.legal_moves(s, 0))
        self.assertRegex(move, r"^play:[^,]+x1$")

    def test_plays_are_legal_and_deterministic_for_every_tendency(self):
        for k in range(12):
            s = mk(4, f"pp{k}")
            for t in ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (1.0, 0.0)):
                for seed in ("a", "b", "c"):
                    move = doubt.policy_play(doubt.view(s, 0), t, seed)
                    self.assertIn(move, plays(s, 0))
                    self.assertEqual(move, doubt.policy_play(doubt.view(s, 0), t, seed))

    def test_policy_play_uses_only_the_seat_view(self):
        s = mk(3, "view-only")
        v = doubt.view(s, 0)
        self.assertIn(doubt.policy_play(v, (0.7, 0.3), "x"), doubt.legal_moves(s, 0))

    def test_impossible_matches_the_card_counts(self):
        checked = 0
        for n, decks in ((3, 1), (6, 2)):
            for k in range(40):
                for count in (1, 2, 3, 4):
                    s = mk(n, f"imp{n}-{k}")
                    s = doubt.apply(s, 0, mv(hand(s, 0)[:count]))
                    for obs in range(1, n):
                        v = doubt.view(s, obs)
                        own = Counter(c[0] for c in v.hand)["A"]
                        want = own + count > 4 * decks
                        self.assertEqual(doubt.impossible(v), want, (n, k, count, obs))
                        checked += want
        self.assertGreater(checked, 5)  # the property was exercised on genuinely impossible claims

    def test_impossible_hand_built_boundaries(self):
        def v(decks, mine, count):
            hand = tuple(("A", "♠") for _ in range(mine)) + (("5", "♠"),)
            return doubt.View(1, (9, 9, 9), (True,) * 3, 2, "2", count, doubt.Window(0, count, "A"), (), False, None,
                              decks, hand)

        self.assertTrue(doubt.impossible(v(1, 2, 3)))
        self.assertFalse(doubt.impossible(v(1, 1, 3)))  # 1 + 3 == 4 copies: exactly possible
        self.assertFalse(doubt.impossible(v(1, 0, 4)))
        self.assertTrue(doubt.impossible(v(1, 1, 4)))
        self.assertFalse(doubt.impossible(v(2, 4, 4)))  # 8 copies in two decks
        self.assertTrue(doubt.impossible(v(2, 5, 4)))

    def test_one_card_claims_can_be_doubted_by_suspicious_characters_only(self):
        v = doubt.View(1, (10, 10, 10), (True,) * 3, 2, "2", 1, doubt.Window(0, 1, "A"), (), False, None, 1,
                       (("5", "♠"), ("7", "♥")))
        self.assertGreaterEqual(doubt.suspicion(v, (0.5, 0.7)), doubt.TRIAGE_MIN)
        self.assertGreaterEqual(doubt.suspicion(v, (0.5, 0.7)), doubt.DOUBT_BELOW)
        self.assertEqual(doubt.triage([(1, doubt.suspicion(v, (0.5, 0.8)), False)]), [1])
        self.assertLess(doubt.suspicion(v, (0.5, 0.3)), doubt.DOUBT_BELOW)
        self.assertLess(doubt.suspicion(v, (0.5, 0.3)), doubt.TRIAGE_MIN)
        self.assertTrue(any(doubt.policy_doubt(v, (0.5, 0.9), f"d{k}") for k in range(60)))
        self.assertFalse(any(doubt.policy_doubt(v, (0.5, 0.2), f"d{k}") for k in range(60)))

    def test_impossible_is_false_without_an_open_claim(self):
        self.assertFalse(doubt.impossible(doubt.view(mk(3), 1)))

    def test_suspicion_is_bounded_and_deterministic(self):
        for k in range(10):
            s = doubt.apply(mk(4, f"su{k}"), 0, mv(hand(mk(4, f"su{k}"), 0)[:2]))
            v = doubt.view(s, 1)
            for t in ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)):
                x = doubt.suspicion(v, t)
                self.assertTrue(0.0 <= x <= 1.0)
                self.assertEqual(x, doubt.suspicion(v, t))

    def test_suspicion_rises_with_claim_size_and_tendency(self):
        def small_and_big(sd):
            s = mk(3, sd)
            return Counter(c[0] for c in hand(s, 1))["A"] == 0 and Counter(c[0] for c in hand(s, 0))["A"] == 0

        seed = seed_where(small_and_big)
        base = mk(3, seed)
        one = doubt.view(doubt.apply(base, 0, mv(hand(base, 0)[:1])), 1)
        four = doubt.view(doubt.apply(base, 0, mv(hand(base, 0)[:4])), 1)
        self.assertFalse(doubt.impossible(four))
        t = (0.5, 0.5)
        self.assertGreater(doubt.suspicion(four, t), doubt.suspicion(one, t))
        self.assertGreater(doubt.suspicion(four, (0.5, 0.9)), doubt.suspicion(four, (0.5, 0.1)))

    def test_suspicion_is_higher_when_the_claimer_emptied_their_hand(self):
        def ok(sd):
            st, _ = drive_to_empty(sd)
            return not doubt.impossible(doubt.view(st, (doubt.view(st).window.claimer + 1) % 3))

        sd = seed_where(ok)
        s, _ = drive_to_empty(sd)
        c = doubt.view(s).window.claimer
        end = doubt.view(s, (c + 1) % 3)
        start_state = mk(3, sd)
        start = doubt.view(doubt.apply(start_state, 0, mv(hand(start_state, 0)[:1])), 1)
        self.assertFalse(doubt.impossible(end))
        self.assertGreater(doubt.suspicion(end, (0.5, 0.5)), doubt.suspicion(start, (0.5, 0.5)))

    def test_doubt_is_certain_on_an_impossible_claim(self):
        def has_ace_for_obs(sd):
            return Counter(c[0] for c in hand(mk(3, sd), 1))["A"] >= 1

        s = mk(3, seed_where(has_ace_for_obs))
        s = doubt.apply(s, 0, mv(hand(s, 0)[:4]))
        v = doubt.view(s, 1)
        self.assertTrue(doubt.impossible(v))
        for t in ((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)):
            for seed in ("a", "b", "c", "d"):
                self.assertIs(doubt.policy_doubt(v, t, seed), True)

    def test_policy_doubt_is_a_deterministic_bool(self):
        for k in range(10):
            s = mk(5, f"pd{k}")
            s = doubt.apply(s, 0, mv(hand(s, 0)[:2]))
            v = doubt.view(s, 2)
            out = doubt.policy_doubt(v, (0.4, 0.6), "z")
            self.assertIsInstance(out, bool)
            self.assertEqual(out, doubt.policy_doubt(v, (0.4, 0.6), "z"))

    def test_policies_can_play_whole_turns_legally(self):
        for sd in ("g1", "g2", "g3"):
            n = 4
            s = mk(n, sd, stake=2)
            tend = [doubt.tendencies(i) for i in range(n)]
            for step in range(300):
                if doubt.view(s).finished:
                    break
                v = doubt.view(s)
                claimer = v.window.claimer if v.window else None
                doubted = False
                if v.window:
                    for i in range(n):
                        if i != claimer and v.active[i] and doubt.policy_doubt(doubt.view(s, i), tend[i], f"{sd}d{step}{i}"):
                            s = doubt.apply(s, i, "doubt")
                            doubted = True
                            break
                if doubted:
                    continue
                if v.window and v.counts[claimer] == 0:
                    s = doubt.apply(s, claimer, "close")
                    continue
                seat = v.turn
                move = doubt.policy_play(doubt.view(s, seat), tend[seat], f"{sd}p{step}")
                self.assertIn(move, doubt.legal_moves(s, seat))
                s = doubt.apply(s, seat, move)
            total = sum(doubt.view(s).counts) + doubt.view(s).pile
            self.assertEqual(total, 52)


class TriageTests(unittest.TestCase):
    def test_top_k_above_threshold_by_score(self):
        scored = [(0, 1.0, False), (1, 0.9999, False), (2, 0.9998, False), (3, 0.0, False)]
        self.assertEqual(sorted(doubt.triage(scored)), [0, 1])
        self.assertEqual(sorted(doubt.triage(scored, k=1)), [0])
        self.assertEqual(sorted(doubt.triage(scored, k=3)), [0, 1, 2])

    def test_ties_break_by_seat_index(self):
        scored = [(3, 1.0, False), (1, 1.0, False), (2, 1.0, False)]
        self.assertEqual(sorted(doubt.triage(scored)), [1, 2])
        self.assertEqual(sorted(doubt.triage(list(reversed(scored)))), [1, 2])

    def test_low_scores_are_never_asked(self):
        self.assertEqual(doubt.triage([(0, 0.0, False), (1, 0.0, False)]), [])
        self.assertEqual(doubt.triage([]), [])

    def test_impossible_seats_are_always_included(self):
        scored = [(0, 0.0, True), (1, 1.0, False), (2, 0.9999, False), (3, 0.9998, False), (4, 0.0, True)]
        got = doubt.triage(scored)
        self.assertTrue({0, 4} <= set(got))
        self.assertIn(1, got)
        self.assertNotIn(3, got)
        self.assertEqual(set(doubt.triage(scored, k=0)), {0, 4})

    def test_is_deterministic(self):
        scored = [(2, 1.0, False), (0, 1.0, False), (1, 0.5, True)]
        self.assertEqual(doubt.triage(scored), doubt.triage(scored))
        self.assertEqual(len(set(doubt.triage(scored))), len(doubt.triage(scored)))


class ShuffleContractTests(unittest.TestCase):
    def test_the_deal_is_a_shuffle_of_one_seeded_deck(self):
        """Different seeds shuffle differently and the same seed reproduces (shuffled is the only randomness)."""
        deck = list(range(52))
        self.assertEqual(shuffled(deck, "q"), shuffled(deck, "q"))
        self.assertNotEqual([hand(mk(3, "q"), i) for i in range(3)], [hand(mk(3, "r"), i) for i in range(3)])


if __name__ == "__main__":
    unittest.main()
