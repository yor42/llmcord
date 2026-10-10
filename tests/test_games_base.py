"""Seeds, the stable shuffle and the default policy (llmcord_core/games/base.py)."""
import asyncio
import unittest
from unittest import mock

from llmcord_core.games import base
from llmcord_core.games import DefaultPolicy, IllegalMove, Seat, new_seed, seed_hash, short_seed_hash, shuffled


class SeedTests(unittest.TestCase):
    def test_new_seed_is_128_bit_hex_and_unique(self):
        a, b = new_seed(), new_seed()
        self.assertEqual(len(a), 32)
        int(a, 16)
        self.assertNotEqual(a, b)

    def test_seed_hash_is_sha256_and_short_form_is_prefix(self):
        self.assertEqual(
            seed_hash("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.assertEqual(short_seed_hash("abc"), "ba7816bf8f01")

    def test_shuffle_golden(self):
        self.assertEqual(shuffled(range(10), "golden"), [2, 1, 6, 5, 8, 3, 4, 0, 9, 7])

    def test_shuffle_is_permutation_stable_and_seed_dependent(self):
        items = list(range(312))
        a = shuffled(items, "s1")
        self.assertEqual(sorted(a), items)
        self.assertEqual(a, shuffled(items, "s1"))
        self.assertNotEqual(a, shuffled(items, "s2"))
        self.assertEqual(items, list(range(312)))

    def test_shuffle_small_inputs(self):
        self.assertEqual(shuffled([], "x"), [])
        self.assertEqual(shuffled([7], "x"), [7])


class ShuffleDistributionTests(unittest.TestCase):
    def test_rejection_branch_fires_and_result_is_permutation(self):
        pulled = []

        def stream(seed):
            for v in (2**32 - 1, 2**32 - 2, 1, 0, 5, 7):
                pulled.append(v)
                yield v

        with mock.patch.object(base, "_randoms", stream):
            out = shuffled([0, 1, 2], "x")
        self.assertEqual(sorted(out), [0, 1, 2])
        # n=3 rejects values >= 2**32-1: the first draw is thrown away, so more than 2 values are used
        self.assertGreater(len(pulled), 2)

    def test_three_item_orders_roughly_uniform(self):
        counts = {}
        for n in range(3000):
            key = tuple(shuffled([0, 1, 2], f"seed-{n}"))
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(counts), 6)
        for v in counts.values():
            self.assertTrue(400 < v < 600, counts)


class SeatTests(unittest.TestCase):
    def test_validation(self):
        Seat(0, "member", 1, 5)
        Seat(1, "character", 2, 5)
        with self.assertRaises(ValueError):
            Seat(0, "bot", 1, 5)
        Seat(2, "member", 3, 0)  # FEAT-21: an ante of 0 is a valid stake (games without a bet)
        for bad in (-5, -1, True, 5.0, "5", None):
            with self.assertRaises(ValueError):
                Seat(0, "member", 1, bad)


class BlackjackStakeTests(unittest.TestCase):
    def test_blackjack_still_rejects_a_zero_stake(self):
        from llmcord_core.games import blackjack
        blackjack.new("s", [Seat(0, "member", 1, 5)])
        with self.assertRaises(ValueError):
            blackjack.new("s", [Seat(0, "member", 1, 0)])
        with self.assertRaises(ValueError):
            blackjack.new("s", [Seat(0, "member", 1, 5), Seat(1, "member", 2, 0)])


class PolicyTests(unittest.TestCase):
    class V:
        def __init__(self, total):
            self.total = total

    def test_hits_below_17_stands_otherwise_never_doubles(self):
        legal = ("hit", "stand", "double")
        self.assertEqual(DefaultPolicy.pick(self.V(16), legal), "hit")
        self.assertEqual(DefaultPolicy.pick(self.V(17), legal), "stand")
        self.assertEqual(DefaultPolicy.pick(self.V(9), ("stand",)), "stand")
        self.assertEqual(asyncio.run(DefaultPolicy().choose(self.V(12), legal)), "hit")

    def test_no_legal_moves_raises(self):
        with self.assertRaises(IllegalMove):
            DefaultPolicy.pick(self.V(10), ())
