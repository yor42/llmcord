import json
import math
import random
import tempfile
import unittest
from pathlib import Path

from llmcord_core.spoken import blackjack
from llmcord_core.spoken.embedder import FakeEmbedder
from llmcord_core.spoken.head import BLACKJACK_PATH, BLACKJACK_SHA256, Head, HeadError, blackjack_head, load_head, parse_head
from llmcord_core.spoken.interpret import decide

TH = {"t_sure": 0.4, "margin": 0.1, "g_sure": 0.0, "t_floor": 0.2, "g_floor": -0.3}


def tiny(probs, th=None):
    """A one-dim head whose output probabilities are `probs` ({class: p}) for any vector."""
    classes = tuple(probs)
    return Head(classes, (0.0,), (1.0,), ((0.0,) * len(classes),), tuple(math.log(p) for p in probs.values()), 1.0, dict(th or TH), "m", "r", "query: ")


def raw_head():
    return json.loads(Path(BLACKJACK_PATH).read_text())


class MaskingTests(unittest.TestCase):
    def test_illegal_moves_dropped_and_renormalized(self):
        head = tiny({"hit": 0.4, "stand": 0.2, "double": 0.3, "talk": 0.1})
        d = decide([0.0], ["hit", "stand"], head)
        # kept: hit .4, stand .2, talk .1 -> total .7
        self.assertAlmostEqual(d.features.p1, 0.4 / 0.7, places=9)
        self.assertAlmostEqual(d.features.margin, 0.2 / 0.7, places=9)
        self.assertAlmostEqual(d.features.talk_gap, 0.3 / 0.7, places=9)
        self.assertNotIn("double", d.moves)

    def test_unknown_and_empty_legal(self):
        head = tiny({"hit": 0.9, "talk": 0.1})
        self.assertEqual(decide([0.0], [], head).band, "silent")
        self.assertEqual(decide([0.0], ["raise"], head).moves, ())
        self.assertEqual(decide([0.0], ["talk"], head).band, "silent")

    def test_insurance_phase_only_insurance_moves(self):
        head = blackjack_head()
        legal = blackjack.INSURANCE_MOVES
        r = random.Random(7)
        for _ in range(5):
            v = [m + s * r.gauss(0, 2) for m, s in zip(head.mu, head.sd)]
            self.assertTrue(set(decide(v, legal, head).moves) <= set(legal))

    def test_single_legal_move_at_most_one_button(self):
        head = tiny({"hit": 0.3, "stand": 0.5, "talk": 0.2}, {**TH, "t_sure": 0.9})
        d = decide([0.0], ["hit"], head)
        self.assertEqual(d.band, "unsure")
        self.assertEqual(d.moves, ("hit",))


class BandTests(unittest.TestCase):
    def head(self, **th):
        return tiny({"hit": 0.5, "stand": 0.3, "talk": 0.2}, {**TH, **th})

    def test_sure(self):
        d = decide([0.0], ["hit", "stand"], self.head())
        self.assertEqual((d.band, d.moves), ("sure", ("hit",)))

    def test_sure_edges(self):
        legal = ["hit", "stand"]  # renormalized over all three: p1 .5, margin .2, gap .3
        self.assertEqual(decide([0.0], legal, self.head(t_sure=0.5 - 1e-6)).band, "sure")
        self.assertEqual(decide([0.0], legal, self.head(t_sure=0.5 + 1e-6)).band, "unsure")
        self.assertEqual(decide([0.0], legal, self.head(margin=0.2 - 1e-6)).band, "sure")
        self.assertEqual(decide([0.0], legal, self.head(margin=0.2 + 1e-6)).band, "unsure")
        self.assertEqual(decide([0.0], legal, self.head(g_sure=0.3 - 1e-6)).band, "sure")
        self.assertEqual(decide([0.0], legal, self.head(g_sure=0.3 + 1e-6)).band, "unsure")

    def test_unsure_offers_two_buttons(self):
        d = decide([0.0], ["hit", "stand"], self.head(margin=0.5))
        self.assertEqual((d.band, d.moves), ("unsure", ("hit", "stand")))

    def test_silent_edges(self):
        legal = ["hit", "stand"]
        loose = dict(t_sure=2.0)
        self.assertEqual(decide([0.0], legal, self.head(**loose, t_floor=0.5 - 1e-6)).band, "unsure")
        self.assertEqual(decide([0.0], legal, self.head(**loose, t_floor=0.5 + 1e-6)).band, "silent")
        self.assertEqual(decide([0.0], legal, self.head(**loose, g_floor=0.3 - 1e-6)).band, "unsure")
        self.assertEqual(decide([0.0], legal, self.head(**loose, g_floor=0.3 + 1e-6)).band, "silent")

    def test_talk_gap_floor(self):
        head = tiny({"hit": 0.3, "stand": 0.1, "talk": 0.6}, {**TH, "t_sure": 2.0, "t_floor": 0.1, "g_floor": -0.2})
        self.assertEqual(decide([0.0], ["hit", "stand"], head).band, "silent")  # gap -0.3 < -0.2
        head = tiny({"hit": 0.3, "stand": 0.1, "talk": 0.6}, {**TH, "t_sure": 2.0, "t_floor": 0.1, "g_floor": -0.4})
        self.assertEqual(decide([0.0], ["hit", "stand"], head).band, "unsure")


class HeadLoadTests(unittest.TestCase):
    def test_shipped_head_loads(self):
        head = blackjack_head()
        self.assertIs(head, blackjack_head())
        self.assertIn("talk", head.classes)
        self.assertEqual((head.dim, len(head.W[0])), (384, len(head.classes)))
        self.assertEqual(head.prefix, "query: ")
        self.assertTrue(set(blackjack.MOVES) <= set(head.classes))

    def test_sha_mismatch_rejected(self):
        with self.assertRaises(HeadError):
            load_head(BLACKJACK_PATH, "0" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "h.json"
            p.write_text(json.dumps({**raw_head(), "T": 2.0}))
            with self.assertRaises(HeadError):
                load_head(p, BLACKJACK_SHA256)

    def test_malformed_rejected(self):
        good = raw_head()
        parse_head(good)
        bad = [
            {**good, "classes": [c for c in good["classes"] if c != "talk"]},
            {**good, "W": good["W"][:-1]},
            {**good, "b": good["b"][:-1]},
            {**good, "sd": good["sd"][:-1]},
            {**good, "thresholds": {k: v for k, v in good["thresholds"].items() if k != "margin"}},
            {k: v for k, v in good.items() if k != "model_revision"},
            {k: v for k, v in good.items() if k != "prefix"},
            {**good, "model": ""},
            {**good, "T": 0},
            {**good, "W": [r[:-1] for r in good["W"]]},
            [],
        ]
        for data in bad:
            with self.assertRaises(HeadError, msg=str(type(data))):
                parse_head(data)

    def test_wrong_vector_size(self):
        with self.assertRaises(HeadError):
            decide([0.0] * 3, ["hit"], blackjack_head())


class GoldenTests(unittest.TestCase):
    """Expected values computed with numpy in the prototype venv (prototypes/spoken_moves bands.py + run.py head_proba)."""
    CLASSES = ["double", "even_money", "hit", "insure", "no_insurance", "stand", "surrender", "talk"]
    CASES = [
        (1, None, 0, [0.002513, 0.002274, 0.005098, 0.001929, 0.004135, 0.011456, 0.005746, 0.966848], ("silent", ()), ("silent", ())),
        (2, "hit", 10, [6.1e-05, 2.4e-05, 0.998636, 2e-05, 5e-06, 2.8e-05, 5.1e-05, 0.001175], ("sure", ("hit",)), ("silent", ())),
        (3, "double", 6, [0.296261, 0.003201, 0.005669, 0.003955, 0.004468, 0.004122, 0.009687, 0.672638], ("sure", ("double",)), ("silent", ())),
        (4, "hit", 4, [0.006846, 0.00447, 0.270705, 0.004779, 0.002257, 0.03463, 0.006374, 0.669941], ("sure", ("hit",)), ("silent", ())),
        (5, "insure", 8, [0.006773, 0.004732, 0.008445, 0.339003, 0.003503, 0.014369, 0.018344, 0.604831], ("silent", ()), ("sure", ("insure",))),
    ]
    PLAY = ["hit", "stand", "double", "surrender"]

    def vector(self, head, seed, cls, k):
        r = random.Random(seed)
        g = [r.gauss(0, 1) for _ in range(head.dim)]
        j = head.classes.index(cls) if cls else None
        return [m + s * (0.5 * x + (k * head.W[i][j] if cls else 0)) for i, (m, s, x) in enumerate(zip(head.mu, head.sd, g))]

    def test_matches_prototype(self):
        head = blackjack_head()
        self.assertEqual(list(head.classes), self.CLASSES)
        for seed, cls, k, probs, play, ins in self.CASES:
            v = self.vector(head, seed, cls, k)
            got = head.probabilities(v)
            for c, p in zip(self.CLASSES, probs):
                self.assertAlmostEqual(got[c], p, delta=2e-6, msg=f"{seed} {c}")
            for legal, want in ((self.PLAY, play), (blackjack.INSURANCE_MOVES, ins)):
                d = decide(v, legal, head)
                self.assertEqual((d.band, d.moves), want, msg=f"{seed} {legal[0]}")

    def test_masked_features_match_prototype(self):
        head = blackjack_head()
        d = decide(self.vector(head, 2, "hit", 10), self.PLAY, head)
        for got, want in zip((d.features.p1, d.features.margin, d.features.talk_gap), (0.998686, 0.998625, 0.997511)):
            self.assertAlmostEqual(got, want, delta=2e-6)


class SeamTests(unittest.TestCase):
    def test_fake_embedder(self):
        e = FakeEmbedder({"hit me": [1.0, 2.0]}, dim=2)
        self.assertEqual(e.embed(["hit me", "other"]), [[1.0, 2.0], [0.0, 0.0]])
        self.assertEqual(e.prefix, "query: ")

    def test_blackjack_labels(self):
        self.assertEqual(set(blackjack.LABELS), set(blackjack.MOVES))
        self.assertEqual(blackjack.LABELS["no_insurance"], "No insurance")


if __name__ == "__main__":
    unittest.main()
