"""Play style traits (llmcord_core/games/playstyle.py), FEAT-32 part A (D31).

Pure functions: no Discord, DB, model or clock.
"""
import dataclasses
import json
import unittest

from llmcord_core.games import playstyle as ps

TRAIT_NAMES = (
    "looseness", "aggression", "bluffing", "stubbornness",
    "suspicion", "risk", "judgment", "unpredictability",
)


def _style(**kw):
    return ps.PlayStyle(**kw)


class ConstantsTests(unittest.TestCase):
    def test_traits_neutral_groups(self):
        """FEAT-32: public constants match the contract."""
        self.assertEqual(ps.TRAITS, TRAIT_NAMES)
        self.assertEqual(ps.NEUTRAL, 50)
        self.assertEqual(ps.GROUPS, {
            "Style": ("looseness", "aggression", "bluffing", "risk"),
            "Reads": ("suspicion", "stubbornness"),
            "Mind": ("judgment", "unpredictability"),
        })
        grouped = [t for g in ps.GROUPS.values() for t in g]
        self.assertCountEqual(grouped, ps.TRAITS)

    def test_dataclass_defaults_frozen_unit(self):
        """FEAT-32: PlayStyle is frozen, defaults to 50, unit is value/100."""
        s = ps.PlayStyle()
        for t in ps.TRAITS:
            self.assertEqual(getattr(s, t), 50)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            s.risk = 3
        self.assertEqual(_style(risk=80).unit("risk"), 0.8)
        self.assertEqual(_style(risk=0).unit("risk"), 0.0)
        self.assertEqual(_style(risk=100).unit("risk"), 1.0)


class ParseTests(unittest.TestCase):
    def test_missing_traits_default_neutral(self):
        self.assertEqual(ps.parse({}), ps.PlayStyle())
        s = ps.parse({"risk": 10})
        self.assertEqual(s.risk, 10)
        self.assertEqual(s.bluffing, 50)

    def test_bounds_accepted(self):
        s = ps.parse({"risk": 0, "judgment": 100})
        self.assertEqual((s.risk, s.judgment), (0, 100))

    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError):
            ps.parse({"charisma": 50})

    def test_non_int_rejected(self):
        for bad in ("50", 50.0, 50.5, None, True, False, [50]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ps.parse({"risk": bad})

    def test_out_of_range_rejected(self):
        for bad in (-1, 101, 1000):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ps.parse({"risk": bad})


class ClampTests(unittest.TestCase):
    def test_lenient_numbers(self):
        s = ps.clamp({"risk": 150, "looseness": -20, "aggression": 70.6,
                      "bluffing": "30", "suspicion": " 80 ", "judgment": "42.4"})
        self.assertEqual(s.risk, 100)
        self.assertEqual(s.looseness, 0)
        self.assertEqual(s.aggression, 71)
        self.assertEqual(s.bluffing, 30)
        self.assertEqual(s.suspicion, 80)
        self.assertEqual(s.judgment, 42)

    def test_unknown_ignored_missing_neutral(self):
        s = ps.clamp({"charisma": 99, "risk": 10})
        self.assertEqual(s.risk, 10)
        self.assertEqual(s.stubbornness, 50)

    def test_garbage_becomes_neutral(self):
        for bad in (True, False, None, "abc", "", float("nan"), float("inf"),
                    float("-inf"), "nan", "inf", [], {}):
            with self.subTest(bad=bad):
                self.assertEqual(ps.clamp({"risk": bad}).risk, 50)

    def test_non_dict_raises(self):
        for bad in (None, [], "x", 5, [("risk", 1)]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ps.clamp(bad)

    def test_result_is_always_valid(self):
        s = ps.clamp({t: 1e9 for t in ps.TRAITS})
        self.assertEqual(ps.parse(dataclasses.asdict(s)), s)


class SerializationTests(unittest.TestCase):
    def test_dumps_canonical(self):
        text = ps.dumps(_style(risk=7))
        data = json.loads(text)
        self.assertEqual(set(data), set(ps.TRAITS))
        self.assertEqual(data["risk"], 7)
        self.assertEqual(text, json.dumps(data, sort_keys=True,
                                          separators=(",", ":")) if " " not in text
                         else json.dumps(data, sort_keys=True))
        self.assertEqual(list(data), sorted(data))

    def test_round_trip(self):
        s = ps.PlayStyle(0, 100, 1, 99, 50, 33, 67, 12)
        self.assertEqual(ps.loads(ps.dumps(s)), s)

    def test_loads_empty_is_none(self):
        self.assertIsNone(ps.loads(""))
        self.assertIsNone(ps.loads(None))

    def test_loads_invalid_is_none(self):
        for bad in ("{", "not json", "[]", "5", "null", '{"risk": 101}',
                    '{"risk": "x"}', '{"nope": 1}', '{"risk": true}'):
            with self.subTest(bad=bad):
                self.assertIsNone(ps.loads(bad))


class DerivedTests(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(ps.derived_from_id(42), ps.derived_from_id(42))

    def test_only_bluffing_and_suspicion_vary(self):
        for cid in range(1, 201):
            s = ps.derived_from_id(cid)
            for t in ps.TRAITS:
                if t not in ("bluffing", "suspicion"):
                    self.assertEqual(getattr(s, t), 50, (cid, t))
                self.assertTrue(0 <= getattr(s, t) <= 100)

    def test_variety(self):
        styles = [ps.derived_from_id(i) for i in range(1, 201)]
        self.assertGreaterEqual(len({s.bluffing for s in styles}), 20)
        self.assertGreaterEqual(len({s.suspicion for s in styles}), 20)


class ResolveTests(unittest.TestCase):
    def test_saved_wins(self):
        s = _style(risk=3)
        self.assertEqual(ps.resolve(ps.dumps(s), 7), s)

    def test_fallback_derived(self):
        for text in ("", None, "garbage", '{"risk": 500}'):
            with self.subTest(text=text):
                self.assertEqual(ps.resolve(text, 7), ps.derived_from_id(7))


class RollTests(unittest.TestCase):
    def test_deterministic_and_range(self):
        self.assertEqual(ps.roll("s", "t"), ps.roll("s", "t"))
        for i in range(200):
            self.assertTrue(0.0 <= ps.roll("seed", f"t{i}") < 1.0)

    def test_differs(self):
        self.assertNotEqual(ps.roll("s", "a"), ps.roll("s", "b"))
        self.assertNotEqual(ps.roll("s1", "a"), ps.roll("s2", "a"))

    def test_roughly_uniform(self):
        vals = [ps.roll("seed", f"tag{i}") for i in range(2000)]
        self.assertTrue(0.45 < sum(vals) / len(vals) < 0.55)


class ErraticTests(unittest.TestCase):
    def test_cap_constant(self):
        self.assertEqual(ps.MAX_ERRATIC, 0.35)

    def test_zero_never(self):
        s = _style(unpredictability=0)
        self.assertFalse(any(ps.erratic(s, "seed", f"t{i}") for i in range(500)))

    def test_max_fraction(self):
        s = _style(unpredictability=100)
        n = sum(ps.erratic(s, "seed", f"t{i}") for i in range(2000))
        self.assertTrue(0.2 < n / 2000 < 0.5, n)

    def test_neutral_strictly_between(self):
        n0 = 0
        n50 = sum(ps.erratic(_style(), "seed", f"t{i}") for i in range(2000))
        n100 = sum(ps.erratic(_style(unpredictability=100), "seed", f"t{i}")
                   for i in range(2000))
        self.assertGreater(n50, n0)
        self.assertLess(n50, n100)

    def test_deterministic(self):
        s = _style(unpredictability=80)
        a = [ps.erratic(s, "seed", f"t{i}") for i in range(100)]
        b = [ps.erratic(s, "seed", f"t{i}") for i in range(100)]
        self.assertEqual(a, b)
        self.assertIsInstance(a[0], bool)


class MisjudgeTests(unittest.TestCase):
    def test_sharp_nearly_exact(self):
        s = _style(judgment=100)
        for i in range(200):
            for strength in (0.0, 0.3, 0.7, 1.0):
                out = ps.misjudge(strength, s, "seed", f"t{i}")
                self.assertLessEqual(abs(out - strength), 0.02 + 1e-9)
                self.assertTrue(0.0 <= out <= 1.0)

    def test_neutral_close(self):
        s = _style()
        for i in range(200):
            for strength in (0.0, 0.3, 0.7, 1.0):
                out = ps.misjudge(strength, s, "seed", f"t{i}")
                self.assertLessEqual(abs(out - strength), 0.05 + 1e-9)

    def test_foolish_overvalues_weak(self):
        s = _style(judgment=0)
        outs = [ps.misjudge(0.3, s, "seed", f"t{i}") for i in range(500)]
        self.assertTrue(all(0.0 <= o <= 1.0 for o in outs))
        self.assertGreaterEqual(sum(outs) / len(outs) - 0.3, 0.15)

    def test_bias_systematic_across_seeds(self):
        s = _style(judgment=0)
        for seed in ("a", "b", "c", "d"):
            outs = [ps.misjudge(0.3, s, seed, f"t{i}") for i in range(300)]
            self.assertGreater(sum(outs) / len(outs), 0.3)

    def test_clamps_input_and_output(self):
        s = _style(judgment=0)
        for strength in (-5.0, 0.0, 1.0, 7.0):
            for i in range(100):
                out = ps.misjudge(strength, s, "seed", f"t{i}")
                self.assertTrue(0.0 <= out <= 1.0)
        self.assertEqual(ps.misjudge(2.0, _style(judgment=100), "s", "t") <= 1.0, True)
        self.assertGreaterEqual(ps.misjudge(2.0, _style(judgment=100), "s", "t"), 0.98)

    def test_deterministic(self):
        s = _style(judgment=10)
        self.assertEqual(ps.misjudge(0.4, s, "seed", "t"), ps.misjudge(0.4, s, "seed", "t"))


class DescribeTests(unittest.TestCase):
    def test_neutral_balanced(self):
        self.assertEqual(ps.describe(ps.PlayStyle()), "balanced")

    def test_neutral_band_omitted(self):
        for v in (40, 50, 60):
            self.assertEqual(ps.describe(_style(looseness=v, risk=v)), "balanced")

    def test_strong_words(self):
        cases = {
            ("looseness", 75): "loose", ("looseness", 25): "tight",
            ("aggression", 90): "aggressive", ("aggression", 10): "passive",
            ("bluffing", 100): "bluffs often", ("bluffing", 0): "rarely bluffs",
            ("stubbornness", 80): "stubborn", ("stubbornness", 20): "quick to fold",
            ("suspicion", 75): "suspicious", ("suspicion", 25): "trusting",
            ("risk", 99): "reckless", ("risk", 1): "cautious",
            ("judgment", 25): "foolish", ("judgment", 75): "sharp",
            ("unpredictability", 75): "erratic", ("unpredictability", 25): "steady",
        }
        for (trait, value), word in cases.items():
            with self.subTest(trait=trait, value=value):
                self.assertEqual(ps.describe(_style(**{trait: value})), word)

    def test_moderate_prefix(self):
        for value in (26, 39, 61, 74):
            text = ps.describe(_style(looseness=value))
            self.assertEqual(text, "somewhat " + ("tight" if value < 50 else "loose"))
        self.assertEqual(ps.describe(_style(bluffing=65)), "somewhat bluffs often")
        self.assertEqual(ps.describe(_style(judgment=30)), "somewhat foolish")

    def test_order_follows_traits(self):
        text = ps.describe(_style(unpredictability=90, looseness=90, judgment=10))
        words = [w.strip() for w in text.split(",")]
        self.assertEqual(words, ["loose", "foolish", "erratic"])

    def test_returns_string_without_balanced_when_any(self):
        self.assertNotIn("balanced", ps.describe(_style(risk=90)))


class NeverFoldFreeTests(unittest.TestCase):
    def test_fold_becomes_check(self):
        self.assertEqual(ps.never_fold_free(["fold", "check", "raise"], "fold"), "check")

    def test_fold_kept_when_cannot_check(self):
        self.assertEqual(ps.never_fold_free(["fold", "call", "raise"], "fold"), "fold")

    def test_other_moves_unchanged(self):
        for move in ("check", "call", "raise", "stand", "hit"):
            self.assertEqual(ps.never_fold_free(["fold", "check"], move), move)


if __name__ == "__main__":
    unittest.main()
