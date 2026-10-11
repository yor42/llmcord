"""FEAT-36 prototype harness: band selection, pass rule, threshold fit and CV folds (prototypes/spoken_moves/bands.py).

Pure python on fake scores; the embedding models themselves are not exercised here.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prototypes" / "spoken_moves"))

import bands  # noqa: E402
import wordrule  # noqa: E402

TH = {"t_sure": 0.8, "margin": 0.1, "g_sure": 0.05, "t_floor": 0.6, "g_floor": 0.0}


def row(label, scores, talk=0.0, accept=(), word=None, lang="en", rid="x"):
    return {"id": rid, "text": rid, "lang": lang, "label": label, "accept": tuple(accept), "scores": scores,
            "talk": talk, "word": word}


class BandTests(unittest.TestCase):
    def band(self, scores, talk, **kw):
        return bands.band_of(bands.features(scores, talk), {**TH, **kw})

    def test_features_rank_moves_and_gaps(self):
        m1, m2, s1, margin, gap = bands.features({"hit": 0.9, "stand": 0.5, "double": 0.7}, 0.4)
        self.assertEqual((m1, m2), ("hit", "double"))
        self.assertAlmostEqual(margin, 0.2)
        self.assertAlmostEqual(gap, 0.5)
        self.assertEqual(bands.features({"hit": 0.9}, 0.4)[1], None)

    def test_sure_needs_score_margin_and_talk_gap(self):
        self.assertEqual(self.band({"hit": 0.9, "stand": 0.5}, 0.4), ("sure", ["hit"]))
        self.assertEqual(self.band({"hit": 0.9, "stand": 0.85}, 0.4), ("unsure", ["hit", "stand"]))
        self.assertEqual(self.band({"hit": 0.9, "stand": 0.5}, 0.95), ("silent", []))
        self.assertEqual(self.band({"hit": 0.7, "stand": 0.3}, 0.2), ("unsure", ["hit", "stand"]))

    def test_silent_below_floor_or_talk_gap_floor(self):
        self.assertEqual(self.band({"hit": 0.5, "stand": 0.3}, 0.1), ("silent", []))
        self.assertEqual(self.band({"hit": 0.7, "stand": 0.3}, 0.9), ("silent", []))

    def test_single_legal_move_gives_one_button(self):
        self.assertEqual(self.band({"hit": 0.7}, 0.2), ("unsure", ["hit"]))

    def test_pass_rule(self):
        mv = row("hit", {}, accept=("stand",))
        self.assertTrue(bands.passes(mv, ["hit"]))
        self.assertTrue(bands.passes(mv, ["double", "stand"]))
        self.assertFalse(bands.passes(mv, ["double"]))
        self.assertFalse(bands.passes(mv, []))
        talk = row("talk", {}, accept=("hit",))
        self.assertTrue(bands.passes(talk, []))
        self.assertTrue(bands.passes(talk, ["hit"]))
        self.assertFalse(bands.passes(talk, ["stand"]))
        self.assertFalse(bands.passes(row("talk", {}), ["hit"]))

    def test_cascade_word_list_first_and_pad_unsure(self):
        r = row("stand", {"hit": 0.9, "stand": 0.5, "double": 0.1}, word=("sure", ["stand"]))
        self.assertEqual(bands.decide(r, TH, True), ("sure", ["stand"]))
        self.assertEqual(bands.decide(r, TH, False)[0], "sure")
        self.assertEqual(bands.decide(r, TH, False)[1], ["hit"])
        u = row("stand", {"hit": 0.9, "stand": 0.5}, word=("unsure", ["stand"]))
        self.assertEqual(bands.decide(u, TH, True), ("unsure", ["stand", "hit"]))
        two = row("stand", {"hit": 0.9, "stand": 0.5, "double": 0.4}, word=("unsure", ["double", "stand"]))
        self.assertEqual(bands.decide(two, TH, True), ("unsure", ["double", "stand"]))
        miss = row("stand", {"hit": 0.9, "stand": 0.5}, word=None)
        self.assertEqual(bands.decide(miss, TH, True), bands.decide(miss, TH, False))

    def test_evaluate_counts_sure_wrong_and_breakdowns(self):
        rows = [row("hit", {}, lang="en", rid="a"), row("stand", {}, lang="ko", rid="b"), row("talk", {}, lang="ko", rid="c")]
        m = bands.evaluate(rows, [("sure", ["hit"]), ("sure", ["hit"]), ("silent", [])])
        self.assertEqual((m["move_pass"], m["move_n"], m["sure_right"], m["sure_wrong"]), (1, 2, 1, 1))
        self.assertEqual((m["talk_silent"], m["talk_n"]), (1, 1))
        self.assertEqual(m["by_lang"], {"en": (1, 1), "ko": (2, 1)})
        self.assertEqual([f[0]["id"] for f in m["failures"]], ["b"])


def separable(n=20):
    """Moves score high with a talk gap, talk scores high on s1 but the gap is small for some."""
    rows = []
    for i in range(n):
        rows.append(row("hit", {"hit": 0.9, "stand": 0.3}, talk=0.4, rid=f"h{i}"))
        rows.append(row("stand", {"hit": 0.3, "stand": 0.88}, talk=0.4, rid=f"s{i}"))
        rows.append(row("talk", {"hit": 0.5 + i * 0.01, "stand": 0.3}, talk=0.7, rid=f"t{i}"))
    return rows


class FitTests(unittest.TestCase):
    def test_fit_honours_move_constraint_and_silences_talk(self):
        rows = separable()
        th, m = bands.fit(rows, False)
        self.assertGreaterEqual(m["move_rate"], 0.95)
        self.assertGreaterEqual(m["talk_rate"], 0.9)
        self.assertEqual(bands.run(rows, th, False)["move_pass"], m["move_pass"])

    def test_fit_constraint_beats_talk_silence(self):
        rows = [row("hit", {"hit": 0.6, "stand": 0.3}, talk=0.59, rid=f"h{i}") for i in range(20)]
        rows += [row("talk", {"hit": 0.6, "stand": 0.3}, talk=0.59, rid=f"t{i}") for i in range(5)]
        _, m = bands.fit(rows, False)
        self.assertGreaterEqual(m["move_rate"], 0.95)
        self.assertEqual(m["talk_rate"], 0.0)

    def test_fit_without_feasible_point_maximizes_move_pass(self):
        rows = [row("hit", {"hit": 0.1, "stand": 0.9}, talk=0.0, rid="a"), row("hit", {"hit": 0.9, "stand": 0.1}, rid="b")]
        _, m = bands.fit(rows, False, min_move=1.01)
        self.assertEqual(m["move_pass"], 2)

    def test_cascade_fit_counts_word_rows_as_fixed(self):
        rows = separable(10)
        rows[0]["word"] = ("sure", ["hit"])
        _, m = bands.fit(rows, True)
        self.assertGreaterEqual(m["move_rate"], 0.95)
        self.assertEqual(m["sure_wrong"], 0)

    def test_fast_fit_matches_decide(self):
        rows = separable(8)
        rows[1]["word"] = ("unsure", ["stand"])
        for cascade in (False, True):
            th, m = bands.fit(rows, cascade)
            again = bands.evaluate(rows, [bands.decide(r, th, cascade) for r in rows])
            self.assertEqual((m["move_pass"], m["talk_silent"]), (again["move_pass"], again["talk_silent"]))


class FoldTests(unittest.TestCase):
    def test_folds_partition_and_stratify(self):
        rows = separable(10)
        folds = bands.stratified_folds(rows, 5, seed=1)
        flat = sorted(i for f in folds for i in f)
        self.assertEqual(flat, list(range(len(rows))))
        for f in folds:
            labels = [rows[i]["label"] for i in f]
            for label in ("hit", "stand", "talk"):
                self.assertEqual(labels.count(label), 2)

    def test_folds_are_deterministic_and_handle_uneven_labels(self):
        rows = [row("a", {}, rid=str(i)) for i in range(7)] + [row("b", {}, rid=str(i)) for i in range(3)]
        self.assertEqual(bands.stratified_folds(rows, 5, 3), bands.stratified_folds(rows, 5, 3))
        sizes = [len(f) for f in bands.stratified_folds(rows, 5, 3)]
        self.assertEqual(sum(sizes), 10)
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_cross_validate_scores_every_row_once(self):
        rows = separable(10)
        m, ths = bands.cross_validate(rows, False, grid_n=5)
        self.assertEqual(m["move_n"] + m["talk_n"], len(rows))
        self.assertEqual(len(ths), 5)


LEX = {"hit": {"en": ["hit", "hit me"], "ko": ["히트"]}, "stand": {"en": ["stand"], "ko": []}}


def extra(text, kinds=("hit", "stand")):
    return wordrule.extra_tokens(text, LEX, list(kinds))


class WordRuleTests(unittest.TestCase):
    def test_move_word_alone_or_with_fillers_has_no_extras(self):
        for text in ("hit", "Hit me!", "ok hit me please", "히트", "히트요", "히트할게요", "그냥 히트 ㅋㅋㅋ", "stand lol 😀", "yeah i'll stand"):
            self.assertEqual(extra(text), 0, text)

    def test_other_words_are_counted(self):
        self.assertEqual(extra("hit me up later"), 2)
        self.assertEqual(extra("that song is a hit"), 4)
        self.assertEqual(extra("히트곡 좋아"), 1)
        self.assertEqual(extra("sure hit"), 1)

    def test_attached_suffix_belongs_to_the_match(self):
        self.assertEqual(extra("standing"), 0)
        self.assertEqual(extra("히트요 더"), 1)

    def test_laugh_runs_and_punctuation_are_ignored(self):
        self.assertEqual(wordrule.other_tokens("ㅋㅋ ㅎㅎㅎ ... !!", []), [])
        self.assertEqual(wordrule.other_tokens("ㅋㅎ x", []), ["ㅋㅎ", "x"])

    def test_cascade_limit_gates_the_word_reading(self):
        r = row("hit", {"hit": 0.2, "stand": 0.1}, talk=0.0, word=("sure", ["stand"]))
        r["extra"] = 2
        self.assertEqual(bands.decide(r, TH, 1)[1], [])
        self.assertEqual(bands.decide(r, TH, 2), ("sure", ["stand"]))
        self.assertEqual(bands.decide(r, TH, True)[1], [])
        self.assertEqual(bands.decide(r, TH, False)[1], [])
        r["extra"] = 0
        self.assertEqual(bands.decide(r, TH, 0), ("sure", ["stand"]))


class HeadMaskTests(unittest.TestCase):
    def test_illegal_moves_dropped_and_renormalized(self):
        scores, talk = bands.mask_renorm({"hit": 0.5, "stand": 0.2, "double": 0.2, "talk": 0.1}, ["hit", "stand"])
        self.assertEqual(set(scores), {"hit", "stand"})
        self.assertAlmostEqual(sum(scores.values()) + talk, 1.0)
        self.assertAlmostEqual(scores["hit"], 0.5 / 0.8)
        self.assertAlmostEqual(talk, 0.1 / 0.8)

    def test_all_mass_on_illegal_moves_does_not_divide_by_zero(self):
        scores, talk = bands.mask_renorm({"hit": 1.0, "stand": 0.0, "talk": 0.0}, ["stand"])
        self.assertEqual((scores, talk), ({"stand": 0.0}, 0.0))

    def test_probability_features_feed_the_bands(self):
        scores, talk = bands.mask_renorm({"hit": 0.6, "stand": 0.1, "talk": 0.3}, ["hit", "stand"])
        _, _, p1, margin, gap = bands.features(scores, talk)
        self.assertAlmostEqual(margin, 0.5)
        self.assertAlmostEqual(gap, 0.3)
        self.assertAlmostEqual(p1, 0.6)


if __name__ == "__main__":
    unittest.main()
