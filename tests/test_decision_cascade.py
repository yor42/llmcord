"""FEAT-44: e5-first cascade replay (prototypes/decision_eval/cascade.py). Offline, pure functions, no keys.

These tests DEFINE the interface; the module is written to satisfy them. Research only.

  apply_policy(e5, reader, policy) -> {"band", "moves", "called"}
  reading_of(record) -> {"probs": {...}} | None
  evaluate(items, finals) -> dict; finals maps item id -> {"band","moves","called"}. Keys: move_pass, move_n, talk_silent
      (no buttons), talk_silent_or_accepted (no buttons or an accepted move among them), talk_n, sure_wrong, by_lang {lang: [pass, n]}
      (move rows), calls, calls_per_row.
  cost_per_1000_messages(calls, rows, per_call_usd) -> float
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototypes" / "decision_eval"))

import cascade as cs  # noqa: E402

UNSURE_ALL = {"cover": "all", "tau": 0.5}
UNSURE_ONLY = {"cover": "unsure", "tau": 0.5}


def e5(band, *moves):
    return {"band": band, "moves": list(moves)}


def item(iid, label, lang="en", accept=(), task="bj", legal=("hit", "stand", "double")):
    return {"id": iid, "task": task, "lang": lang, "label": label, "accept": list(accept), "legal": list(legal), "text": iid}


def final(band, *moves, called=False):
    return {"band": band, "moves": list(moves), "called": called}


class ApplyPolicyTests(unittest.TestCase):
    def test_silent_never_asks_reader(self):
        out = cs.apply_policy(e5("silent"), {"probs": {"hit": 1.0}}, UNSURE_ALL)
        self.assertEqual(out, {"band": "silent", "moves": [], "called": False})

    def test_uncovered_sure_band_unchanged(self):
        out = cs.apply_policy(e5("sure", "hit"), {"probs": {"hit": 0.0}}, UNSURE_ONLY)
        self.assertEqual(out, {"band": "sure", "moves": ["hit"], "called": False})

    def test_unsure_band_is_covered_by_unsure_policy(self):
        out = cs.apply_policy(e5("unsure", "hit", "stand"), {"probs": {"hit": 0.9, "stand": 0.1}}, UNSURE_ONLY)
        self.assertTrue(out["called"])
        self.assertEqual(out["moves"], ["hit"])

    def test_all_covers_sure_band(self):
        out = cs.apply_policy(e5("sure", "hit"), {"probs": {"hit": 0.9}}, UNSURE_ALL)
        self.assertEqual(out, {"band": "sure", "moves": ["hit"], "called": True})

    def test_none_left_goes_silent_but_called(self):
        out = cs.apply_policy(e5("unsure", "hit", "stand"), {"probs": {"hit": 0.2, "stand": 0.3}}, UNSURE_ONLY)
        self.assertEqual(out, {"band": "silent", "moves": [], "called": True})

    def test_tau_boundary_keeps_equal(self):
        out = cs.apply_policy(e5("unsure", "hit", "stand"), {"probs": {"hit": 0.5, "stand": 0.4999}}, UNSURE_ONLY)
        self.assertEqual(out["moves"], ["hit"])

    def test_reorders_by_reader_prob_desc(self):
        out = cs.apply_policy(e5("unsure", "hit", "stand"), {"probs": {"hit": 0.6, "stand": 0.9}}, UNSURE_ONLY)
        self.assertEqual(out["moves"], ["stand", "hit"])
        self.assertEqual(out["band"], "unsure")

    def test_ties_keep_e5_order(self):
        out = cs.apply_policy(e5("unsure", "stand", "hit", "double"),
                              {"probs": {"hit": 0.8, "stand": 0.8, "double": 0.8}}, UNSURE_ONLY)
        self.assertEqual(out["moves"], ["stand", "hit", "double"])

    def test_reader_moves_outside_e5_are_ignored(self):
        out = cs.apply_policy(e5("unsure", "hit"), {"probs": {"hit": 0.7, "double": 0.99}}, UNSURE_ONLY)
        self.assertEqual(out["moves"], ["hit"])

    def test_missing_move_counts_as_zero(self):
        out = cs.apply_policy(e5("unsure", "hit", "stand"), {"probs": {"stand": 0.7}}, UNSURE_ONLY)
        self.assertEqual(out["moves"], ["stand"])

    def test_empty_probs_goes_silent(self):
        out = cs.apply_policy(e5("unsure", "hit"), {"probs": {}}, UNSURE_ONLY)
        self.assertEqual(out, {"band": "silent", "moves": [], "called": True})

    def test_tau_zero_keeps_everything(self):
        out = cs.apply_policy(e5("unsure", "hit", "stand"), {"probs": {}}, {"cover": "unsure", "tau": 0.0})
        self.assertEqual(out["moves"], ["hit", "stand"])

    def test_sure_single_move_stays_sure(self):
        out = cs.apply_policy(e5("sure", "hit"), {"probs": {"hit": 0.5}}, UNSURE_ALL)
        self.assertEqual(out["band"], "sure")

    def test_sure_demoted_when_reader_reorders_or_adds_nothing_new(self):
        # e5 sure with two moves is unusual but must not stay sure when two survive
        out = cs.apply_policy(e5("sure", "hit", "stand"), {"probs": {"hit": 0.9, "stand": 0.9}}, UNSURE_ALL)
        self.assertEqual(out["band"], "unsure")

    def test_sure_demoted_when_a_different_single_move_remains(self):
        out = cs.apply_policy(e5("sure", "hit", "stand"), {"probs": {"hit": 0.1, "stand": 0.9}}, UNSURE_ALL)
        self.assertEqual(out["moves"], ["stand"])
        self.assertEqual(out["band"], "unsure")

    def test_unsure_single_survivor_stays_unsure(self):
        out = cs.apply_policy(e5("unsure", "hit", "stand"), {"probs": {"hit": 0.9}}, UNSURE_ONLY)
        self.assertEqual(out["band"], "unsure")

    def test_single_legal_move(self):
        out = cs.apply_policy(e5("unsure", "hit"), {"probs": {"hit": 0.51}}, UNSURE_ONLY)
        self.assertEqual(out, {"band": "unsure", "moves": ["hit"], "called": True})

    def test_reader_none_fails_open(self):
        out = cs.apply_policy(e5("unsure", "hit", "stand"), None, UNSURE_ONLY)
        self.assertEqual(out, {"band": "unsure", "moves": ["hit", "stand"], "called": True})

    def test_reader_none_uncovered_not_called(self):
        out = cs.apply_policy(e5("sure", "hit"), None, UNSURE_ONLY)
        self.assertEqual(out, {"band": "sure", "moves": ["hit"], "called": False})

    def test_reader_none_on_silent_not_called(self):
        out = cs.apply_policy(e5("silent"), None, UNSURE_ALL)
        self.assertEqual(out, {"band": "silent", "moves": [], "called": False})

    def test_does_not_mutate_inputs(self):
        row = e5("unsure", "hit", "stand")
        cs.apply_policy(row, {"probs": {"stand": 0.9}}, UNSURE_ONLY)
        self.assertEqual(row, e5("unsure", "hit", "stand"))


class ReadingOfTests(unittest.TestCase):
    def test_returns_probs(self):
        rec = {"item": "a", "candidate": "or-d1-noul", "k": 0, "answer": "hit", "p": 0.9,
               "probs": {"hit": 0.9, "stand": 0.1}, "error": None}
        self.assertEqual(cs.reading_of(rec), {"probs": {"hit": 0.9, "stand": 0.1}})

    def test_none_record(self):
        self.assertIsNone(cs.reading_of(None))

    def test_errored_record(self):
        self.assertIsNone(cs.reading_of({"item": "a", "probs": {"hit": 0.9}, "error": "HTTP 500"}))

    def test_empty_probs_is_a_reading(self):
        self.assertEqual(cs.reading_of({"item": "a", "probs": {}, "error": None}), {"probs": {}})

    def test_missing_error_key_is_ok(self):
        self.assertEqual(cs.reading_of({"item": "a", "probs": {"hit": 0.2}}), {"probs": {"hit": 0.2}})


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            item("m1", "hit"),
            item("m2", "stand", lang="ko", accept=["double"]),
            item("m3", "double", lang="ko"),
            item("t1", "talk"),
            item("t2", "talk", accept=["hit"]),
        ]

    def test_counts_and_by_lang(self):
        finals = {
            "m1": final("sure", "hit", called=True),
            "m2": final("unsure", "double", "hit"),      # accepted move among buttons
            "m3": final("silent"),                        # miss
            "t1": final("silent"),
            "t2": final("silent"),
        }
        r = cs.evaluate(self.items, finals)
        self.assertEqual((r["move_pass"], r["move_n"]), (2, 3))
        self.assertEqual((r["talk_silent"], r["talk_n"]), (2, 2))
        self.assertEqual(r["by_lang"], {"en": [1, 1], "ko": [1, 2]})
        self.assertEqual(r["calls"], 1)
        self.assertAlmostEqual(r["calls_per_row"], 1 / 5)
        self.assertEqual(r["sure_wrong"], 0)

    def test_move_row_label_among_several_buttons_passes(self):
        r = cs.evaluate([item("m1", "hit")], {"m1": final("unsure", "stand", "hit")})
        self.assertEqual(r["move_pass"], 1)

    def test_move_row_wrong_buttons_fail(self):
        r = cs.evaluate([item("m1", "hit")], {"m1": final("unsure", "stand")})
        self.assertEqual(r["move_pass"], 0)

    def test_talk_row_with_buttons_not_silent(self):
        r = cs.evaluate([item("t1", "talk")], {"t1": final("unsure", "hit")})
        self.assertEqual(r["talk_silent"], 0)
        self.assertEqual(r["talk_silent_or_accepted"], 0)

    def test_talk_row_accepted_move_counts_only_for_silent_or_accepted(self):
        r = cs.evaluate([item("t2", "talk", accept=["hit"])], {"t2": final("unsure", "hit")})
        self.assertEqual(r["talk_silent"], 0)
        self.assertEqual(r["talk_silent_or_accepted"], 1)
        self.assertEqual(r["talk_n"], 1)

    def test_sure_wrong_on_move_row(self):
        r = cs.evaluate([item("m1", "hit")], {"m1": final("sure", "stand")})
        self.assertEqual(r["sure_wrong"], 1)

    def test_sure_accepted_move_is_not_wrong(self):
        r = cs.evaluate([item("m2", "stand", accept=["double"])], {"m2": final("sure", "double")})
        self.assertEqual(r["sure_wrong"], 0)

    def test_sure_on_talk_row_is_wrong_unless_accepted(self):
        items = [item("t1", "talk"), item("t2", "talk", accept=["hit"])]
        r = cs.evaluate(items, {"t1": final("sure", "hit"), "t2": final("sure", "hit")})
        self.assertEqual(r["sure_wrong"], 1)

    def test_unsure_wrong_is_not_sure_wrong(self):
        r = cs.evaluate([item("m1", "hit")], {"m1": final("unsure", "stand")})
        self.assertEqual(r["sure_wrong"], 0)

    def test_empty_items(self):
        r = cs.evaluate([], {})
        self.assertEqual((r["move_n"], r["talk_n"], r["calls"]), (0, 0, 0))
        self.assertEqual(r["calls_per_row"], 0)

    def test_korean_split_in_by_lang(self):
        items = [item("k1", "hit", lang="ko"), item("k2", "hit", lang="ko"), item("e1", "hit", lang="en")]
        finals = {"k1": final("sure", "hit"), "k2": final("silent"), "e1": final("sure", "hit")}
        self.assertEqual(cs.evaluate(items, finals)["by_lang"], {"ko": [1, 2], "en": [1, 1]})

    def test_talk_rows_not_in_by_lang(self):
        r = cs.evaluate([item("t1", "talk", lang="ko")], {"t1": final("silent")})
        self.assertEqual(r["by_lang"], {})

    def test_calls_count_called_true(self):
        items = [item("a", "hit"), item("b", "hit"), item("c", "talk"), item("d", "talk")]
        finals = {k: final("silent", called=(k in "ac")) for k in "abcd"}
        r = cs.evaluate(items, finals)
        self.assertEqual(r["calls"], 2)
        self.assertAlmostEqual(r["calls_per_row"], 0.5)


class CostTests(unittest.TestCase):
    def test_cost_per_1000(self):
        self.assertAlmostEqual(cs.cost_per_1000_messages(50, 100, 0.002), 1.0)

    def test_zero_calls(self):
        self.assertEqual(cs.cost_per_1000_messages(0, 100, 0.002), 0.0)

    def test_zero_rows_does_not_divide(self):
        self.assertEqual(cs.cost_per_1000_messages(0, 0, 0.002), 0.0)


# --- FEAT-44 replay layer (reader_inputs, replay, render, inside_repo, cmd_replay) ------------------------------------------
import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402


def rec(item_id, reader, *, k=0, task="bj", answer=None, probs=None, cost=0.001, error=None, estimated=False):
    return {"item": item_id, "candidate": reader, "task": task, "k": k, "answer": answer, "probs": probs or {},
            "cost": cost, "error": error, "cost_estimated": estimated}


def all_reader_records(item_ids, estimated_reader=None):
    out = []
    for reader in cs.READERS:
        for i in item_ids:
            out.append(rec(i, reader, answer="hit", cost=0.002, estimated=(reader == estimated_reader)))
    return out


class ReaderInputsTests(unittest.TestCase):
    def setUp(self):
        self.items = [item("a", "hit"), item("b", "hit", legal=("hit", "stand"))]
        self.reader = cs.READERS[0]

    def test_chat_answer_is_probability_one_on_that_move_zero_on_other_legal(self):
        readings, _, _ = cs.reader_inputs([rec("a", self.reader, answer="stand")], self.reader, self.items)
        self.assertEqual(readings["a"], {"probs": {"hit": 0.0, "stand": 1.0, "double": 0.0}})

    def test_invalid_answer_none_is_no_reading(self):
        readings, _, _ = cs.reader_inputs([rec("a", self.reader, answer=None)], self.reader, self.items)
        self.assertIsNone(readings["a"])

    def test_errored_record_is_no_reading(self):
        readings, _, _ = cs.reader_inputs([rec("a", self.reader, answer="hit", error="HTTP 500")], self.reader, self.items)
        self.assertIsNone(readings["a"])

    def test_probs_record_is_used_as_is(self):
        readings, _, _ = cs.reader_inputs([rec("a", self.reader, probs={"hit": 0.7})], self.reader, self.items)
        self.assertEqual(readings["a"], {"probs": {"hit": 0.7}})

    def test_only_k0_bj_records_of_that_reader(self):
        records = [rec("a", self.reader, k=1, answer="hit"), rec("b", self.reader, task="talk", answer="hit"),
                   rec("a", cs.READERS[1], answer="hit")]
        readings, _, _ = cs.reader_inputs(records, self.reader, self.items)
        self.assertEqual(readings, {})

    def test_mean_cost_and_measured_flag(self):
        records = [rec("a", self.reader, answer="hit", cost=0.001), rec("b", self.reader, answer="hit", cost=0.003)]
        _, per_call, price = cs.reader_inputs(records, self.reader, self.items)
        self.assertAlmostEqual(per_call, 0.002)
        self.assertEqual(price, "measured")

    def test_errored_record_cost_excluded_from_mean(self):
        records = [rec("a", self.reader, answer="hit", cost=0.001), rec("b", self.reader, error="x", cost=9.0)]
        _, per_call, _ = cs.reader_inputs(records, self.reader, self.items)
        self.assertAlmostEqual(per_call, 0.001)

    def test_estimated_flag_when_any_record_estimated(self):
        records = [rec("a", self.reader, answer="hit", cost=0.001), rec("b", self.reader, answer="hit", cost=0.001, estimated=True)]
        _, _, price = cs.reader_inputs(records, self.reader, self.items)
        self.assertEqual(price, "estimated")

    def test_no_costs_is_estimated(self):
        _, _, price = cs.reader_inputs([], self.reader, self.items)
        self.assertEqual(price, "estimated")

    def test_duplicate_record_keeps_last(self):
        records = [rec("a", self.reader, answer="hit"), rec("a", self.reader, answer="stand")]
        with contextlib.redirect_stderr(io.StringIO()):
            readings, _, _ = cs.reader_inputs(records, self.reader, self.items)
        self.assertEqual(readings["a"]["probs"]["stand"], 1.0)
        self.assertEqual(readings["a"]["probs"]["hit"], 0.0)


class ReplayAndRenderTests(unittest.TestCase):
    def setUp(self):
        self.items = [item("m1", "hit"), item("t1", "talk"), item("t2", "talk")]
        self.e5 = {"m1": e5("sure", "hit"), "t1": e5("sure", "hit"), "t2": e5("silent")}
        self.taus = [0.5, 0.9]

    def table(self, estimated_reader=None):
        records = all_reader_records(["m1", "t1", "t2"], estimated_reader)
        return cs.replay(self.items, self.e5, records, self.taus)

    def test_e5_alone_row_first_then_reader_cover_tau_order(self):
        t = self.table()
        self.assertIsNone(t[0]["reader"])
        self.assertIsNone(t[0]["tau"])
        self.assertEqual(t[0]["per_call"], 0.0)
        got = [(r["reader"], r["cover"], r["tau"]) for r in t[1:]]
        want = [(rd, cv, tau) for rd in cs.READERS for cv in cs.COVERS for tau in self.taus]
        self.assertEqual(got[:len(want)], want[:len(got)])
        self.assertEqual(got[0], want[0])

    def test_finals_keyed_by_item_id(self):
        for row in self.table():
            self.assertEqual(set(row["finals"]), {"m1", "t1", "t2"})

    def test_e5_alone_finals_make_no_calls(self):
        self.assertEqual(self.table()[0]["result"]["calls"], 0)

    def test_all_cover_asks_reader_on_non_silent_rows_only(self):
        row = next(r for r in self.table() if r["reader"] == cs.READERS[0] and r["cover"] == cs.COVERS[-1] and r["tau"] == 0.5)
        self.assertEqual(row["result"]["calls"], 2)  # m1 and t1; t2 is silent

    def test_render_est_marker_only_for_estimated_prices(self):
        lines = cs.render(self.items, self.table(estimated_reader=cs.READERS[1])).splitlines()
        for reader in cs.READERS:
            rows = [ln for ln in lines if ln.startswith(f"| {reader} ")]
            self.assertTrue(rows)
            for ln in rows:
                self.assertEqual("(est.)" in ln, reader == cs.READERS[1], ln)

    def test_render_e5_alone_cost_is_zero(self):
        lines = cs.render(self.items, self.table()).splitlines()
        row = next(ln for ln in lines if ln.startswith("| e5 alone "))
        self.assertTrue(row.rstrip().endswith("| 0 |"), row)
        self.assertNotIn("(est.)", row)

    def test_render_chatter_section_lists_leaked_talk_row(self):
        text = cs.render(self.items, self.table())
        self.assertIn("Chatter e5 let through (1)", text)
        self.assertIn('- t1 "t1" (e5 sure hit):', text)
        self.assertNotIn('- t2 "t2"', text)
        self.assertNotIn('- m1 "m1"', text)


class InsideRepoTests(unittest.TestCase):
    def test_repo_paths_are_inside(self):
        self.assertTrue(cs.inside_repo(ROOT))
        self.assertTrue(cs.inside_repo(ROOT / "tests" / "nope" / "x.json"))

    def test_temp_dir_is_outside(self):
        with tempfile.TemporaryDirectory() as d:
            if cs.inside_repo(d):
                self.skipTest("temp dir is inside the repo")
            self.assertFalse(cs.inside_repo(Path(d) / "x.json"))

    def test_sibling_with_same_prefix_is_outside(self):
        self.assertFalse(cs.inside_repo(str(ROOT) + "-other"))


class CmdReplayTests(unittest.TestCase):
    def run_cmd(self, **kw):
        a = argparse.Namespace(**{"taus": "0.5", "out": None, **kw})
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cs.cmd_replay(a)
        return code, out.getvalue(), err.getvalue()

    def files(self, d, rows):
        e5_path, res_path = Path(d) / "e5.json", Path(d) / "results.json"
        e5_path.write_text(json.dumps(rows), encoding="utf-8")
        res_path.write_text(json.dumps({"records": []}), encoding="utf-8")
        return str(e5_path), str(res_path)

    def full_rows(self):
        return {it["id"]: {"band": "silent", "moves": []} for it in cs.ec.load_items() if it["task"] == "bj"}

    def test_bad_taus_exit_2(self):
        for taus in ("abc", ",", ""):
            code, _, err = self.run_cmd(taus=taus, e5="unused", results="unused")
            self.assertEqual(code, 2, taus)
            self.assertIn("taus", err)

    def test_e5_file_missing_rows_exit_2(self):
        with tempfile.TemporaryDirectory() as d:
            e5_path, res_path = self.files(d, {})
            code, out, err = self.run_cmd(e5=e5_path, results=res_path)
        self.assertEqual(code, 2)
        self.assertIn("lacks", err)
        self.assertEqual(out, "")

    def test_out_inside_repo_refused_and_nothing_written(self):
        target = ROOT / "tests" / "_cascade_out_must_not_exist"
        with tempfile.TemporaryDirectory() as d:
            e5_path, res_path = self.files(d, self.full_rows())
            code, _, err = self.run_cmd(e5=e5_path, results=res_path, out=str(target))
        self.assertEqual(code, 2)
        self.assertIn("refusing", err)
        self.assertFalse(target.exists())

    def test_out_outside_repo_writes_cascade_md(self):
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as o:
            if cs.inside_repo(o):
                self.skipTest("temp dir is inside the repo")
            e5_path, res_path = self.files(d, self.full_rows())
            code, out, _ = self.run_cmd(e5=e5_path, results=res_path, out=str(Path(o) / "sub"))
            self.assertEqual(code, 0)
            written = (Path(o) / "sub" / "cascade.md").read_text(encoding="utf-8")
        self.assertIn("e5 alone", written)
        self.assertIn("e5 alone", out)


if __name__ == "__main__":
    unittest.main()
