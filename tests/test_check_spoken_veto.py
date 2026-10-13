"""FEAT-49c: scripts/check_spoken_veto.py, the operator-run drift check of the decision-model veto. Written ahead of the implementation.

Interface these tests require: ``main(argv, env=None, transport=None) -> int`` (``env`` defaults to ``os.environ``; ``transport`` is handed
to the production ``veto_client.from_env(env, transport=transport)``); it prints to ``sys.stdout`` / ``sys.stderr``.

* ``--e5 PATH`` (required; the JSON of ``cascade.py dump-e5``: ``{"bj<id>": {"band", "moves", ...}}``), ``--phrases PATH`` (CSV with the
  phrases.csv header; default prototypes/spoken_moves/phrases.csv; row ids become ``bj<id>``), ``--tau`` (default 0.5), ``--yes``.
* Exit codes: 0 fine (also a dry run), 1 drift flagged, 2 refused or bad input. Every refusal makes no network call.
* Refused (2, stderr contains "refusing"): a --phrases path (resolved, so ``..`` counts) under ``<repo>/data`` or under the directory of
  ``LLMCORD_DATABASE_PATH`` from the env. Bad input (2): a missing file, or an e5 file lacking rows of the phrases.
* Without --yes: prints ``estimated calls: N`` (non-silent e5 rows, one request each) and ``estimated cost: $X`` (X per 1,000 messages,
  cascade.cost_per_1000_messages), makes zero requests, exits 0.
* With --yes and ``LLMCORD_SPOKEN_VETO=openrouter-d1`` plus ``OPENROUTER_API_KEY`` (else exit 2, stderr names the missing variable): one
  request per non-silent e5 row through the real client, the cascade policy (cover sure+unsure, reader or-d1-noul, the given tau), then
  the cascade render table and the lines ``right move: P% (a/b)``, ``silent on chatter: P% (a/b)``, ``sure wrong: N``, ``lost moves: N``,
  ``errors: N`` and the baseline line BASELINE below. DRIFT when silent on chatter < 95% or lost moves > 8: a line starting ``DRIFT``
  and exit 1; otherwise a line starting ``OK``. The key is never printed.
"""
import contextlib
import csv
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "FAKE-veto-key-not-a-real-key"
ENV = {"LLMCORD_SPOKEN_VETO": "openrouter-d1", "OPENROUTER_API_KEY": FAKE_KEY}
BASELINE = "baseline (Decision-1, tau 0.5, 195 rows): right move 90%, silent on chatter 100%, lost moves 4"
HEADER = ["id", "text", "lang", "label", "accept", "legal", "note", "ok", "fix"]


def tool():
    from scripts import check_spoken_veto
    return check_spoken_veto


class Rig(unittest.TestCase):
    """A temp dir with a small synthetic phrases.csv (10 move rows, 20 chatter rows) and a matching e5 file."""

    MOVES, TALKS = 10, 20

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.tmp = Path(self.dir.name)
        self.requests = []
        self.move_texts = [f"move phrase {n}" for n in range(self.MOVES)]
        self.talk_texts = [f"chatter phrase {n}" for n in range(self.TALKS)]
        self.rows = [("hit", t) for t in self.move_texts] + [("talk", t) for t in self.talk_texts]
        self.phrases = self.tmp / "phrases.csv"
        with open(self.phrases, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            for n, (label, text) in enumerate(self.rows, 1):
                w.writerow([n, text, "en", label, "", "hit stand double", "", "y", ""])
        self.e5 = self.write_e5(lambda label: {"band": "unsure", "moves": ["hit", "stand"] if label == "hit" else ["hit"]})
        self.probs = lambda text, legal: {m: (0.9 if m == "hit" else 0.1) for m in legal} if text in self.move_texts else {m: 0.1 for m in legal}
        self.status = 200

    def write_e5(self, fn, name="e5.json"):
        path = self.tmp / name
        path.write_text(json.dumps({f"bj{n}": fn(label) for n, (label, _) in enumerate(self.rows, 1)}), encoding="utf-8")
        return path

    def handler(self, request):
        self.requests.append(request)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "nope"})
        body = json.loads(request.content)
        legal = [k[len("move_"):] for k in body["questions"]]
        probs = self.probs(body["state"]["message"], legal)
        return httpx.Response(200, json={"answers": {f"move_{m}": {"type": "noul", "noul": p} for m, p in probs.items()}})

    def run_tool(self, *args, env=None, e5=None, phrases=True, yes=True):
        argv = ["--e5", str(e5 or self.e5)]
        if phrases:
            argv += ["--phrases", str(self.phrases)]
        if yes:
            argv.append("--yes")
        argv += list(args)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = tool().main(argv, env=dict(ENV if env is None else env), transport=httpx.MockTransport(self.handler))
        return code, out.getvalue(), err.getvalue()


class DryRunTests(Rig):
    def test_dry_run_makes_no_request_and_prints_the_estimate(self):
        """FEAT-49c: without --yes only calls and cost are estimated: one call per non-silent e5 row."""
        code, out, err = self.run_tool(yes=False)
        self.assertEqual((code, self.requests), (0, []))
        self.assertIn(f"estimated calls: {self.MOVES + self.TALKS}", out)
        self.assertRegex(out, r"estimated cost: \$[0-9]*\.?[0-9]+")
        self.assertNotIn(FAKE_KEY, out + err)

    def test_silent_e5_rows_are_not_counted(self):
        """FEAT-49c: rows e5 calls silent never reach the reader."""
        e5 = self.write_e5(lambda label: {"band": "silent", "moves": []} if label == "talk" else {"band": "sure", "moves": ["hit"]}, "e5b.json")
        code, out, _ = self.run_tool(yes=False, e5=e5)
        self.assertEqual(code, 0)
        self.assertIn(f"estimated calls: {self.MOVES}", out)
        self.assertEqual(self.requests, [])

    def test_default_phrases_file_is_the_prototype_csv(self):
        """FEAT-49c: with no --phrases the 195 synthetic rows of prototypes/spoken_moves/phrases.csv are used (an e5 file for them works)."""
        with open(ROOT / "prototypes" / "spoken_moves" / "phrases.csv", encoding="utf-8", newline="") as f:
            ids = [r["id"] for r in csv.DictReader(f)]
        path = self.tmp / "e5-full.json"
        path.write_text(json.dumps({f"bj{i}": {"band": "unsure", "moves": ["hit"]} if n < 3 else {"band": "silent", "moves": []}
                                    for n, i in enumerate(ids)}), encoding="utf-8")
        code, out, _ = self.run_tool(yes=False, e5=path, phrases=False)
        self.assertEqual((code, self.requests), (0, []))
        self.assertIn("estimated calls: 3", out)


class RunTests(Rig):
    def test_healthy_run_prints_metrics_and_the_baseline(self):
        """FEAT-49c: --yes asks once per non-silent row and prints the cascade metrics; no drift, exit 0."""
        code, out, err = self.run_tool()
        self.assertEqual(code, 0, out + err)
        self.assertEqual(len(self.requests), self.MOVES + self.TALKS)
        for line in (f"right move: 100% ({self.MOVES}/{self.MOVES})", f"silent on chatter: 100% ({self.TALKS}/{self.TALKS})", "sure wrong: 0",
                     "lost moves: 0", "errors: 0", BASELINE):
            self.assertIn(line, out)
        self.assertRegex(out, r"(?m)^OK")
        self.assertNotRegex(out, r"(?m)^DRIFT")
        self.assertIn("| or-d1-noul | sure+unsure | 0.5 |", out)  # the cascade render table
        self.assertEqual({str(r.url) for r in self.requests}, {"https://openrouter.ai/api/alpha/decisions"})
        self.assertNotIn(FAKE_KEY, out + err)

    def test_chatter_that_is_no_longer_silenced_is_drift(self):
        """FEAT-49c: silent on chatter below 95% flags drift and exits 1."""
        self.probs = lambda text, legal: {m: 0.9 for m in legal}
        code, out, _ = self.run_tool()
        self.assertEqual(code, 1)
        self.assertIn("silent on chatter: 0% (0/20)", out)
        self.assertRegex(out, r"(?m)^DRIFT")
        self.assertNotRegex(out, r"(?m)^OK")

    def test_lost_moves_are_drift(self):
        """FEAT-49c: a reader that drops the moves e5 had right loses them: 10 lost moves is drift."""
        self.probs = lambda text, legal: {m: 0.1 for m in legal}
        code, out, _ = self.run_tool()
        self.assertEqual(code, 1)
        self.assertIn(f"lost moves: {self.MOVES}", out)
        self.assertIn("right move: 0%", out)
        self.assertRegex(out, r"(?m)^DRIFT")

    def test_drift_thresholds_are_exclusive(self):
        """FEAT-49c: exactly 8 lost moves and exactly 95% silent on chatter (19/20) are still fine; 9 lost and 18/20 are drift."""
        for lost, leaked, drift in ((8, 1, False), (9, 0, True), (0, 2, True)):
            with self.subTest(lost=lost, leaked=leaked):
                self.requests.clear()
                self.probs = lambda text, legal, lost=lost, leaked=leaked: (
                    {m: 0.1 for m in legal} if text in self.move_texts[:lost] else
                    {m: 0.9 for m in legal} if text in self.talk_texts[:leaked] or text in self.move_texts else {m: 0.1 for m in legal})
                code, out, _ = self.run_tool()
                self.assertIn(f"lost moves: {lost}", out)
                self.assertEqual(code, 1 if drift else 0, out)
                self.assertRegex(out, r"(?m)^DRIFT" if drift else r"(?m)^OK")

    def test_tau_option_reaches_the_policy(self):
        """FEAT-49c: --tau is the cut-off: at 0.95 a 0.9 reader drops the moves."""
        code, out, _ = self.run_tool("--tau", "0.95")
        self.assertEqual(code, 1)
        self.assertIn(f"lost moves: {self.MOVES}", out)
        self.assertIn("| or-d1-noul | sure+unsure | 0.95 |", out)

    def test_failing_requests_are_counted_and_fail_open(self):
        """FEAT-49c: a provider failure is counted as an error (e5 stands for that row) and the key stays out of the output."""
        self.status = 500
        code, out, err = self.run_tool()
        self.assertIn(f"errors: {self.MOVES + self.TALKS}", out)
        self.assertIn("lost moves: 0", out)
        self.assertNotIn(FAKE_KEY, out + err)
        self.assertIsInstance(code, int)

    def test_rows_with_silent_e5_are_not_asked(self):
        """FEAT-49c: only non-silent rows cost a request."""
        e5 = self.write_e5(lambda label: {"band": "silent", "moves": []} if label == "talk" else {"band": "unsure", "moves": ["hit", "stand"]}, "e5c.json")
        self.run_tool(e5=e5)
        self.assertEqual(len(self.requests), self.MOVES)


class RefusalTests(Rig):
    def refused(self, code, err, needle):
        self.assertEqual(code, 2)
        self.assertIn(needle, err)
        self.assertEqual(self.requests, [])

    def test_wrong_switch_is_refused(self):
        """FEAT-49c: the operator switch must be openrouter-d1; the key is not echoed."""
        for value in (None, "off", "openrouter"):
            env = {k: v for k, v in ENV.items() if k != "LLMCORD_SPOKEN_VETO"}
            if value:
                env["LLMCORD_SPOKEN_VETO"] = value
            code, out, err = self.run_tool(env=env)
            self.refused(code, err, "LLMCORD_SPOKEN_VETO")
            self.assertNotIn(FAKE_KEY, out + err)

    def test_missing_key_is_refused(self):
        """FEAT-49c: no key, no run."""
        code, out, err = self.run_tool(env={"LLMCORD_SPOKEN_VETO": "openrouter-d1"})
        self.refused(code, err, "OPENROUTER_API_KEY")
        self.assertEqual(out.count("right move"), 0)

    def test_phrases_under_the_repo_data_dir_are_refused(self):
        """FEAT-49c: real user data is never read, even by a path with .. in it."""
        for path in (ROOT / "data" / "phrases.csv", ROOT / "tests" / ".." / "data" / "kept.csv"):
            with self.subTest(path=str(path)):
                argv = ["--e5", str(self.e5), "--phrases", str(path), "--yes"]
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = tool().main(argv, env=dict(ENV), transport=httpx.MockTransport(self.handler))
                self.refused(code, err.getvalue(), "refusing")

    def test_phrases_under_the_configured_data_dir_are_refused(self):
        """FEAT-49c: the directory of LLMCORD_DATABASE_PATH counts as the data dir."""
        db_dir = self.tmp / "livedata"
        db_dir.mkdir()
        copied = (db_dir / "phrases.csv")
        copied.write_text(self.phrases.read_text(encoding="utf-8"), encoding="utf-8")
        argv = ["--e5", str(self.e5), "--phrases", str(copied), "--yes"]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = tool().main(argv, env={**ENV, "LLMCORD_DATABASE_PATH": str(db_dir / "llmcord.db")}, transport=httpx.MockTransport(self.handler))
        self.refused(code, err.getvalue(), "refusing")

    def test_missing_files_are_refused(self):
        """FEAT-49c: a missing e5 or phrases file is a clean exit 2 with no request."""
        code, _, err = self.run_tool(e5=self.tmp / "nope.json")
        self.assertEqual((code, self.requests), (2, []))
        self.assertTrue(err)
        phrases = self.phrases
        self.phrases = self.tmp / "nope.csv"
        code, _, err = self.run_tool()
        self.phrases = phrases
        self.assertEqual((code, self.requests), (2, []))
        self.assertTrue(err)

    def test_e5_file_lacking_rows_is_refused(self):
        """FEAT-49c: every phrase needs its e5 reading, as in cascade replay."""
        path = self.tmp / "short.json"
        path.write_text(json.dumps({"bj1": {"band": "silent", "moves": []}}), encoding="utf-8")
        code, _, err = self.run_tool(e5=path)
        self.refused(code, err, "lacks")

    def test_dry_run_also_refuses_bad_paths(self):
        """FEAT-49c: refusals do not depend on --yes."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = tool().main(["--e5", str(self.e5), "--phrases", str(ROOT / "data" / "x.csv")], env=dict(ENV), transport=httpx.MockTransport(self.handler))
        self.refused(code, err.getvalue(), "refusing")


class KeyHygieneTests(Rig):
    def test_key_never_appears_in_output_on_any_path(self):
        """FEAT-49c: not on success, drift, provider errors, a dry run or a refusal."""
        outputs = []
        outputs.append(self.run_tool())
        self.probs = lambda text, legal: {m: 0.1 for m in legal}
        outputs.append(self.run_tool())
        self.status = 401
        outputs.append(self.run_tool())
        outputs.append(self.run_tool(yes=False))
        outputs.append(self.run_tool(env={**ENV, "LLMCORD_SPOKEN_VETO": "off"}))
        for code, out, err in outputs:
            self.assertNotIn(FAKE_KEY, out + err)
            self.assertNotIn("Bearer", out + err)
        self.assertTrue(re.search(r"estimated calls", outputs[3][1]))


if __name__ == "__main__":
    unittest.main()
