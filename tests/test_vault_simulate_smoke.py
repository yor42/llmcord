"""Smoke test for scripts/simulate_vault.py (FEAT-62, decision D43), run as a subprocess with a tiny sample.

The simulator is not part of the product; this only pins that the CLI runs, is deterministic for a seed, writes a
Markdown report, keeps progress on stderr and rejects bad arguments. The report layout is not pinned beyond
headings and the names of the configurations. Lines that carry wall-clock timings are ignored when comparing runs.
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "simulate_vault.py"


def run(*args, timeout=60):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def stable_lines(text):
    """Report lines without wall-clock timings."""
    return [ln for ln in text.splitlines() if not any(w in ln.lower() for w in ("wall", "time", "second", " ms"))]


class SimulateVaultSmokeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def args(self, name, *extra):
        return [
            "--length", "4", "--protocol", "standard", "--turns", "9", "--strategy", "optimal",
            "--samples", "3", "--seed", "7", "--output", str(self.dir / name), *extra,
        ]

    def test_script_exists(self):
        self.assertTrue(SCRIPT.is_file())

    def test_writes_a_markdown_report(self):
        proc = run(*self.args("a.md"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = (self.dir / "a.md").read_text(encoding="utf-8")
        self.assertTrue(report.lstrip().startswith("#"))
        self.assertIn("standard", report.lower())
        self.assertIn("optimal", report.lower())
        self.assertIn("completion", report.lower())

    def test_progress_goes_to_stderr_not_stdout(self):
        proc = run(*self.args("b.md"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("optimal", proc.stdout.lower())

    def test_same_seed_gives_the_same_report(self):
        one, two = run(*self.args("c1.md")), run(*self.args("c2.md"))
        self.assertEqual((one.returncode, two.returncode), (0, 0), one.stderr + two.stderr)
        first = stable_lines((self.dir / "c1.md").read_text(encoding="utf-8"))
        second = stable_lines((self.dir / "c2.md").read_text(encoding="utf-8"))
        self.assertEqual(first, second)

    def test_casual_strategy_and_other_protocol_and_tools_run(self):
        proc = run(
            "--length", "4", "--protocol", "total_match", "--turns", "9", "--strategy", "casual",
            "--tools", "1", "--clues", "2", "--samples", "3", "--seed", "3",
            "--output", str(self.dir / "d.md"),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("total_match", (self.dir / "d.md").read_text(encoding="utf-8").lower())

    def test_five_digit_run_uses_sampling(self):
        proc = run(
            "--length", "5", "--protocol", "standard", "--turns", "9", "--strategy", "optimal",
            "--samples", "2", "--seed", "1", "--output", str(self.dir / "e.md"),
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue((self.dir / "e.md").is_file())

    def test_unknown_strategy_is_rejected(self):
        proc = run("--strategy", "telepathic", "--samples", "1", "--output", str(self.dir / "f.md"))
        self.assertEqual(proc.returncode, 2, proc.stderr)  # argparse usage error
        self.assertIn("usage", proc.stderr.lower())
        self.assertFalse((self.dir / "f.md").exists())

    def test_unknown_protocol_is_rejected(self):
        proc = run("--protocol", "nonsense", "--samples", "1", "--output", str(self.dir / "g.md"))
        self.assertEqual(proc.returncode, 2, proc.stderr)  # argparse usage error
        self.assertIn("usage", proc.stderr.lower())
        self.assertFalse((self.dir / "g.md").exists())

    def test_exhaustive_sampling_is_only_for_four_digits(self):
        proc = run("--length", "5", "--samples", "all", "--output", str(self.dir / "h.md"))
        self.assertEqual(proc.returncode, 2, proc.stderr)  # argparse usage error
        self.assertIn("usage", proc.stderr.lower())
        self.assertFalse((self.dir / "h.md").exists())


if __name__ == "__main__":
    unittest.main()
