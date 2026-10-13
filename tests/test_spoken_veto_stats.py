"""FEAT-49c: shadow-mode telemetry and the live drift signal of the decision-model veto. Written ahead of the implementation.

Seams these tests require (names are the contract):

* ``llmcord_core/spoken/veto_stats.py``: pure, standard library only. ``VetoStats(clock=..., today=...)`` where ``today()`` returns the
  current day (any hashable that formats as text, tests use "d1"/"d2"). ``COUNTERS`` is the tuple of counter names ``offered, vetoed, silenced,
  errors, pressed_ok, pressed_vetoed, unpressed_vetoed``; ``MIN_CALLS = 20``. Methods: ``record(guild, name, n=1)`` (counts under ``today()``;
  unknown name -> ValueError), ``count(guild, day, name)``, ``rate(guild, day)`` (vetoed / offered, None when nothing was offered),
  ``suspicious(guild, day)`` (offered >= MIN_CALLS and the veto rate is exactly 0% or exactly 100%), ``snapshot()`` ({(guild, day): {name: n}},
  a copy), ``lines()`` (one count-only string per (guild, day): ``"... guild=1 day=d1 offered=3 vetoed=1 ..."`` carrying every counter as
  ``name=n``), ``rollover()`` (the lines of every finished day, once, and those days are dropped).
* ``SpokenMoves.stats`` (a VetoStats, tests replace it), ``SpokenMoves.shadow`` (bool, tests assign it or build it from the env var
  ``LLMCORD_SPOKEN_VETO_SHADOW``), see test_spoken_veto_stats_wiring.py for the Discord side.
"""
import unittest


def stats_module():
    from llmcord_core.spoken import veto_stats
    return veto_stats


class StatsBase(unittest.TestCase):
    def setUp(self):
        self.vs = stats_module()
        self.day = ["d1"]
        self.stats = self.vs.VetoStats(clock=lambda: 0.0, today=lambda: self.day[0])

    def add(self, guild, name, n=1):
        for _ in range(n):
            self.stats.record(guild, name)


class CounterTests(StatsBase):
    def test_counter_names(self):
        """FEAT-49c: the seven counters of the design."""
        self.assertEqual(tuple(self.vs.COUNTERS), ("offered", "vetoed", "silenced", "errors", "pressed_ok", "pressed_vetoed", "unpressed_vetoed"))
        self.assertEqual(self.vs.MIN_CALLS, 20)

    def test_records_count_per_guild_and_day(self):
        """FEAT-49c: counters are keyed by (guild, day) and start at zero."""
        self.add(1, "offered", 3)
        self.add(2, "offered")
        self.day[0] = "d2"
        self.add(1, "offered", 2)
        self.assertEqual([self.stats.count(1, "d1", "offered"), self.stats.count(2, "d1", "offered"), self.stats.count(1, "d2", "offered")], [3, 1, 2])
        self.assertEqual([self.stats.count(1, "d1", n) for n in self.vs.COUNTERS[1:]], [0] * 6)
        self.assertEqual(self.stats.count(9, "d1", "offered"), 0)

    def test_record_with_a_count(self):
        """FEAT-49c: record(guild, name, n) adds n."""
        self.stats.record(1, "errors", 4)
        self.assertEqual(self.stats.count(1, "d1", "errors"), 4)

    def test_unknown_counter_is_refused(self):
        """FEAT-49c: a typo in a counter name is an error, not a silent new counter."""
        with self.assertRaises(ValueError):
            self.stats.record(1, "text")

    def test_rate(self):
        """FEAT-49c: rate is vetoed / offered; None when nothing was offered."""
        self.assertIsNone(self.stats.rate(1, "d1"))
        self.add(1, "offered", 4)
        self.add(1, "vetoed")
        self.assertEqual(self.stats.rate(1, "d1"), 0.25)

    def test_snapshot_is_a_copy_with_every_counter(self):
        """FEAT-49c: snapshot() maps (guild, day) to all seven counters and cannot be used to change the stats."""
        self.add(1, "offered", 2)
        shot = self.stats.snapshot()
        self.assertEqual(set(shot), {(1, "d1")})
        self.assertEqual(shot[(1, "d1")], {**{n: 0 for n in self.vs.COUNTERS}, "offered": 2})
        shot[(1, "d1")]["offered"] = 99
        self.assertEqual(self.stats.count(1, "d1", "offered"), 2)


class SuspiciousTests(StatsBase):
    def test_needs_twenty_calls(self):
        """FEAT-49c: 19 offered readings with a 0% (or 100%) veto rate are not yet a signal; 20 are."""
        self.add(1, "offered", 19)
        self.assertFalse(self.stats.suspicious(1, "d1"))
        self.add(1, "offered")
        self.assertTrue(self.stats.suspicious(1, "d1"))

    def test_hundred_percent_is_suspicious(self):
        """FEAT-49c: everything vetoed looks like moved weights."""
        self.add(1, "offered", 20)
        self.add(1, "vetoed", 20)
        self.assertTrue(self.stats.suspicious(1, "d1"))

    def test_ordinary_rates_are_not_suspicious(self):
        """FEAT-49c: a mixed day is fine, and so is a day with nothing offered."""
        self.add(1, "offered", 20)
        self.add(1, "vetoed", 10)
        self.assertFalse(self.stats.suspicious(1, "d1"))
        self.assertFalse(self.stats.suspicious(2, "d1"))

    def test_is_per_guild_and_day(self):
        """FEAT-49c: another server's or day's counts do not add up."""
        self.add(1, "offered", 12)
        self.add(2, "offered", 12)
        self.day[0] = "d2"
        self.add(1, "offered", 12)
        for guild, day in ((1, "d1"), (2, "d1"), (1, "d2")):
            self.assertFalse(self.stats.suspicious(guild, day))


class LinesAndRolloverTests(StatsBase):
    def test_lines_are_count_only(self):
        """FEAT-49c: one line per guild and day carrying every counter as name=n."""
        self.add(1, "offered", 3)
        self.add(1, "vetoed")
        self.add(2, "errors")
        lines = self.stats.lines()
        self.assertEqual(len(lines), 2)
        mine = next(ln for ln in lines if "guild=1" in ln)
        for token in ("day=d1", "offered=3", "vetoed=1", "silenced=0", "errors=0", "pressed_ok=0", "pressed_vetoed=0", "unpressed_vetoed=0"):
            self.assertIn(token, mine.split())
        self.assertIn("errors=1", next(ln for ln in lines if "guild=2" in ln).split())

    def test_no_activity_no_lines(self):
        """FEAT-49c: nothing recorded, nothing to log."""
        self.assertEqual((self.stats.lines(), self.stats.rollover(), self.stats.snapshot()), ([], [], {}))

    def test_rollover_returns_the_finished_day_once(self):
        """FEAT-49c: after the day changes, rollover() returns the old day's summary exactly once and drops it."""
        self.add(1, "offered", 2)
        self.assertEqual(self.stats.rollover(), [])  # the day is not over
        self.day[0] = "d2"
        self.add(1, "offered")
        out = self.stats.rollover()
        self.assertEqual(len(out), 1)
        self.assertIn("day=d1", out[0].split())
        self.assertIn("offered=2", out[0].split())
        self.assertEqual(self.stats.rollover(), [])
        self.assertEqual(set(self.stats.snapshot()), {(1, "d2")})
        self.assertEqual([ln for ln in self.stats.lines() if "day=d1" in ln.split()], [])

    def test_rollover_covers_every_guild_of_the_finished_day(self):
        """FEAT-49c: each guild's finished day is one line."""
        self.add(1, "offered")
        self.add(2, "offered")
        self.day[0] = "d2"
        self.assertEqual(len(self.stats.rollover()), 2)

    def test_no_text_or_user_ids_in_lines_or_snapshot(self):
        """FEAT-49c: only counts and the guild id; a name-like field cannot appear."""
        self.add(1, "offered")
        flat = "\n".join(self.stats.lines()) + repr(self.stats.snapshot())
        self.assertNotIn("user", flat.lower())


if __name__ == "__main__":
    unittest.main()
