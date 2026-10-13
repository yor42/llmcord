"""FEAT-49c: shadow mode and the veto counters in the Discord wiring. Written ahead of the implementation.

Required seams (state names), besides ``VetoStats`` (see test_spoken_veto_stats.py):

* ``SpokenMoves.shadow``: True when env ``LLMCORD_SPOKEN_VETO_SHADOW`` is one of "1", "true", "yes", "on" (``SpokenMoves(bot, env=...)``,
  falling back to ``os.environ`` like ``veto``); tests also assign it. It does not need the veto client to exist.
* ``SpokenMoves.stats``: the ``VetoStats`` the wiring records into; tests replace it with one whose ``today`` they control.
* Recording: ``offered`` just before the reader is asked (so not for gated, limited or notice-only messages), then ``errors`` on any
  failure, else ``vetoed`` when the policy dropped at least one e5 move and ``silenced`` when none was left. Shadow mode records the
  same and then posts e5's own buttons. For a check made from a shadow reading: a successful move press records ``pressed_vetoed``
  (move among those the reader would have dropped) or ``pressed_ok``; a check of a shadow-vetoed reading that ends without a move press
  on it (table-talk button, superseded by a newer phrase, deleted because the table moved on) records ``unpressed_vetoed`` once.
* ``close()`` and a day change (checked when a counter is recorded) log ``stats.lines()`` / ``stats.rollover()`` at INFO; once a guild-day
  is ``suspicious`` a WARNING containing the word "suspicious" and ``guild=<id>`` is logged once for it. No text, ever.
"""
import asyncio
import logging
import unittest

from test_spoken_veto_wiring import SENTINEL, VetoCase

from llmcord_core import games_spoken_discord as gsd
from llmcord_core.spoken.veto import VetoError
from llmcord_core.spoken.veto_limits import VetoLimiter
from test_games_spoken_discord import ALICE, BOB, G, Decision

D1 = "d1"


def new_stats(day):
    from llmcord_core.spoken import veto_stats
    return veto_stats.VetoStats(clock=lambda: 0.0, today=lambda: day[0])


def capture_logs(case):
    records = []

    class Grab(logging.Handler):
        def emit(self, record):
            records.append(record)

    root, handler = logging.getLogger(), Grab(logging.DEBUG)
    old = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    case.addCleanup(root.setLevel, old)
    case.addCleanup(root.removeHandler, handler)
    return records


class StatsCase(VetoCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.day = [D1]
        self.stats = self.spoken.stats = new_stats(self.day)

    def n(self, name, day=D1, guild=G):
        return self.stats.count(guild, day, name)

    def counts(self):
        return {name: self.n(name) for name in self.stats_names()}

    @staticmethod
    def stats_names():
        from llmcord_core.spoken import veto_stats
        return veto_stats.COUNTERS

    def assert_counts(self, **want):
        self.assertEqual(self.counts(), {**{n: 0 for n in self.stats_names()}, **want})

    def contents(self, msg):
        return [p.content for p in msg.posts]


class ShadowCase(StatsCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.spoken.shadow = True


class EnvTests(StatsCase):
    async def test_shadow_env_values(self):
        """FEAT-49c: the operator switch accepts 1/true/yes/on only; default off."""
        self.assertFalse(self.spoken.shadow)
        for value in ("1", "true", "yes", "on"):
            self.assertTrue(gsd.SpokenMoves(self.bot, env={"LLMCORD_SPOKEN_VETO_SHADOW": value}).shadow, value)
        for env in ({}, {"LLMCORD_SPOKEN_VETO_SHADOW": ""}, {"LLMCORD_SPOKEN_VETO_SHADOW": "0"}, {"LLMCORD_SPOKEN_VETO_SHADOW": "off"},
                    {"LLMCORD_SPOKEN_VETO_SHADOW": "no"}, {"LLMCORD_SPOKEN_VETO_SHADOW": "false"}):
            self.assertFalse(gsd.SpokenMoves(self.bot, env=env).shadow, env)

    async def test_shadow_does_not_create_a_client(self):
        """FEAT-49c: shadow needs the same operator switch and key as live; the flag alone makes no client."""
        self.assertIsNone(gsd.SpokenMoves(self.bot, env={"LLMCORD_SPOKEN_VETO_SHADOW": "1"}).veto)


class LiveRecordingTests(StatsCase):
    async def test_all_kept_counts_offered_only(self):
        """FEAT-49c: a reader that agrees with e5 is offered and nothing else."""
        await self.deal()
        self.assert_e5_buttons(await self.say(ALICE))
        self.assert_counts(offered=1)

    async def test_dropped_move_counts_vetoed(self):
        """FEAT-49c: live, dropping one of two moves is a veto, not a silence."""
        await self.deal()
        self.fake.probs = {"hit": 0.9, "stand": 0.1}
        msg = await self.say(ALICE)
        self.assertEqual(self.labels(msg), ["Hit", "Just table talk"])
        self.assert_counts(offered=1, vetoed=1)

    async def test_all_dropped_counts_vetoed_and_silenced(self):
        """FEAT-49c: nothing left is both."""
        await self.deal()
        self.fake.probs = {"hit": 0.1, "stand": 0.1}
        await self.say(ALICE)
        self.assert_counts(offered=1, vetoed=1, silenced=1)

    async def test_reorder_only_is_not_a_veto(self):
        """FEAT-49c: reordering drops nothing."""
        await self.deal()
        self.fake.probs = {"hit": 0.6, "stand": 0.8}
        await self.say(ALICE)
        self.assert_counts(offered=1)

    async def test_errors_are_counted_and_fail_open(self):
        """FEAT-49c: error, timeout and unexpected failures each count one error (and the call was offered)."""
        await self.deal()
        for n, error in enumerate((VetoError("status 500"), asyncio.TimeoutError(), RuntimeError("boom")), 1):
            self.fake.error = error
            self.assert_e5_buttons(await self.say(ALICE))
            self.assert_counts(offered=n, errors=n)

    async def test_slow_client_counts_an_error(self):
        """FEAT-49c: the veto_timeout expiring is an error."""
        await self.deal()
        self.spoken.veto_timeout = 0.05
        self.fake.gate = asyncio.Event()
        await self.say(ALICE)
        self.assert_counts(offered=1, errors=1)

    async def test_gated_and_silent_messages_are_not_offered(self):
        """FEAT-49c: no reader call, no count: e5 silent, guild switch off, breaker open, limiter spent."""
        await self.deal()
        self.decision = Decision("silent", ())
        await self.say(ALICE)
        self.decision = Decision("unsure", ("hit", "stand"))
        self.store.set_spoken_veto_enabled(G, False)
        await self.say(ALICE)
        self.store.set_spoken_veto_enabled(G, True)
        self.fake.breaker_open = True
        await self.say(ALICE)
        self.fake.breaker_open = False
        self.spoken.limiter = VetoLimiter(per_minute=1, per_day=1, clock=lambda: 0.0, today=lambda: 1)
        self.spoken.limiter.allow(G)
        await self.say(ALICE)
        self.assertEqual(self.fake.calls, [])
        self.assert_counts()

    async def test_live_checks_never_record_button_outcomes(self):
        """FEAT-49c: press outcomes are a shadow-mode measure; a live veto records no unpressed or false-veto count."""
        await self.deal()
        self.fake.probs = {"hit": 0.9, "stand": 0.1}
        msg = await self.say(ALICE)
        await self.click(ALICE, msg, None)
        self.assertEqual((self.n("pressed_vetoed"), self.n("unpressed_vetoed")), (0, 0))


class ShadowTests(ShadowCase):
    async def test_vetoed_reading_still_gets_e5_buttons(self):
        """FEAT-49c: shadow calls the reader, records the veto and applies nothing."""
        await self.deal()
        self.fake.probs = {"hit": 0.9, "stand": 0.1}
        self.assert_e5_buttons(await self.say(ALICE))
        self.assert_called_once_for("hit me")
        self.assert_counts(offered=1, vetoed=1)

    async def test_silenced_reading_still_gets_e5_buttons(self):
        """FEAT-49c: even a reading the reader would silence posts e5's check in shadow."""
        await self.deal()
        self.fake.probs = {"hit": 0.1, "stand": 0.1}
        self.assert_e5_buttons(await self.say(ALICE))
        self.assert_counts(offered=1, vetoed=1, silenced=1)
        self.assertEqual(self.spoken.pending[self.key()].offered, ("hit", "stand"))

    async def test_sure_vetoed_reading_still_gets_the_sure_check(self):
        """FEAT-49c: a sure e5 reading the reader would drop keeps its band and its button."""
        await self.deal()
        self.decision = Decision("sure", ("hit",))
        self.fake.probs = {"hit": 0.1}
        msg = await self.say(ALICE)
        self.assertEqual(self.labels(msg), ["Hit", "Just table talk"])
        self.assertEqual(self.spoken.pending[self.key()].band, "sure")
        self.assert_counts(offered=1, vetoed=1, silenced=1)

    async def test_reorder_is_not_applied_and_not_a_veto(self):
        """FEAT-49c: e5's order stands; reordering is no veto."""
        await self.deal()
        self.fake.probs = {"hit": 0.6, "stand": 0.8}
        self.assert_e5_buttons(await self.say(ALICE))
        self.assert_counts(offered=1)

    async def test_errors_are_counted(self):
        """FEAT-49c: failures count as in live mode and e5's buttons stand."""
        await self.deal()
        self.fake.error = VetoError("status 500")
        self.assert_e5_buttons(await self.say(ALICE))
        self.assert_counts(offered=1, errors=1)

    async def test_gates_are_identical_to_live_mode(self):
        """FEAT-49c: every gate holds in shadow: same posts, same calls, same counters as live, over each gate scenario."""
        scenarios = {
            "guild switch off": lambda c: c.store.set_spoken_veto_enabled(G, False),
            "member opt-out": lambda c: c.store.set_spoken_veto_member_on(G, ALICE, False),
            "notice not posted yet": lambda c: c.spoken.veto_notified.clear(),
            "spoken moves off for the member": lambda c: c.store.set_spoken_member_on(G, ALICE, False),
            "spoken moves off for the server": lambda c: c.store.set_spoken_moves_enabled(G, False),
            "breaker open": lambda c: setattr(c.fake, "breaker_open", True),
            "limiter spent": lambda c: c.spoken.__setattr__("limiter", _spent_limiter()),
            "e5 silent": lambda c: setattr(c, "decision", Decision("silent", ())),
            "no client": lambda c: setattr(c.spoken, "veto", None),
        }
        await self.deal()
        for name, setup in scenarios.items():
            outcomes = []
            for shadow in (False, True):
                self.fake.calls.clear()
                self.spoken.pending.clear()
                self.stats = self.spoken.stats = new_stats(self.day)
                self.spoken.shadow = shadow
                self.spoken.veto = self.fake
                self.spoken.veto_notified.update({(G, ALICE), (G, BOB)})
                self.fake.breaker_open = False
                self.decision = Decision("unsure", ("hit", "stand"))
                self.store.set_spoken_veto_enabled(G, True)
                self.store.set_spoken_veto_member_on(G, ALICE, True)
                self.store.set_spoken_member_on(G, ALICE, True)
                self.store.set_spoken_moves_enabled(G, True)
                self.spoken.limiter = VetoLimiter(clock=lambda: 0.0, today=lambda: 1)
                self.fake.probs = {"hit": 0.9, "stand": 0.9}
                setup(self)
                msg = await self.say(ALICE)
                outcomes.append((self.contents(msg), [c[0] for c in self.fake.calls], self.counts()))
            with self.subTest(gate=name):
                self.assertEqual(outcomes[0][1], [])  # the scenario really gates the call (live)
                self.assertEqual(outcomes[1], outcomes[0])

    async def test_notice_comes_first_in_shadow_too(self):
        """FEAT-49c: shadow needs the same opt-in layers: the notice is posted, nothing is sent with that message, then e5's check."""
        await self.deal()
        self.spoken.veto_notified.clear()
        msg = await self.say(ALICE)
        self.assertEqual(len(msg.posts), 2)
        self.assertIn("third-party", msg.posts[0].content)
        self.assertEqual((self.fake.calls, self.counts()), ([], {n: 0 for n in self.stats_names()}))


def _spent_limiter():
    limiter = VetoLimiter(per_minute=1, per_day=1, clock=lambda: 0.0, today=lambda: 1)
    limiter.allow(G)
    return limiter


class ShadowOutcomeTests(ShadowCase):
    async def vetoed_check(self, text="hit me", probs=None):
        self.fake.probs = probs or {"hit": 0.9, "stand": 0.1}
        return await self.say(ALICE, text)

    async def test_pressing_a_dropped_move_is_a_false_veto(self):
        """FEAT-49c: the player pressed what the reader would have dropped."""
        await self.deal()
        msg = await self.vetoed_check()
        await self.click(ALICE, msg, "stand")
        self.assert_counts(offered=1, vetoed=1, pressed_vetoed=1)

    async def test_pressing_a_kept_move_is_ok(self):
        """FEAT-49c: the player pressed a move the reader kept."""
        await self.deal()
        msg = await self.vetoed_check()
        await self.click(ALICE, msg, "hit")
        self.assert_counts(offered=1, vetoed=1, pressed_ok=1)

    async def test_pressing_on_an_unchanged_reading_is_ok(self):
        """FEAT-49c: a shadow reading that dropped nothing and was pressed counts as pressed_ok."""
        await self.deal()
        msg = await self.vetoed_check(probs={"hit": 0.9, "stand": 0.9})
        await self.click(ALICE, msg, "stand")
        self.assert_counts(offered=1, pressed_ok=1)

    async def test_table_talk_button_is_an_unpressed_veto(self):
        """FEAT-49c: 'Just table talk' on a shadow-vetoed check: the veto was probably right."""
        await self.deal()
        msg = await self.vetoed_check()
        await self.click(ALICE, msg, None)
        self.assert_counts(offered=1, vetoed=1, unpressed_vetoed=1)

    async def test_unchanged_reading_without_a_press_counts_nothing_more(self):
        """FEAT-49c: unpressed_vetoed is for vetoed readings only."""
        await self.deal()
        msg = await self.vetoed_check(probs={"hit": 0.9, "stand": 0.9})
        await self.click(ALICE, msg, None)
        self.assert_counts(offered=1)

    async def test_superseded_check_is_an_unpressed_veto(self):
        """FEAT-49c: a newer phrase replaces the vetoed check unanswered."""
        await self.deal()
        await self.vetoed_check("one")
        self.fake.probs = {"hit": 0.9, "stand": 0.9}
        await self.say(ALICE, "two")
        self.assert_counts(offered=2, vetoed=1, unpressed_vetoed=1)

    async def test_table_moving_on_is_an_unpressed_veto(self):
        """FEAT-49c: the member used the table's own button instead; the vetoed check expires."""
        await self.deal()
        await self.vetoed_check()
        await self.press(ALICE, "Hit")
        await self.settle()
        self.assert_counts(offered=1, vetoed=1, unpressed_vetoed=1)

    async def test_a_press_is_counted_once(self):
        """FEAT-49c: the press that moves the table does not also count as an expiry."""
        await self.deal()
        msg = await self.vetoed_check()
        await self.click(ALICE, msg, "stand")
        await self.settle()
        self.assert_counts(offered=1, vetoed=1, pressed_vetoed=1)

    async def test_someone_elses_press_counts_nothing(self):
        """FEAT-49c: a bystander cannot answer the member's check, so nothing is recorded."""
        await self.deal()
        msg = await self.vetoed_check()
        await self.click(BOB, msg, "stand")
        self.assert_counts(offered=1, vetoed=1)
        self.assertIn(self.key(), self.spoken.pending)


class LoggingTests(StatsCase):
    async def test_close_logs_the_counts_at_info(self):
        """FEAT-49c: close() writes the summary lines, counts only."""
        await self.deal()
        self.fake.probs = {"hit": 0.9, "stand": 0.1}
        await self.say(ALICE, f"x {SENTINEL}")
        with self.assertLogs(level="INFO") as logs:
            await self.spoken.close()
        lines = [r for r in logs.records if r.levelno == logging.INFO and "offered=1" in r.getMessage()]
        self.assertEqual(len(lines), 1)
        message = lines[0].getMessage().split()
        for token in (f"guild={G}", f"day={D1}", "vetoed=1"):
            self.assertIn(token, message)
        self.assertNotIn(SENTINEL, "\n".join(r.getMessage() for r in logs.records))

    async def test_day_rollover_logs_the_finished_day_once(self):
        """FEAT-49c: the first counter of a new day logs the old day's summary; later ones do not repeat it."""
        await self.deal()
        await self.say(ALICE)
        self.day[0] = "d2"
        with self.assertLogs(level="INFO") as logs:
            await self.say(ALICE, "again")
        finished = [r.getMessage() for r in logs.records if "day=d1" in r.getMessage().split()]
        self.assertEqual(len(finished), 1)
        self.assertIn("offered=1", finished[0].split())
        with self.assertLogs(level="INFO") as logs:
            await self.say(ALICE, "third")
        self.assertEqual([r for r in logs.records if "day=d1" in r.getMessage().split()], [])

    async def test_suspicious_rate_warns_once_per_guild_and_day(self):
        """FEAT-49c: after 20 offered readings the reader never vetoed: one WARNING, no text."""
        await self.deal()
        self.spoken.limiter = VetoLimiter(per_minute=1000, per_day=1000, clock=lambda: 0.0, today=lambda: 1)
        with self.assertLogs(level="WARNING") as logs:
            for n in range(22):
                await self.say(ALICE, f"hit me {n} {SENTINEL}")
        hits = [r for r in logs.records if "suspicious" in r.getMessage()]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].levelno, logging.WARNING)
        self.assertIn(f"guild={G}", hits[0].getMessage())
        self.assertNotIn(SENTINEL, "\n".join(r.getMessage() for r in logs.records))
        self.assertTrue(self.stats.suspicious(G, D1))

    async def test_suspicious_veto_everything_warns(self):
        """FEAT-49c: a 100% veto rate over 20 readings is also suspicious."""
        await self.deal()
        self.spoken.limiter = VetoLimiter(per_minute=1000, per_day=1000, clock=lambda: 0.0, today=lambda: 1)
        self.fake.probs = {"hit": 0.1, "stand": 0.1}
        with self.assertLogs(level="WARNING") as logs:
            for n in range(21):
                await self.say(ALICE, f"hit me {n}")
        self.assertEqual(len([r for r in logs.records if "suspicious" in r.getMessage()]), 1)

    async def test_a_new_day_can_warn_again(self):
        """FEAT-49c: the once-only rule is per guild and day."""
        await self.deal()
        self.spoken.limiter = VetoLimiter(per_minute=1000, per_day=1000, clock=lambda: 0.0, today=lambda: 1)
        with self.assertLogs(level="WARNING") as logs:
            for day in ("d1", "d2"):
                self.day[0] = day
                for n in range(21):
                    await self.say(ALICE, f"hit me {n}")
        self.assertEqual(len([r for r in logs.records if "suspicious" in r.getMessage()]), 2)

    async def test_few_readings_do_not_warn(self):
        """FEAT-49c: under 20 readings nothing is suspicious."""
        await self.deal()
        self.spoken.limiter = VetoLimiter(per_minute=1000, per_day=1000, clock=lambda: 0.0, today=lambda: 1)
        records = capture_logs(self)
        for n in range(19):
            await self.say(ALICE, f"hit me {n}")
        self.assertEqual([r for r in records if r.levelno >= logging.WARNING], [])


class PrivacyTests(StatsCase):
    async def run_modes(self, shadow):
        records = capture_logs(self)
        self.spoken.shadow = shadow
        await self.deal()
        self.fake.probs = {"hit": 0.9, "stand": 0.1}
        msg = await self.say(ALICE, f"success {SENTINEL}")
        self.fake.probs = {"hit": 0.1, "stand": 0.1}
        await self.say(ALICE, f"dropped {SENTINEL}")
        for error in (VetoError("status 500"), asyncio.TimeoutError(), RuntimeError("boom")):
            self.fake.error = error
            await self.say(ALICE, f"error {SENTINEL}")
        self.fake.error = None
        self.fake.probs = {"hit": 0.9, "stand": 0.1}
        await self.say(ALICE, f"press {SENTINEL}")
        self.day[0] = "d2"
        msg = await self.say(ALICE, f"next day {SENTINEL}")
        await self.click(ALICE, msg, "stand" if shadow else "hit")
        await self.spoken.close()
        self.assertGreater(self.n("offered", "d2") + self.n("offered"), 0)
        self.assertNotIn(SENTINEL, "\n".join(self.stats.lines()))
        self.assertNotIn(SENTINEL, repr(self.stats.snapshot()))
        self.assertNotIn(SENTINEL, "\n".join(logging.Formatter().format(r) for r in records))

    async def test_live_counters_and_logs_hold_no_text(self):
        """FEAT-49c: live mode, success, drop, every error path, rollover and close: no message text anywhere."""
        await self.run_modes(False)

    async def test_shadow_counters_and_logs_hold_no_text(self):
        """FEAT-49c: the same in shadow mode, including the button outcomes."""
        await self.run_modes(True)


if __name__ == "__main__":
    unittest.main()
