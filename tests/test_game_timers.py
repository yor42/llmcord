"""Characterization (MNT-50): timer, lock and close behavior shared by the Blackjack and I Doubt It adapters.

Pins what ``GameTables`` and ``DoubtTables`` do through the shared ``TableRuntime``. Tests drive ``arm``/``disarm``/``close``/``lock``
and the ``on_*_timer`` hooks. The migration surface, i.e. everything below that a refactor may have to touch, is confined to
the two adapter wrappers and these helpers:

- ``new_tasks()``: a snapshot of ``asyncio.all_tasks()`` taken in ``asyncSetUp``; "no live timer task" means no new
  pending task other than the test's own and ``tables.sweeper``. No task names are matched.
- ``timer_present()``: reads ``tables.runtime.timers`` (key shape ``((guild, table), "table")`` for Blackjack, ``((guild, table), kind)``
  for I Doubt It). The only other direct reads are ``tables.closed``, ``tables.sweeper`` and, for one Blackjack test,
  ``tables.table(key).deadline``.
- ``set_retry``/``mark_closed``/``lock`` go through ``tables.runtime`` (``retry_seconds``, ``closed``, ``lock``); disarm goes
  through ``tables.disarm(key)`` for Blackjack and ``tables.runtime.disarm(key, kind)`` for I Doubt It.
- hooks (``on_*_timer``) and, for I Doubt It, ``_reap`` are replaced on the instance with scripted fakes.
- ``SWEEP_SECONDS`` and the per-kind ``*_SECONDS`` (Blackjack only) are set on the instance.
"""
import asyncio
import unittest
from types import SimpleNamespace

from llmcord_core.games_discord import GameTables
from llmcord_core.games_doubt_discord import DoubtTables

DELAY = 0.01


class _Adapter:
    """Uniform driver over one adapter: arm/disarm a kind for a table, with scripted hooks."""

    def __init__(self, tables, kind, other_kind):
        self.tables, self.key, self.kind, self.other_kind = tables, (1, 7), kind, other_kind
        self.calls: list[str] = []
        self.failures: dict[str, int] = {}  # kind -> how many more calls raise
        self.script = None  # optional async callable run inside every hook, before any scripted failure
        self.baseline: set = set()
        self.set_retry(DELAY)
        tables.SWEEP_SECONDS = 3600

    def set_retry(self, seconds):
        self.tables.runtime.retry_seconds = seconds

    def mark_closed(self):
        self.tables.runtime.closed = True

    def lock(self, key):
        return self.tables.runtime.lock(key)

    def hook(self, kind):
        async def hook(*_args):
            self.calls.append(kind)
            if self.script is not None:
                await self.script()
            if self.failures.get(kind, 0) > 0:
                self.failures[kind] -= 1
                raise RuntimeError("boom")
        return hook

    def snapshot(self):
        self.baseline = set(asyncio.all_tasks())

    def new_tasks(self):
        """Pending tasks created since the snapshot, other than the running test and the table sweeper."""
        ignored = self.baseline | {asyncio.current_task(), self.tables.sweeper}
        return [t for t in asyncio.all_tasks() if t not in ignored and not t.done()]


class _Blackjack(_Adapter):
    seconds = {"join": "JOIN_SECONDS", "turn": "TURN_SECONDS", "idle": "IDLE_SECONDS", "character": "CHARACTER_SECONDS"}

    def __init__(self, **kwargs):
        tables = GameTables(SimpleNamespace(store=None), **kwargs)
        super().__init__(tables, "turn", "join")
        for name, kind in (("on_join_timer", "join"), ("on_turn_timer", "turn"), ("on_character_timer", "character"), ("on_idle_timer", "idle")):
            setattr(tables, name, self.hook(kind))

    def arm(self, kind=None, delay=DELAY, key=None):
        kind = kind or self.kind
        setattr(self.tables, self.seconds[kind], delay)
        self.tables.arm(key or self.key, kind, 1)

    def disarm(self, kind=None, key=None):
        self.tables.disarm(key or self.key)

    def timer_present(self, kind=None, key=None) -> bool:
        return ((key or self.key), self.tables.SLOT) in self.tables.runtime.timers


class _Doubt(_Adapter):
    def __init__(self):
        tables = DoubtTables(SimpleNamespace(store=None))
        super().__init__(tables, "turn", "window")
        for name, kind in (("on_lobby_timer", "lobby"), ("on_window_timer", "window"), ("on_turn_timer", "turn"),
                           ("on_character_timer", "character"), ("on_archive_timer", "archive")):
            setattr(tables, name, self.hook(kind))
        self.reaped: list[tuple] = []  # every _reap call, including ones that raised
        self.reap_failures = 0

        async def reap(key):
            self.reaped.append(key)
            if self.reap_failures > 0:
                self.reap_failures -= 1
                raise RuntimeError("reap boom")
        tables._reap = reap

    def arm(self, kind=None, delay=DELAY, key=None):
        self.tables.arm(key or self.key, kind or self.kind, delay)

    def disarm(self, kind=None, key=None):
        self.tables.runtime.disarm(key or self.key, kind or self.kind)

    def timer_present(self, kind=None, key=None) -> bool:
        return ((key or self.key), kind or self.kind) in self.tables.runtime.timers


async def settle(seconds=DELAY * 6):
    """A short fixed wait, only for asserting that nothing more happens."""
    await asyncio.sleep(seconds)


async def until(predicate, within=2.0):
    """Polls for something that should happen; fails the test loudly when it does not within the limit."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + within
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for the expected state")
        await asyncio.sleep(0.005)


class _TimerCases:
    make = None

    async def asyncSetUp(self):
        self.a = self.make()
        self.a.snapshot()

    async def asyncTearDown(self):
        await self.a.tables.close()

    async def test_arm_creates_exactly_one_new_task(self):
        """Characterization (MNT-50): positive control for the task snapshot: one armed timer is one live task."""
        self.assertEqual(self.a.new_tasks(), [])
        self.a.arm(delay=3600)
        self.assertEqual(len(self.a.new_tasks()), 1)

    async def test_hook_fires_once_then_timer_removes_itself(self):
        """Characterization (MNT-50): the hook runs once after its delay and nothing stays tracked."""
        self.a.arm()
        self.assertTrue(self.a.timer_present())
        await until(lambda: self.a.calls == [self.a.kind] and not self.a.timer_present())
        await settle()
        self.assertEqual(self.a.calls, [self.a.kind])
        self.assertEqual(self.a.new_tasks(), [])

    async def test_failing_hook_is_retried_once_and_succeeds(self):
        """Characterization (MNT-50): a hook that raises once is retried once; the failure is logged once."""
        self.a.failures[self.a.kind] = 1
        with self.assertLogs(level="ERROR") as logs:
            self.a.arm()
            await until(lambda: len(self.a.calls) == 2 and not self.a.timer_present())
        await settle()
        self.assertEqual(self.a.calls, [self.a.kind, self.a.kind])
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(self.a.new_tasks(), [])

    async def test_hook_failing_twice_is_logged_twice_and_never_run_a_third_time(self):
        """Characterization (MNT-50): two failures are two logs, no third run, nothing left tracked."""
        self.a.failures[self.a.kind] = 99
        with self.assertLogs(level="ERROR") as logs:
            self.a.arm()
            await until(lambda: len(self.a.calls) == 2 and not self.a.timer_present())
            await settle()
        self.assertEqual(self.a.calls, [self.a.kind, self.a.kind])
        self.assertEqual(len(logs.records), 2)
        self.assertEqual(self.a.new_tasks(), [])

    async def test_disarm_during_retry_sleep_cancels_the_retry(self):
        """Characterization (MNT-50): cancellation alone stops a disarmed timer's retry (not the stale-timer guard)."""
        self.a.failures[self.a.kind] = 99
        self.a.set_retry(5)  # long: only cancellation can keep the retry from running
        with self.assertLogs(level="ERROR"):
            self.a.arm()
            await until(lambda: self.a.calls == [self.a.kind])
            self.a.disarm()
            await settle(DELAY * 2)
        self.assertEqual(self.a.calls, [self.a.kind])
        self.assertEqual(self.a.new_tasks(), [])

    async def test_replacement_during_retry_sleep_cancels_the_stale_retry(self):
        """Characterization (MNT-50): cancellation alone stops a replaced timer's retry (not the stale-timer guard)."""
        self.a.failures[self.a.kind] = 1
        self.a.set_retry(5)  # long: only cancellation can keep the retry from running
        with self.assertLogs(level="ERROR"):
            self.a.arm()
            await until(lambda: self.a.calls == [self.a.kind])
            self.a.arm(delay=3600)  # replaces the retrying timer with a long one
            await settle(DELAY * 2)
        self.assertEqual(self.a.calls, [self.a.kind])
        self.assertTrue(self.a.timer_present())
        self.assertEqual(len(self.a.new_tasks()), 1)

    async def test_hook_that_disarms_its_own_timer_and_fails_is_not_retried(self):
        """Characterization (MNT-50): the stale-timer guard stops the retry of a timer that disarmed itself from inside its hook."""
        self.a.failures[self.a.kind] = 99
        self.a.script = self.a_disarm_own
        with self.assertLogs(level="ERROR") as logs:
            self.a.arm()
            await until(lambda: self.a.calls == [self.a.kind])
            await settle(DELAY * 8)
        self.assertEqual(self.a.calls, [self.a.kind])
        self.assertEqual(len(logs.records), 1)
        self.assertFalse(self.a.timer_present())
        self.assertEqual(self.a.new_tasks(), [])

    async def test_hook_that_rearms_its_own_key_and_fails_leaves_the_new_timer_alone(self):
        """Characterization (MNT-50): a failing hook that re-armed its own key is not retried and does not cancel the new timer."""
        self.a.failures[self.a.kind] = 99
        self.a.script = self.a_rearm_own
        with self.assertLogs(level="ERROR") as logs:
            self.a.arm()
            await until(lambda: self.a.calls == [self.a.kind])
            await settle(DELAY * 8)
        self.assertEqual(self.a.calls, [self.a.kind])
        self.assertEqual(len(logs.records), 1)
        self.assertTrue(self.a.timer_present())
        pending = self.a.new_tasks()
        self.assertEqual(len(pending), 1)
        self.assertFalse(pending[0].cancelled())

    async def test_hook_that_fails_after_the_adapter_closed_is_not_retried(self):
        """Characterization (MNT-50): the stale-timer guard also stops the retry once the adapter is closed."""
        self.a.failures[self.a.kind] = 99

        async def close_then_fail():
            self.a.mark_closed()  # what close() sets first; calling close() from inside its own task would cancel the hook
        self.a.script = close_then_fail
        with self.assertLogs(level="ERROR") as logs:
            self.a.arm()
            await until(lambda: self.a.calls == [self.a.kind])
            await settle(DELAY * 8)
        self.assertEqual(self.a.calls, [self.a.kind])
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(self.a.new_tasks(), [])

    async def test_hook_may_disarm_and_rearm_without_cancelling_itself(self):
        """Characterization (MNT-50): disarm of the running timer's own key never cancels the running hook."""
        other = (1, 8)

        async def churn():
            self.a.disarm()
            self.a.arm(self.a.other_kind, delay=3600, key=other)
            await asyncio.sleep(0)
            self.a.calls.append("completed")
        self.a.script = churn
        self.a.arm()
        await until(lambda: "completed" in self.a.calls)
        self.assertEqual(self.a.calls, [self.a.kind, "completed"])
        self.assertTrue(self.a.timer_present(self.a.other_kind, other))

    async def test_disarm_cancels_pending_timer_and_is_harmless_when_none(self):
        """Characterization (MNT-50): disarm stops a pending hook; disarming with nothing armed does nothing."""
        self.a.disarm()  # nothing armed
        self.a.arm()
        self.a.disarm()
        self.assertFalse(self.a.timer_present())
        await settle()
        self.assertEqual(self.a.calls, [])
        self.assertEqual(self.a.new_tasks(), [])

    async def test_close_cancels_timers_and_blocks_later_arming(self):
        """Characterization (MNT-50): close cancels every timer, sets closed, and later arm calls are no-ops."""
        self.a.arm(delay=3600)
        self.a.arm(self.a.other_kind, delay=3600, key=(1, 8))
        self.assertEqual(len(self.a.new_tasks()), 2)
        await self.a.tables.close()
        self.assertTrue(self.a.tables.closed)
        self.assertEqual(self.a.new_tasks(), [])
        self.assertFalse(self.a.timer_present())
        self.a.arm()
        await settle()
        self.assertEqual(self.a.calls, [])
        self.assertFalse(self.a.timer_present())
        self.assertEqual(self.a.new_tasks(), [])

    async def test_close_cancels_a_hook_that_is_mid_run(self):
        """Characterization (MNT-50): close() cancels a timer whose hook is running (awaiting) and leaves no task behind."""
        gate = asyncio.Event()  # never set
        self.a.script = gate.wait
        self.a.arm()
        await until(lambda: self.a.calls == [self.a.kind])
        await self.a.tables.close()
        self.assertEqual(self.a.calls, [self.a.kind])
        self.assertFalse(self.a.timer_present())
        self.assertEqual(self.a.new_tasks(), [])

    async def test_lock_is_shared_per_key_while_referenced(self):
        """Characterization (MNT-50): the same key gives the same Lock while referenced; other keys get other locks."""
        first = self.a.lock((1, 7))
        self.assertIsInstance(first, asyncio.Lock)
        self.assertIs(self.a.lock((1, 7)), first)
        self.assertIsNot(self.a.lock((1, 8)), first)

    # scripts that act on the timer's own key from inside its hook (kind of the running hook is self.a.kind)
    async def a_disarm_own(self):
        self.a.disarm()

    async def a_rearm_own(self):
        self.a.arm(delay=3600)


class BlackjackTimerTests(_TimerCases, unittest.IsolatedAsyncioTestCase):
    make = staticmethod(_Blackjack)

    async def test_arming_any_kind_replaces_the_tables_previous_timer(self):
        """Characterization (MNT-50): one timer per Blackjack table, whatever the kind."""
        self.a.arm("join", delay=3600)
        self.a.arm("turn")
        await asyncio.sleep(0)  # let the cancelled join timer finish
        self.assertEqual(len(self.a.new_tasks()), 1)
        await until(lambda: self.a.calls == ["turn"])
        await settle()
        self.assertEqual(self.a.calls, ["turn"])
        self.assertEqual(self.a.new_tasks(), [])

    async def test_arm_sets_the_table_deadline_from_the_clock(self):
        """Characterization (MNT-50): arm stamps table.deadline = clock() + the kind's delay (shown as a countdown on the table)."""
        a = _Blackjack(clock=lambda: 1000.0)
        try:
            a.arm("turn", delay=60)
            self.assertEqual(a.tables.table(a.key).deadline, 1060.0)
        finally:
            await a.tables.close()


class DoubtTimerTests(_TimerCases, unittest.IsolatedAsyncioTestCase):
    make = staticmethod(_Doubt)

    async def test_same_kind_replaces_but_different_kinds_coexist(self):
        """Characterization (MNT-50): one I Doubt It timer per (table, kind)."""
        self.a.arm("turn", delay=3600)
        self.a.arm("turn")  # replaces the long one
        self.a.arm("window", delay=3600)
        await asyncio.sleep(0)  # let the replaced timer finish cancelling
        self.assertEqual(len(self.a.new_tasks()), 2)
        await until(lambda: self.a.calls == ["turn"] and not self.a.timer_present("turn"))
        self.assertTrue(self.a.timer_present("window"))
        self.a.disarm("window")
        self.assertFalse(self.a.timer_present("window"))
        await asyncio.sleep(0)  # let the cancelled window timer finish
        self.assertEqual(self.a.new_tasks(), [])

    async def test_reap_runs_after_play_hooks_but_not_after_archive(self):
        """Characterization (MNT-50): a finished play hook reaps the table; the archive hook does not."""
        self.a.arm("turn")
        await until(lambda: self.a.reaped == [(1, 7)])
        self.a.arm("archive")
        await until(lambda: self.a.calls == ["turn", "archive"])
        await settle()
        self.assertEqual(self.a.reaped, [(1, 7)])

    async def test_reap_runs_once_when_the_retry_succeeds(self):
        """Characterization (MNT-50): a hook that failed once and then succeeded reaps exactly once, after the success."""
        self.a.failures["turn"] = 1
        with self.assertLogs(level="ERROR"):
            self.a.arm("turn")
            await until(lambda: len(self.a.calls) == 2 and self.a.reaped)
        await settle()
        self.assertEqual(self.a.reaped, [(1, 7)])

    async def test_reap_does_not_run_when_the_hook_fails_twice(self):
        """Characterization (MNT-50): no reap after a hook that failed both attempts."""
        self.a.failures["turn"] = 99
        with self.assertLogs(level="ERROR"):
            self.a.arm("turn")
            await until(lambda: len(self.a.calls) == 2 and not self.a.timer_present("turn"))
        await settle()
        self.assertEqual(self.a.reaped, [])

    async def test_reap_failure_after_a_successful_hook_reruns_the_whole_attempt(self):
        """Characterization (MNT-50): if _reap raises, the hook is retried too, so it runs twice (and is then reaped again)."""
        self.a.reap_failures = 1
        with self.assertLogs(level="ERROR") as logs:
            self.a.arm("turn")
            await until(lambda: len(self.a.reaped) == 2 and not self.a.timer_present("turn"))
        await settle()
        self.assertEqual(self.a.calls, ["turn", "turn"])
        self.assertEqual(self.a.reaped, [(1, 7), (1, 7)])
        self.assertEqual(len(logs.records), 1)

    async def test_reap_failing_on_both_attempts_runs_the_hook_and_the_reap_twice(self):
        """Characterization (MNT-50): a _reap that always raises makes both attempts fail after the hook succeeded: two hook runs, two reaps, two logs, no third try."""
        self.a.reap_failures = 99
        with self.assertLogs(level="ERROR") as logs:
            self.a.arm("turn")
            await until(lambda: len(self.a.reaped) == 2 and not self.a.timer_present("turn"))
            await settle()
        self.assertEqual(self.a.calls, ["turn", "turn"])
        self.assertEqual(self.a.reaped, [(1, 7), (1, 7)])
        self.assertEqual(len(logs.records), 2)
        self.assertEqual(self.a.new_tasks(), [])
