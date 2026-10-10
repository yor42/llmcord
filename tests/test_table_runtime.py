"""MNT-50: tests of the shared ``TableRuntime`` on its own: no bot, no Discord, no real time.

Sleeping is injected: ``FakeSleep`` records each requested delay and holds the sleeper until the test releases it, so the
tests are deterministic and instant.
"""
import asyncio
import gc
import unittest

from llmcord_core.table_runtime import TableRuntime

KEY = (1, 7)


class FakeSleep:
    def __init__(self):
        self.delays: list[float] = []
        self.waiters: list[asyncio.Future] = []

    async def __call__(self, delay):
        waiter = asyncio.get_running_loop().create_future()
        self.delays.append(delay)
        self.waiters.append(waiter)
        await waiter

    def release(self, index):
        self.waiters[index].set_result(None)


async def until(predicate):
    """Lets the loop run (no real waiting) until the predicate holds."""
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("the expected state was never reached")


class TableRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sleep = FakeSleep()
        self.rt = TableRuntime(label="Test", sleep=self.sleep, retry_seconds=0.5)
        self.baseline = set(asyncio.all_tasks())
        self.calls: list[str] = []
        self.failures = 0
        self.script = None

    async def asyncTearDown(self):
        await self.rt.close()

    def new_tasks(self):
        ignored = self.baseline | {asyncio.current_task()}
        return [t for t in asyncio.all_tasks() if t not in ignored and not t.done()]

    async def hook(self, *args):
        self.calls.append("hook")
        if self.script is not None:
            await self.script()
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("boom")

    async def first_sleep_released(self, kind="turn", delay=3.0):
        self.rt.arm(KEY, kind, delay, self.hook)
        await until(lambda: len(self.sleep.delays) == 1)
        self.sleep.release(0)

    async def test_hook_fires_after_the_requested_delay_and_the_timer_removes_itself(self):
        """Characterization (MNT-50): the first sleep is the requested delay; the hook runs once after it; the slot is then free."""
        self.rt.arm(KEY, "turn", 3.0, self.hook)
        await until(lambda: self.sleep.delays == [3.0])
        self.assertEqual(self.calls, [])
        self.assertIn((KEY, "turn"), self.rt.timers)
        self.sleep.release(0)
        await until(lambda: not self.rt.timers)
        self.assertEqual(self.calls, ["hook"])
        self.assertEqual(self.new_tasks(), [])

    async def test_hook_receives_the_extra_arguments(self):
        """Characterization (MNT-50): arm passes its trailing arguments to the hook and nothing else."""
        seen = []

        async def hook(*args):
            seen.append(args)
        self.rt.arm(KEY, "turn", 1, hook, "a", 2)
        await until(lambda: self.sleep.waiters)
        self.sleep.release(0)
        await until(lambda: not self.rt.timers)
        self.assertEqual(seen, [("a", 2)])

    async def test_failed_hook_is_retried_once_after_retry_seconds_and_succeeds(self):
        """Characterization (MNT-50): the retry sleeps retry_seconds; one failure is logged once; the second run succeeds."""
        self.failures = 1
        with self.assertLogs(level="ERROR") as logs:
            await self.first_sleep_released()
            await until(lambda: len(self.sleep.delays) == 2)
            self.assertEqual(self.sleep.delays, [3.0, 0.5])
            self.assertEqual(self.calls, ["hook"])
            self.sleep.release(1)
            await until(lambda: not self.rt.timers)
        self.assertEqual(self.calls, ["hook", "hook"])
        self.assertEqual([r.getMessage() for r in logs.records], ["Test turn timer failed (table 7, attempt 1)"])
        self.assertEqual(self.new_tasks(), [])

    async def test_hook_failing_twice_is_logged_twice_and_run_exactly_twice(self):
        """Characterization (MNT-50): two failures give two logs with the attempt number, two runs, no third run, nothing tracked."""
        self.failures = 99
        with self.assertLogs(level="ERROR") as logs:
            await self.first_sleep_released()
            await until(lambda: len(self.sleep.delays) == 2)
            self.sleep.release(1)
            await until(lambda: not self.rt.timers)
        self.assertEqual(self.calls, ["hook", "hook"])
        self.assertEqual([r.getMessage() for r in logs.records],
                         ["Test turn timer failed (table 7, attempt 1)", "Test turn timer failed (table 7, attempt 2)"])
        self.assertEqual(len(self.sleep.delays), 2)
        self.assertEqual(self.new_tasks(), [])

    async def test_hook_that_disarms_its_own_slot_and_fails_is_not_retried(self):
        """Characterization (MNT-50): the stale guard stops the retry of a timer that disarmed itself."""
        self.failures = 99

        async def script():
            self.rt.disarm(KEY, "turn")
        self.script = script
        with self.assertLogs(level="ERROR") as logs:
            await self.first_sleep_released()
            await until(lambda: not self.new_tasks())
        self.assertEqual(self.calls, ["hook"])
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(len(self.sleep.delays), 1)  # no retry sleep was ever started

    async def test_hook_that_rearms_its_own_slot_and_fails_leaves_the_new_timer_alone(self):
        """Characterization (MNT-50): a failing hook that re-armed its slot is not retried and does not cancel the new timer."""
        self.failures = 99

        async def noop():
            pass

        async def script():
            self.rt.arm(KEY, "turn", 99, noop)
        self.script = script
        with self.assertLogs(level="ERROR") as logs:
            await self.first_sleep_released()
            await until(lambda: len(self.sleep.delays) == 2)
            await until(lambda: len(self.new_tasks()) == 1)
        self.assertEqual(self.sleep.delays, [3.0, 99])  # the second sleep is the new timer's own
        self.assertEqual(self.calls, ["hook"])
        self.assertEqual(len(logs.records), 1)
        (pending,) = self.new_tasks()
        self.assertFalse(pending.cancelled())
        self.assertIs(self.rt.timers[(KEY, "turn")], pending)

    async def test_hook_that_fails_after_closed_is_set_is_not_retried(self):
        """Characterization (MNT-50): the stale guard also stops the retry once closed is set."""
        self.failures = 99

        async def script():
            self.rt.closed = True
        self.script = script
        with self.assertLogs(level="ERROR") as logs:
            await self.first_sleep_released()
            await until(lambda: not self.new_tasks())
        self.assertEqual(self.calls, ["hook"])
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(len(self.sleep.delays), 1)

    async def test_hook_may_disarm_and_rearm_without_cancelling_itself(self):
        """Characterization (MNT-50): disarm of the running timer's own slot never cancels the running hook."""
        async def noop():
            pass

        async def script():
            self.rt.disarm(KEY, "turn")
            self.rt.arm((1, 8), "turn", 99, noop)
            await asyncio.sleep(0)
            self.calls.append("completed")
        self.script = script
        await self.first_sleep_released()
        await until(lambda: "completed" in self.calls)
        self.assertEqual(self.calls, ["hook", "completed"])
        self.assertIn(((1, 8), "turn"), self.rt.timers)

    async def test_replacing_the_same_slot_cancels_the_old_timer_and_other_kinds_coexist(self):
        """Characterization (MNT-50): one timer per (key, kind): the same slot is replaced, another kind for the key coexists."""
        self.rt.arm(KEY, "turn", 10, self.hook)
        old = self.rt.timers[(KEY, "turn")]
        self.rt.arm(KEY, "window", 20, self.hook)
        self.rt.arm(KEY, "turn", 30, self.hook)
        await until(lambda: old.cancelled())
        self.assertEqual(set(self.rt.timers), {(KEY, "turn"), (KEY, "window")})
        await until(lambda: len(self.sleep.delays) == 2)
        self.assertEqual(sorted(self.sleep.delays), [20, 30])
        self.assertEqual(len(self.new_tasks()), 2)
        self.assertEqual(self.calls, [])

    async def test_disarm_cancels_the_timer_and_is_harmless_when_none(self):
        """Characterization (MNT-50): disarm stops a pending hook; disarming an empty slot does nothing."""
        self.rt.disarm(KEY, "turn")
        self.rt.arm(KEY, "turn", 3, self.hook)
        self.rt.disarm(KEY, "turn")
        self.assertEqual(self.rt.timers, {})
        await until(lambda: not self.new_tasks())
        self.assertEqual(self.calls, [])

    async def test_arm_after_close_is_a_no_op(self):
        """Characterization (MNT-50): once closed, arm creates no timer and no task."""
        await self.rt.close()
        self.assertTrue(self.rt.closed)
        self.rt.arm(KEY, "turn", 3, self.hook)
        self.assertEqual(self.rt.timers, {})
        await asyncio.sleep(0)
        self.assertEqual(self.sleep.delays, [])
        self.assertEqual(self.new_tasks(), [])

    async def test_close_cancels_pending_timers_and_a_hook_that_is_mid_run(self):
        """Characterization (MNT-50): close cancels timers still sleeping and a hook awaiting something, leaving no task."""
        gate = asyncio.Event()  # never set

        async def stuck():
            self.calls.append("stuck")
            await gate.wait()
        self.rt.arm(KEY, "turn", 1, stuck)
        self.rt.arm((1, 8), "turn", 99, self.hook)
        await until(lambda: len(self.sleep.delays) == 2)
        self.sleep.release(0)
        await until(lambda: self.calls == ["stuck"])
        await self.rt.close()
        self.assertEqual(self.rt.timers, {})
        self.assertEqual(self.new_tasks(), [])
        self.assertEqual(self.calls, ["stuck"])

    async def test_close_during_the_retry_sleep_stops_the_retry(self):
        """Characterization (MNT-50): close() cancels a timer waiting to retry, so the hook is not run again."""
        self.failures = 99
        with self.assertLogs(level="ERROR"):
            await self.first_sleep_released()
            await until(lambda: len(self.sleep.delays) == 2)
            await self.rt.close()
        self.assertEqual(self.calls, ["hook"])
        self.assertEqual(self.new_tasks(), [])

    async def test_task_name_defaults_from_label_kind_and_table_and_an_explicit_name_wins(self):
        """Characterization (MNT-50): default task name is f'{label}-{kind}-{key[1]}' lowercased; name= overrides it."""
        self.rt.arm(KEY, "Turn", 1, self.hook)
        self.rt.arm((1, 8), "turn", 1, self.hook, name="custom-name")
        self.assertEqual(self.rt.timers[(KEY, "Turn")].get_name(), "test-turn-7")
        self.assertEqual(self.rt.timers[((1, 8), "turn")].get_name(), "custom-name")

    async def test_lock_is_shared_per_key_while_referenced(self):
        """Characterization (MNT-50): the same key gives the same Lock while referenced; different keys get different locks."""
        first = self.rt.lock(KEY)
        self.assertIsInstance(first, asyncio.Lock)
        self.assertIs(self.rt.lock(KEY), first)
        self.assertIsNot(self.rt.lock((1, 8)), first)

    async def test_lock_is_dropped_when_the_last_reference_goes(self):
        """Characterization (MNT-50): locks are weakly held, so an unreferenced table's lock does not accumulate."""
        lock = self.rt.lock(KEY)
        self.assertIn(KEY, self.rt.locks)
        del lock
        gc.collect()
        self.assertNotIn(KEY, self.rt.locks)
