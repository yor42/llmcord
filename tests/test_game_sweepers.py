"""Characterization (MNT-54): the table sweeper of the Blackjack and I Doubt It adapters.

Pins what ``GameTables`` and ``DoubtTables`` do today so the sweeper loop can move into the shared table runtime. The
adapters' ``sweep`` is replaced on the instance with a recorder, so no store is needed. The migration surface, i.e.
everything below that a refactor may have to touch, is confined to the two adapter wrappers and these helpers:

- ``sweeper`` (the task attribute), ``_ensure_sweeper()`` and ``SWEEP_SECONDS`` (set on the instance before the sweeper starts).
- ``tables`` (a dict the tests fill through ``table(key)`` and empty with ``pop``), plus, for I Doubt It only, ``threads``
  (thread -> table registrations) and ``arm(..., kind, ...)`` timers, which decide when the loop goes idle (``_known``).
- ``closed`` and ``close()``.
- ``new_tasks()``: a snapshot of ``asyncio.all_tasks()`` taken in ``asyncSetUp``; no task names are matched.
"""
import asyncio
import unittest
from types import SimpleNamespace

from llmcord_core.games_discord import GameTables
from llmcord_core.games_doubt_discord import DoubtTables

INTERVAL = 0.01
KEY = (1, 7)


class _Adapter:
    """Uniform driver over one adapter with a recording ``sweep``."""

    failure_text = ""

    def __init__(self, tables):
        self.tables = tables
        self.sweeps = 0
        self.fail_next = 0
        self.baseline: set = set()
        tables.SWEEP_SECONDS = INTERVAL

        async def sweep():
            self.sweeps += 1
            if self.fail_next > 0:
                self.fail_next -= 1
                raise RuntimeError("sweep boom")
        tables.sweep = sweep

    def snapshot(self):
        self.baseline = set(asyncio.all_tasks())

    def new_tasks(self):
        ignored = self.baseline | {asyncio.current_task()}
        return [t for t in asyncio.all_tasks() if t not in ignored and not t.done()]

    def hold(self, key=KEY):
        """Puts a table in memory the way the adapter does, then makes sure the sweeper is started."""
        self.tables.table(key)
        self.tables._ensure_sweeper()

    def release(self, key=KEY):
        self.tables.tables.pop(key, None)


class _Blackjack(_Adapter):
    failure_text = "Blackjack table sweep failed"

    def __init__(self):
        super().__init__(GameTables(SimpleNamespace(store=None)))


class _Doubt(_Adapter):
    failure_text = "I Doubt It table sweep failed"

    def __init__(self):
        super().__init__(DoubtTables(SimpleNamespace(store=None)))


async def until(predicate, within=2.0):
    """Polls for something that should happen; fails the test loudly when it does not within the limit."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + within
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for the expected state")
        await asyncio.sleep(0.005)


async def settle(seconds=INTERVAL * 6):
    """A short fixed wait, only for asserting that nothing more happens."""
    await asyncio.sleep(seconds)


class _SweeperCases:
    make = None

    async def asyncSetUp(self):
        self.a = self.make()
        self.a.snapshot()

    async def asyncTearDown(self):
        await self.a.tables.close()

    async def test_nothing_starts_when_no_table_is_held(self):
        """Characterization (MNT-54): _ensure_sweeper does nothing while no table is in memory."""
        self.a.tables._ensure_sweeper()
        self.assertIsNone(self.a.tables.sweeper)
        self.assertEqual(self.a.new_tasks(), [])

    async def test_nothing_starts_when_closed(self):
        """Characterization (MNT-54): a closed adapter never starts a sweeper, even with a table held."""
        await self.a.tables.close()
        self.a.hold()  # table() plus _ensure_sweeper()
        self.assertIsNone(self.a.tables.sweeper)
        self.assertEqual(self.a.new_tasks(), [])

    async def test_starts_exactly_one_task_and_keeps_a_live_sweeper(self):
        """Characterization (MNT-54): a held table starts one sweeper task; further calls keep that same task."""
        self.a.hold()
        first = self.a.tables.sweeper
        self.assertIsInstance(first, asyncio.Task)
        self.assertEqual(self.a.new_tasks(), [first])
        self.a.tables._ensure_sweeper()
        self.a.hold((1, 8))
        self.assertIs(self.a.tables.sweeper, first)
        self.assertEqual(self.a.new_tasks(), [first])

    async def test_the_loop_calls_sweep_repeatedly(self):
        """Characterization (MNT-54): the sweeper calls sweep again and again, once per interval."""
        self.a.hold()
        await until(lambda: self.a.sweeps >= 3)

    async def test_a_failing_sweep_is_logged_once_per_failure_and_the_loop_goes_on(self):
        """Characterization (MNT-54): each failed sweep logs the adapter's message once; the loop keeps calling sweep."""
        self.a.fail_next = 2
        with self.assertLogs(level="ERROR") as logs:
            self.a.hold()
            await until(lambda: self.a.sweeps >= 4)
        self.assertEqual([r.getMessage() for r in logs.records], [self.a.failure_text] * 2)
        self.assertEqual(self.a.new_tasks(), [self.a.tables.sweeper])

    async def test_the_loop_exits_and_clears_sweeper_when_nothing_is_held(self):
        """Characterization (MNT-54): once no table is held the loop ends by itself and sweeper becomes None."""
        self.a.hold()
        await until(lambda: self.a.sweeps >= 1)
        self.a.release()
        await until(lambda: self.a.tables.sweeper is None)
        await settle()
        self.assertEqual(self.a.new_tasks(), [])

    async def test_a_new_table_after_an_idle_exit_starts_a_new_sweeper(self):
        """Characterization (MNT-54): after the idle exit a later table starts a fresh sweeper task."""
        self.a.hold()
        old = self.a.tables.sweeper
        self.a.release()
        await until(lambda: self.a.tables.sweeper is None)
        self.a.hold((1, 8))
        self.assertIsNotNone(self.a.tables.sweeper)
        self.assertIsNot(self.a.tables.sweeper, old)
        self.assertEqual(self.a.new_tasks(), [self.a.tables.sweeper])

    async def test_the_interval_is_read_again_on_every_iteration(self):
        """Characterization (MNT-54): SWEEP_SECONDS is read each time the loop sleeps, so raising it after the start stops further sweeps."""
        self.a.hold()
        await until(lambda: self.a.sweeps >= 2)
        self.a.tables.SWEEP_SECONDS = 3600
        await settle()  # the sleep already under way may finish once more
        count = self.a.sweeps
        await settle()
        self.assertEqual(self.a.sweeps, count)
        self.assertIsNotNone(self.a.tables.sweeper)

    async def test_close_cancels_the_sweeper_and_leaves_no_task(self):
        """Characterization (MNT-54): close() cancels the sweeper, so no new task is left and sweep stops being called."""
        self.a.hold()
        sweeper = self.a.tables.sweeper
        await until(lambda: self.a.sweeps >= 1)
        await self.a.tables.close()
        self.assertTrue(sweeper.cancelled())
        self.assertEqual(self.a.new_tasks(), [])
        count = self.a.sweeps
        await settle()
        self.assertEqual(self.a.sweeps, count)


class BlackjackSweeperTests(_SweeperCases, unittest.IsolatedAsyncioTestCase):
    make = staticmethod(_Blackjack)

    async def test_table_starts_the_sweeper_by_itself(self):
        """Characterization (MNT-54): Blackjack's table() starts the sweeper without an explicit _ensure_sweeper call."""
        self.a.tables.table(KEY)
        self.assertIsInstance(self.a.tables.sweeper, asyncio.Task)

    async def test_close_resets_the_sweeper_attribute_to_none(self):
        """Characterization (MNT-54): Blackjack's close() sets sweeper back to None."""
        self.a.hold()
        await self.a.tables.close()
        self.assertIsNone(self.a.tables.sweeper)


class DoubtSweeperTests(_SweeperCases, unittest.IsolatedAsyncioTestCase):
    make = staticmethod(_Doubt)

    async def test_table_does_not_start_the_sweeper_by_itself(self):
        """Characterization (MNT-54): I Doubt It's table() does not start the sweeper; start_command calls _ensure_sweeper."""
        self.a.tables.table(KEY)
        self.assertIsNone(self.a.tables.sweeper)
        self.assertEqual(self.a.new_tasks(), [])

    async def test_close_leaves_the_cancelled_sweeper_attribute_set(self):
        """Characterization (MNT-54): I Doubt It's close() cancels the sweeper but does not reset the attribute (unlike Blackjack)."""
        self.a.hold()
        sweeper = self.a.tables.sweeper
        await self.a.tables.close()
        self.assertIs(self.a.tables.sweeper, sweeper)
        self.assertTrue(sweeper.done())

    async def test_a_thread_registration_keeps_the_loop_alive_without_a_table(self):
        """Characterization (MNT-54): with the table gone from memory, a thread registration still counts as held, so the loop goes on until it is dropped."""
        self.a.hold()
        self.a.tables.threads[555] = KEY
        self.a.release()
        mark = self.a.sweeps
        await until(lambda: self.a.sweeps >= mark + 2)
        self.assertIsNotNone(self.a.tables.sweeper)
        self.a.tables.threads.clear()
        await until(lambda: self.a.tables.sweeper is None)

    async def test_a_play_timer_keeps_the_loop_alive_but_an_archive_timer_does_not(self):
        """Characterization (MNT-54): a non-archive timer counts as held; a table with only an archive timer lets the loop go idle."""
        self.a.hold()
        self.a.tables.arm(KEY, "turn", 3600, 1, 0)
        self.a.release()
        mark = self.a.sweeps
        await until(lambda: self.a.sweeps >= mark + 2)
        self.assertIsNotNone(self.a.tables.sweeper)
        self.a.tables.runtime.disarm(KEY, "turn")
        self.a.tables.arm(KEY, "archive", 3600)
        await until(lambda: self.a.tables.sweeper is None)


class SweeperOutsideLoopTests(unittest.TestCase):
    def test_blackjack_table_outside_a_running_loop_starts_nothing(self):
        """Characterization (MNT-54): without a running event loop _ensure_sweeper is a no-op (Blackjack's table() calls it)."""
        a = _Blackjack()
        a.tables.table(KEY)
        self.assertIsNone(a.tables.sweeper)

    def test_doubt_ensure_outside_a_running_loop_starts_nothing(self):
        """Characterization (MNT-54): without a running event loop _ensure_sweeper is a no-op for I Doubt It too."""
        a = _Doubt()
        a.tables.table(KEY)
        a.tables._ensure_sweeper()
        self.assertIsNone(a.tables.sweeper)
