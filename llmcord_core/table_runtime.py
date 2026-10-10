"""Shared table runtime (MNT-50): per-table locks and one-shot timers with a single retry, cancelled on close.

Knows no game, kind names, store or Discord concepts: an adapter arms ``(table_key, kind)`` timers with its own hook.
"""
from __future__ import annotations

import asyncio
import logging
import weakref


class TableRuntime:
    def __init__(self, *, label: str, sleep=None, retry_seconds: float = 5.0):
        self.label, self.sleep, self.retry_seconds = label, sleep, retry_seconds
        self.closed = False
        self.timers: dict[tuple, asyncio.Task] = {}
        self.sweeper: asyncio.Task | None = None
        self.locks: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

    async def _sleep(self, seconds) -> None:
        await (self.sleep or asyncio.sleep)(seconds)  # resolved per call, so patching asyncio.sleep still reaches timers

    def lock(self, key) -> asyncio.Lock:
        lock = self.locks.get(key)
        if lock is None:
            lock = self.locks[key] = asyncio.Lock()
        return lock

    def arm(self, key, kind, delay, hook, *args, name: str | None = None) -> None:
        if self.closed:
            return
        self.disarm(key, kind)
        slot = (key, kind)
        task = self.timers[slot] = asyncio.create_task(self._timer(key, kind, delay, hook, args), name=name or f"{self.label}-{kind}-{key[1]}".lower())
        task.add_done_callback(lambda done: self.timers.get(slot) is done and self.timers.pop(slot, None))

    def disarm(self, key, kind) -> None:
        old = self.timers.pop((key, kind), None)
        if old is not None and old is not asyncio.current_task():
            old.cancel()

    def current(self, key, kind) -> bool:
        return not self.closed and self.timers.get((key, kind)) is asyncio.current_task()

    async def _timer(self, key, kind, delay, hook, args) -> None:
        await self._sleep(delay)
        for attempt in (1, 2):  # one retry, so held bets do not wait for a restart after a transient failure
            try:
                await hook(*args)
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("%s %s timer failed (table %s, attempt %d)", self.label, kind, key[1], attempt)
                if attempt == 1:
                    if not self.current(key, kind):  # replaced, disarmed or closed: the retry would be untracked
                        return
                    await self._sleep(self.retry_seconds)  # still the table's timer, so disarm and close() cancel it
                    if not self.current(key, kind):
                        return

    def ensure_sweeper(self, *, interval, held, sweep, name: str) -> None:
        """Starts the periodic sweep unless closed, already running or without a loop. It ends itself once ``held()`` is false."""
        if self.closed or (self.sweeper is not None and not self.sweeper.done()):
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self.sweeper = asyncio.create_task(self._sweep_loop(interval, held, sweep), name=name)

    async def _sweep_loop(self, interval, held, sweep) -> None:
        while True:
            await self._sleep(interval())
            if not held():  # nothing in memory: stay idle until the adapter starts a sweeper again
                self.sweeper = None
                return
            try:
                await sweep()
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("%s table sweep failed", self.label)

    async def close(self) -> None:
        self.closed = True
        tasks = list(self.timers.values())
        if self.sweeper is not None:
            tasks.append(self.sweeper)
        self.timers.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
