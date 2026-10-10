"""Shared table runtime (MNT-50): per-table locks and one-shot timers with a single retry, cancelled on close.

Knows no game, kind names, store or Discord concepts: an adapter arms ``(table_key, kind)`` timers with its own hook.
"""
from __future__ import annotations

import asyncio
import logging
import weakref


class TableRuntime:
    def __init__(self, *, label: str, sleep=asyncio.sleep, retry_seconds: float = 5.0):
        self.label, self.sleep, self.retry_seconds = label, sleep, retry_seconds
        self.closed = False
        self.timers: dict[tuple, asyncio.Task] = {}
        self.locks: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

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
        await self.sleep(delay)
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
                    await self.sleep(self.retry_seconds)  # still the table's timer, so disarm and close() cancel it
                    if not self.current(key, kind):
                        return

    async def close(self) -> None:
        self.closed = True
        tasks = list(self.timers.values())
        self.timers.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
