"""Spending-cap period math and state; pure, no Discord."""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime, timezone


def _today(now: float) -> date:
    return datetime.fromtimestamp(now, timezone.utc).date()


def _add_month(d: date) -> date:
    return date(d.year + (d.month == 12), d.month % 12 + 1, d.day)


def period_start(now: float, reset_day: int) -> date:
    today = _today(now)
    if today.day >= reset_day:
        return today.replace(day=reset_day)
    return date(today.year - (today.month == 1), (today.month - 2) % 12 + 1, reset_day)


def next_reset(now: float, reset_day: int) -> date:
    return _add_month(period_start(now, reset_day))


def period_key(start: date) -> str:
    return start.isoformat()


@dataclass(frozen=True)
class BudgetState:
    soft_cap_usd: float | None
    hard_cap_usd: float | None
    reset_day: int
    channel_notice: bool
    spent_usd: float
    unpriced_calls: int
    period: str
    resets_on: date
    revision: int

    @property
    def soft_reached(self) -> bool:
        return self.soft_cap_usd is not None and self.spent_usd >= self.soft_cap_usd

    @property
    def hard_reached(self) -> bool:
        return self.hard_cap_usd is not None and self.spent_usd >= self.hard_cap_usd


class BudgetExceeded(Exception):
    def __init__(self, state: BudgetState):
        super().__init__('Spending hard cap reached')
        self.state = state


_admitted: ContextVar[bool] = ContextVar('budget_admitted', default=False)


def admitted() -> bool:
    return _admitted.get()


@contextmanager
def admit():
    """Pass for a started turn: model calls in this context (and tasks created in it) skip the hard-cap gate."""
    token = _admitted.set(True)
    try:
        yield
    finally:
        _admitted.reset(token)


def state(store, now: float | None = None) -> BudgetState:
    now = time.time() if now is None else now
    s = store.budget_settings()
    start = period_start(now, s['reset_day'])
    end = _add_month(start)
    spent, unpriced = store.period_spend(start, end)
    return BudgetState(s['soft_cap_usd'], s['hard_cap_usd'], s['reset_day'], bool(s['channel_notice']),
                       spent, unpriced, period_key(start), end, s['revision'])
