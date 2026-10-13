"""In-memory call limits for the spoken-move veto: a token bucket per guild and one daily count for the process. Nothing is stored."""
from __future__ import annotations

import datetime
import time


class VetoLimiter:
    def __init__(self, per_minute=30, per_day=5000, clock=time.monotonic, today=datetime.date.today):
        self.per_minute, self.per_day, self.clock, self.today = per_minute, per_day, clock, today
        self.buckets: dict = {}  # guild id -> (tokens, last refill time)
        self.day, self.used = today(), 0

    def allow(self, guild_id) -> bool:
        """Consume one call from the guild's bucket and the daily count; False when either is spent (nothing is consumed then)."""
        day = self.today()
        if day != self.day:
            self.day, self.used = day, 0
        if self.used >= self.per_day:
            return False
        now = self.clock()
        tokens, last = self.buckets.get(guild_id, (float(self.per_minute), now))
        tokens = min(float(self.per_minute), tokens + max(0.0, now - last) * self.per_minute / 60.0)
        if tokens < 1.0:
            self.buckets[guild_id] = (tokens, now)
            return False
        self.buckets[guild_id] = (tokens - 1.0, now)
        self.used += 1
        return True
