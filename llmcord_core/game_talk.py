"""Daily game-character model-call budget (MNT-59): one failed-attempt counter per (guild, server day) shared by every table game.

Failed attempts leave no usage row, so they are counted here and added to ``game_talk_calls_today`` against ``game_talk_daily_limit``.
"""
from __future__ import annotations

import time


class TalkBudget:
    def __init__(self, store):
        self.store = store  # a Store or a zero-argument callable returning one
        self.failed: dict[tuple[int, object], int] = {}

    def _store(self):
        return self.store() if callable(self.store) else self.store

    def marker(self, guild_id) -> tuple[int, object]:
        marker = (guild_id, self._store()._server_day(guild_id, time.time())[1])
        for old in [m for m in self.failed if m != marker]:
            del self.failed[old]
        return marker

    def used(self, guild_id) -> int:
        return self._store().game_talk_calls_today(guild_id) + self.failed.get(self.marker(guild_id), 0)

    def remaining(self, guild_id) -> int:
        return max(self._store().game_talk_daily_limit(guild_id) - self.used(guild_id), 0)

    def note_failure(self, guild_id) -> None:
        marker = self.marker(guild_id)
        self.failed[marker] = self.failed.get(marker, 0) + 1
