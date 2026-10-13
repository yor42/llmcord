"""FEAT-49c: in-memory, count-only statistics of the decision-model veto, per server and day. Standard library only.

Nothing here knows a message, a user, a response or a key: only a server id, a day and seven counters. They are logged
(operator-facing English, not translated) when the process closes and when a day ends, and a day whose veto rate is exactly
0% or 100% over at least MIN_CALLS readings is flagged as suspicious (a reader that stopped answering sensibly, or moved weights).
"""
import datetime

COUNTERS = ("offered", "vetoed", "silenced", "errors", "pressed_ok", "pressed_vetoed", "unpressed_vetoed")
MIN_CALLS = 20
BAD_COUNTER = "unknown counter"  # internal diagnostic, never shown


def _local_day():
    return datetime.date.today().isoformat()


class VetoStats:
    def __init__(self, clock=None, today=_local_day):
        self.clock = clock  # kept for symmetry with VetoLimiter; nothing here needs it yet
        self.today = today
        self._data: dict[tuple, dict[str, int]] = {}

    def record(self, guild, name, n=1) -> None:
        if name not in COUNTERS:
            raise ValueError(f"{BAD_COUNTER} {name!r}")
        row = self._data.setdefault((guild, self.today()), dict.fromkeys(COUNTERS, 0))
        row[name] += n

    def count(self, guild, day, name) -> int:
        return self._data.get((guild, day), {}).get(name, 0)

    def rate(self, guild, day):
        offered = self.count(guild, day, "offered")
        return self.count(guild, day, "vetoed") / offered if offered else None

    def suspicious(self, guild, day) -> bool:
        return self.count(guild, day, "offered") >= MIN_CALLS and self.rate(guild, day) in (0, 1)

    def snapshot(self) -> dict:
        return {key: dict(row) for key, row in self._data.items()}

    def _line(self, key) -> str:
        guild, day = key
        return f"veto stats guild={guild} day={day} " + " ".join(f"{name}={self._data[key][name]}" for name in COUNTERS)

    def lines(self) -> list:
        return [self._line(key) for key in self._data]

    def rollover(self) -> list:
        """The lines of every finished day, once; those days are then dropped."""
        today = self.today()
        old = [key for key in self._data if key[1] != today]
        out = [self._line(key) for key in old]
        for key in old:
            del self._data[key]
        return out
