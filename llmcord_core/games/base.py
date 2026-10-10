"""Shared pieces for the minigame rules engines: seats, errors, seeds, the stable shuffle, choosers.

Pure Python: no Discord, database, model or clock. Randomness comes only from a seed string.
"""
import hashlib
import secrets
from dataclasses import dataclass
from typing import Protocol, Sequence

SEAT_KINDS = ("member", "character")


class IllegalMove(Exception):
    """A move that the rules do not allow right now (wrong seat, wrong phase, not legal)."""


@dataclass(frozen=True)
class Seat:
    index: int
    kind: str  # 'member' | 'character'
    ref: int  # member id or character id
    stake: int

    def __post_init__(self):
        if self.kind not in SEAT_KINDS:
            raise ValueError(f"seat kind must be one of {SEAT_KINDS}")
        if type(self.stake) is not int or self.stake < 0:
            raise ValueError("seat stake must be a non-negative int")


class Chooser(Protocol):
    """Picks one move for a seat from what that seat may see. Must return an element of legal_moves."""

    async def choose(self, view, legal_moves: Sequence[str]) -> str: ...


class DefaultPolicy:
    """Deterministic fallback (timeouts, bad model output): decline insurance, hit below 17, else stand; never double or surrender."""

    async def choose(self, view, legal_moves):
        return self.pick(view, legal_moves)

    @staticmethod
    def pick(view, legal_moves):
        legal = tuple(legal_moves)
        if not legal:
            raise IllegalMove("no legal moves")
        if "no_insurance" in legal:
            return "no_insurance"
        total = getattr(view, "total", None)
        if "hit" in legal and total is not None and total < 17:
            return "hit"
        if "stand" in legal:
            return "stand"
        return next((m for m in legal if m not in ("double", "surrender", "insure", "even_money")), legal[0])


def new_seed():
    return secrets.token_hex(16)


def seed_hash(seed):
    return hashlib.sha256(seed.encode()).hexdigest()


def short_seed_hash(seed):
    return seed_hash(seed)[:12]


def _randoms(seed):
    """Endless stream of uniform 32-bit integers derived from sha256(seed, counter)."""
    base = seed.encode()
    counter = 0
    while True:
        block = hashlib.sha256(base + b"/" + counter.to_bytes(8, "big")).digest()
        for i in range(0, 32, 4):
            yield int.from_bytes(block[i:i + 4], "big")
        counter += 1


def shuffled(items, seed):
    """Fisher-Yates driven by sha256 with rejection sampling: identical on every Python/platform."""
    out = list(items)
    stream = _randoms(seed)
    for i in range(len(out) - 1, 0, -1):
        n = i + 1
        limit = (1 << 32) - ((1 << 32) % n)
        r = next(stream)
        while r >= limit:
            r = next(stream)
        j = r % n
        out[i], out[j] = out[j], out[i]
    return out
