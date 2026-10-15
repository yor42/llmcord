"""Vault partner clues: true public facts about the secret that never give an exact position digit (FEAT-62, D43).

Templates are CountEven, CountLow, DigitPresent, PositionParity and PositionHalf. A clue is skipped when it repeats a
``taken`` or earlier clue or narrows nothing beyond them, so the set is consistent and every clue is worth giving.
Deterministic for a given rng seed.
"""
import random

from .constraints import CountEven, CountLow, DigitPresent, PositionHalf, PositionParity, filter_candidates
from .rules import DIGITS, all_secrets


def _possible(secret: str) -> list:
    n = len(secret)
    out = [CountEven(sum(int(c) % 2 == 0 for c in secret)), CountLow(sum(c <= "4" for c in secret))]
    out += [DigitPresent(d, d in secret) for d in DIGITS]
    for pos in range(1, n + 1):
        out.append(PositionParity(pos, int(secret[pos - 1]) % 2 == 1))
        out.append(PositionHalf(pos, secret[pos - 1] >= "5"))
    return out


def make_clues(secret: str, count: int, rng: random.Random, taken=()) -> list:
    pool = _possible(secret)
    rng.shuffle(pool)
    given = list(taken)
    left = filter_candidates(given, len(secret)) if given else list(all_secrets(len(secret)))
    clues = []
    for clue in pool:
        if len(clues) >= count:
            break
        if clue in given:
            continue
        narrowed = [s for s in left if clue.holds(s)]
        if len(narrowed) == len(left):
            continue
        left = narrowed
        given.append(clue)
        clues.append(clue)
    return clues
