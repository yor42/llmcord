"""Vault constraints: public facts a candidate secret must satisfy (FEAT-62, D43).

Each is a frozen dataclass with ``holds(secret)``. Positions are 1-based, digits one-character strings, and the code
length is read from the secret, so every constraint works for 4..6 digits. ``filter_candidates`` narrows the secret
space with them; nothing here ever receives the real secret except to answer ``holds`` in tests and tool effects.
"""
from dataclasses import dataclass

from .rules import all_secrets, feedback


@dataclass(frozen=True)
class GuessResult:
    guess: str
    protocol: str
    result: tuple

    def holds(self, secret: str) -> bool:
        return feedback(self.protocol, secret, self.guess) == self.result


@dataclass(frozen=True)
class DigitPresent:
    digit: str
    present: bool

    def holds(self, secret: str) -> bool:
        return (self.digit in secret) == self.present


@dataclass(frozen=True)
class DigitAt:
    position: int
    digit: str

    def holds(self, secret: str) -> bool:
        return secret[self.position - 1] == self.digit


@dataclass(frozen=True)
class PositionHalf:
    position: int
    high: bool  # low is 0-4, high is 5-9

    def holds(self, secret: str) -> bool:
        return (secret[self.position - 1] >= "5") == self.high


@dataclass(frozen=True)
class PairCount:
    a: str
    b: str
    count: int  # how many of the two digits the secret contains

    def holds(self, secret: str) -> bool:
        return (self.a in secret) + (self.b in secret) == self.count


@dataclass(frozen=True)
class CommonCount:
    digits: str  # the digits two guesses share, sorted (cross-validation, FEAT-66)
    count: int  # how many of them the secret contains

    def holds(self, secret: str) -> bool:
        return len(set(self.digits) & set(secret)) == self.count


@dataclass(frozen=True)
class CountEven:
    count: int

    def holds(self, secret: str) -> bool:
        return sum(int(c) % 2 == 0 for c in secret) == self.count


@dataclass(frozen=True)
class CountLow:
    count: int  # digits 0-4

    def holds(self, secret: str) -> bool:
        return sum(c <= "4" for c in secret) == self.count


@dataclass(frozen=True)
class PositionParity:
    position: int
    odd: bool

    def holds(self, secret: str) -> bool:
        return (int(secret[self.position - 1]) % 2 == 1) == self.odd


@dataclass(frozen=True)
class NotSecret:
    guess: str  # a wrong guess is not the secret; GuessResult alone keeps it under Total Match (its own feedback)

    def holds(self, secret: str) -> bool:
        return secret != self.guess


@dataclass(frozen=True)
class OpenDigit:
    digit: str

    def holds(self, secret: str) -> bool:
        return self.digit in secret


def played(protocol: str, guess: str, result: tuple, won: bool = False) -> list:
    """The public constraints a played guess adds. Always record guesses through this: a wrong guess also adds
    NotSecret, otherwise Total Match keeps re-recommending a guess already known to be wrong (Position-Only
    feedback already excludes it), so callers should always use this rather than building constraints by hand."""
    cons = [GuessResult(guess, protocol, result)]
    return cons if won else cons + [NotSecret(guess)]


def filter_candidates(constraints, length: int = 4, base: list[str] | None = None) -> list[str]:
    """Secrets satisfying every constraint, lexicographic. ``base`` is a list already filtered, narrowed further."""
    left = all_secrets(length) if base is None else base
    for con in constraints:
        left = [s for s in left if con.holds(s)]
    return list(left)
