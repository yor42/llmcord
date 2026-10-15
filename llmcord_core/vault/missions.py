"""Vault support missions: the pure rules (FEAT-67, D43).

Pure Python: no Discord, database, model, clock, i18n or network. Randomness comes only from a passed ``random.Random`` or a seed string.
Mission B is a 3-digit Bulls and Cows game with its own helpers (the main game's ``LENGTHS`` and budgets are untouched). Mission A,
record forensics, shows a few recorded guess/feedback lines of one 4-digit secret with one result hidden; the solver proves the hidden
result is the only one the visible lines allow. Refusals are ``VaultError`` codes, never words.
"""
import itertools
import random
import secrets
from dataclasses import dataclass

from .rules import DIGITS, VaultError, all_secrets, check_guess, evaluate, feedback, random_secret

MISSION_TYPES = ("forensics", "bulls_cows")
BULLS_COWS_LENGTH = 3
BULLS_COWS_TURNS = 6
FORENSICS_LENGTH = 4
FORENSICS_LINES = 4
FORENSICS_ATTEMPTS = 2
FORENSICS_TRIES = 500

_BULLS_COWS: list[tuple[str, ...]] = []


def bulls_cows_secrets() -> tuple[str, ...]:
    """Every secret of 3 unique digits in lexicographic order (leading zeros are valid), built once."""
    if not _BULLS_COWS:
        _BULLS_COWS.append(tuple("".join(p) for p in itertools.permutations(DIGITS, BULLS_COWS_LENGTH)))
    return _BULLS_COWS[0]


def bulls_cows_secret(rng: random.Random | None = None) -> str:
    return (rng or secrets.SystemRandom()).choice(bulls_cows_secrets())


def bulls_cows_feedback(secret: str, guess: str) -> tuple[int, int]:
    return evaluate(secret, guess)


def check_bulls_cows_guess(guess: str) -> str:
    return check_guess(guess, BULLS_COWS_LENGTH)


@dataclass(frozen=True)
class ForensicsPuzzle:
    records: tuple  # (guess, (strikes, balls)) lines; exactly one has None as its result
    hidden: int     # index of that line
    answer: tuple   # the true (strikes, balls) of the hidden line


def forensics_answers(records) -> frozenset:
    """The results the single hidden (None) record can have over the secrets consistent with the visible ones."""
    hidden = [guess for guess, result in records if result is None]
    if len(hidden) != 1:
        raise VaultError("bad_records")
    visible = [(guess, tuple(result)) for guess, result in records if result is not None]
    return frozenset(evaluate(secret, hidden[0]) for secret in all_secrets(FORENSICS_LENGTH)
                     if all(evaluate(secret, guess) == result for guess, result in visible))


def valid_forensics_answer(strikes, balls) -> bool:
    if type(strikes) is not int or type(balls) is not int:
        return False
    return strikes >= 0 and balls >= 0 and strikes + balls <= FORENSICS_LENGTH and (strikes, balls) != (FORENSICS_LENGTH - 1, 1)


def generate_forensics(seed: str) -> ForensicsPuzzle:
    """Deterministic from ``seed`` (about 0.2 s of CPU). Pure, so the Discord layer runs it via asyncio.to_thread and hands the result to the store (store calls themselves stay on the loop: shared sqlite connection). Retries until the hidden result is the only one the visible lines allow."""
    rng = random.Random(seed)
    for _ in range(FORENSICS_TRIES):
        code = random_secret(FORENSICS_LENGTH, rng)
        guesses = []
        while len(guesses) < FORENSICS_LINES:
            guess = random_secret(FORENSICS_LENGTH, rng)
            if guess not in guesses:
                guesses.append(guess)
        hidden = rng.randrange(FORENSICS_LINES)
        answer = feedback("standard", code, guesses[hidden])
        records = tuple((g, None if i == hidden else feedback("standard", code, g)) for i, g in enumerate(guesses))
        if forensics_answers(records) == frozenset({answer}):
            return ForensicsPuzzle(records, hidden, answer)
    raise VaultError("forensics_unsolvable")
