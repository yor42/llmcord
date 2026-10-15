"""Vault rules: the secret space, Strike/Ball evaluation and protocol feedback (FEAT-62, D43).

Pure Python: no Discord, database, model, clock, i18n or network. Randomness comes only from a passed
``random.Random`` (``random_secret`` defaults to ``secrets.SystemRandom``). A secret is a string of unique digits,
4 by default and 4..6 supported; leading zeros are valid. The engine raises ``VaultError`` with a lowercase snake
``code`` and params, never user-facing text: the Discord and dashboard edges turn codes into words.
"""
import itertools
import random
import secrets

PROTOCOLS = ("standard", "total_match", "position_only", "open")
DEFAULT_LENGTH = 4
LENGTHS = (4, 5, 6)
DEFAULT_TURNS = {"standard": 9, "total_match": 9, "position_only": 9, "open": 8}
DIGITS = "0123456789"

_SPACES: dict[int, tuple[str, ...]] = {}


class VaultError(ValueError):
    """A rule refusal. ``code`` is a lowercase snake identifier, ``params`` carries the values for the message."""

    def __init__(self, code: str, **params):
        super().__init__(code)
        self.code = code
        self.params = params


def all_secrets(length: int = DEFAULT_LENGTH) -> tuple[str, ...]:
    """Every secret of ``length`` in lexicographic order, built once per length and cached."""
    if length not in LENGTHS:
        raise VaultError("bad_length", length=length)
    space = _SPACES.get(length)
    if space is None:
        space = _SPACES[length] = tuple("".join(p) for p in itertools.permutations(DIGITS, length))
    return space


def valid_code(code: str, length: int = DEFAULT_LENGTH) -> bool:
    return len(code) == length and set(code) <= set(DIGITS) and len(set(code)) == length


def random_secret(length: int = DEFAULT_LENGTH, rng: random.Random | None = None) -> str:
    if length not in LENGTHS:
        raise VaultError("bad_length", length=length)
    return "".join((rng or secrets.SystemRandom()).sample(DIGITS, length))


def evaluate(secret: str, guess: str) -> tuple[int, int]:
    """(strikes, balls): strikes are right digit and place, balls right digit in another place."""
    strikes = sum(a == b for a, b in zip(secret, guess))
    return strikes, len(set(secret) & set(guess)) - strikes


def check_guess(guess: str, length: int) -> str:
    """Return ``guess`` when it is a legal guess of ``length`` unique digits, else raise a VaultError."""
    if len(guess) != length:
        raise VaultError("guess_length", length=length)
    if not set(guess) <= set(DIGITS):
        raise VaultError("guess_not_digits")
    if len(set(guess)) != length:
        raise VaultError("guess_repeated_digit")
    return guess


def feedback(protocol: str, secret: str, guess: str) -> tuple:
    """What a guess reveals under ``protocol``, as a hashable tuple. Open feedback is Standard feedback."""
    strikes, balls = evaluate(secret, guess)
    if protocol == "total_match":
        return (strikes + balls,)
    if protocol == "position_only":
        return (strikes,)
    return strikes, balls


def is_win(secret: str, guess: str) -> bool:
    return secret == guess


def open_digit(secret: str, rng: random.Random) -> str:
    """The Open protocol's starting hint: one digit of the secret, with no position."""
    return rng.choice(secret)
