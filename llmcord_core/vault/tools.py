"""Vault tools: five one-shot effects that each reveal a public fact about the secret (FEAT-62, D43).

``evaluate_tool`` returns a ``ToolResult`` whose ``constraint`` joins the public record, a ``text_key`` for the edge to
word, and ``data`` for the words. Params are plain dicts (1-based int positions, one-character string digits).
``history`` is the past ``GuessResult`` entries plus previously disclosed ``DigitAt`` constraints. Invalid use raises
``VaultError``. The caller owns the turn rule (``tool_fits``) and spends the item; this module has no state.
"""
from dataclasses import dataclass, field

from .constraints import DigitAt, DigitPresent, GuessResult, PairCount, PositionHalf
from .rules import DIGITS, VaultError, feedback


@dataclass(frozen=True)
class ToolSpec:
    key: str
    turn_cost: int
    needs: tuple[str, ...]  # param names


@dataclass(frozen=True)
class ToolResult:
    constraint: object
    text_key: str
    data: dict = field(default_factory=dict)


TOOLS = {
    spec.key: spec
    for spec in (
        ToolSpec("vault_residual_scanner", 1, ("guess", "digit")),
        ToolSpec("vault_spectrum_analyzer", 1, ("position",)),
        ToolSpec("vault_comparison_circuit", 1, ("a", "b")),
        ToolSpec("vault_position_verifier", 1, ("guess",)),
        ToolSpec("vault_master_decoder", 2, ("position",)),
    )
}


def tool_fits(key: str, turns_left: int) -> bool:
    """A tool must fit the budget; the decoder costs two and must leave one turn for a final guess."""
    if key not in TOOLS:
        raise VaultError("bad_tool", key=key)
    return turns_left >= TOOLS[key].turn_cost + (1 if key == "vault_master_decoder" else 0)


def _position(params: dict, length: int) -> int:
    pos = params.get("position")
    if isinstance(pos, bool) or not isinstance(pos, int) or not 1 <= pos <= length:
        raise VaultError("bad_position", length=length)
    return pos


def _digit(value) -> str:
    if not isinstance(value, str) or len(value) != 1 or value not in DIGITS:
        raise VaultError("bad_digit")
    return value


def _past_guess(params: dict, history) -> GuessResult:
    guess = params.get("guess")
    for item in history:
        if isinstance(item, GuessResult) and item.guess == guess:
            return item
    raise VaultError("guess_not_in_history")


def _disclosed(history) -> set[int]:
    return {h.position for h in history if isinstance(h, DigitAt)}


def evaluate_tool(secret: str, key: str, params: dict, history) -> ToolResult:
    if key == "vault_residual_scanner":
        digit = _digit(params.get("digit"))
        past = _past_guess(params, history)
        if digit not in past.guess:
            raise VaultError("digit_not_in_guess")
        present = digit in secret
        return ToolResult(
            DigitPresent(digit, present),
            "vault_scanner_present" if present else "vault_scanner_absent",
            {"guess": past.guess, "digit": digit, "present": present},
        )
    if key == "vault_spectrum_analyzer":
        pos = _position(params, len(secret))
        high = secret[pos - 1] >= "5"
        return ToolResult(
            PositionHalf(pos, high), "vault_spectrum_high" if high else "vault_spectrum_low",
            {"position": pos, "high": high},
        )
    if key == "vault_comparison_circuit":
        a, b = _digit(params.get("a")), _digit(params.get("b"))
        if a == b:
            raise VaultError("digits_equal")
        count = (a in secret) + (b in secret)
        return ToolResult(PairCount(a, b, count), f"vault_comparison_{count}", {"a": a, "b": b, "count": count})
    if key == "vault_position_verifier":
        past = _past_guess(params, history)
        strikes = [i + 1 for i, (s, g) in enumerate(zip(secret, past.guess)) if s == g]
        if not strikes:
            return ToolResult(
                GuessResult(past.guess, "position_only", (0,)), "vault_verifier_none",
                {"guess": past.guess, "position": None, "digit": None},
            )
        fresh = [p for p in strikes if p not in _disclosed(history)]
        if not fresh:
            raise VaultError("nothing_new")
        pos = fresh[0]
        return ToolResult(
            DigitAt(pos, secret[pos - 1]), "vault_verifier_position",
            {"guess": past.guess, "position": pos, "digit": secret[pos - 1]},
        )
    if key == "vault_master_decoder":
        pos = _position(params, len(secret))
        if pos in _disclosed(history):
            raise VaultError("already_revealed", position=pos)
        return ToolResult(
            DigitAt(pos, secret[pos - 1]), "vault_decoder_digit", {"position": pos, "digit": secret[pos - 1]}
        )
    raise VaultError("unknown_tool", key=key)


def is_redundant(key: str, params: dict, candidates, history=()) -> bool:
    """True when every candidate gives the same answer, or every candidate refuses the query (say, a position
    already revealed in ``history``), so asking teaches nothing (public data only)."""
    guess = params.get("guess")
    answers, refused = set(), 0
    for cand in candidates:
        hist = list(history)
        if isinstance(guess, str) and not any(isinstance(h, GuessResult) and h.guess == guess for h in hist):
            hist.append(GuessResult(guess, "standard", feedback("standard", cand, guess)))
        try:
            answers.add(evaluate_tool(cand, key, params, hist).constraint)
        except VaultError:
            refused += 1
            continue
        if len(answers) > 1:
            return False
    return len(answers) <= 1 and (not answers or not refused)


def cross_validation_common(guess_a: str, guess_b: str) -> frozenset:
    """Digits that two guesses share, whatever the secret."""
    return frozenset(set(guess_a) & set(guess_b))


def cross_validation(secret: str, guess_a: str, guess_b: str) -> int:
    """How many of the digits two guesses share are in the secret."""
    return len(cross_validation_common(guess_a, guess_b) & set(secret))
