"""Vault (금고): a Bulls and Cows game engine (FEAT-62, D43). Pure Python, no Discord, database, model or network."""
from .advice import ADVICE_TEMPLATE_KEY, AdviceFacts, advice_template, build_advice_facts, check_wording
from .clues import make_clues
from .constraints import (
    CommonCount,
    CountEven,
    CountLow,
    DigitAt,
    DigitPresent,
    GuessResult,
    NotSecret,
    OpenDigit,
    PairCount,
    PositionHalf,
    PositionParity,
    filter_candidates,
    played,
)
from .rules import (
    DEFAULT_LENGTH,
    DEFAULT_TURNS,
    LENGTHS,
    PROTOCOLS,
    VaultError,
    all_secrets,
    check_guess,
    evaluate,
    feedback,
    is_win,
    open_digit,
    random_secret,
    valid_code,
)
from .missions import (
    BULLS_COWS_LENGTH,
    BULLS_COWS_TURNS,
    FORENSICS_ATTEMPTS,
    MISSION_TYPES,
    ForensicsPuzzle,
    bulls_cows_feedback,
    bulls_cows_secret,
    bulls_cows_secrets,
    check_bulls_cows_guess,
    forensics_answers,
    generate_forensics,
    valid_forensics_answer,
)
from .solver import Suggestion, rank_guesses, suggest_guess, weight_from_style
from .tools import (
    TOOLS,
    ToolResult,
    ToolSpec,
    cross_validation,
    cross_validation_common,
    evaluate_tool,
    is_redundant,
    tool_fits,
)

__all__ = [
    "ADVICE_TEMPLATE_KEY", "AdviceFacts", "advice_template", "build_advice_facts", "check_wording",
    "DEFAULT_LENGTH", "DEFAULT_TURNS", "LENGTHS", "PROTOCOLS", "TOOLS",
    "CommonCount", "CountEven", "CountLow", "DigitAt", "DigitPresent", "GuessResult", "NotSecret", "OpenDigit", "PairCount", "PositionHalf",
    "PositionParity", "Suggestion", "ToolResult", "ToolSpec", "VaultError",
    "BULLS_COWS_LENGTH", "BULLS_COWS_TURNS", "FORENSICS_ATTEMPTS", "MISSION_TYPES", "ForensicsPuzzle", "bulls_cows_feedback", "bulls_cows_secret",
    "bulls_cows_secrets", "check_bulls_cows_guess", "forensics_answers", "generate_forensics", "valid_forensics_answer",
    "all_secrets", "check_guess", "cross_validation", "cross_validation_common", "evaluate", "evaluate_tool",
    "feedback", "filter_candidates", "is_redundant", "is_win", "make_clues", "open_digit", "played", "random_secret",
    "rank_guesses", "suggest_guess", "tool_fits", "valid_code", "weight_from_style",
]
