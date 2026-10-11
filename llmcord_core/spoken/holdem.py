"""Hold'em adapter for spoken moves: which kinds are legal now, reading chat, and resolving to engine moves."""
import json
from functools import lru_cache
from pathlib import Path

from ..games import holdem
from . import words
from .moves import Reading, Resolution

_DIR = Path(__file__).parent / "lexicon"


@lru_cache(maxsize=1)
def _lexicons():
    load = lambda name: json.loads((_DIR / name).read_text(encoding="utf-8"))  # noqa: E731
    return load("holdem.json"), load("common.json")["negations"]


def _raises(legal):
    return [int(m[6:]) for m in legal if m.startswith("raise:")]


def kinds_for_view(v):
    legal = holdem.legal_moves_for_view(v)
    kinds = [k for k, ok in (
        ("fold", "fold" in legal), ("check", "check" in legal), ("call", "call" in legal or "check" in legal),
        ("raise", bool(_raises(legal))), ("all_in", bool(_raises(legal)) or "call" in legal)) if ok]
    return tuple(kinds)


def read_holdem(text, v):
    lexicon, negations = _lexicons()
    r = words.read(text, lexicon, kinds_for_view(v), negations)
    if r and any(c.kind == "raise" and c.amount is not None and (c.mode is None or c.preset) for c in r.candidates):
        return Reading(r.candidates, False)
    return r


def _targets(move, v, raises):
    if move.preset == "min" and move.amount is None:
        return [min(raises)]
    if move.amount is not None:
        to, by = move.amount, v.level + move.amount
        return {"to": [to], "by": [by]}.get(move.mode, [to, by])
    if not move.preset:
        return []
    full = v.pot + max(0, v.level - v.bets[v.seat])
    return [v.level + (full // 2 if move.preset == "half_pot" else full)]


def resolve(move, v):
    legal = holdem.legal_moves_for_view(v)
    if move.kind in ("fold", "check"):
        return Resolution((move.kind,) if move.kind in legal else ())
    if move.kind == "call":
        return Resolution(tuple(m for m in ("call", "check") if m in legal)[:1])
    raises = _raises(legal)
    top = v.bets[v.seat] + v.stacks[v.seat] if v.seat is not None else 0
    if move.kind == "all_in":
        return Resolution((f"raise:{top}",) if top in raises else ("call",) if "call" in legal else ())
    if move.kind != "raise" or not raises:
        return Resolution(())
    ok = [t for t in dict.fromkeys(_targets(move, v, raises)) if t == top or v.min_raise_to <= t <= top]
    return Resolution(tuple(f"raise:{t}" for t in ok), not ok)
