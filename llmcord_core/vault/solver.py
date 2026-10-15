"""Vault solver: ranks guesses on public constraints only (FEAT-62, D43).

``rank_guesses`` scores guesses by how they split the remaining candidates; ``suggest_guess`` blends that with the
chance of winning outright from Play style: weight 0 is minimax (information), weight 1 only guesses a candidate (win).
Nothing here takes the secret. Callers must record every guess with ``constraints.played``
(a wrong guess adds ``NotSecret``), or Total Match keeps recommending a guess already known
to be wrong. When candidates exceed ``SCORE_CAP`` they are scored on a reproducible sample
and the sizes scaled back up, which keeps 5- and 6-digit games fast on a Raspberry Pi.
"""
import random
from dataclasses import dataclass
from operator import eq

from .rules import all_secrets

SCORE_CAP = 2000
MAX_ERRATIC = 0.35  # mirrors games/playstyle.MAX_ERRATIC (that module pulls in i18n, so it is not imported here)


@dataclass(frozen=True)
class Suggestion:
    guess: str
    in_candidates: bool
    worst_case: int
    expected_remaining: float


def _pool(candidates: list[str], length: int, limit: int, rng: random.Random) -> list[str]:
    if len(candidates) <= limit:
        return list(candidates)
    take = min(len(candidates), limit * 2 // 3)
    pool = rng.sample(candidates, take)
    seen, members = set(pool), set(candidates)
    space = all_secrets(length)
    for _ in range(limit * 3):
        if len(pool) >= limit:
            break
        pick = rng.choice(space)
        if pick not in seen and pick not in members:
            seen.add(pick)
            pool.append(pick)
    for pick in rng.sample(candidates, len(candidates)):
        if len(pool) >= limit:
            break
        if pick not in seen:
            seen.add(pick)
            pool.append(pick)
    return pool


def rank_guesses(candidates, protocol: str, length: int, pool_limit: int = 300, rng: random.Random | None = None):
    candidates = list(candidates)
    if not candidates:
        return []
    rng = rng or random.Random(0)
    pool = _pool(candidates, length, pool_limit, rng)
    n = len(candidates)
    scored = candidates if n <= SCORE_CAP else rng.sample(candidates, SCORE_CAP)
    scale = n / len(scored)
    info = [(c, frozenset(c)) for c in scored]
    members = set(candidates)
    out = []
    for guess in pool:
        gset, groups = frozenset(guess), {}
        for cand, cset in info:
            shared = len(gset & cset)
            if protocol == "total_match":
                key = shared
            else:
                strikes = sum(map(eq, guess, cand))
                key = strikes if protocol == "position_only" else strikes * 16 + shared
            groups[key] = groups.get(key, 0) + 1
        sizes = groups.values()
        out.append(Suggestion(
            guess, guess in members, round(max(sizes) * scale) if scale != 1 else max(sizes),
            sum(s * s for s in sizes) * scale / len(scored),
        ))
    out.sort(key=lambda s: (s.worst_case, s.expected_remaining, not s.in_candidates, s.guess))
    return out


def weight_from_style(risk: float, aggression: float) -> float:
    """Win-versus-information weight in [0, 1] from the playstyle traits (0..100)."""
    return ((risk + aggression) / 2) / 100


def suggest_guess(
    candidates, protocol: str, length: int, weight: float, unpredictability: float = 0,
    seed: str = "", pool_limit: int = 300,
) -> str:
    """One guess. Weight 0 is minimax (worst case, then expected remaining); weight 1 is a candidate; in between,
    guesses are ordered by expected remaining plus ``weight * N`` for a non-candidate (N = candidates left), so a
    heavier weight leans on guesses that can win now. ``unpredictability`` (0..100) sometimes takes the 2nd or 3rd
    best instead, with a seeded roll capped at MAX_ERRATIC. Deterministic for the same arguments."""
    candidates = list(candidates)
    if len(candidates) == 1:
        return candidates[0]
    rng = random.Random(f"vault-pool/{length}/{len(candidates)}/{candidates[0]}")
    ranked = rank_guesses(candidates, protocol, length, pool_limit, rng)
    n = len(candidates)
    if weight >= 1:
        ranked = [s for s in ranked if s.in_candidates] or ranked
        order = ranked
    elif weight <= 0:
        order = ranked
    else:
        order = sorted(ranked, key=lambda s: (
            s.expected_remaining + (0 if s.in_candidates else weight * n), s.worst_case, s.guess))
    if unpredictability > 0 and len(order) > 1:
        roll = random.Random(f"vault-suggest/{seed}")
        if roll.random() < MAX_ERRATIC * min(unpredictability, 100) / 100:
            return roll.choice(order[1:3]).guess
    return order[0].guess
