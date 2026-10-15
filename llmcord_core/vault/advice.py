"""Vault NPC advice facts (FEAT-65, D43). Pure: public constraints in, a recommended guess out; no secret, model or translated text.

A model may only word these facts; ``check_wording`` rejects anything that names another code or makes a digit claim, and the
caller then uses ``advice_template`` (data, translated at the Discord edge).
"""
import re
from dataclasses import dataclass

from .constraints import filter_candidates
from .solver import suggest_guess, weight_from_style

ADVICE_TEMPLATE_KEY = "vault_advice_template"
_RUNS = re.compile(r"\d+")


@dataclass(frozen=True)
class AdviceFacts:
    guess: str
    remaining: int
    weight: float
    in_candidates: bool


def build_advice_facts(constraints, protocol, length, *, risk=50, aggression=50, unpredictability=0, seed=""):
    candidates = filter_candidates(constraints, length)
    weight = weight_from_style(risk, aggression)
    guess = suggest_guess(candidates, protocol, length, weight, unpredictability, seed)
    return AdviceFacts(guess, len(candidates), weight, guess in candidates)


def check_wording(text, facts):
    if not isinstance(text, str) or facts.guess not in text:
        return False
    count = str(facts.remaining)
    allowed = {facts.guess} | ({count} if len(count) < len(facts.guess) else set())
    return all(run in allowed for run in _RUNS.findall(text))


def advice_template(facts):
    return ADVICE_TEMPLATE_KEY, {"guess": facts.guess, "remaining": facts.remaining}
