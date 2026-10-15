"""Vault NPC advice facts (FEAT-65, D43). Pure: public constraints in, a recommended guess out; no secret, model or translated text.

A model may only word these facts; ``check_wording`` rejects anything that names another code, makes a digit claim or spells a number out, and the
caller then uses ``advice_template`` (data, translated at the Discord edge).
"""
import re
from dataclasses import dataclass

from .constraints import filter_candidates
from .solver import suggest_guess, weight_from_style

ADVICE_TEMPLATE_KEY = "vault_advice_template"
_RUNS = re.compile(r"\d+")
_EN_NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen "
                    "twenty thirty forty fifty sixty seventy eighty ninety hundred").split()
_SPELLED = re.compile(r"\b(?:" + "|".join(_EN_NUMBER_WORDS) + r")\b|(?<![가-힣])(?:(?:하나|둘|셋|넷|다섯|여섯|일곱|여덟|아홉|스물)(?:가|는|은|를|을|의|이|도|만|로|에|와|과)?|열)(?![가-힣])",
                      re.IGNORECASE)


def spells_number(text) -> bool:
    """True if the text spells a number out as a word (English digit words, tens and teens; Korean native numerals), which a digit check cannot see."""
    return isinstance(text, str) and _SPELLED.search(text) is not None


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
    if not isinstance(text, str) or facts.guess not in text or spells_number(text):
        return False
    count = str(facts.remaining)
    allowed = {facts.guess} | ({count} if len(count) < len(facts.guess) else set())
    return all(run in allowed for run in _RUNS.findall(text))


def advice_template(facts):
    return ADVICE_TEMPLATE_KEY, {"guess": facts.guess, "remaining": facts.remaining}
