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
_EN_NOT_ONE = r"(?<!\bthis )(?<!\bthat )(?<!\bno )(?<!\bany )(?<!\bevery )(?<!\bwhich )(?<!\bsome )(?<!\bthe )(?<!\banother )"
_EN = r"\b(?:(?:" + "|".join(w for w in _EN_NUMBER_WORDS if w != "one") + r")\b|" + _EN_NOT_ONE + r"one\b(?!\s+of\b))"
_NATIVE = "하나|둘|셋|넷|다섯|여섯|일곱|여덟|아홉|스물"
_SUFFIXES = "가|는|은|를|을|의|이에요|예요|이야|이다|이고|이라고|이라|이랑|이나|이란|이|야|도|만|으로|로|에|와|과|랑|나|인|부터|까지|째|씩|쯤|뿐|밖에|보다|처럼|라고"
_COUNTERS = "번째|개|명|번|마리|자리|칸|가지|시간|살|장|대|곳|층|잔|권|줄|배|단계|턴"
_HANGUL = "가-힣"
_SINO = "일이삼사오육칠팔구"
_SPELLED = re.compile("|".join((
    _EN,
    rf"(?<![{_HANGUL}])(?:{_NATIVE})(?:(?<!하나)(?:{_SUFFIXES})|(?<=하나)(?!도)(?:{_SUFFIXES}))?(?![{_HANGUL}])(?!\s+다(?![{_HANGUL}]))",   # 하나도, 둘 다 are not counts
    rf"(?<![{_HANGUL}])열(?:(?:{_SUFFIXES})|\s*(?:{_COUNTERS}))?(?![{_HANGUL}])(?=\s*(?:[^\w\s{_HANGUL}]|$)|\s*(?:{_COUNTERS})|(?<=열)(?:{_SUFFIXES}))",   # 열 as a verb ("열 수", "열어") is not
    rf"(?<![{_HANGUL}])(?:한\s+(?:{_COUNTERS})|(?:한|두|세|네|석|넉|스무)\s*(?:{_COUNTERS})(?<!한번))(?![{_HANGUL}])",
    rf"(?<![{_HANGUL}])(?:[{_SINO}]*십[{_SINO}]*|[{_SINO}]{{3,}}|백[{_SINO}십]*)(?:{_SUFFIXES})?(?![{_HANGUL}])",
    rf"(?<![{_HANGUL}])[{_SINO}](?:\s+[{_SINO}])+(?![{_HANGUL}])",
    rf"(?<![{_HANGUL}])[삼육칠팔구]\s*(?:번째|번|개|명|자리)(?![{_HANGUL}])",
)), re.IGNORECASE)


def spells_number(text) -> bool:
    """True if the text spells a number out as a word, which a digit check cannot see. Best effort, not a parser: English number words (but not
    "this one", "no one", "one of"), Korean native numerals with common particles, copula and counters (셋이에요, 둘이야, 한 개, 세 번, 열째), and Sino-Korean
    numerals only as runs (삼십, 삼칠팔, 삼 칠) or with a counter (삼 번), never a bare 이/일/사/오. "열 수", "열어", "둘 다" and "하나도" are let through."""
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
