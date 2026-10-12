"""I Doubt It spoken reader (D36): a word list for doubt only, no model; table talk stays silent."""
import json
import re
from functools import lru_cache
from pathlib import Path

from . import words

_DIR = Path(__file__).parent / "lexicon"
_KO_NOT = re.compile(r"아(?:니|님|냐|닌)|지\s?마")


@lru_cache(maxsize=1)
def _lexicons():
    load = lambda name: json.loads((_DIR / name).read_text(encoding="utf-8"))  # noqa: E731
    lex = load("doubt.json")
    return lex["doubt"], [words.normalize(x) for x in lex["_exclude"]], load("common.json")["negations"]


def _has(text, phrase):
    """Exclude phrase present, word-bounded on its ASCII ends."""
    pre = r"(?<![a-z0-9])" if phrase[0].isascii() else ""
    post = r"(?![a-z0-9])" if phrase[-1].isascii() else ""
    return re.search(pre + re.escape(phrase) + post, text) is not None


def read_doubt(text):
    """'sure' for a doubt word, 'ask' when it is phrased as a question, None for table talk, negations, excludes."""
    lex, exclude, negations = _lexicons()
    text = words.normalize(text)
    if any(_has(text, x) for x in exclude):
        return None
    hits = words._found(text, lex, exclude)
    if not hits:
        return None
    start, length = min(hits)
    # Korean negations (안, 말, 못) match anywhere in the text on purpose: it fails safe, a doubt is never called by mistake.
    if words.negated(text, negations) or _KO_NOT.search(text[start + length:]):
        return None
    return "ask" if "?" in text or "？" in text else "sure"
