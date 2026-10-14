"""One Card spoken reader (D40): a word list for the declaration and the missed-declaration callout, no model."""
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import words

_DIR = Path(__file__).parent / "lexicon"
DEFAULT_PHRASES = ("원카드", "one card")
MAX_LEN = 40
_QUOTES = re.compile(r"\"[^\"]*\"|“[^”]*”|「[^」]*」|(?<!\w)'[^']*'(?!\w)")
_PARTICLES = ("이", "가", "은", "는", "도", "랑", "아", "야", "를", "을", "의", "씨", "님", "한테", "에게")


@dataclass(frozen=True)
class OneCardReading:
    kind: str = "none"
    outcome: str = "none"
    target_kind: str = "none"
    target_seat: int | None = None


@lru_cache(maxsize=1)
def _lexicons():
    lex = json.loads((_DIR / "onecard.json").read_text(encoding="utf-8"))
    rx = lambda key: re.compile("|".join(lex[key]["en"] + lex[key]["ko"]) or "(?!)")  # noqa: E731
    return {k: rx(k) for k in ("callout", "callout_alone", "negation")}, lex["mention"], json.loads((_DIR / "common.json").read_text(encoding="utf-8"))["negations"]


def _mention_rx(mention):
    en = "|".join(rf"(?<![a-z0-9]){x}(?![a-z0-9])" for x in mention["en"])
    return re.compile("|".join(filter(None, [en, "|".join(map(re.escape, mention["ko"]))])))


@lru_cache(maxsize=1)
def _mention():
    return _mention_rx(_lexicons()[1])


def _norm(text):
    return words.normalize(unicodedata.normalize("NFKC", text))


@lru_cache(maxsize=16)
def _phrase_rx(phrases):
    """Source of a pattern for any phrase; spacing between characters is free next to Hangul or where the phrase has a space."""
    out = []
    for p in phrases:
        p = _norm(p)
        if not p.replace(" ", ""):
            continue
        body, prev, space = "", "", False
        for c in p:
            if c == " ":
                space = True
                continue
            if prev and (space or not (c.isascii() and prev.isascii())):
                body += r"\s*"
            body += re.escape(c)
            prev, space = c, False
        pre = r"(?<![a-z0-9])" if p[0].isascii() else ""
        post = r"(?![a-z0-9])" if p[-1].isascii() else ""
        out.append(pre + body + post)
    return "|".join(out) or "(?!)"


def _name_hit(text, name):
    name = _norm(name)
    if not name:
        return False
    if name[0].isascii() and name[-1].isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", text) is not None
    for m in re.finditer(rf"(?<![가-힣a-z0-9]){re.escape(name)}", text):
        rest = text[m.end():]
        if not rest or not "가" <= rest[0] <= "힣" or rest.startswith(_PARTICLES):
            return True
    return False


def read_onecard(text, seats=(), reply_to=None, phrases=None):
    """Reading of one chat message: a declaration, or a callout at a seat that did not declare. Only 'sure' acts."""
    none = OneCardReading()
    raw = unicodedata.normalize("NFKC", text).strip()
    if not raw or len(raw) > MAX_LEN or "`" in raw or raw.startswith(">"):
        return none
    rxs, mention, negations = _lexicons()
    phrase = _phrase_rx(tuple(phrases) if phrases is not None else DEFAULT_PHRASES)
    hit = lambda t: re.search(phrase, t) or rxs["callout_alone"].search(t)  # noqa: E731
    text = words.normalize(raw)
    plain = " ".join(_QUOTES.sub(" ", text).split())
    if not hit(plain) or _mention().search(plain):
        return none
    text = plain
    has_phrase = re.search(phrase, text) is not None
    callout = rxs["callout_alone"].search(text) is not None or (
        has_phrase and (rxs["callout"].search(text) is not None or re.search(rf"(?<![a-z0-9])no\s+(?:{phrase})", text) is not None))
    kind = "callout" if callout else "declaration"
    bare = re.sub(phrase, " ", text)  # a seat named "One" must not match inside the declaration phrase
    named = {i for i, names in seats if any(_name_hit(bare, n) for n in names)} if callout or has_phrase else set()
    if named and has_phrase:
        kind = "callout"
    if kind == "declaration" and (words.negated(text, negations) or rxs["negation"].search(text)):
        return none
    ask = "?" in text
    if kind == "declaration":
        return OneCardReading(kind, "ask" if ask or reply_to is not None else "sure")
    if len(named) > 1 or (named and reply_to is not None and reply_to not in named):
        return OneCardReading(kind, "ask")
    if named:
        return OneCardReading(kind, "ask" if ask else "sure", "name", next(iter(named)))
    if reply_to is not None:
        return OneCardReading(kind, "ask" if ask else "sure", "reply", reply_to)
    return OneCardReading(kind, "ask" if ask else "sure")
