"""Game-agnostic word-list matcher: finds which move kinds a chat message names (D32)."""
import re
import unicodedata

from .amounts import parse_amount
from .moves import Reading, SpokenMove

_SUFFIX = "s|es|ing"
_KO_CONT = "해|할|함|하|고|아|자|돼|되"
# Hangul that may follow a Korean lexicon word in the same token: polite/verb endings and particles.
KO_END = ("요", "해", "할", "함", "하", "합", "했", "고", "이", "가", "은", "는", "을", "를", "로", "으로", "도", "만",
           "게", "겠", "지", "자", "죠", "임", "다", "아", "돼", "되", "래", "한", "인", "입", "야", "간", "갈", "갑", "줘",
           "볼", "각")
_KO_END = KO_END
# Single-syllable negations match standalone or with these endings only ('말해' = say, '못생겼' = ugly).
_KO_NEG_END = {"말": "고|아|자|라|요|래|게|지|까|았", "못": "해|하|할|함|했|한"}
_HANGUL = "가-힣"
_CLAUSE_BREAK = re.compile(r"[,.!?;，。！？；]")


def normalize(text):
    text = unicodedata.normalize("NFC", text).casefold().replace("’", "'")
    return " ".join(re.sub(r"[-_]", " ", text).split())


def _en(entry, exact=False):
    """Word-boundary pattern; entries of 4+ letters also match -s/-es/-ing (so 'bet' and 'jam' stay exact)."""
    e = normalize(entry)
    body = rf"(?<!\w){re.escape(e)}"
    if exact or len(e) < 4:
        return re.compile(body + r"(?!\w)")
    alt = rf"{body}(?:{_SUFFIX}|{re.escape(e[-1])}ing)?"
    if e.endswith("e"):
        alt += rf"|(?<!\w){re.escape(e[:-1])}ing"
    return re.compile(rf"(?:{alt})(?!\w)")


def _ko(entry):
    return re.compile(rf"(?<!\S){re.escape(normalize(entry))}")


def _ko_ends_ok(rest):
    return not rest or not "가" <= rest[0] <= "힣" or rest.startswith(KO_END)


def found(text, lex, exclude, exact=False):
    """[(start, length)] of every match of a {'en': [...], 'ko': [...]} word list."""
    out = [(m.start(), m.end() - m.start()) for e in lex.get("en", ()) for m in _en(e, exact).finditer(text)]
    for e in lex.get("ko", ()):
        for m in _ko(e).finditer(text):
            token = text[m.start():].split(" ", 1)[0]
            rest = text[m.end():].split(" ", 1)[0]
            if not any(token.startswith(x) for x in exclude) and _ko_ends_ok(rest):
                out.append((m.start(), len(normalize(e))))
    return out


_found = found


def negated(text, negations):
    for e in negations.get("en", ()):
        if _en(e, True).search(text):
            return True
    for e in negations.get("ko", ()):
        e = normalize(e)
        if len(e.replace(" ", "")) > 1:
            if re.search(re.escape(e).replace(r"\ ", r"\s?"), text):
                return True
        elif e in _KO_NEG_END:
            if re.search(rf"(?<!\S){e}(?:(?![{_HANGUL}])|{_KO_NEG_END[e]})", text):
                return True
        elif re.search(rf"(?<!\S){re.escape(e)}(?:(?!\S)|\s?(?:{_KO_CONT}))", text):
            return True
    return False


_negated = negated


def _first(text, lex, exclude):
    hits = _found(text, lex, exclude)
    return min((s for s, _ in hits), default=None)


def _near(text, hits, s, n):
    """True when the preset at (s, n) is within two words of a raise hit with no clause break between."""
    for rs, rl in hits:
        if s >= rs + rl:
            gap = text[rs + rl:s]
        elif rs >= s + n:
            gap = text[s + n:rs]
        else:
            return True
        if not _CLAUSE_BREAK.search(gap) and len(gap.split()) <= 1:
            return True
    return False


def _preset(text, presets, exclude, near=None):
    """(start, name) of the earliest preset word, the longest on a tie; only within two words of near hits if given."""
    best = None
    for name, lex in presets.items():
        for s, n in _found(text, lex, exclude, True):
            if near is not None and not _near(text, near, s, n):
                continue
            if best is None or (s, -n) < best[:2]:
                best = (s, -n, name)
    return (best[0], best[2]) if best else None


def read(text, lexicon, kinds, negations):
    """Reading of the kinds in text (at most two, in order of appearance), or None when no move is named."""
    text = normalize(text)
    exclude = [normalize(x) for x in lexicon.get("_exclude", ())]
    amount_kinds = lexicon.get("_amount_kinds", ())
    hits = sorted((s, k) for k in kinds if (s := _first(text, lexicon.get(k, {}), exclude)) is not None)
    names = [k for _, k in hits]
    presets, implied = lexicon.get("_presets", {}), False
    if "raise" not in names:
        alone = {lang: {normalize(x) for x in xs} for lang, xs in lexicon.get("_preset_alone", {}).items()}
        presets = {n: {lang: [x for x in xs if normalize(x) in alone.get(lang, ())] for lang, xs in lx.items()}
                   for n, lx in presets.items()}
    amount = parse_amount(text)
    near = _found(text, lexicon.get("raise", {}), exclude) if "raise" in names and amount is None else None
    preset = _preset(text, presets, exclude, near)
    if preset and "raise" in kinds and not names:
        hits, names, implied = [(preset[0], "raise")], ["raise"], True
    if not hits:
        return None
    cands = tuple(
        SpokenMove(k, *(amount if amount and k in amount_kinds else (None, None)),
                   preset[1] if preset and k == "raise" else None)
        for k in names[:2])
    sure = len(names) == 1 and not implied and not negated(text, negations) and "?" not in text and "？" not in text
    return Reading(cands, sure)
