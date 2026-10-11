"""Game-agnostic word-list matcher: finds which move kinds a chat message names (D32)."""
import re
import unicodedata

from .amounts import parse_amount
from .moves import Reading, SpokenMove

_SUFFIX = "s|es|ing"
_KO_CONT = "해|할|함|하|고|아|자|돼|되"


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


def _found(text, lex, exclude, exact=False):
    """[(start, length)] of every match of a {'en': [...], 'ko': [...]} word list."""
    out = [(m.start(), m.end() - m.start()) for e in lex.get("en", ()) for m in _en(e, exact).finditer(text)]
    for e in lex.get("ko", ()):
        for m in _ko(e).finditer(text):
            token = text[m.start():].split(" ", 1)[0]
            if not any(token.startswith(x) for x in exclude):
                out.append((m.start(), len(normalize(e))))
    return out


def _negated(text, negations):
    for e in negations.get("en", ()):
        if _en(e, True).search(text):
            return True
    for e in negations.get("ko", ()):
        e = normalize(e)
        if len(e.replace(" ", "")) > 1:
            if re.search(re.escape(e).replace(r"\ ", r"\s?"), text):
                return True
        elif re.search(rf"(?<!\S){re.escape(e)}(?:(?!\S)|\s?(?:{_KO_CONT}))", text):
            return True
    return False


def _first(text, lex, exclude):
    hits = _found(text, lex, exclude)
    return min((s for s, _ in hits), default=None)


def _preset(text, presets, exclude):
    """(start, name) of the earliest preset word, the longest on a tie."""
    best = None
    for name, lex in presets.items():
        for s, n in _found(text, lex, exclude, True):
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
    preset = _preset(text, presets, exclude)
    if preset and "raise" in kinds and not names:
        hits, names, implied = [(preset[0], "raise")], ["raise"], True
    if not hits:
        return None
    amount = parse_amount(text)
    cands = tuple(
        SpokenMove(k, *(amount if amount and k in amount_kinds else (None, None)),
                   preset[1] if preset and k == "raise" else None)
        for k in names[:2])
    sure = len(names) == 1 and not implied and not _negated(text, negations) and "?" not in text and "？" not in text
    return Reading(cands, sure)
