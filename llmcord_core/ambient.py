"""Ambient invitation rule (D37): a pure EN/KO check for whether chat invites a character to speak; no model, no store."""
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .spoken import words

_HANGUL = re.compile("[가-힣]")


@dataclass(frozen=True)
class Invite:
    signals: tuple = ()  # which of 'name', 'phrase', 'follow_up', 'favorite' fired
    named: tuple = ()  # indices into the names given, in order, of every name found

    def __bool__(self):
        return bool(self.signals)


@lru_cache(maxsize=1)
def _lex():
    data = json.loads((Path(__file__).with_name("ambient_invite.json")).read_text(encoding="utf-8"))
    data["exclude"] = [words.normalize(x) for x in data["exclude"]]
    data["question_end"] = re.compile(rf"(?:{data['question_end']})[\s.!~]*$")
    data["reaction_re"] = re.compile(data["reaction"]["pattern"])
    data["particles"] = (*words.KO_END, *data["ko_particles"])
    reactions = {words.normalize(x) for x in (*data["reaction"]["en"], *data["reaction"]["ko"])}
    data["reaction_set"] = {x.replace(" ", "_") for x in reactions}
    return data


def _text(text):
    return words.normalize((text or "").replace("？", "?"))


def _name_in(text, name):
    """Names are word-bounded; a Hangul name must end the word or take a particle/vocative (루나는, 루나야, 하늘의)."""
    particles = _lex()["particles"]
    for m in re.finditer(re.escape(name), text):
        if m.start() and text[m.start() - 1].isalnum():
            continue
        rest = text[m.end():].split(" ", 1)[0]
        if name[-1].isalnum() and rest and rest[0].isalnum() and not (_HANGUL.match(rest[0]) and rest.startswith(particles)):
            continue
        return True
    return False


def _name_forms(name):
    """The full name, plus its first word that is not a title or article and has 3+ letters (2+ syllables in Hangul)."""
    name = words.normalize(name)
    forms = [name] if name else []
    parts = name.split()
    if len(parts) > 1:
        for part in parts:
            hangul = len(_HANGUL.findall(part))
            if part in _lex()["title_words"]:
                continue
            if hangul >= 2 or (not hangul and sum(c.isalpha() for c in part) >= 3):
                forms.append(part)
                break
    return forms


def _question(text):
    return "?" in re.sub(r"https?://\S+", "", text) or _lex()["question_end"].search(text) is not None


def _reaction(text):
    """Only reaction words (at most 3), once punctuation and emoji are dropped."""
    lex = _lex()
    tokens = re.findall(r"[a-z_ㄱ-ㅣ가-힣']+", text.replace("no way", "no_way"))
    if not tokens or len(text.split()) > lex["reaction_max_words"]:
        return False
    return all(t in lex["reaction_set"] or lex["reaction_re"].fullmatch(t) for t in tokens)


def _spans(text, lexicon):
    return words.found(text, lexicon, [], True)


def _phrase(text, question):
    lex = _lex()
    hits = _spans(text, lex["invite"]) + (_spans(text, lex["invite_question"]) if question else [])
    banned = [(m.start(), m.end()) for x in lex["exclude"] for m in re.finditer(re.escape(x), text)]
    return any(not any(s < e2 and b < s + n for b, e2 in banned) for s, n in hits)


def invite(latest, recent_texts, names, *, favorite=False, follow_up=False) -> Invite:
    """Signals that invite a character to speak: a name in the latest or last two recent lines, an invite phrase in
    the latest, a question or short reaction right after a character spoke (follow_up), or a question from a member
    with a favorite in the cast (favorite)."""
    now = _text(latest)
    seen = [now, *(_text(t) for t in list(recent_texts)[-2:])]
    named = tuple(i for i, name in enumerate(names) if any(_name_in(t, form) for form in _name_forms(name) for t in seen))
    signals = []
    if named:
        signals.append("name")
    question = _question(now)
    if _phrase(now, question):
        signals.append("phrase")
    if follow_up and (question or _reaction(now)):
        signals.append("follow_up")
    if favorite and question:
        signals.append("favorite")
    return Invite(tuple(signals), named)


def invited(latest, recent_texts, names, *, favorite=False, follow_up=False) -> bool:
    return bool(invite(latest, recent_texts, names, favorite=favorite, follow_up=follow_up))
