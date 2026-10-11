"""FEAT-36 second pass: the word list counts as 'sure' only when the move word is most of the message.

Pure python. `other_tokens` drops the matched move phrases (and any particle attached to them) plus general fillers
and punctuation/emoji, and returns what is left; a message with more leftovers than N is not read by the word list.
"""
import re

FILLERS = frozenset((
    "ok okay fine yeah yea ya yes just then please pls plz i i'll ill let's lets me it one lol haha go "
    "ㅇㅋ 오케이 그냥 일단 할게 할게요 갈게 요 해 ㄱㄱ ㄱ 좀").split())
_LAUGH = re.compile(r"ㅋ+|ㅎ+")
_NOISE = re.compile(r"[^\w']+")


def extend_to_token_end(text, spans):
    """Merged (start, end) character ranges, each extended to the end of its whitespace-delimited token."""
    out = []
    for start, length in sorted(spans):
        end = start + length
        while end < len(text) and not text[end].isspace():
            end += 1
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def other_tokens(text, spans):
    """Tokens of the normalized text outside the matched spans, minus fillers, laughs and punctuation."""
    keep, pos = [], 0
    for start, end in extend_to_token_end(text, spans):
        keep.append(text[pos:start])
        pos = end
    keep.append(text[pos:])
    tokens = _NOISE.sub(" ", " ".join(keep)).split()
    return [t.strip("'") for t in tokens if t.strip("'") and t.strip("'") not in FILLERS and not _LAUGH.fullmatch(t)]


def match_spans(text, lexicon, kinds):
    """[(start, length)] of every word-list match for kinds in the normalized text (llmcord_core matcher)."""
    from llmcord_core.spoken import words
    exclude = [words.normalize(x) for x in lexicon.get("_exclude", ())]
    return [s for k in kinds for s in words._found(text, lexicon.get(k, {}), exclude)]


def extra_tokens(text, lexicon, kinds):
    """Number of non-move, non-filler tokens in the message."""
    from llmcord_core.spoken import words
    norm = words.normalize(text)
    return len(other_tokens(norm, match_spans(norm, lexicon, kinds)))
