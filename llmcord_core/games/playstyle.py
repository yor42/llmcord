"""Play style: eight 0..100 traits (50 neutral) that shape how a character plays a minigame (FEAT-32, D31).

Pure Python: no Discord, database, model or clock. Randomness comes only from a seed string.

Judgment and unpredictability are deliberately different. Judgment is SYSTEMATIC: a foolish
character consistently overrates weak hands (a bias that points the same way every time, plus a
little noise), a sharp one reads them almost exactly. Unpredictability is RANDOM: the chance of an
out-of-pattern move on a given decision, capped at MAX_ERRATIC. A game policy asks ``misjudge`` for
its hand-strength estimate and ``erratic`` whether to deviate; an erratic move must still be legal,
and ``never_fold_free`` stops a fold when checking is free.
"""
import dataclasses
import hashlib
import json
import math

from ..i18n import Phrase, terms
from .base import seed_hash

TRAITS = (
    "looseness", "aggression", "bluffing", "stubbornness",
    "suspicion", "risk", "judgment", "unpredictability",
)
NEUTRAL = 50
GROUPS = {  # keys are Msg (str) terms: the dashboard shows them through t()
    terms.PLAY_STYLE_GROUP_STYLE: ("looseness", "aggression", "bluffing", "risk"),
    terms.PLAY_STYLE_GROUP_READS: ("suspicion", "stubbornness"),
    terms.PLAY_STYLE_GROUP_MIND: ("judgment", "unpredictability"),
}
MAX_ERRATIC = 0.35
MAX_BIAS = 0.25  # upward estimate bias at judgment 0
NOISE_SHARP, NOISE_FOOL = 0.02, 0.05

_WORDS = terms.PLAY_STYLE_WORDS


@dataclasses.dataclass(frozen=True)
class PlayStyle:
    looseness: int = NEUTRAL
    aggression: int = NEUTRAL
    bluffing: int = NEUTRAL
    stubbornness: int = NEUTRAL
    suspicion: int = NEUTRAL
    risk: int = NEUTRAL
    judgment: int = NEUTRAL
    unpredictability: int = NEUTRAL

    def unit(self, trait):
        return getattr(self, trait) / 100


def parse(data):
    """Strict: unknown keys, non-ints (bool included) and values outside 0..100 raise ValueError."""
    unknown = set(data) - set(TRAITS)
    if unknown:
        raise ValueError(f"unknown play style traits: {sorted(unknown)}")
    for name, value in data.items():
        if type(value) is not int or not 0 <= value <= 100:
            raise ValueError(f"{name} must be an integer 0..100")
    return PlayStyle(**data)


def _lenient(value):
    if isinstance(value, bool):
        return NEUTRAL
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return NEUTRAL
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return NEUTRAL
    return max(0, min(100, round(value)))


def clamp(data):
    """Lenient: ignore unknown keys, turn garbage into neutral, round and clamp numbers."""
    if not isinstance(data, dict):
        raise ValueError("play style must be an object")
    return PlayStyle(**{t: _lenient(data[t]) for t in TRAITS if t in data})


def dumps(style):
    return json.dumps(dataclasses.asdict(style), sort_keys=True, separators=(",", ":"))


def loads(text):
    if not text:
        return None
    try:
        data = json.loads(text)
        return parse(data) if isinstance(data, dict) else None
    except (ValueError, TypeError):
        return None


def derived_from_id(character_id):
    """The unset default: today's Doubt bluff/suspicion derivation from the character id, other traits neutral."""
    h = seed_hash(f"doubt-persona:{character_id}")
    return PlayStyle(
        bluffing=round(100 * int(h[:8], 16) / 0xFFFFFFFF),
        suspicion=round(100 * int(h[8:16], 16) / 0xFFFFFFFF),
    )


def resolve(text, character_id):
    return loads(text) or derived_from_id(character_id)


def roll(seed, tag):
    """Deterministic uniform float in [0, 1)."""
    digest = hashlib.sha256(f"{seed}/{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def erratic(style, seed, tag):
    """True when this decision should be an out-of-pattern (but legal) move."""
    return roll(seed, f"erratic/{tag}") < MAX_ERRATIC * style.unit("unpredictability")


def misjudge(strength, style, seed, tag):
    """Hand-strength estimate in 0..1: foolish judgment adds a consistent upward bias, plus small noise."""
    strength = max(0.0, min(1.0, strength))
    j = style.unit("judgment")
    bias = MAX_BIAS * max(0.0, 1 - 2 * j)
    amp = NOISE_SHARP + (NOISE_FOOL - NOISE_SHARP) * (1 - j)
    noise = (roll(seed, f"misjudge/{tag}") * 2 - 1) * amp
    return max(0.0, min(1.0, strength + bias + noise))


def describe_phrases(style):
    """The trait words worth saying, each a Msg or a Phrase for Translator.render; [balanced] when none is."""
    parts = []
    for t in TRAITS:
        v = getattr(style, t)
        if 40 <= v <= 60:
            continue
        word = _WORDS[t][0 if v > 50 else 1]
        parts.append(word if v >= 75 or v <= 25 else Phrase(terms.PLAY_STYLE_SOMEWHAT, word=word))
    return parts or [terms.PLAY_STYLE_BALANCED]


def describe(style):
    return ", ".join(str(p) for p in describe_phrases(style))


def never_fold_free(legal, move):
    """A fold is pointless when checking is free: turn it into a check."""
    return "check" if move == "fold" and "check" in legal else move
