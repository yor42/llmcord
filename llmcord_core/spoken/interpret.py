"""Decide sure / unsure / silent for one message vector over the legal moves (ported from the FEAT-36 harness)."""
from dataclasses import dataclass

# Cosine similarity at which a server example settles a message (e5-small vectors are bunched high, so this is far above 0.5).
# Calibrated on the FEAT-36 prototype set (673 phrases, leave-one-out nearest neighbour): 0.96 -> 109 matches, 98.2% same label;
# 0.962 -> 91 matches, 100%. Server examples are near-boundary phrases by construction, so the higher value is used; re-check with real server examples.
EXAMPLE_MATCH = 0.962


@dataclass(frozen=True)
class Features:
    p1: float
    margin: float
    talk_gap: float


@dataclass(frozen=True)
class Decision:
    band: str  # 'sure' | 'unsure' | 'silent'
    moves: tuple
    features: Features | None = None


SILENT = Decision("silent", ())


def _example(vector, legal, examples):
    best, label = EXAMPLE_MATCH, None  # pure-Python dot products: cost grows with the example count (SPOKEN_EXAMPLES_MAX = 200 x 384 dims)
    for vec, name in examples:
        if name != "talk" and name not in legal:
            continue
        sim = sum(a * b for a, b in zip(vector, vec))
        if sim >= best:
            best, label = sim, name
    return label


def decide(vector, legal, head, examples=()):
    """`examples`: (unit vector, label) pairs from the server; a near match to a 'talk' or legal-move example settles the band."""
    legal = {m for m in legal if m in head.classes and m != "talk"}
    if not legal:
        return SILENT
    probs = head.probabilities(vector)
    keep = {m: p for m, p in probs.items() if m == "talk" or m in legal}
    total = sum(keep.values()) or 1.0
    scores = {m: p / total for m, p in keep.items() if m != "talk"}
    talk = keep.get("talk", 0.0) / total
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    m1, p1 = ranked[0]
    m2, p2 = ranked[1] if len(ranked) > 1 else (None, 0.0)
    feat = Features(p1, p1 - p2, p1 - talk)
    label = _example(vector, legal, examples) if examples else None
    if label is not None:
        return Decision("silent", (), feat) if label == "talk" else Decision("sure", (label,), feat)
    th = head.thresholds
    if p1 >= th["t_sure"] and feat.margin >= th["margin"] and feat.talk_gap >= th["g_sure"]:
        return Decision("sure", (m1,), feat)
    if p1 >= th["t_floor"] and feat.talk_gap >= th["g_floor"]:
        return Decision("unsure", (m1, m2) if m2 else (m1,), feat)
    return Decision("silent", (), feat)
