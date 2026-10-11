"""Decide sure / unsure / silent for one message vector over the legal moves (ported from the FEAT-36 harness)."""
from dataclasses import dataclass


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


def decide(vector, legal, head):
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
    th = head.thresholds
    if p1 >= th["t_sure"] and feat.margin >= th["margin"] and feat.talk_gap >= th["g_sure"]:
        return Decision("sure", (m1,), feat)
    if p1 >= th["t_floor"] and feat.talk_gap >= th["g_floor"]:
        return Decision("unsure", (m1, m2) if m2 else (m1,), feat)
    return Decision("silent", (), feat)
