"""The trained spoken-move head (FEAT-36 export): standardize, linear layer, softmax. Pure python."""
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BLACKJACK_PATH = Path(__file__).parent / "heads" / "head-e5-blackjack.json"
BLACKJACK_SHA256 = "52b1a1b97ce3fb9c625045dd144d9ac2ed0f480245eb90be296f40496261cb9c"
THRESHOLD_KEYS = ("t_sure", "margin", "g_sure", "t_floor", "g_floor")


class HeadError(ValueError):
    pass


@dataclass(frozen=True)
class Head:
    classes: tuple
    mu: tuple
    sd: tuple
    W: tuple  # dim rows of len(classes)
    b: tuple
    T: float
    thresholds: dict
    model: str
    revision: str
    prefix: str

    @property
    def dim(self):
        return len(self.mu)

    def probabilities(self, vector):
        """{class: p} for one message vector."""
        if len(vector) != self.dim:
            raise HeadError(f"vector has {len(vector)} dims, head wants {self.dim}")
        logits = list(self.b)
        for x, m, s, row in zip(vector, self.mu, self.sd, self.W):
            z = (x - m) / s
            for j, w in enumerate(row):
                logits[j] += z * w
        logits = [v / self.T for v in logits]
        top = max(logits)
        exps = [math.exp(v - top) for v in logits]
        total = sum(exps)
        return {c: e / total for c, e in zip(self.classes, exps)}


def _numbers(value, n):
    if not isinstance(value, list) or len(value) != n or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        raise HeadError("malformed head arrays")
    return tuple(float(v) for v in value)


def parse_head(data):
    if not isinstance(data, dict):
        raise HeadError("head is not an object")
    classes = data.get("classes")
    if not isinstance(classes, list) or not all(isinstance(c, str) for c in classes) or "talk" not in classes or len(set(classes)) != len(classes):
        raise HeadError("head classes must be unique strings including 'talk'")
    for key in ("model", "model_revision", "prefix"):
        if not isinstance(data.get(key), str) or (key != "prefix" and not data[key]):
            raise HeadError(f"head is missing {key}")
    th = data.get("thresholds")
    if not isinstance(th, dict) or any(not isinstance(th.get(k), (int, float)) or isinstance(th.get(k), bool) for k in THRESHOLD_KEYS):
        raise HeadError("head thresholds incomplete")
    mu = data.get("mu")
    if not isinstance(mu, list) or not mu:
        raise HeadError("malformed head arrays")
    dim, n = len(mu), len(classes)
    W = data.get("W")
    if not isinstance(W, list) or len(W) != dim:
        raise HeadError("head W shape does not match")
    sd = _numbers(data.get("sd"), dim)
    if any(s == 0 for s in sd):
        raise HeadError("head sd has zeros")
    T = data.get("T", 1.0)
    if not isinstance(T, (int, float)) or isinstance(T, bool) or T <= 0:
        raise HeadError("head temperature must be positive")
    return Head(tuple(classes), _numbers(mu, dim), sd, tuple(_numbers(row, n) for row in W), _numbers(data.get("b"), n), float(T),
                {k: float(th[k]) for k in THRESHOLD_KEYS}, data["model"], data["model_revision"], data["prefix"])


def load_head(path, sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise HeadError("head file hash mismatch")
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise HeadError("head is not valid JSON") from exc
    return parse_head(data)


@lru_cache(maxsize=None)
def blackjack_head():
    return load_head(BLACKJACK_PATH, BLACKJACK_SHA256)
