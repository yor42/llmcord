"""Embedder seam: the real model arrives in a later part; tests use FakeEmbedder."""
from typing import Protocol


class Embedder(Protocol):
    prefix: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbedder:
    def __init__(self, vectors=None, dim=384, prefix="query: "):
        self.vectors, self.dim, self.prefix, self.calls = dict(vectors or {}), dim, prefix, []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [list(self.vectors.get(t, [0.0] * self.dim)) for t in texts]
