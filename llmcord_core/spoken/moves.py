"""Data shapes shared by the spoken-move readers."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SpokenMove:
    kind: str
    amount: int | None = None
    mode: str | None = None  # 'to' | 'by' | None (amount given, meaning unclear)
    preset: str | None = None  # 'min' | 'half_pot' | 'pot' | None


@dataclass(frozen=True)
class Reading:
    candidates: tuple  # 1..2 SpokenMove, in order of first appearance
    sure: bool


@dataclass(frozen=True)
class Resolution:
    moves: tuple  # legal engine move strings, best first
    needs_picker: bool = False
