"""The games a table can run: one frozen entry per game, looked up by key."""
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

from . import blackjack


@dataclass(frozen=True)
class Game:
    key: str
    name: str
    engine: ModuleType
    normalize_rules: Callable[[Any], dict]
    max_seats: int
    default_enabled: bool
    ledger_label: str


_GAMES = {
    "blackjack": Game("blackjack", "Blackjack", blackjack, blackjack.normalize_rules, blackjack.MAX_SEATS, True, "Blackjack"),
}


def get(key):
    return _GAMES[key]


def keys():
    return tuple(_GAMES)
