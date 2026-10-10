"""The games a table can run: one frozen entry per game, looked up by key.

Per-game hooks keep the store free of game branches: public_legal (the moves the snapshot shows; never anything hidden),
summary, settlements, round_facts (+ fact_keys, for the table summary), apply_move and illegal_text.
"""
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Optional

from . import blackjack, doubt


@dataclass(frozen=True)
class Game:
    key: str
    name: str
    engine: ModuleType
    normalize_rules: Callable[[Any], dict]
    max_seats: int
    default_enabled: bool
    ledger_label: str
    min_seats: int = 1
    ante: bool = False  # seats stake the round's ante (rules snapshot), not a bet
    public_legal: Optional[Callable] = None  # (state, can_double(), can_insure()) -> {seat: moves}
    summary: Optional[Callable] = None  # (state) -> str | None
    settlements: Optional[Callable] = None  # (state, ledger_label, label) -> [(seat, outcome, stake, returned, reason)]
    round_facts: Optional[Callable] = None  # (state) -> dict merged into the table summary row
    fact_keys: tuple = ()
    apply_move: Optional[Callable] = None  # (state, seat, move, can_double()) -> state
    illegal_text: Optional[Callable] = None  # (state, seat, move) -> str


def _bj_legal(state, can_double, can_insure):
    return {state.turn: blackjack.legal_moves(state, state.turn, can_double(), can_insure())} if state.turn is not None else {}


def _bj_settlements(state, ledger_label, label):
    out = []
    for res in blackjack.result(state).seats:
        note = ', insurance paid' if res.insurance and state.dealer_natural else ''
        out.append((res.seat, res.outcome, res.stake, res.returned, f'{ledger_label} payout ({res.outcome}), {label}{note}'))
    return out


def _bj_illegal(state, seat, move):
    return "It's not your turn." if state.turn != seat else "That move isn't allowed right now."


def _doubt_settlements(state, ledger_label, label):
    return [(res.seat, res.outcome, state.seats[res.seat].stake, res.payout, f'{ledger_label} payout ({label})') for res in doubt.result(state).seats]


def _doubt_illegal(state, seat, move):
    if state.finished:
        return 'This game is over.'
    if move == 'doubt':
        return 'There is nothing to doubt, or it is your own claim.'
    if move == 'close':
        return 'There is nothing for you to close.'
    return "It's not your turn." if state.turn != seat else "That move isn't allowed right now."


_GAMES = {
    "blackjack": Game("blackjack", "Blackjack", blackjack, blackjack.normalize_rules, blackjack.MAX_SEATS, True, "Blackjack",
                      public_legal=_bj_legal, summary=lambda state: blackjack.result(state).summary, settlements=_bj_settlements,
                      round_facts=lambda state: {'dealer_total': blackjack.hand_total(state.dealer)[0]}, fact_keys=('dealer_total',),
                      apply_move=lambda state, seat, move, can_double: blackjack.apply(state, seat, move, can_double()), illegal_text=_bj_illegal),
    "doubt": Game("doubt", "I Doubt It", doubt, doubt.normalize_rules, doubt.MAX_SEATS, True, "I Doubt It",
                  min_seats=doubt.MIN_SEATS, ante=True,
                  public_legal=lambda state, can_double, can_insure: {}, summary=lambda state: None, settlements=_doubt_settlements,
                  round_facts=lambda state: {'winner': state.winner, 'pot': sum(s.stake for s in state.seats)}, fact_keys=('winner', 'pot'),
                  apply_move=lambda state, seat, move, can_double: doubt.apply(state, seat, move), illegal_text=_doubt_illegal),
}


def get(key):
    return _GAMES[key]


def keys():
    return tuple(_GAMES)
