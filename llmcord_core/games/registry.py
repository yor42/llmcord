"""The games a table can run: one frozen entry per game, looked up by key.

Per-game hooks keep the store free of game branches: public_legal (the moves the snapshot shows; never anything hidden),
summary, settlements, round_facts (+ fact_keys, for the table summary), apply_move and illegal_text.
"""
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Optional

from ..i18n import codes, terms
from . import blackjack, doubt, holdem, onecard


@dataclass(frozen=True)
class Game:
    key: str
    name: str
    engine: ModuleType
    normalize_rules: Callable[[Any], dict]
    max_seats: int
    default_enabled: bool
    ledger_label: str
    noun: str = ''  # the game's name inside a sentence
    min_seats: int = 1
    ante: bool = False  # seats stake the round's ante (rules snapshot), not a bet
    cash: bool = False  # a cash table: buy-in into a stack that carries across hands; the wallet moves only at buy-in, rebuy and cash-out
    public_legal: Optional[Callable] = None  # (state, can_double(), can_insure()) -> {seat: moves}
    summary: Optional[Callable] = None  # (state) -> str | None
    settlements: Optional[Callable] = None  # (state, ledger_label, label) -> [(seat, outcome, stake, returned, reason)]; None for a cash game
    round_facts: Optional[Callable] = None  # (state) -> dict merged into the table summary row
    fact_keys: tuple = ()
    apply_move: Optional[Callable] = None  # (state, seat, move, can_double()) -> state
    illegal_text: Optional[Callable] = None  # (state, seat, move) -> ErrorCode (GameError(code) shows it; str() is English)


def _bj_legal(state, can_double, can_insure):
    return {state.turn: blackjack.legal_moves(state, state.turn, can_double(), can_insure())} if state.turn is not None else {}


def _bj_settlements(state, ledger_label, label):
    out = []
    for res in blackjack.result(state).seats:
        note = ', insurance paid' if res.insurance and state.dealer_natural else ''
        out.append((res.seat, res.outcome, res.stake, res.returned, f'{ledger_label} payout ({res.outcome}), {label}{note}'))
    return out


def _bj_illegal(state, seat, move):
    return codes.ENGINE_NOT_YOUR_TURN if state.turn != seat else codes.ENGINE_MOVE_NOT_ALLOWED


def _doubt_settlements(state, ledger_label, label):
    return [(res.seat, res.outcome, state.seats[res.seat].stake, res.payout, f'{ledger_label} payout ({label})') for res in doubt.result(state).seats]


def _doubt_illegal(state, seat, move):
    if state.finished:
        return codes.ENGINE_GAME_OVER
    if move == 'doubt':
        return codes.ENGINE_NOTHING_TO_DOUBT
    if move == 'close':
        return codes.ENGINE_NOTHING_TO_CLOSE
    return codes.ENGINE_NOT_YOUR_TURN if state.turn != seat else codes.ENGINE_MOVE_NOT_ALLOWED


def _holdem_facts(state):
    res = holdem.result(state)
    return {'pot': sum(pot.amount for pot in res.pots), 'winners': sorted({w for pot in res.pots for w in pot.winners}),
            'hand': holdem.hand_name(res.pots[0].hand) if res.pots and res.pots[0].hand is not None else None}


def _holdem_illegal(state, seat, move):
    if state.finished:
        return codes.ENGINE_HAND_OVER
    return codes.ENGINE_NOT_YOUR_TURN if state.turn != seat else codes.ENGINE_MOVE_NOT_ALLOWED


def _onecard_settlements(state, ledger_label, label):
    return [(res.seat, res.outcome, state.seats[res.seat].stake, res.payout, f'{ledger_label} payout ({label})') for res in onecard.result(state).seats]


def _onecard_illegal(state, seat, move):
    if state.finished:
        return codes.ENGINE_GAME_OVER
    return codes.ENGINE_NOT_YOUR_TURN if state.turn != seat else codes.ENGINE_MOVE_NOT_ALLOWED


_GAMES = {
    "blackjack": Game("blackjack", terms.GAME_BLACKJACK, blackjack, blackjack.normalize_rules, blackjack.MAX_SEATS, True, "Blackjack", noun=terms.NOUN_BLACKJACK,
                      public_legal=_bj_legal, summary=lambda state: blackjack.result(state).summary, settlements=_bj_settlements,
                      round_facts=lambda state: {'dealer_total': blackjack.hand_total(state.dealer)[0]}, fact_keys=('dealer_total',),
                      apply_move=lambda state, seat, move, can_double: blackjack.apply(state, seat, move, can_double()), illegal_text=_bj_illegal),
    "doubt": Game("doubt", terms.GAME_DOUBT, doubt, doubt.normalize_rules, doubt.MAX_SEATS, True, "I Doubt It", noun=terms.NOUN_DOUBT,
                  min_seats=doubt.MIN_SEATS, ante=True,
                  public_legal=lambda state, can_double, can_insure: {}, summary=lambda state: None, settlements=_doubt_settlements,
                  round_facts=lambda state: {'winner': state.winner, 'pot': sum(s.stake for s in state.seats)}, fact_keys=('winner', 'pot'),
                  apply_move=lambda state, seat, move, can_double: doubt.apply(state, seat, move), illegal_text=_doubt_illegal),
    "holdem": Game("holdem", terms.GAME_HOLDEM, holdem, holdem.normalize_rules, holdem.MAX_SEATS, True, "Texas Hold'em", noun=terms.NOUN_HOLDEM,
                   min_seats=holdem.MIN_SEATS, cash=True,
                   public_legal=lambda state, can_double, can_insure: {state.turn: holdem.legal_moves(state, state.turn)} if state.turn is not None else {},
                   summary=lambda state: None, round_facts=_holdem_facts, fact_keys=('pot', 'winners', 'hand'),
                   apply_move=lambda state, seat, move, can_double: holdem.apply(state, seat, move), illegal_text=_holdem_illegal),
    "onecard": Game("onecard", terms.GAME_ONECARD, onecard, onecard.normalize_rules, onecard.MAX_SEATS, True, "One Card", noun=terms.NOUN_ONECARD,
                    min_seats=onecard.MIN_SEATS, ante=True,
                    public_legal=lambda state, can_double, can_insure: {}, summary=lambda state: None, settlements=_onecard_settlements,
                    round_facts=lambda state: {'winner': state.winner, 'pot': sum(s.stake for s in state.seats)}, fact_keys=('winner', 'pot'),
                    apply_move=lambda state, seat, move, can_double: onecard.apply(state, seat, move), illegal_text=_onecard_illegal),
}


def get(key):
    return _GAMES[key]


def keys():
    return tuple(_GAMES)
