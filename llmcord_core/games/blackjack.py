"""Blackjack rules engine: six-deck shoe, dealer stands on soft 17, 3:2 natural, hit/stand/double.

Pure functions over an immutable ``State``. Cards are ints 0..51: rank = card % 13 (0 = ace,
9..12 = ten-value), suit = card // 13. A round is fully determined by (seed, seats, moves).
"""
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

from .base import IllegalMove, Seat, short_seed_hash, shuffled

DECKS = 6
MAX_SEATS = 7
DEALER_STANDS_ON = 17
RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
SUITS = ("♠", "♥", "♦", "♣")
MOVES = ("hit", "stand", "double")


def new_shoe(seed):
    return tuple(shuffled(list(range(52)) * DECKS, seed))


def card_str(card):
    return RANKS[card % 13] + SUITS[card // 13]


def hand_str(cards):
    return " ".join(card_str(c) for c in cards)


def hand_total(cards):
    """(best total, soft) where soft means an ace is currently counted as 11."""
    total = sum(min(c % 13 + 1, 10) for c in cards)
    if any(c % 13 == 0 for c in cards) and total + 10 <= 21:
        return total + 10, True
    return total, False


def total_str(cards):
    total, soft = hand_total(cards)
    return f"soft {total}" if soft else str(total)


def is_natural(cards):
    return len(cards) == 2 and hand_total(cards)[0] == 21


@dataclass(frozen=True)
class State:
    seed: str = field(repr=False)
    seats: tuple
    shoe: tuple = field(repr=False)
    pos: int
    hands: tuple  # per seat: tuple of cards
    stakes: tuple  # per seat: current stake (doubled after a double)
    status: tuple  # per seat: 'playing' | 'stand' | 'bust' | 'blackjack'
    doubled: tuple
    dealer: tuple = field(repr=False)  # hole card stays out of repr
    turn: Optional[int]  # seat index on turn, None once finished
    finished: bool
    dealer_natural: bool


@dataclass(frozen=True)
class View:
    seat: Optional[int]  # viewer's seat index, None for spectators / the table message
    hands: tuple
    totals: tuple  # per seat (total, soft)
    status: tuple
    stakes: tuple
    dealer: tuple  # only the up card until the round is finished
    dealer_hidden: bool
    dealer_total: Optional[tuple]  # (total, soft), None while the hole card is hidden
    turn: Optional[int]
    legal: tuple  # viewer's legal moves (empty unless on turn)
    finished: bool
    seed_hash: str
    seed: Optional[str]  # revealed only when finished
    total: Optional[int] = None  # viewer's own best total (None for spectators)
    soft: bool = False


@dataclass(frozen=True)
class SeatResult:
    seat: int
    outcome: str  # 'blackjack' | 'win' | 'push' | 'lose' | 'bust'
    stake: int  # final stake (doubled if doubled)
    returned: int  # amount handed back to the seat, stake included; 0 on loss


@dataclass(frozen=True)
class Result:
    seats: tuple
    dealer: tuple
    dealer_total: int
    seed: str
    seed_hash: str
    summary: str


def _draw(state):
    if state.pos >= len(state.shoe):
        raise IllegalMove("shoe exhausted")
    return state.shoe[state.pos], replace(state, pos=state.pos + 1)


def _next_turn(state, after):
    for i in range(after, len(state.seats)):
        if state.status[i] == "playing":
            return i
    return None


def _finish(state):
    """Dealer plays (only if some seat is still live), then the round ends."""
    dealer = state.dealer
    if any(s == "stand" for s in state.status) and not state.dealer_natural:
        while hand_total(dealer)[0] < DEALER_STANDS_ON:
            card, state = _draw(replace(state, dealer=dealer))
            dealer = dealer + (card,)
    return replace(state, dealer=dealer, turn=None, finished=True)


def _advance(state, after):
    turn = _next_turn(state, after)
    if turn is None:
        return _finish(state)
    return replace(state, turn=turn)


def new(seed, seats: Sequence[Seat]):
    seats = tuple(seats)
    if not 1 <= len(seats) <= MAX_SEATS:
        raise ValueError(f"blackjack needs 1 to {MAX_SEATS} seats")
    if [s.index for s in seats] != list(range(len(seats))):
        raise ValueError("seat indexes must be 0..n-1 in order")
    shoe = new_shoe(seed)
    n = len(seats)
    # deal order: one card to each seat, dealer up card, second card to each seat, dealer hole card
    hands = [[shoe[i]] for i in range(n)]
    up = shoe[n]
    for i in range(n):
        hands[i].append(shoe[n + 1 + i])
    dealer = (up, shoe[2 * n + 1])
    hands = tuple(tuple(h) for h in hands)
    natural = is_natural(dealer)
    status = tuple("blackjack" if is_natural(h) else "playing" for h in hands)
    state = State(
        seed=seed, seats=seats, shoe=shoe, pos=2 * n + 2, hands=hands,
        stakes=tuple(s.stake for s in seats), status=status, doubled=(False,) * n,
        dealer=dealer, turn=None, finished=False, dealer_natural=natural,
    )
    if natural:
        status = tuple("blackjack" if s == "blackjack" else "stand" for s in status)
        return replace(state, status=status, turn=None, finished=True)
    return _advance(state, 0)


def legal_moves(state, seat, can_double=True):
    if state.finished or state.turn != seat:
        return ()
    if can_double and len(state.hands[seat]) == 2:
        return ("hit", "stand", "double")
    return ("hit", "stand")


def apply(state, seat, move, can_double=True):
    if state.finished:
        raise IllegalMove("the round is over")
    if state.turn != seat:
        raise IllegalMove(f"seat {seat} is not on turn")
    if move not in legal_moves(state, seat, can_double):
        raise IllegalMove(f"{move!r} is not legal for seat {seat}")
    status = list(state.status)
    if move == "stand":
        status[seat] = "stand"
        return _advance(replace(state, status=tuple(status)), seat + 1)
    card, state = _draw(state)
    hands = list(state.hands)
    hands[seat] = hands[seat] + (card,)
    busted = hand_total(hands[seat])[0] > 21
    if move == "double":
        stakes = list(state.stakes)
        stakes[seat] *= 2
        doubled = list(state.doubled)
        doubled[seat] = True
        state = replace(state, stakes=tuple(stakes), doubled=tuple(doubled))
        status[seat] = "bust" if busted else "stand"
        return _advance(replace(state, hands=tuple(hands), status=tuple(status)), seat + 1)
    if busted:
        status[seat] = "bust"
        return _advance(replace(state, hands=tuple(hands), status=tuple(status)), seat + 1)
    return replace(state, hands=tuple(hands))


def replay(seed, seats, moves):
    """Rebuild a state from its seed, seats and (seat, move) list; raises IllegalMove on a bad log."""
    state = new(seed, seats)
    for seat, move in moves:
        state = apply(state, seat, move)
    return state


def view(state, seat=None):
    if seat is not None and not 0 <= seat < len(state.seats):
        raise ValueError(f"no seat {seat}")
    revealed = state.finished
    dealer = state.dealer if revealed else state.dealer[:1]
    own = state.hands[seat] if seat is not None else None
    total, soft = hand_total(own) if own is not None else (None, False)
    return View(
        seat=seat,
        hands=state.hands,
        totals=tuple(hand_total(h) for h in state.hands),
        status=state.status,
        stakes=state.stakes,
        dealer=dealer,
        dealer_hidden=not revealed,
        dealer_total=hand_total(dealer) if revealed else None,
        turn=state.turn,
        legal=legal_moves(state, seat) if seat is not None else (),
        finished=state.finished,
        seed_hash=short_seed_hash(state.seed),
        seed=state.seed if revealed else None,
        total=total,
        soft=soft,
    )


def _settle(state, i):
    stake = state.stakes[i]
    st = state.status[i]
    dealer_total = hand_total(state.dealer)[0]
    if st == "bust":
        return "bust", 0
    if state.dealer_natural:
        return ("push", stake) if st == "blackjack" else ("lose", 0)
    if st == "blackjack":
        return "blackjack", stake + stake * 3 // 2
    total = hand_total(state.hands[i])[0]
    if dealer_total > 21 or total > dealer_total:
        return "win", stake * 2
    if total == dealer_total:
        return "push", stake
    return "lose", 0


def result(state):
    if not state.finished:
        raise IllegalMove("the round is not finished")
    seats = []
    lines = []
    for i in range(len(state.seats)):
        outcome, returned = _settle(state, i)
        seats.append(SeatResult(i, outcome, state.stakes[i], returned))
        lines.append(f"Seat {i + 1}: {hand_str(state.hands[i])} ({total_str(state.hands[i])}) {outcome}")
    dealer_total = hand_total(state.dealer)[0]
    head = f"Dealer: {hand_str(state.dealer)} ({total_str(state.dealer)})"
    if dealer_total > 21:
        head += " bust"
    elif state.dealer_natural:
        head += " blackjack"
    return Result(
        seats=tuple(seats), dealer=state.dealer, dealer_total=dealer_total,
        seed=state.seed, seed_hash=short_seed_hash(state.seed),
        summary="\n".join([head, *lines]),
    )
