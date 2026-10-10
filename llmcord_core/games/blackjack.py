"""Blackjack rules engine: six-deck shoe, hit/stand/double, optional house rules (see normalize_rules).

Pure functions over an immutable ``State``. Cards are ints 0..51: rank = card % 13 (0 = ace,
9..12 = ten-value), suit = card // 13. A round is fully determined by (seed, seats, rules, moves).
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
INSURANCE_MOVES = ("insure", "even_money", "no_insurance")
CLASSIC_RULES = {"stand_on": DEALER_STANDS_ON, "hit_soft_17": False, "blackjack_pays": "3:2", "ties": "push", "insurance": False, "surrender": False}
DEFAULT_RULES = {**CLASSIC_RULES, "double": True, "split": False, "decks": DECKS}
_LEGACY_KEYS = {"dealer_stands_on": "stand_on", "dealer_hits_soft_17": "hit_soft_17"}


def normalize_rules(value=None):
    """The full rules dict for a stored/posted value (None or {} is the classic table); ValueError with a readable message on a bad one."""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("The blackjack rules must be a set of options.")
    unknown = sorted(str(k) for k in value if k not in CLASSIC_RULES)
    if unknown:
        raise ValueError(f"Unknown blackjack rule: {', '.join(unknown)}.")
    r = {**CLASSIC_RULES, **value}
    if type(r["stand_on"]) is not int or r["stand_on"] not in (16, 17, 18):
        raise ValueError("The dealer must stand on 16, 17 or 18.")
    for key, label in (("hit_soft_17", "Dealer on soft 17"), ("insurance", "Insurance"), ("surrender", "Late surrender")):
        if type(r[key]) is not bool:
            raise ValueError(f"{label} must be on or off.")
    if r["blackjack_pays"] not in ("3:2", "6:5"):
        raise ValueError("Blackjack must pay 3:2 or 6:5.")
    if r["ties"] not in ("push", "dealer"):
        raise ValueError("Ties must push or go to the dealer.")
    if r["stand_on"] != 17:
        r["hit_soft_17"] = False
    return {k: r[k] for k in CLASSIC_RULES}


def _display_rules(rules):
    r = {**DEFAULT_RULES}
    for key, value in (rules or {}).items():
        r[_LEGACY_KEYS.get(key, key)] = value
    if r["stand_on"] != 17:
        r["hit_soft_17"] = False
    return r


def rules_text(rules=None):
    """The table's rules in plain words (display only)."""
    r = _display_rules(rules)
    dealer = "Dealer hits soft 17" if r["hit_soft_17"] else "Dealer stands on all 17s" if r["stand_on"] == 17 else f"Dealer stands on {r['stand_on']}"
    parts = [dealer, f"Blackjack pays {r['blackjack_pays']}", "Dealer wins ties" if r["ties"] == "dealer" else "Ties push",
             "Double on the first two cards" if r["double"] else "No doubling", "Splitting allowed" if r["split"] else "No splitting"]
    if r["insurance"]:
        parts.append("Insurance")
    if r["surrender"]:
        parts.append("Late surrender")
    parts.append(f"{r['decks']} deck{'' if r['decks'] == 1 else 's'}")
    return " · ".join(parts)


def how_to_play(rules=None):
    r = _display_rules(rules)
    draws = f"it draws until {r['stand_on']} or more" + (", and also hits a soft 17." if r["hit_soft_17"] else ".")
    lines = [
        "**Blackjack in short**",
        "Get closer to 21 than the dealer without going over.",
        "• Cards: 2–10 are face value, J/Q/K are 10, an ace is 1 or 11.",
        "• Hit takes a card. Stand keeps your hand." + (" Double doubles your bet, takes one card and ends your turn (first two cards only)." if r["double"] else ""),
        f"• Over 21 is a bust: you lose. An ace with a ten-value card on your first two cards is blackjack and pays {r['blackjack_pays']}.",
        f"• The dealer plays after everyone: {draws}",
        "• Win: you get your bet back plus the same again. " + ("Tie: the dealer wins." if r["ties"] == "dealer" else "Tie: your bet comes back."),
    ]
    if r["insurance"]:
        lines.append("• Insurance: when the dealer shows an ace you may put up half your bet (bets of 2 or more) that the dealer has blackjack; it pays 2 to 1. With a blackjack you may take even money (paid 1 to 1 right away) instead.")
    if r["surrender"]:
        lines.append("• Surrender: on your first two cards, once the dealer has checked for blackjack, you may give up the hand and get half your bet back.")
    return "\n".join(lines)


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
    status: tuple  # per seat: 'playing' | 'stand' | 'bust' | 'blackjack' | 'surrender'
    doubled: tuple
    dealer: tuple = field(repr=False)  # hole card stays out of repr
    turn: Optional[int]  # seat index on turn, None once finished
    finished: bool
    dealer_natural: bool
    rules: dict = field(default_factory=lambda: dict(CLASSIC_RULES))
    phase: str = "play"  # 'insurance' while seats are asked (dealer shows an ace), then 'play'
    insured: tuple = ()  # per seat: insurance cost paid (0 if none)
    even_money: tuple = ()  # per seat: took even money on a blackjack


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
    phase: str = "play"
    insured: tuple = ()
    even_money: tuple = ()
    rules: dict = field(default_factory=lambda: dict(CLASSIC_RULES))


@dataclass(frozen=True)
class SeatResult:
    seat: int
    outcome: str  # 'blackjack' | 'win' | 'push' | 'lose' | 'bust' | 'surrender' | 'even_money'
    stake: int  # final stake (doubled if doubled)
    returned: int  # amount handed back to the seat, stake and insurance returns included; 0 on a total loss
    insurance: int = 0  # insurance cost the seat paid (already part of what it put in, not of stake)


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


def _dealer_draws(rules, dealer):
    total, soft = hand_total(dealer)
    return total < rules["stand_on"] or (rules["hit_soft_17"] and total == 17 and soft)


def _finish(state):
    """Dealer plays (only if some seat is still live), then the round ends."""
    dealer = state.dealer
    if any(s == "stand" for s in state.status) and not state.dealer_natural:
        while _dealer_draws(state.rules, dealer):
            card, state = _draw(replace(state, dealer=dealer))
            dealer = dealer + (card,)
    return replace(state, dealer=dealer, turn=None, finished=True)


def _advance(state, after):
    turn = _next_turn(state, after)
    if turn is None:
        return _finish(state)
    return replace(state, turn=turn)


def _peek(state):
    """The dealer checks the hole card: a natural ends the round, otherwise the seats play."""
    if state.dealer_natural:
        status = tuple("blackjack" if s == "blackjack" else "stand" for s in state.status)
        return replace(state, status=status, turn=None, finished=True, phase="play")
    return _advance(replace(state, phase="play"), 0)


def new(seed, seats: Sequence[Seat], rules=None):
    seats = tuple(seats)
    rules = normalize_rules(rules)
    if not 1 <= len(seats) <= MAX_SEATS:
        raise ValueError(f"blackjack needs 1 to {MAX_SEATS} seats")
    if [s.index for s in seats] != list(range(len(seats))):
        raise ValueError("seat indexes must be 0..n-1 in order")
    if any(s.stake <= 0 for s in seats):
        raise ValueError("blackjack seat stake must be a positive int")
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
        rules=rules, insured=(0,) * n, even_money=(False,) * n,
    )
    if rules["insurance"] and up % 13 == 0:
        return replace(state, phase="insurance", turn=0)
    return _peek(state)


def legal_moves(state, seat, can_double=True, can_insure=True):
    if state.finished or state.turn != seat:
        return ()
    if state.phase == "insurance":
        if state.status[seat] == "blackjack":
            return ("even_money", "no_insurance")
        if can_insure and state.stakes[seat] >= 2:
            return ("insure", "no_insurance")
        return ("no_insurance",)
    moves = ["hit", "stand"]
    if can_double and len(state.hands[seat]) == 2:
        moves.append("double")
    if state.rules["surrender"] and len(state.hands[seat]) == 2:
        moves.append("surrender")
    return tuple(moves)


def _apply_insurance(state, seat, move):
    insured, even = list(state.insured), list(state.even_money)
    if move == "insure":
        insured[seat] = state.stakes[seat] // 2
    elif move == "even_money":
        even[seat] = True
    state = replace(state, insured=tuple(insured), even_money=tuple(even))
    if seat + 1 < len(state.seats):
        return replace(state, turn=seat + 1)
    return _peek(replace(state, turn=None))


def apply(state, seat, move, can_double=True, can_insure=True):
    if state.finished:
        raise IllegalMove("the round is over")
    if state.turn != seat:
        raise IllegalMove(f"seat {seat} is not on turn")
    if move not in legal_moves(state, seat, can_double, can_insure):
        raise IllegalMove(f"{move!r} is not legal for seat {seat}")
    if state.phase == "insurance":
        return _apply_insurance(state, seat, move)
    status = list(state.status)
    if move == "stand":
        status[seat] = "stand"
        return _advance(replace(state, status=tuple(status)), seat + 1)
    if move == "surrender":
        status[seat] = "surrender"
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


def replay(seed, seats, moves, rules=None):
    """Rebuild a state from its seed, seats, rules and (seat, move) list; raises IllegalMove on a bad log."""
    state = new(seed, seats, rules)
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
        phase=state.phase,
        insured=state.insured,
        even_money=state.even_money,
        rules=dict(state.rules),
    )


def _settle(state, i):
    """(outcome, returned on the main bet); insurance returns are added in result()."""
    stake = state.stakes[i]
    st = state.status[i]
    rules = state.rules
    dealer_total = hand_total(state.dealer)[0]
    tie = ("lose", 0) if rules["ties"] == "dealer" else ("push", stake)
    if st == "surrender":
        return "surrender", stake // 2
    if st == "bust":
        return "bust", 0
    if state.even_money[i]:
        return "even_money", stake * 2
    if state.dealer_natural:
        return tie if st == "blackjack" else ("lose", 0)
    if st == "blackjack":
        return "blackjack", stake + (stake * 6 // 5 if rules["blackjack_pays"] == "6:5" else stake * 3 // 2)
    total = hand_total(state.hands[i])[0]
    if dealer_total > 21 or total > dealer_total:
        return "win", stake * 2
    if total == dealer_total:
        return tie
    return "lose", 0


def result(state):
    if not state.finished:
        raise IllegalMove("the round is not finished")
    seats = []
    lines = []
    for i in range(len(state.seats)):
        outcome, returned = _settle(state, i)
        cost = state.insured[i]
        if cost and state.dealer_natural:
            returned += cost * 3
        seats.append(SeatResult(i, outcome, state.stakes[i], returned, cost))
        note = f", insurance {'paid 2:1' if state.dealer_natural else 'lost'}" if cost else ""
        lines.append(f"Seat {i + 1}: {hand_str(state.hands[i])} ({total_str(state.hands[i])}) {outcome}{note}")
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
