"""Blackjack rules engine: six-deck shoe, hit/stand/double, optional house rules (see normalize_rules).

Pure functions over an immutable ``State``. Cards are ints 0..51: rank = card % 13 (0 = ace,
9..12 = ten-value), suit = card // 13. A round is fully determined by (seed, seats, rules, moves).
"""
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

from ..i18n import Phrase, UserError, codes, terms
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
        raise UserError(codes.RULES_BLACKJACK_OPTIONS)
    unknown = sorted(str(k) for k in value if k not in CLASSIC_RULES)
    if unknown:
        raise UserError(codes.RULES_BLACKJACK_UNKNOWN, names=", ".join(unknown))
    r = {**CLASSIC_RULES, **value}
    if type(r["stand_on"]) is not int or r["stand_on"] not in (16, 17, 18):
        raise UserError(codes.RULES_BLACKJACK_STAND_ON)
    for key, code in (("hit_soft_17", codes.RULES_BLACKJACK_SOFT_17_FLAG), ("insurance", codes.RULES_BLACKJACK_INSURANCE_FLAG),
                      ("surrender", codes.RULES_BLACKJACK_SURRENDER_FLAG)):
        if type(r[key]) is not bool:
            raise UserError(code)
    if r["blackjack_pays"] not in ("3:2", "6:5"):
        raise UserError(codes.RULES_BLACKJACK_PAYS)
    if r["ties"] not in ("push", "dealer"):
        raise UserError(codes.RULES_BLACKJACK_TIES)
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


def rules_phrases(rules=None):
    """The table's rules in plain words, one Phrase per part (display only); join with " · "."""
    r = _display_rules(rules)
    if r["hit_soft_17"]:
        dealer = Phrase(terms.BJ_RULE_DEALER_HITS_S17)
    elif r["stand_on"] == 17:
        dealer = Phrase(terms.BJ_RULE_DEALER_STANDS_ALL_17)
    else:
        dealer = Phrase(terms.BJ_RULE_DEALER_STANDS_ON, total=r["stand_on"])
    parts = [dealer, Phrase(terms.BJ_RULE_PAYS, ratio=r["blackjack_pays"]),
             Phrase(terms.BJ_RULE_TIES_DEALER if r["ties"] == "dealer" else terms.BJ_RULE_TIES_PUSH),
             Phrase(terms.BJ_RULE_DOUBLE if r["double"] else terms.BJ_RULE_NO_DOUBLE),
             Phrase(terms.BJ_RULE_SPLIT if r["split"] else terms.BJ_RULE_NO_SPLIT)]
    if r["insurance"]:
        parts.append(Phrase(terms.BJ_RULE_INSURANCE))
    if r["surrender"]:
        parts.append(Phrase(terms.BJ_RULE_SURRENDER))
    parts.append(Phrase(terms.BJ_RULE_DECK_ONE if r["decks"] == 1 else terms.BJ_RULE_DECKS, count=r["decks"]))  # engine plural, branch-chosen
    return parts


def rules_text(rules=None):
    """The table's rules in plain words (display only)."""
    return " · ".join(str(p) for p in rules_phrases(rules))


def how_to_play_phrases(rules=None):
    """The how-to-play lines, one Phrase per line (display only); join with a newline."""
    r = _display_rules(rules)
    lines = [
        Phrase(terms.BJ_HOW_TITLE),
        Phrase(terms.BJ_HOW_GOAL),
        Phrase(terms.BJ_HOW_CARDS),
        Phrase(terms.BJ_HOW_HIT_STAND_DOUBLE if r["double"] else terms.BJ_HOW_HIT_STAND),
        Phrase(terms.BJ_HOW_BUST, ratio=r["blackjack_pays"]),
        Phrase(terms.BJ_HOW_DEALER_SOFT_17 if r["hit_soft_17"] else terms.BJ_HOW_DEALER, stand_on=r["stand_on"]),
        Phrase(terms.BJ_HOW_WIN_TIE_DEALER if r["ties"] == "dealer" else terms.BJ_HOW_WIN_TIE_PUSH),
    ]
    if r["insurance"]:
        lines.append(Phrase(terms.BJ_HOW_INSURANCE))
    if r["surrender"]:
        lines.append(Phrase(terms.BJ_HOW_SURRENDER))
    return lines


def how_to_play(rules=None):
    return "\n".join(str(p) for p in how_to_play_phrases(rules))


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


def total_phrase(cards):
    """The hand's total for a message param: a Phrase when soft, else the number as plain text."""
    total, soft = hand_total(cards)
    return Phrase(terms.BJ_TOTAL_SOFT, total=total) if soft else str(total)


def total_str(cards):
    return str(total_phrase(cards))


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
    summary_phrases: tuple = ()  # the summary lines as Phrases (English join is `summary`)


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
        line = terms.BJ_SUMMARY_SEAT
        if cost:
            line = terms.BJ_SUMMARY_SEAT_INSURANCE_PAID if state.dealer_natural else terms.BJ_SUMMARY_SEAT_INSURANCE_LOST
        lines.append(Phrase(line, seat=i + 1, hand=hand_str(state.hands[i]), total=total_phrase(state.hands[i]), outcome=terms.BJ_OUTCOMES[outcome]))
    dealer_total = hand_total(state.dealer)[0]
    head = terms.BJ_SUMMARY_DEALER
    if dealer_total > 21:
        head = terms.BJ_SUMMARY_DEALER_BUST
    elif state.dealer_natural:
        head = terms.BJ_SUMMARY_DEALER_BLACKJACK
    phrases = (Phrase(head, hand=hand_str(state.dealer), total=total_phrase(state.dealer)), *lines)
    return Result(
        seats=tuple(seats), dealer=state.dealer, dealer_total=dealer_total,
        seed=state.seed, seed_hash=short_seed_hash(state.seed),
        summary="\n".join(str(p) for p in phrases), summary_phrases=phrases,
    )
