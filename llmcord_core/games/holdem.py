"""No-limit Texas Hold'em rules engine, assist odds and fallback policy (FEAT-33 part A).

Pure functions over an immutable ``State``. Cards are (rank, suit) tuples (ranks 2..10, J, Q, K, A). A hand is fully
determined by (seed, seats, rules, button, moves). Moves: ``fold``, ``check``, ``call`` and ``raise:<n>`` where ``n``
is the seat's total bet for the street.
"""
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import combinations
from typing import Optional

from ..i18n import Phrase, UserError, codes, terms
from . import playstyle
from .base import IllegalMove, shuffled

RANKS = (2, 3, 4, 5, 6, 7, 8, 9, 10, "J", "Q", "K", "A")
SUITS = ("♠", "♥", "♦", "♣")
DECK = tuple((r, s) for s in SUITS for r in RANKS)
MIN_SEATS = 2
MAX_SEATS = 9
DEFAULT_RULES = {"small_blind": 5, "min_buy_in_bb": 40, "max_buy_in_bb": 200, "action_timeout": 45, "assist": False}
TIMEOUT_RANGE = (15, 300)
STREETS = ("preflop", "flop", "turn", "river")
BOARD_LEN = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
_VAL = {**{i: i for i in range(2, 11)}, "J": 11, "Q": 12, "K": 13, "A": 14}
_NAME = terms.POKER_RANK
_PLURAL = terms.POKER_RANK_PLURAL
CATEGORIES = terms.POKER_CATEGORIES
_RAISE = re.compile(r"raise:([1-9][0-9]{0,11})")


def normalize_rules(value=None):
    """The full rules dict for a stored/posted value (None or {} is the default table); ValueError on a bad one."""
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise UserError(codes.RULES_HOLDEM_OPTIONS)
    r = {**DEFAULT_RULES, **{k: v for k, v in value.items() if k in DEFAULT_RULES}}

    def whole(name):
        if type(r[name]) is not int:
            raise UserError(codes.RULES_HOLDEM_WHOLE, name=name)
        return r[name]

    if not 1 <= whole("small_blind") <= 10 ** 6:
        raise UserError(codes.RULES_HOLDEM_SMALL_BLIND)
    lo, hi = whole("min_buy_in_bb"), whole("max_buy_in_bb")
    if lo < 20 or hi > 1000 or lo > hi:
        raise UserError(codes.RULES_HOLDEM_BUY_IN)
    lo, hi = TIMEOUT_RANGE
    if not lo <= whole("action_timeout") <= hi:
        raise UserError(codes.RULES_HOLDEM_TIMEOUT, low=lo, high=hi)
    if type(r["assist"]) is not bool:
        raise UserError(codes.RULES_HOLDEM_ASSIST_FLAG)
    return {k: r[k] for k in DEFAULT_RULES}


def rules_phrases(rules=None):
    """The table's rules in plain words, one Phrase per part (display only); join with " · "."""
    r = normalize_rules(rules)
    sb = r["small_blind"]
    return [Phrase(terms.HOLDEM_RULE_BLINDS, small=sb, big=2 * sb),
            Phrase(terms.HOLDEM_RULE_BUY_IN, low=r["min_buy_in_bb"], high=r["max_buy_in_bb"]),
            Phrase(terms.HOLDEM_RULE_TIMEOUT, seconds=r["action_timeout"]),
            Phrase(terms.HOLDEM_RULE_ASSIST_ON if r["assist"] else terms.HOLDEM_RULE_ASSIST_OFF)]


def rules_text(rules=None):
    """The table's rules in plain words (display only), one line."""
    return " · ".join(str(p) for p in rules_phrases(rules))


def how_to_play_phrases(rules=None):
    """The how-to-play lines, one Phrase per line (display only); join with a newline."""
    r = normalize_rules(rules)
    sb = r["small_blind"]
    return [
        Phrase(terms.HOLDEM_HOW_TITLE),
        Phrase(terms.HOLDEM_HOW_GOAL),
        Phrase(terms.HOLDEM_HOW_BLINDS, small=sb, big=2 * sb),
        Phrase(terms.HOLDEM_HOW_STREETS),
        Phrase(terms.HOLDEM_HOW_RAISE),
        Phrase(terms.HOLDEM_HOW_SHOWDOWN),
        Phrase(terms.HOLDEM_HOW_SIDE_POT),
        Phrase(terms.HOLDEM_HOW_TIMEOUT_ASSIST if r["assist"] else terms.HOLDEM_HOW_TIMEOUT, seconds=r["action_timeout"]),
    ]


def how_to_play(rules=None):
    return "\n".join(str(p) for p in how_to_play_phrases(rules))


@dataclass(frozen=True)
class State:
    seed: str = field(repr=False)
    seats: tuple
    rules: dict
    button: int
    stacks: tuple
    bets: tuple
    contributed: tuple
    folded: tuple
    all_in: tuple
    holes: tuple = field(repr=False)
    board_full: tuple = field(repr=False)
    street: str
    turn: Optional[int]
    to_call: int
    min_raise_to: int
    history: tuple
    finished: bool
    acted: tuple = field(repr=False)
    last_raise: int = field(repr=False)
    shown_n: int = field(repr=False)
    aggressor: Optional[int] = field(repr=False)


@dataclass(frozen=True)
class View:
    seat: Optional[int]
    seats: tuple
    rules: dict
    button: int
    board: tuple
    pot: int
    stacks: tuple
    bets: tuple
    contributed: tuple
    folded: tuple
    all_in: tuple
    street: str
    turn: Optional[int]
    to_call: int
    level: int
    min_raise_to: int
    can_raise: bool
    history: tuple
    finished: bool
    hole: Optional[tuple] = field(default=None, repr=False)


@dataclass(frozen=True)
class Pot:
    amount: int
    eligible: tuple
    winners: tuple
    hand: Optional[tuple]


@dataclass(frozen=True)
class SeatResult:
    seat: int
    start: int
    end: int
    net: int
    outcome: Optional[str]  # 'win' | 'split' | 'lose' | 'fold'; None while the hand runs


@dataclass(frozen=True)
class Result:
    finished: bool
    board: tuple
    pots: tuple
    seats: tuple
    shown: tuple


@dataclass(frozen=True)
class Assist:
    current: str  # English; current_phrase is the same words for the edge to render
    chances: tuple  # (category, probability): the category is a Msg term, a str
    win: float
    current_phrase: Optional[Phrase] = None


# ---- hand evaluation -------------------------------------------------------------------------------------------

def _straight_high(values):
    if 14 in values:
        values = values | {1}
    for high in range(14, 4, -1):
        if all(high - k in values for k in range(5)):
            return high
    return 0


def evaluate(cards):
    """Comparable ``Rank`` tuple of 5..7 cards: (category 0..8, tiebreak ranks...); the wheel is the lowest straight."""
    cards = list(cards)
    if not 5 <= len(cards) <= 7:
        raise ValueError("evaluate takes 5 to 7 cards")
    vals = [_VAL[r] for r, _ in cards]
    flush = None
    by_suit = {}
    for (r, s) in cards:
        by_suit.setdefault(s, []).append(_VAL[r])
    for group in by_suit.values():
        if len(group) >= 5:
            flush = sorted(group, reverse=True)
    if flush:
        high = _straight_high(set(flush))
        if high:
            return (8, high)
    groups = sorted(Counter(vals).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    top, tcount = groups[0]
    if tcount == 4:
        return (7, top, max(v for v in vals if v != top))
    if tcount == 3 and len(groups) > 1 and groups[1][1] >= 2:
        return (6, top, groups[1][0])
    if flush:
        return (5, *flush[:5])
    high = _straight_high(set(vals))
    if high:
        return (4, high)
    if tcount == 3:
        return (3, top, *sorted((v for v in vals if v != top), reverse=True)[:2])
    if tcount == 2:
        if groups[1][1] == 2:
            second = groups[1][0]
            return (2, top, second, max(v for v in vals if v != top and v != second))
        return (1, top, *sorted((v for v in vals if v != top), reverse=True)[:3])
    return (0, *sorted(vals, reverse=True)[:5])


def best_hand(cards):
    """The five cards that make the best hand."""
    cards = tuple(cards)
    rank = evaluate(cards)
    return next(c for c in combinations(cards, 5) if evaluate(c) == rank)


def hand_phrase(rank):
    """The hand's name as words for the edge: a Phrase whose rank params are Msg terms."""
    cat = rank[0]
    if cat == 0:
        return Phrase(terms.HAND_HIGH_CARD, rank=_NAME[rank[1]])
    if cat == 1:
        return Phrase(terms.HAND_PAIR, rank=_PLURAL[rank[1]])
    if cat == 2:
        return Phrase(terms.HAND_TWO_PAIR, high=_PLURAL[rank[1]], low=_PLURAL[rank[2]])
    if cat == 3:
        return Phrase(terms.HAND_THREE_OF_A_KIND, rank=_PLURAL[rank[1]])
    if cat == 4:
        return Phrase(terms.HAND_STRAIGHT, rank=_NAME[rank[1]])
    if cat == 5:
        return Phrase(terms.HAND_FLUSH, rank=_NAME[rank[1]])
    if cat == 6:
        return Phrase(terms.HAND_FULL_HOUSE, trips=_PLURAL[rank[1]], pair=_PLURAL[rank[2]])
    if cat == 7:
        return Phrase(terms.HAND_FOUR_OF_A_KIND, rank=_PLURAL[rank[1]])
    return Phrase(terms.HAND_ROYAL_FLUSH) if rank[1] == 14 else Phrase(terms.HAND_STRAIGHT_FLUSH, rank=_NAME[rank[1]])


def hand_name(rank):
    return str(hand_phrase(rank))


# ---- betting ---------------------------------------------------------------------------------------------------

def _bb(rules):
    return 2 * rules["small_blind"]


def _level(s):
    top = max(s.bets)
    if s.street != "preflop" or sum(1 for i in range(len(s.seats)) if not s.folded[i] and not s.all_in[i]) < 2:
        return top
    return max(top, _bb(s.rules))


def _menu(bet, stack, level, min_to, can_raise, pot):
    out = ["check"] if bet >= level else ["fold", "call"]
    hi = bet + stack
    if can_raise and stack > 0 and hi > level:
        if hi < min_to:
            amounts = [hi]
        else:
            full = pot + max(0, level - bet)
            amounts = sorted(a for a in {min_to, hi, level + full // 2, level + full} if min_to <= a <= hi)
        out += [f"raise:{a}" for a in amounts]
    return out


def _can_raise(s, seat):
    return not s.acted[seat] and any(
        j != seat and not s.folded[j] and not s.all_in[j] for j in range(len(s.seats)))


def legal_moves(state, seat):
    if state.finished or seat != state.turn:
        return []
    level = _level(state)
    return _menu(state.bets[seat], state.stacks[seat], level, state.min_raise_to, _can_raise(state, seat),
                 sum(state.contributed))


def _next_actor(s, after):
    n = len(s.seats)
    can = [i for i in range(n) if not s.folded[i] and not s.all_in[i]]
    level = _level(s)
    for k in range(1, n + 1):
        j = (after + k) % n
        if j in can and (s.bets[j] < level or (not s.acted[j] and len(can) >= 2)):
            return j
    return None


def _settle(s, after):
    n = len(s.seats)
    while True:
        live = [i for i in range(n) if not s.folded[i]]
        if len(live) == 1:
            return replace(s, street="done", turn=None, to_call=0, min_raise_to=0, finished=True)
        nxt = _next_actor(s, after)
        if nxt is not None:
            level = _level(s)
            return replace(s, turn=nxt, to_call=min(level - s.bets[nxt], s.stacks[nxt]),
                           min_raise_to=level + s.last_raise)
        if s.street == "river":
            return replace(s, street="showdown", turn=None, to_call=0, min_raise_to=0, finished=True)
        street = STREETS[STREETS.index(s.street) + 1]
        s = replace(s, street=street, shown_n=BOARD_LEN[street])
        if sum(1 for i in live if not s.all_in[i]) >= 2:
            s = replace(s, bets=(0,) * n, acted=(False,) * n, last_raise=_bb(s.rules), aggressor=None)
        after = s.button


def new(seed, seats, rules=None, button=0):
    seats = tuple(sorted(seats, key=lambda x: x.index))
    n = len(seats)
    if not MIN_SEATS <= n <= MAX_SEATS:
        raise ValueError(f"A Hold'em table needs {MIN_SEATS} to {MAX_SEATS} seats.")
    if [x.index for x in seats] != list(range(n)):
        raise ValueError("Seat indexes must be 0..n-1 without gaps or duplicates.")
    if any(x.stake <= 0 for x in seats):
        raise ValueError("Every seat needs a positive stack.")
    if type(button) is not int or not 0 <= button < n:
        raise ValueError("The button must be a seat index.")
    rules = normalize_rules(rules)
    deck = shuffled(DECK, str(seed))
    holes = [None] * n
    for i in range(n):
        seat = (button + 1 + i) % n
        holes[seat] = (deck[i], deck[n + i])
    sb, bb = (button, (button + 1) % n) if n == 2 else ((button + 1) % n, (button + 2) % n)
    stacks = [x.stake for x in seats]
    bets = [0] * n
    for seat, amount in ((sb, rules["small_blind"]), (bb, _bb(rules))):
        bets[seat] = min(amount, stacks[seat])
        stacks[seat] -= bets[seat]
    state = State(
        seed=str(seed), seats=seats, rules=rules, button=button, stacks=tuple(stacks), bets=tuple(bets),
        contributed=tuple(bets), folded=(False,) * n, all_in=tuple(x == 0 for x in stacks),
        holes=tuple(holes), board_full=tuple(deck[2 * n:2 * n + 5]), street="preflop", turn=None, to_call=0,
        min_raise_to=0, history=(), finished=False, acted=(False,) * n, last_raise=_bb(rules), shown_n=0,
        aggressor=None,
    )
    return _settle(state, bb)


def apply(state, seat, move):
    if state.finished:
        raise IllegalMove("The hand is over.")
    if seat != state.turn:
        raise IllegalMove("It is not that seat's turn.")
    if not isinstance(move, str):
        raise IllegalMove("Unknown move.")
    n = len(state.seats)
    legal = legal_moves(state, seat)
    level = _level(state)
    stacks, bets, contributed = list(state.stacks), list(state.bets), list(state.contributed)
    folded, all_in, acted = list(state.folded), list(state.all_in), list(state.acted)
    last_raise, aggressor = state.last_raise, state.aggressor
    if move in ("fold", "check", "call"):
        if move not in legal:
            raise IllegalMove(f"{move} is not legal now.")
        if move == "fold":
            folded[seat] = True
        elif move == "check":
            acted[seat] = True
        else:
            amount = min(level - bets[seat], stacks[seat])
            stacks[seat] -= amount
            bets[seat] += amount
            contributed[seat] += amount
            acted[seat] = True
    else:
        m = _RAISE.fullmatch(move)
        if not m or not any(x.startswith("raise:") for x in legal):
            raise IllegalMove(f"{move} is not legal now.")
        target = int(m.group(1))
        hi = bets[seat] + stacks[seat]
        if not (state.min_raise_to <= target <= hi or target == hi):
            raise IllegalMove(f"A raise must be to {state.min_raise_to}..{hi}.")
        amount = target - bets[seat]
        stacks[seat] -= amount
        bets[seat] = target
        contributed[seat] += amount
        increase = target - level
        if increase >= last_raise:
            last_raise = increase
            acted = [False] * n
        acted[seat] = True
        aggressor = seat
    if stacks[seat] == 0 and not folded[seat]:
        all_in[seat] = True
    new_state = replace(
        state, stacks=tuple(stacks), bets=tuple(bets), contributed=tuple(contributed), folded=tuple(folded),
        all_in=tuple(all_in), acted=tuple(acted), last_raise=last_raise, aggressor=aggressor,
        history=state.history + ((seat, move),),
    )
    settled = _settle(new_state, seat)
    if settled.finished and settled.street == "showdown" and state.street != "showdown":
        settled = replace(settled, bets=(0,) * n)
    return settled


def replay(seed, seats, moves, rules=None, button=0):
    state = new(seed, seats, rules, button)
    for seat, move in moves:
        state = apply(state, seat, move)
    return state


def view(state, seat=None):
    n = len(state.seats)
    hole = state.holes[seat] if seat is not None and 0 <= seat < n else None
    level = _level(state)
    turn = state.turn
    return View(
        seat=seat, seats=state.seats, rules=state.rules, button=state.button,
        board=state.board_full[:state.shown_n], pot=sum(state.contributed), stacks=state.stacks, bets=state.bets,
        contributed=state.contributed, folded=state.folded, all_in=state.all_in, street=state.street, turn=turn,
        to_call=state.to_call, level=level, min_raise_to=state.min_raise_to,
        can_raise=turn is not None and _can_raise(state, turn), history=state.history, finished=state.finished,
        hole=hole,
    )


def legal_moves_for_view(v):
    """The viewing seat's menu, from its own view only (empty when it is not that seat's turn)."""
    if v.finished or v.seat is None or v.seat != v.turn:
        return []
    return _menu(v.bets[v.seat], v.stacks[v.seat], v.level, v.min_raise_to, v.can_raise, v.pot)


def timeout_move_for_view(v):
    seat = v.seat if v.seat is not None else v.turn
    return "check" if seat is not None and v.bets[seat] >= v.level else "fold"


def result(state):
    n = len(state.seats)
    start = [x.stake for x in state.seats]
    board = state.board_full[:state.shown_n]
    if not state.finished:
        return Result(False, board, (), tuple(SeatResult(i, start[i], start[i], 0, None) for i in range(n)), ())
    c = list(state.contributed)
    order = sorted(range(n), key=lambda i: -c[i])
    refund = [0] * n
    if c[order[0]] > c[order[1]]:
        refund[order[0]] = c[order[0]] - c[order[1]]
        c[order[0]] = c[order[1]]
    live = [i for i in range(n) if not state.folded[i]]
    layers, prev, carry = [], 0, 0
    for lv in sorted({x for x in c if x > 0}):
        amount = sum(min(x, lv) - min(x, prev) for x in c) + carry
        prev, carry = lv, 0
        eligible = tuple(i for i in live if c[i] >= lv)
        if not eligible:
            carry = amount
        elif layers and layers[-1][1] == eligible:
            layers[-1][0] += amount
        else:
            layers.append([amount, eligible])
    if carry and layers:
        layers[-1][0] += carry
    from_button = [(state.button + 1 + k) % n for k in range(n)]
    ranks = {i: evaluate(state.holes[i] + state.board_full) for i in live} if len(live) > 1 else {}
    pots, payout, won, shared = [], [0] * n, set(), set()
    for amount, eligible in layers:
        if ranks:
            best = max(ranks[i] for i in eligible)
            winners = tuple(i for i in from_button if i in eligible and ranks[i] == best)
        else:
            best, winners = None, eligible
        share, extra = divmod(amount, len(winners))
        for k, w in enumerate(winners):
            payout[w] += share + (1 if k < extra else 0)
        won.update(winners)
        if len(winners) > 1:
            shared.update(winners)
        pots.append(Pot(amount, eligible, winners, best))
    seats = []
    for i in range(n):
        end = state.stacks[i] + refund[i] + payout[i]
        outcome = ("fold" if state.folded[i] else "split" if i in shared else "win" if i in won and end > start[i]
                   else "lose")
        seats.append(SeatResult(i, start[i], end, end - start[i], outcome))
    shown = ()
    if len(live) > 1:
        first = state.aggressor if state.aggressor in live else (state.button + 1) % n
        shown = tuple(j for j in ((first + k) % n for k in range(n)) if j in live)
    return Result(True, board, tuple(pots), tuple(seats), shown)


# ---- assist ----------------------------------------------------------------------------------------------------

def _made_phrase(cards):
    if len(cards) >= 5:
        return hand_phrase(evaluate(cards))
    vals = sorted((_VAL[r] for r, _ in cards), reverse=True)
    if len(vals) == 2 and vals[0] == vals[1]:
        return Phrase(terms.HAND_PAIR, rank=_PLURAL[vals[0]])
    return Phrase(terms.HAND_HIGH_CARD, rank=_NAME[vals[0]])


def assist(hole, board, opponents, seed, samples=2000):
    """The made hand now, the three likeliest final categories and the chance to win against random hands."""
    hole, board = tuple(hole), tuple(board)
    if len(hole) != 2 or len(board) not in (0, 3, 4, 5):
        raise ValueError("assist needs two hole cards and a board of 0, 3, 4 or 5 cards")
    if type(opponents) is not int or opponents < 1:
        raise ValueError("opponents must be at least 1")
    known = set(hole) | set(board)
    rest = [c for c in DECK if c not in known]
    if len(known) != len(hole) + len(board) or len(rest) < 5 - len(board) + 2 * opponents:
        raise ValueError("duplicate cards or too many opponents")
    key = f"{seed}/{sorted(map(str, hole))}/{sorted(map(str, board))}"
    missing = 5 - len(board)
    counts = Counter()
    if len(board) >= 3:
        for extra in combinations(rest, missing):
            counts[evaluate(hole + board + extra)[0]] += 1
    else:
        rng = random.Random(f"{key}/chances")
        for _ in range(samples):
            counts[evaluate(hole + tuple(rng.sample(rest, 5)))[0]] += 1
    total = sum(counts.values())
    top = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))[:3]
    probs = [round(n / total, 3) for _, n in top]
    while sum(probs) > 1 + 1e-9:
        k = max(range(len(top)), key=lambda i: probs[i] - top[i][1] / total)
        probs[k] = round(probs[k] - 0.001, 3)
    chances = tuple((CATEGORIES[cat], p) for (cat, _), p in zip(top, probs))
    rng = random.Random(f"{key}/win")
    score = 0.0
    for _ in range(samples):
        draw = rng.sample(rest, missing + 2 * opponents)
        full = board + tuple(draw[:missing])
        mine = evaluate(hole + full)
        theirs = [evaluate(tuple(draw[missing + 2 * k:missing + 2 * k + 2]) + full) for k in range(opponents)]
        best = max(theirs)
        if mine > best:
            score += 1
        elif mine == best:
            score += 1 / (1 + theirs.count(best))
    made = _made_phrase(hole + board)
    return Assist(str(made), chances, round(score / samples, 3), made)


# ---- fallback policy -------------------------------------------------------------------------------------------

def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _chen(hole):
    a, b = sorted((_VAL[hole[0][0]], _VAL[hole[1][0]]), reverse=True)
    score = {14: 10, 13: 8, 12: 7, 11: 6}.get(a, a / 2)
    if a == b:
        return max(5, score * 2)
    if hole[0][1] == hole[1][1]:
        score += 2
    gap = a - b - 1
    score -= (0, 1, 2, 4, 5)[min(gap, 4)]
    if gap <= 1 and a < 12:
        score += 1
    return score


@lru_cache(maxsize=4096)
def _postflop(hole, board):
    """Made-hand strength 0..1 plus a share of the draw outs."""
    rank = evaluate(hole + board)
    cat = rank[0]
    hv = [_VAL[r] for r, _ in hole]
    bv = sorted((_VAL[r] for r, _ in board), reverse=True)
    if len(board) == 5 and evaluate(board) == rank:
        base = 0.2 if cat < 4 else 0.3
    elif cat >= 7:
        base = 0.98
    elif cat == 6:
        base = 0.95
    elif cat == 5:
        base = 0.9
    elif cat == 4:
        base = 0.85
    elif cat == 3:
        base = 0.8 if rank[1] in hv else 0.4
    elif cat == 2:
        base = {2: 0.72, 1: 0.55, 0: 0.28}[sum(1 for p in rank[1:3] if p in hv)]
    elif cat == 1:
        p = rank[1]
        if hv[0] == hv[1]:
            base = 0.7 if p > bv[0] else 0.45 if p > bv[1] else 0.35
        elif p in hv:
            kicker = hv[0] + hv[1] - p
            base = 0.55 + 0.1 * (kicker - 2) / 12 if p >= bv[0] else 0.4 if p >= bv[1] else 0.3
        else:
            base = 0.12 + 0.1 * (max(hv) - 2) / 12
    else:
        base = 0.06 + 0.12 * (max(hv) - 2) / 12 + 0.03 * sum(1 for v in hv if v > bv[0])
    if len(board) < 5 and cat < 5:
        known = hv + bv
        outs = 0
        suits = Counter(s for _, s in hole + board)
        if any(n == 4 and any(s == h[1] for h in hole) for s, n in suits.items()):
            outs += 9
        have, board_only = set(known), set(bv)
        if not _straight_high(have):
            for v in range(2, 15):
                if _straight_high(have | {v}) and not _straight_high(board_only | {v}):
                    outs += 4 - known.count(v)
        eq = min(0.6, outs * (0.04 if len(board) == 3 else 0.02))
        base += (1 - base) * eq * 0.85
    return _clamp(base)


def policy_action(v, style, seed):
    """One legal move for the viewing seat from its own view; every random draw comes from the secret round seed."""
    legal = legal_moves_for_view(v)
    if not legal:
        raise IllegalMove("It is not that seat's turn.")
    style = style or playstyle.PlayStyle()
    seat = v.seat
    tag = f"{v.street}/{len(v.history)}"
    L, A, B, S, R = (style.unit(t) for t in ("looseness", "aggression", "bluffing", "stubbornness", "risk"))
    opponents = max(1, sum(1 for i, f in enumerate(v.folded) if not f and i != seat))
    raw = _clamp((_chen(v.hole) + 3) / 22) if not v.board else _postflop(tuple(v.hole), tuple(v.board))
    s = playstyle.misjudge(raw ** (1 + 0.1 * (opponents - 1)), style, seed, tag)
    stack, bet = v.stacks[seat], v.bets[seat]
    free = "check" in legal
    odds = 0 if free else v.to_call / (v.pot + v.to_call)
    invested = v.contributed[seat] / (v.contributed[seat] + stack)
    need = 0.40 - 0.40 * L + 0.5 * odds * (1 - 0.5 * L) - 0.3 * S * invested
    need += 0.4 * min(1, v.to_call / stack) * (1 - R) if not free else 0
    raises = [m for m in legal if m.startswith("raise:")]
    bluff = s < need and playstyle.roll(seed, f"bluff/{tag}") < 0.25 * B * (0.4 + 0.6 * A)
    if raises and (s >= 0.9 - 0.45 * A or bluff):
        cap = 0.15 + 0.7 * R + 0.3 * s * s
        fits = [m for m in raises if int(m[6:]) - bet <= cap * stack] or raises[:1]
        u = _clamp(0.5 * A + 0.8 * (s - 0.5) + 0.25)
        move = fits[round((len(fits) - 1) * u)]
    elif free:
        move = "check"
    else:
        move = "call" if s >= need else "fold"
    if playstyle.erratic(style, seed, tag):
        others = [m for m in legal if m != move]
        if others:
            move = others[int(playstyle.roll(seed, f"pick/{tag}") * len(others))]
    move = playstyle.never_fold_free(legal, move)
    return move if move in legal else legal[0]
