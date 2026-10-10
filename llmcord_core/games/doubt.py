"""I Doubt It rules engine and fallback policy (FEAT-21, D29).

Pure functions over an immutable ``State``. Cards are (rank, suit) tuples. A game is fully determined by
(seed, seats, rules, moves). Moves: ``play:<rank>x<n>[,<rank>x<n>...]`` (ranks in RANKS order), ``doubt``,
``close`` (the claimer ends the doubt window) and ``forfeit`` (a lifecycle action, not listed by legal_moves).
"""
import json
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

from .base import IllegalMove, Seat, seed_hash, shuffled

RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
SUITS = ("♠", "♥", "♦", "♣")
MIN_SEATS = 3
MAX_SEATS = 8
MAX_PLAY = 4
DEFAULT_RULES = {"claim_rule": "sequence", "ante": 0, "turn_timeout": 60}
ANTE_MAX = 10 ** 9
TIMEOUT_RANGE = (15, 300)
TRIAGE_MIN = 0.3  # suspicion a seat needs before it is worth a model call
DOUBT_BELOW = 0.3  # the fallback policy never doubts under this suspicion


def normalize_rules(value=None):
    """The full rules dict for a stored/posted value (None or {} is the default table); ValueError on a bad one."""
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("The I Doubt It rules must be a set of options.")
    r = {**DEFAULT_RULES, **{k: v for k, v in value.items() if k in DEFAULT_RULES}}
    if r["claim_rule"] != "sequence":
        raise ValueError("The claim rule must be sequence.")
    if type(r["ante"]) is not int or not 0 <= r["ante"] <= ANTE_MAX:
        raise ValueError(f"The ante must be a whole number from 0 to {ANTE_MAX}.")
    lo, hi = TIMEOUT_RANGE
    if type(r["turn_timeout"]) is not int or not lo <= r["turn_timeout"] <= hi:
        raise ValueError(f"The turn timeout must be a whole number of seconds from {lo} to {hi}.")
    return {k: r[k] for k in DEFAULT_RULES}


def rules_text(rules=None):
    """The table's rules in plain words (display only), one line."""
    r = normalize_rules(rules)
    ante = f"Ante {r['ante']}" if r["ante"] else "No ante"
    return f"Claims follow the sequence A to K · {ante} · {r['turn_timeout']} s per turn"


def how_to_play(rules=None):
    r = normalize_rules(rules)
    return "\n".join([
        "**I Doubt It in short**",
        "Be the first to get rid of all your cards.",
        "• On your turn, put 1 to 4 cards face down on the pile and claim they are all the forced rank. The rank goes A, 2, 3 ... K, then A again, one step per claim.",
        "• You can't pass. If you hold none of the rank, you bluff.",
        "• Anyone else can doubt the claim. The cards are shown: if any is the wrong rank, the claimer takes the whole pile. If the claim was true, the doubter takes it.",
        "• If nobody doubts and the next player plays, the claim stands.",
        "• Empty your hand and survive the doubt window to win" + (f" the pot ({r['ante']} each)." if r["ante"] else "."),
        f"• Take longer than {r['turn_timeout']} s and a card is played for you.",
    ])


@dataclass(frozen=True)
class Window:
    claimer: int
    count: int
    rank: str


@dataclass(frozen=True)
class Play:
    seat: int
    count: int
    rank: str
    kind: str = "play"


@dataclass(frozen=True)
class Reveal:
    claimer: int
    doubter: int
    rank: str
    cards: tuple
    bluff: bool
    kind: str = "reveal"


@dataclass(frozen=True)
class State:
    seed: str = field(repr=False)
    seats: tuple
    rules: dict
    decks: int
    hands: tuple = field(repr=False)  # per seat: tuple of cards in hand order
    pile: tuple = field(repr=False)
    active: tuple
    rank_idx: int
    turn: Optional[int]
    window: Optional[tuple] = None  # (claimer, rank, cards) of the open claim
    history: tuple = ()
    finished: bool = False
    winner: Optional[int] = None


@dataclass(frozen=True)
class View:
    seat: Optional[int]
    counts: tuple
    active: tuple
    turn: Optional[int]
    rank: str
    pile: int
    window: Optional[Window]
    history: tuple
    finished: bool
    winner: Optional[int]
    decks: int
    hand: Optional[tuple] = None


@dataclass(frozen=True)
class SeatResult:
    seat: int
    outcome: Optional[str]  # 'win' | 'lose' | 'forfeit'; None while the game runs
    payout: int


@dataclass(frozen=True)
class Result:
    finished: bool
    winner: Optional[int]
    seats: tuple
    pot: int


def _sort_key(card):
    return RANKS.index(card[0]), SUITS.index(card[1])


def _next_active(active, after):
    n = len(active)
    for k in range(1, n + 1):
        j = (after + k) % n
        if active[j]:
            return j
    return None


def new(seed, seats: Sequence[Seat], rules=None):
    seats = tuple(seats)
    rules = normalize_rules(rules)
    n = len(seats)
    if not MIN_SEATS <= n <= MAX_SEATS:
        raise ValueError(f"I Doubt It needs {MIN_SEATS} to {MAX_SEATS} seats")
    if [s.index for s in seats] != list(range(n)):
        raise ValueError("seat indexes must be 0..n-1 in order")
    decks = 1 if n <= 5 else 2
    deck = shuffled([(r, s) for s in SUITS for r in RANKS] * decks, seed)
    hands = tuple(tuple(deck[i::n]) for i in range(n))
    return State(seed=seed, seats=seats, rules=rules, decks=decks, hands=hands, pile=(), active=(True,) * n,
                 rank_idx=0, turn=0)


def _plays(hand):
    """Every canonical play of 1..4 cards from a hand, as (move, ranks-counts) in a stable order."""
    held = {}
    for c in hand:
        held[c[0]] = held.get(c[0], 0) + 1
    ranks = [r for r in RANKS if r in held]
    out = []

    def walk(i, left, picked):
        if i == len(ranks):
            if picked:
                out.append("play:" + ",".join(f"{r}x{n}" for r, n in picked))
            return
        r = ranks[i]
        for n in range(0, min(held[r], left) + 1):
            walk(i + 1, left - n, picked + [(r, n)] if n else picked)

    walk(0, MAX_PLAY, [])
    return tuple(out)


def _can_play(state, seat):
    if state.finished or not state.active[seat] or state.turn != seat or not state.hands[seat]:
        return False
    w = state.window
    return w is None or not (len(state.hands[w[0]]) == 0)


def legal_moves(state, seat):
    """Moves the seat may make now. ``forfeit`` is always allowed for an active seat but is not listed."""
    if state.finished or type(seat) is not int or not 0 <= seat < len(state.seats) or not state.active[seat]:
        return ()
    moves = []
    w = state.window
    if w is not None:
        moves.append("close" if seat == w[0] else "doubt")
    if _can_play(state, seat):
        moves.extend(_plays(state.hands[seat]))
    return tuple(moves)


def _parse_play(move):
    """[(rank, n)] for a canonical play string; IllegalMove otherwise."""
    if not move.startswith("play:"):
        raise IllegalMove(f"unknown move {move!r}")
    parts = []
    for part in move[5:].split(","):
        rank, sep, n = part.rpartition("x")
        if not sep or rank not in RANKS or n not in ("1", "2", "3", "4"):
            raise IllegalMove(f"bad play {move!r}")
        parts.append((rank, int(n)))
    ranks = [RANKS.index(r) for r, _ in parts]
    if ranks != sorted(set(ranks)) or sum(n for _, n in parts) > MAX_PLAY:
        raise IllegalMove(f"bad play {move!r}")
    return parts


def _take(hand, parts):
    """(remaining hand, taken cards): the first matching cards in hand order."""
    need = dict(parts)
    left, taken = [], []
    for c in hand:
        if need.get(c[0], 0) > 0:
            need[c[0]] -= 1
            taken.append(c)
        else:
            left.append(c)
    if any(need.values()):
        raise IllegalMove("you don't hold those cards")
    return tuple(left), tuple(taken)


def _set(tup, i, value):
    return tup[:i] + (value,) + tup[i + 1:]


def _finish(state, winner):
    return replace(state, finished=True, winner=winner, turn=None, window=None)


def apply(state, seat, move):
    if state.finished:
        raise IllegalMove("the game is over")
    if type(seat) is not int or not 0 <= seat < len(state.seats):
        raise IllegalMove("no such seat")
    if not isinstance(move, str):
        raise IllegalMove("bad move")
    if not state.active[seat]:
        raise IllegalMove("that seat has left the game")
    w = state.window
    if move == "forfeit":
        return _forfeit(state, seat)
    if move == "close":
        if w is None or seat != w[0]:
            raise IllegalMove("nothing to close")
        if not state.hands[seat]:
            return _finish(state, seat)
        return replace(state, window=None)
    if move == "doubt":
        if w is None or seat == w[0]:
            raise IllegalMove("nothing to doubt")
        return _doubt(state, seat)
    if not _can_play(state, seat):
        raise IllegalMove("not your turn to play")
    parts = _parse_play(move)
    rest, taken = _take(state.hands[seat], parts)
    rank = RANKS[state.rank_idx % len(RANKS)]
    return replace(
        state, hands=_set(state.hands, seat, rest), pile=state.pile + taken, rank_idx=state.rank_idx + 1,
        turn=_next_active(state.active, seat), window=(seat, rank, taken),
        history=state.history + (Play(seat, len(taken), rank),),
    )


def _doubt(state, doubter):
    claimer, rank, cards = state.window
    bluff = any(c[0] != rank for c in cards)
    history = state.history + (Reveal(claimer, doubter, rank, tuple(sorted(cards, key=_sort_key)), bluff),)
    state = replace(state, history=history)
    if not bluff and not state.hands[claimer]:
        return _finish(state, claimer)
    loser = claimer if bluff else doubter
    hands = _set(state.hands, loser, state.hands[loser] + state.pile)
    return replace(state, hands=hands, pile=(), window=None, turn=_next_active(state.active, claimer))


def _forfeit(state, seat):
    active = _set(state.active, seat, False)
    state = replace(state, active=active, hands=_set(state.hands, seat, ()))
    alive = [i for i, a in enumerate(active) if a]
    if len(alive) == 1:
        return _finish(state, alive[0])
    window = None if state.window and state.window[0] == seat else state.window
    turn = _next_active(active, seat) if state.turn == seat else state.turn
    return replace(state, window=window, turn=turn)


def replay(seed, seats, moves, rules=None):
    state = new(seed, seats, rules)
    for seat, move in moves:
        state = apply(state, seat, move)
    return state


def view(state, seat=None):
    if seat is not None and not 0 <= seat < len(state.seats):
        raise ValueError("no such seat")
    w = state.window
    return View(
        seat=seat, counts=tuple(len(h) for h in state.hands), active=state.active, turn=state.turn,
        rank=RANKS[state.rank_idx % len(RANKS)], pile=len(state.pile),
        window=Window(w[0], len(w[2]), w[1]) if w else None, history=state.history,
        finished=state.finished, winner=state.winner, decks=state.decks,
        hand=None if seat is None else tuple(sorted(state.hands[seat], key=_sort_key)),
    )


def result(state):
    pot = sum(s.stake for s in state.seats)
    seats = []
    for i in range(len(state.seats)):
        if not state.active[i]:
            outcome = "forfeit"
        elif not state.finished:
            outcome = None
        else:
            outcome = "win" if i == state.winner else "lose"
        seats.append(SeatResult(i, outcome, pot if outcome == "win" else 0))
    return Result(state.finished, state.winner, tuple(seats), pot)


def timeout_move(state, seat, seed):
    """What a member's turn timeout plays: the forced rank once if held, else one random card as a bluff."""
    if not _can_play(state, seat):
        raise IllegalMove("that seat is not on turn")
    rank = RANKS[state.rank_idx % len(RANKS)]
    hand = state.hands[seat]
    if any(c[0] == rank for c in hand):
        return f"play:{rank}x1"
    return f"play:{shuffled(hand, seed)[0][0]}x1"


# ---- fallback policy: reads only a seat's own view ----

def _roll(seed, tag):
    return int(seed_hash(f"{seed}/{tag}")[:8], 16) / 2 ** 32


def tendencies(character_id):
    h = seed_hash(f"doubt-persona:{character_id}")
    return int(h[:8], 16) / 0xFFFFFFFF, int(h[8:16], 16) / 0xFFFFFFFF


def _move(cards):
    counts = {}
    for c in cards:
        counts[c[0]] = counts.get(c[0], 0) + 1
    return "play:" + ",".join(f"{r}x{counts[r]}" for r in RANKS if r in counts)


def policy_play(view, tendencies, seed):
    bluff = tendencies[0]
    hand = list(view.hand)
    own = [c for c in hand if c[0] == view.rank][:MAX_PLAY]
    others = [c for c in hand if c[0] != view.rank]
    if own:
        picked = own
        if len(own) < MAX_PLAY and others and _roll(seed, "extra") < bluff * 0.5:
            picked = own + [shuffled(others, f"{seed}/extra-card")[0]]
        return _move(picked)
    pool = shuffled(hand, f"{seed}/bluff-card")
    if len(pool) > 1 and bluff >= 0.7 and _roll(seed, "two") < bluff - 0.5:
        return _move(pool[:2])
    return _move(pool[:1])


def impossible(view):
    """True when the open claim cannot be true: my copies of the claimed rank plus the claim exceed the deck(s)."""
    w = view.window
    if w is None or view.hand is None:
        return False
    mine = sum(1 for c in view.hand if c[0] == w.rank)
    return mine + w.count > 4 * view.decks


def suspicion(view, tendencies):
    """0..1: how much this seat doubts the open claim."""
    w = view.window
    if w is None or view.hand is None:
        return 0.0
    if impossible(view):
        return 1.0
    mine = sum(1 for c in view.hand if c[0] == w.rank)
    unseen = max(4 * view.decks - mine, 1)
    left = view.counts[w.claimer]
    score = 0.20 + 0.15 * (w.count - 1) + 0.2 * min(1.0, w.count / unseen)
    score += 0.25 if left == 0 else 0.10 if left <= 2 else 0.0
    return max(0.0, min(1.0, score * (0.4 + 1.4 * tendencies[1])))


def policy_doubt(view, tendencies, seed):
    w = view.window
    if w is None or w.claimer == view.seat:
        return False
    if impossible(view):
        return True
    s = suspicion(view, tendencies)
    if s < DOUBT_BELOW:
        return False
    return bool(_roll(seed, "doubt") < (s - DOUBT_BELOW) / (1 - DOUBT_BELOW) * 0.8 + 0.1)


def triage(scored, k=2):
    """Seats worth a model call: every impossible claim plus the k most suspicious at or above TRIAGE_MIN."""
    sure = sorted(i for i, _, imp in scored if imp)
    rest = sorted(((i, s) for i, s, imp in scored if not imp and s >= TRIAGE_MIN), key=lambda p: (-p[1], p[0]))
    return sure + [i for i, _ in rest[:max(k, 0)]]
