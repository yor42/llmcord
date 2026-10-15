"""Yut (윷놀이) rules engine (FEAT-56, D42).

Pure functions over an immutable ``State``. A throw is a pure function of (seed, throw index). A piece position is a point id
("P0".."P19" ring, "a1","a2","b1","b2","c1","c2","d1","d2" diagonals, "C" center), ``None`` (unstarted) or "done". Each piece keeps its
``trail``: every single-step point it stepped through since the start, which is what a 빽도 retraces. Moves: ``throw``,
``move:<piece>:<throw>`` and ``forfeit`` (any active seat, any time).
"""
import json
import math
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

from ..i18n import Phrase, UserError, codes, terms
from . import playstyle
from .base import IllegalMove, Seat

MIN_SEATS = 2
MAX_SEATS = 4
P_FLAT_UP = math.pi / (2 + math.pi)
THROWS = ("do", "gae", "geol", "yut", "mo", "backdo")
STEPS = {"do": 1, "gae": 2, "geol": 3, "yut": 4, "mo": 5}
_BY_FLAT = ("mo", "do", "gae", "geol", "yut")
DEFAULT_RULES = {"backdo": True, "pieces": 4, "ante": 0, "turn_timeout": 60}
NOT_ON_TURN = "it is not this seat's turn"  # developer text: the store turns an IllegalMove into an error code
ANTE_MAX = 10 ** 9
PIECES_RANGE = (2, 5)
TIMEOUT_RANGE = (15, 300)

_RING = tuple(f"P{i}" for i in range(1, 20)) + ("P0",)
_A = ("a1", "a2", "C", "b1", "b2", "P15", "P16", "P17", "P18", "P19", "P0")  # from P5
_B = ("c1", "c2", "C", "d1", "d2", "P0")  # from P10
_ROUTES = {"a1": _A[1:], "a2": _A[2:], "b1": _A[4:], "b2": _A[5:], "c1": _B[1:], "c2": _B[2:], "C": _B[3:], "d1": _B[4:], "d2": _B[5:],
           "P5": _A, "P10": _B, "P0": ()}
_ROUTES.update({f"P{i}": _RING[i:] for i in range(1, 20) if f"P{i}" not in _ROUTES})
_ROUTES[None] = _RING


def throw_result(seed, n, backdo=True):
    """(name, sticks) of throw number n: four sticks, each flat-up with probability P_FLAT_UP; the flat count names the throw."""
    sticks = tuple(playstyle.roll(f"{seed}/throw/{n}", f"stick{i}") < P_FLAT_UP for i in range(4))
    flat = sum(sticks)
    if flat == 1 and sticks[0] and backdo:
        return "backdo", sticks
    return _BY_FLAT[flat], sticks


def normalize_rules(value=None):
    """The full rules dict for a stored/posted value (None or {} is the default table); UserError (a ValueError subclass) on a bad one."""
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise UserError(codes.RULES_YUT_OPTIONS)
    r = {**DEFAULT_RULES, **{k: v for k, v in value.items() if k in DEFAULT_RULES}}
    if type(r["backdo"]) is not bool:
        raise UserError(codes.RULES_YUT_FLAG, name="backdo")
    for k, (lo, hi), label in (("pieces", PIECES_RANGE, codes.RULES_YUT_PIECES), ("turn_timeout", TIMEOUT_RANGE, codes.RULES_YUT_TIMEOUT)):
        if type(r[k]) is not int or not lo <= r[k] <= hi:
            raise UserError(label, low=lo, high=hi)
    if type(r["ante"]) is not int or not 0 <= r["ante"] <= ANTE_MAX:
        raise UserError(codes.RULES_YUT_ANTE, limit=ANTE_MAX)
    return {k: r[k] for k in DEFAULT_RULES}


@dataclass(frozen=True)
class Thrown:
    seat: int
    name: str
    sticks: tuple
    kind: str = "throw"


@dataclass(frozen=True)
class Moved:
    seat: int
    pieces: tuple  # the piece indexes that moved together
    throw: str
    start: Optional[str]
    end: Optional[str]  # a point id, None (back at the start) or "done"
    captured: tuple = ()  # (seat, piece) pairs sent back to the start
    kind: str = "move"


@dataclass(frozen=True)
class Dropped:
    seat: int
    kind: str = "drop"  # a banked 빽도 that no piece could take


@dataclass(frozen=True)
class Forfeit:
    seat: int
    kind: str = "forfeit"


@dataclass(frozen=True)
class Win:
    seat: int
    kind: str = "win"


_EVENTS = {c.__dataclass_fields__["kind"].default: c for c in (Thrown, Moved, Dropped, Forfeit, Win)}


@dataclass(frozen=True)
class State:
    seed: str = field(repr=False)
    seats: tuple
    rules: dict
    pieces: tuple
    trails: tuple
    active: tuple
    turn: Optional[int]
    bank: tuple = ()
    phase: str = "throw"
    to_throw: int = 1
    throws_made: int = 0
    last_throw: Optional[tuple] = None
    history: tuple = ()
    finished: bool = False
    winner: Optional[int] = None


@dataclass(frozen=True)
class View:
    seat: Optional[int]
    active: tuple
    turn: Optional[int]
    phase: str
    pieces: tuple
    trails: tuple
    bank: tuple
    to_throw: int
    last_throw: Optional[tuple]
    history: tuple
    finished: bool
    winner: Optional[int]


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


def new(seed, seats: Sequence[Seat], rules=None):
    seats = tuple(seats)
    rules = normalize_rules(rules)
    n = len(seats)
    if not MIN_SEATS <= n <= MAX_SEATS:
        raise ValueError(f"Yut needs {MIN_SEATS} to {MAX_SEATS} seats")
    if [s.index for s in seats] != list(range(n)):
        raise ValueError("seat indexes must be 0..n-1 in order")
    k = rules["pieces"]
    return State(seed=seed, seats=seats, rules=rules, pieces=((None,) * k,) * n, trails=(((),) * k,) * n, active=(True,) * n, turn=0)


def _set(seq, i, value):
    return seq[:i] + (value,) + seq[i + 1:]


def _on_board(state, seat):
    return [i for i, p in enumerate(state.pieces[seat]) if p not in (None, "done")]


def legal_moves(state, seat):
    """The moves of a seat (``forfeit`` last). Reads only what a ``View`` has too, so it works on a State or a View."""
    if type(seat) is not int or not 0 <= seat < len(state.active):
        raise ValueError("no such seat")
    if state.finished or not state.active[seat]:
        return ()
    out = []
    if state.turn == seat and state.phase == "throw":
        out.append("throw")
    elif state.turn == seat and state.phase == "move":
        for i, p in enumerate(state.pieces[seat]):
            if p == "done":
                continue
            for name in dict.fromkeys(state.bank):
                if name != "backdo" or p is not None:
                    out.append(f"move:{i}:{name}")
    out.append("forfeit")
    return tuple(out)


def choices(state, seat):
    """The distinct moves a seat may make, never a forfeit: pieces on the same point with the same trail move alike, so only the first is offered.
    Works on a State or a View."""
    out, seen = [], set()
    for move in legal_moves(state, seat):
        if move == "forfeit":
            continue
        if move.startswith("move:"):
            _, piece, name = move.split(":")
            same = (state.pieces[seat][int(piece)], state.trails[seat][int(piece)], name)
            if same in seen:
                continue
            seen.add(same)
        out.append(move)
    return tuple(out)


def _next_seat(active, pos):
    n = len(active)
    pos = (pos + 1) % n
    while not active[pos]:
        pos = (pos + 1) % n
    return pos


def _settle(state):
    """Drop a banked 빽도 nobody can take, then enter the throw or move phase, or pass the turn when nothing is left."""
    seat = state.turn
    bank = state.bank
    history = state.history
    if state.to_throw > 0:
        return replace(state, phase="throw")
    if "backdo" in bank and not _on_board(state, seat):
        history += (Dropped(seat),) * bank.count("backdo")
        bank = tuple(b for b in bank if b != "backdo")
    if bank:
        return replace(state, bank=bank, phase="move", history=history)
    return replace(state, bank=(), phase="throw", to_throw=1, turn=_next_seat(state.active, seat), history=history)


def _throw(state, seat):
    name, sticks = throw_result(state.seed, state.throws_made, state.rules["backdo"])
    owed = state.to_throw - 1 + (1 if name in ("yut", "mo") else 0)
    state = replace(state, bank=state.bank + (name,), to_throw=owed, throws_made=state.throws_made + 1, last_throw=(name, sticks),
                    history=state.history + (Thrown(seat, name, sticks),))
    return _settle(state)


def _advance(pos, trail, name):
    """(new position, new trail) of a piece stepping by a throw. Where the route starts is decided by `pos` alone."""
    if name == "backdo":
        trail = trail[:-1]
        return (trail[-1] if trail else None), trail
    route = _ROUTES[pos]
    n = STEPS[name]
    if n > len(route):  # past P0: finished
        return "done", ()
    return route[n - 1], trail + route[:n]


def preview(state, seat, move):
    """What a ``move:<piece>:<throw>`` would do, without doing it (a State or a View): where the stack stands and lands, how many pieces
    move together, how many opposing pieces it captures, whether it reaches home."""
    _, piece, name = move.split(":")
    piece = int(piece)
    pos, trail = state.pieces[seat][piece], state.trails[seat][piece]
    end, _ = _advance(pos, trail, name)
    moving = sum(p == pos for p in state.pieces[seat]) if pos is not None else 1
    captured = 0
    if end not in (None, "done"):
        captured = sum(p == end for o in range(len(state.active)) if o != seat and state.active[o] for p in state.pieces[o])
    return {"piece": piece, "throw": name, "start": pos, "end": end, "moving": moving, "captured": captured, "home": end == "done"}


def _move(state, seat, piece, name):
    pos, trail = state.pieces[seat][piece], state.trails[seat][piece]
    stack = [i for i, p in enumerate(state.pieces[seat]) if p == pos] if pos is not None else [piece]
    end, new_trail = _advance(pos, trail, name)
    # A stack never splits: every member takes the moved piece's new trail (diverging trails of stack members, a case the rules
    # leave open, are resolved in favor of the piece that was chosen).
    pieces, trails = list(state.pieces), list(state.trails)
    mine, mine_t = list(pieces[seat]), list(trails[seat])
    for i in stack:
        mine[i], mine_t[i] = end, new_trail
    pieces[seat], trails[seat] = tuple(mine), tuple(mine_t)
    captured = []
    if end not in (None, "done"):
        for o in range(len(state.seats)):
            if o == seat or not state.active[o]:
                continue
            hit = [i for i, p in enumerate(pieces[o]) if p == end]
            if hit:
                captured += [(o, i) for i in hit]
                pieces[o] = tuple(None if i in hit else p for i, p in enumerate(pieces[o]))
                trails[o] = tuple(() if i in hit else t for i, t in enumerate(trails[o]))
    bank = list(state.bank)
    bank.remove(name)
    owed = state.to_throw + (1 if captured and name not in ("yut", "mo") else 0)
    state = replace(state, pieces=tuple(pieces), trails=tuple(trails), bank=tuple(bank), to_throw=owed,
                    history=state.history + (Moved(seat, tuple(stack), name, pos, end, tuple(captured)),))
    if all(p == "done" for p in state.pieces[seat]):
        return replace(state, finished=True, winner=seat, turn=None, bank=(), to_throw=0, history=state.history + (Win(seat),))
    return _settle(state)


def _forfeit(state, seat):
    k = state.rules["pieces"]
    on_turn = state.turn == seat
    state = replace(state, active=_set(state.active, seat, False), pieces=_set(state.pieces, seat, (None,) * k),
                    trails=_set(state.trails, seat, ((),) * k), history=state.history + (Forfeit(seat),))
    left = [i for i, a in enumerate(state.active) if a]
    if len(left) == 1:
        return replace(state, finished=True, winner=left[0], turn=None, bank=(), to_throw=0, history=state.history + (Win(left[0]),))
    if on_turn:
        state = replace(state, turn=_next_seat(state.active, seat), phase="throw", to_throw=1, bank=())
    return state


def apply(state, seat, move):
    if type(seat) is not int or not 0 <= seat < len(state.seats):
        raise IllegalMove("no such seat")
    if type(move) is not str:
        raise IllegalMove("a move is text")
    if state.finished:
        raise IllegalMove("the game is over")
    if not state.active[seat]:
        raise IllegalMove("that seat is out of the game")
    if move not in legal_moves(state, seat):
        raise IllegalMove("that move is not allowed right now")
    if move == "forfeit":
        return _forfeit(state, seat)
    if move == "throw":
        return _throw(state, seat)
    _, piece, name = move.split(":")
    return _move(state, seat, int(piece), name)


def replay(seed, seats, moves, rules=None):
    state = new(seed, seats, rules)
    for seat, move in moves:
        state = apply(state, seat, move)
    return state


def view(state, seat=None):
    if seat is not None and (type(seat) is not int or not 0 <= seat < len(state.seats)):
        raise ValueError("no such seat")
    return View(seat=seat, active=state.active, turn=state.turn, phase=state.phase, pieces=state.pieces, trails=state.trails,
                bank=state.bank, to_throw=state.to_throw, last_throw=state.last_throw, history=state.history,
                finished=state.finished, winner=state.winner)


def board_view(state):
    """Structured positions for a renderer: per-seat piece positions, stacks of 2+ as (point, seat, piece indexes), the bank, the last throw."""
    stacks = []
    for s, row in enumerate(state.pieces):
        for point in dict.fromkeys(p for p in row if p not in (None, "done")):
            members = tuple(i for i, p in enumerate(row) if p == point)
            if len(members) > 1:
                stacks.append((point, s, members))
    last = None if state.last_throw is None else {"name": state.last_throw[0], "sticks": state.last_throw[1]}
    return {"pieces": state.pieces, "stacks": tuple(stacks), "bank": state.bank, "to_throw": state.to_throw, "phase": state.phase,
            "turn": state.turn, "last_throw": last}


def _flat_odds():
    p, q = P_FLAT_UP, 1 - P_FLAT_UP
    return {"mo": q ** 4, "do": 3 * p * q ** 3, "backdo": p * q ** 3, "gae": 6 * p ** 2 * q ** 2, "geol": 4 * p ** 3 * q, "yut": p ** 4}


_ODDS = _flat_odds()


def _dist(point):
    return 0 if point == "done" else len(_ROUTES[point])


def _threat(state, seat, end):
    """The chance that an opposing piece lands on ``end`` with its next throw (a throw counted once; extra throws are ignored)."""
    hit = 0.0
    for name, odds in _ODDS.items():
        if any(_advance(p, state.trails[o][i], name)[0] == end
               for o in range(len(state.active)) if o != seat and state.active[o]
               for i, p in enumerate(state.pieces[o]) if p != "done" and (p is not None or name != "backdo")):
            hit += odds
    return hit


def _score(state, seat, move, bold):
    info = preview(state, seat, move)
    start, end = info["start"], info["end"]
    score = 3.0 * (_dist(start) - _dist(end)) + 0.1 * (20 - _dist(end))
    score += info["captured"] * (45 + 70 * bold) + (60 if info["home"] else 0)
    if start in ("P5", "P10") and end not in (None, "done") and not end.startswith("P"):
        score += 12 + 40 * bold  # took the shortcut
    if end not in (None, "done"):
        mine = [i for i, p in enumerate(state.pieces[seat]) if p == end]
        if mine:
            score += (8 + 40 * (1 - bold)) * (len(mine) + info["moving"] - 1)  # stacking
        size = info["moving"] + len(mine)
        score -= _threat(state, seat, end) * size * (25 + 5 * (20 - _dist(end))) * (0.4 + 1.2 * (1 - bold))
    return score


def policy_move(state, seat, style=None, seed=None):
    """One legal move (never a forfeit) for a seat from what is on the table (a State or a View: Yut hides nothing). The throw is always made.
    A move is scored: a capture, bringing a stack home and a shortcut gain, advancing the front piece gains a little, and a landing an opposing
    piece can hit costs its chance of being hit. ``style`` shifts the weights: bold (aggression and risk) values captures and shortcuts,
    cautious values stacking and safe landings; an erratic style may deviate to another legal move. Deterministic in (state, seat, style, seed)."""
    options = choices(state, seat)
    if not options:
        raise IllegalMove(NOT_ON_TURN)
    if len(options) == 1:
        return options[0]
    bold = 0.5 if style is None else (style.unit("aggression") + style.unit("risk")) / 2
    best = max(range(len(options)), key=lambda i: (_score(state, seat, options[i], bold), -i))
    if style is not None:
        seed = seed if seed is not None else f"{getattr(state, 'seed', 'yut')}/policy/{len(state.history)}"
        if playstyle.erratic(style, seed, "move"):
            others = [m for i, m in enumerate(options) if i != best]
            return others[int(playstyle.roll(seed, "pick") * len(others))]
    return options[best]


def result(state):
    pot = sum(s.stake for s in state.seats)
    out = []
    for i in range(len(state.seats)):
        if not state.active[i]:
            out.append(SeatResult(i, "forfeit", 0))
        elif not state.finished:
            out.append(SeatResult(i, None, 0))
        elif i == state.winner:
            out.append(SeatResult(i, "win", pot))
        else:
            out.append(SeatResult(i, "lose", 0))
    return Result(state.finished, state.winner, tuple(out), pot)


def rules_phrases(rules=None):
    """The table's rules in plain words, one Phrase per part (display only); join with " · "."""
    r = normalize_rules(rules)
    return [Phrase(terms.DOUBT_RULE_ANTE, ante=r["ante"]) if r["ante"] else Phrase(terms.DOUBT_RULE_NO_ANTE), Phrase(terms.DOUBT_RULE_TIMEOUT, seconds=r["turn_timeout"]),
            Phrase(terms.YUT_RULE_BACKDO_ON if r["backdo"] else terms.YUT_RULE_BACKDO_OFF), Phrase(terms.YUT_RULE_PIECES, count=r["pieces"])]


def how_to_play_phrases(rules=None):
    """The how-to-play lines, one Phrase per line (display only); join with a newline."""
    r = normalize_rules(rules)
    lines = [Phrase(terms.YUT_HOW_TITLE), Phrase(terms.YUT_HOW_GOAL), Phrase(terms.YUT_HOW_EXTRA), Phrase(terms.YUT_HOW_BOARD)]
    if r["backdo"]:
        lines.append(Phrase(terms.YUT_HOW_BACKDO))
    lines += [Phrase(terms.YUT_HOW_FINISH), Phrase(terms.YUT_HOW_TIMEOUT, seconds=r["turn_timeout"]),
              Phrase(terms.YUT_HOW_WIN_POT, ante=r["ante"]) if r["ante"] else Phrase(terms.YUT_HOW_WIN)]
    return lines


def _deep(x):
    return tuple(_deep(v) for v in x) if isinstance(x, list) else x


def to_json(state):
    return json.dumps({
        "seed": state.seed, "seats": [[s.index, s.kind, s.ref, s.stake] for s in state.seats], "rules": state.rules,
        "pieces": state.pieces, "trails": state.trails, "active": state.active, "turn": state.turn, "bank": state.bank,
        "phase": state.phase, "to_throw": state.to_throw, "throws_made": state.throws_made, "last_throw": state.last_throw,
        "history": [{f: getattr(e, f) for f in e.__dataclass_fields__} for e in state.history], "finished": state.finished,
        "winner": state.winner})


def from_json(text):
    d = json.loads(text)
    history = []
    for e in d["history"]:
        e = dict(e)
        cls = _EVENTS[e.pop("kind")]
        history.append(cls(**{k: _deep(v) for k, v in e.items()}))
    return State(
        seed=d["seed"], seats=tuple(Seat(*s) for s in d["seats"]), rules=d["rules"], pieces=_deep(d["pieces"]), trails=_deep(d["trails"]),
        active=_deep(d["active"]), turn=d["turn"], bank=_deep(d["bank"]), phase=d["phase"], to_throw=d["to_throw"],
        throws_made=d["throws_made"], last_throw=_deep(d["last_throw"]), history=tuple(history), finished=d["finished"], winner=d["winner"])
