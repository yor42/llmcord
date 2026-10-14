"""One Card (원카드) rules engine (FEAT-43, D39).

Pure functions over an immutable ``State``. Cards are (rank, suit) tuples; jokers are ("JK", "black") and ("JK", "color").
A game is fully determined by (seed, seats, rules, moves). Moves: ``play:<card>`` (``play:7<suit>=<named suit>`` for a 7),
``draw``, ``timeout`` (the turn seat's missed turn, draws like ``draw``), ``declare`` and ``call_out:<seat>`` (any active
seat, on or off turn; neither advances the turn), ``forfeit`` (any active seat, on or off turn: it leaves the game).
"""
import json
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

from .base import IllegalMove, Seat, shuffled

RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
SUITS = ("♠", "♥", "♦", "♣")
PLAIN = ("3", "4", "5", "6", "8", "9", "10")
JOKERS = (("JK", "black"), ("JK", "color"))
MIN_SEATS = 2
MAX_SEATS = 8
HAND = 7
DEFAULT_RULES = {"spade_ace_ten": False, "defense": True, "king_skips_two": True, "last_card": "none", "bankruptcy": False,
                 "bankruptcy_limit": 20, "call_out_penalty": 1, "ante": 0, "turn_timeout": 60}
ANTE_MAX = 10 ** 9
TIMEOUT_RANGE = (15, 300)
BANKRUPTCY_RANGE = (15, 30)
PENALTY_RANGE = (1, 3)
LAST_CARD = ("none", "no_attack", "no_attack_or_joker")
_BOOLS = ("spade_ace_ten", "defense", "king_skips_two", "bankruptcy")
_INTS = (("bankruptcy_limit", BANKRUPTCY_RANGE, "The bankruptcy limit"), ("call_out_penalty", PENALTY_RANGE, "The call-out penalty"),
         ("turn_timeout", TIMEOUT_RANGE, "The turn timeout"))


def normalize_rules(value=None):
    """The full rules dict for a stored/posted value (None or {} is the default table); ValueError on a bad one."""
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("The One Card rules must be a set of options.")
    r = {**DEFAULT_RULES, **{k: v for k, v in value.items() if k in DEFAULT_RULES}}
    for k in _BOOLS:
        if type(r[k]) is not bool:
            raise ValueError(f"The option {k} must be on or off.")
    if r["last_card"] not in LAST_CARD:
        raise ValueError(f"The last card rule must be one of {', '.join(LAST_CARD)}.")
    for k, (lo, hi), label in _INTS:
        if type(r[k]) is not int or not lo <= r[k] <= hi:
            raise ValueError(f"{label} must be a whole number from {lo} to {hi}.")
    if type(r["ante"]) is not int or not 0 <= r["ante"] <= ANTE_MAX:
        raise ValueError(f"The ante must be a whole number from 0 to {ANTE_MAX}.")
    return {k: r[k] for k in DEFAULT_RULES}


@dataclass(frozen=True)
class Play:
    seat: int
    card: tuple
    suit: Optional[str] = None
    kind: str = "play"


@dataclass(frozen=True)
class Draw:
    seat: int
    count: int
    reason: str  # 'pass' | 'attack' | 'timeout' | 'penalty'
    kind: str = "draw"


@dataclass(frozen=True)
class Declare:
    seat: int
    kind: str = "declare"


@dataclass(frozen=True)
class Callout:
    caller: int
    target: int
    hit: bool
    penalty: int  # cards the penalised seat actually drew
    kind: str = "callout"


@dataclass(frozen=True)
class Bankrupt:
    seat: int
    kind: str = "bankrupt"


@dataclass(frozen=True)
class Forfeit:
    seat: int
    kind: str = "forfeit"


@dataclass(frozen=True)
class Reshuffle:
    kind: str = "reshuffle"


@dataclass(frozen=True)
class Win:
    seat: int
    kind: str = "win"


_EVENTS = {c.__dataclass_fields__["kind"].default: c for c in (Play, Draw, Declare, Callout, Bankrupt, Forfeit, Reshuffle, Win)}


@dataclass(frozen=True)
class State:
    seed: str = field(repr=False)
    seats: tuple
    rules: dict
    decks: int
    hands: tuple = field(repr=False)
    draw: tuple = field(repr=False)
    discard: tuple
    active: tuple
    turn: Optional[int]
    direction: int = 1
    suit: Optional[str] = None
    pending: int = 0
    declared: tuple = ()
    exposed: tuple = ()
    history: tuple = ()
    finished: bool = False
    winner: Optional[int] = None


@dataclass(frozen=True)
class View:
    seat: Optional[int]
    counts: tuple
    active: tuple
    turn: Optional[int]
    direction: int
    top: tuple
    suit: Optional[str]
    pending: int
    draw: int
    discard: int
    declared: tuple
    exposed: tuple
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


def card_text(card):
    if card[0] == "JK":
        return "JKb" if card[1] == "black" else "JKc"
    return f"{card[0]}{card[1]}"


def play_move(card, suit=None):
    return f"play:{card_text(card)}" + (f"={suit}" if suit else "")


def _parse_card(text):
    if text == "JKb":
        return JOKERS[0]
    if text == "JKc":
        return JOKERS[1]
    return (text[:-1], text[-1])


def _sort_key(card):
    return (RANKS.index(card[0]), SUITS.index(card[1])) if card[0] != "JK" else (len(RANKS), card[1] == "color")


def new(seed, seats: Sequence[Seat], rules=None):
    seats = tuple(seats)
    rules = normalize_rules(rules)
    n = len(seats)
    if not MIN_SEATS <= n <= MAX_SEATS:
        raise ValueError(f"One Card needs {MIN_SEATS} to {MAX_SEATS} seats")
    if [s.index for s in seats] != list(range(n)):
        raise ValueError("seat indexes must be 0..n-1 in order")
    decks = 1 if n <= 4 else 2
    deck = shuffled(([(r, s) for s in SUITS for r in RANKS] + list(JOKERS)) * decks, seed)
    hands = tuple(tuple(deck[i * HAND:(i + 1) * HAND]) for i in range(n))
    rest = deck[n * HAND:]
    k = next(i for i, card in enumerate(rest) if card[0] in PLAIN)
    return State(seed=seed, seats=seats, rules=rules, decks=decks, hands=hands, draw=tuple(rest[k + 1:] + rest[:k]),
                 discard=(rest[k],), active=(True,) * n, turn=0, declared=(False,) * n)


def _value(card, rules):
    rank, suit = card
    if rank == "JK":
        return 5 if suit == "black" else 7
    if rank == "A":
        return (10 if rules["spade_ace_ten"] else 5) if suit == "♠" else 3
    return 2 if rank == "2" else 0


def _defends(top, card):
    if top[0] == "JK":
        return card[0] == "JK"
    if top[0] == "A" and top[1] == "♠":
        return card == JOKERS[0]
    if top[0] == "A":
        return card[0] in ("A", "JK")
    return card[0] in ("2", "JK") or (card[0] == "A" and card[1] == top[1])


def _matches(top, named, card):
    if card[0] == "JK" or top[0] == "JK":
        return True
    if named is not None:
        return card[1] == named
    return card[1] == top[1] or card[0] == top[0]


def _banned_last(rules, card):
    rule = rules["last_card"]
    if rule == "none":
        return False
    return card[0] in ("2", "A") or (rule == "no_attack_or_joker" and card[0] == "JK")


def _can_act(state, seat):
    return not state.finished and state.turn == seat and state.active[seat]


def _playable(hand, top, named, pending, rules):
    """The distinct cards of a hand that may be played now (a last card may be banned by the rules)."""
    out = []
    for card in dict.fromkeys(hand):
        ok = rules["defense"] and _defends(top, card) if pending else _matches(top, named, card)
        if ok and not (len(hand) == 1 and _banned_last(rules, card)):
            out.append(card)
    return out


def legal_moves(state, seat):
    if not _can_act(state, seat):
        return ()
    out = []
    for card in _playable(state.hands[seat], state.discard[-1], state.suit, state.pending, state.rules):
        out.extend([play_move(card, s) for s in SUITS] if card[0] == "7" else [play_move(card)])
    return tuple(out) + ("draw",)


def playable_cards(view, rules):
    """The distinct cards a seat's own view could play now, as card texts (empty off turn); the named suit is the last play's."""
    if view.hand is None or view.finished or view.turn != view.seat or not view.active[view.seat]:
        return ()
    named = next((e.suit for e in reversed(view.history) if e.kind == "play"), None)
    return tuple(card_text(c) for c in _playable(view.hand, view.top, named, view.pending, normalize_rules(rules)))


def _next(active, pos, direction, steps):
    n = len(active)
    for _ in range(steps):
        pos = (pos + direction) % n
        while not active[pos]:
            pos = (pos + direction) % n
    return pos


def _set(seq, i, value):
    return seq[:i] + (value,) + seq[i + 1:]


def _take(state, seat, n):
    """Draw up to n cards for a seat, reshuffling the discard under its top when the pile runs out: (state, drawn)."""
    draw, discard, hand, history = state.draw, state.discard, state.hands[seat], state.history
    drawn = 0
    while drawn < n:
        if not draw and len(discard) > 1:
            k = sum(1 for e in history if e.kind == "reshuffle")
            draw = tuple(shuffled(discard[:-1], f"{state.seed}/reshuffle/{k}"))
            discard = discard[-1:]
            history += (Reshuffle(),)
        if not draw:
            break
        hand += (draw[0],)
        draw = draw[1:]
        drawn += 1
    declared = _set(state.declared, seat, False) if len(hand) >= 3 else state.declared
    return replace(state, draw=draw, discard=discard, hands=_set(state.hands, seat, hand), declared=declared, history=history), drawn


def _finish(state, winner):
    return replace(state, finished=True, winner=winner, turn=None, history=state.history + (Win(winner),))


def _eliminate(state, seat, event, tag):
    """Take a seat out: its hand goes under the draw pile (seed-shuffled); the lone seat left wins."""
    back = tuple(shuffled(state.hands[seat], f"{state.seed}/{tag}/{seat}"))
    state = replace(
        state, active=_set(state.active, seat, False), hands=_set(state.hands, seat, ()), draw=state.draw + back,
        declared=_set(state.declared, seat, False), exposed=tuple(x for x in state.exposed if x != seat),
        history=state.history + (event,))
    left = [i for i, a in enumerate(state.active) if a]
    return _finish(state, left[0]) if len(left) == 1 else state


def _bust(state, seat):
    """Eliminate a seat that reached the bankruptcy limit; the lone seat left wins."""
    r = state.rules
    if not r["bankruptcy"] or not state.active[seat] or len(state.hands[seat]) < r["bankruptcy_limit"]:
        return state
    return _eliminate(state, seat, Bankrupt(seat), "bankrupt")


def _forfeit(state, seat):
    on_turn = state.turn == seat
    state = _eliminate(state, seat, Forfeit(seat), "forfeit")
    if not state.finished and on_turn:  # `pending` is kept, as after a bust: the next seat inherits an unanswered attack
        state = replace(state, turn=_next(state.active, seat, state.direction, 1))
    return state


def _parse_target(text):
    if not (text.isascii() and text.isdigit()) or len(text) > 2:
        raise IllegalMove("call_out needs a seat number")
    return int(text)


def apply(state, seat, move):
    if type(seat) is not int or not 0 <= seat < len(state.seats):
        raise IllegalMove("no such seat")
    if type(move) is not str:
        raise IllegalMove("a move is text")
    if state.finished:
        raise IllegalMove("the game is over")
    if not state.active[seat]:
        raise IllegalMove("that seat is out of the game")
    if move == "forfeit":
        return _forfeit(state, seat)
    if move == "declare":
        return _declare(state, seat)
    if move.startswith("call_out:"):
        return _call_out(state, seat, _parse_target(move[9:]))
    if move == "timeout" and state.turn == seat:
        return _draw(state, seat, "timeout")
    if move not in legal_moves(state, seat):
        raise IllegalMove("that move is not allowed right now")
    if move == "draw":
        return _draw(state, seat, "attack" if state.pending else "pass")
    ctext, _, named = move[5:].partition("=")
    return _play(state, seat, _parse_card(ctext), named or None)


def _declare(state, seat):
    if len(state.hands[seat]) not in (1, 2) or state.declared[seat]:
        raise IllegalMove("nothing to declare")
    return replace(state, declared=_set(state.declared, seat, True), exposed=tuple(x for x in state.exposed if x != seat),
                   history=state.history + (Declare(seat),))


def _call_out(state, caller, target):
    if target == caller or not 0 <= target < len(state.seats) or not state.active[target]:
        raise IllegalMove("that call-out has no target")
    hit = target in state.exposed
    who = target if hit else caller
    state, drawn = _take(state, who, state.rules["call_out_penalty"])
    if hit:
        state = replace(state, exposed=())
    state = _bust(replace(state, history=state.history + (Callout(caller, target, hit, drawn),)), who)
    if not state.finished and not state.active[state.turn]:
        # a bust on the turn seat passes the turn on; `pending` is kept, so the next seat inherits an unanswered attack
        state = replace(state, turn=_next(state.active, state.turn, state.direction, 1))
    return state


def _draw(state, seat, reason):
    state = replace(state, exposed=())
    state, drawn = _take(state, seat, state.pending or 1)
    state = replace(state, pending=0, history=state.history + (Draw(seat, drawn, reason),))
    state = _bust(state, seat)
    if state.finished:
        return state
    return replace(state, turn=_next(state.active, seat, state.direction, 1))


def _play(state, seat, card, named):
    hand = list(state.hands[seat])
    hand.remove(card)
    state = replace(state, hands=_set(state.hands, seat, tuple(hand)), discard=state.discard + (card,), suit=named, exposed=(),
                    history=state.history + (Play(seat, card, named),))
    if not hand:
        return _finish(state, seat)
    if len(hand) == 1 and not state.declared[seat]:
        state = replace(state, exposed=(seat,))
    rank, direction, steps = card[0], state.direction, 1
    pending = state.pending + _value(card, state.rules)
    if rank == "J":
        steps = 2
    elif rank == "Q":
        direction = -direction
        steps = 2 if sum(state.active) == 2 else 1
    elif rank == "K" and state.rules["king_skips_two"] and sum(state.active) >= 4:
        steps = 3
    return replace(state, direction=direction, pending=pending, turn=_next(state.active, seat, direction, steps))


def rules_text(rules=None):
    """The table's rules in plain words (display only), one line. No card symbols: the line is shown in public messages."""
    r = normalize_rules(rules)
    bits = [f"Ante {r['ante']}" if r["ante"] else "No ante", f"{r['turn_timeout']} s per turn",
            "defense on" if r["defense"] else "defense off", "king skips two" if r["king_skips_two"] else "king has no effect",
            {"none": "last card unrestricted", "no_attack": "no attack as the last card", "no_attack_or_joker": "no attack or joker as the last card"}[r["last_card"]],
            "spade ace worth 10" if r["spade_ace_ten"] else "spade ace worth 5"]
    if r["bankruptcy"]:
        bits.append(f"bankruptcy at {r['bankruptcy_limit']} cards")
    bits.append(f"call-out penalty {r['call_out_penalty']}")
    return " · ".join(bits)


def how_to_play(rules=None):
    r = normalize_rules(rules)
    lines = [
        "**One Card in short**",
        "Be the first to play your last card. Everyone starts with 7 cards; play a card that matches the top card's suit or rank, or draw one and pass. Jokers play on anything.",
        "• Attacks: 2 = 2 cards, A = 3, ♠A = " + ("10" if r["spade_ace_ten"] else "5") + ", black joker = 5, color joker = 7. They add up until someone draws the total and loses the turn."
        + (" You can answer an attack with a stronger or equal defense: a 2 answers a 2 (or the A of the same suit, or a joker), an A answers an A or a joker, ♠A only a black joker, a joker another joker." if r["defense"] else " Defenses are off: you can only draw."),
        "• J skips the next player. Q reverses the direction (with two players you go again). "
        + ("K skips two players (with four or more)." if r["king_skips_two"] else "K has no special effect."),
        "• 7 lets you name the suit the next play must follow.",
        "• At one card, declare it (chat declarations are coming); a player who didn't can be called out, and the caller makes them draw "
        + f"{r['call_out_penalty']}. A wrong call-out costs the caller the same.",
    ]
    if r["last_card"] != "none":
        lines.append("• You can't finish on an attack card" + (" or a joker" if r["last_card"] == "no_attack_or_joker" else "") + ".")
    if r["bankruptcy"]:
        lines.append(f"• Reach {r['bankruptcy_limit']} cards and you go bankrupt: you are out and your cards go back to the pile.")
    lines.append(f"• Take longer than {r['turn_timeout']} s and you draw and lose the turn; miss two turns in a row and you forfeit your seat.")
    lines.append("• The last player standing, or whoever plays their last card, takes the pot" + (f" ({r['ante']} each)." if r["ante"] else "."))
    return "\n".join(lines)


def replay(seed, seats, moves, rules=None):
    state = new(seed, seats, rules)
    for seat, move in moves:
        state = apply(state, seat, move)
    return state


def view(state, seat=None):
    if seat is not None and (type(seat) is not int or not 0 <= seat < len(state.seats)):
        raise ValueError("no such seat")
    top = state.discard[-1]
    return View(
        seat=seat, counts=tuple(len(h) for h in state.hands), active=state.active, turn=state.turn, direction=state.direction,
        top=top, suit=state.suit or (None if top[0] == "JK" else top[1]), pending=state.pending, draw=len(state.draw),
        discard=len(state.discard), declared=state.declared, exposed=state.exposed, history=state.history,
        finished=state.finished, winner=state.winner, decks=state.decks,
        hand=None if seat is None else tuple(sorted(state.hands[seat], key=_sort_key)))


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


def to_json(state):
    events = []
    for e in state.history:
        d = {f: getattr(e, f) for f in e.__dataclass_fields__}
        if e.kind == "play":
            d["card"] = list(e.card)
        events.append(d)
    return json.dumps({
        "seed": state.seed, "seats": [[s.index, s.kind, s.ref, s.stake] for s in state.seats], "rules": state.rules,
        "decks": state.decks, "hands": [[list(c) for c in h] for h in state.hands], "draw": [list(c) for c in state.draw],
        "discard": [list(c) for c in state.discard], "active": list(state.active), "turn": state.turn,
        "direction": state.direction, "suit": state.suit, "pending": state.pending, "declared": list(state.declared),
        "exposed": list(state.exposed), "history": events, "finished": state.finished, "winner": state.winner})


def from_json(text):
    d = json.loads(text)
    cards = lambda seq: tuple(tuple(c) for c in seq)  # noqa: E731
    history = []
    for e in d["history"]:
        e = dict(e)
        cls = _EVENTS[e.pop("kind")]
        if cls is Play:
            e["card"] = tuple(e["card"])
        history.append(cls(**e))
    return State(
        seed=d["seed"], seats=tuple(Seat(*s) for s in d["seats"]), rules=d["rules"], decks=d["decks"],
        hands=tuple(cards(h) for h in d["hands"]), draw=cards(d["draw"]), discard=cards(d["discard"]),
        active=tuple(d["active"]), turn=d["turn"], direction=d["direction"], suit=d["suit"], pending=d["pending"],
        declared=tuple(d["declared"]), exposed=tuple(d["exposed"]), history=tuple(history), finished=d["finished"],
        winner=d["winner"])
