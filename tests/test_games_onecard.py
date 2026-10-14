"""One Card rules engine (llmcord_core/games/onecard.py + registry entry), FEAT-43, decision D39.

These tests define the engine API; they fail today because the module does not exist. The engine mirrors
``games/doubt.py`` and ``games/holdem.py``: pure functions over an immutable ``State``, a game fully determined by
(seed, seats, rules, moves). ``apply`` returns the new state; events are appended to ``state.history``
(``after.history[len(before.history):]`` is what one move did). A bad move raises ``base.IllegalMove``.

Cards are (rank, suit) tuples. Ranks "A","2".."10","J","Q","K"; suits "♠","♥","♦","♣". Jokers are ("JK", "black")
and ("JK", "color"). A deck is 52 cards plus the 2 jokers; two decks from 5 seats (1 deck for 2..4 seats).
``card_text(card)``: "A♠", "10♥", "JKb" (black joker), "JKc" (color joker).

Module API
  MIN_SEATS = 2, MAX_SEATS = 8, DEFAULT_RULES, TIMEOUT_RANGE = (15, 300), BANKRUPTCY_RANGE = (15, 30), PENALTY_RANGE = (1, 3)
  normalize_rules(value=None) -> dict   (None / {} / dict / JSON string; unknown keys dropped; ValueError on a bad value)
  new(seed, seats, rules=None) -> State
  legal_moves(state, seat) -> tuple[str]   (only the turn seat's moves: the plays and "draw"; () for everyone else)
  apply(state, seat, move) -> State
  view(state, seat=None) -> View           (public when seat is None; ValueError on a seat that does not exist)
  result(state) -> Result                  (finished, winner, seats[i].outcome / .payout, pot)
  replay(seed, seats, moves, rules=None) -> State        (moves: [(seat, move), ...])
  to_json(state) -> str, from_json(text) -> State        (lossless: from_json(to_json(s)) == s)
  card_text(card) -> str, play_move(card, suit=None) -> str

Moves (strings)
  play:<card_text>            e.g. "play:9♥", "play:JKb"
  play:7<suit>=<named suit>   a 7 must name a suit: "play:7♥=♠"; naming on any other card, or none on a 7, is illegal
  draw                        no play: draw 1 and pass (while an attack is pending: draw the whole sum and lose the turn)
  timeout                     the turn seat's missed turn: draws exactly like "draw" (the event reason is "timeout")
  declare                     any active seat holding 1 or 2 cards, on or off turn
  call_out:<target seat>      any active seat other than the target, on or off turn; neither advances the turn

State (frozen dataclass; seed/hands/draw are repr=False). Fields:
  seed, seats, rules, decks, hands (tuple of tuples), draw (draw pile, draw[0] is drawn first, drawn cards are
  appended to the end of the hand), discard (discard[-1] is the top card), active (tuple[bool]), turn (None when
  finished), direction (+1 / -1), suit (the suit a 7 named, else None), pending (cards owed to an attack, 0 when none),
  declared (tuple[bool]), exposed (tuple of seats that may be called out now), history, finished, winner.
Events in history (dataclasses with ``kind``):
  play (seat, card, suit), draw (seat, count, reason: "pass" | "attack" | "timeout" | "penalty"; no card identity),
  declare (seat), callout (caller, target, hit: bool, penalty), bankrupt (seat), reshuffle, win (seat).
View: seat, counts, active, turn, direction, top, suit (the suit the next play must match: the named one, else the
  top card's own suit, None on a joker), pending, draw (pile size), discard (pile size), declared, exposed, history,
  finished, winner, decks, hand (own cards sorted; None in the public view). Other seats' hands are never in a view.

Rules settled here (the backlog leaves them open); the implementer builds to these
  * Rules keys and defaults: spade_ace_ten False, defense True, king_skips_two True, last_card "none"
    ("no_attack" | "no_attack_or_joker"), bankruptcy False, bankruptcy_limit 20 (15..30), call_out_penalty 1 (1..3),
    turn_timeout 60 (15..300).
  * Deal: 7 cards each from the shuffled deck(s); the starter is the first card that is a plain rank (3-6, 8-10), cards
    skipped over go to the bottom of the draw pile. Seat 0 starts, direction +1. All cards are conserved.
  * A play must match the top card's suit or rank. Jokers are playable on anything, and anything is playable on a joker.
    While a 7's named suit is active only that suit (or a joker) may be played: rank matching is off. The named suit
    ends with the next play. After an attack is drawn the next seat plays against the top card as usual.
  * Attack values: 2 = 2, A = 3, spade ace = 5 (10 with spade_ace_ten), black joker 5, color joker 7. Attacks add up.
    While an attack is pending the only legal moves are a defense and "draw". Defenses are judged against the top
    card: 2 <- any 2, the A of the same suit, any joker; A <- any A, any joker; spade ace <- black joker only;
    a joker <- any joker. A defense adds its own value and passes the turn on. defense=False leaves only "draw".
  * J skips the next seat, Q reverses (with 2 seats the player goes again, as with J), K skips two (king_skips_two=False:
    no effect). 7 names a suit. None of them can answer a pending attack.
  * last_card: with exactly one card in hand, "no_attack" forbids finishing on 2, A, spade ace (jokers allowed);
    "no_attack_or_joker" also forbids jokers. The seat can still draw.
  * Declaration: "declare" is legal with 1 or 2 cards. It holds while the hand stays at 1 or 2 cards (drawing to 3 clears it).
    A seat that plays down to exactly one card undeclared is exposed until the next play, draw or timeout by any seat
    (a call-out or declare does not expire it). A call-out on an exposed seat makes the target draw the penalty; any
    other call-out (target not exposed, e.g. declared or with other than one card) makes the caller draw it.
  * Bankruptcy (on): a seat reaching bankruptcy_limit cards is eliminated (active False, hand empty, its cards shuffled
    back into the draw pile with a seed-derived shuffle) and a lone remaining seat wins. Its outcome in result() is "forfeit".
  * Draw pile exhausted: the discard minus its top is shuffled (seed-derived) into the draw pile. If nothing is left
    a draw simply gives fewer cards and the turn still passes.
  * result(): the winner's payout is the pot (sum of seat stakes); the others "lose"; outcome None while running.
Registry: key "onecard", name "One Card", noun "One Card", ledger_label "One Card", min_seats 2, max_seats 8,
  ante True, cash False, normalize_rules is onecard.normalize_rules, engine is the module.
Schema: none needed. ``game`` is free TEXT in game_tables / game_settings and the store resolves games through
  ``registry.get`` / ``registry.keys()``.
"""
import dataclasses
import json
import unittest
from collections import Counter

from llmcord_core.games import IllegalMove, Seat, registry
from llmcord_core.games import onecard
from test_game_tables import ALICE, CH, G, TableCase

SUITS = ("♠", "♥", "♦", "♣")
RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
PLAIN = ("3", "4", "5", "6", "8", "9", "10")
FULL_DECK = Counter([(r, s) for s in SUITS for r in RANKS] + [("JK", "black"), ("JK", "color")])
FILL = tuple((r, s) for s in "♣♦♠♥" for r in PLAIN)  # 28 known plain cards, the staged draw pile
SPARE = (("9", "♣"), ("10", "♣"), ("8", "♣"))


def c(text):
    if text == "JKb":
        return ("JK", "black")
    if text == "JKc":
        return ("JK", "color")
    return (text[:-1], text[-1])


def cs(*texts):
    return tuple(c(t) for t in texts)


def seats(n, stake=0):
    return [Seat(i, "member", 100 + i, stake) for i in range(n)]


def mk(n=3, seed="s", stake=0, rules=None):
    return onecard.new(seed, seats(n, stake), rules)


def stage(n=3, hands=(), top="5♥", draw=None, turn=0, rules=None, stake=0, **fields):
    """A state with chosen hands (seat -> card texts), top card and draw pile; other seats hold three clubs."""
    s = mk(n, "stage", stake, rules)
    hs = []
    for i in range(n):
        hs.append(cs(*hands[i]) if i < len(hands) and hands[i] is not None else SPARE)
    return dataclasses.replace(
        s, hands=tuple(hs), discard=(c(top),), draw=FILL if draw is None else tuple(draw), turn=turn, direction=1, suit=None,
        declared=(False,) * n, exposed=(), history=(), active=(True,) * n, **{"pending": 0, **fields})


def hand(state, i):
    return list(onecard.view(state, i).hand)


def legal(state, seat):
    return set(onecard.legal_moves(state, seat))


def M(text, suit=None):
    return onecard.play_move(c(text), suit)


def play(state, seat, text, suit=None):
    return onecard.apply(state, seat, M(text, suit))


def new_events(before, after):
    return after.history[len(before.history):]


def all_cards(state):
    return Counter([card for h in state.hands for card in h] + list(state.draw) + list(state.discard))


def walk_cards(obj, skip=()):
    if isinstance(obj, tuple) and len(obj) == 2 and obj[0] in RANKS + ("JK",):
        return [obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        items = [(f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj)]
    elif isinstance(obj, (list, tuple)):
        items = [(None, x) for x in obj]
    else:
        return []
    found = []
    for name, value in items:
        if name not in skip:
            found.extend(walk_cards(value))
    return found


def value(card):
    rank, suit = card
    if rank == "JK":
        return 5 if suit == "black" else 7
    if rank == "A":
        return 5 if suit == "♠" else 3
    return 2 if rank == "2" else 0


def pick_move(state):
    """A simple deterministic player: the first play (sorted), else draw."""
    seat = state.turn
    moves = sorted(m for m in onecard.legal_moves(state, seat) if m.startswith("play:"))
    return seat, (moves[0] if moves else "draw")


class DealTests(unittest.TestCase):
    def test_counts_decks_and_card_conservation_per_seat_count(self):
        for n in range(2, 9):
            with self.subTest(seats=n):
                decks = 1 if n <= 4 else 2
                s = mk(n)
                self.assertEqual(s.decks, decks)
                self.assertEqual([len(h) for h in s.hands], [7] * n)
                self.assertEqual(len(s.discard), 1)
                self.assertEqual(len(s.draw), 54 * decks - 7 * n - 1)
                self.assertEqual(all_cards(s), Counter({k: v * decks for k, v in FULL_DECK.items()}))

    def test_starter_is_a_plain_card_and_the_opening_state(self):
        for k in range(30):
            s = mk(4, f"seed-{k}")
            self.assertIn(s.discard[-1][0], PLAIN)
            v = onecard.view(s)
            self.assertEqual((v.turn, v.direction, v.pending, v.finished, v.winner, v.exposed), (0, 1, 0, False, None, ()))
            self.assertEqual(v.top, s.discard[-1])
            self.assertEqual(v.counts, (7, 7, 7, 7))
            self.assertEqual(v.suit, v.top[1])

    def test_deal_is_deterministic_and_seed_dependent(self):
        self.assertEqual(mk(4, "a"), mk(4, "a"))
        self.assertEqual(hand(mk(4, "a"), 0), hand(mk(4, "a"), 0))
        self.assertNotEqual([hand(mk(4, "a"), i) for i in range(4)], [hand(mk(4, "b"), i) for i in range(4)])

    def test_seat_count_limits_and_indexes(self):
        self.assertEqual((onecard.MIN_SEATS, onecard.MAX_SEATS), (2, 8))
        for n in (0, 1, 9):
            with self.assertRaises(ValueError):
                onecard.new("s", seats(n))
        with self.assertRaises(ValueError):
            onecard.new("s", [Seat(0, "member", 1, 0), Seat(2, "member", 2, 0)])

    def test_repr_hides_seed_hands_and_piles(self):
        s = mk(3, "very-secret-seed")
        text = repr(s)
        self.assertNotIn("very-secret-seed", text)
        self.assertNotIn(str(s.hands[0][0]), text)


class RulesTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(onecard.normalize_rules(), {
            "spade_ace_ten": False, "defense": True, "king_skips_two": True, "last_card": "none", "bankruptcy": False,
            "bankruptcy_limit": 20, "call_out_penalty": 1, "ante": 0, "turn_timeout": 60})
        self.assertEqual(onecard.normalize_rules({}), onecard.normalize_rules(None))
        self.assertEqual(onecard.normalize_rules(onecard.DEFAULT_RULES), onecard.normalize_rules())

    def test_partial_override_json_string_and_unknown_keys_dropped(self):
        got = onecard.normalize_rules(json.dumps({"defense": False, "call_out_penalty": 3, "bogus": 1}))
        self.assertEqual(got["defense"], False)
        self.assertEqual(got["call_out_penalty"], 3)
        self.assertNotIn("bogus", got)
        self.assertEqual(got["turn_timeout"], 60)

    def test_ranges_and_choices_are_validated(self):
        for ok in ({"bankruptcy_limit": 15}, {"bankruptcy_limit": 30}, {"call_out_penalty": 1}, {"call_out_penalty": 3},
                   {"turn_timeout": 15}, {"turn_timeout": 300}, {"last_card": "no_attack"}, {"last_card": "no_attack_or_joker"}):
            onecard.normalize_rules(ok)
        for bad in ({"bankruptcy_limit": 14}, {"bankruptcy_limit": 31}, {"call_out_penalty": 0}, {"call_out_penalty": 4},
                    {"turn_timeout": 14}, {"turn_timeout": 301}, {"last_card": "never"}, {"defense": "yes"},
                    {"king_skips_two": 1}, {"spade_ace_ten": None}, {"bankruptcy": "on"}, {"bankruptcy_limit": 20.5}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                onecard.normalize_rules(bad)

    def test_non_object_values_raise(self):
        for bad in ([], 5, "not json", "[1]"):
            with self.assertRaises(ValueError):
                onecard.normalize_rules(bad)

    def test_new_snapshots_normalized_rules_and_rejects_bad_ones(self):
        self.assertEqual(mk(3, rules={"last_card": "no_attack"}).rules["last_card"], "no_attack")
        with self.assertRaises(ValueError):
            mk(3, rules={"call_out_penalty": 9})


class PlayTests(unittest.TestCase):
    def test_play_format(self):
        self.assertEqual(onecard.card_text(c("10♥")), "10♥")
        self.assertEqual(onecard.card_text(c("JKb")), "JKb")
        self.assertEqual(onecard.card_text(c("JKc")), "JKc")
        self.assertEqual(onecard.play_move(c("A♠")), "play:A♠")
        self.assertEqual(onecard.play_move(c("7♥"), "♠"), "play:7♥=♠")

    def test_suit_or_rank_must_match(self):
        s = stage(hands=[["K♥", "5♣", "9♠", "7♦"]], top="5♥")
        self.assertEqual({m for m in legal(s, 0) if m.startswith("play:")}, {M("K♥"), M("5♣")})
        for text in ("9♠", "7♦"):
            with self.assertRaises(IllegalMove):
                play(s, 0, text)

    def test_jokers_play_on_anything_and_anything_plays_on_a_joker(self):
        s = stage(hands=[["JKb", "JKc", "9♠"]], top="5♥")
        self.assertIn(M("JKb"), legal(s, 0))
        self.assertIn(M("JKc"), legal(s, 0))
        s = stage(hands=[["9♠", "K♦", "3♣"]], top="JKb")
        self.assertEqual({m for m in legal(s, 0) if m.startswith("play:")}, {M("9♠"), M("K♦"), M("3♣")})

    def test_playing_moves_the_card_to_the_top_and_passes_the_turn(self):
        s = stage(hands=[["K♥", "5♣"]], top="5♥")
        t = play(s, 0, "K♥")
        self.assertEqual((hand(t, 0), t.discard[-1], t.turn), ([c("5♣")], c("K♥"), 1))
        ev = new_events(s, t)
        self.assertEqual([e.kind for e in ev], ["play"])
        self.assertEqual((ev[0].seat, ev[0].card), (0, c("K♥")))
        self.assertEqual(s.turn, 0)  # the old state is untouched
        self.assertEqual(len(s.hands[0]), 2)

    def test_only_the_turn_seat_moves(self):
        s = stage(hands=[["K♥", "5♣"], ["K♥", "5♣"]], top="5♥")
        self.assertEqual(onecard.legal_moves(s, 1), ())
        self.assertIn("draw", legal(s, 0))
        for move in (M("K♥"), "draw", "timeout"):
            with self.assertRaises(IllegalMove):
                onecard.apply(s, 1, move)

    def test_bad_moves_raise(self):
        s = stage(hands=[["K♥", "5♣"]], top="5♥")
        for seat, move in ((0, M("K♦")), (0, "nonsense"), (0, "play:"), (0, 5), (7, "draw"), (-1, "draw"), ("0", "draw")):
            with self.subTest(move=move), self.assertRaises(IllegalMove):
                onecard.apply(s, seat, move)

    def test_draw_gives_the_next_card_and_passes_even_with_a_play_in_hand(self):
        s = stage(hands=[["K♥", "5♣"]], top="5♥")
        t = onecard.apply(s, 0, "draw")
        self.assertEqual(len(t.hands[0]), 3)
        self.assertEqual(t.hands[0][-1], FILL[0])
        self.assertEqual((t.draw, t.turn, t.pending, t.discard), (FILL[1:], 1, 0, s.discard))
        (e,) = new_events(s, t)
        self.assertEqual((e.kind, e.seat, e.count, e.reason), ("draw", 0, 1, "pass"))
        self.assertEqual(walk_cards(e), [])

    def test_two_seat_turns_alternate(self):
        s = stage(n=2, hands=[["K♥", "5♣"], ["K♦", "5♦"]], top="5♥")
        s = onecard.apply(s, 0, "draw")
        self.assertEqual(s.turn, 1)
        s = onecard.apply(s, 1, "draw")
        self.assertEqual(s.turn, 0)

    def test_after_an_attack_is_drawn_the_next_seat_plays_on_the_top_card(self):
        s = stage(hands=[["2♥", "3♣"], ["9♣", "4♣"], ["3♥", "9♣", "J♦"]], top="5♥")
        s = onecard.apply(play(s, 0, "2♥"), 1, "draw")
        self.assertEqual((s.turn, s.pending), (2, 0))
        self.assertEqual({m for m in legal(s, 2) if m.startswith("play:")}, {M("3♥")})


class AttackTests(unittest.TestCase):
    def test_attack_values_and_the_next_seat_must_answer(self):
        for card, top, expect in (("2♥", "5♥", 2), ("A♥", "5♥", 3), ("A♠", "5♠", 5), ("JKb", "5♥", 5), ("JKc", "5♥", 7)):
            with self.subTest(card=card):
                s = stage(hands=[[card, "3♣"]], top=top)
                t = play(s, 0, card)
                self.assertEqual((t.pending, t.turn), (expect, 1))

    def test_spade_ace_ten_setting(self):
        s = stage(hands=[["A♠", "3♣"]], top="5♠", rules={"spade_ace_ten": True})
        self.assertEqual(play(s, 0, "A♠").pending, 10)
        s = stage(hands=[["A♥", "3♣"]], top="5♥", rules={"spade_ace_ten": True})
        self.assertEqual(play(s, 0, "A♥").pending, 3)

    def test_only_defenses_and_draw_are_legal_under_attack(self):
        s = stage(hands=[["9♥", "J♥", "Q♥", "K♥", "7♥", "5♥"]], top="2♥", pending=2)
        self.assertEqual(legal(s, 0), {"draw"})
        for text in ("9♥", "J♥", "K♥", "5♥"):
            with self.assertRaises(IllegalMove):
                play(s, 0, text)
        with self.assertRaises(IllegalMove):
            play(s, 0, "7♥", "♠")

    def test_defense_matrix(self):
        table = [
            ("2♥", 2, "2♦", True), ("2♥", 2, "A♥", True), ("2♥", 2, "A♦", False), ("2♥", 2, "A♠", False),
            ("2♠", 2, "A♠", True), ("2♥", 2, "JKb", True), ("2♥", 2, "JKc", True), ("2♥", 2, "3♥", False),
            ("2♥", 2, "K♥", False), ("2♥", 2, "7♥", False), ("2♥", 2, "J♥", False),
            ("A♥", 3, "A♦", True), ("A♥", 3, "A♠", True), ("A♥", 3, "JKb", True), ("A♥", 3, "JKc", True),
            ("A♥", 3, "2♥", False), ("A♥", 3, "K♥", False), ("A♥", 3, "5♥", False),
            ("A♠", 5, "JKb", True), ("A♠", 5, "JKc", False), ("A♠", 5, "A♥", False), ("A♠", 5, "A♠", False),
            ("A♠", 5, "2♠", False),
            ("JKb", 5, "JKc", True), ("JKb", 5, "JKb", True), ("JKc", 7, "JKb", True), ("JKc", 7, "JKc", True),
            ("JKb", 5, "A♠", False), ("JKb", 5, "2♥", False), ("JKc", 7, "9♥", False),
        ]
        for top, pend, cand, ok in table:
            with self.subTest(top=top, candidate=cand):
                s = stage(hands=[[cand, "9♣"]], top=top, pending=pend)
                self.assertEqual(M(cand) in legal(s, 0), ok)
                if ok:
                    t = play(s, 0, cand)
                    self.assertEqual((t.pending, t.turn, t.discard[-1]), (pend + value(c(cand)), 1, c(cand)))
                else:
                    with self.assertRaises(IllegalMove):
                        play(s, 0, cand)

    def test_a_stacked_attack_ends_with_the_whole_sum_drawn(self):
        s = stage(hands=[["2♥", "3♣", "4♣"], ["2♦", "9♣", "8♣"], ["A♦", "9♣", "8♣"]], top="5♥")
        s = play(s, 0, "2♥")
        s = play(s, 1, "2♦")
        self.assertEqual(s.pending, 4)
        s = play(s, 2, "A♦")  # the A of the same suit as the 2
        self.assertEqual((s.pending, s.turn), (7, 0))
        t = onecard.apply(s, 0, "draw")
        self.assertEqual(list(t.hands[0][-7:]), list(FILL[:7]))
        self.assertEqual((len(t.hands[0]), t.pending, t.turn, t.discard[-1]), (9, 0, 1, c("A♦")))
        (e,) = new_events(s, t)
        self.assertEqual((e.kind, e.seat, e.count, e.reason), ("draw", 0, 7, "attack"))

    def test_defense_off_leaves_only_drawing(self):
        rules = {"defense": False}
        s = stage(hands=[["2♥", "3♣"], ["2♦", "A♦", "JKb", "9♣"]], top="5♥", rules=rules)
        s = play(s, 0, "2♥")
        self.assertEqual(s.pending, 2)
        self.assertEqual(legal(s, 1), {"draw"})
        with self.assertRaises(IllegalMove):
            play(s, 1, "2♦")
        t = onecard.apply(s, 1, "draw")
        self.assertEqual((len(t.hands[1]), t.pending, t.turn), (6, 0, 2))

    def test_two_seat_attack_draw_comes_back_to_the_attacker(self):
        s = stage(n=2, hands=[["2♥", "3♣"], ["9♣", "8♣"]], top="5♥")
        s = onecard.apply(play(s, 0, "2♥"), 1, "draw")
        self.assertEqual((len(s.hands[1]), s.turn), (4, 0))

    def test_a_joker_attack_is_playable_on_any_card(self):
        s = stage(hands=[["JKc", "3♣"]], top="9♠")
        self.assertEqual(play(s, 0, "JKc").pending, 7)


class SpecialTests(unittest.TestCase):
    def test_jack_skips_the_next_seat(self):
        s = stage(hands=[["J♥", "3♣"]], top="5♥")
        self.assertEqual(play(s, 0, "J♥").turn, 2)

    def test_queen_reverses_the_direction(self):
        s = stage(n=4, hands=[["Q♥", "3♣"]], top="5♥")
        t = play(s, 0, "Q♥")
        self.assertEqual((t.turn, t.direction, onecard.view(t).direction), (3, -1, -1))
        t = onecard.apply(t, 3, "draw")
        self.assertEqual(t.turn, 2)

    def test_two_seats_jack_and_queen_give_the_turn_back(self):
        for text in ("J♥", "Q♥"):
            with self.subTest(card=text):
                s = stage(n=2, hands=[[text, "3♣"]], top="5♥")
                self.assertEqual(play(s, 0, text).turn, 0)

    def test_king_skips_two_by_default(self):
        s = stage(n=4, hands=[["K♥", "3♣"]], top="5♥")
        self.assertEqual(play(s, 0, "K♥").turn, 3)

    def test_king_does_nothing_when_the_toggle_is_off(self):
        s = stage(n=4, hands=[["K♥", "3♣"]], top="5♥", rules={"king_skips_two": False})
        self.assertEqual(play(s, 0, "K♥").turn, 1)

    def test_seven_names_a_suit_the_next_play_must_match(self):
        s = stage(hands=[["7♥", "3♣"], ["9♠", "5♥", "7♦", "JKb", "2♠"]], top="5♥")
        t = play(s, 0, "7♥", "♠")
        self.assertEqual((t.suit, t.turn, t.discard[-1], onecard.view(t).suit), ("♠", 1, c("7♥"), "♠"))
        self.assertEqual(new_events(s, t)[0].suit, "♠")
        plays = {m for m in legal(t, 1) if m.startswith("play:")}
        self.assertEqual(plays, {M("9♠"), M("JKb"), M("2♠")})  # not the ♥ nor the 7 by rank
        u = play(t, 1, "9♠")
        self.assertEqual((u.suit, onecard.view(u).suit), (None, "♠"))
        self.assertEqual(u.discard[-1], c("9♠"))

    def test_seven_must_name_exactly_one_suit(self):
        s = stage(hands=[["7♥", "3♥", "9♣"]], top="5♥")
        self.assertEqual({m for m in legal(s, 0) if m.startswith("play:7")}, {M("7♥", x) for x in SUITS})
        for args in (("7♥", None), ("7♥", "★"), ("3♥", "♠")):
            with self.subTest(args=args), self.assertRaises(IllegalMove):
                play(s, 0, *args)

    def test_a_named_suit_does_not_block_a_joker_attack(self):
        s = stage(hands=[["7♥", "3♣"], ["JKc", "9♣"]], top="5♥")
        t = play(play(s, 0, "7♥", "♠"), 1, "JKc")
        self.assertEqual(t.pending, 7)


class LastCardTests(unittest.TestCase):
    CARDS = ("2♠", "A♠", "JKb", "JKc", "K♠", "J♠", "Q♠", "9♠")

    def _allowed(self, rule):
        out = set()
        for text in self.CARDS + ("7♠",):
            s = stage(hands=[[text]], top="5♠", rules={"last_card": rule})
            move = M(text, "♥" if text == "7♠" else None)
            if move in legal(s, 0):
                out.add(text)
            else:
                with self.assertRaises(IllegalMove):
                    onecard.apply(s, 0, move)
            self.assertIn("draw", legal(s, 0))
        return out

    def test_none_allows_every_last_card(self):
        self.assertEqual(self._allowed("none"), set(self.CARDS) | {"7♠"})

    def test_no_attack_blocks_2_ace_and_spade_ace_only(self):
        self.assertEqual(self._allowed("no_attack"), {"JKb", "JKc", "K♠", "J♠", "Q♠", "9♠", "7♠"})

    def test_no_attack_or_joker_also_blocks_jokers(self):
        self.assertEqual(self._allowed("no_attack_or_joker"), {"K♠", "J♠", "Q♠", "9♠", "7♠"})

    def test_the_restriction_only_applies_to_the_last_card(self):
        for rule in ("no_attack", "no_attack_or_joker"):
            s = stage(hands=[["2♠", "JKb", "9♣"]], top="5♠", rules={"last_card": rule})
            self.assertIn(M("2♠"), legal(s, 0))
            self.assertIn(M("JKb"), legal(s, 0))

    def test_a_banned_last_card_cannot_defend_either(self):
        s = stage(hands=[["2♦"]], top="2♥", pending=2, rules={"last_card": "no_attack"})
        self.assertEqual(legal(s, 0), {"draw"})


class WinTests(unittest.TestCase):
    def test_emptying_the_hand_wins(self):
        s = stage(hands=[["9♥"]], top="5♥", stake=5)
        t = play(s, 0, "9♥")
        self.assertEqual((t.finished, t.winner, t.turn), (True, 0, None))
        self.assertIn("win", [e.kind for e in new_events(s, t)])
        self.assertEqual(onecard.legal_moves(t, 1), ())
        for seat, move in ((1, "draw"), (0, "draw"), (1, "declare"), (1, "call_out:0"), (1, "timeout")):
            with self.assertRaises(IllegalMove):
                onecard.apply(t, seat, move)
        r = onecard.result(t)
        self.assertEqual((r.finished, r.winner, r.pot), (True, 0, 15))
        self.assertEqual([(x.outcome, x.payout) for x in r.seats], [("win", 15), ("lose", 0), ("lose", 0)])

    def test_result_while_running(self):
        r = onecard.result(mk(3, stake=5))
        self.assertEqual((r.finished, r.winner, r.pot), (False, None, 15))
        self.assertEqual([(x.outcome, x.payout) for x in r.seats], [(None, 0)] * 3)

    def test_winning_on_an_attack_card_is_fine_without_a_restriction(self):
        s = stage(hands=[["2♥"]], top="5♥")
        self.assertTrue(play(s, 0, "2♥").finished)


class DrawPileTests(unittest.TestCase):
    def _empty_pile(self, **kw):
        return stage(hands=[["K♥", "5♣"]], top="5♥", draw=(), **kw)

    def test_exhausted_pile_reshuffles_the_discard_under_the_top(self):
        s = dataclasses.replace(self._empty_pile(), discard=cs("3♦", "4♦", "6♦", "8♦", "5♥"))
        t = onecard.apply(s, 0, "draw")
        self.assertEqual(t.discard, (c("5♥"),))
        self.assertEqual(len(t.hands[0]), 3)
        self.assertIn(t.hands[0][-1], cs("3♦", "4♦", "6♦", "8♦"))
        self.assertEqual(len(t.draw), 3)
        self.assertEqual(all_cards(t), all_cards(s))
        self.assertIn("reshuffle", [e.kind for e in new_events(s, t)])
        self.assertEqual(t, onecard.apply(s, 0, "draw"))  # deterministic

    def test_the_reshuffle_depends_on_the_seed(self):
        discard = tuple((r, s_) for s_ in SUITS[:3] for r in PLAIN) + (c("5♥"),)
        orders = set()
        for seed in ("a", "b", "c", "d"):
            s = dataclasses.replace(self._empty_pile(), seed=seed, discard=discard)
            orders.add(onecard.apply(s, 0, "draw").draw)
        self.assertGreater(len(orders), 1)

    def test_a_pending_draw_tops_up_from_the_reshuffle(self):
        s = stage(hands=[["K♥", "5♣"]], top="2♥", pending=5, draw=cs("3♣", "4♣"))
        s = dataclasses.replace(s, discard=cs("3♦", "4♦", "6♦", "8♦", "9♦", "2♥"))
        t = onecard.apply(s, 0, "draw")
        self.assertEqual(len(t.hands[0]), 7)
        self.assertEqual(list(t.hands[0][2:4]), list(cs("3♣", "4♣")))
        self.assertEqual((t.pending, t.discard, t.turn), (0, (c("2♥"),), 1))
        self.assertEqual(all_cards(t), all_cards(s))

    def test_nothing_left_to_draw_still_passes_the_turn(self):
        s = self._empty_pile()
        t = onecard.apply(s, 0, "draw")
        self.assertEqual((len(t.hands[0]), t.turn), (2, 1))


class BankruptcyTests(unittest.TestCase):
    RULES = {"bankruptcy": True, "bankruptcy_limit": 15}

    def _big(self, size, rules=None, n=3, turn=1):
        spare = tuple(["9♣"] * size)
        hands = [None] * n
        hands[turn] = spare
        return stage(n=n, hands=hands, top="5♥", turn=turn, rules=rules)

    def test_reaching_the_limit_eliminates_the_seat_and_returns_its_cards(self):
        s = self._big(14, self.RULES)
        t = onecard.apply(s, 1, "draw")
        self.assertEqual((t.active, t.hands[1], t.turn, t.finished), ((True, False, True), (), 2, False))
        self.assertEqual(len(t.draw), len(s.draw) + 14)
        self.assertEqual(all_cards(t), all_cards(s))
        self.assertIn("bankrupt", [e.kind for e in new_events(s, t)])
        self.assertEqual(onecard.view(t).counts[1], 0)
        self.assertEqual(t, onecard.apply(s, 1, "draw"))
        self.assertEqual(onecard.result(t).seats[1].outcome, "forfeit")

    def test_one_card_short_of_the_limit_stays_in(self):
        t = onecard.apply(self._big(13, self.RULES), 1, "draw")
        self.assertEqual((t.active, len(t.hands[1])), ((True, True, True), 14))

    def test_a_pending_attack_can_bankrupt_a_seat(self):
        s = dataclasses.replace(self._big(13, self.RULES), discard=cs("2♥"), pending=2)
        t = onecard.apply(s, 1, "draw")
        self.assertFalse(t.active[1])

    def test_off_by_default_big_hands_stay(self):
        t = onecard.apply(self._big(24), 1, "draw")
        self.assertEqual((t.active, len(t.hands[1])), ((True, True, True), 25))
        t = onecard.apply(self._big(29, {"bankruptcy_limit": 15}), 1, "draw")
        self.assertTrue(t.active[1])

    def test_default_limit_is_twenty(self):
        t = onecard.apply(self._big(19, {"bankruptcy": True}), 1, "draw")
        self.assertFalse(t.active[1])
        t = onecard.apply(self._big(18, {"bankruptcy": True}), 1, "draw")
        self.assertTrue(t.active[1])

    def test_turn_order_skips_an_eliminated_seat(self):
        t = onecard.apply(self._big(14, self.RULES), 1, "draw")
        t = onecard.apply(t, 2, "draw")
        self.assertEqual(t.turn, 0)
        t = onecard.apply(t, 0, "draw")
        self.assertEqual(t.turn, 2)
        self.assertEqual(onecard.legal_moves(t, 1), ())
        with self.assertRaises(IllegalMove):
            onecard.apply(t, 1, "declare")

    def test_the_last_seat_standing_wins(self):
        s = self._big(14, self.RULES, n=2, turn=1)
        t = onecard.apply(s, 1, "draw")
        self.assertEqual((t.finished, t.winner, t.turn), (True, 0, None))


class DeclarationTests(unittest.TestCase):
    def after_play_to_one(self, **kw):
        s = stage(hands=[["9♥", "3♣"], ["4♥", "3♦", "8♦"], ["5♣", "9♦"]], top="5♥", **kw)
        return s, play(s, 0, "9♥")

    def test_playing_down_to_one_undeclared_exposes_the_seat(self):
        s, t = self.after_play_to_one()
        self.assertEqual((t.exposed, t.declared[0], onecard.view(t).exposed), ((0,), False, (0,)))

    def test_call_out_on_an_exposed_seat_makes_it_draw_the_penalty(self):
        s, t = self.after_play_to_one()
        u = onecard.apply(t, 2, "call_out:0")  # off turn is fine
        self.assertEqual(u.hands[0], (c("3♣"),) + FILL[:1])
        self.assertEqual((len(u.hands[2]), u.turn, u.exposed), (2, 1, ()))
        (e,) = new_events(t, u)
        self.assertEqual((e.kind, e.caller, e.target, e.hit, e.penalty), ("callout", 2, 0, True, 1))

    def test_penalty_setting_scales_the_draw(self):
        for penalty in (2, 3):
            with self.subTest(penalty=penalty):
                s, t = self.after_play_to_one(rules={"call_out_penalty": penalty})
                u = onecard.apply(t, 1, "call_out:0")
                self.assertEqual(u.hands[0], (c("3♣"),) + FILL[:penalty])
                self.assertEqual(u.turn, 1)
                s, t = self.after_play_to_one(rules={"call_out_penalty": penalty})
                u = onecard.apply(t, 2, "call_out:1")  # false: seat 1 holds three cards
                self.assertEqual(len(u.hands[2]), 2 + penalty)
                self.assertEqual(len(u.hands[1]), 3)

    def test_false_call_out_makes_the_caller_draw(self):
        s, t = self.after_play_to_one()
        u = onecard.apply(t, 2, "call_out:1")
        self.assertEqual(u.hands[2], cs("5♣", "9♦") + FILL[:1])
        self.assertEqual(u.hands[1], t.hands[1])
        (e,) = new_events(t, u)
        self.assertEqual((e.kind, e.caller, e.target, e.hit, e.penalty), ("callout", 2, 1, False, 1))

    def test_calling_out_a_declared_seat_is_false(self):
        s, t = self.after_play_to_one()
        t = onecard.apply(t, 0, "declare")
        self.assertEqual((t.exposed, t.declared[0]), ((), True))
        u = onecard.apply(t, 2, "call_out:0")
        self.assertEqual((len(u.hands[2]), len(u.hands[0])), (3, 1))

    def test_a_declaration_made_before_the_play_counts(self):
        s = stage(hands=[["9♥", "3♣"]], top="5♥")
        s = onecard.apply(s, 0, "declare")
        self.assertTrue(s.declared[0])
        self.assertIn("declare", [e.kind for e in s.history])
        t = play(s, 0, "9♥")
        self.assertEqual((t.exposed, t.declared[0]), ((), True))
        u = onecard.apply(t, 2, "call_out:0")
        self.assertEqual((len(u.hands[2]), len(u.hands[0])), (4, 1))  # seat 2 holds the three spare clubs

    def test_declare_works_off_turn_and_with_one_or_two_cards_only(self):
        s = stage(hands=[["9♥", "3♣", "4♣"], ["4♥", "3♦"], ["5♣"]], top="5♥")
        self.assertTrue(onecard.apply(s, 1, "declare").declared[1])
        self.assertTrue(onecard.apply(s, 2, "declare").declared[2])
        with self.assertRaises(IllegalMove):
            onecard.apply(s, 0, "declare")

    def test_a_declaration_does_not_survive_drawing_to_three_cards(self):
        s = stage(hands=[["9♥", "3♣"]], top="5♥")
        s = onecard.apply(onecard.apply(s, 0, "declare"), 0, "draw")
        self.assertEqual((len(s.hands[0]), s.declared[0]), (3, False))

    def test_exposure_ends_when_the_next_seat_plays(self):
        s, t = self.after_play_to_one()
        u = play(t, 1, "4♥")
        self.assertEqual(u.exposed, ())
        v = onecard.apply(u, 2, "call_out:0")
        self.assertEqual((len(v.hands[0]), len(v.hands[2])), (1, 3))

    def test_exposure_ends_when_the_next_seat_draws(self):
        s, t = self.after_play_to_one()
        u = onecard.apply(t, 1, "draw")
        self.assertEqual(u.exposed, ())
        v = onecard.apply(u, 2, "call_out:0")
        self.assertEqual((len(v.hands[0]), len(v.hands[2])), (1, 3))

    def test_exposure_ends_on_a_timeout(self):
        s, t = self.after_play_to_one()
        u = onecard.apply(t, 1, "timeout")
        self.assertEqual(u.exposed, ())
        v = onecard.apply(u, 2, "call_out:0")
        self.assertEqual((len(v.hands[0]), len(v.hands[2])), (1, 3))

    def test_a_call_out_does_not_expire_the_exposure(self):
        s, t = self.after_play_to_one()
        u = onecard.apply(t, 2, "call_out:1")  # false
        self.assertEqual(u.exposed, (0,))
        v = onecard.apply(u, 2, "call_out:0")  # true
        self.assertEqual(len(v.hands[0]), 2)

    def test_call_out_target_rules(self):
        s, t = self.after_play_to_one()
        for caller, target in ((0, 0), (2, 2), (2, 7), (2, -1), (9, 0)):
            with self.subTest(caller=caller, target=target), self.assertRaises(IllegalMove):
                onecard.apply(t, caller, f"call_out:{target}")
        for move in ("call_out:", "call_out:x"):
            with self.assertRaises(IllegalMove):
                onecard.apply(t, 2, move)

    def test_two_seat_jack_to_one_card_is_exposed_until_the_same_seat_acts_again(self):
        s = stage(n=2, hands=[["J♥", "3♣"], ["4♥", "3♦"]], top="5♥")
        t = play(s, 0, "J♥")
        self.assertEqual((t.exposed, t.turn), ((0,), 0))
        self.assertEqual(onecard.apply(t, 0, "draw").exposed, ())


class TimeoutTests(unittest.TestCase):
    def test_timeout_draws_one_and_passes(self):
        s = stage(hands=[["K♥", "5♣"]], top="5♥")
        t = onecard.apply(s, 0, "timeout")
        self.assertEqual((t.hands[0], t.turn, t.pending), (cs("K♥", "5♣") + FILL[:1], 1, 0))
        (e,) = new_events(s, t)
        self.assertEqual((e.kind, e.seat, e.count, e.reason), ("draw", 0, 1, "timeout"))

    def test_timeout_draws_the_pending_attack(self):
        s = stage(hands=[["2♦", "5♣"]], top="2♥", pending=6)
        t = onecard.apply(s, 0, "timeout")
        self.assertEqual((len(t.hands[0]), t.pending, t.turn), (8, 0, 1))
        (e,) = new_events(s, t)
        self.assertEqual((e.count, e.reason), (6, "timeout"))

    def test_timeout_for_a_seat_not_on_turn_is_illegal(self):
        with self.assertRaises(IllegalMove):
            onecard.apply(stage(), 1, "timeout")


class ViewTests(unittest.TestCase):
    def test_public_view_has_counts_and_no_hands(self):
        s = mk(4, "views")
        v = onecard.view(s)
        self.assertIsNone(v.hand)
        self.assertEqual((v.seat, v.counts, v.decks, v.draw, v.discard), (None, (7,) * 4, 1, len(s.draw), 1))
        self.assertEqual(walk_cards(v, skip=("top", "history")), [])

    def test_seat_view_shows_only_its_own_hand(self):
        s = mk(4, "views")
        for i in range(4):
            v = onecard.view(s, i)
            self.assertEqual(sorted(v.hand), sorted(s.hands[i]))
            self.assertEqual(walk_cards(v, skip=("top", "history", "hand")), [])

    def test_history_never_leaks_drawn_cards(self):
        s = stage(hands=[["K♥", "5♣"]], top="5♥")
        t = onecard.apply(s, 0, "draw")
        self.assertEqual(walk_cards(onecard.view(t).history), [])

    def test_bad_seat(self):
        for bad in (-1, 3, 99):
            with self.assertRaises(ValueError):
                onecard.view(mk(3), bad)

    def test_non_int_seat_is_a_value_error(self):
        for bad in (True, 1.0, "1", [0]):
            with self.assertRaises(ValueError):
                onecard.view(mk(3), bad)

    def test_joker_top_has_no_suit(self):
        self.assertIsNone(onecard.view(stage(top="JKb")).suit)


class AnteAndTargetTests(unittest.TestCase):
    def test_ante_default_and_validation(self):
        self.assertEqual(onecard.DEFAULT_RULES["ante"], 0)
        self.assertEqual(onecard.normalize_rules()["ante"], 0)
        self.assertEqual(onecard.normalize_rules({"ante": 25})["ante"], 25)
        self.assertEqual(onecard.normalize_rules({"ante": onecard.ANTE_MAX})["ante"], onecard.ANTE_MAX)
        for bad in (-1, onecard.ANTE_MAX + 1, True, 1.5, "5", None):
            with self.assertRaises(ValueError):
                onecard.normalize_rules({"ante": bad})

    def test_huge_call_out_target_is_illegal(self):
        s = mk(3)
        for bad in ("call_out:" + "9" * 5000, "call_out:100", "call_out:001"):
            with self.assertRaises(IllegalMove):
                onecard.apply(s, 0, bad)

    def test_king_with_exactly_three_active_seats_advances_one_seat(self):
        s = stage(n=3, hands=[["K♥", "5♣"]], top="5♥")
        t = play(s, 0, "K♥")
        self.assertEqual(t.turn, 1)

    def test_call_out_bust_with_pending_attack_passes_the_attack_on(self):
        s = dataclasses.replace(
            stage(n=3, hands=[None, ["9♣"] * 14], top="2♥", turn=1, rules={"bankruptcy": True, "bankruptcy_limit": 15}), pending=2)
        t = onecard.apply(s, 1, "call_out:2")
        self.assertEqual((t.active, t.turn, t.pending), ((True, False, True), 2, 2))


class StoreAnteTests(TableCase):
    def test_onecard_round_stakes_the_ante_without_a_key_error(self):
        self.store.set_game_rules(G, "onecard", {"ante": 10})
        snap = self.store.open_table(G, CH, "onecard", ALICE, seed="a")
        self.fund(ALICE, 100)
        self.store.join_round(G, snap["round_id"], ALICE, 0)
        self.assertEqual(self.store.balance(G, ALICE), 90)


class FlowTests(unittest.TestCase):
    def _play_out(self, seed, n, rules=None, limit=4000):
        s = mk(n, seed, rules=rules)
        total = all_cards(s)
        moves = []
        for _ in range(limit):
            if s.finished:
                break
            seat, move = pick_move(s)
            moves.append((seat, move))
            s = onecard.apply(s, seat, move)
            self.assertEqual(all_cards(s), total)
        return s, moves

    def test_games_play_to_a_winner_and_cards_are_conserved(self):
        for n in (2, 3, 5, 8):
            with self.subTest(seats=n):
                s, moves = self._play_out(f"flow-{n}", n)
                self.assertTrue(s.finished)
                self.assertEqual(len(s.hands[s.winner]), 0)
                self.assertEqual(onecard.replay(f"flow-{n}", seats(n), moves), s)

    def test_games_with_all_house_rules_on_still_finish(self):
        rules = {"bankruptcy": True, "bankruptcy_limit": 15, "last_card": "no_attack_or_joker", "spade_ace_ten": True,
                 "call_out_penalty": 2}
        s, _ = self._play_out("house", 4, rules)
        self.assertTrue(s.finished)

    def test_json_round_trip_mid_game_and_at_the_end(self):
        s = mk(3, "json")
        total = all_cards(s)
        for k in range(12):
            seat, move = pick_move(s)
            s = onecard.apply(s, seat, move)
        text = onecard.to_json(s)
        self.assertIsInstance(json.loads(text), dict)
        back = onecard.from_json(text)
        self.assertEqual(back, s)
        self.assertEqual(all_cards(back), total)
        seat, move = pick_move(s)
        self.assertEqual(onecard.apply(back, seat, move), onecard.apply(s, seat, move))
        won = play(stage(hands=[["9♥"]], top="5♥"), 0, "9♥")
        self.assertEqual(onecard.from_json(onecard.to_json(won)), won)

    def test_replay_matches_and_rejects_illegal_logs(self):
        s = mk(3, "rp")
        seat, move = pick_move(s)
        self.assertEqual(onecard.replay("rp", seats(3), [(seat, move)]), onecard.apply(s, seat, move))
        with self.assertRaises(IllegalMove):
            onecard.replay("rp", seats(3), [(1, "draw")])


class ForfeitTests(unittest.TestCase):
    """FEAT-44: ``forfeit`` takes an active seat out like bankruptcy (any time, on or off turn); outcome "forfeit"."""

    def test_the_turn_seat_forfeits_and_the_turn_passes_on(self):
        s = stage(hands=[("9♣", "5♦"), ("3♠",), ("4♠",)])
        t = onecard.apply(s, 0, "forfeit")
        self.assertEqual((t.active, t.hands[0], t.turn, t.finished), ((False, True, True), (), 1, False))
        self.assertEqual(len(t.draw), len(s.draw) + 2)
        self.assertEqual(all_cards(t), all_cards(s))
        self.assertEqual([e.kind for e in new_events(s, t)], ["forfeit"])
        self.assertEqual(onecard.result(t).seats[0].outcome, "forfeit")
        self.assertEqual(t, onecard.apply(s, 0, "forfeit"))

    def test_a_seat_off_turn_can_forfeit_without_moving_the_turn(self):
        s = stage(turn=1)
        t = onecard.apply(s, 2, "forfeit")
        self.assertEqual((t.active, t.turn, t.hands[2]), ((True, True, False), 1, ()))
        t = onecard.apply(t, 1, "draw")
        self.assertEqual(t.turn, 0)

    def test_the_last_active_seat_wins(self):
        s = stage(n=2)
        t = onecard.apply(s, 0, "forfeit")
        self.assertEqual((t.finished, t.winner, t.turn), (True, 1, None))
        self.assertEqual([r.outcome for r in onecard.result(t).seats], ["forfeit", "win"])
        s = stage(n=3, stake=10)
        t = onecard.apply(onecard.apply(s, 0, "forfeit"), 1, "forfeit")
        self.assertEqual((t.finished, t.winner), (True, 2))
        self.assertEqual([(r.outcome, r.payout) for r in onecard.result(t).seats], [("forfeit", 0), ("forfeit", 0), ("win", 30)])

    def test_a_pending_attack_goes_to_the_next_seat(self):
        s = dataclasses.replace(stage(n=3, hands=[("9♣",), ("3♠",), ("4♠",)], top="2♥"), pending=2)
        t = onecard.apply(s, 0, "forfeit")
        self.assertEqual((t.turn, t.pending), (1, 2))
        off = onecard.apply(s, 2, "forfeit")
        self.assertEqual((off.turn, off.pending), (0, 2))

    def test_a_forfeited_seat_has_no_more_moves_and_is_skipped(self):
        s = stage()
        t = onecard.apply(s, 1, "forfeit")
        for move in ("forfeit", "draw", "declare"):
            with self.assertRaises(IllegalMove):
                onecard.apply(t, 1, move)
        self.assertEqual(onecard.legal_moves(t, 1), ())
        self.assertEqual(onecard.apply(t, 0, "draw").turn, 2)

    def test_the_cards_go_under_the_pile_and_the_marks_clear(self):
        s = dataclasses.replace(stage(hands=[("5♥",), ("9♣", "5♦"), None], turn=1), exposed=(0,), declared=(True, False, False))
        t = onecard.apply(s, 0, "forfeit")
        self.assertEqual((t.exposed, t.declared[0]), ((), False))
        self.assertEqual(t.draw[-1:], s.hands[0])

    def test_it_is_deterministic_and_replays_and_serializes(self):
        s = mk(4, "forfeit-seed")
        moves = [(0, "draw"), (1, "forfeit"), (2, "draw")]
        a, b = onecard.replay("forfeit-seed", seats(4), moves), onecard.replay("forfeit-seed", seats(4), moves)
        self.assertEqual(a, b)
        self.assertEqual((a.active, a.turn), ((True, False, True, True), 3))
        self.assertEqual(onecard.from_json(onecard.to_json(a)), a)
        self.assertIn("forfeit", [e.kind for e in a.history])
        self.assertEqual(all_cards(a), all_cards(s))

    def test_a_finished_game_takes_no_forfeit(self):
        t = onecard.apply(stage(n=2), 0, "forfeit")
        with self.assertRaises(IllegalMove):
            onecard.apply(t, 1, "forfeit")


class RegistryTests(unittest.TestCase):
    def test_onecard_entry(self):
        self.assertIn("onecard", registry.keys())
        g = registry.get("onecard")
        self.assertEqual((g.key, g.name, g.noun, g.ledger_label), ("onecard", "One Card", "One Card", "One Card"))
        self.assertIs(g.engine, onecard)
        self.assertIs(g.normalize_rules, onecard.normalize_rules)
        self.assertEqual((g.min_seats, g.max_seats, g.ante, g.cash), (2, 8, True, False))

    def test_existing_games_keep_their_entries(self):
        self.assertTrue({"blackjack", "doubt", "holdem"} <= set(registry.keys()))


if __name__ == "__main__":
    unittest.main()
