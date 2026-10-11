"""Texas Hold'em rules engine, assist and fallback policy (llmcord_core/games/holdem.py), FEAT-33 part A.

Written against the spec in docs/engineering/minigames-plan.md ("Hold'em engine API"). Assumed shapes: the
frozen ``State`` exposes ``seed, seats, rules, button, stacks, bets, contributed, folded, all_in, holes,
board_full, street, turn, min_raise_to, finished``; ``view(...)`` exposes ``board``-like cards, ``pot`` and
``turn``; ``result(...)`` returns ``Result(finished, board, pots, seats, shown)`` with ``Pot(amount, eligible,
winners, hand)`` and ``SeatResult(seat, start, end, net, outcome)``. Cards are (rank, suit) tuples (ranks
2..10 as ints, J/Q/K/A as strings). Hidden-information checks walk the objects for card tuples rather than
assuming field names. Where the spec is silent the tests assert invariants, not exact numbers.
"""
import dataclasses
import json
import random
import unittest

from llmcord_core.games import IllegalMove, Seat, playstyle
from llmcord_core.games import holdem

SUITS = ("♠", "♥", "♦", "♣")
SUIT_OF = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
VALUE = {**{n: n for n in range(2, 11)}, "J": 11, "Q": 12, "K": 13, "A": 14}


def C(text):
    """'As 10d 2c' -> list of (rank, suit) cards."""
    out = []
    for tok in text.split():
        rank, suit = tok[:-1], tok[-1]
        out.append((int(rank) if rank.isdigit() else rank, SUIT_OF[suit]))
    return out


def ev(text):
    return holdem.evaluate(C(text))


def seats(stacks):
    return [Seat(i, "member", 100 + i, s) for i, s in enumerate(stacks)]


def mk(stacks=(1000, 1000), seed="s", button=0, rules=None):
    return holdem.new(seed, seats(stacks), rules, button)


def act(state, *moves):
    for m in moves:
        state = holdem.apply(state, state.turn, m)
    return state


def check_down(state):
    while not state.finished:
        legal = holdem.legal_moves(state, state.turn)
        state = holdem.apply(state, state.turn, "check" if "check" in legal else "call")
    return state


def raises(state, seat=None):
    seat = state.turn if seat is None else seat
    return [int(m.split(":")[1]) for m in holdem.legal_moves(state, seat) if m.startswith("raise:")]


def cards_in(obj):
    """Every card tuple reachable inside a view (dataclass fields, containers)."""
    if isinstance(obj, tuple) and len(obj) == 2 and obj[1] in SUITS:
        return [obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        items = [getattr(obj, f.name) for f in dataclasses.fields(obj)]
    elif isinstance(obj, dict):
        items = list(obj.values())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        items = list(obj)
    else:
        return []
    return [c for item in items for c in cards_in(item)]


def eval7(state, i):
    return holdem.evaluate(list(state.holes[i]) + list(state.board_full))


def random_move(state, rng):
    seat = state.turn
    legal = holdem.legal_moves(state, seat)
    if any(m.startswith("raise:") for m in legal) and rng.random() < 0.3:
        hi = state.bets[seat] + state.stacks[seat]
        return f"raise:{rng.randint(min(state.min_raise_to, hi), hi)}"
    return rng.choice(legal)


def random_hand(index):
    """A seeded random-play hand: (start state, final state, moves)."""
    rng = random.Random(index)
    n = 2 + index % 8
    stacks = [rng.choice([20, 35, 60, 100, 250, 1000]) for _ in range(n)]
    state = mk(stacks, f"sweep-{index}", button=index % n)
    start, moves = state, []
    for _ in range(600):
        if state.finished:
            break
        move = random_move(state, rng)
        moves.append((state.turn, move))
        state = holdem.apply(state, state.turn, move)
    else:
        raise AssertionError("hand did not finish")
    return start, state, moves


def seed_where(pred, limit=6000, prefix="k"):
    for k in range(limit):
        seed = f"{prefix}{k}"
        if pred(seed):
            return seed
    raise AssertionError("no seed found")


class EvaluateTests(unittest.TestCase):
    LADDER = [
        "As Kd 9c 7h 4d", "As Ad 9c 7h 4d", "As Ad 9c 9h 4d", "As Ad Ac 7h 4d", "5s 6d 7c 8h 9d",
        "As Ks 9s 7s 4s", "As Ad Ac 7h 7d", "As Ad Ac Ah 4d", "5s 6s 7s 8s 9s",
    ]

    def test_categories_run_zero_to_eight_and_ascend(self):
        ranks = [ev(t) for t in self.LADDER]
        self.assertEqual([r[0] for r in ranks], list(range(9)))
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(len(set(ranks)), 9)

    def test_wheel_is_the_lowest_straight(self):
        wheel = ev("As 2d 3c 4h 5s")
        self.assertEqual(wheel[0], 4)
        self.assertLess(wheel, ev("2s 3d 4c 5h 6s"))
        self.assertGreater(wheel, ev("As Ad Ac 7h 4d"))
        self.assertEqual(ev("As 2d 3c 4h 5s Kd Kc"), wheel)

    def test_ace_does_not_wrap_around(self):
        self.assertEqual(ev("Qs Kd As 2d 3c")[0], 0)
        self.assertEqual(ev("Js Qs Ks As 2d")[0], 0)

    def test_ace_high_straight_beats_king_high(self):
        self.assertGreater(ev("10s Jd Qc Kh As"), ev("9s 10d Jc Qh Ks"))

    def test_seven_cards_take_the_highest_straight(self):
        self.assertEqual(ev("2s 3d 4c 5h 6d 7c 9h"), ev("3s 4d 5c 6h 7d"))

    def test_straight_flush_wheel_and_royal(self):
        wheel_sf = ev("As 2s 3s 4s 5s")
        self.assertEqual(wheel_sf[0], 8)
        self.assertLess(wheel_sf, ev("2s 3s 4s 5s 6s"))
        self.assertGreater(wheel_sf, ev("As Ad Ac Ah 4d"))
        royal = ev("10s Js Qs Ks As")
        self.assertEqual(royal[0], 8)
        self.assertGreater(royal, ev("9s 10s Js Qs Ks"))

    def test_royal_found_in_seven_cards_ignoring_hole_cards(self):
        self.assertEqual(ev("2c 3d As Ks Qs Js 10s"), ev("10s Js Qs Ks As"))

    def test_straight_flush_beats_a_higher_flush_in_seven_cards(self):
        got = ev("9h 8h 7h 6h 5h Ah Ad")
        self.assertEqual(got[0], 8)
        self.assertEqual(got, ev("5h 6h 7h 8h 9h"))

    def test_flush_beats_straight_and_trips_beats_nothing_less(self):
        self.assertEqual(ev("2s 3s 4s 5s 9s 6d Kh")[0], 5)
        self.assertEqual(ev("5s 6d 7c 8h 9d 9c 9h")[0], 4)

    def test_pair_ordering_and_kickers(self):
        self.assertGreater(ev("As Ad Kc Qh 7d"), ev("As Ad Kc Qh 6d"))
        self.assertGreater(ev("3s 3d 4c 5h 7d"), ev("2s 2d Ac Kh Qd"))
        self.assertGreater(ev("As Ad Kc 2h 3d"), ev("As Ad Qc Jh 10d"))

    def test_unused_cards_do_not_affect_the_rank(self):
        self.assertEqual(ev("Ah Kd 9c 7s 5d 3h 2c"), ev("As Kc 9d 7h 5c 3d 4s"))

    def test_two_pair_tiebreaks(self):
        self.assertGreater(ev("Ks Kd 5c 5h Ad"), ev("Ks Kd 5c 5h Qd"))
        self.assertGreater(ev("Ks Kd 6c 6h 2d"), ev("Ks Kd 5c 5h Ad"))
        self.assertGreater(ev("As Ad 2c 2h Kd"), ev("Ks Kd Qc Qh Jd"))

    def test_three_pairs_use_the_best_two_and_best_kicker(self):
        self.assertEqual(ev("As Ad Kc Kd Qs Qh 2c"), ev("As Ad Kc Kd Qs"))

    def test_trips_kickers(self):
        self.assertGreater(ev("7s 7d 7c Ah Kd"), ev("7s 7d 7c Ah Qd"))
        self.assertGreater(ev("8s 8d 8c 2h 3d"), ev("7s 7d 7c Ah Kd"))

    def test_full_house_tiebreaks(self):
        self.assertGreater(ev("Ks Kd Kc 2h 2d"), ev("Qs Qd Qc As Ah"))
        self.assertGreater(ev("Ks Kd Kc 3h 3d"), ev("Ks Kd Kc 2h 2d"))

    def test_full_house_from_two_trips_or_two_pairs_plus_trips(self):
        self.assertEqual(ev("As Ad Ac Ks Kd Kc 2h"), ev("As Ad Ac Ks Kd"))
        self.assertEqual(ev("Ks Kd Kc 2h 2d 3c 3s"), ev("Ks Kd Kc 3h 3d"))

    def test_quads_rank_then_kicker(self):
        self.assertGreater(ev("8s 8d 8c 8h 2d"), ev("7s 7d 7c 7h Ad"))
        self.assertGreater(ev("7s 7d 7c 7h Ac"), ev("7s 7d 7c 7h Kc"))
        self.assertEqual(ev("7s 7d 7c 7h Ac Kd 2s"), ev("7s 7d 7c 7h Ac"))

    def test_flush_compares_card_by_card(self):
        self.assertGreater(ev("As 9s 7s 4s 2s"), ev("Ks Qs Js 9s 7s"))
        self.assertGreater(ev("As Ks Qs Js 8s"), ev("As Ks Qs Js 7s"))
        self.assertEqual(ev("As Ks Qs 9s 7s 4s 2s"), ev("As Ks Qs 9s 7s"))

    def test_input_order_and_container_do_not_matter(self):
        cards = C("Ah Kd 9c 7s 5d 3h 2c")
        self.assertEqual(holdem.evaluate(cards), holdem.evaluate(list(reversed(cards))))
        self.assertEqual(holdem.evaluate(tuple(cards)), holdem.evaluate(cards))

    def test_wrong_card_counts_raise_value_error(self):
        with self.assertRaises(ValueError):
            holdem.evaluate(C("As Kd 9c 7h"))
        with self.assertRaises(ValueError):
            holdem.evaluate(C("As Kd 9c 7h 4d 3c 2s 5s"))

    def test_best_hand_returns_the_five_cards_used(self):
        cards = C("As Ad Kc Kd Qs Qh 2c")
        best = holdem.best_hand(cards)
        self.assertEqual(len(best), 5)
        self.assertTrue(set(best) <= set(cards))
        self.assertEqual(holdem.evaluate(best), holdem.evaluate(cards))
        self.assertNotIn((2, "♣"), best)

    def test_hand_name_text(self):
        self.assertEqual(holdem.hand_name(ev("Ks Kd 5c 5h 2d")), "Two pair, Kings and Fives")
        names = [holdem.hand_name(ev(t)) for t in self.LADDER]
        self.assertEqual(len(set(names)), 9)
        self.assertIn("pair", holdem.hand_name(ev("As Ad 9c 7h 4d")).lower())
        self.assertIn("Aces", holdem.hand_name(ev("As Ad 9c 7h 4d")))
        self.assertIn("full house", holdem.hand_name(ev("As Ad Ac 7h 7d")).lower())
        self.assertIn("five", holdem.hand_name(ev("As 2d 3c 4h 5s")).lower())
        royal = holdem.hand_name(ev("10s Js Qs Ks As")).lower()
        self.assertTrue("royal" in royal or "straight flush" in royal)


class RulesTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual((holdem.MIN_SEATS, holdem.MAX_SEATS), (2, 9))
        self.assertEqual(holdem.TIMEOUT_RANGE, (15, 300))
        self.assertEqual(holdem.DEFAULT_RULES, {
            "small_blind": 5, "min_buy_in_bb": 40, "max_buy_in_bb": 200, "action_timeout": 45, "assist": False,
        })

    def test_defaults_for_none_and_empty(self):
        self.assertEqual(holdem.normalize_rules(), holdem.DEFAULT_RULES)
        self.assertEqual(holdem.normalize_rules(None), holdem.DEFAULT_RULES)
        self.assertEqual(holdem.normalize_rules({}), holdem.DEFAULT_RULES)

    def test_returns_a_copy_not_the_default_dict(self):
        got = holdem.normalize_rules()
        got["small_blind"] = 99
        self.assertEqual(holdem.DEFAULT_RULES["small_blind"], 5)

    def test_json_string_and_partial_override(self):
        got = holdem.normalize_rules(json.dumps({"small_blind": 25, "assist": True}))
        self.assertEqual(got, {**holdem.DEFAULT_RULES, "small_blind": 25, "assist": True})
        self.assertEqual(holdem.normalize_rules("{}"), holdem.DEFAULT_RULES)

    def test_unknown_keys_are_ignored(self):
        got = holdem.normalize_rules({"small_blind": 7, "ante": 3, "big_blind": 99, "rake": 1})
        self.assertEqual(got, {**holdem.DEFAULT_RULES, "small_blind": 7})

    def test_boundary_values_accepted(self):
        for rules in (
            {"small_blind": 1}, {"small_blind": 10 ** 6}, {"min_buy_in_bb": 20}, {"max_buy_in_bb": 1000},
            {"min_buy_in_bb": 100, "max_buy_in_bb": 100}, {"action_timeout": 15}, {"action_timeout": 300},
            {"assist": True}, {"assist": False},
        ):
            with self.subTest(rules=rules):
                holdem.normalize_rules(rules)

    def test_each_bad_value_raises_value_error(self):
        bad = [
            {"small_blind": 0}, {"small_blind": -5}, {"small_blind": 10 ** 6 + 1}, {"small_blind": "5"},
            {"small_blind": True}, {"small_blind": None},
            {"min_buy_in_bb": 19}, {"min_buy_in_bb": "40"}, {"max_buy_in_bb": 1001},
            {"min_buy_in_bb": 50, "max_buy_in_bb": 49}, {"min_buy_in_bb": 201},
            {"action_timeout": 14}, {"action_timeout": 301}, {"action_timeout": "45"},
            {"assist": "yes"}, {"assist": 1}, {"assist": 0}, {"assist": None},
        ]
        for rules in bad:
            with self.subTest(rules=rules), self.assertRaises(ValueError):
                holdem.normalize_rules(rules)

    def test_non_object_or_invalid_json_raises_value_error(self):
        for value in ([], [1], 5, "not json", "[1]", '"x"'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                holdem.normalize_rules(value)

    def test_text_helpers_return_text_and_validate(self):
        self.assertIsInstance(holdem.rules_text(None), str)
        self.assertIn("\n", holdem.how_to_play(None))
        self.assertNotIn("\n", holdem.rules_text(None))
        self.assertIn("10", holdem.rules_text({"small_blind": 5}))
        with self.assertRaises(ValueError):
            holdem.rules_text({"small_blind": 0})
        with self.assertRaises(ValueError):
            holdem.how_to_play({"assist": "yes"})

    def test_big_blind_is_twice_the_small_blind(self):
        for sb in (1, 7, 25):
            with self.subTest(sb=sb):
                s = mk((1000, 1000), rules={"small_blind": sb})
                self.assertEqual(sorted(s.bets), [sb, 2 * sb])


class DealTests(unittest.TestCase):
    def test_deterministic_by_seed(self):
        self.assertEqual(mk(seed="same"), mk(seed="same"))
        deals = {mk(seed=f"d{k}").holes for k in range(5)}
        self.assertGreater(len(deals), 1)

    def test_all_dealt_cards_are_distinct_and_valid(self):
        s = mk((1000,) * 9, seed="full")
        cards = [c for h in s.holes for c in h] + list(s.board_full)
        self.assertEqual(len(cards), 9 * 2 + 5)
        self.assertEqual(len(set(cards)), len(cards))
        for rank, suit in cards:
            self.assertIn(rank, list(range(2, 11)) + ["J", "Q", "K", "A"])
            self.assertIn(suit, SUITS)
        self.assertTrue(all(len(h) == 2 for h in s.holes))

    def test_deal_order_is_one_card_at_a_time_from_left_of_the_button(self):
        s9 = mk((1000,) * 9, seed="order", button=0)
        deck = [s9.holes[(1 + i) % 9][0] for i in range(9)] + [s9.holes[(1 + i) % 9][1] for i in range(9)]
        deck += list(s9.board_full)
        for n in range(2, 10):
            for button in range(n):
                with self.subTest(n=n, button=button):
                    s = mk((1000,) * n, seed="order", button=button)
                    for i in range(n):
                        seat = (button + 1 + i) % n
                        self.assertEqual(tuple(s.holes[seat]), (deck[i], deck[n + i]))
                    self.assertEqual(tuple(s.board_full), tuple(deck[2 * n:2 * n + 5]))

    def test_heads_up_button_is_small_blind_and_acts_first(self):
        s = mk((1000, 1000), button=1)
        self.assertEqual((s.bets[1], s.bets[0]), (5, 10))
        self.assertEqual((s.stacks[1], s.stacks[0]), (995, 990))
        self.assertEqual(s.contributed, s.bets)
        self.assertEqual(s.turn, 1)
        self.assertEqual(s.street, "preflop")
        self.assertFalse(s.finished)
        self.assertEqual(mk((1000, 1000), button=0).turn, 0)

    def test_three_or_more_seats_blinds_and_first_actor(self):
        s = mk((1000,) * 3, button=0)
        self.assertEqual(s.bets, (0, 5, 10))
        self.assertEqual(s.turn, 0)
        s = mk((1000,) * 4, button=0)
        self.assertEqual(s.bets, (0, 5, 10, 0))
        self.assertEqual(s.turn, 3)
        s = mk((1000,) * 6, button=4)
        self.assertEqual((s.bets[5], s.bets[0]), (5, 10))
        self.assertEqual(sum(s.bets), 15)
        self.assertEqual(s.turn, 1)

    def test_standard_minimum_raise_after_the_blinds(self):
        s = mk((1000, 1000))
        self.assertEqual(s.min_raise_to, 20)
        self.assertIn("raise:20", holdem.legal_moves(s, s.turn))

    def test_short_stack_posts_what_it_has_and_is_all_in(self):
        s = mk((3, 1000), button=0)
        self.assertEqual(s.bets[0], 3)
        self.assertEqual(s.stacks[0], 0)
        self.assertTrue(s.all_in[0])
        self.assertEqual(s.contributed[0], 3)
        self.assertFalse(s.all_in[1])

    def test_bad_inputs_raise_value_error(self):
        good = seats((100, 100))
        cases = {
            "one seat": (seats((100,)), {}),
            "ten seats": (seats((100,) * 10), {}),
            "no seats": ([], {}),
            "gap in indexes": ([Seat(0, "member", 1, 100), Seat(2, "member", 2, 100)], {}),
            "duplicate indexes": ([Seat(0, "member", 1, 100), Seat(0, "member", 2, 100)], {}),
            "zero stack": (seats((100, 0)), {}),
            "button too high": (good, {"button": 2}),
            "negative button": (good, {"button": -1}),
            "bad rules": (good, {"rules": {"small_blind": 0}}),
        }
        for name, (given, kw) in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                holdem.new("s", given, **kw)

    def test_custom_rules_are_normalized_into_the_state(self):
        s = mk(rules={"small_blind": 7, "junk": 1})
        self.assertEqual(s.rules, holdem.normalize_rules({"small_blind": 7}))

    def test_repr_hides_seed_holes_and_board(self):
        s = mk((1000, 1000, 1000), seed="SECRET-SEED-XYZ")
        text = repr(s)
        self.assertNotIn("SECRET-SEED-XYZ", text)
        for card in [c for h in s.holes for c in h] + list(s.board_full):
            self.assertNotIn(repr(card), text)

    def test_state_is_immutable_and_apply_returns_a_new_state(self):
        s = mk()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            s.turn = 1
        before = (s.turn, s.bets, s.stacks, s.history)
        n = holdem.apply(s, s.turn, "call")
        self.assertIsNot(n, s)
        self.assertEqual((s.turn, s.bets, s.stacks, s.history), before)


class BettingTests(unittest.TestCase):
    def test_fold_is_omitted_when_check_is_available(self):
        s = act(mk(), "call")  # button limps, big blind has the option
        legal = holdem.legal_moves(s, s.turn)
        self.assertIn("check", legal)
        self.assertNotIn("fold", legal)
        self.assertNotIn("call", legal)

    def test_fold_and_call_listed_when_facing_a_bet(self):
        s = mk()
        legal = holdem.legal_moves(s, s.turn)
        self.assertEqual(legal[:2], ["fold", "call"])
        self.assertNotIn("check", legal)
        self.assertTrue(all(m.startswith("raise:") for m in legal[2:]))
        with self.assertRaises(IllegalMove):
            holdem.apply(s, s.turn, "check")

    def test_postflop_first_actor_can_check_but_not_fold(self):
        s = act(mk(), "call", "check")
        legal = holdem.legal_moves(s, s.turn)
        self.assertEqual(legal[0], "check")
        self.assertNotIn("fold", legal)

    def test_raise_menu_with_nobody_having_bet(self):
        s = act(mk(), "raise:60", "call")  # pot 120, stacks 940
        self.assertEqual(s.street, "flop")
        self.assertEqual(holdem.legal_moves(s, s.turn), ["check", "raise:10", "raise:60", "raise:120", "raise:940"])

    def test_raise_menu_dedupes_min_and_half_pot(self):
        s = act(mk(), "call", "check")  # pot 20: half pot == min bet 10
        self.assertEqual(holdem.legal_moves(s, s.turn), ["check", "raise:10", "raise:20", "raise:990"])

    def test_raise_menu_amounts_are_ascending_distinct_and_in_range(self):
        s = mk((1000, 1000, 1000, 1000))
        for move in ("raise:40", "call"):
            s = holdem.apply(s, s.turn, move)
        amounts = raises(s)
        seat = s.turn
        hi = s.bets[seat] + s.stacks[seat]
        self.assertEqual(amounts, sorted(set(amounts)))
        self.assertEqual(amounts[0], s.min_raise_to)
        self.assertEqual(amounts[-1], hi)
        self.assertTrue(all(s.min_raise_to <= a <= hi for a in amounts))

    def test_exact_raise_between_min_and_all_in_is_accepted(self):
        s = mk()
        n = holdem.apply(s, s.turn, "raise:37")
        self.assertEqual(n.bets[s.turn], 37)
        self.assertEqual(n.stacks[s.turn], 1000 - 37)
        self.assertEqual(n.contributed[s.turn], 37)
        self.assertNotEqual(n.turn, s.turn)
        self.assertEqual(holdem.apply(s, s.turn, "raise:1000").stacks[s.turn], 0)

    def test_illegal_moves_raise_illegal_move(self):
        s = mk()
        for move in ("raise:19", "raise:1001", "raise:10", "raise:0", "raise:-5", "raise:abc", "raise:", "raise",
                     "bet", "", "CALL", "all_in", "check", "raise:30.5"):
            with self.subTest(move=move), self.assertRaises(IllegalMove):
                holdem.apply(s, s.turn, move)

    def test_wrong_seat_raises_illegal_move(self):
        s = mk((1000,) * 3)
        for seat in range(3):
            if seat != s.turn:
                with self.subTest(seat=seat), self.assertRaises(IllegalMove):
                    holdem.apply(s, seat, "call")

    def test_finished_hand_accepts_no_move(self):
        s = act(mk(), "fold")
        self.assertTrue(s.finished)
        for seat in (0, 1):
            with self.assertRaises(IllegalMove):
                holdem.apply(s, seat, "check")

    def test_short_all_in_below_min_raise_is_listed_and_legal(self):
        s = mk((12, 1000), button=0)  # button posts 5, has 7 more: all-in total 12 < min raise-to 20
        self.assertEqual(raises(s), [12])
        for bad in ("raise:11", "raise:13", "raise:20"):
            with self.subTest(bad=bad), self.assertRaises(IllegalMove):
                holdem.apply(s, s.turn, bad)
        n = holdem.apply(s, s.turn, "raise:12")
        self.assertTrue(n.all_in[0])
        self.assertEqual(n.stacks[0], 0)

    def test_call_for_less_when_short(self):
        s = act(mk((100, 30)), "raise:100")
        self.assertEqual(raises(s), [])  # cannot raise: its whole stack is under the bet
        n = holdem.apply(s, s.turn, "call")
        self.assertTrue(n.finished)  # nobody can act any more
        self.assertEqual(n.contributed[1], 30)
        self.assertTrue(n.all_in[1])

    def test_streets_progress_and_board_grows(self):
        s = mk()
        sizes = []
        s = act(s, "call", "check")
        sizes.append((s.street, len(holdem.view(s).board)))
        s = act(s, "check", "check")
        sizes.append((s.street, len(holdem.view(s).board)))
        s = act(s, "check", "check")
        sizes.append((s.street, len(holdem.view(s).board)))
        self.assertEqual(sizes, [("flop", 3), ("turn", 4), ("river", 5)])
        self.assertEqual(holdem.view(mk()).board, ())
        s = act(s, "check", "check")
        self.assertTrue(s.finished)
        self.assertIn(s.street, ("showdown", "done"))

    def test_postflop_action_order(self):
        s = act(mk(), "call", "check")
        self.assertEqual(s.turn, 1)  # heads-up: the big blind acts first after the flop
        s = act(mk((1000,) * 3), "call", "call", "check")
        self.assertEqual((s.street, s.turn), ("flop", 1))  # first live seat left of the button
        s = act(mk((1000,) * 3, button=2), "call", "call", "check")
        self.assertEqual((s.street, s.turn), ("flop", 0))

    def test_big_blind_has_the_option_preflop(self):
        s = act(mk((1000,) * 3), "call", "call")
        self.assertEqual((s.street, s.turn), ("preflop", 2))
        legal = holdem.legal_moves(s, 2)
        self.assertIn("check", legal)
        self.assertTrue(any(m.startswith("raise:") for m in legal))
        n = holdem.apply(s, 2, "raise:30")
        self.assertEqual(n.street, "preflop")
        self.assertEqual(n.turn, 0)

    def test_folded_seat_is_skipped_in_later_streets(self):
        s = act(mk((1000,) * 3), "call", "fold", "check")
        self.assertTrue(s.folded[1])
        seen = set()
        while not s.finished:
            seen.add(s.turn)
            s = act(s, "check")
        self.assertEqual(seen, {0, 2})

    def test_everyone_all_in_runs_the_board_out(self):
        s = act(mk((100, 100)), "raise:100", "call")
        self.assertTrue(s.finished)
        self.assertEqual(len(holdem.view(s).board), 5)
        r = holdem.result(s)
        self.assertEqual(sum(p.amount for p in r.pots), 200)

    def test_all_in_seats_are_skipped(self):
        s = act(mk((30, 1000, 1000)), "raise:30", "call", "call")
        self.assertTrue(s.all_in[0])
        self.assertEqual(s.street, "flop")
        while not s.finished:
            self.assertNotEqual(s.turn, 0)
            s = act(s, "check")
        self.assertEqual(sum(holdem.result(s).seats[i].end for i in range(3)), 2030)

    def test_short_all_in_raise_does_not_reopen_betting(self):
        # Seat 1 bets 100 on the flop, seat 2 calls, seat 0 shoves 120 (a 20 raise, short of a full 100).
        s = act(mk((130, 1000, 1000)), "call", "call", "check")
        self.assertEqual((s.street, s.turn), ("flop", 1))
        s = act(s, "raise:100", "call", "raise:120")
        self.assertTrue(s.all_in[0])
        self.assertEqual(s.turn, 1)  # seat 1 bet 100 and has acted since: it may call or fold only
        self.assertEqual(holdem.legal_moves(s, 1)[:2], ["fold", "call"])
        self.assertEqual(raises(s, 1), [])
        with self.assertRaises(IllegalMove):
            holdem.apply(s, 1, "raise:240")
        s = holdem.apply(s, 1, "call")
        self.assertEqual(s.turn, 2)
        self.assertEqual(raises(s, 2), [])

    def test_full_raise_does_reopen_betting(self):
        s = act(mk((1000, 1000, 1000)), "call", "call", "check", "raise:100", "call", "raise:250")
        self.assertEqual(s.turn, 1)
        self.assertTrue(raises(s, 1))


class SidePotTests(unittest.TestCase):
    def shove_three(self, *moves):
        s = act(mk((50, 100, 200)), *moves)
        self.assertTrue(s.finished)
        return holdem.result(s)

    def test_three_different_all_in_stacks_make_main_and_side_pot(self):
        r = self.shove_three("raise:50", "raise:100", "call")
        self.assertEqual([p.amount for p in r.pots], [150, 100])
        self.assertEqual(set(r.pots[0].eligible), {0, 1, 2})
        self.assertEqual(set(r.pots[1].eligible), {1, 2})
        self.assertEqual(sum(sr.end for sr in r.seats), 350)

    def test_uncalled_excess_is_returned_not_a_pot(self):
        r = self.shove_three("raise:50", "call", "raise:200", "call")
        self.assertEqual([p.amount for p in r.pots], [150, 100])
        self.assertTrue(all(set(p.eligible) != {2} for p in r.pots))
        self.assertGreaterEqual(r.seats[2].end, 100)  # at least the returned 100
        self.assertEqual(sum(sr.end for sr in r.seats), 350)
        self.assertEqual([sr.net for sr in r.seats], [sr.end - sr.start for sr in r.seats])

    def test_folded_chips_stay_in_the_pot_but_the_seat_is_not_eligible(self):
        s = act(mk((1000,) * 3), "raise:30", "call", "fold")
        s = check_down(s)
        r = holdem.result(s)
        self.assertEqual(sum(p.amount for p in r.pots), 70)
        for p in r.pots:
            self.assertNotIn(2, p.eligible)
            self.assertNotIn(2, p.winners)
        self.assertEqual(r.seats[2].outcome, "fold")
        self.assertEqual(r.seats[2].end, 990)
        self.assertEqual(sum(sr.end for sr in r.seats), 3000)

    def test_all_in_seat_with_fewer_chips_only_wins_what_it_covered(self):
        # Whoever wins, nobody collects more from a seat than that seat put in.
        for k in range(40):
            s = act(mk((50, 500, 500), seed=f"cover{k}"), "raise:50", "call", "call")
            s = check_down(s)
            r = holdem.result(s)
            self.assertEqual(sum(sr.end for sr in r.seats), 1050)
            self.assertLessEqual(r.seats[0].end, 150)


class TieTests(unittest.TestCase):
    def test_heads_up_split_pot(self):
        seed = seed_where(lambda sd: eval7(mk((100, 100), sd), 0) == eval7(mk((100, 100), sd), 1), prefix="tie")
        s = act(mk((100, 100), seed), "raise:100", "call")
        r = holdem.result(s)
        self.assertEqual(set(r.pots[0].winners), {0, 1})
        self.assertEqual([sr.end for sr in r.seats], [100, 100])
        self.assertEqual([sr.outcome for sr in r.seats], ["split", "split"])

    def _odd_chip(self, script, tied, expected_bonus):
        def tie(sd):
            s = mk((100,) * 4, sd)
            return eval7(s, tied[0]) == eval7(s, tied[1])
        seed = seed_where(tie, prefix="odd")
        s = act(mk((100,) * 4, seed), *script)
        r = holdem.result(check_down(s))
        pot = sum(p.amount for p in r.pots)
        self.assertEqual(pot % 2, 1)
        self.assertEqual(set(r.pots[0].winners), set(tied))
        half = pot // 2
        other = tied[0] if expected_bonus == tied[1] else tied[1]
        paid_in = 30
        self.assertEqual(r.seats[expected_bonus].end, 100 - paid_in + half + 1)
        self.assertEqual(r.seats[other].end, 100 - paid_in + half)
        self.assertEqual(sum(sr.end for sr in r.seats), 400)

    def test_odd_chip_goes_to_first_winner_left_of_the_button(self):
        """Button 0: seat 3 raises, the button calls, both blinds fold (pot 75). Order from the button is 1, 2, 3, 0."""
        self._odd_chip(["raise:30", "call", "fold", "fold"], (3, 0), 3)

    def test_odd_chip_with_a_blind_among_the_winners(self):
        """Seat 3 folds, the button raises, the small blind folds, the big blind calls (pot 65): the big blind is first."""
        self._odd_chip(["fold", "raise:30", "fold", "call"], (0, 2), 2)


class ShowdownTests(unittest.TestCase):
    def test_nothing_is_shown_when_everyone_else_folds(self):
        s = act(mk((1000,) * 3), "fold", "fold")
        self.assertTrue(s.finished)
        r = holdem.result(s)
        self.assertEqual(r.shown, ())
        self.assertEqual([sr.outcome for sr in r.seats], ["fold", "fold", "win"])
        self.assertEqual(r.pots[0].winners, (2,))
        self.assertEqual(sum(sr.end for sr in r.seats), 3000)

    def test_last_aggressor_on_the_river_shows_first(self):
        wins = 0
        for k in range(40):
            s = act(mk(seed=f"agg{k}"), "raise:30", "call", "check", "check", "check", "check")
            self.assertEqual((s.street, s.turn), ("river", 1))
            s = act(s, "raise:20", "call")
            r = holdem.result(s)
            winners = {w for p in r.pots for w in p.winners}
            self.assertTrue(winners <= set(r.shown))
            if 1 in r.shown:
                self.assertEqual(r.shown[0], 1)
            if 1 in winners:
                wins += 1
                self.assertEqual(r.shown[0], 1)
        self.assertGreater(wins, 0)

    def test_without_a_bet_on_the_last_street_the_first_seat_left_of_the_button_shows_first(self):
        hits = 0
        for k in range(40):
            # The preflop raise does not count: aggression on the final street is what matters.
            s = check_down(act(mk(seed=f"first{k}"), "raise:30", "call"))
            r = holdem.result(s)
            winners = {w for p in r.pots for w in p.winners}
            self.assertTrue(winners <= set(r.shown))
            if 1 in winners:
                hits += 1
                self.assertEqual(r.shown[0], 1)
        self.assertGreater(hits, 0)

    def test_shown_has_no_duplicates_or_folded_seats(self):
        for i in range(40):
            _, final, _ = random_hand(i)
            r = holdem.result(final)
            self.assertEqual(len(r.shown), len(set(r.shown)))
            self.assertFalse(any(final.folded[i] for i in r.shown))


class ReplayAndViewTests(unittest.TestCase):
    def test_replay_equals_the_live_state(self):
        for i in range(0, 40, 3):
            start, final, moves = random_hand(i)
            again = holdem.replay(start.seed, start.seats, iter(moves), start.rules, start.button)
            self.assertEqual(again, final)
            self.assertEqual(holdem.result(again), holdem.result(final))

    def test_replay_of_a_prefix_matches_the_intermediate_state(self):
        start, final, moves = random_hand(5)
        live = start
        for k, (seat, move) in enumerate(moves[:6], start=1):
            live = holdem.apply(live, seat, move)
            self.assertEqual(holdem.replay(start.seed, start.seats, moves[:k], start.rules, start.button), live)

    def test_replay_defaults_and_illegal_move(self):
        s = mk()
        self.assertEqual(holdem.replay(s.seed, s.seats, []), s)
        with self.assertRaises(IllegalMove):
            holdem.replay(s.seed, s.seats, [(s.turn, "check")])

    def test_view_hides_other_holes_and_future_board(self):
        s = act(mk((1000,) * 4, seed="hide"), "call", "call", "call", "check")  # flop
        self.assertEqual(s.street, "flop")
        shown_board = set(s.board_full[:3])
        future = set(s.board_full[3:])
        for seat in range(4):
            found = set(cards_in(holdem.view(s, seat)))
            self.assertTrue(set(s.holes[seat]) <= found)
            self.assertTrue(shown_board <= found)
            self.assertFalse(found & future)
            for other in range(4):
                if other != seat:
                    self.assertFalse(found & set(s.holes[other]), (seat, other))

    def test_spectator_view_has_no_hole_cards(self):
        s = mk((1000,) * 3, seed="spec")
        self.assertEqual(cards_in(holdem.view(s)), [])
        self.assertEqual(cards_in(holdem.view(s, None)), [])
        s = act(s, "call", "call", "check")
        self.assertEqual(set(cards_in(holdem.view(s))), set(s.board_full[:3]))

    def test_view_carries_pot_and_turn(self):
        s = act(mk((1000,) * 3), "raise:40", "call")
        v = holdem.view(s, 1)
        self.assertEqual(v.pot, sum(s.contributed))
        self.assertEqual(v.turn, s.turn)


class InvariantSweepTests(unittest.TestCase):
    HANDS = 240

    def test_chips_are_conserved_and_results_are_consistent(self):
        for i in range(self.HANDS):
            start, final, _ = random_hand(i)
            with self.subTest(hand=i):
                r = holdem.result(final)
                total = sum(s.stake for s in start.seats)
                self.assertTrue(r.finished)
                self.assertEqual(sum(sr.start for sr in r.seats), total)
                self.assertEqual(sum(sr.end for sr in r.seats), total)
                for sr in r.seats:
                    self.assertEqual(sr.net, sr.end - sr.start)
                    self.assertGreaterEqual(sr.end, 0)
                self._check_pots(final, r)

    def _check_pots(self, final, r):
        n = len(final.seats)
        live = [i for i in range(n) if not final.folded[i]]
        contrib = sorted(final.contributed, reverse=True)
        uncalled = max(0, contrib[0] - contrib[1])
        self.assertEqual(sum(p.amount for p in r.pots), sum(final.contributed) - uncalled)
        previous = set(range(n))
        winners_all = set()
        for p in r.pots:
            self.assertGreater(p.amount, 0)
            self.assertTrue(p.eligible)
            self.assertTrue(set(p.eligible) <= set(live))
            self.assertTrue(set(p.eligible) <= previous)
            previous = set(p.eligible)
            self.assertTrue(p.winners)
            self.assertTrue(set(p.winners) <= set(p.eligible))
            winners_all |= set(p.winners)
            if len(live) > 1:
                best = max(eval7(final, e) for e in p.eligible)
                self.assertEqual(set(p.winners), {e for e in p.eligible if eval7(final, e) == best})
        for sr in r.seats:
            if final.folded[sr.seat]:
                self.assertEqual(sr.outcome, "fold")
            elif sr.seat in winners_all:
                if sr.net > 0:
                    self.assertIn(sr.outcome, ("win", "split"))
                else:
                    self.assertIn(sr.outcome, ("win", "split", "lose"))
            else:
                self.assertEqual(sr.outcome, "lose")
        if len(live) == 1:
            self.assertEqual(r.shown, ())
            self.assertEqual(winners_all, set(live))
        else:
            self.assertTrue(winners_all <= set(r.shown))
            self.assertFalse(set(r.shown) & {i for i in range(n) if final.folded[i]})

    def test_state_invariants_hold_after_every_move(self):
        for i in range(0, self.HANDS, 4):
            rng = random.Random(i)
            n = 2 + i % 8
            stacks = [rng.choice([20, 60, 250, 1000]) for _ in range(n)]
            s = mk(stacks, f"inv-{i}", button=i % n)
            total = sum(stacks)
            while not s.finished:
                legal = holdem.legal_moves(s, s.turn)
                self.assertTrue(legal)
                self.assertEqual(len(legal), len(set(legal)))
                am = [int(m.split(":")[1]) for m in legal if m.startswith("raise:")]
                self.assertEqual(am, sorted(set(am)))
                hi = s.bets[s.turn] + s.stacks[s.turn]
                self.assertTrue(all(min(s.min_raise_to, hi) <= a <= hi for a in am))
                self.assertEqual(("check" in legal), ("fold" not in legal) and ("call" not in legal))
                s = holdem.apply(s, s.turn, random_move(s, rng))
                if not s.finished:
                    self.assertEqual(sum(s.stacks) + sum(s.contributed), total)
                    self.assertFalse(s.folded[s.turn] or s.all_in[s.turn])


class AssistTests(unittest.TestCase):
    def test_deterministic_for_the_same_inputs(self):
        a = holdem.assist(C("As Kd"), [], 3, "seed", samples=300)
        b = holdem.assist(C("As Kd"), [], 3, "seed", samples=300)
        self.assertEqual(a, b)
        c = holdem.assist(C("As Kd"), C("2c 7d 9h"), 2, "seed", samples=300)
        self.assertEqual(c, holdem.assist(C("As Kd"), C("2c 7d 9h"), 2, "seed", samples=300))

    def test_chances_shape(self):
        for board in ("", "2c 7d 9h", "2c 7d 9h Jc", "2c 7d 9h Jc 3s"):
            with self.subTest(board=board):
                a = holdem.assist(C("As Kd"), C(board), 2, "seed", samples=300)
                self.assertLessEqual(len(a.chances), 3)
                self.assertGreaterEqual(len(a.chances), 1)
                probs = [p for _, p in a.chances]
                self.assertEqual(probs, sorted(probs, reverse=True))
                self.assertLessEqual(sum(probs), 1 + 1e-9)
                self.assertTrue(all(0 <= p <= 1 for p in probs))
                self.assertTrue(all(round(p, 3) == p for p in probs))
                self.assertEqual(len({name for name, _ in a.chances}), len(a.chances))
                self.assertIsInstance(a.current, str)

    def test_exact_enumeration_on_the_flop_and_turn_ignores_the_seed(self):
        for board in ("2c 7d 9h", "2c 7d 9h Jc"):
            with self.subTest(board=board):
                a = holdem.assist(C("As Kd"), C(board), 2, "one")
                b = holdem.assist(C("As Kd"), C(board), 2, "two")
                self.assertEqual((a.current, a.chances), (b.current, b.chances))

    def test_known_turn_probabilities(self):
        """Trip twos on the turn (board 2d 7c 9s Jc, 46 unseen): 36/46 stay trips,
        9/46 fill up (three 7s, 9s or Js left), 1/46 quads (2c)."""
        a = holdem.assist(C("2s 2h"), C("2d 7c 9s Jc"), 1, "x")
        probs = [p for _, p in a.chances]
        self.assertAlmostEqual(probs[0], 36 / 46, delta=0.001)
        self.assertAlmostEqual(probs[1], 9 / 46, delta=0.001)
        self.assertAlmostEqual(probs[2], 1 / 46, delta=0.001)
        self.assertIn("three", a.chances[0][0].lower())
        self.assertIn("full house", a.chances[1][0].lower())
        self.assertIn("four", a.chances[2][0].lower())
        self.assertAlmostEqual(sum(probs), 1.0, delta=0.002)

    def test_river_is_certain(self):
        a = holdem.assist(C("As Ks"), C("Qs Js 10s 2d 3c"), 1, "x")
        self.assertEqual(len(a.chances), 1)
        self.assertEqual(a.chances[0][1], 1.0)

    def test_current_hand_name(self):
        self.assertIn("pair", holdem.assist(C("7s 7h"), [], 1, "x", samples=50).current.lower())
        self.assertTrue(holdem.assist(C("As 2h"), [], 1, "x", samples=50).current.lower().startswith("high card"))
        made = holdem.assist(C("As Kd"), C("Ac 5d 9c"), 1, "x").current
        self.assertEqual(made, holdem.hand_name(holdem.evaluate(C("As Kd Ac 5d 9c"))))

    def test_chances_do_not_depend_on_the_opponent_count(self):
        a = holdem.assist(C("As Kd"), C("2c 7d 9h"), 1, "x")
        b = holdem.assist(C("As Kd"), C("2c 7d 9h"), 6, "x")
        self.assertEqual((a.current, a.chances), (b.current, b.chances))
        p = holdem.assist(C("As Ad"), [], 1, "x", samples=200)
        q = holdem.assist(C("As Ad"), [], 6, "x", samples=200)
        self.assertEqual(p.chances, q.chances)

    def test_win_chance_is_deterministic_and_more_opponents_lower_it(self):
        a = holdem.assist(C("As Ad"), C("2c 7d 9h"), 1, "x", samples=400)
        self.assertEqual(a.win, holdem.assist(C("As Ad"), C("2c 7d 9h"), 1, "x", samples=400).win)
        b = holdem.assist(C("As Ad"), C("2c 7d 9h"), 5, "x", samples=400)
        self.assertTrue(0 <= b.win <= a.win <= 1)

    def test_inputs_are_not_mutated(self):
        hole, board = C("As Kd"), C("2c 7d 9h")
        holdem.assist(hole, board, 1, "x")
        self.assertEqual((hole, board), (C("As Kd"), C("2c 7d 9h")))


STYLES = [
    playstyle.PlayStyle(),
    playstyle.PlayStyle(**{t: 0 for t in playstyle.TRAITS}),
    playstyle.PlayStyle(**{t: 100 for t in playstyle.TRAITS}),
    playstyle.PlayStyle(looseness=0, unpredictability=100, judgment=0),
    playstyle.PlayStyle(aggression=100, bluffing=100, risk=100, looseness=100, judgment=100, unpredictability=0),
    playstyle.PlayStyle(stubbornness=100, unpredictability=100, risk=0),
]


class PolicyTests(unittest.TestCase):
    def test_always_a_legal_move_deterministic_and_never_folds_free(self):
        checked = free = 0
        for i in range(30):
            rng = random.Random(i)
            n = 2 + i % 8
            s = mk([rng.choice([20, 60, 250, 1000]) for _ in range(n)], f"pol-{i}", button=i % n)
            while not s.finished:
                seat = s.turn
                legal = holdem.legal_moves(s, seat)
                v = holdem.view(s, seat)
                self.assertEqual(
                    holdem.timeout_move_for_view(v), "check" if "check" in legal else "fold")
                for j, style in enumerate(STYLES):
                    move = holdem.policy_action(v, style, f"p{i}-{j}")
                    self.assertIn(move, legal)
                    self.assertEqual(move, holdem.policy_action(v, style, f"p{i}-{j}"))
                    if "check" in legal:
                        free += 1
                        self.assertNotEqual(move, "fold")
                    checked += 1
                s = holdem.apply(s, seat, random_move(s, rng))
        self.assertGreater(checked, 300)
        self.assertGreater(free, 50)

    def test_policy_can_play_every_seat_of_a_hand(self):
        for i in range(12):
            n = 2 + i % 8
            s = mk((200,) * n, f"auto-{i}", button=i % n)
            for step in range(300):
                if s.finished:
                    break
                style = STYLES[(s.turn + i) % len(STYLES)]
                s = holdem.apply(s, s.turn, holdem.policy_action(holdem.view(s, s.turn), style, f"a{i}-{step}"))
            self.assertTrue(s.finished)
            r = holdem.result(s)
            self.assertEqual(sum(sr.end for sr in r.seats), 200 * n)

    def test_never_folds_when_checking_is_free_even_at_full_unpredictability(self):
        style = playstyle.PlayStyle(unpredictability=100, looseness=0, aggression=0, judgment=0)
        free = 0
        for k in range(120):
            s = act(mk((1000, 1000), f"free{k}"), "call")  # big blind option
            v = holdem.view(s, s.turn)
            for j in range(4):
                free += 1
                self.assertNotEqual(holdem.policy_action(v, style, f"f{k}-{j}"), "fold")
        self.assertEqual(free, 480)

    def test_tight_style_folds_trash_facing_a_raise_more_than_a_loose_one(self):
        def trash(hole):
            (r1, s1), (r2, s2) = hole
            a, b = VALUE[r1], VALUE[r2]
            return max(a, b) <= 9 and abs(a - b) >= 3 and s1 != s2

        tight = playstyle.PlayStyle(looseness=0, unpredictability=0)
        loose = playstyle.PlayStyle(looseness=100, unpredictability=0)
        tight_folds = loose_folds = hands = 0
        for k in range(2000):
            s = mk((1000,) * 3, f"trash{k}")
            if not trash(s.holes[1]):
                continue
            s = holdem.apply(s, 0, "raise:30")  # seat 1 (small blind) faces the raise
            v = holdem.view(s, 1)
            hands += 1
            tight_folds += holdem.policy_action(v, tight, f"t{k}") == "fold"
            loose_folds += holdem.policy_action(v, loose, f"t{k}") == "fold"
            if hands == 80:
                break
        self.assertEqual(hands, 80)
        self.assertGreater(tight_folds, loose_folds)

    def test_aggressive_style_raises_more_with_a_strong_hand(self):
        def strong(hole):
            (r1, _), (r2, _) = hole
            return (r1 == r2 and VALUE[r1] >= 12) or {r1, r2} == {"A", "K"}

        calm = playstyle.PlayStyle(aggression=0, unpredictability=0)
        bold = playstyle.PlayStyle(aggression=100, unpredictability=0)
        calm_raises = bold_raises = hands = 0
        for k in range(4000):
            s = mk((1000, 1000), f"strong{k}")  # button acts first
            if not strong(s.holes[0]):
                continue
            v = holdem.view(s, 0)
            calm_raises += holdem.policy_action(v, calm, f"s{k}").startswith("raise:")
            bold_raises += holdem.policy_action(v, bold, f"s{k}").startswith("raise:")
            hands += 1
            if hands == 40:
                break
        self.assertEqual(hands, 40)
        self.assertGreater(bold_raises, calm_raises)

    def test_timeout_move_is_check_else_fold(self):
        s = mk()
        self.assertEqual(holdem.timeout_move_for_view(holdem.view(s, s.turn)), "fold")
        s = act(s, "call")
        self.assertEqual(holdem.timeout_move_for_view(holdem.view(s, s.turn)), "check")
        s = act(s, "check")
        self.assertEqual(holdem.timeout_move_for_view(holdem.view(s, s.turn)), "check")


def find_seed(stacks, pred, limit=2000):
    """First seed f's{n}' whose deal satisfies pred(state); fails loudly when none does."""
    for n in range(limit):
        seed = f"s{n}"
        if pred(mk(stacks, seed)):
            return seed
    raise AssertionError("no seed found")


def assert_conserved(test, state):
    r = holdem.result(state)
    total = sum(x.stake for x in state.seats)
    test.assertTrue(r.finished)
    test.assertEqual(sum(sr.start for sr in r.seats), total)
    test.assertEqual(sum(sr.end for sr in r.seats), total)
    for sr in r.seats:
        test.assertEqual(sr.net, sr.end - sr.start)
        test.assertGreaterEqual(sr.end, 0)
    return r


class ShortBlindTests(unittest.TestCase):
    """Review finding 1: a big blind below the small blind or the big blind must not make the other seat act."""

    def test_known_defect_heads_up_big_blind_below_small_blind_asks_nobody(self):
        """FEAT-33 (Opus review 1): the small blind already matched all the big blind can win; no decision is asked."""
        s = mk((1000, 3), button=0)  # seat 0 posts 5, seat 1 is all-in for 3: nothing left to decide
        self.assertTrue(s.finished)
        self.assertIsNone(s.turn)
        self.assertEqual(holdem.legal_moves(s, 0), [])
        r = assert_conserved(self, s)
        self.assertGreaterEqual(r.seats[0].end, 997)  # at most the 3 that were matched is at risk
        self.assertNotIn("fold", [m for _, m in s.history])

    def test_three_handed_short_big_blind_still_asks_the_button_to_match_the_full_blind(self):
        s = mk((1000, 1000, 3), button=0)  # seat 1 posts 5, seat 2 is all-in for 3
        self.assertEqual((s.turn, s.to_call), (0, 10))
        self.assertEqual(holdem.legal_moves(s, 0)[:2], ["fold", "call"])

    def test_known_defect_three_handed_after_the_button_folds_the_small_blind_is_not_asked(self):
        """FEAT-33 (Opus review 1): small blind 5 vs a 3 all-in with nobody else live has nothing to decide."""
        s = act(mk((1000, 1000, 3), button=0), "fold")
        self.assertTrue(s.finished)
        self.assertIsNone(s.turn)
        assert_conserved(self, s)

    def test_known_defect_heads_up_big_blind_all_in_for_seven_costs_the_small_blind_two(self):
        """FEAT-33 (Opus review 1): to call is 2 (the 7 the big blind could post minus the 5 already in), not 5."""
        s = mk((1000, 7), button=0)
        self.assertEqual(s.turn, 0)
        self.assertEqual(holdem.view(s, 0).to_call, 2)
        self.assertIn("call", holdem.legal_moves(s, 0))
        n = holdem.apply(s, 0, "call")
        self.assertEqual((n.contributed[0], n.stacks[0]), (7, 993))
        self.assertTrue(n.finished)
        assert_conserved(self, n)


class NoRaiseWithoutResponderTests(unittest.TestCase):
    def setUp(self):
        # Seat 0 shoves 100, the small blind folds: the big blind faces an all-in with nobody left to answer a raise.
        self.s = act(mk((100, 1000, 1000)), "raise:100", "fold")

    def test_known_defect_no_raise_options_when_every_other_live_seat_is_all_in(self):
        """FEAT-33 (Opus review 2): a raise nobody can respond to is not offered."""
        self.assertEqual(self.s.turn, 2)
        legal = holdem.legal_moves(self.s, 2)
        self.assertEqual(legal[:2], ["fold", "call"])
        self.assertEqual([m for m in legal if m.startswith("raise:")], [])
        v = holdem.view(self.s, 2)
        self.assertFalse(v.can_raise)
        self.assertEqual([m for m in holdem.legal_moves_for_view(v) if m.startswith("raise:")], [])

    def test_known_defect_applying_a_raise_with_no_responder_is_illegal(self):
        """FEAT-33 (Opus review 2): a raise with no live seat able to respond raises IllegalMove."""
        with self.assertRaises(IllegalMove):
            holdem.apply(self.s, 2, "raise:300")


class RaiseParsingTests(unittest.TestCase):
    def test_known_defect_leading_zeros_are_illegal(self):
        """FEAT-33 (Opus review 3): raise:020 is not the canonical spelling of raise:20."""
        s = mk()
        for move in ("raise:020", "raise:0020", "raise:00"):
            with self.subTest(move=move), self.assertRaises(IllegalMove):
                holdem.apply(s, s.turn, move)

    def test_thirteen_digit_amount_is_illegal(self):
        s = mk()
        with self.assertRaises(IllegalMove):
            holdem.apply(s, s.turn, "raise:" + "9" * 13)

    def test_known_defect_absurdly_long_amount_is_illegal_not_value_error(self):
        """FEAT-33 (Opus review 3): int() of a 4301+ digit string raises ValueError, which escapes apply()."""
        s = mk()
        for move in ("raise:" + "9" * 5000, "raise:" + "1" * 4301):
            with self.subTest(length=len(move)), self.assertRaises(IllegalMove):
                holdem.apply(s, s.turn, move)

    def test_a_valid_raise_is_stored_canonically(self):
        s = mk()
        n = holdem.apply(s, s.turn, "raise:37")
        self.assertEqual(n.history[-1], (s.turn, "raise:37"))


class RunOutViewTests(unittest.TestCase):
    def test_known_defect_view_bets_are_zero_once_the_board_has_run_out(self):
        """FEAT-33 (Opus review 4): a finished all-in hand shows no open bets."""
        s = act(mk((100, 100)), "raise:100", "call")
        self.assertTrue(s.finished)
        self.assertIn(s.street, ("showdown", "done"))
        self.assertEqual(holdem.view(s).bets, (0, 0))

    def test_run_out_keeps_the_last_aggressor_showing_first(self):
        s = act(mk((100, 100)), "raise:100", "call")
        self.assertEqual(holdem.result(s).shown, (0, 1))
        s = act(mk((100, 100), button=1), "raise:100", "call")  # seat 1 is the button and shoves first
        self.assertEqual(holdem.result(s).shown, (1, 0))
        self.assertEqual(holdem.result(s).shown[0], s.aggressor)


class ViewSafetyTests(unittest.TestCase):
    def test_known_defect_view_repr_does_not_show_the_hole_cards(self):
        """FEAT-33 (Opus review 5): printing a View must not leak the viewer's hole cards."""
        s = mk((1000, 1000, 1000), seed="repr-hole")
        v = holdem.view(s, 0)
        self.assertEqual(tuple(v.hole), tuple(s.holes[0]))
        text = repr(v)
        for card in s.holes[0]:
            self.assertNotIn(repr(card), text)

    def test_view_of_a_negative_or_out_of_range_seat_is_a_spectator_view(self):
        """Characterization (Opus review 5): view() does not validate the seat; unknown seats see no hole cards."""
        s = mk((1000, 1000, 1000), seed="oob")
        for seat in (-1, -3, 3, 99):
            with self.subTest(seat=seat):
                v = holdem.view(s, seat)
                self.assertIsNone(v.hole)
                self.assertEqual(cards_in(v), [])
                self.assertEqual(holdem.legal_moves_for_view(v), [])


class AlreadyFinishedDealTests(unittest.TestCase):
    def test_blinds_that_put_everyone_all_in_finish_the_hand_at_the_deal(self):
        cases = [((3, 1000), 0), ((1000, 3), 1), ((3, 4), 0), ((3, 4), 1), ((4, 3), 0), ((4, 3), 1)]
        for stacks, button in cases:
            with self.subTest(stacks=stacks, button=button):
                s = mk(stacks, button=button)
                self.assertTrue(s.finished)
                self.assertIsNone(s.turn)
                for seat in (0, 1):
                    self.assertEqual(holdem.legal_moves(s, seat), [])
                    with self.assertRaises(IllegalMove):
                        holdem.apply(s, seat, "call")
                assert_conserved(self, s)
                self.assertEqual(len(holdem.view(s).board), 5)


class ExactSidePotTests(unittest.TestCase):
    def test_several_all_ins_and_a_fold_between_the_levels(self):
        stacks = (50, 100, 200, 1000)
        seed = find_seed(stacks, lambda st: eval7(st, 0) > eval7(st, 2) > eval7(st, 1))
        s = act(mk(stacks, seed), "raise:75", "call", "raise:100", "raise:200", "fold")
        self.assertTrue(s.finished)
        self.assertEqual(s.contributed, (50, 100, 200, 75))
        r = holdem.result(s)
        # Levels 50 (everyone), 75 (folded seat 3 stops) and 100 (seat 2's extra 100 is uncalled and refunded).
        self.assertEqual([(p.amount, p.eligible) for p in r.pots], [(200, (0, 1, 2)), (125, (1, 2))])
        self.assertEqual([p.winners for p in r.pots], [(0,), (2,)])
        self.assertEqual([sr.end for sr in r.seats], [200, 0, 225, 925])
        self.assertEqual([sr.net for sr in r.seats], [150, -100, 25, -75])
        self.assertEqual([sr.outcome for sr in r.seats], ["win", "lose", "win", "fold"])

    def test_side_pot_won_by_a_different_seat_than_the_main_pot(self):
        """Reading: outcome is 'win'/'split' only for a seat with net > 0, 'lose' otherwise, 'fold' for folded seats."""
        stacks = (50, 100, 100)
        seed = find_seed(stacks, lambda st: eval7(st, 0) > eval7(st, 1) > eval7(st, 2))
        s = act(mk(stacks, seed), "raise:50", "raise:100", "call")
        r = holdem.result(s)
        self.assertEqual([(p.amount, p.eligible, p.winners) for p in r.pots],
                         [(150, (0, 1, 2), (0,)), (100, (1, 2), (1,))])
        self.assertEqual([sr.end for sr in r.seats], [150, 100, 0])
        self.assertEqual([sr.net for sr in r.seats], [100, 0, -100])
        self.assertEqual(r.seats[0].outcome, "win")
        self.assertEqual(r.seats[2].outcome, "lose")

    def test_known_defect_seat_that_only_gets_its_own_chips_back_is_not_a_winner(self):
        """FEAT-33 (Opus review 7b): winning a side pot with net 0 is labelled 'lose', not 'win'."""
        stacks = (50, 100, 100)
        seed = find_seed(stacks, lambda st: eval7(st, 0) > eval7(st, 1) > eval7(st, 2))
        r = holdem.result(act(mk(stacks, seed), "raise:50", "raise:100", "call"))
        self.assertEqual(r.seats[1].net, 0)
        self.assertEqual(r.seats[1].outcome, "lose")

    def test_three_way_split_with_remainder_two_and_an_odd_chip_in_the_side_pot(self):
        base = mk((11, 30, 30, 100), button=0)
        royal = tuple(C("As Ks Qs Js 10s"))
        holes = (tuple(C("2h 3d")), tuple(C("2d 3h")), tuple(C("2c 3c")), tuple(C("4h 4d")))
        s = dataclasses.replace(
            base, holes=holes, board_full=royal, shown_n=5, street="showdown", turn=None, finished=True,
            contributed=(11, 30, 30, 12), stacks=(0, 0, 0, 88), bets=(0, 0, 0, 0), folded=(False, False, False, True),
            all_in=(True, True, True, False), aggressor=None,
        )
        r = holdem.result(s)
        # Level 12 and below: 44 for seats 0-2 (seat 3's 12 is dead money), then 39 for seats 1-2 only.
        self.assertEqual([(p.amount, p.eligible) for p in r.pots], [(44, (0, 1, 2)), (39, (1, 2))])
        self.assertEqual([set(p.winners) for p in r.pots], [{0, 1, 2}, {1, 2}])
        # Odd chips go to the first winners left of the button (order 1, 2, 3, 0): 44 -> 15, 15, 14; 39 -> 20, 19.
        self.assertEqual([sr.end for sr in r.seats], [14, 35, 34, 88])
        self.assertEqual(sum(sr.end for sr in r.seats), 171)
        self.assertEqual([sr.outcome for sr in r.seats], ["split", "split", "split", "fold"])

    def test_everyone_folds_to_a_raise_the_raiser_gets_the_excess_back(self):
        s = act(mk((1000, 1000, 1000)), "raise:60", "fold", "fold")
        r = holdem.result(s)
        # The pot holds the blinds plus the raiser's own matched 10; its other 50 comes back.
        self.assertEqual([(p.amount, p.eligible, p.winners) for p in r.pots], [(25, (0,), (0,))])
        self.assertEqual([sr.end for sr in r.seats], [1015, 995, 990])
        self.assertEqual(r.seats[0].net, 15)
        self.assertEqual(r.shown, ())


class CumulativeShortAllInTests(unittest.TestCase):
    def test_first_bettor_may_only_call_or_fold_after_two_short_all_ins(self):
        """Characterization (Opus review 8): bet 100, all-in to 150 (+50), all-in to 210 (+60). Together the two
        add 110 (more than a full 100 raise) but neither is a full raise alone, so betting does not reopen for the
        first bettor. Documented rules choice (a strict per-raise comparison)."""
        s = act(mk((220, 1000, 160)), "call", "call", "check")
        self.assertEqual((s.street, s.turn), ("flop", 1))
        s = act(s, "raise:100", "raise:150", "raise:210")
        self.assertTrue(s.all_in[0] and s.all_in[2])
        self.assertEqual(s.turn, 1)
        self.assertEqual(holdem.legal_moves(s, 1), ["fold", "call"])
        self.assertFalse(holdem.view(s, 1).can_raise)
        with self.assertRaises(IllegalMove):
            holdem.apply(s, 1, "raise:400")
        n = holdem.apply(s, 1, "call")
        self.assertTrue(n.finished)
        assert_conserved(self, n)


if __name__ == "__main__":
    unittest.main()
