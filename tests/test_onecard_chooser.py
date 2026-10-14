"""FEAT-44 step B: the pure side of One Card character seats (the Play style policy, the model prompt/schema, the ask call).

TDD-red tests written before the implementation. Offline and pure: rigged engine states (``rig``) stand in for deals, the structured
model call is a fake. Mirrors ``test_doubt_chooser.py`` / ``test_holdem_chooser.py``; the table-level behavior is in ``test_onecard_characters.py``.

Assumed API (a proposal, pinned here):

  ``llmcord_core/games/onecard.py`` (pure; next to ``playable_cards``):
    ``legal_moves_for_view(view, rules) -> tuple``    exactly ``legal_moves(state, view.seat)`` (same order) computed from the seat's own
                                                    ``View`` plus the round's rules dict; ``()`` off turn, when out, or when finished.
                                                    (``View`` has no rules, hence the extra argument, as for ``playable_cards``.)
    ``policy_move(view, rules, style, seed) -> str``  one legal move (``play:<card>``, ``play:7<suit>=<named suit>`` or ``draw``) for the
                                                    viewing seat from its own view only; ``style`` is a ``PlayStyle`` or ``None`` (neutral);
                                                    every random draw comes from ``seed``; raises ``IllegalMove`` when it is not that
                                                    seat's turn. Heuristic: finish when the last card can be played; answer a pending
                                                    attack with a defense when there is one; else hit with an attack card (2, A, joker...)
                                                    when it can; else dump a special card (7, J, Q, K) before a plain one; after a 7 name
                                                    the suit it holds most of; ``draw`` when nothing is playable. Erratic play (trait
                                                    ``unpredictability``) may deviate to another legal move; the tests that pin a specific
                                                    move use ``PlayStyle(unpredictability=0)``.
  ``llmcord_core/game_chooser.py`` (next to ``ask_holdem``):
    ``onecard_schema(legal) -> dict``                 same shape as ``schema(legal)``: ``move`` (enum = the legal moves) and ``line``,
                                                    both required, no extras.
    ``onecard_table_text(view, names, legal) -> str`` ``names`` is a list of display names indexed by seat (characters by name, members
                                                    "Player 1", "Player 2", ... in seat order). Lines the tests rely on:
                                                      ``Your hand: 9♥, JKb, ...``    (``onecard.card_text`` tokens, in the view's order)
                                                      ``Top card: <card>``  ``Suit: <suit>``
                                                      ``Pending attack: <n>``       only while an attack is pending
                                                      ``<name>: <n> card(s)``       for every other seat
                                                      ``Draw pile: <n> card(s)``    a count, never its cards
                                                      ``<name> played <card>`` / ``<name> drew``   for recent public events
                                                      ``Legal moves: <move>, <move>, ...``  exactly the menu
    ``async ask_onecard(bot, guild_id, channel_id, character, view, names, legal, style=None) -> dict``
                                                    like ``ask_holdem``: system message "You are <name>, ..." + persona (+ ``Play style:``
                                                    for a saved style), user message = ``onecard_table_text``, role ``director``, structured
                                                    name ``onecard_move``, usage feature/purpose ``game``; returns the raw answer.
"""
import dataclasses
import itertools
import unittest

from test_doubt_chooser import CARD, fake_bot

from llmcord_core import game_chooser
from llmcord_core.games import onecard
from llmcord_core.games.base import IllegalMove, Seat, shuffled
from llmcord_core.games.playstyle import PlayStyle, describe

G, CH = 1, 100
STEADY = PlayStyle(unpredictability=0)
FULL_DECK = [(r, s) for s in onecard.SUITS for r in onecard.RANKS] + list(onecard.JOKERS)


def c(text):
    """A card from its text (``9♥``, ``10♠``, ``JKb``, ``JKc``)."""
    if text in ("JKb", "JKc"):
        return onecard.JOKERS[0 if text == "JKb" else 1]
    return (text[:-1], text[-1])


def seats_of(n):
    return [Seat(k, "member", k + 1, 0) for k in range(n)]


def rig(hands, top, *, pending=0, suit=None, direction=1, rules=None, turn=0, seed="rig"):
    """A one-deck state (2 to 4 seats) with the given hands (card texts) and top card; the draw pile is the rest of the deck."""
    state = onecard.new(seed, seats_of(len(hands)), rules)
    hands = tuple(tuple(c(t) for t in hand) for hand in hands)
    left = list(FULL_DECK)
    for card in [*itertools.chain.from_iterable(hands), c(top)]:
        left.remove(card)
    return dataclasses.replace(state, hands=hands, draw=tuple(shuffled(left, seed)), discard=(c(top),), pending=pending,
                               suit=suit, turn=turn, direction=direction)


def view_of(state, seat=0):
    return onecard.view(state, seat)


def move_of(hand, top, *, seat_style=STEADY, seed="p", **kwargs):
    state = rig([hand, ["3♣", "8♠", "8♦"]], top, **kwargs)
    return onecard.policy_move(view_of(state), state.rules, seat_style, seed)


class LegalForViewTests(unittest.TestCase):
    def test_it_matches_the_engine_on_dealt_and_rigged_states(self):
        states = [onecard.new(f"fv-{i}", seats_of(3)) for i in range(25)]
        states += [rig([["2♥", "7♥", "9♣", "JKb"], ["4♣"]], "5♥"), rig([["2♣", "A♥", "JKc"], ["4♣"]], "2♥", pending=2),
                   rig([["9♦", "10♠"], ["4♣"]], "5♥", suit="♠"), rig([["9♦", "10♠"], ["4♣"]], "7♥", suit="♠")]
        for state in states:
            for seat in range(len(state.seats)):
                self.assertEqual(onecard.legal_moves_for_view(onecard.view(state, seat), state.rules), onecard.legal_moves(state, seat))

    def test_it_follows_the_rules_snapshot(self):
        rules = {"defense": False, "last_card": "no_attack_or_joker"}
        state = rig([["2♣", "9♠"], ["4♣"]], "2♥", pending=2, rules=rules)
        self.assertEqual(onecard.legal_moves_for_view(view_of(state), state.rules), ("draw",))
        state = rig([["2♥"], ["4♣"]], "5♥", rules=rules)  # a last card may not be an attack
        self.assertEqual(onecard.legal_moves_for_view(view_of(state), state.rules), ("draw",))

    def test_it_is_empty_off_turn_and_when_finished(self):
        state = rig([["2♥", "9♥"], ["4♣"]], "5♥", turn=1)
        self.assertEqual(onecard.legal_moves_for_view(view_of(state, 0), state.rules), ())
        state = onecard.apply(rig([["9♥"], ["4♣"]], "5♥"), 0, "play:9♥")
        self.assertTrue(state.finished)
        self.assertEqual(onecard.legal_moves_for_view(view_of(state, 1), state.rules), ())


class PolicyLegalityTests(unittest.TestCase):
    STYLES = (None, STEADY, PlayStyle(unpredictability=100), PlayStyle(judgment=0, aggression=100, risk=100), PlayStyle(looseness=0, aggression=0))
    RULES = ({}, {"defense": False}, {"last_card": "no_attack_or_joker"}, {"bankruptcy": True, "bankruptcy_limit": 15}, {"king_skips_two": False})

    def play_game(self, seed, n, style, rules, limit=600):
        state, taken = onecard.new(seed, seats_of(n), rules), 0
        while not state.finished and taken < limit:
            seat = state.turn
            move = onecard.policy_move(onecard.view(state, seat), state.rules, style, f"{seed}/{taken}")
            self.assertIn(move, onecard.legal_moves(state, seat), f"{seed} move {taken}")
            state = onecard.apply(state, seat, move)
            taken += 1
        return state, taken

    def test_every_move_of_a_whole_game_is_legal_for_any_style_and_rules(self):
        for style, rules in itertools.product(self.STYLES, self.RULES):
            for seed, n in (("leg-a", 2), ("leg-b", 3), ("leg-c", 4)):
                with self.subTest(style=style, rules=rules, seed=seed):
                    self.play_game(seed, n, style, rules)

    def test_policy_play_finishes_games(self):
        """A steady policy table ends in a win within a bounded number of moves (it draws when stuck and does not loop)."""
        finished = sum(self.play_game(f"fin-{i}", 3, STEADY, {})[0].finished for i in range(6))
        self.assertGreaterEqual(finished, 4)

    def test_the_same_view_and_seed_give_the_same_move(self):
        state = onecard.new("det", seats_of(3))
        view = onecard.view(state, 0)
        self.assertEqual({onecard.policy_move(view, state.rules, PlayStyle(), "same") for _ in range(5)}, {onecard.policy_move(view, state.rules, PlayStyle(), "same")})

    def test_it_refuses_a_seat_that_is_not_on_turn(self):
        state = rig([["2♥", "9♥"], ["4♣", "6♠"]], "5♥", turn=1)
        with self.assertRaises(IllegalMove):
            onecard.policy_move(view_of(state, 0), state.rules, STEADY, "s")

    def test_it_does_not_read_the_other_hands(self):
        """Two deals that differ only in the other seats' cards give the same move for the same view and seed."""
        a = rig([["2♥", "9♥", "J♣", "4♠"], ["4♣", "6♠", "6♦"]], "5♥")
        b = rig([["2♥", "9♥", "J♣", "4♠"], ["K♠", "A♦", "3♣", "JKb"]], "5♥")
        self.assertEqual(onecard.policy_move(view_of(a), a.rules, PlayStyle(), "same"), onecard.policy_move(view_of(b), b.rules, PlayStyle(), "same"))


class PolicyHeuristicTests(unittest.TestCase):
    def test_it_draws_when_nothing_is_playable(self):
        self.assertEqual(move_of(["9♣", "10♠", "K♦"], "5♥"), "draw")

    def test_it_draws_under_an_attack_it_cannot_answer(self):
        self.assertEqual(move_of(["9♥", "10♥", "K♥"], "2♥", pending=2), "draw")

    def test_it_draws_under_an_attack_when_defense_is_off(self):
        self.assertEqual(move_of(["2♣", "9♥"], "2♥", pending=2, rules={"defense": False}), "draw")

    def test_it_defends_with_an_attack_when_it_can(self):
        self.assertEqual(move_of(["2♣", "5♠", "9♦"], "2♥", pending=2), "play:2♣")
        self.assertEqual(move_of(["A♥", "5♠", "9♦"], "2♥", pending=2), "play:A♥")  # an ace of the attack's suit answers a 2
        self.assertEqual(move_of(["JKc", "4♠", "9♦"], "A♥", pending=3), "play:JKc")

    def test_it_defends_with_the_only_legal_answer_even_against_a_big_attack(self):
        self.assertEqual(move_of(["JKb", "4♠", "9♦"], "A♠", pending=5), "play:JKb")

    def test_it_hits_with_an_attack_card_when_it_can(self):
        self.assertEqual(move_of(["4♥", "2♥", "9♥"], "5♥"), "play:2♥")
        self.assertEqual(move_of(["9♦", "A♦", "4♦"], "5♦"), "play:A♦")

    def test_an_attack_is_preferred_to_a_special_card(self):
        self.assertEqual(move_of(["J♥", "2♥", "9♣"], "5♥"), "play:2♥")

    def test_it_dumps_a_special_card_before_a_plain_one(self):
        self.assertEqual(move_of(["4♥", "9♥", "J♥"], "5♥"), "play:J♥")
        self.assertEqual(move_of(["4♥", "Q♥", "10♥"], "5♥"), "play:Q♥")

    def test_it_plays_the_only_playable_card(self):
        self.assertEqual(move_of(["4♥", "9♣", "K♠"], "5♥"), "play:4♥")

    def test_it_plays_its_last_card_to_win(self):
        self.assertEqual(move_of(["9♥"], "5♥"), "play:9♥")
        self.assertIn(move_of(["7♥"], "5♥"), [f"play:7♥={s}" for s in onecard.SUITS])

    def test_it_names_the_suit_it_holds_most_after_a_seven(self):
        self.assertEqual(move_of(["7♥", "9♣", "10♣", "4♣", "K♠"], "5♥"), "play:7♥=♣")
        self.assertEqual(move_of(["7♥", "9♠", "K♠", "4♦"], "5♥"), "play:7♥=♠")

    def test_a_tie_between_suits_names_one_of_them_every_time(self):
        for seed in ("a", "b", "c", "d"):
            move = move_of(["7♥", "9♣", "10♣", "4♠", "K♠"], "5♥", seed=seed)
            self.assertIn(move, ("play:7♥=♣", "play:7♥=♠"))
            self.assertEqual(move, move_of(["7♥", "9♣", "10♣", "4♠", "K♠"], "5♥", seed=seed))

    def test_it_follows_a_named_suit(self):
        # after a 7 named spades the 9♥ matches nothing, the 4♠ does
        self.assertEqual(move_of(["9♥", "4♠", "K♦"], "7♥", suit="♠"), "play:4♠")

    def test_a_joker_is_playable_on_anything_and_never_illegal(self):
        move = move_of(["JKb", "9♣"], "5♥")
        self.assertIn(move, ("play:JKb", "draw"))  # the exact choice is the policy's, but it must be legal
        state = rig([["JKb", "9♣"], ["4♣"]], "5♥")
        self.assertIn(onecard.policy_move(view_of(state), state.rules, STEADY, "s"), onecard.legal_moves(state, 0))

    def test_an_erratic_style_still_moves_legally_and_can_deviate(self):
        state = rig([["2♥", "9♥", "J♥", "4♥"], ["4♣"]], "5♥")
        moves = {onecard.policy_move(view_of(state), state.rules, PlayStyle(unpredictability=100), f"e{i}") for i in range(60)}
        self.assertLessEqual(moves, set(onecard.legal_moves(state, 0)))
        self.assertGreater(len(moves), 1)  # unpredictability 100 deviates from the steady pick on some decisions


class SchemaTests(unittest.TestCase):
    def test_schema_is_the_move_menu_and_a_line(self):
        legal = ("play:2♥", "play:7♥=♠", "draw")
        self.assertEqual(game_chooser.onecard_schema(legal), {"type": "object", "properties": {"move": {"type": "string", "enum": list(legal)}, "line": {"type": "string"}},
                                                              "required": ["move", "line"], "additionalProperties": False})


class PromptTests(unittest.TestCase):
    NAMES = ["Player 1", "Ann", "Player 2"]

    def setUp(self):
        self.state = rig([["2♥", "7♠", "9♣", "JKb"], ["K♠", "A♦", "3♣", "10♥", "JKc"], ["4♠"]], "5♥", turn=1)
        self.view = view_of(self.state, 1)
        self.legal = onecard.legal_moves_for_view(self.view, self.state.rules)

    def text(self, state=None, seat=1):
        state = state or self.state
        view = view_of(state, seat)
        return game_chooser.onecard_table_text(view, self.NAMES, onecard.legal_moves_for_view(view, state.rules))

    def test_the_prompt_holds_own_hand_top_card_suit_and_public_counts(self):
        text = self.text()
        self.assertIn("Your hand: " + ", ".join(onecard.card_text(card) for card in self.view.hand), text)
        self.assertIn("Top card: 5♥", text)
        self.assertIn("Suit: ♥", text)
        self.assertRegex(text, r"Player 1: 4 cards")
        self.assertRegex(text, r"Player 2: 1 card\b")
        self.assertRegex(text, rf"Draw pile: {self.view.draw} cards")
        self.assertNotRegex(text, r"Ann: \d+ card")  # the viewer is not listed among the others

    def test_the_legal_move_menu_is_listed_exactly(self):
        self.assertIn("Legal moves: " + ", ".join(self.legal), self.text())

    def test_a_pending_attack_is_shown_only_while_one_is_pending(self):
        self.assertNotIn("Pending attack", self.text())
        state = rig([["2♥", "9♣"], ["2♣", "5♠", "9♦"]], "2♦", pending=4, turn=1)
        self.assertIn("Pending attack: 4", self.text(state))

    def test_recent_public_plays_and_draws_are_told_by_name(self):
        state = onecard.apply(rig([["9♥", "4♣", "J♣"], ["K♠", "A♦", "3♣"], ["4♠", "6♦"]], "5♥"), 0, "play:9♥")
        state = onecard.apply(state, 1, "draw")
        text = self.text(state, seat=2)
        self.assertIn("Player 1 played 9♥", text)
        self.assertIn("Ann drew", text)

    def test_no_other_seat_hand_or_draw_pile_card_is_in_the_prompt(self):
        text = self.text()
        mine = set(self.view.hand)
        hidden = {card for hand in self.state.hands for card in hand} | set(self.state.draw)
        for card in hidden - mine - {self.state.discard[-1]}:
            self.assertNotIn(onecard.card_text(card), text)
        self.assertNotIn("<@", text)
        self.assertNotIn("seed", text.lower())

    def test_an_eliminated_seat_is_marked_out(self):
        state = onecard.apply(rig([["9♥", "4♣"], ["K♠", "A♦"], ["4♠", "6♦"]], "5♥"), 0, "forfeit")
        self.assertRegex(self.text(state, seat=1), r"Player 1: 0 cards? \(out of the game\)|Player 1.*out")


class AskTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.state = rig([["2♥", "7♠", "9♣"], ["K♠", "A♦", "3♣"], ["4♠"]], "5♥", turn=1)
        self.view = view_of(self.state, 1)
        self.legal = onecard.legal_moves_for_view(self.view, self.state.rules)
        self.names = ["Player 1", "Ann", "Player 2"]
        self.character = {"id": 7, "name": "Ann", "card": CARD}

    async def test_one_structured_call_on_the_seat_view(self):
        bot, calls = fake_bot({"move": "draw", "line": ""})
        result = await game_chooser.ask_onecard(bot, G, CH, self.character, self.view, self.names, self.legal)
        self.assertEqual(result, {"move": "draw", "line": ""})
        (role, request, name, schema, feature, purpose), = calls
        text = "\n".join(m.text for m in request.messages)
        self.assertEqual((role, name, feature, purpose), ("director", "onecard_move", "game", "game"))
        self.assertEqual(schema, game_chooser.onecard_schema(self.legal))
        self.assertIn("You are Ann", request.messages[0].text)
        self.assertIn("cheerful gambler named Ann", text)
        self.assertNotIn("Play style", text)
        self.assertEqual(request.messages[-1].text, game_chooser.onecard_table_text(self.view, self.names, self.legal))

    async def test_a_saved_style_is_told_in_words(self):
        bot, calls = fake_bot({"move": "draw", "line": ""})
        style = PlayStyle(aggression=95, risk=90)
        await game_chooser.ask_onecard(bot, G, CH, self.character, self.view, self.names, self.legal, style)
        self.assertIn(f"Play style: {describe(style)}.", "\n".join(m.text for m in calls[0][1].messages))

    async def test_a_failing_model_raises_for_the_caller_to_fall_back(self):
        bot, _ = fake_bot(None)

        async def boom(*args):
            raise RuntimeError("down")
        bot.models.structured_compiled = boom
        with self.assertRaises(RuntimeError):
            await game_chooser.ask_onecard(bot, G, CH, self.character, self.view, self.names, self.legal)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
