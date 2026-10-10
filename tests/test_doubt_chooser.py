"""FEAT-21 part B2: model-chosen character moves for I Doubt It, with doubt triage.

TDD-red tests written before the implementation. Offline: the structured model call is a fake
(``bot.models.structured_compiled``), table fixtures come from ``test_doubt_discord`` (a table with character seats).
Timers are driven by calling their hooks, never by waiting.

Assumed API (a proposal, pinned here):

  ``llmcord_core/game_chooser.py`` (next to blackjack's ``ask``; same ROLE, ``persona()``, usage wrappers):
    ``doubt_play_schema(view) -> dict``      strict object; properties = the ranks the seat holds, each
                                             ``{"type": "integer", "enum": [0..min(held, 4)]}``, all required, no extras.
    ``doubt_doubt_schema() -> dict``         ``{"type": "object", "properties": {"doubt": {"type": "boolean"}},
                                             "required": ["doubt"], "additionalProperties": False}``.
    ``doubt_table_text(view, names, purpose) -> str``   purpose ``'play'`` or ``'doubt'``; ``view`` is a seat's
                                             ``doubt.View``; ``names`` is a list of display names indexed by seat
                                             (characters by name, members "Player 1", "Player 2", ... in seat order).
                                             Lines the tests rely on:
                                               ``Your hand: A x2, 3 x1, ...``      (ranks in doubt.RANKS order, counts only)
                                               ``Rank to claim: <rank>``
                                               ``<name>: <n> card(s)``              for every other seat
                                               ``Pile: <n> card(s)``
                                               ``<name> claimed <count> <rank>``    for each of the last few plays
                                               ``<card>`` as rank+suit (``7♠``) for cards revealed by a doubt
                                               doubt purpose also: ``Claim to judge: <name> claims <count> <rank>``
    ``async ask_doubt_play(bot, guild_id, channel_id, character, view, names) -> dict``
    ``async ask_doubt_doubt(bot, guild_id, channel_id, character, view, names) -> bool``  (non-bool answer raises)
    Both: system message "You are <name>, ..." plus the persona, user message = ``doubt_table_text``; role ``director``;
    usage feature/purpose ``game`` on a pinned backend snapshot.

  ``DoubtTables.MODEL_SECONDS`` (class constant, 20): bound for a play decision; doubt decisions are bounded by
  ``min(MODEL_SECONDS, WINDOW_SECONDS - 3)``. The timer hooks keep their names and signatures. Gating: model used only
  when ``game_character_talk`` is on and ``game_talk_calls_today + failed attempts < game_talk_daily_limit``; the spending
  hard cap surfaces as ``BudgetExceeded`` from the gateway and is just another fallback. Every attempt (success or
  failure) counts toward the limit.
"""
import asyncio
import functools
import json
import time
from types import SimpleNamespace

import test_doubt_discord as base
from test_doubt_discord import ALICE, BOB, G, counts_of, parse_move

from llmcord_core import game_chooser
from llmcord_core.budget import BudgetExceeded
from llmcord_core.games import doubt
from llmcord_core.games.base import Seat
from llmcord_core.usage import log_attribution

CARD = json.dumps({"description": "A cheerful gambler named {{char}}.", "personality": "Superstitious and chatty."})


def deal(seed, n):
    return doubt.new(seed, [Seat(k, "member", k + 1, 0) for k in range(n)])


def find_seed(n, ok, prefix):
    for i in range(20000):
        seed = f"{prefix}-{i}"
        if ok(deal(seed, n).hands):
            return seed
    raise AssertionError("no suitable seed")


def aces(hand):
    return sum(c[0] == "A" for c in hand)


@functools.lru_cache(maxsize=None)
def quiet_seed():
    """3 seats: seat 1 (Ann) holds no ace, so a claim of four aces is not impossible for her but is suspicious."""
    return find_seed(3, lambda h: aces(h[1]) == 0 and len(h[0]) >= 4, "doubt-b2-quiet")


@functools.lru_cache(maxsize=None)
def five_seed():
    """5 seats (characters at 1, 3, 4): none of them holds an ace, so nobody sees an impossible claim of four aces."""
    return find_seed(5, lambda h: aces(h[1]) == aces(h[3]) == aces(h[4]) == 0 and len(h[0]) >= 4, "doubt-b2-five")


@functools.lru_cache(maxsize=None)
def deep_seed():
    """6 seats (two decks): some seat holds five or more of one rank."""
    return find_seed(6, lambda h: any(max(counts_of(hand).values()) >= 5 for hand in h), "doubt-b2-deep")


def fake_bot(ask_result):
    """The smallest bot the ask functions need: a backend with pin(), a model that records its call."""
    calls = []

    async def model(role, request, name, schema):
        calls.append((role, request, name, schema, log_attribution()["feature"], log_attribution()["purpose"]))
        return ask_result

    class Pin:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return SimpleNamespace(backend=SimpleNamespace(pin=lambda: Pin()), models=SimpleNamespace(structured_compiled=model)), calls


class BuilderCase(base.DoubtCase):
    """Schema, prompt and ask functions on seat views of an engine deal; no table needed."""

    def view_of(self, seed, n, seat):
        return doubt.view(deal(seed, n), seat)

    def names(self, n):
        return [f"Player {i + 1}" for i in range(n)]


class SchemaTests(BuilderCase):
    def test_play_schema_has_only_the_held_ranks_all_required(self):
        view = self.view_of(self.seed, 3, 1)
        schema = game_chooser.doubt_play_schema(view)
        held = counts_of(view.hand)
        self.assertEqual(schema["type"], "object")
        self.assertEqual(set(schema["properties"]), set(held))
        self.assertEqual(set(schema["required"]), set(held))
        self.assertIs(schema["additionalProperties"], False)

    def test_play_schema_counts_are_integer_enums_capped_at_held_and_four(self):
        view = self.view_of(deep_seed(), 6, next(i for i, hand in enumerate(deal(deep_seed(), 6).hands) if max(counts_of(hand).values()) >= 5))
        schema = game_chooser.doubt_play_schema(view)
        held = counts_of(view.hand)
        self.assertGreaterEqual(max(held.values()), 5)
        for rank, n in held.items():
            self.assertEqual(schema["properties"][rank], {"type": "integer", "enum": list(range(0, min(n, 4) + 1))})

    def test_doubt_schema_is_one_boolean(self):
        self.assertEqual(game_chooser.doubt_doubt_schema(), {"type": "object", "properties": {"doubt": {"type": "boolean"}},
                                                             "required": ["doubt"], "additionalProperties": False})


class PromptTests(BuilderCase):
    def hands(self, n=3):
        return [list(h) for h in deal(self.seed, n).hands]

    def test_play_prompt_holds_own_hand_forced_rank_and_public_counts(self):
        view = self.view_of(self.seed, 3, 1)
        text = game_chooser.doubt_table_text(view, ["Player 1", "Ann", "Player 2"], "play")
        held = counts_of(view.hand)
        self.assertIn("Your hand: " + ", ".join(f"{rank} x{n}" for rank, n in held.items()), text)
        self.assertIn(f"Rank to claim: {view.rank}", text)
        self.assertIn(f"Player 1: {view.counts[0]} card", text)
        self.assertIn(f"Player 2: {view.counts[2]} card", text)
        self.assertIn(f"Pile: {view.pile} card", text)

    def test_prompts_never_show_another_seats_cards_or_member_identity(self):
        hands = self.hands()
        for purpose in ("play", "doubt"):
            view = self.view_of(self.seed, 3, 1)
            text = game_chooser.doubt_table_text(view, ["Player 1", "Ann", "Player 2"], purpose)
            for seat in (0, 2):
                for rank, suit in hands[seat]:
                    self.assertNotIn(f"{rank}{suit}", text)
            self.assertNotIn("<@", text)

    def test_doubt_prompt_names_the_claim_being_judged(self):
        state = deal(self.seed, 3)
        state = doubt.apply(state, 0, doubt.timeout_move(state, 0, "s"))
        view = doubt.view(state, 1)
        w = view.window
        text = game_chooser.doubt_table_text(view, ["Player 1", "Ann", "Player 2"], "doubt")
        self.assertIn(f"Claim to judge: Player 1 claims {w.count} {w.rank}", text)
        self.assertIn(f"Player 1 claimed {w.count} {w.rank}", text)

    def test_cards_revealed_by_a_doubt_appear_in_the_prompt(self):
        state = deal(self.seed, 3)
        state = doubt.apply(state, 0, doubt.timeout_move(state, 0, "s"))
        state = doubt.apply(state, 1, "doubt")
        reveal = state.history[-1]
        text = game_chooser.doubt_table_text(doubt.view(state, 2), ["Player 1", "Ann", "Player 2"], "play")
        for rank, suit in reveal.cards:
            self.assertIn(f"{rank}{suit}", text)


class AskTests(BuilderCase):
    async def test_ask_play_is_one_structured_call_on_the_seat_view(self):
        bot, calls = fake_bot({"A": 1})
        view = self.view_of(self.seed, 3, 1)
        self.store.db.execute("UPDATE characters SET card=? WHERE id=?", (CARD, self.ann))
        self.store.db.commit()
        character = self.store.character_by_id(self.ann)
        result = await game_chooser.ask_doubt_play(bot, G, base.CH, character, view, self.names_for_ann())
        self.assertEqual(result, {"A": 1})
        (role, request, name, schema, feature, purpose), = calls
        text = "\n".join(m.text for m in request.messages)
        self.assertEqual((role, feature, purpose), ("director", "game", "game"))
        self.assertEqual(schema, game_chooser.doubt_play_schema(view))
        self.assertIn("You are Ann", request.messages[0].text)
        self.assertIn("cheerful gambler named Ann", text)
        self.assertEqual(request.messages[-1].text, game_chooser.doubt_table_text(view, self.names_for_ann(), "play"))

    def names_for_ann(self):
        return ["Player 1", "Ann", "Player 2"]

    async def test_ask_doubt_returns_the_models_boolean(self):
        view = self.view_of(self.seed, 3, 1)
        character = self.store.character_by_id(self.ann)
        for answer in (True, False):
            bot, calls = fake_bot({"doubt": answer})
            self.assertIs(await game_chooser.ask_doubt_doubt(bot, G, base.CH, character, view, self.names_for_ann()), answer)
            self.assertEqual(calls[0][3], game_chooser.doubt_doubt_schema())

    async def test_ask_doubt_refuses_a_non_boolean_answer(self):
        view = self.view_of(self.seed, 3, 1)
        character = self.store.character_by_id(self.ann)
        for answer in ({"doubt": "yes"}, {}, None, "doubt"):
            with self.subTest(answer=answer):
                bot, _ = fake_bot(answer)
                with self.assertRaises(Exception):
                    await game_chooser.ask_doubt_doubt(bot, G, base.CH, character, view, self.names_for_ann())


class ModelCase(base.DoubtCase):
    """A table with character seats and a fake model. ``answers[kind]`` is a value, an exception, or ``f(name, schema)``."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.store.set_game_character_talk(G, True)
        self.calls, self.lock_refs, self.answers = [], [], {"play": {}, "doubt": {"doubt": False}}
        self.in_flight = self.max_in_flight = 0
        self.record_usage = False
        original = self.doubt.runtime.lock

        def spy(key):
            lock = original(key)
            self.lock_refs.append(lock)  # a strong reference, so the weak lock registry cannot hide a held lock
            return lock
        self.doubt.runtime.lock = spy
        self.bot.models.structured_compiled = self.model

    async def model(self, role, request, schema_name, schema):
        text = "\n".join(m.text for m in request.messages)
        name = request.messages[0].text.split("You are ", 1)[1].split(",")[0].split("\n")[0]
        kind = "doubt" if "doubt" in schema["properties"] else "play"
        self.calls.append({"kind": kind, "name": name, "text": text, "schema": schema, "role": role,
                           "locked": any(lock.locked() for lock in self.lock_refs)})
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0)
            answer = self.answers[kind]
            if callable(answer):
                answer = answer(name, schema)
                if asyncio.iscoroutine(answer):
                    answer = await answer
            if isinstance(answer, BaseException):
                raise answer
            if self.record_usage:
                self.store.db.execute("INSERT INTO model_usage(guild_id,profile,model,role,cached_tokens,reasoning_tokens,cost_basis,created_at,feature) VALUES(?,?,?,?,0,0,?,?,?)",
                                      (G, "p", "m", "director", "none", time.time(), "game"))
                self.store.db.commit()
            return answer
        finally:
            self.in_flight -= 1

    def add_usage(self, count):
        for _ in range(count):
            self.store.db.execute("INSERT INTO model_usage(guild_id,profile,model,role,cached_tokens,reasoning_tokens,cost_basis,created_at,feature) VALUES(?,?,?,?,0,0,?,?,?)",
                                  (G, "p", "m", "director", "none", time.time(), "game"))
        self.store.db.commit()

    # --- tables ------------------------------------------------------------------------------

    async def table3(self, seed=None):
        """Alice 0, Ann 1 (Alice's favorite), Bob 2."""
        if seed:
            self.seed = seed
        await self.start(favorites=True)
        await self.push(BOB, "Join", self.lobby())
        await self.push(ALICE, "Start", self.lobby())

    async def table5(self):
        """Alice 0, Ann 1, Bob 2, Bo 3, Cy 4 (the last two are house seats)."""
        self.seed = five_seed()
        await self.start(favorites=True)
        await self.push(BOB, "Join", self.lobby())
        await self.push(ALICE, "Fill seats", self.lobby())
        await self.push(ALICE, "Fill seats", self.lobby())
        await self.push(ALICE, "Start", self.lobby())
        self.assertEqual([s["name"] for s in self.snap()["seats"] if s["kind"] == "character"], ["Ann", "Bo", "Cy"])

    async def ann_on_turn(self, seed=None):
        await self.table3(seed)
        await self.pick_play(ALICE, {"A": 1})
        await self.close_window()
        self.assertEqual(self.snap()["view"]["turn"], 1)

    async def close_window(self):
        snap = self.snap()
        await self.doubt.on_window_timer(G, snap["table_id"], snap["round_id"], snap["moves"])

    async def fire(self):
        snap = self.snap()
        await self.doubt.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"])

    def held(self, seat):
        return counts_of(self.hand(seat))

    def answer_for(self, seat, **picks):
        """A schema-shaped play answer for the seat: every held rank, the picked counts, zero elsewhere."""
        return {rank: picks.get(rank, 0) for rank in self.held(seat)}

    def other_rank(self, seat, avoid):
        return next(rank for rank in self.held(seat) if rank != avoid)

    def policy_play_move(self, seat):
        snap = self.snap()
        seen = self.store.seat_view(G, snap["round_id"], seat)
        return doubt.policy_play(seen, doubt.tendencies(snap["seats"][seat]["ref_id"]), self.store.policy_seed(G, snap["round_id"], f"{snap['moves']}/{seat}/play"))

    def policy_would_doubt(self, seat):
        snap = self.snap()
        seen = self.store.seat_view(G, snap["round_id"], seat)
        return doubt.policy_doubt(seen, doubt.tendencies(snap["seats"][seat]["ref_id"]), self.store.policy_seed(G, snap["round_id"], f"{snap['moves']}/{seat}/doubt"))

    def calls_of(self, kind):
        return [c for c in self.calls if c["kind"] == kind]


class PlayDecisionTests(ModelCase):
    async def test_a_valid_model_play_is_applied_for_the_character(self):
        await self.ann_on_turn()
        bluff = self.other_rank(1, "2")
        self.answers["play"] = self.answer_for(1, **{bluff: 1})
        await self.fire()
        seat, move, actor = self.moves()[-1]
        self.assertEqual((seat, actor, parse_move(move)), (1, "character", {bluff: 1}))
        (call,) = self.calls
        self.assertEqual((call["kind"], call["name"], call["role"], call["locked"]), ("play", "Ann", "director", False))

    async def test_a_valid_multi_rank_play_is_applied_in_canonical_form(self):
        await self.ann_on_turn()
        bluff = self.other_rank(1, "2")
        self.answers["play"] = self.answer_for(1, **{"2": 1, bluff: 1})
        await self.fire()
        seat, move, actor = self.moves()[-1]
        self.assertEqual((seat, actor, parse_move(move)), (1, "character", {"2": 1, bluff: 1}))

    async def expect_policy_play(self):
        expected = self.policy_play_move(1)
        before = len(self.moves())
        await self.fire()
        self.assertEqual(len(self.moves()), before + 1)
        seat, move, actor = self.moves()[-1]
        self.assertEqual((seat, actor, parse_move(move)), (1, "character", parse_move(expected)))

    async def test_a_play_of_no_cards_falls_back(self):
        await self.ann_on_turn()
        self.answers["play"] = self.answer_for(1)
        await self.expect_policy_play()
        self.assertEqual(len(self.calls), 1)

    async def test_a_play_of_more_than_four_cards_falls_back(self):
        await self.ann_on_turn()
        held = self.held(1)
        ranks = sorted(held, key=lambda r: -held[r])
        picks = {}
        for rank in ranks:  # at most the held count per rank, five or more in all
            picks[rank] = min(held[rank], 4)
            if sum(picks.values()) >= 5:
                break
        self.assertGreaterEqual(sum(picks.values()), 5)
        self.answers["play"] = self.answer_for(1, **picks)
        await self.expect_policy_play()

    async def test_a_play_of_more_cards_than_held_falls_back(self):
        await self.ann_on_turn()
        rank = next(iter(self.held(1)))
        answer = self.answer_for(1)
        answer[rank] = self.held(1)[rank] + 1
        self.answers["play"] = answer
        await self.expect_policy_play()

    async def test_a_play_of_an_unheld_rank_falls_back(self):
        await self.ann_on_turn()
        missing = [rank for rank in doubt.RANKS if rank not in self.held(1)]
        self.assertTrue(missing)
        self.answers["play"] = {**self.answer_for(1, **{"2": 1}), missing[0]: 1}
        await self.expect_policy_play()

    async def test_a_non_object_answer_falls_back(self):
        await self.ann_on_turn()
        self.answers["play"] = ["2"]
        await self.expect_policy_play()

    async def test_a_model_error_falls_back(self):
        await self.ann_on_turn()
        self.answers["play"] = RuntimeError("provider down")
        await self.expect_policy_play()

    async def test_a_spending_hard_cap_falls_back(self):
        await self.ann_on_turn()
        self.answers["play"] = BudgetExceeded(SimpleNamespace())
        await self.expect_policy_play()

    async def test_a_slow_model_times_out_and_falls_back(self):
        await self.ann_on_turn()
        self.doubt.MODEL_SECONDS = 0.05

        async def slow(name, schema):
            await asyncio.sleep(5)
        self.answers["play"] = slow
        await self.expect_policy_play()

    async def test_the_prompt_holds_only_ann_view(self):
        await self.ann_on_turn()
        held, forced = self.held(1), self.snap()["view"]["rank"]  # read before the character plays
        await self.fire()
        text = self.calls[0]["text"]
        self.assertIn("Your hand: " + ", ".join(f"{rank} x{n}" for rank, n in held.items()), text)
        self.assertIn(f"Rank to claim: {forced}", text)
        self.assertIn("Player 1: ", text)  # Alice, the first member
        self.assertIn("Player 2: ", text)  # Bob
        for seat in (0, 2):
            for rank, suit in self.hand(seat):
                self.assertNotIn(f"{rank}{suit}", text)
        self.assertNotIn("<@", text)
        for uid in (ALICE, BOB):  # bare ids collide with card ranks, so look for every id form instead
            for needle in (f"<@{uid}>", f"<@!{uid}>", f"@{uid}", f"id {uid}", f"id={uid}", f"user {uid}", f"user_id {uid}"):
                self.assertNotIn(needle, text)
        for needle in ("Alice", "Bob"):
            self.assertNotIn(needle, text)

    async def test_a_late_second_timer_for_the_same_turn_is_dropped(self):
        await self.ann_on_turn()
        bluff = self.other_rank(1, "2")
        self.answers["play"] = self.answer_for(1, **{bluff: 1})
        before = len(self.moves())
        snap = self.snap()
        await asyncio.gather(*(self.doubt.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"]) for _ in range(2)))
        self.assertEqual(len(self.moves()), before + 1)  # one decision applied; the other result was stale

    async def test_no_model_call_runs_under_the_table_lock(self):
        await self.ann_on_turn()
        self.answers["play"] = self.answer_for(1, **{"2": 1})
        await self.fire()
        self.assertTrue(self.calls)
        self.assertEqual([c["locked"] for c in self.calls], [False] * len(self.calls))
        self.assertTrue(self.lock_refs)  # the spy saw the lock being used


class GatingTests(ModelCase):
    async def test_talk_off_plays_the_fallback_without_a_model_call(self):
        self.store.set_game_character_talk(G, False)
        await self.ann_on_turn()
        expected = self.policy_play_move(1)
        await self.fire()
        self.assertEqual(self.calls, [])
        self.assertEqual((self.moves()[-1][0], self.moves()[-1][2], parse_move(self.moves()[-1][1])), (1, "character", parse_move(expected)))

    async def test_talk_off_asks_no_doubt_question_either(self):
        self.store.set_game_character_talk(G, False)
        await self.table3(quiet_seed())
        await self.pick_play(ALICE, counts_of(self.hand(0)[:4]))
        await self.fire()
        self.assertEqual(self.calls, [])

    async def test_at_the_daily_limit_no_model_call_is_made(self):
        self.store.set_game_talk_daily_limit(G, 2)
        self.add_usage(2)
        await self.ann_on_turn()
        expected = self.policy_play_move(1)
        self.answers["play"] = self.answer_for(1, **{self.other_rank(1, "2"): 1})
        await self.fire()
        self.assertEqual(self.calls, [])
        self.assertEqual(parse_move(self.moves()[-1][1]), parse_move(expected))

    async def test_below_the_daily_limit_the_model_is_asked(self):
        self.store.set_game_talk_daily_limit(G, 2)
        self.add_usage(1)
        await self.ann_on_turn()
        await self.fire()
        self.assertEqual(len(self.calls), 1)

    async def to_ann_again(self):
        """Ann played and the window closed: Bob plays, Alice plays, Ann is on turn again."""
        await self.fire()
        await self.close_window()
        await self.pick_play(BOB, self.any_card(2))
        await self.close_window()
        await self.pick_play(ALICE, self.any_card(0))
        await self.close_window()
        self.assertEqual(self.snap()["view"]["turn"], 1)

    async def test_a_failed_call_counts_toward_the_daily_limit(self):
        self.store.set_game_talk_daily_limit(G, 1)
        await self.ann_on_turn()
        self.answers["play"] = RuntimeError("down")
        await self.to_ann_again()
        self.assertEqual(len(self.calls), 1)
        await self.fire()
        self.assertEqual(len(self.calls), 1)  # the limit of one was used up by the failed attempt

    async def test_a_successful_call_counts_toward_the_daily_limit(self):
        self.store.set_game_talk_daily_limit(G, 1)
        self.record_usage = True  # the real call records usage as feature 'game'
        await self.ann_on_turn()
        self.answers["play"] = self.answer_for(1, **{"2": 1})
        await self.to_ann_again()
        self.assertEqual(len(self.calls), 1)
        await self.fire()
        self.assertEqual(len(self.calls), 1)

    async def test_a_timed_out_call_counts_toward_the_daily_limit(self):
        self.store.set_game_talk_daily_limit(G, 1)
        self.doubt.MODEL_SECONDS = 0.05

        async def slow(name, schema):
            await asyncio.sleep(5)
        self.answers["play"] = slow
        await self.ann_on_turn()
        await self.to_ann_again()
        await self.fire()
        self.assertEqual(len(self.calls), 1)


class DoubtDecisionTests(ModelCase):
    async def claim_four_aces(self):
        await self.table3(quiet_seed())
        await self.pick_play(ALICE, counts_of(self.hand(0)[:4]))
        self.assertIn(((G, self.snap()["table_id"]), "character"), self.doubt.runtime.timers)

    async def test_the_model_saying_doubt_makes_the_character_doubt(self):
        await self.claim_four_aces()
        self.answers["doubt"] = {"doubt": True}
        before = len(self.moves())
        await self.fire()
        (call,) = self.calls
        self.assertEqual((call["kind"], call["name"], call["locked"]), ("doubt", "Ann", False))
        self.assertEqual(self.moves()[before:], [(1, "doubt", "character")])

    async def test_the_model_saying_no_doubt_leaves_the_window_open(self):
        await self.claim_four_aces()
        before = len(self.moves())
        await self.fire()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.moves()), before)
        self.assertIsNotNone(self.snap()["view"]["window"])

    async def expect_policy_doubt(self):
        await self.claim_four_aces()
        expected = self.policy_would_doubt(1)
        before = len(self.moves())
        await self.fire()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.moves()[before:], [(1, "doubt", "character")] if expected else [])

    async def test_a_model_error_falls_back_to_the_policy_doubt(self):
        self.answers["doubt"] = RuntimeError("down")
        await self.expect_policy_doubt()

    async def test_a_spending_hard_cap_falls_back_to_the_policy_doubt(self):
        self.answers["doubt"] = BudgetExceeded(SimpleNamespace())
        await self.expect_policy_doubt()

    async def test_a_malformed_answer_falls_back_to_the_policy_doubt(self):
        self.answers["doubt"] = {"doubt": "yes"}
        await self.expect_policy_doubt()

    async def test_a_slow_doubt_call_times_out_to_the_policy(self):
        self.doubt.WINDOW_SECONDS = 3.05
        self.doubt.MODEL_SECONDS = 0.05

        async def slow(name, schema):
            await asyncio.sleep(5)
        self.answers["doubt"] = slow
        await self.claim_four_aces()
        expected = self.policy_would_doubt(1)
        before = len(self.moves())
        await self.fire()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.moves()[before:], [(1, "doubt", "character")] if expected else [])

    async def test_an_impossible_claim_is_doubted_without_a_model_call(self):
        await self.table3()  # Ann holds an ace
        await self.pick_play(ALICE, counts_of(self.hand(0)[:4]))
        await self.fire()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.moves()[-1], (1, "doubt", "character"))

    async def test_the_doubt_prompt_is_ann_view_only(self):
        await self.claim_four_aces()
        await self.fire()
        text = self.calls[0]["text"]
        self.assertIn("Claim to judge: Player 1 claims 4 A", text)
        for seat in (0, 2):
            for rank, suit in self.hand(seat):
                self.assertNotIn(f"{rank}{suit}", text)
        for uid in (ALICE, BOB):
            for needle in (f"<@{uid}>", f"<@!{uid}>", f"@{uid}", f"id {uid}", f"id={uid}", f"user {uid}", f"user_id {uid}"):
                self.assertNotIn(needle, text)
        for needle in ("<@", "Alice", "Bob"):
            self.assertNotIn(needle, text)

    async def test_at_the_daily_limit_no_doubt_question_is_asked(self):
        self.store.set_game_talk_daily_limit(G, 1)
        self.add_usage(1)
        await self.claim_four_aces()
        await self.fire()
        self.assertEqual(self.calls, [])


class TriageTests(ModelCase):
    async def claim(self):
        await self.table5()
        await self.pick_play(ALICE, counts_of(self.hand(0)[:4]))
        snap = self.snap()
        scored = []
        for seat in (1, 3, 4):
            seen = self.store.seat_view(G, snap["round_id"], seat)
            scored.append((seat, doubt.suspicion(seen, doubt.tendencies(snap["seats"][seat]["ref_id"])), doubt.impossible(seen)))
        self.assertFalse(any(imp for _, _, imp in scored))
        self.assertTrue(all(s >= doubt.TRIAGE_MIN for _, s, _ in scored))
        self.selected = set(doubt.triage(scored, k=2))
        self.assertEqual(len(self.selected), 2)
        self.names = {s["index"]: s["name"] for s in snap["seats"]}

    async def test_only_the_two_most_suspicious_character_seats_are_asked(self):
        await self.claim()
        before = len(self.moves())
        await self.fire()
        self.assertEqual(sorted(c["name"] for c in self.calls), sorted(self.names[i] for i in self.selected))
        self.assertEqual(len({c["name"] for c in self.calls}), 2)
        self.assertEqual(len(self.moves()), before)  # all said no: unselected seats pass, nobody doubts

    async def test_the_asked_seats_are_decided_in_parallel(self):
        await self.claim()
        self.answers["doubt"] = lambda name, schema: asyncio.sleep(0.02, {"doubt": False})
        await self.fire()
        self.assertEqual(self.max_in_flight, 2)

    async def test_the_first_yes_is_applied_and_the_second_is_dropped(self):
        await self.claim()
        self.answers["doubt"] = lambda name, schema: asyncio.sleep(0.01 if name == self.names[min(self.selected)] else 0.03, {"doubt": True})
        before = len(self.moves())
        await self.fire()
        applied = self.moves()[before:]
        self.assertEqual(len(applied), 1)
        seat, move, actor = applied[0]
        self.assertEqual((move, actor), ("doubt", "character"))
        self.assertIn(seat, self.selected)
        self.assertEqual(sum("Doubt!" in text for text in self.thread_texts()), 1)

    async def test_a_dropped_second_call_still_counts_toward_the_daily_limit(self):
        await self.claim()
        self.record_usage = True
        self.answers["doubt"] = lambda name, schema: asyncio.sleep(0.01 if name == self.names[min(self.selected)] else 0.2, {"doubt": True})
        before = self.doubt._talk_budget(G)
        self.bot.run_table_event = lambda *args, **kwargs: asyncio.sleep(0)  # the reveal's event turn (FEAT-21 C) is not what is counted here
        await self.fire()
        self.assertEqual(self.doubt._talk_budget(G), before - 2)  # one usage row for the answer, one failure for the dropped call

    async def test_cancelling_the_timer_mid_decision_applies_nothing_and_counts_the_calls(self):
        await self.claim()
        started = asyncio.Event()

        async def blocked(name, schema):
            started.set()
            await asyncio.sleep(5)
        self.answers["doubt"] = blocked
        before, budget = len(self.moves()), self.doubt._talk_budget(G)
        task = asyncio.create_task(self.fire())
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(self.moves()), before)
        self.assertEqual(self.in_flight, 0)
        self.assertEqual(self.doubt._talk_budget(G), budget - 2)

    async def test_a_yes_after_a_no_still_doubts(self):
        await self.claim()
        yes = self.names[max(self.selected)]
        self.answers["doubt"] = lambda name, schema: {"doubt": name == yes}
        before = len(self.moves())
        await self.fire()
        self.assertEqual(self.moves()[before:], [(max(self.selected), "doubt", "character")])

    async def test_a_member_doubt_during_the_model_calls_is_not_delayed_and_wins(self):
        await self.claim()
        started, release = asyncio.Event(), asyncio.Event()

        async def blocked(name, schema):
            started.set()
            await release.wait()
            return {"doubt": True}
        self.answers["doubt"] = blocked
        before = len(self.moves())
        task = asyncio.create_task(self.fire())
        await asyncio.wait_for(started.wait(), 1)
        click = await self.doubt_press(BOB)  # seat 2: the table lock is free while the models think
        self.assertEqual(self.moves()[before:], [(2, "doubt", "member")])
        self.assertTrue(base.says(click) == [] or not any("moved on" in t.lower() for t in base.says(click)))
        release.set()
        await task
        self.assertEqual(self.moves()[before:], [(2, "doubt", "member")])  # the late character yes was stale

    async def test_no_model_call_runs_under_the_table_lock(self):
        await self.claim()
        await self.fire()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual([c["locked"] for c in self.calls], [False, False])

    async def test_each_asked_seat_prompt_holds_only_its_own_cards(self):
        await self.claim()
        await self.fire()
        for call in self.calls:
            seat = next(i for i, name in self.names.items() if name == call["name"])
            for other in range(5):
                if other == seat:
                    continue
                for rank, suit in self.hand(other):
                    self.assertNotIn(f"{rank}{suit}", call["text"])

    async def test_talk_off_keeps_the_per_seat_fallback_policy(self):
        self.store.set_game_character_talk(G, False)
        await self.claim()
        snap = self.snap()
        expected = None
        n = len(snap["seats"])
        for k in range(1, n):
            i = (snap["view"]["window"]["claimer"] + k) % n
            if snap["seats"][i]["kind"] == "character" and snap["view"]["active"][i] and self.policy_would_doubt(i):
                expected = (i, "doubt", "character")
                break
        before = len(self.moves())
        await self.fire()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.moves()[before:], [expected] if expected else [])


if __name__ == "__main__":  # pragma: no cover
    import unittest
    unittest.main()
