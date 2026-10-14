"""FEAT-44 step B: character seats at One Card tables (favorites, the Character-moves chooser, pacing, table talk, money).

TDD-red tests written before the implementation. Offline: fake Discord and in-memory store from ``test_onecard_discord``; the model is a
fake (``bot.models.structured_compiled``); deals are rigged by patching ``onecard.new`` (the store replays every state through it) so a
test names the hands and the top card; the pure engine is the oracle. Timers are driven by calling their hooks, never by waiting.
Mirrors ``test_doubt_chooser.py``, ``test_doubt_table_talk.py`` and the character parts of ``test_doubt_discord.py``; the pure policy and
prompt tests are in ``test_onecard_chooser.py`` (its header lists the pure API).

Assumed API (a proposal, pinned here; it mirrors ``DoubtTables`` names as closely as the game allows):

  Seating. ``/game start game:onecard favorites:true`` seats the starter's eligible favorites exactly as I Doubt It does
    (``store.join_with_favorites``: each antes from its own wallet, brought_by = the starter, skipped for broke / archived / full, the reply is
    ``GameTables._favorites_note``); in a channel without a world binding it says ``NO_CHARACTER_SEATS``. The step-A note
    (``CHARACTERS_LATER``) goes away. NOT assumed: a "Fill seats" lobby button (the brief does not ask for house seats; the lobby keeps
    ``Join Leave Start How to play``).
  Pacing. ``OneCardTables.CHARACTER_SECONDS = 2``, ``MODEL_SECONDS = 20``, ``EVENT_COOLDOWN_SECONDS = 20`` (measured on ``clock``).
    ``_after`` arms the timer kind ``"character"`` (``arm(key, "character", CHARACTER_SECONDS, round_id, moves)``) when the seat on turn is a
    character and the ``"turn"`` timer only when it is a member; a timer of the other kind does nothing. Hook:
    ``await on_character_timer(guild_id, table_id, round_id, moves)`` (stale-safe; no model call and no Discord call under the table lock).
  Chooser. In the hook, with ``store.game_character_moves(guild) == "style"`` or character table talk off or the daily limit reached
    (``_talk_budget(guild_id)``, ``talk_allowed(guild_id)``, the shared ``bot.talk_budget``) the move is
    ``onecard.policy_move(seen, snap["rules"], store.character_play_style(guild, character_id), store.policy_seed(guild, round_id, f"{moves}/{seat}/character"))``
    with actor ``character``. Otherwise one ``game_chooser.ask_onecard(bot, guild, channel, character, seen, names, legal, style)`` call outside
    the lock, bounded by ``MODEL_SECONDS``; its ``move`` is accepted only when it is one of the engine's legal moves for the seat (a 7 only
    with a suit), any other answer, error, ``BudgetExceeded`` or timeout counts one failed attempt (``talk.note_failure``) and the policy move is
    applied instead; the move is applied only if the table is still at the same move count. Last resorts when the store refuses a move: the
    engine ``timeout`` or ``draw``, then ``forfeit`` with actor ``timeout`` (as doubt's ``_apply_chain``).
  Talk. ``talk`` (``TalkBudget``), ``talk_failed``, ``talk_allowed(guild_id)``, ``table_talk_context(guild_id, thread_id)`` (a dict with
    ``table_id round_id status seated_character_ids``; joining, playing and the banter after the game until the archive timer),
    ``table_block(guild_id, thread_id, character_id)`` (public state, the character's own hand as ``Your hand: <cards>`` and a line
    ``Keep your hand secret``; ``None`` for anyone not seated), ``banter``, ``event_at``, ``event_tasks``, ``EVENT_COOLDOWN_SECONDS``.
    ``SkitBot._table_game`` and ``_table_block`` consult ``bot.onecard`` as well as doubt and holdem.
    Key moments schedule ``bot.run_table_event(thread, text)`` (fire and forget, one per move): an attack card played (2, A, joker), a skip or
    reverse (J, Q), a draw of an attack's pile of 5 or more cards, a seat down to one card (``"<name> is down to one card."``), and the win
    (``"Game over — <name> won the game."``); all gated by ``talk_allowed``; every moment except the win also by the cooldown. A plain
    play or a single-card draw is not a moment. Event texts hold no mentions, no markdown and no hidden card.
  Money as I Doubt It: a character antes from its wallet, the winner's wallet (member or character) gets the pot (ledger reason
    ``One Card payout (table N, round M)``), a lobby that closes refunds the characters' antes.
"""
import asyncio
import dataclasses
import itertools
import logging
import time
from types import SimpleNamespace
from unittest import mock

from helpers import CompiledAdapter, FakeWebhook, drain_memory_tasks, thread_message
from test_doubt_discord import blob, labels, says
from test_onecard_chooser import FULL_DECK, c
from test_table_view import KEYS, MESSY, CLEAN, lookup, rename
from test_onecard_discord import ALICE, ANTE, ANTE_REASON, BOB, CH, CH2, G, HTTP, PAYOUT_REASON, OneCardCase, pick_move

from llmcord_core.admin_store import GameError
from llmcord_core.budget import BudgetExceeded
from llmcord_core.games import onecard
from llmcord_core.games.base import shuffled
from llmcord_core.games.playstyle import PlayStyle, describe
from llmcord_core.games_discord import NO_CHARACTER_SEATS
from llmcord_core.games_onecard_discord import OneCardTables

REAL_NEW = onecard.new
BOT = SimpleNamespace(id=999, bot=True, mention="<@999>", display_name="Bot")
NAMES = {ALICE: "Alice", BOB: "Bob"}

# Alice 0, Ann 1, Bob 2; the top card is 5♥ and no card appears twice. Alice opens with the 9♥ (top 9♥, suit ♥).
ALICE_HAND = ["9♥", "3♣", "K♠", "8♦", "6♠", "10♦", "4♣"]
BOB_HAND = ["A♦", "5♦", "8♣", "Q♣", "10♣", "4♠", "6♥"]
ANN_PLAYS = ["J♥", "2♥", "7♠", "9♣", "K♦", "4♦", "3♦"]  # after the 9♥: playable J♥ 2♥ 9♣ (policy: the attack 2♥)
ANN_STUCK = ["K♦", "4♦", "3♦", "7♠", "8♠", "10♠", "Q♠"]  # nothing plays on a 9♥
ANN_SEVEN = ["7♥", "9♣", "K♦", "4♦", "3♦", "J♦", "Q♠"]  # after the 9♥ the 7♥ plays (suit named by the mover)


def tokens(cards):
    return [onecard.card_text(card) for card in cards]


def rigged_new(hands, top, pending=0, suit=None, direction=1):
    def new(seed, seats, rules=None):
        state = REAL_NEW(seed, seats, rules)
        if len(state.seats) != len(hands):
            return state
        cards = tuple(tuple(c(t) for t in hand) for hand in hands)
        left = list(FULL_DECK)
        for card in [*itertools.chain.from_iterable(cards), c(top)]:
            left.remove(card)
        return dataclasses.replace(state, hands=cards, draw=tuple(shuffled(left, seed)), discard=(c(top),), pending=pending, suit=suit, turn=0, direction=direction)
    return new


class CharCase(OneCardCase):
    """Alice 0, Ann 1 (Alice's favorite), Bob 2; Bo is in the cast, with a wallet, but not a favorite. Talk is off until a test turns it on."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.bo = self.store.create_character(G, self.world, "Bo")
        self.store.change_character_balance(G, self.bo, 100, "start", 1)
        self.store.set_cast(CH, None, [self.ann, self.bo])
        self.onecard.CHARACTER_SECONDS = 3600
        self.now = [10_000.0]
        self.onecard.clock = lambda: self.now[0]
        self.events = []
        self.hold = None

        async def spy(thread, text, speaker_hint=None, spectators=False):
            snap = self.store.latest_round(G, self.tid)
            state = self.engine_state(snap)
            hidden = set(tokens([card for hand in state.hands for card in hand] + list(state.draw)))
            self.events.append({"thread": thread, "text": text, "hidden": hidden, "finished": snap["status"] != "playing", "moves": snap["moves"]})
            if self.hold is not None:
                await self.hold.wait()
        self.bot.run_table_event = spy
        self.addAsyncCleanup(self.release)
        self.tid = None

    async def release(self):
        if self.hold is not None:
            self.hold.set()

    # --- tables ------------------------------------------------------------------------------

    def rig(self, hands, top, **kwargs):
        patcher = mock.patch.object(onecard, "new", rigged_new(hands, top, **kwargs))
        patcher.start()
        self.addCleanup(patcher.stop)

    async def scene(self, hands=None, top="5♥", *, favorites=True, **kwargs):
        """Alice 0, Ann 1, Bob 2 playing the (rigged) deal; ``hands`` is [alice, ann, bob] card texts."""
        if hands is not None:
            self.rig(hands, top, **kwargs)
        await self.start(favorites=favorites)
        await self.push(BOB, "Join", self.lobby())
        await self.push(ALICE, "Start", self.lobby())
        self.tid = self.table()["id"]

    def three(self, ann=ANN_PLAYS, alice=ALICE_HAND, bob=BOB_HAND):
        if ann is ANN_PLAYS:  # a card Alice is given instead (2♥, J♥) must not also be in the default Ann hand: no card appears twice
            spare = iter(["5♠", "9♦"])
            ann = [next(spare) if card in alice else card for card in ann]
        return [alice, ann, bob]

    async def ann_on_turn(self, ann=ANN_PLAYS, **kwargs):
        await self.scene(self.three(ann, **kwargs))
        await self.pick_play(ALICE, "play:9♥")
        self.assertEqual(self.snap()["view"]["turn"], 1)

    async def fire(self):
        snap = self.snap()
        await self.onecard.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"])

    async def settle(self):
        for _ in range(12):
            await asyncio.sleep(0)

    async def draw_until(self, seat):
        """Members draw (always legal) until the seat is on turn."""
        for _ in range(8):
            snap = self.snap()
            turn = snap["view"]["turn"]
            if turn == seat:
                return
            self.assertEqual(snap["seats"][turn]["kind"], "member")
            await self.pick_play(snap["seats"][turn]["ref_id"], "draw")
        self.fail("the seat never came on turn")

    def held(self, seat):
        return tokens(self.hand(seat))

    def policy_for(self, seat):
        snap = self.snap()
        seen = self.store.seat_view(G, snap["round_id"], seat)
        style = self.store.character_play_style(G, snap["seats"][seat]["ref_id"])
        return onecard.policy_move(seen, snap["rules"], style, self.store.policy_seed(G, snap["round_id"], f"{snap['moves']}/{seat}/character"))

    def legal_for(self, seat):
        return onecard.legal_moves(self.engine_state(), seat)

    def hidden_from(self, seat):
        """Every card token another seat or the draw pile holds and this seat does not (and that is not on the discard pile)."""
        state = self.engine_state()
        theirs = {t for i, hand in enumerate(state.hands) if i != seat for t in tokens(hand)} | set(tokens(state.draw))
        return theirs - set(tokens(state.hands[seat])) - set(tokens(state.discard))


# ----------------------------------------------------------------------------------------------------------------------------------
# Seating
# ----------------------------------------------------------------------------------------------------------------------------------
class SeatingTests(CharCase):
    async def test_favorites_seat_the_starters_favorites_with_their_ante(self):
        click = await self.start(favorites=True)
        self.assertEqual(self.seats(), [("member", ALICE, None), ("character", self.ann, ALICE)])
        self.assertEqual(self.store.character_balance(G, self.ann), 100 - ANTE)
        self.assertEqual(self.store.ledger(G, character_id=self.ann)[0]["reason"], ANTE_REASON)
        text = "\n".join(says(click))
        self.assertIn("Your favorites joined too", text)
        self.assertIn("Ann", text)
        self.assertNotIn("yet", text)  # the step-A "can't sit at One Card tables yet" note is gone
        self.assertNotIn("only players take seats", text)

    async def test_the_lobby_names_the_character_and_the_pot_counts_her_ante(self):
        await self.start(favorites=True)
        self.assertIn("Ann", self.lobby().content)
        self.assertNotIn(f"<@{self.ann}>", self.lobby().content)
        self.assertIn(f"{2 * ANTE:,}", self.lobby().content)  # the pot: Alice's and Ann's ante
        self.assertEqual(set(labels(self.lobby())), {"Join", "Leave", "Start", "How to play"})

    async def test_without_the_favorites_option_no_character_sits(self):
        await self.start()
        self.assertEqual(self.seats(), [("member", ALICE, None)])
        self.assertEqual(self.store.character_balance(G, self.ann), 100)

    async def test_a_channel_without_a_world_seats_no_character_and_says_so(self):
        click = await self.start(channel=self.channel2, favorites=True)
        self.assertEqual(self.seats(self.store.latest_round(G, self.store.open_table_for(G, CH2)["id"])), [("member", ALICE, None)])
        self.assertTrue(any(NO_CHARACTER_SEATS in text for text in says(click)))

    async def test_a_broke_favorite_is_skipped_with_a_reason_and_keeps_her_money(self):
        self.store.change_character_balance(G, self.ann, -95, "spent", 1)
        click = await self.start(favorites=True)
        self.assertEqual(self.seats(), [("member", ALICE, None)])
        self.assertEqual(self.store.character_balance(G, self.ann), 5)
        self.assertTrue(any("Not seated" in t and "Ann" in t for t in says(click)))

    async def test_a_second_favorite_takes_the_next_seat(self):
        self.store.add_favorite(G, ALICE, self.bo)
        await self.start(favorites=True)
        self.assertEqual({kind for kind, *_ in self.seats()}, {"member", "character"})
        self.assertEqual(sorted(ref for kind, ref, _ in self.seats() if kind == "character"), sorted([self.ann, self.bo]))

    async def test_the_starter_leaving_takes_the_characters_she_brought_and_refunds_them(self):
        await self.start(favorites=True)
        await self.push(BOB, "Join", self.lobby())
        await self.push(ALICE, "Leave", self.lobby())
        self.assertEqual(self.seats(), [("member", BOB, None)])
        self.assertEqual(self.store.character_balance(G, self.ann), 100)
        self.assertEqual(self.balance(ALICE), 100)

    async def test_an_idle_lobby_closes_and_refunds_the_characters_ante(self):
        await self.start(favorites=True)
        snap = self.snap()
        await self.onecard.on_lobby_timer(G, snap["table_id"], snap["round_id"])
        self.assertEqual(self.store.character_balance(G, self.ann), 100)
        self.assertEqual(self.balance(ALICE), 100)

    async def test_alice_and_ann_alone_can_start_a_two_seat_game(self):
        await self.start(favorites=True)
        await self.push(ALICE, "Start", self.lobby())
        self.assertEqual(self.snap()["status"], "playing")
        self.assertEqual(len(self.snap()["seats"]), 2)

    async def test_the_board_and_turn_message_name_the_character_not_a_mention(self):
        await self.scene(self.three())
        text = self.board().content + "\n" + self.turn().content
        self.assertIn("Ann", text)
        self.assertNotIn(f"<@{self.ann}>", text)
        self.assertIn(f"<@{ALICE}>", self.board().content)

    async def test_nobody_sees_the_characters_hand_at_the_start(self):
        await self.scene(self.three())
        public = self.everything_public()
        for card in ANN_PLAYS:
            self.assertNotIn(card, public)
        click = self.click(ALICE, channel=self.thread())
        await self.onecard.hand_command(click)
        shown = blob(click)
        for card in ANN_PLAYS:
            self.assertNotIn(card, shown)
        for card in ALICE_HAND:
            self.assertIn(card, shown)


# ----------------------------------------------------------------------------------------------------------------------------------
# A character's turn without a model (Play style policy)
# ----------------------------------------------------------------------------------------------------------------------------------
class TurnTests(CharCase):
    def test_the_constants_mirror_doubts(self):
        self.assertEqual((OneCardTables.CHARACTER_SECONDS, OneCardTables.MODEL_SECONDS, OneCardTables.EVENT_COOLDOWN_SECONDS), (2, 20, 20))

    async def test_a_member_turn_arms_the_turn_timer_and_a_character_turn_the_character_timer(self):
        await self.scene(self.three())
        key = (G, self.tid)
        self.assertIn((key, "turn"), self.onecard.runtime.timers)
        self.assertNotIn((key, "character"), self.onecard.runtime.timers)
        await self.pick_play(ALICE, "play:9♥")
        self.assertIn((key, "character"), self.onecard.runtime.timers)
        self.assertNotIn((key, "turn"), self.onecard.runtime.timers)
        await self.fire()  # Ann plays; Bob is next
        self.assertIn((key, "turn"), self.onecard.runtime.timers)
        self.assertNotIn((key, "character"), self.onecard.runtime.timers)

    async def test_the_character_timer_is_armed_with_the_character_delay(self):
        self.onecard.CHARACTER_SECONDS = 2
        armed = []
        real = self.onecard.arm
        self.onecard.arm = lambda key, kind, delay, *args: (armed.append((kind, delay)), real(key, kind, delay, *args))[1]
        await self.scene(self.three())
        await self.pick_play(ALICE, "play:9♥")
        self.assertIn(("character", 2), armed)

    async def test_two_characters_in_a_row_each_get_their_timer(self):
        self.store.add_favorite(G, ALICE, self.bo)
        hands = [ALICE_HAND, ANN_PLAYS, ["2♠", "5♠", "J♣", "Q♦", "A♣", "10♥", "6♦"], BOB_HAND]
        self.rig(hands, "5♥")
        await self.start(favorites=True)
        await self.push(BOB, "Join", self.lobby())
        await self.push(ALICE, "Start", self.lobby())
        kinds = [s["kind"] for s in self.snap()["seats"]]
        self.assertEqual(kinds.count("character"), 2)
        await self.pick_play(ALICE, "play:9♥")
        key = (G, self.table()["id"])
        for _ in range(2):
            self.assertEqual(self.snap()["seats"][self.snap()["view"]["turn"]]["kind"], "character")
            self.assertIn((key, "character"), self.onecard.runtime.timers)
            await self.fire()
        self.assertEqual(self.snap()["seats"][self.snap()["view"]["turn"]]["kind"], "member")

    async def test_the_policy_move_is_applied_for_the_character(self):
        await self.ann_on_turn()
        expected = self.policy_for(1)
        self.assertEqual(expected, "play:2♥")  # an attack when it can hit
        await self.fire()
        self.assertEqual(self.moves()[-1], (1, "play:2♥", "character"))
        self.assertEqual(self.snap()["view"]["turn"], 2)
        self.assertEqual(self.snap()["view"]["pending"], 2)
        await self.onecard.flush()
        self.assertIn("**Ann** played 2♥", self.turn().content)
        self.assertEqual(self.calls_made(), [])

    def calls_made(self):
        return list(self.bot.models.structured_compiled.await_args_list)

    async def test_a_character_with_nothing_to_play_draws_and_passes(self):
        await self.ann_on_turn(ann=ANN_STUCK)
        self.assertEqual(self.legal_for(1), ("draw",))
        await self.fire()
        self.assertEqual(self.moves()[-1], (1, "draw", "character"))
        self.assertEqual(len(self.hand(1)), 8)
        self.assertEqual(self.snap()["view"]["turn"], 2)
        self.assertIn("**Ann** drew a card.", self.turn().content)

    async def test_a_character_defends_an_attack_when_she_can(self):
        await self.scene(self.three(ann=["2♣", "9♣", "7♠", "K♦", "4♦", "3♦", "J♦"], alice=["2♥", *ALICE_HAND[1:]]))
        await self.pick_play(ALICE, "play:2♥")
        self.assertEqual(self.snap()["view"]["pending"], 2)
        await self.fire()
        self.assertEqual(self.moves()[-1], (1, "play:2♣", "character"))
        self.assertEqual(self.snap()["view"]["pending"], 4)

    async def test_a_character_without_a_defense_draws_the_pile(self):
        await self.scene(self.three(ann=["9♣", "7♠", "K♦", "4♦", "3♦", "J♦", "Q♠"], alice=["2♥", *ALICE_HAND[1:]]))
        await self.pick_play(ALICE, "play:2♥")
        await self.fire()
        self.assertEqual(self.moves()[-1], (1, "draw", "character"))
        self.assertEqual(len(self.hand(1)), 9)
        self.assertEqual(self.snap()["view"]["pending"], 0)
        self.assertIn("**Ann** drew 2 cards.", self.turn().content)

    async def test_the_saved_play_style_shapes_the_move_the_same_way_the_policy_computes_it(self):
        await self.ann_on_turn()
        cid = self.ann
        self.store.set_character_play_style(G, cid, PlayStyle(unpredictability=100), self.store.owner_revision(G, "character", cid))
        expected = self.policy_for(1)
        await self.fire()
        self.assertEqual(self.moves()[-1], (1, expected, "character"))

    async def test_a_stale_character_timer_does_nothing(self):
        await self.ann_on_turn()
        snap = self.snap()
        before = len(self.moves())
        await self.onecard.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"] - 1)
        self.assertEqual(len(self.moves()), before)

    async def test_a_late_second_timer_for_the_same_turn_is_dropped(self):
        await self.ann_on_turn()
        snap = self.snap()
        before = len(self.moves())
        await asyncio.gather(*(self.onecard.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"]) for _ in range(2)))
        self.assertEqual(len(self.moves()), before + 1)

    async def test_the_character_timer_does_nothing_on_a_members_turn(self):
        await self.scene(self.three())
        before = len(self.moves())
        await self.fire()
        self.assertEqual(len(self.moves()), before)

    async def test_the_turn_timer_leaves_a_character_alone(self):
        await self.ann_on_turn()
        snap = self.snap()
        before = len(self.moves())
        await self.onecard.on_turn_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        self.assertEqual(len(self.moves()), before)
        self.assertEqual(self.snap()["view"]["turn"], 1)

    async def test_a_character_timer_after_the_table_closed_does_nothing(self):
        await self.ann_on_turn()
        snap = self.snap()
        self.store.close_table(G, snap["table_id"])
        before = len(self.moves())
        await self.onecard.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        self.assertEqual(len(self.moves()), before)

    async def test_a_move_the_store_refuses_falls_back_to_a_draw(self):
        await self.ann_on_turn()
        with mock.patch.object(onecard, "policy_move", return_value="play:A♠"):  # not held, not legal
            await self.fire()
        seat, move, actor = self.moves()[-1]
        self.assertEqual((seat, actor), (1, "character"))
        self.assertIn(move, ("draw", "timeout"))
        self.assertEqual(self.snap()["view"]["turn"], 2)

    async def test_when_nothing_is_accepted_the_seat_forfeits_so_the_table_never_hangs(self):
        await self.ann_on_turn()
        real = self.store.play

        def play(guild_id, round_id, seat, move, actor, expected):
            if move != "forfeit":
                raise GameError("refused")
            return real(guild_id, round_id, seat, move, actor, expected)
        self.store.play = play
        await self.fire()
        self.assertEqual(self.moves()[-1], (1, "forfeit", "timeout"))
        self.assertFalse(self.snap()["view"]["active"][1])

    async def test_no_discord_call_is_made_under_a_table_lock(self):
        await self.ann_on_turn()
        await self.fire()
        await self.onecard.flush()
        self.assertTrue(HTTP)
        self.assertFalse([kind for kind, locked in HTTP if locked])


# ----------------------------------------------------------------------------------------------------------------------------------
# Character moves by model
# ----------------------------------------------------------------------------------------------------------------------------------
class ModelCase(CharCase):
    """Talk on, Character moves "model", a fake model. ``answers["move"]`` is a value, an exception, or ``f(name, schema)``."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.store.set_game_character_talk(G, True)
        self.calls, self.lock_refs = [], []
        self.answers = {"move": {"move": "draw", "line": ""}}
        self.in_flight = 0
        self.record_usage = False
        original = self.onecard.runtime.lock

        def spy(key):
            lock = original(key)
            self.lock_refs.append(lock)  # a strong reference, so the weak lock registry cannot hide a held lock
            return lock
        self.onecard.runtime.lock = spy
        self.bot.models.structured_compiled = self.model

    async def model(self, role, request, schema_name, schema):
        text = "\n".join(m.text for m in request.messages)
        name = request.messages[0].text.split("You are ", 1)[1].split(",")[0].split("\n")[0]
        self.calls.append({"name": name, "text": text, "schema": schema, "schema_name": schema_name, "role": role, "menu": schema["properties"]["move"]["enum"],
                           "locked": any(lock.locked() for lock in self.lock_refs)})
        self.in_flight += 1
        try:
            await asyncio.sleep(0)
            answer = self.answers["move"]
            if callable(answer):
                answer = answer(name, schema)
                if asyncio.iscoroutine(answer):
                    answer = await answer
            if isinstance(answer, BaseException):
                raise answer
            if self.record_usage:
                self.add_usage(1)
            return answer
        finally:
            self.in_flight -= 1

    def add_usage(self, count):
        for _ in range(count):
            self.store.db.execute("INSERT INTO model_usage(guild_id,profile,model,role,cached_tokens,reasoning_tokens,cost_basis,created_at,feature) VALUES(?,?,?,?,0,0,?,?,?)",
                                  (G, "p", "m", "director", "none", time.time(), "game"))
        self.store.db.commit()

    def patch_spending(self, result):
        def state(store, now):
            if isinstance(result, BaseException):
                raise result
            return result
        patcher = mock.patch("llmcord_core.discord_bot.budget.state", state)
        patcher.start()
        self.addCleanup(patcher.stop)

    def failures(self):
        return sum(self.onecard.talk_failed.values())

    async def expect_policy_play(self, failures=1):
        expected = self.policy_for(1)
        before = len(self.moves())
        await self.fire()
        self.assertEqual(len(self.moves()), before + 1)
        self.assertEqual(self.moves()[-1], (1, expected, "character"))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.failures(), failures)


class ModelMoveTests(ModelCase):
    async def test_a_legal_model_move_is_applied_for_the_character(self):
        await self.ann_on_turn()
        self.assertNotEqual(self.policy_for(1), "play:9♣")
        self.answers["move"] = {"move": "play:9♣", "line": ""}
        await self.fire()
        self.assertEqual(self.moves()[-1], (1, "play:9♣", "character"))
        (call,) = self.calls
        self.assertEqual((call["name"], call["role"], call["schema_name"], call["locked"]), ("Ann", "director", "onecard_move", False))
        self.assertEqual(self.failures(), 0)

    async def test_a_model_draw_is_applied(self):
        await self.ann_on_turn()
        self.answers["move"] = {"move": "draw", "line": ""}
        await self.fire()
        self.assertEqual(self.moves()[-1], (1, "draw", "character"))

    async def test_a_seven_with_a_named_suit_is_applied_and_names_the_suit(self):
        await self.ann_on_turn(ann=ANN_SEVEN)
        self.answers["move"] = {"move": "play:7♥=♠", "line": ""}
        await self.fire()
        self.assertEqual(self.moves()[-1], (1, "play:7♥=♠", "character"))
        self.assertEqual(self.snap()["view"]["suit"], "♠")

    async def test_the_menu_is_exactly_the_engines_legal_moves(self):
        await self.ann_on_turn(ann=ANN_SEVEN)
        legal = list(self.legal_for(1))
        await self.fire()
        (call,) = self.calls
        self.assertEqual(call["menu"], legal)
        self.assertIn("Legal moves: " + ", ".join(legal), call["text"])
        self.assertIn("play:7♥=♠", call["menu"])
        self.assertNotIn("play:7♥", call["menu"])  # a seven only with a suit

    async def test_a_card_that_is_not_held_or_not_playable_falls_back(self):
        await self.ann_on_turn()
        for bogus in ("play:A♠", "play:K♦", "play:JKb"):  # not held / held but not playable / not held
            self.assertNotIn(bogus, self.legal_for(1))
        self.answers["move"] = {"move": "play:K♦", "line": ""}
        await self.expect_policy_play()

    async def test_a_seven_without_a_suit_falls_back(self):
        await self.ann_on_turn(ann=ANN_SEVEN)
        self.answers["move"] = {"move": "play:7♥", "line": ""}
        await self.expect_policy_play()

    async def test_moves_outside_the_menu_fall_back(self):
        for n, move in enumerate(("forfeit", "timeout", "declare", "call_out:0", "")):
            with self.subTest(move=move):
                self.calls.clear()
                self.onecard.talk_failed.clear()
                self.answers["move"] = {"move": move, "line": ""}
                if n == 0:
                    await self.ann_on_turn()
                else:  # a fresh Ann turn: Ann draws, then the members draw back to her
                    await self.draw_until(1)
                self.assertEqual(self.snap()["view"]["turn"], 1)
                expected = self.policy_for(1)
                await self.fire()
                self.assertEqual(self.moves()[-1][1:], (expected, "character"))
                self.assertNotIn(self.moves()[-1][1], ("forfeit", "timeout", "declare"))
                self.assertEqual(self.failures(), 1)

    async def test_malformed_answers_fall_back(self):
        await self.ann_on_turn()
        self.answers["move"] = ["draw"]
        await self.expect_policy_play()

    async def test_a_move_that_is_not_text_falls_back(self):
        await self.ann_on_turn()
        self.answers["move"] = {"move": ["draw"], "line": ""}
        await self.expect_policy_play()

    async def test_a_line_is_never_read_as_a_move(self):
        await self.ann_on_turn()
        self.answers["move"] = {"line": "draw"}
        await self.expect_policy_play()

    async def test_a_model_error_falls_back(self):
        await self.ann_on_turn()
        self.answers["move"] = RuntimeError("provider down")
        await self.expect_policy_play()

    async def test_a_spending_hard_cap_error_falls_back(self):
        await self.ann_on_turn()
        self.answers["move"] = BudgetExceeded(SimpleNamespace())
        await self.expect_policy_play()

    async def test_a_slow_model_times_out_and_falls_back(self):
        await self.ann_on_turn()
        self.onecard.MODEL_SECONDS = 0.05

        async def slow(name, schema):
            await asyncio.sleep(5)
        self.answers["move"] = slow
        await self.expect_policy_play()

    async def test_the_saved_play_style_is_in_the_prompt_in_words(self):
        await self.ann_on_turn()
        style = PlayStyle(aggression=95, risk=90)
        self.store.set_character_play_style(G, self.ann, style, self.store.owner_revision(G, "character", self.ann))
        await self.fire()
        self.assertIn(f"Play style: {describe(style)}.", self.calls[0]["text"])

    async def test_a_late_second_timer_for_the_same_turn_is_dropped(self):
        await self.ann_on_turn()
        self.answers["move"] = {"move": "play:9♣", "line": ""}
        snap, before = self.snap(), len(self.moves())
        await asyncio.gather(*(self.onecard.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"]) for _ in range(2)))
        self.assertEqual(len(self.moves()), before + 1)

    async def test_a_member_who_moves_during_the_model_call_wins_the_race(self):
        """The decision is made for a move count; if the table moved on (a forfeit from outside) nothing is applied."""
        await self.ann_on_turn()
        started, release = asyncio.Event(), asyncio.Event()

        async def blocked(name, schema):
            started.set()
            await release.wait()
            return {"move": "play:9♣", "line": ""}
        self.answers["move"] = blocked
        task = asyncio.create_task(self.fire())
        await asyncio.wait_for(started.wait(), 1)
        self.store_play(2, "forfeit", "member")  # Bob leaves while Ann thinks (off turn, allowed): the move count changed
        release.set()
        await task
        self.assertEqual(self.moves()[-1], (2, "forfeit", "member"))

    async def test_cancelling_the_timer_mid_decision_applies_nothing_and_counts_the_call(self):
        await self.ann_on_turn()
        started = asyncio.Event()

        async def blocked(name, schema):
            started.set()
            await asyncio.sleep(5)
        self.answers["move"] = blocked
        before = len(self.moves())
        task = asyncio.create_task(self.fire())
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(self.moves()), before)
        self.assertEqual(self.in_flight, 0)
        self.assertEqual(self.failures(), 1)

    async def test_no_model_call_runs_under_the_table_lock(self):
        await self.ann_on_turn()
        await self.fire()
        self.assertTrue(self.calls)
        self.assertEqual([call["locked"] for call in self.calls], [False] * len(self.calls))
        self.assertTrue(self.lock_refs)


class ModelPromptTests(ModelCase):
    async def test_the_prompt_holds_only_anns_view(self):
        await self.ann_on_turn()
        mine, hidden = self.held(1), self.hidden_from(1)
        legal = list(self.legal_for(1))
        await self.fire()
        text = self.calls[0]["text"]
        self.assertIn("Your hand: " + ", ".join(mine), text)
        self.assertIn("Top card: 9♥", text)
        self.assertIn("Legal moves: " + ", ".join(legal), text)
        self.assertIn("Player 1: 6 cards", text)  # Alice, after playing the 9♥
        self.assertIn("Player 2: 7 cards", text)  # Bob
        self.assertIn("Player 1 played 9♥", text)
        self.assertTrue(hidden)
        for card in hidden:
            self.assertNotIn(card, text)
        self.assertNotIn("<@", text)
        for uid in (ALICE, BOB):
            for needle in (f"<@{uid}>", f"<@!{uid}>", f"@{uid}", f"id {uid}", f"id={uid}", f"user {uid}", f"user_id {uid}"):
                self.assertNotIn(needle, text)
        for needle in ("Alice", "Bob", "seed"):
            self.assertNotIn(needle, text)

    async def test_two_characters_each_get_only_their_own_hand(self):
        self.store.add_favorite(G, ALICE, self.bo)
        bo_hand = ["2♠", "5♠", "J♣", "Q♦", "A♣", "10♥", "6♦"]
        self.rig([ALICE_HAND, ANN_PLAYS, bo_hand, BOB_HAND], "5♥")
        await self.start(favorites=True)
        await self.push(BOB, "Join", self.lobby())
        await self.push(ALICE, "Start", self.lobby())
        await self.pick_play(ALICE, "play:9♥")
        for _ in range(2):
            self.calls.clear()
            seat = self.snap()["view"]["turn"]
            name = self.snap()["seats"][seat]["name"]
            self.assertIn(name, ("Ann", "Bo"))
            hidden = self.hidden_from(seat)
            await self.fire()
            (call,) = self.calls
            self.assertEqual(call["name"], name)
            for card in hidden:
                self.assertNotIn(card, call["text"])
        self.assertEqual({s["name"] for s in self.snap()["seats"] if s["kind"] == "character"}, {"Ann", "Bo"})


class GatingTests(ModelCase):
    async def test_style_mode_never_calls_the_model_even_with_talk_on(self):
        self.store.set_game_character_moves(G, "style")
        await self.ann_on_turn()
        expected = self.policy_for(1)
        await self.fire()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.moves()[-1], (1, expected, "character"))
        self.assertEqual(self.failures(), 0)

    async def test_talk_off_plays_the_policy_without_a_model_call(self):
        self.store.set_game_character_talk(G, False)
        await self.ann_on_turn()
        expected = self.policy_for(1)
        await self.fire()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.moves()[-1], (1, expected, "character"))

    async def test_at_the_daily_limit_no_model_call_is_made(self):
        self.store.set_game_talk_daily_limit(G, 2)
        self.add_usage(2)
        await self.ann_on_turn()
        expected = self.policy_for(1)
        self.answers["move"] = {"move": "play:9♣", "line": ""}
        await self.fire()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.moves()[-1][1], expected)

    async def test_below_the_daily_limit_the_model_is_asked(self):
        self.store.set_game_talk_daily_limit(G, 2)
        self.add_usage(1)
        await self.ann_on_turn()
        await self.fire()
        self.assertEqual(len(self.calls), 1)

    async def test_the_hard_spending_cap_plays_the_policy_without_a_call_or_a_failure(self):
        self.patch_spending(SimpleNamespace(hard_reached=True, soft_reached=True))
        await self.ann_on_turn()
        expected = self.policy_for(1)
        await self.fire()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.moves()[-1][1], expected)
        self.assertEqual(self.failures(), 0)

    async def test_an_unreadable_spending_state_fails_open(self):
        self.patch_spending(RuntimeError("db"))
        await self.ann_on_turn()
        await self.fire()
        self.assertEqual(len(self.calls), 1)

    async def test_a_spending_state_below_the_cap_asks_the_model(self):
        self.patch_spending(SimpleNamespace(hard_reached=False, soft_reached=False))
        await self.ann_on_turn()
        await self.fire()
        self.assertEqual(len(self.calls), 1)

    async def test_talk_allowed_follows_the_setting_the_limit_and_failed_attempts(self):
        self.assertTrue(self.onecard.talk_allowed(G))
        self.store.set_game_talk_daily_limit(G, 2)
        self.onecard.talk.note_failure(G)
        self.assertTrue(self.onecard.talk_allowed(G))
        self.onecard.talk.note_failure(G)
        self.assertFalse(self.onecard.talk_allowed(G))
        self.assertEqual(self.onecard._talk_budget(G), 0)
        self.store.set_game_talk_daily_limit(G, 5)
        self.assertTrue(self.onecard.talk_allowed(G))
        self.store.set_game_character_talk(G, False)
        self.assertFalse(self.onecard.talk_allowed(G))

    async def test_the_failed_attempts_are_shared_with_the_other_table_games(self):
        self.assertIs(self.onecard.talk, self.bot.doubt.talk)
        self.assertIs(self.onecard.talk, self.bot.holdem.talk)

    async def to_ann_again(self):
        await self.fire()
        await self.draw_until(1)

    async def test_a_failed_call_counts_toward_the_daily_limit(self):
        self.store.set_game_talk_daily_limit(G, 1)
        await self.ann_on_turn()
        self.answers["move"] = RuntimeError("down")
        await self.to_ann_again()
        self.assertEqual(len(self.calls), 1)
        await self.fire()
        self.assertEqual(len(self.calls), 1)  # the limit of one was used up by the failed attempt

    async def test_a_successful_call_counts_toward_the_daily_limit(self):
        self.store.set_game_talk_daily_limit(G, 1)
        self.record_usage = True  # the real call records usage as feature 'game'
        await self.ann_on_turn()
        await self.to_ann_again()
        self.assertEqual(len(self.calls), 1)
        await self.fire()
        self.assertEqual(len(self.calls), 1)

    async def test_a_timed_out_call_counts_toward_the_daily_limit(self):
        self.store.set_game_talk_daily_limit(G, 1)
        self.onecard.MODEL_SECONDS = 0.05

        async def slow(name, schema):
            await asyncio.sleep(5)
        self.answers["move"] = slow
        await self.ann_on_turn()
        await self.to_ann_again()
        await self.fire()
        self.assertEqual(len(self.calls), 1)


# ----------------------------------------------------------------------------------------------------------------------------------
# Whole games, money
# ----------------------------------------------------------------------------------------------------------------------------------
class MoneyTests(CharCase):
    async def test_a_character_who_wins_takes_the_pot(self):
        await self.scene(self.three(ann=["9♣"]))  # Ann's only card matches the 9♥ Alice opens with
        thread = self.thread()
        await self.pick_play(ALICE, "play:9♥")
        await self.fire()
        self.assertEqual(self.store.latest_round(G, self.tid)["status"], "settled")
        self.assertEqual(self.store.character_balance(G, self.ann), 100 - ANTE + 3 * ANTE)
        self.assertEqual(self.balance(ALICE), 100 - ANTE)
        self.assertEqual(self.balance(BOB), 100 - ANTE)
        self.assertEqual(self.store.ledger(G, character_id=self.ann)[0]["reason"], PAYOUT_REASON)
        self.assertTrue(any("Ann" in text for text in self.thread_texts(thread)))

    async def test_a_member_who_wins_leaves_the_character_with_less(self):
        await self.scene(self.three(alice=["9♥"]))
        await self.pick_play(ALICE, "play:9♥")
        self.assertEqual(self.balance(ALICE), 100 - ANTE + 3 * ANTE)
        self.assertEqual(self.store.character_balance(G, self.ann), 100 - ANTE)
        self.assertEqual(self.balance(BOB), 100 - ANTE)

    async def test_a_forfeiting_character_loses_her_ante_and_the_game_goes_on(self):
        await self.ann_on_turn()
        real = self.store.play

        def play(guild_id, round_id, seat, move, actor, expected):
            if move != "forfeit":
                raise GameError("refused")
            return real(guild_id, round_id, seat, move, actor, expected)
        self.store.play = play
        await self.fire()
        self.store.play = real
        self.assertEqual(self.store.latest_round(G, self.tid)["status"], "playing")
        self.assertEqual(self.store.character_balance(G, self.ann), 100 - ANTE)

    async def test_a_scripted_game_with_a_character_ends_once_and_conserves_the_money(self):
        wallets = lambda: self.balance(ALICE) + self.balance(BOB) + self.store.character_balance(G, self.ann)  # noqa: E731
        self.assertEqual(wallets(), 300)
        await self.start(favorites=True)
        await self.push(BOB, "Join", self.lobby())
        await self.push(ALICE, "Start", self.lobby())
        tid, users, thread = self.table()["id"], {0: ALICE, 2: BOB}, self.thread()
        for _ in range(1500):
            snap = self.store.latest_round(G, tid)
            if snap["status"] != "playing":
                break
            state = self.engine_state(snap)
            if state.turn == 1:
                await self.onecard.on_character_timer(G, tid, snap["round_id"], snap["moves"])
            else:
                _, move = pick_move(state)
                await self.onecard.play_from_picker(self.click(users[state.turn], channel=thread), tid, snap["round_id"], snap["moves"], move)
        self.assertEqual(self.store.latest_round(G, tid)["status"], "settled")
        self.assertEqual(wallets(), 300)
        self.assertEqual(sorted([self.balance(ALICE), self.balance(BOB), self.store.character_balance(G, self.ann)]), [90, 90, 120])  # the winner took the pot


# ----------------------------------------------------------------------------------------------------------------------------------
# Table talk
# ----------------------------------------------------------------------------------------------------------------------------------
class EventCase(CharCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.store.set_game_character_talk(G, True)
        self.store.set_game_character_moves(G, "style")  # the policy plays: these tests are about talk, not model moves

    def assert_clean(self, event):
        text = event["text"]
        self.assertTrue(text.strip())
        for card in event["hidden"]:
            self.assertNotIn(card, text)
        self.assertNotIn("<@", text)
        self.assertNotIn("*", text)


class EventMomentTests(EventCase):
    async def test_an_attack_card_is_a_key_moment(self):
        await self.scene(self.three(alice=["2♥", *ALICE_HAND[1:]]))
        await self.pick_play(ALICE, "play:2♥")
        await self.settle()
        (event,) = self.events
        self.assertIs(event["thread"], self.thread())
        self.assert_clean(event)

    async def test_a_plain_card_and_a_single_card_draw_say_nothing(self):
        await self.scene(self.three(alice=["4♥", *ALICE_HAND[1:]], ann=ANN_STUCK))
        await self.pick_play(ALICE, "play:4♥")  # a 4 is plain
        await self.fire()  # Ann plays her plain 4♦ (or draws)
        await self.pick_play(BOB, "draw")  # no attack pending: a single card
        await self.settle()
        self.assertEqual(self.events, [])

    async def test_a_skip_and_a_reverse_are_moments(self):
        for card in ("J♥", "Q♥"):
            with self.subTest(card=card):
                self.events.clear()
                self.now[0] += 1000
                await self.fresh_table(card)
                await self.pick_play(ALICE, f"play:{card}")
                await self.settle()
                (event,) = self.events
                self.assert_clean(event)

    async def fresh_table(self, card):
        """A new table (the previous one is closed first) where Alice opens with ``card`` on the 5♥."""
        for table in [self.store.open_table_for(G, CH)]:
            if table is not None:
                self.store.close_table(G, table["id"])
        patcher = mock.patch.object(onecard, "new", rigged_new(self.three(alice=[card, *ALICE_HAND[1:]]), "5♥"))
        patcher.start()
        self.addCleanup(patcher.stop)
        await self.start(favorites=True)
        await self.push(BOB, "Join", self.lobby())
        await self.push(ALICE, "Start", self.lobby())
        self.tid = self.table()["id"]

    async def test_drawing_a_big_attack_pile_is_a_moment(self):
        await self.scene(self.three(), top="A♠", pending=5)
        await self.pick_play(ALICE, "draw")
        await self.settle()
        (event,) = self.events
        self.assert_clean(event)

    async def test_a_seat_down_to_one_card_is_a_moment_and_names_her(self):
        await self.ann_on_turn(ann=["9♣", "K♦"])
        await self.settle()
        self.events.clear()
        self.now[0] += 100
        await self.fire()  # Ann plays her 9♣ and has the K♦ left
        await self.settle()
        (event,) = self.events
        self.assertIn("Ann", event["text"])
        self.assertIn("one card", event["text"].lower())
        self.assertNotIn("K♦", event["text"])
        self.assert_clean(event)

    async def test_a_member_down_to_one_card_is_a_moment(self):
        await self.scene(self.three(alice=["9♥", "3♣"]))
        await self.pick_play(ALICE, "play:9♥")
        await self.settle()
        (event,) = self.events
        self.assertIn("one card", event["text"].lower())
        self.assertNotIn("3♣", event["text"])
        self.assert_clean(event)

    async def test_the_win_is_a_moment_and_names_the_winner(self):
        await self.scene(self.three(ann=["9♣"]))
        await self.pick_play(ALICE, "play:9♥")
        await self.settle()
        self.events.clear()
        await self.fire()
        await self.settle()
        (event,) = self.events
        self.assertTrue(event["finished"])
        self.assertIn("Ann", event["text"])
        self.assertIn("won", event["text"])
        self.assert_clean(event)

    async def test_the_win_ignores_the_cooldown(self):
        await self.scene(self.three(alice=["2♥", "9♥"], ann=ANN_STUCK))
        await self.pick_play(ALICE, "play:2♥")  # attack: a moment (and Alice is down to one card)
        await self.settle()
        self.assertEqual(len(self.events), 1)
        await self.fire()  # Ann cannot defend: she draws 2
        await self.pick_play(BOB, "draw")  # a single card
        await self.pick_play(ALICE, "play:9♥")  # the win, inside the cooldown
        await self.settle()
        self.assertTrue(self.events[-1]["finished"])
        self.assertEqual(sum(e["finished"] for e in self.events), 1)


class EventGateTests(EventCase):
    async def test_a_second_moment_inside_the_cooldown_is_dropped_and_one_after_it_fires(self):
        hands = self.three(alice=["2♥", *ALICE_HAND[1:]], ann=["2♣", "9♣", "7♠", "K♦", "4♦", "3♦", "J♦"], bob=["JKb", *BOB_HAND[1:]])
        await self.scene(hands)
        await self.pick_play(ALICE, "play:2♥")
        await self.settle()
        self.assertEqual(len(self.events), 1)
        self.now[0] += 19
        await self.fire()  # Ann defends with the 2♣: an attack inside the cooldown
        await self.settle()
        self.assertEqual(self.moves()[-1][1], "play:2♣")
        self.assertEqual(len(self.events), 1)
        self.now[0] += 2
        await self.pick_play(BOB, "play:JKb")  # a joker answers: an attack after the cooldown
        await self.settle()
        self.assertEqual(len(self.events), 2)
        for event in self.events:
            self.assert_clean(event)

    async def test_talk_off_means_no_event_at_all_not_even_the_win(self):
        self.store.set_game_character_talk(G, False)
        await self.scene(self.three(alice=["2♥", "9♥"], ann=["9♣"]))
        await self.pick_play(ALICE, "play:2♥")
        await self.fire()  # Ann cannot defend with her 9♣: she draws 2
        await self.draw_until(0)
        await self.pick_play(ALICE, "play:9♥")
        await self.settle()
        self.assertEqual(self.store.latest_round(G, self.tid)["status"], "settled")
        self.assertEqual(self.events, [])

    async def test_at_the_daily_limit_the_table_stays_silent(self):
        self.store.set_game_talk_daily_limit(G, 1)
        self.add_usage(1)
        await self.scene(self.three(alice=["2♥", *ALICE_HAND[1:]]))
        await self.pick_play(ALICE, "play:2♥")
        await self.settle()
        self.assertEqual(self.events, [])

    def add_usage(self, count):
        for _ in range(count):
            self.store.db.execute("INSERT INTO model_usage(guild_id,profile,model,role,cached_tokens,reasoning_tokens,cost_basis,created_at,feature) VALUES(?,?,?,?,0,0,?,?,?)",
                                  (G, "p", "m", "director", "none", time.time(), "game"))
        self.store.db.commit()

    async def test_failed_move_attempts_count_toward_the_limit(self):
        self.store.set_game_talk_daily_limit(G, 2)
        self.onecard.talk.note_failure(G)
        self.onecard.talk.note_failure(G)
        await self.scene(self.three(alice=["2♥", *ALICE_HAND[1:]]))
        await self.pick_play(ALICE, "play:2♥")
        await self.settle()
        self.assertEqual(self.events, [])

    async def test_a_moment_suppressed_by_the_gate_does_not_use_up_the_cooldown(self):
        self.store.set_game_character_talk(G, False)
        await self.scene(self.three(alice=["2♥", *ALICE_HAND[1:]], ann=["2♣", "9♣", "7♠", "K♦", "4♦", "3♦", "J♦"]))
        await self.pick_play(ALICE, "play:2♥")
        await self.settle()
        self.assertEqual(self.events, [])
        self.store.set_game_character_talk(G, True)
        await self.fire()  # Ann defends: a moment, right after the suppressed one
        await self.settle()
        self.assertEqual(len(self.events), 1)

    async def test_a_move_does_not_wait_for_the_event_turn(self):
        self.hold = asyncio.Event()
        await self.scene(self.three(alice=["2♥", *ALICE_HAND[1:]]))
        await asyncio.wait_for(self.pick_play(ALICE, "play:2♥"), 1)
        await self.settle()
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.moves()[-1][1], "play:2♥")

    async def test_a_failing_event_turn_does_not_break_the_game(self):
        async def boom(thread, text, speaker_hint=None, spectators=False):
            raise RuntimeError("talk down")
        self.bot.run_table_event = boom
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)
        await self.scene(self.three(alice=["2♥", *ALICE_HAND[1:]]))
        await self.pick_play(ALICE, "play:2♥")
        await self.settle()
        self.assertEqual(self.snap()["status"], "playing")
        self.assertEqual(self.snap()["view"]["turn"], 1)

    async def test_a_scripted_game_never_says_a_hidden_card_in_an_event(self):
        self.store.set_game_talk_daily_limit(G, 10_000)
        step = [0]

        def clock():
            step[0] += 25
            return 10_000.0 + step[0]
        self.onecard.clock = clock
        await self.start(favorites=True)
        await self.push(BOB, "Join", self.lobby())
        await self.push(ALICE, "Start", self.lobby())
        self.tid = tid = self.table()["id"]
        users, thread = {0: ALICE, 2: BOB}, self.thread()
        for n in range(1500):
            snap = self.store.latest_round(G, tid)
            if snap["status"] != "playing":
                break
            state = self.engine_state(snap)
            if state.turn == 1:
                await self.onecard.on_character_timer(G, tid, snap["round_id"], snap["moves"])
            else:
                _, move = pick_move(state)
                await self.onecard.play_from_picker(self.click(users[state.turn], channel=thread), tid, snap["round_id"], snap["moves"], move)
            await self.settle()
        self.assertEqual(self.store.latest_round(G, tid)["status"], "settled")
        finals = [e for e in self.events if e["finished"]]
        self.assertEqual(len(finals), 1)
        self.assertIs(self.events[-1], finals[0])
        for event in self.events:
            self.assert_clean(event)


# ----------------------------------------------------------------------------------------------------------------------------------
# Context and block, then real talk turns
# ----------------------------------------------------------------------------------------------------------------------------------
class BlockTests(CharCase):
    async def test_a_joining_lobby_thread_already_reports_its_seated_characters(self):
        await self.start(favorites=True)
        thread = self.thread()
        ctx = self.onecard.table_talk_context(G, thread.id)
        self.assertEqual((ctx["status"], ctx["seated_character_ids"]), ("joining", [self.ann]))
        self.assertEqual(set(ctx), {"table_id", "round_id", "status", "seated_character_ids"})

    async def test_a_playing_table_reports_its_context_and_other_threads_have_none(self):
        await self.scene(self.three())
        ctx = self.onecard.table_talk_context(G, self.thread().id)
        self.assertEqual((ctx["table_id"], ctx["round_id"], ctx["status"], ctx["seated_character_ids"]),
                         (self.tid, self.snap()["round_id"], "playing", [self.ann]))
        self.assertIsNone(self.onecard.table_talk_context(G, 424242))
        self.assertIsNone(self.onecard.table_talk_context(G + 1, self.thread().id))
        self.assertIsNone(self.bot.doubt.table_talk_context(G, self.thread().id))  # a One Card thread is not a Doubt table

    async def test_the_banter_after_the_game_keeps_the_context_until_the_archive_timer(self):
        await self.scene(self.three(alice=["9♥"]))
        thread_id = self.thread().id
        await self.pick_play(ALICE, "play:9♥")
        ctx = self.onecard.table_talk_context(G, thread_id)
        self.assertEqual(ctx["status"], "settled")
        self.assertEqual(ctx["seated_character_ids"], [self.ann])
        await self.onecard.on_archive_timer(G, self.tid)
        self.assertIsNone(self.onecard.table_talk_context(G, thread_id))

    async def test_the_block_holds_public_state_and_only_the_speakers_own_hand(self):
        await self.scene(self.three())
        block = self.onecard.table_block(G, self.thread().id, self.ann)
        self.assertIn("One Card", block)
        self.assertIn("5♥", block)
        self.assertRegex(block, r"\b7 cards?\b")
        self.assertIn("Ann", block)
        self.assertEqual(block.count("Your hand:"), 1)
        self.assertIn("Your hand: " + ", ".join(self.held(1)), block)
        self.assertIn("Keep your hand secret", block)
        for card in self.hidden_from(1):
            self.assertNotIn(card, block)
        self.assertNotIn("<@", block)

    async def test_the_block_follows_the_public_play(self):
        await self.ann_on_turn()
        block = self.onecard.table_block(G, self.thread().id, self.ann)
        self.assertIn("9♥", block)
        self.assertIn("on turn", block)
        for card in self.hidden_from(1):
            self.assertNotIn(card, block)

    async def test_the_block_is_none_for_anyone_not_seated_and_for_other_threads_and_guilds(self):
        await self.scene(self.three())
        thread = self.thread()
        self.assertIsNone(self.onecard.table_block(G, thread.id, self.bo))
        self.assertIsNone(self.onecard.table_block(G, 424242, self.ann))
        self.assertIsNone(self.onecard.table_block(G + 1, thread.id, self.ann))
        self.assertIsNone(self.bot.doubt.table_block(G, thread.id, self.ann))

    async def test_the_block_after_the_game_has_no_hand_line(self):
        await self.scene(self.three(alice=["9♥"]))
        thread_id = self.thread().id
        await self.pick_play(ALICE, "play:9♥")
        block = self.onecard.table_block(G, thread_id, self.ann)
        self.assertIsNotNone(block)
        self.assertNotIn("Your hand:", block)

    async def test_members_are_named_by_display_name_when_known_and_never_mentioned(self):
        await self.scene(self.three())
        guild = SimpleNamespace(get_member=lambda ident: SimpleNamespace(display_name=NAMES.get(ident)) if ident in (ALICE, BOB) else None)
        self.bot.get_guild = lambda gid: guild
        block = self.onecard.table_block(G, self.thread().id, self.ann)
        for name in ("Alice", "Bob"):
            self.assertIn(name, block)
        self.assertNotIn("Player", block)
        self.assertNotIn("<@", block)

    async def test_the_bot_finds_the_one_card_table_for_a_thread(self):
        await self.scene(self.three())
        game, ctx = self.bot._table_game(G, self.thread().id)
        self.assertIs(game, self.onecard)
        self.assertEqual(ctx["seated_character_ids"], [self.ann])
        self.assertEqual(self.bot._table_block(G, self.thread().id, self.ann), self.onecard.table_block(G, self.thread().id, self.ann))
        self.assertIsNone(self.bot._table_block(G, self.thread().id, self.bo))


class TalkModels(CompiledAdapter):
    """Records every request (director / dialogue / memory) with the speaker (set by the webhook lookup before each dialogue call)."""

    def __init__(self):
        self.calls, self.pick, self.speaker = [], [], None

    def record(self, kind, request, role):
        entry = {"kind": kind, "role": role, "text": "\n".join(m.text for m in request.messages), "speaker": self.speaker}
        self.calls.append(entry)
        return entry

    async def structured_compiled(self, role, request, name, schema):
        if name == "choose_speakers":
            self.record("director", request, role)
            return {"speakers": list(self.pick)}
        if name == "extract_memory":
            self.record("extract_memory", request, role)
            return {"shared_facts": [], "personal_facts": [], "encounter_facts": []}
        self.record("other:" + name, request, role)
        raise RuntimeError("no character-move model in the talk tests")

    async def text_compiled(self, role, request, max_tokens=None):
        self.record("summary", request, role)
        return "Earlier scene summary"

    async def stream_compiled(self, role, request):
        self.record("dialogue", request, role)
        yield "<emotion>neutral</emotion>\nHello."

    async def close(self):
        pass

    def of(self, kind):
        return [call for call in self.calls if call["kind"] == kind]


class TalkTurnTests(CharCase):
    """Real table-thread turns through ``SkitBot.on_message`` / ``run_table_event`` with the One Card table's block in the speaker's prompt."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        del self.bot.run_table_event  # the real one
        self.store.set_game_character_talk(G, True)
        self.models = TalkModels()
        self.bot.engine.models = self.bot.models = self.models
        self.bot._connection.user = BOT
        self.hooks = {cid: FakeWebhook(60000 + 1000 * cid) for cid in (self.ann, self.bo)}

        async def webhook(channel, character):
            self.models.speaker = character["id"]
            return self.hooks[character["id"]]
        self.bot._webhook = webhook
        logging.disable(logging.WARNING)
        self.addCleanup(logging.disable, logging.NOTSET)

        async def drain():
            await drain_memory_tasks(self.bot)
        self.addAsyncCleanup(drain)

    def spoke(self):
        return [cid for cid, hook in self.hooks.items() if hook.posts]

    async def say(self, user=ALICE, text="Hello there"):
        thread = self.thread()

        async def history(**kwargs):
            return
            yield
        thread.history = history
        message = thread_message(thread, user, "<@999> " + text)
        message.author.display_name = NAMES.get(user, f"User {user}")
        message.mentions = [BOT]
        await self.bot.on_message(message)
        await drain_memory_tasks(self.bot)

    async def test_a_message_in_the_thread_is_answered_by_the_seated_character_with_her_own_hand(self):
        await self.scene(self.three())
        self.models.pick = [self.ann]
        await self.say()
        (dialogue,) = self.models.of("dialogue")
        self.assertEqual(dialogue["speaker"], self.ann)
        block = self.onecard.table_block(G, self.thread().id, self.ann)
        self.assertIn(block, dialogue["text"])
        self.assertIn("Your hand: " + ", ".join(self.held(1)), dialogue["text"])
        for card in self.hidden_from(1):
            self.assertNotIn(card, dialogue["text"])
        self.assertEqual(self.spoke(), [self.ann])

    async def test_a_cast_member_who_is_not_seated_never_speaks_even_when_chosen(self):
        await self.scene(self.three())
        self.models.pick = [self.bo, self.ann]
        await self.say()
        self.assertEqual(self.spoke(), [self.ann])
        for call in self.models.calls:
            for card in self.hidden_from(1):
                self.assertNotIn(card, call["text"], f"a hidden card leaked into a {call['kind']} request")

    async def test_the_director_request_has_no_table_block_or_hand(self):
        await self.scene(self.three())
        self.models.pick = [self.ann]
        await self.say()
        for call in self.models.of("director"):
            self.assertNotIn("Your hand:", call["text"])
            self.assertNotIn("Keep your hand secret", call["text"])

    async def test_with_character_talk_off_the_characters_stay_silent(self):
        self.store.set_game_character_talk(G, False)
        await self.scene(self.three())
        self.models.pick = [self.ann]
        await self.say()
        self.assertEqual(self.models.of("dialogue"), [])
        self.assertEqual(self.spoke(), [])

    async def test_an_event_turn_runs_one_director_turn_restricted_to_the_seated_characters(self):
        await self.scene(self.three())
        self.models.pick = [self.bo, self.ann]
        await self.bot.run_table_event(self.thread(), "Alice played a 2. Ann is down to one card.")
        await drain_memory_tasks(self.bot)
        self.assertEqual(len(self.models.of("director")), 1)
        self.assertEqual(self.spoke(), [self.ann])
        for call in self.models.calls:
            for card in self.hidden_from(1):
                self.assertNotIn(card, call["text"])

    async def test_an_event_turn_follows_the_talk_gate(self):
        await self.scene(self.three())
        self.store.set_game_character_talk(G, False)
        self.models.pick = [self.ann]
        await self.bot.run_table_event(self.thread(), "Alice played a 2.")
        self.assertEqual(self.models.calls, [])

    async def test_a_move_by_the_policy_schedules_the_event_through_the_real_bot(self):
        """End to end: an attack played by a member makes the bot run an event turn in the thread (Ann may answer)."""
        self.store.set_game_character_moves(G, "style")
        await self.scene(self.three(alice=["2♥", *ALICE_HAND[1:]]))
        self.models.pick = [self.ann]
        await self.pick_play(ALICE, "play:2♥")
        await asyncio.gather(*self.onecard.event_tasks)
        await drain_memory_tasks(self.bot)
        self.assertEqual(len(self.models.of("director")), 1)
        self.assertEqual(self.spoke(), [self.ann])
        for call in self.models.calls:
            for card in self.hidden_from(1):
                self.assertNotIn(card, call["text"])


# ----------------------------------------------------------------------------------------------------------------------------------
# FEAT-48: spectators (the public view; a cast member who is not seated)
# ----------------------------------------------------------------------------------------------------------------------------------
class PublicViewTests(CharCase):
    """``table_view.public_table_view`` covers ``onecard``: the top card, suit, direction, pending attack, hand sizes and whose turn, never a
    hand, a draw pile card or the seed."""

    def view(self, channel=None, parent=None, guild=G):
        return lookup(self.bot, guild, (channel or self.thread()).id if not isinstance(channel, int) else channel, parent)

    def public_tokens(self):
        """Cards a spectator may read: the discard pile and every card ever played."""
        state = self.engine_state()
        return set(tokens(state.discard)) | {onecard.card_text(e.card) for e in state.history if e.kind == "play"}

    def assert_public(self, text):
        state = self.engine_state()
        hidden = (set(tokens([card for hand in state.hands for card in hand])) | set(tokens(state.draw))) - self.public_tokens()
        self.assertTrue(hidden)
        for card in hidden:
            self.assertNotIn(card, text, f"{card} (hidden) leaked")
        snap = self.snap()
        for marker in ("Your hand", "(you)", "<@", "Keep your hand secret", state.seed, snap["seed_hash"][:12]):
            self.assertNotIn(marker, text)

    async def test_shape_and_resolution_from_thread_parent_and_channel(self):
        await self.scene(self.three())
        thread, snap = self.thread(), self.snap()
        found = self.view(thread)
        self.assertEqual(set(found), KEYS)
        self.assertEqual((found["game"], found["table_id"], found["thread_id"], found["status"], found["seated_character_ids"], found["moment"]),
                         ("onecard", snap["table_id"], thread.id, "playing", [self.ann], None))
        self.assertEqual(self.view(thread, CH), found)
        self.assertEqual(self.view(CH), found)
        plain = await self.ch.make_thread("plain chat")
        self.assertEqual(self.view(plain, CH), found)

    async def test_the_text_has_the_public_state_and_no_hidden_card(self):
        await self.scene(self.three(), top="5♥")
        await self.pick_play(ALICE, "play:9♥")
        text = self.view()["text"]
        self.assert_public(text)
        self.assertIn("One Card", text)
        self.assertIn("9♥", text)
        self.assertIn("clockwise", text)
        self.assertRegex(text, r"\b6 cards\b")  # Alice, after the 9♥
        self.assertRegex(text, r"\b7 cards\b")  # Ann and Bob
        self.assertIn("Ann", text)
        self.assertIn("(on turn)", text)
        self.assertNotIn("Draw pile", text)

    async def test_a_pending_attack_and_a_named_suit_are_public(self):
        await self.scene(self.three(alice=["2♥", *ALICE_HAND[1:]]))
        await self.pick_play(ALICE, "play:2♥")
        text = self.view()["text"]
        self.assertIn("Pending attack: 2", text)
        self.assert_public(text)

    async def test_a_scripted_game_never_shows_a_hidden_card_publicly(self):
        await self.start(favorites=True)
        await self.push(BOB, "Join", self.lobby())
        await self.push(ALICE, "Start", self.lobby())
        self.tid = tid = self.table()["id"]
        users, thread = {0: ALICE, 2: BOB}, self.thread()
        for _ in range(400):
            snap = self.store.latest_round(G, tid)
            if snap["status"] != "playing":
                break
            self.assert_public(self.onecard.public_block(G, (G, tid)))
            state = self.engine_state(snap)
            if state.turn == 1:
                await self.onecard.on_character_timer(G, tid, snap["round_id"], snap["moves"])
            else:
                _, move = pick_move(state)
                await self.onecard.play_from_picker(self.click(users[state.turn], channel=thread), tid, snap["round_id"], snap["moves"], move)
        self.assertEqual(self.store.latest_round(G, tid)["status"], "settled")

    async def test_joining_lobby_resolves_with_status_joining(self):
        await self.start(favorites=True)
        found = self.view()
        self.assertEqual((found["status"], found["seated_character_ids"]), ("joining", [self.ann]))
        self.assertIn("Ann", found["text"])
        self.assertNotIn("Your hand", found["text"])

    async def test_guild_isolation_and_no_table(self):
        self.assertIsNone(self.view(CH))
        await self.scene(self.three())
        thread = self.thread()
        self.assertIsNone(self.view(thread, CH, guild=G + 1))
        self.assertIsNone(self.view(CH, guild=G + 1))
        self.assertIsNone(lookup(self.bot, G + 1, thread.id))
        plain = await self.channel2.make_thread("elsewhere")
        self.assertIsNone(self.view(plain, CH2))

    async def test_names_are_sanitised(self):
        await self.scene(self.three())
        rename(self.store, self.ann, MESSY)
        text = self.view()["text"]
        self.assertIn(CLEAN, text)
        for bad in ("<@", "*", "_", MESSY):
            self.assertNotIn(bad, text)

    async def test_the_seated_block_is_unchanged_by_the_public_block(self):
        await self.scene(self.three())
        block = self.onecard.table_block(G, self.thread().id, self.ann)
        self.assertEqual(block.count("Your hand:"), 1)
        self.assertIn("Ann (you)", block)


class SpectatorTalkTests(CharCase):
    """A cast member who is not seated (Bo) watches the table: she gets the public view in her prompt, never Ann's or anyone's hand, and
    the key moments schedule the spectator reaction after the seated one."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.store.set_game_character_talk(G, True)
        self.store.set_game_character_moves(G, "style")
        self.store.set_cast_watches_games(G, True)

    async def test_a_key_moment_schedules_the_seated_turn_then_one_spectator_turn(self):
        calls = []

        async def spy(thread, text, speaker_hint=None, spectators=False):
            calls.append((text, spectators))
        self.bot.run_table_event = spy
        await self.scene(self.three(alice=["2♥", *ALICE_HAND[1:]]))
        await self.pick_play(ALICE, "play:2♥")
        await self.settle()
        await asyncio.gather(*self.onecard.event_tasks)
        self.assertEqual([spectators for _, spectators in calls], [False, True])
        self.assertEqual(len({text for text, _ in calls}), 1)

    async def test_with_the_switch_off_there_is_no_spectator_turn(self):
        self.store.set_cast_watches_games(G, False)
        calls = []

        async def spy(thread, text, speaker_hint=None, spectators=False):
            calls.append(spectators)
        self.bot.run_table_event = spy
        await self.scene(self.three(alice=["2♥", *ALICE_HAND[1:]]))
        await self.pick_play(ALICE, "play:2♥")
        await self.settle()
        await asyncio.gather(*self.onecard.event_tasks)
        self.assertEqual(calls, [False])

    async def test_the_spectators_prompt_has_the_public_view_and_no_hand(self):
        del self.bot.run_table_event
        models = TalkModels()
        self.bot.engine.models = self.bot.models = models
        self.bot._connection.user = BOT
        hooks = {cid: FakeWebhook(60000 + 1000 * cid) for cid in (self.ann, self.bo)}

        async def webhook(channel, character):
            models.speaker = character["id"]
            return hooks[character["id"]]
        self.bot._webhook = webhook
        logging.disable(logging.WARNING)
        self.addCleanup(logging.disable, logging.NOTSET)
        self.addAsyncCleanup(lambda: drain_memory_tasks(self.bot))
        await self.scene(self.three())
        models.pick = [self.bo]
        await self.bot.run_table_event(self.thread(), "Alice played a 2.", spectators=True)
        await drain_memory_tasks(self.bot)
        (dialogue,) = models.of("dialogue")
        self.assertEqual(dialogue["speaker"], self.bo)
        self.assertIn("Game table you are watching", dialogue["text"])
        self.assertIn("One Card", dialogue["text"])
        self.assertNotIn("Your hand", dialogue["text"])
        state = self.engine_state()
        hidden = set(tokens([card for hand in state.hands for card in hand] + list(state.draw))) - set(tokens(state.discard))
        for call in models.calls:
            for card in hidden:
                self.assertNotIn(card, call["text"], f"a hidden card leaked into a {call['kind']} request")


if __name__ == "__main__":  # pragma: no cover
    import unittest
    unittest.main()
