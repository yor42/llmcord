"""FEAT-60 (D42): character seats at Yut tables (favorites, ante and payout, the ~2 s character timer, policy and model moves with fallback,
key-moment table talk, public/seated table context).

Offline: fake Discord and in-memory store from ``test_yut_discord``; the model is a fake (``bot.models.structured_compiled``); a rigged
``yut.new`` puts pieces on the board so a move phase has several choices. Timers are driven by calling their hooks, never by waiting.
Modelled on ``test_onecard_characters.py``; the pure policy is in ``test_games_yut_policy.py``, the prompt in ``test_yut_chooser.py``.
"""
import asyncio
import dataclasses
from unittest import mock

from helpers import FakeWebhook
from test_doubt_discord import says
from test_games_yut_policy import ring
from test_yut_discord import ALICE, ANTE, BOB, CH, G, YutCase

from llmcord_core.admin_store import GameError
from llmcord_core.games import yut
from llmcord_core.games_discord import NO_CHARACTER_SEATS
from llmcord_core.games_yut_discord import YutTables

REAL_NEW = yut.new


def rigged_new(rows, turn=1):
    """Two seats; ``rows`` are the seats' ring points (None = unstarted, "done"); the seat ``turn`` throws first."""
    def new(seed, seats, rules=None):
        state = REAL_NEW(seed, seats, rules)
        if len(state.seats) != 2:
            return state
        k = state.rules["pieces"]
        pieces = tuple(tuple(r) + (None,) * (k - len(r)) for r in rows)
        trails = tuple(tuple((ring(p) if p not in (None, "done", "P0") else ("P19", "P0") if p == "P0" else ()) for p in row) + ((),) * (k - len(row)) for row in rows)
        return dataclasses.replace(state, pieces=pieces, trails=trails, turn=turn)
    return new


class CharCase(YutCase):
    """Alice 0 and Ann 1 (Alice's favorite) in a channel bound to a world; Bo is in the cast but not a favorite. Talk is off until a test turns it on."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        s = self.store
        self.world = s.create_space(G, "W", "world")
        s.bind_channel(G, CH, self.world)
        self.ann = s.create_character(G, self.world, "Ann")
        self.bo = s.create_character(G, self.world, "Bo")
        for cid in (self.ann, self.bo):
            s.change_character_balance(G, cid, 100, "start", 1)
        s.add_favorite(G, ALICE, self.ann)
        s.set_cast(CH, None, [self.ann, self.bo])
        self.bot.models.structured_compiled = mock.AsyncMock(side_effect=RuntimeError("no model"))
        self.events = []

        async def spy(thread, text, speaker_hint=None, spectators=False):
            self.events.append((text, spectators))
        self.bot.run_table_event = spy
        self.tid = None

    def rig(self, rows, turn=1):
        patcher = mock.patch.object(yut, "new", rigged_new(rows, turn))
        patcher.start()
        self.addCleanup(patcher.stop)

    async def sit(self, favorites=True):
        await self.start(favorites=favorites)
        await self.push(ALICE, "Start", self.lobby())
        self.tid = self.table()["id"]

    async def fire(self):
        snap = self.store.latest_round(G, self.tid)
        await self.yut.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"])

    def seats(self):
        return [(s["kind"], s["ref_id"], s["brought_by"]) for s in self.snap()["seats"]]

    def moves(self):
        return self.log()

    def failures(self):
        return sum(self.yut.talk_failed.values())

    async def ann_to_move(self):
        """Fire Ann's timer until she holds a banked throw to move."""
        for _ in range(12):
            snap = self.snap()
            if snap["view"]["turn"] == 1 and snap["view"]["phase"] == "move":
                return snap
            self.assertEqual(snap["view"]["turn"], 1)
            await self.fire()
        self.fail("Ann never reached the move phase")


class SeatingTests(CharCase):
    async def test_favorites_seat_the_starters_favorites_with_their_ante(self):
        click = await self.start(favorites=True)
        self.assertEqual(self.seats(), [("member", ALICE, None), ("character", self.ann, ALICE)])
        self.assertEqual(self.store.character_balance(G, self.ann), 100 - ANTE)
        text = "\n".join(says(click))
        self.assertIn("Your favorites joined too", text)
        self.assertNotIn("yet", text)
        self.assertIn("Ann", self.lobby().content)
        self.assertNotIn(f"<@{self.ann}>", self.lobby().content)

    async def test_without_the_option_no_character_sits(self):
        await self.start()
        self.assertEqual(self.seats(), [("member", ALICE, None)])
        self.assertEqual(self.store.character_balance(G, self.ann), 100)

    async def test_a_broke_favorite_is_skipped_with_a_reason(self):
        self.store.change_character_balance(G, self.ann, -95, "spent", 1)
        click = await self.start(favorites=True)
        self.assertEqual(self.seats(), [("member", ALICE, None)])
        self.assertTrue(any("Not seated" in t and "Ann" in t for t in says(click)))

    async def test_at_most_four_seats_in_all(self):
        extra = [self.store.create_character(G, self.world, n) for n in ("C1", "C2", "C3")]
        for cid in (self.bo, *extra):
            self.store.change_character_balance(G, cid, 100, "start", 1)
            self.store.add_favorite(G, ALICE, cid)
        click = await self.start(favorites=True)
        self.assertEqual(len(self.snap()["seats"]), yut.MAX_SEATS)
        self.assertTrue(any("Not seated" in t for t in says(click)))

    async def test_the_starter_leaving_takes_her_characters_and_refunds_them(self):
        await self.start(favorites=True)
        await self.push(BOB, "Join", self.lobby())
        await self.push(ALICE, "Leave", self.lobby())
        self.assertEqual(self.seats(), [("member", BOB, None)])
        self.assertEqual(self.store.character_balance(G, self.ann), 100)

    async def test_an_idle_lobby_refunds_the_characters_ante(self):
        await self.start(favorites=True)
        snap = self.snap()
        await self.yut.on_lobby_timer(G, snap["table_id"], snap["round_id"])
        self.assertEqual(self.store.character_balance(G, self.ann), 100)

    async def test_alice_and_ann_alone_can_start(self):
        await self.sit()
        self.assertEqual(self.snap()["status"], "playing")


class UnboundChannelTests(YutCase):
    async def test_a_channel_without_a_world_seats_no_character_and_says_so(self):
        click = await self.start(favorites=True)
        self.assertEqual(len(self.snap()["seats"]), 1)
        self.assertTrue(any(NO_CHARACTER_SEATS in t for t in says(click)))


class TurnTests(CharCase):
    def test_the_constants(self):
        self.assertEqual((YutTables.CHARACTER_SECONDS, YutTables.MODEL_SECONDS), (2, 20))

    async def test_a_member_turn_arms_the_turn_timer_and_a_character_turn_the_character_timer(self):
        self.yut.CHARACTER_SECONDS = 3600
        self.rig([[], []], turn=0)
        await self.sit()
        key = (G, self.tid)
        self.assertIn((key, "turn"), self.yut.runtime.timers)
        self.assertNotIn((key, "character"), self.yut.runtime.timers)
        await self.throw(ALICE)
        for _ in range(8):
            if self.snap()["view"]["turn"] == 1:
                break
            snap = self.snap()
            first = next(m for m in snap["legal"][0] if m != "forfeit")
            self.yut._apply(key, snap, 0, first, "member")
        self.assertEqual(self.snap()["view"]["turn"], 1)
        self.assertIn((key, "character"), self.yut.runtime.timers)
        self.assertNotIn((key, "turn"), self.yut.runtime.timers)

    async def test_the_character_timer_uses_the_character_delay(self):
        armed = []
        real = self.yut.arm
        self.yut.arm = lambda key, kind, delay, *args: (armed.append((kind, delay)), real(key, kind, delay, *args))[1]
        self.rig([[], []], turn=1)
        await self.sit()
        self.assertIn(("character", 2), armed)

    async def test_a_character_throws_then_moves_by_the_policy_and_the_turn_comes_back(self):
        self.rig([[], []], turn=1)
        await self.sit()
        await self.fire()
        self.assertEqual(self.moves()[-1], (1, "throw", "character"))
        for _ in range(20):
            if self.snap()["view"]["turn"] == 0:
                break
            await self.fire()
            self.assertEqual(self.moves()[-1][0], 1)
            self.assertEqual(self.moves()[-1][2], "character")
        self.assertEqual(self.snap()["view"]["turn"], 0)
        self.assertEqual(self.calls(), [])
        self.assertTrue(all(actor == "character" for seat, _, actor in self.moves() if seat == 1))

    def calls(self):
        return list(self.bot.models.structured_compiled.await_args_list)

    async def test_the_policy_move_is_the_one_applied(self):
        self.rig([["P8", "P11"], ["P3", "P6"]], turn=1)
        await self.sit()
        snap = await self.ann_to_move()
        seen = self.store.seat_view(G, snap["round_id"], 1)
        style = self.store.character_play_style(G, self.ann)
        expected = yut.policy_move(seen, 1, style, self.store.policy_seed(G, snap["round_id"], f"{snap['moves']}/1/character"))
        await self.fire()
        self.assertEqual(self.moves()[-1], (1, expected, "character"))

    async def test_a_character_never_times_out_and_the_turn_timer_leaves_her_alone(self):
        self.rig([[], []], turn=1)
        await self.sit()
        snap = self.snap()
        before = len(self.moves())
        await self.yut.on_turn_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        self.assertEqual(len(self.moves()), before)
        self.assertEqual(self.snap()["view"]["turn"], 1)

    async def test_the_character_timer_does_nothing_on_a_members_turn_or_when_stale_or_closed(self):
        self.rig([[], []], turn=0)
        await self.sit()
        before = len(self.moves())
        await self.fire()
        self.assertEqual(len(self.moves()), before)
        self.store.close_table(G, self.tid)
        await self.fire()
        self.assertEqual(len(self.moves()), before)

    async def test_a_stale_character_timer_does_nothing(self):
        self.rig([[], []], turn=1)
        await self.sit()
        snap = self.snap()
        await self.yut.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"] - 1)
        self.assertEqual(self.moves(), [])

    async def test_two_timers_for_the_same_decision_apply_once(self):
        self.rig([[], []], turn=1)
        await self.sit()
        snap = self.snap()
        await asyncio.gather(*(self.yut.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"]) for _ in range(2)))
        self.assertEqual(len(self.moves()), 1)

    async def test_a_move_the_store_refuses_falls_back_to_the_first_legal_move(self):
        self.rig([["P8", "P11"], ["P3", "P6"]], turn=1)
        await self.sit()
        snap = await self.ann_to_move()
        legal = yut.choices(self.store.seat_view(G, snap["round_id"], 1), 1)
        with mock.patch.object(yut, "policy_move", return_value="move:9:gae"):
            await self.fire()
        seat, move, actor = self.moves()[-1]
        self.assertEqual((seat, actor), (1, "character"))
        self.assertEqual(move, legal[0])

    async def test_when_nothing_is_accepted_the_seat_forfeits_so_the_table_never_hangs(self):
        self.rig([[], []], turn=1)
        await self.sit()
        real = self.store.play

        def play(guild_id, round_id, seat, move, actor, expected):
            if move != "forfeit":
                raise GameError("refused")
            return real(guild_id, round_id, seat, move, actor, expected)
        self.store.play = play
        await self.fire()
        self.assertEqual(self.moves()[-1], (1, "forfeit", "timeout"))
        self.assertEqual(self.store.latest_round(G, self.tid)["status"], "settled")

    async def test_the_turn_message_has_no_throw_or_move_button_for_a_character(self):
        self.rig([[], []], turn=1)
        await self.sit()
        from test_doubt_discord import labels
        self.assertEqual(set(labels(self.turn())), {"Forfeit"})


class WinTests(CharCase):
    async def test_a_character_winning_takes_the_pot_and_the_table_talks(self):
        self.store.set_game_character_talk(G, True)
        self.rig([["P3"], ["done", "done", "done", "P0"]], turn=1)
        await self.sit()
        await self.fire()  # throws
        await self.fire()  # brings the last piece home
        snap = self.store.latest_round(G, self.tid)
        self.assertEqual((snap["status"], snap["view"]["winner"]), ("settled", 1))
        self.assertEqual(self.store.character_balance(G, self.ann), 100 - ANTE + 2 * ANTE)
        await asyncio.gather(*self.yut.event_tasks)
        self.assertIn(("Game over — Ann won the game.", False), self.events)


class ModelCase(CharCase):
    """Talk on, Character moves "model" (the default), a fake model."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.store.set_game_character_talk(G, True)
        self.answer = None
        self.asked = []

        async def model(role, request, name, schema):
            self.asked.append((name, schema["properties"]["move"]["enum"], request.messages[-1].text))
            answer = self.answer(schema["properties"]["move"]["enum"]) if callable(self.answer) else self.answer
            if isinstance(answer, BaseException):
                raise answer
            return answer
        self.bot.models.structured_compiled = model
        self.webhook = FakeWebhook(9000)
        self.bot._webhook = mock.AsyncMock(return_value=self.webhook)

    async def at_move(self):
        self.rig([["P8", "P11"], ["P3", "P6"]], turn=1)
        await self.sit()
        snap = await self.ann_to_move()
        self.asked.clear()
        return snap


class ModelTests(ModelCase):
    async def test_a_throw_needs_no_model_call(self):
        self.answer = {"move": "throw", "line": ""}
        self.rig([[], []], turn=1)
        await self.sit()
        await self.fire()
        self.assertEqual(self.asked, [])
        self.assertEqual(self.moves()[-1], (1, "throw", "character"))

    async def test_the_model_chooses_among_the_legal_moves_and_its_line_is_posted_by_her_webhook(self):
        snap = await self.at_move()
        seen = self.store.seat_view(G, snap["round_id"], 1)
        legal = yut.choices(seen, 1)
        pick = legal[-1]
        self.answer = {"move": pick, "line": "  Watch this!  "}
        await self.fire()
        self.assertEqual(self.moves()[-1], (1, pick, "character"))
        self.assertEqual(self.asked[0][1], list(legal))
        self.assertEqual(self.asked[0][0], "yut_move")
        self.assertEqual([m.content for m in self.webhook.posts], ["Watch this!"])
        self.assertEqual(self.failures(), 0)

    async def test_an_illegal_answer_falls_back_to_the_policy_and_counts_a_failure(self):
        snap = await self.at_move()
        seen = self.store.seat_view(G, snap["round_id"], 1)
        expected = yut.policy_move(seen, 1, self.store.character_play_style(G, self.ann), self.store.policy_seed(G, snap["round_id"], f"{snap['moves']}/1/character"))
        self.answer = {"move": "forfeit", "line": "bye"}
        await self.fire()
        self.assertEqual(self.moves()[-1], (1, expected, "character"))
        self.assertEqual(self.failures(), 1)
        self.assertEqual(self.webhook.posts, [])

    async def test_a_model_error_falls_back_to_the_policy_and_counts_a_failure(self):
        await self.at_move()
        self.answer = RuntimeError("down")
        await self.fire()
        self.assertEqual(self.moves()[-1][0::2], (1, "character"))
        self.assertEqual(self.failures(), 1)

    async def assert_policy_plays(self):
        snap = await self.at_move()
        self.answer = {"move": yut.choices(self.store.seat_view(G, snap["round_id"], 1), 1)[-1], "line": ""}
        await self.fire()
        self.assertEqual(self.asked, [])
        self.assertEqual(self.moves()[-1][0::2], (1, "character"))
        self.assertEqual(self.failures(), 0)

    async def test_style_mode_never_calls_the_model_even_with_talk_on(self):
        self.store.set_game_character_moves(G, "style")
        await self.assert_policy_plays()

    async def test_talk_off_plays_the_policy_without_a_model_call(self):
        self.store.set_game_character_talk(G, False)
        await self.assert_policy_plays()

    async def test_at_the_daily_limit_no_model_call_is_made(self):
        self.store.set_game_talk_daily_limit(G, 1)
        self.yut.talk.failed[self.yut.talk.marker(G)] = 1
        await self.at_move()
        await self.fire()
        self.assertEqual(self.asked, [])

    async def test_a_play_style_lookup_error_falls_through_to_the_policy(self):
        await self.at_move()
        self.answer = RuntimeError("down")
        real = self.store.character_play_style
        calls = []

        def style(guild_id, cid):
            calls.append(cid)
            if len(calls) == 1:
                raise RuntimeError("store")
            return real(guild_id, cid)
        self.store.character_play_style = style
        await self.fire()
        self.assertEqual(self.moves()[-1][0::2], (1, "character"))

    async def test_an_answer_that_arrives_after_the_table_moved_on_is_dropped_and_counted(self):
        snap = await self.at_move()
        legal = yut.choices(self.store.seat_view(G, snap["round_id"], 1), 1)

        def answer(enum):
            self.store.play(G, snap["round_id"], 0, "forfeit", "member", self.snap()["moves"])  # Alice leaves while the model thinks
            return {"move": legal[0], "line": "late"}
        self.answer = answer
        before = len(self.moves())
        await self.fire()
        self.assertEqual(len(self.moves()), before + 1)  # only Alice's forfeit
        self.assertEqual(self.failures(), 1)
        self.assertEqual(self.webhook.posts, [])

    async def test_no_discord_or_model_call_is_made_under_the_table_lock(self):
        await self.at_move()
        key = (G, self.tid)
        seen = []
        original = self.bot.models.structured_compiled

        async def spy(*args):
            seen.append(self.yut.runtime.lock(key).locked())
            return await original(*args)
        self.bot.models.structured_compiled = spy
        self.answer = {"move": "forfeit", "line": ""}
        await self.fire()
        self.assertEqual(seen, [False])


class TalkTests(CharCase):
    def snap_of(self, before_history, events, names=("Alice", "Ann")):
        seats = [{"kind": "member", "ref_id": ALICE, "name": None}, {"kind": "character", "ref_id": self.ann, "name": "Ann"}]
        return {"seats": seats, "view": {"history": before_history}}, {"seats": seats, "view": {"history": before_history + events}}

    def test_key_moments_are_a_capture_a_piece_home_and_a_yut_or_mo(self):
        key = (G, 1)
        throw = lambda name: {"kind": "throw", "seat": 1, "name": name, "sticks": (True,) * 4}
        move = lambda **kw: {"kind": "move", "seat": 1, "pieces": (0,), "throw": "gae", "start": "P1", "end": "P3", "captured": (), **kw}
        cases = [([throw("yut")], "Ann threw Yut and gets another throw."), ([throw("mo")], "Ann threw Mo and gets another throw."),
                 ([move(captured=((0, 1),))], "Ann captured a piece of Player 1."), ([move(end="done")], "Ann brought a piece home."),
                 ([throw("gae")], None), ([move()], None)]
        for events, expected in cases:
            before, new = self.snap_of([], events)
            text = self.yut._moment(key, before, new)
            self.assertEqual(text, expected, events)

    async def test_the_table_talk_context_lists_the_seated_characters(self):
        await self.sit()
        context = self.yut.table_talk_context(G, self.table()["thread_id"])
        self.assertEqual(context["seated_character_ids"], [self.ann])
        self.assertEqual(self.bot._table_fields(context)["allowed_ids"], (self.ann,))

    async def test_a_seated_character_gets_the_public_table_and_nobody_else_does(self):
        await self.sit()
        thread = self.table()["thread_id"]
        block = self.yut.table_block(G, thread, self.ann)
        self.assertIn("You are sitting at a table of Yut", block)
        self.assertIn("Ann (you)", block)
        self.assertIn("Player 1", block)
        self.assertIsNone(self.yut.table_block(G, thread, self.bo))
        public = self.yut.public_block(G, (G, self.tid))
        self.assertNotIn("(you)", public)
        self.assertNotIn("Seed", block)
        self.assertEqual(self.bot._table_block(G, thread, self.ann), block)
