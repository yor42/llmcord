"""FEAT-21 part C: table talk in the I Doubt It table thread.

TDD-red tests written before the implementation. Offline: fake Discord (``helpers``), the doubt table fixtures of
``test_doubt_discord``, and a recording model gateway that sees every director, dialogue and memory request of a turn.
Timers are never waited for; the DoubtTables clock is a settable fake.

Assumed API (proposed, pinned here; the implementer creates it):

  ``SceneContext.allowed_ids`` (tuple of character ids, ``None`` = unrestricted, the default): when set, ``Engine.speakers``
    and ``eligible`` offer only those characters (no favorites extras, no step-in, no archived favorites).
  ``SceneContext.usage_feature`` (``None`` = the old ambient/summon/reply choice): ``SkitBot.run_scene`` wraps the turn in
    ``capture_usage(..., usage_feature)`` when it is set, so the director, the dialogue and the memory calls of a table-thread
    turn are attributed to feature ``'game'`` (``log_attribution()['feature']``) and count toward the daily table-talk limit.
  ``DoubtTables.table_talk_context(guild_id, thread_id)`` (sync) -> ``None`` or a dict with keys ``table_id``, ``round_id``,
    ``seated_character_ids`` (list of character ids seated at the table's latest round, in seat order) and ``status`` (the
    round status: ``'joining'``, ``'playing'`` or ``'settled'``). Guild-scoped. Covers joining, playing and the post-game banter
    until the archive timer ran; ``None`` for every other thread (also after the archive timer, and for another guild).
  ``DoubtTables.table_block(guild_id, thread_id, character_id)`` (sync) -> ``None`` (not a table thread, or the character is not
    seated) or the per-speaker text: game name, forced rank, every seat's name and card count, the pile size, recent claims and
    reveals, whose turn it is, and ``Your hand: <rank><suit>, ...`` (``7♠`` style tokens) for that speaker only. Once the round is
    over there is no hand line.
  The block is injected into the dialogue prompt of that speaker whatever the server's preset says; it is never part of the
    director request, the saved nodes, the memory jobs or the catch-up.
  ``DoubtTables.EVENT_COOLDOWN_SECONDS = 20`` (measured on ``DoubtTables.clock``): at most one event turn per table per cooldown.
  ``bot.run_table_event(thread, text, speaker_hint=None)`` (coroutine): scheduled (fire-and-forget) by DoubtTables on a doubt's
    reveal, on a seat reaching one card and on the game's end, with public text only. It skips (never waits) when the thread's
    channel lock is busy, applies the same gating as a member message, and runs one ambient director turn restricted to the
    seated characters with the event text in the speaker's prompt.
  Gating of a table-thread turn: ``game_character_talk`` on, and ``game_talk_calls_today + DoubtTables.talk_failed`` below
    ``game_talk_daily_limit``, and the usual spending hard cap; otherwise the characters stay silent (no model call, no status
    message in the thread). Members' messages are still counted by ``note_thread_message``.
  After the archive timer the thread behaves like any normal thread (the whole cast, feature ``'reply'``, no block).
"""
import asyncio
from unittest import mock
import re
import time
import json
import logging
from types import SimpleNamespace

import test_doubt_discord as base
from helpers import CompiledAdapter, FakeGameClick, FakeWebhook, drain_memory_tasks, invoke, thread_message
from test_doubt_chooser import five_seed
from test_doubt_discord import ALICE, BOB, CARA, CH, G

from llmcord_core.games import doubt
from llmcord_core.games_doubt_discord import DoubtButton
from llmcord_core.prompts import default_bundle
from llmcord_core.usage import log_attribution

NAMES = {ALICE: "Alice", BOB: "Bob", CARA: "Cara", 9: "Zed"}
BOT = SimpleNamespace(id=999, bot=True, mention="<@999>", display_name="Bot")
COUNT_LINE = re.compile(r"\b(?:10|[2-9AJQK])\s*x\s*\d")
NO_MODEL = {"extract_memory"}


def tokens(cards):
    return [f"{rank}{suit}" for rank, suit in cards]


class TalkModels(CompiledAdapter):
    """Records every request (kind director / dialogue / extract_memory / summary / other) with the usage feature that was
    active, the speaker (set by the webhook lookup that precedes each dialogue call) and whether a table lock was held."""

    def __init__(self, case):
        self.case, self.calls, self.pick, self.gate, self.entered = case, [], [], None, asyncio.Event()
        self.speaker = None

    def record(self, kind, request, role):
        text = "\n".join(m.text for m in request.messages)
        entry = {"kind": kind, "role": role, "text": text, "feature": log_attribution()["feature"], "speaker": self.speaker,
                 "table_locked": any(lock.locked() for lock in self.case.doubt.runtime.locks.values()), "request": request}
        self.calls.append(entry)
        return entry

    async def structured_compiled(self, role, request, name, schema):
        if name == "choose_speakers":
            entry = self.record("director", request, role)
            user = next(m.text for m in request.messages if m.role != "system")
            entry["payload"] = json.loads(user)
            return {"speakers": list(self.pick)}
        if name == "extract_memory":
            self.record("extract_memory", request, role)
            return {"shared_facts": [], "personal_facts": [], "encounter_facts": []}
        self.record("other:" + name, request, role)
        raise RuntimeError("no character-move model in table talk tests")  # the chooser falls back to its policy

    async def text_compiled(self, role, request, max_tokens=None):
        self.record("summary", request, role)
        return "Earlier scene summary"

    async def stream_compiled(self, role, request):
        self.record("dialogue", request, role)
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        yield "<emotion>neutral</emotion>\nHello."

    async def close(self):
        pass

    def of(self, kind):
        return [c for c in self.calls if c["kind"] == kind]


class TalkCase(base.DoubtCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        s = self.store
        s.set_game_character_talk(G, True)
        self.dee = s.create_character(G, self.world, "Dee")  # in the cast, never seated at the tables below
        s.set_cast(CH, None, [self.ann, self.bo, self.cy, self.dee])
        s.change_character_balance(G, self.dee, 100, "start", 1)
        self.models = TalkModels(self)
        self.bot.engine.models = self.bot.models = self.models
        self.bot._connection.user = BOT
        self.hooks = {cid: FakeWebhook(60000 + 1000 * cid) for cid in (self.ann, self.bo, self.cy, self.dee)}

        async def webhook(channel, character):
            self.models.speaker = character["id"]
            return self.hooks[character["id"]]
        self.bot._webhook = webhook
        logging.disable(logging.WARNING)  # the move model is absent on purpose; the chooser logs each fallback
        self.addCleanup(logging.disable, logging.NOTSET)
        self.now = [10_000.0]
        self.doubt.clock = lambda: self.now[0]

        async def drain():
            await drain_memory_tasks(self.bot)
        self.addAsyncCleanup(drain)

    def thread(self):
        """The table thread; remembered while the table is open, since a finished game closes the table row."""
        if self.table() is not None:
            self._table_thread = super().thread()
        return self._table_thread

    # --- tables ------------------------------------------------------------------------------

    async def table3(self, seed=None):
        """Alice 0, Ann 1 (Alice's favorite), Bob 2; Bo, Cy and Dee are in the cast but not seated."""
        if seed:
            self.seed = seed
        await self.start(favorites=True)
        await self.push(BOB, "Join", self.lobby())
        await self.push(ALICE, "Start", self.lobby())
        self.thread()

    async def table5(self):
        """Alice 0, Ann 1, Bob 2, Bo 3, Cy 4; Dee is in the cast but not seated."""
        self.seed = five_seed()
        await self.start(favorites=True)
        await self.push(BOB, "Join", self.lobby())
        await self.push(ALICE, "Fill seats", self.lobby())
        await self.push(ALICE, "Fill seats", self.lobby())
        await self.push(ALICE, "Start", self.lobby())
        self.thread()

    async def fire(self):
        snap = self.snap()
        await self.doubt.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"])

    def seat_of(self, character_id):
        return next(s["index"] for s in self.snap()["seats"] if s["kind"] == "character" and s["ref_id"] == character_id)

    def seated(self):
        return [s["ref_id"] for s in self.snap()["seats"] if s["kind"] == "character"]

    def cards_of(self, seat):
        return set(tokens(self.hand(seat)))

    def others_cards(self, seat):
        """Every card token held by another seat and not by this one (the hidden information for this speaker)."""
        n = len(self.snap()["seats"])
        mine = self.cards_of(seat)
        return {t for i in range(n) if i != seat for t in self.cards_of(i)} - mine

    # --- driving the bot ---------------------------------------------------------------------

    @staticmethod
    def silent_history(thread):
        async def history(**kwargs):
            return
            yield  # an empty async generator
        thread.history = history

    async def say(self, user=ALICE, text="Hello there", *, thread=None, mention=True, drain=True):
        thread = thread or self.thread()
        self.silent_history(thread)
        message = thread_message(thread, user, ("<@999> " if mention else "") + text)
        message.author.display_name = NAMES.get(user, f"User {user}")
        message.mentions = [BOT] if mention else []
        await self.bot.on_message(message)
        if drain:
            await drain_memory_tasks(self.bot)
        return message

    def spoke(self):
        return [cid for cid, hook in self.hooks.items() if hook.posts]

    def announcements(self, thread=None):
        return [m.content for m in (thread or self.thread()).sent if not m.deleted and m.content]

    def dialogue_of(self, character_id):
        return next(c for c in self.models.of("dialogue") if c["speaker"] == character_id)

    def assert_no_hidden(self, call, seat):
        for card in self.others_cards(seat):
            self.assertNotIn(card, call["text"], f"{card} of another seat leaked into a {call['kind']} request")
        for name in {"Ann", "Bo", "Cy", "Alice", "Bob", "Player 1", "Player 2"}:
            for line in call["text"].splitlines():
                if name in line and COUNT_LINE.search(line) and "Your hand:" not in line:
                    self.fail(f"a hand-count line about {name} in a {call['kind']} request: {line!r}")

    # --- spending ----------------------------------------------------------------------------

    def add_usage(self, count):
        for _ in range(count):
            self.store.db.execute("INSERT INTO model_usage(guild_id,profile,model,role,cached_tokens,reasoning_tokens,cost_basis,created_at,feature) VALUES(?,?,?,?,0,0,?,?,?)",
                                  (G, "p", "m", "director", "none", time.time(), "game"))
        self.store.db.commit()

    def hard_cap(self):
        from datetime import datetime, timezone
        self.store.save_budget(None, 1.0, 1, False, self.store.budget_settings()["revision"])
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self.store.db:
            self.store.db.execute("INSERT INTO spend_days(day,cost_usd,unpriced_calls) VALUES(?,?,?) ON CONFLICT(day) DO UPDATE SET cost_usd=cost_usd+excluded.cost_usd", (day, 5.0, 0))

    # --- finishing a game --------------------------------------------------------------------

    async def finish_by_forfeits(self):
        """Table3 to its end: Ann forfeits, Alice misses two turns, so Bob wins; the last move goes through the bot's timer hooks."""
        snap = self.snap()
        tid, rid = snap["table_id"], snap["round_id"]
        self.store_play(1, "forfeit", "timeout")
        await self.doubt.on_turn_timer(G, tid, rid, self.snap()["moves"])  # Alice's first missed turn plays a card
        self.assertEqual(self.snap()["status"], "playing")
        await self.doubt.on_window_timer(G, tid, rid, self.snap()["moves"])
        seat = self.snap()["view"]["turn"]
        self.assertEqual(seat, 2)
        self.store_play(2, f"play:{self.hand(2)[0][0]}x1")
        await self.doubt.on_window_timer(G, tid, rid, self.snap()["moves"])
        await self.doubt.on_turn_timer(G, tid, rid, self.snap()["moves"])
        self.assertEqual(self.store.latest_round(G, tid)["status"], "settled")
        await asyncio.gather(*self.doubt.event_tasks)  # the game-end event turn the table scheduled; later asserts count their own calls
        await drain_memory_tasks(self.bot)
        self.models.calls.clear()


class ContextTests(TalkCase):
    async def test_a_playing_table_thread_reports_its_seated_characters(self):
        await self.table3()
        ctx = self.doubt.table_talk_context(G, self.thread().id)
        snap = self.snap()
        self.assertEqual(ctx["table_id"], snap["table_id"])
        self.assertEqual(ctx["round_id"], snap["round_id"])
        self.assertEqual(ctx["seated_character_ids"], [self.ann])
        self.assertEqual(ctx["status"], "playing")

    async def test_a_joining_lobby_thread_already_reports_its_seated_characters(self):
        await self.start(favorites=True)
        ctx = self.doubt.table_talk_context(G, self.thread().id)
        self.assertEqual((ctx["seated_character_ids"], ctx["status"]), ([self.ann], "joining"))

    async def test_every_character_seat_is_listed_in_seat_order(self):
        await self.table5()
        ctx = self.doubt.table_talk_context(G, self.thread().id)
        self.assertEqual(ctx["seated_character_ids"], [self.ann, self.bo, self.cy])

    async def test_other_threads_other_guilds_and_unknown_ids_have_no_context(self):
        await self.table3()
        plain = await self.ch.make_thread("plain chat")
        self.assertIsNone(self.doubt.table_talk_context(G, plain.id))
        self.assertIsNone(self.doubt.table_talk_context(G + 1, self.thread().id))
        self.assertIsNone(self.doubt.table_talk_context(G, 424242))

    async def test_the_banter_after_the_game_keeps_the_context_until_the_archive_timer(self):
        """Characterization (FEAT-21 C): after the game the closed table's thread still talks until it is archived."""
        await self.table3()
        thread, tid = self.thread(), self.table()["id"]
        await self.finish_by_forfeits()
        ctx = self.doubt.table_talk_context(G, thread.id)
        self.assertEqual((ctx["seated_character_ids"], ctx["status"]), ([self.ann], "settled"))
        await self.doubt.on_archive_timer(G, tid)
        self.assertIsNone(self.doubt.table_talk_context(G, thread.id))

    async def test_a_bot_restart_recovery_leaves_no_context(self):
        await self.table3()
        thread = self.thread()
        await self.doubt.recover()
        self.assertIsNone(self.doubt.table_talk_context(G, thread.id))


class BlockTests(TalkCase):
    async def test_the_block_holds_public_state_and_only_the_speakers_own_hand(self):
        await self.table3()
        block = self.doubt.table_block(G, self.thread().id, self.ann)
        view = self.snap()["view"]
        self.assertIn("I Doubt It", block)
        self.assertIn(view["rank"], block)
        self.assertIn("Ann", block)
        self.assertRegex(block, rf"\b{view['pile']} cards?\b")
        for count in set(view["counts"]):
            self.assertRegex(block, rf"\b{count} cards?\b")
        self.assertEqual(block.count("Your hand:"), 1)
        for card in self.cards_of(1):
            self.assertIn(card, block)
        for card in self.others_cards(1):
            self.assertNotIn(card, block)

    async def test_two_seated_characters_get_their_own_blocks(self):
        await self.table5()
        thread = self.thread()
        blocks = {cid: self.doubt.table_block(G, thread.id, cid) for cid in (self.ann, self.bo, self.cy)}
        for cid, block in blocks.items():
            seat = self.seat_of(cid)
            for card in self.cards_of(seat):
                self.assertIn(card, block)
            for card in self.others_cards(seat):
                self.assertNotIn(card, block)
            self.assertNotIn("<@", block)

    async def test_the_block_is_none_for_a_character_that_is_not_seated_and_for_other_threads(self):
        await self.table3()
        thread = self.thread()
        for cid in (self.bo, self.cy, self.dee):
            self.assertIsNone(self.doubt.table_block(G, thread.id, cid))
        plain = await self.ch.make_thread("plain chat")
        self.assertIsNone(self.doubt.table_block(G, plain.id, self.ann))
        self.assertIsNone(self.doubt.table_block(G + 1, thread.id, self.ann))

    async def test_the_block_follows_the_public_play_and_a_reveal_shows_only_the_revealed_cards(self):
        await self.table3()
        snap = self.snap()
        claimed = self.any_card(0)
        await self.pick_play(ALICE, claimed)
        block = self.doubt.table_block(G, self.thread().id, self.ann)
        self.assertRegex(block, rf"\b{self.snap()['view']['pile']} cards?\b")
        await self.doubt_press(BOB)
        reveal = [h for h in self.snap()["view"]["history"] if h["kind"] == "reveal"][-1]
        block = self.doubt.table_block(G, self.thread().id, self.ann)
        for card in tokens(reveal["cards"]):
            self.assertIn(card, block)
        revealed = set(tokens(reveal["cards"]))
        for card in self.others_cards(1) - revealed:
            self.assertNotIn(card, block)
        self.assertEqual(snap["table_id"], self.snap()["table_id"])

    async def test_the_block_after_the_game_has_no_hand_line(self):
        await self.table3()
        thread = self.thread()
        await self.finish_by_forfeits()
        block = self.doubt.table_block(G, thread.id, self.ann)
        self.assertTrue(block is None or "Your hand:" not in block)


class SpeakerRestrictionTests(TalkCase):
    async def test_only_the_seated_character_is_offered_to_the_director_and_speaks(self):
        await self.table3()
        self.models.pick = [self.dee, self.bo, self.ann, self.cy]  # a director gone wrong still cannot add a stranger
        await self.say()
        (director,) = self.models.of("director")
        self.assertEqual([option["id"] for option in director["payload"]["cast"]], [self.ann])
        self.assertEqual(self.spoke(), [self.ann])

    async def test_a_cast_member_that_is_not_seated_never_speaks_even_when_named(self):
        await self.table3()
        self.models.pick = [self.bo]
        await self.say(text="Bo, what do you think?")
        self.assertNotIn(self.bo, self.spoke())
        self.assertTrue(all(option["id"] in (self.ann,) for c in self.models.of("director") for option in c["payload"]["cast"]))

    async def test_a_favorite_that_could_step_in_is_ignored_at_a_table(self):
        await self.table3()
        self.store.add_favorite(G, ALICE, self.dee)
        self.store.set_favorites_mode(G, ALICE, "step_in")
        self.models.pick = [self.dee, self.ann]
        await self.say()
        (director,) = self.models.of("director")
        self.assertEqual([option["id"] for option in director["payload"]["cast"]], [self.ann])
        self.assertEqual(self.spoke(), [self.ann])

    async def test_an_archived_favorite_does_not_step_in_at_a_table(self):
        await self.table3()
        self.store.add_favorite(G, ALICE, self.dee)
        self.store.db.execute("UPDATE characters SET archived=1 WHERE id=?", (self.dee,))
        self.store.db.commit()
        self.models.pick = [self.dee, self.ann]
        await self.say()
        self.assertNotIn(self.dee, self.spoke())
        (director,) = self.models.of("director")
        self.assertEqual([option["id"] for option in director["payload"]["cast"]], [self.ann])

    async def test_all_seated_characters_may_speak_and_no_one_else(self):
        await self.table5()
        self.models.pick = [self.dee, self.ann, self.bo, self.cy]
        await self.say()
        (director,) = self.models.of("director")
        self.assertEqual({option["id"] for option in director["payload"]["cast"]}, {self.ann, self.bo, self.cy})
        self.assertEqual(sorted(self.spoke()), sorted([self.ann, self.bo, self.cy]))
        self.assertEqual(self.hooks[self.dee].posts, [])

    async def test_a_message_without_an_invitation_follows_the_parents_ambient_rule(self):
        await self.table3()
        self.models.pick = [self.ann]
        await self.say(text="just chatting", mention=False)
        self.assertEqual(self.models.calls, [])  # the parent is not ambient: no mention, no reply, no turn

    async def test_with_an_ambient_parent_an_invited_message_gets_a_restricted_turn(self):
        await self.table3()
        self.store.set_ambient(CH, True)
        self.models.pick = [self.dee, self.ann]
        await self.say(text="Ann, what do you think?", mention=False)
        self.assertEqual(self.spoke(), [self.ann])
        (director,) = self.models.of("director")
        self.assertEqual([option["id"] for option in director["payload"]["cast"]], [self.ann])

    async def test_summoning_a_character_that_is_not_seated_is_refused_privately(self):
        await self.table3()
        thread = self.thread()
        self.silent_history(thread)
        click = FakeGameClick(self.bot, ALICE, channel=thread)
        await invoke(self.bot, "summon", click, character="Dee", prompt="Join us")
        self.assertEqual(self.models.calls, [])
        self.assertEqual(self.spoke(), [])
        replies = click.response.sent + click.followup.sent
        self.assertEqual(len(replies), 1)
        text, kwargs = replies[0]
        self.assertTrue(kwargs.get("ephemeral"))
        self.assertLess(len(text), 200)

    async def test_summoning_a_seated_character_works_and_counts_as_table_talk(self):
        await self.table3()
        thread = self.thread()
        self.silent_history(thread)
        click = FakeGameClick(self.bot, ALICE, channel=thread)
        await invoke(self.bot, "summon", click, character="Ann", prompt="Join us")
        await drain_memory_tasks(self.bot)
        self.assertEqual(self.spoke(), [self.ann])
        self.assertTrue(self.models.calls)
        self.assertEqual({c["feature"] for c in self.models.calls if c["kind"] in ("director", "dialogue")}, {"game"})


class HiddenInformationTests(TalkCase):
    async def full_turn(self):
        await self.table5()
        self.models.pick = [self.ann, self.bo, self.cy]
        await self.say()

    async def test_each_speaker_prompt_holds_its_own_hand_and_nobody_elses(self):
        await self.full_turn()
        dialogues = self.models.of("dialogue")
        self.assertEqual([c["speaker"] for c in dialogues], [self.ann, self.bo, self.cy])
        for call in dialogues:
            seat = self.seat_of(call["speaker"])
            self.assertEqual(call["text"].count("Your hand:"), 1)
            for card in self.cards_of(seat):
                self.assertIn(card, call["text"])
            self.assert_no_hidden(call, seat)

    async def test_the_block_is_in_the_speakers_prompt_exactly_as_the_table_gives_it(self):
        await self.full_turn()
        for call in self.models.of("dialogue"):
            block = self.doubt.table_block(G, self.thread().id, call["speaker"])
            self.assertIn(block, call["text"])

    async def test_no_request_of_the_turn_holds_another_seats_cards(self):
        await self.full_turn()
        self.assertTrue(self.models.calls)
        for call in self.models.calls:
            if call["kind"] == "dialogue":
                continue
            for card in set().union(*(self.cards_of(i) for i in range(5))):
                self.assertNotIn(card, call["text"], f"{card} in a {call['kind']} request")
            self.assertNotIn("Your hand:", call["text"])

    async def test_the_director_request_has_no_table_block(self):
        await self.full_turn()
        (director,) = self.models.of("director")
        for cid in (self.ann, self.bo, self.cy):
            self.assertNotIn(self.doubt.table_block(G, self.thread().id, cid), director["text"])
        self.assertNotIn("Your hand", director["text"])

    async def test_nothing_of_the_block_is_saved_in_the_scene_or_posted(self):
        await self.full_turn()
        rows = self.store.all("SELECT * FROM nodes")
        self.assertTrue(rows)
        saved = "\n".join(str(value) for row in rows for value in tuple(row))
        every_card = set().union(*(self.cards_of(i) for i in range(5)))
        for card in every_card:
            self.assertNotIn(card, saved)
        self.assertNotIn("Your hand", saved)
        posted = "\n".join(m.content for hook in self.hooks.values() for m in hook.posts)
        for card in every_card:
            self.assertNotIn(card, posted)
        for table in (self.store.all("SELECT * FROM node_summaries") if self.store.one("SELECT name FROM sqlite_master WHERE name='node_summaries'") else []):
            self.assertNotIn("Your hand", str(tuple(table)))

    async def test_the_second_turn_does_not_carry_the_first_turns_hand_into_history(self):
        await self.table3()
        self.models.pick = [self.ann]
        first = await self.say()
        before = self.cards_of(1)
        self.store_play(0, "play:" + f"{self.hand(0)[0][0]}x1")  # the table moves on; Ann's hand is unchanged
        await self.say(text="and now?")
        for call in self.models.of("dialogue"):
            self.assertEqual(call["text"].count("Your hand:"), 1)
        self.assertEqual(before, self.cards_of(1))
        self.assertIsNotNone(first)

    async def test_the_block_reaches_the_prompt_whatever_the_servers_preset_says(self):
        await self.table3()
        bundle = default_bundle()
        bundle["purposes"]["dialogue"][0]["content"] = "Custom instructions for {{user}}."
        ident, revision = self.store.save_preset(G, "Bare", bundle)
        self.store.activate_preset(G, ident, revision, {"dialogue": "compatible"})
        self.models.pick = [self.ann]
        await self.say()
        (call,) = self.models.of("dialogue")
        self.assertIn("Custom instructions", call["text"])
        self.assertIn(self.doubt.table_block(G, self.thread().id, self.ann), call["text"])

    async def test_a_reveal_shows_the_revealed_cards_but_still_no_other_hand(self):
        await self.table3()
        await self.pick_play(ALICE, self.any_card(0))
        await self.doubt_press(BOB)
        reveal = [h for h in self.snap()["view"]["history"] if h["kind"] == "reveal"][-1]
        self.models.pick = [self.ann]
        await self.say()
        (call,) = self.models.of("dialogue")
        for card in tokens(reveal["cards"]):
            self.assertIn(card, call["text"])
        for card in self.others_cards(1) - set(tokens(reveal["cards"])):
            self.assertNotIn(card, call["text"])


class GatingTests(TalkCase):
    async def silent(self, note):
        before = self.announcements()
        key = (G, self.table()["id"])
        talk = self.doubt.table(key).talk
        await self.say()
        self.assertEqual(self.models.calls, [], note)
        self.assertEqual(self.spoke(), [], note)
        self.assertEqual(self.announcements(), before, note)  # no status line, no "No active character", no notice
        self.assertEqual(self.doubt.table(key).talk, talk + 1, "members' chat is still counted")

    async def test_with_character_table_talk_off_the_characters_stay_silent(self):
        await self.table3()
        self.store.set_game_character_talk(G, False)
        self.models.pick = [self.ann]
        await self.silent("talk off")

    async def test_at_the_daily_limit_the_characters_stay_silent(self):
        await self.table3()
        self.store.set_game_talk_daily_limit(G, 3)
        self.add_usage(3)
        self.models.pick = [self.ann]
        await self.silent("limit reached")

    async def test_failed_move_attempts_count_toward_the_limit(self):
        await self.table3()
        self.store.set_game_talk_daily_limit(G, 3)
        self.add_usage(1)
        marker = (G, self.store._server_day(G, time.time())[1])
        self.doubt.talk_failed[marker] = 2
        self.models.pick = [self.ann]
        await self.silent("1 used + 2 failed of 3")

    async def test_below_the_limit_the_characters_talk(self):
        await self.table3()
        self.store.set_game_talk_daily_limit(G, 3)
        self.add_usage(2)
        self.models.pick = [self.ann]
        await self.say()
        self.assertEqual(self.spoke(), [self.ann])

    async def test_the_spending_hard_cap_keeps_the_characters_silent(self):
        await self.table3()
        self.hard_cap()
        self.models.pick = [self.ann]
        await self.say()
        self.assertEqual(self.models.calls, [])
        self.assertEqual(self.spoke(), [])

    async def test_a_silent_table_still_plays(self):
        await self.table3()
        self.store.set_game_character_talk(G, False)
        await self.say()
        await self.pick_play(ALICE, self.any_card(0))
        self.assertEqual(self.moves()[-1][0], 0)


class UsageTests(TalkCase):
    async def test_every_model_call_of_a_table_turn_counts_as_table_talk(self):
        await self.table3()
        self.models.pick = [self.ann]
        await self.say()
        turn = [c for c in self.models.calls if c["kind"] in ("director", "dialogue")]
        self.assertEqual([c["kind"] for c in turn], ["director", "dialogue"])
        self.assertEqual({c["feature"] for c in turn}, {"game"})

    async def test_an_ambient_table_turn_counts_as_table_talk_too(self):
        await self.table3()
        self.store.set_ambient(CH, True)
        self.models.pick = [self.ann]
        await self.say(text="Ann, what do you think?", mention=False)
        self.assertEqual({c["feature"] for c in self.models.calls if c["kind"] in ("director", "dialogue")}, {"game"})

    async def test_table_talk_never_runs_under_the_table_lock(self):
        await self.table3()
        self.models.pick = [self.ann]
        await self.say()
        self.assertTrue(self.models.calls)
        self.assertFalse(any(c["table_locked"] for c in self.models.calls))

    async def test_a_move_does_not_wait_for_a_talk_turn_in_flight(self):
        await self.table3()
        self.models.pick = [self.ann]
        self.models.gate = asyncio.Event()
        task = asyncio.create_task(self.say())
        try:
            await asyncio.wait_for(self.models.entered.wait(), 1)
            await asyncio.wait_for(self.pick_play(ALICE, self.any_card(0)), 1)
            self.assertEqual(self.moves()[-1][0], 0)
        finally:
            self.models.gate.set()
            await task


class NormalThreadTests(TalkCase):
    async def test_a_thread_that_is_not_a_table_keeps_the_whole_cast_and_the_old_usage_label(self):
        """Regression (FEAT-21 C): every non-table location is unchanged."""
        await self.table3()
        self.store.set_game_character_talk(G, False)  # table talk settings do not touch ordinary threads
        self.store.set_game_talk_daily_limit(G, 1)
        self.add_usage(5)
        thread = await self.ch.make_thread("plain chat")
        self.models.pick = [self.dee, self.bo]
        await self.say(thread=thread)
        (director,) = self.models.of("director")
        self.assertEqual({option["id"] for option in director["payload"]["cast"]}, {self.ann, self.bo, self.cy, self.dee})
        self.assertEqual(sorted(self.spoke()), sorted([self.dee, self.bo]))
        for call in self.models.of("dialogue"):
            self.assertNotIn("Your hand:", call["text"])
            self.assertNotIn("I Doubt It", call["text"])
        self.assertEqual({c["feature"] for c in self.models.calls if c["kind"] in ("director", "dialogue")}, {"reply"})

    async def test_the_table_thread_next_to_it_stays_restricted(self):
        await self.table3()
        thread = await self.ch.make_thread("plain chat")
        self.models.pick = [self.dee]
        await self.say(thread=thread)
        self.assertEqual(self.spoke(), [self.dee])
        self.hooks[self.dee].posts.clear()
        await self.say()
        self.assertEqual(self.hooks[self.dee].posts, [])


class AfterTheTableTests(TalkCase):
    async def assert_normal_thread(self):
        self.models.pick = [self.dee, self.bo]
        await self.say()
        (director,) = self.models.of("director")
        self.assertEqual({option["id"] for option in director["payload"]["cast"]}, {self.ann, self.bo, self.cy, self.dee})
        self.assertEqual(sorted(self.spoke()), sorted([self.dee, self.bo]))
        for call in self.models.of("dialogue"):
            self.assertNotIn("Your hand:", call["text"])
        self.assertEqual({c["feature"] for c in self.models.calls if c["kind"] in ("director", "dialogue")}, {"reply"})

    async def test_after_the_archive_timer_the_thread_is_a_normal_thread(self):
        """Characterization (FEAT-21 C): once the table is archived its thread talks like any other thread of the channel."""
        await self.table3()
        tid = self.table()["id"]
        await self.finish_by_forfeits()
        await self.doubt.on_archive_timer(G, tid)
        await self.assert_normal_thread()

    async def test_after_a_restart_recovery_the_thread_is_a_normal_thread(self):
        """Characterization (FEAT-21 C): a table forgotten by recovery no longer restricts its thread."""
        await self.table3()
        await self.doubt.recover()
        await self.assert_normal_thread()

    async def test_the_banter_after_the_game_is_restricted_and_has_no_hand(self):
        await self.table3()
        await self.finish_by_forfeits()
        self.models.pick = [self.dee, self.ann]
        await self.say()
        (director,) = self.models.of("director")
        self.assertEqual([option["id"] for option in director["payload"]["cast"]], [self.ann])
        self.assertEqual(self.spoke(), [self.ann])
        (call,) = self.models.of("dialogue")
        self.assertNotIn("Your hand:", call["text"])
        self.assertEqual(call["feature"], "game")

    async def test_the_joining_lobby_thread_is_restricted_and_has_no_hand_yet(self):
        await self.start(favorites=True)
        self.models.pick = [self.dee, self.ann]
        await self.say()
        self.assertEqual(self.spoke(), [self.ann])
        (call,) = self.models.of("dialogue")
        self.assertNotIn("Your hand:", call["text"])


class EventTurnTests(TalkCase):
    EVENT = "Doubt! <@5> doubted <@6>'s one 3. They were A♥ — <@6> takes the pile (1 card)."

    async def run_event(self, thread=None, text=None):
        await self.bot.run_table_event(thread or self.thread(), text or self.EVENT)
        await drain_memory_tasks(self.bot)

    async def test_an_event_runs_one_ambient_director_turn_restricted_to_the_seated_characters(self):
        await self.table3()
        self.models.pick = [self.dee, self.ann]
        await self.run_event()
        (director,) = self.models.of("director")  # the event text names no one: it must still reach the director
        self.assertEqual([option["id"] for option in director["payload"]["cast"]], [self.ann])
        self.assertTrue(director["payload"]["ambient"])
        self.assertEqual(self.spoke(), [self.ann])
        (call,) = self.models.of("dialogue")
        self.assertIn("one 3", call["text"])
        self.assertIn(self.doubt.table_block(G, self.thread().id, self.ann), call["text"])
        self.assertEqual({c["feature"] for c in self.models.calls if c["kind"] in ("director", "dialogue")}, {"game"})
        self.assertFalse(any(c["table_locked"] for c in self.models.calls))

    async def test_an_event_turn_never_shows_another_seat_the_hand_of_the_speaker(self):
        await self.table5()
        self.models.pick = [self.ann, self.bo, self.cy]
        await self.run_event(text="Doubt! A claim was doubted and the pile changed hands.")  # a fixed card in the text could be another seat's
        dialogues = self.models.of("dialogue")
        self.assertEqual(len(dialogues), 3)
        for call in dialogues:
            self.assert_no_hidden(call, self.seat_of(call["speaker"]))

    async def test_a_director_that_stays_quiet_leaves_the_thread_untouched(self):
        await self.table3()
        before = self.announcements()
        self.models.pick = []
        await self.bot.run_table_event(self.thread(), self.EVENT)
        self.assertEqual(self.spoke(), [])
        self.assertEqual(self.announcements(), before)

    async def test_an_event_is_skipped_without_waiting_when_the_thread_is_busy(self):
        await self.table3()
        self.models.pick = [self.ann]
        async with self.bot.channel_locks.lock(self.thread().id):
            await asyncio.wait_for(self.bot.run_table_event(self.thread(), self.EVENT), 1)
        self.assertEqual(self.models.calls, [])
        self.assertEqual(self.spoke(), [])

    async def test_an_event_follows_the_same_gating_as_a_member_message(self):
        await self.table3()
        self.models.pick = [self.ann]
        self.store.set_game_talk_daily_limit(G, 2)
        self.add_usage(2)
        await self.run_event()
        self.assertEqual(self.models.calls, [])
        self.store.set_game_talk_daily_limit(G, 100)
        self.store.set_game_character_talk(G, False)
        await self.run_event()
        self.assertEqual(self.models.calls, [])
        self.store.set_game_character_talk(G, True)
        self.hard_cap()
        await self.run_event()
        self.assertEqual(self.models.calls, [])

    async def test_an_event_in_the_banter_after_the_game_is_restricted_and_has_no_hand(self):
        await self.table3()
        await self.finish_by_forfeits()
        self.models.pick = [self.dee, self.ann]
        await self.run_event(text="Bob won the game.")
        self.assertEqual(self.spoke(), [self.ann])
        self.assertNotIn("Your hand:", self.models.of("dialogue")[0]["text"])

    async def test_an_event_in_a_thread_that_is_not_a_table_does_nothing(self):
        await self.table3()
        self.models.pick = [self.ann]
        plain = await self.ch.make_thread("plain chat")
        await self.run_event(thread=plain)
        self.assertEqual(self.models.calls, [])


class EventScheduleTests(TalkCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.events, self.hold = [], None

        async def spy(thread, text, speaker_hint=None):
            snap = self.store.latest_round(G, self.tid)
            view = snap["view"]
            hidden = set()
            if snap["status"] == "playing":
                state = self.engine_state(snap)
                hidden = set(tokens([c for hand in state.hands for c in hand] + list(state.pile)))
                for item in state.history:
                    if getattr(item, "kind", "") == "reveal":
                        hidden -= set(tokens(item.cards))
            self.events.append({"thread": thread, "text": text, "hint": speaker_hint, "hidden": hidden,
                                "counts": view["counts"] if view else (), "finished": snap["status"] != "playing",
                                "last": self.moves()[-1][1] if self.moves() else None})
            if self.hold is not None:
                await self.hold.wait()
        self.bot.run_table_event = spy
        self.addAsyncCleanup(self.release)

    async def release(self):
        if self.hold is not None:
            self.hold.set()

    async def table3(self, seed=None):
        await super().table3(seed)
        self.tid = self.table()["id"]

    async def settle(self):
        for _ in range(12):
            await asyncio.sleep(0)

    async def doubt_step(self):
        """Doubts the next claim (Ann's through her timer, a member's through the picker): one reveal."""
        users = {0: ALICE, 2: BOB}
        for _ in range(4):
            view = self.snap()["view"]
            if view["window"]:
                claimer = view["window"]["claimer"]
                await self.doubt_press(users[next(seat for seat in users if seat != claimer)])
                return
            seat = view["turn"]
            if seat == 1:
                await self.fire()
            else:
                await self.pick_play(users[seat], {self.hand(seat)[0][0]: 1})
        self.fail("no claim to doubt")

    async def test_the_cooldown_is_twenty_seconds(self):
        self.assertEqual(self.doubt.EVENT_COOLDOWN_SECONDS, 20)

    async def test_a_reveal_schedules_one_public_event_turn_in_the_table_thread(self):
        await self.table3()
        await self.pick_play(ALICE, self.any_card(0))
        await self.settle()
        self.assertEqual(self.events, [])  # a plain claim is not an event
        await self.doubt_press(BOB)
        await self.settle()
        (event,) = self.events
        self.assertIs(event["thread"], self.thread())
        self.assertTrue(event["text"].strip())
        for card in event["hidden"]:
            self.assertNotIn(card, event["text"])

    async def test_events_are_rate_limited_per_table(self):
        await self.table3()
        await self.pick_play(ALICE, self.any_card(0))
        await self.doubt_press(BOB)
        await self.settle()
        self.assertEqual(len(self.events), 1)
        await self.doubt_step()
        await self.settle()
        self.assertEqual(len(self.events), 1, "a second reveal inside the cooldown")
        self.now[0] += 19
        await self.doubt_step()
        await self.settle()
        self.assertEqual(len(self.events), 1)
        self.now[0] += 2
        await self.doubt_step()
        await self.settle()
        self.assertEqual(len(self.events), 2)
        for event in self.events:
            for card in event["hidden"]:
                self.assertNotIn(card, event["text"])

    async def test_a_move_does_not_wait_for_the_event_turn(self):
        await self.table3()
        self.hold = asyncio.Event()
        await self.pick_play(ALICE, self.any_card(0))
        await asyncio.wait_for(self.doubt_press(BOB), 1)
        await self.settle()
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.moves()[-1][1], "doubt")

    async def test_the_game_end_schedules_an_event_even_when_it_comes_from_forfeits(self):
        await self.table3()
        self.now[0] += 100
        await self.finish_by_forfeits()
        await self.settle()
        self.assertTrue(self.events)
        self.assertTrue(self.events[-1]["finished"])
        self.assertEqual(sum(e["finished"] for e in self.events), 1)

    async def test_a_scripted_game_schedules_events_only_for_reveals_one_card_seats_and_the_end(self):
        await self.table3()
        self.store.set_game_talk_daily_limit(G, 10_000)
        step = [0]

        def clock():
            step[0] += 25
            return 10_000.0 + step[0]
        self.doubt.clock = clock
        tid, users, thread = self.tid, {0: ALICE, 2: BOB}, self.thread()
        for n in range(900):
            snap = self.store.latest_round(G, tid)
            if snap["status"] != "playing":
                break
            view, state, moves = snap["view"], self.engine_state(snap), snap["moves"]
            seat, window = view["turn"], view["window"]
            if window is not None and not state.hands[window["claimer"]]:
                await self.doubt.on_window_timer(G, tid, snap["round_id"], moves)
            elif seat == 1:
                await self.fire()
                if self.store.latest_round(G, tid)["moves"] == moves:
                    await self.doubt.on_window_timer(G, tid, snap["round_id"], moves)
            elif window is not None and any(c[0] != window["rank"] for c in state.window[2]):
                await DoubtButton("doubt", tid, snap["round_id"], moves).callback(self.click(users[seat], channel=thread))
            else:
                move = doubt.timeout_move(state, seat, f"talk/{n}")
                await self.doubt.play_from_picker(self.click(users[seat], channel=thread), tid, snap["round_id"], moves, base.parse_move(move))
            await self.settle()
        self.assertEqual(self.store.latest_round(G, tid)["status"], "settled")
        reveals = [e for e in self.events if e["last"] == "doubt" and not e["finished"]]
        plain = [e for e in self.events if e["last"] != "doubt" and not e["finished"]]
        finals = [e for e in self.events if e["finished"]]
        self.assertTrue(reveals)
        self.assertEqual(len(finals), 1)
        self.assertIs(self.events[-1], finals[0])
        self.assertEqual(len(plain), 1, [e["text"] for e in plain])  # the one time a seat reached its last card
        self.assertEqual(min(plain[0]["counts"]), 1)
        for event in self.events:
            for card in event["hidden"]:
                self.assertNotIn(card, event["text"])

    async def test_a_one_card_event_suppressed_by_the_gate_can_still_fire_later(self):
        await self.table3()
        real = self.store.play

        def play(*args, **kwargs):
            snap = real(*args, **kwargs)
            if snap["status"] == "playing":
                snap["view"]["counts"] = [1, *list(snap["view"]["counts"][1:])]
            return snap
        self.store.play = play
        self.store.set_game_character_talk(G, False)
        await self.pick_play(ALICE, self.any_card(0))
        await self.settle()
        self.assertEqual(self.events, [])
        self.assertFalse(self.doubt.table((G, self.tid)).one_card)
        self.store.set_game_character_talk(G, True)
        snap = self.snap()
        await self.doubt.on_window_timer(G, self.tid, snap["round_id"], snap["moves"])
        await self.fire()
        await self.settle()
        self.assertTrue(self.events)
        self.assertIn("down to one card", self.events[-1]["text"])


class ReviewFixTests(TalkCase):
    """FEAT-21 C review fixes: prompt-safe names, event time, no memory work, failed attempts, banter, the secrecy line."""

    def members(self):
        guild = SimpleNamespace(get_member=lambda ident: SimpleNamespace(display_name=NAMES.get(ident)) if ident in (ALICE, BOB) else None)
        self.bot.get_guild = lambda gid: guild

    async def settle_tasks(self):
        for _ in range(12):
            await asyncio.sleep(0)

    async def test_the_block_names_members_by_display_name_and_tells_the_character_to_keep_the_hand_secret(self):
        await self.table3()
        self.members()
        block = self.doubt.table_block(G, self.thread().id, self.ann)
        for name in ("Alice", "Bob"):
            self.assertIn(name, block)
        self.assertNotIn("Player", block)
        self.assertNotIn("<@", block)
        self.assertIn("Keep your hand secret", block)

    async def test_event_texts_and_the_event_turn_hold_no_mentions_or_markdown(self):
        await self.table3()
        self.members()
        events = []

        async def spy(thread, text, speaker_hint=None):
            events.append((thread, text))
        real = type(self.bot).run_table_event
        self.bot.run_table_event = spy
        await self.pick_play(ALICE, self.any_card(0))
        await self.doubt_press(BOB)
        await self.settle_tasks()
        ((thread, text),) = events
        self.assertNotIn("<@", text)
        self.assertNotIn("**", text)
        self.assertIn("Bob", text)
        self.models.pick = [self.ann]
        await real(self.bot, thread, text)
        await drain_memory_tasks(self.bot)
        (director,) = self.models.of("director")
        self.assertNotIn(f"<@{ALICE}>", director["text"])
        self.assertNotIn(f"<@{BOB}>", director["text"])
        self.assertTrue(all("<@" not in row["content"] for row in self.store.all("SELECT content FROM nodes")))

    async def test_an_event_turn_shows_the_current_time_not_the_year_2001(self):
        await self.table3()
        self.models.pick = [self.ann]
        await self.bot.run_table_event(self.thread(), "Bob won the game.")
        await drain_memory_tasks(self.bot)
        from datetime import datetime, timezone
        text = self.models.of("dialogue")[0]["text"]
        self.assertIn(str(datetime.now(timezone.utc).year), text)
        self.assertNotIn("2001", text)

    async def test_an_event_turn_time_is_the_clock_not_a_decoded_synthetic_id(self):
        """The event carries an explicit created_at; the negative synthetic id is only a unique node id."""
        from datetime import datetime, timezone
        from llmcord_core import discord_bot
        await self.table3()
        self.models.pick = [self.ann]
        moment = datetime(2031, 3, 4, 12, 30, tzinfo=timezone.utc).timestamp()
        fake = mock.Mock(wraps=time)
        fake.time.return_value = moment
        with mock.patch.object(discord_bot, "time", fake):
            await self.bot.run_table_event(self.thread(), "Bob won the game.")
        await drain_memory_tasks(self.bot)
        text = self.models.of("dialogue")[0]["text"]
        self.assertIn("March 4, 2031", text)
        self.assertIn("12:30 PM", text)

    def test_time_values_prefers_created_at_and_never_decodes_a_negative_id(self):
        from datetime import datetime, timezone
        from llmcord_core.prompts import time_values
        moment = datetime(2031, 3, 4, 12, 30, tzinfo=timezone.utc).timestamp()
        self.assertEqual(time_values(-5, "UTC", moment)["local_time"], "Tuesday, March 4, 2031, 12:30 PM (UTC)")
        self.assertEqual(time_values(-5, "UTC", moment)["isodate"], "2031-03-04")
        snowflake = ((int(datetime(2026, 10, 8, 8, 39, tzinfo=timezone.utc).timestamp() * 1000) - 1420070400000) << 22)
        self.assertEqual(time_values(snowflake, "UTC")["isotime"], "08:39")

    async def test_table_talk_schedules_no_memory_work_but_a_normal_thread_still_does(self):
        await self.table3()
        scheduled = []
        real = self.bot._schedule_memory
        self.bot._schedule_memory = lambda *args: (scheduled.append(args[0]), real(*args))[1]
        self.models.pick = [self.ann]
        await self.say()
        self.assertEqual(scheduled, [])
        self.assertEqual([c["kind"] for c in self.models.calls if c["kind"] in ("extract_memory", "summary")], [])
        thread = await self.ch.make_thread("plain chat")
        self.models.pick = [self.dee]
        await self.say(thread=thread)
        self.assertEqual(scheduled, [thread.id])

    async def test_failed_talk_turns_count_toward_the_daily_limit(self):
        await self.table3()
        self.models.pick = [self.ann]

        async def broken(role, request):
            raise RuntimeError("provider down")
            yield
        self.models.stream_compiled = broken
        await self.say()
        self.assertEqual(sum(self.doubt.talk_failed.values()), 1)

    async def test_a_failing_director_counts_as_a_failed_attempt(self):
        await self.table3()

        async def broken(role, request, name, schema):
            raise RuntimeError("provider down")
        self.models.structured_compiled = broken
        await self.say()
        self.assertGreaterEqual(sum(self.doubt.talk_failed.values()), 1)

    async def test_the_banter_survives_a_forget_and_ends_only_with_the_archive_timer(self):
        await self.table3()
        thread, tid = self.thread(), self.table()["id"]
        await self.finish_by_forfeits()
        self.doubt._forget((G, tid))
        self.assertIsNotNone(self.doubt.table_talk_context(G, thread.id))
        await self.doubt.on_archive_timer(G, tid)
        self.assertIsNone(self.doubt.table_talk_context(G, thread.id))
