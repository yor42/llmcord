"""FEAT-55 / D41 (step 3): spectators speak in a table thread when addressed, and react once per key moment.

Written before the implementation (TDD red). Offline: the Doubt and Hold'em table-talk fixtures (``TalkCase``) with a recording model gateway.

Assumed seams (proposed, pinned here; the implementer creates or adjusts them):

  Routing (``SkitBot.on_message`` / ``_table_talk_open`` / ``_table_fields``): with ``Store.cast_watches_games`` ON, an EXPLICIT message in a
    table thread (bot mention, or a reply to a character's node) is run with ``allowed_ids`` = seated characters + spectators, where the
    spectators are the thread's channel cast (``get_cast``) members that are not seated, plus the same-world favorites of the message's author
    (``eligible_favorites``; a favorite whose home world is not the channel's world is not eligible; another member's favorite is not eligible).
    The director still chooses. Non-explicit (ambient) thread messages keep seated-only ``allowed_ids``. With nobody seated an explicit message
    may still run (spectators only); a non-explicit one still returns early. Switch OFF: exactly today's behavior. ``talk_allowed``, the
    spending hard cap and the TalkBudget gate spectator turns exactly like seated ones. The usage feature stays ``'game'``.
  The spectator's dialogue prompt carries the PUBLIC view (the step-2 label ``Game table you are watching`` + ``<game>.public_block``), never
    ``Your hand`` / hole cards / any seat's cards. A seated speaker of the same turn keeps its own private block.
  Reactions: ``bot.run_table_event(thread, text, speaker_hint=None, spectators=False)``. With ``spectators=True`` the scene is an ambient event
    turn whose ``allowed_ids`` are the NON-seated channel cast only (no member is asking, so no favorites), at most ONE speaker, same gating,
    same ``'game'`` feature, a failed director/dialogue calls ``talk_budget.note_failure``. It does nothing when the switch is off or when
    no non-seated cast member exists. The two games schedule it (fire and forget, same ``event_tasks``) only when the switch is ON and
    ``game_character_moves`` is ``'style'``, at the same key moments and under the same EVENT_COOLDOWN / ``talk_allowed`` gate as the seated
    event, as a SEPARATE call after the unchanged seated call (same text). Because ``run_table_event`` skips a busy thread, the two calls
    must run one after the other, not concurrently.
  Blackjack (channel, no thread) schedules no spectator turn.
"""
import asyncio
from types import SimpleNamespace

import test_doubt_table_talk as dt
import test_games_discord as gd
import test_holdem_table_talk as ht
from helpers import drain_memory_tasks, thread_message

G, CH = dt.G, dt.CH
ALICE, BOB = dt.ALICE, dt.BOB
SPEC_LABEL = "Game table you are watching"


class Spectators:
    """The scenarios, shared by both games. Concrete classes provide the game specifics (see the hooks at the bottom)."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        s = self.store
        s.set_game_character_moves(G, "style")
        self.eve = s.create_character(G, self.world, "Eve")  # same world, not in the cast
        self.hal = s.create_character(G, self.world, "Hal")  # same world, nobody's favorite, not in the cast
        self.gus = s.create_character(G, self.world, "Gus")  # same world, Bob's favorite
        self.far_world = s.create_space(G, "Elsewhere", "world")
        self.far = s.create_character(G, self.far_world, "Far")  # another world, Alice's favorite
        s.add_favorite(G, ALICE, self.eve)
        s.add_favorite(G, ALICE, self.far)
        s.add_favorite(G, BOB, self.gus)
        for cid in (self.eve, self.hal, self.gus, self.far):
            self.hooks[cid] = dt.FakeWebhook(70000 + 1000 * cid)

    # --- driving ------------------------------------------------------------------------------

    async def chat(self, text="Hello there", *, user=ALICE, mention=True, reference_id=None):
        thread = self.game_thread()
        self.silent_history(thread)
        message = thread_message(thread, user, ("<@999> " if mention else "") + text)
        message.author.display_name = {ALICE: "Alice", BOB: "Bob"}.get(user, f"User {user}")
        message.mentions = [dt.BOT] if mention else []
        if reference_id is not None:
            message.reference = SimpleNamespace(message_id=reference_id)
        await self.bot.on_message(message)
        await drain_memory_tasks(self.bot)
        return message

    def cast_ids(self):
        directors = self.models.of("director")
        self.assertEqual(len(directors), 1, "expected exactly one director request")
        return {option["id"] for option in directors[0]["payload"]["cast"]}

    async def settle(self):
        for _ in range(12):
            await asyncio.sleep(0)
        if self.game.event_tasks:
            await asyncio.gather(*self.game.event_tasks, return_exceptions=True)

    def spy_events(self):
        self.events = []

        async def spy(thread, text, speaker_hint=None, spectators=False, **kwargs):
            self.events.append({"thread": thread, "text": text, "spectators": bool(spectators)})
        self.bot.run_table_event = spy

    def reactions(self):
        return [e for e in self.events if e["spectators"]]

    def seated_events(self):
        return [e for e in self.events if not e["spectators"]]

    def add_usage(self, count):
        import time
        for _ in range(count):
            self.store.db.execute("INSERT INTO model_usage(guild_id,profile,model,role,cached_tokens,reasoning_tokens,cost_basis,created_at,feature) VALUES(?,?,?,?,0,0,?,?,?)",
                                  (G, "p", "m", "director", "none", time.time(), "game"))
        self.store.db.commit()

    def failed(self):
        return sum(self.game.talk_failed.values())

    # --- A. mention routing ---------------------------------------------------------------------

    async def test_a_mention_with_the_switch_on_lets_the_channel_cast_spectators_reply(self):
        """FEAT-55: an explicit thread message offers seated + non-seated channel cast to the director."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table()
        self.models.pick = [self.dee]
        await self.chat("Dee, what do you make of this?")
        self.assertEqual(self.spoke(), [self.dee])
        self.assertEqual(self.cast_ids(), {self.ann, self.bo, self.cy, self.dee, self.eve})
        self.assertEqual({c["feature"] for c in self.models.calls if c["kind"] in ("director", "dialogue")}, {"game"})

    async def test_the_asking_members_same_world_favorite_may_reply_but_other_worlds_and_other_members_favorites_may_not(self):
        """FEAT-55: eligible = channel cast + the ASKER's same-world favorites; Far (another world), Gus (Bob's) and Hal (nobody's) are out."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table()
        self.models.pick = [self.eve, self.far, self.gus, self.hal]
        await self.chat("Eve, join in")
        ids = self.cast_ids()
        self.assertIn(self.eve, ids)
        self.assertTrue({self.far, self.gus, self.hal}.isdisjoint(ids))
        self.assertEqual(self.spoke(), [self.eve])

    async def test_another_member_asking_brings_their_own_favorite_not_alices(self):
        """FEAT-55: the pool follows the message's author."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table()
        self.models.pick = [self.gus]
        await self.chat("Gus?", user=BOB)
        ids = self.cast_ids()
        self.assertIn(self.gus, ids)
        self.assertNotIn(self.eve, ids)
        self.assertEqual(self.spoke(), [self.gus])

    async def test_a_reply_to_a_spectators_message_wakes_the_spectators_again(self):
        """FEAT-55: replying to a character's node is explicit too (no mention needed)."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table()
        self.models.pick = [self.dee]
        await self.chat("Dee?")
        self.assertEqual(len(self.hooks[self.dee].posts), 1)
        first = self.hooks[self.dee].posts[0]
        self.models.calls.clear()
        await self.chat("And now?", mention=False, reference_id=first.id)
        self.assertEqual(len(self.hooks[self.dee].posts), 2)
        self.assertIn(self.dee, self.cast_ids())

    async def test_an_ambient_thread_message_never_wakes_spectators(self):
        """FEAT-55: with the parent ambient, an invited but non-explicit message stays seated-only."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table()
        self.store.set_ambient(CH, True)
        self.models.pick = [self.dee, self.ann]
        await self.chat("Ann, what do you think?", mention=False)
        self.assertEqual(self.cast_ids(), {self.ann})
        self.assertEqual(self.spoke(), [self.ann])

    async def test_the_switch_off_keeps_spectators_silent_on_a_mention(self):
        """Characterization (FEAT-55): the switch is off by default; a mention offers only the seated characters."""
        await self.open_table()
        self.models.pick = [self.dee, self.ann]
        await self.chat("Dee?")
        self.assertEqual(self.cast_ids(), {self.ann})
        self.assertEqual(self.spoke(), [self.ann])

    async def test_a_mention_when_nobody_is_seated_reaches_the_spectators_only_with_the_switch(self):
        """FEAT-55: a table of members only is silent today; with the switch an explicit message reaches the cast."""
        await self.open_table_humans()
        self.models.pick = [self.dee]
        await self.chat("Dee?")
        self.assertEqual(self.models.calls, [])  # switch off: today's behavior
        self.store.set_cast_watches_games(G, True)
        await self.chat("Dee?")
        self.assertEqual(self.cast_ids(), {self.bo, self.cy, self.dee, self.ann, self.eve})
        self.assertEqual(self.spoke(), [self.dee])

    async def test_a_non_explicit_message_when_nobody_is_seated_still_returns_early(self):
        """Regression (FEAT-55): no mention, no reply, nobody seated: no turn even with the switch on and an ambient parent."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table_humans()
        self.store.set_ambient(CH, True)
        self.models.pick = [self.dee]
        await self.chat("Dee, what do you think?", mention=False)
        self.assertEqual(self.models.calls, [])
        self.assertEqual(self.spoke(), [])

    async def test_the_daily_limit_and_the_hard_cap_silence_spectators_too(self):
        """Regression (FEAT-55): talk_allowed and the spending cap gate spectator turns."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table()
        self.models.pick = [self.dee]
        self.store.set_game_talk_daily_limit(G, 2)
        self.add_usage(2)
        await self.chat("Dee?")
        self.assertEqual(self.models.calls, [])
        self.store.set_game_talk_daily_limit(G, 100)
        self.hard_cap()
        await self.chat("Dee?")
        self.assertEqual(self.models.calls, [])
        self.assertEqual(self.spoke(), [])

    async def test_a_failed_spectator_turn_counts_toward_the_daily_limit(self):
        """FEAT-55: a broken dialogue of a spectator counts as a failed attempt (nobody is seated, so only the new routing can reach it)."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table_humans()
        self.models.pick = [self.dee]

        async def broken(role, request):
            raise RuntimeError("provider down")
            yield
        self.models.stream_compiled = broken
        await self.chat("Dee?")
        self.assertEqual(self.failed(), 1)

    async def test_a_spectator_prompt_holds_the_public_view_and_no_hand_of_any_seat(self):
        """FEAT-55: Dee gets the watching block (public text), never a hand; the seated speaker of the same turn keeps its own block."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table()
        self.models.pick = [self.dee, self.ann]
        await self.chat("Dee and Ann?")
        self.assertCountEqual(self.spoke(), [self.dee, self.ann])
        dee = self.dialogue_of(self.dee)
        ann = self.dialogue_of(self.ann)
        self.assertIn(SPEC_LABEL, dee["text"])
        self.assertIn(self.public_text(), dee["text"])
        self.assertNotIn("Your hand", dee["text"])
        self.assertNotIn("Your hole cards", dee["text"])
        self.assertNotIn("Keep your hand secret", dee["text"])
        self.assert_no_cards(dee)
        self.assertIn(self.game.table_block(G, self.game_thread().id, self.ann), ann["text"])
        self.assertNotIn(SPEC_LABEL, ann["text"])

    # --- B. reactions ---------------------------------------------------------------------------

    async def test_a_spectator_event_turn_is_one_capped_ambient_turn_among_the_non_seated_cast(self):
        """FEAT-55: run_table_event(spectators=True): non-seated cast only, one speaker, one director + one dialogue call, public prompt."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table()
        self.models.pick = [self.ann, self.dee, self.bo, self.cy]
        await self.bot.run_table_event(self.game_thread(), self.EVENT, spectators=True)
        await drain_memory_tasks(self.bot)
        self.assertEqual(self.cast_ids(), {self.bo, self.cy, self.dee})
        (director,) = self.models.of("director")
        self.assertTrue(director["payload"]["ambient"])
        self.assertEqual(len(self.spoke()), 1)
        self.assertIn(self.spoke()[0], {self.bo, self.cy, self.dee})
        self.assertEqual(sorted(c["kind"] for c in self.models.calls), ["dialogue", "director"])
        (dialogue,) = self.models.of("dialogue")
        self.assertIn(self.EVENT, dialogue["text"])
        self.assertIn(SPEC_LABEL, dialogue["text"])
        self.assertNotIn("Your hand", dialogue["text"])
        self.assert_no_cards(dialogue)
        self.assertEqual({c["feature"] for c in self.models.calls}, {"game"})
        self.assertFalse(any(c["table_locked"] for c in self.models.calls))

    async def test_a_spectator_event_turn_with_the_switch_off_does_nothing(self):
        """FEAT-55: defensive: the switch gates the turn itself, not only the scheduling."""
        await self.open_table()
        self.models.pick = [self.dee]
        await self.bot.run_table_event(self.game_thread(), self.EVENT, spectators=True)
        self.assertEqual(self.models.calls, [])
        self.assertEqual(self.spoke(), [])

    async def test_a_spectator_event_turn_needs_a_non_seated_cast_member(self):
        """FEAT-55: with everyone in the cast seated there is nobody to ask and no model call."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table()
        self.store.set_cast(CH, None, [self.ann])
        self.models.pick = [self.ann]
        await self.bot.run_table_event(self.game_thread(), self.EVENT, spectators=True)
        self.assertEqual(self.models.calls, [])

    async def test_a_spectator_event_turn_follows_the_same_gating_and_skips_a_busy_thread(self):
        """FEAT-55: talk off, daily limit, hard cap and a busy thread all skip without a model call."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table()
        self.models.pick = [self.dee]
        async with self.bot.channel_locks.lock(self.game_thread().id):
            await asyncio.wait_for(self.bot.run_table_event(self.game_thread(), self.EVENT, spectators=True), 1)
        self.assertEqual(self.models.calls, [])
        self.store.set_game_talk_daily_limit(G, 2)
        self.add_usage(2)
        await self.bot.run_table_event(self.game_thread(), self.EVENT, spectators=True)
        self.assertEqual(self.models.calls, [])
        self.store.set_game_talk_daily_limit(G, 100)
        self.store.set_game_character_talk(G, False)
        await self.bot.run_table_event(self.game_thread(), self.EVENT, spectators=True)
        self.assertEqual(self.models.calls, [])
        self.store.set_game_character_talk(G, True)
        self.hard_cap()
        await self.bot.run_table_event(self.game_thread(), self.EVENT, spectators=True)
        self.assertEqual(self.models.calls, [])

    async def test_a_failed_spectator_event_turn_is_noted_as_a_failed_attempt(self):
        """FEAT-55: a broken director counts toward the daily limit through note_failure."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table()

        async def broken(role, request, name, schema):
            raise RuntimeError("provider down")
        self.models.structured_compiled = broken
        before = self.failed()
        await self.bot.run_table_event(self.game_thread(), self.EVENT, spectators=True)
        self.assertEqual(self.failed(), before + 1)
        self.assertEqual(self.spoke(), [])

    async def test_the_seated_event_turn_stays_seated_only_with_the_switch_on(self):
        """Regression (FEAT-55): a plain event (no spectators flag) offers only the seated characters, the switch on or not."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table()
        self.models.pick = [self.dee, self.ann]
        await self.bot.run_table_event(self.game_thread(), self.EVENT)
        self.assertEqual(self.cast_ids(), {self.ann})
        self.assertEqual(self.spoke(), [self.ann])

    async def test_a_key_moment_schedules_one_seated_and_one_spectator_event(self):
        """FEAT-55: switch on, style moves: the unchanged seated call plus exactly one spectator call with the same public text."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table()
        self.spy_events()
        await self.key_moment()
        await self.settle()
        self.assertEqual(len(self.seated_events()), 1)
        self.assertEqual(len(self.reactions()), 1)
        reaction = self.reactions()[0]
        self.assertEqual(reaction["text"], self.seated_events()[0]["text"])
        self.assertIs(reaction["thread"], self.game_thread())

    async def test_with_the_switch_off_a_key_moment_schedules_only_the_seated_event(self):
        """Characterization (FEAT-55): today's schedule, unchanged."""
        await self.open_table()
        self.spy_events()
        await self.key_moment()
        await self.settle()
        self.assertEqual(len(self.seated_events()), 1)
        self.assertEqual(self.reactions(), [])

    async def test_no_reaction_when_the_budget_is_used_up_or_talk_is_off(self):
        """FEAT-55: the daily limit and the table-talk switch silence the seated event and the reaction alike."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table()
        self.spy_events()
        self.store.set_game_talk_daily_limit(G, 2)
        self.add_usage(2)
        await self.key_moment()
        await self.settle()
        self.assertEqual(self.events, [])

    async def test_reactions_share_the_event_cooldown(self):
        """FEAT-55: a second key moment inside 20 s reacts neither seated nor spectator; after the cooldown both come back."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table()
        self.spy_events()
        await self.key_moment()
        await self.settle()
        self.assertEqual((len(self.seated_events()), len(self.reactions())), (1, 1))
        await self.another_moment()
        await self.settle()
        self.assertEqual((len(self.seated_events()), len(self.reactions())), (1, 1))
        self.advance(21)
        await self.another_moment()
        await self.settle()
        self.assertEqual((len(self.seated_events()), len(self.reactions())), (2, 2))

    async def test_a_reaction_runs_after_the_seated_event_not_under_it(self):
        """FEAT-55: through the real run_table_event both turns can speak for one moment (the second does not hit the busy lock)."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table()
        self.models.pick = [self.ann, self.dee]
        await self.key_moment()
        await self.settle()
        await drain_memory_tasks(self.bot)
        self.assertEqual(self.spoke().count(self.ann), 1)
        self.assertEqual(len([cid for cid in self.spoke() if cid in (self.bo, self.cy, self.dee)]), 1)


class DoubtSpectators(Spectators, dt.TalkCase):
    EVENT = "Doubt! A claim was doubted and the pile changed hands."

    @property
    def game(self):
        return self.doubt

    async def open_table(self):
        await self.table3()
        self.tid = self.table()["id"]

    async def open_table_humans(self):
        await self.start()
        await self.push(BOB, "Join", self.lobby())
        await self.push(ALICE, "Start", self.lobby())
        self.thread()
        self.tid = self.table()["id"]

    def game_thread(self):
        return self.thread()

    def public_text(self):
        return self.doubt.public_block(G, (G, self.tid))

    def assert_no_cards(self, call):
        for seat in range(len(self.snap()["seats"])):
            for token in self.cards_of(seat):
                self.assertNotIn(token, call["text"], f"{token} of a seat leaked into a spectator request")

    def advance(self, seconds):
        self.now[0] += seconds

    async def key_moment(self):
        await self.pick_play(ALICE, self.any_card(0))
        await self.doubt_press(BOB)

    async def another_moment(self):
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

    async def test_the_game_end_is_a_key_moment_for_a_spectator_reaction(self):
        """FEAT-55: the game's end schedules the seated event and one spectator reaction."""
        self.store.set_cast_watches_games(G, True)
        await self.open_table()
        self.spy_events()
        snap = self.snap()
        tid, rid = snap["table_id"], snap["round_id"]
        self.store_play(1, "forfeit", "timeout")
        await self.doubt.on_turn_timer(G, tid, rid, self.snap()["moves"])
        await self.doubt.on_window_timer(G, tid, rid, self.snap()["moves"])
        self.store_play(2, f"play:{self.hand(2)[0][0]}x1")
        await self.doubt.on_window_timer(G, tid, rid, self.snap()["moves"])
        await self.doubt.on_turn_timer(G, tid, rid, self.snap()["moves"])
        await self.settle()
        self.assertEqual(self.store.latest_round(G, tid)["status"], "settled")
        self.assertEqual(sum("won the game" in e["text"] for e in self.reactions()), 1)
        self.assertEqual(sum("won the game" in e["text"] for e in self.seated_events()), 1)


class HoldemSpectators(Spectators, ht.TalkCase):
    EVENT = "Showdown! Bob wins a pot of 240 with a pair of kings."

    @property
    def game(self):
        return self.holdem

    async def open_table(self):
        await self.table3("mnt57-b")  # a seed whose first three hands do not bust a seat (the cooldown test plays two more)

    def dialogue_of(self, character_id):
        return next(c for c in self.models.of("dialogue") if c["speaker"] == character_id)

    async def open_table_humans(self):
        await self.deal_table(sits=((ALICE, 1000), (BOB, 1000)))
        self.remember()

    def game_thread(self):
        return self.th

    def public_text(self):
        return self.holdem.public_block(G, (G, self.tid))

    def assert_no_cards(self, call):
        self.assert_clean(call)

    def advance(self, seconds):
        self.now_t[0] += seconds

    async def key_moment(self):
        await self.play_hand(ht.call_down_chooser)

    async def another_moment(self):
        await self.next_hand()
        self.remember()
        await self.play_hand(ht.call_down_chooser)


class BlackjackSpectators(gd.GameCase):
    async def test_a_blackjack_settle_triggers_no_spectator_turn(self):
        """Characterization (FEAT-55): blackjack has no thread and no unprompted reactions; the cast only sees the public view in prompts."""
        self.store.set_cast_watches_games(G, True)
        events, scenes = [], []

        async def spy(*args, **kwargs):
            events.append((args, kwargs))

        async def scene(*args, **kwargs):
            scenes.append(args)
        self.bot.run_table_event = spy
        self.bot.run_scene = scene
        self.rig("10", "9", "9", "7")
        await self.bet(gd.ALICE, 10)
        await self.press(gd.ALICE, "Deal now")
        await self.press(gd.ALICE, "Stand")
        self.assertEqual(self.latest()["status"], "settled")
        for _ in range(12):
            await asyncio.sleep(0)
        self.assertEqual((events, scenes), ([], []))
