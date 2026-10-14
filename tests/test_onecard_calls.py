"""FEAT-45 step 2 (D39): One Card declarations and call-outs by chat message, the Discord side (``games_onecard_discord.OneCardTables``).

TDD-red tests written before the implementation. Offline: the fakes, the rigged deals and the fake clock of ``test_onecard_characters``; the pure
reader (``spoken.onecard.read_onecard``, tests in ``test_spoken_onecard.py``) and the engine (``games.onecard``) are the oracles. Timers are
driven by calling their hooks, never by waiting.

Assumed API (a proposal, pinned here):

  Hook. ``await OneCardTables.on_thread_message(message)``: reads one chat message of a table thread and applies what it says before returning
    (no background task; it never raises into ``on_message``, which awaits it next to ``note_thread_message``; the wiring itself is not tested
    here). Cheap memory filters first (``threads``: bot / webhook / foreign guild / unknown thread), then the store (the table is open, the round
    is playing, the author holds a seat, ``store.game_enabled(guild, "onecard")``). Reading is ``spoken.onecard.read_onecard(text, seats,
    reply_to, phrases)`` with ``seats = [(index, [name]), ...]`` (members by guild display name, characters by name), ``reply_to`` the seat of
    the message replied to and ``phrases`` the rules' ``declare_phrases``. The spoken-move switches (``spoken_member_on`` etc.) are never
    consulted. Only a ``sure`` reading acts; ``ask`` and ``none`` have no effect at all (no reply, no reaction, no move).
  Reply targets. ``message.reference.resolved`` is the replied-to message: a human's message maps to that member's seat (``author.id``); a
    character's webhook message (``webhook_id`` set) maps to the seat of the character whose name is ``author.name``; any other message (the
    bot's board and turn messages) is no target.
  Applying. ``declare`` / ``call_out:<seat>`` go through the table's normal step (table lock, ``store.play`` with actor ``member``, board and
    turn message repainted, timers re-armed); the author's message gets one reaction (``message.add_reaction``) after the lock is released and
    nothing else is said. A refused move (engine ``IllegalMove``, stale count) is dropped silently.
  Held declaration. A sure declaration by a seat with exactly 2 cards is kept in memory with ``self.clock()`` at receipt; it is applied as
    ``declare`` (actor ``member``) right after that seat's next play when that leaves exactly one card and the play is at most
    ``DECLARE_WINDOW = 10`` s after the message (inclusive). It is dropped when that seat draws, when any other seat moves first, or when
    the hand is not 1 afterwards. 3 or more cards: ignored. Reaction on the original message, when it is applied.
  Rules. ``games.onecard.normalize_rules`` gains ``declare_phrases`` (default ``["원카드", "One Card"]``, a list of strings) stored with the
    game's rules (``store.set_game_rules`` / ``game_rules``, per round in ``snap["rules"]``) and passed to the reader as ``phrases``.
  NPC seats. ``CALLOUT_SECONDS = 3``; a timer of kind ``"callout"`` (``arm(key, "callout", CALLOUT_SECONDS, round_id, moves)``) is armed when a
    move leaves a member exposed while a character is seated; hook ``await on_callout_timer(guild_id, table_id, round_id, moves)``
    (stale-safe: nothing happens when the table moved on; Discord work and the character line happen outside the table lock). The first
    active character seat calls out the exposed member (engine ``call_out:<seat>``, actor ``character``) and says a line through
    ``bot.run_table_event(thread, text, speaker_hint=<character id>)`` (the existing talk path; the text holds no hand, no mention). Needs
    ``talk_allowed(guild)``; with talk off or the daily limit reached characters stay silent and do not call out. Characters call out humans only.
    A character that plays down to one card decides on ``playstyle.roll(store.policy_seed(guild, round, f"{moves}/{seat}/declare"), ...)``:
    it forgets with a chance of at most 0.30 (scaled by its Play style), so a roll of 0.31 or more always declares and a roll of 0.0 forgets.
    Declaring applies engine ``declare`` (actor ``character``) and, when talk is allowed, posts the line "One card!" through the same
    ``run_table_event(..., speaker_hint=<character id>)``; forgetting leaves the seat exposed and says nothing.
  Help. ``_help_text`` (the "How to play" button) says how to declare ("원카드" / "One Card") and call out, and no longer says declarations
    are coming (the engine's ``how_to_play`` text changes).
"""
import dataclasses
from types import SimpleNamespace
from unittest import mock

from helpers import thread_message

from test_onecard_characters import BOB, CharCase, G, NAMES
from test_onecard_discord import ALICE, CARA

from llmcord_core.games import onecard, playstyle
from llmcord_core.games.playstyle import PlayStyle
from llmcord_core.games_onecard_discord import OneCardTables

FILLER = ["A♦", "5♦", "8♣", "Q♣", "10♣", "4♠", "3♠"]  # a seven-card hand with nothing in common with the test hands below
SEVEN = ["3♣", "K♠", "8♦", "6♠", "10♦", "4♣"]  # six more cards for a seven-card Alice who plays the 9♥ first
WEBHOOK_ID = 777


class CallCase(CharCase):
    """Alice 0 and Bob 1 (``duo``) or Alice 0, Ann 1, Bob 2 (``trio``). The top card is 5♥. Names: Alice, Bob, Ann."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        guild = SimpleNamespace(get_member=lambda ident: SimpleNamespace(display_name=NAMES[ident]) if ident in NAMES else None)
        self.bot.get_guild = lambda gid: guild
        self.lines, self.lock_seen = [], []

        async def spy(thread, text, speaker_hint=None, spectators=False):
            self.lines.append({"text": text, "hint": speaker_hint, "locked": self.onecard.runtime.lock((G, self.tid)).locked()})
        self.bot.run_table_event = spy
        self.onecard.CALLOUT_SECONDS = 3600

    # --- tables ------------------------------------------------------------------------------

    async def duo(self, alice, bob=None):
        await self.scene([alice, bob or FILLER], favorites=False)

    async def trio(self, alice, ann, bob=None):
        await self.scene([alice, ann, bob or FILLER], favorites=True)

    # --- messages ----------------------------------------------------------------------------

    async def say(self, user, text, *, reply=None, guild_id=G, thread=None, bot=False, webhook=False):
        message = thread_message(thread or self.thread(), user, text, bot=bot, guild_id=guild_id)
        message.add_reaction = mock.AsyncMock(side_effect=lambda emoji: self.lock_seen.append(self.onecard.runtime.lock((G, self.tid)).locked()))
        message.webhook_id = WEBHOOK_ID if webhook else None
        if reply is not None:
            message.reference = SimpleNamespace(message_id=reply.id, resolved=reply)
        await self.onecard.on_thread_message(message)
        return message

    def human(self, user, text="hmm"):
        return thread_message(self.thread(), user, text)

    def hook_message(self, name):
        """A character's webhook message as Discord delivers it: the webhook's id and the character's name."""
        thread = self.thread()
        return SimpleNamespace(id=thread.next_id(), webhook_id=WEBHOOK_ID, author=SimpleNamespace(id=WEBHOOK_ID, bot=True, name=name, display_name=name), content="...")

    # --- reading state -----------------------------------------------------------------------

    def log(self):
        return self.moves()

    def counts(self):
        return list(self.snap()["view"]["counts"])

    def view(self):
        view = dict(self.snap()["view"])
        view["exposed"] = list(view["exposed"])
        return view

    async def tick(self, seconds):
        self.now[0] += seconds

    async def callout_timer(self, moves=None):
        snap = self.snap()
        await self.onecard.on_callout_timer(G, self.tid, snap["round_id"], snap["moves"] if moves is None else moves)

    async def ann_turn(self):
        snap = self.snap()
        await self.onecard.on_character_timer(G, self.tid, snap["round_id"], snap["moves"])

    def force_roll(self, value, seat=1):
        """Every roll on Ann's declaration seed (whatever the move count it is tagged with) returns ``value``; other rolls are the real ones."""
        snap = self.snap()
        seeds = {self.store.policy_seed(G, snap["round_id"], f"{m}/{seat}/declare") for m in range(0, 14)}
        real = playstyle.roll
        patcher = mock.patch.object(playstyle, "roll", lambda seed, tag: value if seed in seeds else real(seed, tag))
        patcher.start()
        self.addCleanup(patcher.stop)

    def hinted(self, hint=None):
        return [line for line in self.lines if line["hint"] == (self.ann if hint is None else hint)]


class ReadingTests(CallCase):
    async def test_a_sure_declaration_at_one_card_applies_at_once_with_a_reaction_and_no_reply(self):
        """FEAT-45: "원카드" from a seat holding one card is the engine's declare; a small reaction, nothing said."""
        await self.duo(["9♥"])
        before = self.thread_texts()
        message = await self.say(ALICE, "원카드")
        self.assertEqual(self.log(), [(0, "declare", "member")])
        self.assertTrue(self.view()["declared"][0])
        message.add_reaction.assert_awaited_once()
        self.assertEqual(self.thread_texts(), before)
        self.assertEqual(self.lock_seen, [False])

    async def test_english_phrase_declares_too(self):
        """FEAT-45: "One card!" reads the same."""
        await self.duo(["9♥"])
        await self.say(ALICE, "One card!")
        self.assertEqual(self.log(), [(0, "declare", "member")])

    async def test_ask_and_none_readings_have_no_side_effects(self):
        """FEAT-45: a question, a mention of the game and unrelated chat do nothing: no move, no reaction, no reply, no reminder."""
        await self.duo(["9♥"])
        before, sent = self.thread_texts(), len(self.thread().sent)
        for text in ("원카드?", "원카드 아니야", "hello there", "원카드 규칙이 뭐야"):
            message = await self.say(ALICE, text)
            message.add_reaction.assert_not_awaited()
        self.assertEqual(self.log(), [])
        self.assertEqual(self.thread_texts(), before)
        self.assertEqual(len(self.thread().sent), sent)

    async def test_the_games_own_switch_is_the_consent(self):
        """FEAT-45: with One Card turned off for the server nothing is read."""
        await self.duo(["9♥"])
        thread = self.thread()
        self.store.set_game_enabled(G, "onecard", False)
        message = await self.say(ALICE, "원카드", thread=thread)
        self.assertEqual(self.moves(), [])
        message.add_reaction.assert_not_awaited()

    async def test_only_seated_human_authors_act(self):
        """FEAT-45: bots, webhook messages (a character's own "원카드!"), unseated members and other guilds are ignored."""
        await self.duo(["9♥"])
        other = self.channel2
        stranger = await other.make_thread("elsewhere") if hasattr(other, "make_thread") else None
        for kwargs in ({"user": CARA}, {"user": ALICE, "bot": True}, {"user": ALICE, "webhook": True}, {"user": ALICE, "guild_id": G + 1}):
            kwargs = dict(kwargs)
            message = await self.say(kwargs.pop("user"), "원카드", **kwargs)
            message.add_reaction.assert_not_awaited()
        if stranger is not None:
            message = await self.say(ALICE, "원카드", thread=stranger)
            message.add_reaction.assert_not_awaited()
        self.assertEqual(self.log(), [])

    async def test_a_settled_round_ignores_late_messages(self):
        """FEAT-45: a declaration or call-out arriving after the round settled changes nothing (the winner played the last card)."""
        await self.duo(["9♥"])
        thread, tid = self.thread(), self.tid
        await self.pick_play(ALICE, "play:9♥")
        moves = self.store.latest_round(G, tid)["moves"]
        for user, text in ((ALICE, "원카드"), (BOB, "Alice forgot one card"), (BOB, "원카드 안했네")):
            message = await self.say(user, text, thread=thread)
            message.add_reaction.assert_not_awaited()
        self.assertEqual(self.store.latest_round(G, tid)["moves"], moves)

    async def test_a_message_that_loses_the_race_to_the_move_is_dropped_silently(self):
        """FEAT-45: a declaration for a seat that already declared is refused by the engine; it is dropped without a reaction or an error."""
        await self.duo(["9♥"])
        await self.say(ALICE, "원카드")
        second = await self.say(ALICE, "원카드")
        second.add_reaction.assert_not_awaited()
        self.assertEqual(self.log(), [(0, "declare", "member")])

    async def test_the_server_phrase_list_replaces_the_default(self):
        """FEAT-45: the rules hold ``declare_phrases`` (default 원카드 / One Card); a custom list is what the reader gets."""
        self.assertEqual(onecard.normalize_rules(None)["declare_phrases"], ["원카드", "One Card"])
        self.store.set_game_rules(G, "onecard", {"ante": 10, "declare_phrases": ["유노"]})
        self.assertEqual(self.store.game_rules(G, "onecard")["declare_phrases"], ["유노"])
        await self.duo(["9♥"])
        plain = await self.say(ALICE, "원카드")
        plain.add_reaction.assert_not_awaited()
        await self.say(ALICE, "유노")
        self.assertEqual(self.log(), [(0, "declare", "member")])

    async def test_declare_phrases_must_be_a_list_of_text(self):
        """FEAT-45: a bad phrase list is a ValueError like any other bad rule."""
        for bad in ("원카드", [1], [""], 5):
            with self.assertRaises(ValueError):
                onecard.normalize_rules({"declare_phrases": bad})

    async def test_no_card_text_in_any_message_after_calls(self):
        """FEAT-45 (privacy): neither a declaration nor a call-out puts anyone's remaining card in a public message."""
        await self.duo(["9♥", "4♥"])
        await self.say(ALICE, "원카드")
        await self.pick_play(ALICE, "play:9♥")
        await self.onecard.flush()
        self.assertNotIn("4♥", self.everything_public())


class DeclarationTests(CallCase):
    async def test_two_cards_hold_the_declaration_until_the_next_play(self):
        """FEAT-45: at two cards nothing is applied (no reaction yet); the play that leaves one card applies declare right after it."""
        await self.duo(["9♥", "4♥"])
        message = await self.say(ALICE, "원카드")
        self.assertEqual(self.log(), [])
        message.add_reaction.assert_not_awaited()
        await self.tick(5)
        await self.pick_play(ALICE, "play:9♥")
        self.assertEqual(self.log(), [(0, "play:9♥", "member"), (0, "declare", "member")])
        self.assertTrue(self.view()["declared"][0])
        self.assertEqual(self.view()["exposed"], [])
        message.add_reaction.assert_awaited_once()

    async def test_the_window_is_ten_seconds_inclusive(self):
        """FEAT-45: a play exactly 10 s after the message still counts."""
        await self.duo(["9♥", "4♥"])
        await self.say(ALICE, "원카드")
        await self.tick(10)
        await self.pick_play(ALICE, "play:9♥")
        self.assertTrue(self.view()["declared"][0])

    async def test_a_held_declaration_older_than_ten_seconds_is_ignored(self):
        """FEAT-45: a stale message does not count: the seat is exposed after its play."""
        await self.duo(["9♥", "4♥"])
        message = await self.say(ALICE, "원카드")
        await self.tick(10.5)
        await self.pick_play(ALICE, "play:9♥")
        self.assertFalse(self.view()["declared"][0])
        self.assertEqual(self.view()["exposed"], [0])
        message.add_reaction.assert_not_awaited()

    async def test_a_draw_drops_the_held_declaration(self):
        """FEAT-45: drawing means the declaration was not for the later play, even within 10 s."""
        await self.duo(["9♥", "4♥"])
        await self.say(ALICE, "원카드")
        await self.pick_play(ALICE, "draw")
        await self.pick_play(BOB, "draw")
        await self.pick_play(ALICE, "play:9♥")
        await self.pick_play(BOB, "draw")
        await self.pick_play(ALICE, "play:4♥")
        self.assertEqual(self.counts()[0], 1)
        self.assertFalse(self.view()["declared"][0])
        self.assertEqual(self.view()["exposed"], [0])

    async def test_another_seat_moving_first_drops_the_held_declaration(self):
        """FEAT-45 (D39 "until the next seat acts"): Bob says it at two cards while Alice is on turn; Alice moves first, so it is dropped."""
        await self.duo(["9♥", *SEVEN], ["4♥", "K♥"])
        await self.say(BOB, "원카드")
        await self.pick_play(ALICE, "play:9♥")
        await self.pick_play(BOB, "play:4♥")
        self.assertFalse(self.view()["declared"][1])
        self.assertEqual(self.view()["exposed"], [1])

    async def test_three_or_more_cards_do_nothing(self):
        """FEAT-45: nothing is held at three cards: no reaction, and a later play to two cards is not a declaration."""
        await self.duo(["9♥", "4♥", "6♥"])
        message = await self.say(ALICE, "원카드")
        message.add_reaction.assert_not_awaited()
        await self.pick_play(ALICE, "play:9♥")
        self.assertEqual(self.log(), [(0, "play:9♥", "member")])
        self.assertFalse(self.view()["declared"][0])

    async def test_declaring_removes_the_exposure_mark_from_the_board(self):
        """FEAT-45: the board shows "exposed" after an undeclared drop to one card and "declared" once the seat declares."""
        await self.duo(["9♥", "4♥"])
        await self.pick_play(ALICE, "play:9♥")
        await self.onecard.flush()
        self.assertIn("exposed", self.board().content)
        await self.say(ALICE, "원카드")
        await self.onecard.flush()
        self.assertIn("declared", self.board().content)
        self.assertNotIn("exposed", self.board().content)


class HeldPerSeatTests(CallCase):
    async def test_another_seats_two_card_declaration_does_not_overwrite_a_hold(self):
        """FEAT-45: Bob's free "원카드" at two cards keeps Alice's hold; her play to one card still declares."""
        await self.duo(["9♥", "4♥"], ["K♥", "Q♥"])
        first = await self.say(ALICE, "원카드")
        await self.say(BOB, "원카드")
        await self.pick_play(ALICE, "play:9♥")
        self.assertTrue(self.view()["declared"][0])
        first.add_reaction.assert_awaited_once()

    async def test_a_seat_named_one_does_not_make_a_declaration_a_call_out(self):
        """FEAT-45: with a seat named "One", Alice's "one card" is still her declaration."""
        names = dict(NAMES)
        names[BOB] = "One"
        with mock.patch.dict(NAMES, names):
            await self.duo(["9♥"])
            await self.say(ALICE, "one card")
        self.assertEqual(self.log(), [(0, "declare", "member")])


class CallOutTests(CallCase):
    async def exposed_alice(self):
        await self.duo(["9♥", "4♥"])
        await self.pick_play(ALICE, "play:9♥")
        self.assertEqual(self.view()["exposed"], [0])

    async def test_a_call_out_by_name_hits_and_the_target_draws_the_penalty(self):
        """FEAT-45: "Alice forgot one card" by Bob applies call_out:0 at once; Alice draws the penalty; a reaction; nothing said."""
        await self.exposed_alice()
        before = self.thread_texts()
        message = await self.say(BOB, "Alice forgot one card")
        self.assertEqual(self.log()[-1], (1, "call_out:0", "member"))
        self.assertEqual(self.counts()[0], 2)
        self.assertEqual(self.view()["exposed"], [])
        message.add_reaction.assert_awaited_once()
        self.assertEqual(self.thread_texts(), before)
        self.assertEqual(self.lock_seen, [False])

    async def test_the_call_out_penalty_rule_applies(self):
        """FEAT-45: the table's call_out_penalty (here 3) is what the target draws."""
        self.store.set_game_rules(G, "onecard", {"ante": 10, "call_out_penalty": 3})
        await self.exposed_alice()
        await self.say(BOB, "Alice forgot one card")
        self.assertEqual(self.counts()[0], 4)

    async def test_an_implicit_call_out_hits_the_one_exposed_seat(self):
        """FEAT-45: "원카드 안했네" with no name and no reply, exactly one seat exposed."""
        await self.exposed_alice()
        await self.say(BOB, "원카드 안했네")
        self.assertEqual(self.log()[-1], (1, "call_out:0", "member"))

    async def test_an_implicit_call_out_with_nobody_exposed_is_ignored(self):
        """FEAT-45: no exposed seat and no target: nothing, so a vague remark never costs the caller a card."""
        await self.duo(["9♥", "4♥"])
        message = await self.say(BOB, "원카드 안했네")
        message.add_reaction.assert_not_awaited()
        self.assertEqual(self.log(), [])

    async def test_a_second_implicit_call_out_applies_once(self):
        """FEAT-45: two members call out the same exposed seat; the first hits, the second finds nobody exposed and is ignored."""
        await self.exposed_alice()
        first = await self.say(BOB, "원카드 안했네")
        second = await self.say(BOB, "원카드 안했네")
        first.add_reaction.assert_awaited_once()
        second.add_reaction.assert_not_awaited()
        self.assertEqual([m for m in self.log() if m[1].startswith("call_out")], [(1, "call_out:0", "member")])

    async def test_a_reply_to_a_members_message_targets_that_member(self):
        """FEAT-45: replying to Alice's own message with a call-out hits Alice."""
        await self.exposed_alice()
        earlier = self.human(ALICE)
        await self.say(BOB, "원카드 안 했잖아", reply=earlier)
        self.assertEqual(self.log()[-1], (1, "call_out:0", "member"))

    async def test_a_reply_to_a_character_webhook_message_targets_that_character(self):
        """FEAT-45: replying to Ann's webhook message targets Ann; she is not exposed, so it is a false call-out and the caller draws."""
        await self.trio(["9♥"], ["9♣", "K♦"])
        hooked = self.hook_message("Ann")
        await self.say(BOB, "원카드 안 했잖아", reply=hooked)
        self.assertEqual(self.log(), [(2, "call_out:1", "member")])
        self.assertEqual(self.counts()[2], 8)

    async def test_a_reply_to_the_board_or_turn_message_is_no_target(self):
        """FEAT-45: replying to the table's own messages names nobody; with nobody exposed nothing happens."""
        await self.duo(["9♥", "4♥"])
        for posted in (self.board(), self.turn()):
            message = await self.say(BOB, "원카드 안 했잖아", reply=SimpleNamespace(id=posted.id, webhook_id=None, author=SimpleNamespace(id=999, bot=True, name="Bot", display_name="Bot")))
            message.add_reaction.assert_not_awaited()
        self.assertEqual(self.log(), [])

    async def test_a_false_call_out_by_name_costs_the_caller(self):
        """FEAT-45: calling out a seat that declared makes the caller draw the penalty (the engine's rule)."""
        await self.duo(["9♥"])
        await self.say(ALICE, "원카드")
        await self.say(BOB, "Alice forgot one card")
        self.assertEqual(self.log()[-1], (1, "call_out:0", "member"))
        self.assertEqual(self.counts(), [1, 8])

    async def test_a_call_out_on_yourself_is_dropped(self):
        """FEAT-45: naming yourself is refused by the engine; no error, no reaction."""
        await self.exposed_alice()
        message = await self.say(ALICE, "Alice forgot one card")
        message.add_reaction.assert_not_awaited()
        self.assertEqual(self.view()["exposed"], [0])

    async def test_a_question_is_not_a_call_out(self):
        """FEAT-45: "Alice 원카드 안 했어?" is an ask reading: nothing happens."""
        await self.exposed_alice()
        message = await self.say(BOB, "Alice 원카드 안 했어?")
        message.add_reaction.assert_not_awaited()
        self.assertEqual(self.view()["exposed"], [0])

    async def test_only_seated_members_can_call_out(self):
        """FEAT-45: Cara has no seat."""
        await self.exposed_alice()
        message = await self.say(CARA, "Alice forgot one card")
        message.add_reaction.assert_not_awaited()
        self.assertEqual(self.view()["exposed"], [0])

    async def test_the_board_is_repainted_after_a_call_out(self):
        """FEAT-45: the board shows the new hand size and loses the exposed mark."""
        await self.exposed_alice()
        await self.onecard.flush()
        self.assertIn("exposed", self.board().content)
        await self.say(BOB, "Alice forgot one card")
        await self.onecard.flush()
        self.assertIn("2 cards", self.board().content)
        self.assertNotIn("exposed", self.board().content)


class CharacterSeatTests(CallCase):
    ANN = ["9♣", "K♦"]

    async def ann_down_to_one(self, roll=None):
        await self.trio(ALICE_HAND_PLAYS, self.ANN)
        if roll is not None:
            self.force_roll(roll)
        await self.pick_play(ALICE, "play:9♥")
        await self.ann_turn()

    async def test_a_character_declares_unless_it_rolls_a_forgetful_one(self):
        """FEAT-45: a roll of 0.99 declares: the engine's declare (actor character) follows her play."""
        await self.ann_down_to_one(0.99)
        self.assertEqual(self.log()[-2:], [(1, "play:9♣", "character"), (1, "declare", "character")])
        self.assertEqual(self.view()["exposed"], [])

    async def test_a_forgetful_roll_leaves_the_character_exposed_and_silent(self):
        """FEAT-45: a roll of 0.0 forgets (the chance is above zero for a neutral style): exposed, no declare, no line."""
        self.store.set_game_character_talk(G, True)
        await self.ann_down_to_one(0.0)
        self.assertEqual(self.view()["exposed"], [1])
        self.assertFalse(self.view()["declared"][1])
        self.assertEqual(self.hinted(), [])

    async def capped(self, value):
        style = PlayStyle(**{f.name: value for f in dataclasses.fields(PlayStyle)})
        await self.trio(ALICE_HAND_PLAYS, self.ANN)
        self.store.set_character_play_style(G, self.ann, style, self.store.owner_revision(G, "character", self.ann))
        self.force_roll(0.31)
        await self.pick_play(ALICE, "play:9♥")
        await self.ann_turn()
        self.assertTrue(self.view()["declared"][1], f"traits at {value}")

    async def test_the_forgetting_chance_is_capped_for_the_lowest_style(self):
        """FEAT-45: a roll of 0.31 declares whatever the Play style (every trait at 0)."""
        await self.capped(0)

    async def test_the_forgetting_chance_is_capped_for_the_highest_style(self):
        """FEAT-45: a roll of 0.31 declares whatever the Play style (every trait at 100)."""
        await self.capped(100)

    async def test_a_declaring_character_says_her_line_when_talk_is_allowed(self):
        """FEAT-45: through the table talk path, with her as the hinted speaker, outside the table lock; the line is "One card!"."""
        self.store.set_game_character_talk(G, True)
        self.store.set_game_character_moves(G, "style")
        await self.ann_down_to_one(0.99)
        lines = self.hinted()
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0]["text"], r"(?i)one card|원카드")
        self.assertFalse(lines[0]["locked"])
        self.assertNotIn("<@", lines[0]["text"])

    async def test_a_declaring_character_is_silent_when_talk_is_off(self):
        """FEAT-45: the declare still counts (it is game state), but no line is posted while character table talk is off."""
        await self.ann_down_to_one(0.99)
        self.assertTrue(self.view()["declared"][1])
        self.assertEqual(self.hinted(), [])

    async def test_a_character_never_calls_out_a_character(self):
        """FEAT-45: Ann forgets and stays exposed; no callout timer is armed and the hook does nothing."""
        self.store.set_game_character_talk(G, True)
        await self.ann_down_to_one(0.0)
        self.assertNotIn(((G, self.tid), "callout"), self.onecard.runtime.timers)
        await self.callout_timer()
        self.assertFalse([m for m in self.log() if m[1].startswith("call_out")])


ALICE_HAND_PLAYS = ["9♥", *SEVEN]


class CharacterCallOutTests(CallCase):
    async def alice_exposed(self, *, talk=True):
        self.store.set_game_character_talk(G, talk)
        await self.trio(["9♥", "4♥"], ["Q♠", "K♦"])
        await self.pick_play(ALICE, "play:9♥")
        self.assertEqual(self.view()["exposed"], [0])

    async def test_an_exposed_member_arms_the_callout_timer_when_a_character_sits(self):
        """FEAT-45: the characters' call-out waits CALLOUT_SECONDS (3) so a human can go first."""
        self.assertEqual(OneCardTables.CALLOUT_SECONDS, 3)
        await self.alice_exposed()
        self.assertIn(((G, self.tid), "callout"), self.onecard.runtime.timers)

    async def test_no_callout_timer_without_a_seated_character(self):
        """FEAT-45: a table of members only needs no timer."""
        await self.duo(["9♥", "4♥"])
        await self.pick_play(ALICE, "play:9♥")
        self.assertNotIn(((G, self.tid), "callout"), self.onecard.runtime.timers)

    async def test_after_the_delay_a_character_calls_out_the_exposed_member_with_a_line(self):
        """FEAT-45: call_out:0 from Ann's seat (actor character); Alice draws; Ann's line goes through the talk path outside the lock."""
        await self.alice_exposed()
        await self.callout_timer()
        self.assertEqual(self.log()[-1], (1, "call_out:0", "character"))
        self.assertEqual(self.counts()[0], 2)
        lines = self.hinted()
        self.assertEqual(len(lines), 1)
        self.assertFalse(lines[0]["locked"])
        self.assertNotIn("<@", lines[0]["text"])
        self.assertNotIn("4♥", lines[0]["text"])

    async def test_a_human_who_called_out_first_makes_the_timer_stale(self):
        """FEAT-45: Bob called out in the 3 s; the timer for the old move count does nothing, so Alice draws once."""
        await self.alice_exposed()
        snap = self.snap()
        await self.say(BOB, "Alice forgot one card")
        await self.onecard.on_callout_timer(G, self.tid, snap["round_id"], snap["moves"])
        self.assertEqual([m for m in self.log() if m[1].startswith("call_out")], [(2, "call_out:0", "member")])
        self.assertEqual(self.hinted(), [])
        self.assertEqual(self.counts()[0], 2)

    async def test_a_declaration_before_the_timer_makes_it_stale(self):
        """FEAT-45: Alice declares in time (late but before the character): the timer finds nobody exposed."""
        await self.alice_exposed()
        snap = self.snap()
        await self.say(ALICE, "원카드")
        await self.onecard.on_callout_timer(G, self.tid, snap["round_id"], snap["moves"])
        self.assertFalse([m for m in self.log() if m[1].startswith("call_out")])

    async def test_characters_stay_silent_and_do_not_call_out_when_talk_is_off(self):
        """FEAT-45: call-outs are table talk and respect its switch (and the daily limit through talk_allowed)."""
        await self.alice_exposed(talk=False)
        await self.callout_timer()
        self.assertFalse([m for m in self.log() if m[1].startswith("call_out")])
        self.assertEqual(self.hinted(), [])

    async def test_a_stale_timer_for_an_older_move_count_does_nothing(self):
        """FEAT-45: the timer carries the move count it was armed for."""
        await self.alice_exposed()
        await self.callout_timer(moves=0)
        self.assertFalse([m for m in self.log() if m[1].startswith("call_out")])


class HelpTextTests(CallCase):
    async def test_how_to_play_explains_declaring_and_calling_out_and_no_longer_says_coming(self):
        """FEAT-45: the table's help text (and the engine guide behind it) says how to declare and call out."""
        for text in (self.onecard._help_text(G, 1, 1), onecard.how_to_play()):
            low = text.lower()
            self.assertNotIn("coming", low)
            self.assertTrue("원카드" in text or "one card" in low)
            self.assertIn("call", low)
