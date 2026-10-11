"""FEAT-33 part D1: table talk in the Texas Hold'em table thread (mirrors ``test_doubt_table_talk.py``, FEAT-21 part C).

TDD-red tests written before the implementation (plain failing tests). Offline: fake Discord (``helpers``), the Hold'em fixtures of
``test_holdem_discord_hand`` (``HandCase``) and a recording model gateway that sees every director, dialogue and memory request of a turn.
Timers are never waited for; ``HoldemTables.clock`` is a settable fake.

Assumed API (proposed, pinned here; the implementer creates it). It is the Doubt API (see the module docstring of ``test_doubt_table_talk``)
on ``bot.holdem``, and the bot routes a thread to whichever game's table owns it:

  ``HoldemTables.table_talk_context(guild_id, thread_id)`` (sync) -> ``None`` or a dict with ``table_id``, ``round_id``, ``status`` (the newest
    round's status: ``'joining'`` / ``'playing'`` / ``'settled'``) and ``seated_character_ids`` (seat order). Guild-scoped. Covers the lobby, the
    hands, the pauses between hands and the banter after the table closed until the archive timer ran (or recovery closed it); ``None`` otherwise.
  ``HoldemTables.table_block(guild_id, thread_id, character_id)`` (sync) -> ``None`` (not a Hold'em table thread, wrong guild, or the character is
    not seated) or the per-speaker text: game name, street, board, pot, every seat's name and stack and whether it folded or is all-in, whose
    turn it is, the recent public actions, and that speaker's OWN two hole cards (``7♠`` tokens) while a hand is being played. Never another
    seat's hole cards (a revealed showdown hand is public and may appear), never Assist odds, never a seed or seed hash, no mentions.
    Between hands (and after the table closed) it holds no hole cards that were not published in the thread.
  ``bot.engine.table_block`` is the router: the dialogue prompt of a seated speaker carries that speaker's block whatever the preset says; the
    block is never in the director request, the saved nodes, the memory jobs or the catch-up. ``SceneContext.allowed_ids`` is the seated
    characters and ``usage_feature`` is ``'game'`` (as for Doubt).
  ``bot.run_table_event(thread, text, speaker_hint=None)`` runs an ambient turn in a Hold'em table thread with the same gating as a member message.
  ``HoldemTables.EVENT_COOLDOWN_SECONDS = 20`` (measured on ``HoldemTables.clock``) and ``HoldemTables.event_tasks``: at most one event turn per
    table per cooldown, fire and forget, scheduled with public text only on: a hand that reaches a showdown, a seat that busts, and a seat
    going all-in ("big pot"). Which of these is the exact trigger set is an assumption, see the notes in the report.
  Gating: ``game_character_talk`` on, and ``game_talk_calls_today + HoldemTables.talk_failed`` below ``game_talk_daily_limit``, and the spending
    hard cap; otherwise the characters stay silent (no model call, no status message) while the game goes on.
"""
import asyncio
import logging
import re
import time
from types import SimpleNamespace

import test_holdem_discord_hand as hand
from helpers import FakeGameClick, FakeWebhook, drain_memory_tasks, invoke, thread_message
from test_doubt_table_talk import TalkModels as DoubtTalkModels
from test_holdem_discord_hand import call_down, card, find_seed
from test_holdem_discord_lobby import ALICE, BOB, CARA, CH, G, G2, item_of, labels

from llmcord_core.games import holdem, seed_hash
from llmcord_core.games.base import Seat
from llmcord_core.prompts import default_bundle
from llmcord_core.usage import log_attribution

NAMES = {ALICE: "Alice", BOB: "Bob", CARA: "Cara"}
BOT = SimpleNamespace(id=999, bot=True, mention="<@999>", display_name="Bot")
SHOVE = hand.shove


class TalkModels(DoubtTalkModels):
    """The Doubt recorder, with the table-lock probe pointed at the Hold'em runtime."""

    def record(self, kind, request, role):
        text = "\n".join(m.text for m in request.messages)
        entry = {"kind": kind, "role": role, "text": text, "feature": log_attribution()["feature"], "speaker": self.speaker,
                 "table_locked": any(lock.locked() for lock in self.case.holdem.runtime.locks.values()), "request": request}
        self.calls.append(entry)
        return entry


class TalkCase(hand.HandCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        s = self.store
        s.set_game_character_talk(G, True)
        s.set_game_character_moves(G, "style")  # character moves never call the model unless a test says so
        self.dee = s.create_character(G, self.world, "Dee")  # in the cast, never seated at the tables below
        s.set_cast(CH, None, [self.ann, self.bo, self.cy, self.dee])
        s.change_character_balance(G, self.dee, 5000, "start", 1)
        self.models = TalkModels(self)
        self.bot.engine.models = self.bot.models = self.models
        self.bot._connection.user = BOT
        self.hooks = {cid: FakeWebhook(60000 + 1000 * cid) for cid in (self.ann, self.bo, self.cy, self.dee)}

        async def webhook(channel, character):
            self.models.speaker = character["id"]
            return self.hooks[character["id"]]
        self.bot._webhook = webhook
        logging.disable(logging.WARNING)
        self.addCleanup(logging.disable, logging.NOTSET)
        self.now_t = [10_000.0]
        self.holdem.clock = lambda: self.now_t[0]

        async def drain():
            await drain_memory_tasks(self.bot)
        self.addAsyncCleanup(drain)

    # --- tables ------------------------------------------------------------------------------

    def remember(self):
        """Remembers the current hand's hole cards and seed (the hidden information a prompt must never carry)."""
        self.dealt = [{card(c) for c in hole} for hole in self.holes()]
        self.seeds_in_play = [r["seed"] for r in self.store.all("SELECT seed FROM game_rounds WHERE guild_id=? AND status IN ('joining','playing')", (G,))]

    async def table3(self, seed=None):
        """Alice 0, Ann 1 (Alice's favorite), Bob 2; Bo, Cy and Dee are in the cast but not seated."""
        if seed:
            self.seed = seed
        await self.deal_table(sits=((BOB, 1000),), favorites=True)
        self.remember()

    async def table5(self):
        """Alice, Ann, Bob, Bo, Cy (Fill seats takes the cast in order); Dee is in the cast but not seated."""
        await self.start(favorites=True)
        await self.sit(BOB)
        await self.push(ALICE, "Fill seats")
        await self.push(ALICE, "Fill seats")
        await self.push(ALICE, "Deal")
        self.tid, self.th = self.table()["id"], self.thread()
        self.rid = self.snap()["round_id"]
        self.remember()

    async def lobby_only(self):
        await self.start(favorites=True)
        self.tid, self.th = self.table()["id"], self.thread()

    def seat_of(self, character_id):
        return next(s["index"] for s in self.now()["seats"] if s["kind"] == "character" and s["ref_id"] == character_id)

    def seated(self):
        return [s["ref_id"] for s in self.now()["seats"] if s["kind"] == "character"]

    def own(self, seat):
        return self.dealt[seat]

    def secret_from(self, seat):
        """Hole cards of the other seats that were never published in the thread (so no speaker at ``seat`` may see them)."""
        public = self.text()
        others = {t for i, cards in enumerate(self.dealt) if i != seat for t in cards} - self.dealt[seat]
        return {t for t in others if t not in public}

    def unpublished(self):
        """Every dealt hole card (own included) that no one has published in the thread."""
        public = self.text()
        return {t for cards in self.dealt for t in cards if t not in public}

    async def to_character(self, cid):
        """Plays the others (members call down, other characters through the timer) until ``cid`` is on turn."""
        for _ in range(40):
            snap = self.now()
            self.assertEqual(snap["status"], "playing", "the hand ended before the character came on turn")
            row = snap["seats"][snap["view"]["turn"]]
            if row["kind"] == "character" and row["ref_id"] == cid:
                return snap
            if row["kind"] == "character":
                await self.holdem.on_character_timer(G, self.tid, self.rid, snap["moves"])
            else:
                await self.do(row["ref_id"], call_down(self.menu(snap)))
        self.fail("the character never came on turn")

    async def to_member(self):
        """Plays characters through their timers until a member is on turn; returns the snapshot."""
        for _ in range(40):
            snap = self.now()
            row = snap["seats"][snap["view"]["turn"]]
            if row["kind"] == "member":
                return snap
            await self.holdem.on_character_timer(G, self.tid, self.rid, snap["moves"])
        self.fail("no member came on turn")

    async def to_street(self, street, folders=()):
        """Everybody checks or calls, straight through the store (the fallback policy of a character could shove the hand away).
        The turn message is not repainted, so its buttons are stale afterwards."""
        for _ in range(60):
            snap = self.now()
            if snap["status"] != "playing" or snap["view"]["street"] == street:
                break
            row = snap["seats"][snap["view"]["turn"]]
            menu = self.menu(snap)
            move = "fold" if ("character", row["ref_id"]) in folders and "fold" in menu else call_down(menu)
            self.store.play(G, self.rid, snap["view"]["turn"], move, row["kind"], snap["moves"])
        self.assertEqual(self.now()["view"]["street"], street)

    async def showdown_hand(self):
        """Table5 plus Cara: Cara folds at once, Alice and Bob call down, so there is a showdown and Cara's cards are never published."""
        await self.start(favorites=True)
        await self.sit(BOB)
        await self.sit(CARA)
        await self.push(ALICE, "Fill seats")
        await self.push(ALICE, "Deal")
        self.tid, self.th = self.table()["id"], self.thread()
        self.rid = self.snap()["round_id"]
        self.remember()
        snap = await self.play_hand(lambda menu, snap, seat: "fold" if snap["seats"][seat]["ref_id"] == CARA and "fold" in menu else call_down(menu))
        self.assertGreater(len(holdem.result(self.replayed(self.rid)).shown), 1, "expected a showdown")
        return snap

    def replayed(self, rid):
        snap = self.store.round_snapshot(G, rid)
        seats = [Seat(s["index"], s["kind"], s["ref_id"], s["stack"]) for s in snap["seats"]]
        moves = [(r["seat_index"], r["move"]) for r in self.store.all("SELECT seat_index,move FROM game_moves WHERE guild_id=? AND round_id=? ORDER BY id", (G, rid))]
        return holdem.replay(snap["seed"], seats, moves, snap["rules"], snap["rules"]["button"])

    def public_after(self, rid):
        """Cards a settled hand published: the board and the hands the result showed."""
        state = self.replayed(rid)
        shown = holdem.result(state).shown
        return {card(c) for i in shown for c in state.holes[i]} | {card(c) for c in state.board_full[:state.shown_n]}

    # --- driving the bot ---------------------------------------------------------------------

    @staticmethod
    def silent_history(thread):
        async def history(**kwargs):
            return
            yield  # an empty async generator
        thread.history = history

    async def say(self, user=ALICE, text="Hello there", *, thread=None, mention=True, drain=True, guild_id=G):
        thread = thread or self.th
        self.silent_history(thread)
        message = thread_message(thread, user, ("<@999> " if mention else "") + text, guild_id=guild_id)
        message.author.display_name = NAMES.get(user, f"User {user}")
        message.mentions = [BOT] if mention else []
        await self.bot.on_message(message)
        if drain:
            await drain_memory_tasks(self.bot)
        return message

    def spoke(self):
        return [cid for cid, h in self.hooks.items() if h.posts]

    def announcements(self, thread=None):
        return [m.content for m in (thread or self.th).sent if not m.deleted and m.content]

    def block(self, cid, thread=None):
        return self.holdem.table_block(G, (thread or self.th).id, cid)

    def assert_clean(self, call, seat=None):
        """One model request: no seed or hash of a hand in play, no cards nobody published, no mentions."""
        text = call["text"]
        for seed in self.seeds_in_play:
            self.assertNotIn(seed, text, f"a seed leaked into a {call['kind']} request")
            self.assertNotIn(seed_hash(seed)[:12], text, f"a seed hash leaked into a {call['kind']} request")
        hidden = self.unpublished() if seat is None else self.secret_from(seat)
        for token in hidden:
            if seat is not None and token in self.dealt[seat]:
                continue
            self.assertNotIn(token, text, f"{token} leaked into a {call['kind']} request")

    # --- spending ----------------------------------------------------------------------------

    def hard_cap(self):
        from datetime import datetime, timezone
        self.store.save_budget(None, 1.0, 1, False, self.store.budget_settings()["revision"])
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self.store.db:
            self.store.db.execute("INSERT INTO spend_days(day,cost_usd,unpriced_calls) VALUES(?,?,?) ON CONFLICT(day) DO UPDATE SET cost_usd=cost_usd+excluded.cost_usd", (day, 5.0, 0))


class ContextTests(TalkCase):
    async def test_a_playing_table_thread_reports_its_seated_characters(self):
        await self.table3()
        ctx = self.holdem.table_talk_context(G, self.th.id)
        snap = self.now()
        self.assertEqual(ctx["table_id"], self.tid)
        self.assertEqual(ctx["round_id"], snap["round_id"])
        self.assertEqual(ctx["seated_character_ids"], [self.ann])
        self.assertEqual(ctx["status"], "playing")

    async def test_a_lobby_thread_already_reports_its_seated_characters(self):
        await self.lobby_only()
        ctx = self.holdem.table_talk_context(G, self.th.id)
        self.assertEqual((ctx["seated_character_ids"], ctx["status"]), ([self.ann], "joining"))

    async def test_every_character_seat_is_listed_in_seat_order(self):
        await self.table5()
        ctx = self.holdem.table_talk_context(G, self.th.id)
        self.assertEqual(ctx["seated_character_ids"], [self.ann, self.bo, self.cy])

    async def test_other_threads_other_guilds_and_unknown_ids_have_no_context(self):
        await self.table3()
        plain = await self.ch.make_thread("plain chat")
        self.assertIsNone(self.holdem.table_talk_context(G, plain.id))
        self.assertIsNone(self.holdem.table_talk_context(G2, self.th.id))
        self.assertIsNone(self.holdem.table_talk_context(G, 424242))

    async def test_the_pause_between_hands_keeps_the_context(self):
        await self.table3()
        await self.play_hand(lambda menu, snap, seat: "fold" if "fold" in menu else call_down(menu))
        ctx = self.holdem.table_talk_context(G, self.th.id)
        self.assertEqual(ctx["seated_character_ids"], [self.ann])
        self.assertEqual(ctx["status"], "joining")

    async def test_the_banter_after_the_table_closed_keeps_the_context_until_the_archive_timer(self):
        """Characterization (FEAT-33 D1): like I Doubt It, a closed table's thread still talks (restricted) until it is archived."""
        await self.lobby_only()
        await self.push(ALICE, "Close table")
        self.assertIsNone(self.table())
        ctx = self.holdem.table_talk_context(G, self.th.id)
        self.assertEqual(ctx["seated_character_ids"], [self.ann])
        await self.holdem.on_archive_timer(G, self.tid)
        self.assertIsNone(self.holdem.table_talk_context(G, self.th.id))

    async def test_a_bot_restart_recovery_leaves_no_context(self):
        await self.table3()
        await self.holdem.recover()
        self.assertIsNone(self.holdem.table_talk_context(G, self.th.id))


class BlockTests(TalkCase):
    async def test_the_block_holds_public_state_and_only_the_speakers_own_cards(self):
        await self.table3()
        block = self.block(self.ann)
        view = self.now()["view"]
        self.assertIn("Hold'em", block)
        self.assertIn("Ann", block)
        self.assertRegex(block, rf"\b{view['pot']:,}|\b{view['pot']}\b")  # the pot (the blinds) is public
        for seat in self.now()["seats"]:
            stack = view["stacks"][seat["index"]]
            self.assertRegex(block, rf"\b{stack:,}\b|\b{stack}\b")
        ann = self.seat_of(self.ann)
        for token in self.own(ann):
            self.assertEqual(block.count(token), 1)
        for token in self.secret_from(ann):
            self.assertNotIn(token, block)

    async def test_the_block_never_carries_member_mentions_a_seed_or_a_hash(self):
        await self.table3()
        block = self.block(self.ann)
        self.assertNotIn("<@", block)
        for seed in self.seeds_in_play:
            self.assertNotIn(seed, block)
            self.assertNotIn(seed_hash(seed)[:12], block)

    async def test_each_seated_character_gets_its_own_block(self):
        await self.table5()
        for cid in (self.ann, self.bo, self.cy):
            seat, block = self.seat_of(cid), self.block(cid)
            for token in self.own(seat):
                self.assertIn(token, block)
            for token in self.secret_from(seat):
                self.assertNotIn(token, block)

    async def test_the_block_is_none_for_a_character_that_is_not_seated_and_for_other_threads(self):
        await self.table3()
        for cid in (self.bo, self.cy, self.dee):
            self.assertIsNone(self.block(cid))
        plain = await self.ch.make_thread("plain chat")
        self.assertIsNone(self.block(self.ann, plain))
        self.assertIsNone(self.holdem.table_block(G2, self.th.id, self.ann))

    async def test_the_block_follows_the_public_play_and_shows_the_flop(self):
        await self.table3()
        await self.to_street("flop")
        view = self.now()["view"]
        block = self.block(self.ann)
        for c in view["board"]:
            self.assertIn(card(c), block)
        self.assertRegex(block, rf"\b{view['pot']:,}|\b{view['pot']}\b")
        ann = self.seat_of(self.ann)
        for token in self.secret_from(ann):
            self.assertNotIn(token, block)

    async def test_a_folded_seats_cards_stay_out_of_everyone_elses_block(self):
        await self.table5()
        await self.to_street("flop", folders={("character", self.bo), ("character", self.cy)})
        snap = self.now()
        folded = [i for i in range(len(snap["seats"])) if snap["view"]["folded"][i]]
        self.assertTrue(folded, "a facing-a-bet character folds")
        for cid in self.seated():
            seat = self.seat_of(cid)
            if snap["view"]["folded"][seat]:
                continue
            for i in folded:
                for token in self.dealt[i] - self.dealt[seat]:
                    self.assertNotIn(token, self.block(cid))

    async def test_assist_odds_never_reach_the_block(self):
        self.store.set_game_rules(G, "holdem", {"assist": True})
        await self.start(favorites=True)
        await self.push(ALICE, "Assist on")
        await self.sit(BOB)
        await self.push(ALICE, "Deal")
        self.tid, self.th, self.rid = self.table()["id"], self.thread(), self.snap()["round_id"]
        self.remember()
        snap = self.now()
        press = self.click(ALICE, channel=self.th, message=self.turn_msg())
        await find_label(self.turn_msg(), "My cards").callback(press)
        odds = " ".join(text for text, _ in press.response.sent + press.followup.sent)
        self.assertIn("Assist", odds)  # the member does see the odds
        block = self.block(self.ann)
        self.assertNotIn("Assist", block)
        self.assertNotIn("%", block)
        self.assertNotIn(re.search(r"you have: ([^.]+)\.", odds).group(1), block)
        self.assertEqual(snap["status"], "playing")

    async def test_between_hands_the_block_holds_no_unpublished_hole_cards(self):
        await self.table3()
        await self.play_hand(lambda menu, snap, seat: "fold" if "fold" in menu else call_down(menu))
        block = self.block(self.ann)
        self.assertIsNotNone(block)
        for token in self.unpublished():
            self.assertNotIn(token, block)

    async def test_the_next_hand_gets_its_own_cards_and_forgets_the_old_ones(self):
        await self.table3()
        await self.play_hand(lambda menu, snap, seat: "fold" if "fold" in menu else call_down(menu))
        old = self.unpublished()
        await self.next_hand()
        self.remember()
        ann = self.seat_of(self.ann)
        block = self.block(self.ann)
        for token in self.own(ann):
            self.assertIn(token, block)
        for token in old - self.own(ann):
            self.assertNotIn(token, block)
        for token in self.secret_from(ann):
            self.assertNotIn(token, block)

    async def test_after_a_showdown_only_published_hands_may_appear(self):
        """Replay: the showdown publishes the live hands; a seat that folded keeps its cards secret from every later block."""
        await self.showdown_hand()
        hidden = self.unpublished()
        self.assertTrue(hidden, "Cara folded: her cards were never published")
        for cid in self.seated():
            block = self.block(cid)
            for token in hidden:
                self.assertNotIn(token, block)


def find_label(message, label):
    for child in message.view.children if message.view else ():
        if item_of(child).label == label:
            return child
    raise AssertionError(f"no {label!r} button among {labels(message)}")


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

    async def test_a_favorite_that_could_step_in_is_ignored_at_a_table(self):
        await self.table3()
        self.store.add_favorite(G, ALICE, self.dee)
        self.store.set_favorites_mode(G, ALICE, "step_in")
        self.models.pick = [self.dee, self.ann]
        await self.say()
        (director,) = self.models.of("director")
        self.assertEqual([option["id"] for option in director["payload"]["cast"]], [self.ann])
        self.assertEqual(self.spoke(), [self.ann])

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
        self.assertEqual(self.models.calls, [])

    async def test_with_an_ambient_parent_an_invited_message_gets_a_restricted_turn(self):
        await self.table3()
        self.store.set_ambient(CH, True)
        self.models.pick = [self.dee, self.ann]
        await self.say(text="Ann, what do you think?", mention=False)
        self.assertEqual(self.spoke(), [self.ann])

    async def test_summoning_a_character_that_is_not_seated_is_refused_privately(self):
        await self.table3()
        self.silent_history(self.th)
        click = FakeGameClick(self.bot, ALICE, channel=self.th)
        await invoke(self.bot, "summon", click, character="Dee", prompt="Join us")
        self.assertEqual(self.models.calls, [])
        self.assertEqual(self.spoke(), [])
        replies = click.response.sent + click.followup.sent
        self.assertEqual(len(replies), 1)
        self.assertTrue(replies[0][1].get("ephemeral"))

    async def test_summoning_a_seated_character_works_and_counts_as_table_talk(self):
        await self.table3()
        self.silent_history(self.th)
        click = FakeGameClick(self.bot, ALICE, channel=self.th)
        await invoke(self.bot, "summon", click, character="Ann", prompt="Join us")
        await drain_memory_tasks(self.bot)
        self.assertEqual(self.spoke(), [self.ann])
        self.assertEqual({c["feature"] for c in self.models.calls if c["kind"] in ("director", "dialogue")}, {"game"})


class HiddenInformationTests(TalkCase):
    async def full_turn(self):
        await self.table5()
        self.models.pick = [self.ann, self.bo, self.cy]
        await self.say()

    async def test_each_speaker_prompt_holds_its_own_cards_and_nobody_elses(self):
        await self.full_turn()
        dialogues = self.models.of("dialogue")
        self.assertEqual([c["speaker"] for c in dialogues], [self.ann, self.bo, self.cy])
        for call in dialogues:
            seat = self.seat_of(call["speaker"])
            for token in self.own(seat):
                self.assertIn(token, call["text"])
            self.assert_clean(call, seat)

    async def test_the_block_is_in_the_speakers_prompt_exactly_as_the_table_gives_it(self):
        await self.full_turn()
        for call in self.models.of("dialogue"):
            self.assertIn(self.block(call["speaker"]), call["text"])

    async def test_no_non_dialogue_request_of_the_turn_holds_any_hole_card_seed_or_block(self):
        await self.full_turn()
        self.assertTrue(self.models.of("director"))
        for call in self.models.calls:
            if call["kind"] == "dialogue":
                continue
            for cards in self.dealt:
                for token in cards:
                    self.assertNotIn(token, call["text"], f"{token} in a {call['kind']} request")
            self.assert_clean(call)
            for cid in (self.ann, self.bo, self.cy):
                self.assertNotIn(self.block(cid), call["text"])
            self.assertNotIn("Hold'em", call["text"])

    async def test_the_director_request_has_no_table_block_and_names_no_cards(self):
        await self.full_turn()
        (director,) = self.models.of("director")
        for cid in (self.ann, self.bo, self.cy):
            self.assertNotIn(self.block(cid), director["text"])
        for cards in self.dealt:
            for token in cards:
                self.assertNotIn(token, director["text"])

    async def test_a_dialogue_prompt_never_carries_a_fellow_speakers_block(self):
        await self.full_turn()
        for call in self.models.of("dialogue"):
            for cid in (self.ann, self.bo, self.cy):
                if cid != call["speaker"]:
                    self.assertNotIn(self.block(cid), call["text"])

    async def test_nothing_of_the_block_is_saved_in_the_scene_or_posted(self):
        await self.full_turn()
        rows = self.store.all("SELECT * FROM nodes")
        self.assertTrue(rows)
        saved = "\n".join(str(value) for row in rows for value in tuple(row))
        every_card = set().union(*self.dealt)
        for token in every_card:
            self.assertNotIn(token, saved)
        posted = "\n".join(m.content for h in self.hooks.values() for m in h.posts)
        for token in every_card:
            self.assertNotIn(token, posted)

    async def test_a_later_turn_does_not_carry_the_earlier_blocks_into_history(self):
        await self.table3()
        self.models.pick = [self.ann]
        await self.say()
        first = self.block(self.ann)
        await self.to_member()
        snap = self.now()
        row = snap["seats"][snap["view"]["turn"]]
        await self.do(row["ref_id"], call_down(self.menu(snap)))
        self.assertNotEqual(first, self.block(self.ann))
        self.models.calls.clear()
        await self.say(text="and now?")
        for call in self.models.of("dialogue"):
            self.assertNotIn(first, call["text"])
            self.assertIn(self.block(self.ann), call["text"])

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
        self.assertIn(self.block(self.ann), call["text"])

    async def test_the_flop_is_in_the_prompt_and_the_other_hands_still_are_not(self):
        await self.table5()
        await self.to_street("flop")
        self.models.pick = [self.ann, self.bo, self.cy]
        await self.say()
        for call in self.models.calls:
            self.assert_clean(call, self.seat_of(call["speaker"]) if call["kind"] == "dialogue" else None)
        for call in self.models.of("dialogue"):
            for c in self.now()["view"]["board"]:
                self.assertIn(card(c), call["text"])

    async def test_assist_odds_never_reach_any_prompt(self):
        self.store.set_game_rules(G, "holdem", {"assist": True})
        await self.start(favorites=True)
        await self.push(ALICE, "Assist on")
        await self.sit(BOB)
        await self.push(ALICE, "Deal")
        self.tid, self.th, self.rid = self.table()["id"], self.thread(), self.snap()["round_id"]
        self.remember()
        press = self.click(ALICE, channel=self.th, message=self.turn_msg())
        await find_label(self.turn_msg(), "My cards").callback(press)
        odds = " ".join(text for text, _ in press.response.sent + press.followup.sent)
        self.assertIn("Assist", odds)
        self.models.pick = [self.ann]
        await self.say()
        self.assertTrue(self.models.calls)
        for call in self.models.calls:
            self.assertNotIn("Assist", call["text"])
            self.assertNotRegex(call["text"], r"\d+%")
            self.assertNotIn("Chance to win", call["text"])

    async def test_after_a_showdown_replay_the_banter_prompts_hold_no_unpublished_card_and_no_seed(self):
        await self.showdown_hand()
        hidden = self.unpublished()
        self.assertTrue(hidden)
        self.models.pick = [self.ann, self.bo]
        await self.say()
        self.assertTrue(self.models.calls)
        for call in self.models.calls:
            for token in hidden:
                self.assertNotIn(token, call["text"], f"{token} in a {call['kind']} request")
            self.assert_clean(call)  # the next hand's seed (not yet revealed) is not in any request either

    async def test_a_folded_players_cards_never_leak_into_a_mid_hand_prompt(self):
        await self.table5()
        await self.to_street("flop", folders={("character", self.bo), ("character", self.cy)})
        self.assertTrue(any(self.now()["view"]["folded"]))
        self.models.pick = [self.ann, self.bo, self.cy]  # even a director that names the folded seats
        await self.say()
        for call in self.models.calls:
            self.assert_clean(call, self.seat_of(call["speaker"]) if call["kind"] == "dialogue" else None)

    async def test_the_next_hand_prompt_holds_the_new_cards_only(self):
        await self.table3()
        await self.play_hand(lambda menu, snap, seat: "fold" if "fold" in menu else call_down(menu))
        old = self.unpublished()
        await self.next_hand()
        self.remember()
        self.models.pick = [self.ann]
        await self.say()
        (call,) = self.models.of("dialogue")
        ann = self.seat_of(self.ann)
        for token in self.own(ann):
            self.assertIn(token, call["text"])
        for token in old - self.own(ann):
            self.assertNotIn(token, call["text"])
        self.assert_clean(call, ann)


class GatingTests(TalkCase):
    async def silent(self, note):
        before = self.announcements()
        await self.say()
        self.assertEqual(self.models.calls, [], note)
        self.assertEqual(self.spoke(), [], note)
        self.assertEqual(self.announcements(), before, note)  # no status line, no "No active character", no notice

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
        self.holdem.talk_failed[(G, self.store._server_day(G, time.time())[1])] = 2
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
        await self.silent("hard cap")

    async def test_talk_calls_of_another_guild_do_not_use_up_this_guilds_limit(self):
        """Guild isolation: usage rows of guild G2 do not silence guild G."""
        await self.table3()
        self.store.set_game_talk_daily_limit(G, 3)
        self.add_usage(50, guild=G2)
        self.models.pick = [self.ann]
        await self.say()
        self.assertEqual(self.spoke(), [self.ann])

    async def test_another_guilds_talk_setting_does_not_silence_this_guild(self):
        await self.table3()
        self.store.set_game_character_talk(G2, False)
        self.models.pick = [self.ann]
        await self.say()
        self.assertEqual(self.spoke(), [self.ann])

    async def test_a_silent_table_still_plays_without_a_model_call(self):
        """Talk off, model-move mode: the character's seat still plays its turn through the policy, silently."""
        await self.table3()
        self.store.set_game_character_talk(G, False)
        self.store.set_game_character_moves(G, "model")
        snap = await self.to_character(self.ann)
        await self.holdem.on_character_timer(G, self.tid, self.rid, snap["moves"])
        self.assertGreater(self.now()["moves"], snap["moves"])
        self.assertEqual(self.models.calls, [])
        self.assertEqual(self.spoke(), [])

    async def test_a_table_at_the_limit_still_plays_to_the_end_of_the_hand(self):
        await self.table3()
        self.store.set_game_talk_daily_limit(G, 1)
        self.add_usage(1)
        await self.play_hand(call_down_chooser)
        self.assertEqual(self.models.calls, [])
        self.assertEqual(self.spoke(), [])

    def add_usage(self, count, guild=G):
        for _ in range(count):
            self.store.db.execute("INSERT INTO model_usage(guild_id,profile,model,role,cached_tokens,reasoning_tokens,cost_basis,created_at,feature) VALUES(?,?,?,?,0,0,?,?,?)",
                                  (guild, "p", "m", "director", "none", time.time(), "game"))
        self.store.db.commit()


def call_down_chooser(menu, snap, seat):
    return call_down(menu)


class UsageTests(TalkCase):
    async def test_every_model_call_of_a_table_turn_counts_as_table_talk(self):
        await self.table3()
        self.models.pick = [self.ann]
        await self.say()
        turn = [c for c in self.models.calls if c["kind"] in ("director", "dialogue")]
        self.assertEqual([c["kind"] for c in turn], ["director", "dialogue"])
        self.assertEqual({c["feature"] for c in turn}, {"game"})

    async def test_table_talk_never_runs_under_the_table_lock(self):
        await self.table3()
        self.models.pick = [self.ann]
        await self.say()
        self.assertTrue(self.models.calls)
        self.assertFalse(any(c["table_locked"] for c in self.models.calls))

    async def test_a_move_does_not_wait_for_a_talk_turn_in_flight(self):
        await self.table3()
        snap = await self.to_member()
        self.models.pick = [self.ann]
        self.models.gate = asyncio.Event()
        task = asyncio.create_task(self.say())
        try:
            await asyncio.wait_for(self.models.entered.wait(), 1)
            row = snap["seats"][snap["view"]["turn"]]
            await asyncio.wait_for(self.do(row["ref_id"], call_down(self.menu(snap))), 1)
            self.assertGreater(self.now()["moves"], snap["moves"])
        finally:
            self.models.gate.set()
            await task


class NormalThreadTests(TalkCase):
    async def test_a_thread_that_is_not_a_table_keeps_the_whole_cast_and_the_old_usage_label(self):
        """Regression (FEAT-33 D1): every non-table location is unchanged and ignores the table talk settings."""
        await self.table3()
        self.store.set_game_character_talk(G, False)
        self.store.set_game_talk_daily_limit(G, 1)
        thread = await self.ch.make_thread("plain chat")
        self.models.pick = [self.dee, self.bo]
        await self.say(thread=thread)
        (director,) = self.models.of("director")
        self.assertEqual({option["id"] for option in director["payload"]["cast"]}, {self.ann, self.bo, self.cy, self.dee})
        self.assertEqual(sorted(self.spoke()), sorted([self.dee, self.bo]))
        for call in self.models.of("dialogue"):
            self.assertNotIn("Hold'em", call["text"])
            for cards in self.dealt:
                for token in cards:
                    self.assertNotIn(token, call["text"])
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
            self.assertNotIn("Hold'em", call["text"])
        self.assertEqual({c["feature"] for c in self.models.calls if c["kind"] in ("director", "dialogue")}, {"reply"})

    async def test_after_the_archive_timer_the_thread_is_a_normal_thread(self):
        """Characterization (FEAT-33 D1): once the table is archived its thread talks like any other thread of the channel."""
        await self.lobby_only()
        await self.push(ALICE, "Close table")
        await self.holdem.on_archive_timer(G, self.tid)
        await self.assert_normal_thread()

    async def test_after_a_restart_recovery_the_thread_is_a_normal_thread(self):
        """Characterization (FEAT-33 D1): a table closed by recovery no longer restricts its thread."""
        await self.table3()
        await self.holdem.recover()
        await self.assert_normal_thread()

    async def test_the_banter_after_the_table_closed_is_restricted_and_holds_no_hole_cards(self):
        await self.table3()
        await self.holdem.close_for_channel(G, CH, self.tid)
        self.models.pick = [self.dee, self.ann]
        await self.say()
        (director,) = self.models.of("director")
        self.assertEqual([option["id"] for option in director["payload"]["cast"]], [self.ann])
        self.assertEqual(self.spoke(), [self.ann])
        (call,) = self.models.of("dialogue")
        self.assertEqual(call["feature"], "game")
        for token in self.unpublished():
            self.assertNotIn(token, call["text"])

    async def test_the_lobby_thread_is_restricted_and_has_no_cards_yet(self):
        await self.lobby_only()
        self.models.pick = [self.dee, self.ann]
        await self.say()
        self.assertEqual(self.spoke(), [self.ann])
        (call,) = self.models.of("dialogue")
        self.assertEqual(call["feature"], "game")
        self.assertNotRegex(call["text"], r"(?:10|[2-9AJQK])[♠♥♦♣]")


class EventTurnTests(TalkCase):
    EVENT = "Showdown! Bob wins a pot of 240 with a pair of kings."

    async def run_event(self, thread=None, text=None):
        await self.bot.run_table_event(thread or self.th, text or self.EVENT)
        await drain_memory_tasks(self.bot)

    async def test_an_event_runs_one_ambient_director_turn_restricted_to_the_seated_characters(self):
        await self.table3()
        self.models.pick = [self.dee, self.ann]
        await self.run_event()
        (director,) = self.models.of("director")
        self.assertEqual([option["id"] for option in director["payload"]["cast"]], [self.ann])
        self.assertTrue(director["payload"]["ambient"])
        self.assertEqual(self.spoke(), [self.ann])
        (call,) = self.models.of("dialogue")
        self.assertIn("pair of kings", call["text"])
        self.assertIn(self.block(self.ann), call["text"])
        self.assertEqual({c["feature"] for c in self.models.calls if c["kind"] in ("director", "dialogue")}, {"game"})
        self.assertFalse(any(c["table_locked"] for c in self.models.calls))

    async def test_an_event_turn_never_shows_a_seat_another_hand(self):
        await self.table5()
        self.models.pick = [self.ann, self.bo, self.cy]
        await self.run_event()
        dialogues = self.models.of("dialogue")
        self.assertEqual(len(dialogues), 3)
        for call in dialogues:
            self.assert_clean(call, self.seat_of(call["speaker"]))
        for call in self.models.of("director"):
            self.assert_clean(call)

    async def test_a_director_that_stays_quiet_leaves_the_thread_untouched(self):
        await self.table3()
        before = self.announcements()
        self.models.pick = []
        await self.bot.run_table_event(self.th, self.EVENT)
        self.assertEqual(self.spoke(), [])
        self.assertEqual(self.announcements(), before)

    async def test_an_event_is_skipped_without_waiting_when_the_thread_is_busy(self):
        await self.table3()
        self.models.pick = [self.ann]
        async with self.bot.channel_locks.lock(self.th.id):
            await asyncio.wait_for(self.bot.run_table_event(self.th, self.EVENT), 1)
        self.assertEqual(self.models.calls, [])
        self.assertEqual(self.spoke(), [])

    async def test_an_event_follows_the_same_gating_as_a_member_message(self):
        await self.table3()
        self.models.pick = [self.ann]
        self.store.set_game_talk_daily_limit(G, 2)
        GatingTests.add_usage(self, 2)
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

    async def test_an_event_in_the_banter_after_the_table_closed_is_restricted(self):
        await self.table3()
        await self.holdem.close_for_channel(G, CH, self.tid)
        self.models.pick = [self.dee, self.ann]
        await self.run_event(text="The table is closed.")
        self.assertEqual(self.spoke(), [self.ann])

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
            snap = self.store.round_snapshot(G, self.rid)
            public = self.public_after(self.rid) if snap["status"] == "settled" else {card(c) for c in self.now()["view"]["board"]}
            self.events.append({"thread": thread, "text": text, "hint": speaker_hint, "hidden": set().union(*self.dealt) - public,
                                "settled": snap["status"] == "settled"})
            if self.hold is not None:
                await self.hold.wait()
        self.bot.run_table_event = spy
        self.addAsyncCleanup(self.release)

    async def release(self):
        if self.hold is not None:
            self.hold.set()

    async def settle(self):
        for _ in range(12):
            await asyncio.sleep(0)

    def assert_public_only(self):
        for event in self.events:
            self.assertTrue(event["text"].strip())
            for token in event["hidden"]:
                self.assertNotIn(token, event["text"], "an event text named a card nobody had published")
            self.assertNotIn("<@", event["text"])

    async def heads_up(self):
        await self.deal_table(sits=((ALICE, 1000), (BOB, 1000)))
        self.remember()

    async def test_the_cooldown_is_twenty_seconds(self):
        self.assertEqual(self.holdem.EVENT_COOLDOWN_SECONDS, 20)

    async def test_a_hand_that_reaches_a_showdown_schedules_one_public_event_turn_in_the_thread(self):
        await self.heads_up()
        await self.play_hand(call_down_chooser)
        await self.settle()
        self.assertEqual(len(self.events), 1)
        self.assertIs(self.events[0]["thread"], self.th)
        self.assert_public_only()

    async def test_a_plain_call_down_mid_hand_is_not_an_event(self):
        await self.heads_up()
        snap = self.now()
        row = snap["seats"][snap["view"]["turn"]]
        await self.do(row["ref_id"], "call")
        await self.settle()
        self.assertEqual(self.events, [])

    async def test_a_hand_won_by_folds_never_names_the_folded_cards(self):
        await self.table3()
        await self.play_hand(lambda menu, snap, seat: "fold" if "fold" in menu else call_down(menu))
        await self.settle()
        self.assert_public_only()  # whether or not a fold-out hand is an event, its text holds no card nobody published

    async def test_a_bust_schedules_an_event(self):
        self.seed = find_seed("bust")
        await self.deal_table(sits=((ALICE, 400), (BOB, 700), (CARA, 1000)))
        self.remember()
        await self.play_hand(SHOVE_CHOOSER)
        await self.settle()
        self.assertEqual(len(self.events), 1)  # showdown and bust of one hand are one event turn, not two
        self.assert_public_only()

    async def test_an_all_in_is_a_big_pot_event(self):
        """Assumption: a seat committing its whole stack is the 'big pot' trigger; the text names no card."""
        await self.heads_up()
        snap = self.now()
        row = snap["seats"][snap["view"]["turn"]]
        await self.do(row["ref_id"], SHOVE(self.menu(snap)))
        await self.settle()
        self.assertEqual(len(self.events), 1)
        self.assertFalse(self.events[0]["settled"])
        self.assert_public_only()

    async def test_events_are_rate_limited_per_table(self):
        await self.heads_up()
        await self.play_hand(call_down_chooser)
        await self.settle()
        self.assertEqual(len(self.events), 1)
        await self.next_hand()
        self.remember()
        await self.play_hand(call_down_chooser)
        await self.settle()
        self.assertEqual(len(self.events), 1, "a second showdown inside the cooldown")
        self.now_t[0] += 19
        await self.next_hand()
        self.remember()
        await self.play_hand(call_down_chooser)
        await self.settle()
        self.assertEqual(len(self.events), 1)
        self.now_t[0] += 2
        await self.next_hand()
        self.remember()
        await self.play_hand(call_down_chooser)
        await self.settle()
        self.assertEqual(len(self.events), 2)
        self.assert_public_only()

    async def test_a_move_does_not_wait_for_the_event_turn(self):
        await self.heads_up()
        self.hold = asyncio.Event()
        snap = self.now()
        row = snap["seats"][snap["view"]["turn"]]
        await asyncio.wait_for(self.do(row["ref_id"], SHOVE(self.menu(snap))), 1)
        await self.settle()
        self.assertEqual(len(self.events), 1)
        self.assertGreater(self.now()["moves"], snap["moves"])


def SHOVE_CHOOSER(menu, snap, seat):
    return SHOVE(menu)
