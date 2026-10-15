"""FEAT-37 part D: the blackjack move check in Discord. Offline: a stub runtime, a stubbed decision, fake messages and interactions."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from llmcord_core import games_spoken_discord as gsd
from llmcord_core.spoken.interpret import Decision, Features
from llmcord_core.spoken.onnx_embedder import ModelError
from helpers import command, invoke
from llmcord_core.games_doubt_discord import SPOKEN_OFF, SPOKEN_ON, SPOKEN_SERVER_OFF
from test_games_discord import ALICE, BOB, CARA, CH, G, STALE, Click, GameCase


class Runtime:
    def __init__(self):
        self.texts, self.error, self.gate, self.up, self.ready_gate, self.closed = [], None, None, True, None, False

    def close(self):
        self.closed = True

    async def ready(self):
        if self.ready_gate is not None:
            await self.ready_gate.wait()
        return self.up

    async def embed(self, text):
        self.texts.append(text)
        if self.gate is not None:
            await self.gate.wait()
        if self.error:
            raise self.error
        return [0.0]


class Posted:
    _next = 9000

    def __init__(self, content, view):
        Posted._next += 1
        self.id, self.content, self.view, self.deleted = Posted._next, content, view, False

    async def delete(self):
        self.deleted = True


class Said:
    """A member's chat message in the table channel."""

    def __init__(self, user, text, channel_id=CH, bot=False, webhook=None):
        self.guild, self.channel = SimpleNamespace(id=G), SimpleNamespace(id=channel_id)
        self.author, self.webhook_id, self.content = SimpleNamespace(id=user, bot=bot), webhook, text
        self.posts, self.on_reply = [], None

    async def reply(self, content, view=None, **kwargs):
        post = Posted(content, view)
        if self.on_reply is not None:
            await self.on_reply()
        self.posts.append(post)
        return post


class SpokenClick(Click):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deleted = False

    async def delete_original_response(self):
        self.deleted = True


class SpokenCase(GameCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.spoken = self.bot.spoken
        self.runtime = self.spoken.runtime = Runtime()
        self.decision = Decision("sure", ("hit",))
        self.legal_seen = []

        self.examples_seen = []

        def decide(vector, legal, head, examples=()):
            self.legal_seen.append(sorted(legal))
            self.examples_seen.append(list(examples))
            return self.decision

        for patcher in (mock.patch.object(gsd.interpret, "decide", side_effect=decide), mock.patch.object(gsd.spoken_head, "blackjack_head", return_value=object())):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.spoken.close)
        self.store.set_spoken_moves_enabled(G, True)

    async def deal(self, *ranks):
        self.rig(*(ranks or ("9", "10", "10", "6", "8", "7")))
        await self.bet(ALICE, 10)
        await self.bet(BOB, 10)
        await self.press(BOB, "Deal now")

    async def say(self, user, text="hit me", **kw):
        msg = Said(user, text, **kw)
        self.bot.spoken.note_message(msg)
        await self.settle()
        return msg

    async def settle(self):
        await asyncio.gather(*list(self.spoken.tasks))

    def key(self):
        return (G, self.latest()["table_id"])

    def check_button(self, msg, move):
        children = msg.posts[-1].view.children
        return next(c for c in children if c.move == move and c.action == ("move" if move else "talk"))

    async def click(self, user, msg, move, guild_id=G, post=None):
        click = SpokenClick(self.bot, user, guild_id=guild_id)
        click.message = post or msg.posts[-1]
        await self.check_button(msg, move).callback(click)
        return click

    def labels(self, msg):
        return [c.item.label for c in msg.posts[-1].view.children]


class CheckTests(SpokenCase):
    async def test_sure_offers_the_move_and_table_talk(self):
        await self.deal()
        msg = await self.say(ALICE)
        self.assertEqual(len(msg.posts), 1)
        self.assertEqual(msg.posts[0].content, "Did you mean **Hit**?")
        self.assertEqual(self.labels(msg), ["Hit", "Just table talk"])
        self.assertEqual(self.runtime.texts, ["hit me"])
        self.assertIn(self.key(), self.spoken.pending)

    async def test_unsure_offers_two_moves_and_table_talk(self):
        await self.deal()
        self.decision = Decision("unsure", ("hit", "stand"))
        msg = await self.say(ALICE)
        self.assertEqual(msg.posts[0].content, "Did you mean **Hit** or **Stand**?")
        self.assertEqual(self.labels(msg), ["Hit", "Stand", "Just table talk"])

    async def test_silent_posts_nothing(self):
        await self.deal()
        self.decision = Decision("silent", ())
        msg = await self.say(ALICE)
        self.assertEqual(msg.posts, [])
        self.assertEqual(self.spoken.pending, {})

    async def test_insurance_phase_offers_only_insurance_moves_with_the_cost(self):
        self.store.set_blackjack_rules(G, {"insurance": True})
        await self.deal("9", "10", "A", "6", "8", "7")
        self.assertEqual(self.latest()["phase"], "insurance")
        self.decision = Decision("unsure", ("insure", "no_insurance"))
        msg = await self.say(ALICE, "insure me")
        self.assertEqual(self.legal_seen[-1], ["insure", "no_insurance"])
        self.assertEqual(self.labels(msg), ["Insure (5)", "No insurance", "Just table talk"])
        self.assertEqual(msg.posts[0].content, "Did you mean **Insure** or **No insurance**?")

    async def test_custom_ids_stay_within_the_limit(self):
        await self.deal()
        self.decision = Decision("unsure", ("hit", "stand"))
        msg = await self.say(ALICE)
        for child in msg.posts[0].view.children:
            self.assertLessEqual(len(child.item.custom_id), 100)
            self.assertTrue(child.item.custom_id.startswith("llmcord:bjs:"))


class GateTests(SpokenCase):
    async def test_switch_off_does_nothing_and_never_loads_the_runtime(self):
        await self.deal()
        self.store.set_spoken_moves_enabled(G, False)
        self.spoken.runtime = None
        with mock.patch("llmcord_core.spoken.onnx_embedder.get_runtime") as get:
            msg = await self.say(ALICE)
        get.assert_not_called()
        self.assertEqual((msg.posts, self.spoken.tasks, self.runtime.texts), ([], set(), []))

    async def test_enabled_loads_the_runtime_lazily_on_the_first_message(self):
        await self.deal()
        self.spoken.runtime = None
        with mock.patch("llmcord_core.spoken.onnx_embedder.get_runtime", return_value=self.runtime) as get:
            await self.say(ALICE)
        get.assert_called_once()

    async def test_ignored_before_any_inference(self):
        await self.deal()
        for msg in (Said(BOB, "hit"), Said(CARA, "hit"), Said(ALICE, "hit", bot=True), Said(ALICE, "hit", webhook=5), Said(ALICE, "   "),
                    Said(ALICE, "x" * (gsd.MAX_CHARS + 1)), Said(ALICE, "hit", channel_id=CH + 1)):
            self.spoken.note_message(msg)
        self.assertEqual((self.spoken.tasks, self.runtime.texts), (set(), []))
        msg = Said(ALICE, "x" * gsd.MAX_CHARS)
        self.spoken.note_message(msg)
        self.assertEqual(len(self.spoken.tasks), 1)
        await self.settle()

    async def test_no_table_or_not_playing_is_ignored(self):
        self.spoken.note_message(Said(ALICE, "hit"))
        await self.bet(ALICE, 10)  # joining, nobody on turn
        self.spoken.note_message(Said(ALICE, "hit"))
        self.assertEqual(self.spoken.tasks, set())

    async def test_runtime_unavailable_logs_once_and_posts_nothing(self):
        await self.deal()
        self.runtime.up = False
        with self.assertLogs(level="WARNING") as logs:
            first = await self.say(ALICE)
            second = await self.say(ALICE)
        self.assertEqual((first.posts, second.posts), ([], []))
        self.assertEqual(len([r for r in logs.records if gsd.RUNTIME_OFF in r.getMessage()]), 1)

    async def test_model_error_posts_nothing(self):
        await self.deal()
        self.runtime.error = ModelError("bad vector")
        msg = await self.say(ALICE)
        self.assertEqual(msg.posts, [])

    async def test_model_error_leaves_the_normal_scene_path_running(self):
        await self.deal()
        self.runtime.error = ModelError("bad vector")
        message = SimpleNamespace(guild=SimpleNamespace(id=G), author=SimpleNamespace(id=ALICE, bot=False), webhook_id=None, content="hit me",
                                  channel=SimpleNamespace(id=CH), reference=None, mentions=[])
        message.reply = Said(ALICE, "hit me").reply
        seen = []
        self.bot.location = lambda channel: seen.append(channel.id) or (None, None)
        self.bot.doubt.note_thread_message = mock.AsyncMock()
        await self.bot.on_message(message)
        await self.settle()
        self.assertEqual(seen, [CH])  # on_message went on to location() after scheduling the check

    async def test_scheduling_errors_never_stop_on_message(self):
        message = SimpleNamespace(guild=SimpleNamespace(id=G), author=SimpleNamespace(id=ALICE, bot=False), webhook_id=None, content="x",
                                  channel=SimpleNamespace(id=CH), reference=None, mentions=[])
        seen = []
        self.bot.location = lambda channel: seen.append(1) or (None, None)
        self.bot.doubt.note_thread_message = mock.AsyncMock()
        with mock.patch.object(self.spoken, "note_message", side_effect=RuntimeError("boom")), self.assertLogs(level="ERROR"):
            await self.bot.on_message(message)
        self.assertEqual(seen, [1])


class RaceTests(SpokenCase):
    async def test_move_changed_during_inference_posts_nothing(self):
        await self.deal()
        self.runtime.gate = asyncio.Event()
        msg = Said(ALICE, "hit me")
        self.spoken.note_message(msg)
        await asyncio.sleep(0)
        await self.press(ALICE, "Hit")
        self.runtime.gate.set()
        await self.settle()
        self.assertEqual((msg.posts, self.spoken.pending), ([], {}))

    async def test_member_turned_off_during_inference_posts_and_keeps_nothing(self):
        await self.deal()
        self.runtime.gate = asyncio.Event()
        msg = Said(ALICE, "hit me")
        self.spoken.note_message(msg)
        await asyncio.sleep(0)
        self.store.set_spoken_member_on(G, ALICE, False)
        self.runtime.gate.set()
        await self.settle()
        self.assertEqual((msg.posts, self.spoken.pending, self.spoken.readings), ([], {}, {}))

    async def test_member_turned_off_while_the_reply_is_in_flight_deletes_it(self):
        await self.deal()
        msg = Said(ALICE, "hit me")

        async def off():
            self.store.set_spoken_member_on(G, ALICE, False)

        msg.on_reply = off
        self.spoken.note_message(msg)
        await self.settle()
        self.assertTrue(msg.posts[0].deleted)
        self.assertEqual(self.spoken.pending, {})

    async def test_a_newer_phrase_replaces_the_pending_check(self):
        await self.deal()
        first = await self.say(ALICE, "hit me")
        second = await self.say(ALICE, "one more card")
        self.assertTrue(first.posts[0].deleted)
        self.assertFalse(second.posts[0].deleted)
        self.assertIs(self.spoken.pending[self.key()].message, second.posts[0])

    async def test_an_older_slow_reading_drops_itself_when_a_newer_phrase_arrives(self):
        await self.deal()
        self.runtime.gate = asyncio.Event()
        old, new = Said(ALICE, "hit"), Said(ALICE, "stand")
        self.spoken.note_message(old)
        self.spoken.note_message(new)
        await asyncio.sleep(0)
        self.runtime.gate.set()
        await self.settle()
        self.assertEqual((old.posts, len(new.posts)), ([], 1))

    async def test_turn_end_deletes_the_check(self):
        await self.deal()
        msg = await self.say(ALICE)
        await self.press(ALICE, "Stand")
        await self.settle()
        self.assertTrue(msg.posts[0].deleted)
        self.assertEqual(self.spoken.pending, {})

    async def test_turn_timeout_and_table_close_delete_the_check(self):
        await self.deal()
        msg = await self.say(ALICE)
        snap = self.latest()
        await self.games.on_turn_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        await self.settle()
        self.assertTrue(msg.posts[0].deleted)
        msg = await self.say(ALICE)  # the default policy hit and alice is still on turn
        self.assertEqual(len(msg.posts), 1)
        await self.games.set_channel(G, CH, False)
        await self.settle()
        self.assertTrue(msg.posts[0].deleted)

    async def test_the_turn_index_follows_the_table(self):
        await self.deal()
        key = self.key()
        self.assertEqual(self.games.turns[key].member_id, ALICE)
        self.assertEqual(self.games.turn_channels[(G, CH)], key)
        await self.press(ALICE, "Stand")
        self.assertEqual(self.games.turns[key].member_id, BOB)
        await self.press(BOB, "Stand")
        self.assertEqual((self.games.turns, self.games.turn_channels), ({}, {}))


class PressTests(SpokenCase):
    async def test_another_member_is_refused(self):
        await self.deal()
        msg = await self.say(ALICE)
        before = self.latest()["moves"]
        click = await self.click(BOB, msg, "hit")
        self.assertEqual(click.replies, [gsd.NOT_YOURS])
        self.assertEqual(self.latest()["moves"], before)
        self.assertIn(self.key(), self.spoken.pending)
        talk = await self.click(BOB, msg, None)
        self.assertEqual(talk.replies, [gsd.NOT_YOURS])
        self.assertFalse(msg.posts[0].deleted)

    async def test_the_turn_member_plays_the_move_through_the_normal_path(self):
        await self.deal()
        msg = await self.say(ALICE)
        before = self.latest()["moves"]
        click = await self.click(ALICE, msg, "hit")
        self.assertEqual(self.latest()["moves"], before + 1)
        self.assertEqual(click.response.edited[-1], {"content": "✔ Hit", "view": None})
        self.assertFalse(msg.posts[0].deleted)
        self.assertEqual(self.spoken.pending, {})
        self.assertEqual(self.store.db.execute("SELECT move,actor FROM game_moves ORDER BY id").fetchall()[-1][:], ("hit", "member"))
        self.assertTrue(self.texts())  # the table message was repainted

    async def test_a_stale_press_says_the_table_moved_on(self):
        await self.deal()
        msg = await self.say(ALICE)
        button = self.check_button(msg, "hit")
        self.spoken.pending.clear()  # as after a restart: no record of the check
        await self.press(ALICE, "Hit")  # alice moves on her own
        click = SpokenClick(self.bot, ALICE)
        click.message = msg.posts[0]
        await button.callback(click)
        self.assertEqual(click.replies, [STALE])
        self.assertTrue(click.deleted)

    async def test_table_talk_deletes_the_check_and_answers_privately(self):
        await self.deal()
        msg = await self.say(ALICE)
        before = self.latest()["moves"]
        click = await self.click(ALICE, msg, None)
        self.assertEqual(click.replies, [gsd.TALK_KEPT])
        self.assertTrue(msg.posts[0].deleted)
        self.assertEqual((self.latest()["moves"], self.spoken.pending), (before, {}))


class ReviewTests(SpokenCase):
    async def test_turn_changing_while_the_reply_is_in_flight_deletes_the_own_reply(self):
        await self.deal()
        msg = Said(ALICE, "hit me")
        msg.on_reply = lambda: self.press(ALICE, "Stand")
        self.spoken.note_message(msg)
        await self.settle()
        self.assertTrue(msg.posts[0].deleted)
        self.assertEqual(self.spoken.pending, {})

    async def test_pressing_a_replaced_check_plays_but_talk_deletes_only_the_pressed_one(self):
        await self.deal()
        first = await self.say(ALICE, "hit me")
        second = await self.say(ALICE, "one more card")
        key = self.key()
        talk = await self.click(ALICE, first, None, post=first.posts[0])
        self.assertEqual(talk.replies, [STALE])
        self.assertTrue(first.posts[0].deleted)
        self.assertFalse(second.posts[0].deleted)
        self.assertIs(self.spoken.pending[key].message, second.posts[0])
        before = self.latest()["moves"]
        click = await self.click(ALICE, first, "hit", post=first.posts[0])
        self.assertEqual(self.latest()["moves"], before + 1)
        self.assertEqual(click.response.edited[-1]["content"], "✔ Hit")
        await self.settle()
        self.assertTrue(second.posts[0].deleted)  # by the turn change, not by the old check's press

    async def test_a_bystander_press_with_no_record_leaves_the_check(self):
        await self.deal()
        msg = await self.say(ALICE)
        self.spoken.pending.clear()
        click = await self.click(BOB, msg, "hit")
        self.assertEqual(click.replies, ["It's not your turn."])
        self.assertFalse(click.deleted)

    async def test_close_cancels_an_in_flight_reading_and_closes_the_runtime(self):
        await self.deal()
        self.runtime.gate = asyncio.Event()
        msg = Said(ALICE, "hit me")
        self.spoken.note_message(msg)
        await asyncio.sleep(0)
        await self.spoken.close()
        self.assertTrue(all(t.done() for t in self.spoken.tasks) and msg.posts == [] and self.runtime.closed)

    async def test_a_superseded_reading_does_no_model_work(self):
        await self.deal()
        self.runtime.ready_gate = asyncio.Event()
        self.spoken.note_message(Said(ALICE, "old"))
        self.spoken.note_message(Said(ALICE, "new"))
        await asyncio.sleep(0)
        self.runtime.ready_gate.set()
        await self.settle()
        self.assertEqual(self.runtime.texts, ["new"])

    async def test_the_same_table_id_in_another_guild_does_not_cross(self):
        await self.deal()
        msg = await self.say(ALICE)
        other = Said(ALICE, "hit me")
        other.guild = SimpleNamespace(id=G + 1)
        self.spoken.note_message(other)
        self.assertEqual(self.runtime.texts, ["hit me"])
        before = self.latest()["moves"]
        click = await self.click(ALICE, msg, "hit", guild_id=G + 1)
        self.assertEqual(self.latest()["moves"], before)
        self.assertIn(self.key(), self.spoken.pending)
        self.assertFalse(msg.posts[0].deleted)
        self.assertEqual(len(click.replies), 1)

    async def test_latest_is_pruned_when_the_turn_ends(self):
        await self.deal()
        await self.say(ALICE)
        self.assertIn(self.key(), self.spoken.latest)
        await self.press(ALICE, "Stand")
        await self.press(BOB, "Stand")
        self.assertEqual(self.spoken.latest, {})

    async def test_a_bad_head_warns_once(self):
        await self.deal()
        error = gsd.spoken_head.HeadError("bad head")
        with mock.patch.object(gsd.spoken_head, "blackjack_head", side_effect=error), self.assertLogs(level="WARNING") as logs:
            first, second = await self.say(ALICE), await self.say(ALICE)
        self.assertEqual((first.posts, second.posts, self.runtime.texts), ([], [], []))
        self.assertEqual(len([r for r in logs.records if "bad head" in r.getMessage()]), 1)


FEATURES = Features(0.912345678, 0.5, 0.25)


class KeepCase(SpokenCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.store.set_spoken_keep_phrases(G, True)
        self.sure = Decision("sure", ("hit",), FEATURES)
        self.unsure = Decision("unsure", ("hit", "stand"), FEATURES)
        self.silent = Decision("silent", (), FEATURES)

    def kept(self):
        return self.store.spoken_phrases(G)

    async def table(self, user, label):
        await self.press(user, label)
        await self.settle()


class CaptureTests(KeepCase):
    async def test_talk_press_keeps_a_false_positive(self):
        await self.deal()
        self.decision = self.unsure
        msg = await self.say(ALICE, "maybe later")
        await self.click(ALICE, msg, None)
        (row,) = self.kept()
        self.assertEqual((row["text"], row["kind"], row["label"], row["band"], row["offered"]), ("maybe later", "false_positive", "talk", "unsure", ["hit", "stand"]))
        self.assertEqual(row["scores"], {"p1": 0.9123, "margin": 0.5, "talk_gap": 0.25})

    async def test_unsure_second_move_keeps_a_false_positive_but_the_first_does_not(self):
        await self.deal()
        self.decision = self.unsure
        msg = await self.say(ALICE, "hmm")
        await self.click(ALICE, msg, "hit")
        self.assertEqual(self.kept(), [])
        msg = await self.say(ALICE, "hmm ok")
        await self.click(ALICE, msg, "stand")
        (row,) = self.kept()
        self.assertEqual((row["text"], row["kind"], row["label"]), ("hmm ok", "false_positive", "stand"))

    async def test_a_sure_check_answered_with_its_move_keeps_nothing(self):
        await self.deal()
        self.decision = self.sure
        msg = await self.say(ALICE)
        await self.click(ALICE, msg, "hit")
        self.assertEqual(self.kept(), [])

    async def test_a_silent_reading_then_a_table_button_keeps_a_false_negative(self):
        await self.deal()
        self.decision = self.silent
        await self.say(ALICE, "ok I'll stay put")
        await self.table(ALICE, "Stand")
        (row,) = self.kept()
        self.assertEqual((row["text"], row["kind"], row["label"], row["band"], row["offered"]), ("ok I'll stay put", "false_negative", "stand", "silent", []))
        self.assertEqual(row["scores"]["p1"], 0.9123)

    async def test_a_move_the_check_did_not_offer_keeps_a_false_negative(self):
        await self.deal()
        self.decision = self.sure
        msg = await self.say(ALICE, "gimme")
        await self.table(ALICE, "Stand")
        (row,) = self.kept()
        self.assertEqual((row["kind"], row["label"], row["band"], row["offered"]), ("false_negative", "stand", "sure", ["hit"]))
        self.assertTrue(msg.posts[0].deleted)

    async def test_a_table_button_of_an_offered_move_keeps_nothing(self):
        await self.deal()
        self.decision = self.unsure
        await self.say(ALICE, "hmm")
        await self.table(ALICE, "Hit")
        self.assertEqual(self.kept(), [])

    async def test_a_later_turn_without_a_reading_keeps_nothing(self):
        await self.deal()
        self.decision = self.silent
        await self.say(ALICE, "pass")
        await self.table(ALICE, "Hit")
        self.assertEqual(len(self.kept()), 1)
        await self.table(ALICE, "Stand")
        await self.table(BOB, "Stand")
        self.assertEqual((len(self.kept()), self.spoken.ended), (1, {}))

    async def test_hook_needs_the_same_round_moves_and_member(self):
        from llmcord_core.games_discord import Turn
        key = (G, 1)
        reading = lambda: gsd._Reading(Turn(5, 2, ALICE, CH), "t", "silent", (), {"p1": 0.1})
        for args in ((6, 2, ALICE), (5, 3, ALICE), (5, 2, BOB)):
            self.spoken.ended[key] = reading()
            self.assertIsNone(self.spoken.claim_table_move(key, *args, "hit"), args)
        self.spoken.ended[key] = reading()
        self.assertIsNone(self.spoken.claim_table_move(key, 5, 2, ALICE, "deal"))
        self.spoken.ended[key] = reading()
        claim = self.spoken.claim_table_move(key, 5, 2, ALICE, "hit")
        self.assertEqual(self.kept(), [])  # claiming is memory only
        self.spoken.keep_claimed(claim)
        self.assertEqual(len(self.kept()), 1)
        self.assertEqual(self.spoken.ended, {})

    async def test_a_reading_still_in_flight_keeps_nothing(self):
        await self.deal()
        self.decision = self.silent
        await self.say(ALICE, "old phrase")
        self.runtime.gate = asyncio.Event()
        self.spoken.note_message(Said(ALICE, "new phrase"))
        await asyncio.sleep(0)
        await self.press(ALICE, "Stand")
        self.runtime.gate.set()
        await self.settle()
        self.assertEqual(self.kept(), [])

    async def test_a_turn_timeout_keeps_nothing(self):
        await self.deal()
        self.decision = self.silent
        await self.say(ALICE, "thinking")
        snap = self.latest()
        await self.games.on_turn_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        await self.settle()
        self.assertEqual(self.kept(), [])

    async def test_nothing_is_remembered_or_kept_when_the_switch_is_off(self):
        await self.deal()
        self.store.set_spoken_moves_enabled(G, False)
        await self.say(ALICE, "pass")
        await self.table(ALICE, "Stand")
        self.assertEqual((self.spoken.readings, self.spoken.ended, self.kept()), ({}, {}, []))

    async def test_keeping_off_or_member_off_stores_nothing(self):
        await self.deal()
        self.decision = self.unsure
        msg = await self.say(ALICE, "hmm")
        self.store.set_spoken_keep_phrases(G, False)
        await self.click(ALICE, msg, None)
        self.assertEqual(self.kept(), [])
        self.store.set_spoken_keep_phrases(G, True)
        self.store.set_spoken_member_on(G, ALICE, False)
        self.decision = self.silent
        await self.say(ALICE, "pass")
        self.assertNotIn(self.key(), self.spoken.readings)
        await self.table(ALICE, "Stand")
        self.assertEqual(self.kept(), [])

    async def test_the_same_scenario_with_the_member_on_keeps_a_row(self):
        await self.deal()
        self.decision = self.silent
        await self.say(ALICE, "pass")
        await self.table(ALICE, "Stand")
        self.assertEqual(len(self.kept()), 1)

    async def test_talk_press_then_a_table_move_keeps_only_the_false_positive(self):
        await self.deal()
        self.decision = self.sure
        msg = await self.say(ALICE, "maybe")
        await self.click(ALICE, msg, None)
        await self.table(ALICE, "Stand")
        (row,) = self.kept()
        self.assertEqual((row["kind"], row["label"]), ("false_positive", "talk"))

    async def test_the_keep_comes_after_the_answer(self):
        await self.deal()
        self.decision = self.silent
        await self.say(ALICE, "pass")
        order, real, finish = [], self.store.keep_spoken_phrase, self.games._finish

        async def late(*a, **k):
            await finish(*a, **k)
            order.append("finish")

        self.games._finish = late
        self.store.keep_spoken_phrase = lambda *a: order.append("keep") or real(*a)
        await self.table(ALICE, "Stand")
        self.assertEqual(order, ["finish", "keep"])

    async def test_a_failing_keep_never_breaks_the_press(self):
        await self.deal()
        self.decision = self.unsure
        msg = await self.say(ALICE, "hmm")
        before = self.latest()["moves"]
        with mock.patch.object(self.store, "keep_spoken_phrase", side_effect=RuntimeError("db")), self.assertLogs(level="ERROR"):
            click = await self.click(ALICE, msg, "stand")
        self.assertEqual((self.latest()["moves"], click.response.edited[-1]["content"]), (before + 1, "✔ Stand"))
        self.decision = self.silent
        await self.say(BOB, "pass")
        with mock.patch.object(self.store, "keep_spoken_phrase", side_effect=RuntimeError("db")), self.assertLogs(level="ERROR"):
            await self.table(BOB, "Stand")
        self.assertEqual(self.latest()["status"], "settled")

    async def test_keep_runs_outside_the_table_lock(self):
        await self.deal()
        self.decision = self.silent
        await self.say(ALICE, "pass")
        locked = []
        real = self.store.keep_spoken_phrase
        self.store.keep_spoken_phrase = lambda *a: locked.append(self.games.runtime.lock(self.key()).locked()) or real(*a)
        await self.table(ALICE, "Stand")
        self.assertEqual(locked, [False])

    async def test_memory_is_bounded_and_dropped_with_the_table(self):
        await self.deal()
        self.decision = self.silent
        await self.say(ALICE, "pass")
        key = self.key()
        self.assertIn(key, self.spoken.readings)
        snap = self.latest()
        await self.games.on_turn_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        self.assertEqual((list(self.spoken.readings), list(self.spoken.ended)), ([], [key]))
        await self.games.set_channel(G, CH, False)
        self.assertEqual((self.spoken.readings, self.spoken.ended), ({}, {}))


class ExampleCacheTests(KeepCase):
    def add(self, text, label):
        ident = self.store.keep_spoken_phrase(G, ALICE, "blackjack", text, [], "silent", {}, "false_negative", label)
        return self.store.promote_spoken_phrase(G, ident, label, ALICE)

    async def test_no_examples_pass_an_empty_list_and_embed_nothing_extra(self):
        await self.deal()
        await self.say(ALICE, "hit me")
        self.assertEqual((self.examples_seen, self.runtime.texts), ([[]], ["hit me"]))
        self.assertEqual(self.spoken.examples, {})

    async def test_examples_are_embedded_once_per_revision(self):
        await self.deal()
        self.add("one more", "hit")
        await self.say(ALICE, "a")
        self.assertEqual(self.runtime.texts, ["one more", "a"])
        self.assertEqual(self.examples_seen[-1], [([0.0], "hit")])
        await self.say(ALICE, "b")
        self.assertEqual(self.runtime.texts, ["one more", "a", "b"])
        self.add("chat", "talk")
        await self.say(ALICE, "c")
        self.assertEqual(self.runtime.texts, ["one more", "a", "b", "chat", "c"])  # only the new id
        self.assertEqual(self.examples_seen[-1], [([0.0], "hit"), ([0.0], "talk")])

    async def test_a_deleted_example_is_dropped_from_the_cache(self):
        await self.deal()
        first, second = self.add("one more", "hit"), self.add("chat", "talk")
        await self.say(ALICE, "a")
        self.store.delete_spoken_example(G, first)
        await self.say(ALICE, "b")
        self.assertEqual(self.examples_seen[-1], [([0.0], "talk")])
        self.assertEqual([i for i, _, _ in self.spoken.examples[G][1]], [second])

    async def test_a_failed_embed_is_retried_next_time(self):
        await self.deal()
        self.add("one more", "hit")
        self.runtime.error = ModelError("bad")
        await self.say(ALICE, "a")
        self.runtime.error = None
        await self.say(ALICE, "b")
        self.assertEqual(self.examples_seen[-1], [([0.0], "hit")])


class RegistrationTests(unittest.TestCase):
    def test_template_matches_both_shapes_and_not_the_table_buttons(self):
        t = gsd.SpokenButton.__discord_ui_compiled_template__
        self.assertTrue(t.fullmatch("llmcord:bjs:move:3:4:5:hit"))
        self.assertTrue(t.fullmatch("llmcord:bjs:talk:3:4:5"))
        self.assertFalse(t.fullmatch("llmcord:bj:hit:3:4:5"))

    def test_bot_registers_the_dynamic_item(self):
        import inspect
        from llmcord_core import discord_bot
        self.assertIn("SpokenButton", inspect.getsource(discord_bot.SkitBot.setup_hook))

class MemberSwitchTests(SpokenCase):
    async def switch(self, user, setting, guild_id=G):
        click = Click(self.bot, user, guild_id=guild_id)
        await invoke(self.bot, "game spoken_moves", click, setting=setting)
        return click

    def keep(self, guild, user, text):
        self.store.set_spoken_keep_phrases(guild, True)
        return self.store.keep_spoken_phrase(guild, user, "blackjack", text, ["hit"], "unsure", {}, "false_positive", "talk")

    def test_registered_under_game_with_on_and_off_choices(self):
        cmd = command(self.bot, "game spoken_moves")
        self.assertEqual(cmd.description, "Turn move checks from your chat on or off for you")
        self.assertEqual([(c.name, c.value) for c in cmd._params["setting"].choices], [("On", "on"), ("Off", "off")])
        self.assertLessEqual(len(self.bot.tree.get_command("game").commands), 25)

    async def test_off_stores_off_and_purges_only_that_members_phrases(self):
        self.store.set_spoken_moves_enabled(2, True)
        self.keep(G, ALICE, "mine"), self.keep(G, BOB, "his"), self.keep(2, ALICE, "other server")
        click = await self.switch(ALICE, "off")
        self.assertEqual(click.replies, [SPOKEN_OFF])
        self.assertFalse(self.store.spoken_member_on(G, ALICE))
        self.assertEqual([p["text"] for p in self.store.spoken_phrases(G)], ["his"])
        self.assertEqual([p["text"] for p in self.store.spoken_phrases(2)], ["other server"])
        self.assertTrue(self.store.spoken_member_on(G, BOB) and self.store.spoken_member_on(2, ALICE))

    async def test_on_stores_on(self):
        await self.switch(ALICE, "off")
        click = await self.switch(ALICE, "on")
        self.assertEqual(click.replies, [SPOKEN_ON])
        self.assertTrue(self.store.spoken_member_on(G, ALICE))

    async def test_server_switch_off_adds_one_note(self):
        self.store.set_spoken_moves_enabled(G, False)
        self.assertEqual((await self.switch(ALICE, "on")).replies, [SPOKEN_ON + SPOKEN_SERVER_OFF])
        self.assertEqual((await self.switch(ALICE, "off")).replies, [SPOKEN_OFF + SPOKEN_SERVER_OFF])

    async def test_no_server_note_while_either_server_switch_is_on(self):
        self.store.set_spoken_moves_enabled(G, False)
        self.store.set_spoken_doubt_enabled(G, True)
        self.assertEqual((await self.switch(ALICE, "on")).replies, [SPOKEN_ON])
        self.store.set_spoken_doubt_enabled(G, False)
        self.store.set_spoken_moves_enabled(G, True)
        self.assertEqual((await self.switch(ALICE, "off")).replies, [SPOKEN_OFF])

    async def test_member_off_is_ignored_without_touching_the_runtime(self):
        await self.deal()
        await self.switch(ALICE, "off")
        with mock.patch.object(self.spoken, "_spawn") as spawn:
            self.spoken.note_message(Said(ALICE, "hit me"))
        spawn.assert_not_called()
        self.assertEqual((self.runtime.texts, self.spoken.latest, self.spoken.pending), ([], {}, {}))
        self.assertEqual(len((await self.say(BOB, "hit me")).posts) + len((await self.say(ALICE)).posts), 0)

    async def test_an_open_check_still_works_after_the_member_turns_off(self):
        await self.deal()
        msg = await self.say(ALICE)
        await self.switch(ALICE, "off")
        self.assertIn(self.key(), self.spoken.pending)
        click = await self.click(ALICE, msg, "hit")
        self.assertNotEqual(click.replies, [STALE])

    async def test_talk_reply_points_to_the_switch(self):
        await self.deal()
        msg = await self.say(ALICE)
        click = await self.click(ALICE, msg, None)
        self.assertEqual(click.replies, ["Got it \u2014 left as table talk. Tired of these? Turn them off with /game spoken_moves."])



if __name__ == "__main__":
    unittest.main()
