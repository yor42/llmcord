"""FEAT-37 part D: the blackjack move check in Discord. Offline: a stub runtime, a stubbed decision, fake messages and interactions."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from llmcord_core import games_spoken_discord as gsd
from llmcord_core.spoken.interpret import Decision
from llmcord_core.spoken.onnx_embedder import ModelError
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

        def decide(vector, legal, head):
            self.legal_seen.append(sorted(legal))
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

    async def test_outcomes_go_through_the_seam(self):
        await self.deal()
        msg = await self.say(ALICE)
        with mock.patch.object(self.spoken, "_note_outcome") as seam:
            await self.click(ALICE, msg, None)
        seam.assert_called_once_with(G, "hit me", ("hit",), "talk")


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


if __name__ == "__main__":
    unittest.main()
