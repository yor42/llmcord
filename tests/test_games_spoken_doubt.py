"""FEAT-38 part B: spoken doubts in I Doubt It table threads (D36). Offline: fake threads, messages and clicks; a real in-memory store."""
import asyncio
import unittest
from unittest import mock

import discord
from helpers import FakeGameClick, FakeThread, thread_message
from test_doubt_discord import DAN, G, MEMBERS, DoubtCase, button, item_of

from llmcord_core import games_spoken_doubt as gsd
from llmcord_core import games_ui
from llmcord_core.admin_store import ConflictError
from llmcord_core.games import registry
from llmcord_core.games_discord import STALE
from llmcord_core.store import Store
from llmcord_core.games_doubt_discord import DOUBTED_FIRST, MOVED_ON


class Chat:
    """A member's message in the table thread, as ``on_message`` hands it over, able to take a reply."""

    def __init__(self, thread, user, text, **kw):
        self.msg = thread_message(thread, user, text, **kw)
        self.msg.reply = self.reply
        self.thread, self.posts = thread, []

    async def reply(self, content, view=None, **kwargs):
        post = await self.thread.send(content, view=view, **kwargs)
        self.posts.append(post)
        return post

    def __getattr__(self, name):
        return getattr(self.msg, name)


def make_click(case, user, post, guild_id=G):
    click = case.click(user, channel=case.thread(), message=post)

    async def delete_original_response():
        click.message.deleted = True

    click.delete_original_response = delete_original_response
    click.guild_id = guild_id
    return click


class SpokenDoubtCase(DoubtCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.spoken = self.bot.spoken_doubt
        self.addAsyncCleanup(self.spoken.close)
        self.store.set_spoken_doubt_enabled(G, True)

    async def window(self):
        """A three-member table with a doubt window open on the first player's claim. Returns that player's seat."""
        await self.playing()
        seat = self.snap()["view"]["turn"]
        await self.pick_play(MEMBERS[seat], self.any_card(seat))
        self.assertIsNotNone(self.snap()["view"]["window"])
        return seat

    def say(self, user, text="I doubt it", **kw):
        chat = Chat(self.thread(), user, text, **kw)
        self.bot.spoken_doubt.note_message(chat)
        return chat

    async def settle(self):
        while pending := [t for t in self.spoken.tasks if not t.done()]:
            await asyncio.gather(*pending)
        await asyncio.sleep(0)

    async def said(self, user, text="I doubt it", **kw):
        chat = self.say(user, text, **kw)
        await self.settle()
        return chat

    def doubter(self, claimer):
        return next(u for i, u in enumerate(MEMBERS) if i != claimer)

    def others(self, claimer):
        return [u for i, u in enumerate(MEMBERS) if i != claimer]

    async def press(self, user, post, label):
        click = make_click(self, user, post)
        await button(post, label).callback(click)
        return click

    def last_check(self, chat):
        return chat.posts[-1]


class SwitchTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)

    def test_default_off_and_round_trip(self):
        self.assertIs(self.store.spoken_doubt_enabled(G), False)
        self.assertIs(self.store.set_spoken_doubt_enabled(G, True), True)
        self.assertIs(self.store.spoken_doubt_enabled(G), True)
        self.assertIs(self.store.set_spoken_doubt_enabled(G, False, expected=True), False)
        self.assertIs(self.store.spoken_doubt_enabled(G), False)

    def test_stale_expected_raises_and_non_bool_is_refused(self):
        with self.assertRaises(ConflictError):
            self.store.set_spoken_doubt_enabled(G, True, expected=True)
        with self.assertRaises(ValueError):
            self.store.set_spoken_doubt_enabled(G, 1)

    def test_guild_isolation_and_independence_from_the_moves_switch(self):
        self.store.set_spoken_doubt_enabled(G, True)
        self.assertIs(self.store.spoken_doubt_enabled(2), False)
        self.assertIs(self.store.spoken_moves_enabled(G), False)
        self.store.set_spoken_moves_enabled(G, True)
        self.store.set_spoken_doubt_enabled(G, False)
        self.assertIs(self.store.spoken_moves_enabled(G), True)

    def test_the_pseudo_row_is_not_a_game(self):
        self.store.set_spoken_doubt_enabled(G, True)
        self.assertNotIn(self.store._SPOKEN_DOUBT_KEY, registry.keys())
        self.assertNotIn(self.store._SPOKEN_DOUBT_KEY, games_ui.GAME_SECTIONS)


class GateTests(SpokenDoubtCase):
    async def ignored(self, user, text="I doubt it", **kw):
        with mock.patch.object(self.spoken, "_spawn") as spawn:
            self.say(user, text, **kw)
        spawn.assert_not_called()
        self.assertEqual((self.spoken.pending, self.spoken.latest), ({}, {}))

    async def test_switch_off(self):
        seat = await self.window()
        self.store.set_spoken_doubt_enabled(G, False)
        await self.ignored(self.doubter(seat))

    async def test_member_off(self):
        seat = await self.window()
        user = self.doubter(seat)
        self.store.set_spoken_member_on(G, user, False)
        await self.ignored(user)

    async def test_no_window_means_no_store_read(self):
        await self.playing()
        seat = self.snap()["view"]["turn"]
        with mock.patch.object(self.store, "latest_round") as latest, mock.patch.object(self.store, "spoken_doubt_enabled") as enabled:
            await self.ignored(self.doubter(seat))
        latest.assert_not_called()
        enabled.assert_not_called()

    async def test_no_doubt_word_means_no_store_read(self):
        seat = await self.window()
        with mock.patch.object(self.store, "latest_round") as latest, mock.patch.object(self.store, "spoken_doubt_enabled") as enabled:
            await self.ignored(self.doubter(seat), "nice weather today")
        latest.assert_not_called()
        enabled.assert_not_called()

    async def test_the_claimer_is_ignored(self):
        seat = await self.window()
        await self.ignored(MEMBERS[seat])

    async def test_a_member_not_seated_is_ignored(self):
        await self.window()
        await self.ignored(DAN)

    async def test_bots_and_webhooks_are_ignored(self):
        seat = await self.window()
        await self.ignored(self.doubter(seat), bot=True)
        chat = Chat(self.thread(), self.doubter(seat), "I doubt it")
        chat.msg.webhook_id = 5
        with mock.patch.object(self.spoken, "_spawn") as spawn:
            self.spoken.note_message(chat)
        spawn.assert_not_called()

    async def test_another_guilds_thread_is_ignored(self):
        seat = await self.window()
        await self.ignored(self.doubter(seat), guild_id=2)

    async def test_a_plain_channel_and_an_unknown_thread_are_ignored(self):
        seat = await self.window()
        user = self.doubter(seat)
        chat = Chat(self.ch, user, "I doubt it")
        with mock.patch.object(self.spoken, "_spawn") as spawn:
            self.spoken.note_message(chat)
        spawn.assert_not_called()
        before = dict(self.doubt.tables)
        thread = FakeThread(999999, parent_id=self.ch.id)
        thread.next_id = lambda: 1
        stranger = Chat(thread, user, "I doubt it")
        with mock.patch.object(self.spoken, "_spawn") as spawn:
            self.spoken.note_message(stranger)
        spawn.assert_not_called()
        self.assertEqual(self.doubt.tables, before)

    async def test_long_text_is_ignored(self):
        seat = await self.window()
        await self.ignored(self.doubter(seat), "I doubt it " + "x" * 200)

    async def test_the_message_still_reaches_the_bot_flow(self):
        seat = await self.window()
        chat = Chat(self.thread(), self.doubter(seat), "I doubt it")
        with mock.patch.object(self.bot.spoken, "note_message") as moves, mock.patch.object(self.doubt, "note_thread_message", new=mock.AsyncMock()) as talk:
            await self.bot.on_message(chat)
        moves.assert_called_once()
        talk.assert_awaited_once()
        await self.settle()
        self.assertEqual(len(chat.posts), 1)

    async def test_a_failing_check_never_raises_into_on_message(self):
        seat = await self.window()
        chat = Chat(self.thread(), self.doubter(seat), "I doubt it")
        with mock.patch.object(self.spoken, "note_message", side_effect=RuntimeError("boom")):
            await self.bot.on_message(chat)


class CheckTests(SpokenDoubtCase):
    async def test_posts_the_check_with_the_right_buttons_and_ids(self):
        seat = await self.window()
        user = self.doubter(seat)
        chat = await self.said(user)
        post = self.last_check(chat)
        self.assertEqual(post.content, "Did you mean **Doubt it!**?")
        snap = self.snap()
        ids = [item_of(c).custom_id for c in post.view.children]
        base = f"{snap['table_id']}:{snap['round_id']}:{snap['moves']}:{user}"
        self.assertEqual(ids, [f"llmcord:dts:doubt:{base}", f"llmcord:dts:talk:{base}"])
        self.assertEqual([item_of(c).label for c in post.view.children], ["Doubt it!", "Just table talk"])
        self.assertEqual([item_of(c).style for c in post.view.children], [discord.ButtonStyle.danger, discord.ButtonStyle.secondary])
        self.assertTrue(all(gsd.SpokenDoubtButton.__discord_ui_compiled_template__.fullmatch(i) for i in ids))

    async def test_a_question_is_also_offered(self):
        seat = await self.window()
        chat = await self.said(self.doubter(seat), "do you really have those?? I doubt it?")
        self.assertEqual(len(chat.posts), 1)

    async def test_a_second_phrase_replaces_the_first(self):
        seat = await self.window()
        user = self.doubter(seat)
        first = await self.said(user)
        second = await self.said(user, "doubt it!")
        await self.settle()
        self.assertTrue(self.last_check(first).deleted)
        self.assertFalse(self.last_check(second).deleted)
        self.assertEqual(len(self.spoken.pending), 1)

    async def test_two_members_get_one_check_each(self):
        seat = await self.window()
        a, b = self.others(seat)
        one, two = await self.said(a), await self.said(b)
        self.assertFalse(self.last_check(one).deleted or self.last_check(two).deleted)
        self.assertEqual(len(self.spoken.pending), 2)

    async def test_member_turned_off_before_the_post_or_during_it_leaves_no_check(self):
        seat = await self.window()
        user = self.doubter(seat)
        chat = self.say(user)
        self.store.set_spoken_member_on(G, user, False)
        await self.settle()
        self.assertEqual((chat.posts, self.spoken.pending), ([], {}))
        self.store.set_spoken_member_on(G, user, True)
        chat = Chat(self.thread(), user, "I doubt it")
        reply = chat.reply

        async def flip(content, view=None, **kwargs):
            post = await reply(content, view=view, **kwargs)
            self.store.set_spoken_member_on(G, user, False)
            return post

        chat.reply = flip
        self.spoken.note_message(chat)
        await self.settle()
        self.assertTrue(chat.posts[-1].deleted)
        self.assertEqual(self.spoken.pending, {})

    async def test_a_check_posted_after_the_window_moved_on_is_deleted(self):
        seat = await self.window()
        user = self.doubter(seat)
        chat = Chat(self.thread(), user, "I doubt it")
        gate = asyncio.Event()
        reply = chat.reply

        async def slow(content, view=None, **kwargs):
            post = await reply(content, view=view, **kwargs)
            await gate.wait()
            return post

        chat.msg.reply = slow
        self.spoken.note_message(chat)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await self.doubt.on_window_timer(G, self.snap()["table_id"], self.snap()["round_id"], self.snap()["moves"])
        gate.set()
        await self.settle()
        self.assertTrue(chat.posts[-1].deleted)
        self.assertEqual(self.spoken.pending, {})

    async def test_no_discord_call_is_made_under_a_table_lock(self):
        seat = await self.window()
        user = self.doubter(seat)
        from test_doubt_discord import HTTP
        HTTP.clear()
        chat = await self.said(user)
        await self.press(user, self.last_check(chat), "Doubt it!")
        await self.settle()
        self.assertTrue(HTTP)
        self.assertFalse([kind for kind, held in HTTP if held])


class PressTests(SpokenDoubtCase):
    async def test_the_doubt_press_doubts_once_and_edits_the_check(self):
        seat = await self.window()
        user = self.doubter(seat)
        chat = await self.said(user)
        post = self.last_check(chat)
        before = self.snap()["moves"]
        click = await self.press(user, post, "Doubt it!")
        self.assertEqual(self.snap()["moves"], before + 1)
        self.assertEqual(self.moves()[-1][1], "doubt")
        self.assertEqual(post.content, "✔ Doubt it!")
        self.assertIsNone(post.view)
        self.assertFalse(post.deleted)
        self.assertEqual(click.response.sent + click.followup.sent, [])

    async def test_the_second_member_gets_doubted_first_and_the_check_goes(self):
        seat = await self.window()
        a, b = self.others(seat)
        one, two = await self.said(a), await self.said(b)
        await self.press(a, self.last_check(one), "Doubt it!")
        click = await self.press(b, self.last_check(two), "Doubt it!")
        await self.settle()
        self.assertEqual([c for c, _ in click.followup.sent], [DOUBTED_FIRST])
        self.assertTrue(self.last_check(two).deleted)
        self.assertEqual([m[1] for m in self.moves()].count("doubt"), 1)

    async def test_the_window_closing_by_a_doubt_deletes_the_others_checks(self):
        seat = await self.window()
        a, b = self.others(seat)
        one, two = await self.said(a), await self.said(b)
        await self.press(a, self.last_check(one), "Doubt it!")
        await self.settle()
        self.assertTrue(self.last_check(two).deleted)
        self.assertFalse(self.last_check(one).deleted)
        self.assertEqual(self.spoken.pending, {})

    async def test_someone_elses_press_gets_not_yours_and_changes_nothing(self):
        seat = await self.window()
        a, b = self.others(seat)
        chat = await self.said(a)
        post = self.last_check(chat)
        before = self.snap()["moves"]
        click = await self.press(b, post, "Doubt it!")
        self.assertEqual(click.response.sent[0][0], gsd.NOT_YOURS)
        self.assertTrue(click.response.sent[0][1]["ephemeral"])
        self.assertEqual(self.snap()["moves"], before)
        self.assertFalse(post.deleted)
        self.assertEqual(len(self.spoken.pending), 1)
        click = await self.press(b, post, "Just table talk")
        self.assertEqual(click.response.sent[0][0], gsd.NOT_YOURS)
        self.assertFalse(post.deleted)

    async def test_talk_press_deletes_the_check_with_a_private_hint(self):
        seat = await self.window()
        user = self.doubter(seat)
        chat = await self.said(user)
        post = self.last_check(chat)
        before = self.snap()["moves"]
        click = await self.press(user, post, "Just table talk")
        self.assertTrue(post.deleted)
        self.assertEqual(click.response.sent[0][0], gsd.TALK_KEPT)
        self.assertTrue(click.response.sent[0][1]["ephemeral"])
        self.assertEqual(self.snap()["moves"], before)
        self.assertEqual(self.spoken.pending, {})

    async def test_a_lingering_check_of_an_old_window_is_refused_and_deleted(self):
        seat = await self.window()
        user = self.doubter(seat)
        chat = await self.said(user)
        post = self.last_check(chat)
        snap = self.snap()
        self.spoken.pending.clear()  # as if the drop lost the race: the button still arrives
        await self.doubt.on_window_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        click = await self.press(user, post, "Doubt it!")
        self.assertEqual([c for c, _ in click.followup.sent], [MOVED_ON])
        self.assertTrue(post.deleted)

    async def test_a_talk_press_on_an_untracked_check_is_kept_as_table_talk(self):
        seat = await self.window()
        user = self.doubter(seat)
        post = self.last_check(await self.said(user))
        self.spoken.pending.clear()
        click = await self.press(user, post, "Just table talk")
        self.assertEqual(click.response.sent[0][0], gsd.TALK_KEPT)
        self.assertTrue(post.deleted)

    async def test_a_failing_doubt_press_is_logged_and_deletes_the_check(self):
        seat = await self.window()
        user = self.doubter(seat)
        post = self.last_check(await self.said(user))
        with mock.patch.object(self.doubt, "press_doubt", side_effect=RuntimeError("boom")), self.assertLogs(level="ERROR"):
            await self.press(user, post, "Doubt it!")
        self.assertTrue(post.deleted)

    async def test_press_doubt_returns_stale_without_speaking_for_a_bot(self):
        click = FakeGameClick(self.bot, 99, channel=self.ch)
        click.user.bot = True
        self.assertEqual(await self.doubt.press_doubt(click, 1, 1, 0), STALE)
        self.assertEqual(click.response.sent + click.followup.sent, [])

    async def test_the_claimer_pressing_is_refused_with_the_table_text(self):
        seat = await self.window()
        snap = self.snap()
        post = await self.thread().send("x", view=discord.ui.View())
        click = make_click(self, MEMBERS[seat], post)
        await self.spoken.press(click, "doubt", snap["table_id"], snap["round_id"], snap["moves"], MEMBERS[seat])
        self.assertEqual(click.followup.sent[0][0], "You can't doubt your own claim.")
        self.assertTrue(post.deleted)


class WindowTests(SpokenDoubtCase):
    async def test_the_window_timer_deletes_pending_checks(self):
        seat = await self.window()
        chat = await self.said(self.doubter(seat))
        snap = self.snap()
        await self.doubt.on_window_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        await self.settle()
        self.assertTrue(self.last_check(chat).deleted)
        self.assertEqual((self.spoken.pending, self.spoken.windows), ({}, {}))

    async def test_a_play_closes_the_window_and_deletes_pending_checks(self):
        seat = await self.window()
        chat = await self.said(self.doubter(seat))
        next_seat = self.snap()["view"]["turn"]
        await self.pick_play(MEMBERS[next_seat], self.any_card(next_seat))
        await self.settle()
        self.assertTrue(self.last_check(chat).deleted)

    async def test_the_timer_is_not_extended_by_a_pending_check(self):
        seat = await self.window()
        table = self.doubt.tables[next(iter(self.doubt.tables))]
        deadline = table.deadlines["window"]
        await self.said(self.doubter(seat))
        self.assertEqual(table.deadlines["window"], deadline)

    async def test_closing_the_table_drops_the_checks(self):
        seat = await self.window()
        chat = await self.said(self.doubter(seat))
        key = (G, self.snap()["table_id"])
        self.doubt._forget(key)
        await self.settle()
        self.assertTrue(self.last_check(chat).deleted)
        self.assertEqual((self.spoken.pending, self.spoken.windows), ({}, {}))

    async def test_close_for_channel_drops_the_checks(self):
        seat = await self.window()
        chat = await self.said(self.doubter(seat))
        await self.doubt.close_for_channel(G, self.ch.id, self.snap()["table_id"])
        await self.settle()
        self.assertTrue(self.last_check(chat).deleted)

    async def test_close_cancels_tasks_and_clears(self):
        seat = await self.window()
        await self.said(self.doubter(seat))
        await self.spoken.close()
        self.assertEqual((self.spoken.pending, self.spoken.windows, self.spoken.latest, self.spoken.tasks), ({}, {}, {}, set()))
        self.spoken.note_message(Chat(self.thread(), self.doubter(seat), "I doubt it"))
        self.assertEqual(self.spoken.latest, {})


class RegistrationTests(unittest.TestCase):
    def test_template_matches_its_shape_and_not_the_other_buttons(self):
        t = gsd.SpokenDoubtButton.__discord_ui_compiled_template__
        self.assertTrue(t.fullmatch("llmcord:dts:doubt:3:4:5:6"))
        self.assertTrue(t.fullmatch("llmcord:dts:talk:3:4:5:6"))
        self.assertFalse(t.fullmatch("llmcord:dts:doubt:3:4:5"))
        self.assertFalse(t.fullmatch("llmcord:bjs:talk:3:4:5"))
        self.assertFalse(t.fullmatch("llmcord:doubt:doubt:3:4:5"))

    def test_bot_registers_the_dynamic_item_and_closes_the_checker(self):
        import inspect
        from llmcord_core import discord_bot
        self.assertIn("SpokenDoubtButton", inspect.getsource(discord_bot.SkitBot.setup_hook))
        self.assertIn("spoken_doubt.close", inspect.getsource(discord_bot.SkitBot.close))


if __name__ == "__main__":
    unittest.main()
