"""FEAT-49b3: opt-in and consent for the decision-model veto (docs/engineering/research/spoken-veto-design.md, "Opt-in and consent").

Written ahead of the implementation. Names these tests require:

* ``/game spoken_veto setting`` (member): ``setting`` choices On/Off/Status (values ``on``/``off``/``status``); stores through
  ``store.set_spoken_veto_member_on``; an ephemeral reply in the invoker's language that ALWAYS carries the notice line.
* ``/admin games spoken_veto setting`` (administrator): ``setting`` choices On/Off; stores through ``store.set_spoken_veto_enabled``;
  an ephemeral reply with the disclosure sentence naming "OpenRouter (Microsoft Decision-1)". When the operator has not configured the
  veto the reply also says it has no effect until the operator configures it. "Configured" is ``bot.spoken.veto is not None``
  (``SpokenMoves(bot, env=...)`` builds it with ``veto_client.from_env``); the tests also patch ``os.environ`` to agree.
* ``SpokenMoves.veto_notified``: a set of ``(guild_id, user_id)`` held in memory only. The first message of a member who passes every
  veto gate (including an e5 reading that is not silent) posts the one-line notice in the channel (``message.reply``, ``channel_t``
  at render time), adds the member to the set, and does not call the veto for that message. Later messages call it normally.
* Games tab: a switch row for the guild veto (``tests/test_dashboard_browser.py``, ``tests/dashboard_server.py``).
"""
import asyncio
import os
import re
import unittest
from unittest import mock

import discord
from helpers import FakeInteraction, command, invoke

import test_games_spoken_discord as tgsd
import test_spoken_veto_wiring as wiring
from llmcord_core import games_spoken_discord as gsd
from llmcord_core.admin_store import ConflictError
from llmcord_core.i18n import discord_locale as dl
from test_game_tables import TableCase

G, ALICE, BOB = tgsd.G, tgsd.ALICE, tgsd.BOB
PROVIDER = "OpenRouter (Microsoft Decision-1)"
DISCLOSURE = (f"When on, a message that the local reader takes for a spoken move is sent to {PROVIDER}, a third-party service, to double-check it. "
              "Only the message text and the legal moves are sent, only for members who turned on spoken moves, and only on their own turn. Off by default.")
OPT_OUT = "/game spoken_veto off"
HANGUL = re.compile("[가-힣]")
FAKE_KEY = "FAKE-veto-key-not-a-real-key"
CONFIGURED_ENV = {"LLMCORD_SPOKEN_VETO": "openrouter-d1", "OPENROUTER_API_KEY": FAKE_KEY}


def has_notice(text: str) -> bool:
    """The member notice: may be double-checked by a third-party service, and how to opt out."""
    return "third-party service" in text and "double-check" in text and OPT_OUT in text


def korean(interaction):
    interaction.locale = discord.Locale.korean
    return interaction


class StoreSettingTests(TableCase):
    """The Games tab switch saves through set_spoken_veto_enabled (same shape as the neighbouring spoken switches' tests)."""

    def test_default_off_save_round_trip_stays_in_guild(self):
        """FEAT-49b3: the Games tab veto switch is off by default, saves per server and leaves guild 2 alone."""
        self.assertFalse(self.store.spoken_veto_enabled(G))
        self.assertIs(self.store.set_spoken_veto_enabled(G, True, False), True)
        self.assertTrue(self.store.spoken_veto_enabled(G))
        self.assertFalse(self.store.spoken_veto_enabled(2))
        self.assertIs(self.store.set_spoken_veto_enabled(G, False, True), False)

    def test_stale_save_conflicts_with_a_reload_message(self):
        """FEAT-49b3: a switch saved against a stale value is refused with the reload message and keeps the stored value."""
        self.store.set_spoken_veto_enabled(G, True)
        with self.assertRaisesRegex(ConflictError, "^The decision-model check setting was changed elsewhere. Reload the page and try again.$"):
            self.store.set_spoken_veto_enabled(G, False, False)
        self.assertTrue(self.store.spoken_veto_enabled(G))


SERVER_OFF = "(This server has the decision-model check turned off right now.)"


class MemberCommandTests(tgsd.SpokenCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.store.set_spoken_veto_enabled(G, True)  # FEAT-49 review: with the server switch off the reply differs (see ServerOffReplyTests)

    async def switch(self, user, setting, guild_id=G, interaction=None):
        click = interaction or FakeInteraction(guild_id=guild_id, user_id=user)
        await invoke(self.bot, "game spoken_veto", click, setting=setting)
        return click

    def test_registered_under_game_with_on_off_status_choices(self):
        """FEAT-49b3: /game spoken_veto takes setting on/off/status, and the /game group stays within Discord's 25 subcommands."""
        cmd = command(self.bot, "game spoken_veto")
        self.assertEqual([(c.name, c.value) for c in cmd._params["setting"].choices], [("On", "on"), ("Off", "off"), ("Status", "status")])
        self.assertLessEqual(len(self.bot.tree.get_command("game").commands), 25)

    async def test_off_and_on_store_for_that_member_only_and_reply_privately_with_the_notice(self):
        """FEAT-49b3: off then on is stored for this guild and member only; every reply is ephemeral and carries the notice line."""
        self.store.set_spoken_veto_member_on(2, ALICE, True)
        off = await self.switch(ALICE, "off")
        self.assertIs(self.store.spoken_veto_member_on(G, ALICE), False)
        self.assertTrue(self.store.spoken_veto_member_on(G, BOB) and self.store.spoken_veto_member_on(2, ALICE))
        on = await self.switch(ALICE, "on")
        self.assertIs(self.store.spoken_veto_member_on(G, ALICE), True)
        for click in (off, on):
            self.assertEqual(len(click.replies), 1)
            self.assertTrue(has_notice(click.replies[0]), click.replies[0])
            self.assertIs(click.response.sent[0][1]["ephemeral"], True)

    async def test_status_changes_nothing_and_still_carries_the_notice(self):
        """FEAT-49b3: status reads the member's switch without writing, and the notice line is in every reply while the server switch is on (it is omitted, with a server-off sentence, when the switch is off)."""
        for stored in (True, False):
            self.store.set_spoken_veto_member_on(G, ALICE, stored)
            click = await self.switch(ALICE, "status")
            self.assertIs(self.store.spoken_veto_member_on(G, ALICE), stored)
            self.assertTrue(has_notice(click.replies[0]), click.replies[0])
            self.assertIs(click.response.sent[0][1]["ephemeral"], True)

    async def test_status_does_not_create_a_row_for_a_new_member(self):
        """FEAT-49b3: looking at the status of a member with no settings row stores nothing (the default stays 'on')."""
        before = self.store.one("SELECT COUNT(*) FROM member_settings WHERE guild_id=? AND user_id=?", (G, BOB))[0]
        await self.switch(BOB, "status")
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM member_settings WHERE guild_id=? AND user_id=?", (G, BOB))[0], before)

    async def test_the_reply_uses_the_invokers_language_and_keeps_the_command_literal(self):
        """FEAT-49b3: with the invoker locale ko the reply is Korean (a catalog entry exists) and still names the literal command."""
        english = (await self.switch(ALICE, "off")).replies[0]
        reply = (await self.switch(ALICE, "off", interaction=korean(FakeInteraction(user_id=ALICE)))).replies[0]
        self.assertNotEqual(reply, english)
        self.assertRegex(reply, HANGUL)
        self.assertIn(OPT_OUT, reply)


class ServerOffReplyTests(tgsd.SpokenCase):
    async def test_replies_say_the_server_has_it_off_and_omit_the_notice(self):
        """FEAT-49 review: with the server switch off, on/off/status replies end with the server-off sentence and carry no notice line."""
        self.assertFalse(self.store.spoken_veto_enabled(G))
        for setting in ("on", "off", "status"):
            click = FakeInteraction(user_id=ALICE)
            await invoke(self.bot, "game spoken_veto", click, setting=setting)
            reply = click.replies[0]
            with self.subTest(setting=setting):
                self.assertIn(SERVER_OFF, reply)
                self.assertFalse(has_notice(reply), reply)
                self.assertIs(click.response.sent[0][1]["ephemeral"], True)

    async def test_server_on_has_no_server_off_sentence(self):
        """FEAT-49 review: with the server switch on the reply is the notice reply, without the server-off sentence."""
        self.store.set_spoken_veto_enabled(G, True)
        click = FakeInteraction(user_id=ALICE)
        await invoke(self.bot, "game spoken_veto", click, setting="status")
        self.assertNotIn("turned off right now", click.replies[0])


class AdminCommandTests(tgsd.SpokenCase):
    def configured(self, on: bool):
        """Agree with whichever way the command decides 'configured': the live client attribute and the process environment."""
        self.spoken.veto = wiring.FakeVeto() if on else None
        env = {k: v for k, v in os.environ.items() if k not in CONFIGURED_ENV}
        patcher = mock.patch.dict(os.environ, {**env, **CONFIGURED_ENV} if on else env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def run_admin(self, setting, admin=True, interaction=None):
        click = interaction or FakeInteraction(admin=admin)
        await invoke(self.bot, "admin games spoken_veto", click, setting=setting)
        return click

    def test_registered_under_admin_games_with_on_off_choices(self):
        """FEAT-49b3: /admin games spoken_veto takes only setting on/off, like /game spoken_moves."""
        cmd = command(self.bot, "admin games spoken_veto")
        self.assertEqual([(c.name, c.value) for c in cmd._params["setting"].choices], [("On", "on"), ("Off", "off")])
        self.assertEqual([o["name"] for o in cmd.to_dict(self.bot.tree).get("options", [])], ["setting"])

    async def test_non_admin_is_refused_and_nothing_is_stored(self):
        """FEAT-49b3: write_admin rules as for the neighbouring admin games commands: the refusal text, no change."""
        self.configured(True)
        member = await self.run_admin("on", admin=False)
        self.assertEqual(member.replies, ["Only server administrators can use that command."])
        self.assertFalse(self.store.spoken_veto_enabled(G))

    async def test_on_and_off_save_and_reply_privately_with_the_disclosure(self):
        """FEAT-49b3: the switch saves per guild, and both replies are ephemeral and contain the design doc's disclosure sentence."""
        self.configured(True)
        on = await self.run_admin("on")
        self.assertIs(self.store.spoken_veto_enabled(G), True)
        self.assertFalse(self.store.spoken_veto_enabled(2))
        off = await self.run_admin("off")
        self.assertIs(self.store.spoken_veto_enabled(G), False)
        for admin in (on, off):
            self.assertEqual(len(admin.replies), 1)
            self.assertIn(DISCLOSURE, admin.replies[0])
            self.assertIs(admin.response.sent[0][1]["ephemeral"], True)
            self.assertNotIn("no effect", admin.replies[0])

    async def test_unconfigured_operator_saves_but_says_it_has_no_effect(self):
        """FEAT-49b3: without LLMCORD_SPOKEN_VETO=openrouter-d1 and a key the switch still saves, and the reply says it does nothing until the operator configures it."""
        self.configured(False)
        admin = await self.run_admin("on")
        self.assertIs(self.store.spoken_veto_enabled(G), True)
        self.assertIn(DISCLOSURE, admin.replies[0])
        self.assertIn("no effect", admin.replies[0])
        self.assertIn("operator", admin.replies[0])

    async def test_the_reply_uses_the_invokers_language_but_keeps_the_provider_name(self):
        """FEAT-49b3: with the invoker locale ko the disclosure is Korean (a catalog entry exists) and the provider name stays as written."""
        self.configured(True)
        admin = await self.run_admin("on", interaction=korean(FakeInteraction(admin=True)))
        reply = admin.replies[0]
        self.assertRegex(reply, HANGUL)
        self.assertIn(PROVIDER, reply)
        self.assertNotIn(DISCLOSURE, reply)


class NoticeCase(wiring.VetoCase):
    """VetoCase with the member notice NOT pre-seeded (the shared wiring tests pre-seed it)."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.spoken.veto_notified.clear()

    @staticmethod
    def notices(msg):
        return [p for p in msg.posts if has_notice(p.content)]

    @staticmethod
    def checks(msg):
        return [p for p in msg.posts if p.view is not None]

    def assert_check_labels(self, msg, labels):
        self.assertEqual(len(self.checks(msg)), 1)
        self.assertEqual([c.item.label for c in self.checks(msg)[0].view.children], labels)

    async def next_member_on_turn(self):
        await self.press(ALICE, "Stand")
        self.assertEqual(self.bot.games.turns[self.bot.games.turn_channels[(G, tgsd.CH)]].member_id, BOB)


class NoticeTests(NoticeCase):
    async def test_first_eligible_message_posts_the_notice_and_does_not_call(self):
        """FEAT-49b3: the first message that would leave the Pi gets the notice and e5's buttons; nothing is sent to the service."""
        await self.deal()
        msg = await self.say(ALICE, f"  {wiring.SENTINEL}  ")
        self.assertEqual(len(self.notices(msg)), 1)
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(self.runtime.texts, [wiring.SENTINEL])
        self.assert_check_labels(msg, ["Hit", "Stand", "Just table talk"])
        self.assertNotIn(wiring.SENTINEL, self.notices(msg)[0].content)
        self.assertEqual(self.spoken.veto_notified, {(G, ALICE)})

    async def test_second_message_uses_the_veto_without_a_second_notice(self):
        """FEAT-49b3: after the notice the member's later messages in this process are vetoed normally and get no notice."""
        await self.deal()
        await self.say(ALICE, "one")
        self.fake.probs = {"hit": 0.9, "stand": 0.1}
        second = await self.say(ALICE, "two")
        self.assertEqual([c[0] for c in self.fake.calls], ["two"])
        self.assertEqual(self.notices(second), [])
        self.assert_check_labels(second, ["Hit", "Just table talk"])

    async def test_a_different_member_gets_their_own_notice(self):
        """FEAT-49b3: the mark is per member: the next player's first eligible message is notified and not sent."""
        await self.deal()
        await self.say(ALICE, "one")
        await self.next_member_on_turn()
        first = await self.say(BOB, "bob one")
        self.assertEqual((len(self.notices(first)), self.fake.calls), (1, []))
        await self.say(BOB, "bob two")
        self.assertEqual([c[0] for c in self.fake.calls], ["bob two"])
        self.assertEqual(self.spoken.veto_notified, {(G, ALICE), (G, BOB)})

    async def test_a_member_notified_in_another_guild_is_still_notified_here(self):
        """FEAT-49b3: the mark is per guild and member, so the same user in another server does not skip this server's notice."""
        await self.deal()
        self.spoken.veto_notified.add((2, ALICE))
        msg = await self.say(ALICE)
        self.assertEqual((len(self.notices(msg)), self.fake.calls), (1, []))
        self.assertEqual(self.spoken.veto_notified, {(2, ALICE), (G, ALICE)})

    async def test_opted_out_member_gets_no_notice_and_no_call(self):
        """FEAT-49b3: a member who opted out of the veto sees no notice (nothing leaves) and is not marked."""
        await self.deal()
        self.store.set_spoken_veto_member_on(G, ALICE, False)
        msg = await self.say(ALICE)
        self.assertEqual((self.notices(msg), self.fake.calls, self.spoken.veto_notified), ([], [], set()))
        self.assert_check_labels(msg, ["Hit", "Stand", "Just table talk"])

    async def test_guild_switch_off_gets_no_notice_and_no_call(self):
        """FEAT-49b3: with the server switch off nothing is announced and nothing is sent."""
        await self.deal()
        self.store.set_spoken_veto_enabled(G, False)
        msg = await self.say(ALICE)
        self.assertEqual((self.notices(msg), self.fake.calls, self.spoken.veto_notified), ([], [], set()))
        self.assert_check_labels(msg, ["Hit", "Stand", "Just table talk"])

    async def test_no_client_gets_no_notice(self):
        """FEAT-49b3: with the operator switch off (no client) nobody is told about a service that is never called."""
        await self.deal()
        self.spoken.veto = None
        msg = await self.say(ALICE)
        self.assertEqual((self.notices(msg), self.spoken.veto_notified), ([], set()))

    async def test_e5_silent_first_message_neither_notifies_nor_marks(self):
        """FEAT-49b3: a message e5 calls silent would not be sent, so no notice; the first message that would be sent gets it."""
        await self.deal()
        self.decision = tgsd.Decision("silent", ())
        silent = await self.say(ALICE, "lol")
        self.assertEqual((silent.posts, self.fake.calls, self.spoken.veto_notified), ([], [], set()))
        self.decision = tgsd.Decision("unsure", ("hit", "stand"))
        msg = await self.say(ALICE, "hit me")
        self.assertEqual((len(self.notices(msg)), self.fake.calls), (1, []))

    async def _hold_notice(self, msg, fail=False):
        """Make msg's first reply (the notice) wait for ``gate``; with fail=True it then raises. Returns (entered, gate)."""
        entered, gate, calls = asyncio.Event(), asyncio.Event(), []

        async def on_reply():
            calls.append(1)
            if len(calls) == 1:
                entered.set()
                await gate.wait()
                if fail:
                    raise RuntimeError("discord is down")

        msg.on_reply = on_reply
        return entered, gate

    async def test_second_message_while_the_notice_is_in_flight_posts_no_second_notice_and_no_call(self):
        """FEAT-49 review: while the notice post is in flight nothing is sent and no second notice is posted; the mark is set only after it succeeded."""
        await self.deal()
        first = tgsd.Said(ALICE, "one")
        entered, gate = await self._hold_notice(first)
        self.spoken.note_message(first)
        await asyncio.wait_for(entered.wait(), 1)
        second = tgsd.Said(ALICE, "two")
        before = set(self.spoken.tasks)
        self.spoken.note_message(second)
        await asyncio.wait_for(asyncio.gather(*(self.spoken.tasks - before)), 1)
        self.assertEqual((self.fake.calls, self.notices(second)), ([], []))
        self.assert_check_labels(second, ["Hit", "Stand", "Just table talk"])
        self.assertEqual(self.spoken.veto_notified, set())
        gate.set()
        await self.settle()
        self.assertEqual(len(self.notices(first)), 1)
        self.assertEqual((self.fake.calls, self.spoken.veto_notified), ([], {(G, ALICE)}))
        await self.say(ALICE, "three")
        self.assertEqual([c[0] for c in self.fake.calls], ["three"])

    async def test_failed_notice_does_not_mark_and_is_retried(self):
        """FEAT-49 review: if the notice post raises, the member is not marked, nothing was sent for either message, and the next eligible message posts the notice again."""
        await self.deal()
        first = tgsd.Said(ALICE, "one")
        entered, gate = await self._hold_notice(first, fail=True)
        self.spoken.note_message(first)
        await asyncio.wait_for(entered.wait(), 1)
        second = tgsd.Said(ALICE, "two")
        before = set(self.spoken.tasks)
        self.spoken.note_message(second)
        await asyncio.wait_for(asyncio.gather(*(self.spoken.tasks - before)), 1)
        gate.set()
        await self.settle()
        self.assertEqual((self.fake.calls, self.spoken.veto_notified, self.notices(first)), ([], set(), []))
        third = await self.say(ALICE, "three")
        self.assertEqual((len(self.notices(third)), self.fake.calls), (1, []))
        self.assertEqual(self.spoken.veto_notified, {(G, ALICE)})
        await self.say(ALICE, "four")
        self.assertEqual([c[0] for c in self.fake.calls], ["four"])

    async def test_marks_reset_with_a_new_instance(self):
        """FEAT-49b3: the marks live in memory only: a new SpokenMoves (a restart) notifies again."""
        await self.deal()
        await self.say(ALICE, "one")
        self.assertEqual(self.spoken.veto_notified, {(G, ALICE)})
        fresh = gsd.SpokenMoves(self.bot, env={})
        self.addCleanup(fresh.close)
        self.assertEqual(fresh.veto_notified, set())
        fresh.runtime, fresh.veto = tgsd.Runtime(), wiring.FakeVeto()
        fresh.veto.probs = {"hit": 0.9, "stand": 0.9}
        self.bot.spoken = fresh
        msg = tgsd.Said(ALICE, "again")
        fresh.note_message(msg)
        await asyncio.gather(*list(fresh.tasks))
        self.assertEqual((len(self.notices(msg)), fresh.veto.calls), (1, []))


class NoticeLanguageTests(NoticeCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        dl.set_server_language_source(self.store.server_language)  # SkitBot.setup_hook registers it in production
        self.addCleanup(dl.set_server_language_source, None)

    async def test_the_notice_follows_the_channel_language_at_render_time(self):
        """FEAT-49b3: the notice is the channel's (channel_t(guild)), resolved when posted: English first, Korean after the server language changes."""
        await self.deal()
        first = await self.say(ALICE, "one")
        self.assertEqual(len(self.notices(first)), 1)
        self.assertNotRegex(self.notices(first)[0].content, HANGUL)
        self.store.set_server_language(G, "ko")
        await self.next_member_on_turn()
        second = await self.say(BOB, "bob one")
        posts = [p for p in second.posts if p.view is None]
        self.assertEqual(len(posts), 1)
        self.assertRegex(posts[0].content, HANGUL)
        self.assertIn(OPT_OUT, posts[0].content)


if __name__ == "__main__":
    unittest.main()
