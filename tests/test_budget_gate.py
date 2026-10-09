"""FEAT-16 step 6: spending-cap gate, operator DMs and member notices in SkitBot."""
from __future__ import annotations

import asyncio
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import discord

from helpers import FakeInteraction, FakeModels, FakeTextChannel, invoke, make_settings
from llmcord_core.discord_bot import SkitBot
from llmcord_core.engine import SceneContext

CHANNEL = 100
OPS = frozenset({5, 6})


class FakeUser:
    def __init__(self, error=None):
        self.sent, self.error = [], error

    async def send(self, text, **kwargs):
        if self.error:
            raise self.error
        self.sent.append((text, kwargs))


def forbidden():
    return discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), "Cannot send")


class GateCase(unittest.IsolatedAsyncioTestCase):
    operators = OPS

    def setUp(self):
        self.bot = SkitBot(replace(make_settings(), operator_ids=self.operators))
        self.store = self.bot.store
        self.world = self.store.create_space(1, "World", "world")
        self.store.bind_channel(1, CHANNEL, self.world)
        self.alice = self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])
        self.store.set_cast(CHANNEL, None, [self.alice])
        self.channel = FakeTextChannel(CHANNEL)
        self.models = self.bot.engine.models = self.bot.models = FakeModels([self.alice])
        self.users = {i: FakeUser() for i in self.operators}
        self.bot.get_user = self.users.get

    async def asyncTearDown(self):
        await self.drain()
        for task in self.bot.memory_tasks.values():
            task.cancel()
        await asyncio.sleep(0)

    def tearDown(self):
        self.store.close()

    async def drain(self):
        while pending := [t for t in (*self.bot.note_tasks, *self.bot.memory_tasks.values()) if not t.done()]:
            await asyncio.gather(*pending, return_exceptions=True)

    def caps(self, soft=None, hard=None, notice=False):
        rev = self.store.budget_settings()['revision']
        self.store.save_budget(soft, hard, 1, notice, rev)

    def spend(self, cost, unpriced=0):
        day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        with self.store.db:
            self.store.db.execute('INSERT INTO spend_days(day,cost_usd,unpriced_calls) VALUES(?,?,?) '
                                  'ON CONFLICT(day) DO UPDATE SET cost_usd=cost_usd+excluded.cost_usd,'
                                  'unpriced_calls=unpriced_calls+excluded.unpriced_calls', (day, cost, unpriced))

    async def turn(self, ident=1000, ambient=False, interaction=None):
        scene = SceneContext(1, CHANNEL, None, self.world, 9, ident, "Hi", None, [], [], ambient=ambient)
        await self.bot.run_scene(scene, self.channel, interaction)
        await self.drain()

    @property
    def replies(self):
        return [post for hook in self.channel.hooks for post in hook.posts]

    def dms(self, ident):
        return [text for text, _ in self.users[ident].sent]


class HardCapTests(GateCase):
    async def test_hard_cap_blocks_explicit_turn_silently_when_notice_off(self):
        self.caps(hard=1.0)
        self.spend(2.0)
        await self.turn()
        self.assertEqual(self.models.calls, [])
        self.assertEqual(self.replies, [])
        self.assertEqual(self.channel.sent, [])

    async def test_channel_notice_posts_once_per_hour(self):
        self.caps(hard=0, notice=True)
        with patch('llmcord_core.discord_bot.time.time', return_value=1_800_000_000.0):
            await self.turn(1)
            await self.turn(2)
        self.assertEqual(len(self.channel.sent), 1)
        self.assertIn("Character replies are paused", self.channel.sent[0].content)
        self.assertEqual(self.models.calls, [])
        with patch('llmcord_core.discord_bot.time.time', return_value=1_800_000_000.0 + 3599):
            await self.turn(3)
        self.assertEqual(len(self.channel.sent), 1)
        with patch('llmcord_core.discord_bot.time.time', return_value=1_800_000_000.0 + 3600):
            await self.turn(4)
        self.assertEqual(len(self.channel.sent), 2)

    async def test_ambient_scene_in_run_scene_is_silent_even_with_notice_on(self):
        self.caps(hard=0, notice=True)
        await self.turn(ambient=True)
        self.assertEqual((self.models.calls, self.channel.sent), ([], []))

    async def test_ambient_message_does_not_count_or_call_model(self):
        self.store.set_ambient(CHANNEL, True)
        self.caps(hard=0)
        user = SimpleNamespace(id=9, display_name='Sam', bot=False)

        class Chan:
            id = CHANNEL

        message = SimpleNamespace(id=1000, guild=SimpleNamespace(id=1), channel=Chan(), author=user, content='hey',
                                  mentions=[], webhook_id=None, reference=None, attachments=[],
                                  created_at=datetime.now(timezone.utc))
        before = self.store.ambient_state(CHANNEL)
        await self.bot.on_message(message)
        self.assertEqual(self.store.ambient_state(CHANNEL), before)
        self.assertEqual(self.models.calls, [])

    async def test_summon_under_hard_cap_replies_ephemerally(self):
        self.caps(hard=0)
        interaction = FakeInteraction(channel=self.channel)
        reads = []

        async def history(**kwargs):
            reads.append(kwargs)
            return
            yield
        self.channel.history = history
        await invoke(self.bot, "summon", interaction, "Alice", "Wave")
        self.assertEqual(reads, [])
        (content, kwargs), = interaction.response.sent
        self.assertIn("Character replies are paused", content)
        self.assertTrue(kwargs["ephemeral"])
        self.assertEqual((self.models.calls, self.channel.sent, self.replies), ([], [], []))

    async def test_run_scene_with_interaction_uses_followup_when_response_done(self):
        self.caps(hard=0)
        interaction = FakeInteraction(channel=self.channel)
        await interaction.response.defer()
        await self.turn(interaction=interaction)
        (content, kwargs), = interaction.followup.sent
        self.assertTrue(kwargs["ephemeral"])
        self.assertEqual(self.models.calls, [])

    async def test_raising_hard_cap_lets_next_turn_run(self):
        self.caps(hard=1.0)
        self.spend(2.0)
        await self.turn(1)
        self.assertEqual(self.replies, [])
        self.caps(hard=10.0)
        await self.turn(2)
        self.assertTrue(self.replies)

    async def test_caps_off_runs_normally_without_dms(self):
        self.spend(500.0)
        await self.turn()
        self.assertTrue(self.replies)
        self.assertEqual([self.dms(i) for i in OPS], [[], []])

    async def test_one_hard_dm_per_operator_and_no_later_soft_dm(self):
        self.caps(soft=1.0, hard=2.0)
        self.spend(3.0)
        await self.turn(1)
        await self.turn(2)
        for ident in OPS:
            texts = self.dms(ident)
            self.assertEqual(len(texts), 1)
            self.assertTrue(texts[0].startswith("Spending limit reached"))
            self.assertIn("$3.00", texts[0])
            self.assertIn("hard cap of $2.00", texts[0])
        self.assertEqual(self.users[5].sent[0][1]["allowed_mentions"].everyone, False)


class SoftCapTests(GateCase):
    async def test_soft_cap_never_blocks_and_dms_once_per_operator(self):
        self.caps(soft=1.0)
        self.spend(1234.5)
        await self.turn(1)
        self.assertTrue(self.replies)
        await self.turn(2)
        for ident in OPS:
            texts = self.dms(ident)
            self.assertEqual(len(texts), 1)
            self.assertIn("soft cap of $1.00", texts[0])
            self.assertIn("$1,234.50", texts[0])
            self.assertIn("Hard cap: off.", texts[0])
            self.assertRegex(texts[0], r"resets on \d{4}-\d{2}-\d{2} \(UTC\)")

    async def test_unpriced_suffix_singular_and_plural(self):
        self.caps(soft=1.0)
        self.spend(5.0, unpriced=1)
        self.bot._budget_check()
        await self.drain()
        self.assertTrue(self.dms(5)[0].endswith("1 model call without a cost estimate was counted as $0."))
        self.spend(0, unpriced=2)
        state = self.bot._budget_check()
        from llmcord_core.discord_bot import _operator_text
        self.assertTrue(_operator_text('soft', state).endswith("3 model calls without a cost estimate were counted as $0."))

    async def test_dm_failure_is_logged_and_turn_still_runs(self):
        self.caps(soft=1.0)
        self.spend(5.0)
        self.users[5].error = forbidden()
        self.users[6].error = discord.HTTPException(SimpleNamespace(status=500, reason="x"), "boom")
        with self.assertLogs(level="WARNING") as logs:
            await self.turn()
        self.assertTrue(self.replies)
        self.assertIn("Operator spending DM", "\n".join(logs.output))

    async def test_dm_failure_with_hard_cap_still_blocks(self):
        self.caps(hard=1.0)
        self.spend(5.0)
        self.users[5].error = forbidden()
        with self.assertLogs(level="WARNING"):
            await self.turn()
        self.assertEqual((self.models.calls, self.replies), ([], []))

    async def test_fetch_user_used_when_not_cached(self):
        self.caps(soft=1.0)
        self.spend(5.0)
        fetched = FakeUser()

        async def fetch_user(ident):
            return fetched
        self.bot.get_user = lambda ident: None
        self.bot.fetch_user = fetch_user
        self.bot._budget_check()
        await self.drain()
        self.assertEqual(len(fetched.sent), 2)

    async def test_budget_read_failure_fails_open(self):
        with patch('llmcord_core.discord_bot.budget.state', side_effect=RuntimeError("db gone")):
            with self.assertLogs(level="ERROR") as logs:
                await self.turn()
        self.assertIn("Budget check failed", "\n".join(logs.output))
        self.assertTrue(self.replies)


class NoOperatorTests(GateCase):
    operators = frozenset()

    async def test_no_operators_means_no_dms_and_hard_cap_still_blocks(self):
        self.caps(soft=1.0, hard=2.0)
        self.spend(5.0)
        await self.turn()
        self.assertEqual(self.bot.note_tasks, set())
        self.assertEqual(self.models.calls, [])


if __name__ == "__main__":
    unittest.main()
