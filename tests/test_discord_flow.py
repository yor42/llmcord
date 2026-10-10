import unittest
import asyncio

from llmcord_core.discord_bot import SkitBot, split_discord
from llmcord_core.engine import SceneContext
from helpers import FakeChannel, make_settings, FakeMessage, FakeWebhook, FlowFakeModels, drain_memory_tasks


class DiscordFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_turn_keeps_preset_snapshot_and_emotion_across_chunks(self):
        settings = make_settings(max_output_tokens=3000)
        bot = SkitBot(settings)
        world = bot.store.create_space(1, 'W', 'world')
        bot.store.bind_channel(1, 100, world)
        alice = bot.store.add_character(1, world, 'Alice', {'name': 'Alice'}, None, [])
        bob = bot.store.add_character(1, world, 'Bob', {'name': 'Bob'}, None, [])
        bot.store.set_cast(100, None, [alice, bob])
        from llmcord_core.prompts import default_bundle
        first = default_bundle()
        first['purposes']['dialogue'][0]['content'] = 'ORIGINAL PRESET'
        ident, revision = bot.store.save_preset(1, 'First', first)
        bot.store.activate_preset(1, ident, revision, {'dialogue': 'compatible'})
        changed = default_bundle()
        changed['purposes']['dialogue'][0]['content'] = 'NEW PRESET'
        other, newer = bot.store.save_preset(1, 'Other', changed)
        calls = []
        class Models(FlowFakeModels):
            async def stream_text(self, role, system, messages):
                calls.append(system)
                if len(calls) == 1:
                    bot.store.activate_preset(1, other, newer, {'dialogue': 'compatible'})
                yield '<emo'
                yield 'tion>neutral</emotion>\n' + 'word ' * 600
        models = Models([alice, bob])
        bot.engine.models = bot.models = models
        hooks = {alice: FakeWebhook(2000), bob: FakeWebhook(3000)}
        async def webhook(channel, character):
            return hooks[character['id']]
        bot._webhook = webhook
        channel = FakeChannel(100)
        try:
            await bot.run_scene(SceneContext(1, 100, None, world, 9, 1000, 'Start', None, [], []), channel)
            self.assertEqual(channel.errors, [])
            self.assertTrue(all('ORIGINAL PRESET' in call and 'NEW PRESET' not in call for call in calls))
            self.assertTrue(all('<emotion>' not in msg.content for hook in hooks.values() for msg in hook.posts))
            self.assertTrue(all(bot.store.trace(msg.id)['preset']['id'] == ident for hook in hooks.values() for msg in hook.posts))
            self.assertTrue(all(opt['username'] == 'Alice' for opt in hooks[alice].options))
            self.assertGreater(len(hooks[alice].posts), 1)
            self.assertTrue(all('thread' not in options for hook in hooks.values() for options in hook.options))
        finally:
            await drain_memory_tasks(bot)  # REL-02: let background memory work finish before the store closes
            bot.store.close()

    async def test_sequential_webhook_lines_and_parent_recovery(self):
        settings = make_settings()
        bot = SkitBot(settings)
        world = bot.store.create_space(1, "World", "world")
        bot.store.bind_channel(1, 100, world)
        alice = bot.store.add_character(1, world, "Alice", {"name": "Alice"}, None, [])
        bob = bot.store.add_character(1, world, "Bob", {"name": "Bob"}, None, [])
        bot.store.set_cast(100, None, [alice, bob])
        fake_models = FlowFakeModels([alice, bob])
        bot.engine.models = bot.models = fake_models
        hooks = {alice: FakeWebhook(2000), bob: FakeWebhook(3000)}

        async def webhook(channel, character):
            return hooks[character["id"]]

        bot._webhook = webhook
        channel = FakeChannel(100)
        scene = SceneContext(1, 100, None, world, 9, 1000, "Start a skit", None, [], [])
        try:
            await bot.run_scene(scene, channel)
            await drain_memory_tasks(bot)  # REL-02: the summary may run as a background task
            self.assertEqual(channel.errors, [])
            self.assertEqual(hooks[alice].posts[0].content.split('\n\n-# ')[0], "First line")
            self.assertEqual(hooks[bob].posts[0].content.split('\n\n-# ')[0], "Second line")
            self.assertEqual(bot.store.node(2000)["parent_id"], 1000)
            self.assertEqual(bot.store.node(3000)["parent_id"], 2000)
            self.assertEqual(bot.store.ancestors(3000)[-1]["character_id"], bob)
            self.assertEqual(bot.store.summary(3000), "The group met.")
            self.assertEqual(bot.store.trace(3000)["character_id"], bob)
            self.assertEqual(len(channel.messages), 1)
            self.assertTrue(channel.messages[0].deleted)
            self.assertIsNone(bot.store.node(channel.messages[0].id))
            self.assertTrue(channel.options[0]['silent'])
        finally:
            bot.store.close()

    def test_discord_split_respects_limit(self):
        chunks = split_discord("word " * 800, 100)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertEqual(" ".join(chunks).split(), ("word " * 800).split())

    async def test_failure_reports_stage_and_preserves_original_cleanup_error(self):
        settings = make_settings()
        bot = SkitBot(settings)
        world = bot.store.create_space(1, 'World', 'world')
        bot.store.bind_channel(1, 100, world)
        alice = bot.store.add_character(1, world, 'Alice', {'name': 'Alice'}, None, [])
        bot.store.set_cast(100, None, [alice])
        bot.engine.models = bot.models = FlowFakeModels([alice])
        class BrokenMessage(FakeMessage):
            async def edit(self, **kwargs):
                raise RuntimeError('Webhook edit rejected')
            async def delete(self):
                raise ValueError('Cleanup also failed')
        class BrokenWebhook:
            async def send(self, *args, **kwargs):
                return BrokenMessage(2000, '…')
        async def webhook(*args):
            return BrokenWebhook()
        bot._webhook = webhook
        channel = FakeChannel(100)
        try:
            # The stage and original error are pinned in the logged ERROR line, which carries them both before
            # and after D3 (public turn failures become generic; see tests/test_error_mapping.py, UX-07/SEC-05).
            with self.assertLogs(level='ERROR') as logs:
                await bot.run_scene(SceneContext(1, 100, None, world, 9, 1000, 'Hello', None, [], []), channel)
            failures = [record.getMessage() for record in logs.records if record.levelname == 'ERROR']
            self.assertEqual(len(failures), 1, failures)
            self.assertIn('webhook delivery', failures[0])
            self.assertIn('RuntimeError: Webhook edit rejected', failures[0])
            self.assertNotIn('Cleanup also failed', failures[0])
            self.assertEqual(len(channel.errors), 1)
            self.assertNotIn('Cleanup also failed', channel.errors[0])
            self.assertFalse(channel.messages[0].deleted)
            self.assertEqual(len(channel.messages), 1)
        finally:
            bot.store.close()

    async def test_status_appears_before_slow_director_and_clears_for_no_cast(self):
        settings = make_settings()
        bot = SkitBot(settings)
        world = bot.store.create_space(1, 'World', 'world')
        channel = FakeChannel(100)
        entered = asyncio.Event()
        release = asyncio.Event()
        async def slow_speakers(scene):
            entered.set()
            await release.wait()
            return []
        bot.engine.speakers = slow_speakers
        task = asyncio.create_task(bot.run_scene(SceneContext(1, 100, None, world, 9, 1000, 'Hello', None, [], []), channel))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            self.assertTrue(channel.messages[0].content.startswith('⏳ Generating a reply…'))
            self.assertIn('test · 24h tracked: 0 tokens', channel.messages[0].content)
            self.assertFalse(task.done())
            release.set()
            await task
            self.assertEqual(len(channel.messages), 1)
            self.assertIn('No active character', channel.messages[0].content)
        finally:
            release.set()
            await task
            bot.store.close()

    async def test_silent_ambient_turn_does_not_post_status(self):
        settings = make_settings()
        bot = SkitBot(settings)
        world = bot.store.create_space(1, 'World', 'world')
        channel = FakeChannel(100)
        try:
            await bot.run_scene(SceneContext(1, 100, None, world, 9, 1000, 'Hello', None, [], [], ambient=True), channel)
            self.assertEqual(channel.messages, [])
        finally:
            bot.store.close()

    async def test_ambient_turn_with_speaker_posts_status_after_director_and_marks_response(self):
        """Characterization (ARCH-01): an ambient turn that has a speaker posts the "Generating a reply…" status
        only after the director returned, clears it at the end, and records the ambient response."""
        settings = make_settings()
        bot = SkitBot(settings)
        world = bot.store.create_space(1, 'World', 'world')
        bot.store.bind_channel(1, 100, world)
        alice = bot.store.add_character(1, world, 'Alice', {'name': 'Alice'}, None, [])
        bot.store.set_cast(100, None, [alice])
        bot.engine.models = bot.models = FlowFakeModels([alice])
        hook = FakeWebhook(2000)
        async def webhook(channel, character):
            return hook
        bot._webhook = webhook
        channel = FakeChannel(100)
        seen = []
        real_speakers = bot.engine.speakers
        async def speakers(scene):
            seen.append(len(channel.messages))
            result = await real_speakers(scene)
            seen.append(len(channel.messages))
            return result
        bot.engine.speakers = speakers
        try:
            self.assertEqual(bot.store.ambient_state(100)[1], 0.0)
            await bot.run_scene(SceneContext(1, 100, None, world, 9, 1000, 'Alice, join us', None, [], [], ambient=True), channel)
            await drain_memory_tasks(bot)
            self.assertEqual(seen, [0, 0])
            self.assertEqual(channel.errors, [])
            self.assertEqual(len(channel.messages), 1)
            self.assertTrue(channel.messages[0].content.startswith('⏳'))
            self.assertTrue(all(message.deleted for message in channel.messages))
            self.assertEqual(hook.posts[0].content.split('\n\n-# ')[0], 'First line')
            count, last = bot.store.ambient_state(100)
            self.assertEqual(count, 0)
            self.assertGreater(last, 0)
        finally:
            bot.store.close()


if __name__ == "__main__":
    unittest.main()
