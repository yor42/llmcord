import unittest
import asyncio
import discord
from pathlib import Path

from llmcord_core.config import ModelProfile, Settings
from llmcord_core.discord_bot import SkitBot, split_discord
from llmcord_core.engine import SceneContext
from helpers import CompiledAdapter, drain_memory_tasks, is_turn_failure


class FakeModels(CompiledAdapter):
    def __init__(self, speakers):
        self.speakers = speakers
        self.lines = iter(["First line", "Second line"])

    async def structured(self, role, system, messages, schema_name, schema):
        if schema_name == "choose_speakers":
            return {"speakers": self.speakers}
        return {"shared_facts": [], "personal_facts": [], "encounter_facts": []}

    async def stream_text(self, role, system, messages):
        yield next(self.lines)

    async def text(self, role, system, messages, max_tokens=None):
        return "The group met."


class FakeMessage:
    def __init__(self, ident, content):
        self.id, self.content = ident, content
        self.deleted = False
        self.edits = []

    async def edit(self, *, content, **kwargs):
        self.content = content
        self.edits.append(content)

    async def delete(self):
        self.deleted = True


class FakeWebhook:
    def __init__(self, start):
        self.next_id = start
        self.posts = []
        self.options = []

    async def send(self, content, **kwargs):
        # discord.py dereferences thread.id when the keyword is present.
        if kwargs.get('thread', discord.utils.MISSING) is None:
            raise AttributeError("'NoneType' object has no attribute 'id'")
        result = FakeMessage(self.next_id, content)
        self.next_id += 1
        self.posts.append(result)
        self.options.append(kwargs)
        return result


class FakeChannel:
    def __init__(self, ident):
        self.id = ident
        self.messages = []
        self.options = []

    @property
    def errors(self):
        return [message.content for message in self.messages
                if not message.deleted and is_turn_failure(message.content)]

    async def send(self, content, **kwargs):
        message = FakeMessage(5000 + len(self.messages), content)
        self.messages.append(message)
        self.options.append(kwargs)
        return message


class DiscordFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_turn_keeps_preset_snapshot_and_emotion_across_chunks(self):
        profile = ModelProfile('compatible', 'test', 16000, False, base_url='http://localhost/v1')
        settings = Settings('token', None, ':memory:', 90, {'test': profile}, 'test', 'test', 'test',
            {'max_input_tokens': 12000, 'max_output_tokens': 3000, 'max_images': 3, 'max_attachment_bytes': 8388608,
             'max_speakers': 3, 'recent_messages': 12, 'recent_window_seconds': 600, 'ambient_cooldown_seconds': 120})
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
        class Models(FakeModels):
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
        profile = ModelProfile("compatible", "test", 16000, False,
            base_url="http://localhost/v1")
        settings = Settings("token", None, Path(":memory:"), 90,
            {"test": profile}, "test", "test", "test",
            {"max_input_tokens": 12000, "max_output_tokens": 700,
             "max_images": 3, "max_attachment_bytes": 8388608,
             "max_speakers": 3, "recent_messages": 12,
             "recent_window_seconds": 600, "ambient_cooldown_seconds": 120})
        bot = SkitBot(settings)
        world = bot.store.create_space(1, "World", "world")
        bot.store.bind_channel(1, 100, world)
        alice = bot.store.add_character(1, world, "Alice", {"name": "Alice"}, None, [])
        bob = bot.store.add_character(1, world, "Bob", {"name": "Bob"}, None, [])
        bot.store.set_cast(100, None, [alice, bob])
        fake_models = FakeModels([alice, bob])
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
        profile = ModelProfile('compatible', 'test', 16000, False, base_url='http://localhost/v1')
        settings = Settings('token', None, ':memory:', 90, {'test': profile}, 'test', 'test', 'test',
            {'max_input_tokens': 12000, 'max_output_tokens': 700, 'max_speakers': 3})
        bot = SkitBot(settings)
        world = bot.store.create_space(1, 'World', 'world')
        bot.store.bind_channel(1, 100, world)
        alice = bot.store.add_character(1, world, 'Alice', {'name': 'Alice'}, None, [])
        bot.store.set_cast(100, None, [alice])
        bot.engine.models = bot.models = FakeModels([alice])
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
        profile = ModelProfile('compatible', 'test', 16000, False, base_url='http://localhost/v1')
        settings = Settings('token', None, ':memory:', 90, {'test': profile}, 'test', 'test', 'test',
                            {'max_input_tokens': 12000, 'max_output_tokens': 700, 'max_speakers': 3})
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
        profile = ModelProfile('compatible', 'test', 16000, False, base_url='http://localhost/v1')
        settings = Settings('token', None, ':memory:', 90, {'test': profile}, 'test', 'test', 'test',
                            {'max_input_tokens': 12000, 'max_output_tokens': 700, 'max_speakers': 3})
        bot = SkitBot(settings)
        world = bot.store.create_space(1, 'World', 'world')
        channel = FakeChannel(100)
        try:
            await bot.run_scene(SceneContext(1, 100, None, world, 9, 1000, 'Hello', None, [], [], ambient=True), channel)
            self.assertEqual(channel.messages, [])
        finally:
            bot.store.close()


if __name__ == "__main__":
    unittest.main()
