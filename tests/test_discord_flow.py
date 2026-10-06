import unittest
from pathlib import Path

from llmcord_core.config import ModelProfile, Settings
from llmcord_core.discord_bot import SkitBot, split_discord
from llmcord_core.engine import SceneContext


class FakeModels:
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

    async def edit(self, *, content, **kwargs):
        self.content = content

    async def delete(self):
        pass


class FakeWebhook:
    def __init__(self, start):
        self.next_id = start
        self.posts = []

    async def send(self, content, **kwargs):
        result = FakeMessage(self.next_id, content)
        self.next_id += 1
        self.posts.append(result)
        return result


class FakeChannel:
    def __init__(self, ident):
        self.id = ident
        self.errors = []

    async def send(self, content, **kwargs):
        self.errors.append(content)


class DiscordFlowTests(unittest.IsolatedAsyncioTestCase):
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
            self.assertEqual(channel.errors, [])
            self.assertEqual(hooks[alice].posts[0].content, "First line")
            self.assertEqual(hooks[bob].posts[0].content, "Second line")
            self.assertEqual(bot.store.node(2000)["parent_id"], 1000)
            self.assertEqual(bot.store.node(3000)["parent_id"], 2000)
            self.assertEqual(bot.store.ancestors(3000)[-1]["character_id"], bob)
            self.assertEqual(bot.store.summary(3000), "The group met.")
            self.assertEqual(bot.store.trace(3000)["character_id"], bob)
        finally:
            bot.store.close()

    def test_discord_split_respects_limit(self):
        chunks = split_discord("word " * 800, 100)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertEqual(" ".join(chunks).split(), ("word " * 800).split())


if __name__ == "__main__":
    unittest.main()
