import hashlib
import unittest
from io import BytesIO
from types import SimpleNamespace

import discord
from PIL import Image

from llmcord_core.admin_store import ConflictError
from llmcord_core.avatars import normalize_avatar
from llmcord_core.discord_bot import SkitBot
from llmcord_core.engine import SceneContext
from llmcord_core.store import Store
from test_core import settings
from test_discord_flow import FakeChannel, FakeModels, FakeWebhook


def image(color):
    output = BytesIO()
    Image.new('RGB', (40, 50), color).save(output, 'PNG')
    return normalize_avatar(output.getvalue())


class StaticAvatarTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.world = self.store.create_space(1, 'World', 'world')
        self.character = self.store.add_character(1, self.world, 'Alice', {'name': 'Alice'}, image('red'), [])

    def tearDown(self):
        self.store.close()

    def test_card_portrait_is_fallback_and_emotion_images_are_independent(self):
        self.assertEqual(self.store.character_by_id(self.character)['avatar'], image('red'))
        self.assertIsNone(self.store.avatar_slot(1, self.character, 'neutral')['image'])
        self.store.save_avatar(1, self.character, 'neutral', 'Neutral', '', image('green'))
        self.store.save_avatar(1, self.character, 'happy', 'Happy', '', image('blue'))
        revision = self.store.owner_revision(1, 'character', self.character)
        self.store.save_static_avatar(1, self.character, image('yellow'), revision)
        self.assertEqual(self.store.avatar_slot(1, self.character, 'neutral')['image'], image('green'))
        self.assertEqual(self.store.avatar_slot(1, self.character, 'happy')['image'], image('blue'))
        self.assertEqual(self.store.character_by_id(self.character)['avatar'], image('yellow'))
        self.store.clear_avatar_image(1, self.character, 'neutral', 1)
        self.assertIsNone(self.store.avatar_slot(1, self.character, 'neutral')['image'])
        self.assertEqual(self.store.character_by_id(self.character)['avatar'], image('yellow'))
        self.store.save_static_avatar(1, self.character, None, self.store.owner_revision(1, 'character', self.character))
        self.assertIsNone(self.store.character_by_id(self.character)['avatar'])
        self.assertEqual(self.store.avatar_slot(1, self.character, 'happy')['image'], image('blue'))

    def test_fallback_and_emotion_changes_reject_stale_or_cross_server_edits(self):
        revision = self.store.owner_revision(1, 'character', self.character)
        with self.assertRaises(ValueError):
            self.store.save_static_avatar(2, self.character, image('yellow'), revision)
        with self.assertRaises(ValueError):
            self.store.save_static_avatar(1, self.character, b'not an image', revision)
        self.store.save_avatar(1, self.character, 'neutral', 'Neutral', '', image('green'))
        with self.assertRaises(ConflictError):
            self.store.save_static_avatar(1, self.character, image('yellow'), revision)
        with self.assertRaises(ConflictError):
            self.store.clear_avatar_image(1, self.character, 'neutral', 0)
        with self.assertRaises(ValueError):
            self.store.clear_avatar_image(2, self.character, 'neutral', 1)
        self.assertEqual(self.store.character_by_id(self.character)['avatar'], image('red'))
        self.assertEqual(self.store.avatar_slot(1, self.character, 'neutral')['image'], image('green'))


class AvatarDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_emotion_then_static_fallback_across_streaming_and_missing_assets(self):
        bot = SkitBot(settings())
        store = bot.store
        world = store.create_space(1, 'World', 'world')
        store.bind_channel(1, 100, world)
        character = store.add_character(1, world, 'Alice', {'name': 'Alice'}, image('red'), [])
        store.set_cast(100, None, [character])

        class Hook(FakeWebhook):
            id, token = 999, 'test-token'
            default_avatar = None
            async def edit(self, *, name, avatar):
                self.default_avatar = avatar
                return self

        hook = Hook(2000)
        class Channel(FakeChannel):
            created = False
            missing = set()
            async def webhooks(self):
                return [hook] if self.created else []
            async def create_webhook(self, **kwargs):
                self.created = True
                hook.default_avatar = kwargs['avatar']
                return hook
            async def fetch_message(self, ident):
                if ident in self.missing:
                    raise discord.NotFound(SimpleNamespace(status=404, reason='Missing'), 'Asset deleted')
                return SimpleNamespace(id=ident)

        channel = Channel(100)
        bot.get_channel = lambda _: channel
        async def webhook(channel, row):
            return await bot._webhook_locked(channel, row, (100, character))
        bot._webhook = webhook
        class Models(FakeModels):
            reply = ''
            async def stream_text(self, role, system, messages):
                # Split the emotion header and force continuation messages.
                yield self.reply[:5]
                yield self.reply[5:] + 'word ' * 600
        models = Models([character])
        bot.engine.models = bot.models = models

        def publish(key, color, message):
            store.save_avatar(1, character, key, key.title(), '', image(color))
            return store.execute('INSERT INTO avatar_assets(guild_id,character_id,slot_key,image_hash,channel_id,message_id,url,created_at) VALUES(?,?,?,?,?,?,?,?)',
                (1, character, key, hashlib.sha256(image(color)).hexdigest(), 100, message, f'https://cdn.discordapp.com/attachments/100/{message}/avatar.png', 0))

        turn = 0
        async def check(header, expected_source, expected_url, expected_fallback):
            nonlocal turn
            turn += 1
            models.reply = header + '\nHello '
            bot.checked_avatar_assets.clear()  # Simulate first use after restart.
            start = len(hook.posts)
            await bot.run_scene(SceneContext(1, 100, None, world, 9, 1000 + turn, 'Hi', None, [], []), channel)
            self.assertFalse(channel.errors)
            self.assertGreater(len(hook.posts) - start, 1)
            self.assertEqual(hook.default_avatar, expected_fallback)
            for posted, options in zip(hook.posts[start:], hook.options[start:]):
                self.assertEqual(options['avatar_url'], expected_url)
                self.assertNotIn('<emotion>', posted.content)
                self.assertEqual(store.trace(posted.id)['avatar_source'], expected_source)

        try:
            await check('<emotion>neutral</emotion>', 'static', None, image('red'))
            publish('neutral', 'green', 701)
            happy = publish('happy', 'blue', 702)
            await check('<emotion>happy</emotion>', 'emotion', 'https://cdn.discordapp.com/attachments/100/702/avatar.png', image('red'))
            await check('No emotion header', 'emotion', 'https://cdn.discordapp.com/attachments/100/701/avatar.png', image('red'))
            channel.missing.add(702)
            # The static portrait wins when the chosen emotion fails, even if neutral has an image.
            await check('<emotion>happy</emotion>', 'static', None, image('red'))
            store.execute('DELETE FROM avatar_assets WHERE id=?', (happy,))
            resolved = await bot.resolve_avatar({'slot_key': 'happy', 'asset_id': happy, 'url': 'old-url'})
            self.assertIsNone(resolved['url'])
            self.assertIsNone(resolved['asset_id'])
            store.clear_avatar_image(1, character, 'happy', 1)
            store.clear_avatar_image(1, character, 'neutral', 1)
            store.save_static_avatar(1, character, image('yellow'), store.owner_revision(1, 'character', character))
            await check('<emotion>neutral</emotion>', 'static', None, image('yellow'))
            store.save_static_avatar(1, character, None, store.owner_revision(1, 'character', character))
            await check('<emotion>neutral</emotion>', 'default', None, None)
        finally:
            store.close()
