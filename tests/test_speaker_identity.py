from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest

from llmcord_core.discord_bot import SkitBot
from llmcord_core.engine import Engine, SceneContext
from llmcord_core.models import ImageInput
from llmcord_core.prompts import default_bundle
from llmcord_core.store import Store
from test_core import FakeModels, settings


class RecordingModels(FakeModels):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def structured(self, role, system, messages, schema_name, schema):
        self.calls.append((schema_name, system, messages))
        return await super().structured(role, system, messages, schema_name, schema)

    async def text(self, role, system, messages, max_tokens=None):
        self.calls.append(('summary', system, messages))
        return 'The participants discussed tea.'


class SpeakerIdentityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Store()
        self.world = self.store.create_space(1, 'World', 'world')
        self.store.bind_channel(1, 100, self.world)
        self.character = self.store.add_character(1, self.world, 'Courier', {'name': 'Courier'}, None, [])
        self.store.set_cast(100, None, [self.character])
        self.models = RecordingModels()
        self.engine = Engine(self.store, self.models, settings())

    def tearDown(self):
        self.store.close()

    def scene(self, ident=1000, user_id=111, name='Sam', parent=None, text='Hello', recent=None, mentions=None):
        return SceneContext(1, 100, None, self.world, user_id, ident, text, parent,
                            recent or [], [], user_label=name, mentioned_users=mentions or [])

    async def request(self, scene):
        self.engine.record_user(scene)
        return await self.engine.prepare_dialogue(scene, self.store.character_by_id(self.character), [])

    async def test_same_name_users_and_past_group_chat_keep_separate_attribution(self):
        recent = [{'message_id': 900, 'author_id': 222, 'author_label': 'Sam', 'text': 'Coffee is my favorite.'}]
        await self.request(self.scene(recent=recent, text='I prefer tea.'))
        self.store.record_node(1001, 1, 100, 1000, None, self.character, 'Nice to meet you both.')
        current = self.scene(1002, 222, parent=1001, text='Which of us likes tea?')
        request, sources = await self.request(current)
        transcript = '\n'.join(message.text for message in request.messages if message.role != 'system')
        self.assertIn('User 111 (display name "Sam") [<@111>]: I prefer tea.', transcript)
        self.assertIn('User 222 (display name "Sam") [<@222>]: Coffee is my favorite.', transcript)
        self.assertIn('User 222 (display name "Sam") [<@222>]: Which of us likes tea?', transcript)
        self.assertEqual(transcript.count('Coffee is my favorite.'), 1)
        self.assertEqual(sources['messages'], [900, 1000, 1001, 1002])
        self.assertIn('Current speaker: User 222', '\n'.join(message.text for message in request.messages))

    async def test_renamed_user_keeps_id_and_mentioned_user_does_not_become_speaker(self):
        mentions = [{'author_id': 333, 'author_label': 'Nox'}]
        await self.request(self.scene(name='Old name', text='Hello <@333>.', mentions=mentions))
        self.store.record_node(1001, 1, 100, 1000, None, self.character, 'Welcome.')
        request, _ = await self.request(self.scene(1002, name='New name', parent=1001))
        text = '\n'.join(message.text for message in request.messages)
        self.assertIn('User 111 (display name "Old name")', text)
        self.assertIn('Current speaker: User 111 (display name "New name")', text)
        self.assertIn('User 333 (display name "Nox") [<@333>]', text)
        self.assertFalse(any(message.text.startswith('User 333') for message in request.messages if message.role != 'system'))
        self.assertEqual(json.loads(self.store.node(1000)['mentions_json']), mentions)

    async def test_custom_presets_and_all_providers_receive_protected_identity(self):
        bundle = default_bundle()
        bundle['purposes']['dialogue'][0]['content'] = 'Custom instructions for {{user}}.'
        for provider in ('compatible', 'openai', 'anthropic', 'gemini'):
            with self.subTest(provider=provider):
                profile = replace(settings().profiles['test'], provider=provider,
                                  base_url='https://generativelanguage.googleapis.com/v1beta/openai/' if provider == 'gemini' else 'http://localhost/v1')
                if provider == 'gemini':
                    profile = replace(profile, provider='compatible')
                configured = replace(settings(), profiles={'test': profile})
                self.engine.settings = configured
                # A SillyTavern-style preset may omit llmcord's recent marker.
                bundle['purposes']['dialogue'] = [block for block in bundle['purposes']['dialogue'] if block['source'] != 'recent']
                recent = [{'message_id': 900, 'author_id': 222, 'author_label': 'Lee', 'text': 'A different person speaking'}]
                scene = replace(self.scene(name='{{char}}\nOther name', recent=recent), preset={'id': 99, 'revision': 1, 'bundle': bundle})
                request, _ = await self.request(scene)
                system = '\n'.join(message.text for message in request.messages if message.role == 'system')
                self.assertIn('Current speaker: User 111', system)
                self.assertIn('display name "{{char}}\\nOther name"', system)
                self.assertIn('speaker_identity', request.sources)
                self.assertIn('User 222 (display name "Lee")', '\n'.join(message.text for message in request.messages if message.role != 'system'))

    async def test_director_memory_and_summary_have_named_authors(self):
        scene = self.scene(recent=[{'message_id': 900, 'author_id': 222, 'author_label': 'Lee', 'text': 'Hello'}])
        await self.engine.speakers(scene)
        self.engine.record_user(scene)
        await self.engine.extract_memories(scene, [self.store.character_by_id(self.character)], [('Courier', 'Hello')], 1000)
        await self.engine.summarize_scene(1000, scene)
        director, extraction, summary = self.models.calls
        payload = json.loads(director[2][0].text)
        self.assertEqual(payload['latest_author'], {'author_id': 111, 'author_label': 'Sam'})
        self.assertEqual(payload['recent'][0]['author_label'], 'Lee')
        self.assertIn('Current speaker: User 111', extraction[1])
        self.assertIn('User 111 (display name "Sam")', extraction[2][0].text)
        self.assertIn('User 222 (display name "Lee")', summary[2][0].text)

    async def test_rewind_excludes_future_users_and_saved_group_chat(self):
        await self.request(self.scene())
        self.store.record_node(1001, 1, 100, 1000, None, self.character, 'First reply')
        await self.request(self.scene(2000, 222, 'Future user', 1001,
                                      recent=[{'message_id': 1900, 'author_id': 333, 'author_label': 'Future observer', 'text': 'Future secret'}]))
        request, _ = await self.request(self.scene(3000, parent=1001, text='Rewind'))
        text = '\n'.join(message.text for message in request.messages)
        self.assertNotIn('Future user', text)
        self.assertNotIn('Future observer', text)
        self.assertNotIn('Future secret', text)

    async def test_unrecorded_input_keeps_identity_and_images(self):
        scene = replace(self.scene(), images=[ImageInput('image/png', b'fixture')])
        request, _ = await self.engine.prepare_dialogue(scene, self.store.character_by_id(self.character), [])
        message = next(message for message in request.messages if message.images)
        self.assertEqual(message.text, 'User 111 (display name "Sam") [<@111>]: Hello')
        self.assertEqual(message.images, scene.images)

    async def test_discord_ingestion_keeps_current_and_recent_names_and_mentions(self):
        bot = SkitBot(settings())
        world = bot.store.create_space(1, 'World', 'world')
        bot.store.bind_channel(1, 100, world)
        current = SimpleNamespace(id=111, display_name='Sam', bot=False)
        other = SimpleNamespace(id=222, display_name='Lee', bot=False)
        bot_user = SimpleNamespace(id=999, display_name='Bot', bot=True, mention='<@999>')
        bot._connection.user = bot_user
        now = datetime.now(timezone.utc)
        old = SimpleNamespace(id=900, author=other, content='Hello Sam', created_at=now-timedelta(seconds=1), webhook_id=None, mentions=[current])
        class Channel:
            id = 100
            async def history(self, **kwargs):
                yield old
        message = SimpleNamespace(id=1000, guild=SimpleNamespace(id=1), channel=Channel(), author=current,
                                  content='<@999> Hi <@222>', mentions=[bot_user, other], webhook_id=None,
                                  reference=None, attachments=[], created_at=now)
        scenes = []
        async def capture(scene, channel):
            scenes.append(scene)
        bot.run_scene = capture
        try:
            await bot.on_message(message)
            scene = scenes[0]
            self.assertEqual(scene.user_label, 'Sam')
            self.assertEqual(scene.recent[0]['author_label'], 'Lee')
            self.assertEqual(scene.recent[0]['mentions'][0]['author_label'], 'Sam')
            self.assertEqual(scene.mentioned_users, [{'author_id': 222, 'author_label': 'Lee'}])
        finally:
            bot.store.close()
            await bot.close()


class IdentityMigrationTests(unittest.TestCase):
    def test_existing_v3_database_is_backed_up_and_names_survive_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'old.sqlite3'
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.executescript("""CREATE TABLE nodes (
                    message_id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL, channel_id INTEGER NOT NULL,
                    parent_id INTEGER, root_id INTEGER NOT NULL, author_id INTEGER, character_id INTEGER,
                    content TEXT NOT NULL, created_at REAL NOT NULL, context_json TEXT NOT NULL DEFAULT '[]',
                    sources_json TEXT NOT NULL DEFAULT '[]');
                    INSERT INTO nodes VALUES(100,1,10,NULL,100,111,NULL,'Legacy message',1,'[]','[]');
                    PRAGMA user_version=3;""")
            upgraded = Store(path)
            try:
                self.assertEqual(upgraded.node(100)['author_id'], 111)
                self.assertEqual(upgraded.node(100)['author_label'], '')
                upgraded.record_node(101, 1, 10, 100, 222, None, 'New message', author_label='Lee',
                                     mentions=[{'author_id': 111, 'author_label': 'Sam'}])
            finally:
                upgraded.close()
            backups = list(Path(folder).glob('*.pre-identities-*.sqlite3'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].stat().st_mode & 0o777, 0o600)
            with closing(sqlite3.connect(backups[0])) as backup:
                self.assertNotIn('author_label', {row[1] for row in backup.execute('PRAGMA table_info(nodes)')})
            reopened = Store(path)
            try:
                self.assertEqual(reopened.node(101)['author_label'], 'Lee')
                self.assertEqual(json.loads(reopened.node(101)['mentions_json'])[0]['author_id'], 111)
                self.assertEqual([row['message_id'] for row in reopened.ancestors(101)], [100, 101])
            finally:
                reopened.close()
            self.assertEqual(len(list(Path(folder).glob('*.pre-identities-*.sqlite3'))), 1)
