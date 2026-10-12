"""MNT-77, MNT-80 (D39): ModelConfigError carries codes, and the Monitoring turn log renders its stored English stage text per viewer."""
from __future__ import annotations

import os
import pickle
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from test_backend_resolver import FakeAnthropic, FakeOpenAI

from llmcord_core import admin, i18n
from llmcord_core.admin import AdminService
from llmcord_core.backend import _ProbeSource
from llmcord_core.config import Settings, key_hosts_from_env, profile_from_mapping
from llmcord_core.dashboard import turn_log_row
from llmcord_core.errors import ModelConfigError
from llmcord_core.models import ModelGateway, TurnMessage
from llmcord_core.store import Store

G = 1
HOSTS = key_hosts_from_env({})
OPENAI = {'provider': 'openai', 'model': 'gpt-x', 'context_tokens': 16000, 'api_key_env': 'OPENAI_API_KEY'}
UNPINNED = {'provider': 'compatible', 'model': 'm', 'context_tokens': 16000, 'base_url': 'https://tmp.example/v1', 'api_key_env': 'FAKE_API_KEY'}


class PseudoCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {'LLMCORD_I18N_PSEUDO': '1'})
        patcher.start()
        i18n.clear_cache()
        self.addCleanup(i18n.clear_cache)
        self.addCleanup(patcher.stop)
        self.pseudo, self.en = i18n.Translator('pseudo'), i18n.english()
        for target, fake in (('AsyncOpenAI', FakeOpenAI), ('AsyncAnthropic', FakeAnthropic)):
            p = mock.patch(f'llmcord_core.models.{target}', fake, create=True)
            p.start()
            self.addCleanup(p.stop)

    async def raised(self, mapping, name='cloud', environ=None, **kwargs):
        """The ModelConfigError the gateway raises for a dashboard profile (no network: it fails before any client call)."""
        profile = profile_from_mapping(name, mapping, source='dashboard')
        settings = Settings('', None, Path(':memory:'), 0, {name: profile}, name, name, name, {})
        gateway = ModelGateway(_ProbeSource(settings, HOSTS, environ or {}), **kwargs)
        try:
            with self.assertRaises(ModelConfigError) as caught:
                await gateway.text('dialogue', '', [TurnMessage('user', 'hi')])
        finally:
            await gateway.close()
        return caught.exception


class ModelConfigErrorTests(PseudoCase):
    async def test_missing_variable_text_type_and_code(self):
        """MNT-77: the missing-variable error keeps its English text, stays a RuntimeError, and carries a code and params."""
        for kwargs, where in (({}, "the bot's"), ({'environment': 'the dashboard'}, "the dashboard's")):
            with self.subTest(where=where):
                error = await self.raised(OPENAI, **kwargs)
                self.assertIsInstance(error, RuntimeError)
                self.assertEqual(str(error), f"OPENAI_API_KEY is not set in {where} environment (profile cloud)")
                self.assertIsNotNone(error.code)
                self.assertEqual(self.en.error(error), str(error))
                rendered = self.pseudo.error(error)
                self.assertTrue(rendered.startswith('[!! '), rendered)
                self.assertIn('OPENAI_API_KEY', rendered)
                self.assertIn('cloud', rendered)

    async def test_unpinned_key_text_type_and_code(self):
        """MNT-77: the pin error keeps 'Profile {name}: {inner}' in English and renders in the viewer's language."""
        error = await self.raised(UNPINNED, name='local', environ={'FAKE_API_KEY': 'fake-value-12345'})
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(str(error), 'Profile local: FAKE_API_KEY may only be sent to its pinned hosts; tmp.example is not one of them. '
                                     'Add it to LLMCORD_KEY_HOSTS to allow it.')
        self.assertIsNotNone(error.code)
        self.assertEqual(self.en.error(error), str(error))
        rendered = self.pseudo.error(error)
        self.assertTrue(rendered.startswith('[!! '), rendered)
        self.assertIn('FAKE_API_KEY', rendered)
        self.assertNotIn('Add it to LLMCORD_KEY_HOSTS', rendered)  # the inner message is rendered too, not pasted as English
        self.assertNotIn('fake-value-12345', rendered)

    def test_legacy_single_string_form_and_pickling(self):
        """MNT-77: ModelConfigError('text') still works (code None, str is the text) and coded errors survive pickling."""
        legacy = ModelConfigError('Profile x: broken')
        self.assertIsInstance(legacy, RuntimeError)
        self.assertEqual((str(legacy), legacy.code, legacy.params), ('Profile x: broken', None, {}))
        self.assertEqual(self.pseudo.error(legacy), 'Profile x: broken')
        self.assertEqual(str(pickle.loads(pickle.dumps(legacy))), 'Profile x: broken')

    async def test_coded_error_pickles_and_keeps_its_text(self):
        """MNT-77: a coded ModelConfigError round-trips through pickle with the same text, code and params."""
        error = await self.raised(OPENAI)
        again = pickle.loads(pickle.dumps(error))
        self.assertIsInstance(again, ModelConfigError)
        self.assertEqual((str(again), again.code, again.params), (str(error), error.code, error.params))


class ModelConfigEnvWholeMessageTests(PseudoCase):
    async def test_known_environments_use_whole_message_codes(self):
        """MNT-77 (i18n whole messages): 'the bot' and 'the dashboard' pick their own code with no spliced environment param; str is unchanged."""
        for environment, expected in (('the bot', "OPENAI_API_KEY is not set in the bot's environment (profile cloud)"),
                                      ('the dashboard', "OPENAI_API_KEY is not set in the dashboard's environment (profile cloud)")):
            with self.subTest(environment=environment):
                error = await self.raised(OPENAI, environment=environment)
                self.assertEqual(str(error), expected)
                self.assertNotIn('environment', error.params)
                self.assertEqual(self.en.error(error), expected)


    def setUp(self):
        super().setUp()
        self.store = Store(':memory:')
        self.addCleanup(self.store.close)
        world = self.store.create_space(G, 'W', 'world')
        self.ann = self.store.create_character(G, world, 'Ann')
        self.store.db.execute('UPDATE characters SET card=? WHERE id=?', ('{"description": "A gambler.", "personality": "Superstitious."}', self.ann))
        self.store.db.commit()
        app = SimpleNamespace(state=SimpleNamespace(store=self.store, auth=None, http=None, bot_token='x', config_profiles={}, config_roles={}, config_limits={}))
        self.service = AdminService(app)

    async def draft_with(self, error):
        class Gateway:
            async def structured_compiled(self, *args):
                raise error

            async def close(self):
                pass
        with mock.patch.object(admin, 'dashboard_gateway', lambda *a, **k: Gateway()):
            with self.assertRaises(ValueError) as caught:
                await self.service.draft_play_style(G, self.ann)
        return caught.exception

    async def test_value_error_keeps_english_and_renders_per_viewer(self):
        """MNT-77: the play-style path still raises a ValueError with the same English, and the pseudo viewer gets a translated message."""
        error = await self.draft_with(await self.raised(OPENAI))
        self.assertEqual(str(error), "OPENAI_API_KEY is not set in the bot's environment (profile cloud)")
        self.assertEqual(self.en.error(error), str(error))
        rendered = self.pseudo.error(error)
        self.assertTrue(rendered.startswith('[!! '), rendered)
        self.assertIn('OPENAI_API_KEY', rendered)

    async def test_long_message_is_still_capped_at_300_characters(self):
        """MNT-77: the admin-facing text keeps its 300-character cap."""
        error = await self.draft_with(await self.raised(UNPINNED, name='p' * 400, environ={'FAKE_API_KEY': 'fake-value-12345'}))
        self.assertEqual(len(str(error)), 300)
        self.assertTrue(str(error).startswith('Profile ' + 'p' * 50))

    async def test_legacy_string_error_is_capped_too(self):
        """MNT-77: a legacy single-string ModelConfigError reaches the admin as the same capped English ValueError."""
        error = await self.draft_with(ModelConfigError('x' * 500))
        self.assertEqual(str(error), 'x' * 300)
        self.assertEqual(self.pseudo.error(error), 'x' * 300)


def row(stage):
    return {'created_at': 1_700_000_000, 'channel_id': 5, 'stage': stage, 'model': 'm', 'input_tokens': 1, 'output_tokens': 2, 'status': 'ok',
            'reference_id': '', 'preview': ''}


class TurnLogStageTests(PseudoCase):
    def stage(self, text, t=None):
        return turn_log_row(row(text), 'UTC', lambda cid: '#c', t=t)['stage']

    def test_english_rows_render_exactly_as_stored(self):
        """MNT-80: in en every stored stage form (feature, purpose or role, failed at stage) shows byte-identical."""
        for text in ('reply · dialogue', 'reply · director', 'ambient · memory', 'memory · extraction', 'catchup · catchup', 'game · dialogue',
                     'reply · failed at dialogue generation', 'reply · failed at speaker selection', 'reply · failed at webhook delivery',
                     'summon · failed at saving character reply', 'catchup · failed at catchup', 'playstyle · dialogue'):
            with self.subTest(text=text):
                self.assertEqual(self.stage(text), text)
                self.assertEqual(self.stage(text, self.en), text)

    def test_empty_stage_is_a_dash(self):
        """MNT-80: an empty stage still shows the dash."""
        self.assertEqual(self.stage(''), '—')
        self.assertEqual(self.stage(None, self.pseudo), '—')

    def test_known_parts_render_in_the_viewers_language(self):
        """MNT-80: a known feature and role both render pseudo, joined by the same ' · ' separator."""
        out = self.stage('reply · dialogue', self.pseudo)
        parts = out.split(' · ')
        self.assertEqual(len(parts), 2, out)
        self.assertTrue(all(p.startswith('[!! ') for p in parts), out)

    def test_failed_at_is_one_message_and_the_stage_is_translated(self):
        """MNT-80: 'failed at {stage}' renders as a whole message with the stage in the viewer's language."""
        out = self.stage('reply · failed at dialogue generation', self.pseudo)
        feature, failed = out.split(' · ')
        self.assertTrue(feature.startswith('[!! '), out)
        self.assertTrue(failed.startswith('[!! '), out)
        self.assertNotIn('failed at', failed)
        self.assertNotIn('dialogue generation', failed)

    def test_unknown_parts_pass_through_unchanged(self):
        """MNT-80: old rows and custom text (unknown feature, purpose or no separator) are shown exactly as stored."""
        self.assertEqual(self.stage('custom · dialogue', self.pseudo).split(' · ', 1)[0], 'custom')
        self.assertTrue(self.stage('custom · dialogue', self.pseudo).split(' · ', 1)[1].startswith('[!! '))
        self.assertEqual(self.stage('reply · custom thing', self.pseudo).split(' · ', 1)[1], 'custom thing')
        for text in ('#scene reply', 'something else entirely', 'a · b', '/command', '  odd  ·  spacing  '):
            with self.subTest(text=text):
                self.assertEqual(self.stage(text, self.pseudo), text)

    def test_extra_separator_inside_a_part_is_handled(self):
        """MNT-80: the stage splits on the first ' · ' only; a further ' · ' inside the second part neither crashes nor loses text."""
        for text in ('reply · custom · thing', 'reply · failed at odd · stage', 'x · y · z · w'):
            with self.subTest(text=text):
                self.assertEqual(self.stage(text), text)
        self.assertTrue(self.stage('reply · failed at odd · stage', self.pseudo).startswith('[!! '))
        self.assertEqual(self.stage('x · y · z · w', self.pseudo), 'x · y · z · w')
        self.assertEqual(self.stage('reply · custom · thing', self.pseudo).split(' · ', 1)[1], 'custom · thing')


if __name__ == '__main__':
    unittest.main()
