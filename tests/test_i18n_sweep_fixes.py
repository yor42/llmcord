import os
import tempfile
import types
import unittest
from unittest import mock

import httpx

from llmcord_core import catchup, i18n, web
from llmcord_core.admin import AdminService
from llmcord_core.dashboard import USAGE_ROLE_WORDS
from llmcord_core.prompts import default_bundle
from llmcord_core.web import create_app

OWNER_ROWS = dict(
    spaces=[{'id': 1, 'kind': 'world', 'name': 'Annex'}, {'id': 2, 'kind': 'hub', 'name': 'Core'}],
    characters=[{'id': 3, 'name': 'Mira'}], channels=[{'channel_id': 9}], lorebooks=[{'id': 4, 'name': 'Tales'}], thread_scopes=[55551234, 55555678])


def owners(t):
    admin = object.__new__(AdminService)
    r = OWNER_ROWS
    return admin.owners_from(1, r['spaces'], r['characters'], r['channels'], r['lorebooks'], r['thread_scopes'], {9: '#general'}, {55555678: '#t'}, t=t)


class SweepFixTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {'LLMCORD_I18N_PSEUDO': '1'})
        patcher.start()
        i18n.clear_cache()
        self.addCleanup(i18n.clear_cache)
        self.addCleanup(patcher.stop)
        self.pseudo = i18n.Translator('pseudo')

    def test_owner_labels(self):
        """FEAT-44/45: owner labels are whole translated messages; English and the select keys are unchanged."""
        labels = [o['label'] for o in owners(None)]
        self.assertEqual(labels, ['World: Annex', 'Hub: Core', 'Character: Mira', 'Channel: #general', 'Server: Server-wide lore', 'Lorebook: Tales',
                                  'Archived thread …1234', 'Thread #t'])
        self.assertEqual([(o['kind'], o['id']) for o in owners(None)][:2], [('space', 1), ('space', 2)])
        self.assertTrue(all(o['label'].startswith('[!! ') for o in owners(self.pseudo)))

    def test_thread_label(self):
        """FEAT-44/45: thread labels (unknown, archived, named) render in the given language."""
        english = [AdminService.thread_label(5551234, n) for n in (None, {}, {5551234: '#x'})]
        self.assertEqual(english, ['Thread …1234', 'Archived thread …1234', 'Thread #x'])
        self.assertTrue(AdminService.thread_label(5551234, {}, t=self.pseudo).startswith('[!! '))

    def test_preview_sample_values(self):
        """FEAT-44/45: the preview's sample user, character and world come from the translator."""
        import asyncio

        async def run(t):
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[]))) as http:
                app = create_app(':memory:', 'https://pi.test', 'c', 's', 'b', http, enable_dashboard=False)
                try:
                    request = app.state.admin.preview_prompt(1, default_bundle(), 'dialogue', None, 'Sample', 'Earlier sample', **({'t': t} if t else {}))
                finally:
                    app.state.store.close()
            return '\n'.join(m.text for m in request.messages)
        english = asyncio.run(run(None))
        self.assertIn('Local time for Sample user: ', english)
        self.assertIn('Sample user', asyncio.run(run(i18n.english())))
        self.assertNotIn('Sample user', asyncio.run(run(self.pseudo)))

    def test_usage_roles_and_billing_tier_words(self):
        """FEAT-44/45: the Used for column and Billing tier options render through the translator, English equals the raw keys."""
        english = i18n.english()
        self.assertEqual(', '.join(english.render(USAGE_ROLE_WORDS[k]) for k in ('dialogue', 'director', 'memory')), 'dialogue, director, memory')
        self.assertTrue(self.pseudo.render(USAGE_ROLE_WORDS['memory']).startswith('[!! '))
        self.assertEqual([english.p('billing tier', 'paid'), english.p('billing tier', 'free')], ['paid', 'free'])
        self.assertTrue(self.pseudo.p('billing tier', 'paid').startswith('[!! '))

    def test_config_error_keeps_english_text_and_a_renderable_value(self):
        """FEAT-44/45: app.state.config_error stays English; config_error_value renders in the viewer's language."""
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'bad.yaml')
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write('models: [unclosed')
            app = types.SimpleNamespace(state=types.SimpleNamespace())
            web._load_config(app, path)
        self.assertEqual(app.state.config_error, 'the file could not be read')
        self.assertIs(app.state.config_error_value, web.CONFIG_UNREADABLE)
        self.assertEqual(i18n.english().render(app.state.config_error_value), 'the file could not be read')
        self.assertTrue(self.pseudo.render(app.state.config_error_value).startswith('[!! '))

    def test_config_error_keeps_a_profile_value_error_as_is(self):
        """FEAT-45: a ValueError from profile loading is kept as config_error text and config_error_value."""
        error = ValueError('bad profile')
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'ok.yaml')
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write('models: {}\n')
            app = types.SimpleNamespace(state=types.SimpleNamespace())
            with mock.patch.object(web, 'profiles_from_models', side_effect=error):
                web._load_config(app, path)
        self.assertEqual(app.state.config_error, 'bad profile')
        self.assertIs(app.state.config_error_value, error)
        self.assertEqual(app.state.config_profiles, {})

    def test_sample_input_default(self):
        """FEAT-44/45: the preview's Sample input default is a translated message, English 'Hello!'."""
        self.assertEqual(i18n.english().p('sample input', 'Hello!'), 'Hello!')
        self.assertTrue(self.pseudo.p('sample input', 'Hello!').startswith('[!! '))

    def test_nothing_new(self):
        """FEAT-44/45: catchup returns a Phrase for the edge; nothing_new() still gives the English text."""
        for hours, text in ((None, 'Nothing new here since you last spoke.'), (1, 'Nothing new here in the last hour.'), (6, 'Nothing new here in the last 6 hours.')):
            self.assertEqual(catchup.nothing_new(hours), text)
            self.assertEqual(i18n.english().render(catchup.nothing_new_phrase(hours)), text)
            self.assertTrue(self.pseudo.render(catchup.nothing_new_phrase(hours)).startswith('[!! '))


if __name__ == '__main__':
    unittest.main()
