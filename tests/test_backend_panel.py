"""D24 step 4 (FEAT-11): Backend tab view-model helpers, form-to-mapping diff, and the prompt preview following dashboard roles."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from llmcord_core.backend import effective_backend
from llmcord_core.config import profile_from_mapping
from llmcord_core.dashboard import changed_fields, editor_initial, role_options, key_status, profile_mapping, profile_mapping_of, profile_view, roles_detail
from llmcord_core.prompts import default_bundle
from llmcord_core.web import create_app

CONFIG = {
    'main': profile_from_mapping('main', {'provider': 'openai', 'model': 'gpt-x', 'context_tokens': 8000, 'api_key_env': 'OPENAI_API_KEY'}),
    'local': profile_from_mapping('local', {'provider': 'compatible', 'model': 'm', 'context_tokens': 4000, 'base_url': 'http://localhost:1/v1'}),
}
FORM = {'provider': 'compatible', 'model': ' m2 ', 'context_tokens': 4000.0, 'base_url': ' http://localhost:2/v1 ', 'api_key': '', 'supports_images': False,
        'stream_usage': True, 'billing_tier': 'paid', 'timeout_seconds': 120, 'max_retries': 1}


def roles(**kw):
    return {'dialogue': None, 'director': None, 'memory': None, 'revision': 0, 'version': 0, **kw}


def row(name, data, revision=1):
    return {'name': name, 'data': data, 'revision': revision, 'updated_at': 0}


class ViewModelTests(unittest.TestCase):
    def test_sources_roles_and_key_status(self):
        """FEAT-11: config, dashboard-only and overriding profiles are labelled; roles become badges; keys show set/missing/none."""
        rows = [row('main', {'provider': 'openai', 'model': 'gpt-y', 'context_tokens': 9000, 'api_key_env': 'OPENAI_API_KEY'}, 3),
                row('new', {'provider': 'compatible', 'model': 'n', 'context_tokens': 1000, 'base_url': 'http://h/v1'}, 5)]
        backend = effective_backend(CONFIG, {'dialogue': 'main'}, rows, roles(director='new'))
        views = {v['name']: v for v in profile_view(backend, CONFIG, rows, {'OPENAI_API_KEY': 'sk-secret-value'})}
        self.assertEqual(list(views), ['local', 'main', 'new'])
        self.assertEqual((views['local']['source'], views['main']['source'], views['new']['source']),
                         ('config.yaml', 'Dashboard (overrides config.yaml)', 'Dashboard'))
        self.assertEqual((views['main']['roles'], views['new']['roles'], views['local']['roles']), (['dialogue'], ['director'], []))
        self.assertEqual((views['main']['revision'], views['new']['revision'], views['local']['revision']), (3, 5, None))
        self.assertEqual((views['main']['key'], views['main']['key_state'], views['local']['key']), ('${OPENAI_API_KEY} (set)', 'set', 'No key'))
        self.assertEqual(key_status('OPENAI_API_KEY', {})[1], '${OPENAI_API_KEY} (missing)')

    def test_no_environment_value_reaches_the_view_model(self):
        """FEAT-11 (D17): the view-model only knows whether a variable has a value, never the value."""
        rows = [row('main', {'provider': 'openai', 'model': 'g', 'context_tokens': 9000, 'api_key_env': 'OPENAI_API_KEY'})]
        backend = effective_backend({}, {}, rows, roles())
        views = profile_view(backend, {}, rows, {'OPENAI_API_KEY': 'sk-canary-123'})
        self.assertNotIn('sk-canary-123', repr(views))

    def test_skipped_rows_stay_listed(self):
        """FEAT-11: a saved row the merge skipped is listed so it can be deleted."""
        rows = [row('broken', {'provider': 'nope'}, 2), row('Bad Name', {}, 1)]
        backend = effective_backend({}, {}, rows, roles())
        views = profile_view(backend, {}, rows)
        self.assertEqual([(v['name'], v['kind'], v['revision']) for v in views], [('broken', 'skipped', 2)])
        self.assertTrue(backend.problems)

    def test_roles_detail_lists_only_changed_roles(self):
        self.assertEqual(roles_detail(roles(director='a'), {'dialogue': None, 'director': 'b', 'memory': None}), {'director': {'from': 'a', 'to': 'b'}})


class EditorHelperTests(unittest.TestCase):
    def test_role_options_list_effective_profiles_only(self):
        """FEAT-11: role options are the config default plus the effective profiles; a skipped saved row is not offered."""
        rows = [row('broken', {'provider': 'nope'}, 2), row('fine', {'provider': 'compatible', 'model': 'm', 'context_tokens': 100, 'base_url': 'http://h/v1'})]
        backend = effective_backend(CONFIG, {'dialogue': 'main'}, rows, roles())
        options = role_options(None, backend.profiles, 'main')
        self.assertEqual(sorted(options), ['', 'fine', 'local', 'main'])
        self.assertEqual(options[''], 'Use config.yaml (main)')
        self.assertNotIn('broken', options)
        self.assertEqual(role_options(None, {}, None), {'': 'Use config.yaml (none)'})

    def test_role_options_mark_an_unavailable_saved_value(self):
        """FEAT-11: a saved role naming a profile that no longer resolves stays selectable, labelled not available."""
        options = role_options('gone', {'main': CONFIG['main']}, 'main')
        self.assertEqual(options['gone'], 'gone (not available)')
        self.assertEqual(role_options('main', {'main': CONFIG['main']}, 'main')['main'], 'main')

    def test_editor_initial_coerces_unknown_choices(self):
        """FEAT-11: an unknown provider or billing tier becomes the default, an unknown reasoning effort is dropped, and the input is not mutated."""
        saved = {'provider': 'mystery', 'billing_tier': 'gold', 'reasoning_effort': 'extreme', 'model': 'm'}
        initial = editor_initial(saved)
        self.assertEqual(initial, {'provider': 'compatible', 'billing_tier': 'paid', 'model': 'm'})
        self.assertEqual(saved['provider'], 'mystery')
        good = {'provider': 'anthropic', 'billing_tier': 'free', 'reasoning_effort': 'high'}
        self.assertEqual(editor_initial(good), good)
        self.assertIsNot(editor_initial(good), good)


class FormMappingTests(unittest.TestCase):
    def test_mapping_trims_converts_and_sets_only_applicable_keys(self):
        mapping = profile_mapping(FORM)
        self.assertEqual(mapping['model'], 'm2')
        self.assertEqual(mapping['base_url'], 'http://localhost:2/v1')
        self.assertIs(type(mapping['context_tokens']), int)
        self.assertNotIn('api_key_env', mapping)
        self.assertNotIn('reasoning_effort', mapping)
        self.assertNotIn('input_cost_per_million', mapping)
        anthropic = profile_mapping({**FORM, 'provider': 'anthropic', 'reasoning_effort': 'high', 'api_key': '${ANTHROPIC_API_KEY}'})
        self.assertEqual((anthropic.get('base_url'), anthropic.get('reasoning_effort'), anthropic['api_key_env']), (None, None, 'ANTHROPIC_API_KEY'))
        self.assertEqual(profile_mapping({**FORM, 'reasoning_effort': 'low', 'input_cost_per_million': '1.5'})['input_cost_per_million'], 1.5)
        profile_from_mapping('x', profile_mapping(FORM), source='dashboard')

    def test_errors_never_echo_a_pasted_key(self):
        with self.assertRaises(ValueError) as caught:
            profile_mapping({**FORM, 'api_key': 'sk-pasted-secret'})
        self.assertNotIn('sk-pasted-secret', str(caught.exception))
        for bad in ({'model': ' '}, {'context_tokens': None}, {'context_tokens': 1.5}, {'context_tokens': 0}, {'timeout_seconds': 'abc'}, {'max_retries': 1.5}):
            with self.assertRaises(ValueError, msg=bad):
                profile_mapping({**FORM, **bad})

    def test_changed_fields_are_names_only(self):
        initial = profile_mapping_of(CONFIG['local'])
        same = profile_mapping({**FORM, 'model': 'm', 'base_url': 'http://localhost:1/v1'})
        self.assertEqual(changed_fields(same, initial), [])
        edited = profile_mapping({**FORM, 'model': 'm', 'base_url': 'http://localhost:1/v1', 'api_key': '${OPENAI_API_KEY}', 'context_tokens': 5000})
        self.assertEqual(changed_fields(edited, initial), ['api_key_env', 'context_tokens'])
        self.assertEqual(changed_fields(same, None), sorted(same))


class PreviewFollowsBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_dashboard_role_changes_preview_provider_and_budget(self):
        """FEAT-11 (D24): the prompt preview uses the effective backend, so assigning a dashboard profile to a role changes its provider and budget."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yaml'
            path.write_text('models:\n  dialogue: main\n  director: main\n  memory: main\n  profiles:\n    main:\n      provider: openai\n      model: g\n      context_tokens: 30000\n'
                            'limits:\n  max_input_tokens: 12000\n  max_output_tokens: 700\n', encoding='utf-8')
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[]))) as http:
                app = create_app(':memory:', 'https://pi.test', 'c', 's', 'b', http, enable_dashboard=False, config_path=str(path))
                store, admin = app.state.store, app.state.admin
                self.assertEqual((admin.providers['director'], admin.budgets['director']), ('openai', 12000))
                store.save_model_profile('claude', {'provider': 'anthropic', 'model': 'c', 'context_tokens': 5000}, None)
                store.save_model_roles({'director': 'claude'}, 0, set(app.state.config_profiles))
                seen = {}
                with patch('llmcord_core.admin.compile_prompt') as compile_prompt:
                    admin.preview_prompt(1, default_bundle(), 'director', None, 'Sample', '')
                    seen['director'] = compile_prompt.call_args.args[4:6]
                    admin.preview_prompt(1, default_bundle(), 'dialogue', None, 'Sample', '')
                    seen['dialogue'] = compile_prompt.call_args.args[4:6]
                self.assertEqual(seen['director'], ('anthropic', 5000 - 700))
                self.assertEqual(seen['dialogue'], ('openai', 12000))
                store.close()

    async def test_unreadable_config_leaves_an_empty_set_and_a_message(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.yaml'
            path.write_text('models:\n  profiles:\n    broken:\n      provider: nope\n', encoding='utf-8')
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[]))) as http:
                app = create_app(':memory:', 'https://pi.test', 'c', 's', 'b', http, enable_dashboard=False, config_path=str(path))
                self.assertEqual((app.state.config_profiles, app.state.config_roles), ({}, {}))
                self.assertTrue(app.state.config_error)
                self.assertEqual(app.state.admin.providers['dialogue'], 'compatible')
                app.state.store.close()
    async def test_malformed_config_still_starts_with_a_generic_message(self):
        """FEAT-11: invalid YAML, a models list and a null profiles section each leave empty sets and a message that quotes nothing from the file."""
        for text in ('models: [unclosed SECRET-TEXT\n  : :', 'models: [1]\n', 'models:\n  profiles: null\n  dialogue: SECRET-TEXT\n'):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'config.yaml'
                path.write_text(text, encoding='utf-8')
                async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[]))) as http:
                    app = create_app(':memory:', 'https://pi.test', 'c', 's', 'b', http, enable_dashboard=False, config_path=str(path))
                    self.assertEqual((app.state.config_profiles, app.state.config_roles), ({}, {}))
                    self.assertTrue(app.state.config_error)
                    self.assertNotIn('SECRET-TEXT', app.state.config_error)
                    self.assertNotIn('unclosed', app.state.config_error)
                    app.state.store.close()


if __name__ == '__main__':
    unittest.main()
