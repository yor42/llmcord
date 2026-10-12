"""FEAT-41 (D38): the Backend editor's Fetch models lists a provider's models with the unsaved profile; the key goes only in a header."""
import asyncio
import unittest
import unittest.mock

import httpx

from llmcord_core.backend import MODEL_LIST_MAX_BYTES, ModelInfo, list_profile_models
from llmcord_core.config import key_hosts_from_env

FAKE_VALUE = 'fake-listing-value-123'
ENV = {'FIXTURE_API_KEY': FAKE_VALUE, 'OPENAI_API_KEY': FAKE_VALUE, 'ANTHROPIC_API_KEY': FAKE_VALUE}


def mapping(provider='compatible', **extra):
    return {'provider': provider, 'model': 'model-list', 'context_tokens': 1, **extra}


def run(coro):
    return asyncio.run(coro)


class Recorder:
    def __init__(self, handler):
        self.requests, self.handler = [], handler

    def __call__(self, request):
        self.requests.append(request)
        return self.handler(request)

    @property
    def transport(self):
        return httpx.MockTransport(self)


def hosts(**pins):
    return key_hosts_from_env({'LLMCORD_KEY_HOSTS': ','.join(f'{k}={v}' for k, v in pins.items())})


class ModelListTests(unittest.TestCase):
    def test_openai_lists_sorted_deduped_with_bearer_header_only(self):
        rec = Recorder(lambda r: httpx.Response(200, json={'data': [{'id': 'gpt-b'}, {'id': 'gpt-a'}, {'id': 'gpt-b'}]}))
        result = run(list_profile_models('p', mapping('openai', api_key_env='OPENAI_API_KEY'), hosts(), ENV, rec.transport))
        self.assertTrue(result.ok)
        self.assertEqual([m.id for m in result.models], ['gpt-a', 'gpt-b'])
        request = rec.requests[0]
        self.assertEqual((request.method, str(request.url)), ('GET', 'https://api.openai.com/v1/models'))
        self.assertEqual(request.headers['authorization'], f'Bearer {FAKE_VALUE}')
        self.assertNotIn(FAKE_VALUE, str(request.url) + result.message)

    def test_compatible_strips_models_prefix_and_reads_openrouter_fields(self):
        body = {'data': [{'id': 'models/gemini-x'}, {'id': 'vendor/vision', 'context_length': 128000, 'architecture': {'input_modalities': ['text', 'image']}},
                         {'id': 'vendor/text', 'context_length': 'big', 'architecture': {'input_modalities': ['text']}}]}
        rec = Recorder(lambda r: httpx.Response(200, json=body))
        result = run(list_profile_models('p', mapping(base_url='https://lister.example/v1/'), hosts(), ENV, rec.transport))
        self.assertEqual(result.models, (ModelInfo('gemini-x'), ModelInfo('vendor/text', None, False), ModelInfo('vendor/vision', 128000, True)))
        self.assertEqual(str(rec.requests[0].url), 'https://lister.example/v1/models')
        self.assertNotIn('authorization', rec.requests[0].headers)

    def test_compatible_with_key_sends_bearer(self):
        rec = Recorder(lambda r: httpx.Response(200, json={'data': [{'id': 'm'}]}))
        result = run(list_profile_models('p', mapping(base_url='https://lister.example/v1', api_key_env='FIXTURE_API_KEY'),
                                         hosts(FIXTURE_API_KEY='https://lister.example'), ENV, rec.transport))
        self.assertTrue(result.ok)
        self.assertEqual(rec.requests[0].headers['authorization'], f'Bearer {FAKE_VALUE}')

    def test_anthropic_paginates_and_reads_optional_fields(self):
        pages = {
            None: {'data': [{'id': 'claude-b', 'max_input_tokens': 200000, 'capabilities': {'image_input': {'supported': True}}}], 'has_more': True, 'last_id': 'claude-b'},
            'claude-b': {'data': [{'id': 'claude-a'}], 'has_more': False},
        }
        rec = Recorder(lambda r: httpx.Response(200, json=pages[r.url.params.get('after_id')]))
        result = run(list_profile_models('p', mapping('anthropic', api_key_env='ANTHROPIC_API_KEY'), hosts(), ENV, rec.transport))
        self.assertEqual(result.models, (ModelInfo('claude-a'), ModelInfo('claude-b', 200000, True)))
        self.assertEqual(len(rec.requests), 2)
        first = rec.requests[0]
        self.assertEqual((first.url.host, first.url.params['limit']), ('api.anthropic.com', '1000'))
        self.assertEqual((first.headers['x-api-key'], first.headers['anthropic-version']), (FAKE_VALUE, '2023-06-01'))
        self.assertNotIn(FAKE_VALUE, ''.join(str(r.url) for r in rec.requests))

    def test_anthropic_pagination_is_capped(self):
        rec = Recorder(lambda r: httpx.Response(200, json={'data': [{'id': f'm{len(rec.requests)}'}], 'has_more': True, 'last_id': 'x'}))
        result = run(list_profile_models('p', mapping('anthropic', api_key_env='ANTHROPIC_API_KEY'), hosts(), ENV, rec.transport))
        self.assertTrue(result.ok)
        self.assertLessEqual(len(rec.requests), 5)

    def test_key_pin_refusal_makes_no_request(self):
        rec = Recorder(lambda r: httpx.Response(200, json={'data': []}))
        result = run(list_profile_models('p', mapping(base_url='https://evil.example/v1', api_key_env='OPENAI_API_KEY'), hosts(OPENAI_API_KEY='https://api.openai.com'), ENV, rec.transport))
        self.assertFalse(result.ok)
        self.assertIn('pinned', result.message)
        self.assertEqual(rec.requests, [])
        self.assertNotIn(FAKE_VALUE, result.message)

    def test_missing_key_variable_makes_no_request(self):
        rec = Recorder(lambda r: httpx.Response(200, json={'data': []}))
        result = run(list_profile_models('p', mapping('openai', api_key_env='OPENAI_API_KEY'), hosts(), {}, rec.transport))
        self.assertEqual((result.ok, result.message, rec.requests), (False, "OPENAI_API_KEY is not set in the dashboard's environment.", []))

    def test_redirect_is_not_followed_and_key_not_sent_on(self):
        rec = Recorder(lambda r: httpx.Response(302, headers={'location': 'https://evil.example/models'}) if r.url.host != 'evil.example' else httpx.Response(200, json={'data': [{'id': 'x'}]}))
        result = run(list_profile_models('p', mapping('openai', api_key_env='OPENAI_API_KEY'), hosts(), ENV, rec.transport))
        self.assertFalse(result.ok)
        self.assertEqual([r.url.host for r in rec.requests], ['api.openai.com'])
        self.assertNotIn(FAKE_VALUE, result.message)

    def test_compatible_404_and_garbage_say_to_type_the_id(self):
        for response in (httpx.Response(404, text='nope'), httpx.Response(200, text='<html>not json</html>'), httpx.Response(200, json={'weird': True})):
            rec = Recorder(lambda r, response=response: response)
            result = run(list_profile_models('p', mapping(base_url='https://lister.example/v1'), hosts(), ENV, rec.transport))
            self.assertEqual((result.ok, result.message), (False, "This provider doesn't list its models; type the ID."))

    def test_error_status_and_empty_list_are_failures_without_the_key(self):
        rec = Recorder(lambda r: httpx.Response(401, text=f'bad key {FAKE_VALUE}'))
        result = run(list_profile_models('p', mapping('openai', api_key_env='OPENAI_API_KEY'), hosts(), ENV, rec.transport))
        self.assertFalse(result.ok)
        self.assertIn('401', result.message)
        self.assertNotIn(FAKE_VALUE, result.message)
        rec = Recorder(lambda r: httpx.Response(200, json={'data': []}))
        result = run(list_profile_models('p', mapping('openai', api_key_env='OPENAI_API_KEY'), hosts(), ENV, rec.transport))
        self.assertFalse(result.ok)
        self.assertIn('no models', result.message)

    def test_requests_ask_for_identity_encoding(self):
        rec = Recorder(lambda r: httpx.Response(200, json={'data': [{'id': 'm'}]}))
        run(list_profile_models('p', mapping('openai', api_key_env='OPENAI_API_KEY'), hosts(), ENV, rec.transport))
        self.assertEqual(rec.requests[0].headers['accept-encoding'], 'identity')

    def test_401_without_a_key_says_a_key_is_needed(self):
        rec = Recorder(lambda r: httpx.Response(401))
        result = run(list_profile_models('p', mapping(base_url='https://lister.example/v1'), hosts(), ENV, rec.transport))
        self.assertEqual(result.message, 'The provider needs an API key (HTTP 401).')
        keyed = run(list_profile_models('p', mapping('openai', api_key_env='OPENAI_API_KEY'), hosts(), ENV, rec.transport))
        self.assertEqual(keyed.message, 'The provider refused the key (HTTP 401).')

    def test_overall_deadline_stops_a_slow_provider(self):
        async def slow(request):
            await asyncio.sleep(5)
            return httpx.Response(200, json={'data': [{'id': 'm'}]})
        with unittest.mock.patch('llmcord_core.backend.MODEL_LIST_DEADLINE_SECONDS', 0.05):
            result = run(list_profile_models('p', mapping(base_url='https://lister.example/v1'), hosts(), ENV, httpx.MockTransport(slow)))
        self.assertEqual((result.ok, result.message), (False, 'The provider did not respond in time.'))

    def test_oversized_body_is_refused(self):
        rec = Recorder(lambda r: httpx.Response(200, content=b'[' + b' ' * (MODEL_LIST_MAX_BYTES + 10) + b']'))
        result = run(list_profile_models('p', mapping(base_url='https://lister.example/v1'), hosts(), ENV, rec.transport))
        self.assertEqual((result.ok, 'too large' in result.message), (False, True))

    def test_network_error_never_raises_or_leaks_the_key(self):
        def boom(request):
            raise httpx.ConnectError(f'failed for {FAKE_VALUE}')
        with self.assertNoLogs(level='WARNING'):
            result = run(list_profile_models('p', mapping('openai', api_key_env='OPENAI_API_KEY'), hosts(), ENV, httpx.MockTransport(boom)))
        self.assertEqual((result.ok, FAKE_VALUE in result.message), (False, False))

    def test_blank_model_form_values_are_accepted_by_the_dashboard_helper(self):
        from llmcord_core.dashboard import profile_mapping
        with self.assertRaises(ValueError):
            profile_mapping({'provider': 'openai', 'model': '', 'context_tokens': 1})
        self.assertTrue(profile_mapping({'provider': 'openai', 'model': 'model-list', 'context_tokens': 1}))

    def test_single_model_message_is_singular(self):
        rec = Recorder(lambda r: httpx.Response(200, json={'data': [{'id': 'only'}]}))
        result = run(list_profile_models('p', mapping(base_url='https://lister.example/v1'), hosts(), ENV, rec.transport))
        self.assertEqual(result.message, '1 model found.')


if __name__ == '__main__':
    unittest.main()
