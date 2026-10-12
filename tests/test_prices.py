"""FEAT-42 (D38): the Backend editor's Look up prices reads public price lists (LiteLLM, or OpenRouter's own list); no key, GET only, cached 24 h."""
import asyncio
import unittest
import unittest.mock

import httpx

from llmcord_core import prices
from llmcord_core.prices import LITELLM_URL, NOT_FOUND, PRICE_MAX_BYTES, suggest_prices

LITELLM = {
    'sample_spec': {'input_cost_per_token': 0.5, 'output_cost_per_token': 0.5},
    'gpt-x': {'input_cost_per_token': 2.5e-06, 'output_cost_per_token': 1e-05, 'cache_read_input_token_cost': 1.25e-06},
    'gpt-x-mini': {'input_cost_per_token': 1.5e-07, 'output_cost_per_token': 6e-07},
    'anthropic/claude-x': {'input_cost_per_token': 3e-06, 'output_cost_per_token': 1.5e-05, 'cache_read_input_token_cost': 3e-07},
    'gemini/gemini-x': {'input_cost_per_token': 1e-07, 'output_cost_per_token': 4e-07, 'litellm_provider': 'gemini'},
    'gemini-x': {'input_cost_per_token': 9e-07, 'output_cost_per_token': 9e-06, 'litellm_provider': 'vertex_ai-language-models'},
    'vertex-only': {'input_cost_per_token': 1e-06, 'output_cost_per_token': 2e-06, 'litellm_provider': 'vertex_ai-language-models'},
    'plain-openai': {'input_cost_per_token': 1e-06, 'output_cost_per_token': 2e-06, 'litellm_provider': 'openai'},
    'openai/only-prefixed': {'input_cost_per_token': 1e-06, 'output_cost_per_token': 2e-06},
    'free-model': {'input_cost_per_token': 0, 'output_cost_per_token': 0},
    'no-prices': {'max_tokens': 4096},
    'negative': {'input_cost_per_token': -1, 'output_cost_per_token': 'x'},
}
OPENROUTER = {'data': [
    {'id': 'vendor/model', 'pricing': {'prompt': '0.000003', 'completion': '0.000015', 'input_cache_read': '0.0000003'}},
    {'id': 'vendor/no-cache', 'pricing': {'prompt': '0.000001', 'completion': '0.000002'}},
    {'id': 'vendor/model-pro', 'pricing': {'prompt': '0.1', 'completion': '0.2'}},
]}


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


def litellm_recorder(body=None):
    return Recorder(lambda r: httpx.Response(200, json=LITELLM if body is None else body))


class PriceTests(unittest.TestCase):
    def setUp(self):
        prices.clear_cache()
        self.addCleanup(prices.clear_cache)

    def lookup(self, provider, base_url, model, rec, now=lambda: 1_700_000_000.0):
        return run(suggest_prices(provider, base_url, model, rec.transport, now))

    def test_exact_match_converts_per_token_to_per_million_without_float_noise(self):
        rec = litellm_recorder()
        result = self.lookup('compatible', 'https://x.example/v1', 'gpt-x', rec)
        self.assertEqual((result.found, result.input, result.output, result.cached_input), (True, 2.5, 10.0, 1.25))
        self.assertEqual((result.source, result.fetched, result.message), ('LiteLLM', '2023-11-14', ''))
        self.assertEqual((rec.requests[0].method, str(rec.requests[0].url)), ('GET', LITELLM_URL))

    def test_missing_cached_price_is_none_and_zero_prices_are_kept(self):
        rec = litellm_recorder()
        mini = self.lookup('openai', None, 'gpt-x-mini', rec)
        self.assertEqual((mini.found, mini.input, mini.output, mini.cached_input), (True, 0.15, 0.6, None))
        free = self.lookup('openai', None, 'free-model', rec)
        self.assertEqual((free.found, free.input, free.output), (True, 0.0, 0.0))

    def test_provider_prefixed_match(self):
        rec = litellm_recorder()
        self.assertEqual(self.lookup('anthropic', None, 'claude-x', rec).input, 3.0)
        self.assertEqual(self.lookup('openai', None, 'only-prefixed', rec).output, 2.0)
        gemini = self.lookup('compatible', 'https://generativelanguage.googleapis.com/v1beta/openai', 'models/gemini-x', rec)
        self.assertEqual((gemini.found, gemini.input, gemini.output), (True, 0.1, 0.4))
        other = self.lookup('compatible', 'https://x.example/v1', 'gemini-x', rec)
        self.assertEqual((other.found, other.input), (True, 0.9))

    def test_prefixed_entry_wins_over_a_different_providers_exact_entry(self):
        rec = litellm_recorder()
        gemini = self.lookup('compatible', 'https://generativelanguage.googleapis.com/v1beta/openai', 'gemini-x', rec)
        self.assertEqual((gemini.input, gemini.output), (0.1, 0.4))

    def test_exact_entry_from_a_mismatched_provider_is_not_found(self):
        rec = litellm_recorder()
        self.assertFalse(self.lookup('compatible', 'https://generativelanguage.googleapis.com/v1beta/openai', 'vertex-only', rec).found)
        self.assertFalse(self.lookup('anthropic', None, 'plain-openai', rec).found)
        self.assertTrue(self.lookup('openai', None, 'plain-openai', rec).found)
        self.assertTrue(self.lookup('compatible', 'https://x.example/v1', 'vertex-only', rec).found)

    def test_no_near_miss_matching(self):
        rec = litellm_recorder({'gpt-x': LITELLM['gpt-x']})
        for model in ('gpt-x-mini', 'GPT-X', 'gpt', 'gpt-x-2026-01-01'):
            result = self.lookup('openai', None, model, rec)
            self.assertEqual((result.found, result.message), (False, NOT_FOUND), model)
            self.assertEqual((result.input, result.output, result.cached_input), (None, None, None))

    def test_not_found_and_entries_without_usable_prices(self):
        rec = litellm_recorder()
        for model in ('unknown', 'no-prices', 'negative', 'sample_spec', ''):
            result = self.lookup('openai', None, model, rec)
            self.assertEqual((result.found, result.message), (False, NOT_FOUND), model)

    def test_openrouter_host_uses_its_own_list_with_string_prices(self):
        rec = Recorder(lambda r: httpx.Response(200, json=OPENROUTER))
        result = self.lookup('compatible', 'https://openrouter.ai/api/v1/', 'vendor/model', rec)
        self.assertEqual((result.found, result.input, result.output, result.cached_input, result.source), (True, 3.0, 15.0, 0.3, 'OpenRouter'))
        self.assertEqual(str(rec.requests[0].url), 'https://openrouter.ai/api/v1/models')
        self.assertIsNone(self.lookup('compatible', 'https://openrouter.ai/api/v1', 'vendor/no-cache', rec).cached_input)
        self.assertFalse(self.lookup('compatible', 'https://openrouter.ai/api/v1', 'vendor/mod', rec).found)
        self.assertEqual(self.lookup('compatible', 'https://openrouter.ai/api/v1', 'vendor/model-pro', rec).input, 100000.0)
        self.assertEqual(len(rec.requests), 1)

    def test_openrouter_url_is_fixed_whatever_base_url_is_typed(self):
        for base in ('https://u:p@openrouter.ai/x?y=1', 'http://openrouter.ai/other', 'https://openrouter.ai:8443/a/b/', 'https://OpenRouter.ai'):
            prices.clear_cache()
            rec = Recorder(lambda r: httpx.Response(200, json=OPENROUTER))
            result = self.lookup('compatible', base, 'vendor/model', rec)
            self.assertTrue(result.found, base)
            self.assertEqual([str(r.url) for r in rec.requests], ['https://openrouter.ai/api/v1/models'], base)
            self.assertNotIn('authorization', rec.requests[0].headers)

    def test_no_authorization_header_or_key_is_ever_sent(self):
        rec = litellm_recorder()
        self.lookup('openai', None, 'gpt-x', rec)
        orec = Recorder(lambda r: httpx.Response(200, json=OPENROUTER))
        self.lookup('compatible', 'https://openrouter.ai/api/v1', 'vendor/model', orec)
        for request in rec.requests + orec.requests:
            self.assertNotIn('authorization', request.headers)
            self.assertNotIn('x-api-key', request.headers)

    def test_redirect_is_not_followed(self):
        rec = Recorder(lambda r: httpx.Response(302, headers={'location': 'https://evil.example/x'}))
        result = self.lookup('openai', None, 'gpt-x', rec)
        self.assertEqual((result.found, result.message), (False, 'The price list could not be downloaded.'))
        self.assertEqual(len(rec.requests), 1)

    def test_http_error_and_bad_json_have_fixed_messages(self):
        self.assertEqual(self.lookup('openai', None, 'gpt-x', Recorder(lambda r: httpx.Response(500, text='secret detail'))).message,
                         'The price list could not be downloaded.')
        self.assertEqual(self.lookup('openai', None, 'gpt-x', Recorder(lambda r: httpx.Response(200, text='not json'))).message,
                         'The price list could not be read.')
        self.assertEqual(self.lookup('openai', None, 'gpt-x', Recorder(lambda r: httpx.Response(200, json=[1]))).message,
                         'The price list could not be read.')

        def boom(request):
            raise httpx.ConnectError('dns failure for host')
        result = self.lookup('openai', None, 'gpt-x', Recorder(boom))
        self.assertEqual((result.found, result.message), (False, 'The price list could not be downloaded.'))

    def test_oversized_body_is_refused(self):
        with unittest.mock.patch.object(prices, 'PRICE_MAX_BYTES', 1000):
            result = self.lookup('openai', None, 'gpt-x', Recorder(lambda r: httpx.Response(200, content=b'{"a": "' + b'x' * 2000 + b'"}')))
        self.assertEqual((result.found, result.message), (False, 'The price list is too large to read.'))
        self.assertGreater(PRICE_MAX_BYTES, 1_000_000)

    def test_deadline_gives_up(self):
        async def slow_transport():
            class Slow(httpx.AsyncBaseTransport):
                async def handle_async_request(self, request):
                    await asyncio.sleep(5)
            return await suggest_prices('openai', None, 'gpt-x', Slow())
        with unittest.mock.patch.object(prices, 'PRICE_DEADLINE_SECONDS', 0.05):
            result = run(slow_transport())
        self.assertEqual((result.found, result.message), (False, 'The price list did not respond in time.'))

    def test_cache_serves_two_lookups_within_24_hours_then_refetches(self):
        rec = litellm_recorder()
        clock = [1_700_000_000.0]
        self.lookup('openai', None, 'gpt-x', rec, lambda: clock[0])
        clock[0] += 3600
        again = self.lookup('openai', None, 'gpt-x-mini', rec, lambda: clock[0])
        self.assertEqual((len(rec.requests), again.found, again.fetched), (1, True, '2023-11-14'))
        clock[0] += 24 * 3600
        later = self.lookup('openai', None, 'gpt-x', rec, lambda: clock[0])
        self.assertEqual((len(rec.requests), later.fetched), (2, '2023-11-15'))

    def test_failures_are_not_cached(self):
        answers = [httpx.Response(500), httpx.Response(200, json=LITELLM)]
        rec = Recorder(lambda r: answers.pop(0))
        self.assertFalse(self.lookup('openai', None, 'gpt-x', rec).found)
        self.assertTrue(self.lookup('openai', None, 'gpt-x', rec).found)
        self.assertEqual(len(rec.requests), 2)

    def test_concurrent_lookups_download_once(self):
        async def slow(request):
            await asyncio.sleep(0.05)
            return httpx.Response(200, json=LITELLM)
        seen = []

        class Transport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                seen.append(request)
                return await slow(request)

        async def both():
            transport = Transport()
            return await asyncio.gather(*(suggest_prices('openai', None, 'gpt-x', transport) for _ in range(3)))
        results = run(both())
        self.assertEqual((len(seen), [r.found for r in results]), (1, [True] * 3))

    def test_each_source_is_cached_separately(self):
        rec = litellm_recorder()
        orec = Recorder(lambda r: httpx.Response(200, json=OPENROUTER))
        self.lookup('openai', None, 'gpt-x', rec)
        self.lookup('compatible', 'https://openrouter.ai/api/v1', 'vendor/model', orec)
        self.lookup('openai', None, 'gpt-x', rec)
        self.lookup('compatible', 'https://openrouter.ai/api/v1', 'vendor/model', orec)
        self.assertEqual((len(rec.requests), len(orec.requests)), (1, 1))


if __name__ == '__main__':
    unittest.main()
