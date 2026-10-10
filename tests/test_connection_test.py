"""FEAT-11 (D24 step 5): ``test_profile_connection`` probes an unsaved profile form with one tiny request.

Seam: the helper in ``llmcord_core.backend`` (imported as ``probe`` so unittest does not collect it), with
``llmcord_core.models.AsyncOpenAI`` / ``AsyncAnthropic`` replaced by recording fakes. No network, no Store.
"""
from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from llmcord_core.backend import test_profile_connection as probe
from llmcord_core.config import key_hosts_from_env
from llmcord_core.models import ModelGateway
from test_backend_resolver import FakeAnthropic, FakeOpenAI

SECRET = 'sk-probe-secret-0123456789'
HOSTS = key_hosts_from_env({})
OPENAI = {'provider': 'openai', 'model': 'gpt-x', 'context_tokens': 16000, 'api_key_env': 'OPENAI_API_KEY'}
ENV = {'OPENAI_API_KEY': SECRET}


class ProbeCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeOpenAI.created, FakeOpenAI.models, FakeAnthropic.created, FakeAnthropic.models = [], [], [], []
        for target, fake in (('AsyncOpenAI', FakeOpenAI), ('AsyncAnthropic', FakeAnthropic)):
            patcher = patch(f'llmcord_core.models.{target}', fake, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def fail_requests(self, message):
        original = FakeOpenAI.__init__

        def init(client, **kwargs):
            original(client, **kwargs)

            async def boom(**request):
                raise RuntimeError(message)
            client.responses.create = boom
            client.chat.completions.create = boom
        patcher = patch.object(FakeOpenAI, '__init__', init)
        patcher.start()
        self.addCleanup(patcher.stop)


class ProbeTests(ProbeCase):
    async def test_success_reports_model_and_latency(self):
        """FEAT-11: a working profile gives ok=True, an integer latency and 'Connected. <model> replied in N ms.'."""
        result = await probe('cloud', OPENAI, HOSTS, ENV)
        self.assertTrue(result.ok)
        self.assertIsInstance(result.latency_ms, int)
        self.assertGreaterEqual(result.latency_ms, 0)
        self.assertEqual(result.message, f'Connected. gpt-x replied in {result.latency_ms} ms.')
        self.assertEqual(FakeOpenAI.models, ['gpt-x'])
        self.assertEqual(FakeOpenAI.created[0].kwargs['api_key'], SECRET)

    async def test_provider_error_echoing_the_key_is_redacted(self):
        """FEAT-11: a provider error containing the key value is returned redacted, capped at 300 characters."""
        self.fail_requests(f'Incorrect API key provided: {SECRET}. ' + 'x' * 600)
        result = await probe('cloud', OPENAI, HOSTS, ENV)
        self.assertFalse(result.ok)
        self.assertIsNone(result.latency_ms)
        self.assertNotIn(SECRET, result.message)
        self.assertIn('Incorrect API key', result.message)
        self.assertLessEqual(len(result.message), 300)

    async def test_pin_refusal_builds_no_client(self):
        """FEAT-11: a base_url not pinned for the key is refused with the pin message and no client or request."""
        result = await probe('cloud', {**OPENAI, 'base_url': 'https://evil.example/v1'}, HOSTS, ENV)
        self.assertFalse(result.ok)
        self.assertIn('evil.example', result.message)
        self.assertNotIn(SECRET, result.message)
        self.assertEqual((FakeOpenAI.created, FakeOpenAI.models), ([], []))

    async def test_missing_variable_is_named_for_the_dashboard_environment(self):
        """FEAT-11: a key variable unset or empty in the web environment is named, with no request sent."""
        for environ in ({}, {'OPENAI_API_KEY': ''}):
            with self.subTest(environ=environ):
                result = await probe('cloud', OPENAI, HOSTS, environ)
                self.assertFalse(result.ok)
                self.assertIn('OPENAI_API_KEY', result.message)
                self.assertIn("dashboard's environment", result.message)
        self.assertEqual(FakeOpenAI.created, [])

    async def test_validation_error_builds_no_client(self):
        """FEAT-11: a form that fails profile validation (bad base_url) reports the error and sends nothing."""
        result = await probe('cloud', {'provider': 'compatible', 'model': 'm', 'context_tokens': 1000, 'base_url': 'ftp://nope'}, HOSTS, ENV)
        self.assertFalse(result.ok)
        self.assertTrue(result.message)
        self.assertEqual((FakeOpenAI.created, FakeOpenAI.models), ([], []))

    async def test_client_timeout_is_capped_and_retries_disabled(self):
        """FEAT-11: the one-off client uses min(timeout, 20) seconds and max_retries 0."""
        await probe('cloud', {**OPENAI, 'timeout_seconds': 120, 'max_retries': 5}, HOSTS, ENV)
        await probe('cloud', {**OPENAI, 'timeout_seconds': 5}, HOSTS, ENV)
        first, second = (client.kwargs for client in FakeOpenAI.created)
        self.assertEqual((first['timeout'], first['max_retries']), (20, 0))
        self.assertEqual((second['timeout'], second['max_retries']), (5, 0))

    async def test_anthropic_profile_uses_the_pinned_origin(self):
        """FEAT-11: an anthropic form probes through the anthropic client with the explicit origin."""
        data = {'provider': 'anthropic', 'model': 'claude-x', 'context_tokens': 16000, 'api_key_env': 'ANTHROPIC_API_KEY'}
        result = await probe('claude', data, HOSTS, {'ANTHROPIC_API_KEY': SECRET})
        self.assertTrue(result.ok, result.message)
        self.assertEqual(FakeAnthropic.created[0].kwargs['base_url'], 'https://api.anthropic.com')

    async def test_client_is_closed_on_success_and_failure(self):
        """FEAT-11: the one-off gateway closes its client after a success and after a failed request."""
        await probe('cloud', OPENAI, HOSTS, ENV)
        self.fail_requests('provider down')
        await probe('cloud', OPENAI, HOSTS, ENV)
        self.assertEqual([client.closed for client in FakeOpenAI.created], [True, True])

    async def test_overall_limit_maps_to_timeout_message_and_closes_client(self):
        """FEAT-11: a call that outlives the overall probe limit gives the timeout failure and still closes the client."""
        original = FakeOpenAI.__init__

        def init(client, **kwargs):
            original(client, **kwargs)

            async def slow(**request):
                await asyncio.sleep(5)
            client.responses.create = slow
            client.chat.completions.create = slow
        with patch.object(FakeOpenAI, '__init__', init), patch('llmcord_core.backend.PROBE_TIMEOUT_SECONDS', 0.05):
            result = await probe('cloud', OPENAI, HOSTS, ENV)
        self.assertFalse(result.ok)
        self.assertIsNone(result.latency_ms)
        self.assertIn('did not respond in time', result.message)
        self.assertEqual([client.closed for client in FakeOpenAI.created], [True])

    async def test_provider_timeout_exception_maps_to_timeout_message_and_closes_client(self):
        """MNT-32: the SDK's own timeout error (not the overall limit) gives the same failure line and still closes the client."""
        import httpx
        original = FakeOpenAI.__init__

        def init(client, **kwargs):
            original(client, **kwargs)

            async def boom(**request):
                raise httpx.ReadTimeout('timed out')
            client.responses.create = boom
            client.chat.completions.create = boom
        with patch.object(FakeOpenAI, '__init__', init):
            result = await probe('cloud', OPENAI, HOSTS, ENV)
        self.assertFalse(result.ok)
        self.assertIsNone(result.latency_ms)
        self.assertIn('did not respond in time', result.message)
        self.assertEqual([client.closed for client in FakeOpenAI.created], [True])

    async def test_gateway_names_its_environment_when_the_variable_is_missing(self):
        """MNT-32: the gateway's missing-variable error names the process it runs in; the default stays 'the bot'."""
        from llmcord_core.backend import _ProbeSource
        from llmcord_core.config import Settings, profile_from_mapping
        from llmcord_core.errors import ModelConfigError
        from llmcord_core.models import TurnMessage
        from pathlib import Path
        profile = profile_from_mapping('cloud', OPENAI, source='dashboard')
        settings = Settings('', None, Path(':memory:'), 0, {'cloud': profile}, 'cloud', 'cloud', 'cloud', {})
        for kwargs, where in (({}, "the bot's environment"), ({'environment': 'the dashboard'}, "the dashboard's environment")):
            with self.subTest(where=where):
                gateway = ModelGateway(_ProbeSource(settings, HOSTS, {}), **kwargs)
                with self.assertRaises(ModelConfigError) as caught:
                    await gateway.text('dialogue', '', [TurnMessage('user', 'hi')])
                self.assertIn(f'OPENAI_API_KEY is not set in {where}', str(caught.exception))
                await gateway.close()

    async def test_no_usage_budget_or_log_hooks(self):
        """FEAT-11: the probe gateway has no usage sink, budget gate or turn log sink."""
        seen, original = [], ModelGateway.__init__

        def init(gateway, *args, **kwargs):
            seen.append((args, kwargs))
            original(gateway, *args, **kwargs)
        with patch.object(ModelGateway, '__init__', init):
            await probe('cloud', OPENAI, HOSTS, ENV)
        self.assertEqual(len(seen), 1)
        args, kwargs = seen[0]
        self.assertEqual(len(args), 1)  # settings only (the patched init excludes self)
        for hook in ('usage_sink', 'budget_gate', 'log_sink'):
            self.assertIsNone(kwargs.get(hook))

    async def test_unexpected_exception_becomes_a_generic_failure(self):
        """FEAT-11: the helper never raises; an internal error gives ok=False and a generic message without the detail."""
        with patch('llmcord_core.backend.check_key_pin', side_effect=RuntimeError('internal-detail-xyz'), create=True):
            result = await probe('cloud', OPENAI, HOSTS, ENV)
        self.assertFalse(result.ok)
        self.assertTrue(result.message)
        self.assertNotIn('internal-detail-xyz', result.message)


if __name__ == '__main__':
    unittest.main()
