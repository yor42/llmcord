import asyncio
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from llmcord_core.config import ModelProfile
from llmcord_core.discord_bot import SkitBot
from llmcord_core.engine import SceneContext
from llmcord_core.models import ModelGateway, TurnMessage
from llmcord_core.store import Store
from llmcord_core.usage import ModelUsage, capture_usage, collect_usage, rates, reply_footer
from test_core import settings
from test_discord_flow import FakeChannel, FakeWebhook


def gemini(**kwargs):
    return ModelProfile('compatible', 'gemini-3.8-flash', 16000, True,
                        base_url='https://generativelanguage.googleapis.com/v1beta/openai/', structured_outputs=True, **kwargs)


def counts(inputs=100, outputs=500):
    return SimpleNamespace(prompt_tokens=inputs, completion_tokens=outputs,
                           prompt_tokens_details=SimpleNamespace(cached_tokens=40),
                           completion_tokens_details=SimpleNamespace(reasoning_tokens=30))


class UsageTests(unittest.TestCase):
    def test_prices_support_published_schedule_free_tier_overrides_and_unknown_models(self):
        now = datetime(2026, 10, 7, tzinfo=timezone.utc).timestamp()
        next_year = datetime(2027, 1, 1, tzinfo=timezone.utc).timestamp()
        self.assertEqual(rates(gemini(), now)[0], (0.75, 3.75, 0.075))
        self.assertEqual(rates(gemini(), next_year)[0], (1.5, 7.5, 0.15))
        self.assertEqual(rates(gemini(billing_tier='free'), now)[0], (0, 0, 0))
        self.assertEqual(rates(gemini(input_cost_per_million=1, output_cost_per_million=2, cached_input_cost_per_million=0.1), now)[0], (1, 2, 0.1))
        unknown = replace(gemini(), model='unknown')
        self.assertEqual(rates(unknown, now)[0], (None, None, None))
        with capture_usage(1):
            usage = collect_usage('test', unknown, 'dialogue', counts())
        self.assertIsNone(usage.cost_usd)
        self.assertIn('cost: unavailable', reply_footer('unknown', usage))

    def test_reported_tokens_cached_pricing_and_reasoning_are_not_double_counted(self):
        now = datetime(2026, 10, 7, tzinfo=timezone.utc).timestamp()
        with patch('llmcord_core.usage.time.time', return_value=now), capture_usage(1) as records:
            usage = collect_usage('test', gemini(), 'dialogue', counts())
        self.assertEqual(records, [usage])
        self.assertEqual((usage.input_tokens, usage.output_tokens, usage.cached_tokens, usage.reasoning_tokens), (100, 500, 40, 30))
        self.assertAlmostEqual(usage.cost_usd, 0.001923)
        self.assertIn('incl. 30 thinking', reply_footer('gemini', usage))
        empty = collect_usage('test', gemini(), 'dialogue', None)
        self.assertIsNone(empty.input_tokens)
        self.assertIsNone(empty.output_tokens)
        self.assertIsNone(empty.cost_usd)

    def test_rolling_totals_are_scoped_persistent_and_include_internal_calls(self):
        store = Store()
        try:
            now = 200000.0
            for guild, model, created, role, inputs, outputs, cost in (
                (1, 'model', now, 'dialogue', 100, 50, 0.01),
                (1, 'model', now, 'director', 20, 10, 0.002),
                (1, 'model', now - 86401, 'memory', 999, 999, 1),
                (2, 'model', now, 'dialogue', 888, 888, 1),
                (1, 'different-model', now, 'dialogue', 777, 777, 1),
                (1, 'model', now, 'memory', None, None, None),
            ):
                store.record_model_usage(ModelUsage(guild, 'test', model, role, inputs, outputs, 0, 0, cost, 'configured', created))
            summary = store.model_usage_summary(1, 'test', 'model', now - 86400)
            self.assertEqual((summary['input_tokens'], summary['output_tokens'], summary['requests']), (120, 60, 3))
            self.assertAlmostEqual(summary['cost_usd'], 0.012)
            self.assertEqual((summary['unreported'], summary['unpriced']), (1, 1))
        finally:
            store.close()


class UsageAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_streams_capture_final_usage_and_anthropic_cache_tokens(self):
        for provider in ('openai', 'anthropic'):
            with self.subTest(provider=provider):
                profile = replace(gemini(), provider=provider, input_cost_per_million=1, output_cost_per_million=2, cached_input_cost_per_million=0.1)
                gateway = ModelGateway(replace(settings(), profiles={'test': profile}))
                if provider == 'openai':
                    async def stream():
                        yield SimpleNamespace(type='response.output_text.delta', delta='Hello')
                        yield SimpleNamespace(type='response.completed', response=SimpleNamespace(usage=SimpleNamespace(input_tokens=100, output_tokens=10, input_tokens_details=SimpleNamespace(cached_tokens=20))))
                    gateway.clients['test'] = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=stream())))
                else:
                    class Stream:
                        async def __aenter__(self):
                            return self
                        async def __aexit__(self, *args):
                            pass
                        @property
                        def text_stream(self):
                            async def text():
                                yield 'Hello'
                            return text()
                        async def get_final_message(self):
                            return SimpleNamespace(usage=SimpleNamespace(input_tokens=100, output_tokens=10, cache_read_input_tokens=30, cache_creation_input_tokens=20))
                    gateway.clients['test'] = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: Stream()))
                with capture_usage(1) as records:
                    self.assertEqual(''.join([part async for part in gateway.stream_text('dialogue', '', [TurnMessage('user', 'Hi')])]), 'Hello')
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0].output_tokens, 10)
                self.assertEqual(records[0].input_tokens, 150 if provider == 'anthropic' else 100)
                self.assertEqual(records[0].cached_tokens, 30 if provider == 'anthropic' else 20)

    async def test_usage_only_trailer_is_captured_and_concurrent_guilds_do_not_mix(self):
        gateway = ModelGateway(replace(settings(), profiles={'test': gemini()}))
        async def create(**kwargs):
            if kwargs.get('stream'):
                async def stream():
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='Hello'))], usage=None)
                    yield SimpleNamespace(choices=[], usage=counts())
                return stream()
            await asyncio.sleep(0)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Hello'))], usage=counts())
        api = AsyncMock(side_effect=create)
        gateway.clients['test'] = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=api)))
        async def run(guild):
            with capture_usage(guild) as records:
                if guild == 1:
                    self.assertEqual(''.join([part async for part in gateway.stream_text('dialogue', '', [TurnMessage('user', 'Hi')])]), 'Hello')
                else:
                    await gateway.text('memory', '', [TurnMessage('user', 'Hi')])
            return records
        first, second = await asyncio.gather(run(1), run(2))
        self.assertEqual([record.guild_id for record in first], [1])
        self.assertEqual([record.guild_id for record in second], [2])
        self.assertEqual(first[0].output_tokens, 500)
        stream_call = next(call.kwargs for call in api.call_args_list if call.kwargs.get('stream'))
        self.assertEqual(stream_call['stream_options'], {'include_usage': True})

    async def test_usage_footer_keeps_history_clean_and_is_one_call_for_all_chunks(self):
        bot = SkitBot(replace(settings(), profiles={'test': gemini()}))
        store = bot.store
        world = store.create_space(1, 'World', 'world')
        store.bind_channel(1, 100, world)
        character = store.create_character(1, world, 'Alice')
        store.set_cast(100, None, [character])
        hook = FakeWebhook(2000)
        async def webhook(channel, row):
            return hook
        bot._webhook = webhook
        async def create(**kwargs):
            if kwargs.get('stream'):
                async def stream():
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='<emotion>neutral</emotion>\n' + 'word ' * 600))], usage=None)
                    yield SimpleNamespace(choices=[], usage=counts())
                return stream()
            schema = kwargs.get('response_format', {}).get('json_schema', {}).get('schema', {})
            text = '{"speakers":[' + str(character) + ']}' if 'speakers' in schema.get('properties', {}) else '{"shared_facts":[],"personal_facts":[],"encounter_facts":[]}' if schema else 'Summary'
            self.assertFalse(any('-# ' in str(message['content']) for message in kwargs['messages']))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason='stop')], usage=counts(10, 5))
        bot.models.clients['test'] = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=create))))
        channel = FakeChannel(100)
        now = datetime(2026, 10, 7, tzinfo=timezone.utc).timestamp()
        try:
            with patch('llmcord_core.usage.time.time', return_value=now):
                await bot.run_scene(SceneContext(1, 100, None, world, 9, 1000, 'Hello', None, [], []), channel)
            self.assertFalse(channel.errors)
            self.assertGreater(len(hook.posts), 1)
            for posted in hook.posts:
                self.assertLessEqual(len(posted.content), 2000)
                self.assertIn('Reply input: 100', posted.content)
                self.assertIn('Output: 500', posted.content)
                self.assertIn('$0.001923', posted.content)
                self.assertNotIn('-# ', store.node(posted.id)['content'])
                self.assertEqual(store.trace(posted.id)['usage']['output_tokens'], 500)
            self.assertEqual(len(store.all("SELECT * FROM model_usage WHERE role='dialogue'")), 1)
            self.assertTrue(any('Streaming' in edit and 'gemini-3.8-flash' in edit and '24h tracked:' in edit for message in channel.messages for edit in message.edits))
            self.assertGreater(store.model_usage_summary(1, 'test', 'gemini-3.8-flash', now - 86400)['requests'], 1)
        finally:
            store.close()
