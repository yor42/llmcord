"""FEAT-10 (D24 step 3): the bot resolves dashboard model profiles once per turn and checks key hosts at call time.

Seams: ``BackendResolver(store, base_settings)`` (``snapshot`` / ``pin`` / ``current`` / ``key_hosts``), ``ModelGateway``
and ``Engine`` built over a resolver, and ``SkitBot.run_scene`` end to end (observed through ``bot.settings``). The SDK
client classes are replaced by recording fakes (``llmcord_core.models.AsyncOpenAI`` / ``AsyncAnthropic``), the store is
in-memory and the environment is patched; nothing touches the network.
"""
from __future__ import annotations

import asyncio
import dataclasses
import os
import sqlite3
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from helpers import FakeTextChannel, MemoryModels, drain_memory_tasks, make_settings
from llmcord_core.backend import BackendResolver
from llmcord_core.discord_bot import SkitBot
from llmcord_core.engine import Engine, SceneContext
from llmcord_core.errors import ModelConfigError
from llmcord_core.models import ModelGateway, TurnMessage
from llmcord_core.store import Store

MSG = [TurnMessage('user', 'hi')]
SECRET = 'sk-dashboard-secret-0123456789'
CHANNEL = 100


def dash_data(model='m1', **extra):
    return {'provider': 'compatible', 'model': model, 'context_tokens': 16000, 'base_url': 'http://localhost/v1', **extra}


def save_dash(store, name, data):
    """Save through the validated store path (insert, or update at the current revision)."""
    row = next((r for r in store.model_profile_rows() if r['name'] == name), None)
    store.save_model_profile(name, data, row['revision'] if row else None)


def write_raw(store, name, data):
    """Write a profile row directly, bypassing save-time validation (simulates a changed environment or a bad row)."""
    import json
    with store.write_admin():
        store.db.execute('INSERT INTO model_profiles(name, data_json, revision, updated_at) VALUES(?,?,?,?) '
                         'ON CONFLICT(name) DO UPDATE SET data_json=excluded.data_json',
                         (name, json.dumps(data), 1, 0.0))
        store.db.execute('UPDATE model_roles SET version=version+1 WHERE id=1')


class FakeSdk:
    """Records constructed clients and every request; stands in for AsyncOpenAI and AsyncAnthropic."""
    created: list = []
    models: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        type(self).created.append(self)
        sdk = type(self)

        async def chat(**request):
            sdk.models.append(request['model'])
            if request.get('stream'):
                async def chunks():
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='ok'))], usage=None)
                return chunks()
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok'), finish_reason='stop')], usage=None)

        async def responses(**request):
            sdk.models.append(request['model'])
            return SimpleNamespace(output_text='ok', usage=None)

        async def messages(**request):
            sdk.models.append(request['model'])
            return SimpleNamespace(content=[SimpleNamespace(type='text', text='ok')], usage=None)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=chat))
        self.responses = SimpleNamespace(create=responses)
        self.messages = SimpleNamespace(create=messages)

        self.closed = False

    async def close(self):
        self.closed = True


class FakeOpenAI(FakeSdk):
    created, models = [], []


class FakeAnthropic(FakeSdk):
    created, models = [], []


async def _empty():
    return
    yield


class BackendCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)
        self.base = make_settings()
        FakeOpenAI.created, FakeOpenAI.models, FakeAnthropic.created, FakeAnthropic.models = [], [], [], []
        env = patch.dict(os.environ, {})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop('LLMCORD_KEY_HOSTS', None)
        for name in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY'):
            os.environ.pop(name, None)
        for target, fake in (('AsyncOpenAI', FakeOpenAI), ('AsyncAnthropic', FakeAnthropic)):
            patcher = patch(f'llmcord_core.models.{target}', fake, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def resolver(self, base=None):
        return BackendResolver(self.store, base or self.base)

    def with_profile(self, **profile):
        """Base settings whose single config profile 'test' is replaced by one built from ``profile`` fields."""
        replaced = dataclasses.replace(self.base.profiles['test'], **profile)
        return dataclasses.replace(self.base, profiles={'test': replaced})


class ResolverTests(BackendCase):
    def test_unchanged_version_returns_the_cached_snapshot_without_rereading_rows(self):
        """FEAT-10: with an unchanged backend version, snapshot() reuses the cached Settings (one cheap version read)."""
        resolver = self.resolver()
        first = resolver.snapshot()
        with patch.object(self.store, 'model_profile_rows', wraps=self.store.model_profile_rows) as rows, \
                patch.object(self.store, 'model_roles', wraps=self.store.model_roles) as roles:
            self.assertIs(resolver.snapshot(), first)
            self.assertIs(resolver.snapshot(), first)
        self.assertEqual((rows.call_count, roles.call_count), (0, 0))

    def test_changed_version_reloads_and_old_snapshot_is_untouched(self):
        """FEAT-10: a saved profile bumps the version; the next snapshot() carries it, the earlier object does not change."""
        resolver = self.resolver()
        before = resolver.snapshot()
        save_dash(self.store, 'test', dash_data('m2'))
        after = resolver.snapshot()
        self.assertEqual(after.profile('dialogue').model, 'm2')
        self.assertEqual(after.profile('dialogue').source, 'dashboard')
        self.assertEqual(before.profile('dialogue').model, 'test')
        self.assertEqual(before.profiles['test'].source, 'config')

    def test_role_assignment_change_is_picked_up(self):
        """FEAT-10: a role assignment saved in the database switches that role on the next snapshot."""
        resolver = self.resolver()
        save_dash(self.store, 'fast', dash_data('quick'))
        roles = self.store.model_roles()
        self.store.save_model_roles({'dialogue': 'fast', 'director': None, 'memory': None}, roles['revision'], {'test'})
        settings = resolver.snapshot()
        self.assertEqual((settings.dialogue, settings.director, settings.memory), ('fast', 'test', 'test'))

    def test_invalid_row_is_logged_once_and_config_profile_applies(self):
        """FEAT-10: an invalid saved row falls back to the config profile and its problem is logged once at WARNING."""
        write_raw(self.store, 'test', {'provider': 'compatible', 'model': 'no-tokens-or-base'})
        resolver = self.resolver()
        with self.assertLogs(level='WARNING') as logs:
            first = resolver.snapshot()
            resolver.snapshot()
            resolver.snapshot()
        self.assertEqual(first.profile('dialogue'), self.base.profile('dialogue'))
        problems = [line for line in logs.output if 'test' in line]
        self.assertEqual(len(problems), 1, logs.output)

    def test_version_change_during_reload_is_seen_by_the_next_snapshot(self):
        """FEAT-10: the cache is keyed by the version read first, so an edit saved while reloading is not lost."""
        resolver = self.resolver()
        original, saved = self.store.model_profile_rows, []

        def rows_then_edit():
            result = original()
            if not saved:
                saved.append(True)
                save_dash(self.store, 'extra', dash_data('late'))
            return result

        with patch.object(self.store, 'model_profile_rows', rows_then_edit):
            first = resolver.snapshot()
        self.assertNotIn('extra', first.profiles)
        self.assertEqual(resolver.snapshot().profiles['extra'].model, 'late')

    def test_database_error_keeps_last_good_snapshot(self):
        """FEAT-10: a DB error logs the exception type only and returns the last good snapshot."""
        resolver = self.resolver()
        save_dash(self.store, 'test', dash_data('m2'))
        good = resolver.snapshot()
        with patch.object(self.store, 'model_backend_version', side_effect=sqlite3.OperationalError('disk secret detail')), \
                self.assertLogs(level='WARNING') as logs:
            self.assertEqual(resolver.snapshot().profile('dialogue').model, 'm2')
        self.assertEqual(good.profile('dialogue').model, 'm2')
        text = '\n'.join(logs.output)
        self.assertIn('OperationalError', text)
        self.assertNotIn('disk secret detail', text)

    def test_database_error_before_any_snapshot_returns_base(self):
        """FEAT-10: a DB error with no earlier snapshot serves the startup (config.yaml) settings."""
        resolver = self.resolver()
        with patch.object(self.store, 'model_backend_version', side_effect=sqlite3.OperationalError('x')), self.assertLogs(level='WARNING'):
            settings = resolver.snapshot()
        self.assertEqual(settings.profiles, self.base.profiles)
        self.assertEqual(settings.dialogue, 'test')

    def test_key_hosts_has_builtin_pins_and_reads_environment_once(self):
        """FEAT-10: key_hosts is parsed from the environment when the resolver is built."""
        with patch.dict(os.environ, {'LLMCORD_KEY_HOSTS': 'MY_API_KEY=llm.example'}):
            resolver = self.resolver()
        self.assertIn(('https', 'api.openai.com', 443), resolver.key_hosts['OPENAI_API_KEY'])
        self.assertIn(('https', 'llm.example', 443), resolver.key_hosts['MY_API_KEY'])

    def test_malformed_key_hosts_stops_startup_naming_the_variable(self):
        """FEAT-10: a malformed LLMCORD_KEY_HOSTS fails at construction and names the variable."""
        with patch.dict(os.environ, {'LLMCORD_KEY_HOSTS': 'not valid'}), self.assertRaises(ValueError) as caught:
            self.resolver()
        self.assertIn('LLMCORD_KEY_HOSTS', str(caught.exception))

    def test_current_is_pinned_inside_and_live_outside(self):
        """FEAT-10: current() is the pinned snapshot inside pin() (nested pins keep the outer one) and live outside."""
        resolver = self.resolver()
        save_dash(self.store, 'test', dash_data('m1'))
        with resolver.pin():
            save_dash(self.store, 'test', dash_data('m2'))
            self.assertEqual(resolver.current().profile('dialogue').model, 'm1')
            with resolver.pin():
                self.assertEqual(resolver.current().profile('dialogue').model, 'm1')
            self.assertEqual(resolver.current().profile('dialogue').model, 'm1')
        self.assertEqual(resolver.current().profile('dialogue').model, 'm2')

    def test_settings_source_wrappers_accept_settings_or_resolver(self):
        """FEAT-10: ModelGateway and Engine keep working with a plain Settings and follow a resolver's current Settings."""
        self.assertEqual(ModelGateway(self.base).settings.dialogue, 'test')
        self.assertEqual(Engine(self.store, None, self.base).settings.profile('dialogue').model, 'test')
        resolver = self.resolver()
        gateway = ModelGateway(resolver)
        engine = Engine(self.store, gateway, resolver)
        save_dash(self.store, 'test', dash_data('m2'))
        self.assertEqual(gateway.settings.profile('dialogue').model, 'm2')
        self.assertEqual(engine.settings.profile('dialogue').model, 'm2')
        with resolver.pin():
            save_dash(self.store, 'test', dash_data('m3'))
            self.assertEqual(engine.settings.profile('dialogue').model, 'm2')


class SnapshotTests(BackendCase):
    async def test_profile_saved_mid_turn_does_not_affect_the_second_call(self):
        """FEAT-10: inside pin(), a profile saved between two gateway calls is not used until the next turn."""
        save_dash(self.store, 'test', dash_data('m1'))
        resolver = self.resolver()
        gateway = ModelGateway(resolver)
        with resolver.pin():
            await gateway.text('dialogue', '', MSG)
            save_dash(self.store, 'test', dash_data('m2'))
            await gateway.text('dialogue', '', MSG)
        self.assertEqual(FakeOpenAI.models, ['m1', 'm1'])
        with resolver.pin():
            await gateway.text('dialogue', '', MSG)
        self.assertEqual(FakeOpenAI.models, ['m1', 'm1', 'm2'])

    async def test_task_created_in_a_pinned_block_keeps_the_turn_snapshot(self):
        """FEAT-10: a task created inside pin() inherits the snapshot even if it runs after the block exits."""
        save_dash(self.store, 'test', dash_data('m1'))
        resolver = self.resolver()
        gateway = ModelGateway(resolver)
        release = asyncio.Event()

        async def work():
            await release.wait()
            await gateway.text('memory', '', MSG)
            return resolver.current().profile('memory').model

        with resolver.pin():
            task = asyncio.create_task(work())
        save_dash(self.store, 'test', dash_data('m2'))
        release.set()
        self.assertEqual(await asyncio.wait_for(task, 1), 'm1')
        self.assertEqual(FakeOpenAI.models, ['m1'])


class ClientCacheTests(BackendCase):
    async def test_client_rebuilt_only_when_fingerprint_changes(self):
        """FEAT-10: the client cache is keyed by name plus (provider, base_url, key env, timeout, retries, source)."""
        save_dash(self.store, 'test', dash_data('m1'))
        gateway = ModelGateway(self.resolver())
        await gateway.text('dialogue', '', MSG)
        await gateway.text('dialogue', '', MSG)
        self.assertEqual(len(FakeOpenAI.created), 1)
        save_dash(self.store, 'test', dash_data('m2'))
        await gateway.text('dialogue', '', MSG)
        self.assertEqual(len(FakeOpenAI.created), 1, 'a model change must reuse the client')
        for extra, expected in (({'timeout_seconds': 30}, 2), ({'timeout_seconds': 30, 'max_retries': 3}, 3)):
            save_dash(self.store, 'test', dash_data('m2', **extra))
            await gateway.text('dialogue', '', MSG)
            self.assertEqual(len(FakeOpenAI.created), expected)
        self.assertEqual(FakeOpenAI.created[-1].kwargs['timeout'], 30)
        self.assertEqual(FakeOpenAI.created[-1].kwargs['max_retries'], 3)
        save_dash(self.store, 'test', dash_data('m2', timeout_seconds=30, max_retries=3, base_url='http://other.local/v1'))
        await gateway.text('dialogue', '', MSG)
        self.assertEqual(FakeOpenAI.created[-1].kwargs['base_url'], 'http://other.local/v1')
        self.assertEqual(len(FakeOpenAI.created), 4)

    async def test_replaced_client_is_closed_by_gateway_close(self):
        """FEAT-10: a client replaced after a fingerprint change is still closed by gateway.close()."""
        save_dash(self.store, 'test', dash_data('m1'))
        gateway = ModelGateway(self.resolver())
        await gateway.text('dialogue', '', MSG)
        save_dash(self.store, 'test', dash_data('m1', timeout_seconds=30))
        await gateway.text('dialogue', '', MSG)
        self.assertEqual(len(FakeOpenAI.created), 2)
        await gateway.close()
        self.assertEqual([client.closed for client in FakeOpenAI.created], [True, True])

    async def test_plain_settings_still_cache_one_client_per_profile(self):
        """FEAT-10: a gateway built on a plain Settings keeps one client per profile."""
        gateway = ModelGateway(self.base)
        await gateway.text('dialogue', '', MSG)
        await gateway.text('memory', '', MSG)
        self.assertEqual(len(FakeOpenAI.created), 1)


class ClientLifecycleTests(BackendCase):
    """MNT-31: retired clients are closed after their last in-flight use; close() closes everything."""

    def setUp(self):
        super().setUp()
        self.gate = asyncio.Event()
        self.started = asyncio.Event()
        patcher = patch('llmcord_core.models.AsyncOpenAI', self.gated_sdk, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def gated_sdk(self, **kwargs):
        client = FakeOpenAI(**kwargs)
        create = client.chat.completions.create

        async def gated(**request):
            self.started.set()
            await self.gate.wait()
            if getattr(client, 'fail', False):
                raise RuntimeError('provider failure')
            return await create(**request)
        client.chat.completions.create = gated
        return client

    def variant(self, timeout):
        return self.with_profile(timeout_seconds=timeout)

    async def settle(self):
        for _ in range(5):
            await asyncio.sleep(0)

    async def test_retired_client_is_closed_after_its_last_in_flight_call(self):
        """MNT-31 (fixed): a replaced client stays open while a call uses it, then is closed and dropped."""
        gateway = ModelGateway(self.variant(10))
        call = asyncio.create_task(gateway.text('dialogue', '', MSG, settings=self.variant(10)))
        await self.started.wait()
        newer = asyncio.create_task(gateway.text('dialogue', '', MSG, settings=self.variant(20)))
        await self.settle()
        old = FakeOpenAI.created[0]
        self.assertEqual(len(FakeOpenAI.created), 2)
        self.assertFalse(old.closed, 'a client with a call in flight must stay open')
        self.gate.set()
        await asyncio.gather(call, newer)
        await self.settle()
        self.assertTrue(old.closed)
        self.assertFalse(FakeOpenAI.created[1].closed)
        self.assertEqual(gateway.retired, {})
        await gateway.close()

    async def test_idle_replaced_client_is_closed_at_once(self):
        """MNT-31 (fixed): a replaced client with no call in flight is closed without waiting for shutdown."""
        gateway = ModelGateway(self.variant(10))
        self.gate.set()
        await gateway.text('dialogue', '', MSG, settings=self.variant(10))
        await gateway.text('dialogue', '', MSG, settings=self.variant(20))
        await self.settle()
        self.assertEqual([c.closed for c in FakeOpenAI.created], [True, False])
        self.assertEqual(gateway.retired, {})

    async def test_retired_client_is_closed_when_a_stream_is_closed_early(self):
        """MNT-31 (fixed): a stream closed before completion still releases its client."""
        gateway = ModelGateway(self.variant(10))
        self.gate.set()
        stream = gateway.stream_text('dialogue', '', MSG, settings=self.variant(10))
        self.assertEqual(await anext(stream), 'ok')
        await gateway.text('dialogue', '', MSG, settings=self.variant(20))
        await self.settle()
        old = FakeOpenAI.created[0]
        self.assertFalse(old.closed, 'an open stream keeps its client alive')
        await stream.aclose()
        await self.settle()
        self.assertTrue(old.closed)
        self.assertEqual(gateway.retired, {})

    async def test_alternating_fingerprints_do_not_keep_building_clients(self):
        """MNT-31 (fixed, was suspicion c): an older pinned snapshot and the current one share stable clients instead of retiring each other."""
        gateway = ModelGateway(self.variant(10))
        hold = asyncio.create_task(gateway.text('dialogue', '', MSG, settings=self.variant(10)))
        await self.started.wait()
        for _ in range(3):
            for timeout in (20, 10):
                call = asyncio.create_task(gateway.text('dialogue', '', MSG, settings=self.variant(timeout)))
                await self.settle()
                call.cancel()
                await asyncio.gather(call, return_exceptions=True)
        self.assertEqual(len(FakeOpenAI.created), 2)
        self.gate.set()
        await hold
        await gateway.close()

    async def test_failing_call_releases_and_closes_its_retired_client(self):
        """MNT-31 (fixed): a provider call that raises still releases its client, so a retired one is closed."""
        gateway = ModelGateway(self.variant(10))
        call = asyncio.create_task(gateway.text('dialogue', '', MSG, settings=self.variant(10)))
        await self.started.wait()
        old = FakeOpenAI.created[0]
        old.fail = True
        newer = asyncio.create_task(gateway.text('dialogue', '', MSG, settings=self.variant(20)))
        await self.settle()
        self.assertFalse(old.closed)
        self.gate.set()
        with self.assertRaises(RuntimeError):
            await call
        await newer
        await self.settle()
        self.assertEqual(gateway._inflight, {})
        self.assertTrue(old.closed)

    async def test_structured_text_fallback_counts_nested_use_then_closes(self):
        """MNT-31 (fixed): the compatible structured->text fallback holds its client twice, then releases to zero and closes it."""
        gateway = ModelGateway(self.variant(10))
        call = asyncio.create_task(gateway.structured('dialogue', '', MSG, 'x', {'type': 'string'}, settings=self.variant(10)))
        await self.started.wait()
        old = FakeOpenAI.created[0]
        self.assertEqual(gateway._inflight, {id(old): 2})
        newer = asyncio.create_task(gateway.text('dialogue', '', MSG, settings=self.variant(20)))
        await self.settle()
        self.assertFalse(old.closed)
        self.gate.set()
        with self.assertRaises(ValueError):
            await call
        await newer
        await self.settle()
        self.assertEqual(gateway._inflight, {})
        self.assertTrue(old.closed)

    async def test_anthropic_stream_releases_its_client(self):
        """MNT-31 (fixed): the anthropic messages.stream path releases its client when the stream ends."""
        profile = dataclasses.replace(self.base.profiles['test'], provider='anthropic', base_url=None)
        settings = dataclasses.replace(self.base, profiles={'test': profile})

        class Stream:
            text_stream = _empty()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get_final_message(self):
                return SimpleNamespace(usage=None)
        gateway = ModelGateway(settings)
        client = gateway._client('test')
        client.messages.stream = lambda **kw: Stream()
        self.assertEqual([d async for d in gateway.stream_text('dialogue', '', MSG, settings=settings)], [])
        self.assertEqual(gateway._inflight, {})
        await gateway.close()

    async def test_close_awaits_a_pending_scheduled_close(self):
        """MNT-31 (fixed): close() waits for a replaced client's scheduled close and still closes the rest."""
        gateway = ModelGateway(self.variant(10))
        self.gate.set()
        await gateway.text('dialogue', '', MSG, settings=self.variant(10))
        release = asyncio.Event()
        first = FakeOpenAI.created[0]
        real = first.close

        async def slow_close():
            await release.wait()
            await real()
        first.close = slow_close
        await gateway.text('dialogue', '', MSG, settings=self.variant(20))
        await self.settle()
        self.assertFalse(first.closed)
        closing = asyncio.create_task(gateway.close())
        await self.settle()
        self.assertFalse(closing.done())
        release.set()
        await closing
        self.assertTrue(first.closed)
        self.assertTrue(FakeOpenAI.created[1].closed)

    async def test_close_continues_past_a_failing_client_and_logs_type_only(self):
        """MNT-31 (fixed): close() closes every client, logs failures by exception type only, and does not stop at the first."""
        gateway = ModelGateway(self.variant(10))
        self.gate.set()
        await gateway.text('dialogue', '', MSG, settings=self.variant(10))
        first = FakeOpenAI.created[0]
        gateway.retired[('test', 'other')] = second = FakeOpenAI()

        async def broken():
            raise RuntimeError('provider said: private text')
        first.close = broken
        with self.assertLogs(level='WARNING') as logs:
            await gateway.close()
        self.assertTrue(second.closed)
        output = '\n'.join(logs.output)
        self.assertIn('RuntimeError', output)
        self.assertNotIn('private text', output)


class UnpinnedCallTests(BackendCase):
    async def test_each_unpinned_call_reads_current_once_and_logs_the_profile_used(self):
        """FEAT-10: outside a turn every gateway entry point reads current() exactly once, and the log entry matches the request."""
        base = self.with_profile(structured_outputs=True)
        save_dash(self.store, 'test', dash_data('m0', structured_outputs=True))
        resolver = self.resolver(base)
        entries = []
        gateway = ModelGateway(resolver, log_sink=entries.append)
        request = SimpleNamespace(messages=MSG)
        real, count = resolver.current, [0]

        def current_then_edit():
            # Every read is followed by a dashboard edit, so a second read within one call would see another model.
            count[0] += 1
            settings = real()
            save_dash(self.store, 'test', dash_data(f'm{count[0]}', structured_outputs=True))
            return settings

        async def drain(stream):
            return [piece async for piece in stream]

        calls = {
            'text': lambda: gateway.text('dialogue', '', MSG),
            'structured': lambda: gateway.structured('director', '', MSG, 'choose_speakers', {'type': 'object'}),
            'stream_text': lambda: drain(gateway.stream_text('dialogue', '', MSG)),
            'text_compiled': lambda: gateway.text_compiled('dialogue', request),
            'structured_compiled': lambda: gateway.structured_compiled('director', request, 'choose_speakers', {'type': 'object'}),
            'stream_compiled': lambda: drain(gateway.stream_compiled('dialogue', request)),
        }
        for name, call in calls.items():
            with self.subTest(call=name), patch.object(resolver, 'current', current_then_edit):
                count[0], before = 0, len(entries)
                try:
                    await call()
                except Exception:
                    pass  # the fake returns plain text; structured calls may reject it after the request was made
                self.assertEqual(count[0], 1)
                self.assertEqual(len(entries) - before, 1)
                self.assertEqual(entries[-1]['model'], FakeOpenAI.models[-1])


class PinEnforcementTests(BackendCase):
    EVIL = {'provider': 'openai', 'model': 'gpt', 'context_tokens': 16000, 'base_url': 'https://evil.example/v1', 'api_key_env': 'OPENAI_API_KEY'}

    async def test_dashboard_profile_with_unpinned_host_is_refused_at_call_time(self):
        """FEAT-10: a stored dashboard profile sending OPENAI_API_KEY to an unpinned host raises, builds no client, and leaks no key."""
        os.environ['OPENAI_API_KEY'] = SECRET
        write_raw(self.store, 'test', self.EVIL)
        gateway = ModelGateway(self.resolver())
        for call in (lambda: gateway.text('dialogue', '', MSG),
                     lambda: gateway.structured('director', '', MSG, 'choose_speakers', {'type': 'object'})):
            with self.subTest(call=call), self.assertRaises(Exception) as caught:
                await call()
            message = str(caught.exception)
            self.assertIn('evil.example', message)
            self.assertIn('test', message)
            self.assertNotIn(SECRET, message)
        self.assertEqual(FakeOpenAI.created, [])
        self.assertEqual(FakeOpenAI.models, [])

    async def test_missing_key_variable_names_the_variable(self):
        """FEAT-10: a dashboard profile whose key variable is unset or empty fails naming it, with no placeholder key."""
        write_raw(self.store, 'test', {**self.EVIL, 'base_url': 'https://api.openai.com/v1'})
        gateway = ModelGateway(self.resolver())
        for value in (None, ''):
            if value is not None:
                os.environ['OPENAI_API_KEY'] = value
            with self.subTest(value=value), self.assertRaises(Exception) as caught:
                await gateway.text('dialogue', '', MSG)
            self.assertIn('OPENAI_API_KEY', str(caught.exception))
        self.assertEqual(FakeOpenAI.created, [])

    async def test_config_profile_with_the_same_shape_is_trusted(self):
        """FEAT-10: config.yaml profiles are not pinned: the same evil-host shape builds a client with the env key."""
        os.environ['OPENAI_API_KEY'] = SECRET
        base = self.with_profile(provider='openai', base_url='https://evil.example/v1', api_key_env='OPENAI_API_KEY')
        await ModelGateway(self.resolver(base)).text('dialogue', '', MSG)
        kwargs = FakeOpenAI.created[0].kwargs
        self.assertEqual((kwargs['api_key'], kwargs['base_url']), (SECRET, 'https://evil.example/v1'))

    async def test_missing_key_keeps_placeholder_for_config_and_keyless_dashboard_profiles(self):
        """FEAT-10: the local-no-key fallback stays for config profiles and for dashboard profiles with no key variable."""
        base = self.with_profile(api_key_env='OPENAI_API_KEY')
        await ModelGateway(self.resolver(base)).text('dialogue', '', MSG)
        self.assertEqual(FakeOpenAI.created[0].kwargs['api_key'], 'local-no-key')
        save_dash(self.store, 'test', dash_data('m1'))
        await ModelGateway(self.resolver()).text('dialogue', '', MSG)
        self.assertEqual(FakeOpenAI.created[1].kwargs['api_key'], 'local-no-key')


class ExplicitOriginTests(BackendCase):
    async def test_dashboard_profiles_pass_an_explicit_base_url(self):
        """FEAT-10: dashboard openai/anthropic/compatible clients get the resolved origin so env base URLs cannot redirect the key."""
        os.environ.update({'OPENAI_API_KEY': SECRET, 'ANTHROPIC_API_KEY': SECRET + 'a'})
        with patch.dict(os.environ, {'OPENAI_BASE_URL': 'https://evil.example/v1', 'ANTHROPIC_BASE_URL': 'https://evil.example'}):
            write_raw(self.store, 'test', {'provider': 'openai', 'model': 'gpt', 'context_tokens': 16000, 'api_key_env': 'OPENAI_API_KEY'})
            gateway = ModelGateway(self.resolver())
            await gateway.text('dialogue', '', MSG)
            write_raw(self.store, 'test', {'provider': 'anthropic', 'model': 'claude', 'context_tokens': 16000, 'api_key_env': 'ANTHROPIC_API_KEY'})
            await gateway.text('dialogue', '', MSG)
            write_raw(self.store, 'test', dash_data('m1'))
            await gateway.text('dialogue', '', MSG)
        self.assertEqual(FakeOpenAI.created[0].kwargs['base_url'], 'https://api.openai.com/v1')
        self.assertEqual(FakeOpenAI.created[0].kwargs['api_key'], SECRET)
        self.assertEqual(FakeAnthropic.created[0].kwargs['base_url'], 'https://api.anthropic.com')
        self.assertEqual(FakeAnthropic.created[0].kwargs['api_key'], SECRET + 'a')
        self.assertEqual(FakeOpenAI.created[1].kwargs['base_url'], 'http://localhost/v1')

    async def test_config_profiles_keep_todays_construction(self):
        """FEAT-10: config.yaml openai/anthropic profiles without base_url pass no base_url (SDK defaults and env apply)."""
        for provider, fake in (('openai', FakeOpenAI), ('anthropic', FakeAnthropic)):
            base = self.with_profile(provider=provider, base_url=None)
            await ModelGateway(self.resolver(base)).text('dialogue', '', MSG)
            self.assertNotIn('base_url', fake.created[-1].kwargs)
            self.assertEqual(sorted(fake.created[-1].kwargs), ['api_key', 'max_retries', 'timeout'])


class BotWiringTests(BackendCase):
    def setUp(self):
        super().setUp()
        self.bot = SkitBot(make_settings())
        self.addCleanup(self.bot.store.close)
        self.store = self.bot.store
        self.world = self.store.create_space(1, 'World', 'world')
        self.store.bind_channel(1, CHANNEL, self.world)
        self.alice = self.store.add_character(1, self.world, 'Alice', {'name': 'Alice'}, None, [])
        self.store.set_cast(CHANNEL, None, [self.alice])
        self.channel = FakeTextChannel(CHANNEL)

    async def asyncTearDown(self):
        for task in self.bot.memory_tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.sleep(0)

    async def test_one_snapshot_for_the_whole_turn_including_memory_work(self):
        """FEAT-10: director, speakers and the memory/summary task a turn starts all use the turn's snapshot; the next turn sees the edit."""
        save_dash(self.store, 'test', dash_data('m1'))
        bot, store = self.bot, self.store
        seen, gate = [], asyncio.Event()

        class Models(MemoryModels):
            changed = False

            def note(self, kind, role):
                seen.append((kind, role, bot.settings.profile(role).model))

            async def structured(self, role, system, messages, schema_name, schema):
                self.note('structured', role)
                return await super().structured(role, system, messages, schema_name, schema)

            async def stream_text(self, role, system, messages):
                self.note('stream', role)
                if not self.changed:
                    self.changed = True
                    save_dash(store, 'test', dash_data('m2'))
                async for piece in super().stream_text(role, system, messages):
                    yield piece

            async def text(self, role, system, messages, max_tokens=None):
                await gate.wait()
                self.note('text', role)
                return await super().text(role, system, messages, max_tokens)

        models = Models()
        models.chosen = [self.alice]
        bot.engine.models = bot.models = models

        async def turn(ident):
            scene = SceneContext(1, CHANNEL, None, self.world, 9, ident, 'Hi', None, [], [])
            async with bot.channel_locks.setdefault(CHANNEL, asyncio.Lock()):
                await asyncio.wait_for(bot.run_scene(scene, self.channel), 1.0)

        await turn(1001)
        gate.set()
        await drain_memory_tasks(bot)
        self.assertEqual(self.channel.errors, [])
        kinds = {(kind, role) for kind, role, _ in seen}
        self.assertTrue({('structured', 'director'), ('stream', 'dialogue'), ('text', 'memory')} <= kinds, seen)
        self.assertEqual({model for _, _, model in seen}, {'m1'}, seen)
        turn_one = len(seen)
        await turn(1002)
        await drain_memory_tasks(bot)
        self.assertEqual({model for _, _, model in seen[turn_one:]}, {'m2'}, seen)

    async def test_on_message_reads_the_backend_version_at_most_once_outside_a_turn(self):
        """FEAT-10: a message rejected before any turn (unsupported image) costs at most one model_backend_version read."""
        self.store.bind_channel(1, 200, self.world)
        user = SimpleNamespace(id=111, display_name='Sam', bot=False)
        bot_user = SimpleNamespace(id=999, display_name='Bot', bot=True, mention='<@999>')
        self.bot._connection.user = bot_user
        attachment = SimpleNamespace(size=10, content_type='image/png', read=AsyncMock(return_value=b'x'))
        reply = AsyncMock()
        message = SimpleNamespace(id=1000, guild=SimpleNamespace(id=1), channel=FakeTextChannel(200), author=user, content='<@999> look',
                                  mentions=[bot_user], webhook_id=None, reference=None, attachments=[attachment],
                                  created_at=datetime.now(timezone.utc), reply=reply)
        message.channel.history = lambda **kwargs: _empty()
        with patch.object(self.store, 'model_backend_version', wraps=self.store.model_backend_version) as reads:
            await self.bot.on_message(message)
        reply.assert_awaited_once()
        self.assertLessEqual(reads.call_count, 1)

    async def test_model_config_error_is_generic_for_members_and_detailed_in_the_log(self):
        """FEAT-10: a ModelConfigError shows members only the generic setup notice with a reference ID; the log keeps the detail."""
        detail = 'Profile test: OPENAI_API_KEY may only be sent to its pinned hosts; evil.example is not one of them'

        class Models(MemoryModels):
            async def stream_text(self, role, system, messages):
                raise ModelConfigError(detail)
                yield ''

        models = Models()
        models.chosen = [self.alice]
        self.bot.engine.models = self.bot.models = models
        followup = SimpleNamespace(send=AsyncMock())
        interaction = SimpleNamespace(followup=followup, response=SimpleNamespace(is_done=lambda: True))
        scene = SceneContext(1, CHANNEL, None, self.world, 9, 1001, 'Hi', None, [], [])
        async with self.bot.channel_locks.setdefault(CHANNEL, asyncio.Lock()):
            with self.assertLogs(level='ERROR') as logs:
                await asyncio.wait_for(self.bot.run_scene(scene, self.channel, interaction), 1.0)
        shown = [call.args[0] for call in followup.send.await_args_list] + [m.content for m in self.channel.sent]
        text = '\n'.join(shown)
        self.assertRegex(text, r"\(ref [0-9a-f]{6}\): The bot's model setup needs attention\. An admin can look it up with the reference ID\.")
        for secret in ('evil.example', 'OPENAI_API_KEY', 'pinned'):
            self.assertNotIn(secret, text)
        self.assertIn('evil.example', '\n'.join(logs.output))

    def test_log_entry_secret_list_includes_dashboard_profile_keys(self):
        """FEAT-10: the turn log redacts the key of a dashboard profile (not only config.yaml profiles)."""
        os.environ['OPENAI_API_KEY'] = SECRET
        write_raw(self.store, 'cloud', {'provider': 'openai', 'model': 'gpt', 'context_tokens': 16000, 'api_key_env': 'OPENAI_API_KEY'})
        self.store.set_turn_log(1, True, 14)
        with patch.object(self.store, 'add_turn_log', wraps=self.store.add_turn_log) as add:
            self.bot._log_entry(1, CHANNEL, 5, [], stage='test', request_text=f'key {SECRET} end')
        self.assertIn(SECRET, add.call_args.kwargs['secret_values'])
        entry = self.store.turn_log_entry(1, self.store.turn_log_page(1)[0]['id'])
        self.assertNotIn(SECRET, entry['request_text'])


if __name__ == '__main__':
    unittest.main()
