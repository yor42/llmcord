"""FEAT-15 step 1: the spending hard cap is enforced in ModelGateway; started turns hold a pass."""
from __future__ import annotations

import asyncio
import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import test_budget_gate
from llmcord_core import budget
from llmcord_core.config import ModelProfile, Settings
from llmcord_core.models import ModelGateway, TurnMessage

MSG = [TurnMessage('user', 'hi')]


def make_state(spent=0.0, soft=None, hard=None):
    return budget.BudgetState(soft, hard, 1, False, spent, 0, '2026-10-01', None, 0)


def completion(content='ok'):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason='stop')], usage=None)


class Harness:
    def __init__(self, gate, with_gate=True):
        profile = ModelProfile('compatible', 'm', 16000, True, base_url='http://localhost/v1', structured_outputs=True)
        settings = Settings('token', None, ':memory:', 90, {'test': profile}, 'test', 'test', 'test', {'max_output_tokens': 700})
        self.gate_calls = 0

        def counted():
            self.gate_calls += 1
            if isinstance(gate, Exception):
                raise gate
            return gate

        self.gateway = ModelGateway(settings, budget_gate=counted) if with_gate else ModelGateway(settings)

        async def create(**kwargs):
            if kwargs.get('stream'):
                async def stream():
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='ok'))], usage=None)
                return stream()
            return completion('{"speakers":[1]}' if 'response_format' in kwargs else 'ok')

        self.create = AsyncMock(side_effect=create)
        self.gateway.clients['test'] = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))

    async def call_all(self):
        g = self.gateway
        out = [await g.text('dialogue', '', MSG),
               (await g.structured('director', '', MSG, 'choose_speakers', {'type': 'object'}))['speakers'],
               ''.join([p async for p in g.stream_text('dialogue', '', MSG)])]
        return out


class GatewayBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_hard_cap_blocks_every_entry_point_before_the_provider(self):
        h = Harness(make_state(2.0, hard=1.0))
        for call in (lambda: h.gateway.text('dialogue', '', MSG),
                     lambda: h.gateway.structured('director', '', MSG, 'choose_speakers', {'type': 'object'}),
                     lambda: h.gateway.stream_text('dialogue', '', MSG).__anext__()):
            with self.subTest(call=call), self.assertRaises(budget.BudgetExceeded) as caught:
                await call()
            self.assertTrue(caught.exception.state.hard_reached)
        self.assertEqual(h.create.call_count, 0)

    async def test_compiled_wrappers_are_gated(self):
        h = Harness(make_state(2.0, hard=1.0))
        request = SimpleNamespace(messages=MSG)
        with self.assertRaises(budget.BudgetExceeded):
            await h.gateway.text_compiled('dialogue', request)
        self.assertEqual(h.create.call_count, 0)

    async def test_admitted_calls_go_through_without_consulting_the_gate(self):
        h = Harness(make_state(2.0, hard=1.0))
        with budget.admit():
            self.assertTrue(budget.admitted())
            self.assertEqual(await h.call_all(), ['ok', [1], 'ok'])
        self.assertFalse(budget.admitted())
        self.assertEqual((h.create.call_count, h.gate_calls), (3, 0))

    async def test_soft_cap_only_does_not_block(self):
        h = Harness(make_state(2.0, soft=1.0, hard=10.0))
        self.assertEqual(await h.call_all(), ['ok', [1], 'ok'])
        self.assertEqual(h.gate_calls, 3)

    async def test_gate_returning_none_or_raising_fails_open(self):
        for gate in (None, RuntimeError('db down')):
            with self.subTest(gate=gate):
                h = Harness(gate)
                with self.assertLogs(level='ERROR') if gate else self.assertNoLogs(level='ERROR'):
                    self.assertEqual(await h.call_all(), ['ok', [1], 'ok'])
                self.assertEqual((h.create.call_count, h.gate_calls), (3, 3))

    async def test_no_gate_means_no_check(self):
        h = Harness(make_state(2.0, hard=1.0), with_gate=False)
        self.assertEqual(await h.call_all(), ['ok', [1], 'ok'])
        self.assertEqual(h.gate_calls, 0)

    async def test_pass_is_inherited_by_tasks_created_inside_but_does_not_leak_after(self):
        h = Harness(make_state(2.0, hard=1.0))
        started = asyncio.Event()

        async def work():
            await started.wait()
            return budget.admitted(), await h.gateway.text('memory', '', MSG)

        with budget.admit():
            task = asyncio.create_task(work())
        self.assertFalse(budget.admitted())
        with self.assertRaises(budget.BudgetExceeded):
            await h.gateway.text('dialogue', '', MSG)
        started.set()
        self.assertEqual(await task, (True, 'ok'))
        self.assertEqual(h.create.call_count, 1)


class MidTurnCapTests(test_budget_gate.GateCase):
    async def test_admitted_turn_finishes_all_speakers_and_memory_after_cap_is_crossed(self):
        bob = self.store.add_character(1, self.world, 'Bob', {'name': 'Bob'}, None, [])
        self.store.set_cast(test_budget_gate.CHANNEL, None, [self.alice, bob])
        profiles = self.bot.settings.profiles
        profiles['test'] = dataclasses.replace(profiles['test'], structured_outputs=True)
        gateway = ModelGateway(self.bot.settings, usage_sink=self.store.record_model_usage, budget_gate=self.bot._budget_check)
        self.bot.models = self.bot.engine.models = gateway
        kinds = []

        async def create(**kwargs):
            schema = kwargs.get('response_format', {}).get('json_schema', {}).get('schema', {})
            if kwargs.get('stream'):
                kinds.append('stream')
                if kinds.count('stream') == 1:
                    self.spend(5.0)
                async def stream():
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                        content='<emotion>neutral</emotion>\nHello.'))], usage=None)
                return stream()
            if 'speakers' in schema.get('properties', {}):
                kinds.append('director')
                text = '{"speakers":[%d,%d]}' % (self.alice, bob)
            elif schema:
                kinds.append('extract')
                text = '{"shared_facts":[],"personal_facts":[],"encounter_facts":[]}'
            else:
                kinds.append('summary')
                text = 'Summary'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason='stop')], usage=None)

        client = AsyncMock(side_effect=create)
        gateway.clients['test'] = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=client)))
        self.caps(hard=1.0)
        self.spend(0.5)
        await self.turn()
        self.assertTrue(self.bot._budget_check().hard_reached)
        self.assertEqual(kinds.count('stream'), 2)
        self.assertEqual(len(self.replies), 2)
        self.assertIn('extract', kinds)
        self.assertIn('summary', kinds)
        before = client.call_count
        await self.turn(1001)
        self.assertEqual(client.call_count, before)
        self.assertEqual(len(self.replies), 2)


if __name__ == '__main__':
    unittest.main()
