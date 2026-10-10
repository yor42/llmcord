"""FEAT-06: model calls and turn failures are recorded in the per-server turn log, with personal facts masked."""
from __future__ import annotations

import asyncio
import json
import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from helpers import FakeChannel, FakeInteraction, FakeTextChannel, FakeWebhook, core_settings, drain_memory_tasks, invoke
from llmcord_core.config import ModelProfile
from llmcord_core.discord_bot import SkitBot
from llmcord_core.engine import SceneContext
from llmcord_core.models import ImageInput, ModelGateway, TurnMessage
from llmcord_core import budget
from llmcord_core.usage import capture_usage, log_attribution
from test_catchup import ME, granted, recent

KEY = 'sk-test-abcdefghijklmnop'
FACT = 'secretly afraid of herons'
NEW_FACT = 'loves marmalade toast'


def profile():
    return ModelProfile('compatible', 'log-model', 16000, True, base_url='http://localhost/v1', structured_outputs=True,
                        api_key_env='LOGTEST_API_KEY')


def usage(inputs=10, outputs=5):
    return SimpleNamespace(prompt_tokens=inputs, completion_tokens=outputs)


def completion(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason='stop')], usage=usage())


class TurnLogBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {'LOGTEST_API_KEY': KEY})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.bot = SkitBot(replace(core_settings(), profiles={'test': profile()}))
        self.store = self.bot.store
        self.addCleanup(self.store.close)
        self.world = self.store.create_space(1, 'World', 'world')
        self.store.bind_channel(1, 100, self.world)
        self.alice = self.store.create_character(1, self.world, 'Alice')
        self.bob = self.store.create_character(1, self.world, 'Bob')
        self.store.set_cast(100, None, [self.alice, self.bob])
        self.prompts, self.extraction, self.fail_dialogue = [], {'shared_facts': [], 'personal_facts': [NEW_FACT], 'encounter_facts': []}, None
        self.hook = FakeWebhook(2000)

        async def webhook(channel, row):
            return self.hook
        self.bot._webhook = webhook

        async def create(**kwargs):
            self.prompts.append(kwargs)
            if kwargs.get('stream'):
                if self.fail_dialogue:
                    raise self.fail_dialogue

                async def stream():
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='<emotion>neutral</emotion>\nHello '))], usage=None)
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='there.'))], usage=None)
                    yield SimpleNamespace(choices=[], usage=usage(20, 7))
                return stream()
            schema = kwargs.get('response_format', {}).get('json_schema', {}).get('schema', {})
            if 'speakers' in schema.get('properties', {}):
                return completion(json.dumps({'speakers': [self.alice]}))
            if schema:
                return completion(json.dumps(self.extraction))
            return completion('Summary.')
        self.create = AsyncMock(side_effect=create)
        self.bot.models.clients['test'] = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))

    def scene(self):
        return SceneContext(1, 100, None, self.world, 9, 1000, 'Hello', None, [], [])

    def rows(self):
        return [dict(r) for r in self.store.all('SELECT * FROM turn_log ORDER BY id')]

    async def turn(self):
        channel = FakeChannel(100)
        await self.bot.run_scene(self.scene(), channel)
        await drain_memory_tasks(self.bot)
        return channel


class RecordingTurnTests(TurnLogBase):
    async def test_reply_turn_logs_every_model_call_with_attribution_and_tokens(self):
        """FEAT-06: director, dialogue and memory extraction are logged with guild, channel, message, stage, model and tokens."""
        self.store.set_turn_log(1, True, 14)
        await self.turn()
        rows = self.rows()
        by_stage = {r['stage']: r for r in rows}
        self.assertEqual(set(by_stage), {'reply · director', 'reply · dialogue', 'memory · extraction', 'memory · summary'})
        for row in rows:
            self.assertEqual((row['guild_id'], row['channel_id'], row['message_id'], row['profile'], row['model'], row['status']),
                             (1, 100, 1000, 'test', 'log-model', 'ok'))
            self.assertEqual((row['input_tokens'], row['output_tokens']), (20, 7) if row['stage'].endswith('dialogue') else (10, 5))
        self.assertIn('Hello there.', by_stage['reply · dialogue']['response_text'])
        self.assertIn('[user]', by_stage['reply · dialogue']['request_text'])
        self.assertEqual(len([r for r in rows if r['stage'] == 'reply · dialogue']), 1)

    async def test_nothing_is_logged_when_the_switch_is_off(self):
        """FEAT-06: with the server's turn log off no entry is written."""
        await self.turn()
        self.assertEqual(self.rows(), [])
        self.assertGreater(self.create.call_count, 1)

    async def test_personal_facts_are_masked_in_requests_and_extraction_responses(self):
        """FEAT-06: a consented member's stored fact and a freshly extracted fact never reach the log."""
        self.store.set_turn_log(1, True, 14)
        self.store.set_consent(1, 9, True)
        self.store.record_node(1, 1, 100, None, 9, None, 'earlier')
        self.store.add_personal(1, 9, self.alice, FACT, 1)
        await self.turn()
        dialogue = next(p for p in self.prompts if p.get('stream'))
        self.assertIn(FACT, json.dumps(dialogue['messages']))
        rows = self.rows()
        self.assertEqual(len(rows), 4)
        for row in rows:
            for column in ('request_text', 'response_text', 'error_detail'):
                self.assertNotIn(FACT, row[column])
                self.assertNotIn(NEW_FACT, row[column])
        self.assertIn('[personal fact hidden]', next(r for r in rows if r['stage'] == 'reply · dialogue')['request_text'])
        self.assertIn('[personal fact hidden]', next(r for r in rows if r['stage'] == 'memory · extraction')['response_text'])
        self.assertTrue(self.store.personal(1, 9))

    async def test_failed_turn_adds_an_error_entry_with_the_reference_the_member_saw(self):
        """FEAT-06: a turn failure is logged with its reference id, stage, detail and stack; the failed call is logged too."""
        self.store.set_turn_log(1, True, 14)
        self.fail_dialogue = RuntimeError('provider exploded')
        channel = await self.turn()
        ref = channel.errors[0].split('ref ')[1].split(')')[0]
        failure = next(r for r in self.rows() if r['reference_id'])
        self.assertEqual(failure['reference_id'], ref)
        self.assertEqual((failure['status'], failure['stage'], failure['message_id'], failure['channel_id']),
                         ('error', 'reply · failed at dialogue generation', 1000, 100))
        self.assertIn('provider exploded', failure['error_detail'])
        self.assertIn('File', failure['error_detail'])
        self.assertEqual(failure['request_text'] + failure['response_text'], '')
        call = next(r for r in self.rows() if r['stage'] == 'reply · dialogue')
        self.assertEqual((call['status'], call['reference_id']), ('error', ''))
        self.assertIn('provider exploded', call['error_detail'])

    async def test_catchup_is_logged_with_its_facts_masked(self):
        """FEAT-06: /catchup logs stage 'catchup · catchup' and masks the member's facts."""
        self.store.set_turn_log(1, True, 14)
        self.store.set_consent(1, ME, True)
        self.store.record_node(1, 1, 100, None, ME, None, 'earlier')
        self.store.add_personal(1, ME, self.alice, FACT, 1)
        self.create.side_effect = None
        self.create.return_value = completion('You missed a duel.')
        it = granted(FakeInteraction(channel=FakeTextChannel(100, [recent(1, 'hello', uid=5)]), user_id=ME))
        it.id = 777
        await invoke(self.bot, 'catchup', it)
        self.assertEqual(it.replies, ['You missed a duel.'])
        self.assertIn(FACT, json.dumps(self.create.call_args.kwargs['messages']))
        (row,) = self.rows()
        self.assertEqual((row['stage'], row['message_id'], row['channel_id'], row['guild_id']), ('catchup · catchup', 777, 100, 1))
        self.assertNotIn(FACT, row['request_text'])
        self.assertIn('[personal fact hidden]', row['request_text'])

    async def test_catchup_masks_long_and_reformatted_facts_as_rendered(self):
        """FEAT-06: facts clipped or whitespace-collapsed by /catchup are fully absent from the stored request."""
        long_fact = 'zebra ' * 41
        spaced = 'likes  tea\nwith   lemon'
        self.store.set_turn_log(1, True, 14)
        self.store.set_consent(1, ME, True)
        self.store.record_node(1, 1, 100, None, ME, None, 'earlier')
        for fact in (long_fact.strip(), spaced):
            self.store.add_personal(1, ME, self.alice, fact, 1)
        self.create.side_effect = None
        self.create.return_value = completion('ok')
        it = granted(FakeInteraction(channel=FakeTextChannel(100, [recent(1, 'hello', uid=5)]), user_id=ME))
        await invoke(self.bot, 'catchup', it)
        sent = json.dumps(self.create.call_args.kwargs['messages'])
        self.assertIn('likes tea with lemon', sent)
        (row,) = self.rows()
        for needle in ('zebra zebra', 'likes tea', 'with lemon', 'likes  tea'):
            self.assertNotIn(needle, row['request_text'])

    async def test_catchup_failure_masks_facts_in_error_text(self):
        """FEAT-06: a failed /catchup whose error echoes a fact stores it masked."""
        self.store.set_turn_log(1, True, 14)
        self.store.set_consent(1, ME, True)
        self.store.record_node(1, 1, 100, None, ME, None, 'earlier')
        self.store.add_personal(1, ME, self.alice, FACT, 1)
        self.create.side_effect = RuntimeError(f'provider echoed {FACT}')
        it = granted(FakeInteraction(channel=FakeTextChannel(100, [recent(1, 'hello', uid=5)]), user_id=ME))
        await invoke(self.bot, 'catchup', it)
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertNotIn(FACT, row['error_detail'])

    async def test_turns_and_guilds_keep_separate_mask_sets_and_the_footer_still_gets_usage(self):
        """FEAT-06: one turn's facts are not masked in another turn's log, the memory task keeps its own turn's set, and the usage footer still works."""
        self.store.set_turn_log(1, True, 14)
        self.store.set_turn_log(2, True, 14)
        self.store.set_usage_footer(1, True)
        self.store.set_consent(1, 9, True)
        self.store.record_node(1, 1, 100, None, 9, None, 'earlier')
        self.store.add_personal(1, 9, self.alice, FACT, 1)
        await self.turn()
        self.assertIn('Reply input: 20', self.hook.posts[0].content)
        self.assertTrue(any(r['stage'] == 'reply · dialogue' for r in self.rows()))
        before = len(self.rows())
        # a second turn by a member without consent mentioning the same words is not masked
        world2 = self.store.create_space(2, 'W2', 'world')
        self.store.bind_channel(2, 200, world2)
        alice2 = self.store.create_character(2, world2, 'Alice2')
        self.store.set_cast(200, None, [alice2])
        scene = SceneContext(2, 200, None, world2, 10, 2000, f'I am {FACT}', None, [], [])
        await self.bot.run_scene(scene, FakeChannel(200))
        await drain_memory_tasks(self.bot)
        later = self.rows()[before:]
        self.assertTrue(later)
        self.assertTrue(all(r['guild_id'] == 2 and r['message_id'] == 2000 for r in later))
        self.assertTrue(any(FACT in r['request_text'] for r in later))

    async def test_held_memory_task_keeps_its_own_turns_mask_set(self):
        """FEAT-06: a memory task finishing after another turn masks its own turn's facts, and the other turn is unaffected."""
        self.store.set_turn_log(1, True, 14)
        self.store.set_turn_log(2, True, 14)
        self.store.set_consent(1, 9, True)
        self.store.record_node(1, 1, 100, None, 9, None, 'earlier')
        self.store.add_personal(1, 9, self.alice, FACT, 1)
        self.extraction = {'shared_facts': [], 'personal_facts': [FACT], 'encounter_facts': []}
        gate, held = asyncio.Event(), []
        inner = self.create.side_effect

        async def gated(**kwargs):
            if kwargs.get('response_format', {}).get('json_schema', {}).get('schema', {}).get('properties', {}).get('personal_facts') and not held:
                held.append(1)
                await gate.wait()
            return await inner(**kwargs)
        self.create.side_effect = gated
        await self.bot.run_scene(self.scene(), FakeChannel(100))
        await asyncio.sleep(0.05)
        self.assertEqual(held, [1])
        world2 = self.store.create_space(2, 'W2', 'world')
        self.store.bind_channel(2, 200, world2)
        alice2 = self.store.create_character(2, world2, 'Alice2')
        self.store.set_cast(200, None, [alice2])
        await self.bot.run_scene(SceneContext(2, 200, None, world2, 10, 2000, f'I am {FACT}', None, [], []), FakeChannel(200))
        gate.set()
        await drain_memory_tasks(self.bot, within=3)
        rows = self.rows()
        mine = [r for r in rows if r['guild_id'] == 1 and r['stage'] == 'memory · extraction']
        self.assertEqual(len(mine), 1)
        self.assertNotIn(FACT, mine[0]['response_text'])
        self.assertIn('[personal fact hidden]', mine[0]['response_text'])
        other = [r for r in rows if r['guild_id'] == 2]
        self.assertTrue(any(FACT in r['request_text'] for r in other))

    async def test_catchup_failure_logs_its_reference(self):
        """FEAT-06: a failed /catchup logs an error entry with the ref shown to the member."""
        self.store.set_turn_log(1, True, 14)
        self.create.side_effect = RuntimeError('nope')
        it = granted(FakeInteraction(channel=FakeTextChannel(100, [recent(1, 'hello', uid=5)]), user_id=ME))
        it.id = 5
        await invoke(self.bot, 'catchup', it)
        ref = it.replies[0].split('ref ')[1].rstrip(')')
        stages = {r['stage']: r for r in self.rows()}
        self.assertEqual(stages['catchup · failed at catchup']['reference_id'], ref)
        self.assertEqual(stages['catchup · catchup']['status'], 'error')


class FailureWordingTests(TurnLogBase):
    DASHBOARD = 'An admin can find details in the dashboard under Monitoring → Log.'
    BOT_LOG = 'An admin can find details in the bot log.'

    async def command_error_text(self):
        from discord import app_commands
        it = FakeInteraction(guild_id=1)
        cmd = SimpleNamespace(name='x', qualified_name='admin x')
        await self.bot.tree.on_error(it, app_commands.CommandInvokeError(cmd, RuntimeError('boom')))
        return it.replies[0]

    async def test_scene_failure_points_to_the_dashboard_when_the_log_is_on(self):
        """UI-27: with the turn log on, the public failure notice names Monitoring → Log."""
        self.store.set_turn_log(1, True, 14)
        self.fail_dialogue = RuntimeError('provider exploded')
        channel = await self.turn()
        self.assertIn('(ref ', channel.errors[0])
        self.assertTrue(channel.errors[0].endswith(self.DASHBOARD), channel.errors[0])

    async def test_scene_failure_keeps_bot_log_wording_when_the_log_is_off(self):
        """UI-27: with the turn log off, the public failure notice still points at the bot log."""
        self.fail_dialogue = RuntimeError('provider exploded')
        channel = await self.turn()
        self.assertTrue(channel.errors[0].endswith(self.BOT_LOG), channel.errors[0])

    async def test_scene_failure_still_posts_when_the_setting_read_raises(self):
        """UI-27: a failing setting read falls back to bot-log wording and the notice still posts."""
        self.store.set_turn_log(1, True, 14)
        self.fail_dialogue = RuntimeError('provider exploded')
        with patch.object(self.store, 'turn_log_settings', side_effect=RuntimeError('db gone')):
            channel = await self.turn()
        self.assertEqual(len(channel.errors), 1)
        self.assertTrue(channel.errors[0].endswith(self.BOT_LOG), channel.errors[0])

    async def test_private_internal_error_detail_follows_the_same_choice(self):
        """UI-27: the private interaction detail for an internal error names Monitoring → Log when the log is on."""
        self.store.set_turn_log(1, True, 14)
        self.fail_dialogue = RuntimeError('provider exploded')
        it = FakeInteraction(guild_id=1)
        with patch.object(self.bot.engine, 'prepare_dialogue', side_effect=RuntimeError('internal')):
            await self.bot.run_scene(self.scene(), FakeChannel(100), it)
        self.assertIn(f'internal error. {self.DASHBOARD}', it.replies[0])

    async def test_command_error_wording(self):
        """UI-27: command errors name Monitoring → Log only when the log is on; DM, off and read failures keep the old text."""
        old = 'The error was logged.'
        self.assertTrue((await self.command_error_text()).endswith(old))
        self.store.set_turn_log(1, True, 14)
        self.assertTrue((await self.command_error_text()).endswith(self.DASHBOARD))
        with patch.object(self.store, 'turn_log_settings', side_effect=RuntimeError('db gone')):
            self.assertTrue((await self.command_error_text()).endswith(old))
        from discord import app_commands
        dm = FakeInteraction(guild_id=None)
        await self.bot.tree.on_error(dm, app_commands.CommandInvokeError(SimpleNamespace(name='x', qualified_name='x'), RuntimeError('boom')))
        self.assertTrue(dm.replies[0].endswith(old))


class GatewayLogTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.entries = []
        self.gateway = ModelGateway(replace(core_settings(), profiles={'test': profile()}), log_sink=self.entries.append)
        self.create = AsyncMock(return_value=completion('fine'))
        self.gateway.clients['test'] = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))

    async def test_images_render_as_placeholders_never_base64(self):
        """FEAT-06: image bytes are logged as a size placeholder."""
        message = TurnMessage('user', 'look', [ImageInput('image/png', b'\x89PNG' * 10)])
        await self.gateway.text('dialogue', 'be nice', [message])
        (entry,) = self.entries
        self.assertEqual(entry['render_request'](), 'be nice\n\n[user]\nlook\n[image: image/png, 40 bytes]')
        self.assertNotIn('base64', entry['render_request']())

    async def test_provider_error_is_logged_and_still_raised(self):
        """FEAT-06: a provider exception logs status 'error' with detail and the call still raises."""
        self.create.side_effect = RuntimeError('boom')
        with self.assertRaises(RuntimeError):
            await self.gateway.text('dialogue', '', [TurnMessage('user', 'hi')])
        (entry,) = self.entries
        self.assertEqual(entry['status'], 'error')
        self.assertIn('boom', entry['error_detail'])
        self.assertIsNone(entry['input_tokens'])

    async def test_sink_failure_never_breaks_the_call(self):
        """FEAT-06: a log sink exception is swallowed for text and structured calls."""
        def broken(entry):
            raise RuntimeError('disk full')
        self.gateway.log_sink = broken
        self.create.return_value = completion('{"speakers":[1]}')
        self.assertEqual(await self.gateway.text('dialogue', '', [TurnMessage('user', 'hi')]), '{"speakers":[1]}')
        self.assertEqual((await self.gateway.structured('director', '', [TurnMessage('user', 'hi')], 'n', {'type': 'object'}))['speakers'], [1])

    async def test_stream_logs_the_received_text_once(self):
        """FEAT-06: a stream is logged once, with the text received and the final usage."""
        async def stream():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='ab'))], usage=None)
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='cd'))], usage=None)
            yield SimpleNamespace(choices=[], usage=usage(3, 4))
        self.create.side_effect = None
        self.create.return_value = stream()
        self.assertEqual(''.join([d async for d in self.gateway.stream_text('dialogue', '', [TurnMessage('user', 'hi')])]), 'abcd')
        (entry,) = self.entries
        self.assertEqual((entry['response_text'], entry['input_tokens'], entry['output_tokens'], entry['status']), ('abcd', 3, 4, 'ok'))

    async def test_compatible_fallback_logs_one_entry_per_provider_request(self):
        """FEAT-06: the JSON-prompt structured fallback logs each actual request once."""
        self.gateway.settings.profiles['test'] = replace(profile(), structured_outputs=False)
        self.create.side_effect = [completion('not json'), completion('{"speakers":[2]}')]
        result = await self.gateway.structured('director', 'sys', [TurnMessage('user', 'hi')], 'n', {'type': 'object'})
        self.assertEqual(result, {'speakers': [2]})
        self.assertEqual([e['response_text'] for e in self.entries], ['not json', '{"speakers": [2]}'])
        self.assertIn('Try again', self.entries[1]['render_request']())

    async def test_sink_failure_never_breaks_a_stream(self):
        """FEAT-06: a log sink exception is swallowed for a stream call."""
        def broken(entry):
            raise RuntimeError('disk full')
        self.gateway.log_sink = broken
        self.create.side_effect = None
        self.create.return_value = chat_stream('ab', 'cd', final=usage(1, 1))
        self.assertEqual(''.join([d async for d in self.gateway.stream_text('dialogue', '', [TurnMessage('user', 'hi')])]), 'abcd')

    async def test_stream_closed_early_logs_one_error_entry_with_partial_text(self):
        """FEAT-06: a stream the consumer abandons logs once, as an error, with the partial text; usage is recorded once."""
        records = []
        self.gateway.usage_sink = records.append
        self.create.side_effect = None
        self.create.return_value = chat_stream('ab', 'cd', final=usage(3, 4))
        stream = self.gateway.stream_text('dialogue', '', [TurnMessage('user', 'hi')])
        self.assertEqual(await stream.__anext__(), 'ab')
        await stream.aclose()
        (entry,) = self.entries
        self.assertEqual((entry['status'], entry['response_text']), ('error', 'ab'))
        self.assertIn('stream closed before completion', entry['error_detail'])
        self.assertEqual(len(records), 1)

    async def test_cancelled_stream_is_logged_as_an_error(self):
        """FEAT-06: a cancelled stream logs 'stream closed before completion' and re-raises."""
        gate = asyncio.Event()

        async def slow():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='ab'))], usage=None)
            await gate.wait()
        self.create.side_effect = None
        self.create.return_value = slow()

        async def consume():
            return [d async for d in self.gateway.stream_text('dialogue', '', [TurnMessage('user', 'hi')])]
        task = asyncio.create_task(consume())
        await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        (entry,) = self.entries
        self.assertEqual((entry['status'], entry['response_text']), ('error', 'ab'))
        self.assertIn('CancelledError', entry['error_detail'])

    async def test_invalid_fallback_json_logs_the_final_error(self):
        """FEAT-06: when the compatible fallback gives up, the last attempt is logged as the error."""
        self.gateway.settings.profiles['test'] = replace(profile(), structured_outputs=False)
        self.create.side_effect = [completion('nope'), completion('still nope')]
        with self.assertRaises(ValueError):
            await self.gateway.structured('director', '', [TurnMessage('user', 'hi')], 'n', {'type': 'object'})
        self.assertEqual([e['status'] for e in self.entries], ['ok', 'error'])
        self.assertIn('invalid JSON', self.entries[1]['error_detail'])

    async def test_budget_refusal_logs_nothing_and_sends_nothing(self):
        """FEAT-06: a hard-cap refusal raises before any provider call or log entry, including between fallback attempts."""
        state = SimpleNamespace(hard_reached=True)
        self.gateway.budget_gate = lambda: state
        for call in (lambda: self.gateway.text('dialogue', '', [TurnMessage('user', 'hi')]),
                     lambda: self.gateway.structured('director', '', [TurnMessage('user', 'hi')], 'n', {'type': 'object'}),
                     lambda: self.gateway.stream_text('dialogue', '', [TurnMessage('user', 'hi')]).__anext__()):
            with self.assertRaises(budget.BudgetExceeded):
                await call()
        self.gateway.settings.profiles['test'] = replace(profile(), structured_outputs=False)
        self.gateway.budget_gate = iter([None, None, state]).__next__
        self.create.return_value = completion('nope')
        with self.assertRaises(budget.BudgetExceeded):
            await self.gateway.structured('director', '', [TurnMessage('user', 'hi')], 'n', {'type': 'object'})
        self.assertEqual(len(self.entries), 1)
        self.assertEqual(self.create.call_count, 1)

    async def test_no_sink_makes_no_sink_call(self):
        """FEAT-06: without a log sink the gateway behaves as before and renders nothing."""
        self.gateway.log_sink = None
        with patch.object(ModelGateway, '_render', side_effect=AssertionError):
            self.assertEqual(await self.gateway.text('dialogue', '', [TurnMessage('user', 'hi')]), 'fine')

    async def test_sink_runs_in_the_context_the_call_started_in(self):
        """FEAT-06: attribution is captured at call start, even if the stream is finalized elsewhere."""
        seen = []
        self.gateway.log_sink = lambda entry: seen.append(log_attribution()['guild_id'])
        self.create.side_effect = None
        self.create.return_value = chat_stream('ab', final=usage(1, 1))
        with capture_usage(7):
            stream = self.gateway.stream_text('dialogue', '', [TurnMessage('user', 'hi')])
            await stream.__anext__()
        await asyncio.create_task(stream.aclose())
        self.assertEqual(seen, [7])


def chat_stream(*texts, final=None):
    async def stream():
        for text in texts:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))], usage=None)
        if final:
            yield SimpleNamespace(choices=[], usage=final)
    return stream()


class SecretRedactionTests(TurnLogBase):
    async def test_configured_api_key_is_redacted_from_logged_text(self):
        """FEAT-06: the configured provider key never appears in a stored entry."""
        self.store.set_turn_log(1, True, 14)
        with capture_usage(1, 100, 'reply'):
            self.bot._log_model_call({'role': 'dialogue', 'profile': 'test', 'model': 'm', 'render_request': lambda: f'key {KEY} here',
                                      'response_text': '', 'input_tokens': None, 'output_tokens': None, 'status': 'ok', 'error_detail': ''})
        (row,) = self.rows()
        self.assertNotIn(KEY, row['request_text'])
        self.assertIn('[redacted]', row['request_text'])


if __name__ == '__main__':
    unittest.main()
