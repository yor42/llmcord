import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from types import SimpleNamespace

from llmcord_core.config import load_settings, prompt_provider
from llmcord_core.models import ModelGateway, TurnMessage
from llmcord_core.prompts import compile_prompt, default_bundle


class HostingTests(unittest.IsolatedAsyncioTestCase):
    def config(self):
        return Path(__file__).resolve().parents[1] / 'config-gemini.yaml'

    def test_environment_overrides_shared_database_and_guild(self):
        with patch.dict(os.environ, {'DISCORD_BOT_TOKEN': 'test', 'GEMINI_API_KEY': 'test',
                                    'DISCORD_GUILD_ID': '123', 'LLMCORD_DATABASE_PATH': '/tmp/shared.sqlite3'}, clear=True):
            settings = load_settings(self.config())
        self.assertEqual(settings.development_guild_id, 123)
        self.assertEqual(settings.database_path, Path('/tmp/shared.sqlite3'))

    def test_cloud_compatible_profile_requires_its_key(self):
        with patch.dict(os.environ, {'DISCORD_BOT_TOKEN': 'test'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'API key environment variable is missing'):
                load_settings(self.config())

    def test_gemini_prompt_rules_apply_only_to_the_official_compatible_host(self):
        self.assertEqual(prompt_provider('compatible', 'https://generativelanguage.googleapis.com/v1beta/openai/'), 'gemini')
        self.assertEqual(prompt_provider('compatible', 'https://generativelanguage.googleapis.com.example.com/v1/'), 'compatible')
        self.assertEqual(prompt_provider('compatible', 'http://localhost:11434/v1'), 'compatible')
        self.assertEqual(prompt_provider('anthropic'), 'anthropic')

    async def test_gemini_compiled_text_json_and_stream_keep_all_system_blocks(self):
        with patch.dict(os.environ, {'DISCORD_BOT_TOKEN': 'test', 'GEMINI_API_KEY': 'test'}, clear=True):
            settings = load_settings(self.config())
        model = ModelGateway(settings)
        create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"speakers":[1]}'))]))
        model.clients['google/flash'] = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        request = compile_prompt(default_bundle(), 'dialogue', {'char': 'Alice', 'description': 'BACKGROUND', 'card_post_history': 'FINAL'}, [TurnMessage('user', 'INPUT')], settings.profile('dialogue').prompt_provider, 4000, contract='CONTRACT')
        await model.text_compiled('dialogue', request)
        sent = create.call_args.kwargs['messages']
        self.assertEqual([m['role'] for m in sent], ['system', 'user'])
        for text in ('CONTRACT', 'BACKGROUND', 'Alice', 'FINAL'):
            self.assertIn(text, sent[0]['content'])
        self.assertTrue(sent[0]['content'].endswith('FINAL'))
        await model.structured_compiled('director', request, 'test', {'type': 'object'})
        self.assertEqual(create.call_args.kwargs['messages'], sent)
        async def stream():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='Hello'))])
        create.return_value = stream()
        self.assertEqual(''.join([part async for part in model.stream_compiled('dialogue', request)]), 'Hello')
        self.assertEqual(create.call_args.kwargs['messages'], sent)

    def test_local_compatible_profile_remains_keyless(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text('discord: {}\nmodels:\n  dialogue: local\n  profiles:\n    local:\n      provider: compatible\n      model: local\n      context_tokens: 8192\n      base_url: http://localhost:11434/v1\n')
            with patch.dict(os.environ, {'DISCORD_BOT_TOKEN': 'test'}, clear=True):
                self.assertIsNone(load_settings(path).profile('dialogue').api_key_env)

    async def test_gemini_reasoning_setting_reaches_text_json_and_stream(self):
        with patch.dict(os.environ, {'DISCORD_BOT_TOKEN': 'test', 'GEMINI_API_KEY': 'test'}, clear=True):
            settings = load_settings(self.config())
        gateway = ModelGateway(settings)
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"speakers":[1]}'), finish_reason='stop')])
        create = AsyncMock(return_value=response)
        gateway.clients['google/flash'] = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        messages = [TurnMessage('user', 'hello')]
        await gateway.text('dialogue', 'rules', messages)
        self.assertEqual(create.call_args.kwargs['reasoning_effort'], 'low')
        await gateway.structured('director', 'rules', messages, 'test', {'type': 'object'})
        self.assertEqual(create.call_args.kwargs['reasoning_effort'], 'low')
        self.assertEqual(create.call_args.kwargs['response_format']['type'], 'json_schema')
        self.assertEqual(create.call_args.kwargs['response_format']['json_schema']['schema'], {'type': 'object'})
        async def stream():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='Hello'))])
        create.return_value = stream()
        self.assertEqual(''.join([part async for part in gateway.stream_text('dialogue', 'rules', messages)]), 'Hello')
        self.assertEqual(create.call_args.kwargs['reasoning_effort'], 'low')


if __name__ == '__main__':
    unittest.main()
