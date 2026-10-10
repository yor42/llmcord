"""MNT-16: configured key values are redacted whatever their variable is called; provider stages are named constants."""
from __future__ import annotations

import json
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

from helpers import FakeChannel, FakeInteraction, core_settings, make_settings
from llmcord_core import discord_bot, errors
from llmcord_core.backend import BackendResolver
from llmcord_core.config import ModelProfile
from llmcord_core.discord_bot import SkitBot
from llmcord_core.errors import error_detail, register_secret_names

VALUE = 'plain-odd-value-12345'


def profile(env='LLM_APIKEY'):
    return ModelProfile('compatible', 'm', 16000, True, base_url='http://localhost/v1', api_key_env=env)


class RegistryBase(unittest.TestCase):
    def setUp(self):
        for patcher in (patch.object(errors, '_SECRET_NAMES', set()), patch.dict(os.environ, {'LLM_APIKEY': VALUE, 'SHORT_NAME': 'ollama'})):
            patcher.start()
            self.addCleanup(patcher.stop)


class RedactRegistryTests(RegistryBase):
    def test_unregistered_non_suffix_name_is_not_known(self):
        self.assertIn(VALUE, error_detail(RuntimeError(f'bad {VALUE}')))

    def test_registered_name_is_redacted_from_error_detail(self):
        register_secret_names(['LLM_APIKEY'])
        detail = error_detail(RuntimeError(f'bad {VALUE}'))
        self.assertNotIn(VALUE, detail)
        self.assertIn('[redacted]', detail)

    def test_short_registered_value_is_not_redacted(self):
        register_secret_names(['SHORT_NAME', 'MISSING_NAME', ''])
        self.assertIn('ollama', error_detail(RuntimeError('ollama stays')))

    def test_config_profile_is_registered_on_load_and_redacted_in_error_detail_and_turn_log(self):
        import tempfile
        from pathlib import Path
        from llmcord_core.config import load_settings
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'DISCORD_BOT_TOKEN': 'tok'}):
            path = Path(folder) / 'config.yaml'
            path.write_text('models:\n  profiles:\n    p: {provider: compatible, model: m, context_tokens: 16000, base_url: "http://localhost/v1", api_key_env: LLM_APIKEY}\n'
                            '  dialogue: p\n  director: p\n  memory: p\n', encoding='utf-8')
            settings = load_settings(path)
        self.assertEqual(settings.profiles['p'].api_key_env, 'LLM_APIKEY')
        self.assertNotIn(VALUE, error_detail(RuntimeError(f'bad {VALUE}')))
        bot = SkitBot(replace(core_settings(), profiles=settings.profiles))
        self.addCleanup(bot.store.close)
        bot.store.set_turn_log(1, True, 14)
        bot._log_entry(1, 100, 5, [], stage='test', request_text=f'key {VALUE}', error_detail=lambda: error_detail(RuntimeError(VALUE)))
        entry = bot.store.turn_log_entry(1, bot.store.turn_log_page(1)[0]['id'])
        # Characterization: _log_entry already passes every profile's key to the store; this pins it for non-suffix names.
        self.assertNotIn(VALUE, json.dumps(entry))

    def test_resolver_registers_config_profile_names(self):
        """Only the config_profiles path is proven here; a dashboard profile's name must end in _API_KEY, so the suffix rule already covers it."""
        from llmcord_core.store import Store
        store = Store(':memory:')
        self.addCleanup(store.close)
        self.assertIn(VALUE, error_detail(RuntimeError(VALUE)))
        resolver = BackendResolver(store, make_settings(), config_profiles={'p': profile()})
        self.assertNotIn(VALUE, error_detail(RuntimeError(VALUE)))
        self.assertIn('p', resolver.snapshot().profiles)


class StageNoticeTests(unittest.IsolatedAsyncioTestCase):
    async def notice(self, stage):
        bot = SkitBot(core_settings())
        self.addCleanup(bot.store.close)
        it = FakeInteraction(guild_id=1)
        progress = type('P', (), {'message': None})()
        with self.assertLogs(level='ERROR'):
            await bot._report_scene_failure(RuntimeError('provider says no'), stage, progress, FakeChannel(100), it)
        return it.replies[0]

    async def test_provider_stages_show_the_provider_detail(self):
        for stage in (discord_bot.STAGE_SPEAKER_SELECTION, discord_bot.STAGE_IMAGE_DESCRIPTION, discord_bot.STAGE_DIALOGUE):
            text = await self.notice(stage)
            self.assertIn('RuntimeError: provider says no', text, stage)
            self.assertNotIn('internal error', text, stage)

    async def test_stage_labels_are_unchanged(self):
        self.assertEqual((discord_bot.STAGE_SPEAKER_SELECTION, discord_bot.STAGE_IMAGE_DESCRIPTION, discord_bot.STAGE_DIALOGUE),
                         ('speaker selection', 'image description', 'dialogue generation'))

    async def test_other_stages_say_internal_error(self):
        text = await self.notice('webhook delivery')
        self.assertIn('internal error', text)
        self.assertNotIn('provider says no', text)


if __name__ == '__main__':
    unittest.main()
