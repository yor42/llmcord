import os
import unittest
from pathlib import Path
from unittest.mock import patch

from llmcord_core.config import load_settings, operator_ids_from_env
from llmcord_core.web import create_app


def parse(value):
    return operator_ids_from_env({} if value is None else {'LLMCORD_OPERATOR_IDS': value})


class OperatorIdTests(unittest.TestCase):
    def test_unset_and_empty_mean_no_operators(self):
        self.assertEqual(parse(None), frozenset())
        self.assertEqual(parse(''), frozenset())
        self.assertEqual(parse(' , ,'), frozenset())

    def test_whitespace_and_trailing_comma_are_ignored(self):
        self.assertEqual(parse(' 1 , 2, '), frozenset({1, 2}))
        self.assertEqual(parse('7,'), frozenset({7}))

    def test_two_ids(self):
        self.assertEqual(parse('111,222'), frozenset({111, 222}))

    def test_malformed_entries_raise_naming_variable(self):
        for bad in ('abc', '-5', '12x', '1 2', '+3', '1,x', '１２', '²', '١٢', '1_000', '0', '000'):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'LLMCORD_OPERATOR_IDS'):
                parse(bad)

    def test_overlong_entries_raise_short_message(self):
        for bad in ('1' * 21, '1' * 5000):
            with self.subTest(n=len(bad)), self.assertRaisesRegex(ValueError, 'LLMCORD_OPERATOR_IDS') as ctx:
                parse(bad)
            self.assertLess(len(str(ctx.exception)), 200)

    def test_leading_zeros_duplicates_and_max_length(self):
        self.assertEqual(parse('007'), frozenset({7}))
        self.assertEqual(parse('5,5'), frozenset({5}))
        self.assertEqual(parse('1' * 20), frozenset({int('1' * 20)}))

    def test_message_names_bad_entry(self):
        with self.assertRaisesRegex(ValueError, "'12x'"):
            parse('1,12x')

    def test_default_environment_is_process_environment(self):
        with patch.dict(os.environ, {'LLMCORD_OPERATOR_IDS': '5'}):
            self.assertEqual(operator_ids_from_env(), frozenset({5}))

    def test_load_settings_exposes_operator_ids(self):
        config = Path(__file__).resolve().parents[1] / 'config-gemini.yaml'
        env = {'DISCORD_BOT_TOKEN': 'test', 'GEMINI_API_KEY': 'test'}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(load_settings(config).operator_ids, frozenset())
        with patch.dict(os.environ, {**env, 'LLMCORD_OPERATOR_IDS': '10, 20'}, clear=True):
            self.assertEqual(load_settings(config).operator_ids, frozenset({10, 20}))
        with patch.dict(os.environ, {**env, 'LLMCORD_OPERATOR_IDS': 'nope'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'LLMCORD_OPERATOR_IDS'):
                load_settings(config)

    def test_create_app_stores_operator_ids(self):
        args = (':memory:', 'https://pi.test', 'client', 'secret', 'bot')
        for kwargs, expected in (({}, frozenset()), ({'operator_ids': frozenset({9})}, frozenset({9}))):
            app = create_app(*args, enable_dashboard=False, **kwargs)
            self.addCleanup(app.state.store.close)
            self.assertEqual(app.state.operator_ids, expected)


if __name__ == '__main__':
    unittest.main()
