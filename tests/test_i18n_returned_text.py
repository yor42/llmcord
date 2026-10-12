"""MNT-76 (D39): text that shared modules return (not raise) also comes as a structured Phrase, so the dashboard can translate it."""
import asyncio
import inspect
import os
import unittest
from unittest import mock

import httpx

from llmcord_core import backend as backend_module
from llmcord_core import dashboard, i18n, prices, prompts
from llmcord_core.backend import effective_backend, list_profile_models
from llmcord_core.config import key_hosts_from_env
from llmcord_core.i18n import Phrase
from llmcord_core.prices import LITELLM_URL, NOT_FOUND, PriceResult, suggest_prices
from llmcord_core.prompts import block, compatibility, compile_prompt, default_bundle

FAKE_VALUE = 'fake-returned-text-value'
ENV = {'OPENAI_API_KEY': FAKE_VALUE}
NO_ROLES = {'dialogue': None, 'director': None, 'memory': None, 'revision': 0, 'version': 0}


def run(coro):
    return asyncio.run(coro)


def pseudo_translator():
    return i18n.Translator('pseudo')


class PseudoMixin:
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {'LLMCORD_I18N_PSEUDO': '1'})
        patcher.start()
        i18n.clear_cache()
        self.addCleanup(i18n.clear_cache)
        self.addCleanup(patcher.stop)
        self.pseudo = pseudo_translator()
        self.english = i18n.english()


def every_problem_bundle():
    bundle = default_bundle()
    bundle['purposes']['dialogue'] = [
        block('hist', 'history', role='user'),
        block('bad', 'bogus'),
        block('mac', content='Hi {{zzz}} {{aaa}}'),
        block('trig', raw={'injection_trigger': ['continue']}),
        block('ext', raw={'extension': True}),
        block('late', content='x', role='system'),
        block('lore', 'lore_in_chat', role='user')]
    bundle['purposes']['director'] = [block('main', 'description', placement='in_chat'), block('payload', 'payload', role='system')]
    bundle['source'] = {'assistant_prefill': 'x'}
    return bundle


def expected_problems(article, label):
    return [
        'dialogue/Bad: remap unsupported marker bogus',
        f'dialogue/Bad: choose {article} {label} placement adaptation',
        'dialogue/Mac: unsupported macros aaa, zzz',
        f'dialogue/Mac: choose {article} {label} placement adaptation',
        'dialogue/Trig: remap generation triggers',
        f'dialogue/Trig: choose {article} {label} placement adaptation',
        'dialogue/Ext: remove extension dependency',
        f'dialogue/Ext: choose {article} {label} placement adaptation',
        f'dialogue/Late: choose {article} {label} placement adaptation',
        f'dialogue/Lore: choose {article} {label} placement adaptation',
        'director/Main: remap unsupported marker description',
        'director/Main: use relative placement; this purpose has no conversation-history marker',
        f'director/Main: choose {article} {label} placement adaptation',
        'director/Payload: the required input payload must use the user role',
        'Disable or remap imported assistant prefill']


class CompatibilityTests(PseudoMixin, unittest.TestCase):
    def test_compatibility_strings_are_unchanged_for_every_problem_kind(self):
        """MNT-76: compatibility() still returns the exact English strings, for Anthropic and for Gemini."""
        bundle = every_problem_bundle()
        self.assertEqual(compatibility(bundle, {'dialogue': 'anthropic', 'director': 'anthropic'}), expected_problems('an', 'Anthropic'))
        self.assertEqual(compatibility(bundle, {'dialogue': 'gemini', 'director': 'gemini'}), expected_problems('a', 'Gemini'))

    def test_phrases_render_to_the_same_english(self):
        """MNT-76: compatibility_phrases() gives one Phrase per problem; str() and the en translator equal the old strings."""
        bundle = every_problem_bundle()
        for provider, article, label in (('anthropic', 'an', 'Anthropic'), ('gemini', 'a', 'Gemini')):
            phrases = prompts.compatibility_phrases(bundle, {'dialogue': provider, 'director': provider})
            self.assertTrue(all(isinstance(p, Phrase) for p in phrases))
            self.assertEqual([str(p) for p in phrases], expected_problems(article, label))
            self.assertEqual([self.english.render(p) for p in phrases], expected_problems(article, label))

    def test_phrases_render_pseudo_text(self):
        """MNT-76: every compatibility phrase renders as pseudo text, so none is finished English."""
        phrases = prompts.compatibility_phrases(every_problem_bundle(), {'dialogue': 'anthropic', 'director': 'gemini'})
        self.assertEqual(len(phrases), 15)
        for phrase in phrases:
            self.assertTrue(self.pseudo.render(phrase).startswith('[!! '), phrase)

    def test_provider_choice_is_two_whole_messages(self):
        """MNT-76: the Anthropic and Gemini placement problems are different whole messages, not one message with a spliced label."""
        bundle = every_problem_bundle()
        anthropic = prompts.compatibility_phrases(bundle, {'dialogue': 'anthropic'})
        gemini = prompts.compatibility_phrases(bundle, {'dialogue': 'gemini'})
        pairs = [(a, g) for a, g in zip(anthropic, gemini) if str(a) != str(g)]
        self.assertTrue(pairs)
        for a, g in pairs:
            self.assertNotEqual(str(a.msg), str(g.msg))

    def test_no_problems_gives_no_phrases(self):
        """MNT-76: a clean bundle yields an empty list from both functions."""
        self.assertEqual(prompts.compatibility_phrases(default_bundle(), {'dialogue': 'compatible'}), [])
        self.assertEqual(compatibility(default_bundle(), {'dialogue': 'compatible'}), [])

    def test_compile_prompt_error_text_is_unchanged(self):
        """MNT-76: compile_prompt still raises ValueError with the old joined English."""
        bundle = default_bundle()
        bundle['purposes']['dialogue'] = [block('hist', 'history', role='user'), block('mac', content='Hi {{zzz}}')]
        with self.assertRaises(ValueError) as caught:
            compile_prompt(bundle, 'dialogue', {}, [], 'compatible', 4000)
        self.assertEqual(str(caught.exception), 'dialogue/Mac: unsupported macros zzz.')


def row(name, data, revision=1):
    return {'name': name, 'data': data, 'revision': revision, 'updated_at': 0}


class BackendProblemTests(PseudoMixin, unittest.TestCase):
    def backend(self):
        rows = [row('Bad Name', {}), row('broken', {'provider': 'nope'}), row('unreadable', 'not a mapping')]
        return effective_backend({}, {}, rows, {**NO_ROLES, 'director': 'ghost'})

    def test_problem_phrases_match_problems_in_english(self):
        """MNT-76: Backend.problems stays str; problem_phrases is the parallel tuple of Phrase that renders the same English."""
        result = self.backend()
        self.assertEqual(len(result.problems), 4)
        self.assertIsInstance(result.problems, tuple)
        self.assertTrue(all(isinstance(p, str) for p in result.problems))
        self.assertIsInstance(result.problem_phrases, tuple)
        self.assertTrue(all(isinstance(p, Phrase) for p in result.problem_phrases))
        self.assertEqual([str(p) for p in result.problem_phrases], list(result.problems))
        self.assertEqual([self.english.render(p) for p in result.problem_phrases], list(result.problems))

    def test_problem_texts_are_unchanged(self):
        """MNT-76: the English problem strings for the invalid name, unreadable row and missing role stay byte-identical."""
        problems = self.backend().problems
        self.assertEqual(problems[0], 'A saved profile with an invalid name was skipped.')
        self.assertEqual(problems[2], 'Profile unreadable was skipped: its saved settings could not be read')
        self.assertEqual(problems[3], 'The director role names profile ghost, which is not available; using the config.yaml setting.')

    def test_problem_phrases_render_pseudo_text(self):
        """MNT-76: every Backend problem renders as pseudo text, including a skip reason that came from a coded error."""
        for phrase in self.backend().problem_phrases:
            self.assertTrue(self.pseudo.render(phrase).startswith('[!! '), phrase)

    def test_default_problem_phrases_is_empty(self):
        """MNT-76: a Backend built without problems has an empty problem_phrases tuple by default."""
        result = effective_backend({}, {}, [], NO_ROLES)
        self.assertEqual((result.problems, result.problem_phrases), ((), ()))
        direct = backend_module.Backend({}, {}, (), 0)
        self.assertEqual(direct.problem_phrases, ())


class Recorder:
    def __init__(self, handler):
        self.handler = handler

    @property
    def transport(self):
        return httpx.MockTransport(self.handler)


def mapping(provider='compatible', **extra):
    return {'provider': provider, 'model': 'model-list', 'context_tokens': 1, **extra}


def hosts(**pins):
    return key_hosts_from_env({'LLMCORD_KEY_HOSTS': ','.join(f'{k}={v}' for k, v in pins.items())})


def listing(provider_mapping, handler, environ=ENV):
    return run(list_profile_models('p', provider_mapping, hosts(), environ, Recorder(handler).transport))


def unreachable(request):
    raise httpx.ConnectError('down')


class ModelListPhraseTests(PseudoMixin, unittest.TestCase):
    def fixed_results(self):
        compatible = mapping(base_url='https://lister.example/v1')
        ok = lambda n: lambda r: httpx.Response(200, json={'data': [{'id': f'm{i}'} for i in range(n)]})  # noqa: E731
        return {
            'one': listing(compatible, ok(1)),
            'two': listing(compatible, ok(2)),
            'none': listing(compatible, ok(0)),
            'not_found_compatible': listing(compatible, lambda r: httpx.Response(404)),
            'not_found_openai': listing(mapping('openai'), lambda r: httpx.Response(404)),
            'unparseable': listing(compatible, lambda r: httpx.Response(200, content=b'not json')),
            'refused': listing(mapping('openai', api_key_env='OPENAI_API_KEY'), lambda r: httpx.Response(401)),
            'needs_key': listing(compatible, lambda r: httpx.Response(403)),
            'server_error': listing(compatible, lambda r: httpx.Response(500)),
            'unreachable': listing(compatible, unreachable),
            'env_missing': listing(mapping('openai', api_key_env='OPENAI_API_KEY'), lambda r: httpx.Response(200, json=[]), environ={}),
        }

    def test_fixed_messages_carry_a_phrase_equal_to_the_message(self):
        """MNT-76: every fixed model-list message has a phrase whose English rendering equals message byte for byte."""
        for name, result in self.fixed_results().items():
            self.assertIsInstance(result.phrase, Phrase, name)
            self.assertEqual(str(result.phrase), result.message, name)
            self.assertEqual(self.english.render(result.phrase), result.message, name)
            self.assertTrue(self.pseudo.render(result.phrase).startswith('[!! '), name)

    def test_messages_are_unchanged(self):
        """MNT-76: the English message text of the model list stays byte-identical, including the plural."""
        results = self.fixed_results()
        self.assertEqual(results['one'].message, '1 model found.')
        self.assertEqual(results['two'].message, '2 models found.')
        self.assertEqual(results['none'].message, 'The provider returned no models; type the ID.')
        self.assertEqual(results['not_found_compatible'].message, "This provider doesn't list its models; type the ID.")
        self.assertEqual(results['not_found_openai'].message, "The provider's answer was not a model list; type the ID.")
        self.assertEqual(results['refused'].message, 'The provider refused the key (HTTP 401).')
        self.assertEqual(results['needs_key'].message, 'The provider needs an API key (HTTP 403).')
        self.assertEqual(results['server_error'].message, 'The provider answered HTTP 500 instead of a model list.')
        self.assertEqual(results['unreachable'].message, 'Could not reach the provider.')
        self.assertEqual(results['env_missing'].message, "OPENAI_API_KEY is not set in the dashboard's environment.")

    def test_phrase_defaults_to_none(self):
        """MNT-76: ModelListResult keeps working with the old three arguments; phrase is last and defaults to None."""
        result = backend_module.ModelListResult(True, (), 'x')
        self.assertIsNone(result.phrase)

    def test_plural_phrase_follows_the_count(self):
        """MNT-76: the found-count message is one plural message, so the count decides the form."""
        results = self.fixed_results()
        self.assertEqual(results['one'].phrase.params.get('count'), 1)
        self.assertEqual(results['two'].phrase.params.get('count'), 2)

    def test_env_name_is_a_param_not_spliced(self):
        """MNT-76: the missing-variable message keeps the variable name as a named param."""
        phrase = self.fixed_results()['env_missing'].phrase
        self.assertIn('OPENAI_API_KEY', phrase.params.values())
        self.assertNotIn('OPENAI_API_KEY', str(phrase.msg))

    def test_key_value_never_reaches_a_phrase(self):
        """MNT-76: the key value from the environment is in neither the message nor the phrase params."""
        for result in self.fixed_results().values():
            self.assertNotIn(FAKE_VALUE, result.message + repr(result.phrase))


LITELLM = {'gpt-x': {'input_cost_per_token': 2.5e-06, 'output_cost_per_token': 1e-05}, 'no-prices': {'max_tokens': 4096}}


def lookup(model, handler=None, provider='openai'):
    prices.clear_cache()
    handler = handler or (lambda r: httpx.Response(200, json=LITELLM))
    return run(suggest_prices(provider, None, model, httpx.MockTransport(handler), lambda: 1_700_000_000.0))


class PricePhraseTests(PseudoMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(prices.clear_cache)

    def test_not_found_flag_on_every_not_found_path(self):
        """MNT-76: not_found is True for an empty model, a model missing from the list, a model with no prices and a bare entry."""
        for result in (lookup(''), lookup('missing-model'), lookup('no-prices'),
                       prices._result({}, ('input_cost_per_token', 'output_cost_per_token', 'cache_read_input_token_cost'), 'LiteLLM', '')):
            self.assertTrue(result.not_found)
            self.assertFalse(result.found)
            self.assertEqual(result.message, NOT_FOUND)
            self.assertIsInstance(result.phrase, Phrase)
            self.assertEqual(str(result.phrase), NOT_FOUND)
            self.assertTrue(self.pseudo.render(result.phrase).startswith('[!! '))

    def test_other_outcomes_are_not_not_found(self):
        """MNT-76: a hit and a download failure leave not_found False; the failure still carries a phrase equal to its message."""
        hit = lookup('gpt-x')
        self.assertEqual((hit.found, hit.not_found, hit.message), (True, False, ''))
        self.assertIsNone(hit.phrase)
        failed = lookup('gpt-x', lambda r: httpx.Response(500))
        self.assertEqual((failed.found, failed.not_found), (False, False))
        self.assertEqual(failed.message, 'The price list could not be downloaded.')
        self.assertEqual(str(failed.phrase), failed.message)
        unreadable = lookup('gpt-x', lambda r: httpx.Response(200, content=b'not json'))
        self.assertEqual((unreadable.not_found, unreadable.message), (False, 'The price list could not be read.'))
        self.assertEqual(str(unreadable.phrase), unreadable.message)

    def test_openrouter_not_found_sets_flag(self):
        """MNT-76: the OpenRouter source also sets not_found, with the OpenRouter source name kept."""
        prices.clear_cache()
        result = run(suggest_prices('compatible', 'https://openrouter.ai/api/v1', 'vendor/none', httpx.MockTransport(lambda r: httpx.Response(200, json={'data': []})),
                                    lambda: 1_700_000_000.0))
        self.assertEqual((result.source, result.not_found, result.message), ('OpenRouter', True, NOT_FOUND))

    def test_price_result_old_construction_still_works(self):
        """MNT-76: PriceResult built with the old seven fields defaults phrase to None and not_found to False; LITELLM_URL stays exported."""
        result = PriceResult(False, None, None, None, 'LiteLLM', '', 'x')
        self.assertEqual((result.phrase, result.not_found), (None, False))
        self.assertTrue(LITELLM_URL.startswith('https://'))
        old_not_found = PriceResult(False, None, None, None, 'LiteLLM', '2026-10-12', NOT_FOUND)
        self.assertTrue(old_not_found.not_found)

    def test_dashboard_uses_the_flag_not_the_english_text(self):
        """MNT-76: the Backend tab decides on result.not_found, not on comparing English text, and renders the phrase."""
        source = inspect.getsource(dashboard)
        self.assertIn('result.not_found', source)
        self.assertNotIn('result.message == NOT_FOUND', source)
        self.assertNotIn("startswith('No price found')", source)
        self.assertIn('t.render(result.phrase)', source)
