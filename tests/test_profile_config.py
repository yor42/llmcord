import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmcord_core import config
from llmcord_core.config import load_settings

LITERAL = 'sk-abc123'


def mapping(**over):
    base = {'provider': 'openai', 'model': 'gpt-x', 'context_tokens': 8000}
    base.update(over)
    return base


def write_yaml(test, text):
    folder = tempfile.TemporaryDirectory()
    test.addCleanup(folder.cleanup)
    path = Path(folder.name) / 'cfg.yaml'
    path.write_text(text, encoding='utf-8')
    return path


YAML_OK = """
models:
  dialogue: main
  profiles:
    main:
      provider: compatible
      model: m
      context_tokens: 4000
      base_url: http://localhost:11434/v1
"""


def yaml_with(extra):
    return YAML_OK + ''.join(f'      {line}\n' for line in extra)


def profile(provider='openai', base_url=None, api_key_env=None):
    return config.ModelProfile(provider=provider, model='m', context_tokens=1000, supports_images=False,
                               api_key_env=api_key_env, base_url=base_url)


# Item 1: ModelProfile.source
class SourceFieldTests(unittest.TestCase):
    def test_source_defaults_to_config_and_is_last_field(self):
        self.assertEqual(profile().source, 'config')
        self.assertEqual(list(config.ModelProfile.__dataclass_fields__)[-1], 'source')


# Item 2: profile_from_mapping
class ProfileFromMappingTests(unittest.TestCase):
    def build(self, value=None, **kw):
        return config.profile_from_mapping('p', mapping() if value is None else value, **kw)

    def test_minimal_mapping_uses_defaults(self):
        p = self.build()
        self.assertEqual((p.provider, p.model, p.context_tokens), ('openai', 'gpt-x', 8000))
        self.assertEqual((p.supports_images, p.api_key_env, p.base_url, p.timeout_seconds, p.max_retries, p.billing_tier),
                         (False, None, None, 120.0, 1, 'paid'))
        self.assertEqual(p.source, 'config')

    def test_full_mapping_round_trips(self):
        p = self.build(mapping(
            provider='compatible', base_url='https://example.com/v1', api_key_env='X_API_KEY', supports_images=True,
            reasoning_effort='low', structured_outputs=True, stream_usage=False, billing_tier='free',
            input_cost_per_million=1, output_cost_per_million=2.5, cached_input_cost_per_million=0,
            timeout_seconds=30, max_retries=0), source='dashboard')
        self.assertEqual(p.source, 'dashboard')
        self.assertEqual((p.base_url, p.api_key_env, p.reasoning_effort, p.billing_tier), ('https://example.com/v1', 'X_API_KEY', 'low', 'free'))
        self.assertTrue(p.supports_images and p.structured_outputs)
        self.assertFalse(p.stream_usage)
        self.assertEqual((p.input_cost_per_million, p.output_cost_per_million, p.cached_input_cost_per_million), (1, 2.5, 0))
        self.assertEqual((p.timeout_seconds, p.max_retries), (30, 0))

    def test_missing_key_env_is_not_checked_here(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.build(mapping(api_key_env='NOPE_API_KEY')).api_key_env, 'NOPE_API_KEY')

    def assertRejects(self, value, pattern, **kw):
        with self.assertRaisesRegex(ValueError, pattern):
            self.build(value, **kw)

    def test_existing_messages_kept(self):
        self.assertRejects(mapping(billing_tier='gold'), 'Invalid billing_tier or stream_usage in p')
        self.assertRejects(mapping(stream_usage='yes'), 'Invalid billing_tier or stream_usage in p')
        self.assertRejects(mapping(input_cost_per_million=-1), 'Invalid token price in p')
        self.assertRejects(mapping(output_cost_per_million=float('nan')), 'Invalid token price in p')
        self.assertRejects(mapping(timeout_seconds=0), 'timeout_seconds in p must be a finite number greater than 0')
        self.assertRejects(mapping(timeout_seconds=True), 'timeout_seconds in p must be a finite number greater than 0')
        self.assertRejects(mapping(max_retries=-1), 'max_retries in p must be a whole number of at least 0')
        self.assertRejects(mapping(max_retries=1.5), 'max_retries in p must be a whole number of at least 0')
        self.assertRejects(mapping(provider='other'), 'Unsupported provider in p')
        self.assertRejects(mapping(provider='compatible'), 'base_url is required for p')
        self.assertRejects(mapping(reasoning_effort='low'), 'Invalid compatible reasoning_effort in p')
        self.assertRejects(mapping(provider='compatible', base_url='http://h/v1', reasoning_effort='max'),
                           'Invalid compatible reasoning_effort in p')

    def test_model_must_be_nonempty_string_up_to_200(self):
        for bad in ('', None, 5, ['x']):
            with self.subTest(bad=bad):
                self.assertRejects(mapping(model=bad), 'model', source='dashboard')
        self.assertRejects(mapping(model='m' * 201), 'model', source='dashboard')
        self.assertEqual(self.build(mapping(model='m' * 200), source='dashboard').model, 'm' * 200)

    def test_context_tokens_must_be_positive_int_not_bool(self):
        for bad in (0, -5, True, False, 1.5, '8000', 8000.0, None):
            with self.subTest(bad=bad):
                self.assertRejects(mapping(context_tokens=bad), 'context_tokens', source='dashboard')
        self.assertEqual(self.build(mapping(context_tokens=1), source='dashboard').context_tokens, 1)

    def test_base_url_must_be_http_or_https_with_hostname(self):
        for bad in ('ftp://example.com', 'example.com/v1', 'http://', 'https:///v1', 'file:///etc/passwd', 'javascript:alert(1)'):
            with self.subTest(bad=bad):
                self.assertRejects(mapping(provider='compatible', base_url=bad), 'base_url', source='dashboard')
        for good in ('http://localhost:11434/v1', 'https://example.com/v1'):
            self.assertEqual(self.build(mapping(provider='compatible', base_url=good), source='dashboard').base_url, good)

    def test_unknown_key_rejected_for_dashboard_only_and_named(self):
        value = mapping(surprise=1)
        with self.assertRaisesRegex(ValueError, 'surprise'):
            self.build(value, source='dashboard')
        self.assertEqual(self.build(value).model, 'gpt-x')
        self.assertEqual(self.build(value, source='config').model, 'gpt-x')

    def test_known_keys_pass_for_dashboard(self):
        self.assertEqual(self.build(mapping(supports_images=True, api_key_env='OPENAI_API_KEY'), source='dashboard').source, 'dashboard')


# Item 2: load_settings still rejects what it did and uses the shared validation
class LoadSettingsStillStrictTests(unittest.TestCase):
    def load(self, extra=(), env=None):
        path = write_yaml(self, yaml_with(extra))
        with patch.dict(os.environ, {'DISCORD_BOT_TOKEN': 't', **(env or {})}, clear=True):
            return load_settings(path)

    def test_valid_yaml_loads_with_config_source(self):
        settings = self.load()
        self.assertEqual(settings.profiles['main'].source, 'config')

    def test_rejections_unchanged(self):
        cases = [
            (['billing_tier: gold'], 'Invalid billing_tier or stream_usage in main'),
            (['input_cost_per_million: -1'], 'Invalid token price in main'),
            (['timeout_seconds: 0'], 'timeout_seconds in main must be a finite number greater than 0'),
            (['max_retries: -1'], 'max_retries in main must be a whole number of at least 0'),
            (['reasoning_effort: bogus'], 'Invalid compatible reasoning_effort in main'),
        ]
        for extra, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.load(extra)

    def test_missing_token_and_unsupported_provider_and_missing_key_still_rejected(self):
        path = write_yaml(self, yaml_with([]))
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'Discord token'):
            load_settings(path)
        path = write_yaml(self, YAML_OK.replace('provider: compatible', 'provider: nope'))
        with patch.dict(os.environ, {'DISCORD_BOT_TOKEN': 't'}, clear=True), self.assertRaisesRegex(ValueError, 'Unsupported provider in main'):
            load_settings(path)
        with self.assertRaisesRegex(ValueError, 'API key environment variable is missing for main'):
            self.load(['api_key_env: MAIN_API_KEY'])
        self.assertEqual(self.load(['api_key_env: MAIN_API_KEY'], env={'MAIN_API_KEY': 'x'}).profiles['main'].api_key_env, 'MAIN_API_KEY')

    def test_missing_key_ignored_for_unassigned_profile(self):
        text = yaml_with([]) + """    spare:
      provider: openai
      model: m
      context_tokens: 1000
      api_key_env: SPARE_API_KEY
"""
        path = write_yaml(self, text)
        with patch.dict(os.environ, {'DISCORD_BOT_TOKEN': 't'}, clear=True):
            self.assertIn('spare', load_settings(path).profiles)

    def test_yaml_is_trusted_unknown_keys_pinning_and_allowlist_not_applied(self):
        settings = self.load(['mystery: 1', 'api_key_env: lowercase_key', 'base_url: http://evil.example/v1'],
                             env={'lowercase_key': 'x'})
        self.assertEqual(settings.profiles['main'].base_url, 'http://evil.example/v1')

    def test_yaml_stays_lenient_on_context_tokens_model_base_url_and_key_name(self):
        self.assertEqual(self.load(['context_tokens: "8000"']).profiles['main'].context_tokens, 8000)
        self.assertEqual(self.load(['context_tokens: 8000.0']).profiles['main'].context_tokens, 8000)
        for bad in ('true', '0', '-3'):
            with self.subTest(context_tokens=bad), self.assertRaises(ValueError):
                self.load([f'context_tokens: {bad}'])
        long_model = 'm' * 300
        self.assertEqual(self.load([f'model: {long_model}']).profiles['main'].model, long_model)
        self.assertEqual(self.load(['base_url: ftp://x/v1']).profiles['main'].base_url, 'ftp://x/v1')

    def test_lenient_split_in_profile_from_mapping(self):
        base = mapping(provider='compatible', base_url='http://h/v1')
        for source in ('config', 'dashboard'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                config.profile_from_mapping('p', {**base, 'context_tokens': True}, source=source)
            with self.subTest(source=source, floor=True), self.assertRaises(ValueError):
                config.profile_from_mapping('p', {**base, 'context_tokens': 0}, source=source)
        self.assertEqual(config.profile_from_mapping('p', {**base, 'context_tokens': '8000'}).context_tokens, 8000)
        self.assertEqual(config.profile_from_mapping('p', {**base, 'context_tokens': 8000.0}).context_tokens, 8000)
        self.assertEqual(config.profile_from_mapping('p', {**base, 'model': 'm' * 300}).model, 'm' * 300)
        self.assertEqual(config.profile_from_mapping('p', {**base, 'base_url': 'ftp://x/v1'}).base_url, 'ftp://x/v1')
        self.assertEqual(config.profile_from_mapping('p', {**base, 'api_key_env': 'lower'}).api_key_env, 'lower')


# Item 3: load_profiles
class LoadProfilesTests(unittest.TestCase):
    def test_reads_models_without_token_or_keys(self):
        text = """
models:
  dialogue: a
  director: b
  profiles:
    a:
      provider: openai
      model: m
      context_tokens: 1000
      api_key_env: A_API_KEY
    b:
      provider: compatible
      model: m2
      context_tokens: 2000
      base_url: http://localhost:1/v1
"""
        path = write_yaml(self, text)
        with patch.dict(os.environ, {}, clear=True):
            profiles, roles = config.load_profiles(path)
        self.assertEqual(set(profiles), {'a', 'b'})
        self.assertEqual(profiles['b'].model, 'm2')
        self.assertEqual(profiles['a'].source, 'config')
        self.assertEqual(roles, {'dialogue': 'a', 'director': 'b', 'memory': 'a'})

    def test_roles_fall_back_to_dialogue(self):
        path = write_yaml(self, YAML_OK)
        profiles, roles = config.load_profiles(path)
        self.assertEqual(roles, {'dialogue': 'main', 'director': 'main', 'memory': 'main'})

    def test_missing_file_gives_empty(self):
        self.assertEqual(config.load_profiles(Path(tempfile.gettempdir()) / 'llmcord-no-such-config.yaml'), ({}, {}))

    def test_invalid_profile_still_rejected(self):
        path = write_yaml(self, yaml_with(['timeout_seconds: -1']))
        with self.assertRaisesRegex(ValueError, 'timeout_seconds in main'):
            config.load_profiles(path)

    def test_no_models_section_is_empty(self):
        path = write_yaml(self, 'limits: {}\n')
        profiles, _roles = config.load_profiles(path)
        self.assertEqual(profiles, {})


# Item 4: profile names
class ProfileNameTests(unittest.TestCase):
    def test_regex_matches_spec(self):
        self.assertEqual(config.PROFILE_NAME_RE.pattern, r'^[a-z0-9][a-z0-9._/-]{0,63}$')

    def test_valid_names(self):
        for name in ('a', '0', 'gemini-flash', 'my.profile_1', 'org/model-x', 'a' * 64):
            with self.subTest(name=name):
                config.validate_profile_name(name)

    def test_invalid_names(self):
        for name in ('', 'A', 'Upper', '-lead', '.lead', '_lead', '/lead', 'has space', 'a' * 65, 'ünï', 'a\n', 'a$b', 'a:b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                config.validate_profile_name(name)


# Item 5: key references
class KeyRefTests(unittest.TestCase):
    MESSAGE = 'API key must be a ${NAME} reference to a variable ending in _API_KEY'

    def test_empty_means_no_key(self):
        for empty in ('', None):
            self.assertIsNone(config.parse_key_ref(empty))

    def test_valid_refs(self):
        self.assertEqual(config.parse_key_ref('${OPENAI_API_KEY}'), 'OPENAI_API_KEY')
        self.assertEqual(config.parse_key_ref('  ${X_API_KEY}\n'), 'X_API_KEY')
        self.assertEqual(config.parse_key_ref('${A_API_KEY}'), 'A_API_KEY')
        self.assertEqual(config.parse_key_ref('${' + 'A' * 53 + '_API_KEY}'), 'A' * 53 + '_API_KEY')

    def test_invalid_refs_raise_exact_message_without_echo(self):
        bad = [LITERAL, 'OPENAI_API_KEY', '${DISCORD_BOT_TOKEN}', '${openai_api_key}', ' ${X_API_KEY} trailing',
               '${X_API_KEY}${Y_API_KEY}', '$X_API_KEY', '${_API_KEY}', '${1X_API_KEY}', '${X_API_KEY', '${}',
               '${X-Y_API_KEY}', 'sk-proj-SECRETVALUE9999', '${' + 'A' * 70 + '_API_KEY}']
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError) as ctx:
                config.parse_key_ref(text)
            self.assertEqual(str(ctx.exception), self.MESSAGE)
            self.assertNotIn(text.strip(), str(ctx.exception))

    def test_literal_never_in_message(self):
        with self.assertRaises(ValueError) as ctx:
            config.parse_key_ref(LITERAL)
        self.assertNotIn(LITERAL, str(ctx.exception))
        self.assertNotIn('abc123', str(ctx.exception))

    def test_key_ref_regex_and_format(self):
        self.assertEqual(config.KEY_REF_RE.pattern, r'^[A-Z][A-Z0-9_]{0,60}_API_KEY$')
        self.assertEqual(config.format_key_ref('OPENAI_API_KEY'), '${OPENAI_API_KEY}')
        self.assertEqual(config.parse_key_ref(config.format_key_ref('Z9_API_KEY')), 'Z9_API_KEY')


# Items 6 and 7: built-in pins and LLMCORD_KEY_HOSTS
OPENAI = ('https', 'api.openai.com', 443)
GEMINI = ('https', 'generativelanguage.googleapis.com', 443)


def hosts(value=None):
    return config.key_hosts_from_env({} if value is None else {'LLMCORD_KEY_HOSTS': value})


class KeyHostsTests(unittest.TestCase):
    def test_builtin_constant(self):
        self.assertEqual(config.BUILTIN_KEY_HOSTS, {
            'OPENAI_API_KEY': ('https://api.openai.com',),
            'ANTHROPIC_API_KEY': ('https://api.anthropic.com',),
            'GEMINI_API_KEY': ('https://generativelanguage.googleapis.com',),
            'GOOGLE_API_KEY': ('https://generativelanguage.googleapis.com',),
        })

    def test_unset_and_empty_give_builtins_only(self):
        for value in (None, '', '  '):
            with self.subTest(value=value):
                result = hosts(value)
                self.assertEqual(set(result), {'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY', 'GOOGLE_API_KEY'})
                self.assertEqual(result['OPENAI_API_KEY'], frozenset({OPENAI}))
                self.assertEqual(result['ANTHROPIC_API_KEY'], frozenset({('https', 'api.anthropic.com', 443)}))
                self.assertEqual(result['GEMINI_API_KEY'], frozenset({GEMINI}))
                self.assertEqual(result['GOOGLE_API_KEY'], frozenset({GEMINI}))
                self.assertIsInstance(result['OPENAI_API_KEY'], frozenset)

    def test_bare_host_means_https_with_default_port(self):
        self.assertEqual(hosts('MY_API_KEY=Llm.Example.COM')['MY_API_KEY'], frozenset({('https', 'llm.example.com', 443)}))

    def test_ports_and_schemes(self):
        result = hosts('A_API_KEY=example.com:8443|http://localhost:11434|http://plain.example|https://s.example:443')
        self.assertEqual(result['A_API_KEY'], frozenset({
            ('https', 'example.com', 8443), ('http', 'localhost', 11434),
            ('http', 'plain.example', 80), ('https', 's.example', 443)}))

    def test_multiple_names_and_whitespace(self):
        result = hosts(' A_API_KEY = a.example | b.example , B_API_KEY=c.example ')
        self.assertEqual(result['A_API_KEY'], frozenset({('https', 'a.example', 443), ('https', 'b.example', 443)}))
        self.assertEqual(result['B_API_KEY'], frozenset({('https', 'c.example', 443)}))

    def test_merges_with_builtins(self):
        result = hosts('OPENAI_API_KEY=proxy.example')
        self.assertEqual(result['OPENAI_API_KEY'], frozenset({OPENAI, ('https', 'proxy.example', 443)}))
        self.assertEqual(result['GEMINI_API_KEY'], frozenset({GEMINI}))

    def test_malformed_values_raise_naming_variable(self):
        bad = ['A_API_KEY=example.com/path', 'A_API_KEY=https://example.com/v1', 'A_API_KEY=user@example.com',
               'A_API_KEY=https://user:pw@example.com', 'A_API_KEY=example.com?x=1', 'A_API_KEY=example.com#f',
               'lower_api_key=example.com', 'DISCORD_BOT_TOKEN=example.com', 'A_API_KEY=', 'A_API_KEY=a.example|',
               'A_API_KEY=|a.example', 'A_API_KEY', '=example.com', 'A_API_KEY=ftp://example.com',
               'A_API_KEY=example.com:abc', 'A_API_KEY=example.com:0', 'A_API_KEY=example.com:99999', 'A_API_KEY=a b.example',
               'A_API_KEY=example.com,garbage']
        for value in bad:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'LLMCORD_KEY_HOSTS'):
                hosts(value)

    def test_error_does_not_echo_long_or_secret_values(self):
        with self.assertRaises(ValueError) as ctx:
            hosts('A_API_KEY=' + 'x' * 5000 + '/p')
        self.assertLess(len(str(ctx.exception)), 300)

    def test_default_environment_is_process_environment(self):
        with patch.dict(os.environ, {'LLMCORD_KEY_HOSTS': 'Q_API_KEY=q.example'}):
            self.assertIn('Q_API_KEY', config.key_hosts_from_env())


# Item 8: profile_origin
class ProfileOriginTests(unittest.TestCase):
    def test_provider_defaults(self):
        self.assertEqual(config.profile_origin(profile('openai')), OPENAI)
        self.assertEqual(config.profile_origin(profile('anthropic')), ('https', 'api.anthropic.com', 443))

    def test_base_url_wins_and_is_normalized(self):
        self.assertEqual(config.profile_origin(profile('compatible', 'http://LocalHost:11434/v1')), ('http', 'localhost', 11434))
        self.assertEqual(config.profile_origin(profile('compatible', 'https://Example.com/v1/x')), ('https', 'example.com', 443))
        self.assertEqual(config.profile_origin(profile('compatible', 'http://example.com')), ('http', 'example.com', 80))
        self.assertEqual(config.profile_origin(profile('openai', 'https://proxy.example/v1')), ('https', 'proxy.example', 443))

    def test_compatible_without_base_url_raises(self):
        with self.assertRaises(ValueError):
            config.profile_origin(profile('compatible'))


# Item 9: check_key_pin
class CheckKeyPinTests(unittest.TestCase):
    def check(self, p, value=None):
        return config.check_key_pin(p, hosts(value))

    def test_openai_default_host_ok(self):
        self.assertIsNone(self.check(profile('openai', api_key_env='OPENAI_API_KEY')))
        self.assertIsNone(self.check(profile('anthropic', api_key_env='ANTHROPIC_API_KEY')))

    def test_evil_host_refused_with_host_not_key(self):
        p = profile('openai', 'https://evil.example/v1', 'OPENAI_API_KEY')
        with self.assertRaises(ValueError) as ctx:
            self.check(p)
        message = str(ctx.exception)
        self.assertEqual(message, 'OPENAI_API_KEY may only be sent to its pinned hosts; evil.example is not one of them. '
                                  'Add it to LLMCORD_KEY_HOSTS to allow it.')
        self.assertNotIn('sk-', message)

    def test_gemini_compatible_endpoint_ok(self):
        p = profile('compatible', 'https://generativelanguage.googleapis.com/v1beta/openai/', 'GEMINI_API_KEY')
        self.assertIsNone(self.check(p))
        p = profile('compatible', 'https://generativelanguage.googleapis.com/v1beta/openai/', 'GOOGLE_API_KEY')
        self.assertIsNone(self.check(p))

    def test_gemini_key_cannot_go_to_openai(self):
        with self.assertRaises(ValueError):
            self.check(profile('openai', api_key_env='GEMINI_API_KEY'))

    def test_no_key_allows_any_host_and_scheme(self):
        self.assertIsNone(self.check(profile('compatible', 'http://localhost:11434/v1')))
        self.assertIsNone(self.check(profile('compatible', 'http://evil.example/v1', None)))
        self.assertIsNone(self.check(profile('openai')))

    def test_port_mismatch_refused(self):
        with self.assertRaises(ValueError):
            self.check(profile('compatible', 'https://api.openai.com:8443/v1', 'OPENAI_API_KEY'))

    def test_scheme_mismatch_refused(self):
        with self.assertRaises(ValueError):
            self.check(profile('compatible', 'http://api.openai.com/v1', 'OPENAI_API_KEY'))
        with self.assertRaises(ValueError):
            self.check(profile('compatible', 'http://api.openai.com:443/v1', 'OPENAI_API_KEY'))

    def test_host_comparison_is_case_insensitive(self):
        self.assertIsNone(self.check(profile('compatible', 'https://API.OpenAI.com/v1', 'OPENAI_API_KEY')))

    def test_lookalike_hosts_refused(self):
        for url in ('https://api.openai.com.evil.example/v1', 'https://evilapi.openai.com/v1', 'https://api.openai.com@evil.example/v1'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.check(profile('compatible', url, 'OPENAI_API_KEY'))

    def test_env_added_origin_accepted(self):
        p = profile('compatible', 'http://llm.internal:8080/v1', 'LOCAL_API_KEY')
        with self.assertRaises(ValueError):
            self.check(p)
        self.assertIsNone(self.check(p, 'LOCAL_API_KEY=http://llm.internal:8080'))
        self.assertIsNone(self.check(profile('compatible', 'https://proxy.example/v1', 'OPENAI_API_KEY'), 'OPENAI_API_KEY=proxy.example'))

    def test_key_name_outside_allowlist_pattern_refused(self):
        for name in ('DISCORD_BOT_TOKEN', 'openai_api_key', 'PATH'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.check(profile('openai', api_key_env=name))

    def test_applies_regardless_of_source(self):
        p = config.ModelProfile(provider='compatible', model='m', context_tokens=1, supports_images=False,
                                api_key_env='OPENAI_API_KEY', base_url='https://evil.example/v1', source='config')
        with self.assertRaises(ValueError):
            self.check(p)


# Opus review follow-ups (A-F)
SECRET = 'sk-secret'


def anthropic_profile(base_url, key):
    return config.ModelProfile(provider='anthropic', model='m', context_tokens=1000, supports_images=False,
                               api_key_env=key, base_url=base_url, source='config')


def compatible_profile(base_url, key='OPENAI_API_KEY'):
    return config.ModelProfile(provider='compatible', model='m', context_tokens=1000, supports_images=False,
                               api_key_env=key, base_url=base_url, source='config')


def dash(**over):
    return config.profile_from_mapping('p', mapping(**over), source='dashboard')


class AnthropicOriginTests(unittest.TestCase):
    def test_origin_ignores_base_url(self):
        for url in (None, 'https://api.openai.com/v1', 'http://evil.example'):
            with self.subTest(url=url):
                self.assertEqual(config.profile_origin(anthropic_profile(url, None)), ('https', 'api.anthropic.com', 443))

    def test_openai_key_with_anthropic_provider_refused(self):
        with self.assertRaises(ValueError):
            config.check_key_pin(anthropic_profile('https://api.openai.com/v1', 'OPENAI_API_KEY'), hosts())

    def test_anthropic_key_with_anthropic_provider_passes(self):
        self.assertIsNone(config.check_key_pin(anthropic_profile('https://api.openai.com/v1', 'ANTHROPIC_API_KEY'), hosts()))

    def test_dashboard_rejects_base_url_on_anthropic(self):
        with self.assertRaisesRegex(ValueError, 'base_url'):
            dash(provider='anthropic', base_url='https://api.anthropic.com')


class PortZeroTests(unittest.TestCase):
    def test_dashboard_rejects_port_zero(self):
        with self.assertRaisesRegex(ValueError, 'base_url'):
            dash(provider='compatible', base_url='https://api.openai.com:0/v1')

    def test_check_key_pin_raises_for_port_zero(self):
        with self.assertRaises(ValueError):
            config.check_key_pin(compatible_profile('https://api.openai.com:0/v1'), hosts())


class NoEchoTests(unittest.TestCase):
    URLS = ('https://sk-secret\uff20example.com/v1', 'https://sk-secret\u2100example.com/v1')

    def assertClean(self, call):
        with self.assertRaises(ValueError) as ctx:
            call()
        error = ctx.exception
        self.assertNotIn(SECRET, str(error))
        for chained in (error.__cause__, error.__context__):
            self.assertTrue(chained is None or SECRET not in str(chained))
        return error

    def test_profile_from_mapping_gives_fixed_base_url_message(self):
        for url in self.URLS:
            with self.subTest(url=url):
                error = self.assertClean(lambda: dash(provider='compatible', base_url=url))
                self.assertIn('base_url', str(error))

    def test_profile_origin_and_check_key_pin_do_not_echo(self):
        for url in self.URLS:
            p = compatible_profile(url)
            with self.subTest(url=url):
                self.assertClean(lambda: config.profile_origin(p))
                self.assertClean(lambda: config.check_key_pin(p, hosts()))


class DashboardHardeningTests(unittest.TestCase):
    def test_api_key_env_must_match_key_ref_pattern(self):
        for bad in ('DISCORD_BOT_TOKEN', 'openai_api_key', 'PATH', '', 5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                dash(api_key_env=bad)
        self.assertEqual(dash(api_key_env='OPENAI_API_KEY').api_key_env, 'OPENAI_API_KEY')

    def test_base_url_userinfo_whitespace_control_rejected(self):
        for bad in ('https://u:p@api.openai.com/v1', 'https://x@api.openai.com/v1', 'https://example.com/ v1',
                    ' https://example.com/v1', 'https://example.com/v1\n', 'https://exa\tmple.com/v1', 'https://example.com/\x00'):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'base_url'):
                dash(provider='compatible', base_url=bad)

    def test_source_must_be_config_or_dashboard(self):
        for bad in ('Dashboard', 'web', '', None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                config.profile_from_mapping('p', mapping(), source=bad)

    def test_trailing_dot_host_refused_by_pin(self):
        with self.assertRaises(ValueError):
            config.check_key_pin(compatible_profile('https://api.openai.com./v1'), hosts())


if __name__ == '__main__':
    unittest.main()
