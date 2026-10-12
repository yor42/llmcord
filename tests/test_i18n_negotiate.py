import importlib
import subprocess
import sys
import unittest

neg = importlib.import_module('llmcord_core.i18n.negotiate')


class FakeLocale:
    def __init__(self, value):
        self.value = value


class NormalizeTests(unittest.TestCase):
    """FEAT-43: tag normalization accepts only language[-script][-region]."""

    def test_valid_forms(self):
        cases = {
            'ko': 'ko', 'KO': 'ko', 'en-US': 'en_US', 'en_us': 'en_US', 'zh-tw': 'zh_TW',
            'zh-Hant-TW': 'zh_Hant_TW', 'zh_hant': 'zh_Hant', 'es-419': 'es_419', ' ko ': 'ko',
            'fil': 'fil',
        }
        for raw, want in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(neg.normalize(raw), want)

    def test_pseudo_any_case(self):
        self.assertEqual(neg.normalize('PSEUDO'), 'pseudo')
        self.assertEqual(neg.normalize('Pseudo'), 'pseudo')

    def test_object_with_value(self):
        self.assertEqual(neg.normalize(FakeLocale('ja')), 'ja')
        self.assertEqual(neg.normalize(FakeLocale('en-US')), 'en_US')

    def test_invalid_returns_none(self):
        for raw in (None, '', '  ', '*', 'e', 'english', 'k0', 'en-US-posix', 'de-1996-x', 'en-x-private',
                    'en-USA', 'en-41', 'zh-Hant-TW-x', 'ko_KR_', 12, FakeLocale(None), '한국어'):
            with self.subTest(raw=raw):
                self.assertIsNone(neg.normalize(raw))


class CandidatesTests(unittest.TestCase):
    """FEAT-43: lookup order for one tag."""

    def test_orders(self):
        self.assertEqual(neg.candidates('zh-Hant-TW'), ['zh_Hant_TW', 'zh_TW', 'zh_Hant', 'zh'])
        self.assertEqual(neg.candidates('en-US'), ['en_US', 'en'])
        self.assertEqual(neg.candidates('ko'), ['ko'])
        self.assertEqual(neg.candidates('zh-Hant'), ['zh_Hant', 'zh'])

    def test_invalid_is_empty(self):
        self.assertEqual(neg.candidates('*'), [])
        self.assertEqual(neg.candidates(None), [])
        self.assertEqual(neg.candidates('en-x-y'), [])

    def test_no_duplicates(self):
        for tag in ('zh-Hant-TW', 'en-US', 'ko', 'es-419'):
            got = neg.candidates(tag)
            self.assertEqual(len(got), len(set(got)))


class NegotiateTests(unittest.TestCase):
    """FEAT-43: negotiation walks preferred tags in order with per-tag fallback."""

    def test_exact_and_language_fallback(self):
        self.assertEqual(neg.negotiate(['ko'], {'en', 'ko'}), 'ko')
        self.assertEqual(neg.negotiate(['ko-KR'], {'en', 'ko'}), 'ko')
        self.assertEqual(neg.negotiate(['zh-Hant-TW'], ['en', 'zh_TW']), 'zh_TW')

    def test_per_tag_fallback(self):
        self.assertEqual(neg.negotiate(['en-GB', 'ko'], {'en', 'ko'}), 'en')
        self.assertEqual(neg.negotiate(['ja', 'ko'], {'en', 'ko'}), 'ko')

    def test_default_when_nothing_matches(self):
        self.assertEqual(neg.negotiate(['ja'], {'ko'}), 'en')
        self.assertEqual(neg.negotiate(['ja'], {'ko'}, default='ko'), 'ko')
        self.assertEqual(neg.negotiate([], {'ko'}), 'en')

    def test_none_and_invalid_skipped(self):
        self.assertEqual(neg.negotiate([None, '*', 'bad tag', 'ko'], {'ko'}), 'ko')

    def test_accepts_discord_style_objects(self):
        self.assertEqual(neg.negotiate([FakeLocale('ko')], {'en', 'ko'}), 'ko')

    def test_pseudo_only_when_available(self):
        self.assertEqual(neg.negotiate(['pseudo'], {'en', 'pseudo'}), 'pseudo')
        self.assertEqual(neg.negotiate(['pseudo'], {'en'}), 'en')


class ParseAcceptLanguageTests(unittest.TestCase):
    """FEAT-43: RFC 9110 Accept-Language parsing with limits."""

    def test_empty(self):
        self.assertEqual(neg.parse_accept_language(None), [])
        self.assertEqual(neg.parse_accept_language(''), [])

    def test_q_ordering_and_default(self):
        self.assertEqual(neg.parse_accept_language('en;q=0.5, ko, ja;q=0.8'), ['ko', 'ja', 'en'])

    def test_stable_for_equal_q(self):
        self.assertEqual(neg.parse_accept_language('fr, de, es;q=1.0'), ['fr', 'de', 'es'])

    def test_drops_bad_entries(self):
        header = 'ko;q=0, *, en;q=abc, ja;q=1.5, e!, de;q=0.3, en-US;q=0.9'
        self.assertEqual(neg.parse_accept_language(header), ['en-US', 'de'])

    def test_raw_tags_returned(self):
        self.assertEqual(neg.parse_accept_language('en-us;q=0.9, KO'), ['KO', 'en-us'])

    def test_entry_limit(self):
        header = ', '.join(f'a{chr(97 + i)}' for i in range(25))
        got = neg.parse_accept_language(header)
        self.assertEqual(len(got), 20)
        self.assertNotIn('ay', got)

    def test_length_limit(self):
        header = 'ko,' + ' ' * 1000 + 'ja'
        self.assertEqual(neg.parse_accept_language(header), ['ko'])


class DisplayNameTests(unittest.TestCase):
    """FEAT-43: display names come from CLDR in the language itself."""

    def test_known(self):
        self.assertEqual(neg.display_name('en'), 'English')
        self.assertEqual(neg.display_name('ko'), '한국어')
        self.assertEqual(neg.display_name('ja'), '日本語')

    def test_pseudo(self):
        self.assertEqual(neg.display_name('pseudo'), 'Pseudo-locale')

    def test_unknown_returns_code(self):
        self.assertEqual(neg.display_name('!!'), '!!')
        self.assertEqual(neg.display_name('zzzz_ZZ'), 'zzzz_ZZ')


class LazyBabelTests(unittest.TestCase):
    """FEAT-43: importing the i18n package does not import babel."""

    def test_import_does_not_load_babel(self):
        code = (
            'import sys, llmcord_core.i18n, llmcord_core.i18n.negotiate, llmcord_core.i18n.format, '
            'llmcord_core.i18n.catalog\n'
            'sys.exit(1 if "babel" in sys.modules else 0)'
        )
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
