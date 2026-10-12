"""FEAT-43: the public i18n API (Translator, Msg, UserError, pseudo-locale, lazy Babel import)."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_i18n_catalog import write_domain

from llmcord_core.i18n import (
    E_,
    NP_,
    ErrorCode,
    MessageFormatError,
    N_,
    Msg,
    Translator,
    UserError,
    clear_cache,
)

LOGGER = 'llmcord_core.i18n'
REPO = Path(__file__).resolve().parent.parent
LONG = NP_('menu', 'Open')


class TranslatorCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        write_domain(self.root, 'ko', 'bot', [
            (('Hello {name}',), {'string': '안녕 {name}'}),
            (('Plain',), {'string': '평범'}),
            (('Open',), {'string': '열기', 'context': 'menu'}),
            (('Open',), {'string': '개봉'}),
            ((('One file', '{count} files'),), {'string': ('파일 {count}개',)}),
            ((('One item', '{count} items'),), {'string': ('항목 {count}개',), 'context': 'inv'}),
            (('{name:이/가} won',), {'string': '{name:이/가} 이겼어요'}),
            (('Show {total}',), {'string': '표시 {totl}'}),
            (('Braces {{x}}',), {'string': '중괄호 {{x}}'}),
        ])
        clear_cache()
        self.addCleanup(clear_cache)
        self.ko = Translator('ko', root=self.root)
        self.en = Translator('en', root=self.root)


class EnglishTests(TranslatorCase):
    """FEAT-43: English renders the msgid byte-identically."""

    def test_identity_and_params(self):
        self.assertEqual(self.en('Plain'), 'Plain')
        self.assertEqual(self.en('Hello {name}', name='Luna'), 'Hello Luna')
        self.assertEqual(self.en('Hello {name}', name='Luna', extra=1), 'Hello Luna')
        self.assertEqual(self.en('Tab\t "q" é {{x}}'), 'Tab\t "q" é {x}')
        self.assertEqual(self.en('Total {amount:,}', amount=1234567), 'Total 1,234,567')

    def test_default_locale_is_en(self):
        self.assertEqual(Translator().locale, 'en')

    def test_missing_param_raises(self):
        with self.assertRaises(MessageFormatError):
            self.en('Hello {name}')

    def test_plurals(self):
        self.assertEqual(self.en.n('One file', '{count} files', 1), 'One file')
        self.assertEqual(self.en.n('One file', '{count} files', 3), '3 files')
        self.assertEqual(self.en.n('One file', '{count} files', 0), '0 files')
        self.assertEqual(self.en.np('inv', 'One item', '{count} items', 2), '2 items')

    def test_locale_without_catalog_behaves_as_english(self):
        fr = Translator('fr', root=self.root)
        self.assertEqual(fr('Hello {name}', name='x'), 'Hello x')
        self.assertEqual(fr.n('One file', '{count} files', 2), '2 files')


class KoreanTests(TranslatorCase):
    """FEAT-43: lookup, context, plurals and fallbacks in a translated locale."""

    def test_lookup(self):
        self.assertEqual(self.ko('Plain'), '평범')
        self.assertEqual(self.ko('Hello {name}', name='Luna'), '안녕 Luna')

    def test_josa_in_translation(self):
        self.assertEqual(self.ko('{name:이/가} won', name='루나'), '루나가 이겼어요')

    def test_context_via_p_and_constant(self):
        self.assertEqual(self.ko.p('menu', 'Open'), '열기')
        self.assertEqual(self.ko('Open'), '개봉')
        self.assertEqual(self.ko(LONG), '열기')
        self.assertEqual(self.en(LONG), 'Open')

    def test_plurals(self):
        self.assertEqual(self.ko.n('One file', '{count} files', 1), '파일 1개')
        self.assertEqual(self.ko.n('One file', '{count} files', 7), '파일 7개')
        self.assertEqual(self.ko.np('inv', 'One item', '{count} items', 4), '항목 4개')

    def test_explicit_count_param_wins(self):
        self.assertEqual(self.en.n('One file', '{count} files', 2, count='two'), 'two files')

    def test_missing_translation_falls_back_to_english(self):
        self.assertEqual(self.ko('Untranslated {x}', x=1), 'Untranslated 1')
        self.assertEqual(self.ko.n('One cat', '{count} cats', 2), '2 cats')
        self.assertEqual(self.ko.p('other', 'Open'), 'Open')

    def test_wrong_field_falls_back_and_warns_once_without_values(self):
        with self.assertLogs(LOGGER, 'WARNING') as logs:
            first = self.ko('Show {total}', total='SECRET-VALUE-1')
            second = self.ko('Show {total}', total='SECRET-VALUE-2')
        self.assertEqual((first, second), ('Show SECRET-VALUE-1', 'Show SECRET-VALUE-2'))
        self.assertEqual(len(logs.records), 1)
        self.assertNotIn('SECRET-VALUE', logs.output[0])
        self.assertIn('Show {total}', logs.output[0])
        self.assertIn('ko', logs.output[0])

    def test_doubled_braces(self):
        self.assertEqual(self.ko('Braces {{x}}'), '중괄호 {x}')

    def test_number_and_list(self):
        self.assertEqual(self.ko.number(1234567), '1,234,567')
        self.assertEqual(self.ko.list(['A', 'B', 'C']), 'A, B 및 C')
        self.assertEqual(self.en.list(['A', 'B', 'C']), 'A, B, and C')
        self.assertEqual(self.en.list([]), '')


class PseudoTests(TranslatorCase):
    """FEAT-43: the pseudo-locale wraps English and protects fields, tags and code."""

    def setUp(self):
        super().setUp()
        self.ps = Translator('pseudo', root=self.root)

    def test_wrap(self):
        out = self.ps('Plain')
        self.assertTrue(out.startswith('[!! ') and out.endswith(' !!]'), out)
        self.assertNotIn('Plain', out)
        self.assertFalse(out.isascii())

    def test_ignores_catalog_translations(self):
        self.assertNotIn('평범', self.ps('Plain'))

    def test_preserves_fields_tags_and_code(self):
        out = self.ps('Hi {name} <@123> `code span` {{x}}', name='Luna')
        self.assertIn('Luna', out)
        self.assertIn('<@123>', out)
        self.assertIn('`code span`', out)
        self.assertNotIn('{{', out)
        self.assertNotIn('{name}', out)

    def test_plural_and_formatting(self):
        self.assertIn('3', self.ps.n('One file', '{count} files', 3))
        self.assertNotIn('{count}', self.ps.n('One file', '{count} files', 3))


class ErrorTests(TranslatorCase):
    """FEAT-43: UserError carries code and params, english text, rendered at the edge."""

    GREETING = E_('test.greeting', 'Hello {name}')

    def test_user_error_is_english_value_error(self):
        err = UserError(self.GREETING, name='Luna')
        self.assertIsInstance(err, ValueError)
        self.assertEqual(str(err), 'Hello Luna')
        self.assertEqual(err.args[0], 'Hello Luna')
        self.assertIs(err.code, self.GREETING)
        self.assertEqual(err.params, {'name': 'Luna'})
        with self.assertRaisesRegex(ValueError, 'Hello Luna'):
            raise err

    def test_user_error_with_missing_param_uses_bare_msgid(self):
        self.assertEqual(str(UserError(self.GREETING)), 'Hello {name}')

    def test_translator_error(self):
        err = UserError(E_('test.hello', 'Hello {name}'), name='Luna')
        self.assertEqual(self.ko.error(err), '안녕 Luna')
        self.assertEqual(self.en.error(err), 'Hello Luna')
        self.assertEqual(self.ko.error(KeyError('boom')), "'boom'")
        self.assertEqual(self.ko.error(ValueError('plain')), 'plain')

    def test_error_code_validation(self):
        self.assertEqual(E_('a.b_c.d1', 'x'), ErrorCode('a.b_c.d1', 'x'))
        for bad in ('nodot', 'A.b', '.a.b', 'a.', 'a..b', '1a.b', 'a.B', 'a b.c', ''):
            with self.assertRaises(ValueError, msg=bad):
                E_(bad, 'x')


class MsgTests(unittest.TestCase):
    """FEAT-43: N_/NP_ markers return str subclasses equal to the msgid."""

    def test_msg_is_str(self):
        msg = N_('Hello')
        self.assertIsInstance(msg, str)
        self.assertEqual(msg, 'Hello')
        self.assertEqual(f'{msg}!', 'Hello!')
        self.assertIsNone(msg.context)
        self.assertIsInstance(msg, Msg)

    def test_np_carries_context(self):
        msg = NP_('menu', 'Open')
        self.assertEqual(msg, 'Open')
        self.assertEqual(msg.context, 'menu')


class LazyBabelTests(unittest.TestCase):
    """FEAT-43: importing llmcord_core.i18n must not import Babel."""

    def test_import_does_not_load_babel(self):
        code = (
            'import sys, llmcord_core.i18n as m\n'
            'print(any(k == "babel" or k.startswith("babel.") for k in sys.modules))\n'
            'm.Translator("en")("Plain"); m.available_locales()\n'
            'print(any(k == "babel" or k.startswith("babel.") for k in sys.modules))\n'
        )
        done = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        lines = done.stdout.split()
        self.assertEqual(lines[0], 'False')
        self.assertEqual(lines[1], 'False')


if __name__ == '__main__':
    unittest.main()
