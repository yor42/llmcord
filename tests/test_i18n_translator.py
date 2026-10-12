"""FEAT-43: the public i18n API (Translator, Msg, UserError, pseudo-locale, lazy Babel import)."""
from __future__ import annotations

import copy
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_i18n_catalog import write_domain

from llmcord_core import i18n
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


class PickledConflict(UserError):
    """Module level so pickle can find it."""


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
            ((('One box', '{count} boxes'),), {'string': ('상자 {cnt}개',)}),
        ])
        clear_cache()
        i18n._warned.clear()
        self.addCleanup(i18n._warned.clear)
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

    def test_distinct_bad_keys_each_warn_once(self):
        with self.assertLogs(LOGGER, 'WARNING') as logs:
            self.ko('Show {total}', total=1)
            self.ko.n('One box', '{count} boxes', 3)
            self.ko('Show {total}', total=2)
        self.assertEqual(len(logs.records), 2)

    def test_missing_param_still_raises_in_translated_locale(self):
        with self.assertLogs(LOGGER, 'WARNING'), self.assertRaises(MessageFormatError):
            self.ko('Hello {name}')

    def test_bad_plural_translation_falls_back_to_english_and_warns_once(self):
        with self.assertLogs(LOGGER, 'WARNING') as logs:
            self.assertEqual(self.ko.n('One box', '{count} boxes', 3), '3 boxes')
        self.assertEqual(len(logs.records), 1)
        self.assertNotIn('3', logs.output[0])
        with self.assertNoLogs(LOGGER, 'WARNING'):
            self.assertEqual(self.ko.n('One box', '{count} boxes', 4), '4 boxes')

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

    def test_error_with_unusable_params_returns_bare_msgid(self):
        """FEAT-43: an error UserError tolerated at raise time must not raise again while it is rendered."""
        bad_spec = UserError(E_('test.coins', 'You have {n:,} coins'), n='lots')
        for tr in (self.en, self.ko):
            with self.subTest(locale=tr.locale):
                with self.assertLogs(LOGGER, 'WARNING') as logs:
                    self.assertEqual(tr.error(UserError(self.GREETING)), 'Hello {name}')
                self.assertIn('test.greeting', logs.output[-1])
                with self.assertLogs(LOGGER, 'WARNING') as logs:
                    self.assertEqual(tr.error(bad_spec), 'You have {n:,} coins')
                self.assertNotIn('lots', '\n'.join(logs.output))

    def test_user_error_survives_copy_and_pickle(self):
        """FEAT-43: UserError rebuilds from code and params, not from its English text."""
        err = UserError(self.GREETING, name='Luna')
        for clone in (copy.copy(err), copy.deepcopy(err), pickle.loads(pickle.dumps(err))):
            self.assertIsInstance(clone, UserError)
            self.assertEqual((clone.code, clone.params, str(clone)), (err.code, err.params, 'Hello Luna'))

    def test_error_code_validation(self):
        self.assertEqual(E_('a.b_c.d1', 'x'), ErrorCode('a.b_c.d1', 'x'))
        for bad in ('nodot', 'A.b', '.a.b', 'a.', 'a..b', '1a.b', 'a.B', 'a b.c', ''):
            with self.assertRaises(ValueError, msg=bad):
                E_(bad, 'x')


class EnglishTranslatorTests(unittest.TestCase):
    """FEAT-43: english() is the shared Phase 1 translator."""

    def test_english_is_en_translator(self):
        tr = i18n.english()
        self.assertIsInstance(tr, Translator)
        self.assertEqual(tr.locale, 'en')

    def test_english_is_shared(self):
        self.assertIs(i18n.english(), i18n.english())

    def test_english_renders_params(self):
        self.assertEqual(i18n.english()('Hello {name}', name='x'), 'Hello x')


class Conflict(UserError):
    pass


class LegacyUserErrorTests(TranslatorCase):
    """FEAT-43: UserError('plain text') is the legacy form so error classes can subclass it before conversion."""

    def test_plain_text_form(self):
        err = UserError('Plain text')
        self.assertEqual(str(err), 'Plain text')
        self.assertEqual(err.args, ('Plain text',))
        self.assertIsNone(err.code)
        self.assertEqual(err.params, {})
        self.assertIsInstance(err, ValueError)

    def test_plain_text_rejects_params(self):
        with self.assertRaises(TypeError):
            UserError('x', a=1)

    def test_subclass_works_in_both_forms(self):
        for args, kwargs in ((('text',), {}), ((E_('test.hello', 'Hello {name}'),), {'name': 'Luna'})):
            with self.subTest(args=args):
                for catch in (Conflict, ValueError):
                    with self.assertRaises(catch):
                        raise Conflict(*args, **kwargs)
        self.assertEqual(str(Conflict('text')), 'text')
        self.assertEqual(str(Conflict(E_('test.hello', 'Hello {name}'), name='Luna')), 'Hello Luna')

    def test_legacy_form_survives_copy_and_pickle(self):
        for cls in (UserError, PickledConflict):
            err = cls('Plain text')
            clones = [copy.copy(err), copy.deepcopy(err), pickle.loads(pickle.dumps(err))]
            for clone in clones:
                with self.subTest(cls=cls.__name__):
                    self.assertIs(type(clone), cls)
                    self.assertEqual(clone.args, ('Plain text',))
                    self.assertIsNone(clone.code)
                    self.assertEqual(clone.params, {})

    def test_translator_error_returns_plain_text(self):
        for tr in (self.en, self.ko):
            with self.subTest(locale=tr.locale), self.assertNoLogs(LOGGER, 'WARNING'):
                self.assertEqual(tr.error(UserError('Plain text')), 'Plain text')
                self.assertEqual(tr.error(Conflict('Plain')), 'Plain')


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
