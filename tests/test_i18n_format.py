import unittest

from llmcord_core.i18n import format as fmt
from llmcord_core.i18n.format import MessageFormatError


class FieldsTests(unittest.TestCase):
    """FEAT-43: fields() lists names and rejects anything but bare named fields."""

    def test_names(self):
        self.assertEqual(fmt.fields('hello'), set())
        self.assertEqual(fmt.fields('{a} and {b_2} {a}'), {'a', 'b_2'})
        self.assertEqual(fmt.fields('{{literal}} {n:,} {x:이/가}'), {'n', 'x'})

    def test_rejections(self):
        bad = {
            'unbalanced open': '{a',
            'unbalanced close': 'a}',
            'attribute': '{a.b}',
            'dunder attribute': '{a.__class__}',
            'index': '{a[0]}',
            'key': '{a[x]}',
            'positional empty': '{}',
            'positional number': '{0}',
            'conversion r': '{a!r}',
            'conversion s': '{a!s}',
            'conversion a': '{a!a}',
            'nested spec': '{a:{b}}',
            'nested spec brace': '{a:>{w}}',
            'leading digit': '{1a}',
            'space': '{ a }',
        }
        for label, template in bad.items():
            with self.subTest(label):
                with self.assertRaises(MessageFormatError):
                    fmt.fields(template)
                with self.assertRaises(MessageFormatError):
                    fmt.format_message(template, {'a': 1, 'b': 2, 'w': 3})

    def test_error_is_value_error(self):
        self.assertTrue(issubclass(MessageFormatError, ValueError))


class FormatMessageTests(unittest.TestCase):
    """FEAT-43: formatting with named params, numeric specs and josa specs."""

    def test_plain_and_escaped_braces(self):
        self.assertEqual(fmt.format_message('plain', {}), 'plain')
        self.assertEqual(fmt.format_message('{{x}} {a}', {'a': 1}), '{x} 1')

    def test_extra_params_ignored(self):
        self.assertEqual(fmt.format_message('{a}', {'a': 'x', 'b': 'y'}), 'x')

    def test_missing_param(self):
        with self.assertRaises(MessageFormatError):
            fmt.format_message('{a} {b}', {'a': 1})

    def test_numeric_specs(self):
        self.assertEqual(fmt.format_message('{amount:,}', {'amount': 1234567}), '1,234,567')
        self.assertEqual(fmt.format_message('{v:.2f}', {'v': 3.14159}), '3.14')
        self.assertEqual(fmt.format_message('{v:>5}', {'v': 'ab'}), '   ab')
        self.assertEqual(fmt.format_message('{v:05d}', {'v': 42}), '00042')

    def test_bad_spec(self):
        with self.assertRaises(MessageFormatError):
            fmt.format_message('{v:d}', {'v': 'text'})
        with self.assertRaises(MessageFormatError):
            fmt.format_message('{v:,d}', {'v': 'text'})

    def test_no_attribute_leak_through_value(self):
        with self.assertRaises(MessageFormatError):
            fmt.format_message('{a.__class__}', {'a': 1})

    def test_contract_examples(self):
        f = fmt.format_message
        self.assertEqual(f('{name:이/가} 이겼어요.', {'name': '루나'}), '루나가 이겼어요.')
        self.assertEqual(f('{name:이/가}', {'name': '하늘'}), '하늘이')
        self.assertEqual(f('{p:으로/로}', {'p': '서울'}), '서울로')
        self.assertEqual(f('{p:으로/로}', {'p': '부산'}), '부산으로')
        self.assertEqual(f('{p:으로/로}', {'p': '집'}), '집으로')
        self.assertEqual(f('{n:을/를}', {'n': 10}), '10을')
        self.assertEqual(f('{n:을/를}', {'n': 2}), '2를')
        self.assertEqual(f('{name:을/를}', {'name': 'Luna'}), 'Luna을(를)')

    def test_locale_argument_accepted(self):
        self.assertEqual(fmt.format_message('{a}', {'a': 1}, 'ko'), '1')


class JosaTableTests(unittest.TestCase):
    """FEAT-43: the full particle table in both orders."""

    TABLE = (
        ('이', '가', '이(가)'), ('은', '는', '은(는)'), ('을', '를', '을(를)'), ('과', '와', '과(와)'),
        ('아', '야', '아(야)'), ('이나', '나', '(이)나'), ('이랑', '랑', '(이)랑'),
        ('이에요', '예요', '(이)에요'), ('이라', '라', '(이)라'), ('으로', '로', '(으)로'),
    )

    def test_table(self):
        for a, b, combined in self.TABLE:
            for spec in (f'{a}/{b}', f'{b}/{a}'):
                with self.subTest(spec=spec):
                    self.assertTrue(fmt.is_josa_spec(spec))
                    self.assertEqual(fmt.JOSA[spec], (a, b, combined))
                    self.assertEqual(fmt.josa('책', spec), a)
                    self.assertEqual(fmt.josa('Luna', spec), combined)
                    self.assertEqual(fmt.josa('', spec), combined)

    def test_vowel_forms(self):
        for a, b, _ in self.TABLE:
            for spec in (f'{a}/{b}', f'{b}/{a}'):
                with self.subTest(spec=spec):
                    self.assertEqual(fmt.josa('나무', spec), b)

    def test_euro_rieul(self):
        for spec in ('으로/로', '로/으로'):
            self.assertEqual(fmt.josa('서울', spec), '로')
            self.assertEqual(fmt.josa('집', spec), '으로')
            self.assertEqual(fmt.josa('나무', spec), '로')

    def test_not_josa(self):
        for spec in ('', ',', '이/', '가/가', 'a/b', '이가'):
            self.assertFalse(fmt.is_josa_spec(spec))

    def test_rendered_with_all_specs(self):
        for spec, (a, _, _) in fmt.JOSA.items():
            with self.subTest(spec=spec):
                self.assertEqual(fmt.format_message('{w:' + spec + '}', {'w': '밥'}), '밥' + a)


class FinalSoundTests(unittest.TestCase):
    """FEAT-43: final_sound decides batchim from Hangul, jamo, digits."""

    def test_syllables(self):
        self.assertEqual(fmt.final_sound('나무'), (False, False))
        self.assertEqual(fmt.final_sound('집'), (True, False))
        self.assertEqual(fmt.final_sound('서울'), (True, True))
        self.assertEqual(fmt.final_sound('하늘'), (True, True))
        self.assertEqual(fmt.final_sound('가'), (False, False))
        self.assertEqual(fmt.final_sound('힣'), (True, False))

    def test_compatibility_jamo(self):
        self.assertEqual(fmt.final_sound('ㄱ'), (True, False))
        self.assertEqual(fmt.final_sound('ㄹ'), (True, True))
        self.assertEqual(fmt.final_sound('ㅎ'), (True, False))
        self.assertEqual(fmt.final_sound('ㅏ'), (False, False))
        self.assertEqual(fmt.final_sound('ㅣ'), (False, False))

    def test_single_digits(self):
        want = {
            '0': (True, False), '1': (True, True), '2': (False, False), '3': (True, False),
            '4': (False, False), '5': (False, False), '6': (True, False), '7': (True, True),
            '8': (True, True), '9': (False, False),
        }
        for digit, sound in want.items():
            with self.subTest(digit=digit):
                self.assertEqual(fmt.final_sound(digit), sound)
                self.assertEqual(fmt.final_sound('12' + digit if digit != '0' else '12'),
                                 sound if digit != '0' else (False, False))

    def test_trailing_zeros(self):
        cases = {
            '10': (True, False),
            '100': (True, False),
            '1000': (True, False),
            '20000': (True, False),
            '100000000': (True, False),
            '1' + '0' * 11: (True, False),
            '1' + '0' * 12: (False, False),
            '1' + '0' * 15: (False, False),
            '0': (True, False),
            '000': (True, False),
        }
        for word, sound in cases.items():
            with self.subTest(word=word):
                self.assertEqual(fmt.final_sound(word), sound)

    def test_too_large_unknown(self):
        self.assertIsNone(fmt.final_sound('1' + '0' * 16))

    def test_comma_numbers(self):
        self.assertEqual(fmt.final_sound('1,000'), (True, False))
        self.assertEqual(fmt.final_sound('1,234'), (False, False))
        self.assertEqual(fmt.final_sound('20,000'), (True, False))

    def test_decimals_read_last_digit(self):
        self.assertEqual(fmt.final_sound('3.10'), (True, False))
        self.assertEqual(fmt.final_sound('3.5'), (False, False))
        self.assertEqual(fmt.final_sound('3.1'), (True, True))
        self.assertEqual(fmt.final_sound('3.20'), (True, False))

    def test_closers_and_whitespace_stripped(self):
        self.assertEqual(fmt.final_sound('집)'), (True, False))
        self.assertEqual(fmt.final_sound('「나무」'), (False, False))
        self.assertEqual(fmt.final_sound('"서울"  '), (True, True))
        self.assertEqual(fmt.final_sound('[10]'), (True, False))
        self.assertEqual(fmt.final_sound('《집》'), (True, False))
        self.assertEqual(fmt.final_sound('집’'), (True, False))

    def test_unknown(self):
        for word in ('Luna', '', '   ', ')', '😀', 'abc!'):
            with self.subTest(word=word):
                self.assertIsNone(fmt.final_sound(word))

    def test_josa_unknown_gives_combined(self):
        self.assertEqual(fmt.josa('Luna', '이/가'), '이(가)')
        self.assertEqual(fmt.josa('Luna', '으로/로'), '(으)로')


class NumberAndListTests(unittest.TestCase):
    """FEAT-43: Babel-backed number and list formatting."""

    def test_babel_locale(self):
        self.assertEqual(fmt.babel_locale('ko'), 'ko')
        self.assertEqual(fmt.babel_locale('pseudo'), 'en')
        self.assertEqual(fmt.babel_locale('!!'), 'en')
        self.assertEqual(fmt.babel_locale('zh_Hant_TW'), 'zh_Hant_TW')

    def test_format_number(self):
        self.assertEqual(fmt.format_number(1234567, 'en'), '1,234,567')
        self.assertEqual(fmt.format_number(1234567, 'ko'), '1,234,567')
        self.assertEqual(fmt.format_number(1234567, 'pseudo'), '1,234,567')
        self.assertEqual(fmt.format_number(1234567, '!!'), '1,234,567')

    def test_format_list(self):
        self.assertEqual(fmt.format_list(['A', 'B', 'C'], 'en'), 'A, B, and C')
        self.assertEqual(fmt.format_list(['A', 'B', 'C'], 'ko'), 'A, B 및 C')
        self.assertEqual(fmt.format_list(['A', 'B'], 'en'), 'A and B')
        self.assertEqual(fmt.format_list(['A'], 'en'), 'A')
        self.assertEqual(fmt.format_list([], 'en'), '')
        self.assertEqual(fmt.format_list([], 'ko'), '')
        self.assertEqual(fmt.format_list([1, 2], 'en'), '1 and 2')


if __name__ == '__main__':
    unittest.main()
