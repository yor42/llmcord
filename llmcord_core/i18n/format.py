"""Message formatting: str.format with named fields, Korean particles (josa), Babel number/list helpers."""
from __future__ import annotations

import re
import string

_NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_CLOSERS = ')]}>"\'”’」』》〉'

# spec -> (form after a final consonant, form after a vowel, combined)
JOSA: dict[str, tuple[str, str, str]] = {}
for _a, _b, _c in (
    ('이', '가', '이(가)'), ('은', '는', '은(는)'), ('을', '를', '을(를)'), ('과', '와', '과(와)'),
    ('아', '야', '아(야)'), ('이나', '나', '(이)나'), ('이랑', '랑', '(이)랑'),
    ('이에요', '예요', '(이)에요'), ('이라', '라', '(이)라'), ('으로', '로', '(으)로'),
):
    JOSA[f'{_a}/{_b}'] = JOSA[f'{_b}/{_a}'] = (_a, _b, _c)

_DIGITS = {
    '0': (True, False), '1': (True, True), '2': (False, False), '3': (True, False), '4': (False, False),
    '5': (False, False), '6': (True, False), '7': (True, True), '8': (True, True), '9': (False, False),
}


class MessageFormatError(ValueError):
    pass


def _parse(template: str) -> list[tuple[str, str, str]]:
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as error:
        raise MessageFormatError(f'bad message syntax: {error}') from None
    out = []
    for _, name, spec, conv in parsed:
        if name is None:
            continue
        if not _NAME.match(name):
            raise MessageFormatError(f'field name must be a bare identifier: {name!r}')
        if conv:
            raise MessageFormatError(f'conversion not allowed in field {name!r}')
        spec = spec or ''
        if '{' in spec or '}' in spec:
            raise MessageFormatError(f'nested format spec in field {name!r}')
        out.append((name, spec, conv))
    return out


def fields(template: str) -> set[str]:
    return {name for name, _, _ in _parse(template)}


def is_josa_spec(spec: str) -> bool:
    return spec in JOSA


def final_sound(word: str) -> tuple[bool, bool] | None:
    word = word.rstrip()
    word = word.rstrip(_CLOSERS).rstrip()
    if not word:
        return None
    ch = word[-1]
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        jong = (code - 0xAC00) % 28
        return jong != 0, jong == 8
    if 0x3131 <= code <= 0x314E:
        return True, ch == 'ㄹ'
    if 0x314F <= code <= 0x3163:
        return False, False
    if ch not in _DIGITS:
        return None
    match = re.search(r'[0-9,]+$', word)
    run = match.group()
    if match.start() > 0 and word[match.start() - 1] == '.':
        return _DIGITS[ch]
    digits = run.replace(',', '').lstrip('0')
    if not digits:
        return True, False
    k = len(digits) - len(digits.rstrip('0'))
    if k == 0:
        return _DIGITS[ch]
    if k <= 3 or 4 <= k <= 11:
        return True, False
    if k <= 15:
        return False, False
    return None


def josa(word: str, spec: str) -> str:
    a, b, combined = JOSA[spec]
    sound = final_sound(word)
    if sound is None:
        return combined
    final, rieul = sound
    if a == '으로':
        return '로' if not final or rieul else '으로'
    return a if final else b


class SafeFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        try:
            return kwargs[key]
        except KeyError:
            raise MessageFormatError(f'missing parameter {key!r}') from None

    def format_field(self, value, format_spec):
        if is_josa_spec(format_spec):
            text = str(value)
            return text + josa(text, format_spec)
        try:
            return format(value, format_spec)
        except (ValueError, TypeError) as error:
            raise MessageFormatError(f'bad format spec {format_spec!r}: {error}') from None


_FORMATTER = SafeFormatter()


def format_message(template: str, params, locale=None) -> str:
    _parse(template)
    try:
        return _FORMATTER.vformat(template, (), dict(params or {}))
    except MessageFormatError:
        raise
    except (ValueError, KeyError, IndexError, AttributeError, TypeError) as error:
        raise MessageFormatError(f'cannot format message: {error}') from None


def babel_locale(locale) -> str:
    try:
        from babel import Locale
        return str(Locale.parse(str(locale)))
    except Exception:
        return 'en'


def format_number(value, locale) -> str:
    try:
        from babel.numbers import format_decimal
        return format_decimal(value, locale=babel_locale(locale))
    except Exception:
        return f'{value:,}'


def format_list(items, locale) -> str:
    items = [str(i) for i in items]
    if not items:
        return ''
    from babel.lists import format_list as babel_format_list
    return babel_format_list(items, locale=babel_locale(locale))
