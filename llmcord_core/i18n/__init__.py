"""Translation at the edge: gettext catalogs keyed by the English text, errors as codes (FEAT-43, D39)."""
from __future__ import annotations

import logging
import re
from typing import NamedTuple

from .catalog import DOMAINS, LOCALES_DIR, available_locales, clear_cache, load_catalog
from .format import MessageFormatError, format_message
from .negotiate import DEFAULT_LOCALE, PSEUDO_LOCALE, display_name, negotiate, normalize, parse_accept_language

__all__ = [
    'DEFAULT_LOCALE', 'PSEUDO_LOCALE', 'DOMAINS', 'LOCALES_DIR', 'normalize', 'negotiate', 'parse_accept_language',
    'display_name', 'available_locales', 'load_catalog', 'clear_cache', 'MessageFormatError',
    'Msg', 'N_', 'NP_', 'ErrorCode', 'E_', 'UserError', 'Translator', 'pseudo',
]

log = logging.getLogger('llmcord_core.i18n')

CODE_RE = re.compile(r'^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$')
_PROTECT = re.compile(r'\{\{|\}\}|\{[^{}]*\}|<[^<>]*>|`[^`]*`')
_FROM = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
_TO = 'àḅċḋéḟġħíjķḷṁñöṗqŕšṫüṽẁẍýžÀḄĊḊÉḞĠĦÍJĶḶṀÑÖṖQŔŠṪÜṼẀẌÝŽ'
_PSEUDO = str.maketrans(_FROM, _TO)
_warned: set = set()


class Msg(str):
    """English msgid usable as plain text, carrying its gettext context."""

    context: str | None

    def __new__(cls, msgid, context=None):
        self = super().__new__(cls, msgid)
        self.context = context
        return self


def N_(msgid, /) -> Msg:
    return Msg(msgid)


def NP_(context, msgid, /) -> Msg:
    return Msg(msgid, context)


class ErrorCode(NamedTuple):
    code: str
    msgid: str


def E_(code, msgid, /) -> ErrorCode:
    if not CODE_RE.match(code):
        raise ValueError(f'invalid error code: {code!r}')
    return ErrorCode(code, msgid)


class UserError(ValueError):
    """An error meant for the user, raised as a code; str() is the English message."""

    def __init__(self, code, /, **params):
        try:
            text = format_message(code.msgid, params)
        except MessageFormatError:
            text = code.msgid
        super().__init__(text)
        self.code = code
        self.params = params


def pseudo(text: str) -> str:
    out, pos = [], 0
    for match in _PROTECT.finditer(text):
        out.append(text[pos:match.start()].translate(_PSEUDO))
        out.append(match.group())
        pos = match.end()
    out.append(text[pos:].translate(_PSEUDO))
    return '[!! ' + ''.join(out) + ' ' + '~' * max(1, len(text) * 3 // 10) + ' !!]'


class Translator:
    def __init__(self, locale=DEFAULT_LOCALE, *, root=None):
        self.locale = locale
        self._root = root
        self._catalog = None

    def _cat(self):
        if self._catalog is None:
            self._catalog = load_catalog(self.locale, self._root)
        return self._catalog

    def _render(self, context, singular, plural, count, params):
        english = singular if plural is None or count == 1 else plural
        if plural is None:
            found = self._cat().get(context, singular)
        else:
            found = self._cat().get_plural(context, singular, count)
        source = pseudo(english) if self.locale == PSEUDO_LOCALE else english
        if found is not None:
            try:
                return format_message(found, params, self.locale)
            except MessageFormatError:
                key = (self.locale, context, singular)
                if key not in _warned:
                    _warned.add(key)
                    log.warning('bad translation: locale=%s msgctxt=%r msgid=%r', self.locale, context, singular)
        return format_message(source, params, self.locale)

    def __call__(self, msgid, /, **params) -> str:
        return self._render(getattr(msgid, 'context', None), str(msgid), None, None, params)

    def n(self, singular, plural, count, /, **params) -> str:
        params.setdefault('count', count)
        return self._render(getattr(singular, 'context', None), str(singular), plural, count, params)

    def p(self, context, msgid, /, **params) -> str:
        return self._render(context, str(msgid), None, None, params)

    def np(self, context, singular, plural, count, /, **params) -> str:
        params.setdefault('count', count)
        return self._render(context, str(singular), plural, count, params)

    def error(self, err) -> str:
        if isinstance(err, UserError):
            return self(err.code.msgid, **err.params)
        return str(err)

    def number(self, value) -> str:
        from .format import format_number
        return format_number(value, self.locale)

    def list(self, items) -> str:
        from .format import format_list
        return format_list(items, self.locale)
