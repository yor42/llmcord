"""Translation at the edge: gettext catalogs keyed by the English text, errors as codes (FEAT-43, D39)."""
from __future__ import annotations

import functools
import logging
import re
from typing import NamedTuple

from .catalog import DOMAINS, LOCALES_DIR, available_locales, clear_cache, load_catalog
from .format import MessageFormatError, format_message
from .negotiate import DEFAULT_LOCALE, PSEUDO_LOCALE, display_name, negotiate, normalize, parse_accept_language

__all__ = [
    'DEFAULT_LOCALE', 'PSEUDO_LOCALE', 'DOMAINS', 'LOCALES_DIR', 'normalize', 'negotiate', 'parse_accept_language',
    'display_name', 'available_locales', 'load_catalog', 'clear_cache', 'MessageFormatError',
    'Msg', 'N_', 'NP_', 'Phrase', 'ErrorCode', 'E_', 'UserError', 'CodedError', 'Translator', 'pseudo', 'english',
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


class Phrase:
    """Words for the edge to render later: a Msg (from N_/NP_) plus named params. Immutable; str() is the English text.

    A param that is a Msg or a Phrase is rendered in the same locale as the phrase (Translator.render).
    """

    __slots__ = ('_msg', '_params')

    def __init__(self, msg, /, **params):
        object.__setattr__(self, '_msg', msg)
        object.__setattr__(self, '_params', dict(params))

    @property
    def msg(self):
        return self._msg

    @property
    def params(self) -> dict:
        return dict(self._params)

    def __setattr__(self, name, value):
        raise AttributeError('Phrase is immutable')

    def __delattr__(self, name):
        raise AttributeError('Phrase is immutable')

    def _key(self):
        return str(self._msg), getattr(self._msg, 'context', None), self._params

    def __eq__(self, other):
        return isinstance(other, Phrase) and self._key() == other._key()

    def __hash__(self):
        text, context, params = self._key()
        return hash((text, context, tuple(sorted(params))))

    def __repr__(self):
        context = getattr(self._msg, 'context', None)
        head = repr(str(self._msg)) if context is None else f'NP_({context!r}, {str(self._msg)!r})'
        return 'Phrase(' + ''.join([head, *(f', {k}={v!r}' for k, v in self._params.items())]) + ')'

    def __str__(self):
        return english().render(self)

    def __reduce__(self):
        return (_rebuild_phrase, (self._msg, self._params))


def _rebuild_phrase(msg, params):
    return Phrase(msg, **params)


def _coded_text(code, params):
    rendered = {k: english().render(v) if isinstance(v, (Msg, Phrase)) else v for k, v in params.items()}
    try:
        return format_message(code.msgid, rendered)
    except MessageFormatError:
        return code.msgid


def _init_coded(self, code, params):
    if isinstance(code, str):
        if params:
            raise TypeError('params are not allowed with a plain-text message')
        BaseException.__init__(self, str(code))
        self.code = None
        self.params = {}
        return
    BaseException.__init__(self, _coded_text(code, params))
    self.code = code
    self.params = params


def _reduce_coded(self):
    if self.code is None:
        return (_rebuild_user_error, (type(self), str(self.args[0]), {}))
    return (_rebuild_user_error, (type(self), self.code, self.params))


class UserError(ValueError):
    """An error meant for the user, raised as a code; str() is the English message.

    Legacy form: UserError('plain text') has code None and no params, so error classes can
    subclass it before every raise site is converted; Translator.error returns str(err) for it.
    """

    def __init__(self, code, /, **params):
        _init_coded(self, code, params)

    def __reduce__(self):
        return _reduce_coded(self)


class CodedError:
    """Mixin for a coded exception that is not a UserError (an engine's IllegalMove): class X(CodedError, Exception).

    Same shape as UserError: X(code, **params) or the legacy X('text'); Translator.error renders it.
    """

    __slots__ = ()

    def __init__(self, code, /, **params):
        _init_coded(self, code, params)

    def __reduce__(self):
        return _reduce_coded(self)


def _rebuild_user_error(cls, code, params):
    return cls(code, **params)


@functools.cache
def english() -> Translator:
    """The English translator. Phase 1 code that has no audience yet uses it; it never changes language."""
    return Translator(DEFAULT_LOCALE)


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
        if any(isinstance(v, (Msg, Phrase)) for v in params.values()):
            params = {k: self.render(v) if isinstance(v, (Msg, Phrase)) else v for k, v in params.items()}
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

    def render(self, value) -> str:
        """Words for a Phrase (with its params), a Msg (with its context); anything else is str(value), untouched."""
        if isinstance(value, Phrase):
            msg = value.msg
            return self._render(getattr(msg, 'context', None), str(msg), None, None, value.params)
        if isinstance(value, Msg):
            return self(value)
        return str(value)

    def error(self, err) -> str:
        code, params = getattr(err, 'code', None), getattr(err, 'params', None)
        if isinstance(code, ErrorCode) and isinstance(params, dict):
            try:
                return self(code.msgid, **params)
            except MessageFormatError:
                log.warning('bad error params: code=%s', code.code)
                return str(err)
        return str(err)

    def number(self, value) -> str:
        from .format import format_number
        return format_number(value, self.locale)

    def list(self, items) -> str:
        from .format import format_list
        return format_list(items, self.locale)
