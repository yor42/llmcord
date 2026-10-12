"""Dashboard edge: per-client locale and Quasar language (FEAT-43, D39). NiceGUI is imported lazily."""
from __future__ import annotations

import copy
import typing
from urllib.parse import urlsplit

from . import Translator
from .negotiate import negotiate, parse_accept_language

COOKIE = 'llmcord_lang'
COOKIE_MAX_AGE = 365 * 86400


def request_locale(cookie=None, account_locale=None, accept_language=None, available=None) -> str:
    from .catalog import available_locales
    return negotiate([cookie, account_locale, *parse_accept_language(accept_language)], available or available_locales())


def locale_for_request(request, session=None) -> str:
    """lang query > cookie > account locale > Accept-Language > en; values that name no available locale are skipped."""
    from .catalog import available_locales
    account = ((session or {}).get('user') or {}).get('locale')
    query = request.query_params.get('lang')
    return negotiate([query, request.cookies.get(COOKIE), account, *parse_accept_language(request.headers.get('accept-language'))], available_locales())


def request_t(request, session=None) -> Translator:
    return Translator(locale_for_request(request, session))


def client_t(client) -> Translator:
    return Translator(getattr(client, 'llmcord_locale', 'en'))


def safe_next(value) -> str:
    """A same-site absolute path or '/': no scheme, host, backslash or control character."""
    if not isinstance(value, str) or not value.startswith('/') or value.startswith('//') or '\\' in value:
        return '/'
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        return '/'
    parts = urlsplit(value)
    return value if not parts.scheme and not parts.netloc else '/'


def quasar_language(locale) -> str:
    from babel.core import get_global
    from nicegui.language import Language
    listed = typing.get_args(Language)
    if locale == 'pseudo':
        return 'en-US'
    tag = str(locale).replace('_', '-')
    if tag in listed:
        return tag
    lang = tag.split('-')[0]
    likely = get_global('likely_subtags').get(lang)
    region = likely.split('_')[-1] if likely else ''
    if region and f'{lang}-{region}' in listed:
        return f'{lang}-{region}'
    if lang in listed:
        return lang
    return next((v for v in listed if v.startswith(lang + '-')), 'en-US')


def bind_client(client, locale) -> None:
    client.llmcord_locale = locale
    page = copy.copy(client.page)
    page.language = quasar_language(locale)
    client.page = page


def current_locale() -> str:
    from nicegui.slot import Slot
    stack = Slot.get_stack()
    if not stack:
        return 'en'
    try:
        return getattr(stack[-1].parent.client, 'llmcord_locale', 'en')
    except RuntimeError:
        return 'en'


def current() -> Translator:
    return Translator(current_locale())
