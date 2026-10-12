"""Dashboard edge: per-client locale and Quasar language (FEAT-43, D39). NiceGUI is imported lazily."""
from __future__ import annotations

import copy
import typing

from . import Translator
from .negotiate import negotiate, parse_accept_language

COOKIE = 'llmcord_lang'
COOKIE_MAX_AGE = 365 * 86400


def request_locale(cookie=None, account_locale=None, accept_language=None, available=None) -> str:
    from .catalog import available_locales
    return negotiate([cookie, account_locale, *parse_accept_language(accept_language)], available or available_locales())


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
    from nicegui import ui
    try:
        return getattr(ui.context.client, 'llmcord_locale', 'en')
    except RuntimeError:
        return 'en'


def current() -> Translator:
    return Translator(current_locale())
