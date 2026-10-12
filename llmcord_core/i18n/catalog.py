"""gettext .po catalogs: discovery of available locales and cached loading (Babel is imported lazily)."""
from __future__ import annotations

import functools
import gettext
import logging
import os
import re
from pathlib import Path

from .negotiate import DEFAULT_LOCALE, PSEUDO_LOCALE, normalize

log = logging.getLogger('llmcord_core.i18n')

LOCALES_DIR = Path(__file__).resolve().parent.parent / 'locales'
DOMAINS = ('bot', 'commands', 'web', 'errors', 'common')
ENGLISH_PLURAL = 'n != 1'
_NPLURALS = re.compile(r'nplurals\s*=\s*(\d+)')
_PLURAL = re.compile(r'plural\s*=\s*([^;]*)')


def _english(n: int) -> int:
    return 0 if n == 1 else 1


class Catalog:
    """Translated messages of one locale, keyed by (msgctxt or None, English singular msgid)."""

    def __init__(self, locale: str = DEFAULT_LOCALE, messages=None, plural=_english, num_plurals: int = 2):
        self.locale = locale
        self.num_plurals = num_plurals
        self._messages = messages if messages is not None else {}
        self._plural = plural

    def get(self, context, msgid):
        entry = self._messages.get((context, msgid))
        if isinstance(entry, tuple):
            return entry[0]
        return entry

    def get_plural(self, context, msgid, n):
        entry = self._messages.get((context, msgid))
        if entry is None:
            return None
        if not isinstance(entry, tuple):
            return entry
        try:
            index = int(self._plural(n))
        except Exception:
            index = _english(n)
        return entry[max(0, min(index, len(entry) - 1))]

    def __len__(self):
        return len(self._messages)


def available_locales(root=None) -> tuple[str, ...]:
    base = Path(root) if root is not None else LOCALES_DIR
    found = []
    if base.is_dir():
        for path in sorted(base.iterdir()):
            name = path.name
            if name in ('en', 'templates', PSEUDO_LOCALE) or normalize(name) != name or not path.is_dir():
                continue
            if any((path / 'LC_MESSAGES' / f'{d}.po').is_file() for d in DOMAINS):
                found.append(name)
    wanted = os.environ.get('LLMCORD_LOCALES', '').strip()
    if wanted:
        keep = {normalize(x) for x in wanted.split(',') if x.strip()}
        found = [x for x in found if x in keep]
    result = ['en', *found]
    if os.environ.get('LLMCORD_I18N_PSEUDO') == '1':
        result.append(PSEUDO_LOCALE)
    return tuple(result)


def _plural_rule(header, locale):
    """Return (callable, nplurals) from a Plural-Forms header, else Babel's CLDR default."""
    if header:
        count, expr = _NPLURALS.search(header), _PLURAL.search(header)
        if count and expr:
            try:
                return gettext.c2py(expr.group(1).strip()), int(count.group(1))
            except Exception:
                log.warning('bad Plural-Forms in locale %s', locale)
    from babel.core import Locale
    from babel.messages.plurals import get_plural
    num, expr = get_plural(Locale.parse(locale))
    return gettext.c2py(expr), num


def load_catalog(locale, root=None) -> Catalog:
    if locale != PSEUDO_LOCALE and (not isinstance(locale, str) or normalize(locale) != locale):
        return Catalog(DEFAULT_LOCALE)
    base = Path(root) if root is not None else LOCALES_DIR
    if locale not in (DEFAULT_LOCALE, PSEUDO_LOCALE) and not (base / locale / 'LC_MESSAGES').is_dir():
        return Catalog(DEFAULT_LOCALE)
    return _load_catalog(locale, root)


@functools.lru_cache(maxsize=None)
def _load_catalog(locale, root=None) -> Catalog:
    base = Path(root) if root is not None else LOCALES_DIR
    directory = base / locale / 'LC_MESSAGES'
    if locale in (DEFAULT_LOCALE, PSEUDO_LOCALE) or not directory.is_dir():
        return Catalog(locale)
    from babel.messages.pofile import read_po
    messages: dict = {}
    header = None
    for domain in DOMAINS:
        path = directory / f'{domain}.po'
        if not path.is_file():
            continue
        try:
            with open(path, 'rb') as fh:
                po = read_po(fh, locale=locale, abort_invalid=True)
        except Exception as error:
            log.error('cannot read catalog %s: %s', path, type(error).__name__)
            continue
        header = header or dict(po.mime_headers).get('Plural-Forms')
        for message in po:
            if not message.id or message.fuzzy:
                continue
            singular = message.id[0] if isinstance(message.id, (tuple, list)) else message.id
            if isinstance(message.string, (tuple, list)):
                value = tuple(message.string)
                if not all(value):
                    continue
            else:
                value = message.string
                if not value:
                    continue
            key = (message.context or None, singular)
            if key in messages:
                if messages[key] != value:
                    log.warning('conflicting translation in %s: locale=%s msgctxt=%r msgid=%r', domain, locale, key[0], key[1])
                continue
            messages[key] = value
    try:
        plural, num = _plural_rule(header, locale)
    except Exception:
        plural, num = _english, 2
    return Catalog(locale, messages, plural, num)


def clear_cache() -> None:
    _load_catalog.cache_clear()
