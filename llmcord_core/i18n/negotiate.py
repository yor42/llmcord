"""Locale tag normalization and negotiation. Babel is imported lazily."""
from __future__ import annotations

import re

DEFAULT_LOCALE = 'en'
PSEUDO_LOCALE = 'pseudo'

_LANG = re.compile(r'[A-Za-z]{2,3}')
_SCRIPT = re.compile(r'[A-Za-z]{4}')
_REGION = re.compile(r'[A-Za-z]{2}|[0-9]{3}')
_Q = re.compile(r'(?:0(?:\.\d{0,3})?|1(?:\.0{0,3})?)')


def normalize(tag) -> str | None:
    tag = getattr(tag, 'value', tag)
    if not isinstance(tag, str):
        return None
    tag = tag.strip().replace('-', '_')
    if not tag or tag == '*':
        return None
    if tag.lower() == PSEUDO_LOCALE:
        return PSEUDO_LOCALE
    parts = tag.split('_')
    if not _LANG.fullmatch(parts[0]):
        return None
    out = [parts[0].lower()]
    rest = parts[1:]
    if rest and _SCRIPT.fullmatch(rest[0]):
        out.append(rest.pop(0).title())
    if rest and _REGION.fullmatch(rest[0]):
        out.append(rest.pop(0).upper())
    if rest:
        return None
    return '_'.join(out)


def candidates(tag) -> list[str]:
    norm = normalize(tag)
    if norm is None:
        return []
    if norm == PSEUDO_LOCALE:
        return [norm]
    parts = norm.split('_')
    lang, rest = parts[0], parts[1:]
    script = next((p for p in rest if len(p) == 4), None)
    region = next((p for p in rest if len(p) != 4), None)
    out = [norm]
    if region:
        out.append(f'{lang}_{region}')
    if script:
        out.append(f'{lang}_{script}')
    out.append(lang)
    return list(dict.fromkeys(out))


def negotiate(preferred, available, default: str = DEFAULT_LOCALE) -> str:
    have = set(available)
    for tag in preferred or ():
        if tag is None:
            continue
        for cand in candidates(tag):
            if cand in have:
                return cand
    return default


def parse_accept_language(header) -> list[str]:
    if not header:
        return []
    found: list[tuple[float, str]] = []
    for entry in header[:1000].split(',')[:20]:
        tag, *params = (p.strip() for p in entry.split(';'))
        q = 1.0
        bad = False
        for param in params:
            key, _, value = param.partition('=')
            if key.strip().lower() == 'q':
                value = value.strip()
                if not _Q.fullmatch(value):
                    bad = True
                    break
                q = float(value)
        if bad or q <= 0 or tag == '*' or normalize(tag) is None:
            continue
        found.append((q, tag))
    found.sort(key=lambda item: -item[0])
    return [tag for _, tag in found]


def display_name(locale: str) -> str:
    if locale == PSEUDO_LOCALE:
        return 'Pseudo-locale'
    try:
        from babel import Locale
        name = Locale.parse(locale).get_display_name(locale)
        return (name[0].upper() + name[1:]) if name else locale
    except Exception:
        return locale
