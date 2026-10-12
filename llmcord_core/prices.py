"""FEAT-42 (D38): suggested model prices for the Backend editor. Public price lists only, GET only, no API key; nothing at runtime looks prices up."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

from .i18n import Phrase, terms

LITELLM_URL = 'https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json'
PRICE_MAX_BYTES = 10 * 1024 * 1024
OPENROUTER_URL = 'https://openrouter.ai/api/v1/models'
PRICE_DEADLINE_SECONDS = 30
PRICE_TIMEOUT_SECONDS = 15
PRICE_CACHE_SECONDS = 24 * 3600
NOT_FOUND = terms.PRICE_NOT_FOUND


@dataclass(frozen=True)
class PriceResult:
    found: bool
    input: float | None
    output: float | None
    cached_input: float | None
    source: str
    fetched: str
    message: str
    phrase: Phrase | None = None
    not_found: bool = False

    def __post_init__(self):
        if not self.not_found and not self.found and self.message == NOT_FOUND:
            object.__setattr__(self, "not_found", True)


class _PriceFailure(Exception):
    def __init__(self, phrase: Phrase):
        super().__init__(str(phrase))
        self.phrase = phrase


def _failed(source, date, phrase, not_found=False):
    return PriceResult(False, None, None, None, source, date, str(phrase), phrase, not_found)


def _not_found(source, date):
    return _failed(source, date, Phrase(NOT_FOUND), True)


_cache: dict[str, tuple[float, str, dict]] = {}
_locks: 'weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]' = weakref.WeakKeyDictionary()


def clear_cache():
    _cache.clear()


def _lock():
    loop = asyncio.get_running_loop()
    if loop not in _locks:
        _locks[loop] = asyncio.Lock()
    return _locks[loop]


async def _download(url, transport):
    try:
        async with httpx.AsyncClient(transport=transport, follow_redirects=False, timeout=PRICE_TIMEOUT_SECONDS) as client:
            async with client.stream('GET', url, headers={'Accept': 'application/json'}) as response:
                if response.status_code != 200:
                    raise _PriceFailure(Phrase(terms.PRICE_DOWNLOAD))
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body += chunk
                    if len(body) > PRICE_MAX_BYTES:
                        raise _PriceFailure(Phrase(terms.PRICE_TOO_LARGE))
    except _PriceFailure:
        raise
    except (httpx.TimeoutException, asyncio.TimeoutError):
        raise _PriceFailure(Phrase(terms.PRICE_TIMEOUT)) from None
    except httpx.HTTPError:
        raise _PriceFailure(Phrase(terms.PRICE_DOWNLOAD)) from None
    return bytes(body)


def _index(url, raw):
    try:
        data = json.loads(raw)
    except ValueError:
        raise _PriceFailure(Phrase(terms.PRICE_UNREADABLE)) from None
    if url == LITELLM_URL:
        if not isinstance(data, dict):
            raise _PriceFailure(Phrase(terms.PRICE_UNREADABLE))
        return {key: value for key, value in data.items() if key != 'sample_spec' and isinstance(value, dict)}
    items = data.get('data') if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise _PriceFailure(Phrase(terms.PRICE_UNREADABLE))
    return {item['id']: item for item in items if isinstance(item, dict) and isinstance(item.get('id'), str)}


async def _load(url, transport, now):
    """The parsed price list for `url`, cached for 24 hours; concurrent callers share one download and failures are not cached."""
    async with _lock():
        hit = _cache.get(url)
        if hit and now() - hit[0] < PRICE_CACHE_SECONDS:
            return hit[1], hit[2]
        stamp = now()
        table = await asyncio.to_thread(_index, url, await _download(url, transport))
        date = datetime.fromtimestamp(stamp, timezone.utc).strftime('%Y-%m-%d')
        _cache[url] = (stamp, date, table)
        return date, table


def _per_million(value):
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return None
    return round(value * 1_000_000, 6)


def _result(entry, fields, source, date):
    prices = [_per_million(entry.get(field)) for field in fields]
    if prices[0] is None and prices[1] is None:
        return _not_found(source, date)
    return PriceResult(True, *prices, source, date, '')


def _litellm_family(provider, base_url):
    host = (urlparse(base_url or '').hostname or '').lower()
    if provider in ('anthropic', 'openai'):
        return provider
    return 'gemini' if host == 'generativelanguage.googleapis.com' else None


async def suggest_prices(provider, base_url, model, transport=None, now=time.time) -> PriceResult:
    """Look up suggested per-million prices for `model` in a public price list; never raises and never sends a key."""
    source, date = 'LiteLLM', ''
    try:
        model = (model or '').strip().removeprefix('models/')
        if not model:
            return _not_found(source, date)
        base = (base_url or '').strip().rstrip('/')
        openrouter = provider != 'anthropic' and (urlparse(base).hostname or '').lower() == 'openrouter.ai'
        if openrouter:
            source, url, fields = 'OpenRouter', OPENROUTER_URL, ('prompt', 'completion', 'input_cache_read')
        else:
            url, fields = LITELLM_URL, ('input_cost_per_token', 'output_cost_per_token', 'cache_read_input_token_cost')
        try:
            date, table = await asyncio.wait_for(_load(url, transport, now), PRICE_DEADLINE_SECONDS)
        except asyncio.TimeoutError:
            return _failed(source, date, Phrase(terms.PRICE_TIMEOUT))
        except _PriceFailure as error:
            return _failed(source, date, error.phrase)
        family = None if openrouter else _litellm_family(provider, base)
        for candidate in [model] if openrouter else ([f'{family}/{model}'] if family else []) + [model]:
            entry = table.get(candidate)
            if entry is None:
                continue
            if openrouter:
                entry = entry.get('pricing') if isinstance(entry.get('pricing'), dict) else {}
            elif family and candidate == model and entry.get('litellm_provider') not in (None, family):
                continue
            result = _result(entry, fields, source, date)
            if result.found:
                return result
        return _not_found(source, date)
    except Exception as error:
        logging.warning('Price lookup failed unexpectedly: %s', type(error).__name__)
        return _failed(source, date, Phrase(terms.PRICE_FAILED))
