"""Pure merge of config.yaml profiles/roles with dashboard-edited database rows (D24)."""
from __future__ import annotations

import contextlib
import asyncio
import contextvars
import json
import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import quote

import httpx

from .config import LIMIT_DEFAULTS, ModelProfile, Settings, check_key_pin, key_hosts_from_env, profile_from_mapping, validate_profile_name
from .errors import error_detail, is_timeout, redact, register_secret_names

ROLES = ('dialogue', 'director', 'memory')
PROBE_TIMEOUT_SECONDS = 25


@dataclass(frozen=True)
class Backend:
    """`roles` holds only resolvable roles; a role with no valid database or config profile is absent."""
    profiles: dict[str, ModelProfile]
    roles: dict[str, str]
    problems: tuple[str, ...]
    version: int


def effective_backend(config_profiles, config_roles, rows, roles) -> Backend:
    profiles, problems = dict(config_profiles), []
    for row in rows:
        name = row['name']
        try:
            try:
                validate_profile_name(name)
            except ValueError:
                problems.append('A saved profile with an invalid name was skipped.')
                continue
            if not isinstance(row['data'], dict):
                raise ValueError('its saved settings could not be read')
            profiles[name] = profile_from_mapping(name, row['data'], source='dashboard')
        except ValueError as error:
            problems.append(f'Profile {name} was skipped: {error}')
    merged = {}
    for role in ROLES:
        chosen = roles.get(role)
        if chosen is not None and chosen not in profiles:
            problems.append(f'The {role} role names profile {chosen}, which is not available; using the config.yaml setting.')
            chosen = None
        chosen = chosen or config_roles.get(role)
        if chosen in profiles:
            merged[role] = chosen
    return Backend(profiles, merged, tuple(problems), roles.get('version', 0))


def apply_backend(settings: Settings, backend: Backend) -> Settings:
    return replace(settings, profiles=backend.profiles, **{role: backend.roles.get(role, getattr(settings, role)) for role in ROLES})


class StaticSource:
    """A fixed Settings behind the resolver interface (no dashboard profiles, so nothing to pin)."""
    key_hosts: dict = {}

    def __init__(self, settings: Settings):
        self.settings = settings

    def current(self) -> Settings:
        return self.settings

    def pin(self):
        return contextlib.nullcontext()


def settings_source(value):
    return value if hasattr(value, 'current') else StaticSource(value)


class BackendResolver:
    """Resolves the effective Settings from config.yaml plus dashboard rows; `pin()` holds one snapshot for a turn."""

    def __init__(self, store, base_settings: Settings, config_profiles=None, config_roles=None):
        self.store, self.base = store, base_settings
        self.config_profiles = dict(base_settings.profiles if config_profiles is None else config_profiles)
        self.config_roles = dict(config_roles) if config_roles is not None else {role: getattr(base_settings, role) for role in ROLES}
        self.key_hosts = key_hosts_from_env()
        register_secret_names(p.api_key_env for p in self.config_profiles.values())
        self._cached: tuple[int, Settings] | None = None
        self._pinned = contextvars.ContextVar(f'backend_snapshot_{id(self)}', default=None)

    def snapshot(self) -> Settings:
        try:
            version = self.store.model_backend_version()
            if self._cached is not None and self._cached[0] == version:
                return self._cached[1]
            backend = effective_backend(self.config_profiles, self.config_roles, self.store.model_profile_rows(), self.store.model_roles())
        except Exception as error:
            logging.warning('Model backend could not be read from the database: %s', type(error).__name__)
            return self._cached[1] if self._cached else self.base
        for problem in backend.problems:
            logging.warning('Model backend: %s', problem)
        settings = apply_backend(self.base, backend)
        register_secret_names(p.api_key_env for p in settings.profiles.values())
        self._cached = (version, settings)
        return settings

    @contextlib.contextmanager
    def pin(self):
        if self._pinned.get() is not None:
            yield
            return
        token = self._pinned.set(self.snapshot())
        try:
            yield
        finally:
            self._pinned.reset(token)

    def current(self) -> Settings:
        return self._pinned.get() or self.snapshot()


@dataclass(frozen=True)
class ConnectionResult:
    ok: bool
    latency_ms: int | None
    message: str


class _ProbeSource(StaticSource):
    """One-off settings for a connection test: its own key hosts and environment, no pinning context manager needed."""

    def __init__(self, settings: Settings, key_hosts, environ):
        super().__init__(settings)
        self.key_hosts, self.environ = key_hosts, environ


def _failure(message: str, secret: str | None = None) -> ConnectionResult:
    return ConnectionResult(False, None, redact(message, extra_values=[secret], strict=True)[:300])


async def test_profile_connection(name, mapping, key_hosts, environ=os.environ) -> ConnectionResult:
    """Send one tiny request with the unsaved profile `mapping`; never raises and never returns the key value."""
    from .models import ModelGateway, TurnMessage
    gateway, secret = None, None
    try:
        try:
            profile = profile_from_mapping(name, mapping, source='dashboard')
            check_key_pin(profile, key_hosts)
        except ValueError as error:
            return _failure(str(error))
        if profile.api_key_env:
            secret = environ.get(profile.api_key_env)
            if not secret:
                return _failure(f"{profile.api_key_env} is not set in the dashboard's environment.")
        profile = replace(profile, timeout_seconds=min(profile.timeout_seconds, 20), max_retries=0)
        settings = Settings('', None, Path(':memory:'), 0, {name: profile}, name, name, name, {})
        gateway = ModelGateway(_ProbeSource(settings, key_hosts, environ), environment='the dashboard')
        started = time.perf_counter()
        try:
            await asyncio.wait_for(gateway.text('dialogue', 'Reply with the word OK.', [TurnMessage('user', 'OK?')], max_tokens=16), PROBE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            return _failure('The model provider did not respond in time')
        except Exception as error:
            return _failure('The model provider did not respond in time' if is_timeout(error) else error_detail(error, 10000), secret)
        latency = int((time.perf_counter() - started) * 1000)
        return ConnectionResult(True, latency, f'Connected. {profile.model} replied in {latency} ms.')
    except Exception as error:
        logging.warning('Connection test failed unexpectedly: %s', type(error).__name__)
        return ConnectionResult(False, None, 'The test could not run; see the dashboard log.')
    finally:
        if gateway is not None:
            try:
                await gateway.close()
            except Exception as error:
                logging.warning('Connection test client did not close: %s', type(error).__name__)


MODEL_LIST_TIMEOUT_SECONDS = 15
MODEL_LIST_DEADLINE_SECONDS = 20
MODEL_LIST_MAX_BYTES = 8 * 1024 * 1024
MODEL_LIST_MAX_PAGES = 5


@dataclass(frozen=True)
class ModelInfo:
    id: str
    context_tokens: int | None = None
    supports_images: bool | None = None


@dataclass(frozen=True)
class ModelListResult:
    ok: bool
    models: tuple[ModelInfo, ...]
    message: str


def _list_failure(message: str, secret: str | None = None) -> ModelListResult:
    return ModelListResult(False, (), redact(message, extra_values=[secret], strict=True)[:300])


class _ListFailure(Exception):
    pass


async def _get_json(client, url, headers, keyed=True):
    """GET `url` without following redirects and read at most MODEL_LIST_MAX_BYTES; raises _ListFailure with a fixed, key-free message."""
    try:
        async with client.stream('GET', url, headers=headers) as response:
            if response.status_code == 404:
                raise _ListFailure('not_found')
            if response.status_code in (401, 403):
                raise _ListFailure(f'The provider refused the key (HTTP {response.status_code}).' if keyed else f'The provider needs an API key (HTTP {response.status_code}).')
            if response.status_code != 200:
                raise _ListFailure(f'The provider answered HTTP {response.status_code} instead of a model list.')
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body += chunk
                if len(body) > MODEL_LIST_MAX_BYTES:
                    raise _ListFailure('The provider\'s model list is too large to read.')
    except _ListFailure:
        raise
    except (httpx.TimeoutException, asyncio.TimeoutError):
        raise _ListFailure('The provider did not respond in time.') from None
    except httpx.HTTPError:
        raise _ListFailure('Could not reach the provider.') from None
    try:
        return json.loads(bytes(body))
    except ValueError:
        raise _ListFailure('unparseable') from None


def _positive_int(value):
    return value if type(value) is int and value > 0 else None


def _compatible_info(item) -> ModelInfo | None:
    if isinstance(item, str):
        item = {'id': item}
    if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id'].strip():
        return None
    model_id = item['id'].strip().removeprefix('models/')
    modalities = (item.get('architecture') or {}).get('input_modalities') if isinstance(item.get('architecture'), dict) else None
    images = ('image' in modalities) if isinstance(modalities, list) else None
    return ModelInfo(model_id, _positive_int(item.get('context_length')), images)


def _anthropic_info(item) -> ModelInfo | None:
    if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id'].strip():
        return None
    capabilities = item.get('capabilities')
    image = capabilities.get('image_input') if isinstance(capabilities, dict) else None
    images = image.get('supported') if isinstance(image, dict) and type(image.get('supported')) is bool else None
    return ModelInfo(item['id'].strip(), _positive_int(item.get('max_input_tokens')), images)


async def list_profile_models(name, mapping, key_hosts, environ=os.environ, transport=None) -> ModelListResult:
    """List the provider's models for the unsaved profile `mapping` (GET only, no redirects); never raises and never returns the key value."""
    secret = None
    try:
        try:
            profile = profile_from_mapping(name, mapping, source='dashboard')
            check_key_pin(profile, key_hosts)
        except ValueError as error:
            return _list_failure(str(error))
        if profile.api_key_env:
            secret = environ.get(profile.api_key_env)
            if not secret:
                return _list_failure(f"{profile.api_key_env} is not set in the dashboard's environment.")
        base = (profile.base_url or '').rstrip('/')
        if profile.provider == 'anthropic':
            url, headers = 'https://api.anthropic.com/v1/models?limit=1000', {'x-api-key': secret or '', 'anthropic-version': '2023-06-01'}
        elif profile.provider == 'openai':
            url, headers = f'{base or "https://api.openai.com/v1"}/models', {'Authorization': f'Bearer {secret}'} if secret else {}
        else:
            url, headers = f'{base}/models', {'Authorization': f'Bearer {secret}'} if secret else {}
        headers = {**headers, 'Accept-Encoding': 'identity'}
        found: dict[str, ModelInfo] = {}
        async def collect(url):
            async with httpx.AsyncClient(transport=transport, follow_redirects=False, timeout=MODEL_LIST_TIMEOUT_SECONDS) as client:
                for _page in range(MODEL_LIST_MAX_PAGES if profile.provider == 'anthropic' else 1):
                    data = await _get_json(client, url, headers, bool(secret))
                    if profile.provider == 'anthropic':
                        items = data.get('data') if isinstance(data, dict) else None
                        parse = _anthropic_info
                    else:
                        items = data.get('data') if isinstance(data, dict) else data
                        parse = _compatible_info
                    if not isinstance(items, list):
                        raise _ListFailure('unparseable')
                    for item in items:
                        info = parse(item)
                        if info is not None:
                            found.setdefault(info.id, info)
                    last = data.get('last_id') if isinstance(data, dict) else None
                    if profile.provider != 'anthropic' or not (isinstance(data, dict) and data.get('has_more') is True and isinstance(last, str) and last):
                        break
                    url = f'https://api.anthropic.com/v1/models?limit=1000&after_id={quote(last, safe="")}'

        try:
            await asyncio.wait_for(collect(url), MODEL_LIST_DEADLINE_SECONDS)
        except asyncio.TimeoutError:
            return _list_failure('The provider did not respond in time.')
        except _ListFailure as error:
            text = str(error)
            if text in ('not_found', 'unparseable'):
                if profile.provider == 'compatible':
                    return _list_failure("This provider doesn't list its models; type the ID.")
                return _list_failure("The provider's answer was not a model list; type the ID.")
            return _list_failure(text, secret)
        if not found:
            return _list_failure('The provider returned no models; type the ID.')
        return ModelListResult(True, tuple(found[key] for key in sorted(found)), f'{len(found)} model{"" if len(found) == 1 else "s"} found.')
    except Exception as error:
        logging.warning('Model list failed unexpectedly: %s', type(error).__name__)
        return ModelListResult(False, (), 'The model list could not be fetched; see the dashboard log.')


def dashboard_gateway(backend: Backend, limits, store, key_hosts=None, environ=os.environ):
    """A gateway for a model call made by the dashboard (FEAT-32 D): the director profile (else dialogue), key pins checked on every call,
    recorded in model_usage and refused at the spending hard cap, like the bot's own calls. ValueError when no model is set up."""
    from . import budget
    from .models import ModelGateway
    name = backend.roles.get('director') or backend.roles.get('dialogue')
    if name is None or name not in backend.profiles:
        raise ValueError('No model is set up for this bot. Choose one on the Backend tab.')
    settings = Settings('', None, Path(':memory:'), 0, backend.profiles, name, name, name, {**LIMIT_DEFAULTS, **limits})
    source = _ProbeSource(settings, key_hosts_from_env() if key_hosts is None else key_hosts, environ)
    return ModelGateway(source, usage_sink=store.record_model_usage, budget_gate=lambda: budget.state(store), environment='the dashboard')
