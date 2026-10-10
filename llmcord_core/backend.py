"""Pure merge of config.yaml profiles/roles with dashboard-edited database rows (D24)."""
from __future__ import annotations

import contextlib
import asyncio
import contextvars
import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

from .config import ModelProfile, Settings, check_key_pin, key_hosts_from_env, profile_from_mapping, validate_profile_name
from .errors import error_detail, is_timeout, redact

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
