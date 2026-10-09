"""Pure merge of config.yaml profiles/roles with dashboard-edited database rows (D24)."""
from __future__ import annotations

import contextlib
import contextvars
import logging
from dataclasses import dataclass, replace

from .config import ModelProfile, Settings, key_hosts_from_env, profile_from_mapping, validate_profile_name

ROLES = ('dialogue', 'director', 'memory')


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
