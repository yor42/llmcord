"""Pure merge of config.yaml profiles/roles with dashboard-edited database rows (D24)."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .config import ModelProfile, Settings, profile_from_mapping, validate_profile_name

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
