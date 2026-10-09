from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def prompt_provider(provider: str, base_url: str | None = None) -> str:
    """Use Gemini's system-instruction rules for its official compatible endpoint."""
    if provider == 'compatible' and urlparse(base_url or '').hostname == 'generativelanguage.googleapis.com':
        return 'gemini'
    return provider


@dataclass(frozen=True)
class ModelProfile:
    provider: str
    model: str
    context_tokens: int
    supports_images: bool
    api_key_env: str | None = None
    base_url: str | None = None
    reasoning_effort: str | None = None
    structured_outputs: bool = False
    stream_usage: bool = True
    billing_tier: str = 'paid'
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    cached_input_cost_per_million: float | None = None
    timeout_seconds: float = 120.0
    max_retries: int = 1
    source: str = 'config'

    @property
    def prompt_provider(self) -> str:
        return prompt_provider(self.provider, self.base_url)


@dataclass(frozen=True)
class Settings:
    token: str
    development_guild_id: int | None
    database_path: Path
    history_retention_days: int
    profiles: dict[str, ModelProfile]
    dialogue: str
    director: str
    memory: str
    limits: dict[str, int]
    operator_ids: frozenset[int] = frozenset()

    def profile(self, role: str) -> ModelProfile:
        return self.profiles[getattr(self, role)]


DEFAULT_DATABASE_PATH = "data/llmcord.sqlite3"
LIMIT_DEFAULTS = {
    "max_input_tokens": 12000, "max_output_tokens": 700,
    "max_images": 3, "max_attachment_bytes": 8 * 1024 * 1024,
    "max_speakers": 3, "recent_messages": 12,
    "recent_window_seconds": 600, "ambient_cooldown_seconds": 120,
    "memory_input_tokens": 6000, "memory_output_tokens": 550,
    "summary_every_messages": 1, "extraction_every_turns": 1,
}


def _database_path(raw: dict[str, Any]) -> Path:
    return Path(os.environ.get("LLMCORD_DATABASE_PATH") or raw.get("database_path", DEFAULT_DATABASE_PATH))


def resolve_database_path(config_path: str | Path = "config.yaml") -> Path:
    """LLMCORD_DATABASE_PATH, else config.yaml database_path, else the default; invalid YAML raises."""
    import yaml

    if os.environ.get("LLMCORD_DATABASE_PATH"):
        return _database_path({})
    try:
        with open(config_path, encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
    except FileNotFoundError:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a mapping")
    return _database_path(raw)


def operator_ids_from_env(environ=os.environ) -> frozenset[int]:
    """Comma-separated Discord user IDs in LLMCORD_OPERATOR_IDS; unset or empty means no operators."""
    ids = set()
    for entry in (environ.get("LLMCORD_OPERATOR_IDS") or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if len(entry) > 20 or not (entry.isascii() and entry.isdigit() and int(entry)):
            shown = entry if len(entry) <= 24 else entry[:24] + "…"
            raise ValueError(f"LLMCORD_OPERATOR_IDS must be comma-separated Discord user IDs; bad entry: {shown!r}")
        ids.add(int(entry))
    return frozenset(ids)


PROFILE_NAME_RE = re.compile(r'^[a-z0-9][a-z0-9._/-]{0,63}$')
KEY_REF_RE = re.compile(r'^[A-Z][A-Z0-9_]{0,60}_API_KEY$')
BUILTIN_KEY_HOSTS = {
    'OPENAI_API_KEY': ('https://api.openai.com',),
    'ANTHROPIC_API_KEY': ('https://api.anthropic.com',),
    'GEMINI_API_KEY': ('https://generativelanguage.googleapis.com',),
    'GOOGLE_API_KEY': ('https://generativelanguage.googleapis.com',),
}
PROVIDER_ORIGINS = {'openai': 'https://api.openai.com', 'anthropic': 'https://api.anthropic.com'}
PROFILE_KEYS = frozenset({
    'provider', 'model', 'context_tokens', 'supports_images', 'api_key_env', 'base_url', 'reasoning_effort',
    'structured_outputs', 'stream_usage', 'billing_tier', 'input_cost_per_million', 'output_cost_per_million',
    'cached_input_cost_per_million', 'timeout_seconds', 'max_retries',
})
_KEY_REF_TEXT = 'API key must be a ${NAME} reference to a variable ending in _API_KEY'
_ORIGIN_RE = re.compile(r'(?:(https?)://)?([a-z0-9][a-z0-9.-]*)(?::([0-9]{1,5}))?', re.ASCII | re.IGNORECASE)


def validate_profile_name(name: str) -> None:
    if not isinstance(name, str) or not PROFILE_NAME_RE.fullmatch(name):
        raise ValueError('Profile name must be 1-64 characters: lowercase letters, digits, and . _ / -, starting with a letter or digit')


def parse_key_ref(text: str | None) -> str | None:
    """`${NAME}` -> NAME (an allowlisted *_API_KEY variable); empty -> None. Never echoes the input."""
    if text is None or (isinstance(text, str) and not text.strip()):
        return None
    match = re.fullmatch(r'\$\{([^{}]*)\}', text.strip()) if isinstance(text, str) else None
    if not match or not KEY_REF_RE.fullmatch(match.group(1)):
        raise ValueError(_KEY_REF_TEXT)
    return match.group(1)


def format_key_ref(name: str) -> str:
    return '${' + name + '}'


def _split_origin(url: str) -> tuple[str, str, int]:
    bad = False
    try:
        parsed = urlparse(url)
        scheme, host, port = parsed.scheme.lower(), parsed.hostname, parsed.port
        bad = scheme not in {'http', 'https'} or not host or port == 0
    except (ValueError, AttributeError, TypeError):
        bad = True
    if bad:
        raise ValueError('base_url must be an http or https URL with a host and a valid port')
    return scheme, host.lower(), port if port is not None else (443 if scheme == 'https' else 80)


def key_hosts_from_env(environ=os.environ) -> dict[str, frozenset[tuple[str, str, int]]]:
    """Built-in key pins merged with LLMCORD_KEY_HOSTS (`NAME=origin|origin,NAME2=origin`)."""
    result = {name: {_split_origin(origin) for origin in origins} for name, origins in BUILTIN_KEY_HOSTS.items()}
    raw = (environ.get('LLMCORD_KEY_HOSTS') or '').strip()
    if raw:
        for number, entry in enumerate(raw.split(','), 1):
            name, equals, origins = entry.partition('=')
            name = name.strip()
            parsed = set()
            if equals and KEY_REF_RE.fullmatch(name):
                for origin in origins.split('|'):
                    match = _ORIGIN_RE.fullmatch(origin.strip())
                    port = int(match.group(3)) if match and match.group(3) else None
                    if not match or (port is not None and not 0 < port <= 65535):
                        parsed = None
                        break
                    scheme = (match.group(1) or 'https').lower()
                    parsed.add((scheme, match.group(2).lower(), port or (443 if scheme == 'https' else 80)))
            else:
                parsed = None
            if not parsed:
                raise ValueError(f'LLMCORD_KEY_HOSTS must look like NAME_API_KEY=host[:port]|https://host2,...; entry {number} is invalid')
            result.setdefault(name, set()).update(parsed)
    return {name: frozenset(origins) for name, origins in result.items()}


def profile_origin(profile: ModelProfile) -> tuple[str, str, int]:
    if profile.provider == 'anthropic':
        return _split_origin(PROVIDER_ORIGINS['anthropic'])
    if profile.base_url:
        return _split_origin(profile.base_url)
    if profile.provider not in PROVIDER_ORIGINS:
        raise ValueError(f'base_url is required for {profile.provider} profiles')
    return _split_origin(PROVIDER_ORIGINS[profile.provider])


def check_key_pin(profile: ModelProfile, key_hosts: dict[str, frozenset[tuple[str, str, int]]]) -> None:
    name = profile.api_key_env
    if name is None:
        return
    if not KEY_REF_RE.fullmatch(name):
        raise ValueError('API key variable must end in _API_KEY and use capital letters, digits and underscores')
    origin = profile_origin(profile)
    if origin not in key_hosts.get(name, frozenset()):
        raise ValueError(f'{name} may only be sent to its pinned hosts; {origin[1]} is not one of them. '
                         'Add it to LLMCORD_KEY_HOSTS to allow it.')


def profile_from_mapping(name: str, value: dict, *, source: str = 'config') -> ModelProfile:
    """Build and validate one profile from a config.yaml-shaped mapping (the key's presence is not checked)."""
    if source not in {'config', 'dashboard'}:
        raise ValueError("source must be 'config' or 'dashboard'")
    if not isinstance(value, dict):
        raise ValueError(f'Profile {name} must be a mapping')
    model, tokens, base_url, api_key_env = value.get('model'), value.get('context_tokens'), value.get('base_url'), value.get('api_key_env')
    if source == 'dashboard':
        unknown = sorted(str(key) for key in value if key not in PROFILE_KEYS)
        if unknown:
            raise ValueError(f'Unknown setting in {name}: {", ".join(unknown)}')
        if not isinstance(model, str) or not model.strip() or len(model) > 200:
            raise ValueError(f'model in {name} must be a name of 1 to 200 characters')
        if type(tokens) is not int:
            raise ValueError(f'context_tokens in {name} must be a whole number of at least 1')
        if api_key_env is not None and (not isinstance(api_key_env, str) or not KEY_REF_RE.fullmatch(api_key_env)):
            raise ValueError(f'api_key_env in {name} must be a variable ending in _API_KEY')
        if base_url is not None:
            if value.get('provider') == 'anthropic':
                raise ValueError(f'base_url is not used by anthropic profiles ({name})')
            bad = not isinstance(base_url, str) or any(c.isspace() or not c.isprintable() for c in base_url)
            if not bad:
                try:
                    _split_origin(base_url)
                    bad = '@' in urlparse(base_url).netloc
                except ValueError:
                    bad = True
            if bad:
                raise ValueError(f'base_url in {name} must be an http or https URL with a host and no login')
    else:
        try:
            tokens = int(tokens) if type(tokens) is not bool else None
        except (TypeError, ValueError):
            tokens = None
        if model is None:
            raise ValueError(f'model is required for {name}')
    if type(tokens) is not int or tokens < 1:
        raise ValueError(f'context_tokens in {name} must be a whole number of at least 1')
    profile = ModelProfile(
        provider=value.get('provider'), model=model, context_tokens=tokens,
        supports_images=bool(value.get('supports_images', False)),
        api_key_env=api_key_env, base_url=base_url,
        reasoning_effort=value.get('reasoning_effort'),
        structured_outputs=bool(value.get('structured_outputs', False)),
        stream_usage=value.get('stream_usage', True), billing_tier=value.get('billing_tier', 'paid'),
        input_cost_per_million=value.get('input_cost_per_million'), output_cost_per_million=value.get('output_cost_per_million'),
        cached_input_cost_per_million=value.get('cached_input_cost_per_million'),
        timeout_seconds=value.get('timeout_seconds', 120.0), max_retries=value.get('max_retries', 1),
        source=source,
    )
    if profile.billing_tier not in {'paid', 'free'} or type(profile.stream_usage) is not bool:
        raise ValueError(f'Invalid billing_tier or stream_usage in {name}')
    for rate in (profile.input_cost_per_million, profile.output_cost_per_million, profile.cached_input_cost_per_million):
        if rate is not None and (type(rate) not in {int, float} or not math.isfinite(rate) or rate < 0):
            raise ValueError(f'Invalid token price in {name}')
    timeout = profile.timeout_seconds
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f'timeout_seconds in {name} must be a finite number greater than 0')
    if type(profile.max_retries) is not int or profile.max_retries < 0:
        raise ValueError(f'max_retries in {name} must be a whole number of at least 0')
    if profile.provider not in {"openai", "anthropic", "compatible"}:
        raise ValueError(f"Unsupported provider in {name}")
    if profile.provider == "compatible" and not profile.base_url:
        raise ValueError(f"base_url is required for {name}")
    if profile.reasoning_effort is not None and (profile.provider != "compatible" or profile.reasoning_effort not in {"none", "minimal", "low", "medium", "high"}):
        raise ValueError(f"Invalid compatible reasoning_effort in {name}")
    return profile


def _profiles_and_roles(models: dict[str, Any]) -> tuple[dict[str, ModelProfile], dict[str, str]]:
    profiles = {name: profile_from_mapping(name, value) for name, value in models.get("profiles", {}).items()}
    return profiles, {role: models.get(role, models.get("dialogue")) for role in ("dialogue", "director", "memory")}


def load_profiles(path: str | Path = "config.yaml") -> tuple[dict[str, ModelProfile], dict[str, str]]:
    """Profiles and role choices from the models section only (no token or key checks)."""
    import yaml

    try:
        with open(path, encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
    except FileNotFoundError:
        return {}, {}
    profiles, choices = _profiles_and_roles(raw.get("models") or {})
    return profiles, {role: value for role, value in choices.items() if value is not None}


def load_settings(path: str | Path = "config.yaml") -> Settings:
    import yaml

    with open(path, encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}
    discord = raw.get("discord", {})
    token = os.environ.get(discord.get("token_env", "DISCORD_BOT_TOKEN"), "")
    if not token:
        raise ValueError("Discord token environment variable is missing")
    models = raw.get("models", {})
    profiles, choices = _profiles_and_roles(models)
    if not profiles or any(value not in profiles for value in choices.values()):
        raise ValueError("Each model role must name a configured profile")
    for name, profile in profiles.items():
        if name in choices.values() and (profile.provider != "compatible" or profile.api_key_env) and not os.environ.get(profile.api_key_env or ""):
            raise ValueError(f"API key environment variable is missing for {name}")
    limits = {key: int(raw.get("limits", {}).get(key, value)) for key, value in LIMIT_DEFAULTS.items()}
    if any(value <= 0 for value in limits.values()) or limits["max_speakers"] > 3:
        raise ValueError("Limits must be positive and max_speakers cannot exceed 3")
    if limits["summary_every_messages"] > 100 or limits["extraction_every_turns"] > 100:
        raise ValueError("summary_every_messages and extraction_every_turns must be between 1 and 100")
    guild_id = os.environ.get("DISCORD_GUILD_ID") or discord.get("development_guild_id")
    if guild_id and int(guild_id) <= 0:
        raise ValueError("DISCORD_GUILD_ID must be a positive server ID")
    retention = int(raw.get("history_retention_days", 90))
    if retention <= 0:
        raise ValueError("history_retention_days must be positive")
    return Settings(
        token=token, development_guild_id=int(guild_id) if guild_id else None,
        database_path=_database_path(raw),
        history_retention_days=retention, profiles=profiles,
        dialogue=choices["dialogue"], director=choices["director"],
        memory=choices["memory"], limits=limits, operator_ids=operator_ids_from_env(),
    )
