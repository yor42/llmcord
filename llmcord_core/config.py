from __future__ import annotations

import os
import math
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
        if len(entry) > 20 or not (entry.isascii() and entry.isdigit()):
            shown = entry if len(entry) <= 24 else entry[:24] + "…"
            raise ValueError(f"LLMCORD_OPERATOR_IDS must be comma-separated Discord user IDs; bad entry: {shown!r}")
        ids.add(int(entry))
    return frozenset(ids)


def load_settings(path: str | Path = "config.yaml") -> Settings:
    import yaml

    with open(path, encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file) or {}
    discord = raw.get("discord", {})
    token = os.environ.get(discord.get("token_env", "DISCORD_BOT_TOKEN"), "")
    if not token:
        raise ValueError("Discord token environment variable is missing")
    models = raw.get("models", {})
    profiles = {
        name: ModelProfile(
            provider=value["provider"], model=value["model"],
            context_tokens=int(value["context_tokens"]),
            supports_images=bool(value.get("supports_images", False)),
            api_key_env=value.get("api_key_env"), base_url=value.get("base_url"),
            reasoning_effort=value.get("reasoning_effort"),
            structured_outputs=bool(value.get("structured_outputs", False)),
            stream_usage=value.get('stream_usage', True), billing_tier=value.get('billing_tier', 'paid'),
            input_cost_per_million=value.get('input_cost_per_million'), output_cost_per_million=value.get('output_cost_per_million'),
            cached_input_cost_per_million=value.get('cached_input_cost_per_million'),
            timeout_seconds=value.get('timeout_seconds', 120.0), max_retries=value.get('max_retries', 1),
        )
        for name, value in models.get("profiles", {}).items()
    }
    choices = {role: models.get(role, models.get("dialogue")) for role in ("dialogue", "director", "memory")}
    if not profiles or any(value not in profiles for value in choices.values()):
        raise ValueError("Each model role must name a configured profile")
    for name, profile in profiles.items():
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
        if name in choices.values() and (profile.provider != "compatible" or profile.api_key_env) and not os.environ.get(profile.api_key_env or ""):
            raise ValueError(f"API key environment variable is missing for {name}")
        if profile.provider == "compatible" and not profile.base_url:
            raise ValueError(f"base_url is required for {name}")
        if profile.reasoning_effort is not None and (profile.provider != "compatible" or profile.reasoning_effort not in {"none", "minimal", "low", "medium", "high"}):
            raise ValueError(f"Invalid compatible reasoning_effort in {name}")
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
