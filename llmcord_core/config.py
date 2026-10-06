from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    provider: str
    model: str
    context_tokens: int
    supports_images: bool
    api_key_env: str | None = None
    base_url: str | None = None


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

    def profile(self, role: str) -> ModelProfile:
        return self.profiles[getattr(self, role)]


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
        )
        for name, value in models.get("profiles", {}).items()
    }
    choices = {role: models.get(role, models.get("dialogue")) for role in ("dialogue", "director", "memory")}
    if not profiles or any(value not in profiles for value in choices.values()):
        raise ValueError("Each model role must name a configured profile")
    for name, profile in profiles.items():
        if profile.provider not in {"openai", "anthropic", "compatible"}:
            raise ValueError(f"Unsupported provider in {name}")
        if name in choices.values() and profile.provider != "compatible" and not os.environ.get(profile.api_key_env or ""):
            raise ValueError(f"API key environment variable is missing for {name}")
        if profile.provider == "compatible" and not profile.base_url:
            raise ValueError(f"base_url is required for {name}")
    defaults = {
        "max_input_tokens": 12000, "max_output_tokens": 700,
        "max_images": 3, "max_attachment_bytes": 8 * 1024 * 1024,
        "max_speakers": 3, "recent_messages": 12,
        "recent_window_seconds": 600, "ambient_cooldown_seconds": 120,
    }
    limits = {key: int(raw.get("limits", {}).get(key, value)) for key, value in defaults.items()}
    if any(value <= 0 for value in limits.values()) or limits["max_speakers"] > 3:
        raise ValueError("Limits must be positive and max_speakers cannot exceed 3")
    guild_id = discord.get("development_guild_id")
    retention = int(raw.get("history_retention_days", 90))
    if retention <= 0:
        raise ValueError("history_retention_days must be positive")
    return Settings(
        token=token, development_guild_id=int(guild_id) if guild_id else None,
        database_path=Path(raw.get("database_path", "data/llmcord.sqlite3")),
        history_retention_days=retention, profiles=profiles,
        dialogue=choices["dialogue"], director=choices["director"],
        memory=choices["memory"], limits=limits,
    )
