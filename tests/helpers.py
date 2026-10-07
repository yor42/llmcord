"""Shared fakes for new tests. Older test files keep their local helpers for now."""
from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timezone
from types import SimpleNamespace

import discord
import httpx

from llmcord_core.config import ModelProfile, Settings

LIMITS = {"max_input_tokens": 12000, "max_output_tokens": 700, "max_images": 3,
          "max_attachment_bytes": 8388608, "max_speakers": 3, "recent_messages": 12,
          "recent_window_seconds": 600, "ambient_cooldown_seconds": 120}


def make_settings(**limits) -> Settings:
    profile = ModelProfile("compatible", "test", 16000, False, base_url="http://localhost/v1")
    return Settings("token", None, ":memory:", 90, {"test": profile}, "test", "test", "test", {**LIMITS, **limits})


class FakeModels:
    """Deterministic model gateway: fixed director choice, empty extraction, fixed summary."""

    def __init__(self, speakers=None, summary="Earlier scene summary"):
        self.chosen, self.summary_text, self.calls = speakers, summary, []

    async def structured(self, role, system, messages, schema_name, schema):
        self.calls.append(("structured", schema_name))
        if schema_name == "choose_speakers":
            return {"speakers": self.chosen or []}
        return {"shared_facts": [], "personal_facts": [], "encounter_facts": []}

    async def text(self, role, system, messages, max_tokens=None):
        self.calls.append(("text", role))
        return self.summary_text

    async def stream_text(self, role, system, messages):
        self.calls.append(("stream", role))
        yield "<emotion>neutral</emotion>\nHello."

    async def close(self):
        pass


class FakeResponse:
    def __init__(self):
        self.sent, self._done = [], False

    def is_done(self):
        return self._done

    async def send_message(self, content=None, **kwargs):
        self.sent.append((content, kwargs))
        self._done = True

    async def defer(self, **kwargs):
        self._done = True


class FakeFollowup:
    def __init__(self):
        self.sent = []

    async def send(self, content=None, **kwargs):
        self.sent.append((content, kwargs))


class FakeInteraction:
    """Minimal discord.Interaction stand-in for invoking app-command callbacks directly."""

    def __init__(self, guild_id=1, channel_id=100, user_id=9, admin=False):
        self.guild_id = guild_id
        self.guild = SimpleNamespace(id=guild_id, get_member=lambda _ident: None) if guild_id else None
        self.channel = SimpleNamespace(id=channel_id, mention=f"<#{channel_id}>") if channel_id else None
        self.user = SimpleNamespace(id=user_id, display_name="Tester", bot=False)
        self.permissions = discord.Permissions(administrator=admin)
        self.created_at = datetime.now(timezone.utc)
        self.response, self.followup = FakeResponse(), FakeFollowup()

    @property
    def replies(self) -> list[str]:
        return [content for content, _ in self.response.sent + self.followup.sent]


def command(bot, path: str):
    """Look up an app command by its space-separated path, e.g. ``command(bot, "lore add")``."""
    parent, *rest = path.split()
    found = bot.tree.get_command(parent)
    for name in rest:
        found = found.get_command(name)
    if found is None:
        raise LookupError(path)
    return found


async def invoke(bot, path: str, interaction: FakeInteraction, *args, **kwargs):
    """Run checks then the callback, routing errors through the tree's error handler like discord.py."""
    from discord import app_commands
    cmd = command(bot, path)
    try:
        for check in cmd.checks:
            check(interaction)
        await cmd.callback(interaction, *args, **kwargs)
    except app_commands.AppCommandError as error:
        await bot.tree.on_error(interaction, error)
    except Exception as error:  # discord.py wraps callback errors the same way
        await bot.tree.on_error(interaction, app_commands.CommandInvokeError(cmd, error))


def discord_transport(admin_guilds=(1,), calls: Counter | None = None, rate_limited=False):
    """httpx MockTransport imitating the Discord endpoints the dashboard uses, counting calls per path."""
    calls = calls if calls is not None else Counter()

    def handler(request: httpx.Request):
        path = request.url.path.removeprefix("/api/v10")
        calls[f"{request.method} {path}"] += 1
        headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset-After": "0"} if rate_limited else {}
        if path == "/users/@me/guilds":
            return httpx.Response(200, headers=headers, json=[
                {"id": str(g), "name": f"Guild {g}", "permissions": "8"} for g in admin_guilds])
        if path == "/oauth2/token":
            return httpx.Response(200, json={"access_token": "access", "refresh_token": "refresh", "expires_in": 3600})
        if path == "/users/@me":
            return httpx.Response(200, json={"id": "4", "username": "Admin"})
        if path.endswith("/channels"):
            return httpx.Response(200, json=[{"id": "100", "name": "scene", "type": 0}])
        return httpx.Response(404, json={})

    return httpx.MockTransport(handler), calls


def install_session(app, ident="session", user_id="4", csrf="csrf"):
    app.state.sessions[ident] = {"user": {"id": user_id, "username": "Admin"}, "expires": time.time() + 3600,
                                 "token_expires": time.time() + 3600, "csrf": csrf, "access": "access", "refresh": "refresh"}
    return ident
