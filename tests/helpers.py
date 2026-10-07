"""Shared fakes for new tests. Older test files keep their local helpers for now."""
from __future__ import annotations

import asyncio
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


def discord_transport(admin_guilds=(1,), calls: Counter | None = None, rate_limited=False, *,
                      guilds_by_token=None, guild_failures=None, token_status=200, yield_on_guilds=False):
    """httpx MockTransport imitating the Discord endpoints the dashboard uses, counting calls per path.

    ``admin_guilds`` is read on every request, so pass a list and mutate it to simulate Discord-side
    permission changes. ``guilds_by_token`` maps an access token (without ``Bearer``) to its guild ids and
    takes precedence; guild-list calls are then also counted under ``"GET /users/@me/guilds <token>"``.
    ``guild_failures`` is a list consumed one item per guild-list call before normal answers: an int status
    code (429 answers carry ``Retry-After: 0``) or ``"network"`` for a connection error.
    ``token_status`` is the status of ``POST /oauth2/token`` (>= 400 makes refreshes fail).
    ``yield_on_guilds`` makes guild-list calls yield to the event loop once (the mock otherwise answers without
    suspending, so ``asyncio.gather`` of guards would run them one after another instead of overlapping).
    """
    calls = calls if calls is not None else Counter()

    def handler(request: httpx.Request):
        path = request.url.path.removeprefix("/api/v10")
        calls[f"{request.method} {path}"] += 1
        headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset-After": "0"} if rate_limited else {}
        if path == "/users/@me/guilds":
            token = request.headers.get("Authorization", "").removeprefix("Bearer ")
            if guilds_by_token is not None:
                calls[f"{request.method} {path} {token}"] += 1
            if guild_failures:
                failure = guild_failures.pop(0)
                if failure == "network":
                    raise httpx.ConnectError("offline", request=request)
                return httpx.Response(failure, headers={"Retry-After": "0"} if failure == 429 else {}, json={})
            guilds = guilds_by_token.get(token, ()) if guilds_by_token is not None else admin_guilds
            return httpx.Response(200, headers=headers, json=[
                {"id": str(g), "name": f"Guild {g}", "permissions": "8"} for g in guilds])
        if path == "/oauth2/token":
            if token_status >= 400:
                return httpx.Response(token_status, json={"error": "invalid_grant"})
            return httpx.Response(200, json={"access_token": "access", "refresh_token": "refresh", "expires_in": 3600})
        if path == "/users/@me":
            return httpx.Response(200, json={"id": "4", "username": "Admin"})
        if path.endswith("/channels"):
            return httpx.Response(200, json=[{"id": "100", "name": "scene", "type": 0}])
        return httpx.Response(404, json={})

    if yield_on_guilds:
        async def async_handler(request: httpx.Request):
            if request.url.path.endswith("/users/@me/guilds"):
                await asyncio.sleep(0)
            return handler(request)

        return httpx.MockTransport(async_handler), calls
    return httpx.MockTransport(handler), calls


class ShiftedClock:
    """Shift ``time.time`` and ``time.monotonic`` forward without sleeping (``with ShiftedClock() as clock:
    clock.advance(301)``). Patches the ``time`` module itself so code using ``time.time()`` /
    ``time.monotonic()`` sees the shift; real time keeps flowing underneath, so event loops stay sane.
    Note: it patches ``time.monotonic`` process-wide, so asyncio/anyio loop timers scheduled before ``advance()`` fire early."""

    def __init__(self):
        self.offset = 0.0
        self._real_time, self._real_monotonic = time.time, time.monotonic

    def advance(self, seconds: float):
        self.offset += seconds

    def __enter__(self):
        time.time = lambda: self._real_time() + self.offset
        time.monotonic = lambda: self._real_monotonic() + self.offset
        return self

    def __exit__(self, *exc):
        time.time, time.monotonic = self._real_time, self._real_monotonic
        return False


def install_session(app, ident="session", user_id="4", csrf="csrf", access="access"):
    app.state.sessions[ident] = {"user": {"id": user_id, "username": "Admin"}, "expires": time.time() + 3600,
                                 "token_expires": time.time() + 3600, "csrf": csrf, "access": access, "refresh": "refresh"}
    return ident
