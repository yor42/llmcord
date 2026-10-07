"""Shared fakes for new tests. Older test files keep their local helpers for now."""
from __future__ import annotations

import asyncio
import re
import time
from collections import Counter
from datetime import datetime, timezone
from types import SimpleNamespace

import discord
import httpx

from llmcord_core.config import ModelProfile, Settings

LIMITS = {"max_input_tokens": 12000, "max_output_tokens": 700, "max_images": 3,
          "max_attachment_bytes": 8388608, "max_speakers": 3, "recent_messages": 12,
          "recent_window_seconds": 600, "ambient_cooldown_seconds": 120,
          # BUG-03 / D4 memory cadence + budget keys (defaults match load_settings).
          "memory_input_tokens": 6000, "memory_output_tokens": 550,
          "summary_every_messages": 1, "extraction_every_turns": 1}


def make_settings(model: str = "test", **limits) -> Settings:
    profile = ModelProfile("compatible", model, 16000, False, base_url="http://localhost/v1")
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


class MemoryModels(FakeModels):
    """FakeModels that records summary/extraction calls with their full arguments.

    ``summaries`` holds ``{"text", "max_tokens"}`` per text call (``text`` joins system + message texts);
    ``extractions`` holds ``{"text"}`` per ``extract_memory`` call. ``extraction_result`` is returned from
    extraction calls; ``summary_failures`` is the number of upcoming text calls that raise ``RuntimeError``.
    """

    def __init__(self, summary="Fresh summary", extraction_result=None, summary_failures=0):
        super().__init__(summary=summary)
        self.summaries, self.extractions = [], []
        self.extraction_result = extraction_result or {"shared_facts": [], "personal_facts": [], "encounter_facts": []}
        self.summary_failures = summary_failures

    @staticmethod
    def _joined(system, messages):
        return "\n".join([system, *(message.text for message in messages)])

    async def structured(self, role, system, messages, schema_name, schema):
        if schema_name == "extract_memory":
            self.calls.append(("structured", schema_name))
            self.extractions.append({"text": self._joined(system, messages)})
            return self.extraction_result
        return await super().structured(role, system, messages, schema_name, schema)

    async def text(self, role, system, messages, max_tokens=None):
        self.calls.append(("text", role))
        self.summaries.append({"text": self._joined(system, messages), "max_tokens": max_tokens})
        if self.summary_failures:
            self.summary_failures -= 1
            raise RuntimeError("summary provider down")
        return self.summary_text


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

    def __init__(self, guild_id=1, channel_id=100, user_id=9, admin=False, channel=None):
        self.guild_id = guild_id
        self.guild = SimpleNamespace(id=guild_id, get_member=lambda _ident: None) if guild_id else None
        if channel is None and channel_id:
            channel = SimpleNamespace(id=channel_id, mention=f"<#{channel_id}>")
        self.channel = channel
        self.user = SimpleNamespace(id=user_id, display_name="Tester", bot=False)
        self.permissions = discord.Permissions(administrator=admin)
        self.created_at = datetime.now(timezone.utc)
        self.response, self.followup = FakeResponse(), FakeFollowup()

    @property
    def replies(self) -> list[str]:
        return [content for content, _ in self.response.sent + self.followup.sent]

    async def original_response(self):
        """The public reply sent through ``response.send_message`` (``/summon`` uses its id as the scene input)."""
        return SimpleNamespace(id=7000 + len(self.response.sent))


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
                      guilds_by_token=None, guild_failures=None, token_status=200, yield_on_guilds=False, retry_after=0):
    """httpx MockTransport imitating the Discord endpoints the dashboard uses, counting calls per path.

    ``admin_guilds`` is read on every request, so pass a list and mutate it to simulate Discord-side
    permission changes. ``guilds_by_token`` maps an access token (without ``Bearer``) to its guild ids and
    takes precedence; guild-list calls are then also counted under ``"GET /users/@me/guilds <token>"``.
    ``guild_failures`` is a list consumed one item per guild-list call before normal answers: an int status
    code (429 answers carry ``Retry-After: <retry_after>``, default 0) or ``"network"`` for a connection error.
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
                return httpx.Response(failure, headers={"Retry-After": str(retry_after)} if failure == 429 else {}, json={})
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


REFERENCE_PATTERN = re.compile(r"(?i)\bref\b[\s:#.-]*([0-9a-z]{4,})")


def reference_ids(text: str) -> list[str]:
    """Reference tokens in a user-facing error (``ref a1b2c3``, ``Ref: A1B2C3``; UX-07/SEC-05 per D3). The format
    is deliberately loose: tests pin that the same token appears in the reply and the log, not its shape."""
    return REFERENCE_PATTERN.findall(text or "")


def is_turn_failure(content: str) -> bool:
    """A public turn-failure message: the pre-D3 ``Character response failed during <stage>: <detail>`` text, or a
    generic message carrying a reference id (D3)."""
    return bool(content) and (content.startswith("Character response failed") or bool(reference_ids(content)))


def not_found(text="Unknown Webhook") -> discord.NotFound:
    return discord.NotFound(SimpleNamespace(status=404, reason="Not Found"), text)


class FakeSentMessage:
    """A message posted by a fake webhook or channel; records edits and deletion."""

    def __init__(self, ident, content):
        self.id, self.content, self.edits, self.deleted = ident, content, [], False

    async def edit(self, *, content, **kwargs):
        self.content = content
        self.edits.append(content)

    async def delete(self):
        self.deleted = True


class FakeHook:
    """Fake ``discord.Webhook`` owned by a ``FakeTextChannel``. A dead hook (deleted on Discord) raises
    ``discord.NotFound`` from ``send``/``edit``. ``fail_next_chunk`` makes the next non-placeholder send raise
    ``NotFound`` and kills the hook (deleted mid-turn)."""

    def __init__(self, channel, ident, name, avatar, token="hook-token"):
        self.channel, self.id, self.token, self.name, self.avatar = channel, ident, token, name, avatar
        self.dead, self.fail_next_chunk = False, False
        self.send_attempts, self.posts, self.options, self.edits = 0, [], [], []

    def _touch(self):
        if self in self.channel.doomed:
            self.channel.kill(self)
        if self.dead:
            raise not_found()

    async def edit(self, *, name=None, avatar=None, **kwargs):
        self._touch()
        self.edits.append({"name": name, "avatar": avatar})
        self.name, self.avatar = name, avatar
        return self

    async def send(self, content, **kwargs):
        self.send_attempts += 1
        self._touch()
        if self.fail_next_chunk and content != "…":
            self.fail_next_chunk = False
            self.channel.kill(self)
            raise not_found()
        message = FakeSentMessage(self.channel.next_id(), content)
        self.posts.append(message)
        self.options.append(kwargs)
        return message


class FakeTextChannel(discord.TextChannel):
    """Passes ``isinstance(channel, discord.TextChannel)`` so ``SkitBot._webhook`` runs for real.

    ``hooks`` is the live webhook list Discord would return; ``listings``/``creates`` count API calls.
    ``doom(hook)`` models a webhook deleted on Discord right before the bot's next touch: the next
    ``webhooks()`` listing still includes it (then it dies), and any ``send``/``edit`` on it raises ``NotFound``.
    ``create_error`` is raised by ``create_webhook``; ``dead_on_create`` makes new hooks already deleted.
    ``fetch_message`` raises ``NotFound`` for ids in ``missing`` and counts calls in ``fetches``."""

    def __init__(self, ident=100):  # deliberately skips discord.TextChannel.__init__
        self.id = ident
        self.hooks, self.doomed, self.sent, self.missing = [], set(), [], set()
        self.listings = self.creates = self.fetches = 0
        self.create_error, self.dead_on_create = None, False
        self._next_id = 2000

    def next_id(self):
        self._next_id += 1
        return self._next_id

    def kill(self, hook):
        hook.dead = True
        self.doomed.discard(hook)
        if hook in self.hooks:
            self.hooks.remove(hook)

    def doom(self, hook):
        self.doomed.add(hook)

    @property
    def errors(self):
        return [m.content for m in self.sent if not m.deleted and is_turn_failure(m.content)]

    async def webhooks(self):
        self.listings += 1
        snapshot = list(self.hooks)
        for hook in list(self.doomed):
            self.kill(hook)
        return snapshot

    async def create_webhook(self, *, name, avatar=None, reason=None):
        self.creates += 1
        if self.create_error:
            raise self.create_error
        hook = FakeHook(self, 900 + self.creates, name, avatar)
        if self.dead_on_create:
            hook.dead = True
        else:
            self.hooks.append(hook)
        return hook

    async def send(self, content=None, **kwargs):
        message = FakeSentMessage(self.next_id(), content)
        self.sent.append(message)
        return message

    async def history(self, **kwargs):
        """Empty channel history (``/summon`` scans it for recent context)."""
        return
        yield

    async def fetch_message(self, ident):
        self.fetches += 1
        if ident in self.missing:
            raise discord.NotFound(SimpleNamespace(status=404, reason="Not Found"), "Unknown Message")
        return SimpleNamespace(id=ident)


def memory_tasks(bot) -> list[asyncio.Task]:
    """Background memory tasks the bot is tracking (REL-02 seam ``bot.memory_tasks``: channel id -> task, or an
    iterable of tasks). Empty while extraction/summary still run inline inside ``run_scene``."""
    found = []
    for value in getattr(bot, "memory_tasks", {}).values():
        found.extend(value if isinstance(value, (list, tuple, set)) else [value])
    return [task for task in found if isinstance(task, asyncio.Future)]


async def drain_memory_tasks(bot, within: float = 1.0) -> None:
    """Wait (bounded) until the bot's background memory tasks finish, without retrieving their exceptions.
    A no-op when extraction/summary run inline."""
    deadline = asyncio.get_running_loop().time() + within
    while pending := [task for task in memory_tasks(bot) if not task.done()]:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise AssertionError(f"{len(pending)} memory task(s) still pending after {within}s")
        await asyncio.wait(pending, timeout=remaining)
