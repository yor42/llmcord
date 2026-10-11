"""Shared fakes for new tests. Older test files keep their local helpers for now."""
from __future__ import annotations

import asyncio
import os
import re
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import discord
import httpx
from PIL import Image

from llmcord_core.avatars import normalize_avatar
from llmcord_core.config import ModelProfile, Settings, load_settings

LIMITS = {"max_input_tokens": 12000, "max_output_tokens": 700, "max_images": 3,
          "max_attachment_bytes": 8388608, "max_speakers": 3, "recent_messages": 12,
          "recent_window_seconds": 600, "ambient_cooldown_seconds": 120,
          # BUG-03 / D4 memory cadence + budget keys (defaults match load_settings).
          "memory_input_tokens": 6000, "memory_output_tokens": 550,
          "summary_every_messages": 1, "extraction_every_turns": 1}


def make_settings(model: str = "test", **limits) -> Settings:
    profile = ModelProfile("compatible", model, 16000, False, base_url="http://localhost/v1")
    return Settings("token", None, ":memory:", 90, {"test": profile}, "test", "test", "test", {**LIMITS, **limits})


class CompiledAdapter:
    """Adapts compiled requests to the fakes' (system, messages) signature, mirroring ModelGateway.compiled_input.

    ``provider = 'anthropic'`` (default) splits system messages into the system string; any other value
    ('openai', 'compatible') mirrors that path: empty system, messages passed through inline."""

    provider = 'anthropic'

    def _split(self, request):
        if self.provider != 'anthropic':
            return '', request.messages
        return '\n\n'.join(m.text for m in request.messages if m.role == 'system'), [m for m in request.messages if m.role != 'system']

    async def text_compiled(self, role, request, max_tokens=None):
        system, messages = self._split(request)
        return await self.text(role, system, messages, max_tokens)

    async def structured_compiled(self, role, request, schema_name, schema):
        system, messages = self._split(request)
        return await self.structured(role, system, messages, schema_name, schema)

    async def stream_compiled(self, role, request):
        system, messages = self._split(request)
        async for delta in self.stream_text(role, system, messages):
            yield delta


class FakeModels(CompiledAdapter):
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
        self.sent, self._done, self.choices = [], False, None

    def is_done(self):
        return self._done

    async def send_message(self, content=None, **kwargs):
        self.sent.append((content, kwargs))
        self._done = True

    async def defer(self, **kwargs):
        self._done = True

    async def autocomplete(self, choices):
        """Records the choices an autocomplete callback answered with (``None`` until it answers)."""
        self.choices = list(choices)
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
    """Look up an app command by its space-separated path, e.g. ``command(bot, "admin lore add")``.
    Raises ``LookupError(path)`` when any segment is not registered."""
    parent, *rest = path.split()
    found = bot.tree.get_command(parent)
    for name in rest:
        if found is None or not hasattr(found, "get_command"):
            raise LookupError(path)
        found = found.get_command(name)
    if found is None:
        raise LookupError(path)
    return found


def leaf_commands(bot) -> dict:
    """Every invocable slash command in the live tree, keyed by qualified path (``"admin lore add"``, ``"summon"``)."""
    from discord import app_commands
    return {cmd.qualified_name: cmd for cmd in bot.tree.walk_commands() if isinstance(cmd, app_commands.Command)}


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


async def autocomplete(bot, path: str, option: str, interaction: FakeInteraction, current: str) -> list:
    """Run the autocomplete callback of ``option`` on command ``path`` the way discord.py 2.6 dispatches it
    (``Command._invoke_autocomplete``: autocomplete checks, cog binding, then ``interaction.response.autocomplete``)
    and return the choices it answered with. ``option`` is the synced (Discord-facing) option name. Raises
    ``app_commands.CommandSignatureMismatch`` when the option has no autocomplete callback, and propagates any
    exception the callback raises."""
    interaction.response.choices = None
    await command(bot, path)._invoke_autocomplete(interaction, option, SimpleNamespace(**{option: current}))
    return interaction.response.choices


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


_DASHBOARD = {}


def shared_dashboard():
    """``(app, client)`` for the one NiceGUI-mounted app this process can build (NiceGUI's global app cannot be
    mounted twice, so every test that needs ``enable_dashboard=True`` shares it). The client's cookies are reset on
    each call; install your own session with ``install_session(app, ident)``."""
    if not _DASHBOARD:
        import atexit

        from fastapi.testclient import TestClient

        from llmcord_core.web import create_app
        transport, _ = discord_transport(admin_guilds=(1,))
        app = create_app(":memory:", "https://pi.test", "client", "secret", "bot", httpx.AsyncClient(transport=transport),
                         enable_dashboard=True, config_path="tests/nonexistent-config.yaml")
        client = TestClient(app, base_url="https://pi.test")
        client.__enter__()
        atexit.register(client.__exit__, None, None, None)
        _DASHBOARD.update(app=app, client=client)
    _DASHBOARD["client"].cookies.clear()
    return _DASHBOARD["app"], _DASHBOARD["client"]


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


def not_found(text="Unknown Webhook", code=10015) -> discord.NotFound:
    """A 404 as Discord sends it; ``code`` is the JSON error code (10015 Unknown Webhook, 10008 Unknown Message)."""
    return discord.NotFound(SimpleNamespace(status=404, reason="Not Found"), {"message": text, "code": code})


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
    ``fetch_message`` raises ``NotFound`` for ids in ``missing`` and counts calls in ``fetches``.
    ``history(**kwargs)`` yields ``messages`` (or raises ``error`` when called) and records kwargs in ``history_calls``."""

    def __init__(self, ident=100, messages=(), error=None):  # deliberately skips discord.TextChannel.__init__
        self.id = ident
        self.messages, self.error, self.history_calls = list(messages), error, []
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

    def history(self, **kwargs):
        """Channel history (``/summon`` scans it for recent context); empty unless ``messages`` is given."""
        self.history_calls.append(kwargs)
        if self.error:
            raise self.error
        return self._replay()

    async def _replay(self):
        for message in self.messages:
            yield message

    async def fetch_message(self, ident):
        self.fetches += 1
        if ident in self.missing:
            raise discord.NotFound(SimpleNamespace(status=404, reason="Not Found"), "Unknown Message")
        return SimpleNamespace(id=ident)


class FakeThread(discord.Thread):
    """Passes ``isinstance(channel, discord.Thread)`` so ``SkitBot.location`` / ``local_scope`` take the thread path.
    Pass it as ``FakeInteraction(channel=FakeThread(101, parent_id=100))``."""

    def __init__(self, ident=101, parent_id=100):  # deliberately skips discord.Thread.__init__
        self.id, self.parent_id = ident, parent_id


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


def core_settings():
    """Settings used by test_core/scene/identity tests (vision on, no memory-cadence keys)."""
    profile = ModelProfile("compatible", "test", 16000, True, base_url="http://localhost/v1")
    return Settings("test", None, ":memory:", 90, {"test": profile}, "test", "test", "test",
        {"max_input_tokens": 12000, "max_output_tokens": 700, "max_images": 3,
         "max_attachment_bytes": 8388608, "max_speakers": 3, "recent_messages": 12,
         "recent_window_seconds": 600, "ambient_cooldown_seconds": 120})


def image(color):
    output = BytesIO()
    Image.new('RGB', (40, 50), color).save(output, 'PNG')
    return normalize_avatar(output.getvalue())


ENV = {"DISCORD_BOT_TOKEN": "test", "TEST_OPENAI_KEY": "sk-test", "TEST_ANTHROPIC_KEY": "sk-ant-test"}

PROFILES = {
    "compatible": "      provider: compatible\n      model: local\n      context_tokens: 8192\n"
                  "      base_url: http://localhost:11434/v1\n",
    "openai": "      provider: openai\n      model: gpt-test\n      context_tokens: 8192\n"
              "      api_key_env: TEST_OPENAI_KEY\n",
    "anthropic": "      provider: anthropic\n      model: claude-test\n      context_tokens: 8192\n"
                 "      api_key_env: TEST_ANTHROPIC_KEY\n",
}


def write_config(directory: str, provider: str = "compatible", extra: str = "") -> Path:
    path = Path(directory) / "config.yaml"
    path.write_text("discord: {}\nmodels:\n  dialogue: main\n  profiles:\n    main:\n"
                    + PROFILES[provider] + extra, encoding="utf-8")
    return path


def settings_for(provider: str = "compatible", extra: str = ""):
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ENV, clear=True):
        return load_settings(write_config(directory, provider, extra))


class FlowFakeModels(CompiledAdapter):
    def __init__(self, speakers):
        self.speakers = speakers
        self.lines = iter(["First line", "Second line"])

    async def structured(self, role, system, messages, schema_name, schema):
        if schema_name == "choose_speakers":
            return {"speakers": self.speakers}
        return {"shared_facts": [], "personal_facts": [], "encounter_facts": []}

    async def stream_text(self, role, system, messages):
        yield next(self.lines)

    async def text(self, role, system, messages, max_tokens=None):
        return "The group met."


class FakeMessage:
    def __init__(self, ident, content):
        self.id, self.content = ident, content
        self.deleted = False
        self.edits = []

    async def edit(self, *, content, **kwargs):
        self.content = content
        self.edits.append(content)

    async def delete(self):
        self.deleted = True


class FakeWebhook:
    def __init__(self, start):
        self.next_id = start
        self.posts = []
        self.options = []

    async def send(self, content, **kwargs):
        # discord.py dereferences thread.id when the keyword is present.
        if kwargs.get('thread', discord.utils.MISSING) is None:
            raise AttributeError("'NoneType' object has no attribute 'id'")
        result = FakeMessage(self.next_id, content)
        self.next_id += 1
        self.posts.append(result)
        self.options.append(kwargs)
        return result


class FakeChannel:
    def __init__(self, ident):
        self.id = ident
        self.messages = []
        self.options = []

    @property
    def errors(self):
        return [message.content for message in self.messages
                if not message.deleted and is_turn_failure(message.content)]

    async def send(self, content, **kwargs):
        message = FakeMessage(5000 + len(self.messages), content)
        self.messages.append(message)
        self.options.append(kwargs)
        return message


class CoreFakeModels(CompiledAdapter):
    """FakeModels variant without call recording; subclasses record their own calls."""

    def __init__(self, speakers=None):
        self.chosen = speakers

    async def structured(self, role, system, messages, schema_name, schema):
        if schema_name == "choose_speakers":
            return {"speakers": self.chosen or []}
        return {"shared_facts": [], "personal_facts": [], "encounter_facts": []}

    async def text(self, role, system, messages, max_tokens=None):
        return "Earlier scene summary"


# ---- table games with public threads (FEAT-21 part B1) -------------------------------------------------

def forbidden(text="Missing Permissions", code=50013) -> discord.Forbidden:
    """A 403 as Discord sends it (``code`` 50013: Missing Permissions)."""
    return discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), {"message": text, "code": code})


class FakePosted:
    """A message posted to a ``FakeGameChannel`` / ``FakeTableThread``: records edits (``edits``: the kwargs of each),
    pinning and deletion. ``content``/``view`` follow the last edit that set them (``view=None`` removes the buttons).
    ``create_thread`` is available on messages of a channel."""

    def __init__(self, host, ident, content, view=None, **kwargs):
        self.host, self.id, self.content, self.view, self.kwargs = host, ident, content, view, kwargs
        self.edits, self.pinned, self.deleted = [], False, False

    def apply(self, kwargs):
        if kwargs.get("content", discord.utils.MISSING) is not discord.utils.MISSING:
            self.content = kwargs["content"]
        if kwargs.get("view", discord.utils.MISSING) is not discord.utils.MISSING:
            self.view = kwargs["view"]
        self.edits.append(dict(kwargs))

    def _alive(self):
        if self.deleted:
            raise not_found("Unknown Message", 10008)

    async def edit(self, **kwargs):
        await self.host.touch("edit")
        self._alive()
        if self.host.edit_error:
            raise self.host.edit_error
        self.apply(kwargs)
        return self

    async def delete(self, **kwargs):
        await self.host.touch("delete")
        self._alive()
        self.deleted = True

    async def pin(self, **kwargs):
        await self.host.touch("pin")
        self._alive()
        if self.host.pin_error:
            raise self.host.pin_error
        self.pinned = True

    async def create_thread(self, *, name, **kwargs):
        self._alive()
        return await self.host.make_thread(name, **kwargs)


class FakePartial:
    """``get_partial_message(id)``: edit/delete/pin/create_thread reach the posted message, or raise NotFound when it is gone."""

    def __init__(self, host, ident):
        self.host, self.id = host, ident

    def _target(self):
        message = self.host.messages.get(self.id)
        if message is None:
            raise not_found("Unknown Message", 10008)
        return message

    async def edit(self, **kwargs):
        return await self._target().edit(**kwargs)

    async def delete(self, **kwargs):
        return await self._target().delete(**kwargs)

    async def pin(self, **kwargs):
        return await self._target().pin(**kwargs)

    async def create_thread(self, *, name, **kwargs):
        return await self._target().create_thread(name=name, **kwargs)


class FakeMessageHost:
    """Mixin for fake channels and threads that hold posted messages. ``calls`` lists every Discord call kind
    (``send``, ``edit``, ``delete``, ``pin``, ``fetch``, ``create_thread``, ``thread_edit``); ``on_call(kind)`` is invoked for each
    (a thread falls back to its parent channel's hook), so a test can check what was held while it ran. Errors to raise:
    ``send_error``, ``edit_error``, ``pin_error``. Every call yields to the event loop once, like a real request."""

    def _init_host(self, first_id):
        self.messages, self.sent, self.calls = {}, [], []
        self.on_call = self.parent_host = None
        self.send_error = self.edit_error = self.pin_error = None
        self._ids = first_id

    def next_id(self):
        self._ids += 1
        return self._ids

    def call(self, kind):
        self.calls.append(kind)
        hook = self.on_call or (self.parent_host.on_call if self.parent_host is not None else None)
        if hook:
            hook(kind)

    async def touch(self, kind):
        self.call(kind)
        await asyncio.sleep(0)

    def post(self, content, view=None, **kwargs):
        """Adds a message without a Discord call (the public response of an interaction)."""
        message = FakePosted(self, self.next_id(), content, view, **kwargs)
        self.messages[message.id] = message
        self.sent.append(message)
        return message

    async def send(self, content=None, **kwargs):
        await self.touch("send")
        if self.send_error:
            raise self.send_error
        return self.post(content, **kwargs)

    def get_partial_message(self, ident):
        return FakePartial(self, ident)

    async def fetch_message(self, ident):
        await self.touch("fetch")
        message = self.messages.get(ident)
        if message is None or message.deleted:
            raise not_found("Unknown Message", 10008)
        return message


class FakeGameChannel(FakeMessageHost):
    """A text channel for table games. The first message posted gets id 7001. ``create_thread`` (on the channel or on one
    of its messages) makes a ``FakeTableThread`` unless ``thread_error`` is set; ``threads`` maps thread id -> thread."""

    def __init__(self, ident=100, guild_id=1):
        self.id, self.mention, self.guild_id = ident, f"<#{ident}>", guild_id
        self._init_host(7000)
        self.threads, self.thread_error, self._thread_seq = {}, None, 0

    async def make_thread(self, name, **kwargs):
        await self.touch("create_thread")
        if self.thread_error:
            raise self.thread_error
        self._thread_seq += 1
        thread = FakeTableThread(self.id * 100 + self._thread_seq, self, name, kwargs)
        self.threads[thread.id] = thread
        return thread

    async def create_thread(self, *, name, message=None, **kwargs):
        return await self.make_thread(name, **kwargs)


class FakeTableThread(FakeMessageHost, FakeThread):
    """A thread made by ``FakeGameChannel`` (passes ``isinstance(..., discord.Thread)``). ``edit_calls`` holds the kwargs of each
    ``edit`` (archiving is ``edit(archived=True, locked=True)``); ``thread_edit_error`` makes it raise."""

    def __init__(self, ident, parent, name, create_kwargs):
        FakeThread.__init__(self, ident, parent.id)
        self._init_host(ident * 1000)
        self.parent_host, self.name, self.create_kwargs = parent, name, create_kwargs
        self.archived = self.locked = False
        self.guild = SimpleNamespace(id=parent.guild_id)
        self.edit_calls, self.thread_edit_error = [], None

    async def edit(self, **kwargs):
        await self.touch("thread_edit")
        if self.thread_edit_error:
            raise self.thread_edit_error
        self.edit_calls.append(dict(kwargs))
        self.archived = kwargs.get("archived", self.archived)
        self.locked = kwargs.get("locked", self.locked)
        return self


class FakeGameResponse(FakeResponse):
    def __init__(self, click):
        super().__init__()
        self.click, self.edited, self.modals = click, [], []

    async def send_modal(self, modal):
        """Records the modal a component press opened (``modals``); it counts as the interaction's one response."""
        self.click.call("send_modal")
        await asyncio.sleep(0)
        self.modals.append(modal)
        self._done = True

    async def defer(self, **kwargs):
        self.click.call("defer")
        await super().defer(**kwargs)

    async def send_message(self, content=None, **kwargs):
        self.click.call("send_message")
        await asyncio.sleep(0)
        await super().send_message(content, **kwargs)
        if not kwargs.get("ephemeral"):  # a public response is the table message
            self.click.message = self.click.channel.post(content, **kwargs)

    async def edit_message(self, **kwargs):
        """Editing the message a component sits on (also how an ephemeral picker is replaced); counted among ``sent`` as private."""
        self.click.call("edit_message")
        await asyncio.sleep(0)
        self.edited.append(kwargs)
        self.sent.append((kwargs.get("content"), {**kwargs, "_edit": True}))
        self._done = True


class FakeGameFollowup(FakeFollowup):
    def __init__(self, click, sent):
        super().__init__()
        self.click, self.sent = click, sent

    async def send(self, content=None, **kwargs):
        self.click.call("followup")
        await asyncio.sleep(0)
        self.sent.append((content, kwargs))
        return SimpleNamespace(id=6000 + len(self.sent))


class FakeGameClick(FakeInteraction):
    """A slash command or button press on a table game. ``channel`` is a ``FakeGameChannel`` or ``FakeTableThread``;
    ``message`` is the message a component sits on (``edit_original_response`` edits it). A public ``send_message`` posts into
    the channel and becomes ``message`` (what ``original_response()`` returns). ``on_call(kind)`` sees every Discord call."""

    def __init__(self, bot, user_id, *, guild_id=1, channel=None, message=None, on_call=None):
        super().__init__(guild_id=guild_id, channel_id=None, user_id=user_id, channel=channel)
        self.channel_id, self.client, self.message, self.on_call = channel.id, bot, message, on_call
        self.user.mention = f"<@{user_id}>"
        self.response, self.followup = FakeGameResponse(self), FakeGameFollowup(self, self.followup.sent)

    def call(self, kind):
        if self.on_call:
            self.on_call(kind)

    async def edit_original_response(self, **kwargs):
        self.call("edit_original_response")
        await asyncio.sleep(0)
        self.response.edited.append(kwargs)
        if self.message is not None:
            self.message._alive()
            self.message.apply(kwargs)

    async def original_response(self):
        return self.message


def thread_message(thread, user_id, content="hello", *, bot=False, guild_id=1):
    """A chat message in a table thread as ``on_message`` / ``note_thread_message`` receive it (its id follows the thread's own posts)."""
    return SimpleNamespace(id=thread.next_id(), guild=SimpleNamespace(id=guild_id), channel=thread, content=content,
                           author=SimpleNamespace(id=user_id, bot=bot), webhook_id=None, mentions=[], attachments=[],
                           reference=None, created_at=datetime.now(timezone.utc))
