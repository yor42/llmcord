"""REL-02 (part 1): model clients use explicit timeout / max_retries from the profile config.

Settings are built through the real `load_settings` on a temp YAML so the tests fail on
behavior (SDK defaults of 600 s / 2 retries) rather than on a missing constructor argument.
No network: the hanging-provider test patches `httpx.AsyncHTTPTransport.handle_async_request`
(the default transport both SDKs build) with a fake that honours the request's read timeout.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import anthropic
import httpx
import openai

from llmcord_core.config import load_settings
from llmcord_core.models import ModelGateway, TurnMessage

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


def seconds(timeout) -> float | None:
    """SDK clients store a float or an httpx.Timeout; compare on the read timeout."""
    return timeout.read if isinstance(timeout, httpx.Timeout) else timeout


class ProfileTimeoutConfigTests(unittest.TestCase):
    def test_profile_without_keys_gets_bounded_defaults(self):
        """REL-02 (fixed): a profile with no timeout keys defaults to timeout_seconds=120.0 and max_retries=1."""
        profile = settings_for().profile("dialogue")
        self.assertEqual(getattr(profile, "timeout_seconds", None), 120.0)
        self.assertEqual(getattr(profile, "max_retries", None), 1)

    def test_profile_reads_explicit_timeout_and_retries(self):
        """REL-02 (fixed): models.profiles.<name>.timeout_seconds / max_retries are read from config YAML."""
        profile = settings_for(extra="      timeout_seconds: 45.5\n      max_retries: 0\n").profile("dialogue")
        self.assertEqual(getattr(profile, "timeout_seconds", None), 45.5)
        self.assertEqual(getattr(profile, "max_retries", None), 0)

    def test_invalid_timeout_or_retries_rejected_at_load(self):
        """REL-02 (fixed): non-positive / non-finite / non-numeric timeout_seconds and negative / non-int max_retries raise ValueError at load."""
        bad = {
            "zero timeout": "      timeout_seconds: 0\n",
            "negative timeout": "      timeout_seconds: -5\n",
            "non-numeric timeout": "      timeout_seconds: soon\n",
            "negative retries": "      max_retries: -1\n",
            "nan timeout": "      timeout_seconds: .nan\n",
            "infinite timeout": "      timeout_seconds: .inf\n",
            "bool timeout": "      timeout_seconds: true\n",
            "bool retries": "      max_retries: true\n",
            "fractional retries": "      max_retries: 1.5\n",
            "string retries": "      max_retries: \"2\"\n",
        }
        for label, extra in bad.items():
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    settings_for(extra=extra)


class ClientConstructionTests(unittest.IsolatedAsyncioTestCase):
    async def client_for(self, provider: str, extra: str = ""):
        settings = settings_for(provider, extra)
        gateway = ModelGateway(settings)
        with patch.dict(os.environ, ENV, clear=True):
            client = gateway._client("main")
        self.addAsyncCleanup(gateway.close)
        return client

    async def test_sdk_clients_use_profile_timeout_and_retries(self):
        """REL-02 (fixed): openai / compatible / anthropic SDK clients are built with the profile's timeout and max_retries."""
        for provider in ("compatible", "openai", "anthropic"):
            with self.subTest(provider):
                client = await self.client_for(provider, "      timeout_seconds: 7.5\n      max_retries: 3\n")
                self.assertEqual(seconds(client.timeout), 7.5)
                self.assertEqual(client.max_retries, 3)

    async def test_sdk_clients_default_to_bounded_timeout(self):
        """REL-02 (fixed): with no keys the SDK client gets 120 s / 1 retry, not the SDK's 600 s / 2 retries."""
        for provider in ("compatible", "anthropic"):
            with self.subTest(provider):
                client = await self.client_for(provider)
                self.assertEqual(seconds(client.timeout), 120.0)
                self.assertEqual(client.max_retries, 1)


class HangingProviderTests(unittest.IsolatedAsyncioTestCase):
    """A provider that never answers must surface as an exception from the model call."""

    async def run_hanging_call(self, provider: str, expected: type[Exception]):
        attempts = []

        async def hanging(transport, request: httpx.Request):
            # Emulate a server that accepts the request but never sends a response:
            # wait out the client's read timeout, then fail the way httpcore does.
            attempts.append(request.url.host)
            await asyncio.sleep(request.extensions["timeout"]["read"])
            raise httpx.ReadTimeout("fake provider never responded", request=request)

        settings = settings_for(provider, "      timeout_seconds: 0.05\n      max_retries: 0\n")
        gateway = ModelGateway(settings)
        self.addAsyncCleanup(gateway.close)
        started = time.monotonic()
        with patch.dict(os.environ, ENV, clear=True), \
                patch.object(httpx.AsyncHTTPTransport, "handle_async_request", hanging), \
                self.assertRaises(expected):
            # The outer 1 s bound keeps the test finite; it raises asyncio.TimeoutError, not `expected`.
            await asyncio.wait_for(gateway.text("dialogue", "rules", [TurnMessage("user", "hi")]), 1.0)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 1.0)
        self.assertEqual(len(attempts), 1, "max_retries=0 should make exactly one attempt")

    async def test_hanging_compatible_provider_raises_timeout(self):
        """REL-02 (fixed): a hung compatible (OpenAI SDK) call raises APITimeoutError within the profile timeout."""
        await self.run_hanging_call("compatible", openai.APITimeoutError)

    async def test_hanging_anthropic_provider_raises_timeout(self):
        """REL-02 (fixed): a hung Anthropic call raises APITimeoutError within the profile timeout."""
        await self.run_hanging_call("anthropic", anthropic.APITimeoutError)


if __name__ == "__main__":
    unittest.main()
