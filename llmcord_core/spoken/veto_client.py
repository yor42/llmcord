"""FEAT-49a: async client for the decision-model veto (see veto.py). Not wired into the bot yet.

The host is fixed in code, redirects are not followed, there are no retries, and a circuit breaker stops calls after repeated
failures. Errors and logs carry at most a status code or an exception class name, never the key, the message or the response body.
"""
import json
import logging
import time

import httpx

from . import veto
from .veto import VetoError

log = logging.getLogger(__name__)

ENV_SWITCH = "LLMCORD_SPOKEN_VETO"
ENV_VALUE = "openrouter-d1"
ENV_KEY = "OPENROUTER_API_KEY"


class VetoClient:
    def __init__(self, api_key, *, transport=None, clock=time.monotonic, timeout=1.0, connect_timeout=0.3, max_bytes=65536, fail_limit=5, cooldown=60.0):
        self._http = httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(timeout, connect=connect_timeout), follow_redirects=False,
                                       headers={"Authorization": f"Bearer {api_key}"})
        self._clock, self._max_bytes, self._fail_limit, self._cooldown = clock, max_bytes, fail_limit, cooldown
        self._fails, self._opened_at, self._closed = 0, None, False

    @property
    def breaker_open(self):
        return self._opened_at is not None and self._clock() - self._opened_at < self._cooldown

    async def ask(self, text, legal):
        """{move: probability} for the legal moves; raises VetoError on bad input, an open breaker, or any failed call."""
        body = veto.build_request(text, legal)
        if self._closed:
            raise VetoError("client closed")
        if self.breaker_open:
            raise VetoError("breaker open")
        try:
            probs = veto.parse_response(await self._post(body), legal)
        except VetoError:
            self._fail()
            raise
        self._fails, self._opened_at = 0, None
        return probs

    async def _post(self, body):
        try:
            async with self._http.stream("POST", veto.ORIGIN + veto.PATH, json=body) as resp:
                if resp.status_code != 200:
                    log.warning("veto call failed: status %s", resp.status_code)
                    raise VetoError(f"status {resp.status_code}")
                raw = b""
                async for chunk in resp.aiter_bytes():
                    raw += chunk
                    if len(raw) > self._max_bytes:
                        log.warning("veto call failed: response too large")
                        raise VetoError("response too large")
        except VetoError:
            raise
        except Exception as e:
            log.warning("veto call failed: %s", type(e).__name__)
            raise VetoError(f"request failed: {type(e).__name__}") from None
        try:
            return json.loads(raw)
        except ValueError:
            log.warning("veto call failed: invalid json")
            raise VetoError("invalid json") from None

    def _fail(self):
        self._fails += 1
        if self._fails >= self._fail_limit:
            self._opened_at = self._clock()

    async def close(self):
        self._closed = True
        await self._http.aclose()


def from_env(env, **kw):
    """A client when env turns the veto on and holds a printable-ASCII key, else None (never raises). Reads only the mapping it is given."""
    key = (env.get(ENV_KEY) or "").strip()
    if env.get(ENV_SWITCH) != ENV_VALUE or not key or any(not 0x21 <= ord(c) <= 0x7E for c in key):
        return None
    return VetoClient(key, **kw)
