import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException

from helpers import ShiftedClock

from llmcord_core.auth import AuthService


class AuthRateLimitTests(unittest.IsolatedAsyncioTestCase):
    def service(self, handler):
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(http.aclose)
        app = SimpleNamespace(state=SimpleNamespace(http=http, sessions={}, base_url='https://private.test'))
        return AuthService(app)

    async def test_retries_rate_limit_without_reusing_old_permissions(self):
        responses = [httpx.Response(429, json={'retry_after': 0.01}),
                     httpx.Response(200, json=[{'id': '1', 'permissions': '8'}]),
                     httpx.Response(200, json=[{'id': '1', 'permissions': '0'}])]
        calls = []
        def handler(request):
            calls.append(request)
            return responses.pop(0)
        auth = self.service(handler)
        # Session and token outlive the clock shift below, so only the guild-list cache goes stale.
        auth.app.state.sessions['session'] = {'expires': time.time() + 3600,
            'token_expires': time.time() + 3600, 'access': 'token', 'csrf': 'csrf'}
        with ShiftedClock() as clock:
            await auth.guard('session', 1)
            # PERF-01 / D1: guild lists are cached for 300 s, so revocation only shows up after the TTL.
            # Advance past it so the second guard refetches instead of reusing the stale admin list.
            clock.advance(301)
            with self.assertRaises(HTTPException) as error:
                await auth.guard('session', 1)
        self.assertEqual(error.exception.status_code, 403)
        self.assertEqual(len(calls), 3)

    async def test_obeys_exhausted_bucket_header_before_next_check(self):
        calls = 0
        def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=[], headers={'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset-After': '0.05'})
        auth = self.service(handler)
        await auth.discord_get('/users/@me/guilds', 'Bearer token')
        with patch('llmcord_core.auth.asyncio.sleep', new_callable=AsyncMock) as sleep:
            await auth.discord_get('/users/@me/guilds', 'Bearer token')
        self.assertEqual(calls, 2)
        self.assertGreater(sleep.await_args.args[0], 0)

    async def test_long_rate_limit_fails_closed_without_waiting_or_hammering(self):
        calls = 0
        def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(429, headers={'Retry-After': '60'}, json={})
        auth = self.service(handler)
        for _ in range(2):
            with self.assertRaises(HTTPException) as error:
                await auth.discord_get('/users/@me/guilds', 'Bearer token')
            self.assertEqual(error.exception.status_code, 503)
            self.assertIn('Retry-After', error.exception.headers)
        self.assertEqual(calls, 1)

    async def test_concurrent_checks_are_serialized_per_credential(self):
        active = peak = calls = 0
        async def handler(request):
            nonlocal active, peak, calls
            active += 1
            peak = max(peak, active)
            calls += 1
            await asyncio.sleep(0.01)
            active -= 1
            return httpx.Response(200, json=[])
        auth = self.service(handler)
        await asyncio.gather(*(auth.discord_get('/users/@me/guilds', 'Bearer token') for _ in range(3)))
        self.assertEqual((calls, peak), (3, 1))

    async def test_unauthorized_response_invalidates_only_rejected_session(self):
        auth = self.service(lambda _: httpx.Response(401, json={}))
        auth.app.state.sessions.update(bad={'access': 'bad'}, good={'access': 'good'})
        with self.assertRaises(HTTPException) as error:
            await auth.discord_get('/users/@me/guilds', 'Bearer bad')
        self.assertEqual(error.exception.status_code, 401)
        self.assertEqual(set(auth.app.state.sessions), {'good'})

    async def test_server_failure_does_not_authorize_operation(self):
        auth = self.service(lambda _: httpx.Response(500, json={}))
        with self.assertRaises(HTTPException) as error:
            await auth.discord_get('/users/@me/guilds', 'Bearer token')
        self.assertEqual(error.exception.status_code, 502)


if __name__ == '__main__':
    unittest.main()
