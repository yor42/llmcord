"""PERF-01 / roadmap R2.1: per-session Discord guild-list cache in ``AuthService`` (TTL 300 s, decision D1).

Call-count tests guarantee the cache saves Discord calls (one guild-list fetch per session per TTL, concurrent
cold guards coalesced). The other tests pin authorization invariants the cache keeps: per-session isolation,
revocation within the TTL, immediate invalidation on 401 / sign-out / expiry / failed refresh, errors never cached, and cheap
cookie/csrf/origin checks on every call. Time is shifted with ``ShiftedClock``; nothing sleeps.
"""
import asyncio
import unittest
from collections import Counter

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient

from helpers import ShiftedClock, discord_transport, install_session

from llmcord_core.web import create_app

GUILDS = "GET /users/@me/guilds"
TTL = 300


def make_app(transport):
    return create_app(":memory:", "https://pi.test", "client", "secret", "bot",
                      httpx.AsyncClient(transport=transport), enable_dashboard=False,
                      config_path="tests/nonexistent-config.yaml")


class GuardCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls = Counter()
        self.admin_guilds = [1]
        self.failures = []
        transport, _ = discord_transport(admin_guilds=self.admin_guilds, calls=self.calls,
                                         guild_failures=self.failures)
        self.app = make_app(transport)
        self.auth = self.app.state.auth
        self.session = install_session(self.app)

    async def asyncTearDown(self):
        await self.app.state.http.aclose()
        self.app.state.store.close()

    async def assertStatus(self, status, awaitable):
        with self.assertRaises(HTTPException) as caught:
            await awaitable
        self.assertEqual(caught.exception.status_code, status)

    # --- call-count guarantees: one guild-list fetch per session per TTL ---

    async def test_repeat_guards_within_ttl_fetch_guilds_once(self):
        """PERF-01 (fixed): repeated guards for one session within the TTL (allowed and denied, any guild) cost one
        Discord guild-list call in total."""
        await self.auth.guard(self.session, 1)
        await self.auth.guard(self.session, 1, csrf="csrf", origin="https://pi.test")
        await self.assertStatus(403, self.auth.guard(self.session, 2))
        await self.auth.guard(self.session, 1)
        await self.assertStatus(403, self.auth.guard(self.session, 3))
        self.assertEqual(self.calls[GUILDS], 1)

    async def test_repeat_guards_just_before_ttl_fetch_once(self):
        """PERF-01 (fixed): a guard 299 s after the first one is still served from the per-session cache (D1: 300 s)."""
        with ShiftedClock() as clock:
            await self.auth.guard(self.session, 1)
            clock.advance(TTL - 1)
            await self.auth.guard(self.session, 1)
        self.assertEqual(self.calls[GUILDS], 1)

    async def test_concurrent_cold_guards_coalesce_into_one_call(self):
        """PERF-01 (fixed): five concurrent guards for one session with a cold cache cost exactly one Discord call
        (the guild-list mock yields once so the guards really overlap on the session lock)."""
        transport, _ = discord_transport(admin_guilds=self.admin_guilds, calls=self.calls, yield_on_guilds=True)
        await self.app.state.http.aclose()
        self.app.state.http = httpx.AsyncClient(transport=transport)
        results = await asyncio.gather(*(self.auth.guard(self.session, 1) for _ in range(5)))
        self.assertEqual(len(results), 5)
        self.assertEqual(self.calls[GUILDS], 1)

    # --- authorization invariants the cache keeps ---

    async def test_guard_allows_admin_guild_and_denies_others(self):
        session = await self.auth.guard(self.session, 1)
        self.assertIs(session, self.app.state.sessions[self.session])
        await self.assertStatus(403, self.auth.guard(self.session, 2))

    async def test_removed_admin_loses_access_after_ttl(self):
        """PERF-01 / D1: once the TTL has elapsed the next guard refetches, so revocation takes at most 300 s."""
        with ShiftedClock() as clock:
            await self.auth.guard(self.session, 1)
            self.admin_guilds.remove(1)
            clock.advance(TTL + 1)
            await self.assertStatus(403, self.auth.guard(self.session, 1))
        self.assertEqual(self.calls[GUILDS], 2)

    async def test_newly_granted_guild_appears_after_ttl(self):
        """PERF-01 / D1: a guild granted on Discord is usable after the TTL without signing in again."""
        with ShiftedClock() as clock:
            await self.auth.guard(self.session, 1)
            self.admin_guilds.append(2)
            clock.advance(TTL + 1)
            await self.auth.guard(self.session, 2)

    async def test_expired_session_rejected_without_discord_call_even_when_warm(self):
        """PERF-01: session expiry invalidates immediately; no Discord call is made for an expired session."""
        await self.auth.guard(self.session, 1)
        self.app.state.sessions[self.session]["expires"] = 0
        await self.assertStatus(401, self.auth.guard(self.session, 1))
        self.assertEqual(self.calls[GUILDS], 1)

    async def test_expired_session_cold_rejected_without_discord_call(self):
        self.app.state.sessions[self.session]["expires"] = 0
        await self.assertStatus(401, self.auth.guard(self.session, 1))
        self.assertEqual(self.calls[GUILDS], 0)

    async def test_discord_401_invalidates_session_and_its_cached_guilds(self):
        """PERF-01: a Discord 401 pops the session; a later session under the same ident does not see the
        old guild list."""
        with ShiftedClock() as clock:
            await self.auth.guard(self.session, 1)
            clock.advance(TTL + 1)
            self.failures.append(401)
            await self.assertStatus(401, self.auth.guard(self.session, 1))
            self.assertNotIn(self.session, self.app.state.sessions)
            await self.assertStatus(401, self.auth.guard(self.session, 1))
            self.admin_guilds[:] = [2]
            install_session(self.app, self.session)
            await self.auth.guard(self.session, 2)
            await self.assertStatus(403, self.auth.guard(self.session, 1))

    async def test_queued_guards_after_401_do_not_refetch(self):
        """PERF-01 (fixed): when a cold-cache guild fetch gets a Discord 401, guards queued on the same session's
        lock fail 401 without calling Discord again (3 guards cost 1 call)."""
        transport, _ = discord_transport(admin_guilds=self.admin_guilds, calls=self.calls,
                                         guild_failures=[401], yield_on_guilds=True)
        await self.app.state.http.aclose()
        self.app.state.http = httpx.AsyncClient(transport=transport)
        results = await asyncio.gather(*(self.auth.guard(self.session, 1) for _ in range(3)),
                                       return_exceptions=True)
        self.assertEqual([getattr(r, "status_code", r) for r in results], [401, 401, 401])
        self.assertTrue(all(isinstance(r, HTTPException) for r in results))
        self.assertEqual(self.calls[GUILDS], 1)

    async def test_failed_token_refresh_invalidates_cached_guilds(self):
        """PERF-01: a failed token refresh drops the session and its cache immediately."""
        transport, _ = discord_transport(admin_guilds=self.admin_guilds, calls=self.calls, token_status=400)
        await self.app.state.http.aclose()
        self.app.state.http = httpx.AsyncClient(transport=transport)
        await self.auth.guard(self.session, 1)
        self.app.state.sessions[self.session]["token_expires"] = 0
        await self.assertStatus(401, self.auth.guard(self.session, 1))
        self.assertNotIn(self.session, self.app.state.sessions)
        self.admin_guilds[:] = [2]
        install_session(self.app, self.session)
        await self.auth.guard(self.session, 2)
        await self.assertStatus(403, self.auth.guard(self.session, 1))

    async def test_new_sign_in_does_not_reuse_another_sessions_cache(self):
        """PERF-01: the cache is per session; a brand-new session (new ident, even with the same access
        token) fetches its own guild list."""
        await self.auth.guard(self.session, 1)
        self.admin_guilds[:] = [2]
        other = install_session(self.app, "fresh-session")
        await self.auth.guard(other, 2)
        await self.assertStatus(403, self.auth.guard(other, 1))
        self.assertGreaterEqual(self.calls[GUILDS], 2)

    async def test_errors_are_not_cached(self):
        """PERF-01: 403 / 5xx / 429 / network failures propagate and the next guard retries Discord."""
        cases = {403: [403], 502: [500], 503: [429, 429, 429], "network": ["network"]}
        for name, failures in cases.items():
            with self.subTest(failure=name):
                ident = install_session(self.app, f"session-{name}")
                self.failures[:] = failures
                expected = 503 if name == "network" else name
                await self.assertStatus(expected, self.auth.guard(ident, 1))
                before = self.calls[GUILDS]
                self.assertIs(await self.auth.guard(ident, 1), self.app.state.sessions[ident])
                self.assertEqual(self.calls[GUILDS], before + 1)

    async def test_cookie_csrf_and_origin_checked_every_call_without_discord(self):
        """PERF-01: binding checks stay per call and never need Discord, warm cache or not."""
        await self.auth.guard(self.session, 1, csrf="csrf", origin="https://pi.test")
        await self.assertStatus(403, self.auth.guard(self.session, 1, csrf="wrong"))
        await self.assertStatus(403, self.auth.guard(self.session, 1, csrf="csrf", origin="https://evil.test"))
        await self.assertStatus(401, self.auth.guard("unknown-session", 1))
        await self.assertStatus(401, self.auth.guard("", 1))
        self.assertEqual(self.calls[GUILDS], 1)


class PerSessionIsolationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls = Counter()
        transport, _ = discord_transport(calls=self.calls, guilds_by_token={"token-a": (1,), "token-b": (2,)})
        self.app = make_app(transport)
        self.auth = self.app.state.auth
        self.a = install_session(self.app, "session-a", user_id="10", access="token-a")
        self.b = install_session(self.app, "session-b", user_id="20", access="token-b")

    async def asyncTearDown(self):
        await self.app.state.http.aclose()
        self.app.state.store.close()

    async def run_both(self):
        results = {}
        for ident in (self.a, self.b):
            for guild_id in (1, 2, 1, 2):
                try:
                    await self.auth.guard(ident, guild_id)
                    results.setdefault(ident, []).append((guild_id, "ok"))
                except HTTPException as error:
                    results.setdefault(ident, []).append((guild_id, error.status_code))
        return results

    async def test_sessions_never_share_guild_lists(self):
        """PERF-01: two sessions with different tokens each get their own guild list."""
        results = await self.run_both()
        self.assertEqual(results[self.a], [(1, "ok"), (2, 403), (1, "ok"), (2, 403)])
        self.assertEqual(results[self.b], [(1, 403), (2, "ok"), (1, 403), (2, "ok")])
        self.assertGreaterEqual(self.calls[f"{GUILDS} token-a"], 1)
        self.assertGreaterEqual(self.calls[f"{GUILDS} token-b"], 1)

    async def test_two_sessions_each_fetch_once(self):
        """PERF-01 (fixed): with a per-session cache each session fetches its guild list exactly once."""
        await self.run_both()
        self.assertEqual(self.calls[f"{GUILDS} token-a"], 1)
        self.assertEqual(self.calls[f"{GUILDS} token-b"], 1)


class HttpCacheTests(unittest.TestCase):
    def setUp(self):
        self.calls = Counter()
        self.admin_guilds = [1]
        transport, _ = discord_transport(admin_guilds=self.admin_guilds, calls=self.calls)
        self.app = make_app(transport)
        self.client = TestClient(self.app, base_url="https://pi.test")
        self.client.__enter__()
        store = self.app.state.store
        world = store.create_space(1, "World", "world")
        self.character = store.add_character(1, world, "Alice", {"name": "Alice"}, None, [])
        from io import BytesIO

        from PIL import Image

        from llmcord_core.avatars import normalize_avatar
        png = BytesIO()
        Image.new("RGB", (8, 8)).save(png, "PNG")
        store.db.execute("UPDATE characters SET avatar=? WHERE id=?", (normalize_avatar(png.getvalue()), self.character))
        store.db.commit()
        self.session = install_session(self.app)
        self.client.cookies.set("llmcord_session", self.session)

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_index_then_guild_page_costs_one_guild_call(self):
        """PERF-01 (fixed): the legacy guild picker (`GET /`) and the guild page share the session cache."""
        index = self.client.get("/")
        self.assertEqual(index.status_code, 200)
        self.assertIn("Guild 1", index.text)
        self.assertEqual(self.client.get("/guild/1").status_code, 200)
        self.assertEqual(self.calls[GUILDS], 1)

    def test_guild_page_then_index_costs_one_guild_call(self):
        """PERF-01 (fixed): a guard warms the cache for the guild picker too."""
        self.assertEqual(self.client.get("/guild/1").status_code, 200)
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.calls[GUILDS], 1)

    def test_logout_clears_cached_guilds(self):
        """PERF-01: sign-out invalidates immediately; a later session under the same ident refetches."""
        avatar = f"/guild/1/characters/{self.character}/avatar"
        self.assertEqual(self.client.get(avatar).status_code, 200)
        self.assertEqual(self.client.post("/logout", data={"csrf": "csrf"}, follow_redirects=False).status_code, 303)
        self.assertNotIn(self.session, self.app.state.sessions)
        self.assertEqual(self.client.get(avatar).status_code, 401)
        self.admin_guilds[:] = [2]
        install_session(self.app, self.session)
        self.client.cookies.set("llmcord_session", self.session)
        self.assertEqual(self.client.get(avatar).status_code, 403)
        self.assertEqual(self.calls[GUILDS], 2)

    def test_removed_admin_loses_http_access_after_ttl(self):
        """PERF-01 / D1: revocation reaches HTTP routes within the TTL."""
        avatar = f"/guild/1/characters/{self.character}/avatar"
        with ShiftedClock() as clock:
            self.assertEqual(self.client.get(avatar).status_code, 200)
            self.admin_guilds.clear()
            clock.advance(TTL + 1)
            self.assertEqual(self.client.get(avatar).status_code, 403)


if __name__ == "__main__":
    unittest.main()
