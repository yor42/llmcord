"""SEC-04 / roadmap R2.6: in-memory auth state in ``AuthService`` must stay bounded.

``app.state.sessions``, ``refresh_locks`` (keyed by session ident), ``request_locks`` and ``retry_at`` (keyed by a
Discord-token-derived key) used to grow forever. The intended behaviour pinned here:

1. an ordinary access to the auth service after a session's ``expires`` drops that session (valid ones survive);
2. removing a session (logout, Discord 401, failed refresh, expiry prune) drops its ``refresh_locks`` entry;
3. ``request_locks`` / ``retry_at`` entries for keys no live session uses are dropped once their ``retry_at`` passed,
   while a pending future ``retry_at`` survives so the rate-limit backoff is kept;
4. a held lock is never pruned in a way that lets two refreshes or two requests for one key run at once.

The implementer may prune on access in ``AuthService`` or sweep on new sign-in, so ``touch()`` does both: a
``session()`` lookup for a valid session plus a full OAuth sign-in through the real routes. State dicts are only
inspected for size / key membership. Time is shifted with ``ShiftedClock``; nothing sleeps.
"""
import asyncio
import time
import unittest
from collections import Counter

import httpx
from fastapi import HTTPException

from helpers import ShiftedClock, discord_transport, install_session

from llmcord_core.auth import token_key
from llmcord_core.web import create_app

GUILDS = "GET /users/@me/guilds"


class GatedTransport:
    """Wraps a ``discord_transport`` so matching requests wait on ``release`` while counting overlap."""

    def __init__(self, inner, matches):
        self.inner, self.matches = inner, matches
        self.armed = False
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.count = self.active = self.peak = 0

    async def handler(self, request):
        if self.armed and self.matches(request):
            self.count += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.entered.set()
            try:
                await self.release.wait()
            finally:
                self.active -= 1
        return self.inner.handler(request)


def is_refresh(request):
    return request.url.path.endswith("/oauth2/token") and b"grant_type=refresh_token" in request.content


def is_guild_list(request):
    return request.url.path.endswith("/users/@me/guilds")


class PruneTestCase(unittest.IsolatedAsyncioTestCase):
    transport_options = {}

    def setUp(self):
        self.calls = Counter()
        self.failures = []
        self.inner, _ = discord_transport(calls=self.calls, guild_failures=self.failures, **self.transport_options)
        self.gate = GatedTransport(self.inner, is_refresh)
        self.app = create_app(":memory:", "https://pi.test", "client", "secret", "bot",
                              httpx.AsyncClient(transport=httpx.MockTransport(self.gate.handler)),
                              enable_dashboard=False, config_path="tests/nonexistent-config.yaml")
        self.auth = self.app.state.auth
        self.sessions = self.app.state.sessions

    async def asyncSetUp(self):
        # ShiftedClock jumps make debug-mode asyncio report the running task as "slow"; silence that noise.
        asyncio.get_running_loop().slow_callback_duration = float("inf")

    async def asyncTearDown(self):
        self.gate.release.set()
        await self.app.state.http.aclose()
        self.app.state.store.close()

    def web(self):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="https://pi.test")

    async def login(self):
        """Full OAuth sign-in through ``/login`` + ``/auth/callback``; returns the new session ident."""
        async with self.web() as client:
            state = (await client.get("/login")).cookies["llmcord_oauth_state"]
            response = await client.get("/auth/callback", params={"code": "code", "state": state})
            self.assertEqual(response.status_code, 303, response.text)
            return response.cookies["llmcord_session"]

    async def logout(self, ident):
        async with self.web() as client:
            client.cookies.set("llmcord_session", ident)
            response = await client.post("/logout", data={"csrf": self.sessions[ident]["csrf"]})
            self.assertEqual(response.status_code, 303, response.text)

    async def touch(self, live):
        """An ordinary access (valid session lookup) plus a new sign-in: whichever prunes, one of them runs."""
        self.assertIs(await self.auth.session(live), self.sessions[live])
        return await self.login()

    def short_lived(self, ident, access, seconds=60):
        install_session(self.app, ident, access=access)
        self.sessions[ident]["expires"] = time.time() + seconds
        return ident

    async def refreshed(self, ident):
        """Force a token refresh so ``refresh_locks`` gets an entry for ``ident``."""
        self.sessions[ident]["token_expires"] = 0
        await self.auth.session(ident)


class SessionExpiryTests(PruneTestCase):
    async def test_expired_sessions_pruned_on_next_access(self):
        """SEC-04 (fixed): sessions whose ``expires`` passed are removed by the next ordinary auth access / sign-in."""
        with ShiftedClock() as clock:
            live = install_session(self.app, "live", access="tok-live")
            self.short_lived("old-a", "tok-a")
            self.short_lived("old-b", "tok-b")
            clock.advance(120)
            await self.touch(live)
        self.assertNotIn("old-a", self.sessions)
        self.assertNotIn("old-b", self.sessions)
        self.assertIn(live, self.sessions)

    async def test_valid_sessions_survive_pruning(self):
        """Characterization (SEC-04): pruning never drops an unexpired session; each still passes a guard afterwards."""
        with ShiftedClock() as clock:
            live = [install_session(self.app, f"live-{i}", access=f"tok-{i}") for i in range(3)]
            self.short_lived("old", "tok-old")
            clock.advance(120)
            fresh = await self.touch(live[0])
            for ident in [*live, fresh]:
                self.assertIs(await self.auth.guard(ident, 1), self.sessions[ident])

    async def test_session_churn_stays_bounded(self):
        """SEC-04 (fixed): after many short sessions expire, every auth dict is bounded by the live sessions."""
        with ShiftedClock() as clock:
            live = install_session(self.app, "live", access="tok-live")
            for i in range(10):
                ident = self.short_lived(f"old-{i}", f"tok-{i}")
                await self.auth.guard(ident, 1)
                await self.refreshed(ident)
            clock.advance(120)
            await self.touch(live)
        self.assertEqual(len(self.sessions), 2)  # "live" plus the sign-in done by touch()
        self.assertLessEqual(len(self.auth.refresh_locks), len(self.sessions))
        self.assertLessEqual(len(self.auth.request_locks), 2 * len(self.sessions))
        self.assertLessEqual(len(self.auth.retry_at), 2 * len(self.sessions))


class RefreshLockRemovalTests(PruneTestCase):
    def setUp(self):
        super().setUp()
        self.ident = install_session(self.app, "victim", access="tok-victim")

    async def test_logout_drops_refresh_lock(self):
        """SEC-04 (fixed): signing out removes the session's ``refresh_locks`` entry."""
        await self.refreshed(self.ident)
        await self.logout(self.ident)
        self.assertNotIn(self.ident, self.sessions)
        self.assertNotIn(self.ident, self.auth.refresh_locks)

    async def test_discord_401_drops_refresh_lock(self):
        """SEC-04 (fixed): a Discord 401 that invalidates the session also removes its ``refresh_locks`` entry."""
        await self.refreshed(self.ident)
        self.failures.append(401)
        with self.assertRaises(HTTPException) as caught:
            await self.auth.guard(self.ident, 1)
        self.assertEqual(caught.exception.status_code, 401)
        self.assertNotIn(self.ident, self.sessions)
        self.assertNotIn(self.ident, self.auth.refresh_locks)

    async def test_expiry_prune_drops_refresh_lock(self):
        """SEC-04 (fixed): pruning an expired session removes its ``refresh_locks`` entry too."""
        with ShiftedClock() as clock:
            live = install_session(self.app, "live", access="tok-live")
            await self.refreshed(self.ident)
            self.sessions[self.ident]["expires"] = time.time() + 60
            clock.advance(120)
            await self.touch(live)
        self.assertNotIn(self.ident, self.sessions)
        self.assertNotIn(self.ident, self.auth.refresh_locks)


class FailedRefreshLockTests(PruneTestCase):
    transport_options = {"token_status": 400}

    async def test_failed_refresh_drops_refresh_lock(self):
        """SEC-04 (fixed): a failed token refresh drops the session and its ``refresh_locks`` entry."""
        ident = install_session(self.app, "victim", access="tok-victim")
        self.sessions[ident]["token_expires"] = 0
        with self.assertRaises(HTTPException) as caught:
            await self.auth.session(ident)
        self.assertEqual(caught.exception.status_code, 401)
        self.assertNotIn(ident, self.sessions)
        self.assertNotIn(ident, self.auth.refresh_locks)


class OrphanRequestStateTests(PruneTestCase):
    # Every answer carries an exhausted bucket with Reset-After 0, so each Discord call writes a ``retry_at``
    # entry whose time has already passed by the next clock tick.
    transport_options = {"rate_limited": True}

    async def keys_used_by(self, ident):
        before = set(self.auth.request_locks) | set(self.auth.retry_at)
        await self.auth.guard(ident, 1)
        keys = (set(self.auth.request_locks) | set(self.auth.retry_at)) - before
        self.assertTrue(keys)
        return keys

    def assertPruned(self, keys):
        self.assertTrue(keys.isdisjoint(self.auth.request_locks), "request_locks kept a dead key")
        self.assertTrue(keys.isdisjoint(self.auth.retry_at), "retry_at kept a dead key")

    async def test_expired_sessions_request_state_pruned(self):
        """SEC-04 (fixed): request locks / retry times of an expired session's token go once its retry time passed."""
        with ShiftedClock() as clock:
            live = install_session(self.app, "live", access="tok-live")
            keys = await self.keys_used_by(self.short_lived("old", "tok-old"))
            clock.advance(120)
            await self.touch(live)
        self.assertPruned(keys)

    async def test_signed_out_sessions_request_state_pruned(self):
        """SEC-04 (fixed): request locks / retry times of a signed-out session's token are pruned on later access."""
        with ShiftedClock() as clock:
            live = install_session(self.app, "live", access="tok-live")
            gone = install_session(self.app, "gone", access="tok-gone")
            keys = await self.keys_used_by(gone)
            await self.logout(gone)
            clock.advance(1)
            await self.touch(live)
        self.assertPruned(keys)

    async def test_replaced_access_token_request_state_pruned(self):
        """SEC-04 (fixed): after a token refresh no live session uses the old access token, so its entries are pruned."""
        with ShiftedClock() as clock:
            live = install_session(self.app, "live", access="tok-live")
            ident = install_session(self.app, "rotating", access="tok-before-refresh")
            keys = await self.keys_used_by(ident)
            await self.refreshed(ident)
            clock.advance(1)
            await self.touch(live)
        self.assertNotEqual(self.sessions[ident]["access"], "tok-before-refresh")
        self.assertPruned(keys)

    async def test_live_sessions_request_state_survives(self):
        """Characterization (SEC-04): a live session's own token keeps working (and stays rate-limit aware) after pruning."""
        with ShiftedClock() as clock:
            live = install_session(self.app, "live", access="tok-live")
            await self.auth.guard(live, 1)
            self.short_lived("old", "tok-old")
            clock.advance(301)  # also past the guild cache TTL, so the next guard calls Discord again
            await self.touch(live)
            await self.auth.guard(live, 1)
        self.assertEqual(self.calls[GUILDS], 2)


class PendingBackoffTests(PruneTestCase):
    transport_options = {"retry_after": 60}

    async def rate_limited_keys(self):
        ident = self.short_lived("old", "tok-old", seconds=10)
        self.failures.append(429)
        with self.assertRaises(HTTPException) as caught:
            await self.auth.guard(ident, 1)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(self.calls[GUILDS], 1)
        return set(self.auth.retry_at)

    async def test_pending_backoff_survives_prune(self):
        """Characterization (SEC-04): a future ``retry_at`` survives pruning even when its session is gone, so a new
        request with that token still fails closed without calling Discord."""
        with ShiftedClock() as clock:
            live = install_session(self.app, "live", access="tok-live")
            keys = await self.rate_limited_keys()
            clock.advance(30)  # session expired, backoff (60 s) still pending
            await self.touch(live)
            self.assertLessEqual(keys, set(self.auth.retry_at))
            with self.assertRaises(HTTPException) as caught:
                await self.auth.discord_get("/users/@me/guilds", "Bearer tok-old")
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(self.calls[GUILDS], 1)

    async def test_backoff_pruned_after_retry_time_passes(self):
        """SEC-04 (fixed): once the dead token's backoff has elapsed its ``retry_at`` entry is pruned."""
        with ShiftedClock() as clock:
            live = install_session(self.app, "live", access="tok-live")
            keys = await self.rate_limited_keys()
            clock.advance(120)
            await self.touch(live)
        self.assertTrue(keys.isdisjoint(self.auth.retry_at))
        self.assertTrue(keys.isdisjoint(self.auth.request_locks))


class HeldLockTests(PruneTestCase):
    async def test_concurrent_refreshes_coalesce_across_prune(self):
        """Characterization (SEC-04): a prune pass (which drops another, expired session's idle refresh lock) while a
        live session's refresh is in flight does not let a second refresh start; later concurrent refreshes still
        coalesce into one token call each. The live session's lock is kept because the session is live, not via
        ``busy()``; ``test_dropped_session_keeps_busy_refresh_lock`` covers ``busy()`` on refresh locks."""
        with ShiftedClock() as clock:
            live = install_session(self.app, "live", access="tok-live")
            target = install_session(self.app, "target", access="tok-target")
            dying = self.short_lived("dying", "tok-dying")
            await self.refreshed(dying)  # gives the prune pass a refresh lock to drop
            self.sessions[target]["token_expires"] = 0
            self.gate.armed = True
            first = asyncio.create_task(self.auth.session(target))
            await asyncio.wait_for(self.gate.entered.wait(), 1)  # first refresh holds the lock, waiting on Discord
            clock.advance(120)
            await self.touch(live)
            second = asyncio.create_task(self.auth.session(target))
            for _ in range(10):
                await asyncio.sleep(0)
            self.assertEqual(self.gate.peak, 1)
            self.gate.release.set()
            results = await asyncio.gather(first, second)
            self.assertTrue(all(r is self.sessions[target] for r in results))
            self.assertEqual(self.gate.count, 1)

            # After the prune pass, a fresh burst of concurrent refreshes still costs one token call.
            self.sessions[target]["token_expires"] = 0
            results = await asyncio.gather(*(self.auth.session(target) for _ in range(3)))
        self.assertTrue(all(r is self.sessions[target] for r in results))
        self.assertEqual(self.gate.count, 2)
        self.assertEqual(self.gate.peak, 1)

    async def test_held_request_lock_survives_prune(self):
        """Characterization (SEC-04): a request lock held by an in-flight Discord call (here a token no session owns yet,
        as during sign-in) is not pruned out from under it; a second call for that key waits its turn."""
        self.gate.matches = is_guild_list
        with ShiftedClock() as clock:
            live = install_session(self.app, "live", access="tok-live")
            self.short_lived("dying", "tok-dying")
            self.gate.armed = True
            first = asyncio.create_task(self.auth.discord_get("/users/@me/guilds", "Bearer orphan"))
            await asyncio.wait_for(self.gate.entered.wait(), 1)
            clock.advance(120)
            await self.touch(live)
            second = asyncio.create_task(self.auth.discord_get("/users/@me/guilds", "Bearer orphan"))
            for _ in range(10):
                await asyncio.sleep(0)
            self.assertEqual(self.gate.peak, 1)
            self.gate.release.set()
            await asyncio.gather(first, second)
        self.assertEqual((self.gate.count, self.gate.peak), (2, 1))


    async def test_dropped_session_keeps_busy_refresh_lock(self):
        """Characterization (SEC-04): signing a session out while its refresh holds the lock and another refresh waits
        keeps the ``refresh_locks`` entry until the lock is idle; only one refresh POST runs and both callers get 401.
        The next prune pass then drops the idle entry."""
        target = install_session(self.app, "target", access="tok-target")
        self.sessions[target]["token_expires"] = 0
        self.gate.armed = True
        first = asyncio.create_task(self.auth.session(target))
        await asyncio.wait_for(self.gate.entered.wait(), 1)
        second = asyncio.create_task(self.auth.session(target))
        for _ in range(10):
            await asyncio.sleep(0)
        self.assertFalse(second.done())
        self.auth.drop_session(target)  # what /logout does
        self.auth.prune()
        self.assertNotIn(target, self.sessions)
        self.assertIn(target, self.auth.refresh_locks)
        self.gate.release.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
        self.assertEqual([getattr(r, "status_code", r) for r in results], [401, 401])
        self.assertEqual((self.gate.count, self.gate.peak), (1, 1))
        self.auth.prune()
        self.assertNotIn(target, self.auth.refresh_locks)

    async def test_released_request_lock_with_queued_waiter_survives_prune(self):
        """Characterization (SEC-04): in the window after a request lock is released to a queued waiter that has not
        resumed yet (``locked()`` is already False), a prune pass keeps the entry, so a third caller for the same key
        queues behind the waiter instead of running alongside it. The key belongs to no live session and its retry
        time has passed, so only ``busy()`` protects it."""
        self.gate.matches = is_guild_list
        key = ("/users/@me/guilds", token_key("Bearer orphan"))
        lock = self.auth.request_locks.setdefault(key, asyncio.Lock())
        await lock.acquire()
        self.auth.retry_at[key] = time.monotonic() - 1
        self.gate.armed = True
        waiter = asyncio.create_task(self.auth.discord_get("/users/@me/guilds", "Bearer orphan"))
        for _ in range(10):
            await asyncio.sleep(0)
        self.assertFalse(waiter.done())
        self.assertEqual(self.gate.count, 0)
        lock.release()  # hands the lock to the waiter, which has not run yet
        self.assertFalse(lock.locked())
        self.auth.prune()  # no await between release and prune: still inside the window
        self.assertIs(self.auth.request_locks.get(key), lock)
        third = asyncio.create_task(self.auth.discord_get("/users/@me/guilds", "Bearer orphan"))
        await asyncio.wait_for(self.gate.entered.wait(), 1)
        for _ in range(10):
            await asyncio.sleep(0)
        self.assertEqual(self.gate.peak, 1)
        self.gate.release.set()
        await asyncio.gather(waiter, third)
        self.assertEqual((self.gate.count, self.gate.peak), (2, 1))


class LockInternalsTests(unittest.TestCase):
    def test_asyncio_lock_has_waiters_attribute(self):
        """SEC-04: ``llmcord_core.auth.busy()`` reads the private ``asyncio.Lock._waiters`` to see queued waiters
        after a release; if a Python upgrade removes it, ``busy()`` silently degrades to ``locked()``, so fail loudly."""
        self.assertTrue(hasattr(asyncio.Lock(), "_waiters"))


if __name__ == "__main__":
    unittest.main()
