"""Discord authorization shared by ordinary requests and live UI callbacks."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import time

import httpx
from fastapi import HTTPException

DISCORD_API = 'https://discord.com/api/v10'
GUILD_CACHE_TTL = 300  # seconds a session's guild list is reused (roadmap D1)


def token_key(token):
    return hashlib.sha256(token.encode()).digest()


def busy(lock):
    # Held or awaited locks must stay shared, or two refreshes/requests for one key could run at once (SEC-04).
    return lock.locked() or bool(getattr(lock, '_waiters', None))


class AuthService:
    def __init__(self, app):
        self.app = app
        self.refresh_locks = {}
        self.request_locks = {}
        self.retry_at = {}

    @staticmethod
    def retry_delay(response):
        value = response.headers.get('Retry-After') or response.headers.get('X-RateLimit-Reset-After')
        if value is None:
            try:
                value = response.json().get('retry_after', 1)
            except (ValueError, AttributeError):
                value = 1
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 1.0

    async def discord_get(self, path, token):
        key = (path, token_key(token))
        deadline = time.monotonic() + 20
        async with self.request_locks.setdefault(key, asyncio.Lock()):
            for attempt in range(3):
                delay = max(0, self.retry_at.get(key, 0) - time.monotonic())
                if delay:
                    if time.monotonic() + delay >= deadline:
                        raise HTTPException(503, 'Discord permission checks are temporarily rate limited. Please try again shortly.',
                                            headers={'Retry-After': str(max(1, int(delay) + 1))})
                    await asyncio.sleep(delay)
                try:
                    response = await self.app.state.http.get(DISCORD_API + path, headers={'Authorization': token})
                except httpx.RequestError:
                    logging.warning('Discord permission request failed: path=%s network_error', path)
                    raise HTTPException(503, 'Cannot reach Discord to check permissions. Please try again shortly.') from None
                if response.headers.get('X-RateLimit-Remaining') == '0' or response.status_code == 429:
                    self.retry_at[key] = time.monotonic() + self.retry_delay(response)
                if response.status_code == 429:
                    logging.warning('Discord permission request rate limited: path=%s retry_after=%.2f', path, self.retry_delay(response))
                    continue
                if response.status_code == 401:
                    # Invalidate only sessions using the rejected access token.
                    for ident, session in list(self.app.state.sessions.items()):
                        if 'Bearer ' + session['access'] == token:
                            self.drop_session(ident)
                    raise HTTPException(401, 'Discord session expired. Please sign in again.')
                if response.status_code >= 400:
                    logging.warning('Discord permission request failed: path=%s status=%d', path, response.status_code)
                    detail = 'Discord denied the permission check. Please sign in again.' if response.status_code == 403 else 'Discord permission checks are unavailable. Please try again shortly.'
                    raise HTTPException(403 if response.status_code == 403 else 502, detail)
                return response.json()
        raise HTTPException(503, 'Discord permission checks are temporarily rate limited. Please try again shortly.')

    def drop_session(self, ident):
        self.app.state.sessions.pop(ident, None)
        lock = self.refresh_locks.get(ident)
        if lock is not None and not busy(lock):
            del self.refresh_locks[ident]

    def prune(self):
        """Drop expired sessions and lock/backoff state no live session uses (SEC-04); O(sessions + keys)."""
        sessions, now = self.app.state.sessions, time.time()
        for ident, session in list(sessions.items()):
            if session.get('expires', now) < now:
                self.drop_session(ident)
        for ident, lock in list(self.refresh_locks.items()):
            if ident not in sessions and not busy(lock):
                del self.refresh_locks[ident]
        live = {token_key('Bearer ' + s['access']) for s in sessions.values() if 'access' in s}
        mono = time.monotonic()
        for key in list({*self.request_locks, *self.retry_at}):
            lock = self.request_locks.get(key)
            if key[1] in live or self.retry_at.get(key, 0) > mono or (lock is not None and busy(lock)):
                continue
            self.request_locks.pop(key, None)
            self.retry_at.pop(key, None)

    async def session(self, ident):
        self.prune()
        session = self.app.state.sessions.get(ident)
        if not session or session['expires'] < time.time():
            raise HTTPException(401, 'Sign in with Discord')
        if session['token_expires'] < time.time() + 30:
            failed = False
            async with self.refresh_locks.setdefault(ident, asyncio.Lock()):
                if session['token_expires'] < time.time() + 30:
                    response = await self.app.state.http.post(DISCORD_API + '/oauth2/token', data={
                        'grant_type': 'refresh_token', 'refresh_token': session['refresh'],
                        'client_id': self.app.state.client_id, 'client_secret': self.app.state.client_secret})
                    if response.status_code >= 400:
                        self.app.state.sessions.pop(ident, None)
                        failed = True
                    else:
                        tokens = response.json()
                        session.update(access=tokens['access_token'], refresh=tokens['refresh_token'], token_expires=time.time() + int(tokens['expires_in']))
            if failed:
                self.drop_session(ident)  # after releasing the lock, so an idle lock is dropped with the session
                raise HTTPException(401, 'Discord session expired')
        if self.app.state.sessions.get(ident) is not session or session['expires'] < time.time():
            raise HTTPException(401, 'Discord session expired')
        return session

    async def session_for(self, request):
        return await self.session(request.cookies.get('llmcord_session', ''))

    def current(self, session):
        return session['expires'] >= time.time() and any(s is session for s in self.app.state.sessions.values())

    def forget_guilds(self, session=None):
        for s in [session] if session is not None else list(self.app.state.sessions.values()):
            s.pop('guild_cache', None)

    async def guilds(self, session):
        # Cached on the session dict, so sign-out, 401, failed refresh and expiry drop it with the session.
        async with session.setdefault('guild_lock', asyncio.Lock()):
            if not self.current(session):  # e.g. queued behind a fetch that got a Discord 401
                raise HTTPException(401, 'Discord session expired')
            cached = session.get('guild_cache')
            if cached and time.monotonic() - cached[0] < GUILD_CACHE_TTL:
                return cached[1]
            fetched_at = time.monotonic()
            guilds = await self.discord_get('/users/@me/guilds', 'Bearer ' + session['access'])
            if self.current(session):
                session['guild_cache'] = (fetched_at, guilds)
            return guilds

    def check_origin(self, origin):
        if origin and origin.rstrip('/') != self.app.state.base_url:
            raise HTTPException(403, 'Invalid request origin')

    async def guard(self, ident, guild_id, csrf=None, origin=None):
        session = await self.session(ident)
        self.check_origin(origin)
        if csrf is not None and not secrets.compare_digest(str(csrf), session['csrf']):
            raise HTTPException(403, 'Invalid form token')
        guilds = await self.guilds(session)
        if self.app.state.sessions.get(ident) is not session or session['expires'] < time.time():
            raise HTTPException(401, 'Discord session expired')
        if not any(int(g['id']) == guild_id and (g.get('owner') or int(g.get('permissions', '0')) & 8) for g in guilds):
            raise HTTPException(403, 'Server administrator permission required')
        return session

    async def require_admin(self, request, guild_id, mutate=False):
        ident = request.cookies.get('llmcord_session', '')
        origin, csrf = request.headers.get('origin'), None
        if mutate:
            # Reject unauthenticated or cross-origin posts before parsing the body (SEC-01).
            await self.session(ident)
            self.check_origin(origin)
            csrf = str((await request.form()).get('csrf', ''))
        return await self.guard(ident, guild_id, csrf, origin)
