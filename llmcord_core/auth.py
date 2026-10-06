"""Discord authorization shared by ordinary requests and live UI callbacks."""
from __future__ import annotations

import asyncio
import secrets
import time

from fastapi import HTTPException

DISCORD_API = 'https://discord.com/api/v10'


class AuthService:
    def __init__(self, app):
        self.app = app
        self.refresh_locks = {}

    async def discord_get(self, path, token):
        response = await self.app.state.http.get(DISCORD_API + path, headers={'Authorization': token})
        if response.status_code >= 400:
            raise HTTPException(502, 'Discord authentication is unavailable')
        return response.json()

    async def session(self, ident):
        session = self.app.state.sessions.get(ident)
        if not session or session['expires'] < time.time():
            raise HTTPException(401, 'Sign in with Discord')
        if session['token_expires'] < time.time() + 30:
            async with self.refresh_locks.setdefault(ident, asyncio.Lock()):
                if session['token_expires'] < time.time() + 30:
                    response = await self.app.state.http.post(DISCORD_API + '/oauth2/token', data={
                        'grant_type': 'refresh_token', 'refresh_token': session['refresh'],
                        'client_id': self.app.state.client_id, 'client_secret': self.app.state.client_secret})
                    if response.status_code >= 400:
                        self.app.state.sessions.pop(ident, None)
                        raise HTTPException(401, 'Discord session expired')
                    tokens = response.json()
                    session.update(access=tokens['access_token'], refresh=tokens['refresh_token'], token_expires=time.time() + int(tokens['expires_in']))
        if self.app.state.sessions.get(ident) is not session or session['expires'] < time.time():
            raise HTTPException(401, 'Discord session expired')
        return session

    async def session_for(self, request):
        return await self.session(request.cookies.get('llmcord_session', ''))

    async def guard(self, ident, guild_id, csrf=None, origin=None):
        session = await self.session(ident)
        if origin and origin.rstrip('/') != self.app.state.base_url:
            raise HTTPException(403, 'Invalid request origin')
        if csrf is not None and not secrets.compare_digest(str(csrf), session['csrf']):
            raise HTTPException(403, 'Invalid form token')
        guilds = await self.discord_get('/users/@me/guilds', 'Bearer ' + session['access'])
        if self.app.state.sessions.get(ident) is not session or session['expires'] < time.time():
            raise HTTPException(401, 'Discord session expired')
        if not any(int(g['id']) == guild_id and (g.get('owner') or int(g.get('permissions', '0')) & 8) for g in guilds):
            raise HTTPException(403, 'Server administrator permission required')
        return session

    async def require_admin(self, request, guild_id, mutate=False):
        csrf = str((await request.form()).get('csrf', '')) if mutate else None
        return await self.guard(request.cookies.get('llmcord_session', ''), guild_id, csrf, request.headers.get('origin'))
