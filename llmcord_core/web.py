"""Private, Discord-authenticated administration for one llmcord installation."""
from __future__ import annotations

import secrets
import sqlite3
import time
import re
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode, urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, RedirectResponse, Response
from starlette.datastructures import MutableHeaders
from starlette.middleware.gzip import DEFAULT_EXCLUDED_CONTENT_TYPES, GZipMiddleware

from .store import Store
from .auth import DISCORD_API, AuthService
from .admin import AdminService


AVATAR_CACHE = {"Cache-Control": "private, max-age=300"}


class SecurityHeaders:
    """Plain ASGI middleware: adds security headers; only /admin text/html is buffered to add CSP nonces."""

    def __init__(self, app, base_url: str, cacheable: set, versioned: dict):
        self.app, self.cacheable, self.versioned = app, cacheable, versioned
        self.connect = base_url.replace("https://", "wss://")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        admin = scope["path"].startswith("/admin")
        nonce = secrets.token_urlsafe(24) if admin else ""
        policy = "default-src 'self'; style-src 'self' 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'"
        if admin:
            policy += f"; script-src 'self' 'nonce-{nonce}' 'unsafe-eval'; img-src 'self' data: blob: https://cdn.discordapp.com; connect-src 'self' {self.connect}; font-src 'self' data:"
        start, chunks, rewrite = None, [], False

        async def wrapped(message):
            nonlocal start, rewrite
            if message["type"] == "http.response.start":
                start = {**message, "headers": list(message.get("headers", []))}
                headers = MutableHeaders(raw=start["headers"])
                prefix = self.versioned.get("prefix")
                if (prefix and scope["path"].startswith(prefix) and not scope["path"].startswith(prefix + "dynamic_resources/")
                        and start["status"] in (200, 304)):
                    headers["Cache-Control"] = self.versioned["directives"]  # versioned URLs are immutable
                elif scope.get("endpoint") not in self.cacheable or "cache-control" not in headers:
                    headers["Cache-Control"] = "no-store"
                headers["X-Content-Type-Options"] = "nosniff"
                headers["Referrer-Policy"] = "same-origin"
                headers["Content-Security-Policy"] = policy
                rewrite = admin and "text/html" in headers.get("content-type", "")
                if not rewrite:
                    await send(start)
            elif not rewrite:
                await send(message)
            else:
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    body = b"".join(chunks).decode("utf-8")
                    body = re.sub(r"<script(?=[\s>])", f'<script nonce="{nonce}"', body).encode("utf-8")
                    if scope["method"] != "HEAD":
                        MutableHeaders(raw=start["headers"])["content-length"] = str(len(body))
                    await send(start)
                    await send({"type": "http.response.body", "body": body})

        await self.app(scope, receive, wrapped)


def create_app(database_path: str | Path, base_url: str, client_id: str,
               client_secret: str, bot_token: str, oauth_http: httpx.AsyncClient | None = None, *, enable_dashboard: bool = True, config_path: str = "config.yaml", operator_ids: frozenset[int] = frozenset()) -> FastAPI:
    base_url = base_url.rstrip("/")
    parsed_url = urlparse(base_url)
    if parsed_url.scheme != "https" or not parsed_url.hostname:
        raise ValueError("WEB_BASE_URL must be a private HTTPS URL")
    if not client_id or not client_secret or not bot_token:
        raise ValueError("Discord OAuth and bot credentials are required")
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            yield
        finally:
            application.state.store.close()
            if application.state.owns_http:
                await application.state.http.aclose()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.store = Store(database_path)
    app.state.http = oauth_http or httpx.AsyncClient(timeout=10)
    app.state.owns_http = oauth_http is None
    app.state.operator_ids = frozenset(operator_ids)
    app.state.sessions = {}
    app.state.states = {}
    app.state.base_url = base_url
    app.state.client_id = client_id
    app.state.client_secret = client_secret
    app.state.bot_token = bot_token

    # ConflictError is a ValueError; no parent-app route writes, so none reaches here (NiceGUI actions handle it).
    @app.exception_handler(ValueError)
    async def invalid_value(_request: Request, error: ValueError):
        return PlainTextResponse(f"Invalid input: {str(error)}", status_code=400)

    @app.exception_handler(sqlite3.IntegrityError)
    async def duplicate_value(_request: Request, _error: sqlite3.IntegrityError):
        return PlainTextResponse("This name or entry already exists", status_code=409)

    cacheable, versioned = set(), {}
    app.add_middleware(SecurityHeaders, base_url=base_url, cacheable=cacheable, versioned=versioned)

    app.state.auth = AuthService(app)
    app.state.admin = AdminService(app, config_path)
    discord_get = app.state.auth.discord_get
    session_for = app.state.auth.session_for
    require_admin = app.state.auth.require_admin

    @app.get("/")
    async def index():
        return RedirectResponse("/admin/", status_code=303)

    @app.get("/login")
    async def login():
        state = secrets.token_urlsafe(32)
        app.state.states[state] = time.time() + 600
        app.state.states = {key: expiry for key, expiry in app.state.states.items()
            if expiry >= time.time()}
        query = urlencode({"response_type": "code", "client_id": client_id,
            "scope": "identify guilds", "state": state,
            "redirect_uri": base_url + "/auth/callback"})
        result = RedirectResponse("https://discord.com/oauth2/authorize?" + query)
        result.set_cookie("llmcord_oauth_state", state, secure=True, httponly=True,
            samesite="lax", max_age=600)
        return result

    @app.get("/auth/callback")
    async def callback(request: Request, code: str, state: str):
        if not secrets.compare_digest(request.cookies.get("llmcord_oauth_state", ""), state):
            raise HTTPException(403, "Invalid OAuth state")
        expiry = app.state.states.pop(state, 0)
        if expiry < time.time():
            raise HTTPException(403, "Invalid OAuth state")
        response = await app.state.http.post(DISCORD_API + "/oauth2/token",
            data={"grant_type": "authorization_code", "code": code,
                  "redirect_uri": base_url + "/auth/callback",
                  "client_id": client_id, "client_secret": client_secret})
        if response.status_code >= 400:
            raise HTTPException(401, "Discord sign-in failed")
        tokens = response.json()
        user = await discord_get("/users/@me", "Bearer " + tokens["access_token"])
        ident = secrets.token_urlsafe(40)
        app.state.auth.prune()
        app.state.sessions[ident] = {"user": user, "access": tokens["access_token"],
            "refresh": tokens["refresh_token"],
            "token_expires": time.time() + int(tokens["expires_in"]),
            "expires": time.time() + 7 * 86400, "csrf": secrets.token_urlsafe(32)}
        result = RedirectResponse("/", status_code=303)
        result.set_cookie("llmcord_session", ident, secure=True, httponly=True,
            samesite="lax", max_age=7 * 86400)
        result.delete_cookie("llmcord_oauth_state", secure=True, httponly=True,
            samesite="lax")
        return result

    @app.post("/logout")
    async def logout(request: Request):
        session = await session_for(request)
        app.state.auth.check_origin(request.headers.get("origin"))
        form = await request.form()
        if not secrets.compare_digest(str(form.get("csrf", "")), session["csrf"]):
            raise HTTPException(403, "Invalid form token")
        app.state.auth.drop_session(request.cookies.get("llmcord_session", ""))
        result = RedirectResponse("/", status_code=303)
        result.delete_cookie("llmcord_session")
        return result

    @app.get("/guild/{guild_id}")
    async def guild_page(guild_id: int):
        return RedirectResponse(f"/admin/guild/{guild_id}", status_code=303)

    @app.get("/guild/{guild_id}/characters/{character_id}/avatar")
    async def character_avatar(request: Request, guild_id: int, character_id: int):
        await require_admin(request, guild_id)
        row = app.state.store.one("SELECT avatar FROM characters WHERE guild_id=? AND id=?",
            (guild_id, character_id))
        if not row or not row["avatar"]:
            raise HTTPException(404, "Avatar not found")
        return Response(row["avatar"], media_type="image/png", headers=AVATAR_CACHE)

    @app.get('/guild/{guild_id}/characters/{character_id}/avatars/{slot_key}')
    async def emotion_avatar(request: Request, guild_id: int, character_id: int, slot_key: str):
        await require_admin(request, guild_id)
        row = app.state.store.avatar_slot(guild_id, character_id, slot_key)
        if not row['image']:
            raise HTTPException(404, 'Avatar not found')
        return Response(row['image'], media_type='image/png', headers=AVATAR_CACHE)

    cacheable.update((character_avatar, emotion_avatar))

    if enable_dashboard:
        from .dashboard import mount_dashboard
        mount_dashboard(app)
        from nicegui import core
        from nicegui.version import __version__
        versioned.update(prefix=f"/admin/_nicegui/{__version__}/", directives=core.app.config.cache_control_directives)
    # Added last so it is outermost: the CSP rewrite sees plain HTML. text/html is excluded (BREACH: CSRF token, reflected params, socket.io polling text).
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6,
                       exclude_content_types=(*DEFAULT_EXCLUDED_CONTENT_TYPES, "text/html", "text/plain", "application/octet-stream"))
    return app
