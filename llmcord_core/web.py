"""Private, Discord-authenticated administration for one llmcord installation."""
from __future__ import annotations

import json
import secrets
import sqlite3
import time
import re
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode, urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from .avatars import avatar_version
from .cards import parse_card
from .lorebooks import MAX_BOOK_BYTES, parse_lorebook
from .store import Store
from .auth import AuthService
from .admin import AdminService
from .admin_store import ConflictError


DISCORD_API = "https://discord.com/api/v10"
ADMINISTRATOR = 1 << 3
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
TEMPLATES.env.filters["avatar_version"] = avatar_version
AVATAR_CACHE = {"Cache-Control": "private, max-age=300"}


def create_app(database_path: str | Path, base_url: str, client_id: str,
               client_secret: str, bot_token: str, oauth_http: httpx.AsyncClient | None = None, *, enable_dashboard: bool = True, config_path: str = "config.yaml") -> FastAPI:
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
    app.state.sessions = {}
    app.state.states = {}
    app.state.base_url = base_url
    app.state.client_id = client_id
    app.state.client_secret = client_secret
    app.state.bot_token = bot_token

    @app.exception_handler(ValueError)
    async def invalid_value(_request: Request, error: ValueError):
        return PlainTextResponse(f"Invalid input: {str(error)}", status_code=400)

    @app.exception_handler(sqlite3.IntegrityError)
    async def duplicate_value(_request: Request, _error: sqlite3.IntegrityError):
        return PlainTextResponse("This name or entry already exists", status_code=409)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        if request.scope.get("endpoint") not in cacheable or "cache-control" not in response.headers:
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        policy = "default-src 'self'; style-src 'self' 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'"
        if request.url.path.startswith('/admin'):
            nonce = secrets.token_urlsafe(24)
            policy += f"; script-src 'self' 'nonce-{nonce}' 'unsafe-eval'; img-src 'self' data: blob:; connect-src 'self' {base_url.replace('https://', 'wss://')}; font-src 'self' data:"
            if 'text/html' in response.headers.get('content-type', ''):
                body = b''.join([part async for part in response.body_iterator]).decode('utf-8')
                body = re.sub(r'<script(?=[\s>])', f'<script nonce="{nonce}"', body)
                headers = dict(response.headers)
                headers.pop('content-length', None)
                response = Response(body, status_code=response.status_code, headers=headers, background=response.background)
        response.headers["Content-Security-Policy"] = policy
        return response

    app.state.auth = AuthService(app)
    app.state.admin = AdminService(app, config_path)
    discord_get = app.state.auth.discord_get
    session_for = app.state.auth.session_for
    require_admin = app.state.auth.require_admin

    @app.exception_handler(ConflictError)
    async def conflicting_value(_request, error):
        return PlainTextResponse(str(error), status_code=409)

    def redirect(guild_id: int):
        return RedirectResponse(f"/guild/{guild_id}", status_code=303)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        if enable_dashboard:
            return RedirectResponse('/admin/', status_code=303)
        try:
            session = await session_for(request)
        except HTTPException:
            return TEMPLATES.TemplateResponse(request, "login.html", {"base_url": base_url})
        guilds = await app.state.auth.guilds(session)
        allowed = [guild for guild in guilds if guild.get("owner") or
            int(guild.get("permissions", "0")) & ADMINISTRATOR]
        return TEMPLATES.TemplateResponse(request, "index.html",
            {"guilds": allowed, "user": session["user"], "csrf": session["csrf"]})

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
        form = await request.form()
        if not secrets.compare_digest(str(form.get("csrf", "")), session["csrf"]):
            raise HTTPException(403, "Invalid form token")
        app.state.auth.drop_session(request.cookies.get("llmcord_session", ""))
        result = RedirectResponse("/", status_code=303)
        result.delete_cookie("llmcord_session")
        return result

    @app.get("/guild/{guild_id}", response_class=HTMLResponse)
    async def guild_page(request: Request, guild_id: int):
        session = await require_admin(request, guild_id)
        if enable_dashboard:
            return RedirectResponse(f'/admin/guild/{guild_id}', status_code=303)
        store = app.state.store
        channels_response = await app.state.http.get(
            f"{DISCORD_API}/guilds/{guild_id}/channels",
            headers={"Authorization": "Bot " + bot_token})
        channels = [item for item in channels_response.json() if item.get("type") == 0] if channels_response.status_code == 200 else []
        spaces = store.list_spaces(guild_id)
        bindings = store.all("SELECT * FROM channels WHERE guild_id=? ORDER BY channel_id", (guild_id,))
        characters = store.all("SELECT * FROM characters WHERE guild_id=? ORDER BY name", (guild_id,))
        lore = store.all("SELECT * FROM lore WHERE guild_id=? AND scope_kind IN ('channel','space') ORDER BY id DESC", (guild_id,))
        books = store.list_lorebooks(guild_id)
        cast_choices = {binding["channel_id"]: store.eligible_characters(
            guild_id, binding["space_id"]) for binding in bindings}
        cast_selected = {binding["channel_id"]: json.loads(binding["default_cast"])
            for binding in bindings}
        return TEMPLATES.TemplateResponse(request, "guild.html", {
            "guild_id": guild_id, "user": session["user"], "csrf": session["csrf"],
            "spaces": spaces, "channels": channels, "bindings": bindings,
            "characters": characters, "lore": lore, "books": books,
            "card_fields": {row["id"]: json.loads(row["card"]) for row in characters},
            "cast_choices": cast_choices, "cast_selected": cast_selected,
            "book_entries": {book["id"]: store.lorebook_entries(book["id"]) for book in books},
            "book_entry_rules": {entry["id"]: json.loads(entry["rule_json"])
                for book in books for entry in store.lorebook_entries(book["id"])},
            "links": {book["id"]: store.lorebook_links(book["id"]) for book in books},
            "hub_links": {space["id"]: store.allowed_worlds(space["id"]) for space in spaces if space["kind"] == "hub"},
            "store": store})

    async def posted(request: Request, guild_id: int):
        session = await require_admin(request, guild_id, True)
        return session, await request.form()

    @app.post("/guild/{guild_id}/spaces")
    async def create_space(request: Request, guild_id: int):
        session, form = await posted(request, guild_id)
        ident = app.state.store.create_space(guild_id, str(form.get("name", "")), str(form.get("kind", "")))
        app.state.store.audit(guild_id, int(session["user"]["id"]), "space.create", {"id": ident})
        return redirect(guild_id)

    @app.post("/guild/{guild_id}/links")
    async def link_world(request: Request, guild_id: int):
        session, form = await posted(request, guild_id)
        hub_id, world_id = int(form["hub_id"]), int(form["world_id"])
        enabled = form.get("enabled") == "yes"
        (app.state.store.link_world if enabled else app.state.store.unlink_world)(guild_id, hub_id, world_id)
        app.state.store.audit(guild_id, int(session["user"]["id"]), "hub.link", {"hub": hub_id, "world": world_id, "enabled": enabled})
        return redirect(guild_id)

    @app.post("/guild/{guild_id}/bindings")
    async def bind_channel(request: Request, guild_id: int):
        session, form = await posted(request, guild_id)
        channel_id, space_id = int(form["channel_id"]), int(form["space_id"])
        response = await app.state.http.get(f"{DISCORD_API}/guilds/{guild_id}/channels",
            headers={"Authorization": "Bot " + bot_token})
        if response.status_code != 200 or not any(int(item["id"]) == channel_id and item["type"] == 0 for item in response.json()):
            raise HTTPException(400, "Choose a text channel in this server")
        app.state.store.bind_channel(guild_id, channel_id, space_id)
        app.state.store.audit(guild_id, int(session["user"]["id"]), "channel.bind", {"channel": channel_id, "space": space_id})
        return redirect(guild_id)

    @app.post("/guild/{guild_id}/characters/import")
    async def preview_character(request: Request, guild_id: int):
        session, form = await posted(request, guild_id)
        upload: UploadFile = form["file"]
        if not upload.filename or not upload.filename.lower().endswith((".json", ".png")):
            raise HTTPException(400, "Upload a JSON or PNG character card")
        data = await upload.read(8 * 1024 * 1024 + 1)
        await upload.close()
        card = parse_card(upload.filename, data)
        world_id = int(form["world_id"])
        world = app.state.store.space_by_id(world_id)
        if not world or world["guild_id"] != guild_id or world["kind"] != "world":
            raise HTTPException(400, "Choose a home world in this server")
        existing = app.state.store.character(guild_id, card.name)
        session["card_preview"] = {"world_id": world_id, "card": card, "changes": app.state.store.preview_card(guild_id, world_id, card),
            "expires": time.time() + 900}
        return TEMPLATES.TemplateResponse(request, "card_preview.html", {
            "guild_id": guild_id, "card": card, "world": world,
            "existing": existing, "csrf": session["csrf"], 'changes': session['card_preview']['changes']})

    @app.post("/guild/{guild_id}/characters/apply")
    async def apply_character(request: Request, guild_id: int):
        session, form = await posted(request, guild_id)
        preview = session.get("card_preview")
        if not preview or preview["expires"] < time.time():
            raise HTTPException(409, "Card preview expired; upload again")
        card = preview["card"]
        existing = app.state.store.character(guild_id, card.name)
        if existing and form.get("replace") != "yes":
            raise HTTPException(409, "Confirm replacing the existing card")
        ident = app.state.store.apply_card(guild_id, preview["world_id"], card, {key[8:]: str(value) for key, value in form.items() if key.startswith("resolve:")}, preview["changes"]["revision"])
        session.pop("card_preview", None)
        app.state.store.audit(guild_id, int(session["user"]["id"]), "character.import", {"id": ident})
        return redirect(guild_id)

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

    @app.get("/guild/{guild_id}/card-preview/avatar")
    async def preview_avatar(request: Request, guild_id: int):
        session = await require_admin(request, guild_id)
        preview = session.get("card_preview")
        if not preview or preview["expires"] < time.time() or not preview["card"].avatar:
            raise HTTPException(404, "Preview avatar not found")
        return Response(preview["card"].avatar, media_type="image/png")

    cacheable = {character_avatar, emotion_avatar}

    @app.post("/guild/{guild_id}/characters/{character_id}")
    async def edit_character(request: Request, guild_id: int, character_id: int):
        session, form = await posted(request, guild_id)
        store = app.state.store
        row = store.one("SELECT * FROM characters WHERE guild_id=? AND id=?", (guild_id, character_id))
        if not row:
            raise HTTPException(404, "Character not found")
        card = json.loads(row["card"])
        for field in ("description", "personality", "scenario", "first_mes", "mes_example"):
            card[field] = str(form.get(field, card.get(field, "")))
        name = str(form.get("name", row["name"])).strip()
        card["name"] = name
        world_id = int(form.get("world_id", row["world_id"]))
        if world_id != row["world_id"] and form.get("confirm_move") != "yes":
            impact = store.cast_impact(guild_id, character_id, world_id)
            raise HTTPException(409, f"Moving worlds affects {len(impact)} channel casts; confirm the move")
        store.update_character(guild_id, character_id, world_id, name, card)
        store.audit(guild_id, int(session["user"]["id"]), "character.edit", {"id": character_id, "world": world_id})
        return redirect(guild_id)

    @app.post("/guild/{guild_id}/characters/{character_id}/archive")
    async def archive_character(request: Request, guild_id: int, character_id: int):
        session, form = await posted(request, guild_id)
        app.state.store.archive_character(guild_id, character_id, form.get("archived") == "yes")
        app.state.store.audit(guild_id, int(session["user"]["id"]), "character.archive", {"id": character_id})
        return redirect(guild_id)

    @app.post("/guild/{guild_id}/casts")
    async def set_cast(request: Request, guild_id: int):
        session, form = await posted(request, guild_id)
        channel_id = int(form["channel_id"])
        binding = app.state.store.channel(channel_id)
        if not binding or binding["guild_id"] != guild_id:
            raise HTTPException(400, "Channel not bound in this server")
        ids = [int(value) for value in form.getlist("character_id")]
        app.state.store.set_cast(channel_id, None, ids, default=True)
        app.state.store.audit(guild_id, int(session["user"]["id"]), "cast.default", {"channel": channel_id, "characters": ids})
        return redirect(guild_id)

    @app.post("/guild/{guild_id}/ambient")
    async def set_ambient(request: Request, guild_id: int):
        session, form = await posted(request, guild_id)
        channel_id = int(form["channel_id"])
        binding = app.state.store.channel(channel_id)
        if not binding or binding["guild_id"] != guild_id:
            raise HTTPException(400, "Channel not bound in this server")
        enabled = form.get("enabled") == "yes"
        app.state.store.set_ambient(channel_id, enabled)
        app.state.store.audit(guild_id, int(session["user"]["id"]), "channel.ambient",
            {"channel": channel_id, "enabled": enabled})
        return redirect(guild_id)

    @app.post("/guild/{guild_id}/lore")
    async def add_lore(request: Request, guild_id: int):
        session, form = await posted(request, guild_id)
        kind, raw_ident = str(form["scope"]).split(":", 1)
        ident = int(raw_ident)
        if kind == "space":
            scope = app.state.store.space_by_id(ident)
            valid = scope and scope["guild_id"] == guild_id
        elif kind == "channel":
            scope = app.state.store.channel(ident)
            valid = scope and scope["guild_id"] == guild_id
        else:
            valid = False
        if not valid:
            raise HTTPException(400, "Invalid lore scope")
        keys = [value.strip() for value in str(form.get("keys", "")).split(",") if value.strip()]
        lore_id = app.state.store.add_lore(guild_id, kind, ident, str(form.get("content", "")),
            keys, constant=not keys)
        app.state.store.audit(guild_id, int(session["user"]["id"]), "lore.add", {"id": lore_id})
        return redirect(guild_id)

    @app.post("/guild/{guild_id}/lore/{lore_id}")
    async def edit_lore(request: Request, guild_id: int, lore_id: int):
        session, form = await posted(request, guild_id)
        app.state.store.edit_lore(guild_id, lore_id, str(form.get("content", "")))
        app.state.store.audit(guild_id, int(session["user"]["id"]), "lore.edit", {"id": lore_id})
        return redirect(guild_id)

    @app.post("/guild/{guild_id}/lore/{lore_id}/delete")
    async def delete_lore(request: Request, guild_id: int, lore_id: int):
        session, _ = await posted(request, guild_id)
        app.state.store.delete_lore(guild_id, lore_id)
        app.state.store.audit(guild_id, int(session["user"]["id"]), "lore.delete", {"id": lore_id})
        return redirect(guild_id)

    @app.post("/guild/{guild_id}/lore/{lore_id}/pin")
    async def pin_lore(request: Request, guild_id: int, lore_id: int):
        session, _ = await posted(request, guild_id)
        if not app.state.store.lore_row(guild_id, lore_id):
            raise HTTPException(404, "Lore entry not found")
        app.state.store.pin_lore(guild_id, lore_id)
        app.state.store.audit(guild_id, int(session["user"]["id"]), "lore.pin", {"id": lore_id})
        return redirect(guild_id)

    @app.post("/guild/{guild_id}/lore/{lore_id}/promote")
    async def promote_lore(request: Request, guild_id: int, lore_id: int):
        session, form = await posted(request, guild_id)
        kind, raw_ident = str(form["scope"]).split(":", 1)
        ident = int(raw_ident)
        scope = (app.state.store.space_by_id(ident) if kind == "space" else
                 app.state.store.channel(ident) if kind == "channel" else None)
        if not scope or scope["guild_id"] != guild_id:
            raise HTTPException(400, "Choose a world, hub, or bound channel in this server")
        new_id = app.state.store.promote_lore(guild_id, lore_id, kind, ident)
        app.state.store.audit(guild_id, int(session["user"]["id"]), "lore.promote",
            {"source": lore_id, "new_id": new_id, "scope": kind, "scope_id": ident})
        return redirect(guild_id)

    @app.post("/guild/{guild_id}/books")
    async def create_book(request: Request, guild_id: int):
        session, form = await posted(request, guild_id)
        ident = app.state.store.create_lorebook(guild_id, str(form.get("name", "")),
            str(form.get("target_kind", "")), int(form.get("target_id", 0) or 0))
        app.state.store.audit(guild_id, int(session["user"]["id"]), "book.create", {"id": ident})
        return redirect(guild_id)

    @app.post("/guild/{guild_id}/books/{book_id}/assign")
    async def assign_book(request: Request, guild_id: int, book_id: int):
        session, form = await posted(request, guild_id)
        space_id = int(form["space_id"])
        enabled = form.get("enabled") == "yes"
        app.state.store.set_lorebook_space(guild_id, book_id, space_id, enabled)
        app.state.store.audit(guild_id, int(session["user"]["id"]), "book.assign", {"id": book_id, "space": space_id, "enabled": enabled})
        return redirect(guild_id)

    @app.post("/guild/{guild_id}/books/{book_id}/preview", response_class=HTMLResponse)
    async def preview_book(request: Request, guild_id: int, book_id: int):
        session, form = await posted(request, guild_id)
        upload: UploadFile = form["file"]
        if not upload.filename or not upload.filename.lower().endswith(".json"):
            raise HTTPException(400, "Upload a JSON lorebook")
        data = await upload.read(MAX_BOOK_BYTES + 1)
        await upload.close()
        imported = parse_lorebook(data)
        book = app.state.store.lorebook(guild_id, book_id)
        if not book:
            raise HTTPException(404, "Lorebook not found")
        changes = app.state.store.preview_lorebook_sync(guild_id, book_id, imported)
        session["preview"] = {"book_id": book_id, "revision": book["revision"],
            "imported": imported, "expires": time.time() + 900}
        return TEMPLATES.TemplateResponse(request, "preview.html", {
            "guild_id": guild_id, "book": book, "changes": changes, "csrf": session["csrf"]})

    @app.post("/guild/{guild_id}/books/{book_id}/apply")
    async def apply_book(request: Request, guild_id: int, book_id: int):
        session, form = await posted(request, guild_id)
        preview = session.get("preview")
        if not preview or preview["book_id"] != book_id or preview["expires"] < time.time():
            raise HTTPException(409, "Import preview expired; upload again")
        resolutions = {key[8:]: str(value) for key, value in form.items() if key.startswith("resolve:")}
        changes = app.state.store.sync_lorebook(guild_id, book_id, preview["imported"],
            resolutions, preview["revision"])
        session.pop("preview", None)
        app.state.store.audit(guild_id, int(session["user"]["id"]), "book.sync", {"id": book_id,
            "changes": [{"uid": change["uid"], "status": change["status"]} for change in changes]})
        return redirect(guild_id)

    @app.post("/guild/{guild_id}/books/{book_id}/entries/{entry_id}")
    async def edit_book_entry(request: Request, guild_id: int, book_id: int, entry_id: int):
        session, form = await posted(request, guild_id)
        row = app.state.store.one("SELECT * FROM lorebook_entries WHERE id=? AND book_id=?", (entry_id, book_id))
        if not row or not app.state.store.lorebook(guild_id, book_id):
            raise HTTPException(404, "Entry not found")
        app.state.store.edit_lorebook_entry(guild_id, entry_id, str(form.get("content", "")))
        app.state.store.audit(guild_id, int(session["user"]["id"]), "book.entry.edit", {"id": entry_id})
        return redirect(guild_id)

    if enable_dashboard:
        from .dashboard import mount_dashboard
        mount_dashboard(app)
    return app
