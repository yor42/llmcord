"""Component-based private administration mounted under /admin."""
from __future__ import annotations

import contextlib
import copy
import functools
import json
import sqlite3
import inspect
import types
import zoneinfo
import httpx
from urllib.parse import parse_qs
from http.cookies import SimpleCookie

from fastapi import HTTPException, Request
from fastapi.responses import PlainTextResponse

from .auth import guild_icon_url, is_server_admin
from .avatars import MAX_AVATAR_BYTES, avatar_version, normalize_avatar
from .cards import parse_card
from .lorebooks import parse_lorebook
from .prompts import PURPOSES, SOURCES, block, compatibility, default_bundle, export_preset, parse_preset
from .scene_ui import confirm_dialog, delete_book_dialog, delete_space_dialog, direct_import_dialog, guideline_editor


class LiveContext:
    def __init__(self, app, request, guild_id, session):
        self.app, self.guild_id = app, guild_id
        self.ident = request.cookies.get('llmcord_session', '')
        self.csrf = session['csrf']
        self.service = app.state.admin
        self.store = self.service.store
        self.lore_owner = request.query_params.get('owner')
        self.channel_names = {}
        self.selector, self.containers, self.builders, self.built = None, {}, {}, set()

    async def load_channel_names(self):
        # One Discord channel fetch per page render; a failure leaves the mapping empty.
        from nicegui import ui
        try:
            channels = await self.run(lambda: self.service.avatars.channels(self.guild_id)) or []
        except httpx.HTTPError:
            ui.notify('Discord channels are unavailable', type='negative', timeout=8000)
            channels = []
        self.channel_names = {int(c['id']): '#' + c['name'] for c in channels if c['type'] == 0}
        return self.channel_names

    @functools.cached_property
    def snapshot(self):
        # Render-time lookups for one page build; callbacks must use live store reads.
        store, gid = self.store, self.guild_id
        spaces, characters, channels = store.list_spaces(gid), store.list_characters(gid), store.list_channels(gid)
        lorebooks, scopes = store.list_lorebooks(gid), store.thread_lore_scopes(gid)
        return types.SimpleNamespace(spaces=spaces, characters=characters, channels=channels, lorebooks=lorebooks,
            worlds={r['id']: r['name'] for r in spaces if r['kind'] == 'world'},
            hubs={r['id']: r['name'] for r in spaces if r['kind'] == 'hub'},
            owners=self.service.owners_from(gid, spaces, characters, channels, lorebooks, scopes, self.channel_names))

    def set_url(self, **params):
        sets = ''.join(f'u.searchParams.set({json.dumps(k)}, {json.dumps(v)});' for k, v in params.items())
        self.selector.client.run_javascript(f'const u = new URL(location.href);{sets} history.replaceState(history.state, "", u);')

    async def build(self, tab):
        """Build a tab's panel once; a fresh snapshot is read per build because panels may be built long after page load."""
        container = self.containers.get(tab)
        if container is None or tab in self.built:
            return
        self.built.add(tab)
        self.__dict__.pop('snapshot', None)
        container.clear()
        try:
            with container:
                result = self.builders[tab]()
                if inspect.isawaitable(result):
                    await result  # builders must not otherwise await: a refresh during an awaited build would double-fill
        except BaseException:
            self.built.discard(tab)
            raise

    async def refresh(self, tab=None, owner=None):
        """In-page replacement for a browser reload: stale every built panel, optionally switch tab, rebuild the visible one."""
        if owner:
            self.lore_owner = owner
            self.set_url(owner=owner)
        for built in self.built:
            self.containers[built].clear()
        self.built.clear()
        if tab and tab != self.selector.value:
            self.selector.set_value(tab)  # returns before the build: the tab_panels change handler builds it
        else:
            await self.build(self.selector.value)

    async def run(self, operation, action=None, detail=None):
        return (await self._attempt(operation, action, detail))[1]

    async def _attempt(self, operation, action=None, detail=None):
        """Run an operation, returning (succeeded, result); a failure is notified and yields (False, None)."""
        from nicegui import ui
        try:
            return True, await self.service.run(self.ident, self.guild_id, operation, action, detail)
        except (ValueError, HTTPException, sqlite3.IntegrityError, TypeError, KeyError) as error:
            message = error.detail if isinstance(error, HTTPException) else 'This name already exists' if isinstance(error, sqlite3.IntegrityError) else 'Choose valid values for all required fields' if isinstance(error, (TypeError, KeyError)) else str(error)
            ui.notify(message, type='negative', timeout=8000)
            return False, None

    def button(self, text, operation, action=None, detail=None, then=None, success=None, **kwargs):
        from nicegui import ui
        async def clicked():
            ok, result = await self._attempt(operation, action, detail)
            if not ok:
                return
            if success:
                ui.notify(success, type='positive')
            if then:
                followup = then(result)
                if inspect.isawaitable(followup):
                    await followup
        return ui.button(text, on_click=clicked, **kwargs)

    def upload(self, handler, label, action=None, detail=None, then=None, success=None):
        from nicegui import ui
        async def uploaded(event):
            # Upload endpoint also validates session binding and CSRF before reading the body.
            ok, result = await self._attempt(lambda: handler(event), action, detail)
            if not ok:
                return
            if success:
                ui.notify(success, type='positive')
            if then:
                followup = then(result)
                if inspect.isawaitable(followup):
                    await followup
        control = ui.upload(label=label, on_upload=uploaded, auto_upload=True, max_file_size=MAX_AVATAR_BYTES, max_files=1)
        control._props['headers'] = [{'name': 'X-CSRF-Token', 'value': self.csrf}]
        return control


def pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def rejection_notice(error):
    """Negative-notification text for a live event rejected after the client was confirmed bound to its session."""
    if not isinstance(error, HTTPException):
        return None
    if error.status_code == 401:
        return 'Your Discord sign-in expired. Sign in again.'
    return str(error.detail or 'This action was rejected')


# Discord-like style tokens (UI-28, D20). Later steps reuse these names.
THEME_BODY, THEME_HEADER, THEME_CARD = '#1e1f22', '#111214', '#2b2d31'
THEME_CARD_HOVER, THEME_TILE, THEME_TILE_TEXT = '#35373c', '#404249', '#dbdee1'
THEME_TEXT_MUTED, THEME_TEXT_FAINT, THEME_BORDER = '#b5bac1', '#949ba4', '#4e5058'
THEME_PRIMARY, THEME_NEGATIVE = '#5865f2', '#da373c'
THEME_DIVIDER, THEME_NESTED = '#3f4147', '#313338'

_THEME_CSS = f"""
body.body--dark, body.body--dark .q-page, body.body--dark .q-layout {{ background: {THEME_BODY}; }}
.q-header {{ background: {THEME_HEADER}; }}
body.body--dark .q-card {{ background: {THEME_CARD}; }}
body.body--dark .q-card .q-card, body.body--dark .q-card .q-expansion-item {{ background: {THEME_NESTED}; }}
.ll-section {{ width: 100%; padding: 24px; gap: 16px; margin-bottom: 20px; align-items: flex-start; }}
.ll-section-title {{ font-size: 20px; font-weight: 700; line-height: 1.3; }}
.ll-form-row {{ width: 100%; display: flex; flex-flow: row wrap; align-items: flex-end; gap: 16px; }}
.ll-form-row .q-field {{ flex: 0 1 16rem; min-width: min(16rem, 100%); }}
.ll-section > .q-expansion-item, .ll-subpanel {{ border: 1px solid {THEME_DIVIDER}; border-radius: 8px; }}
.ll-stack {{ width: 100%; box-sizing: border-box; display: flex; flex-direction: column; align-items: stretch; gap: 16px; padding: 4px 0 8px; }}
.ll-stack .q-uploader {{ max-width: min(20rem, 100%); }}
.ll-subtitle {{ font-size: 16px; font-weight: 700; line-height: 1.3; }}
@media (max-width: 600px) {{ .ll-section {{ padding: 16px; }} .ll-form-row .q-field {{ flex-basis: 100%; }} }}
body.body--dark .q-tab-panels, body.body--dark .q-tab-panel {{ background: transparent; }}
.ll-page .q-tab-panel {{ padding-left: 0; padding-right: 0; }}
.ll-tabs {{ border-bottom: 1px solid {THEME_DIVIDER}; }}
.ll-tabs .q-tab {{ color: {THEME_TEXT_MUTED}; }}
.ll-tabs .q-tab--active {{ color: #fff; }}
.ll-tabs .q-tab__indicator {{ background: {THEME_PRIMARY}; height: 3px; }}
.ll-page {{ width: 100%; padding: 32px 48px; gap: 4px; align-items: stretch; }}
@media (max-width: 600px) {{ .ll-page {{ padding: 16px; }} }}
.ll-crumb {{ display: flex; align-items: center; gap: 8px; min-width: 0; flex: 1 1 auto; }}
.ll-crumb a {{ color: #fff !important; text-decoration: none !important; font-size: 18px; font-weight: 600; }}
.ll-crumb-sep {{ color: {THEME_TEXT_FAINT}; }}
.ll-crumb-icon {{ flex: none; width: 24px; height: 24px; border-radius: 8px; object-fit: cover; }}
.ll-crumb-tile {{ flex: none; width: 24px; height: 24px; border-radius: 8px; background: {THEME_TILE}; color: {THEME_TILE_TEXT}; font-size: 11px; font-weight: 600; display: flex; align-items: center; justify-content: center; }}
.ll-crumb-name {{ min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 600; }}
.q-card {{ border-radius: 12px; }}
.q-btn, .q-tab {{ text-transform: none; }}
.q-field--outlined .q-field__control {{ background: {THEME_BODY}; border-radius: 8px; }}
.q-field--outlined .q-field__control:before {{ border-color: {THEME_BORDER}; border-radius: 8px; }}
.q-field--outlined .q-field__control:after {{ border-radius: 8px; }}
.ll-tabs .q-tabs__arrow--left, .ll-tabs .q-tabs__arrow--right {{ color: {THEME_TEXT_MUTED}; }}
.ll-muted {{ color: {THEME_TEXT_MUTED}; }}
.ll-faint {{ color: {THEME_TEXT_FAINT}; }}
.ll-signin-card {{ width: 100%; max-width: 420px; padding: 40px; margin: 80px auto 0; align-items: stretch; text-align: center; gap: 12px; }}
.ll-logo {{ width: 56px; height: 56px; border-radius: 16px; background: {THEME_PRIMARY}; color: #fff; font-size: 30px; font-weight: 700; display: flex; align-items: center; justify-content: center; margin: 0 auto; }}
.ll-signin-btn {{ display: flex; align-items: center; justify-content: center; width: 100%; height: 48px; margin-top: 12px; border-radius: 8px; background: {THEME_PRIMARY}; color: #fff !important; font-weight: 600; text-decoration: none !important; }}
.ll-signin-btn:hover {{ background: #4752c4; }}
.ll-servers {{ width: 100%; max-width: 1100px; margin: 0 auto; padding: 16px; }}
.ll-server-card {{ display: flex; align-items: center; gap: 16px; padding: 20px; border-radius: 12px; background: {THEME_CARD}; color: #fff !important; text-decoration: none !important; font-size: 18px; font-weight: 600; min-width: 0; }}
.ll-server-card:hover {{ background: {THEME_CARD_HOVER}; }}
.ll-server-icon {{ flex: none; width: 56px; height: 56px; border-radius: 16px; object-fit: cover; }}
.ll-server-tile {{ flex: none; width: 56px; height: 56px; border-radius: 16px; background: {THEME_TILE}; color: {THEME_TILE_TEXT}; font-size: 20px; font-weight: 600; display: flex; align-items: center; justify-content: center; }}
.ll-server-name {{ min-width: 0; overflow-wrap: anywhere; }}
"""


@contextlib.contextmanager
def section(title):
    """A top-level settings card with a 20 px heading; reuse it for each tab section."""
    from nicegui import ui
    with ui.card().classes('ll-section w-full'):
        ui.label(title).classes('ll-section-title')
        yield


def apply_theme():
    from nicegui import ui
    ui.colors(primary=THEME_PRIMARY, negative=THEME_NEGATIVE, dark=THEME_CARD)
    ui.add_css(_THEME_CSS)


def server_initials(name):
    """Up to two uppercase initials from a server name; '?' when blank."""
    words = str(name or '').split()
    return ''.join(word[0] for word in words[:2]).upper() or '?'


def mount_dashboard(app):
    from nicegui import ui

    # Outlined fields app-wide (UI-28); default_props is a dict update, so repeating it is harmless.
    ui.input.default_props('outlined dense')
    ui.select.default_props('outlined dense')
    ui.textarea.default_props('outlined dense')
    _install_socket_auth(app)
    _install_upload_guard(app)
    _register_pages(app)
    ui.run_with(app, mount_path='/admin', title='llmcord admin', dark=True, reconnect_timeout=15,
                gzip_middleware_factory=None, on_air=None, show_welcome_message=False)


def _install_socket_auth(app):
    from nicegui import Client, core, ui

    core.sio.eio.cors_allowed_origins = [app.state.base_url]
    original_event = core.sio.handlers['/']['event']
    original_handshake = core.sio.handlers['/']['handshake']

    async def socket_check(sid, message, supplied_environ=None, *, check_permissions=True):
        # The live-call security boundary (SEC-03): the client must be bound to this cookie's session; CORS above limits origins.
        # Returns (allowed, client, error); client/error are set only once the client is confirmed bound to this session.
        client = Client.instances.get(message.get('client_id', ''))
        if not client:
            return False, None, None
        binding = getattr(client, 'llmcord_binding', None)
        if binding is None:
            return False, None, None
        environ = supplied_environ or core.sio.get_environ(sid) or {}
        cookies = SimpleCookie()
        cookies.load(environ.get('HTTP_COOKIE', ''))
        ident = cookies.get('llmcord_session')
        if not ident or ident.value != binding[0]:
            return False, None, None
        try:
            if binding[1] is not None and check_permissions:
                await app.state.auth.guard(binding[0], binding[1])
            else:
                await app.state.auth.session(binding[0])
            return True, client, None
        except HTTPException as error:
            return False, client, error

    async def socket_allowed(sid, message, supplied_environ=None, *, check_permissions=True):
        return (await socket_check(sid, message, supplied_environ, check_permissions=check_permissions))[0]

    @core.sio.on('handshake')
    async def handshake(sid, message):
        if not await socket_allowed(sid, message):
            return False
        return await original_handshake(sid, message)

    @core.sio.on('event')
    async def event(sid, message):
        allowed, client, error = await socket_check(sid, message)
        if allowed:
            original_event(sid, message)
        elif client is not None and (notice := rejection_notice(error)):
            # The event itself is still dropped; only the bound client is told why.
            with client:
                ui.notify(notice, type='negative', timeout=8000)

    original_connect = core.sio.handlers['/']['connect']
    @core.sio.on('connect')
    async def connect(sid, environ, auth=None):
        query = {k: v[0] for k, v in parse_qs(environ.get('QUERY_STRING', '')).items()}
        if query.get('implicit_handshake') == 'true' and not await socket_allowed(sid, query, environ):
            return False
        return await original_connect(sid, environ, auth)

    for name in ('javascript_response', 'ack', 'log'):
        original = core.sio.handlers['/'][name]
        async def guarded_socket(sid, message, original=original):
            # Delivery acknowledgements and JS results do not read guild data or
            # invoke admin actions. Recheck session/cookie binding here; action
            # events, uploads and AdminService operations still check Discord.
            if await socket_allowed(sid, message, check_permissions=False):
                value = original(sid, message)
                if inspect.isawaitable(value):
                    await value
        core.sio.on(name, guarded_socket)


def _install_upload_guard(app):
    from nicegui import Client

    @app.middleware('http')
    async def protected_uploads(request, call_next):
        path = request.url.path
        if path.startswith('/admin/_nicegui/client/') and '/upload/' in path:
            parts = path.split('/')
            client = Client.instances.get(parts[4]) if len(parts) > 4 else None
            binding = getattr(client, 'llmcord_binding', None)
            if not binding or request.cookies.get('llmcord_session') != binding[0]:
                return PlainTextResponse('Sign in with Discord', 401)
            try:
                await app.state.auth.guard(binding[0], binding[1], request.headers.get('x-csrf-token', ''), request.headers.get('origin'))
            except HTTPException as error:
                return PlainTextResponse(error.detail, error.status_code)
            if int(request.headers.get('content-length', 0)) > MAX_AVATAR_BYTES + 65536:
                return PlainTextResponse('Upload exceeds 8 MiB', 413)
            received = 0
            receive = request._receive
            async def bounded_receive():
                nonlocal received
                message = await receive()
                received += len(message.get('body', b''))
                if received > MAX_AVATAR_BYTES + 65536:
                    raise HTTPException(413, 'Upload exceeds 8 MiB')
                return message
            request._receive = bounded_receive
        return await call_next(request)


def _register_pages(app):
    from nicegui import ui

    @ui.page('/')
    async def servers(request: Request):
        apply_theme()
        try:
            session = await app.state.auth.session_for(request)
        except HTTPException:
            with ui.card().classes('ll-signin-card'):
                ui.label('l').classes('ll-logo').props('aria-hidden=true')
                ui.label('llmcord').classes('text-3xl font-bold')
                ui.label('Manage your characters, worlds, lore, and prompt presets.').classes('ll-muted')
                ui.link('Sign in with Discord', app.state.base_url + '/login').classes('ll-signin-btn')
                ui.label('You will see the servers where you are an administrator.').classes('ll-faint text-sm')
            return
        ui.context.client.llmcord_binding = (request.cookies['llmcord_session'], None)
        guilds = await app.state.auth.guilds(session)
        with ui.column().classes('ll-servers gap-1'):
            ui.label('Your servers').classes('text-3xl font-bold')
            ui.label('Servers where you are an administrator.').classes('ll-muted')
            with ui.element('div').classes('grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 w-full mt-4'):
                for guild in guilds:
                    if not is_server_admin(guild):
                        continue
                    card = ui.link(target=f"/guild/{guild['id']}").classes('ll-server-card')  # ui.link adds the /admin mount prefix
                    card.props['aria-label'] = str(guild['name'])
                    with card:
                        icon = guild_icon_url(guild, 128)
                        if icon:
                            ui.element('img').classes('ll-server-icon').props(f'src="{icon}" alt=""')
                        else:
                            ui.label(server_initials(guild['name'])).classes('ll-server-tile').props('aria-hidden=true')
                        ui.label(str(guild['name'])).classes('ll-server-name')
        signout(app, session)

    @ui.page('/guild/{guild_id}', response_timeout=30)
    async def guild(request: Request, guild_id: int):
        apply_theme()
        session = await app.state.auth.require_admin(request, guild_id)
        ui.context.client.llmcord_binding = (request.cookies['llmcord_session'], guild_id)
        ctx = LiveContext(app, request, guild_id, session)
        try:  # cached by require_admin; the header must not fail the page if a refetch does
            current = next((g for g in await app.state.auth.guilds(session) if str(g.get('id')) == str(guild_id)), None)
        except HTTPException:
            current = None
        with ui.header().classes('items-center justify-between no-wrap'):
            with ui.element('div').classes('ll-crumb'):
                crumb = ui.link('llmcord', '/')
                crumb.props['aria-label'] = 'llmcord / Servers'
                if current:
                    ui.label('/').classes('ll-crumb-sep').props('aria-hidden=true')
                    # guild_icon_url only returns a CDN URL built from a digit snowflake and a hex hash, so it is safe inside the quoted prop.
                    icon = guild_icon_url(current, 128)
                    if icon:
                        ui.element('img').classes('ll-crumb-icon').props(f'src="{icon}" alt=""')
                    else:
                        ui.label(server_initials(current.get('name'))).classes('ll-crumb-tile').props('aria-hidden=true')
                    ui.label(str(current.get('name', ''))).classes('ll-crumb-name')
            ui.label(session['user']['username']).classes('ml-2')
        with ui.column().classes('ll-page'):
            ui.label('Server administration').classes('text-3xl font-bold')
            ui.label('Changes apply on the next bot turn. Prompt drafts require activation.').classes('ll-muted')
            with ui.tabs().classes('w-full ll-tabs').props('align=left outside-arrows mobile-arrows') as tabs:
                setup = ui.tab('setup', 'Server setup')
                characters = ui.tab('characters', 'Characters')
                lore = ui.tab('lore', 'Lore')
                imports = ui.tab('imports', 'Imports')
                prompts = ui.tab('prompts', 'Prompt presets')
            selected_tab = request.query_params.get('tab')
            if selected_tab not in ('setup', 'characters', 'lore', 'imports', 'prompts'):
                selected_tab = 'setup'
            await ctx.load_channel_names()  # must precede any ctx.snapshot access: it bakes channel names into owner labels
            ui.add_css('.lore-drop-zone:empty::before { content: "Drop entries here"; color: #94a3b8; pointer-events: none; }')
            ui.add_css('.q-select:not(.owner-heading) { min-width: min(16rem, 100%); }')
            async def changed(event):
                ctx.set_url(tab=event.value)
                await ctx.build(event.value)
            ctx.builders = {'setup': lambda: setup_panel(ctx), 'characters': lambda: characters_panel(ctx),
                            'lore': lambda: lore_panel(ctx, on_import=lambda: ctx.selector.set_value('imports')),
                            'imports': lambda: imports_panel(ctx), 'prompts': lambda: presets_panel(ctx)}
            with ui.tab_panels(tabs, value=selected_tab, on_change=changed).classes('w-full') as ctx.selector:
                for tab in (setup, characters, lore, imports, prompts):
                    ctx.containers[tab.props['name']] = ui.tab_panel(tab)
            await ctx.build(selected_tab)
        signout(app, session)


def link_operation(store, guild_id, hub, world, enabled):
    pruned = (store.link_world if enabled else store.unlink_world)(guild_id, hub, world) or 0
    return {'hub': hub, 'world': world, 'enabled': enabled, 'pruned': pruned}


def link_detail(result):
    return {'hub': result['hub'], 'world': result['world'], 'enabled': result['enabled'], 'pruned': result['pruned']}


def bind_operation(store, guild_id, channel, space):
    dropped = store.bind_channel(guild_id, channel, space)
    names = [row['name'] for ident in dropped if (row := store.character_by_id(ident)) and row['guild_id'] == guild_id]
    return {'channel': channel, 'space': space, 'dropped': dropped, 'names': names}


def bind_detail(result):
    return {'channel': result['channel'], 'space': result['space'], 'dropped': result['dropped']}


_TIMEZONE_OPTIONS: list[str] = []


def timezone_options(current):
    if not _TIMEZONE_OPTIONS:
        _TIMEZONE_OPTIONS.extend(sorted(zoneinfo.available_timezones()))
    if isinstance(current, str) and current and current not in _TIMEZONE_OPTIONS:
        return sorted([*_TIMEZONE_OPTIONS, current])
    return list(_TIMEZONE_OPTIONS)


def timezone_operation(store, guild_id, name):
    old = store.guild_timezone(guild_id)
    store.set_guild_timezone(guild_id, name or '')
    return {'old': old, 'new': name or ''}


def timezone_detail(result):
    return {'old': result['old'], 'new': result['new']}


def signout(app, session):
    from nicegui import ui
    # A normal POST retains the existing origin and CSRF checks.
    import html
    ui.html('<form action="/logout" method="post"><input type="hidden" name="csrf" value="' + html.escape(session['csrf'], quote=True) + '"><button type="submit">Sign out</button></form>', sanitize=False)


async def setup_panel(ctx):
    from nicegui import ui
    store, gid = ctx.store, ctx.guild_id
    spaces = {r['id']: r['name'] + ' (' + r['kind'] + ')' for r in ctx.snapshot.spaces}
    channel_names = ctx.channel_names
    with section('Models and usage · last 24 hours'):
        models = ctx.service.model_config
        grouped = {}
        for role in ('dialogue', 'director', 'memory'):
            ident = models.get(role, models.get('dialogue'))
            if ident:
                grouped.setdefault(ident, []).append(role)
        rows = []
        for ident, roles in grouped.items():
            model = models.get('profiles', {}).get(ident, {}).get('model', '')
            summary = store.model_usage_summary(gid, ident, model)
            cost = f"${summary['cost_usd']:.6f}" + (f" + {summary['unpriced']} unpriced calls" if summary['unpriced'] else '')
            rows.append({'profile': ident, 'model': model, 'roles': ', '.join(roles), 'input': summary['input_tokens'], 'output': summary['output_tokens'], 'cost': cost, 'unreported': summary['unreported']})
        if rows:
            ui.table(columns=[{'name': key, 'field': key, 'label': label, 'align': 'left'} for key, label in
                              (('model', 'Model'), ('roles', 'Used for'), ('input', 'Input tokens'), ('output', 'Output tokens'), ('cost', 'Estimated USD'), ('unreported', 'Unreported calls'))], rows=rows, row_key='profile').classes('w-full')
            ui.label('Tracked for this server, including internal model calls. Cost uses list rates or your configured rates; it is not a billing statement. Refresh to update totals.').classes('ll-muted')
        else:
            ui.label('No model profiles are configured for the dashboard.')
    with section('Spaces'):
        for space in ctx.snapshot.spaces:
            with ui.expansion(f"{space['name']} · {space['kind']}").classes('space-card w-full rounded-lg'):
                guideline_editor(ctx, 'space', space['id'], 'World guidelines' if space['kind'] == 'world' else 'Hub guidelines')
                ui.button('Delete ' + space['kind'], icon='delete', color='negative', on_click=lambda space=space: delete_space_dialog(ctx, space))
        with ui.element('div').classes('ll-form-row'):
            name = ui.input('Space name')
            kind = ui.select(['world', 'hub'], value='world', label='Kind')
            ctx.button('Create space', lambda: store.create_space(gid, name.value or '', kind.value), 'space.create', then=lambda _: ctx.refresh())
    with section('Hub links'):
        hubs, worlds = ctx.snapshot.hubs, ctx.snapshot.worlds
        with ui.element('div').classes('ll-form-row'):
            hub = ui.select(hubs, label='Hub')
            world = ui.select(worlds, label='World')
            def link(enabled):
                return link_operation(store, gid, hub.value, world.value, enabled)
            def linked(result):
                if result['pruned']:
                    ui.notify(f"Removed {result['pruned']} cast entries no longer available in the hub.")
                return ctx.refresh()
            ctx.button('Link', lambda: link(True), 'hub.link', link_detail, then=linked)
            ctx.button('Unlink', lambda: link(False), 'hub.unlink', link_detail, then=linked).props('outline')
        for ident in hubs:
            ui.label(hubs[ident] + ': ' + ', '.join(worlds.get(x, str(x)) for x in store.allowed_worlds(ident)))
    with section('Channels and casts'):
        with ui.element('div').classes('ll-form-row'):
            channel = ui.select(channel_names, label='Discord text channel')
            space = ui.select(spaces, label='World or hub')
            def bind():
                if channel.value not in channel_names:
                    raise ValueError('Choose a text channel in this server')
                return bind_operation(store, gid, channel.value, space.value)
            def bound(result):
                names = result['names']
                if names:
                    shown = ', '.join(names[:5]) + (f' ...and {len(names) - 5} more' if len(names) > 5 else '')
                    ui.notify(f'Removed from the cast (not available there): {shown}.')
                else:
                    ui.notify('Cast kept.')
                return ctx.refresh()
            ctx.button('Bind channel', bind, 'channel.bind', bind_detail, then=bound)
        ui.label('Rebinding a channel keeps its ambient mode and removes cast members not available in the new space.').classes('ll-muted')
        for binding in ctx.snapshot.channels:
            with ui.card().classes('w-full channel-card'):
                ui.label(channel_names.get(binding['channel_id'], str(binding['channel_id']))).classes('text-lg font-bold')
                guideline_editor(ctx, 'channel', binding['channel_id'], 'Channel guidelines')
                options = {r['id']: r['name'] for r in store.eligible_characters(gid, binding['space_id'])}
                cast = ui.select(options, value=json.loads(binding['default_cast']), multiple=True, label='Default cast (up to five)').classes('w-full')
                def save_cast(binding=binding, cast=cast):
                    store.set_cast(binding['channel_id'], None, cast.value or [], default=True)
                    return True
                ctx.button('Save cast', save_cast, 'cast.default', success='Cast saved')
                ambient = ui.switch('Ambient participation', value=bool(binding['ambient']))
                def save_ambient(binding=binding, ambient=ambient):
                    store.set_ambient(binding['channel_id'], ambient.value)
                    return True
                ctx.button('Save ambient setting', save_ambient, 'channel.ambient', success='Ambient setting saved')
    with section('Reply footer'):
        footer = ui.switch('Show model and cost footer on replies', value=store.usage_footer_enabled(gid))
        def save_footer():
            store.set_usage_footer(gid, footer.value)
            return True
        ctx.button('Save footer setting', save_footer, 'settings.footer', success='Footer setting saved')
    with section('Timezone'):
        current = store.guild_timezone(gid)
        timezone = ui.select(timezone_options(current), value=current or None,
                             label='Server timezone', with_input=True, clearable=True).classes('w-full max-w-sm')
        ui.label('Characters use this for members who have not chosen their own with /time set. Without it they use UTC.').classes('ll-muted')
        ctx.button('Save timezone', lambda: timezone_operation(store, gid, timezone.value), 'settings.timezone',
                   timezone_detail, success='Timezone saved')
    with section('Avatar asset channel'):
        current_asset = store.asset_channel_id(gid)
        asset_channel = ui.select(channel_names, value=current_asset if current_asset in channel_names else None, label='Private text channel')
        async def configure():
            await ctx.service.avatars.configure(gid, asset_channel.value)
            return True
        ctx.button('Save asset channel', configure, 'avatar.channel', success='Asset channel saved')
        ui.label('Deny View Channel to @everyone and allow the bot to upload images. Images become Discord CDN assets.').classes('ll-muted')


def characters_panel(ctx):
    from nicegui import ui
    store, gid = ctx.store, ctx.guild_id
    worlds = ctx.snapshot.worlds
    def new_character():
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-lg'):
            ui.label('Create character').classes('text-xl font-bold')
            ui.label('Start with an empty card, then add a description, personality, and dialogue examples.').classes('ll-muted')
            name = ui.input('Character name').classes('w-full')
            world = ui.select(worlds, value=next(iter(worlds), None), label='Home world').classes('w-full')
            with ui.element('div').classes('ll-form-row justify-end'):
                ui.button('Cancel', on_click=dialog.close).props('flat')
                ctx.button('Create', lambda: store.create_character(gid, world.value, name.value or ''),
                           'character.create', then=lambda _: ctx.refresh('characters'))
        dialog.open()
    rows = ctx.snapshot.characters
    with section('Characters'):
        ui.button('Create character', icon='add', on_click=new_character).set_enabled(bool(worlds))
        if not worlds:
            ui.label('Create a home world in Server setup first.').classes('ll-muted')
        for row in rows:
            character_card(ctx, row, worlds)
        if not rows:
            ui.label('Create an empty character here or import a character card in Imports to get started.').classes('ll-muted')


def character_card(ctx, row, worlds):
    from nicegui import ui
    store, gid = ctx.store, ctx.guild_id
    with ui.expansion(row['name'] + (' · Archived' if row['archived'] else '')).classes('w-full character-card'):
        with ui.element('div').classes('ll-stack'):
            card = json.loads(row['card'])
            with ui.element('div').classes('ll-form-row'):
                name = ui.input('Name', value=row['name'])
                world = ui.select(worlds, value=row['world_id'], label='Home world')
            fields = {field: ui.textarea(label, value=card.get(field, '')).classes('w-full') for field, label in
                [('description', 'Description'), ('personality', 'Personality'), ('scenario', 'Scenario'), ('first_mes', 'Opening line'), ('mes_example', 'Example dialogue'), ('system_prompt', 'Card instructions'), ('post_history_instructions', 'Card post-history instructions')]}
            confirm = ui.checkbox('Confirm moving worlds; ineligible casts will be cleared')
            revision = store.owner_revision(gid, 'character', row['id'])
            def save(row=row, card=card, name=name, world=world, fields=fields, confirm=confirm, revision=revision):
                if world.value != row['world_id'] and not confirm.value:
                    raise ValueError('Confirm the world move before saving')
                new = {**card, **{key: control.value or '' for key, control in fields.items()}, 'name': name.value}
                store.update_character(gid, row['id'], world.value, name.value or '', new, expected_revision=revision)
                return True
            with ui.element('div').classes('ll-form-row'):
                ctx.button('Save character', save, 'character.edit', {'id': row['id']}, then=lambda _: ctx.refresh('characters'))
                def archive(row=row):
                    store.archive_character(gid, row['id'], not row['archived'])
                    return True
                ctx.button('Restore' if row['archived'] else 'Archive', archive, 'character.archive', {'id': row['id']}, then=lambda _: ctx.refresh('characters')).props('outline')
                def confirm_delete(row=row, revision=revision):
                    confirm_dialog(ctx, f"Delete {row['name']}?",
                                   ['Permanently delete this character, its lore, memories, and saved avatars, and remove it from all casts. Past Discord messages remain.',
                                    'This cannot be undone. Use Archive if you may want to restore the character later.'],
                                   'Delete permanently', lambda: store.delete_character(gid, row['id'], revision),
                                   'character.delete', {'id': row['id']}, then=lambda _: ctx.refresh('characters'))
                ui.button('Delete character', icon='delete', color='negative', on_click=confirm_delete)
            static_avatar_editor(ctx, row)
            ui.label('Emotion avatars').classes('ll-subtitle')
            ui.label('The selected emotion image is used first. If unavailable, the fallback static avatar is used.').classes('ll-muted')
            for slot in store.avatar_slots(gid, row['id']):
                avatar_editor(ctx, row['id'], slot)
            with ui.element('div').classes('ll-form-row'):
                key = ui.input('New emotion key').props('hint="lowercase letters, digits, underscores, hyphens"')
                label = ui.input('Emotion label')
                def add_slot(row=row, key=key, label=label):
                    if any(s['slot_key'] == key.value for s in store.avatar_slots(gid, row['id'])):
                        raise ValueError('That emotion key already exists; edit that emotion instead')
                    store.save_avatar(gid, row['id'], key.value or '', label.value or '', '')
                    return True
                ctx.button('Add emotion', add_slot, 'avatar.slot.create', then=lambda _: ctx.refresh())


def static_avatar_editor(ctx, character):
    from nicegui import ui
    with ui.expansion('Fallback static avatar').classes('w-full fallback-avatar ll-subpanel'), ui.element('div').classes('ll-stack'):
        if character['avatar']:
            ui.image(ctx.app.state.base_url + f"/guild/{ctx.guild_id}/characters/{character['id']}/avatar?v={avatar_version(character['avatar'])}").classes('w-24 h-24 rounded-lg')
        else:
            ui.label('No static fallback set. Import a card portrait or upload one here.').classes('ll-muted')
        ui.label('Used when the selected emotion has no usable image. No asset channel or publication is needed.').classes('ll-muted')
        revision = ctx.store.owner_revision(ctx.guild_id, 'character', character['id'])
        def reload(_):
            return ctx.refresh('characters')
        async def upload(event):
            ctx.store.save_static_avatar(ctx.guild_id, character['id'], normalize_avatar(await event.file.read()), revision)
            return True
        with ui.element('div').classes('ll-form-row'):
            ctx.upload(upload, 'Upload fallback avatar', 'avatar.fallback.edit', {'character': character['id']}, then=reload, success='Fallback avatar saved')
            if character['avatar']:
                def remove():
                    ctx.store.save_static_avatar(ctx.guild_id, character['id'], None, revision)
                    return True
                ui.button('Remove fallback avatar', icon='delete', color='negative', on_click=lambda: confirm_dialog(
                    ctx, 'Remove fallback avatar?', ['The character falls back to emotion images only. Emotion images stay.'],
                    'Remove fallback avatar', remove, 'avatar.fallback.delete', {'character': character['id']}, then=reload))


def avatar_editor(ctx, character_id, slot):
    from nicegui import ui
    with ui.expansion(slot['label']).classes('w-full ll-subpanel'), ui.element('div').classes('ll-stack'):
        if slot['has_image']:
            ui.image(ctx.app.state.base_url + f"/guild/{ctx.guild_id}/characters/{character_id}/avatars/{slot['slot_key']}?v={slot['image_version']}").classes('w-24 h-24 rounded-lg')
        ui.label('Stable key: ' + slot['slot_key']).classes('ll-muted')
        with ui.element('div').classes('ll-form-row'):
            label = ui.input('Label', value=slot['label'])
            description = ui.input('When to use this emotion', value=slot['description'])
        async def upload(event):
            image = normalize_avatar(await event.file.read())
            ctx.store.save_avatar(ctx.guild_id, character_id, slot['slot_key'], label.value or '', description.value or '', image, slot['revision'])
            return True
        ctx.upload(upload, 'Upload avatar image', 'avatar.slot.edit', {'character': character_id, 'slot': slot['slot_key']},
                   then=lambda _: ctx.refresh(), success='Emotion image saved')
        with ui.element('div').classes('ll-form-row'):
            def save():
                ctx.store.save_avatar(ctx.guild_id, character_id, slot['slot_key'], label.value or '', description.value or '', None, slot['revision'])
                return True
            ctx.button('Save emotion', save, 'avatar.slot.edit', {'character': character_id, 'slot': slot['slot_key']}, then=lambda _: ctx.refresh())
            async def publish():
                return await ctx.service.avatars.publish(ctx.guild_id, character_id, slot['slot_key'], repair=True)
            ctx.button('Publish / repair image', publish, 'avatar.publish', {'character': character_id, 'slot': slot['slot_key']}, success='Image published').props('outline')
            if slot['has_image']:
                def remove_image():
                    ctx.store.clear_avatar_image(ctx.guild_id, character_id, slot['slot_key'], slot['revision'])
                    return True
                ui.button('Remove emotion image', icon='delete', color='negative', on_click=lambda: confirm_dialog(
                    ctx, f"Remove the image from {slot['label']}?", ['The emotion and its label stay; only the image is removed.'],
                    'Remove emotion image', remove_image, 'avatar.image.delete', {'character': character_id, 'slot': slot['slot_key']},
                    then=lambda _: ctx.refresh('characters')))
            if slot['slot_key'] != 'neutral':
                def delete():
                    ctx.store.delete_avatar(ctx.guild_id, character_id, slot['slot_key'], slot['revision'])
                    return True
                ui.button('Remove emotion', icon='delete', color='negative', on_click=lambda: confirm_dialog(
                    ctx, f"Remove emotion {slot['label']}?", ['This removes the emotion and its image. The neutral emotion and the fallback avatar stay.'],
                    'Remove emotion', delete, 'avatar.slot.delete', then=lambda _: ctx.refresh()))


def lore_panel(ctx, on_import=None):
    from .lore_workspace import render_lore_workspace
    render_lore_workspace(ctx, entry_editor, on_import=on_import,
                          on_entry_import=lambda kind, ident, refresh: direct_import_dialog(ctx, kind, ident, refresh))


def entry_editor(ctx, entry, owners, refresh):
    from nicegui import ui
    with ui.card().classes('w-full') as panel:
        ui.label('Edit lore' if entry.get('ref') else 'Create lore').classes('text-xl font-bold')
        text = ui.textarea('Content', value=entry['content']).classes('w-full')
        rule = copy.deepcopy(entry['rule'])
        controls = {}
        for key in ('keys', 'secondary_keys'):
            controls[key] = ui.input(key.replace('_', ' ').title(), value=pretty(rule[key])).props('hint="JSON string array; preserves commas inside regex"').classes('w-full')
        controls['order'] = ui.number('Priority / insertion order (higher values appear later)', value=rule['order'], precision=0)
        with ui.row():
            controls['enabled'] = ui.checkbox('Enabled', value=rule['enabled'])
            controls['constant'] = ui.checkbox('Always active', value=rule['constant'])
            pinned = ui.checkbox('Pinned', value=entry['pinned'])
        with ui.expansion('Advanced activation and placement rules').classes('w-full'):
            for key, value in rule.items():
                if key in controls or key == 'unsupported':
                    continue
                if key == 'original':
                    with ui.expansion('Preserved import fields / unsupported feature remapping').classes('w-full'):
                        controls[key] = ui.textarea('Original fields (JSON)', value=pretty(value)).classes('w-full')
                elif key == 'regex_enabled':
                    controls[key] = ui.select({'auto': 'Detect /pattern/flags automatically', 'regex': 'Regex patterns', 'literal': 'Literal keywords'},
                        value='auto' if value is None else 'regex' if value else 'literal', label='Keyword matching')
                elif type(value) is bool:
                    controls[key] = ui.checkbox(key.replace('_', ' ').title(), value=value)
                elif type(value) is int:
                    controls[key] = ui.number(key.replace('_', ' ').title(), value=value, precision=0)
                elif key == 'role':
                    controls[key] = ui.select(['system', 'user', 'assistant'], value=value, label='Message role')
                elif key == 'position':
                    controls[key] = ui.select(list(dict.fromkeys(['before_char', 'after_char', 'before_examples', 'after_examples', 'in_chat', value])), value=value, label='Placement')
                elif key == 'scan_depth' or isinstance(value, list):
                    controls[key] = ui.input(key.replace('_', ' ').title(), value=pretty(value))
            for warning in rule.get('unsupported', []):
                ui.label(warning).classes('text-amber-300')
        def collect():
            for key, control in controls.items():
                if key == 'regex_enabled':
                    rule[key] = {'auto': None, 'regex': True, 'literal': False}[control.value]
                elif key == 'original' or key == 'scan_depth' or isinstance(rule.get(key), list):
                    rule[key] = json.loads(control.value or ('null' if key == 'scan_depth' else '{}'))
                elif type(rule.get(key)) is int:
                    rule[key] = int(control.value)
                else:
                    rule[key] = control.value
            return rule
        def save():
            ref = ctx.store.save_entry(ctx.guild_id, entry['owner_kind'], entry['owner_id'], text.value or '', collect(), pinned.value, entry.get('ref'), entry.get('revision'))
            return ref
        def saved(_):
            panel.delete()
            refresh()
        ctx.button('Save lore', save, 'lore.edit', then=saved)
        if entry.get('ref'):
            destination = ui.select(owners, label='Destination', with_input=True).classes('w-full')
            def transfer(copy):
                if not destination.value:
                    raise ValueError('Choose a destination')
                kind, ident = destination.value.split(':', 1)
                return ctx.store.transfer_entry(ctx.guild_id, entry['ref'], kind, int(ident), entry['revision'], copy=copy)
            ctx.button('Move', lambda: transfer(False), 'lore.move', then=saved)
            ctx.button('Copy', lambda: transfer(True), 'lore.copy', then=saved)
            def delete():
                ctx.store.delete_entry(ctx.guild_id, entry['ref'], entry['revision'])
                return True
            ui.button('Delete', color='negative', on_click=lambda: confirm_dialog(
                ctx, 'Delete this lore entry?', ['This also keeps your deletion choice for future reimports.'],
                'Delete entry', delete, 'lore.delete', then=saved))


def import_changes(changes):
    from nicegui import ui
    resolutions = {}
    for change in changes:
        with ui.expansion(f"{change['uid']} · {change['status']} · {change.get('disposition', '')}").classes('w-full'):
            ui.label('Current: ' + change.get('before', '')).classes('whitespace-pre-wrap')
            ui.label('Incoming: ' + change.get('after', '')).classes('whitespace-pre-wrap')
            for warning in change.get('warnings', []):
                ui.label(warning).classes('text-amber-300')
            if change['status'] == 'conflict':
                resolutions[change['uid']] = ui.select({'keep': 'Keep current entry', 'import': 'Use imported entry'}, label='Resolve conflict')
    return resolutions


def imports_panel(ctx):
    from nicegui import ui
    store, gid = ctx.store, ctx.guild_id
    ui.label('Character cards').classes('text-xl font-bold')
    worlds = ctx.snapshot.worlds
    world = ui.select(worlds, label='Home world')
    preview_area = ui.column().classes('w-full')
    async def card_uploaded(event):
        card = parse_card(event.file.name, await event.file.read())
        selected_world = world.value
        preview = store.preview_card(gid, selected_world, card)
        preview_area.clear()
        with preview_area:
            ui.label('Preview: ' + card.name).classes('text-xl font-bold')
            ui.label(card.data.get('description', '')).classes('whitespace-pre-wrap')
            if card.avatar:
                import base64
                ui.image('data:image/png;base64,' + base64.b64encode(card.avatar).decode()).classes('w-24 h-24')
            decisions = {key: ui.select({'keep': 'Keep current ' + key, 'import': 'Use imported ' + key}, label=f'Resolve {key} conflict') for key in preview['conflicts']}
            decisions.update(import_changes(preview['changes']))
            confirm = ui.checkbox('Confirm updating the existing character and home world') if preview['character_id'] else None
            expires = __import__('time').time() + 900
            def apply():
                if __import__('time').time() > expires:
                    raise ValueError('Preview expired; upload again')
                if confirm and not confirm.value:
                    raise ValueError('Confirm updating the existing character')
                return store.apply_card(gid, selected_world, card, {key: c.value for key, c in decisions.items()}, preview['revision'])
            ctx.button('Apply card import', apply, 'character.import', then=lambda _: ctx.refresh())
    ctx.upload(card_uploaded, 'Upload V2/V3 JSON or PNG card')
    ui.separator()
    ui.label('Direct lore entry import').classes('text-xl font-bold')
    owners = {f"{owner['kind']}:{owner['id']}": owner['label'] for owner in ctx.snapshot.owners}
    destination = ui.select(owners, label='Destination owner').classes('w-full')
    def import_into_owner():
        if destination.value not in owners:
            raise ValueError('Choose a destination owner')
        kind, ident = destination.value.split(':', 1)
        direct_import_dialog(ctx, kind, int(ident))
    ui.button('Import entries into selected owner', icon='upload_file', on_click=import_into_owner)
    ui.separator()
    ui.label('Named lorebooks').classes('text-xl font-bold')
    with ui.row().classes('items-end'):
        book_name = ui.input('Lorebook name')
        target = ui.select({'guild': 'Server lorebook', 'channel': 'Channel lorebook'}, value='guild', label='Lorebook scope')
        channels = {r['channel_id']: ctx.channel_names.get(r['channel_id'], str(r['channel_id'])) for r in ctx.snapshot.channels}
        channel = ui.select(channels, label='Channel (channel lorebooks only)').style('min-width: 20rem; max-width: 100%')
        ctx.button('Create lorebook', lambda: store.create_lorebook(gid, book_name.value or '', target.value, channel.value or 0), 'book.create', then=lambda _: ctx.refresh())
    for book in ctx.snapshot.lorebooks:
        with ui.expansion(book['name'] + ' · ' + ('server' if book['target_kind'] == 'guild' else book['target_kind'])).classes('w-full'):
            ui.button('Edit entries in Lore', icon='edit', on_click=lambda book=book: ctx.refresh('lore', owner=f"book:{book['id']}"))
            ui.button('Delete lorebook', icon='delete', color='negative', on_click=lambda book=book: delete_book_dialog(ctx, book))
            if book['target_kind'] == 'guild':
                linked_spaces = store.lorebook_links(book['id'])
                for space in ctx.snapshot.spaces:
                    enabled = space['id'] in linked_spaces
                    def assign(book=book, space=space, enabled=enabled):
                        store.set_lorebook_space(gid, book['id'], space['id'], not enabled)
                        return True
                    ctx.button(('Disable in ' if enabled else 'Enable in ') + space['name'], assign, 'book.assign', then=lambda _: ctx.refresh())
            area = ui.column().classes('w-full')
            async def book_uploaded(event, book=book, area=area):
                imported = parse_lorebook(await event.file.read())
                revision = store.owner_revision(gid, 'book', book['id'])
                changes = store.preview_import(gid, 'book', book['id'], imported)
                area.clear()
                with area:
                    label = 'RisuAI' if imported.source_format == 'risu' else 'SillyTavern'
                    ui.label(f'{label} lorebook detected · {len(imported.entries)} entries').classes('font-bold')
                    decisions = import_changes(changes)
                    expires = __import__('time').time() + 900
                    def apply():
                        if __import__('time').time() > expires:
                            raise ValueError('Preview expired; upload again')
                        return store.sync_lorebook(gid, book['id'], imported, {key: c.value for key, c in decisions.items()}, revision)
                    ctx.button('Apply lorebook sync', apply, 'book.sync', {'id': book['id']}, then=lambda _: ctx.refresh())
            ctx.upload(book_uploaded, 'Upload JSON lorebook for preview')


def presets_panel(ctx):
    from nicegui import ui
    store, gid = ctx.store, ctx.guild_id
    presets = store.list_presets(gid)
    choices = {0: 'Built-in default', **{r['id']: f"{r['name']} · draft {r['revision']}" for r in presets}}
    active = store.active_preset(gid)
    ui.label(f"Active preset: {choices.get(active['id'], str(active['id']))} · revision {active['revision']}").classes('text-xl font-bold')
    select = ui.select(choices, value=active['id'], label='Preset library').classes('w-full')
    purpose = ui.select({p: p.title() for p in PURPOSES}, value='dialogue', label='Purpose')
    selected = next((r for r in presets if r['id'] == active['id']), None)
    state = {'id': active['id'], 'revision': selected['revision'] if selected else 0, 'bundle': store.preset_bundle(gid, active['id']), 'raw_controls': {}}
    name = ui.input('Preset name', value=selected['name'] if selected else 'Default copy').classes('w-full')
    editor = ui.column().classes('w-full')
    diagnostics = ui.column().classes('w-full')

    def collect():
        for b, control in state['raw_controls'].values():
            b['raw'] = json.loads(control.value or '{}')
        return copy.deepcopy(state['bundle'])

    render_editor = _preset_editor(ctx, state, purpose, collect)

    async def load():
        bundle = await ctx.run(lambda: store.preset_bundle(gid, select.value))
        if bundle is None:
            return
        selected = next((r for r in store.list_presets(gid) if r['id'] == select.value), None)
        state.update(id=select.value, revision=selected['revision'] if selected else 0, bundle=bundle)
        name.value = selected['name'] if selected else 'Default copy'
        render_editor.refresh()

    def save():
        return store.save_preset(gid, name.value or '', collect(), state['id'] or None, state['revision'] if state['id'] else None)

    def saved(result):
        state['id'], state['revision'] = result
        fresh = store.list_presets(gid)
        select.set_options({0: 'Built-in default', **{r['id']: f"{r['name']} · draft {r['revision']}" for r in fresh}}, value=state['id'])
        ui.notify('Draft saved. Activate it when ready.', type='positive')

    ctx.button('Load selected preset', lambda: load())
    with editor:
        render_editor()
    async def change_purpose():
        if await ctx.run(lambda: True):
            collect()
            render_editor.refresh()
    purpose.on_value_change(lambda _: change_purpose())
    _preset_actions(ctx, state, name, collect, save, saved)
    _preset_import(ctx, state, name, render_editor)
    _preset_export(ctx, collect)
    _preset_preview(ctx, purpose, collect, diagnostics)


def _preset_editor(ctx, state, purpose, collect):
    from nicegui import ui

    @ui.refreshable
    def render_editor():
        state['raw_controls'] = {}
        blocks = state['bundle']['purposes'].setdefault(purpose.value, copy.deepcopy(default_bundle()['purposes'][purpose.value]))
        with ui.column().classes('w-full') as ordered:
            for b in blocks:
                with ui.card().classes('w-full prompt-block') as card:
                    card.prompt_id = b['id']
                    with ui.row().classes('items-center w-full'):
                        ui.icon('drag_indicator').classes('prompt-handle cursor-grab')
                        ui.input('Block name').bind_value(b, 'name').classes('flex-1')
                        ui.switch('Enabled').bind_value(b, 'enabled')
                    ui.label('Stable ID: ' + b['id']).classes('text-slate-400')
                    sources = list(dict.fromkeys(sorted(SOURCES) + [b['source']]))
                    ui.select(sources, label='Context source').bind_value(b, 'source').classes('w-full')
                    ui.textarea('Prompt text / template').bind_value(b, 'content').classes('w-full')
                    with ui.row():
                        ui.select(['system', 'user', 'assistant'], label='Role').bind_value(b, 'role')
                        ui.select(['relative', 'in_chat'], label='Placement').bind_value(b, 'placement')
                        ui.number('Depth', precision=0).bind_value(b, 'depth', backward=lambda v: int(v or 0))
                        ui.number('Injection order', precision=0).bind_value(b, 'order', backward=lambda v: int(v or 0))
                        ui.number('Trimming priority', precision=0).bind_value(b, 'priority', backward=lambda v: int(v or 0))
                    ui.select({'': 'Exact placement', 'top_system': 'Move to top-level system instructions', 'user': 'Convert late system block to user instructions'}, label='System instruction placement').bind_value(b, 'adaptation').classes('w-full')
                    with ui.expansion('Preserved import fields and compatibility remapping').classes('w-full'):
                        raw = ui.textarea('Original prompt fields (JSON)', value=pretty(b['raw'])).classes('w-full')
                        state['raw_controls'][b['id']] = (b, raw)
                        ui.label('To remap triggers/extensions: clear injection_trigger and set extension to false.')
                    async def remove(b=b):
                        if await ctx.run(lambda: True):
                            collect()
                            blocks.remove(b)
                            render_editor.refresh()
                    ui.button('Remove block', on_click=remove, color='negative')
            async def reordered(event):
                if await ctx.run(lambda: True):
                    collect()
                    by_id = {b['id']: b for b in blocks}
                    blocks[:] = [by_id[child.prompt_id] for child in ordered if hasattr(child, 'prompt_id')]
                    render_editor.refresh()
            ordered.make_sortable(handle='.prompt-handle', on_end=reordered, options={'draggable': '.prompt-block'})
        async def add():
            if await ctx.run(lambda: True):
                collect()
                import uuid
                blocks.append(block('custom-' + uuid.uuid4().hex[:8]))
                render_editor.refresh()
        ui.button('Add prompt block', on_click=add)
        for problem in compatibility(state['bundle'], ctx.service.providers):
            ui.label(problem).classes('text-amber-300')
        source = state['bundle'].setdefault('source', {})
        if source.get('assistant_prefill'):
            ui.checkbox('Disable imported assistant prefill').bind_value(source, 'prefill_disabled')

    return render_editor


def _preset_actions(ctx, state, name, collect, save, saved):
    from nicegui import ui
    store, gid = ctx.store, ctx.guild_id
    with ui.row():
        ctx.button('Save draft', save, 'preset.save', then=saved)
        def duplicate():
            return store.save_preset(gid, name.value or '', collect())
        ctx.button('Save as new preset', duplicate, 'preset.create', then=saved)
        def activate():
            store.activate_preset(gid, state['id'], state['revision'], ctx.service.providers)
            return True
        ctx.button('Activate saved revision', activate, 'preset.activate', then=lambda _: ctx.refresh())
        def delete():
            store.delete_preset(gid, state['id'])
            return True
        def ask_delete():
            if not state['id']:
                ui.notify('The built-in default cannot be deleted', type='negative', timeout=8000)
                return
            current = next((r for r in store.list_presets(gid) if r['id'] == state['id']), None)
            confirm_dialog(ctx, f"Delete preset {current['name'] if current else state['id']}?",
                           ['This permanently deletes the preset. The active preset cannot be deleted; activate another one first.'],
                           'Delete preset', delete, 'preset.delete', then=lambda _: ctx.refresh())
        ui.button('Delete preset', icon='delete', color='negative', on_click=ask_delete)


def _preset_import(ctx, state, name, render_editor):
    from nicegui import ui
    ui.separator()
    ui.label('Import preset').classes('text-xl font-bold')
    order_area = ui.column().classes('w-full')
    async def uploaded(event):
        data = await event.file.read()
        bundle, profiles = parse_preset(data)
        if profiles:
            order_area.clear()
            with order_area:
                order = ui.select({p['index']: 'SillyTavern order ' + p['label'] for p in profiles}, label='Choose an order profile')
                def choose():
                    if order.value is None:
                        raise ValueError('Choose an order profile')
                    return parse_preset(data, order.value)[0]
                ctx.button('Preview selected profile', choose, then=imported)
        else:
            imported(bundle)
    def imported(bundle):
        state.update(id=0, revision=0, bundle=bundle)
        name.value = 'Imported preset'
        render_editor.refresh()
        ui.notify('Import preview loaded. Resolve compatibility items, save a draft, then activate.')
    ctx.upload(uploaded, 'Upload native or SillyTavern Chat Completion JSON')


def _preset_export(ctx, collect):
    from nicegui import ui
    ui.separator()
    ui.label('Export preset').classes('text-xl font-bold')
    def native_export():
        ui.download.content(pretty(export_preset(collect())), 'llmcord-preset.json', 'application/json')
        return True
    ctx.button('Export full native bundle', native_export)
    omit = ui.input('Explicitly omit nonportable block IDs (JSON array)', value='[]').classes('w-full')
    def st_export():
        ui.download.content(pretty(export_preset(collect(), sillytavern=True, omit=json.loads(omit.value))), 'sillytavern-preset.json', 'application/json')
        return True
    ctx.button('Export SillyTavern dialogue preset', st_export)


def _preset_preview(ctx, purpose, collect, diagnostics):
    from nicegui import ui
    ui.separator()
    ui.label('Assembled request preview').classes('text-xl font-bold')
    chars = {r['id']: r['name'] for r in ctx.snapshot.characters}
    character = ui.select(chars, label='Sample character')
    channel = ui.select({row['channel_id']: ctx.channel_names.get(row['channel_id'], str(row['channel_id'])) for row in ctx.snapshot.channels}, label='Sample channel (optional)')
    sample = ui.textarea('Sample input', value='Hello!').classes('w-full')
    history = ui.textarea('Sample history (one message per line)').classes('w-full')
    def preview():
        request = ctx.service.preview_prompt(ctx.guild_id, collect(), purpose.value, character.value, sample.value or '', history.value or '', channel.value)
        diagnostics.clear()
        with diagnostics:
            ui.label(f'Estimated input tokens: {request.estimated_tokens}')
            ui.label('Omitted: ' + ', '.join(request.omitted))
            ui.label('Adaptations: ' + ', '.join(request.adaptations))
            for message in request.messages:
                with ui.expansion(message.role).classes('w-full'):
                    ui.label(message.text).classes('whitespace-pre-wrap')
        return True
    ctx.button('Preview without a model call', preview)
