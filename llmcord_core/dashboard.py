"""Component-based private administration mounted under /admin."""
from __future__ import annotations

import copy
import json
import sqlite3
import inspect
from urllib.parse import parse_qs
from http.cookies import SimpleCookie

from fastapi import HTTPException, Request
from fastapi.responses import PlainTextResponse

from .avatars import MAX_AVATAR_BYTES, avatar_version, normalize_avatar
from .cards import parse_card
from .lorebooks import parse_lorebook
from .prompts import PURPOSES, SOURCES, block, compatibility, default_bundle, export_preset, parse_preset
from .scene_ui import delete_book_dialog, delete_space_dialog, direct_import_dialog, guideline_editor


class LiveContext:
    def __init__(self, app, request, guild_id, session):
        self.app, self.guild_id = app, guild_id
        self.ident = request.cookies.get('llmcord_session', '')
        self.csrf = session['csrf']
        self.service = app.state.admin
        self.store = self.service.store
        self.lore_owner = request.query_params.get('owner')

    async def run(self, operation, action=None, detail=None):
        from nicegui import ui
        try:
            return await self.service.run(self.ident, self.guild_id, operation, action, detail)
        except (ValueError, HTTPException, sqlite3.IntegrityError, TypeError, KeyError) as error:
            message = error.detail if isinstance(error, HTTPException) else 'This name already exists' if isinstance(error, sqlite3.IntegrityError) else 'Choose valid values for all required fields' if isinstance(error, (TypeError, KeyError)) else str(error)
            ui.notify(message, type='negative', timeout=8000)
            return None

    def button(self, text, operation, action=None, detail=None, then=None, **kwargs):
        from nicegui import ui
        async def clicked():
            result = await self.run(operation, action, detail)
            if result is not None and then:
                then(result)
        return ui.button(text, on_click=clicked, **kwargs)

    def upload(self, handler, label):
        from nicegui import ui
        async def uploaded(event):
            # Upload endpoint also validates session binding and CSRF before reading the body.
            await self.run(lambda: handler(event))
        control = ui.upload(label=label, on_upload=uploaded, auto_upload=True, max_file_size=MAX_AVATAR_BYTES, max_files=1)
        control._props['headers'] = [{'name': 'X-CSRF-Token', 'value': self.csrf}]
        return control


def pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def mount_dashboard(app):
    from nicegui import Client, core, ui

    core.sio.eio.cors_allowed_origins = [app.state.base_url]
    original_event = core.sio.handlers['/']['event']
    original_handshake = core.sio.handlers['/']['handshake']

    async def socket_allowed(sid, message, supplied_environ=None, *, check_permissions=True):
        # The live-call security boundary (SEC-03): the client must be bound to this cookie's session; CORS above limits origins.
        client = Client.instances.get(message.get('client_id', ''))
        if not client:
            return False
        binding = getattr(client, 'llmcord_binding', None)
        if binding is None:
            return False
        environ = supplied_environ or core.sio.get_environ(sid) or {}
        cookies = SimpleCookie()
        cookies.load(environ.get('HTTP_COOKIE', ''))
        ident = cookies.get('llmcord_session')
        if not ident or ident.value != binding[0]:
            return False
        try:
            if binding[1] is not None and check_permissions:
                await app.state.auth.guard(binding[0], binding[1])
            else:
                await app.state.auth.session(binding[0])
            return True
        except HTTPException:
            return False

    @core.sio.on('handshake')
    async def handshake(sid, message):
        if not await socket_allowed(sid, message):
            return False
        return await original_handshake(sid, message)

    @core.sio.on('event')
    async def event(sid, message):
        if await socket_allowed(sid, message):
            original_event(sid, message)

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

    @ui.page('/')
    async def servers(request: Request):
        try:
            session = await app.state.auth.session_for(request)
        except HTTPException:
            with ui.card().classes('mx-auto mt-20 p-8 max-w-xl'):
                ui.label('llmcord').classes('text-3xl font-bold')
                ui.label('Manage your characters, worlds, lore, and prompt presets.')
                ui.link('Sign in with Discord', app.state.base_url + '/login').classes('text-lg')
            return
        ui.context.client.llmcord_binding = (request.cookies['llmcord_session'], None)
        guilds = await app.state.auth.guilds(session)
        ui.label('Your servers').classes('text-3xl font-bold')
        for guild in guilds:
            if guild.get('owner') or int(guild.get('permissions', '0')) & 8:
                with ui.card().classes('w-full max-w-xl'):
                    ui.link(guild['name'], f"/guild/{guild['id']}").classes('text-xl')
        signout(app, session)

    @ui.page('/guild/{guild_id}', response_timeout=30)
    async def guild(request: Request, guild_id: int):
        session = await app.state.auth.require_admin(request, guild_id)
        ui.context.client.llmcord_binding = (request.cookies['llmcord_session'], guild_id)
        ctx = LiveContext(app, request, guild_id, session)
        with ui.header().classes('items-center justify-between bg-slate-900'):
            ui.link('llmcord / Servers', '/').classes('text-white text-xl')
            ui.label(session['user']['username'])
        ui.label('Server administration').classes('text-3xl font-bold mt-4')
        ui.label('Changes apply on the next bot turn. Prompt drafts require activation.').classes('text-slate-400')
        with ui.tabs().classes('w-full') as tabs:
            setup = ui.tab('Server setup')
            characters = ui.tab('Characters')
            lore = ui.tab('Lore')
            imports = ui.tab('Imports')
            prompts = ui.tab('Prompt presets')
        selected_tab = {'characters': characters, 'lore': lore, 'imports': imports}.get(request.query_params.get('tab'), setup)
        with ui.tab_panels(tabs, value=selected_tab).classes('w-full'):
            with ui.tab_panel(setup):
                await setup_panel(ctx)
            with ui.tab_panel(characters):
                characters_panel(ctx)
            with ui.tab_panel(lore):
                lore_panel(ctx, on_import=lambda: tabs.set_value(imports))
            with ui.tab_panel(imports):
                imports_panel(ctx)
            with ui.tab_panel(prompts):
                presets_panel(ctx)
        signout(app, session)

    ui.run_with(app, mount_path='/admin', title='llmcord admin', dark=True, reconnect_timeout=15,
                gzip_middleware_factory=None, on_air=None, show_welcome_message=False)


def signout(app, session):
    from nicegui import ui
    # A normal POST retains the existing origin and CSRF checks.
    import html
    ui.html('<form action="/logout" method="post"><input type="hidden" name="csrf" value="' + html.escape(session['csrf'], quote=True) + '"><button type="submit">Sign out</button></form>', sanitize=False)


async def setup_panel(ctx):
    from nicegui import ui
    store, gid = ctx.store, ctx.guild_id
    spaces = {r['id']: r['name'] + ' (' + r['kind'] + ')' for r in store.list_spaces(gid)}
    channels = await ctx.run(lambda: ctx.service.avatars.channels(gid)) or []
    channel_names = {int(c['id']): '#' + c['name'] for c in channels if c['type'] == 0}
    ui.label('Models and usage · last 24 hours').classes('text-xl font-bold')
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
        ui.label('Tracked for this server, including internal model calls. Cost uses list rates or your configured rates; it is not a billing statement. Refresh to update totals.').classes('text-slate-400')
    else:
        ui.label('No model profiles are configured for the dashboard.')
    ui.separator()
    ui.label('Spaces').classes('text-xl font-bold')
    for space in store.list_spaces(gid):
        with ui.expansion(f"{space['name']} · {space['kind']}").classes('space-card w-full border rounded-lg'):
            guideline_editor(ctx, 'space', space['id'], 'World guidelines' if space['kind'] == 'world' else 'Hub guidelines')
            ui.button('Delete ' + space['kind'], icon='delete', color='negative', on_click=lambda space=space: delete_space_dialog(ctx, space))
    with ui.row().classes('items-end'):
        name = ui.input('Space name')
        kind = ui.select(['world', 'hub'], value='world', label='Kind')
        ctx.button('Create space', lambda: store.create_space(gid, name.value or '', kind.value), 'space.create', then=lambda _: ui.navigate.reload())
    ui.separator()
    ui.label('Hub links').classes('text-xl font-bold')
    hubs = {r['id']: r['name'] for r in store.list_spaces(gid) if r['kind'] == 'hub'}
    worlds = {r['id']: r['name'] for r in store.list_spaces(gid) if r['kind'] == 'world'}
    with ui.row().classes('items-end'):
        hub = ui.select(hubs, label='Hub')
        world = ui.select(worlds, label='World')
        def link(enabled):
            (store.link_world if enabled else store.unlink_world)(gid, hub.value, world.value)
            return True
        ctx.button('Link', lambda: link(True), 'hub.link', then=lambda _: ui.navigate.reload())
        ctx.button('Unlink', lambda: link(False), 'hub.unlink', then=lambda _: ui.navigate.reload())
    for ident in hubs:
        ui.label(hubs[ident] + ': ' + ', '.join(worlds.get(x, str(x)) for x in store.allowed_worlds(ident)))
    ui.separator()
    ui.label('Channels and casts').classes('text-xl font-bold')
    with ui.row().classes('items-end'):
        channel = ui.select(channel_names, label='Discord text channel')
        space = ui.select(spaces, label='World or hub')
        def bind():
            if channel.value not in channel_names:
                raise ValueError('Choose a text channel in this server')
            store.bind_channel(gid, channel.value, space.value)
            return True
        ctx.button('Bind channel', bind, 'channel.bind', then=lambda _: ui.navigate.reload())
    ui.label('Rebinding a channel resets its casts and ambient mode.').classes('text-amber-300')
    for binding in store.all('SELECT * FROM channels WHERE guild_id=?', (gid,)):
        with ui.card().classes('w-full channel-card'):
            ui.label(channel_names.get(binding['channel_id'], str(binding['channel_id']))).classes('text-lg font-bold')
            guideline_editor(ctx, 'channel', binding['channel_id'], 'Channel guidelines')
            options = {r['id']: r['name'] for r in store.eligible_characters(gid, binding['space_id'])}
            cast = ui.select(options, value=json.loads(binding['default_cast']), multiple=True, label='Default cast (up to five)').classes('w-full')
            def save_cast(binding=binding, cast=cast):
                store.set_cast(binding['channel_id'], None, cast.value or [], default=True)
                return True
            ctx.button('Save cast', save_cast, 'cast.default')
            ambient = ui.switch('Ambient participation', value=bool(binding['ambient']))
            def save_ambient(binding=binding, ambient=ambient):
                store.set_ambient(binding['channel_id'], ambient.value)
                return True
            ctx.button('Save ambient setting', save_ambient, 'channel.ambient')
    ui.separator()
    ui.label('Avatar asset channel').classes('text-xl font-bold')
    setting = store.one('SELECT asset_channel_id FROM guild_settings WHERE guild_id=?', (gid,))
    asset_channel = ui.select(channel_names, value=setting['asset_channel_id'] if setting and setting['asset_channel_id'] in channel_names else None, label='Private text channel')
    async def configure():
        await ctx.service.avatars.configure(gid, asset_channel.value)
        return True
    ctx.button('Save asset channel', configure, 'avatar.channel')
    ui.label('Deny View Channel to @everyone and allow the bot to upload images. Images become Discord CDN assets.')


def characters_panel(ctx):
    from nicegui import ui
    store, gid = ctx.store, ctx.guild_id
    def reload_characters(_):
        ui.navigate.to(f'/guild/{gid}?tab=characters')
    worlds = {r['id']: r['name'] for r in store.list_spaces(gid) if r['kind'] == 'world'}
    def new_character():
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-lg'):
            ui.label('Create character').classes('text-xl font-bold')
            ui.label('Start with an empty card, then add a description, personality, and dialogue examples.')
            name = ui.input('Character name').classes('w-full')
            world = ui.select(worlds, value=next(iter(worlds), None), label='Home world').classes('w-full')
            with ui.row():
                ui.button('Cancel', on_click=dialog.close)
                ctx.button('Create', lambda: store.create_character(gid, world.value, name.value or ''),
                           'character.create', then=reload_characters)
        dialog.open()
    ui.button('Create character', icon='add', on_click=new_character).set_enabled(bool(worlds))
    if not worlds:
        ui.label('Create a home world in Server setup first.')
    rows = store.all('SELECT * FROM characters WHERE guild_id=? ORDER BY name', (gid,))
    for row in rows:
        with ui.expansion(row['name'] + (' · Archived' if row['archived'] else '')).classes('w-full border rounded-lg character-card'):
            card = json.loads(row['card'])
            name = ui.input('Name', value=row['name'])
            world = ui.select(worlds, value=row['world_id'], label='Home world')
            fields = {field: ui.textarea(label, value=card.get(field, '')).classes('w-full') for field, label in
                [('description', 'Description'), ('personality', 'Personality'), ('scenario', 'Scenario'), ('first_mes', 'Opening line'), ('mes_example', 'Example dialogue'), ('system_prompt', 'Card instructions'), ('post_history_instructions', 'Card post-history instructions')]}
            confirm = ui.checkbox('Confirm moving worlds; ineligible casts will be cleared')
            revision = store.owner_revision(gid, 'character', row['id'])
            def save(row=row, card=card, name=name, world=world, fields=fields, confirm=confirm, revision=revision):
                if world.value != row['world_id'] and not confirm.value:
                    raise ValueError('Confirm the world move before saving')
                with store.write_admin():
                    if store.owner_revision(gid, 'character', row['id']) != revision:
                        raise ValueError('Character changed; reload before saving')
                    new = {**card, **{key: control.value or '' for key, control in fields.items()}, 'name': name.value}
                    target = store.space_by_id(world.value)
                    if not target or target['guild_id'] != gid or target['kind'] != 'world' or not name.value:
                        raise ValueError('Choose a home world and name')
                    store.db.execute('UPDATE characters SET card=?,name=?,world_id=? WHERE guild_id=? AND id=?', (json.dumps(new), name.value.strip(), world.value, gid, row['id']))
                    store._prune_character_casts(gid, row['id'])
                    if world.value != row['world_id']:
                        store._remove_from_thread_casts(row['id'])
                    store.bump_owner(gid, 'character', row['id'])
                return True
            ctx.button('Save character', save, 'character.edit', {'id': row['id']}, then=reload_characters)
            def archive(row=row):
                store.archive_character(gid, row['id'], not row['archived'])
                return True
            ctx.button('Restore' if row['archived'] else 'Archive', archive, 'character.archive', {'id': row['id']}, then=reload_characters)
            def confirm_delete(row=row, revision=revision):
                with ui.dialog() as dialog, ui.card().classes('w-full max-w-lg'):
                    ui.label(f"Delete {row['name']}?").classes('text-xl font-bold')
                    ui.label('Permanently delete this character, its lore, memories, and saved avatars, and remove it from all casts. Past Discord messages remain.')
                    ui.label('This cannot be undone. Use Archive if you may want to restore the character later.')
                    with ui.row():
                        ui.button('Cancel', on_click=dialog.close)
                        ctx.button('Delete permanently', lambda: store.delete_character(gid, row['id'], revision),
                                   'character.delete', {'id': row['id']}, then=reload_characters, color='negative')
                dialog.open()
            ui.button('Delete character', icon='delete', color='negative', on_click=confirm_delete)
            static_avatar_editor(ctx, row)
            ui.label('Emotion avatars').classes('text-xl font-bold')
            ui.label('The selected emotion image is used first. If unavailable, the fallback static avatar is used.')
            for slot in store.avatar_slots(gid, row['id']):
                avatar_editor(ctx, row['id'], slot)
            with ui.row().classes('items-end'):
                key = ui.input('New stable slot key').props('hint="lowercase letters, digits, underscores, hyphens"')
                label = ui.input('Emotion label')
                def add_slot(row=row, key=key, label=label):
                    if any(s['slot_key'] == key.value for s in store.avatar_slots(gid, row['id'])):
                        raise ValueError('That emotion key already exists; edit its slot instead')
                    store.save_avatar(gid, row['id'], key.value or '', label.value or '', '')
                    return True
                ctx.button('Add emotion', add_slot, 'avatar.slot.create', then=lambda _: ui.navigate.reload())
    if not rows:
        ui.label('Create an empty character here or import a character card in Imports to get started.')


def static_avatar_editor(ctx, character):
    from nicegui import ui
    with ui.expansion('Fallback static avatar').classes('w-full fallback-avatar'):
        if character['avatar']:
            ui.image(ctx.app.state.base_url + f"/guild/{ctx.guild_id}/characters/{character['id']}/avatar?v={avatar_version(character['avatar'])}").classes('w-24 h-24')
        else:
            ui.label('No static fallback set. Import a card portrait or upload one here.')
        ui.label('Used when the selected emotion has no usable image. No asset channel or publication is needed.')
        state = {'image': None}
        revision = ctx.store.owner_revision(ctx.guild_id, 'character', character['id'])
        async def upload(event):
            state['image'] = normalize_avatar(await event.file.read())
            ui.notify('Fallback image ready; save it to keep it')
        ctx.upload(upload, 'Upload fallback avatar')
        def save():
            if state['image'] is None:
                raise ValueError('Upload a fallback image before saving')
            ctx.store.save_static_avatar(ctx.guild_id, character['id'], state['image'], revision)
            return True
        def reload(_):
            ui.navigate.to(f'/guild/{ctx.guild_id}?tab=characters')
        ctx.button('Save fallback avatar', save, 'avatar.fallback.edit', {'character': character['id']}, then=reload)
        if character['avatar']:
            def remove():
                ctx.store.save_static_avatar(ctx.guild_id, character['id'], None, revision)
                return True
            ctx.button('Remove fallback avatar', remove, 'avatar.fallback.delete', {'character': character['id']}, then=reload)


def avatar_editor(ctx, character_id, slot):
    from nicegui import ui
    with ui.expansion(slot['label']).classes('w-full'):
        if slot['image']:
            ui.image(ctx.app.state.base_url + f"/guild/{ctx.guild_id}/characters/{character_id}/avatars/{slot['slot_key']}?v={avatar_version(slot['image'])}").classes('w-24 h-24')
        ui.label('Stable key: ' + slot['slot_key'])
        label = ui.input('Label', value=slot['label'])
        description = ui.input('When to use this emotion', value=slot['description'])
        state = {'image': None}
        async def upload(event):
            state['image'] = normalize_avatar(await event.file.read())
            ui.notify('Image ready; save the slot to keep it')
        ctx.upload(upload, 'Upload avatar image')
        def save():
            ctx.store.save_avatar(ctx.guild_id, character_id, slot['slot_key'], label.value or '', description.value or '', state['image'], slot['revision'])
            return True
        ctx.button('Save slot', save, 'avatar.slot.edit', {'character': character_id, 'slot': slot['slot_key']}, then=lambda _: ui.navigate.reload())
        async def publish():
            return await ctx.service.avatars.publish(ctx.guild_id, character_id, slot['slot_key'], repair=True)
        ctx.button('Publish / repair image', publish, 'avatar.publish', {'character': character_id, 'slot': slot['slot_key']})
        if slot['image']:
            def remove_image():
                ctx.store.clear_avatar_image(ctx.guild_id, character_id, slot['slot_key'], slot['revision'])
                return True
            ctx.button('Remove emotion image', remove_image, 'avatar.image.delete', {'character': character_id, 'slot': slot['slot_key']}, then=lambda _: ui.navigate.to(f'/guild/{ctx.guild_id}?tab=characters'))
        if slot['slot_key'] != 'neutral':
            def delete():
                ctx.store.delete_avatar(ctx.guild_id, character_id, slot['slot_key'], slot['revision'])
                return True
            ctx.button('Remove emotion', delete, 'avatar.slot.delete', then=lambda _: ui.navigate.reload())


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
            confirm = ui.checkbox('Confirm deleting this entry')
            def delete():
                if not confirm.value:
                    raise ValueError('Confirm deletion')
                ctx.store.delete_entry(ctx.guild_id, entry['ref'], entry['revision'])
                return True
            ctx.button('Delete', delete, 'lore.delete', then=saved, color='negative')


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
                resolutions[change['uid']] = ui.select({'keep': 'Keep local decision', 'import': 'Use imported entry'}, label='Resolve conflict')
    return resolutions


def imports_panel(ctx):
    from nicegui import ui
    store, gid = ctx.store, ctx.guild_id
    ui.label('Character cards').classes('text-xl font-bold')
    worlds = {r['id']: r['name'] for r in store.list_spaces(gid) if r['kind'] == 'world'}
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
            ctx.button('Apply card import', apply, 'character.import', then=lambda _: ui.navigate.reload())
    ctx.upload(card_uploaded, 'Upload V2/V3 JSON or PNG card')
    ui.separator()
    ui.label('Direct lore entry import').classes('text-xl font-bold')
    owners = {f"{owner['kind']}:{owner['id']}": owner['label'] for owner in ctx.service.owners(gid)}
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
        book_name = ui.input('Book name')
        target = ui.select(['guild', 'channel'], value='guild', label='Book scope')
        channels = {r['channel_id']: str(r['channel_id']) for r in store.all('SELECT * FROM channels WHERE guild_id=?', (gid,))}
        channel = ui.select(channels, label='Channel (for channel books)')
        ctx.button('Create book', lambda: store.create_lorebook(gid, book_name.value or '', target.value, channel.value or 0), 'book.create', then=lambda _: ui.navigate.reload())
    for book in store.list_lorebooks(gid):
        with ui.expansion(book['name'] + ' · ' + book['target_kind']).classes('w-full'):
            ui.button('Delete book', icon='delete', color='negative', on_click=lambda book=book: delete_book_dialog(ctx, book))
            if book['target_kind'] == 'guild':
                for space in store.list_spaces(gid):
                    enabled = space['id'] in store.lorebook_links(book['id'])
                    def assign(book=book, space=space, enabled=enabled):
                        store.set_lorebook_space(gid, book['id'], space['id'], not enabled)
                        return True
                    ctx.button(('Disable in ' if enabled else 'Enable in ') + space['name'], assign, 'book.assign', then=lambda _: ui.navigate.reload())
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
                    ctx.button('Apply lorebook sync', apply, 'book.sync', {'id': book['id']}, then=lambda _: ui.navigate.reload())
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
    with ui.row():
        ctx.button('Save draft', save, 'preset.save', then=saved)
        def duplicate():
            return store.save_preset(gid, name.value or '', collect())
        ctx.button('Save as new preset', duplicate, 'preset.create', then=saved)
        def activate():
            store.activate_preset(gid, state['id'], state['revision'], ctx.service.providers)
            return True
        ctx.button('Activate saved revision', activate, 'preset.activate', then=lambda _: ui.navigate.reload())
        def delete():
            if not state['id']:
                raise ValueError('The built-in default cannot be deleted')
            store.delete_preset(gid, state['id'])
            return True
        ctx.button('Delete preset', delete, 'preset.delete', then=lambda _: ui.navigate.reload(), color='negative')
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
    ui.separator()
    ui.label('Assembled request preview').classes('text-xl font-bold')
    chars = {r['id']: r['name'] for r in store.all('SELECT * FROM characters WHERE guild_id=?', (gid,))}
    character = ui.select(chars, label='Sample character')
    channel = ui.select({row['channel_id']: str(row['channel_id']) for row in store.all('SELECT channel_id FROM channels WHERE guild_id=?', (gid,))}, label='Sample channel (optional)')
    sample = ui.textarea('Sample input', value='Hello!').classes('w-full')
    history = ui.textarea('Sample history (one message per line)').classes('w-full')
    def preview():
        request = ctx.service.preview_prompt(gid, collect(), purpose.value, character.value, sample.value or '', history.value or '', channel.value)
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
