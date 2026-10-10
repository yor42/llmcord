"""Component-based private administration mounted under /admin."""
from __future__ import annotations

import contextlib
import copy
import functools
import json
import logging
import math
import os
import re
import sqlite3
import time
import inspect
import types
import zoneinfo
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import httpx
from urllib.parse import parse_qs
from http.cookies import SimpleCookie

from fastapi import HTTPException, Request
from fastapi.responses import PlainTextResponse

from .auth import guild_icon_url, is_server_admin, user_avatar_url
from .admin_store import ConflictError
from .backend import ROLES, effective_backend
from .avatars import MAX_AVATAR_BYTES, avatar_version, normalize_avatar
from .cards import parse_card
from .config import PROFILE_KEYS, format_key_ref, parse_key_ref, validate_profile_name
from .icons import icon_css, lucide, lucide_button, more_menu
from .lorebooks import parse_lorebook
from .macro_highlight import MacroHighlight
from .prompts import PURPOSES, SOURCES, block, compatibility, default_bundle, ST_MARKERS, export_preset, parse_preset, purpose_blocks
from .savebar import SaveBar
from .scene_ui import confirm_dialog, delete_book_dialog, delete_space_dialog, direct_import_dialog, guideline_editor


class OperatorContext:
    """Live context for the /operator page: every action re-checks the operator allowlist (AdminService.run_operator)."""
    def __init__(self, app, request):
        self.service = app.state.admin
        self.store = self.service.store
        self.ident = request.cookies.get('llmcord_session', '')

    async def _attempt(self, operation, action=None, detail=None):
        from nicegui import ui
        try:
            return True, await self.service.run_operator(self.ident, operation, action, detail)
        except (ValueError, HTTPException) as error:
            ui.notify(error.detail if isinstance(error, HTTPException) else str(error), type='negative', timeout=8000)
            return False, error

    async def read(self, operation):
        """A guarded read: the result, or None after a notified failure."""
        ok, result = await self._attempt(operation)
        return result if ok else None

    def button(self, text, operation, action=None, detail=None, then=None, success=None, conflict=None, **kwargs):
        from nicegui import ui
        async def clicked():
            ok, result = await self._attempt(operation, action, detail)
            if not ok:
                if conflict and isinstance(result, ConflictError):
                    followup = conflict()
                    if inspect.isawaitable(followup):
                        await followup
                return
            if success:
                ui.notify(success, type='positive')
            if then:
                followup = then(result)
                if inspect.isawaitable(followup):
                    await followup
        return ui.button(text, on_click=clicked, **kwargs)


def budget_panel(ctx):
    from nicegui import ui
    from . import budget
    with section('Spending caps'):
        body = ui.column().classes('ll-stack w-full')

    def render():
        state = budget.state(ctx.store)
        body.clear()
        with body:
            ui.label('Applies to the whole bot. Operators get a DM when spending passes each cap. At the hard cap, new character replies, '
                     'ambient turns, catch-ups and memory updates pause until the reset day or until you raise the cap.').classes('ll-muted')
            ui.label(f'{format_usd(state.spent_usd)} spent since {state.period}. The period resets on {state.resets_on.isoformat()} (UTC).')
            if state.hard_reached:
                ui.label('Hard cap reached: character replies are paused.').classes('font-bold')
            elif state.soft_reached:
                ui.label('Soft cap reached: operators were warned.').classes('font-bold')
            if state.unpriced_calls:
                n = state.unpriced_calls
                ui.label(f'{n} model call{"" if n == 1 else "s"} without a cost estimate this period {"was" if n == 1 else "were"} counted as $0. '
                         'Set prices for those models to include them.').classes('text-warning')
            with ui.element('div').classes('ll-form-row'):
                soft = ui.number('Soft cap (USD)', value=state.soft_cap_usd, min=0, step=1).props('hint="Blank means off" clearable')
                hard = ui.number('Hard cap (USD)', value=state.hard_cap_usd, min=0, step=1).props('hint="Blank means off" clearable')
                day = ui.select(list(range(1, 29)), value=state.reset_day, label='Reset day').props('hint="Day of the month, UTC"')
            notice = ui.switch('Post a notice in the channel when the hard cap pauses a reply (at most once an hour per channel)', value=state.channel_notice)

            def values():
                def cap(control, name):
                    if control.value in (None, ''):
                        return None
                    try:
                        return float(control.value)
                    except (TypeError, ValueError):
                        raise ValueError(f'{name} cap must be blank or a number of dollars, zero or more.')
                if isinstance(day.value, bool) or not isinstance(day.value, int) or not 1 <= day.value <= 28:
                    raise ValueError('Reset day must be a whole number from 1 to 28.')
                return cap(soft, 'Soft'), cap(hard, 'Hard'), day.value, bool(notice.value)

            def save():
                return ctx.store.save_budget(*values(), state.revision)

            def detail():
                soft_cap, hard_cap, reset_day, channel_notice = values()
                return {'soft_cap_usd': soft_cap, 'hard_cap_usd': hard_cap, 'reset_day': reset_day, 'channel_notice': channel_notice}
            with ui.element('div').classes('ll-form-row'):
                ctx.button('Save', save, 'budget.settings', lambda _result: detail(), then=lambda _result: render(), success='Spending caps saved.', conflict=render)
    render()

ROLE_LABELS = {'dialogue': 'Dialogue', 'director': 'Director', 'memory': 'Memory'}
PROFILE_SOURCES = {'config': 'config.yaml', 'dashboard': 'Dashboard', 'override': 'Dashboard (overrides config.yaml)', 'skipped': 'Dashboard (skipped, see warning)'}
BACKEND_NOTE = ('Changes apply from the next message. "Set" means this dashboard\'s environment has the variable; '
                'the bot checks its own environment when it calls the model.')
KEY_HINT = 'A ${NAME} reference to a variable ending in _API_KEY. Never paste the key itself.'


def profile_mapping_of(profile):
    """The stored shape of a ModelProfile: PROFILE_KEYS with a value."""
    return {key: getattr(profile, key) for key in PROFILE_KEYS if getattr(profile, key) is not None}


def key_status(name, environ=os.environ):
    """('none' | 'set' | 'missing', text) for a profile's key reference; only whether the variable has a value is read."""
    if not name:
        return 'none', 'No key'
    state = 'set' if environ.get(name) else 'missing'
    return state, f'{format_key_ref(name)} ({state})'


def profile_view(backend, config_profiles, db_rows, environ=os.environ):
    """Row view-model for the Profiles list, sorted by name; never carries an environment value."""
    revisions = {row['name']: row['revision'] for row in db_rows}
    badges = {}
    for role in ROLES:
        if backend.roles.get(role):
            badges.setdefault(backend.roles[role], []).append(role)
    views = []
    for name in sorted(backend.profiles):
        profile = backend.profiles[name]
        kind = ('override' if name in config_profiles else 'dashboard') if profile.source == 'dashboard' else 'config'
        state, key = key_status(profile.api_key_env, environ)
        views.append({'name': name, 'provider': profile.provider, 'model': profile.model, 'key': key, 'key_state': state, 'kind': kind,
                      'source': PROFILE_SOURCES[kind], 'revision': revisions.get(name) if kind != 'config' else None,
                      'roles': badges.get(name, []), 'initial': profile_mapping_of(profile)})
    for row in db_rows:  # saved rows the merge skipped stay listed so they can be fixed or deleted
        name = row['name']
        try:
            validate_profile_name(name)
        except ValueError:
            continue
        if name in backend.profiles and backend.profiles[name].source == 'dashboard':
            continue
        data = row['data'] if isinstance(row['data'], dict) else {}
        ref = data.get('api_key_env') if isinstance(data.get('api_key_env'), str) else None
        state, key = key_status(ref, environ)
        views.append({'name': name, 'provider': str(data.get('provider', '')), 'model': str(data.get('model', '')), 'key': key, 'key_state': state,
                      'kind': 'skipped', 'source': PROFILE_SOURCES['skipped'], 'revision': row['revision'], 'roles': [],
                      'initial': {k: v for k, v in data.items() if k in PROFILE_KEYS}})
    return sorted(views, key=lambda view: view['name'])


def role_options(saved_value, backend_profiles, config_role):
    """Options for a role select: use config.yaml, each available profile, and a saved value that no longer resolves."""
    options = {'': f'Use config.yaml ({config_role or "none"})'}
    options.update({name: name for name in backend_profiles})
    if saved_value and saved_value not in options:
        options[saved_value] = f'{saved_value} (not available)'
    return options


def editor_initial(mapping):
    """Editor starting values: a saved provider, billing tier or reasoning effort the selects cannot show falls back to the default (or is dropped)."""
    initial = dict(mapping)
    for field, allowed, default in (('provider', ('openai', 'anthropic', 'compatible'), 'compatible'), ('billing_tier', ('paid', 'free'), 'paid'),
                                    ('reasoning_effort', ('none', 'minimal', 'low', 'medium', 'high'), None)):
        if field in initial and initial[field] not in allowed:
            initial.pop(field)
            if default:
                initial[field] = default
    return initial


def _number(value, label, whole=False):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        if isinstance(value, bool):
            raise ValueError
        number = float(value.strip() if isinstance(value, str) else value)
        if not math.isfinite(number):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f'{label} must be a number.') from None
    if whole:
        if not number.is_integer():
            raise ValueError(f'{label} must be a whole number.')
        return int(number)
    return number


def profile_mapping(values):
    """Form values -> a config.yaml-shaped mapping with only the keys that apply. The key text is parsed, never echoed in errors."""
    provider = values.get('provider')
    model = (values.get('model') or '').strip()
    if not model:
        raise ValueError('Model is required.')
    tokens = _number(values.get('context_tokens'), 'Context tokens', whole=True)
    if tokens is None or tokens < 1:
        raise ValueError('Context tokens must be a whole number of at least 1.')
    mapping = {'provider': provider, 'model': model, 'context_tokens': tokens, 'supports_images': bool(values.get('supports_images')),
               'structured_outputs': bool(values.get('structured_outputs')), 'stream_usage': bool(values.get('stream_usage', True)),
               'billing_tier': values.get('billing_tier') or 'paid'}
    key = parse_key_ref(values.get('api_key'))
    if key:
        mapping['api_key_env'] = key
    base_url = (values.get('base_url') or '').strip()
    if base_url and provider != 'anthropic':
        mapping['base_url'] = base_url
    if values.get('reasoning_effort') and provider == 'compatible':
        mapping['reasoning_effort'] = values['reasoning_effort']
    for field, label in (('input_cost_per_million', 'Input cost'), ('output_cost_per_million', 'Output cost'), ('cached_input_cost_per_million', 'Cached input cost'),
                         ('timeout_seconds', 'Timeout')):
        number = _number(values.get(field), label)
        if number is not None:
            mapping[field] = number
    retries = _number(values.get('max_retries'), 'Max retries', whole=True)
    if retries is not None:
        mapping['max_retries'] = retries
    return mapping


def changed_fields(mapping, initial):
    """Names of the fields that differ from `initial` (all of them for a new profile); names only, never values."""
    initial = initial or {}
    return sorted(key for key in set(mapping) | set(initial) if mapping.get(key) != initial.get(key))


def roles_detail(old, new):
    return {role: {'from': old.get(role), 'to': new.get(role)} for role in ROLES if old.get(role) != new.get(role)}


async def backend_panel(ctx, config):
    """Backend tab: role assignment and the profile list/editor. `config` has .profiles, .roles and .error from config.yaml."""
    from nicegui import ui
    from .scene_ui import confirm_dialog
    config_names = set(config.profiles)
    current_views = []
    with section('Roles'):
        roles_body = ui.column().classes('ll-stack w-full')
    with section('Profiles'):
        profiles_body = ui.column().classes('ll-stack w-full')

    async def render():
        state = await ctx.read(lambda: (ctx.store.model_profile_rows(), ctx.store.model_roles()))
        if state is None:
            return
        rows, saved = state
        backend = effective_backend(config.profiles, config.roles, rows, saved)
        views = profile_view(backend, config.profiles, rows)
        current_views[:] = views
        roles_body.clear()
        profiles_body.clear()
        with roles_body:
            render_roles(saved, backend)
        with profiles_body:
            render_profiles(views, backend)

    def render_roles(saved, backend):
        selects = {}
        with ui.element('div').classes('ll-form-row'):
            for role in ROLES:
                selects[role] = ui.select(role_options(saved.get(role), backend.profiles, config.roles.get(role)), value=saved.get(role) or '', label=ROLE_LABELS[role]).classes('ll-wide')
        ui.label('A role set here replaces the config.yaml role of the same name.').classes('ll-muted')

        def chosen():
            return {role: selects[role].value or None for role in ROLES}
        with ui.element('div').classes('ll-form-row'):
            ctx.button('Save roles', lambda: ctx.store.save_model_roles(chosen(), saved['revision'], config_names),
                       'backend.roles', lambda _result: roles_detail(saved, chosen()), then=lambda _result: render(), success='Roles saved.', conflict=render)

    def render_profiles(views, backend):
        if config.error:
            ui.label(f'config.yaml could not be read, so only dashboard profiles are listed. {config.error}').classes('text-warning')
        for problem in backend.problems:
            ui.label(problem).classes('text-warning')
        if not views:
            ui.label('No profiles yet.').classes('ll-muted')
        for view in views:
            with ui.element('div').classes('ll-stack ll-result w-full'):
                with ui.element('div').classes('ll-form-row'):
                    ui.label(view['name']).classes('font-bold')
                    for role in view['roles']:
                        ui.badge(f'{ROLE_LABELS[role]} role').props('outline')
                ui.label(f"{view['provider']} · {view['model']}")
                ui.label(f"Key: {view['key']}").classes('ll-muted')
                ui.label(f"From: {view['source']}").classes('ll-muted')
                with ui.element('div').classes('ll-form-row'):
                    edit = ui.button('Edit', on_click=lambda v=view: open_editor(v)).props('outline size=sm')
                    edit.props['aria-label'] = f"Edit {view['name']}"
                    if view['kind'] in ('dashboard', 'skipped'):
                        drop = ui.button('Delete', on_click=lambda v=view: confirm_drop(v)).props('outline size=sm color=negative')
                        drop.props['aria-label'] = f"Delete {view['name']}"
                    elif view['kind'] == 'override':
                        reset = ui.button('Reset to config.yaml', on_click=lambda v=view: confirm_drop(v)).props('outline size=sm')
                        reset.props['aria-label'] = f"Reset to config.yaml for {view['name']}"
        with ui.element('div').classes('ll-form-row'):
            lucide_button('Add profile', 'plus', on_click=lambda: open_editor(None))
        ui.label(BACKEND_NOTE).classes('ll-muted')

    def confirm_drop(view):
        name, reset = view['name'], view['kind'] == 'override'
        lines = (['The config.yaml version of this profile applies again. Edits made on the dashboard are lost.'] if reset else
                 ['This removes the saved profile. Roles assigned to it must be changed first.'])
        confirm_dialog(ctx, f'Reset {name} to config.yaml?' if reset else f'Delete {name}?', lines, 'Reset profile' if reset else 'Delete profile',
                       lambda: ctx.store.delete_model_profile(name, view['revision'], config_names), 'backend.profile_delete', {'name': name},
                       then=lambda _result: render(), conflict=lambda: render())

    def open_editor(view):
        editing = view is not None
        initial = editor_initial(view['initial']) if editing else {}
        revision = view['revision'] if editing else None
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-2xl'):
            ui.label(f"Edit {view['name']}" if editing else 'Add profile').classes('text-xl font-bold')
            with ui.column().classes('ll-stack w-full'):
                name = ui.input('Name', value=view['name'] if editing else '').classes('w-full')
                if editing:
                    name.props('readonly')
                else:
                    name.props('hint="Lowercase letters, digits and . _ / -"')
                provider = ui.select(['openai', 'anthropic', 'compatible'], value=initial.get('provider', 'compatible'), label='Provider').classes('w-full')
                model = ui.input('Model', value=initial.get('model', '')).classes('w-full')
                key = ui.input('API key', value=format_key_ref(initial['api_key_env']) if initial.get('api_key_env') else '',
                               placeholder='${OPENAI_API_KEY}').classes('w-full').props(f'hint="{KEY_HINT}"')
                base = ui.input('Base URL', value=initial.get('base_url', '')).classes('w-full').props('hint="Required for compatible providers."')
                base.bind_visibility_from(provider, 'value', lambda value: value != 'anthropic')
                tokens = ui.number('Context tokens', value=initial.get('context_tokens'), min=1, step=1, format='%d').classes('w-full')
                images = ui.switch('Supports images', value=initial.get('supports_images', False))
                with ui.expansion('Advanced').classes('w-full'):
                    with ui.column().classes('ll-stack w-full'):
                        effort = ui.select(['', 'none', 'minimal', 'low', 'medium', 'high'], value=initial.get('reasoning_effort') or '',
                                           label='Reasoning effort').classes('w-full').props('hint="Compatible providers only"')
                        effort.bind_visibility_from(provider, 'value', lambda value: value == 'compatible')
                        structured = ui.switch('Structured outputs', value=initial.get('structured_outputs', False))
                        usage = ui.switch('Stream usage', value=initial.get('stream_usage', True))
                        tier = ui.select(['paid', 'free'], value=initial.get('billing_tier', 'paid'), label='Billing tier').classes('w-full')
                        costs = {field: ui.number(label, value=initial.get(field), min=0).props('clearable hint="Blank means none"').classes('w-full')
                                 for field, label in (('input_cost_per_million', 'Input cost per million (USD)'), ('output_cost_per_million', 'Output cost per million (USD)'),
                                                      ('cached_input_cost_per_million', 'Cached input cost per million (USD)'))}
                        timeout = ui.number('Timeout (seconds)', value=initial.get('timeout_seconds', 120), min=0).classes('w-full')
                        retries = ui.number('Max retries', value=initial.get('max_retries', 1), min=0, step=1, format='%d').classes('w-full')

            def mapping():
                return profile_mapping({'provider': provider.value, 'model': model.value, 'api_key': key.value, 'base_url': base.value, 'context_tokens': tokens.value,
                                        'supports_images': images.value, 'reasoning_effort': effort.value, 'structured_outputs': structured.value,
                                        'stream_usage': usage.value, 'billing_tier': tier.value, 'timeout_seconds': timeout.value,
                                        'max_retries': retries.value, **{field: control.value for field, control in costs.items()}})

            def save():
                new_name = (name.value or '').strip()
                if not editing:
                    validate_profile_name(new_name)
                    if new_name in config_names or any(v['name'] == new_name for v in current_views):
                        raise ValueError(f'A profile named {new_name} already exists. Use Edit to change it.')
                return ctx.store.save_model_profile(new_name, mapping(), revision)

            def detail(_result):
                return {'name': (name.value or '').strip(), 'created': revision is None, 'fields': changed_fields(mapping(), initial if editing else None)}

            async def saved(_result):
                dialog.close()
                await render()

            async def stale():
                if editing:
                    dialog.close()
                    await render()
            running = False

            async def probe():
                nonlocal running
                from . import backend as backend_module
                from .config import key_hosts_from_env
                if running:
                    raise ValueError('A connection test is already running.')
                running = True
                try:
                    probe_name = (name.value or '').strip()
                    if not editing:
                        validate_profile_name(probe_name)
                    values = mapping()
                    # Test seam only: tests/dashboard_server.py sets this; production uses the real probe.
                    tester = getattr(ctx.service.app.state, 'connection_tester', None) or backend_module.test_profile_connection
                    test.disable()
                    return await tester(probe_name or 'profile', values, key_hosts_from_env())
                finally:
                    running = False
                    if not dialog.is_deleted:
                        test.enable()

            def tested(result):
                if dialog.is_deleted:
                    return
                outcome.clear()
                with outcome:
                    ui.label(result.message).classes('text-positive' if result.ok else 'text-negative').props('role=status')
            with ui.element('div').classes('ll-form-row justify-end'):
                ui.button('Cancel', on_click=dialog.close).props('flat')
                test = ctx.button('Test connection', probe, 'backend.test', lambda result: {'name': (name.value or '').strip(), 'ok': result.ok}, then=tested).props('outline')
                ctx.button('Save profile', save, 'backend.profile', detail, then=saved, success='Profile saved.', conflict=stale)
            outcome = ui.column().classes('w-full')
            ui.label('Sends one short request from the dashboard with these settings. It is not counted in usage.').classes('ll-muted')
        dialog.on('hide', dialog.delete)
        dialog.open()

    await render()


OPERATOR_USAGE_PERIODS = {'period': 'This spending period', '7': 'Last 7 days', '30': 'Last 30 days'}


def operator_usage_since(key, reset_day, now):
    from . import budget
    if key == 'period':
        return datetime.combine(budget.period_start(now, reset_day), datetime.min.time(), tzinfo=ZoneInfo('UTC')).timestamp()
    return now - int(key) * 86400


def server_usage_label(guild_id, names):
    if not guild_id:
        return 'No server'
    return names.get(str(guild_id)) or f'Server (ID …{str(guild_id)[-4:]})'


async def usage_by_server_panel(ctx, guild_names):
    from nicegui import ui
    try:  # names come from the operator's own Discord guild list; any failure falls back to the server ID
        names = {str(g.get('id')): str(g.get('name', '')) for g in await guild_names() if g.get('name')}
    except (httpx.HTTPError, ValueError, HTTPException):
        names = {}
    with section('Usage by server'):
        select = ui.select(OPERATOR_USAGE_PERIODS, value='period', label='Period')
        body = ui.column().classes('w-full gap-4')

        async def render():
            now = time.time()
            report = await ctx.read(lambda: ctx.store.usage_by_guild(operator_usage_since(select.value, ctx.store.budget_settings()['reset_day'], now), now))
            body.clear()
            if report is None:
                return
            with body:
                if not report['rows']:
                    ui.label('No model calls in this period.')
                    return
                def cells(label, r):
                    return {'key': label, 'server': label, 'requests': f"{r['requests']:,}", 'input': f"{r['input_tokens']:,}", 'output': f"{r['output_tokens']:,}",
                            'cost': format_usd(r['cost_usd']), 'unpriced': r['unpriced']}
                rows = [cells(server_usage_label(r['guild_id'], names), r) | {'key': str(r['guild_id'])} for r in report['rows']]
                usage_table([('server', 'Server'), ('requests', 'Requests'), ('input', 'Input tokens'), ('output', 'Output tokens'), ('cost', 'Estimated USD'), ('unpriced', 'Without cost estimate')],
                            [*rows, cells('Total', report['totals']) | {'key': 'total'}], 'key')
                ui.label('Each server keeps usage for its own period (Server settings), so this table can show less than the Spending tab.').classes('ll-muted')
                ui.label('Cost uses list rates or your configured rates; it is not a billing statement.').classes('ll-muted')
        select.on_value_change(render)
        await render()


class AttemptFailed(tuple):
    """(False, None) that remembers the error, so a caller can tell its own attempt's failure kind."""
    def __new__(cls, pair, error):
        self = super().__new__(cls, pair)
        self.error = error
        return self


class LiveContext:
    def __init__(self, app, request, guild_id, session):
        self.app, self.guild_id = app, guild_id
        self.is_operator = False  # set by the page, which knows the session's role
        self.ident = request.cookies.get('llmcord_session', '')
        self.csrf = session['csrf']
        self.service = app.state.admin
        self.store = self.service.store
        self.lore_owner = request.query_params.get('owner')
        self.channel_names, self.thread_names, self.thread_scopes = {}, {}, None
        self.channels_failed = False
        self.selector, self.containers, self.builders, self.built = None, {}, {}, set()
        self.savebar = None

    async def load_channel_names(self):
        # One Discord channel fetch per page render; a failure leaves the mapping empty.
        from nicegui import ui
        try:
            channels = await self.service.run(self.ident, self.guild_id, lambda: self.service.avatars.channels(self.guild_id)) or []
        except (httpx.HTTPError, ValueError, HTTPException):
            self.channels_failed = True
            ui.notify('Discord channels could not be loaded. Showing channel IDs.', type='negative', timeout=8000)
            channels = []
        self.channel_names = {int(c['id']): '#' + c['name'] for c in channels if c['type'] == 0}
        return self.channel_names

    async def load_thread_names(self):
        # UI-03: one extra fetch, only when the guild has thread lore scopes; a failure sets the mapping to None.
        self.thread_scopes = self.store.thread_lore_scopes(self.guild_id)  # snapshot reuses this single read
        if not self.thread_scopes:
            return self.thread_names
        from nicegui import ui
        try:
            threads = await self.service.run(self.ident, self.guild_id, lambda: self.service.avatars.active_threads(self.guild_id)) or []
        except (httpx.HTTPError, ValueError, HTTPException):
            self.thread_names = None  # unknown, not archived: labels fall back to "Thread ...NNNN"
            if not self.channels_failed:  # a channel failure already toasted; one notice per page load
                ui.notify('Thread names could not be loaded. Showing thread IDs.', type='warning', timeout=8000)
            return self.thread_names
        self.thread_names = {int(t['id']): '#' + t['name'] for t in threads}
        return self.thread_names

    @functools.cached_property
    def snapshot(self):
        # Render-time lookups for one page build; callbacks must use live store reads.
        store, gid = self.store, self.guild_id
        spaces, characters, channels = store.list_spaces(gid), store.list_characters(gid), store.list_channels(gid)
        lorebooks, scopes = store.list_lorebooks(gid), self.thread_scopes if self.thread_scopes is not None else store.thread_lore_scopes(gid)
        return types.SimpleNamespace(spaces=spaces, characters=characters, channels=channels, lorebooks=lorebooks,
            worlds={r['id']: r['name'] for r in spaces if r['kind'] == 'world'},
            hubs={r['id']: r['name'] for r in spaces if r['kind'] == 'hub'},
            owners=self.service.owners_from(gid, spaces, characters, channels, lorebooks, scopes, self.channel_names, self.thread_names))

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
        if self.savebar and self.savebar.refuse():
            return
        if owner:
            self.lore_owner = owner
            self.set_url(owner=owner)
        if self.savebar:
            self.savebar.clear()
        for built in self.built:
            self.containers[built].clear()
        self.built.clear()
        if tab and tab != self.selector.value:
            self.selector.set_value(tab)  # returns before the build: the tab_panels change handler builds it
        else:
            await self.build(self.selector.value)

    async def run(self, operation, action=None, detail=None):
        if action is not None and self.savebar and self.savebar.refuse():
            return None
        return (await self._attempt(operation, action, detail))[1]

    async def _attempt(self, operation, action=None, detail=None):
        """Run an operation, returning (succeeded, result); a failure is notified and yields (False, None), a tuple carrying the error as .error."""
        from nicegui import ui
        try:
            return True, await self.service.run(self.ident, self.guild_id, operation, action, detail)
        except (ValueError, HTTPException, sqlite3.IntegrityError, TypeError, KeyError) as error:
            message = error.detail if isinstance(error, HTTPException) else 'This name already exists' if isinstance(error, sqlite3.IntegrityError) else 'Choose valid values for all required fields' if isinstance(error, (TypeError, KeyError)) else str(error)
            ui.notify(message, type='negative', timeout=8000)
            return AttemptFailed((False, None), error)

    def button(self, text, operation, action=None, detail=None, then=None, success=None, prepare=None, **kwargs):
        from nicegui import ui
        async def clicked():
            if self.savebar and self.savebar.refuse():
                return
            if prepare:
                try:
                    await prepare()
                except Exception:
                    logging.exception('Form check before saving failed')
                    ui.notify('Could not check the form before saving. Try again', type='negative', timeout=8000)
                    return
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
            if self.savebar and self.savebar.refuse():
                control.reset()
                return
            ok, result = await self._attempt(lambda: handler(event), action, detail)
            control.reset()  # max_files=1 drops the input after one file; a second upload in this page view needs it back
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


NOTICE_WINDOW = 8.0  # seconds; matches the 8000 ms toast timeout


def should_notify(last, notice, now, window=NOTICE_WINDOW):
    """UI-04: show a rejection notice unless the same text was shown to this client within the toast lifetime.

    `last` is the client's previous (notice, time) or None.
    """
    return last is None or last[0] != notice or now - last[1] >= window


# Discord-like style tokens (UI-28, D20). Later steps reuse these names.
THEME_BODY, THEME_HEADER, THEME_CARD = '#1e1f22', '#111214', '#2b2d31'
THEME_CARD_HOVER, THEME_TILE, THEME_TILE_TEXT = '#35373c', '#404249', '#dbdee1'
THEME_TEXT_MUTED, THEME_TEXT_FAINT, THEME_BORDER = '#b5bac1', '#949ba4', '#4e5058'
THEME_PRIMARY, THEME_NEGATIVE = '#5865f2', '#da373c'
THEME_DIVIDER, THEME_NESTED = '#3f4147', '#313338'
THEME_MACRO_KNOWN, THEME_MACRO_UNKNOWN, THEME_MACRO_TEXT = '#7983f5', '#f23f43', '#ffffff'

_THEME_CSS = f"""
body.body--dark, body.body--dark .q-page, body.body--dark .q-layout {{ background: {THEME_BODY}; }}
.q-header {{ background: {THEME_HEADER}; }}
body.body--dark .q-card {{ background: {THEME_CARD}; }}
body.body--dark .q-card .q-card, body.body--dark .q-card .q-expansion-item {{ background: {THEME_NESTED}; }}
.ll-section {{ width: 100%; padding: 24px; gap: 16px; margin-bottom: 20px; align-items: flex-start; }}
.ll-section-title {{ font-size: 20px; font-weight: 700; line-height: 1.3; }}
.ll-form-row {{ width: 100%; display: flex; flex-flow: row wrap; align-items: flex-end; gap: 16px; }}
.ll-form-row .q-field {{ flex: 0 1 16rem; min-width: min(16rem, 100%); }}
.ll-form-row .ll-wide {{ flex-basis: 22rem; }}
.ll-form-row.ll-field-row {{ align-items: center; }}
.ll-section > .q-expansion-item, .ll-subpanel {{ border: 1px solid {THEME_DIVIDER}; border-radius: 8px; }}
.ll-usage .q-table__grid-content {{ gap: 8px; }}
.ll-usage .q-table__grid-item-card {{ background: {THEME_NESTED}; border: 1px solid {THEME_DIVIDER}; border-radius: 8px; box-shadow: none; }}
.ll-log-row {{ border: 1px solid {THEME_DIVIDER}; border-radius: 8px; }}
.ll-log-row.ll-log-error {{ border-left: 3px solid {THEME_NEGATIVE}; }}
.ll-log-head {{ min-width: 0; flex: 1 1 0; gap: 2px; overflow-wrap: anywhere; }}
.ll-log-block {{ font-family: ui-monospace, monospace; font-size: 13px; white-space: pre-wrap; overflow-wrap: anywhere; max-height: 16rem; overflow: auto; width: 100%; box-sizing: border-box; padding: 8px; background: {THEME_BODY}; border-radius: 6px; margin: 0; }}
.ll-stack {{ width: 100%; box-sizing: border-box; display: flex; flex-direction: column; align-items: stretch; gap: 16px; padding: 4px 0 8px; }}
.ll-stack .q-uploader {{ max-width: min(20rem, 100%); }}
.q-uploader:not(:has(.q-uploader__file)) .q-uploader__subtitle {{ display: none; }}
.q-uploader:not(:has(.q-uploader__file)) .q-uploader__list::before {{ content: "Drop a file here or choose +"; display: block; width: 100%; text-align: center; font-size: 13px; color: {THEME_TEXT_MUTED}; padding: 16px 8px; }}
.ll-result.q-card {{ padding: 16px; box-shadow: none; border: 1px solid {THEME_DIVIDER}; border-radius: 8px; }}
.ll-subtitle {{ font-size: 16px; font-weight: 700; line-height: 1.3; }}
.ll-drop.nicegui-column {{ background: {THEME_BODY}; border: 1px dashed {THEME_BORDER}; border-radius: 8px; gap: 8px; }}
.ll-entry.q-card {{ padding: 4px 12px; gap: 8px; flex-direction: row; flex-wrap: wrap; align-items: center; border: 1px solid {THEME_DIVIDER}; border-radius: 8px; box-shadow: none; }}
.ll-entry-text {{ flex: 1 1 12rem; min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }}
.ll-entry-text > .q-field, .ll-entry-text > div {{ display: inline; }}
.ll-entry-actions {{ flex: 0 0 auto; }}
.ll-pill {{ padding: 1px 8px; border-radius: 999px; border: 1px solid {THEME_DIVIDER}; font-size: 12px; white-space: nowrap; color: {THEME_TEXT_MUTED}; }}
.ll-pill-off {{ color: {THEME_NEGATIVE}; border-color: {THEME_NEGATIVE}; }}
@media (max-width: 600px) {{ .ll-entry-text {{ flex-basis: calc(100% - 4.5rem); }} .ll-entry-actions {{ margin-left: auto; }} }}
.lore-panel {{ min-width: min(20rem, 100%); }}
@media (max-width: 600px) {{ .lore-board {{ flex-direction: column; overflow: visible; }} .lore-board > .lore-panel {{ flex: 0 0 auto; width: 100%; }}
.ll-bulk > .q-btn {{ flex: 1 1 calc(50% - 8px); }} .ll-bulk > .ll-muted {{ flex-basis: 100%; }} }}
.ll-savebar {{ position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%); z-index: 2000; box-sizing: border-box; width: calc(100% - 32px); max-width: 960px; display: flex; flex-flow: row wrap; align-items: center; justify-content: space-between; gap: 8px 16px; padding: 12px 16px; background: {THEME_HEADER}; color: {THEME_TILE_TEXT}; border: 1px solid {THEME_DIVIDER}; border-radius: 12px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4); }}
.ll-savebar.hidden {{ display: none; }}
body:has(.ll-savebar:not(.hidden)) .ll-page {{ padding-bottom: 120px; }}
.ll-savebar.ll-savebar-alert {{ border-color: {THEME_NEGATIVE}; box-shadow: 0 0 0 2px {THEME_NEGATIVE}; }}
.ll-savebar .ll-savebar-reset {{ color: {THEME_TILE_TEXT}; }}
.ll-savebar-actions {{ display: flex; gap: 8px; }}
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
.ll-account {{ flex: none; color: #fff !important; padding: 4px 8px; border-radius: 20px; }}
.ll-account .q-btn__content {{ flex-wrap: nowrap; gap: 8px; }}
.ll-avatar {{ flex: none; width: 32px; height: 32px; border-radius: 50%; object-fit: cover; }}
.ll-avatar-tile {{ flex: none; width: 32px; height: 32px; border-radius: 50%; background: {THEME_TILE}; color: {THEME_TILE_TEXT}; font-size: 13px; font-weight: 600; display: flex; align-items: center; justify-content: center; }}
.ll-account-name {{ color: #fff; font-weight: 600; max-width: 12rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
@media (max-width: 600px) {{ .ll-account-name {{ display: none; }} }}
.ll-menu.q-menu {{ background: {THEME_CARD}; color: #fff; min-width: 200px; }}
.ll-menu-head {{ padding: 12px 16px; display: flex; flex-direction: column; gap: 2px; }}
.ll-menu form {{ margin: 0; }}
.ll-menu-item {{ display: block; width: 100%; padding: 10px 16px; border: 0; background: transparent; color: #fff; font: inherit; text-align: left; cursor: pointer; }}
.ll-menu-item:hover, .ll-menu-item:focus-visible {{ background: {THEME_CARD_HOVER}; }}
.ll-card-header {{ display: flex; align-items: center; width: 100%; min-width: 0; }}
.ll-card-title {{ flex: 1 1 auto; min-width: 0; font-weight: 500; }}
.ll-more {{ margin-right: 4px; }}
.ll-block-head {{ display: flex; align-items: center; gap: 8px; width: 100%; min-width: 0; }}
.ll-block-toggle {{ flex: 1 1 0; min-width: 0; display: flex; align-items: center; gap: 8px; padding: 6px 4px; border: 0; background: transparent; color: inherit; font: inherit; text-align: left; cursor: pointer; border-radius: 8px; }}
.ll-block-toggle:hover {{ background: {THEME_CARD_HOVER}; }}
.ll-block-toggle:focus-visible {{ outline: 2px solid {THEME_PRIMARY}; }}
.ll-block-name {{ flex: 0 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 500; }}
.ll-block-meta .ll-block-missing {{ color: {THEME_NEGATIVE}; }}
.ll-block-meta {{ display: flex; flex: 0 100 auto; min-width: 0; overflow: hidden; white-space: pre; color: {THEME_TEXT_MUTED}; font-size: 13px; }}
.ll-block-caret {{ margin: 0 0 0 auto !important; transition: transform .15s; }}
.ll-block-open .ll-block-caret {{ transform: rotate(180deg); }}
.ll-collapsed {{ display: none !important; }}
.mh-mirror {{ position: absolute; pointer-events: none; user-select: none; overflow: hidden; box-sizing: border-box; color: {THEME_MACRO_TEXT}; border: 0; margin: 0; white-space: pre-wrap; overflow-wrap: break-word; }}
.ll-macro textarea.mh-ta {{ position: relative; z-index: 1; background: transparent; color: transparent; caret-color: {THEME_MACRO_TEXT}; }}
.ll-macro textarea.mh-ta::selection {{ background: rgba(88, 101, 242, 0.45); color: transparent; }}
.mh-known {{ background: color-mix(in srgb, {THEME_MACRO_KNOWN} 35%, transparent); border-radius: 3px; }}
.mh-unknown {{ background: color-mix(in srgb, {THEME_MACRO_UNKNOWN} 25%, transparent); border-radius: 3px; text-decoration: underline wavy {THEME_MACRO_UNKNOWN}; }}
.q-card {{ border-radius: 12px; }}
.q-btn, .q-tab {{ text-transform: none; }}
.q-field--outlined .q-field__control {{ background: {THEME_BODY}; border-radius: 8px; }}
.q-field--outlined .q-field__control:before {{ border-color: {THEME_BORDER}; border-radius: 8px; }}
.q-field--outlined .q-field__control:after {{ border-radius: 8px; }}
.ll-tabs .q-tabs__arrow--left, .ll-tabs .q-tabs__arrow--right {{ color: {THEME_TEXT_MUTED}; }}
@media (max-width: 599px) {{
.prompt-toggle {{ display: grid; grid-template-columns: minmax(0, 1fr) auto auto; align-items: center; column-gap: 8px; }}
.prompt-toggle .ll-block-name {{ grid-column: 1; grid-row: 1; }}
.prompt-toggle .ll-block-meta {{ grid-column: 1; grid-row: 2; }}
.prompt-toggle .ll-pill {{ grid-column: 2; grid-row: 1 / span 2; }}
.prompt-toggle .ll-block-caret {{ grid-column: 3; grid-row: 1 / span 2; }}
.prompt-handle {{ box-sizing: content-box; padding: 11.25px; margin: -11.25px -2px -11.25px -11.25px; touch-action: none; }}
.ll-tabs .q-tabs__arrow, .ll-tabs .q-tabs__arrow--faded {{ display: none; }}
.ll-tabs .q-tabs__content {{ flex-wrap: wrap; overflow: visible; }}
.ll-tabs .q-tab {{ flex: 0 0 auto; padding: 0 12px; min-height: 40px; }}
}}
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
def section(title, header=None):
    """A top-level settings card with a 20 px heading; reuse it for each tab section.

    header: optional callable run at the right end of the title row (e.g. a ``with more_menu(...)`` block)."""
    from nicegui import ui
    with ui.card().classes('ll-section w-full'):
        if header is None:
            ui.label(title).classes('ll-section-title')
        else:
            with ui.element('div').classes('ll-card-header'):
                ui.label(title).classes('ll-section-title grow')
                header()
        yield


def apply_theme():
    from nicegui import ui
    ui.colors(primary=THEME_PRIMARY, negative=THEME_NEGATIVE, dark=THEME_CARD)
    ui.add_css(_THEME_CSS)
    ui.add_css(icon_css())


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
    ui.number.default_props('outlined dense')
    _install_socket_auth(app)
    _install_upload_guard(app)
    _register_pages(app)
    ui.run_with(app, mount_path='/admin', title='llmcord admin', dark=True, reconnect_timeout=15,
                gzip_middleware_factory=None, on_air=None, show_welcome_message=False)


OPERATOR = object()  # llmcord_binding[1] for the bot-wide operator page; never a guild id


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
            if binding[1] is OPERATOR:
                if check_permissions:
                    await app.state.auth.guard_operator(binding[0])
                else:
                    await app.state.auth.session(binding[0])
            elif binding[1] is not None and check_permissions:
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
            now = time.monotonic()
            last = getattr(client, 'llmcord_last_notice', None)  # lives and dies with the client
            if should_notify(last, notice, now):
                client.llmcord_last_notice = (notice, now)
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
            if binding[1] is OPERATOR:
                return PlainTextResponse('Uploads are not available here', 403)
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
        with ui.header().classes('items-center justify-between no-wrap'):
            header_crumb()
            account_menu(session, app.state.auth.is_operator(session))
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

    @ui.page('/operator', response_timeout=30)
    async def operator(request: Request):
        try:
            session = await app.state.auth.guard_operator(request.cookies.get('llmcord_session', ''), origin=request.headers.get('origin'))
        except HTTPException:
            raise HTTPException(404)  # indistinguishable from an unknown path
        apply_theme()
        ui.context.client.llmcord_binding = (request.cookies['llmcord_session'], OPERATOR)
        ui.page_title('Bot settings')
        with ui.header().classes('items-center justify-between no-wrap'):
            header_crumb()
            account_menu(session, True)
        with ui.column().classes('ll-page'):
            ui.label('Bot settings').classes('text-3xl font-bold')
            ui.label('Bot-wide settings. Only operators can see this page.').classes('ll-muted')
            ctx = OperatorContext(app, request)
            selected_tab = request.query_params.get('tab')
            if selected_tab not in ('spending', 'usage', 'backend'):
                selected_tab = 'spending'
            with ui.tabs().classes('w-full ll-tabs').props('align=left outside-arrows mobile-arrows') as tabs:
                spending = ui.tab('spending', 'Spending')
                usage = ui.tab('usage', 'Usage by server')
                backend = ui.tab('backend', 'Backend')
            built = set()
            async def build(tab):
                if tab in built:
                    return
                built.add(tab)
                try:
                    with panels[tab]:
                        if tab == 'spending':
                            budget_panel(ctx)
                        elif tab == 'backend':
                            await backend_panel(ctx, types.SimpleNamespace(profiles=app.state.config_profiles, roles=app.state.config_roles, error=app.state.config_error))
                        else:
                            await usage_by_server_panel(ctx, lambda: app.state.auth.guilds(session))
                except BaseException:
                    built.discard(tab)
                    raise
            async def changed(event):
                ui.run_javascript(f'const u = new URL(location.href); u.searchParams.set("tab", {json.dumps(event.value)}); history.replaceState(history.state, "", u);')
                await build(event.value)
            with ui.tab_panels(tabs, value=selected_tab, on_change=changed).classes('w-full'):
                panels = {'spending': ui.tab_panel(spending), 'usage': ui.tab_panel(usage), 'backend': ui.tab_panel(backend)}
            await build(selected_tab)

    @ui.page('/guild/{guild_id}', response_timeout=30)
    async def guild(request: Request, guild_id: int):
        apply_theme()
        session = await app.state.auth.require_admin(request, guild_id)
        ui.context.client.llmcord_binding = (request.cookies['llmcord_session'], guild_id)
        ctx = LiveContext(app, request, guild_id, session)
        ctx.is_operator = app.state.auth.is_operator(session)
        try:  # cached by require_admin; the header must not fail the page if a refetch does
            current = next((g for g in await app.state.auth.guilds(session) if str(g.get('id')) == str(guild_id)), None)
        except HTTPException:
            current = None
        with ui.header().classes('items-center justify-between no-wrap'):
            header_crumb(current)
            account_menu(session, app.state.auth.is_operator(session))
        with ui.column().classes('ll-page'):
            ui.label('Server administration').classes('text-3xl font-bold')
            ui.label('Changes apply on the next bot turn. Prompt drafts require activation.').classes('ll-muted')
            with ui.tabs().classes('w-full ll-tabs').props('align=left outside-arrows mobile-arrows') as tabs:
                setup = ui.tab('setup', 'Server setup')
                characters = ui.tab('characters', 'Characters')
                lore = ui.tab('lore', 'Lore')
                imports = ui.tab('imports', 'Imports')
                prompts = ui.tab('prompts', 'Prompt presets')
                monitoring = ui.tab('monitoring', 'Monitoring')
            selected_tab = request.query_params.get('tab')
            if selected_tab not in ('setup', 'characters', 'lore', 'imports', 'prompts', 'monitoring'):
                selected_tab = 'setup'
            await ctx.load_channel_names()  # must precede any ctx.snapshot access: it bakes channel names into owner labels
            await ctx.load_thread_names()
            ui.add_css('.lore-drop-zone:empty::before { content: "Drop entries here"; color: #94a3b8; pointer-events: none; }')
            ui.add_css('.q-select:not(.owner-heading) { min-width: min(16rem, 100%); }')
            async def changed(event):
                ctx.set_url(tab=event.value)
                await ctx.build(event.value)
            ctx.builders = {'setup': lambda: setup_panel(ctx), 'characters': lambda: characters_panel(ctx),
                            'lore': lambda: lore_panel(ctx, on_import=lambda: ctx.selector.set_value('imports')),
                            'imports': lambda: imports_panel(ctx), 'prompts': lambda: presets_panel(ctx),
                            'monitoring': lambda: monitoring_panel(ctx)}
            with ui.tab_panels(tabs, value=selected_tab, on_change=changed).classes('w-full') as ctx.selector:
                for tab in (setup, characters, lore, imports, prompts, monitoring):
                    ctx.containers[tab.props['name']] = ui.tab_panel(tab)
            ctx.savebar = SaveBar(ctx)
            await ctx.build(selected_tab)


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


def header_crumb(current=None):
    """Header crumb: the llmcord link, plus the current server's icon (or initials tile) and name when given."""
    from nicegui import ui
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


def account_menu(session, is_operator=False):
    """Header account button (avatar plus name) opening a menu with the name and a Sign out item."""
    from nicegui import ui
    import html
    user = session['user']
    name = str(user.get('global_name') or user['username'])
    with ui.button().props('flat no-caps aria-label="Account menu"').classes('ll-account'):
        # user_avatar_url only returns a CDN URL built from a digit snowflake and a hex hash, so it is safe inside the quoted prop.
        url = user_avatar_url(user, 64)
        if url:
            ui.element('img').classes('ll-avatar').props(f'src="{url}" alt=""')
        else:
            ui.label(server_initials(name)).classes('ll-avatar-tile').props('aria-hidden=true')
        ui.label(name).classes('ll-account-name')
        # The button is centred in the header with about 16 px below it; the 20 px offset puts the menu 4 px under the header's bottom edge.
        with ui.menu().props('anchor="bottom right" self="top right" :offset="[0, 20]"').classes('ll-menu'):
            with ui.element('div').classes('ll-menu-head'):
                ui.label(name).classes('font-semibold')
                if str(user['username']) != name:
                    ui.label('@' + str(user['username'])).classes('ll-muted text-sm')
            if is_operator:
                ui.link('Bot settings', '/operator').classes('ll-menu-item no-underline')  # ui.link adds the /admin mount prefix
            ui.separator()
            # A normal POST retains the existing origin and CSRF checks.
            ui.html('<form action="/logout" method="post"><input type="hidden" name="csrf" value="' + html.escape(session['csrf'], quote=True) + '"><button type="submit" class="ll-menu-item">Sign out</button></form>', sanitize=False)


def format_usd(value):
    """Dashboard cost: $0.00, two decimals from a cent up, else up to four decimals trimmed (<$0.0001 below that)."""
    value = float(value or 0)
    if not math.isfinite(value) or value <= 0:
        return '$0.00'
    if value >= 0.01:
        return f'${value:,.2f}'
    text = f'{value:.4f}'.rstrip('0')
    if text == '0.':
        return '<$0.0001'
    return '$' + text


RETENTION_OPTIONS = {7: '7 days', 14: '14 days', 30: '30 days', 0: 'Full month'}
USAGE_FEATURES = {'reply': 'Replies', 'ambient': 'Ambient turns', 'summon': 'Summons', 'memory': 'Memory updates', 'catchup': 'Catch-ups'}
USAGE_RANGES = {'day': ('Last 24 hours', 1), '7': ('Last 7 days', 7), '14': ('Last 14 days', 14), '30': ('Last 30 days', 30), 'month': ('This month', 31)}
USAGE_UNKNOWN = 'Unknown (before this update)'


def usage_since(zone_name, key, now):
    """Start of a usage range: a rolling window, or the 1st of the month in the server timezone."""
    if key == 'month':
        try:
            zone = ZoneInfo(zone_name or 'UTC')
        except Exception:
            zone = ZoneInfo('UTC')
        return datetime.fromtimestamp(now, zone).replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
    return now - USAGE_RANGES[key][1] * 86400


def usage_chart_options(report, since, now):
    zone = ZoneInfo(report['zone'])
    day, last, days = datetime.fromtimestamp(since, zone).date(), datetime.fromtimestamp(now, zone).date(), []
    while day <= last:
        days.append(day.isoformat()); day += timedelta(days=1)
    palette = [THEME_PRIMARY, '#3ba55d', '#faa61a', THEME_NEGATIVE, '#9b84ee', THEME_TEXT_MUTED]
    series = []
    for index, (model, cells) in enumerate(sorted(report['daily'].items())):
        color = palette[index % len(palette)]
        for pos, label in ((0, 'Input'), (1, 'Output')):
            series.append({'name': f'{model} · {label}', 'type': 'bar', 'stack': label, 'data': [cells.get(d, [0, 0])[pos] for d in days],
                           'itemStyle': {'color': color, 'opacity': 1 if pos == 0 else 0.6}})
    axis = {'axisLabel': {'color': THEME_TEXT_MUTED}, 'axisLine': {'lineStyle': {'color': THEME_BORDER}}}
    return {'backgroundColor': 'transparent', 'textStyle': {'color': THEME_TEXT_MUTED}, 'color': palette,
            'legend': {'type': 'scroll', 'bottom': 0, 'textStyle': {'color': THEME_TEXT_MUTED}},
            'tooltip': {'trigger': 'axis', 'axisPointer': {'type': 'shadow'}, 'backgroundColor': THEME_CARD_HOVER, 'borderColor': THEME_BORDER, 'textStyle': {'color': THEME_TILE_TEXT},
                        ':formatter': "params => params[0].axisValueLabel + '<br>' + params.filter(p => p.value > 0).map(p => p.marker + p.seriesName + ': ' + Number(p.value).toLocaleString('en-US') + ' tokens').join('<br>')"},
            'grid': {'left': 8, 'right': 8, 'top': 16, 'bottom': 48, 'containLabel': True},
            'xAxis': {'type': 'category', 'data': [d[5:] for d in days], **axis},
            'yAxis': {'type': 'value', 'splitLine': {'lineStyle': {'color': THEME_DIVIDER}}, 'axisLabel': {'color': THEME_TEXT_MUTED, ':formatter': "v => v >= 1000000 ? v / 1000000 + 'M' : v >= 1000 ? v / 1000 + 'k' : v"}},
            'series': series}


def usage_range_keys(days):
    """Period options for a retention of `days` (0 = current month only): rolling ranges that fit, and 'month' only at 30."""
    if days == 0:
        return ['day', '7', 'month']
    return [k for k in USAGE_RANGES if k == 'month' and days == 30 or k != 'month' and USAGE_RANGES[k][1] <= days]


def usage_feature_rows(rows):
    """By-feature rows with display labels: empty -> before-update bucket; unrecognised names merge into one 'Unknown feature' row."""
    merged = {}
    for r in rows:
        feature = r['feature']
        label = USAGE_FEATURES.get(feature) or (USAGE_UNKNOWN if not feature else 'Unknown feature')
        row = merged.setdefault(label, {'feature': label, 'requests': 0, 'input_tokens': 0, 'output_tokens': 0, 'cost_usd': 0.0, 'unpriced': 0, 'unreported': 0})
        for field in ('requests', 'input_tokens', 'output_tokens', 'unpriced', 'unreported'):
            row[field] += r[field] or 0
        if r['cost_usd'] is None:
            row['unpriced'] += r['requests']
        else:
            row['cost_usd'] += r['cost_usd']
    return list(merged.values())


def usage_table(columns, rows, key):
    from nicegui import ui
    ui.table(columns=[{'name': name, 'field': name, 'label': label, 'align': 'left'} for name, label in columns], rows=rows, row_key=key
             ).classes('w-full ll-usage').props(':grid="Quasar.Screen.lt.sm" hide-pagination')


TURN_LOG_PAGE = 50


def normalize_reference(text):
    """A pasted reference ID: the 6-hex token if there is one ('(ref 06F1EE)' -> '06f1ee'), else the trimmed text, lowercased."""
    text = (text or '').strip()
    found = re.search(r'\b[0-9a-fA-F]{6}\b', text)
    return (found.group(0) if found else text.strip(' \t#[](){}.,:;')).lower()


def turn_log_row(entry, zone_name, channel_label):
    """Display strings for one turn log summary row (time in the server timezone, falling back to UTC)."""
    try:
        zone = ZoneInfo(zone_name or 'UTC')
    except Exception:
        zone = ZoneInfo('UTC')
    inputs, outputs = entry['input_tokens'], entry['output_tokens']
    tokens = '—' if inputs is None and outputs is None else f"{'—' if inputs is None else format(inputs, ',')} in / {'—' if outputs is None else format(outputs, ',')} out"
    return {'time': datetime.fromtimestamp(entry['created_at'], zone).strftime('%Y-%m-%d %H:%M'), 'channel': channel_label(entry['channel_id']),
            'stage': entry['stage'] or '—', 'model': entry['model'] or '—', 'tokens': tokens, 'error': entry['status'] == 'error',
            'status': 'Error' if entry['status'] == 'error' else 'OK', 'reference': entry['reference_id'] or '', 'preview': (entry['preview'] or '').strip()}


async def turn_log_section(ctx, channel_label):
    from nicegui import ui
    store, gid = ctx.store, ctx.guild_id
    days = store.turn_log_settings(gid)['days']
    zone_name = store.guild_timezone(gid)
    with section('Log'):
        ui.label('Personal facts are hidden and API keys removed. Entries are kept for ' + ('the current month' if days == 0 else f'{days} days') + ' (Server settings).').classes('ll-muted')
        with ui.element('div').classes('ll-form-row ll-field-row'):
            channel = ui.select({None: 'All channels'}, value=None, label='Channel')
            errors_only = ui.switch('Errors only')
            reference = ui.input('Reference ID')
        off_note = ui.label('Logging is off; showing earlier entries.').classes('ll-muted')
        off_note.set_visibility(False)
        body = ui.column().classes('w-full gap-2')
        more = ui.button('Load more', on_click=lambda: load(False)).props('outline')
        more.set_visibility(False)
        state = {'last': None, 'applied': None, 'gen': 0, 'busy': False, 'shown': None, 'shown_last': None}

        def filters():
            return {'channel_id': channel.value, 'errors_only': bool(errors_only.value), 'reference_id': normalize_reference(reference.value) or None}

        async def open_entry(entry_id, holder):
            result = await ctx.run(lambda: (store.turn_log_entry(gid, entry_id),))
            holder.clear()
            if result is None:
                return False  # the guarded read failed (already notified): let reopening retry
            full = result[0]
            with holder:
                if full is None:
                    ui.label('This entry is no longer available.').classes('ll-muted')
                    return True
                for title, text in (('Request', full['request_text']), ('Response', full['response_text']), ('Error', full['error_detail'])):
                    if text or title != 'Error':
                        ui.label(title).classes('ll-subtitle')
                        ui.label(text or '(empty)').classes('ll-log-block')
            return True

        def add_row(entry):
            row = turn_log_row(entry, zone_name, channel_label)
            with ui.expansion().classes('ll-log-row w-full' + (' ll-log-error' if row['error'] else '')).props('dense') as expansion:
                with expansion.add_slot('header'):
                    with ui.column().classes('ll-log-head'):
                        ui.label(f"{row['time']} · {row['channel']} · {row['stage']}").classes('text-sm')
                        ui.label(f"{row['model']} · {row['tokens']}").classes('ll-muted text-sm')
                        if row['error']:
                            ui.label('Error' + (f" · Ref {row['reference']}" if row['reference'] else '')).classes('text-negative text-sm')
                        elif row['reference']:
                            ui.label(f"OK · Ref {row['reference']}").classes('ll-muted text-sm')
                        else:
                            ui.label('OK').classes('ll-muted text-sm')
                        if row['preview']:
                            ui.label(row['preview']).classes('ll-muted text-sm')
                holder = ui.column().classes('w-full gap-2 q-pa-sm')
            built = []
            async def toggled(event):
                if event.value and not built:
                    built.append(True)
                    if not await open_entry(entry['id'], holder):
                        built.clear()
            expansion.on_value_change(toggled)

        async def load(reset):
            if reset:
                state['gen'] += 1
                state['last'] = None
                state['applied'] = filters()
            elif state['busy']:
                return
            generation, chosen, before = state['gen'], state['applied'], state['last']
            state['busy'] = True
            more.disable()
            try:
                page = await ctx.run(lambda: (store.turn_log_page(gid, **chosen, before_id=before, limit=TURN_LOG_PAGE), store.turn_log_settings(gid)['enabled'],
                                              store.turn_log_channels(gid) if reset else None))
                if generation != state['gen']:
                    return  # a newer filter was applied while this read ran
                if page is None:
                    if reset:  # keep the controls and the list in agreement: put the controls back without another read
                        # restore what the list shows; the handlers that follow see filters() == applied and read nothing
                        restored = state['shown'] or {'channel_id': None, 'errors_only': False, 'reference_id': None}
                        state['applied'], state['last'] = restored, state['shown_last']
                        channel.value, errors_only.value, reference.value = restored['channel_id'], restored['errors_only'], restored['reference_id'] or ''
                    return
                entries, enabled, known = page
                if reset:
                    keep = [channel.value] if channel.value is not None and channel.value not in known else []
                    channel.set_options({None: 'All channels', **{c: channel_label(c) for c in [*known, *keep]}}, value=channel.value)
                    state['shown'] = chosen
                    body.clear()
                    off_note.set_visibility(not enabled and bool(entries))
                with body:
                    for entry in entries:
                        add_row(entry)
                    if reset and not entries:
                        filtered = chosen['channel_id'] is not None or chosen['errors_only'] or chosen['reference_id']
                        ui.label('No entries match these filters.' if filtered else 'No log entries yet.' if enabled
                                 else 'The turn log is off. Turn it on in Server settings to record model calls.').classes('ll-muted')
                if entries:
                    state['last'] = entries[-1]['id']
                if reset:
                    state['shown_last'] = state['last']
                more.set_visibility(len(entries) >= TURN_LOG_PAGE)
            finally:
                if generation == state['gen']:
                    state['busy'] = False
                    more.enable()

        async def apply(_=None):
            if filters() != state['applied']:
                await load(True)
        channel.on_value_change(apply)
        errors_only.on_value_change(apply)
        reference.on('keydown.enter', apply)
        reference.on('blur', apply)
        await load(True)


def usage_role_labels(backend_roles):
    """Profile name -> the roles it serves; a role without its own profile uses dialogue's."""
    roles = {}
    for role in ('dialogue', 'director', 'memory'):
        ident = backend_roles.get(role, backend_roles.get('dialogue'))
        if ident:
            roles.setdefault(ident, []).append(role)
    return roles


async def monitoring_panel(ctx):
    from nicegui import ui
    store, gid = ctx.store, ctx.guild_id
    days = store.turn_log_settings(gid)['days']
    keys = usage_range_keys(days)
    roles = usage_role_labels(ctx.service.effective_backend().roles)
    thread_names = ctx.thread_names or {}
    def channel_label(cid):
        if cid is None:
            return USAGE_UNKNOWN
        name = ctx.channel_names.get(cid) or thread_names.get(cid)
        if name:
            return name
        digits = str(cid)
        return f"Deleted channel (ID …{digits[-4:]})"
    def money(row):
        return format_usd(row['cost_usd'])
    if ctx.is_operator:  # re-checked per build; the number is bot-wide, so it is read through the operator guard
        from . import budget
        try:
            state = await ctx.service.run_operator(ctx.ident, lambda: budget.state(store))
        except HTTPException:
            state = None
        if state:
            with ui.element('div').classes('ll-form-row'):
                ui.label(f'Bot-wide spending: {format_usd(state.spent_usd)} of {format_usd(state.hard_cap_usd)} hard cap this period' if state.hard_cap_usd is not None
                         else f'Bot-wide spending: {format_usd(state.spent_usd)} this period; no hard cap set').classes('ll-muted')
                ui.link('Open Bot settings', '/operator')  # ui.link adds the /admin mount prefix
    with section('Usage'):
        select = ui.select({k: USAGE_RANGES[k][0] for k in keys}, value='7', label='Period')
        if len(keys) < len(USAGE_RANGES):
            ui.label('Usage is kept for ' + ('the current month' if days == 0 else f'{days} days') + ' (Server settings).').classes('ll-muted')
        body = ui.column().classes('w-full gap-4')
        def render():
            body.clear()
            now = time.time()
            since = usage_since(store.guild_timezone(gid), select.value, now)
            report = store.usage_report(gid, since, now)
            with body:
                if not report['totals']['requests']:
                    ui.label('No model calls in this period.')
                    return
                ui.echart(usage_chart_options(report, report['since'], now)).classes('w-full h-80')
                ui.label('By model').classes('ll-subtitle')
                usage_table([('model', 'Model'), ('roles', 'Used for'), ('requests', 'Requests'), ('input', 'Input tokens'), ('output', 'Output tokens'), ('cost', 'Estimated USD'), ('unpriced', 'Without cost estimate')],
                            [{'key': f"{r['profile']}/{r['model']}", 'model': r['model'], 'roles': ', '.join(roles.get(r['profile'], [])), 'requests': f"{r['requests']:,}", 'input': f"{r['input_tokens']:,}",
                              'output': f"{r['output_tokens']:,}", 'cost': money(r), 'unpriced': r['unpriced']} for r in report['by_model']], 'key')
                ui.label('By channel').classes('ll-subtitle')
                usage_table([('channel', 'Channel'), ('requests', 'Requests'), ('input', 'Input tokens'), ('output', 'Output tokens'), ('cost', 'Estimated USD'), ('unpriced', 'Without cost estimate')],
                            [{'key': str(r['channel_id']), 'channel': channel_label(r['channel_id']), 'requests': f"{r['requests']:,}", 'input': f"{r['input_tokens']:,}",
                              'output': f"{r['output_tokens']:,}", 'cost': money(r), 'unpriced': r['unpriced']} for r in report['by_channel']], 'key')
                ui.label('By feature').classes('ll-subtitle')
                usage_table([('feature', 'Feature'), ('requests', 'Requests'), ('input', 'Input tokens'), ('output', 'Output tokens'), ('cost', 'Estimated USD'), ('unpriced', 'Without cost estimate')],
                            [{'key': r['feature'], 'feature': r['feature'], 'requests': f"{r['requests']:,}", 'input': f"{r['input_tokens']:,}",
                              'output': f"{r['output_tokens']:,}", 'cost': money(r), 'unpriced': r['unpriced']} for r in usage_feature_rows(report['by_feature'])], 'key')
                if report['totals']['unreported']:
                    ui.label(f"{report['totals']['unreported']:,} calls did not report token counts, so their tokens are not included.").classes('ll-muted')
                ui.label('Tracked for this server, including internal model calls. Cost uses list rates or your configured rates; it is not a billing statement.').classes('ll-muted')
        select.on_value_change(lambda _: render())
        render()
    await turn_log_section(ctx, channel_label)


async def setup_panel(ctx):
    from nicegui import ui
    store, gid = ctx.store, ctx.guild_id
    spaces = {r['id']: r['name'] + ' (' + r['kind'] + ')' for r in ctx.snapshot.spaces}
    channel_names = ctx.channel_names
    with section('Worlds and hubs'):
        for space in ctx.snapshot.spaces:
            with ui.expansion(f"{space['name']} · {space['kind']}").classes('space-card w-full rounded-lg'):
                guideline_editor(ctx, 'space', space['id'], 'World guidelines' if space['kind'] == 'world' else 'Hub guidelines')
                lucide_button('Delete ' + space['kind'], 'trash-2', color='negative', on_click=lambda space=space: delete_space_dialog(ctx, space))
        with ui.element('div').classes('ll-form-row ll-field-row'):
            name = ui.input('Name')
            kind = ui.select(['world', 'hub'], value='world', label='Kind')
            ctx.button('Create', lambda: store.create_space(gid, name.value or '', kind.value), 'space.create', then=lambda _: ctx.refresh())
    with section('Hub links'):
        hubs, worlds = ctx.snapshot.hubs, ctx.snapshot.worlds
        with ui.element('div').classes('ll-form-row ll-field-row'):
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
        with ui.element('div').classes('ll-form-row ll-field-row'):
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
        ui.label('Rebinding a channel keeps its ambient mode and removes cast members not available in the new world or hub.').classes('ll-muted')
        for binding in ctx.snapshot.channels:
            channel_card(ctx, binding, channel_names, spaces)
    with section('Server settings'):
        footer = ui.switch('Show model and cost footer on replies', value=store.usage_footer_enabled(gid))
        catchup = ui.switch('Allow /catchup in channels without characters', value=store.catchup_anywhere(gid))
        ui.label('Members can then get a private summary of any channel they can read. Its messages are sent to the summary model.').classes('ll-muted')
        turn_log = store.turn_log_settings(gid)
        log_switch = ui.switch('Keep a turn log', value=turn_log['enabled'])
        with ui.element('div').classes('ll-form-row ll-field-row'):
            log_days = ui.select(RETENTION_OPTIONS, value=turn_log['days'], label='Keep usage and log for')
        ui.label('The turn log stores the prompt and reply of every model call on this server, including /catchup, so admins can check errors. Personal facts are hidden. Usage history is kept for the same period.').classes('ll-muted')
        current = store.guild_timezone(gid)
        with ui.element('div').classes('ll-form-row ll-field-row'):
            timezone = ui.select(timezone_options(current), value=current or None,
                                 label='Server timezone', with_input=True, clearable=True)
        ui.label('Characters use this for members who have not chosen their own with /time set. Without it they use UTC.').classes('ll-muted')
        current_asset = store.asset_channel_id(gid)
        saved_asset = current_asset if current_asset in channel_names else None
        with ui.element('div').classes('ll-form-row ll-field-row'):
            asset_channel = ui.select(channel_names, value=saved_asset, label='Private text channel')
        ui.label('Avatar asset channel: deny View Channel to @everyone and allow the bot to upload images. Images become Discord CDN assets.').classes('ll-muted')
        def save_footer():
            store.set_usage_footer(gid, footer.value)
            return True
        def save_catchup():
            store.set_catchup_anywhere(gid, catchup.value)
            return True
        def save_turn_log():
            store.set_turn_log(gid, bool(log_switch.value), int(log_days.value))
            if 'monitoring' in ctx.built:  # rebuild on next show so its retention notes and periods follow
                ctx.containers['monitoring'].clear()
                ctx.built.discard('monitoring')
            return True
        async def configure():
            await ctx.service.avatars.configure(gid, asset_channel.value)
            return True
        async def show_monitoring(_):
            # same in-flight caveat as refresh(): a Monitoring build still awaiting its reads when this save lands could double-fill
            if ctx.selector.value == 'monitoring':
                await ctx.build('monitoring')
        ctx.savebar.track('Server settings', {}, then=show_monitoring, parts=[
            {'controls': {footer: store.usage_footer_enabled(gid)}, 'operation': save_footer, 'action': 'settings.footer',
             'detail': lambda _: {'enabled': bool(footer.value)}},
            {'controls': {catchup: store.catchup_anywhere(gid)}, 'operation': save_catchup, 'action': 'settings.catchup',
             'detail': lambda _: {'enabled': bool(catchup.value)}},
            {'controls': {log_switch: turn_log['enabled'], log_days: turn_log['days']}, 'operation': save_turn_log, 'action': 'settings.turn_log',
             'detail': lambda _: {'enabled': bool(log_switch.value), 'days': int(log_days.value)}},
            {'controls': {timezone: current or None}, 'operation': lambda: timezone_operation(store, gid, timezone.value),
             'action': 'settings.timezone', 'detail': timezone_detail},
            {'controls': {asset_channel: saved_asset}, 'operation': configure, 'action': 'avatar.channel'}],
            success='Server settings saved')


def channel_savers(store, gid, cid, cast_value, ambient_value, cast, ambient):
    """Save operations for a channel card's default cast and ambient mode, with a stale-overwrite guard (UI-42).

    There is no revision column: each operation re-reads the guild's binding and compares it with the value the card was
    built from (or last saved), raising ConflictError if it changed elsewhere. The compare and the write run inside
    store.write_admin() (BEGIN IMMEDIATE), so the bot process cannot write between them. store.execute's own commit ends
    that transaction after the first UPDATE, which is the one that changes the compared column.
    """
    baseline = {'cast': list(cast_value), 'ambient': bool(ambient_value)}
    def current():
        row = store.channel(cid)
        if row is None or row['guild_id'] != gid:
            raise ConflictError('This channel is no longer bound in Discord. Reload the page and try again.')
        return row
    def save_cast():
        value = list(cast())
        with store.write_admin():
            if json.loads(current()['default_cast']) != baseline['cast']:
                raise ConflictError('Default cast changed in Discord. Reload the page and try again.')
            store.set_cast(cid, None, value, default=True)
        baseline['cast'] = list(dict.fromkeys(value))
        return True
    def save_ambient():
        value = bool(ambient())
        with store.write_admin():
            if bool(current()['ambient']) != baseline['ambient']:
                raise ConflictError('Ambient participation changed in Discord. Reload the page and try again.')
            store.set_ambient(cid, value)
        baseline['ambient'] = value
        return True
    return save_cast, save_ambient


def binding_space_name(spaces, space_id):
    """The bound world or hub's name for a channel summary row, or '' when it no longer exists (UI-42)."""
    return spaces[space_id].rsplit(' (', 1)[0] if space_id in spaces else ''


def channel_card(ctx, binding, channel_names, spaces):
    """One channel binding: a collapsed summary row that expands to a single save-bar editor (guidelines, cast, ambient)."""
    from nicegui import ui
    store, gid = ctx.store, ctx.guild_id
    cid = binding['channel_id']
    name = channel_names.get(cid, str(cid))
    space_name = binding_space_name(spaces, binding['space_id'])
    with ui.card().classes('w-full channel-card ll-stack ll-result') as card:
        with ui.element('div').classes('ll-block-head'):
            toggle = ui.element('button').classes('ll-block-toggle channel-toggle')
            toggle.props['type'] = 'button'
            toggle.props['aria-expanded'] = 'false'
        body = ui.element('div').classes('ll-stack w-full ll-collapsed')
        with body:
            control, current = guideline_editor(ctx, 'channel', cid, 'Channel guidelines', button=False)
            options = {r['id']: r['name'] for r in store.eligible_characters(gid, binding['space_id'])}
            saved_cast = json.loads(binding['default_cast'])
            cast = ui.select(options, value=saved_cast, multiple=True, label='Default cast (up to five)').classes('w-full')
            ambient = ui.switch('Ambient participation', value=bool(binding['ambient']))
        with toggle:
            _span(name + ' ').classes('ll-block-name')
            with ui.element('span').classes('ll-block-meta'):
                _span(' \u00b7 ')
                _span(space_name or 'World or hub missing').classes('' if space_name else 'll-block-missing')
                _span(' \u00b7 ')
                _span().bind_text_from(cast, 'value', backward=lambda v: f"{len(v or [])} in cast ")
            _span('Ambient').classes('ll-pill').bind_visibility_from(ambient, 'value')
            lucide('chevron-down', '1.25em').classes('ll-icon-solo ll-block-caret')
        def flip():
            now_open = toggle.props['aria-expanded'] != 'true'
            card.classes(add='ll-block-open') if now_open else card.classes(remove='ll-block-open')
            body.classes(remove='ll-collapsed') if now_open else body.classes(add='ll-collapsed')
            toggle.props['aria-expanded'] = 'true' if now_open else 'false'
            toggle.update()
        toggle.on('click', flip)
    revision = {'n': current['revision']}
    def save_guidelines():
        result = store.save_guidelines(gid, 'channel', cid, control.value or '', revision['n'])
        revision['n'] = store.guidelines(gid, 'channel', cid)['revision']
        return result
    save_cast, save_ambient = channel_savers(store, gid, cid, saved_cast, binding['ambient'],
                                             lambda: cast.value or [], lambda: ambient.value)
    ctx.savebar.track(name, {}, parts=[
        {'controls': {control: current['content']}, 'operation': save_guidelines, 'action': 'guidelines.edit',
         'detail': {'kind': 'channel', 'id': cid}},
        {'controls': {cast: saved_cast}, 'operation': save_cast, 'action': 'cast.default'},
        {'controls': {ambient: bool(binding['ambient'])}, 'operation': save_ambient, 'action': 'channel.ambient'}],
        success='Channel settings saved')


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
        dialog.on('hide', dialog.delete)
        dialog.open()
    rows = ctx.snapshot.characters
    MacroHighlight()
    with section('Characters'):
        lucide_button('Create character', 'plus', on_click=new_character).set_enabled(bool(worlds))
        if not worlds:
            ui.label('Create a home world in Server setup first.').classes('ll-muted')
        for row in rows:
            character_card(ctx, row, worlds)
        if not rows:
            ui.label('Create an empty character here or import a character card in Imports to get started.').classes('ll-muted')


def character_card(ctx, row, worlds):
    from nicegui import ui
    store, gid = ctx.store, ctx.guild_id
    revision = store.owner_revision(gid, 'character', row['id'])
    with ui.expansion().classes('w-full character-card') as card_panel:
        with card_panel.add_slot('header'), ui.element('div').classes('ll-card-header'):
            ui.label(row['name'] + (' · Archived' if row['archived'] else '')).classes('ll-card-title')
            with more_menu(row['name']):
                async def archive_item(row=row):
                    def archive():
                        store.archive_character(gid, row['id'], not row['archived'])
                        return True
                    if await ctx.run(archive, 'character.archive', {'id': row['id']}):
                        await ctx.refresh('characters')
                ui.menu_item('Restore' if row['archived'] else 'Archive', on_click=archive_item).props('role=menuitem')
                ui.separator()
                def confirm_delete(row=row, revision=revision):
                    confirm_dialog(ctx, f"Delete {row['name']}?",
                                   ['Permanently delete this character, its lore, memories, and saved avatars, and remove it from all casts. Past Discord messages remain.',
                                    'This cannot be undone. Use Archive if you may want to restore the character later.'],
                                   'Delete permanently', lambda: store.delete_character(gid, row['id'], revision),
                                   'character.delete', {'id': row['id']}, then=lambda _: ctx.refresh('characters'))
                ui.menu_item('Delete character', on_click=confirm_delete).classes('text-negative').props('role=menuitem')
        with ui.element('div').classes('ll-stack'):
            card = json.loads(row['card'])
            with ui.element('div').classes('ll-form-row'):
                name = ui.input('Name', value=row['name'])
                world = ui.select(worlds, value=row['world_id'], label='Home world')
            fields = {field: ui.textarea(label, value=card.get(field, '')).classes('w-full ll-macro') for field, label in
                [('description', 'Description'), ('personality', 'Personality'), ('scenario', 'Scenario'), ('first_mes', 'Opening line'), ('mes_example', 'Example dialogue'), ('system_prompt', 'Card instructions'), ('post_history_instructions', 'Card post-history instructions')]}
            confirm = ui.checkbox('Confirm moving worlds; ineligible casts will be cleared')
            confirm.bind_visibility_from(world, 'value', backward=lambda value, saved=row['world_id']: value != saved)
            world.on_value_change(lambda _: confirm.set_value(False))  # a world move needs a fresh tick
            def save(row=row, card=card, name=name, world=world, fields=fields, confirm=confirm, revision=revision):
                if world.value != row['world_id'] and not confirm.value:
                    raise ValueError('Confirm the world move before saving')
                new = {**card, **{key: control.value or '' for key, control in fields.items()}, 'name': name.value}
                store.update_character(gid, row['id'], world.value, name.value or '', new, expected_revision=revision)
                return True
            tracked = {name: row['name'], world: row['world_id'], **{control: card.get(key, '') for key, control in fields.items()}}
            ctx.savebar.track(row['name'], tracked, save, 'character.edit', {'id': row['id']}, then=lambda _: ctx.refresh('characters'), on_reset=lambda: confirm.set_value(False), reload=lambda: ctx.refresh('characters'))
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
                lucide_button('Remove fallback avatar', 'trash-2', color='negative', on_click=lambda: confirm_dialog(
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
                lucide_button('Remove emotion image', 'trash-2', color='negative', on_click=lambda: confirm_dialog(
                    ctx, f"Remove the image from {slot['label']}?", ['The emotion and its label stay; only the image is removed.'],
                    'Remove emotion image', remove_image, 'avatar.image.delete', {'character': character_id, 'slot': slot['slot_key']},
                    then=lambda _: ctx.refresh('characters')))
            if slot['slot_key'] != 'neutral':
                def delete():
                    ctx.store.delete_avatar(ctx.guild_id, character_id, slot['slot_key'], slot['revision'])
                    return True
                lucide_button('Remove emotion', 'trash-2', color='negative', on_click=lambda: confirm_dialog(
                    ctx, f"Remove emotion {slot['label']}?", ['This removes the emotion and its image. The neutral emotion and the fallback avatar stay.'],
                    'Remove emotion', delete, 'avatar.slot.delete', then=lambda _: ctx.refresh()))


def lore_panel(ctx, on_import=None):
    from .lore_workspace import render_lore_workspace
    MacroHighlight()
    render_lore_workspace(ctx, entry_editor, on_import=on_import,
                          on_entry_import=lambda kind, ident, refresh: direct_import_dialog(ctx, kind, ident, refresh))


RULE_LABELS = {
    'selective': 'Selective', 'selective_logic': 'Secondary key logic', 'case_sensitive': 'Case sensitive',
    'whole_words': 'Whole words only', 'depth': 'Depth', 'scan_depth': 'Scan depth (JSON, null for default)',
    'probability': 'Probability (%)', 'use_probability': 'Use probability', 'group': 'Groups (JSON list)',
    'group_weight': 'Group weight', 'group_override': 'Group override', 'use_group_scoring': 'Use group scoring',
    'exclude_recursion': 'Exclude from recursion', 'prevent_recursion': 'Prevent recursion',
    'delay_until_recursion': 'Delay until recursion', 'recursion_level': 'Recursion level',
    'sticky': 'Sticky turns', 'cooldown': 'Cooldown turns', 'delay': 'Delay turns',
    'character_filter_names': 'Character filter names (JSON list)', 'character_filter_tags': 'Character filter tags (JSON list)',
    'character_filter_exclude': 'Exclude filtered characters', 'match_character_description': 'Match character description',
    'match_character_personality': 'Match character personality', 'match_scenario': 'Match scenario',
}


def rule_label(key):
    return RULE_LABELS.get(key) or key.replace('_', ' ').capitalize()


def entry_editor(ctx, entry, owners, refresh):
    from nicegui import ui
    with ui.card().classes('w-full ll-stack').style('padding: 16px') as panel:
        ui.label('Edit lore' if entry.get('ref') else 'Create lore').classes('ll-subtitle')
        text = ui.textarea('Content', value=entry['content']).classes('w-full ll-macro')
        rule = copy.deepcopy(entry['rule'])
        controls = {}
        for key, label in (('keys', 'Keys'), ('secondary_keys', 'Secondary keys')):
            controls[key] = ui.input_chips(label, value=[str(item) for item in rule.get(key) or []], new_value_mode='add-unique') \
                .props('outlined dense hint="Press Enter after each key"').classes('w-full')
            # Typed text that was never confirmed with Enter becomes a chip when the field loses focus.
            # Quasar clears its input before it emits blur, so the pending text is tracked from input-value.
            element_id = controls[key].id
            controls[key].on('input-value', js_handler=f"""(value) => {{
                window.llPendingKeys = window.llPendingKeys || {{}};
                window.llPendingKeys[{element_id}] = value || '';
            }}""")
            controls[key].on('blur', js_handler=f"""() => {{
                const text = ((window.llPendingKeys || {{}})[{element_id}] || '').trim();
                if (window.llPendingKeys) window.llPendingKeys[{element_id}] = '';
                if (text) getElement({element_id}).add(text, true);
            }}""")
        controls['order'] = ui.number('Order', value=rule['order'], precision=0).props('hint="Higher values appear later"')
        with ui.row().classes('gap-x-3 gap-y-0'):
            controls['enabled'] = ui.checkbox('Enabled', value=rule['enabled']).classes('whitespace-nowrap')
            controls['constant'] = ui.checkbox('Always active', value=rule['constant']).classes('whitespace-nowrap')
            pinned = ui.checkbox('Pinned', value=entry['pinned']).classes('whitespace-nowrap')
        with ui.expansion('Advanced activation and placement rules').classes('w-full ll-subpanel'), ui.element('div').classes('ll-stack px-4'):
            for key, value in rule.items():
                if key in controls or key == 'unsupported':
                    continue
                if key == 'original':
                    with ui.expansion('Preserved import fields / unsupported feature remapping').classes('w-full ll-subpanel'):
                        controls[key] = ui.textarea('Original fields (JSON)', value=pretty(value)).classes('w-full')
                elif key == 'regex_enabled':
                    controls[key] = ui.select({'auto': 'Detect /pattern/flags automatically', 'regex': 'Regex patterns', 'literal': 'Literal keywords'},
                        value='auto' if value is None else 'regex' if value else 'literal', label='Keyword matching')
                elif type(value) is bool:
                    controls[key] = ui.checkbox(rule_label(key), value=value)
                elif type(value) is int:
                    controls[key] = ui.number(rule_label(key), value=value, precision=0)
                elif key == 'role':
                    controls[key] = ui.select(_prompt_options(PROMPT_ROLE_LABELS, value), value=value, label='Message role')
                elif key == 'position':
                    controls[key] = ui.select(_prompt_options(LORE_PLACEMENT_LABELS, value), value=value, label='Placement')
                elif key == 'scan_depth' or isinstance(value, list):
                    controls[key] = ui.input(rule_label(key), value=pretty(value))
            for warning in rule.get('unsupported', []):
                ui.label(warning).classes('text-amber-300')
        def collect():
            for key, control in controls.items():
                if key == 'regex_enabled':
                    rule[key] = {'auto': None, 'regex': True, 'literal': False}[control.value]
                elif key in ('keys', 'secondary_keys'):
                    rule[key] = [item.strip() for item in control.value or [] if isinstance(item, str) and item.strip()]
                elif key == 'original' or key == 'scan_depth' or isinstance(rule.get(key), list):
                    rule[key] = json.loads(control.value or ('null' if key == 'scan_depth' else '{}'))
                elif type(rule.get(key)) is int:
                    rule[key] = int(control.value)
                else:
                    rule[key] = control.value
            return rule
        async def flush_pending_keys():
            # Blur can lose the race with a click on Save, so take any typed-but-unconfirmed key text now.
            from nicegui import ui as nicegui_ui
            ids = {key: controls[key].id for key in ('keys', 'secondary_keys')}
            try:
                pending = await nicegui_ui.run_javascript(
                    'const p = window.llPendingKeys || {}; const out = {};'
                    f'for (const [name, id] of Object.entries({json.dumps(ids)})) {{ out[name] = String(p[id] || ""); p[id] = ""; '
                    'try { getElement(id).$refs.qRef.updateInputValue("", true); } catch (e) {} }'
                    'return out;', timeout=3.0)
            except (TimeoutError, RuntimeError):
                return
            for key, text in (pending if isinstance(pending, dict) else {}).items():
                text = text.strip() if isinstance(text, str) else ''
                if key in controls and text and text not in controls[key].value:
                    controls[key].value = [*controls[key].value, text]
        def save():
            ref = ctx.store.save_entry(ctx.guild_id, entry['owner_kind'], entry['owner_id'], text.value or '', collect(), pinned.value, entry.get('ref'), entry.get('revision'))
            return ref
        def saved(_):
            panel.delete()
            refresh()
        if entry.get('ref'):
            destination = ui.select(owners, label='Destination', with_input=True).classes('w-full')
            def transfer(copy):
                if not destination.value:
                    raise ValueError('Choose where to move or copy the entry.')
                kind, ident = destination.value.split(':', 1)
                return ctx.store.transfer_entry(ctx.guild_id, entry['ref'], kind, int(ident), entry['revision'], copy=copy)
            def delete():
                ctx.store.delete_entry(ctx.guild_id, entry['ref'], entry['revision'])
                return True
        with ui.element('div').classes('ll-form-row'):
            ctx.button('Save lore', save, 'lore.edit', then=saved, prepare=flush_pending_keys)
            if entry.get('ref'):
                ctx.button('Move', lambda: transfer(False), 'lore.move', then=saved).props('outline')
                ctx.button('Copy', lambda: transfer(True), 'lore.copy', then=saved).props('outline')
                ui.button('Delete', color='negative', on_click=lambda: confirm_dialog(
                    ctx, 'Delete this lore entry?', ['This also keeps your deletion choice for future reimports.'],
                    'Delete entry', delete, 'lore.delete', then=saved))


def import_changes(changes):
    from nicegui import ui
    resolutions = {}
    for change in changes:
        with ui.expansion(f"{change['uid']} · {change['status']} · {change.get('disposition', '')}").classes('w-full ll-subpanel'):
            with ui.element('div').classes('ll-stack'):
                ui.label('Current: ' + (change.get('before') or '(none)')).classes('whitespace-pre-wrap ll-muted')
                ui.label('Incoming: ' + change.get('after', '')).classes('whitespace-pre-wrap ll-muted')
                for warning in change.get('warnings', []):
                    ui.label(warning).classes('text-amber-300')
                if change['status'] == 'conflict':
                    with ui.element('div').classes('ll-form-row'):
                        resolutions[change['uid']] = ui.select({'keep': 'Keep current entry', 'import': 'Use imported entry'}, label='Resolve conflict')
    return resolutions


def imports_panel(ctx):
    from nicegui import ui
    store, gid = ctx.store, ctx.guild_id
    with section('Character cards'):
        worlds = ctx.snapshot.worlds
        world = ui.select(worlds, label='Home world')
        preview_area = ui.column().classes('w-full')
        async def card_uploaded(event):
            card = parse_card(event.file.name, await event.file.read())
            selected_world = world.value
            preview = store.preview_card(gid, selected_world, card)
            preview_area.clear()
            with preview_area, ui.card().classes('w-full ll-stack ll-result'):
                ui.label('Preview: ' + card.name).classes('ll-subtitle')
                ui.label(card.data.get('description', '')).classes('whitespace-pre-wrap ll-muted')
                if card.avatar:
                    import base64
                    ui.image('data:image/png;base64,' + base64.b64encode(card.avatar).decode()).classes('w-24 h-24 rounded-lg')
                with ui.element('div').classes('ll-form-row'):
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
                with ui.element('div').classes('ll-form-row'):
                    ctx.button('Apply card import', apply, 'character.import', then=lambda _: ctx.refresh())
        with ui.element('div').classes('ll-stack'):
            ctx.upload(card_uploaded, 'Upload V2/V3 JSON or PNG card')
    with section('Direct lore entry import'):
        owners = {f"{owner['kind']}:{owner['id']}": owner['label'] for owner in ctx.snapshot.owners}
        with ui.element('div').classes('ll-form-row'):
            destination = ui.select(owners, label='Destination owner')
            def import_into_owner():
                if destination.value not in owners:
                    raise ValueError('Choose a destination owner.')
                kind, ident = destination.value.split(':', 1)
                direct_import_dialog(ctx, kind, int(ident))
            lucide_button('Import entries into selected owner', 'upload', on_click=import_into_owner)
    with section('Named lorebooks'):
        with ui.element('div').classes('ll-form-row'):
            book_name = ui.input('Lorebook name')
            target = ui.select({'guild': 'Server lorebook', 'channel': 'Channel lorebook'}, value='guild', label='Kind')
            channels = {r['channel_id']: ctx.channel_names.get(r['channel_id'], str(r['channel_id'])) for r in ctx.snapshot.channels}
            channel = ui.select(channels, label='Channel').classes('ll-wide')
            channel.bind_visibility_from(target, 'value', backward=lambda kind: kind == 'channel')
            ctx.button('Create lorebook', lambda: store.create_lorebook(gid, book_name.value or '', target.value, channel.value or 0), 'book.create', then=lambda _: ctx.refresh())
        for book in ctx.snapshot.lorebooks:
            title = book['name'] + ' · ' + ('Server lorebook' if book['target_kind'] == 'guild' else 'Channel lorebook')
            if book['target_kind'] == 'channel' and ctx.channel_names.get(book['target_id']):
                title += ' · ' + ctx.channel_names[book['target_id']]
            with ui.expansion(title).classes('w-full'):
                with ui.element('div').classes('ll-stack'):
                    with ui.element('div').classes('ll-form-row'):
                        lucide_button('Edit entries in Lore', 'pencil', on_click=lambda book=book: ctx.refresh('lore', owner=f"book:{book['id']}")).props('outline')
                        lucide_button('Delete lorebook', 'trash-2', color='negative', on_click=lambda book=book: delete_book_dialog(ctx, book))
                    if book['target_kind'] == 'guild':
                        linked_spaces = store.lorebook_links(book['id'])
                        with ui.element('div').classes('ll-form-row'):
                            for space in ctx.snapshot.spaces:
                                enabled = space['id'] in linked_spaces
                                def assign(book=book, space=space, enabled=enabled):
                                    store.set_lorebook_space(gid, book['id'], space['id'], not enabled)
                                    return True
                                ctx.button(('Disable in ' if enabled else 'Enable in ') + space['name'], assign, 'book.assign', then=lambda _: ctx.refresh()).props('outline')
                    area = ui.column().classes('w-full')
                    async def book_uploaded(event, book=book, area=area):
                        imported = parse_lorebook(await event.file.read())
                        revision = store.owner_revision(gid, 'book', book['id'])
                        changes = store.preview_import(gid, 'book', book['id'], imported)
                        area.clear()
                        with area, ui.element('div').classes('ll-stack'):
                            label = 'RisuAI' if imported.source_format == 'risu' else 'SillyTavern'
                            ui.label(f'{label} lorebook detected · {len(imported.entries)} entries').classes('ll-subtitle')
                            decisions = import_changes(changes)
                            expires = __import__('time').time() + 900
                            def apply():
                                if __import__('time').time() > expires:
                                    raise ValueError('Preview expired; upload again')
                                return store.sync_lorebook(gid, book['id'], imported, {key: c.value for key, c in decisions.items()}, revision)
                            with ui.element('div').classes('ll-form-row'):
                                ctx.button('Apply lorebook sync', apply, 'book.sync', {'id': book['id']}, then=lambda _: ctx.refresh())
                    ctx.upload(book_uploaded, 'Upload JSON lorebook for preview')


def _preset_snapshot(state, name):
    """Text snapshot of the editable preset state (name, bundle, raw-JSON textareas as typed) for the save bar's dirty check.

    Viewing another Purpose fills in its default blocks, so every purpose is normalised first; raw JSON is compared as text and never parsed."""
    def plain(value):
        if isinstance(value, float) and value == int(value):
            return int(value)
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, list):
            return [plain(item) for item in value]
        return value
    bundle = copy.deepcopy(state['bundle'])
    purposes = bundle.get('purposes', {})
    for purpose in PURPOSES:
        if purposes.get(purpose) is None:
            purposes[purpose] = copy.deepcopy(default_bundle()['purposes'][purpose])
    if not bundle.get('source'):
        bundle.pop('source', None)
    raw = {}
    for blocks in purposes.values():
        for b in blocks:
            control = state['raw_controls'].get(b['id'])
            text = control[1].value if control and control[0] is not None else pretty(b.get('raw'))
            raw[b['id']] = text
            b.pop('raw', None)
    return json.dumps({'name': name.value or '', 'bundle': plain(bundle), 'raw': raw}, sort_keys=True, default=str)


def presets_panel(ctx):
    from nicegui import ui
    store, gid = ctx.store, ctx.guild_id
    presets = store.list_presets(gid)
    IMPORTED = -1

    def library_options(rows, preview=False):
        return {0: 'Built-in default', **{r['id']: f"{r['name']} · draft {r['revision']}" for r in rows}, **({IMPORTED: 'Imported (not saved)'} if preview else {})}

    def subtitle_text(rows):
        current = store.active_preset(gid)
        return f"Active preset: {library_options(rows).get(current['id'], str(current['id']))} · revision {current['revision']}"
    choices = library_options(presets)
    active = store.active_preset(gid)
    selected = next((r for r in presets if r['id'] == active['id']), None)
    state = {'id': active['id'], 'revision': selected['revision'] if selected else 0, 'bundle': store.preset_bundle(gid, active['id']), 'raw_controls': {},
             'controls': [], 'after_render': None, 'saved': None, 'preview': False, 'label': selected['name'] if selected else 'Built-in default'}
    editor = None
    baseline_label = {'text': state['label']}

    def collect():
        for b, control in state['raw_controls'].values():
            b['raw'] = json.loads(control.value or '{}')
        return copy.deepcopy(state['bundle'])

    def preset_menu():
        with more_menu('preset'):
            ui.menu_item('Save as new preset', on_click=lambda: save_new()).props('role=menuitem')
            ui.menu_item('Activate saved revision', on_click=lambda: activate_saved()).props('role=menuitem')
            ui.separator()
            ui.menu_item('Delete preset', on_click=lambda: ask_delete()).classes('text-negative').props('role=menuitem')

    with section('Preset library', header=preset_menu):
        subtitle = ui.label(subtitle_text(presets)).classes('ll-subtitle')
        select = ui.select(choices, value=active['id'], label='Preset library').classes('w-full')
        name = ui.input('Preset name', value=selected['name'] if selected else 'Default copy').classes('w-full')

    def rebase(label):
        """Take the saved baseline (after a render) that Reset and the dirty check compare against."""
        state['label'] = baseline_label['text'] = label
        state['saved'] = {'snapshot': _preset_snapshot(state, name), 'bundle': copy.deepcopy(state['bundle']), 'name': name.value, 'id': state['id'], 'revision': state['revision']}
        if editor is not None:
            editor.name = label

    def dirty():
        return _preset_snapshot(state, name) != state['saved']['snapshot']

    def tracked():
        return {control: None for control in [name, *state['controls']]}

    def after_render():
        if editor is not None:
            ctx.savebar.retrack(editor, tracked())
            if not state.get('quiet'):
                ctx.savebar.check(editor)

    def show_library(rows=None):
        """Point the library select at the loaded preset, or at 'Imported (not saved)' while an import preview is unsaved."""
        select.set_options(library_options(rows if rows is not None else store.list_presets(gid), state['preview']), value=IMPORTED if state['preview'] else state['id'])

    def reset():
        baseline = state['saved']
        was_preview = state['preview']
        state.update(id=baseline['id'], revision=baseline['revision'], bundle=copy.deepcopy(baseline['bundle']), label=baseline_label['text'], preview=False)
        editor.name = baseline_label['text']
        if was_preview:
            show_library()  # drops the 'Imported (not saved)' option
        else:
            select.value = baseline['id']
        name.value = baseline['name']
        render_editor.refresh()

    def adopt(label):
        """Show a freshly loaded, saved or imported preset as the new unmodified baseline."""
        state['quiet'] = True
        try:
            render_editor.refresh()
        finally:
            state['quiet'] = False
        rebase(label)
        ctx.savebar.check(editor)

    async def load():
        shown = IMPORTED if state['preview'] else state['id']
        if select.value == shown:
            return
        if ctx.savebar.refuse():
            select.value = shown
            return
        bundle = await ctx.run(lambda: store.preset_bundle(gid, select.value))
        if bundle is None:
            select.value = shown
            return
        chosen = next((r for r in store.list_presets(gid) if r['id'] == select.value), None)
        state.update(id=select.value, revision=chosen['revision'] if chosen else 0, bundle=bundle, preview=False)
        show_library()
        state.get('open_blocks', set()).clear()
        name.value = chosen['name'] if chosen else 'Default copy'
        adopt(chosen['name'] if chosen else 'Built-in default')

    select.on_value_change(lambda _: load())

    def save():
        return store.save_preset(gid, name.value or '', collect(), state['id'] or None, state['revision'] if state['id'] else None)

    def saved(result):
        state['id'], state['revision'] = result
        state['preview'] = False
        fresh = store.list_presets(gid)
        show_library(fresh)
        subtitle.set_text(subtitle_text(fresh))
        rebase(name.value or state['label'])
        ctx.savebar.check(editor)
        ui.notify('Draft saved. Activate it when ready.', type='positive')

    with section('Prompt blocks'):
        purpose = ui.select({p: p.title() for p in PURPOSES}, value='dialogue', label='Purpose')
        MacroHighlight()
        render_editor = _preset_editor(ctx, state, purpose, collect)
        render_editor()
        rebase(state['label'])
        editor = ctx.savebar.track(state['label'], tracked(), save, 'preset.save', then=saved, dirty=dirty, reset=reset, save_label='Save draft')
        state['after_render'] = after_render
        async def change_purpose():
            if await ctx.run(lambda: True):
                collect()
                render_editor.refresh()
        purpose.on_value_change(lambda _: change_purpose())

    async def save_new():
        if ctx.savebar.active is not None and ctx.savebar.active is not editor:
            ctx.savebar.refuse()
            return

        def duplicate():
            taken = {r['name'] for r in store.list_presets(gid)}
            base = (name.value or '').strip()
            if base in taken:  # an unchanged name would hit the unique name; propose the next free "<name> copy"
                base = base[:60]
                proposal = f'{base} copy'
                n = 2
                while proposal in taken:
                    proposal, n = f'{base} copy {n}', n + 1
                name.value = proposal
            return store.save_preset(gid, name.value or '', collect())
        ok, result = await ctx._attempt(duplicate, 'preset.create')
        if ok:
            ctx.savebar.settle()
            saved(result)

    async def activate_saved():
        def activate():
            store.activate_preset(gid, state['id'], state['revision'], ctx.service.providers)
            return True
        if await ctx.run(activate, 'preset.activate'):
            await ctx.refresh()

    def ask_delete():
        if ctx.savebar.refuse():
            return
        def delete():
            store.delete_preset(gid, state['id'])
            return True
        if not state['id']:
            ui.notify('The built-in default cannot be deleted', type='negative', timeout=8000)
            return
        current = next((r for r in store.list_presets(gid) if r['id'] == state['id']), None)
        confirm_dialog(ctx, f"Delete preset {current['name'] if current else state['id']}?",
                       ['This permanently deletes the preset. The active preset cannot be deleted; activate another one first.'],
                       'Delete preset', delete, 'preset.delete', then=lambda _: ctx.refresh())

    def imported(bundle):
        editor.name = state['label'] = 'Imported preset'
        state.update(id=0, revision=0, bundle=bundle, preview=True)
        state.get('open_blocks', set()).clear()
        show_library()
        name.value = 'Imported preset'
        render_editor.refresh()
        ctx.savebar.check(editor)  # an import preview is unsaved: the baseline stays the previous preset
        ui.notify('Import preview loaded. Resolve compatibility items, save a draft, then activate.')

    _preset_import(ctx, state, imported)
    _preset_export(ctx, collect)
    _preset_preview(ctx, purpose, collect)


def _read_button(ctx, text, operation):
    """A button that runs a read-only operation through the guard but stays available while the save bar is active."""
    from nicegui import ui
    async def clicked():
        await ctx.run(operation)
    return ui.button(text, on_click=clicked)


def _span(text=''):
    """An inline text element (``ui.label`` renders a div, which a button may not contain)."""
    from nicegui.elements.mixins.text_element import TextElement
    return TextElement(tag='span', text=text)


PROMPT_ROLE_LABELS = {'system': 'System', 'user': 'User', 'assistant': 'Assistant'}
PROMPT_PLACEMENT_LABELS = {'relative': 'Relative to the prompt', 'in_chat': 'In chat at depth'}
LORE_PLACEMENT_LABELS = {'before_char': 'Before character', 'after_char': 'After character', 'before_examples': 'Before examples',
                         'after_examples': 'After examples', 'in_chat': 'In chat at depth'}
PROMPT_PLACEMENT_SHORT = {'relative': 'Relative', 'in_chat': 'In chat'}
PROMPT_SOURCE_LABELS = {
    'text': 'Custom text', 'history': 'Chat history', 'payload': 'Request payload', 'location': 'Location', 'description': 'Character description',
    'personality': 'Character personality', 'scenario': 'Scenario', 'opening': 'Opening message', 'examples': 'Example dialogue',
    'card_instructions': 'Card instructions', 'card_post_history': 'Card post-history instructions',
    'lore_before_char': 'Lore before character', 'lore_after_char': 'Lore after character', 'lore_before_examples': 'Lore before examples',
    'lore_after_examples': 'Lore after examples', 'lore_in_chat': 'Lore in chat', 'personal': 'Personal facts', 'encounters': 'Encounters',
    'summary': 'Scene summary', 'preceding': 'Preceding messages', 'recent': 'Recent messages'}


def _prompt_options(labels, *current):
    """``{code: label}`` for a select; codes not in ``labels`` (imported presets) stay selectable under their raw code."""
    return {**labels, **{c: c for c in current if c not in labels}}


def _preset_editor(ctx, state, purpose, collect):
    from nicegui import ui

    @ui.refreshable
    def render_editor():
        state['raw_controls'] = {}
        state['controls'] = controls = []
        def tracked(control):
            controls.append(control)
            return control
        blocks = state['bundle']['purposes'].setdefault(purpose.value, copy.deepcopy(default_bundle()['purposes'][purpose.value]))
        with ui.column().classes('w-full gap-4') as ordered:
            for position, b in enumerate(blocks):
                with ui.card().classes('w-full prompt-block ll-stack ll-result') as card:
                    card.prompt_id = b['id']
                    key = (purpose.value, b['id'])  # block ids repeat across purposes
                    opened = key in state.setdefault('open_blocks', set())
                    if opened:
                        card.classes('ll-block-open')
                    with ui.element('div').classes('ll-block-head'):
                        lucide('grip-vertical', '1.25em').classes('prompt-handle cursor-grab self-center ll-icon-solo')
                        toggle = ui.element('button').classes('ll-block-toggle prompt-toggle')
                        toggle.props['type'] = 'button'
                        toggle.props['aria-expanded'] = 'true' if opened else 'false'
                        with toggle:
                            # phrasing content only (a <button> cannot hold divs); trailing spaces keep the accessible name readable
                            _span().classes('ll-block-name').bind_text_from(b, 'name', backward=lambda v: (v or 'Untitled block') + ' ')
                            with ui.element('span').classes('ll-block-meta'):
                                _span().bind_text_from(b, 'role', backward=lambda v: PROMPT_ROLE_LABELS.get(v, v))
                                _span(' \u00b7 ')
                                _span().bind_text_from(b, 'placement', backward=lambda v: f'{PROMPT_PLACEMENT_SHORT.get(v, v)} ')
                            _span('Disabled').classes('ll-pill ll-pill-off').bind_visibility_from(b, 'enabled', backward=lambda v: not v)
                            lucide('chevron-down', '1.25em').classes('ll-icon-solo ll-block-caret')
                        async def remove(b=b, key=key):
                            if await ctx.run(lambda: True):
                                collect()
                                blocks.remove(b)
                                state['open_blocks'].discard(key)
                                render_editor.refresh()
                        async def move(delta, b=b):
                            # same path as a drag: collect the fields, reorder the list, re-render (the save bar sees the new order)
                            if await ctx.run(lambda: True):
                                collect()
                                at = blocks.index(b)
                                if 0 <= at + delta < len(blocks):
                                    blocks.insert(at + delta, blocks.pop(at))
                                render_editor.refresh()
                        with more_menu(b['name'] or 'Untitled block') as menu:
                            more_button = menu.parent_slot.parent
                            up = ui.menu_item('Move up', on_click=lambda b=b: move(-1, b)).props('role=menuitem')
                            down = ui.menu_item('Move down', on_click=lambda b=b: move(1, b)).props('role=menuitem')
                            if position == 0:
                                up.props('disable')
                            if position == len(blocks) - 1:
                                down.props('disable')
                            ui.separator()
                            ui.menu_item('Remove block', on_click=remove).classes('text-negative').props('role=menuitem')
                    body = ui.element('div').classes('ll-stack w-full')
                    if not opened:
                        body.classes('ll-collapsed')
                    def flip(card=card, toggle=toggle, body=body, bid=key):
                        open_ids = state['open_blocks']
                        now_open = bid not in open_ids
                        (open_ids.add if now_open else open_ids.discard)(bid)
                        card.classes(add='ll-block-open') if now_open else card.classes(remove='ll-block-open')
                        body.classes(remove='ll-collapsed') if now_open else body.classes(add='ll-collapsed')
                        toggle.props['aria-expanded'] = 'true' if now_open else 'false'
                        toggle.update()
                    toggle.on('click', flip)
                    with body:
                        with ui.element('div').classes('ll-form-row'):
                            name_input = tracked(ui.input('Block name')).bind_value(b, 'name')
                            def rename(event, button=more_button):
                                button.props['aria-label'] = f"More actions for {event.value or 'Untitled block'}"
                                button.update()
                            name_input.on_value_change(rename)
                            tracked(ui.switch('Enabled')).bind_value(b, 'enabled')
                        sources = _prompt_options({k: PROMPT_SOURCE_LABELS.get(k, k) for k in sorted(SOURCES, key=lambda k: PROMPT_SOURCE_LABELS.get(k, k))}, b['source'])
                        tracked(ui.select(sources, label='Context source')).bind_value(b, 'source').classes('w-full')
                        tracked(ui.textarea('Prompt text / template')).bind_value(b, 'content').classes('w-full ll-macro').props('rows=6')
                        with ui.element('div').classes('ll-form-row'):
                            tracked(ui.select(_prompt_options(PROMPT_ROLE_LABELS, b['role']), label='Role')).bind_value(b, 'role')
                            tracked(ui.select(_prompt_options(PROMPT_PLACEMENT_LABELS, b['placement']), label='Placement')).bind_value(b, 'placement')
                            tracked(ui.number('Depth', precision=0)).bind_value(b, 'depth', backward=lambda v: int(v or 0))
                            tracked(ui.number('Injection order', precision=0)).bind_value(b, 'order', backward=lambda v: int(v or 0))
                            tracked(ui.number('Trimming priority', precision=0)).bind_value(b, 'priority', backward=lambda v: int(v or 0))
                        tracked(ui.select({'': 'Exact placement', 'top_system': 'Move to top-level system instructions', 'user': 'Convert late system block to user instructions'}, label='System instruction placement')).bind_value(b, 'adaptation').classes('w-full')
                        with ui.expansion('Preserved import fields and compatibility remapping').classes('w-full ll-subpanel'), ui.element('div').classes('ll-stack'):
                            raw = tracked(ui.textarea('Original prompt fields (JSON)', value=pretty(b['raw']))).classes('w-full').props('rows=6')
                            state['raw_controls'][b['id']] = (b, raw)
                            ui.label('Block ID: ' + b['id']).classes('ll-muted')
                            ui.label('To remap triggers/extensions: clear injection_trigger and set extension to false.').classes('ll-muted')
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
                fresh = block('custom-' + uuid.uuid4().hex[:8])
                blocks.append(fresh)
                state.setdefault('open_blocks', set()).add((purpose.value, fresh['id']))
                render_editor.refresh()
        with ui.element('div').classes('ll-form-row'):
            ui.button('Add prompt block', on_click=add).props('outline')
        for problem in compatibility(state['bundle'], ctx.service.providers):
            ui.label(problem).classes('text-amber-300')
        source = state['bundle'].setdefault('source', {})
        if source.get('assistant_prefill'):
            tracked(ui.checkbox('Disable imported assistant prefill')).bind_value(source, 'prefill_disabled')
        if state['after_render']:
            state['after_render']()

    return render_editor


def _preset_import(ctx, state, imported):
    from nicegui import ui
    with section('Import preset'):
        order_area = ui.column().classes('w-full')
        async def uploaded(event):
            data = await event.file.read()
            bundle, profiles = parse_preset(data)
            if profiles:
                order_area.clear()
                with order_area, ui.element('div').classes('ll-form-row'):
                    order = ui.select({p['index']: 'SillyTavern order ' + p['label'] for p in profiles}, label='Choose an order profile')
                    def choose():
                        if order.value is None:
                            raise ValueError('Choose a prompt order profile.')
                        return parse_preset(data, order.value)[0]
                    ctx.button('Preview selected profile', choose, then=imported)
            else:
                imported(bundle)
        with ui.element('div').classes('ll-stack'):
            ctx.upload(uploaded, 'Upload native or SillyTavern Chat Completion JSON')


def _preset_export(ctx, collect):
    from nicegui import ui
    with section('Export preset'):
        def native_export():
            ui.download.content(pretty(export_preset(collect())), 'llmcord-preset.json', 'application/json')
            return True
        with ui.element('div').classes('ll-form-row'):
            _read_button(ctx, 'Export full native bundle', native_export).props('outline')
        def block_options():
            try:
                portable = set(ST_MARKERS.values())  # the check export_preset uses
                return {b['id']: f"{b['name'] or 'Untitled block'} ({b['id']})"
                        + ('' if b['source'] == 'text' or b['source'] in portable or b['source'].startswith('unsupported:') else ' \u2014 not portable')
                        for b in purpose_blocks(collect(), 'dialogue')}
            except ValueError:
                return {}
        omit = ui.select(block_options(), multiple=True, value=[], label='Leave out blocks').props('use-chips').classes('w-full')
        omit.tooltip('SillyTavern cannot represent some blocks; choose them here to export without them.')
        def refresh_options():
            omit.set_options(block_options(), value=[v for v in omit.value if v in block_options()])
        omit.on('popup-show', refresh_options)
        def st_export():
            ui.download.content(pretty(export_preset(collect(), sillytavern=True, omit=list(omit.value))), 'sillytavern-preset.json', 'application/json')
            return True
        with ui.element('div').classes('ll-form-row'):
            _read_button(ctx, 'Export SillyTavern dialogue preset', st_export).props('outline')


def _preset_preview(ctx, purpose, collect):
    from nicegui import ui
    with section('Assembled request preview'):
        chars = {r['id']: r['name'] for r in ctx.snapshot.characters}
        with ui.element('div').classes('ll-form-row'):
            character = ui.select(chars, label='Sample character')
            channel = ui.select({row['channel_id']: ctx.channel_names.get(row['channel_id'], str(row['channel_id'])) for row in ctx.snapshot.channels}, label='Sample channel (optional)')
        sample = ui.textarea('Sample input', value='Hello!').classes('w-full')
        history = ui.textarea('Sample history (one message per line)').classes('w-full')
        diagnostics = ui.column().classes('w-full')
        def preview():
            bundle = collect()
            request = ctx.service.preview_prompt(ctx.guild_id, bundle, purpose.value, character.value, sample.value or '', history.value or '', channel.value)
            diagnostics.clear()
            with diagnostics:
                ui.label(f'Estimated input tokens: {request.estimated_tokens}')
                names = {b['id']: b.get('name') or b['id'] for b in purpose_blocks(bundle, purpose.value)}
                adapt_labels = {'top_system': 'moved to top-level system instructions', 'user': 'converted to user instructions'}
                def omitted_text(entry):
                    ident, sep, rest = entry.partition(':')
                    if sep and rest.isdigit():
                        return f'{names.get(ident, ident)} (history message {int(rest) + 1})'
                    return names.get(entry, entry)
                def adapted_text(entry):
                    ident, sep, rest = entry.partition(': ')
                    return f'{names.get(ident, ident)}: {adapt_labels.get(rest, rest)}' if sep else entry
                ui.label('Omitted: ' + (', '.join(omitted_text(e) for e in request.omitted) or 'none'))
                ui.label('Adaptations: ' + (', '.join(adapted_text(e) for e in request.adaptations) or 'none'))
                for message in request.messages:
                    with ui.expansion(PROMPT_ROLE_LABELS.get(message.role, message.role)).classes('w-full'):
                        ui.label(message.text).classes('whitespace-pre-wrap')
            return True
        with ui.element('div').classes('ll-form-row'):
            _read_button(ctx, 'Preview without a model call', preview)
