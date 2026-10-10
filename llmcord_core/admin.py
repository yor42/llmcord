"""UI-independent operations, authorization and metadata-only auditing."""
from __future__ import annotations

import inspect
import json
import logging
import sqlite3
import time
import traceback

from .avatars import AvatarPublisher
from .backend import effective_backend
from .models import TurnMessage
from .prompts import compile_prompt, time_values

PURPOSE_ROLES = {'dialogue': 'dialogue', 'director': 'director', 'extraction': 'memory', 'summary': 'memory', 'images': 'dialogue'}
OPERATOR_AUDIT_GUILD = 0  # no Discord snowflake is 0; operator actions have no server
log = logging.getLogger(__name__)


class AdminService:
    def __init__(self, app, config_path='config.yaml'):
        self.app, self.store, self.auth = app, app.state.store, app.state.auth
        self.avatars = AvatarPublisher(self.store, app.state.http, app.state.bot_token)
        self.model_config = getattr(app.state, 'config_models', {})
        self.limits = getattr(app.state, 'config_limits', {})
        # config.yaml profiles/roles are loaded once by create_app; the dashboard and the preview share them.
        self.config_profiles = getattr(app.state, 'config_profiles', {})
        self.config_roles = getattr(app.state, 'config_roles', {})
        self.config_error = getattr(app.state, 'config_error', None)
        self._purposes = None

    def effective_backend(self):
        return effective_backend(self.config_profiles, self.config_roles, self.store.model_profile_rows(), self.store.model_roles())

    def _purpose_settings(self):
        version = self.store.model_backend_version()
        if self._purposes is None or self._purposes[0] != version:
            backend = self.effective_backend()
            providers, budgets = {}, {}
            for purpose, role in PURPOSE_ROLES.items():
                profile = backend.profiles.get(backend.roles.get(role) or backend.roles.get('dialogue'))
                providers[purpose] = profile.prompt_provider if profile else 'compatible'
                budgets[purpose] = min(self.limits.get('max_input_tokens', 12000), (profile.context_tokens if profile else 32000) - self.limits.get('max_output_tokens', 700))
            self._purposes = (version, providers, budgets)
        return self._purposes

    @property
    def providers(self):
        return self._purpose_settings()[1]

    @property
    def budgets(self):
        return self._purpose_settings()[2]

    async def run(self, ident, guild_id, operation, action=None, detail=None):
        # Live (socket.io) calls carry no per-request CSRF/origin: the boundary is the socket's cookie-to-client
        # binding plus socket.io CORS (cors_allowed_origins, dashboard.mount_dashboard), and this per-action admin guard (SEC-03).
        session = await self.auth.guard(ident, guild_id)
        result = operation()
        if inspect.isawaitable(result):
            result = await result
        if action:
            self._audit(guild_id, int(session['user']['id']), action, detail, result)
        return result

    async def run_operator(self, ident, operation, action=None, detail=None):
        # Live (socket.io) calls carry no per-request CSRF/origin: the boundary is the socket's cookie-to-client
        # binding plus socket.io CORS (cors_allowed_origins, dashboard.mount_dashboard), and this per-action operator guard.
        session = await self.auth.guard_operator(ident)
        result = operation()
        if inspect.isawaitable(result):
            result = await result
        if action:
            self._audit(OPERATOR_AUDIT_GUILD, int(session['user']['id']), action, detail, result)
        return result

    def _audit(self, guild_id, actor_id, action, detail, result):
        # MNT-38: the operation has committed, so a failing audit must not report it as failed (a retry would conflict or duplicate).
        # Logged without the detail or exception messages: they may hold user text. Detail-building errors and the write's sqlite3.Error are absorbed.
        try:
            if callable(detail):  # detail may be derived from the operation result
                detail = detail(result)
            json.dumps(detail or {})  # store.audit serializes; a failure here must not fail the committed action either
        except Exception as error:
            log.error('Audit detail failed after a committed admin action; auditing without it (action=%s guild=%s actor=%s error=%s)\n%s',
                action, guild_id, actor_id, type(error).__name__, ''.join(traceback.format_tb(error.__traceback__)))
            detail = {'detail_error': True}
        try:
            self.store.audit(guild_id, actor_id, action, detail or {})
        except sqlite3.Error:
            log.warning('Audit write failed after a committed admin action (action=%s guild=%s actor=%s)', action, guild_id, actor_id, exc_info=True)

    def owners(self, guild_id, channel_names=None, thread_names=None):
        store = self.store
        return self.owners_from(guild_id, store.list_spaces(guild_id), store.list_characters(guild_id), store.list_channels(guild_id),
            store.list_lorebooks(guild_id), store.thread_lore_scopes(guild_id), channel_names, thread_names)

    @staticmethod
    def thread_label(scope_id, thread_names=None):
        # UI-03/UI-44: thread_names None means the fetch failed or was skipped, so the archived state is unknown.
        name = (thread_names or {}).get(scope_id)
        if name:
            return f"Thread {name}"
        return f"Thread …{str(scope_id)[-4:]}" if thread_names is None else f"Archived thread …{str(scope_id)[-4:]}"

    def owners_from(self, guild_id, spaces, characters, channels, lorebooks, thread_scopes, channel_names=None, thread_names=None):
        owners = []
        for row in spaces:
            owners.append({'kind': 'space', 'id': row['id'], 'label': f"{row['kind'].title()}: {row['name']}"})
        for row in characters:
            owners.append({'kind': 'character', 'id': row['id'], 'label': 'Character: ' + row['name']})
        for row in channels:
            owners.append({'kind': 'channel', 'id': row['channel_id'], 'label': f"Channel: {(channel_names or {}).get(row['channel_id'], row['channel_id'])}"})
        owners.append({'kind': 'guild', 'id': guild_id, 'label': 'Server: Server-wide lore'})
        for row in lorebooks:
            owners.append({'kind': 'book', 'id': row['id'], 'label': 'Lorebook: ' + row['name']})
        for scope_id in thread_scopes:
            owners.append({'kind': 'thread', 'id': scope_id, 'label': self.thread_label(scope_id, thread_names)})
        return owners

    def preview_prompt(self, guild_id, bundle, purpose, character_id, sample, sample_history, channel_id=None):
        # Preview uses supplied sample data only; it never reads scene or personal-memory tables.
        card, character_name = {}, 'Character'
        if character_id:
            self.store.validate_owner(guild_id, 'character', character_id)
            row = self.store.character_by_id(character_id)
            card, character_name = json.loads(row['card']), row['name']
        values = {'char': character_name, 'user': 'Sample user', 'group': character_name,
            'payload': sample, 'description': card.get('description', ''), 'personality': card.get('personality', ''),
            'scenario': card.get('scenario', ''), 'opening': card.get('first_mes', ''), 'examples': card.get('mes_example', ''),
            'mesExamples': card.get('mes_example', ''), 'mesExamplesRaw': card.get('mes_example', ''),
            'card_instructions': card.get('system_prompt', ''), 'card_post_history': card.get('post_history_instructions', ''),
            'location': 'Sample world', 'summary': ''}
        values.update(time_values((int(time.time() * 1000) - 1420070400000) << 22, self.store.resolve_timezone(guild_id, None)[0]))
        if purpose == 'dialogue':
            if channel_id:
                self.store.validate_owner(guild_id, 'channel', channel_id)
                binding = self.store.channel(channel_id)
                guidelines = self.store.scene_guidelines(guild_id, binding['space_id'], channel_id)
            elif character_id:
                guidelines = {'world_guidelines': self.store.guidelines(guild_id, 'space', row['world_id'])}
            else:
                guidelines = {}
            values.update({key: row['content'] for key, row in guidelines.items()})
        history = [TurnMessage('user', text) for text in sample_history.splitlines() if text] + [TurnMessage('user', sample)]
        contract = 'Begin your response with <emotion>neutral</emotion> on its own line, then write your dialogue.' if purpose == 'dialogue' else ''
        if purpose in {'director', 'extraction'}:
            from .models import DIRECTOR_SCHEMA, MEMORY_SCHEMA
            contract = 'Return only the required structured result. Output schema: ' + json.dumps(DIRECTOR_SCHEMA if purpose == 'director' else MEMORY_SCHEMA)
        return compile_prompt(bundle, purpose, values, history, self.providers[purpose], self.budgets[purpose], contract=contract)
