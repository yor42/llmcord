"""UI-independent operations, authorization and metadata-only auditing."""
from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

from .avatars import AvatarPublisher
from .config import prompt_provider
from .models import TurnMessage
from .prompts import compile_prompt, time_values

OPERATOR_AUDIT_GUILD = 0  # no Discord snowflake is 0; operator actions have no server


class AdminService:
    def __init__(self, app, config_path='config.yaml'):
        self.app, self.store, self.auth = app, app.state.store, app.state.auth
        self.avatars = AvatarPublisher(self.store, app.state.http, app.state.bot_token)
        raw = {}
        if Path(config_path).exists():
            import yaml
            raw = yaml.safe_load(Path(config_path).read_text(encoding='utf-8')) or {}
        models = raw.get('models', {})
        self.model_config = models
        profiles = models.get('profiles', {})
        self.providers, self.budgets = {}, {}
        for purpose, role in {'dialogue': 'dialogue', 'director': 'director', 'extraction': 'memory', 'summary': 'memory', 'images': 'dialogue'}.items():
            profile = profiles.get(models.get(role, models.get('dialogue')), {})
            self.providers[purpose] = prompt_provider(profile.get('provider', 'compatible'), profile.get('base_url'))
            self.budgets[purpose] = min(raw.get('limits', {}).get('max_input_tokens', 12000), profile.get('context_tokens', 32000) - raw.get('limits', {}).get('max_output_tokens', 700))

    async def run(self, ident, guild_id, operation, action=None, detail=None):
        # Live (socket.io) calls carry no per-request CSRF/origin: the boundary is the socket's cookie-to-client
        # binding plus socket.io CORS (cors_allowed_origins, dashboard.mount_dashboard), and this per-action admin guard (SEC-03).
        session = await self.auth.guard(ident, guild_id)
        result = operation()
        if inspect.isawaitable(result):
            result = await result
        if action:
            if callable(detail):  # detail may be derived from the operation result
                detail = detail(result)
            self.store.audit(guild_id, int(session['user']['id']), action, detail or {})
        return result

    async def run_operator(self, ident, operation, action=None, detail=None):
        # Live (socket.io) calls carry no per-request CSRF/origin: the boundary is the socket's cookie-to-client
        # binding plus socket.io CORS (cors_allowed_origins, dashboard.mount_dashboard), and this per-action operator guard.
        session = await self.auth.guard_operator(ident)
        result = operation()
        if inspect.isawaitable(result):
            result = await result
        if action:
            if callable(detail):
                detail = detail(result)
            self.store.audit(OPERATOR_AUDIT_GUILD, int(session['user']['id']), action, detail or {})
        return result

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
