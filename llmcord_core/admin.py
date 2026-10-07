"""UI-independent operations, authorization and metadata-only auditing."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

from .avatars import AvatarPublisher
from .config import prompt_provider
from .models import TurnMessage
from .prompts import compile_prompt


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

    async def run(self, ident, guild_id, csrf, operation, action=None, detail=None):
        session = await self.auth.guard(ident, guild_id, csrf, self.app.state.base_url)
        result = operation()
        if inspect.isawaitable(result):
            result = await result
        if action:
            self.store.audit(guild_id, int(session['user']['id']), action, detail or {})
        return result

    def owners(self, guild_id):
        owners = []
        for row in self.store.list_spaces(guild_id):
            owners.append({'kind': 'space', 'id': row['id'], 'label': f"{row['kind'].title()}: {row['name']}"})
        for row in self.store.all('SELECT * FROM characters WHERE guild_id=? ORDER BY name', (guild_id,)):
            owners.append({'kind': 'character', 'id': row['id'], 'label': 'Character: ' + row['name']})
        for row in self.store.all('SELECT * FROM channels WHERE guild_id=? ORDER BY channel_id', (guild_id,)):
            owners.append({'kind': 'channel', 'id': row['channel_id'], 'label': f"Channel: {row['channel_id']}"})
        owners.append({'kind': 'guild', 'id': guild_id, 'label': 'Guild: Server-wide lore'})
        for row in self.store.list_lorebooks(guild_id):
            owners.append({'kind': 'book', 'id': row['id'], 'label': 'Book: ' + row['name']})
        for row in self.store.all("SELECT DISTINCT scope_id FROM lore WHERE guild_id=? AND scope_kind='thread'", (guild_id,)):
            owners.append({'kind': 'thread', 'id': row['scope_id'], 'label': f"Thread: {row['scope_id']}"})
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
