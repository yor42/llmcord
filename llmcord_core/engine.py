from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .config import LIMIT_DEFAULTS, Settings
from .lore import estimate_tokens, lore_scopes
from .world_info import evaluate
from .models import DIRECTOR_SCHEMA, MEMORY_SCHEMA, ImageInput, ModelGateway, TurnMessage
from .store import Store
from .prompts import compile_prompt
from .errors import error_detail
from .identity import speaker_context, user_line


def tail_tokens(text: str, budget: int) -> str:
    """Keep the end of ``text`` so estimate_tokens stays within ``budget``."""
    data = text.encode('utf-8')
    return text if estimate_tokens(text) <= budget else data[-3 * budget:].decode('utf-8', 'ignore')


@dataclass(frozen=True)
class SceneContext:
    guild_id: int
    channel_id: int
    parent_channel_id: int | None
    space_id: int
    user_id: int
    user_message_id: int
    text: str
    parent_message_id: int | None
    recent: list[dict]
    images: list[ImageInput]
    ambient: bool = False
    forced_character_id: int | None = None
    preset: dict | None = None
    user_label: str = ''
    guidelines: dict | None = None
    mentioned_users: list[dict] = field(default_factory=list)


class Engine:
    def __init__(self, store: Store, models: ModelGateway, settings: Settings):
        self.store, self.models, self.settings = store, models, settings

    def eligible(self, scene: SceneContext) -> list:
        return self.store.eligible_characters(scene.guild_id, scene.space_id)

    def compile(self, scene, purpose, values, history=None, contract='', images=None, max_tokens=None, protected_history_index=None, lore_injections=()):
        snapshot = scene.preset or self.store.active_preset(scene.guild_id)
        role = {'extraction': 'memory', 'summary': 'memory', 'images': 'dialogue'}.get(purpose, purpose)
        budget = min(self.settings.limits['max_input_tokens'], self.settings.profile(role).context_tokens - (max_tokens or self.settings.limits['max_output_tokens']))
        return compile_prompt(snapshot['bundle'], purpose, values, history or [], self.settings.profile(role).prompt_provider, budget, contract=contract, images=images, protected_history_index=protected_history_index, lore_injections=lore_injections)

    def identities(self, scene, nodes=()):
        participants = []
        for row in nodes:
            for part in json.loads(row['context_json']):
                participants.extend([part, *part.get('mentions', [])])
            participants.extend(json.loads(row['mentions_json']))
            if row['author_id'] is not None:
                participants.append({'author_id': row['author_id'], 'author_label': row['author_label']})
        for part in scene.recent:
            participants.extend([part, *part.get('mentions', [])])
        participants.extend(scene.mentioned_users)
        return speaker_context(scene.user_id, scene.user_label, participants)

    async def purpose_text(self, scene, purpose, payload, max_tokens, images=None):
        values = {'payload': payload}
        if purpose == 'summary' and scene.user_id:
            values['speaker_identity'] = self.identities(scene)
        request = self.compile(scene, purpose, values, images=images, max_tokens=max_tokens)
        role = 'dialogue' if purpose == 'images' else 'memory'
        if hasattr(self.models, 'text_compiled'):
            return await self.models.text_compiled(role, request, max_tokens)
        return await self.models.text(role, '\n\n'.join(m.text for m in request.messages if m.role == 'system'), [m for m in request.messages if m.role != 'system'], max_tokens)

    async def purpose_structured(self, scene, purpose, payload, name, schema):
        request = self.compile(scene, purpose, {'payload': payload, 'speaker_identity': self.identities(scene)}, contract='Return only the required structured result. Output schema: ' + json.dumps(schema))
        role = 'director' if purpose == 'director' else 'memory'
        if hasattr(self.models, 'structured_compiled'):
            return await self.models.structured_compiled(role, request, name, schema)
        return await self.models.structured(role, '\n\n'.join(m.text for m in request.messages if m.role == 'system'), [m for m in request.messages if m.role != 'system'], name, schema)

    async def speakers(self, scene: SceneContext) -> list:
        eligible = {row["id"]: row for row in self.eligible(scene)}
        cast = [eligible[ident] for ident in self.store.get_cast(scene.channel_id, scene.parent_channel_id) if ident in eligible]
        if scene.forced_character_id is not None:
            if scene.forced_character_id not in eligible:
                raise ValueError("Character cannot join this space")
            forced = eligible[scene.forced_character_id]
            cast = [forced, *[row for row in cast if row["id"] != forced["id"]]]
        if not cast:
            return []
        # A cheap deterministic gate avoids a model call for ambient messages with no invitation.
        if scene.ambient:
            lower = " ".join([*(part.get("text", "") for part in scene.recent[-2:]), scene.text]).casefold()
            if not any(row["name"].casefold() in lower for row in cast) and not any(
                phrase in lower for phrase in ("what do you think", "join us", "come over", "your turn")
            ):
                return []
        options = [{"id": row["id"], "name": row["name"]} for row in cast]
        prompt = json.dumps({"cast": options, "recent": scene.recent[-6:], "latest": scene.text,
            'latest_author': {'author_id': scene.user_id, 'author_label': scene.user_label},
            "ambient": scene.ambient, "forced": scene.forced_character_id}, ensure_ascii=False)
        try:
            decision = await self.purpose_structured(scene, "director", prompt, "choose_speakers", DIRECTOR_SCHEMA)
            ids = decision.get("speakers")
            if not isinstance(ids, list):
                raise ValueError("Invalid director result")
            ids = list(dict.fromkeys(ident for ident in ids if type(ident) is int and ident in {row["id"] for row in cast}))
            if scene.forced_character_id is not None:
                ids = [scene.forced_character_id, *[ident for ident in ids if ident != scene.forced_character_id]]
            if not ids and not scene.ambient:
                ids = [cast[0]["id"]]
            return [eligible[ident] for ident in ids[:self.settings.limits["max_speakers"]]]
        except Exception as error:
            logging.error("Director failed: %s", error_detail(error))
            return [] if scene.ambient else [cast[0]]

    def record_user(self, scene: SceneContext, stored_text: str | None = None) -> int:
        return self.store.record_node(scene.user_message_id, scene.guild_id, scene.channel_id,
            scene.parent_message_id, scene.user_id, None, stored_text or scene.text, scene.recent,
            author_label=scene.user_label, mentions=scene.mentioned_users)

    async def describe_images(self, scene: SceneContext) -> str:
        if not scene.images:
            return ""
        try:
            return (await self.purpose_text(scene, 'images', scene.text or 'Describe these images', 250, scene.images)).strip()[:1500]
        except Exception as error:
            logging.error("Image description failed: %s", error_detail(error))
            return "Image attached; description unavailable."

    def history_messages(self, nodes, target_character_id):
        history, ids, seen = [], [], {row['message_id'] for row in nodes}
        for row in nodes:
            # Recent human chat was observed with this ancestor, so belongs to
            # this branch even on later turns. Avoid duplicate saved messages.
            for part in json.loads(row['context_json']):
                ident = part.get('message_id')
                if ident in seen:
                    continue
                seen.add(ident)
                history.append(TurnMessage('user', user_line(part.get('author_id'), part.get('author_label', ''), part.get('text', ''))))
                ids.append(ident)
            if row['character_id']:
                speaker = self.store.character_by_id(row['character_id'])
                label = speaker['name'] if speaker else f"Character {row['character_id']}"
                role = 'assistant' if row['character_id'] == target_character_id else 'user'
                history.append(TurnMessage(role, f"{label}: {row['content']}"))
            else:
                history.append(TurnMessage('user', user_line(row['author_id'], row['author_label'], row['content'])))
            ids.append(row['message_id'])
        return history, ids

    async def _history(self, scene: SceneContext, target_character_id: int):
        nodes = self.store.ancestors(scene.user_message_id)
        summary = ""
        summary_index = -1
        for index, node in enumerate(nodes[:-1]):
            if saved := self.store.summary(node["message_id"]):
                summary, summary_index = saved, index
        unsummarized = nodes[summary_index + 1:]
        current_summary = self.store.summary(scene.user_message_id)
        expanded = self.history_messages(unsummarized, target_character_id)[0]
        if current_summary:
            summary = current_summary
            unsummarized = unsummarized[-12:]
        elif len(unsummarized) > 12 and (len(expanded) > 20 or sum(estimate_tokens(message.text) for message in expanded) > 5000):
            older, unsummarized = unsummarized[:-12], unsummarized[-12:]
            payload = '\n'.join(message.text for message in self.history_messages(older, target_character_id)[0])
            if summary:
                payload = f"Previous summary: {summary}\n{payload}"
            try:
                summary = await self.purpose_text(scene, 'summary', payload, 600)
                self.store.save_summary(scene.user_message_id, summary)
            except Exception as error:
                logging.error("Branch summarization failed: %s", error_detail(error))
                summary = (summary + "\n" + payload)[-2000:]
        history, history_ids = self.history_messages(unsummarized, target_character_id)
        if scene.images and history:
            history[-1] = TurnMessage(history[-1].role, history[-1].text, scene.images)
        return summary, history, [row["message_id"] for row in nodes], history_ids

    async def prepare_dialogue(self, scene: SceneContext, character, preceding_lines: list[tuple[str, str]]):
        card = json.loads(character["card"])
        space = self.store.space_by_id(scene.space_id)
        query = "\n".join([*(part.get("text", "") for part in scene.recent), scene.text])
        scopes = lore_scopes(self.store, character["id"], scene.space_id, scene.channel_id, scene.parent_channel_id)
        budget = min(self.settings.limits["max_input_tokens"], self.settings.profile("dialogue").context_tokens - self.settings.limits["max_output_tokens"])
        budget -= len(scene.images) * 1500
        summary, history, message_ids, history_ids = await self._history(scene, character["id"])
        ancestor_ids = set(message_ids)
        parent = self.store.node(scene.parent_message_id)
        def visible(source_id: int | None) -> bool:
            if not parent or not source_id:
                return True
            source = self.store.node(source_id)
            if not source:
                return True
            if source["root_id"] == parent["root_id"]:
                return source_id in ancestor_ids
            return source["created_at"] <= parent["created_at"]

        lore = [item for item in evaluate(self.store, scene.guild_id, scopes, query,
            max(300, min(1800, budget // 4)), character=character,
            messages=[*(part.get("text", "") for part in scene.recent),
                      *(row["content"] for row in self.store.ancestors(scene.user_message_id)[-12:])],
            branch_ids=message_ids, response_id=scene.user_message_id ^ character["id"])
            if (row := self.store.entry_by_key(scene.guild_id, item.entry_key)) and
             (row['promoted_from'] is not None or visible(row['source_message_id']))]
        personal = self.store.personal(scene.guild_id, scene.user_id, character["id"]) if self.store.has_consent(scene.guild_id, scene.user_id) else []
        personal = [row for row in personal if visible(row["source_message_id"])]
        encounters = self.store.encounters(scene.guild_id, character["id"], character["world_id"])
        if scene.space_id != character["world_id"]:
            encounters += self.store.encounters(scene.guild_id, character["id"], scene.space_id)
        encounters = [row for row in encounters if visible(row["source_message_id"])]
        location = f"You are in {'hub' if space['kind']=='hub' else 'world'} {space['name']}."
        def lore_text(items):
            return "\n".join(f"[{item.entry_key}] {item.content}" for item in items)
        eligible = {row['id']: row for row in self.eligible(scene)}
        group_ids = list(dict.fromkeys([*self.store.get_cast(scene.channel_id, scene.parent_channel_id), *([scene.forced_character_id] if scene.forced_character_id else [])]))
        values = {'char': character['name'], 'user': scene.user_label or f'User {scene.user_id}',
            'group': ', '.join(eligible[ident]['name'] for ident in group_ids if ident in eligible),
            'location': location, 'description': card.get('description', ''),
            'personality': card.get('personality', ''), 'scenario': card.get('scenario', ''),
            'opening': card.get('first_mes', ''), 'examples': card.get('mes_example', ''),
            'mesExamples': card.get('mes_example', ''), 'mesExamplesRaw': card.get('mes_example', ''),
            'card_instructions': card.get('system_prompt', ''), 'card_post_history': card.get('post_history_instructions', ''),
            'personal': 'Known personal bonds with the current speaker:\n' + '\n'.join(row['content'] for row in personal[-6:]) if personal else '',
            'encounters': 'Your own past encounters:\n' + '\n'.join(row['content'] for row in encounters[-6:]) if encounters else '',
            'summary': summary, 'preceding': '\n'.join(f'{name}: {line}' for name, line in preceding_lines),
            # Recent chat is carried once in labelled history, so imported
            # presets with only a history marker still receive every speaker.
            'recent': '',
            'speaker_identity': self.identities(scene, self.store.ancestors(scene.user_message_id))}
        guidelines = scene.guidelines if scene.guidelines is not None else self.store.scene_guidelines(scene.guild_id, scene.space_id, scene.parent_channel_id or scene.channel_id)
        values.update({key: row['content'] for key, row in guidelines.items()})
        for position in ('before_char', 'after_char', 'before_examples', 'after_examples'):
            values['lore_' + position] = lore_text([item for item in lore if item.position == position])
        if not history:
            recent = [part for part in scene.recent if part.get('message_id') != scene.user_message_id]
            history = [TurnMessage('user', user_line(part.get('author_id'), part.get('author_label', ''), part.get('text', ''))) for part in recent]
            history.append(TurnMessage('user', user_line(scene.user_id, scene.user_label, scene.text), scene.images))
            history_ids = [part.get('message_id') for part in recent] + [scene.user_message_id]
        slots = self.store.usable_avatars(scene.guild_id, character['id'])
        choices = [{'key': row['slot_key'], 'label': row['label'], 'description': row['description']} for row in slots]
        contract = 'Begin your response with exactly <emotion>SLOT_KEY</emotion> on its own line, then write your dialogue. Choose one available avatar emotion for the whole reply: ' + json.dumps(choices, ensure_ascii=False)
        request = self.compile(scene, 'dialogue', values, history, contract, protected_history_index=history_ids.index(scene.user_message_id), lore_injections=[item for item in lore if item.position == 'in_chat'])
        included = set(request.sources)
        used_history_ids = [history_ids[index] for index in request.history_indices]
        used_lore = [item for item in lore if (item.position != 'in_chat' and 'lore_' + item.position in included) or item.entry_key in request.lore_keys]
        snapshot = scene.preset or self.store.active_preset(scene.guild_id)
        sources = {"lore": [{"id": item.id, "scope": item.scope_kind, "scope_id": item.scope_id,
            "reason": item.reason, "entry_key": item.entry_key, "book_name": item.book_name,
            "position": item.position} for item in used_lore],
            "lore_activations": [item.entry_key for item in used_lore],
            "messages": [ident for ident in used_history_ids if isinstance(ident, int)], "summary": "summary" in included,
            "personal": [row["id"] for row in personal[-6:]] if "personal" in included else [],
            "encounters": [row["id"] for row in encounters[-6:]] if "encounters" in included else [],
            "recent": [part.get("message_id") for part in scene.recent if part.get('message_id') in used_history_ids]}
        sources['preset'] = {'id': snapshot['id'], 'revision': snapshot['revision'], **request.trace()}
        sources['guidelines'] = {key: {field: row[field] for field in ('kind', 'owner_id', 'revision')} for key, row in guidelines.items() if key in included}
        return request, sources

    async def prompt_for(self, scene: SceneContext, character, preceding_lines: list[tuple[str, str]]) -> tuple[str, list[TurnMessage], dict]:
        """Compatibility view; delivery uses prepare_dialogue to retain exact order."""
        request, sources = await self.prepare_dialogue(scene, character, preceding_lines)
        return '\n\n'.join(m.text for m in request.messages if m.role == 'system'), [m for m in request.messages if m.role != 'system'], sources

    def limit(self, key: str) -> int:
        return self.settings.limits.get(key, LIMIT_DEFAULTS[key])

    def recent_transcript(self, nodes, budget: int) -> str:
        """Newest whole nodes whose transcript fits ``budget`` tokens; a lone oversize node keeps its end."""
        history, ids = self.history_messages(nodes, 0)
        node_ids = {row['message_id'] for row in nodes}
        groups, current = [], []
        for message, ident in zip(history, ids):
            current.append(message.text)
            if ident in node_ids:
                groups.append('\n'.join(current))
                current = []
        kept, size = [], -1
        for group in reversed(groups):
            size += len(group.encode('utf-8')) + 1
            if kept and size > 3 * budget:
                break
            kept.append(group)
        return tail_tokens('\n'.join(reversed(kept)), budget)

    async def summarize_scene(self, last_message_id: int, scene: SceneContext | None = None) -> None:
        nodes = self.store.ancestors(last_message_id)
        prior_summary, start = "", 0
        for index, node in enumerate(nodes[:-1]):
            if saved := self.store.summary(node["message_id"]):
                prior_summary, start = saved, index + 1
        if len(nodes) - start < self.limit('summary_every_messages'):
            return
        try:
            transcript = self.recent_transcript(nodes[start:], self.limit('memory_input_tokens'))
            if scene is None:
                node = nodes[-1]
                scene = SceneContext(node['guild_id'], node['channel_id'], None, 0, 0, last_message_id, '', None, [], [])
            summary = await self.purpose_text(scene, 'summary', f'Previous summary: {prior_summary}\nNew exchange:\n{transcript}', self.limit('memory_output_tokens'))
            if summary.strip():
                self.store.save_summary(last_message_id, summary.strip())
        except Exception as error:
            logging.error("Scene summarization failed: %s", error_detail(error))

    async def extract_memories(self, scene: SceneContext, speakers: list, lines: list[tuple[str, str]], root_id: int) -> None:
        if not lines:
            return
        every = self.limit('extraction_every_turns')
        if every > 1 and self.store.count_user_ancestors(scene.guild_id, scene.user_message_id) % every:
            return
        try:
            head = user_line(scene.user_id, scene.user_label, scene.text)
            budget = self.limit('memory_input_tokens')
            rest = budget - estimate_tokens(head + '\n')
            body = '\n'.join(f"{name}: {line}" for name, line in lines)
            text = tail_tokens(head + '\n' + body, budget) if rest <= 0 else head + '\n' + tail_tokens(body, rest)
        except Exception as error:
            logging.error("Memory extraction failed: %s", error_detail(error))
            return
        scope_kind = "thread" if scene.parent_channel_id else "channel"
        scope_id = scene.channel_id
        for index, character in enumerate(speakers[:len(lines)]):
            try:
                result = await self.purpose_structured(scene, "extraction", f"Character: {character['name']}\n{text}", "extract_memory", MEMORY_SCHEMA)
                for fact in result.get("shared_facts", []) if index == 0 else []:
                    if isinstance(fact, str) and 3 < len(fact) <= 300:
                        self.store.add_candidate(scene.guild_id, scope_kind, scope_id, fact.strip(), root_id, scene.user_message_id)
                for fact in result.get("personal_facts", []):
                    if isinstance(fact, str) and 3 < len(fact) <= 300:
                        self.store.add_personal(scene.guild_id, scene.user_id, character["id"], fact.strip(), scene.user_message_id)
                for fact in result.get("encounter_facts", []):
                    if isinstance(fact, str) and 3 < len(fact) <= 300:
                        self.store.add_encounter(scene.guild_id, character["id"], scene.space_id, fact.strip(), scene.user_message_id)
            except Exception as error:
                logging.error("Memory extraction failed: %s", error_detail(error))
