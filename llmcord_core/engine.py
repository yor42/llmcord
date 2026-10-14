from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from . import ambient
from .backend import settings_source
from .config import LIMIT_DEFAULTS, Settings
from .lore import estimate_tokens, lore_scopes
from .world_info import evaluate
from .models import DIRECTOR_SCHEMA, MEMORY_SCHEMA, ImageInput, ModelGateway, TurnMessage
from .store import Store
from .prompts import age_label, compile_prompt, time_values
from .errors import error_detail
from .i18n import UserError, codes
from .usage import log_purpose, mask_for_log
from .identity import speaker_context, user_line
from .happenings import recent_happenings, render_happenings


def tail_tokens(text: str, budget: int) -> str:
    """Keep the end of ``text`` so estimate_tokens stays within ``budget``."""
    data = text.encode('utf-8')
    return text if estimate_tokens(text) <= budget else data[-3 * budget:].decode('utf-8', 'ignore')


def structured_contract(schema) -> str:
    return 'Return only the required structured result. Output schema: ' + json.dumps(schema)


EXTRACTION_CONTRACT = structured_contract(MEMORY_SCHEMA)


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
    speaker_ids: tuple = ()
    allowed_ids: tuple | None = None  # table thread: only these characters may speak (None: unrestricted)
    usage_feature: str | None = None  # overrides the ambient/summon/reply usage label of the turn
    event: bool = False  # a table event turn: the ambient cheap gate never silences it
    created_at: float | None = None  # unix time of the turn when the message id is not a snowflake (table events)
    spectator: bool = False  # FEAT-55: allowed_ids also holds non-seated spectators, who get the public table view
    max_speakers: int | None = None  # FEAT-55: caps the turn's speakers below the configured limit
    rewind: bool = False  # a reply to a non-latest character node
    happenings_now: float | None = None  # MNT-84: fixed at the first dialogue prompt that gets the block (switch on, eligible world); every later speaker gets the same rows
    happenings_rows: list | None = field(default=None, compare=False)
    history_now: float | None = field(default=None, compare=False)  # MNT-85: the turn's clock for message ages, fixed at the first dialogue history
    watch_cache: tuple | None = field(default=None, compare=False)  # FEAT-55: (view or None,) from the turn's first dialogue prompt; a 1-tuple so a None lookup is cached too


class Engine:
    def __init__(self, store: Store, models: ModelGateway, settings):
        self.store, self.models = store, models
        self.source = settings_source(settings)
        self.talk_failed = None  # (guild_id) -> counts a failed table-talk attempt toward the daily limit; set by the bot
        self.table_block = None  # (guild_id, thread_id, character_id) -> per-speaker table text or None; set by the bot
        self.watch_lookup = None  # FEAT-55: (guild_id, channel_id, parent_channel_id) -> public table view dict or None; set by the bot

    @property
    def settings(self) -> Settings:
        return self.source.current()

    @settings.setter
    def settings(self, value: Settings) -> None:
        self.source = settings_source(value)

    def eligible(self, scene: SceneContext) -> list:
        rows = self.store.eligible_characters(scene.guild_id, scene.space_id)
        return rows if scene.allowed_ids is None else [row for row in rows if row['id'] in scene.allowed_ids]

    def eligible_with_archived(self, scene: SceneContext) -> dict:
        """Eligible characters plus the author's own archived favorites (FEAT-26): for lookups of speakers already chosen for this author, never for casts."""
        rows = {row["id"]: row for row in self.eligible(scene)}
        if scene.allowed_ids is None:
            rows.update({row["id"]: row for row in self.store.archived_favorite_rows(scene.guild_id, scene.user_id, scene.space_id)})
        return rows

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
        with log_purpose(purpose):
            return await self.models.text_compiled(role, request, max_tokens)

    async def purpose_structured(self, scene, purpose, payload, name, schema):
        request = self.compile(scene, purpose, {'payload': payload, 'speaker_identity': self.identities(scene)}, contract=structured_contract(schema))
        role = 'director' if purpose == 'director' else 'memory'
        with log_purpose(purpose):
            return await self.models.structured_compiled(role, request, name, schema)

    def favorite_available(self, guild_id, user_id, space_id, channel_id, parent_channel_id) -> bool:
        """Cheap check for the ambient threshold: the author has a favorite in the cast, or one that could step in."""
        if not user_id:
            return False
        if not self.store.favorites(guild_id, user_id):
            return False
        favorites = self.store.eligible_favorites(guild_id, user_id, space_id)
        if not favorites:
            return False
        if self.store.favorites_mode(guild_id, user_id) == 'step_in' or any(f['archived'] for f in favorites):
            return True
        cast = set(self.store.get_cast(channel_id, parent_channel_id))
        return any(f['character_id'] in cast for f in favorites)

    def ambient_follow_up(self, scene: SceneContext) -> bool:
        """A character spoke in this channel within the recent window, with at most one human line since."""
        node = self.store.latest_character_node(scene.channel_id)
        if not node or node["created_at"] < time.time() - self.settings.limits.get("recent_window_seconds", 600):
            return False
        if len(scene.recent) < 2:
            return True
        before = scene.recent[-2].get("message_id")
        return before is not None and node["message_id"] > before

    async def speakers(self, scene: SceneContext) -> list:
        eligible = {row["id"]: row for row in self.eligible(scene)}
        restricted = scene.allowed_ids is not None
        cast = [eligible[ident] for ident in (scene.allowed_ids if restricted else self.store.get_cast(scene.channel_id, scene.parent_channel_id)) if ident in eligible]
        if scene.forced_character_id is not None:
            if scene.forced_character_id not in eligible:
                raise UserError(codes.ENGINE_CANNOT_JOIN)
            forced = eligible[scene.forced_character_id]
            cast = [forced, *[row for row in cast if row["id"] != forced["id"]]]
        archived = {} if restricted else {row['id']: row for row in self.store.archived_favorite_rows(scene.guild_id, scene.user_id, scene.space_id)}
        pool = {**eligible, **archived}
        favorites = [f for f in self.store.favorites(scene.guild_id, scene.user_id) if f['character_id'] in pool] if scene.user_id and not restricted else []
        favorite_ids = {f['character_id'] for f in favorites}
        step_in = bool(favorites) and self.store.favorites_mode(scene.guild_id, scene.user_id) == 'step_in'
        in_cast = {row['id'] for row in cast}
        # An archived favorite always steps in (it can never be in a cast), whatever the member's mode.
        extra = [f['character_id'] for f in favorites if f['character_id'] not in in_cast and (step_in or f['character_id'] in archived)]
        if not cast and not extra:
            return []
        options_rows = list(cast) + [pool[ident] for ident in extra[:self.store.cast_limits(scene.guild_id)['max_favorites']]]
        allowed = {row['id'] for row in options_rows}
        # D37: an ambient message gets a model call only when the EN/KO invitation rule fires.
        ask = None
        if scene.ambient and not scene.event:
            ask = ambient.invite(scene.text, [part.get("text", "") for part in scene.recent[-2:]], [row["name"] for row in options_rows],
                favorite=bool(favorite_ids & allowed), follow_up=self.ambient_follow_up(scene))
            if not ask:
                return []
        options = [{"id": row["id"], "name": row["name"], **({"favorite": True} if row["id"] in favorite_ids else {})} for row in options_rows]
        hint = {'favorites_hint': 'The latest author prefers the options marked favorite: prefer them when they fit, without ignoring other members.'} if favorite_ids & allowed else {}
        prompt = json.dumps({"cast": options, **hint, "recent": scene.recent[-6:], "latest": scene.text,
            'latest_author': {'author_id': scene.user_id, 'author_label': scene.user_label},
            "ambient": scene.ambient and ask is None, "forced": scene.forced_character_id}, ensure_ascii=False)
        try:
            decision = await self.purpose_structured(scene, "director", prompt, "choose_speakers", DIRECTOR_SCHEMA)
            ids = decision.get("speakers")
            if not isinstance(ids, list):
                raise ValueError("Invalid director result")
            ids = list(dict.fromkeys(ident for ident in ids if type(ident) is int and ident in allowed))
            if scene.forced_character_id is not None:
                ids = [scene.forced_character_id, *[ident for ident in ids if ident != scene.forced_character_id]]
            if not ids and ask is not None:
                favorite_rows = [row for row in options_rows if row["id"] in favorite_ids]
                ids = [options_rows[ask.named[0]]["id"] if ask.named else (favorite_rows or options_rows)[0]["id"]]
            elif not ids and not scene.ambient:
                ids = [options_rows[0]["id"]]
            return [pool[ident] for ident in ids[:min(self.settings.limits["max_speakers"], scene.max_speakers or 99)]]
        except Exception as error:
            logging.error("Director failed: %s", error_detail(error))
            if scene.usage_feature == 'game' and self.talk_failed:
                self.talk_failed(scene.guild_id)
            return [] if scene.ambient else [options_rows[0]]

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

    def history_messages(self, nodes, target_character_id, now=None):
        """MNT-85: with ``now``, user-role lines (never the target's own) are prefixed with their relative age."""
        history, ids, seen = [], [], {row['message_id'] for row in nodes}
        def aged(role, text, then):
            if now is None or role != 'user':
                return TurnMessage(role, text)
            label = age_label(now, then)
            return TurnMessage(role, f'{label} {text}' if label else text)
        for row in nodes:
            # Recent human chat was observed with this ancestor, so belongs to
            # this branch even on later turns. Avoid duplicate saved messages.
            for part in json.loads(row['context_json']):
                ident = part.get('message_id')
                if ident in seen:
                    continue
                seen.add(ident)
                history.append(aged('user', user_line(part.get('author_id'), part.get('author_label', ''), part.get('text', '')), ((ident >> 22) + 1420070400000) / 1000 if type(ident) is int else row['created_at']))
                ids.append(ident)
            if row['character_id']:
                speaker = self.store.character_by_id(row['character_id'])
                label = speaker['name'] if speaker else f"Character {row['character_id']}"
                role = 'assistant' if row['character_id'] == target_character_id else 'user'
                history.append(aged(role, f"{label}: {row['content']}", row['created_at']))
            else:
                history.append(aged('user', user_line(row['author_id'], row['author_label'], row['content']), row['created_at']))
            ids.append(row['message_id'])
        return history, ids

    async def _history(self, scene: SceneContext, target_character_id: int):
        nodes = self.store.ancestors(scene.user_message_id)
        summary, summary_index = self.prior_summary(nodes)
        unsummarized = nodes[summary_index + 1:]
        current_summary = self.store.summary(scene.user_message_id)
        expanded = self.history_messages(unsummarized, target_character_id)[0]
        if current_summary:
            summary = current_summary
            unsummarized = unsummarized[-12:]
        elif len(unsummarized) > 12 and (len(expanded) > 20 or sum(estimate_tokens(message.text) for message in expanded) > 5000):
            older, unsummarized = unsummarized[:-12], unsummarized[-12:]
            output = self.limit('memory_output_tokens')
            older_text = '\n'.join(message.text for message in self.history_messages(older, target_character_id)[0])
            try:
                budget = self.memory_input_budget(scene, 'summary', output, extra=estimate_tokens(summary) if summary else 0)
                if budget <= 0:
                    self.no_room('summary')
                    summary = (summary + "\n" + older_text)[-2000:]
                else:
                    payload = tail_tokens(older_text, budget)
                    if summary:
                        payload = f"Previous summary: {summary}\n{payload}"
                    summary = await self.purpose_text(scene, 'summary', payload, output)
                    self.store.save_summary(scene.user_message_id, summary)
            except Exception as error:
                logging.error("Branch summarization failed: %s", error_detail(error))
                summary = (summary + "\n" + older_text)[-2000:]
        if scene.history_now is None:
            # SceneContext is frozen; like happenings_now (MNT-84) the clock is set around it so every speaker of the turn reads the same ages. A rewind uses the replied-to node's time.
            parent = self.store.node(scene.parent_message_id) if scene.rewind else None
            object.__setattr__(scene, 'history_now', parent['created_at'] if parent else time.time())
        history, history_ids = self.history_messages(unsummarized, target_character_id, now=scene.history_now)
        if scene.images and history:
            history[-1] = TurnMessage(history[-1].role, history[-1].text, scene.images)
        return summary, history, [row["message_id"] for row in nodes], history_ids

    def happenings_block(self, scene, character, space, parent):
        """FEAT-54: recent item results in this channel, for a speaker whose world is this space or linked into this hub; a rewind anchors at the replied-to node."""
        if not self.store.cast_sees_item_results(scene.guild_id):
            return ''
        if character['world_id'] != scene.space_id and not (space['kind'] == 'hub' and character['world_id'] in self.store.allowed_worlds(scene.space_id)):
            return ''
        before = parent['created_at'] if scene.rewind and parent else None
        if scene.happenings_rows is None:
            # SceneContext is frozen, but this per-turn cache must be set lazily: only a speaker passing the checks above reads item_uses, once per turn (MNT-84/86).
            if scene.happenings_now is None:
                object.__setattr__(scene, 'happenings_now', before if before is not None else time.time())
            object.__setattr__(scene, 'happenings_rows', recent_happenings(self.store, scene.guild_id, scene.channel_id, now=scene.happenings_now, before=before))
        rows = scene.happenings_rows
        names = {scene.user_id: scene.user_label}
        for part in [*scene.recent, *scene.mentioned_users]:
            if part.get('author_id') is not None:
                names.setdefault(part['author_id'], part.get('author_label', ''))
        return render_happenings(rows, names) or ''

    def watch_block(self, scene, character, space):
        """FEAT-55: public text of an open game table for a non-seated speaker in a normal channel; the lookup is fixed at the turn's first prompt."""
        if not self.watch_lookup or not self.store.cast_watches_games(scene.guild_id) or (scene.allowed_ids is not None and not scene.spectator) or scene.rewind:
            return ''
        if scene.watch_cache is None:
            # SceneContext is frozen; like happenings_rows (MNT-84) the cache is set around it so every speaker of the turn sees one view, even a seated first speaker.
            object.__setattr__(scene, 'watch_cache', (self.watch_lookup(scene.guild_id, scene.channel_id, scene.parent_channel_id),))
        view = scene.watch_cache[0]
        if not view or character['world_id'] != scene.space_id and not (space['kind'] == 'hub' and character['world_id'] in self.store.allowed_worlds(scene.space_id)):
            return ''
        if character['id'] in view.get('seated_character_ids', ()):
            return ''
        return str(view.get('text') or '')

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
        mask_for_log(*(row['content'] for row in personal))
        encounters = self.store.encounters(scene.guild_id, character["id"], character["world_id"])
        if scene.space_id != character["world_id"]:
            encounters += self.store.encounters(scene.guild_id, character["id"], scene.space_id)
        encounters = [row for row in encounters if visible(row["source_message_id"])]
        location = f"You are in {'hub' if space['kind']=='hub' else 'world'} {space['name']}."
        if space['kind'] == 'hub' and space['hub_tone'] == 'off_duty':
            location += (" This hub is an off-duty lounge shared with characters from other worlds. Keep your personality and voice, "
                         "but you are off duty here: chat casually, and don't push your world's plot, quests or conflicts.")
        def lore_text(items):
            return "\n".join(f"[{item.entry_key}] {item.content}" for item in items)
        eligible = self.eligible_with_archived(scene)
        group_ids = list(dict.fromkeys([*self.store.get_cast(scene.channel_id, scene.parent_channel_id), *([scene.forced_character_id] if scene.forced_character_id else []), *scene.speaker_ids]))
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
            'speaker_identity': self.identities(scene, self.store.ancestors(scene.user_message_id)),
            # Table thread only: this speaker's own view of the game (public state + own hand), never saved or sent anywhere else.
            'table_state': (self.table_block(scene.guild_id, scene.channel_id, character['id']) or '') if scene.allowed_ids is not None and self.table_block else ''}
        values['happenings'] = self.happenings_block(scene, character, space, parent)
        values['watching'] = self.watch_block(scene, character, space)
        guidelines = scene.guidelines if scene.guidelines is not None else self.store.scene_guidelines(scene.guild_id, scene.space_id, scene.parent_channel_id or scene.channel_id)
        values.update({key: row['content'] for key, row in guidelines.items()})
        zone, _ = self.store.resolve_timezone(scene.guild_id, scene.user_id)
        values.update(time_values(scene.user_message_id, zone, scene.created_at))
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
        if 'happenings' in included:
            sources['happenings'] = True
        if values['watching'].strip():
            sources['watching'] = True
        sources['preset'] = {'id': snapshot['id'], 'revision': snapshot['revision'], **request.trace()}
        sources['guidelines'] = {key: {field: row[field] for field in ('kind', 'owner_id', 'revision')} for key, row in guidelines.items() if key in included}
        return request, sources

    def limit(self, key: str) -> int:
        return self.settings.limits.get(key, LIMIT_DEFAULTS[key])

    def prior_summary(self, nodes) -> tuple[str, int]:
        """Newest saved summary on ``nodes[:-1]`` and its index (-1 if none); falls back to a bounded lookup above the ancestors window."""
        summary, index = "", -1
        for position, node in enumerate(nodes[:-1]):
            if saved := self.store.summary(node["message_id"]):
                summary, index = saved, position
        if index < 0 and nodes and nodes[0]["parent_id"] is not None:
            summary = self.store.latest_summary_at_or_above(nodes[0]["guild_id"], nodes[0]["parent_id"]) or ""
        return summary, index

    def memory_input_budget(self, scene, purpose: str, max_tokens: int, extra: int = 0, values=None, contract: str = '') -> int:
        """memory_input_tokens clamped to what the memory profile's context leaves after the fixed prompt, output and a margin; <= 0 when nothing fits."""
        s = self.settings
        limit = s.limits.get('memory_input_tokens', LIMIT_DEFAULTS['memory_input_tokens'])
        values = {'payload': '', **({'speaker_identity': self.identities(scene)} if scene.user_id or purpose == 'extraction' else {}), **(values or {})}
        fixed = self.compile(scene, purpose, values, contract=contract, max_tokens=max_tokens).estimated_tokens
        window = min(s.limits['max_input_tokens'], s.profile('memory').context_tokens - max_tokens)
        return min(limit, window - fixed - extra - max(64, window // 20))

    def no_room(self, purpose: str) -> None:
        logging.warning("Skipping memory %s: no input room in memory profile %s", purpose, self.settings.memory)

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
        prior_summary, found = self.prior_summary(nodes)
        start = found + 1
        if len(nodes) - start < self.limit('summary_every_messages'):
            return
        try:
            if scene is None:
                node = nodes[-1]
                scene = SceneContext(node['guild_id'], node['channel_id'], None, 0, 0, last_message_id, '', None, [], [])
            output = self.limit('memory_output_tokens')
            budget = self.memory_input_budget(scene, 'summary', output, extra=estimate_tokens(prior_summary))
            if budget <= 0:
                return self.no_room('summary')
            transcript = self.recent_transcript(nodes[start:], budget)
            summary = await self.purpose_text(scene, 'summary', f'Previous summary: {prior_summary}\nNew exchange:\n{transcript}', output)
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
            budget = self.memory_input_budget(scene, 'extraction', self.limit('max_output_tokens'), contract=EXTRACTION_CONTRACT)
            if budget <= 0:
                return self.no_room('extraction')
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
