from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sqlite3
import time
import re
import hashlib
import itertools
import json
from collections import OrderedDict
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from typing import Literal
from zoneinfo import available_timezones

import discord
from discord import app_commands
from discord.ext import commands

from . import budget, catchup
from .backend import BackendResolver
from .cards import parse_card
from .config import Settings
from .engine import Engine, SceneContext
from .models import ImageInput, ModelGateway, TurnMessage
from .prompts import time_values
from .names import resolve, resolve_space, suggest
from .admin_store import ConflictError, CurrencyError
from .store import Store
from .avatars import emotion_stream
from .errors import ModelConfigError, error_detail, error_stack, reference_id, user_detail
from .usage import capture_usage, stage_label, log_attribution, log_purpose, log_scope, mask_for_log, reply_footer
from .identity import discord_identity, message_context
from .favorites_discord import register_favorites_commands
from .game_talk import TalkBudget
from .games_discord import GameButton, GameTables, register_game_commands
from .games_spoken_discord import SpokenButton, SpokenMoves
from .games_spoken_doubt import SpokenDoubtButton, SpokenDoubts
from .games_doubt_discord import DoubtButton, DoubtTables, register_doubt_commands
from .games_holdem_discord import HoldemButton, HoldemTables
from .items_discord import register_items_commands
from llmcord_core import i18n
from llmcord_core.i18n import N_, UserError, codes
from llmcord_core.i18n import discord_locale
from llmcord_core.i18n.discord_locale import C_, LlmcordTranslator, channel_t, invoker_t

AVATAR_ASSET_CHECK_TTL = 600
CACHE_LIMIT = 512
UNKNOWN_WEBHOOK = 10015
MEMORY_WAIT_SECONDS = 15
STAGE_SPEAKER_SELECTION, STAGE_IMAGE_DESCRIPTION, STAGE_DIALOGUE = 'speaker selection', 'image description', 'dialogue generation'
PROVIDER_STAGES = frozenset({STAGE_SPEAKER_SELECTION, STAGE_IMAGE_DESCRIPTION, STAGE_DIALOGUE})
MEMORY_CLOSE_SECONDS = 5
NO_CHARACTER_NOTE_SECONDS = 15


async def recent_human_lines(channel, limits: dict, cutoff, floor: float, before=None) -> list:
    kwargs = {"before": before} if before is not None else {}
    recent = []
    async for old in channel.history(limit=limits["recent_messages"] * 5, **kwargs):
        if old.created_at < cutoff or old.created_at.timestamp() < floor:
            break
        if old.author.bot or old.webhook_id or not old.content:
            continue
        recent.append(message_context(old))
        if len(recent) >= limits["recent_messages"]:
            break
    recent.reverse()
    return recent


def _kind_word(t, kind):
    return t.p('space kind', 'world') if kind == 'world' else t.p('space kind', 'hub') if kind == 'hub' else kind


def _space_ref(t, kind, name):
    kind_word = _kind_word(t, kind)
    return t.p('space ref', '{kind} {name}', kind=kind_word, name=name)


def split_discord(text: str, limit: int = 1900) -> list[str]:
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut].strip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks or ["(no response)"]


class LruDict(OrderedDict):
    """Dict that drops its least recently used entry past ``cap``."""

    def __init__(self, cap=CACHE_LIMIT):
        super().__init__()
        self.cap = cap

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key, default=None):
        return self[key] if key in self else default

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.cap:
            self.popitem(last=False)


_EVENT_COUNTER = itertools.count()


def _event_id() -> int:
    """A synthetic node id for a table event: negative (never a real Discord id) and snowflake-shaped so it stays unique across restarts. Never decoded for its time (SceneContext.created_at)."""
    return -(((int(time.time() * 1000) - 1420070400000) << 22) | (next(_EVENT_COUNTER) & 0x3FFFFF))


class LockPool(dict):
    """Per-key asyncio locks; past ``cap`` the oldest idle locks are dropped, never one held or awaited."""

    def __init__(self, cap=CACHE_LIMIT):
        super().__init__()
        self.cap = cap

    def lock(self, key):
        lock = self.pop(key, None) or asyncio.Lock()
        self[key] = lock
        for old in list(self):
            if len(self) <= self.cap:
                break
            idle = self[old]
            # private asyncio attr: a just-released lock with a woken waiter is unlocked; guarded by test_lock_with_waiters_is_never_evicted
            if old != key and not idle.locked() and not getattr(idle, '_waiters', None):
                del self[old]
        return lock


def _webhook_gone(error):
    """True when Discord says the cached webhook is unusable (Unknown Webhook, or an invalid token)."""
    if isinstance(error, discord.NotFound):
        return error.code == UNKNOWN_WEBHOOK
    return isinstance(error, discord.HTTPException) and error.status == 401


class SkitBot(commands.Bot):
    def __init__(self, settings: Settings):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=commands.when_mentioned, intents=intents)
        self.base_settings = settings
        self.store = Store(settings.database_path)
        self.backend = BackendResolver(self.store, settings)
        self.models = ModelGateway(self.backend, usage_sink=self.store.record_model_usage,
                                  budget_gate=self._budget_check, log_sink=self._log_model_call)
        self.engine = Engine(self.store, self.models, self.backend)
        self.channel_locks = LockPool()
        self.catchup_used: dict[tuple[int, int, int], float] = {}
        self.webhook_locks = LockPool()
        self.webhooks = LruDict()
        self.webhook_defaults = LruDict()
        self.checked_avatar_assets = LruDict()
        self.cleanup_task: asyncio.Task | None = None
        self.tree.allowed_contexts = app_commands.AppCommandContext(guild=True)
        self.memory_tasks: dict[int, asyncio.Task] = {}
        self.note_tasks: set[asyncio.Task] = set()
        self.talk_budget = TalkBudget(lambda: self.store, self._hard_cap_reached)
        self.games = GameTables(self)
        self.spoken = SpokenMoves(self)
        self.spoken_doubt = SpokenDoubts(self)
        self.doubt = DoubtTables(self)
        self.holdem = HoldemTables(self)
        self.engine.table_block = self._table_block
        self.engine.talk_failed = self._note_talk_failure
        register_commands(self)

    async def setup_hook(self):
        await self.tree.set_translator(LlmcordTranslator())
        discord_locale.set_server_language_source(self.store.server_language)
        if self.base_settings.development_guild_id:
            guild = discord.Object(id=self.base_settings.development_guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()
        await self.games.recover()
        await self.doubt.recover()
        await self.holdem.recover()
        self.add_dynamic_items(GameButton, SpokenButton, SpokenDoubtButton, DoubtButton, HoldemButton)
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def on_ready(self):
        logging.info("Discord connected as %s; servers=%d; command scope=%s",
                     self.user, len(self.guilds), self.base_settings.development_guild_id or "global")

    async def _cleanup_loop(self):
        while True:
            try:
                self.store.expire_history(self.base_settings.history_retention_days)
            except Exception:
                logging.exception('History cleanup failed')
            try:
                self.store.expire_monitoring()
            except Exception:
                logging.exception('Monitoring cleanup failed')
            await asyncio.sleep(86400)

    async def close(self):
        if self.cleanup_task:
            self.cleanup_task.cancel()
        try:
            await self.spoken.close()
        except Exception:
            logging.exception('Closing the spoken-move checks failed')
        try:
            await self.spoken_doubt.close()
        except Exception:
            logging.exception('Closing the spoken-doubt checks failed')
        try:
            await self.games.close()
        except Exception:
            logging.exception('Closing the game timers failed')
        try:
            await self.doubt.close()
        except Exception:
            logging.exception('Closing the I Doubt It timers failed')
        try:
            await self.holdem.close()
        except Exception:
            logging.exception("Closing the Texas Hold'em timers failed")
        try:
            pending = [task for task in self.memory_tasks.values() if not task.done()]
            if pending:
                _, pending = await asyncio.wait(pending, timeout=MEMORY_CLOSE_SECONDS)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            notes = [task for task in self.note_tasks if not task.done()]
            for task in notes:
                task.cancel()
            await asyncio.gather(*notes, return_exceptions=True)
            await self.models.close()
        finally:
            try:
                self.store.close()
            finally:
                await super().close()

    def location(self, channel) -> tuple[int | None, object | None]:
        parent_id = channel.parent_id if isinstance(channel, discord.Thread) else None
        return parent_id, self.store.binding(channel.id, parent_id)

    async def on_message(self, message: discord.Message):
        if message.guild is None or message.author.bot or message.webhook_id:
            return
        try:
            await self.doubt.note_thread_message(message)
        except Exception:
            logging.exception("Counting a table thread message failed")
        try:
            self.spoken.note_message(message)  # only schedules a background check; the message carries on to the characters
        except Exception:
            logging.exception("Scheduling a spoken-move check failed")
        try:
            self.spoken_doubt.note_message(message)  # same: a background check at most
        except Exception:
            logging.exception("Scheduling a spoken-doubt check failed")
        parent_id, binding = self.location(message.channel)
        if not binding:
            return
        game, table = self._table_game(message.guild.id, message.channel.id)
        if table is not None and not self._table_talk_open(message.guild.id, table, game):
            return  # the table's characters are silent (talk off, daily limit, spending cap); members still chat
        reference_id = message.reference.message_id if message.reference else None
        referenced = self.store.node(reference_id)
        if referenced and (referenced["guild_id"] != message.guild.id or referenced["channel_id"] != message.channel.id):
            referenced = None
        explicit_reply = bool(referenced and referenced["character_id"])
        mentioned = self.user in message.mentions if self.user else False
        explicit = explicit_reply or mentioned
        if not explicit and not binding["ambient"]:
            return
        if not explicit:
            state = self._budget_check()
            if state and state.hard_reached:
                return
            async with self.channel_locks.lock(message.channel.id):
                self.store.count_ambient_message(message.channel.id)
                count, last = self.store.ambient_state(message.channel.id)
                if time.time() - last < self.base_settings.limits["ambient_cooldown_seconds"]:
                    return
                if count < 2 and not (count == 1 and self.engine.favorite_available(message.guild.id, message.author.id, binding["space_id"], message.channel.id, parent_id)):
                    return
                # A silent director decision still consumes this ambient opportunity.
                self.store.mark_ambient_response(message.channel.id)
        recent = []
        latest = self.store.latest_character_node(message.channel.id)
        # A rewind sees only the ancestry at its target, never newer channel chat.
        is_rewind = bool(explicit_reply and latest and reference_id != latest["message_id"])
        if not is_rewind:
            cutoff = message.created_at - timedelta(seconds=self.base_settings.limits["recent_window_seconds"])
            reset_at = self.store.scene_reset_at(message.channel.id)
            after_parent = referenced["created_at"] if explicit_reply and referenced else 0.0
            recent = await recent_human_lines(message.channel, self.base_settings.limits, cutoff, max(reset_at, after_parent), before=message)
        text = message.content
        if self.user:
            text = text.replace(self.user.mention, "", 1).strip()
        images = []
        current = self.settings if message.attachments else None
        try:
            for attachment in message.attachments:
                if attachment.size > self.base_settings.limits["max_attachment_bytes"]:
                    raise UserError(codes.ATTACHMENT_TOO_BIG)
                if attachment.content_type and attachment.content_type.startswith("image/"):
                    if len(images) >= self.base_settings.limits["max_images"]:
                        raise UserError(codes.TOO_MANY_IMAGES)
                    if not current.profile("dialogue").supports_images:
                        raise UserError(codes.MODEL_NO_IMAGES)
                    images.append(ImageInput(attachment.content_type, await attachment.read()))
                elif attachment.content_type and attachment.content_type.startswith("text/"):
                    text += "\n" + (await attachment.read()).decode("utf-8", errors="replace")[:12000]
                else:
                    raise UserError(codes.ATTACHMENT_TYPE)
        except ValueError as error:
            await message.reply(channel_t(message.guild).error(error), mention_author=False)
            return
        scene = SceneContext(message.guild.id, message.channel.id, parent_id,
            binding["space_id"], message.author.id, message.id, text,
            reference_id if referenced else None, recent, images, ambient=not explicit, user_label=message.author.display_name,
            mentioned_users=[discord_identity(user) for user in message.mentions if not user.bot], **self._table_fields(table))
        async with self.channel_locks.lock(message.channel.id):
            await self.run_scene(scene, message.channel)

    def _table_game(self, guild_id, thread_id):
        """(the game's tables object, its table talk context) for the table that owns this thread, else (None, None)."""
        for game in (self.doubt, self.holdem):
            table = game.table_talk_context(guild_id, thread_id)
            if table is not None:
                return game, table
        return None, None

    def _table_block(self, guild_id, thread_id, character_id):
        block = self.doubt.table_block(guild_id, thread_id, character_id)
        return block if block is not None else self.holdem.table_block(guild_id, thread_id, character_id)

    def _note_talk_failure(self, guild_id) -> None:
        """A failed table talk attempt counts toward the guild's daily table-talk limit, which every table game shares."""
        self.talk_budget.note_failure(guild_id)

    def _table_talk_open(self, guild_id, table, game) -> bool:
        """May the characters of this table thread speak: someone is seated, table talk is on, the daily limit and the spending cap allow it."""
        # The limit is checked once per turn: a turn that passes with one call left may use up to 1 + max_speakers calls
        # (the director plus one dialogue per speaker). The overshoot is bounded and accepted.
        if not table['seated_character_ids'] or not game.talk_allowed(guild_id):
            return False
        state = self._budget_check()
        return not (state and state.hard_reached)

    @staticmethod
    def _table_fields(table) -> dict:
        return {} if table is None else {'allowed_ids': tuple(table['seated_character_ids']), 'usage_feature': 'game'}

    async def run_table_event(self, thread, text, speaker_hint=None):
        """One ambient character turn in a table thread after a public game event (a reveal, a seat down to one card, the game's end).
        Never waits: skipped when the thread is busy or the table's characters may not talk. The event is saved as a user node
        by the table (a synthetic negative message id, author = the bot) so it sits in the scene history like a narration line."""
        guild = getattr(thread, 'guild', None)
        if guild is None or not isinstance(thread, discord.Thread):
            return
        game, table = self._table_game(guild.id, thread.id)
        parent_id, binding = self.location(thread)
        if table is None or not binding or not self._table_talk_open(guild.id, table, game):
            return
        lock = self.channel_locks.lock(thread.id)
        if lock.locked():
            return
        async with lock:
            recent = []
            try:
                cutoff = discord.utils.utcnow() - timedelta(seconds=self.base_settings.limits["recent_window_seconds"])
                recent = await recent_human_lines(thread, self.base_settings.limits, cutoff, self.store.scene_reset_at(thread.id))
            except Exception as error:
                logging.warning('Table event could not read recent chat: %s', error_detail(error))
            scene = SceneContext(guild.id, thread.id, parent_id, binding["space_id"], self.user.id if self.user else 0, _event_id(), text,
                None, recent, [], ambient=True, user_label='Table events', event=True, created_at=time.time(), **self._table_fields(table))
            await self.run_scene(scene, thread)

    @staticmethod
    def _webhook_key(channel, character):
        parent_channel = channel.parent if isinstance(channel, discord.Thread) else channel
        return (getattr(parent_channel, 'id', None), character['id'])

    def _forget_webhook(self, key):
        self.webhooks.pop(key, None)
        self.webhook_defaults.pop(key, None)

    async def _webhook(self, channel, character):
        parent_channel = channel.parent if isinstance(channel, discord.Thread) else channel
        if not isinstance(parent_channel, discord.TextChannel):
            raise ValueError("Character webhooks require a text channel or its thread")
        key = (parent_channel.id, character['id'])
        async with self.webhook_locks.lock(key):
            return await self._webhook_locked(parent_channel, character, key)

    async def _webhook_identity(self, webhook, character, key, identity):
        if self.webhook_defaults.get(key) != identity:
            try:
                webhook = await webhook.edit(name=character['name'][:80], avatar=character['avatar'])
            except discord.HTTPException as error:
                if not _webhook_gone(error):
                    raise
                self._forget_webhook(key)
                return None
            self.webhook_defaults[key] = identity
        self.webhooks[key] = webhook
        return webhook

    async def _webhook_locked(self, parent_channel, character, key):
        identity = (character['name'], hashlib.sha256(character['avatar'] or b'').hexdigest())
        if key in self.webhooks:
            webhook = await self._webhook_identity(self.webhooks[key], character, key, identity)
            if webhook:
                return webhook
        webhook_id = self.store.webhook_id(parent_channel.id, character["id"])
        if webhook_id:
            for webhook in await parent_channel.webhooks():
                if webhook.id == webhook_id and webhook.token:
                    webhook = await self._webhook_identity(webhook, character, key, identity)
                    if webhook:
                        return webhook
                    break
        try:
            webhook = await parent_channel.create_webhook(name=character["name"][:80], avatar=character["avatar"], reason="llmcord character")
        except discord.Forbidden as error:
            raise ValueError("I need Manage Webhooks permission in this channel") from error
        except discord.HTTPException as error:
            raise ValueError("Could not create a character webhook: " + error_detail(error)) from error
        self.store.save_webhook_id(parent_channel.id, character["id"], webhook.id)
        self.webhooks[key] = webhook
        self.webhook_defaults[key] = identity
        return webhook

    async def resolve_avatar(self, selected):
        """Prefer the emotion image; None uses the webhook's static fallback."""
        ident = selected['asset_id']
        if ident:
            checked = self.checked_avatar_assets.get(ident)
            if checked is None or time.monotonic() - checked[1] > AVATAR_ASSET_CHECK_TTL:
                asset = self.store.one('SELECT * FROM avatar_assets WHERE id=?', (ident,))
                try:
                    if asset:
                        channel = self.get_channel(asset['channel_id']) or await self.fetch_channel(asset['channel_id'])
                        await channel.fetch_message(asset['message_id'])
                    self.checked_avatar_assets[ident] = (bool(asset), time.monotonic())
                except (discord.DiscordException, AttributeError):
                    self.checked_avatar_assets[ident] = (False, time.monotonic())
            if not self.checked_avatar_assets[ident][0]:
                return {**selected, 'url': None, 'asset_id': None}
        return selected

    def _budget_check(self, now=None):
        """Current spending state, DMing operators once per period; None (fail open) if it cannot be read."""
        try:
            now = time.time() if now is None else now
            state = budget.state(self.store, now)
            if state.hard_reached:
                kinds = [('soft', False), ('hard', True)]
            elif state.soft_reached:
                kinds = [('soft', True)]
            else:
                kinds = []
            for operator_id in sorted(self.base_settings.operator_ids):
                for kind, send in kinds:
                    if self.store.claim_notice(state.period, kind, operator_id, now) and send:
                        self._send_operator_dm(operator_id, _operator_text(kind, state, t=channel_t(None)))
            return state
        except Exception as error:
            logging.error('Budget check failed: %s\n%s', error_detail(error), error_stack(error))
            return None

    def _hard_cap_reached(self) -> bool:
        """Read-only spending check for the game talk budget: no operator notices, fails open."""
        try:
            return bool(budget.state(self.store, time.time()).hard_reached)
        except Exception as error:
            logging.warning('Game talk budget check failed open: %s', error_detail(error))
            return False

    def _send_operator_dm(self, operator_id, text):
        async def send():
            try:
                user = self.get_user(operator_id) or await self.fetch_user(operator_id)
                await user.send(text, allowed_mentions=discord.AllowedMentions.none())
            except discord.DiscordException as error:
                logging.warning('Operator spending DM to %s failed: %s', operator_id, error_detail(error))
            except Exception as error:
                logging.error('Operator spending DM to %s failed: %s', operator_id, error_detail(error))

        task = asyncio.create_task(send())
        self.note_tasks.add(task)
        task.add_done_callback(self.note_tasks.discard)

    @property
    def settings(self) -> Settings:
        return self.backend.current()

    async def run_scene(self, scene: SceneContext, channel, interaction=None):
        state = self._budget_check()
        if state and state.hard_reached:
            if interaction is not None:
                notice = _hard_cap_notice(state, t=invoker_t(interaction))
                try:
                    target = interaction.followup if interaction.response.is_done() else interaction.response
                    if target is interaction.followup:
                        await target.send(notice, ephemeral=True)
                    else:
                        await target.send_message(notice, ephemeral=True)
                except discord.DiscordException as error:
                    logging.warning('Spending limit notice failed: %s', error_detail(error))
            elif not scene.ambient and state.channel_notice \
                    and self.store.claim_notice(state.period, 'channel', scene.channel_id, time.time(), interval=3600):
                try:
                    await channel.send(_hard_cap_notice(state, t=self._channel_t(channel)), allowed_mentions=discord.AllowedMentions.none())
                except discord.DiscordException as error:
                    logging.warning('Spending limit channel notice failed: %s', error_detail(error))
            return
        previous = self.memory_tasks.get(channel.id)
        if previous and not previous.done():
            _, pending = await asyncio.wait({previous}, timeout=MEMORY_WAIT_SECONDS)
            if pending:
                logging.warning('Memory task for channel %s still running after %ss; continuing without it',
                                channel.id, MEMORY_WAIT_SECONDS)
        with self.backend.pin(), budget.admit(), log_scope(scene.user_message_id), capture_usage(scene.guild_id, scene.channel_id, scene.usage_feature or ('ambient' if scene.ambient else 'summon' if scene.forced_character_id is not None else 'reply')):
            return await self._run_scene(scene, channel, interaction)

    def _log_entry(self, guild_id, channel_id, message_id, facts, **fields):
        # A log failure never touches the turn; nothing is rendered unless the server's turn log is on.
        try:
            if guild_id is None or not self.store.turn_log_settings(guild_id)['enabled']:
                return
            secrets = [os.environ.get(p.api_key_env, '') for p in self.settings.profiles.values() if p.api_key_env]
            fields = {key: value() if key in ('request_text', 'error_detail') and callable(value) else value for key, value in fields.items()}
            self.store.add_turn_log(guild_id, channel_id=channel_id, message_id=message_id, masked_facts=facts,
                                    secret_values=[v for v in secrets if v], **fields)
        except Exception as error:
            logging.warning('Turn log entry could not be saved: %s', type(error).__name__)

    def _log_model_call(self, entry):
        where = log_attribution()
        self._log_entry(where['guild_id'], where['channel_id'], where['message_id'], where['facts'],
                        stage=f"{where['feature']} · {where['purpose'] or entry['role']}", profile=entry['profile'], model=entry['model'],
                        status=entry['status'], error_detail=entry['error_detail'], request_text=entry['render_request'],
                        response_text=entry['response_text'], input_tokens=entry['input_tokens'], output_tokens=entry['output_tokens'])

    def _log_failure(self, error, ref, stage, *, guild_id=None, channel_id=None, feature=None, message_id=None, facts=()):
        where = log_attribution()
        self._log_entry(guild_id if guild_id is not None else where['guild_id'], channel_id if channel_id is not None else where['channel_id'],
                        message_id if message_id is not None else where['message_id'], [*where['facts'], *facts],
                        stage=f"{feature or where['feature']} · failed at {stage}", status='error', reference_id=ref,
                        error_detail=lambda: error_detail(error) + '\n' + error_stack(error))

    def _schedule_memory(self, channel_id, scene, completed, preceding, root_id, parent_message_id):
        previous = self.memory_tasks.get(channel_id)

        async def work():
            if previous and not previous.done():
                try:
                    await asyncio.wait({previous})
                except asyncio.CancelledError:
                    previous.cancel()
                    await asyncio.gather(previous, return_exceptions=True)
                    raise
            with capture_usage(scene.guild_id, scene.channel_id, 'memory'):
                try:
                    await self.engine.extract_memories(scene, completed, preceding, root_id)
                    await self.engine.summarize_scene(parent_message_id, scene)
                except Exception as error:
                    logging.error('Memory update failed in channel %s: %s\n%s',
                                  channel_id, error_detail(error), error_stack(error))

        def forget(done):
            if self.memory_tasks.get(channel_id) is done:
                del self.memory_tasks[channel_id]

        task = asyncio.create_task(work())
        self.memory_tasks[channel_id] = task
        task.add_done_callback(forget)

    def _delete_later(self, message):
        async def delete():
            await asyncio.sleep(NO_CHARACTER_NOTE_SECONDS)
            try:
                await message.delete()
            except discord.DiscordException as error:
                logging.warning('No-character hint cleanup failed: %s', error_detail(error))

        task = asyncio.create_task(delete())
        self.note_tasks.add(task)
        task.add_done_callback(self.note_tasks.discard)

    @staticmethod
    def _channel_t(channel):
        return channel_t(getattr(channel, 'guild', None))

    async def _run_scene(self, scene: SceneContext, channel, interaction=None):
        model_label = discord.utils.escape_markdown(self.settings.profile('dialogue').model[:120])
        progress = _SceneProgress(self.store, self.settings, scene, channel, model_label)
        stage = _Stage(STAGE_SPEAKER_SELECTION)
        try:
            if not scene.ambient:
                await progress.update(progress.status(self._channel_t(channel)('Generating a reply…')))
            scene = replace(scene, preset=scene.preset or self.store.active_preset(scene.guild_id),
                guidelines=scene.guidelines if scene.guidelines is not None else self.store.scene_guidelines(scene.guild_id, scene.space_id, scene.parent_channel_id or scene.channel_id))
            speakers = await self.engine.speakers(scene)
            scene = replace(scene, speaker_ids=tuple(row['id'] for row in speakers))
            if not speakers:
                if not scene.ambient and await progress.update(self._channel_t(channel)("No active character is available here. Use /cast set or /summon.")):
                    self._delete_later(progress.message)
                    progress.message = None
                return
            if scene.ambient:
                await progress.update(progress.status(self._channel_t(channel)('Generating a reply…')))
            stage.name = STAGE_IMAGE_DESCRIPTION
            image_description = await self.engine.describe_images(scene)
            stored_text = scene.text + (f"\n[Image description: {image_description}]" if image_description else "")
            stage.name = 'saving scene input'
            root_id = self.engine.record_user(scene, stored_text)
            preceding: list[tuple[str, str]] = []
            parent_message_id = scene.user_message_id
            completed = []
            for character in speakers:
                name = discord.utils.escape_markdown(character['name'])
                await progress.update(progress.status(self._channel_t(channel)('**{name}** is preparing a reply…', name=name)))
                stage.name = 'preparing character prompt'
                request, sources = await self.engine.prepare_dialogue(scene, character, preceding)
                line, outgoing, chunks, emotion, chosen_avatar, usage = await self._stream_speaker(
                    scene, channel, character, request, progress=progress, stage=stage, name=name, model_label=model_label)
                parent_message_id = self._record_reply(scene, character, sources=sources, outgoing=outgoing, chunks=chunks,
                    emotion=emotion, chosen_avatar=chosen_avatar, usage=usage, parent_message_id=parent_message_id, stage=stage)
                preceding.append((character["name"], line))
                completed.append(character)
            await progress.clear(self._channel_t(channel)('Reply sent.'))
            if scene.ambient:
                self.store.mark_ambient_response(scene.channel_id)
            if scene.allowed_ids is None:  # table-thread talk is neither remembered nor summarized
                self._schedule_memory(channel.id, scene, completed, preceding, root_id, parent_message_id)
        except Exception as error:
            await self._report_scene_failure(error, stage.name, progress, channel, interaction)
        finally:
            await progress.clear()

    async def _stream_speaker(self, scene, channel, character, request, *, progress, stage, name, model_label):
        """Post one character's reply through its webhook, streaming into a placeholder; returns
        (line, outgoing messages, chunks, emotion, chosen avatar, usage)."""
        stage.name = 'webhook setup'
        webhook = await self._webhook(channel, character)
        thread_options = {'thread': channel} if isinstance(channel, discord.Thread) else {}
        slots = {row['slot_key']: row for row in self.store.usable_avatars(scene.guild_id, character['id'])}
        emotion, chosen_avatar, placeholder = 'neutral', slots['neutral'], None
        pieces, last_edit = [], 0.0

        async def send_placeholder():
            return await webhook.send('…', **thread_options, wait=True, silent=True,
                username=character['name'], avatar_url=chosen_avatar['url'], allowed_mentions=discord.AllowedMentions.none())

        try:
            stage.name = STAGE_DIALOGUE
            await progress.update(progress.status(self._channel_t(channel)('Streaming **{name}**', name=name)))
            with capture_usage() as usage_records:
                async with contextlib.aclosing(self.models.stream_compiled('dialogue', request)) as stream:
                    async for event in emotion_stream(stream, slots):
                        if event.emotion is not None:
                            emotion = event.emotion
                            stage.name = 'avatar lookup'
                            chosen_avatar = await self.resolve_avatar(slots.get(emotion, slots['neutral']))
                            stage.name = 'webhook delivery'
                            try:
                                placeholder = await send_placeholder()
                            except discord.HTTPException as error:
                                if not _webhook_gone(error):
                                    raise
                                self._forget_webhook(self._webhook_key(channel, character))
                                stage.name = 'webhook setup'
                                webhook = await self._webhook(channel, character)
                                stage.name = 'webhook delivery'
                                placeholder = await send_placeholder()
                            stage.name = STAGE_DIALOGUE
                            continue
                        pieces.append(event.text)
                        current = "".join(pieces)
                        if time.monotonic() - last_edit > 1.2 and len(current) < 1800:
                            stage.name = 'webhook delivery'
                            await placeholder.edit(content=current + " ▌", allowed_mentions=discord.AllowedMentions.none())
                            last_edit = time.monotonic()
                            stage.name = STAGE_DIALOGUE
            line = re.sub(r'<emotion>[^\n]*?</emotion>\s*', '', "".join(pieces)).strip()
            usage = usage_records[-1] if usage_records else None
            footer = reply_footer(model_label, usage, t=self._channel_t(channel)) if self.store.usage_footer_enabled(scene.guild_id) else ''
            chunks = split_discord(line, limit=min(1900, 2000 - len(footer)))
            stage.name = 'webhook delivery'
            await placeholder.edit(content=chunks[0] + footer, allowed_mentions=discord.AllowedMentions.none())
        except Exception as error:
            if _webhook_gone(error):
                self._forget_webhook(self._webhook_key(channel, character))
            if placeholder:
                try:
                    await placeholder.delete()
                except Exception as cleanup_error:
                    logging.warning('Failed response cleanup: %s', error_detail(cleanup_error))
            raise
        outgoing = [placeholder]
        try:
            for chunk in chunks[1:]:
                outgoing.append(await webhook.send(chunk + footer, **thread_options, wait=True, silent=True,
                    username=character['name'], avatar_url=chosen_avatar['url'],
                    allowed_mentions=discord.AllowedMentions.none()))
        except discord.HTTPException as error:
            if _webhook_gone(error):
                self._forget_webhook(self._webhook_key(channel, character))
            raise
        return line, outgoing, chunks, emotion, chosen_avatar, usage

    def _record_reply(self, scene, character, *, sources, outgoing, chunks, emotion, chosen_avatar, usage,
                      parent_message_id, stage):
        for posted, chunk in zip(outgoing, chunks):
            stage.name = 'saving character reply'
            self.store.record_node(posted.id, scene.guild_id, scene.channel_id,
                parent_message_id, None, character["id"], chunk,
                sources=sources["messages"])
            self.store.save_trace(posted.id, {"character_id": character["id"], 'emotion': emotion,
                'usage': usage.as_dict() if usage else None,
                'avatar_source': 'emotion' if chosen_avatar['asset_id'] else 'static' if character['avatar'] else 'default',
                'avatar_asset_id': chosen_avatar['asset_id'], **sources})
            self.store.save_lore_activations(posted.id, sources.get("lore_activations", []))
            parent_message_id = posted.id
        return parent_message_id

    def _failure_pointer(self, guild_id, fallback=None, *, t=None) -> str:
        t = t or i18n.english()
        # UI-27: point admins at Monitoring → Log only when the turn log is on (so the failure entry exists).
        try:
            if guild_id is not None and self.store.turn_log_settings(guild_id)['enabled']:
                return t("An admin can find details in the dashboard under Monitoring → Log.")
        except Exception:
            pass
        return fallback if fallback is not None else t("An admin can find details in the bot log.")

    async def _report_scene_failure(self, error, stage, progress, channel, interaction):
        ref = reference_id()
        logging.error('Scene failed during %s [ref %s]: %s\n%s', stage, ref, error_detail(error), error_stack(error))
        self._log_failure(error, ref, stage)
        guild_id = log_attribution()['guild_id']
        if log_attribution()['feature'] == 'game':
            self._note_talk_failure(guild_id)
        failure = self._channel_t(channel)("The character couldn't reply (ref {ref}). {pointer}", ref=ref,
                                           pointer=self._failure_pointer(guild_id, t=self._channel_t(channel)))
        try:
            if progress.message:
                try:
                    await progress.message.edit(content=failure, allowed_mentions=discord.AllowedMentions.none())
                    progress.message = None
                except discord.DiscordException:
                    await channel.send(failure, allowed_mentions=discord.AllowedMentions.none())
            else:
                await channel.send(failure, allowed_mentions=discord.AllowedMentions.none())
        except discord.DiscordException as post_error:
            logging.warning('Public failure notice failed [ref %s]: %s', ref, error_detail(post_error))
        if interaction:
            t = invoker_t(interaction)
            detail = (_member_detail(error, ref, t=t) if stage in PROVIDER_STAGES
                      else t('internal error. {pointer}', pointer=self._failure_pointer(guild_id, t=t)))
            try:
                await interaction.followup.send(t("Your turn failed during {stage} (ref {ref}): {detail}", stage=_stage_label(stage, t), ref=ref, detail=detail)[:1900],
                    ephemeral=True, allowed_mentions=discord.AllowedMentions.none())
            except Exception as notify_error:
                logging.warning('Private failure notice failed [ref %s]: %s', ref, error_detail(notify_error))


def _stage_label(stage, t):
    return stage_label(stage, t)


def _member_detail(error, ref, *, t=None):
    t = t or i18n.english()
    if isinstance(error, ModelConfigError):
        return t("The bot's model setup needs attention. An admin can look it up with the reference ID.")
    return user_detail(error)


class _Stage:
    """Mutable label of the turn step in progress; the failure report reads it."""
    def __init__(self, name):
        self.name = name


class _SceneProgress:
    """The public 'generating' status message of one turn."""
    def __init__(self, store, settings, scene, channel, model_label):
        self.store, self.settings, self.scene, self.channel, self.model_label = store, settings, scene, channel, model_label
        self.message = None

    def status(self, phase):
        t = channel_t(getattr(self.channel, 'guild', None))
        summary = self.store.model_usage_summary(self.scene.guild_id, self.settings.dialogue, self.settings.profile('dialogue').model)
        tokens = summary['input_tokens'] + summary['output_tokens']
        if summary['unreported']:
            return t('⏳ {phase} · {model} · 24h tracked: {tokens:,} tokens (+{unreported} unreported calls)',
                     phase=phase, model=self.model_label, tokens=tokens, unreported=summary['unreported'])
        return t('⏳ {phase} · {model} · 24h tracked: {tokens:,} tokens', phase=phase, model=self.model_label, tokens=tokens)

    async def update(self, content):
        try:
            if self.message:
                await self.message.edit(content=content, allowed_mentions=discord.AllowedMentions.none())
            else:
                self.message = await self.channel.send(content, silent=True,
                    allowed_mentions=discord.AllowedMentions.none())
            return True
        except discord.DiscordException as error:
            logging.warning('Generation status update failed: %s', error_detail(error))
            return False

    async def clear(self, fallback=None):
        if fallback is None:
            fallback = channel_t(getattr(self.channel, 'guild', None))('Generation stopped.')
        if self.message:
            try:
                await self.message.delete()
            except discord.DiscordException as error:
                logging.warning('Generation status cleanup failed: %s', error_detail(error))
                await self.update(fallback)
            self.message = None


USE_IN_SERVER_CHANNEL = N_("Use this command in a server channel.")
CATCHUP_COOLDOWN_SECONDS = 300
CATCHUP_NOT_HERE = N_("/catchup works only in character channels on this server. "
                      "A server admin can allow it in other channels under Server settings in the dashboard.")


def _command_context(bot: SkitBot) -> SimpleNamespace:
    async def binding_for(interaction: discord.Interaction):
        if not interaction.guild or not interaction.channel:
            raise UserError(codes.USE_IN_SERVER)
        parent_id, binding = bot.location(interaction.channel)
        if not binding:
            raise UserError(codes.NOT_BOUND)
        return parent_id, binding

    def require_guild(interaction: discord.Interaction) -> None:
        if not interaction.guild:
            raise UserError(codes.USE_IN_SERVER)

    def eligible_rows(interaction: discord.Interaction):
        if not interaction.guild or not interaction.channel:
            return []
        _, binding = bot.location(interaction.channel)
        return bot.store.eligible_characters(interaction.guild_id, binding["space_id"]) if binding else []

    def guild_characters(guild_id: int):
        return bot.store.all("SELECT * FROM characters WHERE guild_id=? AND archived=0 ORDER BY name", (guild_id,))

    def cast_rows(interaction: discord.Interaction):
        """Eligible characters plus current cast members (a member can go stale after a hub unlinks its world)."""
        if not interaction.guild or not interaction.channel:
            return []
        parent_id, binding = bot.location(interaction.channel)
        if not binding:
            return []
        rows = {row["id"]: row for row in bot.store.eligible_characters(interaction.guild_id, binding["space_id"])}
        for ident in bot.store.get_cast(interaction.channel.id, parent_id):
            row = bot.store.character_by_id(ident)
            if row and row["guild_id"] == interaction.guild_id:
                rows.setdefault(ident, row)
        return sorted(rows.values(), key=lambda row: row["name"])

    def eligible_character(interaction: discord.Interaction, binding, text: str):
        return resolve(bot.store.eligible_characters(interaction.guild_id, binding["space_id"]), text, "character",
                       " available here")

    def eligible_ids(interaction: discord.Interaction, binding, text: str) -> list[int]:
        rows = bot.store.eligible_characters(interaction.guild_id, binding["space_id"])
        return [resolve(rows, part, "character", " available here")["id"] for part in text.split(",") if part.strip()]

    def guild_space(interaction: discord.Interaction, text: str, kind: str | None = None):
        return resolve_space(bot.store.list_spaces(interaction.guild_id), text, kind)

    def safe_choices(source):
        async def complete(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
            if not interaction.guild:
                return []
            try:
                return source(interaction, current)
            except Exception as error:
                logging.warning("Autocomplete failed: %s", error_detail(error))
                return []
        return complete

    character_choices = safe_choices(lambda interaction, current: suggest([row["name"] for row in eligible_rows(interaction)], current))
    characters_choices = safe_choices(lambda interaction, current: suggest([row["name"] for row in eligible_rows(interaction)], current, many=True))
    cast_member_choices = safe_choices(lambda interaction, current: suggest([row["name"] for row in cast_rows(interaction)], current))
    guild_character_choices = safe_choices(lambda interaction, current: suggest(
        [row["name"] for row in guild_characters(interaction.guild_id)], current))

    def space_choices(kind: str | None = None):
        return safe_choices(lambda interaction, current: suggest(
            [row["name"] for row in bot.store.list_spaces(interaction.guild_id) if kind is None or row["kind"] == kind], current))

    def local_scope(interaction: discord.Interaction) -> tuple[str, int]:
        return ("thread", interaction.channel.id) if isinstance(interaction.channel, discord.Thread) else ("channel", interaction.channel.id)

    return SimpleNamespace(binding_for=binding_for, require_guild=require_guild, eligible_rows=eligible_rows, guild_characters=guild_characters, cast_rows=cast_rows, eligible_character=eligible_character, eligible_ids=eligible_ids, guild_space=guild_space, safe_choices=safe_choices, character_choices=character_choices, characters_choices=characters_choices, cast_member_choices=cast_member_choices, guild_character_choices=guild_character_choices, space_choices=space_choices, local_scope=local_scope)


def _register_space_commands(bot: SkitBot, ctx: SimpleNamespace, admin_space: app_commands.Group) -> None:
    require_guild, guild_space, space_choices = ctx.require_guild, ctx.guild_space, ctx.space_choices
    space = app_commands.Group(name="space", description=C_("List the server's worlds and hubs"))

    @admin_space.command(name="create", description=C_("Create a world or hub"))
    @app_commands.checks.has_permissions(administrator=True)
    async def space_create(interaction: discord.Interaction, kind: Literal["world", "hub"], name: str):
        require_guild(interaction)
        t = invoker_t(interaction)
        ident = bot.store.create_space(interaction.guild_id, name, kind)
        await interaction.response.send_message(t("Created {space} (#{ident}).", space=_space_ref(t, kind, name), ident=ident), ephemeral=True)

    @space.command(name="list", description=C_("List the server's worlds and hubs"))
    async def space_list(interaction: discord.Interaction):
        require_guild(interaction)
        t = invoker_t(interaction)
        rows = bot.store.list_spaces(interaction.guild_id)
        text = "\n".join(
            t("#{ident} {kind}: {name} (off duty)", ident=row['id'], kind=_kind_word(t, row['kind']), name=row['name'])
            if row['kind'] == 'hub' and row['hub_tone'] == 'off_duty'
            else t("#{ident} {kind}: {name}", ident=row['id'], kind=_kind_word(t, row['kind']), name=row['name'])
            for row in rows) or t("No worlds or hubs yet.")
        await interaction.response.send_message(text[:1900], ephemeral=True)

    @admin_space.command(name="bind", description=C_("Bind a text channel to a world or hub"))
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.autocomplete(space=space_choices())
    async def space_bind(interaction: discord.Interaction, channel: discord.TextChannel, space: str):
        require_guild(interaction)
        t = invoker_t(interaction)
        chosen = guild_space(interaction, space)
        previous = bot.store.channel(channel.id)
        dropped = bot.store.bind_channel(interaction.guild_id, channel.id, chosen["id"])
        target = _space_ref(t, chosen['kind'], chosen['name'])
        if previous and previous["space_id"] == chosen["id"]:
            message = t("{channel} is already bound to {target}; its cast and ambient setting were kept.", channel=channel.mention, target=target)
        elif dropped:
            names = [row["name"] for ident in dropped if (row := bot.store.character_by_id(ident))]
            message = t("Bound {channel} to {target}. Removed from its cast (not available there): {names}.", channel=channel.mention, target=target, names=', '.join(names))
        elif previous:
            message = t("Bound {channel} to {target}. Its cast and ambient setting were kept.", channel=channel.mention, target=target)
        else:
            message = t("Bound {channel} to {target}.", channel=channel.mention, target=target)
        await interaction.response.send_message(message[:1900], ephemeral=True)

    @admin_space.command(name="link_world", description=C_("Link a world to a hub so its characters can appear there"))
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.autocomplete(hub=space_choices("hub"), world=space_choices("world"))
    async def space_link(interaction: discord.Interaction, hub: str, world: str):
        require_guild(interaction)
        t = invoker_t(interaction)
        hub_row, world_row = guild_space(interaction, hub, "hub"), guild_space(interaction, world, "world")
        bot.store.link_world(interaction.guild_id, hub_row["id"], world_row["id"])
        await interaction.response.send_message(t("Linked {world} to {hub}.", world=world_row['name'], hub=hub_row['name']), ephemeral=True)

    @admin_space.command(name="unlink_world", description=C_("Unlink a world from a hub and drop its characters from that hub's channel casts"))
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.autocomplete(hub=space_choices("hub"), world=space_choices("world"))
    async def space_unlink(interaction: discord.Interaction, hub: str, world: str):
        require_guild(interaction)
        t = invoker_t(interaction)
        hub_row, world_row = guild_space(interaction, hub, "hub"), guild_space(interaction, world, "world")
        pruned = bot.store.unlink_world(interaction.guild_id, hub_row["id"], world_row["id"])
        await interaction.response.send_message(
            t.n("{world} is no longer linked to {hub}; removed {count} cast entry.", "{world} is no longer linked to {hub}; removed {count} cast entries.",
                pruned, world=world_row['name'], hub=hub_row['name']), ephemeral=True)

    @admin_space.command(name="tone", description=C_("Set how characters behave in a hub"))
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.autocomplete(hub=space_choices("hub"))
    @app_commands.choices(tone=[app_commands.Choice(name=C_("In character"), value="in_character"), app_commands.Choice(name=C_("Off duty"), value="off_duty")])
    async def space_tone(interaction: discord.Interaction, hub: str, tone: str):
        require_guild(interaction)
        t = invoker_t(interaction)
        hub_row = guild_space(interaction, hub, "hub")
        bot.store.set_hub_tone(interaction.guild_id, hub_row["id"], tone)
        tone_label = {"in_character": t("in character"), "off_duty": t("off duty")}.get(tone, tone.replace('_', ' '))
        await interaction.response.send_message(t("Hub {hub} tone: {tone}.", hub=hub_row['name'], tone=tone_label), ephemeral=True, allowed_mentions=discord.AllowedMentions.none())

    bot.tree.add_command(space)


def _register_character_commands(bot: SkitBot, ctx: SimpleNamespace, admin_character: app_commands.Group) -> None:
    binding_for, require_guild, guild_characters, guild_space, guild_character_choices, space_choices = ctx.binding_for, ctx.require_guild, ctx.guild_characters, ctx.guild_space, ctx.guild_character_choices, ctx.space_choices
    character = app_commands.Group(name="character", description=C_("Inspect characters available here"))

    @admin_character.command(name="import", description=C_("Import a V2/V3 JSON or PNG character card"))
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.autocomplete(world=space_choices("world"))
    async def character_import(interaction: discord.Interaction, world: str, attachment: discord.Attachment):
        require_guild(interaction)
        t = invoker_t(interaction)
        home = guild_space(interaction, world, "world")
        if attachment.size > 8 * 1024 * 1024:
            raise UserError(codes.CARD_TOO_BIG)
        await interaction.response.defer(ephemeral=True)
        parsed = parse_card(attachment.filename, await attachment.read())
        ident = bot.store.add_character(interaction.guild_id, home["id"], parsed.name, parsed.data, parsed.avatar, parsed.entries)
        await interaction.followup.send(t("Imported {name} (#{ident}) into {home} with {count} lore entries.", name=parsed.name, ident=ident, home=home['name'], count=len(parsed.entries)), ephemeral=True)

    @character.command(name="list", description=C_("List characters available here"))
    async def character_list(interaction: discord.Interaction):
        t = invoker_t(interaction)
        _, binding = await binding_for(interaction)
        rows = bot.store.eligible_characters(interaction.guild_id, binding["space_id"])
        text = ", ".join(row["name"] for row in rows) or t("No characters available.")
        await interaction.response.send_message(text[:1900], ephemeral=True)

    @character.command(name="info", description=C_("Show a character's home world"))
    @app_commands.autocomplete(character=guild_character_choices)
    async def character_info(interaction: discord.Interaction, character: str):
        require_guild(interaction)
        t = invoker_t(interaction)
        row = resolve(guild_characters(interaction.guild_id), character, "character", " in this server")
        world = bot.store.space_by_id(row["world_id"])
        await interaction.response.send_message(t("{name} — home world: {world}", name=row['name'], world=world['name']), ephemeral=True)

    bot.tree.add_command(character)


def _register_cast_commands(bot: SkitBot, ctx: SimpleNamespace, admin_cast: app_commands.Group) -> None:
    binding_for, cast_rows, eligible_character, eligible_ids, character_choices, characters_choices, cast_member_choices = ctx.binding_for, ctx.cast_rows, ctx.eligible_character, ctx.eligible_ids, ctx.character_choices, ctx.characters_choices, ctx.cast_member_choices
    cast = app_commands.Group(name="cast", description=C_("Manage this channel or thread's active cast"))

    @cast.command(name="set", description=C_("Set the active cast with comma-separated names"))
    @app_commands.autocomplete(characters=characters_choices)
    async def cast_set(interaction: discord.Interaction, characters: str):
        t = invoker_t(interaction)
        parent_id, binding = await binding_for(interaction)
        ids = eligible_ids(interaction, binding, characters)
        if not ids:
            raise UserError(codes.CAST_NAMES_NEEDED)
        bot.store.set_cast(interaction.channel.id, parent_id, ids)
        names = [bot.store.character_by_id(ident)["name"] for ident in dict.fromkeys(ids)]
        await interaction.response.send_message(t("Active cast: {names}", names=', '.join(names)), ephemeral=True)

    @cast.command(name="add", description=C_("Add an eligible character to the active cast"))
    @app_commands.autocomplete(character=character_choices)
    async def cast_add(interaction: discord.Interaction, character: str):
        t = invoker_t(interaction)
        parent_id, binding = await binding_for(interaction)
        row = eligible_character(interaction, binding, character)
        current = bot.store.get_cast(interaction.channel.id, parent_id)
        bot.store.set_cast(interaction.channel.id, parent_id, [*current, row["id"]])
        await interaction.response.send_message(t("Added {name} to the active cast.", name=row['name']), ephemeral=True)

    @cast.command(name="remove", description=C_("Remove a character from the active cast"))
    @app_commands.autocomplete(character=cast_member_choices)
    async def cast_remove(interaction: discord.Interaction, character: str):
        t = invoker_t(interaction)
        parent_id, binding = await binding_for(interaction)
        row = resolve(cast_rows(interaction), character, "character", " available here or in the cast")
        eligible = {item["id"] for item in bot.store.eligible_characters(interaction.guild_id, binding["space_id"])}
        current = bot.store.get_cast(interaction.channel.id, parent_id)
        kept = [ident for ident in current if ident != row["id"] and ident in eligible]
        bot.store.set_cast(interaction.channel.id, parent_id, kept)
        message = t("Removed {name} from the active cast.", name=row['name'])
        if dropped := len([ident for ident in current if ident != row["id"]]) - len(kept):
            message += " " + t.n("Also dropped {count} character no longer available here.", "Also dropped {count} characters no longer available here.", dropped)
        await interaction.response.send_message(message, ephemeral=True)

    @cast.command(name="show", description=C_("Show this channel or thread's active cast"))
    async def cast_show(interaction: discord.Interaction):
        t = invoker_t(interaction)
        parent_id, _ = await binding_for(interaction)
        ids = bot.store.get_cast(interaction.channel.id, parent_id)
        names = [bot.store.character_by_id(ident)["name"] for ident in ids if bot.store.character_by_id(ident)]
        await interaction.response.send_message(t("Active cast: {names}", names=', '.join(names)) if names else t("The active cast is empty. Use /cast set to choose one."), ephemeral=True)

    @admin_cast.command(name="default", description=C_("Set the channel's default cast"))
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.autocomplete(characters=characters_choices)
    async def cast_default(interaction: discord.Interaction, characters: str):
        t = invoker_t(interaction)
        parent_id, binding = await binding_for(interaction)
        ids = eligible_ids(interaction, binding, characters)
        bot.store.set_cast(interaction.channel.id, parent_id, ids, default=True)
        await interaction.response.send_message(t("Channel default cast updated."), ephemeral=True)

    bot.tree.add_command(cast)


def _register_ambient_commands(bot: SkitBot, ctx: SimpleNamespace, admin_ambient: app_commands.Group) -> None:
    binding_for = ctx.binding_for
    ambient = app_commands.Group(name="ambient", description=C_("Show this channel's ambient setting"))

    @admin_ambient.command(name="on", description=C_("Enable ambient participation in this channel"))
    @app_commands.checks.has_permissions(administrator=True)
    async def ambient_on(interaction: discord.Interaction):
        t = invoker_t(interaction)
        _, binding = await binding_for(interaction)
        bot.store.set_ambient(binding["channel_id"], True)
        await interaction.response.send_message(t("Ambient is on for {channel}.", channel=f"<#{binding['channel_id']}>"), ephemeral=True)

    @admin_ambient.command(name="off", description=C_("Disable ambient participation in this channel"))
    @app_commands.checks.has_permissions(administrator=True)
    async def ambient_off(interaction: discord.Interaction):
        t = invoker_t(interaction)
        _, binding = await binding_for(interaction)
        bot.store.set_ambient(binding["channel_id"], False)
        await interaction.response.send_message(t("Ambient is off for {channel}.", channel=f"<#{binding['channel_id']}>"), ephemeral=True)

    @ambient.command(name="status", description=C_("Show this channel's ambient setting"))
    async def ambient_status(interaction: discord.Interaction):
        t = invoker_t(interaction)
        _, binding = await binding_for(interaction)
        channel_ref = f"<#{binding['channel_id']}>"
        message = t("Ambient is on for {channel}.", channel=channel_ref) if binding["ambient"] else t("Ambient is off for {channel}.", channel=channel_ref)
        await interaction.response.send_message(message, ephemeral=True)

    bot.tree.add_command(ambient)


def _money(amount: float) -> str:
    return f"${amount:,.2f}"


def _hard_cap_notice(state, subject: str | None = None, *, t=None) -> str:
    t = t or i18n.english()
    if subject is None:
        subject = t("Character replies")
    return t("{subject} are paused: this bot reached its spending limit for this period. "
             "They resume on {date} (UTC), or sooner if an operator raises the limit.", subject=subject, date=state.resets_on.isoformat())


def _operator_text(kind: str, state, *, t=None) -> str:
    t = t or i18n.english()
    resets = state.resets_on.isoformat()
    if kind == 'soft':
        # i18n: {hard_cap} is either "off" or a dollar amount
        text = t("Spending warning: this bot has spent {spent} since {period}, "
                 "past the soft cap of {soft_cap}. Hard cap: "
                 "{hard_cap}. The period resets on {resets} (UTC).", spent=_money(state.spent_usd), period=state.period,
                 soft_cap=_money(state.soft_cap_usd), hard_cap=t('off') if state.hard_cap_usd is None else _money(state.hard_cap_usd), resets=resets)
    else:
        text = t("Spending limit reached: this bot has spent {spent} since {period}, "
                 "at or past the hard cap of {hard_cap}. Character replies, ambient turns, catch-ups and "
                 "memory updates are paused until {resets} (UTC) or until an operator raises the cap in the dashboard's Bot settings.",
                 spent=_money(state.spent_usd), period=state.period, hard_cap=_money(state.hard_cap_usd), resets=resets)
    n = state.unpriced_calls
    if n > 0:
        text += " " + t.n("{count} model call without a cost estimate was counted as $0.",
                          "{count} model calls without a cost estimate were counted as $0.", n)
    return text


def _register_summon_command(bot: SkitBot, ctx: SimpleNamespace) -> None:
    binding_for, eligible_character, character_choices = ctx.binding_for, ctx.eligible_character, ctx.character_choices
    @bot.tree.command(name="summon", description=C_("Invite an eligible character for one turn"))
    @app_commands.autocomplete(character=character_choices)
    async def summon(interaction: discord.Interaction, character: str, prompt: str):
        t = invoker_t(interaction)
        parent_id, binding = await binding_for(interaction)
        row = eligible_character(interaction, binding, character)
        game, table = bot._table_game(interaction.guild_id, interaction.channel.id)
        if table is not None:
            if row["id"] not in table["seated_character_ids"]:
                await interaction.response.send_message(t("Only the characters seated at this table can speak in its thread."), ephemeral=True)
                return
        state = bot._budget_check()
        if state and state.hard_reached:
            await interaction.response.send_message(_hard_cap_notice(state, t=t), ephemeral=True)
            return
        if table is not None and not game.talk_allowed(interaction.guild_id):
            await interaction.response.send_message(t("The table's characters aren't talking right now (table talk is off or today's limit is reached)."), ephemeral=True)
            return
        cutoff = interaction.created_at - timedelta(seconds=bot.base_settings.limits["recent_window_seconds"])
        reset_at = bot.store.scene_reset_at(interaction.channel.id)
        latest = bot.store.latest_character_node(interaction.channel.id)
        parent_message_id = latest["message_id"] if latest and latest["created_at"] >= max(cutoff.timestamp(), reset_at) else None
        after_parent = latest["created_at"] if parent_message_id else 0.0
        recent = await recent_human_lines(interaction.channel, bot.base_settings.limits, cutoff, max(reset_at, after_parent))
        await interaction.response.send_message(
            channel_t(interaction.guild)("{user} summons {name}: {prompt}", user=interaction.user.display_name, name=row['name'], prompt=prompt[:1700]),
            allowed_mentions=discord.AllowedMentions.none())
        invitation = await interaction.original_response()
        scene = SceneContext(interaction.guild_id, interaction.channel.id, parent_id,
            binding["space_id"], interaction.user.id, invitation.id, prompt,
            parent_message_id, recent, [], forced_character_id=row["id"], user_label=interaction.user.display_name,
            mentioned_users=[discord_identity(member) for ident in re.findall(r'<@!?(\d+)>', prompt)
                             if (member := interaction.guild.get_member(int(ident))) and not member.bot],
            **bot._table_fields(table))
        async with bot.channel_locks.lock(interaction.channel.id):
            await bot.run_scene(scene, interaction.channel, interaction)


def _register_catchup_command(bot: SkitBot) -> None:
    @bot.tree.command(name="catchup", description=C_("Privately summarize what you missed in this channel"))
    @app_commands.describe(focus=C_("What to emphasize, such as what concerns you"), hours=C_("Look back this many hours instead"))
    async def catchup_command(interaction: discord.Interaction, focus: app_commands.Range[str, 1, 300] | None = None,
                              hours: app_commands.Range[int, 1, 72] | None = None):
        if not interaction.guild_id or not interaction.channel:
            raise UserError(codes.USE_IN_SERVER)
        t = invoker_t(interaction)
        guild_id, channel, user = interaction.guild_id, interaction.channel, interaction.user

        async def reply(text):
            send = interaction.followup.send if interaction.response.is_done() else interaction.response.send_message
            await send(catchup.fit(text), ephemeral=True, allowed_mentions=discord.AllowedMentions.none())

        if not (interaction.permissions.read_message_history or interaction.permissions.administrator):
            return await reply(t("You can't read this channel's message history, so I can't summarize it."))
        if not bot.location(channel)[1] and not bot.store.catchup_anywhere(guild_id):
            return await reply(t(CATCHUP_NOT_HERE))
        state = bot._budget_check()
        if state and state.hard_reached:
            return await reply(_hard_cap_notice(state, t("Catch-ups"), t=t))
        key = (guild_id, channel.id, user.id)
        now_mono = time.monotonic()
        wait = bot.catchup_used[key] + CATCHUP_COOLDOWN_SECONDS - now_mono if key in bot.catchup_used else 0
        if wait > 0:
            minutes = -(-int(wait) // 60) or 1
            return await reply(t.n("You can use /catchup here again in {minutes} minute.", "You can use /catchup here again in {minutes} minutes.", minutes, minutes=minutes))
        bot.catchup_used = {k: v for k, v in bot.catchup_used.items() if now_mono - v < CATCHUP_COOLDOWN_SECONDS}
        bot.catchup_used[key] = now_mono  # claimed before any await so a concurrent call is refused
        attempted = False
        try:
            await interaction.response.defer(ephemeral=True, thinking=True)
            try:
                lines = await catchup.collect(channel.history(limit=catchup.SCAN_LIMIT), user.id,
                                              catchup.cutoff(interaction.created_at, hours), until_own=hours is None)
            except discord.Forbidden:
                return await reply(t("I can't read this channel's message history."))
            if not lines:
                return await reply(t.render(catchup.nothing_new_phrase(hours)))
            facts = ([row["content"] for row in bot.store.personal(guild_id, user.id)[-catchup.MAX_FACTS:]]
                     if bot.store.has_consent(guild_id, user.id) else [])
            message = catchup.user_message(user.display_name, catchup.transcript(lines), focus, facts, hours)
            attempted = True
            try:
                with bot.backend.pin(), capture_usage(guild_id, channel.id, 'catchup'), log_scope(getattr(interaction, 'id', None)), log_purpose('catchup'):
                    mask_for_log(*catchup.fact_variants(facts))
                    text = await bot.models.text('memory', catchup.SYSTEM, [TurnMessage('user', message)],
                                                 max_tokens=catchup.MAX_OUTPUT_TOKENS)
            except budget.BudgetExceeded as error:
                attempted = False
                return await reply(_hard_cap_notice(error.state, t("Catch-ups"), t=t))
            except Exception as error:
                ref = reference_id()
                logging.error('Catchup failed [ref %s]: %s\n%s', ref, error_detail(error), error_stack(error))
                bot._log_failure(error, ref, 'catchup', guild_id=guild_id, channel_id=channel.id, feature='catchup',
                                 message_id=getattr(interaction, 'id', None), facts=catchup.fact_variants(facts))
                return await reply(t("Could not write the catch-up: {detail} (ref {ref})", detail=_member_detail(error, ref, t=t)[:300], ref=ref))
        finally:
            if not attempted:
                bot.catchup_used.pop(key, None)
        await reply(text.strip() or t("The model returned nothing. Try again later."))


def _register_memory_commands(bot: SkitBot, ctx: SimpleNamespace) -> None:
    require_guild = ctx.require_guild
    memory = app_commands.Group(name="memory", description=C_("Control your personal character memories"))

    @memory.command(name="opt_in", description=C_("Allow characters to remember facts you explicitly state"))
    async def memory_opt_in(interaction: discord.Interaction):
        require_guild(interaction)
        t = invoker_t(interaction)
        bot.store.set_consent(interaction.guild_id, interaction.user.id, True)
        await interaction.response.send_message(t("Personal memory enabled. Use /memory list or /memory opt_out anytime."), ephemeral=True)

    @memory.command(name="opt_out", description=C_("Disable and erase your personal memories"))
    async def memory_opt_out(interaction: discord.Interaction):
        require_guild(interaction)
        t = invoker_t(interaction)
        bot.store.set_consent(interaction.guild_id, interaction.user.id, False)
        await interaction.response.send_message(t("Personal memory disabled and your saved personal facts erased."), ephemeral=True)

    @memory.command(name="list", description=C_("List what characters remember about you"))
    async def memory_list(interaction: discord.Interaction):
        require_guild(interaction)
        t = invoker_t(interaction)
        rows = bot.store.personal(interaction.guild_id, interaction.user.id)
        lines = []
        for row in rows:
            character_row = bot.store.character_by_id(row["character_id"])
            lines.append(t("#{ident} {name}: {content}", ident=row['id'], name=character_row['name'] if character_row else t('Unknown'), content=row['content']))
        await interaction.response.send_message("\n".join(lines)[:1900] or t("No personal memories saved."), ephemeral=True)

    @memory.command(name="forget", description=C_("Remove one of your personal memories"))
    async def memory_forget(interaction: discord.Interaction, memory_id: int):
        require_guild(interaction)
        t = invoker_t(interaction)
        if bot.store.forget_personal(interaction.guild_id, interaction.user.id, memory_id):
            message = t("Memory #{memory_id} removed.", memory_id=memory_id)
        else:
            message = t("No memory #{memory_id} of yours was found.", memory_id=memory_id)
        await interaction.response.send_message(message, ephemeral=True)

    bot.tree.add_command(memory)


_TIMEZONE_NAMES: list[str] = []
UNKNOWN_TIMEZONE = N_("Unknown timezone. Pick one from the list, such as Asia/Seoul.")


async def _timezone_choices(interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
    if len(current) > 64:
        return []
    if not _TIMEZONE_NAMES:
        _TIMEZONE_NAMES.extend(sorted(available_timezones()))
    needle = current.lower()
    return [app_commands.Choice(name=z, value=z) for z in _TIMEZONE_NAMES if needle in z.lower()][:25]


def _register_time_commands(bot: SkitBot, ctx: SimpleNamespace) -> None:
    require_guild = ctx.require_guild
    time_group = app_commands.Group(name="time", description=C_("Set the timezone characters use for you"))

    def local_time(interaction: discord.Interaction, zone: str) -> str:
        return time_values(interaction.id, zone)["local_time"]

    @time_group.command(name="set", description=C_("Choose your timezone in this server"))
    @app_commands.describe(zone=C_("IANA timezone name, such as Asia/Seoul"))
    @app_commands.autocomplete(zone=_timezone_choices)
    async def time_set(interaction: discord.Interaction, zone: str):
        require_guild(interaction)
        t = invoker_t(interaction)
        if len(zone) > 64:
            message = t(UNKNOWN_TIMEZONE)
        else:
            try:
                bot.store.set_user_timezone(interaction.guild_id, interaction.user.id, zone)
                message = t("Your timezone here is now {zone}. Local time: {time}.", zone=zone, time=local_time(interaction, zone))
            except ValueError:
                message = t(UNKNOWN_TIMEZONE)
        await interaction.response.send_message(message, ephemeral=True)

    @time_group.command(name="show", description=C_("Show the timezone characters use for you"))
    async def time_show(interaction: discord.Interaction):
        require_guild(interaction)
        t = invoker_t(interaction)
        zone, source = bot.store.resolve_timezone(interaction.guild_id, interaction.user.id)
        now = local_time(interaction, zone)
        if source == "member":
            message = t("Your timezone here is {zone} (your setting). Local time: {time}.", zone=zone, time=now)
        elif source == "server":
            message = t("Your timezone here is {zone} (server default). Local time: {time}. Use /time set to choose your own.", zone=zone, time=now)
        else:
            message = t("Your timezone here is UTC (no timezone set). Local time: {time}. Use /time set to choose yours.", time=now)
        await interaction.response.send_message(message, ephemeral=True)

    @time_group.command(name="clear", description=C_("Remove your timezone in this server"))
    async def time_clear(interaction: discord.Interaction):
        require_guild(interaction)
        t = invoker_t(interaction)
        if not bot.store.clear_user_timezone(interaction.guild_id, interaction.user.id):
            message = t("You had no timezone set here.")
        else:
            zone, source = bot.store.resolve_timezone(interaction.guild_id, interaction.user.id)
            if source == "server":
                message = t("Cleared your timezone. Characters now use {zone} (server default).", zone=zone)
            else:
                message = t("Cleared your timezone. Characters now use UTC (no timezone set).")
        await interaction.response.send_message(message, ephemeral=True)

    bot.tree.add_command(time_group)


def _register_currency_commands(bot: SkitBot, ctx: SimpleNamespace, admin_currency: app_commands.Group) -> None:
    require_guild = ctx.require_guild

    async def reply(interaction: discord.Interaction, message: str) -> None:
        await interaction.response.send_message(message, ephemeral=True, allowed_mentions=discord.AllowedMentions.none())

    @bot.tree.command(name="balance", description=C_("Show your balance, or another member's"))
    @app_commands.describe(member=C_("Whose balance to show (default: yours)"), character=C_("A character whose balance to show"))
    @app_commands.autocomplete(character=ctx.guild_character_choices)
    async def balance(interaction: discord.Interaction, member: discord.Member | None = None, character: str | None = None):
        require_guild(interaction)
        t = invoker_t(interaction)
        guild_id, name = interaction.guild_id, bot.store.currency_name(interaction.guild_id)
        if member is not None and character:
            await reply(interaction, t("Choose a member or a character, not both."))
        elif character:
            row = resolve(ctx.guild_characters(guild_id), character, "character", " in this server")
            await reply(interaction, t("{who} has {amount:,} {currency}.", who=row['name'], amount=bot.store.character_balance(guild_id, row['id']), currency=name))
        elif member is None or member.id == interaction.user.id:
            await reply(interaction, t("You have {amount:,} {currency}.", amount=bot.store.balance(guild_id, interaction.user.id), currency=name))
        else:
            await reply(interaction, t("{who} has {amount:,} {currency}.", who=member.mention, amount=bot.store.balance(guild_id, member.id), currency=name))

    @bot.tree.command(name="daily", description=C_("Claim your daily check-in bonus"))
    async def daily(interaction: discord.Interaction):
        require_guild(interaction)
        t = invoker_t(interaction)
        guild_id = interaction.guild_id
        try:
            result = bot.store.claim_daily(guild_id, interaction.user.id)
        except CurrencyError as error:
            await reply(interaction, t.error(error))
            return
        name = bot.store.currency_name(guild_id)
        if result["status"] == "off":
            await reply(interaction, t("Daily check-ins are off on this server."))
        elif result["status"] == "already":
            await reply(interaction, t("You already checked in today. Next check-in {reset}.", reset=f"<t:{int(result['reset'])}:R>"))
        else:
            await reply(interaction, t("Checked in: +{payout:,} {currency} (day {streak} streak). Balance: {balance:,} {currency}. Next check-in {reset}.",
                                       payout=result['payout'], currency=name, streak=result['streak'], balance=result['balance'], reset=f"<t:{int(result['reset'])}:R>"))

    @admin_currency.command(name="daily", description=C_("Set the daily check-in bonus (amount 0 turns it off)"))
    @app_commands.describe(amount=C_("Currency paid for each check-in (0 turns check-ins off)"), streak_bonus=C_("Extra per consecutive day (default: keep the current value)"),
        streak_days=C_("How many streak days earn the bonus, up to 365 (default: keep the current value)"))
    @app_commands.checks.has_permissions(administrator=True)
    async def currency_daily(interaction: discord.Interaction, amount: app_commands.Range[int, 0, 1_000_000],
            streak_bonus: app_commands.Range[int, 0, 1_000_000] | None = None, streak_days: app_commands.Range[int, 0, 365] | None = None):
        require_guild(interaction)
        t = invoker_t(interaction)
        current = bot.store.daily_settings(interaction.guild_id)
        try:
            saved = bot.store.set_daily_settings(interaction.guild_id, amount, current["streak_bonus"] if streak_bonus is None else streak_bonus,
                current["streak_days"] if streak_days is None else streak_days, expected=current)
        except ConflictError:
            await reply(interaction, t("The daily check-in settings were changed elsewhere. Run the command again."))
            return
        name = bot.store.currency_name(interaction.guild_id)
        if saved["amount"] == 0:
            await reply(interaction, t("Daily check-ins are now off."))
        elif saved["streak_bonus"] == 0 or saved["streak_days"] == 0:
            await reply(interaction, t("Daily check-in: {amount:,} {currency}, no streak bonus.", amount=saved['amount'], currency=name))
        else:
            await reply(interaction, t("Daily check-in: {amount:,} {currency}, +{bonus:,} per streak day for up to {days:,} days.",
                                       amount=saved['amount'], currency=name, bonus=saved['streak_bonus'], days=saved['streak_days']))

    async def change(interaction: discord.Interaction, member: discord.Member, amount: int, reason: str, sign: int) -> None:
        require_guild(interaction)
        t = invoker_t(interaction)
        if member.bot:
            await reply(interaction, t("Bots cannot hold a balance."))
            return
        try:
            row = bot.store.change_balance(interaction.guild_id, member.id, sign * amount, reason, interaction.user.id)
        except CurrencyError as error:
            await reply(interaction, t.error(error))
            return
        name = bot.store.currency_name(interaction.guild_id)
        if sign > 0:
            message = t("Gave {amount:,} {currency} to {member}. New balance: {balance:,} {currency}.", amount=amount, currency=name, member=member.mention, balance=row['balance_after'])
        else:
            message = t("Took {amount:,} {currency} from {member}. New balance: {balance:,} {currency}.", amount=amount, currency=name, member=member.mention, balance=row['balance_after'])
        await reply(interaction, message)

    @admin_currency.command(name="grant", description=C_("Give a member currency"))
    @app_commands.describe(member=C_("Who receives it"), amount=C_("How much to give"), reason=C_("Why (kept in the ledger)"))
    @app_commands.checks.has_permissions(administrator=True)
    async def currency_grant(interaction: discord.Interaction, member: discord.Member, amount: app_commands.Range[int, 1, 1_000_000], reason: str):
        await change(interaction, member, amount, reason, 1)

    @admin_currency.command(name="revoke", description=C_("Take currency from a member"))
    @app_commands.describe(member=C_("Who loses it"), amount=C_("How much to take"), reason=C_("Why (kept in the ledger)"))
    @app_commands.checks.has_permissions(administrator=True)
    async def currency_revoke(interaction: discord.Interaction, member: discord.Member, amount: app_commands.Range[int, 1, 1_000_000], reason: str):
        await change(interaction, member, amount, reason, -1)

    @admin_currency.command(name="name", description=C_("Set what this server calls its currency"))
    @app_commands.describe(name=C_("For example coins or gold (1 to 32 characters)"))
    @app_commands.checks.has_permissions(administrator=True)
    async def currency_name(interaction: discord.Interaction, name: str):
        require_guild(interaction)
        t = invoker_t(interaction)
        saved = bot.store.set_currency_name(interaction.guild_id, name)
        await reply(interaction, t("This server's currency is now called {name}.", name=saved))


def _register_lore_commands(bot: SkitBot, ctx: SimpleNamespace, admin_lore: app_commands.Group) -> None:
    binding_for, require_guild, guild_space, space_choices, local_scope = ctx.binding_for, ctx.require_guild, ctx.guild_space, ctx.space_choices, ctx.local_scope
    lore = app_commands.Group(name="lore", description=C_("Show the lore available here"))

    @admin_lore.command(name="add", description=C_("Add lore to this channel (or thread) or to its world or hub"))
    @app_commands.checks.has_permissions(administrator=True)
    async def lore_add(interaction: discord.Interaction, content: str, keys: str = "", scope: Literal["channel", "space"] = "channel"):
        t = invoker_t(interaction)
        _, binding = await binding_for(interaction)
        scope_kind, scope_id = local_scope(interaction) if scope == "channel" else ("space", binding["space_id"])
        parsed = [key.strip() for key in keys.split(",") if key.strip()]
        home = bot.store.space_by_id(binding["space_id"]) if scope == "space" else None
        label = _kind_word(t, home["kind"]) if home else t.p("lore scope", "channel") if scope == "channel" else t.p("lore scope", "world or hub")
        ident = bot.store.add_lore(interaction.guild_id, scope_kind, scope_id, content, parsed, constant=not parsed, pinned=not parsed)
        await interaction.response.send_message(t("Added {label} lore #{ident}.", label=label, ident=ident), ephemeral=True)

    @lore.command(name="list", description=C_("Show the lore available in this location"))
    async def lore_list(interaction: discord.Interaction):
        t = invoker_t(interaction)
        _, binding = await binding_for(interaction)
        scopes = [("space", binding["space_id"]), ("channel", binding["channel_id"])]
        if isinstance(interaction.channel, discord.Thread):
            scopes.append(("thread", interaction.channel.id))
        home = bot.store.space_by_id(binding["space_id"])
        names = {"space": _kind_word(t, home["kind"]) if home else t.p("lore scope", "world or hub"),
                 "channel": t.p("lore scope", "channel"), "thread": t.p("lore scope", "thread")}
        lines = []
        for kind, ident in scopes:
            lines.extend(t("#{ident} [{scope}] {content}", ident=row['id'], scope=names.get(kind, kind), content=row['content'][:100])
                         for row in bot.store.list_lore(interaction.guild_id, kind, ident))
        await interaction.response.send_message("\n".join(lines)[:1900] or t("No lore for this channel or its world or hub yet."), ephemeral=True)

    @admin_lore.command(name="pin", description=C_("Pin a lore entry"))
    @app_commands.checks.has_permissions(administrator=True)
    async def lore_pin(interaction: discord.Interaction, lore_id: int):
        require_guild(interaction)
        t = invoker_t(interaction)
        if not bot.store.lore_row(interaction.guild_id, lore_id):
            raise UserError(codes.LORE_NOT_FOUND)
        bot.store.pin_lore(interaction.guild_id, lore_id)
        await interaction.response.send_message(t("Pinned lore #{lore_id}.", lore_id=lore_id), ephemeral=True)

    @admin_lore.command(name="edit", description=C_("Correct the text of a lore entry"))
    @app_commands.checks.has_permissions(administrator=True)
    async def lore_edit(interaction: discord.Interaction, lore_id: int, content: str):
        require_guild(interaction)
        t = invoker_t(interaction)
        if not bot.store.lore_row(interaction.guild_id, lore_id):
            raise UserError(codes.LORE_NOT_FOUND)
        bot.store.edit_lore(interaction.guild_id, lore_id, content)
        await interaction.response.send_message(t("Updated lore #{lore_id}.", lore_id=lore_id), ephemeral=True)

    @admin_lore.command(name="promote", description=C_("Copy lore into a channel or world/hub"))
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.autocomplete(space=space_choices())
    async def lore_promote(interaction: discord.Interaction, lore_id: int, destination: Literal["channel", "space"], space: str = ""):
        t = invoker_t(interaction)
        _, binding = await binding_for(interaction)
        if not bot.store.lore_row(interaction.guild_id, lore_id):
            raise UserError(codes.LORE_NOT_FOUND)
        if destination == "channel":
            target_kind, target_id = "channel", binding["channel_id"]
        else:
            target = guild_space(interaction, space) if space.strip() else bot.store.space_by_id(binding["space_id"])
            if not target or target["guild_id"] != interaction.guild_id:
                raise UserError(codes.DESTINATION_NOT_FOUND)
            target_kind, target_id = "space", target["id"]
        ident = bot.store.promote_lore(interaction.guild_id, lore_id, target_kind, target_id)
        where = f"<#{target_id}>" if target_kind == "channel" else _space_ref(t, target['kind'], target['name'])
        await interaction.response.send_message(t("Promoted lore #{lore_id} to {where} as lore #{ident}.", lore_id=lore_id, where=where, ident=ident), ephemeral=True)

    @admin_lore.command(name="delete", description=C_("Delete a lore entry"))
    @app_commands.checks.has_permissions(administrator=True)
    async def lore_delete(interaction: discord.Interaction, lore_id: int):
        require_guild(interaction)
        t = invoker_t(interaction)
        if bot.store.delete_lore(interaction.guild_id, lore_id):
            message = t("Deleted lore #{lore_id}.", lore_id=lore_id)
        else:
            message = t("No lore #{lore_id} in this server.", lore_id=lore_id)
        await interaction.response.send_message(message, ephemeral=True)

    bot.tree.add_command(lore)


def _register_scene_commands(bot: SkitBot, ctx: SimpleNamespace, admin_scene: app_commands.Group) -> None:
    binding_for = ctx.binding_for
    scene = app_commands.Group(name="scene", description=C_("Start a fresh scene"))

    @scene.command(name="reset", description=C_("Start a fresh scene from your next invitation"))
    async def scene_reset(interaction: discord.Interaction):
        t = invoker_t(interaction)
        await binding_for(interaction)
        bot.store.reset_scene(interaction.channel.id)
        await interaction.response.send_message(t("Scene reset. Your next mention or summon starts fresh."), ephemeral=True)

    @admin_scene.command(name="delete", description=C_("Delete a stored message and everything after it in its branch"))
    @app_commands.describe(message_id=C_("Any stored message in this channel's scene"))
    @app_commands.checks.has_permissions(administrator=True)
    async def scene_delete(interaction: discord.Interaction, message_id: str):
        t = invoker_t(interaction)
        await binding_for(interaction)
        try:
            target_id = int(message_id)
        except ValueError as error:
            raise UserError(codes.NUMERIC_MESSAGE_ID) from error
        node = bot.store.node(target_id)
        if not node or node["guild_id"] != interaction.guild_id or node["channel_id"] != interaction.channel.id:
            raise UserError(codes.SCENE_MESSAGE_NOT_FOUND)
        count, personal, encounters = bot.store.delete_subtree_counts(interaction.guild_id, target_id)
        message = t("Deleted {count} stored messages from this scene.", count=count)
        if personal or encounters:
            if personal == 1 and encounters == 1:
                message += " " + t("Also forgot {personal} personal fact and {encounters} encounter.", personal=personal, encounters=encounters)
            elif personal == 1:
                message += " " + t("Also forgot {personal} personal fact and {encounters} encounters.", personal=personal, encounters=encounters)
            elif encounters == 1:
                message += " " + t("Also forgot {personal} personal facts and {encounters} encounter.", personal=personal, encounters=encounters)
            else:
                message += " " + t("Also forgot {personal} personal facts and {encounters} encounters.", personal=personal, encounters=encounters)
        await interaction.response.send_message(message, ephemeral=True)

    bot.tree.add_command(scene)


def _clean(value: str) -> str:
    return discord.utils.escape_mentions(discord.utils.escape_markdown(value))


def _context_lore_line(bot: SkitBot, interaction: discord.Interaction, item: dict, *, t=None) -> str:
    """One /context lore entry (administrators only) as words: its first key or a content excerpt, and its owner, never a raw id."""
    t = t or i18n.english()
    kind, scope_id, guild_id = item["scope"], item["scope_id"], interaction.guild_id

    clean = _clean

    label = ""
    if kind in ("character", "space", "channel", "thread"):
        row = bot.store.lore_row(guild_id, item["id"])
        if row:
            try:
                keys = json.loads(row["keys_json"] or "[]")
            except ValueError:
                keys = []
            if isinstance(keys, list) and keys and isinstance(keys[0], str) and keys[0].strip():
                label = keys[0].strip()[:40]
            else:
                label = " ".join(str(row["content"]).split())[:40]
    if kind in ("lorebook", "guild"):
        ref, sql = (("book-entry", "SELECT e.content FROM lorebook_entries e JOIN lorebooks b ON b.id=e.book_id WHERE b.guild_id=? AND e.id=?")
                    if kind == "lorebook" else ("guild-lore", "SELECT content FROM guild_lore_entries WHERE guild_id=? AND id=?"))
        try:
            entry = bot.store.admin_entry(guild_id, f"{ref}:{item['id']}")
            content, rule = entry["content"], entry["rule"]
        except (ValueError, TypeError, KeyError):
            # Unreadable rule JSON: still a guild-scoped content excerpt.
            found = bot.store.one(sql, (guild_id, item["id"]))
            content, rule = (found["content"] if found else ""), {}
        keys = rule.get("keys") if isinstance(rule, dict) else None
        if isinstance(keys, list) and keys and isinstance(keys[0], str) and keys[0].strip():
            label = keys[0].strip()[:40]
        else:
            label = " ".join(str(content).split())[:40]
    if not label:
        key = str(item.get("entry_key") or "")
        label = "" if key.startswith(("lore:", "book:")) else key
    label = f'"{clean(label)}"' if label else t("an unnamed entry")
    if kind == "space":
        space = bot.store.space_by_id(scope_id)
        if space and space["guild_id"] != guild_id:
            space = None
        owner = _space_ref(t, space['kind'], clean(space['name'])) if space else t("a world or hub")
    elif kind == "character":
        character = bot.store.character_by_id(scope_id)
        if character and character["guild_id"] != guild_id:
            character = None
        owner = t("character {name}", name=clean(character['name'])) if character else t("a character")
    elif kind == "lorebook":
        owner = t("lorebook {name}", name=clean(item['book_name'])) if item.get("book_name") else t("a lorebook")
    elif kind == "guild":
        owner = t("server-wide lore")
    elif kind in ("channel", "thread"):
        channel = interaction.guild.get_channel_or_thread(scope_id) if interaction.guild else None
        name = getattr(channel, "name", "")
        owner = f"#{clean(name)}" if name else (t("a thread") if kind == "thread" else t("#unknown-channel"))
    else:
        owner = t("another owner")
    return t("{label} ({owner}, {reason})", label=label, owner=owner, reason=_lore_reason(item['reason'], clean, t))


def _lore_reason(reason, clean, t):
    reason = str(reason)
    if reason == "sticky":
        return t("sticky")
    if reason == "constant":
        return t("constant")
    if reason.startswith("keyword: "):
        return t("keyword: {word}", word=clean(reason[len("keyword: "):]))
    return clean(reason)


def _register_context_command(bot: SkitBot, ctx: SimpleNamespace) -> None:
    binding_for = ctx.binding_for
    @bot.tree.command(name="context", description=C_("Inspect what informed the last character line"))
    @app_commands.default_permissions(administrator=True)
    @app_commands.checks.has_permissions(administrator=True)
    async def context(interaction: discord.Interaction, message_id: str = ""):
        t = invoker_t(interaction)
        await binding_for(interaction)
        if message_id:
            try:
                ident = int(message_id)
            except ValueError as error:
                raise UserError(codes.NUMERIC_MESSAGE_ID) from error
        else:
            latest = bot.store.latest_character_node(interaction.channel.id)
            ident = latest["message_id"] if latest else 0
        node, trace = bot.store.node(ident), bot.store.trace(ident)
        if not node or node["guild_id"] != interaction.guild_id or node["channel_id"] != interaction.channel.id or not trace:
            raise UserError(codes.NO_SAVED_CONTEXT)
        lore_lines = [_context_lore_line(bot, interaction, item, t=t) for item in trace.get("lore", [])]
        text = t("Lore: {lore}", lore=", ".join(lore_lines) or t("none"))
        text += "\n" + t("Branch messages: {branch}; nearby group messages: {nearby}", branch=len(trace.get('messages', [])), nearby=len(trace.get('recent', [])))
        text += "\n" + t("Personal facts: {facts}; character encounters: {encounters}", facts=len(trace.get('personal', [])), encounters=len(trace.get('encounters', [])))
        preset = trace.get('preset', {})
        if preset:
            clean = _clean
            if preset['id'] == 0:
                name = t("Built-in default")
            else:
                found = next((row for row in bot.store.list_presets(interaction.guild_id) if row['id'] == preset['id']), None)
                name = clean(found['name']) if found else f"#{preset['id']}"
            text += "\n" + t("Preset: {name} (revision {revision}); emotion: {emotion}", name=name, revision=preset['revision'], emotion=trace.get('emotion', 'neutral'))
            text += "\n" + t("Prompt blocks: {blocks}; trimmed blocks: {trimmed}", blocks=len(preset.get('blocks', [])), trimmed=len(preset.get('omitted', [])))
            if preset.get('adaptations'):
                text += "\n" + t("Provider adaptations: {adaptations}", adaptations=clean('; '.join(preset['adaptations'])[:300]))
        await interaction.response.send_message(text[:1900], ephemeral=True, allowed_mentions=discord.AllowedMentions.none())


def _register_error_handler(bot: SkitBot) -> None:
    @bot.tree.error
    async def command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
        t = invoker_t(interaction)
        if isinstance(error, app_commands.MissingPermissions):
            # UI-06: a stale DM client can still invoke a guild-only command; say where to use it, not who may.
            # guild_id, not guild: guild can be None for an uncached guild while guild_id is set.
            message = t(USE_IN_SERVER_CHANNEL) if interaction.guild_id is None else t("Only server administrators can use that command.")
        elif isinstance(error, app_commands.CommandInvokeError) and isinstance(error.original, ValueError):
            message = t.error(error.original)
        elif (isinstance(error, app_commands.CommandInvokeError) and isinstance(error.original, sqlite3.IntegrityError)
              and str(error.original).startswith('UNIQUE')):
            message = t("That already exists.")
        else:
            original = getattr(error, 'original', error)
            ref = reference_id()
            logging.error('Command %s failed [ref %s]: %s\n%s', getattr(getattr(error, 'command', None), 'qualified_name', '?'),
                          ref, error_detail(original), error_stack(original))
            bot._log_failure(original, ref, getattr(getattr(error, 'command', None), 'qualified_name', '?'), guild_id=interaction.guild_id,
                             channel_id=getattr(interaction, 'channel_id', None), feature='command', message_id=getattr(interaction, 'id', None))
            pointer = bot._failure_pointer(interaction.guild_id, t('The error was logged.'), t=t)
            message = t("Something went wrong (ref {ref}). {pointer}", ref=ref, pointer=pointer)
        if interaction.response.is_done():
            await interaction.followup.send(message[:1900], ephemeral=True)
        else:
            await interaction.response.send_message(message[:1900], ephemeral=True)


def register_commands(bot: SkitBot) -> None:
    ctx = _command_context(bot)
    admin = app_commands.Group(name="admin", description=C_("Server administrator commands"),
                               default_permissions=discord.Permissions(administrator=True))
    admin_space = app_commands.Group(name="space", description=C_("Manage worlds, hubs and channel bindings"), parent=admin)
    admin_character = app_commands.Group(name="character", description=C_("Import characters"), parent=admin)
    admin_cast = app_commands.Group(name="cast", description=C_("Set a channel's default cast"), parent=admin)
    admin_ambient = app_commands.Group(name="ambient", description=C_("Turn ambient participation on or off"), parent=admin)
    admin_lore = app_commands.Group(name="lore", description=C_("Add and manage lore"), parent=admin)
    admin_scene = app_commands.Group(name="scene", description=C_("Delete stored scenes"), parent=admin)
    admin_currency = app_commands.Group(name="currency", description=C_("Give, take and name the server currency"), parent=admin)
    admin_games = app_commands.Group(name="games", description=C_("Turn games on in channels and set bet limits"), parent=admin)

    _register_space_commands(bot, ctx, admin_space)
    _register_character_commands(bot, ctx, admin_character)
    _register_cast_commands(bot, ctx, admin_cast)
    _register_ambient_commands(bot, ctx, admin_ambient)
    _register_summon_command(bot, ctx)
    register_favorites_commands(bot, ctx)
    _register_catchup_command(bot)
    _register_memory_commands(bot, ctx)
    _register_time_commands(bot, ctx)
    _register_currency_commands(bot, ctx, admin_currency)
    register_game_commands(bot, ctx, admin_games)
    register_doubt_commands(bot, ctx)
    register_items_commands(bot, ctx)
    _register_lore_commands(bot, ctx, admin_lore)
    _register_scene_commands(bot, ctx, admin_scene)
    bot.tree.add_command(admin)
    _register_context_command(bot, ctx)
    _register_error_handler(bot)
