from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
import re
import hashlib
from dataclasses import replace
from datetime import timedelta
from typing import Literal

import discord
from discord import app_commands
from discord.ext import commands

from .cards import parse_card
from .config import Settings
from .engine import Engine, SceneContext
from .models import ImageInput, ModelGateway
from .names import resolve, resolve_space, suggest
from .store import Store
from .avatars import emotion_stream
from .errors import error_detail, error_stack, reference_id, user_detail
from .usage import capture_usage, reply_footer
from .identity import discord_identity, message_context

AVATAR_ASSET_CHECK_TTL = 600
MEMORY_WAIT_SECONDS = 15
PROVIDER_STAGES = {'speaker selection', 'image description', 'dialogue generation'}
MEMORY_CLOSE_SECONDS = 5
NO_CHARACTER_NOTE_SECONDS = 15


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


class SkitBot(commands.Bot):
    def __init__(self, settings: Settings):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=commands.when_mentioned, intents=intents)
        self.settings = settings
        self.store = Store(settings.database_path)
        self.models = ModelGateway(settings, usage_sink=self.store.record_model_usage)
        self.engine = Engine(self.store, self.models, settings)
        self.channel_locks: dict[int, asyncio.Lock] = {}
        self.webhook_locks = {}
        self.webhooks = {}
        self.webhook_defaults = {}
        self.checked_avatar_assets = {}
        self.cleanup_task: asyncio.Task | None = None
        self.tree.allowed_contexts = app_commands.AppCommandContext(guild=True)
        self.memory_tasks: dict[int, asyncio.Task] = {}
        self.note_tasks: set[asyncio.Task] = set()
        register_commands(self)

    async def setup_hook(self):
        if self.settings.development_guild_id:
            guild = discord.Object(id=self.settings.development_guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def on_ready(self):
        logging.info("Discord connected as %s; servers=%d; command scope=%s",
                     self.user, len(self.guilds), self.settings.development_guild_id or "global")

    async def _cleanup_loop(self):
        while True:
            try:
                self.store.expire_history(self.settings.history_retention_days)
            except Exception:
                logging.exception('History cleanup failed')
            await asyncio.sleep(86400)

    async def close(self):
        if self.cleanup_task:
            self.cleanup_task.cancel()
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
        parent_id, binding = self.location(message.channel)
        if not binding:
            return
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
            async with self.channel_locks.setdefault(message.channel.id, asyncio.Lock()):
                self.store.count_ambient_message(message.channel.id)
                count, last = self.store.ambient_state(message.channel.id)
                if count < 2 or time.time() - last < self.settings.limits["ambient_cooldown_seconds"]:
                    return
                # A silent director decision still consumes this ambient opportunity.
                self.store.mark_ambient_response(message.channel.id)
        recent = []
        latest = self.store.latest_character_node(message.channel.id)
        # A rewind sees only the ancestry at its target, never newer channel chat.
        is_rewind = bool(explicit_reply and latest and reference_id != latest["message_id"])
        if not is_rewind:
            cutoff = message.created_at - timedelta(seconds=self.settings.limits["recent_window_seconds"])
            reset_at = self.store.scene_reset_at(message.channel.id)
            after_parent = referenced["created_at"] if explicit_reply and referenced else 0.0
            async for old in message.channel.history(before=message, limit=self.settings.limits["recent_messages"] * 5):
                if old.created_at < cutoff or old.created_at.timestamp() < max(reset_at, after_parent):
                    break
                if old.author.bot or old.webhook_id or not old.content:
                    continue
                recent.append(message_context(old))
                if len(recent) >= self.settings.limits["recent_messages"]:
                    break
            recent.reverse()
        text = message.content
        if self.user:
            text = text.replace(self.user.mention, "", 1).strip()
        images = []
        try:
            for attachment in message.attachments:
                if attachment.size > self.settings.limits["max_attachment_bytes"]:
                    raise ValueError("An attachment exceeds the configured size limit")
                if attachment.content_type and attachment.content_type.startswith("image/"):
                    if len(images) >= self.settings.limits["max_images"]:
                        raise ValueError("Too many image attachments")
                    if not self.settings.profile("dialogue").supports_images:
                        raise ValueError("The configured dialogue model cannot read images")
                    images.append(ImageInput(attachment.content_type, await attachment.read()))
                elif attachment.content_type and attachment.content_type.startswith("text/"):
                    text += "\n" + (await attachment.read()).decode("utf-8", errors="replace")[:12000]
                else:
                    raise ValueError("Only text and image attachments are supported")
        except ValueError as error:
            await message.reply(str(error), mention_author=False)
            return
        scene = SceneContext(message.guild.id, message.channel.id, parent_id,
            binding["space_id"], message.author.id, message.id, text,
            reference_id if referenced else None, recent, images, ambient=not explicit, user_label=message.author.display_name,
            mentioned_users=[discord_identity(user) for user in message.mentions if not user.bot])
        async with self.channel_locks.setdefault(message.channel.id, asyncio.Lock()):
            await self.run_scene(scene, message.channel)

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
        async with self.webhook_locks.setdefault(key, asyncio.Lock()):
            return await self._webhook_locked(parent_channel, character, key)

    async def _webhook_identity(self, webhook, character, key, identity):
        if self.webhook_defaults.get(key) != identity:
            try:
                webhook = await webhook.edit(name=character['name'][:80], avatar=character['avatar'])
            except discord.NotFound:
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

    async def run_scene(self, scene: SceneContext, channel, interaction=None):
        previous = self.memory_tasks.get(channel.id)
        if previous and not previous.done():
            _, pending = await asyncio.wait({previous}, timeout=MEMORY_WAIT_SECONDS)
            if pending:
                logging.warning('Memory task for channel %s still running after %ss; continuing without it',
                                channel.id, MEMORY_WAIT_SECONDS)
        with capture_usage(scene.guild_id):
            return await self._run_scene(scene, channel, interaction)

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
            with capture_usage(scene.guild_id):
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

    async def _run_scene(self, scene: SceneContext, channel, interaction=None):
        progress = None
        model_label = discord.utils.escape_markdown(self.settings.profile('dialogue').model[:120])

        def status(phase):
            summary = self.store.model_usage_summary(scene.guild_id, self.settings.dialogue, self.settings.profile('dialogue').model)
            tokens = summary['input_tokens'] + summary['output_tokens']
            qualifier = f" (+{summary['unreported']} unreported calls)" if summary['unreported'] else ''
            return f'⏳ {phase} · {model_label} · 24h tracked: {tokens:,} tokens{qualifier}'

        async def update_progress(content):
            nonlocal progress
            try:
                if progress:
                    await progress.edit(content=content, allowed_mentions=discord.AllowedMentions.none())
                else:
                    progress = await channel.send(content, silent=True,
                        allowed_mentions=discord.AllowedMentions.none())
                return True
            except discord.DiscordException as error:
                logging.warning('Generation status update failed: %s', error_detail(error))
                return False

        async def clear_progress(fallback='Generation stopped.'):
            nonlocal progress
            if progress:
                try:
                    await progress.delete()
                except discord.DiscordException as error:
                    logging.warning('Generation status cleanup failed: %s', error_detail(error))
                    await update_progress(fallback)
                progress = None

        stage = 'speaker selection'
        try:
            if not scene.ambient:
                await update_progress(status('Generating a reply…'))
            scene = replace(scene, preset=scene.preset or self.store.active_preset(scene.guild_id),
                guidelines=scene.guidelines if scene.guidelines is not None else self.store.scene_guidelines(scene.guild_id, scene.space_id, scene.parent_channel_id or scene.channel_id))
            speakers = await self.engine.speakers(scene)
            if not speakers:
                if not scene.ambient:
                    if await update_progress("No active character is available here. Use /cast set or /summon."):
                        self._delete_later(progress)
                        progress = None
                return
            if scene.ambient:
                await update_progress(status('Generating a reply…'))
            stage = 'image description'
            image_description = await self.engine.describe_images(scene)
            stored_text = scene.text + (f"\n[Image description: {image_description}]" if image_description else "")
            stage = 'saving scene input'
            root_id = self.engine.record_user(scene, stored_text)
            preceding: list[tuple[str, str]] = []
            parent_message_id = scene.user_message_id
            completed = []
            for character in speakers:
                name = discord.utils.escape_markdown(character['name'])
                await update_progress(status(f'**{name}** is preparing a reply…'))
                stage = 'preparing character prompt'
                request, sources = await self.engine.prepare_dialogue(scene, character, preceding)
                system = '\n\n'.join(m.text for m in request.messages if m.role == 'system')
                messages = [m for m in request.messages if m.role != 'system']
                stage = 'webhook setup'
                webhook = await self._webhook(channel, character)
                thread_options = {'thread': channel} if isinstance(channel, discord.Thread) else {}
                slots = {row['slot_key']: row for row in self.store.usable_avatars(scene.guild_id, character['id'])}
                emotion, chosen_avatar, placeholder = 'neutral', slots['neutral'], None
                pieces, last_edit = [], 0.0
                try:
                    stage = 'dialogue generation'
                    await update_progress(status(f'Streaming **{name}**'))
                    with capture_usage() as usage_records:
                        stream = self.models.stream_compiled('dialogue', request) if hasattr(self.models, 'stream_compiled') else self.models.stream_text('dialogue', system, messages)
                        async for event in emotion_stream(stream, slots):
                            if event.emotion is not None:
                                emotion = event.emotion
                                stage = 'avatar lookup'
                                chosen_avatar = await self.resolve_avatar(slots.get(emotion, slots['neutral']))
                                stage = 'webhook delivery'
                                try:
                                    placeholder = await webhook.send('…', **thread_options, wait=True, silent=True,
                                        username=character['name'], avatar_url=chosen_avatar['url'], allowed_mentions=discord.AllowedMentions.none())
                                except discord.NotFound:
                                    self._forget_webhook(self._webhook_key(channel, character))
                                    stage = 'webhook setup'
                                    webhook = await self._webhook(channel, character)
                                    stage = 'webhook delivery'
                                    placeholder = await webhook.send('…', **thread_options, wait=True, silent=True,
                                        username=character['name'], avatar_url=chosen_avatar['url'], allowed_mentions=discord.AllowedMentions.none())
                                stage = 'dialogue generation'
                                continue
                            pieces.append(event.text)
                            current = "".join(pieces)
                            if time.monotonic() - last_edit > 1.2 and len(current) < 1800:
                                stage = 'webhook delivery'
                                await placeholder.edit(content=current + " ▌", allowed_mentions=discord.AllowedMentions.none())
                                last_edit = time.monotonic()
                                stage = 'dialogue generation'
                    line = re.sub(r'<emotion>[^\n]*?</emotion>\s*', '', "".join(pieces)).strip()
                    usage = usage_records[-1] if usage_records else None
                    footer = reply_footer(model_label, usage) if self.store.usage_footer_enabled(scene.guild_id) else ''
                    chunks = split_discord(line, limit=min(1900, 2000 - len(footer)))
                    stage = 'webhook delivery'
                    await placeholder.edit(content=chunks[0] + footer, allowed_mentions=discord.AllowedMentions.none())
                except Exception as error:
                    if isinstance(error, discord.NotFound):
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
                except discord.NotFound:
                    self._forget_webhook(self._webhook_key(channel, character))
                    raise
                for posted, chunk in zip(outgoing, chunks):
                    stage = 'saving character reply'
                    self.store.record_node(posted.id, scene.guild_id, scene.channel_id,
                        parent_message_id, None, character["id"], chunk,
                        sources=sources["messages"])
                    self.store.save_trace(posted.id, {"character_id": character["id"], 'emotion': emotion,
                        'usage': usage.as_dict() if usage else None,
                        'avatar_source': 'emotion' if chosen_avatar['asset_id'] else 'static' if character['avatar'] else 'default',
                        'avatar_asset_id': chosen_avatar['asset_id'], **sources})
                    self.store.save_lore_activations(posted.id, sources.get("lore_activations", []))
                    parent_message_id = posted.id
                preceding.append((character["name"], line))
                completed.append(character)
            await clear_progress('Reply sent.')
            if scene.ambient:
                self.store.mark_ambient_response(scene.channel_id)
            self._schedule_memory(channel.id, scene, completed, preceding, root_id, parent_message_id)
        except Exception as error:
            ref = reference_id()
            logging.error('Scene failed during %s [ref %s]: %s\n%s', stage, ref, error_detail(error), error_stack(error))
            failure = f"The character couldn't reply (ref {ref}). An admin can find details in the bot log."
            try:
                if progress:
                    try:
                        await progress.edit(content=failure, allowed_mentions=discord.AllowedMentions.none())
                        progress = None
                    except discord.DiscordException:
                        await channel.send(failure, allowed_mentions=discord.AllowedMentions.none())
                else:
                    await channel.send(failure, allowed_mentions=discord.AllowedMentions.none())
            except discord.DiscordException as post_error:
                logging.warning('Public failure notice failed [ref %s]: %s', ref, error_detail(post_error))
            if interaction:
                detail = (user_detail(error) if stage in PROVIDER_STAGES
                          else 'internal error. An admin can find details in the bot log.')
                try:
                    await interaction.followup.send(f"Your turn failed during {stage} (ref {ref}): {detail}"[:1900],
                        ephemeral=True, allowed_mentions=discord.AllowedMentions.none())
                except Exception as notify_error:
                    logging.warning('Private failure notice failed [ref %s]: %s', ref, error_detail(notify_error))
        finally:
            await clear_progress()


def register_commands(bot: SkitBot) -> None:
    async def binding_for(interaction: discord.Interaction):
        if not interaction.guild or not interaction.channel:
            raise ValueError("This command is available in server channels only")
        parent_id, binding = bot.location(interaction.channel)
        if not binding:
            raise ValueError("This channel is not bound to a world or hub")
        return parent_id, binding

    def require_guild(interaction: discord.Interaction) -> None:
        if not interaction.guild:
            raise ValueError("Use this command in a server.")

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

    admin = app_commands.Group(name="admin", description="Server administrator commands",
                               default_permissions=discord.Permissions(administrator=True))
    admin_space = app_commands.Group(name="space", description="Manage worlds, hubs and channel bindings", parent=admin)
    admin_character = app_commands.Group(name="character", description="Import characters", parent=admin)
    admin_cast = app_commands.Group(name="cast", description="Set a channel's default cast", parent=admin)
    admin_ambient = app_commands.Group(name="ambient", description="Turn ambient participation on or off", parent=admin)
    admin_lore = app_commands.Group(name="lore", description="Add and manage lore", parent=admin)
    admin_scene = app_commands.Group(name="scene", description="Delete stored scenes", parent=admin)

    space = app_commands.Group(name="space", description="List the server's worlds and hubs")

    @admin_space.command(name="create", description="Create a world or hub")
    @app_commands.checks.has_permissions(administrator=True)
    async def space_create(interaction: discord.Interaction, kind: Literal["world", "hub"], name: str):
        require_guild(interaction)
        ident = bot.store.create_space(interaction.guild_id, name, kind)
        await interaction.response.send_message(f"Created {kind} {name} (#{ident}).", ephemeral=True)

    @space.command(name="list", description="List the server's worlds and hubs")
    async def space_list(interaction: discord.Interaction):
        require_guild(interaction)
        rows = bot.store.list_spaces(interaction.guild_id)
        text = "\n".join(f"#{row['id']} {row['kind']}: {row['name']}" for row in rows) or "No worlds or hubs yet."
        await interaction.response.send_message(text[:1900], ephemeral=True)

    @admin_space.command(name="bind", description="Bind a text channel to a world or hub")
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.autocomplete(space=space_choices())
    async def space_bind(interaction: discord.Interaction, channel: discord.TextChannel, space: str):
        require_guild(interaction)
        chosen = guild_space(interaction, space)
        previous = bot.store.channel(channel.id)
        dropped = bot.store.bind_channel(interaction.guild_id, channel.id, chosen["id"])
        target = f"{chosen['kind']} {chosen['name']}"
        if previous and previous["space_id"] == chosen["id"]:
            message = f"{channel.mention} is already bound to {target}; its cast and ambient setting were kept."
        elif dropped:
            names = [row["name"] for ident in dropped if (row := bot.store.character_by_id(ident))]
            message = f"Bound {channel.mention} to {target}. Removed from its cast (not available there): {', '.join(names)}."
        elif previous:
            message = f"Bound {channel.mention} to {target}. Its cast and ambient setting were kept."
        else:
            message = f"Bound {channel.mention} to {target}."
        await interaction.response.send_message(message[:1900], ephemeral=True)

    @admin_space.command(name="link_world", description="Link a world to a hub so its characters can appear there")
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.autocomplete(hub=space_choices("hub"), world=space_choices("world"))
    async def space_link(interaction: discord.Interaction, hub: str, world: str):
        require_guild(interaction)
        hub_row, world_row = guild_space(interaction, hub, "hub"), guild_space(interaction, world, "world")
        bot.store.link_world(interaction.guild_id, hub_row["id"], world_row["id"])
        await interaction.response.send_message(f"Linked {world_row['name']} to {hub_row['name']}.", ephemeral=True)

    @admin_space.command(name="unlink_world", description="Unlink a world from a hub and drop its characters from that hub's channel casts")
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.autocomplete(hub=space_choices("hub"), world=space_choices("world"))
    async def space_unlink(interaction: discord.Interaction, hub: str, world: str):
        require_guild(interaction)
        hub_row, world_row = guild_space(interaction, hub, "hub"), guild_space(interaction, world, "world")
        pruned = bot.store.unlink_world(interaction.guild_id, hub_row["id"], world_row["id"])
        entries = "cast entry" if pruned == 1 else "cast entries"
        await interaction.response.send_message(
            f"{world_row['name']} is no longer linked to {hub_row['name']}; removed {pruned} {entries}.", ephemeral=True)

    bot.tree.add_command(space)

    character = app_commands.Group(name="character", description="Inspect characters available here")

    @admin_character.command(name="import", description="Import a V2/V3 JSON or PNG character card")
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.autocomplete(world=space_choices("world"))
    async def character_import(interaction: discord.Interaction, world: str, attachment: discord.Attachment):
        require_guild(interaction)
        home = guild_space(interaction, world, "world")
        if attachment.size > 8 * 1024 * 1024:
            raise ValueError("Card exceeds 8 MiB")
        await interaction.response.defer(ephemeral=True)
        parsed = parse_card(attachment.filename, await attachment.read())
        ident = bot.store.add_character(interaction.guild_id, home["id"], parsed.name, parsed.data, parsed.avatar, parsed.entries)
        await interaction.followup.send(f"Imported {parsed.name} (#{ident}) into {home['name']} with {len(parsed.entries)} lore entries.", ephemeral=True)

    @character.command(name="list", description="List characters available here")
    async def character_list(interaction: discord.Interaction):
        _, binding = await binding_for(interaction)
        rows = bot.store.eligible_characters(interaction.guild_id, binding["space_id"])
        text = ", ".join(row["name"] for row in rows) or "No characters available."
        await interaction.response.send_message(text[:1900], ephemeral=True)

    @character.command(name="info", description="Show a character's home world")
    @app_commands.autocomplete(character=guild_character_choices)
    async def character_info(interaction: discord.Interaction, character: str):
        require_guild(interaction)
        row = resolve(guild_characters(interaction.guild_id), character, "character", " in this server")
        world = bot.store.space_by_id(row["world_id"])
        await interaction.response.send_message(f"{row['name']} — home world: {world['name']}", ephemeral=True)

    bot.tree.add_command(character)

    cast = app_commands.Group(name="cast", description="Manage this channel or thread's active cast")

    @cast.command(name="set", description="Set the active cast with comma-separated names")
    @app_commands.autocomplete(characters=characters_choices)
    async def cast_set(interaction: discord.Interaction, characters: str):
        parent_id, binding = await binding_for(interaction)
        ids = eligible_ids(interaction, binding, characters)
        if not ids:
            raise ValueError("Give one or more character names")
        bot.store.set_cast(interaction.channel.id, parent_id, ids)
        names = [bot.store.character_by_id(ident)["name"] for ident in dict.fromkeys(ids)]
        await interaction.response.send_message(f"Active cast: {', '.join(names)}", ephemeral=True)

    @cast.command(name="add", description="Add an eligible character to the active cast")
    @app_commands.autocomplete(character=character_choices)
    async def cast_add(interaction: discord.Interaction, character: str):
        parent_id, binding = await binding_for(interaction)
        row = eligible_character(interaction, binding, character)
        current = bot.store.get_cast(interaction.channel.id, parent_id)
        bot.store.set_cast(interaction.channel.id, parent_id, [*current, row["id"]])
        await interaction.response.send_message(f"Added {row['name']} to the active cast.", ephemeral=True)

    @cast.command(name="remove", description="Remove a character from the active cast")
    @app_commands.autocomplete(character=cast_member_choices)
    async def cast_remove(interaction: discord.Interaction, character: str):
        parent_id, binding = await binding_for(interaction)
        row = resolve(cast_rows(interaction), character, "character", " available here or in the cast")
        eligible = {item["id"] for item in bot.store.eligible_characters(interaction.guild_id, binding["space_id"])}
        current = bot.store.get_cast(interaction.channel.id, parent_id)
        kept = [ident for ident in current if ident != row["id"] and ident in eligible]
        bot.store.set_cast(interaction.channel.id, parent_id, kept)
        message = f"Removed {row['name']} from the active cast."
        if dropped := len([ident for ident in current if ident != row["id"]]) - len(kept):
            message += f" Also dropped {dropped} character(s) no longer available here."
        await interaction.response.send_message(message, ephemeral=True)

    @cast.command(name="show", description="Show this channel or thread's active cast")
    async def cast_show(interaction: discord.Interaction):
        parent_id, _ = await binding_for(interaction)
        ids = bot.store.get_cast(interaction.channel.id, parent_id)
        names = [bot.store.character_by_id(ident)["name"] for ident in ids if bot.store.character_by_id(ident)]
        await interaction.response.send_message(", ".join(names) or "The cast is empty.", ephemeral=True)

    @admin_cast.command(name="default", description="Set the channel's default cast")
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.autocomplete(characters=characters_choices)
    async def cast_default(interaction: discord.Interaction, characters: str):
        parent_id, binding = await binding_for(interaction)
        ids = eligible_ids(interaction, binding, characters)
        bot.store.set_cast(interaction.channel.id, parent_id, ids, default=True)
        await interaction.response.send_message("Channel default cast updated.", ephemeral=True)

    bot.tree.add_command(cast)

    ambient = app_commands.Group(name="ambient", description="Show this channel's ambient setting")

    @admin_ambient.command(name="on", description="Enable ambient participation in this channel")
    @app_commands.checks.has_permissions(administrator=True)
    async def ambient_on(interaction: discord.Interaction):
        _, binding = await binding_for(interaction)
        bot.store.set_ambient(binding["channel_id"], True)
        await interaction.response.send_message("Ambient participation enabled for this channel.", ephemeral=True)

    @admin_ambient.command(name="off", description="Disable ambient participation in this channel")
    @app_commands.checks.has_permissions(administrator=True)
    async def ambient_off(interaction: discord.Interaction):
        _, binding = await binding_for(interaction)
        bot.store.set_ambient(binding["channel_id"], False)
        await interaction.response.send_message("Ambient participation disabled for this channel.", ephemeral=True)

    @ambient.command(name="status", description="Show this channel's ambient setting")
    async def ambient_status(interaction: discord.Interaction):
        _, binding = await binding_for(interaction)
        await interaction.response.send_message("Ambient is on." if binding["ambient"] else "Ambient is off.", ephemeral=True)

    bot.tree.add_command(ambient)

    @bot.tree.command(name="summon", description="Invite an eligible character for one turn")
    @app_commands.autocomplete(character=character_choices)
    async def summon(interaction: discord.Interaction, character: str, prompt: str):
        parent_id, binding = await binding_for(interaction)
        row = eligible_character(interaction, binding, character)
        cutoff = interaction.created_at - timedelta(seconds=bot.settings.limits["recent_window_seconds"])
        reset_at = bot.store.scene_reset_at(interaction.channel.id)
        latest = bot.store.latest_character_node(interaction.channel.id)
        parent_message_id = latest["message_id"] if latest and latest["created_at"] >= max(cutoff.timestamp(), reset_at) else None
        after_parent = latest["created_at"] if parent_message_id else 0.0
        recent = []
        async for old in interaction.channel.history(limit=bot.settings.limits["recent_messages"] * 5):
            if old.created_at < cutoff or old.created_at.timestamp() < max(reset_at, after_parent):
                break
            if old.author.bot or old.webhook_id or not old.content:
                continue
            recent.append(message_context(old))
            if len(recent) >= bot.settings.limits["recent_messages"]:
                break
        recent.reverse()
        await interaction.response.send_message(
            f"{interaction.user.display_name} summons {row['name']}: {prompt[:1700]}",
            allowed_mentions=discord.AllowedMentions.none())
        invitation = await interaction.original_response()
        scene = SceneContext(interaction.guild_id, interaction.channel.id, parent_id,
            binding["space_id"], interaction.user.id, invitation.id, prompt,
            parent_message_id, recent, [], forced_character_id=row["id"], user_label=interaction.user.display_name,
            mentioned_users=[discord_identity(member) for ident in re.findall(r'<@!?(\d+)>', prompt)
                             if (member := interaction.guild.get_member(int(ident))) and not member.bot])
        async with bot.channel_locks.setdefault(interaction.channel.id, asyncio.Lock()):
            await bot.run_scene(scene, interaction.channel, interaction)

    memory = app_commands.Group(name="memory", description="Control your personal character memories")

    @memory.command(name="opt_in", description="Allow characters to remember facts you explicitly state")
    async def memory_opt_in(interaction: discord.Interaction):
        require_guild(interaction)
        bot.store.set_consent(interaction.guild_id, interaction.user.id, True)
        await interaction.response.send_message("Personal memory enabled. Use /memory list or /memory opt_out anytime.", ephemeral=True)

    @memory.command(name="opt_out", description="Disable and erase your personal memories")
    async def memory_opt_out(interaction: discord.Interaction):
        require_guild(interaction)
        bot.store.set_consent(interaction.guild_id, interaction.user.id, False)
        await interaction.response.send_message("Personal memory disabled and your saved personal facts erased.", ephemeral=True)

    @memory.command(name="list", description="List what characters remember about you")
    async def memory_list(interaction: discord.Interaction):
        require_guild(interaction)
        rows = bot.store.personal(interaction.guild_id, interaction.user.id)
        lines = []
        for row in rows:
            character_row = bot.store.character_by_id(row["character_id"])
            lines.append(f"#{row['id']} {character_row['name'] if character_row else 'Unknown'}: {row['content']}")
        await interaction.response.send_message("\n".join(lines)[:1900] or "No personal memories saved.", ephemeral=True)

    @memory.command(name="forget", description="Remove one of your personal memories")
    async def memory_forget(interaction: discord.Interaction, memory_id: int):
        require_guild(interaction)
        if bot.store.forget_personal(interaction.guild_id, interaction.user.id, memory_id):
            message = f"Memory #{memory_id} removed."
        else:
            message = f"No memory #{memory_id} of yours was found."
        await interaction.response.send_message(message, ephemeral=True)

    bot.tree.add_command(memory)

    lore = app_commands.Group(name="lore", description="Show the lore available here")

    @admin_lore.command(name="add", description="Add lore to this channel (or thread) or to its world or hub")
    @app_commands.checks.has_permissions(administrator=True)
    async def lore_add(interaction: discord.Interaction, content: str, keys: str = "", scope: Literal["channel", "space"] = "channel"):
        _, binding = await binding_for(interaction)
        scope_kind, scope_id = local_scope(interaction) if scope == "channel" else ("space", binding["space_id"])
        parsed = [key.strip() for key in keys.split(",") if key.strip()]
        home = bot.store.space_by_id(binding["space_id"]) if scope == "space" else None
        label = home["kind"] if home else "channel" if scope == "channel" else "space"
        ident = bot.store.add_lore(interaction.guild_id, scope_kind, scope_id, content, parsed, constant=not parsed, pinned=not parsed)
        await interaction.response.send_message(f"Added {label} lore #{ident}.", ephemeral=True)

    @lore.command(name="list", description="Show the lore available in this location")
    async def lore_list(interaction: discord.Interaction):
        _, binding = await binding_for(interaction)
        scopes = [("space", binding["space_id"]), ("channel", binding["channel_id"])]
        if isinstance(interaction.channel, discord.Thread):
            scopes.append(("thread", interaction.channel.id))
        home = bot.store.space_by_id(binding["space_id"])
        names = {"space": home["kind"] if home else "space"}
        lines = []
        for kind, ident in scopes:
            lines.extend(f"#{row['id']} [{names.get(kind, kind)}] {row['content'][:100]}" for row in bot.store.list_lore(interaction.guild_id, kind, ident))
        await interaction.response.send_message("\n".join(lines)[:1900] or "No lore for this channel or its world or hub yet.", ephemeral=True)

    @admin_lore.command(name="pin", description="Pin a lore entry")
    @app_commands.checks.has_permissions(administrator=True)
    async def lore_pin(interaction: discord.Interaction, lore_id: int):
        require_guild(interaction)
        if not bot.store.lore_row(interaction.guild_id, lore_id):
            raise ValueError("Lore entry not found")
        bot.store.pin_lore(interaction.guild_id, lore_id)
        await interaction.response.send_message(f"Pinned lore #{lore_id}.", ephemeral=True)

    @admin_lore.command(name="edit", description="Correct the text of a lore entry")
    @app_commands.checks.has_permissions(administrator=True)
    async def lore_edit(interaction: discord.Interaction, lore_id: int, content: str):
        require_guild(interaction)
        bot.store.edit_lore(interaction.guild_id, lore_id, content)
        await interaction.response.send_message(f"Updated lore #{lore_id}.", ephemeral=True)

    @admin_lore.command(name="promote", description="Copy lore into a channel or world/hub")
    @app_commands.checks.has_permissions(administrator=True)
    @app_commands.autocomplete(space=space_choices())
    async def lore_promote(interaction: discord.Interaction, lore_id: int, destination: Literal["channel", "space"], space: str = ""):
        _, binding = await binding_for(interaction)
        if destination == "channel":
            target_kind, target_id = "channel", binding["channel_id"]
        else:
            target = guild_space(interaction, space) if space.strip() else bot.store.space_by_id(binding["space_id"])
            if not target or target["guild_id"] != interaction.guild_id:
                raise ValueError("Destination space not found")
            target_kind, target_id = "space", target["id"]
        ident = bot.store.promote_lore(interaction.guild_id, lore_id, target_kind, target_id)
        await interaction.response.send_message(f"Promoted lore #{lore_id} to #{ident} in {target_kind}.", ephemeral=True)

    @admin_lore.command(name="delete", description="Delete a lore entry")
    @app_commands.checks.has_permissions(administrator=True)
    async def lore_delete(interaction: discord.Interaction, lore_id: int):
        require_guild(interaction)
        if bot.store.delete_lore(interaction.guild_id, lore_id):
            message = f"Deleted lore #{lore_id}."
        else:
            message = f"No lore #{lore_id} in this server."
        await interaction.response.send_message(message, ephemeral=True)

    bot.tree.add_command(lore)

    scene = app_commands.Group(name="scene", description="Start a fresh scene")

    @scene.command(name="reset", description="Start a fresh scene from your next invitation")
    async def scene_reset(interaction: discord.Interaction):
        await binding_for(interaction)
        bot.store.reset_scene(interaction.channel.id)
        await interaction.response.send_message("Scene reset. Your next mention or summon starts fresh.", ephemeral=True)

    @admin_scene.command(name="delete", description="Delete a stored message and everything after it in its branch")
    @app_commands.describe(message_id="Any stored message in this channel's scene")
    @app_commands.checks.has_permissions(administrator=True)
    async def scene_delete(interaction: discord.Interaction, message_id: str):
        await binding_for(interaction)
        try:
            target_id = int(message_id)
        except ValueError as error:
            raise ValueError("Give a numeric message ID") from error
        node = bot.store.node(target_id)
        if not node or node["guild_id"] != interaction.guild_id or node["channel_id"] != interaction.channel.id:
            raise ValueError("Scene message not found in this channel")
        count = bot.store.delete_subtree(interaction.guild_id, target_id)
        await interaction.response.send_message(f"Deleted {count} stored messages from this scene.", ephemeral=True)

    bot.tree.add_command(scene)
    bot.tree.add_command(admin)

    @bot.tree.command(name="context", description="Inspect what informed the last character line")
    async def context(interaction: discord.Interaction, message_id: str = ""):
        await binding_for(interaction)
        if message_id:
            try:
                ident = int(message_id)
            except ValueError as error:
                raise ValueError("Give a numeric message ID") from error
        else:
            latest = bot.store.latest_character_node(interaction.channel.id)
            ident = latest["message_id"] if latest else 0
        node, trace = bot.store.node(ident), bot.store.trace(ident)
        if not node or node["guild_id"] != interaction.guild_id or node["channel_id"] != interaction.channel.id or not trace:
            raise ValueError("No saved context for that character line")
        lore_lines = []
        for item in trace.get("lore", []):
            scope = item["scope"]
            if scope == "space":
                space = bot.store.space_by_id(item["scope_id"])
                scope = f"{space['kind']} {space['name']}" if space else f"space #{item['scope_id']}"
            elif scope == "character":
                character = bot.store.character_by_id(item["scope_id"])
                scope = f"character {character['name']}" if character else f"character #{item['scope_id']}"
            elif scope == "lorebook":
                scope = f"lorebook {item.get('book_name', item['scope_id'])}"
            else:
                scope = f"{scope} #{item['scope_id']}"
            lore_lines.append(f"#{item['id']} ({scope}, {item['reason']})")
        text = "Lore: " + (", ".join(lore_lines) or "none")
        text += f"\nBranch messages: {len(trace.get('messages', []))}"
        text += f"; nearby group messages: {len(trace.get('recent', []))}"
        text += f"\nPersonal memories: {len(trace.get('personal', []))}; character encounters: {len(trace.get('encounters', []))}"
        preset = trace.get('preset', {})
        if preset:
            text += f"\nPreset: #{preset['id']} revision {preset['revision']}; emotion: {trace.get('emotion', 'neutral')}"
            text += f"\nPrompt blocks: {len(preset.get('blocks', []))}; trimmed blocks: {len(preset.get('omitted', []))}"
            if preset.get('adaptations'):
                text += '\nProvider adaptations: ' + '; '.join(preset['adaptations'])[:300]
        await interaction.response.send_message(text[:1900], ephemeral=True)

    @bot.tree.error
    async def command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
        if isinstance(error, app_commands.MissingPermissions):
            message = "Only server administrators can use that command."
        elif isinstance(error, app_commands.CommandInvokeError) and isinstance(error.original, ValueError):
            message = str(error.original)
        elif (isinstance(error, app_commands.CommandInvokeError) and isinstance(error.original, sqlite3.IntegrityError)
              and str(error.original).startswith('UNIQUE')):
            message = "That already exists."
        else:
            original = getattr(error, 'original', error)
            ref = reference_id()
            logging.error('Command %s failed [ref %s]: %s\n%s', getattr(getattr(error, 'command', None), 'qualified_name', '?'),
                          ref, error_detail(original), error_stack(original))
            message = f"Something went wrong (ref {ref}). The error was logged."
        if interaction.response.is_done():
            await interaction.followup.send(message[:1900], ephemeral=True)
        else:
            await interaction.response.send_message(message[:1900], ephemeral=True)
