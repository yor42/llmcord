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
from .store import Store
from .avatars import emotion_stream
from .errors import error_detail, error_stack
from .usage import capture_usage, reply_footer
from .identity import discord_identity, message_context

AVATAR_ASSET_CHECK_TTL = 600
MEMORY_WAIT_SECONDS = 15
MEMORY_CLOSE_SECONDS = 5


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
        self.memory_tasks: dict[int, asyncio.Task] = {}
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

    async def run_scene(self, scene: SceneContext, channel):
        previous = self.memory_tasks.get(channel.id)
        if previous and not previous.done():
            _, pending = await asyncio.wait({previous}, timeout=MEMORY_WAIT_SECONDS)
            if pending:
                logging.warning('Memory task for channel %s still running after %ss; continuing without it',
                                channel.id, MEMORY_WAIT_SECONDS)
        with capture_usage(scene.guild_id):
            return await self._run_scene(scene, channel)

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

    async def _run_scene(self, scene: SceneContext, channel):
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
                    footer = reply_footer(model_label, usage)
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
            explanation = error_detail(error)
            logging.error('Scene failed during %s: %s\n%s', stage, explanation, error_stack(error))
            failure = f"Character response failed during {stage}: {explanation}"
            if progress:
                try:
                    await progress.edit(content=failure, allowed_mentions=discord.AllowedMentions.none())
                    progress = None
                except discord.DiscordException:
                    await channel.send(failure, allowed_mentions=discord.AllowedMentions.none())
            else:
                await channel.send(failure, allowed_mentions=discord.AllowedMentions.none())
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

    def local_scope(interaction: discord.Interaction) -> tuple[str, int]:
        return ("thread", interaction.channel.id) if isinstance(interaction.channel, discord.Thread) else ("channel", interaction.channel.id)

    space = app_commands.Group(name="space", description="Manage worlds, hubs, and channels")

    @space.command(name="create", description="Create a world or hub")
    @app_commands.checks.has_permissions(administrator=True)
    async def space_create(interaction: discord.Interaction, kind: Literal["world", "hub"], name: str):
        ident = bot.store.create_space(interaction.guild_id, name, kind)
        await interaction.response.send_message(f"Created {kind} {name} (#{ident}).", ephemeral=True)

    @space.command(name="list", description="List the server's worlds and hubs")
    async def space_list(interaction: discord.Interaction):
        rows = bot.store.list_spaces(interaction.guild_id)
        text = "\n".join(f"#{row['id']} {row['kind']}: {row['name']}" for row in rows) or "No spaces yet."
        await interaction.response.send_message(text[:1900], ephemeral=True)

    @space.command(name="bind", description="Bind a text channel to a world or hub")
    @app_commands.checks.has_permissions(administrator=True)
    async def space_bind(interaction: discord.Interaction, channel: discord.TextChannel, space_name: str):
        chosen = bot.store.space(interaction.guild_id, space_name)
        if not chosen:
            raise ValueError("Space not found")
        bot.store.bind_channel(interaction.guild_id, channel.id, chosen["id"])
        await interaction.response.send_message(f"Bound {channel.mention} to {chosen['kind']} {space_name}.", ephemeral=True)

    @space.command(name="allow_world", description="Allow a world's characters in a hub")
    @app_commands.checks.has_permissions(administrator=True)
    async def space_allow(interaction: discord.Interaction, hub_name: str, world_name: str):
        hub, world = bot.store.space(interaction.guild_id, hub_name), bot.store.space(interaction.guild_id, world_name)
        if not hub or not world:
            raise ValueError("Hub or world not found")
        bot.store.link_world(interaction.guild_id, hub["id"], world["id"])
        await interaction.response.send_message(f"{world_name} is now available in {hub_name}.", ephemeral=True)

    @space.command(name="disallow_world", description="Remove a world's characters from a hub")
    @app_commands.checks.has_permissions(administrator=True)
    async def space_disallow(interaction: discord.Interaction, hub_name: str, world_name: str):
        hub, world = bot.store.space(interaction.guild_id, hub_name), bot.store.space(interaction.guild_id, world_name)
        if not hub or not world:
            raise ValueError("Hub or world not found")
        bot.store.unlink_world(interaction.guild_id, hub["id"], world["id"])
        await interaction.response.send_message(f"{world_name} is no longer linked to {hub_name}.", ephemeral=True)

    bot.tree.add_command(space)

    character = app_commands.Group(name="character", description="Import and inspect characters")

    @character.command(name="import", description="Import a V2/V3 JSON or PNG character card")
    @app_commands.checks.has_permissions(administrator=True)
    async def character_import(interaction: discord.Interaction, world_name: str, attachment: discord.Attachment):
        world = bot.store.space(interaction.guild_id, world_name)
        if not world or world["kind"] != "world":
            raise ValueError("Choose an existing world")
        if attachment.size > 8 * 1024 * 1024:
            raise ValueError("Card exceeds 8 MiB")
        await interaction.response.defer(ephemeral=True)
        parsed = parse_card(attachment.filename, await attachment.read())
        ident = bot.store.add_character(interaction.guild_id, world["id"], parsed.name, parsed.data, parsed.avatar, parsed.entries)
        await interaction.followup.send(f"Imported {parsed.name} (#{ident}) into {world_name} with {len(parsed.entries)} lore entries.", ephemeral=True)

    @character.command(name="list", description="List characters available here")
    async def character_list(interaction: discord.Interaction):
        _, binding = await binding_for(interaction)
        rows = bot.store.eligible_characters(interaction.guild_id, binding["space_id"])
        text = ", ".join(row["name"] for row in rows) or "No characters available."
        await interaction.response.send_message(text[:1900], ephemeral=True)

    @character.command(name="info", description="Show a character's home world")
    async def character_info(interaction: discord.Interaction, name: str):
        row = bot.store.character(interaction.guild_id, name)
        if not row:
            raise ValueError("Character not found")
        world = bot.store.space_by_id(row["world_id"])
        await interaction.response.send_message(f"{row['name']} — home world: {world['name']}", ephemeral=True)

    bot.tree.add_command(character)

    cast = app_commands.Group(name="cast", description="Manage this channel or thread's active cast")

    @cast.command(name="set", description="Set the active cast with comma-separated names")
    async def cast_set(interaction: discord.Interaction, names: str):
        parent_id, binding = await binding_for(interaction)
        eligible = {row["name"].casefold(): row for row in bot.store.eligible_characters(interaction.guild_id, binding["space_id"])}
        requested = [part.strip() for part in names.split(",") if part.strip()]
        if not requested:
            raise ValueError("Give one or more character names")
        try:
            ids = [eligible[name.casefold()]["id"] for name in requested]
        except KeyError as error:
            raise ValueError(f"Character not available here: {error.args[0]}") from error
        bot.store.set_cast(interaction.channel.id, parent_id, ids)
        await interaction.response.send_message(f"Active cast: {', '.join(requested)}", ephemeral=True)

    @cast.command(name="add", description="Add an eligible character to the active cast")
    async def cast_add(interaction: discord.Interaction, name: str):
        parent_id, binding = await binding_for(interaction)
        row = bot.store.character(interaction.guild_id, name)
        if not row:
            raise ValueError("Character not found")
        current = bot.store.get_cast(interaction.channel.id, parent_id)
        bot.store.set_cast(interaction.channel.id, parent_id, [*current, row["id"]])
        await interaction.response.send_message(f"Added {name} to the active cast.", ephemeral=True)

    @cast.command(name="remove", description="Remove a character from the active cast")
    async def cast_remove(interaction: discord.Interaction, name: str):
        parent_id, _ = await binding_for(interaction)
        row = bot.store.character(interaction.guild_id, name)
        if not row:
            raise ValueError("Character not found")
        current = bot.store.get_cast(interaction.channel.id, parent_id)
        bot.store.set_cast(interaction.channel.id, parent_id, [ident for ident in current if ident != row["id"]])
        await interaction.response.send_message(f"Removed {name} from the active cast.", ephemeral=True)

    @cast.command(name="show", description="Show this channel or thread's active cast")
    async def cast_show(interaction: discord.Interaction):
        parent_id, _ = await binding_for(interaction)
        ids = bot.store.get_cast(interaction.channel.id, parent_id)
        names = [bot.store.character_by_id(ident)["name"] for ident in ids if bot.store.character_by_id(ident)]
        await interaction.response.send_message(", ".join(names) or "The cast is empty.", ephemeral=True)

    @cast.command(name="default", description="Set the channel's default cast")
    @app_commands.checks.has_permissions(administrator=True)
    async def cast_default(interaction: discord.Interaction, names: str):
        parent_id, binding = await binding_for(interaction)
        eligible = {row["name"].casefold(): row["id"] for row in bot.store.eligible_characters(interaction.guild_id, binding["space_id"])}
        try:
            ids = [eligible[name.strip().casefold()] for name in names.split(",") if name.strip()]
        except KeyError as error:
            raise ValueError(f"Character not available here: {error.args[0]}") from error
        bot.store.set_cast(interaction.channel.id, parent_id, ids, default=True)
        await interaction.response.send_message("Channel default cast updated.", ephemeral=True)

    bot.tree.add_command(cast)

    ambient = app_commands.Group(name="ambient", description="Control opt-in ambient participation")

    @ambient.command(name="on", description="Enable ambient participation in this channel")
    @app_commands.checks.has_permissions(administrator=True)
    async def ambient_on(interaction: discord.Interaction):
        _, binding = await binding_for(interaction)
        bot.store.set_ambient(binding["channel_id"], True)
        await interaction.response.send_message("Ambient participation enabled for this channel.", ephemeral=True)

    @ambient.command(name="off", description="Disable ambient participation in this channel")
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
    async def summon(interaction: discord.Interaction, character_name: str, prompt: str):
        parent_id, binding = await binding_for(interaction)
        row = bot.store.character(interaction.guild_id, character_name)
        if not row:
            raise ValueError("Character not found")
        eligible = {item["id"] for item in bot.store.eligible_characters(interaction.guild_id, binding["space_id"])}
        if row["id"] not in eligible:
            raise ValueError("This character cannot be summoned here")
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
            await bot.run_scene(scene, interaction.channel)

    memory = app_commands.Group(name="memory", description="Control your personal character memories")

    @memory.command(name="opt_in", description="Allow characters to remember facts you explicitly state")
    async def memory_opt_in(interaction: discord.Interaction):
        bot.store.set_consent(interaction.guild_id, interaction.user.id, True)
        await interaction.response.send_message("Personal memory enabled. Use /memory list or /memory opt_out anytime.", ephemeral=True)

    @memory.command(name="opt_out", description="Disable and erase your personal memories")
    async def memory_opt_out(interaction: discord.Interaction):
        bot.store.set_consent(interaction.guild_id, interaction.user.id, False)
        await interaction.response.send_message("Personal memory disabled and your saved personal facts erased.", ephemeral=True)

    @memory.command(name="list", description="List what characters remember about you")
    async def memory_list(interaction: discord.Interaction):
        rows = bot.store.personal(interaction.guild_id, interaction.user.id)
        lines = []
        for row in rows:
            character_row = bot.store.character_by_id(row["character_id"])
            lines.append(f"#{row['id']} {character_row['name'] if character_row else 'Unknown'}: {row['content']}")
        await interaction.response.send_message("\n".join(lines)[:1900] or "No personal memories saved.", ephemeral=True)

    @memory.command(name="forget", description="Remove one of your personal memories")
    async def memory_forget(interaction: discord.Interaction, memory_id: int):
        bot.store.forget_personal(interaction.guild_id, interaction.user.id, memory_id)
        await interaction.response.send_message("Memory removed if it belonged to you.", ephemeral=True)

    bot.tree.add_command(memory)

    lore = app_commands.Group(name="lore", description="Inspect and manage local or shared lore")

    @lore.command(name="add", description="Add a local or space lore entry")
    @app_commands.checks.has_permissions(administrator=True)
    async def lore_add(interaction: discord.Interaction, content: str, keys: str = "", scope: Literal["local", "space"] = "local"):
        _, binding = await binding_for(interaction)
        scope_kind, scope_id = local_scope(interaction) if scope == "local" else ("space", binding["space_id"])
        parsed = [key.strip() for key in keys.split(",") if key.strip()]
        ident = bot.store.add_lore(interaction.guild_id, scope_kind, scope_id, content, parsed, constant=not parsed, pinned=not parsed)
        await interaction.response.send_message(f"Added {scope} lore #{ident}.", ephemeral=True)

    @lore.command(name="list", description="Show the lore available in this location")
    async def lore_list(interaction: discord.Interaction):
        _, binding = await binding_for(interaction)
        scopes = [("space", binding["space_id"]), ("channel", binding["channel_id"])]
        if isinstance(interaction.channel, discord.Thread):
            scopes.append(("thread", interaction.channel.id))
        lines = []
        for kind, ident in scopes:
            lines.extend(f"#{row['id']} [{kind}] {row['content'][:100]}" for row in bot.store.list_lore(interaction.guild_id, kind, ident))
        await interaction.response.send_message("\n".join(lines)[:1900] or "No local or space lore yet.", ephemeral=True)

    @lore.command(name="pin", description="Pin a lore entry")
    @app_commands.checks.has_permissions(administrator=True)
    async def lore_pin(interaction: discord.Interaction, lore_id: int):
        if not bot.store.lore_row(interaction.guild_id, lore_id):
            raise ValueError("Lore entry not found")
        bot.store.pin_lore(interaction.guild_id, lore_id)
        await interaction.response.send_message(f"Pinned lore #{lore_id}.", ephemeral=True)

    @lore.command(name="edit", description="Correct the text of a lore entry")
    @app_commands.checks.has_permissions(administrator=True)
    async def lore_edit(interaction: discord.Interaction, lore_id: int, content: str):
        bot.store.edit_lore(interaction.guild_id, lore_id, content)
        await interaction.response.send_message(f"Updated lore #{lore_id}.", ephemeral=True)

    @lore.command(name="promote", description="Copy lore into a channel or world/hub")
    @app_commands.checks.has_permissions(administrator=True)
    async def lore_promote(interaction: discord.Interaction, lore_id: int, destination: Literal["channel", "space"], space_name: str = ""):
        _, binding = await binding_for(interaction)
        if destination == "channel":
            target_kind, target_id = "channel", binding["channel_id"]
        else:
            target = bot.store.space(interaction.guild_id, space_name) if space_name else bot.store.space_by_id(binding["space_id"])
            if not target or target["guild_id"] != interaction.guild_id:
                raise ValueError("Destination space not found")
            target_kind, target_id = "space", target["id"]
        ident = bot.store.promote_lore(interaction.guild_id, lore_id, target_kind, target_id)
        await interaction.response.send_message(f"Promoted lore #{lore_id} to #{ident} in {target_kind}.", ephemeral=True)

    @lore.command(name="delete", description="Delete a lore entry")
    @app_commands.checks.has_permissions(administrator=True)
    async def lore_delete(interaction: discord.Interaction, lore_id: int):
        bot.store.delete_lore(interaction.guild_id, lore_id)
        await interaction.response.send_message("Lore entry deleted if present.", ephemeral=True)

    bot.tree.add_command(lore)

    scene = app_commands.Group(name="scene", description="Manage stored skit scenes")

    @scene.command(name="reset", description="Start a fresh scene from your next invitation")
    async def scene_reset(interaction: discord.Interaction):
        await binding_for(interaction)
        bot.store.reset_scene(interaction.channel.id)
        await interaction.response.send_message("Scene reset. Your next mention or summon starts fresh.", ephemeral=True)

    @scene.command(name="delete", description="Delete a stored scene and its branches")
    @app_commands.checks.has_permissions(administrator=True)
    async def scene_delete(interaction: discord.Interaction, root_message_id: str):
        await binding_for(interaction)
        try:
            root_id = int(root_message_id)
        except ValueError as error:
            raise ValueError("Give a numeric root message ID") from error
        node = bot.store.node(root_id)
        if not node or node["guild_id"] != interaction.guild_id or node["channel_id"] != interaction.channel.id:
            raise ValueError("Scene root not found in this channel")
        bot.store.delete_scene(interaction.guild_id, node["root_id"])
        await interaction.response.send_message("Stored scene and branches deleted.", ephemeral=True)

    bot.tree.add_command(scene)

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
                scope = f"book {item.get('book_name', item['scope_id'])}"
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
        elif isinstance(error, app_commands.CommandInvokeError) and isinstance(error.original, (ValueError, sqlite3.IntegrityError)):
            message = str(error.original)
        else:
            original = getattr(error, 'original', error)
            detail = error_detail(original)
            logging.error('Command failed: %s\n%s', detail, error_stack(original))
            message = "Command failed: " + detail
        if interaction.response.is_done():
            await interaction.followup.send(message[:1900], ephemeral=True)
        else:
            await interaction.response.send_message(message[:1900], ephemeral=True)
