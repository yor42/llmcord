"""``/favorites``: each member's ordered favorite characters and how far they may go (FEAT-23)."""
from __future__ import annotations

from types import SimpleNamespace

import discord
from discord import app_commands

from .names import NameNotFound, resolve, suggest
from llmcord_core.i18n import N_
from llmcord_core.i18n.discord_locale import C_, invoker_t

NO_MENTIONS = discord.AllowedMentions.none()
NO_FAVORITES = N_('You have no favorites yet. Add one with /favorites add.')


def register_favorites_commands(bot, ctx: SimpleNamespace) -> None:
    require_guild, safe_choices = ctx.require_guild, ctx.safe_choices
    favorites = app_commands.Group(name='favorites', description=C_('Keep favorite characters who are more likely to answer you'))

    async def reply(interaction: discord.Interaction, text: str) -> None:
        await interaction.response.send_message(text, ephemeral=True, allowed_mentions=NO_MENTIONS)

    def addable(guild_id: int):
        return bot.store.all('SELECT * FROM characters WHERE guild_id=? AND (archived=0 OR ?) ORDER BY name', (guild_id, int(bot.store.archived_favorites(guild_id))))

    add_choices = safe_choices(lambda interaction, current: suggest([row['name'] for row in addable(interaction.guild_id)], current))

    own_choices = safe_choices(lambda interaction, current: suggest(
        [row['name'] for row in bot.store.favorites(interaction.guild_id, interaction.user.id)], current))

    @favorites.command(name='add', description=C_('Add a character to your favorites'))
    @app_commands.describe(character=C_('The character to add'))
    @app_commands.autocomplete(character=add_choices)
    async def favorites_add(interaction: discord.Interaction, character: str):
        require_guild(interaction)
        t = invoker_t(interaction)
        row = resolve(addable(interaction.guild_id), character, 'character')
        bot.store.add_favorite(interaction.guild_id, interaction.user.id, row['id'])
        await reply(interaction, t('Added {name} to your favorites.', name=row['name']))

    @favorites.command(name='remove', description=C_('Remove a character from your favorites'))
    @app_commands.describe(character=C_('The favorite to remove'))
    @app_commands.autocomplete(character=own_choices)
    async def favorites_remove(interaction: discord.Interaction, character: str):
        require_guild(interaction)
        t = invoker_t(interaction)
        try:
            row = resolve(bot.store.favorites(interaction.guild_id, interaction.user.id), character, 'character')
        except NameNotFound:
            return await reply(interaction, t('{name} is not one of your favorites.', name=character.strip()))
        bot.store.remove_favorite(interaction.guild_id, interaction.user.id, row['character_id'])
        await reply(interaction, t('Removed {name} from your favorites.', name=row['name']))

    @favorites.command(name='list', description=C_('Show your favorites and your favorites mode'))
    async def favorites_list(interaction: discord.Interaction):
        require_guild(interaction)
        t = invoker_t(interaction)
        rows = bot.store.favorites(interaction.guild_id, interaction.user.id)
        if not rows:
            return await reply(interaction, t(NO_FAVORITES))
        _, binding = bot.location(interaction.channel) if interaction.channel else (None, None)
        eligible = {r['character_id'] for r in bot.store.eligible_favorites(interaction.guild_id, interaction.user.id, binding['space_id'])} if binding else None
        lines = []
        for n, row in enumerate(rows, 1):
            archived, away = bool(row['archived']), eligible is not None and row['character_id'] not in eligible
            if archived and away:
                lines.append(t('{n}. {name} (archived) (not available here)', n=n, name=row['name']))
            elif archived:
                lines.append(t('{n}. {name} (archived)', n=n, name=row['name']))
            elif away:
                lines.append(t('{n}. {name} (not available here)', n=n, name=row['name']))
            else:
                lines.append(t('{n}. {name}', n=n, name=row['name']))
        mode = bot.store.favorites_mode(interaction.guild_id, interaction.user.id)
        if mode == 'lean':
            mode_line = t('Mode: lean — your favorites in the cast are more likely to answer you.')
        else:
            mode_line = t('Mode: step in — your favorites can also answer you when they are not in the cast.')
        await reply(interaction, ('\n'.join(lines) + '\n' + mode_line)[:1900])

    @favorites.command(name='mode', description=C_('Choose how far your favorites may go'))
    @app_commands.describe(mode=C_('Lean: favorites in the cast answer more. Step in: favorites outside the cast can answer too'))
    @app_commands.choices(mode=[app_commands.Choice(name=C_('Lean'), value='lean'), app_commands.Choice(name=C_('Step in'), value='step_in')])
    async def favorites_mode(interaction: discord.Interaction, mode: str):
        require_guild(interaction)
        t = invoker_t(interaction)
        bot.store.set_favorites_mode(interaction.guild_id, interaction.user.id, mode)
        await reply(interaction, t('Favorites mode: lean.') if mode == 'lean' else t('Favorites mode: step in.'))

    @favorites.command(name='clear', description=C_('Remove all your favorites'))
    async def favorites_clear(interaction: discord.Interaction):
        require_guild(interaction)
        t = invoker_t(interaction)
        count = bot.store.clear_favorites(interaction.guild_id, interaction.user.id)
        await reply(interaction, t.n('Cleared {count} favorite.', 'Cleared {count} favorites.', count))

    bot.tree.add_command(favorites)
