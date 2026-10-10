"""``/favorites``: each member's ordered favorite characters and how far they may go (FEAT-23)."""
from __future__ import annotations

from types import SimpleNamespace

import discord
from discord import app_commands

from .names import NameNotFound, resolve, suggest

NO_MENTIONS = discord.AllowedMentions.none()
MODE_LABELS = {'lean': 'lean', 'step_in': 'step in'}
MODE_LINES = {
    'lean': 'Mode: lean — your favorites in the cast are more likely to answer you.',
    'step_in': 'Mode: step in — your favorites can also answer you when they are not in the cast.',
}
NO_FAVORITES = 'You have no favorites yet. Add one with /favorites add.'


def register_favorites_commands(bot, ctx: SimpleNamespace) -> None:
    require_guild, guild_characters, safe_choices = ctx.require_guild, ctx.guild_characters, ctx.safe_choices
    favorites = app_commands.Group(name='favorites', description='Keep favorite characters who are more likely to answer you')

    async def reply(interaction: discord.Interaction, text: str) -> None:
        await interaction.response.send_message(text, ephemeral=True, allowed_mentions=NO_MENTIONS)

    own_choices = safe_choices(lambda interaction, current: suggest(
        [row['name'] for row in bot.store.favorites(interaction.guild_id, interaction.user.id)], current))

    @favorites.command(name='add', description='Add a character to your favorites')
    @app_commands.describe(character='The character to add')
    @app_commands.autocomplete(character=ctx.guild_character_choices)
    async def favorites_add(interaction: discord.Interaction, character: str):
        require_guild(interaction)
        row = resolve(guild_characters(interaction.guild_id), character, 'character')
        bot.store.add_favorite(interaction.guild_id, interaction.user.id, row['id'])
        await reply(interaction, f"Added {row['name']} to your favorites.")

    @favorites.command(name='remove', description='Remove a character from your favorites')
    @app_commands.describe(character='The favorite to remove')
    @app_commands.autocomplete(character=own_choices)
    async def favorites_remove(interaction: discord.Interaction, character: str):
        require_guild(interaction)
        try:
            row = resolve(bot.store.favorites(interaction.guild_id, interaction.user.id), character, 'character')
        except NameNotFound:
            return await reply(interaction, f'{character.strip()} is not one of your favorites.')
        bot.store.remove_favorite(interaction.guild_id, interaction.user.id, row['character_id'])
        await reply(interaction, f"Removed {row['name']} from your favorites.")

    @favorites.command(name='list', description='Show your favorites and your favorites mode')
    async def favorites_list(interaction: discord.Interaction):
        require_guild(interaction)
        rows = bot.store.favorites(interaction.guild_id, interaction.user.id)
        if not rows:
            return await reply(interaction, NO_FAVORITES)
        _, binding = bot.location(interaction.channel) if interaction.channel else (None, None)
        eligible = {r['id'] for r in bot.store.eligible_characters(interaction.guild_id, binding['space_id'])} if binding else None
        lines = [f"{n}. {row['name']}" + (' (not available here)' if eligible is not None and row['character_id'] not in eligible else '')
                 for n, row in enumerate(rows, 1)]
        mode = bot.store.favorites_mode(interaction.guild_id, interaction.user.id)
        await reply(interaction, ('\n'.join(lines) + '\n' + MODE_LINES[mode])[:1900])

    @favorites.command(name='mode', description='Choose how far your favorites may go')
    @app_commands.describe(mode='Lean: favorites in the cast answer more. Step in: favorites outside the cast can answer too')
    @app_commands.choices(mode=[app_commands.Choice(name='Lean', value='lean'), app_commands.Choice(name='Step in', value='step_in')])
    async def favorites_mode(interaction: discord.Interaction, mode: str):
        require_guild(interaction)
        bot.store.set_favorites_mode(interaction.guild_id, interaction.user.id, mode)
        await reply(interaction, f'Favorites mode: {MODE_LABELS[mode]}.')

    @favorites.command(name='clear', description='Remove all your favorites')
    async def favorites_clear(interaction: discord.Interaction):
        require_guild(interaction)
        count = bot.store.clear_favorites(interaction.guild_id, interaction.user.id)
        await reply(interaction, f"Cleared {count} favorite{'' if count == 1 else 's'}.")

    bot.tree.add_command(favorites)
