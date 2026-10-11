"""``/shop``, ``/inventory`` and ``/use``: buying and using toy-shop items (FEAT-22 part C). No model call; every store call is scoped by guild."""
from __future__ import annotations

import logging
import time
import zlib
from types import SimpleNamespace

import discord
from discord import app_commands

from .errors import error_detail
from .items_store import AlreadyOwned, HoldCapReached, ItemError, NotForSale, NotHeld, OnCooldown, PriceChanged, UseDisabled

NO_MENTIONS = discord.AllowedMentions.none()
VIEW_SECONDS = 300
MAX_BUTTONS = 25
MAX_FIELDS = 25
EMBED_BUDGET = 5800
LABEL_LIMIT = 80
PALETTE = (0x5865F2, 0x57F287, 0xFEE75C, 0xEB459E, 0xED4245, 0x3498DB, 0x9B59B6, 0xE67E22)
DESCRIPTIONS = {
    '8ball': 'Ask a question and get an answer.',
    'fortune_cookie': 'Crack it open for a fortune and a lucky number.',
    'dice': 'Roll a die.',
    'd20': 'Roll a twenty-sided die.',
    'tarot': 'Draw a tarot card.',
}
DEFAULT_DESCRIPTION = 'A toy from the shop.'
EMPTY_SHOP = 'Nothing is for sale on this server right now.'
EMPTY_INVENTORY = 'You are not holding any items. Visit /shop to buy some.'
NOT_YOURS = 'This menu belongs to someone else. Run the command yourself to get your own.'
SEND_FAILED = 'Your {name} was used, but the result could not be posted. It is saved in the use log an admin can see.'
SOMETHING_WRONG = 'Something went wrong with {name}. Check /inventory and try again.'
BOUGHT_STALE = 'The menu could not be refreshed. Open /shop again.'


def color_for(key: str) -> int:
    return PALETTE[zlib.crc32(key.encode()) % len(PALETTE)]


def clip(text: str, limit: int = LABEL_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit - 1] + '…'


def fit_embed(embed: discord.Embed, total: int) -> int:
    """Keeps at most 25 fields and the embed under the size budget, noting what was left out. Returns how many fields stay."""
    while len(embed.fields) > MAX_FIELDS or (len(embed) > EMBED_BUDGET and embed.fields):
        embed.remove_field(len(embed.fields) - 1)
    hidden = total - len(embed.fields)
    if hidden:
        embed.description = f'{embed.description}\n+{hidden} more not shown' if embed.description else f'+{hidden} more not shown'
    return len(embed.fields)


def kind_word(consumable: bool) -> str:
    return 'consumable' if consumable else 'reusable'


def buy_error(error: ItemError, name: str) -> str:
    if isinstance(error, NotForSale):
        return f'{name} is not for sale right now.'
    if isinstance(error, AlreadyOwned):
        return f'You already own {name}.'
    if isinstance(error, PriceChanged):
        return str(error)
    if isinstance(error, HoldCapReached):
        return f'You can hold at most {error.cap} of {name}.'
    return str(error)


def use_error(error: ItemError, name: str) -> str:
    if isinstance(error, NotHeld):
        return f'You do not have {name}. Buy one with /shop.'
    if isinstance(error, UseDisabled):
        return f'{name} cannot be used on this server right now.'
    if isinstance(error, OnCooldown):
        return f'{name} is recharging. Try again in {error.remaining} s.'
    return str(error)


def result_embed(user, entry: dict, outcome: dict) -> discord.Embed:
    result = outcome['result']
    embed = discord.Embed(title=f"{entry['emoji']} {entry['name']}", description=result.text, color=color_for(entry['key']))
    for name, value in result.fields:
        embed.add_field(name=name, value=value, inline=True)
    parts = ['Toy shop']
    if result.footer_note:
        parts.append(result.footer_note)
    if entry['consumable']:
        parts.append(f"{outcome['qty']} left")
    embed.set_footer(text=' · '.join(parts))
    embed.set_author(name=getattr(user, 'display_name', None) or 'Someone')
    return embed


def register_items_commands(bot, ctx: SimpleNamespace) -> None:
    require_guild, safe_choices = ctx.require_guild, ctx.safe_choices
    store = bot.store

    def sale_state(entry: dict, qty: int, cap: int, balance: int, currency: str) -> tuple[str, bool]:
        """The Buy button's label and whether it is disabled."""
        if not entry['consumable'] and qty >= 1:
            return 'Owned', True
        if entry['consumable'] and qty >= cap:
            return f'At the limit ({cap})', True
        if balance < entry['price']:
            return clip(f"Need {entry['price'] - balance:,} {currency} more"), True
        return clip(f"Buy {entry['emoji']} {entry['name']} · {entry['price']:,}"), False

    def shop_embed(guild_id: int, user_id: int):
        currency, balance, cap = store.currency_name(guild_id), store.balance(guild_id, user_id), store.item_hold_cap(guild_id)
        listed = [e for e in store.shop_catalog(guild_id) if e['for_sale'] and e['price'] >= 1]
        embed = discord.Embed(title='Toy shop', color=PALETTE[0])
        if not listed:
            embed.description = f'{EMPTY_SHOP}\nYour balance: {balance:,} {currency}'
            return embed, []
        embed.description = f'Your balance: {balance:,} {currency}'
        rows = []
        for entry in listed:
            qty = store.item_qty(guild_id, user_id, entry['key'])
            held = f'You hold {qty}' if qty else 'You hold none'
            description = DESCRIPTIONS.get(entry['key'], DEFAULT_DESCRIPTION)
            embed.add_field(name=f"{entry['emoji']} {entry['name']} · {entry['price']:,} {currency}",
                            value=f"{description}\n{kind_word(entry['consumable'])} · {held}", inline=False)
            rows.append((entry, *sale_state(entry, qty, cap, balance, currency)))
        return embed, rows[:fit_embed(embed, len(rows))]

    def inventory_embed(guild_id: int, user_id: int):
        held = store.inventory(guild_id, user_id)
        if not held:
            return None, []
        now, rows = time.time(), []
        embed = discord.Embed(title='Your items', color=PALETTE[0])
        for item in held:
            entry = store.shop_item(guild_id, item['key'])
            state = [kind_word(item['consumable'])]
            if not entry['allow_use']:
                state.append('use is off right now')
            elif not item['consumable'] and entry['cooldown_seconds'] and item['last_used_at'] is not None:
                ready = item['last_used_at'] + entry['cooldown_seconds']
                state.append(f'recharging, ready <t:{int(ready) + 1}:R>' if now < ready else 'ready')
            elif not item['consumable']:
                state.append('ready')
            embed.add_field(name=f"{item['emoji']} {item['name']} × {item['qty']}", value=' · '.join(state), inline=False)
            rows.append((item, entry))
        return embed, rows[:fit_embed(embed, len(rows))]

    async def send_private(interaction: discord.Interaction, text: str) -> None:
        if interaction.response.is_done():
            await interaction.followup.send(text, ephemeral=True, allowed_mentions=NO_MENTIONS)
        else:
            await interaction.response.send_message(text, ephemeral=True, allowed_mentions=NO_MENTIONS)

    async def use_and_post(interaction: discord.Interaction, guild_id: int, key: str, name: str, via_channel: bool = False) -> bool:
        """Uses the item and posts the result publicly: through the slash response, or (from a menu button) with a channel message.
        Refusals and send failures are shown only to the member. True when the item was used."""
        try:
            outcome = store.use_item(guild_id, interaction.user.id, interaction.channel.id if interaction.channel else 0, key)
        except ItemError as error:
            await send_private(interaction, use_error(error, name))
            return False
        embed = result_embed(interaction.user, outcome['item'], outcome)
        try:
            if via_channel:
                await interaction.channel.send(embed=embed, allowed_mentions=NO_MENTIONS)
            else:
                await interaction.response.send_message(embed=embed, allowed_mentions=NO_MENTIONS)
        except Exception as error:
            logging.warning('Could not post the %s result (use %s): %s', key, outcome['use_id'], error_detail(error))
            try:
                await send_private(interaction, SEND_FAILED.format(name=name))
            except Exception as again:
                logging.warning('Could not tell the member about use %s either: %s', outcome['use_id'], error_detail(again))
        return True

    class OwnedView(discord.ui.View):
        """Base for the member's own menus: only they may press, and it disables itself on timeout."""

        def __init__(self, origin: discord.Interaction, guild_id: int):
            super().__init__(timeout=VIEW_SECONDS)
            self.origin, self.guild_id, self.user_id = origin, guild_id, origin.user.id

        async def interaction_check(self, interaction: discord.Interaction) -> bool:
            if interaction.user.id == self.user_id and interaction.guild_id == self.guild_id:
                return True
            await interaction.response.send_message(NOT_YOURS, ephemeral=True, allowed_mentions=NO_MENTIONS)
            return False

        async def on_timeout(self) -> None:
            for child in self.children:
                child.disabled = True
            try:
                await self.origin.edit_original_response(view=self)
            except Exception as error:
                logging.info('Could not disable an expired items menu: %s', error_detail(error))

    class BuyButton(discord.ui.Button):
        def __init__(self, entry: dict, label: str, disabled: bool, row: int):
            super().__init__(label=label, style=discord.ButtonStyle.success, disabled=disabled, row=row)
            self.entry = entry

        async def callback(self, interaction: discord.Interaction):
            view: ShopView = self.view
            entry, bought, line = self.entry, False, ''
            try:
                try:
                    done = store.buy_item(view.guild_id, view.user_id, entry['key'], expected_price=entry['price'])
                    bought = True
                    line = f"Bought {entry['emoji']} {entry['name']} for {done['price']:,} {store.currency_name(view.guild_id)}."
                except ItemError as error:
                    line = buy_error(error, entry['name'])
                embed, new_view = build_shop(view.origin, view.guild_id)
                await interaction.response.edit_message(content=line, embed=embed, view=new_view)
                view.stop()
            except Exception as error:
                logging.warning('Shop menu failed after %s (bought=%s): %s', entry['key'], bought, error_detail(error))
                try:
                    await send_private(interaction, f'{line} {BOUGHT_STALE}' if bought else SOMETHING_WRONG.format(name=entry['name']))
                except Exception as again:
                    logging.warning('Could not tell the member about the failed shop menu either: %s', error_detail(again))

    class ShopView(OwnedView):
        pass

    def build_shop(origin: discord.Interaction, guild_id: int):
        embed, rows = shop_embed(guild_id, origin.user.id)
        view = ShopView(origin, guild_id)
        for index, (entry, label, disabled) in enumerate(rows):
            view.add_item(BuyButton(entry, label, disabled, index // 5))
        return embed, view

    class UseButton(discord.ui.Button):
        def __init__(self, item: dict, row: int):
            super().__init__(label=clip(f"Use {item['emoji']} {item['name']}"), style=discord.ButtonStyle.primary, row=row)
            self.item = item

        async def callback(self, interaction: discord.Interaction):
            view: InventoryView = self.view
            name = self.item['name']
            try:
                await interaction.response.defer(ephemeral=True)
                await use_and_post(interaction, view.guild_id, self.item['key'], name, via_channel=True)
                embed, new_view = build_inventory(view.origin, view.guild_id)
                if new_view is None:
                    await interaction.edit_original_response(content=EMPTY_INVENTORY, embed=None, view=None)
                else:
                    await interaction.edit_original_response(content=None, embed=embed, view=new_view)
                view.stop()
            except Exception as error:
                logging.warning('Inventory menu failed on %s: %s', self.item['key'], error_detail(error))
                try:
                    await send_private(interaction, SOMETHING_WRONG.format(name=name))
                except Exception as again:
                    logging.warning('Could not tell the member about the failed inventory menu either: %s', error_detail(again))

    class InventoryView(OwnedView):
        pass

    def build_inventory(origin: discord.Interaction, guild_id: int):
        embed, rows = inventory_embed(guild_id, origin.user.id)
        if embed is None:
            return None, None
        view = InventoryView(origin, guild_id)
        usable = [item for item, entry in rows if entry['allow_use']][:MAX_BUTTONS]
        for index, item in enumerate(usable):
            view.add_item(UseButton(item, index // 5))
        return embed, view

    @bot.tree.command(name='shop', description='See what is for sale and buy items')
    async def shop(interaction: discord.Interaction):
        require_guild(interaction)
        embed, view = build_shop(interaction, interaction.guild_id)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True, allowed_mentions=NO_MENTIONS)

    @bot.tree.command(name='inventory', description='See the items you hold and use them')
    async def inventory(interaction: discord.Interaction):
        require_guild(interaction)
        embed, view = build_inventory(interaction, interaction.guild_id)
        if embed is None:
            return await interaction.response.send_message(EMPTY_INVENTORY, ephemeral=True, allowed_mentions=NO_MENTIONS)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True, allowed_mentions=NO_MENTIONS)

    def held_choices(interaction: discord.Interaction, current: str):
        needle = current.strip().casefold()
        held = store.inventory(interaction.guild_id, interaction.user.id)
        return [app_commands.Choice(name=f"{i['emoji']} {i['name']}"[:100], value=i['name'])
                for i in held if needle in i['name'].casefold()][:25]

    @bot.tree.command(name='use', description='Use an item you hold')
    @app_commands.describe(item='The item to use')
    @app_commands.autocomplete(item=safe_choices(held_choices))
    async def use(interaction: discord.Interaction, item: str):
        require_guild(interaction)
        wanted = item.strip().casefold()
        held = store.inventory(interaction.guild_id, interaction.user.id)
        match = next((i for i in held if i['name'].casefold() == wanted or i['key'] == wanted), None)
        if match is None:
            return await send_private(interaction, f'You do not have {item.strip() or "that item"}. Buy one with /shop.')
        await use_and_post(interaction, interaction.guild_id, match['key'], match['name'])
