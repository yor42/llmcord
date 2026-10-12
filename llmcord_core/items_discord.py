"""``/shop``, ``/inventory`` and ``/use``: buying and using toy-shop items (FEAT-22 part C). No model call; every store call is scoped by guild."""
from __future__ import annotations

import copy
import logging
import time
import zlib
from types import SimpleNamespace

import discord
from discord import app_commands

from .errors import error_detail
from .items_store import AlreadyOwned, HoldCapReached, ItemError, NotForSale, NotHeld, OnCooldown, PriceChanged, UseDisabled
from llmcord_core import i18n, items
from llmcord_core.i18n import N_
from llmcord_core.i18n.discord_locale import C_, channel_t, invoker_t

NO_MENTIONS = discord.AllowedMentions.none()
VIEW_SECONDS = 300
MAX_BUTTONS = 25
MAX_FIELDS = 25
EMBED_BUDGET = 5800
LABEL_LIMIT = 80
PALETTE = (0x5865F2, 0x57F287, 0xFEE75C, 0xEB459E, 0xED4245, 0x3498DB, 0x9B59B6, 0xE67E22)
EMPTY_SHOP = N_('Nothing is for sale on this server right now.')
EMPTY_INVENTORY = N_('You are not holding any items. Visit /shop to buy some.')
NOT_YOURS = N_('This menu belongs to someone else. Run the command yourself to get your own.')
SEND_FAILED = N_('Your {name} was used, but the result could not be posted. It is saved in the use log an admin can see.')
SOMETHING_WRONG = N_('Something went wrong with {name}. Check /inventory and try again.')
BOUGHT_STALE = N_('The menu could not be refreshed. Open /shop again.')


def color_for(key: str) -> int:
    return PALETTE[zlib.crc32(key.encode()) % len(PALETTE)]


def clip(text: str, limit: int = LABEL_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit - 1] + '…'


def fit_embed(embed: discord.Embed, total: int, *, t=None) -> int:
    """Keeps at most 25 fields and the embed under the size budget, noting what was left out. Returns how many fields stay."""
    while len(embed.fields) > MAX_FIELDS or (len(embed) > EMBED_BUDGET and embed.fields):
        embed.remove_field(len(embed.fields) - 1)
    t = t or i18n.english()
    hidden = total - len(embed.fields)
    if hidden:
        more = t('+{hidden} more not shown', hidden=hidden)
        embed.description = f'{embed.description}\n{more}' if embed.description else more
    return len(embed.fields)


def kind_word(consumable: bool, *, t=None) -> str:
    t = t or i18n.english()
    return t.p('item kind', 'consumable') if consumable else t.p('item kind', 'reusable')


def item_description(key: str, *, t=None) -> str:
    t = t or i18n.english()
    if key == '8ball':
        return t('Ask a question and get an answer.')
    if key == 'fortune_cookie':
        return t('Crack it open for a fortune and a lucky number.')
    if key == 'dice':
        return t('Roll a die.')
    if key == 'd20':
        return t('Roll a twenty-sided die.')
    if key == 'tarot':
        return t('Draw a tarot card.')
    return t('A toy from the shop.')


def shown_name(entry: dict, t) -> str:
    """The item's name for the audience; the entry's own (English) name when the recipe is unknown."""
    try:
        return items.display_name(items.get_item(entry['key']), t.locale)
    except KeyError:
        return entry['name']


def error_name(error, name: str, t) -> str:
    """`name` unless the error names a recipe (params['item']); then that recipe's name for the audience."""
    key = (getattr(error, 'params', None) or {}).get('item')
    try:
        return items.display_name(items.get_item(key), t.locale) if key else name
    except (KeyError, TypeError):
        return name


def error_text(error, t) -> str:
    """t.error(error) with params['name'] replaced by the audience's item name."""
    params = getattr(error, 'params', None)
    if isinstance(params, dict) and params.get('item') and 'name' in params:
        error = copy.copy(error)
        error.params = {**params, 'name': error_name(error, params['name'], t)}
    return t.error(error)


def buy_error(error: ItemError, name: str, *, t=None) -> str:
    t = t or i18n.english()
    name = error_name(error, name, t)
    if isinstance(error, NotForSale):
        return t('{name} is not for sale right now.', name=name)
    if isinstance(error, AlreadyOwned):
        return t('You already own {name}.', name=name)
    if isinstance(error, PriceChanged):
        return error_text(error, t)
    if isinstance(error, HoldCapReached):
        return t('You can hold at most {cap} of {name}.', cap=error.cap, name=name)
    return error_text(error, t)


def use_error(error: ItemError, name: str, *, t=None) -> str:
    t = t or i18n.english()
    name = error_name(error, name, t)
    if isinstance(error, NotHeld):
        return t('You do not have {name}. Buy one with /shop.', name=name)
    if isinstance(error, UseDisabled):
        return t('{name} cannot be used on this server right now.', name=name)
    if isinstance(error, OnCooldown):
        return t('{name} is recharging. Try again in {remaining} s.', name=name, remaining=error.remaining)
    return error_text(error, t)


def result_embed(user, entry: dict, outcome: dict, *, t=None) -> discord.Embed:
    t = t or i18n.english()
    result = outcome['result']
    embed = discord.Embed(title=f"{entry['emoji']} {shown_name(entry, t)}", description=result.text, color=color_for(entry['key']))
    for name, value in result.fields:
        embed.add_field(name=name, value=value, inline=True)
    parts = [t('Toy shop')]
    if result.footer_note:
        parts.append(result.footer_note)
    if entry['consumable']:
        parts.append(t('{qty} left', qty=outcome['qty']))
    embed.set_footer(text=' · '.join(parts))
    embed.set_author(name=getattr(user, 'display_name', None) or t('Someone'))
    return embed


def register_items_commands(bot, ctx: SimpleNamespace) -> None:
    require_guild, safe_choices = ctx.require_guild, ctx.safe_choices
    store = bot.store

    def sale_state(entry: dict, qty: int, cap: int, balance: int, currency: str, t) -> tuple[str, bool]:
        """The Buy button's label and whether it is disabled."""
        if not entry['consumable'] and qty >= 1:
            return t.p('button.shop', 'Owned'), True
        if entry['consumable'] and qty >= cap:
            return t.p('button.shop', 'At the limit ({cap})', cap=cap), True
        if balance < entry['price']:
            return clip(t.p('button.shop', 'Need {short:,} {currency} more', short=entry['price'] - balance, currency=currency)), True
        name = shown_name(entry, t)
        return clip(t.p('button.shop', 'Buy {emoji} {name} · {price:,}', emoji=entry['emoji'], name=name, price=entry['price'])), False

    def shop_embed(guild_id: int, user_id: int, t):
        currency, balance, cap = store.currency_name(guild_id), store.balance(guild_id, user_id), store.item_hold_cap(guild_id)
        listed = [e for e in store.shop_catalog(guild_id) if e['for_sale'] and e['price'] >= 1]
        embed = discord.Embed(title=t('Toy shop'), color=PALETTE[0])
        if not listed:
            embed.description = f"{t(EMPTY_SHOP)}\n{t('Your balance: {balance:,} {currency}', balance=balance, currency=currency)}"
            return embed, []
        embed.description = t('Your balance: {balance:,} {currency}', balance=balance, currency=currency)
        rows = []
        for entry in listed:
            qty = store.item_qty(guild_id, user_id, entry['key'])
            held = t('You hold {qty}', qty=qty) if qty else t('You hold none')
            description = item_description(entry['key'], t=t)
            embed.add_field(name=f"{entry['emoji']} {shown_name(entry, t)} · {entry['price']:,} {currency}",
                            value=f"{description}\n{kind_word(entry['consumable'], t=t)} · {held}", inline=False)
            rows.append((entry, *sale_state(entry, qty, cap, balance, currency, t)))
        return embed, rows[:fit_embed(embed, len(rows), t=t)]

    def inventory_embed(guild_id: int, user_id: int, t):
        held = store.inventory(guild_id, user_id)
        if not held:
            return None, []
        now, rows = time.time(), []
        embed = discord.Embed(title=t('Your items'), color=PALETTE[0])
        for item in held:
            entry = store.shop_item(guild_id, item['key'])
            state = [kind_word(item['consumable'], t=t)]
            if not entry['allow_use']:
                state.append(t('use is off right now'))
            elif not item['consumable'] and entry['cooldown_seconds'] and item['last_used_at'] is not None:
                ready = item['last_used_at'] + entry['cooldown_seconds']
                state.append(t('recharging, ready {when}', when=f'<t:{int(ready) + 1}:R>') if now < ready else t('ready'))
            elif not item['consumable']:
                state.append(t('ready'))
            embed.add_field(name=f"{item['emoji']} {shown_name(item, t)} × {item['qty']}", value=' · '.join(state), inline=False)
            rows.append((item, entry))
        return embed, rows[:fit_embed(embed, len(rows), t=t)]

    async def send_private(interaction: discord.Interaction, text: str) -> None:
        if interaction.response.is_done():
            await interaction.followup.send(text, ephemeral=True, allowed_mentions=NO_MENTIONS)
        else:
            await interaction.response.send_message(text, ephemeral=True, allowed_mentions=NO_MENTIONS)

    async def use_and_post(interaction: discord.Interaction, guild_id: int, key: str, name: str, via_channel: bool = False) -> bool:
        """Uses the item and posts the result publicly: through the slash response, or (from a menu button) with a channel message.
        Refusals and send failures are shown only to the member. True when the item was used."""
        t = invoker_t(interaction)
        channel = channel_t(interaction.guild)
        name = shown_name({'key': key, 'name': name}, t)
        try:
            outcome = store.use_item(guild_id, interaction.user.id, interaction.channel.id if interaction.channel else 0, key, locale=channel.locale)
        except ItemError as error:
            await send_private(interaction, use_error(error, name, t=t))
            return False
        embed = result_embed(interaction.user, outcome['item'], outcome, t=channel)
        try:
            if via_channel:
                await interaction.channel.send(embed=embed, allowed_mentions=NO_MENTIONS)
            else:
                await interaction.response.send_message(embed=embed, allowed_mentions=NO_MENTIONS)
        except Exception as error:
            logging.warning('Could not post the %s result (use %s): %s', key, outcome['use_id'], error_detail(error))
            try:
                await send_private(interaction, t(SEND_FAILED, name=name))
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
            await interaction.response.send_message(invoker_t(interaction)(NOT_YOURS), ephemeral=True, allowed_mentions=NO_MENTIONS)
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
            t = invoker_t(interaction)
            entry, bought, line = self.entry, False, ''
            try:
                try:
                    done = store.buy_item(view.guild_id, view.user_id, entry['key'], expected_price=entry['price'])
                    bought = True
                    line = t('Bought {emoji} {name} for {price:,} {currency}.', emoji=entry['emoji'], name=shown_name(entry, t), price=done['price'], currency=store.currency_name(view.guild_id))
                except ItemError as error:
                    line = buy_error(error, shown_name(entry, t), t=t)
                embed, new_view = build_shop(view.origin, view.guild_id, t)
                await interaction.response.edit_message(content=line, embed=embed, view=new_view)
                view.stop()
            except Exception as error:
                logging.warning('Shop menu failed after %s (bought=%s): %s', entry['key'], bought, error_detail(error))
                try:
                    await send_private(interaction, f'{line} {t(BOUGHT_STALE)}' if bought else t(SOMETHING_WRONG, name=shown_name(entry, t)))
                except Exception as again:
                    logging.warning('Could not tell the member about the failed shop menu either: %s', error_detail(again))

    class ShopView(OwnedView):
        pass

    def build_shop(origin: discord.Interaction, guild_id: int, t):
        embed, rows = shop_embed(guild_id, origin.user.id, t)
        view = ShopView(origin, guild_id)
        for index, (entry, label, disabled) in enumerate(rows):
            view.add_item(BuyButton(entry, label, disabled, index // 5))
        return embed, view

    class UseButton(discord.ui.Button):
        def __init__(self, item: dict, row: int, t):
            shown = shown_name(item, t)
            super().__init__(label=clip(t.p('button.inventory', 'Use {emoji} {name}', emoji=item['emoji'], name=shown)), style=discord.ButtonStyle.primary, row=row)
            self.item = item

        async def callback(self, interaction: discord.Interaction):
            view: InventoryView = self.view
            t = invoker_t(interaction)
            name = shown_name(self.item, t)
            try:
                await interaction.response.defer(ephemeral=True)
                await use_and_post(interaction, view.guild_id, self.item['key'], name, via_channel=True)
                embed, new_view = build_inventory(view.origin, view.guild_id, t)
                if new_view is None:
                    await interaction.edit_original_response(content=t(EMPTY_INVENTORY), embed=None, view=None)
                else:
                    await interaction.edit_original_response(content=None, embed=embed, view=new_view)
                view.stop()
            except Exception as error:
                logging.warning('Inventory menu failed on %s: %s', self.item['key'], error_detail(error))
                try:
                    await send_private(interaction, t(SOMETHING_WRONG, name=name))
                except Exception as again:
                    logging.warning('Could not tell the member about the failed inventory menu either: %s', error_detail(again))

    class InventoryView(OwnedView):
        pass

    def build_inventory(origin: discord.Interaction, guild_id: int, t):
        embed, rows = inventory_embed(guild_id, origin.user.id, t)
        if embed is None:
            return None, None
        view = InventoryView(origin, guild_id)
        usable = [item for item, entry in rows if entry['allow_use']][:MAX_BUTTONS]
        for index, item in enumerate(usable):
            view.add_item(UseButton(item, index // 5, t))
        return embed, view

    @bot.tree.command(name='shop', description=C_('See what is for sale and buy items'))
    async def shop(interaction: discord.Interaction):
        require_guild(interaction)
        embed, view = build_shop(interaction, interaction.guild_id, invoker_t(interaction))
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True, allowed_mentions=NO_MENTIONS)

    @bot.tree.command(name='inventory', description=C_('See the items you hold and use them'))
    async def inventory(interaction: discord.Interaction):
        require_guild(interaction)
        t = invoker_t(interaction)
        embed, view = build_inventory(interaction, interaction.guild_id, t)
        if embed is None:
            return await interaction.response.send_message(t(EMPTY_INVENTORY), ephemeral=True, allowed_mentions=NO_MENTIONS)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True, allowed_mentions=NO_MENTIONS)

    def held_choices(interaction: discord.Interaction, current: str):
        needle, t = current.strip().casefold(), invoker_t(interaction)
        held = store.inventory(interaction.guild_id, interaction.user.id)
        return [app_commands.Choice(name=f"{i['emoji']} {shown_name(i, t)}"[:100], value=i['name'])
                for i in held if needle in i['name'].casefold() or needle in shown_name(i, t).casefold()][:25]

    @bot.tree.command(name='use', description=C_('Use an item you hold'))
    @app_commands.describe(item=C_('The item to use'))
    @app_commands.autocomplete(item=safe_choices(held_choices))
    async def use(interaction: discord.Interaction, item: str):
        require_guild(interaction)
        t = invoker_t(interaction)
        wanted = item.strip().casefold()
        held = store.inventory(interaction.guild_id, interaction.user.id)
        match = next((i for i in held if wanted in (i['name'].casefold(), shown_name(i, t).casefold(), i['key'])), None)
        if match is None:
            return await send_private(interaction, t('You do not have {name}. Buy one with /shop.', name=item.strip() or t('that item')))
        await use_and_post(interaction, interaction.guild_id, match['key'], match['name'])
