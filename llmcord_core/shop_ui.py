"""Dashboard Shop tab (FEAT-22 part D): the built-in items (price, availability, cooldown, custom lines), the hold cap and the use log.

Custom lines and results are admin or roll text: shown as plain text only, and never written to the audit log.
"""
from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

from . import i18n, items
from .currency_ui import CurrencyPanel, member_label, table
from .i18n import web_locale

PAGE = 50
LOCALE = items.recipe.DEFAULT_LOCALE


def whole(value):
    """A number input's value as an int when it is a whole number; anything else reaches the store as-is so its message is shown."""
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value == int(value) else value


def has_plain_table(recipe):
    """True when custom lines apply: a table of plain strings (not a range or card records, as in tarot, dice and d20)."""
    return any(isinstance(t, dict) and 'range' not in t and all(isinstance(r, str) for r in t.get(LOCALE, ())) and t.get(LOCALE)
               for t in recipe.tables.values())


def lines_of(text):
    return [line.strip() for line in (text or '').splitlines() if line.strip()]


def with_lines(existing, lines):
    """The locale-keyed entries with the default locale replaced by `lines` (removed when empty); other locales are kept."""
    out = {k: v for k, v in (existing or {}).items() if k != LOCALE}
    if lines:
        out[LOCALE] = lines
    return out


def save_item(store, gid, entry, *, for_sale, allow_use, price, cooldown, sides, extra_text, replace_text):
    """Save one built-in item's settings for the server; `entry` is the shop_item the form was built from (its revision guards the write)."""
    recipe = items.get_item(entry['key'])
    params = {}
    if 'sides' in recipe.params and whole(sides) != recipe.params['sides']:
        params['sides'] = whole(sides)
    return store.save_shop_item(gid, entry['key'], expected_revision=entry['revision'], for_sale=bool(for_sale), allow_use=bool(allow_use),
                                price=whole(price), cooldown_seconds=None if recipe.consumable else whole(cooldown), params=params,
                                extra_entries=with_lines(entry['extra_entries'], lines_of(extra_text)) if has_plain_table(recipe) else entry['extra_entries'],
                                replace_entries=with_lines(entry['replace_entries'], lines_of(replace_text)) if has_plain_table(recipe) else entry['replace_entries'])


def item_detail(result):
    """Audit detail for a shop.item save: settings and counts only, never the custom line text."""
    count = lambda entries: sum(len(v) for v in entries.values())
    return {'item_key': result['key'], 'for_sale': result['for_sale'], 'allow_use': result['allow_use'], 'price': result['price_override'],
            'cooldown_seconds': result['cooldown_override'], 'params': dict(result['params']),
            'extra_lines': count(result['extra_entries']), 'replace_lines': count(result['replace_entries']), 'revision': result['revision']}


def use_rows(uses, names, zone_name, *, t=None):
    """Display rows for item_uses entries (newest first)."""
    t = t or i18n.english()
    try:
        zone = ZoneInfo(zone_name or 'UTC')
    except Exception:
        zone = ZoneInfo('UTC')
    rows = []
    for use in uses:
        try:
            recipe = items.get_item(use['item_key'])
            item = f'{recipe.emoji} {recipe.name}'.strip()
        except (KeyError, TypeError):
            item = use['item_key']
        text = ' '.join(str((use['result'] or {}).get('text', '')).replace('**', '').split())
        rows.append({'id': use['id'], 'time': datetime.fromtimestamp(use['used_at'], zone).strftime('%Y-%m-%d %H:%M'),
                     'member': member_label(names, use['user_id'], t=t), 'item': item, 'result': text if len(text) <= 200 else text[:199] + '…'})
    return rows


class ShopPanel:
    lookup = CurrencyPanel.lookup  # the same capped member-name lookup the Currency tab uses

    def __init__(self, ctx):
        self.ctx, self.store, self.gid = ctx, ctx.store, ctx.guild_id
        self.live = ctx.live('shop')
        self.names, self.absent, self.warned = {}, set(), False
        self.uses, self.more = [], False

    def build(self):
        from nicegui import ui
        from .dashboard import section
        t = web_locale.current()
        with section(t('Items')):
            ui.label(t('The built-in items. Members buy them with /shop and use them with /use. A price left blank uses the built-in price. Changes apply at once.')).classes('ll-muted')
            name = self.store.currency_name(self.gid)
            for entry in self.store.shop_catalog(self.gid):
                self.item_card(entry, name)
        with section(t('Settings')):
            self.settings()
        with section(t('Use log')):
            ui.label(t('Every item use on this server, newest first.')).classes('ll-muted')
            self.log_body = ui.column().classes('w-full gap-2')

    def item_card(self, entry, currency):
        from nicegui import ui
        t = web_locale.current()
        ctx, store, gid = self.ctx, self.store, self.gid
        recipe = items.get_item(entry['key'])
        state = {'entry': entry}
        card = ui.element('div').classes('ll-stack ll-result w-full').props('role=group')
        card._props['aria-label'] = recipe.name
        with card:
            ui.label(f'{recipe.emoji} {recipe.name}'.strip()).classes('ll-subtitle')
            kind = recipe.kind.replace('_', ' ').capitalize()
            ui.label(t('{kind} · Consumable', kind=kind) if recipe.consumable else t('{kind} · Reusable', kind=kind)).classes('ll-muted')
            with ui.element('div').classes('ll-form-row ll-field-row'):
                for_sale = ui.switch(t('For sale'), value=entry['for_sale'])
                allow_use = ui.switch(t('Allow use'), value=entry['allow_use'])
            with ui.element('div').classes('ll-form-row ll-field-row'):
                price = ui.number(t.p('field', 'Price'), value=entry['price_override'], min=1, precision=0, format='%d')
                price._props.update(clearable=True, hint=t('Blank = built-in price ({price:,} {currency})', price=recipe.default_price, currency=currency))
                cooldown = sides = None
                if recipe.reusable:
                    cooldown = ui.number(t('Cooldown (seconds)'), value=entry['cooldown_override'], min=0, precision=0, format='%d')
                    cooldown._props.update(clearable=True, hint=t('Blank = built-in ({seconds:,} s)', seconds=recipe.default_cooldown_seconds))
                if 'sides' in recipe.params:
                    sides = ui.number(t.p('field', 'Sides'), value=entry['effective_params']['sides'], min=2, precision=0, format='%d')
            extra = swap = None
            if has_plain_table(recipe):
                with ui.expansion(t('Custom lines')).classes('w-full ll-subpanel'):
                    with ui.element('div').classes('ll-stack w-full'):
                        extra = ui.textarea(t('Extra lines'), value='\n'.join(entry['extra_entries'].get(LOCALE, []))).props('autogrow').classes('w-full')
                        extra._props['hint'] = t('One per line. Added to the built-in lines.')
                        swap = ui.textarea(t('Replace built-in lines'), value='\n'.join(entry['replace_entries'].get(LOCALE, []))).props('autogrow').classes('w-full')
                        swap._props['hint'] = t('One per line. If any are given, only these are used. Leave empty to keep the built-in lines.')
            value = lambda field: field.value if field is not None else None

            def operation():
                saved = save_item(store, gid, state['entry'], for_sale=for_sale.value, allow_use=allow_use.value, price=price.value, cooldown=value(cooldown),
                                  sides=value(sides), extra_text=value(extra), replace_text=value(swap))
                state['entry'] = saved
                return saved
            with ui.element('div').classes('ll-form-row'):
                button = ctx.button(t.p('button', 'Save'), operation, 'shop.item', item_detail, success=t('{name} saved', name=recipe.name))
                button._props['aria-label'] = t('Save {name}', name=recipe.name)

    def settings(self):
        from nicegui import ui
        t = web_locale.current()
        ctx, store, gid = self.ctx, self.store, self.gid
        cap = {'value': store.item_hold_cap(gid)}
        with ui.element('div').classes('ll-form-row ll-field-row'):
            field = ui.number(t('Most of a consumable one member can hold'), value=cap['value'], min=1, max=99, precision=0, format='%d').classes('ll-wide')
        ui.label(t('A member cannot buy another of a consumable item at this number. A reusable item can be owned once.')).classes('ll-muted')

        def operation():
            cap['value'] = store.set_item_hold_cap(gid, whole(field.value), cap['value'])
            return cap['value']
        with ui.element('div').classes('ll-form-row'):
            ctx.button(t.p('button', 'Save'), operation, 'shop.hold_cap', lambda result: {'cap': result}, success=t('Hold cap saved'))

    async def render_log(self, reset=False):
        from nicegui import ui
        t = web_locale.current()
        before = None if reset or not self.uses else self.uses[-1]['id']
        fetched = self.store.item_use_log(self.gid, PAGE + 1, before)
        self.more, fetched = len(fetched) > PAGE, fetched[:PAGE]
        await self.lookup([u['user_id'] for u in fetched])
        if not self.live():
            return
        self.uses = fetched if before is None else self.uses + fetched
        self.log_body.clear()
        with self.log_body:
            if not self.uses:
                ui.label(t('No items have been used yet.')).classes('ll-muted')
                return
            table([('time', t.p('column', 'Time')), ('member', t.p('column', 'Member')), ('item', t.p('column', 'Item')), ('result', t.p('column', 'Result'))],
                  use_rows(self.uses, self.names, self.store.guild_timezone(self.gid), t=t), 'id', 'll-shop-table')
            if self.more:
                ui.button(t.p('button', 'Load more'), on_click=lambda: self.render_log()).props('outline')


async def shop_panel(ctx):
    panel = ShopPanel(ctx)
    panel.build()
    await panel.render_log(reset=True)
