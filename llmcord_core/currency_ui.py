"""Dashboard Currency tab (FEAT-17 part B): name, balances, grant/take form and the ledger with reversals.

Everything the admin typed (currency name, reasons, member names) is shown as plain text: labels and table cells only.
"""
from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from fastapi import HTTPException

from . import i18n
from .admin_store import CurrencyError
from .i18n import UserError, codes, web_locale

PAGE = 50
MAX_CHANGE = 1_000_000
_MENTION = re.compile(r'<@!?(\d+)>')
_MEMBER_INPUT = re.compile(r'<@!?(\d{1,20})>|(\d{1,20})')


def parse_member_id(text):
    """A member ID from a raw ID or a pasted <@id> mention; ValueError (shown to the admin) otherwise."""
    found = _MEMBER_INPUT.fullmatch((text or '').strip())
    member = int(found.group(1) or found.group(2)) if found else 0
    if not 0 < member < 2 ** 63:
        raise UserError(codes.CURRENCY_UI_MEMBER_INVALID)
    return member


def parse_amount(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) or not 1 <= value <= MAX_CHANGE:
        raise UserError(codes.CURRENCY_AMOUNT_RANGE, limit=MAX_CHANGE)
    return int(value)


def save_daily(store, gid, amount, streak_bonus, streak_days, expected):
    """Save the daily check-in settings; blank or fractional input reaches the store as-is so its range message is shown."""
    whole = lambda value: int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value == int(value) else value
    return store.set_daily_settings(gid, whole(amount), whole(streak_bonus), whole(streak_days), expected)


def save_refill_cap(store, gid, cap, expected):
    """Save the character refill cap; blank or fractional input reaches the store as-is so its range message is shown."""
    return store.set_refill_cap(gid, int(cap) if isinstance(cap, (int, float)) and not isinstance(cap, bool) and cap == int(cap) else cap, expected)


def save_bets(store, gid, min_bet, max_bet, expected):
    """Save the game bet limits; blank or fractional input reaches the store as-is so its message is shown."""
    whole = lambda value: int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value == int(value) else value
    return store.set_game_settings(gid, whole(min_bet), whole(max_bet), expected)


def save_channels(store, gid, channel_ids, expected):
    """Save the game channels; the result carries the ids of the tables this closed."""
    return store.set_game_channels(gid, {int(c) for c in channel_ids or ()}, set(expected))


def channels_toast(result, *, t=None):
    t = t or i18n.english()
    closed = len(result['closed_tables'])
    return t('Game channels saved. Closed {closed} open table(s) and refunded their bets.', closed=closed) if closed else t('Game channels saved')


def member_label(names, user_id, *, t=None):
    t = t or i18n.english()
    return names.get(user_id) or t('Member …{last4}', last4=str(user_id)[-4:])


def plain_mentions(text, names, *, t=None):
    """Replace <@id> mention markup (as the store writes it in refusals) with the member's name."""
    return _MENTION.sub(lambda m: member_label(names, int(m.group(1)), t=t), text)


def ledger_rows(entries, names, zone_name, characters=None, *, t=None):
    """Display rows for ledger entries (newest first); `entries` must be a run from the newest entry down, so every reversal of a shown entry is shown."""
    t = t or i18n.english()
    try:
        zone = ZoneInfo(zone_name or 'UTC')
    except Exception:
        zone = ZoneInfo('UTC')
    characters = characters or {}
    holder = lambda e: (characters.get(e['user_id']) or t('Deleted character (#{id})', id=e['user_id'])) if e.get('holder_kind') == 'character' else member_label(names, e['user_id'], t=t)
    reversed_by = {e['reverses_id']: e['id'] for e in entries if e['reverses_id'] is not None}
    rows = []
    for e in entries:
        note = t('Reverses #{id}', id=e['reverses_id']) if e['reverses_id'] is not None else t('Reversed by #{id}', id=reversed_by[e['id']]) if e['id'] in reversed_by else ''
        rows.append({'id': e['id'], 'entry': f"#{e['id']}", 'time': datetime.fromtimestamp(e['created_at'], zone).strftime('%Y-%m-%d %H:%M'),
                     'member': holder(e), 'amount': f"{e['amount']:+,}", 'after': f"{e['balance_after']:,}",
                     'reason': e['reason'], 'admin': member_label(names, e['actor_id'], t=t) if e['actor_id'] else '', 'note': note,
                     'shop': e.get('source') == 'shop', 'can_reverse': e['reverses_id'] is None and e['id'] not in reversed_by and (e.get('holder_kind') != 'character' or e['user_id'] in characters)})
    return rows


def reverse_detail(row, result):
    """Audit detail for a reversal; a reversed shop purchase also records the item and whether it was taken back."""
    detail = {'entry_id': row['id'], 'reversal_id': result['id'], 'user_id': result['user_id'], 'amount': result['amount']}
    if result.get('shop_item'):
        detail['shop_item'] = {'item_key': result['shop_item']['item_key'], 'removed': result['shop_item']['removed']}
    return detail


def reverse_note(result, *, t=None):
    t = t or i18n.english()
    shop = result.get('shop_item')
    if not shop:
        return ''
    return ' ' + t('The shop item was taken back.') if shop['removed'] else ' ' + t('The shop item was already used, so only the currency was returned.')


def friendly(error, names, *, t=None):
    """A currency refusal with the member's name where the store wrote a <@id> mention. A coded refusal stays coded (the member codes map to the
    named-holder twins, which read the same in English), so the surface translates it with t.error."""
    if not isinstance(error, CurrencyError):
        return error
    params = error.params
    if error.code is codes.CURRENCY_MAX_MEMBER:
        return CurrencyError(codes.CURRENCY_MAX_CHARACTER, name=member_label(names, params['user_id'], t=t), limit=params['limit'], currency=params['name'])
    if error.code is codes.CURRENCY_NEGATIVE_MEMBER:
        return CurrencyError(codes.CURRENCY_NEGATIVE_CHARACTER, name=member_label(names, params['user_id'], t=t), current=params['current'], currency=params['name'])
    if error.code is not None:
        return error
    return ValueError(plain_mentions(str(error), names, t=t))


class CurrencyPanel:
    def __init__(self, ctx):
        self.ctx, self.store, self.gid = ctx, ctx.store, ctx.guild_id
        self.live = ctx.live('currency')
        self.names, self.absent, self.warned = {}, set(), False  # absent: left the server, or failed to load in this build
        self.page, self.character_page, self.entries, self.more, self.characters = 0, 0, [], False, {}

    def actor(self):
        return int(self.ctx.app.state.sessions[self.ctx.ident]['user']['id'])

    async def lookup(self, ids):
        """Fetch display names for members not cached yet; a failure warns once per build and leaves IDs."""
        missing = [i for i in dict.fromkeys(ids) if i not in self.names and i not in self.absent]
        if not missing:
            return
        try:
            found, failed = await self.ctx.service.run(self.ctx.ident, self.gid, lambda: self.ctx.service.avatars.member_names(self.gid, missing))
        except (httpx.HTTPError, ValueError, HTTPException):
            found, failed = {}, True
        self.names.update(found)
        self.absent.update(set(missing) - set(found))  # no retry on page turns or writes until the next tab build
        if failed and not self.warned:
            from nicegui import ui
            self.warned = True
            ui.notify(web_locale.current()('Member names could not be loaded. Showing member IDs.'), type='warning', timeout=8000)

    def build(self):
        from nicegui import ui
        from .dashboard import section
        t = web_locale.current()
        ctx, store, gid = self.ctx, self.store, self.gid
        with section(t('Currency name')):
            saved = {'name': store.currency_name(gid)}
            name = ui.input(t('Currency name'), value=saved['name']).props('maxlength=32')
            ui.label(t('What this server calls its currency, for example coins or gold (1 to 32 characters). Members see it in /balance.')).classes('ll-muted')
            def save_name():
                saved['name'] = store.set_currency_name(gid, name.value, saved['name'])
                return saved['name']
            ctx.savebar.track(t('Currency name'), {name: saved['name']}, save=save_name, action='currency.name',
                              detail=lambda result: {'name': result}, success=t('Currency name saved'),
                              dirty=lambda: (name.value or '') != saved['name'],
                              reset=lambda: name.set_value(saved['name']))
        with section(t('Daily check-in')):
            daily = store.daily_settings(gid)
            with ui.element('div').classes('ll-form-row ll-field-row'):
                fields = [ui.number(label, value=daily[key], min=0, precision=0, format='%d') for label, key in
                          ((t('Amount per check-in'), 'amount'), (t('Streak bonus per day'), 'streak_bonus'), (t('Streak bonus grows for up to (days)'), 'streak_days'))]
            ui.label(t("Members claim with /daily once a day; the day ends at midnight in this server's timezone. Set the amount to 0 to turn check-ins off. A missed day starts the streak again.")).classes('ll-muted')
            def save_daily_settings():
                nonlocal daily
                daily = save_daily(store, gid, *(f.value for f in fields), daily)
                self.refill_off.set_visibility(daily['amount'] == 0)
                return daily
            daily_keys = ('amount', 'streak_bonus', 'streak_days')
            def daily_reset():
                for f, key in zip(fields, daily_keys):
                    f.set_value(daily[key])
            ctx.savebar.track(t('Daily check-in'), {f: daily[key] for f, key in zip(fields, daily_keys)}, save=save_daily_settings, action='currency.daily',
                              detail=lambda result: dict(result), success=t('Daily check-in saved'),
                              dirty=lambda: [f.value for f in fields] != [daily[key] for key in daily_keys],
                              reset=daily_reset)
        with section(t('Character wallets')):
            cap = {'value': store.refill_cap(gid)}
            refill_cap = ui.number(t('Character refill cap'), value=cap['value'], min=0, precision=0, format='%d')
            ui.label(t('Each character gets the daily amount once per server day, without the streak bonus, until it reaches this cap. Winnings can take a character above it.')).classes('ll-muted')
            self.refill_off = ui.label(t("Daily check-in is off, so characters don't refill.")).classes('ll-muted')
            self.refill_off.set_visibility(daily['amount'] == 0)
            def save_cap():
                cap['value'] = save_refill_cap(store, gid, refill_cap.value, cap['value'])
                return cap['value']
            ctx.savebar.track(t('Character refill cap'), {refill_cap: cap['value']}, save=save_cap, action='currency.refill_cap',
                              detail=lambda result: {'cap': result}, success=t('Character refill cap saved'),
                              dirty=lambda: refill_cap.value != cap['value'],
                              reset=lambda: refill_cap.set_value(cap['value']))
        ui.label(t('Game channels, bets and table talk are in the Games tab.')).classes('ll-muted')
        with section(t('Balances')):
            self.balances_body = ui.column().classes('w-full gap-2')
        with section(t('Character balances')):
            self.characters_body = ui.column().classes('w-full gap-2')
        with section(t('Give or take currency')):
            self.build_form()
        with section(t.p('section', 'Ledger')):
            ui.label(t('Every change is kept here. A mistake is undone by a reversal entry, never by editing.')).classes('ll-muted')
            self.ledger_body = ui.column().classes('w-full gap-2')

    def build_form(self):
        from nicegui import ui
        t = web_locale.current()
        with ui.element('div').classes('ll-form-row ll-field-row'):
            self.member = ui.input(t('Member ID or mention'))
            self.amount = ui.number(t.p('field', 'Amount'), precision=0, format='%d')
        self.reason = ui.input(t.p('field', 'Reason')).props('maxlength=200')
        with ui.element('div').classes('ll-form-row'):
            self.button(t.p('button', 'Give'), 1)
            self.button(t.p('button', 'Take'), -1).props('outline')

    def button(self, label, sign):
        return self.ctx.button(label, lambda: self.change(sign), 'currency.grant' if sign > 0 else 'currency.revoke',
                               lambda row: {'user_id': row['user_id'], 'amount': row['amount'], 'entry_id': row['id']}, then=lambda row: self.changed(row, sign))

    async def change(self, sign):
        t = web_locale.current()
        member, amount = parse_member_id(self.member.value), parse_amount(self.amount.value)
        await self.lookup([member])
        try:
            return self.store.change_balance(self.gid, member, sign * amount, self.reason.value, self.actor())
        except CurrencyError as error:
            raise friendly(error, self.names, t=t)

    async def changed(self, row, sign):
        from nicegui import ui
        t = web_locale.current()
        name, amount, label = self.store.currency_name(self.gid), abs(row['amount']), member_label(self.names, row['user_id'], t=t)
        balance = row['balance_after']
        ui.notify(t('Gave {amount:,} {name} to {member}. New balance: {balance:,} {name}.', amount=amount, name=name, member=label, balance=balance) if sign > 0
                  else t('Took {amount:,} {name} from {member}. New balance: {balance:,} {name}.', amount=amount, name=name, member=label, balance=balance), type='positive')
        self.amount.set_value(None)
        self.reason.set_value('')
        await self.render_all()

    async def render_all(self):
        await self.render_balances()
        await self.render_character_balances()
        await self.render_ledger(reset=True)

    async def render_balances(self):
        from nicegui import ui
        t = web_locale.current()
        rows = self.store.balances(self.gid, PAGE + 1, self.page * PAGE)
        more, rows = len(rows) > PAGE, rows[:PAGE]
        if not rows and self.page:
            self.page -= 1
            return await self.render_balances()
        await self.lookup([r['user_id'] for r in rows])
        if not self.live():
            return
        self.balances_body.clear()
        with self.balances_body:
            if not rows:
                ui.label(t('No balances yet. Give a member some currency to start.')).classes('ll-muted')
                return
            table([('member', t.p('column', 'Member')), ('balance', t.p('column', 'Balance')), ('id', t.p('column', 'Member ID'))],
                  [{'id': str(r['user_id']), 'member': member_label(self.names, r['user_id'], t=t), 'balance': f"{r['balance']:,}"} for r in rows], 'id')
            if self.page or more:
                with ui.element('div').classes('ll-form-row'):
                    ui.button(t.p('button', 'Previous'), on_click=lambda: self.turn(-1)).props('outline').set_enabled(self.page > 0)
                    ui.label(t('Page {page}', page=self.page + 1)).classes('ll-muted')
                    ui.button(t.p('button', 'Next'), on_click=lambda: self.turn(1)).props('outline').set_enabled(more)

    async def render_character_balances(self):
        from nicegui import ui
        t = web_locale.current()
        rows = self.store.character_balances(self.gid, PAGE + 1, self.character_page * PAGE)
        more, rows = len(rows) > PAGE, rows[:PAGE]
        if not rows and self.character_page:
            self.character_page -= 1
            return await self.render_character_balances()
        if not self.live():
            return
        self.characters_body.clear()
        with self.characters_body:
            if not rows:
                ui.label(t('No character balances yet. Characters refill when they first need money each day.')).classes('ll-muted')
                return
            table([('character', t.p('column', 'Character')), ('balance', t.p('column', 'Balance'))],
                  [{'id': str(r['character_id']), 'character': t('{name} (archived)', name=r['name']) if r['archived'] else r['name'], 'balance': f"{r['balance']:,}"} for r in rows], 'id', 'll-character-table')
            if self.character_page or more:
                with ui.element('div').classes('ll-form-row'):
                    ui.button(t.p('button', 'Previous'), on_click=lambda: self.turn_characters(-1)).props('outline').set_enabled(self.character_page > 0)
                    ui.label(t('Page {page}', page=self.character_page + 1)).classes('ll-muted')
                    ui.button(t.p('button', 'Next'), on_click=lambda: self.turn_characters(1)).props('outline').set_enabled(more)

    async def turn_characters(self, step):
        self.character_page = max(0, self.character_page + step)
        await self.render_character_balances()

    async def turn(self, step):
        self.page = max(0, self.page + step)
        await self.render_balances()

    async def render_ledger(self, reset=False):
        from nicegui import ui
        from html import escape
        t = web_locale.current()
        before = None if reset or not self.entries else self.entries[-1]['id']
        fetched = self.store.ledger(self.gid, limit=PAGE + 1, before_id=before)
        self.more, fetched = len(fetched) > PAGE, fetched[:PAGE]
        entries = fetched if before is None else self.entries + fetched
        await self.lookup([i for e in fetched for i in (e['user_id'] if e['holder_kind'] == 'member' else 0, e['actor_id']) if i])
        if not self.live():
            return
        self.characters = {**self.characters, **self.store.character_names(self.gid, [e['user_id'] for e in fetched if e['holder_kind'] == 'character'])}
        self.entries = entries
        self.ledger_body.clear()
        with self.ledger_body:
            if not entries:
                ui.label(t('No ledger entries yet.')).classes('ll-muted')
                return
            columns = [('entry', t.p('column', 'Entry')), ('time', t.p('column', 'Time')), ('member', t.p('column', 'Member')), ('amount', t.p('column', 'Amount')), ('after', t.p('column', 'Balance after')),
                       ('reason', t.p('column', 'Reason')), ('admin', t.p('column', 'Admin')), ('note', t.p('column', 'Status')), ('action', t.p('column', 'Actions'))]
            grid = table(columns, ledger_rows(entries, self.names, self.store.guild_timezone(self.gid), self.characters, t=t), 'id')
            reverse_label = escape(t.p('button', 'Reverse'))
            reverse_aria = escape(t('Reverse entry {entry}', entry='${props.row.entry}'))
            grid.add_slot('body-cell-action', f'<q-td :props="props"><q-btn v-if="props.row.can_reverse" flat dense no-caps color="negative" label="{reverse_label}" '
                          f':aria-label="`{reverse_aria}`" @click="() => $parent.$emit(\'reverse\', props.row.id)" /></q-td>')
            grid.add_slot('item', '<div class="q-table__grid-item col-12"><div class="q-table__grid-item-card q-pa-sm">'
                          '<div v-for="col in props.cols.filter(c => c.name !== \'action\' && c.value !== \'\')" :key="col.name"><span class="ll-muted">{{ col.label }}:</span> {{ col.value }}</div>'
                          f'<q-btn v-if="props.row.can_reverse" flat dense no-caps color="negative" label="{reverse_label}" :aria-label="`{reverse_aria}`" '
                          '@click="() => $parent.$emit(\'reverse\', props.row.id)" /></div></div>')
            grid.on('reverse', lambda event: self.reverse_requested(event.args))
            if self.more:
                ui.button(t.p('button', 'Load more'), on_click=lambda: self.render_ledger()).props('outline')

    def reverse_requested(self, entry_id):
        if isinstance(entry_id, bool) or not isinstance(entry_id, int):
            return
        row = next((r for r in ledger_rows(self.entries, self.names, self.store.guild_timezone(self.gid), self.characters, t=web_locale.current()) if r['id'] == entry_id and r['can_reverse']), None)
        if row:
            self.reverse_dialog(row)

    def reverse_dialog(self, row):
        from nicegui import ui
        t = web_locale.current()
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-lg'):
            ui.label(t('Reverse entry {entry}', entry=row['entry'])).classes('text-xl font-bold')
            ui.label(t('This adds a reversal entry that undoes {amount} for {member}. The original entry stays in the ledger. This also takes back the item that was bought, if the member still has it.', amount=row['amount'], member=row['member'])
                     if row.get('shop') else
                     t('This adds a reversal entry that undoes {amount} for {member}. The original entry stays in the ledger.', amount=row['amount'], member=row['member'])).classes('ll-muted')
            reason = ui.input(t.p('field', 'Reason')).props('maxlength=200')
            def reverse():
                try:
                    return self.store.reverse_entry(self.gid, row['id'], self.actor(), reason.value)
                except CurrencyError as error:
                    raise friendly(error, self.names, t=t)
            async def done(result):
                dialog.close()
                ui.notify(t('Reversed entry {entry}.', entry=row['entry']) + reverse_note(result, t=t), type='positive')
                await self.render_all()
            with ui.element('div').classes('ll-form-row justify-end'):
                ui.button(t.p('button', 'Cancel'), on_click=dialog.close).props('flat')
                self.ctx.button(t('Reverse entry'), reverse, 'currency.reverse', lambda result: reverse_detail(row, result), then=done, color='negative')
        dialog.on('hide', dialog.delete)
        dialog.open()


def table(columns, rows, key, css='ll-currency-table'):
    from nicegui import ui
    return ui.table(columns=[{'name': n, 'field': n, 'label': label, 'align': 'left', **({'headerClasses': 'sr-only'} if n == 'action' else {})} for n, label in columns], rows=rows, row_key=key,
                    pagination=0).classes(f'w-full {css} ll-usage').props(':grid="Quasar.Screen.lt.sm" flat dense hide-pagination')


async def currency_panel(ctx):
    panel = CurrencyPanel(ctx)
    panel.build()
    await panel.render_all()
