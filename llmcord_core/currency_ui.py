"""Dashboard Currency tab (FEAT-17 part B): name, balances, grant/take form and the ledger with reversals.

Everything the admin typed (currency name, reasons, member names) is shown as plain text: labels and table cells only.
"""
from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from fastapi import HTTPException

from .admin_store import CurrencyError

PAGE = 50
MAX_CHANGE = 1_000_000
_MENTION = re.compile(r'<@!?(\d+)>')
_MEMBER_INPUT = re.compile(r'<@!?(\d{1,20})>|(\d{1,20})')
_WHOLE_AMOUNT = f'The amount must be a whole number from 1 to {MAX_CHANGE:,}.'


def parse_member_id(text):
    """A member ID from a raw ID or a pasted <@id> mention; ValueError (shown to the admin) otherwise."""
    found = _MEMBER_INPUT.fullmatch((text or '').strip())
    member = int(found.group(1) or found.group(2)) if found else 0
    if not 0 < member < 2 ** 63:
        raise ValueError('Enter a member ID (digits only) or paste a mention.')
    return member


def parse_amount(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) or not 1 <= value <= MAX_CHANGE:
        raise ValueError(_WHOLE_AMOUNT)
    return int(value)


def member_label(names, user_id):
    return names.get(user_id) or f'Member …{str(user_id)[-4:]}'


def plain_mentions(text, names):
    """Replace <@id> mention markup (as the store writes it in refusals) with the member's name."""
    return _MENTION.sub(lambda m: member_label(names, int(m.group(1))), text)


def ledger_rows(entries, names, zone_name):
    """Display rows for ledger entries (newest first); `entries` must be a run from the newest entry down, so every reversal of a shown entry is shown."""
    try:
        zone = ZoneInfo(zone_name or 'UTC')
    except Exception:
        zone = ZoneInfo('UTC')
    reversed_by = {e['reverses_id']: e['id'] for e in entries if e['reverses_id'] is not None}
    rows = []
    for e in entries:
        note = f"Reverses #{e['reverses_id']}" if e['reverses_id'] is not None else f"Reversed by #{reversed_by[e['id']]}" if e['id'] in reversed_by else ''
        rows.append({'id': e['id'], 'entry': f"#{e['id']}", 'time': datetime.fromtimestamp(e['created_at'], zone).strftime('%Y-%m-%d %H:%M'),
                     'member': member_label(names, e['user_id']), 'amount': f"{e['amount']:+,}", 'after': f"{e['balance_after']:,}",
                     'reason': e['reason'], 'admin': member_label(names, e['actor_id']) if e['actor_id'] else '', 'note': note,
                     'can_reverse': e['reverses_id'] is None and e['id'] not in reversed_by})
    return rows


def friendly(error, names):
    return ValueError(plain_mentions(str(error), names)) if isinstance(error, CurrencyError) else error


class CurrencyPanel:
    def __init__(self, ctx):
        self.ctx, self.store, self.gid = ctx, ctx.store, ctx.guild_id
        self.live = ctx.live('currency')
        self.names, self.absent, self.warned = {}, set(), False  # absent: left the server, or failed to load in this build
        self.page, self.entries, self.more = 0, [], False

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
            ui.notify('Member names could not be loaded. Showing member IDs.', type='warning', timeout=8000)

    def build(self):
        from nicegui import ui
        from .dashboard import section
        ctx, store, gid = self.ctx, self.store, self.gid
        with section('Currency name'):
            saved = {'name': store.currency_name(gid)}
            name = ui.input('Currency name', value=saved['name']).props('maxlength=32')
            ui.label('What this server calls its currency, for example coins or gold (1 to 32 characters). Members see it in /balance.').classes('ll-muted')
            def save_name():
                saved['name'] = store.set_currency_name(gid, name.value, saved['name'])
                return saved['name']
            ctx.savebar.track('Currency name', {name: saved['name']}, save=save_name, action='currency.name',
                              detail=lambda result: {'name': result}, success='Currency name saved')
        with section('Balances'):
            self.balances_body = ui.column().classes('w-full gap-2')
        with section('Give or take currency'):
            self.build_form()
        with section('Ledger'):
            ui.label('Every change is kept here. A mistake is undone by a reversal entry, never by editing.').classes('ll-muted')
            self.ledger_body = ui.column().classes('w-full gap-2')

    def build_form(self):
        from nicegui import ui
        with ui.element('div').classes('ll-form-row ll-field-row'):
            self.member = ui.input('Member ID or mention')
            self.amount = ui.number('Amount', precision=0, format='%d')
        self.reason = ui.input('Reason').props('maxlength=200')
        with ui.element('div').classes('ll-form-row'):
            self.button('Give', 1)
            self.button('Take', -1).props('outline')

    def button(self, label, sign):
        return self.ctx.button(label, lambda: self.change(sign), 'currency.grant' if sign > 0 else 'currency.revoke',
                               lambda row: {'user_id': row['user_id'], 'amount': row['amount'], 'entry_id': row['id']}, then=lambda row: self.changed(row, sign))

    async def change(self, sign):
        member, amount = parse_member_id(self.member.value), parse_amount(self.amount.value)
        await self.lookup([member])
        try:
            return self.store.change_balance(self.gid, member, sign * amount, self.reason.value, self.actor())
        except CurrencyError as error:
            raise friendly(error, self.names)

    async def changed(self, row, sign):
        from nicegui import ui
        name, amount, label = self.store.currency_name(self.gid), abs(row['amount']), member_label(self.names, row['user_id'])
        verb = f'Gave {amount:,} {name} to {label}' if sign > 0 else f'Took {amount:,} {name} from {label}'
        ui.notify(f"{verb}. New balance: {row['balance_after']:,} {name}.", type='positive')
        self.amount.set_value(None)
        self.reason.set_value('')
        await self.render_all()

    async def render_all(self):
        await self.render_balances()
        await self.render_ledger(reset=True)

    async def render_balances(self):
        from nicegui import ui
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
                ui.label('No balances yet. Give a member some currency to start.').classes('ll-muted')
                return
            table([('member', 'Member'), ('balance', 'Balance'), ('id', 'Member ID')],
                  [{'id': str(r['user_id']), 'member': member_label(self.names, r['user_id']), 'balance': f"{r['balance']:,}"} for r in rows], 'id')
            if self.page or more:
                with ui.element('div').classes('ll-form-row'):
                    ui.button('Previous', on_click=lambda: self.turn(-1)).props('outline').set_enabled(self.page > 0)
                    ui.label(f'Page {self.page + 1}').classes('ll-muted')
                    ui.button('Next', on_click=lambda: self.turn(1)).props('outline').set_enabled(more)

    async def turn(self, step):
        self.page = max(0, self.page + step)
        await self.render_balances()

    async def render_ledger(self, reset=False):
        from nicegui import ui
        before = None if reset or not self.entries else self.entries[-1]['id']
        fetched = self.store.ledger(self.gid, limit=PAGE + 1, before_id=before)
        self.more, fetched = len(fetched) > PAGE, fetched[:PAGE]
        entries = fetched if before is None else self.entries + fetched
        await self.lookup([i for e in fetched for i in (e['user_id'], e['actor_id']) if i])
        if not self.live():
            return
        self.entries = entries
        self.ledger_body.clear()
        with self.ledger_body:
            if not entries:
                ui.label('No ledger entries yet.').classes('ll-muted')
                return
            columns = [('entry', 'Entry'), ('time', 'Time'), ('member', 'Member'), ('amount', 'Amount'), ('after', 'Balance after'),
                       ('reason', 'Reason'), ('admin', 'Admin'), ('note', 'Status'), ('action', 'Actions')]
            grid = table(columns, ledger_rows(entries, self.names, self.store.guild_timezone(self.gid)), 'id')
            grid.add_slot('body-cell-action', '<q-td :props="props"><q-btn v-if="props.row.can_reverse" flat dense no-caps color="negative" label="Reverse" '
                          ':aria-label="`Reverse entry ${props.row.entry}`" @click="() => $parent.$emit(\'reverse\', props.row.id)" /></q-td>')
            grid.add_slot('item', '<div class="q-table__grid-item col-12"><div class="q-table__grid-item-card q-pa-sm">'
                          '<div v-for="col in props.cols.filter(c => c.name !== \'action\' && c.value !== \'\')" :key="col.name"><span class="ll-muted">{{ col.label }}:</span> {{ col.value }}</div>'
                          '<q-btn v-if="props.row.can_reverse" flat dense no-caps color="negative" label="Reverse" :aria-label="`Reverse entry ${props.row.entry}`" '
                          '@click="() => $parent.$emit(\'reverse\', props.row.id)" /></div></div>')
            grid.on('reverse', lambda event: self.reverse_requested(event.args))
            if self.more:
                ui.button('Load more', on_click=lambda: self.render_ledger()).props('outline')

    def reverse_requested(self, entry_id):
        if isinstance(entry_id, bool) or not isinstance(entry_id, int):
            return
        row = next((r for r in ledger_rows(self.entries, self.names, self.store.guild_timezone(self.gid)) if r['id'] == entry_id and r['can_reverse']), None)
        if row:
            self.reverse_dialog(row)

    def reverse_dialog(self, row):
        from nicegui import ui
        with ui.dialog() as dialog, ui.card().classes('w-full max-w-lg'):
            ui.label(f"Reverse entry {row['entry']}").classes('text-xl font-bold')
            ui.label(f"This adds a reversal entry that undoes {row['amount']} for {row['member']}. The original entry stays in the ledger.").classes('ll-muted')
            reason = ui.input('Reason').props('maxlength=200')
            def reverse():
                try:
                    return self.store.reverse_entry(self.gid, row['id'], self.actor(), reason.value)
                except CurrencyError as error:
                    raise friendly(error, self.names)
            async def done(result):
                dialog.close()
                ui.notify(f"Reversed entry {row['entry']}.", type='positive')
                await self.render_all()
            with ui.element('div').classes('ll-form-row justify-end'):
                ui.button('Cancel', on_click=dialog.close).props('flat')
                self.ctx.button('Reverse entry', reverse, 'currency.reverse', lambda result: {'entry_id': row['id'], 'reversal_id': result['id'], 'user_id': result['user_id'], 'amount': result['amount']},
                                then=done, color='negative')
        dialog.on('hide', dialog.delete)
        dialog.open()


def table(columns, rows, key):
    from nicegui import ui
    return ui.table(columns=[{'name': n, 'field': n, 'label': label, 'align': 'left', **({'headerClasses': 'sr-only'} if n == 'action' else {})} for n, label in columns], rows=rows, row_key=key,
                    pagination=0).classes('w-full ll-currency-table ll-usage').props(':grid="Quasar.Screen.lt.sm" flat dense hide-pagination')


async def currency_panel(ctx):
    panel = CurrencyPanel(ctx)
    panel.build()
    await panel.render_all()
