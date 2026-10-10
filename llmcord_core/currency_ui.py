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


def channels_toast(result):
    closed = len(result['closed_tables'])
    return f'Game channels saved. Closed {closed} open table(s) and refunded their bets.' if closed else 'Game channels saved'


def member_label(names, user_id):
    return names.get(user_id) or f'Member …{str(user_id)[-4:]}'


def plain_mentions(text, names):
    """Replace <@id> mention markup (as the store writes it in refusals) with the member's name."""
    return _MENTION.sub(lambda m: member_label(names, int(m.group(1))), text)


def ledger_rows(entries, names, zone_name, characters=None):
    """Display rows for ledger entries (newest first); `entries` must be a run from the newest entry down, so every reversal of a shown entry is shown."""
    try:
        zone = ZoneInfo(zone_name or 'UTC')
    except Exception:
        zone = ZoneInfo('UTC')
    characters = characters or {}
    holder = lambda e: (characters.get(e['user_id']) or f"Deleted character (#{e['user_id']})") if e.get('holder_kind') == 'character' else member_label(names, e['user_id'])
    reversed_by = {e['reverses_id']: e['id'] for e in entries if e['reverses_id'] is not None}
    rows = []
    for e in entries:
        note = f"Reverses #{e['reverses_id']}" if e['reverses_id'] is not None else f"Reversed by #{reversed_by[e['id']]}" if e['id'] in reversed_by else ''
        rows.append({'id': e['id'], 'entry': f"#{e['id']}", 'time': datetime.fromtimestamp(e['created_at'], zone).strftime('%Y-%m-%d %H:%M'),
                     'member': holder(e), 'amount': f"{e['amount']:+,}", 'after': f"{e['balance_after']:,}",
                     'reason': e['reason'], 'admin': member_label(names, e['actor_id']) if e['actor_id'] else '', 'note': note,
                     'can_reverse': e['reverses_id'] is None and e['id'] not in reversed_by and (e.get('holder_kind') != 'character' or e['user_id'] in characters)})
    return rows


def friendly(error, names):
    return ValueError(plain_mentions(str(error), names)) if isinstance(error, CurrencyError) else error


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
                              detail=lambda result: {'name': result}, success='Currency name saved',
                              dirty=lambda: (name.value or '') != saved['name'],
                              reset=lambda: name.set_value(saved['name']))
        with section('Daily check-in'):
            daily = store.daily_settings(gid)
            with ui.element('div').classes('ll-form-row ll-field-row'):
                fields = [ui.number(label, value=daily[key], min=0, precision=0, format='%d') for label, key in
                          (('Amount per check-in', 'amount'), ('Streak bonus per day', 'streak_bonus'), ('Streak bonus grows for up to (days)', 'streak_days'))]
            ui.label("Members claim with /daily once a day; the day ends at midnight in this server's timezone. Set the amount to 0 to turn check-ins off. A missed day starts the streak again.").classes('ll-muted')
            def save_daily_settings():
                nonlocal daily
                daily = save_daily(store, gid, *(f.value for f in fields), daily)
                self.refill_off.set_visibility(daily['amount'] == 0)
                return daily
            daily_keys = ('amount', 'streak_bonus', 'streak_days')
            def daily_reset():
                for f, key in zip(fields, daily_keys):
                    f.set_value(daily[key])
            ctx.savebar.track('Daily check-in', {f: daily[key] for f, key in zip(fields, daily_keys)}, save=save_daily_settings, action='currency.daily',
                              detail=lambda result: dict(result), success='Daily check-in saved',
                              dirty=lambda: [f.value for f in fields] != [daily[key] for key in daily_keys],
                              reset=daily_reset)
        with section('Character wallets'):
            cap = {'value': store.refill_cap(gid)}
            refill_cap = ui.number('Character refill cap', value=cap['value'], min=0, precision=0, format='%d')
            ui.label('Each character gets the daily amount once per server day, without the streak bonus, until it reaches this cap. Winnings can take a character above it.').classes('ll-muted')
            self.refill_off = ui.label("Daily check-in is off, so characters don't refill.").classes('ll-muted')
            self.refill_off.set_visibility(daily['amount'] == 0)
            def save_cap():
                cap['value'] = save_refill_cap(store, gid, refill_cap.value, cap['value'])
                return cap['value']
            ctx.savebar.track('Character refill cap', {refill_cap: cap['value']}, save=save_cap, action='currency.refill_cap',
                              detail=lambda result: {'cap': result}, success='Character refill cap saved',
                              dirty=lambda: refill_cap.value != cap['value'],
                              reset=lambda: refill_cap.set_value(cap['value']))
        with section('Games'):
            bets = store.game_settings(gid)
            with ui.element('div').classes('ll-form-row ll-field-row'):
                limits = [ui.number(label, value=bets[key], min=1, precision=0, format='%d') for label, key in (('Smallest bet', 'min_bet'), ('Largest bet', 'max_bet'))]
            def save_bet_limits():
                nonlocal bets
                bets = save_bets(store, gid, *(f.value for f in limits), bets)
                return bets
            bet_keys = ('min_bet', 'max_bet')
            def bets_reset():
                for f, key in zip(limits, bet_keys):
                    f.set_value(bets[key])
            ctx.savebar.track('Bet limits', {f: bets[key] for f, key in zip(limits, bet_keys)}, save=save_bet_limits, action='games.bets',
                              detail=lambda result: dict(result), success='Bet limits saved',
                              dirty=lambda: [f.value for f in limits] != [bets[key] for key in bet_keys],
                              reset=bets_reset)
            games = store.game_channels(gid)
            options = {**{c: str(c) for c in sorted(games)}, **ctx.channel_names}
            picker = ui.select(options, value=sorted(games), multiple=True, label='Game channels').props('use-chips').classes('w-full')
            def save_game_channels():
                nonlocal games
                wanted = {int(c) for c in picker.value or ()}
                result = save_channels(store, gid, wanted, games)
                games = wanted
                return result
            ctx.savebar.track('Game channels', {picker: sorted(games)}, save=save_game_channels, action='games.channels',
                              detail=lambda result: {'channels': sorted(games), 'closed_tables': list(result['closed_tables'])},
                              dirty=lambda: {int(c) for c in picker.value or ()} != games,
                              reset=lambda: picker.set_value(sorted(games)),
                              then=lambda result: ui.notify(channels_toast(result), type='positive'))
            ui.label('Members play with /blackjack in game channels. Bets are taken when a member joins and paid when the round ends. Turning a channel off closes its open table and refunds the bets.').classes('ll-muted')
        with section('Balances'):
            self.balances_body = ui.column().classes('w-full gap-2')
        with section('Character balances'):
            self.characters_body = ui.column().classes('w-full gap-2')
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
        await self.render_character_balances()
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

    async def render_character_balances(self):
        from nicegui import ui
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
                ui.label('No character balances yet. Characters refill when they first need money each day.').classes('ll-muted')
                return
            table([('character', 'Character'), ('balance', 'Balance')],
                  [{'id': str(r['character_id']), 'character': r['name'] + (' (archived)' if r['archived'] else ''), 'balance': f"{r['balance']:,}"} for r in rows], 'id', 'll-character-table')
            if self.character_page or more:
                with ui.element('div').classes('ll-form-row'):
                    ui.button('Previous', on_click=lambda: self.turn_characters(-1)).props('outline').set_enabled(self.character_page > 0)
                    ui.label(f'Page {self.character_page + 1}').classes('ll-muted')
                    ui.button('Next', on_click=lambda: self.turn_characters(1)).props('outline').set_enabled(more)

    async def turn_characters(self, step):
        self.character_page = max(0, self.character_page + step)
        await self.render_character_balances()

    async def turn(self, step):
        self.page = max(0, self.page + step)
        await self.render_balances()

    async def render_ledger(self, reset=False):
        from nicegui import ui
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
                ui.label('No ledger entries yet.').classes('ll-muted')
                return
            columns = [('entry', 'Entry'), ('time', 'Time'), ('member', 'Member'), ('amount', 'Amount'), ('after', 'Balance after'),
                       ('reason', 'Reason'), ('admin', 'Admin'), ('note', 'Status'), ('action', 'Actions')]
            grid = table(columns, ledger_rows(entries, self.names, self.store.guild_timezone(self.gid), self.characters), 'id')
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
        row = next((r for r in ledger_rows(self.entries, self.names, self.store.guild_timezone(self.gid), self.characters) if r['id'] == entry_id and r['can_reverse']), None)
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


def table(columns, rows, key, css='ll-currency-table'):
    from nicegui import ui
    return ui.table(columns=[{'name': n, 'field': n, 'label': label, 'align': 'left', **({'headerClasses': 'sr-only'} if n == 'action' else {})} for n, label in columns], rows=rows, row_key=key,
                    pagination=0).classes(f'w-full {css} ll-usage').props(':grid="Quasar.Screen.lt.sm" flat dense hide-pagination')


async def currency_panel(ctx):
    panel = CurrencyPanel(ctx)
    panel.build()
    await panel.render_all()
