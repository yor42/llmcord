"""Dashboard Games tab (FEAT-27): settings shared by all games, then one section per game (blackjack first)."""
from __future__ import annotations

from .currency_ui import channels_toast, save_bets, save_channels

_whole = lambda value: int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value == int(value) else value


def all_games_section(ctx):
    from nicegui import ui
    from .dashboard import section
    store, gid = ctx.store, ctx.guild_id
    whole = _whole
    with section('All games'):
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
                          detail=lambda result: {'enabled': result['enabled'], 'closed_tables': list(result['closed_tables'])}, success='Bet limits saved',
                          dirty=lambda: [f.value for f in limits] != [bets[key] for key in bet_keys],
                          reset=bets_reset)
        talk = {'value': store.game_character_talk(gid)}
        talk_switch = ui.switch('Characters talk at the table', value=talk['value'])
        ui.label('On: a character at a blackjack table asks the model for its move and may say a short line, one model call per decision. Off: characters play by the house rule (hit below 17) and stay quiet.').classes('ll-muted')
        def save_talk():
            talk['value'] = store.set_game_character_talk(gid, bool(talk_switch.value), talk['value'])
            return talk['value']
        ctx.savebar.track('Character table talk', {talk_switch: talk['value']}, save=save_talk, action='games.character_talk',
                          detail=lambda result: {'enabled': result}, success='Character table talk saved',
                          dirty=lambda: bool(talk_switch.value) != talk['value'],
                          reset=lambda: talk_switch.set_value(talk['value']))
        limit = {'value': store.game_talk_daily_limit(gid)}
        limit_field = ui.number('Table talk calls per day', value=limit['value'], min=1, precision=0, format='%d')
        ui.label(f"After about this many table talk calls in a server day (approximate when several tables play at once), characters play by the house rule and stay quiet until the next day. Used today: {store.game_talk_calls_today(gid):,}.").classes('ll-muted')
        def save_limit():
            value = limit_field.value
            limit['value'] = store.set_game_talk_daily_limit(gid, int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value == int(value) else value, limit['value'])
            return limit['value']
        ctx.savebar.track('Table talk limit', {limit_field: limit['value']}, save=save_limit, action='games.talk_daily_limit',
                          detail=lambda result: {'limit': result}, success='Table talk limit saved',
                          dirty=lambda: limit_field.value != limit['value'],
                          reset=lambda: limit_field.set_value(limit['value']))
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

        rounds = {'value': store.game_summary_rounds(gid)}
        with ui.element('div').classes('ll-form-row ll-field-row'):
            rounds_field = ui.number('Rounds in the end-of-table summary', value=rounds['value'], min=1, max=20, precision=0, format='%d').classes('ll-wide')
        ui.label('When a table closes, its message lists this many of the last rounds.').classes('ll-muted')
        def save_rounds():
            rounds['value'] = store.set_game_summary_rounds(gid, whole(rounds_field.value), rounds['value'])
            return rounds['value']
        ctx.savebar.track('Summary rounds', {rounds_field: rounds['value']}, save=save_rounds, action='games.summary_rounds',
                          detail=lambda result: {'rounds': result}, success='Summary rounds saved',
                          dirty=lambda: rounds_field.value != rounds['value'],
                          reset=lambda: rounds_field.set_value(rounds['value']))


def blackjack_section(ctx):
    from nicegui import ui
    from .dashboard import section
    from .games import blackjack
    store, gid = ctx.store, ctx.guild_id
    with section('Blackjack'):
        on = {'value': store.blackjack_enabled(gid)}
        switch = ui.switch('Blackjack is on', value=on['value'])
        ui.label('Off: /blackjack is refused and open blackjack tables close with every bet refunded.').classes('ll-muted')
        def save_on():
            result = store.set_blackjack_enabled(gid, bool(switch.value), on['value'])
            on['value'] = result['enabled']
            return result
        def toast(result):
            closed = len(result['closed_tables'])
            if closed:
                ui.notify(f"Blackjack is off. Closed {closed} open {'table' if closed == 1 else 'tables'}.", type='positive')
        ctx.savebar.track('Blackjack', {switch: on['value']}, save=save_on, action='games.blackjack_enabled',
                          detail=lambda result: dict(result), success='Blackjack setting saved',
                          dirty=lambda: bool(switch.value) != on['value'],
                          reset=lambda: switch.set_value(on['value']),
                          then=toast)

        rules = {'value': store.blackjack_rules(gid)}
        ties = {'push': 'Push: bet comes back', 'dealer': 'Dealer wins'}
        with ui.element('div').classes('ll-form-row ll-field-row'):
            stand_on = ui.select({16: '16', 17: '17', 18: '18'}, value=rules['value']['stand_on'], label='Dealer stands on').classes('ll-wide')
            pays = ui.select({'3:2': '3:2', '6:5': '6:5'}, value=rules['value']['blackjack_pays'], label='Blackjack pays')
            tie = ui.select(ties, value=rules['value']['ties'], label='Ties').classes('ll-wide')
        soft = ui.switch('Dealer hits soft 17', value=rules['value']['hit_soft_17'])
        insurance = ui.switch('Insurance and even money', value=rules['value']['insurance'])
        surrender = ui.switch('Late surrender', value=rules['value']['surrender'])
        controls = (stand_on, soft, pays, tie, insurance, surrender)
        def current():
            return {'stand_on': stand_on.value, 'hit_soft_17': bool(soft.value) and stand_on.value == 17, 'blackjack_pays': pays.value, 'ties': tie.value,
                    'insurance': bool(insurance.value), 'surrender': bool(surrender.value)}
        preview = ui.label().classes('ll-muted')
        def refresh(_=None):
            if stand_on.value != 17:
                soft.set_value(False)
            soft.set_enabled(stand_on.value == 17 and (ctx.savebar.active is None or ctx.savebar.active.name == 'Blackjack rules'))
            preview.set_text('Table rules line: ' + blackjack.rules_text(current()))
        for control in controls:
            control.on_value_change(refresh)
        refresh()
        ui.label('Rule changes apply from the next round. Open rounds keep the rules they were dealt with.').classes('ll-muted')
        def save_rules():
            result = store.set_blackjack_rules(gid, current(), expected=rules['value'])
            rules['value'] = result
            return result
        def reset_rules():
            saved = rules['value']
            for control, key in zip(controls, ('stand_on', 'hit_soft_17', 'blackjack_pays', 'ties', 'insurance', 'surrender')):
                control.set_value(saved[key])
        ctx.savebar.track('Blackjack rules', {c: rules['value'][k] for c, k in zip(controls, ('stand_on', 'hit_soft_17', 'blackjack_pays', 'ties', 'insurance', 'surrender'))},
                          save=save_rules, action='games.blackjack_rules', detail=lambda result: dict(result), success='Blackjack rules saved',
                          dirty=lambda: current() != rules['value'], reset=reset_rules, then=lambda result: refresh(),
                          enabled=lambda control: control is not soft or stand_on.value == 17)  # soft 17 only exists when the dealer stands on 17


async def games_panel(ctx):
    all_games_section(ctx)
    blackjack_section(ctx)
