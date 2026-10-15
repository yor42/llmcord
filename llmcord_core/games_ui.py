"""Dashboard Games tab (FEAT-27): settings shared by all games, then one section per game (blackjack first)."""
from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from . import i18n
from .admin_store import ConflictError
from .currency_ui import channels_toast, save_bets, save_channels
from .i18n import N_, UserError, codes, terms, web_locale

_whole = lambda value: int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value == int(value) else value


def _move_labels(t):
    """The blackjack move names shown to the admin; spoken.blackjack.LABELS stays the English source for the keys."""
    from .spoken.blackjack import LABELS
    ours = {'hit': t.p('blackjack move', 'Hit'), 'stand': t.p('blackjack move', 'Stand'), 'double': t.p('blackjack move', 'Double'), 'surrender': t.p('blackjack move', 'Surrender'),
            'insure': t.p('blackjack move', 'Insure'), 'even_money': t.p('blackjack move', 'Even money'), 'no_insurance': t.p('blackjack move', 'No insurance')}
    return {key: ours.get(key, label) for key, label in LABELS.items()}


def _move_name(label, labels, *, t=None):
    t = t or i18n.english()
    return labels.get(label, t('Table talk'))


def phrase_rows(phrases, zone_name, *, t=None):
    """Display rows for kept phrases (newest first). The message text is shown as-is by the caller; no member is ever named."""
    t = t or i18n.english()
    LABELS = _move_labels(t)
    try:
        zone = ZoneInfo(zone_name or 'UTC')
    except Exception:
        zone = ZoneInfo('UTC')
    rows = []
    for p in phrases:
        offered = [LABELS.get(m, str(m)) for m in p['offered']]
        check = (t('Offered {move}', move=offered[0]) if len(offered) == 1 else t('Offered {moves} or {last}', moves=', '.join(offered[:-1]), last=offered[-1])) if offered else t('No check')
        meant = t('Meant: table talk') if p['label'] == 'talk' else t('Meant: {move}', move=_move_name(p['label'], LABELS, t=t))
        rows.append({'id': p['id'], 'text': p['text'], 'check': check, 'meant': meant, 'label': p['label'] if p['label'] in LABELS else 'talk',
                     'when': datetime.fromtimestamp(p['created_at'], zone).strftime('%Y-%m-%d %H:%M')})
    return rows


def all_games_section(ctx):
    from nicegui import ui
    from .dashboard import section
    t = web_locale.current()
    store, gid = ctx.store, ctx.guild_id
    whole = _whole
    with section(t('All games')):
        bets = store.game_settings(gid)
        with ui.element('div').classes('ll-form-row ll-field-row'):
            limits = [ui.number(label, value=bets[key], min=1, precision=0, format='%d') for label, key in ((t('Smallest bet'), 'min_bet'), (t('Largest bet'), 'max_bet'))]
        def save_bet_limits():
            nonlocal bets
            bets = save_bets(store, gid, *(f.value for f in limits), bets)
            return bets
        bet_keys = ('min_bet', 'max_bet')
        def bets_reset():
            for f, key in zip(limits, bet_keys):
                f.set_value(bets[key])
        ctx.savebar.track(t('Bet limits'), {f: bets[key] for f, key in zip(limits, bet_keys)}, save=save_bet_limits, action='games.bets',
                          detail=lambda result: {'min_bet': result['min_bet'], 'max_bet': result['max_bet']}, success=t('Bet limits saved'),
                          dirty=lambda: [f.value for f in limits] != [bets[key] for key in bet_keys],
                          reset=bets_reset)
        talk = {'value': store.game_character_talk(gid)}
        talk_switch = ui.switch(t('Characters talk at the table'), value=talk['value'])
        ui.label(t('On: a character at a blackjack table asks the model for its move and may say a short line, one model call per decision. Off: characters play by the house rule (hit below 17) and stay quiet.')).classes('ll-muted')
        def save_talk():
            talk['value'] = store.set_game_character_talk(gid, bool(talk_switch.value), talk['value'])
            return talk['value']
        ctx.savebar.track(t('Character table talk'), {talk_switch: talk['value']}, save=save_talk, action='games.character_talk',
                          detail=lambda result: {'enabled': result}, success=t('Character table talk saved'),
                          dirty=lambda: bool(talk_switch.value) != talk['value'],
                          reset=lambda: talk_switch.set_value(talk['value']))
        moves = {'value': store.game_character_moves(gid)}
        moves_select = ui.select({'model': t('Model picks the move'), 'style': t('Play style only')}, value=moves['value'], label=t('Character moves')).classes('ll-wide game-character-moves')
        ui.label(t('Model picks the move: with table talk on and the daily limit not reached, a model chooses each character\'s move; otherwise it plays by its play style (blackjack: the house rule). '
                   'Play style only: I Doubt It characters never call the model for a move, which saves calls (a saved play style then sets their bluffing and suspicion); blackjack characters keep the house rule, and with table talk on the model only adds a line.')).classes('ll-muted')
        def save_moves():
            moves['value'] = store.set_game_character_moves(gid, moves_select.value, moves['value'])
            return moves['value']
        ctx.savebar.track(t('Character moves'), {moves_select: moves['value']}, save=save_moves, action='games.character_moves',
                          detail=lambda result: {'mode': result}, success=t('Character moves saved'),
                          dirty=lambda: moves_select.value != moves['value'],
                          reset=lambda: moves_select.set_value(moves['value']))
        limit = {'value': store.game_talk_daily_limit(gid)}
        limit_field = ui.number(t('Table talk calls per day'), value=limit['value'], min=1, precision=0, format='%d')
        ui.label(t('After about this many table talk calls in a server day (approximate when several tables play at once), characters play by the house rule and stay quiet until the next day. Used today: {count:,}.', count=store.game_talk_calls_today(gid))).classes('ll-muted')
        def save_limit():
            value = limit_field.value
            limit['value'] = store.set_game_talk_daily_limit(gid, int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value == int(value) else value, limit['value'])
            return limit['value']
        ctx.savebar.track(t('Table talk limit'), {limit_field: limit['value']}, save=save_limit, action='games.talk_daily_limit',
                          detail=lambda result: {'limit': result}, success=t('Table talk limit saved'),
                          dirty=lambda: limit_field.value != limit['value'],
                          reset=lambda: limit_field.set_value(limit['value']))
        games = store.game_channels(gid)
        options = {**{c: str(c) for c in sorted(games)}, **ctx.channel_names}
        picker = ui.select(options, value=sorted(games), multiple=True, label=t('Game channels')).props('use-chips').classes('w-full')
        def save_game_channels():
            nonlocal games
            wanted = {int(c) for c in picker.value or ()}
            result = save_channels(store, gid, wanted, games)
            games = wanted
            return result
        ctx.savebar.track(t('Game channels'), {picker: sorted(games)}, save=save_game_channels, action='games.channels',
                          detail=lambda result: {'channels': sorted(games), 'closed_tables': list(result['closed_tables'])},
                          dirty=lambda: {int(c) for c in picker.value or ()} != games,
                          reset=lambda: picker.set_value(sorted(games)),
                          then=lambda result: ui.notify(channels_toast(result, t=t), type='positive'))
        ui.label(t('Members play with /blackjack in game channels. Bets are taken when a member joins and paid when the round ends. Turning a channel off closes its open table and refunds the bets.')).classes('ll-muted')

        rounds = {'value': store.game_summary_rounds(gid)}
        with ui.element('div').classes('ll-form-row ll-field-row'):
            rounds_field = ui.number(t('Rounds in the end-of-table summary'), value=rounds['value'], min=1, max=20, precision=0, format='%d').classes('ll-wide')
        ui.label(t('When a table closes, its message lists this many of the last rounds.')).classes('ll-muted')
        def save_rounds():
            rounds['value'] = store.set_game_summary_rounds(gid, whole(rounds_field.value), rounds['value'])
            return rounds['value']
        ctx.savebar.track(t('Summary rounds'), {rounds_field: rounds['value']}, save=save_rounds, action='games.summary_rounds',
                          detail=lambda result: {'rounds': result}, success=t('Summary rounds saved'),
                          dirty=lambda: rounds_field.value != rounds['value'],
                          reset=lambda: rounds_field.set_value(rounds['value']))

        sees = {'value': store.cast_sees_item_results(gid)}
        sees_switch = ui.switch(t('Cast sees item results (Experimental)'), value=sees['value'])
        ui.label(t('When on, characters in a channel can see the last few results (8-ball, fortune cookie, dice, tarot\u2026) used there in the past 30 minutes, so they can mention them when asked. '
                   'Results are already public in the channel; the cast sees only the item name and result.')).classes('ll-muted')
        def save_sees():
            sees['value'] = store.set_cast_sees_item_results(gid, bool(sees_switch.value), sees['value'])
            return sees['value']
        ctx.savebar.track(t('Cast sees item results'), {sees_switch: sees['value']}, save=save_sees, action='games.cast_sees_item_results',
                          detail=lambda result: {'enabled': result}, success=t('Cast sees item results saved'),
                          dirty=lambda: bool(sees_switch.value) != sees['value'],
                          reset=lambda: sees_switch.set_value(sees['value']))

        watches = {'value': store.cast_watches_games(gid)}
        watches_switch = ui.switch(t('Cast watches games (Experimental)'), value=watches['value'])
        ui.label(t('Characters who are not seated can see the public state of an open game table in normal chat and answer when addressed. Never shows hands or face-down cards.')).classes('ll-muted')
        def save_watches():
            watches['value'] = store.set_cast_watches_games(gid, bool(watches_switch.value), watches['value'])
            return watches['value']
        ctx.savebar.track(t('Cast watches games'), {watches_switch: watches['value']}, save=save_watches, action='games.cast_watches_games',
                          detail=lambda result: {'enabled': result}, success=t('Cast watches games saved'),
                          dirty=lambda: bool(watches_switch.value) != watches['value'],
                          reset=lambda: watches_switch.set_value(watches['value']))


def blackjack_section(ctx):
    from nicegui import ui
    from .dashboard import section
    from .games import blackjack
    t = web_locale.current()
    game = terms.GAME_BLACKJACK
    store, gid = ctx.store, ctx.guild_id
    with section(t(game)):
        on = {'value': store.blackjack_enabled(gid)}
        switch = ui.switch(t('{game} is on', game=game), value=on['value'])
        ui.label(t('Off: /blackjack is refused and open blackjack tables close with every bet refunded.')).classes('ll-muted')
        def save_on():
            result = store.set_blackjack_enabled(gid, bool(switch.value), on['value'])
            on['value'] = result['enabled']
            return result
        def toast(result):
            closed = len(result['closed_tables'])
            if closed:
                ui.notify(t.n('{game} is off. Closed {n} open table.', '{game} is off. Closed {n} open tables.', closed, game=game, n=closed), type='positive')
        ctx.savebar.track(t(game), {switch: on['value']}, save=save_on, action='games.blackjack_enabled',
                          detail=lambda result: dict(result), success=t('{game} setting saved', game=game),
                          dirty=lambda: bool(switch.value) != on['value'],
                          reset=lambda: switch.set_value(on['value']),
                          then=toast)

        rules = {'value': store.blackjack_rules(gid)}
        ties = {'push': t('Push: bet comes back'), 'dealer': t('Dealer wins')}
        with ui.element('div').classes('ll-form-row ll-field-row'):
            stand_on = ui.select({16: '16', 17: '17', 18: '18'}, value=rules['value']['stand_on'], label=t('Dealer stands on')).classes('ll-wide')
            pays = ui.select({'3:2': '3:2', '6:5': '6:5'}, value=rules['value']['blackjack_pays'], label=t('Blackjack pays'))
            tie = ui.select(ties, value=rules['value']['ties'], label=t.p('field', 'Ties')).classes('ll-wide')
        soft = ui.switch(t('Dealer hits soft 17'), value=rules['value']['hit_soft_17'])
        insurance = ui.switch(t('Insurance and even money'), value=rules['value']['insurance'])
        surrender = ui.switch(t('Late surrender'), value=rules['value']['surrender'])
        controls = (stand_on, soft, pays, tie, insurance, surrender)
        def current():
            return {'stand_on': stand_on.value, 'hit_soft_17': bool(soft.value) and stand_on.value == 17, 'blackjack_pays': pays.value, 'ties': tie.value,
                    'insurance': bool(insurance.value), 'surrender': bool(surrender.value)}
        preview = ui.label().classes('ll-muted')
        rules_name = t('{game} rules', game=game)
        def refresh(_=None):
            if stand_on.value != 17:
                soft.set_value(False)
            soft.set_enabled(stand_on.value == 17 and (ctx.savebar.active is None or ctx.savebar.active.name == rules_name))
            rules_text = ' · '.join(t.render(p) for p in blackjack.rules_phrases(current()))
            preview.set_text(t('Table rules line: {rules}', rules=rules_text))
        for control in controls:
            control.on_value_change(refresh)
        refresh()
        ui.label(t('Rule changes apply from the next round. Open rounds keep the rules they were dealt with.')).classes('ll-muted')
        def save_rules():
            result = store.set_blackjack_rules(gid, current(), expected=rules['value'])
            rules['value'] = result
            return result
        def reset_rules():
            saved = rules['value']
            for control, key in zip(controls, ('stand_on', 'hit_soft_17', 'blackjack_pays', 'ties', 'insurance', 'surrender')):
                control.set_value(saved[key])
        ctx.savebar.track(rules_name, {c: rules['value'][k] for c, k in zip(controls, ('stand_on', 'hit_soft_17', 'blackjack_pays', 'ties', 'insurance', 'surrender'))},
                          save=save_rules, action='games.blackjack_rules', detail=lambda result: dict(result), success=t('{game} rules saved', game=game),
                          dirty=lambda: current() != rules['value'], reset=reset_rules, then=lambda result: refresh(),
                          enabled=lambda control: control is not soft or stand_on.value == 17)  # soft 17 only exists when the dealer stands on 17

        spoken = {'value': store.spoken_moves_enabled(gid)}
        spoken_switch = ui.switch(t('Read moves from chat (Experimental)'), value=spoken['value'])
        ui.label(t("On: when it's a member's turn at a blackjack table, the bot reads their chat message and, if it looks like a move, replies with buttons to confirm it (for example Hit / Just table talk). "
                   'Nothing is played until they press a button. English and Korean. Experimental: it can miss or misread moves, and it loads a small language model (about 0.5 GB of memory) the first time it reads a message.')).classes('ll-muted')
        def save_spoken():
            spoken['value'] = store.set_spoken_moves_enabled(gid, bool(spoken_switch.value), spoken['value'])
            return spoken['value']
        ctx.savebar.track(t('Read moves from chat'), {spoken_switch: spoken['value']}, save=save_spoken, action='games.spoken_moves',
                          detail=lambda result: {'enabled': result}, success=t('Read moves from chat saved'),
                          dirty=lambda: bool(spoken_switch.value) != spoken['value'],
                          reset=lambda: spoken_switch.set_value(spoken['value']))

        from .spoken.veto_client import PROVIDER
        veto = {'value': store.spoken_veto_enabled(gid)}
        veto_switch = ui.switch(t('Decision-model check for spoken moves (Experimental)'), value=veto['value'])
        ui.label(t('When on, a message that the local reader takes for a spoken move is sent to {provider}, a third-party service, to double-check it. '
                   'Only the message text and the legal moves are sent, only for members who turned on spoken moves, and only on their own turn. Off by default.', provider=PROVIDER)).classes('ll-muted')
        def save_veto():
            veto['value'] = store.set_spoken_veto_enabled(gid, bool(veto_switch.value), veto['value'])
            return veto['value']
        ctx.savebar.track(t('Decision-model check for spoken moves'), {veto_switch: veto['value']}, save=save_veto, action='games.spoken_veto',
                          detail=lambda result: {'enabled': result}, success=t('Decision-model check for spoken moves saved'),
                          dirty=lambda: bool(veto_switch.value) != veto['value'],
                          reset=lambda: veto_switch.set_value(veto['value']))

        from .icons import lucide_button
        from .scene_ui import confirm_dialog
        labels = _move_labels(t)
        move_options = {**labels, 'talk': t('Table talk')}
        ui.label(t('Tuning')).classes('ll-subtitle')
        keep = {'value': store.spoken_keep_setting(gid)}
        keep_switch = ui.switch(t('Keep phrases for tuning'), value=keep['value'])
        ui.label(t('On: when a move check was wrong (the member pressed Just table talk or picked the other move) or missed a move they then played with a table button, the bot keeps that chat message here '
                   'so you can turn it into an example phrase for this server. Only while Read moves from chat is on. Members can turn it off for themselves with /game spoken_moves off. '
                   'Turning this off deletes every kept phrase. Kept phrases are deleted after the history retention period.')).classes('ll-muted')
        def save_keep():
            keep['value'] = store.set_spoken_keep_phrases(gid, bool(keep_switch.value), keep['value'])
            return keep['value']
        def confirm_keep():
            return keep_confirm(store, gid, keep['value'], bool(keep_switch.value), t=t)
        ctx.savebar.track(t('Keep phrases for tuning'), {keep_switch: keep['value']}, save=save_keep, action='games.spoken_keep',
                          detail=lambda result: {'enabled': result}, success=t('Keep phrases for tuning saved'),
                          dirty=lambda: bool(keep_switch.value) != keep['value'],
                          reset=lambda: keep_switch.set_value(keep['value']),
                          then=lambda result: lists.refresh(), confirm=confirm_keep)

        def actor():
            return int(ctx.app.state.sessions[ctx.ident]['user']['id'])

        async def refreshed(result=None):
            lists.refresh()

        def gone(done):
            return lambda result: (ui.notify(done if result else t('Already removed.'), type='positive' if result else 'warning'), lists.refresh())[1]

        @ui.refreshable
        def lists():
            revision = store.spoken_examples_revision(gid)
            ui.label(t('Kept phrases')).classes('ll-subtitle')
            rows = phrase_rows(store.spoken_phrases(gid, 'blackjack', 200), store.guild_timezone(gid), t=t)
            if not rows:
                ui.label(t('No kept phrases yet.')).classes('ll-muted')
            for row in rows:
                with ui.card().classes('w-full ll-stack ll-result ll-spoken-row'):
                    ui.label(f"\u201c{row['text']}\u201d").classes('break-words')
                    ui.label(f"{row['check']} \u00b7 {row['meant']} \u00b7 {row['when']}").classes('ll-muted')
                    with ui.element('div').classes('ll-form-row ll-field-row'):
                        pick = ui.select(move_options, value=row['label'], label=t.p('field', 'Move')).classes('ll-wide')
                        ctx.button(t('Make example'), lambda row=row, pick=pick: refresh_on_refusal(lambda: store.promote_spoken_phrase(gid, row['id'], pick.value, actor(), revision)),
                                   'games.spoken_example_add', lambda result, pick=pick: {'label': pick.value}, then=gone(t('Example phrase added')), success=None,
                                   conflict=refreshed).props('outline size=sm')
                        lucide_button(t.p('button', 'Delete'), 'trash-2', on_click=lambda row=row: confirm_dialog(
                            ctx, t('Delete kept phrase'), [t('This message is deleted from the kept list.')], t.p('button', 'Delete'),
                            lambda: store.delete_spoken_phrase(gid, row['id']), 'games.spoken_phrase_delete', None,
                            then=gone(t('Kept phrase deleted')), conflict=refreshed)).props('outline size=sm color=negative')
            ui.label(t('Example phrases for this server')).classes('ll-subtitle')
            ui.label(t("When a chat message is very close to one of these, the bot uses this example's move instead of its own guess. Up to 200.")).classes('ll-muted')
            examples = store.spoken_examples(gid, 'blackjack')
            if not examples:
                ui.label(t('No example phrases yet.')).classes('ll-muted')
            for example in examples:
                with ui.card().classes('w-full ll-stack ll-result ll-spoken-example'):
                    ui.label(f"\u201c{example['text']}\u201d").classes('break-words')
                    with ui.element('div').classes('ll-form-row ll-field-row'):
                        ui.label(_move_name(example['label'], labels, t=t)).classes('ll-muted')
                        lucide_button(t.p('button', 'Remove'), 'trash-2', on_click=lambda example=example: confirm_dialog(
                            ctx, t('Remove example phrase'), [t("The bot stops using this example for this server's move checks.")], t.p('button', 'Remove'),
                            lambda: store.delete_spoken_example(gid, example['id'], revision), 'games.spoken_example_delete', None,
                            then=gone(t('Example phrase removed')), conflict=refreshed)).props('outline size=sm color=negative')
        lists()


def keep_confirm(store, gid, was_on, now_on, *, t=None):
    """The savebar question before turning keeping off: how many phrases (all games) go. A failed count still asks, without a number."""
    t = t or i18n.english()
    if not (was_on and not now_on):
        return None
    try:
        count = len(store.spoken_phrases(gid, None, store.SPOKEN_PHRASES_MAX))
    except Exception:
        logging.exception('Could not count kept phrases')
        return t('Turn off keeping?'), [t('Every kept phrase is deleted. Example phrases stay.')], t('Turn off and delete')
    if count:
        return t('Turn off keeping?'), [t.n('{count} kept phrase is deleted. Example phrases stay.', '{count} kept phrases are deleted. Example phrases stay.', count)], t('Turn off and delete')
    return None


def refresh_on_refusal(operation):
    """Runs a store call; a refusal (for example the phrase is gone) is re-raised as a conflict so the button refreshes the lists, with the same message."""
    try:
        return operation()
    except ConflictError:
        raise
    except ValueError as error:
        if getattr(error, 'code', None) is not None:
            raise ConflictError(error.code, **error.params) from error
        raise ConflictError(str(error)) from error


def doubt_section(ctx):
    from nicegui import ui
    from .dashboard import section
    t = web_locale.current()
    game = terms.GAME_DOUBT
    store, gid = ctx.store, ctx.guild_id
    with section(t(game)):
        on = {'value': store.game_enabled(gid, 'doubt')}
        switch = ui.switch(t('{game} is on', game=game), value=on['value'])
        ui.label(t('Off: /game start refuses I Doubt It, and open I Doubt It tables close with every ante refunded.')).classes('ll-muted')
        def save_on():
            result = store.set_game_enabled(gid, 'doubt', bool(switch.value), on['value'])
            on['value'] = result['enabled']
            return result
        def toast(result):
            closed = len(result['closed_tables'])
            if closed:
                ui.notify(t.n('{game} is off. Closed {n} open table.', '{game} is off. Closed {n} open tables.', closed, game=game, n=closed), type='positive')
        ctx.savebar.track(t(game), {switch: on['value']}, save=save_on, action='games.doubt_enabled',
                          detail=lambda result: dict(result), success=t('{game} setting saved', game=game),
                          dirty=lambda: bool(switch.value) != on['value'],
                          reset=lambda: switch.set_value(on['value']),
                          then=toast)

        rules = {'value': store.game_rules(gid, 'doubt')}
        with ui.element('div').classes('ll-form-row ll-field-row'):
            ante = ui.number(t.p('field', 'Ante'), value=rules['value']['ante'], min=0, max=1_000_000_000, precision=0, format='%d').classes('ll-wide')
            timeout = ui.number(t('Turn timeout (seconds)'), value=rules['value']['turn_timeout'], min=15, max=300, precision=0, format='%d').classes('ll-wide')
            ui.input(t('Claim rule'), value=t.p('claim rule', 'Sequence')).props('disable').classes('ll-wide')
        ui.label(t('Ante: each player pays this into the pot when joining; the winner takes the pot. 0 = no ante. Turn timeout: a player who runs out of time plays one card automatically; two automatic plays in a row forfeit the seat. Claims follow the sequence A, 2 … K, then A again. Changes apply to tables opened afterwards.')).classes('ll-muted')
        controls = (ante, timeout)
        keys = ('ante', 'turn_timeout')
        def current():
            return {'claim_rule': 'sequence', 'ante': _whole(ante.value), 'turn_timeout': _whole(timeout.value)}
        def save_rules():
            result = store.set_game_rules(gid, 'doubt', current(), expected=rules['value'])
            rules['value'] = result
            return result
        ctx.savebar.track(t('{game} rules', game=game), {c: rules['value'][k] for c, k in zip(controls, keys)},
                          save=save_rules, action='games.doubt_rules', detail=lambda result: dict(result), success=t('{game} rules saved', game=game),
                          dirty=lambda: current() != rules['value'],
                          reset=lambda: [c.set_value(rules['value'][k]) for c, k in zip(controls, keys)])

        spoken = {'value': store.spoken_doubt_enabled(gid)}
        spoken_switch = ui.switch(t('Read doubts from chat (Experimental)'), value=spoken['value'])
        ui.label(t('On: during the few seconds after a claim, when a member at the table writes something like "liar!" or "구라", the bot replies with buttons to confirm it (Doubt it! / Just table talk). '
                   'Nothing is played until they press a button, and the doubt window does not wait. English and Korean, by word list only (no language model). '
                   'Members can turn move checks off for themselves with /game spoken_moves.')).classes('ll-muted')
        def save_spoken():
            spoken['value'] = store.set_spoken_doubt_enabled(gid, bool(spoken_switch.value), spoken['value'])
            return spoken['value']
        ctx.savebar.track(t('Read doubts from chat'), {spoken_switch: spoken['value']}, save=save_spoken, action='games.spoken_doubt',
                          detail=lambda result: {'enabled': result}, success=t('Read doubts from chat saved'),
                          dirty=lambda: bool(spoken_switch.value) != spoken['value'],
                          reset=lambda: spoken_switch.set_value(spoken['value']))


def parse_phrases(text):
    """Declaration phrases from a textarea: one per line, trimmed, blank lines dropped, repeats dropped (first kept)."""
    return list(dict.fromkeys(line.strip() for line in str(text or '').splitlines() if line.strip()))


ONECARD_LAST_CARD = {'none': N_('No restriction'), 'no_attack': N_('No attack cards'), 'no_attack_or_joker': N_('No attack cards or Jokers')}


def onecard_section(ctx):
    from nicegui import ui
    from .dashboard import section
    from .games import onecard
    t = web_locale.current()
    game = terms.GAME_ONECARD
    store, gid = ctx.store, ctx.guild_id
    with section(t(game)):
        on = {'value': store.game_enabled(gid, 'onecard')}
        switch = ui.switch(t('Enabled'), value=on['value'])
        ui.label(t('Off: /game start refuses One Card, and open One Card tables close with every ante refunded.')).classes('ll-muted')
        def save_on():
            result = store.set_game_enabled(gid, 'onecard', bool(switch.value), on['value'])
            on['value'] = result['enabled']
            return result
        def toast(result):
            closed = len(result['closed_tables'])
            if closed:
                ui.notify(t.n('{game} is off. Closed {n} open table.', '{game} is off. Closed {n} open tables.', closed, game=game, n=closed), type='positive')
        ctx.savebar.track(t(game), {switch: on['value']}, save=save_on, action='games.onecard_enabled',
                          detail=lambda result: dict(result), success=t('{game} setting saved', game=game),
                          dirty=lambda: bool(switch.value) != on['value'],
                          reset=lambda: switch.set_value(on['value']),
                          then=toast)

        rules = {'value': store.game_rules(gid, 'onecard')}
        value = rules['value']
        with ui.element('div').classes('ll-form-row ll-field-row'):
            ante = ui.number(t.p('field', 'Ante'), value=value['ante'], min=0, max=onecard.ANTE_MAX, precision=0, format='%d').classes('ll-wide')
            timeout = ui.number(t('Turn timeout (seconds)'), value=value['turn_timeout'], min=onecard.TIMEOUT_RANGE[0], max=onecard.TIMEOUT_RANGE[1], precision=0, format='%d').classes('ll-wide')
            penalty = ui.number(t('Call-out penalty'), value=value['call_out_penalty'], min=onecard.PENALTY_RANGE[0], max=onecard.PENALTY_RANGE[1], precision=0, format='%d').classes('ll-wide')
            last_card = ui.select({k: t(v) for k, v in ONECARD_LAST_CARD.items()}, label=t('Last card restriction'), value=value['last_card']).classes('ll-wide')
        spade = ui.switch(t('Spade Ace counts +10'), value=value['spade_ace_ten'])
        defense = ui.switch(t('Defense chain'), value=value['defense'])
        king = ui.switch(t('King skips two seats'), value=value['king_skips_two'])
        bankruptcy = ui.switch(t('Bankruptcy'), value=value['bankruptcy'])
        limit = ui.number(t('Bankruptcy limit'), value=value['bankruptcy_limit'], min=onecard.BANKRUPTCY_RANGE[0], max=onecard.BANKRUPTCY_RANGE[1], precision=0, format='%d').classes('ll-wide')
        limit.bind_enabled_from(bankruptcy, 'value')
        phrases = ui.textarea(t('Declaration phrases (one per line)'), value='\n'.join(value['declare_phrases'])).classes('ll-wide')
        ui.label(t('Ante: each player pays this into the pot when joining; the winner takes the pot. 0 = no ante. Turn timeout: a player who runs out of time draws automatically. '
                   'Call-out penalty: cards drawn by a player called out for not declaring. Last card restriction: what cannot be played as the final card. '
                   'Bankruptcy: a player holding the limit or more cards is out. Declaration phrases: what a player writes in chat to declare One Card (up to 8 phrases, each up to 30 characters). Changes apply to tables opened afterwards.')).classes('ll-muted')
        controls = (ante, timeout, penalty, last_card, spade, defense, king, bankruptcy, limit, phrases)
        keys = ('ante', 'turn_timeout', 'call_out_penalty', 'last_card', 'spade_ace_ten', 'defense', 'king_skips_two', 'bankruptcy', 'bankruptcy_limit', 'declare_phrases')
        def shown(key, v):
            return '\n'.join(v) if key == 'declare_phrases' else v
        def current():
            return {'ante': _whole(ante.value), 'turn_timeout': _whole(timeout.value), 'call_out_penalty': _whole(penalty.value), 'last_card': last_card.value,
                    'spade_ace_ten': bool(spade.value), 'defense': bool(defense.value), 'king_skips_two': bool(king.value), 'bankruptcy': bool(bankruptcy.value),
                    'bankruptcy_limit': _whole(limit.value), 'declare_phrases': parse_phrases(phrases.value)}
        def save_rules():
            found = parse_phrases(phrases.value)
            if not found:
                raise UserError(codes.ONECARD_PHRASE_NEEDED)
            if len(found) > onecard.PHRASES_MAX:
                raise UserError(codes.ONECARD_PHRASES_MAX, limit=onecard.PHRASES_MAX)
            result = store.set_game_rules(gid, 'onecard', current(), expected=rules['value'])
            rules['value'] = result
            return result
        ctx.savebar.track(t('{game} rules', game=game), {c: shown(k, rules['value'][k]) for c, k in zip(controls, keys)},
                          save=save_rules, action='games.onecard_rules', detail=lambda result: dict(result), success=t('{game} rules saved', game=game),
                          dirty=lambda: current() != rules['value'],
                          reset=lambda: [c.set_value(shown(k, rules['value'][k])) for c, k in zip(controls, keys)])


def yut_section(ctx):
    from nicegui import ui
    from .dashboard import section
    from .games import yut
    t = web_locale.current()
    game = terms.GAME_YUT
    store, gid = ctx.store, ctx.guild_id
    with section(t(game)):
        on = {'value': store.game_enabled(gid, 'yut')}
        switch = ui.switch(t('Enabled'), value=on['value'])
        ui.label(t('Off: /game start refuses Yut, and open Yut tables close with every ante refunded.')).classes('ll-muted')
        def save_on():
            result = store.set_game_enabled(gid, 'yut', bool(switch.value), on['value'])
            on['value'] = result['enabled']
            return result
        def toast(result):
            closed = len(result['closed_tables'])
            if closed:
                ui.notify(t.n('{game} is off. Closed {n} open table.', '{game} is off. Closed {n} open tables.', closed, game=game, n=closed), type='positive')
        ctx.savebar.track(t(game), {switch: on['value']}, save=save_on, action='games.yut_enabled',
                          detail=lambda result: dict(result), success=t('{game} setting saved', game=game),
                          dirty=lambda: bool(switch.value) != on['value'],
                          reset=lambda: switch.set_value(on['value']),
                          then=toast)

        rules = {'value': store.game_rules(gid, 'yut')}
        value = rules['value']
        with ui.element('div').classes('ll-form-row ll-field-row'):
            ante = ui.number(t.p('field', 'Ante'), value=value['ante'], min=0, max=yut.ANTE_MAX, precision=0, format='%d').classes('ll-wide')
            timeout = ui.number(t('Turn timeout (seconds)'), value=value['turn_timeout'], min=yut.TIMEOUT_RANGE[0], max=yut.TIMEOUT_RANGE[1], precision=0, format='%d').classes('ll-wide')
            pieces = ui.number(t('Pieces per player'), value=value['pieces'], min=yut.PIECES_RANGE[0], max=yut.PIECES_RANGE[1], precision=0, format='%d').classes('ll-wide')
        backdo = ui.switch(t('Back-do (빽도) throw'), value=value['backdo'])
        ui.label(t('Ante: each player pays this into the pot when joining; the winner takes the pot. 0 = no ante. Turn timeout: a player who runs out of time has the throw or move made automatically. '
                   'Pieces per player: how many pieces each player must bring home. Back-do: allow the throw that moves a piece one step back. Changes apply to tables opened afterwards.')).classes('ll-muted')
        controls = (ante, timeout, pieces, backdo)
        keys = ('ante', 'turn_timeout', 'pieces', 'backdo')
        def current():
            return {'ante': _whole(ante.value), 'turn_timeout': _whole(timeout.value), 'pieces': _whole(pieces.value), 'backdo': bool(backdo.value)}
        def save_rules():
            result = store.set_game_rules(gid, 'yut', current(), expected=rules['value'])
            rules['value'] = result
            return result
        ctx.savebar.track(t('{game} rules', game=game), {c: rules['value'][k] for c, k in zip(controls, keys)},
                          save=save_rules, action='games.yut_rules', detail=lambda result: dict(result), success=t('{game} rules saved', game=game),
                          dirty=lambda: current() != rules['value'],
                          reset=lambda: [c.set_value(rules['value'][k]) for c, k in zip(controls, keys)])


def holdem_section(ctx):
    from nicegui import ui
    from .dashboard import section
    t = web_locale.current()
    game = terms.GAME_HOLDEM
    store, gid = ctx.store, ctx.guild_id
    with section(t(game)):
        on = {'value': store.game_enabled(gid, 'holdem')}
        switch = ui.switch(t('{game} is on', game=game), value=on['value'])
        ui.label(t("Off: /game start refuses Texas Hold'em, and open Hold'em tables close with every stack cashed out.")).classes('ll-muted')
        def save_on():
            result = store.set_game_enabled(gid, 'holdem', bool(switch.value), on['value'])
            on['value'] = result['enabled']
            return result
        def toast(result):
            closed = len(result['closed_tables'])
            if closed:
                ui.notify(t.n('{game} is off. Closed {n} open table.', '{game} is off. Closed {n} open tables.', closed, game=game, n=closed), type='positive')
        ctx.savebar.track(t(game), {switch: on['value']}, save=save_on, action='games.holdem_enabled',
                          detail=lambda result: dict(result), success=t('{game} setting saved', game=game),
                          dirty=lambda: bool(switch.value) != on['value'],
                          reset=lambda: switch.set_value(on['value']),
                          then=toast)

        rules = {'value': store.game_rules(gid, 'holdem')}
        with ui.element('div').classes('ll-form-row ll-field-row'):
            blind = ui.number(t('Small blind'), value=rules['value']['small_blind'], min=1, max=1_000_000, precision=0, format='%d').classes('ll-wide')
            low = ui.number(t('Smallest buy-in (big blinds)'), value=rules['value']['min_buy_in_bb'], min=20, max=1000, precision=0, format='%d').classes('ll-wide')
            high = ui.number(t('Largest buy-in (big blinds)'), value=rules['value']['max_buy_in_bb'], min=20, max=1000, precision=0, format='%d').classes('ll-wide')
            timeout = ui.number(t('Action timeout (seconds)'), value=rules['value']['action_timeout'], min=15, max=300, precision=0, format='%d').classes('ll-wide')
        assist = ui.switch(t('Allow assist'), value=rules['value']['assist'])
        ui.label(t('The big blind is twice the small blind. Buy-ins are counted in big blinds. A player who runs out of action time checks, or folds when there is a bet. Allow assist lets the host turn on private hand and odds hints for the table. Changes apply to hands dealt afterwards.')).classes('ll-muted')
        controls = (blind, low, high, timeout, assist)
        keys = ('small_blind', 'min_buy_in_bb', 'max_buy_in_bb', 'action_timeout', 'assist')
        def current():
            return {'small_blind': _whole(blind.value), 'min_buy_in_bb': _whole(low.value), 'max_buy_in_bb': _whole(high.value),
                    'action_timeout': _whole(timeout.value), 'assist': bool(assist.value)}
        def save_rules():
            result = store.set_game_rules(gid, 'holdem', current(), expected=rules['value'])
            rules['value'] = result
            return result
        ctx.savebar.track(t('{game} rules', game=game), {c: rules['value'][k] for c, k in zip(controls, keys)},
                          save=save_rules, action='games.holdem_rules', detail=lambda result: dict(result), success=t('{game} rules saved', game=game),
                          dirty=lambda: current() != rules['value'],
                          reset=lambda: [c.set_value(rules['value'][k]) for c, k in zip(controls, keys)])


VAULT_KEYS = ('enabled', 'protocols', 'turns', 'reward', 'advice_cap', 'clue_voice')
OPERATION_KEYS = ('enabled', 'channel_id', 'week_start', 'stage_protocols', 'contribution_reward', 'team_reward', 'min_guesses', 'support_reward')


def vault_form(store, gid):
    settings = store.vault_settings(gid)
    return {**{k: settings[k] for k in VAULT_KEYS}, 'revision': settings['revision']}


def operation_form(store, gid):
    return dict(store.vault_operation_settings(gid))


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return _whole(value)


def save_vault(store, gid, form, base):
    changes = {k: _clean(form[k]) for k in VAULT_KEYS if k != 'turns' and form[k] is not None and _clean(form[k]) != base[k]}
    turns = {p: _clean(v) for p, v in form['turns'].items() if v is not None and _clean(v) != base['turns'].get(p)}
    if turns:
        changes['turns'] = turns
    if not changes:
        return store.vault_settings(gid)
    return store.set_vault_settings(gid, **changes, expected_revision=base['revision'])


def save_operation(store, gid, form, base):
    changes = {k: _clean(form[k]) for k in OPERATION_KEYS if form[k] is not None and _clean(form[k]) != base[k]}
    if not changes:
        return store.vault_operation_settings(gid)
    return store.set_vault_operation_settings(gid, **changes, expected_revision=base['revision'])


def vault_channel_options(channel_names, thread_names, current=None):
    options = {c: name for c, name in channel_names.items() if c not in (thread_names or {})}
    if current is not None and current not in options:
        options[current] = str(current)
    return options


def week_start_options(t):
    days = (t('Monday'), t('Tuesday'), t('Wednesday'), t('Thursday'), t('Friday'), t('Saturday'), t('Sunday'))
    return dict(enumerate(days))


def vault_protocol_options(t):
    from .vault import PROTOCOLS
    from .i18n.terms import VAULT_PROTOCOLS as LABELS
    return {p: t.render(LABELS[p]) for p in PROTOCOLS}


def vault_section(ctx):
    from nicegui import ui
    from .dashboard import section
    from .vault import PROTOCOLS
    t = web_locale.current()
    store, gid = ctx.store, ctx.guild_id
    protocols = vault_protocol_options(t)
    with section(t('Vault (Experimental)')):
        ui.label(t('A private code-breaking puzzle: a member opens a vault with /vault personal, guesses a secret code with clues, and can bring character partners for advice. '
                   'Off by default. Tool prices are set in the Shop tab.')).classes('ll-muted')
        base = {'value': vault_form(store, gid)}
        form = base['value']
        enabled = ui.switch(t('Vault is on'), value=form['enabled'])
        ui.label(t('Off: /vault refuses to open a vault. Vaults already in progress are not touched.')).classes('ll-muted')
        allowed = ui.select(protocols, value=form['protocols'], multiple=True, label=t('Allowed protocols')).props('use-chips').classes('w-full')
        ui.label(t('Applies to personal vaults only; the three stages of the weekly operation choose their protocols separately. Standard and Open are on by default. Total Match and Position-Only give much less feedback per guess, so they are harder: turn them on only if your members want that.')).classes('ll-muted')
        with ui.element('div').classes('ll-form-row ll-field-row'):
            turns = {p: ui.number(t('Turns: {protocol}', protocol=protocols[p]), value=form['turns'][p], min=1, max=50, precision=0, format='%d').classes('ll-wide') for p in PROTOCOLS}
        with ui.element('div').classes('ll-form-row ll-field-row'):
            reward = ui.number(t('Reward for solving a vault'), value=form['reward'], min=0, precision=0, format='%d').classes('ll-wide')
            cap = ui.number(t('Partner advice per vault'), value=form['advice_cap'], min=0, max=50, precision=0, format='%d').classes('ll-wide')
            voice = ui.select({'template': t('Template'), 'llm': t('Model')}, value=form['clue_voice'], label=t('Clue voice')).classes('ll-wide')
        ui.label(t('Turns: guesses a member gets in a personal vault. Reward: paid once when a member solves a personal vault; 0 = none. '
                   'Partner advice: how many times chosen partners may give advice in one vault. Clue voice: partners speak their clues from a template, or the model words them in character (uses the model).')).classes('ll-muted')
        controls = (enabled, allowed, reward, cap, voice, *turns.values())
        def current():
            return {'enabled': bool(enabled.value), 'protocols': [p for p in PROTOCOLS if p in (allowed.value or ())], 'turns': {p: turns[p].value for p in PROTOCOLS},
                    'reward': reward.value, 'advice_cap': cap.value, 'clue_voice': voice.value}
        def saved():
            return {**base['value'], 'turns': dict(base['value']['turns'])}
        def shown():
            v = base['value']
            return [v['enabled'], v['protocols'], v['reward'], v['advice_cap'], v['clue_voice'], *(v['turns'][p] for p in PROTOCOLS)]
        def save_settings():
            result = save_vault(store, gid, {**current(), 'revision': 0}, saved())
            base['value'] = vault_form(store, gid)
            return result
        def reset():
            for control, value in zip(controls, shown()):
                control.set_value(value)
        def dirty():
            return _clean(current()) != {k: base['value'][k] for k in VAULT_KEYS}
        ctx.savebar.track(t('Vault'), dict(zip(controls, shown())), save=save_settings, action='games.vault_settings',
                          detail=lambda result: {k: result[k] for k in VAULT_KEYS}, success=t('Vault settings saved'), dirty=dirty, reset=reset)

        ui.label(t('Weekly operation')).classes('ll-subtitle')
        ui.label(t('A weekly team operation in one channel: members join and contribute guesses, tools and cross-checks to crack three stages together.')).classes('ll-muted')
        obase = {'value': operation_form(store, gid)}
        oform = obase['value']
        op_on = ui.switch(t('Operation is on'), value=oform['enabled'])
        names = vault_channel_options(ctx.channel_names, ctx.thread_names, oform['channel_id'])
        channel = ui.select(names, value=oform['channel_id'], label=t('Operation channel')).classes('w-full')
        ui.label(t('A text channel; threads cannot be used. /vault operation works only there.')).classes('ll-muted')
        days = week_start_options(t)
        with ui.element('div').classes('ll-form-row ll-field-row'):
            week = ui.select(days, value=oform['week_start'], label=t('Week starts on')).classes('ll-wide')
        ui.label(t('The week turns over at midnight in the server time zone ({zone}).', zone=store.guild_timezone(gid) or 'UTC')).classes('ll-muted')
        with ui.element('div').classes('ll-form-row ll-field-row'):
            stages = [ui.select(protocols, value=oform['stage_protocols'][i], label=t('Stage {n} protocol', n=i + 1)).classes('ll-wide') for i in range(3)]
        with ui.element('div').classes('ll-form-row ll-field-row'):
            contribution = ui.number(t('Reward per contributor'), value=oform['contribution_reward'], min=0, precision=0, format='%d').classes('ll-wide')
            team = ui.number(t('Team reward'), value=oform['team_reward'], min=0, precision=0, format='%d').classes('ll-wide')
            support = ui.number(t('Support reward'), value=oform['support_reward'], min=0, precision=0, format='%d').classes('ll-wide')
            minimum = ui.number(t('Minimum contributions'), value=oform['min_guesses'], min=1, max=30, precision=0, format='%d').classes('ll-wide')
        ui.label(t('Minimum contributions: guesses, tool uses and cross-checks a member needs before being paid. Reward per contributor: paid once to a member who reaches the minimum. '
                   'Team reward: paid to each qualified member when all three stages are cleared. Support reward: paid for a solved support mission. '
                   'A solved support mission counts toward the team reward\'s minimum only, not toward the contribution reward. 0 = none.')).classes('ll-muted')
        ocontrols = (op_on, channel, week, *stages, contribution, team, support, minimum)
        def ocurrent():
            return {'enabled': bool(op_on.value), 'channel_id': channel.value, 'week_start': week.value, 'stage_protocols': [s.value for s in stages],
                    'contribution_reward': contribution.value, 'team_reward': team.value, 'min_guesses': minimum.value, 'support_reward': support.value}
        def oshown():
            v = obase['value']
            return [v['enabled'], v['channel_id'], v['week_start'], *v['stage_protocols'], v['contribution_reward'], v['team_reward'], v['support_reward'], v['min_guesses']]
        def save_op():
            result = save_operation(store, gid, {**ocurrent(), 'revision': 0}, obase['value'])
            obase['value'] = operation_form(store, gid)
            return result
        def oreset():
            for control, value in zip(ocontrols, oshown()):
                control.set_value(value)
        ctx.savebar.track(t('Weekly operation'), dict(zip(ocontrols, oshown())), save=save_op, action='games.vault_operation_settings',
                          detail=lambda result: {k: result[k] for k in OPERATION_KEYS}, success=t('Operation settings saved'),
                          dirty=lambda: _clean(ocurrent()) != {k: obase['value'][k] for k in OPERATION_KEYS}, reset=oreset)


GAME_SECTIONS = {'blackjack': blackjack_section, 'doubt': doubt_section, 'holdem': holdem_section, 'onecard': onecard_section, 'yut': yut_section}


async def games_panel(ctx):
    from .games import registry
    all_games_section(ctx)
    for key in registry.keys():
        if key in GAME_SECTIONS:
            GAME_SECTIONS[key](ctx)
    vault_section(ctx)
