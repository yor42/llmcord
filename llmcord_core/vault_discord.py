"""``/vault personal`` and ``/vault history``: the personal vault on Discord (FEAT-63 phase B, D43, Experimental).

Every message is private to the member (their locale). The command's response is the status message; each later action edits it in place
and sends the per-action line as an ephemeral followup. The store (``vault_store``) is the only source of truth and raises codes; this
layer takes input and builds the words. The secret is shown only from a terminal view and is never logged.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from types import SimpleNamespace

import discord
from discord import app_commands

from . import game_chooser
from .errors import error_detail
from .models import TurnMessage
from .names import resolve, suggest
from .usage import capture_usage, log_purpose, log_scope
from .vault import CountEven, CountLow, DigitPresent, PositionHalf, PositionParity
from .vault import TOOLS, build_advice_facts, spells_number
from .vault_store import MAX_TOOLS, VaultStoreError
from llmcord_core import i18n, items
from llmcord_core.i18n import N_, NP_, UserError, codes
from llmcord_core.i18n.discord_locale import C_, invoker_t
from llmcord_core.i18n.terms import VAULT_PROTOCOLS as PROTOCOLS

PREFIX = 'llmcord:vault'
NO_MENTIONS = discord.AllowedMentions.none()
NOT_YOURS = N_('This vault belongs to someone else. Run /vault personal to open your own.')
NO_VAULTS = N_('You have no vaults yet. Start one with /vault personal.')
SOMETHING_WRONG = N_('Something went wrong with the vault. Try again.')
TOOL_REASONS = {'bad_position': N_('Choose a position from 1 to 4.'), 'bad_digit': N_('Choose a single digit from 0 to 9.'), 'digits_equal': N_('Choose two different digits.'),
                'guess_not_in_history': N_('Choose one of your own guesses.'), 'digit_not_in_guess': N_('That digit is not in that guess.'),
                'nothing_new': N_('That guess has no new strike to show.'), 'already_revealed': N_('That position is already revealed.')}
ADVICE_SECONDS = 20
MAX_WORDING = 200
DIGIT_RUNS = re.compile(r'\d+')
ADVICE_SYSTEM = ('You are {name}, a partner helping a friend open a four-digit vault in a roleplay server. Stay in character.\n{persona}\n'
                 'Answer with one or two short sentences, at most ' + str(MAX_WORDING) + ' characters. Do not mention anyone with @. Return only JSON.')
ADVICE_USER = ('The recommended guess is {guess}.{remaining} Tell your friend to try {guess}, in your own words. '
               'Write the guess exactly as given and use no other numbers.')
ADVICE_COUNT = ' About {remaining} codes are still possible.'
CLUE_USER = ('You pass on this clue about the vault code: "{text}" Say it in your own words in one short sentence. '
             'Keep every number exactly as given and add no other numbers.')
LINE_SCHEMA = {'type': 'object', 'properties': {'line': {'type': 'string'}}, 'required': ['line'], 'additionalProperties': False}
STATUSES = {'active': NP_('vault status', 'In progress'), 'solved': NP_('vault status', 'Solved'),
            'failed': NP_('vault status', 'Failed'), 'expired': NP_('vault status', 'Expired')}


def feedback_text(protocol: str, result, *, t=None) -> str:
    t = t or i18n.english()
    if protocol == 'total_match':
        return t.n('{count} match', '{count} matches', result[0])
    if protocol == 'position_only':
        return t.n('{count} strike', '{count} strikes', result[0])
    return t('{strikes}, {balls}', strikes=t.n('{count} strike', '{count} strikes', result[0]), balls=t.n('{count} ball', '{count} balls', result[1]))


def clue_text(clue, *, t=None) -> str:
    t = t or i18n.english()
    if isinstance(clue, CountEven):
        return t.n('Exactly {count} digit is even.', 'Exactly {count} digits are even.', clue.count)
    if isinstance(clue, CountLow):
        return t.n('Exactly {count} digit is low (0 to 4).', 'Exactly {count} digits are low (0 to 4).', clue.count)
    if isinstance(clue, DigitPresent):
        return t('The code contains the digit {digit}.', digit=clue.digit) if clue.present else t('The code does not contain the digit {digit}.', digit=clue.digit)
    if isinstance(clue, PositionParity):
        return t('Digit {position} is odd.', position=clue.position) if clue.odd else t('Digit {position} is even.', position=clue.position)
    if isinstance(clue, PositionHalf):
        return t('Digit {position} is high (5 to 9).', position=clue.position) if clue.high else t('Digit {position} is low (0 to 4).', position=clue.position)
    return ''


def advice_line(entry: dict, t) -> str:
    """One piece of advice: the model's wording (shown as written, never translated) or the template line that names the guess."""
    name = entry['name'] or t('a deleted character')
    if entry['source'] == 'llm' and entry['text']:
        return t('{name}: {fact}', name=name, fact=entry['text'][:MAX_WORDING])
    return t.n('{name} suggests {guess}. {count} code is still possible.', '{name} suggests {guess}. {count} codes are still possible.',
               entry['remaining'], name=name, guess=entry['guess'])


def clue_wording_ok(wording, template: str) -> bool:
    return isinstance(wording, str) and bool(wording.strip()) and not spells_number(wording) and set(DIGIT_RUNS.findall(wording)) == set(DIGIT_RUNS.findall(template))


def tool_name(key: str, t) -> str:
    try:
        return items.display_name(items.get_item(key), t.locale)
    except KeyError:
        return key


def tool_fact(entry: dict, t) -> str:
    """The disclosed fact of one tool use, from the engine's text_key and data (deterministic template text)."""
    data, key = entry['data'], entry['text_key']
    if key in ('vault_scanner_present', 'vault_scanner_absent'):
        return t('The code contains the digit {digit}.', digit=data['digit']) if data['present'] else t('The code does not contain the digit {digit}.', digit=data['digit'])
    if key in ('vault_spectrum_high', 'vault_spectrum_low'):
        return t('Digit {position} is high (5 to 9).', position=data['position']) if data['high'] else t('Digit {position} is low (0 to 4).', position=data['position'])
    if key == 'vault_comparison_0':
        return t('Of the digits {a} and {b}, none is in the code.', a=data['a'], b=data['b'])
    if key == 'vault_comparison_1':
        return t('Of the digits {a} and {b}, exactly one is in the code.', a=data['a'], b=data['b'])
    if key == 'vault_comparison_2':
        return t('Of the digits {a} and {b}, both are in the code.', a=data['a'], b=data['b'])
    if key == 'vault_verifier_position':
        return t('In guess {guess}, digit {position} is {digit}.', guess=data['guess'], position=data['position'], digit=data['digit'])
    if key == 'vault_verifier_none':
        return t('Guess {guess} has no digit in the right position.', guess=data['guess'])
    if key == 'vault_decoder_digit':
        return t('Digit {position} is {digit}.', position=data['position'], digit=data['digit'])
    return ''


def tool_line(entry: dict, t) -> str:
    return t('{name}: {fact}', name=tool_name(entry['tool'], t), fact=tool_fact(entry, t))


def render_main(view: dict, *, t=None, tools_held: int = 0) -> str:
    """The status message. The secret line appears only once the vault has ended (the store withholds it until then)."""
    t = t or i18n.english()
    protocol = view['protocol']
    lines = [t('Personal vault ({date})', date=view['local_date']), t('Protocol: {protocol}', protocol=t.render(PROTOCOLS[protocol])),
             t('Status: {status}', status=t.render(STATUSES[view['status']])), t('Turns left: {left} of {budget}', left=view['turns_left'], budget=view['turn_budget'])]
    if view['tools_used'] or tools_held:
        lines.append(t('Tools used: {n} of {limit}', n=view['tools_used'], limit=MAX_TOOLS))
    if view['tools']:
        lines.append(t('Tool results:'))
        lines += [tool_line(entry, t) for entry in view['tools']]
    if view['open_digit']:
        lines.append(t('Open digit: {digit}', digit=view['open_digit']))
    deleted = t('a deleted character')
    if view['partners']:
        names = ', '.join(partner['name'] or deleted for partner in view['partners'])
        lines.append(t('Partners: {names}', names=names))
    if view['clues']:
        lines.append(t('Clues revealed: {n} of {total}', n=sum(1 for c in view['clues'] if c['revealed']), total=len(view['clues'])))
        for clue in view['clues']:
            if 'constraint' in clue:
                lines.append(t('Clue from {name}: {text}', name=clue['name'] or deleted, text=clue_text(clue['constraint'], t=t)))
    if view['advice']:
        lines.append(t('Partner advice:'))
        lines += [advice_line(entry, t) for entry in view['advice']]
    if view['guesses']:
        lines.append(t('Guesses:'))
        lines += [t('{n}. {guess} — {feedback}', n=g['number'], guess=g['guess'], feedback=feedback_text(protocol, g['feedback'], t=t)) for g in view['guesses']]
    if view['secret']:
        lines.append(t('Secret: {code}', code=view['secret']))
    return '\n'.join(lines)


def render_buttons(view: dict, *, t=None, tools_held: int = 0):
    """Guess, Reveal clue while a clue is hidden, and Tools while the member holds a tool; None once the vault is no longer active."""
    t = t or i18n.english()
    if view['status'] != 'active':
        return None
    buttons = discord.ui.View(timeout=None)
    buttons.add_item(VaultButton('guess', view['id'], t.p('button.vault', 'Guess'), discord.ButtonStyle.primary))
    if any(not c['revealed'] for c in view['clues']):
        buttons.add_item(VaultButton('clue', view['id'], t.p('button.vault', 'Reveal clue'), discord.ButtonStyle.secondary))
    if tools_held > 0:
        buttons.add_item(VaultButton('tools', view['id'], t.p('button.vault', 'Tools'), discord.ButtonStyle.secondary))
    asked = [p for p in view['partners'] if p['name']]
    for partner in asked:
        buttons.add_item(VaultButton(f"ask_{partner['character_id']}", view['id'], t.p('button.vault', 'Ask {name}', name=partner['name'])[:80]))
    if len(view['partners']) == 2 and len(asked) == 2:
        buttons.add_item(VaultButton('meeting', view['id'], t.p('button.vault', 'Strategy meeting')))
    return buttons


def render_picker(view: dict, held: dict, *, t=None):
    """One button per held tool (in TOOLS order) and Back."""
    t = t or i18n.english()
    buttons = discord.ui.View(timeout=None)
    for key, qty in held.items():
        cost = TOOLS[key].turn_cost
        name = tool_name(key, t)
        label = t.n('{name} ×{qty} ({count} turn)', '{name} ×{qty} ({count} turns)', cost, name=name, qty=qty)
        buttons.add_item(VaultButton(f'use_{key}', view['id'], label[:80]))
    buttons.add_item(VaultButton('back', view['id'], t.p('button.vault', 'Back')))
    return buttons


def guess_log(result: dict, currency: str, *, t=None) -> str:
    t = t or i18n.english()
    session = result['session']
    lines = [t('Guess {n}: {guess} — {feedback}', n=result['number'], guess=result['guess'], feedback=feedback_text(session['protocol'], result['feedback'], t=t))]
    if result['won']:
        lines.append(t('You opened the vault!'))
        if session['reward']:
            lines.append(t('Reward: {amount:,} {currency}', amount=session['reward']['amount'], currency=currency))
        elif result['reward_refused']:
            lines.append(t('Your reward could not be paid because your balance is at its limit.'))
    elif session['status'] == 'failed':
        lines.append(t('Out of turns. The vault stays closed today.'))
    return '\n'.join(lines)


class VaultButton(discord.ui.DynamicItem[discord.ui.Button], template=PREFIX + r':(?!op_)(?P<action>[a-z0-9_]+):(?P<session>\d+)'):
    """One button; its custom_id carries the action and the session, never a member or the code."""

    def __init__(self, action: str, session_id: int, label: str = '', style=discord.ButtonStyle.secondary, disabled: bool = False):
        super().__init__(discord.ui.Button(label=label, style=style, disabled=disabled, custom_id=f'{PREFIX}:{action}:{session_id}'))
        self.action, self.session_id = action, session_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match['action'], int(match['session']), item.label or '', item.style, item.disabled)

    async def callback(self, interaction: discord.Interaction):
        await interaction.client.vault.press(interaction, self.action, self.session_id)


class VaultGuessModal(discord.ui.Modal):
    """One code to type. Bound to the session it was opened for; the store checks that the submitter owns it."""

    def __init__(self, session_id: int, *, t=None):
        t = t or i18n.english()
        super().__init__(title=t.p('modal.vault', 'Guess the code')[:45], timeout=600)
        self.session_id = session_id
        self.guess = discord.ui.TextInput(label=t.p('modal.vault', 'Your guess')[:45], placeholder=t.p('modal.vault', '4 different digits')[:100], required=True, max_length=16)
        self.add_item(self.guess)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.client.vault.submit_guess(interaction, self.session_id, str(self.guess.value or ''))

    async def on_error(self, interaction: discord.Interaction, error: Exception):
        logging.error('Vault guess modal failed: %s', error_detail(error))
        await interaction.client.vault.say(interaction, invoker_t(interaction)(SOMETHING_WRONG))


TOOL_FIELDS = {'guess': N_('Past guess (number)'), 'digit': N_('Digit'), 'position': N_('Position (1 to 4)'), 'a': N_('First digit'), 'b': N_('Second digit')}


class VaultToolModal(discord.ui.Modal):
    """The inputs of one Vault tool, named after the engine's params. Bound to the session it was opened for."""

    def __init__(self, session_id: int, tool_key: str, *, t=None):
        t = t or i18n.english()
        super().__init__(title=tool_name(tool_key, t)[:45], timeout=600)
        self.session_id, self.tool_key = session_id, tool_key
        self.inputs = {name: discord.ui.TextInput(label=t.render(TOOL_FIELDS[name])[:45], required=True, max_length=16) for name in TOOLS[tool_key].needs}
        for item in self.inputs.values():
            self.add_item(item)

    async def on_submit(self, interaction: discord.Interaction):
        values = {name: str(item.value or '') for name, item in self.inputs.items()}
        await interaction.client.vault.submit_tool(interaction, self.session_id, self.tool_key, values)

    async def on_error(self, interaction: discord.Interaction, error: Exception):
        logging.error('Vault tool modal failed: %s', error_detail(error))
        await interaction.client.vault.say(interaction, invoker_t(interaction)(SOMETHING_WRONG))


class VaultPlay:
    def __init__(self, bot, clock=time.time):
        self.bot, self.clock = bot, clock
        self.talk_reserved: dict[int, int] = {}  # clue-voice talk calls in flight per guild (usage rows land when the call returns)

    @property
    def store(self):
        return self.bot.store

    async def say(self, interaction, text: str) -> None:
        """A private reply: the first response, or a followup once the interaction was acknowledged."""
        try:
            if interaction.response.is_done():
                await interaction.followup.send(text, ephemeral=True, allowed_mentions=NO_MENTIONS)
            else:
                await interaction.response.send_message(text, ephemeral=True, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning('Vault reply failed: %s', type(error).__name__)

    def _held(self, interaction) -> dict:
        held = {key: self.store.item_qty(interaction.guild_id, interaction.user.id, key) for key in TOOLS}
        return {key: qty for key, qty in held.items() if qty > 0}

    async def _show(self, interaction, view: dict, t, *, edit: bool, picker: bool = False, original: bool = False) -> None:
        held = self._held(interaction)
        text = render_main(view, t=t, tools_held=sum(held.values()))
        buttons = render_picker(view, held, t=t) if picker else render_buttons(view, t=t, tools_held=sum(held.values()))
        try:
            if original:
                await interaction.edit_original_response(content=text, view=buttons, allowed_mentions=NO_MENTIONS)
            elif edit:
                await interaction.response.edit_message(content=text, view=buttons, allowed_mentions=NO_MENTIONS)
            else:
                extra = {} if buttons is None else {'view': buttons}
                await interaction.response.send_message(text, ephemeral=True, allowed_mentions=NO_MENTIONS, **extra)
        except discord.HTTPException as error:
            logging.warning('Vault status could not be shown: %s', type(error).__name__)

    def _partner_ids(self, interaction, names) -> list[int]:
        names = [n for n in names if n is not None and n.strip()]
        if not names:
            return []
        parent_id, _ = self.bot.location(interaction.channel)
        rows = self.store.vault_partner_choices(interaction.guild_id, interaction.user.id, interaction.channel.id, parent_id)
        return [resolve(rows, name, 'character', ' available here or in the cast')['character_id'] for name in names]

    def partner_choices(self, interaction, current: str):
        parent_id, _ = self.bot.location(interaction.channel)
        rows = self.store.vault_partner_choices(interaction.guild_id, interaction.user.id, interaction.channel.id, parent_id)
        return suggest([r['name'] for r in rows], current)

    def _play(self, interaction, session_id: int, text: str, t):
        """(session view to show or None, text to send). A refusal carries no view, except an ended vault, whose status is refreshed."""
        gid, uid = interaction.guild_id, interaction.user.id
        try:
            result = self.store.submit_vault_guess(gid, uid, session_id, text, action_key=str(interaction.id), now=self.clock())
        except VaultStoreError as error:
            if error.code == codes.VAULT_NOT_FOUND:
                return None, t(NOT_YOURS)
            if error.code == codes.VAULT_NOT_ACTIVE:
                try:
                    return self.store.vault_session(gid, uid, session_id, now=self.clock()), t.error(error)
                except VaultStoreError:
                    return None, t.error(error)
            return None, t.error(error)
        return result['session'], guess_log(result, self.store.currency_name(gid), t=t)

    async def personal_command(self, interaction, protocol=None, partner=None, second_partner=None, guess=None) -> None:
        t = invoker_t(interaction)
        gid, uid = interaction.guild_id, interaction.user.id
        try:
            if interaction.channel is None:
                raise UserError(codes.USE_IN_SERVER)
            resuming = self.store.one('SELECT 1 FROM vault_sessions WHERE guild_id=? AND user_id=? AND expires_at>?', (gid, uid, self.clock())) is not None
            ids = [] if resuming else self._partner_ids(interaction, [partner, second_partner])
            parent_id, _ = self.bot.location(interaction.channel)
            view = self.store.open_personal_vault(gid, uid, interaction.channel.id, protocol=protocol or 'standard', partner_ids=ids, now=self.clock(), parent_id=parent_id)
        except UserError as error:
            return await self.say(interaction, t.error(error))
        line = None
        if guess is not None:
            played, line = self._play(interaction, view['id'], guess, t)
            view = played or view
        await self._show(interaction, view, t, edit=False)
        if line:
            await self.say(interaction, line)

    async def history_command(self, interaction) -> None:
        t = invoker_t(interaction)
        gid = interaction.guild_id
        rows = self.store.vault_history(gid, interaction.user.id, now=self.clock())
        if not rows:
            return await self.say(interaction, t(NO_VAULTS))
        currency, lines = self.store.currency_name(gid), []
        for r in rows:
            parts = [t('{date} · {protocol} · {status} · {used}/{budget} turns', date=r['local_date'], protocol=t.render(PROTOCOLS[r['protocol']]),
                       status=t.render(STATUSES[r['status']]), used=r['turns_used'], budget=r['turn_budget'])]
            if r['secret']:
                parts.append(t('Secret: {code}', code=r['secret']))
            if r['reward']:
                parts.append(t('Reward: {amount:,} {currency} (reversed)', amount=r['reward'], currency=currency) if r['reward_reversed']
                             else t('Reward: {amount:,} {currency}', amount=r['reward'], currency=currency))
            lines.append(' · '.join(parts))
        await self.say(interaction, '\n'.join(lines)[:1900])

    async def press(self, interaction, action: str, session_id: int) -> None:
        t = invoker_t(interaction)
        gid, uid = interaction.guild_id, interaction.user.id
        if gid is None:
            return await self.say(interaction, t.error(UserError(codes.USE_IN_SERVER)))
        try:
            view = self.store.vault_session(gid, uid, session_id, now=self.clock())
        except VaultStoreError:
            return await self.say(interaction, t(NOT_YOURS))
        if action == 'guess':
            try:
                await interaction.response.send_modal(VaultGuessModal(session_id, t=t))
            except discord.HTTPException as error:
                logging.warning('Vault could not open the guess modal: %s', type(error).__name__)
            return
        if action == 'back':
            return await self._show(interaction, view, t, edit=True)
        if action == 'tools' or action.startswith('use_'):
            return await self._press_tool(interaction, action, view, t)
        if action == 'meeting' or (action.startswith('ask_') and action[4:].isascii() and action[4:].isdigit()):
            return await self._press_advice(interaction, action, view, t)
        if action != 'clue':
            return await self.say(interaction, t.error(VaultStoreError(codes.VAULT_NOT_FOUND)))
        hidden = next((c['position'] for c in view['clues'] if not c['revealed']), None)
        refused = voiced = deferred = None
        if view['status'] == 'active' and hidden is not None and self.store.vault_settings(gid)['clue_voice'] == 'llm':
            deferred = await self._defer(interaction)   # acknowledged before the reveal commits; a refused defer still reveals, just without the voice
        if view['status'] != 'active':
            refused = VaultStoreError(codes.VAULT_NOT_ACTIVE)
        elif hidden is not None:
            try:
                self.store.reveal_vault_clue(gid, uid, session_id, hidden, now=self.clock())
            except VaultStoreError as error:
                refused = error
        view = self.store.vault_session(gid, uid, session_id, now=self.clock())
        if deferred and refused is None:
            entry = next((c for c in view['clues'] if c['position'] == hidden and c.get('constraint') is not None), None)
            if entry is not None:
                try:
                    voiced = await self._voice_clue(interaction, entry)
                except Exception as error:
                    logging.error('Vault clue voice failed: %s', type(error).__name__)
        await self._show(interaction, view, t, edit=True, original=bool(deferred))
        if voiced:
            await self.say(interaction, t('{name}: {fact}', name=entry['name'] or t('a deleted character'), fact=voiced))
        if refused is not None:
            await self.say(interaction, t.error(refused))

    @staticmethod
    async def _defer(interaction) -> bool:
        """Acknowledge a press that will wait on the model; False (nothing may be spent) when Discord did not take it."""
        try:
            await interaction.response.defer()
        except discord.HTTPException as error:
            logging.warning('Vault could not acknowledge the press: %s', type(error).__name__)
        return interaction.response.is_done()

    async def _word(self, interaction, character_id: int, user_text: str, *, record: bool = False) -> tuple[str | None, list]:
        """One structured model call that only words public facts: (wording or None, captured usage records). None on any failure or empty
        answer (the caller keeps its template). Call inside a backend pin. The capture has no guild, so the gateway keeps the spend total but
        leaves the model_usage row to the caller (the store inserts it with the advice, or ``record`` lets the gateway write it)."""
        character = self.store.character_by_id(character_id)
        if character is None or character['guild_id'] != interaction.guild_id:
            return None, []
        system = ADVICE_SYSTEM.format(name=character['name'], persona=game_chooser.persona(character))
        request = SimpleNamespace(messages=[TurnMessage('system', system), TurnMessage('user', user_text)])
        guild = interaction.guild_id if record else None
        records = []
        try:
            with capture_usage(guild, interaction.channel_id, 'game') as records, log_scope(None), log_purpose('game'):
                answer = await asyncio.wait_for(self.bot.models.structured_compiled(game_chooser.ROLE, request, 'vault_advice', LINE_SCHEMA), ADVICE_SECONDS)
        except Exception as error:
            logging.info('Vault wording failed, using the template: %s', type(error).__name__)
            return None, records
        line = next((v for v in answer.values() if isinstance(v, str)), None) if isinstance(answer, dict) else None
        return game_chooser.clean_line(line)[:MAX_WORDING] or None, records

    async def _voice_clue(self, interaction, entry: dict) -> str | None:
        template = clue_text(entry['constraint'])
        gid = interaction.guild_id
        if self.store.game_talk_calls_today(gid, self.clock()) + self.talk_reserved.get(gid, 0) + 1 > self.store.game_talk_daily_limit(gid):
            return None
        self.talk_reserved[gid] = self.talk_reserved.get(gid, 0) + 1  # reserved before the call; released once its usage row exists or it never ran
        try:
            with self.bot.backend.pin():
                wording, _ = await self._word(interaction, entry['character_id'], CLUE_USER.format(text=template), record=True)
        finally:
            self.talk_reserved[gid] -= 1
            if self.talk_reserved[gid] <= 0:
                del self.talk_reserved[gid]
        return wording if clue_wording_ok(wording, template) else None

    def _advice_facts(self, view: dict, character_id: int):
        style = self.store.resolved_play_style(view['guild_id'], character_id)
        return build_advice_facts(view['constraints'], view['protocol'], view['length'], risk=style.risk, aggression=style.aggression,
                                  unpredictability=style.unpredictability, seed=f"vault-advice/{view['id']}/{character_id}/{view['turns_used']}")

    def _precheck_advice(self, view: dict, count: int) -> None:
        """The store's cap and talk checks, ahead of any model call, so a refused press asks nothing."""
        gid = view['guild_id']
        cap = self.store.vault_settings(gid)['advice_cap']
        if len(view['advice']) + count > cap:
            raise VaultStoreError(codes.VAULT_ADVICE_LIMIT, limit=cap)
        limit = self.store.game_talk_daily_limit(gid)
        if self.store.game_talk_calls_today(gid, self.clock()) + count > limit:
            raise VaultStoreError(codes.VAULT_ADVICE_TALK_LIMIT, limit=limit)

    async def _press_advice(self, interaction, action: str, view: dict, t) -> None:
        gid, uid, sid, iid = interaction.guild_id, interaction.user.id, view['id'], str(interaction.id)
        if view['status'] != 'active':
            await self._show(interaction, view, t, edit=True)
            return await self.say(interaction, t.error(VaultStoreError(codes.VAULT_NOT_ACTIVE)))
        partner_ids = [p['character_id'] for p in view['partners']]
        asked = partner_ids if action == 'meeting' else [int(action[4:])]
        if action == 'meeting' and len(partner_ids) < 2 or any(c not in partner_ids for c in asked):
            return await self.say(interaction, t.error(VaultStoreError(codes.VAULT_ADVICE_NOT_PARTNER)))
        replayed = self.store.vault_advice_replayed(gid, sid, iid, meeting=action == 'meeting')
        if not replayed:
            try:
                self._precheck_advice(view, len(asked))
            except VaultStoreError as error:
                return await self.say(interaction, t.error(error))
            if not await self._defer(interaction):
                return
        try:
            texts, usage = {}, {}
            if not replayed:
                facts = {c: self._advice_facts(view, c) for c in asked}
                prompts = {c: ADVICE_USER.format(guess=facts[c].guess, remaining=ADVICE_COUNT.format(remaining=facts[c].remaining)
                                                 if len(str(facts[c].remaining)) < len(facts[c].guess) else '') for c in asked}
                with self.bot.backend.pin():
                    words = await asyncio.gather(*(self._word(interaction, c, prompts[c]) for c in asked))
                texts = {c: w[0] for c, w in zip(asked, words)}
                usage = {c: w[1] for c, w in zip(asked, words)}
            try:
                if action == 'meeting':
                    entries = self.store.record_vault_meeting(gid, uid, sid, texts=texts, usage=usage, action_key=iid, now=self.clock())
                else:
                    entries = [self.store.record_vault_advice(gid, uid, sid, asked[0], text=texts.get(asked[0]), usage=usage.get(asked[0]), action_key=iid, now=self.clock())]
            except VaultStoreError as error:
                if error.code == codes.VAULT_NOT_ACTIVE:
                    try:
                        await self._show(interaction, self.store.vault_session(gid, uid, sid, now=self.clock()), t, edit=True, original=not replayed)
                    except VaultStoreError:
                        pass
                return await self.say(interaction, t.error(error))
            await self._show(interaction, self.store.vault_session(gid, uid, sid, now=self.clock()), t, edit=True, original=not replayed)
            await self.say(interaction, '\n'.join(advice_line(entry, t) for entry in entries))
        except Exception as error:
            logging.error('Vault advice failed: %s', type(error).__name__)
            await self.say(interaction, t(SOMETHING_WRONG))

    async def _press_tool(self, interaction, action: str, view: dict, t) -> None:
        key = action[4:]
        if action != 'tools' and key not in TOOLS:
            return await self.say(interaction, t.error(VaultStoreError(codes.VAULT_TOOL_UNKNOWN, tool=key)))
        if view['status'] != 'active':
            await self._show(interaction, view, t, edit=True)
            return await self.say(interaction, t.error(VaultStoreError(codes.VAULT_NOT_ACTIVE)))
        if action == 'tools':
            return await self._show(interaction, view, t, edit=True, picker=True)
        try:
            await interaction.response.send_modal(VaultToolModal(view['id'], key, t=t))
        except discord.HTTPException as error:
            logging.warning('Vault could not open the tool modal: %s', type(error).__name__)

    @staticmethod
    def _tool_params(view: dict, tool_key: str, values: dict) -> dict:
        params = {}
        for name in TOOLS[tool_key].needs:
            text = str(values.get(name, '')).strip()
            if name == 'position':
                params[name] = int(text) if text.isascii() and text.isdigit() else text
            elif name == 'guess':
                found = next((g['guess'] for g in view['guesses'] if text.isascii() and text.isdigit() and str(g['number']) == text), None)
                params[name] = found or text
            else:
                params[name] = text
        return params

    async def submit_tool(self, interaction, session_id: int, tool_key: str, values: dict) -> None:
        t = invoker_t(interaction)
        gid, uid = interaction.guild_id, interaction.user.id
        if gid is None:
            return await self.say(interaction, t.error(UserError(codes.USE_IN_SERVER)))
        try:
            view = self.store.vault_session(gid, uid, session_id, now=self.clock())
        except VaultStoreError:
            return await self.say(interaction, t(NOT_YOURS))
        if tool_key not in TOOLS:
            return await self.say(interaction, t.error(VaultStoreError(codes.VAULT_TOOL_UNKNOWN, tool=str(tool_key))))
        try:
            result = self.store.use_vault_tool(gid, uid, session_id, tool_key, self._tool_params(view, tool_key, values), action_key=str(interaction.id), now=self.clock())
        except VaultStoreError as error:
            if error.code == codes.VAULT_NOT_ACTIVE:
                try:
                    await self._show(interaction, self.store.vault_session(gid, uid, session_id, now=self.clock()), t, edit=True)
                except VaultStoreError:
                    pass
            text = t.error(error)
            reason = TOOL_REASONS.get(error.params.get('reason')) if error.code == codes.VAULT_TOOL_BAD_PARAMS else None
            return await self.say(interaction, f'{text} {t.render(reason)}' if reason else text)
        session = result['session']
        lines = [tool_line(result, t)]
        if session['status'] == 'failed':
            lines.append(t('Out of turns. The vault stays closed today.'))
        await self._show(interaction, session, t, edit=True)
        await self.say(interaction, '\n'.join(lines))

    async def submit_guess(self, interaction, session_id: int, text: str, *, edit: bool = True) -> None:
        t = invoker_t(interaction)
        if interaction.guild_id is None:
            return await self.say(interaction, t.error(UserError(codes.USE_IN_SERVER)))
        view, line = self._play(interaction, session_id, text, t)
        if view is None:
            return await self.say(interaction, line)
        await self._show(interaction, view, t, edit=edit)
        await self.say(interaction, line)


def register_vault_commands(bot, ctx: SimpleNamespace) -> None:
    require_guild, safe_choices = ctx.require_guild, ctx.safe_choices
    bot.vault = play = VaultPlay(bot)
    vault = app_commands.Group(name='vault', description=C_('Experimental: open your personal vault by guessing its code'))
    partner_choices = safe_choices(play.partner_choices)

    @vault.command(name='personal', description=C_("Start or continue today's personal vault"))
    @app_commands.describe(protocol=C_('Protocol for a new vault (default: Standard)'), partner=C_('A favorite or cast character who helps with a new vault'),
                           second_partner=C_('A second partner for a new vault'), guess=C_('Guess the code: 4 different digits'))
    @app_commands.choices(protocol=[app_commands.Choice(name=C_('Standard'), value='standard'), app_commands.Choice(name=C_('Total Match'), value='total_match'),
                                    app_commands.Choice(name=C_('Position-Only'), value='position_only'), app_commands.Choice(name=C_('Open'), value='open')])
    @app_commands.autocomplete(partner=partner_choices, second_partner=partner_choices)
    async def vault_personal(interaction: discord.Interaction, protocol: str | None = None, partner: str | None = None,
                             second_partner: str | None = None, guess: str | None = None):
        require_guild(interaction)
        await play.personal_command(interaction, protocol, partner, second_partner, guess)

    @vault.command(name='history', description=C_('Show your recent vaults'))
    async def vault_history(interaction: discord.Interaction):
        require_guild(interaction)
        await play.history_command(interaction)

    bot.tree.add_command(vault)
