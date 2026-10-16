"""``/vault operation``: the weekly co-op vault operation on Discord (FEAT-66 phase B, D43, Experimental).

One shared status message per operation lives in the operation channel (channel language, resolved at each render) and is edited in place,
debounced. Everything a member sees alone (feedback, clue lines, refusals, pickers) is an ephemeral reply in their language. The store
(``vault_operation_store``) is the only source of truth and raises codes; this layer takes input and builds the words. The secret is shown
only for stages that have ended and never reaches a prompt.
"""
from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace

import discord

from .errors import error_detail
from .vault import TOOLS, build_advice_facts
from .vault_discord import (NO_MENTIONS, PREFIX, PROTOCOLS, SOMETHING_WRONG, STATUSES, TOOL_FIELDS, TOOL_REASONS, ADVICE_COUNT, ADVICE_USER, VaultPlay,
                            advice_line, clue_text, feedback_text, tool_line, tool_name)
from .vault_store import MAX_TOOLS, VaultStoreError
from .vault_support_discord import MISSION_ACTIONS, SupportMissions
from llmcord_core import i18n
from llmcord_core.i18n import UserError, codes
from llmcord_core.i18n.discord_locale import C_, channel_locale, invoker_t

STAGES = 3
ATTEMPTS = 2
MAX_TEXT = 1990


def mention(user_id: int) -> str:
    return f'<@{user_id}>'


def render_status(view: dict, *, t=None) -> str:
    """The shared status message. A stage's secret appears only once that stage has ended (the store withholds it until then).
    Over the limit, the lists shrink to counts first, then the oldest guesses fold into one line; the header, secrets and previous codes always stay."""
    t = t or i18n.english()
    stage = view['stage'] or next((x for x in reversed(view['stages']) if x['status'] != 'locked'), view['stages'][0])
    size = lambda lines: len('\n'.join(lines))  # noqa: E731
    lines = _status_lines(view, t, False, 0)
    if size(lines) > MAX_TEXT:
        lines = _status_lines(view, t, True, 0)
    hidden = 0
    while size(lines) > MAX_TEXT and hidden < len(stage['guesses']):
        hidden = min(len(stage['guesses']), hidden + max(1, (size(lines) - MAX_TEXT) // 40))
        lines = _status_lines(view, t, True, hidden)
    secrets = sum(1 for x in view['stages'] if x['secret']) + sum(1 for x in view['stages'] if x['previous_secret'])
    while size(lines) > MAX_TEXT and len(lines) > 5 + secrets:  # never in practice; whole lines only, the secrets stay
        del lines[len(lines) - secrets - 1]
    return '\n'.join(lines)


def _status_lines(view: dict, t, compact: bool, hidden: int) -> list:
    stage = view['stage'] or next((s for s in reversed(view['stages']) if s['status'] != 'locked'), view['stages'][0])
    deleted = t('a deleted character')
    lines = [t('Weekly operation ({week})', week=view['week_key']), t('Status: {status}', status=t.render(STATUSES[view['status']])),
             t('Stage {n} of {total}: {protocol}', n=stage['number'], total=STAGES, protocol=t.render(PROTOCOLS[stage['protocol']])),
             t('Attempt {n} of {total}', n=stage['retry'] + 1, total=ATTEMPTS),
             t('Shared turns left: {left} of {budget}', left=stage['turns_left'], budget=stage['turn_budget'])]
    if stage['open_digit']:
        lines.append(t('Open digit: {digit}', digit=stage['open_digit']))
    lines.append(t('Team tools used: {n} of {limit}', n=stage['tools_used'], limit=MAX_TOOLS))
    names = ', '.join(n['name'] or deleted for n in view['npcs'])
    lines.append(t('Characters: {names}', names=names))
    people = view['participants']
    if people:
        mentions = str(len(people)) if compact else ', '.join(mention(p['user_id']) for p in people)
        lines.append(t('Participants: {mentions}', mentions=mentions))
    advisors = [f"{mention(p['user_id'])}: {p['advisor']['name'] or deleted}" for p in people if p['advisor']]
    if advisors:
        entries = str(len(advisors)) if compact else '; '.join(advisors)
        lines.append(t('Advisors: {entries}', entries=entries))
    if stage['clues']:
        lines.append(t('Clues revealed: {n} of {total}', n=sum(1 for c in stage['clues'] if c['revealed']), total=len(stage['clues'])))
        for clue in stage['clues']:
            if clue['revealed'] and 'constraint' in clue:
                lines.append(t('Clue from {name}: {text}', name=clue['name'] or deleted, text=clue_text(clue['constraint'], t=t)))
    if stage['tools']:
        lines.append(t('Tool results:'))
        lines += [tool_line(entry, t) for entry in stage['tools']]
    for cross in stage['cross']:
        lines.append(cross_line(cross, t))
    if stage['guesses']:
        lines.append(t('Guesses:'))
        if hidden:
            lines.append(t.n('… {count} earlier guess', '… {count} earlier guesses', hidden))
        lines += [t('{n}. {mention}: {guess} — {feedback}', n=g['number'], mention=mention(g['user_id']), guess=g['guess'],
                     feedback=feedback_text(stage['protocol'], g['feedback'], t=t)) for g in stage['guesses'][hidden:]]
    lines += [t("Previous attempt's code for stage {n}: {code}", n=s['number'], code=s['previous_secret']) for s in view['stages'] if s['previous_secret']]
    lines += [t('Secret of stage {n}: {code}', n=s['number'], code=s['secret']) for s in view['stages'] if s['secret']]
    return lines


def cross_line(cross: dict, t) -> str:
    a, b = cross['numbers']
    digits = ', '.join(cross['common'])
    return t('Cross-check of guesses {a} and {b}: the code has exactly {count} of the digits {digits}.', a=a, b=b, count=cross['count'], digits=digits)


def render_buttons(view: dict, *, t=None):
    """The status buttons; None once the operation is no longer active."""
    t = t or i18n.english()
    if view['status'] != 'active' or view['stage'] is None:
        return None
    oid, stage = view['id'], view['stage']
    buttons = discord.ui.View(timeout=None)
    buttons.add_item(VaultOperationButton('guess', oid, t.p('button.vault', 'Guess'), discord.ButtonStyle.primary))
    if any(not c['revealed'] for c in stage['clues']):
        buttons.add_item(VaultOperationButton('clue', oid, t.p('button.vault', 'Reveal clue')))
    buttons.add_item(VaultOperationButton('tools', oid, t.p('button.vault', 'Tools')))
    buttons.add_item(VaultOperationButton('cross', oid, t.p('button.vault', 'Cross-check')))
    buttons.add_item(VaultOperationButton('mission', oid, t.p('button.vault', 'Support mission')))
    for npc in view['npcs']:
        if npc['name']:
            buttons.add_item(VaultOperationButton(f"ask_{npc['character_id']}", oid, t.p('button.vault', 'Ask {name}', name=npc['name'])[:80]))
    buttons.add_item(VaultOperationButton('ask_advisor', oid, t.p('button.vault', 'Ask my advisor')))
    buttons.add_item(VaultOperationButton('advisor', oid, t.p('button.vault', 'Advisor')))
    buttons.add_item(VaultOperationButton('join', oid, t.p('button.vault', 'Join')))
    buttons.add_item(VaultOperationButton('refresh', oid, t.p('button.vault', 'Refresh')))
    return buttons


def render_picker(operation_id: int, held: dict, *, t=None):
    t = t or i18n.english()
    buttons = discord.ui.View(timeout=None)
    for key, qty in held.items():
        label = t.n('{name} ×{qty} ({count} turn)', '{name} ×{qty} ({count} turns)', TOOLS[key].turn_cost, name=tool_name(key, t), qty=qty)
        buttons.add_item(VaultOperationButton(f'use_{key}', operation_id, label[:80]))
    buttons.add_item(VaultOperationButton('back', operation_id, t.p('button.vault', 'Back')))
    return buttons


def refused_note(result: dict, t) -> str:
    return '\n' + t('Your reward could not be paid because your balance is at its limit.') if result.get('reward_refused') else ''


class VaultOperationButton(discord.ui.DynamicItem[discord.ui.Button], template=PREFIX + r':op_(?P<action>[a-z0-9_]+):(?P<operation>\d+)'):
    """One button; its custom_id carries the action and the operation, never a member or the code."""

    def __init__(self, action: str, operation_id: int, label: str = '', style=discord.ButtonStyle.secondary, disabled: bool = False):
        super().__init__(discord.ui.Button(label=label, style=style, disabled=disabled, custom_id=f'{PREFIX}:op_{action}:{operation_id}'))
        self.action, self.operation_id = action, operation_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match['action'], int(match['operation']), item.label or '', item.style, item.disabled)

    async def callback(self, interaction: discord.Interaction):
        await interaction.client.vault_operation.press(interaction, self.action, self.operation_id)


class _Modal(discord.ui.Modal):
    async def on_error(self, interaction: discord.Interaction, error: Exception):
        logging.error('Vault operation modal failed: %s', error_detail(error))
        await interaction.client.vault_operation.say(interaction, invoker_t(interaction)(SOMETHING_WRONG))


class OperationGuessModal(_Modal):
    def __init__(self, operation_id: int, length: int = 4, *, t=None):
        t = t or i18n.english()
        super().__init__(title=t.p('modal.vault', 'Guess the code')[:45], timeout=600)
        self.operation_id = operation_id
        self.guess = discord.ui.TextInput(label=t.p('modal.vault', 'Your guess')[:45], placeholder=t.p('modal.vault', '{length} different digits', length=length)[:100],
                                          required=True, max_length=16)
        self.add_item(self.guess)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.client.vault_operation.submit_guess(interaction, self.operation_id, str(self.guess.value or ''))


class OperationCrossModal(_Modal):
    def __init__(self, operation_id: int, *, t=None):
        t = t or i18n.english()
        super().__init__(title=t.p('modal.vault', 'Cross-check two guesses')[:45], timeout=600)
        self.operation_id = operation_id
        self.first = discord.ui.TextInput(label=t.p('modal.vault', 'First guess (number)')[:45], required=True, max_length=4)
        self.second = discord.ui.TextInput(label=t.p('modal.vault', 'Second guess (number)')[:45], required=True, max_length=4)
        self.add_item(self.first)
        self.add_item(self.second)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.client.vault_operation.submit_cross(interaction, self.operation_id, str(self.first.value or ''), str(self.second.value or ''))


class OperationToolModal(_Modal):
    def __init__(self, operation_id: int, tool_key: str, *, t=None):
        t = t or i18n.english()
        super().__init__(title=tool_name(tool_key, t)[:45], timeout=600)
        self.operation_id, self.tool_key = operation_id, tool_key
        self.inputs = {name: discord.ui.TextInput(label=t.render(TOOL_FIELDS[name])[:45], required=True, max_length=16) for name in TOOLS[tool_key].needs}
        for item in self.inputs.values():
            self.add_item(item)

    async def on_submit(self, interaction: discord.Interaction):
        values = {name: str(item.value or '') for name, item in self.inputs.items()}
        await interaction.client.vault_operation.submit_tool(interaction, self.operation_id, self.tool_key, values)


class OperationMissionModal(_Modal):
    def __init__(self, operation_id: int, mission_type: str, *, t=None):
        t = t or i18n.english()
        super().__init__(title=t.p('modal.vault', 'Support mission')[:45], timeout=600)
        self.operation_id, self.mission_type = operation_id, mission_type
        if mission_type == 'forensics':
            self.inputs = {'strikes': discord.ui.TextInput(label=t.p('modal.vault', 'Strikes of the hidden line')[:45], required=True, max_length=3),
                           'balls': discord.ui.TextInput(label=t.p('modal.vault', 'Balls of the hidden line')[:45], required=True, max_length=3)}
        else:
            self.inputs = {'code': discord.ui.TextInput(label=t.p('modal.vault', 'Your code')[:45], placeholder=t.p('modal.vault', '{length} different digits', length=3)[:100],
                                                        required=True, max_length=16)}
        for item in self.inputs.values():
            self.add_item(item)

    async def on_submit(self, interaction: discord.Interaction):
        values = {name: str(item.value or '') for name, item in self.inputs.items()}
        await interaction.client.vault_operation.submit_mission(interaction, self.operation_id, self.mission_type, values)


class AdvisorSelect(discord.ui.Select):
    def __init__(self, operation_id: int, options, placeholder: str):
        super().__init__(placeholder=placeholder[:150], min_values=0, max_values=1, options=options)
        self.operation_id = operation_id

    async def callback(self, interaction: discord.Interaction):
        chosen = int(self.values[0]) if self.values else None
        await interaction.client.vault_operation.select_advisor(interaction, self.operation_id, chosen)


class VaultOperationPlay(SupportMissions):
    # The private-reply, defer, model-wording and tool-parameter helpers are the personal vault's; they only need .store and .bot.
    say = VaultPlay.say
    _defer = staticmethod(VaultPlay._defer)
    _word = VaultPlay._word
    _tool_params = staticmethod(VaultPlay._tool_params)

    def __init__(self, bot, clock=time.time, sleep=asyncio.sleep, debounce_seconds: float = 3.0):
        self.bot, self.clock, self.sleep, self.debounce_seconds = bot, clock, sleep, debounce_seconds
        self._status: dict[int, SimpleNamespace] = {}  # operation id -> guild_id, channel_id, message_id, actor (memory only)
        self._timers: dict[int, asyncio.Task] = {}
        self._dirty: set[int] = set()
        self._post_lock = asyncio.Lock()

    @property
    def store(self):
        return self.bot.store

    # ---- status message

    def _channel_t(self, guild_id, channel=None):
        guild = getattr(channel, 'guild', None) or self.bot.get_guild(guild_id) or SimpleNamespace(id=guild_id)
        try:
            language = self.store.server_language(guild_id) or ''
        except Exception as error:
            logging.warning('cannot read the server language of guild %s: %s', guild_id, type(error).__name__)
            language = ''
        return i18n.Translator(channel_locale(language, guild))

    def mark(self, operation_id: int, actor: int) -> None:
        """The status changed: one debounced edit with everything that changes meanwhile."""
        ref = self._status.get(operation_id)
        if ref is None:
            return
        ref.actor = actor
        self._dirty.add(operation_id)
        timer = self._timers.get(operation_id)
        if timer is None or timer.done():
            self._timers[operation_id] = asyncio.get_running_loop().create_task(self._timer(operation_id))

    async def _timer(self, operation_id: int) -> None:
        await self.sleep(self.debounce_seconds)
        self._timers.pop(operation_id, None)
        try:
            await self._edit(operation_id)
        except Exception as error:
            logging.error('Vault operation status edit failed: %s', type(error).__name__)

    async def _edit(self, operation_id: int) -> None:
        self._dirty.discard(operation_id)
        ref = self._status.get(operation_id)
        channel = self.bot.get_channel(ref.channel_id) if ref is not None else None
        if channel is None:
            return
        try:
            view = self.store.vault_operation(ref.guild_id, ref.actor, operation_id, now=self.clock())
        except VaultStoreError:
            return
        t = self._channel_t(ref.guild_id, channel)
        try:
            await channel.get_partial_message(ref.message_id).edit(content=render_status(view, t=t), view=render_buttons(view, t=t), allowed_mentions=NO_MENTIONS)
        except discord.NotFound:
            self._status.pop(operation_id, None)
        except discord.HTTPException as error:
            logging.warning('Vault operation status could not be edited: %s', type(error).__name__)

    async def flush(self, operation_id: int | None = None) -> None:
        """Sends every pending status edit now (or the one of ``operation_id``) and cancels its timer."""
        for oid in [o for o in list(self._dirty) if operation_id in (None, o)]:
            timer = self._timers.pop(oid, None)
            if timer is not None and timer is not asyncio.current_task():
                timer.cancel()
            await self._edit(oid)

    async def close(self) -> None:
        timers = list(self._timers.values())
        self._timers.clear()
        for timer in timers:
            timer.cancel()
        await asyncio.gather(*timers, return_exceptions=True)

    async def _post(self, interaction, view: dict, actor: int) -> None:
        async with self._post_lock:
            if view['id'] in self._status:
                return self.mark(view['id'], actor)
            channel = interaction.channel
            t = self._channel_t(view['guild_id'], channel)
            buttons = render_buttons(view, t=t)
            try:
                message = await channel.send(render_status(view, t=t), allowed_mentions=NO_MENTIONS, **({} if buttons is None else {'view': buttons}))
            except discord.HTTPException as error:
                return logging.warning('Vault operation status could not be posted: %s', type(error).__name__)
            self._status[view['id']] = SimpleNamespace(guild_id=view['guild_id'], channel_id=channel.id, message_id=message.id, actor=actor)

    async def _announce(self, operation: dict, event: str | None, channel) -> None:
        if event is None:
            return
        t = self._channel_t(operation['guild_id'], channel)
        stage = operation['stage']['number'] if operation['stage'] else STAGES
        text = {'stage_solved': lambda: t('Stage {stage} cleared! Stage {next} is open.', stage=stage - 1, next=stage),
                'stage_retry': lambda: t('Stage {stage} ran out of turns. The team gets one more try with a new code.', stage=stage),
                'operation_failed': lambda: t('The operation failed. The vault stays closed this week.'),
                'operation_solved': lambda: t('The operation is solved! All three stages are cleared.')}.get(event)
        channel = self.bot.get_channel(operation['channel_id']) or channel
        if text is None or channel is None:
            return
        try:
            await channel.send(text(), allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning('Vault operation announcement failed: %s', type(error).__name__)

    def _adopt(self, interaction, view: dict, uid: int) -> None:
        """After a restart the status message is the one a button sits on (never an ephemeral picker): keep editing it instead of posting another."""
        message = getattr(interaction, 'message', None)
        if view['id'] in self._status or message is None or getattr(getattr(message, 'flags', None), 'ephemeral', False):
            return
        if getattr(getattr(message, 'channel', None), 'id', None) == view['channel_id']:
            self._status[view['id']] = SimpleNamespace(guild_id=view['guild_id'], channel_id=view['channel_id'], message_id=message.id, actor=uid)

    async def _done(self, interaction, result: dict, uid: int) -> None:
        operation = result['operation']
        self._adopt(interaction, operation, uid)
        self.mark(operation['id'], uid)
        if not result['replayed']:
            await self._announce(operation, result['event'], interaction.channel)

    # ---- command and presses

    def _open(self, interaction, uid: int) -> dict:
        gid, channel = interaction.guild_id, interaction.channel
        if channel is None:
            raise UserError(codes.USE_IN_SERVER)
        if isinstance(channel, discord.Thread):
            if not self.store.vault_settings(gid)['enabled']:
                raise VaultStoreError(codes.VAULT_DISABLED)
            if not self.store.vault_operation_settings(gid)['enabled']:
                raise VaultStoreError(codes.VAULT_OPERATION_DISABLED)
            raise VaultStoreError(codes.VAULT_OPERATION_CHANNEL)
        parent_id, _ = self.bot.location(channel)
        return self.store.open_vault_operation(gid, uid, channel.id, now=self.clock(), parent_id=parent_id)

    def _check_channel(self, gid: int, here=None) -> None:
        """The configured operation channel must be a plain text channel of this guild (the store only checks the id is a number)."""
        settings = self.store.vault_operation_settings(gid)
        if not settings['enabled'] or settings['channel_id'] is None or not self.store.vault_settings(gid)['enabled']:
            return
        channel = here if getattr(here, 'id', None) == settings['channel_id'] else self.bot.get_channel(settings['channel_id'])  # the channel the press came from needs no cache
        guild = getattr(channel, 'guild', None)
        owner = guild.id if guild is not None else getattr(channel, 'guild_id', None)
        other = isinstance(channel, (discord.abc.GuildChannel, discord.Thread)) and not isinstance(channel, discord.TextChannel)
        if channel is None or other or (owner is not None and owner != gid):
            raise VaultStoreError(codes.VAULT_OPERATION_CHANNEL_INVALID)

    async def _enter(self, interaction, t, channel_id=None) -> None:
        uid = interaction.user.id
        try:
            if not isinstance(interaction.channel, discord.Thread):
                self._check_channel(interaction.guild_id, interaction.channel)
            view = self._open(interaction, uid) if channel_id is None else self._open_for(interaction, uid, channel_id)
        except UserError as error:
            return await self.say(interaction, t.error(error))
        await self.say(interaction, t('You joined the operation.') if view['joined'] else t('You are already in the operation.'))
        self._adopt(interaction, view, uid)
        if view['id'] not in self._status:
            await self._post(interaction, view, uid)
        elif view['joined']:
            self.mark(view['id'], uid)

    def _open_for(self, interaction, uid: int, operation_id: int) -> dict:
        gid = interaction.guild_id
        row = self.store.one('SELECT channel_id FROM vault_operations WHERE id=? AND guild_id=?', (operation_id, gid))
        if row is None:
            raise VaultStoreError(codes.VAULT_NOT_FOUND)
        if self.store.current_vault_operation_id(gid, row['channel_id'], now=self.clock()) != operation_id:
            raise VaultStoreError(codes.VAULT_NOT_ACTIVE)
        parent_id, _ = self.bot.location(interaction.channel)
        return self.store.open_vault_operation(gid, uid, row['channel_id'], now=self.clock(), parent_id=parent_id)

    async def open_command(self, interaction) -> None:
        await self._enter(interaction, invoker_t(interaction))

    async def press(self, interaction, action: str, operation_id: int) -> None:
        t = invoker_t(interaction)
        gid, uid = interaction.guild_id, interaction.user.id
        if gid is None:
            return await self.say(interaction, t.error(UserError(codes.USE_IN_SERVER)))
        if action == 'join':
            return await self._enter(interaction, t, operation_id)
        try:
            view = self.store.vault_operation(gid, uid, operation_id, now=self.clock())
        except VaultStoreError as error:
            return await self.say(interaction, t.error(error))
        if action == 'refresh':
            return await self._refresh(interaction, view, uid)
        if action == 'back':
            return await self._back(interaction, view, t)
        if action in ('guess', 'cross', 'tools', 'clue', 'advisor', *MISSION_ACTIONS) or action.startswith(('use_', 'ask_')):
            if view['status'] != 'active' or view['stage'] is None:
                return await self.say(interaction, t.error(VaultStoreError(codes.VAULT_NOT_ACTIVE)))
        else:
            return await self.say(interaction, t.error(VaultStoreError(codes.VAULT_NOT_FOUND)))
        if action in MISSION_ACTIONS:
            return await self._mission_press(interaction, view, action, t)
        if action == 'guess':
            return await self._modal(interaction, OperationGuessModal(operation_id, view['stage']['length'], t=t))
        if action == 'cross':
            return await self._modal(interaction, OperationCrossModal(operation_id, t=t))
        if action == 'tools':
            return await self._picker(interaction, operation_id, t)
        if action.startswith('use_'):
            key = action[4:]
            if key not in TOOLS:
                return await self.say(interaction, t.error(VaultStoreError(codes.VAULT_TOOL_UNKNOWN, tool=key)))
            return await self._modal(interaction, OperationToolModal(operation_id, key, t=t))
        if action == 'clue':
            return await self._clue(interaction, view, t)
        if action == 'advisor':
            return await self._advisor_picker(interaction, view, t)
        if action == 'ask_advisor' or (action[4:].isascii() and action[4:].isdigit()):
            return await self._ask(interaction, view, action, t)
        await self.say(interaction, t.error(VaultStoreError(codes.VAULT_NOT_FOUND)))

    @staticmethod
    async def _modal(interaction, modal) -> None:
        try:
            await interaction.response.send_modal(modal)
        except discord.HTTPException as error:
            logging.warning('Vault could not open the operation modal: %s', type(error).__name__)

    async def _refresh(self, interaction, view: dict, uid: int) -> None:
        self._adopt(interaction, view, uid)
        if not await self._defer(interaction):
            return
        if view['id'] in self._status:
            self._status[view['id']].actor = uid
            self._dirty.add(view['id'])
            await self.flush(view['id'])

    async def _back(self, interaction, view: dict, t) -> None:
        try:
            await interaction.response.edit_message(content=render_status(view, t=t), view=None, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning('Vault operation picker could not be closed: %s', type(error).__name__)

    async def _picker(self, interaction, operation_id: int, t) -> None:
        gid, uid = interaction.guild_id, interaction.user.id
        held = {key: self.store.item_qty(gid, uid, key) for key in TOOLS}
        held = {key: qty for key, qty in held.items() if qty > 0}
        if not held:
            return await self.say(interaction, t.error(VaultStoreError(codes.VAULT_TOOL_NOT_HELD)))
        try:
            await interaction.response.send_message(t('Choose a tool.'), ephemeral=True, view=render_picker(operation_id, held, t=t), allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning('Vault operation picker failed: %s', type(error).__name__)

    async def _clue(self, interaction, view: dict, t) -> None:
        gid, uid = interaction.guild_id, interaction.user.id
        hidden = next((c['position'] for c in view['stage']['clues'] if not c['revealed']), None)
        try:
            if hidden is None:
                raise VaultStoreError(codes.VAULT_CLUE_UNKNOWN)
            entry = self.store.reveal_operation_clue(gid, uid, view['id'], hidden, now=self.clock())
        except VaultStoreError as error:
            return await self.say(interaction, t.error(error))
        self.mark(view['id'], uid)
        await self.say(interaction, t('Clue from {name}: {text}', name=entry['name'] or t('a deleted character'), text=clue_text(entry['constraint'], t=t)))

    async def _advisor_picker(self, interaction, view: dict, t) -> None:
        gid, uid = interaction.guild_id, interaction.user.id
        parent_id, _ = self.bot.location(interaction.channel)
        rows = [r for r in self.store.vault_partner_choices(gid, uid, view['channel_id'], parent_id) if r['source'] == 'favorite']
        if not rows:
            return await self.say(interaction, t('You have no favorite to bring as an advisor here.'))
        select = AdvisorSelect(view['id'], [discord.SelectOption(label=r['name'][:100], value=str(r['character_id'])) for r in rows[:25]],
                               t('Choose a favorite to advise you. Choose none to clear.'))
        picker = discord.ui.View(timeout=600)
        picker.add_item(select)
        try:
            await interaction.response.send_message(t('Choose an advisor.'), ephemeral=True, view=picker, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning('Vault advisor picker failed: %s', type(error).__name__)

    async def select_advisor(self, interaction, operation_id: int, character_id) -> None:
        t = invoker_t(interaction)
        gid, uid = interaction.guild_id, interaction.user.id
        if gid is None:
            return await self.say(interaction, t.error(UserError(codes.USE_IN_SERVER)))
        try:
            view = self.store.set_operation_advisor(gid, uid, operation_id, character_id, now=self.clock())
        except VaultStoreError as error:
            return await self.say(interaction, t.error(error))
        mine = next((p['advisor'] for p in view['participants'] if p['user_id'] == uid), None)
        self.mark(operation_id, uid)
        await self.say(interaction, t('Your advisor: {name}', name=mine['name'] or t('a deleted character')) if mine else t('You no longer have an advisor.'))

    async def _ask(self, interaction, view: dict, action: str, t) -> None:
        gid, uid, oid, iid = interaction.guild_id, interaction.user.id, view['id'], str(interaction.id)
        stage = view['stage']
        mine = next(p['advisor'] for p in view['participants'] if p['user_id'] == uid)
        cid = (mine['character_id'] if mine else None) if action == 'ask_advisor' else int(action[4:])
        allowed = {n['character_id'] for n in view['npcs']} | ({mine['character_id']} if mine else set())
        if cid is None or cid not in allowed:
            return await self.say(interaction, t.error(VaultStoreError(codes.VAULT_ADVICE_NOT_PARTNER)))
        replayed = self.store.one("SELECT 1 FROM vault_operation_actions WHERE guild_id=? AND operation_id=? AND kind='advice' AND action_key=?", (gid, oid, iid)) is not None
        if not replayed:
            cap, limit = self.store.vault_settings(gid)['advice_cap'], self.store.game_talk_daily_limit(gid)
            if len(stage['advice']) + 1 > cap:
                return await self.say(interaction, t.error(VaultStoreError(codes.VAULT_ADVICE_LIMIT, limit=cap)))
            if self.store.game_talk_calls_today(gid, self.clock()) + 1 > limit:
                return await self.say(interaction, t.error(VaultStoreError(codes.VAULT_ADVICE_TALK_LIMIT, limit=limit)))
            try:
                await interaction.response.defer(ephemeral=True, thinking=True)
            except discord.HTTPException as error:
                return logging.warning('Vault could not acknowledge the press: %s', type(error).__name__)
        try:
            text = usage = None
            if not replayed:
                style = self.store.resolved_play_style(gid, cid)
                facts = build_advice_facts(stage['constraints'], stage['protocol'], stage['length'], risk=style.risk, aggression=style.aggression,
                                           unpredictability=style.unpredictability,
                                           seed=f"vault-op-advice/{oid}/{stage['number']}/{stage['retry']}/{cid}/{stage['turns_used']}")
                prompt = ADVICE_USER.format(guess=facts.guess, remaining=ADVICE_COUNT.format(remaining=facts.remaining) if len(str(facts.remaining)) < len(facts.guess) else '')
                with self.bot.backend.pin():
                    text, usage = await self._word(interaction, cid, prompt)
            try:
                entry = self.store.record_operation_advice(gid, uid, oid, cid, text=text, usage=usage, action_key=iid, now=self.clock())
            except VaultStoreError as error:
                return await self.say(interaction, t.error(error))
            await self.say(interaction, advice_line(entry, t))
        except Exception as error:
            logging.error('Vault operation advice failed: %s', type(error).__name__)
            await self.say(interaction, t(SOMETHING_WRONG))

    # ---- modal submits

    async def submit_guess(self, interaction, operation_id: int, text: str) -> None:
        t = invoker_t(interaction)
        gid, uid = interaction.guild_id, interaction.user.id
        if gid is None:
            return await self.say(interaction, t.error(UserError(codes.USE_IN_SERVER)))
        try:
            before = self.store.vault_operation(gid, uid, operation_id, now=self.clock())
            result = self.store.submit_operation_guess(gid, uid, operation_id, text, action_key=str(interaction.id), now=self.clock())
        except VaultStoreError as error:
            return await self.say(interaction, t.error(error))
        protocol = (before['stage'] or before['stages'][0])['protocol']
        await self.say(interaction, t('Guess {n}: {guess} — {feedback}', n=result['number'], guess=result['guess'], feedback=feedback_text(protocol, result['feedback'], t=t)) + refused_note(result, t))
        await self._done(interaction, result, uid)

    async def submit_tool(self, interaction, operation_id: int, tool_key: str, values: dict) -> None:
        t = invoker_t(interaction)
        gid, uid = interaction.guild_id, interaction.user.id
        if gid is None:
            return await self.say(interaction, t.error(UserError(codes.USE_IN_SERVER)))
        try:
            view = self.store.vault_operation(gid, uid, operation_id, now=self.clock())
            if tool_key not in TOOLS:
                raise VaultStoreError(codes.VAULT_TOOL_UNKNOWN, tool=str(tool_key))
            result = self.store.use_operation_tool(gid, uid, operation_id, tool_key, self._tool_params(view['stage'] or {'guesses': []}, tool_key, values),
                                                   action_key=str(interaction.id), now=self.clock())
        except VaultStoreError as error:
            text = t.error(error)
            reason = TOOL_REASONS.get(error.params.get('reason')) if error.code == codes.VAULT_TOOL_BAD_PARAMS else None
            return await self.say(interaction, f'{text} {t.render(reason)}' if reason else text)
        await self.say(interaction, tool_line(result, t) + refused_note(result, t))
        await self._done(interaction, result, uid)

    async def submit_cross(self, interaction, operation_id: int, first: str, second: str) -> None:
        t = invoker_t(interaction)
        gid, uid = interaction.guild_id, interaction.user.id
        if gid is None:
            return await self.say(interaction, t.error(UserError(codes.USE_IN_SERVER)))
        numbers = [int(x) if x.strip().isascii() and x.strip().isdigit() else x for x in (first, second)]
        try:
            result = self.store.operation_cross_validate(gid, uid, operation_id, numbers[0], numbers[1], action_key=str(interaction.id), now=self.clock())
        except VaultStoreError as error:
            return await self.say(interaction, t.error(error))
        await self.say(interaction, cross_line({'numbers': result['numbers'], 'count': result['count'], 'common': result['common']}, t) + refused_note(result, t))
        await self._done(interaction, result, uid)


def register_vault_operation_commands(bot, ctx: SimpleNamespace) -> None:
    require_guild = ctx.require_guild
    bot.vault_operation = play = VaultOperationPlay(bot)
    vault = bot.tree.get_command('vault')

    @vault.command(name='operation', description=C_('Join the weekly co-op vault operation in this channel'))
    async def vault_operation(interaction: discord.Interaction):
        require_guild(interaction)
        await play.open_command(interaction)
