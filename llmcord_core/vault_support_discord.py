"""Vault support missions on Discord (FEAT-67 phase B, D43, Experimental): the Support mission button, the picker, the mission message and the answers.

A mixin of ``VaultOperationPlay``. The store (``vault_support_store``) is the only source of truth and raises codes; this layer takes input and builds
words. Every reply is ephemeral in the invoker's language and nothing is posted in the channel. The answer appears only once the mission is over.
Creating a forensics puzzle takes about 0.2 s: the press is acknowledged first and the store runs on the event loop (the shared sqlite connection is
not safe to write from another thread while the loop commits its own work).
"""
from __future__ import annotations

import logging

import discord

from .vault_discord import NO_MENTIONS, SOMETHING_WRONG, feedback_text
from .vault_store import VaultStoreError
from llmcord_core.i18n import UserError, codes
from llmcord_core.i18n.discord_locale import invoker_t

MISSION_ACTIONS = ('mission', 'mission_forensics', 'mission_bulls_cows', 'mission_answer')


def mission_name(kind: str, t) -> str:
    return t('Record forensics') if kind == 'forensics' else t('Three-digit Bulls and Cows')


def _feedback(result, t) -> str:
    return feedback_text('standard', result, t=t)


def _answer_line(mission: dict, t) -> str:
    if mission['type'] == 'bulls_cows':
        return t('The code was {code}.', code=mission['answer'])
    return t('The hidden result was {feedback}.', feedback=_feedback(mission['answer'], t))


def _left_line(mission: dict, t) -> str:
    return t('Attempts left: {left} of {budget}', left=mission['attempts_left'], budget=mission['turn_budget'])


def render_mission(mission: dict, t) -> str:
    """The mission message. The answer line exists only for a solved or failed mission (the store withholds it until then)."""
    lines = [t('Support mission: {name}', name=mission_name(mission['type'], t))]
    if mission['type'] == 'forensics':
        lines.append(t('Recorded guesses of one secret code:'))
        hidden = mission['puzzle']['hidden']
        for n, record in enumerate(mission['puzzle']['records'], 1):
            lines.append(t('{n}. {guess} — {feedback}', n=n, guess=record['guess'], feedback='?' if record['result'] is None else _feedback(record['result'], t)))
        lines.append(t('What are the strikes and balls of line {n}?', n=hidden + 1))
    else:
        lines.append(t('Guess a code of 3 different digits.'))
        lines += [t('{n}. {guess} — {feedback}', n=a['number'], guess=a['answer'], feedback=_feedback(a['feedback'], t)) for a in mission['history']]
    lines.append(_left_line(mission, t))
    if mission['status'] == 'solved':
        lines += [t('Support mission complete!'), _answer_line(mission, t)]
    elif mission['status'] == 'failed':
        lines += [t('Support mission failed.'), _answer_line(mission, t)]
    return '\n'.join(lines)


def render_result(result: dict, t) -> str:
    mission, event = result['mission'], result['event']
    if event is None:
        entry = mission['history'][result['number'] - 1]
        if mission['type'] == 'bulls_cows':
            first = t('Answer {n}: {guess} — {feedback}', n=result['number'], guess=entry['answer'], feedback=_feedback(result['feedback'], t))
        else:
            first = t('Answer {n}: {feedback}', n=result['number'], feedback=_feedback(entry['answer'], t))
        return '\n'.join([first, _left_line(mission, t)])
    lines = [t('Support mission complete!') if event == 'solved' else t('Support mission failed.'), _answer_line(mission, t)]
    if event == 'solved':
        reward = mission['reward']
        lines.append(t('Reward: {amount}', amount=reward['amount']) if reward and reward['amount'] > 0 else t('Your reward could not be paid because your balance is at its limit.')
                     if result.get('reward_refused') else t('No reward was paid for this mission.'))
    return '\n'.join(lines)


class SupportMissions:
    async def _send(self, interaction, text: str, view=None) -> None:
        try:
            send = interaction.followup.send if interaction.response.is_done() else interaction.response.send_message
            await send(text, ephemeral=True, allowed_mentions=NO_MENTIONS, **({} if view is None else {'view': view}))
        except discord.HTTPException as error:
            logging.warning('Vault support mission reply failed: %s', type(error).__name__)

    async def _mission_show(self, interaction, mission: dict, oid: int, t) -> None:
        from .vault_operation_discord import VaultOperationButton
        view = None
        if mission['status'] == 'active':
            view = discord.ui.View(timeout=None)
            view.add_item(VaultOperationButton('mission_answer', oid, t.p('button.vault', 'Answer'), discord.ButtonStyle.primary))
        await self._send(interaction, render_mission(mission, t), view)

    async def _mission_press(self, interaction, view: dict, action: str, t) -> None:
        from .vault_operation_discord import OperationMissionModal, VaultOperationButton
        gid, uid, oid = interaction.guild_id, interaction.user.id, view['id']
        try:
            mission = self.store.support_mission(gid, uid, oid, now=self.clock())
            if action == 'mission_answer':   # a stale Answer button on a finished mission still opens the modal: the submit replays or refuses
                if mission is None:
                    raise VaultStoreError(codes.VAULT_MISSION_NONE)
                return await self._modal(interaction, OperationMissionModal(oid, mission['type'], t=t))
            if action == 'mission' and mission is None:
                picker = discord.ui.View(timeout=None)
                picker.add_item(VaultOperationButton('mission_forensics', oid, t.p('button.vault', 'Record forensics')))
                picker.add_item(VaultOperationButton('mission_bulls_cows', oid, t.p('button.vault', 'Three-digit Bulls and Cows')))
                return await self._send(interaction, t('Choose a support mission.'), picker)
            if action != 'mission':
                kind = action[len('mission_'):]
                if mission is None and kind == 'forensics' and not await self._defer(interaction):
                    return
                mission = self.store.start_support_mission(gid, uid, oid, kind, now=self.clock())
        except VaultStoreError as error:
            return await self.say(interaction, t.error(error))
        await self._mission_show(interaction, mission, oid, t)

    async def submit_mission(self, interaction, operation_id: int, mission_type: str, values: dict) -> None:
        t = invoker_t(interaction)
        gid, uid = interaction.guild_id, interaction.user.id
        if gid is None:
            return await self.say(interaction, t.error(UserError(codes.USE_IN_SERVER)))
        try:
            if mission_type == 'forensics':
                pair = [str(values.get(name, '')).strip() for name in ('strikes', 'balls')]
                if not all(x.isascii() and x.isdigit() and len(x) <= 3 for x in pair):
                    raise VaultStoreError(codes.VAULT_MISSION_ANSWER_INVALID)
                answer = (int(pair[0]), int(pair[1]))
            else:
                answer = str(values.get('code', '')).strip()
            result = self.store.submit_support_answer(gid, uid, operation_id, answer, action_key=str(interaction.id), now=self.clock())
        except VaultStoreError as error:
            return await self.say(interaction, t.error(error))
        except Exception as error:
            logging.error('Vault support mission failed: %s', type(error).__name__)
            return await self.say(interaction, t(SOMETHING_WRONG))
        await self._send(interaction, render_result(result, t))
