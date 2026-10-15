"""Vault operation persistence (FEAT-66, D43). Mixed into Store next to VaultStore, whose helpers it reuses.

One weekly co-op operation per guild, configured channel and server-local week: three sequential stages with one shared turn budget each,
one automatic retry per stage, two cast characters holding the clues, optional favorite advisors, cross-validation between two members'
guesses. Settings (off by default) sit in vault_operation_settings. A stage row holds the CURRENT attempt; its actions carry the attempt
number, so a retry hides the previous guesses. The secret leaves the store only in the view of an ended stage. Refusals are
VaultStoreError codes; the Discord edge turns them into words. Rewards are paid through the ledger in the transaction that qualifies.
"""
from __future__ import annotations

import dataclasses
import json
import random
import secrets
import time
from datetime import datetime, time as day_time, timedelta

from .admin_store import ConflictError
from .i18n import UserError, codes
from .items_store import NotHeld
from .vault import (DEFAULT_LENGTH, PROTOCOLS, TOOLS, CommonCount, DigitAt, GuessResult, OpenDigit, VaultError, build_advice_facts, check_wording, cross_validation_common,
                    evaluate_tool, feedback, filter_candidates, is_redundant, make_clues, open_digit, played, random_secret, tool_fits)
from .vault_store import CLUE_COUNT, MAX_TOOLS, VaultStore, VaultStoreError

STAGES = 3
MAX_MIN_GUESSES = 30
DEFAULT_STAGE_PROTOCOLS = ['standard', 'standard', 'open']
CONTRIBUTIONS = ('guess', 'tool', 'cross')


class VaultOperationStore(VaultStore):
    def vault_operation_settings(self, guild_id):
        row = self.one('SELECT * FROM vault_operation_settings WHERE guild_id=?', (guild_id,))
        protocols = json.loads(row['stage_protocols_json'] or 'null') if row else None
        return {'enabled': bool(row['enabled']) if row else False, 'channel_id': row['channel_id'] if row else None,
                'week_start': row['week_start'] if row else 0, 'stage_protocols': protocols or list(DEFAULT_STAGE_PROTOCOLS),
                'contribution_reward': row['contribution_reward'] if row else 0, 'team_reward': row['team_reward'] if row else 0,
                'min_guesses': row['min_guesses'] if row else 1, 'support_reward': row['support_reward'] if row else 0, 'revision': row['revision'] if row else 0}

    def set_vault_operation_settings(self, guild_id, *, enabled=None, channel_id=None, week_start=None, stage_protocols=None, contribution_reward=None,
                                     team_reward=None, min_guesses=None, support_reward=None, expected_revision=None):
        def whole(value, low, high):
            return not isinstance(value, bool) and isinstance(value, int) and low <= value <= high
        if enabled is not None and not isinstance(enabled, bool):
            raise UserError(codes.VAULT_SETTINGS_FLAG)
        if channel_id is not None and not whole(channel_id, 1, 2 ** 63 - 1):
            raise UserError(codes.VAULT_OPERATION_SETTINGS_CHANNEL)
        if week_start is not None and not whole(week_start, 0, 6):
            raise UserError(codes.VAULT_OPERATION_SETTINGS_WEEK_START)
        if stage_protocols is not None and (not isinstance(stage_protocols, (list, tuple)) or len(stage_protocols) != STAGES
                                            or not all(isinstance(p, str) and p in PROTOCOLS for p in stage_protocols)):
            raise UserError(codes.VAULT_OPERATION_SETTINGS_PROTOCOLS)
        for reward in (contribution_reward, team_reward, support_reward):
            if reward is not None and not whole(reward, 0, self.MAX_CURRENCY_CHANGE):
                raise UserError(codes.VAULT_SETTINGS_REWARD, limit=self.MAX_CURRENCY_CHANGE)
        if min_guesses is not None and not whole(min_guesses, 1, MAX_MIN_GUESSES):
            raise UserError(codes.VAULT_OPERATION_SETTINGS_MIN_GUESSES, low=1, high=MAX_MIN_GUESSES)
        with self.write_admin():
            current = self.vault_operation_settings(guild_id)
            if expected_revision is not None and current['revision'] != expected_revision:
                raise ConflictError(codes.VAULT_SETTINGS_CHANGED)
            new = {'enabled': current['enabled'] if enabled is None else enabled, 'channel_id': current['channel_id'] if channel_id is None else channel_id,
                   'week_start': current['week_start'] if week_start is None else week_start,
                   'stage_protocols': current['stage_protocols'] if stage_protocols is None else list(stage_protocols),
                   'contribution_reward': current['contribution_reward'] if contribution_reward is None else contribution_reward,
                   'team_reward': current['team_reward'] if team_reward is None else team_reward,
                   'min_guesses': current['min_guesses'] if min_guesses is None else min_guesses,
                   'support_reward': current['support_reward'] if support_reward is None else support_reward, 'revision': current['revision'] + 1}
            self.db.execute('INSERT INTO vault_operation_settings(guild_id,enabled,channel_id,week_start,stage_protocols_json,contribution_reward,team_reward,min_guesses,support_reward,revision,updated_at) '
                'VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(guild_id) DO UPDATE SET enabled=excluded.enabled,channel_id=excluded.channel_id,week_start=excluded.week_start,'
                'stage_protocols_json=excluded.stage_protocols_json,contribution_reward=excluded.contribution_reward,team_reward=excluded.team_reward,'
                'min_guesses=excluded.min_guesses,support_reward=excluded.support_reward,revision=excluded.revision,updated_at=excluded.updated_at',
                (guild_id, int(new['enabled']), new['channel_id'], new['week_start'], json.dumps(new['stage_protocols']), new['contribution_reward'],
                 new['team_reward'], new['min_guesses'], new['support_reward'], new['revision'], time.time()))
        return self.vault_operation_settings(guild_id)

    # ---- lifecycle

    def _op_week(self, guild_id, week_start, now):
        """(week key, Unix start, Unix end): the latest server-local week_start midnight at or before `now`, and seven local calendar days later."""
        zone = self._vault_zone(guild_id)
        local = datetime.fromtimestamp(now, zone)
        first = local.date() - timedelta(days=(local.weekday() - week_start) % 7)
        return (first.isoformat(), datetime.combine(first, day_time(0), zone).timestamp(),
                datetime.combine(first + timedelta(days=7), day_time(0), zone).timestamp())

    def _op_finish(self, guild_id, operation_id, status, now):
        self.db.execute("UPDATE vault_operations SET status=?,finished_at=? WHERE id=? AND guild_id=? AND status='active'", (status, now, operation_id, guild_id))
        self.db.execute("UPDATE vault_operation_stages SET status=? WHERE operation_id=? AND guild_id=? AND status='active'", (status, operation_id, guild_id))

    def _op_expire(self, guild_id, now, channel_id=None):
        who, args = ('', ()) if channel_id is None else (' AND channel_id=?', (channel_id,))
        ids = [r['id'] for r in self.all(f"SELECT id FROM vault_operations WHERE guild_id=? AND status='active' AND expires_at<=?{who}", (guild_id, now, *args))]
        for operation_id in ids:
            self._op_finish(guild_id, operation_id, 'expired', now)

    def _op_npcs(self, guild_id, channel_id, parent_id, npc_ids):
        binding = self.binding(channel_id, parent_id)
        eligible = {c['id'] for c in self.eligible_characters(guild_id, binding['space_id'])} if binding is not None and binding['guild_id'] == guild_id else set()
        cast = [c for c in self.get_cast(channel_id, parent_id) if c in eligible]
        if npc_ids is None:
            chosen = cast[:2]
        else:
            chosen = list(npc_ids) if isinstance(npc_ids, (list, tuple)) else []
            if len(chosen) != 2 or len(set(chosen)) != 2 or any(isinstance(c, bool) or not isinstance(c, int) or c not in cast for c in chosen):
                raise VaultStoreError(codes.VAULT_OPERATION_NPCS)
        if len(chosen) != 2:
            raise VaultStoreError(codes.VAULT_OPERATION_NPCS)
        return chosen

    def open_vault_operation(self, guild_id, user_id, channel_id, *, npc_ids=None, now=None, seed=None, parent_id=None):
        """Joins the current week's operation of the configured channel (costs no turn), creating it first when there is none.
        `seed` is a test seam only: callers must never pass one built from public values (it fixes the stage-1 secret)."""
        now = time.time() if now is None else now
        if not self.vault_settings(guild_id)['enabled']:
            raise VaultStoreError(codes.VAULT_DISABLED)
        settings = self.vault_operation_settings(guild_id)
        if not settings['enabled']:
            raise VaultStoreError(codes.VAULT_OPERATION_DISABLED)
        if settings['channel_id'] is None or settings['channel_id'] != channel_id:
            raise VaultStoreError(codes.VAULT_OPERATION_CHANNEL)
        with self.write_admin():
            self._op_expire(guild_id, now, channel_id)
        with self.write_admin():
            row = self.one('SELECT * FROM vault_operations WHERE guild_id=? AND channel_id=? AND expires_at>? ORDER BY id DESC LIMIT 1', (guild_id, channel_id, now))
            created = False
            if row is not None and row['status'] != 'active':
                raise VaultStoreError(codes.VAULT_NOT_ACTIVE)
            if row is None:
                week_key, starts_at, expires_at = self._op_week(guild_id, settings['week_start'], now)
                row = self.one('SELECT * FROM vault_operations WHERE guild_id=? AND channel_id=? AND week_key=?', (guild_id, channel_id, week_key))  # the timezone moved back onto a used week
                if row is not None and row['status'] != 'active':
                    raise VaultStoreError(codes.VAULT_NOT_ACTIVE)
                if row is None:
                    npcs = self._op_npcs(guild_id, channel_id, parent_id, npc_ids)
                    oid = self.db.execute('INSERT INTO vault_operations(guild_id,channel_id,week_key,starts_at,expires_at,npc_ids_json,protocols_json,created_by,created_at) VALUES(?,?,?,?,?,?,?,?,?)',
                        (guild_id, channel_id, week_key, starts_at, expires_at, json.dumps(npcs), json.dumps(settings['stage_protocols']), user_id, now)).lastrowid
                    row = self.one('SELECT * FROM vault_operations WHERE id=? AND guild_id=?', (oid, guild_id))
                    self._op_start_stage(row, 1, 0, now, seed)
                    created = True
            joined = self.db.execute('INSERT OR IGNORE INTO vault_operation_participants(guild_id,operation_id,user_id,joined_at) VALUES(?,?,?,?)', (guild_id, row['id'], user_id, now)).rowcount > 0
            return self._op_view(row, created=created, joined=joined)

    def current_vault_operation_id(self, guild_id, channel_id, *, now=None):
        now = time.time() if now is None else now
        row = self.one("SELECT id FROM vault_operations WHERE guild_id=? AND channel_id=? AND status='active' AND expires_at>? ORDER BY id DESC LIMIT 1", (guild_id, channel_id, now))
        return row['id'] if row else None

    def _op_load(self, guild_id, user_id, operation_id, now):
        """The operation row for one of its participants, with a lazy expiry committed first."""
        row = self.one('SELECT * FROM vault_operations WHERE id=? AND guild_id=?', (operation_id, guild_id))
        if row is None:
            raise VaultStoreError(codes.VAULT_NOT_FOUND)
        if self.one('SELECT 1 FROM vault_operation_participants WHERE guild_id=? AND operation_id=? AND user_id=?', (guild_id, operation_id, user_id)) is None:
            raise VaultStoreError(codes.VAULT_OPERATION_NOT_PARTICIPANT)
        if row['status'] == 'active' and row['expires_at'] <= now:
            with self.write_admin():
                self._op_finish(guild_id, operation_id, 'expired', now)
            row = self.one('SELECT * FROM vault_operations WHERE id=? AND guild_id=?', (operation_id, guild_id))
        return row

    def vault_operation(self, guild_id, user_id, operation_id, *, now=None):
        now = time.time() if now is None else now
        return self._op_view(self._op_load(guild_id, user_id, operation_id, now))

    def _op_active(self, guild_id, operation_id, now):
        """Inside write_admin: the fresh operation row and its active stage row, or vault.not_active."""
        row = self.one('SELECT * FROM vault_operations WHERE id=? AND guild_id=?', (operation_id, guild_id))
        stage = self.one("SELECT * FROM vault_operation_stages WHERE operation_id=? AND guild_id=? AND status='active'", (operation_id, guild_id))
        if row is None or row['status'] != 'active' or row['expires_at'] <= now or stage is None:
            raise VaultStoreError(codes.VAULT_NOT_ACTIVE)
        return row, stage

    def _op_start_stage(self, row, number, retry, now, seed=None):
        """Inside write_admin: a fresh secret, budget and clues for stage `number` (a retry replaces the stage's current attempt; the failed attempt's code is kept in previous_secret)."""
        guild_id, oid = row['guild_id'], row['id']
        protocol = json.loads(row['protocols_json'])[number - 1]
        rng = random.Random(f'{seed}/{number}/{retry}') if seed is not None else secrets.SystemRandom()
        used = {r['secret'] for r in self.all('SELECT secret FROM vault_operation_stages WHERE operation_id=? AND guild_id=?', (oid, guild_id))}
        code = random_secret(DEFAULT_LENGTH, rng)
        while code in used:
            code = random_secret(DEFAULT_LENGTH, rng)
        shown = open_digit(code, rng) if protocol == 'open' else None
        budget = self.vault_settings(guild_id)['turns'][protocol]
        if retry:
            self.db.execute("UPDATE vault_operation_stages SET retry=?,previous_secret=secret,secret=?,open_digit=?,turn_budget=?,status='active',created_at=? WHERE operation_id=? AND guild_id=? AND number=?",
                (retry, code, shown, budget, now, oid, guild_id, number))
            self.db.execute('DELETE FROM vault_operation_clues WHERE operation_id=? AND guild_id=? AND stage_number=?', (oid, guild_id, number))
        else:
            self.db.execute('INSERT INTO vault_operation_stages(guild_id,operation_id,number,protocol,length,turn_budget,secret,open_digit,created_at) VALUES(?,?,?,?,?,?,?,?,?)',
                (guild_id, oid, number, protocol, DEFAULT_LENGTH, budget, code, shown, now))
        npcs = json.loads(row['npc_ids_json'])
        for i, clue in enumerate(make_clues(code, CLUE_COUNT, rng)):
            self.db.execute('INSERT INTO vault_operation_clues(guild_id,operation_id,stage_number,position,character_id,kind,data_json) VALUES(?,?,?,?,?,?,?)',
                (guild_id, oid, number, i, npcs[i % len(npcs)], type(clue).__name__, json.dumps(dataclasses.asdict(clue), sort_keys=True)))

    # ---- view

    def _op_view(self, row, created=False, joined=False):
        gid, oid = row['guild_id'], row['id']
        npc_ids = json.loads(row['npc_ids_json'])
        protocols = json.loads(row['protocols_json'])
        people = self.all('SELECT * FROM vault_operation_participants WHERE guild_id=? AND operation_id=? ORDER BY id', (gid, oid))
        stages = {s['number']: s for s in self.all('SELECT * FROM vault_operation_stages WHERE guild_id=? AND operation_id=?', (gid, oid))}
        clue_rows = self.all('SELECT * FROM vault_operation_clues WHERE guild_id=? AND operation_id=? ORDER BY stage_number,position', (gid, oid))
        actions = self.all('SELECT * FROM vault_operation_actions WHERE guild_id=? AND operation_id=? ORDER BY id', (gid, oid))
        claims = self.all('SELECT * FROM vault_operation_claims WHERE guild_id=? AND operation_id=?', (gid, oid))
        names = self.character_names(gid, npc_ids + [p['advisor_id'] for p in people if p['advisor_id']] + [a['character_id'] for a in actions if a['character_id']])
        reversed_ids = {r['reverses_id'] for r in self.all(f"SELECT reverses_id FROM currency_ledger WHERE guild_id=? AND reverses_id IN ({','.join('?' * len(claims))})",
                                                            (gid, *[c['ledger_id'] for c in claims]))} if claims else set()
        contributions = {}
        for a in actions:
            if a['kind'] in CONTRIBUTIONS:
                contributions[a['user_id']] = contributions.get(a['user_id'], 0) + 1
        settings = self.vault_operation_settings(gid)
        owed_guesses = dict(contributions)
        solved_missions = set()
        for m in self.all("SELECT user_id FROM vault_support_missions WHERE guild_id=? AND operation_id=? AND status='solved'", (gid, oid)):
            contributions[m['user_id']] = contributions.get(m['user_id'], 0) + 1
            solved_missions.add(m['user_id'])

        def claim(user_id, kind):
            c = next((c for c in claims if c['user_id'] == user_id and c['type'] == kind), None)
            return {'amount': c['amount'], 'ledger_id': c['ledger_id'], 'reversed': c['ledger_id'] in reversed_ids} if c else None

        def refused(user_id):
            """A reward that is owed and unpaid: the claim is written in the same transaction as the credit, so only a balance cap leaves it missing."""
            need = settings['min_guesses']
            return bool((settings['contribution_reward'] > 0 and owed_guesses.get(user_id, 0) >= need and not claim(user_id, 'contribution'))
                        or (settings['team_reward'] > 0 and row['status'] == 'solved' and contributions.get(user_id, 0) >= need and not claim(user_id, 'team'))
                        or (settings['support_reward'] > 0 and user_id in solved_missions and not claim(user_id, 'support')))

        entries = []
        for number in range(1, STAGES + 1):
            s = stages.get(number)
            if s is None:
                entries.append({'number': number, 'protocol': protocols[number - 1], 'length': DEFAULT_LENGTH, 'status': 'locked', 'retry': 0, 'turn_budget': None,
                                'turns_used': 0, 'turns_left': None, 'open_digit': None, 'guesses': [], 'tools': [], 'tools_used': 0, 'cross': [], 'clues': [],
                                'advice': [], 'constraints': [], 'secret': None, 'previous_secret': None})
                continue
            entries.append(self._op_stage_entry(s, [a for a in actions if a['stage_number'] == number and a['attempt'] == s['retry']],
                                                [c for c in clue_rows if c['stage_number'] == number], names))
        return {'id': oid, 'guild_id': gid, 'channel_id': row['channel_id'], 'week_key': row['week_key'], 'starts_at': row['starts_at'], 'expires_at': row['expires_at'],
                'status': row['status'], 'created': created, 'joined': joined, 'npcs': [{'character_id': i, 'name': names.get(i)} for i in npc_ids],
                'participants': [{'user_id': p['user_id'], 'joined_at': p['joined_at'],
                                  'advisor': {'character_id': p['advisor_id'], 'name': names.get(p['advisor_id'])} if p['advisor_id'] else None,
                                  'contributions': contributions.get(p['user_id'], 0), 'reward_contribution': claim(p['user_id'], 'contribution'),
                                  'reward_team': claim(p['user_id'], 'team'), 'reward_support': claim(p['user_id'], 'support'),
                                  'reward_refused': refused(p['user_id'])} for p in people],
                'stage': next((e for e in entries if e['status'] == 'active'), None), 'stages': entries}

    def _op_stage_entry(self, s, actions, clue_rows, names):
        ended = s['status'] != 'active'
        guesses = [{'number': a['number'], 'guess': a['guess'], 'feedback': tuple(json.loads(a['feedback_json'])), 'won': bool(a['won']), 'user_id': a['user_id']}
                   for a in actions if a['kind'] == 'guess']
        tool_rows = [a for a in actions if a['kind'] == 'tool']
        cross_rows = [a for a in actions if a['kind'] == 'cross']
        clues = [self._vault_clue_entry(c, names, ended) for c in clue_rows]
        constraints = ([OpenDigit(s['open_digit'])] if s['open_digit'] else []) + [c['constraint'] for c in clues if 'constraint' in c]
        for g in guesses:
            constraints += played(s['protocol'], g['guess'], g['feedback'], g['won'])
        constraints += [self._vault_tool_constraint(t) for t in tool_rows]
        crosses = []
        for x in cross_rows:
            data = json.loads(x['data_json'])
            constraints.append(CommonCount(**json.loads(x['constraint_json'])))
            crosses.append({'numbers': tuple(data['numbers']), 'common': data['common'], 'count': data['count'], 'user_id': x['user_id']})
        used = len(guesses) + sum(a['turn_cost'] for a in tool_rows + cross_rows)
        return {'number': s['number'], 'protocol': s['protocol'], 'length': s['length'], 'status': s['status'], 'retry': s['retry'], 'turn_budget': s['turn_budget'],
                'turns_used': used, 'turns_left': max(s['turn_budget'] - used, 0), 'open_digit': s['open_digit'], 'guesses': guesses,
                'tools': [{**self._vault_tool_entry(t), 'user_id': t['user_id']} for t in tool_rows], 'tools_used': len(tool_rows), 'cross': crosses, 'clues': clues,
                'advice': [{**self._vault_advice_entry(a, names), 'user_id': a['user_id']} for a in actions if a['kind'] == 'advice'],
                'constraints': constraints, 'secret': s['secret'] if ended else None,
                'previous_secret': s['previous_secret'] if s['retry'] >= 1 else None}

    # ---- shared steps

    def _op_replay(self, guild_id, operation_id, kind, action_key):
        if action_key is None:
            return None
        return self.one('SELECT * FROM vault_operation_actions WHERE guild_id=? AND operation_id=? AND kind=? AND action_key=?', (guild_id, operation_id, kind, action_key))

    def _op_attempt(self, stage, kind=None):
        extra, args = ('', ()) if kind is None else (' AND kind=?', (kind,))
        return self.all(f'SELECT * FROM vault_operation_actions WHERE guild_id=? AND operation_id=? AND stage_number=? AND attempt=?{extra} ORDER BY id',
                        (stage['guild_id'], stage['operation_id'], stage['number'], stage['retry'], *args))

    def _op_used(self, stage):
        return sum(1 if a['kind'] == 'guess' else a['turn_cost'] for a in self._op_attempt(stage))

    def _op_public(self, stage):
        """The public constraints of the stage's current attempt (never the secret)."""
        names = {}
        clues = self.all('SELECT * FROM vault_operation_clues WHERE guild_id=? AND operation_id=? AND stage_number=? ORDER BY position', (stage['guild_id'], stage['operation_id'], stage['number']))
        return self._op_stage_entry(stage, self._op_attempt(stage), clues, names)['constraints']

    def _op_insert(self, stage, user_id, kind, now, action_key=None, **fields):
        columns = ['guild_id', 'operation_id', 'stage_number', 'attempt', 'kind', 'user_id', 'action_key', 'created_at', *fields]
        values = [stage['guild_id'], stage['operation_id'], stage['number'], stage['retry'], kind, user_id, action_key, now, *fields.values()]
        action_id = self.db.execute(f"INSERT INTO vault_operation_actions({','.join(columns)}) VALUES({','.join('?' * len(columns))})", values).lastrowid
        return self.one('SELECT * FROM vault_operation_actions WHERE id=? AND guild_id=?', (action_id, stage['guild_id']))

    def _op_pay(self, row, user_id, kind, amount, now):
        """Inside write_admin: one ledger credit and its unique claim. True when a balance cap refused it, leaving it unpaid (retried on a later action)."""
        if amount <= 0:
            return False
        try:
            entry = self._append_ledger(row['guild_id'], user_id, amount, f"Vault operation {'team ' if kind == 'team' else 'support mission ' if kind == 'support' else ''}reward ({row['week_key']})", user_id, 'vault', None)
        except UserError:
            return True
        self.db.execute('INSERT INTO vault_operation_claims(guild_id,operation_id,user_id,type,amount,ledger_id,created_at) VALUES(?,?,?,?,?,?,?)',
            (row['guild_id'], row['id'], user_id, kind, amount, entry['id'], now))
        return False

    def _op_contributions(self, guild_id, operation_id, user_id, missions=False):
        """Qualifying contributions: guesses, tools and cross-validations; `missions` adds a solved support mission (team reward only, never the contribution reward)."""
        marks = ','.join('?' * len(CONTRIBUTIONS))
        solved = self.one("SELECT COUNT(*) FROM vault_support_missions WHERE guild_id=? AND operation_id=? AND user_id=? AND status='solved'", (guild_id, operation_id, user_id))[0] if missions else 0
        return solved + self.one(f'SELECT COUNT(*) FROM vault_operation_actions WHERE guild_id=? AND operation_id=? AND user_id=? AND kind IN ({marks})', (guild_id, operation_id, user_id, *CONTRIBUTIONS))[0]

    def _op_qualify(self, row, user_id, now, settings=None):
        """Inside write_admin: the contribution reward once the member qualifies. True when a balance cap refused it."""
        settings = settings or self.vault_operation_settings(row['guild_id'])
        if settings['contribution_reward'] <= 0 or self._op_contributions(row['guild_id'], row['id'], user_id) < settings['min_guesses']:
            return False
        if self.one("SELECT 1 FROM vault_operation_claims WHERE guild_id=? AND operation_id=? AND user_id=? AND type='contribution'", (row['guild_id'], row['id'], user_id)) is None:
            return self._op_pay(row, user_id, 'contribution', settings['contribution_reward'], now)
        return False

    def _op_team_reward(self, row, now, settings=None):
        """Inside write_admin: the team reward to every qualified member. Returns the user ids a balance cap refused."""
        settings = settings or self.vault_operation_settings(row['guild_id'])
        refused = set()
        if settings['team_reward'] <= 0:
            return refused
        for p in self.all('SELECT user_id FROM vault_operation_participants WHERE guild_id=? AND operation_id=? ORDER BY id', (row['guild_id'], row['id'])):
            qualified = self._op_contributions(row['guild_id'], row['id'], p['user_id'], missions=True) >= settings['min_guesses']
            if qualified and self.one("SELECT 1 FROM vault_operation_claims WHERE guild_id=? AND operation_id=? AND user_id=? AND type='team'", (row['guild_id'], row['id'], p['user_id'])) is None:
                if self._op_pay(row, p['user_id'], 'team', settings['team_reward'], now):
                    refused.add(p['user_id'])
        return refused

    def _op_advance(self, row, stage, won, now, settings=None, refused=None):
        """Inside write_admin, after a turn was spent: solves, retries or fails the stage. Returns the event name or None; `refused` (a set) collects the user ids whose team reward a cap refused."""
        guild_id, oid = row['guild_id'], row['id']
        if won:
            self.db.execute("UPDATE vault_operation_stages SET status='solved' WHERE id=? AND guild_id=?", (stage['id'], guild_id))
            if stage['number'] < STAGES:
                self._op_start_stage(row, stage['number'] + 1, 0, now)
                return 'stage_solved'
            self.db.execute("UPDATE vault_operations SET status='solved',finished_at=? WHERE id=? AND guild_id=?", (now, oid, guild_id))
            held = self._op_team_reward(row, now, settings)
            if refused is not None:
                refused |= held
            return 'operation_solved'
        if self._op_used(stage) < stage['turn_budget']:
            return None
        if stage['retry'] == 0:
            self._op_start_stage(row, stage['number'], 1, now)
            return 'stage_retry'
        self._op_finish(guild_id, oid, 'failed', now)
        return 'operation_failed'

    def _op_view_now(self, guild_id, operation_id):
        return self._op_view(self.one('SELECT * FROM vault_operations WHERE id=? AND guild_id=?', (operation_id, guild_id)))

    # ---- actions

    def submit_operation_guess(self, guild_id, user_id, operation_id, guess, *, action_key=None, now=None):
        now = time.time() if now is None else now
        self._op_load(guild_id, user_id, operation_id, now)
        with self.write_admin():
            done = self._op_replay(guild_id, operation_id, 'guess', action_key)
            if done is not None:
                return self._op_guess_result(done, guild_id, operation_id, True, None)
            row, stage = self._op_active(guild_id, operation_id, now)
            code = self._vault_valid_guess(stage, guess)
            guesses = self._op_attempt(stage, 'guess')
            if any(g['guess'] == code for g in guesses):
                raise VaultStoreError(codes.VAULT_GUESS_DUPLICATE)
            result = feedback(stage['protocol'], stage['secret'], code)
            won = code == stage['secret']
            action = self._op_insert(stage, user_id, 'guess', now, action_key, number=len(guesses) + 1, guess=code, feedback_json=json.dumps(list(result)), won=int(won))
            settings, owed = self.vault_operation_settings(guild_id), set()
            late = self._op_qualify(row, user_id, now, settings)
            event = self._op_advance(row, stage, won, now, settings, owed)
            return self._op_guess_result(action, guild_id, operation_id, False, event, late or user_id in owed)

    def _op_guess_result(self, action, guild_id, operation_id, replayed, event, refused=False):
        return {'reward_refused': refused, 'number': action['number'], 'guess': action['guess'], 'feedback': tuple(json.loads(action['feedback_json'])), 'won': bool(action['won']),
                'replayed': replayed, 'event': event, 'operation': self._op_view_now(guild_id, operation_id)}

    def reveal_operation_clue(self, guild_id, user_id, operation_id, position, *, now=None):
        """Shows one NPC clue (1-based) to the whole team. Free and idempotent: the first call stores revealed_at."""
        now = time.time() if now is None else now
        self._op_load(guild_id, user_id, operation_id, now)
        with self.write_admin():
            row, stage = self._op_active(guild_id, operation_id, now)
            if isinstance(position, bool) or not isinstance(position, int):
                raise VaultStoreError(codes.VAULT_CLUE_UNKNOWN)
            clue = self.one('SELECT * FROM vault_operation_clues WHERE guild_id=? AND operation_id=? AND stage_number=? AND position=?', (guild_id, operation_id, stage['number'], position - 1))
            if clue is None:
                raise VaultStoreError(codes.VAULT_CLUE_UNKNOWN)
            if clue['revealed_at'] is None:
                self.db.execute('UPDATE vault_operation_clues SET revealed_at=? WHERE id=? AND guild_id=?', (now, clue['id'], guild_id))
                clue = self.one('SELECT * FROM vault_operation_clues WHERE id=? AND guild_id=?', (clue['id'], guild_id))
            return self._vault_clue_entry(clue, self.character_names(guild_id, [clue['character_id']]))

    def set_operation_advisor(self, guild_id, user_id, operation_id, character_id, *, now=None):
        """One favorite per member may advise (never hold a clue); None clears it."""
        now = time.time() if now is None else now
        self._op_load(guild_id, user_id, operation_id, now)
        with self.write_admin():
            row, _ = self._op_active(guild_id, operation_id, now)
            if character_id is not None:
                binding = self.binding(row['channel_id'], None)
                favorites = {f['character_id'] for f in self.eligible_favorites(guild_id, user_id, binding['space_id'])} if binding is not None and binding['guild_id'] == guild_id else set()
                if isinstance(character_id, bool) or not isinstance(character_id, int) or character_id not in favorites:
                    raise VaultStoreError(codes.VAULT_PARTNER_UNAVAILABLE)
            self.db.execute('UPDATE vault_operation_participants SET advisor_id=? WHERE guild_id=? AND operation_id=? AND user_id=?', (character_id, guild_id, operation_id, user_id))
            return self._op_view(row)

    def record_operation_advice(self, guild_id, user_id, operation_id, character_id, *, text=None, usage=None, action_key=None, now=None):
        """One piece of NPC or advisor advice: facts from the public record, spends a team advice cap unit and a talk call, never a turn."""
        now = time.time() if now is None else now
        self._op_load(guild_id, user_id, operation_id, now)
        with self.write_admin():
            done = self._op_replay(guild_id, operation_id, 'advice', action_key)
            if done is not None:
                return {**self._vault_advice_entry(done, self.character_names(guild_id, [done['character_id']]), True), 'user_id': done['user_id']}
            row, stage = self._op_active(guild_id, operation_id, now)
            mine = self.one('SELECT advisor_id FROM vault_operation_participants WHERE guild_id=? AND operation_id=? AND user_id=?', (guild_id, operation_id, user_id))['advisor_id']
            allowed = set(json.loads(row['npc_ids_json'])) | ({mine} if mine else set())
            if isinstance(character_id, bool) or not isinstance(character_id, int) or character_id not in allowed:
                raise VaultStoreError(codes.VAULT_ADVICE_NOT_PARTNER)
            cap = self.vault_settings(guild_id)['advice_cap']
            if len(self._op_attempt(stage, 'advice')) + 1 > cap:
                raise VaultStoreError(codes.VAULT_ADVICE_LIMIT, limit=cap)
            limit = self.game_talk_daily_limit(guild_id)
            if self.game_talk_calls_today(guild_id, now) + 1 > limit:
                raise VaultStoreError(codes.VAULT_ADVICE_TALK_LIMIT, limit=limit)
            style = self.resolved_play_style(guild_id, character_id)
            facts = build_advice_facts(self._op_public(stage), stage['protocol'], stage['length'], risk=style.risk, aggression=style.aggression,
                                       unpredictability=style.unpredictability, seed=f"vault-op-advice/{operation_id}/{stage['number']}/{stage['retry']}/{character_id}/{self._op_used(stage)}")
            source, text = ('llm', text) if check_wording(text, facts) else ('template', None)
            action = self._op_insert(stage, user_id, 'advice', now, action_key, character_id=character_id, guess=facts.guess, remaining=facts.remaining, source=source, text=text)
            self._vault_advice_usage(guild_id, row['channel_id'], usage, now)
            return {**self._vault_advice_entry(action, self.character_names(guild_id, [character_id])), 'user_id': user_id}

    def use_operation_tool(self, guild_id, user_id, operation_id, tool_key, params, *, action_key=None, now=None):
        """One Vault tool for the team: one transaction checks, asks the engine, consumes the CALLER's own inventory unit, records the use and spends shared turns."""
        now = time.time() if now is None else now
        params = dict(params) if isinstance(params, dict) else {}
        self._op_load(guild_id, user_id, operation_id, now)
        with self.write_admin():
            done = self._op_replay(guild_id, operation_id, 'tool', action_key)
            if done is not None:
                return self._op_tool_result(done, True, None)
            row, stage = self._op_active(guild_id, operation_id, now)
            if not isinstance(tool_key, str) or tool_key not in TOOLS:
                raise VaultStoreError(codes.VAULT_TOOL_UNKNOWN, tool=str(tool_key))
            tools = self._op_attempt(stage, 'tool')
            if len(tools) >= MAX_TOOLS:
                raise VaultStoreError(codes.VAULT_TOOL_LIMIT, limit=MAX_TOOLS)
            left = stage['turn_budget'] - self._op_used(stage)
            if not tool_fits(tool_key, left):
                raise VaultStoreError(codes.VAULT_TOOL_NO_TURNS)
            if self.item_qty(guild_id, user_id, tool_key) < 1:
                raise VaultStoreError(codes.VAULT_TOOL_NOT_HELD)
            self._vault_tool_enabled(guild_id, tool_key)
            history = [GuessResult(g['guess'], stage['protocol'], tuple(json.loads(g['feedback_json']))) for g in self._op_attempt(stage, 'guess')]
            history += [c for c in map(self._vault_tool_constraint, tools) if isinstance(c, DigitAt)]
            try:
                found = evaluate_tool(stage['secret'], tool_key, params, history)
            except VaultError as error:
                if error.code != 'nothing_new':
                    raise VaultStoreError(codes.VAULT_TOOL_BAD_PARAMS, reason=error.code) from None
                found = None
            if is_redundant(tool_key, params, filter_candidates(self._op_public(stage), stage['length']), history):
                raise VaultStoreError(codes.VAULT_TOOL_REDUNDANT)
            if found is None:
                found = self._vault_verifier_known(stage['secret'], params['guess'])
            cost = TOOLS[tool_key].turn_cost
            try:
                used = self.consume_item(guild_id, user_id, row['channel_id'], tool_key, now=now, result={'tool': tool_key, 'vault_operation': operation_id, 'turn_cost': cost})
            except NotHeld:
                raise VaultStoreError(codes.VAULT_TOOL_NOT_HELD) from None
            action = self._op_insert(stage, user_id, 'tool', now, action_key, tool_key=tool_key, params_json=json.dumps(params, sort_keys=True), text_key=found.text_key,
                data_json=json.dumps(found.data, sort_keys=True), constraint_kind=type(found.constraint).__name__,
                constraint_json=json.dumps(dataclasses.asdict(found.constraint), sort_keys=True), turn_cost=cost, use_id=used['use_id'])
            settings, owed = self.vault_operation_settings(guild_id), set()
            late = self._op_qualify(row, user_id, now, settings)
            event = self._op_advance(row, stage, False, now, settings, owed)
            return self._op_tool_result(action, False, event, late or user_id in owed)

    def _op_tool_result(self, action, replayed, event, refused=False):
        entry = self._vault_tool_entry(action)
        return {'tool': entry['tool'], 'params': entry['params'], 'text_key': entry['text_key'], 'data': entry['data'], 'turn_cost': entry['turn_cost'],
                'constraint': self._vault_tool_constraint(action), 'qty': self.item_qty(action['guild_id'], action['user_id'], entry['tool']), 'replayed': replayed,
                'event': event, 'reward_refused': refused, 'operation': self._op_view_now(action['guild_id'], action['operation_id'])}

    def operation_cross_validate(self, guild_id, user_id, operation_id, number_a, number_b, *, action_key=None, now=None):
        """Two members' guesses in the current attempt: reveals how many of their shared digits the secret holds; the count becomes public. Costs one shared turn, no item."""
        now = time.time() if now is None else now
        self._op_load(guild_id, user_id, operation_id, now)
        with self.write_admin():
            done = self._op_replay(guild_id, operation_id, 'cross', action_key)
            if done is not None:
                return self._op_cross_result(done, True, None)
            row, stage = self._op_active(guild_id, operation_id, now)
            numbers = (number_a, number_b)
            if any(isinstance(n, bool) or not isinstance(n, int) for n in numbers) or number_a == number_b:
                raise VaultStoreError(codes.VAULT_CROSS_UNKNOWN)
            by_number = {g['number']: g for g in self._op_attempt(stage, 'guess')}
            if number_a not in by_number or number_b not in by_number:
                raise VaultStoreError(codes.VAULT_CROSS_UNKNOWN)
            a, b = by_number[number_a], by_number[number_b]
            if a['user_id'] == b['user_id']:
                raise VaultStoreError(codes.VAULT_CROSS_SAME_MEMBER)
            if self._op_attempt(stage, 'cross'):
                raise VaultStoreError(codes.VAULT_CROSS_USED)
            common = cross_validation_common(a['guess'], b['guess'])
            candidates = filter_candidates(self._op_public(stage), stage['length'])
            if not common or len({len(common & set(c)) for c in candidates}) <= 1:
                raise VaultStoreError(codes.VAULT_CROSS_REDUNDANT)
            if stage['turn_budget'] - self._op_used(stage) < 1:
                raise VaultStoreError(codes.VAULT_TOOL_NO_TURNS)
            digits = ''.join(sorted(common))
            count = len(common & set(stage['secret']))
            action = self._op_insert(stage, user_id, 'cross', now, action_key, turn_cost=1, data_json=json.dumps({'numbers': list(numbers), 'common': list(digits), 'count': count}),
                constraint_kind='CommonCount', constraint_json=json.dumps(dataclasses.asdict(CommonCount(digits, count)), sort_keys=True))
            settings, owed = self.vault_operation_settings(guild_id), set()
            late = self._op_qualify(row, user_id, now, settings)
            event = self._op_advance(row, stage, False, now, settings, owed)
            return self._op_cross_result(action, False, event, late or user_id in owed)

    def _op_cross_result(self, action, replayed, event, refused=False):
        data = json.loads(action['data_json'])
        return {'count': data['count'], 'common': data['common'], 'numbers': tuple(data['numbers']), 'turn_cost': 1, 'replayed': replayed, 'event': event,
                'reward_refused': refused, 'operation': self._op_view_now(action['guild_id'], action['operation_id'])}
