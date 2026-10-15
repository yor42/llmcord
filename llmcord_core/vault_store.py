"""Personal Vault persistence (FEAT-63, D43). Mixed into Store; uses AdminStore's write_admin, ledger and timezone helpers.

One vault per guild, member and server-local day. The secret lives in vault_sessions.secret and leaves the store only in a terminal
view or summary. Guild settings (off by default) sit in vault_settings with a revision. Refusals are VaultStoreError codes; the
Discord edge turns them into words.
A terminal vault shows every clue constraint (the secret is shown then anyway); an active one only the revealed clues.
"""
from __future__ import annotations

import dataclasses
import json
import random
import secrets
import time
from datetime import datetime, time as day_time, timedelta
from zoneinfo import ZoneInfo

from .admin_store import ConflictError
from .i18n import UserError, codes
from .items_store import NotHeld, _recipe
from .vault import (ADVICE_TEMPLATE_KEY, DEFAULT_LENGTH, DEFAULT_TURNS, PROTOCOLS, TOOLS, CountEven, CountLow, DigitAt, DigitPresent, GuessResult, OpenDigit, PairCount,
                    PositionHalf, PositionParity, ToolResult, VaultError, build_advice_facts, check_guess, check_wording, evaluate_tool, feedback, filter_candidates, is_redundant, make_clues,
                    open_digit, played, random_secret, tool_fits)

MAX_PARTNERS = 2
CLUE_COUNT = 2
MAX_TURNS = 30
HISTORY_MAX = 50
CLUE_KINDS = {c.__name__: c for c in (CountEven, CountLow, DigitPresent, PositionHalf, PositionParity)}
TOOL_CONSTRAINTS = {c.__name__: c for c in (DigitAt, DigitPresent, GuessResult, PairCount, PositionHalf)}
MAX_TOOLS = 2
MAX_ADVICE_CAP = 50
CLUE_VOICES = ('template', 'llm')
GUESS_CODES = {'guess_length': codes.VAULT_GUESS_LENGTH, 'guess_not_digits': codes.VAULT_GUESS_NOT_DIGITS, 'guess_repeated_digit': codes.VAULT_GUESS_REPEATED_DIGIT}


class VaultStoreError(UserError):
    """A Vault refusal; the code is one of the vault.* codes and never carries the secret."""


class VaultStore:
    def vault_settings(self, guild_id):
        row = self.one('SELECT * FROM vault_settings WHERE guild_id=?', (guild_id,))
        protocols, turns = ['standard', 'open'], {}
        if row is not None:
            stored = json.loads(row['protocols_json'] or 'null')
            protocols = [p for p in PROTOCOLS if p in stored] if stored else protocols
            turns = {k: v for k, v in json.loads(row['turns_json'] or '{}').items() if k in PROTOCOLS}
        return {'enabled': bool(row['enabled']) if row else False, 'protocols': protocols, 'turns': {**DEFAULT_TURNS, **turns},
                'reward': row['reward'] if row else 0, 'revision': row['revision'] if row else 0,
                'advice_cap': row['advice_cap'] if row else 6, 'clue_voice': row['clue_voice'] if row else 'template'}

    def set_vault_settings(self, guild_id, *, enabled=None, protocols=None, turns=None, reward=None, advice_cap=None, clue_voice=None, expected_revision=None):
        if enabled is not None and not isinstance(enabled, bool):
            raise UserError(codes.VAULT_SETTINGS_FLAG)
        if protocols is not None and (not isinstance(protocols, (list, tuple)) or not protocols or not all(isinstance(p, str) and p in PROTOCOLS for p in protocols)):
            raise UserError(codes.VAULT_SETTINGS_PROTOCOLS)
        if turns is not None and (not isinstance(turns, dict) or any(k not in PROTOCOLS or isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= MAX_TURNS for k, v in turns.items())):
            raise UserError(codes.VAULT_SETTINGS_TURNS, low=1, high=MAX_TURNS)
        if reward is not None and (isinstance(reward, bool) or not isinstance(reward, int) or not 0 <= reward <= self.MAX_CURRENCY_CHANGE):
            raise UserError(codes.VAULT_SETTINGS_REWARD, limit=self.MAX_CURRENCY_CHANGE)
        if advice_cap is not None and (isinstance(advice_cap, bool) or not isinstance(advice_cap, int) or not 0 <= advice_cap <= MAX_ADVICE_CAP):
            raise UserError(codes.VAULT_SETTINGS_ADVICE_CAP, limit=MAX_ADVICE_CAP)
        if clue_voice is not None and (not isinstance(clue_voice, str) or clue_voice not in CLUE_VOICES):
            raise UserError(codes.VAULT_SETTINGS_CLUE_VOICE)
        with self.write_admin():
            current = self.vault_settings(guild_id)
            row = self.one('SELECT turns_json FROM vault_settings WHERE guild_id=?', (guild_id,))
            overrides = json.loads(row['turns_json'] or '{}') if row else {}
            if expected_revision is not None and current['revision'] != expected_revision:
                raise ConflictError(codes.VAULT_SETTINGS_CHANGED)
            new = {'enabled': current['enabled'] if enabled is None else enabled,
                   'protocols': current['protocols'] if protocols is None else [p for p in PROTOCOLS if p in protocols],
                   'turns': overrides if turns is None else {**overrides, **turns},
                   'reward': current['reward'] if reward is None else reward, 'revision': current['revision'] + 1,
                   'advice_cap': current['advice_cap'] if advice_cap is None else advice_cap, 'clue_voice': current['clue_voice'] if clue_voice is None else clue_voice}
            self.db.execute('INSERT INTO vault_settings(guild_id,enabled,protocols_json,turns_json,reward,revision,updated_at,advice_cap,clue_voice) VALUES(?,?,?,?,?,?,?,?,?) '
                'ON CONFLICT(guild_id) DO UPDATE SET enabled=excluded.enabled,protocols_json=excluded.protocols_json,turns_json=excluded.turns_json,'
                'reward=excluded.reward,revision=excluded.revision,updated_at=excluded.updated_at,advice_cap=excluded.advice_cap,clue_voice=excluded.clue_voice',
                (guild_id, int(new['enabled']), json.dumps(new['protocols']), json.dumps(new['turns'], sort_keys=True), new['reward'], new['revision'], time.time(),
                 new['advice_cap'], new['clue_voice']))
        return self.vault_settings(guild_id)

    def _vault_zone(self, guild_id):
        try:
            return ZoneInfo(self.guild_timezone(guild_id) or 'UTC')
        except Exception:
            return ZoneInfo('UTC')

    def _vault_day(self, guild_id, now):
        """(ISO local date, Unix time of the next local midnight) in the server timezone at `now`."""
        zone = self._vault_zone(guild_id)
        day = datetime.fromtimestamp(now, zone).date()
        return day.isoformat(), datetime.combine(day + timedelta(days=1), day_time(0), zone).timestamp()

    def _vault_finish(self, guild_id, session_id, status, now):
        self.db.execute("UPDATE vault_sessions SET status=?,finished_at=? WHERE id=? AND guild_id=? AND status='active'", (status, now, session_id, guild_id))
        self.db.execute("UPDATE vault_stages SET status=? WHERE session_id=? AND guild_id=? AND status='active'", (status, session_id, guild_id))

    def _vault_expire(self, guild_id, now, user_id=None):
        """Inside write_admin: marks the guild's (or one member's) active vaults past their expiry; returns the count."""
        who, args = ('', ()) if user_id is None else (' AND user_id=?', (user_id,))
        ids = [r['id'] for r in self.all(f"SELECT id FROM vault_sessions WHERE guild_id=? AND status='active' AND expires_at<=?{who}", (guild_id, now, *args))]
        for session_id in ids:
            self._vault_finish(guild_id, session_id, 'expired', now)
        return len(ids)

    def expire_vault_sessions(self, guild_id, now=None):
        now = time.time() if now is None else now
        with self.write_admin():
            return self._vault_expire(guild_id, now)

    def _vault_row(self, guild_id, user_id, session_id, now):
        """The member's own session row (None for anyone else's or another server's), with a lazy expiry committed first."""
        row = self.one('SELECT * FROM vault_sessions WHERE id=? AND guild_id=? AND user_id=?', (session_id, guild_id, user_id))
        if row is not None and row['status'] == 'active' and row['expires_at'] <= now:
            with self.write_admin():
                self._vault_finish(guild_id, session_id, 'expired', now)
            row = self.one('SELECT * FROM vault_sessions WHERE id=? AND guild_id=?', (session_id, guild_id))
        return row

    def _vault_view(self, row, created=False, refused=False):
        """Inside or outside a transaction: the public view of a session row. The secret only once it is no longer active."""
        gid, sid = row['guild_id'], row['id']
        partner_ids = json.loads(row['partners_json'])
        clue_rows = self.all('SELECT * FROM vault_clues WHERE guild_id=? AND session_id=? ORDER BY position', (gid, sid))
        actions = self.all('SELECT * FROM vault_actions WHERE guild_id=? AND session_id=? ORDER BY number', (gid, sid))
        advice_rows = self.all('SELECT * FROM vault_advice WHERE guild_id=? AND session_id=? ORDER BY id', (gid, sid))
        names = self.character_names(gid, partner_ids + [c['character_id'] for c in clue_rows] + [a['character_id'] for a in advice_rows])

        clues = [self._vault_clue_entry(c, names, row['status'] != 'active') for c in clue_rows]
        tool_rows = self.all('SELECT * FROM vault_tool_uses WHERE guild_id=? AND session_id=? ORDER BY id', (gid, sid))
        guesses = [{'number': a['number'], 'guess': a['guess'], 'feedback': tuple(json.loads(a['feedback_json'])), 'won': bool(a['won'])} for a in actions]
        constraints = ([OpenDigit(row['open_digit'])] if row['open_digit'] else []) + [c['constraint'] for c in clues if c['revealed'] and 'constraint' in c]
        for g in guesses:
            constraints += played(row['protocol'], g['guess'], g['feedback'], g['won'])
        constraints += [self._vault_tool_constraint(t) for t in tool_rows]
        used = len(guesses) + sum(t['turn_cost'] for t in tool_rows)
        claim = self.one("SELECT amount,ledger_id FROM vault_reward_claims WHERE guild_id=? AND session_id=? AND user_id=? AND type='win'", (gid, sid, row['user_id']))
        reversed_ = bool(claim and self.one('SELECT 1 FROM currency_ledger WHERE guild_id=? AND reverses_id=?', (gid, claim['ledger_id'])))
        return {'id': sid, 'guild_id': gid, 'user_id': row['user_id'], 'channel_id': row['channel_id'], 'local_date': row['local_date'],
                'protocol': row['protocol'], 'length': row['length'], 'status': row['status'], 'turn_budget': row['turn_budget'],
                'turns_used': used, 'turns_left': max(row['turn_budget'] - used, 0), 'created_at': row['created_at'],
                'expires_at': row['expires_at'], 'created': created, 'open_digit': row['open_digit'],
                'partners': [{'character_id': i, 'name': names.get(i)} for i in partner_ids], 'clues': clues, 'guesses': guesses,
                'advice': [self._vault_advice_entry(a, names) for a in advice_rows],
                'tools': [self._vault_tool_entry(t) for t in tool_rows], 'tools_used': len(tool_rows), 'constraints': constraints, 'secret': row['secret'] if row['status'] != 'active' else None,
                'reward': {'amount': claim['amount'], 'ledger_id': claim['ledger_id']} if claim else None,
                'reward_reversed': reversed_, 'reward_refused': refused}

    @staticmethod
    def _vault_tool_constraint(t):
        data = json.loads(t['constraint_json'])
        if t['constraint_kind'] == 'GuessResult':
            data['result'] = tuple(data['result'])
        return TOOL_CONSTRAINTS[t['constraint_kind']](**data)

    @staticmethod
    def _vault_tool_entry(t):
        return {'tool': t['tool_key'], 'params': json.loads(t['params_json']), 'text_key': t['text_key'], 'data': json.loads(t['data_json']), 'turn_cost': t['turn_cost']}

    @staticmethod
    def _vault_clue_entry(c, names, terminal=False):
        """A clue as shown: its constraint only once revealed, or once the vault has ended (the secret is shown then anyway)."""
        entry = {'position': c['position'] + 1, 'character_id': c['character_id'], 'name': names.get(c['character_id']),
                 'revealed': c['revealed_at'] is not None, 'revealed_at': c['revealed_at']}
        if entry['revealed'] or terminal:
            entry['constraint'] = CLUE_KINDS[c['kind']](**json.loads(c['data_json']))
        return entry

    @staticmethod
    def _vault_advice_entry(a, names, replayed=False):
        llm = a['source'] == 'llm'
        return {'id': a['id'], 'character_id': a['character_id'], 'name': names.get(a['character_id']), 'guess': a['guess'], 'remaining': a['remaining'],
                'source': a['source'], 'text': a['text'] if llm else None, 'text_key': None if llm else ADVICE_TEMPLATE_KEY,
                'data': {'guess': a['guess'], 'remaining': a['remaining']}, 'created_at': a['created_at'], 'replayed': replayed}

    def _vault_advice_usage(self, guild_id, channel_id, records, now):
        """The one model_usage row of a delivered advice: the real call's record when the model was asked, else a placeholder for the template."""
        rec = next((r for r in records or () if hasattr(r, 'profile')), None)
        if rec is None:
            self.db.execute('INSERT INTO model_usage(guild_id,profile,model,role,cached_tokens,reasoning_tokens,cost_basis,created_at,channel_id,feature) VALUES(?,?,?,?,0,0,?,?,?,?)',
                            (guild_id, 'vault', 'advice', 'dialogue', 'none', now, channel_id, 'game'))
            return
        self.db.execute('INSERT INTO model_usage(guild_id,profile,model,role,input_tokens,output_tokens,cached_tokens,reasoning_tokens,cost_usd,cost_basis,created_at,channel_id,feature) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                        (guild_id, rec.profile, rec.model, rec.role, rec.input_tokens, rec.output_tokens, rec.cached_tokens, rec.reasoning_tokens, rec.cost_usd, rec.cost_basis, now, channel_id, 'game'))

    def vault_advice_replayed(self, guild_id, session_id, action_key, *, meeting=False):
        """True when this interaction already produced advice (the store owns the action_key format)."""
        pattern = f'meeting:{action_key}:%' if meeting else f'ask:{action_key}'
        return self.one('SELECT 1 FROM vault_advice WHERE guild_id=? AND session_id=? AND action_key LIKE ?', (guild_id, session_id, pattern)) is not None

    def record_vault_advice(self, guild_id, user_id, session_id, character_id, *, text=None, usage=None, action_key=None, now=None):
        """One piece of partner advice: facts from public constraints, the model's wording kept only if it passes check_wording.
        Spends one advice cap unit and one shared talk call atomically; a replayed action_key returns the first advice and spends nothing."""
        now = time.time() if now is None else now
        key = None if action_key is None else f'ask:{action_key}'
        with self.write_admin():
            row = self.one('SELECT * FROM vault_sessions WHERE id=? AND guild_id=? AND user_id=?', (session_id, guild_id, user_id))
            if row is None:
                raise VaultStoreError(codes.VAULT_NOT_FOUND)
            if key is not None:
                done = self.one('SELECT * FROM vault_advice WHERE guild_id=? AND session_id=? AND action_key=?', (guild_id, session_id, key))
                if done is not None:
                    return self._vault_advice_entry(done, self.character_names(guild_id, [done['character_id']]), True)
            return self._vault_advice_give(row, [character_id], {character_id: text}, [key], now, usage={character_id: usage})[0]

    def record_vault_meeting(self, guild_id, user_id, session_id, *, texts=None, usage=None, action_key=None, now=None):
        """Strategy meeting: advice from every chosen partner (at most two), all or nothing, one cap unit and one talk call each."""
        now = time.time() if now is None else now
        texts = texts if isinstance(texts, dict) else {}
        with self.write_admin():
            row = self.one('SELECT * FROM vault_sessions WHERE id=? AND guild_id=? AND user_id=?', (session_id, guild_id, user_id))
            if row is None:
                raise VaultStoreError(codes.VAULT_NOT_FOUND)
            partners = json.loads(row['partners_json'])
            keys = [None if action_key is None else f'meeting:{action_key}:{cid}' for cid in partners]
            if action_key is not None and partners:
                marks = ','.join('?' * len(keys))
                done = self.all(f'SELECT * FROM vault_advice WHERE guild_id=? AND session_id=? AND action_key IN ({marks}) ORDER BY id', (guild_id, session_id, *keys))
                if done:
                    names = self.character_names(guild_id, [d['character_id'] for d in done])
                    return [self._vault_advice_entry(d, names, True) for d in done]
            return self._vault_advice_give(row, partners, texts, keys, now, meeting=True, usage=usage)

    def _vault_advice_give(self, row, character_ids, texts, keys, now, meeting=False, usage=None):
        """Inside write_admin: checks in order, then stores advice and spends cap and talk for every asked partner or for none."""
        guild_id, session_id = row['guild_id'], row['id']
        if row['status'] != 'active' or row['expires_at'] <= now:
            raise VaultStoreError(codes.VAULT_NOT_ACTIVE)
        partners = json.loads(row['partners_json'])
        if not character_ids or any(isinstance(c, bool) or not isinstance(c, int) or c not in partners for c in character_ids):
            raise VaultStoreError(codes.VAULT_ADVICE_NOT_PARTNER)
        cap = self.vault_settings(guild_id)['advice_cap']
        used = self.one('SELECT COUNT(*) FROM vault_advice WHERE guild_id=? AND session_id=?', (guild_id, session_id))[0]
        if used + len(character_ids) > cap:
            raise VaultStoreError(codes.VAULT_ADVICE_LIMIT, limit=cap)
        limit = self.game_talk_daily_limit(guild_id)
        if self.game_talk_calls_today(guild_id, now) + len(character_ids) > limit:
            raise VaultStoreError(codes.VAULT_ADVICE_TALK_LIMIT, limit=limit)
        view = self._vault_view(row)
        out = []
        for cid, key in zip(character_ids, keys):
            style = self.resolved_play_style(guild_id, cid)
            facts = build_advice_facts(view['constraints'], view['protocol'], view['length'], risk=style.risk, aggression=style.aggression,
                                       unpredictability=style.unpredictability, seed=f"vault-advice/{session_id}/{cid}/{view['turns_used']}")
            text = texts.get(cid)
            source, text = ('llm', text) if check_wording(text, facts) else ('template', None)
            advice_id = self.db.execute('INSERT INTO vault_advice(guild_id,session_id,user_id,character_id,guess,remaining,source,text,action_key,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)',
                (guild_id, session_id, row['user_id'], cid, facts.guess, facts.remaining, source, text, key, now)).lastrowid
            self._vault_advice_usage(guild_id, row['channel_id'], (usage or {}).get(cid), now)
            out.append(self._vault_advice_entry(self.one('SELECT * FROM vault_advice WHERE id=? AND guild_id=?', (advice_id, guild_id)), self.character_names(guild_id, [cid])))
        return out

    def reveal_vault_clue(self, guild_id, user_id, session_id, position, *, now=None):
        """Shows one partner clue (1-based). Free and idempotent: the first call stores revealed_at, later calls keep it."""
        now = time.time() if now is None else now
        row = self._vault_row(guild_id, user_id, session_id, now)
        if row is None:
            raise VaultStoreError(codes.VAULT_NOT_FOUND)
        if row['status'] != 'active':
            raise VaultStoreError(codes.VAULT_NOT_ACTIVE)
        if isinstance(position, bool) or not isinstance(position, int):
            raise VaultStoreError(codes.VAULT_CLUE_UNKNOWN)
        with self.write_admin():
            fresh = self.one('SELECT status,expires_at FROM vault_sessions WHERE id=? AND guild_id=? AND user_id=?', (session_id, guild_id, user_id))
            if fresh is None:
                raise VaultStoreError(codes.VAULT_NOT_FOUND)
            lapsed = fresh['status'] == 'active' and fresh['expires_at'] <= now
            if lapsed:
                self._vault_finish(guild_id, session_id, 'expired', now)
            if fresh['status'] == 'active' and not lapsed:
                clue = self.one('SELECT * FROM vault_clues WHERE guild_id=? AND session_id=? AND position=?', (guild_id, session_id, position - 1))
                if clue is None:
                    raise VaultStoreError(codes.VAULT_CLUE_UNKNOWN)
                if clue['revealed_at'] is None:
                    self.db.execute('UPDATE vault_clues SET revealed_at=? WHERE id=? AND guild_id=?', (now, clue['id'], guild_id))
                    clue = self.one('SELECT * FROM vault_clues WHERE id=? AND guild_id=?', (clue['id'], guild_id))
                return self._vault_clue_entry(clue, self.character_names(guild_id, [clue['character_id']]))
        raise VaultStoreError(codes.VAULT_NOT_ACTIVE)

    def vault_session(self, guild_id, user_id, session_id, *, now=None):
        now = time.time() if now is None else now
        row = self._vault_row(guild_id, user_id, session_id, now)
        if row is None:
            raise VaultStoreError(codes.VAULT_NOT_FOUND)
        return self._vault_view(row)

    def vault_partner_choices(self, guild_id, user_id, channel_id, parent_id=None):
        binding = self.binding(channel_id, parent_id)
        if binding is None or binding['guild_id'] != guild_id:
            return []
        out = {f['character_id']: {'character_id': f['character_id'], 'name': f['name'], 'source': 'favorite'}
               for f in self.eligible_favorites(guild_id, user_id, binding['space_id'])}
        cast = set(self.get_cast(channel_id, parent_id))
        for c in self.eligible_characters(guild_id, binding['space_id']):
            if c['id'] in cast and c['id'] not in out:
                out[c['id']] = {'character_id': c['id'], 'name': c['name'], 'source': 'cast'}
        return list(out.values())

    def open_personal_vault(self, guild_id, user_id, channel_id, *, protocol='standard', partner_ids=(), now=None, seed=None, parent_id=None):
        """Starts or resumes the member's vault for the server day (one per local date: a failed or solved one is returned again, never reset)."""
        now = time.time() if now is None else now
        partner_ids = list(partner_ids)
        with self.write_admin():
            self._vault_expire(guild_id, now, user_id)
        with self.write_admin():
            current = self.one('SELECT * FROM vault_sessions WHERE guild_id=? AND user_id=? AND expires_at>? ORDER BY id DESC LIMIT 1', (guild_id, user_id, now))
            if current is not None:
                return self._vault_view(current)
            local_date, expires_at = self._vault_day(guild_id, now)
            clash = self.one('SELECT * FROM vault_sessions WHERE guild_id=? AND user_id=? AND local_date=?', (guild_id, user_id, local_date))
            if clash is not None:  # the timezone moved back onto a date already used
                return self._vault_view(clash)
            settings = self.vault_settings(guild_id)
            if not settings['enabled']:
                raise VaultStoreError(codes.VAULT_DISABLED)
            if protocol not in settings['protocols']:
                raise VaultStoreError(codes.VAULT_PROTOCOL_NOT_ALLOWED)
            if any(isinstance(i, bool) or not isinstance(i, int) for i in partner_ids):
                raise VaultStoreError(codes.VAULT_PARTNER_UNAVAILABLE)
            partners = list(dict.fromkeys(partner_ids))
            if len(partners) > MAX_PARTNERS:
                raise VaultStoreError(codes.VAULT_PARTNERS_TOO_MANY, limit=MAX_PARTNERS)
            if partners and not set(partners) <= {c['character_id'] for c in self.vault_partner_choices(guild_id, user_id, channel_id, parent_id)}:
                raise VaultStoreError(codes.VAULT_PARTNER_UNAVAILABLE)
            rng = random.Random(seed) if seed is not None else secrets.SystemRandom()
            code = random_secret(DEFAULT_LENGTH, rng)
            shown = open_digit(code, rng) if protocol == 'open' else None
            clues = make_clues(code, CLUE_COUNT, rng) if partners else []
            budget = settings['turns'][protocol]
            sid = self.db.execute('INSERT INTO vault_sessions(guild_id,user_id,channel_id,local_date,protocol,length,turn_budget,secret,open_digit,partners_json,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (guild_id, user_id, channel_id, local_date, protocol, DEFAULT_LENGTH, budget, code, shown, json.dumps(partners), now, expires_at)).lastrowid
            self.db.execute('INSERT INTO vault_stages(guild_id,session_id,number,turn_budget,created_at) VALUES(?,?,?,?,?)', (guild_id, sid, 1, budget, now))
            for i, clue in enumerate(clues):
                self.db.execute('INSERT INTO vault_clues(guild_id,session_id,position,character_id,kind,data_json) VALUES(?,?,?,?,?,?)',
                    (guild_id, sid, i, partners[i % len(partners)], type(clue).__name__, json.dumps(dataclasses.asdict(clue), sort_keys=True)))
            return self._vault_view(self.one('SELECT * FROM vault_sessions WHERE id=? AND guild_id=?', (sid, guild_id)), created=True)

    def submit_vault_guess(self, guild_id, user_id, session_id, guess, *, action_key=None, now=None):
        now = time.time() if now is None else now
        with self.write_admin():
            row = self.one('SELECT * FROM vault_sessions WHERE id=? AND guild_id=? AND user_id=?', (session_id, guild_id, user_id))
            if row is None:
                raise VaultStoreError(codes.VAULT_NOT_FOUND)
            if action_key is not None:
                done = self.one('SELECT * FROM vault_actions WHERE guild_id=? AND session_id=? AND action_key=?', (guild_id, session_id, action_key))
                if done is not None:
                    return self._vault_result(row, done, True)
            if row['status'] == 'active' and row['expires_at'] <= now:
                self._vault_finish(guild_id, session_id, 'expired', now)
            elif row['status'] == 'active':
                refused = False
                code = self._vault_valid_guess(row, guess)
                if self.one('SELECT 1 FROM vault_actions WHERE guild_id=? AND session_id=? AND guess=?', (guild_id, session_id, code)):
                    raise VaultStoreError(codes.VAULT_GUESS_DUPLICATE)
                number = self.one('SELECT COUNT(*) FROM vault_actions WHERE session_id=? AND guild_id=?', (session_id, guild_id))[0] + 1
                spent = number + self._vault_tool_turns(session_id, guild_id)
                result = feedback(row['protocol'], row['secret'], code)
                won = code == row['secret']
                stage = self.one('SELECT id FROM vault_stages WHERE session_id=? AND guild_id=? ORDER BY number LIMIT 1', (session_id, guild_id))
                action_id = self.db.execute('INSERT INTO vault_actions(guild_id,session_id,stage_id,number,user_id,guess,feedback_json,won,action_key,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)',
                    (guild_id, session_id, stage['id'], number, user_id, code, json.dumps(list(result)), int(won), action_key, now)).lastrowid
                if won:
                    self._vault_finish(guild_id, session_id, 'solved', now)
                    refused = not self._vault_pay(row, now) and self.vault_settings(guild_id)['reward'] > 0
                elif spent >= row['turn_budget']:
                    self._vault_finish(guild_id, session_id, 'failed', now)
                return self._vault_result(self.one('SELECT * FROM vault_sessions WHERE id=? AND guild_id=?', (session_id, guild_id)), self.one('SELECT * FROM vault_actions WHERE id=? AND guild_id=?', (action_id, guild_id)), False, refused)
        raise VaultStoreError(codes.VAULT_NOT_ACTIVE)

    @staticmethod
    def _vault_valid_guess(row, guess):
        try:
            return check_guess(guess if isinstance(guess, str) else '', row['length'])
        except VaultError as error:
            raise VaultStoreError(GUESS_CODES[error.code], **error.params) from None

    def _vault_tool_turns(self, session_id, guild_id):
        return self.one('SELECT COALESCE(SUM(turn_cost),0) FROM vault_tool_uses WHERE session_id=? AND guild_id=?', (session_id, guild_id))[0]

    def use_vault_tool(self, guild_id, user_id, session_id, tool_key, params, *, action_key=None, now=None):
        """Activates one Vault tool: one transaction checks, asks the engine, consumes one inventory unit, records the use and spends the turns.
        Every refusal is raised before anything is written. A replayed action_key returns the first result and spends nothing."""
        now = time.time() if now is None else now
        params = dict(params) if isinstance(params, dict) else {}
        with self.write_admin():
            row = self.one('SELECT * FROM vault_sessions WHERE id=? AND guild_id=? AND user_id=?', (session_id, guild_id, user_id))
            if row is None:
                raise VaultStoreError(codes.VAULT_NOT_FOUND)
            if action_key is not None:
                done = self.one('SELECT * FROM vault_tool_uses WHERE guild_id=? AND session_id=? AND action_key=?', (guild_id, session_id, action_key))
                if done is not None:
                    return self._vault_tool_result(row, done, True)
            if row['status'] == 'active' and row['expires_at'] <= now:
                self._vault_finish(guild_id, session_id, 'expired', now)
            elif row['status'] == 'active':
                return self._vault_tool_apply(row, tool_key, params, action_key, now)
        raise VaultStoreError(codes.VAULT_NOT_ACTIVE)

    def _vault_tool_apply(self, row, tool_key, params, action_key, now):
        guild_id, session_id, user_id = row['guild_id'], row['id'], row['user_id']
        tools = self.all('SELECT * FROM vault_tool_uses WHERE guild_id=? AND session_id=? ORDER BY id', (guild_id, session_id))
        guesses = self.all('SELECT guess,feedback_json FROM vault_actions WHERE guild_id=? AND session_id=? ORDER BY number', (guild_id, session_id))
        left = row['turn_budget'] - len(guesses) - sum(t['turn_cost'] for t in tools)
        found = self._vault_tool_check(guild_id, user_id, row['secret'], row['protocol'], row['length'], tool_key, params, tools, guesses, left, lambda: self._vault_view(row)['constraints'])
        cost = TOOLS[tool_key].turn_cost
        try:
            used = self.consume_item(guild_id, user_id, row['channel_id'], tool_key, now=now, result={'tool': tool_key, 'vault_session': session_id, 'turn_cost': cost})
        except NotHeld:
            raise VaultStoreError(codes.VAULT_TOOL_NOT_HELD) from None
        stage = self.one('SELECT id FROM vault_stages WHERE session_id=? AND guild_id=? ORDER BY number LIMIT 1', (session_id, guild_id))
        constraint = dataclasses.asdict(found.constraint)
        action_id = self.db.execute('INSERT INTO vault_tool_uses(guild_id,session_id,stage_id,user_id,tool_key,params_json,text_key,data_json,constraint_kind,constraint_json,turn_cost,use_id,action_key,created_at) '
            'VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (guild_id, session_id, stage['id'], user_id, tool_key, json.dumps(params, sort_keys=True), found.text_key,
            json.dumps(found.data, sort_keys=True), type(found.constraint).__name__, json.dumps(constraint, sort_keys=True), cost, used['use_id'], action_key, now)).lastrowid
        if left - cost <= 0:
            self._vault_finish(guild_id, session_id, 'failed', now)
        return self._vault_tool_result(self.one('SELECT * FROM vault_sessions WHERE id=? AND guild_id=?', (session_id, guild_id)),
                                       self.one('SELECT * FROM vault_tool_uses WHERE id=? AND guild_id=?', (action_id, guild_id)), False)

    def _vault_tool_check(self, guild_id, user_id, secret, protocol, length, tool_key, params, tools, guesses, left, public):
        """Shared tool pipeline for a session or an operation stage: every refusal, then the engine's answer. `public` is called only once the engine has answered."""
        if not isinstance(tool_key, str) or tool_key not in TOOLS:
            raise VaultStoreError(codes.VAULT_TOOL_UNKNOWN, tool=str(tool_key))
        if len(tools) >= MAX_TOOLS:
            raise VaultStoreError(codes.VAULT_TOOL_LIMIT, limit=MAX_TOOLS)
        if not tool_fits(tool_key, left):
            raise VaultStoreError(codes.VAULT_TOOL_NO_TURNS)
        if self.item_qty(guild_id, user_id, tool_key) < 1:
            raise VaultStoreError(codes.VAULT_TOOL_NOT_HELD)
        self._vault_tool_enabled(guild_id, tool_key)
        history = [GuessResult(g['guess'], protocol, tuple(json.loads(g['feedback_json']))) for g in guesses]
        history += [c for c in map(self._vault_tool_constraint, tools) if isinstance(c, DigitAt)]
        try:
            found = evaluate_tool(secret, tool_key, params, history)
        except VaultError as error:
            if error.code != 'nothing_new':
                raise VaultStoreError(codes.VAULT_TOOL_BAD_PARAMS, reason=error.code) from None
            found = None
        if is_redundant(tool_key, params, filter_candidates(public(), length), history):
            raise VaultStoreError(codes.VAULT_TOOL_REDUNDANT)
        return found if found is not None else self._vault_verifier_known(secret, params['guess'])

    def _vault_tool_enabled(self, guild_id, tool_key):
        """The shop's Allow use switch also governs Vault tools; inside the caller's transaction, before anything is consumed."""
        recipe = _recipe(tool_key)
        if recipe is not None and not self._shop_entry(recipe, self._shop_row(guild_id, tool_key))['allow_use']:
            raise VaultStoreError(codes.VAULT_TOOL_DISABLED)

    @staticmethod
    def _vault_verifier_known(secret, guess):
        """The verifier's paid answer when every strike is already disclosed: a secret-dependent refusal would be a free leak."""
        pos = next(i + 1 for i, (s, g) in enumerate(zip(secret, guess)) if s == g)
        return ToolResult(DigitAt(pos, secret[pos - 1]), 'vault_verifier_position', {'guess': guess, 'position': pos, 'digit': secret[pos - 1]})

    def _vault_tool_result(self, row, use, replayed):
        entry = self._vault_tool_entry(use)
        return {'tool': entry['tool'], 'params': entry['params'], 'text_key': entry['text_key'], 'data': entry['data'], 'turn_cost': entry['turn_cost'],
                'constraint': self._vault_tool_constraint(use), 'qty': self.item_qty(row['guild_id'], row['user_id'], entry['tool']), 'replayed': replayed,
                'session': self._vault_view(row)}

    def _vault_pay(self, row, now):
        """Inside write_admin: the win reward in force now, once per (session, member, type). A balance cap that refuses it leaves the win unpaid."""
        amount = self.vault_settings(row['guild_id'])['reward']
        if amount <= 0:
            return False
        try:
            entry = self._append_ledger(row['guild_id'], row['user_id'], amount, f"Vault reward ({row['local_date']})", row['user_id'], 'vault', None)
        except UserError:
            return False
        self.db.execute("INSERT INTO vault_reward_claims(guild_id,session_id,user_id,type,amount,ledger_id,created_at) VALUES(?,?,?,'win',?,?,?)",
            (row['guild_id'], row['id'], row['user_id'], amount, entry['id'], now))
        return True

    def _vault_result(self, row, action, replayed, refused=False):
        return {'number': action['number'], 'guess': action['guess'], 'feedback': tuple(json.loads(action['feedback_json'])), 'won': bool(action['won']),
                'replayed': replayed, 'reward_refused': refused, 'session': self._vault_view(row, refused=refused)}

    def vault_history(self, guild_id, user_id, *, limit=10, now=None):
        now = time.time() if now is None else now
        limit = max(1, min(int(limit), HISTORY_MAX))
        with self.write_admin():
            self._vault_expire(guild_id, now, user_id)
        return [{'id': r['id'], 'local_date': r['local_date'], 'protocol': r['protocol'], 'status': r['status'],
                 'turns_used': r['used'], 'turn_budget': r['turn_budget'], 'secret': r['secret'] if r['status'] != 'active' else None,
                 'reward': r['reward'], 'reward_reversed': bool(r['reversed'])} for r in self.all(
            'SELECT s.id,s.local_date,s.protocol,s.status,s.turn_budget,s.secret,'
            '(SELECT COUNT(*) FROM vault_actions a WHERE a.session_id=s.id AND a.guild_id=s.guild_id)'
            '+(SELECT COALESCE(SUM(u.turn_cost),0) FROM vault_tool_uses u WHERE u.session_id=s.id AND u.guild_id=s.guild_id) AS used,'
            "COALESCE((SELECT c.amount FROM vault_reward_claims c WHERE c.session_id=s.id AND c.guild_id=s.guild_id AND c.user_id=s.user_id AND c.type='win'),0) AS reward, "
            "EXISTS(SELECT 1 FROM vault_reward_claims c JOIN currency_ledger l ON l.guild_id=c.guild_id AND l.reverses_id=c.ledger_id WHERE c.session_id=s.id AND c.guild_id=s.guild_id AND c.user_id=s.user_id AND c.type='win') AS reversed "
            'FROM vault_sessions s WHERE s.guild_id=? AND s.user_id=? ORDER BY s.local_date DESC,s.id DESC LIMIT ?', (guild_id, user_id, limit))]
