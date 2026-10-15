"""Vault support missions persistence (FEAT-67, D43). Mixed into Store next to VaultOperationStore, whose helpers it reuses.

A member of an operation may do one support mission per operation: record forensics or 3-digit Bulls and Cows. The puzzle and its answer are
stored at creation and never regenerated; the answer leaves the store only once the mission is solved or failed. A solved mission counts as one
contribution and may pay one support reward through the ledger (a claim of type 'support'; it never pays the main contribution reward but counts toward the team reward's min_guesses). Missions spend no shared stage turn and never touch
a stage secret. The Store shares one sqlite connection (check_same_thread=False) and write_admin only locks other write_admin callers, so store calls must not run on a background thread. The Discord layer defers the interaction, generates the forensics puzzle (pure, about 0.2 s) with asyncio.to_thread, then calls start_support_mission on the loop with it (puzzle=); the store re-checks it with the solver. Refusals are VaultStoreError codes; the Discord edge turns them into words.
"""
from __future__ import annotations

import json
import random
import secrets
import time

from .i18n import codes
from .vault import (BULLS_COWS_LENGTH, BULLS_COWS_TURNS, FORENSICS_ATTEMPTS, MISSION_TYPES, VaultError, bulls_cows_feedback, bulls_cows_secret,
                    check_bulls_cows_guess, forensics_answers, generate_forensics, valid_forensics_answer)
from .vault.missions import FORENSICS_LENGTH, FORENSICS_LINES
from .vault_operation_store import VaultOperationStore
from .vault_store import GUESS_CODES, VaultStoreError

BUDGETS = {'forensics': FORENSICS_ATTEMPTS, 'bulls_cows': BULLS_COWS_TURNS}


class VaultSupportStore(VaultOperationStore):
    def _support_row(self, guild_id, operation_id, user_id):
        return self.one('SELECT * FROM vault_support_missions WHERE guild_id=? AND operation_id=? AND user_id=?', (guild_id, operation_id, user_id))

    def _support_view(self, row, created=False):
        kind, ended = row['mission_type'], row['status'] != 'active'
        puzzle, answer = json.loads(row['puzzle_json']), json.loads(row['answer_json'])
        attempts = self.all('SELECT * FROM vault_support_attempts WHERE guild_id=? AND mission_id=? ORDER BY number', (row['guild_id'], row['id']))
        history = []
        for a in attempts:
            given, found = json.loads(a['answer_json']), json.loads(a['feedback_json']) if a['feedback_json'] else None
            history.append({'number': a['number'], 'answer': given if kind == 'bulls_cows' else tuple(given),
                            'feedback': tuple(found) if found is not None else None, 'correct': bool(a['correct'])})
        if kind == 'forensics':
            puzzle = {'records': [{'guess': g, 'result': tuple(r) if r is not None else None} for g, r in puzzle['records']], 'hidden': puzzle['hidden']}
        shown = (answer if kind == 'bulls_cows' else tuple(answer)) if ended else None
        budget = BUDGETS[kind]
        return {'id': row['id'], 'operation_id': row['operation_id'], 'guild_id': row['guild_id'], 'user_id': row['user_id'], 'type': kind, 'status': row['status'],
                'created': created, 'turn_budget': budget, 'attempts_used': row['attempts_used'], 'attempts_left': max(budget - row['attempts_used'], 0),
                'puzzle': puzzle, 'history': history, 'answer': shown, 'reward': self._support_reward(row)}

    def _support_reward(self, row):
        claim = self.one("SELECT * FROM vault_operation_claims WHERE guild_id=? AND operation_id=? AND user_id=? AND type='support'",
                         (row['guild_id'], row['operation_id'], row['user_id']))
        if claim is None:
            return None
        undone = self.one('SELECT 1 FROM currency_ledger WHERE guild_id=? AND reverses_id=?', (row['guild_id'], claim['ledger_id'])) is not None
        return {'amount': claim['amount'], 'ledger_id': claim['ledger_id'], 'reversed': undone}

    @staticmethod
    def _puzzle_holds(puzzle):
        try:
            records = tuple((g, None if r is None else tuple(r)) for g, r in puzzle.records)
            return (len(records) == FORENSICS_LINES and 0 <= puzzle.hidden < len(records) and records[puzzle.hidden][1] is None
                    and all(isinstance(g, str) and len(g) == FORENSICS_LENGTH for g, _ in records)
                    and forensics_answers(records) == frozenset({tuple(puzzle.answer)}))
        except (AttributeError, TypeError, ValueError, VaultError):
            return False

    def start_support_mission(self, guild_id, user_id, operation_id, mission_type, *, now=None, seed=None, puzzle=None):
        """The member's one mission of this operation: created, or the persisted one (`seed` and `puzzle` ignored then). `seed` is a test seam only.
        `puzzle` is a ForensicsPuzzle the caller generated off the loop (generate_forensics is pure); it is re-checked with the solver here and replaced by a
        freshly generated one if it does not hold. The store itself always runs on the loop (shared sqlite connection)."""
        now = time.time() if now is None else now
        self._op_load(guild_id, user_id, operation_id, now)
        made = None
        if isinstance(mission_type, str) and mission_type == 'forensics' and self._support_row(guild_id, operation_id, user_id) is None:
            made = puzzle if puzzle is not None and self._puzzle_holds(puzzle) else generate_forensics(seed if seed is not None else secrets.token_hex(16))   # pure and slow: before the write lock; discarded if a mission appeared meanwhile
        with self.write_admin():
            self._op_active(guild_id, operation_id, now)
            if not isinstance(mission_type, str) or mission_type not in MISSION_TYPES:
                raise VaultStoreError(codes.VAULT_MISSION_TYPE_UNKNOWN)
            row = self._support_row(guild_id, operation_id, user_id)
            if row is not None:
                if row['mission_type'] != mission_type:
                    raise VaultStoreError(codes.VAULT_MISSION_EXISTS)
                return self._support_view(row)
            if mission_type == 'bulls_cows':
                code = bulls_cows_secret(random.Random(seed) if seed is not None else secrets.SystemRandom())
                puzzle, answer = {'length': BULLS_COWS_LENGTH}, code
            else:
                made = made or generate_forensics(seed if seed is not None else secrets.token_hex(16))
                puzzle, answer = {'records': [[g, list(r) if r is not None else None] for g, r in made.records], 'hidden': made.hidden}, list(made.answer)
            mission_id = self.db.execute('INSERT INTO vault_support_missions(guild_id,operation_id,user_id,mission_type,puzzle_json,answer_json,created_at) VALUES(?,?,?,?,?,?,?)',
                (guild_id, operation_id, user_id, mission_type, json.dumps(puzzle), json.dumps(answer), now)).lastrowid
            return self._support_view(self.one('SELECT * FROM vault_support_missions WHERE id=? AND guild_id=?', (mission_id, guild_id)), created=True)

    def support_mission(self, guild_id, user_id, operation_id, *, now=None):
        now = time.time() if now is None else now
        self._op_load(guild_id, user_id, operation_id, now)
        row = self._support_row(guild_id, operation_id, user_id)
        return None if row is None else self._support_view(row)

    def _support_check(self, row, answer):
        """The normalized answer (a JSON-ready value) of a legal answer, else a refusal; nothing is written."""
        if row['mission_type'] == 'bulls_cows':
            try:
                return check_bulls_cows_guess(answer if isinstance(answer, str) else '')
            except VaultError as error:
                raise VaultStoreError(GUESS_CODES[error.code], **error.params) from None
        if not isinstance(answer, (tuple, list)) or len(answer) != 2 or not valid_forensics_answer(*answer):
            raise VaultStoreError(codes.VAULT_MISSION_ANSWER_INVALID)
        return [answer[0], answer[1]]

    def submit_support_answer(self, guild_id, user_id, operation_id, answer, *, action_key=None, now=None):
        """One answer to the member's own mission. Invalid and repeated answers cost nothing; the last wrong one fails the mission for good."""
        now = time.time() if now is None else now
        self._op_load(guild_id, user_id, operation_id, now)
        with self.write_admin():
            if action_key is not None:
                done = self.one('SELECT a.* FROM vault_support_attempts a JOIN vault_support_missions m ON m.id=a.mission_id WHERE m.guild_id=? AND m.operation_id=? '
                                'AND m.user_id=? AND a.action_key=?', (guild_id, operation_id, user_id, action_key))
                if done is not None:
                    mission = self.one('SELECT * FROM vault_support_missions WHERE id=? AND guild_id=?', (done['mission_id'], guild_id))
                    event = 'solved' if done['correct'] else 'failed' if done['number'] >= BUDGETS[mission['mission_type']] else None
                    return self._support_result(mission, done, True, event)
            self._op_active(guild_id, operation_id, now)
            row = self._support_row(guild_id, operation_id, user_id)
            if row is None:
                raise VaultStoreError(codes.VAULT_MISSION_NONE)
            if row['status'] != 'active':
                raise VaultStoreError(codes.VAULT_MISSION_FINISHED)
            given = self._support_check(row, answer)
            text = json.dumps(given)
            if self.one('SELECT 1 FROM vault_support_attempts WHERE guild_id=? AND mission_id=? AND answer_json=?', (guild_id, row['id'], text)):
                raise VaultStoreError(codes.VAULT_GUESS_DUPLICATE)
            truth = json.loads(row['answer_json'])
            bulls = row['mission_type'] == 'bulls_cows'
            found = bulls_cows_feedback(truth, given) if bulls else None
            correct = given == truth
            number = row['attempts_used'] + 1
            attempt_id = self.db.execute('INSERT INTO vault_support_attempts(guild_id,mission_id,number,answer_json,feedback_json,correct,action_key,created_at) VALUES(?,?,?,?,?,?,?,?)',
                (guild_id, row['id'], number, text, json.dumps(list(found)) if found else None, int(correct), action_key, now)).lastrowid
            event = 'solved' if correct else 'failed' if number >= BUDGETS[row['mission_type']] else None
            self.db.execute('UPDATE vault_support_missions SET attempts_used=?,status=?,finished_at=? WHERE id=? AND guild_id=?',
                (number, event or 'active', now if event else None, row['id'], guild_id))
            refused = False
            if correct:
                operation, _ = self._op_active(guild_id, operation_id, now)
                reward = self.vault_operation_settings(guild_id)['support_reward']
                if reward > 0:
                    refused = self._op_pay(operation, user_id, 'support', reward, now)
            return self._support_result(self.one('SELECT * FROM vault_support_missions WHERE id=? AND guild_id=?', (row['id'], guild_id)),
                                        self.one('SELECT * FROM vault_support_attempts WHERE id=? AND guild_id=?', (attempt_id, guild_id)), False, event, refused)

    def _support_result(self, mission, attempt, replayed, event, refused=False):
        found = json.loads(attempt['feedback_json']) if attempt['feedback_json'] else None
        return {'correct': bool(attempt['correct']), 'feedback': tuple(found) if found is not None else None, 'number': attempt['number'], 'event': event,
                'replayed': replayed, 'reward_refused': refused, 'mission': self._support_view(mission)}
