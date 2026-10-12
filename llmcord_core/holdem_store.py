"""Texas Hold'em cash tables (FEAT-33 part B). Mixed into Store; uses AdminStore's round, ledger and snapshot helpers.

A table is one game_tables row and one game_rounds row per hand. game_seats.stack is a seat's starting stack for that hand, payout its end stack
once settled, stake the total bought in. The wallet moves only at buy-in, rebuy and cash-out (game_cashouts); a hand moves chips between stacks only.
"""
from __future__ import annotations

import json
import time

from .admin_store import GameError
from .games import new_seed, registry, seed_hash
from .i18n import codes


class HoldemStore:
    def _buy_in_range(self, rnd):
        """(min, max) buy-in in currency for a Hold'em round, from the rules it pinned."""
        rules = self._round_rules(rnd)
        if rules is None:
            raise GameError(codes.GAME_RESTORE_FAILED)
        bb = 2 * rules['small_blind']
        return rules['min_buy_in_bb'] * bb, rules['max_buy_in_bb'] * bb

    def _cash_out_seat(self, guild_id, table_id, seat, amount, reason):
        """Pays `amount` of a seat's stack to its holder (clamped at the balance cap, 0 for a deleted character) and records it; returns what was paid."""
        paid = self._pay_seat(guild_id, seat, amount, reason)
        self.db.execute('INSERT INTO game_cashouts(guild_id,table_id,kind,ref_id,bought_in,cashed_out,created_at) VALUES(?,?,?,?,?,?,?)',
            (guild_id, table_id, seat['kind'], seat['ref_id'], seat['stake'], paid, time.time()))
        return paid

    def last_cash_out(self, guild_id, table_id, kind, ref_id):
        """What the seat's newest cash-out at this table actually paid (after any balance-cap clamp), or 0."""
        row = self.one('SELECT cashed_out FROM game_cashouts WHERE guild_id=? AND table_id=? AND kind=? AND ref_id=? ORDER BY id DESC LIMIT 1', (guild_id, table_id, kind, ref_id))
        return row['cashed_out'] if row else 0

    def open_cash_tables(self):
        """Every open table of a cash game in any guild, with the status of its newest round (startup recovery)."""
        keys = [k for k in registry.keys() if registry.get(k).cash]
        marks = ','.join('?' * len(keys))
        return [dict(r) for r in self.all(f"SELECT t.guild_id,t.id AS table_id,t.game,t.channel_id,t.thread_id,t.message_id,(SELECT r.status FROM game_rounds r WHERE r.table_id=t.id AND r.guild_id=t.guild_id ORDER BY r.number DESC LIMIT 1) AS status_of_newest_round FROM game_tables t WHERE t.status='open' AND t.game IN ({marks}) ORDER BY t.guild_id,t.id", keys)]

    @staticmethod
    def _end_stack(rnd, seat):
        """What a seat holds after its round: the end stack of a settled hand, else its starting stack (a joining or voided hand)."""
        return (seat['payout'] or 0) if rnd['status'] == 'settled' else seat['stack']

    def _cash_out_table(self, guild_id, table_id):
        """Inside write_admin: every seat of the newest round is paid its stack and recorded."""
        row = self.one('SELECT id FROM game_rounds WHERE guild_id=? AND table_id=? ORDER BY number DESC LIMIT 1', (guild_id, table_id))
        if row is None:
            return
        rnd = self._game_round(guild_id, row['id'])
        reason = f"{self._round_game(rnd).ledger_label} cash-out ({self._game_label(rnd)})"
        for seat in self._game_seats(guild_id, rnd['id']):
            self._cash_out_seat(guild_id, table_id, seat, self._end_stack(rnd, seat), reason)

    def _next_cash_round(self, guild_id, table_id):
        """Inside write_admin: the next lobby, with the rules the table opened with (only Assist follows the server). Seats are copied with their end stacks (starting stacks after a voided hand); busted seats are cashed out for 0 and dropped, and characters a busted member brought are cashed out to their own wallets."""
        prev = self._game_round(guild_id, self.one('SELECT id FROM game_rounds WHERE guild_id=? AND table_id=? ORDER BY number DESC LIMIT 1', (guild_id, table_id))['id'])
        rules = self._round_rules(prev)  # unreadable pinned rules (None) fall back to the server's current rules
        assist = bool(rules and rules['assist'] and self.game_rules(guild_id, 'holdem')['assist'])
        round_id = self._new_round(guild_id, table_id, new_seed(), time.time(), assist=assist, pinned=rules)
        seats = self._game_seats(guild_id, prev['id'])
        busted = {s['ref_id'] for s in seats if s['kind'] == 'member' and self._end_stack(prev, s) <= 0}
        reason = f"{self._round_game(prev).ledger_label} cash-out ({self._game_label(prev)})"
        for seat in seats:
            stack = self._end_stack(prev, seat)
            if seat['kind'] == 'character' and seat['brought_by'] in busted and stack > 0:
                self._cash_out_seat(guild_id, table_id, seat, stack, reason)  # a character leaves with the member who brought it
                continue
            if stack <= 0:
                self.db.execute('INSERT INTO game_cashouts(guild_id,table_id,kind,ref_id,bought_in,cashed_out,created_at) VALUES(?,?,?,?,?,?,?)',
                    (guild_id, table_id, seat['kind'], seat['ref_id'], seat['stake'], 0, time.time()))
                continue
            index = self.one('SELECT COUNT(*) FROM game_seats WHERE guild_id=? AND round_id=?', (guild_id, round_id))[0]
            self.db.execute('INSERT INTO game_seats(guild_id,round_id,seat_index,kind,ref_id,stake,brought_by,stack) VALUES(?,?,?,?,?,?,?,?)',
                (guild_id, round_id, index, seat['kind'], seat['ref_id'], seat['stake'], seat['brought_by'], stack))
        return round_id

    def _next_button(self, guild_id, rnd, seats):
        """Inside write_admin: the dealer seat of a hand. The first hand draws it from the seed; later hands pass it clockwise from the previous button holder to the next member still seated."""
        n = len(seats)
        prev = self.one('SELECT id,button FROM game_rounds WHERE guild_id=? AND table_id=? AND number<? AND button IS NOT NULL ORDER BY number DESC LIMIT 1', (guild_id, rnd['table_id'], rnd['number']))
        if prev is not None:
            old = self._game_seats(guild_id, prev['id'])
            now = {(s['kind'], s['ref_id']): s['seat_index'] for s in seats}
            for k in range(1, len(old) + 1):
                row = old[(prev['button'] + k) % len(old)]
                if (row['kind'], row['ref_id']) in now:
                    return now[(row['kind'], row['ref_id'])]
        return int(seed_hash(rnd['seed'])[:8], 16) % n

    def _lobby_round(self, guild_id, round_id):
        """Inside write_admin: a Hold'em round that is still a lobby (the only time anything about the table changes)."""
        rnd = self._game_round(guild_id, round_id)
        if not self._round_game(rnd).cash:
            raise GameError(codes.HOLDEM_NOT_A_TABLE)
        if rnd['status'] != 'joining':
            raise GameError(codes.GAME_ROUND_STARTED)
        return rnd

    def rebuy(self, guild_id, round_id, user_id, amount):
        """A seated member adds chips between hands, up to the max buy-in; stack and stake both grow. Returns the round snapshot."""
        if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
            raise GameError(codes.HOLDEM_REBUY_WHOLE)
        with self.write_admin():
            rnd = self._lobby_round(guild_id, round_id)
            seat = self.one("SELECT * FROM game_seats WHERE guild_id=? AND round_id=? AND kind='member' AND ref_id=?", (guild_id, round_id, user_id))
            if seat is None:
                raise GameError(codes.GAME_NOT_SEATED)
            high, name = self._buy_in_range(rnd)[1], self.currency_name(guild_id)
            if seat['stack'] + amount > high:
                raise GameError(codes.HOLDEM_STACK_MAX, high=high, name=name, room=max(high - seat['stack'], 0))
            balance = self.balance(guild_id, user_id)
            if balance < amount:
                raise GameError(codes.HOLDEM_REBUY_SHORT, amount=amount, name=name, balance=balance)
            self._append_ledger(guild_id, user_id, -amount, f"{self._round_game(rnd).ledger_label} rebuy ({self._game_label(rnd)})", user_id, 'game', None)
            self.db.execute('UPDATE game_seats SET stack=stack+?,stake=stake+? WHERE guild_id=? AND round_id=? AND seat_index=?', (amount, amount, guild_id, round_id, seat['seat_index']))
            return self._game_snapshot(guild_id, round_id)

    def set_table_assist(self, guild_id, round_id, user_id, on):
        """The host turns Assist on or off in the lobby; on only while the server allows it. Returns the round snapshot."""
        if not isinstance(on, bool):
            raise GameError(codes.RULES_HOLDEM_ASSIST_FLAG)
        with self.write_admin():
            rnd = self._lobby_round(guild_id, round_id)
            if self._game_table(guild_id, rnd['table_id'])['opened_by'] != user_id:
                raise GameError(codes.HOLDEM_ASSIST_OPENER)
            if on and not self.game_rules(guild_id, 'holdem')['assist']:
                raise GameError(codes.HOLDEM_ASSIST_NOT_ALLOWED)
            rules = self._round_rules(rnd)
            if rules is None:
                raise GameError(codes.GAME_RESTORE_FAILED)
            self.db.execute('UPDATE game_rounds SET rules_json=? WHERE id=? AND guild_id=?', (json.dumps({**rules, 'assist': on}, sort_keys=True), round_id, guild_id))
            return self._game_snapshot(guild_id, round_id)

    def settled_seed(self, guild_id, round_id):
        """Internal (replaying a finished hand): never to be rendered while the table is open."""
        rnd = self._game_round(guild_id, round_id)
        return rnd['seed'] if rnd['status'] in ('settled', 'cancelled') else None

    def table_seeds(self, guild_id, table_id):
        """[(hand number, seed)] of the dealt hands of a closed table, oldest first; nothing while the table is open."""
        if not self._table_closed(guild_id, table_id):
            return []
        return [(r['number'], r['seed']) for r in self.all("SELECT number,seed FROM game_rounds WHERE guild_id=? AND table_id=? AND dealt_at IS NOT NULL AND status IN ('settled','cancelled') ORDER BY number", (guild_id, table_id))]

    def table_cash_summary(self, guild_id, table_id):
        """End-of-table money: {'cashouts': [{kind, ref_id, name, bought_in, cashed_out, net}] (a seat that left and sat again is one row),
        'seated': [{kind, ref_id, name, stack, stake}] for whoever still sits (nobody once the table is closed)}."""
        table = self._game_table(guild_id, table_id)
        totals = {}
        for row in self.all('SELECT kind,ref_id,bought_in,cashed_out FROM game_cashouts WHERE guild_id=? AND table_id=? ORDER BY id', (guild_id, table_id)):
            got = totals.setdefault((row['kind'], row['ref_id']), [0, 0])
            got[0] += row['bought_in']
            got[1] += row['cashed_out']
        seated = []
        if table['status'] == 'open':
            row = self.one('SELECT id FROM game_rounds WHERE guild_id=? AND table_id=? ORDER BY number DESC LIMIT 1', (guild_id, table_id))
            if row is not None:
                rnd = self._game_round(guild_id, row['id'])
                seated = [{'kind': s['kind'], 'ref_id': s['ref_id'], 'stack': self._end_stack(rnd, s), 'stake': s['stake']} for s in self._game_seats(guild_id, rnd['id'])]
        names = self.character_names(guild_id, [ref for kind, ref in totals if kind == 'character'] + [s['ref_id'] for s in seated if s['kind'] == 'character'])

        def name(kind, ref):
            return (names.get(ref) or f'Deleted character (#{ref})') if kind == 'character' else None
        return {'cashouts': [{'kind': kind, 'ref_id': ref, 'name': name(kind, ref), 'bought_in': got[0], 'cashed_out': got[1], 'net': got[1] - got[0]} for (kind, ref), got in totals.items()],
                'seated': [{**s, 'name': name(s['kind'], s['ref_id'])} for s in seated]}
