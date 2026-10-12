"""Items, the shop and inventories (FEAT-22 part B). Mixed into Store; every read and write is scoped by guild_id and every write runs in one write_admin transaction."""
from __future__ import annotations

import json
import math
import time
from dataclasses import replace

from . import items
from .admin_store import ConflictError
from .games import new_seed
from .i18n import ErrorCode, UserError, codes, terms
from .items.recipe import DEFAULT_LOCALE, MAX_COOLDOWN, MAX_PRICE


class _ItemCode(ErrorCode):
    """The ErrorCode of a raised item error; it also equals the stable string (`insufficient`) that callers compare `error.code` with. Only ==, != and ordering accept the string; it hashes as a tuple, so do not use it as a set member or dict key."""

    def __eq__(self, other):
        return other == self.legacy if isinstance(other, str) else tuple.__eq__(self, other)

    def __ne__(self, other):
        return not self == other

    def _key(self, other):
        return other.legacy if isinstance(other, _ItemCode) else other

    def __lt__(self, other):
        return self.legacy < o if isinstance(o := self._key(other), str) else tuple.__lt__(self, other)

    def __gt__(self, other):
        return self.legacy > o if isinstance(o := self._key(other), str) else tuple.__gt__(self, other)

    def __le__(self, other):
        return self == other or self < other

    def __ge__(self, other):
        return self == other or self > other

    __hash__ = tuple.__hash__


def _rebuild_item_error(cls, message, params, state):
    error = cls.__new__(cls)
    ItemError.__init__(error, message, **params)
    error.__dict__.update(state)
    return error


class ItemError(UserError):
    """A shop or item action refused; `code` is stable so callers can phrase it themselves."""
    code = 'item_error'

    def __init__(self, message, /, **params):
        super().__init__(message, **params)
        self._message = message
        if self.code is None:
            self.code = type(self).code
        else:
            self.code = _ItemCode(*self.code)
            self.code.legacy = type(self).code

    def __reduce__(self):
        state = {k: v for k, v in self.__dict__.items() if k not in {'code', 'params', '_message'}}
        return (_rebuild_item_error, (type(self), self._message, self.params, state))


class NotForSale(ItemError):
    code = 'not_for_sale'


class HoldCapReached(ItemError):
    code = 'hold_cap'

    def __init__(self, message, cap, /, **params):
        super().__init__(message, **params)
        self.cap = cap


class AlreadyOwned(ItemError):
    code = 'already_owned'


class InsufficientFunds(ItemError):
    code = 'insufficient'

    def __init__(self, message, balance, price, /, **params):
        super().__init__(message, **params)
        self.balance, self.price = balance, price


class PriceChanged(ItemError):
    code = 'price_changed'

    def __init__(self, message, price, /, **params):
        super().__init__(message, **params)
        self.price = price


class NotHeld(ItemError):
    code = 'not_held'


class UseDisabled(ItemError):
    code = 'use_disabled'


class OnCooldown(ItemError):
    code = 'on_cooldown'

    def __init__(self, message, remaining, /, **params):
        super().__init__(message, **params)
        self.remaining = remaining


HOLD_CAP_MAX = 99
USE_LOG_MAX = 100


def _recipe(item_key):
    try:
        return items.get_item(item_key)
    except (KeyError, TypeError):
        return None


def _loads(text):
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


class ItemsStore:
    def _shop_row(self, guild_id, item_key):
        return self.one('SELECT * FROM shop_items WHERE guild_id=? AND item_key=?', (guild_id, item_key))

    @staticmethod
    def _shop_entry(recipe, row):
        extra, swap, params = (_loads(row[k]) for k in ('extra_entries', 'replace_entries', 'params')) if row else ({}, {}, {})
        cooldown = 0 if recipe.consumable else (row['cooldown_seconds'] if row and row['cooldown_seconds'] is not None else recipe.default_cooldown_seconds)
        return {
            'key': recipe.key, 'name': recipe.name, 'emoji': recipe.emoji, 'kind': recipe.kind, 'consumable': recipe.consumable,
            'for_sale': bool(row['for_sale']) if row else False, 'allow_use': bool(row['allow_use']) if row else True,
            'price': row['price'] if row and row['price'] is not None else recipe.default_price, 'price_override': row['price'] if row else None,
            'default_price': recipe.default_price, 'cooldown_seconds': cooldown, 'cooldown_override': row['cooldown_seconds'] if row else None,
            'default_cooldown_seconds': recipe.default_cooldown_seconds, 'extra_entries': extra, 'replace_entries': swap,
            'params': params, 'effective_params': {**recipe.params, **params}, 'revision': row['revision'] if row else 0,
        }

    def shop_catalog(self, guild_id):
        rows = {r['item_key']: r for r in self.all('SELECT * FROM shop_items WHERE guild_id=?', (guild_id,))}
        return [self._shop_entry(recipe, rows.get(recipe.key)) for recipe in items.all_items()]

    def shop_item(self, guild_id, item_key):
        recipe = _recipe(item_key)
        return None if recipe is None else self._shop_entry(recipe, self._shop_row(guild_id, item_key))

    def save_shop_item(self, guild_id, item_key, *, expected_revision, for_sale, allow_use=True, price=None, cooldown_seconds=None,
                       extra_entries=None, replace_entries=None, params=None):
        """Replaces the guild's whole setting for one built-in item (None price/cooldown = the built-in default). No row yet is revision 0.
        Auditing is the caller's job (AdminService.run), as for the other admin writes."""
        recipe = _recipe(item_key)
        if recipe is None:
            raise UserError(codes.ITEM_UNKNOWN)
        for name, value in ((terms.ITEM_SETTING_FOR_SALE, for_sale), (terms.ITEM_SETTING_ALLOW_USE, allow_use)):
            if not isinstance(value, bool):
                raise UserError(codes.ITEM_FLAG, name=name)
        if price is not None and (isinstance(price, bool) or not isinstance(price, int) or not 1 <= price <= MAX_PRICE):
            raise UserError(codes.ITEM_PRICE_RANGE, limit=MAX_PRICE)
        if cooldown_seconds is not None and (isinstance(cooldown_seconds, bool) or not isinstance(cooldown_seconds, int) or not 0 <= cooldown_seconds <= MAX_COOLDOWN):
            raise UserError(codes.ITEM_COOLDOWN_RANGE, limit=MAX_COOLDOWN)
        params = dict(params or {})
        unknown = set(params) - set(recipe.params)
        if unknown:
            raise UserError(codes.ITEM_NO_SETTING, name=recipe.name, settings=', '.join(sorted(map(str, unknown))))
        items.get_kind(recipe.kind).validate(replace(recipe, params={**recipe.params, **params}))
        extra, swap = items.validate_entries(recipe, extra_entries or None, replace_entries or None)
        with self.write_admin():
            row = self._shop_row(guild_id, item_key)
            if (row['revision'] if row else 0) != expected_revision:
                raise ConflictError(codes.ITEM_CHANGED, name=recipe.name)
            values = (int(for_sale), int(allow_use), price, cooldown_seconds, json.dumps(extra, ensure_ascii=False), json.dumps(swap, ensure_ascii=False),
                      json.dumps(params, sort_keys=True))
            if row:
                self.db.execute('UPDATE shop_items SET for_sale=?,allow_use=?,price=?,cooldown_seconds=?,extra_entries=?,replace_entries=?,params=?,revision=revision+1 WHERE guild_id=? AND item_key=?',
                    (*values, guild_id, item_key))
            else:
                self.db.execute('INSERT INTO shop_items(guild_id,item_key,for_sale,allow_use,price,cooldown_seconds,extra_entries,replace_entries,params,revision) VALUES(?,?,?,?,?,?,?,?,?,1)',
                    (guild_id, item_key, *values))
            return self.shop_item(guild_id, item_key)

    def item_hold_cap(self, guild_id):
        row = self.one('SELECT item_hold_cap FROM guild_settings WHERE guild_id=?', (guild_id,))
        return row['item_hold_cap'] if row else 10

    def set_item_hold_cap(self, guild_id, cap, expected=None):
        if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= HOLD_CAP_MAX:
            raise UserError(codes.ITEM_HOLD_RANGE, limit=HOLD_CAP_MAX)
        with self.write_admin():
            if expected is not None and self.item_hold_cap(guild_id) != expected:
                raise ConflictError(codes.ITEM_HOLD_CHANGED)
            self.db.execute('INSERT INTO guild_settings(guild_id,item_hold_cap) VALUES(?,?) ON CONFLICT(guild_id) DO UPDATE SET item_hold_cap=excluded.item_hold_cap', (guild_id, cap))
        return cap

    def item_qty(self, guild_id, user_id, item_key):
        row = self.one('SELECT qty FROM inventory WHERE guild_id=? AND user_id=? AND item_key=?', (guild_id, user_id, item_key))
        return row['qty'] if row else 0

    def inventory(self, guild_id, user_id):
        """The member's held items (qty > 0), in catalog order, with the recipe's name and emoji."""
        held = {r['item_key']: r for r in self.all('SELECT item_key,qty,last_used_at FROM inventory WHERE guild_id=? AND user_id=? AND qty>0', (guild_id, user_id))}
        out = []
        for recipe in items.all_items():
            if recipe.key in held:
                out.append({'key': recipe.key, 'name': recipe.name, 'emoji': recipe.emoji, 'consumable': recipe.consumable,
                            'qty': held[recipe.key]['qty'], 'last_used_at': held[recipe.key]['last_used_at']})
        return out

    def buy_item(self, guild_id, user_id, item_key, expected_price=None):
        """One transaction: for sale, hold cap, then the ledger debit, the purchase link and the inventory row. Returns {'item','qty','balance','price','ledger_id'}.
        With `expected_price` (the price the buyer was shown), a different current price raises PriceChanged before anything is written."""
        recipe = _recipe(item_key)
        with self.write_admin():
            entry = self._shop_entry(recipe, self._shop_row(guild_id, item_key)) if recipe else None
            if entry is None or not entry['for_sale'] or entry['price'] < 1:
                raise NotForSale(codes.ITEM_NOT_FOR_SALE)
            if expected_price is not None and expected_price != entry['price']:
                raise PriceChanged(codes.ITEM_PRICE_CHANGED, entry['price'], price=entry['price'])
            qty = self.item_qty(guild_id, user_id, item_key)
            if recipe.reusable and qty >= 1:
                raise AlreadyOwned(codes.ITEM_ALREADY_OWNED, name=recipe.name)
            cap = self.item_hold_cap(guild_id)
            if recipe.consumable and qty >= cap:
                raise HoldCapReached(codes.ITEM_HOLD_CAP, cap, cap=cap, name=recipe.name)
            balance = self.balance(guild_id, user_id)
            if balance < entry['price']:
                raise InsufficientFunds(codes.ITEM_INSUFFICIENT, balance, entry['price'], name=recipe.name, price=entry['price'], currency=self.currency_name(guild_id), balance=balance)
            ledger = self._append_ledger(guild_id, user_id, -entry['price'], f'Bought {recipe.name}', user_id, 'shop', None)
            self.db.execute('INSERT INTO shop_purchases(ledger_id,guild_id,user_id,item_key) VALUES(?,?,?,?)', (ledger['id'], guild_id, user_id, item_key))
            self.db.execute('INSERT INTO inventory(guild_id,user_id,item_key,qty) VALUES(?,?,?,1) ON CONFLICT(guild_id,user_id,item_key) DO UPDATE SET qty=qty+1', (guild_id, user_id, item_key))
        return {'item': entry, 'qty': qty + 1, 'balance': ledger['balance_after'], 'price': entry['price'], 'ledger_id': ledger['id']}

    def use_item(self, guild_id, user_id, channel_id, item_key, *, seed=None, now=None, locale=DEFAULT_LOCALE):
        """One transaction: held, use allowed, cooldown over, then the seeded roll, the count or last-used update and the use log row.
        Returns {'result': Result, 'qty': remaining, 'use_id', 'seed', 'item'}; a roll that fails to run writes nothing."""
        now = time.time() if now is None else now
        recipe = _recipe(item_key)
        with self.write_admin():
            held = self.one('SELECT qty,last_used_at FROM inventory WHERE guild_id=? AND user_id=? AND item_key=?', (guild_id, user_id, item_key))
            if recipe is None or held is None or held['qty'] < 1:
                raise NotHeld(codes.ITEM_NOT_HELD)
            entry = self._shop_entry(recipe, self._shop_row(guild_id, item_key))
            if not entry['allow_use']:
                raise UseDisabled(codes.ITEM_USE_DISABLED, name=recipe.name)
            cooldown = entry['cooldown_seconds']
            if recipe.reusable and cooldown and held['last_used_at'] is not None and now < held['last_used_at'] + cooldown:
                remaining = math.ceil(held['last_used_at'] + cooldown - now)
                raise OnCooldown(codes.ITEM_ON_COOLDOWN, remaining, name=recipe.name, remaining=remaining)
            seed = new_seed() if seed is None else str(seed)
            result = items.roll(recipe, seed, locale, entry['extra_entries'], entry['replace_entries'], entry['params'])
            qty = held['qty'] - 1 if recipe.consumable else held['qty']
            self.db.execute('UPDATE inventory SET qty=?,last_used_at=? WHERE guild_id=? AND user_id=? AND item_key=?', (qty, now, guild_id, user_id, item_key))
            use_id = self.db.execute('INSERT INTO item_uses(guild_id,user_id,channel_id,item_key,seed,result,used_at) VALUES(?,?,?,?,?,?,?)',
                (guild_id, user_id, channel_id, item_key, seed, json.dumps(result.to_json(), ensure_ascii=False), now)).lastrowid
        return {'result': result, 'qty': qty, 'use_id': use_id, 'seed': seed, 'item': entry}

    def item_use_log(self, guild_id, limit=50, before_id=None):
        """Newest first; `before_id` pages back. Each row's result is the decoded stored JSON."""
        limit = max(1, min(int(limit), USE_LOG_MAX))
        rows = self.all('SELECT * FROM item_uses WHERE guild_id=? AND (? IS NULL OR id<?) ORDER BY id DESC LIMIT ?', (guild_id, before_id, before_id, limit))
        return [{**dict(r), 'result': _loads(r['result'])} for r in rows]

    def _take_shop_item(self, guild_id, entry):
        """Reversal hook (inside write_admin): for a shop purchase, removes one held item if any is left.
        Returns {'item_key','removed','note'}; note is non-empty when the item was already used. None for any other entry."""
        link = self.one('SELECT item_key FROM shop_purchases WHERE ledger_id=? AND guild_id=?', (entry['id'], guild_id))
        if link is None:
            return None
        key = link['item_key']
        removed = self.db.execute('UPDATE inventory SET qty=qty-1 WHERE guild_id=? AND user_id=? AND item_key=? AND qty>0', (guild_id, entry['user_id'], key)).rowcount > 0
        return {'item_key': key, 'removed': removed, 'note': '' if removed else ' (item already used; coins only)'}
