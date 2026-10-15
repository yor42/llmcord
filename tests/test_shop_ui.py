"""FEAT-22 part D: Shop tab helpers (item save, audit detail, use log rows) and the guarded write path."""
import json
import unittest

import httpx
from fastapi import HTTPException

from helpers import discord_transport, install_session

from llmcord_core import items
from llmcord_core.admin_store import ConflictError
from llmcord_core.currency_ui import ledger_rows, reverse_detail, reverse_note
from llmcord_core.shop_ui import ShopPanel, has_plain_table, item_detail, lines_of, save_item, use_rows, whole, with_lines
from llmcord_core.store import Store
from llmcord_core.web import create_app


def form(**kw):
    return {'for_sale': True, 'allow_use': True, 'price': None, 'cooldown': None, 'sides': 6, 'extra_text': '', 'replace_text': '', **kw}


class HelperTests(unittest.TestCase):
    def test_whole_keeps_blank_and_fractional_input_for_the_store_message(self):
        self.assertEqual((whole(5.0), whole(7), whole(None), whole(2.5), whole(True)), (5, 7, None, 2.5, True))

    def test_non_finite_numbers_are_left_for_the_validation_message(self):
        import math
        for bad in (math.nan, math.inf, -math.inf):
            self.assertIs(whole(bad), bad)
        store = Store()
        self.addCleanup(store.close)
        with self.assertRaisesRegex(ValueError, 'price must be a whole number'):
            save_item(store, 1, store.shop_item(1, 'dice'), **form(price=whole(math.inf)))

    def test_catalog_has_a_fixed_display_order(self):
        self.assertEqual([r.key for r in items.all_items() if r.kind == 'random_message'], ['8ball', 'dice', 'd20', 'fortune_cookie', 'tarot'])
        self.assertEqual([r.key for r in items.all_items()][:5], ['8ball', 'dice', 'd20', 'fortune_cookie', 'tarot'])  # FEAT-64: vault tools follow the toys

    def test_ledger_rows_flag_shop_purchases_for_the_reverse_dialog(self):
        entry = lambda id, source: {'id': id, 'user_id': 5, 'amount': -5, 'balance_after': 1, 'reason': 'r', 'actor_id': 5, 'reverses_id': None, 'created_at': 0, 'source': source}
        rows = ledger_rows([entry(2, 'shop'), entry(1, 'admin')], {}, 'UTC')
        self.assertEqual([r['shop'] for r in rows], [True, False])

    def test_lines_are_trimmed_and_blank_lines_dropped(self):
        self.assertEqual(lines_of('  a \n\n b\r\n   \n'), ['a', 'b'])
        self.assertEqual(lines_of(None), [])

    def test_with_lines_replaces_only_the_default_locale(self):
        self.assertEqual(with_lines({'en': ['x'], 'ko': ['y']}, ['z']), {'en': ['z'], 'ko': ['y']})
        self.assertEqual(with_lines({'en': ['x'], 'ko': ['y']}, []), {'ko': ['y']})

    def test_custom_lines_only_for_plain_string_tables(self):
        plain = {r.key: has_plain_table(r) for r in items.all_items() if r.kind == 'random_message'}
        self.assertEqual(plain, {'8ball': True, 'd20': False, 'dice': False, 'fortune_cookie': True, 'tarot': False})

    def test_detail_holds_counts_never_the_text(self):
        store = Store()
        self.addCleanup(store.close)
        entry = store.shop_item(1, 'fortune_cookie')
        saved = save_item(store, 1, entry, **form(extra_text='SECRET-TEXT a\nSECRET-TEXT b', replace_text='SECRET-TEXT c', price=30))
        detail = item_detail(saved)
        self.assertNotIn('SECRET-TEXT', json.dumps(detail))
        self.assertEqual((detail['item_key'], detail['price'], detail['extra_lines'], detail['replace_lines'], detail['revision']), ('fortune_cookie', 30, 2, 1, 1))

    def test_use_rows_show_names_plain_text_and_the_server_timezone(self):
        uses = [{'id': 3, 'user_id': 5, 'item_key': 'dice', 'used_at': 0, 'result': {'text': 'You rolled a **4**\n(1-6).'}},
                {'id': 2, 'user_id': 6, 'item_key': 'gone', 'used_at': 0, 'result': {'text': 'x' * 300}}]
        rows = use_rows(uses, {5: 'Aria'}, 'Asia/Seoul')
        self.assertEqual((rows[0]['member'], rows[0]['item'], rows[0]['result'], rows[0]['time']), ('Aria', '🎲 Dice', 'You rolled a 4 (1-6).', '1970-01-01 09:00'))
        self.assertEqual((rows[1]['member'], rows[1]['item'], len(rows[1]['result'])), ('Member …6', 'gone', 200))
        self.assertEqual(use_rows(uses[:1], {}, 'No/Such_Zone')[0]['time'], '1970-01-01 00:00')

    def test_panel_borrows_the_currency_name_lookup(self):
        self.assertTrue(callable(ShopPanel.lookup))


class SaveItemTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)

    def test_save_sets_fields_and_blank_means_built_in(self):
        entry = self.store.shop_item(1, 'dice')
        saved = save_item(self.store, 1, entry, **form(price=75.0, cooldown=9, sides=8.0))
        self.assertEqual((saved['for_sale'], saved['price'], saved['cooldown_seconds'], saved['params'], saved['revision']), (True, 75, 9, {'sides': 8}, 1))
        again = save_item(self.store, 1, saved, **form(for_sale=False, allow_use=False, sides=6))
        self.assertEqual((again['price_override'], again['price'], again['cooldown_override'], again['params'], again['allow_use'], again['revision']), (None, 50, None, {}, False, 2))

    def test_consumable_ignores_cooldown_and_sides(self):
        saved = save_item(self.store, 1, self.store.shop_item(1, 'tarot'), **form(cooldown=99, sides=3))
        self.assertEqual((saved['cooldown_override'], saved['params']), (None, {}))

    def test_other_locales_survive_a_save_and_tables_without_plain_lines_are_untouched(self):
        entry = self.store.shop_item(1, 'fortune_cookie')
        saved = save_item(self.store, 1, entry, **form(extra_text='one'))
        self.assertEqual(saved['extra_entries'], {'en': ['one']})
        cleared = save_item(self.store, 1, saved, **form())
        self.assertEqual((cleared['extra_entries'], cleared['replace_entries']), ({}, {}))
        dice = save_item(self.store, 1, self.store.shop_item(1, 'dice'), **form(extra_text='ignored', replace_text='ignored'))
        self.assertEqual((dice['extra_entries'], dice['replace_entries']), ({}, {}))

    def test_bad_values_show_the_store_message_and_write_nothing(self):
        entry = self.store.shop_item(1, 'dice')
        for bad, pattern in ((form(price=0), 'price must be a whole number'), (form(price=2.5), 'price must be a whole number'),
                             (form(cooldown=-1), 'cooldown must be'), (form(sides=None), '.'), (form(sides=1), '.')):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, pattern):
                save_item(self.store, 1, entry, **bad)
        with self.assertRaises(ValueError):
            save_item(self.store, 1, self.store.shop_item(1, 'fortune_cookie'), **form(replace_text='x' * 400))
        self.assertEqual((self.store.shop_item(1, 'dice')['revision'], self.store.shop_item(1, 'fortune_cookie')['revision']), (0, 0))

    def test_a_stale_form_is_refused(self):
        stale = self.store.shop_item(1, 'dice')
        self.store.save_shop_item(1, 'dice', expected_revision=0, for_sale=True, price=60)
        with self.assertRaisesRegex(ConflictError, 'Dice was changed elsewhere. Reload the page and try again.'):
            save_item(self.store, 1, stale, **form(price=70))
        self.assertEqual(self.store.shop_item(1, 'dice')['price'], 60)

    def test_saving_one_server_never_touches_another(self):
        save_item(self.store, 1, self.store.shop_item(1, 'dice'), **form(price=70))
        self.assertEqual((self.store.shop_item(2, 'dice')['revision'], self.store.shop_item(2, 'dice')['for_sale']), (0, False))

    def test_hold_cap_is_guarded_by_the_value_the_form_was_built_from(self):
        self.assertEqual(self.store.set_item_hold_cap(1, 12, 10), 12)
        with self.assertRaises(ConflictError):
            self.store.set_item_hold_cap(1, 5, 10)
        with self.assertRaisesRegex(ValueError, 'from 1 to 99'):
            self.store.set_item_hold_cap(1, whole(100.0), 12)
        self.assertEqual((self.store.item_hold_cap(1), self.store.item_hold_cap(2)), (12, 10))


class ReversalTests(unittest.TestCase):
    def test_reversing_a_purchase_records_the_item_and_names_the_outcome(self):
        store = Store()
        self.addCleanup(store.close)
        store.save_shop_item(1, 'fortune_cookie', expected_revision=0, for_sale=True)
        store.change_balance(1, 5, 100, 'funds', 9)
        bought = store.buy_item(1, 5, 'fortune_cookie')
        entry = {'id': bought['ledger_id']}
        result = store.reverse_entry(1, bought['ledger_id'], 9, 'Mistake')
        self.assertEqual(reverse_detail(entry, result)['shop_item'], {'item_key': 'fortune_cookie', 'removed': True})
        self.assertEqual(reverse_note(result), ' The shop item was taken back.')
        plain = store.change_balance(1, 5, 3, 'grant', 9)
        undone = store.reverse_entry(1, plain['id'], 9, 'Oops')
        self.assertNotIn('shop_item', reverse_detail({'id': plain['id']}, undone))
        self.assertEqual(reverse_note(undone), '')

    def test_an_already_used_item_is_noted(self):
        store = Store()
        self.addCleanup(store.close)
        store.save_shop_item(1, 'fortune_cookie', expected_revision=0, for_sale=True)
        store.change_balance(1, 5, 100, 'funds', 9)
        bought = store.buy_item(1, 5, 'fortune_cookie')
        store.use_item(1, 5, 1, 'fortune_cookie')
        result = store.reverse_entry(1, bought['ledger_id'], 9, 'Refund')
        self.assertEqual(reverse_detail({'id': bought['ledger_id']}, result)['shop_item'], {'item_key': 'fortune_cookie', 'removed': False})
        self.assertIn('already used', reverse_note(result))


class UseLogTests(unittest.TestCase):
    def test_log_pages_newest_first_and_is_per_server(self):
        store = Store()
        self.addCleanup(store.close)
        for guild, n in ((1, 5), (2, 1)):
            for i in range(n):
                store.db.execute('INSERT INTO item_uses(guild_id,user_id,channel_id,item_key,seed,result,used_at) VALUES(?,?,?,?,?,?,?)',
                                 (guild, 5, 1, 'dice', 's', json.dumps({'text': f'g{guild}-{i}'}), i))
        store.db.commit()
        first = store.item_use_log(1, 3)
        rest = store.item_use_log(1, 3, first[-1]['id'])
        self.assertEqual([u['result']['text'] for u in first + rest], [f'g1-{i}' for i in (4, 3, 2, 1, 0)])


class GuardedWriteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        transport, _ = discord_transport(admin_guilds=[1])
        self.app = create_app(':memory:', 'https://pi.test', 'client', 'secret', 'bot', httpx.AsyncClient(transport=transport), enable_dashboard=False,
                              config_path='tests/nonexistent-config.yaml')
        self.service, self.store = self.app.state.admin, self.app.state.store
        self.session = install_session(self.app, user_id='4')

    async def asyncTearDown(self):
        await self.app.state.http.aclose()
        self.store.close()

    def audit(self):
        return [(r['guild_id'], r['action'], r['detail_json']) for r in self.store.all('SELECT * FROM admin_audit ORDER BY id')]

    async def test_admin_save_is_audited_without_the_custom_text(self):
        entry = self.store.shop_item(1, 'fortune_cookie')
        saved = await self.service.run(self.session, 1, lambda: save_item(self.store, 1, entry, **form(price=30, extra_text='SECRET-TEXT')), 'shop.item', item_detail)
        self.assertEqual(saved['price'], 30)
        rows = self.audit()
        self.assertEqual([(g, a) for g, a, _ in rows], [(1, 'shop.item')])
        self.assertEqual(json.loads(rows[0][2])['item_key'], 'fortune_cookie')
        self.assertNotIn('SECRET-TEXT', rows[0][2])

    async def test_a_user_without_access_to_the_server_cannot_save_or_read_it(self):
        entry = self.store.shop_item(2, 'dice')
        with self.assertRaises(HTTPException) as caught:
            await self.service.run(self.session, 2, lambda: save_item(self.store, 2, entry, **form(price=1)), 'shop.item', item_detail)
        self.assertEqual(caught.exception.status_code, 403)
        self.assertEqual((self.store.shop_item(2, 'dice')['revision'], self.audit()), (0, []))

    async def test_failed_and_stale_saves_write_no_audit_row(self):
        entry = self.store.shop_item(1, 'dice')
        with self.assertRaises(ValueError):
            await self.service.run(self.session, 1, lambda: save_item(self.store, 1, entry, **form(price=0)), 'shop.item', item_detail)
        self.store.save_shop_item(1, 'dice', expected_revision=0, for_sale=True)
        with self.assertRaises(ConflictError):
            await self.service.run(self.session, 1, lambda: save_item(self.store, 1, entry, **form(price=9)), 'shop.item', item_detail)
        self.assertEqual(self.audit(), [])


if __name__ == '__main__':
    unittest.main()
