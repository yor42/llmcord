"""FEAT-17 part B: Currency tab helpers, the bot-token member lookup and the guarded currency-name write."""
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx

from llmcord_core.admin_store import ConflictError, CurrencyError
from llmcord_core.avatars import AvatarPublisher
from llmcord_core.currency_ui import CurrencyPanel, friendly, ledger_rows, member_label, parse_amount, parse_member_id, plain_mentions
from llmcord_core.store import Store


class ParseTests(unittest.TestCase):
    def test_member_id_accepts_raw_id_and_mention(self):
        for text, expected in (('123456789012345678', 123456789012345678), ('  <@123>  ', 123), ('<@!456>', 456)):
            self.assertEqual(parse_member_id(text), expected)

    def test_member_id_rejects_non_digits(self):
        for bad in ('', None, 'abc', '12a', '<@&5>', '<@abc>', '0', '-5', '1.5', '@everyone', '9' * 21, '<@5> <@6>'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_member_id(bad)

    def test_amount_is_a_whole_number_in_range(self):
        self.assertEqual((parse_amount(1), parse_amount(50.0), parse_amount(1_000_000)), (1, 50, 1_000_000))
        for bad in (None, 0, -1, 1.5, 1_000_001, True, '5'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_amount(bad)


class LabelTests(unittest.TestCase):
    def test_label_falls_back_to_last_four_digits(self):
        self.assertEqual(member_label({5: 'Aria'}, 5), 'Aria')
        self.assertEqual(member_label({}, 123456789), 'Member …6789')

    def test_mention_markup_is_replaced_with_names(self):
        text = plain_mentions('That would leave <@5> with a negative balance; <@!77777> too.', {5: 'Aria'})
        self.assertEqual(text, 'That would leave Aria with a negative balance; Member …7777 too.')

    def test_friendly_rewrites_only_currency_errors(self):
        self.assertEqual(str(friendly(CurrencyError('Cannot, <@5>.'), {5: 'Aria'})), 'Cannot, Aria.')
        other = ValueError('<@5> stays')
        self.assertIs(friendly(other, {5: 'Aria'}), other)


class LedgerRowTests(unittest.TestCase):
    def entry(self, id, amount, user=5, reverses=None, actor=9):
        return {'id': id, 'user_id': user, 'amount': amount, 'balance_after': 10, 'reason': f'<b>why {id}</b>', 'actor_id': actor, 'reverses_id': reverses, 'created_at': 0}

    def test_reversed_and_reversal_rows_are_marked(self):
        rows = ledger_rows([self.entry(3, -5, reverses=1), self.entry(2, 4), self.entry(1, 5)], {5: 'Aria'}, 'Asia/Seoul')
        self.assertEqual([(r['entry'], r['note'], r['can_reverse']) for r in rows], [('#3', 'Reverses #1', False), ('#2', '', True), ('#1', 'Reversed by #3', False)])
        self.assertEqual((rows[0]['amount'], rows[1]['amount'], rows[0]['time']), ('-5', '+4', '1970-01-01 09:00'))
        self.assertEqual((rows[0]['member'], rows[0]['admin'], rows[0]['reason']), ('Aria', 'Member …9', '<b>why 3</b>'))

    def test_unknown_timezone_falls_back_to_utc(self):
        self.assertEqual(ledger_rows([self.entry(1, 5)], {}, 'No/Such_Zone')[0]['time'], '1970-01-01 00:00')


class MemberNameTests(unittest.IsolatedAsyncioTestCase):
    def publisher(self, handler):
        return AvatarPublisher(None, httpx.AsyncClient(transport=httpx.MockTransport(handler)), 'tok')

    async def test_names_nick_then_global_then_username_and_only_requested_ids(self):
        seen = []
        def handler(request):
            seen.append((request.method, request.url.path, request.headers['Authorization']))
            uid = request.url.path.rsplit('/', 1)[1]
            body = {'1': {'nick': 'Nick', 'user': {'username': 'u1', 'global_name': 'G1'}}, '2': {'nick': None, 'user': {'username': 'u2', 'global_name': 'G2'}},
                    '3': {'user': {'username': 'u3', 'global_name': None}}}
            return httpx.Response(404, json={}) if uid == '4' else httpx.Response(200, json=body[uid])
        names, failed = await self.publisher(handler).member_names(7, [1, 2, 3, 4, 1])
        self.assertEqual((names, failed), ({1: 'Nick', 2: 'G2', 3: 'u3'}, False))
        self.assertEqual(sorted(seen), sorted(('GET', f'/api/v10/guilds/7/members/{i}', 'Bot tok') for i in (1, 2, 3, 4)))

    async def test_failure_is_flagged_and_other_names_kept(self):
        def handler(request):
            return httpx.Response(429, json={}) if request.url.path.endswith('/2') else httpx.Response(200, json={'nick': 'Ok', 'user': {}})
        names, failed = await self.publisher(handler).member_names(7, [1, 2])
        self.assertEqual((names, failed), ({1: 'Ok'}, True))

    async def test_rate_limit_stops_further_requests_and_fails(self):
        seen = []
        def handler(request):
            seen.append(request.url.path)
            return httpx.Response(429, json={}, headers={'Retry-After': '1'})
        names, failed = await self.publisher(handler).member_names(7, list(range(1, 21)))
        self.assertEqual((names, failed), ({}, True))
        self.assertLessEqual(len(seen), 4)

    async def test_ttl_cache_serves_names_and_departed_members_without_http(self):
        seen = []
        def handler(request):
            seen.append(request.url.path)
            return httpx.Response(404, json={}) if request.url.path.endswith('/2') else httpx.Response(200, json={'nick': 'Ok', 'user': {}})
        publisher = self.publisher(handler)
        first = await publisher.member_names(7, [1, 2])
        self.assertEqual(await publisher.member_names(7, [1, 2]), first)
        self.assertEqual(len(seen), 2)
        self.assertEqual(await publisher.member_names(8, [1]), ({1: 'Ok'}, False))
        self.assertEqual(len(seen), 3)

    async def test_transport_error_is_a_failure_not_a_raise(self):
        def handler(request):
            raise httpx.ConnectError('down')
        self.assertEqual(await self.publisher(handler).member_names(7, [1]), ({}, True))


class LookupTests(unittest.IsolatedAsyncioTestCase):
    def panel(self, outcome):
        calls = []
        async def run(ident, guild_id, operation, action=None, detail=None):
            calls.append(operation())
            return await calls[-1]
        avatars = SimpleNamespace(member_names=lambda guild_id, ids: outcome(ids))
        ctx = SimpleNamespace(store=object(), guild_id=1, ident='x', service=SimpleNamespace(run=run, avatars=avatars), live=lambda tab: lambda: True)
        return CurrencyPanel(ctx)

    async def test_cache_skips_known_ids_and_failure_warns_once(self):
        asked = []
        async def outcome(ids):
            asked.append(list(ids))
            return ({5: 'Aria'} if 5 in ids else {}), 6 in ids
        panel, notes = self.panel(outcome), []
        with mock.patch('nicegui.ui.notify', lambda message, **kw: notes.append((message, kw.get('type')))):
            await panel.lookup([5, 5, 6])
            await panel.lookup([5, 6, 7])
            await panel.lookup([5])
        self.assertEqual(asked, [[5, 6], [7]])  # 6 failed in this build: not retried until the next build
        self.assertEqual(notes, [('Member names could not be loaded. Showing member IDs.', 'warning')])
        self.assertEqual(panel.names, {5: 'Aria'})

    async def test_a_departed_member_is_not_asked_again(self):
        asked = []
        async def outcome(ids):
            asked.append(list(ids))
            return {}, False
        panel = self.panel(outcome)
        await panel.lookup([8])
        await panel.lookup([8])
        self.assertEqual(asked, [[8]])


class CurrencyNameGuardTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)

    def test_expected_name_must_still_match(self):
        self.assertEqual(self.store.set_currency_name(1, 'gold', 'coins'), 'gold')
        with self.assertRaises(ConflictError):
            self.store.set_currency_name(1, 'gems', 'coins')
        self.assertEqual(self.store.currency_name(1), 'gold')
        self.assertEqual(self.store.set_currency_name(1, 'gems', 'gold'), 'gems')

    def test_no_expected_value_keeps_the_slash_command_behaviour(self):
        self.store.set_currency_name(1, 'gold')
        self.assertEqual(self.store.set_currency_name(1, 'gems'), 'gems')

    def test_a_refused_change_does_not_write(self):
        with self.assertRaises(ValueError):
            self.store.set_currency_name(1, '<@5>', 'coins')
        self.assertEqual(self.store.currency_name(1), 'coins')

    def test_another_servers_ledger_is_never_listed(self):
        self.store.change_balance(1, 5, 10, 'one', 9)
        self.store.change_balance(2, 6, 7, 'two', 9)
        self.assertEqual([e['reason'] for e in self.store.ledger(2)], ['two'])
        self.assertEqual([b['user_id'] for b in self.store.balances(1)], [5])
        with self.assertRaises(CurrencyError):
            self.store.reverse_entry(2, self.store.ledger(1)[0]['id'], 9, 'cross-server')
