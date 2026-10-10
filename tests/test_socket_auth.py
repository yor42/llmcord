"""MNT-08: offline tests for the dashboard live-socket auth boundary (`dashboard._install_socket_auth`)."""
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from nicegui import Client, core

from llmcord_core import dashboard
from llmcord_core.dashboard import NOTICE_WINDOW, OPERATOR, rejection_notice, should_notify

COOKIE = 'abc123'


class FakeSio:
    def __init__(self, environ):
        self.eio = SimpleNamespace(cors_allowed_origins=None)
        self.environ = environ
        self.on_handlers = {}
        self.handlers = {'/': {name: mock.Mock(name=name) for name in
                               ('event', 'handshake', 'connect', 'javascript_response', 'ack', 'log')}}
        self.handlers['/']['handshake'] = mock.AsyncMock(return_value='handshake-ok')
        self.handlers['/']['connect'] = mock.AsyncMock(return_value='connect-ok')

    def on(self, name, handler=None):
        def register(fn):
            self.on_handlers[name] = fn
            return fn
        return register(handler) if handler else register

    def get_environ(self, sid):
        return self.environ


class FakeClient:
    def __init__(self, binding=None, has_binding=True):
        if has_binding:
            self.llmcord_binding = binding

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeAuth:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def _call(self, name, *args):
        self.calls.append((name, *args))
        if self.error:
            raise self.error

    async def guard(self, session, guild_id):
        await self._call('guard', session, guild_id)

    async def guard_operator(self, session):
        await self._call('guard_operator', session)

    async def session(self, session):
        await self._call('session', session)


class SocketAuthTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sio = FakeSio({'HTTP_COOKIE': f'llmcord_session={COOKIE}'})
        self.auth = FakeAuth()
        self.app = SimpleNamespace(state=SimpleNamespace(base_url='https://example.test', auth=self.auth))
        self.clients = {}
        self.original_event = self.sio.handlers['/']['event']
        self.original_handshake = self.sio.handlers['/']['handshake']
        self.original_connect = self.sio.handlers['/']['connect']
        for patcher in (mock.patch.object(core, 'sio', self.sio),
                        mock.patch.object(Client, 'instances', self.clients)):
            patcher.start()
            self.addCleanup(patcher.stop)
        notify = mock.patch('nicegui.ui.notify')
        self.notify = notify.start()
        self.addCleanup(notify.stop)
        self.clock = [100.0]
        clock = mock.patch.object(dashboard, 'time', SimpleNamespace(monotonic=lambda: self.clock[0]))
        clock.start()
        self.addCleanup(clock.stop)
        dashboard._install_socket_auth(self.app)
        self.event = self.sio.on_handlers['event']
        self.handshake = self.sio.on_handlers['handshake']
        self.connect = self.sio.on_handlers['connect']

    def add_client(self, cid='c1', **kwargs):
        client = FakeClient(**kwargs)
        self.clients[cid] = client
        return client

    def assert_dropped_silently(self):
        self.original_event.assert_not_called()
        self.notify.assert_not_called()

    async def test_cors_origin_is_base_url(self):
        self.assertEqual(self.sio.eio.cors_allowed_origins, ['https://example.test'])

    async def test_unknown_client_id_dropped_silently(self):
        await self.event('sid', {'client_id': 'nope'})
        self.assert_dropped_silently()
        self.assertEqual(self.auth.calls, [])

    async def test_missing_client_id_dropped_silently(self):
        await self.event('sid', {})
        self.assert_dropped_silently()

    async def test_unbound_client_dropped_silently(self):
        self.add_client(has_binding=False)
        await self.event('sid', {'client_id': 'c1'})
        self.assert_dropped_silently()
        self.assertEqual(self.auth.calls, [])

    async def test_binding_none_attribute_dropped_silently(self):
        self.add_client(binding=None)
        await self.event('sid', {'client_id': 'c1'})
        self.assert_dropped_silently()
        self.assertEqual(self.auth.calls, [])

    async def test_missing_cookie_dropped_silently(self):
        self.add_client(binding=(COOKIE, 5))
        self.sio.environ = {}
        await self.event('sid', {'client_id': 'c1'})
        self.assert_dropped_silently()
        self.assertEqual(self.auth.calls, [])

    async def test_mismatched_cookie_dropped_silently(self):
        self.add_client(binding=('other-session', 5))
        await self.event('sid', {'client_id': 'c1'})
        self.assert_dropped_silently()
        self.assertEqual(self.auth.calls, [])

    async def test_guild_binding_allowed_calls_guard_and_original(self):
        self.add_client(binding=(COOKIE, 5))
        message = {'client_id': 'c1', 'id': 1}
        await self.event('sid', message)
        self.original_event.assert_called_once_with('sid', message)
        self.assertEqual(self.auth.calls, [('guard', COOKIE, 5)])
        self.notify.assert_not_called()

    async def test_guild_rejection_notifies_and_dedupes(self):
        self.add_client(binding=(COOKIE, 5))
        self.auth.error = HTTPException(403, 'Manage Server required')
        await self.event('sid', {'client_id': 'c1'})
        self.original_event.assert_not_called()
        self.notify.assert_called_once_with('Manage Server required', type='negative', timeout=8000)
        self.clock[0] += NOTICE_WINDOW - 0.5
        await self.event('sid', {'client_id': 'c1'})
        self.assertEqual(self.notify.call_count, 1)
        self.auth.error = HTTPException(403, 'Different reason')
        await self.event('sid', {'client_id': 'c1'})
        self.assertEqual(self.notify.call_count, 2)
        self.assertEqual(self.notify.call_args.args, ('Different reason',))
        self.original_event.assert_not_called()

    async def test_same_notice_renotified_after_window(self):
        self.add_client(binding=(COOKIE, 5))
        self.auth.error = HTTPException(403, 'nope')
        await self.event('sid', {'client_id': 'c1'})
        self.clock[0] += NOTICE_WINDOW
        await self.event('sid', {'client_id': 'c1'})
        self.assertEqual(self.notify.call_count, 2)

    async def test_notice_dedupe_is_per_client(self):
        self.add_client('c1', binding=(COOKIE, 5))
        self.add_client('c2', binding=(COOKIE, 5))
        self.auth.error = HTTPException(403, 'nope')
        await self.event('sid', {'client_id': 'c1'})
        await self.event('sid', {'client_id': 'c2'})
        self.assertEqual(self.notify.call_count, 2)

    async def test_401_uses_sign_in_expired_text(self):
        self.add_client(binding=(COOKIE, 5))
        self.auth.error = HTTPException(401, 'whatever')
        await self.event('sid', {'client_id': 'c1'})
        self.notify.assert_called_once_with('Your Discord sign-in expired. Sign in again.', type='negative', timeout=8000)

    async def test_non_http_error_is_not_swallowed_or_notified(self):
        """Characterization (MNT-08): only HTTPException is mapped; other auth errors propagate and nothing is forwarded."""
        self.add_client(binding=(COOKIE, 5))
        self.auth.error = RuntimeError('db down')
        with self.assertRaises(RuntimeError):
            await self.event('sid', {'client_id': 'c1'})
        self.assert_dropped_silently()

    async def test_operator_binding_uses_guard_operator(self):
        self.add_client(binding=(COOKIE, OPERATOR))
        await self.event('sid', {'client_id': 'c1'})
        self.assertEqual(self.auth.calls, [('guard_operator', COOKIE)])
        self.original_event.assert_called_once()

    async def test_operator_rejection_notifies(self):
        self.add_client(binding=(COOKIE, OPERATOR))
        self.auth.error = HTTPException(403, 'Operator only')
        await self.event('sid', {'client_id': 'c1'})
        self.original_event.assert_not_called()
        self.notify.assert_called_once_with('Operator only', type='negative', timeout=8000)

    async def test_session_only_binding(self):
        self.add_client(binding=(COOKIE, None))
        await self.event('sid', {'client_id': 'c1'})
        self.assertEqual(self.auth.calls, [('session', COOKIE)])
        self.original_event.assert_called_once()

    async def test_handshake_rejected_returns_false(self):
        self.add_client(binding=(COOKIE, 5))
        self.auth.error = HTTPException(403, 'no')
        self.assertIs(await self.handshake('sid', {'client_id': 'c1'}), False)
        self.original_handshake.assert_not_called()
        self.notify.assert_not_called()

    async def test_handshake_unbound_returns_false(self):
        self.assertIs(await self.handshake('sid', {'client_id': 'zzz'}), False)
        self.original_handshake.assert_not_called()

    async def test_handshake_allowed_delegates(self):
        self.add_client(binding=(COOKIE, 5))
        message = {'client_id': 'c1'}
        self.assertEqual(await self.handshake('sid', message), 'handshake-ok')
        self.original_handshake.assert_awaited_once_with('sid', message)
        self.assertEqual(self.auth.calls, [('guard', COOKIE, 5)])

    async def test_connect_implicit_handshake_bad_cookie_returns_false(self):
        self.add_client(binding=('other-session', 5))
        environ = {'QUERY_STRING': 'client_id=c1&implicit_handshake=true',
                   'HTTP_COOKIE': f'llmcord_session={COOKIE}'}
        self.assertIs(await self.connect('sid', environ), False)
        self.original_connect.assert_not_called()

    async def test_connect_implicit_handshake_good_cookie_delegates(self):
        self.add_client(binding=(COOKIE, 5))
        environ = {'QUERY_STRING': 'client_id=c1&implicit_handshake=true',
                   'HTTP_COOKIE': f'llmcord_session={COOKIE}'}
        self.assertEqual(await self.connect('sid', environ), 'connect-ok')
        self.original_connect.assert_awaited_once_with('sid', environ, None)

    async def test_connect_without_implicit_handshake_skips_auth(self):
        environ = {'QUERY_STRING': 'client_id=c1'}
        self.assertEqual(await self.connect('sid', environ), 'connect-ok')
        self.assertEqual(self.auth.calls, [])

    GUARDED = ('javascript_response', 'ack', 'log')

    async def test_guarded_handlers_drop_unbound_and_bad_cookie(self):
        self.add_client('bound', binding=('other-session', 5))
        self.add_client('unbound', has_binding=False)
        for name in self.GUARDED:
            for cid in ('bound', 'unbound', 'missing'):
                with self.subTest(name=name, client=cid):
                    await self.sio.on_handlers[name]('sid', {'client_id': cid})
                    self.sio.handlers['/'][name].assert_not_called()
        self.assertEqual(self.auth.calls, [])
        self.notify.assert_not_called()

    async def test_guarded_handlers_use_session_only_not_guard(self):
        for name in self.GUARDED:
            for binding, label in (((COOKIE, 5), 'guild'), ((COOKIE, OPERATOR), 'operator')):
                with self.subTest(name=name, binding=label):
                    self.auth.calls.clear()
                    self.add_client(binding=binding)
                    message = {'client_id': 'c1'}
                    await self.sio.on_handlers[name]('sid', message)
                    self.sio.handlers['/'][name].assert_called_with('sid', message)
                    self.assertEqual(self.auth.calls, [('session', COOKIE)])

    async def test_guarded_handlers_do_not_notify_on_rejection(self):
        """Characterization (MNT-08): a rejected session drops the ack/log/js result without a toast."""
        self.add_client(binding=(COOKIE, 5))
        self.auth.error = HTTPException(401, 'expired')
        for name in self.GUARDED:
            with self.subTest(name=name):
                await self.sio.on_handlers[name]('sid', {'client_id': 'c1'})
                self.sio.handlers['/'][name].assert_not_called()
        self.notify.assert_not_called()

    async def test_guarded_handlers_await_async_original(self):
        for name in self.GUARDED:
            with self.subTest(name=name):
                original = mock.AsyncMock()
                self.sio.handlers['/'][name] = original
                dashboard._install_socket_auth(self.app)
                self.add_client(binding=(COOKIE, 5))
                await self.sio.on_handlers[name]('sid', {'client_id': 'c1'})
                original.assert_awaited_once_with('sid', {'client_id': 'c1'})


class NoticeHelperTests(unittest.TestCase):
    def test_rejection_notice_ignores_non_http_errors(self):
        self.assertIsNone(rejection_notice(None))
        self.assertIsNone(rejection_notice(ValueError('x')))

    def test_rejection_notice_texts(self):
        self.assertEqual(rejection_notice(HTTPException(401, 'x')), 'Your Discord sign-in expired. Sign in again.')
        self.assertEqual(rejection_notice(HTTPException(403, '')), 'This action was rejected')
        self.assertEqual(rejection_notice(HTTPException(403, 'Custom')), 'Custom')

    def test_should_notify_window_boundary(self):
        self.assertTrue(should_notify(None, 'a', 10.0))
        self.assertFalse(should_notify(('a', 10.0), 'a', 10.0 + NOTICE_WINDOW - 0.01))
        self.assertTrue(should_notify(('a', 10.0), 'a', 10.0 + NOTICE_WINDOW))
        self.assertTrue(should_notify(('a', 10.0), 'b', 10.0))


if __name__ == '__main__':
    unittest.main()
