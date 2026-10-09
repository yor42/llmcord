"""UX-01: LiveContext.button tells a successful operation that returns None apart from a failed one."""
import unittest
from types import SimpleNamespace
from unittest import mock

import httpx
from fastapi import HTTPException

from llmcord_core.dashboard import LiveContext


class FakeService:
    def __init__(self, outcome):
        self.store, self.outcome = object(), outcome

    async def run(self, ident, guild_id, operation, action=None, detail=None):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class ButtonOutcomeTests(unittest.IsolatedAsyncioTestCase):
    async def click(self, outcome, **options):
        """Build a LiveContext over a fake service, click the button, return (then calls, notifications)."""
        app = SimpleNamespace(state=SimpleNamespace(admin=FakeService(outcome)))
        request = SimpleNamespace(cookies={}, query_params={})
        ctx = LiveContext(app, request, 1, {'csrf': 'x'})
        thens, notes, buttons = [], [], []
        def fake_button(text, on_click=None, **kwargs):
            buttons.append(on_click)
            return object()
        def fake_notify(message, **kwargs):
            notes.append((message, kwargs.get('type')))
        with mock.patch('nicegui.ui.button', fake_button), mock.patch('nicegui.ui.notify', fake_notify):
            ctx.button('Go', lambda: None, 'x.y', then=thens.append, **options)
            await buttons[0]()
        return thens, notes

    async def test_then_runs_after_success_returning_none(self):
        """UX-01: a successful operation that returns None still runs then (and the success toast)."""
        thens, notes = await self.click(None, success='Saved')
        self.assertEqual(thens, [None])
        self.assertEqual(notes, [('Saved', 'positive')])

    async def test_then_gets_truthy_result(self):
        """UX-01: then receives a non-None result and success is optional."""
        thens, notes = await self.click(['kept'])
        self.assertEqual(thens, [['kept']])
        self.assertEqual(notes, [])

    async def test_failure_runs_neither_then_nor_success(self):
        """UX-01: a failed operation notifies the error only; then and success do not run."""
        thens, notes = await self.click(HTTPException(403, 'Nope'), success='Saved')
        self.assertEqual(thens, [])
        self.assertEqual(notes, [('Nope', 'negative')])

    async def test_run_return_value_unchanged(self):
        """UX-01: LiveContext.run still returns the result on success and None after a notified failure."""
        for outcome, expected in ((True, True), (ValueError('bad'), None)):
            app = SimpleNamespace(state=SimpleNamespace(admin=FakeService(outcome)))
            ctx = LiveContext(app, SimpleNamespace(cookies={}, query_params={}), 1, {'csrf': 'x'})
            with mock.patch('nicegui.ui.notify'):
                self.assertEqual(await ctx.run(lambda: None), expected)


class ChannelNamesTests(unittest.IsolatedAsyncioTestCase):
    async def load(self, channels):
        """Return (channel_names, notifications) after load_channel_names over a fake avatars.channels."""
        service = FakeService(None)
        async def run(ident, guild_id, operation, action=None, detail=None):
            return operation()
        service.run = run
        service.avatars = SimpleNamespace(channels=channels)
        ctx = LiveContext(SimpleNamespace(state=SimpleNamespace(admin=service)), SimpleNamespace(cookies={}, query_params={}), 1, {'csrf': 'x'})
        notes = []
        with mock.patch('nicegui.ui.notify', lambda message, **kwargs: notes.append((message, kwargs.get('type')))):
            return await ctx.load_channel_names(), notes

    async def test_transport_error_leaves_names_empty(self):
        """A Discord transport error notifies and renders with numeric ids instead of breaking the page."""
        def channels(guild_id):
            raise httpx.ConnectError('down')
        names, notes = await self.load(channels)
        self.assertEqual(names, {})
        self.assertEqual(notes, [('Discord channels could not be loaded. Showing channel IDs.', 'negative')])

    async def test_value_error_leaves_names_empty(self):
        """A rejected channel fetch (ValueError) is notified by run and leaves names empty."""
        def channels(guild_id):
            raise ValueError('Discord channels are unavailable')
        names, notes = await self.load(channels)
        self.assertEqual(names, {})

    async def test_success_maps_text_channels_only(self):
        """Only type-0 channels are mapped, with int keys and a # prefix."""
        names, notes = await self.load(lambda guild_id: [
            {'id': '10', 'name': 'general', 'type': 0}, {'id': '11', 'name': 'voice', 'type': 2}])
        self.assertEqual(names, {10: '#general'})
        self.assertEqual(notes, [])


class ThreadNamesTests(unittest.IsolatedAsyncioTestCase):
    async def load(self, scopes, active_threads, channels=None):
        """Return (thread_names, fetch count) after load_thread_names over fake store scopes and avatars."""
        service = FakeService(None)
        async def run(ident, guild_id, operation, action=None, detail=None):
            return operation()
        service.run = run
        service.store = SimpleNamespace(thread_lore_scopes=lambda guild_id: scopes)
        fetches = []
        def fetch(guild_id):
            fetches.append(guild_id)
            return active_threads(guild_id)
        service.avatars = SimpleNamespace(active_threads=fetch)
        ctx = LiveContext(SimpleNamespace(state=SimpleNamespace(admin=service)), SimpleNamespace(cookies={}, query_params={}), 1, {'csrf': 'x'})
        self.notes = []
        with mock.patch('nicegui.ui.notify', lambda message, **kwargs: self.notes.append((message, kwargs.get('type')))):
            if channels is not None:
                ctx.channels_failed = channels
            return await ctx.load_thread_names(), fetches

    async def test_no_scopes_makes_no_fetch(self):
        """UI-03: a guild without thread lore scopes never calls Discord for threads."""
        names, fetches = await self.load([], lambda guild_id: [{'id': '5', 'name': 'x'}])
        self.assertEqual((names, fetches), ({}, []))

    async def test_scopes_fetch_once_and_map_names(self):
        """UI-03: thread scopes trigger one fetch; names map by int id with a # prefix."""
        names, fetches = await self.load([5], lambda guild_id: [{'id': '5', 'name': 'x'}])
        self.assertEqual((names, fetches), ({5: '#x'}, [1]))

    async def test_failure_sets_names_none_and_labels_neutral(self):
        """UI-44: a failed thread fetch (transport or rejected) leaves names None, so threads read "Thread ...NNNN"."""
        from llmcord_core.admin import AdminService
        def down(guild_id):
            raise httpx.ConnectError('down')
        def rejected(guild_id):
            raise ValueError('Discord threads are unavailable')
        for fail in (down, rejected):
            names, _ = await self.load([5], fail)
            self.assertIsNone(names)
            self.assertEqual(self.notes, [('Thread names could not be loaded. Showing thread IDs.', 'warning')])
            self.assertEqual(AdminService.thread_label(1234567, names), "Thread …4567")

    async def test_guard_rejection_is_a_fetch_failure(self):
        """UI-44: an HTTPException from the auth guard (503/401) does not fail the page; names become None with one toast."""
        service = FakeService(HTTPException(503, 'Busy'))
        service.store = SimpleNamespace(thread_lore_scopes=lambda guild_id: [5])
        ctx = LiveContext(SimpleNamespace(state=SimpleNamespace(admin=service)), SimpleNamespace(cookies={}, query_params={}), 1, {'csrf': 'x'})
        notes = []
        with mock.patch('nicegui.ui.notify', lambda message, **kwargs: notes.append((message, kwargs.get('type')))):
            self.assertEqual(await ctx.load_channel_names(), {})
            self.assertIsNone(await ctx.load_thread_names())
        self.assertEqual(notes, [('Discord channels could not be loaded. Showing channel IDs.', 'negative')])

    async def test_double_failure_notifies_once(self):
        """UI-44: when the channel fetch already failed, the thread failure adds no second toast."""
        def down(guild_id):
            raise httpx.ConnectError('down')
        names, _ = await self.load([5], down, channels=True)
        self.assertIsNone(names)
        self.assertEqual(self.notes, [])


if __name__ == '__main__':
    unittest.main()
