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
        self.assertEqual(notes, [('Discord channels are unavailable', 'negative')])

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


if __name__ == '__main__':
    unittest.main()
