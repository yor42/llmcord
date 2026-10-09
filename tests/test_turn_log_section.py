"""MNT-28: the Log section's paging guard, tested at the narrowest seam that can hold a read in flight.

A browser cannot reorder two guarded reads (the guild lock serialises them), so this drives ``turn_log_section`` with a minimal
stand-in for ``nicegui.ui`` and a ``ctx.run`` that can hold one read open, over a real in-memory Store.
"""
import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from llmcord_core.dashboard import TURN_LOG_PAGE, turn_log_section
from llmcord_core.store import Store


class Element:
    """Records children created inside ``with element:``; enough of the ui.* surface for turn_log_section."""

    def __init__(self, ui, kind, text='', value=None, on_click=None):
        self.ui, self.kind, self.text, self.value, self.on_click = ui, kind, text, value, on_click
        self.children, self.handlers, self.visible, self.enabled = [], [], True, True
        if ui.stack:
            ui.stack[-1].children.append(self)

    def __enter__(self):
        self.ui.stack.append(self)
        return self

    def __exit__(self, *exc):
        self.ui.stack.pop()

    def classes(self, *_):
        return self

    def props(self, *_):
        return self

    def on(self, *_):
        return self

    def on_value_change(self, handler):
        self.handlers.append(handler)

    def add_slot(self, _name):
        return Element(self.ui, 'slot')

    def set_options(self, options, value=None):
        self.options = options

    def set_visibility(self, visible):
        self.visible = visible

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def clear(self):
        self.children.clear()

    async def change(self, value):
        self.value = value
        for handler in self.handlers:
            await handler(None)


class FakeUI:
    def __init__(self):
        self.stack, self.made = [], {}

    def _make(self, kind, name=None, **kwargs):
        element = Element(self, kind, **kwargs)
        self.made[name or kind] = element
        return element

    def card(self):
        return self._make('card')

    def element(self, _tag):
        return self._make('element')

    def column(self):
        return self._make('column', 'body' if 'body' not in self.made else 'column')

    def label(self, text=''):
        return self._make('label', text=text)

    def select(self, options, value=None, label=''):
        return self._make('select', label, value=value)

    def switch(self, text, value=False):
        return self._make('switch', text, value=value)

    def input(self, label=''):
        return self._make('input', label)

    def button(self, text, on_click=None):
        return self._make('button', text, on_click=on_click)

    def expansion(self):
        return self._make('expansion')

    def labels(self):
        return [e.text for e in self.made.values() if e.kind == 'label']


class TurnLogSectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Store()
        self.store.set_turn_log(1, True, 14)
        base = time.time() - 3600
        # TURN_LOG_PAGE + 8 plain entries and one newer error: the first page is full, so Load more is offered.
        for n in range(TURN_LOG_PAGE + 8):
            self.store.add_turn_log(1, channel_id=100, stage='reply · dialogue', request_text=f'req {n}', response_text=f'reply {n}', now=base + n)
        self.store.add_turn_log(1, channel_id=100, stage='reply · failed at dialogue', status='error', reference_id='06f1ee', error_detail='boom', now=base + 100)
        self.hold = None  # an asyncio.Event the next guarded read waits for
        self.reads = 0

    async def guarded_read(self, operation):
        self.reads += 1
        gate, self.hold = self.hold, None
        if gate is not None:
            await gate.wait()
        return operation()

    async def open_section(self):
        ui = FakeUI()
        ctx = SimpleNamespace(store=self.store, guild_id=1, run=self.guarded_read)
        with mock.patch('nicegui.ui', ui):
            await turn_log_section(ctx, lambda cid: f'#c{cid}')
        return ui

    async def test_filter_change_during_load_more_does_not_append_the_older_page(self):
        """Characterization (MNT-28): a filter applied while a Load more read is in flight wins; the older page is dropped, not appended to the new list."""
        ui = await self.open_section()
        body, more, errors_only = ui.made['body'], ui.made['Load more'], ui.made['Errors only']
        self.assertEqual(len(body.children), TURN_LOG_PAGE)
        self.assertTrue(more.visible)
        self.hold = asyncio.Event()
        gate = self.hold
        loading = asyncio.create_task(more.on_click())
        await asyncio.sleep(0)
        self.assertEqual(self.reads, 2)  # first page (with the channel list), and the held older page
        self.assertFalse(more.enabled)
        await errors_only.change(True)  # the new filter loads while the older page is still held
        self.assertEqual(len(body.children), 1)
        gate.set()
        await loading
        self.assertEqual(len(body.children), 1)
        self.assertFalse(more.visible)
        self.assertTrue(more.enabled)

    async def test_load_more_without_a_filter_change_appends_the_older_page(self):
        """Characterization (MNT-28): the control case for the guard above; Load more alone appends the 8 older entries."""
        ui = await self.open_section()
        body, more = ui.made['body'], ui.made['Load more']
        await more.on_click()
        self.assertEqual(len(body.children), TURN_LOG_PAGE + 8 + 1)
        self.assertFalse(more.visible)


if __name__ == '__main__':
    unittest.main()
