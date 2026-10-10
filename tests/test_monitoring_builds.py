"""MNT-29: only the newest Monitoring build fills its container, and overlapping Log reads stay consistent with the newest read.

Driven at the same fake-``ui`` seam as test_turn_log_section: a service whose reads can be held open and made to fail, so the
order in which reads finish is controlled by the test (a browser cannot reorder them: the guild lock serialises them).
"""
import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from llmcord_core.dashboard import LiveContext, TURN_LOG_PAGE, monitoring_panel, turn_log_section
from llmcord_core.store import Store
from test_turn_log_section import FakeUI


class MonitoringFakeUI(FakeUI):
    def __init__(self):
        super().__init__()
        self.created = []  # every element ever made (FakeUI.made keeps only the last per name)

    def link(self, text='', target=''):
        return self._make('link', text=text)

    def table(self, **kwargs):
        return self._make('table')

    def echart(self, options):
        return self._make('echart')

    def _make(self, kind, name=None, **kwargs):
        element = super()._make(kind, name, **kwargs)
        self.created.append(element)
        return element


class Service:
    """A stand-in for AdminService whose operator read and guarded reads can each be held open or failed."""

    def __init__(self, store):
        self.store, self.gates, self.fail = store, [], []
        self.reads = 0
        self.avatars = None

    def effective_backend(self):
        return SimpleNamespace(roles={})

    async def run_operator(self, ident, operation, action=None, detail=None):
        gate = self.gates.pop(0) if self.gates else None
        if gate is not None:
            await gate.wait()
        return operation()

    async def run(self, ident, guild_id, operation, action=None, detail=None):
        self.reads += 1
        gate = self.gates.pop(0) if self.gates else None
        fail = self.fail.pop(0) if self.fail else False
        if gate is not None:
            await gate.wait()
        if fail:
            raise ValueError('read failed')
        return operation()


def make_ctx(store, service):
    request = SimpleNamespace(cookies={}, query_params={})
    ctx = LiveContext(SimpleNamespace(state=SimpleNamespace(admin=service)), request, 1, {'csrf': ''})
    ctx.service = service
    return ctx


class MonitoringBuildTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Store()
        self.store.set_turn_log(1, True, 14)
        self.service = Service(self.store)
        self.ui = MonitoringFakeUI()
        patcher = mock.patch('nicegui.ui', self.ui)
        patcher.start()
        self.addCleanup(patcher.stop)

    def monitoring_ctx(self, operator):
        ctx = make_ctx(self.store, self.service)
        ctx.is_operator = operator
        ctx.selector = SimpleNamespace(value='monitoring')
        container = self.ui.element('div')
        ctx.containers['monitoring'] = container
        ctx.builders['monitoring'] = lambda: monitoring_panel(ctx)
        return ctx, container

    def usage_sections(self, container):
        return [c for c in container.children if c.kind == 'card']

    async def test_build_overtaken_while_awaiting_does_not_fill_twice(self):
        """MNT-29 (fixed): a Monitoring build still awaiting its operator read when a newer build starts leaves the container filled once."""
        ctx, container = self.monitoring_ctx(operator=True)
        gate = asyncio.Event()
        self.service.gates = [gate]
        old = asyncio.create_task(ctx.build('monitoring'))
        await asyncio.sleep(0)
        ctx.built.discard('monitoring')  # what a retention save does before rebuilding
        container.clear()
        await ctx.build('monitoring')
        filled, made = len(self.usage_sections(container)), len(self.ui.created)
        self.assertGreaterEqual(filled, 1)
        gate.set()
        await old
        self.assertEqual((len(self.usage_sections(container)), len(self.ui.created)), (filled, made))
        self.assertIn('monitoring', ctx.built)

    async def test_build_overtaken_while_loading_the_log_does_not_fill_twice(self):
        """MNT-29 (fixed): the Log's first read is also covered; a stale build does not add rows or the empty note after the newer build."""
        ctx, container = self.monitoring_ctx(operator=False)
        gate = asyncio.Event()
        self.service.gates = [gate]
        old = asyncio.create_task(ctx.build('monitoring'))
        await asyncio.sleep(0)
        ctx.built.discard('monitoring')
        container.clear()
        await ctx.build('monitoring')
        made = len(self.ui.created)
        gate.set()
        await old
        self.assertEqual(len(self.ui.created), made)

    async def test_overtaken_build_that_raises_leaves_the_newer_build_built(self):
        """MNT-29: a stale build failing after a newer one started must not discard the newer build's built mark."""
        ctx, container = self.monitoring_ctx(operator=True)
        gate = asyncio.Event()
        self.service.gates = [gate]
        old = asyncio.create_task(ctx.build('monitoring'))
        await asyncio.sleep(0)
        ctx.built.discard('monitoring')
        container.clear()
        await ctx.build('monitoring')
        self.assertIn('monitoring', ctx.built)
        old.cancel()  # the stale build ends with an exception (here cancellation) after the newer one finished
        with self.assertRaises(asyncio.CancelledError):
            await old
        self.assertIn('monitoring', ctx.built)

    async def test_refresh_overtaking_a_build_discards_the_stale_fill(self):
        """MNT-29 (fixed): TabContext.refresh bumps the build token, so a build it overtook cannot fill the cleared container."""
        ctx, container = self.monitoring_ctx(operator=True)
        gate = asyncio.Event()
        self.service.gates = [gate]
        ctx.selector = SimpleNamespace(value='monitoring', set_value=lambda _v: None)
        old = asyncio.create_task(ctx.build('monitoring'))
        await asyncio.sleep(0)
        await ctx.refresh()
        filled, made = len(self.usage_sections(container)), len(self.ui.created)
        gate.set()
        await old
        self.assertEqual((len(self.usage_sections(container)), len(self.ui.created)), (filled, made))


class OverlappingLogReadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Store()
        self.store.set_turn_log(1, True, 14)
        base = time.time() - 3600
        for n in range(TURN_LOG_PAGE + 8):
            self.store.add_turn_log(1, channel_id=100, stage='reply · dialogue', request_text=f'req {n}', response_text=f'reply {n}', now=base + n)
        self.store.add_turn_log(1, channel_id=100, stage='reply · failed at dialogue', status='error', reference_id='06f1ee', error_detail='boom', now=base + 100)
        self.script = []  # per read: (gate or None, fails)
        self.reads = 0

    async def scripted_read(self, operation):
        self.reads += 1
        gate, fails = self.script.pop(0) if self.script else (None, False)
        if gate is not None:
            await gate.wait()
        return None if fails else operation()

    async def open_section(self):
        ui = FakeUI()
        ctx = SimpleNamespace(store=self.store, guild_id=1, run=self.scripted_read)
        with mock.patch('nicegui.ui', ui):
            await turn_log_section(ctx, lambda cid: f'#c{cid}')
        return ui

    async def test_stale_success_after_a_newer_failure_leaves_the_earlier_list(self):
        """MNT-29: filter A is read slowly, filter B fails first, then A's stale success lands; the list and controls stay at the last shown filter."""
        ui = await self.open_section()
        body, errors_only, channel = ui.made['body'], ui.made['Errors only'], ui.made['Channel']
        self.assertEqual(len(body.children), TURN_LOG_PAGE)
        gate = asyncio.Event()
        self.script = [(gate, False), (None, True)]
        slow = asyncio.create_task(errors_only.change(True))
        await asyncio.sleep(0)
        await channel.change(100)  # the newest read fails
        self.assertEqual((errors_only.value, channel.value), (False, None))
        gate.set()
        await slow
        self.assertEqual(len(body.children), TURN_LOG_PAGE)
        self.assertEqual((errors_only.value, channel.value), (False, None))
        self.assertTrue(ui.made['Load more'].enabled)

    async def test_stale_failure_after_a_newer_success_keeps_the_newest_filter(self):
        """MNT-29: filter A's read fails late after filter B already loaded; A's failure must not reset the controls over B's list."""
        ui = await self.open_section()
        body, errors_only, channel = ui.made['body'], ui.made['Errors only'], ui.made['Channel']
        gate = asyncio.Event()
        self.script = [(gate, True), (None, False)]
        slow = asyncio.create_task(channel.change(100))
        await asyncio.sleep(0)
        await errors_only.change(True)  # the newest read succeeds: the one error entry in channel 100
        self.assertEqual(len(body.children), 1)
        gate.set()
        await slow
        self.assertEqual(len(body.children), 1)
        self.assertEqual((errors_only.value, channel.value), (True, 100))
        self.assertTrue(ui.made['Load more'].enabled)


if __name__ == '__main__':
    unittest.main()
