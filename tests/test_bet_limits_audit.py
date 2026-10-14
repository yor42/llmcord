"""MNT-79: the Games tab audit details are built from what each save returns (bet limits used to drop their detail)."""
import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import httpx
import nicegui

from helpers import discord_transport, install_session

from llmcord_core import dashboard, games_ui
from llmcord_core.web import create_app


class Control:
    """A stand-in NiceGUI element: keeps its value, and every other method returns itself."""

    def __init__(self, value=None):
        self.value = value

    def set_value(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return lambda *args, **kwargs: self


class FakeUi:
    def __getattr__(self, name):
        if name in {'number', 'switch', 'select', 'input', 'textarea'}:
            return lambda *args, value=None, **kwargs: Control(value)
        if name == 'refreshable':
            return _Refreshable
        return lambda *args, **kwargs: Control()


class _Refreshable:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def refresh(self):
        pass


class RecordingSavebar:
    active = None

    def __init__(self):
        self.tracked = []

    def track(self, name, controls, **kwargs):
        self.tracked.append(SimpleNamespace(name=name, controls=controls, **kwargs))


@contextmanager
def _plain_section(title):
    yield


class GamesAuditDetailTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        transport, _ = discord_transport(admin_guilds=[1])
        self.app = create_app(':memory:', 'https://pi.test', 'client', 'secret', 'bot', httpx.AsyncClient(transport=transport),
                              enable_dashboard=False, config_path='tests/nonexistent-config.yaml')
        self.store, self.service = self.app.state.store, self.app.state.admin
        self.session = install_session(self.app, user_id='4')

    async def asyncTearDown(self):
        await self.app.state.http.aclose()
        self.store.close()

    def build(self, *sections):
        """Run the Games tab section builders against a fake UI and return what they registered with the savebar."""
        bar = RecordingSavebar()
        ctx = SimpleNamespace(store=self.store, guild_id=1, savebar=bar, channel_names={}, app=self.app, ident=self.session,
                              button=lambda *args, **kwargs: Control())
        with mock.patch.object(nicegui, 'ui', FakeUi()), mock.patch.object(dashboard, 'section', _plain_section):
            for section in sections:
                section(ctx)
        return {entry.action: entry for entry in bar.tracked}

    def audit_rows(self):
        return [(row['action'], json.loads(row['detail_json'])) for row in self.store.all('SELECT action, detail_json FROM admin_audit ORDER BY id')]

    async def test_saving_bet_limits_audits_the_saved_limits(self):
        """MNT-79: saving the bet limits writes one audit row whose detail carries min_bet and max_bet."""
        bets = self.build(games_ui.all_games_section)['games.bets']
        smallest, largest = bets.controls
        smallest.value, largest.value = 5, 200
        result = await self.service.run(self.session, 1, bets.save, bets.action, bets.detail)
        self.assertEqual(result, {'min_bet': 5, 'max_bet': 200})
        self.assertEqual(self.store.game_settings(1), {'min_bet': 5, 'max_bet': 200})
        self.assertEqual(self.audit_rows(), [('games.bets', {'min_bet': 5, 'max_bet': 200})])

    async def test_a_second_bet_limit_save_audits_its_own_limits(self):
        """MNT-79: the detail follows each save, not the first one."""
        bets = self.build(games_ui.all_games_section)['games.bets']
        smallest, largest = bets.controls
        for low, high in ((2, 50), (3, 75)):
            smallest.value, largest.value = low, high
            await self.service.run(self.session, 1, bets.save, bets.action, bets.detail)
        self.assertEqual(self.audit_rows(), [('games.bets', {'min_bet': 2, 'max_bet': 50}), ('games.bets', {'min_bet': 3, 'max_bet': 75})])

    async def test_every_games_tab_save_audits_with_its_detail(self):
        """MNT-79: no tracked Games tab save falls back to detail_error; each detail builds from its real save result."""
        tracked = self.build(games_ui.all_games_section, *games_ui.GAME_SECTIONS.values())
        self.assertIn('games.bets', tracked)
        self.assertGreaterEqual(len(tracked), 15)
        for action, entry in tracked.items():
            with self.subTest(action):
                before = len(self.audit_rows())
                await self.service.run(self.session, 1, entry.save, action, entry.detail)
                rows = self.audit_rows()
                self.assertEqual(len(rows), before + 1)
                self.assertEqual(rows[-1][0], action)
                self.assertNotIn('detail_error', rows[-1][1])
                self.assertTrue(rows[-1][1], 'the detail is not empty')


if __name__ == '__main__':
    unittest.main()
