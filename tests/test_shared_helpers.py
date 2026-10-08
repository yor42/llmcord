"""Shared helpers consolidated by ARCH-03."""
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

from llmcord_core.auth import ADMINISTRATOR, DISCORD_API, is_server_admin
from llmcord_core.discord_bot import recent_human_lines


class ServerAdminTests(unittest.TestCase):
    def test_is_server_admin(self):
        """ARCH-03: one administrator predicate."""
        self.assertTrue(is_server_admin({'owner': True}))
        self.assertTrue(is_server_admin({'permissions': str(ADMINISTRATOR | 1)}))
        self.assertFalse(is_server_admin({'permissions': '4'}))
        self.assertFalse(is_server_admin({}))

    def test_single_api_base(self):
        """ARCH-03: web and avatars reuse auth.DISCORD_API."""
        from llmcord_core import avatars, web
        self.assertIs(web.DISCORD_API, DISCORD_API)
        self.assertIs(avatars.DISCORD_API, DISCORD_API)


class FakeChannel:
    def __init__(self, messages):
        self.messages, self.calls = messages, []

    async def history(self, **kwargs):
        self.calls.append(kwargs)
        for m in self.messages:
            yield m


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def msg(i, age, text='hi', bot=False, webhook=None):
    return NS(id=i, created_at=NOW - timedelta(seconds=age), content=text, webhook_id=webhook, mentions=[],
              author=NS(id=i, bot=bot, display_name=f'u{i}', name=f'u{i}'))


class RecentHistoryTests(unittest.IsolatedAsyncioTestCase):
    limits = {'recent_messages': 3}

    async def run_h(self, messages, floor=0.0, before=None):
        channel = FakeChannel(messages)
        cutoff = NOW - timedelta(seconds=100)
        return await recent_human_lines(channel, self.limits, cutoff, floor, before=before), channel

    async def test_filters_limit_and_order(self):
        """ARCH-03: skips bots/webhooks/empty, caps at limit, returns chronological order."""
        out, channel = await self.run_h([msg(1, 1), msg(2, 2, bot=True), msg(3, 3, webhook=9), msg(4, 4, ''),
                                         msg(5, 5), msg(6, 6), msg(7, 7)])
        self.assertEqual([m['message_id'] for m in out], [6, 5, 1])
        self.assertEqual(channel.calls, [{'limit': 15}])

    async def test_stops_at_cutoff_and_floor(self):
        """ARCH-03: stop at the first message older than cutoff or before the floor."""
        out, _ = await self.run_h([msg(1, 1), msg(2, 200), msg(3, 3)])
        self.assertEqual([m['message_id'] for m in out], [1])
        floor = (NOW - timedelta(seconds=10)).timestamp()
        out, _ = await self.run_h([msg(1, 1), msg(2, 20), msg(3, 3)], floor=floor)
        self.assertEqual([m['message_id'] for m in out], [1])

    async def test_before_passed_through(self):
        """ARCH-03: before is forwarded only when given."""
        anchor = object()
        _, channel = await self.run_h([], before=anchor)
        self.assertEqual(channel.calls, [{'limit': 15, 'before': anchor}])


if __name__ == '__main__':
    unittest.main()
