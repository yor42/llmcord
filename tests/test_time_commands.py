"""/time set|show|clear member timezone commands (FEAT-03, decision D14).

Replies are ephemeral and pinned exactly. ``interaction.id`` is a fixed snowflake so the local time is deterministic.
"""
import unittest
from datetime import datetime, timezone
from unittest import mock
from zoneinfo import available_timezones

from helpers import FakeInteraction, autocomplete, invoke, make_settings

from llmcord_core.discord_bot import SkitBot
from llmcord_core.prompts import time_values

# 2026-10-08 08:39:00 UTC
MOMENT_MS = int(datetime(2026, 10, 8, 8, 39, tzinfo=timezone.utc).timestamp() * 1000)
SNOWFLAKE = (MOMENT_MS - 1420070400000) << 22
UNKNOWN = "Unknown timezone. Pick one from the list, such as Asia/Seoul."


def interaction(**kwargs):
    made = FakeInteraction(**kwargs)
    made.id = SNOWFLAKE
    return made


def local(zone):
    return time_values(SNOWFLAKE, zone)["local_time"]


class TimeCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store

    def tearDown(self):
        self.store.close()

    async def run_command(self, path, *args, **kwargs):
        made = interaction(**kwargs)
        await invoke(self.bot, path, made, *args)
        return made

    def assert_ephemeral(self, made):
        sent = made.response.sent + made.followup.sent
        self.assertEqual(len(sent), 1, sent)
        self.assertTrue(sent[0][1].get("ephemeral"), sent)

    async def test_set_valid_zone_stores_and_replies_then_show_is_member(self):
        """FEAT-03: /time set stores the zone for this guild and user, then /time show reports it as the member's setting."""
        made = await self.run_command("time set", "Asia/Seoul")
        self.assertEqual(made.replies, [f"Your timezone here is now Asia/Seoul. Local time: {local('Asia/Seoul')}."])
        self.assertEqual(self.store.user_timezone(1, 9), "Asia/Seoul")
        self.assert_ephemeral(made)
        shown = await self.run_command("time show")
        self.assertEqual(shown.replies, [f"Your timezone here is Asia/Seoul (your setting). Local time: {local('Asia/Seoul')}."])
        self.assert_ephemeral(shown)
        self.assertIn("Thursday, October 8, 2026, 5:39 PM (Asia/Seoul)", shown.replies[0])

    async def test_set_invalid_names_reply_generically_and_store_nothing(self):
        """FEAT-03: unknown names, an empty string, and 65 characters get the generic reply; nothing is stored."""
        for value in ("Mars/Olympus", "", "A" * 65, " Asia/Seoul"):
            with self.subTest(value=value[:20]):
                made = await self.run_command("time set", value)
                self.assertEqual(made.replies, [UNKNOWN])
                self.assert_ephemeral(made)
                self.assertEqual(self.store.user_timezone(1, 9), "")

    async def test_over_64_characters_never_reach_the_store(self):
        """FEAT-03: a 65-character zone is rejected before set_user_timezone is called."""
        with mock.patch.object(type(self.store), "set_user_timezone") as spy:
            made = await self.run_command("time set", "A" * 65)
        self.assertEqual(made.replies, [UNKNOWN])
        spy.assert_not_called()

    async def test_show_server_default_and_no_setting(self):
        """FEAT-03: /time show reports the server default, or UTC with a hint when nothing is set."""
        none = await self.run_command("time show")
        self.assertEqual(none.replies, [
            f"Your timezone here is UTC (no timezone set). Local time: {local('UTC')}. Use /time set to choose yours."])
        self.assert_ephemeral(none)
        self.store.set_guild_timezone(1, "America/New_York")
        server = await self.run_command("time show")
        self.assertEqual(server.replies, [
            f"Your timezone here is America/New_York (server default). Local time: {local('America/New_York')}. "
            "Use /time set to choose your own."])
        self.assert_ephemeral(server)

    async def test_clear_falls_back_to_server_default_or_utc(self):
        """FEAT-03: /time clear removes the member row and names the zone now in effect."""
        self.store.set_guild_timezone(1, "America/New_York")
        self.store.set_user_timezone(1, 9, "Asia/Seoul")
        made = await self.run_command("time clear")
        self.assertEqual(made.replies, ["Cleared your timezone. Characters now use America/New_York (server default)."])
        self.assert_ephemeral(made)
        self.assertEqual(self.store.user_timezone(1, 9), "")
        self.store.set_guild_timezone(1, "")
        self.store.set_user_timezone(1, 9, "Asia/Seoul")
        made = await self.run_command("time clear")
        self.assertEqual(made.replies, ["Cleared your timezone. Characters now use UTC (no timezone set)."])

    async def test_clear_with_nothing_set(self):
        """FEAT-03: /time clear with no member timezone says so."""
        made = await self.run_command("time clear")
        self.assertEqual(made.replies, ["You had no timezone set here."])
        self.assert_ephemeral(made)

    async def test_timezone_is_per_guild(self):
        """FEAT-03: guild isolation; a zone set in guild 1 is invisible in guild 2 and clearing there removes nothing."""
        await self.run_command("time set", "Asia/Seoul", guild_id=1)
        other = await self.run_command("time show", guild_id=2)
        self.assertIn("(no timezone set)", other.replies[0])
        self.assertEqual(self.store.user_timezone(2, 9), "")
        cleared = await self.run_command("time clear", guild_id=2)
        self.assertEqual(cleared.replies, ["You had no timezone set here."])
        self.assertEqual(self.store.user_timezone(1, 9), "Asia/Seoul")

    async def test_memory_opt_out_keeps_timezone(self):
        """FEAT-03 (D14): /memory opt_out erases personal memories but not the member's timezone."""
        await self.run_command("time set", "Asia/Seoul")
        await self.run_command("memory opt_in")
        await self.run_command("memory opt_out")
        self.assertEqual(self.store.user_timezone(1, 9), "Asia/Seoul")

    async def test_dm_is_refused_like_other_member_commands(self):
        """FEAT-03: from a DM every /time command says to use a server, ephemerally, and stores nothing."""
        for path, args in (("time set", ("Asia/Seoul",)), ("time show", ()), ("time clear", ())):
            with self.subTest(path=path):
                made = await self.run_command(path, *args, guild_id=None, channel_id=555)
                self.assertEqual(len(made.replies), 1, made.replies)
                self.assertIn("server", made.replies[0].lower())
                self.assert_ephemeral(made)
        self.assertEqual(self.store.user_timezone(None, 9), "")
        self.assertEqual(self.store.user_timezone(1, 9), "")


class TimeAutocompleteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())

    def tearDown(self):
        self.bot.store.close()

    async def zones(self, current):
        choices = await autocomplete(self.bot, "time set", "zone", interaction(), current)
        for choice in choices:
            self.assertEqual(choice.name, choice.value)
        return [choice.value for choice in choices]

    async def test_substring_match_is_case_insensitive(self):
        """FEAT-03: 'seoul' finds Asia/Seoul; every result contains the text; at most 25."""
        values = await self.zones("seoul")
        self.assertIn("Asia/Seoul", values)
        self.assertLessEqual(len(values), 25)
        self.assertTrue(all("seoul" in value.lower() for value in values))

    async def test_empty_input_returns_first_25_sorted(self):
        """FEAT-03: empty input answers the first 25 zone names in sorted order."""
        self.assertEqual(await self.zones(""), sorted(available_timezones())[:25])

    async def test_over_64_characters_returns_nothing(self):
        """FEAT-03: input longer than 64 characters answers an empty list."""
        self.assertEqual(await self.zones("a" * 65), [])


if __name__ == "__main__":
    unittest.main()
