"""FEAT-18 part A: daily check-in bonus with streaks (schema v13, store API, /daily and /admin currency daily)."""
import re
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import discord
from helpers import FakeInteraction, invoke, make_settings

from llmcord_core.admin_store import ConflictError, CurrencyError
from llmcord_core.discord_bot import SkitBot
from llmcord_core.store import Store


def at(year, month, day, hour=12, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp()


class DailyStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.store.set_daily_settings(1, 25, 5, 7)

    def tearDown(self):
        self.store.close()

    def claim(self, day, user=5, guild=1, hour=12):
        return self.store.claim_daily(guild, user, now=at(2026, 10, day, hour))

    def test_off_by_default(self):
        self.assertEqual(self.store.daily_settings(2), {"amount": 0, "streak_bonus": 0, "streak_days": 7})
        result = self.store.claim_daily(2, 5, now=at(2026, 10, 10))
        self.assertEqual(result["status"], "off")
        self.assertEqual(self.store.ledger(2), [])
        self.assertEqual(self.store.balance(2, 5), 0)

    def test_first_claim_pays_the_amount_on_day_one(self):
        result = self.claim(10)
        self.assertEqual((result["status"], result["payout"], result["streak"], result["balance"]), ("paid", 25, 1, 25))
        entry = self.store.ledger(1, 5)[0]
        self.assertEqual((entry["amount"], entry["source"], entry["actor_id"], entry["reason"]), (25, "daily", 5, "Daily check-in (day 1)"))
        self.assertEqual(result["entry"]["id"], entry["id"])

    def test_same_day_is_refused_with_the_next_reset(self):
        self.claim(10, hour=1)
        result = self.claim(10, hour=23)
        self.assertEqual(result["status"], "already")
        self.assertEqual(result["reset"], at(2026, 10, 11, 0))
        self.assertEqual((len(self.store.ledger(1)), self.store.balance(1, 5)), (1, 25))

    def test_next_day_grows_the_streak_with_bonus(self):
        self.claim(10)
        second = self.claim(11)
        third = self.claim(12)
        self.assertEqual([(r["streak"], r["payout"]) for r in (second, third)], [(2, 30), (3, 35)])
        self.assertEqual(self.store.balance(1, 5), 25 + 30 + 35)
        self.assertEqual(self.store.ledger(1, 5)[0]["reason"], "Daily check-in (day 3)")

    def test_a_missed_day_resets_the_streak(self):
        self.claim(10)
        self.claim(11)
        result = self.claim(13)
        self.assertEqual((result["streak"], result["payout"]), (1, 25))

    def test_streak_days_caps_the_bonus(self):
        self.store.set_daily_settings(1, 25, 5, 2)
        payouts = [self.claim(day)["payout"] for day in range(1, 6)]
        self.assertEqual(payouts, [25, 30, 35, 35, 35])
        self.store.set_daily_settings(1, 25, 5, 0)
        self.assertEqual(self.claim(6)["payout"], 25)

    def test_payout_is_capped_at_the_change_limit(self):
        self.store.set_daily_settings(1, 1_000_000, 1_000_000, 365)
        self.claim(10)
        result = self.claim(11)
        self.assertEqual(result["payout"], 1_000_000)
        self.assertEqual(self.store.ledger(1, 5)[0]["amount"], 1_000_000)

    def test_balance_cap_refusal_records_nothing(self):
        self.store.db.execute("INSERT INTO currency_balances(guild_id,user_id,balance,updated_at) VALUES(1,5,1000000000000,0)")
        self.store.db.commit()
        with self.assertRaisesRegex(CurrencyError, "above the maximum"):
            self.claim(10)
        self.assertEqual(self.store.ledger(1), [])
        self.assertIsNone(self.store.one("SELECT 1 FROM currency_daily"))
        self.store.db.execute("UPDATE currency_balances SET balance=0")
        self.store.db.commit()
        self.assertEqual(self.claim(10)["streak"], 1)

    def test_server_timezone_decides_the_day(self):
        self.store.set_guild_timezone(1, "Asia/Seoul")
        first = self.store.claim_daily(1, 5, now=at(2026, 10, 10, 23, 30))
        self.assertEqual(first["status"], "paid")
        self.assertEqual(first["reset"], at(2026, 10, 11, 15))
        self.assertEqual(self.store.claim_daily(1, 5, now=at(2026, 10, 11, 0, 30))["status"], "already")
        self.assertEqual(self.store.claim_daily(1, 5, now=at(2026, 10, 11, 15, 0))["streak"], 2)
        self.assertEqual(self.store.one("SELECT last_day FROM currency_daily")["last_day"], "2026-10-12")

    def test_utc_when_the_timezone_is_unset_or_invalid(self):
        self.assertEqual(self.claim(10, hour=23)["reset"], at(2026, 10, 11, 0))
        self.store.db.execute("UPDATE guild_settings SET timezone='Not/AZone' WHERE guild_id=1")
        self.store.db.commit()
        self.assertEqual(self.store.claim_daily(1, 5, now=at(2026, 10, 11, 23))["streak"], 2)

    def test_reset_follows_dst(self):
        self.store.set_guild_timezone(1, "America/New_York")
        result = self.store.claim_daily(1, 5, now=at(2026, 11, 1, 12))
        self.assertEqual(result["reset"], at(2026, 11, 2, 5))
        self.assertEqual(self.store.claim_daily(1, 5, now=at(2026, 11, 2, 4, 59))["status"], "already")
        self.assertEqual(self.store.claim_daily(1, 5, now=at(2026, 11, 2, 5, 0))["streak"], 2)

    def test_reset_when_midnight_is_skipped(self):
        self.store.set_guild_timezone(1, "America/Santiago")
        self.assertEqual(self.store.claim_daily(1, 5, now=at(2026, 9, 5, 18))["reset"], at(2026, 9, 6, 4))

    def test_moving_the_timezone_back_does_not_allow_a_second_claim(self):
        self.store.set_guild_timezone(1, "Asia/Seoul")
        self.store.claim_daily(1, 5, now=at(2026, 10, 10, 20))
        self.store.set_guild_timezone(1, "America/New_York")
        again = self.store.claim_daily(1, 5, now=at(2026, 10, 10, 20, 5))
        self.assertEqual(again["status"], "already")
        self.assertEqual(again["reset"], at(2026, 10, 12, 4))
        self.assertEqual(len(self.store.ledger(1)), 1)

    def test_two_connections_pay_once(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "shared.sqlite3"
            first, second = Store(path), Store(path)
            try:
                first.set_daily_settings(1, 25, 5, 7)
                results = []
                threads = [threading.Thread(target=lambda s=s: results.append(s.claim_daily(1, 5, now=at(2026, 10, 10))["status"])) for s in (first, second)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                self.assertEqual(sorted(results), ["already", "paid"])
                self.assertEqual((first.balance(1, 5), len(second.ledger(1))), (25, 1))
            finally:
                first.close()
                second.close()

    def test_guild_and_member_isolation(self):
        self.store.set_daily_settings(2, 7, 0, 7)
        self.claim(10)
        self.assertEqual(self.claim(10, user=6)["status"], "paid")
        other = self.store.claim_daily(2, 5, now=at(2026, 10, 10))
        self.assertEqual((other["status"], other["payout"]), ("paid", 7))
        self.assertEqual((self.store.balance(1, 5), self.store.balance(2, 5)), (25, 7))
        self.assertEqual([r["amount"] for r in self.store.ledger(2)], [7])

    def test_settings_validation(self):
        for args in ((-1, 0, 7), (1_000_001, 0, 7), (1, -1, 7), (1, 1_000_001, 7), (1, 0, -1), (1, 0, 366),
                     (True, 0, 7), (1, True, 7), (1, 0, True), (1.5, 0, 7), ("1", 0, 7), (None, 0, 7)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.store.set_daily_settings(1, *args)
        self.assertEqual(self.store.daily_settings(1), {"amount": 25, "streak_bonus": 5, "streak_days": 7})
        self.assertEqual(self.store.set_daily_settings(1, 1_000_000, 1_000_000, 365), {"amount": 1_000_000, "streak_bonus": 1_000_000, "streak_days": 365})

    def test_settings_survive_other_writes_and_stale_expected_conflicts(self):
        self.store.set_currency_name(1, "gold")
        self.store.set_usage_footer(1, False)
        current = self.store.daily_settings(1)
        self.assertEqual(current, {"amount": 25, "streak_bonus": 5, "streak_days": 7})
        self.store.set_daily_settings(1, 10, 0, 3, expected=current)
        with self.assertRaises(ConflictError):
            self.store.set_daily_settings(1, 99, 0, 3, expected=current)
        self.assertEqual(self.store.daily_settings(1)["amount"], 10)


def v12_file(folder):
    path = Path(folder) / "old.sqlite3"
    store = Store(path)
    store.set_currency_name(1, "gold")
    store.change_balance(1, 5, 40, "keep", 9)
    store.close()
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("DROP TABLE currency_daily")
        for column in ("daily_amount", "daily_streak_bonus", "daily_streak_days"):
            db.execute(f"ALTER TABLE guild_settings DROP COLUMN {column}")
        db.execute("PRAGMA user_version=12")
    return path


class DailyMigrationTests(unittest.TestCase):
    def test_v12_file_upgrades_with_a_backup_and_keeps_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = v12_file(folder)
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 13)
                self.assertEqual((store.currency_name(1), store.balance(1, 5)), ("gold", 40))
                self.assertEqual(store.daily_settings(1), {"amount": 0, "streak_bonus": 0, "streak_days": 7})
                store.set_daily_settings(1, 10, 0, 7)
                self.assertEqual(store.claim_daily(1, 5, now=at(2026, 10, 10))["balance"], 50)
            finally:
                store.close()
            backups = list(Path(folder).glob("old.sqlite3.pre-v13-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            with closing(sqlite3.connect(backups[0])) as db:
                self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 12)
                self.assertNotIn("currency_daily", {r[0] for r in db.execute("SELECT name FROM sqlite_master")})
            Store(path).close()
            self.assertEqual(len(list(Path(folder).glob("*.pre-*"))), 1)

    def test_v14_file_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new.sqlite3"
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("PRAGMA user_version=14")
            with self.assertRaisesRegex(ValueError, "newer than this application"):
                Store(path)


class DailyCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store

    def tearDown(self):
        self.store.close()

    async def run_command(self, path, *args, admin=False, user_id=9, guild_id=1, **kwargs):
        interaction = FakeInteraction(guild_id=guild_id, channel_id=100 if guild_id else 555, admin=admin, user_id=user_id)
        await invoke(self.bot, path, interaction, *args, **kwargs)
        return interaction

    def assert_private(self, interaction):
        sent = interaction.response.sent + interaction.followup.sent
        self.assertEqual(len(sent), 1)
        self.assertIs(sent[0][1].get("ephemeral"), True)
        self.assertEqual(sent[0][1].get("allowed_mentions").to_dict(), discord.AllowedMentions.none().to_dict())

    async def test_off_message(self):
        reply = await self.run_command("daily")
        self.assertEqual(reply.replies, ["Daily check-ins are off on this server."])
        self.assert_private(reply)
        self.assertEqual(self.store.ledger(1), [])

    async def test_paid_then_already(self):
        await self.run_command("admin currency name", "gold", admin=True)
        await self.run_command("admin currency daily", 25, admin=True)
        paid = await self.run_command("daily")
        self.assertRegex(paid.replies[0], r"^Checked in: \+25 gold \(day 1 streak\)\. Balance: 25 gold\. Next check-in <t:\d+:R>\.$")
        self.assert_private(paid)
        again = await self.run_command("daily")
        self.assertRegex(again.replies[0], r"^You already checked in today\. Next check-in <t:\d+:R>\.$")
        self.assert_private(again)
        reset = lambda reply: re.search(r"<t:(\d+):R>", reply.replies[0]).group(1)
        self.assertEqual(reset(paid), reset(again))
        self.assertGreater(int(reset(paid)), 0)
        self.assertEqual(len(self.store.ledger(1)), 1)

    async def test_balance_cap_refusal_is_a_plain_reply(self):
        self.store.set_daily_settings(1, 5, 0, 7)
        self.store.db.execute("INSERT INTO currency_balances(guild_id,user_id,balance,updated_at) VALUES(1,9,1000000000000,0)")
        self.store.db.commit()
        reply = await self.run_command("daily")
        self.assertIn("above the maximum", reply.replies[0])
        self.assertEqual(len(self.store.ledger(1)), 0)

    async def test_daily_in_a_dm_is_refused(self):
        reply = await self.run_command("daily", guild_id=None)
        self.assertIn("server", reply.replies[0].lower())

    async def test_admin_daily_saves_and_keeps_omitted_options(self):
        saved = await self.run_command("admin currency daily", 25, streak_bonus=5, streak_days=7, admin=True)
        self.assertEqual(saved.replies, ["Daily check-in: 25 coins, +5 per streak day for up to 7 days."])
        self.assert_private(saved)
        later = await self.run_command("admin currency daily", 30, admin=True)
        self.assertEqual(later.replies, ["Daily check-in: 30 coins, +5 per streak day for up to 7 days."])
        self.assertEqual(self.store.daily_settings(1), {"amount": 30, "streak_bonus": 5, "streak_days": 7})
        off = await self.run_command("admin currency daily", 0, admin=True)
        self.assertEqual(off.replies, ["Daily check-ins are now off."])
        self.assertEqual(self.store.daily_settings(1)["amount"], 0)

    async def test_admin_daily_conflict_is_a_private_reply(self):
        original = self.store.set_daily_settings

        def racing(guild_id, *args, **kwargs):
            original(guild_id, 99, 0, 1)
            return original(guild_id, *args, **kwargs)

        self.store.set_daily_settings = racing
        reply = await self.run_command("admin currency daily", 25, admin=True)
        self.assertEqual(reply.replies, ["The daily check-in settings were changed elsewhere. Run the command again."])
        self.assert_private(reply)
        self.assertEqual(self.store.daily_settings(1)["amount"], 99)

    async def test_admin_daily_without_streak_bonus(self):
        saved = await self.run_command("admin currency daily", 10, streak_bonus=0, admin=True)
        self.assertEqual(saved.replies, ["Daily check-in: 10 coins, no streak bonus."])

    async def test_admin_daily_rejects_non_admins(self):
        refused = await self.run_command("admin currency daily", 25, admin=False)
        self.assertEqual(refused.replies, ["Only server administrators can use that command."])
        self.assertEqual(self.store.daily_settings(1)["amount"], 0)


if __name__ == "__main__":
    unittest.main()
