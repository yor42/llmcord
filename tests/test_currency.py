"""FEAT-17 part A: per-server member currency (schema v12, store API, /balance and /admin currency)."""
import sqlite3
import tempfile
import threading
import unittest
import unittest.mock
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

from helpers import FakeInteraction, invoke, make_settings

from llmcord_core.admin_store import CurrencyError
from llmcord_core.discord_bot import SkitBot
from llmcord_core.store import Store


def member(user_id, bot=False):
    return SimpleNamespace(id=user_id, bot=bot, mention=f"<@{user_id}>")


class CurrencyStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()

    def tearDown(self):
        self.store.close()

    def test_default_name_and_set_name(self):
        """The name is 'coins' until an admin sets one; it is per server and stripped."""
        self.assertEqual(self.store.currency_name(1), "coins")
        self.assertEqual(self.store.set_currency_name(1, "  gold "), "gold")
        self.assertEqual((self.store.currency_name(1), self.store.currency_name(2)), ("gold", "coins"))

    def test_name_validation(self):
        for bad in ("", "   ", "x" * 33, "a\nb", "a\x00b", "a\tb", None, "<@5>", "@everyone", "`x`", "**x**", "a_b", "~x", "a|b"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.store.set_currency_name(1, bad)
        self.assertEqual(self.store.currency_name(1), "coins")
        self.assertEqual(self.store.set_currency_name(1, "x" * 32), "x" * 32)

    def test_name_survives_other_settings_writes(self):
        self.store.set_currency_name(1, "gold")
        self.store.set_usage_footer(1, False)
        self.assertEqual(self.store.currency_name(1), "gold")

    def test_grant_and_revoke_append_ledger_rows(self):
        self.assertEqual(self.store.balance(1, 5), 0)
        grant = self.store.change_balance(1, 5, 50, " welcome ", 9)
        self.assertEqual((grant["amount"], grant["balance_after"], grant["reason"], grant["actor_id"], grant["source"], grant["reverses_id"]),
                         (50, 50, "welcome", 9, "admin", None))
        revoke = self.store.change_balance(1, 5, -20, "fine", 9)
        self.assertEqual(revoke["balance_after"], 30)
        self.assertEqual(self.store.balance(1, 5), 30)
        self.assertEqual([r["amount"] for r in self.store.ledger(1, 5)], [-20, 50])

    def test_negative_balance_is_refused_without_a_ledger_row(self):
        self.store.change_balance(1, 5, 10, "start", 9)
        with self.assertRaisesRegex(CurrencyError, r"negative balance \(current balance: 10 coins\)"):
            self.store.change_balance(1, 5, -11, "too much", 9)
        self.assertEqual(self.store.balance(1, 5), 10)
        self.assertEqual(len(self.store.ledger(1)), 1)
        self.store.change_balance(1, 5, -10, "exact", 9)
        self.assertEqual(self.store.balance(1, 5), 0)

    def test_bounds_and_input_validation(self):
        for amount in (0, 1_000_001, -1_000_001, 1.5, True, "5", None):
            with self.subTest(amount=amount), self.assertRaises(CurrencyError):
                self.store.change_balance(1, 5, amount, "x", 9)
        for reason in ("", "  ", "x" * 201, None):
            with self.subTest(reason=reason), self.assertRaises(CurrencyError):
                self.store.change_balance(1, 5, 1, reason, 9)
        self.assertEqual(self.store.change_balance(1, 5, 1, "x" * 200, 9)["amount"], 1)
        self.assertEqual(len(self.store.ledger(1)), 1)

    def test_balance_ceiling(self):
        self.store.db.execute("INSERT INTO currency_balances(guild_id,user_id,balance,updated_at) VALUES(1,5,999999500000,0)")
        self.store.db.commit()
        with self.assertRaisesRegex(CurrencyError, "above the maximum"):
            self.store.change_balance(1, 5, 1_000_000, "over", 9)
        self.assertEqual(self.store.change_balance(1, 5, 500_000, "to the limit", 9)["balance_after"], 1_000_000_000_000)

    def test_reversal_that_would_exceed_the_maximum_is_refused(self):
        self.store.change_balance(1, 5, 10, "g", 9)
        revoke = self.store.change_balance(1, 5, -5, "r", 9)
        self.store.db.execute("UPDATE currency_balances SET balance=1000000000000 WHERE guild_id=1 AND user_id=5")
        self.store.db.commit()
        with self.assertRaisesRegex(CurrencyError, "above the maximum"):
            self.store.reverse_entry(1, revoke["id"], 9, "undo")
        self.assertEqual(self.store.balance(1, 5), 1_000_000_000_000)
        self.assertEqual(len(self.store.ledger(1)), 2)

    def test_limits_above_500_are_clamped(self):
        for _ in range(3):
            self.store.change_balance(1, 5, 1, "x", 9)
        with unittest.mock.patch.object(self.store, "all", wraps=self.store.all) as spy:
            self.store.ledger(1, limit=10_000)
            self.store.balances(1, limit=10_000)
        self.assertEqual([call.args[1][-1] for call in spy.call_args_list[:1]], [500])
        self.assertEqual(spy.call_args_list[1].args[1][-2:], (500, 0))

    def test_guild_isolation(self):
        self.store.change_balance(1, 5, 10, "a", 9)
        self.store.change_balance(2, 5, 70, "b", 9)
        self.assertEqual((self.store.balance(1, 5), self.store.balance(2, 5), self.store.balance(3, 5)), (10, 70, 0))
        self.assertEqual([r["amount"] for r in self.store.ledger(1)], [10])
        self.assertEqual([r["balance"] for r in self.store.balances(2)], [70])
        other = self.store.ledger(2)[0]["id"]
        with self.assertRaisesRegex(CurrencyError, "No such ledger entry"):
            self.store.reverse_entry(1, other, 9, "cross")
        self.assertEqual(self.store.balance(2, 5), 70)

    def test_reverse_once_only(self):
        entry = self.store.change_balance(1, 5, 40, "oops", 9)
        reversal = self.store.reverse_entry(1, entry["id"], 8, "undo")
        self.assertEqual((reversal["amount"], reversal["source"], reversal["reverses_id"], reversal["balance_after"], reversal["user_id"]),
                         (-40, "reversal", entry["id"], 0, 5))
        with self.assertRaisesRegex(CurrencyError, "already reversed"):
            self.store.reverse_entry(1, entry["id"], 8, "again")
        with self.assertRaisesRegex(CurrencyError, "cannot be reversed"):
            self.store.reverse_entry(1, reversal["id"], 8, "undo undo")
        with self.assertRaisesRegex(CurrencyError, "No such ledger entry"):
            self.store.reverse_entry(1, 9999, 8, "missing")
        self.assertEqual(len(self.store.ledger(1)), 2)

    def test_reversal_of_a_revoke_restores_and_one_that_would_go_negative_is_refused(self):
        grant = self.store.change_balance(1, 5, 40, "g", 9)
        revoke = self.store.change_balance(1, 5, -30, "r", 9)
        with self.assertRaisesRegex(CurrencyError, "negative balance"):
            self.store.reverse_entry(1, grant["id"], 9, "undo grant")
        self.assertEqual(self.store.balance(1, 5), 10)
        self.assertEqual(self.store.reverse_entry(1, revoke["id"], 9, "undo revoke")["balance_after"], 40)
        self.assertEqual(self.store.reverse_entry(1, grant["id"], 9, "undo grant")["balance_after"], 0)

    def test_ledger_paging_and_balance_listing(self):
        for user, amount in ((5, 10), (6, 30), (7, 30), (5, 5)):
            self.store.change_balance(1, user, amount, "x", 9)
        rows = self.store.ledger(1)
        self.assertEqual([r["id"] for r in rows], sorted((r["id"] for r in rows), reverse=True))
        self.assertEqual(len(self.store.ledger(1, limit=2)), 2)
        older = self.store.ledger(1, before_id=rows[1]["id"])
        self.assertEqual([r["id"] for r in older], [r["id"] for r in rows[2:]])
        self.assertEqual({r["user_id"] for r in self.store.ledger(1, user_id=5)}, {5})
        self.assertEqual([(r["user_id"], r["balance"]) for r in self.store.balances(1, 10, 0)], [(6, 30), (7, 30), (5, 15)])
        self.assertEqual([r["user_id"] for r in self.store.balances(1, 1, 1)], [7])

    def test_ledger_is_append_only(self):
        entry = self.store.change_balance(1, 5, 10, "x", 9)
        with self.assertRaisesRegex(sqlite3.DatabaseError, "append-only"):
            self.store.db.execute("UPDATE currency_ledger SET amount=99 WHERE id=?", (entry["id"],))
        with self.assertRaisesRegex(sqlite3.DatabaseError, "append-only"):
            self.store.db.execute("DELETE FROM currency_ledger WHERE id=?", (entry["id"],))
        self.assertEqual(self.store.ledger(1)[0]["amount"], 10)

    def test_schema_refuses_a_negative_balance_directly(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.db.execute("INSERT INTO currency_balances(guild_id,user_id,balance,updated_at) VALUES(1,5,-1,0)")
        self.store.db.rollback()

    def test_two_connections_cannot_overdraw(self):
        """Read, check and write share one BEGIN IMMEDIATE transaction, so concurrent revokes cannot both pass."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "shared.sqlite3"
            first, second = Store(path), Store(path)
            try:
                first.change_balance(1, 5, 100, "start", 9)
                results = []

                def revoke(store):
                    try:
                        store.change_balance(1, 5, -60, "spend", 9)
                        results.append("ok")
                    except CurrencyError:
                        results.append("refused")

                threads = [threading.Thread(target=revoke, args=(store,)) for store in (first, second)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                self.assertEqual(sorted(results), ["ok", "refused"])
                self.assertEqual(first.balance(1, 5), 40)
                self.assertEqual(len(second.ledger(1)), 2)
            finally:
                first.close()
                second.close()


def v11_file(folder):
    path = Path(folder) / "old.sqlite3"
    store = Store(path)
    store.set_usage_footer(1, False)
    store.close()
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("DROP TABLE currency_balances")
        db.execute("DROP TABLE currency_ledger")
        db.execute("ALTER TABLE guild_settings DROP COLUMN currency_name")
        db.execute("PRAGMA user_version=11")
    return path


class CurrencyMigrationTests(unittest.TestCase):
    def test_v11_file_upgrades_with_a_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = v11_file(folder)
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 14)
                self.assertEqual(store.currency_name(1), "coins")
                self.assertFalse(store.usage_footer_enabled(1))
                store.change_balance(1, 5, 3, "ok", 9)
                tables = {r[0] for r in store.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertTrue({"currency_balances", "currency_ledger"} <= tables)
            finally:
                store.close()
            backups = list(Path(folder).glob("old.sqlite3.pre-v12-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            with closing(sqlite3.connect(backups[0])) as db:
                self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 11)
                self.assertNotIn("currency_ledger", {r[0] for r in db.execute("SELECT name FROM sqlite_master")})

    def test_reopening_a_v12_file_makes_no_second_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = v11_file(folder)
            Store(path).close()
            Store(path).close()
            self.assertEqual(len(list(Path(folder).glob("*.pre-*"))), 1)

    def test_v15_file_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new.sqlite3"
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("PRAGMA user_version=15")
            with self.assertRaisesRegex(ValueError, "newer than this application"):
                Store(path)
            self.assertEqual(list(Path(folder).glob("*.pre-*")), [])


class CurrencyCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store

    def tearDown(self):
        self.store.close()

    async def run_command(self, path, *args, admin=True, user_id=9, **kwargs):
        interaction = FakeInteraction(admin=admin, user_id=user_id)
        await invoke(self.bot, path, interaction, *args, **kwargs)
        return interaction

    def assert_private(self, interaction):
        sent = interaction.response.sent + interaction.followup.sent
        self.assertEqual(len(sent), 1)
        self.assertIs(sent[0][1].get("ephemeral"), True)

    async def test_balance_shows_own_and_others(self):
        self.store.change_balance(1, 9, 1234, "x", 1)
        self.store.change_balance(1, 5, 7, "x", 1)
        own = await self.run_command("balance", admin=False)
        self.assertEqual(own.replies, ["You have 1,234 coins."])
        self.assert_private(own)
        other = await self.run_command("balance", member(5), admin=False)
        self.assertEqual(other.replies, ["<@5> has 7 coins."])
        self.assert_private(other)
        self.assertIsNotNone(other.response.sent[0][1].get("allowed_mentions"))
        self.assertEqual((await self.run_command("balance", member(9), admin=False)).replies, ["You have 1,234 coins."])
        self.assertEqual((await self.run_command("balance", member(77), admin=False)).replies, ["<@77> has 0 coins."])

    async def test_grant_in_a_dm_is_refused(self):
        interaction = FakeInteraction(guild_id=None, channel_id=555, admin=True)
        await invoke(self.bot, "admin currency grant", interaction, member(5), 10, "x")
        self.assertIn("server", interaction.replies[0].lower())
        self.assertEqual(self.store.ledger(1), [])

    async def test_balance_in_dm_is_refused(self):
        interaction = FakeInteraction(guild_id=None, channel_id=555)
        await invoke(self.bot, "balance", interaction)
        self.assertIn("server", interaction.replies[0].lower())

    async def test_grant_and_revoke_confirm_with_new_balance(self):
        grant = await self.run_command("admin currency grant", member(5), 150, "welcome")
        self.assertEqual(grant.replies, ["Gave 150 coins to <@5>. New balance: 150 coins."])
        self.assert_private(grant)
        self.assertIsNotNone(grant.response.sent[0][1].get("allowed_mentions"))
        revoke = await self.run_command("admin currency revoke", member(5), 20, "fine")
        self.assertEqual(revoke.replies, ["Took 20 coins from <@5>. New balance: 130 coins."])
        self.assertEqual(self.store.balance(1, 5), 130)
        ledger = self.store.ledger(1, 5)
        self.assertEqual([(r["amount"], r["actor_id"], r["reason"]) for r in ledger], [(-20, 9, "fine"), (150, 9, "welcome")])

    async def test_revoke_below_zero_is_refused(self):
        await self.run_command("admin currency grant", member(5), 10, "start")
        refused = await self.run_command("admin currency revoke", member(5), 11, "too much")
        self.assertEqual(refused.replies, ["That would leave <@5> with a negative balance (current balance: 10 coins)."])
        self.assertEqual(self.store.balance(1, 5), 10)
        self.assertEqual(len(self.store.ledger(1)), 1)

    async def test_bots_cannot_hold_a_balance(self):
        refused = await self.run_command("admin currency grant", member(5, bot=True), 10, "x")
        self.assertEqual(refused.replies, ["Bots cannot hold a balance."])
        self.assertEqual(self.store.ledger(1), [])

    async def test_blank_reason_is_refused(self):
        refused = await self.run_command("admin currency grant", member(5), 10, "   ")
        self.assertEqual(refused.replies, ["The reason must be 1 to 200 characters."])
        self.assertEqual(self.store.balance(1, 5), 0)

    async def test_name_command_renames_and_replies_use_it(self):
        renamed = await self.run_command("admin currency name", " gold ")
        self.assertEqual(renamed.replies, ["This server's currency is now called gold."])
        self.assert_private(renamed)
        grant = await self.run_command("admin currency grant", member(5), 1, "x")
        self.assertEqual(grant.replies, ["Gave 1 gold to <@5>. New balance: 1 gold."])
        self.assertEqual((await self.run_command("balance", admin=False, user_id=5)).replies, ["You have 1 gold."])

    async def test_bad_name_is_refused_through_the_error_handler(self):
        refused = await self.run_command("admin currency name", "x" * 40)
        self.assertEqual(refused.replies, ["The currency name must be 1 to 32 characters on one line."])
        self.assertEqual(self.store.currency_name(1), "coins")

    async def test_admin_commands_reject_non_admins(self):
        for path, args in (("admin currency grant", (member(5), 1, "x")), ("admin currency revoke", (member(5), 1, "x")), ("admin currency name", ("gold",))):
            with self.subTest(path=path):
                interaction = await self.run_command(path, *args, admin=False)
                self.assertEqual(interaction.replies, ["Only server administrators can use that command."])
        self.assertEqual(self.store.balance(1, 5), 0)
        self.assertEqual(self.store.currency_name(1), "coins")


if __name__ == "__main__":
    unittest.main()
