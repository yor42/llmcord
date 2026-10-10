"""FEAT-25: schema v16 and character wallets (store API, migration, /balance character:)."""
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import discord
from helpers import FakeInteraction, autocomplete, invoke, make_settings

from llmcord_core.admin_store import ConflictError, CurrencyError
from llmcord_core.discord_bot import SkitBot
from llmcord_core.store import Store

G, H = 1, 2
MAX = Store.MAX_CURRENCY_BALANCE


def at(day, hour=12, month=10):
    return datetime(2026, month, day, hour, 0, tzinfo=timezone.utc).timestamp()


def member(ident):
    return discord.Object(id=ident)


class WalletCase(unittest.TestCase):
    def setUp(self):
        self.store = s = Store()
        self.w = s.create_space(G, "W", "world")
        self.ann = s.create_character(G, self.w, "Ann")
        self.bo = s.create_character(G, self.w, "Bo")
        self.w2 = s.create_space(H, "W2", "world")
        self.zed = s.create_character(H, self.w2, "Zed")
        s.set_daily_settings(G, 100, 50, 7)
        s.set_daily_settings(H, 0, 0, 7)

    def tearDown(self):
        self.store.close()

    def day(self, d):
        return at(d)

    def rows(self, cid=None):
        return self.store.ledger(G, character_id=cid or self.ann)


class RefillTests(WalletCase):
    def test_first_need_refills_once_with_one_ledger_row(self):
        entry = self.store.refill_character(G, self.ann, now=self.day(10))
        self.assertEqual((entry["amount"], entry["balance_after"], entry["source"], entry["reason"], entry["actor_id"], entry["holder_kind"]),
                         (100, 100, "refill", "Daily refill", 0, "character"))
        self.assertIsNone(self.store.refill_character(G, self.ann, now=self.day(10)))
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.store.character_balance(G, self.ann, now=self.day(10)), 100)

    def test_next_day_refills_again_and_missed_days_do_not_stack(self):
        self.store.refill_character(G, self.ann, now=self.day(10))
        self.store.refill_character(G, self.ann, now=self.day(11))
        self.assertEqual(self.store.character_balance(G, self.ann, now=self.day(11)), 200)
        self.store.refill_character(G, self.ann, now=self.day(20))
        self.assertEqual(self.store.character_balance(G, self.ann, now=self.day(20)), 300)

    def test_refill_stops_at_the_cap(self):
        self.store.set_refill_cap(G, 150)
        self.store.refill_character(G, self.ann, now=self.day(10))
        entry = self.store.refill_character(G, self.ann, now=self.day(11))
        self.assertEqual((entry["amount"], entry["balance_after"]), (50, 150))
        self.assertIsNone(self.store.refill_character(G, self.ann, now=self.day(12)))
        self.assertEqual(len(self.rows()), 2)

    def test_balance_above_cap_is_untouched_but_the_day_is_marked(self):
        self.store.set_refill_cap(G, 100)
        self.store.change_character_balance(G, self.ann, 500, "win", 7, now=self.day(10))
        self.assertEqual(self.store.character_balance(G, self.ann, now=self.day(10)), 600)
        self.assertEqual(self.store.one("SELECT refill_day FROM character_balances WHERE character_id=?", (self.ann,))["refill_day"], "2026-10-10")
        self.assertEqual(self.store.character_balance(G, self.ann, now=self.day(11)), 600)

    def test_daily_off_does_nothing_and_leaves_the_day_unset(self):
        self.assertIsNone(self.store.refill_character(H, self.zed, now=self.day(10)))
        self.assertEqual(self.store.character_balance(H, self.zed, now=self.day(10)), 0)
        self.assertIsNone(self.store.one("SELECT 1 FROM character_balances WHERE character_id=?", (self.zed,)))
        self.store.set_daily_settings(H, 40, 0, 7)
        entry = self.store.refill_character(H, self.zed, now=self.day(10))
        self.assertEqual(entry["amount"], 40)

    def test_cap_zero_never_refills(self):
        self.store.set_refill_cap(G, 0)
        self.assertIsNone(self.store.refill_character(G, self.ann, now=self.day(10)))
        self.assertEqual(self.store.character_balance(G, self.ann, now=self.day(10)), 0)
        self.assertEqual(self.rows(), [])

    def test_effective_balance_read_writes_nothing(self):
        before = self.store.db.total_changes
        self.assertEqual(self.store.character_balance(G, self.ann, now=self.day(10)), 100)
        self.assertEqual(self.store.db.total_changes, before)
        self.assertEqual(self.rows(), [])

    def test_server_day_follows_the_server_timezone(self):
        self.store.set_guild_timezone(G, "Asia/Seoul")
        self.store.refill_character(G, self.ann, now=at(10, 14))  # 23:00 on the 10th in Seoul
        self.assertIsNone(self.store.refill_character(G, self.ann, now=at(10, 14, )))
        self.assertIsNotNone(self.store.refill_character(G, self.ann, now=at(10, 16)))  # 01:00 on the 11th

    def test_refill_uses_the_amount_without_streak_bonus(self):
        self.assertEqual(self.store.refill_character(G, self.ann, now=self.day(10))["amount"], 100)
        self.assertEqual(self.store.refill_character(G, self.ann, now=self.day(11))["amount"], 100)


class ChangeTests(WalletCase):
    def test_change_applies_the_refill_first(self):
        entry = self.store.change_character_balance(G, self.ann, -30, "bet", 7, now=self.day(10))
        self.assertEqual((entry["amount"], entry["balance_after"], entry["holder_kind"], entry["user_id"]), (-30, 70, "character", self.ann))
        self.assertEqual([r["amount"] for r in self.rows()], [-30, 100])

    def test_refusals_name_the_character(self):
        self.store.set_daily_settings(G, 0, 0, 7)
        with self.assertRaisesRegex(CurrencyError, r"^That would leave Ann with a negative balance \(current balance: 0 coins\)\.$"):
            self.store.change_character_balance(G, self.ann, -1, "x", 7)
        self.store.change_character_balance(G, self.ann, 1000, "x", 7)
        with self.store.db:
            self.store.db.execute("UPDATE character_balances SET balance=? WHERE character_id=?", (MAX - 5, self.ann))
        with self.assertRaisesRegex(CurrencyError, r"^That would take Ann above the maximum balance"):
            self.store.change_character_balance(G, self.ann, 6, "x", 7)
        self.assertEqual(len(self.rows()), 1)

    def test_a_refused_change_rolls_back_the_refill(self):
        with self.assertRaises(CurrencyError):
            self.store.change_character_balance(G, self.ann, -500, "x", 7, now=self.day(10))
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.store.character_balance(G, self.ann, now=self.day(10)), 100)

    def test_bad_amount_and_reason(self):
        for bad in (0, True, 2.5, 10 ** 7):
            with self.assertRaises(CurrencyError):
                self.store.change_character_balance(G, self.ann, bad, "x", 7)
        with self.assertRaises(CurrencyError):
            self.store.change_character_balance(G, self.ann, 5, " ", 7)

    def test_another_guilds_character_is_refused_everywhere(self):
        message = r"^That character is not on this server\.$"
        for call in (lambda: self.store.change_character_balance(G, self.zed, 5, "x", 7),
                     lambda: self.store.character_balance(G, self.zed),
                     lambda: self.store.refill_character(G, self.zed)):
            with self.assertRaisesRegex(CurrencyError, message):
                call()
        with self.assertRaisesRegex(CurrencyError, message):
            self.store.change_character_balance(G, 999999, 5, "x", 7)
        self.assertEqual(self.store.ledger(H), [])

    def test_guild_two_is_unaffected_by_guild_one(self):
        self.store.set_refill_cap(G, 5)
        self.assertEqual(self.store.refill_cap(H), 1000)
        self.store.refill_character(G, self.ann, now=self.day(10))
        self.assertEqual(self.store.character_balances(H, now=self.day(10)), [])
        self.assertEqual(self.store.ledger(H), [])

    def test_game_helpers_inside_one_transaction(self):
        with self.store.write_admin():
            debit = self.store._character_debit_locked(G, self.ann, 40, "Blackjack bet", now=self.day(10))
            paid = self.store._character_credit_clamped_locked(G, self.ann, 90, "Blackjack payout", now=self.day(10))
        self.assertEqual((debit["amount"], debit["source"], debit["actor_id"], paid), (-40, "game", 0, 90))
        self.assertEqual(self.store.character_balance(G, self.ann, now=self.day(10)), 150)
        with self.assertRaises(CurrencyError):
            with self.store.write_admin():
                self.store._character_debit_locked(G, self.ann, 5000, "x", now=self.day(10))
        with self.store.db:
            self.store.db.execute("UPDATE character_balances SET balance=? WHERE character_id=?", (MAX - 10, self.ann))
        with self.store.write_admin():
            self.assertEqual(self.store._character_credit_clamped_locked(G, self.ann, 50, "w", now=self.day(10)), 10)
            self.assertEqual(self.store._character_credit_clamped_locked(G, self.ann, 50, "w", now=self.day(10)), 0)
        with self.assertRaises(CurrencyError):
            with self.store.write_admin():
                self.store._character_debit_locked(G, self.zed, 1, "x")


class HardeningTests(WalletCase):
    def test_game_helpers_refuse_bad_amounts(self):
        for bad in (0, -5, True, 2.5, "5", None):
            for call in (self.store._character_debit_locked, self.store._character_credit_clamped_locked):
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, r"^The amount must be a positive whole number\.$"):
                    with self.store.write_admin():
                        call(G, self.ann, bad, "x")
        self.assertEqual(self.rows(), [])

    def test_settings_are_read_once_per_listing(self):
        for name in ("Cy", "Di", "Ed"):
            cid = self.store.create_character(G, self.w, name)
            self.store.refill_character(G, cid, now=self.day(10))
        seen = []
        real = self.store.daily_settings
        self.store.daily_settings = lambda gid: seen.append(gid) or real(gid)
        self.assertEqual(len(self.store.character_balances(G, now=self.day(11))), 3)
        self.assertEqual(len(seen), 1)

    def test_a_deleted_characters_id_is_never_reused(self):
        self.store.change_character_balance(G, self.ann, 5, "x", 9, now=self.day(10))
        top = max(self.ann, self.bo)
        with self.store.write_admin():
            self.store.db.execute("DELETE FROM owner_revisions WHERE kind='character'")
            self.store.db.execute("DELETE FROM characters WHERE id=?", (self.ann,))
        new = self.store.create_character(G, self.w, "New")
        self.assertGreater(new, max(top, self.ann))
        self.assertEqual(self.store.ledger(G, character_id=new), [])


class CapTests(WalletCase):
    def test_default_set_and_validation(self):
        self.assertEqual(self.store.refill_cap(G), 1000)
        self.assertEqual(self.store.set_refill_cap(G, 0), 0)
        self.assertEqual(self.store.set_refill_cap(G, MAX), MAX)
        message = rf"^The refill cap must be a whole number from 0 to {MAX:,}\.$"
        for bad in (-1, MAX + 1, 2.5, "5", None, True):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, message):
                self.store.set_refill_cap(G, bad)

    def test_stale_expected_conflicts(self):
        self.store.set_refill_cap(G, 20)
        with self.assertRaises(ConflictError):
            self.store.set_refill_cap(G, 30, expected=1000)
        self.assertEqual(self.store.refill_cap(G), 20)
        self.assertEqual(self.store.set_refill_cap(G, 30, expected=20), 30)

    def test_daily_settings_are_unchanged(self):
        self.store.set_refill_cap(G, 20)
        self.assertEqual(self.store.daily_settings(G), {"amount": 100, "streak_bonus": 50, "streak_days": 7})


class SeparationTests(WalletCase):
    def test_equal_ids_keep_member_and_character_apart(self):
        self.store.change_balance(G, self.ann, 7, "member", 9)
        self.store.change_character_balance(G, self.ann, 5, "char", 9, now=self.day(10))
        self.assertEqual(self.store.balance(G, self.ann), 7)
        self.assertEqual(self.store.character_balance(G, self.ann, now=self.day(10)), 105)
        self.assertEqual([r["reason"] for r in self.store.ledger(G, user_id=self.ann)], ["member"])
        self.assertEqual([r["reason"] for r in self.store.ledger(G, character_id=self.ann)], ["char", "Daily refill"])
        both = self.store.ledger(G)
        self.assertEqual(sorted({r["holder_kind"] for r in both}), ["character", "member"])
        self.assertEqual([r["user_id"] for r in self.store.balances(G)], [self.ann])

    def test_reverse_a_character_entry(self):
        entry = self.store.change_character_balance(G, self.ann, 30, "win", 9, now=self.day(10))
        reversal = self.store.reverse_entry(G, entry["id"], 9, "undo")
        self.assertEqual((reversal["amount"], reversal["holder_kind"], reversal["user_id"], reversal["reverses_id"], reversal["balance_after"]), (-30, "character", self.ann, entry["id"], 100))
        self.assertEqual(self.store.balance(G, self.ann), 0)
        with self.assertRaisesRegex(CurrencyError, "already reversed"):
            self.store.reverse_entry(G, entry["id"], 9, "again")

    def test_reversal_refusal_names_the_character(self):
        entry = self.store.change_character_balance(G, self.ann, 300, "win", 9, now=self.day(10))
        self.store.change_character_balance(G, self.ann, -350, "bet", 9, now=self.day(10))
        with self.assertRaisesRegex(CurrencyError, r"^That would leave Ann with a negative balance"):
            self.store.reverse_entry(G, entry["id"], 9, "undo")

    def test_another_guild_cannot_reverse_it(self):
        entry = self.store.change_character_balance(G, self.ann, 30, "win", 9, now=self.day(10))
        with self.assertRaisesRegex(CurrencyError, "No such ledger entry"):
            self.store.reverse_entry(H, entry["id"], 9, "undo")

    def test_deleting_a_character_drops_the_wallet_but_keeps_the_ledger(self):
        entry = self.store.change_character_balance(G, self.ann, 30, "win", 9, now=self.day(10))
        self.store.delete_character(G, self.ann, self.store.owner_revision(G, "character", self.ann))
        self.assertIsNone(self.store.one("SELECT 1 FROM character_balances WHERE character_id=?", (self.ann,)))
        self.assertEqual(len(self.rows(self.ann)), 2)
        self.assertEqual(self.store.character_names(G, [self.ann, self.bo]), {self.bo: "Bo"})
        with self.assertRaisesRegex(CurrencyError, r"^That character was deleted, so this entry cannot be reversed\.$"):
            self.store.reverse_entry(G, entry["id"], 9, "undo")

    def test_character_balances_list(self):
        self.store.refill_character(G, self.ann, now=self.day(10))
        self.store.change_character_balance(G, self.bo, 500, "win", 9, now=self.day(10))
        rows = self.store.character_balances(G, now=self.day(10))
        self.assertEqual([(r["name"], r["balance"]) for r in rows], [("Bo", 600), ("Ann", 100)])
        self.assertEqual(set(rows[0]), {"character_id", "name", "archived", "balance", "updated_at"})
        self.assertEqual([r["name"] for r in self.store.character_balances(G, 1, 1, now=self.day(10))], ["Ann"])
        # a pending refill counts in the listed balance
        self.assertEqual([(r["name"], r["balance"]) for r in self.store.character_balances(G, now=self.day(11))], [("Bo", 700), ("Ann", 200)])
        self.assertEqual(self.store.character_balances(H, now=self.day(10)), [])


class MigrationTests(unittest.TestCase):
    def v15_file(self, folder):
        path = Path(folder) / "old.sqlite3"
        store = Store(path)
        store.set_currency_name(G, "gold")
        store.change_balance(G, 5, 40, "grant", 1)
        store.change_balance(G, 5, -10, "take", 1)
        store.close()
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("DROP INDEX currency_ledger_holder")
            db.execute("ALTER TABLE currency_ledger DROP COLUMN holder_kind")
            db.execute("DROP TABLE character_balances")
            db.execute("ALTER TABLE guild_settings DROP COLUMN character_refill_cap")
            db.execute("PRAGMA user_version=15")
        return path

    def test_v15_file_upgrades_with_a_backup_and_keeps_money(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.v15_file(folder)
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 16)
                self.assertEqual(store.balance(G, 5), 30)
                ledger = store.ledger(G, user_id=5)
                self.assertEqual([(r["amount"], r["holder_kind"]) for r in ledger], [(-10, "member"), (40, "member")])
                self.assertEqual(store.refill_cap(G), 1000)
                with self.assertRaisesRegex(sqlite3.DatabaseError, "append-only"):
                    store.db.execute("UPDATE currency_ledger SET reason='x'")
                store.db.rollback()
                store.set_daily_settings(G, 10, 0, 7)
                w = store.create_space(G, "W", "world")
                c = store.create_character(G, w, "Ann")
                self.assertEqual(store.refill_character(G, c, now=at(10))["balance_after"], 10)
            finally:
                store.close()
            backups = list(Path(folder).glob("old.sqlite3.pre-v16-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            with closing(sqlite3.connect(backups[0])) as db:
                self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 15)
                self.assertNotIn("character_balances", {r[0] for r in db.execute("SELECT name FROM sqlite_master")})

    def test_reopening_does_not_back_up_again_and_v17_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.v15_file(folder)
            Store(path).close()
            Store(path).close()
            self.assertEqual(len(list(Path(folder).glob("*.pre-*.sqlite3"))), 1)
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("PRAGMA user_version=17")
            with self.assertRaisesRegex(ValueError, "newer than this application"):
                Store(path)

    def test_the_ledger_columns_and_check_constraint(self):
        store = Store()
        try:
            columns = {r[1]: r for r in store.db.execute("PRAGMA table_info(currency_ledger)")}
            self.assertEqual(columns["holder_kind"][4], "'member'")
            with self.assertRaises(sqlite3.IntegrityError):
                store.db.execute("INSERT INTO currency_ledger(guild_id,user_id,amount,balance_after,reason,actor_id,source,created_at,holder_kind) VALUES(1,1,1,1,'x',0,'x',0,'robot')")
            with self.assertRaises(sqlite3.IntegrityError):
                store.db.execute("INSERT INTO character_balances(guild_id,character_id,balance,updated_at) VALUES(1,999,1,0)")
        finally:
            store.close()


class BalanceCommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = s = self.bot.store
        w = s.create_space(1, "W", "world")
        self.ann = s.create_character(1, w, "Ann")
        self.old = s.create_character(1, w, "Old")
        s.archive_character(1, self.old, True)
        w2 = s.create_space(2, "W2", "world")
        s.create_character(2, w2, "Zed")
        s.set_daily_settings(1, 80, 0, 7)

    def tearDown(self):
        self.store.close()

    async def run_balance(self, **kwargs):
        interaction = FakeInteraction(admin=False)
        await invoke(self.bot, "balance", interaction, **kwargs)
        return interaction

    async def test_character_balance_uses_the_effective_balance(self):
        interaction = await self.run_balance(character="Ann")
        self.assertEqual(interaction.replies, ["Ann has 80 coins."])
        sent = interaction.response.sent[0][1]
        self.assertIs(sent.get("ephemeral"), True)
        self.assertIsNotNone(sent.get("allowed_mentions"))
        self.assertEqual(self.store.ledger(1), [])  # reading does not write

    async def test_name_is_case_insensitive(self):
        self.assertEqual((await self.run_balance(character="ann")).replies, ["Ann has 80 coins."])

    async def test_member_and_character_together_are_refused(self):
        interaction = await self.run_balance(member=member(5), character="Ann")
        self.assertEqual(interaction.replies, ["Choose a member or a character, not both."])

    async def test_unknown_other_guild_and_archived_characters_are_not_found(self):
        for name in ("Nobody", "Zed", "Old"):
            interaction = await self.run_balance(character=name)
            self.assertEqual(interaction.replies, [f"No character named {name} in this server."], name)

    async def test_member_balance_is_unchanged(self):
        self.store.change_balance(1, 9, 12, "x", 1)
        self.assertEqual((await self.run_balance()).replies, ["You have 12 coins."])

    async def test_autocomplete_offers_this_servers_active_characters(self):
        choices = await autocomplete(self.bot, "balance", "character", FakeInteraction(), "")
        self.assertEqual([c.value for c in choices], ["Ann"])


class LedgerRowsTests(unittest.TestCase):
    def test_character_entries_show_the_character_name_or_the_deleted_label(self):
        from llmcord_core.currency_ui import ledger_rows
        base = {"amount": 5, "balance_after": 5, "reason": "r", "actor_id": 0, "source": "refill", "reverses_id": None, "created_at": 0}
        entries = [{**base, "id": 2, "user_id": 7, "holder_kind": "character"}, {**base, "id": 1, "user_id": 8, "holder_kind": "character"}, {**base, "id": 3, "user_id": 7, "holder_kind": "member"}]
        rows = ledger_rows(entries, {7: "Mia"}, "UTC", {7: "Ann"})
        self.assertEqual([r["member"] for r in rows], ["Ann", "Deleted character (#8)", "Mia"])
        self.assertEqual([r["can_reverse"] for r in rows], [True, False, True])


if __name__ == "__main__":
    unittest.main()
