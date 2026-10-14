"""FEAT-22 part B: schema v18 and the items/shop/inventory store API (buy, use, cooldown, hold cap, reversal hook)."""
import json
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from llmcord_core.admin_store import ConflictError, CurrencyError
from llmcord_core.items import RecipeError
from llmcord_core.items_store import (
    AlreadyOwned, HoldCapReached, InsufficientFunds, NotForSale, NotHeld, OnCooldown, UseDisabled,
)
from llmcord_core.store import Store

G, G2, U, U2, CH = 1, 2, 10, 11, 100
BALL, COOKIE, DICE = "8ball", "fortune_cookie", "dice"


def stock(store, guild=G, balance=1000, user=U, **sale):
    """Switch the three test items on for sale (unless `sale` says otherwise) and give `user` coins."""
    for key in (BALL, COOKIE, DICE):
        store.save_shop_item(guild, key, expected_revision=store.shop_item(guild, key)["revision"], for_sale=True, **sale)
    if balance:
        store.change_balance(guild, user, balance, "start", 1)


class SchemaTests(unittest.TestCase):
    def test_fresh_database_is_v20_with_the_item_tables(self):
        store = Store()
        self.addCleanup(store.close)
        self.assertEqual(store.one("PRAGMA user_version")[0], 22)
        names = {r[0] for r in store.all("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertLessEqual({"shop_items", "inventory", "shop_purchases", "item_uses"}, names)
        self.assertEqual(store.item_hold_cap(G), 10)

    def test_v17_file_upgrades_once_with_backup_and_keeps_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v17.sqlite3"
            store = Store(path)
            store.set_currency_name(G, "gold")
            store.close()
            with closing(sqlite3.connect(path)) as db, db:
                for table in ("shop_items", "inventory", "shop_purchases", "item_uses"):
                    db.execute(f"DROP TABLE {table}")
                db.execute("ALTER TABLE guild_settings DROP COLUMN item_hold_cap")
                db.execute("PRAGMA user_version=17")
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 22)
                self.assertEqual(len(list(Path(folder).glob("v17.sqlite3.pre-v18-*.sqlite3"))), 1)
                self.assertEqual(store.currency_name(G), "gold")
                self.assertEqual(store.item_hold_cap(G), 10)
                stock(store)
                self.assertEqual(store.buy_item(G, U, BALL)["qty"], 1)
            finally:
                store.close()
            backup = next(Path(folder).glob("v17.sqlite3.pre-v18-*.sqlite3"))
            with closing(sqlite3.connect(backup)) as old:
                self.assertEqual(old.execute("PRAGMA user_version").fetchone()[0], 17)
                self.assertIsNone(old.execute("SELECT 1 FROM sqlite_master WHERE name='inventory'").fetchone())
            Store(path).close()
            self.assertEqual(len(list(Path(folder).glob("*.pre-v18-*.sqlite3"))), 1)

    def test_newer_v22_file_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new.sqlite3"
            Store(path).close()
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("PRAGMA user_version=23")
            with self.assertRaises(ValueError):
                Store(path)

    def test_item_uses_is_append_only(self):
        store = Store()
        self.addCleanup(store.close)
        stock(store)
        store.buy_item(G, U, BALL)
        store.use_item(G, U, CH, BALL, seed="s", now=1000)
        for sql in ("UPDATE item_uses SET seed='x'", "DELETE FROM item_uses"):
            with self.assertRaises(sqlite3.IntegrityError):
                store.db.execute(sql)
            store.db.rollback()
        self.assertEqual(len(store.item_use_log(G)), 1)

    def test_shop_purchases_is_append_only_but_the_reversal_hook_still_works(self):
        store = Store()
        self.addCleanup(store.close)
        stock(store)
        bought = store.buy_item(G, U, COOKIE)
        for sql in ("UPDATE shop_purchases SET item_key='x'", "DELETE FROM shop_purchases"):
            with self.assertRaises(sqlite3.IntegrityError):
                store.db.execute(sql)
            store.db.rollback()
        self.assertTrue(store.reverse_entry(G, bought["ledger_id"], 1, "refund")["shop_item"]["removed"])


class ShopSettingsTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)

    def test_catalog_defaults_without_rows(self):
        catalog = {e["key"]: e for e in self.store.shop_catalog(G)}
        self.assertEqual(set(catalog), {BALL, COOKIE, DICE, "d20", "tarot"})
        ball = catalog[BALL]
        self.assertFalse(ball["for_sale"])
        self.assertTrue(ball["allow_use"])
        self.assertEqual((ball["price"], ball["revision"]), (ball["default_price"], 0))
        self.assertEqual(ball["cooldown_seconds"], ball["default_cooldown_seconds"])
        self.assertEqual(catalog[COOKIE]["cooldown_seconds"], 0)
        self.assertEqual(catalog[DICE]["effective_params"]["sides"], 6)

    def test_save_overrides_and_bumps_revision(self):
        saved = self.store.save_shop_item(G, DICE, expected_revision=0, for_sale=True, price=77, cooldown_seconds=3,
                                          params={"sides": 12}, extra_entries=None)
        self.assertEqual((saved["revision"], saved["price"], saved["cooldown_seconds"], saved["effective_params"]["sides"]), (1, 77, 3, 12))
        self.assertTrue(saved["for_sale"])
        again = self.store.save_shop_item(G, DICE, expected_revision=1, for_sale=False, allow_use=False)
        self.assertEqual((again["revision"], again["price"], again["allow_use"]), (2, again["default_price"], False))
        self.assertEqual(again["effective_params"]["sides"], 6)

    def test_revision_conflict(self):
        self.store.save_shop_item(G, DICE, expected_revision=0, for_sale=True)
        for stale in (0, 5):
            with self.assertRaises(ConflictError):
                self.store.save_shop_item(G, DICE, expected_revision=stale, for_sale=False)
        self.assertTrue(self.store.shop_item(G, DICE)["for_sale"])

    def test_validation_rejects_bad_input_and_writes_nothing(self):
        bad = [
            {"item_key": "nope"}, {"price": 0}, {"price": True}, {"price": 10 ** 9}, {"cooldown_seconds": -1}, {"cooldown_seconds": 10 ** 9},
            {"params": {"sides": 1}}, {"params": {"sides": "6"}}, {"params": {"colour": 3}}, {"for_sale": 1},
            {"extra_entries": {"en": [""]}}, {"extra_entries": {"fr": ["x"]}}, {"extra_entries": {"en": ["x"] * 600}},
            {"replace_entries": {"en": []}}, {"extra_entries": "text"},
        ]
        for case in bad:
            args = {"item_key": BALL, "for_sale": True, **case}
            key = args.pop("item_key")
            with self.subTest(case), self.assertRaises(ValueError):
                self.store.save_shop_item(G, key, expected_revision=0, **args)
        self.assertEqual(self.store.shop_item(G, BALL)["revision"], 0)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM shop_items")[0], 0)

    def test_entries_validated_at_save_and_stored(self):
        with self.assertRaises(RecipeError):
            self.store.save_shop_item(G, BALL, expected_revision=0, for_sale=True, extra_entries={"en": ["ok", 5]})
        saved = self.store.save_shop_item(G, BALL, expected_revision=0, for_sale=True, extra_entries={"en": ["Custom line"]})
        self.assertEqual(saved["extra_entries"], {"en": ["Custom line"]})

    def test_hold_cap_setting(self):
        self.assertEqual(self.store.set_item_hold_cap(G, 3), 3)
        self.assertEqual((self.store.item_hold_cap(G), self.store.item_hold_cap(G2)), (3, 10))
        for bad in (0, 100, True, "5", 2.5):
            with self.assertRaises(ValueError):
                self.store.set_item_hold_cap(G, bad)
        with self.assertRaises(ConflictError):
            self.store.set_item_hold_cap(G, 4, expected=10)
        self.assertEqual(self.store.set_item_hold_cap(G, 4, expected=3), 4)


class BuyTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)
        stock(self.store)

    def snapshot(self, guild=G):
        s = self.store
        return (s.balance(guild, U), s.one("SELECT COUNT(*) FROM currency_ledger")[0], s.one("SELECT COUNT(*) FROM inventory")[0],
                s.one("SELECT COUNT(*) FROM shop_purchases")[0])

    def test_expected_price_mismatch_refuses_without_writing(self):
        from llmcord_core.items_store import PriceChanged
        price = self.store.shop_item(G, BALL)["price"]
        before = self.snapshot()
        with self.assertRaises(PriceChanged) as caught:
            self.store.buy_item(G, U, BALL, expected_price=price - 1)
        self.assertEqual((caught.exception.code, caught.exception.price), ("price_changed", price))
        self.assertEqual(str(caught.exception), f"The price changed to {price:,}. Open /shop again.")
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.store.buy_item(G, U, BALL, expected_price=price)["price"], price)

    def test_happy_path(self):
        price = self.store.shop_item(G, BALL)["price"]
        bought = self.store.buy_item(G, U, BALL)
        self.assertEqual((bought["qty"], bought["price"], bought["balance"]), (1, price, 1000 - price))
        self.assertEqual(self.store.balance(G, U), 1000 - price)
        ledger = self.store.one("SELECT * FROM currency_ledger WHERE id=?", (bought["ledger_id"],))
        self.assertEqual((ledger["source"], ledger["amount"], ledger["reason"], ledger["user_id"]), ("shop", -price, ledger["reason"], U))
        self.assertTrue(ledger["reason"].startswith("Bought "))
        link = self.store.one("SELECT * FROM shop_purchases WHERE ledger_id=?", (bought["ledger_id"],))
        self.assertEqual((link["guild_id"], link["user_id"], link["item_key"]), (G, U, BALL))
        self.assertEqual([(i["key"], i["qty"]) for i in self.store.inventory(G, U)], [(BALL, 1)])

    def test_price_override_is_charged(self):
        self.store.save_shop_item(G, COOKIE, expected_revision=1, for_sale=True, price=40)
        self.assertEqual(self.store.buy_item(G, U, COOKIE)["balance"], 960)

    def test_rejected_paths_leave_nothing_behind(self):
        self.store.save_shop_item(G, DICE, expected_revision=1, for_sale=False)
        self.store.set_item_hold_cap(G, 1)
        self.store.buy_item(G, U, COOKIE)
        self.store.buy_item(G, U, BALL)
        before = self.snapshot()
        cases = [(NotForSale, DICE), (NotForSale, "nope"), (HoldCapReached, COOKIE), (AlreadyOwned, BALL)]
        for error, key in cases:
            with self.subTest(key), self.assertRaises(error):
                self.store.buy_item(G, U, key)
            self.assertEqual(self.snapshot(), before)
        self.store.change_balance(G, U, -self.store.balance(G, U), "empty", 1)
        before = self.snapshot()
        self.store.save_shop_item(G, "tarot", expected_revision=0, for_sale=True)
        with self.assertRaises(InsufficientFunds) as ctx:
            self.store.buy_item(G, U, "tarot")
        self.assertEqual((ctx.exception.balance, ctx.exception.code), (0, "insufficient"))
        self.assertEqual(self.snapshot(), before)

    def test_error_codes_are_distinct(self):
        codes = {c.code for c in (NotForSale, HoldCapReached, AlreadyOwned, InsufficientFunds, NotHeld, UseDisabled, OnCooldown)}
        self.assertEqual(len(codes), 7)

    def test_consumable_hold_cap_follows_the_setting(self):
        self.store.set_item_hold_cap(G, 3)
        for expected in (1, 2, 3):
            self.assertEqual(self.store.buy_item(G, U, COOKIE)["qty"], expected)
        with self.assertRaises(HoldCapReached) as ctx:
            self.store.buy_item(G, U, COOKIE)
        self.assertEqual(ctx.exception.cap, 3)
        self.assertEqual(self.store.buy_item(G, U2 if self.store.change_balance(G, U2, 100, "s", 1) else 0, COOKIE)["qty"], 1)

    def test_reusable_cap_is_one_whatever_the_setting(self):
        self.store.set_item_hold_cap(G, 99)
        self.store.buy_item(G, U, BALL)
        with self.assertRaises(AlreadyOwned):
            self.store.buy_item(G, U, BALL)

    def test_not_for_sale_after_switching_off_but_owners_keep_it(self):
        self.store.buy_item(G, U, BALL)
        self.store.save_shop_item(G, BALL, expected_revision=1, for_sale=False)
        with self.assertRaises(NotForSale):
            self.store.buy_item(G, U2, BALL)
        self.assertEqual(self.store.use_item(G, U, CH, BALL, seed="s", now=5)["qty"], 1)


class UseTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)
        stock(self.store, balance=5000)

    def test_consumable_decrements_and_logs(self):
        self.store.buy_item(G, U, COOKIE)
        self.store.buy_item(G, U, COOKIE)
        first = self.store.use_item(G, U, CH, COOKIE, seed="seed-a", now=100)
        self.assertEqual((first["qty"], first["seed"]), (1, "seed-a"))
        again = self.store.use_item(G, U, CH, COOKIE, seed="seed-a", now=100)
        self.assertEqual(again["result"], first["result"])
        self.assertEqual(again["qty"], 0)
        with self.assertRaises(NotHeld):
            self.store.use_item(G, U, CH, COOKIE, now=101)
        log = self.store.item_use_log(G)
        self.assertEqual([r["id"] for r in log], [again["use_id"], first["use_id"]])
        self.assertEqual(log[1]["result"], json.loads(json.dumps(first["result"].to_json())))
        self.assertEqual((log[1]["user_id"], log[1]["channel_id"], log[1]["item_key"], log[1]["seed"], log[1]["used_at"]), (U, CH, COOKIE, "seed-a", 100))
        self.assertEqual(self.store.inventory(G, U), [])

    def test_unseeded_use_draws_and_stores_a_seed(self):
        self.store.buy_item(G, U, COOKIE)
        out = self.store.use_item(G, U, CH, COOKIE)
        self.assertTrue(out["seed"])
        self.assertEqual(self.store.item_use_log(G)[0]["seed"], out["seed"])

    def test_reusable_cooldown(self):
        self.store.buy_item(G, U, BALL)
        cooldown = self.store.shop_item(G, BALL)["cooldown_seconds"]
        self.assertGreater(cooldown, 0)
        self.store.use_item(G, U, CH, BALL, seed="s", now=1000)
        with self.assertRaises(OnCooldown) as ctx:
            self.store.use_item(G, U, CH, BALL, seed="s", now=1000 + cooldown - 1.5)
        self.assertEqual(ctx.exception.remaining, 2)
        self.assertEqual(len(self.store.item_use_log(G)), 1)
        out = self.store.use_item(G, U, CH, BALL, seed="s", now=1000 + cooldown)
        self.assertEqual(out["qty"], 1)
        self.assertEqual(self.store.item_qty(G, U, BALL), 1)

    def test_cooldown_override_and_per_member(self):
        self.store.save_shop_item(G, BALL, expected_revision=1, for_sale=True, cooldown_seconds=0)
        self.store.change_balance(G, U2, 500, "s", 1)
        self.store.buy_item(G, U, BALL)
        self.store.buy_item(G, U2, BALL)
        self.store.use_item(G, U, CH, BALL, seed="s", now=1)
        self.store.use_item(G, U, CH, BALL, seed="s", now=1)
        self.store.save_shop_item(G, BALL, expected_revision=2, for_sale=True, cooldown_seconds=60)
        self.store.use_item(G, U, CH, BALL, seed="s", now=100)
        with self.assertRaises(OnCooldown):
            self.store.use_item(G, U, CH, BALL, seed="s", now=101)
        self.store.use_item(G, U2, CH, BALL, seed="s", now=101)

    def test_allow_use_off_blocks_and_nothing_changes(self):
        self.store.buy_item(G, U, COOKIE)
        self.store.save_shop_item(G, COOKIE, expected_revision=1, for_sale=True, allow_use=False)
        with self.assertRaises(UseDisabled):
            self.store.use_item(G, U, CH, COOKIE, seed="s")
        self.assertEqual(self.store.item_qty(G, U, COOKIE), 1)
        self.assertEqual(self.store.item_use_log(G), [])

    def test_a_failing_roll_writes_nothing(self):
        self.store.buy_item(G, U, COOKIE)
        with mock.patch("llmcord_core.items_store.items.roll", side_effect=RecipeError("boom")), self.assertRaises(RecipeError):
            self.store.use_item(G, U, CH, COOKIE, seed="s", now=1)
        self.assertEqual(self.store.item_qty(G, U, COOKIE), 1)
        self.assertEqual(self.store.item_use_log(G), [])
        self.assertIsNone(self.store.one("SELECT last_used_at FROM inventory WHERE item_key=?", (COOKIE,))["last_used_at"])

    def test_unheld_or_unknown_item_rejected(self):
        for key in (BALL, "nope"):
            with self.assertRaises(NotHeld):
                self.store.use_item(G, U, CH, key)

    def test_dice_uses_the_guilds_sides_and_entries(self):
        self.store.save_shop_item(G, DICE, expected_revision=1, for_sale=True, params={"sides": 2}, cooldown_seconds=0)
        self.store.buy_item(G, U, DICE)
        for n in range(8):
            text = self.store.use_item(G, U, CH, DICE, seed=f"s{n}", now=n)["result"].text
            self.assertTrue(text.endswith("(1-2)."), text)
        self.store.save_shop_item(G, BALL, expected_revision=1, for_sale=True, cooldown_seconds=0,
                                  replace_entries={"en": ["Only answer"]})
        self.store.buy_item(G, U, BALL)
        self.assertIn("Only answer", self.store.use_item(G, U, CH, BALL, seed="s", now=1)["result"].text)


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)

    def test_guilds_are_independent(self):
        stock(self.store, G, balance=1000)
        self.store.change_balance(G2, U, 1000, "s", 1)
        self.store.set_item_hold_cap(G, 2)
        self.assertEqual([e["for_sale"] for e in self.store.shop_catalog(G2)], [False] * 5)
        self.assertEqual(self.store.item_hold_cap(G2), 10)
        with self.assertRaises(NotForSale):
            self.store.buy_item(G2, U, BALL)
        self.store.buy_item(G, U, COOKIE)
        self.store.use_item(G, U, CH, COOKIE, seed="s", now=1)
        self.assertEqual(self.store.inventory(G2, U), [])
        self.assertEqual(self.store.item_use_log(G2), [])
        with self.assertRaises(NotHeld):
            self.store.use_item(G2, U, CH, COOKIE)
        stock(self.store, G2, balance=0)
        self.store.buy_item(G2, U, COOKIE)
        self.assertEqual((self.store.item_qty(G, U, COOKIE), self.store.item_qty(G2, U, COOKIE)), (0, 1))
        self.assertEqual(self.store.balance(G, U), 1000 - 25)
        self.assertEqual(self.store.balance(G2, U), 1000 - 25)
        self.assertEqual(len(self.store.item_use_log(G)), 1)

    def test_use_log_paging(self):
        stock(self.store, G, balance=1000)
        self.store.buy_item(G, U, DICE)
        self.store.save_shop_item(G, DICE, expected_revision=1, for_sale=True, cooldown_seconds=0)
        ids = [self.store.use_item(G, U, CH, DICE, seed=str(n), now=n)["use_id"] for n in range(5)]
        page = self.store.item_use_log(G, limit=2)
        self.assertEqual([r["id"] for r in page], [ids[4], ids[3]])
        older = self.store.item_use_log(G, limit=10, before_id=page[-1]["id"])
        self.assertEqual([r["id"] for r in older], [ids[2], ids[1], ids[0]])


class ConcurrencyTests(unittest.TestCase):
    def race(self, folder, prepare, action):
        path = Path(folder) / "shared.sqlite3"
        stores = [Store(path), Store(path)]
        try:
            stock(stores[0], balance=0)
            prepare(stores[0])
            results = []
            barrier = threading.Barrier(2, timeout=0.5)
            for store in stores:
                # Every buy/use reads the shop row first inside its transaction: both threads meet here, so without the
                # transaction lock both would read the old state. With it the second waits at BEGIN and the barrier times out.
                def meet(guild_id, item_key, _read=store._shop_row):
                    row = _read(guild_id, item_key)
                    try:
                        barrier.wait()
                    except threading.BrokenBarrierError:
                        pass
                    return row
                store._shop_row = meet

            def run(store):
                try:
                    action(store)
                    results.append("ok")
                except (CurrencyError, ValueError) as error:
                    results.append(getattr(error, "code", "refused"))

            threads = [threading.Thread(target=run, args=(s,)) for s in stores]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            return stores, sorted(results)
        except BaseException:
            for s in stores:
                s.close()
            raise

    def test_two_buys_with_balance_for_one(self):
        with tempfile.TemporaryDirectory() as folder:
            with closing(Store()) as scratch:
                price = scratch.shop_item(G, COOKIE)["price"]
            stores, results = self.race(folder, lambda s: s.change_balance(G, U, price + 5, "s", 1), lambda s: s.buy_item(G, U, COOKIE))
            try:
                self.assertEqual(results, ["insufficient", "ok"])
                self.assertEqual(stores[0].balance(G, U), 5)
                self.assertEqual(stores[1].item_qty(G, U, COOKIE), 1)
                self.assertEqual(stores[0].one("SELECT COUNT(*) FROM shop_purchases")[0], 1)
                self.assertEqual(stores[0].one("SELECT COUNT(*) FROM currency_ledger WHERE source='shop'")[0], 1)
            finally:
                for s in stores:
                    s.close()

    def test_two_buys_at_a_cap_of_one(self):
        with tempfile.TemporaryDirectory() as folder:
            def prepare(s):
                s.change_balance(G, U, 1000, "s", 1)
                s.set_item_hold_cap(G, 1)
            stores, results = self.race(folder, prepare, lambda s: s.buy_item(G, U, COOKIE))
            try:
                self.assertEqual(results, ["hold_cap", "ok"])
                self.assertEqual(stores[0].item_qty(G, U, COOKIE), 1)
                self.assertEqual(stores[0].balance(G, U), 1000 - stores[0].shop_item(G, COOKIE)["price"])
            finally:
                for s in stores:
                    s.close()

    def test_two_uses_of_the_last_consumable(self):
        with tempfile.TemporaryDirectory() as folder:
            def prepare(s):
                s.change_balance(G, U, 1000, "s", 1)
                s.buy_item(G, U, COOKIE)
            stores, results = self.race(folder, prepare, lambda s: s.use_item(G, U, CH, COOKIE))
            try:
                self.assertEqual(results, ["not_held", "ok"])
                self.assertEqual(stores[0].item_qty(G, U, COOKIE), 0)
                self.assertEqual(len(stores[0].item_use_log(G)), 1)
            finally:
                for s in stores:
                    s.close()

    def test_two_buys_of_a_reusable_item(self):
        with tempfile.TemporaryDirectory() as folder:
            stores, results = self.race(folder, lambda s: s.change_balance(G, U, 1000, "s", 1), lambda s: s.buy_item(G, U, BALL))
            try:
                self.assertEqual(results, ["already_owned", "ok"])
                self.assertEqual(stores[0].item_qty(G, U, BALL), 1)
                self.assertEqual(stores[0].one("SELECT COUNT(*) FROM shop_purchases")[0], 1)
            finally:
                for s in stores:
                    s.close()

    def test_two_uses_of_a_reusable_item_at_once(self):
        with tempfile.TemporaryDirectory() as folder:
            def prepare(s):
                s.change_balance(G, U, 1000, "s", 1)
                s.buy_item(G, U, BALL)
            stores, results = self.race(folder, prepare, lambda s: s.use_item(G, U, CH, BALL, seed="s", now=1000))
            try:
                self.assertEqual(results, ["ok", "on_cooldown"])
                self.assertEqual(len(stores[0].item_use_log(G)), 1)
                self.assertEqual(stores[0].item_qty(G, U, BALL), 1)
            finally:
                for s in stores:
                    s.close()


class ReversalTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)
        stock(self.store, balance=1000)

    def test_reversing_a_purchase_returns_coins_and_removes_the_item(self):
        bought = self.store.buy_item(G, U, COOKIE)
        out = self.store.reverse_entry(G, bought["ledger_id"], 1, "refund")
        self.assertEqual(self.store.balance(G, U), 1000)
        self.assertEqual(self.store.item_qty(G, U, COOKIE), 0)
        self.assertEqual((out["source"], out["reason"], out["shop_item"]["removed"]), ("reversal", "refund", True))
        with self.assertRaises(CurrencyError):
            self.store.reverse_entry(G, bought["ledger_id"], 1, "again")
        with self.assertRaises(CurrencyError):
            self.store.reverse_entry(G, out["id"], 1, "undo")
        self.assertEqual((self.store.balance(G, U), self.store.item_qty(G, U, COOKIE)), (1000, 0))

    def test_reversal_takes_one_of_several(self):
        first = self.store.buy_item(G, U, COOKIE)
        self.store.buy_item(G, U, COOKIE)
        self.store.reverse_entry(G, first["ledger_id"], 1, "refund")
        self.assertEqual(self.store.item_qty(G, U, COOKIE), 1)

    def test_reversal_after_use_returns_coins_only_and_says_so(self):
        bought = self.store.buy_item(G, U, COOKIE)
        self.store.use_item(G, U, CH, COOKIE, seed="s", now=1)
        out = self.store.reverse_entry(G, bought["ledger_id"], 1, "refund")
        self.assertEqual(self.store.balance(G, U), 1000)
        self.assertFalse(out["shop_item"]["removed"])
        self.assertIn("already used", out["reason"])
        self.assertTrue(out["reason"].startswith("refund"))
        self.assertEqual(self.store.one("SELECT reason FROM currency_ledger WHERE id=?", (out["id"],))["reason"], out["reason"])
        self.assertEqual(self.store.item_qty(G, U, COOKIE), 0)

    def test_long_reason_still_fits_with_the_note(self):
        bought = self.store.buy_item(G, U, COOKIE)
        self.store.use_item(G, U, CH, COOKIE, seed="s", now=1)
        out = self.store.reverse_entry(G, bought["ledger_id"], 1, "r" * 200)
        self.assertEqual(len(out["reason"]), 200)
        self.assertIn("already used", out["reason"])

    def test_other_sources_are_unchanged(self):
        grant = self.store.change_balance(G, U, 10, "gift", 1)["id"]
        self.store.buy_item(G, U, COOKIE)
        out = self.store.reverse_entry(G, grant, 1, "oops")
        self.assertNotIn("shop_item", out)
        self.assertEqual(out["reason"], "oops")
        self.assertEqual(self.store.item_qty(G, U, COOKIE), 1)

    def test_reversal_needs_the_same_guild(self):
        bought = self.store.buy_item(G, U, COOKIE)
        with self.assertRaises(CurrencyError):
            self.store.reverse_entry(G2, bought["ledger_id"], 1, "x")
        self.assertEqual(self.store.item_qty(G, U, COOKIE), 1)

    def test_reversal_failure_rolls_the_item_back_in(self):
        bought = self.store.buy_item(G, U, COOKIE)
        with mock.patch.object(self.store, "_append_ledger", side_effect=CurrencyError("boom")), self.assertRaises(CurrencyError):
            self.store.reverse_entry(G, bought["ledger_id"], 1, "refund")
        self.assertEqual(self.store.item_qty(G, U, COOKIE), 1)

if __name__ == "__main__":
    unittest.main()
