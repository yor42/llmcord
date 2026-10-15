"""Vault tools as shop items (FEAT-64 phase A, decision D43). Written before the implementation; they fail until it exists.

CONTRACT
  items registry: a new kind ``vault_tool`` (``items.get_kind("vault_tool")``, audience "none" like the other items).
  Five built-in recipes in ``items/recipes/``: the keys of ``vault.TOOLS``; each consumable, kind ``vault_tool``, a positive
  ``default_price`` (the Master Decoder is the dearest), an English and a Korean name. They load through the normal catalog,
  so they appear in ``items.all_items()`` after the five toy items, in ``shop_catalog`` (not for sale until an admin allows it)
  and, once held, in ``inventory``. They are never rolled.
  items_store: ``consume_item(guild_id, user_id, channel_id, item_key, *, now=None, result=None) -> {"qty": left, "use_id": id}``
      The one primitive that takes one held unit inside the CALLER's transaction: it never begins or commits (the caller's
      ``write_admin()`` does), it decrements the member's own inventory row, sets last_used_at=now, and writes one ``item_uses``
      row (``result`` is a JSON-able dict stored as the use result). Nothing held -> ``NotHeld``; nothing is written then.
  ``use_item`` for a vault_tool raises an ``ItemError`` with ``kind == "use_elsewhere"`` (code ``item.use_vault``) before any
      write; the ``/use`` command turns it into a private reply that points to ``/vault personal``. The inventory menu shows the
      held tools but offers no Use button for them.
  ``/shop`` field text for a tool: a plain description that names its turn cost ("1 turn" / "2 turns").
"""
import unittest

from helpers import FakeInteraction, invoke

from llmcord_core import items
from llmcord_core.items_store import ItemError, NotHeld
from llmcord_core.store import Store
from llmcord_core.vault import TOOLS

from test_items_discord import G, G2, U, ItemsBase

KEYS = tuple(TOOLS)
SCANNER, DECODER = "vault_residual_scanner", "vault_master_decoder"
CH = 100


class CatalogTests(unittest.TestCase):
    def test_the_five_recipes_load_with_the_vault_tool_kind(self):
        by_key = {r.key: r for r in items.all_items()}
        self.assertLessEqual(set(KEYS), set(by_key))
        self.assertEqual(items.get_kind("vault_tool").key, "vault_tool")
        for key in KEYS:
            with self.subTest(key=key):
                recipe = by_key[key]
                self.assertEqual((recipe.kind, recipe.consumable, recipe.audience), ("vault_tool", True, "none"))
                self.assertIsInstance(recipe.default_price, int)
                self.assertGreaterEqual(recipe.default_price, 1)
                self.assertTrue(recipe.names["en"].strip() and recipe.names["ko"].strip())
                self.assertEqual(items.display_name(recipe, "ko"), recipe.names["ko"])

    def test_master_decoder_is_the_dearest_tool(self):
        price = {r.key: r.default_price for r in items.all_items()}
        self.assertTrue(all(price[DECODER] > price[k] for k in KEYS if k != DECODER))

    def test_toy_items_keep_their_place_first(self):
        keys = [r.key for r in items.all_items()]
        self.assertEqual(keys[:5], ["8ball", "dice", "d20", "fortune_cookie", "tarot"])
        self.assertEqual(set(keys[5:]), set(KEYS))

    def test_recipe_with_the_kind_validates_and_wrong_audience_is_rejected(self):
        recipe = next(r for r in items.all_items() if r.key == SCANNER)
        data = {"key": "vault_probe_copy", "kind": "vault_tool", "name": "Probe", "consumable": True, "default_price": recipe.default_price}
        self.assertEqual(items.validate_recipe(data).kind, "vault_tool")
        with self.assertRaises(items.RecipeError):
            items.validate_recipe({**data, "audience": "member"})


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)
        self.store.change_balance(1, 10, 5000, "start", 1)

    def give(self, key, qty=1, user=10, guild=1):
        self.store.db.execute("INSERT INTO inventory(guild_id,user_id,item_key,qty) VALUES(?,?,?,?) ON CONFLICT(guild_id,user_id,item_key) DO UPDATE SET qty=excluded.qty", (guild, user, key, qty))
        self.store.db.commit()

    def test_shop_lists_them_off_sale_and_an_admin_can_sell_and_a_member_buy(self):
        catalog = {e["key"]: e for e in self.store.shop_catalog(1)}
        self.assertTrue(all(catalog[k]["kind"] == "vault_tool" and catalog[k]["consumable"] and not catalog[k]["for_sale"] for k in KEYS))
        self.store.save_shop_item(1, SCANNER, expected_revision=0, for_sale=True)
        bought = self.store.buy_item(1, 10, SCANNER)
        self.assertEqual((bought["qty"], self.store.item_qty(1, 10, SCANNER)), (1, 1))
        self.assertEqual(self.store.balance(1, 10), 5000 - bought["price"])
        self.assertEqual([i["key"] for i in self.store.inventory(1, 10)], [SCANNER])

    def test_use_item_refuses_a_vault_tool_and_changes_nothing(self):
        self.give(SCANNER, 2)
        with self.assertRaises(ItemError) as ctx:
            self.store.use_item(1, 10, CH, SCANNER)
        self.assertEqual(ctx.exception.kind, "use_elsewhere")
        self.assertEqual((self.store.item_qty(1, 10, SCANNER), self.store.item_use_log(1)), (2, []))

    def test_consume_item_takes_one_unit_inside_the_callers_transaction_and_logs_it(self):
        self.give(SCANNER, 2)
        with self.store.write_admin():
            out = self.store.consume_item(1, 10, CH, SCANNER, now=1234.5, result={"tool": SCANNER, "vault_session": 7})
            self.assertTrue(self.store.db.in_transaction, "the primitive must not commit")
        self.assertEqual(out["qty"], 1)
        self.assertEqual(self.store.item_qty(1, 10, SCANNER), 1)
        self.assertEqual(self.store.one("SELECT last_used_at FROM inventory WHERE guild_id=1 AND user_id=10 AND item_key=?", (SCANNER,))[0], 1234.5)
        log = self.store.item_use_log(1)
        self.assertEqual(len(log), 1)
        self.assertEqual((log[0]["id"], log[0]["item_key"], log[0]["user_id"], log[0]["channel_id"], log[0]["used_at"]), (out["use_id"], SCANNER, 10, CH, 1234.5))
        self.assertEqual(log[0]["result"], {"tool": SCANNER, "vault_session": 7})

    def test_consume_item_is_rolled_back_with_the_callers_transaction(self):
        self.give(SCANNER)
        with self.assertRaises(RuntimeError):
            with self.store.write_admin():
                self.store.consume_item(1, 10, CH, SCANNER, now=5.0, result={})
                raise RuntimeError("the caller failed after consuming")
        self.assertEqual((self.store.item_qty(1, 10, SCANNER), self.store.item_use_log(1)), (1, []))

    def test_consume_item_with_nothing_held_raises_and_writes_nothing(self):
        self.give(SCANNER, 0)
        for guild, user in ((1, 10), (1, 11), (2, 10)):
            with self.subTest(guild=guild, user=user):
                with self.assertRaises(NotHeld):
                    with self.store.write_admin():
                        self.store.consume_item(guild, user, CH, SCANNER, now=1.0, result={})
        self.assertEqual(self.store.item_use_log(1) + self.store.item_use_log(2), [])

    def test_consume_item_leaves_other_members_and_guilds_alone(self):
        self.give(SCANNER, 3)
        self.give(SCANNER, 3, user=11)
        self.give(SCANNER, 3, guild=2)
        with self.store.write_admin():
            self.store.consume_item(1, 10, CH, SCANNER, now=1.0, result={})
        self.assertEqual((self.store.item_qty(1, 10, SCANNER), self.store.item_qty(1, 11, SCANNER), self.store.item_qty(2, 10, SCANNER)), (2, 3, 3))
        self.assertEqual(self.store.item_use_log(2), [])


class DiscordTests(ItemsBase):
    def hold(self, key, qty=1):
        self.store.db.execute("INSERT INTO inventory(guild_id,user_id,item_key,qty) VALUES(?,?,?,?) ON CONFLICT(guild_id,user_id,item_key) DO UPDATE SET qty=excluded.qty", (G, U, key, qty))
        self.store.db.commit()

    async def run_cmd(self, path, *args, user_id=U, guild_id=G):
        interaction = FakeInteraction(guild_id=guild_id, user_id=user_id)
        await invoke(self.bot, path, interaction, *args)
        return interaction

    async def test_shop_describes_each_tool_with_its_turn_cost(self):
        for key in KEYS:
            self.store.save_shop_item(G, key, expected_revision=0, for_sale=True)
        shop = await self.run_cmd("shop")
        fields = {f.name: f.value for f in shop.response.sent[0][1]["embed"].fields}
        for key in KEYS:
            name = self.store.shop_item(G, key)["name"]
            value = next(v for n, v in fields.items() if name in n)
            with self.subTest(key=key):
                self.assertNotIn("A toy from the shop.", value)
                self.assertRegex(value, rf"\b{TOOLS[key].turn_cost} turns?\b")
                self.assertIn("consumable", value)
        self.assertRegex(next(v for n, v in fields.items() if self.store.shop_item(G, DECODER)["name"] in n), r"\b2 turns\b")

    async def test_use_points_to_vault_personal_and_keeps_the_item(self):
        self.hold(SCANNER, 2)
        out = await self.run_cmd("use", SCANNER)
        self.assertEqual(len(out.replies), 1)
        self.assertIn("/vault personal", out.replies[0])
        self.assertTrue(out.response.sent[0][1]["ephemeral"])
        self.assertEqual((self.store.item_qty(G, U, SCANNER), self.store.item_use_log(G)), (2, []))

    async def test_use_refusal_is_private_per_guild(self):
        self.hold(SCANNER)
        other = await self.run_cmd("use", SCANNER, guild_id=G2)
        self.assertNotIn("/vault personal", other.replies[0])  # not held in that guild: the ordinary not-held reply

    async def test_inventory_lists_tools_without_a_use_button(self):
        self.hold(SCANNER, 2)
        self.store.buy_item(G, U, "8ball")
        inv = await self.run_cmd("inventory")
        reply = inv.response.sent[0][1]
        name = self.store.shop_item(G, SCANNER)["name"]
        self.assertIn(f"{self.store.shop_item(G, SCANNER)['emoji']} {name} × 2".strip(), [f.name.strip() for f in reply["embed"].fields])
        labels = [b.label for b in reply["view"].children]
        self.assertTrue(any("Magic 8-ball" in label for label in labels))
        self.assertFalse(any(name in label for label in labels))


if __name__ == "__main__":
    unittest.main()
