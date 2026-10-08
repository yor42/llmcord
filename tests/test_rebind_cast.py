"""Rebinding a channel and unlinking a hub world keep or prune the cast, never silently reset it (UX-03; R4 step 4).

Target behavior:
- ``Store.bind_channel`` to the space the channel is already bound to keeps ``default_cast``, ``active_cast`` and
  ``ambient`` exactly. To a different space it prunes both casts to the characters eligible there (order kept) and
  keeps ``ambient``. It returns the dropped characters (ids or rows; an empty list when nothing was dropped).
- ``/admin space bind`` says "already bound ... kept" for the same space, names the dropped characters for a new
  space, and says the cast was kept when nothing was dropped.
- ``Store.unlink_world`` prunes now-ineligible characters from ``default_cast`` and ``active_cast`` of every channel
  bound to that hub in the same guild and returns the number of pruned entries (one per id removed from one cast
  list). ``/admin space unlink_world`` mentions that number. Other spaces and other guilds are untouched.
- The dashboard's bind/unlink buttons call the same store methods (the legacy web routes were retired in SEC-02).

Out of scope (gap): thread casts (``thread_casts`` has no parent column), pinned below as a characterization.
Seams: store methods and slash commands via ``helpers.invoke``.
"""
import json
import unittest
from types import SimpleNamespace

from helpers import FakeInteraction, FakeThread, invoke, make_settings

from llmcord_core.discord_bot import SkitBot
from llmcord_core.store import Store


def dropped_ids(result) -> list[int]:
    """``bind_channel``'s dropped characters as ids, whether it returns ids or rows."""
    return [item if isinstance(item, int) else item["id"] for item in result]


class World:
    """Guild 1: worlds Harbor (Alice, Bob) and Forest (Carol, Dave); hub Plaza links both; hub Square links Harbor.
    Guild 2: world Faraway (Zara), hub Yonder linking it."""

    def build(self, store):
        self.store = store
        self.harbor = store.create_space(1, "Harbor", "world")
        self.forest = store.create_space(1, "Forest", "world")
        self.plaza = store.create_space(1, "Plaza", "hub")
        self.square = store.create_space(1, "Square", "hub")
        store.link_world(1, self.plaza, self.harbor)
        store.link_world(1, self.plaza, self.forest)
        store.link_world(1, self.square, self.harbor)
        add = lambda guild, world, name: store.add_character(guild, world, name, {"name": name}, None, [])  # noqa: E731
        self.alice, self.bob = add(1, self.harbor, "Alice"), add(1, self.harbor, "Bob")
        self.carol, self.dave = add(1, self.forest, "Carol"), add(1, self.forest, "Dave")
        self.faraway = store.create_space(2, "Faraway", "world")
        self.yonder = store.create_space(2, "Yonder", "hub")
        store.link_world(2, self.yonder, self.faraway)
        self.zara = add(2, self.faraway, "Zara")

    def casts(self, channel_id):
        row = self.store.channel(channel_id)
        return json.loads(row["default_cast"]), json.loads(row["active_cast"])

    def set_casts(self, channel_id, default, active, ambient=True):
        self.store.set_cast(channel_id, None, default, default=True)
        self.store.set_cast(channel_id, None, active)
        self.store.set_ambient(channel_id, ambient)


class StoreRebindTests(World, unittest.TestCase):
    def setUp(self):
        self.build(Store())

    def tearDown(self):
        self.store.close()

    def test_rebind_same_space_keeps_cast_and_ambient(self):
        """UX-03: rebinding a channel to its current space keeps both casts and ambient exactly and drops nobody
        (currently the upsert resets them to ``[]``/``[]``/0 and returns ``None``)."""
        self.store.bind_channel(1, 100, self.harbor)
        self.set_casts(100, [self.bob, self.alice], [self.alice])
        result = self.store.bind_channel(1, 100, self.harbor)
        self.assertEqual(self.casts(100), ([self.bob, self.alice], [self.alice]))
        self.assertEqual(self.store.channel(100)["ambient"], 1)
        self.assertEqual(self.store.channel(100)["space_id"], self.harbor)
        self.assertEqual(dropped_ids(result), [])

    def test_rebind_to_other_space_prunes_in_order_and_keeps_ambient(self):
        """UX-03: Plaza -> Harbor drops Forest's Carol from both casts, keeps the remaining order and ambient, and
        returns Carol once (currently both casts are emptied and ambient switched off)."""
        self.store.bind_channel(1, 100, self.plaza)
        self.set_casts(100, [self.bob, self.carol, self.alice], [self.carol, self.bob])
        result = self.store.bind_channel(1, 100, self.harbor)
        self.assertEqual(self.store.channel(100)["space_id"], self.harbor)
        self.assertEqual(self.casts(100), ([self.bob, self.alice], [self.bob]))
        self.assertEqual(self.store.channel(100)["ambient"], 1)
        self.assertEqual(dropped_ids(result), [self.carol])

    def test_rebind_to_wider_space_keeps_everything(self):
        """UX-03: Harbor -> Plaza (which links Harbor) drops nobody: casts and ambient are kept and the result is empty."""
        self.store.bind_channel(1, 100, self.harbor)
        self.set_casts(100, [self.alice, self.bob], [self.bob])
        result = self.store.bind_channel(1, 100, self.plaza)
        self.assertEqual(self.store.channel(100)["space_id"], self.plaza)
        self.assertEqual(self.casts(100), ([self.alice, self.bob], [self.bob]))
        self.assertEqual(self.store.channel(100)["ambient"], 1)
        self.assertEqual(dropped_ids(result), [])

    def test_first_bind_starts_empty(self):
        """Regression (UX-03): a never-bound channel starts with empty casts and ambient off."""
        self.store.bind_channel(1, 100, self.harbor)
        self.assertEqual(self.casts(100), ([], []))
        self.assertEqual(self.store.channel(100)["ambient"], 0)

    def test_rebind_rejects_other_guild_and_leaves_cast(self):
        """Regression (UX-03, guild isolation): guild 2 cannot rebind guild 1's channel, and guild 1 cannot bind to
        guild 2's space; either way the channel's binding and casts are unchanged."""
        self.store.bind_channel(1, 100, self.harbor)
        self.set_casts(100, [self.alice], [self.alice, self.bob])
        with self.assertRaises(ValueError):
            self.store.bind_channel(2, 100, self.faraway)
        with self.assertRaises(ValueError):
            self.store.bind_channel(1, 100, self.faraway)
        self.assertEqual(self.store.channel(100)["space_id"], self.harbor)
        self.assertEqual(self.store.channel(100)["guild_id"], 1)
        self.assertEqual(self.casts(100), ([self.alice], [self.alice, self.bob]))
        self.assertEqual(self.store.channel(100)["ambient"], 1)


class StoreUnlinkTests(World, unittest.TestCase):
    def setUp(self):
        self.build(Store())
        self.store.bind_channel(1, 200, self.plaza)
        self.store.bind_channel(1, 201, self.plaza)
        self.store.bind_channel(1, 100, self.harbor)
        self.store.bind_channel(1, 300, self.square)
        self.store.bind_channel(2, 400, self.yonder)
        self.set_casts(200, [self.alice, self.carol], [self.carol, self.alice])
        self.set_casts(201, [], [self.alice])
        self.set_casts(100, [self.alice], [self.alice, self.bob])
        self.set_casts(300, [self.bob], [self.alice])
        self.set_casts(400, [self.zara], [self.zara])

    def tearDown(self):
        self.store.close()

    def test_unlink_world_prunes_hub_channel_casts(self):
        """UX-03: unlinking Harbor from Plaza removes Alice from both casts of every Plaza channel, keeps Carol and the
        order, and returns 3 (200 default, 200 active, 201 active). Currently Alice stays as a stale member and the
        call returns ``None``."""
        pruned = self.store.unlink_world(1, self.plaza, self.harbor)
        self.assertEqual(self.store.allowed_worlds(self.plaza), {self.forest})
        self.assertEqual(self.casts(200), ([self.carol], [self.carol]))
        self.assertEqual(self.casts(201), ([], []))
        self.assertEqual(pruned, 3)

    def test_unlink_world_leaves_other_spaces_and_guilds(self):
        """Regression (UX-03, guild isolation): unlinking Harbor from Plaza never touches channels bound to Harbor itself,
        to another hub (Square still links Harbor), or in guild 2, nor ambient anywhere."""
        self.store.unlink_world(1, self.plaza, self.harbor)
        self.assertEqual(self.casts(100), ([self.alice], [self.alice, self.bob]))
        self.assertEqual(self.casts(300), ([self.bob], [self.alice]))
        self.assertEqual(self.casts(400), ([self.zara], [self.zara]))
        for channel_id in (100, 200, 201, 300, 400):
            self.assertEqual(self.store.channel(channel_id)["ambient"], 1, channel_id)

    def test_unlink_world_prune_is_scoped_to_the_guild(self):
        """Regression (UX-03, guild isolation): the prune selects channels by guild as well as hub. A channel row in
        guild 2 pointing at guild 1's Plaza (not reachable through ``bind_channel``; written directly) keeps its cast."""
        self.store.execute("""INSERT INTO channels(channel_id,guild_id,space_id,default_cast,active_cast)
                              VALUES(500,2,?,?,?)""", (self.plaza, json.dumps([self.alice]), json.dumps([self.alice])))
        self.store.unlink_world(1, self.plaza, self.harbor)
        self.assertEqual(self.casts(500), ([self.alice], [self.alice]))

    def test_unlink_world_returns_zero_when_nothing_pruned(self):
        """UX-03: unlinking Forest from Square (never linked, no Forest characters cast there) prunes nothing and returns
        0 (currently ``None``)."""
        self.assertEqual(self.store.unlink_world(1, self.square, self.forest), 0)
        self.assertEqual(self.casts(300), ([self.bob], [self.alice]))

    def test_unlink_world_rejects_other_guild_and_leaves_cast(self):
        """Regression (UX-03, guild isolation): guild 2 cannot unlink guild 1's hub world; links and casts stay."""
        with self.assertRaises(ValueError):
            self.store.unlink_world(2, self.plaza, self.harbor)
        self.assertEqual(self.store.allowed_worlds(self.plaza), {self.harbor, self.forest})
        self.assertEqual(self.casts(200), ([self.alice, self.carol], [self.carol, self.alice]))

    def test_unlink_world_rejects_a_non_hub(self):
        """Regression (UX-03): unlinking from a world id (not a hub) is refused and prunes nothing."""
        self.store.bind_channel(1, 101, self.harbor)
        self.set_casts(101, [self.alice], [self.alice])
        with self.assertRaises(ValueError):
            self.store.unlink_world(1, self.harbor, self.forest)
        self.assertEqual(self.casts(101), ([self.alice], [self.alice]))

    def test_unlink_world_leaves_thread_casts(self):
        """Characterization (UX-03, gap): thread casts are not pruned by ``unlink_world`` because ``thread_casts`` has no
        parent column to find the hub's threads; a Plaza thread keeps Alice after Harbor is unlinked."""
        self.store.set_cast(250, 200, [self.alice, self.carol])
        self.store.unlink_world(1, self.plaza, self.harbor)
        self.assertEqual(self.store.get_cast(250, 200), [self.alice, self.carol])


class SlashRebindTests(World, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.build(self.bot.store)

    def tearDown(self):
        self.store.close()

    async def bind(self, space):
        interaction = FakeInteraction(admin=True)
        await invoke(self.bot, "admin space bind", interaction, SimpleNamespace(id=100, mention="<#100>"), space)
        return interaction.replies[0] if interaction.replies else ""

    async def test_bind_same_space_says_already_bound_and_kept(self):
        """UX-03: ``/admin space bind`` to the current space keeps the setup and says it is already bound and kept
        (currently it replies "Bound <#100> to world Harbor." after resetting casts and ambient)."""
        self.store.bind_channel(1, 100, self.harbor)
        self.set_casts(100, [self.bob, self.alice], [self.alice])
        reply = await self.bind("Harbor")
        self.assertEqual(self.casts(100), ([self.bob, self.alice], [self.alice]), reply)
        self.assertEqual(self.store.channel(100)["ambient"], 1)
        self.assertIn("already bound", reply.lower())
        self.assertIn("kept", reply.lower())

    async def test_bind_other_space_names_dropped_characters(self):
        """UX-03: Plaza -> Harbor prunes Carol and the reply names her (currently both casts are emptied silently)."""
        self.store.bind_channel(1, 100, self.plaza)
        self.set_casts(100, [self.carol, self.alice], [self.alice, self.carol])
        reply = await self.bind("Harbor")
        self.assertEqual(self.store.channel(100)["space_id"], self.harbor, reply)
        self.assertEqual(self.casts(100), ([self.alice], [self.alice]), reply)
        self.assertEqual(self.store.channel(100)["ambient"], 1)
        self.assertIn("Carol", reply)

    async def test_bind_other_space_says_cast_kept_when_nothing_dropped(self):
        """UX-03: Harbor -> Plaza drops nobody; casts are kept and the reply says so."""
        self.store.bind_channel(1, 100, self.harbor)
        self.set_casts(100, [self.alice], [self.bob, self.alice])
        reply = await self.bind("Plaza")
        self.assertEqual(self.store.channel(100)["space_id"], self.plaza, reply)
        self.assertEqual(self.casts(100), ([self.alice], [self.bob, self.alice]), reply)
        self.assertIn("kept", reply.lower())

    async def test_unlink_world_prunes_and_reports_count(self):
        """UX-03: ``/admin space unlink_world Plaza Harbor`` prunes Alice from Plaza channels and the reply mentions
        the 3 removed entries (currently Alice stays and the reply has no count)."""
        self.store.bind_channel(1, 200, self.plaza)
        self.store.bind_channel(1, 201, self.plaza)
        self.set_casts(200, [self.alice, self.carol], [self.carol, self.alice])
        self.set_casts(201, [], [self.alice])
        interaction = FakeInteraction(admin=True)
        await invoke(self.bot, "admin space unlink_world", interaction, "Plaza", "Harbor")
        reply = interaction.replies[0]
        self.assertEqual(self.store.allowed_worlds(self.plaza), {self.forest}, reply)
        self.assertEqual(self.casts(200), ([self.carol], [self.carol]), reply)
        self.assertEqual(self.casts(201), ([], []), reply)
        self.assertRegex(reply, r"\b3\b")

    async def test_cast_add_works_after_unlink_world(self):
        """UX-03: after Harbor is unlinked from Plaza, ``/cast add Dave`` in a Plaza channel succeeds. Currently the
        stale Alice stays in the active cast, so ``set_cast`` rejects the whole new cast as ineligible."""
        self.store.bind_channel(1, 200, self.plaza)
        self.store.set_cast(200, None, [self.alice, self.carol])
        await invoke(self.bot, "admin space unlink_world", FakeInteraction(admin=True), "Plaza", "Harbor")
        interaction = FakeInteraction(channel_id=200)
        await invoke(self.bot, "cast add", interaction, "Dave")
        self.assertEqual(self.store.get_cast(200), [self.carol, self.dave], interaction.replies)
        self.assertIn("Added Dave", interaction.replies[0])

    async def test_bind_in_thread_still_targets_the_named_channel(self):
        """Regression (UX-03): the bind command acts on its ``channel`` option, not on where it was run."""
        interaction = FakeInteraction(admin=True, channel=FakeThread(101, parent_id=100))
        await invoke(self.bot, "admin space bind", interaction, SimpleNamespace(id=150, mention="<#150>"), "Harbor")
        self.assertEqual(self.store.channel(150)["space_id"], self.harbor, interaction.replies)
        self.assertIsNone(self.store.channel(101))


if __name__ == "__main__":
    unittest.main()
