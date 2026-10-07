"""Shared name resolution, autocomplete and consistent option names for slash commands (UX-04, UX-06; R4 step 3).

Target behavior:
- One character resolver (``/cast set|add|remove``, ``/admin cast default``, ``/summon``, ``/character info``): an exact
  name wins; otherwise a unique case-insensitive match (casefold, surrounding whitespace stripped) resolves; several
  case-insensitive matches and no exact one is refused with a message listing the candidates; no match says
  "No character named X". Cast commands and ``/summon`` resolve among the characters eligible in the bound space,
  ``/character info`` across the guild. Another guild's characters never resolve.
- One space resolver (``/admin space bind|allow_world|disallow_world``, ``/admin character import``,
  ``/admin lore promote``) with the same rule, plus a kind check (hub / world) with a clear message.
- Autocomplete on every character and space option: at most 25 ``app_commands.Choice`` items, case-insensitive
  substring filter, guild-scoped (cast/summon: eligible characters; hub/world options: that kind), empty in a DM or,
  for options that need a binding, an unbound channel.
- Synced option names: ``character``, ``characters``, ``space``, ``hub``, ``world`` (``/admin space create`` keeps
  ``kind``/``name``).

Seams: slash commands via ``helpers.invoke`` (real checks and tree error handler; arguments passed positionally so the
resolver tests fail for the resolver, not for option renames), autocomplete via ``helpers.autocomplete`` (discord.py's
own ``Command._invoke_autocomplete`` dispatch with a ``FakeInteraction``), synced payloads via ``Command.to_dict``.
Outcomes are read back from the store (casts, bindings, hub links, imported characters, promoted lore) and replies.
"""
import json
import re
import unittest
from types import SimpleNamespace

from discord import app_commands
from helpers import FakeInteraction, FakeModels, FakeTextChannel, autocomplete, command, invoke, make_settings

from llmcord_core.discord_bot import SkitBot

# UX-06: synced option names per command, in order.
OPTIONS = {
    "character info": ["character"],
    "cast add": ["character"],
    "cast remove": ["character"],
    "summon": ["character", "prompt"],
    "cast set": ["characters"],
    "admin cast default": ["characters"],
    "admin space bind": ["channel", "space"],
    "admin lore promote": ["lore_id", "destination", "space"],
    "admin space allow_world": ["hub", "world"],
    "admin space disallow_world": ["hub", "world"],
    "admin character import": ["world", "attachment"],
    "admin space create": ["kind", "name"],
}
CHARACTER_PATHS = ["cast set", "cast add", "cast remove", "admin cast default", "summon", "character info"]
LIST_PATHS = ["cast set", "admin cast default"]  # comma lists; were already case-insensitive before UX-04
SINGLE_PATHS = ["cast add", "cast remove", "summon", "character info"]  # previously an exact name=? lookup
ELIGIBLE_SINGLE_PATHS = ["cast add", "cast remove", "summon"]
# (path, option) -> scope of the suggestions.
CHARACTER_OPTIONS = {("cast set", "characters"): "eligible", ("cast add", "character"): "eligible",
                     ("cast remove", "character"): "eligible", ("admin cast default", "characters"): "eligible",
                     ("summon", "character"): "eligible", ("character info", "character"): "guild"}
SPACE_OPTIONS = {("admin space bind", "space"): None, ("admin lore promote", "space"): None,
                 ("admin space allow_world", "hub"): "hub", ("admin space allow_world", "world"): "world",
                 ("admin space disallow_world", "hub"): "hub", ("admin space disallow_world", "world"): "world",
                 ("admin character import", "world"): "world"}


def card_attachment(name="Newcomer"):
    data = json.dumps({"spec": "chara_card_v2", "data": {"name": name}}).encode()

    async def read():
        return data

    return SimpleNamespace(filename="card.json", size=len(data), read=read)


class BotCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.bot.engine.models = self.bot.models = FakeModels()
        self.harbor = self.store.create_space(1, "Harbor", "world")
        self.other = self.store.create_space(1, "Other", "world")
        self.store.bind_channel(1, 100, self.harbor)
        self.channel = FakeTextChannel(100)

    def tearDown(self):
        self.store.close()

    def add(self, name, world=None, guild=1):
        return self.store.add_character(guild, world or self.harbor, name, {"name": name}, None, [])

    def names(self, ids):
        return [self.store.character_by_id(ident)["name"] for ident in ids]


class CharacterResolutionTests(BotCase):
    async def resolve(self, path, text):
        """Run ``path`` with ``text`` in bound channel 100; return (names it acted on, first reply)."""
        before = [row["id"] for row in self.store.eligible_characters(1, self.harbor)]
        if path == "cast remove":
            self.store.set_cast(100, None, before)
        else:
            self.store.set_cast(100, None, [])
        interaction = FakeInteraction(channel=self.channel, admin=True)
        await invoke(self.bot, path, interaction, *((text, "Wave hello") if path == "summon" else (text,)))
        reply = interaction.replies[0] if interaction.replies else ""
        if path in ("cast set", "cast add"):
            acted = self.names(self.store.get_cast(100))
        elif path == "cast remove":
            after = self.store.get_cast(100)
            acted = self.names([ident for ident in before if ident not in after])
        elif path == "admin cast default":
            acted = self.names(json.loads(self.store.channel(100)["default_cast"]))
        elif path == "summon":
            found = re.match(r"Tester summons (.+?): Wave hello", reply)
            acted = [found.group(1)] if found else []
        else:
            acted = [reply.split(" — home world:")[0]] if " — home world:" in reply else []
        return acted, reply

    async def assert_resolves(self, paths, text, expected):
        for path in paths:
            with self.subTest(path=path, text=text):
                acted, reply = await self.resolve(path, text)
                self.assertEqual(acted, [expected], reply)

    async def assert_refused(self, paths, text, *mentions):
        for path in paths:
            with self.subTest(path=path, text=text):
                acted, reply = await self.resolve(path, text)
                self.assertEqual(acted, [], reply)
                for mention in mentions:
                    self.assertIn(mention, reply)

    async def test_lists_resolve_a_unique_casefold_match(self):
        """Regression (UX-04): ``/cast set`` and ``/admin cast default`` resolve a unique case-insensitive name with
        surrounding whitespace."""
        self.add("Alice")
        await self.assert_resolves(LIST_PATHS, "  aLiCe ", "Alice")

    async def test_single_name_commands_resolve_a_unique_casefold_match(self):
        """UX-04: ``/cast add``, ``/cast remove``, ``/summon`` and ``/character info`` resolve a unique case-insensitive
        name with surrounding whitespace (previously they needed the exact ``name=?`` match and say "Character not found")."""
        self.add("Alice")
        await self.assert_resolves(SINGLE_PATHS, "  aLiCe ", "Alice")

    async def test_exact_match_wins_for_single_name_commands(self):
        """Regression (UX-04): when "Alice" and "alice" both exist, each name resolves to itself."""
        self.add("Alice")
        self.add("alice")
        await self.assert_resolves(SINGLE_PATHS, "Alice", "Alice")
        await self.assert_resolves(SINGLE_PATHS, "alice", "alice")

    async def test_exact_match_wins_for_comma_lists(self):
        """UX-04: with "Alice" and "alice" both eligible, ``/cast set Alice`` and ``/admin cast default Alice`` pick
        "Alice" (previously a casefold dict kept whichever row sorts last, so "Alice" silently becomes "alice")."""
        self.add("Alice")
        self.add("alice")
        await self.assert_resolves(LIST_PATHS, "Alice", "Alice")
        await self.assert_resolves(LIST_PATHS, "alice", "alice")

    async def test_ambiguous_casefold_match_is_refused_with_candidates(self):
        """UX-04: "Alice" and "ALICE" both match "alice" case-insensitively and neither exactly, so every character
        command refuses, changes nothing, and lists both candidates (previously the list commands silently picked one and
        the others said "Character not found")."""
        self.add("Alice")
        self.add("ALICE")
        await self.assert_refused(CHARACTER_PATHS, "alice", "Alice", "ALICE")

    async def test_unknown_character_reply_names_it(self):
        """UX-04: a name with no match is refused with "No character named Zed" (or similar wording naming it); previously
        it was "Character not found" or "Character not available here: zed" (casefolded)."""
        self.add("Alice")
        for path in CHARACTER_PATHS:
            with self.subTest(path=path):
                acted, reply = await self.resolve(path, "Zed")
                self.assertEqual(acted, [], reply)
                self.assertIn("no character", reply.lower())
                self.assertIn("Zed", reply)

    async def test_lists_resolve_among_eligible_characters(self):
        """Regression (UX-04): a same-name-different-case character outside the bound space does not make the list
        commands ambiguous."""
        self.add("Alice")
        self.add("alice", self.other)
        await self.assert_resolves(LIST_PATHS, "ALICE", "Alice")

    async def test_cast_and_summon_resolve_among_eligible_characters(self):
        """UX-04: ``/cast add``, ``/cast remove`` and ``/summon`` resolve among the characters eligible in the bound
        space, so "ALICE" resolves to the eligible "Alice" even though an ineligible "alice" exists elsewhere."""
        self.add("Alice")
        self.add("alice", self.other)
        await self.assert_resolves(ELIGIBLE_SINGLE_PATHS, "ALICE", "Alice")

    async def test_character_info_resolves_across_the_guild(self):
        """UX-04: ``/character info`` resolves across the whole guild: "carol" finds "Carol" in a world not bound here,
        and "ALICE" is ambiguous between "Alice" (here) and "alice" (another world)."""
        self.add("Alice")
        self.add("alice", self.other)
        self.add("Carol", self.other)
        await self.assert_resolves(["character info"], "carol", "Carol")
        await self.assert_refused(["character info"], "ALICE", "Alice", "alice")

    async def test_casefold_match_ignores_other_guilds(self):
        """UX-04 (guild isolation): guild 2's "ALICE" is not a candidate in guild 1, so "alice" resolves uniquely to
        guild 1's "Alice" in the single-name commands."""
        self.add("Alice")
        elsewhere = self.store.create_space(2, "Faraway", "world")
        self.add("ALICE", elsewhere, guild=2)
        await self.assert_resolves(SINGLE_PATHS, "alice", "Alice")

    async def test_other_guild_characters_never_resolve(self):
        """Regression (UX-04, guild isolation): a character that exists only in guild 2 is never resolved from guild 1,
        and its world is never named; guild 2's "ALICE" does not make the list commands ambiguous."""
        self.add("Alice")
        elsewhere = self.store.create_space(2, "Faraway", "world")
        self.add("Zara", elsewhere, guild=2)
        self.add("ALICE", elsewhere, guild=2)
        for path in CHARACTER_PATHS:
            with self.subTest(path=path):
                acted, reply = await self.resolve(path, "Zara")
                self.assertEqual(acted, [], reply)
                self.assertNotIn("Faraway", reply)
        await self.assert_resolves(LIST_PATHS, "alice", "Alice")


    async def test_cast_remove_drops_a_stale_member(self):
        """UX-04: after ``/admin space disallow_world`` unlinks Harbor from hub Plaza, the hub channel's cast still holds
        Harbor's Alice, and ``/cast remove alice`` must remove her (casefold resolution over eligible characters plus
        the current cast). Currently the resolver only sees eligible characters, so the stale member cannot be removed.
        Guild scoping still holds: guild 2's "Zara" is not resolved."""
        alice = self.add("Alice")
        plaza = self.store.create_space(1, "Plaza", "hub")
        self.store.link_world(1, plaza, self.harbor)
        self.store.bind_channel(1, 200, plaza)
        self.store.set_cast(200, None, [alice])
        unlink = FakeInteraction(admin=True)
        await invoke(self.bot, "admin space disallow_world", unlink, "Plaza", "Harbor")
        self.assertEqual(self.store.allowed_worlds(plaza), set(), unlink.replies)
        self.assertEqual(self.store.get_cast(200), [alice], "precondition: Alice is a stale cast member")
        elsewhere = self.store.create_space(2, "Faraway", "world")
        self.add("Zara", elsewhere, guild=2)
        foreign = FakeInteraction(channel_id=200)
        await invoke(self.bot, "cast remove", foreign, "Zara")
        self.assertNotIn("Removed", foreign.replies[0])
        interaction = FakeInteraction(channel_id=200)
        await invoke(self.bot, "cast remove", interaction, "alice")
        self.assertEqual(self.store.get_cast(200), [], interaction.replies)

    async def test_character_info_excludes_archived_characters(self):
        """UX-04: an archived character is not resolved by ``/character info``; the reply is "No character named
        Alice ..." (currently the guild-wide lookup includes archived rows and shows her home world)."""
        alice = self.add("Alice")
        self.store.archive_character(1, alice, True)
        acted, reply = await self.resolve("character info", "Alice")
        self.assertEqual(acted, [], reply)
        self.assertIn("no character named alice", reply.lower())


class SpaceResolutionTests(BotCase):
    def setUp(self):
        super().setUp()
        self.plaza = self.store.create_space(1, "Plaza", "hub")
        self.lore = self.store.add_lore(1, "channel", 100, "The tide is high")

    def space_name(self, ident):
        return self.store.space_by_id(ident)["name"] if ident else None

    def links(self):
        rows = self.store.all("SELECT hub_id, world_id FROM hub_worlds")
        return {(self.space_name(row["hub_id"]), self.space_name(row["world_id"])) for row in rows}

    async def run_command(self, path, *args):
        interaction = FakeInteraction(admin=True)
        await invoke(self.bot, path, interaction, *args)
        return interaction.replies[0] if interaction.replies else ""

    async def resolve(self, path, text):
        """Run a single-space command with ``text``; return (space name it acted on or None, reply)."""
        if path == "admin space bind":
            reply = await self.run_command(path, SimpleNamespace(id=300, mention="<#300>"), text)
            row = self.store.channel(300)
            return (self.space_name(row["space_id"]) if row else None), reply
        if path == "admin lore promote":
            reply = await self.run_command(path, self.lore, "space", text)
            row = self.store.one("SELECT scope_id FROM lore WHERE promoted_from=? AND scope_kind='space'", (self.lore,))
            return (self.space_name(row["scope_id"]) if row else None), reply
        reply = await self.run_command(path, text, card_attachment())
        row = self.store.character(1, "Newcomer")
        return (self.space_name(row["world_id"]) if row else None), reply

    async def link(self, path, hub, world):
        """Run allow/disallow (disallow starts from Plaza<->Harbor linked); return (links after, reply)."""
        self.store.execute("DELETE FROM hub_worlds")
        if path.endswith("disallow_world"):
            self.store.link_world(1, self.plaza, self.harbor)
        reply = await self.run_command(path, hub, world)
        return self.links(), reply

    async def test_space_commands_resolve_a_unique_casefold_match(self):
        """UX-04: every space option resolves a unique case-insensitive name with surrounding whitespace (previously each
        needed the exact ``name=?`` match)."""
        for path, text, expected in [("admin space bind", " plaza ", "Plaza"), ("admin lore promote", " HARBOR", "Harbor"),
                                     ("admin character import", "harbor ", "Harbor")]:
            with self.subTest(path=path):
                acted, reply = await self.resolve(path, text)
                self.assertEqual(acted, expected, reply)
        with self.subTest(path="admin space allow_world"):
            links, reply = await self.link("admin space allow_world", " plaza ", "HARBOR")
            self.assertEqual(links, {("Plaza", "Harbor")}, reply)
        with self.subTest(path="admin space disallow_world"):
            links, reply = await self.link("admin space disallow_world", "PLAZA", " harbor ")
            self.assertEqual(links, set(), reply)

    async def test_exact_space_match_wins(self):
        """Regression (UX-04): with hub "Plaza" and world "plaza" both present, each name resolves to itself."""
        self.store.create_space(1, "plaza", "world")
        for path in ("admin space bind", "admin lore promote"):
            for text in ("Plaza", "plaza"):
                with self.subTest(path=path, text=text):
                    self.store.execute("DELETE FROM lore WHERE promoted_from IS NOT NULL")
                    acted, reply = await self.resolve(path, text)
                    self.assertEqual(acted, text, reply)

    async def test_ambiguous_space_is_refused_with_candidates(self):
        """UX-04: "plaza" matches hubs "Plaza" and "PLAZA" (and "harbor" matches worlds "Harbor" and "HARBOR")
        case-insensitively with no exact match, so the command refuses, changes nothing, and lists both candidates."""
        self.store.create_space(1, "PLAZA", "hub")
        self.store.create_space(1, "HARBOR", "world")
        for path, text, candidates in [("admin space bind", "plaza", ("Plaza", "PLAZA")),
                                       ("admin lore promote", "plaza", ("Plaza", "PLAZA")),
                                       ("admin character import", "harbor", ("Harbor", "HARBOR"))]:
            with self.subTest(path=path):
                acted, reply = await self.resolve(path, text)
                self.assertIsNone(acted, reply)
                for candidate in candidates:
                    self.assertIn(candidate, reply)
        for path in ("admin space allow_world", "admin space disallow_world"):
            for hub, world, candidates in [("plaza", "Harbor", ("Plaza", "PLAZA")), ("Plaza", "harbor", ("Harbor", "HARBOR"))]:
                with self.subTest(path=path, hub=hub, world=world):
                    expected = {("Plaza", "Harbor")} if path.endswith("disallow_world") else set()
                    links, reply = await self.link(path, hub, world)
                    self.assertEqual(links, expected, reply)
                    for candidate in candidates:
                        self.assertIn(candidate, reply)

    async def test_unknown_space_reply_names_it(self):
        """UX-04: a space name with no match is refused with a reply naming it (previously "Space not found", "Hub or world
        not found", "Choose an existing world", "Destination space not found")."""
        for path in ("admin space bind", "admin lore promote", "admin character import"):
            with self.subTest(path=path):
                acted, reply = await self.resolve(path, "Nowhere")
                self.assertIsNone(acted, reply)
                self.assertIn("Nowhere", reply)
        for path in ("admin space allow_world", "admin space disallow_world"):
            for hub, world in [("Nowhere", "Harbor"), ("Plaza", "Nowhere")]:
                with self.subTest(path=path, hub=hub, world=world):
                    expected = {("Plaza", "Harbor")} if path.endswith("disallow_world") else set()
                    links, reply = await self.link(path, hub, world)
                    self.assertEqual(links, expected, reply)
                    self.assertIn("Nowhere", reply)

    async def test_wrong_kind_gets_a_clear_message(self):
        """UX-04: a world given as the hub, or a hub given as the world, is refused with a message naming the space and
        the expected kind, and nothing changes (previously allow_world said "Choose a hub and world in this server",
        disallow_world reports success without checking kinds, import says "Choose an existing world")."""
        cases = [("Harbor", "Harbor", "Harbor", "hub"), ("Plaza", "Plaza", "Plaza", "world"), ("Other", "Plaza", "Other", "hub")]
        for path in ("admin space allow_world", "admin space disallow_world"):
            for hub, world, named, kind in cases:
                with self.subTest(path=path, hub=hub, world=world):
                    expected = {("Plaza", "Harbor")} if path.endswith("disallow_world") else set()
                    links, reply = await self.link(path, hub, world)
                    self.assertEqual(links, expected, reply)
                    self.assertIn(named, reply)
                    self.assertIn(kind, reply.lower())
                    self.assertNotIn("no longer linked", reply)
                    self.assertNotIn("now available", reply)
        with self.subTest(path="admin character import"):
            acted, reply = await self.resolve("admin character import", "Plaza")
            self.assertIsNone(acted, reply)
            self.assertIn("Plaza", reply)
            self.assertIn("world", reply.lower())

    async def test_space_casefold_match_ignores_other_guilds(self):
        """UX-04 (guild isolation): guild 2's "PLAZA" is not a candidate in guild 1, so "plaza" resolves uniquely to
        guild 1's "Plaza"."""
        self.store.create_space(2, "PLAZA", "hub")
        acted, reply = await self.resolve("admin space bind", "plaza")
        self.assertEqual(acted, "Plaza", reply)

    async def test_other_guild_spaces_never_resolve(self):
        """Regression (UX-04, guild isolation): a space that exists only in guild 2 is never bound, linked, promoted to
        or imported into from guild 1."""
        self.store.create_space(2, "Faraway", "world")
        self.store.create_space(2, "Far Plaza", "hub")
        for path in ("admin space bind", "admin lore promote", "admin character import"):
            with self.subTest(path=path):
                acted, reply = await self.resolve(path, "Faraway")
                self.assertIsNone(acted, reply)
        for path in ("admin space allow_world", "admin space disallow_world"):
            for hub, world in [("Plaza", "Faraway"), ("Far Plaza", "Harbor")]:
                with self.subTest(path=path, hub=hub, world=world):
                    expected = {("Plaza", "Harbor")} if path.endswith("disallow_world") else set()
                    links, reply = await self.link(path, hub, world)
                    self.assertEqual(links, expected, reply)


class AutocompleteTests(BotCase):
    def setUp(self):
        super().setUp()
        for name in ("Alice", "Malia", "Bob"):
            self.add(name)
        for name in ("Carol", "Lina"):
            self.add(name, self.other)
        self.plaza = self.store.create_space(1, "Plaza", "hub")
        self.store.link_world(1, self.plaza, self.harbor)
        self.store.bind_channel(1, 200, self.plaza)
        faraway = self.store.create_space(2, "Faraway", "world")
        self.store.create_space(2, "Far Plaza", "hub")
        self.add("Alina", faraway, guild=2)

    async def values(self, path, option, current, **interaction):
        choices = await autocomplete(self.bot, path, option, FakeInteraction(admin=True, **interaction), current)
        self.assertIsInstance(choices, list)
        self.assertLessEqual(len(choices), 25)
        for choice in choices:
            self.assertIsInstance(choice, app_commands.Choice)
        return {choice.value for choice in choices}

    def test_name_options_have_autocomplete(self):
        """UX-04: every character and space option is synced with ``autocomplete: true`` (previously none was)."""
        missing = []
        for path, option in [*CHARACTER_OPTIONS, *SPACE_OPTIONS]:
            options = {item["name"]: item for item in command(self.bot, path).to_dict(self.bot.tree).get("options", [])}
            if not options.get(option, {}).get("autocomplete"):
                missing.append((path, option))
        self.assertEqual(missing, [])

    async def test_character_autocomplete_filters_case_insensitively(self):
        """UX-04: character suggestions are a case-insensitive substring match; cast and summon options offer only the
        characters eligible in channel 100's world (not "Lina" from another world, not guild 2's "Alina");
        ``/character info`` offers the whole guild (including "Lina", never "Alina")."""
        for (path, option), scope in CHARACTER_OPTIONS.items():
            with self.subTest(path=path, option=option):
                expected = {"Alice", "Malia", "Lina"} if scope == "guild" else {"Alice", "Malia"}
                self.assertEqual(await self.values(path, option, "LI"), expected)

    async def test_cast_autocomplete_in_a_hub_uses_linked_worlds(self):
        """UX-04: in a hub-bound channel, cast and summon suggestions are the characters of the hub's linked worlds."""
        for (path, option), scope in CHARACTER_OPTIONS.items():
            if scope == "eligible":
                with self.subTest(path=path, option=option):
                    self.assertEqual(await self.values(path, option, "", channel_id=200), {"Alice", "Malia", "Bob"})

    async def test_space_autocomplete_filters_by_kind_and_guild(self):
        """UX-04: space suggestions are guild 1's spaces only; hub options offer hubs, world options offer worlds, and
        the filter is a case-insensitive substring."""
        expected_all = {None: {"Harbor", "Other", "Plaza"}, "hub": {"Plaza"}, "world": {"Harbor", "Other"}}
        for (path, option), kind in SPACE_OPTIONS.items():
            with self.subTest(path=path, option=option):
                self.assertEqual(await self.values(path, option, ""), expected_all[kind])
                self.assertEqual(await self.values(path, option, "AR"), {"Harbor"} if kind != "hub" else set())
                self.assertEqual(await self.values(path, option, "aZ"), {"Plaza"} if kind != "world" else set())

    async def test_autocomplete_returns_at_most_25_choices(self):
        """UX-04: with 30 matching characters and 30 matching worlds, each option answers 1..25 choices."""
        for index in range(30):
            self.add(f"Extra {index:02}")
            self.store.create_space(1, f"Realm {index:02}", "world")
        for (path, option), kind in [*CHARACTER_OPTIONS.items(), *SPACE_OPTIONS.items()]:
            if kind == "hub":
                continue
            with self.subTest(path=path, option=option):
                current = "extra" if (path, option) in CHARACTER_OPTIONS else "realm"
                found = await self.values(path, option, current)
                self.assertTrue(found, "no suggestions")
                self.assertTrue(all(value.lower().startswith(current) for value in found), found)

    async def test_autocomplete_is_empty_in_a_dm(self):
        """UX-04: every character and space autocomplete answers an empty list in a DM without raising."""
        for path, option in [*CHARACTER_OPTIONS, *SPACE_OPTIONS]:
            with self.subTest(path=path, option=option):
                self.assertEqual(await self.values(path, option, "a", guild_id=None, channel_id=555), set())

    async def test_cast_autocomplete_is_empty_in_an_unbound_channel(self):
        """UX-04: cast and summon suggestions need a bound channel; in an unbound channel they are empty, no error."""
        for (path, option), scope in CHARACTER_OPTIONS.items():
            if scope == "eligible":
                with self.subTest(path=path, option=option):
                    self.assertEqual(await self.values(path, option, "a", channel_id=300), set())

    async def test_space_autocomplete_works_in_an_unbound_channel(self):
        """UX-04: space options do not depend on the current channel's binding (``/admin space bind`` is usually run
        from a channel that is not bound yet), so they still suggest guild 1's spaces in unbound channel 300."""
        for (path, option), kind in SPACE_OPTIONS.items():
            with self.subTest(path=path, option=option):
                expected = {None: {"Harbor", "Other", "Plaza"}, "hub": {"Plaza"}, "world": {"Harbor", "Other"}}[kind]
                self.assertEqual(await self.values(path, option, "", channel_id=300), expected)


    async def test_comma_list_autocomplete_completes_the_last_fragment(self):
        """Regression (UX-04): for ``characters`` options the last comma-separated fragment is completed and the value
        keeps the earlier names ("Alice, ma" -> "Alice, Malia"); names already in the list are not suggested again."""
        for path in ("cast set", "admin cast default"):
            with self.subTest(path=path):
                self.assertIn("Alice, Malia", await self.values(path, "characters", "Alice, ma"))
                found = await self.values(path, "characters", "Alice, ")
                self.assertEqual(found, {"Alice, Malia", "Alice, Bob"})
                self.assertEqual(await self.values(path, "characters", "alice, Bob, "), {"alice, Bob, Malia"})

    async def test_autocomplete_source_failure_returns_empty_and_logs_warning(self):
        """Regression (UX-04): when the store query behind an autocomplete raises, the callback answers ``[]`` and logs a
        WARNING instead of propagating the error."""
        def broken(*args, **kwargs):
            raise RuntimeError("database is locked")

        sources = {("character info", "character"): "all"}
        for path, option in [*CHARACTER_OPTIONS, *SPACE_OPTIONS]:
            method = sources.get((path, option), "eligible_characters" if (path, option) in CHARACTER_OPTIONS else "list_spaces")
            with self.subTest(path=path, option=option, method=method):
                original = getattr(self.store, method)
                setattr(self.store, method, broken)
                try:
                    with self.assertLogs(level="WARNING") as logs:
                        found = await self.values(path, option, "a")
                finally:
                    setattr(self.store, method, original)
                self.assertEqual(found, set())
                self.assertTrue(any(record.levelname == "WARNING" for record in logs.records), logs.output)

    async def test_autocomplete_skips_values_over_100_characters(self):
        """Regression (UX-04): Discord rejects choice values over 100 characters, so a name whose resulting value would
        be longer is not suggested: a 101-character name never; a 95-character name alone, but not after "Alice, "."""
        too_long, fits = "L" * 101, "M" * 95
        self.add(too_long)
        self.add(fits)
        for (path, option), _scope in CHARACTER_OPTIONS.items():
            with self.subTest(path=path, option=option):
                found = await self.values(path, option, "")
                self.assertNotIn(too_long, found)
                self.assertTrue(all(len(value) <= 100 for value in found), found)
                if option == "character":
                    self.assertIn(fits, await self.values(path, option, "mmm"))
                else:
                    self.assertEqual(await self.values(path, option, "Alice, mmm"), set())

    async def test_character_info_autocomplete_excludes_archived_characters(self):
        """UX-04: archived characters never appear in ``/character info`` suggestions (currently they do)."""
        self.store.archive_character(1, self.store.character(1, "Alice")["id"], True)
        self.assertEqual(await self.values("character info", "character", "LI"), {"Malia", "Lina"})


class OptionNameTests(unittest.TestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())

    def tearDown(self):
        self.bot.store.close()

    def test_options_use_consistent_names(self):
        """UX-06: synced option names are ``character`` / ``characters`` / ``space`` / ``hub`` / ``world`` (previously
        ``name``, ``character_name``, ``names``, ``space_name``, ``hub_name``, ``world_name``); ``/admin space create``
        keeps ``kind`` and ``name``."""
        found = {path: [option["name"] for option in command(self.bot, path).to_dict(self.bot.tree).get("options", [])]
                 for path in OPTIONS}
        self.assertEqual(found, OPTIONS)


if __name__ == "__main__":
    unittest.main()
