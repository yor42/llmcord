"""FEAT-46 Korean smoke tests (D39): replies, public tables, josa and command metadata in Korean, English unchanged."""
import asyncio
import unittest
from types import SimpleNamespace

import discord
from discord import app_commands
from helpers import FakeInteraction, invoke, make_settings

import test_games_discord as games
from llmcord_core.discord_bot import SkitBot
from llmcord_core.i18n import Translator, discord_locale
from llmcord_core.i18n.discord_locale import LlmcordTranslator

HANGUL = lambda text: any("가" <= ch <= "힣" for ch in text)  # noqa: E731


def reset_server_language():
    discord_locale.set_server_language_source(None)


class SlashReplyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        reset_server_language()
        self.addCleanup(reset_server_language)
        self.addCleanup(self.bot.store.close)

    async def reply(self, path, locale, **kwargs):
        interaction = FakeInteraction()
        if locale is not None:
            interaction.locale = locale
        await invoke(self.bot, path, interaction, **kwargs)
        return interaction.replies

    async def test_ko_invoker_gets_korean_and_en_us_gets_unchanged_english(self):
        """FEAT-46 Korean smoke tests: /balance, /inventory, /favorites list and a /blackjack refusal follow the invoker locale."""
        cases = (
            ("balance", {}, "내 잔액은 0 coins(이)에요.", "You have 0 coins."),
            ("inventory", {}, "가지고 있는 아이템이 없어요. /shop에서 구매해 보세요.", "You are not holding any items. Visit /shop to buy some."),
            ("favorites list", {}, "아직 즐겨찾기가 없어요. /favorites add로 추가해 주세요.", "You have no favorites yet. Add one with /favorites add."),
            ("blackjack", {"bet": 10}, "이 채널에서는 게임이 꺼져 있어요.", "Games are off in this channel."),
        )
        for path, kwargs, korean, english in cases:
            with self.subTest(path=path):
                self.assertEqual(await self.reply(path, discord.Locale.korean, **kwargs), [korean])
                self.assertEqual(await self.reply(path, discord.Locale.american_english, **kwargs), [english])
                self.assertEqual(await self.reply(path, None, **kwargs), [english])

    async def test_other_locale_falls_back_to_english(self):
        """FEAT-46 Korean smoke tests: a locale with no pack (Japanese) is answered in English."""
        self.assertEqual(await self.reply("balance", discord.Locale.japanese), ["You have 0 coins."])


class PublicTableTests(games.GameCase):
    def setUp(self):
        reset_server_language()
        self.addCleanup(reset_server_language)

    async def open_table(self, server_language, invoker_locale):
        self.store.set_server_language(games.G, server_language)
        self.bot.get_guild = lambda guild_id: SimpleNamespace(id=guild_id, features=(), preferred_locale=discord.Locale.american_english)
        discord_locale.set_server_language_source(self.store.server_language)
        click = games.Click(self.bot, games.ALICE)
        click.locale = invoker_locale
        await invoke(self.bot, "blackjack", click, bet=10)
        return click

    async def test_public_table_in_a_korean_server_is_korean(self):
        """FEAT-46 Korean smoke tests: the public table message follows the server language (ko)."""
        click = await self.open_table("ko", discord.Locale.american_english)
        content, kwargs = click.response.sent[0]
        self.assertTrue(HANGUL(content), content)
        self.assertNotIn("Join with", content)
        labels = [child.item.label for child in kwargs["view"].children]
        self.assertEqual(labels[-1], Translator("ko").p("button.blackjack", "How to play"))

    async def test_public_table_in_an_english_server_is_english(self):
        """FEAT-46 Korean smoke tests: an en server gets the unchanged English table even from a ko invoker (the audience rule)."""
        click = await self.open_table("en", discord.Locale.korean)
        content, kwargs = click.response.sent[0]
        self.assertIn("**Blackjack**", content)
        self.assertRegex(content, r"Join with /blackjack bet:<amount> — dealing <t:\d+:R>\.")
        self.assertFalse(HANGUL(content), content)
        self.assertEqual([child.item.label for child in kwargs["view"].children], ["Deal now", "Leave", "How to play"])

    async def test_ephemeral_join_reply_follows_the_invoker_not_the_server(self):
        """FEAT-46 Korean smoke tests: a ko member joining an en server's table sees the private reply in Korean, the table stays English."""
        await self.open_table("en", discord.Locale.american_english)
        click = games.Click(self.bot, games.BOB)
        click.locale = discord.Locale.korean
        await invoke(self.bot, "blackjack", click, bet=25)
        self.assertTrue(HANGUL(click.replies[0]), click.replies)
        self.assertNotIn("You joined", click.replies[0])
        self.assertFalse(HANGUL(self.channel.edits[-1][1]["content"]))


class JosaTests(unittest.TestCase):
    ko = Translator("ko")

    def test_object_particle_after_consonant_vowel_number_and_latin(self):
        """FEAT-46 Korean smoke tests: 'Added {name} to your favorites.' picks 을/를 from the name's last sound."""
        for name, want in (("리안", "리안을"), ("미나", "미나를"), ("3", "3을"), ("2", "2를"), ("Alice", "Alice을(를)")):
            with self.subTest(name=name):
                self.assertEqual(self.ko("Added {name} to your favorites.", name=name), f"즐겨찾기에 {want} 추가했어요.")

    def test_direction_particle_treats_a_final_rieul_like_a_vowel(self):
        """FEAT-46 Korean smoke tests: 'currency is now called {name}' picks 으로/로; 서울 (ㄹ) takes 로."""
        for name, want in (("리안", "리안으로"), ("미나", "미나로"), ("서울", "서울로"), ("Gold", "Gold(으)로")):
            with self.subTest(name=name):
                self.assertEqual(self.ko("This server's currency is now called {name}.", name=name), f"이 서버의 화폐 이름을 {want} 바꿨어요.")

    def test_subject_particle_in_a_game_message(self):
        """FEAT-46 Korean smoke tests: '{who} claims {claim}.' picks 이/가 and 이라/라 independently."""
        for who, claim, want in (("리안", "에이스", "리안이 에이스라고"), ("미나", "킹", "미나가 킹이라고"), ("Bob", "3", "Bob이(가) 3이라고")):
            with self.subTest(who=who):
                out = self.ko("{who} claims {claim}.", who=who, claim=claim)
                self.assertEqual(out, f"{want} 선언했어요.")


class CommandMetadataTests(unittest.TestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.addCleanup(self.bot.store.close)
        self.translator = LlmcordTranslator()

    def payloads(self):
        async def build():
            return [await cmd.get_translated_payload(self.bot.tree, self.translator) for cmd in self.bot.tree.get_commands()]
        return asyncio.run(build())

    def walk(self, node):
        """Yield every dict in a command payload that can carry localizations."""
        yield node
        for key in ("options", "choices"):
            for child in node.get(key, ()):
                yield from self.walk(child)

    def test_real_commands_have_korean_descriptions_and_only_ko(self):
        """FEAT-46 Korean smoke tests: the registered translator localizes command descriptions to ko and no other locale."""
        by_name = {p["name"]: p for p in self.payloads()}
        for name in ("balance", "inventory", "blackjack", "favorites"):
            with self.subTest(command=name):
                loc = by_name[name]["description_localizations"]
                self.assertEqual(set(loc), {"ko"})
                self.assertTrue(HANGUL(loc["ko"]), loc)
                self.assertNotEqual(loc["ko"], by_name[name]["description"])
        for payload in self.payloads():
            for node in self.walk(payload):
                self.assertLessEqual(set(node.get("description_localizations") or ()), {"ko"})
                self.assertLessEqual(set(node.get("name_localizations") or ()), {"ko"})

    def test_translate_returns_none_for_english_and_unpacked_locales(self):
        """FEAT-46 Korean smoke tests: translate answers None for en-US and a locale without a pack, Korean text for ko."""
        context = app_commands.TranslationContext(app_commands.TranslationContextLocation.command_description, None)
        ko = {p["name"]: p for p in self.payloads()}["balance"]["description_localizations"]["ko"]
        english = {p["name"]: p for p in self.payloads()}["balance"]["description"]
        source = app_commands.locale_str(english)

        async def translate(locale):
            return await self.translator.translate(source, locale, context)
        self.assertEqual(asyncio.run(translate(discord.Locale.korean)), ko)
        self.assertIsNone(asyncio.run(translate(discord.Locale.american_english)))
        self.assertIsNone(asyncio.run(translate(discord.Locale.japanese)))

    def test_korean_descriptions_and_choice_names_fit_discord_limit(self):
        """FEAT-46 Korean smoke tests: every Korean description and choice name is at most 100 characters."""
        checked = 0
        for payload in self.payloads():
            for node in self.walk(payload):
                for field in ("description_localizations", "name_localizations"):
                    text = (node.get(field) or {}).get("ko")
                    if text is not None:
                        checked += 1
                        self.assertLessEqual(len(text), 100, text)
        self.assertGreater(checked, 50)


if __name__ == "__main__":
    unittest.main()
