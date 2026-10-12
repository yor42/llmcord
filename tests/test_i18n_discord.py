import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import discord
from babel.messages.catalog import Catalog
from babel.messages.pofile import write_po
from discord import app_commands

from llmcord_core import i18n
from llmcord_core.i18n import clear_cache
from llmcord_core.i18n import discord_locale as dl
from llmcord_core.i18n.discord_locale import C_, LlmcordTranslator, channel_locale, invoker_locale

LONG = "L" * 101


def write_po_file(root, locale, domain, entries):
    cat = Catalog(locale=locale)
    for msgid, msgstr in entries.items():
        cat.add(msgid, msgstr)
    path = Path(root) / locale / "LC_MESSAGES" / f"{domain}.po"
    path.parent.mkdir(parents=True)
    with open(path, "wb") as f:
        write_po(f, cat, omit_header=False)


def ctx(location):
    return app_commands.TranslationContext(location, None)


class FixtureCase(unittest.TestCase):
    def setUp(self):
        clear_cache()
        self.addCleanup(clear_cache)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        write_po_file(self.root, "ko", "commands", {"Roll dice": "주사위 굴리기", "Sides": "면 수", "Red": "빨강", LONG: "K" * 101, "Greet": "인사", "Name": "이름"})
        patcher = mock.patch.dict("os.environ", {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for var in ("LLMCORD_LOCALES", "LLMCORD_I18N_PSEUDO"):
            __import__("os").environ.pop(var, None)


class TranslatorTests(FixtureCase):
    """FEAT-43: LlmcordTranslator returns catalog text for descriptions only."""

    def translate(self, msg, locale, location=app_commands.TranslationContextLocation.command_description):
        tr = LlmcordTranslator(root=self.root)
        return asyncio.run(tr.translate(msg, locale, ctx(location)))

    def test_c_returns_locale_str(self):
        out = C_("Roll dice")
        self.assertIsInstance(out, app_commands.locale_str)
        self.assertEqual(out.message, "Roll dice")
        self.assertEqual(str(out), "Roll dice")

    def test_korean_description(self):
        self.assertEqual(self.translate(C_("Roll dice"), discord.Locale.korean), "주사위 굴리기")

    def test_names_stay_english(self):
        L = app_commands.TranslationContextLocation
        for loc in (L.command_name, L.group_name, L.parameter_name):
            self.assertIsNone(self.translate(C_("Roll dice"), discord.Locale.korean, loc))

    def test_english_locales_none(self):
        for loc in (discord.Locale.american_english, discord.Locale.british_english):
            self.assertIsNone(self.translate(C_("Roll dice"), loc))

    def test_locale_without_pack_none(self):
        self.assertIsNone(self.translate(C_("Roll dice"), discord.Locale.japanese))

    def test_missing_entry_none(self):
        self.assertIsNone(self.translate(C_("Not in catalog"), discord.Locale.korean))

    def test_long_description_dropped_with_warning(self):
        with self.assertLogs("llmcord_core.i18n", "WARNING") as logs:
            self.assertIsNone(self.translate(C_(LONG), discord.Locale.korean))
        self.assertEqual(len(logs.records), 1)

    def test_long_choice_name_dropped(self):
        L = app_commands.TranslationContextLocation
        with self.assertLogs("llmcord_core.i18n", "WARNING"):
            self.assertIsNone(self.translate(C_(LONG), discord.Locale.korean, L.choice_name))

    def test_context_extra_selects_msgctxt(self):
        self.assertIsNone(self.translate(app_commands.locale_str("Roll dice", context="x"), discord.Locale.korean))


class PayloadSpikeTests(FixtureCase):
    """FEAT-43 spike: discord.py applies the translator to descriptions and choice names, never to names."""

    def payload(self):
        client = discord.Client(intents=discord.Intents.none())
        tree = app_commands.CommandTree(client)

        async def roll(interaction, sides: int, color: str):
            pass

        cmd = app_commands.Command(name="roll", description=C_("Roll dice"), callback=roll)
        cmd = app_commands.describe(sides=C_("Sides"))(cmd)
        cmd = app_commands.choices(color=[app_commands.Choice(name=C_("Red"), value="r")])(cmd)
        tr = LlmcordTranslator(root=self.root)
        return asyncio.run(cmd.get_translated_payload(tree, tr))

    def test_localizations_for_description_parameter_and_choice(self):
        data = self.payload()
        self.assertEqual(data["description_localizations"], {"ko": "주사위 굴리기"})
        self.assertFalse(data.get("name_localizations"))
        sides, color = data["options"]
        self.assertEqual(sides["description_localizations"], {"ko": "면 수"})
        self.assertFalse(sides.get("name_localizations"))
        self.assertFalse(color.get("name_localizations"))
        self.assertEqual(color["choices"][0]["name"], "Red")
        self.assertEqual(color["choices"][0]["name_localizations"], {"ko": "빨강"})


class LocaleHelperTests(FixtureCase):
    """FEAT-43: invoker_locale and channel_locale choose the reply language."""

    def test_invoker_locale(self):
        avail = ("en", "ko")
        for tag, want in ((discord.Locale.korean, "ko"), (discord.Locale.american_english, "en"), (discord.Locale.japanese, "en")):
            self.assertEqual(invoker_locale(SimpleNamespace(locale=tag), avail), want)

    def test_invoker_locale_default_available_uses_catalogs(self):
        with mock.patch.object(dl, "available_locales", return_value=("en", "ko")):
            self.assertEqual(invoker_locale(SimpleNamespace(locale=discord.Locale.korean)), "ko")

    def guild(self, features, preferred):
        return SimpleNamespace(features=features, preferred_locale=preferred)

    def test_channel_locale(self):
        avail = ("en", "ko")
        community = self.guild(["COMMUNITY"], discord.Locale.korean)
        plain = self.guild([], discord.Locale.korean)
        self.assertEqual(channel_locale("ko", plain, avail), "ko")
        self.assertEqual(channel_locale("ko", None, avail), "ko")
        self.assertEqual(channel_locale("ja", plain, avail), "en")
        self.assertEqual(channel_locale("ja", community, avail), "ko")
        self.assertEqual(channel_locale("", community, avail), "ko")
        self.assertEqual(channel_locale(None, plain, avail), "en")
        self.assertEqual(channel_locale(None, None, avail), "en")
        self.assertEqual(channel_locale("", self.guild(["COMMUNITY"], discord.Locale.japanese), avail), "en")


class PhaseOneLocaleTests(unittest.TestCase):
    """FEAT-43: invoker_t and channel_t are English stubs that ignore their argument."""

    def test_invoker_t_is_english_until_feat44(self):
        """FEAT-44 changes this: invoker_t will follow the interaction locale, so update this test then."""
        fake = SimpleNamespace(locale=discord.Locale.korean)
        self.assertIs(dl.invoker_t(fake), i18n.english())
        self.assertEqual(dl.invoker_t(fake).locale, "en")
        self.assertIs(dl.invoker_t(object()), i18n.english())

    def test_channel_t_is_english_until_feat44(self):
        """FEAT-44 changes this: channel_t will follow the server bot language, so update this test then."""
        guild = SimpleNamespace(features=["COMMUNITY"], preferred_locale=discord.Locale.korean)
        self.assertIs(dl.channel_t(guild), i18n.english())
        self.assertEqual(dl.channel_t(guild).locale, "en")
        self.assertIs(dl.channel_t(None), i18n.english())


if __name__ == "__main__":
    unittest.main()
