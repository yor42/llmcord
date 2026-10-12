"""FEAT-44 (batch 2a): server bot language (schema v22) and the real Discord audience resolvers."""
import logging
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import discord
from helpers import make_settings

from llmcord_core import i18n
from llmcord_core.discord_bot import SkitBot
from llmcord_core.i18n import catalog, clear_cache, codes
from llmcord_core.i18n import discord_locale as dl
from llmcord_core.store import Store
from llmcord_core.usage import reply_footer

G1, G2 = 1, 2


def v21_file(folder):
    """A current store with the v22 column removed again and user_version set back to 21."""
    path = Path(folder) / "v21.sqlite3"
    store = Store(path)
    store.set_usage_footer(G1, False)
    store.set_usage_footer(G2, True)
    store.close()
    with closing(sqlite3.connect(path)) as db, db:
        if "language" in {r[1] for r in db.execute("PRAGMA table_info(guild_settings)")}:
            db.execute("ALTER TABLE guild_settings DROP COLUMN language")
        db.execute("PRAGMA user_version=21")
    return path


def settings_rows(path):
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        return [dict(r) for r in db.execute("SELECT * FROM guild_settings ORDER BY guild_id")]


def user_version(path):
    with closing(sqlite3.connect(path)) as db:
        return db.execute("PRAGMA user_version").fetchone()[0]


class LocaleCase(unittest.TestCase):
    """Clean env (no pseudo, no locale filter), clean caches, and no server-language source before and after."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for var in ("LLMCORD_LOCALES", "LLMCORD_I18N_PSEUDO"):
            os.environ.pop(var, None)
        clear_cache()
        self.addCleanup(clear_cache)
        dl.set_server_language_source(None)
        self.addCleanup(dl.set_server_language_source, None)

    def pseudo_on(self):
        os.environ["LLMCORD_I18N_PSEUDO"] = "1"
        clear_cache()


def guild(features=(), preferred=discord.Locale.american_english, guild_id=G1):
    return SimpleNamespace(id=guild_id, features=list(features), preferred_locale=preferred)


class MigrationV22Tests(unittest.TestCase):
    def test_v21_file_upgrades_with_a_backup_and_keeps_rows(self):
        """FEAT-44: a v21 file gets a .pre-v22 backup, the language column, and its guild_settings rows keep language ''."""
        with tempfile.TemporaryDirectory() as folder:
            path = v21_file(folder)
            before = settings_rows(path)
            self.assertEqual(len(before), 2)
            self.assertNotIn("language", before[0])
            Store(path).close()
            self.assertEqual(user_version(path), 22)
            after = settings_rows(path)
            self.assertEqual([r["language"] for r in after], ["", ""])
            self.assertEqual([{k: v for k, v in r.items() if k != "language"} for r in after], before)
            backups = list(Path(folder).glob("v21.sqlite3.pre-v22-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(user_version(backups[0]), 21)
            with closing(sqlite3.connect(path)) as db:
                column = {r[1]: r for r in db.execute("PRAGMA table_info(guild_settings)")}["language"]
            self.assertEqual((column[2], column[3], column[4]), ("TEXT", 1, "''"))

    def test_v22_file_opened_again_does_nothing(self):
        """FEAT-44: reopening a v22 file leaves rows alone and makes no further backup."""
        with tempfile.TemporaryDirectory() as folder:
            path = v21_file(folder)
            Store(path).close()
            first = settings_rows(path)
            Store(path).close()
            Store(path).close()
            self.assertEqual(settings_rows(path), first)
            self.assertEqual(user_version(path), 22)
            self.assertEqual(len(list(Path(folder).glob("*.pre-v22-*.sqlite3"))), 1)

    def test_fresh_store_is_v22_with_the_same_shape_as_an_upgraded_one(self):
        """FEAT-44: a new file and an upgraded file have the same guild_settings.language column, and a new file needs no backup."""
        with tempfile.TemporaryDirectory() as folder:
            fresh = Path(folder) / "fresh.sqlite3"
            Store(fresh).close()
            upgraded = v21_file(folder)
            Store(upgraded).close()
            shapes = []
            for path in (fresh, upgraded):
                self.assertEqual(user_version(path), 22)
                with closing(sqlite3.connect(path)) as db:
                    shapes.append({r[1]: r[2:5] for r in db.execute("PRAGMA table_info(guild_settings)")})
            self.assertEqual(shapes[0], shapes[1])
            self.assertEqual(list(Path(folder).glob("fresh.sqlite3.pre-*")), [])

    def test_in_memory_store_is_v22(self):
        """FEAT-44: an in-memory store reports user_version 22."""
        store = Store()
        try:
            self.assertEqual(store.one("PRAGMA user_version")[0], 22)
        finally:
            store.close()

    def test_a_v23_database_is_refused(self):
        """FEAT-44: the newer-than-application guard moves to 23."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "new.sqlite3"
            Store(path).close()
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("PRAGMA user_version=23")
            with self.assertRaisesRegex(ValueError, "newer than this application"):
                Store(path)


class ServerLanguageStoreTests(LocaleCase):
    def setUp(self):
        super().setUp()
        self.store = Store()
        self.addCleanup(self.store.close)

    def test_default_is_empty_and_round_trips(self):
        """FEAT-44: no row means '', and set_server_language returns and stores the value."""
        self.assertEqual(self.store.server_language(G1), "")
        self.assertEqual(self.store.set_server_language(G1, "ko"), "ko")
        self.assertEqual(self.store.server_language(G1), "ko")
        self.assertEqual(self.store.set_server_language(G1, ""), "")
        self.assertEqual(self.store.server_language(G1), "")

    def test_guild_isolation(self):
        """FEAT-44: one guild's language never shows up in another guild."""
        self.store.set_server_language(G1, "ko")
        self.assertEqual(self.store.server_language(G2), "")
        self.store.set_server_language(G2, "")
        self.assertEqual(self.store.server_language(G1), "ko")

    def test_setting_it_keeps_other_guild_settings(self):
        """FEAT-44: the upsert leaves the other guild_settings columns alone."""
        self.store.set_usage_footer(G1, False)
        self.store.set_server_language(G1, "ko")
        self.assertFalse(self.store.usage_footer_enabled(G1))
        self.store.set_usage_footer(G1, True)
        self.assertEqual(self.store.server_language(G1), "ko")

    def test_normalizes_the_given_tag(self):
        """FEAT-44: 'ko-KR' is stored as 'ko'."""
        self.assertEqual(self.store.set_server_language(G1, "ko-KR"), "ko")
        self.assertEqual(self.store.server_language(G1), "ko")

    def test_unknown_language_raises_a_coded_error(self):
        """FEAT-44: an unavailable locale raises SERVER_LANGUAGE_UNKNOWN with the given value and English text."""
        self.store.set_server_language(G1, "ko")
        with self.assertRaises(i18n.UserError) as raised:
            self.store.set_server_language(G1, "xx-YY")
        err = raised.exception
        self.assertIs(err.code, codes.SERVER_LANGUAGE_UNKNOWN)
        self.assertEqual(err.params, {"language": "xx-YY"})
        self.assertEqual(i18n.english().error(err), "Unknown language: xx-YY")
        self.assertEqual(str(err), "Unknown language: xx-YY")
        self.assertEqual(self.store.server_language(G1), "ko")  # a refused write changes nothing

    def test_pseudo_only_when_available(self):
        """FEAT-44: 'pseudo' is refused unless LLMCORD_I18N_PSEUDO makes it available."""
        with self.assertRaises(i18n.UserError) as raised:
            self.store.set_server_language(G1, "pseudo")
        self.assertIs(raised.exception.code, codes.SERVER_LANGUAGE_UNKNOWN)
        self.pseudo_on()
        self.assertEqual(self.store.set_server_language(G1, "pseudo"), "pseudo")
        self.assertEqual(self.store.server_language(G1), "pseudo")


class InvokerTests(LocaleCase):
    def test_follows_the_interaction_locale(self):
        """FEAT-44: invoker_t returns a Translator for the interaction's locale."""
        t = dl.invoker_t(SimpleNamespace(locale=discord.Locale.korean))
        self.assertIsInstance(t, i18n.Translator)
        self.assertEqual(t.locale, "ko")
        self.assertEqual(dl.invoker_t(SimpleNamespace(locale=discord.Locale.american_english)).locale, "en")
        self.assertEqual(dl.invoker_t(SimpleNamespace(locale=discord.Locale.japanese)).locale, "en")  # no pack

    def test_accepts_a_plain_string_locale(self):
        """FEAT-44: a str locale works as well as the enum."""
        self.assertEqual(dl.invoker_t(SimpleNamespace(locale="ko")).locale, "ko")

    def test_missing_or_unusable_locale_is_english(self):
        """FEAT-44: no .locale attribute, or None, gives en."""
        self.assertEqual(dl.invoker_t(object()).locale, "en")
        self.assertEqual(dl.invoker_t(SimpleNamespace(locale=None)).locale, "en")
        self.assertEqual(dl.invoker_locale(object()), "en")

    def test_pseudo_locale_gives_pseudo_text(self):
        """FEAT-44: with pseudo enabled, an interaction in 'pseudo' is answered in pseudo text."""
        self.pseudo_on()
        t = dl.invoker_t(SimpleNamespace(locale="pseudo"))
        self.assertEqual(t.locale, "pseudo")
        self.assertEqual(t("Reply sent."), i18n.pseudo("Reply sent."))
        self.assertNotEqual(t("Reply sent."), "Reply sent.")
        self.assertEqual(dl.invoker_t(SimpleNamespace(locale=None))("Reply sent."), "Reply sent.")

    def test_pseudo_interaction_is_english_when_pseudo_is_off(self):
        """FEAT-44: without LLMCORD_I18N_PSEUDO, a 'pseudo' locale is not available and gives en."""
        self.assertEqual(dl.invoker_t(SimpleNamespace(locale="pseudo")).locale, "en")


class ChannelTests(LocaleCase):
    def source(self, mapping):
        dl.set_server_language_source(lambda guild_id: mapping.get(guild_id, ""))
        return mapping

    def test_no_guild_is_english(self):
        """FEAT-44: channel_t(None) is English even with a source registered."""
        self.source({G1: "ko"})
        self.assertEqual(dl.channel_t(None).locale, "en")

    def test_no_source_follows_discord_then_english(self):
        """FEAT-44: with no source, a community guild's preferred locale decides, else en."""
        self.assertEqual(dl.channel_t(guild(["COMMUNITY"], discord.Locale.korean)).locale, "ko")
        self.assertEqual(dl.channel_t(guild([], discord.Locale.korean)).locale, "en")

    def test_server_language_wins_over_discord(self):
        """FEAT-44: the stored server language beats the community preferred locale, even for a non-community guild."""
        self.source({G1: "ko"})
        self.assertEqual(dl.channel_t(guild(["COMMUNITY"], discord.Locale.japanese)).locale, "ko")
        self.assertEqual(dl.channel_t(guild([], discord.Locale.japanese)).locale, "ko")

    def test_empty_server_language_uses_community_preferred_locale(self):
        """FEAT-44: '' follows Discord: a COMMUNITY guild's preferred_locale, then en."""
        self.source({})
        self.assertEqual(dl.channel_t(guild(["COMMUNITY"], discord.Locale.korean)).locale, "ko")
        self.assertEqual(dl.channel_t(guild(["COMMUNITY"], discord.Locale.japanese)).locale, "en")
        self.assertEqual(dl.channel_t(guild(["COMMUNITY"], discord.Locale.american_english)).locale, "en")

    def test_non_community_preferred_locale_is_ignored(self):
        """FEAT-44: a guild without COMMUNITY has no meaningful preferred_locale, so it is ignored."""
        self.source({})
        self.assertEqual(dl.channel_t(guild([], discord.Locale.korean)).locale, "en")
        self.assertEqual(dl.channel_t(guild(["NEWS"], discord.Locale.korean)).locale, "en")

    def test_source_receives_the_guild_id_and_is_read_on_every_call(self):
        """FEAT-44: the source is called with guild.id each time; a changed value shows up on the next call."""
        calls = []
        stored = {"value": ""}

        def fn(guild_id):
            calls.append(guild_id)
            return stored["value"]

        dl.set_server_language_source(fn)
        g = guild([], discord.Locale.american_english, guild_id=42)
        self.assertEqual(dl.channel_t(g).locale, "en")
        stored["value"] = "ko"
        self.assertEqual(dl.channel_t(g).locale, "ko")
        stored["value"] = ""
        self.assertEqual(dl.channel_t(g).locale, "en")
        self.assertEqual(calls, [42, 42, 42])

    def test_real_store_change_between_two_calls(self):
        """FEAT-44: wiring the Store as the source, set_server_language takes effect on the very next channel_t."""
        store = Store()
        self.addCleanup(store.close)
        dl.set_server_language_source(store.server_language)
        g = guild([], discord.Locale.american_english)
        self.assertEqual(dl.channel_t(g).locale, "en")
        store.set_server_language(G1, "ko")
        self.assertEqual(dl.channel_t(g).locale, "ko")
        self.assertEqual(dl.channel_t(guild([], discord.Locale.american_english, guild_id=G2)).locale, "en")

    def test_clearing_the_source_stops_using_it(self):
        """FEAT-44: set_server_language_source(None) clears the registration."""
        self.source({G1: "ko"})
        self.assertEqual(dl.channel_t(guild()).locale, "ko")
        dl.set_server_language_source(None)
        self.assertEqual(dl.channel_t(guild()).locale, "en")

    def test_raising_source_logs_a_warning_and_falls_back(self):
        """FEAT-44: a source that raises is logged at warning and treated as ''."""
        def boom(guild_id):
            raise RuntimeError("db is down")

        dl.set_server_language_source(boom)
        with self.assertLogs("llmcord_core.i18n", logging.WARNING) as logs:
            t = dl.channel_t(guild(["COMMUNITY"], discord.Locale.korean, guild_id=987001))
        self.assertEqual(t.locale, "ko")  # fell back to '' then to Discord
        self.assertGreaterEqual(len(logs.records), 1)
        with self.assertLogs("llmcord_core.i18n", logging.WARNING):
            self.assertEqual(dl.channel_t(guild([], guild_id=987002)).locale, "en")

    def test_guild_without_id_falls_back_with_a_source_registered(self):
        """FEAT-44: a guild stub with no id never reaches the source and never raises."""
        self.source({G1: "pseudo"})
        self.assertEqual(dl.channel_t(SimpleNamespace(features=[])).locale, "en")
        stub = SimpleNamespace(features=["COMMUNITY"], preferred_locale=discord.Locale.korean)
        self.assertEqual(dl.channel_t(stub).locale, "ko")

    def test_stored_pseudo_language_renders_pseudo(self):
        """FEAT-44: a representative channel renderer (the usage footer) is pseudo-translated for server language 'pseudo'."""
        self.pseudo_on()
        self.source({G1: "pseudo"})
        usage = SimpleNamespace(input_tokens=10, output_tokens=20, reasoning_tokens=0, cost_usd=0.5, cost_basis="paid-list")
        t = dl.channel_t(guild())
        self.assertEqual(t.locale, "pseudo")
        out = reply_footer("model-x", usage, t=t)
        english = reply_footer("model-x", usage, t=i18n.english())
        self.assertNotEqual(out, english)
        self.assertIn("[!!", out)
        self.assertIn("model-x", out)
        self.assertIn("$0.500000", out)

    def test_english_channel_render_is_byte_identical(self):
        """FEAT-44: in en the same renderer's output is byte-identical to the English translator's."""
        self.source({})
        usage = SimpleNamespace(input_tokens=10, output_tokens=20, reasoning_tokens=3, cost_usd=0.5, cost_basis="paid-list")
        for g in (guild(), guild(["COMMUNITY"], discord.Locale.american_english), None):
            self.assertEqual(reply_footer("model-x", usage, t=dl.channel_t(g)), reply_footer("model-x", usage, t=i18n.english()))
            self.assertEqual(reply_footer("m", None, t=dl.channel_t(g)), reply_footer("m", None))


class SetupHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_setup_hook_registers_the_store_as_the_server_language_source(self):
        """FEAT-44: after setup_hook, channel_t reads the bot store's server language."""
        dl.set_server_language_source(None)
        self.addCleanup(dl.set_server_language_source, None)
        bot = SkitBot(make_settings())
        self.addCleanup(bot.store.close)
        self.addAsyncCleanup(bot.games.close)
        self.addAsyncCleanup(bot.doubt.close)
        self.addAsyncCleanup(bot.holdem.close)
        bot.tree.sync = mock.AsyncMock()
        bot.games.recover, bot.doubt.recover, bot.holdem.recover = mock.AsyncMock(), mock.AsyncMock(), mock.AsyncMock()
        await bot.setup_hook()
        self.addCleanup(bot.cleanup_task.cancel)
        plain = guild([], discord.Locale.american_english)
        self.assertEqual(dl.channel_t(plain).locale, "en")
        bot.store.set_server_language(G1, "ko")
        self.assertEqual(dl.channel_t(plain).locale, "ko")


class AvailableLocalesCacheTests(LocaleCase):
    def make_locale(self, root, name):
        folder = Path(root) / name / "LC_MESSAGES"
        folder.mkdir(parents=True)
        (folder / "commands.po").write_text('msgid ""\nmsgstr ""\n', encoding="utf-8")

    def test_same_tuple_until_the_clear_hook(self):
        """FEAT-44 (MNT-69): available_locales is cached per root, and clear_cache invalidates it."""
        with tempfile.TemporaryDirectory() as folder:
            self.make_locale(folder, "ko")
            first = catalog.available_locales(folder)
            self.assertEqual(first, ("en", "ko"))
            self.assertIs(catalog.available_locales(folder), first)
            self.make_locale(folder, "ja")
            self.assertIs(catalog.available_locales(folder), first)  # the folder scan is not repeated
            clear_cache()
            self.assertEqual(catalog.available_locales(folder), ("en", "ja", "ko"))

    def test_cache_is_per_root(self):
        """FEAT-44 (MNT-69): different roots do not share an entry."""
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            self.make_locale(one, "ko")
            self.assertEqual(catalog.available_locales(one), ("en", "ko"))
            self.assertEqual(catalog.available_locales(two), ("en",))

    def test_cache_is_per_pseudo_setting(self):
        """FEAT-44 (MNT-69): switching LLMCORD_I18N_PSEUDO changes the result without a manual clear."""
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(catalog.available_locales(folder), ("en",))
            os.environ["LLMCORD_I18N_PSEUDO"] = "1"
            self.assertEqual(catalog.available_locales(folder), ("en", "pseudo"))
            del os.environ["LLMCORD_I18N_PSEUDO"]
            self.assertEqual(catalog.available_locales(folder), ("en",))

    def test_default_root_result_is_cached(self):
        """FEAT-44 (MNT-69): the default (bundled) root is cached as well."""
        self.assertIs(catalog.available_locales(), catalog.available_locales())
        self.assertIn("ko", catalog.available_locales())


if __name__ == "__main__":
    unittest.main()
