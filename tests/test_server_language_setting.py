"""FEAT-44 (D39, batch 2c): the Bot language select in Server settings.

Seam: ``server_language_options`` / ``language_operation`` / ``language_detail`` over an in-memory ``Store``.
"""
import unittest

from llmcord_core import i18n
from llmcord_core.dashboard import LANGUAGE_FOLLOW, language_detail, language_operation, server_language_options
from llmcord_core.i18n import UserError, codes
from llmcord_core.store import Store


class ServerLanguageOptionsTests(unittest.TestCase):
    def test_follow_first_then_every_locale_by_own_name(self):
        """FEAT-44: the follow option comes first, then each available locale named in its own language."""
        options, value = server_language_options('')
        self.assertEqual(list(options)[0], LANGUAGE_FOLLOW)
        self.assertEqual(options[LANGUAGE_FOLLOW], "Follow each member's Discord language")
        self.assertEqual(list(options)[1:], list(i18n.available_locales()))
        for code in i18n.available_locales():
            self.assertEqual(options[code], i18n.display_name(code))
        self.assertEqual(value, LANGUAGE_FOLLOW)

    def test_stored_locale_is_selected(self):
        """FEAT-44: an installed stored language is the select's value."""
        self.assertEqual(server_language_options('ko')[1], 'ko')

    def test_uninstalled_stored_language_shows_as_follow(self):
        """FEAT-44: a stored language with no catalog falls back to the follow option."""
        self.assertEqual(server_language_options('xx')[1], LANGUAGE_FOLLOW)


class LanguageOperationTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()

    def test_saves_language_and_detail_is_the_code(self):
        """FEAT-44: choosing a language stores it; the audit detail is the stored code."""
        stored = language_operation(self.store, 1, 'ko')
        self.assertEqual(stored, 'ko')
        self.assertEqual(self.store.server_language(1), 'ko')
        self.assertEqual(language_detail(stored), {'language': 'ko'})

    def test_follow_option_stores_empty(self):
        """FEAT-44: the follow option (or no value) clears the language to ''."""
        for value in (LANGUAGE_FOLLOW, None):
            with self.subTest(value=value):
                self.store.set_server_language(1, 'ko')
                stored = language_operation(self.store, 1, value)
                self.assertEqual(stored, '')
                self.assertEqual(self.store.server_language(1), '')
                self.assertEqual(language_detail(stored), {'language': ''})

    def test_unknown_language_refused_and_store_unchanged(self):
        """FEAT-44: an unknown language raises UserError and keeps the existing value."""
        self.store.set_server_language(1, 'ko')
        with self.assertRaises(UserError) as caught:
            language_operation(self.store, 1, 'xx')
        self.assertEqual(caught.exception.code, codes.SERVER_LANGUAGE_UNKNOWN)
        self.assertEqual(self.store.server_language(1), 'ko')

    def test_guild_isolation(self):
        """FEAT-44: saving one guild's language leaves another guild's untouched."""
        self.store.set_server_language(2, 'ko')
        language_operation(self.store, 1, LANGUAGE_FOLLOW)
        self.assertEqual(self.store.server_language(2), 'ko')


if __name__ == '__main__':
    unittest.main()
