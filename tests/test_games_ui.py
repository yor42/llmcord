"""FEAT-21 part D: the Games tab builds one section per registered game (registry order), and the doubt settings write per guild."""
import unittest

from test_game_tables import G, TableCase

from llmcord_core import games_ui
from llmcord_core.admin_store import ConflictError
from llmcord_core.games import registry


# Registered games with no dashboard section yet; remove 'holdem' when FEAT-33 part D adds its section.
NO_SECTION_YET = ('holdem',)


class GamesTabSections(unittest.TestCase):
    def test_every_registered_game_has_a_section_in_registry_order(self):
        self.assertEqual(tuple(games_ui.GAME_SECTIONS), tuple(k for k in registry.keys() if k not in NO_SECTION_YET))
        self.assertEqual(tuple(games_ui.GAME_SECTIONS), ('blackjack', 'doubt'))
        self.assertTrue(all(callable(f) for f in games_ui.GAME_SECTIONS.values()))


class DoubtSettings(TableCase):
    def test_rules_save_only_for_doubt_in_the_right_guild(self):
        rules = {'claim_rule': 'sequence', 'ante': 50, 'turn_timeout': 90}
        before = self.store.game_rules(G, 'blackjack')
        self.assertEqual(self.store.set_game_rules(G, 'doubt', rules, self.store.game_rules(G, 'doubt')), rules)
        self.assertEqual(self.store.game_rules(G, 'doubt'), rules)
        self.assertEqual(self.store.game_rules(G, 'blackjack'), before)
        self.assertEqual(self.store.game_rules(2, 'doubt')['ante'], 0)

    def test_stale_rules_and_switch_conflict(self):
        stale = self.store.game_rules(G, 'doubt')
        self.store.set_game_rules(G, 'doubt', {'ante': 5})
        with self.assertRaisesRegex(ConflictError, '^The I Doubt It rules were changed elsewhere. Reload the page and try again.$'):
            self.store.set_game_rules(G, 'doubt', {'ante': 9}, stale)
        self.store.set_game_enabled(G, 'doubt', False)
        with self.assertRaisesRegex(ConflictError, '^The I Doubt It setting was changed elsewhere. Reload the page and try again.$'):
            self.store.set_game_enabled(G, 'doubt', True, True)
        with self.assertRaisesRegex(ConflictError, '^The blackjack setting was changed elsewhere. Reload the page and try again.$'):
            self.store.set_game_enabled(G, 'blackjack', False, False)

    def test_cleared_field_gives_a_friendly_value_error(self):
        """A cleared Ante or Turn timeout reaches the store as None; the savebar shows this ValueError text."""
        with self.assertRaisesRegex(ValueError, 'ante must be a whole number'):
            self.store.set_game_rules(G, 'doubt', {'ante': None})
        with self.assertRaisesRegex(ValueError, 'turn timeout must be a whole number'):
            self.store.set_game_rules(G, 'doubt', {'turn_timeout': None})

    def test_bad_ante_and_timeout_are_refused(self):
        for bad in ({'ante': -1}, {'ante': 10 ** 9 + 1}, {'ante': 1.5}, {'turn_timeout': 14}, {'turn_timeout': 301}):
            with self.assertRaises(ValueError):
                self.store.set_game_rules(G, 'doubt', bad)


if __name__ == '__main__':
    unittest.main()
