"""FEAT-59: Yut settings in the Games tab: the store API the section writes through (per guild, revision checked, validated)."""
import unittest

from test_game_tables import G, TableCase

from llmcord_core.admin_store import ConflictError
from llmcord_core.games import yut


class YutSettings(TableCase):
    def test_defaults_enabled_and_per_guild_rules(self):
        """FEAT-59: Yut is on by default with the default rules; saving rules or the switch touches only that guild and game."""
        self.assertTrue(self.store.game_enabled(G, 'yut'))
        self.assertEqual(self.store.game_rules(G, 'yut'), yut.normalize_rules({}))
        before = self.store.game_rules(G, 'onecard')
        rules = yut.normalize_rules({'pieces': 3, 'backdo': False, 'ante': 20, 'turn_timeout': 90})
        self.assertEqual(self.store.set_game_rules(G, 'yut', rules, self.store.game_rules(G, 'yut')), rules)
        self.store.set_game_enabled(G, 'yut', False, True)
        self.assertEqual(self.store.game_rules(G, 'yut'), rules)
        self.assertEqual(self.store.game_rules(2, 'yut'), yut.normalize_rules({}))
        self.assertTrue(self.store.game_enabled(2, 'yut'))
        self.assertEqual(self.store.game_rules(G, 'onecard'), before)
        self.assertTrue(self.store.game_enabled(G, 'onecard'))

    def test_stale_rules_and_switch_conflict(self):
        """FEAT-59: a stale save of the rules or the switch raises the 'changed elsewhere' conflict and writes nothing."""
        stale = self.store.game_rules(G, 'yut')
        self.store.set_game_rules(G, 'yut', {'ante': 5})
        with self.assertRaisesRegex(ConflictError, '^The Yut rules were changed elsewhere. Reload the page and try again.$'):
            self.store.set_game_rules(G, 'yut', {'ante': 9}, stale)
        self.assertEqual(self.store.game_rules(G, 'yut')['ante'], 5)
        self.store.set_game_enabled(G, 'yut', False)
        with self.assertRaisesRegex(ConflictError, '^The Yut setting was changed elsewhere. Reload the page and try again.$'):
            self.store.set_game_enabled(G, 'yut', True, True)

    def test_out_of_range_values_are_refused(self):
        """FEAT-59: pieces 2-5, timeout 15-300, a boolean Back-do and a non-negative ante; nothing is saved on refusal."""
        for bad in ({'pieces': 1}, {'pieces': 6}, {'turn_timeout': 14}, {'turn_timeout': 301}, {'backdo': 'yes'}, {'ante': -1}, {'pieces': None}):
            with self.assertRaises(ValueError, msg=bad):
                self.store.set_game_rules(G, 'yut', bad)
        self.assertEqual(self.store.game_rules(G, 'yut'), yut.normalize_rules({}))
        for ok in ({'pieces': 2}, {'pieces': 5}, {'backdo': False}):
            self.assertEqual(self.store.set_game_rules(G, 'yut', ok)[next(iter(ok))], next(iter(ok.values())))


if __name__ == '__main__':
    unittest.main()
