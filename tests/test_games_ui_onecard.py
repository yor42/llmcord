"""FEAT-46: One Card settings in the Games tab: the store API the section writes through (per guild, revision checked, validated)."""
import unittest

from test_game_tables import G, TableCase

from llmcord_core.admin_store import ConflictError
from llmcord_core.games import onecard


class OneCardSettings(TableCase):
    def test_defaults_enabled_and_per_guild_rules(self):
        """FEAT-46: One Card is on by default with the default rules; saving rules or the switch touches only that guild and game."""
        self.assertTrue(self.store.game_enabled(G, 'onecard'))
        self.assertEqual(self.store.game_rules(G, 'onecard'), onecard.normalize_rules({}))
        before = self.store.game_rules(G, 'doubt')
        rules = onecard.normalize_rules({'call_out_penalty': 2, 'last_card': 'no_attack', 'declare_phrases': ['one card']})
        self.assertEqual(self.store.set_game_rules(G, 'onecard', rules, self.store.game_rules(G, 'onecard')), rules)
        self.store.set_game_enabled(G, 'onecard', False, True)
        self.assertEqual(self.store.game_rules(G, 'onecard'), rules)
        self.assertEqual(self.store.game_rules(2, 'onecard'), onecard.normalize_rules({}))
        self.assertTrue(self.store.game_enabled(2, 'onecard'))
        self.assertEqual(self.store.game_rules(G, 'doubt'), before)
        self.assertTrue(self.store.game_enabled(G, 'doubt'))

    def test_stale_rules_and_switch_conflict(self):
        """FEAT-46: a stale save of the rules or the switch raises the 'changed elsewhere' conflict and writes nothing."""
        stale = self.store.game_rules(G, 'onecard')
        self.store.set_game_rules(G, 'onecard', {'ante': 5})
        with self.assertRaisesRegex(ConflictError, '^The One Card rules were changed elsewhere. Reload the page and try again.$'):
            self.store.set_game_rules(G, 'onecard', {'ante': 9}, stale)
        self.assertEqual(self.store.game_rules(G, 'onecard')['ante'], 5)
        self.store.set_game_enabled(G, 'onecard', False)
        with self.assertRaisesRegex(ConflictError, '^The One Card setting was changed elsewhere. Reload the page and try again.$'):
            self.store.set_game_enabled(G, 'onecard', True, True)

    def test_out_of_range_values_are_refused(self):
        """FEAT-46: bankruptcy limit 15-30, call-out penalty 1-3, timeout 15-300 and a valid last card rule; nothing is saved on refusal."""
        for bad in ({'bankruptcy_limit': 14}, {'bankruptcy_limit': 31}, {'call_out_penalty': 0}, {'call_out_penalty': 4},
                    {'turn_timeout': 14}, {'last_card': 'always'}, {'ante': -1}, {'bankruptcy_limit': None}):
            with self.assertRaises(ValueError, msg=bad):
                self.store.set_game_rules(G, 'onecard', bad)
        for ok in ({'bankruptcy_limit': 15}, {'bankruptcy_limit': 30}, {'call_out_penalty': 3}):
            self.assertEqual(self.store.set_game_rules(G, 'onecard', ok)[next(iter(ok))], next(iter(ok.values())))

    def test_declaration_phrases_must_be_one_to_eight_nonempty_strings(self):
        """FEAT-46: the store keeps 1 to 8 trimmed phrases and refuses an empty list."""
        self.assertEqual(self.store.set_game_rules(G, 'onecard', {'declare_phrases': ['  a  ', 'b']})['declare_phrases'], ['a', 'b'])
        for bad in ([], ['x'] * 9, ['  '], [1]):
            with self.assertRaisesRegex(ValueError, 'declaration phrases'):
                self.store.set_game_rules(G, 'onecard', {'declare_phrases': bad})
        self.assertEqual(self.store.game_rules(G, 'onecard')['declare_phrases'], ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
