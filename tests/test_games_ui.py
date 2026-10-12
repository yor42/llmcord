"""FEAT-21 part D: the Games tab builds one section per registered game (registry order), and the doubt settings write per guild."""
import unittest
from unittest import mock

from test_game_tables import G, TableCase

from llmcord_core import games_ui
from llmcord_core.admin_store import ConflictError
from llmcord_core.games import holdem, registry


class GamesTabSections(unittest.TestCase):
    def test_every_registered_game_has_a_section_in_registry_order(self):
        self.assertEqual(tuple(games_ui.GAME_SECTIONS), tuple(registry.keys()))
        self.assertEqual(tuple(games_ui.GAME_SECTIONS), ('blackjack', 'doubt', 'holdem'))
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


class HoldemSettings(TableCase):
    DEFAULT = {'small_blind': 5, 'min_buy_in_bb': 40, 'max_buy_in_bb': 200, 'action_timeout': 45, 'assist': False}

    def test_defaults_and_enable_switch(self):
        """FEAT-33 D: Hold'em is on by default with the default blinds, buy-in range, timeout and assist off."""
        self.assertTrue(self.store.game_enabled(G, 'holdem'))
        self.assertEqual(self.store.game_rules(G, 'holdem'), self.DEFAULT)

    def test_rules_round_trip_through_normalize_and_stay_in_their_guild_and_game(self):
        rules = {'small_blind': 25, 'min_buy_in_bb': 50, 'max_buy_in_bb': 400, 'action_timeout': 90, 'assist': True}
        before = {g: self.store.game_rules(G, g) for g in ('blackjack', 'doubt')}
        self.assertEqual(self.store.set_game_rules(G, 'holdem', rules, self.store.game_rules(G, 'holdem')), rules)
        self.assertEqual(self.store.game_rules(G, 'holdem'), holdem.normalize_rules(rules))
        self.assertEqual(self.store.game_rules(G, 'holdem'), rules)
        self.assertEqual({g: self.store.game_rules(G, g) for g in ('blackjack', 'doubt')}, before)
        self.assertEqual(self.store.game_rules(2, 'holdem'), self.DEFAULT)
        self.store.set_game_enabled(G, 'holdem', False, True)
        self.assertTrue(self.store.game_enabled(2, 'holdem'))
        self.assertTrue(self.store.game_enabled(G, 'doubt'))
        self.assertEqual(self.store.game_rules(G, 'holdem'), rules)

    def test_stale_rules_and_switch_conflict(self):
        stale = self.store.game_rules(G, 'holdem')
        self.store.set_game_rules(G, 'holdem', {'small_blind': 10})
        with self.assertRaisesRegex(ConflictError, "^The Texas Hold'em rules were changed elsewhere. Reload the page and try again.$"):
            self.store.set_game_rules(G, 'holdem', {'small_blind': 20}, stale)
        self.assertEqual(self.store.game_rules(G, 'holdem')['small_blind'], 10)
        self.store.set_game_enabled(G, 'holdem', False)
        with self.assertRaisesRegex(ConflictError, "^The Texas Hold'em setting was changed elsewhere. Reload the page and try again.$"):
            self.store.set_game_enabled(G, 'holdem', True, True)

    def test_bad_rules_are_refused_and_nothing_is_saved(self):
        for bad in ({'small_blind': 0}, {'small_blind': 10 ** 6 + 1}, {'small_blind': 2.5}, {'small_blind': None},
                    {'min_buy_in_bb': 19}, {'max_buy_in_bb': 1001}, {'min_buy_in_bb': 100, 'max_buy_in_bb': 50},
                    {'action_timeout': 14}, {'action_timeout': 301}, {'action_timeout': None}, {'assist': 1}, {'assist': None}):
            with self.assertRaises(ValueError, msg=bad):
                self.store.set_game_rules(G, 'holdem', bad)
        self.assertEqual(self.store.game_rules(G, 'holdem'), self.DEFAULT)

    def test_rules_at_the_limits_are_accepted(self):
        for ok in ({'small_blind': 1, 'min_buy_in_bb': 20, 'max_buy_in_bb': 20, 'action_timeout': 15},
                   {'small_blind': 10 ** 6, 'min_buy_in_bb': 1000, 'max_buy_in_bb': 1000, 'action_timeout': 300}):
            self.assertEqual(self.store.set_game_rules(G, 'holdem', ok), {**self.DEFAULT, **ok})

    def test_character_moves_setting_is_shared_and_per_guild(self):
        """Character moves is one server-wide setting for all games, not a Hold'em rule."""
        self.assertEqual(self.store.game_character_moves(G), 'model')
        self.assertEqual(self.store.set_game_character_moves(G, 'style', 'model'), 'style')
        self.assertEqual(self.store.game_character_moves(G), 'style')
        self.assertEqual(self.store.game_character_moves(2), 'model')
        self.assertNotIn('character_moves', self.store.game_rules(G, 'holdem'))
        with self.assertRaises(ConflictError):
            self.store.set_game_character_moves(G, 'model', 'model')


class SpokenMovesSetting(TableCase):
    def test_default_off_save_round_trip_stays_in_guild(self):
        """FEAT-37 E: Read moves from chat is off by default, saves per server and leaves guild 2 alone."""
        self.assertFalse(self.store.spoken_moves_enabled(G))
        self.assertIs(self.store.set_spoken_moves_enabled(G, True, False), True)
        self.assertTrue(self.store.spoken_moves_enabled(G))
        self.assertFalse(self.store.spoken_moves_enabled(2))
        self.assertIs(self.store.set_spoken_moves_enabled(G, False, True), False)
        self.assertFalse(self.store.spoken_moves_enabled(G))

    def test_stale_save_conflicts(self):
        self.store.set_spoken_moves_enabled(G, True)
        with self.assertRaises(ConflictError):
            self.store.set_spoken_moves_enabled(G, False, False)
        self.assertTrue(self.store.spoken_moves_enabled(G))


class KeepPhrasesSetting(TableCase):
    def test_default_off_save_round_trip_stays_in_guild(self):
        """FEAT-37 F3: Keep phrases for tuning is off by default, saves per server and leaves guild 2 alone."""
        self.store.set_spoken_moves_enabled(G, True)
        self.store.set_spoken_moves_enabled(2, True)
        self.assertFalse(self.store.spoken_keep_phrases(G))
        self.assertIs(self.store.set_spoken_keep_phrases(G, True, False), True)
        self.assertTrue(self.store.spoken_keep_phrases(G))
        self.assertFalse(self.store.spoken_keep_phrases(2))

    def test_setting_reads_stored_flag_while_read_moves_is_off(self):
        """FEAT-37 F3: the dashboard switch shows the stored flag and saves against it without a false conflict, even while Read moves from chat is off."""
        self.store.set_spoken_moves_enabled(G, True)
        self.store.set_spoken_keep_phrases(G, True)
        self.store.set_spoken_moves_enabled(G, False)
        self.assertTrue(self.store.spoken_keep_setting(G))
        self.assertFalse(self.store.spoken_keep_phrases(G))
        self.assertIs(self.store.set_spoken_keep_phrases(G, False, self.store.spoken_keep_setting(G)), False)
        self.assertFalse(self.store.spoken_keep_setting(G))

    def test_stale_save_conflicts(self):
        self.store.set_spoken_keep_phrases(G, True)
        with self.assertRaises(ConflictError):
            self.store.set_spoken_keep_phrases(G, False, False)


class KeptPhraseRows(TableCase):
    def keep(self, text, offered, label, user=7):
        self.store.set_spoken_moves_enabled(G, True)
        self.store.set_spoken_keep_phrases(G, True)
        return self.store.keep_spoken_phrase(G, user, 'blackjack', text, offered, 'unsure', {}, 'false_positive', label)

    def test_rows_describe_the_check_and_the_meaning_without_naming_anyone(self):
        self.keep('hit me', ['hit'], 'talk')
        self.keep('bust it', ['hit', 'stand'], 'stand')
        self.keep('what', [], 'hit')
        rows = games_ui.phrase_rows(self.store.spoken_phrases(G, 'blackjack', 200), 'UTC')
        self.assertEqual([r['text'] for r in rows], ['what', 'bust it', 'hit me'])
        self.assertEqual([r['check'] for r in rows], ['No check', 'Offered Hit or Stand', 'Offered Hit'])
        self.assertEqual([r['meant'] for r in rows], ['Meant: Hit', 'Meant: Stand', 'Meant: table talk'])
        self.assertTrue(all('user' not in k for r in rows for k in r))

    def test_promote_delete_and_remove_are_guild_scoped_and_revision_checked(self):
        first, second = self.keep('hit me', ['hit'], 'talk'), self.keep('stay put', ['hit'], 'talk')
        revision = self.store.spoken_examples_revision(G)
        self.store.promote_spoken_phrase(G, first, 'hit', 99, revision)
        self.assertEqual([(e['text'], e['label']) for e in self.store.spoken_examples(G, 'blackjack')], [('hit me', 'hit')])
        self.assertEqual([p['id'] for p in self.store.spoken_phrases(G)], [second])
        with self.assertRaises(ConflictError):
            self.store.promote_spoken_phrase(G, second, 'stand', 99, revision)
        self.assertFalse(self.store.delete_spoken_phrase(2, second))
        self.assertTrue(self.store.delete_spoken_phrase(G, second))
        example = self.store.spoken_examples(G, 'blackjack')[0]['id']
        with self.assertRaises(ConflictError):
            self.store.delete_spoken_example(G, example, revision)
        self.assertFalse(self.store.delete_spoken_example(2, example))
        self.assertTrue(self.store.delete_spoken_example(G, example, self.store.spoken_examples_revision(G)))

    def test_keeping_off_deletes_kept_phrases_but_not_examples(self):
        first = self.keep('hit me', ['hit'], 'talk')
        self.keep('stay put', ['hit'], 'talk')
        self.store.promote_spoken_phrase(G, first, 'hit', 99)
        self.store.set_spoken_keep_phrases(G, False, True)
        self.assertEqual(self.store.spoken_phrases(G), [])
        self.assertEqual(len(self.store.spoken_examples(G, 'blackjack')), 1)

class KeepConfirmAndRefusals(TableCase):
    def keep(self, text, user=7):
        self.store.set_spoken_moves_enabled(G, True)
        self.store.set_spoken_keep_phrases(G, True)
        return self.store.keep_spoken_phrase(G, user, 'blackjack', text, ['hit'], 'unsure', {}, 'false_positive', 'talk')

    def test_confirm_counts_every_kept_phrase_in_the_server(self):
        self.keep('one'), self.keep('two')
        self.store.db.execute("UPDATE spoken_phrases SET game='holdem' WHERE text='two'")
        ask = games_ui.keep_confirm(self.store, G, True, False)
        self.assertEqual(ask, ('Turn off keeping?', ['2 kept phrases are deleted. Example phrases stay.'], 'Turn off and delete'))
        self.assertIsNone(games_ui.keep_confirm(self.store, G, True, True))
        self.assertIsNone(games_ui.keep_confirm(self.store, G, False, False))

    def test_confirm_is_silent_with_nothing_kept(self):
        self.assertIsNone(games_ui.keep_confirm(self.store, G, True, False))

    def test_a_failing_count_still_asks_without_a_number(self):
        with mock.patch.object(self.store, 'spoken_phrases', side_effect=RuntimeError('db')), self.assertLogs(level='ERROR'):
            ask = games_ui.keep_confirm(self.store, G, True, False)
        self.assertEqual(ask, ('Turn off keeping?', ['Every kept phrase is deleted. Example phrases stay.'], 'Turn off and delete'))

    def test_a_refusal_becomes_a_conflict_with_the_same_message(self):
        with self.assertRaisesRegex(ConflictError, '^That phrase is no longer there.$'):
            games_ui.refresh_on_refusal(lambda: self.store.promote_spoken_phrase(G, 999, 'hit', 1))
        with self.assertRaises(ConflictError):
            games_ui.refresh_on_refusal(lambda: (_ for _ in ()).throw(ConflictError('x')))
        self.assertEqual(games_ui.refresh_on_refusal(lambda: 5), 5)
        with self.assertRaises(RuntimeError):
            games_ui.refresh_on_refusal(lambda: (_ for _ in ()).throw(RuntimeError('x')))

    def test_help_text_names_the_member_command(self):
        import inspect
        source = inspect.getsource(games_ui)
        self.assertIn('/game spoken_moves off', source)
        self.assertNotIn('/games spoken_moves', source)



if __name__ == '__main__':
    unittest.main()
