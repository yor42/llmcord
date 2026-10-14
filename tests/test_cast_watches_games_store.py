"""FEAT-48 / D40 (step 1): the server switch "Cast watches games" is a ``game_settings`` pseudo flag, default off.

Written before the implementation. Assumed seam (proposed, pinned here; the implementer creates it):

  ``Store.cast_watches_games(guild_id)`` / ``Store.set_cast_watches_games(guild_id, on, expected=None)``: same behavior as
  ``cast_sees_item_results`` (``tests/test_happenings.py``): default off, per guild, ``ConflictError`` on a stale ``expected``,
  ``ValueError`` for a non-bool, and no effect on the games list or on the sibling pseudo rows.
"""
import unittest

from llmcord_core.admin_store import ConflictError
from llmcord_core.store import Store

G, G2 = 1, 2


class WatchSwitchTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)

    def test_default_off_and_per_guild(self):
        self.assertFalse(self.store.cast_watches_games(G))
        self.assertTrue(self.store.set_cast_watches_games(G, True))
        self.assertTrue(self.store.cast_watches_games(G))
        self.assertFalse(self.store.cast_watches_games(G2))
        self.assertFalse(self.store.set_cast_watches_games(G, False))
        self.assertFalse(self.store.cast_watches_games(G))

    def test_pseudo_row_is_not_a_game(self):
        """The Games tab lists registry.keys() and reads game_enabled for each; the pseudo row must stay out of both."""
        from llmcord_core.games import registry
        before = {key: self.store.game_enabled(G, key) for key in registry.keys()}
        self.store.set_cast_watches_games(G, True)
        self.assertNotIn("cast_watches_games", registry.keys())
        self.assertEqual({key: self.store.game_enabled(G, key) for key in registry.keys()}, before)
        self.assertEqual(self.store.game_character_moves(G), "model")  # a sibling pseudo row is untouched

    def test_independent_of_the_item_results_switch(self):
        self.store.set_cast_watches_games(G, True)
        self.assertFalse(self.store.cast_sees_item_results(G))
        self.store.set_cast_sees_item_results(G, True)
        self.store.set_cast_watches_games(G, False)
        self.assertTrue(self.store.cast_sees_item_results(G))
        self.assertFalse(self.store.cast_watches_games(G))

    def test_stale_expected_conflicts_and_keeps_value(self):
        self.store.set_cast_watches_games(G, True)
        with self.assertRaises(ConflictError):
            self.store.set_cast_watches_games(G, False, expected=False)
        self.assertTrue(self.store.cast_watches_games(G))
        self.assertFalse(self.store.set_cast_watches_games(G, False, expected=True))

    def test_matching_expected_default_writes(self):
        self.assertTrue(self.store.set_cast_watches_games(G, True, expected=False))
        self.assertTrue(self.store.cast_watches_games(G))

    def test_expected_is_checked_per_guild(self):
        self.store.set_cast_watches_games(G2, True)
        self.assertTrue(self.store.set_cast_watches_games(G, True, expected=False))  # G2's value is not G's expectation

    def test_non_bool_is_refused(self):
        with self.assertRaises(ValueError):
            self.store.set_cast_watches_games(G, "yes")
        self.assertFalse(self.store.cast_watches_games(G))
