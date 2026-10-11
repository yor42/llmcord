"""FEAT-37 C: the spoken-move server switches are per guild, off by default, and never listed as games."""
import unittest

from test_game_tables import G, TableCase

from llmcord_core import games_ui
from llmcord_core.admin_store import ConflictError
from llmcord_core.games import registry


class SpokenSettings(TableCase):
    def test_default_off(self):
        self.assertIs(self.store.spoken_moves_enabled(G), False)
        self.assertIs(self.store.spoken_keep_phrases(G), False)

    def test_round_trip(self):
        self.assertIs(self.store.set_spoken_moves_enabled(G, True), True)
        self.assertIs(self.store.spoken_moves_enabled(G), True)
        self.assertIs(self.store.set_spoken_moves_enabled(G, False, expected=True), False)
        self.assertIs(self.store.spoken_moves_enabled(G), False)

    def test_conflict_and_bad_value(self):
        with self.assertRaises(ConflictError):
            self.store.set_spoken_moves_enabled(G, True, expected=True)
        with self.assertRaises(ValueError):
            self.store.set_spoken_keep_phrases(G, 1)

    def test_guild_isolation(self):
        self.store.set_spoken_moves_enabled(G, True)
        self.store.set_spoken_keep_phrases(G, True)
        self.assertIs(self.store.spoken_moves_enabled(2), False)
        self.assertIs(self.store.spoken_keep_phrases(2), False)

    def test_keep_phrases_reads_false_while_main_is_off(self):
        self.store.set_spoken_keep_phrases(G, True)
        self.assertIs(self.store.spoken_keep_phrases(G), False)
        self.store.set_spoken_moves_enabled(G, True)
        self.assertIs(self.store.spoken_keep_phrases(G), True)
        self.store.set_spoken_moves_enabled(G, False)
        self.assertIs(self.store.spoken_keep_phrases(G), False)

    def test_pseudo_rows_are_not_games(self):
        self.store.set_spoken_moves_enabled(G, True)
        self.store.set_spoken_keep_phrases(G, True)
        for key in (self.store._SPOKEN_KEY, self.store._SPOKEN_KEEP_KEY, self.store._MOVES_KEY):
            self.assertNotIn(key, registry.keys())
            self.assertNotIn(key, games_ui.GAME_SECTIONS)


if __name__ == '__main__':
    unittest.main()
