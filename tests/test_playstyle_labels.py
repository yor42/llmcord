"""FEAT-32 C: the dashboard's Play style captions cover exactly the play style traits."""
import unittest

from llmcord_core.dashboard import PLAY_STYLE_LABELS
from llmcord_core.games import playstyle


class PlayStyleLabelTests(unittest.TestCase):
    def test_labels_cover_exactly_the_traits_with_low_and_high_captions(self):
        """FEAT-32 C: one (label, low, high) entry per trait, and every grouped trait has one."""
        self.assertEqual(set(PLAY_STYLE_LABELS), set(playstyle.TRAITS))
        self.assertEqual(sorted(t for group in playstyle.GROUPS.values() for t in group), sorted(playstyle.TRAITS))
        for label, low, high in PLAY_STYLE_LABELS.values():
            self.assertTrue(label and low and high and low != high)


if __name__ == '__main__':
    unittest.main()
