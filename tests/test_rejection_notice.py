import unittest

from llmcord_core.dashboard import NOTICE_WINDOW, should_notify


class ShouldNotifyTests(unittest.TestCase):
    def test_burst_of_same_notice_shows_once(self):
        """UI-04: identical rejection notices within the toast lifetime are shown once per client."""
        last, shown = None, 0
        for i in range(20):
            now = 100 + i * 0.1
            if should_notify(last, 'Nope', now):
                last, shown = ('Nope', now), shown + 1
        self.assertEqual(shown, 1)

    def test_different_notice_shows(self):
        """UI-04: a different notice text is not suppressed."""
        self.assertTrue(should_notify(('A', 100), 'B', 100.5))

    def test_notice_shows_again_after_window(self):
        """UI-04: after the 8 s window the same notice can show again."""
        self.assertFalse(should_notify(('A', 100), 'A', 100 + NOTICE_WINDOW - 0.1))
        self.assertTrue(should_notify(('A', 100), 'A', 100 + NOTICE_WINDOW))


if __name__ == '__main__':
    unittest.main()
