"""Dashboard theme constants and server-initials helper (UI-28).

Seam: pure functions/constants in ``llmcord_core.dashboard``.
"""
import unittest

from llmcord_core.dashboard import THEME_NEGATIVE, THEME_PRIMARY, server_initials


class ServerInitialsTests(unittest.TestCase):
    def test_two_words(self):
        """UI-28: two words give the uppercased first letter of each."""
        self.assertEqual(server_initials('Test server'), 'TS')

    def test_only_first_two_words(self):
        """UI-28: words past the second are ignored."""
        self.assertEqual(server_initials('moonlit tavern night'), 'MT')

    def test_single_word(self):
        """UI-28: one word gives just its first character."""
        self.assertEqual(server_initials('Skit'), 'S')

    def test_extra_whitespace(self):
        """UI-28: leading, trailing and repeated whitespace is ignored."""
        self.assertEqual(server_initials('  spaced   out  '), 'SO')

    def test_non_ascii(self):
        """UI-28: non-ASCII first letters are kept and uppercased."""
        self.assertEqual(server_initials('한국 서버'), '한서')
        self.assertEqual(server_initials('émile zola'), 'ÉZ')

    def test_blank_gives_question_mark(self):
        """UI-28: None, empty and whitespace-only names give '?'."""
        for name in (None, '', '   '):
            with self.subTest(name=name):
                self.assertEqual(server_initials(name), '?')


class ThemeConstantTests(unittest.TestCase):
    def test_palette_anchors(self):
        """UI-28 (D20): primary and negative colors that later steps rely on."""
        self.assertEqual(THEME_PRIMARY, '#5865f2')
        self.assertEqual(THEME_NEGATIVE, '#da373c')


if __name__ == '__main__':
    unittest.main()
