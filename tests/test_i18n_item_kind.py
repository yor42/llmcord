import unittest
from types import SimpleNamespace

from llmcord_core.i18n import Translator
from llmcord_core.shop_ui import kind_line


class ItemKindLabelTests(unittest.TestCase):
    def test_english_unchanged(self):
        t = Translator('en')
        self.assertEqual(kind_line(SimpleNamespace(kind='random_message', consumable=False), t), 'Random message · Reusable')
        self.assertEqual(kind_line(SimpleNamespace(kind='random_message', consumable=True), t), 'Random message · Consumable')

    def test_korean_kind_word(self):
        t = Translator('ko')
        for consumable in (False, True):
            self.assertIn('랜덤 메시지', kind_line(SimpleNamespace(kind='random_message', consumable=consumable), t))

    def test_unknown_kind_falls_back_to_derived_text(self):
        line = kind_line(SimpleNamespace(kind='future_thing', consumable=False), Translator('en'))
        self.assertEqual(line, 'Future thing · Reusable')


if __name__ == '__main__':
    unittest.main()
