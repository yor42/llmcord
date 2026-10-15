"""MNT-83 Korean follow-ups: dashboard item-error names, recipe-held tarot orientation words, Currency tab hint."""
import json
import unittest
from pathlib import Path

from llmcord_core import items
from llmcord_core.admin_store import ConflictError
from llmcord_core.i18n import Translator, UserError, codes
from llmcord_core.items import RecipeError, roll, validate_recipe
from llmcord_core.shop_ui import error_text

TAROT = {r.key: r for r in items.all_items()}['tarot']
RECIPES = Path(items.__file__).parent / 'recipes'


def tarot_data(**kw):
    data = json.loads((RECIPES / 'tarot.json').read_text(encoding='utf-8'))
    data.update(kw)
    return data


class DashboardItemErrorNameTests(unittest.TestCase):
    def test_no_setting_error_shows_the_korean_item_name(self):
        """MNT-83a: the dashboard error ITEM_NO_SETTING names the item in the viewer's language."""
        error = UserError(codes.ITEM_NO_SETTING, name='Tarot', item='tarot', settings='x')
        self.assertIn('타로', error_text(error, Translator('ko')))
        self.assertIn('Tarot', error_text(error, Translator('en')))
        self.assertEqual(error.params['name'], 'Tarot')

    def test_changed_error_shows_the_korean_item_name(self):
        """MNT-83a: the dashboard error ITEM_CHANGED names the item in the viewer's language."""
        error = ConflictError(codes.ITEM_CHANGED, name='Tarot', item='tarot')
        self.assertIn('타로', error_text(error, Translator('ko')))

    def test_unknown_item_or_plain_error_is_unchanged(self):
        """MNT-83a: an unknown item key keeps the given name; an error without an item is rendered as before."""
        gone = UserError(codes.ITEM_CHANGED, name='Old', item='gone')
        self.assertIn('Old', error_text(gone, Translator('ko')))
        self.assertEqual(error_text(ValueError('plain'), Translator('ko')), Translator('ko').error(ValueError('plain')))


class OrientationDataTests(unittest.TestCase):
    def test_tarot_recipe_holds_its_orientation_words(self):
        """MNT-83b: the words live in the recipe JSON, locale-keyed."""
        self.assertEqual(dict(TAROT.orientation), {'en': ['Upright', 'Reversed'], 'ko': ['정방향', '역방향']})

    def test_english_and_korean_rolls_use_the_recipe_words(self):
        """MNT-83b: seeded English results are unchanged; Korean uses the Korean words."""
        for seed in range(40):
            en, ko = roll(TAROT, seed, 'en').text, roll(TAROT, seed, 'ko').text
            self.assertTrue('(Upright)' in en or '(Reversed)' in en)
            self.assertTrue('(정방향)' in ko or '(역방향)' in ko)
            self.assertEqual('(Reversed)' in en, '(역방향)' in ko)

    def test_a_new_language_needs_only_data(self):
        """MNT-83b: a recipe adding a locale gets its words; an unlisted locale falls back to English."""
        recipe = validate_recipe(tarot_data(orientation={'en': ['Upright', 'Reversed'], 'fr': ['Droit', 'Renversé']}))
        seen = {('Droit' in roll(recipe, s, 'fr').text, 'Renversé' in roll(recipe, s, 'fr').text) for s in range(40)}
        self.assertEqual(seen, {(True, False), (False, True)})
        self.assertIn('(Upright)' if '(Upright)' in roll(recipe, 1, 'de').text else '(Reversed)', roll(recipe, 1, 'de').text)

    def test_missing_orientation_falls_back_to_english_words(self):
        """MNT-83b: a custom two-sided recipe without the field still rolls with Upright/Reversed."""
        data = tarot_data()
        del data['orientation']
        text = roll(validate_recipe(data), 3, 'ko').text
        self.assertTrue('(Upright)' in text or '(Reversed)' in text)

    def test_validation_rejects_bad_orientation(self):
        """MNT-83b: orientation must be locale-keyed with 'en', each a pair of non-empty short texts; unknown fields still fail."""
        for bad in ({'ko': ['a', 'b']}, {'en': ['a']}, {'en': ['a', 'b', 'c']}, {'en': ['a', '']}, {'en': ['a', 'x' * 41]}, {'en': 'ab'}, ['a', 'b']):
            with self.subTest(bad=bad), self.assertRaises(RecipeError):
                validate_recipe(tarot_data(orientation=bad))
        with self.assertRaises(RecipeError):
            validate_recipe(tarot_data(code='x'))

    def test_orientation_stays_out_of_the_code_module(self):
        """MNT-83b: random_message.py keeps no per-locale word table."""
        from llmcord_core.items import random_message
        self.assertNotIn('정방향', Path(random_message.__file__).read_text(encoding='utf-8'))


class CurrencyHintTests(unittest.TestCase):
    HINT = 'On a Korean-language server, rename the default currency coins to a Korean word such as 코인, so messages read naturally.'

    def test_hint_is_in_the_currency_tab_source_and_translated(self):
        """MNT-83c: the Currency name section carries the hint through the i18n layer, with Korean."""
        source = (Path(items.__file__).parent.parent / 'currency_ui.py').read_text(encoding='utf-8')
        self.assertIn(self.HINT, source)
        po = (Path(items.__file__).parent.parent / 'locales/ko/LC_MESSAGES/web.po').read_text(encoding='utf-8')
        self.assertIn(f'msgid "{self.HINT}"\nmsgstr "한국어', po)


if __name__ == '__main__':
    unittest.main()
