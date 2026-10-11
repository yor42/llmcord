"""FEAT-22 part A: the item kind registry and recipe validation (pure, no Discord or database)."""
import copy
import unittest

from llmcord_core.items import RecipeError, get_kind, validate_recipe
from llmcord_core.items.registry import roll

BASE = {
    'key': 'ball', 'kind': 'random_message', 'name': 'Ball', 'emoji': 'x', 'consumable': False, 'default_price': 10,
    'default_cooldown_seconds': 5, 'audience': 'none',
    'tables': {'answer': {'en': ['yes', 'no']}}, 'template': {'en': {'text': '{answer}'}},
}


def recipe(**changes):
    data = copy.deepcopy(BASE)
    data.update(changes)
    return data


class RegistryTests(unittest.TestCase):
    def test_valid_recipe_loads(self):
        r = validate_recipe(recipe())
        self.assertEqual((r.key, r.kind, r.audience), ('ball', 'random_message', 'none'))

    def test_unknown_kind_rejected(self):
        with self.assertRaises(RecipeError):
            validate_recipe(recipe(kind='run_code'))
        with self.assertRaises(RecipeError):
            get_kind('nope')

    def test_wrong_audience_rejected(self):
        for audience in ('member', 'any', 'npc', 'bogus'):
            with self.assertRaises(RecipeError):
                validate_recipe(recipe(audience=audience))

    def test_bad_keys_rejected(self):
        for key in ('', 'Ball', 'has space', 'a' * 33, 'dash-ed', 5):
            with self.assertRaises(RecipeError, msg=repr(key)):
                validate_recipe(recipe(key=key))

    def test_unknown_top_level_field_and_param_rejected(self):
        with self.assertRaises(RecipeError):
            validate_recipe(recipe(script='print(1)'))
        with self.assertRaises(RecipeError):
            validate_recipe(recipe(params={'evil': 1}))

    def test_size_limits(self):
        with self.assertRaises(RecipeError):
            validate_recipe(recipe(name='n' * 41))
        with self.assertRaises(RecipeError):
            validate_recipe(recipe(tables={'answer': {'en': ['x'] * 501}}))
        with self.assertRaises(RecipeError):
            validate_recipe(recipe(tables={'answer': {'en': ['x' * 301]}}))
        with self.assertRaises(RecipeError):
            validate_recipe(recipe(tables={'answer': {'en': ['x'] * 400}, 'b': {'en': ['y' * 300] * 400}}))
        with self.assertRaises(RecipeError):
            validate_recipe(recipe(tables={'answer': {'en': []}}))

    def test_sides_param_bounds(self):
        tables = {'roll': {'range': [1, 'sides']}}
        template = {'en': 'got {roll}'}
        for sides in (1, 1001, 'six', True, 6.5):
            with self.assertRaises(RecipeError, msg=repr(sides)):
                validate_recipe(recipe(params={'sides': sides}, tables=tables, template=template))
        validate_recipe(recipe(params={'sides': 1000}, tables=tables, template=template))
        with self.assertRaises(RecipeError):
            validate_recipe(recipe(tables=tables, template=template))

    def test_locale_keyed_text_needs_en(self):
        with self.assertRaises(RecipeError):
            validate_recipe(recipe(tables={'answer': {'ko': ['yes']}}))
        with self.assertRaises(RecipeError):
            validate_recipe(recipe(template={'ko': {'text': '{answer}'}}))

    def test_template_placeholders_are_whitelisted(self):
        for text in ('{0.__class__}', '{answer.__class__}', '{answer[0]}', '{missing}', '{answer!r}', '{answer:>10}', '{', '{0}'):
            with self.assertRaises(RecipeError, msg=text):
                validate_recipe(recipe(template={'en': {'text': text}}))
        r = validate_recipe(recipe(template={'en': {'text': '{{literal}} {answer}'}}))
        self.assertTrue(roll(r, 1).text.startswith('{literal} '))

    def test_consumable_and_reusable_flags(self):
        reusable = validate_recipe(recipe())
        self.assertTrue(reusable.reusable)
        self.assertEqual(reusable.hold_cap, 1)
        self.assertEqual(reusable.default_cooldown_seconds, 5)
        consumable = validate_recipe(recipe(consumable=True))
        self.assertFalse(consumable.reusable)
        self.assertIsNone(consumable.hold_cap)
        self.assertEqual(consumable.default_cooldown_seconds, 0)

    def test_wrong_types_rejected(self):
        for changes in ({'default_price': True}, {'consumable': 'yes'}, {'default_price': -1}, {'tables': {'answer': {5: ['x'], 'en': ['y']}}},
                        {'tables': {'answer': {'en': ['y']}}, 'template': {5: {'text': '{answer}'}, 'en': {'text': '{answer}'}}}):
            with self.assertRaises(RecipeError, msg=repr(changes)):
                validate_recipe(recipe(**changes))
        for chance in (True, 2, -1, 'half'):
            with self.assertRaises(RecipeError, msg=repr(chance)):
                validate_recipe(recipe(params={'reversed_chance': chance}))

    def test_recipe_is_immutable(self):
        r = validate_recipe(recipe())
        with self.assertRaises(AttributeError):
            r.name = 'other'

    def test_deeply_nested_recipe_rejected(self):
        deep = []
        for _ in range(5000):
            deep = [deep]
        with self.assertRaises(RecipeError):
            validate_recipe(recipe(params={'sides': deep}))

    def test_non_dict_and_non_json_rejected(self):
        with self.assertRaises(RecipeError):
            validate_recipe([])
        with self.assertRaises(RecipeError):
            validate_recipe(recipe(params={'sides': object()}))


if __name__ == '__main__':
    unittest.main()
