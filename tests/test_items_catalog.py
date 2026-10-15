"""FEAT-22 part A: the built-in catalog and the random_message rolls."""
import json
import unittest

from llmcord_core.items import RecipeError, all_items, get_item, roll, validate_entries

KEYS = {'8ball', 'fortune_cookie', 'dice', 'd20', 'tarot'}
VAULT_KEYS = {'vault_residual_scanner', 'vault_spectrum_analyzer', 'vault_comparison_circuit', 'vault_position_verifier', 'vault_master_decoder'}  # FEAT-64
TOYS = [i for i in all_items() if i.kind == 'random_message']


class CatalogTests(unittest.TestCase):
    def test_catalog_contents_and_flags(self):
        self.assertEqual({i.key for i in all_items()}, KEYS | VAULT_KEYS)
        self.assertEqual({i.key for i in TOYS}, KEYS)
        facts = {i.key: (i.consumable, i.default_price, i.default_cooldown_seconds) for i in TOYS}
        self.assertEqual(facts, {'8ball': (False, 150, 30), 'fortune_cookie': (True, 25, 0), 'dice': (False, 50, 5),
                                 'd20': (False, 50, 5), 'tarot': (True, 60, 0)})
        self.assertEqual(get_item('dice').params['sides'], 6)
        self.assertEqual(get_item('d20').params['sides'], 20)
        self.assertEqual(get_item('8ball').hold_cap, 1)
        self.assertIsNone(get_item('tarot').hold_cap)
        with self.assertRaises(KeyError):
            get_item('nope')

    def test_every_builtin_rolls_200_seeds(self):
        for item in TOYS:
            for seed in range(200):
                res = roll(item, seed)
                self.assertTrue(res.text.strip(), item.key)
                json.dumps(res.to_json())

    def test_same_seed_same_result_and_seeds_differ(self):
        for item in TOYS:
            self.assertEqual(roll(item, 7), roll(item, 7))
            self.assertEqual(roll(item, 7).to_json(), roll(item, 7).to_json())
            self.assertGreater(len({roll(item, s).text for s in range(100)}), 1, item.key)

    def test_golden_values_pin_cross_version_stability(self):
        self.assertEqual(roll(get_item('dice'), 1).text, 'You rolled a **2** (1-6).')
        self.assertEqual(roll(get_item('8ball'), 1).text, 'You may rely on it.')

    def test_validate_entries(self):
        ball, tarot = get_item('8ball'), get_item('tarot')
        validate_entries(ball, {'en': ['ok']}, {'en': ['a']})
        validate_entries(ball)
        bad = [({'en': ['x'] * 481}, None), (None, {'en': []}), ({'en': 'x'}, None), ({'fr': ['x']}, None), (None, {'en': ['x' * 301]}),
               ({'en': ['']}, None), (None, {'en': ['y'] * 501}), ({'en': ['x'], 'ko': [5]}, None)]
        for extra, swap in bad:
            with self.assertRaises(RecipeError, msg=repr((extra, swap))[:50]):
                validate_entries(ball, extra, swap)
            with self.assertRaises(RecipeError):
                roll(ball, 1, extra_entries=extra, replace_entries=swap)
        # replace shrinks the base so extras that fit on top of it are fine; the cap applies per locale
        validate_entries(ball, {'en': ['x'] * 480}, {'en': ['a']})
        with self.assertRaises(RecipeError):
            validate_entries(ball, {'en': ['x'] * 500}, {'en': ['a', 'b']})
        validate_entries(tarot, {'en': ['x']})  # ignored but still shape-checked
        with self.assertRaises(RecipeError):
            validate_entries(tarot, None, {'en': []})

    def test_dice_within_range(self):
        for key, sides in (('dice', 6), ('d20', 20)):
            seen = {int(roll(get_item(key), s).text.split('**')[1]) for s in range(1000)}
            self.assertEqual(seen, set(range(1, sides + 1)))

    def test_sides_override_validated(self):
        from llmcord_core.items import random_message
        dice = get_item('dice')
        for s in range(100):
            n = int(random_message.roll(dice, s, params={'sides': 100}).text.split('**')[1])
            self.assertTrue(1 <= n <= 100)
        with self.assertRaises(RecipeError):
            random_message.roll(dice, 1, params={'sides': 1})
        with self.assertRaises(RecipeError):
            random_message.roll(dice, 1, params={'sides': 5000})

    def test_fortune_cookie_has_lucky_number(self):
        for s in range(100):
            res = roll(get_item('fortune_cookie'), s)
            self.assertTrue(1 <= int(res.footer_note.rsplit(' ', 1)[1]) <= 99)

    def test_tarot_orientations_and_meanings(self):
        tarot = get_item('tarot')
        cards = {c['name']: c for c in tarot.tables['card']['en']}
        self.assertEqual(len(cards), 22)
        seen = set()
        for s in range(400):
            head, meaning = roll(tarot, s).text.split('\n', 1)
            name, orientation = head.strip('*').split('** (')
            orientation = orientation.rstrip(')')
            seen.add(orientation)
            self.assertEqual(meaning, cards[name]['reversed' if orientation == 'Reversed' else 'upright'])
        self.assertEqual(seen, {'Upright', 'Reversed'})

    def test_8ball_extra_and_replace_entries(self):
        ball = get_item('8ball')
        base = set(ball.tables['answer']['en'])
        extra = {'en': ['Ask the cat.']}
        got = {roll(ball, s, extra_entries=extra).text for s in range(2000)}
        self.assertEqual(got, base | {'Ask the cat.'})
        got = {roll(ball, s, replace_entries={'en': ['A', 'B']}).text for s in range(200)}
        self.assertEqual(got, {'A', 'B'})
        got = {roll(ball, s, extra_entries={'en': ['C']}, replace_entries={'en': ['A']}).text for s in range(50)}
        self.assertEqual(got, {'A', 'C'})
        # no change for a roll without entries
        self.assertEqual(roll(ball, 3), roll(ball, 3, extra_entries={}, replace_entries=None))

    def test_extra_entries_ignored_for_tarot_and_dice(self):
        entries = {'en': ['Ask the cat.']}
        for key in ('tarot', 'dice', 'd20'):
            item = get_item(key)
            for s in range(100):
                self.assertEqual(roll(item, s), roll(item, s, extra_entries=entries, replace_entries=entries))

    def test_fortune_cookie_extra_goes_to_fortune_table_only(self):
        cookie = get_item('fortune_cookie')
        got = {roll(cookie, s, extra_entries={'en': ['Zzz.']}).text for s in range(1500)}
        self.assertIn('Zzz.', got)

    def test_bad_entries_rejected(self):
        ball = get_item('8ball')
        for bad in ({'en': ['x' * 301]}, {'en': ['x'] * 501}, {'en': ['']}, {'en': [5]}, {'fr': ['x']}, ['x']):
            with self.assertRaises(RecipeError, msg=repr(bad)[:40]):
                roll(ball, 1, extra_entries=bad)
        with self.assertRaises(RecipeError):
            roll(ball, 1, extra_entries={'en': ['x'] * 481})

    def test_locale_falls_back_to_en(self):
        ball = get_item('8ball')
        self.assertEqual(roll(ball, 5, locale='fr'), roll(ball, 5))
        extra = {'en': ['Ask the cat.']}
        self.assertEqual(roll(ball, 5, locale='fr', extra_entries=extra), roll(ball, 5, extra_entries=extra))
        ko = {'en': ['x'], 'ko': ['y']}
        self.assertEqual(roll(ball, 5, locale='ko', replace_entries=ko).text, 'y')
        self.assertEqual(roll(ball, 5, locale='fr', replace_entries=ko).text, 'x')


if __name__ == '__main__':
    unittest.main()
