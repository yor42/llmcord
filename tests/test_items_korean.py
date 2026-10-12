"""FEAT-46 Korean items (f5): built-in recipes in Korean, locale-keyed names, and the Korean name at the Discord and dashboard edges."""
import copy
import re
import string
import unittest
from collections.abc import Mapping

import discord

from helpers import FakeInteraction, invoke, make_settings

from llmcord_core import items
from llmcord_core.discord_bot import SkitBot
from llmcord_core.i18n import Translator, discord_locale
from llmcord_core.items import RecipeError, display_name, roll, validate_recipe
from llmcord_core.items_store import ItemError
from llmcord_core.shop_ui import use_rows

KO_NAMES = {'8ball': '매직 8볼', 'd20': '20면체 주사위 (d20)', 'dice': '주사위', 'fortune_cookie': '포춘 쿠키', 'tarot': '타로'}
ALL = {r.key: r for r in items.all_items()}
G, U = 1, 9


def placeholders(text):
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def hangul(text):
    return any('가' <= ch <= '힣' for ch in text)


def template_of(recipe, locale):
    raw = recipe.template[locale]
    return {'text': raw} if isinstance(raw, str) else raw


def template_placeholders(raw):
    found = placeholders(raw['text']) | placeholders(raw.get('footer_note', ''))
    for name, value in raw.get('fields', []):
        found |= placeholders(name) | placeholders(value)
    return found


def dice_data(**kw):
    data = {'key': 'dice', 'kind': 'random_message', 'name': 'Dice', 'emoji': '🎲', 'consumable': False, 'default_price': 50,
            'params': {'sides': 6}, 'tables': {'roll': {'range': [1, 'sides']}}, 'template': {'en': {'text': 'You rolled a **{roll}**.'}}}
    data.update(kw)
    return data


class BuiltInKoreanContentTests(unittest.TestCase):
    def test_five_built_ins_exist(self):
        """FEAT-46 Korean items: the catalog still holds exactly the five built-ins."""
        self.assertEqual(set(ALL), set(KO_NAMES))

    def test_every_built_in_has_the_agreed_korean_name(self):
        """FEAT-46 Korean items: each built-in has the glossary Korean name; the English name is unchanged."""
        for key, recipe in ALL.items():
            with self.subTest(key=key):
                self.assertEqual(recipe.names.get('ko'), KO_NAMES[key])
                self.assertEqual(display_name(recipe, 'ko'), KO_NAMES[key])
                self.assertEqual(recipe.names['en'], recipe.name)

    def test_english_names_stay_byte_identical(self):
        """FEAT-46 Korean items: Recipe.name stays the English string."""
        self.assertEqual({k: r.name for k, r in ALL.items()}, {'8ball': 'Magic 8-ball', 'd20': 'd20', 'dice': 'Dice', 'fortune_cookie': 'Fortune cookie', 'tarot': 'Tarot'})

    def test_every_table_with_en_has_ko_with_same_shape(self):
        """FEAT-46 Korean items: each table with an en list has a ko list with the same row count and record fields."""
        for key, recipe in ALL.items():
            for name, table in recipe.tables.items():
                if not (isinstance(table, dict) and 'en' in table):
                    continue
                with self.subTest(key=key, table=name):
                    self.assertIn('ko', table)
                    en, ko = table['en'], table['ko']
                    self.assertEqual(len(ko), len(en))
                    for en_row, ko_row in zip(en, ko):
                        self.assertEqual(type(ko_row), type(en_row))
                        if isinstance(en_row, dict):
                            self.assertEqual(set(ko_row), set(en_row))
                            self.assertTrue(all(isinstance(v, str) and v.strip() for v in ko_row.values()))
                        else:
                            self.assertTrue(ko_row.strip())

    def test_every_template_has_ko_with_same_parts_and_placeholders(self):
        """FEAT-46 Korean items: each ko template has the same parts and placeholders as en."""
        for key, recipe in ALL.items():
            with self.subTest(key=key):
                self.assertIn('ko', recipe.template)
                en, ko = template_of(recipe, 'en'), template_of(recipe, 'ko')
                self.assertEqual(set(ko), set(en))
                self.assertEqual(len(ko.get('fields', [])), len(en.get('fields', [])))
                self.assertEqual(template_placeholders(ko), template_placeholders(en))
                self.assertEqual(placeholders(ko['text']), placeholders(en['text']))
                self.assertEqual(placeholders(ko.get('footer_note', '')), placeholders(en.get('footer_note', '')))

    def test_ko_content_is_korean_text(self):
        """FEAT-46 Korean items: ko table rows and templates contain Hangul and differ from en."""
        for key, recipe in ALL.items():
            with self.subTest(key=key):
                text = template_of(recipe, 'ko')['text']
                self.assertTrue(hangul(text) or re.fullmatch(r'\{\w+\}', text))
                for table in recipe.tables.values():
                    if isinstance(table, dict) and 'ko' in table:
                        row = table['ko'][0]
                        self.assertTrue(hangul(' '.join(row.values()) if isinstance(row, dict) else row))
                        self.assertNotEqual(table['ko'], table['en'])


class KoreanRollTests(unittest.TestCase):
    def test_ko_roll_is_korean_and_deterministic(self):
        """FEAT-46 Korean items: roll(..., 'ko') is deterministic and differs from the English roll."""
        for key, recipe in ALL.items():
            for seed in (1, 2, 3):
                with self.subTest(key=key, seed=seed):
                    first = roll(recipe, seed, 'ko')
                    self.assertEqual(first, roll(recipe, seed, 'ko'))
                    self.assertNotEqual(first.to_json(), roll(recipe, seed, 'en').to_json())

    def test_regional_tag_matches_base_language(self):
        """FEAT-46 Korean items: 'ko-KR' rolls the same as 'ko'."""
        for key, recipe in ALL.items():
            with self.subTest(key=key):
                self.assertEqual(roll(recipe, 5, 'ko-KR'), roll(recipe, 5, 'ko'))

    def test_unknown_locale_falls_back_to_english(self):
        """FEAT-46 Korean items: an unknown locale rolls as English."""
        for key, recipe in ALL.items():
            with self.subTest(key=key):
                self.assertEqual(roll(recipe, 5, 'xx'), roll(recipe, 5, 'en'))
                self.assertEqual(roll(recipe, 5, 'fr-CA'), roll(recipe, 5, 'en'))

    def test_dice_ko_has_the_number_and_sides(self):
        """FEAT-46 Korean items: the Korean dice sentence still carries the roll and the sides."""
        result = roll(ALL['dice'], 4, 'ko')
        number = roll(ALL['dice'], 4, 'en').text.split('**')[1]
        self.assertIn(f'**{number}**', result.text)
        self.assertIn('1-6', result.text)
        self.assertNotIn('rolled', result.text)

    def test_tarot_ko_uses_korean_card_and_orientation_words(self):
        """FEAT-46 Korean items: a Korean tarot draw names a standard Korean card and a Korean orientation."""
        cards = ('바보', '마법사', '여사제', '여황제', '황제', '교황', '연인', '전차', '힘', '은둔자', '운명의 수레바퀴', '정의', '매달린 사람',
                 '죽음', '절제', '악마', '탑', '별', '달', '태양', '심판', '세계')
        seen = set()
        for seed in range(60):
            text = roll(ALL['tarot'], seed, 'ko').text
            self.assertTrue(any(card in text for card in cards), text)
            self.assertNotIn('Upright', text)
            self.assertNotIn('Reversed', text)
            self.assertTrue('정방향' in text or '역방향' in text, text)
            seen.add('정방향' in text)
        self.assertEqual(seen, {True, False})

    def test_english_tarot_orientation_unchanged(self):
        """FEAT-46 Korean items: English tarot still says Upright or Reversed."""
        texts = [roll(ALL['tarot'], seed, 'en').text for seed in range(30)]
        self.assertTrue(any('(Upright)' in t for t in texts) and any('(Reversed)' in t for t in texts))


class NameShapeTests(unittest.TestCase):
    def test_plain_str_name_still_parses(self):
        """FEAT-46 Korean items: a plain-string name parses; names holds only en; display_name returns it for every locale."""
        recipe = validate_recipe(dice_data())
        self.assertEqual(recipe.name, 'Dice')
        self.assertIsInstance(recipe.names, Mapping)
        self.assertEqual(dict(recipe.names), {'en': 'Dice'})
        for locale in ('en', 'ko', 'ko-KR', 'fr', 'xx', 'pseudo'):
            self.assertEqual(display_name(recipe, locale), 'Dice')

    def test_locale_keyed_name_parses(self):
        """FEAT-46 Korean items: a locale-keyed name parses and Recipe.name is the English one."""
        recipe = validate_recipe(dice_data(name={'en': 'Dice', 'ko': '주사위'}))
        self.assertEqual(recipe.name, 'Dice')
        self.assertEqual(dict(recipe.names), {'en': 'Dice', 'ko': '주사위'})

    def test_names_are_read_only(self):
        """FEAT-46 Korean items: Recipe.names cannot be changed in place."""
        recipe = validate_recipe(dice_data(name={'en': 'Dice', 'ko': '주사위'}))
        with self.assertRaises(TypeError):
            recipe.names['ko'] = 'x'

    def test_display_name_fallbacks(self):
        """FEAT-46 Korean items: display_name normalizes the tag, tries the base language, then en."""
        recipe = validate_recipe(dice_data(name={'en': 'Dice', 'ko': '주사위'}))
        for locale in ('ko', 'ko-KR', 'ko_KR', 'KO'):
            self.assertEqual(display_name(recipe, locale), '주사위', locale)
        for locale in ('en', 'en-US', 'fr', 'xx', '', None):
            self.assertEqual(display_name(recipe, locale), 'Dice', locale)

    def test_name_without_en_is_rejected(self):
        """FEAT-46 Korean items: a locale-keyed name without en is a RecipeError."""
        with self.assertRaises(RecipeError):
            validate_recipe(dice_data(name={'ko': '주사위'}))

    def test_locale_keyed_name_checks_each_text(self):
        """FEAT-46 Korean items: every locale's name must be non-empty text within MAX_NAME."""
        for bad in ({'en': 'Dice', 'ko': ''}, {'en': 'Dice', 'ko': 'x' * 41}, {'en': 'Dice', 'ko': 5}, {'en': ''}):
            with self.subTest(bad=bad), self.assertRaises(RecipeError):
                validate_recipe(dice_data(name=bad))
        with self.assertRaises(RecipeError):
            validate_recipe(dice_data(name={'en': 'Dice', 'not a locale!': 'x'}))


class KoreanEdgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        for key in ('8ball', 'dice'):
            self.store.save_shop_item(G, key, expected_revision=0, for_sale=True)
        self.store.set_currency_name(G, 'gold')
        self.store.change_balance(G, U, 1000, 'start', 1)
        discord_locale.set_server_language_source(None)
        self.addCleanup(discord_locale.set_server_language_source, None)

    def tearDown(self):
        self.store.close()

    async def run_cmd(self, path, *args, locale=None):
        interaction = FakeInteraction(guild_id=G, user_id=U)
        if locale:
            interaction.locale = locale
        await invoke(self.bot, path, interaction, *args)
        return interaction

    async def test_shop_for_ko_invoker_shows_korean_names(self):
        """FEAT-46 Korean items: /shop for a ko invoker lists and labels items by their Korean name."""
        shop = await self.run_cmd('shop', locale=discord.Locale.korean)
        reply = shop.response.sent[0][1]
        names = ' '.join(f.name for f in reply['embed'].fields)
        self.assertIn('주사위', names)
        self.assertIn('매직 8볼', names)
        self.assertNotIn('Dice', names)
        labels = ' '.join(b.label for b in reply['view'].children)
        self.assertIn('주사위', labels)
        self.assertNotIn('Dice', labels)

    async def test_shop_for_en_invoker_is_unchanged(self):
        """FEAT-46 Korean items: /shop for an en invoker still uses the English names."""
        for locale in (None, discord.Locale.american_english):
            shop = await self.run_cmd('shop', locale=locale)
            names = [f.name for f in shop.response.sent[0][1]['embed'].fields]
            self.assertIn('🎲 Dice · 50 gold', names)
            self.assertIn('🎱 Magic 8-ball · 150 gold', names)

    async def test_inventory_for_ko_invoker_shows_korean_names(self):
        """FEAT-46 Korean items: /inventory for a ko invoker names items in Korean."""
        self.store.buy_item(G, U, 'dice')
        inv = await self.run_cmd('inventory', locale=discord.Locale.korean)
        names = ' '.join(f.name for f in inv.response.sent[0][1]['embed'].fields)
        self.assertIn('주사위', names)
        self.assertNotIn('Dice', names)

    async def test_use_in_a_korean_server_posts_the_korean_name_and_roll(self):
        """FEAT-46 Korean items: /use by a ko invoker in a ko-language server posts the Korean name and a Korean roll."""
        discord_locale.set_server_language_source(lambda guild_id: 'ko')
        self.store.buy_item(G, U, 'dice')
        use = await self.run_cmd('use', 'dice', locale=discord.Locale.korean)
        embed = use.response.sent[0][1]['embed']
        self.assertEqual(embed.title, '🎲 주사위')
        self.assertNotIn('rolled', embed.description)

    async def test_use_for_en_invoker_is_unchanged(self):
        """FEAT-46 Korean items: /use by an en invoker in an en server posts the English name and roll."""
        self.store.buy_item(G, U, 'dice')
        use = await self.run_cmd('use', 'dice')
        embed = use.response.sent[0][1]['embed']
        self.assertEqual(embed.title, '🎲 Dice')
        self.assertTrue(embed.description.startswith('You rolled a **'))

    async def test_use_error_for_ko_invoker_carries_the_korean_name(self):
        """FEAT-46 Korean items: a private /use refusal (cooldown) for a ko invoker names the item in Korean."""
        self.store.buy_item(G, U, 'dice')
        await self.run_cmd('use', 'dice', locale=discord.Locale.korean)
        wait = await self.run_cmd('use', 'dice', locale=discord.Locale.korean)
        self.assertIn('주사위', wait.replies[0])
        self.assertNotIn('Dice', wait.replies[0])
        self.assertTrue(wait.response.sent[0][1]['ephemeral'])

    async def test_use_error_for_en_invoker_is_unchanged(self):
        """FEAT-46 Korean items: the same refusal for an en invoker is the old English text."""
        self.store.buy_item(G, U, 'dice')
        await self.run_cmd('use', 'dice')
        wait = await self.run_cmd('use', 'dice')
        self.assertRegex(wait.replies[0], r'^Dice is recharging\. Try again in \d+ s\.$')

    async def test_store_errors_stay_english_and_carry_the_item_key(self):
        """FEAT-46 Korean items: store errors keep the English name and add the recipe key as the item param."""
        self.store.buy_item(G, U, 'dice')
        with self.assertRaises(ItemError) as caught:
            self.store.buy_item(G, U, 'dice')
        self.assertEqual(caught.exception.params['name'], 'Dice')
        self.assertEqual(caught.exception.params['item'], 'dice')
        self.assertEqual(str(caught.exception), 'You already own Dice.')
        self.store.use_item(G, U, 100, 'dice', locale='ko')
        with self.assertRaises(ItemError) as cooling:
            self.store.use_item(G, U, 100, 'dice', locale='ko')
        self.assertEqual((cooling.exception.params['name'], cooling.exception.params['item']), ('Dice', 'dice'))

    async def test_use_log_result_follows_the_audience_locale_but_stores_english_name(self):
        """FEAT-46 Korean items: the returned item dict keeps the English name even when the roll is Korean."""
        self.store.buy_item(G, U, 'dice')
        outcome = self.store.use_item(G, U, 100, 'dice', locale='ko')
        self.assertEqual(outcome['item']['name'], 'Dice')
        self.assertNotIn('rolled', outcome['result'].text)


class DashboardKoreanTests(unittest.TestCase):
    USES = [{'id': 3, 'user_id': 5, 'item_key': 'dice', 'used_at': 0, 'result': {'text': 'x'}}]

    def test_use_rows_show_the_korean_name_with_a_ko_translator(self):
        """FEAT-46 Korean items: the dashboard use-log row names the item in Korean for a ko viewer."""
        rows = use_rows(copy.deepcopy(self.USES), {5: 'Aria'}, 'UTC', t=Translator('ko'))
        self.assertEqual(rows[0]['item'], '🎲 주사위')

    def test_use_rows_stay_english_without_a_translator_or_for_en(self):
        """FEAT-46 Korean items: the use-log row is English by default and for an en viewer."""
        self.assertEqual(use_rows(self.USES, {5: 'Aria'}, 'UTC')[0]['item'], '🎲 Dice')
        self.assertEqual(use_rows(self.USES, {5: 'Aria'}, 'UTC', t=Translator('en'))[0]['item'], '🎲 Dice')
