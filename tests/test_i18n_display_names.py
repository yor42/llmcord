"""MNT-81 a/b (D39): built-in prompt block names and default emotion labels are translated at display time; stored values never change."""
import os
import unittest
from unittest.mock import patch

from llmcord_core import i18n
from llmcord_core.avatars import DEFAULT_SLOTS, label_to_store, slot_display_label
from llmcord_core.presets_ui import block_display_name
from llmcord_core.prompts import block, default_bundle


class BlockDisplayNameTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {'LLMCORD_I18N_PSEUDO': '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        i18n.clear_cache()
        self.addCleanup(i18n.clear_cache)
        self.en = i18n.Translator('en')
        self.pseudo = i18n.Translator('pseudo')

    def test_every_built_in_default_is_identical_in_english(self):
        """MNT-81 a: in English every built-in block shows exactly the stored default name."""
        bundle = default_bundle()['purposes']
        count = 0
        for blocks in bundle.values():
            for b in blocks:
                count += 1
                self.assertEqual(block_display_name(b, self.en), b['name'])
        self.assertGreater(count, 20)

    def test_built_in_default_is_translated_in_pseudo(self):
        """MNT-81 a: a built-in block still carrying its default name shows the translated name."""
        for ident, name in (('main', 'Main'), ('lore_before_char', 'Lore Before Char'), ('character_voice', 'Character Voice'), ('history', 'History')):
            b = block(ident)
            self.assertEqual(b['name'], name)
            self.assertEqual(block_display_name(b, self.pseudo), i18n.pseudo(name))

    def test_display_does_not_mutate_the_block(self):
        """MNT-81 a: display translation never changes the stored block."""
        b = block('main')
        before = dict(b)
        block_display_name(b, self.pseudo)
        self.assertEqual(b, before)

    def test_renamed_built_in_is_unchanged(self):
        """MNT-81 a: a built-in block the user renamed keeps the typed name."""
        b = block('main')
        b['name'] = 'My opening rules'
        self.assertEqual(block_display_name(b, self.pseudo), 'My opening rules')

    def test_renamed_to_another_default_is_unchanged(self):
        """MNT-81 a: the default name of a different id does not count as the default for this id."""
        b = block('main')
        b['name'] = 'History'
        self.assertEqual(block_display_name(b, self.pseudo), 'History')

    def test_custom_default_is_translated_with_code(self):
        """MNT-81 a: a custom block with its default name shows the translated 'Custom-{code}' with the code kept."""
        b = block('custom-ab12cd34')
        self.assertEqual(b['name'], 'Custom-Ab12Cd34')
        shown = block_display_name(b, self.pseudo)
        self.assertEqual(shown, self.pseudo('Custom-{code}', code='ab12cd34'))
        self.assertIn('ab12cd34', shown)
        self.assertNotEqual(shown, b['name'])

    def test_custom_default_english_is_identical(self):
        """MNT-81 a: in English a custom block's default name is unchanged."""
        b = block('custom-ab12cd34')
        self.assertEqual(block_display_name(b, self.en), b['name'])

    def test_custom_renamed_or_imported_is_unchanged(self):
        """MNT-81 a: custom and imported blocks with their own names keep them."""
        mine = {**block('custom-ab12cd34'), 'name': 'Narrator notes'}
        imported = {**block('imported-7'), 'name': 'Main Prompt'}
        self.assertEqual(block_display_name(mine, self.pseudo), 'Narrator notes')
        self.assertEqual(block_display_name(imported, self.pseudo), 'Main Prompt')

    def test_empty_name_is_untitled(self):
        """MNT-81 a: an empty name keeps showing the translated 'Untitled block'."""
        b = {**block('main'), 'name': ''}
        self.assertEqual(block_display_name(b, self.en), 'Untitled block')
        self.assertEqual(block_display_name(b, self.pseudo), self.pseudo('Untitled block'))


class SlotDisplayLabelTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {'LLMCORD_I18N_PSEUDO': '1'})
        patcher.start()
        self.addCleanup(patcher.stop)
        i18n.clear_cache()
        self.addCleanup(i18n.clear_cache)
        self.en = i18n.Translator('en')
        self.pseudo = i18n.Translator('pseudo')

    def test_default_labels_are_identical_in_english(self):
        """MNT-81 b: in English every default slot label is shown exactly as stored."""
        for key in DEFAULT_SLOTS:
            self.assertEqual(slot_display_label(key, key.title(), self.en), key.title())

    def test_default_labels_are_translated_in_pseudo(self):
        """MNT-81 b: a default slot still carrying its default label shows the translated word."""
        for key in DEFAULT_SLOTS:
            self.assertEqual(slot_display_label(key, key.title(), self.pseudo), i18n.pseudo(key.title()))

    def test_renamed_default_slot_is_unchanged(self):
        """MNT-81 b: a default slot whose label was edited keeps the typed label."""
        self.assertEqual(slot_display_label('happy', 'Joyful', self.pseudo), 'Joyful')

    def test_default_label_on_another_slot_is_unchanged(self):
        """MNT-81 b: a default word typed as the label of a different slot is not translated."""
        self.assertEqual(slot_display_label('happy', 'Sad', self.pseudo), 'Sad')

    def test_custom_slot_is_unchanged_even_if_title_cased(self):
        """MNT-81 b: a user-added emotion keeps its label, even when it equals key.title()."""
        self.assertEqual(slot_display_label('smug', 'Smug', self.pseudo), 'Smug')
        self.assertEqual(slot_display_label('neutral_2', 'Neutral', self.pseudo), 'Neutral')


class LabelToStoreTests(unittest.TestCase):
    def test_untouched_translated_default_keeps_the_stored_label(self):
        self.assertEqual(label_to_store('Happy', i18n.pseudo('Happy'), i18n.pseudo('Happy')), 'Happy')
        self.assertEqual(label_to_store('Happy', 'Happy', 'Happy'), 'Happy')

    def test_edited_text_is_stored(self):
        self.assertEqual(label_to_store('Happy', i18n.pseudo('Happy'), 'Joyful'), 'Joyful')

    def test_blank_is_stored_as_empty(self):
        self.assertEqual(label_to_store('Happy', i18n.pseudo('Happy'), ''), '')
        self.assertEqual(label_to_store('Happy', 'Happy', None), '')


if __name__ == '__main__':
    unittest.main()
