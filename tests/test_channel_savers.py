"""UI-42: the channel card's default cast and ambient saves refuse to overwrite a change made elsewhere.

Seam: ``channel_savers`` over an in-memory ``Store``.
"""
import json
import unittest

from llmcord_core.admin_store import ConflictError
from llmcord_core.dashboard import binding_space_name, channel_savers
from llmcord_core.store import Store


class ChannelSaversTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.world = self.store.create_space(1, 'World', 'world')
        self.store.bind_channel(1, 100, self.world)
        self.alice = self.store.add_character(1, self.world, 'Alice', {'name': 'Alice'}, None, [])
        self.bob = self.store.add_character(1, self.world, 'Bob', {'name': 'Bob'}, None, [])
        self.cast, self.ambient = [], False
        self.save_cast, self.save_ambient = channel_savers(self.store, 1, 100, [], False, lambda: self.cast, lambda: self.ambient)

    def tearDown(self):
        self.store.close()

    def default_cast(self):
        return json.loads(self.store.channel(100)['default_cast'])

    def test_normal_saves_work_and_repeat(self):
        """UI-42: saves succeed and move the baseline, so a second save from the same card is not a conflict."""
        self.cast, self.ambient = [self.alice], True
        self.assertTrue(self.save_cast())
        self.assertTrue(self.save_ambient())
        self.cast, self.ambient = [self.alice, self.bob], False
        self.assertTrue(self.save_cast())
        self.assertTrue(self.save_ambient())
        self.assertEqual(self.default_cast(), [self.alice, self.bob])
        self.assertFalse(self.store.channel(100)['ambient'])

    def test_cast_changed_elsewhere_conflicts_without_overwrite(self):
        """UI-42: a cast changed behind the card raises ConflictError and keeps the other change."""
        self.store.set_cast(100, None, [self.bob], default=True)
        self.cast = [self.alice]
        with self.assertRaisesRegex(ConflictError, 'Default cast changed in Discord. Reload the page and try again.'):
            self.save_cast()
        self.assertEqual(self.default_cast(), [self.bob])

    def test_ambient_changed_elsewhere_conflicts_without_overwrite(self):
        """UI-42: ambient changed behind the card raises ConflictError and keeps the other change."""
        self.store.set_ambient(100, True)
        self.ambient = False
        with self.assertRaisesRegex(ConflictError, 'Ambient participation changed in Discord'):
            self.save_ambient()
        self.assertTrue(self.store.channel(100)['ambient'])

    def test_other_guild_binding_is_a_conflict(self):
        """UI-42: a binding that belongs to another guild is treated as gone, never written."""
        gone = 'This channel is no longer bound in Discord. Reload the page and try again.'
        save_cast, save_ambient = channel_savers(self.store, 2, 100, [], False, lambda: [self.alice], lambda: True)
        for save in (save_cast, save_ambient):
            with self.assertRaisesRegex(ConflictError, gone):
                save()
        self.assertEqual(self.default_cast(), [])
        self.assertFalse(self.store.channel(100)['ambient'])

    def test_missing_channel_row_is_a_conflict(self):
        """UI-42: a binding removed behind the card conflicts instead of raising a bare error."""
        save_cast, save_ambient = channel_savers(self.store, 1, 999, [], False, lambda: [self.alice], lambda: True)
        for save in (save_cast, save_ambient):
            with self.assertRaisesRegex(ConflictError, 'no longer bound'):
                save()

    def test_saved_baseline_still_catches_later_external_change(self):
        """UI-42: after a successful save the baseline is the saved value, so a later outside change conflicts."""
        self.cast, self.ambient = [self.alice], True
        self.save_cast()
        self.save_ambient()
        self.store.set_cast(100, None, [self.bob], default=True)
        self.store.set_ambient(100, False)
        self.cast, self.ambient = [self.alice, self.bob], True
        with self.assertRaises(ConflictError):
            self.save_cast()
        with self.assertRaises(ConflictError):
            self.save_ambient()
        self.assertEqual(self.default_cast(), [self.bob])

    def test_missing_space_has_no_name(self):
        """UI-42: a binding whose space is gone yields '' (the row then shows 'World or hub missing')."""
        self.assertEqual(binding_space_name({5: 'World (world)'}, 5), 'World')
        self.assertEqual(binding_space_name({5: 'World (world)'}, 6), '')
