"""Server timezone operation behind the Server setup select (FEAT-04).

Seam: ``timezone_operation`` / ``timezone_detail`` over an in-memory ``Store``.
"""
import unittest
import zoneinfo

from llmcord_core.dashboard import timezone_detail, timezone_operation, timezone_options
from llmcord_core.store import Store


class TimezoneOperationTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()

    def test_set_from_empty(self):
        """FEAT-04: setting a timezone on a guild without one reports old '' and stores the name."""
        result = timezone_operation(self.store, 1, 'Asia/Seoul')
        self.assertEqual(result, {'old': '', 'new': 'Asia/Seoul'})
        self.assertEqual(self.store.guild_timezone(1), 'Asia/Seoul')

    def test_change(self):
        """FEAT-04: changing the timezone reports the previous name."""
        self.store.set_guild_timezone(1, 'Asia/Seoul')
        result = timezone_operation(self.store, 1, 'Europe/Paris')
        self.assertEqual(result, {'old': 'Asia/Seoul', 'new': 'Europe/Paris'})
        self.assertEqual(self.store.guild_timezone(1), 'Europe/Paris')

    def test_clear_with_empty_or_none(self):
        """FEAT-04: clearing the select (None or '') removes the server default."""
        for cleared in (None, ''):
            with self.subTest(cleared=cleared):
                self.store.set_guild_timezone(1, 'Asia/Seoul')
                result = timezone_operation(self.store, 1, cleared)
                self.assertEqual(result, {'old': 'Asia/Seoul', 'new': ''})
                self.assertEqual(self.store.guild_timezone(1), '')

    def test_unknown_name_raises_and_leaves_store_unchanged(self):
        """FEAT-04: an unknown timezone raises ValueError and keeps the existing value."""
        self.store.set_guild_timezone(1, 'Asia/Seoul')
        with self.assertRaises(ValueError):
            timezone_operation(self.store, 1, 'Mars/Olympus')
        self.assertEqual(self.store.guild_timezone(1), 'Asia/Seoul')

    def test_guild_isolation(self):
        """FEAT-04: changing one guild's timezone leaves another guild's untouched."""
        self.store.set_guild_timezone(2, 'Europe/Paris')
        timezone_operation(self.store, 1, 'Asia/Seoul')
        self.assertEqual(self.store.guild_timezone(2), 'Europe/Paris')
        timezone_operation(self.store, 1, '')
        self.assertEqual(self.store.guild_timezone(2), 'Europe/Paris')

    def test_detail_shape(self):
        """FEAT-04: the audit detail carries only old and new."""
        result = {'old': 'Asia/Seoul', 'new': '', 'extra': 1}
        self.assertEqual(timezone_detail(result), {'old': 'Asia/Seoul', 'new': ''})


class TimezoneOptionsTests(unittest.TestCase):
    def test_empty_current_is_sorted_available(self):
        """FEAT-04: with no stored value the options are exactly the sorted available zones."""
        self.assertEqual(timezone_options(''), sorted(zoneinfo.available_timezones()))

    def test_listed_value_not_duplicated(self):
        """FEAT-04: a stored zone already available is not added twice."""
        self.assertEqual(timezone_options('Asia/Seoul').count('Asia/Seoul'), 1)

    def test_unlisted_value_included_without_mutating_cache(self):
        """FEAT-04: an unlisted stored value stays selectable and does not leak into later calls."""
        options = timezone_options('Etc/Unlisted-Test')
        self.assertIn('Etc/Unlisted-Test', options)
        self.assertEqual(options, sorted(options))
        self.assertNotIn('Etc/Unlisted-Test', timezone_options(''))


if __name__ == '__main__':
    unittest.main()
