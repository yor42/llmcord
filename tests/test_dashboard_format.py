"""Dashboard cost formatting via ``format_usd`` (UI-16).

Dashboard-only cost format; Discord footer keeps its own formatting.
Seam: ``format_usd`` function formatting USD values with context-sensitive precision.
Behavior: 0 -> '$0.00'; 0.001 -> '$0.001'; 0.0004 -> '$0.0004'; 0.00004 -> '<$0.0001';
          0.01 -> '$0.01'; 0.0125 -> '$0.01'; 0.1 -> '$0.10'; 1234.5 -> '$1,234.50'.
"""
import unittest

from llmcord_core.dashboard import format_usd, usage_role_labels


class FormatUsdTests(unittest.TestCase):
    """UI-16: Test USD formatting with context-sensitive precision (two decimals from a cent up, up to four below)."""

    def test_format_usd_table(self):
        """UI-16: format_usd matches expected outputs for a range of inputs."""
        test_cases = [
            (0, '$0.00'),
            (None, '$0.00'),
            (-1, '$0.00'),
            (float('nan'), '$0.00'),
            (float('inf'), '$0.00'),
            ('0.001', '$0.001'),
            (0.0001, '$0.0001'),
            (0.0004, '$0.0004'),
            (0.00004, '<$0.0001'),
            (0.009, '$0.009'),
            (0.009999, '$0.01'),
            (0.01, '$0.01'),
            (0.1, '$0.10'),
            (1234.5, '$1,234.50'),
        ]
        for value, expected in test_cases:
            with self.subTest(input=value):
                self.assertEqual(format_usd(value), expected)


class UsageRoleLabelsTests(unittest.TestCase):
    def test_roles_map_to_effective_profiles_with_dialogue_fallback(self):
        """MNT-32: the Monitoring 'Used for' badges follow the effective roles; a missing role uses dialogue's profile."""
        self.assertEqual(usage_role_labels({'dialogue': 'a', 'director': 'b', 'memory': 'a'}), {'a': ['dialogue', 'memory'], 'b': ['director']})
        self.assertEqual(usage_role_labels({'dialogue': 'a'}), {'a': ['dialogue', 'director', 'memory']})
        self.assertEqual(usage_role_labels({'dialogue': 'a', 'memory': 'dash'}), {'a': ['dialogue', 'director'], 'dash': ['memory']})
        self.assertEqual(usage_role_labels({}), {})


if __name__ == '__main__':
    unittest.main()
