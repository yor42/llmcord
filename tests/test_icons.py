"""Tests for vendored Lucide icons (UI-30)."""
import ast
import re
import unittest
from pathlib import Path

from llmcord_core import icons


class IconVendoringTests(unittest.TestCase):
    """Verify that required icons are vendored and properly configured."""

    # Icons expected to be used by the dashboard
    REQUIRED_ICONS = {'trash-2', 'plus', 'upload', 'pencil', 'arrow-left', 'arrow-right', 'grip-vertical'}

    def test_required_icons_are_available(self):
        """All dashboard icons are in icons.available()."""
        available = set(icons.available())
        self.assertTrue(self.REQUIRED_ICONS.issubset(available),
                       f'Missing icons: {self.REQUIRED_ICONS - available}')

    def test_required_icon_files_exist(self):
        """Each required icon has a corresponding .svg file."""
        for name in self.REQUIRED_ICONS:
            path = icons.ICON_DIR / f'{name}.svg'
            self.assertTrue(path.exists(), f'Missing file: {path}')
            self.assertTrue(path.is_file(), f'Not a file: {path}')

    def test_license_file_exists_with_isc_text(self):
        """ISC LICENSE file exists in lucide directory and contains 'ISC'."""
        license_path = icons.ICON_DIR / 'LICENSE'
        self.assertTrue(license_path.exists(), f'Missing LICENSE: {license_path}')
        self.assertTrue(license_path.is_file(), f'LICENSE is not a file: {license_path}')
        content = license_path.read_text(encoding='utf-8')
        self.assertIn('ISC', content, 'LICENSE file does not contain "ISC"')

    def test_icon_css_contains_all_required_rules(self):
        """icon_css() contains .ll-i-NAME rules with data: URLs for each icon."""
        css = icons.icon_css()
        for name in self.REQUIRED_ICONS:
            class_rule = f'.ll-i-{name}'
            self.assertIn(class_rule, css, f'Missing CSS rule: {class_rule}')
        # Verify all rules use data: URLs as the mask source (not http:// URLs)
        self.assertNotIn('url("http://', css, 'icon_css() contains http:// URLs (should use data:)')

    def test_icon_css_uses_data_url_scheme(self):
        """All icon CSS rules use data:image/svg+xml, URLs."""
        css = icons.icon_css()
        # Find all rules for icons (not the generic .ll-icon rules)
        icon_rules = re.findall(r'\.ll-i-[a-z0-9-]+ \{ --ll-mask: url\("([^"]+)"\)', css)
        self.assertTrue(icon_rules, 'No icon-specific CSS rules found')
        for url in icon_rules:
            self.assertTrue(url.startswith('data:image/svg+xml,'),
                          f'Icon CSS URL does not use data:image/svg+xml scheme: {url}')

    def test_data_url_rejects_invalid_names(self):
        """icons._data_url raises KeyError for invalid icon names."""
        with self.assertRaises(KeyError) as error:
            icons._data_url('not-an-icon')
        self.assertIn('not-an-icon', str(error.exception))

    def test_data_url_rejects_empty_name(self):
        """icons._data_url raises KeyError for empty string."""
        with self.assertRaises(KeyError):
            icons._data_url('')

    def test_data_url_rejects_path_traversal_attempt(self):
        """icons._data_url raises KeyError for path traversal attempts."""
        with self.assertRaises(KeyError):
            icons._data_url('../LICENSE')

    def test_all_lucide_calls_in_dashboard_use_vendored_icons(self):
        """Every icon name passed to lucide() or lucide_button() in dashboard.py is vendored."""
        icon_names = self._extract_icon_names_from_file(
            Path(__file__).parent.parent / 'llmcord_core' / 'dashboard.py'
        )
        available = set(icons.available())
        self.assertTrue(icon_names, 'No lucide() or lucide_button() calls found in dashboard.py')
        missing = icon_names - available
        self.assertFalse(missing, f'Unvendored icons used in dashboard.py: {missing}')

    def test_all_lucide_calls_in_lore_workspace_use_vendored_icons(self):
        """Every icon name passed to lucide() or lucide_button() in lore_workspace.py is vendored."""
        icon_names = self._extract_icon_names_from_file(
            Path(__file__).parent.parent / 'llmcord_core' / 'lore_workspace.py'
        )
        available = set(icons.available())
        self.assertTrue(icon_names, 'No lucide() or lucide_button() calls found in lore_workspace.py')
        missing = icon_names - available
        self.assertFalse(missing, f'Unvendored icons used in lore_workspace.py: {missing}')

    @staticmethod
    def _extract_icon_names_from_file(file_path):
        """Collect string icon names passed to lucide() and lucide_button() calls."""
        positions = {'lucide': 0, 'lucide_button': 1}

        def strings(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                yield node.value
            elif isinstance(node, ast.IfExp):
                yield from strings(node.body)
                yield from strings(node.orelse)

        names = set()
        for node in ast.walk(ast.parse(file_path.read_text(encoding='utf-8'))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            fname = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', None)
            if fname not in positions:
                continue
            index = positions[fname]
            arg = node.args[index] if len(node.args) > index else None
            for kw in node.keywords:
                if kw.arg == 'name':
                    arg = kw.value
            if arg is not None:
                names.update(strings(arg))
        return names


if __name__ == '__main__':
    unittest.main()
