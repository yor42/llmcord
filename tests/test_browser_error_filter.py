"""MNT-37: the browser suite tolerates the dropped-NiceGUI-asset cssRules error only when a /_nicegui/ request failed."""
import unittest

from test_dashboard_browser import unexpected_page_errors

CSS_RULES = "SecurityError: Failed to read the 'cssRules' property from 'CSSStyleSheet'"
ASSET = 'https://localhost:1/_nicegui/3.0/static/quasar.prod.css'


class UnexpectedPageErrorsTests(unittest.TestCase):
    def test_cssrules_error_tolerated_when_asset_failed(self):
        self.assertEqual(unexpected_page_errors([CSS_RULES], [ASSET]), [])

    def test_bare_message_without_stack_is_tolerated_too(self):
        self.assertEqual(unexpected_page_errors(["Failed to read the 'cssRules' property"], [ASSET]), [])

    def test_cssrules_error_kept_without_failed_asset(self):
        self.assertEqual(unexpected_page_errors([CSS_RULES], []), [CSS_RULES])

    def test_other_errors_always_kept(self):
        other = 'TypeError: x is undefined'
        self.assertEqual(unexpected_page_errors([CSS_RULES, other], [ASSET]), [other])
        self.assertEqual(unexpected_page_errors([other], [ASSET]), [other])

    def test_no_errors(self):
        self.assertEqual(unexpected_page_errors([], [ASSET]), [])


if __name__ == '__main__':
    unittest.main()
