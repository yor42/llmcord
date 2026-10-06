import os
import unittest
from unittest.mock import patch

from llmcord_core.errors import error_detail


class ErrorDetailTests(unittest.TestCase):
    def test_provider_message_keeps_status_without_prompt_or_credentials(self):
        error = RuntimeError('full response must not appear')
        error.status_code = 400
        error.body = {'error': {'message': 'Unsupported model. key=private-key https://service.test/?key=private-key'},
                      'request': {'prompt': 'private conversation'}}
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'private-key'}):
            detail = error_detail(error)
        self.assertIn('RuntimeError HTTP 400: Unsupported model', detail)
        for private in ('private-key', 'private conversation', 'full response', 'service.test'):
            self.assertNotIn(private, detail)

    def test_discord_error_has_status_code_and_bounded_text(self):
        error = RuntimeError('unused')
        error.status = 403
        error.code = 50013
        error.text = 'Missing Permissions ' + 'x' * 2000
        detail = error_detail(error)
        self.assertIn('HTTP 403 code 50013: Missing Permissions', detail)
        self.assertLessEqual(len(detail), 1000)

    def test_unexpected_error_retains_diagnostic(self):
        self.assertEqual(error_detail(AttributeError("'NoneType' object has no attribute 'id'")),
                         "AttributeError: 'NoneType' object has no attribute 'id'")
