"""FEAT-07: pure helpers behind the Monitoring tab's turn log viewer."""
import unittest

from llmcord_core.dashboard import normalize_reference, turn_log_row


class TurnLogViewTests(unittest.TestCase):
    def test_normalize_reference(self):
        """FEAT-07: the 6-hex reference token is extracted from pasted text (e.g. '(ref 06F1EE)') and lowercased; otherwise the trimmed text is lowercased."""
        for raw, expected in ((' 06f1ee ', '06f1ee'), ('06f1ee', '06f1ee'), ('(ref 06F1EE)', '06f1ee'), ('[ref 06f1ee]', '06f1ee'), ('ref: 06f1ee.', '06f1ee'), ('#06f1ee', '06f1ee'), (' [Nope] ', 'nope'), (None, ''), ('', '')):
            self.assertEqual(normalize_reference(raw), expected, raw)

    def test_turn_log_row_formats_time_tokens_and_status(self):
        """FEAT-07: the row shows the server-timezone time, tokens or a dash, and marks errors."""
        entry = {'created_at': 1767225600, 'channel_id': 5, 'stage': 'reply · dialogue', 'model': 'm', 'status': 'ok', 'reference_id': '', 'input_tokens': 1234, 'output_tokens': None, 'preview': ' hi '}
        row = turn_log_row(entry, 'Asia/Seoul', lambda cid: f'#c{cid}')
        self.assertEqual((row['time'], row['channel'], row['tokens'], row['status'], row['error'], row['preview']), ('2026-01-01 09:00', '#c5', '1,234 in / — out', 'OK', False, 'hi'))
        failed = {**entry, 'status': 'error', 'reference_id': 'ABC', 'input_tokens': None, 'stage': ''}
        row = turn_log_row(failed, 'Not/AZone', str)
        self.assertEqual((row['time'], row['tokens'], row['status'], row['error'], row['reference'], row['stage']), ('2026-01-01 00:00', '—', 'Error', True, 'ABC', '—'))
