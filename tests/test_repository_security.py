import tempfile
import unittest
from pathlib import Path

from scripts.check_repository import is_sqlite, private_path


class RepositorySecurityTests(unittest.TestCase):
    def test_private_paths_blocked_even_if_gitignore_is_changed(self):
        for path in ('nested/.env', '.env.production', 'CONFIG.YAML', 'data/card.json',
                     'backup.db-wal', 'nested/session.sqlite3.backup', '.ssh/id_ed25519',
                     'keys/client.pem', 'credentials.json', '.test-artifacts/server.log'):
            with self.subTest(path=path):
                self.assertTrue(private_path(path))

    def test_templates_fixtures_and_directory_placeholders_remain_allowed(self):
        for path in ('.env.example', '.env.production.example', 'config-example.yaml',
                     'config-gemini.yaml', 'tests/fixtures/risu-lorebook.json',
                     'charcard/.gitignore', 'lorebooks/.gitignore'):
            with self.subTest(path=path):
                self.assertFalse(private_path(path))

    def test_sqlite_header_detected_without_database_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'renamed.bin'
            path.write_bytes(b'SQLite format 3\x00' + b'fixture')
            self.assertTrue(is_sqlite(path))
            path.write_text('A harmless synthetic fixture')
            self.assertFalse(is_sqlite(path))
