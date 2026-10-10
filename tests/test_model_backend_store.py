"""D24 step 2 (FEAT-09): schema v9 model profile/role storage, store helpers and the pure config merge."""
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from llmcord_core.admin_store import ConflictError
from llmcord_core.config import PROFILE_KEYS, ModelProfile, profile_from_mapping
from llmcord_core.store import Store
from tests.helpers import make_settings

OPENAI = {'provider': 'openai', 'model': 'gpt-x', 'context_tokens': 8000, 'api_key_env': 'OPENAI_API_KEY'}
LOCAL = {'provider': 'compatible', 'model': 'local', 'context_tokens': 4000, 'base_url': 'http://localhost:1234/v1'}


def backend_module():
    from llmcord_core import backend
    return backend


class MigrationTests(unittest.TestCase):
    def _make_v8(self, path):
        store = Store(path)
        store.create_space(1, 'World', 'world')
        store.close()
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute('DROP TABLE IF EXISTS model_profiles')
            connection.execute('DROP TABLE IF EXISTS model_roles')
            connection.execute('PRAGMA user_version=8')

    def test_v8_file_upgrades_with_backup_and_keeps_data(self):
        """FEAT-09 (D24): a v8 file gets a pre-v9 backup, user_version 11, the new tables, and keeps its rows."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.sqlite3'
            self._make_v8(path)
            store = Store(path)
            try:
                self.assertEqual(store.one('PRAGMA user_version')[0], 11)
                self.assertEqual(store.one('SELECT COUNT(*) FROM spaces')[0], 1)
                self.assertEqual(store.model_profile_rows(), [])
                roles = store.model_roles()
                self.assertEqual((roles['dialogue'], roles['director'], roles['memory']), (None, None, None))
            finally:
                store.close()
            backups = list(Path(folder).glob('*.pre-v9-*.sqlite3'))
            self.assertEqual(len(backups), 1)
            with closing(sqlite3.connect(backups[0])) as backup:
                self.assertEqual(backup.execute('PRAGMA user_version').fetchone()[0], 8)
                self.assertIsNone(backup.execute("SELECT 1 FROM sqlite_master WHERE name='model_profiles'").fetchone())

    def test_v9_file_opens_without_new_backup(self):
        """FEAT-09 (D24): reopening a v9 file makes no further backup."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.sqlite3'
            self._make_v8(path)
            Store(path).close()
            self.assertEqual(len(list(Path(folder).glob('*.pre-*'))), 1)
            store = Store(path)
            try:
                self.assertEqual(store.one('PRAGMA user_version')[0], 11)
            finally:
                store.close()
            self.assertEqual(len(list(Path(folder).glob('*.pre-*'))), 1)

    def test_v12_file_is_refused(self):
        """FEAT-09 (D24): databases newer than v11 are rejected."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'new.sqlite3'
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute('PRAGMA user_version=12')
            with self.assertRaisesRegex(ValueError, 'newer than this application'):
                Store(path)

    def test_fresh_store_has_single_role_row(self):
        """FEAT-09 (D24): a fresh store has exactly one model_roles row (id 1), version 0."""
        store = Store(':memory:')
        self.assertEqual(store.one('SELECT COUNT(*) FROM model_roles')[0], 1)
        roles = store.model_roles()
        self.assertEqual((roles['revision'], roles['version']), (0, 0))
        with self.assertRaises(sqlite3.IntegrityError):
            store.db.execute('INSERT INTO model_roles(id) VALUES(2)')


class ProfileStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(':memory:')
        patcher = patch.dict('os.environ', {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        import os
        os.environ.pop('LLMCORD_KEY_HOSTS', None)

    def test_create_update_and_list_sorted(self):
        """FEAT-09: create returns a revision, update needs the current revision, rows list sorted by name."""
        first = self.store.save_model_profile('zeta', OPENAI, None, now=10.0)
        self.store.save_model_profile('alpha', LOCAL, None, now=11.0)
        second = self.store.save_model_profile('zeta', {**OPENAI, 'model': 'gpt-y'}, first, now=12.0)
        self.assertGreater(second, first)
        rows = self.store.model_profile_rows()
        self.assertEqual([r['name'] for r in rows], ['alpha', 'zeta'])
        zeta = rows[1]
        self.assertEqual(zeta['data']['model'], 'gpt-y')
        self.assertEqual(zeta['revision'], second)
        self.assertEqual(zeta['updated_at'], 12.0)

    def test_create_existing_name_conflicts(self):
        """FEAT-09: expected_revision=None on an existing name raises ConflictError."""
        self.store.save_model_profile('a', LOCAL, None)
        with self.assertRaises(ConflictError):
            self.store.save_model_profile('a', LOCAL, None)

    def test_stale_or_missing_revision_conflicts(self):
        """FEAT-09: a stale revision, or an update of a missing row, raises ConflictError and changes nothing."""
        revision = self.store.save_model_profile('a', LOCAL, None)
        self.store.save_model_profile('a', {**LOCAL, 'model': 'b'}, revision)
        with self.assertRaises(ConflictError):
            self.store.save_model_profile('a', {**LOCAL, 'model': 'c'}, revision)
        with self.assertRaises(ConflictError):
            self.store.save_model_profile('nope', LOCAL, 1)
        self.assertEqual(self.store.model_profile_rows()[0]['data']['model'], 'b')

    def test_stored_data_is_normalized(self):
        """FEAT-09: None values are dropped; only known keys are stored; stored JSON holds the key NAME."""
        self.store.save_model_profile('a', {**OPENAI, 'base_url': None, 'reasoning_effort': None}, None)
        raw = json.loads(self.store.one('SELECT data_json FROM model_profiles WHERE name=?', ('a',))[0])
        self.assertLessEqual(set(raw), set(PROFILE_KEYS))
        self.assertNotIn(None, raw.values())
        for key, value in OPENAI.items():
            self.assertEqual(raw[key], value)
        for key in ('supports_images', 'structured_outputs', 'stream_usage'):
            if key in raw:
                self.assertIs(type(raw[key]), bool)
        self.assertEqual(raw['api_key_env'], 'OPENAI_API_KEY')

    def test_non_bool_flags_refused_without_echo(self):
        """FEAT-09 (D24): supports_images, structured_outputs and stream_usage must be real bools; the value is not echoed."""
        secret = 'sk-live-abcdef1234567890'
        for key in ('supports_images', 'structured_outputs', 'stream_usage'):
            with self.assertRaises(ValueError) as caught:
                self.store.save_model_profile('a', {**LOCAL, key: secret}, None)
            self.assertNotIn(secret, str(caught.exception))
        self.assertEqual(self.store.model_profile_rows(), [])
        self.assertEqual(self.store.model_backend_version(), 0)

    def test_revision_never_reused_after_delete_and_recreate(self):
        """FEAT-09 (D24): a stale editor cannot overwrite a recreated profile; revisions differ from every earlier one."""
        seen = [self.store.save_model_profile('a', LOCAL, None)]
        self.store.delete_model_profile('a', seen[0], set())
        seen.append(self.store.save_model_profile('a', LOCAL, None))
        self.assertNotIn(seen[1], seen[:1])
        with self.assertRaises(ConflictError):
            self.store.save_model_profile('a', {**LOCAL, 'model': 'stale'}, seen[0])
        self.assertEqual(self.store.model_profile_rows()[0]['data']['model'], 'local')
        updated = self.store.save_model_profile('a', {**LOCAL, 'model': 'new'}, seen[1])
        self.assertNotIn(updated, seen)

    def test_delete_and_role_save_validate_names_without_echo(self):
        """FEAT-09: invalid names raise ValueError (not ConflictError) and are not echoed."""
        bad = 'Bad Name sk-live-abcdef'
        with self.assertRaises(ValueError) as caught:
            self.store.delete_model_profile(bad, 1, set())
        self.assertNotIsInstance(caught.exception, ConflictError)
        self.assertNotIn(bad, str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self.store.save_model_roles({'dialogue': bad, 'director': None, 'memory': None}, 0, {bad})
        self.assertNotIsInstance(caught.exception, ConflictError)
        self.assertNotIn(bad, str(caught.exception))
        self.assertEqual(self.store.model_backend_version(), 0)

    def test_invalid_name_refused_before_anything_stored(self):
        """FEAT-09: a bad name is refused and not echoed into a row."""
        for name in ('Bad Name', '', '../x', 'A'):
            with self.assertRaises(ValueError):
                self.store.save_model_profile(name, LOCAL, None)
        self.assertEqual(self.store.model_profile_rows(), [])
        self.assertEqual(self.store.model_backend_version(), 0)

    def test_invalid_data_refused(self):
        """FEAT-09: dashboard validation applies (unknown key, bad context_tokens)."""
        with self.assertRaises(ValueError):
            self.store.save_model_profile('a', {**LOCAL, 'surprise': 1}, None)
        with self.assertRaises(ValueError):
            self.store.save_model_profile('a', {**LOCAL, 'context_tokens': 0}, None)
        self.assertEqual(self.store.model_profile_rows(), [])

    def test_literal_key_cannot_be_stored(self):
        """FEAT-09 (D24): api_key_env holding a key value is refused, and the value is not echoed or stored."""
        secret = 'sk-live-abcdef1234567890'
        with self.assertRaises(ValueError) as caught:
            self.store.save_model_profile('a', {**OPENAI, 'api_key_env': secret}, None)
        self.assertNotIn(secret, str(caught.exception))
        self.assertEqual(self.store.model_profile_rows(), [])
        self.assertEqual(self.store.all('SELECT data_json FROM model_profiles'), [])
        self.assertFalse(any(secret in r[0] for r in self.store.all('SELECT data_json FROM model_profiles')))

    def test_pinned_key_refuses_foreign_host(self):
        """FEAT-09 (D24): OPENAI_API_KEY with an evil base_url is refused at save; nothing is written."""
        evil = {'provider': 'compatible', 'model': 'm', 'context_tokens': 100,
                'api_key_env': 'OPENAI_API_KEY', 'base_url': 'https://evil.example'}
        with self.assertRaises(ValueError) as caught:
            self.store.save_model_profile('a', evil, None)
        self.assertIn('pinned', str(caught.exception))
        self.assertEqual(self.store.model_profile_rows(), [])
        self.assertEqual(self.store.model_backend_version(), 0)

    def test_pinned_key_allows_its_host(self):
        """FEAT-09 (D24): the same key with the pinned host saves."""
        ok = {**OPENAI, 'provider': 'compatible', 'base_url': 'https://api.openai.com/v1'}
        self.store.save_model_profile('a', ok, None)
        self.assertEqual(len(self.store.model_profile_rows()), 1)

    def test_env_pin_extension_is_honoured(self):
        """FEAT-09 (D24): LLMCORD_KEY_HOSTS in the environment allows an extra host at save time."""
        data = {'provider': 'compatible', 'model': 'm', 'context_tokens': 100,
                'api_key_env': 'OPENAI_API_KEY', 'base_url': 'https://proxy.example/v1'}
        with self.assertRaises(ValueError):
            self.store.save_model_profile('a', data, None)
        with patch.dict('os.environ', {'LLMCORD_KEY_HOSTS': 'OPENAI_API_KEY=proxy.example'}):
            self.store.save_model_profile('a', data, None)
        self.assertEqual(len(self.store.model_profile_rows()), 1)

    def test_delete_profile(self):
        """FEAT-09: delete removes the row; stale or missing revision conflicts."""
        revision = self.store.save_model_profile('a', LOCAL, None)
        with self.assertRaises(ConflictError):
            self.store.delete_model_profile('a', revision + 5, set())
        with self.assertRaises(ConflictError):
            self.store.delete_model_profile('missing', 1, set())
        self.store.delete_model_profile('a', revision, set())
        self.assertEqual(self.store.model_profile_rows(), [])

    def test_delete_refused_while_role_uses_it(self):
        """FEAT-09: deleting a profile assigned to a role raises ValueError naming profile and role."""
        revision = self.store.save_model_profile('a', LOCAL, None)
        self.store.save_model_roles({'dialogue': None, 'director': 'a', 'memory': None}, 0, set())
        with self.assertRaises(ValueError) as caught:
            self.store.delete_model_profile('a', revision, set())
        self.assertNotIsInstance(caught.exception, ConflictError)
        self.assertEqual(str(caught.exception), 'a is assigned to the director role; assign another profile first.')
        self.assertEqual(len(self.store.model_profile_rows()), 1)

    def test_delete_allowed_when_config_has_same_name(self):
        """FEAT-09: a role on a profile also defined in config.yaml may lose its database override."""
        revision = self.store.save_model_profile('a', LOCAL, None)
        self.store.save_model_roles({'dialogue': 'a', 'director': None, 'memory': None}, 0, {'a'})
        self.store.delete_model_profile('a', revision, {'a'})
        self.assertEqual(self.store.model_profile_rows(), [])
        self.assertEqual(self.store.model_roles()['dialogue'], 'a')


class RoleStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(':memory:')

    def test_save_roles_and_revision(self):
        """FEAT-09: saving roles stores them and bumps the revision; a stale revision conflicts."""
        self.store.save_model_roles({'dialogue': 'a', 'director': None, 'memory': 'b'}, 0, {'a', 'b'})
        roles = self.store.model_roles()
        self.assertEqual((roles['dialogue'], roles['director'], roles['memory']), ('a', None, 'b'))
        self.assertEqual(roles['revision'], 1)
        with self.assertRaises(ConflictError):
            self.store.save_model_roles({'dialogue': None, 'director': None, 'memory': None}, 0, {'a'})
        self.assertEqual(self.store.model_roles()['dialogue'], 'a')

    def test_role_accepts_database_profile_or_config_name(self):
        """FEAT-09: a role may name a config.yaml profile or a profile with a database row."""
        self.store.save_model_profile('db', LOCAL, None)
        self.store.save_model_roles({'dialogue': 'db', 'director': 'cfg', 'memory': None}, 0, {'cfg'})
        roles = self.store.model_roles()
        self.assertEqual((roles['dialogue'], roles['director']), ('db', 'cfg'))

    def test_role_save_refused_for_profile_deleted_after_page_load(self):
        """FEAT-09 (D24): a profile deleted after the caller loaded its page is refused inside the transaction; nothing changes."""
        revision = self.store.save_model_profile('gone', LOCAL, None)
        self.store.delete_model_profile('gone', revision, set())
        before = self.store.model_roles()
        with self.assertRaises(ValueError) as caught:
            self.store.save_model_roles({'dialogue': 'gone', 'director': None, 'memory': None}, 0, set())
        self.assertNotIsInstance(caught.exception, ConflictError)
        self.assertIn('dialogue', str(caught.exception))
        self.assertIn('gone', str(caught.exception))
        self.assertEqual(self.store.model_roles(), before)

    def test_unknown_profile_refused_naming_role_and_profile(self):
        """FEAT-09: a role naming a profile outside available_names raises ValueError naming both; nothing changes."""
        with self.assertRaises(ValueError) as caught:
            self.store.save_model_roles({'dialogue': 'ghost', 'director': None, 'memory': None}, 0, {'a'})
        self.assertNotIsInstance(caught.exception, ConflictError)
        self.assertIn('dialogue', str(caught.exception))
        self.assertIn('ghost', str(caught.exception))
        roles = self.store.model_roles()
        self.assertEqual((roles['dialogue'], roles['revision'], roles['version']), (None, 0, 0))

    def test_version_bumps_on_every_write(self):
        """FEAT-09 (D24): model_backend_version rises on profile save, update, role save and profile delete."""
        seen = [self.store.model_backend_version()]
        revision = self.store.save_model_profile('a', LOCAL, None)
        seen.append(self.store.model_backend_version())
        revision = self.store.save_model_profile('a', {**LOCAL, 'model': 'x'}, revision)
        seen.append(self.store.model_backend_version())
        self.store.save_model_roles({'dialogue': 'a', 'director': None, 'memory': None}, 0, set())
        seen.append(self.store.model_backend_version())
        self.store.save_model_roles({'dialogue': None, 'director': None, 'memory': None}, 1, set())
        seen.append(self.store.model_backend_version())
        self.store.delete_model_profile('a', revision, set())
        seen.append(self.store.model_backend_version())
        self.assertEqual(seen, sorted(set(seen)))
        self.assertEqual(len(seen), 6)
        self.assertEqual(self.store.model_roles()['version'], seen[-1])

    def test_failed_writes_do_not_bump_version(self):
        """FEAT-09 (D24): refused writes leave the version unchanged."""
        before = self.store.model_backend_version()
        for call in (lambda: self.store.save_model_profile('Bad', LOCAL, None),
                     lambda: self.store.save_model_roles({'dialogue': 'x', 'director': None, 'memory': None}, 0, set()),
                     lambda: self.store.save_model_roles({'dialogue': None, 'director': None, 'memory': None}, 9, set())):
            with self.assertRaises(ValueError):
                call()
        revision = self.store.save_model_profile('a', LOCAL, None)
        self.store.save_model_roles({'dialogue': 'a', 'director': None, 'memory': None}, 0, set())
        before = self.store.model_backend_version()
        with self.assertRaises(ValueError):
            self.store.delete_model_profile('a', revision, set())
        with self.assertRaises(ConflictError):
            self.store.delete_model_profile('a', revision + 100, set())
        self.assertEqual(self.store.model_backend_version(), before)


def profile(model='m', **extra):
    return profile_from_mapping('p', {'provider': 'compatible', 'model': model, 'context_tokens': 100,
                                      'base_url': 'http://localhost/v1', **extra})


def row(name, data, revision=1):
    return {'name': name, 'data': data, 'revision': revision, 'updated_at': 1.0}


def roles_row(dialogue=None, director=None, memory=None, version=3):
    return {'dialogue': dialogue, 'director': director, 'memory': memory, 'revision': 1, 'version': version}


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.config = {'base': profile('config-model'), 'other': profile('other-model')}
        self.config_roles = {'dialogue': 'base', 'director': 'base', 'memory': 'other'}

    def merge(self, rows, roles=None):
        return backend_module().effective_backend(self.config, self.config_roles, rows, roles or roles_row())

    def test_no_rows_equals_config(self):
        """FEAT-09: with no database rows the backend is the config profiles and roles, no problems."""
        result = self.merge([])
        self.assertEqual(result.profiles, self.config)
        self.assertEqual(result.roles, self.config_roles)
        self.assertEqual(result.problems, ())
        self.assertEqual(result.version, 3)

    def test_row_overrides_config_profile_by_name(self):
        """FEAT-09 (D24): a database row replaces the config profile of the same name and is marked dashboard."""
        data = {'provider': 'compatible', 'model': 'db-model', 'context_tokens': 50, 'base_url': 'http://localhost:9/v1'}
        result = self.merge([row('base', data), row('extra', LOCAL)])
        self.assertEqual(result.profiles['base'].model, 'db-model')
        self.assertEqual(result.profiles['base'].source, 'dashboard')
        self.assertEqual(result.profiles['other'], self.config['other'])
        self.assertEqual(result.profiles['extra'].model, 'local')
        self.assertEqual(result.problems, ())

    def test_invalid_row_skipped_with_problem_and_config_fallback(self):
        """FEAT-09 (D24): an invalid row is skipped, the config profile stays, and a problem names the profile."""
        bad = {'provider': 'compatible', 'model': 'x', 'context_tokens': 0, 'base_url': 'http://localhost/v1'}
        result = self.merge([row('base', bad), row('orphan', {'provider': 'nope', 'model': 'x', 'context_tokens': 1})])
        self.assertEqual(result.profiles['base'], self.config['base'])
        self.assertNotIn('orphan', result.profiles)
        self.assertEqual(len(result.problems), 2)
        self.assertIn('base', result.problems[0])
        self.assertIn('orphan', result.problems[1])

    def test_bad_json_row_is_skipped(self):
        """FEAT-09: a row whose data is not a mapping (corrupt JSON read back) is skipped with a problem."""
        result = self.merge([row('base', 'not a mapping')])
        self.assertEqual(result.profiles['base'], self.config['base'])
        self.assertEqual(len(result.problems), 1)
        self.assertIn('base', result.problems[0])

    def test_problem_never_includes_key_value(self):
        """FEAT-09 (D24): a row carrying a literal key yields a problem without the key text."""
        secret = 'sk-live-abcdef1234567890'
        result = self.merge([row('base', {**LOCAL, 'api_key_env': secret})])
        self.assertEqual(len(result.problems), 1)
        self.assertNotIn(secret, result.problems[0])

    def test_pinning_not_checked_in_merge(self):
        """FEAT-09 (D24): the merge keeps a row whose key/host pin fails; the bot checks at call time (step 3)."""
        data = {'provider': 'compatible', 'model': 'm', 'context_tokens': 10,
                'api_key_env': 'OPENAI_API_KEY', 'base_url': 'https://evil.example'}
        result = self.merge([row('evil', data)])
        self.assertIn('evil', result.profiles)
        self.assertEqual(result.problems, ())

    def test_database_role_overrides_config_role(self):
        """FEAT-09 (D24): a database role wins; NULL roles use the config role."""
        result = self.merge([row('extra', LOCAL)], roles_row(director='extra'))
        self.assertEqual(result.roles, {'dialogue': 'base', 'director': 'extra', 'memory': 'other'})
        self.assertEqual(result.problems, ())

    def test_role_naming_missing_profile_falls_back_with_problem(self):
        """FEAT-09: a database role naming an unknown profile falls back to the config role and reports it."""
        result = self.merge([], roles_row(memory='ghost'))
        self.assertEqual(result.roles['memory'], 'other')
        self.assertEqual(len(result.problems), 1)
        self.assertIn('memory', result.problems[0])
        self.assertIn('ghost', result.problems[0])

    def test_role_pointing_at_skipped_invalid_row_falls_back(self):
        """FEAT-09: a role naming a profile that exists only as an invalid row falls back to the config role."""
        bad = {'provider': 'compatible', 'model': 'x', 'context_tokens': 0, 'base_url': 'http://localhost/v1'}
        result = self.merge([row('broken', bad)], roles_row(dialogue='broken'))
        self.assertEqual(result.roles['dialogue'], 'base')
        self.assertTrue(any('broken' in p for p in result.problems))

    def test_unresolvable_role_is_absent(self):
        """FEAT-09: with empty config and no valid assignment the role is simply absent from roles."""
        result = backend_module().effective_backend({}, {'dialogue': None, 'director': None, 'memory': None}, [row('x', LOCAL)],
                                                    roles_row(dialogue='x', director='ghost'))
        self.assertEqual(result.roles, {'dialogue': 'x'})

    def test_backend_is_frozen(self):
        """FEAT-09: Backend is a frozen dataclass."""
        result = self.merge([])
        with self.assertRaises(Exception):
            result.version = 99

    def test_apply_backend_keeps_settings_role_when_absent(self):
        """FEAT-09: a role missing from backend.roles keeps the existing Settings role."""
        settings = make_settings()
        result = backend_module().Backend(profiles={'test': settings.profiles['test']}, roles={'dialogue': 'test'},
                                          problems=(), version=1)
        applied = backend_module().apply_backend(replace(settings, director='keep-d', memory='keep-m'), result)
        self.assertEqual((applied.dialogue, applied.director, applied.memory), ('test', 'keep-d', 'keep-m'))

    def test_row_with_invalid_name_skipped_without_echo(self):
        """FEAT-09 (D24): a row with an invalid name (inserted by SQL) is skipped; the problem does not echo the name."""
        store = Store(':memory:')
        bad = 'Evil Name sk-live-abcdef'
        store.db.execute('INSERT INTO model_profiles(name,data_json,revision,updated_at) VALUES(?,?,?,?)',
                         (bad, json.dumps(LOCAL), 1, 1.0))
        result = self.merge(store.model_profile_rows())
        self.assertNotIn(bad, result.profiles)
        self.assertEqual(len(result.problems), 1)
        self.assertNotIn(bad, result.problems[0])
        self.assertNotIn('sk-live', result.problems[0])

    def test_apply_backend_replaces_profiles_and_roles_only(self):
        """FEAT-09: apply_backend swaps profiles and the three roles and leaves other Settings fields alone."""
        settings = make_settings()
        result = self.merge([row('extra', LOCAL)], roles_row(dialogue='extra'))
        applied = backend_module().apply_backend(settings, result)
        self.assertEqual(applied.profiles, result.profiles)
        self.assertEqual((applied.dialogue, applied.director, applied.memory), ('extra', 'base', 'other'))
        self.assertEqual(replace(applied, profiles=settings.profiles, dialogue=settings.dialogue,
                                 director=settings.director, memory=settings.memory), settings)
        self.assertIsInstance(applied.profile('dialogue'), ModelProfile)
        self.assertEqual(applied.profile('dialogue').model, 'local')


if __name__ == '__main__':
    unittest.main()
