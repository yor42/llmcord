import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmcord_core.admin_store import ConflictError
from llmcord_core.engine import Engine, SceneContext
from llmcord_core.lore import lore_scopes
from llmcord_core.lorebooks import parse_lorebook
from llmcord_core.models import TurnMessage
from llmcord_core.prompts import block, compile_prompt, default_bundle
from llmcord_core.store import Store
from llmcord_core.world_info import evaluate
from test_core import FakeModels, settings


def imported(content='Imported fact'):
    return parse_lorebook(json.dumps({'entries': {'1': {'content': content, 'constant': True, 'order': 37, 'custom': {'preserved': True}}}}).encode())


class SceneSettingsTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.world = self.store.create_space(1, 'World', 'world')
        self.other = self.store.create_space(2, 'Other', 'world')
        self.store.bind_channel(1, 100, self.world)
        self.character = self.store.create_character(1, self.world, 'Alice')

    def tearDown(self):
        self.store.close()

    def test_guidelines_validate_scope_length_revision_and_clear(self):
        self.store.save_guidelines(1, 'space', self.world, 'World rules', 0)
        self.store.save_guidelines(1, 'channel', 100, 'Channel rules', 0)
        snapshot = self.store.scene_guidelines(1, self.world, 100)
        self.assertEqual(snapshot['world_guidelines']['content'], 'World rules')
        self.assertEqual(snapshot['channel_guidelines']['content'], 'Channel rules')
        for operation in (lambda: self.store.save_guidelines(1, 'space', self.other, 'Bad', 0),
                          lambda: self.store.save_guidelines(2, 'channel', 100, 'Bad', 0),
                          lambda: self.store.save_guidelines(1, 'character', self.character, 'Bad', 0),
                          lambda: self.store.save_guidelines(1, 'space', self.world, 'x' * 6001, 1)):
            with self.assertRaises(ValueError):
                operation()
        with self.assertRaises(ConflictError):
            self.store.save_guidelines(1, 'space', self.world, 'Stale', 0)
        self.store.save_guidelines(1, 'channel', 100, '', 1)
        self.assertNotIn('channel_guidelines', self.store.scene_guidelines(1, self.world, 100))
        self.assertEqual(snapshot['channel_guidelines']['content'], 'Channel rules')

    def test_managed_guidelines_survive_custom_presets_and_trimming_for_each_provider(self):
        bundle = default_bundle()
        bundle['purposes']['dialogue'] = [block('main', content='LONG BACKGROUND ' * 1000), block('history', 'history', role='user'), block('final', content='CARD FINAL', adaptation='top_system')]
        values = {'world_guidelines': 'WORLD RULE {{user}} literal', 'channel_guidelines': 'CHANNEL RULE'}
        for provider in ('compatible', 'anthropic', 'gemini'):
            with self.subTest(provider=provider):
                request = compile_prompt(bundle, 'dialogue', values, [TurnMessage('user', 'INPUT')], provider, 400, contract='CONTRACT')
                text = '\n'.join(message.text for message in request.messages)
                self.assertLess(text.index('CARD FINAL'), text.index('WORLD RULE'))
                self.assertLess(text.index('WORLD RULE'), text.index('CHANNEL RULE'))
                self.assertIn('{{user}} literal', text)
                self.assertIn('main', request.omitted)
                self.assertIn('world_guidelines', request.sources)
                self.assertIn('channel_guidelines', request.sources)
                self.assertIn('CONTRACT', text)
                self.assertIn('INPUT', text)
        with self.assertRaises(ValueError):
            compile_prompt(bundle, 'dialogue', values, [TurnMessage('user', 'INPUT')], 'gemini', 10)
        director = compile_prompt(default_bundle(), 'director', {**values, 'payload': 'INPUT'}, [], 'compatible', 4000)
        self.assertNotIn('world_guidelines', director.sources)

    def test_direct_import_is_additive_idempotent_and_preserves_rules_in_every_owner(self):
        book = self.store.create_lorebook(1, 'Book', 'guild')
        self.store.record_node(9, 1, 101, None, 4, None, 'Thread message')
        for kind, ident in (('guild', 1), ('space', self.world), ('channel', 100), ('character', self.character), ('thread', 101), ('book', book)):
            with self.subTest(kind=kind):
                incoming = imported()
                revision = self.store.owner_revision(1, kind, ident)
                refs = self.store.import_lore_entries(1, kind, ident, incoming, revision)
                row = self.store.admin_entry(1, refs[0])
                self.assertEqual((row['owner_kind'], row['owner_id']), (kind, ident))
                self.assertEqual(row['rule']['order'], 37)
                self.assertEqual(row['rule']['original']['custom'], {'preserved': True})
                self.assertEqual(self.store.preview_entry_import(1, kind, ident, incoming)[0]['status'], 'duplicate')
                self.assertEqual(self.store.import_lore_entries(1, kind, ident, incoming, self.store.owner_revision(1, kind, ident)), [])
                self.store.import_lore_entries(1, kind, ident, imported('Different source file'), self.store.owner_revision(1, kind, ident))
                self.assertEqual(len(self.store.admin_entries(1, kind, ident)), 2)
                with self.assertRaises(ConflictError):
                    self.store.import_lore_entries(1, kind, ident, imported('Stale'), revision)
        self.assertEqual(len(self.store.list_lorebooks(1)), 1)
        with self.assertRaises(ValueError):
            self.store.import_lore_entries(1, 'guild', 2, imported(), 0)
        with self.assertRaises(ValueError):
            self.store.import_lore_entries(2, 'space', self.world, imported(), 0)

    def test_guild_lore_activates_only_in_its_server_and_moves_between_all_tables(self):
        ref = self.store.import_lore_entries(1, 'guild', 1, imported(), 0)[0]
        row = self.store.admin_entry(1, ref)
        other_char = self.store.create_character(2, self.other, 'Bob')
        self.store.bind_channel(2, 200, self.other)
        self.assertEqual([m.content for m in evaluate(self.store, 1, lore_scopes(self.store, self.character, self.world, 100), '', 1000, character=self.store.character_by_id(self.character))], ['Imported fact'])
        self.assertFalse(evaluate(self.store, 2, lore_scopes(self.store, other_char, self.other, 200), '', 1000, character=self.store.character_by_id(other_char)))
        book = self.store.create_lorebook(1, 'Book', 'guild')
        for kind, ident in (('space', self.world), ('guild', 1), ('book', book), ('guild', 1), ('character', self.character)):
            ref = self.store.transfer_entry(1, ref, kind, ident, self.store.admin_entry(1, ref)['revision'])
            self.assertEqual(self.store.admin_entry(1, ref)['entry_key'], row['entry_key'])
        self.assertFalse(self.store.admin_entries(1, 'guild', 1))

    def test_space_deletion_blocks_dependencies_then_cleans_owned_data_without_id_reuse(self):
        book = self.store.create_lorebook(1, 'Shared', 'guild')
        self.store.set_lorebook_space(1, book, self.world, True)
        self.store.save_guidelines(1, 'space', self.world, 'Rules', 0)
        self.store.import_lore_entries(1, 'space', self.world, imported(), self.store.owner_revision(1, 'space', self.world))
        revision = self.store.owner_revision(1, 'space', self.world)
        with self.assertRaises(ValueError):
            self.store.delete_space(1, self.world, revision)
        with self.assertRaises(ValueError):
            self.store.delete_space(2, self.world, revision)
        replacement = self.store.create_space(1, 'Replacement', 'world')
        self.store.update_character(1, self.character, replacement, 'Alice', {'name': 'Alice'})
        self.store.bind_channel(1, 100, replacement)
        self.store.add_encounter(1, self.character, self.world, 'Old encounter', 9)
        self.store.delete_space(1, self.world, revision)
        self.assertIsNone(self.store.space_by_id(self.world))
        self.assertIsNotNone(self.store.lorebook(1, book))
        self.assertEqual(self.store.lorebook_links(book), [])
        self.assertFalse(self.store.all("SELECT * FROM scene_guidelines WHERE kind='space' AND owner_id=?", (self.world,)))
        self.assertFalse(self.store.all('SELECT * FROM encounters WHERE space_id=?', (self.world,)))
        self.assertIsNotNone(self.store.character_by_id(self.character))
        empty = self.store.create_space(1, 'Empty', 'world')
        self.store.delete_space(1, empty, 0)
        self.assertGreater(self.store.create_space(1, 'New', 'world'), empty)

    def test_book_deletion_keeps_moved_entries_and_rolls_back_partial_failure(self):
        book = self.store.create_lorebook(1, 'Book', 'guild')
        data = parse_lorebook(json.dumps({'entries': {'1': {'content': 'Owned'}, '2': {'content': 'Moved'}}}).encode())
        self.store.sync_lorebook(1, book, data, {}, 0)
        entries = {r['uid']: r for r in self.store.admin_entries(1, 'book', book)}
        moved = self.store.transfer_entry(1, entries['2']['ref'], 'space', self.world, entries['2']['revision'])
        self.store.delete_lorebook(1, book, self.store.owner_revision(1, 'book', book))
        self.assertEqual(self.store.admin_entry(1, moved)['content'], 'Moved')
        self.assertIsNone(self.store.lorebook(1, book))
        new = self.store.create_lorebook(1, 'New', 'guild')
        self.assertGreater(new, book)
        self.store.sync_lorebook(1, new, data, {}, 0)
        revision = self.store.owner_revision(1, 'book', new)
        original = self.store._delete_entry_locked
        calls = 0
        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('Simulated failure')
            original(*args)
        with patch.object(self.store, '_delete_entry_locked', side_effect=fail_second):
            with self.assertRaises(RuntimeError):
                self.store.delete_lorebook(1, new, revision)
        self.assertEqual(len(self.store.admin_entries(1, 'book', new)), 2)
        self.assertEqual(self.store.owner_revision(1, 'book', new), revision)

    def test_new_settings_survive_database_reopen_with_existing_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'store.sqlite3'
            store = Store(path)
            world = store.create_space(1, 'World', 'world')
            store.save_guidelines(1, 'space', world, 'Saved', 0)
            ref = store.import_lore_entries(1, 'guild', 1, imported(), 0)[0]
            store.close()
            store = Store(path)
            try:
                self.assertEqual(store.guidelines(1, 'space', world)['content'], 'Saved')
                self.assertEqual(store.admin_entry(1, ref)['content'], 'Imported fact')
            finally:
                store.close()


class GuidelineEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_channel_guidelines_inherit_in_threads_and_use_current_hub_not_home_world(self):
        store = Store()
        try:
            world = store.create_space(1, 'World', 'world')
            hub = store.create_space(1, 'Hub', 'hub')
            store.link_world(1, hub, world)
            store.bind_channel(1, 100, hub)
            char = store.create_character(1, world, 'Alice')
            store.save_guidelines(1, 'space', world, 'HOME RULE', 0)
            store.save_guidelines(1, 'space', hub, 'HUB RULE', 0)
            store.save_guidelines(1, 'channel', 100, 'CHANNEL RULE', 0)
            snapshot = store.scene_guidelines(1, hub, 100)
            store.save_guidelines(1, 'channel', 100, 'NEW RULE', 1)
            engine = Engine(store, FakeModels(), settings())
            scene = SceneContext(1, 101, 100, hub, 9, 12, 'Hello', None, [], [], guidelines=snapshot)
            engine.record_user(scene)
            request, trace = await engine.prepare_dialogue(scene, store.character_by_id(char), [])
            text = '\n'.join(message.text for message in request.messages)
            self.assertIn('HUB RULE', text)
            self.assertIn('CHANNEL RULE', text)
            self.assertNotIn('HOME RULE', text)
            self.assertNotIn('NEW RULE', text)
            self.assertEqual(trace['guidelines']['channel_guidelines']['revision'], 1)
            self.assertNotIn('content', trace['guidelines']['channel_guidelines'])
        finally:
            store.close()
