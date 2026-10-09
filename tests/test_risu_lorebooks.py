import json
import asyncio
import unittest
from pathlib import Path

from llmcord_core.lorebooks import digest, export_entry, parse_lorebook, validate_rule
from llmcord_core.store import Store
from llmcord_core.world_info import evaluate
from llmcord_core.config import ModelProfile, Settings
from llmcord_core.engine import Engine, SceneContext


def risu(entries, **metadata):
    return json.dumps({'type': 'risu', 'ver': 1, 'data': entries, **metadata}).encode()


class RisuLorebookTests(unittest.TestCase):
    def test_validate_rule_texts_say_what_to_do(self):
        with self.assertRaisesRegex(ValueError, 'Fix these values and try again'):
            validate_rule({'probability': 101})
        with self.assertRaisesRegex(ValueError, 'Choose one of these'):
            validate_rule({'role': 'bogus'})

    def test_supplied_export_preserves_all_records_and_folder_metadata(self):
        data = (Path(__file__).parent / 'fixtures/risu-lorebook.json').read_bytes()
        raw = json.loads(data)
        book = parse_lorebook(data)
        self.assertEqual(book.source_format, 'risu')
        self.assertEqual(len(book.entries), 6)
        self.assertEqual(json.loads(book.raw_json), raw)
        for uid, entry in book.entries.items():
            self.assertEqual(entry.rule['original'], raw['data'][int(uid)])
            self.assertEqual(entry.source_hash, digest(raw['data'][int(uid)]))
        self.assertFalse(book.entries['3'].rule['enabled'])
        self.assertTrue(book.entries['3'].warnings)
        self.assertEqual(book.entries['4'].rule['original']['folder'], raw['data'][3]['key'])
        self.assertTrue(book.entries['4'].rule['regex_enabled'])
        self.assertFalse(book.entries['5'].rule['regex_enabled'])

    def test_rules_affect_actual_activation_and_export_keeps_source_fields(self):
        raw = [
            {'id': 'secondary', 'key': 'moon, lunar', 'secondkey': 'red, scarlet', 'selective': True,
             'insertorder': 0, 'content': 'Red moon lore', 'mode': 'normal', 'alwaysActive': False,
             'folder': 'folder:1', 'extra': {'preserve': True}},
            {'id': 'constant', 'key': '', 'secondkey': 'absent', 'selective': True,
             'alwaysActive': True, 'insertorder': 20, 'content': 'Always included'},
            {'id': 'regex', 'key': '/bicycle/i', 'useRegex': True, 'content': 'Regex lore'},
            {'id': 'literal', 'key': '/bicycle/i', 'useRegex': False, 'content': 'Literal lore'},
            {'id': 'folder', 'key': 'folder:1', 'mode': 'folder', 'alwaysActive': True, 'content': 'Folder metadata'},
        ]
        book = parse_lorebook(risu(raw))
        self.assertEqual(book.entries['secondary'].rule['keys'], ['moon', 'lunar'])
        self.assertEqual(book.entries['secondary'].rule['secondary_keys'], ['red', 'scarlet'])
        self.assertEqual(book.entries['secondary'].rule['order'], 0)
        store = Store()
        try:
            world = store.create_space(1, 'World', 'world')
            store.bind_channel(1, 100, world)
            character = store.add_character(1, world, 'Alice', {'name': 'Alice'}, None, [])
            ident = store.create_lorebook(1, 'Risu book', 'channel', 100)
            store.sync_lorebook(1, ident, book, {}, 0)
            def matches(text):
                return {entry.content for entry in evaluate(store, 1, [('space', world), ('channel', 100)],
                    text, 1000, character=store.character_by_id(character))}
            self.assertEqual(matches('red moon BICYCLE'), {'Red moon lore', 'Always included', 'Regex lore'})
            self.assertEqual(matches('moon'), {'Always included'})
            self.assertIn('Literal lore', matches('/bicycle/i'))
            exported = store.export_lore(1, 'book', ident)
            self.assertEqual(exported['entries']['secondary']['folder'], 'folder:1')
            self.assertEqual(exported['entries']['secondary']['extra'], {'preserve': True})
            restored = parse_lorebook(json.dumps(exported).encode())
            self.assertFalse(restored.entries['literal'].rule['regex_enabled'])
            self.assertTrue(restored.entries['regex'].rule['regex_enabled'])
            self.assertFalse(restored.entries['folder'].rule['enabled'])
        finally:
            store.close()

    def test_reimports_detect_local_changes_and_keep_stable_ids(self):
        store = Store()
        try:
            ident = store.create_lorebook(1, 'Book', 'guild')
            first = parse_lorebook(risu([{'id': 'a', 'key': 'moon', 'content': 'Original', 'insertorder': 0}]))
            store.sync_lorebook(1, ident, first, {}, 0)
            row = store.lorebook_entries(ident)[0]
            store.edit_lorebook_entry(1, row['id'], 'Local edit')
            incoming = parse_lorebook(risu([{'id': 'a', 'key': 'moon', 'content': 'Incoming', 'insertorder': 0}]))
            self.assertEqual(store.preview_lorebook_sync(1, ident, incoming)[0]['status'], 'conflict')
            store.sync_lorebook(1, ident, incoming, {'a': 'keep'}, store.owner_revision(1, 'book', ident))
            self.assertEqual(store.lorebook_entries(ident)[0]['content'], 'Local edit')
            self.assertEqual(store.lorebook_entries(ident)[0]['id'], row['id'])
        finally:
            store.close()

    def test_unsupported_behavior_is_inactive_until_explicitly_remapped(self):
        for source in ({'mode': 'child'}, {'content': '@@depth 1\nLore'}, {'content': '{{getvar::secret}}'}):
            with self.subTest(source=source):
                entry = parse_lorebook(risu([{'id': 'a', 'key': 'moon', 'content': 'Lore', 'insertorder': 100, **source}])).entries.popitem()[1]
                self.assertTrue(entry.rule['unsupported'])
                self.assertTrue(validate_rule(entry.rule)['unsupported'])
                if 'content' in source:
                    self.assertFalse(validate_rule(entry.rule, 'Rewritten lore')['unsupported'])
        entry = parse_lorebook(risu([{'key': 'moon', 'content': 'Lore', 'mode': 'folder', 'insertorder': 100}])).entries['0']
        rule = entry.rule
        rule['original']['mode'] = 'normal'
        rule['enabled'] = True
        self.assertFalse(validate_rule(rule)['unsupported'])

    def test_rejects_bad_versions_layouts_duplicates_and_limits(self):
        payloads = [b'{"type":"risu","ver":2,"data":[]}', b'{"type":"risu","ver":1,"data":{}}',
                    risu([None]), risu([{'id': 'a'}, {'id': 'a'}]), risu([{'content': str(i)} for i in range(2001)])]
        for payload in payloads:
            with self.subTest(payload=payload[:60]), self.assertRaises(ValueError):
                parse_lorebook(payload)
        self.assertEqual(len(parse_lorebook(risu([{'content': str(i)} for i in range(2000)])).entries), 2000)

    def test_regex_flag_can_be_edited_without_stale_source_overriding_it(self):
        entry = parse_lorebook(risu([{'key': '/moon/i', 'content': 'Lore', 'useRegex': False}])).entries['0']
        entry.rule['regex_enabled'] = True
        validated = validate_rule(entry.rule)
        self.assertTrue(validated['regex_enabled'])
        self.assertTrue(parse_lorebook(json.dumps([export_entry('Lore', validated)]).encode()).entries['0'].rule['regex_enabled'])

    def test_higher_insertorder_appears_later_in_actual_character_context(self):
        store = Store()
        try:
            world = store.create_space(1, 'World', 'world')
            store.bind_channel(1, 100, world)
            character = store.add_character(1, world, 'Alice', {'name': 'Alice'}, None, [])
            ident = store.create_lorebook(1, 'Risu', 'channel', 100)
            incoming = parse_lorebook(risu([
                {'key': 'moon', 'content': 'HIGH_PRIORITY_LORE', 'insertorder': 200},
                {'key': 'moon', 'content': 'LOW_PRIORITY_LORE', 'insertorder': 0},
            ]))
            store.sync_lorebook(1, ident, incoming, {}, 0)
            profile = ModelProfile('compatible', 'test', 16000, False, base_url='http://localhost/v1')
            settings = Settings('token', None, ':memory:', 90, {'test': profile}, 'test', 'test', 'test',
                                {'max_input_tokens': 12000, 'max_output_tokens': 700})
            scene = SceneContext(1, 100, None, world, 9, 1000, 'moon', None, [], [])
            request, _ = asyncio.run(Engine(store, None, settings).prepare_dialogue(scene, store.character_by_id(character), []))
            context = '\n'.join(message.text for message in request.messages)
            self.assertLess(context.index('LOW_PRIORITY_LORE'), context.index('HIGH_PRIORITY_LORE'))
            selected = evaluate(store, 1, [('space', world), ('channel', 100)], 'moon', 1000,
                                character=store.character_by_id(character), max_entries=1)
            self.assertEqual([entry.content for entry in selected], ['HIGH_PRIORITY_LORE'])
        finally:
            store.close()


if __name__ == '__main__':
    unittest.main()
