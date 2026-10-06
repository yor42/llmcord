import copy
import json
import tempfile
import sqlite3
import time
import unittest
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import httpx
from fastapi import HTTPException
from PIL import Image

from llmcord_core.admin_store import ConflictError
from llmcord_core.avatars import AvatarPublisher, emotion_stream, normalize_avatar
from llmcord_core.cards import ParsedCard, parse_card
from llmcord_core.engine import Engine, SceneContext
from llmcord_core.lore import LoreMatch, lore_scopes
from llmcord_core.lorebooks import export_entry, normalize_entry, parse_lorebook, validate_rule
from llmcord_core.models import TurnMessage
from llmcord_core.prompts import block, compatibility, compile_prompt, default_bundle, export_preset, parse_preset, validate_bundle
from llmcord_core.store import SCHEMA, Store
from llmcord_core.web import create_app
from llmcord_core.world_info import evaluate
from test_core import FakeModels, settings


def book(content='Imported', **rule):
    return parse_lorebook(json.dumps({'entries': {'1': {'content': content, 'key': ['moon'], **rule}}}).encode())


def png(color='red'):
    output = BytesIO()
    Image.new('RGB', (300, 400), color).save(output, 'PNG')
    return normalize_avatar(output.getvalue())


class AdministrationTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.world = self.store.create_space(1, 'World', 'world')
        self.store.bind_channel(1, 100, self.world)
        self.character = self.store.add_character(1, self.world, 'Alice', {'name': 'Alice'}, None, [])
        self.book = self.store.create_lorebook(1, 'Book', 'channel', 100)

    def tearDown(self):
        self.store.close()

    def sync(self, incoming, resolutions=None):
        return self.store.sync_lorebook(1, self.book, incoming, resolutions or {}, self.store.owner_revision(1, 'book', self.book))

    def test_cross_table_moves_preserve_identity_rules_and_provenance(self):
        ident = self.store.add_lore(1, 'channel', 100, 'A branch fact', ['moon'], source_id=42)
        old = self.store.admin_entry(1, f'lore:{ident}')
        rule = {**old['rule'], 'order': 0, 'whole_words': True, 'cooldown': 3}
        ref = self.store.save_entry(1, 'channel', 100, old['content'], rule, ref=old['ref'], expected_revision=old['revision'])
        old = self.store.admin_entry(1, ref)
        moved = self.store.transfer_entry(1, ref, 'book', self.book, old['revision'])
        row = self.store.admin_entry(1, moved)
        self.assertEqual(row['entry_key'], old['entry_key'])
        self.assertEqual(row['source_message_id'], 42)
        self.assertEqual(row['rule']['order'], 0)
        self.assertTrue(row['rule']['whole_words'])
        self.assertFalse(row['pinned'])
        back = self.store.transfer_entry(1, moved, 'character', self.character, row['revision'])
        row = self.store.admin_entry(1, back)
        copied = self.store.transfer_entry(1, back, 'channel', 100, row['revision'], copy=True)
        self.assertNotEqual(row['entry_key'], self.store.admin_entry(1, copied)['entry_key'])
        self.assertEqual(row['rule'], self.store.admin_entry(1, copied)['rule'])

    def test_moves_between_books_and_stale_edits(self):
        self.sync(book())
        row = self.store.admin_entries(1, 'book', self.book)[0]
        other = self.store.create_lorebook(1, 'Other', 'channel', 100)
        moved = self.store.transfer_entry(1, row['ref'], 'book', other, row['revision'])
        self.assertEqual(moved, row['ref'])
        self.assertEqual(self.store.admin_entry(1, moved)['entry_key'], row['entry_key'])
        with self.assertRaises(ConflictError):
            self.store.save_entry(1, 'book', self.book, 'stale', row['rule'], ref=row['ref'], expected_revision=row['revision'])
        with self.assertRaises(ValueError):
            self.store.transfer_entry(2, moved, 'channel', 100, 0)

    def test_sync_preserves_manual_moved_deleted_and_updates_in_place(self):
        self.sync(book())
        original = self.store.admin_entries(1, 'book', self.book)[0]
        self.sync(book('Updated'))
        updated = self.store.admin_entries(1, 'book', self.book)[0]
        self.assertEqual(original['id'], updated['id'])
        self.assertEqual(original['entry_key'], updated['entry_key'])
        self.store.save_entry(1, 'book', self.book, 'Manual', normalize_entry('m', {}).rule)
        moved = self.store.transfer_entry(1, updated['ref'], 'character', self.character, updated['revision'])
        changes = self.store.preview_import(1, 'book', self.book, book('Changed again'))
        self.assertEqual(changes[0]['status'], 'conflict')
        with self.assertRaises(ConflictError):
            self.sync(book('Changed again'))
        self.sync(book('Changed again'), {'1': 'keep'})
        self.assertEqual(self.store.admin_entry(1, moved)['content'], 'Updated')
        self.assertEqual([r['content'] for r in self.store.admin_entries(1, 'book', self.book)], ['Manual'])
        row = self.store.admin_entry(1, moved)
        self.store.delete_entry(1, moved, row['revision'])
        self.sync(book('Changed yet again'), {'1': 'keep'})
        self.assertIsNone(self.store.entry_by_key(1, row['entry_key']))

    def test_owner_export_reimport_preserves_source_ids_and_reviews_manual_collision(self):
        self.sync(book())
        imported = self.store.admin_entries(1, 'book', self.book)[0]
        self.store.save_entry(1, 'book', self.book, 'Manual', normalize_entry('m', {}).rule)
        exported = parse_lorebook(json.dumps(self.store.export_lore(1, 'book', self.book)).encode())
        changes = self.store.preview_import(1, 'book', self.book, exported)
        statuses = {c['uid']: c['status'] for c in changes}
        self.assertIn(statuses['1'], {'unchanged', 'update'})
        manual_uid = next(uid for uid, status in statuses.items() if status == 'conflict')
        self.sync(exported, {manual_uid: 'keep'})
        self.assertEqual(len(self.store.admin_entries(1, 'book', self.book)), 2)
        self.assertEqual(self.store.admin_entry(1, imported['ref'])['entry_key'], imported['entry_key'])

    def test_import_revision_and_zero_priority(self):
        self.sync(book(order=0))
        row = self.store.admin_entries(1, 'book', self.book)[0]
        self.assertEqual(row['rule']['order'], 0)
        revision = self.store.owner_revision(1, 'book', self.book)
        self.store.edit_lorebook_entry(1, row['id'], 'Local')
        self.assertGreater(self.store.owner_revision(1, 'book', self.book), revision)
        with self.assertRaises(ConflictError):
            self.store.sync_lorebook(1, self.book, book('Incoming'), {'1': 'import'}, revision)

    def test_v2_migration_preserves_ids_avatars_traces_and_activation_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.sqlite3'
            rule = normalize_entry('7', {'id': 7, 'content': 'Legacy fact', 'constant': True}).rule
            raw_card = {'name': 'Legacy', 'character_book': {'entries': [rule['original']]}}
            with sqlite3.connect(path) as connection:
                connection.executescript(SCHEMA)
                connection.execute('ALTER TABLE characters ADD COLUMN archived INTEGER NOT NULL DEFAULT 0')
                connection.execute("ALTER TABLE lore ADD COLUMN rule_json TEXT NOT NULL DEFAULT '{}'")
                connection.execute("INSERT INTO spaces VALUES(1,1,'World','world')")
                connection.execute('INSERT INTO characters(id,guild_id,world_id,name,card,avatar) VALUES(5,1,1,?,?,?)', ('Legacy', json.dumps(raw_card), png()))
                connection.execute("INSERT INTO lore(id,guild_id,scope_kind,scope_id,content,constant,rule_json) VALUES(4,1,'character',5,'Legacy fact',1,?)", (json.dumps(rule),))
                connection.execute("INSERT INTO trace VALUES(99,'{\"legacy\":true}')")
                connection.execute("INSERT INTO lore_activations VALUES(99,'lore:4')")
                connection.execute('PRAGMA user_version=2')
            connection.close()
            upgraded = Store(path)
            try:
                self.assertEqual(upgraded.character_by_id(5)['avatar'], png())
                self.assertEqual(upgraded.lore_row(1, 4)['entry_key'], 'lore:4')
                self.assertEqual(upgraded.trace(99), {'legacy': True})
                self.assertEqual(upgraded.branch_lore_activations([99])[0]['entry_key'], 'lore:4')
                self.assertTrue(upgraded.admin_entries(1, 'character', 5))
                self.assertEqual(len(upgraded.avatar_slots(1, 5)), 6)
                self.assertEqual(len(list(Path(directory).glob('*.pre-v3-*.sqlite3'))), 1)
            finally:
                upgraded.close()

    def test_deleted_default_emotion_is_not_recreated(self):
        self.store.delete_avatar(1, self.character, 'happy', 0)
        self.assertNotIn('happy', [s['slot_key'] for s in self.store.avatar_slots(1, self.character)])

    def test_book_parser_accepts_limit_and_rejects_excess(self):
        raw = [{'uid': i, 'content': str(i)} for i in range(2000)]
        self.assertEqual(len(parse_lorebook(json.dumps(raw).encode()).entries), 2000)
        with self.assertRaises(ValueError):
            parse_lorebook(json.dumps(raw + [{'uid': 2000, 'content': 'excess'}]).encode())

    def test_card_reimport_keeps_local_lore_and_manual_avatar(self):
        first = parse_card('alice.json', json.dumps({'name': 'Alice', 'description': 'original', 'character_book': {'entries': [{'id': 1, 'content': 'First', 'keys': ['x']}]}}).encode())
        preview = self.store.preview_card(1, self.world, first)
        self.store.apply_card(1, self.world, first, {'card': 'import'}, preview['revision'])
        lore = self.store.admin_entries(1, 'character', self.character)[0]
        self.store.save_entry(1, 'character', self.character, 'Local lore', lore['rule'], ref=lore['ref'], expected_revision=lore['revision'])
        self.store.save_entry(1, 'character', self.character, 'Manual addition', normalize_entry('manual', {}).rule)
        self.store.save_avatar(1, self.character, 'neutral', 'Neutral', '', png())
        second = ParsedCard(first.name, first.data, [{**first.entries[0], 'content': 'New imported'}], png('blue'))
        preview = self.store.preview_card(1, self.world, second)
        self.assertIn('avatar', preview['conflicts'])
        self.store.apply_card(1, self.world, second, {'avatar': 'keep', '1': 'keep'}, preview['revision'])
        self.assertEqual(self.store.character_by_id(self.character)['avatar'], png())
        self.assertEqual({r['content'] for r in self.store.admin_entries(1, 'character', self.character)}, {'Local lore', 'Manual addition'})

    def test_rule_export_preserves_extensions_and_remaps_warnings(self):
        rule = normalize_entry('x', {'content': 'X', 'vectorized': True, 'custom_extension': {'a': 1}, 'order': 0}).rule
        self.assertTrue(validate_rule(rule)['unsupported'])
        rule['original']['vectorized'] = False
        edited = validate_rule(rule)
        self.assertFalse(edited['unsupported'])
        self.assertEqual(export_entry('X', edited)['custom_extension'], {'a': 1})
        self.assertEqual(export_entry('X', edited)['order'], 0)

    def test_guild_preset_versions_are_immutable_and_isolated(self):
        original = default_bundle()
        ident, revision = self.store.save_preset(1, 'A', original)
        self.store.activate_preset(1, ident, revision, {'dialogue': 'compatible'})
        snapshot = self.store.active_preset(1)
        original['purposes']['dialogue'][0]['content'] = 'Changed'
        self.store.save_preset(1, 'A renamed', original, ident, revision)
        self.assertNotEqual(self.store.preset_bundle(1, ident)['purposes']['dialogue'][0]['content'], snapshot['bundle']['purposes']['dialogue'][0]['content'])
        self.assertEqual(self.store.active_preset(1)['revision'], revision)
        self.assertEqual(self.store.active_preset(2)['id'], 0)
        with self.assertRaises(ValueError):
            self.store.preset_bundle(2, ident)
        with self.assertRaises(ValueError):
            self.store.delete_preset(1, ident)
        with self.assertRaises(ConflictError):
            self.store.save_preset(1, 'A', original, ident, revision)


class PromptTests(unittest.TestCase):
    def test_actual_post_history_and_anthropic_adaptation(self):
        values = {'char': 'Alice', 'card_post_history': 'FINAL'}
        history = [TurnMessage('user', 'INPUT')]
        rendered = compile_prompt(default_bundle(), 'dialogue', values, history, 'compatible', 4000)
        self.assertEqual([m.text for m in rendered.messages][-2:], ['INPUT', 'FINAL'])
        anthropic = compile_prompt(default_bundle(), 'dialogue', values, history, 'anthropic', 4000)
        self.assertIn('card_post_history: top_system', anthropic.adaptations)
        self.assertEqual(anthropic.messages[-1].text, 'INPUT')

    def test_dynamic_lore_requires_explicit_anthropic_mapping(self):
        bundle = default_bundle()
        marker = next(b for b in bundle['purposes']['dialogue'] if b['source'] == 'lore_in_chat')
        marker['adaptation'] = ''
        marker['role'] = 'user'
        self.assertTrue(compatibility(bundle, {'dialogue': 'anthropic'}))
        marker['adaptation'] = 'top_system'
        self.assertFalse(compatibility(bundle, {'dialogue': 'anthropic'}))

    def test_budget_keeps_input_and_contract_and_substitutes_once(self):
        bundle = default_bundle()
        bundle['purposes']['dialogue'] = [block('custom', content='Hi {{char}}'), block('history', 'history')]
        request = compile_prompt(bundle, 'dialogue', {'char': '{{user}}'}, [TurnMessage('user', 'old ' * 200), TurnMessage('user', 'CURRENT')], 'compatible', 80, contract='REQUIRED')
        self.assertEqual(request.history_indices, [1])
        self.assertIn('history:0', request.omitted)
        self.assertIn('Hi {{user}}', [m.text for m in request.messages])
        self.assertIn('REQUIRED', [m.text for m in request.messages])
        with self.assertRaises(ValueError):
            compile_prompt(bundle, 'dialogue', {}, [TurnMessage('user', 'x' * 2000)], 'compatible', 10, contract='REQUIRED')

    def test_lore_depth_injection_and_disabled_marker(self):
        item = LoreMatch(1, 'DEPTH', 'channel', 100, 'constant', 'key', 'in_chat', 1, 'system', rule={'order': 0})
        history = [TurnMessage('user', 'OLD'), TurnMessage('user', 'NOW')]
        request = compile_prompt(default_bundle(), 'dialogue', {}, history, 'compatible', 4000, lore_injections=[item])
        texts = [m.text for m in request.messages]
        self.assertLess(texts.index('World Info [key]: DEPTH'), texts.index('NOW'))
        self.assertGreater(texts.index('World Info [key]: DEPTH'), texts.index('OLD'))
        self.assertEqual(request.lore_keys, ['key'])
        bundle = default_bundle()
        next(b for b in bundle['purposes']['dialogue'] if b['source'] == 'lore_in_chat')['enabled'] = False
        self.assertFalse(compile_prompt(bundle, 'dialogue', {}, history, 'compatible', 4000, lore_injections=[item]).lore_keys)

    def test_depth_zero_order_and_role_groups(self):
        bundle = default_bundle()
        bundle['purposes']['dialogue'] = [block('history', 'history'),
            block('s', content='SYSTEM', placement='in_chat', depth=0, order=1),
            block('u2', content='USER TWO', role='user', placement='in_chat', depth=0, order=2),
            block('u1', content='USER ONE', role='user', placement='in_chat', depth=0, order=1)]
        request = compile_prompt(bundle, 'dialogue', {}, [TurnMessage('user', 'INPUT')], 'compatible', 4000)
        self.assertEqual([m.text for m in request.messages], ['INPUT', 'USER ONE', 'USER TWO', 'SYSTEM'])

    def test_sillytavern_order_selection_roundtrip_and_unknowns(self):
        raw = {'temperature': 0.7, 'custom': {'preserved': True}, 'prompts': [
            {'identifier': 'main', 'name': 'Main', 'content': 'Hi {{char}}', 'role': 'system'},
            {'identifier': 'chatHistory', 'marker': True},
            {'identifier': 'extra', 'content': '{{unknown}}'}],
            'prompt_order': [{'character_id': 100001, 'order': [{'identifier': 'main', 'enabled': True}, {'identifier': 'chatHistory', 'enabled': True}]},
                             {'character_id': 42, 'order': [{'identifier': 'chatHistory', 'enabled': True}]}]}
        data = json.dumps(raw).encode()
        bundle, orders = parse_preset(data)
        self.assertIsNone(bundle)
        self.assertEqual(len(orders), 2)
        bundle, _ = parse_preset(data, 0)
        self.assertFalse(compatibility(bundle, {'dialogue': 'compatible'}))
        extras = [b['id'] for b in bundle['purposes']['dialogue'] if b['id'].startswith('llmcord-')]
        exported = export_preset(bundle, True, extras)
        self.assertEqual(exported['temperature'], 0.7)
        self.assertEqual(exported['custom'], raw['custom'])
        parsed, _ = parse_preset(json.dumps(exported).encode())
        self.assertEqual(parsed['purposes']['dialogue'][0]['content'], 'Hi {{char}}')
        next(b for b in bundle['purposes']['dialogue'] if b['id'] == 'extra')['enabled'] = True
        self.assertTrue(compatibility(bundle, {'dialogue': 'compatible'}))
        self.assertEqual(parse_preset(json.dumps(export_preset(bundle)).encode())[0], bundle)

    def test_nonportable_export_requires_omission(self):
        with self.assertRaises(ValueError):
            export_preset(default_bundle(), True)
        nonportable = [b['id'] for b in default_bundle()['purposes']['dialogue'] if b['source'] not in {'text', 'history', 'description', 'personality', 'scenario', 'examples', 'lore_before_char', 'lore_after_char'}]
        exported = export_preset(default_bundle(), True, nonportable)
        self.assertTrue(exported['prompts'])

    def test_required_payload_marker_cannot_be_removed(self):
        bundle = default_bundle()
        bundle['purposes']['director'] = [block('text', content='No input')]
        with self.assertRaises(ValueError):
            validate_bundle(bundle)

    def test_import_formats_and_additive_card_slots(self):
        raw = {'wi_format': 'Lore: {0}', 'scenario_format': 'Scene: {{scenario}}',
            'prompts': [{'identifier': 'worldInfoBefore', 'marker': True}, {'identifier': 'scenario', 'marker': True}, {'identifier': 'chatHistory', 'marker': True}],
            'prompt_order': [{'identifier': name, 'enabled': True} for name in ('worldInfoBefore', 'scenario', 'chatHistory')]}
        bundle, _ = parse_preset(json.dumps(raw).encode())
        request = compile_prompt(bundle, 'dialogue', {'lore_before_char': '{{user}} literal', 'scenario': 'Tea shop', 'card_instructions': 'CARD', 'card_post_history': 'POST'}, [TurnMessage('user', 'INPUT')], 'compatible', 4000)
        texts = [m.text for m in request.messages]
        self.assertIn('Lore: {{user}} literal', texts)
        self.assertIn('Scene: Tea shop', texts)
        self.assertLess(texts.index('CARD'), texts.index('INPUT'))
        self.assertGreater(texts.index('POST'), texts.index('INPUT'))


class AsyncFeatureTests(unittest.IsolatedAsyncioTestCase):
    async def test_emotion_headers_at_every_chunk_boundary(self):
        async def stream(parts):
            for part in parts:
                yield part
        text = '<emotion>happy</emotion>\nHello world'
        for boundary in range(1, len(text)):
            events = [e async for e in emotion_stream(stream([text[:boundary], text[boundary:]]), {'neutral', 'happy'})]
            self.assertEqual(events[0].emotion, 'happy')
            self.assertEqual(''.join(e.text for e in events), 'Hello world')
        for text, body in [('No header', 'No header'), ('<emotion>unknown</emotion>\nReply', 'Reply'), ('<emotion>broken\nReply', 'Reply'), ('<emotion=happy>\nReply', 'Reply')]:
            events = [e async for e in emotion_stream(stream(list(text)), {'neutral'})]
            self.assertEqual(events[0].emotion, 'neutral')
            self.assertEqual(''.join(e.text for e in events), body)
        events = [e async for e in emotion_stream(stream(['<emotion>' + 'x' * 400, 'x' * 400, '\nReply']), {'neutral'})]
        self.assertEqual(''.join(e.text for e in events), 'Reply')

    async def test_avatar_publication_normalization_and_history(self):
        store = Store()
        w = store.create_space(1, 'W', 'world')
        c = store.add_character(1, w, 'Alice', {'name': 'Alice'}, None, [])
        calls = []
        def api(request):
            calls.append(request)
            if request.method == 'GET':
                return httpx.Response(200, json=[{'id': '100', 'type': 0, 'permission_overwrites': [{'id': '1', 'deny': str(1 << 10)}]}])
            return httpx.Response(200, json={'id': '55', 'attachments': [{'url': 'https://cdn.discordapp.com/attachments/100/55/avatar.png?ex=old&hm=signed'}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as http:
            publisher = AvatarPublisher(store, http, 'test')
            await publisher.configure(1, 100)
            store.save_avatar(1, c, 'happy', 'Happy', 'Smiling', png())
            await publisher.publish(1, c, 'happy')
            self.assertEqual(store.avatar_asset(1, c, 'happy')['url'], 'https://cdn.discordapp.com/attachments/100/55/avatar.png')
            store.save_avatar(1, c, 'happy', 'Happy', 'Smiling', png('blue'))
            self.assertIsNone(store.avatar_asset(1, c, 'happy'))
            self.assertEqual(store.one('SELECT COUNT(*) AS n FROM avatar_assets')['n'], 1)
            self.assertEqual([s['slot_key'] for s in store.usable_avatars(1, c)], ['neutral'])
        store.close()

    async def test_live_service_rechecks_permission_csrf_and_expiry(self):
        allowed = [True]
        async def api(request):
            return httpx.Response(200, json=[{'id': '1', 'permissions': '8' if allowed[0] else '0'}])
        async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as http:
            app = create_app(':memory:', 'https://pi.test', 'client', 'secret', 'bot', http, enable_dashboard=False)
            app.state.sessions['s'] = {'user': {'id': '4'}, 'csrf': 'token', 'expires': time.time() + 100, 'token_expires': time.time() + 100, 'access': 'a'}
            service = app.state.admin
            with self.assertRaises(HTTPException):
                await service.run('s', 1, 'wrong', lambda: service.store.create_space(1, 'Bad', 'world'))
            allowed[0] = False
            with self.assertRaises(HTTPException):
                await service.run('s', 1, 'token', lambda: service.store.create_space(1, 'Bad', 'world'))
            allowed[0] = True
            app.state.sessions['s']['expires'] = 0
            with self.assertRaises(HTTPException):
                await service.run('s', 1, 'token', lambda: service.store.create_space(1, 'Bad', 'world'))
            self.assertFalse(service.store.list_spaces(1))
            service.store.close()

    async def test_preview_does_not_read_personal_memory(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[]))) as http:
            app = create_app(':memory:', 'https://pi.test', 'c', 's', 'b', http, enable_dashboard=False)
            store = app.state.store
            w = store.create_space(1, 'W', 'world')
            c = store.add_character(1, w, 'Alice', {'name': 'Alice'}, None, [])
            store.set_consent(1, 99, True)
            store.add_personal(1, 99, c, 'PRIVATE FACT', 9)
            request = app.state.admin.preview_prompt(1, default_bundle(), 'dialogue', c, 'Sample', 'Earlier sample')
            self.assertNotIn('PRIVATE FACT', str(request))
            store.close()
