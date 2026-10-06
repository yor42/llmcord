import json
import unittest
from unittest.mock import patch

from llmcord_core.admin_store import ConflictError
from llmcord_core.lorebooks import parse_lorebook
from llmcord_core.store import Store


class CharacterTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.world = self.store.create_space(1, 'World', 'world')
        self.hub = self.store.create_space(1, 'Hub', 'hub')
        self.other_world = self.store.create_space(2, 'Other world', 'world')
        self.store.bind_channel(1, 100, self.world)

    def tearDown(self):
        self.store.close()

    def test_create_empty_card_validates_world_and_unique_name_without_overwriting(self):
        character = self.store.create_character(1, self.world, ' Alice ')
        row = self.store.character_by_id(character)
        self.assertEqual(row['name'], 'Alice')
        self.assertEqual(row['world_id'], self.world)
        card = json.loads(row['card'])
        self.assertEqual(card.pop('name'), 'Alice')
        self.assertTrue(card and all(value == '' for value in card.values()))
        self.assertEqual({s['slot_key'] for s in self.store.avatar_slots(1, character)}, {'neutral', 'happy', 'sad', 'angry', 'surprised', 'embarrassed'})
        for world, name in ((self.world, 'Alice'), (self.world, '  '), (self.hub, 'Bob'), (self.other_world, 'Bob'), (999, 'Bob')):
            with self.assertRaises(ValueError):
                self.store.create_character(1, world, name)
        self.assertEqual(len(self.store.all('SELECT * FROM characters')), 1)
        self.assertEqual(self.store.character_by_id(character)['card'], row['card'])
        self.assertIsNotNone(self.store.create_character(2, self.other_world, 'Alice'))

    def test_delete_cleans_owned_data_and_casts_but_keeps_history_and_moved_lore(self):
        character = self.store.create_character(1, self.world, 'Alice')
        survivor = self.store.create_character(1, self.world, 'Bob')
        self.store.set_cast(100, None, [character, survivor], default=True)
        self.store.set_cast(100, None, [character, survivor])
        self.store.set_cast(101, 100, [survivor, character])
        self.store.record_node(12, 1, 100, None, None, character, 'Historical reply')
        self.store.set_consent(1, 4, True)
        self.store.add_personal(1, 4, character, 'Personal fact', 12)
        self.store.add_personal(1, 4, survivor, 'Survivor fact', 12)
        self.store.add_encounter(1, character, self.world, 'Encounter', 12)
        candidate = self.store.add_candidate(1, 'character', character, 'Candidate', 12, 12)
        self.store.save_webhook_id(100, character, 999)
        self.store.execute('INSERT INTO avatar_assets VALUES(1,?,?,?,?,?,?,?,?)', (1, character, 'happy', 'hash', 100, 13, 'https://example.com/image.png', 0))
        book = self.store.create_lorebook(1, 'Book', 'guild')
        imported = parse_lorebook(json.dumps({'entries': {'owned': {'content': 'Owned lore'}, 'moved': {'content': 'Moved lore'}}}).encode())
        self.store.sync_lorebook(1, book, imported, {}, 0)
        rows = {entry['uid']: entry for entry in self.store.admin_entries(1, 'book', book)}
        owned, moved = rows['owned'], rows['moved']
        ref = self.store.transfer_entry(1, owned['ref'], 'character', character, owned['revision'])
        # Lore originating from this card but moved elsewhere is still owned elsewhere.
        self.store.execute("INSERT INTO import_entries VALUES(1,'character',?,'moved',?,'{}','hash','moved')", (character, moved['entry_key']))
        self.store.execute('INSERT INTO card_imports VALUES(?,?)', (character, '{}'))
        revision = self.store.owner_revision(1, 'character', character)
        self.store.delete_character(1, character, revision)
        self.assertIsNone(self.store.character_by_id(character))
        self.assertIsNotNone(self.store.character_by_id(survivor))
        self.assertEqual(self.store.get_cast(100), [survivor])
        self.assertEqual(self.store.get_cast(101, 100), [survivor])
        self.assertEqual(json.loads(self.store.channel(100)['default_cast']), [survivor])
        self.assertEqual(self.store.node(12)['character_id'], character)
        self.assertEqual(self.store.node(12)['content'], 'Historical reply')
        self.assertIsNotNone(self.store.admin_entry(1, moved['ref']))
        with self.assertRaises(ValueError):
            self.store.admin_entry(1, ref)
        self.assertEqual(self.store.one("SELECT disposition FROM import_entries WHERE source_kind='book' AND source_id=? AND uid='owned'", (book,))['disposition'], 'deleted')
        for table in ('personal_memories', 'encounters', 'webhooks', 'avatar_slots', 'avatar_assets', 'card_imports'):
            self.assertFalse(self.store.all(f'SELECT * FROM {table} WHERE character_id=?', (character,)))
        self.assertFalse(self.store.all('SELECT * FROM evidence WHERE candidate_id=?', (candidate,)))
        self.assertFalse(self.store.all("SELECT * FROM import_entries WHERE source_kind='character' AND source_id=?", (character,)))
        self.assertTrue(self.store.personal(1, 4, survivor))
        # The allocator covers both manual creation and card imports, even once history is purged.
        self.store.execute('DELETE FROM nodes')
        self.store.delete_character(1, survivor, self.store.owner_revision(1, 'character', survivor))
        new = self.store.create_character(1, self.world, 'New')
        self.assertGreater(new, survivor)
        self.store.delete_character(1, new, self.store.owner_revision(1, 'character', new))
        imported_id = self.store.add_character(1, self.world, 'Imported', {'name': 'Imported'}, None, [])
        self.assertGreater(imported_id, new)

    def test_delete_rejects_cross_guild_and_stale_revision_and_rolls_back_failures(self):
        character = self.store.create_character(1, self.world, 'Alice')
        revision = self.store.owner_revision(1, 'character', character)
        with self.assertRaises(ValueError):
            self.store.delete_character(2, character, revision)
        self.store.add_lore(1, 'character', character, 'Owned lore', [])
        with self.assertRaises(ConflictError):
            self.store.delete_character(1, character, revision)
        revision = self.store.owner_revision(1, 'character', character)
        self.store.set_cast(100, None, [character])
        with patch.object(self.store, '_prune_character_casts', side_effect=RuntimeError('Simulated failure')):
            with self.assertRaises(RuntimeError):
                self.store.delete_character(1, character, revision)
        self.assertIsNotNone(self.store.character_by_id(character))
        self.assertEqual(len(self.store.admin_entries(1, 'character', character)), 1)
        self.assertEqual(self.store.get_cast(100), [character])
        self.assertEqual(self.store.owner_revision(1, 'character', character), revision)
