import unittest
from unittest.mock import patch

from llmcord_core.admin_store import ConflictError
from llmcord_core.lore_workspace import snapshot
from llmcord_core.lorebooks import normalize_entry, parse_lorebook
from llmcord_core.store import Store


class LoreBulkTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.world = self.store.create_space(1, 'World', 'world')
        self.store.bind_channel(1, 1234567890123456789, self.world)
        self.character = self.store.add_character(1, self.world, 'Alice', {'name': 'Alice'}, None, [])
        self.book = self.store.create_lorebook(1, 'Book', 'guild')
        rule = normalize_entry('x', {'key': ['moon'], 'order': 0, 'cooldown': 3, 'extra': {'preserve': True}}).rule
        self.refs = [self.store.save_entry(1, 'channel', 1234567890123456789, f'Entry {i}', rule, pinned=True) for i in range(3)]

    def tearDown(self):
        self.store.close()

    def rows(self):
        return [self.store.admin_entry(1, ref) for ref in self.refs]

    def test_batch_transfer_preserves_identity_rules_and_unrelated_destination_changes(self):
        rows = self.rows()
        selected = [snapshot(row) for row in rows]
        self.store.save_entry(1, 'book', self.book, 'Independent destination change', rows[0]['rule'])
        results = self.store.transfer_entries(1, selected, 'book', self.book)
        self.assertEqual(len(results), 3)
        for row, ref in zip(rows, results):
            moved = self.store.admin_entry(1, ref)
            self.assertEqual(moved['entry_key'], row['entry_key'])
            self.assertEqual(moved['rule'], row['rule'])
            self.assertTrue(moved['pinned'])
            self.assertEqual(moved['owner_id'], self.book)
        self.assertEqual(len(self.store.admin_entries(1, 'book', self.book)), 4)

    def test_stale_selection_rejects_whole_move_and_delete(self):
        rows = self.rows()
        selected = [snapshot(row) for row in rows]
        self.store.save_entry(1, 'channel', rows[1]['owner_id'], 'Changed', rows[1]['rule'],
                              ref=rows[1]['ref'], expected_revision=rows[1]['revision'])
        for operation in (lambda: self.store.transfer_entries(1, selected, 'book', self.book),
                          lambda: self.store.delete_entries(1, selected)):
            with self.assertRaises(ConflictError):
                operation()
            self.assertEqual(len(self.store.admin_entries(1, 'channel', rows[0]['owner_id'])), 3)
            self.assertEqual(self.store.admin_entries(1, 'book', self.book), [])

    def test_transfer_rollback_after_mid_batch_failure(self):
        rows = self.rows()
        original = self.store._transfer_entry_locked
        calls = 0
        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('Simulated write failure')
            return original(*args)
        with patch.object(self.store, '_transfer_entry_locked', side_effect=fail_second):
            with self.assertRaises(RuntimeError):
                self.store.transfer_entries(1, [snapshot(row) for row in rows], 'book', self.book)
        self.assertEqual([row['entry_key'] for row in self.rows()], [row['entry_key'] for row in rows])
        self.assertEqual(self.store.admin_entries(1, 'book', self.book), [])

    def test_bulk_delete_retains_import_dispositions(self):
        incoming = parse_lorebook(b'{"entries":{"a":{"content":"A"},"b":{"content":"B"}}}')
        self.store.sync_lorebook(1, self.book, incoming, {}, 0)
        selected = [snapshot(row) for row in self.store.admin_entries(1, 'book', self.book)]
        self.assertEqual(self.store.delete_entries(1, selected), 2)
        self.assertEqual(self.store.lorebook_entries(self.book), [])
        dispositions = self.store.all('SELECT disposition FROM import_entries WHERE source_id=?', (self.book,))
        self.assertEqual([row['disposition'] for row in dispositions], ['deleted', 'deleted'])

    def test_mixed_owner_transfer_and_duplicate_or_cross_guild_rejection(self):
        first = self.rows()[0]
        ref = self.store.save_entry(1, 'character', self.character, 'Character lore', first['rule'])
        second = self.store.admin_entry(1, ref)
        with self.assertRaises(ConflictError):
            self.store.delete_entries(1, [snapshot(first), snapshot(first)])
        with self.assertRaises(ConflictError):
            self.store.delete_entries(2, [snapshot(first)])
        moved = self.store.transfer_entries(1, [snapshot(first), snapshot(second)], 'book', self.book)
        self.assertEqual(len(moved), 2)
        self.assertEqual({row['entry_key'] for row in self.store.admin_entries(1, 'book', self.book)},
                         {first['entry_key'], second['entry_key']})


if __name__ == '__main__':
    unittest.main()
