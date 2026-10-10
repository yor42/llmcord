"""MNT-04 (PERF-03): schema v11 stores each avatar slot's sha256 in avatar_slots.image_hash so listings never read the image BLOB."""
import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

from llmcord_core.store import Store

IMAGES = {'neutral': b'neutral-bytes', 'happy': b'happy-bytes'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def seeded(store):
    store.create_space(1, 'World', 'world')
    world = store.one('SELECT id FROM spaces')['id']
    cid = store.create_character(1, world, 'Ann')
    return cid


class UpgradeTests(unittest.TestCase):
    def test_v10_file_backfills_digests_and_writes_backup(self):
        """MNT-04: a v10 file whose slots hold images gets correct sha256 values, NULL where no image, and a pre-v11 backup."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.sqlite3'
            store = Store(path)
            cid = seeded(store)
            store.close()
            with closing(sqlite3.connect(path)) as db, db:
                db.execute('ALTER TABLE avatar_slots DROP COLUMN image_hash')
                for key, blob in IMAGES.items():
                    db.execute('INSERT OR REPLACE INTO avatar_slots(character_id,slot_key,label,image) VALUES(?,?,?,?)', (cid, key, key, blob))
                db.execute("INSERT OR REPLACE INTO avatar_slots(character_id,slot_key,label,image) VALUES(?,?,?,NULL)", (cid, "sad", "sad"))
                db.execute('PRAGMA user_version=10')
            store = Store(path)
            try:
                rows = {r['slot_key']: r['image_hash'] for r in store.all("SELECT slot_key,image_hash FROM avatar_slots WHERE slot_key IN ('neutral','happy','sad')")}
                self.assertEqual(rows, {'neutral': digest(IMAGES['neutral']), 'happy': digest(IMAGES['happy']), 'sad': None})
                self.assertEqual(store.one('PRAGMA user_version')[0], 14)
            finally:
                store.close()
            backups = list(Path(folder).glob('*.pre-v11-*.sqlite3'))
            self.assertEqual(len(backups), 1)
            with closing(sqlite3.connect(backups[0])) as db:
                self.assertEqual(db.execute('PRAGMA user_version').fetchone()[0], 10)

    def test_failed_backfill_rolls_back_and_retry_succeeds(self):
        """MNT-04: a failing backfill leaves the v10 file (schema, rows, version) untouched; a retry backfills."""
        from tests.test_migration_atomic import snapshot
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.sqlite3'
            store = Store(path)
            cid = seeded(store)
            store.close()
            with closing(sqlite3.connect(path)) as db, db:
                db.execute('ALTER TABLE avatar_slots DROP COLUMN image_hash')
                db.execute("UPDATE avatar_slots SET image=? WHERE character_id=? AND slot_key='neutral'", (b'img', cid))
                db.execute('PRAGMA user_version=10')
            before = snapshot(path)
            with mock.patch.object(Store, '_backfill_avatar_hashes', side_effect=RuntimeError('boom')):
                with self.assertRaisesRegex(RuntimeError, 'boom'):
                    Store(path)
            self.assertEqual(snapshot(path), before)
            store = Store(path)
            try:
                self.assertEqual(store.one("SELECT image_hash FROM avatar_slots WHERE character_id=? AND slot_key='neutral'", (cid,))[0], digest(b'img'))
            finally:
                store.close()

    def test_empty_image_backfills_as_null(self):
        """MNT-04: an empty b'' image gets a NULL hash, as save_avatar stores it."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.sqlite3'
            store = Store(path)
            cid = seeded(store)
            store.close()
            with closing(sqlite3.connect(path)) as db, db:
                db.execute('ALTER TABLE avatar_slots DROP COLUMN image_hash')
                db.execute("UPDATE avatar_slots SET image=? WHERE character_id=? AND slot_key='neutral'", (b'', cid))
                db.execute('PRAGMA user_version=10')
            store = Store(path)
            try:
                self.assertIsNone(store.one("SELECT image_hash FROM avatar_slots WHERE character_id=? AND slot_key='neutral'", (cid,))[0])
            finally:
                store.close()

    def test_v15_file_is_refused(self):
        """MNT-04: databases newer than v14 are rejected without a backup."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'new.sqlite3'
            with closing(sqlite3.connect(path)) as db, db:
                db.execute('PRAGMA user_version=15')
            with self.assertRaisesRegex(ValueError, 'newer than this application'):
                Store(path)
            self.assertEqual(list(Path(folder).glob('*.pre-*')), [])


class ListingTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(':memory:')
        self.cid = seeded(self.store)

    def tearDown(self):
        self.store.close()

    def stored(self, key):
        return self.store.one('SELECT image,image_hash FROM avatar_slots WHERE character_id=? AND slot_key=?', (self.cid, key))

    def test_listing_never_selects_the_image_column(self):
        """MNT-04: avatar_slots() reads no image BLOB and returns the same has_image/image_hash/image_version."""
        self.store.save_avatar(1, self.cid, 'neutral', 'Neutral', '', IMAGES['neutral'])
        self.store.save_avatar(1, self.cid, 'happy', 'Happy', '', IMAGES['happy'])
        self.store.save_avatar(1, self.cid, 'sad', 'Sad', '')
        statements = []
        self.store.db.set_trace_callback(statements.append)
        slots = {s['slot_key']: s for s in self.store.avatar_slots(1, self.cid)}
        self.store.db.set_trace_callback(None)
        listing = [s for s in statements if 'avatar_slots' in s]
        self.assertTrue(listing)
        for sql in listing:
            self.assertNotIn('*', sql.split('FROM')[0])
            self.assertNotRegex(sql.split('FROM')[0], r'\bimage\b')
        self.assertNotIn('image', slots['happy'])
        for key, blob in IMAGES.items():
            self.assertTrue(slots[key]['has_image'])
            self.assertEqual(slots[key]['image_hash'], digest(blob))
            self.assertEqual(slots[key]['image_version'], digest(blob)[:12])
        self.assertFalse(slots['sad']['has_image'])
        self.assertIsNone(slots['sad']['image_hash'])
        self.assertIsNone(slots['sad']['image_version'])

    def test_save_replace_and_clear_keep_image_hash_in_sync(self):
        """MNT-04: saving, replacing, keeping (no new image) and clearing an image keep image_hash equal to sha256(image)."""
        self.store.save_avatar(1, self.cid, 'happy', 'Happy', '', b'one')
        self.assertEqual(self.stored('happy')['image_hash'], digest(b'one'))
        self.store.save_avatar(1, self.cid, 'happy', 'Happy 2', 'd', b'two')
        self.assertEqual(self.stored('happy')['image_hash'], digest(b'two'))
        self.store.save_avatar(1, self.cid, 'happy', 'Happy 3', 'd')
        self.assertEqual(self.stored('happy')['image_hash'], digest(b'two'))
        self.assertEqual(self.store.avatar_slot(1, self.cid, 'happy')['image_hash'], digest(b'two'))
        revision = self.store.avatar_slot(1, self.cid, 'happy')['revision']
        self.store.clear_avatar_image(1, self.cid, 'happy', revision)
        row = self.stored('happy')
        self.assertIsNone(row['image'])
        self.assertIsNone(row['image_hash'])
        self.assertFalse(self.store.avatar_slot(1, self.cid, 'happy')['has_image'])


if __name__ == '__main__':
    unittest.main()
