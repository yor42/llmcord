"""PERF-03: avatar listings return no image BLOBs and read the slot table once (blobs are still read to hash them; a stored hash would need a schema change)."""
import hashlib
import unittest
from contextlib import contextmanager

from llmcord_core.admin_store import ConflictError
from llmcord_core.avatars import avatar_version
from llmcord_core.store import Store
from helpers import image

META_KEYS = {'character_id', 'slot_key', 'label', 'description', 'revision', 'has_image', 'image_hash', 'image_version'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


class AvatarQueryTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.world = self.store.create_space(1, 'World', 'world')
        self.other_world = self.store.create_space(2, 'Other', 'world')
        self.character = self.store.add_character(1, self.world, 'Alice', {'name': 'Alice'}, None, [])
        self.foreign = self.store.add_character(2, self.other_world, 'Bob', {'name': 'Bob'}, None, [])
        # add_character seeds default emotion slots; start from an empty slot table.
        self.store.execute('DELETE FROM avatar_slots')

    def tearDown(self):
        self.store.close()

    def fill(self):
        """Neutral plus five emotion slots, all with distinct images."""
        colors = {'neutral': 'gray', 'happy': 'yellow', 'sad': 'blue', 'angry': 'red', 'shy': 'pink', 'proud': 'green'}
        for key, color in colors.items():
            self.store.save_avatar(1, self.character, key, key.title(), key + ' desc', image(color))
        return {key: image(color) for key, color in colors.items()}

    def publish(self, key, data, url, character=None):
        self.store.execute('INSERT INTO avatar_assets(guild_id,character_id,slot_key,image_hash,channel_id,message_id,url,created_at) VALUES(?,?,?,?,?,?,?,?)',
            (1, character or self.character, key, sha(data), 10, 20, url, 1.0))
        return self.store.one('SELECT id FROM avatar_assets WHERE url=?', (url,))['id']

    @contextmanager
    def traced(self):
        statements = []
        self.store.db.set_trace_callback(statements.append)
        try:
            yield statements
        finally:
            self.store.db.set_trace_callback(None)

    @staticmethod
    def slot_statements(statements):
        return [s for s in statements if 'avatar_slots' in s.lower()]

    # avatar_slots metadata
    def test_slots_are_metadata_without_image_blobs(self):
        images = self.fill()
        self.store.save_avatar(1, self.character, 'bare', 'Bare', '')
        rows = {r['slot_key']: r for r in self.store.avatar_slots(1, self.character)}
        self.assertEqual(set(rows), set(images) | {'bare'})
        for row in rows.values():
            self.assertEqual(set(row), META_KEYS)
            self.assertNotIn('image', row)
            self.assertEqual(row['character_id'], self.character)
        happy = rows['happy']
        self.assertTrue(happy['has_image'])
        self.assertEqual(happy['image_hash'], sha(images['happy']))
        self.assertEqual(happy['image_version'], avatar_version(images['happy']))
        self.assertEqual((happy['label'], happy['description'], happy['revision']), ('Happy', 'happy desc', 1))
        bare = rows['bare']
        self.assertEqual((bare['has_image'], bare['image_hash'], bare['image_version']), (False, None, None))

    def test_slots_include_synthetic_neutral_default(self):
        rows = self.store.avatar_slots(1, self.character)
        self.assertEqual([r['slot_key'] for r in rows], ['neutral'])
        neutral = rows[0]
        self.assertEqual(set(neutral), META_KEYS)
        self.assertEqual((neutral['label'], neutral['description'], neutral['revision']), ('Neutral', '', 0))
        self.assertEqual((neutral['has_image'], neutral['image_hash'], neutral['image_version']), (False, None, None))
        self.store.save_avatar(1, self.character, 'happy', 'Happy', '')
        self.assertEqual({r['slot_key'] for r in self.store.avatar_slots(1, self.character)}, {'neutral', 'happy'})

    def test_slots_reject_other_guild_character(self):
        with self.assertRaises(ValueError):
            self.store.avatar_slots(1, self.foreign)

    # avatar_slot
    def test_slot_returns_image_for_requested_key_only(self):
        images = self.fill()
        row = self.store.avatar_slot(1, self.character, 'sad')
        self.assertEqual(row['image'], images['sad'])
        self.assertEqual(row['slot_key'], 'sad')
        self.assertTrue(META_KEYS <= set(row))
        self.assertEqual(row['image_hash'], sha(images['sad']))
        self.assertEqual(row['image_version'], avatar_version(images['sad']))
        self.assertTrue(row['has_image'])

    def test_slot_synthetic_neutral_when_missing(self):
        row = self.store.avatar_slot(1, self.character, 'neutral')
        self.assertEqual((row['slot_key'], row['label'], row['revision']), ('neutral', 'Neutral', 0))
        self.assertIsNone(row['image'])
        self.assertFalse(row['has_image'])

    def test_next_owner_id_rejects_unknown_owner_type_with_next_step(self):
        with self.assertRaisesRegex(ValueError, 'world, hub or lorebook. Reload the page'):
            self.store.next_owner_id('x', 'y')

    def test_slot_unknown_key_and_other_guild_raise(self):
        self.fill()
        with self.assertRaisesRegex(ValueError, 'avatar no longer exists'):
            self.store.avatar_slot(1, self.character, 'missing')
        with self.assertRaises(ValueError):
            self.store.avatar_slot(1, self.foreign, 'neutral')

    def test_slot_lookup_is_keyed_not_a_character_scan(self):
        self.fill()
        with self.traced() as statements:
            self.store.avatar_slot(1, self.character, 'happy')
        touching = self.slot_statements(statements)
        self.assertTrue(touching)
        for sql in touching:
            self.assertIn('slot_key', sql.lower().split('where', 1)[-1], sql)

    def test_save_without_image_keeps_existing_blob(self):
        images = self.fill()
        before = self.store.avatar_slot(1, self.character, 'happy')
        self.store.save_avatar(1, self.character, 'happy', 'Glad', 'new desc', None, before['revision'])
        after = self.store.avatar_slot(1, self.character, 'happy')
        self.assertEqual(after['image'], images['happy'])
        self.assertEqual((after['label'], after['description']), ('Glad', 'new desc'))
        self.assertEqual(after['revision'], before['revision'] + 1)

    def test_stale_revision_conflicts(self):
        self.fill()
        with self.assertRaises(ConflictError):
            self.store.save_avatar(1, self.character, 'happy', 'X', '', None, 99)
        with self.assertRaises(ConflictError):
            self.store.clear_avatar_image(1, self.character, 'happy', 99)
        with self.assertRaises(ConflictError):
            self.store.delete_avatar(1, self.character, 'happy', 99)
        self.assertEqual(self.store.avatar_slot(1, self.character, 'happy')['label'], 'Happy')

    def test_avatar_asset_is_guild_filtered_with_explicit_hash(self):
        data = image('red')
        self.store.execute('INSERT INTO avatar_assets(guild_id,character_id,slot_key,image_hash,channel_id,message_id,url,created_at) VALUES(?,?,?,?,?,?,?,?)',
            (2, self.character, 'happy', sha(data), 10, 20, 'https://cdn.discordapp.com/a/x.png', 1.0))
        self.assertIsNone(self.store.avatar_asset(1, self.character, 'happy', sha(data)))

    # usable_avatars
    def test_usable_avatars_parity_with_published_assets(self):
        images = self.fill()
        neutral_asset = self.publish('neutral', images['neutral'], 'https://cdn.discordapp.com/a/neutral.png')
        happy_asset = self.publish('happy', images['happy'], 'https://cdn.discordapp.com/a/happy.png')
        self.publish('sad', image('black'), 'https://cdn.discordapp.com/a/stale.png')  # hash is not the current image
        rows = {r['slot_key']: r for r in self.store.usable_avatars(1, self.character)}
        self.assertEqual(set(rows), {'neutral', 'happy'})
        self.assertEqual((rows['neutral']['asset_id'], rows['neutral']['url']), (neutral_asset, 'https://cdn.discordapp.com/a/neutral.png'))
        self.assertEqual((rows['happy']['asset_id'], rows['happy']['url']), (happy_asset, 'https://cdn.discordapp.com/a/happy.png'))
        for row in rows.values():
            self.assertNotIn('image', row)
            self.assertEqual(row['label'], row['slot_key'].title())
        self.assertEqual(rows['happy']['description'], 'happy desc')

    def test_usable_avatars_neutral_always_present_without_asset(self):
        rows = self.store.usable_avatars(1, self.character)
        self.assertEqual([r['slot_key'] for r in rows], ['neutral'])
        self.assertEqual((rows[0]['asset_id'], rows[0]['url']), (None, None))
        self.assertNotIn('image', rows[0])

    def test_usable_avatars_prefers_newest_matching_asset(self):
        images = self.fill()
        self.publish('happy', images['happy'], 'https://cdn.discordapp.com/a/old.png')
        newest = self.publish('happy', images['happy'], 'https://cdn.discordapp.com/a/new.png')
        row = next(r for r in self.store.usable_avatars(1, self.character) if r['slot_key'] == 'happy')
        self.assertEqual((row['asset_id'], row['url']), (newest, 'https://cdn.discordapp.com/a/new.png'))

    def test_usable_avatars_reads_slot_table_once(self):
        images = self.fill()
        for key, data in images.items():
            self.publish(key, data, f'https://cdn.discordapp.com/a/{key}.png')
        with self.traced() as statements:
            rows = self.store.usable_avatars(1, self.character)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(self.slot_statements(statements)), 1, self.slot_statements(statements))
