"""PERF-02: lore entry listing is built in one query per kind and paged/filtered in SQL.

Pins `admin_entries` parity with `admin_entry`, and the contract of
`AdminStore.admin_entries_page(guild, kind, owner, query, limit, offset)`.
"""
import json
import unittest

from llmcord_core.lorebooks import normalize_entry
from llmcord_core.store import Store

G, OTHER = 1, 2
CHANNEL = 1234567890123456789
OTHER_CHANNEL = 1234567890123456790


def rule(keys=(), order=100):
    return normalize_entry('x', {'keys': list(keys), 'order': order}).rule


class LorePagingTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        s = self.store
        self.world = s.create_space(G, 'World', 'world')
        s.bind_channel(G, CHANNEL, self.world)
        self.char = s.add_character(G, self.world, 'Alice', {'name': 'Alice'}, None, [])
        self.book = s.create_lorebook(G, 'Book', 'guild')
        self.other_world = s.create_space(OTHER, 'Other', 'world')
        s.bind_channel(OTHER, OTHER_CHANNEL, self.other_world)
        self.other_book = s.create_lorebook(OTHER, 'OBook', 'guild')
        self.owners = {
            'character': (G, self.char),
            'channel': (G, CHANNEL),
            'guild': (G, G),
            'book': (G, self.book),
        }

    def tearDown(self):
        self.store.close()

    def add(self, kind, content, keys=(), order=100, guild=None):
        g, owner = self.owners[kind]
        return self.store.save_entry(guild or g, kind, owner, content, rule(keys, order))

    def listing(self, kind):
        g, owner = self.owners[kind]
        return self.store.admin_entries(g, kind, owner)

    def page(self, kind, query='', limit=50, offset=0):
        g, owner = self.owners[kind]
        return self.store.admin_entries_page(g, kind, owner, query, limit, offset)

    def refs(self, entries):
        return [e['ref'] for e in entries]

    def test_admin_entries_match_admin_entry_and_ordering(self):
        """PERF-02: listing returns exactly what admin_entry returns, in today's order."""
        for kind in self.owners:
            with self.subTest(kind=kind):
                made = [self.add(kind, f'{kind} {i}', ['k%d' % i], order=o) for i, o in enumerate([50, 10, 50, 0])]
                entries = self.listing(kind)
                g = self.owners[kind][0]
                self.assertEqual(entries, [self.store.admin_entry(g, e['ref']) for e in entries])
                if kind == 'book':
                    self.assertEqual(self.refs(entries), made)  # by id
                else:
                    self.assertEqual(self.refs(entries), [made[3], made[1], made[0], made[2]])  # insertion_order,id

    def test_pages_and_totals_for_120_entries(self):
        """PERF-02: 120 entries page as 50/50/20; each page is the matching slice."""
        for kind in self.owners:
            with self.subTest(kind=kind):
                for i in range(120):
                    self.add(kind, f'entry {i}', order=i % 7)
                full = self.listing(kind)
                self.assertEqual(len(full), 120)
                sizes = []
                for offset in (0, 50, 100):
                    entries, total = self.page(kind, limit=50, offset=offset)
                    self.assertEqual(total, 120)
                    self.assertEqual(entries, full[offset:offset + 50])
                    sizes.append(len(entries))
                self.assertEqual(sizes, [50, 50, 20])
                entries, total = self.page(kind, limit=50, offset=150)
                self.assertEqual((entries, total), ([], 120))

    def test_query_matches_content_and_keys_case_insensitively(self):
        """PERF-02: substring match over content and rule keys, casefolded."""
        for kind in self.owners:
            with self.subTest(kind=kind):
                a = self.add(kind, 'The Dragon sleeps')
                b = self.add(kind, 'plain', ['Wyvern', 'moon'])
                self.add(kind, 'unrelated', ['other'])
                for query, want in (('dragon', [a]), ('DRAGON', [a]), ('WYV', [b]), ('moon', [b])):
                    entries, total = self.page(kind, query)
                    self.assertEqual((self.refs(entries), total), (want, 1), query)
                # empty / None query returns everything
                for query in ('', None):
                    entries, total = self.page(kind, query)
                    self.assertEqual(total, 3)
                    self.assertEqual(entries, self.listing(kind))

    def test_query_uses_unicode_casefold(self):
        """PERF-02: 'ß' matches 'SS' and 'Ä' matches 'ä' like str.casefold()."""
        for kind in ('character', 'book'):
            with self.subTest(kind=kind):
                a = self.add(kind, 'Straße am Fluss')
                b = self.add(kind, 'x', ['Ärger'])
                self.add(kind, 'nothing here')
                self.assertEqual(self.refs(self.page(kind, 'STRASSE')[0]), [a])
                self.assertEqual(self.refs(self.page(kind, 'strasse')[0]), [a])
                self.assertEqual(self.refs(self.page(kind, 'ärger')[0]), [b])
                self.assertEqual(self.refs(self.page(kind, 'ÄRGER')[0]), [b])

    def test_query_ignores_json_field_names(self):
        """PERF-02: rule_json field names ('order', 'enabled', 'keys') are not searchable."""
        for kind in self.owners:
            with self.subTest(kind=kind):
                self.add(kind, 'alpha', ['beta'])
                for query in ('order', 'enabled', 'keys', 'constant', '{', '"'):
                    entries, total = self.page(kind, query)
                    self.assertEqual((entries, total), ([], 0), query)

    def test_like_wildcards_are_literal(self):
        """PERF-02: '%' and '_' (and backslash) in the query match literally."""
        for kind in self.owners:
            with self.subTest(kind=kind):
                pct = self.add(kind, '100% sure')
                und = self.add(kind, 'snake_case')
                bs = self.add(kind, r'path\to')
                self.add(kind, '100 sure')
                self.add(kind, 'snakeXcase')
                self.assertEqual(self.refs(self.page(kind, '%')[0]), [pct])
                self.assertEqual(self.refs(self.page(kind, '_')[0]), [und])
                self.assertEqual(self.refs(self.page(kind, 'e_c')[0]), [und])
                self.assertEqual(self.refs(self.page(kind, '\\')[0]), [bs])
                self.assertEqual(self.page(kind, '%%')[1], 0)

    def test_filtered_total_and_paging(self):
        """PERF-02: total is the filtered count; paging applies after filtering."""
        for i in range(70):
            self.add('channel', f'hit {i}')
            self.add('channel', f'miss {i}')
        full = [e for e in self.listing('channel') if 'hit' in e['content']]
        entries, total = self.page('channel', 'HIT', limit=50, offset=50)
        self.assertEqual(total, 70)
        self.assertEqual(entries, full[50:])
        self.assertEqual(self.page('channel', 'HIT', limit=50, offset=100), ([], 70))
        self.assertEqual(self.page('channel', 'nothing', limit=50, offset=0), ([], 0))

    def test_legacy_keys_json_fallback_is_searchable(self):
        """PERF-02: rows with empty rule_json match on keys from keys_json."""
        for kind, table in (('channel', 'lore'), ('guild', 'guild_lore_entries')):
            with self.subTest(kind=kind):
                g, owner = self.owners[kind]
                cols = 'guild_id,scope_kind,scope_id,content,keys_json,rule_json,entry_key'
                self.store.db.execute(
                    f'INSERT INTO {table}({cols}) VALUES(?,?,?,?,?,?,?)',
                    (g, kind, owner, 'legacy body', json.dumps(['Phoenix']), '{}', 'legacy:' + kind))
                self.store.db.commit()
                self.add(kind, 'modern', ['ember'])
                entries, total = self.page(kind, 'phoenix')
                self.assertEqual(total, 1)
                self.assertEqual(entries[0]['content'], 'legacy body')
                self.assertEqual(entries[0]['rule']['keys'], ['Phoenix'])
                self.assertEqual(self.page(kind, 'ember')[1], 1)
                self.assertEqual(self.page(kind, 'rule')[1], 0)

    def test_corrupt_rule_json_is_skipped_by_filter(self):
        """PERF-02: a row with undecodable rule_json is a non-match, not a SQLite error."""
        g, owner = self.owners['channel']
        self.store.db.execute(
            'INSERT INTO lore(guild_id,scope_kind,scope_id,content,keys_json,rule_json,entry_key) VALUES(?,?,?,?,?,?,?)',
            (g, 'channel', owner, 'broken body', '[]', 'not json', 'corrupt:1'))
        self.store.db.commit()
        good = self.add('channel', 'good ember', ['spark'])
        self.assertEqual(self.refs(self.page('channel', 'spark')[0]), [good])
        self.assertEqual(self.page('channel', 'nomatch'), ([], 0))

    def test_guild_isolation(self):
        """PERF-02: other guilds' entries are never listed or counted."""
        for kind in ('channel', 'guild', 'book'):
            self.add(kind, 'mine shared-word')
        self.store.save_entry(OTHER, 'channel', OTHER_CHANNEL, 'theirs shared-word', rule())
        self.store.save_entry(OTHER, 'guild', OTHER, 'theirs shared-word', rule())
        self.store.save_entry(OTHER, 'book', self.other_book, 'theirs shared-word', rule())
        for kind in ('channel', 'guild', 'book'):
            with self.subTest(kind=kind):
                entries, total = self.page(kind, 'shared-word')
                self.assertEqual(total, 1)
                self.assertEqual([e['content'] for e in entries], ['mine shared-word'])

    def test_foreign_owner_rejected(self):
        """PERF-02: an owner from another guild raises ValueError via validate_owner."""
        cases = (('channel', OTHER_CHANNEL), ('book', self.other_book), ('guild', OTHER),
                 ('space', self.other_world), ('character', 999999))
        for kind, owner in cases:
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    self.store.admin_entries_page(G, kind, owner, '', 50, 0)


if __name__ == '__main__':
    unittest.main()
