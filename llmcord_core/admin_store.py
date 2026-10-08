"""Versioned administration data, shared by the bot, HTTP routes and UI."""
from __future__ import annotations

import json
import uuid
import time
from contextlib import contextmanager

from .lorebooks import normalize_entry, parse_lorebook

ADMIN_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_usage (
 id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL,
 profile TEXT NOT NULL, model TEXT NOT NULL, role TEXT NOT NULL,
 input_tokens INTEGER, output_tokens INTEGER, cached_tokens INTEGER NOT NULL,
 reasoning_tokens INTEGER NOT NULL, cost_usd REAL, cost_basis TEXT NOT NULL,
 created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_guild_model_time ON model_usage(guild_id,profile,model,created_at);
CREATE TABLE IF NOT EXISTS scene_guidelines (
 guild_id INTEGER NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('space','channel')),
 owner_id INTEGER NOT NULL, content TEXT NOT NULL, revision INTEGER NOT NULL,
 PRIMARY KEY(guild_id,kind,owner_id)
);
CREATE TABLE IF NOT EXISTS guild_lore_entries (
 id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL,
 scope_kind TEXT NOT NULL DEFAULT 'guild' CHECK(scope_kind='guild'),
 scope_id INTEGER NOT NULL, content TEXT NOT NULL, keys_json TEXT NOT NULL DEFAULT '[]',
 constant INTEGER NOT NULL DEFAULT 0, insertion_order INTEGER NOT NULL DEFAULT 100,
 enabled INTEGER NOT NULL DEFAULT 1, source_message_id INTEGER, promoted_from INTEGER,
 pinned INTEGER NOT NULL DEFAULT 0, rule_json TEXT NOT NULL DEFAULT '{}',
 entry_key TEXT NOT NULL UNIQUE, revision INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS owner_revisions (
 guild_id INTEGER NOT NULL, kind TEXT NOT NULL, owner_id INTEGER NOT NULL,
 revision INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(guild_id,kind,owner_id)
);
CREATE TABLE IF NOT EXISTS import_entries (
 guild_id INTEGER NOT NULL, source_kind TEXT NOT NULL, source_id INTEGER NOT NULL,
 uid TEXT NOT NULL, entry_key TEXT NOT NULL, baseline_json TEXT NOT NULL,
 source_hash TEXT NOT NULL, disposition TEXT NOT NULL DEFAULT 'active',
 PRIMARY KEY(guild_id,source_kind,source_id,uid)
);
CREATE TABLE IF NOT EXISTS card_imports (
 character_id INTEGER PRIMARY KEY, baseline_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prompt_presets (
 id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL, name TEXT NOT NULL,
 UNIQUE(guild_id,name)
);
CREATE TABLE IF NOT EXISTS prompt_revisions (
 preset_id INTEGER NOT NULL REFERENCES prompt_presets(id) ON DELETE CASCADE,
 revision INTEGER NOT NULL, bundle_json TEXT NOT NULL, created_at REAL NOT NULL,
 PRIMARY KEY(preset_id,revision)
);
CREATE TABLE IF NOT EXISTS guild_settings (
 guild_id INTEGER PRIMARY KEY, preset_id INTEGER, preset_revision INTEGER,
 asset_channel_id INTEGER, usage_footer INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS avatar_slots (
 character_id INTEGER NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
 slot_key TEXT NOT NULL, label TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
 image BLOB, revision INTEGER NOT NULL DEFAULT 0,
 PRIMARY KEY(character_id,slot_key)
);
CREATE TABLE IF NOT EXISTS avatar_assets (
 id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL, character_id INTEGER NOT NULL,
 slot_key TEXT NOT NULL, image_hash TEXT NOT NULL, channel_id INTEGER NOT NULL,
 message_id INTEGER NOT NULL, url TEXT NOT NULL, created_at REAL NOT NULL
);
"""


def entry_rule(ident, row):
    rule = json.loads(row['rule_json'])
    if not rule:
        rule = normalize_entry(str(ident), {'keys': json.loads(row['keys_json']), 'constant': bool(row['constant']), 'enabled': bool(row['enabled']), 'order': row['insertion_order']}).rule
    return rule


def entry_match(ident, content, rule_json, keys_json, constant, enabled, order, needle):
    if needle in content.casefold():
        return 1
    try:
        rule = entry_rule(ident, {'rule_json': rule_json, 'keys_json': keys_json, 'constant': constant, 'enabled': enabled, 'insertion_order': order})
        return int(needle in (content + ' ' + ' '.join(rule['keys'])).casefold())
    except (ValueError, TypeError):
        return 0


class ConflictError(ValueError):
    """An optimistic revision or unresolved import conflict rejected a write."""


class AdminStore:
    def record_model_usage(self, usage):
        if usage.guild_id is None:
            return
        self.execute('INSERT INTO model_usage(guild_id,profile,model,role,input_tokens,output_tokens,cached_tokens,reasoning_tokens,cost_usd,cost_basis,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                     (usage.guild_id, usage.profile, usage.model, usage.role, usage.input_tokens, usage.output_tokens, usage.cached_tokens, usage.reasoning_tokens, usage.cost_usd, usage.cost_basis, usage.created_at))

    def model_usage_summary(self, guild_id, profile, model, since=None):
        since = time.time() - 86400 if since is None else since
        return dict(self.one('SELECT COUNT(*) AS requests,COALESCE(SUM(input_tokens),0) AS input_tokens,COALESCE(SUM(output_tokens),0) AS output_tokens,COALESCE(SUM(cost_usd),0) AS cost_usd,COALESCE(SUM(input_tokens IS NULL OR output_tokens IS NULL),0) AS unreported,COALESCE(SUM(cost_usd IS NULL),0) AS unpriced FROM model_usage WHERE guild_id=? AND profile=? AND model=? AND created_at>=?', (guild_id, profile, model, since)))

    def next_owner_id(self, kind, table):
        if (kind, table) not in {('space', 'spaces'), ('book', 'lorebooks')}:
            raise ValueError('Invalid owner table')
        return self.one(f'SELECT COALESCE(MAX(id),0)+1 AS next_id FROM (SELECT id FROM {table} UNION ALL SELECT owner_id FROM owner_revisions WHERE kind=?)', (kind,))['next_id']

    def guidelines(self, guild_id, kind, owner_id):
        if kind not in {'space', 'channel'}:
            raise ValueError('Choose a world, hub, or channel')
        self.validate_owner(guild_id, kind, owner_id)
        row = self.one('SELECT * FROM scene_guidelines WHERE guild_id=? AND kind=? AND owner_id=?', (guild_id, kind, owner_id))
        return dict(row) if row else {'guild_id': guild_id, 'kind': kind, 'owner_id': owner_id, 'content': '', 'revision': 0}

    def save_guidelines(self, guild_id, kind, owner_id, content, expected_revision):
        content = content.strip()
        if len(content.encode()) > 6000:
            raise ValueError('Guidelines exceed 6,000 bytes; shorten them before saving')
        with self.write_admin():
            current = self.guidelines(guild_id, kind, owner_id)
            if current['revision'] != expected_revision:
                raise ConflictError('Guidelines changed; reload before saving')
            self.db.execute('INSERT INTO scene_guidelines VALUES(?,?,?,?,?) ON CONFLICT(guild_id,kind,owner_id) DO UPDATE SET content=excluded.content,revision=excluded.revision', (guild_id, kind, owner_id, content, expected_revision + 1))
            self.bump_owner(guild_id, kind, owner_id)
        return True

    def scene_guidelines(self, guild_id, space_id, channel_id):
        result = {}
        for key, kind, ident in (('world_guidelines', 'space', space_id), ('channel_guidelines', 'channel', channel_id)):
            if kind == 'channel' and not self.channel(ident):
                continue
            row = self.guidelines(guild_id, kind, ident)
            if row['content']:
                result[key] = row
        return result

    def space_delete_impact(self, guild_id, space_id):
        self.validate_owner(guild_id, 'space', space_id)
        return {'characters': [row['name'] for row in self.all('SELECT name FROM characters WHERE guild_id=? AND world_id=?', (guild_id, space_id))],
                'channels': [row['channel_id'] for row in self.all('SELECT channel_id FROM channels WHERE guild_id=? AND space_id=?', (guild_id, space_id))],
                'entries': len(self.admin_entries(guild_id, 'space', space_id)),
                'revision': self.owner_revision(guild_id, 'space', space_id)}

    def delete_space(self, guild_id, space_id, expected_revision):
        with self.write_admin():
            impact = self.space_delete_impact(guild_id, space_id)
            if impact['revision'] != expected_revision:
                raise ConflictError('World or hub changed; reload before deleting')
            if impact['characters'] or impact['channels']:
                raise ValueError('Move or delete its home characters and rebind its channels before deleting this world or hub')
            for entry in self.admin_entries(guild_id, 'space', space_id):
                self._delete_entry_locked(guild_id, entry)
            self.db.execute("DELETE FROM candidates WHERE guild_id=? AND scope_kind='space' AND scope_id=?", (guild_id, space_id))
            self.db.execute('DELETE FROM encounters WHERE guild_id=? AND space_id=?', (guild_id, space_id))
            self.db.execute("DELETE FROM import_entries WHERE guild_id=? AND source_kind='space' AND source_id=?", (guild_id, space_id))
            self.db.execute("DELETE FROM scene_guidelines WHERE guild_id=? AND kind='space' AND owner_id=?", (guild_id, space_id))
            self.db.execute('DELETE FROM spaces WHERE guild_id=? AND id=?', (guild_id, space_id))
            self.bump_owner(guild_id, 'space', space_id)
        return True

    def delete_lorebook(self, guild_id, book_id, expected_revision):
        with self.write_admin():
            self.validate_owner(guild_id, 'book', book_id)
            if self.owner_revision(guild_id, 'book', book_id) != expected_revision:
                raise ConflictError('Lorebook changed; reload before deleting')
            for entry in self.admin_entries(guild_id, 'book', book_id):
                self._delete_entry_locked(guild_id, entry)
            revision = self.owner_revision(guild_id, 'book', book_id) + 1
            self.db.execute("DELETE FROM import_entries WHERE guild_id=? AND source_kind='book' AND source_id=?", (guild_id, book_id))
            self.db.execute('DELETE FROM lorebooks WHERE guild_id=? AND id=?', (guild_id, book_id))
            self.db.execute("INSERT INTO owner_revisions VALUES(?,'book',?,?) ON CONFLICT(guild_id,kind,owner_id) DO UPDATE SET revision=excluded.revision", (guild_id, book_id, revision))
        return True

    def preview_entry_import(self, guild_id, kind, owner_id, imported):
        def signature(content, rule, pinned):
            return json.dumps([content, rule, bool(pinned)], sort_keys=True, ensure_ascii=False)
        seen = {signature(row['content'], row['rule'], row['pinned']) for row in self.admin_entries(guild_id, kind, owner_id)}
        changes = []
        for uid, entry in imported.entries.items():
            pinned = entry.rule.get('original', {}).get('llmcord_pinned') is True
            key = signature(entry.content, entry.rule, pinned)
            changes.append({'uid': uid, 'status': 'duplicate' if key in seen else 'add', 'after': entry.content, 'warnings': entry.rule.get('unsupported', [])})
            seen.add(key)
        return changes

    def import_lore_entries(self, guild_id, kind, owner_id, imported, expected_revision):
        with self.write_admin():
            self.validate_owner(guild_id, kind, owner_id)
            if self.owner_revision(guild_id, kind, owner_id) != expected_revision:
                raise ConflictError('Destination lore changed; preview the import again')
            changes = self.preview_entry_import(guild_id, kind, owner_id, imported)
            refs = []
            for change in changes:
                if change['status'] == 'add':
                    entry = imported.entries[change['uid']]
                    refs.append(self.insert_admin_entry(guild_id, kind, owner_id, entry.content, entry.rule, entry.rule.get('original', {}).get('llmcord_pinned') is True))
            if refs:
                self.bump_owner(guild_id, kind, owner_id)
        return refs

    @contextmanager
    def write_admin(self):
        with self.admin_lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def migrate_admin(self):
        self.db.executescript(ADMIN_SCHEMA)
        for table, additions in {
            'lore': {'entry_key': "TEXT NOT NULL DEFAULT ''", 'revision': 'INTEGER NOT NULL DEFAULT 0'},
            'lorebook_entries': {'entry_key': "TEXT NOT NULL DEFAULT ''", 'revision': 'INTEGER NOT NULL DEFAULT 0', 'pinned': 'INTEGER NOT NULL DEFAULT 0', 'source_message_id': 'INTEGER', 'promoted_from': 'INTEGER'},
            'characters': {'avatar_manual': 'INTEGER NOT NULL DEFAULT 0'},
            'guild_settings': {'usage_footer': 'INTEGER NOT NULL DEFAULT 1'},
        }.items():
            columns = {row[1] for row in self.db.execute(f'PRAGMA table_info({table})')}
            for name, spec in additions.items():
                if name not in columns:
                    self.db.execute(f'ALTER TABLE {table} ADD COLUMN {name} {spec}')
        self.db.execute("UPDATE lore SET entry_key='lore:'||id WHERE entry_key=''")
        self.db.execute("UPDATE lorebook_entries SET entry_key='book:'||book_id||':'||uid WHERE entry_key=''")
        if self.one('PRAGMA user_version')[0] < 3:
            from .avatars import DEFAULT_SLOTS
            for character in self.all('SELECT id FROM characters'):
                for key in DEFAULT_SLOTS:
                    self.db.execute('INSERT OR IGNORE INTO avatar_slots(character_id,slot_key,label) VALUES(?,?,?)', (character['id'], key, key.title()))
        self.db.execute('CREATE UNIQUE INDEX IF NOT EXISTS lore_identity ON lore(entry_key) WHERE entry_key<>\'\'')
        self.db.execute('CREATE UNIQUE INDEX IF NOT EXISTS book_identity ON lorebook_entries(entry_key) WHERE entry_key<>\'\'')
        self.db.executescript("""CREATE TRIGGER IF NOT EXISTS lore_insert_identity AFTER INSERT ON lore WHEN new.entry_key='' BEGIN
            UPDATE lore SET entry_key='entry:'||lower(hex(randomblob(16))),revision=(random() & 4611686018427387903)+1 WHERE id=new.id;
        END;""")
        upgrading = self.one('PRAGMA user_version')[0] < 3
        for book in self.all('SELECT * FROM lorebooks') if upgrading else []:
            try:
                imported = parse_lorebook(book['original_json'].encode())
            except (ValueError, TypeError):
                continue
            for row in self.lorebook_entries(book['id']):
                source = imported.entries.get(row['uid'])
                if source:
                    self.db.execute('INSERT OR IGNORE INTO import_entries VALUES(?,?,?,?,?,?,?,?)',
                        (book['guild_id'], 'book', book['id'], row['uid'], row['entry_key'],
                         json.dumps({'content': source.content, 'rule': source.rule, 'pinned': False}), source.source_hash, 'active'))
        # Embedded card entries had no source IDs in v2. Match known source content
        # conservatively; unmatched rows stay independent rather than being deleted.
        for character in self.all('SELECT * FROM characters') if upgrading else []:
            if self.one("SELECT uid FROM import_entries WHERE source_kind='character' AND source_id=? LIMIT 1", (character['id'],)):
                continue
            card = json.loads(character['card'])
            entries = (card.get('character_book') or {}).get('entries', [])
            rows = self.all("SELECT * FROM lore WHERE guild_id=? AND scope_kind='character' AND scope_id=? ORDER BY id", (character['guild_id'], character['id']))
            for index, raw in enumerate(entries if isinstance(entries, list) else []):
                if not isinstance(raw, dict):
                    continue
                source = normalize_entry(str(raw.get('id', raw.get('uid', index))), raw)
                match = next((r for r in rows if r['content'] == source.content), None)
                if not match:
                    continue
                rows.remove(match)
                self.db.execute('INSERT OR IGNORE INTO import_entries VALUES(?,?,?,?,?,?,?,?)',
                    (character['guild_id'], 'character', character['id'], source.uid, match['entry_key'], json.dumps({'content': source.content, 'rule': source.rule, 'pinned': False}), source.source_hash, 'active'))

    def validate_owner(self, guild_id, kind, owner_id):
        if kind == 'guild':
            row = {'guild_id': guild_id} if owner_id == guild_id else None
        elif kind == 'book':
            row = self.lorebook(guild_id, owner_id)
        elif kind == 'character':
            row = self.character_by_id(owner_id)
        elif kind == 'space':
            row = self.space_by_id(owner_id)
        elif kind == 'channel':
            row = self.channel(owner_id)
        elif kind == 'thread':
            row = self.one("SELECT guild_id FROM lore WHERE guild_id=? AND scope_kind='thread' AND scope_id=? UNION SELECT guild_id FROM nodes WHERE guild_id=? AND channel_id=? LIMIT 1", (guild_id, owner_id, guild_id, owner_id))
        else:
            row = None
        if not row or row['guild_id'] != guild_id:
            raise ValueError('Choose an owner in this server')

    def owner_revision(self, guild_id, kind, owner_id):
        if kind == 'book':
            row = self.lorebook(guild_id, owner_id)
            return row['revision'] if row else -1
        row = self.one('SELECT revision FROM owner_revisions WHERE guild_id=? AND kind=? AND owner_id=?', (guild_id, kind, owner_id))
        return row['revision'] if row else 0

    def bump_owner(self, guild_id, kind, owner_id):
        if kind == 'book':
            self.db.execute('UPDATE lorebooks SET revision=revision+1 WHERE guild_id=? AND id=?', (guild_id, owner_id))
        else:
            self.db.execute('INSERT INTO owner_revisions VALUES(?,?,?,1) ON CONFLICT(guild_id,kind,owner_id) DO UPDATE SET revision=revision+1', (guild_id, kind, owner_id))

    def admin_entry(self, guild_id, ref):
        try:
            kind, raw_id = ref.split(':', 1)
            ident = int(raw_id)
        except (AttributeError, ValueError):
            raise ValueError('Invalid entry reference') from None
        if kind == 'lore':
            row = self.one('SELECT * FROM lore WHERE guild_id=? AND id=?', (guild_id, ident))
        elif kind == 'guild-lore':
            row = self.one('SELECT * FROM guild_lore_entries WHERE guild_id=? AND id=?', (guild_id, ident))
        elif kind == 'book-entry':
            row = self.one('SELECT e.*,b.guild_id FROM lorebook_entries e JOIN lorebooks b ON b.id=e.book_id WHERE b.guild_id=? AND e.id=?', (guild_id, ident))
        else:
            row = None
        if not row:
            raise ValueError('Lore entry not found')
        return self._entry_dict(kind, ref, ident, row)

    def _entry_dict(self, kind, ref, ident, row):
        rule = entry_rule(ident, row)
        return {'ref': ref, 'id': ident, 'table': {'lore': 'lore', 'guild-lore': 'guild_lore_entries', 'book-entry': 'lorebook_entries'}[kind],
                'entry_key': row['entry_key'] or f'lore:{ident}', 'revision': row['revision'],
                'owner_kind': row['scope_kind'] if kind != 'book-entry' else 'book',
                'owner_id': row['scope_id'] if kind != 'book-entry' else row['book_id'],
                'content': row['content'], 'rule': rule, 'pinned': bool(row['pinned']),
                'source_message_id': row['source_message_id'], 'promoted_from': row['promoted_from'],
                'uid': row['uid'] if kind == 'book-entry' else None}

    def entry_by_key(self, guild_id, key):
        row = self.one('SELECT id FROM lore WHERE guild_id=? AND entry_key=?', (guild_id, key))
        if row:
            return self.admin_entry(guild_id, f"lore:{row['id']}")
        row = self.one('SELECT id FROM guild_lore_entries WHERE guild_id=? AND entry_key=?', (guild_id, key))
        if row:
            return self.admin_entry(guild_id, f"guild-lore:{row['id']}")
        row = self.one('SELECT e.id FROM lorebook_entries e JOIN lorebooks b ON b.id=e.book_id WHERE b.guild_id=? AND e.entry_key=?', (guild_id, key))
        return self.admin_entry(guild_id, f"book-entry:{row['id']}") if row else None

    def _entry_query(self, guild_id, kind, owner_id):
        if kind == 'book':
            return 'book-entry', 'FROM lorebook_entries e JOIN lorebooks b ON b.id=e.book_id WHERE b.guild_id=? AND e.book_id=?', (guild_id, owner_id), 'e.id', 'e.*'
        if kind == 'guild':
            return 'guild-lore', 'FROM guild_lore_entries WHERE guild_id=?', (guild_id,), 'insertion_order,id', '*'
        return 'lore', 'FROM lore WHERE guild_id=? AND scope_kind=? AND scope_id=?', (guild_id, kind, owner_id), 'insertion_order,id', '*'

    def admin_entries(self, guild_id, kind, owner_id):
        self.validate_owner(guild_id, kind, owner_id)
        prefix, where, params, order, cols = self._entry_query(guild_id, kind, owner_id)
        return [self._entry_dict(prefix, f"{prefix}:{r['id']}", r['id'], r) for r in self.all(f'SELECT {cols} {where} ORDER BY {order}', params)]

    def admin_entries_page(self, guild_id, kind, owner_id, query='', limit=50, offset=0):
        self.validate_owner(guild_id, kind, owner_id)
        prefix, where, params, order, cols = self._entry_query(guild_id, kind, owner_id)
        col = (lambda c: f'e.{c}' if c in ('id', 'content', 'rule_json') else 'NULL') if kind == 'book' else (lambda c: c)
        if query:
            where += f" AND llmcord_entry_match({col('id')},{col('content')},{col('rule_json')},{col('keys_json')},{col('constant')},{col('enabled')},{col('insertion_order')},?)"
            params = params + (query.casefold(),)
        if query:
            rows = self.all(f'SELECT {cols},COUNT(*) OVER () AS llmcord_total {where} ORDER BY {order} LIMIT ? OFFSET ?', params + (limit, offset))
            total = rows[0]['llmcord_total'] if rows else 0 if offset <= 0 else self.one(f'SELECT COUNT(*) AS n {where}', params)['n']
        else:
            total = self.one(f'SELECT COUNT(*) AS n {where}', params)['n']
            rows = self.all(f'SELECT {cols} {where} ORDER BY {order} LIMIT ? OFFSET ?', params + (limit, offset))
        return [self._entry_dict(prefix, f"{prefix}:{r['id']}", r['id'], r) for r in rows], total

    def insert_admin_entry(self, guild_id, kind, owner_id, content, rule, pinned=False, key=None, source_id=None, promoted_from=None):
        key = key or 'entry:' + uuid.uuid4().hex
        if kind == 'book':
            uid = 'local-' + uuid.uuid4().hex
            ident = self.db.execute('INSERT INTO lorebook_entries(book_id,uid,content,rule_json,source_hash,local_hash,entry_key,pinned) VALUES(?,?,?,?,?,?,?,?)',
                (owner_id, uid, content, json.dumps(rule), '', '', key, int(pinned))).lastrowid
            self.db.execute('UPDATE lorebook_entries SET source_message_id=?,promoted_from=?,revision=? WHERE id=?', (source_id, promoted_from, time.time_ns(), ident))
            return f'book-entry:{ident}'
        table = 'guild_lore_entries' if kind == 'guild' else 'lore'
        ident = self.db.execute(f'INSERT INTO {table}(guild_id,scope_kind,scope_id,content,keys_json,constant,insertion_order,enabled,rule_json,pinned,entry_key) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
            (guild_id, kind, owner_id, content, json.dumps(rule['keys']), int(rule['constant']), rule['order'], int(rule['enabled']), json.dumps(rule), int(pinned), key)).lastrowid
        self.db.execute(f'UPDATE {table} SET source_message_id=?,promoted_from=?,revision=? WHERE id=?', (source_id, promoted_from, time.time_ns(), ident))
        return f"{'guild-lore' if kind == 'guild' else 'lore'}:{ident}"

    def save_entry(self, guild_id, kind, owner_id, content, rule, pinned=False, ref=None, expected_revision=None):
        from .lorebooks import validate_rule
        self.validate_owner(guild_id, kind, owner_id)
        rule = validate_rule(rule, content)
        if not content.strip():
            raise ValueError('Lore content cannot be empty')
        with self.write_admin():
            if ref:
                old = self.admin_entry(guild_id, ref)
                if (old['owner_kind'], old['owner_id']) != (kind, owner_id):
                    raise ConflictError('Entry moved; reload the editor')
                if expected_revision is None or old['revision'] != expected_revision:
                    raise ConflictError('Entry changed; reload the editor')
                self.db.execute(f"UPDATE {old['table']} SET content=?,rule_json=?,pinned=?,revision=revision+1 WHERE id=?", (content.strip(), json.dumps(rule), int(pinned), old['id']))
                if old['table'] != 'lorebook_entries':
                    self.db.execute(f"UPDATE {old['table']} SET keys_json=?,constant=?,insertion_order=?,enabled=? WHERE id=?", (json.dumps(rule['keys']), int(rule['constant']), rule['order'], int(rule['enabled']), old['id']))
            else:
                ref = self.insert_admin_entry(guild_id, kind, owner_id, content.strip(), rule, pinned)
            self.bump_owner(guild_id, kind, owner_id)
        return ref

    def delete_entry(self, guild_id, ref, expected_revision):
        with self.write_admin():
            row = self.admin_entry(guild_id, ref)
            if row['revision'] != expected_revision:
                raise ConflictError('Entry changed; reload before deleting')
            self._delete_entry_locked(guild_id, row)

    def _delete_entry_locked(self, guild_id, row):
        self.db.execute("UPDATE import_entries SET disposition='deleted' WHERE guild_id=? AND entry_key=?", (guild_id, row['entry_key']))
        self.db.execute(f"DELETE FROM {row['table']} WHERE id=?", (row['id'],))
        self.bump_owner(guild_id, row['owner_kind'], row['owner_id'])

    def _selected_entries_locked(self, guild_id, entries):
        if not entries or len(entries) > 2000:
            raise ValueError('Select between 1 and 2000 lore entries')
        rows, seen = [], set()
        for entry in entries:
            row = self.entry_by_key(guild_id, entry['entry_key'])
            if not row or row['entry_key'] in seen or row['revision'] != entry['revision'] or (row['owner_kind'], row['owner_id']) != (entry['owner_kind'], entry['owner_id']):
                raise ConflictError('A selected lore entry changed; select it again before continuing')
            seen.add(row['entry_key'])
            rows.append(row)
        return rows

    def delete_entries(self, guild_id, entries):
        with self.write_admin():
            rows = self._selected_entries_locked(guild_id, entries)
            for row in rows:
                self._delete_entry_locked(guild_id, row)
        return len(rows)

    def transfer_entries(self, guild_id, entries, kind, owner_id):
        # Appending independent entries does not require the destination's
        # snapshot revision. Validate each source inside the same transaction.
        with self.write_admin():
            self.validate_owner(guild_id, kind, owner_id)
            rows = self._selected_entries_locked(guild_id, entries)
            return [self._transfer_entry_locked(guild_id, row, kind, owner_id, False) for row in rows]

    def transfer_entry(self, guild_id, ref, kind, owner_id, expected_revision, copy=False, target_revision=None):
        self.validate_owner(guild_id, kind, owner_id)
        with self.write_admin():
            row = self.admin_entry(guild_id, ref)
            if row['revision'] != expected_revision or (target_revision is not None and self.owner_revision(guild_id, kind, owner_id) != target_revision):
                raise ConflictError('Lore changed; reload before transferring')
            return self._transfer_entry_locked(guild_id, row, kind, owner_id, copy)

    def _transfer_entry_locked(self, guild_id, row, kind, owner_id, copy):
        ref = row['ref']
        if not copy and (kind, owner_id) == (row['owner_kind'], row['owner_id']):
            return ref
        target_table = 'lorebook_entries' if kind == 'book' else 'guild_lore_entries' if kind == 'guild' else 'lore'
        same_table = target_table == row['table']
        new_ref = self.insert_admin_entry(guild_id, kind, owner_id, row['content'], row['rule'], row['pinned'], None if copy else row['entry_key'], row['source_message_id'], row['promoted_from']) if copy or not same_table else ref
        if not copy:
            if new_ref == ref:
                if kind == 'book':
                    self.db.execute('UPDATE lorebook_entries SET book_id=?,uid=?,revision=revision+1 WHERE id=?', (owner_id, 'local-' + uuid.uuid4().hex, row['id']))
                else:
                    self.db.execute(f"UPDATE {row['table']} SET scope_kind=?,scope_id=?,revision=revision+1 WHERE id=?", (kind, owner_id, row['id']))
            else:
                self.db.execute(f"DELETE FROM {row['table']} WHERE id=?", (row['id'],))
            self.db.execute("UPDATE import_entries SET disposition='moved' WHERE guild_id=? AND entry_key=?", (guild_id, row['entry_key']))
            self.bump_owner(guild_id, row['owner_kind'], row['owner_id'])
        self.bump_owner(guild_id, kind, owner_id)
        return new_ref

    def preview_import(self, guild_id, kind, owner_id, imported):
        self.validate_owner(guild_id, kind, owner_id)
        tracked = {r['uid']: r for r in self.all('SELECT * FROM import_entries WHERE guild_id=? AND source_kind=? AND source_id=?', (guild_id, kind, owner_id))}
        changes = []
        for uid in sorted(set(tracked) | set(imported.entries)):
            source, incoming = tracked.get(uid), imported.entries.get(uid)
            current = self.import_current(guild_id, kind, owner_id, uid, source)
            effective = {'content': current['content'], 'rule': current['rule'], 'pinned': current['pinned']} if current else None
            changed = source and (source['disposition'] != 'active' or effective != json.loads(source['baseline_json']))
            status = ('conflict' if not source and current else 'add' if not source else 'unchanged' if (incoming and source['source_hash'] == incoming.source_hash) or (not incoming and source['source_hash'] == '') else 'conflict' if changed else 'update' if incoming else 'remove')
            changes.append({'uid': uid, 'status': status, 'before': current['content'] if current else '', 'after': incoming.content if incoming else '', 'warnings': incoming.warnings if incoming else (), 'disposition': source['disposition'] if source else 'manual' if current else 'new'})
        return changes

    def import_current(self, guild_id, kind, owner_id, uid, origin):
        if origin:
            return self.entry_by_key(guild_id, origin['entry_key'])
        if kind == 'book':
            row = self.one('SELECT id FROM lorebook_entries WHERE book_id=? AND uid=?', (owner_id, uid))
            if row:
                return self.admin_entry(guild_id, f"book-entry:{row['id']}")
        row = self.entry_by_key(guild_id, uid)
        return row if row and (row['owner_kind'], row['owner_id']) == (kind, owner_id) else None

    def apply_import_locked(self, guild_id, kind, owner_id, imported, resolutions):
        changes = self.preview_import(guild_id, kind, owner_id, imported)
        for change in changes:
            uid, status = change['uid'], change['status']
            if status == 'conflict' and resolutions.get(uid) not in {'keep', 'import'}:
                raise ConflictError(f'Resolve conflict for entry {uid}')
        for change in changes:
            uid, status = change['uid'], change['status']
            if status == 'unchanged':
                continue
            incoming = imported.entries.get(uid)
            incoming_pinned = bool(incoming and incoming.rule.get('original', {}).get('llmcord_pinned') is True)
            origin = self.one('SELECT * FROM import_entries WHERE guild_id=? AND source_kind=? AND source_id=? AND uid=?', (guild_id, kind, owner_id, uid))
            current = self.import_current(guild_id, kind, owner_id, uid, origin)
            if status == 'conflict' and resolutions.get(uid) == 'keep':
                # Acknowledge the new source baseline without losing the local disposition.
                if incoming:
                    self.db.execute('UPDATE import_entries SET baseline_json=?,source_hash=? WHERE guild_id=? AND source_kind=? AND source_id=? AND uid=?', (json.dumps({'content': incoming.content, 'rule': incoming.rule, 'pinned': incoming_pinned}), incoming.source_hash, guild_id, kind, owner_id, uid))
                else:
                    self.db.execute("UPDATE import_entries SET source_hash='' WHERE guild_id=? AND source_kind=? AND source_id=? AND uid=?", (guild_id, kind, owner_id, uid))
                continue
            same_owner = bool(current and (current['owner_kind'], current['owner_id']) == (kind, owner_id))
            if current and (not incoming or not same_owner):
                self.db.execute(f"DELETE FROM {current['table']} WHERE id=?", (current['id'],))
                if (current['owner_kind'], current['owner_id']) != (kind, owner_id):
                    self.bump_owner(guild_id, current['owner_kind'], current['owner_id'])
            if incoming:
                key = origin['entry_key'] if origin else current['entry_key'] if current else 'entry:' + uuid.uuid4().hex
                if same_owner:
                    ref = current['ref']
                    self.db.execute(f"UPDATE {current['table']} SET content=?,rule_json=?,pinned=?,revision=revision+1 WHERE id=?", (incoming.content, json.dumps(incoming.rule), int(incoming_pinned), current['id']))
                    if kind != 'book':
                        self.db.execute(f"UPDATE {current['table']} SET keys_json=?,constant=?,enabled=?,insertion_order=? WHERE id=?", (json.dumps(incoming.rule['keys']), int(incoming.rule['constant']), int(incoming.rule['enabled']), incoming.rule['order'], current['id']))
                else:
                    ref = self.insert_admin_entry(guild_id, kind, owner_id, incoming.content, incoming.rule, pinned=incoming_pinned, key=key)
                if kind == 'book':
                    ident = int(ref.split(':')[1])
                    self.db.execute('UPDATE lorebook_entries SET uid=?,source_hash=?,local_hash=? WHERE id=?', (uid, incoming.source_hash, incoming.local_hash, ident))
                self.db.execute('INSERT INTO import_entries VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(guild_id,source_kind,source_id,uid) DO UPDATE SET baseline_json=excluded.baseline_json,source_hash=excluded.source_hash,disposition=excluded.disposition',
                    (guild_id, kind, owner_id, uid, key, json.dumps({'content': incoming.content, 'rule': incoming.rule, 'pinned': incoming_pinned}), incoming.source_hash, 'active'))
            elif origin:
                self.db.execute("UPDATE import_entries SET disposition='deleted',source_hash='' WHERE guild_id=? AND source_kind=? AND source_id=? AND uid=?", (guild_id, kind, owner_id, uid))
        self.bump_owner(guild_id, kind, owner_id)
        return changes

    def sync_admin_book(self, guild_id, book_id, imported, resolutions, expected_revision):
        with self.write_admin():
            if self.owner_revision(guild_id, 'book', book_id) != expected_revision:
                raise ConflictError('Lorebook changed since preview; preview again')
            changes = self.apply_import_locked(guild_id, 'book', book_id, imported, resolutions)
            self.db.execute('UPDATE lorebooks SET original_json=? WHERE id=?', (imported.raw_json, book_id))
        return changes

    def export_lore(self, guild_id, kind, owner_id):
        from .lorebooks import export_entry
        entries = {}
        for row in self.admin_entries(guild_id, kind, owner_id):
            origin = self.one('SELECT uid FROM import_entries WHERE guild_id=? AND source_kind=? AND source_id=? AND entry_key=?', (guild_id, kind, owner_id, row['entry_key']))
            uid = row['uid'] or (origin['uid'] if origin else row['entry_key'])
            entries[uid] = export_entry(row['content'], row['rule'])
            entries[uid]['llmcord_pinned'] = row['pinned']
        return {'entries': entries}

    @staticmethod
    def card_book(card):
        from .lorebooks import ImportedBook
        entries = {}
        for index, item in enumerate(card.entries):
            rule = item.get('rule') or normalize_entry(str(index), item).rule
            uid = str(item.get('uid', index))
            raw = {**rule.get('original', {}), 'content': item['content']}
            normalized = normalize_entry(uid, raw)
            entries[uid] = normalized
        return ImportedBook('{}', entries)

    def preview_card(self, guild_id, world_id, card):
        world = self.space_by_id(world_id)
        if not world or world['guild_id'] != guild_id or world['kind'] != 'world':
            raise ValueError('Choose a home world in this server')
        existing = self.character(guild_id, card.name)
        if not existing:
            return {'character_id': None, 'revision': 0, 'conflicts': [], 'changes': []}
        baseline = self.one('SELECT baseline_json FROM card_imports WHERE character_id=?', (existing['id'],))
        current = json.loads(existing['card'])
        old = json.loads(baseline['baseline_json']) if baseline else None
        text = lambda value: {k: v for k, v in value.items() if k != 'character_book'}
        conflicts = []
        if text(current) != text(card.data) and (old is None or text(current) != text(old)):
            conflicts.append('card')
        if card.avatar and existing['avatar'] and bytes(existing['avatar']) != card.avatar and existing['avatar_manual']:
            conflicts.append('avatar')
        return {'character_id': existing['id'], 'revision': self.owner_revision(guild_id, 'character', existing['id']), 'conflicts': conflicts,
                'changes': self.preview_import(guild_id, 'character', existing['id'], self.card_book(card))}

    def _create_character_locked(self, guild_id, world_id, name, card, avatar=None):
        # Deleted characters retain an owner-revision tombstone. Never reuse an
        # ID: historical replies and a bot turn in progress may still reference it.
        ident = self.one("SELECT COALESCE(MAX(id),0)+1 AS next_id FROM (SELECT id FROM characters UNION ALL SELECT owner_id FROM owner_revisions WHERE kind='character' UNION ALL SELECT character_id FROM nodes WHERE character_id IS NOT NULL)")['next_id']
        self.db.execute('INSERT INTO characters(id,guild_id,world_id,name,card,avatar) VALUES(?,?,?,?,?,?)',
                        (ident, guild_id, world_id, name, json.dumps(card), avatar))
        from .avatars import DEFAULT_SLOTS
        for key in DEFAULT_SLOTS:
            self.db.execute('INSERT INTO avatar_slots(character_id,slot_key,label) VALUES(?,?,?)', (ident, key, key.title()))
        return ident

    def create_character(self, guild_id, world_id, name):
        name = name.strip()
        if not name:
            raise ValueError('Character name cannot be empty')
        with self.write_admin():
            world = self.space_by_id(world_id)
            if not world or world['guild_id'] != guild_id or world['kind'] != 'world':
                raise ValueError('Choose a home world in this server')
            if self.character(guild_id, name):
                raise ValueError('This character name already exists')
            card = {'name': name, **{field: '' for field in ('description', 'personality', 'scenario', 'first_mes', 'mes_example', 'system_prompt', 'post_history_instructions')}}
            ident = self._create_character_locked(guild_id, world_id, name, card)
            self.bump_owner(guild_id, 'character', ident)
        return ident

    def delete_character(self, guild_id, character_id, expected_revision):
        with self.write_admin():
            self.validate_owner(guild_id, 'character', character_id)
            if self.owner_revision(guild_id, 'character', character_id) != expected_revision:
                raise ConflictError('Character changed; reload before deleting')
            for entry in self.admin_entries(guild_id, 'character', character_id):
                self._delete_entry_locked(guild_id, entry)
            self.db.execute("DELETE FROM candidates WHERE guild_id=? AND scope_kind='character' AND scope_id=?", (guild_id, character_id))
            for table in ('personal_memories', 'encounters', 'avatar_assets'):
                self.db.execute(f'DELETE FROM {table} WHERE guild_id=? AND character_id=?', (guild_id, character_id))
            self.db.execute('DELETE FROM webhooks WHERE character_id=?', (character_id,))
            self.db.execute('DELETE FROM card_imports WHERE character_id=?', (character_id,))
            self.db.execute("DELETE FROM import_entries WHERE guild_id=? AND source_kind='character' AND source_id=?", (guild_id, character_id))
            self.db.execute('DELETE FROM characters WHERE guild_id=? AND id=?', (guild_id, character_id))
            self._prune_character_casts(guild_id, character_id)
            self.bump_owner(guild_id, 'character', character_id)
        return True

    def apply_card(self, guild_id, world_id, card, resolutions=None, expected_revision=None):
        resolutions = resolutions or {}
        with self.write_admin():
            preview = self.preview_card(guild_id, world_id, card)
            existing = self.character(guild_id, card.name)
            if existing and expected_revision is not None and preview['revision'] != expected_revision:
                raise ConflictError('Character changed since preview; upload again')
            for conflict in preview['conflicts']:
                if resolutions.get(conflict) not in {'keep', 'import'}:
                    raise ConflictError(f'Resolve {conflict} conflict in the dashboard')
            data = json.loads(existing['card']) if existing and resolutions.get('card') == 'keep' else card.data
            avatar = existing['avatar'] if existing else None
            manual = existing['avatar_manual'] if existing else 0
            if card.avatar and (not manual or resolutions.get('avatar') == 'import'):
                avatar, manual = card.avatar, 0
            if existing:
                ident = existing['id']
                self.db.execute('UPDATE characters SET world_id=?,card=?,avatar=?,avatar_manual=? WHERE id=?',
                                (world_id, json.dumps(data), avatar, manual, ident))
            else:
                ident = self._create_character_locked(guild_id, world_id, card.name, data, avatar)
            self.apply_import_locked(guild_id, 'character', ident, self.card_book(card), resolutions)
            self.db.execute('INSERT INTO card_imports VALUES(?,?) ON CONFLICT(character_id) DO UPDATE SET baseline_json=excluded.baseline_json', (ident, json.dumps(card.data)))
            self._prune_character_casts(guild_id, ident)
            if existing and existing['world_id'] != world_id:
                self._remove_from_thread_casts(guild_id, ident)
        return ident

    def list_presets(self, guild_id):
        return self.all('SELECT p.*,MAX(r.revision) AS revision FROM prompt_presets p JOIN prompt_revisions r ON r.preset_id=p.id WHERE p.guild_id=? GROUP BY p.id ORDER BY p.name', (guild_id,))

    def preset_bundle(self, guild_id, preset_id, revision=None):
        from .prompts import default_bundle
        if preset_id == 0:
            return default_bundle()
        row = self.one('SELECT r.* FROM prompt_revisions r JOIN prompt_presets p ON p.id=r.preset_id WHERE p.guild_id=? AND p.id=? AND (? IS NULL OR r.revision=?) ORDER BY r.revision DESC LIMIT 1', (guild_id, preset_id, revision, revision))
        if not row:
            raise ValueError('Preset revision not found in this server')
        return json.loads(row['bundle_json'])

    def save_preset(self, guild_id, name, bundle, preset_id=None, expected_revision=None):
        from .prompts import validate_bundle
        bundle = validate_bundle(bundle)
        if not name.strip() or len(name) > 80:
            raise ValueError('Preset name must be 1–80 characters')
        with self.write_admin():
            if preset_id:
                rows = self.list_presets(guild_id)
                current = next((r for r in rows if r['id'] == preset_id), None)
                if not current:
                    raise ValueError('Preset not found in this server')
                if expected_revision != current['revision']:
                    raise ConflictError('Preset changed; reload before saving')
                revision = current['revision'] + 1
                self.db.execute('UPDATE prompt_presets SET name=? WHERE id=?', (name.strip(), preset_id))
            else:
                preset_id = self.db.execute('INSERT INTO prompt_presets(guild_id,name) VALUES(?,?)', (guild_id, name.strip())).lastrowid
                revision = 1
            self.db.execute('INSERT INTO prompt_revisions VALUES(?,?,?,?)', (preset_id, revision, json.dumps(bundle), time.time()))
        return preset_id, revision

    def activate_preset(self, guild_id, preset_id, revision, providers):
        from .prompts import compatibility
        bundle = self.preset_bundle(guild_id, preset_id, revision)
        problems = compatibility(bundle, providers)
        if problems:
            raise ValueError('Resolve preset compatibility: ' + '; '.join(problems))
        with self.write_admin():
            self.db.execute('INSERT INTO guild_settings(guild_id,preset_id,preset_revision) VALUES(?,?,?) ON CONFLICT(guild_id) DO UPDATE SET preset_id=excluded.preset_id,preset_revision=excluded.preset_revision', (guild_id, preset_id, revision))

    def usage_footer_enabled(self, guild_id):
        row = self.one('SELECT usage_footer FROM guild_settings WHERE guild_id=?', (guild_id,))
        return bool(row['usage_footer']) if row else True

    def set_usage_footer(self, guild_id, enabled):
        with self.write_admin():
            self.db.execute('INSERT INTO guild_settings(guild_id,usage_footer) VALUES(?,?) ON CONFLICT(guild_id) DO UPDATE SET usage_footer=excluded.usage_footer', (guild_id, int(bool(enabled))))

    def active_preset(self, guild_id):
        row = self.one('SELECT * FROM guild_settings WHERE guild_id=?', (guild_id,))
        ident, revision = (row['preset_id'] or 0, row['preset_revision'] or 0) if row else (0, 0)
        return {'id': ident, 'revision': revision, 'bundle': self.preset_bundle(guild_id, ident, revision)}

    def delete_preset(self, guild_id, preset_id):
        with self.write_admin():
            if self.one('SELECT guild_id FROM guild_settings WHERE guild_id=? AND preset_id=?', (guild_id, preset_id)):
                raise ValueError('Select another preset before deleting the active preset')
            self.db.execute('DELETE FROM prompt_presets WHERE guild_id=? AND id=?', (guild_id, preset_id))

    def avatar_slots(self, guild_id, character_id):
        self.validate_owner(guild_id, 'character', character_id)
        stored = {r['slot_key']: self._slot_meta(dict(r), keep_image=False) for r in self.all('SELECT * FROM avatar_slots WHERE character_id=?', (character_id,))}
        stored.setdefault('neutral', self._slot_meta(self._neutral_slot(character_id), keep_image=False))
        return list(stored.values())

    @staticmethod
    def _neutral_slot(character_id):
        return {'character_id': character_id, 'slot_key': 'neutral', 'label': 'Neutral', 'description': '', 'image': None, 'revision': 0}

    @staticmethod
    def _slot_meta(row, keep_image):
        import hashlib
        image = row['image']
        digest = hashlib.sha256(image).hexdigest() if image else None
        if not keep_image:
            del row['image']
        return {**row, 'has_image': bool(image), 'image_hash': digest, 'image_version': digest[:12] if digest else None}

    def save_static_avatar(self, guild_id, character_id, image, expected_revision):
        from .avatars import normalize_avatar
        image = normalize_avatar(image) if image is not None else None
        with self.write_admin():
            self.validate_owner(guild_id, 'character', character_id)
            if self.owner_revision(guild_id, 'character', character_id) != expected_revision:
                raise ConflictError('Character changed; reload before saving the fallback avatar')
            self.db.execute('UPDATE characters SET avatar=?,avatar_manual=1 WHERE guild_id=? AND id=?', (image, guild_id, character_id))
            self.bump_owner(guild_id, 'character', character_id)

    def avatar_slot(self, guild_id, character_id, key):
        self.validate_owner(guild_id, 'character', character_id)
        row = self.one('SELECT * FROM avatar_slots WHERE character_id=? AND slot_key=?', (character_id, key))
        if row:
            return self._slot_meta(dict(row), keep_image=True)
        if key == 'neutral':
            return self._slot_meta(self._neutral_slot(character_id), keep_image=True)
        raise ValueError('Avatar slot not found')

    def _slot_revision(self, guild_id, character_id, key):
        self.validate_owner(guild_id, 'character', character_id)
        row = self.one('SELECT revision FROM avatar_slots WHERE character_id=? AND slot_key=?', (character_id, key))
        if row:
            return row['revision']
        if key == 'neutral':
            return 0
        raise ValueError('Avatar slot not found')

    def save_avatar(self, guild_id, character_id, key, label, description, image=None, expected_revision=None):
        import re
        self.validate_owner(guild_id, 'character', character_id)
        if not re.fullmatch(r'[a-z0-9_-]{1,40}', key) or not label.strip() or len(label) > 80 or len(description) > 500:
            raise ValueError('Use a stable lowercase key, a label up to 80 characters, and a description up to 500 characters')
        with self.write_admin():
            current = self.one('SELECT image,revision FROM avatar_slots WHERE character_id=? AND slot_key=?', (character_id, key)) or ({'image': None, 'revision': 0} if key == 'neutral' else None)
            if current and expected_revision is not None and current['revision'] != expected_revision:
                raise ConflictError('Avatar slot changed; reload')
            blob = image if image is not None else current['image'] if current else None
            self.db.execute('INSERT INTO avatar_slots VALUES(?,?,?,?,?,1) ON CONFLICT(character_id,slot_key) DO UPDATE SET label=excluded.label,description=excluded.description,image=excluded.image,revision=revision+1',
                (character_id, key, label.strip(), description, blob))
            self.bump_owner(guild_id, 'character', character_id)

    def clear_avatar_image(self, guild_id, character_id, key, expected_revision):
        with self.write_admin():
            if self._slot_revision(guild_id, character_id, key) != expected_revision:
                raise ConflictError('Avatar slot changed; reload')
            self.db.execute('UPDATE avatar_slots SET image=NULL,revision=revision+1 WHERE character_id=? AND slot_key=?', (character_id, key))
            self.bump_owner(guild_id, 'character', character_id)

    def delete_avatar(self, guild_id, character_id, key, expected_revision):
        if key == 'neutral':
            raise ValueError('The neutral slot is required')
        with self.write_admin():
            if self._slot_revision(guild_id, character_id, key) != expected_revision:
                raise ConflictError('Avatar slot changed; reload')
            self.db.execute('DELETE FROM avatar_slots WHERE character_id=? AND slot_key=?', (character_id, key))
            self.bump_owner(guild_id, 'character', character_id)

    def avatar_asset(self, guild_id, character_id, key, image_hash=None):
        # With an explicit image_hash the caller must already have validated the owner (the query is still guild-filtered).
        if image_hash is None:
            image_hash = self.avatar_slot(guild_id, character_id, key)['image_hash']
        if not image_hash:
            return None
        return self.one('SELECT * FROM avatar_assets WHERE guild_id=? AND character_id=? AND slot_key=? AND image_hash=? ORDER BY id DESC LIMIT 1',
            (guild_id, character_id, key, image_hash))

    def usable_avatars(self, guild_id, character_id):
        result = []
        for row in self.avatar_slots(guild_id, character_id):
            asset = self.avatar_asset(guild_id, character_id, row['slot_key'], row['image_hash'] or '')
            if row['slot_key'] == 'neutral' or asset:
                result.append({**row, 'asset_id': asset['id'] if asset else None, 'url': asset['url'] if asset else None})
        return result
