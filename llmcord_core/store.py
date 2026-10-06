from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS spaces (
 id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL, name TEXT NOT NULL,
 kind TEXT NOT NULL CHECK(kind IN ('world','hub')),
 UNIQUE(guild_id,name)
);
CREATE TABLE IF NOT EXISTS hub_worlds (
 hub_id INTEGER NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
 world_id INTEGER NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
 PRIMARY KEY(hub_id,world_id)
);
CREATE TABLE IF NOT EXISTS channels (
 channel_id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL,
 space_id INTEGER NOT NULL REFERENCES spaces(id),
 default_cast TEXT NOT NULL DEFAULT '[]', active_cast TEXT NOT NULL DEFAULT '[]',
 ambient INTEGER NOT NULL DEFAULT 0, ambient_count INTEGER NOT NULL DEFAULT 0,
 last_ambient REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS thread_casts (
 thread_id INTEGER PRIMARY KEY, cast TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ambient_activity (
 channel_id INTEGER PRIMARY KEY, message_count INTEGER NOT NULL DEFAULT 0,
 last_response REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS scene_resets (
 channel_id INTEGER PRIMARY KEY, reset_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS characters (
 id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL,
 world_id INTEGER NOT NULL REFERENCES spaces(id),
 name TEXT NOT NULL, card TEXT NOT NULL, avatar BLOB,
 UNIQUE(guild_id,name)
);
CREATE TABLE IF NOT EXISTS lore (
 id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL,
 scope_kind TEXT NOT NULL CHECK(scope_kind IN ('character','space','channel','thread')),
 scope_id INTEGER NOT NULL, content TEXT NOT NULL, keys_json TEXT NOT NULL DEFAULT '[]',
 constant INTEGER NOT NULL DEFAULT 0, insertion_order INTEGER NOT NULL DEFAULT 100,
 enabled INTEGER NOT NULL DEFAULT 1, source_message_id INTEGER,
 promoted_from INTEGER, pinned INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS nodes (
 message_id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL, channel_id INTEGER NOT NULL,
 parent_id INTEGER, root_id INTEGER NOT NULL, author_id INTEGER, character_id INTEGER,
 content TEXT NOT NULL, created_at REAL NOT NULL, context_json TEXT NOT NULL DEFAULT '[]',
 sources_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS nodes_channel_time ON nodes(channel_id,created_at);
CREATE INDEX IF NOT EXISTS nodes_root ON nodes(root_id);
CREATE TABLE IF NOT EXISTS summaries (
 node_id INTEGER PRIMARY KEY REFERENCES nodes(message_id) ON DELETE CASCADE,
 content TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS consent (
 guild_id INTEGER NOT NULL, user_id INTEGER NOT NULL, enabled INTEGER NOT NULL,
 PRIMARY KEY(guild_id,user_id)
);
CREATE TABLE IF NOT EXISTS personal_memories (
 id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL, user_id INTEGER NOT NULL,
 character_id INTEGER NOT NULL, content TEXT NOT NULL,
 source_message_id INTEGER NOT NULL,
 UNIQUE(guild_id,user_id,character_id,content)
);
CREATE TABLE IF NOT EXISTS encounters (
 id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL, character_id INTEGER NOT NULL,
 space_id INTEGER NOT NULL, content TEXT NOT NULL, source_message_id INTEGER NOT NULL,
 UNIQUE(guild_id,character_id,space_id,content)
);
CREATE TABLE IF NOT EXISTS candidates (
 id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL,
 scope_kind TEXT NOT NULL, scope_id INTEGER NOT NULL,
 content TEXT NOT NULL, evidence_count INTEGER NOT NULL DEFAULT 0,
 promoted INTEGER NOT NULL DEFAULT 0,
 UNIQUE(guild_id,scope_kind,scope_id,content)
);
CREATE TABLE IF NOT EXISTS evidence (
 candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
 root_id INTEGER NOT NULL, source_message_id INTEGER NOT NULL,
 PRIMARY KEY(candidate_id,root_id)
);
CREATE TABLE IF NOT EXISTS webhooks (
 channel_id INTEGER NOT NULL, character_id INTEGER NOT NULL,
 webhook_id INTEGER NOT NULL, PRIMARY KEY(channel_id,character_id)
);
CREATE TABLE IF NOT EXISTS trace (
 response_id INTEGER PRIMARY KEY, details_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lorebooks (
 id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL, name TEXT NOT NULL,
 target_kind TEXT NOT NULL CHECK(target_kind IN ('guild','channel')),
 target_id INTEGER NOT NULL DEFAULT 0, revision INTEGER NOT NULL DEFAULT 0,
 original_json TEXT NOT NULL DEFAULT '{}',
 UNIQUE(guild_id,name,target_kind,target_id)
);
CREATE TABLE IF NOT EXISTS lorebook_entries (
 id INTEGER PRIMARY KEY, book_id INTEGER NOT NULL REFERENCES lorebooks(id) ON DELETE CASCADE,
 uid TEXT NOT NULL, content TEXT NOT NULL, rule_json TEXT NOT NULL,
 source_hash TEXT NOT NULL, local_hash TEXT NOT NULL,
 UNIQUE(book_id,uid)
);
CREATE TABLE IF NOT EXISTS lorebook_space_links (
 book_id INTEGER NOT NULL REFERENCES lorebooks(id) ON DELETE CASCADE,
 space_id INTEGER NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
 PRIMARY KEY(book_id,space_id)
);
CREATE TABLE IF NOT EXISTS lore_activations (
 node_id INTEGER NOT NULL, entry_key TEXT NOT NULL,
 PRIMARY KEY(node_id,entry_key)
);
CREATE TABLE IF NOT EXISTS admin_audit (
 id INTEGER PRIMARY KEY, guild_id INTEGER NOT NULL, actor_id INTEGER NOT NULL,
 action TEXT NOT NULL, detail_json TEXT NOT NULL, created_at REAL NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        existing = str(path) != ":memory:" and Path(path).exists()
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("PRAGMA journal_mode=WAL")
        old_version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if existing and old_version < 2:
            backup = Path(str(path) + f".pre-v2-{int(time.time())}.sqlite3")
            with closing(sqlite3.connect(backup)) as target:
                self.db.backup(target)
        self.db.executescript(SCHEMA)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(characters)")}
        if "archived" not in columns:
            self.db.execute("ALTER TABLE characters ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(lore)")}
        if "rule_json" not in columns:
            self.db.execute("ALTER TABLE lore ADD COLUMN rule_json TEXT NOT NULL DEFAULT '{}'")
        self.db.execute("PRAGMA user_version=2")
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def one(self, sql: str, args: tuple = ()) -> sqlite3.Row | None:
        return self.db.execute(sql, args).fetchone()

    def all(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        return self.db.execute(sql, args).fetchall()

    def execute(self, sql: str, args: tuple = ()) -> int:
        with self.db:
            return self.db.execute(sql, args).lastrowid

    def create_space(self, guild_id: int, name: str, kind: str) -> int:
        if kind not in {"world", "hub"} or not name.strip():
            raise ValueError("Space needs a name and kind world or hub")
        return self.execute("INSERT INTO spaces(guild_id,name,kind) VALUES(?,?,?)", (guild_id, name.strip(), kind))

    def space(self, guild_id: int, name: str) -> sqlite3.Row | None:
        return self.one("SELECT * FROM spaces WHERE guild_id=? AND name=?", (guild_id, name))

    def space_by_id(self, space_id: int) -> sqlite3.Row | None:
        return self.one("SELECT * FROM spaces WHERE id=?", (space_id,))

    def list_spaces(self, guild_id: int) -> list[sqlite3.Row]:
        return self.all("SELECT * FROM spaces WHERE guild_id=? ORDER BY kind,name", (guild_id,))

    def link_world(self, guild_id: int, hub_id: int, world_id: int) -> None:
        hub, world = self.space_by_id(hub_id), self.space_by_id(world_id)
        if not hub or not world or hub["guild_id"] != guild_id or world["guild_id"] != guild_id or hub["kind"] != "hub" or world["kind"] != "world":
            raise ValueError("Choose a hub and world in this server")
        self.execute("INSERT OR IGNORE INTO hub_worlds VALUES(?,?)", (hub_id, world_id))

    def unlink_world(self, guild_id: int, hub_id: int, world_id: int) -> None:
        hub, world = self.space_by_id(hub_id), self.space_by_id(world_id)
        if not hub or not world or hub["guild_id"] != guild_id or world["guild_id"] != guild_id:
            raise ValueError("Choose a hub and world in this server")
        self.execute("DELETE FROM hub_worlds WHERE hub_id=? AND world_id=?", (hub_id, world_id))

    def allowed_worlds(self, hub_id: int) -> set[int]:
        return {row["world_id"] for row in self.all("SELECT world_id FROM hub_worlds WHERE hub_id=?", (hub_id,))}

    def bind_channel(self, guild_id: int, channel_id: int, space_id: int) -> None:
        space = self.space_by_id(space_id)
        if not space or space["guild_id"] != guild_id:
            raise ValueError("Space does not belong to this server")
        self.execute("INSERT INTO channels(channel_id,guild_id,space_id) VALUES(?,?,?) ON CONFLICT(channel_id) DO UPDATE SET space_id=excluded.space_id,default_cast='[]',active_cast='[]',ambient=0", (channel_id, guild_id, space_id))

    def channel(self, channel_id: int) -> sqlite3.Row | None:
        return self.one("SELECT * FROM channels WHERE channel_id=?", (channel_id,))

    def binding(self, channel_id: int, parent_id: int | None = None) -> sqlite3.Row | None:
        return self.channel(parent_id or channel_id)

    def set_ambient(self, channel_id: int, enabled: bool) -> None:
        self.execute("UPDATE channels SET ambient=? WHERE channel_id=?", (int(enabled), channel_id))
        self.execute("DELETE FROM ambient_activity WHERE channel_id=?", (channel_id,))

    def ambient_state(self, channel_id: int) -> tuple[int, float]:
        row = self.one("SELECT * FROM ambient_activity WHERE channel_id=?", (channel_id,))
        return (row["message_count"], row["last_response"]) if row else (0, 0.0)

    def count_ambient_message(self, channel_id: int) -> None:
        self.execute("INSERT INTO ambient_activity(channel_id,message_count) VALUES(?,1) ON CONFLICT(channel_id) DO UPDATE SET message_count=message_count+1", (channel_id,))

    def mark_ambient_response(self, channel_id: int, now: float | None = None) -> None:
        self.execute("INSERT INTO ambient_activity(channel_id,message_count,last_response) VALUES(?,0,?) ON CONFLICT(channel_id) DO UPDATE SET message_count=0,last_response=excluded.last_response", (channel_id, now or time.time()))

    def reset_scene(self, channel_id: int, now: float | None = None) -> None:
        self.execute("INSERT INTO scene_resets(channel_id,reset_at) VALUES(?,?) ON CONFLICT(channel_id) DO UPDATE SET reset_at=excluded.reset_at", (channel_id, now or time.time()))

    def scene_reset_at(self, channel_id: int) -> float:
        row = self.one("SELECT reset_at FROM scene_resets WHERE channel_id=?", (channel_id,))
        return row["reset_at"] if row else 0.0

    def add_character(self, guild_id: int, world_id: int, name: str, card: dict, avatar: bytes | None, entries: list[dict]) -> int:
        world = self.space_by_id(world_id)
        if not world or world["guild_id"] != guild_id or world["kind"] != "world":
            raise ValueError("Characters must have a home world in this server")
        previous = self.character(guild_id, name)
        moved = bool(previous and previous["world_id"] != world_id)
        with self.db:
            self.db.execute("INSERT INTO characters(guild_id,world_id,name,card,avatar) VALUES(?,?,?,?,?) ON CONFLICT(guild_id,name) DO UPDATE SET world_id=excluded.world_id,card=excluded.card,avatar=excluded.avatar", (guild_id, world_id, name, json.dumps(card), avatar))
            row = self.one("SELECT id FROM characters WHERE guild_id=? AND name=?", (guild_id, name))
            character_id = row["id"]
            self.db.execute("DELETE FROM lore WHERE scope_kind='character' AND scope_id=?", (character_id,))
            for entry in entries:
                self.db.execute("INSERT INTO lore(guild_id,scope_kind,scope_id,content,keys_json,constant,insertion_order,enabled,rule_json) VALUES(?,'character',?,?,?,?,?,?,?)", (guild_id, character_id, entry["content"], json.dumps(entry["keys"]), int(entry["constant"]), entry["insertion_order"], int(entry["enabled"]), json.dumps(entry.get("rule", {}))))
            self._prune_character_casts(guild_id, character_id)
            if moved:
                self._remove_from_thread_casts(character_id)
        return character_id

    def character(self, guild_id: int, name: str) -> sqlite3.Row | None:
        return self.one("SELECT * FROM characters WHERE guild_id=? AND name=?", (guild_id, name))

    def character_by_id(self, character_id: int) -> sqlite3.Row | None:
        return self.one("SELECT * FROM characters WHERE id=?", (character_id,))

    def eligible_characters(self, guild_id: int, space_id: int) -> list[sqlite3.Row]:
        space = self.space_by_id(space_id)
        if not space or space["guild_id"] != guild_id:
            return []
        worlds = {space_id} if space["kind"] == "world" else self.allowed_worlds(space_id)
        if not worlds:
            return []
        marks = ",".join("?" for _ in worlds)
        return self.all(f"SELECT * FROM characters WHERE guild_id=? AND archived=0 AND world_id IN ({marks}) ORDER BY name", (guild_id, *worlds))

    def get_cast(self, channel_id: int, parent_id: int | None = None) -> list[int]:
        if parent_id:
            thread = self.one('SELECT "cast" FROM thread_casts WHERE thread_id=?', (channel_id,))
            if thread:
                return json.loads(thread["cast"])
        row = self.binding(channel_id, parent_id)
        return json.loads(row["active_cast"]) if row else []

    def set_cast(self, channel_id: int, parent_id: int | None, cast: list[int], default: bool = False) -> None:
        cast = list(dict.fromkeys(cast))
        if len(cast) > 5:
            raise ValueError("A cast can have at most five characters")
        binding = self.binding(channel_id, parent_id)
        if not binding:
            raise ValueError("Channel is not bound to a space")
        eligible = {row["id"] for row in self.eligible_characters(binding["guild_id"], binding["space_id"])}
        if not set(cast) <= eligible:
            raise ValueError("Character is not eligible in this space")
        if parent_id and not default:
            self.execute("INSERT INTO thread_casts(thread_id,cast) VALUES(?,?) ON CONFLICT(thread_id) DO UPDATE SET cast=excluded.cast", (channel_id, json.dumps(cast)))
        else:
            col = "default_cast" if default else "active_cast"
            self.execute(f"UPDATE channels SET {col}=? WHERE channel_id=?", (json.dumps(cast), binding["channel_id"]))
            if default:
                self.execute("UPDATE channels SET active_cast=? WHERE channel_id=?", (json.dumps(cast), binding["channel_id"]))

    def add_lore(self, guild_id: int, scope_kind: str, scope_id: int, content: str, keys: list[str] | None = None, constant: bool = False, order: int = 100, source_id: int | None = None, promoted_from: int | None = None, pinned: bool = False) -> int:
        if scope_kind not in {"character", "space", "channel", "thread"} or not content.strip():
            raise ValueError("Invalid lore scope or empty content")
        return self.execute("INSERT INTO lore(guild_id,scope_kind,scope_id,content,keys_json,constant,insertion_order,source_message_id,promoted_from,pinned) VALUES(?,?,?,?,?,?,?,?,?,?)", (guild_id, scope_kind, scope_id, content.strip(), json.dumps(keys or []), int(constant), order, source_id, promoted_from, int(pinned)))

    def lore_row(self, guild_id: int, lore_id: int) -> sqlite3.Row | None:
        return self.one("SELECT * FROM lore WHERE guild_id=? AND id=?", (guild_id, lore_id))

    def list_lore(self, guild_id: int, scope_kind: str, scope_id: int) -> list[sqlite3.Row]:
        return self.all("SELECT * FROM lore WHERE guild_id=? AND scope_kind=? AND scope_id=? AND enabled=1 ORDER BY insertion_order,id", (guild_id, scope_kind, scope_id))

    def delete_lore(self, guild_id: int, lore_id: int) -> None:
        self.execute("DELETE FROM lore WHERE guild_id=? AND id=?", (guild_id, lore_id))

    def pin_lore(self, guild_id: int, lore_id: int) -> None:
        self.execute("UPDATE lore SET pinned=1 WHERE guild_id=? AND id=?", (guild_id, lore_id))

    def edit_lore(self, guild_id: int, lore_id: int, content: str) -> None:
        if not content.strip() or not self.lore_row(guild_id, lore_id):
            raise ValueError("Lore entry not found or content is empty")
        self.execute("UPDATE lore SET content=? WHERE guild_id=? AND id=?", (content.strip(), guild_id, lore_id))

    def promote_lore(self, guild_id: int, lore_id: int, scope_kind: str, scope_id: int) -> int:
        source = self.lore_row(guild_id, lore_id)
        if not source:
            raise ValueError("Lore entry not found")
        return self.add_lore(guild_id, scope_kind, scope_id, source["content"], json.loads(source["keys_json"]), bool(source["constant"]), source["insertion_order"], source["source_message_id"], lore_id, True)

    def record_node(self, message_id: int, guild_id: int, channel_id: int, parent_id: int | None, author_id: int | None, character_id: int | None, content: str, context: list[dict] | None = None, sources: list[int] | None = None, created_at: float | None = None) -> int:
        parent = self.node(parent_id) if parent_id else None
        if parent and (parent["guild_id"] != guild_id or parent["channel_id"] != channel_id):
            parent = None
        root_id = parent["root_id"] if parent else message_id
        self.execute("INSERT OR REPLACE INTO nodes(message_id,guild_id,channel_id,parent_id,root_id,author_id,character_id,content,created_at,context_json,sources_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)", (message_id, guild_id, channel_id, parent["message_id"] if parent else None, root_id, author_id, character_id, content, created_at or time.time(), json.dumps(context or []), json.dumps(sources or [])))
        return root_id

    def node(self, message_id: int | None) -> sqlite3.Row | None:
        return self.one("SELECT * FROM nodes WHERE message_id=?", (message_id,)) if message_id else None

    def ancestors(self, message_id: int, limit: int = 200) -> list[sqlite3.Row]:
        result, seen = [], set()
        node = self.node(message_id)
        while node and node["message_id"] not in seen and len(result) < limit:
            result.append(node)
            seen.add(node["message_id"])
            node = self.node(node["parent_id"])
        result.reverse()
        return result

    def latest_character_node(self, channel_id: int) -> sqlite3.Row | None:
        return self.one("SELECT * FROM nodes WHERE channel_id=? AND character_id IS NOT NULL ORDER BY created_at DESC,message_id DESC LIMIT 1", (channel_id,))

    def save_summary(self, node_id: int, content: str) -> None:
        self.execute("INSERT INTO summaries(node_id,content) VALUES(?,?) ON CONFLICT(node_id) DO UPDATE SET content=excluded.content", (node_id, content))

    def summary(self, node_id: int) -> str | None:
        row = self.one("SELECT content FROM summaries WHERE node_id=?", (node_id,))
        return row["content"] if row else None

    def set_consent(self, guild_id: int, user_id: int, enabled: bool) -> None:
        with self.db:
            self.db.execute("INSERT INTO consent VALUES(?,?,?) ON CONFLICT(guild_id,user_id) DO UPDATE SET enabled=excluded.enabled", (guild_id, user_id, int(enabled)))
            if not enabled:
                self.db.execute("DELETE FROM personal_memories WHERE guild_id=? AND user_id=?", (guild_id, user_id))

    def has_consent(self, guild_id: int, user_id: int) -> bool:
        row = self.one("SELECT enabled FROM consent WHERE guild_id=? AND user_id=?", (guild_id, user_id))
        return bool(row and row["enabled"])

    def add_personal(self, guild_id: int, user_id: int, character_id: int, content: str, source_id: int) -> None:
        if self.has_consent(guild_id, user_id):
            self.execute("INSERT OR IGNORE INTO personal_memories(guild_id,user_id,character_id,content,source_message_id) VALUES(?,?,?,?,?)", (guild_id, user_id, character_id, content, source_id))

    def personal(self, guild_id: int, user_id: int, character_id: int | None = None) -> list[sqlite3.Row]:
        if character_id is None:
            return self.all("SELECT * FROM personal_memories WHERE guild_id=? AND user_id=? ORDER BY id", (guild_id, user_id))
        return self.all("SELECT * FROM personal_memories WHERE guild_id=? AND user_id=? AND character_id=? ORDER BY id", (guild_id, user_id, character_id))

    def forget_personal(self, guild_id: int, user_id: int, memory_id: int) -> None:
        self.execute("DELETE FROM personal_memories WHERE guild_id=? AND user_id=? AND id=?", (guild_id, user_id, memory_id))

    def add_encounter(self, guild_id: int, character_id: int, space_id: int, content: str, source_id: int) -> None:
        self.execute("INSERT OR IGNORE INTO encounters(guild_id,character_id,space_id,content,source_message_id) VALUES(?,?,?,?,?)", (guild_id, character_id, space_id, content, source_id))

    def encounters(self, guild_id: int, character_id: int, space_id: int) -> list[sqlite3.Row]:
        return self.all("SELECT * FROM encounters WHERE guild_id=? AND character_id=? AND space_id=? ORDER BY id DESC LIMIT 12", (guild_id, character_id, space_id))

    def add_candidate(self, guild_id: int, scope_kind: str, scope_id: int, content: str, root_id: int, source_id: int) -> int:
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO candidates(guild_id,scope_kind,scope_id,content) VALUES(?,?,?,?)", (guild_id, scope_kind, scope_id, content))
            row = self.one("SELECT id,evidence_count,promoted FROM candidates WHERE guild_id=? AND scope_kind=? AND scope_id=? AND content=?", (guild_id, scope_kind, scope_id, content))
            inserted = self.db.execute("INSERT OR IGNORE INTO evidence(candidate_id,root_id,source_message_id) VALUES(?,?,?)", (row["id"], root_id, source_id)).rowcount
            if inserted:
                self.db.execute("UPDATE candidates SET evidence_count=evidence_count+1 WHERE id=?", (row["id"],))
            if row["evidence_count"] + inserted >= 2 and not row["promoted"]:
                self.db.execute("INSERT INTO lore(guild_id,scope_kind,scope_id,content,source_message_id,constant) VALUES(?,?,?,?,?,1)", (guild_id, scope_kind, scope_id, content, source_id))
                self.db.execute("UPDATE candidates SET promoted=1 WHERE id=?", (row["id"],))
            return row["id"]

    def save_trace(self, response_id: int, details: dict[str, Any]) -> None:
        self.execute("INSERT OR REPLACE INTO trace VALUES(?,?)", (response_id, json.dumps(details)))

    def trace(self, response_id: int) -> dict[str, Any] | None:
        row = self.one("SELECT details_json FROM trace WHERE response_id=?", (response_id,))
        return json.loads(row["details_json"]) if row else None

    def webhook_id(self, channel_id: int, character_id: int) -> int | None:
        row = self.one("SELECT webhook_id FROM webhooks WHERE channel_id=? AND character_id=?", (channel_id, character_id))
        return row["webhook_id"] if row else None

    def save_webhook_id(self, channel_id: int, character_id: int, webhook_id: int) -> None:
        self.execute("INSERT INTO webhooks VALUES(?,?,?) ON CONFLICT(channel_id,character_id) DO UPDATE SET webhook_id=excluded.webhook_id", (channel_id, character_id, webhook_id))

    def delete_scene(self, guild_id: int, root_id: int) -> None:
        with self.db:
            clause = "SELECT message_id FROM nodes WHERE guild_id=? AND root_id=?"
            self.db.execute(f"DELETE FROM evidence WHERE source_message_id IN ({clause})", (guild_id, root_id))
            self.db.execute(f"DELETE FROM trace WHERE response_id IN ({clause})", (guild_id, root_id))
            self.db.execute(f"DELETE FROM lore_activations WHERE node_id IN ({clause})", (guild_id, root_id))
            self.db.execute(f"DELETE FROM summaries WHERE node_id IN ({clause})", (guild_id, root_id))
            self.db.execute("DELETE FROM nodes WHERE guild_id=? AND root_id=?", (guild_id, root_id))
            self.db.execute("UPDATE candidates SET evidence_count=(SELECT COUNT(*) FROM evidence WHERE candidate_id=candidates.id)")
            self.db.execute("DELETE FROM candidates WHERE evidence_count=0 AND promoted=0")

    def expire_history(self, days: int, now: float | None = None) -> int:
        cutoff = (now or time.time()) - days * 86400
        with self.db:
            count = self.one("SELECT COUNT(*) AS n FROM nodes WHERE created_at<?", (cutoff,))["n"]
            clause = "SELECT message_id FROM nodes WHERE created_at<?"
            self.db.execute(f"DELETE FROM evidence WHERE source_message_id IN ({clause})", (cutoff,))
            self.db.execute(f"DELETE FROM trace WHERE response_id IN ({clause})", (cutoff,))
            self.db.execute(f"DELETE FROM lore_activations WHERE node_id IN ({clause})", (cutoff,))
            self.db.execute(f"DELETE FROM summaries WHERE node_id IN ({clause})", (cutoff,))
            self.db.execute("DELETE FROM nodes WHERE created_at<?", (cutoff,))
            self.db.execute("UPDATE candidates SET evidence_count=(SELECT COUNT(*) FROM evidence WHERE candidate_id=candidates.id)")
            self.db.execute("DELETE FROM candidates WHERE evidence_count=0 AND promoted=0")
            return count

    def create_lorebook(self, guild_id: int, name: str, target_kind: str,
                        target_id: int = 0) -> int:
        if target_kind not in {"guild", "channel"} or not name.strip():
            raise ValueError("Choose a guild or channel lorebook and a name")
        if target_kind == "channel":
            channel = self.channel(target_id)
            if not channel or channel["guild_id"] != guild_id:
                raise ValueError("Channel is not bound in this server")
        else:
            target_id = 0
        return self.execute("INSERT INTO lorebooks(guild_id,name,target_kind,target_id) VALUES(?,?,?,?)",
            (guild_id, name.strip(), target_kind, target_id))

    def lorebook(self, guild_id: int, book_id: int) -> sqlite3.Row | None:
        return self.one("SELECT * FROM lorebooks WHERE guild_id=? AND id=?", (guild_id, book_id))

    def list_lorebooks(self, guild_id: int) -> list[sqlite3.Row]:
        return self.all("SELECT * FROM lorebooks WHERE guild_id=? ORDER BY name,id", (guild_id,))

    def lorebook_entries(self, book_id: int) -> list[sqlite3.Row]:
        return self.all("SELECT * FROM lorebook_entries WHERE book_id=? ORDER BY id", (book_id,))

    def set_lorebook_space(self, guild_id: int, book_id: int, space_id: int, enabled: bool) -> None:
        book, space = self.lorebook(guild_id, book_id), self.space_by_id(space_id)
        if not book or book["target_kind"] != "guild" or not space or space["guild_id"] != guild_id:
            raise ValueError("Choose a guild book and space in this server")
        if enabled:
            self.execute("INSERT OR IGNORE INTO lorebook_space_links VALUES(?,?)", (book_id, space_id))
        else:
            self.execute("DELETE FROM lorebook_space_links WHERE book_id=? AND space_id=?", (book_id, space_id))

    def lorebook_links(self, book_id: int) -> list[int]:
        return [row["space_id"] for row in self.all(
            "SELECT space_id FROM lorebook_space_links WHERE book_id=?", (book_id,))]

    def active_lorebook_entries(self, guild_id: int, home_world_id: int,
                                space_id: int, channel_id: int) -> list[sqlite3.Row]:
        # A hub guest carries only books enabled for their own world, plus hub books.
        space_ids = (home_world_id, space_id) if home_world_id != space_id else (space_id,)
        marks = ",".join("?" for _ in space_ids)
        return self.all(f"""SELECT e.*, b.name AS book_name, b.target_kind, b.target_id,
                b.guild_id FROM lorebook_entries e JOIN lorebooks b ON b.id=e.book_id
                WHERE b.guild_id=? AND (
                  (b.target_kind='channel' AND b.target_id=?) OR
                  (b.target_kind='guild' AND b.id IN
                    (SELECT book_id FROM lorebook_space_links WHERE space_id IN ({marks}))))
                ORDER BY e.id""", (guild_id, channel_id, *space_ids))

    def preview_lorebook_sync(self, guild_id: int, book_id: int, imported) -> list[dict]:
        from .lorebooks import digest
        if not self.lorebook(guild_id, book_id):
            raise ValueError("Lorebook not found")
        current = {row["uid"]: row for row in self.lorebook_entries(book_id)}
        changes = []
        for uid, entry in imported.entries.items():
            row = current.pop(uid, None)
            if not row:
                status = "add"
            elif row["source_hash"] == entry.source_hash:
                status = "unchanged"
            elif digest({"content": row["content"], "rule": json.loads(row["rule_json"])}) != row["local_hash"]:
                status = "conflict"
            else:
                status = "update"
            changes.append({"uid": uid, "status": status, "warnings": entry.warnings,
                "before": row["content"] if row else "", "after": entry.content})
        for uid, row in current.items():
            local_changed = digest({"content": row["content"],
                "rule": json.loads(row["rule_json"])}) != row["local_hash"]
            changes.append({"uid": uid, "status": "conflict" if local_changed else "remove",
                "warnings": (), "before": row["content"], "after": ""})
        return changes

    def sync_lorebook(self, guild_id: int, book_id: int, imported,
                      resolutions: dict[str, str], expected_revision: int) -> list[dict]:
        changes = self.preview_lorebook_sync(guild_id, book_id, imported)
        book = self.lorebook(guild_id, book_id)
        if book["revision"] != expected_revision:
            raise ValueError("Lorebook changed since preview; preview again")
        for change in changes:
            if change["status"] == "conflict" and resolutions.get(change["uid"]) not in {"keep", "import"}:
                raise ValueError(f"Resolve conflict for entry {change['uid']}")
        with self.db:
            for change in changes:
                uid, status = change["uid"], change["status"]
                entry = imported.entries.get(uid)
                if status == "unchanged" or (status == "conflict" and resolutions.get(uid) == "keep"):
                    continue
                if entry is None:
                    self.db.execute("DELETE FROM lorebook_entries WHERE book_id=? AND uid=?", (book_id, uid))
                elif status == "add":
                    self.db.execute("""INSERT INTO lorebook_entries
                        (book_id,uid,content,rule_json,source_hash,local_hash)
                        VALUES(?,?,?,?,?,?)""", (book_id, uid, entry.content,
                        json.dumps(entry.rule, ensure_ascii=False), entry.source_hash, entry.local_hash))
                else:
                    self.db.execute("""UPDATE lorebook_entries SET content=?,rule_json=?,
                        source_hash=?,local_hash=? WHERE book_id=? AND uid=?""",
                        (entry.content, json.dumps(entry.rule, ensure_ascii=False),
                         entry.source_hash, entry.local_hash, book_id, uid))
            self.db.execute("UPDATE lorebooks SET original_json=?,revision=revision+1 WHERE id=?",
                (imported.raw_json, book_id))
        return changes

    def edit_lorebook_entry(self, guild_id: int, entry_id: int, content: str) -> None:
        row = self.one("""SELECT e.id FROM lorebook_entries e JOIN lorebooks b ON b.id=e.book_id
            WHERE e.id=? AND b.guild_id=?""", (entry_id, guild_id))
        if not row or not content.strip():
            raise ValueError("Lorebook entry not found or empty")
        self.execute("UPDATE lorebook_entries SET content=? WHERE id=?", (content.strip(), entry_id))

    def save_lore_activations(self, node_id: int, entry_keys: list[str]) -> None:
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO lore_activations VALUES(?,?)",
                [(node_id, key) for key in entry_keys])

    def branch_lore_activations(self, ancestor_ids: list[int]) -> list[sqlite3.Row]:
        if not ancestor_ids:
            return []
        marks = ",".join("?" for _ in ancestor_ids)
        return self.all(f"SELECT * FROM lore_activations WHERE node_id IN ({marks})", tuple(ancestor_ids))

    def audit(self, guild_id: int, actor_id: int, action: str, detail: dict) -> None:
        self.execute("INSERT INTO admin_audit(guild_id,actor_id,action,detail_json,created_at) VALUES(?,?,?,?,?)",
            (guild_id, actor_id, action, json.dumps(detail), time.time()))

    def archive_character(self, guild_id: int, character_id: int, archived: bool) -> None:
        with self.db:
            self.db.execute("UPDATE characters SET archived=? WHERE guild_id=? AND id=?",
                (int(archived), guild_id, character_id))
            if archived:
                self._prune_character_casts(guild_id, character_id)

    def cast_impact(self, guild_id: int, character_id: int, new_world_id: int) -> list[int]:
        impacted = []
        for binding in self.all("SELECT * FROM channels WHERE guild_id=?", (guild_id,)):
            cast = set(json.loads(binding["default_cast"]) + json.loads(binding["active_cast"]))
            if character_id not in cast:
                continue
            space = self.space_by_id(binding["space_id"])
            allowed = ({space["id"]} if space["kind"] == "world"
                       else self.allowed_worlds(space["id"]))
            if new_world_id not in allowed:
                impacted.append(binding["channel_id"])
        return impacted

    def _prune_character_casts(self, guild_id: int, character_id: int) -> None:
        character = self.character_by_id(character_id)
        for binding in self.all("SELECT * FROM channels WHERE guild_id=?", (guild_id,)):
            eligible = {row["id"] for row in self.eligible_characters(guild_id, binding["space_id"])}
            for field in ("default_cast", "active_cast"):
                cast = json.loads(binding[field])
                cleaned = [ident for ident in cast if ident != character_id or ident in eligible]
                if cleaned != cast:
                    self.db.execute(f"UPDATE channels SET {field}=? WHERE channel_id=?",
                        (json.dumps(cleaned), binding["channel_id"]))
        if not character or character["archived"]:
            self._remove_from_thread_casts(character_id)

    def _remove_from_thread_casts(self, character_id: int) -> None:
        rows = self.all('SELECT thread_id,"cast" FROM thread_casts')
        for row in rows:
            cast = json.loads(row["cast"])
            if character_id in cast:
                self.db.execute('UPDATE thread_casts SET "cast"=? WHERE thread_id=?',
                    (json.dumps([ident for ident in cast if ident != character_id]), row["thread_id"]))

    def update_character(self, guild_id: int, character_id: int, world_id: int,
                         name: str, card: dict) -> None:
        world = self.space_by_id(world_id)
        if not world or world["guild_id"] != guild_id or world["kind"] != "world":
            raise ValueError("Choose a home world in this server")
        if not name.strip():
            raise ValueError("Character name cannot be empty")
        previous = self.character_by_id(character_id)
        if not previous or previous["guild_id"] != guild_id:
            raise ValueError("Character not found in this server")
        moved = previous["world_id"] != world_id
        with self.db:
            self.db.execute("UPDATE characters SET world_id=?,name=?,card=? WHERE guild_id=? AND id=?",
                (world_id, name.strip(), json.dumps(card), guild_id, character_id))
            self._prune_character_casts(guild_id, character_id)
            if moved:
                self._remove_from_thread_casts(character_id)
