import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from fastapi.testclient import TestClient

from llmcord_core.lore import lore_scopes
from llmcord_core.lorebooks import parse_lorebook
from llmcord_core.store import Store
from llmcord_core.web import create_app
from llmcord_core.world_info import evaluate


def payload(entries):
    return json.dumps({"entries": entries}).encode()


class LorebookTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.a = self.store.create_space(1, "A", "world")
        self.b = self.store.create_space(1, "B", "world")
        self.h = self.store.create_space(1, "Hub", "hub")
        self.store.link_world(1, self.h, self.a)
        self.store.link_world(1, self.h, self.b)
        for channel, space in ((100, self.a), (200, self.b), (300, self.h)):
            self.store.bind_channel(1, channel, space)
        self.alice = self.store.add_character(1, self.a, "Alice", {"name": "Alice"}, None, [])
        self.bob = self.store.add_character(1, self.b, "Bob", {"name": "Bob"}, None, [])

    def tearDown(self):
        self.store.close()

    def test_parse_formats_and_reject_bad_entries(self):
        book = parse_lorebook(payload({"0": {"key": ["moon"], "content": "Red moon",
            "position": 4, "role": "user", "depth": 1}}))
        self.assertEqual(book.entries["0"].rule["position"], "in_chat")
        self.assertEqual(book.entries["0"].rule["role"], "user")
        self.assertEqual(len(parse_lorebook(b'[{"uid": 7, "key": ["x"], "content": "X"}]').entries), 1)
        with self.assertRaises(ValueError):
            parse_lorebook(b'[null]')

    def test_book_isolation_hub_channel_and_conflict(self):
        book_a = self.store.create_lorebook(1, "A book", "guild")
        book_b = self.store.create_lorebook(1, "B book", "guild")
        book_h = self.store.create_lorebook(1, "Hub book", "guild")
        book_c = self.store.create_lorebook(1, "Cafe", "channel", 300)
        for book, space in ((book_a, self.a), (book_b, self.b), (book_h, self.h)):
            self.store.set_lorebook_space(1, book, space, True)
        for book, text in ((book_a, "A secret"), (book_b, "B secret"),
                           (book_h, "Hub event"), (book_c, "Cafe menu")):
            self.store.sync_lorebook(1, book, parse_lorebook(payload({"1": {
                "key": ["tell"], "content": text}})), {}, self.store.lorebook(1, book)['revision'])
        scopes = lore_scopes(self.store, self.alice, self.h, 300)
        found = evaluate(self.store, 1, scopes, "tell", 1000,
            character=self.store.character_by_id(self.alice))
        self.assertEqual({item.content for item in found}, {"A secret", "Hub event", "Cafe menu"})
        scopes = lore_scopes(self.store, self.bob, self.h, 300)
        found = evaluate(self.store, 1, scopes, "tell", 1000,
            character=self.store.character_by_id(self.bob))
        self.assertEqual({item.content for item in found}, {"B secret", "Hub event", "Cafe menu"})
        entry = self.store.lorebook_entries(book_a)[0]
        self.store.edit_lorebook_entry(1, entry["id"], "Locally edited")
        updated = parse_lorebook(payload({"1": {"key": ["tell"], "content": "New import"}}))
        self.assertEqual(self.store.preview_lorebook_sync(1, book_a, updated)[0]["status"], "conflict")
        with self.assertRaises(ValueError):
            self.store.sync_lorebook(1, book_a, updated, {}, self.store.lorebook(1, book_a)['revision'])
        self.store.sync_lorebook(1, book_a, updated, {"1": "keep"}, self.store.lorebook(1, book_a)["revision"])
        self.assertEqual(self.store.lorebook_entries(book_a)[0]["content"], "Locally edited")

    def test_rules_and_branch_activation(self):
        book = self.store.create_lorebook(1, "Rules", "channel", 100)
        entries = {"regex": {"key": ["/moon/i"], "content": "regex hit"},
            "secondary": {"key": ["moon"], "keysecondary": ["red"], "selective": True,
                          "content": "red only"},
            "constant": {"constant": True, "content": "always"},
            "recursion": {"key": ["regex hit"], "content": "recursive"},
            "unsupported": {"constant": True, "position": 7, "content": "outlet"},
            "cool": {"key": ["moon"], "cooldown": 2, "content": "timed"}}
        self.store.sync_lorebook(1, book, parse_lorebook(payload(entries)), {}, 0)
        character = self.store.character_by_id(self.alice)
        scopes = lore_scopes(self.store, self.alice, self.a, 100)
        first = evaluate(self.store, 1, scopes, "MOON", 1000, character=character,
            response_id=10)
        content = {item.content for item in first}
        self.assertTrue({"regex hit", "always", "recursive", "timed"} <= content)
        self.assertNotIn("outlet", content)
        self.assertNotIn("red only", content)
        self.store.record_node(10, 1, 100, None, 5, None, "MOON")
        self.store.save_lore_activations(10, [item.entry_key for item in first])
        second = evaluate(self.store, 1, scopes, "MOON", 1000, character=character,
            branch_ids=[10], response_id=11)
        self.assertNotIn("timed", {item.content for item in second})
        rewind = evaluate(self.store, 1, scopes, "MOON", 1000, character=character,
            branch_ids=[], response_id=11)
        self.assertIn("timed", {item.content for item in rewind})

    def test_moving_character_prunes_ineligible_casts(self):
        self.store.set_cast(100, None, [self.alice], default=True)
        self.store.set_cast(300, None, [self.alice], default=True)
        self.assertEqual(self.store.cast_impact(1, self.alice, self.b), [100])
        self.store.update_character(1, self.alice, self.b, "Alice", {"name": "Alice"})
        self.assertEqual(self.store.get_cast(100), [])
        self.assertEqual(self.store.get_cast(300), [self.alice])

    def test_versioned_migration_creates_backup(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            database = Path(directory) / "old.sqlite3"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE legacy_marker(value TEXT)")
                connection.execute("INSERT INTO legacy_marker VALUES('before migration')")
                connection.execute("PRAGMA user_version=1")
            connection.close()
            store = Store(database)
            self.assertEqual(store.one("PRAGMA user_version")[0], 6)
            store.close()
            backups = list(Path(directory).glob("old.sqlite3.pre-v3-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            with sqlite3.connect(backups[0]) as original:
                self.assertEqual(original.execute("SELECT value FROM legacy_marker").fetchone()[0],
                    "before migration")
            original.close()


class WebTests(unittest.TestCase):
    def test_oauth_admin_csrf_and_logout(self):
        """SEC-02 (migrated from the legacy-route walk-through): OAuth state check, guild admin scoping on an avatar GET,
        csrf-guarded logout. The legacy form routes this test used to post to were retired."""
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            async def discord_api(request):
                path = request.url.path
                if path == "/api/v10/oauth2/token":
                    return httpx.Response(200, json={"access_token": "a", "refresh_token": "r", "expires_in": 3600})
                if path == "/api/v10/users/@me":
                    return httpx.Response(200, json={"id": "4", "username": "admin"})
                if path == "/api/v10/users/@me/guilds":
                    return httpx.Response(200, json=[{"id": "1", "name": "Server", "permissions": "8"}])
                if path == "/api/v10/guilds/1/channels":
                    return httpx.Response(200, json=[{"id": "100", "name": "chat", "type": 0}])
                return httpx.Response(404)
            http = httpx.AsyncClient(transport=httpx.MockTransport(discord_api))
            app = create_app(Path(directory) / "web.sqlite3", "https://pi.test", "client", "secret", "bot", http, enable_dashboard=False)
            with TestClient(app, base_url="https://pi.test") as client:
                self.assertEqual(client.get("/guild/1/characters/1/avatar").status_code, 401)
                login = client.get("/login", follow_redirects=False)
                state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
                self.assertEqual(client.get("/auth/callback", params={"code": "x", "state": "wrong"}).status_code, 403)
                self.assertEqual(client.get("/auth/callback", params={"code": "x", "state": state}, follow_redirects=False).status_code, 303)
                csrf = next(iter(app.state.sessions.values()))["csrf"]
                self.assertEqual(client.get("/guild/2/characters/1/avatar").status_code, 403)
                self.assertEqual(client.get("/guild/1/characters/1/avatar").status_code, 404)  # admin of guild 1, no such character
                self.assertEqual(client.post("/logout", data={"csrf": "wrong"}, follow_redirects=False).status_code, 403)
                self.assertTrue(app.state.sessions)
                self.assertEqual(client.post("/logout", data={"csrf": csrf}, follow_redirects=False).status_code, 303)
                self.assertFalse(app.state.sessions)
                self.assertEqual(client.get("/guild/1/characters/1/avatar").status_code, 401)


if __name__ == "__main__":
    unittest.main()
