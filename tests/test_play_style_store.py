"""FEAT-32 part B: schema v19 (characters.play_style), the guild-scoped store API and the audited admin operation."""
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import httpx
from fastapi import HTTPException
from helpers import discord_transport, install_session

from llmcord_core.admin_store import ConflictError
from llmcord_core.games import playstyle
from llmcord_core.games.playstyle import PlayStyle
from llmcord_core.store import Store
from llmcord_core.web import create_app

G, H = 1, 2
BOLD = PlayStyle(looseness=80, risk=20, judgment=10)


def columns(db):
    return {row[1] for row in db.execute("PRAGMA table_info(characters)")}


class V18UpgradeTests(unittest.TestCase):
    def v18_file(self, folder):
        path = Path(folder) / "v18.sqlite3"
        store = Store(path)
        world = store.create_space(G, "W", "world")
        store.create_character(G, world, "Ann")
        store.create_character(G, world, "Bo")
        store.close()
        with closing(sqlite3.connect(path)) as db, db:
            if "play_style" in columns(db):
                db.execute("ALTER TABLE characters DROP COLUMN play_style")
            db.execute("PRAGMA user_version=18")
        return path

    def test_v18_file_upgrades_with_a_backup_and_keeps_every_character(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.v18_file(folder)
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 25)
                self.assertIn("play_style", columns(store.db))
                rows = store.all("SELECT name, play_style FROM characters ORDER BY name")
                self.assertEqual([tuple(r) for r in rows], [("Ann", ""), ("Bo", "")])
            finally:
                store.close()
            backups = list(Path(folder).glob("v18.sqlite3.pre-v19-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            with closing(sqlite3.connect(backups[0])) as old:
                self.assertEqual(old.execute("PRAGMA user_version").fetchone()[0], 18)
                self.assertNotIn("play_style", columns(old))

    def test_reopening_is_idempotent_and_does_not_back_up_again(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.v18_file(folder)
            Store(path).close()
            store = Store(path)
            store.close()
            self.assertEqual(len(list(Path(folder).glob("*.pre-v19-*.sqlite3"))), 1)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM characters").fetchone()[0], 2)

    def test_existing_column_at_version_18_is_tolerated(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "half.sqlite3"
            Store(path).close()
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("PRAGMA user_version=18")
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 25)
                self.assertIn("play_style", columns(store.db))
            finally:
                store.close()

    def test_fresh_database_is_current_version_with_the_column(self):
        store = Store()
        try:
            self.assertEqual(store.one("PRAGMA user_version")[0], 25)
            self.assertIn("play_style", columns(store.db))
        finally:
            store.close()


class StyleCase(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)
        self.w = self.store.create_space(G, "W", "world")
        self.ann = self.store.create_character(G, self.w, "Ann")
        self.w2 = self.store.create_space(H, "W2", "world")
        self.zed = self.store.create_character(H, self.w2, "Zed")

    def revision(self, guild=G, cid=None):
        return self.store.owner_revision(guild, "character", cid or self.ann)


class StoreApiTests(StyleCase):
    def test_unset_reads_none_and_resolves_to_the_derived_style(self):
        self.assertIsNone(self.store.character_play_style(G, self.ann))
        self.assertEqual(self.store.resolved_play_style(G, self.ann), playstyle.derived_from_id(self.ann))

    def test_set_get_clear_round_trip(self):
        before = self.revision()
        after = self.store.set_character_play_style(G, self.ann, BOLD, before)
        self.assertEqual(after, self.revision())
        self.assertGreater(after, before)
        self.assertEqual(self.store.character_play_style(G, self.ann), BOLD)
        self.assertEqual(self.store.resolved_play_style(G, self.ann), BOLD)
        self.assertEqual(self.store.one("SELECT play_style FROM characters WHERE id=?", (self.ann,))[0], playstyle.dumps(BOLD))
        cleared = self.store.set_character_play_style(G, self.ann, None, after)
        self.assertGreater(cleared, after)
        self.assertIsNone(self.store.character_play_style(G, self.ann))
        self.assertEqual(self.store.one("SELECT play_style FROM characters WHERE id=?", (self.ann,))[0], "")
        self.assertEqual(self.store.resolved_play_style(G, self.ann), playstyle.derived_from_id(self.ann))

    def test_invalid_stored_text_reads_as_unset(self):
        for text in ("not json", '{"looseness": 101}', '{"nope": 1}', "[1]"):
            self.store.db.execute("UPDATE characters SET play_style=? WHERE id=?", (text, self.ann))
            self.store.db.commit()
            self.assertIsNone(self.store.character_play_style(G, self.ann), text)
            self.assertEqual(self.store.resolved_play_style(G, self.ann), playstyle.derived_from_id(self.ann))

    def test_stale_revision_conflicts_and_writes_nothing(self):
        stale = self.revision()
        self.store.set_character_play_style(G, self.ann, BOLD, stale)
        current = self.revision()
        with self.assertRaises(ConflictError):
            self.store.set_character_play_style(G, self.ann, PlayStyle(), stale)
        self.assertEqual(self.store.character_play_style(G, self.ann), BOLD)
        self.assertEqual(self.revision(), current)

    def test_non_playstyle_input_is_rejected_without_a_write(self):
        before = self.revision()
        for bad in ({"looseness": 80}, "x", 5):
            with self.assertRaises((TypeError, ValueError)):
                self.store.set_character_play_style(G, self.ann, bad, before)
        self.assertIsNone(self.store.character_play_style(G, self.ann))
        self.assertEqual(self.revision(), before)

    def test_unvalidated_style_values_are_rejected_without_a_write(self):
        before = self.revision()
        for bad in (PlayStyle(looseness=101), PlayStyle(looseness=-1), PlayStyle(risk="x"),
                    PlayStyle(risk=50.0), PlayStyle(risk=True)):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.store.set_character_play_style(G, self.ann, bad, before)
        self.assertIsNone(self.store.character_play_style(G, self.ann))
        self.assertEqual(self.store.one("SELECT play_style FROM characters WHERE id=?", (self.ann,))[0], "")
        self.assertEqual(self.revision(), before)

    def test_guild_isolation(self):
        self.store.set_character_play_style(H, self.zed, BOLD, self.revision(H, self.zed))
        before = self.revision(H, self.zed)
        self.assertIsNone(self.store.character_play_style(G, self.zed))
        with self.assertRaises(ValueError):
            self.store.set_character_play_style(G, self.zed, PlayStyle(), before)
        self.assertEqual(self.store.character_play_style(H, self.zed), BOLD)
        self.assertEqual(self.revision(H, self.zed), before)

    def test_missing_character_reads_none_and_refuses_a_write(self):
        self.assertIsNone(self.store.character_play_style(G, 9999))
        with self.assertRaises(ValueError):
            self.store.set_character_play_style(G, 9999, BOLD, 0)

    def test_update_character_keeps_the_style(self):
        self.store.set_character_play_style(G, self.ann, BOLD, self.revision())
        self.store.update_character(G, self.ann, self.w, "Annie", {"name": "Annie"}, expected_revision=self.revision())
        self.assertEqual(self.store.character_play_style(G, self.ann), BOLD)

    def test_card_json_never_carries_the_style(self):
        self.store.set_character_play_style(G, self.ann, BOLD, self.revision())
        card = self.store.character_by_id(self.ann)["card"]
        self.assertNotIn("play_style", card)
        self.assertNotIn("looseness", card)
        self.assertEqual(json.loads(card)["name"], "Ann")

    def test_deleting_the_character_removes_the_style(self):
        self.store.set_character_play_style(G, self.ann, BOLD, self.revision())
        self.store.delete_character(G, self.ann, self.revision())
        self.assertIsNone(self.store.character_play_style(G, self.ann))
        self.assertIsNone(self.store.one("SELECT 1 FROM characters WHERE id=?", (self.ann,)))


def detail_text(row):
    detail = json.loads(row["detail_json"])
    return detail["changes"]


class AdminServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        transport, _ = discord_transport(admin_guilds=[1])
        self.app = create_app(":memory:", "https://pi.test", "client", "secret", "bot",
                              httpx.AsyncClient(transport=transport), enable_dashboard=False,
                              config_path="tests/nonexistent-config.yaml")
        self.service, self.store = self.app.state.admin, self.app.state.store
        self.session = install_session(self.app, user_id="4")
        w = self.store.create_space(G, "W", "world")
        self.ann = self.store.create_character(G, w, "Ann")
        w2 = self.store.create_space(H, "W2", "world")
        self.zed = self.store.create_character(H, w2, "Zed")

    async def asyncTearDown(self):
        await self.app.state.http.aclose()
        self.store.close()

    def audits(self):
        return [r for r in self.store.all("SELECT * FROM admin_audit ORDER BY id") if r["action"] == "character.play_style"]

    def revision(self):
        return self.store.owner_revision(G, "character", self.ann)

    async def test_admin_sets_and_resets_with_audit_detail(self):
        await self.service.set_play_style(self.session, G, self.ann, BOLD, self.revision())
        self.assertEqual(self.store.character_play_style(G, self.ann), BOLD)
        rows = self.audits()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["guild_id"], rows[0]["actor_id"]), (G, 4))
        self.assertEqual(json.loads(rows[0]["detail_json"])["id"], self.ann)
        self.assertEqual(detail_text(rows[0]), "looseness 50→80, risk 50→20, judgment 50→10")
        await self.service.set_play_style(self.session, G, self.ann, None, self.revision())
        self.assertIsNone(self.store.character_play_style(G, self.ann))
        self.assertEqual(detail_text(self.audits()[-1]), "reset")
        self.assertEqual(json.loads(self.audits()[-1]["detail_json"])["id"], self.ann)

    async def test_saving_the_same_style_twice_audits_no_change(self):
        await self.service.set_play_style(self.session, G, self.ann, BOLD, self.revision())
        await self.service.set_play_style(self.session, G, self.ann, BOLD, self.revision())
        rows = self.audits()
        self.assertEqual(len(rows), 2)
        self.assertEqual(detail_text(rows[1]), "no change")
        self.assertEqual(json.loads(rows[1]["detail_json"])["id"], self.ann)

    async def test_stale_revision_conflicts_without_audit(self):
        stale = self.revision()
        await self.service.set_play_style(self.session, G, self.ann, BOLD, stale)
        with self.assertRaises(ConflictError):
            await self.service.set_play_style(self.session, G, self.ann, PlayStyle(), stale)
        self.assertEqual(len(self.audits()), 1)
        self.assertEqual(self.store.character_play_style(G, self.ann), BOLD)

    async def test_non_admin_guild_is_refused_without_a_write_or_audit(self):
        with self.assertRaises(HTTPException) as caught:
            await self.service.set_play_style(self.session, H, self.zed, BOLD, 0)
        self.assertEqual(caught.exception.status_code, 403)
        self.assertIsNone(self.store.character_play_style(H, self.zed))
        self.assertEqual(self.audits(), [])

    async def test_another_guilds_character_is_refused_without_a_write_or_audit(self):
        with self.assertRaises(ValueError):
            await self.service.set_play_style(self.session, G, self.zed, BOLD, 0)
        self.assertIsNone(self.store.character_play_style(H, self.zed))
        self.assertEqual(self.audits(), [])
