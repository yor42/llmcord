"""Characterization of dashboard HTTP authorization cost and boundaries (docs/engineering/audit.md).

The Discord-call counts pin today's behavior (PERF-01). When admin checks become cached, update the
expected counts deliberately and note the change in docs/engineering/perf-baseline.md.
"""
import unittest
from collections import Counter
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from starlette.requests import Request

from helpers import discord_transport, install_session

from llmcord_core.web import create_app

GUILDS = "GET /users/@me/guilds"


class WebAuthBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.calls = Counter()
        transport, _ = discord_transport(admin_guilds=(1,), calls=self.calls)
        self.app = create_app(":memory:", "https://pi.test", "client", "secret", "bot",
                              httpx.AsyncClient(transport=transport), enable_dashboard=False,
                              config_path="tests/nonexistent-config.yaml")
        self.client = TestClient(self.app, base_url="https://pi.test")
        self.client.__enter__()
        store = self.app.state.store
        world = store.create_space(1, "World", "world")
        self.character = store.add_character(1, world, "Alice", {"name": "Alice"}, None, [])
        from llmcord_core.avatars import normalize_avatar
        from io import BytesIO
        from PIL import Image
        png = BytesIO()
        Image.new("RGB", (8, 8)).save(png, "PNG")
        store.db.execute("UPDATE characters SET avatar=? WHERE id=?", (normalize_avatar(png.getvalue()), self.character))
        store.db.commit()
        self.session = install_session(self.app)
        self.client.cookies.set("llmcord_session", self.session)

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def _png(self, color):
        from io import BytesIO

        from PIL import Image

        from llmcord_core.avatars import normalize_avatar
        png = BytesIO()
        Image.new("RGB", (8, 8), color).save(png, "PNG")
        return normalize_avatar(png.getvalue())

    def _assert_private_cacheable(self, response):
        self.assertEqual(response.status_code, 200)
        cache = response.headers["cache-control"]
        self.assertIn("private", cache)
        self.assertIn("max-age=300", cache)
        self.assertNotIn("no-store", cache)
        self.assertNotIn("public", cache)

    def _guild_page_avatar_src(self):
        import html
        import re
        page = self.client.get("/guild/1")
        self.assertEqual(page.status_code, 200)
        match = re.search(r'<img[^>]*src="(/guild/1/characters/%d/avatar[^"]*)"' % self.character, page.text)
        self.assertIsNotNone(match, "guild page has no <img> for the character avatar")
        return html.unescape(match.group(1))

    def test_character_avatar_is_privately_cacheable(self):
        """PERF-01 (fixed): the stored character avatar is sent `Cache-Control: private, max-age=300` (not no-store, never
        public), so the browser stops refetching it on every render."""
        self._assert_private_cacheable(self.client.get(f"/guild/1/characters/{self.character}/avatar"))

    def test_emotion_avatar_is_privately_cacheable(self):
        """PERF-01 (fixed): a stored emotion avatar slot is sent `Cache-Control: private, max-age=300` (not no-store, never
        public)."""
        self.app.state.store.save_avatar(1, self.character, "happy", "Happy", "", image=self._png("red"))
        self._assert_private_cacheable(self.client.get(f"/guild/1/characters/{self.character}/avatars/happy"))

    def test_avatar_error_responses_are_no_store(self):
        """PERF-01: avatar 403, 404 and 401 responses stay `no-store`; only stored images become cacheable."""
        forbidden = self.client.get(f"/guild/2/characters/{self.character}/avatar")
        self.assertEqual(forbidden.status_code, 403)
        self.assertIn("no-store", forbidden.headers["cache-control"])
        missing = self.client.get("/guild/1/characters/999999/avatar")
        self.assertEqual(missing.status_code, 404)
        self.assertIn("no-store", missing.headers["cache-control"])
        empty_slot = self.client.get(f"/guild/1/characters/{self.character}/avatars/neutral")
        self.assertEqual(empty_slot.status_code, 404)
        self.assertIn("no-store", empty_slot.headers["cache-control"])
        self.client.cookies.clear()
        for path in (f"/guild/1/characters/{self.character}/avatar", f"/guild/1/characters/{self.character}/avatars/neutral"):
            unauthenticated = self.client.get(path)
            self.assertEqual(unauthenticated.status_code, 401)
            self.assertIn("no-store", unauthenticated.headers["cache-control"])

    def test_guild_page_is_no_store(self):
        """PERF-01: HTML pages stay `no-store` when avatars become cacheable."""
        page = self.client.get("/guild/1")
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.headers["cache-control"], "no-store")

    def test_card_preview_avatar_is_no_store(self):
        """PERF-01: the card-preview avatar is session-scoped and temporary, so it stays `no-store`."""
        import time
        from types import SimpleNamespace
        image = self._png("blue")
        self.app.state.sessions[self.session]["card_preview"] = {
            "world_id": 1, "card": SimpleNamespace(avatar=image), "expires": time.time() + 600}
        response = self.client.get("/guild/1/card-preview/avatar")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, image)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_avatar_routes_ignore_version_query(self):
        """PERF-01: avatar routes ignore unknown query parameters, so `?v=...` (and no `v`) both serve the image."""
        stored = self.app.state.store.one("SELECT avatar FROM characters WHERE id=?", (self.character,))["avatar"]
        for query in ("", "?v=anything", "?v="):
            response = self.client.get(f"/guild/1/characters/{self.character}/avatar{query}")
            self.assertEqual(response.status_code, 200, query)
            self.assertEqual(response.content, stored)
        slot = self._png("green")
        self.app.state.store.save_avatar(1, self.character, "happy", "Happy", "", image=slot)
        response = self.client.get(f"/guild/1/characters/{self.character}/avatars/happy?v=anything")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, slot)

    def test_guild_page_avatar_url_is_versioned(self):
        """PERF-01 (fixed): the legacy guild page links the character avatar with a non-empty `v=` content version, and the
        version changes when the avatar bytes change, so a cached image is never shown after an upload."""
        from urllib.parse import parse_qs, urlsplit
        first = parse_qs(urlsplit(self._guild_page_avatar_src()).query).get("v", [""])[0]
        self.assertTrue(first, "avatar <img> src has no v= parameter")
        store = self.app.state.store
        store.db.execute("UPDATE characters SET avatar=? WHERE id=?", (self._png("white"), self.character))
        store.db.commit()
        second = parse_qs(urlsplit(self._guild_page_avatar_src()).query).get("v", [""])[0]
        self.assertTrue(second)
        self.assertNotEqual(first, second)

    def test_repeated_avatar_requests_check_discord_once(self):
        """PERF-01 (fixed): N avatar requests in one session within the TTL cost one Discord guild-list call."""
        for _ in range(5):
            self.assertEqual(self.client.get(f"/guild/1/characters/{self.character}/avatar").status_code, 200)
        self.assertEqual(self.calls[GUILDS], 1)

    def test_legacy_guild_page_cost(self):
        """PERF-01: one legacy page render = one user-token check plus one bot-token channel fetch."""
        self.assertEqual(self.client.get("/guild/1").status_code, 200)
        self.assertEqual(self.calls[GUILDS], 1)
        self.assertEqual(self.calls["GET /guilds/1/channels"], 1)

    def test_non_admin_guild_is_forbidden(self):
        self.assertEqual(self.client.get("/guild/2/characters/1/avatar").status_code, 403)

    def test_mutation_requires_csrf(self):
        self.assertEqual(self.client.post("/guild/1/spaces", data={"name": "B", "kind": "world"}).status_code, 403)
        ok = self.client.post("/guild/1/spaces", data={"csrf": "csrf", "name": "B", "kind": "world"}, follow_redirects=False)
        self.assertEqual(ok.status_code, 303)

    def test_cross_origin_mutation_rejected(self):
        response = self.client.post("/guild/1/spaces", data={"csrf": "csrf", "name": "C", "kind": "world"},
                                    headers={"origin": "https://evil.test"})
        self.assertEqual(response.status_code, 403)

    def test_unauthenticated_requests_are_rejected(self):
        self.client.cookies.clear()
        self.assertEqual(self.client.get("/guild/1").status_code, 401)
        self.assertEqual(self.client.post("/guild/1/spaces", data={"name": "X", "kind": "world"}).status_code, 401)
        self.assertEqual(self.calls[GUILDS], 0)

    def test_unauthenticated_body_is_not_parsed(self):
        """SEC-01 (fixed): a mutating request without a session is rejected with 401 before its body is parsed."""
        self.client.cookies.clear()
        parsed = []
        original = Request.form

        def counting_form(request, *args, **kwargs):
            parsed.append(request.url.path)
            return original(request, *args, **kwargs)
        with patch.object(Request, "form", counting_form):
            response = self.client.post("/guild/1/characters/import", files={"file": ("a.json", b"x" * 1024, "application/json")})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(parsed, [])

    def test_expired_session_body_is_not_parsed(self):
        """SEC-01: an expired session is rejected with 401 before the multipart body is parsed."""
        self.app.state.sessions[self.session]["expires"] = 0
        parsed = []
        original = Request.form

        def counting_form(request, *args, **kwargs):
            parsed.append(request.url.path)
            return original(request, *args, **kwargs)
        with patch.object(Request, "form", counting_form):
            response = self.client.post("/guild/1/characters/import", files={"file": ("a.json", b"x" * 1024, "application/json")})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(parsed, [])
        self.assertEqual(self.calls[GUILDS], 0)

    def test_cross_origin_body_is_not_parsed(self):
        """SEC-01 (fixed): a mutating request with a valid session but a foreign Origin is rejected with 403 before its
        multipart body is parsed."""
        parsed = []
        original = Request.form

        def counting_form(request, *args, **kwargs):
            parsed.append(request.url.path)
            return original(request, *args, **kwargs)
        with patch.object(Request, "form", counting_form):
            response = self.client.post("/guild/1/characters/import", headers={"origin": "https://evil.test"},
                                        files={"file": ("a.json", b"x" * 1024, "application/json")})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(parsed, [])

    def test_cross_origin_without_session_is_401(self):
        """SEC-01: a foreign Origin with no session is still 401 (session is checked before origin), body unparsed."""
        self.client.cookies.clear()
        parsed = []
        original = Request.form

        def counting_form(request, *args, **kwargs):
            parsed.append(request.url.path)
            return original(request, *args, **kwargs)
        with patch.object(Request, "form", counting_form):
            response = self.client.post("/guild/1/characters/import", headers={"origin": "https://evil.test"},
                                        files={"file": ("a.json", b"x" * 1024, "application/json")})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(parsed, [])

    def test_logout_requires_csrf_and_clears_session(self):
        self.assertEqual(self.client.post("/logout", data={"csrf": "csrf"}, follow_redirects=False).status_code, 303)
        self.assertNotIn(self.session, self.app.state.sessions)

    def test_logout_cross_origin_body_is_not_parsed(self):
        """SEC-01 (fixed): a `/logout` POST with a valid session but a foreign Origin is rejected with 403 before its
        form body is parsed, and the session survives (a forged cross-site logout must not sign the user out)."""
        parsed = []
        original = Request.form

        def counting_form(request, *args, **kwargs):
            parsed.append(request.url.path)
            return original(request, *args, **kwargs)
        with patch.object(Request, "form", counting_form):
            response = self.client.post("/logout", data={"csrf": "csrf"}, headers={"origin": "https://evil.test"},
                                        follow_redirects=False)
        self.assertEqual(parsed, [])
        self.assertEqual(response.status_code, 403)
        self.assertIn(self.session, self.app.state.sessions)

    def test_same_origin_mutation_with_csrf_succeeds(self):
        """SEC-01: a mutating admin POST with the dashboard's own Origin, a valid session and csrf passes
        `check_origin` and applies the change."""
        response = self.client.post("/guild/1/spaces", data={"csrf": "csrf", "name": "Same", "kind": "world"},
                                    headers={"origin": "https://pi.test"}, follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/guild/1")
        self.assertIn("Same", [space["name"] for space in self.app.state.store.list_spaces(1)])

    def test_same_origin_logout_with_csrf_signs_out(self):
        """SEC-01: `/logout` with the dashboard's own Origin and a valid csrf drops the session and clears the cookie."""
        response = self.client.post("/logout", data={"csrf": "csrf"}, headers={"origin": "https://pi.test"},
                                    follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertNotIn(self.session, self.app.state.sessions)
        self.assertIn("llmcord_session=", response.headers["set-cookie"])
        self.assertEqual(self.client.get("/guild/1").status_code, 401)

    def test_expired_session_rejected_without_discord_call(self):
        self.app.state.sessions[self.session]["expires"] = 0
        self.assertEqual(self.client.get(f"/guild/1/characters/{self.character}/avatar").status_code, 401)
        self.assertEqual(self.calls[GUILDS], 0)


if __name__ == "__main__":
    unittest.main()
