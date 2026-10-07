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

    def test_avatar_responses_are_no_store(self):
        """Characterization (PERF-01): avatar images are served with `Cache-Control: no-store` (changes in R2.4)."""
        response = self.client.get(f"/guild/1/characters/{self.character}/avatar")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")

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

    def test_expired_session_rejected_without_discord_call(self):
        self.app.state.sessions[self.session]["expires"] = 0
        self.assertEqual(self.client.get(f"/guild/1/characters/{self.character}/avatar").status_code, 401)
        self.assertEqual(self.calls[GUILDS], 0)


if __name__ == "__main__":
    unittest.main()
