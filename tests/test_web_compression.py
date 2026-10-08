"""Compression and security-header middleware behavior (PERF-04, docs/engineering/audit.md)."""
import re
import unittest
from io import BytesIO

import httpx
from fastapi.testclient import TestClient

from helpers import discord_transport, install_session

from llmcord_core.web import create_app

BASE_POLICY = "default-src 'self'; style-src 'self' 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'"
ADMIN_SUFFIX = ("; img-src 'self' data: blob:; connect-src 'self' wss://pi.test; font-src 'self' data:")
IMMUTABLE = "public, max-age=31536000, immutable, stale-while-revalidate=31536000"
GZIP = {"accept-encoding": "gzip"}
IDENTITY = {"accept-encoding": "identity"}


def make_client(*, dashboard):
    transport, _ = discord_transport(admin_guilds=(1,))
    app = create_app(":memory:", "https://pi.test", "client", "secret", "bot",
                     httpx.AsyncClient(transport=transport), enable_dashboard=dashboard,
                     config_path="tests/nonexistent-config.yaml")
    client = TestClient(app, base_url="https://pi.test")
    client.__enter__()
    client.cookies.set("llmcord_session", install_session(app))
    return app, client


class DashboardCompressionTests(unittest.TestCase):
    # NiceGUI's global app cannot be mounted twice per process, so build the dashboard app once.
    @classmethod
    def setUpClass(cls):
        cls.app, cls.client = make_client(dashboard=True)
        cls.addClassCleanup(cls.client.__exit__, None, None, None)
        cls.page = cls.client.get("/admin/guild/1", headers=IDENTITY)

    def _static_paths(self):
        return re.findall(r'src="(/admin/_nicegui/[^"]+/static/[^"]+\.js)"', self.page.text)

    def test_static_js_is_gzipped_and_decodes_to_same_bytes(self):
        """PERF-04: NiceGUI static JS is gzip-encoded for clients that accept it and decodes to the identical bytes."""
        paths = self._static_paths()
        self.assertTrue(paths)
        path = next(p for p in paths if p.endswith("quasar.umd.prod.js"))
        plain = self.client.get(path, headers=IDENTITY)
        packed = self.client.get(path, headers=GZIP)
        self.assertEqual(plain.status_code, 200)
        self.assertIsNone(plain.headers.get("content-encoding"))
        self.assertEqual(packed.headers.get("content-encoding"), "gzip")
        self.assertEqual(packed.content, plain.content)
        self.assertEqual(packed.headers["x-content-type-options"], "nosniff")

    def test_admin_html_is_not_compressed_and_nonces_match(self):
        """PERF-04: /admin HTML is never gzip-encoded (BREACH, and CSP nonce rewriting); every <script> carries the
        nonce named in the CSP header and content-length matches the rewritten body."""
        response = self.client.get("/admin/guild/1", headers=GZIP)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.headers.get("content-encoding"))
        nonce = re.search(r"'nonce-([^']+)'", response.headers["content-security-policy"]).group(1)
        scripts = re.findall(r"<script[^>]*>", response.text)
        self.assertTrue(scripts)
        for tag in scripts:
            self.assertIn(f'nonce="{nonce}"', tag)
        if "content-length" in response.headers:
            self.assertEqual(int(response.headers["content-length"]), len(response.content))

    def test_admin_static_headers_unchanged(self):
        """PERF-04: a versioned /admin NiceGUI static asset gets NiceGUI's immutable cache-control and gets nosniff,
        same-origin referrer and the admin CSP (with a nonce); it is not rewritten."""
        path = self._static_paths()[0]
        response = self.client.get(path, headers=IDENTITY)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], IMMUTABLE)
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["referrer-policy"], "same-origin")
        policy = response.headers["content-security-policy"]
        nonce = re.search(r"'nonce-([^']+)'", policy).group(1)
        self.assertEqual(policy, BASE_POLICY + f"; script-src 'self' 'nonce-{nonce}' 'unsafe-eval'" + ADMIN_SUFFIX)
        self.assertNotIn(b'nonce="', response.content)

    def test_unversioned_admin_paths_stay_no_store(self):
        """PERF-04: per-client and unversioned /admin/_nicegui paths are still no-store."""
        for path in ("/admin/_nicegui/client/abc/missing", "/admin/_nicegui/static/none.js"):
            response = self.client.get(path, headers=IDENTITY)
            self.assertEqual(response.headers["cache-control"], "no-store", path)

    def test_versioned_component_is_immutable_and_versioned_404_is_no_store(self):
        """PERF-04: /components and /libraries under the versioned prefix are immutable; a versioned 404 is no-store."""
        paths = re.findall(r'(?:src|href)="(/admin/_nicegui/[^"/]+/(?:components|libraries)/[^"]+)"', self.page.text)
        self.assertTrue(paths)
        response = self.client.get(paths[0], headers=IDENTITY)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], IMMUTABLE)
        prefix = paths[0].rsplit("/components/", 1)[0].rsplit("/libraries/", 1)[0]
        missing = self.client.get(prefix + "/static/definitely-missing.js", headers=IDENTITY)
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.headers["cache-control"], "no-store")

    def test_admin_html_nonce_differs_per_request(self):
        """PERF-04: each /admin HTML response carries a fresh CSP nonce."""
        other = self.client.get("/admin/guild/1", headers=IDENTITY)
        nonces = {re.search(r"'nonce-([^']+)'", r.headers["content-security-policy"]).group(1) for r in (self.page, other)}
        self.assertEqual(len(nonces), 2)

    def test_admin_html_headers_unchanged(self):
        """Characterization (PERF-04): /admin HTML is no-store with the full admin policy."""
        response = self.page
        nonce = re.search(r"'nonce-([^']+)'", response.headers["content-security-policy"]).group(1)
        self.assertEqual(response.headers["content-security-policy"],
                         BASE_POLICY + f"; script-src 'self' 'nonce-{nonce}' 'unsafe-eval'" + ADMIN_SUFFIX)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["referrer-policy"], "same-origin")


class LegacyHeaderTests(unittest.TestCase):
    def setUp(self):
        self.app, self.client = make_client(dashboard=False)
        self.addCleanup(self.client.__exit__, None, None, None)
        store = self.app.state.store
        world = store.create_space(1, "World", "world")
        self.character = store.add_character(1, world, "Alice", {"name": "Alice"}, None, [])
        from PIL import Image

        from llmcord_core.avatars import normalize_avatar
        png = BytesIO()
        Image.new("RGB", (8, 8)).save(png, "PNG")
        store.db.execute("UPDATE characters SET avatar=? WHERE id=?", (normalize_avatar(png.getvalue()), self.character))
        store.db.commit()

    def _assert_common(self, response):
        self.assertEqual(response.headers["content-security-policy"], BASE_POLICY)
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["referrer-policy"], "same-origin")

    def test_legacy_page_headers(self):
        """Characterization (PERF-04): a legacy page is no-store with the base policy (no script-src, no nonce)."""
        response = self.client.get("/guild/1", headers=GZIP)
        self.assertEqual(response.status_code, 200)
        self._assert_common(response)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertIsNone(response.headers.get("content-encoding"))
        self.assertNotIn('nonce="', response.text)

    def test_avatar_route_keeps_private_cache(self):
        """Characterization (PERF-04): the avatar route keeps its own private cache-control and the base policy."""
        response = self.client.get(f"/guild/1/characters/{self.character}/avatar", headers=GZIP)
        self.assertEqual(response.status_code, 200)
        self._assert_common(response)
        self.assertIn("private", response.headers["cache-control"])
        self.assertIn("max-age=300", response.headers["cache-control"])

    def test_error_responses_headers(self):
        """Characterization (PERF-04): plain-text and JSON error responses are no-store with the base policy."""
        forbidden = self.client.get("/guild/2/characters/1/avatar")
        self.assertEqual(forbidden.status_code, 403)
        self._assert_common(forbidden)
        self.assertEqual(forbidden.headers["cache-control"], "no-store")
        self.client.cookies.clear()
        unauth = self.client.get("/guild/1/characters/1/avatar")
        self.assertEqual(unauth.status_code, 401)
        self._assert_common(unauth)
        self.assertEqual(unauth.headers["cache-control"], "no-store")
        invalid = self.client.post("/guild/1/anything-missing")
        self._assert_common(invalid)
        self.assertEqual(invalid.headers["cache-control"], "no-store")
        self.assertEqual(invalid.headers["content-type"].split(";")[0], "application/json")


if __name__ == "__main__":
    unittest.main()
