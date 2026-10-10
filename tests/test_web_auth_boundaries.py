"""Characterization of dashboard HTTP authorization cost and boundaries
(docs/engineering/history/audit-2026-10-07.md).

The Discord-call counts pin today's behavior (PERF-01). When admin checks become cached, update the
expected counts deliberately and note the change in docs/engineering/perf-baseline.md.

SEC-02: the legacy Jinja form routes were retired, so the csrf / origin / session / body-not-parsed properties (SEC-01)
are pinned on the state-changing POST surfaces that remain: ``POST /logout`` and the NiceGUI upload endpoint guarded by
``protected_uploads`` in ``dashboard.py``.
"""
import unittest
from collections import Counter
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from starlette.requests import Request

from helpers import discord_transport, install_session, shared_dashboard

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

    def test_repeated_avatar_requests_check_discord_once(self):
        """PERF-01 (fixed): N avatar requests in one session within the TTL cost one Discord guild-list call."""
        for _ in range(5):
            self.assertEqual(self.client.get(f"/guild/1/characters/{self.character}/avatar").status_code, 200)
        self.assertEqual(self.calls[GUILDS], 1)

    def test_non_admin_guild_is_forbidden(self):
        self.assertEqual(self.client.get("/guild/2/characters/1/avatar").status_code, 403)

    def test_card_preview_avatar_route_is_gone(self):
        """SEC-02 (replaces PERF-01 test_card_preview_avatar_is_no_store): the session-scoped card-preview avatar route was
        retired with the legacy import flow; even with a preview planted in the session it is a 404."""
        import time
        from types import SimpleNamespace
        self.app.state.sessions[self.session]["card_preview"] = {
            "world_id": 1, "card": SimpleNamespace(avatar=self._png("blue")), "expires": time.time() + 600}
        self.assertEqual(self.client.get("/guild/1/card-preview/avatar").status_code, 404)

    def test_avatar_version_changes_with_the_image_bytes(self):
        """PERF-01 (fixed; SEC-02 migration of test_guild_page_avatar_url_is_versioned): the `?v=` the dashboard appends to
        avatar URLs is a non-empty content version that changes when the bytes change, so a cached image is never shown
        after an upload. (The dashboard's own `<img>` is built lazily in the Characters tab, so only the version
        function is reachable offline, so the `?v=` URL itself is not asserted here.)"""
        from llmcord_core.avatars import avatar_version
        first, second = avatar_version(self._png("red")), avatar_version(self._png("white"))
        self.assertTrue(first)
        self.assertTrue(second)
        self.assertNotEqual(first, second)
        self.assertEqual(first, avatar_version(self._png("red")))

    def test_logout_requires_csrf(self):
        """SEC-01 (migrated from a legacy form route): a `/logout` POST without the csrf field is 403 and keeps the
        session; with it, the POST succeeds."""
        self.assertEqual(self.client.post("/logout", follow_redirects=False).status_code, 403)
        self.assertEqual(self.client.post("/logout", data={"csrf": "wrong"}, follow_redirects=False).status_code, 403)
        self.assertIn(self.session, self.app.state.sessions)
        self.assertEqual(self.client.post("/logout", data={"csrf": "csrf"}, follow_redirects=False).status_code, 303)

    def test_cross_origin_logout_rejected(self):
        """SEC-01 (migrated): a cross-origin `/logout` is 403 even with a valid csrf, and the session survives."""
        response = self.client.post("/logout", data={"csrf": "csrf"}, headers={"origin": "https://evil.test"},
                                    follow_redirects=False)
        self.assertEqual(response.status_code, 403)
        self.assertIn(self.session, self.app.state.sessions)

    def test_unauthenticated_requests_are_rejected(self):
        """SEC-01 (migrated): no session means 401 on a state-changing POST and on an avatar GET, with no Discord call."""
        self.client.cookies.clear()
        self.assertEqual(self.client.get(f"/guild/1/characters/{self.character}/avatar").status_code, 401)
        self.assertEqual(self.client.post("/logout", data={"csrf": "csrf"}).status_code, 401)
        self.assertEqual(self.calls[GUILDS], 0)

    def _post_counting_form_parses(self, headers=None, **kwargs):
        parsed = []
        original = Request.form

        def counting_form(request, *args, **kw):
            parsed.append(request.url.path)
            return original(request, *args, **kw)
        with patch.object(Request, "form", counting_form):
            response = self.client.post("/logout", headers=headers or {}, follow_redirects=False,
                                        data={"csrf": "x" * 4096}, **kwargs)
        return response, parsed

    def test_unauthenticated_body_is_not_parsed(self):
        """SEC-01 (fixed; migrated to /logout): a mutating request without a session is 401 before its body is parsed."""
        self.client.cookies.clear()
        response, parsed = self._post_counting_form_parses()
        self.assertEqual(response.status_code, 401)
        self.assertEqual(parsed, [])

    def test_expired_session_body_is_not_parsed(self):
        """SEC-01 (migrated to /logout): an expired session is 401 before the body is parsed, with no Discord call."""
        self.app.state.sessions[self.session]["expires"] = 0
        response, parsed = self._post_counting_form_parses()
        self.assertEqual(response.status_code, 401)
        self.assertEqual(parsed, [])
        self.assertEqual(self.calls[GUILDS], 0)

    def test_cross_origin_body_is_not_parsed(self):
        """SEC-01 (fixed; migrated to /logout): a valid session with a foreign Origin is 403 before the body is parsed."""
        response, parsed = self._post_counting_form_parses({"origin": "https://evil.test"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(parsed, [])

    def test_cross_origin_without_session_is_401(self):
        """SEC-01 (migrated to /logout): a foreign Origin with no session is still 401 (session first), body unparsed."""
        self.client.cookies.clear()
        response, parsed = self._post_counting_form_parses({"origin": "https://evil.test"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(parsed, [])

    def test_logout_requires_csrf_and_clears_session(self):
        self.assertEqual(self.client.post("/logout", data={"csrf": "csrf"}, follow_redirects=False).status_code, 303)
        self.assertNotIn(self.session, self.app.state.sessions)

    def test_logout_cross_origin_body_is_not_parsed(self):
        """SEC-01 (fixed): a `/logout` POST with a valid session but a foreign Origin is rejected with 403 before its
        form body is parsed, and the session survives (a forged cross-site logout must not sign the user out)."""
        response, parsed = self._post_counting_form_parses({"origin": "https://evil.test"})
        self.assertEqual(parsed, [])
        self.assertEqual(response.status_code, 403)
        self.assertIn(self.session, self.app.state.sessions)

    def test_same_origin_logout_with_csrf_signs_out(self):
        """SEC-01: `/logout` with the dashboard's own Origin and a valid csrf drops the session and clears the cookie."""
        response = self.client.post("/logout", data={"csrf": "csrf"}, headers={"origin": "https://pi.test"},
                                    follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertNotIn(self.session, self.app.state.sessions)
        self.assertIn("llmcord_session=", response.headers["set-cookie"])
        self.assertEqual(self.client.get(f"/guild/1/characters/{self.character}/avatar").status_code, 401)

    def test_expired_session_rejected_without_discord_call(self):
        self.app.state.sessions[self.session]["expires"] = 0
        self.assertEqual(self.client.get(f"/guild/1/characters/{self.character}/avatar").status_code, 401)
        self.assertEqual(self.calls[GUILDS], 0)


class RemovedLegacyRouteTests(unittest.TestCase):
    """SEC-02: the legacy Jinja form routes are gone; NiceGUI under /admin is the only write path."""

    REMOVED_POSTS = (
        "/guild/1/spaces", "/guild/1/links", "/guild/1/bindings", "/guild/1/characters/import",
        "/guild/1/characters/apply", "/guild/1/characters/1", "/guild/1/characters/1/archive", "/guild/1/casts",
        "/guild/1/ambient", "/guild/1/lore", "/guild/1/lore/1", "/guild/1/lore/1/delete", "/guild/1/lore/1/pin",
        "/guild/1/lore/1/promote", "/guild/1/books", "/guild/1/books/1/assign", "/guild/1/books/1/preview",
        "/guild/1/books/1/apply", "/guild/1/books/1/entries/1",
    )

    def setUp(self):
        self.calls = Counter()
        transport, _ = discord_transport(admin_guilds=(1,), calls=self.calls)
        self.app = create_app(":memory:", "https://pi.test", "client", "secret", "bot",
                              httpx.AsyncClient(transport=transport), enable_dashboard=False,
                              config_path="tests/nonexistent-config.yaml")
        self.client = TestClient(self.app, base_url="https://pi.test")
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.cookies.set("llmcord_session", install_session(self.app))

    def test_removed_post_paths_are_404_or_405_for_a_signed_in_admin(self):
        """SEC-02: every retired legacy POST path is 404/405 with a valid session, csrf and same Origin, and changes nothing."""
        for path in self.REMOVED_POSTS:
            with self.subTest(path=path):
                response = self.client.post(path, data={"csrf": "csrf", "name": "Nope", "kind": "world"},
                                            headers={"origin": "https://pi.test"}, follow_redirects=False)
                self.assertIn(response.status_code, (404, 405))
        self.assertEqual(self.app.state.store.list_spaces(1), [])

    def test_root_redirects_to_the_dashboard_even_without_it(self):
        """SEC-02: `GET /` is always 303 -> /admin/ (the no-dashboard login/index render was removed), signed in or not."""
        for signed_in in (True, False):
            if not signed_in:
                self.client.cookies.clear()
            response = self.client.get("/", follow_redirects=False)
            self.assertEqual(response.status_code, 303)
            self.assertEqual(response.headers["location"], "/admin/")
        self.assertEqual(self.calls[GUILDS], 0)

    def test_guild_page_redirects_to_the_dashboard_page(self):
        """SEC-02: the old `/guild/{id}` bookmark is a plain 303 -> /admin/guild/{id}; the dashboard target enforces auth,
        so the redirect itself makes no Discord call and needs no session."""
        response = self.client.get("/guild/1", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/admin/guild/1")
        self.client.cookies.clear()
        response = self.client.get("/guild/7", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/admin/guild/7")
        self.assertEqual(self.calls[GUILDS], 0)


class UploadBoundaryTests(unittest.TestCase):
    """SEC-01 / SEC-02: the NiceGUI upload endpoint (``protected_uploads`` in dashboard.py) is the remaining multipart
    surface. The dashboard app is the process-wide shared one (NiceGUI mounts once). A request that passes the guard is
    forwarded to the router, which has no route for the made-up element id, so "allowed" is observed as 404."""

    def setUp(self):
        from nicegui import Client
        self.app, self.client = shared_dashboard()
        self.ident = f"upload-{self.id().rsplit('.', 1)[-1]}"
        install_session(self.app, self.ident)
        self.client.cookies.set("llmcord_session", self.ident)
        self.assertEqual(self.client.get("/admin/guild/1", headers={"accept-encoding": "identity"}).status_code, 200)
        clients = [key for key, value in Client.instances.items() if getattr(value, "llmcord_binding", None) == (self.ident, 1)]
        self.assertTrue(clients, "page render did not bind a NiceGUI client to the session")
        self.url = f"/admin/_nicegui/client/{clients[-1]}/upload/missing-element"
        self.files = {"file": ("a.json", b"{}", "application/json")}
        self.good = {"X-CSRF-Token": "csrf", "Origin": "https://pi.test"}

    def post_counting_form_parses(self, headers=None, files=None):
        parsed = []
        original = Request.form

        def counting_form(request, *args, **kw):
            parsed.append(request.url.path)
            return original(request, *args, **kw)
        with patch.object(Request, "form", counting_form):
            response = self.client.post(self.url, headers=headers or {}, files=files or self.files)
        return response, parsed

    def test_same_origin_upload_with_csrf_passes_the_guard(self):
        """SEC-01: same Origin, valid session cookie and X-CSRF-Token get past the guard (routed on: 404 here)."""
        response, parsed = self.post_counting_form_parses(self.good)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(parsed, [])

    def test_missing_or_wrong_csrf_is_403(self):
        """SEC-01: no or wrong X-CSRF-Token is 403."""
        self.assertEqual(self.client.post(self.url, files=self.files).status_code, 403)
        wrong = self.client.post(self.url, files=self.files, headers={"X-CSRF-Token": "nope"})
        self.assertEqual(wrong.status_code, 403)

    def test_cross_origin_upload_is_403(self):
        """SEC-01: a foreign Origin is 403 even with a valid csrf token, and the multipart body is not parsed."""
        response, parsed = self.post_counting_form_parses({**self.good, "Origin": "https://evil.test"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(parsed, [])

    def test_no_session_cookie_is_401(self):
        """SEC-01: without the session cookie the upload is 401 (before csrf/origin), body unparsed."""
        self.client.cookies.clear()
        response, parsed = self.post_counting_form_parses({**self.good, "Origin": "https://evil.test"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(parsed, [])

    def test_expired_session_is_401(self):
        """SEC-01: an expired session is 401, body unparsed."""
        self.app.state.sessions[self.ident]["expires"] = 0
        response, parsed = self.post_counting_form_parses(self.good)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(parsed, [])

    def test_oversized_upload_is_413_after_guard(self):
        """SEC-01: a body over 8 MiB (+ framing allowance) is 413, but only after the guard passes."""
        big = {"file": ("big.json", b"x" * (8 * 1024 * 1024 + 70000), "application/json")}
        self.assertEqual(self.client.post(self.url, files=big, headers=self.good).status_code, 413)
        self.assertEqual(self.client.post(self.url, files=big, headers={"X-CSRF-Token": "nope"}).status_code, 403)


class OperatorBindingTests(unittest.TestCase):
    """FEAT-08: the /operator page binds its client to the OPERATOR sentinel; socket events and uploads re-check operator
    status (never the guild guard), and the page answers non-operators exactly like an unknown path."""

    def setUp(self):
        from nicegui import Client
        self.app, self.client = shared_dashboard()
        self.addCleanup(setattr, self.app.state, "operator_ids", self.app.state.operator_ids)
        self.app.state.operator_ids = frozenset({9})
        self.ident = f"operator-{self.id().rsplit('.', 1)[-1]}"
        install_session(self.app, self.ident, user_id="9")
        self.client.cookies.set("llmcord_session", self.ident)
        self.assertEqual(self.client.get("/admin/operator", headers={"accept-encoding": "identity"}).status_code, 200)
        from llmcord_core.dashboard import OPERATOR
        clients = [key for key, value in Client.instances.items()
                   if (b := getattr(value, "llmcord_binding", None)) and b[0] == self.ident and b[1] is OPERATOR]
        self.assertTrue(clients, "operator page did not bind a client to the OPERATOR sentinel")
        self.client_id = clients[-1]

    def socket(self, name, patched=None):
        """Run the installed socket handler `name`; True when the guarded original ran."""
        import asyncio
        from unittest.mock import PropertyMock
        from nicegui import Client, core
        client = Client.instances[self.client_id]
        message = {"client_id": self.client_id, "next_message_id": 0}
        environ = {"HTTP_COOKIE": f"llmcord_session={self.ident}"}
        calls = []
        with patch.object(core.sio, "get_environ", return_value=environ), \
                patch.object(client.outbox, "prune_history", side_effect=lambda *a: calls.append(a)), \
                patch.object(client, "handle_event", side_effect=lambda *a: calls.append(a)), \
                patch.object(Client, "has_socket_connection", new_callable=PropertyMock, return_value=True), \
                patch.object(self.app.state.auth, "guard", side_effect=AssertionError("guard called for OPERATOR")):
            asyncio.run(core.sio.handlers["/"][name]("sid", message))
        return bool(calls)

    def test_operator_event_allowed_then_dropped_after_removal(self):
        """FEAT-08: a live event from an operator-bound client runs; once the user leaves operator_ids it is dropped."""
        self.assertTrue(self.socket("event"))
        self.app.state.operator_ids = frozenset()
        self.assertFalse(self.socket("event"))

    def test_session_only_events_do_not_need_operator(self):
        """FEAT-08: acks/JS results (check_permissions=False) only need the session, like the guild-less binding."""
        self.app.state.operator_ids = frozenset()
        self.assertTrue(self.socket("ack"))
        self.app.state.sessions.pop(self.ident)
        self.assertFalse(self.socket("ack"))

    def test_upload_to_operator_client_is_403_without_the_guild_guard(self):
        """FEAT-08: the operator page has no uploads; the guild guard is never consulted with OPERATOR."""
        url = f"/admin/_nicegui/client/{self.client_id}/upload/missing-element"
        with patch.object(self.app.state.auth, "guard", side_effect=AssertionError("guard called for OPERATOR")):
            response = self.client.post(url, files={"file": ("a.json", b"{}", "application/json")},
                                        headers={"X-CSRF-Token": "csrf", "Origin": "https://pi.test"})
        self.assertEqual(response.status_code, 403)

    def test_non_operator_and_signed_out_get_the_unknown_page_response(self):
        """FEAT-08: /admin/operator for a non-operator or signed-out visitor matches an unknown path (status and size)."""
        unknown = self.client.get("/admin/no-such-page")
        self.assertEqual(unknown.status_code, 404)
        self.app.state.operator_ids = frozenset()
        for cookies in ({"llmcord_session": self.ident}, {}):
            self.client.cookies.clear()
            for key, value in cookies.items():
                self.client.cookies.set(key, value)
            response = self.client.get("/admin/operator")
            self.assertEqual((response.status_code, len(response.text)), (404, len(unknown.text)))

    def test_is_operator_tolerates_malformed_ids(self):
        """FEAT-08: a session without a numeric user id is simply not an operator."""
        auth = self.app.state.auth
        self.assertTrue(auth.is_operator({"user": {"id": "9"}}))
        for bad in ({}, {"user": {}}, {"user": {"id": "x"}}, {"user": {"id": None}}):
            self.assertFalse(auth.is_operator(bad))


if __name__ == "__main__":
    unittest.main()
