"""FEAT-45 (D39, batch 2b): the viewer's language in the dashboard and HTTP layer.

Covers locale negotiation for a request, the safe redirect target, ``POST /lang`` (modelled on ``/logout``), translated
HTTP error details, and the language NiceGUI puts on the page. English output with no cookie stays unchanged.
"""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from starlette.requests import Request

from helpers import discord_transport, install_session, shared_dashboard

from llmcord_core import i18n
from llmcord_core.i18n import web_locale
from llmcord_core.web import create_app


def make_request(cookies=None, accept_language=None, query=""):
    headers = []
    if cookies:
        headers.append((b"cookie", "; ".join(f"{k}={v}" for k, v in cookies.items()).encode()))
    if accept_language is not None:
        headers.append((b"accept-language", accept_language.encode()))
    return Request({"type": "http", "method": "GET", "path": "/", "headers": headers, "query_string": query.encode(),
                    "scheme": "https", "server": ("pi.test", 443)})


def account(locale):
    return {"user": {"id": "4", "username": "Admin", "locale": locale}}


class LocaleForRequestTests(unittest.TestCase):
    def test_default_is_english(self):
        """FEAT-45: no cookie, no account locale and no Accept-Language resolves to en."""
        self.assertEqual(web_locale.locale_for_request(make_request()), "en")
        self.assertEqual(web_locale.locale_for_request(make_request(), None), "en")

    def test_accept_language_is_used(self):
        """FEAT-45: Accept-Language picks an available locale when nothing else is set."""
        self.assertEqual(web_locale.locale_for_request(make_request(accept_language="ko-KR,ko;q=0.9,en;q=0.5")), "ko")
        self.assertEqual(web_locale.locale_for_request(make_request(accept_language="en-US,en;q=0.9")), "en")

    def test_account_locale_beats_accept_language(self):
        """FEAT-45: the signed-in account's locale beats Accept-Language."""
        request = make_request(accept_language="en-US")
        self.assertEqual(web_locale.locale_for_request(request, account("ko")), "ko")

    def test_cookie_beats_account_and_accept_language(self):
        """FEAT-45: the language cookie beats the account locale and Accept-Language."""
        request = make_request(cookies={web_locale.COOKIE: "en"}, accept_language="ko")
        self.assertEqual(web_locale.locale_for_request(request, account("ko")), "en")
        request = make_request(cookies={web_locale.COOKIE: "ko"}, accept_language="en")
        self.assertEqual(web_locale.locale_for_request(request, account("en")), "ko")

    def test_lang_query_beats_everything(self):
        """FEAT-45: a `lang` query parameter naming an available locale wins over cookie, account and Accept-Language."""
        request = make_request(cookies={web_locale.COOKIE: "en"}, accept_language="en", query="lang=ko")
        self.assertEqual(web_locale.locale_for_request(request, account("en")), "ko")

    def test_unknown_values_are_skipped(self):
        """FEAT-45: an unavailable query, cookie or account locale is skipped and the next source is used."""
        self.assertEqual(web_locale.locale_for_request(make_request(query="lang=xx", accept_language="ko")), "ko")
        self.assertEqual(web_locale.locale_for_request(make_request(cookies={web_locale.COOKIE: "xx"}), account("ko")), "ko")
        self.assertEqual(web_locale.locale_for_request(make_request(cookies={web_locale.COOKIE: "xx"}, accept_language="zz"), account("qq")), "en")
        self.assertEqual(web_locale.locale_for_request(make_request(query="lang=<script>")), "en")

    def test_session_without_a_user_locale_is_ignored(self):
        """FEAT-45: a session with no locale (or an empty dict) does not break negotiation."""
        self.assertEqual(web_locale.locale_for_request(make_request(accept_language="ko"), {"user": {"id": "4"}}), "ko")
        self.assertEqual(web_locale.locale_for_request(make_request(accept_language="ko"), {}), "ko")

    def test_pseudo_needs_the_environment_switch(self):
        """FEAT-45: pseudo is a choice only when LLMCORD_I18N_PSEUDO=1."""
        request = make_request(cookies={web_locale.COOKIE: "pseudo"}, query="lang=pseudo")
        with patch.dict(os.environ, {"LLMCORD_I18N_PSEUDO": ""}):
            self.assertEqual(web_locale.locale_for_request(request), "en")
        with patch.dict(os.environ, {"LLMCORD_I18N_PSEUDO": "1"}):
            self.assertEqual(web_locale.locale_for_request(request), "pseudo")
            self.assertEqual(web_locale.locale_for_request(make_request(cookies={web_locale.COOKIE: "pseudo"})), "pseudo")


class TranslatorFactoryTests(unittest.TestCase):
    def test_request_t_uses_the_request_locale(self):
        """FEAT-45: request_t returns a Translator for the negotiated locale."""
        t = web_locale.request_t(make_request(accept_language="ko"))
        self.assertIsInstance(t, i18n.Translator)
        self.assertEqual(t.locale, "ko")
        self.assertEqual(web_locale.request_t(make_request(), account("ko")).locale, "ko")
        self.assertEqual(web_locale.request_t(make_request()).locale, "en")

    def test_request_t_english_text_is_unchanged(self):
        """FEAT-45: English through request_t is the message id itself."""
        self.assertEqual(web_locale.request_t(make_request())("Invalid form token"), "Invalid form token")

    def test_client_t_reads_the_bound_locale(self):
        """FEAT-45: client_t uses the locale bound on the client and falls back to en."""
        self.assertEqual(web_locale.client_t(SimpleNamespace(llmcord_locale="ko")).locale, "ko")
        self.assertEqual(web_locale.client_t(SimpleNamespace()).locale, "en")
        self.assertIsInstance(web_locale.client_t(SimpleNamespace()), i18n.Translator)


class LookupTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"LLMCORD_I18N_PSEUDO": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_lookup_pseudo_translates_a_known_msgid(self):
        """FEAT-45: lookup returns pseudo text in the pseudo locale."""
        text = "Server administrator permission required"
        self.assertEqual(i18n.Translator("pseudo").lookup(text), i18n.pseudo(text))

    def test_lookup_does_not_format_braces(self):
        """FEAT-45: lookup leaves braces alone and never raises."""
        self.assertEqual(i18n.Translator("en").lookup("a {b} c"), "a {b} c")

    def test_rejection_notice_translates_the_detail_at_the_edge(self):
        """FEAT-45: a 403 detail raised in English is pseudo-translated for a pseudo viewer, exact English otherwise."""
        from fastapi import HTTPException
        from llmcord_core.dashboard import rejection_notice
        error = HTTPException(403, "Server administrator permission required")
        self.assertEqual(rejection_notice(error, t=i18n.Translator("pseudo")), i18n.pseudo("Server administrator permission required"))
        self.assertEqual(rejection_notice(error, t=i18n.english()), "Server administrator permission required")


class SafeNextTests(unittest.TestCase):
    def test_local_paths_are_kept(self):
        """FEAT-45: a plain local path (with query) is returned unchanged."""
        for value in ("/guild/1?tab=lore", "/", "/operator", "/admin/guild/1", "/guild/1?tab=lore&x=1#top"):
            self.assertEqual(web_locale.safe_next(value), value)

    def test_unsafe_values_become_root(self):
        """FEAT-45: protocol-relative, absolute, backslash, scheme, control-character, empty and non-str values give '/'."""
        for value in ("//evil.example", "https://evil.example", "http://evil.example/x", "/\\evil", "\\evil", "javascript:x",
                      "", None, "/\nfoo", "/\rfoo", "/foo\x00", "/\tfoo", "evil.example", "guild/1", 5, b"/guild/1",
                      "//", "///evil", "/\\\\evil"):
            self.assertEqual(web_locale.safe_next(value), "/", repr(value))


class LangRouteTests(unittest.TestCase):
    def setUp(self):
        transport, _ = discord_transport(admin_guilds=(1,))
        self.app = create_app(":memory:", "https://pi.test", "client", "secret", "bot",
                              httpx.AsyncClient(transport=transport), enable_dashboard=False,
                              config_path="tests/nonexistent-config.yaml")
        self.client = TestClient(self.app, base_url="https://pi.test")
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.session = install_session(self.app)
        self.client.cookies.set("llmcord_session", self.session)

    def post(self, data=None, **kwargs):
        return self.client.post("/lang", data=data, follow_redirects=False, **kwargs)

    def test_switch_sets_cookie_and_redirects(self):
        """FEAT-45: a valid POST answers 303 and sets the language cookie Secure, HttpOnly, SameSite=Lax with a max age."""
        response = self.post({"csrf": "csrf", "locale": "ko", "next": "/guild/1?tab=lore"})
        self.assertEqual(response.status_code, 303)
        self.assertTrue(response.headers["location"].startswith("/"))
        self.assertTrue(response.headers["location"].endswith("/guild/1?tab=lore"))
        cookie = response.headers["set-cookie"]
        self.assertIn(f"{web_locale.COOKIE}=ko", cookie)
        lowered = cookie.lower()
        self.assertIn("secure", lowered)
        self.assertIn("httponly", lowered)
        self.assertIn("samesite=lax", lowered)
        self.assertIn(f"max-age={web_locale.COOKIE_MAX_AGE}", lowered)

    def test_switch_back_to_english(self):
        """FEAT-45: choosing en sets the cookie to en (an explicit choice, not a deletion)."""
        response = self.post({"csrf": "csrf", "locale": "en", "next": "/"})
        self.assertEqual(response.status_code, 303)
        self.assertIn(f"{web_locale.COOKIE}=en", response.headers["set-cookie"])

    def test_cookie_changes_the_next_request_language(self):
        """FEAT-45: after the switch, the cookie jar's language drives the HTTP error text (here pseudo)."""
        with patch.dict(os.environ, {"LLMCORD_I18N_PSEUDO": "1"}):
            self.assertEqual(self.post({"csrf": "csrf", "locale": "pseudo", "next": "/"}).status_code, 303)
            self.assertEqual(self.client.cookies.get(web_locale.COOKIE), "pseudo")
            response = self.client.post("/logout", data={"csrf": "wrong"}, follow_redirects=False)
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], i18n.pseudo("Invalid form token"))

    def test_missing_or_wrong_csrf_is_403(self):
        """FEAT-45: a missing or wrong form token is 403 'Invalid form token' and sets no cookie."""
        for data in ({"locale": "ko", "next": "/"}, {"csrf": "wrong", "locale": "ko", "next": "/"}):
            response = self.post(data)
            self.assertEqual(response.status_code, 403)
            self.assertEqual(response.json()["detail"], "Invalid form token")
            self.assertNotIn("set-cookie", response.headers)

    def test_cross_origin_is_rejected(self):
        """FEAT-45: a foreign Origin is 403 like /logout, even with a valid token, and sets no cookie."""
        response = self.post({"csrf": "csrf", "locale": "ko", "next": "/"}, headers={"origin": "https://evil.test"})
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("set-cookie", response.headers)

    def test_own_origin_is_accepted(self):
        """FEAT-45: the app's own Origin passes the origin check."""
        response = self.post({"csrf": "csrf", "locale": "ko", "next": "/"}, headers={"origin": "https://pi.test"})
        self.assertEqual(response.status_code, 303)

    def test_no_session_is_401(self):
        """FEAT-45: without a session the route is 401 and sets no cookie, as /logout does."""
        self.client.cookies.clear()
        response = self.post({"csrf": "csrf", "locale": "ko", "next": "/"})
        self.assertEqual(response.status_code, 401)
        self.assertNotIn("set-cookie", response.headers)

    def test_session_is_checked_before_origin(self):
        """FEAT-45: a foreign Origin with no session is 401 (session first), the same order as /logout."""
        self.client.cookies.clear()
        response = self.post({"csrf": "csrf", "locale": "ko", "next": "/"}, headers={"origin": "https://evil.test"})
        self.assertEqual(response.status_code, 401)

    def test_unknown_locale_is_400(self):
        """FEAT-45: an unavailable, empty, missing or odd locale is 400 and sets no cookie."""
        for locale in ("xx", "", "pseudo", "../en", "ko\r\nSet-Cookie: a=b"):
            response = self.post({"csrf": "csrf", "locale": locale, "next": "/"})
            self.assertEqual(response.status_code, 400, repr(locale))
            self.assertNotIn("set-cookie", response.headers)
        self.assertEqual(self.post({"csrf": "csrf", "next": "/"}).status_code, 400)

    def test_unknown_locale_message_is_translated(self):
        """FEAT-45: the 400 detail is a plain str produced through t() in the viewer's language."""
        with patch.dict(os.environ, {"LLMCORD_I18N_PSEUDO": "1"}):
            self.client.cookies.set(web_locale.COOKIE, "pseudo")
            detail = self.post({"csrf": "csrf", "locale": "xx", "next": "/"}).json()["detail"]
        self.assertIsInstance(detail, str)
        self.assertTrue(detail.startswith("[!! "), detail)

    def test_open_redirect_attempts_land_on_root(self):
        """FEAT-45: an unsafe next value still sets the cookie but redirects to '/', never off site."""
        for value in ("//evil.example", "https://evil.example", "/\\evil", "javascript:x", "", "/\nfoo", "evil.example"):
            response = self.post({"csrf": "csrf", "locale": "ko", "next": value})
            self.assertEqual(response.status_code, 303, repr(value))
            self.assertEqual(response.headers["location"], "/", repr(value))

    def test_missing_next_lands_on_root(self):
        """FEAT-45: no next field redirects to '/'."""
        response = self.post({"csrf": "csrf", "locale": "ko"})
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/")

    def test_get_is_not_allowed(self):
        """FEAT-45: the switch is POST only."""
        self.assertEqual(self.client.get("/lang", follow_redirects=False).status_code, 405)


class HttpErrorLanguageTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"LLMCORD_I18N_PSEUDO": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)
        transport, _ = discord_transport(admin_guilds=(1,))
        self.app = create_app(":memory:", "https://pi.test", "client", "secret", "bot",
                              httpx.AsyncClient(transport=transport), enable_dashboard=False,
                              config_path="tests/nonexistent-config.yaml")
        self.client = TestClient(self.app, base_url="https://pi.test")
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.client.cookies.set("llmcord_session", install_session(self.app))

    def test_avatar_404_is_pseudo_translated_for_a_pseudo_cookie(self):
        """FEAT-45: an HTTPException detail from web.py is pseudo-translated for a pseudo cookie."""
        self.client.cookies.set(web_locale.COOKIE, "pseudo")
        response = self.client.get("/guild/1/characters/999999/avatar")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], i18n.pseudo("Avatar not found"))

    def test_logout_token_error_is_pseudo_translated(self):
        """FEAT-45: the /logout form-token error follows the cookie language."""
        self.client.cookies.set(web_locale.COOKIE, "pseudo")
        response = self.client.post("/logout", data={"csrf": "wrong"}, follow_redirects=False)
        self.assertEqual(response.json()["detail"], i18n.pseudo("Invalid form token"))

    def test_auth_error_is_pseudo_translated(self):
        """FEAT-45: an auth.py error (no session) follows the cookie language."""
        self.client.cookies.clear()
        self.client.cookies.set(web_locale.COOKIE, "pseudo")
        response = self.client.get("/guild/1/characters/1/avatar")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], i18n.pseudo("Sign in with Discord"))

    def test_accept_language_pseudo_header_is_not_special(self):
        """FEAT-45: without the cookie, an English viewer still gets plain English text."""
        response = self.client.get("/guild/1/characters/999999/avatar", headers={"accept-language": "en-US,en;q=0.9"})
        self.assertEqual(response.json()["detail"], "Avatar not found")

    def test_english_is_byte_identical_without_a_cookie(self):
        """FEAT-45: no cookie and no Accept-Language keeps the exact English detail."""
        response = self.client.get("/guild/1/characters/999999/avatar")
        self.assertEqual(response.json()["detail"], "Avatar not found")
        self.client.cookies.clear()
        self.assertEqual(self.client.post("/logout", data={"csrf": "x"}).json()["detail"], "Sign in with Discord")


class PageLanguageTests(unittest.TestCase):
    """The language NiceGUI puts on the page (Quasar language pack and <html lang>)."""

    def setUp(self):
        patcher = patch.dict(os.environ, {"LLMCORD_I18N_PSEUDO": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app, self.client = shared_dashboard()
        self.ident = f"lang-{self.id().rsplit('.', 1)[-1]}"
        install_session(self.app, self.ident)
        self.client.cookies.set("llmcord_session", self.ident)

    def html(self, path, **headers):
        response = self.client.get(path, headers={"accept-encoding": "identity", **headers})
        self.assertEqual(response.status_code, 200, path)
        return response.text

    def test_default_page_is_en_us(self):
        """FEAT-45: with no cookie and no query the page is lang="en-US"."""
        self.assertIn('lang="en-US"', self.html("/admin/guild/1"))

    def test_lang_query_ko(self):
        """FEAT-45: ?lang=ko renders the guild page with lang="ko-KR"."""
        self.assertIn('lang="ko-KR"', self.html("/admin/guild/1?lang=ko"))

    def test_lang_query_en(self):
        """FEAT-45: ?lang=en renders lang="en-US" even when Accept-Language says Korean."""
        self.assertIn('lang="en-US"', self.html("/admin/guild/1?lang=en", **{"accept-language": "ko"}))

    def test_pseudo_maps_to_en_us(self):
        """FEAT-45: the pseudo locale uses Quasar's en-US pack."""
        self.assertIn('lang="en-US"', self.html("/admin/guild/1?lang=pseudo"))

    def test_cookie_sets_the_page_language(self):
        """FEAT-45: the language cookie sets the page language on the servers page."""
        self.client.cookies.set(web_locale.COOKIE, "ko")
        self.assertIn('lang="ko-KR"', self.html("/admin/"))

    def test_accept_language_sets_the_page_language(self):
        """FEAT-45: Accept-Language ko sets lang="ko-KR" on the guild page."""
        self.assertIn('lang="ko-KR"', self.html("/admin/guild/1", **{"accept-language": "ko-KR,ko;q=0.9"}))

    def test_operator_page_follows_the_language(self):
        """FEAT-45: the operator page binds the viewer's language too."""
        self.app.state.operator_ids = frozenset({4})
        self.addCleanup(setattr, self.app.state, "operator_ids", frozenset())
        self.assertIn('lang="ko-KR"', self.html("/admin/operator?lang=ko"))

    def test_signed_out_sign_in_card_is_translated(self):
        """FEAT-45: the signed-out sign-in page is in the viewer's language (pseudo here) and html lang follows."""
        self.client.cookies.clear()
        page = self.html("/admin/?lang=pseudo")
        self.assertIn(i18n.pseudo("Sign in with Discord"), page)
        self.assertIn('lang="en-US"', page)
        korean = self.html("/admin/", **{"accept-language": "ko"})
        self.assertIn('lang="ko-KR"', korean)

    def test_switcher_next_keeps_the_admin_mount_and_drops_lang(self):
        """FEAT-45: the hidden /lang form's next is the mounted page path with its query, without the lang parameter."""
        page = self.html("/admin/guild/1?tab=lore&lang=ko")
        self.assertIn("/admin/guild/1?tab=lore", page)
        self.assertNotIn("tab=lore&amp;lang", page)
        self.assertNotIn("tab=lore&lang", page)
        self.app.state.operator_ids = frozenset({4})
        self.addCleanup(setattr, self.app.state, "operator_ids", frozenset())
        self.assertIn("ll-lang-form", self.html("/admin/operator?lang=ko"))

    def test_signed_out_english_is_unchanged(self):
        """FEAT-45: the signed-out page in English still has the plain sign-in text."""
        self.client.cookies.clear()
        page = self.html("/admin/")
        self.assertIn("Sign in with Discord", page)
        self.assertNotIn("[!!", page)


if __name__ == "__main__":
    unittest.main()
