import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nicegui.page import page as NiceGUIPage

from llmcord_core.i18n import web_locale

ROOT = Path(__file__).resolve().parent.parent
AVAIL = ("en", "ko", "ja")

SCRIPT = """
import json
from fastapi import FastAPI
from fastapi.testclient import TestClient
from nicegui import ui
from llmcord_core.i18n import web_locale

app = FastAPI()

@ui.page("/probe")
def probe():
    web_locale.bind_client(ui.context.client, "ko")
    ui.label("probe")

@ui.page("/plain")
def plain():
    ui.label("plain")

ui.run_with(app)
out = {}
with TestClient(app) as client:
    for path in ("/probe", "/plain"):
        out[path] = client.get(path).text
print("JSON:" + json.dumps(out))
"""


class RequestLocaleTests(unittest.TestCase):
    """FEAT-43: request_locale order is cookie, account, Accept-Language."""

    def rl(self, **kw):
        return web_locale.request_locale(available=AVAIL, **kw)

    def test_cookie_beats_account_beats_header(self):
        self.assertEqual(self.rl(cookie="ja", account_locale="ko", accept_language="en"), "ja")
        self.assertEqual(self.rl(account_locale="ko", accept_language="ja"), "ko")
        self.assertEqual(self.rl(accept_language="ja;q=0.5, ko;q=0.9"), "ko")

    def test_unavailable_cookie_falls_through(self):
        self.assertEqual(self.rl(cookie="fr", account_locale="ko"), "ko")
        self.assertEqual(self.rl(cookie="fr", accept_language="ja"), "ja")

    def test_malformed_header_and_nothing(self):
        self.assertEqual(self.rl(accept_language=";;;, q=1, *, ko;q=abc"), "en")
        self.assertEqual(self.rl(), "en")
        self.assertEqual(self.rl(accept_language=""), "en")

    def test_regional_header_maps_to_language(self):
        self.assertEqual(self.rl(accept_language="ko-KR,ko;q=0.9,en;q=0.8"), "ko")

    def test_constants(self):
        self.assertEqual(web_locale.COOKIE, "llmcord_lang")
        self.assertEqual(web_locale.COOKIE_MAX_AGE, 365 * 86400)


class QuasarLanguageTests(unittest.TestCase):
    """FEAT-43: our locale codes map onto NiceGUI's Language values."""

    def test_mapping(self):
        for loc, want in (("en", "en-US"), ("ko", "ko-KR"), ("ja", "ja"), ("zh_TW", "zh-TW"), ("zh_CN", "zh-CN"), ("pseudo", "en-US"), ("xx", "en-US")):
            self.assertEqual(web_locale.quasar_language(loc), want, loc)


class BindClientTests(unittest.TestCase):
    """FEAT-43: bind_client localizes one client without mutating the shared page."""

    def test_copies_page(self):
        original = NiceGUIPage("/x")
        original.language = "en-US"
        client = SimpleNamespace(page=original)
        web_locale.bind_client(client, "ko")
        self.assertIsNot(client.page, original)
        self.assertEqual(client.page.language, "ko-KR")
        self.assertEqual(original.language, "en-US")
        self.assertEqual(client.llmcord_locale, "ko")

    def test_current_locale_outside_client_is_en(self):
        self.assertEqual(web_locale.current_locale(), "en")
        self.assertEqual(web_locale.current().locale, "en")


class RenderSpikeTests(unittest.TestCase):
    """FEAT-43 spike (NiceGUI 3.17.1): a bound client renders html lang and loads the Quasar pack."""

    def test_probe_page_is_korean_and_plain_page_is_not(self):
        with tempfile.TemporaryDirectory() as home:
            out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, cwd=ROOT, timeout=90,
                                 env={**__import__("os").environ, "HOME": home, "NICEGUI_STORAGE_PATH": home})
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        pages = json.loads(out.stdout.split("JSON:", 1)[1])
        probe, plain = pages["/probe"], pages["/plain"]
        self.assertIn('<html lang="ko-KR"', probe)
        self.assertIn("ko-KR.umd.prod.js", probe)
        self.assertIn("en-US.umd.prod.js", plain)
        self.assertNotIn("ko-KR", plain)


if __name__ == "__main__":
    unittest.main()
