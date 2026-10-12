"""MNT-81 d (D39): opt-in browser test of the dashboard's own 404 page (same harness and gating as
test_dashboard_browser_i18n.py; the fixture server runs with LLMCORD_I18N_PSEUDO=1)."""
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import httpx

from llmcord_core import i18n

SESSION = 'browser-lang-session'


@unittest.skipUnless(os.environ.get('LLMCORD_BROWSER_TESTS') == '1', 'Set LLMCORD_BROWSER_TESTS=1 to run browser integration tests')
class DashboardNotFoundBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import trustme
        from playwright.sync_api import sync_playwright
        cls.temp = tempfile.TemporaryDirectory()
        folder = Path(cls.temp.name)
        cert = trustme.CA().issue_cert('localhost')
        cert.private_key_pem.write_to_path(folder / 'key.pem')
        cert.cert_chain_pems[0].write_to_path(folder / 'cert.pem')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        cls.url = f'https://localhost:{port}'
        cls.log = (folder / 'server.log').open('w')
        env = {**os.environ, 'LLMCORD_TEST_PORT': str(port), 'LLMCORD_TEST_KEY': str(folder / 'key.pem'),
               'LLMCORD_TEST_CERT': str(folder / 'cert.pem'), 'LLMCORD_I18N_PSEUDO': '1'}
        cls.process = subprocess.Popen([sys.executable, '-m', 'tests.dashboard_server'], cwd=Path(__file__).resolve().parents[1], env=env,
                                       stdout=cls.log, stderr=cls.log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        for _ in range(80):
            try:
                if httpx.get(cls.url + '/_test/state', verify=False, timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        else:
            cls.process.terminate()
            cls.log.close()
            raise RuntimeError((folder / 'server.log').read_text())
        cls.playwright = sync_playwright().start()
        executable = os.environ.get('LLMCORD_BROWSER_EXECUTABLE')
        cls.browser = cls.playwright.chromium.launch(
            executable_path=executable,
            channel=None if executable else os.environ.get('LLMCORD_BROWSER_CHANNEL', 'msedge' if os.name == 'nt' else None),
            headless=True)
        cls.context = cls.browser.new_context(ignore_https_errors=True, viewport={'width': 1400, 'height': 1000}, locale='en-US')
        cls.context.add_cookies([{'name': 'llmcord_session', 'value': SESSION, 'url': cls.url, 'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
        cls.page = cls.context.new_page()

    @classmethod
    def tearDownClass(cls):
        cls.context.request.post(cls.url + '/_test/stop')
        cls.context.close()
        cls.browser.close()
        cls.playwright.stop()
        cls.process.wait(timeout=20)
        cls.log.close()
        cls.temp.cleanup()

    def set_language(self, value):
        self.context.clear_cookies(name='llmcord_lang')
        if value:
            self.context.add_cookies([{'name': 'llmcord_lang', 'value': value, 'url': self.url, 'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])

    def visit(self, path):
        """Returns (status, heading text, visible body text, link hrefs) once the page is rendered."""
        response = self.page.goto(self.url + path)
        heading = self.page.get_by_role('heading').first
        heading.wait_for(timeout=5000)
        hrefs = self.page.eval_on_selector_all('a[href]', 'els => els.map(e => e.getAttribute("href"))')
        return response.status, heading.inner_text(), self.page.locator('body').inner_text(), hrefs

    def test_unknown_and_forbidden_paths_get_the_same_translated_404(self):
        """MNT-81 d: an unknown /admin path and a page the viewer may not open (the operator page for a non-operator) both answer 404 with the same page and heading, in English."""
        self.set_language('en')
        unknown = self.visit('/admin/no-such-page')
        forbidden = self.visit('/admin/operator')
        self.assertEqual(unknown[0], 404)
        self.assertEqual(forbidden[0], 404)
        self.assertEqual(unknown[1], 'Page not found')
        self.assertEqual(unknown[1:], forbidden[1:])
        self.assertTrue(any(h.rstrip('/').endswith('/admin') for h in unknown[3]), unknown[3])

    def test_404_heading_is_pseudo_when_pseudo_language_is_selected(self):
        """MNT-81 d: with the pseudo language selected both 404 cases show the pseudo-translated heading and still match."""
        self.set_language('pseudo')
        unknown = self.visit('/admin/no-such-page')
        forbidden = self.visit('/admin/operator')
        self.assertEqual(unknown[0], 404)
        self.assertEqual(forbidden[0], 404)
        self.assertEqual(unknown[1], i18n.pseudo('Page not found'))
        self.assertEqual(unknown[1:], forbidden[1:])

    def test_404_does_not_reveal_guild_names_or_ids(self):
        """MNT-81 d: the hidden operator page 404 shows nothing from the request path or the viewer servers."""
        self.set_language('en')
        _, _, body, _ = self.visit('/admin/operator')
        self.assertNotIn('Test server', body)
        self.assertNotIn('operator', body)
