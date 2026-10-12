"""FEAT-45 (D39, batch 2b): opt-in browser test of the dashboard language switch (same harness and gating as
test_dashboard_browser.py; the fixture server runs with LLMCORD_I18N_PSEUDO=1)."""
import os
import re
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
TABS = ('setup', 'characters', 'lore', 'imports', 'prompts', 'monitoring', 'currency', 'games', 'shop')
# MNT-75 automation: things that are never translated, each tied to what produces it (tests/dashboard_server.py). Any other run
# of 3+ ASCII letters outside a pseudo marker is reported. Words match case-insensitively; NEVER_TRANSLATED_NODES match a whole
# text node (after pseudo markers are removed) so a real label cannot hide behind a fixture word.
NEVER_TRANSLATED = frozenset("""
alice cheerful courier
scene assets
magic ball fortune cookie tarot
discord nicegui tailscale openai json png jpg url usd utc api cfg
""".split())
# fixture: character Alice ("A cheerful courier"); channels scene and assets; built-in shop catalog item names (Magic 8-ball,
# Fortune cookie, Tarot); brand, format and unit names.
NEVER_TRANSLATED_NODES = tuple(re.compile(pattern) for pattern in (
    r'World',                                        # the fixture's space name
    r'fixture',                                      # lorebook entry key
    r'Fixture entry \d+',                            # lorebook entry content
    r'(fixture|second)-model( · —| ·)?',            # model names in the usage list
    r'FIXTURE-[A-Z-]+|Provider returned 500 FIXTURE-ERROR-DETAIL',  # stored request/response/error text
    r'Log (request|reply) \d+',                      # stored model-call log text
    r'user\d+',                                      # fixture member names
    r'Seed grant \d+',                               # ledger reason typed by the fixture
    r'Aria',                                         # member nickname from the mock Discord API
    r'.*Dice',                                       # built-in shop item name with its emoji (stored catalog name)
    r'You rolled a \d \(1-6\)\.',                    # stored use-log text (SUSPECTED untranslated, see report)
))
COLLECT_TEXT = """
() => {
  const root = [...document.querySelectorAll('.q-tab-panel')].find(p => p.offsetParent !== null);
  if (!root) return null;
  const out = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const text = node.textContent.trim();
    const parent = node.parentElement;
    if (!text || !parent || parent.closest('script, style, textarea, code, pre, kbd, option, .q-icon, .material-icons')) continue;
    const box = parent.getBoundingClientRect();
    const style = getComputedStyle(parent);
    if (style.visibility === 'hidden' || style.display === 'none' || (box.width === 0 && box.height === 0)) continue;
    out.push(text);
  }
  return out;
}
"""


@unittest.skipUnless(os.environ.get('LLMCORD_BROWSER_TESTS') == '1', 'Set LLMCORD_BROWSER_TESTS=1 to run browser integration tests')
class DashboardLanguageBrowserTests(unittest.TestCase):
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

    def load(self, path):
        self.page.goto(self.url + path)
        self.page.wait_for_function('window.did_handshake === true && window.socket?.connected === true')

    def choose(self, name):
        from playwright.sync_api import expect
        self.page.locator('button.ll-account').click()  # its aria-label is translated, so select by class
        menu = self.page.locator('.ll-menu')
        expect(menu).to_be_visible()
        menu.get_by_role('button', name=name, exact=True).click()

    def lang_cookie(self):
        return next((c for c in self.context.cookies(self.url) if c['name'] == 'llmcord_lang'), None)

    def test_switch_to_pseudo_and_back_to_english(self):
        """FEAT-45: the account menu switches the dashboard to the pseudo language (labels, html lang, cookie) and English restores the exact labels."""
        from playwright.sync_api import expect
        pseudo_label = i18n.pseudo('Characters')
        self.assertIsNone(self.lang_cookie())
        self.load('/admin/guild/1')
        expect(self.page.get_by_role('tab', name='Characters', exact=True)).to_be_visible()
        self.choose('Pseudo-locale')
        expect(self.page.get_by_text(pseudo_label, exact=True).first).to_be_visible()
        expect(self.page.get_by_role('tab', name='Characters', exact=True)).to_have_count(0)
        self.page.wait_for_function('window.did_handshake === true && window.socket?.connected === true')
        self.assertEqual(self.page.evaluate('document.documentElement.lang'), 'en-US')
        self.assertIn('/guild/1', self.page.url)
        cookie = self.lang_cookie()
        self.assertEqual(cookie['value'], 'pseudo')
        self.assertTrue(cookie['secure'])
        self.assertTrue(cookie['httpOnly'])
        self.assertEqual(cookie['sameSite'], 'Lax')
        self.page.reload()
        expect(self.page.get_by_text(pseudo_label, exact=True).first).to_be_visible()
        self.choose(i18n.display_name('en'))
        expect(self.page.get_by_role('tab', name='Characters', exact=True)).to_be_visible()
        expect(self.page.get_by_text(pseudo_label, exact=True)).to_have_count(0)
        self.assertEqual(self.lang_cookie()['value'], 'en')
        self.assertEqual(self.page.evaluate('document.documentElement.lang'), 'en-US')

    def test_current_language_is_marked(self):
        """FEAT-45: the account menu marks the current language with aria-current=true on exactly one button."""
        from playwright.sync_api import expect
        self.load('/admin/guild/1')
        self.page.locator('button.ll-account').click()  # its aria-label is translated, so select by class
        menu = self.page.locator('.ll-menu')
        expect(menu.locator('button[form="ll-lang-form"]').first).to_be_visible()
        expect(menu.locator('[aria-current="true"]')).to_have_count(1)
        expect(menu.get_by_role('button', name='Sign out')).to_be_visible()

    def test_bot_language_saves_to_store_with_audit(self):
        """FEAT-44: Server settings has a Bot language select; choosing a language and saving stores it and audits settings.language, and the follow option restores ''."""
        from playwright.sync_api import expect
        import json
        state = lambda: httpx.get(self.url + '/_test/state', verify=False, timeout=5).json()
        def wait_for(predicate):
            for _ in range(50):
                if predicate():
                    return
                self.page.wait_for_timeout(100)
            self.fail('Expected database state was not reached')
        def save(option):
            select = self.page.get_by_label('Bot language', exact=True)
            select.wait_for(timeout=5000)
            select.click()
            self.page.get_by_role('option', name=option, exact=True).click()
            self.page.get_by_role('region', name='Unsaved changes').get_by_role('button', name='Save changes', exact=True).click()
            self.page.get_by_text('Server settings saved', exact=True).wait_for(timeout=5000)
        self.load('/admin/guild/1')
        self.page.get_by_role('tab', name='Server setup', exact=True).click()
        self.assertEqual(state()['guild_language'], '')
        save(i18n.display_name('ko'))
        wait_for(lambda: state()['guild_language'] == 'ko')
        rows = [r for r in state()['audit'] if r['action'] == 'settings.language']
        self.assertEqual([json.loads(r['detail_json']) for r in rows], [{'language': 'ko'}])
        save("Follow each member's Discord language")
        wait_for(lambda: state()['guild_language'] == '')
        expect(self.page.get_by_label('Bot language', exact=True)).to_be_visible()


    def pseudo_texts(self, tab):
        self.load(f'/admin/guild/1?tab={tab}')
        self.page.wait_for_function(f'(() => {{ const t = ({COLLECT_TEXT})(); return t !== null && t.length > 1; }})()')
        texts = None
        for _ in range(40):  # sample until two consecutive reads agree, so a panel that is still filling is not scanned
            current = self.page.evaluate(COLLECT_TEXT)
            if current == texts:
                return current
            texts = current
            self.page.wait_for_timeout(250)
        return texts

    def test_pseudo_marks_visible_text_on_every_tab(self):
        """MNT-75 automation: with pseudo selected every top-level tab shows marked text and no unmarked plain-English text node of 3+ letters."""
        self.context.add_cookies([{'name': 'llmcord_lang', 'value': 'pseudo', 'url': self.url, 'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
        self.addCleanup(self.context.clear_cookies, name='llmcord_lang')
        offenders = {}
        for tab in TABS:
            texts = self.pseudo_texts(tab)
            self.assertTrue(any('[!!' in text for text in texts), f'{tab}: no pseudo-marked text in {texts[:5]}')
            bad = []
            for text in texts:
                rest = re.sub(r'\[!!.*?!!\]', ' ', text, flags=re.S)
                if rest.lstrip().startswith('/'):  # slash commands such as /admin ...
                    continue
                if any(node.fullmatch(' '.join(rest.split())) for node in NEVER_TRANSLATED_NODES):
                    continue
                if [w for w in re.findall(r'[A-Za-z]{3,}', rest) if w.lower() not in NEVER_TRANSLATED]:
                    bad.append(text if len(text) <= 80 else text[:77] + '...')
            if bad:
                offenders[tab] = bad
        self.assertEqual(offenders, {}, 'untranslated visible text under pseudo: ' + '; '.join(f'{k}: {v}' for k, v in offenders.items()))
