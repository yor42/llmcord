"""Opt-in browser integration tests against a disposable mocked Discord server."""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import httpx


@unittest.skipUnless(os.environ.get('LLMCORD_BROWSER_TESTS') == '1', 'Set LLMCORD_BROWSER_TESTS=1 to run browser integration tests')
class DashboardBrowserTests(unittest.TestCase):
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
        env = {**os.environ, 'LLMCORD_TEST_PORT': str(port), 'LLMCORD_TEST_KEY': str(folder / 'key.pem'), 'LLMCORD_TEST_CERT': str(folder / 'cert.pem')}
        cls.process = subprocess.Popen([sys.executable, '-m', 'tests.dashboard_server'], cwd=Path(__file__).resolve().parents[1], env=env, stdout=cls.log, stderr=cls.log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
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
        cls.context = cls.browser.new_context(ignore_https_errors=True, viewport={'width': 1400, 'height': 1000})
        cls.context.add_cookies([{'name': 'llmcord_session', 'value': 'browser-test-session', 'url': cls.url, 'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
        cls.page = cls.context.new_page()
        cls.errors = []
        cls.page.on('pageerror', lambda e: cls.errors.append(e.stack or str(e)))

    @classmethod
    def tearDownClass(cls):
        cls.context.request.post(cls.url + '/_test/stop')
        cls.context.close()
        cls.browser.close()
        cls.playwright.stop()
        cls.process.wait(timeout=20)
        cls.log.close()
        cls.temp.cleanup()

    def state(self):
        return self.context.request.get(self.url + '/_test/state').json()

    def tearDown(self):
        if any(test is self for test, _ in self._outcome.result.failures + self._outcome.result.errors):
            artifacts = Path(__file__).resolve().parents[1] / '.test-artifacts'
            artifacts.mkdir(exist_ok=True)
            self.page.screenshot(path=str(artifacts / 'failure.png'))
            (artifacts / 'server.log').write_text((Path(self.temp.name) / 'server.log').read_text(encoding='utf-8', errors='replace'))
            (artifacts / 'browser-errors.json').write_text(json.dumps(self.errors, indent=2))
            (artifacts / 'state.json').write_text(json.dumps(self.state()))
            (artifacts / 'lore-drag.json').write_text(json.dumps(self.page.evaluate("""() => ({
                started: window.lastLoreDragStarted ?? null, ended: window.lastLoreDrop ?? null,
                lists: [...document.querySelectorAll('.lore-drop-zone')].map(element => ({
                    id: element.id, owner: {...element.dataset}, bounds: element.getBoundingClientRect().toJSON(),
                })),
            })"""), indent=2))

    def wait_for(self, predicate):
        for _ in range(50):
            if predicate():
                return
            self.page.wait_for_timeout(100)
        self.fail('Expected database state was not reached')

    def lore_idle(self, expected_owners=None):
        self.page.evaluate("""async () => {
            const { Sortable } = await import('nicegui-sortable');
            window.loreTestSortable = Sortable;
        }""")
        # Playwright polls synchronous predicates; a Promise is truthy even
        # when its eventual result says the old board is not ready yet.
        self.page.wait_for_function("""expected => {
            const Sortable = window.loreTestSortable;
            const root = document.querySelector('.lore-workspace');
            const lists = document.querySelectorAll('.lore-drop-zone');
            return root && !root.inert && lists.length === 2 && [...lists].every(list => {
                const sortable = Sortable.get(list);
                const side = list.classList.contains('lore-drop-left') ? 'left' : 'right';
                const owner = expected?.[side];
                return list.dataset.dragReady === 'true' && sortable?.el === list &&
                    !sortable.option('disabled') && (!owner ||
                    (list.dataset.ownerKind === owner[0] && list.dataset.ownerId === String(owner[1])));
            });
        }""", arg=expected_owners)

    def choose_lore_owner(self, side, label, kind, ident):
        self.page.get_by_label(f'{side.title()} owner', exact=True).click()
        self.page.get_by_role('option', name=label, exact=True).click()
        # The select changes before the server authorizes and redraws the board.
        # Old lists can still report dragReady during that interval.
        self.lore_idle({side: [kind, ident]})

    def drag_lore(self, tile, destination, edge=False, require_started=True):
        self.page.evaluate("""async () => {
            window.lastLoreDrop = null;
            window.lastLoreDragStarted = null;
            const { Sortable } = await import('nicegui-sortable');
            for (const element of document.querySelectorAll('.lore-drop-zone')) {
                const sortable = Sortable.get(element);
                if (!sortable || sortable.testWrapped) continue;
                const start = sortable.option('onStart');
                sortable.option('onStart', event => {
                    window.lastLoreDragStarted = {source: {...event.from.dataset}, entry: {...event.item.dataset}};
                    start(event);
                });
                const original = sortable.option('onEnd');
                sortable.option('onEnd', event => {
                    window.lastLoreDrop = {source: {...event.from.dataset}, target: {...event.to.dataset}, entry: {...event.item.dataset}};
                    original(event);
                });
                sortable.testWrapped = true;
            }
        }""")
        destination.evaluate("element => window.scrollBy(0, element.getBoundingClientRect().top - 140)")
        tile.locator('.drag-handle').scroll_into_view_if_needed()
        source_box = tile.locator('.drag-handle').bounding_box()
        target_box = destination.bounding_box()
        if source_box is None or target_box is None:
            if require_started:
                self.fail('Lore drag source or destination disappeared before the gesture')
            # During the deliberately overlapping attempt, a committed move
            # may already be replacing the board. There is then nothing to grab.
            return
        self.page.mouse.move(source_box['x'] + source_box['width'] / 2, source_box['y'] + source_box['height'] / 2)
        self.page.mouse.down()
        try:
            self.page.mouse.move(source_box['x'] + source_box['width'] / 2 + 12,
                                 source_box['y'] + source_box['height'] / 2, steps=3)
            if require_started:
                self.page.wait_for_function('window.lastLoreDragStarted !== null', timeout=5000)
            self.page.mouse.move(target_box['x'] + (5 if edge else 30), target_box['y'] + (5 if edge else 30), steps=12)
            self.page.wait_for_timeout(150)
        finally:
            self.page.mouse.up()
        if require_started:
            self.page.wait_for_function('window.lastLoreDrop !== null', timeout=5000)

    def select_lore(self, tile):
        from playwright.sync_api import expect
        checkbox = tile.get_by_role('checkbox', name='Select entry', exact=True)
        checkbox.click()
        expect(checkbox).to_be_checked()

    def test_dashboard_workflows_and_live_authorization(self):
        from urllib.parse import urlparse, parse_qs
        # Exercise the rendered login link, including NiceGUI's mount prefix.
        anonymous = self.browser.new_context(ignore_https_errors=True)
        try:
            login_page = anonymous.new_page()
            anonymous.route('https://discord.com/oauth2/authorize?*',
                            lambda route: route.fulfill(status=200, body='Mock Discord authorization'))
            login_page.goto(self.url + '/admin/')
            # Wait only for the redirect to commit: the assertions need the URL and cookies, not the
            # mocked page's load event, which intermittently never fired on CI (30 s timeout).
            with login_page.expect_navigation(url='https://discord.com/oauth2/authorize?*', wait_until='commit'):
                login_page.get_by_role('link', name='Sign in with Discord', exact=True).click()
            query = parse_qs(urlparse(login_page.url).query)
            self.assertEqual(query['redirect_uri'], [self.url + '/auth/callback'])
            cookies = {c['name']: c for c in anonymous.cookies(self.url)}
            self.assertEqual(cookies['llmcord_oauth_state']['value'], query['state'][0])
            self.assertTrue(cookies['llmcord_oauth_state']['secure'])
            self.assertTrue(cookies['llmcord_oauth_state']['httpOnly'])
        finally:
            anonymous.close()
        page = self.page
        page.goto(self.url + '/admin/')
        page.get_by_role('link', name='Test server', exact=True).click()
        page.wait_for_url(self.url + '/admin/guild/1')
        page.get_by_role('link', name='llmcord / Servers', exact=True).click()
        page.wait_for_url(self.url + '/admin/')
        with page.expect_navigation() as navigation:
            page.get_by_role('link', name='Test server', exact=True).click()
        page.wait_for_url(self.url + '/admin/guild/1')
        response = navigation.value
        self.assertEqual(response.status, 200)
        self.assertIn('nonce-', response.headers['content-security-policy'])
        page.get_by_text('Server administration', exact=True).wait_for()
        page.get_by_text('fixture-model', exact=True).wait_for()
        # Rendered controls precede the connection that delivers their events.
        page.wait_for_function('window.did_handshake === true && window.socket?.connected === true')
        world_panel = page.locator('.space-card').filter(has=page.get_by_text('World · world', exact=True))
        world_panel.get_by_text('World · world', exact=True).click()
        world_panel.get_by_label('World guidelines', exact=True).fill('The setting is a courier guild. Treat users as guild members.')
        world_panel.get_by_role('button', name='Save world guidelines', exact=True).click()
        self.wait_for(lambda: any(row['kind'] == 'space' and 'courier guild' in row['content'] for row in self.state()['guidelines']))
        channel_panel = page.locator('.channel-card').filter(has=page.get_by_text('#scene', exact=True))
        channel_panel.get_by_label('Channel guidelines', exact=True).fill('Occasional fourth-wall jokes are welcome; keep them brief.')
        channel_panel.get_by_role('button', name='Save channel guidelines', exact=True).click()
        self.wait_for(lambda: any(row['kind'] == 'channel' and 'fourth-wall' in row['content'] for row in self.state()['guidelines']))
        page.get_by_label('Space name', exact=True).fill('Browser world')
        page.get_by_role('button', name='Create space', exact=True).click()
        self.wait_for(lambda: any(r['name'] == 'Browser world' for r in self.state()['spaces']))
        unused = page.locator('.space-card').filter(has=page.get_by_text('Browser world · world', exact=True))
        unused.get_by_text('Browser world · world', exact=True).click()
        unused.get_by_role('button', name='Delete world', exact=True).click()
        dialog = page.get_by_role('dialog')
        dialog.get_by_text('Delete world Browser world?', exact=True).wait_for()
        dialog.get_by_role('button', name='Cancel', exact=True).click()
        self.assertTrue(any(row['name'] == 'Browser world' for row in self.state()['spaces']))
        unused.get_by_role('button', name='Delete world', exact=True).click()
        dialog.get_by_role('button', name='Delete world', exact=True).click()
        self.wait_for(lambda: not any(row['name'] == 'Browser world' for row in self.state()['spaces']))
        page.get_by_label('Private text channel', exact=True).click()
        page.get_by_role('option', name='#assets', exact=True).click()
        page.get_by_role('button', name='Save asset channel', exact=True).click()

        page.get_by_role('tab', name='Characters', exact=True).click()
        page.get_by_role('button', name='Create character', exact=True).click()
        dialog = page.get_by_role('dialog')
        dialog.get_by_role('button', name='Create', exact=True).click()
        page.get_by_text('Character name cannot be empty', exact=True).wait_for()
        dialog.get_by_label('Character name', exact=True).fill(' Alice ')
        dialog.get_by_role('button', name='Create', exact=True).click()
        page.get_by_text('This character name already exists', exact=True).wait_for()
        dialog.get_by_label('Character name', exact=True).fill(' Browser blank ')
        dialog.get_by_label('Home world', exact=True).click()
        page.get_by_role('option', name='World', exact=True).click()
        dialog.get_by_role('button', name='Create', exact=True).click()
        self.wait_for(lambda: any(row['name'] == 'Browser blank' for row in self.state()['characters']))
        blank = next(row for row in self.state()['characters'] if row['name'] == 'Browser blank')
        self.assertTrue(all(value == '' for field, value in json.loads(blank['card']).items() if field != 'name'))
        self.assertTrue(any(row['character_id'] == blank['id'] for row in self.state()['slots']))
        character = page.locator('.character-card').filter(has=page.get_by_text('Browser blank', exact=True))
        character.get_by_text('Browser blank', exact=True).click()
        character.get_by_text('Fallback static avatar', exact=True).click()
        fallback = character.locator('.fallback-avatar')
        from PIL import Image
        from io import BytesIO
        portrait = BytesIO()
        Image.new('RGB', (50, 50), 'green').save(portrait, 'PNG')
        fallback.locator('input[type=file]').set_input_files({'name': 'fallback.png', 'mimeType': 'image/png', 'buffer': portrait.getvalue()})
        page.get_by_text('Fallback avatar saved', exact=True).wait_for()
        self.wait_for(lambda: next(row for row in self.state()['characters'] if row['id'] == blank['id'])['has_static_avatar'])
        self.assertFalse(self.state()['assets'])
        self.assertFalse(any(slot['has_image'] for slot in self.state()['slots'] if slot['character_id'] == blank['id']))
        character.get_by_text('Browser blank', exact=True).click()
        character.get_by_label('Description', exact=True).fill('A character created without an import')
        character.get_by_role('button', name='Save character', exact=True).click()
        self.wait_for(lambda: json.loads(next(row for row in self.state()['characters'] if row['id'] == blank['id'])['card'])['description'] == 'A character created without an import')
        self.assertEqual(json.loads(next(row for row in self.state()['characters'] if row['id'] == blank['id'])['card'])['description'], 'A character created without an import')
        character.get_by_text('Browser blank', exact=True).click()
        character.get_by_text('Fallback static avatar', exact=True).click()
        fallback.get_by_role('button', name='Remove fallback avatar', exact=True).click()
        dialog.get_by_role('button', name='Remove fallback avatar', exact=True).click()
        self.wait_for(lambda: not next(row for row in self.state()['characters'] if row['id'] == blank['id'])['has_static_avatar'])
        character.get_by_text('Browser blank', exact=True).click()
        character.get_by_role('button', name='Delete character', exact=True).click()
        dialog.get_by_text('Delete Browser blank?', exact=True).wait_for()
        dialog.get_by_role('button', name='Cancel', exact=True).click()
        self.assertTrue(any(row['id'] == blank['id'] for row in self.state()['characters']))
        character.get_by_role('button', name='Delete character', exact=True).click()
        dialog.get_by_role('button', name='Delete permanently', exact=True).click()
        self.wait_for(lambda: not any(row['id'] == blank['id'] for row in self.state()['characters']))
        self.assertFalse(any(row['character_id'] == blank['id'] for row in self.state()['slots']))
        page.get_by_role('button', name='Create character', exact=True).wait_for()

        books_before = len(self.state()['lorebooks'])
        page.get_by_role('tab', name='Imports', exact=True).click()
        page.get_by_label('Lorebook name', exact=True).fill('Browser empty book')
        page.get_by_role('button', name='Create lorebook', exact=True).click()
        page.get_by_role('tab', name='Imports', exact=True).click()
        empty = page.locator('.q-expansion-item').filter(has=page.get_by_text('Browser empty book · server', exact=True)).first
        empty.get_by_text('Browser empty book · server', exact=True).click()
        empty.get_by_role('button', name='Delete lorebook', exact=True).click()
        dialog.get_by_text('Delete lorebook Browser empty book?', exact=True).wait_for()
        dialog.get_by_role('button', name='Cancel', exact=True).click()
        self.assertTrue(any(row['name'] == 'Browser empty book' for row in self.state()['lorebooks']))
        empty.get_by_role('button', name='Delete lorebook', exact=True).click()
        dialog.get_by_role('button', name='Delete lorebook', exact=True).click()
        self.wait_for(lambda: not any(row['name'] == 'Browser empty book' for row in self.state()['lorebooks']))
        page.get_by_label('Destination owner', exact=True).click()
        page.get_by_role('option', name='Server: Server-wide lore', exact=True).click()
        page.get_by_role('button', name='Import entries into selected owner', exact=True).click()
        data = json.dumps({'type': 'risu', 'ver': 1, 'data': [{'key': '', 'content': 'Direct guild fact', 'alwaysActive': True, 'insertorder': 37}]}).encode()
        dialog.locator('input[type=file]').set_input_files({'name': 'guild.json', 'mimeType': 'application/json', 'buffer': data})
        dialog.get_by_text('RisuAI: 1 new entries, 0 already present', exact=True).wait_for()
        dialog.get_by_role('button', name='Apply entry import', exact=True).click()
        self.wait_for(lambda: any(row['content'] == 'Direct guild fact' for row in self.state()['guild_lore']))
        page.wait_for_function("new URL(location.href).searchParams.get('owner')?.startsWith('guild:')")
        self.assertEqual(len(self.state()['lorebooks']), books_before)
        self.choose_lore_owner('left', 'World: World', 'space', 1)
        data = json.dumps({'entries': [{'content': 'Direct world fact', 'constant': True, 'order': 37}]}).encode()
        for count in (1, 0):
            page.get_by_role('button', name='Import JSON entries', exact=True).first.click()
            dialog.locator('input[type=file]').set_input_files({'name': 'world.json', 'mimeType': 'application/json', 'buffer': data})
            dialog.get_by_text(f'SillyTavern: {count} new entries, {1 - count} already present', exact=True).wait_for()
            dialog.get_by_role('button', name='Apply entry import', exact=True).click()
            self.lore_idle()
        self.assertEqual(sum(row['content'] == 'Direct world fact' for row in self.state()['lore']), 1)
        self.assertEqual(len(self.state()['lorebooks']), books_before)

        page.get_by_role('tab', name='Lore', exact=True).click()
        page.get_by_label('Page', exact=True).wait_for()
        # Defaults use World on the left and the named book on the right.
        page.get_by_role('button', name='New entry', exact=True).first.click()
        page.get_by_label('Content', exact=True).fill('Browser lore')
        page.get_by_role('button', name='Save lore', exact=True).click()
        self.wait_for(lambda: any(r['content'] == 'Browser lore' for r in self.state()['lore']))
        tile = page.locator('.lore-entry').filter(has_text='Browser lore')
        # Use the visible drag handle and target owner container.
        self.lore_idle()
        self.drag_lore(tile, page.locator('.lore-drop-right'))
        self.wait_for(lambda: any(r['content'] == 'Browser lore' for r in self.state()['books']))
        self.lore_idle()
        page.locator('.lore-drop-right .lore-entry').filter(has_text='Browser lore').wait_for()

        def add_lore(content):
            page.get_by_role('button', name='New entry', exact=True).first.click()
            page.get_by_label('Content', exact=True).fill(content)
            page.get_by_role('button', name='Save lore', exact=True).click()
            self.wait_for(lambda: any(row['content'] == content for row in self.state()['lore']))
            self.lore_idle()

        for content in ('Bulk A', 'Bulk B', 'Bulk C'):
            add_lore(content)
        left_list = page.locator('.lore-drop-left')
        right_list = page.locator('.lore-drop-right')
        # Edge drops and another attempted drag while a save is pending must
        # neither lose entries nor submit an operation against stale DOM IDs.
        # (/_test/delay-permission drops the guild-list cache so the save's guard really refetches and is slow.)
        self.assertEqual(self.context.request.post(self.url + '/_test/delay-permission').status, 200)
        self.drag_lore(left_list.locator('.lore-entry').filter(has_text='Bulk A'), right_list, edge=True)
        self.drag_lore(left_list.locator('.lore-entry').filter(has_text='Bulk B'), right_list, edge=True, require_started=False)
        self.wait_for(lambda: any(row['content'] == 'Bulk A' for row in self.state()['books']))
        self.lore_idle()
        self.assertEqual(sum(row['content'] in {'Bulk A', 'Bulk B', 'Bulk C'}
                             for row in self.state()['lore'] + self.state()['books']), 3)
        # The row shortcut works immediately after a completed drag.
        right_list.locator('.lore-entry').filter(has_text='Bulk A').get_by_role('button', name='Move left', exact=True).click()
        self.wait_for(lambda: any(row['content'] == 'Bulk A' for row in self.state()['lore']))
        self.lore_idle()
        if any(row['content'] == 'Bulk B' for row in self.state()['books']):
            right_list.locator('.lore-entry').filter(has_text='Bulk B').get_by_role('button', name='Move left', exact=True).click()
            self.wait_for(lambda: any(row['content'] == 'Bulk B' for row in self.state()['lore']))
            self.lore_idle()
        for content in ('Bulk A', 'Bulk B', 'Bulk C'):
            self.select_lore(left_list.locator('.lore-entry').filter(has_text=content))
        page.get_by_text('3 selected', exact=True).wait_for()
        page.get_by_role('button', name='Move selected right', exact=True).click()
        self.wait_for(lambda: sum(row['content'] in {'Bulk A', 'Bulk B', 'Bulk C'} for row in self.state()['books']) == 3)
        self.lore_idle()
        for content in ('Bulk A', 'Bulk B'):
            self.select_lore(right_list.locator('.lore-entry').filter(has_text=content))
        page.get_by_role('button', name='Move selected left', exact=True).click()
        self.wait_for(lambda: sum(row['content'] in {'Bulk A', 'Bulk B'} for row in self.state()['lore']) == 2)
        self.lore_idle()
        right_list.locator('.lore-entry').filter(has_text='Bulk C').get_by_role('button', name='Delete', exact=True).click()
        page.get_by_role('button', name='Delete 1 entry', exact=True).click()
        self.wait_for(lambda: not any(row['content'] == 'Bulk C' for row in self.state()['books']))
        self.lore_idle()
        for content in ('Bulk A', 'Bulk B'):
            self.select_lore(left_list.locator('.lore-entry').filter(has_text=content))
        page.get_by_role('button', name='Delete selected', exact=True).click()
        page.get_by_role('button', name='Delete 2 entries', exact=True).click()
        self.wait_for(lambda: not any(row['content'] in {'Bulk A', 'Bulk B'} for row in self.state()['lore']))
        self.lore_idle()
        # An external edit must reject the stale drag and restore the source.
        add_lore('Stale test')
        tile = left_list.locator('.lore-entry').filter(has_text='Stale test')
        key = tile.get_attribute('data-entry-key')
        self.assertEqual(self.context.request.post(self.url + '/_test/change-lore',
            data={'key': key, 'content': 'Stale test updated'}).status, 200)
        self.drag_lore(tile, right_list, edge=True)
        page.get_by_text('A selected lore entry changed; select it again before continuing', exact=True).wait_for()
        self.lore_idle()
        left_list.locator('.lore-entry').filter(has_text='Stale test updated').wait_for()
        self.assertFalse(any(row['entry_key'] == key for row in self.state()['books']))
        # Also support dropping into an empty list and page-wide selection.
        # Keep the old, ready book list visible while authorization is pending.
        # This reproduces the race seen on fast CI workers during owner changes.
        self.assertEqual(self.context.request.post(self.url + '/_test/delay-permission').status, 200)
        alice = next(row for row in self.state()['characters'] if row['name'] == 'Alice')
        self.choose_lore_owner('right', 'Character: Alice', 'character', alice['id'])
        self.drag_lore(left_list.locator('.lore-entry').filter(has_text='Stale test updated'), right_list, edge=True)
        self.wait_for(lambda: any(row['entry_key'] == key and row['scope_kind'] == 'character' for row in self.state()['lore']))
        self.lore_idle()
        self.select_lore(right_list.locator('.lore-entry'))
        page.get_by_role('button', name='Move selected left', exact=True).click()
        self.wait_for(lambda: any(row['entry_key'] == key and row['scope_kind'] == 'space' for row in self.state()['lore']))
        self.lore_idle()
        selected_count = left_list.locator('.lore-entry').count()
        page.locator('.lore-panel-left').get_by_role('button', name='Select page', exact=True).click()
        page.get_by_text(f'{selected_count} selected', exact=True).wait_for()
        page.get_by_role('button', name='Clear selection', exact=True).click()
        page.get_by_text('0 selected', exact=True).wait_for()

        page.get_by_role('tab', name='Prompt presets', exact=True).click()
        page.get_by_label('Prompt text / template', exact=True).first.fill('Respond as {{char}}. Browser custom instructions.')
        page.get_by_label('Prompt text / template', exact=True).first.press('Tab')
        cards = page.locator('.prompt-block')
        cards.first.locator('.prompt-handle').drag_to(cards.nth(2))
        page.get_by_label('Preset name', exact=True).fill('Browser preset')
        page.get_by_role('button', name='Save draft', exact=True).click()
        self.wait_for(lambda: any(r['name'] == 'Browser preset' for r in self.state()['presets']))
        page.get_by_role('button', name='Activate saved revision', exact=True).click()
        self.wait_for(lambda: bool(self.state()['active']))
        self.assertTrue(any(b['content'] == 'Respond as {{char}}. Browser custom instructions.' for b in self.state()['active_bundle']['purposes']['dialogue']))
        page.get_by_role('tab', name='Prompt presets', exact=True).click()
        page.get_by_role('button', name='Preview without a model call', exact=True).click()
        page.get_by_text('Estimated input tokens:', exact=False).wait_for()
        with page.expect_download() as download:
            page.get_by_role('button', name='Export full native bundle', exact=True).click()
        native = json.loads(Path(download.value.path()).read_text(encoding='utf-8'))
        self.assertEqual(native['format'], 'llmcord-preset')

        page.get_by_role('tab', name='Imports', exact=True).click()
        page.get_by_label('Home world', exact=True).click()
        page.get_by_role('option', name='World', exact=True).click()
        data = json.dumps({'name': 'Imported via browser', 'description': 'Imported card'}).encode()
        page.locator('input[type=file]').first.set_input_files({'name': 'card.json', 'mimeType': 'application/json', 'buffer': data})
        page.get_by_text('Preview: Imported via browser', exact=True).wait_for()
        page.get_by_role('button', name='Apply card import', exact=True).click()
        self.wait_for(lambda: any(r['name'] == 'Imported via browser' for r in self.state()['characters']))
        page.get_by_role('tab', name='Characters', exact=True).click()
        page.get_by_text('Imported via browser', exact=True).wait_for()
        character = page.locator('.q-expansion-item').filter(has=page.get_by_text('Imported via browser', exact=True)).first
        character.get_by_text('Imported via browser', exact=True).click()
        character.get_by_text('Happy', exact=True).click()
        happy = character.locator('.q-expansion-item').filter(has=page.get_by_text('Happy', exact=True)).first
        happy.get_by_label('Label', exact=True).fill('Joy')
        from PIL import Image
        from io import BytesIO
        image = BytesIO()
        Image.new('RGB', (50, 50), 'red').save(image, 'PNG')
        happy.locator('input[type=file]').set_input_files({'name': 'joy.png', 'mimeType': 'image/png', 'buffer': image.getvalue()})
        page.get_by_text('Emotion image saved', exact=True).wait_for()
        self.wait_for(lambda: any(s['label'] == 'Joy' for s in self.state()['slots']))
        page.get_by_role('tab', name='Characters', exact=True).click()
        character = page.locator('.q-expansion-item').filter(has=page.get_by_text('Imported via browser', exact=True)).first
        character.get_by_text('Imported via browser', exact=True).click()
        character.get_by_text('Joy', exact=True).click()
        joy = character.locator('.q-expansion-item').filter(has=page.get_by_text('Joy', exact=True)).first
        joy.get_by_role('button', name='Publish / repair image', exact=True).click()
        self.wait_for(lambda: bool(self.state()['assets']))
        upload_target = self.context.request.get(self.url + '/_test/uploads').json()[-1]
        upload = {'file': {'name': 'test.json', 'mimeType': 'application/json', 'buffer': b'{}'}}
        self.assertEqual(self.context.request.post(self.url + upload_target, multipart=upload).status, 403)
        self.assertEqual(self.context.request.post(self.url + upload_target, multipart=upload, headers={'X-CSRF-Token': 'browser-test-csrf', 'Origin': 'https://wrong.test'}).status, 403)
        anonymous = self.playwright.request.new_context(ignore_https_errors=True)
        self.assertEqual(anonymous.post(self.url + upload_target, multipart=upload).status, 401)
        anonymous.dispose()
        oversized = {'file': {'name': 'large.json', 'mimeType': 'application/json', 'buffer': b'x' * (8 * 1024 * 1024 + 70000)}}
        self.assertEqual(self.context.request.post(self.url + upload_target, multipart=oversized, headers={'X-CSRF-Token': 'browser-test-csrf'}).status, 413)
        page.wait_for_load_state('networkidle')
        artifacts = Path(__file__).resolve().parents[1] / '.test-artifacts'
        artifacts.mkdir(exist_ok=True)
        page.screenshot(path=str(artifacts / 'dashboard.png'))

        # Reach and exercise named-book import from the Lore workspace.
        page.get_by_role('tab', name='Lore', exact=True).click()
        page.get_by_role('button', name='Import lorebook', exact=True).click()
        page.get_by_label('Lorebook name', exact=True).fill('Browser import')
        page.get_by_role('button', name='Create lorebook', exact=True).click()
        self.wait_for(lambda: any(r['name'] == 'Browser import' for r in self.state()['lorebooks']))
        page.get_by_role('tab', name='Lore', exact=True).click()
        page.get_by_role('button', name='Import lorebook', exact=True).click()
        book = page.locator('.q-expansion-item').filter(has=page.get_by_text('Browser import · server', exact=True)).first
        book.get_by_text('Browser import · server', exact=True).click()
        content = json.dumps({'entries': {'1': {'key': ['moon'], 'content': 'Imported lorebook entry', 'enabled': True}}}).encode()
        book.locator('input[type=file]').set_input_files({'name': 'lorebook.json', 'mimeType': 'application/json', 'buffer': content})
        book.get_by_role('button', name='Apply lorebook sync', exact=True).wait_for()
        entry = book.locator('.q-expansion-item').filter(has=page.get_by_text('Incoming: Imported lorebook entry', exact=True)).last
        entry.locator('.q-item').first.click()
        book.get_by_text('Incoming: Imported lorebook entry', exact=True).wait_for()
        book.get_by_role('button', name='Apply lorebook sync', exact=True).click()
        self.wait_for(lambda: any(row['content'] == 'Imported lorebook entry' for row in self.state()['books']))
        page.get_by_role('tab', name='Lore', exact=True).click()
        page.get_by_role('button', name='Import lorebook', exact=True).click()
        book = page.locator('.q-expansion-item').filter(has=page.get_by_text('Browser import · server', exact=True)).first
        book.get_by_text('Browser import · server', exact=True).click()
        book.get_by_role('button', name='Enable in World', exact=True).click()
        book.get_by_role('button', name='Enable in World', exact=True).wait_for(state='hidden')  # panel rebuilt in place, expansion collapsed
        page.get_by_role('tab', name='Lore', exact=True).click()
        page.get_by_role('button', name='Import lorebook', exact=True).click()
        book = page.locator('.q-expansion-item').filter(has=page.get_by_text('Browser import · server', exact=True)).first
        book.get_by_text('Browser import · server', exact=True).click()
        book.get_by_role('button', name='Disable in World', exact=True).wait_for()

        risu = json.dumps({'type': 'risu', 'ver': 1, 'data': [
            {'id': '1', 'key': 'moon', 'secondkey': '', 'content': 'Imported Risu entry',
             'insertorder': 0, 'mode': 'normal', 'alwaysActive': False, 'selective': False, 'useRegex': False},
            {'id': 'folder', 'key': 'folder:1', 'content': '', 'insertorder': 100, 'mode': 'folder'},
        ]}).encode()
        book.locator('input[type=file]').set_input_files({'name': 'risu.json', 'mimeType': 'application/json', 'buffer': risu})
        book.get_by_text('RisuAI lorebook detected · 2 entries', exact=True).wait_for()
        book.get_by_role('button', name='Apply lorebook sync', exact=True).click()
        self.wait_for(lambda: any(row['content'] == 'Imported Risu entry' for row in self.state()['books']))
        imported = next(row for row in self.state()['books'] if row['content'] == 'Imported Risu entry')
        self.assertEqual(json.loads(imported['rule_json'])['order'], 0)
        self.assertFalse(json.loads(imported['rule_json'])['regex_enabled'])
        folder = next(row for row in self.state()['books'] if row['uid'] == 'folder')
        self.assertFalse(json.loads(folder['rule_json'])['enabled'])

        page.get_by_role('tab', name='Server setup', exact=True).click()
        page.get_by_label('Space name', exact=True).fill('Forbidden')
        page.wait_for_timeout(200)
        # /_test/revoke also drops cached guild lists, simulating the 300 s TTL (PERF-01 / D1) having elapsed;
        # the live event below must then be rejected on the refetch.
        self.context.request.post(self.url + '/_test/revoke')
        page.get_by_role('button', name='Create space', exact=True).click(force=True)
        page.wait_for_timeout(300)
        self.assertFalse(any(r['name'] == 'Forbidden' for r in self.state()['spaces']))
        # /_test/restore likewise drops the cached (non-admin) guild list, as if the TTL had elapsed.
        self.context.request.post(self.url + '/_test/restore')
        page.goto(self.url + '/admin/guild/1')
        page.get_by_label('Space name', exact=True).fill('Forbidden')
        page.wait_for_timeout(200)
        self.context.request.post(self.url + '/_test/expire')
        page.get_by_role('button', name='Create space', exact=True).click(force=True)
        page.wait_for_timeout(300)
        self.assertFalse(any(r['name'] == 'Forbidden' for r in self.state()['spaces']))
        self.assertFalse(self.errors, self.errors)

    def test_rejected_live_event_notifies_bound_client(self):
        """PERF-01 (fixed): a live event rejected by the permission check is dropped but the user is told why.

        A bound client with a matching cookie clicks "Create space" after (1) admin permission is revoked (403),
        (2) Discord keeps rate limiting the guild check (503), and (3) the session expires (401). Each time the
        handler must not run, and the user should see why: the 403/503 detail in a negative notification, and a
        sign-in prompt (notification or redirect) for the 401.
        """
        import re
        from playwright.sync_api import expect
        # Own context and session: this test revokes access and expires its session.
        context = self.browser.new_context(ignore_https_errors=True, viewport={'width': 1400, 'height': 1000})
        context.add_cookies([{'name': 'llmcord_session', 'value': 'browser-reject-session', 'url': self.url,
                              'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
        original_page, errors = self.page, []
        def post(path):
            self.assertEqual(context.request.post(self.url + path).status, 200)
        try:
            post('/_test/restore')
            page = self.page = context.new_page()
            page.on('pageerror', lambda e: errors.append(e.stack or str(e)))
            page.goto(self.url + '/admin/guild/1')
            page.wait_for_function('window.did_handshake === true && window.socket?.connected === true')
            page.get_by_role('tab', name='Server setup', exact=True).click()
            page.get_by_label('Space name', exact=True).fill('Rejected')
            page.wait_for_timeout(200)
            create = page.get_by_role('button', name='Create space', exact=True)
            notification = page.locator('.q-notification')
            def created():
                return any(r['name'] == 'Rejected' for r in self.state()['spaces'])

            # 403: admin permission revoked (revoke also drops the cached guild list, as if the TTL elapsed).
            post('/_test/revoke')
            create.click(force=True)
            expect(notification.filter(has_text=re.compile('administrator permission', re.I))).to_be_visible(timeout=4000)
            self.assertFalse(created())
            post('/_test/restore')

            # 503: Discord rate limits every retry of the guild check.
            post('/_test/rate-limit')
            create.click(force=True)
            expect(notification.filter(has_text=re.compile('try again shortly', re.I))).to_be_visible(timeout=4000)
            self.assertFalse(created())
            post('/_test/restore')

            # 401: the session expired. Either a notification or a redirect to a sign-in page is acceptable.
            post('/_test/expire?session=browser-reject-session')
            create.click(force=True)
            expect(page.get_by_text(re.compile('sign in', re.I)).first).to_be_visible(timeout=4000)
            self.assertFalse(created())
            self.assertFalse(errors, errors)
        finally:
            self.page = original_page
            context.request.post(self.url + '/_test/restore')
            context.close()

    def test_lore_search_is_debounced(self):
        """PERF-01 (fixed): typing in the lore search box re-renders the board once, after typing pauses, not per keystroke.

        The board loads each side through ``AdminStore.admin_entries_page`` (counted under the ``admin_entries`` keys), so one render is two counted calls.
        Ten keystrokes 30 ms apart arrive well inside a ~300 ms debounce window, so a debounced search
        renders once at the end. The bound allows two renders (four calls) for one scheduling hiccup
        mid-burst; before the fix every keystroke rendered (about ten renders, twenty calls).
        """
        # Own context and session: the workflow test revokes access and expires its session.
        context = self.browser.new_context(ignore_https_errors=True, viewport={'width': 1400, 'height': 1000})
        context.add_cookies([{'name': 'llmcord_session', 'value': 'browser-search-session', 'url': self.url,
                              'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
        original_page, errors = self.page, []
        try:
            self.assertEqual(context.request.post(self.url + '/_test/restore').status, 200)
            owner = context.request.post(self.url + '/_test/seed-search-lore').json()['owner']
            page = self.page = context.new_page()
            page.on('pageerror', lambda e: errors.append(e.stack or str(e)))
            page.goto(f'{self.url}/admin/guild/1?owner={owner}')
            page.wait_for_function('window.did_handshake === true && window.socket?.connected === true')
            page.get_by_role('tab', name='Lore', exact=True).click()
            kind, ident = owner.split(':')
            self.lore_idle({'left': [kind, int(ident)]})
            left = page.locator('.lore-drop-left')
            left.locator('.lore-entry').filter(has_text='Lightning storms close the harbor').wait_for()

            self.assertEqual(context.request.post(self.url + '/_test/counters/reset').status, 200)
            box = page.get_by_label('Search content and keywords', exact=True)
            box.click()
            box.press_sequentially('lighthouse', delay=30)

            # Settle: the filtered board is drawn and no further renders arrive for 0.6 s.
            page.wait_for_function("""() => {
                const tiles = [...document.querySelectorAll('.lore-drop-left .lore-entry')].map(t => t.innerText);
                return tiles.length === 2 && tiles.some(t => t.includes('Mara keeps the lighthouse lamp burning'))
                    && tiles.some(t => t.includes('Ships steer by the north beacon'));
            }""", timeout=30000)
            renders, stable = None, 0
            for _ in range(100):
                current = context.request.get(self.url + '/_test/counters').json().get('admin_entries', 0)
                stable = stable + 1 if current == renders else 0
                renders = current
                if stable >= 3:
                    break
                page.wait_for_timeout(200)
            self.lore_idle()

            # Correctness: the full query filters exactly as before (content or keywords, case-insensitive).
            self.assertEqual(box.input_value(), 'lighthouse')
            tiles = left.locator('.lore-entry').all_inner_texts()
            self.assertEqual(len(tiles), 2, tiles)
            self.assertFalse(any('Lightning storms' in tile for tile in tiles), tiles)
            page.locator('.lore-panel-left').get_by_text('2 entries · page 1', exact=True).wait_for()
            self.assertEqual(page.locator('.lore-drop-right .lore-entry').count(), 0)
            self.assertFalse(errors, errors)

            # Debounce: at most two board renders (two admin_entries_page calls each) for the whole burst.
            self.assertLessEqual(renders, 4, f'admin_entries_page called {renders} times while typing 10 characters')
        finally:
            self.page = original_page
            context.close()

    def test_guild_page_load_reads_each_guild_list_once(self):
        """PERF-05: one load of the guild page reads each guild-wide list once, not once per panel.

        Opening every tab (panels build lazily, on first selection) called, before the per-render snapshot,
        ``list_spaces`` about eight times and ``lorebook_links`` once per guild book per space. Targets: one call each to
        ``list_spaces``, ``list_characters``, ``list_channels``, ``list_lorebooks`` and ``thread_lore_scopes``,
        and exactly one ``lorebook_links`` per guild-target book (the fixture has several spaces, so the old
        per-space loop would give books x spaces).
        """
        context = self.browser.new_context(ignore_https_errors=True, viewport={'width': 1400, 'height': 1000})
        context.add_cookies([{'name': 'llmcord_session', 'value': 'browser-snapshot-session', 'url': self.url,
                              'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
        original_page, errors = self.page, []
        try:
            self.assertEqual(context.request.post(self.url + '/_test/restore').status, 200)
            page = self.page = context.new_page()
            page.on('pageerror', lambda e: errors.append(e.stack or str(e)))
            # /_test/state uses the unwrapped store methods, so reading it does not disturb the counters.
            guild_books = sum(1 for book in context.request.get(self.url + '/_test/state').json()['lorebooks']
                              if book['target_kind'] == 'guild')
            self.assertEqual(context.request.post(self.url + '/_test/counters/reset').status, 200)
            page.goto(self.url + '/admin/guild/1')
            page.wait_for_function('window.did_handshake === true && window.socket?.connected === true')
            page.get_by_role('tab', name='Server setup', exact=True).click()
            page.get_by_label('Space name', exact=True).wait_for()
            page.wait_for_timeout(300)
            counters = context.request.get(self.url + '/_test/counters').json()
            self.assertFalse(errors, errors)
            got = {name: counters.get('store:' + name, 0) for name in
                   ('list_spaces', 'list_characters', 'list_channels', 'list_lorebooks', 'thread_lore_scopes', 'lorebook_links')}
            self.assertEqual({k: v for k, v in got.items() if k != 'lorebook_links'},
                             {'list_spaces': 1, 'list_characters': 1, 'list_channels': 1, 'list_lorebooks': 1, 'thread_lore_scopes': 1}, got)
            self.assertEqual(got['lorebook_links'], 0, got)
            # Building the Imports panel takes a fresh snapshot (one more read each) and one lorebook_links per guild book.
            self.assertGreaterEqual(guild_books, 1)
            page.get_by_role('tab', name='Imports', exact=True).click()
            page.get_by_label('Lorebook name', exact=True).wait_for()
            page.wait_for_timeout(300)
            counters = context.request.get(self.url + '/_test/counters').json()
            self.assertEqual(counters.get('store:lorebook_links', 0), guild_books, counters)
            self.assertEqual(counters.get('store:list_spaces', 0), 2, counters)
        finally:
            self.page = original_page
            context.close()

    def ux_page(self, path):
        """Open a page in its own context/session (UX-01 tests); returns (context, page, errors)."""
        context = self.browser.new_context(ignore_https_errors=True, viewport={'width': 1400, 'height': 1000})
        context.add_cookies([{'name': 'llmcord_session', 'value': 'browser-ux-session', 'url': self.url,
                              'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
        errors = []
        network = []
        page = context.new_page()
        page.on('pageerror', lambda e: errors.append(e.stack or str(e)))
        page.on('requestfailed', lambda r: network.append(f'failed {r.url} {r.failure}'))
        page.on('response', lambda r: network.append(f'{r.status} {r.url}') if r.status >= 400 else None)
        page.network = network
        page.goto(self.url + path)
        page.wait_for_function('window.did_handshake === true && window.socket?.connected === true')
        return context, page, errors

    def tab_selected(self, page, name):
        return page.get_by_role('tab', name=name, exact=True).get_attribute('aria-selected') == 'true'

    def test_tab_query_selects_prompt_presets(self):
        """UX-01: ?tab=prompts opens Prompt presets (and an unknown value falls back to Server setup)."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=prompts')
        try:
            page.get_by_role('tab', name='Prompt presets', exact=True).wait_for()
            self.assertTrue(self.tab_selected(page, 'Prompt presets'))
            page.goto(self.url + '/admin/guild/1?tab=bogus')
            page.get_by_role('tab', name='Server setup', exact=True).wait_for()
            self.assertTrue(self.tab_selected(page, 'Server setup'))
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_tab_switch_is_recorded_in_url_and_survives_reload(self):
        """UX-01: switching tabs updates ?tab= in place (keeping owner=), so a reload stays on that tab."""
        context, page, errors = self.ux_page('/admin/guild/1?owner=channel:100')
        try:
            page.get_by_role('tab', name='Characters', exact=True).click()
            page.wait_for_url('**tab=characters*')
            self.assertIn('owner=channel', page.url)
            page.goto(page.url)
            page.wait_for_function('window.did_handshake === true && window.socket?.connected === true')
            page.get_by_role('tab', name='Characters', exact=True).wait_for()
            self.assertTrue(self.tab_selected(page, 'Characters'))
            self.assertIn('tab=characters', page.url)
            self.assertIn('owner=channel', page.url)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_save_cast_shows_success_toast(self):
        """UX-01: Save cast (an operation returning None) reports 'Cast saved'."""
        context, page, errors = self.ux_page('/admin/guild/1')
        try:
            page.get_by_role('button', name='Save cast', exact=True).first.click()
            page.get_by_text('Cast saved', exact=True).wait_for(timeout=5000)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_server_timezone_select_saves_with_toast_and_audit(self):
        """FEAT-04: Server setup has a searchable 'Server timezone' select; Save timezone stores it, toasts, and audits old/new."""
        context, page, errors = self.ux_page('/admin/guild/1')
        try:
            page.get_by_role('tab', name='Server setup', exact=True).click()
            select = page.get_by_label('Server timezone', exact=True)
            select.wait_for(timeout=5000)
            select.click()
            select.fill('Asia/Seoul')
            page.get_by_role('option', name='Asia/Seoul', exact=True).click()
            page.get_by_role('button', name='Save timezone', exact=True).click()
            page.get_by_text('Timezone saved', exact=True).wait_for(timeout=5000)
            self.wait_for(lambda: self.state()['guild_timezone'] == 'Asia/Seoul')
            rows = [r for r in self.state()['audit'] if r['action'] == 'settings.timezone']
            self.assertEqual([json.loads(r['detail_json']) for r in rows], [{'old': '', 'new': 'Asia/Seoul'}])
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_channel_selects_show_names_not_ids(self):
        """UX-01: lore owner, imports channel and presets sample-channel selects show '#scene', not '100'."""
        context, page, errors = self.ux_page('/admin/guild/1')
        try:
            page.get_by_role('tab', name='Lore', exact=True).click()
            page.locator('.lore-drop-zone').first.wait_for(state='attached')
            page.get_by_label('Left owner', exact=True).click()
            page.get_by_role('option', name='Channel: #scene', exact=True).wait_for()
            self.assertEqual(page.get_by_role('option', name='Channel: 100', exact=True).count(), 0)
            page.keyboard.press('Escape')
            page.get_by_role('tab', name='Imports', exact=True).click()
            page.get_by_label('Channel (channel lorebooks only)', exact=True).click()
            page.get_by_role('option', name='#scene', exact=True).wait_for()
            self.assertEqual(page.get_by_role('option', name='100', exact=True).count(), 0)
            page.keyboard.press('Escape')
            page.get_by_role('tab', name='Prompt presets', exact=True).click()
            page.get_by_label('Sample channel (optional)', exact=True).click()
            page.get_by_role('option', name='#scene', exact=True).wait_for()
            self.assertEqual(page.get_by_role('option', name='100', exact=True).count(), 0)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_page_load_fetches_discord_channels_once(self):
        """UX-01: one page load makes exactly one Discord /guilds/1/channels call (shared by setup panel and names)."""
        context, page, errors = self.ux_page('/admin/')
        try:
            self.assertEqual(context.request.post(self.url + '/_test/metrics/reset').status, 200)
            page.goto(self.url + '/admin/guild/1')
            page.wait_for_function('window.did_handshake === true && window.socket?.connected === true')
            page.get_by_label('Space name', exact=True).wait_for()
            page.wait_for_timeout(300)
            calls = context.request.get(self.url + '/_test/metrics').json()
            self.assertEqual(calls.get('GET /guilds/1/channels', 0), 1, calls)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def panel_reads(self, context):
        counters = context.request.get(self.url + '/_test/counters').json()
        return {name: counters.get('store:' + name, 0) for name in ('model_usage_summary', 'list_presets')}

    def test_lazy_panels_build_on_first_selection(self):
        """PERF-01: a page load builds only the selected tab's panel; others are built once, on first selection.

        ``model_usage_summary`` is read only by Server setup and ``list_presets`` only by Prompt presets.
        """
        context, page, errors = self.ux_page('/admin/')
        try:
            self.assertEqual(context.request.post(self.url + '/_test/counters/reset').status, 200)
            page.goto(self.url + '/admin/guild/1?tab=characters')
            page.wait_for_function('window.did_handshake === true && window.socket?.connected === true')
            page.get_by_role('button', name='Create character', exact=True).wait_for()
            page.wait_for_timeout(300)
            self.assertEqual(self.panel_reads(context), {'model_usage_summary': 0, 'list_presets': 0})
            page.get_by_role('tab', name='Prompt presets', exact=True).click()
            page.get_by_label('Sample channel (optional)', exact=True).wait_for()
            page.wait_for_timeout(300)
            built = self.panel_reads(context)
            self.assertEqual(built['model_usage_summary'], 0, built)
            self.assertGreaterEqual(built['list_presets'], 1, built)
            # Leaving and returning does not rebuild the panel.
            page.get_by_role('tab', name='Characters', exact=True).click()
            page.get_by_role('tab', name='Prompt presets', exact=True).click()
            page.get_by_label('Sample channel (optional)', exact=True).wait_for()
            page.wait_for_timeout(300)
            self.assertEqual(self.panel_reads(context), built)
            page.get_by_role('tab', name='Server setup', exact=True).click()
            page.get_by_label('Space name', exact=True).wait_for()
            page.wait_for_timeout(300)
            self.assertEqual(self.panel_reads(context), {**built, 'model_usage_summary': 1})
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def create_space_in_place(self, page, name):
        page.evaluate('window.__marker = 1')
        page.get_by_label('Space name', exact=True).fill(name)
        page.get_by_role('button', name='Create space', exact=True).click()
        page.get_by_text(f'{name} · world', exact=True).wait_for(timeout=5000)

    def test_mutation_updates_page_without_browser_reload(self):
        """PERF-01: Create space refreshes the page in place: no navigation, no new Discord channel fetch."""
        context, page, errors = self.ux_page('/admin/guild/1')
        try:
            page.get_by_label('Space name', exact=True).wait_for()
            self.assertEqual(context.request.post(self.url + '/_test/metrics/reset').status, 200)
            self.create_space_in_place(page, 'Inplace world')
            self.assertEqual(page.evaluate('window.__marker'), 1)
            page.wait_for_timeout(300)
            calls = context.request.get(self.url + '/_test/metrics').json()
            self.assertEqual(calls.get('GET /guilds/1/channels', 0), 0, calls)
            self.assertTrue(self.tab_selected(page, 'Server setup'))
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_other_tabs_show_current_data_after_mutation(self):
        """PERF-01: a panel built before a mutation on another tab shows the new data when selected again."""
        context, page, errors = self.ux_page('/admin/guild/1')
        try:
            page.get_by_role('tab', name='Characters', exact=True).click()
            page.get_by_role('button', name='Create character', exact=True).wait_for()
            page.get_by_role('tab', name='Server setup', exact=True).click()
            self.create_space_in_place(page, 'Fresh world')
            page.get_by_role('tab', name='Characters', exact=True).click()
            page.get_by_role('button', name='Create character', exact=True).click()
            page.get_by_role('dialog').get_by_label('Home world', exact=True).click()
            page.get_by_role('option', name='Fresh world', exact=True).wait_for()
            page.keyboard.press('Escape')
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_lore_import_button_builds_imports_panel(self):
        """PERF-01 / UX-01: the Lore tab's Import lorebook button switches to Imports, building it if it is not yet built."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=lore')
        try:
            page.get_by_role('button', name='Import lorebook', exact=True).wait_for()
            self.assertEqual(page.get_by_label('Channel (channel lorebooks only)', exact=True).count(), 0)
            page.get_by_role('button', name='Import lorebook', exact=True).click()
            page.get_by_label('Channel (channel lorebooks only)', exact=True).wait_for(timeout=5000)
            self.assertTrue(self.tab_selected(page, 'Imports'))
            page.wait_for_url('**tab=imports*')
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_imports_lorebook_edit_entries_opens_lore_on_that_book(self):
        """UX-05: a lorebook's 'Edit entries in Lore' button switches to the Lore tab with that lorebook as the shown owner."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=imports')
        try:
            book = next(row for row in self.state()['lorebooks'] if row['name'] == 'Test book')
            expansion = page.locator('.q-expansion-item').filter(has=page.get_by_text('Test book · channel', exact=True)).first
            expansion.get_by_text('Test book · channel', exact=True).click()
            expansion.get_by_role('button', name='Edit entries in Lore', exact=True).click()
            page.get_by_label('Left owner', exact=True).wait_for(timeout=5000)
            self.assertTrue(self.tab_selected(page, 'Lore'))
            page.wait_for_function(
                "([kind, ident]) => { const left = document.querySelector('.lore-drop-left'); "
                "return left && left.dataset.ownerKind === kind && left.dataset.ownerId === ident; }",
                arg=['book', str(book['id'])], timeout=5000)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_fallback_avatar_upload_saves_without_another_click(self):
        """UX-05: choosing a fallback avatar file saves it at once (notice 'Fallback avatar saved'); no Save button remains."""
        from io import BytesIO
        from PIL import Image
        context, page, errors = self.ux_page('/admin/guild/1?tab=characters')
        try:
            alice = next(row for row in self.state()['characters'] if row['name'] == 'Alice')
            character = page.locator('.character-card').filter(has=page.get_by_text('Alice', exact=True)).first
            character.get_by_text('Alice', exact=True).first.click()
            character.get_by_text('Fallback static avatar', exact=True).click()
            fallback = character.locator('.fallback-avatar')
            self.assertEqual(fallback.get_by_role('button', name='Save fallback avatar', exact=True).count(), 0)
            portrait = BytesIO()
            Image.new('RGB', (50, 50), 'blue').save(portrait, 'PNG')
            fallback.locator('input[type=file]').set_input_files({'name': 'fallback.png', 'mimeType': 'image/png', 'buffer': portrait.getvalue()})
            page.get_by_text('Fallback avatar saved', exact=True).wait_for(timeout=5000)
            self.wait_for(lambda: next(row for row in self.state()['characters'] if row['id'] == alice['id'])['has_static_avatar'])
            # The characters tab was refreshed, so the saved image is rendered.
            character = page.locator('.character-card').filter(has=page.get_by_text('Alice', exact=True)).first
            character.get_by_text('Alice', exact=True).first.click()
            character.get_by_text('Fallback static avatar', exact=True).click()
            character.locator('.fallback-avatar img').first.wait_for(timeout=5000)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_delete_preset_asks_for_confirmation(self):
        """UX-05: 'Delete preset' opens a dialog; Cancel keeps the preset and the red confirm button deletes it."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=prompts')
        try:
            page.get_by_label('Preset name', exact=True).fill('Doomed preset')
            page.get_by_role('button', name='Save as new preset', exact=True).click()
            self.wait_for(lambda: any(r['name'] == 'Doomed preset' for r in self.state()['presets']))
            page.get_by_role('button', name='Delete preset', exact=True).click()
            dialog = page.get_by_role('dialog')
            dialog.get_by_text('Delete preset Doomed preset?', exact=True).wait_for(timeout=5000)
            dialog.get_by_role('button', name='Cancel', exact=True).click()
            dialog.wait_for(state='hidden')
            self.assertTrue(any(r['name'] == 'Doomed preset' for r in self.state()['presets']))
            page.get_by_role('button', name='Delete preset', exact=True).click()
            dialog.get_by_role('button', name='Delete preset', exact=True).click()
            self.wait_for(lambda: not any(r['name'] == 'Doomed preset' for r in self.state()['presets']))
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_lore_entry_delete_goes_through_the_dialog(self):
        """UX-05: deleting a lore entry from its editor opens a confirmation dialog (no checkbox); Cancel keeps it."""
        import re
        context, page, errors = self.ux_page('/admin/guild/1?tab=lore')
        try:
            page.get_by_role('button', name='New entry', exact=True).first.wait_for(timeout=5000)
            page.get_by_role('button', name='New entry', exact=True).first.click()
            page.get_by_label('Content', exact=True).fill('Doomed lore entry')
            page.get_by_role('button', name='Save lore', exact=True).click()
            self.wait_for(lambda: any(r['content'] == 'Doomed lore entry' for r in self.state()['lore']))
            page.locator('.lore-entry').filter(has_text='Doomed lore entry').get_by_role('button', name='Edit / transfer', exact=True).click()
            self.assertEqual(page.get_by_text('Confirm deleting this entry', exact=True).count(), 0)
            editor = page.locator('.q-card').filter(has=page.get_by_text('Edit lore', exact=True)).last
            editor.get_by_role('button', name='Delete', exact=True).click()
            dialog = page.get_by_role('dialog')
            dialog.get_by_role('button', name='Cancel', exact=True).wait_for(timeout=5000)
            dialog.get_by_role('button', name='Cancel', exact=True).click()
            dialog.wait_for(state='hidden')
            self.assertTrue(any(r['content'] == 'Doomed lore entry' for r in self.state()['lore']))
            editor.get_by_role('button', name='Delete', exact=True).click()
            dialog.get_by_role('button', name=re.compile('^Delete')).click()
            self.wait_for(lambda: not any(r['content'] == 'Doomed lore entry' for r in self.state()['lore']))
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    CLIPPED_SELECT_LABELS_JS = """() => [...document.querySelectorAll('.q-select')].flatMap(select => {
        const label = select.querySelector('.q-field__label');
        if (!label || !label.getClientRects().length) return [];
        if (select.classList.contains('q-field--float')) return [];
        return [{text: label.textContent.trim(), scrollWidth: label.scrollWidth, clientWidth: label.clientWidth,
                 selectWidth: select.getBoundingClientRect().width}];
    })"""

    maxDiff = None

    def test_empty_select_labels_are_not_clipped(self):
        """UX-01: an empty dropdown on any guild tab shows its whole floating label (no ellipsis truncation)."""
        context, page, errors = self.ux_page('/admin/guild/1')
        clipped = {}
        measured = 0
        try:
            anchors = {'Server setup': page.get_by_label('Space name', exact=True),
                       'Characters': page.get_by_role('button', name='Create character', exact=True),
                       'Lore': page.get_by_role('button', name='Import lorebook', exact=True),
                       'Imports': page.get_by_label('Channel (channel lorebooks only)', exact=True),
                       'Prompt presets': page.get_by_label('Sample channel (optional)', exact=True)}
            for tab, anchor in anchors.items():
                page.get_by_role('tab', name=tab, exact=True).click()
                anchor.wait_for(timeout=5000)  # panels build lazily on first selection
                page.wait_for_function('() => document.querySelectorAll(".q-tab-panel").length === 1', timeout=5000)
                page.wait_for_timeout(100)
                labels = page.evaluate(self.CLIPPED_SELECT_LABELS_JS)
                measured += len(labels)
                clipped[tab] = [(item['text'], item['scrollWidth'], item['clientWidth'])
                                for item in labels if item['scrollWidth'] > item['clientWidth']]
            self.assertGreater(measured, 5)
            self.assertEqual({tab: items for tab, items in clipped.items() if items}, {})
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()
