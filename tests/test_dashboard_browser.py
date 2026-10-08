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
            login_page.get_by_role('link', name='Sign in with Discord', exact=True).click()
            login_page.wait_for_url('https://discord.com/oauth2/authorize?*')
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
        with page.expect_navigation():
            world_panel.get_by_role('button', name='Save world guidelines', exact=True).click()
        self.assertTrue(any(row['kind'] == 'space' and 'courier guild' in row['content'] for row in self.state()['guidelines']))
        channel_panel = page.locator('.channel-card').filter(has=page.get_by_text('#scene', exact=True))
        channel_panel.get_by_label('Channel guidelines', exact=True).fill('Occasional fourth-wall jokes are welcome; keep them brief.')
        with page.expect_navigation():
            channel_panel.get_by_role('button', name='Save channel guidelines', exact=True).click()
        self.assertTrue(any(row['kind'] == 'channel' and 'fourth-wall' in row['content'] for row in self.state()['guidelines']))
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
        with page.expect_navigation():
            dialog.get_by_role('button', name='Delete world', exact=True).click()
        self.assertFalse(any(row['name'] == 'Browser world' for row in self.state()['spaces']))
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
        with page.expect_navigation():
            dialog.get_by_role('button', name='Create', exact=True).click()
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
        page.get_by_text('Fallback image ready; save it to keep it', exact=True).wait_for()
        fallback.locator('.q-uploader__file--img').wait_for()
        with page.expect_navigation():
            fallback.get_by_role('button', name='Save fallback avatar', exact=True).click()
        self.assertTrue(next(row for row in self.state()['characters'] if row['id'] == blank['id'])['has_static_avatar'])
        self.assertFalse(self.state()['assets'])
        self.assertFalse(any(slot['has_image'] for slot in self.state()['slots'] if slot['character_id'] == blank['id']))
        character.get_by_text('Browser blank', exact=True).click()
        character.get_by_label('Description', exact=True).fill('A character created without an import')
        with page.expect_navigation():
            character.get_by_role('button', name='Save character', exact=True).click()
        self.assertEqual(json.loads(next(row for row in self.state()['characters'] if row['id'] == blank['id'])['card'])['description'], 'A character created without an import')
        character.get_by_text('Browser blank', exact=True).click()
        character.get_by_text('Fallback static avatar', exact=True).click()
        with page.expect_navigation():
            fallback.get_by_role('button', name='Remove fallback avatar', exact=True).click()
        self.assertFalse(next(row for row in self.state()['characters'] if row['id'] == blank['id'])['has_static_avatar'])
        character.get_by_text('Browser blank', exact=True).click()
        character.get_by_role('button', name='Delete character', exact=True).click()
        dialog.get_by_text('Delete Browser blank?', exact=True).wait_for()
        dialog.get_by_role('button', name='Cancel', exact=True).click()
        self.assertTrue(any(row['id'] == blank['id'] for row in self.state()['characters']))
        character.get_by_role('button', name='Delete character', exact=True).click()
        with page.expect_navigation():
            dialog.get_by_role('button', name='Delete permanently', exact=True).click()
        self.assertFalse(any(row['id'] == blank['id'] for row in self.state()['characters']))
        self.assertFalse(any(row['character_id'] == blank['id'] for row in self.state()['slots']))
        page.get_by_role('button', name='Create character', exact=True).wait_for()

        page.get_by_role('tab', name='Imports', exact=True).click()
        page.get_by_label('Book name', exact=True).fill('Browser empty book')
        with page.expect_navigation():
            page.get_by_role('button', name='Create book', exact=True).click()
        page.get_by_role('tab', name='Imports', exact=True).click()
        empty = page.locator('.q-expansion-item').filter(has=page.get_by_text('Browser empty book · guild', exact=True)).first
        empty.get_by_text('Browser empty book · guild', exact=True).click()
        empty.get_by_role('button', name='Delete book', exact=True).click()
        dialog.get_by_text('Delete lorebook Browser empty book?', exact=True).wait_for()
        dialog.get_by_role('button', name='Cancel', exact=True).click()
        self.assertTrue(any(row['name'] == 'Browser empty book' for row in self.state()['lorebooks']))
        empty.get_by_role('button', name='Delete book', exact=True).click()
        with page.expect_navigation():
            dialog.get_by_role('button', name='Delete lorebook', exact=True).click()
        self.assertFalse(any(row['name'] == 'Browser empty book' for row in self.state()['lorebooks']))
        page.get_by_label('Destination owner', exact=True).click()
        page.get_by_role('option', name='Guild: Server-wide lore', exact=True).click()
        page.get_by_role('button', name='Import entries into selected owner', exact=True).click()
        data = json.dumps({'type': 'risu', 'ver': 1, 'data': [{'key': '', 'content': 'Direct guild fact', 'alwaysActive': True, 'insertorder': 37}]}).encode()
        dialog.locator('input[type=file]').set_input_files({'name': 'guild.json', 'mimeType': 'application/json', 'buffer': data})
        dialog.get_by_text('RisuAI: 1 new entries, 0 already present', exact=True).wait_for()
        with page.expect_navigation():
            dialog.get_by_role('button', name='Apply entry import', exact=True).click()
        self.assertTrue(any(row['content'] == 'Direct guild fact' for row in self.state()['guild_lore']))
        self.assertEqual(len(self.state()['lorebooks']), 1)
        self.choose_lore_owner('left', 'World: World', 'space', 1)
        data = json.dumps({'entries': [{'content': 'Direct world fact', 'constant': True, 'order': 37}]}).encode()
        for count in (1, 0):
            page.get_by_role('button', name='Import JSON entries', exact=True).first.click()
            dialog.locator('input[type=file]').set_input_files({'name': 'world.json', 'mimeType': 'application/json', 'buffer': data})
            dialog.get_by_text(f'SillyTavern: {count} new entries, {1 - count} already present', exact=True).wait_for()
            dialog.get_by_role('button', name='Apply entry import', exact=True).click()
            self.lore_idle()
        self.assertEqual(sum(row['content'] == 'Direct world fact' for row in self.state()['lore']), 1)
        self.assertEqual(len(self.state()['lorebooks']), 1)

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
        with page.expect_navigation(wait_until='networkidle'):
            page.get_by_role('button', name='Apply card import', exact=True).click()
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
        page.get_by_text('Image ready; save the slot to keep it', exact=True).wait_for()
        happy.get_by_role('button', name='Save slot', exact=True).click()
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
        page.get_by_label('Book name', exact=True).fill('Browser import')
        with page.expect_navigation(wait_until='networkidle'):
            page.get_by_role('button', name='Create book', exact=True).click()
        page.get_by_role('tab', name='Lore', exact=True).click()
        page.get_by_role('button', name='Import lorebook', exact=True).click()
        book = page.locator('.q-expansion-item').filter(has=page.get_by_text('Browser import · guild', exact=True)).first
        book.get_by_text('Browser import · guild', exact=True).click()
        content = json.dumps({'entries': {'1': {'key': ['moon'], 'content': 'Imported lorebook entry', 'enabled': True}}}).encode()
        book.locator('input[type=file]').set_input_files({'name': 'lorebook.json', 'mimeType': 'application/json', 'buffer': content})
        book.get_by_role('button', name='Apply lorebook sync', exact=True).wait_for()
        entry = book.locator('.q-expansion-item').filter(has=page.get_by_text('Incoming: Imported lorebook entry', exact=True)).last
        entry.locator('.q-item').first.click()
        book.get_by_text('Incoming: Imported lorebook entry', exact=True).wait_for()
        with page.expect_navigation(wait_until='networkidle'):
            book.get_by_role('button', name='Apply lorebook sync', exact=True).click()
        self.assertTrue(any(row['content'] == 'Imported lorebook entry' for row in self.state()['books']))
        page.get_by_role('tab', name='Lore', exact=True).click()
        page.get_by_role('button', name='Import lorebook', exact=True).click()
        book = page.locator('.q-expansion-item').filter(has=page.get_by_text('Browser import · guild', exact=True)).first
        book.get_by_text('Browser import · guild', exact=True).click()
        with page.expect_navigation(wait_until='networkidle'):
            book.get_by_role('button', name='Enable in World', exact=True).click()
        page.get_by_role('tab', name='Lore', exact=True).click()
        page.get_by_role('button', name='Import lorebook', exact=True).click()
        book = page.locator('.q-expansion-item').filter(has=page.get_by_text('Browser import · guild', exact=True)).first
        book.get_by_text('Browser import · guild', exact=True).click()
        book.get_by_role('button', name='Disable in World', exact=True).wait_for()

        risu = json.dumps({'type': 'risu', 'ver': 1, 'data': [
            {'id': '1', 'key': 'moon', 'secondkey': '', 'content': 'Imported Risu entry',
             'insertorder': 0, 'mode': 'normal', 'alwaysActive': False, 'selective': False, 'useRegex': False},
            {'id': 'folder', 'key': 'folder:1', 'content': '', 'insertorder': 100, 'mode': 'folder'},
        ]}).encode()
        book.locator('input[type=file]').set_input_files({'name': 'risu.json', 'mimeType': 'application/json', 'buffer': risu})
        book.get_by_text('RisuAI lorebook detected · 2 entries', exact=True).wait_for()
        with page.expect_navigation(wait_until='networkidle'):
            book.get_by_role('button', name='Apply lorebook sync', exact=True).click()
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
