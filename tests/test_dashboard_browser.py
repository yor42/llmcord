"""Opt-in browser integration tests against a disposable mocked Discord server."""
import json
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


def _section(page, title):
    """The dashboard card titled `title`; scopes generic labels such as 'Name', 'Kind', 'Channel' and 'Create'."""
    return page.locator('.ll-section').filter(has=page.locator('.ll-section-title', has_text=title))


def _worlds_section(page):
    """The Server setup 'Worlds and hubs' card."""
    return _section(page, 'Worlds and hubs')


@unittest.skipUnless(os.environ.get('LLMCORD_BROWSER_TESTS') == '1', 'Set LLMCORD_BROWSER_TESTS=1 to run browser integration tests')
class DashboardBrowserTests(unittest.TestCase):
    SLOW_SERVER_POLLS = 300  # poll count (x 100 ms = 30 s): publishing an emotion image can take seconds on a busy machine

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
        cls.watch(cls.page, cls.errors)

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
            artifacts = Path(__file__).resolve().parents[1] / '.test-artifacts' / self.id()  # MNT-30: one folder per failing test
            artifacts.mkdir(parents=True, exist_ok=True)
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

    def wait_for(self, predicate, polls=50):
        for _ in range(polls):
            if predicate():
                return
            self.page.wait_for_timeout(100)
        self.fail('Expected database state was not reached')

    def settled_height(self, locator, polls=50):
        previous = None
        for _ in range(polls):
            box = locator.bounding_box()
            if box is None:
                self.fail('Element has no layout box (hidden or detached) while measuring its height')
            height = box['height']
            if height == previous:
                return height
            previous = height
            locator.page.wait_for_timeout(100)
        self.fail(f'Height never settled (last {previous})')

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
        self.load(page, self.url + '/admin/')
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
        page.get_by_text('Server settings', exact=True).wait_for()
        # Rendered controls precede the connection that delivers their events.
        page.wait_for_function('window.did_handshake === true && window.socket?.connected === true')
        world_panel = page.locator('.space-card').filter(has=page.get_by_text('World · world', exact=True))
        world_panel.get_by_text('World · world', exact=True).click()
        world_panel.get_by_label('World guidelines', exact=True).fill('The setting is a courier guild. Treat users as guild members.')
        world_panel.get_by_role('button', name='Save world guidelines', exact=True).click()
        self.wait_for(lambda: any(row['kind'] == 'space' and 'courier guild' in row['content'] for row in self.state()['guidelines']))
        channel_panel = page.locator('.channel-card').filter(has=page.locator('.ll-block-name').get_by_text(re.compile(r'^#scene\s*$')))
        channel_panel.locator('button.channel-toggle').click()
        channel_panel.get_by_label('Channel guidelines', exact=True).fill('Occasional fourth-wall jokes are welcome; keep them brief.')
        savebar = page.get_by_role('region', name='Unsaved changes')
        savebar.get_by_role('button', name='Save changes', exact=True).click()
        savebar.wait_for(state='hidden')
        self.wait_for(lambda: any(row['kind'] == 'channel' and 'fourth-wall' in row['content'] for row in self.state()['guidelines']))
        _worlds_section(page).get_by_label('Name', exact=True).fill('Browser world')
        _worlds_section(page).get_by_role('button', name='Create', exact=True).click()
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
        savebar.get_by_role('button', name='Save changes', exact=True).click()
        savebar.wait_for(state='hidden')

        page.get_by_role('tab', name='Characters', exact=True).click()
        page.get_by_role('button', name='Create character', exact=True).click()
        dialog = page.get_by_role('dialog')
        dialog.get_by_role('button', name='Create', exact=True).click()
        page.get_by_text('Character name cannot be empty. Enter a name.', exact=True).wait_for()
        dialog.get_by_label('Character name', exact=True).fill(' Alice ')
        dialog.get_by_role('button', name='Create', exact=True).click()
        page.get_by_text('This character name already exists. Choose another name.', exact=True).wait_for()
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
        savebar = page.get_by_role('region', name='Unsaved changes')
        savebar.get_by_role('button', name='Save changes', exact=True).click()
        savebar.wait_for(state='hidden')
        self.wait_for(lambda: json.loads(next(row for row in self.state()['characters'] if row['id'] == blank['id'])['card'])['description'] == 'A character created without an import')
        self.assertEqual(json.loads(next(row for row in self.state()['characters'] if row['id'] == blank['id'])['card'])['description'], 'A character created without an import')
        character.get_by_text('Browser blank', exact=True).click()
        character.get_by_text('Fallback static avatar', exact=True).click()
        self.open_dialog(fallback.get_by_role('button', name='Remove fallback avatar', exact=True), dialog)
        dialog.get_by_role('button', name='Remove fallback avatar', exact=True).click()
        self.wait_for(lambda: not next(row for row in self.state()['characters'] if row['id'] == blank['id'])['has_static_avatar'])
        character.get_by_text('Browser blank', exact=True).click()
        self.open_more_item(page, character, 'Browser blank', 'Delete character')
        dialog.get_by_text('Delete Browser blank?', exact=True).wait_for()
        dialog.get_by_role('button', name='Cancel', exact=True).click()
        self.assertTrue(any(row['id'] == blank['id'] for row in self.state()['characters']))
        self.open_more_item(page, character, 'Browser blank', 'Delete character')
        dialog.get_by_role('button', name='Delete permanently', exact=True).click()
        self.wait_for(lambda: not any(row['id'] == blank['id'] for row in self.state()['characters']))
        self.assertFalse(any(row['character_id'] == blank['id'] for row in self.state()['slots']))
        page.get_by_role('button', name='Create character', exact=True).wait_for()

        books_before = len(self.state()['lorebooks'])
        page.get_by_role('tab', name='Imports', exact=True).click()
        page.get_by_label('Lorebook name', exact=True).fill('Browser empty book')
        page.get_by_role('button', name='Create lorebook', exact=True).click()
        page.get_by_role('tab', name='Imports', exact=True).click()
        empty = page.locator('.q-expansion-item').filter(has=page.get_by_text('Browser empty book · Server lorebook', exact=True)).first
        empty.get_by_text('Browser empty book · Server lorebook', exact=True).click()
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
        # UI-19: the multi-page count line names the page, and the wider Page select shows its whole label.
        page.locator('.lore-panel-right').get_by_text('75 entries · page 1 of 2', exact=True).wait_for()
        page_select = page.locator('.lore-panel-right .q-select').filter(has=page.get_by_label('Page', exact=True)).first
        label = page_select.locator('.q-field__label')
        self.assertLessEqual(label.evaluate('el => el.scrollWidth'), label.evaluate('el => el.clientWidth'))
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
        self.open_more_item(page, right_list.locator('.lore-entry').filter(has_text='Bulk A'), 'entry', 'Move left')
        self.wait_for(lambda: any(row['content'] == 'Bulk A' for row in self.state()['lore']))
        self.lore_idle()
        if any(row['content'] == 'Bulk B' for row in self.state()['books']):
            self.open_more_item(page, right_list.locator('.lore-entry').filter(has_text='Bulk B'), 'entry', 'Move left')
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
        self.open_more_item(page, right_list.locator('.lore-entry').filter(has_text='Bulk C'), 'entry', 'Delete')
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
        page.locator('.prompt-toggle').first.click()
        page.get_by_label('Prompt text / template', exact=True).first.fill('Respond as {{char}}. Browser custom instructions.')
        page.get_by_label('Prompt text / template', exact=True).first.press('Tab')
        cards = page.locator('.prompt-block')
        cards.first.locator('.prompt-handle').drag_to(cards.nth(2))
        page.get_by_label('Preset name', exact=True).fill('Browser preset')
        page.get_by_role('region', name='Unsaved changes').get_by_role('button', name='Save draft', exact=True).click()
        self.wait_for(lambda: any(r['name'] == 'Browser preset' for r in self.state()['presets']))
        page.get_by_role('region', name='Unsaved changes').wait_for(state='hidden')
        self.open_more_item(page, page, 'preset', 'Activate saved revision')
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
        self.wait_for(lambda: bool(self.state()['assets']), self.SLOW_SERVER_POLLS)
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
        book = page.locator('.q-expansion-item').filter(has=page.get_by_text('Browser import · Server lorebook', exact=True)).first
        book.get_by_text('Browser import · Server lorebook', exact=True).click()
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
        book = page.locator('.q-expansion-item').filter(has=page.get_by_text('Browser import · Server lorebook', exact=True)).first
        book.get_by_text('Browser import · Server lorebook', exact=True).click()
        book.get_by_role('button', name='Enable in World', exact=True).click()
        book.get_by_role('button', name='Enable in World', exact=True).wait_for(state='hidden')  # panel rebuilt in place, expansion collapsed
        page.get_by_role('tab', name='Lore', exact=True).click()
        page.get_by_role('button', name='Import lorebook', exact=True).click()
        book = page.locator('.q-expansion-item').filter(has=page.get_by_text('Browser import · Server lorebook', exact=True)).first
        book.get_by_text('Browser import · Server lorebook', exact=True).click()
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
        _worlds_section(page).get_by_label('Name', exact=True).fill('Forbidden')
        page.wait_for_timeout(200)
        # /_test/revoke also drops cached guild lists, simulating the 300 s TTL (PERF-01 / D1) having elapsed;
        # the live event below must then be rejected on the refetch.
        self.context.request.post(self.url + '/_test/revoke')
        _worlds_section(page).get_by_role('button', name='Create', exact=True).click(force=True)
        page.wait_for_timeout(300)
        self.assertFalse(any(r['name'] == 'Forbidden' for r in self.state()['spaces']))
        # /_test/restore likewise drops the cached (non-admin) guild list, as if the TTL had elapsed.
        self.context.request.post(self.url + '/_test/restore')
        self.load(page, self.url + '/admin/guild/1')
        _worlds_section(page).get_by_label('Name', exact=True).fill('Forbidden')
        page.wait_for_timeout(200)
        self.context.request.post(self.url + '/_test/expire')
        _worlds_section(page).get_by_role('button', name='Create', exact=True).click(force=True)
        page.wait_for_timeout(300)
        self.assertFalse(any(r['name'] == 'Forbidden' for r in self.state()['spaces']))
        self.assertFalse(self.errors, self.errors)

    def test_rejected_live_event_notifies_bound_client(self):
        """PERF-01 (fixed): a live event rejected by the permission check is dropped but the user is told why.

        A bound client with a matching cookie clicks "Create" in Worlds and hubs after (1) admin permission is revoked (403),
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
            self.watch(page, errors)
            self.load(page, self.url + '/admin/guild/1')
            page.get_by_role('tab', name='Server setup', exact=True).click()
            _worlds_section(page).get_by_label('Name', exact=True).fill('Rejected')
            page.wait_for_timeout(200)
            create = _worlds_section(page).get_by_role('button', name='Create', exact=True)
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
            self.watch(page, errors)
            self.load(page, f'{self.url}/admin/guild/1?owner={owner}')
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
            page.locator('.lore-panel-left').get_by_text('2 entries', exact=True).wait_for()
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
            self.watch(page, errors)
            # /_test/state uses the unwrapped store methods, so reading it does not disturb the counters.
            guild_books = sum(1 for book in context.request.get(self.url + '/_test/state').json()['lorebooks']
                              if book['target_kind'] == 'guild')
            self.load(page, self.url + '/admin/guild/1', before_each=lambda: self.assertEqual(
                context.request.post(self.url + '/_test/counters/reset').status, 200))
            page.get_by_role('tab', name='Server setup', exact=True).click()
            _worlds_section(page).get_by_label('Name', exact=True).wait_for()
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

    def open_more_item(self, page, card, name, item):
        """Open a character card's more-actions menu and click `item`. The menu is portalled to the body, so a lost click retries once."""
        from playwright.sync_api import TimeoutError as PlaywrightTimeout
        menuitem = page.get_by_role('menuitem', name=item, exact=True)
        button = card.get_by_role('button', name=f'More actions for {name}', exact=True)
        button.click()
        try:
            menuitem.wait_for(timeout=5000)
        except PlaywrightTimeout:
            button.click()
            menuitem.wait_for(timeout=10000)
        menuitem.click()

    def open_dialog(self, trigger, dialog):
        """Click `trigger` and wait until `dialog` is open and its buttons sit inside the viewport (Quasar animates it in).
        A click that lands while the page is still re-rendering can be lost, so click once more if no dialog appeared."""
        from playwright.sync_api import TimeoutError as PlaywrightTimeout, expect
        trigger.click()
        try:
            dialog.wait_for(timeout=5000)
        except PlaywrightTimeout:
            trigger.click()
            dialog.wait_for(timeout=10000)
        expect(dialog.get_by_role('button').last).to_be_in_viewport(timeout=10000)

    @staticmethod
    def watch(page, errors):
        page.on('pageerror', lambda e: errors.append(e.stack or str(e)))
        page.watched_errors, page.failed_assets = errors, []
        page.on('requestfailed', lambda r: page.failed_assets.append(r.url) if '/_nicegui/' in r.url and 'ERR_ABORTED' not in (r.failure or '') else None)

    def load(self, page, url, before_each=None):
        """goto + live-socket wait. Chromium sometimes fails a NiceGUI static asset itself (net::ERR_TOO_MANY_RETRIES, the request
        never reaches the server) and nicegui.js then throws a cssRules SecurityError. Only that pair (every page error is that
        SecurityError and a /_nicegui/ request failed) is reloaded, up to twice; any other page error is kept for the assertion.
        ``before_each`` runs before every goto attempt (MNT-35), so count-based tests can reset their counters per attempt.
        """
        for attempt in range(3):
            if before_each:
                before_each()
            page.goto(url)
            page.wait_for_function('window.did_handshake === true && window.socket?.connected === true')
            errors = page.watched_errors
            if attempt == 2 or not (page.failed_assets and errors and all("Failed to read the 'cssRules'" in e for e in errors)):
                return
            errors.clear()
            page.failed_assets.clear()

    def ux_page(self, path):
        """Open a page in its own context/session (UX-01 tests); returns (context, page, errors)."""
        context = self.browser.new_context(ignore_https_errors=True, viewport={'width': 1400, 'height': 1000})
        context.add_cookies([{'name': 'llmcord_session', 'value': 'browser-ux-session', 'url': self.url,
                              'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
        errors = []
        network = []
        page = context.new_page()
        self.watch(page, errors)
        page.on('requestfailed', lambda r: network.append(f'failed {r.url} {r.failure}'))
        page.on('response', lambda r: network.append(f'{r.status} {r.url}') if r.status >= 400 else None)
        page.network = network
        self.load(page, self.url + path)
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
            self.load(page, page.url)
            page.get_by_role('tab', name='Characters', exact=True).wait_for()
            self.assertTrue(self.tab_selected(page, 'Characters'))
            self.assertIn('tab=characters', page.url)
            self.assertIn('owner=channel', page.url)
            page.get_by_role('tab', name='Lore', exact=True).click()
            page.wait_for_url('**tab=lore*')
            self.assertIn('owner=channel', page.url)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_save_channel_settings_shows_success_toast(self):
        """UX-01 / UI-15: saving a changed channel card from the save bar reports 'Channel settings saved'."""
        context, page, errors = self.ux_page('/admin/guild/1')
        try:
            card = page.locator('.channel-card').first
            card.locator('button.channel-toggle').click()
            card.get_by_role('switch', name='Ambient participation', exact=True).click()
            page.get_by_role('region', name='Unsaved changes').get_by_role('button', name='Save changes', exact=True).click()
            page.get_by_text('Channel settings saved', exact=True).wait_for(timeout=5000)
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
            page.get_by_role('region', name='Unsaved changes').get_by_role('button', name='Save changes', exact=True).click()
            page.get_by_text('Server settings saved', exact=True).wait_for(timeout=5000)
            self.wait_for(lambda: self.state()['guild_timezone'] == 'Asia/Seoul')
            rows = [r for r in self.state()['audit'] if r['action'] == 'settings.timezone']
            self.assertEqual([json.loads(r['detail_json']) for r in rows], [{'old': '', 'new': 'Asia/Seoul'}])
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_server_footer_switch_audits_enabled_state(self):
        """UI-08: toggling the reply footer switch audits settings.footer with the new on/off value."""
        context, page, errors = self.ux_page('/admin/guild/1')
        try:
            page.get_by_role('tab', name='Server setup', exact=True).click()
            switch = page.get_by_role('switch', name='Show model and cost footer on replies')
            switch.wait_for(timeout=5000)
            was = switch.get_attribute('aria-checked') == 'true'
            switch.click()
            page.get_by_role('region', name='Unsaved changes').get_by_role('button', name='Save changes', exact=True).click()
            page.get_by_text('Server settings saved', exact=True).wait_for(timeout=5000)
            rows = [r for r in self.state()['audit'] if r['action'] == 'settings.footer']
            self.assertEqual([json.loads(r['detail_json']) for r in rows], [{'enabled': not was}])
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_server_catchup_switch_persists(self):
        """FEAT-15: toggling 'Allow /catchup in channels without characters' and saving persists it and audits settings.catchup."""
        context, page, errors = self.ux_page('/admin/guild/1')
        try:
            page.get_by_role('tab', name='Server setup', exact=True).click()
            switch = page.get_by_role('switch', name='Allow /catchup in channels without characters')
            switch.wait_for(timeout=5000)
            self.assertEqual(switch.get_attribute('aria-checked'), 'false')
            switch.click()
            page.get_by_role('region', name='Unsaved changes').get_by_role('button', name='Save changes', exact=True).click()
            page.get_by_text('Server settings saved', exact=True).wait_for(timeout=5000)
            self.wait_for(lambda: self.state()['catchup_anywhere'] is True)
            rows = [r for r in self.state()['audit'] if r['action'] == 'settings.catchup']
            self.assertEqual([json.loads(r['detail_json']) for r in rows], [{'enabled': True}])
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def usage_table(self, page, index):
        """Rows of the nth usage table as lists of cell texts."""
        return page.locator('.ll-usage').nth(index).locator('tbody tr').evaluate_all('rows => rows.map(r => [...r.cells].map(c => c.textContent.trim()))')

    def test_monitoring_tab_shows_usage_for_this_server_only(self):
        """FEAT-12: the Monitoring tab charts the last 7 days and lists usage by model, channel and feature; another server's calls and the old Server setup table are absent."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=monitoring')
        try:
            page.get_by_text('By model', exact=True).wait_for(timeout=5000)
            self.assertEqual(page.locator('.ll-section-title', has_text='Usage').count(), 1)
            page.locator('canvas').first.wait_for()
            self.assertEqual(self.usage_table(page, 0)[0][:3], ['second-model', 'dialogue, director, memory', '2'])
            models = {row[0]: row for row in self.usage_table(page, 0)}
            self.assertEqual(models['fixture-model'][2:], ['5', '4,323', '715', '$0.0085', '0'])
            self.assertEqual(models['second-model'][2:], ['2', '7,000', '900', '$0.01', '1'])
            self.assertEqual(models['fixture-model'][1], 'dialogue, director, memory')
            channels = {row[0]: row[1] for row in self.usage_table(page, 1)}
            self.assertEqual(channels, {'#scene': '4', 'Deleted channel (ID …777)': '1', 'Unknown (before this update)': '2'})
            features = {row[0]: row[1] for row in self.usage_table(page, 2)}
            self.assertEqual(features, {'Replies': '1', 'Ambient turns': '1', 'Summons': '1', 'Memory updates': '1', 'Catch-ups': '1', 'Unknown (before this update)': '2'})
            self.assertEqual(page.get_by_text('other-guild-model').count(), 0)
            page.get_by_text('Tracked for this server, including internal model calls.', exact=False).wait_for()
            page.get_by_role('tab', name='Server setup', exact=True).click()
            _worlds_section(page).get_by_label('Name', exact=True).wait_for()
            self.assertEqual(page.get_by_text('Models and usage', exact=False).count(), 0)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_monitoring_range_change_updates_totals(self):
        """FEAT-12: switching the period re-renders the tables; Last 24 hours only has the three recent calls."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=monitoring')
        try:
            page.get_by_text('By model', exact=True).wait_for(timeout=5000)
            self.assertEqual({row[0] for row in self.usage_table(page, 0)}, {'fixture-model', 'second-model'})
            page.get_by_label('Period', exact=True).click()
            page.get_by_role('option', name='Last 24 hours', exact=True).click()
            self.wait_for(lambda: [row[0] for row in self.usage_table(page, 0)] == ['fixture-model'])
            self.assertEqual(self.usage_table(page, 0)[0][2:5], ['3', '1,623', '345'])
            page.get_by_label('Period', exact=True).click()
            days = self.state()['turn_log']['days']
            shown = {7: 2, 14: 3, 30: 5, 0: 3}[days]  # options longer than the retention are hidden, with a note
            self.assertEqual(page.get_by_role('option').count(), shown)
            page.keyboard.press('Escape')
            if shown < 5:
                page.get_by_text(f'Usage is kept for {days} days (Server settings).' if days else 'Usage is kept for the current month (Server settings).').wait_for()
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_monitoring_fits_phone_width(self):
        """FEAT-12: at 390 px the chart and tables stay inside the page (no horizontal scroll) and the chart is drawn."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=monitoring')
        try:
            page.set_viewport_size({'width': 390, 'height': 844})
            page.get_by_text('By feature', exact=True).wait_for(timeout=5000)
            page.wait_for_timeout(500)
            self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'))
            box = page.locator('canvas').first.bounding_box()
            self.assertTrue(0 < box['width'] <= 390 and box['height'] > 100, box)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def log_rows(self, page):
        return page.locator('.ll-log-row')

    def open_log(self, width=None):
        context, page, errors = self.ux_page('/admin/guild/1?tab=monitoring')
        if width:
            page.set_viewport_size({'width': width, 'height': 844})
        page.get_by_text('Personal facts are hidden and API keys removed.', exact=False).wait_for(timeout=5000)
        self.log_rows(page).first.wait_for(timeout=5000)
        return context, page, errors

    def test_turn_log_lists_this_server_newest_first(self):
        """FEAT-07: the Log section shows 50 guild-1 entries newest first with channel, model and tokens, and never the guild-2 canary."""
        context, page, errors = self.open_log()
        try:
            self.assertEqual(self.log_rows(page).count(), 50)
            self.assertIn('summon · dialogue', self.log_rows(page).first.inner_text())
            self.assertIn('Deleted channel (ID …777)', self.log_rows(page).first.inner_text())
            self.assertIn('12 in / 34 out', self.log_rows(page).first.inner_text())
            self.assertRegex(self.log_rows(page).nth(2).inner_text(), r'\d{4}-\d\d-\d\d \d\d:\d\d · #scene · reply · dialogue')
            page.get_by_text('Entries are kept for', exact=False).wait_for()
            self.assertEqual(page.get_by_text('OTHER-GUILD-CANARY', exact=False).count(), 0)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_turn_log_filters(self):
        """FEAT-07: Errors only, Channel and Reference ID (with a pasted 'ref ' prefix) each narrow the list; no match says so."""
        context, page, errors = self.open_log()
        try:
            page.get_by_role('switch', name='Errors only').click()
            self.wait_for(lambda: self.log_rows(page).count() == 1)
            self.assertIn('Ref 06f1ee', self.log_rows(page).first.inner_text())
            page.get_by_role('switch', name='Errors only').click()
            self.wait_for(lambda: self.log_rows(page).count() == 50)
            page.get_by_label('Reference ID', exact=True).fill('(ref 06F1EE)')
            page.get_by_label('Reference ID', exact=True).press('Enter')
            self.wait_for(lambda: self.log_rows(page).count() == 1)
            page.get_by_label('Reference ID', exact=True).fill('NOPE')
            page.get_by_label('Reference ID', exact=True).press('Enter')
            page.get_by_text('No entries match these filters.', exact=True).wait_for(timeout=5000)
            page.get_by_label('Reference ID', exact=True).fill('')
            page.get_by_label('Reference ID', exact=True).press('Enter')
            self.wait_for(lambda: self.log_rows(page).count() == 50)
            page.get_by_label('Channel', exact=True).click()
            page.get_by_role('option', name='Deleted channel (ID …777)', exact=True).click()
            self.wait_for(lambda: self.log_rows(page).count() == 1)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_turn_log_expand_and_load_more(self):
        """FEAT-07: expanding an entry loads its request, response and error text; Load more appends the 8 older entries and then hides."""
        context, page, errors = self.open_log()
        try:
            self.log_rows(page).first.click()
            page.locator('.ll-log-block', has_text='FIXTURE-REQUEST-TEXT').wait_for(timeout=5000)
            page.locator('.ll-log-block', has_text='FIXTURE-RESPONSE-TEXT').wait_for()
            self.log_rows(page).nth(1).click()
            page.locator('.ll-log-block', has_text='FIXTURE-ERROR-REQUEST').wait_for(timeout=5000)
            page.locator('.ll-log-block', has_text='Provider returned 500 FIXTURE-ERROR-DETAIL').wait_for()
            page.get_by_role('button', name='Load more').click()
            self.wait_for(lambda: self.log_rows(page).count() == 58)
            self.wait_for(lambda: not page.get_by_role('button', name='Load more').is_visible())
            self.assertIn('Log reply 1', self.log_rows(page).last.inner_text())
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_turn_log_fits_phone_width(self):
        """FEAT-07: at 390 px the log, including an expanded entry, causes no horizontal page scroll."""
        context, page, errors = self.open_log(390)
        try:
            self.log_rows(page).nth(1).click()
            page.locator('.ll-log-block', has_text='FIXTURE-ERROR-REQUEST').wait_for(timeout=5000)
            page.wait_for_timeout(300)
            self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'))
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_server_turn_log_settings_persist(self):
        """D22 step 4: the Keep a turn log switch and retention save together as one settings.turn_log audit row."""
        settings_rows = lambda: [r for r in self.state()['audit'] if r['action'] == 'settings.turn_log']
        earlier_rows = len(settings_rows())  # the MNT-28 retention test also saves this setting
        context, page, errors = self.ux_page('/admin/guild/1')
        try:
            page.get_by_role('tab', name='Server setup', exact=True).click()
            switch = page.get_by_role('switch', name='Keep a turn log')
            switch.wait_for(timeout=5000)
            self.assertEqual(switch.get_attribute('aria-checked'), 'false')
            switch.click()
            page.get_by_label('Keep usage and log for', exact=True).click()
            page.get_by_role('option', name='30 days', exact=True).click()
            page.get_by_role('region', name='Unsaved changes').get_by_role('button', name='Save changes', exact=True).click()
            page.get_by_text('Server settings saved', exact=True).wait_for(timeout=5000)
            self.wait_for(lambda: self.state()['turn_log'] == {'enabled': True, 'days': 30})
            rows = settings_rows()[earlier_rows:]
            self.assertEqual([json.loads(r['detail_json']) for r in rows], [{'enabled': True, 'days': 30}])
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def post_hook(self, path, payload=None, **params):
        response = self.context.request.post(self.url + path, data=payload, params=params or None)
        self.assertEqual(response.status, 200, path)

    def set_turn_log(self, enabled, days):
        self.post_hook('/_test/turn-log-settings', {'enabled': enabled, 'days': days})

    def test_turn_log_off_with_earlier_entries_says_so(self):
        """Characterization (MNT-28): with the turn log switched off but entries still kept, the Log lists them and notes that logging is off."""
        before = self.state()['turn_log']
        self.set_turn_log(False, 14)
        try:
            context, page, errors = self.open_log()
            try:
                page.get_by_text('Logging is off; showing earlier entries.', exact=True).wait_for(timeout=5000)
                self.assertEqual(self.log_rows(page).count(), 50)
                self.assertEqual(page.get_by_text('The turn log is off.', exact=False).count(), 0)
                self.assertFalse(errors, (errors, getattr(page, 'network', [])))
            finally:
                context.close()
        finally:
            self.set_turn_log(before['enabled'], before['days'])

    def test_deleted_channels_get_distinct_labels(self):
        """MNT-28: two deleted channels are told apart by the last 4 digits of their ID in the Log's Channel filter and the By channel table; an unknown channel keeps its own label."""
        ids = (900000001234, 900000005678)
        for channel in ids:
            self.post_hook('/_test/monitoring-channel', {'channel_id': channel, 'present': True})
        try:
            context, page, errors = self.open_log()
            try:
                page.get_by_text('By channel', exact=True).wait_for(timeout=5000)
                wanted = {'Deleted channel (ID \u20261234)', 'Deleted channel (ID \u20265678)'}
                channels = [row[0] for row in self.usage_table(page, 1)]
                self.assertLessEqual(wanted, set(channels), channels)
                self.assertIn('Unknown (before this update)', channels)
                page.get_by_label('Channel', exact=True).click()
                page.get_by_role('option').first.wait_for()
                options = page.get_by_role('option').all_inner_texts()
                self.assertLessEqual(wanted, set(options), options)
                page.keyboard.press('Escape')
            finally:
                context.close()
        finally:
            for channel in ids:
                self.post_hook('/_test/monitoring-channel', {'channel_id': channel, 'present': False})

    def test_monitoring_notes_follow_a_retention_change_without_reload(self):
        """MNT-28: after saving a new 'Keep usage and log for' value, opening Monitoring in the same page shows the new retention notes and period list."""
        before = self.state()['turn_log']
        self.set_turn_log(False, 14)
        try:
            context, page, errors = self.ux_page('/admin/guild/1?tab=monitoring')
            try:
                page.get_by_text('Usage is kept for 14 days (Server settings).', exact=True).wait_for(timeout=5000)
                page.get_by_role('tab', name='Server setup', exact=True).click()
                page.get_by_label('Keep usage and log for', exact=True).click()
                page.get_by_role('option', name='7 days', exact=True).click()
                page.get_by_role('region', name='Unsaved changes').get_by_role('button', name='Save changes', exact=True).click()
                page.get_by_text('Server settings saved', exact=True).wait_for(timeout=5000)
                self.wait_for(lambda: self.state()['turn_log']['days'] == 7)
                page.get_by_role('tab', name='Monitoring', exact=True).click()
                page.get_by_text('Personal facts are hidden and API keys removed.', exact=False).wait_for(timeout=5000)
                self.assertEqual(page.get_by_text('Usage is kept for 7 days (Server settings).', exact=True).count(), 1)
                self.assertEqual(page.get_by_text('Entries are kept for 7 days (Server settings).', exact=False).count(), 1)
                page.get_by_label('Period', exact=True).click()
                page.get_by_role('option').first.wait_for()
                self.assertEqual(page.get_by_role('option').count(), 2)
                page.keyboard.press('Escape')
            finally:
                context.close()
        finally:
            self.set_turn_log(before['enabled'], before['days'])

    def test_channel_logging_after_page_build_joins_the_filter(self):
        """MNT-28: a channel that first logs after the page was built shows up in the Log's Channel filter once the list is reloaded, without a page reload."""
        context, page, errors = self.open_log()
        try:
            channel = page.get_by_label('Channel', exact=True)
            channel.click()
            page.get_by_role('option', name='All channels', exact=True).wait_for()
            self.assertEqual(page.get_by_role('option', name='#assets', exact=True).count(), 0)
            page.keyboard.press('Escape')
            self.post_hook('/_test/monitoring-channel', {'channel_id': 200, 'present': True})
            page.get_by_role('switch', name='Errors only').click()
            self.wait_for(lambda: self.log_rows(page).count() == 1)
            page.get_by_role('switch', name='Errors only').click()
            self.wait_for(lambda: self.log_rows(page).count() == 50)
            channel.click()
            page.get_by_role('option', name='All channels', exact=True).wait_for()
            self.assertEqual(page.get_by_role('option', name='#assets', exact=True).count(), 1)
            page.keyboard.press('Escape')
        finally:
            context.close()
            self.post_hook('/_test/monitoring-channel', {'channel_id': 200, 'present': False})

    def test_retention_saved_from_the_monitoring_tab_rebuilds_it(self):
        """MNT-28: editing retention on Server setup, switching to Monitoring and saving from there leaves Monitoring showing the new notes, not blank."""
        before = self.state()['turn_log']
        self.set_turn_log(False, 14)
        try:
            context, page, errors = self.ux_page('/admin/guild/1?tab=monitoring')
            try:
                page.get_by_text('Usage is kept for 14 days (Server settings).', exact=True).wait_for(timeout=5000)
                page.get_by_role('tab', name='Server setup', exact=True).click()
                page.get_by_label('Keep usage and log for', exact=True).click()
                page.get_by_role('option', name='7 days', exact=True).click()
                page.get_by_role('tab', name='Monitoring', exact=True).click()
                page.get_by_role('region', name='Unsaved changes').get_by_role('button', name='Save changes', exact=True).click()
                page.get_by_text('Server settings saved', exact=True).wait_for(timeout=5000)
                page.get_by_text('Usage is kept for 7 days (Server settings).', exact=True).wait_for(timeout=5000)
                self.assertEqual(page.get_by_text('Entries are kept for 7 days (Server settings).', exact=False).count(), 1)
            finally:
                context.close()
        finally:
            self.set_turn_log(before['enabled'], before['days'])

    def test_retention_saved_before_a_failing_part_still_rebuilds_monitoring(self):
        """MNT-29: when the retention part saves but a later Server settings part fails while Monitoring is shown, Monitoring is rebuilt with the new notes, not left blank; the failed part stays unsaved."""
        before = self.state()['turn_log']
        self.set_turn_log(False, 14)
        try:
            context, page, errors = self.ux_page('/admin/guild/1?tab=monitoring')
            try:
                page.get_by_text('Usage is kept for 14 days (Server settings).', exact=True).wait_for(timeout=5000)
                page.get_by_role('tab', name='Server setup', exact=True).click()
                page.get_by_label('Keep usage and log for', exact=True).click()
                page.get_by_role('option', name='7 days', exact=True).click()
                select = page.get_by_label('Server timezone', exact=True)
                select.click()
                select.fill('Asia/Seoul')
                page.get_by_role('option', name='Asia/Seoul', exact=True).click()
                page.get_by_role('tab', name='Monitoring', exact=True).click()
                self.post_hook('/_test/fail-timezone', n=1)
                page.get_by_role('region', name='Unsaved changes').get_by_role('button', name='Save changes', exact=True).click()
                page.get_by_text('Fixture timezone failure', exact=True).wait_for(timeout=5000)
                page.get_by_text('Usage is kept for 7 days (Server settings).', exact=True).wait_for(timeout=5000)
                self.assertEqual(page.get_by_text('Entries are kept for 7 days (Server settings).', exact=False).count(), 1)
                self.assertEqual(self.state()['turn_log']['days'], 7)
                self.assertEqual(self.state()['guild_timezone'], '')
                self.assertTrue(page.get_by_role('region', name='Unsaved changes').is_visible())
            finally:
                context.close()
        finally:
            self.post_hook('/_test/fail-timezone', n=0)
            self.set_turn_log(before['enabled'], before['days'])

    def test_failed_filter_read_restores_the_controls(self):
        """MNT-28: when the read for a changed filter fails, the filter controls go back to the applied values so they agree with the list."""
        from playwright.sync_api import expect
        context, page, errors = self.open_log()
        try:
            switch = page.get_by_role('switch', name='Errors only')
            self.post_hook('/_test/fail-turn-log-page', n=1)
            switch.click()
            page.get_by_text('Fixture turn log read failure', exact=True).wait_for(timeout=5000)
            self.assertEqual(self.log_rows(page).count(), 50)  # the list still shows the earlier filter
            expect(switch).to_have_attribute('aria-checked', 'false', timeout=2000)
        finally:
            context.close()
            self.post_hook('/_test/fail-turn-log-page', n=0)

    def test_channel_card_is_one_save_bar_editor(self):
        """UI-15 / MNT-26: a collapsed channel card saves guidelines and ambient as one edit with one audit row each, can be saved again without a conflict, and Reset restores."""
        context, page, errors = self.ux_page('/admin/guild/1')
        try:
            from playwright.sync_api import expect
            def count(action):
                return len([r for r in self.state()['audit'] if r['action'] == action])
            card = page.locator('.channel-card').first
            toggle = card.locator('button.channel-toggle')
            bar = page.get_by_role('region', name='Unsaved changes')
            guidelines = card.get_by_label('Channel guidelines', exact=True)
            self.assertEqual(toggle.get_attribute('aria-expanded'), 'false')
            toggle.click()
            expect(toggle).to_have_attribute('aria-expanded', 'true')
            before = {a: count(a) for a in ('guidelines.edit', 'channel.ambient')}
            guidelines.fill('First edit.')
            card.get_by_role('switch', name='Ambient participation', exact=True).click()
            bar.get_by_role('button', name='Save changes', exact=True).click()
            page.get_by_text('Channel settings saved', exact=True).wait_for(timeout=5000)
            bar.wait_for(state='hidden')
            self.wait_for(lambda: count('guidelines.edit') == before['guidelines.edit'] + 1)
            self.assertEqual(count('channel.ambient'), before['channel.ambient'] + 1)
            toggle.get_by_text('Ambient', exact=True).wait_for()
            guidelines.fill('Second edit.')
            bar.get_by_role('button', name='Save changes', exact=True).click()
            bar.wait_for(state='hidden')
            self.wait_for(lambda: count('guidelines.edit') == before['guidelines.edit'] + 2)
            self.assertEqual(count('channel.ambient'), before['channel.ambient'] + 1)
            self.assertEqual(page.get_by_text('Guidelines changed; reload before saving', exact=False).count(), 0)
            guidelines.fill('Discarded edit.')
            bar.get_by_role('button', name='Reset', exact=True).click()
            expect(guidelines).to_have_value('Second edit.')
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
            named = _section(page, 'Named lorebooks')
            self.assertFalse(named.get_by_label('Channel', exact=True).is_visible())
            named.get_by_label('Kind', exact=True).click()
            page.get_by_role('option', name='Channel lorebook', exact=True).click()
            named.get_by_label('Channel', exact=True).wait_for(state='visible')
            named.get_by_label('Channel', exact=True).click()
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
            self.load(page, self.url + '/admin/guild/1')
            _worlds_section(page).get_by_label('Name', exact=True).wait_for()
            page.wait_for_timeout(300)
            calls = context.request.get(self.url + '/_test/metrics').json()
            self.assertEqual(calls.get('GET /guilds/1/channels', 0), 1, calls)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_server_name_with_markup_renders_as_text(self):
        """UI-28: a server name with markup and quotes is plain text on the servers page and the server header."""
        from playwright.sync_api import expect
        name = '<b>"Moon</b> & Tavern'
        initials = '<&'  # first character of the first two whitespace-separated words
        context, page, errors = self.ux_page('/admin/')
        try:
            self.assertEqual(context.request.post(self.url + '/_test/guild-name', data={'name': name}).status, 200)
            page.goto(self.url + '/admin/')
            card = page.get_by_role('link', name=name, exact=True)
            expect(card).to_have_count(1)
            self.assertEqual(page.locator('a.ll-server-card').count(), 1)
            expect(card.locator('.ll-server-name')).to_have_text(name)
            expect(card.locator('.ll-server-tile')).to_have_text(initials)
            self.assertEqual(card.locator('b').count(), 0)
            self.assertEqual(card.locator('img.ll-server-icon').count(), 0)
            card.click()
            page.wait_for_url('**/admin/guild/1')
            crumb = page.locator('.ll-crumb')
            expect(crumb.locator('.ll-crumb-name')).to_have_text(name)
            expect(crumb.locator('.ll-crumb-tile')).to_have_text(initials)
            self.assertEqual(crumb.locator('b').count(), 0)
            expect(page.get_by_role('link', name='llmcord / Servers', exact=True)).to_have_count(1)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.request.post(self.url + '/_test/guild-name', data={'name': 'Test server'})
            context.close()

    def test_account_menu_in_header_with_sign_out(self):
        """UI-29: the header account button opens a menu with the display name and a CSRF-carrying Sign out form; the old bottom button is gone."""
        from playwright.sync_api import expect
        context = self.browser.new_context(ignore_https_errors=True, viewport={'width': 1400, 'height': 1000})
        context.add_cookies([{'name': 'llmcord_session', 'value': 'browser-account-session', 'url': self.url,
                              'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
        page = context.new_page()
        errors = []
        self.watch(page, errors)
        try:
            page.goto(self.url + '/admin/')
            expect(page.get_by_role('link', name='llmcord / Servers', exact=True)).to_be_visible()
            expect(page.get_by_role('button', name='Account menu')).to_be_visible()
            expect(page.get_by_text('Sign out')).to_have_count(0)
            page.goto(self.url + '/admin/guild/1')
            button = page.get_by_role('button', name='Account menu')
            expect(button).to_be_visible()
            button.click()
            menu = page.locator('.ll-menu')
            expect(menu.get_by_text('Test admin', exact=True)).to_be_visible()
            sign_out = menu.get_by_role('button', name='Sign out')
            expect(sign_out).to_be_visible()
            form = menu.locator('form')
            self.assertEqual(form.get_attribute('method'), 'post')
            self.assertTrue(form.get_attribute('action').endswith('/logout'))
            self.assertEqual(form.locator('input[name=csrf]').get_attribute('type'), 'hidden')
            self.assertEqual(form.locator('input[name=csrf]').get_attribute('value'), 'browser-account-csrf')
            sign_out.click()
            page.wait_for_url(lambda url: not url.rstrip('/').endswith('/guild/1'))
            expect(page.get_by_text('Sign in with Discord')).to_be_visible()
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_operator_spending_caps_panel(self):
        """FEAT-16 step 7: the operator saves caps, reset day and the channel notice; invalid and stale saves toast and write nothing."""
        from playwright.sync_api import expect
        context = self.browser.new_context(ignore_https_errors=True, viewport={'width': 1000, 'height': 900})
        context.add_cookies([{'name': 'llmcord_session', 'value': 'browser-operator-session', 'url': self.url,
                              'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
        page = context.new_page()
        errors = []
        self.watch(page, errors)
        def settings():
            return context.request.get(self.url + '/_test/budget').json()
        try:
            page.goto(self.url + '/admin/operator')
            expect(page.get_by_text('$0.001 spent since')).to_be_visible()
            expect(page.get_by_text('(UTC).')).to_be_visible()
            expect(page.get_by_label('Soft cap (USD)')).to_have_value('')
            expect(page.get_by_label('Hard cap (USD)')).to_have_value('')
            expect(page.get_by_text('Blank means off')).to_have_count(2)
            expect(page.get_by_text('without a cost estimate')).to_have_count(0)
            self.assertEqual(settings()['revision'], 0)
            page.get_by_label('Soft cap (USD)').fill('10')
            page.get_by_label('Hard cap (USD)').fill('5')
            page.get_by_role('button', name='Save').click()
            expect(page.get_by_text('The soft cap cannot be higher than the hard cap.')).to_be_visible()
            self.assertEqual(settings()['revision'], 0)
            page.get_by_label('Soft cap (USD)').fill('5')
            page.get_by_label('Hard cap (USD)').fill('10')
            page.get_by_label('Reset day').click()
            page.get_by_role('option', name='15', exact=True).click()
            page.get_by_role('switch').click()
            page.get_by_role('button', name='Save').click()
            expect(page.get_by_text('Spending caps saved.')).to_be_visible()
            saved = settings()
            self.assertEqual((saved['revision'], saved['soft_cap_usd'], saved['hard_cap_usd'], saved['reset_day'], saved['channel_notice']), (1, 5.0, 10.0, 15, 1))
            page.reload()
            expect(page.get_by_label('Soft cap (USD)')).to_have_value('5')
            expect(page.get_by_label('Hard cap (USD)')).to_have_value('10')
            expect(page.get_by_label('Reset day')).to_have_value('15')
            expect(page.get_by_role('switch')).to_have_attribute('aria-checked', 'true')
            # A change made behind the page's back makes the open form stale.
            self.assertTrue(context.request.post(self.url + '/_test/budget-save').ok)
            page.get_by_role('button', name='Save').click()
            expect(page.get_by_text('Bot settings changed; reload before saving')).to_be_visible()
            self.assertEqual(settings()['revision'], 2)
            expect(page.get_by_label('Soft cap (USD)')).to_have_value('5')
            page.get_by_role('button', name='Save').click()
            expect(page.get_by_text('Spending caps saved.')).to_be_visible()
            self.assertEqual(settings()['revision'], 3)
            # Unpriced calls show a warning.
            self.assertTrue(context.request.post(self.url + '/_test/budget-unpriced?n=2').ok)
            page.reload()
            expect(page.get_by_text('2 model calls without a cost estimate this period were counted as $0.')).to_be_visible()
            # Phone width: nothing scrolls sideways.
            page.set_viewport_size({'width': 390, 'height': 900})
            expect(page.get_by_role('button', name='Save')).to_be_visible()
            self.assertLessEqual(page.evaluate('document.documentElement.scrollWidth'), 390)
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_operator_bot_settings_page_and_menu(self):
        """FEAT-08: operators get a Bot settings item in the account menu that opens /operator; others (and signed-out visitors) get the unknown-page 404."""
        from playwright.sync_api import expect
        def plain_404(page, path):
            response = page.goto(self.url + path)
            return response.status, len(page.content())
        context = self.browser.new_context(ignore_https_errors=True, viewport={'width': 1000, 'height': 800})
        context.add_cookies([{'name': 'llmcord_session', 'value': 'browser-operator-session', 'url': self.url,
                              'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
        page = context.new_page()
        errors = []
        self.watch(page, errors)
        try:
            page.goto(self.url + '/admin/guild/1')
            page.get_by_role('button', name='Account menu').click()
            item = page.locator('.ll-menu').get_by_role('link', name='Bot settings')
            expect(item).to_be_visible()
            item.click()
            page.wait_for_url('**/admin/operator')
            expect(page.get_by_text('Bot-wide settings. Only operators can see this page.')).to_be_visible()
            expect(page.get_by_text('Spending caps', exact=True)).to_be_visible()
            self.assertEqual(page.title(), 'Bot settings')
            self.assertFalse(errors, errors)
        finally:
            context.close()
        for session in ('browser-plain-session', None):
            context = self.browser.new_context(ignore_https_errors=True)
            if session:
                context.add_cookies([{'name': 'llmcord_session', 'value': session, 'url': self.url,
                                      'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
            page = context.new_page()
            try:
                if session:
                    page.goto(self.url + '/admin/')
                    page.get_by_role('button', name='Account menu').click()
                    expect(page.get_by_text('Sign out')).to_be_visible()
                    expect(page.get_by_text('Bot settings')).to_have_count(0)
                self.assertEqual(plain_404(page, '/admin/operator'), plain_404(page, '/admin/no-such-page'))
                self.assertEqual(plain_404(page, '/admin/operator')[0], 404)
            finally:
                context.close()

    def operator_page(self, path, width=1000):
        context = self.browser.new_context(ignore_https_errors=True, viewport={'width': width, 'height': 900})
        context.add_cookies([{'name': 'llmcord_session', 'value': 'browser-operator-session', 'url': self.url,
                              'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
        page = context.new_page()
        errors = []
        self.watch(page, errors)
        page.goto(self.url + path)
        return context, page, errors

    def test_operator_bot_settings_tabs_and_usage_by_server(self):
        """FEAT-13: Bot settings has Spending and Usage by server tabs; the table names known servers, falls back to the ID, sorts by cost and follows the period."""
        from playwright.sync_api import expect
        context, page, errors = self.operator_page('/admin/operator')
        try:
            expect(page.get_by_role('tab', name='Spending')).to_be_visible()
            expect(page.get_by_role('tab', name='Usage by server')).to_be_visible()
            expect(page.get_by_text('Spending caps', exact=True)).to_be_visible()
            page.get_by_role('tab', name='Usage by server').click()
            expect(page.get_by_role('combobox', name='Period')).to_have_value('This spending period')
            expect(page.get_by_text('Each server keeps usage for its own period (Server settings), so this table can show less than the Spending tab.')).to_be_visible()
            rows = page.locator('tbody tr')
            expect(rows.first).to_contain_text('Server (ID …2)')
            expect(rows.first).to_contain_text('$5.00')
            expect(page.locator('tbody tr', has_text='Test server')).to_have_count(1)
            self.assertIn('tab=usage', page.url)
            page.get_by_label('Period').click()
            page.get_by_role('option', name='Last 7 days').click()
            expect(page.locator('tbody tr', has_text='Total')).to_contain_text('$5.02')
            expect(page.locator('tbody tr', has_text='Server (ID …3)')).to_have_count(0)
            page.get_by_label('Period').click()
            page.get_by_role('option', name='Last 30 days').click()
            expect(page.locator('tbody tr', has_text='Server (ID …3)')).to_have_count(1)
            expect(page.locator('tbody tr', has_text='Total')).to_contain_text('$5.27')
            page.get_by_role('tab', name='Spending').click()
            expect(page.get_by_label('Soft cap (USD)')).to_be_visible()
            page.goto(self.url + '/admin/operator?tab=usage')
            expect(page.get_by_text('Each server keeps usage for its own period')).to_be_visible()
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_operator_usage_by_server_fits_phone_width(self):
        """FEAT-13: at 390 px the Usage by server tab shows its servers and does not scroll sideways."""
        from playwright.sync_api import expect
        context, page, errors = self.operator_page('/admin/operator?tab=usage', 390)
        try:
            expect(page.get_by_text('Each server keeps usage for its own period')).to_be_visible()
            page.get_by_label('Period').click()
            page.get_by_role('option', name='Last 7 days').click()
            expect(page.get_by_text('Server (ID …2)')).to_be_visible()
            page.wait_for_timeout(500)
            self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'))
            self.assertFalse(errors, errors)
        finally:
            context.close()

    @staticmethod
    def shot(page, name, **options):
        """Documentation screenshots are written only when LLMCORD_SHOTS_DIR is set."""
        folder = os.environ.get('LLMCORD_SHOTS_DIR')
        if folder:
            Path(folder).mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(Path(folder) / name), **options)

    @staticmethod
    def backend_row(page, name):
        """The Profiles list row for `name` (row texts such as 'From: config.yaml' repeat, so rows are found by name)."""
        return page.locator('.ll-result').filter(has=page.get_by_text(name, exact=True))

    def fill_profile_editor(self, page, name, model, tokens, key='', base=''):
        dialog = page.get_by_role('dialog')
        dialog.get_by_label('Name', exact=True).fill(name)
        dialog.get_by_label('Model', exact=True).fill(model)
        if key:
            dialog.get_by_label('API key', exact=True).fill(key)
        if base:
            dialog.get_by_label('Base URL', exact=True).fill(base)
        dialog.get_by_label('Context tokens', exact=True).fill(str(tokens))
        return dialog

    def test_operator_backend_tab_lists_config_profiles(self):
        """D24 step 4 (FEAT-11): the Backend tab lists a config.yaml profile with its key status and source, and the role selects offer the config default."""
        from playwright.sync_api import expect
        context, page, errors = self.operator_page('/admin/operator?tab=backend')
        try:
            expect(page.get_by_role('tab', name='Backend')).to_be_visible()
            expect(page.get_by_text('Roles', exact=True)).to_be_visible()
            expect(page.get_by_text('Profiles', exact=True)).to_be_visible()
            row = self.backend_row(page, 'cfg-main')
            expect(row).to_contain_text('openai · cfg-model')
            expect(row).to_contain_text('Key: ${FIXTURE_API_KEY} (missing)')
            expect(row).to_contain_text('From: config.yaml')
            for role in ('Dialogue', 'Director', 'Memory'):
                expect(row).to_contain_text(f'{role} role')
                expect(page.get_by_label(role, exact=True)).to_have_value('Use config.yaml (cfg-main)')
            expect(self.backend_row(page, 'cfg-second')).to_contain_text('Key: No key')
            expect(page.get_by_role('button', name='Edit cfg-main')).to_be_visible()
            expect(page.get_by_role('button', name='Delete cfg-main')).to_have_count(0)
            expect(page.get_by_role('button', name='Reset to config.yaml for cfg-main')).to_have_count(0)
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_operator_backend_add_keyless_profile_and_assign_director(self):
        """D24 step 4 (FEAT-11): add a compatible profile without a key, assign it to Director; it survives a reload and the audit rows carry names only."""
        from playwright.sync_api import expect
        context, page, errors = self.operator_page('/admin/operator?tab=backend')
        before = len(self.state()['audit'])
        try:
            expect(page.get_by_role('button', name='Add profile')).to_be_visible()
            page.get_by_role('button', name='Add profile').click()
            dialog = self.fill_profile_editor(page, 'tmp-keyless', 'tmp-secret-model-name', 4096, base='https://tmp.example/v1')
            dialog.get_by_role('button', name='Save profile').click()
            expect(page.get_by_text('Profile saved.')).to_be_visible()
            expect(page.get_by_role('dialog')).to_have_count(0)
            row = self.backend_row(page, 'tmp-keyless')
            expect(row).to_contain_text('From: Dashboard')
            page.get_by_label('Director', exact=True).click()
            page.get_by_role('option', name='tmp-keyless', exact=True).click()
            page.get_by_role('button', name='Save roles').click()
            expect(page.get_by_text('Roles saved.')).to_be_visible()
            page.reload()
            row = self.backend_row(page, 'tmp-keyless')
            expect(row).to_contain_text('From: Dashboard')
            expect(row).to_contain_text('Key: No key')
            expect(row).to_contain_text('Director role')
            expect(self.backend_row(page, 'cfg-main')).not_to_contain_text('Director role')
            expect(page.get_by_label('Director', exact=True)).to_have_value('tmp-keyless')
            state = self.state()
            self.assertEqual([r['name'] for r in state['model_profiles']], ['tmp-keyless'])
            self.assertEqual(state['model_roles']['director'], 'tmp-keyless')
            audit = {r['action']: json.loads(r['detail_json']) for r in state['audit'][before:] if r['action'].startswith('backend.')}
            self.assertEqual(set(audit), {'backend.profile', 'backend.roles'})
            self.assertEqual((audit['backend.profile']['name'], audit['backend.profile']['created']), ('tmp-keyless', True))
            self.assertEqual(set(audit['backend.profile']), {'name', 'created', 'fields'})
            self.assertEqual(audit['backend.roles'], {'director': {'from': None, 'to': 'tmp-keyless'}})
            raw = json.dumps(audit)
            for value in ('tmp-secret-model-name', 'tmp.example', '4096'):
                self.assertNotIn(value, raw)
            # Screenshots (light theme): the tab at both widths.
            page.set_viewport_size({'width': 1000, 'height': 1100})
            expect(row).to_be_visible()
            self.shot(page, 'backend-tab-1000.png', full_page=True)
            page.set_viewport_size({'width': 390, 'height': 1100})
            expect(row).to_be_visible()
            page.wait_for_function('document.documentElement.scrollWidth <= window.innerWidth')
            self.shot(page, 'backend-tab-390.png', full_page=True)
            self.assertEqual(page.evaluate('[document.documentElement.scrollWidth, window.innerWidth]'), [390, 390])
            page.set_viewport_size({'width': 1000, 'height': 1100})
            page.get_by_role('button', name='Edit tmp-keyless').click()
            dialog = page.get_by_role('dialog')
            page.set_viewport_size({'width': 1000, 'height': 1500})
            dialog.get_by_text('Advanced', exact=True).click()
            expect(dialog.locator('.q-expansion-item--expanded')).to_have_count(1)
            expect(dialog.get_by_label('Max retries', exact=True)).to_be_visible()
            # Quasar animates the height in JS, so wait until the open content is fully inside the card (the card grew past its collapsed size).
            page.wait_for_function("() => { const c = document.querySelector('.q-dialog .q-card'); const x = c.querySelector('.q-expansion-item__content'); return !x.getAttribute('style') && c.getBoundingClientRect().bottom >= x.getBoundingClientRect().bottom; }")
            self.shot(page, 'backend-editor-advanced-1000.png')
            page.set_viewport_size({'width': 390, 'height': 844})
            expect(dialog.get_by_label('Max retries', exact=True)).to_be_attached()
            page.wait_for_function('document.documentElement.scrollWidth <= window.innerWidth')
            self.shot(page, 'backend-editor-390.png')
            # On a phone the open editor is taller than the screen: Save profile must still be reachable by scrolling, and nothing covers it.
            save = dialog.get_by_role('button', name='Save profile')
            save.scroll_into_view_if_needed()
            expect(save).to_be_in_viewport()
            save.click(trial=True)
            dialog.get_by_role('button', name='Cancel').click()
            expect(page.get_by_role('dialog')).to_have_count(0)
            self.assertFalse(errors, errors)
        finally:
            # Leave the shared fixture clean: Director back to config.yaml, then drop the profile. If the UI cleanup fails, fall back to the store.
            try:
                try:
                    page.goto(self.url + '/admin/operator?tab=backend')
                    page.get_by_label('Director', exact=True).click()
                    page.get_by_role('option', name='Use config.yaml (cfg-main)', exact=True).click()
                    page.get_by_role('button', name='Save roles').click()
                    expect(page.get_by_text('Roles saved.')).to_be_visible()
                    page.get_by_role('button', name='Delete tmp-keyless').click()
                    page.get_by_role('dialog').get_by_role('button', name='Delete profile').click()
                    expect(self.backend_row(page, 'tmp-keyless')).to_have_count(0)
                except Exception:
                    self.post_hook('/_test/backend-reset')
                    raise
            finally:
                context.close()

    def test_operator_backend_pinned_key_refused(self):
        """D24 step 4 (FEAT-11): a ${OPENAI_API_KEY} reference with a foreign base URL is refused with a toast; nothing is stored and the dialog keeps the input."""
        from playwright.sync_api import expect
        context, page, errors = self.operator_page('/admin/operator?tab=backend')
        before = self.state()
        try:
            page.get_by_role('button', name='Add profile').click()
            dialog = self.fill_profile_editor(page, 'tmp-evil', 'evil-model', 1000, key='${OPENAI_API_KEY}', base='https://evil.example/v1')
            dialog.get_by_role('button', name='Save profile').click()
            expect(page.get_by_text('OPENAI_API_KEY may only be sent to its pinned hosts')).to_be_visible()
            expect(dialog).to_be_visible()
            expect(dialog.get_by_label('Name', exact=True)).to_have_value('tmp-evil')
            expect(dialog.get_by_label('API key', exact=True)).to_have_value('${OPENAI_API_KEY}')
            expect(dialog.get_by_label('Base URL', exact=True)).to_have_value('https://evil.example/v1')
            after = self.state()
            self.assertEqual(after['model_profiles'], before['model_profiles'])
            self.assertEqual(after['model_roles'], before['model_roles'])
            self.assertEqual(len(after['audit']), len(before['audit']))
            dialog.get_by_role('button', name='Cancel').click()
            expect(self.backend_row(page, 'tmp-evil')).to_have_count(0)
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_operator_backend_override_and_reset_to_config(self):
        """D24 step 4 (FEAT-11): editing a config.yaml profile saves an override (Reset appears); Reset to config.yaml removes it and restores the file's version."""
        from playwright.sync_api import expect
        context, page, errors = self.operator_page('/admin/operator?tab=backend')
        before = len(self.state()['audit'])
        try:
            row = self.backend_row(page, 'cfg-second')
            expect(row).to_contain_text('From: config.yaml')
            page.get_by_role('button', name='Edit cfg-second').click()
            dialog = page.get_by_role('dialog')
            expect(dialog.get_by_label('Name', exact=True)).to_have_value('cfg-second')
            expect(dialog.get_by_label('Model', exact=True)).to_have_value('second-model')
            dialog.get_by_label('Model', exact=True).fill('overridden-model')
            dialog.get_by_role('button', name='Save profile').click()
            expect(page.get_by_text('Profile saved.')).to_be_visible()
            expect(row).to_contain_text('From: Dashboard (overrides config.yaml)')
            expect(row).to_contain_text('overridden-model')
            expect(page.get_by_role('button', name='Delete cfg-second')).to_have_count(0)
            page.reload()
            expect(self.backend_row(page, 'cfg-second')).to_contain_text('overridden-model')
            self.assertEqual([r['name'] for r in self.state()['model_profiles']], ['cfg-second'])
            page.get_by_role('button', name='Reset to config.yaml for cfg-second').click()
            page.get_by_role('dialog').get_by_role('button', name='Reset profile').click()
            row = self.backend_row(page, 'cfg-second')
            expect(row).to_contain_text('From: config.yaml')
            expect(row).to_contain_text('second-model')
            expect(row).not_to_contain_text('overridden-model')
            self.assertEqual(self.state()['model_profiles'], [])
            actions = [(r['action'], json.loads(r['detail_json'])) for r in self.state()['audit'][before:] if r['action'].startswith('backend.')]
            self.assertEqual([a for a, _ in actions], ['backend.profile', 'backend.profile_delete'])
            self.assertEqual(actions[1][1], {'name': 'cfg-second'})
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_operator_backend_duplicate_name_is_refused(self):
        """D24 step 4 (FEAT-11): Add profile with the name of an existing profile toasts that it exists and stores nothing."""
        from playwright.sync_api import expect
        context, page, errors = self.operator_page('/admin/operator?tab=backend')
        before = self.state()
        try:
            page.get_by_role('button', name='Add profile').click()
            dialog = self.fill_profile_editor(page, 'cfg-main', 'dup-model', 1000, base='https://dup.example/v1')
            dialog.get_by_role('button', name='Save profile').click()
            expect(page.get_by_text('A profile named cfg-main already exists. Use Edit to change it.')).to_be_visible()
            expect(dialog).to_be_visible()
            after = self.state()
            self.assertEqual((after['model_profiles'], after['model_roles']), (before['model_profiles'], before['model_roles']))
            self.assertEqual(len(after['audit']), len(before['audit']))
            dialog.get_by_role('button', name='Cancel').click()
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_operator_backend_delete_conflict_refreshes_the_list(self):
        """D24 step 4 (FEAT-11): deleting a profile changed behind the page's back shows the conflict toast, closes the dialog and redraws the list."""
        from playwright.sync_api import expect
        self.post_hook('/_test/backend-profile', {'name': 'tmp-stale', 'model': 'old-model'})
        context, page, errors = self.operator_page('/admin/operator?tab=backend')
        try:
            expect(self.backend_row(page, 'tmp-stale')).to_contain_text('old-model')
            page.get_by_role('button', name='Delete tmp-stale').click()
            self.post_hook('/_test/backend-profile', {'name': 'tmp-stale', 'model': 'newer-model'})
            page.get_by_role('dialog').get_by_role('button', name='Delete profile').click()
            expect(page.locator('.q-notification').filter(has_text=re.compile('changed|reload', re.I))).to_be_visible()
            expect(page.get_by_role('dialog')).to_have_count(0)
            expect(self.backend_row(page, 'tmp-stale')).to_contain_text('newer-model')
            self.assertEqual([r['name'] for r in self.state()['model_profiles']], ['tmp-stale'])
            self.assertFalse(errors, errors)
        finally:
            self.post_hook('/_test/backend-reset')
            context.close()

    def test_operator_backend_test_connection(self):
        """D24 step 5 (FEAT-11): Test connection sends the unsaved form values to the tester, shows its success or failure line, audits name and ok only, and stores nothing."""
        from playwright.sync_api import expect
        context, page, errors = self.operator_page('/admin/operator?tab=backend')
        before = self.state()
        try:
            self.post_hook('/_test/backend-tester', {'mode': 'ok'})
            page.get_by_role('button', name='Add profile').click()
            dialog = self.fill_profile_editor(page, 'tmp-probe', 'typed-not-saved', 2048, base='https://probe.example/v1')
            expect(dialog.get_by_text('Sends one short request from the dashboard with these settings. It is not counted in usage.')).to_be_visible()
            button = dialog.get_by_role('button', name='Test connection')
            button.click()
            result = dialog.get_by_role('status')
            expect(result).to_have_text('Connected. typed-not-saved replied in 42 ms.')
            expect(result).to_have_class(re.compile('text-positive'))
            expect(button).to_be_enabled()
            self.shot(page, 'backend-test-ok-1000.png')
            calls = self.context.request.get(self.url + '/_test/backend-tester-calls').json()
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['name'], 'tmp-probe')
            self.assertEqual((calls[0]['mapping']['model'], calls[0]['mapping']['context_tokens'], calls[0]['mapping']['base_url']),
                             ('typed-not-saved', 2048, 'https://probe.example/v1'))
            self.post_hook('/_test/backend-tester', {'mode': 'fail'})
            button.click()
            expect(result).to_have_text('Provider returned 401: bad credentials')
            expect(result).to_have_class(re.compile('text-negative'))
            self.shot(page, 'backend-test-fail-1000.png')
            after = self.state()
            self.assertEqual((after['model_profiles'], after['model_roles']), (before['model_profiles'], before['model_roles']))
            tests = [(r['guild_id'], json.loads(r['detail_json'])) for r in after['audit'][len(before['audit']):] if r['action'] == 'backend.test']
            self.assertEqual(tests, [(0, {'name': 'tmp-probe', 'ok': True}), (0, {'name': 'tmp-probe', 'ok': False})])
            self.assertFalse([r for r in after['audit'][len(before['audit']):] if r['action'] != 'backend.test'])
            dialog.get_by_role('button', name='Cancel').click()
            self.assertFalse(errors, errors)
        finally:
            self.post_hook('/_test/backend-tester', {'mode': 'ok'})
            context.close()

    def test_operator_backend_test_connection_button_disabled_while_running(self):
        """D24 step 5 (FEAT-11): while a test is running the button is disabled; it comes back with the result once the tester answers."""
        from playwright.sync_api import expect
        context, page, errors = self.operator_page('/admin/operator?tab=backend')
        try:
            self.post_hook('/_test/backend-tester', {'mode': 'ok', 'hold': True})
            page.get_by_role('button', name='Add profile').click()
            dialog = self.fill_profile_editor(page, 'tmp-slow', 'slow-model', 1000, base='https://slow.example/v1')
            button = dialog.get_by_role('button', name='Test connection')
            button.click()
            expect(button).to_be_disabled()
            expect(dialog.get_by_role('status')).to_have_count(0)
            self.post_hook('/_test/backend-tester-release')
            expect(dialog.get_by_role('status')).to_have_text('Connected. slow-model replied in 42 ms.')
            expect(button).to_be_enabled()
            dialog.get_by_role('button', name='Cancel').click()
            self.assertFalse(errors, errors)
        finally:
            self.post_hook('/_test/backend-tester', {'mode': 'ok'})
            context.close()

    def test_operator_backend_test_connection_survives_closing_the_dialog(self):
        """MNT-32: closing the editor while a test is still running leaves no error, no stray toast, and a usable page once the tester answers."""
        from playwright.sync_api import expect
        context, page, errors = self.operator_page('/admin/operator?tab=backend')
        try:
            self.post_hook('/_test/backend-tester', {'mode': 'ok', 'hold': True})
            page.get_by_role('button', name='Add profile').click()
            dialog = self.fill_profile_editor(page, 'tmp-closed', 'closed-model', 1000, base='https://closed.example/v1')
            dialog.get_by_role('button', name='Test connection').click()
            expect(dialog.get_by_role('button', name='Test connection')).to_be_disabled()
            dialog.get_by_role('button', name='Cancel').click()
            expect(page.get_by_role('dialog')).to_have_count(0)
            self.post_hook('/_test/backend-tester-release')
            page.wait_for_timeout(800)
            self.assertEqual(page.locator('.q-notification').count(), 0)
            page.get_by_role('button', name='Add profile').click()
            expect(page.get_by_role('dialog')).to_be_visible()
            expect(page.get_by_role('dialog').get_by_role('status')).to_have_count(0)
            page.get_by_role('dialog').get_by_role('button', name='Cancel').click()
            self.assertFalse(errors, errors)
        finally:
            self.post_hook('/_test/backend-tester', {'mode': 'ok'})
            context.close()

    def test_operator_backend_test_connection_rejects_invalid_name_without_audit(self):
        """D24 step 5 (FEAT-11): an invalid profile name in the Add form is refused before the tester runs and writes no backend.test audit row."""
        context, page, errors = self.operator_page('/admin/operator?tab=backend')
        before = self.state()
        try:
            self.post_hook('/_test/backend-tester', {'mode': 'ok'})
            calls_before = len(self.context.request.get(self.url + '/_test/backend-tester-calls').json())
            page.get_by_role('button', name='Add profile').click()
            dialog = self.fill_profile_editor(page, 'Bad Name!', 'm', 1000, base='https://bad.example/v1')
            dialog.get_by_role('button', name='Test connection').click()
            page.wait_for_timeout(800)
            after = self.state()
            self.assertEqual([r for r in after['audit'][len(before['audit']):] if r['action'] == 'backend.test'], [])
            self.assertEqual(len(self.context.request.get(self.url + '/_test/backend-tester-calls').json()), calls_before)
            self.assertEqual(dialog.get_by_role('status').count(), 0)
            dialog.get_by_role('button', name='Cancel').click()
        finally:
            context.close()

    def test_operator_backend_route_stays_404_for_non_operators(self):
        """D24 step 4 (FEAT-11): the Backend tab adds no new door; /operator?tab=backend is still the unknown-page 404 for a plain server admin. The operator-only 404 itself is pinned by test_operator_bot_settings_page_and_menu."""
        context = self.browser.new_context(ignore_https_errors=True)
        context.add_cookies([{'name': 'llmcord_session', 'value': 'browser-plain-session', 'url': self.url, 'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
        page = context.new_page()
        try:
            response = page.goto(self.url + '/admin/operator?tab=backend')
            self.assertEqual(response.status, 404)
            self.assertNotIn('Add profile', page.content())
        finally:
            context.close()

    def test_monitoring_bot_wide_spending_line_for_operators_only(self):
        """FEAT-13: an operator sees the bot-wide spending line and an Open Bot settings link on a server's Monitoring tab; a plain server admin gets neither in the page."""
        import re
        from playwright.sync_api import expect
        context, page, errors = self.operator_page('/admin/guild/1?tab=monitoring')
        try:
            line = page.get_by_text(re.compile(r'Bot-wide spending: \$\S+ (of \$\S+ hard cap this period|this period; no hard cap set)'))
            expect(line).to_be_visible()
            page.get_by_role('link', name='Open Bot settings').click()
            page.wait_for_url('**/admin/operator')
            self.assertFalse(errors, errors)
        finally:
            context.close()
        context = self.browser.new_context(ignore_https_errors=True, viewport={'width': 1000, 'height': 900})
        context.add_cookies([{'name': 'llmcord_session', 'value': 'browser-plain-session', 'url': self.url,
                              'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
        page = context.new_page()
        try:
            page.goto(self.url + '/admin/guild/1?tab=monitoring')
            expect(page.get_by_text('By feature', exact=True)).to_be_visible()
            content = page.content()
            self.assertNotIn('Bot-wide spending', content)
            self.assertNotIn('Open Bot settings', content)
            self.assertNotIn('hard cap this period', content)
            self.assertNotIn('no hard cap set', content)
            self.assertNotIn('$5.02', content)  # the all-servers total
        finally:
            context.close()

    def test_monitoring_spending_line_goes_when_the_operator_is_removed(self):
        """FEAT-13: the operator check is repeated when Monitoring is rebuilt, so a viewer removed from the operator list loses the bot-wide line without reloading."""
        import re
        from playwright.sync_api import expect
        before = self.state()['turn_log']
        self.set_turn_log(False, 14)
        context, page, errors = self.operator_page('/admin/guild/1?tab=monitoring')
        try:
            expect(page.get_by_text(re.compile(r'Bot-wide spending: '))).to_be_visible()
            self.post_hook('/_test/operators', {'ids': []})
            page.get_by_role('tab', name='Server setup', exact=True).click()
            page.get_by_label('Keep usage and log for', exact=True).click()
            page.get_by_role('option', name='7 days', exact=True).click()
            page.get_by_role('tab', name='Monitoring', exact=True).click()
            page.get_by_role('region', name='Unsaved changes').get_by_role('button', name='Save changes', exact=True).click()
            expect(page.get_by_text('Usage is kept for 7 days (Server settings).', exact=True)).to_be_visible()
            expect(page.get_by_text('Bot-wide spending')).to_have_count(0)
            expect(page.get_by_text('Open Bot settings')).to_have_count(0)
        finally:
            self.post_hook('/_test/operators', {'ids': [6]})
            self.set_turn_log(before['enabled'], before['days'])
            context.close()

    def test_account_menu_profile_lines_avatar_and_position(self):
        """UI-37: global_name and @username lines, the avatar image src, and the menu opening below the header at both widths."""
        from playwright.sync_api import expect
        for width, height in ((1280, 900), (390, 800)):
            context = self.browser.new_context(ignore_https_errors=True, viewport={'width': width, 'height': height})
            context.add_cookies([{'name': 'llmcord_session', 'value': 'browser-profile-session', 'url': self.url,
                                  'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
            page = context.new_page()
            page.route('https://cdn.discordapp.com/**', lambda route: route.abort())  # fake data: never fetch the CDN
            errors = []
            self.watch(page, errors)
            try:
                page.goto(self.url + '/admin/')
                button = page.get_by_role('button', name='Account menu')
                expect(button).to_be_visible()
                src = button.locator('img.ll-avatar').get_attribute('src')
                self.assertEqual(src, 'https://cdn.discordapp.com/avatars/5/' + 'a' * 32 + '.png?size=64')
                self.assertEqual(button.locator('.ll-avatar-tile').count(), 0)
                button.click()
                menu = page.locator('.ll-menu')
                expect(menu.get_by_text('Moon Display', exact=True)).to_be_visible()
                expect(menu.get_by_text('@moonuser', exact=True)).to_be_visible()
                page.wait_for_timeout(300)
                header = page.locator('header').bounding_box()
                box = menu.bounding_box()
                self.assertGreaterEqual(box['y'], header['y'] + header['height'], (width, header, box))
                self.assertGreaterEqual(box['x'], 0, box)
                self.assertLessEqual(box['x'] + box['width'], width, box)
                self.assertLessEqual(box['y'] + box['height'], height, box)
                self.assertFalse(errors, errors)
            finally:
                context.close()

    def panel_reads(self, context):
        counters = context.request.get(self.url + '/_test/counters').json()
        return {name: counters.get('store:' + name, 0) for name in ('usage_report', 'list_presets')}

    def test_lazy_panels_build_on_first_selection(self):
        """PERF-01: a page load builds only the selected tab's panel; others are built once, on first selection.

        ``usage_report`` is read only by Monitoring and ``list_presets`` only by Prompt presets.
        """
        context, page, errors = self.ux_page('/admin/')
        try:
            self.load(page, self.url + '/admin/guild/1?tab=characters', before_each=lambda: self.assertEqual(
                context.request.post(self.url + '/_test/counters/reset').status, 200))
            page.get_by_role('button', name='Create character', exact=True).wait_for()
            page.wait_for_timeout(300)
            self.assertEqual(self.panel_reads(context), {'usage_report': 0, 'list_presets': 0})
            page.get_by_role('tab', name='Prompt presets', exact=True).click()
            page.get_by_label('Sample channel (optional)', exact=True).wait_for()
            page.wait_for_timeout(300)
            built = self.panel_reads(context)
            self.assertEqual(built['usage_report'], 0, built)
            self.assertGreaterEqual(built['list_presets'], 1, built)
            # Leaving and returning does not rebuild the panel.
            page.get_by_role('tab', name='Characters', exact=True).click()
            page.get_by_role('tab', name='Prompt presets', exact=True).click()
            page.get_by_label('Sample channel (optional)', exact=True).wait_for()
            page.wait_for_timeout(300)
            self.assertEqual(self.panel_reads(context), built)
            page.get_by_role('tab', name='Monitoring', exact=True).click()
            page.get_by_text('By model', exact=True).wait_for()
            page.wait_for_timeout(300)
            self.assertEqual(self.panel_reads(context), {**built, 'usage_report': 1})
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def create_space_in_place(self, page, name):
        page.evaluate('window.__marker = 1')
        _worlds_section(page).get_by_label('Name', exact=True).fill(name)
        _worlds_section(page).get_by_role('button', name='Create', exact=True).click()
        page.get_by_text(f'{name} · world', exact=True).wait_for(timeout=5000)

    def test_mutation_updates_page_without_browser_reload(self):
        """PERF-01: Create (Worlds and hubs) refreshes the page in place: no navigation, no new Discord channel fetch."""
        context, page, errors = self.ux_page('/admin/guild/1')
        try:
            _worlds_section(page).get_by_label('Name', exact=True).wait_for()
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
            self.assertEqual(_section(page, 'Named lorebooks').get_by_label('Kind', exact=True).count(), 0)
            page.get_by_role('button', name='Import lorebook', exact=True).click()
            _section(page, 'Named lorebooks').get_by_label('Kind', exact=True).wait_for(timeout=5000)
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
            expansion = page.locator('.q-expansion-item').filter(has=page.get_by_text('Test book · Channel lorebook · #scene', exact=True)).first
            expansion.get_by_text('Test book · Channel lorebook · #scene', exact=True).click()
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

    def audit_mark(self):
        """Highest audit id so far; pass it to audit_rows so a test sees only rows it caused (the fixture DB is shared)."""
        return max((r['id'] for r in self.state()['audit']), default=0)

    def audit_rows(self, action, after=0):
        return [r for r in self.state()['audit'] if r['action'] == action and r['id'] > after]

    def alice_slot(self, page, label):
        """Alice's card with the emotion `label` expanded (the characters tab rebuilds after writes, so this re-opens only what is closed)."""
        card = page.locator('.character-card').filter(has=page.get_by_text('Alice', exact=True)).first
        def attempt():
            header = card.locator('.q-item').filter(has=page.get_by_text(label, exact=True)).first
            if not header.is_visible():
                if not card.get_by_label('Description', exact=True).is_visible():
                    card.get_by_text('Alice', exact=True).first.click()
                header.wait_for(state='visible', timeout=3000)
            if header.get_attribute('aria-expanded', timeout=3000) != 'true':
                header.click(timeout=5000)
            return card.locator('.q-expansion-item').filter(has=page.get_by_text(label, exact=True)).last
        for _ in range(5):
            try:
                emotion = attempt()
                if emotion.locator('input[type=file]').count():
                    return card, emotion
            except Exception:
                pass
            page.wait_for_timeout(300)
        return card, attempt()

    def ui09_cleanup(self):
        """Best-effort reset of what the UI-09 tests change (extra emotions, 'Audit hub', #scene binding, Alice's fallback avatar)."""
        try:
            response = self.context.request.post(self.url + '/_test/cleanup-ui09')
            if response.status != 200:
                print(f'warning: /_test/cleanup-ui09 returned {response.status}', file=sys.stderr)
        except Exception as exc:
            print(f'warning: /_test/cleanup-ui09 failed: {exc!r}', file=sys.stderr)

    def delete_lore(self, *contents):
        """Best-effort removal of lore entries a test created (by exact content)."""
        try:
            response = self.context.request.post(self.url + '/_test/delete-lore', data={'contents': list(contents)})
            if response.status != 200:
                print(f'warning: /_test/delete-lore returned {response.status}', file=sys.stderr)
        except Exception as exc:
            print(f'warning: /_test/delete-lore failed: {exc!r}', file=sys.stderr)

    def png(self, color):
        from io import BytesIO
        from PIL import Image
        buffer = BytesIO()
        Image.new('RGB', (50, 50), color).save(buffer, 'PNG')
        return {'name': f'{color}.png', 'mimeType': 'image/png', 'buffer': buffer.getvalue()}

    def test_emotion_removal_dialogs_cancel_confirm_and_audit(self):
        """UI-09: 'Remove emotion image' and 'Remove emotion' each ask first; Cancel keeps the data, the confirm button removes it and audits avatar.image.delete / avatar.slot.delete."""
        context, page, errors, card, _ = self.open_alice()
        mark = self.audit_mark()
        try:
            alice = self.alice_row()
            def slot():
                return next((s for s in self.state()['slots'] if s['character_id'] == alice['id'] and s['slot_key'] == 'testy'), None)
            card.get_by_label('New emotion key', exact=True).fill('testy')
            card.get_by_label('Emotion label', exact=True).fill('Testy')
            card.get_by_role('button', name='Add emotion', exact=True).click()
            self.wait_for(lambda: slot() is not None)
            card, emotion = self.alice_slot(page, 'Testy')
            emotion.locator('input[type=file]').set_input_files(self.png('red'))
            page.get_by_text('Emotion image saved', exact=True).wait_for(timeout=5000)
            self.wait_for(lambda: slot()['has_image'])
            dialog = page.get_by_role('dialog')
            # Remove emotion image: Cancel keeps the image.
            card, emotion = self.alice_slot(page, 'Testy')
            self.open_dialog(emotion.get_by_role('button', name='Remove emotion image', exact=True), dialog)
            dialog.get_by_text('Remove the image from Testy?', exact=True).wait_for()
            dialog.get_by_role('button', name='Cancel', exact=True).click()
            dialog.wait_for(state='hidden')
            self.assertTrue(slot()['has_image'])
            self.assertEqual(self.audit_rows('avatar.image.delete', mark), [])
            # Confirm removes only the image and audits it.
            self.open_dialog(emotion.get_by_role('button', name='Remove emotion image', exact=True), dialog)
            dialog.get_by_role('button', name='Remove emotion image', exact=True).click()
            self.wait_for(lambda: slot() and not slot()['has_image'])
            self.assertEqual([json.loads(r['detail_json']) for r in self.audit_rows('avatar.image.delete', mark)], [{'character': alice['id'], 'slot': 'testy'}])
            # Remove emotion: Cancel keeps the slot.
            card, emotion = self.alice_slot(page, 'Testy')
            self.assertEqual(emotion.get_by_role('button', name='Remove emotion image', exact=True).count(), 0)
            self.open_dialog(emotion.get_by_role('button', name='Remove emotion', exact=True), dialog)
            dialog.get_by_text('Remove emotion Testy?', exact=True).wait_for()
            dialog.get_by_role('button', name='Cancel', exact=True).click()
            dialog.wait_for(state='hidden')
            self.assertIsNotNone(slot())
            self.assertEqual(self.audit_rows('avatar.slot.delete', mark), [])
            # Confirm deletes the slot and audits it.
            self.open_dialog(emotion.get_by_role('button', name='Remove emotion', exact=True), dialog)
            dialog.get_by_role('button', name='Remove emotion', exact=True).click()
            self.wait_for(lambda: slot() is None)
            self.assertEqual(len(self.audit_rows('avatar.slot.delete', mark)), 1)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()
            self.ui09_cleanup()

    def test_avatar_image_urls_carry_a_version_that_changes_with_the_image(self):
        """UI-09: the emotion and fallback avatar <img> sources end in ?v=<12 hex> and the value changes when the image changes."""
        context, page, errors, card, _ = self.open_alice()
        try:
            alice = self.alice_row()
            card.get_by_label('New emotion key', exact=True).fill('vtest')
            card.get_by_label('Emotion label', exact=True).fill('Vtest')
            card.get_by_role('button', name='Add emotion', exact=True).click()
            self.wait_for(lambda: any(s['slot_key'] == 'vtest' and s['character_id'] == alice['id'] for s in self.state()['slots']))
            def emotion_src(expected_count):
                card, emotion = self.alice_slot(page, 'Vtest')
                image = emotion.locator('img')
                self.wait_for(lambda: image.count() == expected_count)
                return image.first.get_attribute('src')
            card, emotion = self.alice_slot(page, 'Vtest')
            self.assertEqual(emotion.locator('img').count(), 0)
            emotion.locator('input[type=file]').set_input_files(self.png('red'))
            page.get_by_text('Emotion image saved', exact=True).wait_for(timeout=5000)
            first = emotion_src(1)
            self.assertRegex(first, rf'/characters/{alice["id"]}/avatars/vtest\?v=[0-9a-f]{{12}}$')
            card, emotion = self.alice_slot(page, 'Vtest')
            emotion.locator('input[type=file]').set_input_files(self.png('blue'))
            self.wait_for(lambda: emotion_src(1) != first)
            second = emotion_src(1)
            self.assertRegex(second, r'\?v=[0-9a-f]{12}$')
            self.assertNotEqual(first.split('?v=')[1], second.split('?v=')[1])
            # The fallback avatar uses the same scheme.
            def fallback_src():
                card = page.locator('.character-card').filter(has=page.get_by_text('Alice', exact=True)).first
                fallback = card.locator('.fallback-avatar')
                for _ in range(10):
                    try:
                        if not card.get_by_label('Description', exact=True).is_visible():
                            card.get_by_text('Alice', exact=True).first.click(timeout=3000)
                        if not fallback.get_by_text('Used when the selected emotion', exact=False).is_visible():
                            card.get_by_text('Fallback static avatar', exact=True).click(timeout=3000)
                        fallback.get_by_text('Used when the selected emotion', exact=False).wait_for(state='visible', timeout=3000)
                        break
                    except Exception:
                        page.wait_for_timeout(300)
                return card, fallback
            card, fallback = fallback_src()
            fallback.locator('input[type=file]').set_input_files(self.png('green'))
            page.get_by_text('Fallback avatar saved', exact=True).wait_for(timeout=5000)
            card, fallback = fallback_src()
            fallback.locator('img').first.wait_for(timeout=5000)
            one = fallback.locator('img').first.get_attribute('src')
            self.assertRegex(one, rf'/characters/{alice["id"]}/avatar\?v=[0-9a-f]{{12}}$')
            fallback.locator('input[type=file]').set_input_files(self.png('yellow'))
            self.wait_for(lambda: fallback_src()[1].locator('img').first.get_attribute('src') != one)
            two = fallback_src()[1].locator('img').first.get_attribute('src')
            self.assertRegex(two, rf'/characters/{alice["id"]}/avatar\?v=[0-9a-f]{{12}}$')
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()
            self.ui09_cleanup()

    def test_hub_link_unlink_and_bind_channel_write_audit_rows(self):
        """UI-09: Link, Unlink and Bind channel each write one audit row ('hub.link', 'hub.unlink', 'channel.bind') carrying the ids acted on."""
        context, page, errors = self.ux_page('/admin/guild/1')
        mark = self.audit_mark()
        try:
            world = next(r for r in self.state()['spaces'] if r['name'] == 'World')
            annex = next(r for r in self.state()['spaces'] if r['name'] == 'Annex')
            worlds = _worlds_section(page)
            worlds.get_by_label('Name', exact=True).fill('Audit hub')
            worlds.get_by_label('Kind', exact=True).click()
            page.get_by_role('option', name='hub', exact=True).click()
            worlds.get_by_role('button', name='Create', exact=True).click()
            self.wait_for(lambda: any(r['name'] == 'Audit hub' for r in self.state()['spaces']))
            hub = next(r for r in self.state()['spaces'] if r['name'] == 'Audit hub')
            links = _section(page, 'Hub links')
            def choose(section, label, option):
                section.get_by_label(label, exact=True).click()
                page.get_by_role('option', name=option, exact=True).click()
            choose(links, 'Hub', 'Audit hub')
            choose(links, 'World', 'World')
            links.get_by_role('button', name='Link', exact=True).click()
            self.wait_for(lambda: len(self.audit_rows('hub.link', mark)) == 1)
            self.assertEqual(json.loads(self.audit_rows('hub.link', mark)[0]['detail_json']), {'hub': hub['id'], 'world': world['id'], 'enabled': True, 'pruned': 0})
            links = _section(page, 'Hub links')
            choose(links, 'Hub', 'Audit hub')
            choose(links, 'World', 'World')
            links.get_by_role('button', name='Unlink', exact=True).click()
            self.wait_for(lambda: len(self.audit_rows('hub.unlink', mark)) == 1)
            self.assertEqual(json.loads(self.audit_rows('hub.unlink', mark)[0]['detail_json']), {'hub': hub['id'], 'world': world['id'], 'enabled': False, 'pruned': 0})
            self.assertEqual(len(self.audit_rows('hub.link', mark)), 1)
            # Bind #scene (100) to Annex and back to World: one row each.
            def bind(option):
                channels = _section(page, 'Channels and casts')
                choose(channels, 'Discord text channel', '#scene')
                choose(channels, 'World or hub', option)
                channels.get_by_role('button', name='Bind channel', exact=True).click()
            before = len(self.audit_rows('channel.bind', mark))
            bind('Annex (world)')
            self.wait_for(lambda: len(self.audit_rows('channel.bind', mark)) == before + 1)
            bind('World (world)')
            self.wait_for(lambda: len(self.audit_rows('channel.bind', mark)) == before + 2)
            details = [json.loads(r['detail_json']) for r in self.audit_rows('channel.bind', mark)[before:]]
            self.assertEqual(details, [{'channel': 100, 'space': annex['id'], 'dropped': []}, {'channel': 100, 'space': world['id'], 'dropped': []}])
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()
            self.ui09_cleanup()

    def test_delete_preset_asks_for_confirmation(self):
        """UX-05: 'Delete preset' opens a dialog; Cancel keeps the preset and the red confirm button deletes it."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=prompts')
        try:
            page.get_by_label('Preset name', exact=True).fill('Doomed preset')
            self.open_more_item(page, page, 'preset', 'Save as new preset')
            self.wait_for(lambda: any(r['name'] == 'Doomed preset' for r in self.state()['presets']))
            page.get_by_role('region', name='Unsaved changes').wait_for(state='hidden')
            dialog = page.get_by_role('dialog')
            self.open_more_item(page, page, 'preset', 'Delete preset')
            dialog.wait_for(timeout=5000)
            dialog.get_by_text('Delete preset Doomed preset?', exact=True).wait_for(timeout=5000)
            dialog.get_by_role('button', name='Cancel', exact=True).click()
            dialog.wait_for(state='hidden')
            self.assertTrue(any(r['name'] == 'Doomed preset' for r in self.state()['presets']))
            self.open_more_item(page, page, 'preset', 'Delete preset')
            dialog.wait_for(timeout=5000)
            dialog.get_by_role('button', name='Delete preset', exact=True).click()
            self.wait_for(lambda: not any(r['name'] == 'Doomed preset' for r in self.state()['presets']))
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_preset_save_bar_tracks_unsaved_edits(self):
        """UI-23: the preset save bar appears only for real edits, Preview works while dirty, a dirty library switch is refused, Reset restores."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=prompts')
        try:
            bar = page.get_by_role('region', name='Unsaved changes')
            field = page.get_by_label('Prompt text / template', exact=True).first
            page.locator('.prompt-toggle').first.click()
            field.wait_for(timeout=5000)
            page.get_by_label('Preset name', exact=True).fill('Bar preset')
            self.open_more_item(page, page, 'preset', 'Save as new preset')
            self.wait_for(lambda: any(r['name'] == 'Bar preset' for r in self.state()['presets']))
            bar.wait_for(state='hidden')
            before_presets = self.state()['presets']
            original = field.input_value()
            self.assertEqual(bar.count(), 0)
            purpose = page.get_by_label('Purpose', exact=True)
            purpose.click()
            page.get_by_role('option').nth(1).click()
            self.wait_for(lambda: field.input_value() != original)
            self.assertEqual(bar.count(), 0)
            purpose.click()
            page.get_by_role('option', name='Dialogue', exact=True).click()
            self.wait_for(lambda: field.input_value() == original)
            self.assertEqual(bar.count(), 0)
            field.fill(original + ' edited')
            field.press('Tab')
            bar.wait_for(timeout=5000)
            bar.get_by_text('Bar preset has unsaved changes.', exact=True).wait_for()
            bar.get_by_role('button', name='Save draft', exact=True).wait_for()
            page.get_by_role('button', name='Preview without a model call', exact=True).click()
            page.get_by_text('Estimated input tokens:', exact=False).wait_for(timeout=5000)
            library = page.get_by_label('Preset library', exact=True)
            shown = library.input_value()
            library.click()
            page.get_by_role('option', name='Built-in default', exact=True).click()
            page.get_by_text('Save or reset your changes to', exact=False).wait_for(timeout=5000)
            self.assertEqual(library.input_value(), shown)
            bar.get_by_role('button', name='Reset', exact=True).click()
            bar.wait_for(state='hidden')
            self.assertEqual(page.get_by_label('Prompt text / template', exact=True).first.input_value(), original)
            self.assertEqual(self.state()['presets'], before_presets)
            dialog = page.get_by_role('dialog')
            self.open_more_item(page, page, 'preset', 'Delete preset')
            dialog.wait_for(timeout=5000)
            dialog.get_by_role('button', name='Delete preset', exact=True).click()
            self.wait_for(lambda: not any(r['name'] == 'Bar preset' for r in self.state()['presets']))
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_ui31_preset_prompt_text_highlights_known_and_unknown_macros(self):
        """UI-31: the prompt text mirror marks known and unknown {{macros}} live; collapsed blocks are not attached; no CSP violation."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=prompts')
        try:
            console = []
            page.on('console', lambda m: console.append(m.text) if m.type == 'error' else None)
            page.locator('.prompt-toggle').first.wait_for()
            self.assertEqual(page.locator('.mh-mirror').count(), 0)
            page.locator('.prompt-toggle').first.click()
            field = page.get_by_label('Prompt text / template', exact=True).first
            field.wait_for(timeout=5000)
            original = field.input_value()
            plain = page.get_by_label('Block name', exact=True).first.evaluate('el => getComputedStyle(el).color')  # an unfocused field's normal text colour
            field.fill('Hi {{user}} {{bogus}}')
            mirror = page.locator('.mh-mirror')
            mirror.wait_for(timeout=5000)
            self.assertEqual(mirror.count(), 1)
            self.assertEqual(mirror.get_attribute('aria-hidden'), 'true')
            self.assertTrue(field.evaluate('el => el === document.activeElement'))
            self.assertEqual(mirror.evaluate('el => getComputedStyle(el).color'), plain)
            self.assertEqual(field.evaluate('el => getComputedStyle(el).caretColor'), plain)
            self.assertEqual(mirror.locator('.mh-known').all_inner_texts(), ['{{user}}'])
            self.assertEqual(mirror.locator('.mh-unknown').all_inner_texts(), ['{{bogus}}'])
            field.press_sequentially(' {{char}}')
            page.wait_for_function("() => document.querySelectorAll('.mh-mirror .mh-known').length === 2")
            field.press_sequentially(' <b>&')
            self.assertEqual(field.input_value(), 'Hi {{user}} {{bogus}} {{char}} <b>&')
            self.assertEqual(mirror.text_content(), field.input_value())
            self.assertEqual(mirror.locator('b').count(), 0)
            bar = page.get_by_role('region', name='Unsaved changes')
            bar.wait_for(timeout=5000)
            bar.get_by_role('button', name='Reset', exact=True).click()
            bar.wait_for(state='hidden')
            page.wait_for_function('text => document.querySelector(".ll-macro textarea").value === text && document.querySelector(".mh-mirror")?.textContent === text', arg=original)
            self.assertEqual(page.locator('.mh-mirror').count(), 1)
            field.fill('\n'.join(f'line {i} {{{{user}}}}' for i in range(40)))
            page.wait_for_function('() => document.querySelector(".mh-mirror")?.querySelectorAll(".mh-known").length === 40')
            field.evaluate('el => { el.scrollTop = 120; }')
            page.wait_for_function('() => { const t = document.querySelector(".ll-macro textarea"); return t.scrollTop > 0 && document.querySelector(".mh-mirror").scrollTop === t.scrollTop; }')
            self.assertEqual([m for m in console if 'Content Security Policy' in m], [])
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_ui31_lore_content_highlights_known_and_unknown_macros(self):
        """UI-31 step 2: the lore Content field is mirrored with known and unknown {{macros}}; closed editors have no mirror."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=lore')
        try:
            page.get_by_role('button', name='New entry', exact=True).first.wait_for(timeout=5000)
            self.assertEqual(page.locator('.mh-mirror').count(), 0)
            page.get_by_role('button', name='New entry', exact=True).first.click()
            editor = page.locator('.q-card').filter(has=page.get_by_text('Create lore', exact=True)).last
            field = editor.get_by_label('Content', exact=True)
            field.fill('{{char}} meets {{user}} {{bogus}}')
            mirror = page.locator('.mh-mirror')
            mirror.wait_for(timeout=5000)
            page.wait_for_function("() => document.querySelectorAll('.mh-mirror .mh-known').length === 2")
            self.assertEqual(mirror.count(), 1)
            self.assertEqual(mirror.locator('.mh-known').all_inner_texts(), ['{{char}}', '{{user}}'])
            self.assertEqual(mirror.locator('.mh-unknown').all_inner_texts(), ['{{bogus}}'])
            self.assertEqual(mirror.text_content(), field.input_value())
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_ui31_both_panels_on_one_page_attach_each_lore_textarea_once(self):
        """UI-31 step 2: after visiting Prompt presets then Lore (two MacroHighlight instances), a lore editor has exactly one mirror, also when reopened."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=prompts')
        try:
            console = []
            page.on('console', lambda m: console.append(m.text) if m.type == 'error' else None)
            page.locator('.prompt-toggle').first.wait_for()
            page.get_by_role('tab', name='Lore', exact=True).click()
            page.get_by_role('button', name='New entry', exact=True).first.click()
            editor = page.locator('.q-card').filter(has=page.get_by_text('Create lore', exact=True)).last
            editor.get_by_label('Content', exact=True).fill('Dual {{char}} {{user}} {{bogus}}')
            page.wait_for_function("() => document.querySelectorAll('.mh-mirror .mh-known').length === 2")
            self.assertEqual(page.locator('.mh-mirror').count(), 1)
            self.assertEqual(page.locator('.mh-mirror .mh-unknown').all_inner_texts(), ['{{bogus}}'])
            page.get_by_role('button', name='Save lore', exact=True).click()
            page.get_by_text('Create lore', exact=True).wait_for(state='detached', timeout=5000)
            row = page.locator('.lore-entry').filter(has_text='Dual')
            row.get_by_role('button', name='Edit entry', exact=True).click()
            page.get_by_text('Edit lore', exact=True).wait_for(timeout=5000)
            page.get_by_label('Content', exact=True).scroll_into_view_if_needed()  # attach is lazy: an editor below the fold waits until it is visible
            page.wait_for_function("() => document.querySelectorAll('.mh-mirror .mh-known').length === 2")
            self.assertEqual(page.locator('.mh-mirror').count(), 1)
            self.assertEqual([m for m in console if 'Content Security Policy' in m], [])
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_ui31_character_description_highlights_known_and_unknown_macros(self):
        """UI-31 step 3: a character card field is mirrored with known and unknown {{macros}}; collapsed character cards have no mirror."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=characters')
        try:
            console = []
            page.on('console', lambda m: console.append(m.text) if m.type == 'error' else None)
            card = page.locator('.character-card').filter(has=page.get_by_text('Alice', exact=True)).first
            card.get_by_text('Alice', exact=True).first.wait_for(timeout=5000)
            self.assertEqual(page.locator('.mh-mirror').count(), 0)
            card.get_by_text('Alice', exact=True).first.click()
            description = card.get_by_label('Description', exact=True)
            description.wait_for()
            description.fill('{{char}} greets {{user}} {{bogus}}')
            mirror = description.locator('xpath=preceding-sibling::div[contains(@class, "mh-mirror")]')
            page.wait_for_function("() => document.querySelectorAll('.character-card .mh-mirror .mh-known').length === 2")
            self.assertEqual(mirror.locator('.mh-known').all_inner_texts(), ['{{char}}', '{{user}}'])
            self.assertEqual(mirror.locator('.mh-unknown').all_inner_texts(), ['{{bogus}}'])
            self.assertEqual(mirror.text_content(), description.input_value())
            self.assertEqual([m for m in console if 'Content Security Policy' in m], [])
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def preset_file(self, marker):
        from llmcord_core.prompts import default_bundle, export_preset
        bundle = default_bundle()
        bundle['purposes']['dialogue'][0]['content'] = marker
        return {'name': 'preset.json', 'mimeType': 'application/json', 'buffer': json.dumps(export_preset(bundle)).encode()}

    def test_preset_import_is_an_unsaved_preview_that_reset_undoes(self):
        """UI-41: an import shows 'Imported (not saved)' in the library, Activate/Delete are refused while it is dirty, export while dirty
        exports the unsaved preview (characterization), Reset restores the previous preset, and a second import works without a reload."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=prompts')
        try:
            bar = page.get_by_role('region', name='Unsaved changes')
            library = page.get_by_label('Preset library', exact=True)
            name = page.get_by_label('Preset name', exact=True)
            uploader = page.locator('input[type=file]').first
            before_library, before_name, before_state = library.input_value(), name.input_value(), self.state()
            uploader.set_input_files(self.preset_file('First imported marker'))
            bar.wait_for(timeout=5000)
            self.wait_for(lambda: library.input_value() == 'Imported (not saved)')
            self.assertEqual(name.input_value(), 'Imported preset')
            bar.get_by_text('Imported preset has unsaved changes.', exact=True).wait_for()

            refusal = page.get_by_text('Save or reset your changes to Imported preset first.', exact=True)
            badge = page.locator('.q-notification__badge')
            self.open_more_item(page, page, 'preset', 'Activate saved revision')
            self.wait_for(lambda: refusal.count() >= 1)
            self.open_more_item(page, page, 'preset', 'Delete preset')
            self.wait_for(lambda: badge.count() and badge.first.inner_text() == '2')  # Quasar groups identical toasts under a counter
            self.assertEqual(page.get_by_role('dialog').count(), 0)
            library.click()
            page.get_by_role('option', name='Built-in default', exact=True).click()
            self.wait_for(lambda: badge.count() and badge.first.inner_text() == '3')
            self.wait_for(lambda: library.input_value() == 'Imported (not saved)')
            self.assertEqual(self.state()['active'], before_state['active'])
            self.assertEqual(self.state()['presets'], before_state['presets'])

            with page.expect_download() as download:
                page.get_by_role('button', name='Export full native bundle', exact=True).click()
            self.assertIn('First imported marker', Path(download.value.path()).read_text(encoding='utf-8'))

            bar.get_by_role('button', name='Reset', exact=True).click()
            bar.wait_for(state='hidden')
            self.wait_for(lambda: library.input_value() == before_library)
            self.assertEqual(name.input_value(), before_name)
            with page.expect_download() as download:
                page.get_by_role('button', name='Export full native bundle', exact=True).click()
            self.assertNotIn('First imported marker', Path(download.value.path()).read_text(encoding='utf-8'))

            page.locator('input[type=file]').first.set_input_files(self.preset_file('Second imported marker'))
            bar.wait_for(timeout=5000)
            self.wait_for(lambda: library.input_value() == 'Imported (not saved)')
            with page.expect_download() as download:
                page.get_by_role('button', name='Export full native bundle', exact=True).click()
            self.assertIn('Second imported marker', Path(download.value.path()).read_text(encoding='utf-8'))
            bar.get_by_role('button', name='Reset', exact=True).click()
            bar.wait_for(state='hidden')
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def check_audit_failure(self, kind):
        """MNT-30 / MNT-38: a save whose audit write fails afterwards ends consistently. A busy database is absorbed (saved, bar settled, warning logged);
        any other error is toasted, the bar stays dirty with Save enabled, and nothing unhandled reaches server.log."""
        active_before = self.state()['active']
        context, page, errors = self.ux_page('/admin/guild/1?tab=prompts')
        try:
            bar = page.get_by_role('region', name='Unsaved changes')
            save = bar.get_by_role('button', name='Save draft', exact=True)
            page.locator('.ll-subtitle').filter(has_text='Active preset:').first.wait_for(timeout=30000)  # the tab is built
            page.get_by_label('Preset name', exact=True).fill(f'MNT30 audit failure {kind}')
            bar.wait_for(timeout=30000)
            self.post_hook('/_test/fail-next-audit', n=1, kind=kind)
            save.click()
            if kind == 'busy':
                bar.wait_for(state='hidden', timeout=30000)
                self.wait_for(lambda: any(r['name'] == 'MNT30 audit failure busy' for r in self.state()['presets']), self.SLOW_SERVER_POLLS)
            else:
                page.get_by_text('Something went wrong. Try again.', exact=True).wait_for(timeout=10000)
                self.assertTrue(bar.is_visible())
                self.wait_for(lambda: save.is_enabled(), self.SLOW_SERVER_POLLS)
            log = (Path(self.temp.name) / 'server.log').read_text(encoding='utf-8', errors='replace')
            self.assertNotIn('ERROR:nicegui', log)
            self.assertNotIn('Task exception was never retrieved', log)
            self.assertIn('Admin action failed' if kind == 'bug' else 'Audit write failed after a committed admin action', log)
            if kind == 'busy':
                self.assertNotIn('Admin action hit a database error', log)
                self.assertEqual(page.get_by_text('The database is busy. Try again.').count(), 0)
        finally:
            self.post_hook('/_test/fail-next-audit', n=0)
            context.close()
            self.context.request.post(self.url + '/_test/cleanup-presets', params={'prefix': 'MNT30 ', 'active': active_before})

    def test_save_with_a_busy_audit_write_still_reports_success(self):
        """MNT-38: sqlite3.OperationalError from the audit write after the commit is logged, not toasted; the row is saved and the bar settles."""
        self.check_audit_failure('busy')

    def test_save_with_an_unexpected_error_ends_with_a_toast_and_a_dirty_bar(self):
        """MNT-30: any other unexpected error is toasted generically, logged, and never escapes the handler; the bar stays dirty with Save enabled."""
        self.check_audit_failure('bug')

    def test_preset_save_as_new_proposes_a_copy_name_and_save_draft_refreshes_subtitle(self):
        """UI-41: Save as new preset with an unchanged name saves '<name> copy' (then 'copy 2'); Save draft on the active preset refreshes the
        'Active preset' subtitle; saving an import selects the new preset in the library."""
        active_before = self.state()['active']
        context, page, errors = self.ux_page('/admin/guild/1?tab=prompts')
        try:
            bar = page.get_by_role('region', name='Unsaved changes')
            name = page.get_by_label('Preset name', exact=True)
            library = page.get_by_label('Preset library', exact=True)
            names = lambda: {r['name'] for r in self.state()['presets']}
            name.fill('UI41 copy source')
            self.open_more_item(page, page, 'preset', 'Save as new preset')
            self.wait_for(lambda: 'UI41 copy source' in names(), self.SLOW_SERVER_POLLS)
            bar.wait_for(state='hidden', timeout=30000)
            self.open_more_item(page, page, 'preset', 'Save as new preset')
            self.wait_for(lambda: 'UI41 copy source copy' in names(), self.SLOW_SERVER_POLLS)
            # The row is saved before the page has re-rendered: wait for the field, not just the database.
            self.wait_for(lambda: name.input_value() == 'UI41 copy source copy', self.SLOW_SERVER_POLLS)
            bar.wait_for(state='hidden', timeout=30000)
            name.fill('UI41 copy source')
            self.open_more_item(page, page, 'preset', 'Save as new preset')
            self.wait_for(lambda: 'UI41 copy source copy 2' in names(), self.SLOW_SERVER_POLLS)
            self.wait_for(lambda: name.input_value() == 'UI41 copy source copy 2', self.SLOW_SERVER_POLLS)
            self.wait_for(lambda: 'UI41 copy source copy 2' in library.input_value(), self.SLOW_SERVER_POLLS)
            bar.wait_for(state='hidden', timeout=30000)

            self.open_more_item(page, page, 'preset', 'Activate saved revision')
            subtitle = page.locator('.ll-subtitle').filter(has_text='Active preset:').first
            self.wait_for(lambda: bool(self.state()['active']), self.SLOW_SERVER_POLLS)
            self.wait_for(lambda: 'UI41 copy source copy 2 · draft 1' in subtitle.inner_text(), self.SLOW_SERVER_POLLS)
            name.fill('UI41 renamed active')
            bar.get_by_role('button', name='Save draft', exact=True).click()
            self.wait_for(lambda: 'UI41 renamed active' in names(), self.SLOW_SERVER_POLLS)
            self.wait_for(lambda: 'UI41 renamed active · draft 2' in subtitle.inner_text(), self.SLOW_SERVER_POLLS)

            page.locator('input[type=file]').first.set_input_files(self.preset_file('Saved import marker'))
            self.wait_for(lambda: library.input_value() == 'Imported (not saved)', self.SLOW_SERVER_POLLS)
            self.wait_for(lambda: name.input_value() == 'Imported preset', self.SLOW_SERVER_POLLS)
            bar.wait_for(timeout=30000)
            name.fill('UI41 saved import')
            bar.get_by_role('button', name='Save draft', exact=True).click()
            self.wait_for(lambda: 'UI41 saved import' in names(), self.SLOW_SERVER_POLLS)
            bar.wait_for(state='hidden', timeout=30000)
            self.wait_for(lambda: library.input_value() == 'UI41 saved import · draft 1', self.SLOW_SERVER_POLLS)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()
            try:
                response = self.context.request.post(self.url + '/_test/cleanup-presets', params={'prefix': 'UI41 ', 'active': active_before})
                if response.status != 200:
                    print(f'warning: /_test/cleanup-presets returned {response.status}', file=sys.stderr)
            except Exception as exc:
                print(f'warning: /_test/cleanup-presets failed: {exc!r}', file=sys.stderr)

    def test_prompt_blocks_are_collapsed_rows_with_live_summary(self):
        """UI-23: prompt blocks start collapsed, toggle by keyboard, keep their summary live, stay open across re-renders and remove via the menu."""
        import re
        from playwright.sync_api import expect
        context, page, errors = self.ux_page('/admin/guild/1?tab=prompts')
        try:
            cards = page.locator('.prompt-block')
            cards.first.wait_for(timeout=5000)
            first = cards.first
            toggle = first.locator('.prompt-toggle')
            field = first.get_by_label('Prompt text / template', exact=True)
            bar = page.get_by_role('region', name='Unsaved changes')
            self.assertEqual(toggle.get_attribute('aria-expanded'), 'false')
            self.assertFalse(field.is_visible())
            toggle.focus()
            page.keyboard.press('Enter')
            field.wait_for(timeout=5000)
            expect(toggle).to_have_attribute('aria-expanded', 'true')
            expect(first.get_by_role('button', name=re.compile(r'System\s+\u00b7\s+Relative'))).to_have_class(re.compile('prompt-toggle'))
            name = first.get_by_label('Block name', exact=True)
            name.fill('Renamed block')
            expect(toggle).to_contain_text('Renamed block')
            expect(first.get_by_role('button', name='More actions for Renamed block', exact=True)).to_be_visible()
            pill = toggle.locator('.ll-pill-off')
            enabled = first.get_by_role('switch', name='Enabled', exact=True)
            expect(enabled).to_be_checked()
            expect(pill).to_be_hidden()
            enabled.click()
            expect(pill).to_have_text('Disabled')
            enabled.click()
            expect(pill).to_be_hidden()
            # a re-render (Purpose away and back) keeps the open block open
            purpose = page.get_by_label('Purpose', exact=True)
            purpose.click()
            page.get_by_role('option', name='Director', exact=True).click()
            director_toggle = page.locator('.prompt-block').first.locator('.prompt-toggle')
            expect(director_toggle).not_to_contain_text('Renamed block')
            # open state is per purpose: Director's block with the same id stays collapsed
            expect(director_toggle).to_have_attribute('aria-expanded', 'false')
            purpose.click()
            page.get_by_role('option', name='Dialogue', exact=True).click()
            expect(page.locator('.prompt-toggle').first).to_contain_text('Renamed block')
            toggle = page.locator('.prompt-toggle').first
            expect(toggle).to_have_attribute('aria-expanded', 'true')
            self.assertTrue(page.locator('.prompt-block').first.get_by_label('Prompt text / template', exact=True).is_visible())
            # Add prompt block appends a new, expanded block
            count = cards.count()
            page.get_by_role('button', name='Add prompt block', exact=True).click()
            expect(cards).to_have_count(count + 1)
            added = cards.last
            self.assertEqual(added.locator('.prompt-toggle').get_attribute('aria-expanded'), 'true')
            self.assertTrue(added.get_by_label('Prompt text / template', exact=True).is_visible())
            # Remove block via the menu drops that block and raises the save bar
            added_name = added.get_by_label('Block name', exact=True).input_value() or 'Untitled block'
            self.open_more_item(page, added, added_name, 'Remove block')
            expect(cards).to_have_count(count)
            bar.wait_for(timeout=5000)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def block_names(self, page):
        return [t.strip() for t in page.locator('.prompt-block .ll-block-name').all_inner_texts()]

    def test_prompt_block_menu_lists_move_and_remove_in_order(self):
        """UI-36: each block's menu holds Move up, Move down, Remove block; Move up is disabled on the first block and Move down on the last."""
        from playwright.sync_api import expect
        context, page, errors = self.ux_page('/admin/guild/1?tab=prompts')
        try:
            cards = page.locator('.prompt-block')
            cards.first.wait_for(timeout=5000)
            self.assertGreaterEqual(cards.count(), 2)
            names = self.block_names(page)
            items = page.get_by_role('menuitem')

            def open_menu(index):
                button = cards.nth(index).get_by_role('button', name=f'More actions for {names[index]}', exact=True)
                button.click()
                items.first.wait_for(timeout=5000)

            open_menu(0)
            self.assertEqual([t.strip() for t in items.all_inner_texts()], ['Move up', 'Move down', 'Remove block'])
            expect(page.get_by_role('menuitem', name='Move up', exact=True)).to_be_disabled()
            expect(page.get_by_role('menuitem', name='Move down', exact=True)).to_be_enabled()
            expect(page.get_by_role('menuitem', name='Remove block', exact=True)).to_be_enabled()
            page.keyboard.press('Escape')
            items.first.wait_for(state='hidden')
            open_menu(len(names) - 1)
            expect(page.get_by_role('menuitem', name='Move down', exact=True)).to_be_disabled()
            expect(page.get_by_role('menuitem', name='Move up', exact=True)).to_be_enabled()
            page.keyboard.press('Escape')
            self.assertFalse(page.get_by_role('region', name='Unsaved changes').count())
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_prompt_block_move_down_reorders_and_reset_restores(self):
        """UI-36: Move down swaps the first two blocks, raises the save bar, and Reset restores the order without touching the stored preset."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=prompts')
        try:
            cards = page.locator('.prompt-block')
            cards.first.wait_for(timeout=5000)
            bar = page.get_by_role('region', name='Unsaved changes')
            before_presets = self.state()['presets']
            original = self.block_names(page)
            self.assertGreaterEqual(len(original), 2)
            self.open_more_item(page, cards.first, original[0], 'Move down')
            expected = [original[1], original[0]] + original[2:]
            self.wait_for(lambda: self.block_names(page) == expected)
            bar.wait_for(timeout=5000)
            bar.get_by_text('has unsaved changes.', exact=False).wait_for()
            self.assertEqual(self.state()['presets'], before_presets)
            bar.get_by_role('button', name='Reset', exact=True).click()
            bar.wait_for(state='hidden')
            self.wait_for(lambda: self.block_names(page) == original)
            self.assertEqual(self.state()['presets'], before_presets)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_prompt_block_rows_fit_phone_width(self):
        """UI-36: at 390 px a block's summary sits on a second line under its name and the drag handle has a >= 40 px hit area."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=prompts')
        try:
            page.set_viewport_size({'width': 390, 'height': 844})
            row = page.locator('.prompt-block').first.locator('.prompt-toggle')
            row.wait_for(timeout=5000)
            name = row.locator('.ll-block-name').bounding_box()
            meta = row.locator('.ll-block-meta').bounding_box()
            self.assertGreaterEqual(meta['y'], name['y'] + name['height'] - 1, (name, meta))
            handle = page.locator('.prompt-block').first.locator('.prompt-handle').bounding_box()
            # measured 40 x 40 (1.25em icon + 11.25 px padding each side = 40 px)
            self.assertGreaterEqual(handle['width'], 39.5, handle)
            self.assertGreaterEqual(handle['height'], 39.5, handle)
            self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'))
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_prompt_selects_show_labels_and_export_leave_out_works(self):
        """UI-24: block Placement shows a readable label (stored value unchanged) and Export's "Leave out blocks" select drops the chosen block from the SillyTavern JSON."""
        from playwright.sync_api import expect
        context, page, errors = self.ux_page('/admin/guild/1?tab=prompts')
        try:
            cards = page.locator('.prompt-block')
            cards.first.wait_for(timeout=5000)
            first = cards.first
            first.locator('.prompt-toggle').click()
            expect(first.get_by_label('Placement', exact=True)).to_have_value('Relative to the prompt')
            expect(first.get_by_label('Role', exact=True)).to_have_value('System')

            def export():
                with page.expect_download() as download:
                    page.get_by_role('button', name='Export SillyTavern dialogue preset', exact=True).click()
                return json.loads(Path(download.value.path()).read_text(encoding='utf-8'))

            nonportable = ['location', 'opening', 'card_instructions', 'lore_before_examples', 'lore_after_examples', 'lore_in_chat',
                           'personal', 'encounters', 'summary', 'preceding', 'recent', 'card_post_history']
            select = page.get_by_label('Leave out blocks', exact=True)

            def leave_out(*ids):
                select.click()
                for ident in ids:
                    page.get_by_role('option').filter(has_text=f'({ident})').click()
                page.keyboard.press('Escape')

            leave_out(*nonportable)
            full = export()
            names = [p['name'] for p in full['prompts']]
            self.assertIn('Character Voice', names)
            leave_out('character_voice')
            left_out = export()
            self.assertNotIn('Character Voice', [p['name'] for p in left_out['prompts']])
            self.assertEqual(len(left_out['prompts']), len(names) - 1)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_disabled_lore_entry_shows_the_disabled_pill(self):
        """UI-40: a disabled lore entry's row carries the 'Disabled' pill (ll-pill ll-pill-off); an enabled entry's row does not."""
        from playwright.sync_api import expect
        context, page, errors = self.ux_page('/admin/guild/1?tab=lore')
        try:
            page.get_by_role('button', name='New entry', exact=True).first.wait_for(timeout=5000)
            for content in ('Pill enabled entry', 'Pill disabled entry'):
                page.get_by_role('button', name='New entry', exact=True).first.click()
                page.get_by_label('Content', exact=True).fill(content)
                page.get_by_role('button', name='Save lore', exact=True).click()
                self.wait_for(lambda: any(r['content'] == content for r in self.state()['lore']))
            enabled_row = page.locator('.lore-drop-left .lore-entry').filter(has_text='Pill enabled entry')
            disabled_row = page.locator('.lore-drop-left .lore-entry').filter(has_text='Pill disabled entry')
            expect(disabled_row).to_be_visible()
            expect(disabled_row.locator('.ll-pill-off')).to_have_count(0)
            self.assertEqual(self.context.request.post(self.url + '/_test/set-lore-enabled', data={'content': 'Pill disabled entry', 'enabled': False}).status, 200)
            try:
                self.load(page, self.url + '/admin/guild/1?tab=lore')
                expect(disabled_row.locator('.ll-pill.ll-pill-off')).to_have_text('Disabled')
                expect(enabled_row).to_be_visible()
                expect(enabled_row.locator('.ll-pill-off')).to_have_count(0)
                expect(enabled_row.get_by_text('Disabled', exact=True)).to_have_count(0)
            finally:
                self.context.request.post(self.url + '/_test/set-lore-enabled', data={'content': 'Pill disabled entry', 'enabled': True})
            self.assertFalse(errors, errors)
        finally:
            context.close()
            self.delete_lore('Pill enabled entry', 'Pill disabled entry')

    def test_lore_move_item_is_disabled_when_both_panels_show_the_same_owner(self):
        """UI-40: with the same owner in both panels, the entry menu's Move item is disabled; with different owners it is enabled."""
        import re
        from playwright.sync_api import expect
        context, page, errors = self.ux_page('/admin/guild/1?tab=lore')
        original_page = self.page
        self.page = page
        try:
            page.get_by_role('button', name='New entry', exact=True).first.wait_for(timeout=5000)
            page.get_by_role('button', name='New entry', exact=True).first.click()
            page.get_by_label('Content', exact=True).fill('Same owner entry')
            page.get_by_role('button', name='Save lore', exact=True).click()
            self.wait_for(lambda: any(r['content'] == 'Same owner entry' for r in self.state()['lore']))
            entry = page.locator('.lore-drop-left .lore-entry').filter(has_text='Same owner entry')
            entry.wait_for(timeout=5000)
            menuitem = page.get_by_role('menuitem', name='Move right', exact=True)

            def open_menu():
                entry.get_by_role('button', name='More actions for entry', exact=True).click()
                menuitem.wait_for(timeout=5000)

            open_menu()
            expect(menuitem).not_to_have_class(re.compile('disabled'))
            page.keyboard.press('Escape')
            expect(menuitem).to_have_count(0)
            # Point the right panel at whatever owner the left one shows (its kind and id are on the drop zone).
            left_zone = page.locator('.lore-drop-left')
            kind, ident = left_zone.get_attribute('data-owner-kind'), left_zone.get_attribute('data-owner-id')
            self.choose_lore_owner('right', page.get_by_label('Left owner', exact=True).input_value(), kind, int(ident))
            entry = page.locator('.lore-drop-left .lore-entry').filter(has_text='Same owner entry')
            entry.wait_for(timeout=5000)
            open_menu()
            expect(menuitem).to_have_class(re.compile('disabled'))
            before = self.state()
            menuitem.click(force=True)
            self.assertEqual(self.state()['lore'], before['lore'])
            self.assertFalse(errors, errors)
        finally:
            self.page = original_page
            context.close()
            self.delete_lore('Same owner entry')

    def test_ui46_lore_and_preset_selects_show_labels_not_codes(self):
        """UI-46: lore Message role/Placement selects show labels and saving 'After examples' stores after_examples; the preset block form hides
        'Stable ID' (Block ID lives in the preserved-fields expansion); the preview's expansion titles are role labels and Omitted/Adaptations have no ':N' codes."""
        import re
        from playwright.sync_api import expect
        context, page, errors = self.ux_page('/admin/guild/1?tab=lore')
        try:
            page.get_by_role('button', name='New entry', exact=True).first.wait_for(timeout=5000)
            page.get_by_role('button', name='New entry', exact=True).first.click()
            page.get_by_label('Content', exact=True).fill('UI46 placement entry')
            page.get_by_role('button', name='Save lore', exact=True).click()
            self.wait_for(lambda: any(r['content'] == 'UI46 placement entry' for r in self.state()['lore']))
            row = page.locator('.lore-drop-left .lore-entry').filter(has_text='UI46 placement entry')
            row.get_by_role('button', name='Edit entry', exact=True).click()
            page.get_by_text('Advanced activation and placement rules', exact=True).click()
            role = page.get_by_label('Message role', exact=True)
            placement = page.get_by_label('Placement', exact=True)
            role.wait_for(timeout=5000)
            self.assertIn(role.input_value(), ('System', 'User', 'Assistant'))
            self.assertIn(placement.input_value(), ('Before character', 'After character', 'Before examples', 'After examples', 'In chat at depth'))
            placement.click()
            page.get_by_role('option', name='After examples', exact=True).click()
            expect(placement).to_have_value('After examples')
            page.get_by_role('button', name='Save lore', exact=True).click()
            self.wait_for(lambda: any(r['content'] == 'UI46 placement entry' and json.loads(r['rule_json']).get('position') == 'after_examples'
                                      for r in self.state()['lore']))
            self.assertFalse(errors, errors)
        finally:
            context.close()
            self.delete_lore('UI46 placement entry')
        context, page, errors = self.ux_page('/admin/guild/1?tab=prompts')
        try:
            toggle = page.locator('.prompt-toggle').first
            toggle.click()
            page.get_by_label('Prompt text / template', exact=True).first.wait_for(timeout=5000)
            card = page.locator('.ll-block-open').first
            self.assertEqual(card.get_by_text('Stable ID', exact=False).count(), 0)
            block_id = card.get_by_text(re.compile(r'^Block ID: \S+'))
            expect(block_id).to_be_hidden()
            card.get_by_text('Preserved import fields and compatibility remapping', exact=True).click()
            expect(block_id).to_be_visible()
            self.assertIn('Block ID: ', card.inner_text())
            page.get_by_role('button', name='Preview without a model call', exact=True).click()
            page.get_by_text('Estimated input tokens:', exact=False).wait_for(timeout=5000)
            omitted = page.get_by_text(re.compile(r'^Omitted: '))
            adaptations = page.get_by_text(re.compile(r'^Adaptations: '))
            omitted.wait_for(timeout=5000)
            self.assertEqual(omitted.inner_text().strip(), 'Omitted: none')
            self.assertEqual(adaptations.inner_text().strip(), 'Adaptations: none')
            self.assertNotRegex(omitted.inner_text(), r':\d')
            self.assertNotRegex(adaptations.inner_text(), r'(top_system|:\d)')
            titles = page.locator('.q-expansion-item__container .q-item__label').all_inner_texts()
            self.assertTrue({'System', 'User', 'Assistant'} & {t.strip() for t in titles}, titles)
            self.assertFalse(errors, errors)
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
            row = page.locator('.lore-entry').filter(has_text='Doomed lore entry')
            edit = row.get_by_role('button', name='Edit entry', exact=True)
            self.assertEqual(edit.inner_text().strip(), '')
            self.assertEqual(edit.get_attribute('aria-label'), 'Edit entry')
            # Right after Save lore the list re-renders while the editor closes, so
            # a single early measurement can catch a transient taller layout
            # (MNT-36). Wait for two equal readings, then assert the settled height.
            self.assertLess(self.settled_height(row), 70)
            self.assertEqual(row.get_by_text('Disabled', exact=True).count(), 0)
            edit.click()
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

    def test_cancelled_confirmation_dialogs_leave_the_dom(self):
        """UI-05: a one-shot confirmation dialog is removed from the page after Cancel or Escape, without a refresh."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=lore')
        try:
            page.get_by_role('button', name='New entry', exact=True).first.wait_for(timeout=5000)
            page.get_by_role('button', name='New entry', exact=True).first.click()
            page.get_by_label('Content', exact=True).fill('Dialog cleanup entry')
            page.get_by_role('button', name='Save lore', exact=True).click()
            self.wait_for(lambda: any(r['content'] == 'Dialog cleanup entry' for r in self.state()['lore']))
            page.locator('.lore-entry').filter(has_text='Dialog cleanup entry').get_by_role('button', name='Edit entry', exact=True).click()
            editor = page.locator('.q-card').filter(has=page.get_by_text('Edit lore', exact=True)).last
            dialogs = "() => Object.values(document.querySelector('#app').__vue_app__._container._vnode.component.proxy.elements).filter(e => e.tag === 'nicegui-dialog').length"
            self.assertEqual(page.evaluate(dialogs), 0)
            for close in ('cancel', 'escape'):
                editor.get_by_role('button', name='Delete', exact=True).click()
                dialog = page.get_by_role('dialog')
                dialog.get_by_role('button', name='Cancel', exact=True).wait_for(timeout=5000)
                if close == 'cancel':
                    dialog.get_by_role('button', name='Cancel', exact=True).click()
                else:
                    page.keyboard.press('Escape')
                page.wait_for_function("() => document.querySelectorAll('.q-dialog').length === 0", timeout=5000)
                page.wait_for_timeout(500)
                self.assertEqual(page.evaluate(dialogs), 0, close)
            self.assertTrue(any(r['content'] == 'Dialog cleanup entry' for r in self.state()['lore']))
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_lore_editor_keys_are_chips(self):
        """UI-21: Keys / Secondary keys are chip inputs; a key containing a comma stays ONE key, old keys survive an edit."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=lore')
        try:
            def add_chip(label, text):
                field = page.get_by_label(label, exact=True)
                field.fill(text)
                field.press('Enter')
            page.get_by_role('button', name='New entry', exact=True).first.wait_for(timeout=5000)
            page.get_by_role('button', name='New entry', exact=True).first.click()
            page.get_by_label('Content', exact=True).fill('Chip lore entry')
            add_chip('Keys', 'old key')
            page.get_by_role('button', name='Save lore', exact=True).click()
            self.wait_for(lambda: any(r['content'] == 'Chip lore entry' for r in self.state()['lore']))
            row = page.locator('.lore-entry').filter(has_text='Chip lore entry')
            row.get_by_role('button', name='Edit entry', exact=True).click()
            editor = page.locator('.q-card').filter(has=page.get_by_text('Edit lore', exact=True)).last
            editor.locator('.q-chip').filter(has_text='old key').wait_for(timeout=5000)
            add_chip('Keys', '/a,b/')
            add_chip('Secondary keys', 'second')
            page.get_by_role('button', name='Save lore', exact=True).click()

            def entry():
                return next((r for r in self.state()['lore'] if r['content'] == 'Chip lore entry'), None)
            self.wait_for(lambda: entry() and '/a,b/' in json.loads(entry()['keys_json']))
            self.assertEqual(json.loads(entry()['keys_json']), ['old key', '/a,b/'])
            page.get_by_text('Edit lore', exact=True).wait_for(state='detached', timeout=5000)
            row = page.locator('.lore-entry').filter(has_text='Chip lore entry')
            row.get_by_role('button', name='Edit entry', exact=True).click()
            editor = page.locator('.q-card').filter(has=page.get_by_text('Edit lore', exact=True)).last
            for chip in ('old key', '/a,b/', 'second'):
                editor.locator('.q-chip').filter(has_text=chip).wait_for(timeout=5000)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_lore_editor_commits_pending_key_text_on_save(self):
        """UI-21: key text typed without Enter is committed as a chip when Save lore is clicked directly (no duplicates)."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=lore')
        try:
            page.get_by_role('button', name='New entry', exact=True).first.wait_for(timeout=5000)
            page.get_by_role('button', name='New entry', exact=True).first.click()
            page.get_by_label('Content', exact=True).fill('Pending chip entry')
            page.get_by_label('Keys', exact=True).fill('pending')
            page.get_by_label('Secondary keys', exact=True).fill('/x,y/')
            page.get_by_role('button', name='Save lore', exact=True).click()
            self.wait_for(lambda: any(r['content'] == 'Pending chip entry' for r in self.state()['lore']))
            row = next(r for r in self.state()['lore'] if r['content'] == 'Pending chip entry')
            self.assertEqual(json.loads(row['keys_json']), ['pending'])
            self.assertEqual(json.loads(row['rule_json'])['secondary_keys'], ['/x,y/'])
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_lore_editor_tab_away_then_save_keeps_key_text(self):
        """UI-45: key text left with Tab (blur) and then Save is stored once, trimmed."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=lore')
        try:
            page.get_by_role('button', name='New entry', exact=True).first.wait_for(timeout=5000)
            page.get_by_role('button', name='New entry', exact=True).first.click()
            page.get_by_label('Content', exact=True).fill('Tab away entry')
            page.get_by_label('Keys', exact=True).fill('  tabbed  ')
            page.get_by_label('Keys', exact=True).press('Tab')
            page.get_by_role('button', name='Save lore', exact=True).click()
            self.wait_for(lambda: any(r['content'] == 'Tab away entry' for r in self.state()['lore']))
            row = next(r for r in self.state()['lore'] if r['content'] == 'Tab away entry')
            self.assertEqual(json.loads(row['keys_json']), ['tabbed'])
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    def test_lore_editor_failed_save_after_flush_keeps_chips(self):
        """UI-45: a save that fails after pending key text was flushed keeps the chip and clears the typed text."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=lore')
        try:
            page.get_by_role('button', name='New entry', exact=True).first.wait_for(timeout=5000)
            page.get_by_role('button', name='New entry', exact=True).first.click()
            page.get_by_label('Content', exact=True).fill('Failed flush entry')
            page.get_by_text('Advanced activation and placement rules', exact=True).click()
            page.get_by_label('Scan depth (JSON, null for default)', exact=True).fill('{bad')
            page.get_by_label('Keys', exact=True).fill('kept')
            page.get_by_role('button', name='Save lore', exact=True).click()
            editor = page.locator('.q-card').filter(has=page.get_by_text('Create lore', exact=True)).last
            editor.locator('.q-chip').filter(has_text='kept').wait_for(timeout=5000)
            page.wait_for_timeout(500)
            self.assertEqual(page.get_by_label('Keys', exact=True).input_value(), '')
            self.assertFalse(any(r['content'] == 'Failed flush entry' for r in self.state()['lore']))
            self.assertEqual(editor.locator('.q-chip').filter(has_text='kept').count(), 1)
        finally:
            context.close()

    def test_lore_editor_checkboxes_stay_inside_card_on_phones(self):
        """UI-45/UI-50: at 390 px each of Enabled, Always active and Pinned lies inside its card with its label on one line; the row wraps whole checkboxes (Pinned drops below at 390 px, all share a line at 1000 px)."""
        context, page, errors = self.ux_page('/admin/guild/1?tab=lore')
        try:
            page.set_viewport_size({'width': 390, 'height': 844})
            page.get_by_role('button', name='New entry', exact=True).first.wait_for(timeout=5000)
            page.get_by_role('button', name='New entry', exact=True).first.click()
            card = page.locator('.q-card').filter(has=page.get_by_text('Create lore', exact=True)).last
            card.wait_for(timeout=5000)
            card_right = card.bounding_box()['x'] + card.bounding_box()['width']
            heights = {}
            for name in ('Enabled', 'Always active', 'Pinned'):
                box = page.get_by_role('checkbox', name=name, exact=True).bounding_box()
                self.assertLessEqual(box['x'] + box['width'], card_right + 0.5, (name, box, card_right))
                rects = page.evaluate("""(name) => {
                    const label = [...document.querySelectorAll('.q-checkbox__label')].filter(e => e.textContent.trim() === name).pop();
                    return {rects: label.getClientRects().length, height: label.getBoundingClientRect().height};
                }""", name)
                self.assertEqual(rects['rects'], 1, (name, rects))
                heights[name] = rects['height']
            for name, height in heights.items():
                self.assertAlmostEqual(height, heights['Enabled'], delta=2, msg=(name, heights))
            tops = self.checkbox_tops(page)
            self.assertGreater(tops['Pinned'], tops['Enabled'] + 2, tops)  # Pinned wrapped to a second line
            self.assertAlmostEqual(tops['Always active'], tops['Enabled'], delta=2, msg=tops)
            self.assertFalse(page.evaluate('() => document.documentElement.scrollWidth > innerWidth'))
            page.set_viewport_size({'width': 1000, 'height': 844})
            page.wait_for_function(
                '''() => { const t = ['Enabled', 'Always active', 'Pinned'].map(n => [...document.querySelectorAll('.q-checkbox')]
                    .filter(e => e.textContent.trim() === n).pop().getBoundingClientRect().top);
                    return Math.max(...t) - Math.min(...t) <= 2; }''', timeout=5000)
            tops = self.checkbox_tops(page)
            self.assertLessEqual(max(tops.values()) - min(tops.values()), 2, tops)
            self.assertFalse(errors, (errors, getattr(page, 'network', [])))
        finally:
            context.close()

    @staticmethod
    def checkbox_tops(page):
        return {name: page.get_by_role('checkbox', name=name, exact=True).bounding_box()['y']
                for name in ('Enabled', 'Always active', 'Pinned')}

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
            anchors = {'Server setup': _worlds_section(page).get_by_label('Name', exact=True),
                       'Characters': page.get_by_role('button', name='Create character', exact=True),
                       'Lore': page.get_by_role('button', name='Import lorebook', exact=True),
                       'Imports': _section(page, 'Named lorebooks').get_by_label('Kind', exact=True),
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

    # MNT-21: the editor save bar (D21), first user: the Character editor.
    def open_alice(self, path='/admin/guild/1?tab=characters'):
        context, page, errors = self.ux_page(path)
        page.get_by_role('tab', name='Characters', exact=True).click()
        card = page.locator('.character-card').filter(has=page.get_by_text('Alice', exact=True)).first
        card.get_by_text('Alice', exact=True).first.click()
        description = card.get_by_label('Description', exact=True)
        description.wait_for()
        return context, page, errors, card, description

    def alice_row(self):
        return next(row for row in self.state()['characters'] if row['name'] == 'Alice')

    def restore_alice(self, original):
        self.context.request.post(self.url + '/_test/change-character', data={'name': 'Alice', 'description': original})

    def test_savebar_hidden_until_character_field_differs(self):
        """MNT-21: no bar on load; editing Description shows 'Alice has unsaved changes.'; typing the old value back hides it."""
        context, page, errors, card, description = self.open_alice()
        try:
            bar = page.get_by_role('region', name='Unsaved changes')
            self.assertFalse(bar.is_visible())
            original = description.input_value()
            description.fill(original + ' edited')
            bar.wait_for(state='visible', timeout=5000)
            self.assertEqual(bar.get_by_text('Alice has unsaved changes.', exact=True).count(), 1)
            description.fill(original)
            bar.wait_for(state='hidden', timeout=5000)
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_savebar_reset_restores_fields_without_store_change(self):
        """MNT-21: Reset puts the field back, hides the bar, and makes no store call or audit row."""
        context, page, errors, card, description = self.open_alice()
        try:
            bar = page.get_by_role('region', name='Unsaved changes')
            original = description.input_value()
            before = self.state()
            description.fill(original + ' discarded')
            bar.wait_for(state='visible', timeout=5000)
            page.wait_for_function('() => window.onbeforeunload !== null', timeout=5000)
            bar.get_by_role('button', name='Reset', exact=True).click()
            bar.wait_for(state='hidden', timeout=5000)
            page.wait_for_function('() => window.onbeforeunload === null', timeout=5000)
            self.assertEqual(description.input_value(), original)
            after = self.state()
            self.assertEqual(after['characters'], before['characters'])
            self.assertEqual(after['audit'], before['audit'])
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_savebar_save_changes_stores_edit_and_audits(self):
        """MNT-21: Save changes stores the edit, hides the bar, and writes a character.edit audit row."""
        context, page, errors, card, description = self.open_alice()
        original = description.input_value()
        try:
            bar = page.get_by_role('region', name='Unsaved changes')
            audit_before = len([r for r in self.state()['audit'] if r['action'] == 'character.edit'])
            description.fill('A courier edited through the save bar')
            bar.wait_for(state='visible', timeout=5000)
            bar.get_by_role('button', name='Save changes', exact=True).click()
            bar.wait_for(state='hidden', timeout=5000)
            self.wait_for(lambda: json.loads(self.alice_row()['card'])['description'] == 'A courier edited through the save bar')
            rows = [r for r in self.state()['audit'] if r['action'] == 'character.edit']
            self.assertEqual(len(rows), audit_before + 1)
            self.assertEqual(json.loads(rows[-1]['detail_json']), {'id': self.alice_row()['id']})
            self.assertFalse(errors, errors)
        finally:
            context.close()
            self.restore_alice(original)

    def test_savebar_conflict_keeps_bar_and_edit(self):
        """MNT-21: a stale save shows the conflict toast; the bar stays and the edit stays in the field."""
        context, page, errors, card, description = self.open_alice()
        original = description.input_value()
        try:
            bar = page.get_by_role('region', name='Unsaved changes')
            description.fill('My edit that will conflict')
            bar.wait_for(state='visible', timeout=5000)
            response = self.context.request.post(self.url + '/_test/change-character', data={'name': 'Alice', 'description': 'Another admin was here'})
            self.assertTrue(response.ok)
            bar.get_by_role('button', name='Save changes', exact=True).click()
            page.get_by_text('Character changed; reload before saving', exact=True).wait_for(timeout=5000)
            self.assertTrue(bar.is_visible())
            self.assertEqual(description.input_value(), 'My edit that will conflict')
            self.assertEqual(json.loads(self.alice_row()['card'])['description'], 'Another admin was here')
            self.assertFalse(errors, errors)
        finally:
            context.close()
            self.restore_alice(original)

    def test_savebar_conflict_turns_reset_into_reload(self):
        """MNT-22: after a conflict the bar says the character was changed elsewhere and Reset becomes Reload; Reload shows the other admin's value and the next edit saves."""
        context, page, errors, card, description = self.open_alice()
        original = description.input_value()
        try:
            bar = page.get_by_role('region', name='Unsaved changes')
            description.fill('My edit that will conflict')
            bar.wait_for(state='visible', timeout=5000)
            self.assertTrue(self.context.request.post(self.url + '/_test/change-character', data={'name': 'Alice', 'description': 'Another admin was here'}).ok)
            bar.get_by_role('button', name='Save changes', exact=True).click()
            page.get_by_text('Character changed; reload before saving', exact=True).wait_for(timeout=5000)
            bar.get_by_text('Alice was changed somewhere else.', exact=True).wait_for(timeout=5000)
            self.assertEqual(bar.get_by_role('button', name='Reset', exact=True).count(), 0)
            bar.get_by_role('button', name='Reload', exact=True).click()
            bar.wait_for(state='hidden', timeout=5000)
            card = page.locator('.character-card').filter(has=page.get_by_text('Alice', exact=True)).first
            card.get_by_text('Alice', exact=True).first.click()
            description = card.get_by_label('Description', exact=True)
            description.wait_for()
            self.wait_for(lambda: description.input_value() == 'Another admin was here')
            description.fill('Edited after reload')
            bar.wait_for(state='visible', timeout=5000)
            bar.get_by_role('button', name='Save changes', exact=True).click()
            bar.wait_for(state='hidden', timeout=5000)
            self.wait_for(lambda: json.loads(self.alice_row()['card'])['description'] == 'Edited after reload')
            self.assertFalse(errors, errors)
        finally:
            context.close()
            self.restore_alice(original)

    def test_savebar_refuses_upload_and_confirm_dialog_while_dirty(self):
        """MNT-22: an avatar upload (ctx.upload) and a confirm dialog action (Delete permanently) are refused while Alice is dirty; nothing is stored."""
        context, page, errors, card, description = self.open_alice()
        try:
            bar = page.get_by_role('region', name='Unsaved changes')
            description.fill(description.input_value() + ' dirty')
            bar.wait_for(state='visible', timeout=5000)
            before = self.state()
            fallback = card.locator('.q-uploader').filter(has_text='Upload fallback avatar').first
            fallback.locator('input[type=file]').set_input_files(self.png('green'))
            page.get_by_text('Save or reset your changes to Alice first.', exact=True).first.wait_for(timeout=5000)
            self.open_more_item(page, card, 'Alice', 'Delete character')
            page.get_by_role('button', name='Delete permanently', exact=True).click()
            page.get_by_text('Save or reset your changes to Alice first.', exact=True).first.wait_for(timeout=5000)
            page.wait_for_timeout(500)
            after = self.state()
            self.assertEqual(after['characters'], before['characters'])
            self.assertEqual(after['audit'], before['audit'])
            self.assertTrue(bar.is_visible())
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_savebar_disables_other_editors_while_dirty(self):
        """MNT-22: while Alice is dirty the Server settings switches (another tracked editor) are disabled; Reset enables them again."""
        context, page, errors, card, description = self.open_alice()
        try:
            bar = page.get_by_role('region', name='Unsaved changes')
            description.fill(description.input_value() + ' dirty')
            bar.wait_for(state='visible', timeout=5000)
            page.get_by_role('tab', name='Server setup', exact=True).click()
            footer = page.get_by_role('switch', name='Show model and cost footer on replies')
            footer.wait_for(timeout=5000)
            self.assertTrue(footer.is_disabled())
            bar.get_by_role('button', name='Reset', exact=True).click()
            bar.wait_for(state='hidden', timeout=5000)
            self.wait_for(lambda: footer.is_enabled())
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_savebar_blocks_other_writes_while_dirty(self):
        """MNT-21: while Alice is dirty another write is refused (toast, alert class, no store call); edits survive a tab switch."""
        context, page, errors, card, description = self.open_alice()
        try:
            bar = page.get_by_role('region', name='Unsaved changes')
            description.fill(description.input_value() + ' dirty')
            bar.wait_for(state='visible', timeout=5000)
            before = self.state()
            self.open_more_item(page, card, 'Alice', 'Archive')
            page.get_by_text('Save or reset your changes to Alice first.', exact=True).wait_for(timeout=5000)
            page.wait_for_function("() => document.querySelector('.ll-savebar')?.classList.contains('ll-savebar-alert')", timeout=2000)
            after = self.state()
            self.assertEqual(after['characters'], before['characters'])
            self.assertFalse(self.alice_row()['archived'])
            self.assertEqual(after['audit'], before['audit'])
            dirty_value = description.input_value()
            page.get_by_role('tab', name='Server setup', exact=True).click()
            page.wait_for_function("() => document.querySelector('[role=tab][aria-selected=true]')?.textContent.includes('Server setup')", timeout=5000)
            self.assertTrue(bar.is_visible())
            page.get_by_role('tab', name='Characters', exact=True).click()
            page.wait_for_function("() => document.querySelector('[role=tab][aria-selected=true]')?.textContent.includes('Characters')", timeout=5000)
            self.assertTrue(bar.is_visible())
            self.assertEqual(description.input_value(), dirty_value)
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_savebar_blocks_lore_move_while_dirty(self):
        """MNT-21: a lore Move (LiveContext.run with an action) is refused while Alice is dirty; lore rows are unchanged."""
        context, page, errors, card, description = self.open_alice('/admin/guild/1?tab=lore')
        try:
            page.get_by_role('tab', name='Lore', exact=True).click()
            page.get_by_role('button', name='New entry', exact=True).first.wait_for(timeout=5000)
            page.get_by_role('button', name='New entry', exact=True).first.click()
            page.get_by_label('Content', exact=True).fill('Savebar guarded lore')
            page.get_by_role('button', name='Save lore', exact=True).click()
            self.wait_for(lambda: any(r['content'] == 'Savebar guarded lore' for r in self.state()['lore']))
            page.get_by_role('tab', name='Characters', exact=True).click()
            bar = page.get_by_role('region', name='Unsaved changes')
            description.fill(description.input_value() + ' dirty')
            bar.wait_for(state='visible', timeout=5000)
            page.get_by_role('tab', name='Lore', exact=True).click()
            entry = page.locator('.lore-drop-left .lore-entry').filter(has_text='Savebar guarded lore')
            entry.wait_for(timeout=5000)
            before = self.state()
            self.open_more_item(page, entry, 'entry', 'Move right')
            page.get_by_text('Save or reset your changes to Alice first.', exact=True).wait_for(timeout=5000)
            after = self.state()
            self.assertEqual(after['lore'], before['lore'])
            self.assertEqual(after['books'], before['books'])
            bar.get_by_role('button', name='Reset', exact=True).click()
            bar.wait_for(state='hidden', timeout=5000)
            self.open_more_item(page, entry, 'entry', 'Delete')
            page.get_by_role('button', name='Delete 1 entry', exact=True).click()
            self.wait_for(lambda: not any(r['content'] == 'Savebar guarded lore' for r in self.state()['lore']))
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_savebar_refuses_refresh_navigation_while_dirty(self):
        """MNT-21: ctx.refresh() (Imports 'Edit entries in Lore') is refused while Alice is dirty; the edit stays and the tab does not change."""
        context, page, errors, card, description = self.open_alice()
        try:
            bar = page.get_by_role('region', name='Unsaved changes')
            description.fill(description.input_value() + ' dirty')
            bar.wait_for(state='visible', timeout=5000)
            dirty_value = description.input_value()
            page.get_by_role('tab', name='Imports', exact=True).click()
            expansion = page.locator('.q-expansion-item').filter(has=page.get_by_text('Test book · Channel lorebook · #scene', exact=True)).first
            expansion.get_by_text('Test book · Channel lorebook · #scene', exact=True).click()
            expansion.get_by_role('button', name='Edit entries in Lore', exact=True).click()
            page.get_by_text('Save or reset your changes to Alice first.', exact=True).wait_for(timeout=5000)
            self.assertTrue(self.tab_selected(page, 'Imports'))
            self.assertTrue(bar.is_visible())
            page.get_by_role('tab', name='Characters', exact=True).click()
            self.assertEqual(description.input_value(), dirty_value)
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_savebar_reset_unchecks_world_move_confirmation(self):
        """MNT-21: Reset also unticks 'Confirm moving worlds; ineligible casts will be cleared'."""
        context, page, errors, card, description = self.open_alice()
        try:
            bar = page.get_by_role('region', name='Unsaved changes')
            confirm = card.get_by_role('checkbox', name='Confirm moving worlds; ineligible casts will be cleared')
            card.get_by_label('Home world', exact=True).click()
            page.get_by_role('option', name='Annex', exact=True).click()
            confirm.wait_for(state='visible', timeout=5000)
            confirm.click()
            page.wait_for_function("() => [...document.querySelectorAll('.character-card [role=checkbox]')].some(c => c.getAttribute('aria-checked') === 'true')", timeout=5000)
            description.fill(description.input_value() + ' dirty')
            bar.wait_for(state='visible', timeout=5000)
            bar.get_by_role('button', name='Reset', exact=True).click()
            bar.wait_for(state='hidden', timeout=5000)
            page.wait_for_function("() => ![...document.querySelectorAll('.character-card [role=checkbox]')].some(c => c.getAttribute('aria-checked') === 'true')", timeout=5000)
            confirm.wait_for(state='hidden', timeout=5000)
            page.wait_for_function("() => ![...document.querySelectorAll('.character-card [role=checkbox]')].some(c => c.getAttribute('aria-checked') === 'true')", timeout=5000)
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_world_move_confirmation_shows_only_when_world_changes(self):
        """UI-25: the world-move checkbox is hidden on open, shown after choosing Annex, hidden after choosing the saved world again."""
        context, page, errors, card, description = self.open_alice()
        try:
            confirm = card.get_by_role('checkbox', name='Confirm moving worlds; ineligible casts will be cleared')
            self.assertFalse(confirm.is_visible())
            saved = card.get_by_label('Home world', exact=True).input_value()
            card.get_by_label('Home world', exact=True).click()
            page.get_by_role('option', name='Annex', exact=True).click()
            confirm.wait_for(state='visible', timeout=5000)
            card.get_by_label('Home world', exact=True).click()
            page.get_by_role('option', name=saved, exact=True).click()
            confirm.wait_for(state='hidden', timeout=5000)
            card.get_by_label('Home world', exact=True).click()
            page.get_by_role('option', name='Annex', exact=True).click()
            confirm.wait_for(state='visible', timeout=5000)
            confirm.click()
            ticked = "() => [...document.querySelectorAll('.character-card [role=checkbox]')].some(c => c.getAttribute('aria-checked') === 'true')"
            page.wait_for_function(ticked, timeout=5000)
            card.get_by_label('Home world', exact=True).click()
            page.get_by_role('option', name=saved, exact=True).click()
            confirm.wait_for(state='hidden', timeout=5000)
            card.get_by_label('Home world', exact=True).click()
            page.get_by_role('option', name='Annex', exact=True).click()
            confirm.wait_for(state='visible', timeout=5000)
            page.wait_for_function('() => ![...document.querySelectorAll(".character-card [role=checkbox]")].some(c => c.getAttribute("aria-checked") === "true")', timeout=5000)
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_more_menu_keeps_card_open_and_archive_restore_round_trip(self):
        """UI-25: the more-actions button does not collapse the card and lists Archive and Delete character; Archive marks the header and audits, Restore undoes it."""
        context, page, errors, card, description = self.open_alice()
        try:
            self.assertEqual(card.get_by_role('button', name='More actions for Alice', exact=True).get_attribute('aria-label'), 'More actions for Alice')
            card.get_by_role('button', name='More actions for Alice', exact=True).click()
            page.get_by_role('menuitem', name='Archive', exact=True).wait_for(timeout=5000)
            self.assertTrue(page.get_by_role('menuitem', name='Delete character', exact=True).is_visible())
            self.assertTrue(description.is_visible())
            page.keyboard.press('Escape')
            page.get_by_role('menuitem', name='Archive', exact=True).wait_for(state='hidden', timeout=5000)
            audit_before = len([r for r in self.state()['audit'] if r['action'] == 'character.archive'])
            self.open_more_item(page, card, 'Alice', 'Archive')
            archived = page.locator('.character-card').filter(has=page.get_by_text('Alice · Archived', exact=True)).first
            archived.wait_for(timeout=5000)
            self.wait_for(lambda: self.alice_row()['archived'])
            self.assertEqual(len([r for r in self.state()['audit'] if r['action'] == 'character.archive']), audit_before + 1)
            self.open_more_item(page, archived, 'Alice', 'Restore')
            page.locator('.character-card').filter(has=page.get_by_text('Alice', exact=True)).first.wait_for(timeout=5000)
            self.wait_for(lambda: not self.alice_row()['archived'])
            self.assertEqual(page.get_by_text('Alice · Archived', exact=True).count(), 0)
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_archive_conflict_refreshes_the_list_and_retry_succeeds(self):
        """MNT-05: Archive from a stale card shows the conflict toast and archives nothing, then the list refreshes so Archive works on retry."""
        context, page, errors, card, description = self.open_alice()
        original = description.input_value()
        try:
            self.assertTrue(self.context.request.post(self.url + '/_test/change-character', data={'name': 'Alice', 'description': 'Changed behind the page'}).ok)
            audit_before = len([r for r in self.state()['audit'] if r['action'] == 'character.archive'])
            page.evaluate("document.querySelector('.character-card').dataset.stale = '1'")
            self.open_more_item(page, card, 'Alice', 'Archive')
            page.get_by_text('Character changed; reload before archiving', exact=True).wait_for(timeout=5000)
            self.assertFalse(self.alice_row()['archived'])
            self.assertEqual(len([r for r in self.state()['audit'] if r['action'] == 'character.archive']), audit_before)
            self.assertEqual(page.get_by_text('Alice · Archived', exact=True).count(), 0)
            # The refresh replaces the card element; the marker on the old element survives only if it never ran.
            page.wait_for_function("() => !document.querySelector('.character-card[data-stale]')", timeout=3000)
            alice = page.locator('.character-card').filter(has=page.get_by_text('Alice', exact=True)).first
            self.open_more_item(page, alice, 'Alice', 'Archive')
            page.get_by_text('Alice · Archived', exact=True).wait_for(timeout=5000)
            self.wait_for(lambda: self.alice_row()['archived'])
            self.assertFalse(errors, errors)
        finally:
            context.close()
            if self.alice_row()['archived']:
                ctx2, page2, _ = self.ux_page('/admin/guild/1?tab=characters')
                try:
                    archived = page2.locator('.character-card').filter(has=page2.get_by_text('Alice · Archived', exact=True)).first
                    self.open_more_item(page2, archived, 'Alice', 'Restore')
                    self.wait_for(lambda: not self.alice_row()['archived'])
                finally:
                    ctx2.close()
            self.restore_alice(original)

    def test_more_menu_hides_tooltip_and_keyboard_returns_focus(self):
        """UI-39: no "More actions" tooltip stays over the open menu; Tab, Enter opens it, Escape closes it and focus returns to the button."""
        context, page, errors, card, description = self.open_alice()
        try:
            button = card.get_by_role('button', name='More actions for Alice', exact=True)
            button.hover()
            page.locator('.q-tooltip', has_text='More actions').wait_for(state='visible', timeout=5000)
            button.click()
            page.get_by_role('menuitem', name='Archive', exact=True).wait_for(timeout=5000)
            page.wait_for_timeout(500)
            self.assertEqual(page.locator('.q-tooltip:visible', has_text='More actions').count(), 0)
            page.keyboard.press('Escape')
            page.get_by_role('menuitem', name='Archive', exact=True).wait_for(state='hidden', timeout=5000)
            page.mouse.move(0, 0)
            button.focus()
            page.keyboard.press('Enter')
            page.get_by_role('menuitem', name='Archive', exact=True).wait_for(timeout=5000)
            page.keyboard.press('Escape')
            page.get_by_role('menuitem', name='Archive', exact=True).wait_for(state='hidden', timeout=5000)
            self.assertEqual(page.evaluate('() => document.activeElement && document.activeElement.getAttribute("aria-label")'), 'More actions for Alice')
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_savebar_name_edit_shows_saved_name(self):
        """MNT-21: editing Name shows the bar with the SAVED name, not the edited one."""
        context, page, errors, card, description = self.open_alice()
        try:
            bar = page.get_by_role('region', name='Unsaved changes')
            name = card.get_by_label('Name', exact=True)
            name.fill('Alicia')
            bar.wait_for(state='visible', timeout=5000)
            self.assertEqual(bar.get_by_text('Alice has unsaved changes.', exact=True).count(), 1)
            bar.get_by_role('button', name='Reset', exact=True).click()
            bar.wait_for(state='hidden', timeout=5000)
            self.assertEqual(name.input_value(), 'Alice')
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_savebar_unconfirmed_world_move_is_refused(self):
        """MNT-21: changing Home world without the confirm box makes Save changes toast 'Confirm the world move before saving'; bar stays, store unchanged. (Fixture has a second world, 'Annex'.)"""
        context, page, errors, card, description = self.open_alice()
        try:
            bar = page.get_by_role('region', name='Unsaved changes')
            before = self.alice_row()
            card.get_by_label('Home world', exact=True).click()
            page.get_by_role('option', name='Annex', exact=True).click()
            bar.wait_for(state='visible', timeout=5000)
            bar.get_by_role('button', name='Save changes', exact=True).click()
            page.get_by_text('Confirm the world move before saving', exact=True).wait_for(timeout=5000)
            self.assertTrue(bar.is_visible())
            self.assertEqual(self.alice_row(), before)
            bar.get_by_role('button', name='Reset', exact=True).click()
            bar.wait_for(state='hidden', timeout=5000)
            self.assertFalse(errors, errors)
        finally:
            context.close()

    def test_savebar_fits_phone_width(self):
        """MNT-21: at 390 px the bar's buttons are visible and the page does not scroll horizontally."""
        context, page, errors, card, description = self.open_alice()
        try:
            page.set_viewport_size({'width': 390, 'height': 800})
            bar = page.get_by_role('region', name='Unsaved changes')
            description.fill(description.input_value() + ' phone')
            bar.wait_for(state='visible', timeout=5000)
            self.assertTrue(bar.get_by_role('button', name='Save changes', exact=True).is_visible())
            self.assertTrue(bar.get_by_role('button', name='Reset', exact=True).is_visible())
            self.assertTrue(page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'))
            box = bar.bounding_box()
            width, height = page.evaluate('[window.innerWidth, window.innerHeight]')
            self.assertGreaterEqual(box['x'], 0)
            self.assertLessEqual(box['x'] + box['width'], width)
            self.assertLessEqual(box['y'] + box['height'], height)
            self.assertFalse(errors, errors)
        finally:
            context.close()
