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

    def wait_for(self, predicate, polls=50):
        for _ in range(polls):
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
        page.get_by_text('fixture-model', exact=True).wait_for()
        # Rendered controls precede the connection that delivers their events.
        page.wait_for_function('window.did_handshake === true && window.socket?.connected === true')
        world_panel = page.locator('.space-card').filter(has=page.get_by_text('World · world', exact=True))
        world_panel.get_by_text('World · world', exact=True).click()
        world_panel.get_by_label('World guidelines', exact=True).fill('The setting is a courier guild. Treat users as guild members.')
        world_panel.get_by_role('button', name='Save world guidelines', exact=True).click()
        self.wait_for(lambda: any(row['kind'] == 'space' and 'courier guild' in row['content'] for row in self.state()['guidelines']))
        channel_panel = page.locator('.channel-card').filter(has_text='#scene')
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
            self.assertEqual(context.request.post(self.url + '/_test/counters/reset').status, 200)
            self.load(page, self.url + '/admin/guild/1')
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

    def load(self, page, url):
        """goto + live-socket wait. Chromium sometimes fails a NiceGUI static asset itself (net::ERR_TOO_MANY_RETRIES, the request
        never reaches the server) and nicegui.js then throws a cssRules SecurityError. Only that pair (every page error is that
        SecurityError and a /_nicegui/ request failed) is reloaded, up to twice; any other page error is kept for the assertion.
        """
        for attempt in range(3):
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
            self.load(page, self.url + '/admin/guild/1?tab=characters')
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
            _worlds_section(page).get_by_label('Name', exact=True).wait_for()
            page.wait_for_timeout(300)
            self.assertEqual(self.panel_reads(context), {**built, 'model_usage_summary': 1})
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
        header = card.locator('.q-item').filter(has=page.get_by_text(label, exact=True)).first
        if not header.is_visible():
            card.get_by_text('Alice', exact=True).first.click()
            header.wait_for(state='visible')
        if header.get_attribute('aria-expanded') != 'true':
            header.click()
        return card, card.locator('.q-expansion-item').filter(has=page.get_by_text(label, exact=True)).last

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
                if not card.get_by_text('Fallback static avatar', exact=True).is_visible():
                    card.get_by_text('Alice', exact=True).first.click()
                fallback = card.locator('.fallback-avatar')
                if not fallback.get_by_text('Used when the selected emotion', exact=False).is_visible():
                    card.get_by_text('Fallback static avatar', exact=True).click()
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
            self.assertLess(row.bounding_box()['height'], 70)
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
