"""Save full-page screenshots of every dashboard tab at desktop and phone width.

Usage: python scripts/screenshot_dashboard.py [--out DIR] [--tab NAME]... [--viewport desktop|phone]... [--strict]

Starts tests/dashboard_server.py (mock Discord, in-memory database, throwaway TLS cert), opens
/admin/guild/1?tab=<name> in headless Chromium and writes <tab>-<viewport>.png (full page) and
<tab>-<viewport>-top.png (first screen only, readable when the page is very tall) plus manifest.json
(per shot: path, top_path, tab, viewport, page_height in CSS px, page JS errors, and whether the page scrolls horizontally, which is
a hint for layout work, not a failure). Exits non-zero if the server fails to start or a tab never
loads; page JS errors are printed as warnings and only fail the run with --strict. Set LLMCORD_BROWSER_EXECUTABLE to use a specific Chromium.
Never point this at a real Discord application or production database.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
TABS = ('setup', 'characters', 'lore', 'imports', 'prompts', 'monitoring', 'currency')
VIEWPORTS = {
    'desktop': {'viewport': {'width': 1400, 'height': 1000}},
    'phone': {'viewport': {'width': 390, 'height': 844}, 'is_mobile': True, 'has_touch': True, 'device_scale_factor': 2},
}
SETTLE_SECONDS = 1.0


def start_server(folder: Path) -> tuple[subprocess.Popen, str]:
    import trustme
    cert = trustme.CA().issue_cert('localhost')
    cert.private_key_pem.write_to_path(folder / 'key.pem')
    cert.cert_chain_pems[0].write_to_path(folder / 'cert.pem')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'https://localhost:{port}'
    env = {**os.environ, 'LLMCORD_TEST_PORT': str(port), 'LLMCORD_TEST_KEY': str(folder / 'key.pem'),
           'LLMCORD_TEST_CERT': str(folder / 'cert.pem')}
    env.pop('LLMCORD_BENCH', None)
    log = (folder / 'server.log').open('w')
    process = subprocess.Popen([sys.executable, '-m', 'tests.dashboard_server'], cwd=ROOT, env=env, stdout=log, stderr=log)
    process.log = log
    try:
        for _ in range(300):
            try:
                if httpx.get(url + '/_test/state', verify=False, timeout=1).status_code == 200:
                    return process, url
            except httpx.HTTPError:
                pass
            if process.poll() is not None:
                break
            time.sleep(0.1)
    except BaseException:
        kill_server(process)
        raise
    kill_server(process)
    raise RuntimeError('dashboard fixture failed to start:\n' + (folder / 'server.log').read_text(errors='replace')[-4000:])


def kill_server(process: subprocess.Popen) -> None:
    try:
        process.terminate()
        process.wait(timeout=10)
    except BaseException:
        process.kill()
        process.wait(timeout=10)
    finally:
        process.log.close()


def stop_server(process: subprocess.Popen, url: str) -> None:
    try:
        httpx.post(url + '/_test/stop', verify=False, timeout=5)
        process.wait(timeout=10)
        process.log.close()
    except BaseException as exc:
        kill_server(process)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        print(f'server stop failed: {exc!r}', file=sys.stderr)


def failed_entry(out: Path, tab, viewport, error: str) -> dict:
    name = f'{tab}-{viewport}' if tab else None
    return {'path': str(out / f'{name}.png') if name else None, 'top_path': str(out / f'{name}-top.png') if name else None,
            'tab': tab, 'viewport': viewport, 'page_height': None, 'errors': [error],
            'horizontal_scroll': None, 'loaded': False}


def capture(browser, url: str, out: Path, tab: str, viewport: str) -> dict:
    context = browser.new_context(ignore_https_errors=True, **VIEWPORTS[viewport])
    errors: list[str] = []
    entry = {'path': str(out / f'{tab}-{viewport}.png'), 'top_path': str(out / f'{tab}-{viewport}-top.png'),
             'tab': tab, 'viewport': viewport, 'page_height': None, 'errors': errors,
             'horizontal_scroll': None, 'loaded': False}
    try:
        context.add_cookies([{'name': 'llmcord_session', 'value': 'browser-test-session', 'url': url,
                              'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
        page = context.new_page()
        page.on('pageerror', lambda e: errors.append(e.stack or str(e)))
        try:
            page.goto(f'{url}/admin/guild/1?tab={tab}', wait_until='networkidle', timeout=30000)
            page.wait_for_function('window.did_handshake === true && window.socket?.connected === true', timeout=30000)
            page.wait_for_function(
                "() => { const p = document.querySelector('.q-tab-panel'); return !!p && p.innerText.trim().length > 0; }",
                timeout=30000)
            entry['loaded'] = True
        except Exception as exc:
            errors.append(f'tab did not load: {exc}')
        time.sleep(SETTLE_SECONDS)
        entry['horizontal_scroll'] = page.evaluate('document.documentElement.scrollWidth > document.documentElement.clientWidth')
        entry['page_height'] = page.evaluate('document.documentElement.scrollHeight')
        page.screenshot(path=entry['path'], full_page=True)
        page.evaluate('window.scrollTo(0, 0)')
        page.screenshot(path=entry['top_path'])
    finally:
        context.close()
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default=str(ROOT / '.test-artifacts' / 'screenshots'))
    parser.add_argument('--tab', action='append', choices=TABS, help='repeatable; default all')
    parser.add_argument('--viewport', action='append', choices=tuple(VIEWPORTS), help='repeatable; default both')
    parser.add_argument('--strict', action='store_true', help='exit non-zero on page JS errors too')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tabs, viewports = args.tab or TABS, args.viewport or tuple(VIEWPORTS)
    from playwright.sync_api import sync_playwright
    shots: list[dict] = []
    with tempfile.TemporaryDirectory() as temp:
        try:
            process, url = start_server(Path(temp))
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
        interrupted = None
        try:
            with sync_playwright() as playwright:
                executable = os.environ.get('LLMCORD_BROWSER_EXECUTABLE')
                browser = playwright.chromium.launch(executable_path=executable, headless=True)
                try:
                    plan = [(tab, viewport) for tab in tabs for viewport in viewports]
                    for i, (tab, viewport) in enumerate(plan):
                        try:
                            shots.append(capture(browser, url, out, tab, viewport))
                        except BaseException as exc:
                            shots.append(failed_entry(out, tab, viewport, f'capture failed: {exc!r}'))
                            if isinstance(exc, KeyboardInterrupt):
                                interrupted = exc
                        print(f"{tab}-{viewport}: {'ok' if shots[-1]['loaded'] else 'FAILED'}", flush=True)
                        if not shots[-1]['loaded']:
                            print('skipping: ' + ', '.join(f'{t}-{v}' for t, v in plan[i + 1:]), file=sys.stderr)
                            break
                finally:
                    browser.close()
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                interrupted = exc
            else:
                shots.append(failed_entry(out, None, None, f'browser failed: {exc!r}'))
        finally:
            stop_server(process, url)
    (out / 'manifest.json').write_text(json.dumps(shots, indent=2))
    if interrupted:
        raise interrupted
    unloaded = [s for s in shots if not s['loaded']]
    noisy = [s for s in shots if s['errors']]
    for shot in noisy:
        print(f"warning {shot['tab']}-{shot['viewport']}: {shot['errors']}", file=sys.stderr)
    print(f'{len(shots)} screenshots in {out}; {len(unloaded)} failed to load; {len(noisy)} with page errors')
    return 1 if unloaded or (args.strict and noisy) else 0


if __name__ == '__main__':
    raise SystemExit(main())
