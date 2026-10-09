"""Measure dashboard latency and Discord API calls against the mocked fixture server.

Usage: python scripts/bench_dashboard.py [--profile latency|ratelimited|all] [--seed 10,500] [--json out.json]

Starts tests/dashboard_server.py in benchmark mode (mock Discord with configurable latency and
rate-limit reset), drives it with headless Chromium, and prints one JSON object per scenario.
"Settle" is when the fixture's Discord call counter stops changing for SETTLE_QUIET seconds,
so it covers the work the server does after the browser considers the action finished.
Never point this at a real Discord application or production database.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SETTLE_QUIET = 1.5
PROFILES = {
    # Network latency only: Discord answers in 150 ms and never asks the client to wait.
    'latency': {'LLMCORD_BENCH_DISCORD_LATENCY': '0.15', 'LLMCORD_BENCH_RESET_AFTER': '0'},
    # Same latency plus Remaining: 0 / Reset-After: 1 s on every user-guild check (assumed Discord-like).
    'ratelimited': {'LLMCORD_BENCH_DISCORD_LATENCY': '0.15', 'LLMCORD_BENCH_RESET_AFTER': '1.0'},
}


def start_server(folder: Path, env_extra: dict) -> tuple[subprocess.Popen, str]:
    import trustme
    cert = trustme.CA().issue_cert('localhost')
    cert.private_key_pem.write_to_path(folder / 'key.pem')
    cert.cert_chain_pems[0].write_to_path(folder / 'cert.pem')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'https://localhost:{port}'
    env = {**os.environ, 'LLMCORD_TEST_PORT': str(port), 'LLMCORD_TEST_KEY': str(folder / 'key.pem'),
           'LLMCORD_TEST_CERT': str(folder / 'cert.pem'), 'LLMCORD_BENCH': '1', **env_extra}
    log = (folder / 'server.log').open('w')
    process = subprocess.Popen([sys.executable, '-m', 'tests.dashboard_server'], cwd=ROOT, env=env, stdout=log, stderr=log)
    for _ in range(300):
        try:
            if httpx.get(url + '/_test/metrics', verify=False, timeout=1).status_code == 200:
                return process, url
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    process.terminate()
    raise RuntimeError((folder / 'server.log').read_text())


def run_profile(name: str, seed: str) -> list[dict]:
    from playwright.sync_api import sync_playwright
    results = []
    with tempfile.TemporaryDirectory() as temp:
        process, url = start_server(Path(temp), {**PROFILES[name], 'LLMCORD_BENCH_SEED': seed})
        http = httpx.Client(verify=False, base_url=url)

        def metrics():
            return http.get('/_test/metrics').json()

        def settle(started):
            last, changed = metrics(), time.monotonic()
            while time.monotonic() - changed < SETTLE_QUIET:
                time.sleep(0.1)
                current = metrics()
                if current != last:
                    last, changed = current, time.monotonic()
            return last, changed - started

        def measure(scenario, action):
            http.post('/_test/metrics/reset')
            started = time.monotonic()
            action()
            ready = time.monotonic() - started
            calls, settled = settle(started)
            results.append({'profile': name, 'scenario': scenario, 'ready_s': round(ready, 2),
                            'settled_s': round(max(settled, ready), 2), 'discord_calls': calls,
                            'user_guild_checks': calls.get('GET /users/@me/guilds', 0)})

        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                context = browser.new_context(ignore_https_errors=True, viewport={'width': 1400, 'height': 1000})
                context.add_cookies([{'name': 'llmcord_session', 'value': 'browser-test-session', 'url': url,
                                      'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
                page = context.new_page()

                def cold_load():
                    page.goto(url + '/admin/guild/1', wait_until='networkidle', timeout=120000)
                    page.get_by_text('Server administration').wait_for(timeout=120000)
                measure('guild page cold load', cold_load)

                def characters_tab():
                    page.get_by_role('tab', name='Characters').click()
                    page.wait_for_load_state('networkidle')
                measure('switch to Characters tab', characters_tab)

                def expand_character():
                    page.locator('.character-card').filter(has_text='Bench 000').first.click()
                    page.wait_for_load_state('networkidle')
                measure('expand one character', expand_character)

                def lore_panel_build():
                    page.get_by_role('tab', name='Lore').click()
                    page.get_by_label('Search content and keywords').first.wait_for(timeout=60000)
                    page.wait_for_load_state('networkidle')
                measure('open Lore tab (panel build)', lore_panel_build)

                def lore_search():
                    box = page.get_by_label('Search content and keywords').first
                    box.press_sequentially('lighthouse', delay=80)
                    page.wait_for_load_state('networkidle')
                measure('type 10 chars in lore search', lore_search)

                def save_cast():
                    page.get_by_role('tab', name='Server setup').click()
                    page.locator('button.channel-toggle').first.click()
                    page.get_by_role('switch', name='Ambient participation').first.click()
                    page.get_by_role('button', name='Save changes', exact=True).click()
                    page.get_by_text('Channel settings saved', exact=True).wait_for(timeout=60000)
                measure('Save channel settings', save_cast)

                browser.close()
        except Exception:
            print((Path(temp) / 'server.log').read_text()[-4000:], file=sys.stderr)
            raise
        finally:
            try:
                http.post('/_test/stop', timeout=5)
                process.wait(timeout=10)
            except (httpx.HTTPError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=10)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--profile', choices=[*PROFILES, 'all'], default='all')
    parser.add_argument('--seed', default='10,500', help='extra characters,lore entries to seed')
    parser.add_argument('--json', type=Path, help='also write results to this file')
    args = parser.parse_args()
    results = []
    for name in PROFILES if args.profile == 'all' else [args.profile]:
        results.extend(run_profile(name, args.seed))
    report = {'host': platform.platform(), 'python': platform.python_version(), 'seed': args.seed, 'results': results}
    for row in results:
        print(json.dumps(row))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
