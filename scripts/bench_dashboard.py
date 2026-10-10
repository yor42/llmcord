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


# Records every socket frame (direction, size, performance.now()) and which transport carries them.
WS_PROBE = """
window.__frames = [];
window.__transports = [];
const NativeWS = window.WebSocket;
window.WebSocket = function(...args) {
  const ws = new NativeWS(...args);
  window.__transports.push('websocket ' + String(args[0]).split('?')[0]);
  ws.addEventListener('message', e => window.__frames.push({d: 'in', t: performance.now(), n: String(e.data).length}));
  const send = ws.send.bind(ws);
  ws.send = data => { window.__frames.push({d: 'out', t: performance.now(), n: String(data).length}); return send(data); };
  return ws;
};
window.WebSocket.prototype = NativeWS.prototype;
Object.assign(window.WebSocket, {CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3});
const NativeOpen = XMLHttpRequest.prototype.open;
XMLHttpRequest.prototype.open = function(m, u, ...r) {
  if (String(u).includes('socket.io')) window.__transports.push('xhr ' + m);
  return NativeOpen.call(this, m, u, ...r);
};
"""

# Clicks in the page so the timestamps share one clock with the frame log, then waits for the expand to finish.
EXPAND_PROBE = """async (el) => {
  const frames = window.__frames, mark = frames.length;
  const t0 = performance.now();
  const longTasks = [], mutations = [], ticks = [];
  const po = new PerformanceObserver(list => list.getEntries().forEach(e => longTasks.push([Math.round(e.startTime - t0), Math.round(e.duration)])));
  try { po.observe({entryTypes: ['longtask']}); } catch (e) {}
  const mo = new MutationObserver(() => mutations.push(Math.round(performance.now() - t0)));
  mo.observe(el, {childList: true, subtree: true, attributes: true});
  (el.querySelector('.q-item') || el).click();
  const t1 = performance.now();
  const initial = el.getBoundingClientRect().height;
  let last = initial, changedAt = null, stableSince = t1, end = t1;
  await new Promise(resolve => {
    const tick = () => {
      const now = performance.now(), height = el.getBoundingClientRect().height;
      ticks.push(Math.round(now - t0));
      if (height !== last) { last = height; stableSince = now; if (changedAt === null) changedAt = now; }
      if ((changedAt !== null && now - stableSince > 100) || now - t0 > 3000) { end = stableSince; resolve(); } else requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
  po.takeRecords().forEach(e => longTasks.push([Math.round(e.startTime - t0), Math.round(e.duration)]));
  mo.takeRecords().forEach(() => mutations.push(Math.round(performance.now() - t0)));
  po.disconnect(); mo.disconnect();
  const mine = frames.slice(mark);
  const out = mine.find(f => f.d === 'out'), back = mine.find(f => f.d === 'in' && out && f.t >= out.t);
  const rel = f => f ? Math.round((f.t - t0) * 10) / 10 : null;
  return {click_call_ms: t1 - t0, send_ms: rel(out), reply_ms: rel(back), first_mutation_ms: mutations[0] ?? null,
          mutation_count: mutations.length,
          height_change_ms: changedAt === null ? null : changedAt - t0, animation_end_ms: end - t0,
          frame_ticks_ms: ticks.slice(0, 8), long_tasks: longTasks.slice(0, 8), frames: mine.map(f => [f.d, rel(f), f.n]),
          transports: window.__transports};
}"""


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

        expand_detail = {}

        def measure(scenario, action):
            http.post('/_test/metrics/reset')
            started = time.monotonic()
            action()
            ready = time.monotonic() - started
            calls, settled = settle(started)
            row = {'profile': name, 'scenario': scenario, 'ready_s': round(ready, 2),
                   'settled_s': round(max(settled, ready), 2), 'discord_calls': calls,
                   'user_guild_checks': calls.get('GET /users/@me/guilds', 0)}
            if expand_detail:
                row['parts'] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in expand_detail.items()}
                expand_detail.clear()
            results.append(row)

        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                context = browser.new_context(ignore_https_errors=True, viewport={'width': 1400, 'height': 1000})
                context.add_cookies([{'name': 'llmcord_session', 'value': 'browser-test-session', 'url': url,
                                      'secure': True, 'httpOnly': True, 'sameSite': 'Lax'}])
                context.add_init_script(WS_PROBE)
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

                def expand_parts():
                    # Same click sequence as above, split into locator / click-to-reply / animation parts.
                    page.get_by_text('Bench 000').first.click()  # collapse the previous card first
                    page.wait_for_timeout(1000)
                    started = time.perf_counter()
                    card = page.locator('.character-card').filter(has_text='Bench 001').first
                    handle = card.element_handle()
                    resolved = time.perf_counter() - started
                    probe = handle.evaluate(EXPAND_PROBE)
                    waited = time.perf_counter()
                    page.wait_for_load_state('networkidle')
                    probe.update(locator_ms=resolved * 1000, networkidle_ms=(time.perf_counter() - waited) * 1000)
                    expand_detail.update(probe)
                measure('expand one character (split)', expand_parts)

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
