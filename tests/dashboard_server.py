"""Isolated HTTPS browser-test fixture. Never used by production entry points."""
import asyncio
import json
import os
import time
from collections import Counter

import httpx
import uvicorn
from fastapi import Request

from llmcord_core.web import create_app


def main():
    port = int(os.environ['LLMCORD_TEST_PORT'])
    state = {'admin': True, 'guild_checks': 0, 'permission_delay': 0, 'rate_limited': False, 'guild_name': 'Test server'}
    # Opt-in benchmark knobs (scripts/bench_dashboard.py); unset keeps the browser-test behavior.
    bench = os.environ.get('LLMCORD_BENCH') == '1'
    latency = float(os.environ.get('LLMCORD_BENCH_DISCORD_LATENCY', '0'))
    reset_after = os.environ.get('LLMCORD_BENCH_RESET_AFTER', '0.05')
    calls = Counter()
    async def discord_api(request):
        calls[request.method + ' ' + request.url.path.removeprefix('/api/v10')] += 1
        if latency:
            await asyncio.sleep(latency)
        if request.url.path.endswith('/users/@me/guilds'):
            delay = state['permission_delay']
            state['permission_delay'] = 0
            if delay:
                await asyncio.sleep(delay)
            state['guild_checks'] += 1
            if state['rate_limited']:
                # /_test/rate-limit: Discord keeps answering 429, so the guard gives up with a 503.
                return httpx.Response(429, json={'retry_after': 0.01})
            if state['guild_checks'] == 1 and not bench:
                return httpx.Response(429, json={'retry_after': 0.05})
            return httpx.Response(200, json=[{'id': '1', 'name': state['guild_name'], 'permissions': '8' if state['admin'] else '0'}],
                                  headers={'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset-After': reset_after})
        if request.method == 'POST' and request.url.path.endswith('/messages'):
            return httpx.Response(200, json={'id': '900', 'attachments': [{'url': 'https://cdn.discordapp.com/attachments/100/900/image.png?ex=test'}]})
        return httpx.Response(200, json=[{'id': '100', 'name': 'scene', 'type': 0},
            {'id': '200', 'name': 'assets', 'type': 0, 'permission_overwrites': [{'id': '1', 'deny': str(1 << 10)}]}])
    app = create_app(':memory:', f'https://localhost:{port}', 'test-client', 'test-secret', 'test-token', httpx.AsyncClient(transport=httpx.MockTransport(discord_api)), config_path='tests/nonexistent-config.yaml', operator_ids=frozenset({6}))
    store = app.state.store
    app.state.admin.model_config = {'dialogue': 'fixture', 'director': 'fixture', 'memory': 'fixture', 'profiles': {'fixture': {'model': 'fixture-model'}}}
    from llmcord_core.usage import ModelUsage
    store.record_model_usage(ModelUsage(1, 'fixture', 'fixture-model', 'dialogue', 123, 45, 0, 0, 0.001, 'configured', time.time()))
    # Monitoring tab data: two models over several days, two channels, every feature, one old row and one unpriced call; guild 2 must never show.
    # Inserted directly so the operator's spend_days (global caps panel) still sees only the one recorded call above.
    for guild, days_ago, profile, model, channel, feature, inputs, outputs, cost in (
            (1, 0.1, 'fixture', 'fixture-model', 100, 'reply', 1000, 200, 0.002), (1, 0.2, 'fixture', 'fixture-model', 100, 'ambient', 500, 100, 0.001),
            (1, 1.5, 'fixture', 'fixture-model', 777, 'summon', 2000, 300, 0.004), (1, 2.5, 'fixture', 'second-model', 100, 'memory', 3000, 400, None),
            (1, 2.6, 'fixture', 'second-model', 100, 'catchup', 4000, 500, 0.01), (1, 3.5, 'fixture', 'fixture-model', None, '', 700, 70, 0.0005),
            (2, 0.04, 'other', 'other-guild-model', 100, 'reply', 999999, 888888, 5.0)):
        store.db.execute("INSERT INTO model_usage(guild_id,profile,model,role,input_tokens,output_tokens,cached_tokens,reasoning_tokens,cost_usd,cost_basis,created_at,channel_id,feature) VALUES(?,?,?,'dialogue',?,?,0,0,?,'configured',?,?,?)",
                         (guild, profile, model, inputs, outputs, cost, time.time() - days_ago * 86400, channel, feature))
    store.db.commit()
    world = store.create_space(1, 'World', 'world')
    store.bind_channel(1, 100, world)
    store.add_character(1, world, 'Alice', {'name': 'Alice', 'description': 'A cheerful courier'}, None, [])
    store.add_lore(1, 'channel', 100, 'The moon is red', ['moon'])
    book_id = store.create_lorebook(1, 'Test book', 'channel', 100)
    from llmcord_core.lorebooks import parse_lorebook
    # Zero-padded uids: the import sorts entries by uid as strings, so this keeps the book in numeric order (1, 2, ... 10).
    store.sync_lorebook(1, book_id, parse_lorebook(json.dumps({'entries': {f'{i:02d}': {'key': ['fixture'], 'content': f'Fixture entry {i}'} for i in range(75)}}).encode()), {}, 0)
    # PERF-05: a guild-target book (named to sort before 'Test book', which the lore board picks as the default right owner) and a second space, so lorebook_links is observably once per book (not books x spaces).
    store.create_space(1, 'Annex', 'world')
    store.create_lorebook(1, 'Guild book', 'guild')
    if bench:
        seed_benchmark(store, world, os.environ.get('LLMCORD_BENCH_SEED', '10,500'))
    app.state.sessions['browser-test-session'] = {'user': {'id': '4', 'username': 'Test admin'}, 'expires': time.time() + 3600,
        'token_expires': time.time() + 3600, 'csrf': 'browser-test-csrf', 'access': 'test', 'refresh': 'test'}
    # A second session for tests that must not depend on the workflow test (which expires the first one).
    app.state.sessions['browser-search-session'] = {'user': {'id': '4', 'username': 'Test admin'}, 'expires': time.time() + 3600,
        'token_expires': time.time() + 3600, 'csrf': 'browser-search-csrf', 'access': 'test', 'refresh': 'test'}
    # Its own session for the rejected-live-event test, which revokes, rate-limits and finally expires it.
    app.state.sessions['browser-reject-session'] = {'user': {'id': '4', 'username': 'Test admin'}, 'expires': time.time() + 3600,
        'token_expires': time.time() + 3600, 'csrf': 'browser-reject-csrf', 'access': 'test', 'refresh': 'test'}

    # Its own session for the guild-lookup counting test (PERF-05).
    app.state.sessions['browser-snapshot-session'] = {'user': {'id': '4', 'username': 'Test admin'}, 'expires': time.time() + 3600,
        'token_expires': time.time() + 3600, 'csrf': 'browser-snapshot-csrf', 'access': 'test', 'refresh': 'test'}

    # Its own session for the UX-01 tests (tab URL sync, success toasts, channel names).
    app.state.sessions['browser-ux-session'] = {'user': {'id': '4', 'username': 'Test admin'}, 'expires': time.time() + 3600,
        'token_expires': time.time() + 3600, 'csrf': 'browser-ux-csrf', 'access': 'test', 'refresh': 'test'}

    # Its own session for the UI-29 account menu test, which signs out.
    app.state.sessions['browser-account-session'] = {'user': {'id': '4', 'username': 'Test admin'}, 'expires': time.time() + 3600,
        'token_expires': time.time() + 3600, 'csrf': 'browser-account-csrf', 'access': 'test', 'refresh': 'test'}

    # UI-37: a session with global_name and an avatar hash (fake data; the CDN URL is never fetched).
    app.state.sessions['browser-profile-session'] = {'user': {'id': '5', 'username': 'moonuser', 'global_name': 'Moon Display', 'avatar': 'a' * 32},
        'expires': time.time() + 3600, 'token_expires': time.time() + 3600, 'csrf': 'browser-profile-csrf', 'access': 'test', 'refresh': 'test'}

    # FEAT-08: an operator (user id 6, in operator_ids) and a plain server admin for the Bot settings page.
    app.state.sessions['browser-operator-session'] = {'user': {'id': '6', 'username': 'Test operator'}, 'expires': time.time() + 3600,
        'token_expires': time.time() + 3600, 'csrf': 'browser-operator-csrf', 'access': 'test', 'refresh': 'test'}
    app.state.sessions['browser-plain-session'] = {'user': {'id': '4', 'username': 'Test admin'}, 'expires': time.time() + 3600,
        'token_expires': time.time() + 3600, 'csrf': 'browser-plain-csrf', 'access': 'test', 'refresh': 'test'}

    # Count lore board renders (PERF-01/02): render_board loads each side through AdminStore.admin_entries_page,
    # counted here under the 'admin_entries' counter keys (one per side per render).
    counters = Counter()
    admin_store = app.state.admin.store
    original_admin_entries_page = admin_store.admin_entries_page
    def counting_admin_entries_page(guild_id, kind, owner_id, *args, **kwargs):
        counters['admin_entries'] += 1
        counters[f'admin_entries:{kind}:{owner_id}'] += 1
        return original_admin_entries_page(guild_id, kind, owner_id, *args, **kwargs)
    admin_store.admin_entries_page = counting_admin_entries_page

    # PERF-05: count guild-wide lookups under 'store:<name>'. app.state.store and app.state.admin.store are the same
    # object, so wrapping it once covers every dashboard call. /_test/state keeps using the unwrapped originals.
    assert admin_store is store
    originals = {}
    def count_store_method(name):
        original = originals[name] = getattr(store, name)
        def counting(*args, **kwargs):
            counters[f'store:{name}'] += 1
            return original(*args, **kwargs)
        setattr(store, name, counting)
    for name in ('list_spaces', 'list_characters', 'list_channels', 'list_lorebooks', 'thread_lore_scopes', 'lorebook_links',
                 'usage_report', 'list_presets'):  # the last two are panel-specific (Server setup / Prompt presets): R5 step 6
        count_store_method(name)

    @app.get('/_test/state')
    async def snapshot():
        return {'spaces': [dict(r) for r in originals['list_spaces'](1)],
            'guidelines': [dict(r) for r in store.all('SELECT * FROM scene_guidelines')],
            'guild_lore': [dict(r) for r in store.all('SELECT * FROM guild_lore_entries')],
            'lorebooks': [dict(r) for r in originals['list_lorebooks'](1)],
            'characters': [dict(r) for r in store.all('SELECT id,guild_id,world_id,name,card,archived,avatar IS NOT NULL AS has_static_avatar FROM characters')],
            'lore': [dict(r) for r in store.all('SELECT * FROM lore')],
            'books': [dict(r) for r in store.all('SELECT * FROM lorebook_entries')],
            'presets': [dict(r) for r in store.list_presets(1)], 'active': store.active_preset(1)['id'],
            'active_bundle': store.active_preset(1)['bundle'], 'assets': [dict(r) for r in store.all('SELECT * FROM avatar_assets')],
            'guild_timezone': store.guild_timezone(1), 'catchup_anywhere': store.catchup_anywhere(1), 'turn_log': store.turn_log_settings(1), 'audit': [dict(r) for r in store.all('SELECT * FROM admin_audit ORDER BY id')],
            'slots': [{'character_id': r['character_id'], 'slot_key': r['slot_key'], 'label': r['label'], 'has_image': bool(r['image'])} for r in store.all('SELECT * FROM avatar_slots')]}

    @app.get('/_test/uploads')
    async def uploads():
        from nicegui import Client
        return [f'/admin{element._registered_url}' for client in Client.instances.values()
            if getattr(client, 'llmcord_binding', None) == ('browser-test-session', 1)
            for element in client.elements.values() if hasattr(element, '_registered_url')]

    @app.get('/_test/metrics')
    async def metrics():
        return dict(calls)

    @app.post('/_test/metrics/reset')
    async def reset_metrics():
        calls.clear()
        return {'ok': True}

    @app.get('/_test/counters')
    async def get_counters():
        return dict(counters)

    @app.post('/_test/counters/reset')
    async def reset_counters():
        counters.clear()
        return {'ok': True}

    @app.post('/_test/seed-search-lore')
    async def seed_search_lore():
        # World (space) lore for the debounced-search test; only the first two contain the full query.
        for content, keys in (('Mara keeps the lighthouse lamp burning', ['keeper']),
                              ('Ships steer by the north beacon', ['lighthouse']),
                              ('Lightning storms close the harbor', ['weather'])):
            store.add_lore(1, 'space', world, content, keys)
        return {'ok': True, 'owner': f'space:{world}'}

    @app.post('/_test/revoke')
    async def revoke():
        state['admin'] = False
        # Stand-in for the 300 s guild-list TTL (PERF-01 / D1) elapsing after Discord revoked access:
        # drop every cached guild list so the next guard refetches and sees the revocation.
        app.state.auth.forget_guilds()
        return {'ok': True}

    @app.post('/_test/change-lore')
    async def change_lore(request: Request):
        value = await request.json()
        row = store.entry_by_key(1, value['key'])
        store.save_entry(1, row['owner_kind'], row['owner_id'], value['content'], row['rule'],
                         ref=row['ref'], expected_revision=row['revision'])
        return {'ok': True}

    @app.post('/_test/set-lore-enabled')
    async def set_lore_enabled(request: Request):
        # Enable or disable one lore entry (by its content) through the real store API (UI-40 Disabled pill test).
        value = await request.json()
        row = next(r for r in store.all('SELECT entry_key FROM lore WHERE guild_id = 1 AND content = ?', (value['content'],)))
        entry = store.entry_by_key(1, row['entry_key'])
        store.save_entry(1, entry['owner_kind'], entry['owner_id'], entry['content'], {**entry['rule'], 'enabled': bool(value['enabled'])},
                         ref=entry['ref'], expected_revision=entry['revision'])
        return {'ok': True}

    @app.post('/_test/delete-lore')
    async def delete_lore(request: Request):
        # Delete guild-1 lore rows whose content exactly matches one of `contents` (browser tests clean up the entries they create).
        contents = (await request.json())['contents']
        for content in contents:
            for row in store.all('SELECT entry_key FROM lore WHERE guild_id = 1 AND content = ?', (content,)):
                entry = store.entry_by_key(1, row['entry_key'])
                store.delete_entry(1, entry['ref'], entry['revision'])
        return {'ok': True}

    @app.post('/_test/change-character')
    async def change_character(request: Request):
        # Another admin edits a character's description behind the open page's back (MNT-21 conflict test).
        value = await request.json()
        row = next(r for r in store.list_characters(1) if r['name'] == value['name'])
        card = {**json.loads(row['card']), 'description': value['description']}
        store.update_character(1, row['id'], row['world_id'], row['name'], card,
                               expected_revision=store.owner_revision(1, 'character', row['id']))
        return {'ok': True}

    @app.post('/_test/cleanup-presets')
    async def cleanup_presets(prefix: str, active: int = 0):
        # Best-effort reset after the UI-41 preset tests: re-activate the preset that was active before, then drop every preset whose name starts with `prefix`.
        row = store.one('SELECT MAX(revision) AS revision FROM prompt_revisions WHERE preset_id = ?', (active,)) if active else None
        with store.write_admin():
            store.db.execute('UPDATE guild_settings SET preset_id = ?, preset_revision = ? WHERE guild_id = 1', (active, row['revision'] if row else 0))
        for preset in store.list_presets(1):
            if preset['name'].startswith(prefix) and store.active_preset(1)['id'] != preset['id']:
                store.delete_preset(1, preset['id'])
        return {'ok': True}

    @app.post('/_test/cleanup-ui09')
    async def cleanup_ui09():
        # Best-effort reset after the UI-09 tests (even when one failed midway): drop their extra emotions and hub, put #scene back
        # on World, and clear Alice's fallback avatar (the fixture starts without one; this leaves avatar_manual=1 and a newer
        # owner revision, which no test depends on). Reads use raw SQL to leave the PERF counters alone.
        alice = next(r for r in store.all('SELECT id FROM characters WHERE name = ?', ('Alice',)))
        for slot in store.avatar_slots(1, alice['id']):
            if slot['slot_key'] in ('testy', 'vtest'):
                store.delete_avatar(1, alice['id'], slot['slot_key'], slot['revision'])
        for hub in store.all("SELECT id FROM spaces WHERE guild_id = 1 AND name = 'Audit hub'"):
            for linked in store.allowed_worlds(hub['id']):
                store.unlink_world(1, hub['id'], linked)
            store.delete_space(1, hub['id'], store.space_delete_impact(1, hub['id'])['revision'])
        store.bind_channel(1, 100, world)
        store.save_static_avatar(1, alice['id'], None, store.owner_revision(1, 'character', alice['id']))
        return {'ok': True}

    @app.post('/_test/delay-permission')
    async def delay_permission():
        state['permission_delay'] = 0.5
        # The delay only applies to a real Discord fetch; with the 300 s guild-list cache (PERF-01 / D1) the next
        # guard would hit the cache, so drop it to keep the next save pending for the delay.
        app.state.auth.forget_guilds()
        return {'ok': True}

    @app.post('/_test/rate-limit')
    async def rate_limit():
        state['rate_limited'] = True
        # Drop cached guild lists (as if the TTL elapsed) so the next guard asks Discord and gets the 429s.
        app.state.auth.forget_guilds()
        return {'ok': True}

    @app.post('/_test/restore')
    async def restore():
        state['admin'] = True
        state['rate_limited'] = False
        # Stand-in for the 300 s guild-list TTL (PERF-01 / D1) elapsing after access was restored: the revoke
        # check cached a non-admin guild list, so drop it for the next guard to see the restored permission.
        app.state.auth.forget_guilds()
        return {'ok': True}

    @app.post('/_test/guild-name')
    async def guild_name(request: Request):
        state['guild_name'] = (await request.json())['name']
        # Drop cached guild lists so the next page load shows the new name.
        app.state.auth.forget_guilds()
        return {'ok': True}

    @app.post('/_test/expire')
    async def expire(session: str = 'browser-test-session'):
        app.state.sessions[session]['expires'] = 0
        return {'ok': True}

    @app.post('/_test/budget-unpriced')
    async def budget_unpriced(n: int = 2):
        # FEAT-16: model calls without a cost estimate in the current period.
        for _ in range(n):
            store.record_model_usage(ModelUsage(1, 'fixture', 'fixture-model', 'dialogue', 10, 5, 0, 0, None, 'unconfigured', time.time()))
        return {'ok': True}

    @app.get('/_test/budget')
    async def budget_settings():
        return dict(store.budget_settings())

    @app.post('/_test/budget-save')
    async def budget_save():
        # Another operator saves behind the open page's back (stale revision test).
        current = store.budget_settings()
        store.save_budget(current['soft_cap_usd'], current['hard_cap_usd'], current['reset_day'], bool(current['channel_notice']), current['revision'])
        return {'ok': True}

    @app.post('/_test/stop')
    async def stop():
        server.should_exit = True
        return {'ok': True}

    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, ssl_keyfile=os.environ['LLMCORD_TEST_KEY'], ssl_certfile=os.environ['LLMCORD_TEST_CERT'], access_log=False, log_level='warning',
        # uvicorn's 5s default closes idle keep-alive sockets just as Chromium reuses them, which fails a request with ERR_TOO_MANY_RETRIES (a failed quasar css @import then throws a cssRules SecurityError in nicegui.js).
        timeout_keep_alive=120))
    server.run()


def seed_benchmark(store, world, spec):
    """Seed N extra characters (each with a static and an emotion avatar) and M channel lore entries."""
    from io import BytesIO
    from PIL import Image
    from llmcord_core.avatars import normalize_avatar
    characters, lore = (int(part) for part in spec.split(','))
    buffer = BytesIO()
    Image.new('RGB', (64, 64), (90, 120, 200)).save(buffer, 'PNG')
    image = normalize_avatar(buffer.getvalue())
    for index in range(characters):
        ident = store.add_character(1, world, f'Bench {index:03d}', {'name': f'Bench {index:03d}', 'description': 'x' * 400}, image, [])
        store.save_avatar(1, ident, 'happy', 'Happy', '', image)
    for index in range(lore):
        store.add_lore(1, 'channel', 100, f'Bench lore {index} ' + 'y' * 200, [f'key{index}'])


if __name__ == '__main__':
    main()
