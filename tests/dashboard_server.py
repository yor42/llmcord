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
    state = {'admin': True, 'guild_checks': 0, 'permission_delay': 0}
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
            if state['guild_checks'] == 1 and not bench:
                return httpx.Response(429, json={'retry_after': 0.05})
            return httpx.Response(200, json=[{'id': '1', 'name': 'Test server', 'permissions': '8' if state['admin'] else '0'}],
                                  headers={'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset-After': reset_after})
        if request.method == 'POST' and request.url.path.endswith('/messages'):
            return httpx.Response(200, json={'id': '900', 'attachments': [{'url': 'https://cdn.discordapp.com/attachments/100/900/image.png?ex=test'}]})
        return httpx.Response(200, json=[{'id': '100', 'name': 'scene', 'type': 0},
            {'id': '200', 'name': 'assets', 'type': 0, 'permission_overwrites': [{'id': '1', 'deny': str(1 << 10)}]}])
    app = create_app(':memory:', f'https://localhost:{port}', 'test-client', 'test-secret', 'test-token', httpx.AsyncClient(transport=httpx.MockTransport(discord_api)), config_path='tests/nonexistent-config.yaml')
    store = app.state.store
    app.state.admin.model_config = {'dialogue': 'fixture', 'director': 'fixture', 'memory': 'fixture', 'profiles': {'fixture': {'model': 'fixture-model'}}}
    from llmcord_core.usage import ModelUsage
    store.record_model_usage(ModelUsage(1, 'fixture', 'fixture-model', 'dialogue', 123, 45, 0, 0, 0.001, 'configured', time.time()))
    world = store.create_space(1, 'World', 'world')
    store.bind_channel(1, 100, world)
    store.add_character(1, world, 'Alice', {'name': 'Alice', 'description': 'A cheerful courier'}, None, [])
    store.add_lore(1, 'channel', 100, 'The moon is red', ['moon'])
    book_id = store.create_lorebook(1, 'Test book', 'channel', 100)
    from llmcord_core.lorebooks import parse_lorebook
    store.sync_lorebook(1, book_id, parse_lorebook(json.dumps({'entries': {str(i): {'key': ['fixture'], 'content': f'Fixture entry {i}'} for i in range(75)}}).encode()), {}, 0)
    if bench:
        seed_benchmark(store, world, os.environ.get('LLMCORD_BENCH_SEED', '10,500'))
    app.state.sessions['browser-test-session'] = {'user': {'id': '4', 'username': 'Test admin'}, 'expires': time.time() + 3600,
        'token_expires': time.time() + 3600, 'csrf': 'browser-test-csrf', 'access': 'test', 'refresh': 'test'}

    @app.get('/_test/state')
    async def snapshot():
        return {'spaces': [dict(r) for r in store.list_spaces(1)],
            'guidelines': [dict(r) for r in store.all('SELECT * FROM scene_guidelines')],
            'guild_lore': [dict(r) for r in store.all('SELECT * FROM guild_lore_entries')],
            'lorebooks': [dict(r) for r in store.list_lorebooks(1)],
            'characters': [dict(r) for r in store.all('SELECT id,guild_id,world_id,name,card,archived,avatar IS NOT NULL AS has_static_avatar FROM characters')],
            'lore': [dict(r) for r in store.all('SELECT * FROM lore')],
            'books': [dict(r) for r in store.all('SELECT * FROM lorebook_entries')],
            'presets': [dict(r) for r in store.list_presets(1)], 'active': store.active_preset(1)['id'],
            'active_bundle': store.active_preset(1)['bundle'], 'assets': [dict(r) for r in store.all('SELECT * FROM avatar_assets')],
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

    @app.post('/_test/revoke')
    async def revoke():
        state['admin'] = False
        return {'ok': True}

    @app.post('/_test/change-lore')
    async def change_lore(request: Request):
        value = await request.json()
        row = store.entry_by_key(1, value['key'])
        store.save_entry(1, row['owner_kind'], row['owner_id'], value['content'], row['rule'],
                         ref=row['ref'], expected_revision=row['revision'])
        return {'ok': True}

    @app.post('/_test/delay-permission')
    async def delay_permission():
        state['permission_delay'] = 0.5
        return {'ok': True}

    @app.post('/_test/restore')
    async def restore():
        state['admin'] = True
        return {'ok': True}

    @app.post('/_test/expire')
    async def expire():
        app.state.sessions['browser-test-session']['expires'] = 0
        return {'ok': True}

    @app.post('/_test/stop')
    async def stop():
        server.should_exit = True
        return {'ok': True}

    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, ssl_keyfile=os.environ['LLMCORD_TEST_KEY'], ssl_certfile=os.environ['LLMCORD_TEST_CERT'], access_log=False, log_level='warning'))
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
