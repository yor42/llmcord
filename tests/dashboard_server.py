"""Isolated HTTPS browser-test fixture. Never used by production entry points."""
import json
import os
import time

import httpx
import uvicorn
from fastapi import Request

from llmcord_core.web import create_app


def main():
    port = int(os.environ['LLMCORD_TEST_PORT'])
    state = {'admin': True, 'guild_checks': 0}
    async def discord_api(request):
        if request.url.path.endswith('/users/@me/guilds'):
            state['guild_checks'] += 1
            if state['guild_checks'] == 1:
                return httpx.Response(429, json={'retry_after': 0.05})
            return httpx.Response(200, json=[{'id': '1', 'name': 'Test server', 'permissions': '8' if state['admin'] else '0'}],
                                  headers={'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset-After': '0.05'})
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


if __name__ == '__main__':
    main()
