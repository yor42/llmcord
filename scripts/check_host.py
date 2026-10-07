"""Check deployment configuration; --live tests APIs without posting messages."""
from __future__ import annotations

import argparse
import asyncio
import os
from urllib.parse import urlparse

from llmcord_core.config import load_settings, resolve_database_path


def require(name):
    value = os.environ.get(name, '')
    if not value or value.startswith('replace-me'):
        raise ValueError(f'Set {name} in the service environment')
    return value


def check_web():
    for name in ('DISCORD_BOT_TOKEN', 'DISCORD_CLIENT_ID', 'DISCORD_CLIENT_SECRET'):
        require(name)
    parsed = urlparse(require('WEB_BASE_URL'))
    if parsed.scheme != 'https' or not parsed.hostname or parsed.path not in ('', '/') or parsed.query or parsed.fragment:
        raise ValueError('WEB_BASE_URL must be an HTTPS origin without a path')
    if os.environ.get('WEB_HOST', '127.0.0.1') not in ('127.0.0.1', '::1'):
        raise ValueError('Native dashboard hosting must bind to loopback')
    print('PASS: dashboard environment; register WEB_BASE_URL + /auth/callback in Discord')


async def live_check(settings):
    import httpx
    from llmcord_core.models import ModelGateway, TurnMessage, DIRECTOR_SCHEMA, MEMORY_SCHEMA

    async with httpx.AsyncClient(base_url='https://discord.com/api/v10/', timeout=20,
                                 headers={'Authorization': f'Bot {settings.token}'}) as http:
        async def get(path):
            response = await http.get(path)
            if response.status_code != 200:
                raise ValueError(f'Discord {path}: HTTP {response.status_code}')
            return response.json()
        identity = await get('users/@me')
        app = await get('oauth2/applications/@me')
        client_id = os.environ.get('DISCORD_CLIENT_ID')
        if client_id and str(app['id']) != client_id:
            raise ValueError('OAuth client ID and bot token belong to different applications')
        # Discord application flags for full or limited message-content access.
        if not int(app.get('flags', 0)) & ((1 << 18) | (1 << 19)):
            raise ValueError('Enable Message Content Intent in the Discord developer portal')
        print(f"PASS: Discord bot authenticated ({identity['username']}); message-content intent enabled")
        if settings.development_guild_id:
            await get(f'guilds/{settings.development_guild_id}')
            print('PASS: bot is a member of the private test server')
        else:
            raise ValueError('Set DISCORD_GUILD_ID before live private-server testing')

    models = ModelGateway(settings)
    try:
        for role, schema, prompt in (
            ('director', DIRECTOR_SCHEMA, 'Return speakers [1].'),
            ('memory', MEMORY_SCHEMA, 'No facts to remember. Return empty arrays for all fact lists.'),
        ):
            await models.structured(role, 'Follow the requested output schema.',
                                    [TurnMessage('user', prompt)], f'check_{role}', schema)
            print(f'PASS: {role} model returns validated JSON')
        content = ''.join([part async for part in models.stream_text(
            'dialogue', 'Reply briefly.', [TurnMessage('user', 'Say hello in one sentence.')])])
        if not content.strip():
            raise ValueError('Dialogue stream returned no visible text; check output budget/model access')
        print('PASS: dialogue model streams visible text')
    finally:
        await models.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Makes small billable model calls and read-only Discord requests')
    parser.add_argument('--require-guild', action='store_true')
    parser.add_argument('--web-only', action='store_true')
    parser.add_argument('--web', action='store_true')
    args = parser.parse_args()
    try:
        if args.web_only:
            check_web()
            return
        require('DISCORD_BOT_TOKEN')
        settings = load_settings()
        if args.require_guild and not settings.development_guild_id:
            raise ValueError('Set DISCORD_GUILD_ID for fast private-server command registration')
        for role in ('dialogue', 'director', 'memory'):
            profile = settings.profile(role)
            if profile.api_key_env:
                require(profile.api_key_env)
            print(f'PASS: {role}: {profile.model} ({profile.provider})')
        database = settings.database_path
        database.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(database.parent, os.W_OK) or (database.exists() and not os.access(database, os.W_OK)):
            raise ValueError('SQLite database and its parent directory must be writable')
        if resolve_database_path().resolve() != database.resolve():
            raise ValueError('Bot and dashboard database paths differ; set LLMCORD_DATABASE_PATH')
        print('PASS: shared SQLite location is writable')
        if args.web:
            check_web()
        if args.live:
            asyncio.run(live_check(settings))
        else:
            print('API access unverified; use --live after supplying credentials')
    except Exception as error:
        # Provider exception bodies can contain prompts or URLs; keep diagnostics bounded.
        detail = str(error) if isinstance(error, (ValueError, FileNotFoundError)) else type(error).__name__
        raise SystemExit(f'FAIL: {detail}') from None


if __name__ == '__main__':
    main()
