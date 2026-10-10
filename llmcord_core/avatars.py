"""Bounded avatar uploads, Discord publication, and emotion stream parsing."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from io import BytesIO
from urllib.parse import urlsplit

from PIL import Image, ImageOps

from .discord_api import DISCORD_API

log = logging.getLogger(__name__)
DEFAULT_SLOTS = ('neutral', 'happy', 'sad', 'angry', 'surprised', 'embarrassed')
MAX_AVATAR_BYTES = 8 * 1024 * 1024


def avatar_version(data):
    return hashlib.sha256(data).hexdigest()[:12]


def normalize_avatar(data):
    if len(data) > MAX_AVATAR_BYTES:
        raise ValueError('Avatar exceeds 8 MiB. Upload a smaller image.')
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in {'PNG', 'JPEG', 'WEBP'} or getattr(image, 'n_frames', 1) != 1:
                raise ValueError('Upload a static PNG, JPEG or WebP image.')
            if image.width * image.height > 16_000_000:
                raise ValueError('Avatar exceeds 16 megapixels. Upload a smaller image.')
            image = ImageOps.exif_transpose(image)
            image.thumbnail((256, 256))
            output = BytesIO()
            image.convert('RGBA').save(output, 'PNG', optimize=True)
            return output.getvalue()
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError('That avatar image is invalid. Upload a PNG, JPEG or WebP image.') from error


@dataclass(frozen=True)
class DialogueEvent:
    emotion: str | None = None
    text: str = ''


async def emotion_stream(stream, allowed):
    """Do not create a message until its avatar has been resolved."""
    buffer, started, discarding, strip_body_start = '', False, False, False
    async for delta in stream:
        if started:
            if discarding:
                if '\n' not in delta:
                    continue
                delta = delta.partition('\n')[2]
                discarding = False
            if strip_body_start:
                delta = delta.lstrip('\r\n')
                if not delta:
                    continue
                strip_body_start = False
            yield DialogueEvent(text=delta)
            continue
        buffer += delta
        candidate = buffer.lstrip()
        prefix = '<emotion>'
        if prefix.startswith(candidate) and len(candidate) < len(prefix):
            continue
        if candidate.startswith(prefix):
            end = candidate.find('</emotion>')
            if end < 0 and len(candidate) <= 256 and '\n' not in candidate:
                continue
            if end >= 0 and end <= 256:
                label = candidate[len(prefix):end].strip()
                body = candidate[end + len('</emotion>'):].lstrip('\r\n')
            else:
                label = 'neutral'
                body = candidate.partition('\n')[2]
                discarding = '\n' not in candidate
            yield DialogueEvent(emotion=label if label in allowed else 'neutral')
            strip_body_start = not body
            if body:
                yield DialogueEvent(text=body)
        elif candidate.startswith('<emotion'):
            if '\n' not in candidate and len(candidate) <= 256:
                continue
            yield DialogueEvent(emotion='neutral')
            discarding = '\n' not in candidate
            body = candidate.partition('\n')[2]
            if body:
                yield DialogueEvent(text=body)
        else:
            yield DialogueEvent(emotion='neutral')
            if buffer:
                yield DialogueEvent(text=buffer)
        started = True
    if not started:
        yield DialogueEvent(emotion='neutral')
        if buffer and not buffer.lstrip().startswith('<emotion'):
            yield DialogueEvent(text=buffer)


class AvatarPublisher:
    def __init__(self, store, http, bot_token):
        self.store, self.http, self.bot_token = store, http, bot_token

    async def channels(self, guild_id):
        result = await self.http.get(f'{DISCORD_API}/guilds/{guild_id}/channels', headers={'Authorization': 'Bot ' + self.bot_token})
        if result.status_code != 200:
            raise ValueError('Discord channels are unavailable. Try again in a moment.')
        return result.json()

    async def active_threads(self, guild_id):
        result = await self.http.get(f'{DISCORD_API}/guilds/{guild_id}/threads/active', headers={'Authorization': 'Bot ' + self.bot_token})
        if result.status_code != 200:
            raise ValueError('Discord threads are unavailable. Try again in a moment.')
        return result.json().get('threads', [])

    MEMBER_TTL, MEMBER_CACHE_MAX = 600, 1000

    async def member_names(self, guild_id, user_ids):
        """Display names for the given members, one bot-token lookup each (never the privileged member list).

        Returns (names, failed): a member who left (404) is simply absent; any other failure sets failed. Answers, including
        "left", are cached for 10 minutes (bounded); after a rate limit no further request is made in this call."""
        cache = self.__dict__.setdefault('_members', {})
        now, gate, limited = time.monotonic(), asyncio.Semaphore(4), []
        async def one(user_id):
            hit = cache.get((guild_id, user_id))
            if hit and hit[0] > now:
                return hit[1]
            async with gate:
                if limited:
                    raise ValueError('Discord members are rate limited.')
                result = await self.http.get(f'{DISCORD_API}/guilds/{guild_id}/members/{user_id}', headers={'Authorization': 'Bot ' + self.bot_token})
            if result.status_code == 429 or (result.headers.get('X-RateLimit-Remaining') == '0' and result.headers.get('Retry-After')):
                if not limited:
                    limited.append(True)
                    log.warning('Discord rate-limited the member name lookup (guild %s)', guild_id)
            if result.status_code == 404:
                name = None
            elif result.status_code != 200:
                raise ValueError('Discord members are unavailable.')
            else:
                body = result.json()
                user = body.get('user') or {}
                name = body.get('nick') or user.get('global_name') or user.get('username') or None
            if len(cache) >= self.MEMBER_CACHE_MAX:
                for key in [k for k, v in cache.items() if v[0] <= now] or [next(iter(cache))]:
                    del cache[key]
            cache[(guild_id, user_id)] = (now + self.MEMBER_TTL, name)
            return name
        ids = list(dict.fromkeys(user_ids))
        found = await asyncio.gather(*(one(i) for i in ids), return_exceptions=True)
        failed = any(isinstance(r, BaseException) for r in found)
        return {i: r for i, r in zip(ids, found) if isinstance(r, str)}, failed

    async def configure(self, guild_id, channel_id):
        channels = await self.channels(guild_id)
        channel = next((c for c in channels if int(c['id']) == channel_id and c['type'] == 0), None)
        if not channel:
            raise ValueError('Choose an asset text channel in this server.')
        everyone = next((x for x in channel.get('permission_overwrites', []) if str(x['id']) == str(guild_id)), None)
        if not everyone or not int(everyone['deny']) & (1 << 10):
            raise ValueError('The asset channel must deny View Channel to @everyone. Change its permissions in Discord, then choose it again.')
        self.store.execute('INSERT INTO guild_settings(guild_id,asset_channel_id) VALUES(?,?) ON CONFLICT(guild_id) DO UPDATE SET asset_channel_id=excluded.asset_channel_id', (guild_id, channel_id))

    async def publish(self, guild_id, character_id, slot_key, repair=False):
        slot = self.store.avatar_slot(guild_id, character_id, slot_key)
        if not slot['image']:
            raise ValueError('Upload an image for this slot first.')
        settings = self.store.one('SELECT asset_channel_id FROM guild_settings WHERE guild_id=?', (guild_id,))
        if not settings or not settings['asset_channel_id']:
            raise ValueError('Choose an avatar asset channel first.')
        image_hash = slot['image_hash']
        asset = self.store.avatar_asset(guild_id, character_id, slot_key, image_hash)
        if asset and asset['image_hash'] == image_hash and not repair:
            return asset['url']
        channel_id = settings['asset_channel_id']
        response = await self.http.post(f'{DISCORD_API}/channels/{channel_id}/messages', headers={'Authorization': 'Bot ' + self.bot_token},
            data={'payload_json': json.dumps({'content': f'Character {character_id} / {slot_key}', 'allowed_mentions': {'parse': []}})},
            files={'files[0]': (f'{image_hash}.png', slot['image'], 'image/png')})
        if response.status_code >= 400:
            raise ValueError('Avatar publication failed. Check that the bot can send messages and attach files in the asset channel.')
        message = response.json()
        attachment = message['attachments'][0]
        url = urlsplit(attachment['url'])
        if url.scheme != 'https' or url.hostname not in {'cdn.discordapp.com', 'media.discordapp.net'}:
            raise ValueError('Discord returned an unexpected asset URL. Try again.')
        canonical = f'{url.scheme}://{url.netloc}{url.path}'
        self.store.execute('INSERT INTO avatar_assets(guild_id,character_id,slot_key,image_hash,channel_id,message_id,url,created_at) VALUES(?,?,?,?,?,?,?,?)',
            (guild_id, character_id, slot_key, image_hash, channel_id, int(message['id']), canonical, time.time()))
        return canonical
