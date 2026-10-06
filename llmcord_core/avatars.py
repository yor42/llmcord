"""Bounded avatar uploads, Discord publication, and emotion stream parsing."""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from io import BytesIO
from urllib.parse import urlsplit

from PIL import Image, ImageOps

DEFAULT_SLOTS = ('neutral', 'happy', 'sad', 'angry', 'surprised', 'embarrassed')
MAX_AVATAR_BYTES = 8 * 1024 * 1024


def normalize_avatar(data):
    if len(data) > MAX_AVATAR_BYTES:
        raise ValueError('Avatar exceeds 8 MiB')
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in {'PNG', 'JPEG', 'WEBP'} or getattr(image, 'n_frames', 1) != 1:
                raise ValueError('Upload a static PNG, JPEG or WebP image')
            if image.width * image.height > 16_000_000:
                raise ValueError('Avatar exceeds 16 megapixels')
            image = ImageOps.exif_transpose(image)
            image.thumbnail((256, 256))
            output = BytesIO()
            image.convert('RGBA').save(output, 'PNG', optimize=True)
            return output.getvalue()
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError('Invalid avatar image') from error


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
        result = await self.http.get(f'https://discord.com/api/v10/guilds/{guild_id}/channels', headers={'Authorization': 'Bot ' + self.bot_token})
        if result.status_code != 200:
            raise ValueError('Discord channels are unavailable')
        return result.json()

    async def configure(self, guild_id, channel_id):
        channels = await self.channels(guild_id)
        channel = next((c for c in channels if int(c['id']) == channel_id and c['type'] == 0), None)
        if not channel:
            raise ValueError('Choose an asset text channel in this server')
        everyone = next((x for x in channel.get('permission_overwrites', []) if str(x['id']) == str(guild_id)), None)
        if not everyone or not int(everyone['deny']) & (1 << 10):
            raise ValueError('The asset channel must explicitly deny View Channel to @everyone')
        self.store.execute('INSERT INTO guild_settings(guild_id,asset_channel_id) VALUES(?,?) ON CONFLICT(guild_id) DO UPDATE SET asset_channel_id=excluded.asset_channel_id', (guild_id, channel_id))

    async def publish(self, guild_id, character_id, slot_key, repair=False):
        slot = self.store.avatar_slot(guild_id, character_id, slot_key)
        if not slot['image']:
            raise ValueError('Upload an image for this slot first')
        settings = self.store.one('SELECT asset_channel_id FROM guild_settings WHERE guild_id=?', (guild_id,))
        if not settings or not settings['asset_channel_id']:
            raise ValueError('Choose an avatar asset channel first')
        image_hash = hashlib.sha256(slot['image']).hexdigest()
        asset = self.store.avatar_asset(guild_id, character_id, slot_key)
        if asset and asset['image_hash'] == image_hash and not repair:
            return asset['url']
        channel_id = settings['asset_channel_id']
        response = await self.http.post(f'https://discord.com/api/v10/channels/{channel_id}/messages', headers={'Authorization': 'Bot ' + self.bot_token},
            data={'payload_json': json.dumps({'content': f'Character {character_id} / {slot_key}', 'allowed_mentions': {'parse': []}})},
            files={'files[0]': (f'{image_hash}.png', slot['image'], 'image/png')})
        if response.status_code >= 400:
            raise ValueError('Avatar publication failed; check asset-channel permissions')
        message = response.json()
        attachment = message['attachments'][0]
        url = urlsplit(attachment['url'])
        if url.scheme != 'https' or url.hostname not in {'cdn.discordapp.com', 'media.discordapp.net'}:
            raise ValueError('Discord returned an unexpected asset URL')
        canonical = f'{url.scheme}://{url.netloc}{url.path}'
        self.store.execute('INSERT INTO avatar_assets(guild_id,character_id,slot_key,image_hash,channel_id,message_id,url,created_at) VALUES(?,?,?,?,?,?,?,?)',
            (guild_id, character_id, slot_key, image_hash, channel_id, int(message['id']), canonical, time.time()))
        return canonical
