"""FEAT-47 / D40: "Recent happenings" - what items said in this channel lately, as a short data block for the cast's dialogue prompt."""
import json

from .identity import user_identity
from .items.catalog import get_item

NAME_MAX, TEXT_MAX = 40, 200


def _line(text, cap):
    text = ' '.join(str(text).replace('"', "'").split())
    return text[:cap - 1] + '…' if len(text) > cap else text


def recent_happenings(store, guild_id, channel_id, *, now, before=None, limit=3, window_seconds=1800):
    """The newest ``limit`` item uses in this guild and channel within the window (oldest first); rows with unreadable results are skipped."""
    high = now if before is None else min(now, before)
    out = []
    for row in store.all('SELECT user_id,item_key,result,used_at FROM item_uses WHERE guild_id=? AND channel_id=? AND used_at<=? ORDER BY id DESC LIMIT 50',
                         (guild_id, channel_id, high)):
        if row['used_at'] < now - window_seconds:
            break
        try:
            text = json.loads(row['result']).get('text')
        except (ValueError, AttributeError, RecursionError):
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        out.append({'user_id': row['user_id'], 'item_key': row['item_key'], 'text': text})
        if len(out) >= limit:
            break
    return out[::-1]


def render_happenings(rows, names):
    """One single-line entry per use, result text only; names and text are quoted data (quotes and line breaks removed)."""
    lines = []
    for row in rows:
        try:
            item = get_item(row['item_key']).name
        except KeyError:
            item = row['item_key']
        who = user_identity(row['user_id'], names.get(row['user_id']) or '')
        lines.append(f'{who} used the {_line(item, NAME_MAX)}: "{_line(row["text"], TEXT_MAX)}"')
    return '\n'.join(lines) or None
