from __future__ import annotations

import base64
import json
from dataclasses import dataclass

from .lorebooks import normalize_entry


MAX_CARD_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class ParsedCard:
    name: str
    data: dict
    entries: list[dict]
    avatar: bytes | None


def _embedded_png_card(data: bytes) -> dict:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Not a PNG file. Upload a V2/V3 PNG or JSON card.")
    offset = 8
    found: dict[str, dict] = {}
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset:offset + 4], "big")
        kind = data[offset + 4:offset + 8]
        end = offset + 12 + length
        if length > MAX_CARD_BYTES or end > len(data):
            raise ValueError("The PNG card is corrupt. Export it again and upload the new copy.")
        if kind == b"tEXt":
            chunk = data[offset + 8:offset + 8 + length]
            keyword, sep, value = chunk.partition(b"\0")
            if sep and keyword in {b"ccv3", b"chara"}:
                found[keyword.decode("ascii")] = json.loads(base64.b64decode(value, validate=True))
        offset = end
        if kind == b"IEND":
            break
    if "ccv3" in found:
        return found["ccv3"]
    if "chara" in found:
        return found["chara"]
    raise ValueError("No character data found in that PNG. Upload a V2/V3 PNG or JSON card.")


def parse_card(filename: str, data: bytes) -> ParsedCard:
    if len(data) > MAX_CARD_BYTES:
        raise ValueError("Card exceeds 8 MiB. Upload a smaller file.")
    filename = filename.lower()
    if filename.endswith(".png"):
        raw = _embedded_png_card(data)
        from .avatars import normalize_avatar
        avatar = normalize_avatar(data)
    elif filename.endswith(".json"):
        raw, avatar = json.loads(data.decode("utf-8")), None
    else:
        raise ValueError("Upload a V2/V3 PNG or JSON card.")
    if not isinstance(raw, dict):
        raise ValueError("Card must contain a JSON object. Upload a V2/V3 PNG or JSON card.")
    spec = raw.get("spec")
    if spec and spec not in {"chara_card_v2", "chara_card_v3"}:
        raise ValueError(f"Unsupported character card spec: {spec}. Upload a V2/V3 card.")
    card = raw.get("data", raw)
    if not isinstance(card, dict) or not isinstance(card.get("name"), str) or not card["name"].strip():
        raise ValueError("Card has no character name. Add one and upload it again.")
    name = card["name"].strip()
    if len(name) > 80:
        raise ValueError("Character name exceeds 80 characters. Shorten it and upload again.")
    book = card.get("character_book") or {}
    if not isinstance(book, dict):
        book = {}
    entries = []
    for index, entry in enumerate(book.get("entries", [])):
        if not isinstance(entry, dict) or not isinstance(entry.get("content"), str):
            continue
        keys = entry.get("keys", entry.get("key", []))
        if isinstance(keys, str):
            keys = [part.strip() for part in keys.split(",")]
        if not isinstance(keys, list):
            keys = []
        normalized = normalize_entry(str(entry.get("id", entry.get('uid', index))), entry)
        if any(old['uid'] == normalized.uid for old in entries):
            raise ValueError('The card repeats a lore entry ID. Remove the duplicate and upload again.')
        entries.append({
            "uid": normalized.uid,
            "content": entry["content"],
            "keys": [str(key).strip() for key in keys if str(key).strip()],
            "constant": bool(entry.get("constant", entry.get("alwaysActive", False))),
            "enabled": bool(entry.get("enabled", True)),
            "insertion_order": normalized.rule['order'],
            "rule": normalized.rule,
        })
    return ParsedCard(name=name, data=card, entries=entries, avatar=avatar)
