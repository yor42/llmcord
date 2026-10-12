from __future__ import annotations

import base64
import json
from dataclasses import dataclass

from .i18n import UserError, codes
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
        raise UserError(codes.CARD_NOT_PNG)
    offset = 8
    found: dict[str, dict] = {}
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset:offset + 4], "big")
        kind = data[offset + 4:offset + 8]
        end = offset + 12 + length
        if length > MAX_CARD_BYTES or end > len(data):
            raise UserError(codes.CARD_PNG_CORRUPT)
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
    raise UserError(codes.CARD_PNG_NO_DATA)


def parse_card(filename: str, data: bytes) -> ParsedCard:
    if len(data) > MAX_CARD_BYTES:
        raise UserError(codes.CARD_TOO_BIG)
    filename = filename.lower()
    if filename.endswith(".png"):
        raw = _embedded_png_card(data)
        from .avatars import normalize_avatar
        avatar = normalize_avatar(data)
    elif filename.endswith(".json"):
        raw, avatar = json.loads(data.decode("utf-8")), None
    else:
        raise UserError(codes.CARD_UPLOAD_KIND)
    if not isinstance(raw, dict):
        raise UserError(codes.CARD_NOT_OBJECT)
    spec = raw.get("spec")
    if spec and spec not in {"chara_card_v2", "chara_card_v3"}:
        raise UserError(codes.CARD_SPEC, spec=spec)
    card = raw.get("data", raw)
    if not isinstance(card, dict) or not isinstance(card.get("name"), str) or not card["name"].strip():
        raise UserError(codes.CARD_NO_NAME)
    name = card["name"].strip()
    if len(name) > 80:
        raise UserError(codes.CARD_NAME_TOO_LONG)
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
            raise UserError(codes.CARD_LORE_DUPLICATE)
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
