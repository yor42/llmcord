"""SillyTavern and RisuAI lorebook import and normalization.

The original entry is retained so a later release can interpret fields that have
no Discord prompt equivalent. Unsupported placements are deliberately inactive.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


MAX_BOOK_BYTES = 8 * 1024 * 1024
MAX_ENTRIES = 2000


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ImportedEntry:
    uid: str
    content: str
    rule: dict
    source_hash: str
    local_hash: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ImportedBook:
    raw_json: str
    entries: dict[str, ImportedEntry]
    source_format: str = 'sillytavern'


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()] if isinstance(value, list) else []


def normalize_entry(uid: str, entry: dict, *, source_format: str | None = None) -> ImportedEntry:
    content = entry.get("content", "")
    if not isinstance(content, str):
        raise ValueError(f"Entry {uid} has no text content")
    position = entry.get("position", "after_char")
    # Native ST positions: 0 before card, 1 after card, 2 before examples,
    # 3 after examples, 4 at depth; A/N and outlets need ST-only surfaces.
    positions = {0: "before_char", 1: "after_char", 2: "before_examples",
        3: "after_examples", 4: "in_chat", 5: "author_note_top",
        6: "author_note_bottom", 7: "outlet"}
    if isinstance(position, str) and position.isdecimal():
        position = int(position)
    position = positions.get(position, position)
    unsupported = []
    risu = source_format == 'risu' or any(key in entry for key in ('insertorder', 'secondkey', 'useRegex', 'mode'))
    mode = entry.get('mode', 'normal')
    if risu and mode != 'normal':
        unsupported.append('RisuAI folder marker is preserved as inactive metadata' if mode == 'folder'
                           else f'RisuAI mode {mode} requires unsupported activation behavior')
    if risu and re.search(r'(?m)^\s*@@[A-Za-z_]', content):
        unsupported.append('RisuAI @@ decorators require explicit remapping')
    if risu and any(macro.strip() not in {'char', 'user'} for macro in re.findall(r'\{\{(.*?)\}\}', content, re.DOTALL)):
        unsupported.append('RisuAI content macros require explicit remapping')
    if position not in {"before_char", "after_char", "before_examples",
            "after_examples", "in_chat"}:
        unsupported.append(f"Placement {position} needs a SillyTavern-only prompt surface")
    if entry.get("vectorized"):
        unsupported.append("Vector matching needs SillyTavern's vector extension")
    if entry.get("outletName") or entry.get("outlet_name"):
        unsupported.append("Outlet macros are unavailable")
    if entry.get("automationId") or entry.get("automation_id"):
        unsupported.append("SillyTavern automation IDs are unavailable")
    character_filter = entry.get("characterFilter") or {}
    if not isinstance(character_filter, dict):
        character_filter = {}
    role = entry.get("role", "system")
    role = {0: "system", 1: "user", 2: "assistant"}.get(role, role)
    rule = {
        "keys": _strings(entry.get("key", entry.get("keys", []))),
        "secondary_keys": _strings(entry.get("keysecondary", entry.get("secondary_keys", entry.get('secondkey', [])))),
        "constant": bool(entry.get("constant", entry.get("alwaysActive", False))),
        "enabled": bool(entry.get("enabled", not entry.get("disable", False))) and not (risu and mode == 'folder'),
        "order": int(entry.get("order", entry.get("insertion_order", entry.get('insertorder', 100))) if entry.get("order", entry.get("insertion_order", entry.get('insertorder', 100))) is not None else 100),
        "regex_enabled": entry.get('use_regex', entry.get('useRegex', False if risu else None)),
        "selective": bool(entry.get("selective", False)),
        "selective_logic": int(entry.get("selectiveLogic", entry.get("selective_logic", 0)) or 0),
        "case_sensitive": bool(entry.get("caseSensitive", entry.get("case_sensitive", False))),
        "whole_words": bool(entry.get("matchWholeWords", entry.get("match_whole_words", False))),
        "position": position,
        "role": role if role in {"system", "user", "assistant"} else "system",
        "depth": int(entry.get("depth", 0) or 0),
        "scan_depth": entry.get("scanDepth", entry.get("scan_depth")),
        "probability": max(0, min(100, int(entry.get("probability", 100) or 0))),
        "use_probability": bool(entry.get("useProbability", entry.get("use_probability", False))),
        "group": _strings(entry.get("group", "")),
        "group_weight": max(0, int(entry.get("groupWeight", entry.get("group_weight", 100)) or 0)),
        "group_override": bool(entry.get("groupOverride", False)),
        "use_group_scoring": bool(entry.get("useGroupScoring", False)),
        "exclude_recursion": bool(entry.get("excludeRecursion", False)),
        "prevent_recursion": bool(entry.get("preventRecursion", False)),
        "delay_until_recursion": bool(entry.get("delayUntilRecursion", False)),
        "recursion_level": int(entry.get("recursionLevel", 0) or 0),
        "sticky": max(0, int(entry.get("sticky", 0) or 0)),
        "cooldown": max(0, int(entry.get("cooldown", 0) or 0)),
        "delay": max(0, int(entry.get("delay", 0) or 0)),
        "character_filter_names": _strings(entry.get("characterFilterNames",
            character_filter.get("names", []))),
        "character_filter_tags": _strings(entry.get("characterFilterTags",
            character_filter.get("tags", []))),
        "character_filter_exclude": bool(entry.get("characterFilterExclude",
            character_filter.get("isExclude", False))),
        "match_character_description": bool(entry.get("matchCharacterDescription", False)),
        "match_character_personality": bool(entry.get("matchCharacterPersonality", False)),
        "match_scenario": bool(entry.get("matchScenario", False)),
        "original": entry,
        "unsupported": unsupported,
    }
    return ImportedEntry(uid, content, rule, digest(entry), digest({"content": content,
        "rule": rule}), tuple(unsupported))


def export_entry(content: str, rule: dict) -> dict:
    """Overlay effective fields while retaining fields belonging to other tools."""
    raw = dict(rule.get('original', {}))
    aliases = {'keys': 'key', 'secondary_keys': 'keysecondary', 'order': 'order', 'regex_enabled': 'use_regex',
        'selective_logic': 'selectiveLogic', 'case_sensitive': 'caseSensitive',
        'whole_words': 'matchWholeWords', 'scan_depth': 'scanDepth',
        'use_probability': 'useProbability', 'group_weight': 'groupWeight',
        'group_override': 'groupOverride', 'use_group_scoring': 'useGroupScoring',
        'exclude_recursion': 'excludeRecursion', 'prevent_recursion': 'preventRecursion',
        'delay_until_recursion': 'delayUntilRecursion', 'recursion_level': 'recursionLevel',
        'character_filter_names': 'characterFilterNames', 'character_filter_tags': 'characterFilterTags',
        'character_filter_exclude': 'characterFilterExclude',
        'match_character_description': 'matchCharacterDescription',
        'match_character_personality': 'matchCharacterPersonality', 'match_scenario': 'matchScenario'}
    for key, value in rule.items():
        if key not in {'original', 'unsupported'}:
            raw[aliases.get(key, key)] = value
    raw.update(content=content, enabled=rule.get('enabled', True), disable=not rule.get('enabled', True))
    return raw


def validate_rule(rule: dict, content: str | None = None) -> dict:
    if not isinstance(rule, dict):
        raise ValueError('Rule must be an object')
    default = normalize_entry('validation', {}).rule
    unknown = set(rule) - set(default)
    if unknown:
        raise ValueError('Unknown effective rule fields: ' + ', '.join(sorted(unknown)))
    effective = {**default, **rule}
    for key, value in default.items():
        actual = effective[key]
        if isinstance(value, bool) and not isinstance(actual, bool):
            raise ValueError(f'{key} must be true or false')
        if type(value) is int and type(actual) is not int:
            raise ValueError(f'{key} must be an integer')
        if isinstance(value, list) and key != 'unsupported' and (not isinstance(actual, list) or not all(isinstance(x, str) for x in actual)):
            raise ValueError(f'{key} must be a list of strings')
    for key in ('depth', 'sticky', 'cooldown', 'delay', 'recursion_level', 'group_weight'):
        if effective[key] < 0:
            raise ValueError(f'{key} cannot be negative')
    if effective['scan_depth'] is not None and (type(effective['scan_depth']) is not int or effective['scan_depth'] < 0):
        raise ValueError('Scan depth must be a nonnegative integer or null')
    if effective['regex_enabled'] is not None and type(effective['regex_enabled']) is not bool:
        raise ValueError('Regex enabled must be true, false, or null')
    if not 0 <= effective['probability'] <= 100 or effective['selective_logic'] not in range(4):
        raise ValueError('Invalid probability or secondary matching logic')
    if effective['role'] not in {'system', 'user', 'assistant'}:
        raise ValueError('Invalid message role')
    if not isinstance(effective['original'], dict):
        raise ValueError('Original import must be an object')
    # Recompute warnings from the actual effective settings, not stale warnings.
    normalized = normalize_entry('validation', export_entry(
        effective['original'].get('content', '') if content is None else content, effective)).rule
    effective['unsupported'] = normalized['unsupported']
    return effective


def parse_lorebook(data: bytes) -> ImportedBook:
    if len(data) > MAX_BOOK_BYTES:
        raise ValueError("Lorebook exceeds 8 MiB")
    raw = json.loads(data.decode("utf-8-sig"))
    source_format = 'sillytavern'
    if isinstance(raw, dict) and raw.get('type') == 'risu':
        source_format = 'risu'
        if type(raw.get('ver')) is not int or raw['ver'] != 1:
            raise ValueError('Only RisuAI version-1 lorebook exports are supported')
        if not isinstance(raw.get('data'), list):
            raise ValueError('RisuAI lorebook data must be an entry array')
        source = []
        for index, item in enumerate(raw['data']):
            ident = item.get('uid', item.get('id')) if isinstance(item, dict) else None
            if ident is None or ident == '':
                ident = index
            # Child-mode records can deliberately reference another entry ID.
            if isinstance(item, dict) and item.get('mode') == 'child':
                ident = f'{ident}:child:{index}'
            source.append((str(ident), item))
    elif isinstance(raw, list):
        source = [(str(item.get("uid", index)) if isinstance(item, dict) else str(index), item)
                  for index, item in enumerate(raw)]
    elif isinstance(raw, dict) and isinstance(raw.get("entries"), dict):
        source = [(str(uid), item) for uid, item in raw["entries"].items()]
    elif isinstance(raw, dict) and isinstance(raw.get("entries"), list):
        source = [(str(item.get("id", index)) if isinstance(item, dict) else str(index), item)
                  for index, item in enumerate(raw["entries"])]
    else:
        raise ValueError("Expected an entry array, an object with entries, or a RisuAI version-1 lorebook export")
    if len(source) > MAX_ENTRIES:
        raise ValueError("Lorebook has too many entries")
    entries = {}
    for uid, item in source:
        if not isinstance(item, dict) or uid in entries:
            raise ValueError(f"Invalid or duplicate lorebook entry {uid}")
        entries[uid] = normalize_entry(uid, item, source_format=source_format)
    return ImportedBook(json.dumps(raw, ensure_ascii=False), entries, source_format)
