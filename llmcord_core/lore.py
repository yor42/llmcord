from __future__ import annotations

import json
from dataclasses import dataclass, field

from .store import Store


def estimate_tokens(text: str) -> int:
    # Conservative estimate across English and multibyte text; adapters may trim again.
    return max(1, (len(text.encode("utf-8")) + 2) // 3)


@dataclass(frozen=True)
class LoreMatch:
    id: int
    content: str
    scope_kind: str
    scope_id: int
    reason: str
    entry_key: str = ""
    position: str = "after_char"
    depth: int = 0
    role: str = "system"
    book_name: str = ""
    rule: dict = field(default_factory=dict)


def lore_scopes(store: Store, character_id: int, space_id: int, channel_id: int, parent_id: int | None = None) -> list[tuple[str, int]]:
    character = store.character_by_id(character_id)
    space = store.space_by_id(space_id)
    if not character or not space:
        return []
    scopes = [("guild", character['guild_id']), ("character", character_id), ("space", character["world_id"])]
    if space["kind"] == "hub":
        scopes.append(("space", space_id))
    scopes.append(("channel", parent_id or channel_id))
    if parent_id:
        scopes.append(("thread", channel_id))
    return scopes


def retrieve_lore(store: Store, guild_id: int, scopes: list[tuple[str, int]], query: str, budget_tokens: int = 2000, max_entries: int = 12) -> list[LoreMatch]:
    query_folded = query.casefold()
    candidates: list[tuple[int, int, object, str]] = []
    seen = set()
    for scope_index, (scope_kind, scope_id) in enumerate(scopes):
        for row in store.list_lore(guild_id, scope_kind, scope_id):
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            keys = json.loads(row["keys_json"])
            matched = next((key for key in keys if key and key.casefold() in query_folded), None)
            if row["constant"] or row["pinned"] or matched:
                reason = "pinned" if row["pinned"] else "constant" if row["constant"] else f"keyword: {matched}"
                # Constants first, then the spec's lower insertion order first.
                candidates.append((0 if row["constant"] or row["pinned"] else 1, row["insertion_order"], row, reason))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]["id"]))
    selected, used = [], 0
    for _, _, row, reason in candidates:
        cost = estimate_tokens(row["content"])
        if len(selected) < max_entries and used + cost <= budget_tokens:
            selected.append(LoreMatch(row["id"], row["content"], row["scope_kind"], row["scope_id"], reason))
            used += cost
    return selected
