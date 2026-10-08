from __future__ import annotations

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
