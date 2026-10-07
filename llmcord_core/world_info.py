"""Evaluate Discord-applicable SillyTavern World Info rules on one branch."""
from __future__ import annotations

import atexit
import hashlib
import json
import random
import re

from .lore import LoreMatch, estimate_tokens


def _seed(turn_id: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{turn_id}:{key}".encode()).digest()[:8], "big")


# One isolate per process, used only from the event-loop thread (evaluation is synchronous), so no locking.
# Never evaluate user-supplied JS here (only RegExp construction from data): the shared context keeps RegExp.lastMatch etc.
_ISOLATE: list = []
_JS_HELPER = """var __wiCache = new Map();
function __wiTest(p, f, t) {
  var k = f + "/" + p, r = __wiCache.get(k);
  if (r === undefined) {
    if (__wiCache.size >= 512) __wiCache.clear();
    r = new RegExp(p, f);
    __wiCache.set(k, r);
  }
  r.lastIndex = 0;
  return r.test(t) === true;
}"""


def _discard_isolate() -> None:
    while _ISOLATE:
        try:
            _ISOLATE.pop().close()
        except Exception:
            pass


atexit.register(_discard_isolate)


def _regex_test(pattern: str, flags: str, text: str) -> bool:
    try:
        import py_mini_racer
    except Exception:
        return False
    try:
        if not _ISOLATE:
            _ISOLATE.append(py_mini_racer.MiniRacer())
            try:
                _ISOLATE[0].eval(_JS_HELPER)
            except Exception:
                _discard_isolate()
                raise
        return bool(_ISOLATE[0].eval(
            f"__wiTest({json.dumps(pattern)}, {json.dumps(flags)}, {json.dumps(text)})", timeout_sec=0.05))
    except Exception as exc:
        # A bad pattern is a plain JS SyntaxError and leaves the isolate usable; anything else discards it.
        if not isinstance(exc, py_mini_racer.JSEvalException) or isinstance(
                exc, (py_mini_racer.JSTimeoutException, py_mini_racer.JSOOMException)):
            _discard_isolate()
        return False


def _key_matches(key: str, text: str, rule: dict) -> bool:
    if not key:
        return False
    regex_key = key.startswith("/") and key.rfind("/") > 0
    if rule.get('regex_enabled') is True and not regex_key:
        return False
    if rule.get('regex_enabled') is not False and regex_key:
        end = key.rfind("/")
        pattern, flags = key[1:end], key[end + 1:]
        if len(pattern) > 500 or not set(flags) <= set("gimsuyd"):
            return False
        return _regex_test(pattern, flags, text)
    if not rule.get("case_sensitive"):
        key, text = key.casefold(), text.casefold()
    if rule.get("whole_words") and " " not in key:
        return re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", text) is not None
    return key in text


def _secondary_matches(rule: dict, text: str) -> bool:
    keys = rule.get("secondary_keys", [])
    if not rule.get("selective") or not keys:
        return True
    hits = [_key_matches(key, text, rule) for key in keys]
    logic = rule.get("selective_logic", 0)
    return (any(hits) if logic == 0 else all(hits) if logic == 1 else
            not any(hits) if logic == 2 else not all(hits))


def collect_entries(store, guild_id: int, scopes: list[tuple[str, int]], character) -> list[LoreMatch]:
    rows = []
    for kind, scope_id in scopes:
        for row in store.list_lore(guild_id, kind, scope_id):
            rule = json.loads(row["rule_json"] or "{}")
            if not rule:
                rule = {"keys": json.loads(row["keys_json"]), "constant": bool(row["constant"]),
                    "enabled": bool(row["enabled"]), "order": row["insertion_order"]}
            if row["pinned"]:
                rule = {**rule, "constant": True}
            rows.append(LoreMatch(row["id"], row["content"], kind, scope_id, "",
                row['entry_key'] or f"lore:{row['id']}", rule.get("position", "after_char"),
                int(rule.get("depth", 0)), str(rule.get("role", "system")), "", rule))
    if character:
        world_id = character["world_id"]
        current_space = next((value for kind, value in reversed(scopes) if kind == "space"), world_id)
        channel_id = next((value for kind, value in scopes if kind == "channel"), 0)
        for row in store.active_lorebook_entries(guild_id, world_id, current_space, channel_id):
            rule = json.loads(row["rule_json"])
            if row['pinned']:
                rule = {**rule, 'constant': True}
            rows.append(LoreMatch(row["id"], row["content"], "lorebook", row["book_id"], "",
                row['entry_key'] or f"book:{row['book_id']}:{row['uid']}", rule.get("position", "after_char"),
                int(rule.get("depth", 0)), str(rule.get("role", "system")), row["book_name"], rule))
    return rows


def evaluate(store, guild_id: int, scopes: list[tuple[str, int]], query: str,
             budget_tokens: int, *, character=None, messages: list[str] | None = None,
             branch_ids: list[int] | None = None, response_id: int = 0,
             max_entries: int = 12, max_recursion_steps: int = 4) -> list[LoreMatch]:
    entries = collect_entries(store, guild_id, scopes, character)
    messages = messages or [query]
    branch_ids = branch_ids or []
    positions = {ident: index for index, ident in enumerate(branch_ids)}
    previous: dict[str, int] = {}
    for event in store.branch_lore_activations(branch_ids):
        previous[event["entry_key"]] = max(previous.get(event["entry_key"], -1),
            positions[event["node_id"]])
    chosen: dict[str, LoreMatch] = {}
    scan_text = "\n".join(messages)
    recursive_text = ""
    groups_taken = set()
    card = json.loads(character["card"]) if character else {}
    for level in range(max_recursion_steps + 1):
        matches = []
        for item in entries:
            rule = item.rule
            if item.entry_key in chosen or not rule.get("enabled", True) or rule.get("unsupported"):
                continue
            if level and rule.get("exclude_recursion"):
                continue
            if rule.get("delay_until_recursion") and level < max(1, rule.get("recursion_level", 0)):
                continue
            if rule.get("delay", 0) > len(branch_ids):
                continue
            if character and (rule.get("character_filter_names") or rule.get("character_filter_tags")):
                names = {name.casefold() for name in rule.get("character_filter_names", [])}
                tags = {tag.casefold() for tag in rule.get("character_filter_tags", [])}
                card_tags = {str(tag).casefold() for tag in card.get("tags", [])}
                found = character["name"].casefold() in names or bool(tags & card_tags)
                if found == bool(rule.get("character_filter_exclude")):
                    continue
            depth = rule.get("scan_depth")
            base_text = "\n".join(messages[-int(depth):]) if depth is not None and int(depth) > 0 else scan_text
            text = base_text + ("\n" + recursive_text if recursive_text else "")
            if character:
                for flag, field in (("match_character_description", "description"),
                                    ("match_character_personality", "personality"),
                                    ("match_scenario", "scenario")):
                    if rule.get(flag):
                        text += "\n" + str(card.get(field, ""))
            matched = [key for key in rule.get("keys", []) if _key_matches(key, text, rule)]
            last = previous.get(item.entry_key)
            age = len(branch_ids) - last if last is not None else 10**9
            sticky = last is not None and age <= rule.get("sticky", 0)
            cooldown = last is not None and not sticky and age <= (
                rule.get("sticky", 0) + rule.get("cooldown", 0))
            if cooldown or (not rule.get('constant') and not _secondary_matches(rule, text)) or not (rule.get("constant") or matched or sticky):
                continue
            probability = rule.get("probability", 100) if rule.get("use_probability") else 100
            if not sticky and random.Random(_seed(response_id, item.entry_key)).randrange(100) >= probability:
                continue
            reason = "sticky" if sticky else "constant" if rule.get("constant") else f"keyword: {matched[0]}"
            matches.append(LoreMatch(item.id, item.content, item.scope_kind, item.scope_id,
                reason, item.entry_key, item.position, item.depth, item.role, item.book_name, rule))
        matches.sort(key=lambda item: (-int(item.rule.get("constant", False)),
            -int(item.rule.get("order", 100)), item.entry_key))
        added = []
        for item in matches:
            groups = set(item.rule.get("group", []))
            if groups & groups_taken:
                continue
            if groups and not item.rule.get("group_override"):
                rivals = [other for other in matches if groups & set(other.rule.get("group", []))]
                if len(rivals) > 1:
                    if item.rule.get("use_group_scoring"):
                        winner = max(rivals, key=lambda other: (
                            sum(_key_matches(key, scan_text, other.rule)
                                for key in other.rule.get("keys", [])),
                            int(other.rule.get("group_weight", 100)), other.entry_key))
                    else:
                        rng = random.Random(_seed(response_id, ":".join(sorted(groups))))
                        weights = [max(0, int(other.rule.get("group_weight", 100))) for other in rivals]
                        winner = rng.choices(rivals, weights=weights if sum(weights) else None)[0]
                    if winner.entry_key != item.entry_key:
                        continue
            chosen[item.entry_key] = item
            groups_taken.update(groups)
            if not item.rule.get("prevent_recursion"):
                added.append(item.content)
        if not added:
            break
        recursive_text += "\n" + "\n".join(added)
    ranked = sorted(chosen.values(), key=lambda item: (
        0 if item.rule.get("constant") else 1, -int(item.rule.get("order", 100)), item.entry_key))
    selected, used = [], 0
    for item in ranked:
        cost = estimate_tokens(item.content)
        if len(selected) < max_entries and used + cost <= budget_tokens:
            selected.append(item)
            used += cost
    return sorted(selected, key=lambda item: (int(item.rule.get("order", 100)), item.entry_key))
