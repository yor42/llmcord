# Architecture

llmcord keeps prompt state in SQLite and assembles a fresh prompt for each character response. Bot and dashboard are separate processes that share the database. The model provider does not hold the canonical conversation state.

```mermaid
flowchart LR
  Discord[Discord channels and threads] --> Router[Discord router]
  Browser[Admin browser] --> Serve[Tailscale Serve] --> Web[OAuth dashboard]
  Router --> Engine[Scene engine]
  Web --> DB[(SQLite)]
  Engine <--> DB
  Engine --> Director[Director model call]
  Engine --> Dialogue[Sequential character calls]
  Dialogue --> Hooks[Character webhooks] --> Discord
  Engine --> Memory[Summary and fact extraction]
  Memory --> DB
```

## Modules

| File | Responsibility |
| --- | --- |
| [`llmcord.py`](../llmcord.py) | Bot entry point. |
| [`web_main.py`](../web_main.py) | Web entry point. |
| [`migrate.py`](../migrate.py) | Migration entry point used by Compose. |
| [`llmcord_core/config.py`](../llmcord_core/config.py) | YAML settings and environment-backed secrets. |
| [`llmcord_core/discord_bot.py`](../llmcord_core/discord_bot.py) | Discord events, slash commands, webhooks, and message delivery. |
| [`llmcord_core/engine.py`](../llmcord_core/engine.py) | Director, prompt assembly, branches, and memory extraction. |
| [`llmcord_core/models.py`](../llmcord_core/models.py) | OpenAI, Anthropic, and OpenAI-compatible adapters. |
| [`llmcord_core/store.py`](../llmcord_core/store.py) | SQLite schema, migrations, and scoped data access. |
| [`llmcord_core/lore.py`](../llmcord_core/lore.py) | Lore scope calculation and token estimates. |
| [`llmcord_core/world_info.py`](../llmcord_core/world_info.py) | Runtime World Info evaluation. |
| [`llmcord_core/lorebooks.py`](../llmcord_core/lorebooks.py) | Standalone lorebook parsing and rule normalization. |
| [`llmcord_core/cards.py`](../llmcord_core/cards.py) | V2/V3 JSON and PNG character-card parsing. |
| [`llmcord_core/web.py`](../llmcord_core/web.py) | Discord OAuth, administrator checks, CSRF, and dashboard actions. |
| [`llmcord_core/templates/`](../llmcord_core/templates/) | Server-rendered dashboard pages. |

## Scene flow

1. Discord routing accepts an explicit invitation or considers an ambient turn in a bound channel. Threads resolve their space through the parent channel.
2. The engine filters the cast to eligible characters. A structured director result chooses up to the configured three speakers. Ambient turns may select none.
3. The human message is stored as a node. Each speaker gets its own card, permitted lore, opted-in personal facts, encounters, branch summary, and visible message history. A hub guest's prompt does not include another guest's private world lore.
4. The dialogue adapter streams one line. A character webhook posts it; the bot saves its Discord message ID, parent ID, source trace, and World Info activations. The next speaker can react to earlier lines from the same turn.
5. The memory adapter extracts facts and a branch summary. Repeated shared facts become local durable lore after evidence from separate scene roots. Personal facts are written only with consent.

SQLite runs in WAL mode with a busy timeout. Schema version is held in `PRAGMA user_version`; migration backs up an existing older database before upgrading. Compose orders the migration service before bot and web startup.

## Rewinds and retention

Conversation nodes store `parent_id` and `root_id`. A reply to an earlier saved character line walks that line's ancestors. Later sibling nodes are excluded. Source message IDs on local facts and memories are checked against the selected branch where relevant. World Info timing reads saved activations from branch nodes, so a rewind follows its own activation history. `/context` reads the response trace attached to a saved webhook message.

The daily cleanup removes old conversation nodes, summaries tied to them, response traces, and World Info activation rows. It does not remove durable lore or opted-in personal facts. See [Lore, imports, and memory](lore-and-memory.md) for the user-facing controls.

## Current scope

The code targets casual text skits with bounded image attachments. Voice chat, TRPG mechanics, external tools, and semantic lore search are future work. SillyTavern-only prompt surfaces are preserved on import and shown as inactive rather than silently mapped to a different location.
