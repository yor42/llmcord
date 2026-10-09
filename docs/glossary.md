# Glossary

These are the words the Discord commands, the admin dashboard and these docs use. Code names can differ; the last column lists them for contributors.

| Term | Meaning | Not | Code name |
| --- | --- | --- | --- |
| **Server** | A Discord server the bot is in. Dashboard pages, settings and lore are per server. | "guild" (Discord's API name) | `guild`, `guild_id` |
| **World** | A setting that characters belong to. Every character has one home world. | | `space` with `kind='world'` |
| **Hub** | A shared setting where characters from several linked worlds can meet. | | `space` with `kind='hub'` |
| **Space** | A world or a hub, when either one fits (for example `/admin space bind`). Prefer "world or hub" in sentences. | | `space` |
| **Link** | Let a world's characters appear in a hub (`/admin space link_world`, dashboard **Link**). **Unlink** removes them and drops them from that hub's channel casts. | "allow", "disallow" | `link_world`, `unlink_world`, `hub_worlds` |
| **Bind** | Attach a Discord text channel to one world or hub. Scenes in that channel use its characters and lore. | | `bind_channel`, `channels` |
| **Cast** | The characters who can speak in a channel. The **default cast** is set by admins; members change the **active cast** of a scene. | | `default_cast`, `active_cast` |
| **Ambient participation** | Characters may join in without being mentioned. | | `ambient` |
| **Lore** | Facts the bot can put into a prompt. Each lore entry belongs to one owner. | "world info" (only for SillyTavern imports) | `lore`, `world_info` |
| **Owner** | Where a lore entry belongs: a world, hub, character, channel, thread, the server, or a lorebook. | "scope" in user text | `owner_kind`, `scope_kind` |
| **Channel lore** | Lore owned by one channel, or by one thread when added inside a thread (`/admin lore add scope:channel`). | "local" | `channel`, `thread` scopes |
| **Server-wide lore** | Lore owned by the server, available in every world and hub. | "guild lore" | owner kind `guild` |
| **Lorebook** | A named, reusable set of lore entries, usually imported from SillyTavern or RisuAI. A **server lorebook** is enabled per world or hub; a **channel lorebook** applies to one channel. | "book" alone | `lorebooks`, owner kind `book` |
| **Character card** | A V2/V3 character file (`.json` or `.png`) imported into a home world. | | `cards.py` |
| **Emotion avatar** | A per-emotion image for a character. The **fallback avatar** is used when no emotion image is usable. | "slot" in sentences | `avatar_slots`, `avatar` |
| **Prompt preset** | The ordered prompt blocks a server uses. A **draft** becomes live only after **Activate**. | | `presets`, `preset_revisions` |
| **Scene** | One conversation thread of saved lines. Replying to an earlier line starts a **branch** from it. | | `nodes`, `root_id`, `parent_id` |
| **Personal fact** | Something a member said about themselves, kept only after `/memory opt_in`. | | `memories` |
| **Operator** | Someone who runs the bot itself and manages bot-wide settings on the dashboard's **Bot settings** page. Listed in `LLMCORD_OPERATOR_IDS`; separate from server admins. | "owner", "maintainer" | `operator_ids`, `guard_operator` |
| **Spending cap** | A bot-wide limit on estimated model cost per period. The **soft cap** warns operators; the **hard cap** pauses new character replies until the reset day. | "budget" in user text | `bot_settings`, `budget.py` |
| **Catch-up** | A private summary of what a member missed in one channel or thread (`/catchup`). Not a scene turn: it uses its own instruction, not the character prompt, and saves nothing. | "recap" (the scene summary memory keeps) | `catchup.py`, `catchup_anywhere` |
| **Monitoring** | The dashboard tab with a server's model usage (chart and breakdowns) and its turn log. | "stats" | `monitoring_panel`, `usage_report` |
| **Turn log** | A per-server record of model calls and failed turns, kept for the server's retention period, with personal facts hidden. Off until an admin turns on **Keep a turn log**. | "bot log" (the process log on the host) | `turn_log`, `add_turn_log` |

## Writing rules

- Say "server", never "guild", in anything a user reads.
- Name the kind: "world", "hub" or "world or hub". Use "space" only where a command already uses it.
- Say "lorebook", not "book", except in a short list where the context is clear.
- Deletion dialogs use the same shape: the title "Delete *name*?", what is removed and what stays, then **Cancel** and a red **Delete …** button.
