# Maintenance and UI polish backlog

**Mode since 2026-10-08:** maintenance and UI polish. The hardening rework (R1–R6) is closed; its audit, roadmap and decisions D1–D12 are in [history/](history/). This file replaces both: one list of small items, each done through the `orchestrate-change` skill.

## Rules
- **IDs:** `MNT-NN` for maintenance (correctness, reliability, structure, tooling, dependencies), `UI-NN` for dashboard and Discord-facing polish, `FEAT-NN` for small features the user asked for (each step still one item). Never renumber; mark an item **Done (commit)** and keep it until the next tidy. Old IDs (`BUG-01`, `SEC-02`, …) still mean what [the audit](history/audit-2026-10-07.md) says; a follow-up carried from them names its origin.
- **Size:** one item = one small change, green on `scripts/verify.sh` at the level the `verify` skill requires. Anything bigger, any schema change, and any change to an invariant in `CLAUDE.md` needs the user's go-ahead first.
- **UI items** follow [the UI guide](ui-guide.md) and the [glossary](../glossary.md), and carry before/after evidence (screenshot or exact reply text) in the report.
- **Priority:** `now` (user asked or user-visible harm), `next`, `later`. The user picks what moves to `now`.
- **New items:** add the problem, evidence (`path:line` or a test), and the smallest fix you can see. Mark guesses SUSPECTED.

## Maintenance

| ID | Item | Origin | Priority |
| --- | --- | --- | --- |
| MNT-01 | Decide whether store calls move to an executor. Needs slow-sqlite WARNING logs from the Pi (`llmcord_core.store`, ≥ 50 ms). | REL-01 | next (blocked on logs) |
| MNT-02 | Bound the bot's per-channel caches (`channel_locks`, `webhook_locks`, `webhook_defaults`, `webhooks`, `checked_avatar_assets`); drop a cached webhook on 401 too, and only on Unknown Webhook (10015) for NotFound. | REL-04, R3 step 2 | later |
| MNT-03 | The channel lock is released after delivery, but a turn still holds it across all its model calls; revisit only if long turns block channels in practice. | REL-02 | later |
| MNT-04 | Avatar listings read every blob to hash it; a stored `image_hash` column (schema bump, needs approval) would remove that. | PERF-03 | later |
| MNT-05 | `archive_character` writes with `with self.db`, not `write_admin()`, and has no revision check; `thread_casts` has no `guild_id` column (scope relies on character ownership). | SEC-06 | later |
| MNT-06 | `add_personal`'s consent check and insert are two statements (a cross-process opt-out could race it by microseconds). Put both in one transaction. | ARCH-05 | next |
| MNT-07 | `avatars.py` imports `auth.py` only for `DISCORD_API`, so the bot process loads FastAPI; move shared constants to a leaf module. `test_single_api_base` checks string identity and would miss a re-added equal literal. | ARCH-03 | later |
| MNT-08 | Socket-auth handlers (`dashboard._install_socket_auth`) and `rejection_notice` are covered only by the browser suite; add offline tests, including the silent path for unbound clients. | ARCH-01 step 5c, PERF-01 step 5 | next |
| MNT-09 | Not every `_run_scene` failure stage label is pinned (speaker selection, image description, saving scene input, preparing character prompt, webhook setup); `_stream_speaker`/`_record_reply` take many positional args. | ARCH-01 step 5b | later |
| MNT-10 | `create_app` (137 lines) and `setup_panel` (100 lines) are the remaining long functions; `setup_panel` calls `eligible_characters`/`allowed_worlds` per channel and hub. Split only alongside other work there. | ARCH-01, PERF-05 | later |
| MNT-11 | Test tidy: a few inline `Settings(...)` variants and two parallel channel/webhook fakes remain next to `tests/helpers.py`; `CompiledAdapter` mirrors only the Anthropic system-message split. | TOOL-01, ARCH-04 | later |
| MNT-12 | Some test docstrings cite `docs/engineering/audit.md` / `roadmap.md` (now stubs). Point them at `history/…` and delete the stubs. | 2026-10-08 restructure | later |
| MNT-13 | Production web server: consider `timeout_keep_alive` in `web_main.py` if Tailscale Serve reuses upstream connections longer than uvicorn's 5 s (the browser fixture hit this race; no production symptom seen). | R5 flake fix | later (SUSPECTED) |
| MNT-14 | `protected_uploads` 401/413 responses lack the security headers (`BaseHTTPMiddleware` sits outside `SecurityHeaders`). | PERF-04 | later |
| MNT-15 | Memory: a small `context_tokens` memory profile can fail at compile with the 6000-token default; a branch 200+ nodes past its last summary is summarized without the prior summary; `_history`'s inline summarization ignores the new limits. | BUG-03 | later |
| MNT-16 | Error redaction keys off env-var name suffixes, not the configured `api_key_env` values; `PROVIDER_STAGES` matches stage label strings. | SEC-05, UX-07 | later |
| MNT-17 | Dependency and CI upkeep: review Dependabot PRs, keep Python 3.12/3.13 and the arm64 container green. | standing | standing |
| MNT-18 | Discord images on the dashboard: the admin CSP is `img-src 'self' data: blob:` (`web.py:40`), so server icons and user avatars from `cdn.discordapp.com` are blocked. Decided (user 2026-10-08): add `https://cdn.discordapp.com` to the admin `img-src`, nothing else. Build icon and avatar URLs from the hashes already in `/users/@me` (kept in the session, `web.py:156`) and `/users/@me/guilds`; no new Discord calls. Opus review (CSP). | UI-28, UI-29 | now |

## UI polish

| ID | Item | Origin | Priority |
| --- | --- | --- | --- |
| UI-01 | Screenshot tooling: `scripts/screenshot_dashboard.py` starts the browser fixture and saves each tab at desktop and phone width (full page and first screen) with a manifest. | new | **Done** (2026-10-08) |
| UI-02 | First UI survey: walk every tab (Server setup, Characters, Lore, Imports, Prompt presets) and the member/admin slash-command replies; list concrete issues here as new `UI-NN` items with screenshots. Found UI-14..UI-27 (screenshots: `scripts/screenshot_dashboard.py` plus an expanded character and the lore edit form, at both widths). | new | **Done** (2026-10-08) |
| UI-03 | Threads still show as raw IDs in lore owner labels and selects (channels show `#name`). | UX-01 | next |
| UI-04 | A burst of rejected live events shows one toast per event; deduplicate. | PERF-01 step 5 | next |
| UI-05 | Cancelled confirmation dialogs stay in the DOM until the next panel refresh. | UX-05 | later |
| UI-06 | An admin command used from a stale DM client says "Only server administrators…" instead of "use this in a server". | R4 step 2b | later |
| UI-07 | `/context` output falls back to raw scope ids (`guild #…`, `channel #…`, `space #…`) and lists lore as bare `#id` (`discord_bot.py` `_register_context_command`). Use the glossary words and entry names. | UX-06 | next |
| UI-08 | The `settings.footer` audit row has no on/off detail; dashboard actions that navigate with `?tab=` drop `owner=`. | R4 step 5, UX-01 | later |
| UI-09 | Coverage for polish-sensitive flows: only the Save cast success toast is browser-tested; emotion-removal dialogs and the button-to-audit wiring (link/unlink/bind) are untested; the avatar `?v=` URLs are not asserted. | UX-01, UX-05, SEC-02 | next |
| UI-10 | Time the Lore panel build and the character expand separately (character expand went 0.05 → 0.30 s at the end of R5, not profiled). | R5 stage end | later |
| UI-11 | Phone width (390 px): the tab bar shows four tabs and Quasar's scroll arrow is drawn over the fourth label ("›MI…"); Prompt presets is off-screen with no hint. Consider shorter labels or a wrapped/stacked tab bar on narrow screens. Evidence: `scripts/screenshot_dashboard.py --viewport phone`, any `*-phone-top.png`. | UI-01 run | next |
| UI-12 | Lore and Prompt presets are very tall (page heights 10.7k / 17.3k CSS px on desktop, 15.4k / 16.8k on phone, with the fixture's 75-entry lorebook and the default preset). Look at collapsing, paging or sticky controls in the UI-02 survey. | UI-01 run | next (survey) |
| UI-13 | One run logged `SecurityError: Failed to read the 'cssRules' property from 'CSSStyleSheet'` from `nicegui.js` on Prompt presets at phone width; not reproduced in two later runs. | UI-01 run | later (SUSPECTED, intermittent) |
| UI-14 | Server setup breaks the glossary's "name the kind" rule: section **Spaces**, field **Space name**, button **Create space**, and the rebind warning "…not available in the new space". Use "Worlds and hubs", "Name", "Create", "…in the new world or hub". | UI-02 | next |
| UI-15 | Server setup has six separate Save buttons (channel guidelines, cast, ambient, footer, asset channel, plus Bind) and mixes full-width fields with ~250 px selects (Kind, Hub, World, channel pickers); **Sign out** sits unstyled under the card. Judgement call (one Save per channel card would change audit rows, so that part is MNT); show the user two layouts first. | UI-02 | later |
| UI-16 | Usage table: Estimated USD shows six decimals (`$0.001000`); at phone width the table scrolls sideways inside its card with Input tokens cut off and no hint. | UI-02 | next |
| UI-17 | File uploaders (Imports > Character cards, Prompt presets > Import preset) show Quasar's raw `0.0B / 0.00%` header before any file is picked, over an empty box. | UI-02 | next |
| UI-18 | Imports > Named lorebooks: the label **Lorebook scope** says "scope" (glossary: avoid in user text); the **Channel (channel lorebooks only)** select stays visible for a server lorebook. | UI-02 | next |
| UI-19 | Lore: the pager select's label `Page` is clipped to "P" (`lore_workspace.py:191`); an owner with one page shows "page 1" and no pager. CONFIRMED in `lore-desktop-top.png`. | UI-02 | next |
| UI-20 | Lore entry cards are ~195 px each (checkbox row, content, "Priority 100 · Enabled", three full-size buttons), so a 50-entry page is ~10k px. A compact row (content first line, priority, icon buttons with tooltips) would cut it to a few thousand. Entries seem to sort as text ("Fixture entry 5" after "…49"), SUSPECTED; check the sort key. Main fix for UI-12 on Lore. | UI-02 | next |
| UI-21 | Lore edit form: **Keys** and **Secondary Keys** are raw JSON arrays (`[ "fixture"]`, hint "JSON string array; preserves commas inside regex"); "Secondary Keys" is title-cased; the **Priority / insertion order (hig…** label is truncated. A chip input would avoid hand-typed JSON. | UI-02 | next |
| UI-22 | Lore at phone width: the search field shares a row with **Right owner** and shrinks to "Searc…"; the four bulk buttons stack one per row above the lists. | UI-02 | later |
| UI-23 | Prompt presets: the preset actions (**Load selected preset**, **Save draft**, **Save as new preset**, **Activate saved revision**, **Delete preset**) sit at the bottom, below every block (~17k px), and **Load selected preset** is far from the **Preset library** select it acts on. Blocks are always fully expanded. Move the actions to a bar next to the library select (or make it sticky) and collapse blocks to one summary line. Main fix for UI-12 on Prompt presets. | UI-02 | next |
| UI-24 | Prompt presets: selects show code values (`text`, `system`, `relative`, `card_post_history`); export has a raw JSON field "Explicitly omit nonportable block IDs (JSON array)". | UI-02 | later |
| UI-25 | Character panel: **Save character**, **Archive** and **Delete character** stack one per line and **Archive** looks like a primary action; the "Confirm moving worlds; ineligible casts will be cleared" checkbox shows even when the home world is unchanged. | UI-02 | next |
| UI-26 | Discord reply wording: `ValueError` replies lack a final period and vary in voice ("Lore entry not found", "Destination space not found", "Give a numeric message ID", "Card exceeds 8 MiB") while other replies are sentences; `/admin lore promote` ends "…in space." / "…in channel."; `/cast show` returns bare names, unlike the "Active cast: …" reply at `discord_bot.py:688`; ambient replies say both "Ambient participation enabled…" and "Ambient is on."; "character(s)" at `:710`. | UI-02 | next |
| UI-27 | Failure notices say "An admin can find details in the bot log" (`discord_bot.py:443`, `:457`); point them at the Log tab once FEAT-07 lands. | UI-02 | queued (with FEAT-07) |
| UI-28 | General styling pass across the dashboard: one visual style (colour tokens, card and section spacing, headings, button weights, form widths) applied to every page, starting with a styled sign-in card (Discord-style button) and a grid of server cards with the real server icon (initials when the server has none), replacing the blue text links (`dashboard.py:273`, `:281`). Record the style in the ui-guide; do one page per step with before/after screenshots. Server icons need MNT-18. | user 2026-10-08 (UI-02 batch) | now |
| UI-29 | Header: the signed-in user's Discord avatar and display name at the top right (`dashboard.py:290-291` shows the username as text), opening a small menu with **Sign out**; drop the unstyled **Sign out** link under each page (`dashboard.py:338`). Needs MNT-18. | user 2026-10-08 (UI-02 batch) | now |
| UI-30 | Icons: use Lucide icons instead of the Material set in tabs, buttons and list actions. The admin CSP allows only same-origin fonts and scripts, so vendor the SVGs that are used (Lucide is ISC-licensed; keep its licence file) and add a small helper; no CDN. Pick the icon for each action in the ui-guide. | user 2026-10-08 (UI-02 batch) | now |
| UI-31 | Macro highlighting in editors: highlight `{{user}}`, `{{char}}` and the other macros `prompts.substitute` knows in the character fields, the lore edit form and prompt block text, and mark unknown `{{…}}` differently. Spike first: NiceGUI's bundled CodeMirror with a match decorator (small same-origin JS, like `lore_drag.js`) against a highlight layer behind the textarea. It must keep each field's label (browser tests and screen readers find fields by label) and only load in open editors, not in the 50-entry lore list. | user 2026-10-08 (UI-02 batch) | next (spike, then one editor at a time) |
| UI-32 | Editing flow: one convention for where actions go, so editing needs less scrolling. Each editor gets one action bar beside its title (primary **Save** first, secondary actions next, destructive action last and red); lists use compact rows with icon buttons and tooltips. Done through UI-15, UI-20, UI-23 and UI-25. Show the user two mock layouts before the first one and record the choice in the ui-guide. | user 2026-10-08 (UI-02 batch) | now (layout choice first) |

## Features

**Local time (D14).** Characters know the local time of the member whose message started the turn. Discord does not expose a user's timezone, so it resolves in two steps: the member's own setting for this server, else the server default, else UTC.

| ID | Item | Origin | Priority |
| --- | --- | --- | --- |
| FEAT-01 | Schema 4 → 5, one backup-before-upgrade migration for both features (D15): `guild_settings.timezone` (IANA name, empty = UTC), `user_timezones(guild_id, user_id, timezone)`, `guild_settings.turn_log_enabled` (default off) and `turn_log_days` (14 or 30, default 14), and the `turn_log` table (guild, channel, message/node ID, stage, model profile and model, status, reference ID, error detail, request and response text, token counts, created time; index on guild + time). Add the `tzdata` package (the slim container may lack system zone files). Store helpers for the timezone part validate names with `zoneinfo`. Opus review (migration). | user 2026-10-08 | now |
| FEAT-02 | Prompt: compute the time from the triggering message's timestamp (so a rewind or retry sees the same time), in the resolved zone. Add macros `{{time}}`, `{{date}}`, `{{weekday}}`, `{{isotime}}`, `{{isodate}}` (SillyTavern names) and `{{local_time}}`; add a short `time` block to the default dialogue bundle ("Local time for {{user}}: …"). Existing saved presets get the macros only. Document in `admin-console.md`. | user 2026-10-08 | now (after FEAT-01) |
| FEAT-03 | Member commands `/time set <zone>` (autocomplete over zone names), `/time show` (shows the zone in use and where it came from), `/time clear`. Replies are ephemeral. Document in `server-guide.md`. | user 2026-10-08 | now (after FEAT-01) |
| FEAT-04 | Dashboard: a server timezone select in Server setup next to the reply footer, saved through `ctx.button` with audit action `settings.timezone` (detail: old → new). Browser test and screenshots. | user 2026-10-08 | now (after FEAT-01) |

**Turn log (D15).** A `Log` dashboard tab where a server's admins review their own server's recent turns: errors with full detail (stage, provider error, stack, the reference ID the member saw) and each model call's input and output (director, speakers, summaries, image descriptions). Today an error leaves only a log line and a reference ID (`discord_bot.py:441`, `:980`), and `trace` keeps only which sources a reply used (`store.py:101`), not the prompt or raw output.

Settled 2026-10-08 (D15): a per-server switch, off until an admin turns it on; entries kept 14 or 30 days (per-server choice, default 14); personal facts are masked when the entry is written (the prompt keeps a placeholder such as `[3 personal facts hidden]`), so the log never holds them and `/memory opt_out` has nothing to purge there; the tables ride FEAT-01's schema bump.

| ID | Item | Origin | Priority |
| --- | --- | --- | --- |
| FEAT-05 | Turn log store helpers on the FEAT-01 table: insert with a per-entry size cap, secret redaction (see MNT-16) and personal-fact masking; guild-scoped reads (newest first, by channel, errors only, by reference ID); deletion past each server's `turn_log_days` in the daily cleanup; nothing written while the switch is off. Opus review (consent, isolation). | user 2026-10-08 | next (after FEAT-01) |
| FEAT-06 | Engine and bot: record each model call and each `_run_scene` failure into the log without slowing the turn (one insert per call, after it returns). | user 2026-10-08 | next (after FEAT-05) |
| FEAT-07 | Dashboard: a `Log` tab (newest first, filter by channel, errors only, and by reference ID; collapsed rows that expand to the full input and output). Read-only, through `AuthService.guard`; the on/off switch and 14/30-day choice live in Server setup and are saved through `ctx.button` with audit `settings.turn_log`; browser test and screenshots. | user 2026-10-08 | next (after FEAT-06) |

## Decisions

Decisions D1–D12 from the rework stay in force ([history/roadmap-r1-r6.md](history/roadmap-r1-r6.md#decisions-taken-2026-10-07)). New decisions continue the numbering here.

| ID | Date | Decision | Applies to |
| --- | --- | --- | --- |
| D13 | 2026-10-08 | The project moves from hardening to maintenance and UI polish; work is tracked as `MNT-*`/`UI-*` items in this file. | all |
| D14 | 2026-10-08 | Local time: a member's timezone is stored per server (`/time set`), falls back to a server default set in the dashboard, then UTC. It is an explicit setting, not a personal memory: it does not need `/memory opt_in` and `/memory opt_out` does not clear it (`/time clear` does). Characters get it through a default prompt block and SillyTavern-style time macros, using the timezone of the member whose message started the turn. | FEAT-01..04 |
| D15 | 2026-10-08 | Turn log: per-server switch (off by default), 14- or 30-day retention per server (default 14), personal facts masked at write time so the log never stores them, and admins see only their own server's entries. Its tables share schema version 5 with the local-time feature (one migration). | FEAT-01, FEAT-05..07 |
