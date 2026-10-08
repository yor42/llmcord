# Rework roadmap

**Status:** approved by the user on 2026-10-07. R1–R5 are complete and merged into `main`; R5 was fast-forwarded to `d2dec8a` on 2026-10-08. R6 is in progress on `rework/r6-legacy-structure`. All decisions D1–D12 are taken (see "Decisions taken").

Every phase goes through the `orchestrate-change` skill: needed tests → scoped worker → final verification → review. Model routing and evidence ownership follow [the Claude workflow](claude-workflow.md). Finding IDs refer to `audit.md`.

**Phase rules**
- A phase is one or more small PRs, each green on `scripts/verify.sh` (plus `--browser` when it touches dashboard files, plus `--bench` when it claims performance).
- When a known defect is fixed, its `expectedFailure` decorator is removed in the same change.
- When behavior changes on purpose, its characterization test is updated in the same change, and the reason goes in the PR.

## Order at a glance

| Order | Phase | Findings | Size | Risk |
| --- | --- | --- | --- | --- |
| 1 | R1 Correctness quick wins | BUG-01, BUG-02, BUG-04, BUG-05, SEC-01, REL-05 | S | low |
| 2 | R2 Dashboard authorization performance | PERF-01, SEC-03, SEC-04 | M | medium |
| 3 | R3 Bot turn reliability | REL-02, REL-03, REL-04, BUG-03, REL-01 (measure) | M | medium |
| 4 | R4 Discord command surface and messages | SEC-05, UX-03, UX-04, UX-06, UX-07, UX-08, UX-09 | M | low–medium |
| 5 | R5 Dashboard data layer and UX | PERF-02, PERF-03, PERF-04, PERF-05, ARCH-02, UX-01, UX-02, UX-05 | L | medium |
| 6 | R6 Legacy retirement and structure | SEC-02, ARCH-01, ARCH-03, ARCH-04, ARCH-05, SEC-06, TOOL-01 | L | medium–high |

R1 and R2 are independent and could swap. R1 goes first because its tests already exist (four `test_known_defect_*`), so it proves the agent workflow on the smallest diffs.

---

## R1: Correctness quick wins

- **Problem:**
  - **BUG-01:** `/lore add` keys are ignored.
  - **BUG-02:** the cleanup loop dies on its first error.
  - **BUG-04:** a node replace deletes its summary.
  - **BUG-05:** migrate uses a different database path resolution from the bot.
  - **SEC-01:** the form is parsed before auth.
  - **REL-05:** close order can skip `store.close()`.
- **Impact:**
  - Keyword lore works as documented.
  - Retention keeps running.
  - Summaries survive.
  - Unauthenticated uploads no longer hit the disk.
- **Dependencies:** none. Tests for BUG-01/02/04 and SEC-01 already exist as `expectedFailure`.
- **Risk:** low.
  - BUG-01 changes behavior for future `/lore add` calls only. Rows already created with keys stay pinned unless a migration unpins them (open decision D7).
  - BUG-05 must not move an existing deployment's database: the env var keeps precedence.
- **Status:** complete, committed as `1f21be3` on branch `rework/r1-correctness`; `verify.sh --browser` PASS, 165 tests, 0 expected failures.
- **Progress:** SEC-01 done (reviewed; follow-ups: expired-session ordering test, origin check before form parse → R2). BUG-02 + REL-05 done (reviewed). BUG-04 done (reviewed). BUG-01 done (reviewed). BUG-05 done (reviewed; deployment note in audit). Test tidy pass done (fixed known-defect tests renamed per `tests/CLAUDE.md`).
- **Order inside the phase:** SEC-01, then BUG-02 + REL-05, then BUG-04, then BUG-01, then BUG-05.
- **Verification:**
  - The four decorators are removed, `verify.sh` is green, and expected failures equal 0.
  - New tests for REL-05 (models.close raising) and BUG-05 (path resolution precedence).
  - `--browser`, because `auth.py` changes.

## R2: Dashboard authorization performance

- **Problem:** PERF-01. Every socket event, `ctx.run`, handshake and HTTP route (including avatar images) calls Discord `/users/@me/guilds`, uncached and serialized per token. Per-keystroke lore search multiplies the cost. Failed checks silently drop events.
  - Related: SEC-03 (redundant csrf/origin arguments on live calls) and SEC-04 (unbounded auth maps).
- **Impact:** from `perf-baseline.md`:
  - lore search goes from 21 checks / 23 s settled (rate limited) to an expected ≤1 check per TTL window;
  - Save cast goes from 3 checks / 2.4 s to an expected 0–1 checks.
  - This is the main reported pain.
- **Dependencies:** none in code. Product decision D1 (the acceptable revocation delay) sets the TTL.
- **Risk:** medium. This is the authorization boundary; `dashboard.py` monkey-patches every socket handler.
  - **Mitigations:**
    - characterize the current guard call counts first (`tests/test_web_auth_boundaries.py`);
    - keep the cookie-binding checks per event (cheap, no Discord call);
    - invalidate the cache on 401, sign-out and session expiry.
- **Order:**
  1. A per-session guild-list cache with a TTL inside `AuthService` (one change point).
  2. Drop the duplicate check in `AdminService.run` for live calls.
  3. Debounce lore search and stop the `ctx.run(lambda: True)` probe.
  4. Make avatar GETs `private, max-age` (images only).
  5. Notify the user when an event is rejected, instead of dropping it silently.
  6. Prune `sessions`, `request_locks`, `retry_at` and `refresh_locks`.
- **Verification:**
  - Count-based tests are updated deliberately.
  - `--browser`.
  - `--bench`, twice, on the same host, with a new dated section in `perf-baseline.md`.
  - A test that a revoked admin loses access within the TTL, and immediately on a 401.
- **Status:** complete on branch `rework/r2-auth-perf` (steps 1–6 + SEC-01 logout follow-up); `verify.sh --browser` PASS, 224 tests, 0 expected failures; bench twice, see `perf-baseline.md` (lore search settled 23.1 s → 1.5 s rate limited; 0 guild checks after cold load).
- **Progress:** step 1 done (reviewed twice): `AuthService.guilds()` caches the guild list per session for 300 s (D1), coalesces concurrent fetches, never caches errors, and is dropped with the session on 401, sign-out, expiry or failed refresh; `forget_guilds()` resets it (used by the browser fixture). `verify.sh --browser` PASS, 186 tests, 0 expected failures. Bench deferred to the end of R2. Step 2 done (reviewed): origin is checked before the form body (session → origin → body); SEC-03 resolved (`AdminService.run` drops csrf, live calls guard without csrf/origin, real boundary documented); 195 tests, 0 expected failures. Follow-ups (both done after step 6): `/logout` origin-before-body, and a positive test for a mutate with the correct Origin. Note: with the cache, step 2's "drop the duplicate check" became "keep the per-action guard", since a cached check costs no Discord call. Step 3 done (reviewed): lore search debounced at 300 ms (`lore_workspace.py`, Quasar prop); a 10-character burst = 1 board render (was 10), pinned by browser test `test_lore_search_is_debounced`; probes kept (free under the cache, they surface errors). Steps 1–3 committed `2bcf139`. Step 4 done (reviewed): stored avatars `private, max-age=300` (endpoint allow-list, everything else `no-store`), versioned `?v=` URLs in `guild.html` and the dashboard; 202 tests, 0 expected failures. Untested: the dashboard `?v=` URLs (browser-only) and HEAD/405 no-store (checked by the reviewer by hand). Step 4 committed `11ef19f`. Step 5 done (reviewed): rejected live events from a bound client show a negative notification (401 sign in again; else the detail), unbound clients stay silent; 203 tests, 0 expected failures. Follow-ups: dedupe repeated toasts, offline test for `rejection_notice` and the silent path. Step 5 committed `b24f96f`. Step 6 done (reviewed): SEC-04 resolved — `AuthService.prune()` on access and sign-in, `drop_session()` as the single removal path (logout now uses it), held or waited-on locks never dropped, pending backoff kept.

## R3: Bot turn reliability

- **Status:** complete on branch `rework/r3-turn-reliability` (steps 1–6); `verify.sh --browser` PASS, 306 tests, 0 expected failures. Manual private-Discord checklist (`docs/verification.md` items 3, 6, 9) still to run by the user.
- **Progress:** branch `rework/r3-turn-reliability` (from `main` at `827f183`). Step 1 done (reviewed): each model profile has `timeout_seconds` (default 120) and `max_retries` (default 1), validated at load and passed to `AsyncAnthropic`/`AsyncOpenAI` (was SDK default 600 s, 2 retries). The timeout is per read (time to first byte, gaps between chunks), so a stream that keeps sending is not cut off. Pinned by `tests/test_model_timeouts.py` (7 tests, hanging fake transport); 231 tests, 0 expected failures. Follow-ups: a mid-stream timeout surfaces as `ReadTimeout`, a pre-response one as `APITimeoutError` (unify wording with R4 error mapping); slow local `compatible` models may need a higher `timeout_seconds` (documented). Step 1 committed `23a1ea1`. Step 2 done (reviewed): webhook objects cached per (parent channel, character); a cache hit makes no `webhooks()` listing; identity edits still apply; NotFound on edit or the placeholder send drops the entry, re-gets the webhook and retries the send once (no duplicate message: a 404 send posts nothing); later NotFound drops the entry and propagates; emotion-asset checks expire after 600 s. Pinned by `tests/test_webhook_cache.py` (14 tests); 245 tests, 0 expected failures. Follow-ups (low): forget by compare-and-pop to avoid a thread/parent race evicting a fresh webhook; drop (no retry) on 401 too; restrict forget to Unknown Webhook (10015); caches unbounded (channels × characters); no thread-delivery test. Step 2 committed `b7aac21`. Step 3 done (reviewed twice): World Info regex keys share one lazily created MiniRacer isolate with a bounded RegExp cache (`lastIndex` reset, so g/y stay stateless); a timeout, OOM or non-JS error discards the isolate, a plain SyntaxError keeps it; a missing/broken `py_mini_racer` is a non-match; closed at exit (an open isolate hangs interpreter exit). 20 regex keys: 68.6 ms → 3.1 ms per pass. Pinned by `tests/test_world_info_regex.py` (18 tests, incl. isolate counts and injection-shaped text); 263 tests, 0 expected failures. Step 3 committed `02c7050`. Step 4 done (reviewed twice): BUG-03 per D6 + D4 — `summarize_scene` no longer stops past 30 nodes; it summarizes the newest nodes whose transcript fits `limits.memory_input_tokens` (6000), after the previous summary; new limits `memory_output_tokens` (550), `summary_every_messages` and `extraction_every_turns` (1, range 1–100); extraction stays single-turn (consent), keeps the user line when trimmed, and counts turns over the whole branch (`Store.count_user_ancestors`, guild-scoped recursive CTE). Transcript/text-building failures are logged, not raised. Defaults keep today's behaviour except very long transcripts (> ~18 KB) are trimmed instead of failing at compile. Pinned by `tests/test_memory_cadence.py` (24 tests, incl. consent and 266-node cadence); 286 tests, 0 expected failures. Follow-ups (low): a memory profile with a small `context_tokens` can still fail at compile with the 6000 default (document or clamp to the compile budget); a branch 200+ nodes past its last summary is summarized without the prior summary (`ancestors()` window); `summary_every_messages` still counts within that window; `_history`'s inline summarization does not use the new limits. Step 4 committed `0e5d9bd`. Step 5 done (reviewed): the channel lock is released after delivery; extraction + summary run as one background task per turn (`SkitBot.memory_tasks`, chained per channel so summaries stay in order, usage still attributed per guild); the next turn waits up to `MEMORY_WAIT_SECONDS` (15 s) for the channel's pending task, then proceeds without the newest summary (WARNING). Memory/summary failures are logged, no longer posted in the channel. `close()` waits up to 5 s, then cancels (cascading to queued tasks) before closing models and store. Consent holds (`add_personal` checks consent synchronously at write). Pinned by `tests/test_memory_background.py` (10 tests); 296 tests, 0 expected failures. Follow-ups: `/scene delete` racing an in-flight extraction can re-add candidates/encounters from the deleted scene (pre-existing, window now longer; fix with ARCH-05 in R6); scheduling sits inside `_run_scene`'s try (unreachable failure mislabelled). Step 5 committed `693a56d`. Step 6 done (reviewed): `Store` connects with `TimedConnection`, which times execute/executemany/executescript/commit and `with conn:` commit/rollback; calls ≥ `SLOW_QUERY_SECONDS` (50 ms) log one WARNING on `llmcord_core.store` with the duration and ≤ 80 chars of SQL, never parameters; `Store.timing_stats()` exposes calls/total/max/slow. Overhead ≈ 1.5 µs per call on the Pi. Row fetching is not timed; counters are approximate across threads. Pinned by `tests/test_store_timing.py` (10 tests); 306 tests, 0 expected failures; `verify.sh --browser` PASS. REL-01 decision pending real logs from the Pi: if slow-call warnings show event-loop stalls, move store calls to an executor (later phase).
- **Problem:**
  - **REL-02:** the channel lock is held across all model calls, and no client timeouts are set.
  - **REL-03:** a new V8 isolate per regex match.
  - **REL-04:** unbounded caches, webhooks listed every turn, and the avatar-asset check never expires.
  - **BUG-03:** the summary stops after 30 nodes.
  - **REL-01:** synchronous sqlite on the loop. This phase only measures it.
- **Impact:**
  - A hung provider no longer freezes a channel for up to about 30 minutes.
  - Turns are faster: one webhook lookup, cheaper regex.
  - Long branches keep a summary.
- **Dependencies:** D4 (extraction and summary cost per turn) and D6 (whether the 30-node cap is intended) decide BUG-03 and whether extraction moves off the critical path.
- **Risk:** medium. Moving extraction and summary after the lock release changes ordering guarantees between consecutive turns in a channel. A characterization test must pin that the next turn sees the previous summary, or the change must state that it no longer does.
- **Order:**
  1. Explicit provider timeouts and `max_retries`, from config.
  2. Webhook object cache, with invalidation on 404.
  3. Reuse the MiniRacer isolate and cache patterns.
  4. BUG-03 per decision D6.
  5. Lock scope and background memory tasks.
  6. Add timing logs around sqlite calls, to decide whether REL-01 needs an executor.
- **Verification:**
  - New tests with fake models that hang or raise.
  - A regex evaluation count test.
  - `verify.sh`.
  - A manual private-Discord checklist run (`docs/verification.md` items 3, 6 and 9).

## R4: Discord command surface and messages

- **Status:** complete (steps 1–5); `verify.sh --browser` PASS, 439 tests, 0 expected failures. Merged into `main` (fast-forward to `9e3e4be`) on 2026-10-08.
- **Progress:** branch `rework/r4-command-surface` (from `main` at `3ea7e7a`). Step 1 done (reviewed): per D3, a failed turn posts a generic public message with a reference ID (`ref xxxxxx`); the ERROR log carries the same ref, the stage, the full detail and the stack. `/summon` invokers also get an ephemeral notice with the stage and ref; the sanitized provider detail is shown only for provider stages (speaker selection, image description, dialogue generation), timeouts read "The model provider did not respond in time" (unifies the R3 `ReadTimeout`/`APITimeoutError` wording), other stages say "internal error". A failing public post or followup is logged, never raised. Slash command errors: our `ValueError` text passes through, UNIQUE `IntegrityError` → "That already exists.", anything else → generic message with a logged ref. `/memory forget` and `/lore delete` state the outcome (store methods return bool; scoping unchanged — `/lore delete` is guild-wide). Pinned by `tests/test_error_mapping.py` (25 tests); 329 tests, 0 expected failures. Follow-ups (low): `PROVIDER_STAGES` matches stage label strings; error redaction still keys off env var name suffixes, not the configured `api_key_env` values.
  - Step 2a done (reviewed): the tree's `allowed_contexts` is guild-only, so every synced command is unavailable in DMs; a `require_guild` guard answers "Use this command in a server." before any store call in the 14 callbacks that don't use `binding_for`. Pinned by `tests/test_command_contexts.py`. Committed `8d5e980`.
  - Step 2b done (reviewed): per D11, the 14 admin commands live under `/admin` (`default_permissions(administrator)`, runtime `has_permissions` kept); member commands keep their paths; group descriptions reworded; user docs updated (operators should re-check Integrations overrides). Pinned by `tests/test_admin_commands.py` (15 tests, incl. exact command surface and a DM sweep of every member command); 356 tests, 0 expected failures. Follow-ups (low): an admin command used from a stale DM client says "Only server administrators…" rather than "use a server"; guards check `interaction.guild`, not `guild_id`; two test docstrings (`test_admin_commands.py` module, `test_error_mapping.py:124`) still carry stale wording.
  - Step 3 done (reviewed): one resolver (`llmcord_core/names.py`) for character and space options — exact match wins, else a unique casefold match, ambiguity refused with candidates, clear not-found/wrong-kind messages; cast and summon resolve among eligible characters (`/cast remove` also among current cast members, so stale members can be removed), `/character info` across the server excluding archived characters. All 13 character/space options autocomplete (≤ 25, substring, guild-scoped, empty in DMs; comma lists complete the last name). Options renamed: `character`, `characters`, `space`, `hub`, `world` (UX-06). `disallow_world` now checks kinds. Pinned by `tests/test_name_resolution.py`; 389 tests, 0 expected failures. Follow-ups: `unlink_world` leaves stale hub cast members and `set_cast` then rejects `/cast add` (fold into step 4 pruning); autocomplete runs sync sqlite per keystroke (REL-01 data).
  - Step 4 done (reviewed): per UX-03, rebinding to the same space keeps cast and ambient; rebinding to another space keeps ambient and prunes cast members not eligible there, and the reply (slash) or a notification (dashboard) names them; `unlink_world` prunes now-ineligible hub channel casts and reports the count, and refuses a non-hub id; legacy web routes record `dropped`/`pruned` in the audit. Pinned by `tests/test_rebind_cast.py` (22 tests). Committed `f268328` + review follow-up. Follow-ups (low): thread casts are not pruned (no parent column; the engine ignores ineligible members at turn time); a same-space rebind keeps already-stale members; `bind_channel` reads the cast outside `BEGIN IMMEDIATE`.
  - Step 5c done (reviewed): per D5/UX-09, `/admin scene delete message_id:` removes that stored message and its descendants (`Store.delete_subtree`, guild-scoped recursive CTE; evidence/trace/activations/summaries of those nodes go too); ancestors and sibling branches stay; Discord messages are not deleted; the reply gives the count. Pinned by `tests/test_scene_delete.py` (12 tests). Follow-ups (low): a delete racing a pending memory task can hit the summaries FK or re-add candidates/encounters (ARCH-05, R6); the candidate recount is not guild-scoped (idempotent); `delete_scene` has no callers left; a reply to a deleted message starts a new root (unpinned).
  - Step 5a/5b done (reviewed, Opus: schema migration): per D2, `guild_settings.usage_footer` (default on) toggles the public model/usage/cost footer per server from the dashboard Server setup tab (`settings.footer` via `AdminService.run`); usage is still recorded. Schema is now v4: a v3 file is backed up to `.pre-v4-*` before the column is added; a v4 database refuses to open in an older build (roll back from the backup). Per D12, the "No active character" hint deletes itself after 15 s (`NO_CHARACTER_NOTE_SECONDS`, background task; close() cancels it). Pinned by `tests/test_usage_footer.py` (15 tests); `verify.sh --browser` PASS, 439 tests, 0 expected failures. Follow-ups (low): the hint task catches only `DiscordException`; close() snapshots hint tasks once (a hint created during the memory drain can be destroyed pending); the `settings.footer` audit has no on/off detail; no unit test for the dashboard switch path or for close() cancelling a hint; the footer flag is read per speaker, not per turn; `migrate_admin` check-then-ALTER runs outside `BEGIN IMMEDIATE` (two processes opening a v3 file at once — compose runs migrate first).
- **Problem:**
  - **SEC-05:** commands are not guild-only and have no `default_permissions`; provider errors are posted publicly.
  - **UX-06:** admin commands are visible to everyone; option names are inconsistent.
  - **UX-07:** raw errors reach users.
  - **UX-04:** inconsistent name matching.
  - **UX-03:** rebind silently resets the cast.
  - **UX-08:** public footer and "no character" message.
  - **UX-09:** `/scene delete` scope.
- **Impact:** fewer confusing or leaky messages, and admin commands hidden from members.
- **Dependencies:** decisions D2 (cost footer), D3 (public vs admin-only error detail) and D5 (`/scene delete` scope). Renaming command options changes the synced command signatures, so users see renamed options after the next sync.
- **Risk:** low–medium.
  - `default_permissions` is only a default, and server admins can override it in Discord. Keep the runtime `has_permissions` checks.
  - The new guild-only rule must not break the existing threads behavior.
- **Order:**
  1. Error mapping (UX-07, SEC-05 errors).
  2. Guild-only + `default_permissions`.
  3. Shared name resolver with autocomplete (UX-04).
  4. Rebind preserves or prunes the cast (UX-03).
  5. Footer, no-character message and scene delete, per decisions.
- **Verification:**
  - Update the characterization tests in `tests/test_slash_commands.py`, which already pin UX-03, UX-04 and UX-07.
  - A DM invocation test.
  - `verify.sh`.

## R5: Dashboard data layer and UX

- **Problem:**
  - **PERF-02:** lore N+1 and Python-side search.
  - **PERF-03:** blob over-fetch.
  - **PERF-04:** no gzip, buffered CSP rewrite.
  - **PERF-05:** repeated lookups.
  - **ARCH-02:** raw SQL and private helpers in the UI.
  - **UX-01:** no success feedback, wrong tab after reload, raw IDs.
  - **UX-02:** terminology.
  - **UX-05:** split flows.
- **Impact:**
  - Large lorebooks stay responsive.
  - Dashboard writes go through revision-checked store methods (admin-writes invariant).
  - Admins get clear feedback.
- **Dependencies:** R2 first, so that perf measurements are not dominated by auth latency. A glossary (D8) before UX-02 string changes.
- **Status:** complete (steps 1–8); `verify.sh --browser` PASS, 519 tests, 0 expected failures; bench twice, see `perf-baseline.md` ("end of R5"). Merged into `main` (fast-forward to `d2dec8a`) on 2026-10-08.
- **Progress:** branch `rework/r5-dashboard-data` (from `main` at `9e3e4be`), started 2026-10-08.
  - Step 1 done (reviewed): ARCH-02. `Store.update_character` runs under `write_admin()` and takes `expected_revision` (stale → `ConflictError('Character changed; reload before saving')`); the dashboard character save calls it instead of a raw UPDATE and private cast helpers. New guild-scoped read helpers `list_characters`, `list_channels`, `thread_lore_scopes`, `asset_channel_id` replace every raw SQL call in `dashboard.py` and `admin.py`. User-visible: channel bindings list ordered by channel id; an invalid name/world on save shows the store's message. Pinned by `tests/test_store_queries.py` (15 tests, incl. an `AdminService.owners` characterization and a source guard). Follow-ups (low): validation now runs before the revision check; `expected_revision=None` (legacy `web.py` route) still skips the check until R6; the card's `name` field is not stripped (pre-existing); the source guard is a substring denylist (could be a regex on `.execute(`/`SELECT`).
  - Step 2 done (reviewed): PERF-02. `admin_entries` builds entries from one query per owner kind (no N+1). New `admin_entries_page(guild, kind, owner, query, limit, offset) -> (entries, total)` filters, counts and pages in SQL; matching keeps the old casefold substring over content + keys (incl. the `keys_json` fallback) through the `llmcord_entry_match` SQL function registered in `Store.__init__`, with a content short-circuit and one pass per query (`COUNT(*) OVER ()`). Rows with undecodable rule JSON are skipped by a filter. The lore board uses it; the browser counter counts page calls. Store micro-bench at 5000 entries: unfiltered page 174 → 6.5 ms, filtered 175 → 62 ms (`perf-baseline.md`). Pinned by `tests/test_lore_paging.py` (11 tests). Follow-ups (low): filtered search is still linear per owner (FTS5 would change substring semantics); a corrupt row whose content matches is still returned and then fails to decode, as before; no tests for space/thread owners.
  - Step 3 done (reviewed): PERF-03. `avatar_slots` reads the slot table once and returns metadata only (`has_image`, `image_hash`, `image_version`, no blob); `avatar_slot` is a keyed single-row query that still returns the image; `avatar_asset` takes a precomputed hash; `usable_avatars` (called twice per bot turn) reads the slot table once instead of ~9 times for a 6-slot character; clear/delete read only the revision. Dashboard avatar URLs are unchanged (`?v=` = sha256[:12]). Pinned by `tests/test_avatar_queries.py`. Follow-up (low): listings still read every blob to hash it; a stored `image_hash` column (schema bump) would remove that.
  - Step 4 done (reviewed): PERF-05. `LiveContext.snapshot` (a per-render `cached_property`, built for the page's guild only) loads spaces, characters, channels, lorebooks and thread scopes once and derives worlds, hubs and owners from them; `AdminService.owners_from(...)` builds the owner list from preloaded rows and `owners(guild_id)` delegates to it (output unchanged). One guild page load now calls each guild-wide list once (was `list_spaces` 8, `list_characters` 4, `list_channels` 5, `list_lorebooks` 3, `thread_lore_scopes` 2) and `lorebook_links` once per guild-target book (was books × spaces). Callbacks and dialogs still read live (`direct_import_dialog` calls `owners` live). Pinned by browser test `test_guild_page_load_reads_each_guild_list_once` (store call counters in `tests/dashboard_server.py`) and `OwnersFromRowsTests`. The browser fixture gained a second world (`Annex`) and a guild-target book (`Guild book`); the bench seed shares this fixture. Follow-up (low): `setup_panel` still calls `eligible_characters`/`allowed_worlds` per channel and hub.
  - Step 5 done (reviewed): UX-01. `LiveContext.button` uses a private `_attempt() -> (ok, result)`: `then` runs after every success, including a None result (previously skipped, so e.g. guideline saves, creates and deletes never reloaded), never after a failure; `run` is unchanged. New `success=` toast on Save cast, Save ambient setting, Save footer setting, Save asset channel, Publish / repair image. Tabs are named `setup|characters|lore|imports|prompts`; `?tab=` selects any of them (unknown → setup) and a tab change rewrites `?tab=` with `history.replaceState` (other params kept), so `navigate.reload()` stays on the tab. `LiveContext.load_channel_names()` fetches Discord channels once per page load before any snapshot access; a failed fetch (non-200 or transport error) notifies and falls back to ids. Channels show as `#name` in lore owner labels (`owners_from(..., channel_names=None)`), the Imports channel select and the presets sample-channel select. Pinned by `tests/test_dashboard_context.py`, `OwnersChannelNamesTests`, and five browser tests. Follow-ups (low): thread owners still show IDs; only the Save cast toast is browser-tested; the lore-panel programmatic tab switch is not pinned by a test; `navigate.to(...?tab=...)` callers drop `owner=` (pre-existing).
  - Step 6 done (reviewed): PERF-01 "full reloads". The guild page builds only the selected tab panel; another panel is built on first selection (click, `?tab=`, or programmatic switch) with a fresh snapshot per build; Discord channel names are still fetched once per page load (renamed/new Discord channels show after a browser reload). `LiveContext.refresh(tab=None, owner=None)` clears every built panel and rebuilds the selected one in place; it replaces every `ui.navigate.reload()` and same-page `navigate.to` in `dashboard.py` and `scene_ui.py` (no `navigate` calls remain), and keeps `?tab=`/`owner=` in the URL via `set_url` (`history.replaceState`). `ctx.button` awaits an awaitable `then`. A builder that raises is retried on next selection; unknown tab names are ignored. Pinned by four browser tests (lazy build counters, no-reload marker + zero extra Discord channel fetches, fresh data across tabs, lore → Imports switch) and an `owner=` URL check; the workflow test's 16 `expect_navigation` waits became state waits. CI fix in the same commit: `test_tab_switch_is_recorded_in_url_and_survives_reload` uses `goto(page.url)` instead of `reload()` (a CI-only NiceGUI `cssRules` SecurityError after reload, not reproduced locally) and UX tests report failed/≥400 requests with page errors. Follow-ups (low): builders must not await during a build (a refresh mid-build would double-fill; none do today); `refresh(tab)` to another tab returns before that build finishes; three workflow waits are weak (rely on the next locator's auto-wait).
  - Step 7 done (reviewed): PERF-04. GZip outermost on the FastAPI app (level 6, ≥1 KiB; HTML, text/plain and octet-stream excluded for BREACH); `security_headers` is now the plain-ASGI `SecurityHeaders` (headers unchanged; only `/admin` HTML buffered for nonces; HEAD keeps `content-length`). Scope addition (user): versioned `/admin/_nicegui/<ver>/` assets get NiceGUI's immutable cache-control on 200/304 (was `no-store`; `dynamic_resources` and 404s stay `no-store`). `quasar.umd.prod.js` 503 KB → 155 KB on the wire, and cached. Pinned by `tests/test_web_compression.py` (10 tests). Follow-ups (low): `protected_uploads` (`BaseHTTPMiddleware`, outside `SecurityHeaders`) 401/413 responses lack security headers (pre-existing); the NiceGUI `cssRules` SecurityError after a failed stylesheet load (`ERR_TOO_MANY_RETRIES` on `quasar.unimportant.prod.css`) reproduced locally once in four browser runs (`test_lore_import_button_builds_imports_panel`) — a pre-existing flake, fixed after R5 on `fix/browser-css-flake`: the browser-test server used uvicorn's 5 s keep-alive, which closed idle sockets just as Chromium reused them (reproduced 1 in 12 with a 4.95 s gap between navigations; 0 in 24 after the fixture set `timeout_keep_alive=120`). Production runs behind Tailscale Serve and was left unchanged; if the proxy reuses upstream connections longer than 5 s it can hit the same race (no symptom seen; follow-up: consider `timeout_keep_alive` in `web_main.py`).
  - Step 8 done (reviewed): UX-02 and UX-05. Glossary first (D8): `docs/glossary.md`, linked from `docs/README.md`. `/admin space allow_world|disallow_world` → `link_world|unlink_world` (no alias; reply "Linked {world} to {hub}."); `/admin lore add scope:local` → `scope:channel` (thread when run in a thread; reply names channel/world/hub); `/lore list` labels the space row world/hub; empty `/space list` and `/lore list` replies reworded. Dashboard says server/lorebook/emotion (owner labels "Server: Server-wide lore", "Lorebook: X"; Imports "Lorebook name", "Server lorebook"/"Channel lorebook", "Create/Delete lorebook"; "Save emotion"). UX-05: Imports lorebooks get **Edit entries in Lore**; fallback and emotion images save on upload (`LiveContext.upload` takes `action/detail/then/success` like `button`, audited and guarded); one `scene_ui.confirm_dialog` for every delete/remove, newly covering preset delete, emotion removal, emotion image and fallback avatar removal; the lore editor's confirm checkbox became a dialog. User docs updated. Pinned by 5 new offline tests in `test_command_contexts.py`, `ListReplyTests`, and 4 new browser tests; renamed commands updated across the command tests. Verify: 519 offline OK (18 skipped), 18 browser OK. Follow-ups (low): legacy Jinja templates still say "guild book" (retired in R6); the emotion-removal dialogs are not browser-tested; cancelled dialogs stay in the DOM until the next panel refresh (as before).
  - Stage end: `verify.sh --bench` twice at `2d5ebf7` (`perf-baseline.md`, "end of R5"). Cold load 2.0 → 1.4 s settled; lore search settled 1.5 → 1.7 s, because the Lore tab click now builds the lazy Lore panel inside the measurement; 0 guild checks after cold load. Expanding a character went 0.05 → 0.30 s (not profiled). Follow-ups (low): time the Lore panel build and the character expand on their own; a bench seed large enough to show PERF-02.
- **Risk:** medium. The browser suite is the only dashboard coverage (TOOL-01), so add focused tests before refactoring `presets_panel` and `render_lore_workspace`.
- **Order:**
  1. Move ARCH-02 writes into the store.
  2. SQL pagination and search (PERF-02).
  3. Slot metadata queries (PERF-03).
  4. A per-render snapshot (PERF-05).
  5. UX-01 feedback and tab map.
  6. Lazy tab rendering or targeted refresh instead of `navigate.reload`.
  7. PERF-04.
  8. UX-02 and UX-05.
- **Verification:**
  - `--browser`.
  - `--bench` with a new dated section; lore search settled time is the target metric.
  - Store-level tests for the new queries, guild-scoped.

## R6: Legacy retirement and structure

- **Problem:**
  - **SEC-02:** legacy Jinja POST routes are live, and the legacy character edit skips the revision check.
  - **ARCH-01:** oversized functions.
  - **ARCH-03:** duplication.
  - **ARCH-04:** dead code and `hasattr` fallbacks.
  - **ARCH-05:** delete and expiry leave derived memory.
  - **SEC-06:** an unscoped thread-cast write.
  - **TOOL-01:** test-helper duplication.
- **Impact:** a smaller attack surface, one write path per operation, and code that agents can change in smaller pieces.
- **Dependencies:** D9 (retire the legacy routes?) and the ARCH-05 per-artifact retention decision. R4 and R5 should land first, so that splitting `register_commands`, `create_app` and `mount_dashboard` happens after their behavior settles.
- **Risk:** medium–high, because of removal.
  - Follow the CLAUDE.md rule: grep callers, including `tests/dashboard_server.py` and docs; add a characterization test; then delete.
  - The browser fixture may depend on legacy routes.
- **Order:**
  1. SEC-06 scoping.
  2. Legacy route removal, or a revision check on legacy edit if they are kept.
  3. ARCH-04 dead code, with its tests.
  4. ARCH-03 constants and helpers.
  5. ARCH-01 splits along the seams R4/R5 created.
  6. ARCH-05 per decision.
  7. Migrate older tests to `tests/helpers.py` where touched.
- **Verification:**
  - `verify.sh --browser`.
  - A grep proof that no caller remains.
  - Docs updated: `docs/admin-console.md` and `docs/architecture.md`.
- **Progress:** branch `rework/r6-legacy-structure` (from `main` at `048ffab`), started 2026-10-08.
  - Step 1 done (reviewed, Opus for the isolation invariant): SEC-06. `archive_character` validates the owner and raises `ValueError` for an unknown or other-server character, writing nothing; thread-cast pruning skips characters owned by another server. Pinned by 4 new SEC-06 tests and one changed test in `tests/test_store_lifecycle.py`; 523 tests, 0 expected failures. Follow-ups (low): archive is not a `write_admin()` write and has no revision check; no `thread_casts.guild_id` column.
  - Step 2 done (reviewed, Opus for the security surface): SEC-02 per D9. Removed the 19 legacy POST routes, the card-preview avatar route and `llmcord_core/templates/`; `GET /` → `/admin/`, `GET /guild/{id}` → `/admin/guild/{id}`; kept `/login`, `/auth/callback`, `POST /logout`, the avatar GETs. Grep proof: no app, script, fixture or bench caller of the removed routes or templates. SEC-01 tests retargeted to `/logout` and the NiceGUI upload guard (`tests/helpers.py::shared_dashboard()`, one NiceGUI app per process); a table test pins 404/405 for every removed path. Review follow-up: dashboard link/unlink/bind audit the legacy detail again (`AdminService.run` callable detail; `tests/test_dashboard_audit_detail.py`). Removed tests: legacy guild-page cost, picker+guild-page cache sharing, card-preview avatar, legacy rebind routes. Jinja2 stays pinned (NiceGUI requires it). `verify.sh --browser`: 529 offline tests OK, 18 browser OK. Follow-ups (low): dead `require_admin(mutate=True)` branch → step 3; avatar `?v=` not asserted; the button-to-builder wiring itself is untested (the builders and the `AdminService.run` path are).
  - Step 3 done (reviewed, Opus for the auth path and isolation tests): ARCH-04. Removed `lore.retrieve_lore`, `Engine.prompt_for`, `cards.character_prompt`, the `*_compiled` `hasattr` fallbacks in `engine.py`/`discord_bot.py`, `require_admin(mutate=...)` and the FastAPI `ConflictError` handler. Grep proof: no remaining references in app code, scripts or tests. Test fakes share `tests/helpers.py::CompiledAdapter`. Four `test_production_*` tests in `tests/test_core.py` re-pin hub-guest lore isolation, lore order/budget, local candidate activation and branch-rewind isolation via `lore_scopes` + `world_info.evaluate` and `Engine.prepare_dialogue`; the old tests lost only their dead-code assertions and two were renamed. Production lore order is insertion order (constants win only under a tight budget). `verify.sh --browser`: 532 offline OK, 18 browser OK. Follow-ups (low): `CompiledAdapter` mirrors the Anthropic system split, not the inline form other providers get; the hub-guest production test calls `evaluate` without `character`/`messages`.
  - Step 4 done (reviewed): ARCH-03. Shared `auth.DISCORD_API`, `auth.is_server_admin` (`ADMINISTRATOR = 1 << 3`) and `discord_bot.recent_human_lines`; no behaviour change. Grep proof: one `discord.com/api` literal, no `& 8`. Pinned by `tests/test_shared_helpers.py` (5 tests); 537 offline OK, 18 browser OK. `models.py` provider branches left. Follow-ups (low): bot-side `avatars.py` → `auth.py` import; weak URL-identity test.
  - Step 5a done (reviewed): ARCH-01, `register_commands` (459 lines) split into `_command_context` (shared helpers/autocompleters) and one `_register_*` function per command family (24–72 lines), entry point 22 lines; bodies moved verbatim (line-multiset check), registration order unchanged. Pinned by `tests/test_command_tree.py` (snapshot of 30 commands, 14 groups, params, checks), passing before and after; 540 offline OK.
  - Step 5b done (reviewed, Opus for the delivery/error paths): ARCH-01, `SkitBot._run_scene` (163 lines) → 44-line outline plus `_SceneProgress`, `_Stage` (shared stage label read by the failure path), `_stream_speaker` (67), `_record_reply`, `_report_scene_failure`; stage labels and progress-message state unchanged path by path. New characterization tests: avatar-lookup failure stage (`test_error_mapping`), placeholder NotFound retry with a multi-chunk reply (`test_webhook_cache`), ambient turn with a speaker (`test_discord_flow`); 543 offline OK. Follow-ups (low): `_report_scene_failure` and the no-speakers path set `progress.message` directly; `_stream_speaker`/`_record_reply` take many positional args; speaker selection, image description, saving scene input, preparing character prompt and webhook setup failure labels are not each pinned.

---

## Decisions taken (2026-10-07)

| ID | Decision | Consequence for the plan |
| --- | --- | --- |
| D1 | 5 minutes | R2 guild-list cache TTL = 300 s; a 401, sign-out or session expiry still invalidates immediately. |
| D2 | Per-server setting, default on | R4 adds a per-server setting for the usage/cost footer (dashboard-editable); default on, so existing servers see no change. |
| D3 | Generic public message; details private via ephemeral reply | R4 error mapping: public channel gets a generic message; the invoker gets the stage/provider detail ephemerally (interactions), detail also logged. |
| D4 | Configurable, including a max token budget option | R3 makes extraction/summary cadence configurable and adds a max-token budget for those calls. |
| D5 | Subtree from a message; database only | R4 `/scene delete` removes the given node and its descendants; earlier lines and sibling branches stay; Discord messages are not deleted. |
| D6 | Summarize the latest window | R3 BUG-03: past the cap, summarize the most recent window instead of skipping; the window size follows the D4 budget config. |
| D7 | Fix new entries only | R1 BUG-01 changes `/lore add` for future calls; no migration of existing rows. |
| D8 | "Server"; "channel" lore scope; "Link" | R5 UX-02: user-facing text says "server" (code keeps `guild`); `/lore add` scope `local` becomes `channel`; `/space allow` becomes or is aliased to `link`. Glossary in `docs/` first. |
| D9 | Retire the legacy Jinja routes | R6 removes legacy POST routes after grep + characterization tests; NiceGUI is the only write path. |
| D10 | `/scene delete` also removes personal facts and encounters sourced only from the deleted nodes; promoted lore stays; expiry keeps derived data | R6 ARCH-05. |
| D11 (2026-10-08) | Admin commands move under one `/admin` group | R4 step 2: admin subcommands become `/admin space create\|bind\|allow_world\|disallow_world`, `/admin character import`, `/admin cast default`, `/admin ambient on\|off`, `/admin lore add\|pin\|edit\|promote\|delete`, `/admin scene delete`; `/admin` has `default_permissions(administrator)` and keeps the runtime `has_permissions` checks; member commands keep their paths. |
| D12 (2026-10-08) | "No active character" note is short-lived | R4 step 5 UX-08: a mention in a channel with no active character still gets the public hint, but the bot deletes it after about 15 s. |

## Open product decisions

| ID | Decision | Blocks | Options and notes |
| --- | --- | --- | --- |
| D1 | How long may a removed Discord admin keep dashboard access? | R2 | The guild-list cache TTL: e.g. 30 s, 60 s or 5 min. A 401 always invalidates immediately. |
| D2 | Keep the public usage/cost footer on every reply? | R4 | Keep / admin-only via `/context` / config toggle per guild (UX-08) |
| D3 | Public vs admin-only error details | R4 | Public generic + ephemeral or log detail / keep public detail (SEC-05, UX-07) |
| D4 | Run extraction and summary on every turn? | R3 | Every turn (current cost) / every N turns / only when the summary gap is large / config |
| D5 | `/scene delete` scope | R4 | Whole root tree (current) / the invoker's branch only / subtree from a message; also delete the Discord messages? (UX-09) |
| D6 | Is the BUG-03 30-node cap intended? | R3 | If it is a cost guard: summarize the last window instead of skipping. If not: remove. |
| D7 | Unpin existing `/lore add` rows that have keys? | R1 | Migration to unpin / leave existing rows and fix only new ones (BUG-01) |
| D8 | Canonical terms (guild vs server, space/world/hub, "Link" vs "allow", local vs channel) | R5 (UX-02) | Pick one user-facing vocabulary; code names can stay |
| D9 | Retire the legacy Jinja routes? | R6 | Retire / keep read-only / keep with revision checks (SEC-02) |
| D10 | Should scene deletion and history expiry also remove personal facts, encounters and promoted lore derived from the deleted nodes? | R6 (ARCH-05) | Per artifact; consent rules already apply to personal facts |
