# Engineering audit

Snapshot: 2026-10-07, branch `bootstrap/engineering` (commit `1270c09`). Line numbers refer to that commit.

Each finding has an ID that tests reference in their docstrings (`BUG-01: ...`, `Characterization (UX-03): ...`). Do not renumber; retire an ID by marking it **Resolved (commit)**.

**Confidence**
- **CONFIRMED:** reproduced by a test, or the mechanism was read in code and has no other plausible interpretation.
- **SUSPECTED:** plausible from reading, but not reproduced, or the production trigger is unverified.

**Label** (one per finding)
- **incorrect:** wrong behavior.
- **fragile:** works today but breaks easily.
- **unconventional:** works, but surprises maintainers.
- **preference:** taste. Fix only alongside other work.

**Severity**
- **high:** user-visible harm or data loss now.
- **medium:** real cost, with a workaround.
- **low**

## Summary

| ID | Title | Sev | Conf | Label |
| --- | --- | --- | --- | --- |
| PERF-01 | Discord guild check on every event, request and action — **Resolved in R2 (steps 1, 3–5)** | high | CONFIRMED | incorrect |
| PERF-02 | Lore workspace N+1 and in-Python search/pagination | medium | CONFIRMED | fragile |
| PERF-03 | Avatar slot reads load every blob | low | CONFIRMED | fragile |
| PERF-04 | No gzip; CSP middleware buffers every `/admin` HTML response | low | CONFIRMED | unconventional |
| PERF-05 | Same lookups repeated per page render | low | CONFIRMED | preference |
| BUG-01 | `/lore add` keys are ignored — **Resolved (R1, branch `rework/r1-correctness`)** | high | CONFIRMED (test) | incorrect |
| BUG-02 | Daily cleanup loop dies permanently on one error — **Resolved (R1, branch `rework/r1-correctness`)** | medium | CONFIRMED (test) | incorrect |
| BUG-03 | Scene summary stops after 30 unsummarized nodes | medium | SUSPECTED (characterized) | incorrect |
| BUG-04 | `record_node` replace cascade-deletes the node's summary — **Resolved (R1, branch `rework/r1-correctness`)** | medium | CONFIRMED mechanism (test) | incorrect |
| BUG-05 | `migrate.py` ignores `config.yaml` `database_path` — **Resolved (R1, branch `rework/r1-correctness`)** | low | CONFIRMED | fragile |
| SEC-01 | Mutating legacy routes parse the body before auth — **Resolved (R1, branch `rework/r1-correctness`)** | medium | CONFIRMED (test) | incorrect |
| SEC-02 | Legacy Jinja POST routes still live | medium | CONFIRMED | fragile |
| SEC-03 | Live-call CSRF and origin checks pass by construction — **Resolved (R2 step 2)** | low | CONFIRMED | unconventional |
| SEC-04 | Unbounded in-memory auth state; sessions lost on restart — **Resolved in R2 step 6 (pruning; restart sign-out remains, by design)** | low | CONFIRMED | fragile |
| SEC-05 | Slash commands not guild-only or permission-gated; errors public | medium | CONFIRMED | incorrect |
| SEC-06 | `archive_character` cross-guild touches global thread casts | low | CONFIRMED (test) | fragile |
| REL-01 | Synchronous sqlite on the event loop | medium | CONFIRMED | fragile |
| REL-02 | Channel lock held across all model calls; no client timeouts | high | CONFIRMED | fragile |
| REL-03 | New V8 isolate per regex key match | medium | CONFIRMED | fragile |
| REL-04 | Unbounded and stale bot caches; webhooks listed every turn | low | CONFIRMED | fragile |
| REL-05 | Shutdown order can skip `store.close()` — **Resolved (R1, branch `rework/r1-correctness`)** | low | CONFIRMED | fragile |
| ARCH-01 | Oversized functions | medium | CONFIRMED | fragile |
| ARCH-02 | Raw SQL and private store helpers in UI code | medium | CONFIRMED | fragile |
| ARCH-03 | Duplicated constants and logic | low | CONFIRMED | preference |
| ARCH-04 | Production-dead code and test-double fallbacks | low | CONFIRMED | unconventional |
| ARCH-05 | Scene deletion and expiry leave derived memory | medium | SUSPECTED | incorrect |
| UX-01 | Missing success feedback; wrong tab after reload; raw IDs | medium | CONFIRMED | incorrect |
| UX-02 | Inconsistent terminology | medium | CONFIRMED | preference |
| UX-03 | Rebinding a channel silently resets cast and ambient | medium | CONFIRMED (test) | incorrect |
| UX-04 | Inconsistent name matching across commands | low | CONFIRMED (test) | incorrect |
| UX-05 | Lorebook and avatar flows split across tabs and steps | low | CONFIRMED | preference |
| UX-06 | Admin commands visible to all; inconsistent option names | low | CONFIRMED | preference |
| UX-07 | Raw and ambiguous error text | medium | CONFIRMED (test) | incorrect |
| UX-08 | Public cost footer and public "no character" message | low | CONFIRMED | preference |
| UX-09 | `/scene delete` removes the whole root tree | medium | CONFIRMED | fragile |
| TOOL-01 | Tooling, test-helper and docs gaps | low | CONFIRMED | fragile |

---

## Performance

### PERF-01: Discord guild check on every event, request and action
- **Severity:** high. **Confidence:** CONFIRMED (code + bench + count test in `tests/test_web_auth_boundaries.py`). **Label:** incorrect.
- **Root cause of the slow dashboard.** `AuthService.guard` calls Discord `GET /users/@me/guilds` on every invocation with no cache (`llmcord_core/auth.py:92-103`). `discord_get` serializes calls per (path, token) through `request_locks`. It also sleeps until `retry_at` once Discord returns `X-RateLimit-Remaining: 0` (`auth.py:36-67`).
- **Where `guard` runs:**
  - **Every socket.io `event`** (`llmcord_core/dashboard.py:96-99` via `socket_allowed`, `:68-87`). NiceGUI value elements emit an event per keystroke.
  - **Every `ctx.run`.** `LiveContext.run` → `AdminService.run` (`llmcord_core/admin.py:32`). A button click is therefore checked twice: once for the event, once for the action.
  - **Socket handshake and implicit-handshake connect** (`dashboard.py:89-108`).
  - **Every HTTP route** through `require_admin` (`auth.py:105-107`). This includes each avatar image GET (`llmcord_core/web.py:264-279`). Those images are also sent with `Cache-Control: no-store` (`web.py:70`), so the browser refetches them, and re-checks Discord, on every render.
- **Amplifiers:**
  - **Lore search.** Each value change calls `ctx.run(lambda: True)`, then re-renders the board (`llmcord_core/lore_workspace.py:220-225`).
  - **Silent drops.** A failed or rate-limited check makes `socket_allowed` return False, and the event is dropped with no message (`dashboard.py:85-87`, `:98-99`). *(Fixed in R2 step 5, see Status.)*
  - **Full reloads.** Most mutations end in `ui.navigate.reload()`, which re-renders all five tab panels eagerly (`dashboard.py:183-194`; e.g. `:242`, `:253`, `:267`, `:370`, `:575`, `:712`).
- **Why it matters:** see `perf-baseline.md`. Typing 10 characters triggers 21 guild checks. When Discord reports Remaining 0, they queue behind each other and the search settles about 20 s later.
- **Direction:**
  - Cache the per-session guild list with a short TTL (about 30–60 s), invalidated on 401 and on sign-out.
  - Check once per action, not per event plus per action.
  - Debounce search.
  - Let avatar GETs be privately cacheable.
  - Surface rejected events to the user.
  - Revocation latency becomes the TTL. That is a product decision (see the roadmap).
- **Status:** Partly resolved in R2 step 1 (branch `rework/r2-auth-perf`): the guild list is cached per session for 300 s (D1), invalidated on 401, sign-out, expiry and failed refresh; concurrent checks share one call; the picker pages use the same cache. Step 3: the lore search input is debounced (Quasar `debounce=300`), so a typing burst costs one board render instead of one per keystroke; the per-action guard and `ctx.run(lambda: True)` probes stay because they no longer call Discord within the TTL. Step 4: stored avatar 200 responses (`/characters/{c}/avatar`, `/avatars/{slot}`) are `private, max-age=300` via an endpoint allow-list in `security_headers`; all other responses stay `no-store`; image URLs carry `?v=<sha256[:12]>` so uploads show at once. A cached image stays viewable in the same browser profile for up to 300 s after logout or revocation (consistent with D1; no `Vary: Cookie`). Step 5: a rejected live event from a client confirmed bound to the requesting session (live `client_id`, binding present, cookie equals binding) now shows a negative notification — 401 "Your Discord sign-in expired. Sign in again.", other HTTPExceptions their detail (403 admin permission, 502/503 try again shortly); the event is still dropped. Unknown, unbound or cookie-mismatched clients stay silent. Pinned by browser test `test_rejected_live_event_notifies_bound_client`. Follow-ups (low): repeated rejections are not deduplicated (one toast per event); no offline test of `rejection_notice` or of the silent path; the guarded `javascript_response`/`ack`/`log` handlers still drop silently.

### PERF-02: Lore workspace N+1 and in-Python search/pagination
- **Status:** Resolved in R5 step 2: `admin_entries` uses one query per owner kind; the workspace pages through `admin_entries_page` (filter, count and LIMIT/OFFSET in SQL, same casefold substring semantics via a registered SQL function). Filtered search stays linear per owner; numbers in `perf-baseline.md` (2026-10-08).
- **Severity:** medium. **Confidence:** CONFIRMED. **Label:** fragile.
- **Evidence:**
  - `AdminStore.admin_entries` selects ids, then calls `admin_entry` per id (`llmcord_core/admin_store.py:320-328`).
  - The workspace filters by substring and slices pages in Python after loading every entry (`lore_workspace.py:151-156`).
- **Why it matters:** cost grows linearly with book size on every keystroke and refresh. The bench seeds 500 entries.
- **Direction:** one query per owner kind, with `LIMIT/OFFSET` and a `LIKE` (or FTS5) filter in SQL.

### PERF-03: Avatar slot reads load every blob
- **Severity:** low. **Confidence:** CONFIRMED. **Label:** fragile.
- **Evidence:**
  - `avatar_slots` runs `SELECT *`, which includes the `image` BLOBs (`admin_store.py:674-678`).
  - `avatar_slot(key)` loads all slots, then picks one (`:690-694`).
  - The per-image route (`web.py:273-279`) therefore reads every image of the character to serve one.
- **Direction:** select metadata columns for listings, and use a keyed single-row query for the image.

### PERF-04: No gzip; CSP middleware buffers every `/admin` HTML response
- **Severity:** low. **Confidence:** CONFIRMED. **Label:** unconventional.
- **Evidence:**
  - `ui.run_with(..., gzip_middleware_factory=None)` (`dashboard.py:197-198`).
  - `security_headers` joins the full body of every `/admin` `text/html` response, regex-inserts nonces, and rebuilds the `Response` (`web.py:67-85`).
- **Direction:**
  - Re-enable compression after the CSP rewrite (order matters).
  - Consider NiceGUI's own nonce or hash support if available.
  - Low priority over Tailscale on a LAN.

### PERF-05: Same lookups repeated per page render
- **Severity:** low. **Confidence:** CONFIRMED. **Label:** preference.
- **Evidence:**
  - `store.list_spaces(gid)` has 8 call sites in one guild page render: `dashboard.py:211,235,245,246,300,532,580`, plus inside `AdminService.owners`.
  - `owners()` is called twice (`lore_workspace.py:18`, `dashboard.py:560`).
  - The brief said "4× / 2×". The recount is higher; some sites run only on some panels.
- **Direction:** build one per-render snapshot and pass it to the panels.

## Defects

### BUG-01: `/lore add` keys are ignored
- **Status:** Resolved in R1 (branch `rework/r1-correctness`): `/lore add` with keys creates an unpinned, keyword-gated entry; without keys it stays constant + pinned. Per D7, existing rows were not migrated. Test renamed to `test_lore_add_with_keys_is_keyword_triggered`.
- **Severity:** high. **Confidence:** CONFIRMED (test `tests/test_slash_commands.py::test_known_defect_lore_add_with_keys_is_keyword_triggered`, expectedFailure). **Label:** incorrect.
- **Evidence:**
  - `/lore add` always passes `pinned=True` (`llmcord_core/discord_bot.py:588-589`).
  - `world_info` forces pinned rows to constant (`llmcord_core/world_info.py:60-61`).
  - So an entry added with keywords is injected every turn.
- **Why it matters:** silent token cost, and lore that leaks into unrelated scenes.
- **Direction:**
  - Pin only when no keys are given (or add an explicit `pinned` option).
  - Decide whether existing rows created with keys should be unpinned by a migration. That is a product decision.

### BUG-02: Daily cleanup loop dies permanently on one error
- **Status:** Resolved in R1 (branch `rework/r1-correctness`): `_cleanup_loop` logs a failed pass and retries next cycle.
- **Severity:** medium. **Confidence:** CONFIRMED (test in `tests/test_store_lifecycle.py`, expectedFailure). **Label:** incorrect.
- **Evidence:** `_cleanup_loop` has no try/except (`discord_bot.py:72-75`). One `sqlite3.OperationalError` (for example `database is locked` while the web process writes) ends the task, and history retention stops until restart. Nobody awaits the task, so the error surfaces only as "Task exception was never retrieved".
- **Direction:** log and continue inside the loop.

### BUG-03: Scene summary stops after 30 unsummarized nodes
- **Severity:** medium. **Confidence:** SUSPECTED, i.e. characterized but intent unknown. **Label:** incorrect.
- **Evidence:** `summarize_scene` returns early when more than 30 nodes follow the last summary (`llmcord_core/engine.py:273-274`). If a summary fails, or a branch grows beyond 30 nodes before the first summary, that branch is never summarized again, because the gap only grows.
- **Direction:** summarize the most recent window, or chunk. Needs a product decision on whether the cap is a cost guard.
- **Status:** Resolved in R3 step 4 (branch `rework/r3-turn-reliability`) per D6: past the budget, the most recent window that fits `memory_input_tokens` is summarized, so a branch never stalls; summary/extraction cadence and budget are configurable (D4). Stalled branches in existing databases resume on their next turn.

### BUG-04: `record_node` replace cascade-deletes the node's summary
- **Status:** Resolved in R1 (branch `rework/r1-correctness`): `record_node` upserts with `ON CONFLICT(message_id) DO UPDATE`, so the summary survives a re-record. Note: if a re-record ever changed the node's parent, the kept summary would describe the old ancestry (no current caller does this).
- **Severity:** medium. **Confidence:** CONFIRMED mechanism (test in `tests/test_store_lifecycle.py`, expectedFailure); production trigger unverified. **Label:** incorrect.
- **Evidence:**
  - `INSERT OR REPLACE INTO nodes` (`llmcord_core/store.py:342`) deletes and re-inserts the row.
  - `summaries.node_id` is `ON DELETE CASCADE` (`store.py:66`), so re-recording an existing message id drops its summary. Other FK children may be affected too.
  - A Discord message id should be recorded once, so the trigger needs a retry or duplicate delivery.
- **Direction:** `INSERT ... ON CONFLICT(message_id) DO UPDATE`.

### BUG-05: `migrate.py` ignores `config.yaml` `database_path`
- **Status:** Resolved in R1 (branch `rework/r1-correctness`): `config.resolve_database_path()` (env → YAML → default) is shared by bot, web, migrate and `scripts/check_host.py`. Deployment note: with the env var unset and a YAML `database_path` other than the default, web and migrate now open the YAML file. YAML `database_path:` null/empty still crashes at startup, as the bot always did.
- **Severity:** low. **Confidence:** CONFIRMED. **Label:** fragile.
- **Evidence:** `migrate.py:10` and `web_main.py` read only `LLMCORD_DATABASE_PATH` (default `data/llmcord.sqlite3`). The bot reads `database_path` through `load_settings`. A deployment that sets only the YAML key migrates a different file than the bot opens.
- **Direction:** one resolver for the database path, shared by all three entry points.

## Security

### SEC-01: Mutating legacy routes parse the body before auth
- **Status:** Resolved in R1 (branch `rework/r1-correctness`): `require_admin(mutate=True)` validates the session before `request.form()`. R2 step 2 added the origin check before the body too (session → origin → body). R2 follow-up: `/logout` now checks origin too (it previously never did, so a cross-site POST with a valid csrf could sign the user out); all 19 POST routes run session → origin → body. Positive same-origin tests added for a mutate and for logout.
- **Severity:** medium. **Confidence:** CONFIRMED (test in `tests/test_web_auth_boundaries.py`, expectedFailure). **Label:** incorrect.
- **Evidence:** `require_admin(mutate=True)` awaits `request.form()` before `guard` checks the session (`auth.py:105-107`). Unauthenticated multipart bodies are parsed and spooled to disk. The suite shows `ResourceWarning: unclosed SpooledTemporaryFile`.
- **Why it matters:** an unauthenticated client can make the Pi write upload bodies to disk, bounded only by python-multipart limits.
- **Direction:**
  - Check the session cookie (no Discord call) before reading the body.
  - Close the form.
  - Or move the CSRF token to a header.

### SEC-02: Legacy Jinja POST routes still live
- **Severity:** medium. **Confidence:** CONFIRMED. **Label:** fragile.
- **Evidence:**
  - 19 `@app.post` routes in `web.py:201-456` remain mounted next to NiceGUI.
  - They duplicate dashboard behavior with separate validation.
  - Legacy `edit_character` (`web.py:289-307`) writes without a revision check, which contradicts the admin-writes invariant.
- **Direction:**
  - Retire them, gated by a product decision.
  - Before removal, add characterization tests and grep for callers (the browser fixture, docs).

### SEC-03: Live-call CSRF and origin checks pass by construction
- **Severity:** low. **Confidence:** CONFIRMED. **Label:** unconventional.
- **Evidence:**
  - `AdminService.run` passes `self.app.state.base_url` as the origin (`admin.py:32`).
  - `LiveContext` passes the session's own csrf (`dashboard.py:25`, `:33`).
  - So the checks in `guard` can never fail for live calls.
  - Protection actually comes from the cookie-to-client binding (`dashboard.py:76-79`) and socket.io CORS (`:64`).
- **Direction:** document it as the real boundary, and drop the redundant arguments so the code does not imply a check that does not exist.
- **Status:** Resolved in R2 step 2 (branch `rework/r2-auth-perf`): `AdminService.run(ident, guild_id, operation, ...)` no longer takes csrf and calls `guard(ident, guild_id)`; comments in `admin.py` and `dashboard.py` `socket_allowed` name the real boundary (cookie-to-client binding, socket.io CORS, per-action admin guard). Upload requests still check `X-CSRF-Token` and Origin. Note: a missing `Origin` header is accepted (csrf still has to match on form routes and uploads).

### SEC-04: Unbounded in-memory auth state; sessions lost on restart
- **Severity:** low. **Confidence:** CONFIRMED. **Label:** fragile.
- **Evidence:** `app.state.sessions`, `request_locks`, `retry_at` and `refresh_locks` (`auth.py:18-21`) are never pruned. Sessions live only in memory, so every restart signs everyone out.
- **Direction:** prune expired entries on access or with a periodic sweep. Persisting sessions is optional.
- **Status:** Resolved in R2 step 6 (branch `rework/r2-auth-perf`). `AuthService.prune()` runs on every `session()` call and on each OAuth sign-in: it drops expired sessions, refresh locks with no session, and `request_locks`/`retry_at` keys no live session uses whose retry time has passed. `drop_session()` is the single removal path (logout, Discord 401, failed refresh, expiry) and also drops the session's refresh lock. A lock that is held or has waiters is never dropped (`busy()` reads `locked()` and the private `asyncio.Lock._waiters`, pinned by a test). Pending backoff survives with or without a session. Pinned by `tests/test_auth_prune.py`. Not done (optional): persisting sessions across restarts. Low follow-ups: a queued waiter after a failed refresh sends one more refresh POST (pre-existing); `prune()` is O(sessions + keys) per check.

### SEC-05: Slash commands not guild-only or permission-gated; errors public
- **Status:** error part resolved in R4 step 1 (provider detail no longer posted publicly; details ephemeral to the `/summon` invoker for provider stages, and logged). Guild-only (step 2a) and the hidden `/admin` group with `default_permissions(administrator)` (step 2b, D11) resolved; runtime checks kept.
- **Severity:** medium. **Confidence:** CONFIRMED. **Label:** incorrect.
- **Evidence:**
  - No `default_permissions` and no `guild_only`/`allowed_contexts` on any group.
  - Admin commands rely only on the `has_permissions` runtime check (14 sites, e.g. `discord_bot.py:364`).
  - `/memory` in a DM reaches the store with `guild_id=None`, and the raw `IntegrityError` text is returned.
  - Turn failures post `error_detail(error)` publicly in the channel (`discord_bot.py:333-344`). That is up to 1000 characters of provider or Discord error detail; `errors.py` strips credentials but not internal details such as model names or quota messages.
- **Direction:**
  - Make the commands guild-only, with `default_permissions(administrator)` on admin groups.
  - Show generic public errors, with details ephemeral or in logs. Needs a product decision.

### SEC-06: `archive_character` cross-guild touches global thread casts
- **Severity:** low. **Confidence:** CONFIRMED (test in `tests/test_store_lifecycle.py` shows a wrong-guild call does not archive). **Label:** fragile.
- **Evidence:**
  - The guild-scoped `UPDATE` is a no-op for a wrong guild.
  - But `_prune_character_casts` → `_remove_from_thread_casts` scans `thread_casts` without a guild filter (`store.py:526-566`).
  - Practically harmless, because character ids are global. It still breaks the "every store write is guild-scoped" invariant.
- **Direction:** validate the owner first and scope `thread_casts` by guild.

## Reliability

### REL-01: Synchronous sqlite on the event loop
- **Severity:** medium. **Confidence:** CONFIRMED. **Label:** fragile.
- **Evidence:**
  - `Store` uses one `sqlite3` connection with `busy_timeout=5000` (`store.py:140`), called directly from async code in the bot and in the web process.
  - A lock held by the other process can stall the whole event loop for up to 5 s: the Discord heartbeat, or every dashboard socket.
- **Direction:** short-term, keep transactions tiny and measure. Longer-term, use a thread executor for store calls.
- **Status:** Measured in R3 step 6 (branch `rework/r3-turn-reliability`): sqlite calls (including `with conn:` commits) ≥ 50 ms log a WARNING on `llmcord_core.store` (SQL prefix only, never parameters), and `Store.timing_stats()` keeps counters. Decide on an executor once production logs exist.

### REL-02: Channel lock held across all model calls; no client timeouts
- **Severity:** high. **Confidence:** CONFIRMED. **Label:** fragile.
- **Evidence:**
  - `on_message` holds `channel_locks[channel]` for the whole `run_scene` (`discord_bot.py:154-155`): director, image description, N dialogue streams, extraction and summary (`:205-346`).
  - `AsyncAnthropic`/`AsyncOpenAI` are built without `timeout` or `max_retries` (`llmcord_core/models.py:70`, `:76`). The SDK defaults are a 10-minute timeout with 2 retries.
  - One hung provider call blocks the channel for up to about 30 minutes.
- **Direction:**
  - Set explicit timeouts.
  - Release the lock after delivery.
  - Run extraction and summary as a follow-up task.
- **Status:** Partly resolved in R3 step 1 (branch `rework/r3-turn-reliability`): explicit per-profile `timeout_seconds` (120) and `max_retries` (1); worst case for one hung call is about 2 × 120 s instead of about 30 min. The channel lock is still held across all model calls (R3 step 5).
- **Status (R3 step 5):** Resolved. The lock is released after delivery; extraction and summary run as a chained per-channel background task, and the next turn waits at most 15 s for it. New related note: a `/scene delete` during an in-flight extraction can re-add candidates/encounters sourced from the deleted scene (pre-existing race, longer window now); track with ARCH-05.

### REL-03: New V8 isolate per regex key match
- **Severity:** medium. **Confidence:** CONFIRMED. **Label:** fragile.
- **Evidence:** `world_info` creates a `MiniRacer()` for every regex-key evaluation, synchronously (`llmcord_core/world_info.py:28-32`). A large regex-heavy lorebook multiplies isolate start-up cost per turn, and that time blocks the event loop.
- **Direction:** reuse one isolate, cache compiled patterns, or evaluate with Python `re` where the syntax allows.
- **Status:** Resolved in R3 step 3 (branch `rework/r3-turn-reliability`): one shared isolate per process with cached RegExp objects (about 22× faster for 20 keys); a timed-out pattern discards and recreates the isolate; evaluation is still synchronous on the event loop, capped at 50 ms per key. The shared isolate must never evaluate user-supplied JS.

### REL-04: Unbounded and stale bot caches; webhooks listed every turn
- **Severity:** low. **Confidence:** CONFIRMED. **Label:** fragile.
- **Evidence:**
  - `channel_locks`, `webhook_locks`, `webhook_defaults` and `checked_avatar_assets` grow forever.
  - `checked_avatar_assets` never re-validates, so an asset message deleted later is still used (`discord_bot.py:184-199`).
  - `_webhook_locked` calls `parent_channel.webhooks()` for every speaker on every turn (`:168`).
- **Direction:** cache the webhook objects, add TTLs, and re-check an asset after a delivery failure.
- **Status:** Mostly resolved in R3 step 2 (branch `rework/r3-turn-reliability`): webhook objects are cached (no listing per speaker per turn), a deleted webhook is replaced and the send retried once in the same turn, and emotion-asset checks expire after 600 s. Still open (low): `channel_locks`, `webhook_locks`, `webhook_defaults`, `webhooks` and `checked_avatar_assets` are not bounded; only NotFound invalidates a cached webhook.

### REL-05: Shutdown order can skip `store.close()`
- **Status:** Resolved in R1 (branch `rework/r1-correctness`): `close()` uses try/finally so the store and parent close always run.
- **Severity:** low. **Confidence:** CONFIRMED. **Label:** fragile.
- **Evidence:**
  - `close()` awaits `models.close()` before `store.close()`, without try/finally (`discord_bot.py:77-82`).
  - Migration (`store.py:142-169`) is not one transaction. It is idempotent and backed up first, so this is acceptable.
- **Direction:** use try/finally.

## Architecture

### ARCH-01: Oversized functions
- **Severity:** medium. **Confidence:** CONFIRMED. **Label:** fragile.
- **Evidence:**
  - `register_commands` (`discord_bot.py:349-716`, about 370 lines).
  - `_run_scene` (`:205-346`).
  - `create_app` (`web.py:31-461`).
  - `mount_dashboard` (`dashboard.py:61-198`).
  - `render_lore_workspace` and `presets_panel`.
- **Why it matters:** they cannot be unit-tested in parts, and every change touches a large closure scope.
- **Direction:** extract along existing seams (command groups, scene stages, route groups) only when a roadmap phase already touches them.

### ARCH-02: Raw SQL and private store helpers in UI code
- **Status:** Resolved in R5 step 1 for `dashboard.py` and `admin.py` (store methods with a revision check on the character save). Legacy `web.py` routes keep their raw SQL until R6 (SEC-02).
- **Severity:** medium. **Confidence:** CONFIRMED. **Label:** fragile.
- **Evidence:**
  - The dashboard character save reimplements `store.update_character` with a raw `UPDATE` and calls `store._prune_character_casts` and `store._remove_from_thread_casts` (`dashboard.py:325-340`, `:335-338`).
  - Other raw `store.all/one('SELECT ...')` calls appear in `dashboard.py` (`:269`, `:286`, `:315`, `:573`, `:755`) and `admin.py:43-52`.
- **Direction:** route the logic through `Store`/`AdminStore` methods with revision checks.

### ARCH-03: Duplicated constants and logic
- **Severity:** low. **Confidence:** CONFIRMED. **Label:** preference.
- **Evidence:**
  - **Discord API base URL:** `auth.py:13`, `web.py:26`, plus literals in `avatars.py:102,129`.
  - **Administrator bit check:** `auth.py:101`, `dashboard.py:162`, `web.py:109`. The brief said 4×; 3 code sites confirmed.
  - **Guild channel fetch:** `web.py:172`, `web.py:221`, `avatars.py:102`.
  - **Recent-history loop:** in both `on_message` (`discord_bot.py:116-126`) and `/summon`.
  - **Provider branches:** repeated per method in `models.py`.
- **Direction:** consolidate when a phase touches the code.

### ARCH-04: Production-dead code and test-double fallbacks
- **Severity:** low. **Confidence:** CONFIRMED. **Label:** unconventional.
- **Evidence:**
  - `lore.retrieve_lore` (`llmcord_core/lore.py:43`), `Engine.prompt_for` (`engine.py:262`) and `cards.character_prompt` (`cards.py:97`) have no production callers. `tests/test_core.py` uses the first two.
  - `hasattr(self.models, '*_compiled')` fallbacks exist only for older fakes (`engine.py:69`, `:76`; `discord_bot.py` stream).
- **Direction:** follow the CLAUDE.md rule: grep, characterize, then delete together with the tests that pin them.

### ARCH-05: Scene deletion and expiry leave derived memory
- **Severity:** medium. **Confidence:** SUSPECTED. **Label:** incorrect.
- **Evidence:**
  - `delete_scene` and `expire_history` (`store.py:422-447`) remove nodes, evidence, traces, activations and summaries.
  - They keep personal facts, encounters and promoted lore derived from those nodes.
  - Retention docs say durable lore and opted-in facts are kept on purpose, but `/scene delete` reads like "forget this scene".
- **Direction:** decide per artifact (product), then make delete consistent.

## UX

### UX-01: Missing success feedback; wrong tab after reload; raw IDs
- **Severity:** medium. **Confidence:** CONFIRMED. **Label:** incorrect.
- **Evidence:**
  - Save cast, Save ambient, asset channel and Publish give no success toast.
  - `LiveContext.button` runs `then` only when the result is not None (`dashboard.py:43`). SUSPECTED: this is why operations returning None show nothing.
  - `?tab=prompts` is not mapped (`dashboard.py:183`), so a reload after a preset action lands on Server setup.
  - Channels and threads are shown as raw IDs (`admin.py:47-51`, `dashboard.py:573`, `:757`).
- **Direction:** a standard success notify in `LiveContext.run`, the full tab map, and channel names from the guild channel fetch.

### UX-02: Inconsistent terminology
- **Severity:** medium. **Confidence:** CONFIRMED. **Label:** preference.
- **Evidence:**
  - The UI mixes guild/server/space/world/hub/owner/book.
  - Discord says `/space allow` where the dashboard says "Link".
  - `/lore add` says "local" while `/lore promote` says "channel".
- **Direction:** a glossary in `docs/`, then rename strings in one pass.

### UX-03: Rebinding a channel silently resets cast and ambient
- **Status:** resolved in R4 step 4: same-space rebind keeps the cast; a cross-space rebind or a hub unlink prunes ineligible members and says which.
- **Severity:** medium. **Confidence:** CONFIRMED (characterization test). **Label:** incorrect.
- **Evidence:** the `bind_channel` upsert clears `default_cast`, `active_cast` and `ambient` even when rebinding to the same space (`store.py:224`).
- **Direction:** keep them when the space is unchanged; when it changes, prune ineligible characters and say so.

### UX-04: Inconsistent name matching across commands
- **Status:** resolved in R4 step 3 (shared resolver + autocomplete).
- **Severity:** low. **Confidence:** CONFIRMED (characterization test). **Label:** incorrect.
- **Evidence:** `/cast set` and `/cast default` match names case-insensitively. `/cast add`, `/cast remove`, `/summon` and `/character info` need an exact match.
- **Direction:** one resolver, with autocomplete.

### UX-05: Lorebook and avatar flows split across tabs and steps
- **Severity:** low. **Confidence:** CONFIRMED. **Label:** preference.
- **Evidence:**
  - Books are created under Imports but edited under Lore.
  - An avatar upload needs a separate Save.
  - Delete confirmations differ between panels.
- **Direction:** part of the dashboard UX phase.

### UX-06: Admin commands visible to all; inconsistent option names
- **Status:** admin visibility resolved in R4 step 2b (`/admin` group, D11). Option names made consistent in R4 step 3; `/context` jargon pending (R5 wording).
- **Severity:** low. **Confidence:** CONFIRMED. **Label:** preference.
- **Evidence:**
  - Admin commands are visible to everyone (see SEC-05).
  - Option names vary: `name`, `character_name`, `space_name`, `hub_name`.
  - `/context` output uses internal jargon.
- **Direction:** fix with SEC-05.

### UX-07: Raw and ambiguous error text
- **Status:** resolved in R4 step 1 (D3): generic public/ephemeral messages with a logged reference ID; UNIQUE conflicts say "already exists"; `/memory forget` and `/lore delete` report definite outcomes.
- **Severity:** medium. **Confidence:** CONFIRMED (characterization tests). **Label:** incorrect.
- **Evidence:**
  - Users see "UNIQUE constraint failed: ..." and "Command failed: KeyError: ..." (`discord_bot.py:703-712`).
  - Some messages are ambiguous: "Memory removed if it belonged to you", "deleted if present".
- **Direction:** map `IntegrityError` to "already exists", give a generic message plus a log id for the rest, and use definite outcomes.

### UX-08: Public cost footer and public "no character" message
- **Status:** resolved in R4 step 5 (D2, D12): per-server footer setting (default on); the no-character hint deletes itself after 15 s.
- **Severity:** low. **Confidence:** CONFIRMED. **Label:** preference.
- **Evidence:**
  - Every reply carries a model/usage footer (`discord_bot.py:297-310`).
  - "No active character…" is posted to the channel (`:247`).
- **Direction:** product decision (see the roadmap).

### UX-09: `/scene delete` removes the whole root tree
- **Status:** resolved in R4 step 5c (D5): deletes the given message's subtree only and reports the count.
- **Severity:** medium. **Confidence:** CONFIRMED. **Label:** fragile.
- **Evidence:**
  - `delete_scene(guild, root_id)` deletes every node under the root, including other users' branches (`store.py:422-431`).
  - The Discord messages stay.
- **Direction:** product decision on scope; at minimum, confirm with a count.

## Tooling

### TOOL-01: Tooling, test-helper and docs gaps
- **Severity:** low. **Confidence:** CONFIRMED. **Label:** fragile.
- **Evidence:**
  - Before bootstrap there was no lint or type tooling. Since bootstrap: `ruff.toml` (bug-class rules) and `scripts/verify.sh`. CI runs ruff since bootstrap step 3 (`250e9be`).
  - Test helpers are duplicated: two `FakeModels`, `Settings` built inline in 4 files, and test files importing each other. `tests/helpers.py` now exists for new tests.
  - The NiceGUI dashboard is covered only by the opt-in browser test.
  - Docs are stale: `docs/verification.md` says "Python 3.12", and the `docs/architecture.md` module table missed usage/identity/errors/scene_ui/lore_workspace/lore_drag. Fixed in this bootstrap.
- **Direction:** (ruff in CI: done in `250e9be`.) Migrate tests to the helpers opportunistically.
