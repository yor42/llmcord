# Dashboard performance baseline

Recorded 2026-10-07 at commit `1270c09` (branch `bootstrap/engineering`), before any performance work. The numbers belong to finding **PERF-01** in the [hardening audit](history/audit-2026-10-07.md).

## How to reproduce

```bash
scripts/verify.sh --bench                               # unit tests + bench (all profiles)
.venv/bin/python scripts/bench_dashboard.py --profile all --seed 10,500 --json out.json
```

The bench starts `tests/dashboard_server.py` in benchmark mode:
- mocked Discord;
- an in-memory DB seeded with 10 extra characters and 500 lore entries;
- localhost HTTPS with a throwaway certificate.

It drives the server with headless Chromium. Never point it at real Discord or `data/`.

**Columns**
- `ready`: the time until Playwright's action and wait return.
- `settled`: the time until the fixture's Discord call counter has been unchanged for 1.5 s. It therefore includes server work that continues after the browser looks done.
- `checks`: the number of `GET /users/@me/guilds` calls, i.e. `AuthService.guard` → Discord.

**Profiles**

| Profile | Mock Discord behavior |
| --- | --- |
| `latency` | 150 ms per call, never rate limited. |
| `ratelimited` | 150 ms per call, and every user-guild response carries `X-RateLimit-Remaining: 0` and `X-RateLimit-Reset-After: 1`. **The 1 s reset is an assumption about Discord's per-token bucket, not a measurement.** Real behavior may be better or worse. |

## Host and conditions

- **Hardware:** Raspberry Pi 5 Model B Rev 1.0, 4 cores, 4 GiB RAM.
- **Software:** Linux 6.18.33+rpt-rpi-2712 aarch64 (glibc 2.41). Python 3.13.5 (`.venv`). Chromium headless via Playwright 1.63.0.
- **Load:** the host was otherwise idle (load average 0.2–0.3). The development session was open, and nothing else was benchmarking.
- **Thermal:** SoC at 59–61 °C at the start of each run. `vcgencmd get_throttled` = `0x50000`: under-voltage and throttling *had occurred* since boot, but neither was active during the runs.
- **Duration:** run 1 started at 19:52:25 KST and took 78 s wall. Run 2 started at 19:53:43 and took 76 s.

## Results (seed `10,500`)

Two consecutive runs. Times in seconds.

### `latency`

| Scenario | Run 1 ready / settled | Run 2 ready / settled | Checks | Other Discord calls |
| --- | --- | --- | --- | --- |
| Guild page cold load | 2.43 / 2.44 | 2.17 / 2.17 | 4 | 1 channels |
| Switch to Characters tab | 0.07 / 0.07 | 0.08 / 0.09 | 1 | — |
| Expand one character | 0.57 / 0.57 | 0.48 / 0.48 | 1 | — |
| Type 10 chars in lore search | 1.64 / 3.69 | 1.65 / 3.70 | 21 | — |
| Click Save cast | 0.70 / 0.91 | 0.71 / 0.92 | 3 | — |

### `ratelimited`

| Scenario | Run 1 ready / settled | Run 2 ready / settled | Checks | Other Discord calls |
| --- | --- | --- | --- | --- |
| Guild page cold load | 3.22 / 3.94 | 3.15 / 3.87 | 4 | 1 channels |
| Switch to Characters tab | 0.07 / 0.08 | 0.07 / 0.08 | 1 | — |
| Expand one character | 0.55 / 0.56 | 0.56 / 0.56 | 1 | — |
| Type 10 chars in lore search | 1.65 / **23.14** | 1.63 / **23.17** | 21 | — |
| Click Save cast | 0.71 / 2.36 | 0.70 / 2.35 | 3 | — |

The pre-bootstrap run, with the same seed and host, gave: cold load 2.17 s (latency) and 3.93 s settled (rate limited); lore search 21 checks, settled 3.73 s and 23.17 s; Save cast 3 checks, settled 2.43 s (rate limited). Today's runs agree to within about 0.1 s. Call counts are identical.

## Reading the numbers

- **Run-to-run variation:**
  - Cold load varies by about 0.3 s between runs.
  - Every other value varies by 0.1 s or less.
  - Check counts are deterministic.
- **Lore search:**
  - The browser is "ready" after about 1.6 s, but the server keeps working through 21 serialized guild checks: about 2 per keystroke (the event plus `ctx.run`), plus the initial tab switch.
  - With a 1 s reset per check, that becomes a 23 s queue. The user sees results late, and other events from the same session are delayed behind it.
- **Save cast:**
  - Makes 3 checks: the click event, `AdminService.run`, and one more event/handshake.
  - Costs 2.4 s under rate limiting for a single trivial write.
- **Cold load:**
  - Makes 4 checks: the page route, the socket connect, the handshake, and the first event.
  - Makes 1 bot-token channels call.
- **Not measured here:**
  - avatar image GETs, because the seed has no avatar images. In production each image would add a check;
  - real Discord rate-limit behavior;
  - Tailscale latency.

## Updating this baseline

A change claiming a dashboard performance improvement must re-run the bench with the same seed on the same host. It then adds a new dated section here, rather than overwriting this one, and updates any count-based characterization tests in `tests/test_web_auth_boundaries.py` in the same change (see `tests/CLAUDE.md`).

## 2026-10-07: R2 step 1, guild-list cache (count tests only, bench pending)

`AuthService.guilds(session)` caches `/users/@me/guilds` per session for 300 s (D1). The count-based tests changed deliberately:
- `test_web_auth_boundaries.py`: 5 avatar GETs went from 5 guild checks to 1 (`test_repeated_avatar_requests_check_discord_once`). The `no-store` header is unchanged until R2 step 4.
- `test_auth_cache.py` (new): repeat guards within the TTL cost 1 call, concurrent cold guards cost 1, the picker page plus a guild page cost 1, and guards queued behind a 401 cost 1. (R6 step 2 retired the legacy guild page with SEC-02, so the picker-plus-guild-page test and `test_legacy_guild_page_cost` were removed.)

The bench has not been re-run for this step. The dated bench section (two runs, same seed and host) comes at the end of R2, after the debounce and probe removal (steps 2–3), which also affect the lore-search numbers.

R2 step 3 (same day): the lore search box is debounced (300 ms). The browser test `test_lore_search_is_debounced` counts `AdminStore.admin_entries` calls through a test-only wrapper in `tests/dashboard_server.py`: a 10-character burst at 30 ms per key gives 2 calls (1 board render), down from 20 (10 renders). The bench server now runs through that counting wrapper too; its overhead is negligible.

## 2026-10-07: end of R2 (steps 1–6 + logout follow-up)

Same host, same seed (`10,500`), two consecutive `scripts/verify.sh --bench` runs on `rework/r2-auth-perf` at `df8260f`. Times in seconds; "Checks" = `GET /users/@me/guilds` calls.

### `latency`

| Scenario | Run 1 ready / settled | Run 2 ready / settled | Checks (was) | Other Discord calls |
| --- | --- | --- | --- | --- |
| Guild page cold load | 2.16 / 2.16 | 2.02 / 2.02 | 1 (4) | 1 channels |
| Switch to Characters tab | 0.55 / 0.55 | 1.36 / 1.36 | 0 (1) | — |
| Expand one character | 0.04 / 0.05 | 0.05 / 0.06 | 0 (1) | — |
| Type 10 chars in lore search | 1.54 / 1.54 | 1.50 / 1.50 | 0 (21) | — |
| Click Save cast | 0.55 / 0.55 | 0.56 / 0.56 | 0 (3) | — |

### `ratelimited`

| Scenario | Run 1 ready / settled | Run 2 ready / settled | Checks (was) | Other Discord calls |
| --- | --- | --- | --- | --- |
| Guild page cold load | 2.01 / 2.01 | 2.02 / 2.02 | 1 (4) | 1 channels |
| Switch to Characters tab | 0.56 / 0.57 | 0.54 / 0.55 | 0 (1) | — |
| Expand one character | 0.04 / 0.04 | 0.04 / 0.04 | 0 (1) | — |
| Type 10 chars in lore search | 1.51 / 1.52 | 1.50 / 1.50 | 0 (21) | — |
| Click Save cast | 0.55 / 0.55 | 0.60 / 0.60 | 0 (3) | — |

Reading:
- Guild checks per session dropped to 1 on cold load and 0 for every later interaction within the 300 s TTL (D1). The rate-limited profile now matches the latency profile, since nothing waits on Discord after the first check.
- Lore search settled: 23.1 s → 1.5 s (rate limited), 3.7 s → 1.5 s (latency). The remaining time includes the 300 ms debounce.
- Save cast settled: 2.36 s → 0.55–0.60 s (rate limited).
- Cold load (rate limited): 3.9 s → 2.0 s settled.
- The Characters tab and Expand one character moved in opposite directions (tab ~0.07 → ~0.55, expand ~0.55 → ~0.05). Their sum is about the same, so the render work seems to have shifted from the expand step to the tab switch; this was not investigated. The latency run 2 tab value (1.36 s) is a single outlier; the other three runs are 0.54–0.56 s.

## 2026-10-08: R5 step 2 (PERF-02, lore paging in SQL)

One `scripts/verify.sh --bench` run on `rework/r5-dashboard-data` before the single-pass follow-up (same host and seed). Every scenario is within ±0.05 s of the end-of-R2 runs: lore search 1.49 / 1.49 (latency) and 1.51 / 1.51 (rate limited), 0 checks. The bench seeds 500 entries, and at that size the 300 ms debounce and page render dominate. The end-to-end bench therefore cannot show this change.

The store-level micro-benchmark used a throwaway script with no repo code changes and an in-memory `Store`. It measured one character owner and a 50-entry page, with the mean of 5 calls, in ms. "Before" is `cec5002`, which loaded every entry through `admin_entries` (N+1) and then filtered and sliced in Python. "After" is `admin_entries_page`.

| Entries | Query | Before | After |
| --- | --- | --- | --- |
| 500 | none | 15.3 | 1.8 |
| 500 | hit (`key12`) | 15.6 | 6.1 |
| 500 | no match | 15.5 | 11.6 |
| 5000 | none | 173.9 | 6.5 |
| 5000 | hit (`key12`) | 173.4 | 62.5 |
| 5000 | no match | 178.2 | 61.0 |

Reading:
- An unfiltered page no longer grows with book size beyond one indexed COUNT.
- A filtered page is still linear, because the casefold match runs as a Python SQL function on every row of the owner. It runs once per row, since the count comes from a window function in the same query.
- FTS5 would make filtered search sub-linear, but its tokenization would change the substring semantics. It is not needed at current sizes.

## 2026-10-08: end of R5 (steps 1–8)

Same host and seed (`10,500`), two consecutive runs on `rework/r5-dashboard-data` at `2d5ebf7`: run 1 is `scripts/verify.sh --bench` (14:12 KST, 62 s wall), run 2 is `scripts/bench_dashboard.py` alone (14:13 KST, 47 s wall). Load average 0.2–0.6, SoC 59 °C, `get_throttled` = `0x50000` (nothing active), as before. Times in seconds; "Checks" = `GET /users/@me/guilds` calls. "End of R2" is the mean of its two runs.

### `latency`

| Scenario | Run 1 ready / settled | Run 2 ready / settled | End of R2 settled | Checks | Other Discord calls |
| --- | --- | --- | --- | --- | --- |
| Guild page cold load | 1.39 / 1.39 | 1.36 / 1.36 | 2.09 | 1 | 1 channels |
| Switch to Characters tab | 0.06 / 0.31 | 0.06 / 0.32 | 0.96 | 0 | — |
| Expand one character | 0.31 / 0.31 | 0.31 / 0.31 | 0.06 | 0 | — |
| Type 10 chars in lore search | 1.68 / 1.68 | 1.71 / 1.71 | 1.52 | 0 | — |
| Click Save cast | 0.64 / 0.67 | 0.62 / 0.63 | 0.56 | 0 | — |

### `ratelimited`

| Scenario | Run 1 ready / settled | Run 2 ready / settled | End of R2 settled | Checks | Other Discord calls |
| --- | --- | --- | --- | --- | --- |
| Guild page cold load | 1.38 / 1.38 | 1.38 / 1.38 | 2.02 | 1 | 1 channels |
| Switch to Characters tab | 0.07 / 0.32 | 0.07 / 0.32 | 0.56 | 0 | — |
| Expand one character | 0.29 / 0.30 | 0.35 / 0.35 | 0.04 | 0 | — |
| Type 10 chars in lore search | 1.69 / 1.69 | 1.68 / 1.69 | 1.51 | 0 | — |
| Click Save cast | 0.63 / 0.64 | 0.60 / 0.61 | 0.58 | 0 | — |

Reading:
- Cold load: 2.0–2.1 s → 1.4 s settled in both profiles. Since step 6 the page builds only the Server setup panel at load.
- The target metric, lore search settled, went **up** by about 0.17 s (1.5 → 1.7 s). The scenario clicks the Lore tab inside the measurement, and since step 6 that click builds the Lore panel (500 entries, first page) instead of the cold load doing it. This attribution follows from the code and the matching drop in cold load; the panel build was not timed on its own. Discord checks stay at 0, so lore search no longer waits on Discord at all (it was 21 checks and 23 s at the original baseline).
- The Characters tab now settles in 0.31 s (lazy build), and expanding a character takes 0.30 s instead of 0.05 s. Each expanded card now renders its emotion editor and confirm dialogs; this was not profiled.
- Save cast is 0.05–0.10 s slower, within the spread of the earlier runs plus the new success toast.
- Summed over the five scenarios (settled): 4.4 s now in both profiles, vs 5.2 s (latency; 4.8 s without the 1.36 s tab outlier) and 4.7 s (rate limited) at the end of R2.
- The 500-entry seed is too small to show PERF-02 end to end; see the store micro-benchmark in the step 2 section. PERF-04 (gzip, cached assets) cannot show here either: the bench runs on localhost, where transfer size barely matters, and each run starts with an empty browser cache.

Note (after the end-of-R5 runs): the fixture's uvicorn keep-alive went from 5 s to 120 s to fix a browser-test flake. The bench uses the same fixture, so later runs reuse connections longer than the runs above. Expect a small difference at most; re-run before comparing a later perf claim against these numbers.

Note (UI-15, 2026-10-09): the "Click Save cast" scenario no longer exists; Server setup saves through the save bar. `scripts/bench_dashboard.py` now runs **Save channel settings** (open the first channel card, toggle Ambient participation, click **Save changes**, wait for the "Channel settings saved" toast). It measured 1.24 s ready and settled in both profiles, but it includes the card-open round trip and a different write. It is not comparable with the Save cast rows above; use it as the new baseline for this step. Same run: cold load 1.44 s, Characters tab 0.41 / 0.38 s settled, expand one character 0.42 / 0.38 s, lore search 1.88 s (log not kept in the repo).

## 2026-10-09: UI-10 (Lore panel build timed on its own; character expand profiled)

`scripts/bench_dashboard.py` now has an "open Lore tab (panel build)" scenario, and "type 10 chars in lore search" starts from the open panel. One `scripts/verify.sh --bench` run, same host and seed (`10,500`). Times are ready / settled, in seconds.

| Scenario | `latency` | `ratelimited` |
| --- | --- | --- |
| Guild page cold load | 1.45 / 1.45 | 1.44 / 1.44 |
| Switch to Characters tab | 0.08 / 0.42 | 0.06 / 0.38 |
| Expand one character | 0.42 / 0.42 | 0.38 / 0.38 |
| Open Lore tab (panel build) | 0.98 / 0.98 | 0.98 / 0.98 |
| Type 10 chars in lore search | 0.91 / 0.92 | 0.95 / 0.96 |
| Save channel settings | 1.25 / 1.25 | 1.27 / 1.28 |

Reading:
- The panel build plus the search (about 1.9 s) matches the old combined lore-search figure (1.7–1.9 s).
- Character expand is not server work.
  - Over 4 clicks, cProfile measured 13 ms of server CPU, about 1 ms per event (guard 0.1–0.2 ms, emit 0.4 ms).
  - The server sends its 440-byte update within 1 ms of the event. The browser receives it 155–175 ms later, with or without mock Discord latency. The cause of that gap is unknown.
  - In the browser, script time is 1–2 ms and layout/style 8–17 ms. The rest is the 300 ms Quasar expand animation, which Playwright waits out before it treats the element as stable.
- Part of the expand figure is bench overhead:
  - Resolving the `filter(has_text=…)` locator the first time costs 0.2–0.3 s.
  - A click right after another click waits for the previous animation.
  - Whether the 0.06 → 0.31 s change at the end of R5 comes from the dashboard or from Playwright is still open. See UI-48.

## 2026-10-10: UI-48 (character expand attributed)

`scripts/bench_dashboard.py` keeps the **Expand one character** row unchanged and adds **expand one character (split)**. That row collapses the card, waits 1 s, then clicks it from inside the page and times each step with `performance.now()`; its own ready time includes the 1 s wait, so do not compare it with the other row. An init script logs socket frames and long tasks. One run per profile, same host and seed (`10,500`). Times are milliseconds after the click.

| Step | `latency` | `ratelimited` |
| --- | --- | --- |
| Locator resolve (before the click) | 35 | 26 |
| Socket message sent | 0.3 | 0.3 |
| 440-byte update received by the page | 2.5 | 2.4 |
| Main-thread long task (start → end) | 2 → 179 | 2 → 186 |
| First DOM change | 169 | 176 |
| Height starts changing | 289 | 307 |
| Animation stable | 589 | 591 |

Reading:
- The 155–175 ms "delivery gap" from UI-10 is not delivery. The page has the update after about 2 ms over a single websocket (no long-polling); NiceGUI's outbox sends on an event, not on a timer.
- The browser then spends 160–235 ms (varies by run) in one main-thread task handling that update; nothing reaches the DOM until it ends. A CPU profile puts most of it under Vue's render/patch frames (about 210 ms inclusive, profiler-inflated), which fits the expanded card building its 7 text areas, select and checkbox only on expand; the split between Vue component setup and native DOM/style work was not measured. Stubbing out the Tailwind runtime did not shorten it (174 ms).
- The bench clicks from inside the page, skipping Playwright's actionability waits; UI-10's 155–175 ms figure was measured at the first DOM change, so it was this task, not delivery.
- After that comes the 300 ms Quasar expand animation. The Tailwind runtime runs during the animation, but blocking it saved only about 20 ms.
- Not split further: native DOM, style and component setup inside the long task (about 340 ms of `(program)` samples, inflated by the profiler).
- A lighter card body at expand time could save up to about 170 ms; see UI-51.
