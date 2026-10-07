# Dashboard performance baseline

Recorded 2026-10-07 at commit `1270c09` (branch `bootstrap/engineering`), before any performance work. The numbers belong to finding **PERF-01** in `audit.md`.

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
