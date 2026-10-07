---
name: dashboard-perf
description: Measure llmcord dashboard performance with the mocked-Discord Playwright bench and compare against the recorded baseline. Use when investigating dashboard slowness or before/after any change claimed to improve it.
---

# Dashboard performance

## Run
```
.venv/bin/python scripts/bench_dashboard.py --profile all --seed 10,500 --json /path/outside/repo/bench.json
```
- Profiles:
  - `latency`: mock Discord at 150 ms, no rate-limit waits.
  - `ratelimited`: the same, plus `X-RateLimit-Remaining: 0` and `Reset-After: 1 s` on user-guild checks. This rate limit is an assumption that mimics Discord, not a measurement.
- `--seed N,M` adds N characters (each with avatars) and M channel lore entries.
- Each scenario prints `ready_s`, `settled_s` and `discord_calls`.
  - `ready_s` is the time until the browser finishes the action.
  - `settled_s` is the time until the server's Discord call counter has been quiet for 1.5 s. It catches queued work hidden behind the UI.

## Compare
- The baseline numbers and conditions are in `docs/engineering/perf-baseline.md`.
- Compare the same profile, seed and host, and run each side at least twice.
- Discord call counts are deterministic. Treat any change in them as a real result.
- Timings on a Pi vary by about ±15%. Claim a timing win only when it beats that margin on both runs.

## Interpret
- The known dominant cost is PERF-01: every socket event, `ctx.run`, page load and avatar request re-runs `AuthService.guard`, which calls Discord `/users/@me/guilds`. These calls are serialized per token and wait out the rate limit.
- Under `ratelimited`, `settled_s` grows linearly with `user_guild_checks`.
- Offline counters live in `tests/test_web_auth_boundaries.py`. If a change alters Discord call counts, update those assertions and perf-baseline.md in the same change.
- To investigate a new scenario, add it to `run_profile` in `scripts/bench_dashboard.py` using `measure(name, action)`. Keep scenarios deterministic, and use only the fixture's mock Discord.

## Limits
- The bench uses fake Discord and an in-memory DB. It does not capture real Discord latency, Tailscale overhead or SQLite contention with the live bot.
- For field evidence, ask the user for browser devtools timings from the real deployment. Do not instrument production without approval.
