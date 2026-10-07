---
name: verify
description: Run and interpret llmcord's verification suite (lint, repo hygiene, offline unittest, Playwright dashboard suite, perf bench). Use before declaring any change done, or when checks fail and you need to classify why.
---

# Verify llmcord

## Choose the level
| Changed | Run |
| --- | --- |
| docs / `.claude/` only | nothing (or `ruff` if scripts changed) |
| bot, engine, store, models, prompts, world_info, cards, lorebooks | `scripts/verify.sh` |
| web.py, auth.py, admin*.py, dashboard.py, scene_ui.py, lore_workspace.py, lore_drag.*, templates, tests/dashboard_server.py | `scripts/verify.sh --browser` |
| a performance claim | `scripts/verify.sh --bench` (compare with `docs/engineering/perf-baseline.md`, see the `dashboard-perf` skill) |

## Read the results
- The unittest summary looks like `OK (skipped=1, expected failures=N)`. `skipped=1` is the browser test in default mode. **N must equal the number of `test_known_defect_*` tests** (`grep -rc "def test_known_defect_" tests/`).
- `unexpected successes` means a known defect was fixed: remove its `@unittest.expectedFailure` and update audit.md, or find out why it passes.
- Noise you can ignore: discord.py `DeprecationWarning: 'count' is passed as positional argument`, `StarletteDeprecationWarning ... httpx2`, and logged `ERROR:root:Scene failed ...` lines from failure-path tests.
- Browser suite failure: look in `.test-artifacts/` (failure.png, server.log, browser-errors.json, state.json).

## Classify a failure before acting
1. **Regression from the change:** fix in the change.
2. **Environment:** e.g. Playwright browser missing (`.venv/bin/python -m playwright install chromium`) or the venv missing deps (`.venv/bin/pip install -r requirements-dev.txt`).
3. **Flaky:** rerun once. If it passes, note it; don't ignore a second failure.
4. **Pre-existing:** confirm on a clean `git stash` base before claiming this.

Never skip, delete, or loosen a test to get green. Report failures verbatim.

## CI parity
CI (`.github/workflows/ci.yml`) runs the unittest suite on 3.12 and 3.13, the browser suite, gitleaks with `scripts/check_repository.py`, and arm64/amd64 container smoke. Local `verify.sh` covers everything except gitleaks and the container build.
