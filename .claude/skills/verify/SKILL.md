---
name: verify
description: Run and interpret llmcord's verification suite (lint, repo hygiene, offline unittest, Playwright dashboard suite, perf bench). Use before declaring any change done, or when checks fail and you need to classify why.
---

# Verify llmcord

## Choose the level
| Changed | Run |
| --- | --- |
| docs / `.claude/` only | `git diff --check`; parse settings JSON and agent/skill YAML; `claude plugin validate .claude/agents --strict` and `claude plugin validate .claude/skills --strict`; inspect scope/model routing |
| bot, engine, store, models, prompts, world_info, cards, lorebooks | `scripts/verify.sh` |
| web.py, auth.py, admin*.py, dashboard.py, scene_ui.py, lore_workspace.py, lore_drag.*, tests/dashboard_server.py, tests/test_dashboard_browser.py, a UI screenshot script | `scripts/verify.sh --browser` |
| Discord reply wording only | `scripts/verify.sh` (slash-command tests pin the text) |
| a performance claim | `scripts/verify.sh --bench` (compare with `docs/engineering/perf-baseline.md`, see the `dashboard-perf` skill) |

Changes to other executable scripts or tests require the default suite; browser/perf fixtures require the corresponding additional level. Use `scripts/verify.sh --browser --bench` when both apply. A `.claude/` hook script change also needs its syntax check and a controlled fixture invocation; metadata validation alone is insufficient.

During orchestration, `verify-runner` owns the final full run and saves verbose output outside the repo. The main session checks the actual evidence and final diff; workers run targeted checks and the reviewer consumes evidence. Reuse passing evidence only while the checked state is unchanged. No additional agent is needed just to invoke a short local validation command.

## Read the results
- The unittest summary looks like `OK (skipped=1, expected failures=N)`. `skipped=1` is the browser test in default mode. **N must equal the number of `test_known_defect_*` tests** (`grep -rc "def test_known_defect_" tests/`).
- `unexpected successes` means a known defect was fixed: remove its `@unittest.expectedFailure` and mark the item done in `docs/engineering/backlog.md`, or find out why it passes.
- Noise you can ignore: discord.py `DeprecationWarning: 'count' is passed as positional argument`, `StarletteDeprecationWarning ... httpx2`, and logged `ERROR:root:Scene failed ...` lines from failure-path tests.
- Browser suite failure: look in `.test-artifacts/` (failure.png, server.log, browser-errors.json, state.json).

## Classify a failure before acting
1. **Regression from the change:** fix in the change.
2. **Environment:** e.g. Playwright browser missing (`.venv/bin/python -m playwright install chromium`) or the venv missing deps (`.venv/bin/pip install -r requirements-dev.txt`).
3. **Flaky:** rerun once. If it passes, note it; don't ignore a second failure.
4. **Pre-existing:** confirm using the recorded baseline or an isolated checkout of the intended base. Preserve dirty user work; do not stash/reset it merely to classify a failure.

Never skip, delete, or loosen a test to get green. Report failures verbatim.

## CI parity
CI (`.github/workflows/ci.yml`) runs the unittest suite on 3.12 and 3.13, the browser suite, gitleaks with `scripts/check_repository.py`, and arm64/amd64 container smoke. Local `verify.sh` covers everything except gitleaks and the container build.
