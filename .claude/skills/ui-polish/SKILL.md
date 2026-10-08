---
name: ui-polish
description: Plan, brief and accept a llmcord UI polish item (dashboard tabs, labels, toasts, dialogs, layout, or Discord reply wording) with before/after evidence. Use for any UI-NN backlog item or a user request to improve how the dashboard or bot replies look or read.
---

# UI polish

UI items run through `orchestrate-change`; this skill adds what is specific to them. Conventions: `docs/engineering/ui-guide.md`. Words: `docs/glossary.md`. Items: `docs/engineering/backlog.md` (`UI-NN`).

## 1. Frame the item
- Name the surface (tab/panel/dialog, or slash command) and the exact before state: screenshot, or the reply text quoted from code (`path:line`).
- Write the target as concrete text and layout ("button 'Save' → 'Save cast'; toast 'Cast saved.'"), not as "make it nicer". If the target is a judgement call (wording, layout choice, a new convention), show the user two options and let them pick before briefing.
- Check it is presentation only. If it needs a store, guard, audit, consent or write-path change, split off that part as an `MNT` item first.
- Grep `tests/` for the current text: list the browser and slash-command tests that select it. Those updates go to `test-writer` in the same item.

## 2. Brief `ui-polisher`
Use the `orchestrate-change` brief plus: the target text/layout, the tests that select it, which user docs may be edited, and where to save screenshots (under `$CLAUDE_JOB_DIR/tmp` or another path outside the repo).

## 3. Evidence
- Dashboard: before and after screenshots at 1400 × 1000 and about 390 px wide. Use the screenshot script from UI-01 once it exists. Until then the worker drives the browser fixture the way `tests/test_dashboard_browser.py` does: start `python -m tests.dashboard_server` with `LLMCORD_TEST_PORT`, `LLMCORD_TEST_KEY`, `LLMCORD_TEST_CERT` (a throwaway `trustme` pair in a temp dir, as `setUpClass` does), add the `llmcord_session=browser-test-session` cookie, open `/admin/guild/1?tab=<name>`, and stop it with `POST /_test/stop`. It uses mock Discord and an in-memory DB only.
- Discord: exact before/after strings.
- Look at the screenshots yourself (Read the PNGs) before accepting; check spacing, wrapping at phone width, truncated labels, and that nothing else on the page moved.

## 4. Verify and review
- Dashboard files: `scripts/verify.sh --browser`. Reply wording only: `scripts/verify.sh`. A layout change that could cost time on tab build: add `--bench` and compare with `docs/engineering/perf-baseline.md`.
- `change-reviewer` gets the diff, evidence paths and the UI checklist item; Sonnet is enough unless the item touched a guarded path.

## 5. Close
Mark the item done in the backlog, add any issues the worker noticed as new `UI-NN` items, update the UI guide if a convention changed, and update user docs that name changed text.
