---
name: orchestrate-change
description: The required workflow for making any code change in llmcord — orchestrator briefs test-writer, implementer, and change-reviewer subagents and verifies between steps. Use whenever a roadmap item, finding ID (BUG-/PERF-/SEC-/UX-/ARCH-), or bug fix needs code edits.
---

# Orchestrate a change

The main session coordinates and does not edit `llmcord_core/`, entry points, or tests itself.

## 0. Preconditions
- The item comes from `docs/engineering/roadmap.md` (or the user named it), and its roadmap phase is approved by the user.
- `git status` is clean or only has expected work. `scripts/verify.sh` passes on the base (record expected-failure count).

## 1. Write the brief (in your own context, ~10 lines)
```
ID / title:
Goal (observable outcome):
Files in scope:
Out of scope / must not change:
Tests that must flip (known-defect IDs) or stay green:
User-visible changes allowed:
Verification level: default | --browser | --bench
```
If the item needs a product decision (see "Open product decisions" in roadmap.md), ask the user before continuing.

## 2. Pin behavior → `test-writer`
Send the brief. Skip only if an existing test already pins the exact behavior (name it in the brief). Check the report: are the tests observable-behavior tests, and was each expected-failure validated?

## 3. Implement → `implementer`
Send the brief plus the test names from step 2. One item per implementer run. For independent items, use separate runs (in a worktree via `isolation: "worktree"` if they touch the same files).

## 4. Verify yourself
Re-run `scripts/verify.sh` (with the brief's level). Don't rely on the agent's report alone. Compare the expected-failure count: it should drop by exactly the defects fixed.

## 5. Review → `change-reviewer`
Send the brief. On **request changes**, send the findings back to `implementer` (max 2 rounds, then escalate to the user).

## 6. Report to the user
ID, what changed (files), user-visible changes, verification output summary, reviewer verdict, and follow-ups. Update the item's status in `docs/engineering/roadmap.md` and, if it was a finding, mark it in `audit.md`. Commit only when the user asks.
