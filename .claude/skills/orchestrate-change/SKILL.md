---
name: orchestrate-change
description: Required workflow for llmcord code changes (maintenance and UI polish items). Routes bounded work to Haiku, behavioral and UI implementation/tests to Sonnet, and difficult reasoning to Opus; verifies and reviews the final diff.
---

# Orchestrate a change

The main session coordinates and does not edit `llmcord_core/`, entry points, scripts, or tests itself. Delegated workers execute their brief directly; they must not restart this workflow.

## 0. Establish scope and baseline
- Work within the user's request or a backlog item (`docs/engineering/backlog.md`, `MNT-NN`/`UI-NN`). A direct request authorizes its scoped change. Anything larger than one small item, any schema change, and any change to a `CLAUDE.md` invariant needs the user's go-ahead first. Work found along the way becomes a new backlog item, not scope creep.
- Inspect `git status` and the existing diff. Preserve unrelated/pre-existing edits; record a starting snapshot and include new files. Never stash or reset the user's work to obtain a clean base.
- For code changes, establish a passing `scripts/verify.sh` baseline once per unchanged starting state or approved batch. Record skips and expected failures. Use `verify-runner`; reuse that evidence until the base changes.
- Read docs/config directly. For a simple lookup, search directly; for noisy bounded investigation, use `Explore`. Broad debugging or architecture belongs to Sonnet/Opus.

## 1. Write a brief (~10 lines)
```
Backlog ID / title (or user request):
Goal (observable outcome):
Files in scope / starting snapshot:
Out of scope / behavior to preserve:
Existing coverage / tests needed / known defects to fix:
User-visible changes allowed (UI items: exact text/layout, evidence paths):
Risk and reason:
Agents / model choices and reason:
Targeted checks / final level: default | --browser | --bench | both
Evidence/log paths outside the repo:
```
Resolve necessary product decisions before dependent work; continue independent work while waiting.

## 2. Choose the smallest adequate route
| Work | Route |
| --- | --- |
| Docs or Claude config only | Main session edits; metadata/diff validation via `verify` |
| Exact low-risk replacement/format/syntax fix, covered behavior | `quick-editor` (Haiku) → final verification → `change-reviewer` (Sonnet) |
| Behavior change, bug fix, refactor, dependency/tooling fix (`MNT-*`) | `test-writer` when needed (Sonnet) → `implementer` (Sonnet) → final verification → `change-reviewer` (Sonnet) |
| Presentation-only UI polish (`UI-*`: layout, labels, wording, toasts, dialogs) | `ui-polisher` (Sonnet, before/after evidence) → `test-writer` for tests that select renamed text → `verify-runner` `--browser` (dashboard) → `change-reviewer` (Sonnet, UI checklist). Follow the `ui-polish` skill |
| UI item that needs data, guard, audit or write-path changes | Split it: the data part goes the `MNT` route first, then the presentation part |
| Auth/security, isolation, consent, migration, concurrency, or unresolved cross-module reasoning | Sonnet workers; Opus for the difficult reasoning/review, with a reason in the brief |

- Every custom agent has a `model`; never set `inherit` or override all invocations to the parent's model. Leave the invocation's model unset to use the role default; supply an explicit override only for a justified escalation.
- Haiku may write a narrowly specified test by overriding `test-writer` only when the assertion, seam, and existing test pattern are supplied. It must not determine intended behavior or review security invariants on its own.
- Use `Explore` for bounded inventories and sanitized summaries. Run repetitive transformations with a deterministic script/command when possible; use Haiku to summarize exceptions. Do not spawn one agent per row/file/query.
- Built-in Plan still inherits: if used, explicitly select Sonnet for routine planning or Opus for difficult reasoning. Generic agents also need an intentional task/model choice. Do not use forks, agent teams, or background swarms by default.

## 3. Pin and implement
- Skip `test-writer` when existing tests pin the exact behavior (name them), or when no executable behavior changes. Otherwise supply the brief and inspect its behavioral tests and expected-failure evidence.
- Send `implementer` the brief and exact test names. Use `quick-editor` only when its eligibility rules hold. One scoped item per worker; workers run targeted checks, not repeated full suites.
- Default to sequential work. Parallelize only independent tasks with separate file ownership and a clear benefit. If isolated worktrees are needed, explicitly select the intended base, include required existing changes, and integrate all worker edits before final verification; dependent test/implementation stages share one checkout.
- Respect `maxTurns`. Partial output requires inspection and a narrower brief, one justified continuation, or escalation; do not blindly resume to bypass the limit. Haiku uncertainty goes to Sonnet; unresolved Sonnet reasoning goes to Opus. Carry forward existing evidence rather than restarting the search.

## 4. Verify the final state once
Assign the brief's full verification level to `verify-runner`. Inspect its exit codes, saved log, counts, and final diff yourself; do not rely on a prose claim of success. Expected failures should drop by exactly the defects fixed. Use `--browser --bench` when both requirements apply.

After any subsequent executable/test edit, repeat the required checks for the new final state. Reuse evidence only when the checked state is unchanged. No duplicate full run by implementer, reviewer, and orchestrator just to repeat the same passing result.

## 5. Review
Give `change-reviewer` the brief, saved task diff, new-file inventory, pre-existing changes, and verification evidence. It has no command or editing tools. For high-risk work, invoke it with `model: opus` and record why; routine review uses its Sonnet default.

On **request changes**, send findings to the appropriate worker (test findings to `test-writer`). Reverify after fixes and resume the reviewer with the delta; at most two correction rounds before surfacing the unresolved issue to the user. Never report a partial review as approval.

## 6. Report
State the change, files, user-visible behavior, verification evidence, reviewer verdict, and follow-ups. Record roles/models and escalation reasons. Mark the backlog item **Done (commit)** and add any follow-ups as new `MNT-NN`/`UI-NN` items. For UI items include the before/after evidence. Commit after each finished item on its branch; push and offer the merge when the batch the user asked for is done.
