# UI guide

Conventions for polish work on the two user surfaces: the NiceGUI dashboard under `/admin` and the bot's Discord replies. They describe what the code does today; when a polish item changes a convention, update this file in the same change. Words come from [the glossary](../glossary.md).

## Dashboard (NiceGUI + Quasar, Tailwind classes)

**Structure**
- The server page (`dashboard.py`, `_register_pages`) has five tabs, named `setup`, `characters`, `lore`, `imports`, `prompts` and labelled Server setup, Characters, Lore, Imports, Prompt presets. `?tab=` selects one and the URL follows the selected tab.
- Panels build lazily on first selection. `LiveContext.refresh(tab=None, owner=None)` rebuilds in place; there are no full-page reloads or `ui.navigate` calls. Keep it that way.
- A builder must not `await` during a build (a refresh mid-build would double-fill).

**Actions and feedback**
- Every write goes through `ctx.button(...)` or `ctx.upload(...)` (guarded, audited via `AdminService.run`). Never wire a raw `ui.button` to a store write.
- Success: pass `success='…'` for a positive toast when the result is not otherwise visible. Failure: the context shows a negative toast (8 s) with the error text; write that text for the user (see "Messages").
- Every delete or remove uses `scene_ui.confirm_dialog` with a title, what will be lost, and a verb button ("Delete lorebook", not "OK").
- Uploads save on upload; do not add a second save step.

**Text**
- User-facing words: server (not guild), world, hub, channel, lorebook, emotion, owner, link/unlink. Code names may stay `guild`, `space`, `book`.
- Channels show as `#name`; never show a raw ID where a name is known (threads are a known gap, UI-03).
- Buttons are verbs in sentence case ("Save cast", "Create space"). Labels are nouns. Keep them short.

**Changing elements safely**
- The Playwright suite (`tests/test_dashboard_browser.py`) selects by role, accessible name, label text and a few classes. Renaming a button or label is a test change: grep the browser tests first and send the test update to `test-writer` in the same item.
- Keep inputs labelled (Quasar `label=`), so they keep an accessible name.
- Check the page at desktop (1400 × 1000, the test viewport) and phone width (about 390 px) before and after; attach both in the report. UI-01 adds a script for this; until then use the fixture server the browser tests start.

## Discord replies

- Errors: public messages are generic and carry a reference ID; detail goes to the invoker ephemerally and to the log (D3). Don't put provider or internal detail in a public message.
- Slash-command `ValueError` text is shown to the user as written: write it as a sentence that says what to do ("Give a numeric message ID").
- Outcomes are definite (for example "Linked {world} to {hub}."), with counts pluralised correctly.
- Admin commands live under `/admin` (D11); member commands keep their own groups. Option names: `character`, `characters`, `space`, `hub`, `world`.
- The public usage footer is a per-server setting (D2); the "no active character" hint deletes itself after about 15 s (D12).
- Reply text is pinned by slash-command tests: changing wording is a test change too.

## Evidence for a UI item

The report for a `UI-*` item includes:
1. Before and after screenshots (dashboard) or the exact before/after reply text (Discord).
2. The browser or slash-command tests that pin the changed text or flow, and whether they changed.
3. `scripts/verify.sh --browser` for any dashboard file (see the `verify` skill).
4. User docs updated when a visible name, label or command changed (`docs/server-guide.md`, `docs/admin-console.md`, `docs/glossary.md`).
