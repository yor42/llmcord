# UI guide

Conventions for polish work on the two user surfaces: the NiceGUI dashboard under `/admin` and the bot's Discord replies. They describe what the code does today; when a polish item changes a convention, update this file in the same change. Words come from [the glossary](../glossary.md).

## Dashboard (NiceGUI + Quasar, Tailwind classes)

**Structure**
- The server page (`dashboard.py`, `_register_pages`) has five tabs, named `setup`, `characters`, `lore`, `imports`, `prompts` and labelled Server setup, Characters, Lore, Imports, Prompt presets. `?tab=` selects one and the URL follows the selected tab.
- Panels build lazily on first selection. `LiveContext.refresh(tab=None, owner=None)` rebuilds in place; there are no full-page reloads or `ui.navigate` calls. Keep it that way.
- A builder must not `await` during a build (a refresh mid-build would double-fill).

**Style (D20, UI-28)**
- One dark Discord-like style. Tokens are the `THEME_*` constants in `dashboard.py` (page `#1e1f22`, header `#111214`, cards `#2b2d31`, primary `#5865f2`, destructive `#da373c`, muted text `#b5bac1`, faint `#949ba4`). `apply_theme()` runs first in every `@ui.page`; it sets `ui.colors` and one CSS block. Add new shared classes there with the `ll-` prefix instead of per-page colours.
- Buttons and tabs are sentence case (no uppercase transform); cards have a 12 px radius.
- Server page frame: header breadcrumb (`ll-crumb`: link "llmcord" with `aria-label="llmcord / Servers"`, aria-hidden "/", 24 px server icon `ll-crumb-icon` or initials tile `ll-crumb-tile`, name `ll-crumb-name`); content sits in `ll-page` (32/48 px padding, 16 px on phones); tabs use `ll-tabs` (muted inactive, white active, primary indicator, `THEME_DIVIDER` bottom border). Tab panels are transparent so top-level cards (`THEME_CARD`) stand out; nested cards (`.q-card .q-card`) use `THEME_NESTED`. Tokens added: `THEME_DIVIDER` `#3f4147`, `THEME_NESTED` `#313338`.
- Account menu (UI-29): both signed-in pages (`/` and the server page) have a `ui.header`; the right side is `account_menu(session)`: a flat button with `aria-label="Account menu"`, a 32 px round avatar (`ll-avatar`, from `auth.user_avatar_url(user, 64)`; initials tile `ll-avatar-tile` if none) and the display name (`global_name` or `username`, `ll-account-name`, hidden under 600 px). It opens `ll-menu` (name, `@username` when different, separator, **Sign out**). Sign out stays a plain POST form to `/logout` with the escaped csrf field, styled as `ll-menu-item`; no JS. A `ui.button` in the header needs explicit white text (`ll-account`), or Quasar tints it primary.
- Sections: wrap each tab section in `with section('Title'):` (a top-level `ll-section` card: 24 px padding, 16 px on phones, 16 px gap, 20 px bold `ll-section-title`) instead of `ui.separator()`. A row of fields plus its button goes in `ui.element('div').classes('ll-form-row')` (full-width wrapping row, bottom-aligned, fields 16rem, full width on phones); a bare `ui.row()` inside a tab panel shrink-wraps and stacks its children. Expansions and cards inside a section use `THEME_NESTED`; expansions directly in a section get a `THEME_DIVIDER` border, never a white one. Tab panels have no side padding, so cards line up with the tabs.
- Fields are outlined and dense everywhere (`default_props` set once in `mount_dashboard`): `THEME_BODY` fill, `THEME_BORDER` outline, 8 px radius. The main action in a row is a filled primary button; secondary actions use `.props('outline')`; destructive ones use `color='negative'`. Tabs use `outside-arrows mobile-arrows` so the phone scroll arrow never covers a label.
- Editing flow (D21, UI-32): editors follow the Discord settings pattern. The title row of an editor holds one ⋯ menu (`more-horizontal` icon, `aria-label="More actions for <name>"`) with its other actions: reversible ones first, a separator, then the destructive one in red; deletes still go through `confirm_dialog`. Save is not a button in the form: a sticky bottom bar (`<name> has unsaved changes.` · Reset · Save changes) appears only while the open editor differs from its saved values (MNT-21 builds the helper). List rows are compact, one line per item: content, a priority or status pill, an Edit icon button with a tooltip, and a ⋯ menu for Move and Delete. Until an editor is converted (UI-25, UI-20, UI-23, UI-15), the older one-row-of-buttons rule below still applies to it.
- Inside an expansion, wrap the content in `ll-stack` (16 px column gap; uploaders capped at 20rem). Expansions nested inside another expansion get `ll-subpanel` (`THEME_DIVIDER` border, 8 px radius). Sub-headings inside a section use `ll-subtitle` (16 px bold). Put a record's actions in one `ll-form-row`: save (filled), reversible actions such as Archive (`outline`), delete (negative). Dialog buttons sit right-aligned (`ll-form-row justify-end`): Cancel `flat`, the verb filled.
- Lore workspace: drop zones use `ll-drop` (`THEME_BODY` fill, dashed `THEME_BORDER`); keep their `p-4`, `min-h-48`, classes and nesting, because drag and edge-drop tests depend on that geometry. Entry tiles use `ll-entry` (compact nested tile; `size=sm` buttons: Edit filled, Move outline, Delete negative).
- Result blocks (the card import preview) are nested cards with `ll-stack ll-result` (16 px padding, `THEME_DIVIDER` border, 8 px radius, no shadow). Inside an `ll-stack` a lone button stretches to full width, so put it in an `ll-form-row`. Give a select in an `ll-form-row` the `ll-wide` class (22rem basis) when its label is long, so the empty-state label is not clipped.
- `ui.number` has no app-wide default (only input, select and textarea do), so number fields need `.props('outlined dense')` until UI-35 adds the default. Prompt preset blocks are nested `ll-stack ll-result` cards; each top-level area of the Prompt presets tab (library, blocks, import, export, preview) is its own `section()`.
- Icons (UI-30): our own icons are Lucide SVGs (lucide-static 1.53.0, ISC, vendored in `llmcord_core/icons/lucide/` with its LICENSE; no CDN). `llmcord_core/icons.py` draws them as `ll-icon` spans (CSS mask of a `data:` SVG, `currentColor`, `aria-hidden`, 1.715em like Quasar button icons; CSS added by `apply_theme`). Use `lucide_button(text, name, **kw)` instead of `ui.button(text, icon=...)`, and `lucide(name, size)` for a bare icon such as a drag handle (add `ll-icon-solo` when it is not inside a button). Unknown names raise `KeyError`. Actions: delete/remove `trash-2`, create `plus`, import/upload `upload`, edit `pencil`, move left/right `arrow-left`/`arrow-right`, drag handle `grip-vertical`. To add one, download `https://unpkg.com/lucide-static@1.53.0/icons/<name>.svg` into that folder (same version) and use its name. Quasar's built-in icons (expansion carets, select arrows, checkbox ticks, tab arrows) stay as they are.
- Server cards: one link per server whose accessible name is the server name (`aria-label`); the icon comes from `auth.guild_icon_url(guild, 128)`, else an `aria-hidden` initials tile (`server_initials`).

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
- Check the page at desktop (1400 × 1000, the test viewport) and phone width (about 390 px) before and after; attach both in the report. `scripts/screenshot_dashboard.py --out <dir>` saves every tab at both widths from the browser fixture (mock data only).

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
