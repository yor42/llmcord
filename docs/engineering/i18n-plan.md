# Multi-language support (i18n): architecture briefing and implementation spec

**Status:** approved by the user on 2026-10-12 as **D39** in [backlog.md](backlog.md), with decisions A–E as recommended and F allowing parallel Phase 1 batches on separate files. Phase 0 is FEAT-43; the other phases are FEAT-44..46 and MNT-66..75.
**Scope:** the Discord bot and the admin dashboard, Korean (`ko`) and English (`en`) first. Adding a language later must not need core code changes.
**Method:** a read-only scan of the repository at `dcb4d29`, an AST inventory of string literals (section 1.3, appendix A), and the library behaviour this spec depends on, checked against the pinned versions (appendix B).

## 0. Summary

- **There is no JavaScript frontend.** The dashboard is NiceGUI 3.17.1: Python builds every element, and Vue/Quasar only render them. i18next, vue-i18n and svelte-i18n do not apply. Both surfaces are Python, so **one catalog system serves the bot and the dashboard**.
- **Recommended stack:** GNU gettext `.po` catalogs, with the **English UI text as the message id**. Babel (CLDR data) does extraction, plural rules, lists, numbers and language names. A thin `llmcord_core/i18n/` package wraps it. Slash commands use Discord's own localization through `app_commands.Translator`.
- **Two rules carry the design.**
  1. **Translate at the edge.** Stores, game engines and model code never produce translated text. They raise coded errors and return structured results. Only the code that sends a Discord message or builds a dashboard element turns them into words, because only that code knows who will read them.
  2. **The audience decides the language.** A message only the invoker sees uses their Discord client language (`interaction.locale`). A message the channel sees uses the server's bot language. Character lines and anything a person typed are never translated.
- **Size:** about **1,840 user-facing strings in 46 modules**. Of these, 433 are f-strings, about 110 are slash-command metadata and about 300 are raised errors. About 100 sentences are built from fragments, and Korean word order breaks every one of those. The dashboard holds 745 strings, the Discord modules 548, the shared core 418 and the pure engines 128.
- **The main constraint is the tests.** About 470 assertions in 69 of the 142 test files, plus about 1,300 Playwright selectors, match English text. The extraction phase must therefore keep English output **byte-identical**, and tests keep running in `en`. This is why English text is the message id: an untranslated message is its own English text, so wrapping a string changes nothing a test can see.
- **Plan:** Phase 0 builds the foundations. Phase 1 extracts strings with no visible change. Phase 2 adds locale plumbing and the server language setting (schema v22). Phase 3 adds the Korean pack. Phase 4 automates the checks. Phase 5 holds optional extras. Section 9 lists the decisions that need the user's go-ahead.

## 1. Project structure analysis

### 1.1 Stack as found

| Layer | What it is | i18n consequence |
| --- | --- | --- |
| Discord bot | discord.py 2.6.4, `SkitBot` (`discord_bot.py`), slash commands built in `register_commands`, persistent `DynamicItem` buttons | Native locale data on every interaction. Command metadata can be localized by Discord itself. |
| Admin dashboard | NiceGUI 3.17.1 mounted under `/admin` (`dashboard.mount_dashboard`, `ui.run_with`, `dashboard.py:1009`). Python builds pages per client, and Quasar renders them. | Strings live in Python, so the Python catalog serves them. Quasar's own strings come from its language pack, which NiceGUI selects per page. |
| HTTP routes | FastAPI 0.142.2: OAuth (`web.py:102-155`), avatars, `HTTPException` details from `auth.py` | Error details reach people as toasts and HTTP responses. |
| Browser JS | Two small files (`lore_drag.js`, `macro_highlight.js`) and one inline ECharts formatter (`dashboard.py:1391`) | Three strings. Python injects them; no JS i18n library is needed. |
| Persistence | One SQLite file, `guild_settings` per server (`admin_store.py:82`) | A server bot-language column is a schema change (v22). |
| Jinja2 | Still in `requirements.txt`, but the legacy routes were retired (R6) | Nothing to localize. |

### 1.2 Language-aware pieces that already exist (reuse them)

- **Item recipes are already stored per locale** (D30): `items/recipe.py` defines `LOCALE_RE`, `DEFAULT_LOCALE = 'en'` and `lookup()`, which falls back to `en`. Two gaps remain: `items_store.py:214` always rolls with `DEFAULT_LOCALE`, and `shop_ui.py:15` edits only the `en` lines.
- **Spoken-move lexicons** (`spoken/lexicon/*.json`) **and the ambient invitation rule** (`ambient_invite.json`) are EN/KO **input recognition**, not UI text. They stay a separate track, but the Korean button labels must match them (section 5).
- **Discord already sends the locale.** Every interaction has `locale`, the user's client language, and `guild_locale`. Every guild has `preferred_locale` (discord.py `interactions.py:233-238`, `guild.py:625`).
- **The dashboard session already holds the user's Discord language.** `web.py:130` stores the `/users/@me` payload, and with the `identify` scope that payload includes `locale`. No extra API call is needed.
- **NiceGUI ships Quasar's `ko-KR` language pack** and Korean "connection lost" strings (`nicegui/translations.py`).

### 1.3 Inventory

The method was an AST walk over `llmcord_core/`, `web_main.py` and `llmcord.py`. It took every string literal and f-string that reads like prose or is passed to a UI or Discord call (labels, toasts, `send_message`, `Button`, `Embed`, `describe`, `Choice`, exception constructors, and so on). Docstrings and logging calls were left out. It then removed 81 model-facing prompt strings, 23 internal diagnostics and 7 SQL fragments.

The scan **undercounts** single-word labels kept in data tuples, such as `("sit", "Sit", ...)` at `games_holdem_discord.py:369`. Expect about 2,000 messages in total. Per-module counts are in appendix A.

| Surface | Strings | Main files |
| --- | ---: | --- |
| Dashboard / web | 745 | `dashboard.py` (403), `games_ui.py` (108), `presets_ui.py`, `currency_ui.py` (65 each), `shop_ui.py`, `lore_workspace.py`, `scene_ui.py`, `web.py`, `savebar.py` |
| Discord surface | 548 | `discord_bot.py` (188, 72 of them command metadata), `games_holdem_discord.py` (117), `games_doubt_discord.py` (111), `games_discord.py` (65), `items_discord.py`, `favorites_discord.py`, `usage.py`, spoken-move modules |
| Shared core (errors that reach both surfaces) | 418 | `admin_store.py` (168, 91 of them raised), `config.py`, `store.py`, `prompts.py` (validation only), `lorebooks.py`, `auth.py`, `avatars.py`, `backend.py`, `prices.py`, `holdem_store.py`, `items_store.py` |
| Pure engines | 128 | `games/blackjack.py`, `games/holdem.py`, `games/doubt.py`, `games/registry.py`, `items/*` |
| **Total** | **1,839** | 433 f-strings, about 110 command-metadata strings, about 300 raised errors |

### 1.4 String patterns and what each needs

| # | Pattern | Example | Fix pattern |
| --- | --- | --- | --- |
| a | Ephemeral replies, through `reply()` helpers or inline `send_message(..., ephemeral=True)`. There are 45 sites in `discord_bot.py` and 22 in the other modules. | `discord_bot.py:1315` `reply()` | Render with the invoker's locale. |
| b | Command metadata: `Group(description=)`, `@describe`, `Choice(name=)` | `discord_bot.py:897`, `:956`, `:1647-1656` | Wrap in `C_()`, which returns a `locale_str`, and let the tree's Translator localize it. |
| c | Public game messages from renderers that return `(text, view)`, with button labels in spec tuples | `games_holdem_discord.py:353` `render_lobby`, `:369` | Pass the channel locale into each renderer. |
| d | Pure engines that emit English | `games/holdem.py:32` `CATEGORIES`, `:230-245` hand names; `games/blackjack.py:56` `rules_text`; move errors in `games/registry.py` | Return codes and values (`HandRank(category, ranks)`). The edge renders them. |
| e | Store and validation errors written as finished English sentences and shown verbatim on both surfaces | 91 raises in `admin_store.py`. The dashboard shows `str(error)` (`dashboard.py:61`, `:758`) and so does the bot (`discord_bot.py:1626`). | Raise `UserError(code, **params)`, a `ValueError` subclass whose `str()` stays English. |
| f | Sentences built from fragments (about 100) | `games_holdem_discord.py:367` `"; the table closes {stamp} if no hand starts."`, `dashboard.py:1456`, `currency_ui.py:106`, `discord_bot.py:1600-1611` | Rewrite each as one whole message per case. Korean puts the verb last, so fragments cannot be translated. |
| g | Hand-made plurals and number words | `discord_bot.py:1036`, `:1175`, `:1510`; `games_doubt_discord.py:45` (`WORDS`), `:136` | `t.n(singular, plural, n)`. Number words become per-locale messages. |
| h | English possessives and word order baked in | `games_doubt_discord.py:502` `"{doubter} doubted {claimer}'s {claim}"` | One whole message. Korean particles go through the josa format (section 5). |
| i | The public usage footer, on character messages | `usage.py:127` | Channel locale, with Babel number formatting. |
| j | Dashboard frame, labels, hints, toasts, dialogs and aria-labels | tabs `dashboard.py:1231-1239`, `section()` `:970`, `confirm_dialog` `scene_ui.py:23`, `aria-label="Account menu"` `dashboard.py:1324` | `ctx.t(...)`. Translate aria-labels too, because screen readers read them. |
| k | JS strings and the ECharts formatter | `lore_drag.js:29`, `:36`; `toLocaleString('en-US')` and `' tokens'` at `dashboard.py:1391` | Pass the strings in as element props and the locale into the formatter. |
| l | Dates and numbers | `%Y-%m-%d %H:%M` (`dashboard.py:1445`, `currency_ui.py:87`), `{n:,}`, `format_usd` | Babel by locale. Keep Discord's `<t:…:R>` timestamps, which each client already shows in its own language. |
| m | Built-in defaults shown as text | `DEFAULT_CURRENCY = 'coins'` (`admin_store.py:1112`, stored as `''` and applied when read); `"Built-in default"` (`discord_bot.py:1607`) | Translate when shown and the stored value is empty. A name an admin typed stays as typed. |

### 1.5 Out of scope: never translated by this work

| What | Why |
| --- | --- |
| Model-facing prompt text (`prompts.py` default blocks, `game_chooser.py`, `playstyle_generate.py`, `catchup.py`, `identity.py`, director text in `engine.py`): 81 strings | The model reads it, not a person. Characters already "follow the conversation's language" (`prompts.py:98`). Changing prompts is a model-quality question, not an i18n one. See Phase 5 for a catch-up language hint. |
| Character lines (webhooks) and everything admins or members typed: cards, lore, presets, custom item text, currency names, the audit `detail` already stored | It is content, not UI, and was written in its own language. |
| Logs, reference IDs, `error_detail`, operator CLI scripts | These are diagnostics for the operator. English keeps them searchable. |
| Spoken-move and ambient lexicons | They recognize input. Adding Japanese there needs morphology code (`spoken/words.py`), which is a separate track. |
| User docs and the wiki (`docs/`, `scripts/build_wiki.py`) | A later, separate track (Phase 5). |
| Slash command **names** and option **names** | Decision B, section 9. |

## 2. Tech stack recommendation

### 2.1 Comparison

| Criterion | **gettext `.po` + Babel (recommended)** | Keyed JSON (i18next-style, `python-i18n`) | Fluent (`fluent.runtime`) | ICU MessageFormat (PyICU) |
| --- | --- | --- | --- | --- |
| Test churn in Phase 1 | **None.** An untranslated message is its own English text. | Every string needs a key name and an `en.json` entry. That is about 2,000 names to agree on and review. | Same as keyed JSON | Same as keyed JSON |
| Noticing that a translation is stale after an English wording change (UI polish changes wording often) | **Built in:** `update` marks changed entries fuzzy, and fuzzy entries are not shipped. | Needs custom tooling | Needs custom tooling | Needs custom tooling |
| Plurals for future languages (ru 3 forms, ar 6) | `Plural-Forms` header written from CLDR | Needs a plural library | Very good | Very good |
| Korean particles after placeholders | Custom format spec (section 5) | Custom | Custom function | Custom |
| Translator tools | Poedit, Weblate, Crowdin, Lokalise all support `.po` natively | Good | Few tools | Few tools |
| Dependency on a Raspberry Pi 5 (arm64) | Babel is pure Python, BSD-3 | Light | Unmaintained since 2023 | C++ ICU build on arm64 |
| Static checks | `pybabel extract` plus ruff `INT` rules | Key-existence checks | Weak | Weak |

**Recommendation:** gettext catalogs with English message ids, Babel at runtime and in tooling, Python's `string.Formatter` for named placeholders, and discord.py's `Translator` for command metadata. Keep JSON for **data** that is already locale-keyed (item recipes) and for recognition lexicons, because those are content, not UI messages.

### 2.2 Components (checked against pinned versions; see appendix B)

- **Babel** (`Babel==2.18.0`, current at the time of writing; add to `requirements.txt`): `pybabel extract`/`update`, CLDR plural rules, `format_list` (`A, B, and C` / `A, B 및 C`), `format_decimal`, `format_datetime`, and language names in their own language (`ko` gives `한국어`, `ja` gives `日本語`).
- **Runtime loading:** read the `.po` files with Babel's `read_po` once per process, skip fuzzy entries, and merge them into one lookup per locale. **No `.mo` binaries are committed and there is no build step.** Parsing about 2,000 entries for each of two locales should take a fraction of a second on a Pi 5 (Phase 0 measures it). The Docker image needs no change if the catalogs live inside `llmcord_core/`, which is already copied (`Dockerfile`).
- **discord.py 2.6.4:** `app_commands.locale_str`, a `Translator` subclass registered with `CommandTree.set_translator()` before `tree.sync()` in `setup_hook` (`discord_bot.py:176-182`), and `Locale.korean == 'ko'`.
- **NiceGUI 3.17.1:** `Client.page.resolve_language()` is read when the response is built (`nicegui/client.py:223`). That sets `<html lang>` and loads `static/lang/<language>.umd.prod.js` (`templates/index.html:135`). A per-client copy of the page with its own `language` therefore localizes Quasar for that client only. Phase 0 spikes this and pins it with a test.
- **ruff `INT` rules** (flake8-gettext, ruff 0.16.10): configured for our marker names, they reject `t(f'…')` and `t('…'.format(…))`, which Babel would otherwise drop without a word. They do **not** see method calls such as `t.n(f'…')`. A small custom check covers those (section 7).

## 3. File and directory layout

```
llmcord_core/
  i18n/
    __init__.py      # public API: Translator, N_, NP_, E_, UserError, resolve helpers
    catalog.py       # discover locales/*/LC_MESSAGES/*.po, load with Babel, merge per locale, cache
    negotiate.py     # Discord / Accept-Language / BCP 47 tags -> supported locale; fallback chain
    format.py        # safe named-placeholder formatter, Korean particle spec, numbers and lists
    codes.py         # shared user-facing error codes (errors domain); the only place E_() appears
    terms.py         # shared vocabulary (common domain): game names, hand and rank words, On/Off, default currency
    discord_locale.py  # app_commands.Translator adapter, C_(); audience resolvers for interactions and guilds
    web_locale.py      # NiceGUI client binding, cookie/Accept-Language resolution, Quasar language mapping
  locales/
    templates/       # generated .pot files, committed so CI can tell when they are stale
      bot.pot  commands.pot  web.pot  errors.pot  common.pot
    ko/LC_MESSAGES/
      bot.po  commands.po  web.po  errors.po  common.po
    ja/LC_MESSAGES/  # a future language pack: add the folder, no code change
scripts/i18n.py      # extract | update | init <lang> | check | stats | ratchet [--update [--allow-increase]]
scripts/i18n_check.py  # the checks behind `check` and the string ratchet
scripts/i18n_ratchet.json  # ratchet baseline: bare UI and raise literals per module
```

English needs no catalog: it is the source. `en` is always available and is the final fallback.

**Domains** are a source-side split, so translators and reviewers work per surface. At runtime a locale's domains merge into one lookup keyed by `(msgctxt, msgid)`. The same English text may appear in two domains only with the same translation. If two translations differ, the entries need a `msgctxt`, and `check` enforces this.

| Domain | Contents | Extracted from |
| --- | --- | --- |
| `bot` | Discord replies, public table and lobby text, buttons, select options, modals, embeds, footers | Discord-surface modules (path globs in `scripts/i18n.py`) |
| `commands` | Slash command, group and option descriptions, and choice names | `C_()` calls |
| `web` | Dashboard text, toasts, dialogs, aria-labels, JS-injected strings, HTTP error pages | Dashboard and web modules |
| `errors` | Shared `UserError` codes shown on both surfaces (validation, conflicts, permissions) | `i18n/codes.py` only |
| `common` | Vocabulary shown on both surfaces | `i18n/terms.py` only |

`check` fails when a `.py` file outside the mapped globs contains a marker call (`C_` is exempt: command metadata may be built in any module). A new module therefore cannot leak strings that no domain extracts.

**`.po` conventions**

- The `msgid` is the exact English text. Placeholders are always named (`{world}`), never positional.
- Short or ambiguous strings get a `msgctxt` (`"tab"`, `"blackjack action"`, `"holdem action"`, `"button"`). "Call" in Hold'em and "Call" as a verb need different Korean.
- Translator notes come from `# i18n:` comments above the call (`pybabel extract -c i18n:`). For example: `# i18n: {claim} is like "two 7s"`.
- The `Plural-Forms` header comes from CLDR when the language is created (`nplurals=1; plural=0;` for ko, ja and zh).
- An error code reads like this: `CURRENCY_NAME_CHANGED = E_('currency.name_changed', 'The currency name was changed elsewhere. Reload the page and try again.')`. The code is stable. Logs, tests and the audit trail use it, and it does not change when the wording does.

**Locale tags.** A catalog folder uses the POSIX form of Discord's tag: the bare language where Discord has one (`ko`, `ja`), and the region where Discord tells them apart (`zh_CN`, `zh_TW`, `pt_BR`, `es_ES`, `es_419`). Negotiation tries the exact tag, then the language only, then `en`. For example, `en-GB` resolves to `en`, and `zh-TW` resolves to `zh_TW` when that pack exists.

## 4. Context-aware translation strategy

### 4.1 One API for both surfaces

```python
from llmcord_core import i18n

t = i18n.Translator(loc)                       # loc comes from an audience resolver (4.2, 4.3)
t('Linked {world} to {hub}.', world=w, hub=h)  # whole sentence, named placeholders
t.n('{count} character', '{count} characters', count, count=count)
t.p('holdem action', 'Call')                   # msgctxt
t.np('holdem', '{count} seat', '{count} seats', count)  # msgctxt + plural; count is passed as {count} automatically
t.error(err)                                   # renders a UserError in this locale
t.list(names); t.number(1500)                  # CLDR list and number formats

BUSY = i18n.N_('The server is busy. Try again.')   # module constant: a str (English) marked for extraction, translated later with t(BUSY)
STAND = i18n.NP_('blackjack action', 'Stand')       # the same with a msgctxt

raise i18n.UserError(codes.CURRENCY_NAME_CHANGED)   # store or engine layer: never translates
```

- Babel extracts with the keywords `t`, `n:1,2`, `p:1c,2`, `np:1c,2,3`, `N_`, `NP_:1c,2`, `C_` and `E_:2`. It matches the last part of a dotted call (`t.n(…)`, `self.t(…)`), as checked in appendix B. `check` confirms that no other function in `llmcord_core` is named `t`, `n`, `p` or `np`.
- `UserError` subclasses `ValueError`, and its `str()` is the English text. Every existing `except ValueError` path, log line and `assertRaisesRegex` keeps working, so raise sites can move over one at a time.
- **The translator is passed explicitly, not held in a context variable.** asyncio copies context into each task when the task is created. A game timer started inside a button handler would inherit the clicker's locale and post public table messages in it. Explicit passing makes that bug impossible, and `t` is easy to see in reviews and tests.
- **Format strings in catalogs are untrusted input.** The formatter allows bare field names only. `{x.attr}`, `{x[0]}` and conversions are refused, and `check` also refuses any field that is not in the `msgid`. Escaping user values for Discord markdown and mentions (`_clean`, `allowed_mentions`) still happens in code before formatting. Catalog text never goes into `ui.html`.

### 4.2 Discord bot

**Where the locale comes from**

| Source | Meaning | Caveat |
| --- | --- | --- |
| `interaction.locale` | The invoking user's Discord client language | Present on every slash command, button, select and modal interaction |
| `guild_locale` / `guild.preferred_locale` | The server's "primary language" | Real only for Community servers. Other servers report `en-US`, so a small private Korean server would get English. |
| **Server bot language** (new) | `guild_settings.language`: `''` means follow Discord, otherwise a supported locale | A schema change (v22, decision C). Set in the dashboard's Server settings editor. The bot reads it fresh on each turn, like every other setting. |

The channel locale is the first of these that is set: the server bot language, then `guild.preferred_locale` if the server is a Community server, then `en`.

**The audience decides the language**

| Message | Language | Examples |
| --- | --- | --- |
| Ephemeral reply, follow-up, modal or autocomplete shown to the invoker only | `interaction.locale` | Command errors (`discord_bot.py:1619-1642`), `/memory`, `/catchup` replies, Hold'em **My cards**, the raise modal |
| Public message from a command | Channel locale, even though an interaction exists | Opening a blackjack, I Doubt It or Hold'em table; `/shop` and `/use` results if they are public |
| Public message with no interaction (`on_message`, timers, recovery after a restart, the cleanup loop, sweepers) | Channel locale, resolved at each render, never captured in a closure | Table and lobby updates, move checks, doubt checks, the "no active character" hint, the usage footer |
| Buttons on a public message | Channel locale. The ephemeral reply to a click uses the **clicker's** locale. | `HoldemButton`, `DoubtButton`, `GameButton` |
| Character lines (webhooks) | Never translated | Model output follows the conversation's language |

**Command metadata.** `C_()` wraps every description and choice name. `LlmcordTranslator.translate()` returns the catalog string for supported locales and `None` for the rest, so Discord shows the default. The tree already syncs at every start (`discord_bot.py:176-182`), so a catalog change applies after a restart. Discord limits: descriptions and choice names 1–100 characters, button labels 80, select options 100, modal titles and text-input labels 45, message content 2,000 (code cuts at 1,900). `LlmcordTranslator` enforces the 100-character limit at runtime for command metadata. `check` enforces the component limits only by `msgctxt` prefix (section 6.1), so every button label, select option and modal text needs a `button…`, `select…` or `modal…` context.

**Components.** Every `custom_id` is already language-neutral (`games_discord.py:98`, `games_doubt_discord.py:48`, `games_holdem_discord.py:55`, `games_spoken_discord.py:54`, `games_spoken_doubt.py:34`). Keep it that way: a custom ID must never contain text. A message that was already posted keeps its old labels until the bot next edits it, and that is acceptable.

**Error handler.** The ephemeral reply comes from `t.error(original)` in the invoker's locale. The "Something went wrong (ref …)" text is translated, and the reference ID and log detail stay as they are.

**Items.** Pass the audience locale to `items.roll` (`items_store.py:214`), add `ko` text to the built-in recipes (`items/recipes/*.json`), and later give the Shop tab editor a locale selector.

### 4.3 Admin dashboard (NiceGUI)

**Resolution order:** the `llmcord_lang` cookie (an explicit choice), then the Discord account `locale` from the session (`web.py:130`), then `Accept-Language`, then `en`.

**Why a cookie and not `localStorage`.** The dashboard is rendered on the server, so the server must know the language before it sends the first byte. `localStorage` is never sent with a request. Reading it would need a round trip after load and a re-render, so every page would flash in English first. It would also break the ui-guide rule that a builder must not `await` during a build. A cookie gives the same per-browser persistence and the server can read it.

- **The cookie:** `llmcord_lang`, one year, `Secure`, `HttpOnly`, `SameSite=Lax`, kept separate from the session so it survives sign-out. Phase 5 can add a per-user preference stored in the database to follow an admin across devices (a schema change).
- **The switcher:** the account menu (`account_menu`, `dashboard.py:1318`) gets a **Language** item that lists the discovered locales by their own names. Each entry is a plain POST form to a new `/lang` route that checks the CSRF token and origin exactly like `/logout` (`web.py:144`), with no JS. `next` must be a same-site path that starts with `/` and not `//`, to block open redirects. The route answers with a 303 back to the same page.
- **The full reload is deliberate.** It rebuilds the header, the tabs, Quasar's language pack and `<html lang>` in one step. It is a navigation, like sign-out, not a panel refresh, so it is a documented exception to the ui-guide's "no full-page reloads" rule (decision E). While an editor has unsaved changes, the save bar's `beforeunload` warning protects them, or the menu item can be disabled while dirty, matching how the save bar refuses other actions.
- **Binding:** the three page handlers (`dashboard.py:1132`, `:1167`, `:1213`) set `client.llmcord_locale` next to the existing `client.llmcord_binding`. `LiveContext` exposes `ctx.t`, and helpers without `ctx` call `i18n.web_locale.current()`, which reads the client from NiceGUI's slot stack and returns `en` outside a client. NiceGUI runs event handlers and `ui.timer` callbacks in the client's context, so toasts after a click come out in the right language. `_attempt` (`dashboard.py:56`, `:752`) renders `UserError` with `t.error`.
- **Quasar and `<html lang>`:** the page handler swaps in a shallow copy of `client.page` with `language` mapped from our locale (`ko` to `ko-KR`). The mapping is derived from NiceGUI's `Language` literal, so a new pack needs no table edit. `lang="ko"` also lets screen readers pronounce Korean correctly.
- **Routes:** no locale prefix in URLs. `?tab=` keys stay code names, so a link one admin sends opens in the other admin's own language. There is no SEO concern, because the dashboard is private behind Tailscale. An optional `?lang=` sets the cookie, for testing and `scripts/screenshot_dashboard.py`.
- **JS:** `lore_drag.js` gets its two messages as props. The ECharts formatter gets the page locale and the translated unit.
- **Typography:** fonts are not loaded from a CDN, following the project's vendoring rule. A system fallback stack is enough: `"Noto Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif` after the current fonts. Add `:lang(ko) { word-break: keep-all; }` so Korean words are not split mid-word, and check that the tab row still wraps cleanly at 390 px.

### 4.4 Shared error codes

- `UserError(code, **params)`: `code` is an `ErrorCode` from `i18n/codes.py`, which holds a stable dotted ID and its English `msgid`.
- It is rendered by the dashboard toasts (`dashboard.py:61`, `:758`), the bot's command error handler (`discord_bot.py:1626`), interaction handlers in the game modules, and `web.py:212` ("Invalid input").
- The `HTTPException` details in `auth.py` ("Sign in with Discord", "Server administrator permission required", and so on) become codes too. The dashboard toast and a FastAPI exception handler render them.
- Audit and turn-log rows already store English text, which stays as written because it is a historical record. New audit rows may store the code and parameters later, but that is not part of this plan.

## 5. Korean specifics

- **Register:** polite 해요체 in both surfaces (`저장했어요.`, `다시 시도해 주세요.`). Buttons are short nouns or verbs (`저장`, `삭제`, `취소`, `변경 사항 저장`). This is decision D.
- **Particles (조사):** the right particle depends on whether the previous syllable ends in a consonant (받침), so a placeholder such as a character name needs help. The formatter accepts a pair in the format spec: `{world:을/를}`, `{name:이/가}`, `{name:은/는}`, `{name:과/와}`, `{name:아/야}`, `{place:으로/로}`.
  - Hangul is exact: `(code - 0xAC00) % 28`. The (으)로 pair also treats a final ㄹ as no consonant.
  - Numbers follow their Sino-Korean reading, including 십/백/천/만/억 at the end.
  - Anything else, such as a Latin name, gets the combined form `을(를)`.
  - Example: `msgid "Linked {world} to {hub}."` becomes `msgstr "{world:을/를} {hub}에 연결했어요."`.
  - `check` accepts a spec only if it is a known pair. Unit tests cover batchim, no batchim, ㄹ with 로, digits and the Latin fallback.
- **Whole sentences only.** Korean puts the verb last and attaches particles to nouns, so the fragments in pattern f and the possessives in pattern h must become complete messages first. Phase 1 does this while English stays unchanged.
- **Plurals and counters:** Korean has one plural form, but code still calls `t.n` for English. Counters (명, 장, 개) go inside the Korean text (`{count}장`). Number words ("one", "two") become per-locale messages, and Korean uses digits.
- **Game terms:** use the Korean words players already use, which are also the words the spoken-move lexicon reads.
  - Blackjack: 히트, 스탠드, 더블다운, 스플릿, 서렌더, 인슈어런스, 이븐 머니.
  - Hold'em: 폴드, 체크, 콜, 레이즈, 올인, 블라인드, 바이인, 스택, 팟.
  - Hands: 하이카드, 원페어, 투페어, 트리플, 스트레이트, 플러시, 풀하우스, 포카드, 스트레이트 플러시, 로열 플러시.
  - I Doubt It is known in Korea as 다우트.
  - A test checks that the Korean label of every move button appears in `spoken/lexicon/*.json`, so a player who types the button's label is read as making that move.
- **Proposed glossary** (needs the user's approval; it will become a `ko` column in [the glossary](../glossary.md)):

| English | Korean | English | Korean |
| --- | --- | --- | --- |
| Server | 서버 | Character card | 캐릭터 카드 |
| World / Hub | 월드 / 허브 | Emotion avatar | 감정 아바타 |
| Bind (a channel) | 채널 지정 | Prompt preset | 프롬프트 프리셋 |
| Link / Unlink | 연결 / 연결 해제 | Scene / Branch | 장면 / 분기 |
| Cast (default, active) | 출연진 (기본, 현재) | Personal fact | 개인 기억 |
| Lore / Lorebook | 로어 / 로어북 | Favorite | 즐겨찾기 |
| Owner (of lore) | 소유 범위 | Currency / Ledger | 화폐 / 거래 내역 |
| Character wallet | 캐릭터 지갑 | Item / Inventory / Shop | 아이템 / 인벤토리 / 상점 |
| Operator / Server admin | 운영자 / 서버 관리자 | Spending cap | 지출 한도 |
| Turn log | 턴 로그 | Catch-up | 놓친 대화 요약 |
| Ambient participation | 자율 참여 | Table talk | 테이블 잡담 |
| Spoken move / Move check | 채팅 무브 / 무브 확인 *(to confirm)* | Claim / Doubt / Pile | 선언 / 다우트 / 더미 |

## 6. Migration plan

Every step goes through the `orchestrate-change` workflow (brief, `test-writer` if needed, `implementer`, `verify-runner`, `change-reviewer`). The backlog IDs below were added with D39.

| Phase | Items (proposed) | What changes | Exit criteria |
| --- | --- | --- | --- |
| **0 Foundations** (done, section 6.1) | FEAT-43 | `llmcord_core/i18n/` (negotiation, catalog loader, safe formatter with josa, `Translator`, `UserError`, `N_`/`C_`/`E_`); Babel added to requirements; `scripts/i18n.py`; ruff `INT` config; the ratchet check with a baseline allowlist; empty `ko` catalogs; unit tests. Spikes pinned by tests: NiceGUI per-client `language` and the command-metadata Translator. The live sync on the development guild and the catalog load time on the Pi moved to FEAT-44. | Nothing visible changes. `scripts/verify.sh` is green. |
| **1 Extraction** (English byte-identical) | MNT-66 … MNT-74, one per batch below | Wrap strings, rewrite fragments and plurals as whole messages, move engines to structured results and stores to `UserError`. The ratchet allowlist shrinks with each batch. | **Existing tests pass unchanged.** A test that needs editing is a stop signal: English output changed. |
| **2 Locale plumbing** | FEAT-44 (bot), FEAT-45 (dashboard) | Audience resolvers and explicit `t` through every Discord renderer and handler; `set_translator`; schema **v22** `guild_settings.language` with a backup-before-upgrade migration; the Server settings field; cookie, `/lang` route and account-menu switcher; per-client Quasar language; `<html lang>`; fonts and `keep-all`; the item roll locale. Run everything end to end with the **pseudo-locale** before any Korean exists. | Pseudo-locale screenshots and a Discord walkthrough show no unwrapped string. The browser suite passes in `en`. |
| **3 Korean pack** | FEAT-46 | Approve the glossary and register; translate every domain; add `ko` to the built-in item recipes; native review by the user; dashboard screenshots at desktop and phone width; a Discord development-guild pass with a `ko` client; the lexicon-alignment test. | `check` passes at 100% `ko` coverage and Discord limits hold. User sign-off. |
| **4 Automation** | MNT-75 | An i18n stage in `verify.sh` and CI; a coverage report per locale; a pseudo-locale browser smoke test; docs (`server-guide.md`, `admin-console.md`, `ui-guide.md` i18n rules, the glossary `ko` column); the language-pack runbook (section 8). | CI blocks stale templates, broken placeholders, fuzzy entries and limit overruns. |
| **5 Optional** | separate decisions | Localized command names; a language hint for `/catchup` and memory summaries (FEAT-39 saw Korean summaries come out in English); a per-user language stored in the database; Japanese and Chinese packs; translated user docs. | Each is its own decision. |

**Phase 1 batches** (in this order; sizes from appendix A)

| Batch | Scope | Strings |
| --- | --- | ---: |
| 1a | `errors` domain, part 1: `codes.py` plus `admin_store.py` and `store.py` | ~200 |
| 1b | `errors` domain, part 2, and the other shared modules: `config.py`, `prompts.py` validation, `lorebooks.py`, `cards.py`, `avatars.py`, `auth.py`, `web.py`, `models.py`, `items/*`, `holdem_store.py`, `items_store.py`, `backend.py`, `prices.py`, `admin.py`, `engine.py`, `names.py`, `world_info.py`, `budget.py` | ~260 |
| 1c | Engines to structured results: `games/blackjack.py`, `holdem.py`, `doubt.py`, `registry.py`, `base.py`; vocabulary into `terms.py` | ~105 |
| 1d | Command metadata through `C_()`, with Translator registration that returns English unchanged | ~110 |
| 1e | `discord_bot.py` replies, `favorites_discord.py`, `items_discord.py`, `usage.py` | ~160 |
| 1f | Game surfaces: `games_discord.py`, `games_doubt_discord.py`, `games_holdem_discord.py`, spoken-move modules | ~280 |
| 1g | `dashboard.py`, part 1: the frame, Server setup, Characters, operator pages | ~200 |
| 1h | `dashboard.py`, part 2: the remaining tabs | ~200 |
| 1i | `games_ui.py`, `presets_ui.py`, `currency_ui.py`, `shop_ui.py`, `lore_workspace.py`, `scene_ui.py`, `savebar.py`, `icons.py`, JS strings | ~325 |

**Agent routing** (following `orchestrate-change`): Sonnet `implementer` for each batch, since fragment rewrites need judgment; `test-writer` for the i18n package and the pseudo-locale checks; `verify-runner` with `scripts/verify.sh`, plus `--browser` for the dashboard batches; Sonnet `change-reviewer`. Opus reviews Phase 0's formatter (untrusted format strings) and Phase 2's `/lang` route and schema migration (CSRF, open redirect, migration).

**Parallelism:** CLAUDE.md asks for one small item at a time, but the user chose parallel batches (decision F, D39): batches that own separate files (for example 1a and 1i) may run in parallel worktrees. `dashboard.py` stays sequential because it is 2,405 lines and a conflict hotspot.

### 6.1 Phase 0 as built (FEAT-43)

Phase 0 landed on 2026-10-12. It went through a multi-lens review (security, contract, Korean, tests, tooling), adversarial checks of every finding, and fixes. Nothing outside `tests/` and `scripts/` imports the package yet. Phase 1 batches must follow these rules.

**Runtime**

- **Missing translation:** the English text is used.
- **A translation that fails to format** (an unknown field or a bad spec) falls back to English. It logs one warning per locale, msgctxt and msgid, with no parameter values.
- **English source that fails to format** raises `MessageFormatError`. That is a code bug.
- **`t.error(err)` never raises.** If a `UserError`'s parameters do not fit its message, it logs the error code only and returns the English msgid.
- **`load_catalog(locale)`** accepts a locale only if it is `en`, `pseudo`, or an already-normalized tag that has a `locales/<tag>/LC_MESSAGES` folder. Anything else gets an empty catalog, which is not cached and reads no file. Callers still negotiate first: this is a backstop, not a resolver.
- **Formatter:**
  - A format spec with four or more consecutive digits is refused (`{x:>9999}`), so a translation cannot request a huge allocation.
  - Field names must be bare, as section 4.1 says.
- **`web_locale.current_locale()`** reads the client from NiceGUI's slot stack. It returns `en` when there is no client, and also when the slot's element has been deleted.
- **Not yet done:** the Discord development-guild sync spike and the catalog load time on the Pi. Both need the real bot and hardware, so they move to FEAT-44. The NiceGUI per-client `language` spike is pinned by a test (`tests/test_i18n_web.py`).

**Korean particles (josa)**

- **Stripped before the last sound is read:** trailing spaces, closing brackets and quotes (ASCII and fullwidth), and the Discord markdown markers `* _ ~ |` and backtick. `**루나**` reads as 루나.
- **The combined form (`이(가)`) is used** when the reading is unknown:
  - Latin words.
  - Emoji.
  - Discord mentions, channels, roles, custom emoji and timestamps, including wrapped ones such as `**<@id>**`.
  - ASCII letters followed by digits (`GPT-3`, `v1.0`).
  - Numbers ending in 16 or more zeros (the reading of 경 and above is not covered).
  - Decomposed (NFD) Hangul. This is a known limitation: it is safe but could be exact.
- **A josa spec replaces any numeric spec.** `{n:,:을/를}` is refused. For a number that takes a particle, pass a formatted string: `t('You paid {amount}.', amount=t.number(n))` with msgstr `{amount:을/를} 냈어요.`. In English, `t.number` gives the same `1,500` as `{n:,}`.

**Catalog conventions `check` relies on**

- **Component labels need a context.** A Discord button label, select option or modal text must have a `msgctxt` that starts with `button`, `select` or `modal` (for example `NP_('button.holdem', 'Call')`). Only those contexts are length-checked.
- **Command metadata is never formatted.** `C_()` strings go to Discord as written, so they must not contain `{` or `}`. `check` does not enforce this yet.
- **Message ids must be literals.** A marker call takes a string literal. `t('A' if x else 'B')`, `t(a or b)` and `t(names[i])` are `[non-literal]` errors: wrap each branch in its own `t(...)`.
- **`check` output:**
  - Errors are prefixed with an ID: `[stale-template]`, `[stale-catalog]`, `[po-parse]`, `[plural-forms]`, `[obsolete]`, `[placeholder]`, `[field-syntax]`, `[josa]`, `[limit]`, `[conflict]`, `[unmapped]`, `[non-literal]`, `[name-collision]`, `[ratchet]`.
  - `[fuzzy]` is a warning only.
  - A Korean spec that is not a known pair, such as `{name:이가}`, is a `[josa]` error.

**String ratchet (`scripts/i18n_ratchet.json`)**

- **`ui` count:** bare text literals passed positionally, or through a text keyword (`label=`, `text=`, `placeholder=`, `title=` and so on), to a UI or Discord call (`UI_CALLS` in `scripts/i18n_check.py`); `describe` counts every keyword. A literal counts only if it contains a letter or Hangul and does not look like an identifier (lowercase letters, digits and `_.:/#-` only, such as `'save'` or `'tab:lore'`), so a one-word lowercase label is not counted either. Every module outside `i18n/` is counted.
- **`raise` count:** `raise X('text')` for any exception class except `NotImplementedError`, `AssertionError`, `StopIteration`, `SystemExit`, `KeyboardInterrupt` and `UserError`. Only the first positional argument is read; for `HTTPException`, the second argument or `detail=`.
- **Phase 0 baseline:** 693 `ui` and 529 `raise` literals in 46 modules. The review raised the baseline from 655 and 373 because it closed counting gaps.
- **Not counted** (Phase 1 reviewers grep for these by hand):
  - Text inside `.props('aria-label="…"')`.
  - `ctx.run(..., success='…')`.
  - Labels kept in data tuples, such as `("sit", "Sit", …)`.
- **Updating the baseline after a batch:** run `python scripts/i18n.py ratchet --update`. It refuses to raise any count unless you pass `--allow-increase`, which needs a stated reason in the batch report. It also refuses an unreadable baseline, which `--allow-increase` rebuilds.
- **Parallel batches:** when two batches conflict on the baseline file, take either side and rerun `ratchet --update` on the merged tree. Counts only go down, so the guard passes.

**Known gaps, deferred to MNT-75**

- `check` does not detect duplicate msgid blocks in a `.po` file.
- `check` does not enforce the no-braces rule for `C_` strings.
- The tooling follows symlinks under `locales/`.
- A baseline that is valid JSON of the wrong shape (for example `{"files": []}`) makes `check` and `ratchet --update` stop with a traceback instead of the "unreadable" message. They still exit non-zero and leave the file unchanged.

### 6.2 Phase 1 conventions (MNT-66..74)

Written when Phase 1 started (2026-10-12, decision G). Every batch follows these rules, so batches that run in parallel come out alike.

**Where `t` comes from.** In Phase 1 every translator is English. Phase 2 changes only the resolvers below, not the call sites. Each call site states its audience now, while someone is reading it.

| Code | Gets `t` from | Phase 2 (FEAT-44/45) makes it |
| --- | --- | --- |
| Dashboard builders and their handlers | `t = web_locale.current()` at the top of the function. Closures inside a builder may keep that `t`: one client never changes language without a reload (decision E). | The viewer's language, once the page handlers call `bind_client` |
| HTTP routes and auth (`web.py`, `auth.py`) | `t = i18n.english()` with `# i18n: FEAT-45 request locale` | `request_locale(...)` |
| Discord text only the invoker sees: ephemeral replies, modals, autocomplete | `t = discord_locale.invoker_t(interaction)` | `interaction.locale` |
| Discord text the channel sees: public messages, edits of public messages, timers, `on_message` paths | `t = discord_locale.channel_t(guild)` in the function that has the guild (`interaction.guild`, `channel.guild`, `message.guild`; `None` is allowed). Resolve it at each render, and never keep it in a timer, a closure or an object that outlives the render. | The server bot language |
| A renderer or helper that builds text for its caller | A keyword-only parameter, `def render_lobby(table, *, t=None): t = t or i18n.english()`. Production callers always pass `t=`. The default exists only so existing tests that call the renderer stay unchanged. | Callers pass the audience's `t` |
| Stores, engines, `items/`, `games/`, `spoken/` (no domain) | Never. They raise `UserError(codes.X, **params)` or return structured values. Shared words go in `i18n/terms.py`. | Unchanged |

**Strings.**
- **The msgid is the exact current English text**, with named placeholders (`{world}`). Literal braces are written `{{` and `}}`.
- **f-strings:** every f-string expression becomes a named parameter computed before the call: `t('Linked {world} to {hub}.', world=w.name, hub=h.name)`. Keep format specs such as `{n:,}`. In Phase 1, do not swap in `t.number` or `t.list` unless the English output is identical. For example, Babel's English list adds "and", so `', '.join(names)` stays as it is.
- **Fragments and possessives become whole messages**, one per case. For example, `"Table closes" + (" soon" if x else "")` becomes `t('Table closes soon.') if x else t('Table closes.')`, with the exact old text in each branch.
- **Plurals:** use `t.n(singular, plural, n, **params)` only where the old English used the plural form for every count except 1, which is CLDR English. Otherwise keep the old branch and call `t(...)` in each branch.
- **Short or ambiguous text** gets a context: `t.p('tab', 'Lore')`. Component text needs a `button…`, `select…` or `modal…` context (`t.p('button.holdem', 'Call')`) so `check` enforces Discord's limits.
- **Module-level text** becomes `N_('…')` or `NP_('context', '…')` constants, rendered with `t(CONST)` where they are shown. A constant shown without `t()` is a bug, even though English looks the same.
- **Translator notes:** put a `# i18n: …` comment directly above a call whose placeholder meaning is not obvious, for example `# i18n: {claim} is like "two 7s"`.
- **Name clashes:** in a function that uses the translator, no other variable may be named `t`. Rename it, for example `for table in tables`.
- **Never translated:** model-facing prompt text, log lines, `custom_id`s, identifiers and stored values, anything a person typed, character lines, Discord `<t:…>` timestamps, and the reference ID in error replies.

**Errors.**
- **Defining a code:** `NAME = E_('area.name', 'English message with {param}.')` in `i18n/codes.py`, in the batch's own section. Each batch appends only to its own section, so parallel batches merge cleanly.
- **Code names:** IDs are lowercase and dotted (`currency.name_changed`). They are stable, so logs and tests can cite them.
- **Converting a raise:** a raise whose text can reach a person becomes `raise UserError(codes.NAME, **params)`.
- **Project error classes** (`ConflictError`, `CurrencyError`, `GameError`, `ItemError`, `RecipeError`) become subclasses of `UserError`. `UserError` accepts a legacy plain message during Phase 1, so unconverted raises such as `raise ConflictError('…')` keep working. Converted raises pass a code: `raise ConflictError(codes.NAME, **params)`. `except ConflictError` and `except ValueError` are unaffected, and `str(e)` and `e.args` stay the same.
- **Internal diagnostics stay English** and are not converted: impossible states, programming errors, and messages only logs see. The batch report lists what it left and why.
- **Showing an error:** where a surface shows `str(err)` for a `UserError`, it switches to `t.error(err)` in the batch that owns that surface file.

**Each batch.**
- **Isolation:** it runs in its own git worktree under `.worktrees/` on branch `i18n/<batch>`. It edits only its own files, the generated catalogs and the ratchet baseline.
- **No existing test is edited.** A failing test means English changed. Fix the code, or stop and report. New tests are allowed.
- **Before review**, run, in the worktree:
  1. `scripts/i18n.py extract`;
  2. `scripts/i18n.py update`;
  3. `scripts/i18n.py ratchet --update`, where counts only go down;
  4. `scripts/i18n.py check`;
  5. `scripts/verify.sh`, plus `--browser` for dashboard and web batches.
- **Review:** a reviewer checks that every new msgid equals the old English byte for byte. It also checks the audience of each `t`, and the internal-diagnostic exceptions.
- **Merge:** the orchestrator merges each finished batch into the working branch. Conflicts in generated files (`locales/templates/*.pot`, `locales/ko/LC_MESSAGES/*.po`, `scripts/i18n_ratchet.json`) are resolved by regenerating them on the merged tree, never by hand. Then `scripts/verify.sh` runs again.

**Order.** Two batches run at a time, and two batches that edit the same source file never run together:

| Batch | Can start after | Shares files with |
| --- | --- | --- |
| 1a MNT-66 | prep | 1b (`codes.py` sections only) |
| 1d MNT-69 | prep | 1c, 1e, 1f (Discord modules) |
| 1b MNT-67 | 1a, which sets the error-code pattern | 1a |
| 1c MNT-68 | 1d | 1f, 1i (engine callers in `games_*_discord.py`, `games_ui.py`) |
| 1e MNT-70 | 1d | 1d |
| 1g MNT-72 | prep | 1h (`dashboard.py`) |
| 1h MNT-73 | 1g | 1g |
| 1f MNT-71 | 1c and 1d | 1c, 1d |
| 1i MNT-74 | 1c | 1c |

**End of Phase 1.** The ratchet's `ui` counts are zero except for reviewed exceptions. The `raise` counts that remain are internal diagnostics, each listed in its batch report. Section 7 originally said the allowlist "must be empty". That predates the review fix that counts every exception class, so internal errors now count too.

## 7. Quality gates and automation

| Check | Where | Catches |
| --- | --- | --- |
| ruff `INT001`–`INT003` for `t`, `N_` and `C_` | `ruff check` (verify and CI) | f-strings, `.format` and `%` inside a marker |
| `scripts/i18n.py check` | verify stage "i18n catalogs and string ratchet" and CI (since Phase 0) | Templates out of date with the source (extract and diff); `.po` files that do not parse; placeholder names in `msgstr` that differ from `msgid`; unknown josa specs; disallowed field syntax; obsolete entries; fuzzy entries (a warning only: they are never shipped); a `Plural-Forms` header that differs from CLDR; Discord length limits for `button…`/`select…`/`modal…` contexts in every locale; the same `msgid` translated two ways without `msgctxt`; marker calls in unmapped files; f-strings in `t.n` and `t.p` (the case ruff misses) |
| Ratchet check | `check` | Any **new** bare string passed to a UI or Discord call (`ui.label`, `ui.notify`, `send_message`, `Button(label=)`, and so on) in any module, and any new `raise` with a bare English message. Section 6.1 has the counting rules and gaps. A baseline allowlist shrinks with each Phase 1 batch and must be empty at the end of Phase 1. |
| Pseudo-locale | tests and `screenshot_dashboard.py --lang pseudo` | Every message rendered as `[!! Šéŕṽéŕ šéṫüṗ ~~~ !!]` with about 30% padding. A string left as plain English was not extracted, a broken bracket shows a fragment, and the expansion exposes layout overflow. It is generated, never committed, and enabled only with `LLMCORD_I18N_PSEUDO=1`. |
| Korean smoke tests | unit tests and one browser test file | A slash reply in `ko`, a public table in a `ko` server, the dashboard tab labels with the `ko` cookie, josa rendering |
| English tests | unchanged | Pin current behaviour. Tests run in `en` by default. |

## 8. Adding a language later (runbook)

1. `python scripts/i18n.py init ja` creates `llmcord_core/locales/ja/LC_MESSAGES/*.po` from the templates, with `Plural-Forms` from CLDR.
2. Translate, in Poedit, Weblate or any `.po` editor, using the glossary.
3. Run `python scripts/i18n.py check` and the pseudo-locale and screenshot review.
4. Restart. The pack is found at startup. The dashboard menu shows it as `日本語` (from CLDR), Discord's `ja` maps to it, Quasar's `ja` pack is picked automatically, and the server language select lists it.

There is no code change. An optional `LLMCORD_LOCALES=en,ko` environment allowlist can hide a pack that is not finished. Recognition lexicons (spoken moves, ambient) are a separate track and are not needed for UI support.

## 9. Decisions requested before any code changes

| | Decision | Recommendation |
| --- | --- | --- |
| A | Adopt this architecture as **D39**: gettext with English message ids, Babel at runtime, translate at the edge, the audience decides the language. Add Babel as a dependency. | Yes |
| B | Slash command **names and option names**: keep them English, and localize only descriptions and choices | Keep English for now (docs, tests and muscle memory use `/admin space bind`). Revisit in Phase 5. |
| C | **Schema v22**: `guild_settings.language` (`''` means follow Discord), because non-Community servers report `en-US` | Yes; Korean public messages need it |
| D | Korean register: 해요체 throughout; the glossary in section 5 | Yes; the user reviews the terms marked *to confirm* |
| E | The language switch is a POST and redirect (a full reload), an explicit exception to the ui-guide's no-reload rule | Yes |
| F | Run Phase 1 batches one at a time (the CLAUDE.md default), or allow parallel worktrees for batches on separate files | One at a time, unless speed matters more than review load. **User chose parallel** (2026-10-12). |
| G | New CLAUDE.md invariants once Phase 0 lands: "translate at the edge", "the audience decides the language", "English text is the message id; tests run in `en`" | Yes |

## 10. Risks

| Risk | Mitigation |
| --- | --- |
| Phase 1 changes English text by accident (about 2,000 edits) | Existing tests are the guard: any test edit in Phase 1 stops the batch. Batches stay small and get a reviewer. |
| A missed string ships untranslated | The ratchet check, then the pseudo-locale run in Phase 2 |
| A public message renders in the wrong language through inherited context | An explicit `Translator`, no context variable; audience resolvers are the only way to get a locale; tests for timers and recovery |
| A malicious or broken catalog | The formatter allows bare field names only; `check` validates fields; catalog text never goes into `ui.html` |
| Korean text overflows Discord limits or the phone layout | Limit checks per locale; screenshots at 390 px |
| A NiceGUI upgrade breaks the per-client language override | Pinned by a test from the Phase 0 spike. If it breaks, fall back to setting `lang` on the client side and passing Quasar labels explicitly. |
| A wording change during UI polish leaves Korean stale | `update` marks it fuzzy, fuzzy entries are not shipped (English shows), and `check` reports it |
| Startup cost on the Pi | Measured in Phase 0. If needed, compile `.mo` files in the Docker build. |

## Appendix A: inventory by module

The counts are heuristic (see 1.3). "Errors" counts strings passed to exception constructors. "Commands" counts slash-command metadata.

| Module | Strings | f-strings | Commands | Errors |
| --- | ---: | ---: | ---: | ---: |
| `dashboard.py` | 403 | 24 | | 20 |
| `discord_bot.py` | 188 | 58 | 72 | 15 |
| `admin_store.py` | 168 | 44 | | 91 |
| `games_holdem_discord.py` | 117 | 58 | | |
| `games_doubt_discord.py` | 111 | 43 | 15 | |
| `games_ui.py` | 108 | 4 | | |
| `currency_ui.py` | 65 | 13 | | 1 |
| `games_discord.py` | 65 | 21 | 9 | |
| `presets_ui.py` | 65 | 3 | | 1 |
| `games/blackjack.py` | 39 | 13 | | 7 |
| `games/holdem.py` | 35 | 16 | | 10 |
| `items_discord.py` | 33 | 13 | 4 | |
| `config.py` | 32 | 21 | | 31 |
| `store.py` | 31 | 7 | | 26 |
| `shop_ui.py` | 29 | 3 | | |
| `lore_workspace.py` | 28 | 3 | | |
| `prompts.py` (validation only) | 23 | 0 | | 23 |
| `backend.py` | 22 | 8 | | 1 |
| `lorebooks.py` | 22 | 4 | | 14 |
| `auth.py` | 19 | 1 | | 17 |
| `favorites_discord.py` | 19 | 4 | 11 | |
| `web.py` | 19 | 2 | | 8 |
| `games/registry.py` | 17 | 0 | | |
| `scene_ui.py` | 17 | 4 | | 1 |
| `games/doubt.py` | 15 | 11 | | 5 |
| `avatars.py` | 14 | 0 | | 14 |
| `items_store.py` | 14 | 7 | | 5 |
| `admin.py` | 13 | 4 | | 6 |
| `holdem_store.py` | 12 | 3 | | |
| `engine.py` | 11 | 5 | | 2 |
| `prices.py` | 11 | 0 | | |
| `cards.py` | 10 | 1 | | 10 |
| `items/random_message.py` | 10 | 8 | | 6 |
| `models.py` | 8 | 3 | | 7 |
| `games_spoken_discord.py` | 7 | 2 | | |
| `items/recipe.py` | 7 | 6 | | 7 |
| `savebar.py` | 6 | 1 | | |
| `icons.py` | 5 | 2 | | 1 |
| `games_spoken_doubt.py` | 4 | 0 | | |
| `usage.py` | 4 | 2 | | |
| `world_info.py` | 4 | 3 | | |
| `names.py` | 3 | 3 | | 2 |
| `items/catalog.py`, `items/registry.py` | 4 | 4 | | 4 |
| `budget.py`, `games/base.py` | 2 | 1 | | 1 |
| JS (`lore_drag.js`, ECharts formatter) | 3 | | | |

Text-pinning tests: about 470 assertions with English literals in 69 of 142 test files, and about 1,300 Playwright selectors by name, text or label in `test_dashboard_browser.py`.

## Appendix B: verified behaviour this spec relies on

These were checked in a scratch environment with the pinned versions, without touching the repository.

- **Babel 2.18.0 `pybabel extract`**
  - Extracts `t('…')`, `t.n('…', '…', n)` with the keyword `n:1,2`, `t.p('ctx', '…')` with `p:1c,2`, `self.t('…')`, `N_`, `C_` and `E_:2`. It flags named placeholders as `python-brace-format`.
  - Does **not** extract dotted keywords configured as `B.n`.
  - **Silently skips** `_(f'…')`.
- **Babel CLDR data:** plural forms are 1 for `ko`, `ja`, `zh_TW` and `zh_CN`, and 2 for `en`. `format_list(['A','B','C'], locale='ko')` gives `A, B 및 C`. Language names in their own language: `한국어`, `日本語`, `中文 (繁體, 台灣)`.
- **ruff 0.16.10 with `INT`** and `function-names = ["t", …]`: reports `t(f'…')` (INT001) and `t('…'.format())` (INT002). It does not report `t.n(f'…')`.
- **NiceGUI 3.17.1:** `Client.page` is a plain attribute. `resolve_language()` is called in `build_response` and sets `<html lang>` and the Quasar language pack. `static/lang/ko-KR.umd.prod.js` and the `ko-KR` connection strings ship with it.
- **discord.py 2.6.4:** `Interaction.locale` and `Interaction.guild_locale` (it is `None` outside a guild), `Guild.preferred_locale`, `Locale.korean == 'ko'`, `Locale.japanese == 'ja'`, `zh-CN` and `zh-TW`. `CommandTree.set_translator()` and the `Translator.translate()` hook exist.
