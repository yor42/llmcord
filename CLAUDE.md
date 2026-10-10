# llmcord

Discord roleplay "skit" bot (characters speak through webhooks, director model picks speakers, SQLite-held scene trees) plus a private admin dashboard (FastAPI + NiceGUI under `/admin`, Discord OAuth, served via Tailscale). Self-hosted on a Raspberry Pi 5 (arm64), Python 3.12/3.13.

## Working model: orchestrator + agents
The main session **orchestrates; it does not edit application code directly.** For any change to `llmcord_core/`, entry points, scripts, or tests, follow the `orchestrate-change` skill:
brief → `test-writer` if coverage is needed → `implementer`, `ui-polisher` or `quick-editor` → `verify-runner` → `change-reviewer` → report.
The main session may directly edit docs, `CLAUDE.md`, and `.claude/` config.

**Current mode: maintenance and UI polish** (since 2026-10-08; the R1–R6 hardening rework is closed). Work comes from the user or `docs/engineering/backlog.md` (`MNT-NN`, `UI-NN`, `FEAT-NN`), one small item at a time. UI items also follow the `ui-polish` skill and `docs/engineering/ui-guide.md`. Anything larger than one item, a schema change, or a change to an invariant below needs the user's go-ahead first.

Use Sonnet for routine coordination, implementation, tests, and review; Haiku for bounded exploration (`Explore`), exact mechanical edits (`quick-editor`), and verification summaries (`verify-runner`). Reserve Opus for difficult reasoning or review of security, isolation, consent, migration, or concurrency changes. Each project agent has an explicit model and turn limit. Follow the routing and escalation rules in `orchestrate-change`; do not override agents to match the parent model or start agent teams by default. A directly answerable lookup needs no agent. See `docs/engineering/claude-workflow.md` for the audit, provider/version caveats, and how to check actual model usage.

## Processes (no IPC — they share one SQLite file)
- `llmcord.py` → `discord_bot.SkitBot`: events, slash commands (`register_commands`), webhooks, daily cleanup task.
- `web_main.py` → `web.create_app`: OAuth, avatar images, mounts NiceGUI (`dashboard.mount_dashboard`).
- `migrate.py` → `Store(path)`: creates/upgrades schema (`PRAGMA user_version`, currently 15; backs up before upgrading).
The bot reads DB state fresh each turn, so dashboard edits apply on the next turn.

## Module map (`llmcord_core/`)
| Area | Files |
| --- | --- |
| Discord routing, commands, delivery | `discord_bot.py`, `identity.py`, `usage.py` |
| Turn engine (director, prompt, memory) | `engine.py`, `prompts.py`, `world_info.py`, `lore.py` |
| Model providers | `models.py`, `config.py` (profile validation, key pins), `backend.py` (dashboard profiles merged over `config.yaml`, per-turn snapshot), `errors.py` |
| Persistence | `store.py` (scene/lore/core), `admin_store.py` (presets, lore identities, avatars, revisions) |
| Dashboard | `web.py`, `auth.py`, `admin.py`, `dashboard.py`, `scene_ui.py`, `lore_workspace.py`, `lore_drag.{py,js}` |
| Imports/assets | `cards.py`, `lorebooks.py`, `avatars.py` |
| Minigames (FEAT-19) | `games/` (pure seeded rules engine, blackjack), `games_discord.py` (`/blackjack`, buttons, per-table locks and timers); store API in `admin_store.py`; dashboard settings in `currency_ui.py` |

## Commands
- Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt` (browser tests also need `.venv/bin/python -m playwright install chromium`).
- **Verify before declaring a change done:** choose the level in the `verify` skill. Application/script/test changes require `scripts/verify.sh` — ruff, repo hygiene, compileall, offline unittest suite. Docs and Claude config changes require metadata/diff checks instead.
  - `scripts/verify.sh --browser` adds the Playwright dashboard suite (~2 min) — required when touching `web.py`, `auth.py`, `dashboard.py`, `scene_ui.py`, `lore_*`.
  - `scripts/verify.sh --bench` adds `scripts/bench_dashboard.py` — required when claiming a dashboard perf change.
  - Every level also runs a Gitleaks secret scan when `$GITLEAKS` or `gitleaks` on PATH exists (CI always does); otherwise the summary shows SKIP.
- Single test file: `.venv/bin/python -m unittest discover -s tests -p test_slash_commands.py -v`
- Tests are stdlib `unittest` (not pytest). "expected failures" in output are documented known defects, not regressions.

## Invariants (break these and it is a bug)
- **Guild isolation:** every store read/write is scoped by `guild_id`; dashboard actions go through `AuthService.guard` / `AdminService.run` and `validate_owner`.
- **World/hub lore isolation:** a hub guest gets its own world's lore + hub/channel lore, never another linked world's private lore.
- **Consent:** personal facts are stored only after `/memory opt_in`; opt-out deletes them.
- **Branches:** prompts follow `ancestors(parent_id)`; later sibling nodes are never visible to a rewind.
- **Preset snapshot:** a scene captures one preset revision before its first model call; all speakers in that turn use it.
- **Model snapshot and key pins:** a turn resolves model profiles once (`BackendResolver.pin`), and the memory work it starts uses the same snapshot. A dashboard profile's `${NAME}_API_KEY` is sent only to a host pinned for that key (`check_key_pin`, at save and again before every call); key values are never stored, shown, audited or logged.
- **Admin writes:** use `store.write_admin()` + revision checks (`ConflictError`), never blind overwrite.
- **Schema:** bump `user_version` with a backup-before-upgrade migration; never edit existing migrations destructively.

## Care areas
- `dashboard.py` monkey-patches NiceGUI socket handlers for auth — changes there affect every live event.
- All sqlite calls are synchronous on the event loop; the bot holds a per-channel lock across all model calls.
- Never read, print, or edit `.env`, `config.yaml`, `*.key`/`*.crt`, or anything under `data/` (real user data). Tests use `:memory:` or temp dirs and fake Discord/model transports only.
- Don't delete "dead" code or legacy routes without grep + a characterization test. NiceGUI is the only admin write path; the legacy Jinja routes were retired in R6 (SEC-02).

## Engineering docs
`docs/engineering/` — `backlog.md` (current `MNT-*`/`UI-*`/`FEAT-*` items and decisions from D13), `ui-guide.md` (dashboard and reply conventions), system map, perf baseline, Claude workflow. `history/` holds the closed hardening audit (finding IDs like `BUG-01`, `PERF-01`, `SEC-01` that test docstrings cite), the R1–R6 roadmap with decisions D1–D12, and the first verification baseline. User docs live in `docs/` (words: `docs/glossary.md`).
