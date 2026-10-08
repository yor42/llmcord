# llmcord

Discord roleplay "skit" bot (characters speak through webhooks, director model picks speakers, SQLite-held scene trees) plus a private admin dashboard (FastAPI + NiceGUI under `/admin`, Discord OAuth, served via Tailscale). Self-hosted on a Raspberry Pi 5 (arm64), Python 3.12/3.13.

## Working model: orchestrator + agents
The main session **orchestrates; it does not edit application code directly.** For any change to `llmcord_core/`, entry points, or tests, follow the `orchestrate-change` skill:
brief → `test-writer` (pin behavior) → `implementer` (change) → `scripts/verify.sh` → `change-reviewer` (read-only) → report.
The main session may directly edit docs, `CLAUDE.md`, and `.claude/` config. Large rework needs explicit user approval of the roadmap first.

## Processes (no IPC — they share one SQLite file)
- `llmcord.py` → `discord_bot.SkitBot`: events, slash commands (`register_commands`), webhooks, daily cleanup task.
- `web_main.py` → `web.create_app`: OAuth, legacy Jinja form routes, mounts NiceGUI (`dashboard.mount_dashboard`).
- `migrate.py` → `Store(path)`: creates/upgrades schema (`PRAGMA user_version`, currently 4; backs up before upgrading).
The bot reads DB state fresh each turn, so dashboard edits apply on the next turn.

## Module map (`llmcord_core/`)
| Area | Files |
| --- | --- |
| Discord routing, commands, delivery | `discord_bot.py`, `identity.py`, `usage.py` |
| Turn engine (director, prompt, memory) | `engine.py`, `prompts.py`, `world_info.py`, `lore.py` |
| Model providers | `models.py`, `config.py`, `errors.py` |
| Persistence | `store.py` (scene/lore/core), `admin_store.py` (presets, lore identities, avatars, revisions) |
| Dashboard | `web.py`, `auth.py`, `admin.py`, `dashboard.py`, `scene_ui.py`, `lore_workspace.py`, `lore_drag.{py,js}`, `templates/` |
| Imports/assets | `cards.py`, `lorebooks.py`, `avatars.py` |

## Commands
- Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt` (browser tests also need `.venv/bin/python -m playwright install chromium`).
- **Verify (run before declaring any change done):** `scripts/verify.sh` — ruff, repo hygiene, compileall, offline unittest suite.
  - `scripts/verify.sh --browser` adds the Playwright dashboard suite (~2 min) — required when touching `web.py`, `auth.py`, `dashboard.py`, `scene_ui.py`, `lore_*`, templates.
  - `scripts/verify.sh --bench` adds `scripts/bench_dashboard.py` — required when claiming a dashboard perf change.
- Single test file: `.venv/bin/python -m unittest discover -s tests -p test_slash_commands.py -v`
- Tests are stdlib `unittest` (not pytest). "expected failures" in output are documented known defects, not regressions.

## Invariants (break these and it is a bug)
- **Guild isolation:** every store read/write is scoped by `guild_id`; dashboard actions go through `AuthService.guard` / `AdminService.run` and `validate_owner`.
- **World/hub lore isolation:** a hub guest gets its own world's lore + hub/channel lore, never another linked world's private lore.
- **Consent:** personal facts are stored only after `/memory opt_in`; opt-out deletes them.
- **Branches:** prompts follow `ancestors(parent_id)`; later sibling nodes are never visible to a rewind.
- **Preset snapshot:** a scene captures one preset revision before its first model call; all speakers in that turn use it.
- **Admin writes:** use `store.write_admin()` + revision checks (`ConflictError`), never blind overwrite.
- **Schema:** bump `user_version` with a backup-before-upgrade migration; never edit existing migrations destructively.

## Care areas
- `dashboard.py` monkey-patches NiceGUI socket handlers for auth — changes there affect every live event.
- All sqlite calls are synchronous on the event loop; the bot holds a per-channel lock across all model calls.
- Never read, print, or edit `.env`, `config.yaml`, `*.key`/`*.crt`, or anything under `data/` (real user data). Tests use `:memory:` or temp dirs and fake Discord/model transports only.
- Don't delete "dead" code or legacy routes without grep + a characterization test; legacy Jinja routes are still live.

## Engineering docs
`docs/engineering/` — system map, audit (finding IDs like `BUG-01`, `PERF-01`, `SEC-01`), roadmap, baselines. Test docstrings reference these IDs. User docs live in `docs/`.
