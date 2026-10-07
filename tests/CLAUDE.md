# Test conventions

- stdlib `unittest`; async tests use `IsolatedAsyncioTestCase`. Run from repo root with `discover -s tests` (this puts `tests/` on `sys.path`, so import helpers as `from helpers import ...`).
- New tests use `tests/helpers.py`: `make_settings()`, `FakeModels`, `FakeInteraction` + `invoke(bot, "lore add", interaction, ...)` for slash commands (runs real permission checks and the tree's error handler), `discord_transport()` (counting httpx MockTransport) + `install_session()` for web tests. Older files keep local fakes; don't mass-migrate them unasked.
- Never touch the network or real data: `Store()` is in-memory; web tests pass `httpx.AsyncClient(transport=...)` and `config_path="tests/nonexistent-config.yaml"`.
- **Characterization tests** pin current behavior (even odd behavior) and say so in the docstring with a finding ID, e.g. `"""Characterization (UX-03): ..."""`.
- **Known objectively-incorrect behavior** is a `@unittest.expectedFailure` test named `test_known_defect_*` asserting the *intended* behavior, docstring starting with the finding ID (`BUG-01: ...`). When a fix lands, remove the decorator in the same change — "unexpected success" means someone fixed it without updating the test.
- Verify an expected-failure fails for the stated reason (run the body manually once), not because of a typo.
- Count-based perf characterizations (e.g. Discord calls per request in `test_web_auth_boundaries.py`) must be updated deliberately alongside `docs/engineering/perf-baseline.md`.
- Browser suite: `test_dashboard_browser.py` + fixture server `dashboard_server.py` (mock Discord, `:memory:` DB, `/_test/*` control endpoints; `LLMCORD_BENCH*` env vars are bench-only). Opt-in via `LLMCORD_BROWSER_TESTS=1`; failure artifacts go to `.test-artifacts/`.
