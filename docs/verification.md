# Verification

## Offline suite

From the repository root in a Python 3.12 environment with dependencies installed:

```bash
python -m unittest discover -s tests -v
```

The suite covers character-card JSON/PNG parsing, lorebook formats and sync conflicts, world and hub isolation, World Info activation and branch timing, cast changes, consent, branch rewinds, SQLite migration backups, dashboard OAuth/authorization/CSRF, provider adapter responses and failures, webhook message chains, and Discord-safe line splitting. Offline tests use fake provider and Discord responses; they do not send network requests to your real server.

New checks cover stable lore identities across transfers, manual/imported entry separation, stale revisions, priority zero, v2-to-v3 migration, immutable guild presets, actual post-history placement, one-pass macros, provider adaptations, trimming, prompt/lore depth injection, emotion parsing at arbitrary stream boundaries, avatar publication, and a preset snapshot shared by every speaker in a turn.

## Browser integration

Install `requirements-dev.txt`. On Windows the browser test defaults to installed Microsoft Edge. On other hosts, run `python -m playwright install chromium`; `LLMCORD_BROWSER_CHANNEL` can select an installed Chrome or Edge channel.

In PowerShell:

```powershell
$env:LLMCORD_BROWSER_TESTS = '1'
python -m unittest discover -s tests -p test_dashboard_browser.py -v
```

On Linux/macOS:

```bash
LLMCORD_BROWSER_TESTS=1 python -m unittest discover -s tests -p test_dashboard_browser.py -v
```

The test starts a disposable localhost HTTPS service with mocked Discord responses and an in-memory database. It checks the real NiceGUI assets/security policy, live events, space creation, paginated lore dragging, prompt block editing/reordering, preset activation and export, sample previews, character-card uploads, avatar uploads/publication, and revoked/expired sessions. It does not use real Discord or model credentials. Test fixture entry points under `tests/` must never be used as production servers.

## Private Discord checklist

Run these checks in a private text channel before inviting the bot into a busy server or enabling ambient mode:

1. Start the bot and dashboard, log in through the Pi's Tailscale URL, and confirm a non-admin Discord account cannot open the server dashboard.
2. Create a world and bind the private channel. Import a JSON card and a PNG card through the dashboard; confirm the preview, avatar, and home-world assignment.
3. Set the default cast, mention the bot, and confirm each line appears under the intended character webhook identity. Reply to one line and verify it routes back to the scene.
4. Create World A, World B, and a hub. Link both worlds, summon guests, and check `/context`. Confirm a World A guest receives its own world lore and hub/channel lore without World B-only lore.
5. Import a named guild lorebook and a channel book. Preview a reimport after a local edit, resolve the conflict, and check which entries `/context` reports.
6. Reply to an older character line after more messages have arrived. Confirm the new line follows the earlier branch rather than recalling later events. Check a thread separately from its parent channel.
7. Test `/memory opt_in`, `/memory list`, `/memory forget`, and `/memory opt_out` with a test account. Confirm opt-out removes its saved personal facts.
8. Restart both services and repeat a reply and `/context` lookup to check SQLite and webhook recovery.
9. Enable ambient mode only after explicit turns work; verify the two-human-message threshold, cooldown, and silent director behavior.
10. Open all five NiceGUI sections through Tailscale. Test a real card import conflict, a lore move/copy, and a large-book search/page change. Sign out while another console tab is open and verify it cannot mutate data.
11. Configure a private avatar asset channel, upload/publish two emotions, and verify per-message avatars in a channel and concurrent threads. Remove a backing asset, restart the bot, verify default fallback, and repair it from the console.
12. Import a SillyTavern Chat Completion preset, select its order profile, resolve compatibility mappings, save/activate it, and verify the next turn's `/context` trace. Activate another revision during a multi-speaker turn and verify that turn retains its original revision.

The code has not been verified against a live Pi, Tailscale OAuth callback, or Discord channel from this repository alone. These checks require your host, Discord application, and test server.
