# Native Pi hosting and private Discord testing

This path uses a virtual environment and two systemd user services on 64-bit Debian, without Docker. Python 3.13 is supported by the installed ARM64 dependencies. Both services restart after failures, share SQLite, and read credentials from the ignored `.env` file. The web service binds to `127.0.0.1:8080`.

## Install and configure

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp config-gemini.yaml config.yaml
cp .env.example .env
chmod 600 .env
.venv/bin/python -m scripts.install_services
```

Copy the example files only for an initial setup; preserve your existing configuration on upgrades. The Gemini preset selects Google AI Studio's `gemini-3.8-flash` for dialogue, director, and memory through Google's [OpenAI-compatible endpoint](https://ai.google.dev/gemini-api/docs/openai). It uses low reasoning effort and a 4096-token output budget to leave room for thinking and structured results. `structured_outputs: true` enables native JSON schemas for director and memory calls; leave it off for compatible servers that lack JSON schema support. The application input budget remains 12000 tokens. Credentials are mandatory for this cloud compatible profile; local keyless profiles still work.

Edit `.env` privately on the Pi and set:

| Variable | Value |
| --- | --- |
| `DISCORD_BOT_TOKEN` | Bot token from the Discord application |
| `DISCORD_GUILD_ID` | Private server ID; enables immediate guild command registration |
| `GEMINI_API_KEY` | Google AI Studio API key with access/quota for the selected model |
| `DISCORD_CLIENT_ID` | Application ID of the same Discord bot, for dashboard OAuth |
| `DISCORD_CLIENT_SECRET` | OAuth client secret, for dashboard OAuth |
| `WEB_BASE_URL` | Exact Tailscale HTTPS origin, including a non-default port if used |
| `LLMCORD_DATABASE_PATH` | `data/llmcord.sqlite3` for all processes |

The bot can run with the first three variables alone. Dashboard OAuth additionally needs the client ID, secret, and HTTPS origin. The bot's environment database path overrides YAML, preventing the bot and dashboard from using different databases. The guild ID controls command registration; Discord channel permissions control who can interact with the bot.

User services read `.env` through systemd's `EnvironmentFile`. Use plain `NAME=value` assignments with no `export` prefix, shell substitutions, or commands. Restart services after changing it. A variable exported in an SSH terminal does not automatically reach an already running service. For terminal checks, load your trusted `.env`:

```bash
set -a
. ./.env
set +a
.venv/bin/python -m scripts.check_host --require-guild
.venv/bin/python -m scripts.check_host --require-guild --live
```

`--live` authenticates the Discord bot, checks its application ID and message-content intent, confirms private-server membership, then makes small billable model calls for director JSON, memory JSON, and streaming dialogue. It does not post Discord messages or change server data. Model output and credentials are omitted from its report. Channel permissions, webhook delivery, OAuth login, images, and emotion avatars need the interactive checks below.

## Discord application and invite

In the [Discord developer portal](https://discord.com/developers/applications), create/select the bot application. Enable **Message Content Intent** on its Bot page. Turn off Public Bot for a private application. Under OAuth2 URL Generator select `bot` and `applications.commands`, then grant View Channels, Send Messages, Send Messages in Threads, Read Message History, Attach Files, Embed Links, and Manage Webhooks. Invite it to your private server. Enable Developer Mode in Discord and copy the server ID into `.env`.

Allow those permissions only in your test channels and avatar asset channel. Give the bot access to private threads used for testing. Administrator permission is not required for the bot. Your dashboard account must be a server administrator.

For this Pi the existing main Tailscale HTTPS route belongs to another application. Use a separate listener:

```text
WEB_BASE_URL=https://my-pi.my-tailnet.ts.net:8443
```

Register this exact OAuth redirect on the same Discord application:

```text
https://my-pi.my-tailnet.ts.net:8443/auth/callback
```

Check existing proxy settings before applying a listener on another host:

```bash
tailscale serve status
tailscale serve --bg --https=8443 http://127.0.0.1:8080
```

If Tailscale requires administrator privileges, run the Serve command with `sudo` on the Pi. Use Serve (private to the tailnet). The existing port 443 proxy is preserved. Join your browser device to the same tailnet and allow port 8443 in any custom tailnet access rules.

## Start and maintain

```bash
systemctl --user enable --now llmcord-bot.service
# After configuring OAuth and the private HTTPS listener:
systemctl --user enable --now llmcord-web.service
systemctl --user status llmcord-bot llmcord-web
journalctl --user -u llmcord-bot -u llmcord-web -n 80 --no-pager
```

User lingering must be enabled so services run at boot and after SSH logout. Check `loginctl show-user "$USER" -p Linger`; this Pi already has `Linger=yes`. On a new host, an administrator can enable it with `sudo loginctl enable-linger "$USER"`. Installation alone does not enable/start services; credentials must be set first.

Missing credentials fail the pre-start check. After correcting them, run:

```bash
systemctl --user reset-failed llmcord-bot llmcord-web
systemctl --user restart llmcord-bot llmcord-web
```

To upgrade, stop both services, back up the entire `data/` directory, update code/dependencies, and restart. SQLite migrations also create backups when upgrading an existing schema. Do not delete Discord webhook or avatar asset messages between restart tests.

## First real conversation

Keep ambient mode off during explicit testing. In a private test text channel:

1. Run `/space create` with kind `world` and name `Test World`.
2. Run `/space bind` with the current channel and `Test World`.
3. Run `/character import` with `Test World` and upload `examples/cards/mira.json` (or your own JSON/PNG card). Confirm the actual character name with `/character list`.
4. Run `/cast default` with that name, then mention the bot with a short greeting. A generation status appears before model calls and updates to name the current character. It disappears after replies finish, or becomes an error message if generation fails. Ambient turns show status only when a speaker is selected.
5. Confirm the answer uses the character's webhook name. Reply to it and check `/context` for the resulting line.
6. Restart the bot service, then reply again. Confirm the cast, previous scene, and webhook identity survive.

Continue with the [full live checklist](verification.md#private-discord-checklist): worlds/hub isolation, thread routing and older-message branches, memory opt-in/out, import conflicts, lore moves and copies, prompt activation, dashboard session expiry, published emotion avatars, split messages, and ambient mode. Record each result; API preflight success alone does not establish these behaviors.

Failed replies report the failing step, exception type, and provider or Discord error detail, including HTTP status/code when available. Credentials, URLs, and full provider response bodies are omitted. Service logs include stack locations and detailed warnings for director or memory fallbacks.

Offline checks on the Pi:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Browser tests require the optional development dependencies and Chromium; see [verification](verification.md#browser-integration).

On a Pi with system Chromium installed, you can use it without a separate browser download:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
LLMCORD_BROWSER_TESTS=1 LLMCORD_BROWSER_EXECUTABLE=/usr/bin/chromium .venv/bin/python -m unittest discover -s tests -p test_dashboard_browser.py -v
```
