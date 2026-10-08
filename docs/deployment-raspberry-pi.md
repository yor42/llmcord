# Deploy on a Raspberry Pi 5

This deployment runs the bot and admin dashboard as separate Docker Compose services against one SQLite database. The web service listens on the Pi's loopback interface and Tailscale Serve publishes it to your tailnet over HTTPS.

For a native installation without Docker, including a Gemini 3.8 Flash configuration, use [Native Pi hosting and live testing](live-testing-pi.md).

The image includes the application code. Compose mounts only private configuration and persistent data. To use a published version instead of building on the Pi, follow [GitHub CI and container releases](github-ci.md#use-a-published-image-on-a-pi).

## Before you start

- Use a **64-bit** Raspberry Pi OS. Docker directs 64-bit Raspberry Pi OS users to its [Debian arm64 installation guide](https://docs.docker.com/engine/install/debian/).
- Install Docker Engine with the Compose plugin and install Tailscale on the Pi **host**. Join the Pi and the device you will use for administration to the same tailnet.
- Enable [MagicDNS and HTTPS certificates](https://tailscale.com/docs/how-to/set-up-https-certificates) for that tailnet. Note the Pi's full `https://...ts.net` name.
- Create a Discord application with a bot, enable Message Content Intent, and invite the bot with the permissions in [Getting started](getting-started.md#requirements).

The actual ARM64 container build, OAuth callback, and Discord webhook behavior still need validation on your Pi and private server; the repository's automated tests run offline.

## Configure the repository

In the repository root on the Pi:

```bash
cp config-example.yaml config.yaml
cp .env.example .env
mkdir -p data
chmod 600 .env
```

Edit `config.yaml` to select real model IDs for `dialogue`, `director`, and `memory`. Edit `.env` with `DISCORD_BOT_TOKEN`, `DISCORD_CLIENT_ID`, `DISCORD_CLIENT_SECRET`, `WEB_BASE_URL`, and any selected cloud model API keys. Set `WEB_BASE_URL` to the Pi's exact Tailscale HTTPS origin, with no trailing slash or path, for example `https://my-pi.my-tailnet.ts.net`.

All services resolve the database path the same way: `LLMCORD_DATABASE_PATH` when set, otherwise `database_path` from `config.yaml`, otherwise `data/llmcord.sqlite3`. If you set `database_path` in `config.yaml` without the environment variable, note that before this rule was unified the dashboard and migration ignored it and used `data/llmcord.sqlite3`; copy or merge that file first if they ever diverged. Both containers mount `./data` for SQLite, including its WAL files and migration backups.

Do not commit `.env`, `config.yaml`, the database, or its backups. These paths are covered by `.gitignore` or live under the ignored `data/` directory.

### Model endpoint from a container

If Ollama runs on the Pi host, set a compatible profile's `base_url` to `http://host.docker.internal:11434/v1`. Compose maps that hostname to the host gateway for the bot. Ollama must listen on an address the container can reach. If the model runs on another machine, use that machine's reachable LAN or tailnet URL. `localhost` from inside the bot container refers to the container, not the host.

## Set up Discord OAuth

In the Discord developer portal for the **same application as the bot**, register this exact redirect URI, replacing the example hostname:

```text
https://my-pi.my-tailnet.ts.net/auth/callback
```

The scheme, hostname, and path must match `WEB_BASE_URL` plus `/auth/callback`. The dashboard asks Discord for `identify` and `guilds` and checks server administrator permission on every protected request. It uses a browser-bound OAuth state, secure HTTP-only session cookie, and re-checks administrator permission on every change; uploads and sign-out also require a CSRF token. A Tailscale login alone does not grant dashboard access; the Discord account also needs administrator permission in the server.

## Start and check services

```bash
docker compose up --build -d
docker compose ps
docker compose logs --tail=100 migrate bot web
```

Compose starts `migrate` first, then `bot` and `web`. The web port maps to `127.0.0.1:8080` on the Pi. Run Tailscale Serve **on the Pi host**:

```bash
sudo tailscale serve --bg 8080
tailscale serve status
```

Tailscale's [Serve command](https://tailscale.com/docs/reference/tailscale-cli/serve) proxies the local port to the Pi's tailnet HTTPS name; `--bg` retains the configuration across a restart. Open `WEB_BASE_URL` from a tailnet device and complete Discord login. You do not need public router port forwarding.

Use the dashboard to create spaces and bind channels, or use the matching Discord commands. Start in a private channel, import one card, set a cast, and mention the bot. Confirm webhook identity, reply routing, `/context`, and a restart before enabling ambient mode.

## Backups and upgrades

Before an upgrade, stop writers and copy the entire `data/` directory, including any SQLite `-wal` and `-shm` files:

```bash
docker compose stop bot web
cp -a data ../llmcord-data-backup
docker compose up --build -d
```

The example keeps the manual backup outside the Git repository; use a new destination name for each backup. The migration service runs before the other services. When it encounters an existing database below the current schema version, it also writes a dated `*.pre-v3-*`, `*.pre-v4-*` or `*.pre-v5-*.sqlite3` backup beside the database before changing the schema. Keep your own backup as well. If you change `LLMCORD_DATABASE_PATH` or `database_path`, update both together before restarting.

## Common problems

| Symptom | Check |
| --- | --- |
| Discord rejects the OAuth callback | `WEB_BASE_URL` and the registered redirect URI use the same exact HTTPS hostname and `/auth/callback` path. |
| Dashboard says administrator permission is required | Sign in with a Discord account that is an administrator of that server. |
| Dashboard shows no Discord channels | Confirm the bot token in `.env` and that the bot is in that server. |
| Login works, then a restart signs you out | Sessions live in the web process's memory; sign in again. |
| Character response reports a webhook problem | Give the bot Manage Webhooks in that text channel and check its webhook capacity. |
| Model requests cannot reach Ollama | Check `base_url` from inside the bot container and Ollama's listening address. |
| A book entry never activates | Check that the book is enabled for this space or bound to the channel, its keywords match, and its rule has no SillyTavern-only warning. |
| Bot never speaks ambiently | Check the channel's ambient toggle, current cast, two-message threshold, cooldown, and the director's invitation decision. |
