# Getting started

This guide runs the Discord bot locally. The dashboard also needs Discord OAuth and an HTTPS URL; see [Raspberry Pi deployment](deployment-raspberry-pi.md) for that setup.

## Requirements

- Python 3.12.
- A Discord application and bot token.
- A model reachable through OpenAI, Anthropic, or an OpenAI-compatible endpoint. The selected model profiles must support the calls used for dialogue, director decisions, and memory extraction.
- A Discord server where you can invite the bot and manage its channels.

In the Discord developer portal, enable **Message Content Intent** for the bot. Invite it with access to the intended channels and threads, including **View Channel**, **Read Message History**, **Send Messages**, **Use Application Commands**, and **Manage Webhooks**. Character lines are posted through per-character webhooks.

## Install and configure

From the repository root:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp config-example.yaml config.yaml
```

On Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1` and copy the file with `Copy-Item config-example.yaml config.yaml`.

Edit `config.yaml`:

1. Set `models.dialogue`, `models.director`, and `models.memory` to profile names under `models.profiles`. They may all use one profile.
2. Replace the selected profile's placeholder `model` name with a model available to your account or endpoint.
3. Set `discord.development_guild_id` to your server ID while testing. The bot then syncs commands to that server during startup.
4. Keep `database_path` at `data/llmcord.sqlite3` unless you have a reason to move it.

Set credentials as environment variables; do not put them in `config.yaml` or commit them. For example, in PowerShell:

```powershell
$env:DISCORD_BOT_TOKEN = "your-bot-token"
$env:OPENAI_API_KEY = "your-api-key"
python llmcord.py
```

For Anthropic, select an `anthropic` profile and set its `api_key_env` variable. For Ollama or another OpenAI-compatible server, select a `compatible` profile and set `base_url` to the endpoint's OpenAI-style `/v1` URL. A compatible endpoint does not require a cloud API key in the current config loader. Image attachments require `supports_images: true` on the dialogue profile and actual image support from that model.

Run the bot with `python llmcord.py`. It creates the SQLite database if needed. To use the dashboard locally as well, it still requires a valid HTTPS `WEB_BASE_URL` for secure OAuth cookies; the [Pi guide](deployment-raspberry-pi.md) shows the intended Tailscale setup.

## First server setup

Create three Discord text channels such as `#world-a`, `#world-b`, and `#hub`. As a server administrator:

1. Run `/admin space create` three times: worlds `A` and `B`, then hub `Hub`.
2. Run `/admin space bind` in any channel, choosing each Discord channel and its matching space.
3. Run `/admin space allow_world` twice to link `A` and `B` to `Hub`.
4. Import a V2/V3 card with `/admin character import`, choosing its home world. The [example cards](../examples/cards) are small JSON samples.
5. Use `/admin cast default` in each channel to choose its default cast. Names are comma separated; the maximum cast size is five.
6. Mention the bot or reply to a character line in a bound channel. Use `/context` after a response to inspect the saved lore and memory references.

Members can use `/cast set`, `/cast add`, and `/cast remove` to adjust the active cast, or `/summon` to invite one eligible character for a turn. See the [server guide](server-guide.md) for the complete command map. Leave ambient mode off until explicit turns work in a private test channel.

## Configuration reference

| Setting | Meaning |
| --- | --- |
| `discord.token_env` | Name of the environment variable containing the bot token; defaults to `DISCORD_BOT_TOKEN`. |
| `discord.development_guild_id` | Optional server ID for fast development command sync. |
| `database_path` | SQLite file used by the bot, dashboard and migration (`LLMCORD_DATABASE_PATH` overrides it). |
| `history_retention_days` | Age limit for stored conversation text and response traces; default `90`. |
| `models.dialogue` | Profile used for character lines and image descriptions. |
| `models.director` | Profile used for structured speaker selection. |
| `models.memory` | Profile used for summaries and structured fact extraction. |
| `models.profiles.<name>.provider` | `openai`, `anthropic`, or `compatible`. |
| `models.profiles.<name>.model` | Provider-specific model ID. |
| `models.profiles.<name>.context_tokens` | Approximate context window used for prompt budgeting. |
| `models.profiles.<name>.supports_images` | Enables bounded image attachments for dialogue. |
| `models.profiles.<name>.api_key_env` | Environment variable holding a cloud provider key. |
| `models.profiles.<name>.base_url` | OpenAI-compatible endpoint URL, or an optional OpenAI API override. |
| `models.profiles.<name>.timeout_seconds` | Seconds a model request may wait for data (first token or next chunk) before failing; a stream that keeps sending is not cut off. Default `120`; raise it for slow local models, where prompt processing before the first token counts. |
| `models.profiles.<name>.max_retries` | Automatic retries after a failed or timed-out request; default `1` (`0` disables; consider `0` for slow local models so a slow request is not repeated). |

The `limits` block controls input/output token budgets, image count and attachment size, speaker count, nearby-message count and time window, and ambient cooldown. Use [config-example.yaml](../config-example.yaml) for the exact keys and defaults. `max_speakers` cannot exceed three.

Memory limits (all optional, positive whole numbers):

| Key | Meaning |
| --- | --- |
| `limits.memory_input_tokens` | Estimated tokens of transcript sent to one scene-summary or memory-extraction call; default `6000`. Long unsummarized stretches are summarized from their most recent messages that fit. |
| `limits.memory_output_tokens` | Maximum output tokens for a scene-summary call; default `550`. |
| `limits.summary_every_messages` | Summarize only once at least this many messages follow the last saved summary on the branch; `1`–`100`, default `1` (every turn). |
| `limits.extraction_every_turns` | Run memory extraction only on every Nth user turn of the branch; `1`–`100`, default `1` (every turn). |
