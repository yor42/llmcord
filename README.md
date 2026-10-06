# llmcord

A Discord character bot for small group skits. Channels can belong to a **world** or a **hub**. Worlds have their own lore and home characters; a hub can invite characters from selected worlds without mixing their private world lore. Every channel can add local lore and a cast. Threads inherit their parent channel's space and lore, while keeping their own cast and conversation branches.

## Documentation

Start with the [documentation index](docs/README.md). It links to [getting started](docs/getting-started.md), the [server guide](docs/server-guide.md), [lore and memory](docs/lore-and-memory.md), [Raspberry Pi deployment](docs/deployment-raspberry-pi.md), [architecture](docs/architecture.md), and [verification](docs/verification.md).

## Set up

For native Raspberry Pi hosting with Google AI Studio and Gemini 3.8 Flash, use the [Pi hosting and live-testing guide](docs/live-testing-pi.md) and `config-gemini.yaml`.

1. Use Python 3.12 or 3.13 and install the pinned packages: `python -m pip install -r requirements.txt`.
2. Copy `config-example.yaml` to `config.yaml`. Set the `dialogue`, `director`, and `memory` model profiles and model names. Only profiles selected for those roles need credentials. For Ollama or another OpenAI-compatible server, set `provider: compatible` and its `base_url`.
3. Set `DISCORD_BOT_TOKEN` and the API key environment variable for each selected cloud profile. Keep tokens out of the YAML file and version control.
4. In the Discord developer portal, enable **Message Content Intent**. Invite the bot with permissions to read and send messages, read history, use application commands, and manage webhooks in the channels where characters will speak. Give it access to threads that will host scenes.
5. Run `python llmcord.py`. Set `discord.development_guild_id` in `config.yaml` while testing to sync slash commands quickly to one server.

The database is created at `data/llmcord.sqlite3` by default. Run `python -m unittest discover -s tests -v` for offline tests. Docker users can copy `.env` with their credentials and run `docker compose up --build`; a model server on the Docker host may need `host.docker.internal` in its `base_url`.

## Set up a server

For three channels, `#world-a`, `#world-b`, and `#hub`:

1. `/space create world A`, `/space create world B`, `/space create hub Hub`.
2. Bind each text channel using `/space bind`.
3. `/space allow_world Hub A` and `/space allow_world Hub B`.
4. Import cards with `/character import` and the home world name. JSON and PNG V2/V3 cards work; PNGs supply the avatar. The `examples/cards` directory has two small example cards.
5. Set a channel's default cast with `/cast default` and comma-separated character names. Members can use `/cast set`, `/cast add`, and `/cast remove` to change the active cast in a channel or thread. A hub's `/summon` can invite any eligible guest for one turn without changing that cast.
6. Mention the bot or reply to one of its character lines to begin. `/ambient on` is an optional admin setting per channel. Ambient participation waits for at least two human messages and a 120-second cooldown, and the director can stay silent. Test explicit turns first.

World channels only admit their home world's characters. Hub channels admit characters from linked worlds, but only active cast members join ambiently. Threads inherit their parent channel's space, lore, and ambient setting; they keep separate casts and message histories.

## Memory and lore

Character prompts include the card, relevant character and home-world lore, the current hub and channel lore, and branch history. A hub guest does not receive another guest's world lore. Character encounters are scoped to a space. A person's memories can follow the same character between its home world and a hub only after `/memory opt_in`. `/memory list`, `/memory forget`, and `/memory opt_out` give that person control; opting out erases their personal facts.

The bot extracts short scene facts after replies. A repeated fact in two separate scenes becomes durable lore **in that channel or thread only**. Admins can use `/lore add`, `/lore edit`, `/lore pin`, `/lore delete`, and `/lore promote` to control shared facts and explicitly copy them into a channel, world, or hub. `/context` shows which lore and memories informed a saved character line.

Replies to an older character line branch from that line's saved parent chain. A new mention can also use up to 12 recent human messages from the last 10 minutes as group context. `/scene reset` starts the next invitation without the channel's recent context. Admins can use `/scene delete` to remove a stored scene and its branches. Conversation text and response traces expire after 90 days by default; change `history_retention_days` in `config.yaml`. Durable lore and opted-in personal facts stay until removed with their controls. Application logs do not include raw chat transcripts.

## Model providers

The `openai` adapter uses Responses, `anthropic` uses Messages, and `compatible` uses an OpenAI-style chat endpoint. All prompt state comes from SQLite, so changing providers does not depend on provider-hosted conversation state. Image attachments are bounded by the configured count and size and require a dialogue profile with `supports_images: true`. Text attachments are supported. Replies stream through character webhooks and are split below Discord's message limit.

The offline suite covers spaces, lore isolation and promotion, branch histories, consent, director choices, card parsing, lorebook sync, and web authorization. The NiceGUI admin console also provides a full lore-rule editor, drag-and-drop transfers, customizable emotion avatars, and versioned prompt presets per server; see [Admin console](docs/admin-console.md). Before using ambient mode in a real server, validate webhook permissions and identities, reply routing in channels and threads, and restart recovery in a private test channel. Those live Discord checks require your bot token and server and are not part of the offline suite.

## Private admin dashboard on a Raspberry Pi 5

Use a 64-bit Raspberry Pi OS. Install Docker Engine and the Compose plugin following the [Debian arm64 instructions](https://docs.docker.com/engine/install/debian/). Install Tailscale on the **Pi host** and join the Pi and your admin devices to the same tailnet. Enable [MagicDNS and HTTPS certificates](https://tailscale.com/docs/how-to/set-up-https-certificates) in the tailnet.

1. Copy `config-example.yaml` to `config.yaml` and `.env.example` to `.env`. Enter your Discord bot token, OAuth client ID and secret, model credentials, and the Pi's exact Tailscale HTTPS name in `WEB_BASE_URL`, such as `https://my-pi.my-tailnet.ts.net`. Keep `.env` private. Set `LLMCORD_DATABASE_PATH` for all services; it overrides the bot's YAML database path.
2. In the Discord developer portal, register the exact OAuth redirect `https://my-pi.my-tailnet.ts.net/auth/callback` for the same Discord application as the bot. Replace the example name with your Pi's actual HTTPS name. The dashboard requests `identify` and `guilds`; each page and change checks the user's server administrator permission with Discord.
3. Run `docker compose up --build -d`. The migration service runs before the bot and web services and creates a dated SQLite backup when upgrading an existing database. Both services share `./data`. The dashboard is published only on the Pi's `127.0.0.1:8080`.
4. On the Pi host, run `sudo tailscale serve --bg 8080`. Check `tailscale serve status`, then open `WEB_BASE_URL` from a tailnet device. [Tailscale Serve](https://tailscale.com/docs/reference/tailscale-cli/serve) provides the private HTTPS proxy; no public router port forwarding is needed.

In the dashboard, create worlds and a hub, link worlds to the hub, bind Discord text channels, import V2/V3 JSON or PNG character cards, choose default casts, and edit characters or channel lore. Card import has a preview; reusing a name requires explicit replacement. Characters always have a home world. The dashboard can also toggle ambient mode per channel; first check explicit conversations in a private Discord test channel.

Create a named **guild lorebook** and enable it for selected worlds or hubs, or create a **channel lorebook** for one bound channel and its threads. Import a JSON entry array, SillyTavern `entries` object/array, or RisuAI version-1 lorebook export. The preview lists adds, updates, removals, and conflicts. Locally edited imported entries require a keep/import choice. Handwritten lore remains separate. A hub guest receives books enabled for its own home world, hub books, and the current channel book; it does not receive another guest's world books. `/context` records activated entries and their source book on each saved response.

The World Info evaluator supports keyword and JavaScript regex matching, constants, secondary filters, ordering, probability, groups, recursion, timing, character filters, and Discord-applicable prompt positions. Rules tied to SillyTavern-only surfaces such as author-note slots, outlets, vectors, or automation IDs are preserved and flagged in the dashboard; they remain inactive until remapped. Behavior that depends on SillyTavern's exact prompt assembly may differ in Discord scenes, so review imported entries in a private test channel.

If Ollama runs on the Pi host rather than in the bot container, use `http://host.docker.internal:11434/v1` as the compatible model `base_url` and configure Ollama to listen on a host-reachable interface. If it runs on another machine, use its reachable LAN or tailnet URL. `localhost` inside the bot container points to the container itself.
