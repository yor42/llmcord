llmcord is a Discord character bot for casual group skits. Characters speak through webhooks with their own names and avatars, a director model picks who answers, and every scene is kept in SQLite so replies can branch from any earlier line. A private admin dashboard (Discord sign-in, served over Tailscale) manages worlds, characters, lore, prompts and games. It is built to self-host on a Raspberry Pi 5.

Administrators arrange characters and lore into **worlds**, **hubs** and **channels**. Members change a scene's cast, summon characters, keep favorites, play card games with the characters, and control their own personal memory.

## Where to start

| You want to… | Read |
| --- | --- |
| Run the bot for the first time | [Installation](Installation) → [First server setup](First-Server-Setup) → [Configuration](Configuration) |
| Host it on a Raspberry Pi | [Docker on a Raspberry Pi](Docker-on-Raspberry-Pi) or [Native Pi hosting](Native-Pi-Hosting); [Releases and CI](Releases-and-CI) for published images |
| Fix something that doesn't work | [Troubleshooting](Troubleshooting) |
| Learn the commands | [Discord commands](Discord-Commands) |
| Understand worlds, hubs and casts | [Worlds, hubs and casts](Worlds-Hubs-and-Casts), [Scenes and ambient mode](Scenes-and-Ambient-Mode), [Favorites](Favorites) |
| Know what a character remembers | [Lore and memory](Lore-and-Memory) |
| Play games and spend currency | [Blackjack](Blackjack), [I Doubt It](I-Doubt-It), [Texas Hold'em](Texas-Holdem), [One Card](One-Card), [Yut](Yut), [Items and the shop](Items-and-Shop) |
| Run a server from the dashboard | [Dashboard overview](Dashboard-Overview), then the dashboard pages in the sidebar |
| Operate the bot (caps, model profiles) | [Bot settings](Bot-Settings), [Monitoring and costs](Monitoring-and-Costs) |
| Work on the code | [Architecture](Architecture), [Testing and verification](Testing-and-Verification), [Engineering docs](Engineering-Docs) |

The words used in commands, the dashboard and these pages are defined in the [Glossary](Glossary).

## Feature overview

- **Worlds and hubs.** A world is a setting with its own characters and lore. A hub links several worlds so their characters can meet without sharing their private world lore.
- **Scenes that branch.** Reply to any earlier character line and the conversation continues from there; later lines on the other branch are not seen.
- **Ambient mode.** Optionally, characters join a channel's chat on their own when someone names them or asks the room (English and Korean).
- **Lore and memory.** Hand-written lore, SillyTavern and RisuAI lorebook imports with conflict review, World Info rules, scene summaries, and personal facts kept only after `/memory opt_in`.
- **Prompt presets.** Per-server, versioned prompt structures, with SillyTavern Chat Completion preset import.
- **Emotion avatars.** A character can change its avatar by mood for each line.
- **Currency, items and games.** Daily check-ins, a toy shop, and blackjack, I Doubt It, Texas Hold'em, One Card and Yut tables where characters play and talk.
- **Costs under control.** Per-server usage reports, an optional turn log, and bot-wide spending caps.

## Source and license

The code lives at [yor42/llmcord](https://github.com/yor42/llmcord). It began as a fork of [llmcord](https://github.com/jakobdylanc/llmcord) by jakobdylanc and is MIT licensed; see the repository README for the full notice.
