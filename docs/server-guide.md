# Running a server

## Conversation participants

Human messages sent to the model include the speaker's Discord display name and stable user ID. Names are saved with each message, so two members sharing a nickname remain distinct and a renamed member keeps the same identity. Mentioned users are identified separately from the person speaking. This is ordinary conversation attribution; world/channel guidelines still supply the roleplay setting.

Recent human chat observed before a bot turn is kept with that branch and included in later replies and summaries, within the configured context and retention limits. Rewinding to an earlier reply excludes later participants and messages. Identity guidance also applies to custom prompt presets; `{{user}}` means the current speaker.

Older saved messages from before this feature retain their user IDs; historical display names that were never recorded cannot be recovered. Speaker names and mention metadata expire with conversation history. Personal facts remain scoped to the consenting user ID and character.

## Worlds, hubs and casts

| Scope | What it controls |
| --- | --- |
| World | Home for characters and world lore. A world channel can use only characters from that world. |
| Hub | A crossover setting. Administrators choose its allowed worlds; a hub guest keeps its own home-world lore and receives hub lore. |
| Bound channel | Chooses one world or hub and adds channel lore, a default cast, an active cast, and an ambient setting. Multiple channels can share one world or hub. |
| Thread | Inherits its parent channel's world or hub, channel lore, and ambient setting. It has its own cast and conversation history. |

A default cast is an administrator's starting selection for a channel. Members can change the active cast without editing that default. Setting a new default also resets the channel's active cast to it. A thread uses the parent channel's active cast until someone sets a thread cast. A cast can contain up to five eligible characters; the director selects at most three to speak in one turn.

`/summon` invites an eligible character for one turn. In a hub, this allows a linked-world guest outside the current cast to join an explicit skit. Summoning does not change the cast. A hub character outside the active cast will not join ambiently.

## Starting and continuing scenes

- Mention the bot in a bound channel, reply to a saved character webhook message, or run `/summon` with a prompt. These are explicit invitations.
- Replying to an older character message follows that message's saved parent chain. Later messages on the other branch are excluded. The bot does not need Discord's visually nested replies for its character lines; it stores the parent message IDs internally.
- A new invitation may include up to 12 recent human messages from the previous 10 minutes. A current-branch reply can also use intervening group messages. `/scene reset` prevents the next invitation from using earlier nearby context.
- The director chooses one or more speakers from the active cast. Each chosen character receives a separate sequential model call, so later speakers can react to lines already posted in the turn.
- Character lines use webhooks with the character name and PNG avatar when present. If the bot cannot manage webhooks or a channel has reached its webhook limit, it reports the failure in the channel.

Ambient mode is off until an administrator enables it in a channel. It waits for two new human messages and a 120-second channel cooldown by default. The director may remain silent; the code also applies a conservative invitation check before calling the director. Check explicit turns before enabling ambient mode.

Text attachments are read up to the configured size and added to the turn. Image attachments are bounded by count and size and need an image-capable dialogue model. Other attachment types are rejected. Long character lines are split into Discord-safe message chunks.

## Discord commands

Commands marked **Admin** live under `/admin` and require Discord server administrator permission. Discord hides `/admin` from members without that permission by default (a server can change this under Integrations); the bot still checks the permission on every use. Other commands are available to members in their bound channel or thread, unless noted. All commands work in servers only, not in direct messages. If you upgraded from a version where admin commands were top-level (for example `/lore add`), re-check any per-command overrides under Server Settings → Integrations: they do not carry over to the new `/admin …` commands.

| Command | Who | Effect |
| --- | --- | --- |
| `/admin space create` | Admin | Create a named world or hub. |
| `/space list` | Member | List the server's worlds and hubs. |
| `/admin space bind` | Admin | Bind a text channel to a world or hub; rebinding keeps its ambient setting and drops cast members not available in the new world or hub. |
| `/admin space link_world`, `/admin space unlink_world` | Admin | Link a world to a hub so its characters can appear there, or unlink it; unlinking also drops that world's characters from the hub's channel casts. |
| `/admin character import` | Admin | Import a V2/V3 JSON or PNG card into a home world. |
| `/character list`, `/character info` | Member | List eligible characters or inspect one character's home world. |
| `/cast set`, `/cast add`, `/cast remove`, `/cast show` | Member | Manage or view the active cast for the channel or thread. |
| `/admin cast default` | Admin | Set a channel's default and current cast. |
| `/summon` | Member | Invite one eligible character with a prompt, without changing the cast. |
| `/admin ambient on`, `/admin ambient off` | Admin | Toggle ambient participation in the current bound channel. |
| `/ambient status` | Member | Show the ambient setting. |
| `/admin lore add`, `/admin lore edit`, `/admin lore pin`, `/admin lore promote`, `/admin lore delete` | Admin | Manage shared lore. `add` takes `scope:channel` (this channel, or this thread inside a thread) or `scope:space` (the bound world or hub); `promote` copies an entry to another owner. |
| `/lore list` | Member | Show the lore of this channel or thread and its world or hub. |
| `/catchup [focus] [hours]` | Member | A private summary of what you missed in this channel or thread, told from your point of view: by default since your last message here (at most 24 hours and 200 messages), or the last `hours` hours (1–72). `focus` steers it, for example "What did I miss that directly concerns me?". It reads the messages members can see, character lines included (you need Read Message History in that channel), and uses your personal memories only if you ran `/memory opt_in`. Once per member per channel every 5 minutes. Works in character channels; in other channels only when a server admin turns on **Allow /catchup in channels without characters**. Paused like replies when the bot's spending cap is reached. |
| `/memory opt_in`, `/memory opt_out`, `/memory list`, `/memory forget` | Member | Control that member's own personal memories. |
| `/time set`, `/time show`, `/time clear` | Member | Set, view, or remove the member's own timezone for this server (works in any channel). Characters use it for the local time in their prompts; without one they use the server default (set in the dashboard under Server setup → **Server settings** → **Server timezone**), else UTC. `set` suggests IANA names such as `Asia/Seoul` as you type. The timezone is a setting, not a personal memory: it needs no `/memory opt_in`, and `/memory opt_out` does not remove it. |
| `/balance [member]` | Member | Show your balance in the server currency, or another member's. Works in any channel; the reply is private. |
| `/daily` | Member | Claim the daily check-in bonus (once per server day, set by an admin; the day follows the server timezone, UTC if none). Checking in on consecutive days earns an extra bonus; a missed day starts over at day 1. The reply is private and shows your balance and when the next check-in opens. Off until an admin sets an amount. |
| `/admin currency grant`, `/admin currency revoke`, `/admin currency name` | Admin | Give a member currency or take it away (1–1,000,000 at a time, with a reason of up to 200 characters), or set what this server calls its currency (default "coins", up to 32 characters on one line, without `@`, `<` or markdown symbols). A balance never goes below zero: a revoke larger than the balance is refused. Every change is kept in the server's currency ledger with its amount, reason, admin and time; ledger entries are never edited or deleted. Balances stay when a member leaves the server. Members earn currency with `/daily` and bet it in [blackjack](#blackjack). |
| `/admin currency daily amount [streak_bonus] [streak_days]` | Admin | Turn on the daily check-in: `amount` (0 to 1,000,000; 0 turns it off) is paid for each check-in, plus `streak_bonus` for every consecutive day after the first, for up to `streak_days` days (0 to 365, default 7). Options you leave out keep their current value. Each payout is a ledger entry with source "daily". |
| `/blackjack bet` | Member | Join or open a blackjack table in this channel with a bet (see [Blackjack](#blackjack)). Only in channels an admin turned games on for. The table message is public and updates in place; refusals are private. |
| `/admin games channel enabled [channel]` | Admin | Turn games on or off in a channel (default: this one). Turning them off closes that channel's open table and refunds the bets of an unfinished round. |
| `/admin games bets min max` | Admin | Set the smallest and largest bet (1 to 100,000, smallest at most largest; default 1 to 1,000). |
| `/scene reset` | Member | Start fresh context on the next invitation. |
| `/admin scene delete` | Admin | Delete a stored message and everything after it in its branch, by message ID. Earlier messages and other branches stay; Discord messages are not deleted. Personal facts and character encounters first learned from the deleted messages are forgotten too; lore already promoted from them stays. |
| `/context` | Admin | Inspect the last saved character line, or supply a message ID. |

Character and space options (`character`, `characters`, `space`, `hub`, `world`) suggest matching names as you type. A name matches exactly first, then ignoring case; if several names differ only by case, type the exact one. In `characters`, separate names with commas; suggestions complete the last name.

Discord presents the parameters for each slash command. Most command responses are ephemeral, including context and memory listings. Character skit lines appear in the channel.

## Blackjack

Blackjack is the first game that uses the server currency (see `/balance` and `/daily`). Players bet against the bot's dealer; no model is involved, so a result never depends on a character or a provider.

- **Turn it on.** An admin runs `/admin games channel enabled:True` in the channel. Bet limits come from `/admin games bets`. The same settings are in the **Games** section of the dashboard's Currency tab.
- **Play.** `/blackjack bet` opens a table in the channel, or joins the open one. Others join with their own bet for 30 seconds, or anyone seated presses **Deal now**; **Leave** refunds a bet before the deal. When the cards are dealt, each player in turn presses **Hit**, **Stand** or **Double** (only on the first two cards, and only if you can pay the extra stake). Only the player on turn can press; a player who does nothing for 60 seconds is played automatically (hit below 17, else stand), and each automatic or manual move gives the next decision another 60 seconds. After a round, **Play again (same bet)** joins the next round and **Leave table** leaves. The table closes after 2 minutes with nobody joining a new round.
- **Rules.** Six decks reshuffled every round, the dealer stands on soft 17, a natural blackjack pays 3:2 (rounded down), a push returns the bet. No splitting or insurance yet.
- **Money.** The bet leaves your balance when you join and the payout is added when the round ends, each as a ledger entry. If the bot restarts or an admin turns games off in the channel, unfinished rounds are cancelled and every bet is refunded.
- **Check the deck.** Each round shows a short fingerprint of its secret seed ("Seed hash") when the cards are dealt, and the seed itself when the round ends. Run SHA-256 over the revealed seed text and compare the first 12 characters with the hash shown earlier: they match, so the deck was fixed before the first card.

## Admin dashboard

The dashboard is designed for the Pi's Tailscale HTTPS address. A visitor signs in with Discord. Each server page and change checks that their Discord account is an administrator of the selected server; changes are checked again on every action, and uploads and sign-out also require a CSRF token. See [deployment](deployment-raspberry-pi.md) for login setup.

From a server page, an admin can create worlds and hubs, link worlds to hubs, bind channels, choose default casts, toggle ambient mode, turn the reply footer on or off, choose the server timezone, add/edit/pin/promote/delete shared lore, manage characters, and manage named lorebooks. Changes are read from SQLite on the bot's next turn; there is no separate publish step.

For a card, select its home world and upload a V2/V3 `.json` or `.png` file. The preview shows its name, description, personality, scenario, opening line, lore-entry count, and PNG avatar when present. Saving an existing name requires a replacement confirmation. Reimports preview local conflicts and preserve manual additions and avatars unless explicitly replaced. Editing the home world can remove the character from casts where it is no longer eligible. Archived characters are not eligible for scenes.

The dashboard edits the main text fields of an imported card. Use the Lore workspace to edit its embedded character entries. For server and channel lorebooks, use the [import workflow](lore-and-memory.md#importing-sillytavern-lorebooks).

The bot's operators (who run the bot itself, not per-server admins) also get a **Bot settings** page with bot-wide spending caps. When the hard cap is reached, characters stop replying until the reset day; `/summon` tells the member privately, and other messages get no reply unless the operator turned on the channel notice. See [Bot settings and spending caps](admin-console.md#bot-settings-and-spending-caps).

See [Admin console](admin-console.md) for the NiceGUI workspace, rule transfers, emotion avatars, and per-server prompt presets.
