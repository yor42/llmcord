# Running a server

## Conversation participants

Human messages sent to the model include the speaker's Discord display name and stable user ID. Names are saved with each message, so two members sharing a nickname remain distinct and a renamed member keeps the same identity. Mentioned users are identified separately from the person speaking. This is ordinary conversation attribution; world/channel guidelines still supply the roleplay setting.

Recent human chat observed before a bot turn is kept with that branch and included in later replies and summaries, within the configured context and retention limits. Rewinding to an earlier reply excludes later participants and messages. Identity guidance also applies to custom prompt presets; `{{user}}` means the current speaker.

Older saved messages from before this feature retain their user IDs; historical display names that were never recorded cannot be recovered. Speaker names and mention metadata expire with conversation history. Personal memories remain scoped to the consenting user ID and character.

## Spaces and casts

| Scope | What it controls |
| --- | --- |
| World | Home for characters and world lore. A world channel can use only characters from that world. |
| Hub | A crossover space. Administrators choose its allowed worlds; a hub guest keeps its own home-world lore and receives hub lore. |
| Bound channel | Chooses one world or hub and adds channel lore, a default cast, an active cast, and an ambient setting. Multiple channels can share one space. |
| Thread | Inherits its parent channel's space, channel lore, and ambient setting. It has its own cast and conversation history. |

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

Commands marked **Admin** require Discord server administrator permission. Other commands are available to members in their bound channel or thread, unless noted.

| Command | Who | Effect |
| --- | --- | --- |
| `/space create` | Admin | Create a named world or hub. |
| `/space list` | Member | List spaces. |
| `/space bind` | Admin | Bind a text channel to a space; rebinding resets its casts and ambient setting. |
| `/space allow_world`, `/space disallow_world` | Admin | Add or remove a world from a hub's guest list. |
| `/character import` | Admin | Import a V2/V3 JSON or PNG card into a home world. |
| `/character list`, `/character info` | Member | List eligible characters or inspect one character's home world. |
| `/cast set`, `/cast add`, `/cast remove`, `/cast show` | Member | Manage or view the active cast for the channel or thread. |
| `/cast default` | Admin | Set a channel's default and current cast. |
| `/summon` | Member | Invite one eligible character with a prompt, without changing the cast. |
| `/ambient on`, `/ambient off` | Admin | Toggle ambient participation in the current bound channel. |
| `/ambient status` | Member | Show the ambient setting. |
| `/lore add`, `/lore edit`, `/lore pin`, `/lore promote`, `/lore delete` | Admin | Manage shared lore and explicit scope promotion. |
| `/lore list` | Member | Show lore in the current space and local channel or thread. |
| `/memory opt_in`, `/memory opt_out`, `/memory list`, `/memory forget` | Member | Control that member's own personal memories. |
| `/scene reset` | Member | Start fresh context on the next invitation. |
| `/scene delete` | Admin | Delete a stored scene and its branches by root message ID. |
| `/context` | Member | Inspect the last saved character line, or supply a message ID. |

Discord presents the parameters for each slash command. Most command responses are ephemeral, including context and memory listings. Character skit lines appear in the channel.

## Admin dashboard

The dashboard is designed for the Pi's Tailscale HTTPS address. A visitor signs in with Discord. Each server page and change checks that their Discord account is an administrator of the selected server; forms also require a CSRF token. See [deployment](deployment-raspberry-pi.md) for login setup.

From a server page, an admin can create spaces, link worlds to hubs, bind channels, choose default casts, toggle ambient mode, add/edit/pin/promote/delete shared lore, manage characters, and manage named lorebooks. Changes are read from SQLite on the bot's next turn; there is no separate publish step.

For a card, select its home world and upload a V2/V3 `.json` or `.png` file. The preview shows its name, description, personality, scenario, opening line, lore-entry count, and PNG avatar when present. Saving an existing name requires a replacement confirmation. Reimports preview local conflicts and preserve manual additions and avatars unless explicitly replaced. Editing the home world can remove the character from casts where it is no longer eligible. Archived characters are not eligible for scenes.

The dashboard edits the main text fields of an imported card. Use the Lore workspace to edit its embedded character entries. For named guild and channel lorebooks, use the [import workflow](lore-and-memory.md#importing-sillytavern-lorebooks).

See [Admin console](admin-console.md) for the NiceGUI workspace, rule transfers, emotion avatars, and per-server prompt presets.
