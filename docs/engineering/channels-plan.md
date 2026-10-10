# Channel kinds, favorites and character wallets: plan (proposed 2026-10-10)

Status: **decided (D27); FEAT-23 in progress.** Comes before character seats (FEAT-20) because those seats use favorites and character wallets.

## Channel kinds

A channel's kind follows from its binding. There is no separate setting, so the kind can never disagree with the binding.

| Kind | Binding | Who speaks | Lore | Games |
| --- | --- | --- | --- | --- |
| **Members only** | Not bound | Members only. The bot does no roleplay here (as today). | None | Allowed if the channel is a game channel; members only, never character seats. |
| **World channel** | Bound to a world | That world's characters | The world's lore, plus server, channel and lorebook lore | Allowed if it is a game channel. Characters from this world may take seats. |
| **Hub** (the "lobby") | Bound to a hub | Characters from the linked worlds | Hub lore, plus each guest's own world lore (isolation unchanged) | Allowed if it is a game channel. Characters from the linked worlds may take seats. |

Game channels keep the D26 rule: an admin turns games on per channel, in any number of channels. Only the binding decides whether characters can sit at a table.

User docs explain hubs as "the lobby"; commands and the glossary keep the word "hub".

## Hub tone (per hub)

Each hub has a **tone** setting:

- **In character** (default, today's behaviour).
- **Off duty**: characters keep their personality but know they are in a shared lounge with characters from other worlds. They chat casually and don't push their world's plot.

The tone changes only the prompt instruction for scenes in that hub's channels. World channels are always in character. The setting is edited from the dashboard (Worlds and hubs) and `/admin space`, with a revision check like other admin writes.

## Favorites (per member)

- Each member keeps an ordered list of favorite characters for each server, up to the server's **favorites limit**.
- Commands are `/favorites add|remove|list|mode`, plus the member's own page later if one exists. Replies are ephemeral, so only the owner sees the list.
- A favorite has an effect only in channels where that character can already appear (the world or hub eligibility rules). A favorite from world A never appears in world B's channels.
- The list is a preference, not a personal fact, so it needs no `/memory opt_in`. It is deleted on request and when the character is deleted.
- **Mode (per member):** how far the member's favorites may go when that member posts.
  - **Lean** (default): favorites in the channel's cast are more likely to be picked. The director is told who the member's favorites are, and they get a better chance at ambient replies.
  - **Step in**: also allows a favorite who is eligible in the channel but not in its cast to be picked for this member's message. This does not change the channel's saved cast.
- **Fairness:** the director still answers everyone. When members with different favorites post in the same channel, each message weighs only its own author's favorites, and the ambient cooldown still applies.
- **Games:** when joining a table a member can bring their favorites (e.g. `/blackjack bet:10 favorites:true`). Favorites who can appear in that channel and can afford a bet fill empty seats.

## Character wallets

- Each character has its own balance per server, separate from member balances. It moves only through the same append-only ledger: a holder kind of member or character on each entry.
- **Refill:** a character gets the server's daily check-in **amount** (no streak bonus) once per server day, up to the server's **character refill cap**. Refills are applied lazily the first time the character needs money that day, and missed days don't add up.
- **Winnings:** may take a balance above the refill cap. The cap only limits refills. The usual 10^12 balance cap still applies.
- **Daily check-in off:** if the server's daily amount is 0, characters don't refill. The dashboard says so next to the refill cap.
- **Betting:** a character bets the same amount as the member who brought it, lowered to its balance and the table's largest bet. It doesn't sit if it can't cover the smallest bet.
- **Visibility:** admins see character balances and ledger entries in the Currency tab; members see a character's balance with `/balance character:<name>`.

## Limits (per server, user 2026-10-10)

Two server settings in the dashboard, each a whole number from 1 to 15 (default 5), each with a note on cost:

- **Largest cast** (replaces the fixed limit of 5 characters per cast). Note: "A larger cast gives the director more characters to choose from, so more characters may answer a message. Each answer is a separate model call, and every prompt lists the cast."
- **Favorites per member.** Note: "With "step in", a member's favorites outside the cast can answer too, so more favorites can mean more model calls."

Lowering a limit keeps existing casts and favorite lists. They can shrink but not grow until they are within the limit.

## Schema

**v15** (FEAT-23), with a backup before the upgrade like v13 and v14:

- `member_favorites(guild_id, user_id, character_id, position)`
- the member's favorites mode
- `guild_settings` max cast and max favorites
- `spaces.hub_tone` (used by FEAT-24)

**v16** (FEAT-25): character balances, refill day, ledger holder kind and the refill cap. These get their own migration and review, since they touch the money tables.

## Items and order

| Item | Content | Review |
| --- | --- | --- |
| FEAT-23 | Schema v15 + favorites: store, `/favorites`, director hint and ambient weighting, the "step in" mode, cast and favorites limits | Opus (migration, isolation) |
| FEAT-24 | Hub tone: setting, prompt instruction, dashboard and `/admin space` | Sonnet |
| FEAT-25 | Schema v16 + character wallets: balances, lazy refill, refill cap setting, Currency tab view, `/balance character:` | Opus (money) |
| FEAT-20 | Character seats in blackjack: "bring favorites" at join, stakes from the wallet, the model picks a legal move, an in-character line through the webhook | Opus (money, concurrency) |
