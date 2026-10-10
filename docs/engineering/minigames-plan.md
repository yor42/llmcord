# Minigames and items: plan (proposed 2026-10-10)

Status: **phase 1 done 2026-10-10 (FEAT-19, D26)**; later phases are a plan. Character seats (FEAT-20) now build on favorites and character wallets: see [channels-plan.md](channels-plan.md) (D27). Builds on FEAT-17 (currency) and FEAT-18 (daily check-in).

## Principle

The game engine decides everything that has a result: the deck order, legal moves, the dealer's play, who wins, and payouts. A model never decides an outcome. When a character holds a seat, the model gets only what that seat may see plus the list of legal moves, and picks one move id from the list (and optionally a short in-character line). Anything else it returns (an illegal move, a timeout, a provider error, a spending cap) falls back to a fixed default policy, so a game never stalls on a model.

## Layers

1. **Rules engine** (`llmcord_core/games/`, pure Python, no Discord, no database, no model).
   Each game is a set of pure functions over a state: `new(seed, seats, rules)`, `view(state, seat)` (what that seat may see), `legal_moves(state, seat)`, `apply(state, seat, move)`, `result(state)` (finished? payouts per seat).
   Randomness comes only from a seed: the deck is a seeded shuffle, so a game replays exactly from its seed plus its move list. Tests can pin every hand.
2. **Tables** (store, schema v14). `game_tables` (guild, channel, game, rules snapshot, seed, status, created_at), `game_seats` (human member or character, stake), `game_moves` (append-only, like the currency ledger). The live state is rebuilt by replaying moves from the seed, so a bot restart resumes or refunds a table instead of losing it. Every write is guild-scoped and goes through `write_admin`.
3. **Money.** Joining takes the stake with a ledger entry (`source='game'`, reason names the table); settling pays out with another entry in the same transaction that records the result. A table cancelled or timed out refunds stakes. The house is the server itself: winnings are created and losses removed, as with grants and revokes. Bet limits are per-server settings (1 ≤ min ≤ max ≤ 100,000). A payout or refund that would take a balance above the 10^12 cap is paid up to the cap, and the seat records the amount actually paid. Turning games off in a channel closes its open table and refunds the unfinished round.
4. **Discord surface.** `/blackjack [bet]` opens a table in the channel. Moves are buttons (`discord.ui.View`, new to this codebase). Only the seat's own member may press them, and each press is checked against `legal_moves`. A turn timer applies a default move (for example "stand"). A table runs under its own per-table lock, never the channel's skit lock, so a game does not block roleplay or the reverse.
5. **Character seats** (later phase). On a character's turn the bot calls the model with: the character card (personality), the seat's `view`, the legal moves, and a structured-output schema (`engine.structured_contract`) that allows only those move ids. The move goes through the same `apply` as a human's. The optional line is posted through the character's webhook, like skit replies. Model calls go through `BackendResolver` (key pins) and count toward spending caps; a server can set a cheap profile for game moves.

## Phases

| Phase | Item | Content |
| --- | --- | --- |
| 1 | FEAT-19 | Game framework (engine interface, seeded deck, tables/seats/moves schema v14, stake and payout through the ledger, restart recovery, timeouts) plus **blackjack, humans only**: one table per channel, 1 to N players against the engine dealer. |
| 2 | FEAT-20 | Character seats in blackjack: the "pick a legal move" model call, the fallback policy, an in-character line, a per-server switch and model profile. Blackjack is a good test bed because each seat plays only against the dealer, so a weak model choice harms only its own seat. |
| 3 | FEAT-21 | A game that needs 3+ players, where characters fill empty seats (candidates: a simple poker variant, liar's dice, or a trick-taking game). Needs hidden information per seat, which `view()` already models. |
| later | FEAT-22 | Items: catalogue, inventory, buying with currency. Waiting for the user's item ideas. |

## Blackjack rules proposed for phase 1

Six-deck shoe shuffled per round from the seed; dealer stands on soft 17; blackjack pays 3:2 (rounded down); hit, stand, double down on the first two cards; no split in phase 1 (split can follow); insurance, even money, surrender and the other rule options came with FEAT-28 (D28). Push returns the stake. A round settles when every seat has stood, busted, or timed out.

## Decisions (D26, user 2026-10-10)

- **Open table.** `/blackjack` opens a table in the channel. Others join with their own bet during a short join window before the deal. After each round, seated players deal again or leave. The table closes when everyone has left or after an idle timeout.
- **Admin-marked channels.** Games open only in channels an admin turned games on for (dashboard and `/admin`).
- **Groundwork first.** Phase 1 builds no character play. It lays the groundwork for character seats: seats have a kind (`member` or `character`), each seat gets a `view()` of only what it may see, and moves come from a chooser interface. Phase 1 implements only the member chooser (buttons) and a deterministic default policy (used on timeouts). How characters stake is decided in phase 2.
- **Schema v14** for tables, seats and an append-only move log.
- **Seed hash.** Each round shows a short SHA-256 hash of its seed when it deals and reveals the seed when it settles, so anyone can check the deck was fixed before the first card.

## Phase 1 parts

| Part | Content | Review |
| --- | --- | --- |
| A | `llmcord_core/games/`: seeded shoe, blackjack rules engine (`new`, `view`, `legal_moves`, `apply`, `result`, replay from seed + moves), seat kinds, chooser interface with the default policy. Pure Python, no Discord, database or model. | Sonnet |
| B | Schema v14 and store API: game channels, bet limits, tables, seats, rounds, moves; stake taken at join and paid at settle through the ledger in one transaction; refunds on cancel; finding unfinished rounds after a restart. | Opus (migration, money, concurrency) |
| C | Discord: `/blackjack bet`, buttons, join window, turn timer (default policy on timeout), deal again / leave, per-table lock, restart recovery (refund unfinished rounds), `/admin games channel` and bet limits. | Opus (concurrency) |
| D | Dashboard: game channel switches and bet limits. | Sonnet |

## Games tab and blackjack rule options (D28, user 2026-10-10)

**Games tab.** The dashboard gets a Games tab with one section per game. Blackjack is the first. The Currency tab keeps money only: the name, daily check-in, character wallets, balances and the ledger. The Games tab has:

- **Shared settings:** game channels, bet limits, character table talk, the daily table-talk limit, and how many rounds the end-of-table summary shows (default 5).
- **Blackjack:**
  - The game on or off for the whole server. Turning it off closes open tables with refunds, as turning a channel off does today.
  - The rule options below.

**Rule options.** Each round stores the rules it was dealt with, in `game_rounds.rules_json`. A rules change applies from the next round, and replays of older rounds stay correct. An empty rules value means the classic rules, so existing rounds replay unchanged.

| Option | Choices | Default (today's rules) |
| --- | --- | --- |
| Dealer stands on | total 16, 17 or 18 | 17 |
| Dealer on soft 17 | stand or hit (H17); only when the stand total is 17 | stand |
| Blackjack pays | 3:2 or 6:5 | 3:2 |
| Ties | push (bet back) or the dealer wins | push |
| Insurance and even money | off or on. When the dealer shows an ace, each seat in turn may insure for half its bet, which pays 2:1 if the dealer has blackjack. A seat with blackjack is offered even money (1:1 now) instead. A timeout or a character with table talk off declines. | off |
| Late surrender | off or on. On the first two cards, after the dealer checks for blackjack, give up the hand and get half the bet back. | off |

- **Rounding:** halves and 6:5 payouts round down, so the house keeps any odd unit. Insurance needs a bet of at least 2.
- **Rules line:** the table message shows the round's rules as a gray `-# ` line, for example "Dealer stands on 17 · Blackjack pays 3:2 · Ties push · Double on the first two cards · No splitting · 6 decks".
- **Character seats:** the model's legal moves include insurance and surrender when they are allowed. The house-rule fallback never insures and never surrenders.

**Items:**

- UI-53 to UI-56: table polish (end-of-table summary, dealer pacing, how-to-play button, rules line). These come first.
- FEAT-27: the Games tab and the server switch.
- FEAT-28: the rule options. Schema columns go into v16 while it is unreleased.

## I Doubt It (FEAT-21, D29, user 2026-10-11)

Phase 3's first game, and the start of a library of multiplayer games built around table talk. The terms (**I Doubt It**, **Claim**, **Doubt**, **Pile**, **Doubt window**, **Table thread**, **Ante / Pot**) are in the [glossary](../glossary.md).

**Rules (v1).**

- 3–8 seats. One deck for 3–5 seats, two decks for 6–8. The whole deck is dealt.
- On your turn you play 1–4 cards face down and claim that many of the **forced rank**. The rank follows the previous claim: A, 2 … K, A …. You can't pass, and you bluff if you hold none of the rank.
- After each play there is a **doubt window** of 15 s. Any other seated player may doubt, and the first press wins. The window closes on a doubt, when the next player plays (which counts as passing on the doubt), or on timeout. The next player's turn timer starts when the window closes.
- On a doubt, the played cards are revealed to everyone. Whoever loses the doubt takes the pile.
- A player who empties their hand wins once the window on that last play closes. If the play is doubted and was truthful, they win. If it was a bluff, they take the pile and play on.
- **Rule key:** `claim_rule: sequence` is stored and normalized from day one. `free` (claim any rank) is the end goal and comes as a later rule-options item. Only `sequence` is accepted in v1.
- **Ante:** optional, off (0) by default. Each seat pays it at the deal, from a member's or a character's own wallet. The winner takes the pot. Losing a doubt costs no currency.
- **Timeouts:** a member's turn timeout (default 60 s, a Games tab setting) plays the lowest card truthfully if they can, or else a random card as a bluff. Two timeouts in a row forfeit the seat, and that seat's cards leave play.

**Seats.**

- **Bound game channels** (world or hub, D27) seat members and characters together. A member brings their favorites (`favorites:True`), marked "(with @member)". The host can press **Fill seats** in the lobby to add characters from the channel's active cast, marked "(house)", up to the minimum or a chosen count. Character seats ante from their own wallet.
- **Unbound game channels** are **humans-only tables**, with at least 3 members and no character seats.
- Joining is lobby-only. Nobody joins mid-game.

**Who decides what.** D26 holds. The engine owns everything random or resulting: the shuffle, the deal, reveals, who takes the pile, the winner and the pot. A character's model picks one legal move for its seat (which cards to play, whether to doubt) from that seat's view only: its own hand, the hand counts, the pile size, the claim history and the revealed cards. The fallback is a deterministic policy with per-character bluff and suspicion tendencies seeded from the character id. It plays whenever the model is unavailable (error, timeout, daily talk limit, spending cap).

**Doubt triage.** Asking every character "doubt?" after every play would be up to 7 model calls per play. Instead, the policy scores each character seat's suspicion from public information plus its own hand. Only the top 1–2 most suspicious seats get a model call, and the others pass. An impossible claim (more of a rank than that seat can't see) is always asked. These calls run in parallel inside the window, and the first doubt wins.

**Table thread.**

- `/game start` posts the lobby in the game channel and opens a public thread on it.
- **Board message:** pinned at the top of the thread and edited on game events, debounced. It shows hand counts, turn order, the pile size and the current rank.
- **Turn message:** a single message edited in place. It shows the last claim, the pile, who's next, and the **Doubt** and **My hand** buttons. It is reposted at the bottom only when buried (8 or more thread messages since it was posted), at most once per turn, so a quiet table never reposts.
- **My hand** (or `/game hand`) opens a private picker that groups the hand by rank: choose how many of each rank to play. A hand has at most 13 distinct ranks, so one select fits.
- At the end, the result is posted in the thread and the lobby is edited to a one-line result. The thread stays open 10 minutes for banter, then is archived and locked.
- New bot permissions: **Create Public Threads** and **Send Messages in Threads**.

**Table talk.**

- The table thread is a normal skit thread. It inherits the parent's binding, cast, lore and preset, and has its own scene and lock.
- Only seated characters speak. Their prompt gets a table block with public state plus that speaker's own hand.
- Human messages, from players and spectators, trigger the usual director turn. Game events (a doubt, a reveal, a near-win) also trigger an ambient character turn.
- Talk counts against the daily table-talk limit and the spending cap.
- Game moves never wait for talk.
- Talk needs a bound parent and character table talk turned on. Without them, characters still play, silently.

**Commands.**

- `/game start game:<blackjack|doubt> [ante|bet] [favorites]`
- `/game hand` works in any table thread.
- `/game rules game:<name>`
- `/blackjack` stays as an alias of `/game start game:blackjack`.
- An option that doesn't apply to the chosen game gets a short private reply.

**Game registry.** Each game registers its engine module, Discord adapter, Games tab section and settings. `/game`, the Games tab and button prefixes (`llmcord:<game>:…`) come from the registry. Blackjack moves onto it, keeping its existing `llmcord:bj` custom ids so live buttons keep working.

**Schema v17** (backup before upgrade):

- New `game_settings(guild_id, game, enabled, rules_json, updated_at)`, without a revision column: writes keep today's expected-value checks (`ConflictError`). Blackjack's `blackjack_enabled` and `blackjack_rules` are copied into its row, and every read and write moves to the new table. The old columns stay, unused and commented, because dropping them would be destructive.
- `game_tables` gains nullable `thread_id`, `board_message_id` and `turn_message_id`, and `open_table` accepts `doubt`.
- The ante goes in `game_seats.stake`, and the pot is the sum of stakes. An ante of 0 needs a stake of 0, which v16's `CHECK(stake > 0)` refuses and SQLite cannot alter, so v17 rebuilds `game_seats` with `CHECK(stake >= 0)` and keeps every row. Blackjack still rejects a stake of 0 in its engine and store.
- Hands are never stored. They replay from seed plus moves.
- Ledger reasons: "I Doubt It ante / payout / refund (table T, round N)".

**Games tab:** an I Doubt It section with on/off, ante (0 = off), turn timeout, and the claim rule shown read-only as "Sequence". The doubt window (15 s), the bump threshold (8) and the deck count are fixed. The shared settings apply as they are.

| Part | Content | Review |
| --- | --- | --- |
| A | Engine (`games/doubt.py`: `new`, `view`, `legal_moves`, `apply`, `result`, replay, rule normalizer), the seeded fallback policy and doubt triage scoring, the game registry, schema v17 (`game_settings` with blackjack moved over, `game_tables` thread columns), store API, ledger. | Opus (migration, money) |
| B | Discord: `/game` (with `/blackjack` alias), lobby, Fill seats, table thread, board and turn messages with bump, rank-count hand picker, doubt window, timers, forfeits, model chooser with triage, restart recovery. | Opus (concurrency) |
| C | Table talk: table threads as skit threads, the seated-only cast, the table prompt block (seat view only), event-triggered ambient turns, talk limits. | Opus (skit turn path, hidden information) |
| D | Games tab section through the registry, user docs (how to play, permissions), glossary code names. | Sonnet |
