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
| later | FEAT-22 | Items and the shop: designed in [items-plan.md](items-plan.md) (D30). |

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

**Who decides what.** D26 holds. The engine owns everything random or resulting: the shuffle, the deal, reveals, who takes the pile, the winner and the pot. A character's model picks one legal move for its seat (which cards to play, whether to doubt) from that seat's view only: its own hand, the hand counts, the pile size, the claim history and the revealed cards. The fallback is a deterministic policy with per-character bluff and suspicion tendencies seeded from the character id. Its per-move randomness comes from the secret round seed, never from the seed hash shown on the board, so players cannot predict a character's bluffs or doubts; it stays replayable once the seed is revealed after the game. It plays whenever the model is unavailable (error, timeout, daily talk limit, spending cap).

**Doubt triage.** Asking every character "doubt?" after every play would be up to 7 model calls per play. Instead, the policy scores each character seat's suspicion from public information plus its own hand. Only the top 1–2 most suspicious seats get a model call, and the others pass. A seat that sees an impossible claim (its own copies of the rank plus the claimed count exceed the deck) doubts it at once without a model call. The other calls run in parallel inside the window, and the first doubt wins. Every call, including one cancelled because another seat doubted first, counts toward the daily table-talk limit.

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

- `/game start game:<blackjack|doubt> [bet] [favorites]`. I Doubt It has no per-table ante in v1: the server's ante (Games tab), captured when the table opens, applies, and each seat pays it on joining (refunded on leaving or cancel).
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

## Play style and Texas Hold'em (FEAT-32, FEAT-33, FEAT-34, D31, user 2026-10-11)

Terms (**Texas Hold'em**, **Buy-in**, **Stack**, **Blinds**, **Dealer button**, **Assist**, **Play style**, and the widened **Pot**) are in the [glossary](../glossary.md). Order: FEAT-32 (Play style), then FEAT-33 (Hold'em), then FEAT-34 (Doubt on Play style).

### Play style (FEAT-32, schema v19)

A character's way of playing is a small set of numbers, so the fallback policy can be in character instead of seeded from the character id.

- **Traits, each 0–100, 50 neutral:** Looseness (how weak a hand it plays), Aggression (raise over call, and size), Bluffing, Stubbornness (reluctance to fold after investing), Suspicion (how readily it doubts), Risk (how much of its stack it commits), **Judgment** (foolish ↔ sharp: systematic misreading of its own hand, draws and pot odds) and **Unpredictability** (steady ↔ erratic: chance of an out-of-pattern move). A game uses the traits it needs.
- **Judgment is systematic, Unpredictability is random.** A foolish character makes the same kind of mistake every time; an erratic one is sometimes odd even when sharp. Erratic moves are always legal moves, and the policy never folds when checking is free.
- **Not fully deterministic.** Fully deterministic play is exploitable. The traits set thresholds and probabilities; the action is the threshold rule plus noise drawn from the **secret round seed** (as in D29). The same seed and moves replay exactly; players cannot predict it.
- **Storage:** `characters.play_style_json`, one style per character (not per server or game), not part of the card export. Edited in the character editor under "Play style" (sliders grouped Style / Reads / Mind, with "Reset to default"). Saves use `store.write_admin()` with a revision check.
- **Unset style:** keeps today's behavior (traits derived from the character id), so no character is blank and existing Doubt tables are unchanged. A saved style replaces the derived one in every game.
- **Generate from description:** a button sends the character's description and personality to a model with a structured-output schema (eight integers with ranges, plus a one-line reason each). The result fills the sliders as an **unsaved draft**; the admin reviews and saves. The model comes from the server's profile through `BackendResolver` (key pins apply), with an optional cheaper-profile setting, and counts toward the spending cap. The character text is untrusted: the output is validated and clamped as integers, so an injection can change only numbers the admin sees before saving. Reasons are shown, not stored. A failure shows a toast and leaves the sliders alone.
- **Style versus the model.** D26 holds: with a model available the model picks one legal move, and the prompt carries the play style in words. The numbers drive the fallback. A Games tab setting, **Character moves: model / style only** (default model), selects style only: the policy always plays and the model is used for table talk alone, to save calls.

### Texas Hold'em (FEAT-33, schema v20)

**Rules (v1).**

- No-limit, 2–9 seats, heads-up allowed, standard rules for the button, blinds and short all-ins. Best five of seven, ties split, the odd chip goes to the first seat left of the dealer button. Side pots are in v1.
- **Cash table.** A seat pays a buy-in from its wallet (a member's or, for a character, its own) into a **stack**. Stacks carry across hands. Leaving or busting cashes the stack out to the wallet. Between hands members can sit, leave, rebuy or top up (up to the max buy-in); nobody joins mid-hand. A character seat never rebuys: it leaves on a bust. The table closes with fewer than 2 seats or after an idle timeout.
- **Blinds and buy-in:** fixed per server, no escalation, set in the Games tab. Small blind S (default 5), big blind 2S, buy-in 40–200 big blinds. No ante, rake, straddle, run-it-twice, sit-out, pre-actions, tournaments, pot-limit or fixed-limit in v1.
- **Actions:** Fold, Check or Call, Raise. Raise opens a private picker with min, ½ pot, pot and all-in, plus a modal for an exact amount.
- **Showdown:** only seats that must show to win a pot reveal; the last aggressor shows first and a seat that cannot win may muck. If everyone else folds nothing is revealed, and the winner may press **Show** once.
- **Timers:** a member's action timer defaults to 45 s (Games tab). A timeout checks if free, else folds. Two in a row remove the seat (cashed out after the hand).
- **Seed hash:** as D26: a short hash of each hand's seed is shown at the deal and the seed is revealed after the hand settles.

**Assist.** The host picks it in the lobby, only if the server allows it (Games tab, off by default); the table shows "Assist: on". In **My cards** (and `/game hand`) a member sees their current best hand and their chances of finishing with each of the three likeliest hand types by the river, against random opponent hands. It uses only the member's own cards and the board. The odds are exact or a capped Monte Carlo with a fixed per-hand seed, so numbers do not change between presses. Characters never get it, and it never goes into a prompt.

**Seats and surface.** The same as I Doubt It (D29): humans-only in an unbound game channel; in a bound one members bring favorites and the host can Fill seats; `/game start game:holdem`; a table thread with a pinned board (community cards, pot, stacks, button, blinds), one edited turn message with the buttons, a hand-by-hand one-line result, and an end-of-table summary. Table talk is seated characters only. A character's prompt gets the public state plus its own hole cards, never another seat's, and never the assist odds.

**Characters.** The engine owns the cards, the legal actions, the pot and the result. A model sees its seat's view only and picks one legal action id (`fold`, `call`, `raise:<amount>` from the menu min / ½ pot / pot / all-in); an optional in-character line goes through the webhook. Any error, timeout, daily limit or spending cap, or the **style only** setting, uses the Play style policy.

**Store (schema v20, backup before upgrade).** No hands are stored: each hand replays from seed plus the append-only move log. `game_seats` gains `stack` (`stake` stays the total bought in); `game_rounds` holds each hand's seed, rules snapshot (blinds, assist), starting stacks and button. The wallet moves only at sit (buy-in), rebuy, leave or bust (cash-out) and cancel (refund), each in one transaction with its stack change; per-hand wins and losses change stacks inside the table, so the ledger gets no entry per bet. Reasons: "Texas Hold'em buy-in / rebuy / cash-out / refund (table T)". At recovery an unfinished hand is voided (stacks return to their pre-hand values), then the table closes and every stack is cashed out.

| Part | Content | Review |
| --- | --- | --- |
| A | `games/holdem.py`: deal, betting rounds, side pots, hand evaluation, showdown, replay, rule normalizer; the assist odds; the fallback policy reading `PlayStyle`. Pure Python. | Sonnet; Opus pass on side pots |
| B | Schema v20, store API, buy-in / rebuy / cash-out ledger, recovery. | Opus (migration, money) |
| C | Discord: lobby, Fill seats, thread, board and turn messages, raise picker, My cards with Assist, timers, restart recovery, model chooser. | Opus (concurrency) |
| D | Table talk with hidden-information handling, Games tab section (blinds, buy-in range, timer, assist allowed, character moves), user docs. | Opus for the prompt block; Sonnet for the rest |

FEAT-32 parts: (A) `games/playstyle.py` (traits, validation, defaults, derived-from-id) and the policy noise helpers; (B) schema v19, store API, revision checks; (C) character editor section; (D) "Generate from description" and the Character moves setting. Review: Opus for B and D (schema, model call on untrusted text), Sonnet for A and C.

