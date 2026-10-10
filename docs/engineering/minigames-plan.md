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

Six-deck shoe shuffled per round from the seed; dealer stands on soft 17; blackjack pays 3:2 (rounded down); hit, stand, double down on the first two cards; no split or insurance in phase 1 (split can follow). Push returns the stake. A round settles when every seat has stood, busted, or timed out.

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
