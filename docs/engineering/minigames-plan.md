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
- **Seed hash:** a short hash of each hand's seed is shown at the deal; the seeds are revealed only when the table closes (D33), all hands at once in the closing posts, so folded hole cards cannot be replayed while play continues. Blackjack and I Doubt It still reveal per round (D26).

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


### Hold'em engine API (FEAT-33 part A)

`llmcord_core/games/holdem.py`, pure and immutable like `doubt.py`. Cards are `(rank, suit)` tuples, ranks `2..10, J, Q, K, A`, suits `♠ ♥ ♦ ♣`.

- **Constants:** `MIN_SEATS = 2`, `MAX_SEATS = 9`, `DEFAULT_RULES = {"small_blind": 5, "min_buy_in_bb": 40, "max_buy_in_bb": 200, "action_timeout": 45, "assist": False}`, `TIMEOUT_RANGE = (15, 300)`. The big blind is always 2 × small blind. `normalize_rules(value=None)` accepts a dict or JSON string, ignores unknown keys, raises `ValueError` on a bad value (small blind a whole number 1..10**6, min ≤ max buy-in with min ≥ 20 and max ≤ 1000, `assist` a real bool). `rules_text(rules)` and `how_to_play(rules)` give the one-line and the long text.
- **Deal:** `new(seed, seats, rules=None, button=0)`. `seats` are `Seat` objects whose `stake` is the seat's **starting stack for this hand** (all > 0; seats that are busted or out are the caller's to leave out). Deck is `shuffled(52 cards, seed)`; hole cards are dealt one at a time starting left of the button, then the five board cards come off the same deck (no burn cards). Heads-up the button posts the small blind and acts first before the flop; otherwise small blind is left of the button, big blind next. A short stack posts what it has and is all-in. Blinds are posted by `new`.
- **State** (frozen): `seed, seats, rules, button, stacks, bets (this street), contributed (this hand), folded, all_in, holes, board_full, street ('preflop'|'flop'|'turn'|'river'|'showdown'|'done'), turn, to_call, min_raise_to, history, finished`. Never print `seed`/`holes`/`board_full` in repr.
- **Moves** (strings): `fold`, `check`, `call` (all-in for less when short), `raise:<n>` where `n` is the seat's **total bet for the street** (a bet when nobody has bet). `legal_moves(state, seat)` returns the ordered menu: `fold` (omitted when `check` is available), `check` or `call`, then `raise:<n>` for the distinct amounts among min raise, ½ pot, pot, and all-in that fall inside the legal range (an all-in below the min raise is legal and listed). `apply` accepts any integer `raise:<n>` between the min raise-to and all-in (or the all-in alone when below the min). A raise short of a full raise does not reopen the betting to seats that already acted. `IllegalMove` for anything else, including the wrong seat or a finished hand. `apply(state, seat, move)` returns the new state, advances streets, skips seats that are all-in, deals out the board when everyone left is all-in, and finishes the hand.
- **Replay and views:** `replay(seed, seats, moves, rules=None, button=0)` with `moves` an iterable of `(seat, move)`. `view(state, seat=None)` returns a `View`: board shown so far, pot, stacks, bets, folded/all-in flags, turn, to-call, min raise-to, history, and that seat's own two hole cards (`None` for `seat=None` or a spectator). Hole cards of others appear in `Result.shown` after showdown; once the table closes, the revealed seeds also let anyone replay folded hole cards (D33: not before).
- **Result:** `result(state)` returns `Result(finished, board, pots, seats, shown)`. `pots` is a tuple of `Pot(amount, eligible, winners, hand)` main pot first, built from `contributed` (side pots; an uncalled excess is returned to its owner, never a pot; folded seats' chips stay in the pots they fed but they are not eligible). `seats` has a `SeatResult(seat, start, end, net, outcome)` where `outcome` is `win`, `split`, `lose` or `fold` and `end` is the stack after the hand (so the sum of `end` equals the sum of `start`). `shown` lists the seats that reveal at showdown: all seats still in when betting ended with a contested pot, in order starting with the last aggressor on the final street (or the first seat left of the button when nobody bet), and a seat that cannot win any pot may be left out. Nothing is revealed when everyone else folded (`shown = ()`). Ties split a pot evenly; the odd chip goes to the first winner left of the button.
- **Hand evaluation:** `evaluate(cards)` takes 5..7 cards and returns a comparable `Rank` tuple (category 0..8, then tiebreak ranks) with the wheel (A-2-3-4-5) as the lowest straight; `hand_name(rank)` returns text such as "Two pair, Kings and Fives". `best_hand(cards)` gives the five cards used.
- **Assist:** `assist(hole, board, opponents, seed, samples=2000)` returns `Assist(current, chances)`: `current` is the made-hand name now (or "High card, X"), `chances` the three likeliest final hand categories by the river as `(name, probability)` pairs, summing to at most 1, probabilities rounded to 3 places. Exact enumeration when the unknown cards allow it (flop and later with one or two cards to come), otherwise a Monte Carlo of `samples` runs seeded from `seed` plus the cards, so the same inputs always give the same numbers. `opponents` is the count of random opponent hands (≥ 1), used only for the optional "chance to win" figure `win` on `Assist` (also deterministic).
- **Policy:** `policy_action(view, style, seed)` returns one legal move for the viewing seat, from its own view only. `style` is a `playstyle.PlayStyle`. Strength comes from a hand-strength estimate (a preflop starting-hand score; postflop made-hand category plus draw outs) passed through `playstyle.misjudge`; looseness lowers the strength needed to play, aggression raises over calls and sizes raises up the menu, bluffing allows a raise with a weak hand, stubbornness lowers the fold threshold when the seat has already put chips in, risk raises the share of the stack it will commit (all-in appetite), `playstyle.erratic` swaps in another legal move, and `playstyle.never_fold_free` is applied last. All noise comes from the secret round `seed` with a per-decision tag (street and history length), so replays are exact. `unset` styles use `PlayStyle()` (neutral). `timeout_move_for_view(view)` is `check` when legal, else `fold`.

## Spoken moves (FEAT-36, FEAT-37, FEAT-38, D32, user 2026-10-11)

A member can play a turn move by chatting ("fine, hit me", "한 장 더") instead of pressing a button. The bot never plays it directly: it answers with a **move check** whose buttons the member presses. Model research: [research/nl-game-actions-models.md](research/nl-game-actions-models.md).

- **One shared interpreter.** Each game declares its legal turn moves for the current state and built-in EN/KO example phrases (repo JSON, like `items/recipes/*.json`). The interpreter returns a move and an optional amount (unused by blackjack; Hold'em "raise" without a clear amount opens its raise picker). Blackjack first (FEAT-37), then I Doubt It (FEAT-38); Hold'em support is part of FEAT-33's own work. Turn moves only in v1; table moves (join, deal, again, leave, close) stay on buttons.
- **Cascade, cheapest first.** (1) A new first step in `on_message`, beside `note_thread_message` and before the bound-channel check (blackjack can run in unbound channels): is a table in this channel waiting on this author? Only on yes: (2) word list (EN/KO, single-jamo forms like "ㄱㄱ"); a match counts as sure, a negation marker (don't/no/not/안/말/하지 마) drops to the A-or-B check; (3) multilingual embedding, nearest neighbour over example phrases masked to the legal moves, "table talk" as negative examples plus a threshold; (4) move check or silence. No LLM in v1; an optional capped LLM step through the existing model profiles (key pins apply) only if logs show the right move is often missing.
- **Bands.** Sure: `[Hit] [Just table talk]`. Unsure: `[Hit] [Stand] [Just table talk]`. Below the floor: silent. Score is top-1 similarity plus the margin to the best different move; thresholds per model, fitted from logged presses.
- **Move check.** Always shown in v1, even for exact matches. A public message (a plain message cannot get an ephemeral reply) whose buttons only the member whose turn it is can press; others get a private "This isn't your turn". English labels matching the table's buttons (localization comes with FEAT-30). Answered → edited to "✔ Hit"; "Just table talk" → deleted, with a private hint "Tired of these? `/games spoken_moves off`". A newer spoken move replaces a pending check. It belongs to the turn: when the turn ends for any reason it is deleted, a late press gets "That turn is over", and the turn timer does not pause.
- **Table talk still reaches characters.** In table threads the message continues to the skit engine as usual; the interpreter only adds a move check.
- **Switches.** Server: **Read moves from chat** in the Games tab, off by default. Member: `/games spoken_moves off|on`.
- **Kept phrases.** Server setting **Keep phrases for tuning**, off by default, retention follows the server's retention period (like the turn log). Keeps the phrase, scores, band and button pressed; never from opted-out members; from silent messages only near misses just under the floor. An admin can promote a kept phrase into a per-server example phrase (guild-scoped, added to the built-in set). Not tied to `/memory opt_in`, which covers personal facts only. Schema change in FEAT-37.
- **Runtime.** ONNX Runtime + tokenizer in the bot process (no PyTorch), inference in a worker thread, loaded lazily only when some server has the setting on. The model is baked into the Docker image at a pinned revision with a checksum; no runtime download. Pi 5 CPU; no Jetson.
- **Prototype gate (FEAT-36).** About 150 labelled blackjack phrases (EN, KO, mixed, slang, negations, table talk), Korean reviewed by members. Compare `multilingual-e5-small` and `paraphrase-multilingual-MiniLM-L12-v2` as int8 arm64 ONNX. Pass bar: right move among the buttons for ≥ 95% of real moves, silent on ≥ 85% of table talk, < 100 ms per message on the Pi. **Result (D34, 2026-10-12):** `multilingual-e5-small` int8 with a small trained head (logistic regression on the embeddings, trained on the example phrases and ~370 table-talk lines) reaches about 95% right move / 70% silent on table talk at about 6 ms; the user accepted that bar and the feature ships **Experimental**, opt-in. The word-list step did not help blackjack and is not used for it ([research/spoken-moves-prototype.md](research/spoken-moves-prototype.md)).

### FEAT-37 parts (blackjack spoken moves, D32/D34)

Each part is committed on its own; the server switch stays off by default (Experimental), so partial merges are safe.

- **A. Pure interpreter** (`llmcord_core/spoken/`: `head.py` loads and validates the exported head JSON, `interpret.py` masks to legal moves + talk, renormalizes and picks the band, `embedder.py` Protocol + `FakeEmbedder`, `blackjack.py` move list and button labels, `data/head-e5-blackjack.json` + sha256). No numpy, no model; a golden test against prototype outputs.
- **B. Real embedder and image** (`onnx_embedder.py`: lazy ONNX Runtime + tokenizers + numpy imports, one worker thread, `query: ` prefix, mean pooling, 128 tokens). Pins onnxruntime 1.22.1, tokenizers 0.21.2, numpy 2.3.2. The Docker build fetches the pinned e5 revision, verifies sha256, makes the int8 model and verifies it; no runtime download. About +200 MB image, about 580 MB RSS when loaded (lazy: only when a server has it on).
- **C. Settings accessors** (no schema): `spoken_moves_enabled` / `spoken_keep_phrases` per guild as `game_settings` pseudo-rows like `character_moves`, both off by default.
- **D. Move check in Discord** (`games_spoken_discord.py`): `on_message` hook after `note_thread_message`, before the bound-channel check; in-memory "table waiting on this author" lookup before any inference; re-check round and move count after inference; buttons `llmcord:blackjack:spoken:...` through the existing play path (`expected_moves`); delete on turn end; never swallows the message (table talk still reaches characters).
- **E. Games tab**: "Read moves from chat (Experimental)" and "Keep phrases for tuning" switches.
- **F. Kept phrases and per-server examples** (schema v21, approved D35): `spoken_phrases`, `spoken_examples`, member opt-out column; retention purge in the daily cleanup; promote a kept phrase into a guild-scoped example. Kept phrases record false positives (Just table talk pressed, or the other move of an unsure check) and false negatives (silent or wrong, then the member pressed a table button in the same turn; label = that move), shown in the Games tab for an admin to promote; never automatic.
- **G. `/games spoken_moves off|on`**: member opt-out (purges that member's kept phrases).
- **H. Docs**: glossary, CLAUDE.md module map, this plan.

### Hold'em store API (FEAT-33 part B, schema v20)

**Model.** A Hold'em table is one `game_tables` row (game `holdem`, opened by the host) and one `game_rounds` row per **hand**. The newest round is either `joining` (the lobby, and the gap between hands: the only time anyone sits, leaves, rebuys or tops up) or `playing`. Each hand's `game_seats` rows are its players: `stack` is the seat's **starting stack for that hand**, `payout` its **end stack** once the hand settles, `stake` the total bought in at the table so far (carried forward, grows on a rebuy). `next_round` creates the next lobby with the seats copied (stack := last end stack, or the starting stack when the last hand was voided), busted seats dropped. No wallet moves per hand: only at buy-in, rebuy and cash-out. Hands are never stored (replay from seed + `game_moves`).

**Schema v20** (backup before upgrade; `user_version` 20; bump the version check in `Store._upgrade`): `game_seats.stack INTEGER NOT NULL DEFAULT 0 CHECK(stack>=0)` (ADMIN_SCHEMA and the ALTER list; the v17 rebuild still copies old rows), `game_rounds.button INTEGER` (nullable), and a new `game_cashouts(id, guild_id, table_id, kind, ref_id, bought_in, cashed_out, created_at)` with an index on `(guild_id, table_id)`, one row per seat leaving the table (leave, bust, table close). Existing columns and data are untouched; v19 databases upgrade in place.

**Code layout.** A new `llmcord_core/holdem_store.py` with `class HoldemStore` mixed into `Store` (like `ItemsStore`: `class Store(ItemsStore, HoldemStore, AdminStore)`); it uses AdminStore's helpers (`write_admin`, `_append_ledger`, `_character_debit_locked`, `_pay_seat`, `_game_round`, snapshots). The registry gets a `holdem` entry (`cash=True` flag on `Game`, `min_seats=2`, `max_seats=9`, `ledger_label='Texas Hold\'em'`) whose hooks are the Hold'em engine's. Blackjack and I Doubt It behavior must not change (all existing tests keep passing).

**Method contract** (all inside `write_admin()`, scoped by `guild_id`, `GameError` for every refusal with a plain-words message; "joining only" means `GameError('This round has already started.')`):

- `open_table(..., 'holdem', ...)`: as today; `_new_round` pins the full normalized rules. The first lobby has `assist` False.
- `join_round(guild, round, user, stake)`: for holdem `stake` is the **buy-in**, checked against the round rules (`min_buy_in_bb * 2 * small_blind` .. `max_buy_in_bb * 2 * small_blind`) and the wallet; ledger "Texas Hold'em buy-in (table T)"; row `stake = stack = buy_in`. Joining only; one seat per member; max 9.
- `join_with_favorites(...)` and `fill_seats(...)`: the same for holdem. Character buy-in is `min(member's buy-in, character wallet, max buy-in)` (favorites) or `min(100 big blinds, character wallet, max buy-in)` (house seats), and a character below the minimum buy-in is skipped as `'broke'`. House seats (`fill_seats`) are host-only; both use the character's own wallet.
- `rebuy(guild, round, user, amount)`: joining only, the member must hold a seat; `stack + amount <= max buy-in`; amount > 0 and the wallet must cover it; ledger "Texas Hold'em rebuy (table T)"; `stack` and `stake` both grow. Characters never rebuy.
- `leave_round(guild, round, user)`: joining only; for holdem pays the seat's `stack` back (clamped at the balance cap like other payouts; ledger "Texas Hold'em cash-out (table T)") and the stacks of characters the member brought, writes `game_cashouts` rows (bought_in = the seat's `stake`, cashed_out = what was paid), deletes the seats, renumbers `seat_index` 0..n-1.
- `set_table_assist(guild, round, user, on)`: host only, joining only; raises `GameError` unless the server's holdem rules allow assist; updates the round's rules snapshot (`assist`). `next_round` carries the choice forward while still allowed.
- `deal(guild, round)`: joining only; needs 2..9 seats, all with `stack > 0`; sets `button` (first hand: `int(seed_hash(seed)[:8], 16) % n`; later hands: the next seat clockwise after the previous hand's button holder who is still seated); status `playing`; builds the engine state with `Seat(index, kind, ref, stack)` and the pinned rules; settles at once if the engine finished the hand at the deal.
- `play(...)`: unchanged contract (`expected_moves` conflict check, actor rules, `IllegalMove` → `GameError(spec.illegal_text(...))`); on `after.finished` calls the cash settle below.
- **Settle (cash games):** for each seat set `outcome` (the engine's `win` / `split` / `lose` / `fold`) and `payout := end stack`; **no wallet movement, no ledger rows**; round `settled`. Seats whose end stack is 0 are "busted" (the snapshot shows `busted: True`; they are dropped when `next_round` copies).
- `next_round(guild, table_id)`: for holdem, requires the newest round to be `settled` or `cancelled` (voided) and creates the lobby copy described above (`brought_by` kept; busted seats get a `game_cashouts` row with `cashed_out = 0` and are dropped; characters a busted member brought are cashed out to their own wallets at their end stacks, with a `game_cashouts` row, and not copied: D33/MNT-57). If the table is closed, `GameError`. The new lobby carries the previous round's pinned rules (blinds, buy-in range, timeout; D33), so later server changes apply to new tables only; only `assist` is re-checked against the server as above. Rebuy and top-up bounds come from the round's pinned rules (`_buy_in_range`).
- `cancel_round(guild, round, reason)` and `close_table(guild, table_id)` for holdem: any unfinished round is voided (`cancelled`, seats keep their starting stacks, no per-hand result), then **every seat is cashed out** at its `stack` (ledger "Texas Hold'em cash-out (table T)", `game_cashouts` rows) and the table closes. `cancel_round` on a holdem round therefore ends the table. Cash-out never fails on the balance cap (pays what fits, like `_pay_clamped`). A deleted character's wallet is gone (pays 0, as today).
- Snapshot (`round_snapshot` / `latest_round`) for holdem: seats carry `stack` (starting), `end` (payout, once settled), `busted`, `stake`; `rules` includes `button` and `assist`; `view` is the engine's public view; `legal` the public menu for the turn seat (the menu reveals nothing hidden). `seat_view` returns that seat's engine view with its own hole cards. A damaged move log or unreadable pinned rules is a `GameError` like today.
- `game_table_summary` for holdem: per hand the facts `{'pot': total pot, 'winners': [seat indexes], 'hand': winning hand name or None}` and each seat's net = `end stack - starting stack` (not `payout - stake`). Blackjack and I Doubt It summaries unchanged.
- `table_cash_summary(guild, table_id)`: the end-of-table list of `{kind, ref_id, name, bought_in, cashed_out, net}` aggregated from `game_cashouts` (a seat that left and sat again is one row), plus the seats still sitting with their current stacks and stake.
- `unfinished_rounds()`: includes holdem rounds (`joining` or `playing`); the Discord layer closes such tables at startup (void, cash out). No other recovery code in the store.

Hold'em settings are `game_settings` rows (game `holdem`) through the existing `game_enabled` / `set_game_enabled` / `game_rules` / `set_game_rules` (registry-driven); the rules are `holdem.normalize_rules`, where `assist` means "the host may turn it on".

### Hold'em Discord layer (FEAT-33 part C)

New `llmcord_core/games_holdem_discord.py` with `HoldemTables`, built on the same pieces as `games_doubt_discord.py` (`TableRuntime`, one lock per table with store work only, `_Out` records sent after the lock, edit sequence numbers, debounced board, recovery, sweeper). The bot gets `bot.holdem = HoldemTables(bot)` and the same hooks the Doubt tables have (recover on startup, close on shutdown, `close_for_channel` and `sweep` from blackjack's channel code, thread-message notes, dynamic button registration). Custom ids: `llmcord:holdem:<action>:<table>:<round>:<moves>`. Split in three steps:

- **C1a, tables and seats.** `/game start game:holdem [favorites]` (a `bet` gets the note "Texas Hold'em uses the server's blinds; you choose your buy-in when you sit."), `/game rules game:holdem`, `/game hand` in a Hold'em thread. Lobby message and table thread as I Doubt It. Lobby buttons: **Sit** (a modal for the buy-in, default 100 big blinds, shown with the allowed range), **Leave** (cash out), **Fill seats** (host, bound channels), **Assist on/off** (host, only when the server allows it; the lobby shows "Assist: on"), **Deal** (host), **Close table** (host; cashes out). Pinned board message: blinds, buy-in range, seats with stacks, dealer button, Assist state. The lobby waits 5 minutes for a first hand, then closes the table and cashes out.
- **C1b, the hand.** Deal posts the board and a single **turn message** (edited in place, bumped when buried like I Doubt It): street, community cards, pot, whose turn, stacks and bets, the last action, and the buttons **Fold**, **Check / Call <amount>**, **Raise** (an ephemeral picker with the menu amounts min / ½ pot / pot / all-in and an **Exact amount** modal; illegal amounts get a private reply), **My cards** (ephemeral: hole cards, and when Assist is on the member's current hand and the odds, computed off the event loop and cached per hand, seat and street). Only the seat on turn can act; double presses apply once (`expected_moves`). A member's action timer is the table's `action_timeout`: a timeout checks if free, else folds; two timeouts in a row cash the seat out after the hand. Character seats play after a short pause by `holdem.policy_action` with the character's saved Play style (the model chooser is C2). Showdown: a result message with the board, each shown hand and its name, the pots and who won them, and each stack change; if everyone else folded nothing is shown, and the winner gets a **Show** button that posts their two cards once. After a hand settles, `next_round` opens the next lobby in the same thread: the turn message becomes "Next hand in 15 s" with **Sit**, **Leave**, **Rebuy** (modal, top up to the max buy-in) and **Deal now**; at the end of the pause the next hand deals automatically if two or more seats have chips, otherwise the table waits (members can sit) and closes after 10 minutes idle. A bust removes a seat (a character leaves, a member may sit again). The table ends when the host closes it, when it idles out, or when fewer than two seats remain: the end-of-table summary (`table_cash_summary`) is posted in the thread and the thread archives after 10 minutes.
- **C2, character decisions.** The model chooser: a character on turn sends its seat view (its hole cards, public state, legal action ids) to the model, which returns one legal action id and an optional in-character line. Errors, timeouts, the daily table-talk limit, the spending cap and the style-only setting all use `holdem.policy_action`. The play-style line goes in the prompt for characters with a saved style. This follows what `game_chooser.py` does for I Doubt It.
- **Restart recovery.** At startup an unfinished Hold'em hand is voided, the table is closed (the store cashes out every stack), and one line is posted in the thread. The thread is archived.
- **Out of scope for C:** table talk (part D), the Games tab section (part D).
