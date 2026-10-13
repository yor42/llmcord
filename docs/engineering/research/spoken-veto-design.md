# Design: decision-model veto behind the local spoken-move reader (FEAT-49, proposed)

Status: **decided by the owner on 2026-10-13 (see [Decisions](#decisions-owner-2026-10-13)); FEAT-49a in progress, 45b waits for the schema go-ahead below.** It follows from FEAT-47/44 ([results](decision-models-eval.md)): behind e5, a per-move yes/no reader removed all 12 leaked chatter lines on the synthetic set for 4 lost moves out of 124 (Decision-1 `noul`, tau 0.5).

## Goal and non-goals
- Goal: fewer wrong move-check buttons on table chat. e5 alone gave a button to about 1 chatter line in 5.
- Only vetoes. The reader can drop or reorder the moves e5 offered; it never adds a move, never rescues a message e5 called silent, never raises `unsure` to `sure`.
- Blackjack only. Doubts keep the `read_doubt` word rule (every LLM was worse). Hold'em is not wired.
- Not a replacement for e5. With the veto off, or any failure, behaviour is exactly today's.

## What leaves the Pi
Only when all hold: the operator configured a provider and key, a server admin turned the veto on, the member has spoken moves on (`spoken_member_on`, on by default) and has not opted out of the veto, it is that member's turn, the message is at most 200 characters, and e5 did **not** call it silent (about 72% of messages on the test set; far fewer in real chat, where most lines are not move-like).

Sent: the message text, the legal move names of the current hand, and a fixed context sentence. Not sent: user, member, guild or channel ids or names, table or hand state, cards, history, other messages. The request has no identifier of ours. The provider still sees the bot's IP and its own account.

Not logged: message text, provider response bodies, keys. Logged: counts, status codes, latency, band before and after.

## Where it goes (`games_spoken_discord.py`)
In `_read`, after `interpret.decide(...)` (line 203) and the silent-band return (208), before the lock and button build (213-222):

1. `decision.band != "silent"` and veto is on for the guild and the breaker is closed → `await veto.ask(text, legal)` with a hard timeout.
2. `veto.apply_policy(e5_reading, reading)` (the function validated in `prototypes/decision_eval/cascade.py`) → new band and moves.
3. Silent after the veto → return as for e5-silent (no buttons). Otherwise build the buttons from the new moves.
4. After the await, re-run `_current()` staleness checks as lines 185/191 do, because the table may have moved on.

(The "member" gate above means members of a server where spoken moves are on, which is every member unless they switched it off.) The e5 reading already exists before the call, so a timeout, error or open breaker simply continues with e5's buttons (**fail-open**). The player sees no difference except the extra button that was always there today. Each message spawns its own check, and a superseded reading already inside `ask` stays in flight until it answers or times out; only the per-guild limiter bounds the number of calls.

## New code (small, separable)
| Piece | Where | Notes |
| --- | --- | --- |
| Pure request/response/policy | `llmcord_core/spoken/veto.py` | Builds the provider bodies, parses answers to `{move: p}`, `apply_policy`. No I/O. Ported from `evalcore.build_request`/`parse_answer` and `cascade.apply_policy`, with the prototype's tests moved over. |
| HTTP client | `llmcord_core/spoken/veto_client.py` | `httpx.AsyncClient` (already pinned in `requirements.txt`), no redirects, total timeout 1.0 s (connect 0.3 s), **no retries**, response size cap, exceptions mapped to one `VetoError`. Circuit breaker: 5 consecutive errors → skip the call for 60 s. |
| Wiring | `games_spoken_discord.py` | Step above; the client is created lazily and closed in `close()` with the other tasks. |
| Switch | `admin_store.py` pseudo row `spoken_veto` in `game_settings` | Same helper pattern as `spoken_keep_phrases` (`_pseudo_flag`/`_set_pseudo_flag`, `write_admin`, revision check). Default off. **No schema change, no `user_version` bump.** |
| UI and command | `games_ui.py` Games tab beside the spoken switches (176-205); a `/game` subcommand beside `spoken_moves` | Wording below. |
| Docs | `docs/server-guide.md`, `docs/admin-console.md`, `docs/glossary.md` | Disclosure text. |

It does not go through `ModelGateway`/`models.py`: the Decisions wire formats are not Chat Completions, the veto has no dialogue/budget semantics, and a dashboard profile must never be able to point it at another host. If the owner prefers one place for provider code, a `decisions` provider kind in `models.py` is the alternative; it costs a profile-validation change and key-pin plumbing for no extra behaviour here.

## Provider and key handling
- Operator, not dashboard, chooses: `LLMCORD_SPOKEN_VETO=off|openrouter-d1|openai-predicate` (default `off`), keys from `OPENROUTER_API_KEY` / `OPENAI_API_KEY`. Never stored in the database, never shown, audited or logged.
- **The host is fixed in code**, not configurable: `openrouter.ai` for Decision-1 (`/api/alpha/decisions`, one `noul` question per legal move), `api.openai.com` for `gpt-6-luna` (`/v1/decisions`, one `predicate` per legal move). The key is sent only to that host, checked before every call, matching the key-pin invariant. It does not use dashboard profiles or `LLMCORD_KEY_HOSTS`, so there is no way to redirect the key.
- Default recommendation: Decision-1 via OpenRouter (best numbers, about $0.005 per 1,000 messages). Risks to accept knowingly: alpha endpoint, unversioned weights, one upstream provider. OpenAI `predicate` is the fallback choice (public beta, about 0.27 s p50 but looser bands: 3 wrong `sure` rows left at tau 0.5).
- Policy constants in code, not in the dashboard: cover = every non-silent e5 reading, tau = 0.5. They come from the same 195 rows they were picked on, so they are provisional (see shadow mode).

## Opt-in and consent (separate from `/memory opt_in`)
Three layers, all default off:
1. **Operator**: provider and key configured in the environment.
2. **Server admin** switches "Decision-model check for spoken moves" on, in the Games tab or by the slash command. The label and reply must say plainly: *"When on, a message that the local reader takes for a spoken move is sent to {provider}, a third-party service, to double-check it. Only the message text and the legal moves are sent, only for members who turned on spoken moves, and only on their own turn. Off by default."*
3. **Member**: the existing `spoken_member_on` gate (on by default), plus a per-member **opt-out of the veto** (`/game spoken_veto` for the member; owner decision: server-wide default on once the admin enables it, each member can opt out). Members get a one-line notice that their message may be double-checked by a third-party service and how to opt out.

**Schema:** the per-member opt-out needs a new `member_settings` column (`spoken_veto`), i.e. migration v23 (main took v22 for the server language) with backup-before-upgrade. `CLAUDE.md` requires the owner's go-ahead for a schema change; that go-ahead is requested in the FEAT-49b brief and is not assumed. Alternative without a schema change: none that is clean (overloading `spoken_moves` values would be a hack).

Provider terms: the owner reports that the only provider for Decision-1 on OpenRouter is Azure and that its listing says: "Private: this provider does not train on prompts and does not retain prompt data." I have not independently verified this; it is the owner's reading of the OpenRouter provider page. The OpenRouter request must not opt into prompt logging, and the docs should quote the same line with the date it was checked.

## Limits and cost
There is no per-user or per-table rate limit today (only one pending check per table). The veto adds an in-memory per-guild token bucket (default 30 calls/minute) and a per-process daily call cap (default 5,000); beyond either, fall back to e5. At about $0.005 per 1,000 messages the money is negligible, the caps are for outages, bugs and abuse. Counters reset on restart; they are not persisted.

## Shadow mode (replaces the unavailable D35 check)
I could not validate thresholds on kept phrases because those are production data. Instead, an operator setting `LLMCORD_SPOKEN_VETO_SHADOW=1` makes the bot call the reader and **not** apply the result. It records counts only, no text: per guild and day, rows e5 offered, rows the reader would have vetoed, and for each whether the player then pressed a button. A veto the player would have overridden by tapping anyway is a false veto; a vetoed row nobody pressed is probably chatter. That gives real-chat precision and recall for the veto without storing any message. Shadow mode sends the same data as live mode, so it needs the same layers 1 and 2 opt-in. Run it for a week or two per server before turning the veto live.

## Drift guard
Decision-1 weights change without notice. Add `scripts/check_spoken_veto.py` (operator-run, uses only the synthetic phrases from `phrases.csv`, prints the same metrics as the prototype): run before enabling, after any provider notice, and weekly while enabled. Live, a rolling veto rate per guild is logged; a sudden jump to near 0% or near 100% means the weights moved, and the operator turns the switch off.

## Tests (all offline, fake transport)
- `veto.py`: body builders for both providers, parsing of every answer shape including refusal, missing and non-finite probabilities (→ error → fail-open), `apply_policy` cases from `tests/test_decision_cascade.py`.
- Client: `httpx.MockTransport` for success, timeout, 4xx/5xx, oversized body, redirect refused, breaker open/close, the key present only in the header and never in an exception string or log record.
- Wiring (`tests/test_games_spoken_discord.py` style, existing fake `Runtime`): veto off → identical to today; veto on drops buttons; error/timeout/open breaker → e5 buttons; table moves on during the await → no stale buttons; text never reaches logs; no call when e5 is silent, it is not the member's turn, or the member switch is off.
- Store/UI: switch default off, guild isolation (`guild_id` scoping), revision conflict through `write_admin`, Games tab and slash-command wording. Browser suite only if the Games tab layout changes beyond a switch row.
- Fake keys only (`FAKE-...-not-a-real-key`), as in `tests/test_decision_eval.py`.

## Proposed split (each one small item)
1. **FEAT-49a** pure `veto.py` + client + tests, not wired. No behaviour change.
2. **FEAT-49b** switch, Games tab row, slash command, disclosure text, wiring, limits, breaker; default off.
3. **FEAT-49c** shadow-mode counters and `scripts/check_spoken_veto.py`.
4. Rollout: operator enables shadow on one server → review counts → decide live and tau.
Item 2 touches consent and an outbound data path, so it should get an Opus review.

## Risks
- A vetoed real move costs the player a tap on the normal button (4 of 124 on the test set at tau 0.5; 3 of the 4 were Korean).
- Short elliptical stands and slang were the hard rows for every reader ("됐어요", "I'm happy with this", "send it").
- Alpha/beta endpoints can change shape; the parser must fail closed to e5 on anything unexpected.
- Real chat is wider than 195 synthetic phrases; the numbers above are optimistic until shadow mode says otherwise.
- Provider outage or slowness is invisible to players by design, but then the benefit is lost silently; log the breaker state change once.

## Decisions (owner, 2026-10-13)
1. Sending the text is acceptable **if opt-in**: server admin enables it (server-wide default for its members), each member can opt out.
2. Provider: **Decision-1 via OpenRouter**; the non-release (alpha) endpoint is acceptable. OpenAI `predicate` is not built.
3. Members get a one-line notice.
4. The bot currently runs on a single server, a testbed. Taken as: no separate shadow period is required before going live there. The count-only shadow telemetry (FEAT-49c) stays proposed as the way to validate tau on real chat, but it no longer gates 45a/45b.
5. Code stays out of `ModelGateway` (`spoken/veto.py`, `spoken/veto_client.py`).
6. Provider terms: the owner's reading of the OpenRouter provider page (Azure, "does not train on prompts and does not retain prompt data").

7. Migration v23 (`member_settings.spoken_veto`, default on) approved 2026-10-13.

Implemented differently from the proposal on one point: the member notice is a separate one-line channel reply posted **before** anything is sent. The first time a member would be affected since the bot started, the bot posts the notice and skips the veto for that message; later messages use it. The notice is also in every `/game spoken_veto` reply. The mark is in memory only, so a restart repeats the notice. The client enforces its own 1.0 s total deadline, which counts as a breaker failure; the 1.5 s guard around the call is only the outer limit. A message holding a mention, channel or custom-emoji token is never sent, and while a member's notice post is in flight their other messages are not sent either (the mark is set only once the notice was posted).

Status: FEAT-49a/b built (fbff1c5, 9c28524, dbd8631, 601b240, 6f84205). Nothing is enabled until the operator sets `LLMCORD_SPOKEN_VETO=openrouter-d1` and `OPENROUTER_API_KEY`. Shadow telemetry (49c) and the drift script stay proposed.
