# Decision-model evaluation (FEAT-43)

Research prototype, nothing in `llmcord_core/` uses it. Compares OpenAI Decisions (`gpt-6-luna`), Microsoft Decision-1 (via OpenRouter),
Haiku 5.5 and Gemini 3.5 Flash-Lite on the spoken-move sets: 195 blackjack rows (`prototypes/spoken_moves/phrases.csv`) and the 116 doubt
phrases of `tests/test_spoken_doubt.py`. Doubt language: Hangul incl. compatibility jamo (U+3131-318E) is ko, plus ASCII letters is mix. Design and wire formats: `docs/engineering/research/decision-models-eval.md`.

- `evalcore.py`: pure helpers (items, request building, answer parsing, scoring, statistics).
- `run_eval.py`: the CLI. Tests: `tests/test_decision_eval.py` (fake transport, no keys).

## Use
    python prototypes/decision_eval/run_eval.py                    # dry run: call counts and estimated USD, no key, no request
    python prototypes/decision_eval/run_eval.py --run --limit 20 --candidates or-haiku,oai-choice   # small paid run
Flags: `--candidates` (ids: oai-choice oai-predicate or-d1-choice or-d1-noul or-haiku or-flashlite), `--shuffles K` (3), `--limit N`,
`--out DIR`, `--max-usd` (0.50), `--tau` (0.5), `--run`.

## Safety
- Keys come from `OPENAI_API_KEY` (`oai-*`) and `OPENROUTER_API_KEY` (the rest), environment only. A missing one stops the run before any request.
- Keys go to the poster only; never printed, logged, written, or put in a URL. Errors are scrubbed of the key value.
- `--out` defaults to `~/decision-eval-out` and is refused inside the repo. Results hold phrases, ids and answers, no headers.
- `--max-usd` stops the run once summed cost passes it (provider usage cost, else the estimate).
- Estimate prices per 1M tokens (in/out): OpenAI Decisions $0.10/0 and Decision-1 $0.042/0 from the research note; Haiku $0.10/$0.50 and Flash-Lite $0.30/$2.50 are OpenRouter list prices, 2026-10-12. Flash-Lite reasoning is mandatory, so it is sent as effort minimal, excluded, max_tokens 256, and costed at 100 output tokens.
- `--run` refuses (exit 2, before any call) when the dry-run estimate exceeds `--max-usd`; a dry run prints an "exceeds" line.
- `results.json` is rewritten as the run goes (and on an interrupt or crash), so paid results are never lost.
- Errored calls add nothing to the spent total (providers do not normally bill errors); five consecutive errors skip the candidate.
- Calls are sequential. An error counts as an invalid answer; 5 in a row skip the rest of that candidate.

- No extra smoke calls: a small run (`--limit 6 --candidates <one>`) is the smoke test.

## Reading the output
`results.json` has one record per call; `summary.json` (and the markdown summary printed to stdout) has, per candidate and task (bj, doubt): right-move and
silent-on-chatter rates (with en/ko/mix splits), invalid/refusal/error counts, p50/p95 latency, cost per 1,000 calls, order stability
(shuffled choice candidates), AUROC of top probability vs correctness, accuracy per probability bucket, and a tau sweep. For the
shuffled candidates an item's answer is the majority over its K orders (tie: order 0); latency and cost use every call.
Latency is per request including connection setup (a fresh httpx client per call), so p50/p95 are cold-connection numbers.
Silent-on-chatter counts only a strict `talk` answer, while the pass rule also accepts an `accept`-listed move on talk rows.
Cost per 1,000 calls is the mean over non-errored calls; cost per decision is total cost over attempted items (shuffled candidates pay K calls per decision).
Unattempted items count as invalid in the denominators; coverage shows attempted/total.
