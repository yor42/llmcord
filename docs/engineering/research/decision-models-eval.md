# Decision models for spoken-move interpretation (research note, 2026-10-12)

Status: findings and benchmark plan only. Nothing is wired into the bot. Needs a go-ahead before it becomes a backlog item (new provider wire format, and sending player chat to a third party).

## Question
Could OpenAI's Decisions API (`gpt-6-luna`) or Microsoft Decision-1 replace or back up the local e5 interpreter, compared with Haiku 5.5 and Gemini Flash-Lite (D37 assumes these are better at similar cost; never measured)?

## Baseline (from repo)
- Shipped path has no LLM: e5-small int8 ONNX + linear head, about 6 ms/vector on the Pi. About 95% right move, 70% of table talk silent (D34).
- Output is a band (`sure`/`unsure`/`silent`) plus one or two moves (`spoken/interpret.py:38`).
- FEAT-39: small local LLMs failed to stay silent on ambient chat (Qwen 3.5 4B: 71% right move, 95% silent). Silence on chatter is the metric that matters.

## OpenAI Decisions API (official docs: developers.openai.com/api/docs/guides/decisions)
- `POST /v1/decisions`, SDK `client.decisions.create` (Python SDK 3.26.0+). Model `gpt-6-luna` only. Public beta, GA "in the coming weeks".
- $0.10/1M input tokens, no output or cache charges.
- Request: `input` (text or user messages) + `questions[]` of type `predicate`, `choice` (`choices[{value,description}]`) or `score` (`levels[{label,description}]`).
- Response: `answers[]`. Predicate gives `probability`. Choice gives `choice`, `probabilities[]`, `confidence`. Docs advise an `other` fallback choice.
- Community (anecdotal): `choice` was badly miscalibrated and order-dependent (70/30 coin came back about 98% heads); `predicate` matched. Test both, with shuffled option order.
- ZDR available to eligible customers. Latency claims: up to 10x faster than Luna; anecdotal 0.3 to 0.8 s.

## Microsoft Decision-1 (secondary sources only; openrouter.ai/microsoft/microsoft-decision-1 returned 404 for the researcher, and `/api/v1/models` listed no "decision" model)
- Reported: Qwen3.5-9B fine-tune, about $0.042/1M input, free output, model id `microsoft/microsoft-decision-1`, released 2026-10-09.
- Endpoint reports conflict: OpenRouter `/api/alpha/decisions` vs `/providers/microsoft/v1/systemone`. Verify on the real OpenRouter docs first.
- Same three question types (one source says "noul" instead of predicate; likely a garble).
- Accuracy and 35x-speed claims are Microsoft's own. One independent test measured 459 ms.
- Unknown: whether weights are released (would make it a local-model candidate).

## Integration cost if adopted
- `compatible` provider is Chat Completions (`models.py:300`); neither API uses it. Needs a new provider kind in `models.py`, a key pin per host (`config.py:109`, `LLMCORD_KEY_HOSTS`), fake-transport tests, and an e5 fallback when the API is down or slow.
- Privacy: ambient chat leaves the Pi. Needs an opt-in decision (consent invariant) before any production use.

## Benchmark plan (no production wiring)
1. Reuse the FEAT-39 harness and fixtures: blackjack moves, table talk, Korean, and the I Doubt It doubt set.
2. Candidates: Decisions API (choice with `other`, and per-move predicates), Decision-1, Haiku 5.5, Gemini Flash-Lite. e5 as baseline.
3. Metrics: right-move rate, silent-on-chatter rate, Korean split, p50/p95 latency, cost per 1,000 calls, `unsure` band usefulness (do probabilities separate sure from unsure?).
4. Keys come from the environment only (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`); never print them or write them to the repo.
5. Likely role if it wins: tiebreaker behind e5 for the `unsure` band, not a replacement.
