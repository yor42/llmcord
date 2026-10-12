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

## Microsoft Decision-1 (first-hand check, 2026-10-12; supersedes the secondary-source notes)
Checked against OpenRouter's own pages and Microsoft's announcement and Learn docs. The earlier 404 does not reproduce: the model page now loads.
- **Model:** `microsoft/microsoft-decision-1` on OpenRouter ([model page](https://openrouter.ai/microsoft/microsoft-decision-1)), released 2026-10-09, post-trained from Qwen3.5-9B, context 32,768, $0.042/1M input and $0 output. One provider (Azure), page shows P50 latency 0.22 s. Microsoft says it will later rebase it on MAI and OpenAI models, and "weights are updated continually while the API shape stays the same", so scores and thresholds can drift under us.
- **It is not in `/api/v1/models`** (458 models listed, none "decision"), which is why the remote session did not find it. It is a separate router.
- **Endpoints, both real, for different hosts:**
  - OpenRouter: `POST https://openrouter.ai/api/alpha/decisions` (OpenAPI page "Submit a Decisions request", Python SDK `open_router.alpha.decisions.create`). An unauthenticated POST gets 401, other guesses get 404. Bearer `OPENROUTER_API_KEY`. We use this one.
  - Azure Foundry: `<resource-endpoint>/providers/microsoft/v1/systemone` ([Learn](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision)). Needs an Azure subscription and deployment; not used here. The OpenRouter OpenAPI also has a `SystemOne` tag for the same wire shape ("models such as Jev").
- **Wire format (differs from OpenAI's):** request `{model, state, questions}` where `state` is a string or JSON and `questions` is an object keyed by question name. Types: `noul` (yes/no; "noul" is the real name, not a garble), `choice` (`criteria` is a map option to description), `score` (`criteria` is an ordered list). Response `answers[name]`: `noul` gives `{noul: p}`, `choice` gives `{choice, confidence, probabilities{option: p}}`, `score` gives `{score, confidence, legend, probabilities}`; plus `id`, `model`, `provider`, `usage{cost,input_tokens,output_tokens}`. Several questions can share one request.
- **Weights are not released.** Hosted only (Foundry and OpenRouter); no Hugging Face repo found. It cannot become a local-model candidate. Open-weight decision models exist (secondary sources name Cloudflare Clef and Amazon Strands Decider 2B; not verified, out of scope).
- Microsoft's docs advise an abstention option (`cannot tell`), randomizing option order and testing order sensitivity, and warn that "scores can change based on how you phrase or order questions". Same advice as for OpenAI's `choice`.
- Accuracy and speed claims are Microsoft's own; no published benchmarks that we could verify.
- Another model on the same OpenRouter router is `typesafe/jev-1.13` (the doc example). Not in scope unless the user asks.

## Comparison of the two wire formats
| | OpenAI Decisions | OpenRouter Decisions (Decision-1) |
| --- | --- | --- |
| URL | `https://api.openai.com/v1/decisions` | `https://openrouter.ai/api/alpha/decisions` |
| Input field | `input` | `state` |
| Questions | array with `name`, `type`, `instructions` | object keyed by name, with `type`, `instructions`, `criteria` |
| Yes/no type | `predicate` (`probability`) | `noul` (`noul`) |
| Choice options | `choices[{value, description}]` | `criteria{option: description}` |
| Choice result | `choice`, `probabilities[{value, probability}]`, `confidence` | `choice`, `probabilities{option: p}`, `confidence` |
| Price per 1M input | $0.10 (output free) | $0.042 (output free) |

## Integration cost if adopted
- `compatible` provider is Chat Completions (`models.py:300`); neither API uses it. Needs a new provider kind in `models.py`, a key pin per host (`config.py:109`, `LLMCORD_KEY_HOSTS`), fake-transport tests, and an e5 fallback when the API is down or slow.
- Privacy: ambient chat leaves the Pi. Needs an opt-in decision (consent invariant) before any production use.

## Benchmark plan (no production wiring)
1. Reuse the FEAT-39 harness and fixtures: blackjack moves, table talk, Korean, and the I Doubt It doubt set.
2. Candidates: Decisions API (choice with `other`, and per-move predicates), Decision-1, Haiku 5.5, Gemini Flash-Lite. e5 as baseline.
3. Metrics: right-move rate, silent-on-chatter rate, Korean split, p50/p95 latency, cost per 1,000 calls, `unsure` band usefulness (do probabilities separate sure from unsure?).
4. Keys come from the environment only (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`); never print them or write them to the repo.
5. Likely role if it wins: tiebreaker behind e5 for the `unsure` band, not a replacement.
