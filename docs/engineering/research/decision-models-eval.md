# Decision models for spoken-move interpretation (research note, 2026-10-12)

Status: benchmark done (FEAT-43, 2026-10-12, results and recommendation at the end). Nothing is wired into the bot. Using any result in the bot needs the user's opt-in decision (new provider wire format, and sending player chat to a third party).

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

## Harness design (FEAT-43)
Code: `prototypes/decision_eval/` (`evalcore.py` pure helpers, `run_eval.py` CLI). Tests: `tests/test_decision_eval.py` (offline, fake transport). Nothing under `llmcord_core/` changes.

### Items
Reuse the labelled sets, no new phrases and no real chat:
- **bj**: all 195 rows of `prototypes/spoken_moves/phrases.csv` (133 move rows, 62 table talk; en 71, ko 101, mixed 23). Each row has `legal` moves; pass rule as `bands.passes` (move row: answer is the label or an accepted move; talk row: `talk`, or an accepted move).
- **doubt**: the 116 phrases of `tests/test_spoken_doubt.py` (60 doubt, 56 not), parsed with `ast` as in `prototypes/local_director/spoken_llm.py`. Language by script: any Hangul and any ASCII letter is `mix`, Hangul only is `ko`, else `en`.
- Move meanings (`MEANING`) and the two system texts are the ones in `spoken_llm.py`, so the numbers sit next to the Qwen 3.5 4B round and the recorded e5 control (95% right move, 70% silent; D34).

### Candidates (names are CLI ids)
| id | Endpoint | Question shape | Reading |
| --- | --- | --- | --- |
| `oai-choice` | OpenAI `POST /v1/decisions`, model `gpt-6-luna` | one `choice` question: each legal move plus `other` ("table talk, a question, a joke, or not an instruction to act now"); option order shuffled, K orders per item | argmax; `other` is `talk` |
| `oai-predicate` | same | one request, one `predicate` question per legal move ("Is the player choosing to <meaning> right now?") | argmax probability if >= tau, else `talk` |
| `or-d1-choice` | OpenRouter `POST /api/alpha/decisions`, model `microsoft/microsoft-decision-1` | same as `oai-choice` in the OpenRouter wire format (`state`, `criteria` map) | as above |
| `or-d1-noul` | same | one `noul` question per legal move in one request | as `oai-predicate` |
| `or-haiku` | OpenRouter `POST /api/v1/chat/completions`, `anthropic/claude-haiku-5.5` | JSON-schema enum of legal moves plus `talk`, same system text as the Qwen round, temperature 0 | the enum |
| `or-flashlite` | same, `google/gemini-3.5-flash-lite` | same | the enum |
For doubt items the choice options are `doubt` and `other`; the predicate/noul question is "Is the player accusing another player of lying or bluffing right now?". Auth: `OPENAI_API_KEY` for `oai-*`, `OPENROUTER_API_KEY` for the rest; read from the environment only, never printed, logged, stored in results or put in a URL.

### Metrics (per candidate and task)
Right-move rate, silent-on-chatter rate, both split by en/ko/mix, invalid/refusal/error counts, p50/p95 wall latency per request (sequential, so it is not skewed by our own concurrency), cost per 1,000 calls from the provider's `usage` (else tokens x price), order stability for the shuffled choice runs (share of items whose answer is the same across the K orders, and the spread of the top probability), and probability separation: the top probability against correctness, as an AUROC, a table of accuracy by probability bucket (<0.5, 0.5-0.7, 0.7-0.9, 0.9-0.99, >=0.99), and a tau sweep for the predicate/noul and choice modes (move-right and talk-silent if an answer under tau counts as silent).
The e5 control is the recorded D34 number. e5 per-row scores are not reproduced here (needs the ONNX model), so "behind e5" is judged from the recorded numbers and the separation tables, not row by row.

### Safety rules for the script
Dry run by default (prints call count and a cost estimate, makes no request and needs no key); a paid run needs `--run`. A `--max-usd` budget aborts a run once the reported usage cost passes it. Results go to `--out` (default outside the repo), never into the repo; they hold phrases, ids and answers, no headers. Each model is smoke-tested with a few calls before the full run.

## Results (FEAT-43 full run, 2026-10-12)
3,110 calls, 0 errors, $0.080 spent (estimate was $0.166). Items: the 195 blackjack rows (133 moves, 62 table talk) and 116 doubt phrases. Raw per-call results stayed outside the repo (`~/decision-eval-out`). Reproduce: `prototypes/decision_eval/README.md`.

How to read: every decision candidate gives one answer per item; shuffled `choice` candidates are asked in 3 option orders and the majority answer counts. "Right move" is the share of the 133 move rows where the answer is the label or an accepted move. "Silent" is the share of the 62 talk rows answered `talk` (strict). One answer per item is not the same as e5's "right move among up to two buttons", so the e5 row is only a rough yardstick.

### Blackjack moves
| Candidate | Right move | Silent on chatter | Right move ko | Silent ko | p50 / p95 latency | Cost per 1,000 decisions |
| --- | --- | --- | --- | --- | --- | --- |
| e5 int8 + head (D34, recorded, up to 2 buttons) | about 95% | about 70% | n/a | n/a | tens of ms on the Pi | $0 |
| Qwen 3.5 4B local (FEAT-39) | 71% | 95% | n/a | n/a | 0.66 s (remote GPU) | $0 |
| OpenAI `choice` + `other` | 81% (108/133) | 100% (62/62) | 82% | 100% | 0.29 / 0.37 s | about $0.10 (3 calls; estimated) |
| **OpenAI per-move `predicate`** | **97% (129/133)** | 97% (60/62) | 97% | 100% | 0.27 / 0.35 s | about $0.026 (estimated) |
| Decision-1 `choice` + `other` | 93% (124/133) | 100% (62/62) | 90% | 100% | 0.68 / 0.77 s | $0.024 (3 calls) |
| **Decision-1 per-move `noul`** | **98% (130/133)** | **100% (62/62)** | 96% | 100% | 0.71 / 0.81 s | **$0.007** |
| Haiku 5.5 (reasoning off) | 94% (125/133) | 100% (62/62) | 96% | 100% | 1.36 / 2.23 s | $0.046 |
| Gemini 3.5 Flash-Lite | 97% (129/133) | 98% (61/62) | 99% | 97% | 1.04 / 1.23 s | $0.084 |

Costs come from provider usage where given. OpenAI's Decisions response carries no cost field, so its costs are estimates from the $0.10/M input price. Latency is per request from this Windows machine over the public internet, with a fresh connection per call.

### I Doubt It doubts (116 phrases)
| Candidate | Doubt recall | Silent on non-doubts | Silent ko |
| --- | --- | --- | --- |
| `read_doubt` word rule (recorded) | 100% (60/60) | 100% (56/56) | 100% |
| OpenAI `choice` | 92% | 86% | 70% |
| OpenAI `predicate` | 100% | 77% | 67% |
| Decision-1 `choice` | 78% | 88% | 81% |
| Decision-1 `noul` | 95% | 77% | 74% |
| Haiku 5.5 | 97% | 82% | 67% |
| Flash-Lite | 90% | 86% | 74% |

### Findings
1. **Per-move yes/no questions beat one `choice` question.** For OpenAI the gap is large (97% vs 81% right move): its `choice` answer leans toward `other` on clear moves and its probabilities are under-confident (the 0.9-0.99 bucket was only 92% right). Decision-1 `choice` is better (93%) but still below its own `noul` mode (98%). This agrees with the community reports that `choice` is the weaker mode. Order dependence turned out small: 96% of bj items (98% of doubts) gave the same OpenAI `choice` answer in all 3 option orders, Decision-1 100% (99%); the top probability moved by 0.07 and 0.01 on average.
2. **Decision-1 `noul` per move is the best single reader here**: 98% right, 100% silent, $0.007 per 1,000 messages. Its probabilities separate the bands cleanly. Top move probability >= 0.9: 121 rows, all moves. From 0.5 to 0.9: 9 rows, all moves. Below 0.5: 65 rows, of which 62 are chatter and 3 are moves (the only 3 misses). So `sure` >= 0.9, `unsure` 0.5-0.9, `silent` < 0.5 would have had no wrong `sure` on this set. OpenAI `predicate` is almost as good (AUROC 0.997 vs 1.000) but its band edges are looser: the 0.7-0.9 bucket is 94% moves, and two chatter rows crossed ("wait 잠깐" and "insurance is a scam lol").
3. **The decision models, Haiku and Flash-Lite make mostly different mistakes.** Decision-1 `noul` and Haiku never miss the same move (3 + 8 misses, 11 distinct rows); Decision-1 `noul` and OpenAI `predicate` share 1 of 6. No move was missed by all six candidates. The six hard rows (missed by 3 or more candidates) are short or elliptical stands and slang: "됐어요", "I'm happy with this", "나 이걸로 할게", "노노 됐어", plus "send it" and "ㅎㅌ". A cascade of two readers is cheap and useful; one weak reader is not.
4. **Korean was not worse than English** for the `noul` and `predicate` modes on blackjack (96-97% right, 100% silent). On doubts, Korean silence was the weakest for every LLM (67-74%).
5. **Every LLM reader loses to the word rule on doubts.** None got above 88% silent on non-doubts (negations such as "never lies" and "구라 아니야", and lexical false friends). Keep `read_doubt` as is; do not use any of these for doubts.
6. **Baselines (D37):** Haiku 5.5 and Gemini 3.5 Flash-Lite are both good (94-97% right move, 98-100% silent), far better than the local Qwen 3.5 4B (71% right) and about as good as the best decision mode. D37's expectation that a cheap hosted model beats a local one is confirmed on quality. Their cost is 6 to 12 times Decision-1's but still only $0.05-0.08 per 1,000 messages. They are slower (1.0-1.4 s p50, p95 up to 2.2 s) and give no probabilities, so there is no `sure`/`unsure` split.
7. **Speed:** OpenAI Decisions answers in about 0.27 s (p95 0.35 s). Decision-1 through OpenRouter took 0.7 s (OpenRouter's page quotes 0.22 s at the provider; the rest is routing plus our fresh connection per call). Both Decisions APIs are faster than the chat baselines.

### Recommendation
- **Do not replace e5.** The recorded e5 numbers (95% right / 70% silent) belong to a local, free, offline, private reader, and nothing here is a reason to send every message off the Pi.
- **Best fit: a second reader behind e5, only for messages e5 did not call silent.** e5 is silent on only about 70% of chatter, so about 30% of chatter wrongly gets a move check. A per-move yes/no reader (Decision-1 `noul`, or OpenAI `predicate`) was 97-100% silent on chatter on this set and costs well under a cent per 1,000 messages. It could veto most false checks and choose between the two buttons of an unsure check. Only messages e5 would act on go out, which keeps volume and exposure small. A missed real move only costs the player one button tap.
- **Ranking:** (1) Decision-1 `noul` per move: best numbers, cheapest, cleanest probabilities. Risks: unversioned weights ("updated continually"), an `/api/alpha` endpoint, one upstream provider. (2) OpenAI `predicate` per move: fastest (0.27 s) and almost as good. Risks: public beta, a single model. Test both for drift before relying on a threshold. (3) Haiku or Flash-Lite: they run on the existing Chat Completions path (`compatible` provider, no new wire format) but give no probabilities.
- **Not recommended:** `choice` questions on either Decisions API, any LLM for doubts, or reading every message with an LLM.

### Caveats
- n = 133 moves + 62 chatter rows (a 95% interval is roughly +-3 to 4 points). The phrases are synthetic, written for the e5 work. Real channel chatter is more varied, so real false-positive rates need checking on kept phrases (D35) before any threshold is trusted.
- e5's per-row scores were not reproduced here (it needs the ONNX model), so the cascade is argued from the recorded e5 numbers and the LLM separation tables, not simulated row by row. The honest next step is to dump e5's per-row bands from `prototypes/spoken_moves/run.py` and replay the cascade on the same rows.
- Prompts were fixed once after a 6-item smoke test. The OpenAI inputs initially lacked context and scored 0/3 on moves (now one shared context sentence for all decision candidates). Haiku 5.5 needed reasoning switched off, because otherwise it spends its token budget on hidden reasoning and returns nothing. The full set was not used for tuning. The chat baselines' system prompts carry slightly less context than the decision candidates' framing.
- One run and one sample per call. The Decisions APIs have no temperature setting, and the shuffled `choice` majority hides within-item variation. Decision-1 weights can change under a stable API shape.

### Integration and privacy if adopted (not started)
- It needs a new provider kind in `models.py` for the Decisions wire formats (both differ from Chat Completions), key pins for `api.openai.com` and `openrouter.ai` (`config.py:109`, `LLMCORD_KEY_HOSTS`), fake-transport tests, and an e5-only fallback when the API is down, slow (budget about 1 s) or over a per-server rate.
- Messages from the member whose turn it is would leave the Pi. That needs an explicit per-server opt-in, separate from `/memory opt_in`, shown in the Games tab and in `/games spoken_moves`, with the chosen provider's data-retention and training terms checked first.

## Cascade replay (FEAT-44, 2026-10-12)
No new paid calls. The shipped local reader was run on the same 195 blackjack rows (`prototypes/decision_eval/cascade.py dump-e5`: int8 e5, shipped head, no server examples, as `games_spoken_discord._read` does; model hashes matched the pins), and the recorded FEAT-43 answers were replayed behind it (`cascade.py replay`). The second reader only sees rows e5 did not call silent and can only drop or reorder e5's moves.

| Reader and policy | Right move | Silent on chatter | Sure wrong (moves / chatter) | Ko moves right | Asked (share of all messages) | Cost per 1,000 messages |
| --- | --- | --- | --- | --- | --- | --- |
| e5 alone | 93% (124/133) | 81% (50/62) | 5 / 10 | 66/71 | 0 | $0 |
| Decision-1 `noul`, tau 0.5 | 90% (120/133) | 100% (62/62) | 0 / 0 | 63/71 | 72% | $0.005 |
| Decision-1 `noul`, tau 0.7 | 88% (117/133) | 100% | 0 / 0 | 60/71 | 72% | $0.005 |
| Decision-1 `noul`, tau 0.9 | 86% (114/133) | 100% | 0 / 0 | 57/71 | 72% | $0.005 |
| OpenAI `predicate`, tau 0.5 | 89% (119/133) | 98% (61/62) | 2 / 1 | 63/71 | 72% | about $0.019 (estimate) |
| OpenAI `predicate`, tau 0.7 | 87% (116/133) | 100% | 2 / 0 | 61/71 | 72% | about $0.019 (estimate) |
| Haiku 5.5 (answer read as 1.0 / 0.0) | 87% (116/133) | 100% | 0 / 0 | 62/71 | 72% | $0.033 |

Asking only when e5 says `unsure` changes at most one row (not shown above): e5 called just 1 of 195 rows `unsure` (140 `sure`, 54 `silent`), so there is no tiebreaker work for a second reader on this set.

### What it shows
1. **The shipped e5 alone, measured here: 93% right move and 81% silent on chatter**, with 12 chatter rows let through, every one of them in the `sure` band ("hit me up later", "that song is a hit", "what does double do?", "gg", "스탠드 조명 샀어", "insurance is a scam lol"). The recorded D34 figures (95% / 70%) were for up to two buttons and used the FEAT-36 setup, so these are not directly comparable. The two fresh numbers are the ones to use.
2. **A second reader does what it was proposed for.** Decision-1 `noul` at tau 0.5 silences all 12 leaked chatter rows and leaves no wrong `sure` band. Every reader and tau silenced 11 of them; "wait 잠깐" is the one that OpenAI at tau 0.5 let through.
3. **It is paid for with recall.** Decision-1 at 0.5 drops 4 real moves (bj29, 45, 60, 158) of 124 that e5 had right; higher taus drop more (7 at 0.7, 10 at 0.9). OpenAI `predicate` drops 5 at 0.5 and 8 at 0.7 (18 at 0.9, not in the table above). Haiku drops 8 at any tau. A dropped move costs the player a tap on the normal button.
4. **Decision-1 `noul` at tau 0.5 is the best trade here**: 12 chatter rows that e5 gave a button (10 of them a wrong button) removed for 4 lost moves, at about half a cent per 1,000 messages (calls only for the 72% of messages e5 did not call silent). The tau values above were chosen on these same 195 rows, so none is validated. This also shows the earlier "unsure band" idea does not apply: the reader works as a veto on `sure`, not as a tiebreaker.
5. **Korean move rows right** go from 66/71 (e5 alone) to 63/71 with Decision-1 at 0.5, so 3 of its 4 lost moves are Korean; the set has 71 Korean and 47 English move rows, so this is a small-number signal, not a finding.
6. **The cascade cannot raise recall.** It never rescues a message e5 called silent (e5 called 4 move rows silent and gave 5 more the wrong move) and never turns `unsure` into `sure`.

### Caveats
- The e5 head trains on `examples.json` / `talk_examples.json` and `phrases.csv` is its labelled test set, but the same author wrote both, so the e5 numbers are optimistic (near-duplicates exist). The decision models are zero-shot on the same rows.
- 195 synthetic rows; checking against real kept phrases (D35) was not done and cannot be done here because those live under the production data, which this work does not read. That check needs the owner to run `dump-e5`/`replay`-style scoring on their own data, or to supply an anonymised set.
- The replay uses answers recorded once in FEAT-43; Decision-1 weights can change under a stable API shape, so thresholds must be re-checked before relying on them.
- Haiku gives no probabilities, so its tau rows are identical.

### Recommendation (updated)
- A cheap per-move yes/no reader (Decision-1 `noul` first, OpenAI `predicate` second) behind e5 is worth a production design decision **only as a veto on `sure` and `unsure` readings**, and only if false move checks on real chat turn out to matter as much as here (about 1 in 5 chatter lines got a button from e5 alone). Expected effect on this set: 81% → 100% silent on chatter for 3 points of right-move rate.
- It is not a replacement for e5, not a way to improve recall, and not for doubts.
- Production use still needs your opt-in decision (player chat leaves the Pi), a new provider kind in `models.py`, key pins, a per-server switch separate from `/memory opt_in`, and an e5-only fallback. I have not started any of it.
