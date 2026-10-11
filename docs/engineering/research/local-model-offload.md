# Offloading llmcord's small LLM calls (director and memory) to local models or deterministic logic

Date: 2026-10-11. Scope: the "director" call (picks 1–3 speaker character IDs from a transcript, structured JSON; `llmcord_core/engine.py` ~line 165, `DIRECTOR_SCHEMA`) and the "memory" role (summaries and personal-fact extraction). Hardware: Jetson Nano 4GB, Jetson TX2 8GB, Raspberry Pi 5. Companion to `docs/engineering/research/nl-game-actions-models.md`, which covers the Jetson JetPack 4 / Python 3.6 constraints and the e5 embedding classifier. This note does not repeat those; it corrects and extends them where noted.

Legend: [V] verified from a primary source (URL in Sources); [E] my estimate or inference; [U] unverified or no primary source found.

## 1. Summary and recommendation

- **Cost is not a reason to offload.** Real spend (coordinator data, all on `gemini-3.8-flash` at list rates). The bot has been live under a week with 3 active members, so this is a small sample and not a monthly figure. Per call: director about $0.00045 ($0.0009 from 2027); memory about $0.0015; dialogue about $0.0034. Role shares: dialogue 53%, memory 41%, director 6% (49 director calls, 109 memory, 61 dialogue; total about $0.39 over the period) [user-supplied, not independently verified]. Self-hosting only pays when its fixed cost is covered by director calls (section 5.2).
- **Jetson Nano and TX2: do not use.** JetPack 4.6.6 is the final JetPack 4 release and security patches stop after EOL [V]. The llama.cpp CUDA 10.2 path is community-only and unreliable [secondary]. A local director on the Pi is possible but slow, and the Korean quality of sub-1B models is unproven.
- **Pi 5 CPU is the only plausible local host, and only for privacy or offline resilience.** A 0.5–1.5B model can produce a 1–3 speaker choice in seconds, not milliseconds (section 3). The Korean evidence for these sizes is thin (section 4).
- **The largest lever is the memory role, not the director:** 41% of spend and 109 calls against 49 director calls. It is still only about $0.16 per month.
- **Recommendation:** keep the director on the existing hosted model. Use the deterministic ambient gate that already exists (section 6). Do not build a local model for cost. Revisit local only if privacy (transcripts leaving the box) or offline behaviour becomes a requirement, and then benchmark first.

## 2. Real call profile (corrects the brief)

The brief assumed about 2k input and 50 output tokens per director call. Real figures from the coordinator's data are about 560 input and 7 output tokens per director call (27,582 / 49 = 563 in; 327 / 49 = 6.7 out) [user-supplied]. Memory averages about 816 input and 233 output per call (88,957 / 109; 25,400 / 109). Dialogue averages about 3,800 input and 150 output per call (232,587 / 61; 9,244 / 61).

Both the 560-token and 2k-token cases are priced in section 5 so the conclusion holds either way.

## 3. Hardware (question 1)

### 3.1 Jetson Nano 4GB and TX2 8GB
- JetPack 4.6.6 is "the final release for JetPack 4 series"; after EOL "you will no longer receive any critical updates" [V, NVIDIA forum EOL announcement]. The TX2 is listed in the same table [V].
- llama.cpp on CUDA 10.2 / gcc 8: a community gist reports a build with nvcc 10.2 and gcc 8.5, then GPU offload crashing with `GGML_ASSERT: ggml-cuda.cu ... CUDA error` at `--n-gpu-layers 1`; the author also reports Ollama running CPU-only on the Nano [secondary, gist]. A fork (LLabsTech/Llama.cpp) claims JetPack 4.6.1 support; not verified [U].
- The llama.cpp build docs state no minimum CUDA toolkit or compiler version [V, github.com/ggml-org/llama.cpp docs/build.md]. So there is no primary statement that current llama.cpp supports CUDA 10.2. Treat it as unsupported [E].
- Power: Jetson Nano 4GB configurable 5 W or 10 W [secondary, retailer and blog pages; NVIDIA datasheet not fetched]. TX2 power modes not confirmed [U].
- Ollama: no JetPack 4 support found in primary sources [U].
- **Conclusion:** no. The sibling note's Python 3.6 finding stands, and the EOL status adds a maintenance reason on top.

### 3.2 Raspberry Pi 5 (Cortex-A76, 8GB) CPU
Benchmarks come from one secondary table (benchmarklist.com, llama_cpp backend, Pi 5) [secondary]. The page calls the columns "Prefill" and "Decode" and does not state units. I read them as tokens per second [E], but this is unverified [U].

| Model, quant | Prefill (tok/s, units unstated) | Decode (tok/s) |
|---|---|---|
| Qwen2.5-0.5B Q4_K_M | 52.9 | 22.3 |
| Qwen2.5-0.5B Q4_0_4_4 | 189.9 | 26.7 |
| Qwen2.5-1.5B Q4_K_M | 22.3 | 9.6 |
| Qwen2.5-1.5B Q4_0_4_4 | 65.9 | 10.1 |
| Qwen2.5-3B Q4_K_M | 10.6 | 4.8 |
| Llama-3.2-1B Q4_K_M | 30.5 | 11.5 |
| SmolLM2-1.7B Q4_K_M | 17.7 | 8.5 |

- Latency for one director call (about 560 input, about 7 output) on these numbers [E]: Qwen2.5-1.5B Q4_K_M about 25 s prefill plus under 1 s decode; Qwen2.5-0.5B Q4_0_4_4 about 3 s; Qwen2.5-3B Q4_K_M about 55 s. These are too slow for chat unless they run in the background.
- Q4_0_4_4 is an ARM-specific quantization format [V? not checked; the table uses it, so it is listed]; use it only with a build that supports it.
- The sibling note's "Pi 5 8 to 19.4 tok/s for qwen2.5:0.5b via Ollama" is not contradicted by this table, but both are secondary [U]. Benchmarking on the actual Pi is still needed.
- Pi 5 power: about 2.7–3.6 W idle and about 8.8 W with four cores loaded, from third-party measurements [secondary]. The Raspberry Pi site itself was not fetched. A 27 W official PSU is recommended [secondary].
- RAM: 8GB is enough for a 0.5–1.5B Q4 model next to the bot [E].

### 3.3 Electricity and effort (rough)
- Extra electricity for a Pi that already runs the bot 24/7 is close to zero: director bursts are about 49 calls per month, each a few seconds of CPU [E].
- If the Pi were a dedicated inference box at about 8.8 W continuous, that is about 6.4 kWh per 30 days, or roughly KRW 1,000–1,300 at an assumed KRW 150–200 per kWh [E; tariff assumed, not verified [U]]. Idle at about 3 W gives about 2.2 kWh per 30 days.
- Effort: model download, quantization choice, llama.cpp build, latency tuning, Korean quality testing, and ongoing OS and llama.cpp updates on the Pi [E].

## 4. Models (question 2)

Korean evidence from vendor model cards and benchmark tables (not Korean-slang tests):

| Model | Korean evidence found | License | Notes |
|---|---|---|---|
| EXAONE 4.0 1.2B | KMMLU-Pro 42.7 (reasoning) / 37.5 (non-reasoning); KMMLU-Redux 46.9 / 40.4; KSM 60.6 / 26.3 [V] | EXAONE AI Model License Agreement 1.2 - NC [V] | Korean is one of three stated languages [V] |
| EXAONE 3.5 2.4B | KoMT-Bench 7.24; LogicKor 8.51 [V] | EXAONE AI Model License Agreement 1.1 - NC [V] | Research only; commercial use prohibited [V, LICENSE summary]; the model name must start with "EXAONE" for derivatives [V] |
| Qwen2.5 / Qwen3 (0.5–4B) | No Korean number found in the cards [U]. Qwen3-4B card says "100+ languages and dialects" with no list; Korean not named [V] | Apache-2.0 [V] | Korean ability unknown from primary sources |
| Gemma 3 1B / 4B | Korean not named; "over 140 languages" [V]. 4B PT: Global-MMLU-Lite 57.0, MGSM 34.7 (not Korean-specific) [V] | Gemma terms (gated download, Google usage license) [V] | Use-restriction terms need a separate read before any redistribution [U] |
| Llama 3.2 1–3B, Phi-3.5/4-mini | Not fetched [U] | Llama community licence / MIT [U] | Not checked this pass |

- **Correction to the sibling note:** it said the Pi speed for Qwen2.5-0.5B conflicts across sources and is secondary. That still holds. Its licence notes for e5 and the Jetson note are not affected.
- The task needs Korean reading, grammar and JSON-constrained output. llama.cpp supports GBNF and JSON-schema grammar constraints [U, not fetched this pass; check the llama.cpp grammar docs before relying on it].
- **Pick for local:** EXAONE 4.0 1.2B is the only one with Korean numbers, but its NC licence bars commercial use [V]. Private, non-commercial use of this bot is probably fine, but that is a judgment for the user and not a legal opinion [E]. Qwen2.5-0.5B/1.5B is Apache-2.0 but with no Korean evidence [U].

### 4.1 Non-LLM speaker selection
- The director already has a deterministic ambient gate in `engine.py` (lines ~152–160): it skips the model call for ambient messages with no cast-name mention, no invitation phrase, and no favourite. This covers the common case without any model [V, code].
- Name and mention matching for non-ambient turns is cheap and reliable; the director is only needed when several characters are plausible [E].
- Embedding similarity (e5 kNN, per the sibling note) could rank characters by message similarity to their persona. It needs per-character exemplars and is only worth it if deterministic rules miss too often [E]. Latency is in the low tens of milliseconds on Pi (sibling note's estimate, unmeasured [U]).

## 5. Hosted baseline (question 3)

All prices are USD per 1M tokens, standard tier, from vendor pricing pages. Checked 2026-10-11.

| Option | Input | Output | Source and date | JSON schema / structured output |
|---|---|---|---|---|
| Claude Haiku 5.5 (prompts up to 100k) | $0.10 | $0.50 | platform.claude.com pricing [V] | Not verified this pass [U] |
| Claude Haiku 4.5 | $1.00 | $5.00 | same [V] | [U] |
| Gemini 3.8 Flash (current usage) | $0.75 through 2026-12-31; $1.50 from 2027-01-01 | $3.75 ($7.50 from 2027) | ai.google.dev pricing, page updated 2026-10-09 [V] | [U] |
| Gemini 3.5 Flash-Lite | $0.30 | $2.50 | same [V] | [U] |
| Gemini 3.1 Flash-Lite | $0.25 | $1.50 | same [V] | [U] |
| Gemini 2.5 Flash-Lite | $0.10 | $0.40 | same [V] | [U] |
| OpenAI gpt-5-nano | $0.05 (cached $0.005) | $0.40 | developers.openai.com pricing, no date on page [V] | [U] |
| OpenAI gpt-4.1-nano | $0.10 | $0.40 | same [V] | [U] |
| OpenAI gpt-5-mini | $0.25 | $2.00 | same [V] | [U] |
| DeepSeek deepseek-flash (cache miss) | $0.30 peak, $0.15 off-peak | $1.20 peak, $0.60 off-peak | api-docs.deepseek.com pricing [V]; cache hit $0.006 | [U] |
| Groq, Cerebras, OpenRouter | Not fetched (Groq's page is a marketing homepage with no prices) [U] | | | [U] |

Korean ability of these hosted models was not checked in primary sources [U]. Their Korean quality is likely better than any sub-4B local model [E], and this is the strongest argument for hosted over local.

Structured output support: the pricing pages do not state it. Verifying it means reading each provider's structured-output docs, which I did not do in this pass [U].

### 5.1 Worked example: cost per 1,000 director calls

Per call, using the USD per 1M token rates above:

| Option | 560 in / 7 out (real) | 2k in / 50 out (brief) |
|---|---|---|
| Gemini 3.8 Flash ($0.75/$3.75) | $0.45 | $1.69 |
| Gemini 3.8 Flash from 2027 ($1.50/$7.50) | $0.90 | $3.38 |
| Gemini 3.1 Flash-Lite ($0.25/$1.50) | $0.15 | $0.58 |
| Gemini 2.5 Flash-Lite ($0.10/$0.40) | $0.06 | $0.22 |
| Claude Haiku 5.5 ($0.10/$0.50) | $0.06 | $0.23 |
| Claude Haiku 4.5 ($1/$5) | $0.60 | $2.25 |
| gpt-5-nano ($0.05/$0.40) | $0.03 | $0.12 |
| gpt-5-mini ($0.25/$2) | $0.15 | $0.60 |
| DeepSeek deepseek-flash, peak ($0.30/$1.20) | $0.18 | $0.66 |

Check: the real 49 director calls at $0.00045 each come to about $0.022 over the under-one-week period, consistent with the coordinator's total [consistent].

### 5.2 Break-even: director calls per month needed to cover self-hosting

Break-even calls per month = monthly self-hosting cost / hosted cost per call. Hosted cost per call is $0.00045 today ($0.0009 from 2027, Gemini 3.8 Flash).

- Electricity for a Pi that already runs the bot: effectively zero extra [E]. Break-even on electricity is 0 calls.
- Electricity for a dedicated Pi at 8.8 W continuous, about KRW 1,000–1,300 per month (about $0.70–0.90 at an assumed KRW 1,400 per USD [U]): about 1,500–2,000 calls per month to match hosted cost.
- Add upkeep: break-even = (electricity + upkeep value) / $0.00045. Example [E, illustrative]: 1 hour per month of maintenance (OS and llama.cpp updates, model swaps) valued at $20 adds about 44,000 calls per month.
- Current volume: 49 director calls in under a week, so roughly 200 per 30 days if the rate holds [E; too small a sample to rely on]. Far below any break-even above.

Conclusion: self-hosting the director does not pay at this volume, even before upkeep.

## 6. Recommendation for llmcord

Ranked, for cost and fit:

1. **Keep the director on the current hosted model, and rely on the existing deterministic gate.** Savings from any alternative are at most a few cents per 100 calls at this volume. Quality matters more than price here. Do nothing unless quality is a problem.
2. **If you want cheaper hosted calls later:** a cheaper hosted model (Gemini 3.1 Flash-Lite at $0.15 per 1,000 calls, or Claude Haiku 5.5 at $0.06 per 1,000 calls) cuts the per-call director cost by roughly 3–7x. Test Korean speaker selection before switching. The 2027 price doubling for Gemini 3.8 Flash only doubles a small per-call cost, so it does not justify a change by itself.
3. **Memory role, if you want a real lever:** it is 41% of spend, about $0.0015 per call (109 calls in the sample). Options: batch summaries less often, reduce summary input, or use a cheaper hosted model. Any local model for personal-fact extraction must respect the consent invariant (CLAUDE.md: personal facts only after `/memory opt_in`) and needs Korean quality testing first [E].
4. **Local model on the Pi 5, only for non-cost reasons:**
   - Privacy: the director sees recent transcript text, and sending that to a vendor may conflict with the user's expectations. Only the user can decide this.
   - Offline resilience: the bot keeps choosing speakers when the API is down. Note that the current code already falls back to the first cast member on director failure (`engine.py` exception branch) [V, code].
   - Rate limits and latency: hosted calls add network latency; the local 1.5B estimate (about 25 s prefill) is worse, so a local model is not a latency win [E].
   - If pursued, use a 0.5–1.5B Q4 model on the Pi with grammar-constrained output, benchmark Korean on about 100 real transcripts, and benchmark latency on the Pi itself.
5. **Jetson Nano / TX2: reject.**

**Non-cost reasons someone might still offload:** privacy of chat content, offline resilience, avoiding rate limits and provider outages. None of these is quantified here.

## 7. Open questions for the user

- Is privacy of chat transcripts a requirement? If not, offloading has no meaningful benefit.
- Is offline or API-outage resilience a requirement for the director?
- Which hosted provider and model is actually used for the director today, and is any Korean quality issue observed? (The coordinator's data says all roles use `gemini-3.8-flash`.)
- Does the user accept EXAONE's NC licence for a local model if privacy becomes a requirement?
- Is the 2027 Gemini price increase planned for?
- Should memory summaries run less often, given they are the largest slice of spend?

## Sources

- Gemini API pricing (page updated 2026-10-09 UTC): https://ai.google.dev/gemini-api/docs/pricing
- Claude API pricing (includes Haiku 5.5 and Haiku 4.5 rows; checked 2026-10-11): https://platform.claude.com/docs/en/about-claude/pricing
- OpenAI API pricing (gpt-5-nano, gpt-5-mini, gpt-4.1-nano; no date shown): https://developers.openai.com/api/docs/pricing
- DeepSeek API pricing (deepseek-flash, deepseek-v4-pro; peak and off-peak): https://api-docs.deepseek.com/quick_start/pricing
- Groq pricing: https://groq.com/pricing (marketing homepage only, no prices retrieved)
- NVIDIA JetPack 4 end of life (JetPack 4.6.6 final): https://forums.developer.nvidia.com/t/announcing-end-of-life-for-nvidia-jetpack-4-with-the-release-of-jetpack-4-6-6/314302
- llama.cpp build docs (no minimum CUDA version stated): https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- llama.cpp on Jetson Nano CUDA 10.2 (community gist, unreliable): https://gist.github.com/kreier ; fork: https://github.com/LLabsTech/Llama.cpp
- Pi 5 llama.cpp prefill/decode table (secondary, units unstated): https://benchmarklist.com/benchmarks/edge_llm_raspberry_pi_5/
- Pi 5 power (secondary): https://raspberry.tips/en/?p=9083
- Jetson Nano power modes (secondary): none authoritative found; NVIDIA datasheet not fetched [U]
- EXAONE 3.5 2.4B model card (KoMT-Bench, LogicKor, NC licence): https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
- EXAONE 3.5 licence text summary: https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct/raw/main/LICENSE
- EXAONE 4.0 1.2B model card (KMMLU-Pro, KMMLU-Redux, KSM, NC licence 1.2): https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B
- Qwen3-4B model card (Apache-2.0, 100+ languages, Korean not named): https://huggingface.co/Qwen/Qwen3-4B
- Qwen3-1.7B model card: https://huggingface.co/Qwen/Qwen3-1.7B
- Gemma 3 1B model card (multilingual table, licence): https://huggingface.co/google/gemma-3-1b-it
- Gemma 3 4B model card (PT multilingual table): https://huggingface.co/google/gemma-3-4b-it
- Sibling note: `docs/engineering/research/nl-game-actions-models.md`
- Director code: `llmcord_core/engine.py` (ambient gate and failure fallback around lines 150–185)
