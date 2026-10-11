# Small multilingual models for classifying chat into game actions

Date: 2026-10-11. Scope: English + Korean casual Discord messages ("한 장 더", "ㄱㄱ", "그만", "콜") mapped to a small closed, state-dependent set of legal actions plus "none / table talk", on CPU (Pi 5 8GB; Jetson Nano / TX2 considered).
Legend: [V] verified from a primary source (URL in Sources); [E] my own estimate or inference; [U] unverified.

## 1. Summary and recommendation

- **Pi 5 CPU is sufficient. Do not use the Jetson Nano / TX2.** JetPack 4.x ships Python 3.6 and CUDA 10.2. NVIDIA only provides a Python 3.6 onnxruntime wheel (1.11.0, per a forum moderator), and a Python 3.6 / protobuf conflict was reported [V, forum, secondary]. The classifier only runs for the one seated player whose turn it is, on messages of about 10 tokens, so GPU is unnecessary.
- **Primary recommendation: exemplar nearest-neighbour (kNN) over an int8 ONNX `intfloat/multilingual-e5-small`** (MIT, 118M params, 384-dim, official `onnx/` folder including `model_qint8_avx512_vnni.onnx`) [V]. Candidate alternative: `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, same 118M, official `model_qint8_arm64.onnx`) [V]. I could not find a head-to-head Korean comparison of the two in primary sources. Run both on a small labelled Korean set and pick (see section 4).
- Nearest-neighbour lets the label set be filtered per game state (only legal actions compete), allows adding phrases from the dashboard without retraining, and gives an open-set reject for free (low similarity means silent or "table talk").
- Do not use zero-shot NLI (mDeBERTa-xnli) or a small LLM as layer 2. Both are heavier and weaker on calibration; keep the LLM API as layer 3.
- Do not use jina-embeddings-v3 (CC-BY-NC-4.0) [V]; EmbeddingGemma needs the Gemma license and gated download [V], and is bigger.
- Calibration: use top-1 cosine plus top1-minus-top2 margin (margin computed across *distinct actions*, not distinct exemplars), fit thresholds on logged button outcomes. Details in section 3.

## 2. Candidate comparison

Params are from the Hugging Face API safetensors totals [V] unless noted. fp32 disk is roughly 4 bytes x params [E]. For multilingual-e5-small, about 96M of the 118M parameters are the 250k-token embedding matrix [E, from XLM-R vocab size 250,002 x 384], so per-token compute is that of a roughly 21M-parameter transformer. This is why it is cheap on CPU despite the "118M" label.

| Model | Params | License | Korean evidence | ONNX / int8 | Notes |
|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | 117.7M, 12 layers, 384-d [V] | MIT [V] | Mr.TyDi ko MRR@10 54.3 (retrieval; base 55.8, large 61.6) [V]; MTEB MassiveIntent ko accuracy 63.79 [V]; STS17 ko-ko cos Spearman 78.87 [V] | Official `onnx/model.onnx`, `model_O4.onnx`, `model_qint8_avx512_vnni.onnx`, plus openvino [V]. No arm64 int8 file; make one with Optimum (`arm64` preset) [V that the preset exists] | Needs `query: ` prefix on every input, even non-English, "for tasks other than retrieval ... `query: `" [V]. Trained for retrieval; short slang may be weaker [E] |
| intfloat/multilingual-e5-base | 278.0M [V] | MIT [V] | Mr.TyDi ko 55.8 [V] (barely above small) | Same set of ONNX files as small [V] | About 2.4x params, small gain on that benchmark; only if small fails |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 117.7M, 384-d, max seq 128 [V] | Apache-2.0 [V] | Card says 50+ languages without naming them [V partially]; no Korean number found in primary sources [U] | Official `onnx/model_qint8_arm64.onnx`, `_avx512`, `_avx512_vnni`, O1-O4 [V] | Paraphrase-trained, so short-sentence similarity is its design target [E]. Best "ready-made ARM int8" option |
| sentence-transformers/LaBSE | 470.9M [V] | Apache-2.0 [V] | 109 languages [V]; designed for translation-pair matching, not intent similarity [E] | `onnx/model.onnx` only [V] | Larger and not obviously better for this; skip |
| BAAI/bge-m3 | about 568M (same size as its Korean fine-tunes KURE-v1 / BGE-m3-ko) [V via API] | MIT [V] | MIRACL results shown only as an image on the card; no numeric Korean score read [U] | `onnx/model.onnx` + external data [V] | Too large for a hot path on Pi (about 2.3GB fp32) [E]; sensible only as an offline labeller |
| jhgan/ko-sroberta-multitask | 110.6M [V] | Not stated on the card (API shows none) [V]: treat as unlicensed | KorSTS cosine Pearson 84.77 / Spearman 85.60 [V] | Official `onnx/model.onnx`, `_O4`, `_qint8_avx512_vnni` [V] | Korean only (RoBERTa, Korean vocab); English handling [U]. Licence ambiguity is a blocker for shipping |
| BM-K/KoSimCSE-roberta-multitask | 110.6M [V] | Not stated in API [V] | Not fetched [U] | n/a | Same caveats; Korean only |
| nlpai-lab/KURE-v1, dragonkue/BGE-m3-ko | 567.8M [V] | MIT / Apache-2.0 [V] | Korean retrieval fine-tunes of bge-m3; scores not checked [U] | n/a | Too large |
| google/embeddinggemma-300m | 302.9M [V], 768-d, MRL to 128 [V] | Gemma terms, gated download [V] | MTEB multilingual v2 mean 61.15 (not Korean-specific) [V]; QAT Q4_0 60.62, Q8_0 60.93 [V] | QAT checkpoints Q4_0 / Q8_0 [V] | Strong, on-device oriented, but licence friction and 2.5x the compute of e5-small [E] |
| Snowflake/snowflake-arctic-embed-m-v2.0 | 305.4M [V] | Apache-2.0 [V] | Not checked [U] | Not checked [U] | Possible alternative; not evaluated |
| Qwen/Qwen3-Embedding-0.6B | 595.8M [V] | Apache-2.0 [V] | Not checked [U] | n/a | Too large |
| jinaai/jina-embeddings-v3 | 572.3M [V] | CC-BY-NC-4.0 [V] | n/a | n/a | Non-commercial: exclude |
| MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (zero-shot NLI) | about 0.3B [V] | MIT [V] | **Korean is not among XNLI languages and the card reports no Korean score**; card says out-of-XNLI languages are "most likely lower" [V] | n/a | One forward pass per (message, label) pair, so N labels means N passes [E]. Not recommended |
| Qwen/Qwen2.5-0.5B-Instruct (constrained output) | 494.0M [V] | Apache-2.0 [V] | Not verified for slang Korean [U] | GGUF Q4_K_M exists (bartowski) [V] | Pi 5 speeds found: 8 to 19.4 tok/s for qwen2.5:0.5b via Ollama, quantization unstated, sources conflict [V secondary]. Prompt processing of a rules-plus-state prompt is the real cost [E]. Qwen3-0.6B (751.6M, Apache-2.0) similar [V] |

### Latency and memory on Pi 5

- No primary-source or third-party Pi 5 benchmark of any of these embedders was found [U]. Searches returned none.
- Back-of-envelope [E]: e5-small / MiniLM per-token compute is about 21M parameters, so a 16-token message is about 0.7 GFLOP. A Pi 5 (4x Cortex-A76 with dot-product instructions) should do this in tens of milliseconds with int8 ONNX Runtime at 2 to 4 threads. This is a guess; measure it.
- ONNX Runtime docs say Arm processors with dot-product instructions generally perform better with quantization, and the U8S8 saturation problem is x86 AVX2/AVX512-specific ("no such issue ... Arm") [V]. They recommend dynamic quantization for transformers [V].
- Do not use the `*_avx512_vnni.onnx` files on the Pi (x86-only instruction preset) [E]; produce an `arm64` file with sentence-transformers' `export_dynamic_quantized_onnx_model(..., "arm64")` [V that the preset exists; speedup figure not retrieved, U]. The MiniLM repo already ships one.
- RAM: fp32 weights about 470MB, int8 weights likely 120 to 250MB depending on whether the embedding matrix is quantized (dynamic quantization may leave the Gather embedding in fp32) [E, U]. Either fits easily in 8GB next to the bot and dashboard.
- The embedding table dominates size. If RAM or load time matters, you can prune the vocabulary to tokens seen in Korean/English text [E, not evaluated].
- First-call latency: load once at bot start, keep the session resident; the bot is a single-event-loop process, so run inference in a thread executor (CLAUDE.md notes all sqlite is sync on the loop; ONNX calls would add to that) [E].

### Jetson Nano / TX2

- Jetson Nano and TX2 are stuck on JetPack 4.x (CUDA 10.2, Python 3.6 base) [V via NVIDIA forum threads]. Current onnxruntime, sentence-transformers and transformers need newer Python. A forum thread reports a user claiming CUDA 10.2 support ended at onnxruntime 1.6 [U]; NVIDIA staff mention a cp36 wheel for 1.11.0; another user built 1.10.0 from source [V, secondary].
- Conclusion: realistic only as a separate Python 3.6 sidecar with pinned old libraries, which is not worth it. Pi 5 CPU suffices [E].

## 3. Approaches

### A. Exemplar nearest-neighbour (recommended first)
- Store 10 to 40 example phrases per action (stored per game, editable from the dashboard), embed once, cache. At runtime embed the message, cosine-score against exemplars of **only the currently legal actions** plus a "table talk" exemplar set, aggregate per action (max, or mean of top-k).
- Pros: no training, instantly editable, state filtering is trivial (drop illegal actions before scoring), open-set rejection is a threshold, runs in sub-millisecond after the embed, easy to log and explain.
- Cons: needs good exemplars for slang; "table talk" is an unbounded class, so use it as a *negative exemplar set* (real logged chatter) plus the threshold, not as one more class.
- Embedding caveat: paraphrase/retrieval embedders put "stand" and "hit" closer than intent warrants because both are blackjack vocabulary [E]. Include contrastive exemplars (e.g. "더 안 받아" for stand versus "한 장 더" for hit) and check confusion pairs.

### B. SetFit few-shot fine-tuning
- SetFit fine-tunes a Sentence Transformer contrastively on few labelled pairs, then trains a classification head, "without prompts"; multilingual use means swapping in a multilingual body [V: arXiv 2209.11055, HF blog]. The paper's multilingual experiments used paraphrase-multilingual-mpnet-base-v2 on MARC; the MiniLM-L12 pairing appears only in community cards [V, so no peer-reviewed number for it].
- Pros: better separation than raw kNN once you have a few hundred real labelled messages (the button log is exactly that).
- Cons: closed-set softmax head has no native "none" (add a table-talk class trained on real chatter, or threshold the probability); a fixed head cannot honour per-state legal sets except by masking logits and renormalizing; every label-set change needs retraining; training on-Pi is slow, so train off-box [E].
- Suggested path: ship A, log outcomes, then train SetFit offline on the logged data if A's accuracy plateaus.

### C. Zero-shot NLI
- mDeBERTa xnli: Korean unsupported in XNLI evaluation and no reported score [V]; N forward passes per message. Rejected.

### D. Tiny LLM with constrained output
- Qwen2.5-0.5B / Qwen3-0.6B with grammar-constrained decoding over the legal actions. Pros: handles context ("one more" after a hand), arbitrary phrasing. Cons: prompt processing cost on Pi (unmeasured here [U]); Korean slang quality unverified [U]; confidence needs token log-probs, whose calibration is poor for small models [E]. Keep the existing API LLM as rare layer 3 instead.

### State-dependence and open-set rejection
- Treat legality as a mask applied *before* ranking: if only [hit, stand] are legal, "double" or "split" phrases cannot win, which also raises effective margin.
- A message can be both an action and table talk ("콜? 아직 모르겠다"). The prompt design (always offering [Just table talk]) already covers this, so the model only has to rank, not decide.
- Rely on the word-list layer first for unambiguous tokens ("ㄱㄱ", "콜", "stand"). Note: single-jamo or very short tokens ("ㄱㄱ") are poorly represented by subword embedders [E]; the word-list should own them, and exemplars should be matched on normalized text.

## 4. Calibration of the three bands

All of this is standard practice [E] unless a source is cited; no primary source was gathered for these specifics.

1. **Scores.** For each legal action a, s(a) = max cosine over its exemplars. Also compute s(talk) against negative exemplars. Features: s1 (top action), margin m = s1 - s2 where s2 is the best *different* action's score, and s1 - s(talk).
2. **Bands.** High = s1 >= t_hi and m >= m_hi (single-action prompt). Medium = s1 >= t_lo and (m < m_hi or s1 < t_hi) (two-action prompt, top two actions). Low = else, silent. Use *both* statistics: similarity alone is high for in-domain-looking talk, margin alone is high for far-away junk.
3. **Do not compare raw cosine across models.** E5 cosines are compressed into a narrow high range (the card says to expect scores in roughly 0.7 to 1.0 [V: e5 card notes on temperature 0.01]); thresholds must be fit per model. Re-fit if the model or exemplar set changes.
4. **Temperature / Platt scaling.** Convert scores into probabilities with softmax(s/T) over legal actions plus a talk logit, fit T (and a bias for talk) by minimizing log loss on logged outcomes. Platt/isotonic on s1 and m is the lightweight alternative; with fewer than about 200 labels prefer a 1- or 2-parameter fit over isotonic [E].
5. **Labels from button presses.** Chosen action = positive for that action; "Just table talk" = negative (all actions wrong); ignored prompt = weak/unlabelled (do not use as negative, since it conflates "not interested" with "wrong"). Silent (low band) messages that were later followed by the player pressing a real button manually are valuable false-negative labels. Caveat: selection bias, since you only see labels for messages that already scored high enough to prompt; occasionally (e.g. 5%) shadow-log scores for silent messages and, if consented per the project's privacy rules, label them offline.
6. **Tuning targets.** Choose t_hi so the precision of the single-action prompt is at least about 90% (the player rarely has to press "table talk"); choose t_lo for medium-band recall of the true action within the top two at about 90%; accept more silence. Re-tune monthly; keep thresholds in a settings table.
7. **Privacy.** Logged chat text is user data. Per CLAUDE.md's consent invariant, store only scores, predicted/chosen action and a hash or nothing of the text unless the opt-in rules allow it [E, project policy to be checked].

## 5. Prototype first (in order)

1. Offline harness: collect about 150 hand-written EN/KO/mixed messages (actions + talk) per game; evaluate kNN with e5-small (with `query: `), MiniLM-L12, and ko-sroberta (research only, license) on top-1 accuracy, top-2 recall, and talk-rejection AUROC. Choose the embedder with that data, not the Mr.TyDi/Massive numbers.
2. Convert chosen model to int8 `arm64` ONNX; measure p50/p95 latency and RSS on the actual Pi 5 at 1, 2, 4 threads (batch 1, 8 to 20 tokens).
3. Implement state-masked kNN with margin-based bands and a decision log table (scores, band, prompt shown, button pressed).
4. After a few hundred logged presses, fit T / thresholds; consider SetFit offline if kNN confusion pairs persist.
5. Only if needed: compare with multilingual-e5-base, and benchmark Qwen2.5-0.5B GGUF prompt-processing time as a layer-2.5.

## 6. Open questions

- No Korean-slang evaluation for any candidate exists; every Korean number above is on formal-text benchmarks (KorSTS, Mr.TyDi, MASSIVE).
- Pi 5 latency for these embedders: not found, measure.
- MTEB leaderboard Korean-task numbers for e5-small vs MiniLM-L12 were not retrieved [U]; the leaderboard is a JS app and was not scraped.
- ko-sroberta / KoSimCSE license is not stated on the cards; unresolved.
- Whether dynamic int8 shrinks the 96M-parameter embedding table: unverified.
- How the planned Texas Hold'em adds actions ("콜", "레이즈", "올인"): check that "콜" does not collide with "call a friend/other" chatter; solved by masking and by the word list.

## Sources

- multilingual-e5-small model card (languages, prefix rule, Mr.TyDi table, MTEB metrics, ONNX files): https://huggingface.co/intfloat/multilingual-e5-small , raw: https://huggingface.co/intfloat/multilingual-e5-small/raw/main/README.md
- multilingual-e5-base: https://huggingface.co/intfloat/multilingual-e5-base
- Model metadata via HF API (params, license, file lists): https://huggingface.co/api/models/<id> for each model listed above
- paraphrase-multilingual-MiniLM-L12-v2: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- LaBSE: https://huggingface.co/sentence-transformers/LaBSE
- bge-m3: https://huggingface.co/BAAI/bge-m3
- ko-sroberta-multitask: https://huggingface.co/jhgan/ko-sroberta-multitask
- mDeBERTa-v3-base-mnli-xnli: https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli
- EmbeddingGemma: https://huggingface.co/google/embeddinggemma-300m
- Qwen2.5-0.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct ; GGUF: https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF
- SetFit paper: https://arxiv.org/abs/2209.11055 ; blog: https://huggingface.co/blog/setfit
- ONNX Runtime quantization docs: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- Sentence Transformers efficiency (ONNX, quantization presets): https://sbert.net/docs/sentence_transformer/usage/efficiency.html
- KorNLI/KorSTS paper: https://aclanthology.org/2020.findings-emnlp.39
- Pi 5 LLM speed (secondary): https://www.dfrobot.com/blog-15784.html , https://benchmarklist.com/benchmarks/edge_llm_raspberry_pi_5/
- Jetson Nano / JetPack 4.6 onnxruntime (secondary, NVIDIA forum): https://forums.developer.nvidia.com/t/can-jetson-nano-jetpack-4-6-install-onnxruntime-1-10-can-use-gpu-if-yes-please-help-me-thanks/256870 , https://forums.developer.nvidia.com/t/jetson-nano-fails-to-install-pre-built-onnxruntime/259049
