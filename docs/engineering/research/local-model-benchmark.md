# Local small-model director benchmark (FEAT-39, decision D36)

Date: 2026-10-12. Benchmark only; no llmcord code was changed. Follows `local-model-offload.md`. Legend: **[M]** measured on this machine in this run, **[E]** estimate or inference.

## 1. Setup

- Machine [M]: Raspberry Pi 5, **4 GB RAM** (the offload note assumed 8 GB), 4 cores, about 2.4 GB MemAvailable with production running. Benchmark used 3 threads, context 2048, one slot, `cache_prompt: false` (every call prefills the full prompt), temperature 0, `max_tokens` 64.
- Runtime: llama.cpp **b11559** (commit f2cc7282c, "0.6.0-dev"), prebuilt `llama-b11559-bin-ubuntu-arm64.tar.gz` from github.com/ggml-org/llama.cpp, sha256 `9a6d241056ae96ba72dda763b09a27eec38bbf005f25a5a9904ae2985bbc1f23` (matches the release digest). CPU only. This build ships several ARM CPU backends (armv8.0 to armv9.2) and picks one at run time; I did not check which one was chosen [E].
- Models (official Qwen GGUF repos, pinned revisions):

| Model file | HF repo @ revision | Size | sha256 |
|---|---|---|---|
| qwen2.5-0.5b-instruct-q4_k_m.gguf | Qwen/Qwen2.5-0.5B-Instruct-GGUF @ 9217f5db79a29953eb74d5343926648285ec7e67 | 491 MB | 74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db |
| qwen2.5-0.5b-instruct-q4_0.gguf | same | 429 MB | 7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed |
| qwen2.5-1.5b-instruct-q4_k_m.gguf | Qwen/Qwen2.5-1.5B-Instruct-GGUF @ 91cad51170dc346986eccefdc2dd33a9da36ead9 | 705 MB | 6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e |

  3B was skipped (RAM). Weights and runtime lived on the SD card (`~/feat39-bench-data`, since deleted), not in the 2 GB RAM-backed /tmp, to keep them out of RAM.
- Prompt [M]: system = the `director` preset `main` text from `prompts.py:116` + "Return only the required structured result. Output schema: " + `DIRECTOR_SCHEMA` JSON (as `engine.structured_contract`); user = the JSON payload of `engine.speakers` (`cast`, `recent` last 6, `latest`, `latest_author`, `ambient`, `forced`). Median **262 prompt tokens** and 15 output tokens (the hosted call profile in the offload note is about 560 in / 7 out, so these prompts are shorter and decode longer).
- Constrained run: llama-server `response_format` `json_schema` (strict). Unconstrained run: same prompt, no schema. One warm-up call, then 30 calls per mode per model.
- Scoring: model output is filtered to cast ids and an empty non-ambient answer falls back to the first cast member, exactly as `engine.speakers` does. "Exact" = set equals expected. "Contains addressed" = expected speakers are all in the answer (only for cases with a clear expected speaker: 24 of 30; the 6 ambient cases are scored by silence instead). "Ambient silence" = answer is empty for the 6 ambient chit-chat cases. Case 15 ("Good morning everyone!") has no single right answer, so it counts as exact for any non-empty answer.
- Thermal [M]: the Pi reached **83 C and was soft-throttling during every run** (`get_throttled` 0xe0008 for the 0.5B runs, 0xf0008 for 1.5B, meaning the soft temperature limit was active at the end and frequency capping, throttling and the soft limit had occurred; 0xf0008 adds that under-voltage had occurred, so the 1.5B timings may also be affected by supply voltage). Production containers were also running. I waited for under 65 C before each model, but the first 30 seconds of each run is the only cool part. **All latencies below are therefore pessimistic**, probably by a modest factor with a good heatsink and fan [E].
- Memory guard: MemAvailable stayed above 600 MB (checked before every call). Peak RSS (VmHWM, includes mmap'd model pages) was 733 / 805 / 1947 MB; the 1.5B run exceeded the 1.6 GB target.

## 2. Cases

30 synthetic cases (15 EN, 10 KO, 5 mixed), fictional characters and users only, expected answers written before any run (`cases.py`). Casts of 2 to 5; kinds: addr (clearly addressed), multi (two named), follow (follow-up to a character's last line, character not named in the latest message), amb (ambient chit-chat, expected []), invite (ambient with an explicit invitation). The recent-turns context is in `cases.py` and omitted here. Answers shown are the constrained run, after engine filtering. INVALID means the output was cut off at 64 tokens in a repeating id loop, so it was not valid JSON.

| # | lang | kind | cast | latest message (recent turns omitted) | expected | 0.5B Q4_K_M | 0.5B Q4_0 | 1.5B Q4_K_M |
|---|---|---|---|---|---|---|---|---|
| 1 | en | addr | Mira,Bram | Mira, can you help me pick a gift for my sister? | Mira | Mira,Bram | INVALID | Mira,Bram |
| 2 | en | addr | Mira,Bram,Luna | Bram what did you buy today? | Bram | Mira,Bram,Luna | INVALID | Mira,Bram,Luna |
| 3 | en | addr | Luna,Kai | Hey Kai, you still owe me a rematch. | Kai | Luna,Kai | Luna,Kai | Luna,Kai |
| 4 | en | multi | Mira,Bram,Luna | Mira and Luna, settle this for us: pineapple on pizza, yes or no? | Mira,Luna | Mira,Bram,Luna | Mira,Bram,Luna | Mira,Bram,Luna |
| 5 | en | multi | Mira,Bram,Luna,Kai | Bram and Kai, race you to the lighthouse! | Bram,Kai | Mira,Bram,Luna,Kai | Mira,Bram,Luna,Kai | Mira,Luna,Kai |
| 6 | en | follow | Mira,Bram | Yes he did, right? | Bram | Mira,Bram | INVALID | Mira,Bram |
| 7 | en | follow | Luna,Kai,Otto | Why did it collapse? | Otto | Luna,Kai,Otto | INVALID | Luna,Kai,Otto |
| 8 | en | follow | Mira,Kai | Haha nice. And the food? | Kai | Mira,Kai | INVALID | Kai,Mira |
| 9 | en | amb | Mira,Bram | did anyone watch the game last night | [] | Mira,Bram | INVALID | Mira,Bram |
| 10 | en | amb | Mira,Bram,Luna | my cat knocked over a plant again | [] | Mira,Bram,Luna | INVALID | Mira,Bram,Luna |
| 11 | en | amb | Kai,Otto | we should plan a trip next month | [] | Kai,Otto | Kai,Otto | Kai,Otto |
| 12 | en | invite | Mira,Luna | Luna, come over and settle this for us! | Luna | Mira,Luna | INVALID | Luna,Mira |
| 13 | en | invite | Bram,Kai,Otto | Otto, your turn, give us a riddle. | Otto | Bram,Kai,Otto | INVALID | Kai,Bram,Otto |
| 14 | en | addr | Mira,Bram,Luna,Kai,Otto | Luna, what is the capital of Iceland? | Luna | Bram,Luna,Kai,Otto | INVALID | Luna,Mira,Kai,Otto |
| 15 | en | open | Mira,Bram | Good morning everyone! | any one | Mira,Bram | INVALID | Mira,Bram |
| 16 | ko | addr | 하늘,민준 | 하늘아, 오늘 뭐 했어? | 하늘 | 하늘,민준 | 하늘,민준 | 하늘,민준 |
| 17 | ko | addr | 하늘,민준,서연 | 민준아 같이 산책하자 | 민준 | 하늘,민준,서연 | INVALID | 민준,서연,하늘 |
| 18 | ko | addr | 서연,도윤 | 도윤 씨, 이 문제 좀 도와주세요. | 도윤 | 서연,도윤 | 서연,도윤 | 서연,도윤 |
| 19 | ko | multi | 하늘,민준,서연 | 하늘이랑 서연이는 어떻게 생각해? | 하늘,서연 | 하늘,민준,서연 | 하늘,민준,서연 | 하늘,민준,서연 |
| 20 | ko | follow | 하늘,민준 | 그래서 하늘이는 왜 안 갔대? | 하늘 | 하늘,민준 | 하늘,민준 | 하늘,민준 |
| 21 | ko | follow | 서연,도윤,지우 | 오 대박 몇 점 나왔어? | 지우 | 서연,도윤,지우 | 서연,도윤,지우 | 서연,도윤,지우 |
| 22 | ko | amb | 하늘,민준 | 오늘 점심 뭐 먹지 | [] | 하늘,민준 | INVALID | 하늘,민준 |
| 23 | ko | amb | 서연,도윤,지우 | 주말에 비 온대 | [] | 서연,도윤,지우 | 서연,도윤,지우 | 서연,도윤,지우 |
| 24 | ko | invite | 하늘,서연 | 서연아 와서 좀 봐줘! | 서연 | 하늘,서연 | INVALID | 서연,하늘 |
| 25 | ko | addr | 하늘,민준,서연,도윤 | 도윤아 너는 어디야? | 도윤 | 하늘,민준,서연,도윤 | 하늘,민준,서연,도윤 | 하늘,민준,서연 |
| 26 | mix | addr | Mira,하늘 | Mira, 오늘 저녁 뭐 먹을까? | Mira | Mira,하늘 | INVALID | Mira,하늘 |
| 27 | mix | addr | Bram,민준,Luna | 민준아, can you tell Bram what happened? | 민준 | Bram,민준,Luna | INVALID | 민준,Bram,Luna |
| 28 | mix | multi | Mira,하늘,Kai | Kai랑 하늘이, 둘이 같이 노래 한 곡 불러줘 | Kai,하늘 | Mira,하늘,Kai | Mira,하늘,Kai | Mira,하늘,Kai |
| 29 | mix | follow | Luna,서연 | Luna, yes please say something! | Luna | Luna,서연 | INVALID | Luna,서연 |
| 30 | mix | amb | Mira,하늘 | 아 진짜 졸리다 so tired | [] | Mira,하늘 | INVALID | Mira,하늘 |


## 3. Numbers

### Constrained (json_schema) [M]

| metric | 0.5B Q4_K_M | 0.5B Q4_0 | 1.5B Q4_K_M |
|---|---|---|---|
| prompt tokens (median) | 262 | 262 | 262 |
| output tokens (median) | 15 | 64 | 15 |
| prefill ms (median) | 4516 | 1050 | 8239 |
| decode ms (median) | 603 | 2061 | 1551 |
| wall s per call (median) | 5.12 | 2.91 | 9.84 |
| wall s per call (p90) | 6.18 | 3.29 | 14.82 |
| valid JSON with speakers array | 100% | 40% | 100% |
| exact match all | 1/30 | 0/30 | 1/30 |
| exact match en | 1/15 | 0/15 | 1/15 |
| exact match ko | 0/10 | 0/10 | 0/10 |
| exact match mix | 0/5 | 0/5 | 0/5 |
| contains addressed all | 24/24 | 10/24 | 22/24 |
| contains addressed en | 12/12 | 3/12 | 11/12 |
| contains addressed ko | 8/8 | 6/8 | 7/8 |
| contains addressed mix | 4/4 | 1/4 | 4/4 |
| ambient silence all | 0/6 | 0/6 | 0/6 |
| ambient silence en | 0/3 | 0/3 | 0/3 |
| ambient silence ko | 0/2 | 0/2 | 0/2 |
| ambient silence mix | 0/1 | 0/1 | 0/1 |

### Unconstrained [M]

| metric | 0.5B Q4_K_M | 0.5B Q4_0 | 1.5B Q4_K_M |
|---|---|---|---|
| prompt tokens (median) | 262 | 262 | 262 |
| output tokens (median) | 15 | 36 | 15 |
| prefill ms (median) | 4547 | 1085 | 7396 |
| decode ms (median) | 601 | 1281 | 1419 |
| wall s per call (median) | 5.19 | 2.41 | 8.86 |
| wall s per call (p90) | 6.07 | 2.64 | 10.46 |
| valid JSON with speakers array | 100% | 10% | 100% |
| exact match all | 1/30 | 0/30 | 1/30 |
| exact match en | 1/15 | 0/15 | 1/15 |
| exact match ko | 0/10 | 0/10 | 0/10 |
| exact match mix | 0/5 | 0/5 | 0/5 |
| contains addressed all | 24/24 | 3/24 | 22/24 |
| contains addressed en | 12/12 | 1/12 | 11/12 |
| contains addressed ko | 8/8 | 1/8 | 7/8 |
| contains addressed mix | 4/4 | 1/4 | 4/4 |
| ambient silence all | 0/6 | 0/6 | 0/6 |
| ambient silence en | 0/3 | 0/3 | 0/3 |
| ambient silence ko | 0/2 | 0/2 | 0/2 |
| ambient silence mix | 0/1 | 0/1 | 0/1 |

### Load and memory [M]

| | 0.5B Q4_K_M | 0.5B Q4_0 | 1.5B Q4_K_M |
|---|---|---|---|
| load s | 16.4 | 6.5 | 24.1 |
| peak RSS MB (VmHWM) | 733 | 805 | 1947 |
| temp / throttled at end | 83.4'C / 0xe0008 | 82.9'C / 0xe0008 | 83.4'C / 0xf0008 |


Prefill speed from the medians: 0.5B Q4_K_M about 58 tokens/s, 1.5B Q4_K_M about 32 tokens/s, 0.5B Q4_0 about 250 tokens/s [M, derived, throttled]. The Q4_0 speedup is consistent with ARM weight repacking (run-time repack of Q4_0, which replaces the old Q4_0_4_4 files in the offload note) [E, not verified in the server log].

## 4. Findings

- **Quality is the blocker, not speed.** Every Q4_K_M answer was valid JSON (unconstrained too), but the models almost always return the whole cast: 29 of 30 answers for 0.5B and 27 of 30 for 1.5B. The apparent "contains addressed" rates (24/24 and 22/24) are mostly that artifact; exact match is 1/30 for both (the single hit is case 15, which accepts any non-empty answer, so the real count is 0/29). Ambient silence is **0/6 for all three** models, so the "choose no speaker" rule is not followed at all. 1.5B drops a character in a few cases and misses the addressed one in 2 of 24 (case 5 omits Bram, case 25 omits Doyun/도윤 and answers 하늘, 민준, 서연), but never gets a case exactly right.
- **Korean:** no better or worse than English here, because neither works. The addressed Korean speaker was present in 7/8 (1.5B) and 8/8 (0.5B) only because the whole cast was returned. Korean addressing is **not reliable** with these models and this prompt [M].
- **Constrained decoding matters only for Q4_0.** Q4_K_M never needed it. Q4_0 0.5B degenerated into endless id lists (valid 40% constrained, 10% unconstrained, and unconstrained it echoes the schema), so Q4_0 is unusable here even though it is 2 to 4 times faster [M].
- **Latency vs the offload note [M, throttled]:** one call with a 262-token prompt took a median 5.1 s (0.5B Q4_K_M, p90 6.2 s) and 9.8 s (1.5B Q4_K_M, p90 14.8 s), plus 6 to 24 s model load. The note's estimates (about 3 s and about 25 s for 560 tokens, section 3.2) are in the same range once the shorter prompt is accounted for. A local director does fit in "several seconds" for 0.5B and "about 10 s" for 1.5B, but it does not fit in a good chat experience, and it would also compete with the production containers for 4 GB of RAM (the 1.5B run left about 1.4 GB available).
- **Conclusion against the research note:** the note's recommendation stands. Keep the hosted director plus the deterministic ambient gate in `engine.py`. A 0.5 to 1.5B Qwen2.5 director, as prompted today, would behave like "everyone speaks every turn". Making a local model work would need a different prompt (few-shot examples, or a name-matching pre-pass with the model only breaking ties) and a larger or Korean-trained model (EXAONE 4.0 1.2B has an NC licence; Qwen3 or a 3B+ would not fit the 4 GB box) [E, untested]. Not worth pursuing for cost; revisit only for a privacy or offline requirement.
- Limits of this run: one prompt format, temperature 0, one seed, 30 cases, a throttled CPU, no few-shot tuning. Hard-to-parse effects (e.g. Qwen chat template, system-prompt weight) were not explored.

## 5. Appendix: scripts and raw results

Scripts are in `prototypes/local_director/` (they expect the runtime and models under `~/feat39-bench-data/`; raw results were not kept in the repo)

- `cases.py` (the 30 cases and expected sets), `bench.py` (starts llama-server, runs the calls, records timings and RSS), `runall.sh` (waits for under 65 C between models), `score.py` (scoring), `tables.md` (tables used here).
- `results/` is created by `bench.py` (raw per-call text and timings, server logs); not committed.
- Downloaded GGUF files and the runtime were deleted after the run.

## Round 2: Qwen 3.5 4B on Ollama (remote)

Run 2026-10-12. Everything below is **[M] measured** unless marked **[E] estimated**. Scripts: `prototypes/local_director/` (`ollama_bench.py` director + client, `memory_cases.py`, `spoken_llm.py`, `stub_ollama.py` offline dry-run server). Raw per-call results were written to a scratchpad directory, not the repo.

### Setup
- Server: Ollama 0.40.3 on the user's other device (reached over the tailnet; set `OLLAMA_URL` to run the scripts), model tag `qwen3.5:4b` (4.2B, Q4_K_M, 3.3 GB file, capabilities completion/vision/tools/thinking).
- `/api/ps` after the director run: `size` 8.64 GB, `size_vram` 5.52 GB, context 131072. So the model was **not fully in VRAM**; about 3.1 GB (the remainder, likely the large KV cache for the 131k context) sat in system RAM. The GPU model is unknown; the latency below is for this unknown machine, **not** the Pi.
- Every call: `POST /api/chat`, `stream:false`, `format:<JSON schema>` (omitted for the free-text summary), `options {temperature:0, seed:1}`, `think:false` (main runs). One warm-up call first. Per-call counters recorded: wall, `prompt_eval_count/duration`, `eval_count/duration`, `load_duration`.
- Prompts are the real ones: the director system prompt and payload shape of round 1 (30 cases), the `summary` and `extraction` presets from `llmcord_core/prompts.py` with `MEMORY_SCHEMA`.

### 1. Director (30 cases, think off)

| metric | 0.5B Q4_K_M (R1) | 1.5B Q4_K_M (R1) | Qwen 3.5 4B (R2) |
|---|---|---|---|
| valid JSON | 30/30 | 30/30 | 30/30 |
| exact all | 1/30 | 1/30 | 21/30 |
| exact en / ko / mix | 1/15, 0/10, 0/5 | 1/15, 0/10, 0/5 | 11/15, 7/10, 3/5 |
| contains addressed (all) | 24/24 | 22/24 | 20/24 (en 10/12, ko 7/8, mix 3/4) |
| ambient silence | 0/6 | 0/6 | **0/6** |
| wall s per call (median) | 5.1 | 9.8 | 0.70 (max 0.88) |
| prompt tokens / output tokens (median) | 262 / 15 | 262 / 15 | 267 / 9 |
| prefill ms / decode ms (median) | 4516 / 603 | 8239 / 1551 | 275 / 377 |

The 4B no longer returns the whole cast; it picks the right speaker or pair in 21/30 cases. It still **never stays silent** on ambient chit-chat: in all 6 cases it named one speaker (the director's silence behavior is the point of ambient mode, so this is the same blocking defect as round 1, only less visible). All 9 misses are accounted for: the 6 ambient cases (one speaker named instead of none) and 3 follow-up cases where it picked the wrong character (#6 Mira instead of Bram, #20 민준 instead of 하늘, #29 서연 instead of Luna). Every addressed-by-name case with a single target was right, including Korean and mixed.

Thinking on (`think:true`, 10 cases = every 3rd, `num_predict` 2048): **unusable**. 9 of 10 calls spent the whole 2048-token budget thinking (about 115 s each, ~18 tokens/s) and returned empty content; only the first finished (1122 tokens). Valid 1/10. So thinking mode would cost minutes per director call on this machine; not pursued further. Not run for memory or spoken moves.

### 2. Memory (12 synthetic transcripts: 5 EN, 5 KO, 2 mixed; keyword checklist, no judge)

| metric (think off) | all | EN | KO | mixed |
|---|---|---|---|---|
| extraction schema-valid | 12/12 | 5/5 | 5/5 | 2/2 |
| summary fact recall (keyword groups) | 24/31 | 12/12 | 6/13 | 6/6 |
| shared/encounter fact recall | 5/10 | 2/4 | 2/4 | 1/2 |
| personal fact recall (user's own facts, in `personal_facts`) | 4/11 | 1/4 | 3/5 | 0/2 |
| personal facts returned on "none" cases (false positives) | 0 | 0 | 0 | 0 |
| forbidden items in personal_facts | 1 | 0 | 1 | 0 |
| latency per call (median / max, 24 calls) | 9.7 s / 29.6 s | | | |

Latency split: the summary takes 15 to 30 s (median ~185 output tokens, the model writes far more than needed), extraction 1 to 5 s.

Reading the numbers (the keyword scorer is blunt, so these are judgments from reading the outputs):
- **Korean summaries came out in English** with romanized names (e.g. "Taewoo", "Su-a", "Borigae"). That is why KO recall is 6/13: the content is mostly right but the Korean keywords are absent. For a Korean server this is a real defect; a prompt that says "write in the transcript's language" would be the first thing to try (not tested).
- **Invention.** The automated invented-items counter reported 17 for summaries and 2 for extraction, but almost all are false alarms (sentence-initial words such as "This", "Despite"; romanized Korean names). Real problems seen by reading: summaries **embellish** ("long-standing camaraderie", "fracture under the weight of their conflicting natures", a "musician" status), and one extraction invented events ("민준과 태오는 떡볶이를 먹으러 갔다", they only talked about going). In the prompt-injection case (en4) the summary dramatized the attack text, but the injected fact (the password) was **not** saved; extraction kept only "Eli likes chess".
- **Fact routing is weak.** The model often files the user's own facts under `encounter_facts` instead of `personal_facts` (en2, ko2, ko1), so personal recall is 4/11.
- Consent: the extraction prompt has no consent distinction (opt-in is enforced later in the store), so a consented vs not-consented case was not applicable at the prompt level.

### 3. Game-action interpretation

Test set: all **195** rows of `prototypes/spoken_moves/phrases.csv` (133 move rows, 62 table-talk rows; the labelled set the FEAT-36/D34 numbers use). The brief's "673 phrases" is the size of the whole data directory including the example lists the e5 head trains on; scoring on those would be circular, so they were not used. No sampling. Model got the phrase plus the row's legal moves (with a one-line meaning each) and answered with a schema enum of legal moves plus `talk`. Metric as `bands.passes`: move rows right when the answer is the label or an accepted move; talk rows silent when the answer is `talk`.

| metric | e5 int8 + head + word rule (FEAT-36/D34, as recorded) | Qwen 3.5 4B, think off |
|---|---|---|
| right move on real moves | about 95% | **70.7%** (94/133) |
| silent on table talk | about 70% | **95.2%** (59/62; 96.8% counting accepted moves) |
| by language, pass overall | n/a | en 62/71, ko 73/101, mixed 19/23 |
| wall s per phrase (median / max) | tens of ms on the Pi [E] | 0.66 s / 0.93 s |
| invalid answers | n/a | 0/195 |

The LLM errs the opposite way from e5: it is conservative. Most of its 39 move misses are `talk` on valid phrases, especially Korean slang and short stands ("ㄱㄱ", "고고", "ㅎㅌ", "서렌 ㄱ", "됐어요", "그만할게", "nah im good", "that's enough") and negated hits ("don't hit me", "안 받아"). A smaller group is the wrong move: insurance vs even money ("이븐 머니요" read as no_insurance), "따블" and "dd". Since a missed move only means the player taps a button, and a false move on table talk is the costly error, this profile (95% silent) is nicer than the e5 head's 70% silent, but the right-move rate is 25 points worse.

Doubt reader (I Doubt It; 116 phrases from `tests/test_spoken_doubt.py`, 60 doubt, 56 not):

| | doubt recall | silent on non-doubt |
|---|---|---|
| `read_doubt` (word rule) | 60/60 | 56/56 |
| Qwen 3.5 4B, think off | 60/60 | 31/56 |

The model answered "doubt" for every negation and exclude phrase ("never lies", "구라 아니야", "거짓말하지마", "Without doubt, nice hand", "의심할 여지가 없어") and for lexical false friends ("lying on the couch", "the answer lies in luck", "뻥!"). The word rule is clearly better here and costs nothing.

### Conclusions
- **Director:** a large improvement over the 0.5B/1.5B in speaker picking (21/30 exact vs 1/30) and fast (0.7 s) on the remote GPU, but it still cannot do ambient silence (0/6) and is not better than the hosted director where silence matters. Thinking mode is unusable. It is not a drop-in replacement for the hosted call. A partial design that might work (not tested): keep the ambient/silence decision in a cheap rule or the hosted model, and use the local model only for choosing among addressed speakers. The latency is for the user's other machine over the tailnet; on the Pi itself a 4B model is not tested and, from round 1's scaling, would be far slower **[E]**.
- **Memory:** not good enough to replace the hosted memory role. Schema validity is perfect, but Korean summaries come out in English, summaries embellish, user facts land in the wrong array, and a summary costs ~15 to 30 s. Extraction alone (1 to 5 s) is the more plausible local job, after prompt tuning.
- **Game actions:** not a replacement for the e5 head on right-move rate (71% vs 95%), better on table-talk silence (95% vs 70%), and 0.66 s vs tens of ms. Possible as a fallback only for utterances the head cannot place; for doubts, keep `read_doubt`.
- Caveats: one model, one run (temperature 0, seed 1), small hand-written sets, a keyword scorer for memory, remote GPU of unknown type partly spilling to RAM. Differences of a case or two are noise.
