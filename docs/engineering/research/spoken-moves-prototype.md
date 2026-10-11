# Spoken moves prototype: first run (FEAT-36)

Date: 2026-10-12. Harness and data: `prototypes/spoken_moves/` (README there). Design: [minigames-plan.md](../minigames-plan.md#spoken-moves-feat-36-feat-37-feat-38-d32-user-2026-10-11), D32. Model research: [nl-game-actions-models.md](nl-game-actions-models.md).

**Result: the gate is not met.** Speed is far inside the limit. The models find the right move well enough. They are not silent often enough on table talk.

## Setup

- Data, all drafts written by the orchestrator and not yet reviewed by members:
  - `phrases.csv`: 150 labelled blackjack phrases. 96 are moves (34 hit, 34 stand, 17 double, 11 surrender) and 54 are table talk. They mix EN, KO and EN/KO, and include slang, negations and lookalikes such as "hit me up later" and "더블 치즈버거".
  - `examples.json`: 54 move examples and 28 table-talk examples.
  - `wordlist.json`: the word list.
- Models:
  - `intfloat/multilingual-e5-small`, quantized to int8 locally with ONNX Runtime dynamic quantization.
  - `paraphrase-multilingual-MiniLM-L12-v2`, using the official `model_qint8_arm64.onnx`.
  - Revisions and checksums are pinned in `models.lock.json`.
- Method:
  - Nearest example per legal move, scored on three numbers: top score, margin to the next move, and gap to table talk.
  - The three bands (sure, unsure, silent) are fitted by grid search to maximise talk silence while keeping at least 95% of moves right.
  - The cascade is the word list first (`llmcord_core.spoken.words.read`), then the embedding.
  - The honest numbers are 5-fold cross-validated.

## Numbers (5-fold CV)

| Model / mode | Right move among buttons (gate ≥ 95%) | Silent on table talk (gate ≥ 85%) |
| --- | --- | --- |
| e5, embedding only | 92.7% | 38.9% |
| e5, word list + embedding | 93.8% | 38.9% |
| MiniLM, embedding only | 89.6% | 1.9% |
| MiniLM, word list + embedding | 92.7% | 11.1% |

Speed on the Pi 5, warm, one message at a time (tokenize, infer and pool):

| Model | p50 | p95 | Load time | Peak RSS |
| --- | --- | --- | --- | --- |
| e5 | 5–7 ms | ≤ 8 ms | 1 s | about 580 MB |
| MiniLM | 5–7 ms | ≤ 8 ms | 1 s | about 580 MB |

The speed gate (< 100 ms) passes by more than 10x.

A first run scored e5 cascade talk silence at 53.7%. That run used a word list that had been written with the test phrases in view (it included "패스", "스톱" and a "히트곡" exclusion). Those words were removed and the run repeated. The figure moving that much on three words shows how small and noisy this set is.

## What fails

- **e5 is clearly better than MiniLM.** MiniLM's similarity scores do not separate table talk at all, so drop it.
- **Lookalikes:**
  - The word list rule is "a match is sure", so "hit me up later", "that song is a hit", "double check the rules", "더블 치즈버거 먹고싶다", "스탠드 조명 샀어" and "두 배로 맛있다" get a sure move check.
  - These cases are structural: no threshold fixes them.
- **Generic chat lands near moves.** With only 28 table-talk examples, short chat lands closer to "hit" or "stand" than to talk: "잠깐만", "one sec", "thinking...", "몇 점이지?", "wait 잠깐" and "brb 화장실".
  - e5 scores sit in a narrow 0.84–0.93 range, so the margins are tiny (often < 0.02).
- **Missed moves:** "hold", "that's enough", "패스", "스톱", "fold" (as surrender) and "ㅈㅈ".

## Options for a second pass (not decided)

1. **Word list "sure" only when the move word is most of the message.** For example, at most about 3 words besides fillers; otherwise it becomes an unsure check or goes to the embedding. This targets the lookalike class.
2. **Many more table-talk examples.** Generic EN/KO chat, a few hundred lines, written or collected with consent, not from `data/`. kNN silence depends on how well "talk" is covered.
3. **A small trained head instead of nearest example.** Logistic regression on the e5 vectors (move vs talk, then which move), fitted on examples only. It costs microseconds at runtime and gives calibrated probabilities instead of raw cosines.
4. **Revisit the gate.** The move check is cheap to dismiss ("Just table talk" deletes it), so 85% silence may be stricter than players need. A lower bar with a per-member off switch is a product call.
5. **Reviewed phrases and a fresh held-out set** from real players before any gate decision. The draft data and examples share an author, so the numbers above are optimistic.

## Caveats

- n = 150. The fold thresholds vary, and a coarser or finer grid moves talk silence by several points.
- "Silent" is counted strictly: a talk row whose accepted button is shown passes the per-label table but does not count as silent.
- The local e5 int8 file is not byte-reproducible across ONNX Runtime versions. Its hash is recorded but not enforced. FEAT-37 bakes one verified file into the image anyway.
