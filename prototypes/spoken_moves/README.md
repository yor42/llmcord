# Spoken moves prototype (FEAT-36, D32)

Throwaway harness that answers one question on the Pi 5 (4 GB, arm64): can a small multilingual embedding model, behind the word list, pick the right blackjack move from chat text? Nothing here is imported by the bot. Design: `docs/engineering/minigames-plan.md` ("Spoken moves"); model research: `docs/engineering/research/nl-game-actions-models.md`.

## The gate

- the right move is among the buttons for >= 95% of real moves,
- the bot stays silent on >= 85% of table talk,
- < 100 ms per message.

Buttons: one for a "sure" band, two for an "unsure" band, none when silent. Cascade = word list first (`llmcord_core.spoken.words.read`), then the embedding bands.

## Files

- `phrases.csv`: labelled test phrases (`id,text,lang,label,accept,legal,note,ok,fix`). `label` is a move or `talk`; `accept` lists extra moves that also count as right; `legal` lists the moves the table allows for that phrase.
- `examples.json`: example phrases per move plus `table_talk`; the embedding step's reference set. Must not repeat any phrase text (eval prints overlaps).
- `wordlist.json`: blackjack word list (same shape as `llmcord_core/spoken/lexicon/holdem.json`).
- `bands.py`: band selection, pass rule, threshold grid fit, stratified CV folds (pure python; tests in `tests/test_spoken_bands.py`).
- `run.py`: `fetch`, `eval`, `bench`.
- `models.lock.json`: pinned model revisions and file sha256 (later fetches verify against it).

## Setup

```
python3 -m venv prototypes/spoken_moves/.venv
prototypes/spoken_moves/.venv/bin/pip install -r prototypes/spoken_moves/requirements.txt
prototypes/spoken_moves/.venv/bin/python prototypes/spoken_moves/run.py fetch
```

The venv is gitignored and separate from the main one. `fetch` downloads (plain urllib, no huggingface_hub) into `~/.cache/llmcord-spoken/<model>/`, about 600 MB, and makes an int8 copy of the e5 model locally (needs the `onnx` package).

## Commands

```
run.py fetch [--model e5|minilm|both]
run.py eval  [--model e5|minilm|both] [--out report.md]
run.py bench [--model e5|minilm|both] [--out report.md]
```

`eval` fits the thresholds (grid search: maximize talk-silent rate subject to >= 95% move pass, tie-break on more right "sure" bands) and reports honest 5-fold stratified cross-validation next to the in-sample fit, for "embedding only" and "cascade", with per-language and per-label breakdowns and every failing row. Trust the cross-validated numbers; the in-sample fit overstates. `bench` reports load time, peak RSS and warm per-message latency (tokenize + infer + pool) at 1, 2 and 4 threads.

## Reviewing phrases.csv

Open it in a spreadsheet (UTF-8). For each row fill:

- `ok`: `y` if the label and legal moves are right and the phrase sounds like something a player would type, `n` if not.
- `fix`: a better phrasing or the label you would give it (leave `ok` as `n`).

Korean reviewers: please check especially that the Korean and mixed Korean/English phrases sound natural, not translated, and that slang (`ㅈㅈ`, `가자`, ...) is labelled the way players mean it. Do not edit the rows to make the numbers pass; change the data only when it is wrong.

`eval` reads every row and ignores `ok` and `fix`: the reviewed edits are applied to the rows by hand before the gate run.

Caveat: the draft word list and example phrases were written by the same author as `phrases.csv`, so they are not fully independent of it (near-duplicates such as "대박" / "대박이다 진짜" exist). Treat the draft numbers as optimistic, and decide the gate on reviewed phrases, ideally with a fresh held-out set from real players.
