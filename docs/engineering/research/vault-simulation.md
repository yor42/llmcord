# Vault simulation (FEAT-62, D43)

First measured report of the pure Vault engine (`llmcord_core/vault/`) against the D43 bar. Produced by
`scripts/simulate_vault.py` on the Raspberry Pi 5 (arm64), Python 3.13, seed `d43`. Offline and deterministic: the same
command and seed reproduce every number except the wall-clock lines. `DEFAULT_TURNS` was **not** changed to chase the bar.

## Commands

```
.venv/bin/python scripts/simulate_vault.py --length 4 --protocol standard,total_match,position_only --strategy optimal,casual --samples 500 --seed d43 --output /tmp/vs/r1.md
.venv/bin/python scripts/simulate_vault.py --length 4 --protocol open --strategy optimal,casual --samples 500 --seed d43 --output /tmp/vs/r2.md
.venv/bin/python scripts/simulate_vault.py --length 5,6 --protocol standard --strategy optimal,casual --samples 60 --seed d43 --output /tmp/vs/r3.md
.venv/bin/python scripts/simulate_vault.py --length 4 --protocol standard --strategy optimal --tools 2 --clues 2 --samples 100 --seed d43 --output /tmp/vs/r4.md
.venv/bin/python scripts/simulate_vault.py --length 4 --protocol total_match --strategy optimal --tools 2,4 --turns 9,15 --samples 100 --seed d43 --output /tmp/vs/r5.md
```

Each run samples secrets reproducibly from the seed (500 of the 5040 four-digit secrets; only 60 at 5 and 6 digits
because the casual model filters up to 151200 codes per turn). Every game is played to a 20-turn horizon so the
report can say what budget would be needed. Completion is the share opened within the budget; mean turns counts only
games opened within 20 turns, so it understates hard configurations (the unsolved count is in the generated report).

Player models. **Optimal**: minimax guess on the public record (weight 0). **Casual**: a random guess consistent with
only the last two feedbacks (plus the starting clues and tool facts), never repeating a past guess. These are the
only two models the bar names; a human sits somewhere between them.

## Result against the D43 bar

Budgets 11, 13 and 15 are the larger budgets the simulator reports for every run (no tools, no clues).

| Configuration | Model | Bar | At default budget | At 11 / 13 / 15 turns | Verdict |
| --- | --- | --- | --- | --- | --- |
| Standard, 4 digits, 9 turns | optimal | 99% | 100.0% | 100 / 100 / 100% | pass |
| Standard, 4 digits, 9 turns | casual | about 60% | 39.2% | 49.8 / 60.2 / 66.2% | **miss** (13 turns meets it) |
| Total Match, 4 digits, 9 turns | optimal | 95% | 20.0% | 29.0 / 37.8 / 45.2% | **miss** (not met by 15 turns; 67.4% at the 20-turn horizon) |
| Total Match, 4 digits, 9 turns | casual | 30% | 6.4% | 9.8 / 13.4 / 17.4% | **miss** |
| Position-Only, 4 digits, 9 turns | optimal | 95% | 60.6% | 99.0 / 100 / 100% | **miss** (11 turns meets it) |
| Position-Only, 4 digits, 9 turns | casual | 30% | 6.6% | 10.8 / 13.4 / 17.2% | **miss** (not met at 15) |
| Open, 4 digits, 8 turns | optimal | 95% | 100.0% | 100 / 100 / 100% | pass |
| Open, 4 digits, 8 turns | casual | 30% | 50.8% | 73.8 / 82.8 / 89.4% | pass |
| Standard, 5 digits, 9 turns | optimal | (no bar) | 100.0% (60 games) | 100 / 100 / 100% | informational |
| Standard, 5 digits, 9 turns | casual | (no bar) | 11.7% | 20.0 / 38.3 / 46.7% | informational |
| Standard, 6 digits, 9 turns | optimal | (no bar) | 100.0% (60 games) | 100 / 100 / 100% | informational |
| Standard, 6 digits, 9 turns | casual | (no bar) | 10.0% | 13.3 / 16.7 / 18.3% | informational |

Optimal turns to open: Standard mean 5.36 (p95 7); Open mean 4.94 (p95 6); Position-Only mean 8.86 (median 9, p90 and
p95 11); 5 digits mean 6.00 (p95 7); 6 digits mean 6.42 (p95 8).

## Notes on the misses

- **Standard casual (39.2% against about 60%).** The model forgets everything older than two guesses, so it keeps
  re-testing codes the full record rules out. It reaches 60.2% at 13 turns. The bar's 60% most likely assumed a
  player who remembers more; a model that keeps the whole record is the optimal model, which already clears 99%.
  Decide whether the bar or the casual model is the thing to adjust before touching the budget. Open at 8 turns
  (50.8%) is easier than Standard because one digit is given.
- **Total Match is far harder than Standard, and tools are what fix it.** Feedback is the number of shared digits,
  which does not depend on where the digits sit, so a guess never narrows the order of digits it has already
  identified. The solver finds the digit set in about five guesses; the order is then found by trying the remaining
  permutations (up to 24 for four digits), at most one per turn. Completion without tools is 20.0% at 9 turns, 29.0%
  at 11, 37.8% at 13, 45.2% at 15 and 67.4% at the 20-turn horizon (median 17 turns), so no budget up to 15 reaches
  the 95% bar, and neither would the 20-turn horizon. With the greedy tool user: 2 tools give 42.0% at 9 turns and 67.0% at 15; 4
  tools give 59.0% at 9 and 100% at 15 (the `--turns 9,15` rows of run r5). The 94.0% (11) and 99.0% (13) figures
  for 4 tools are budget-9 play scored at larger cutoffs (tools fire only while fewer than 9 turns are used), so
  they are indicative, not a measured 11- or 13-turn budget. Four tools therefore meet the 95% bar at a 15-turn
  budget; rerun with `--turns 11,13` before choosing a smaller one. The casual model stays at 17.4% at 15 turns
  without tools. The generated per-run reports (r1-r5) are not kept in the repo; rerun the commands to rebuild them.
  **Correction.** An earlier version of this report said Total Match was "unwinnable" with "no budget helping" (4.6%).
  That was a simulator and engine-usage defect: a wrong guess satisfied its own Total Match result, so the solver
  kept re-recommending it. Guesses are now recorded with `played()` (a wrong guess also adds `NotSecret`), and the
  numbers above come from the corrected run. The other rows were rerun and are unchanged.
- **Position-Only optimal (60.6% at 9 turns).** Needs 11 turns for 99.0%. Mean 8.86 turns, median 9, so about
  four in ten games exceed 9. Raising its budget to 11 meets the bar for the optimal solver. The casual model
  stays near 17% even at 15 turns, so it would remain under the 30% casual bar unless clues or tools are included.
- **Caveat:** the simulator's tool user is a greedy best-reduction-per-turn picker, not a tuned strategy, so its
  Total Match numbers are a floor.

## Tools and clues

Standard, 4 digits, 9 turns, optimal, 2 tools and 2 clues, 100 games: 100.0% completion, mean 5.58 turns (p95 7).
Mean candidates left after each guess: 203.4, 11.3, 1.3, 1.0. Mean candidate reduction per use: comparison circuit
113.7 to 42.4 (63% fewer, 124 uses), position cross-checker 140.2 to 41.4 (70% fewer, 51 uses), residual scanner 9.4
to 4.6 (5 uses), spectrum analyzer 11.6 to 5.7 (14 uses). Tools cost turns, so with the optimal solver they do not
raise Standard completion; they matter for Total Match (run r5, above): comparison circuit 1061 to 404 candidates
(62% fewer, 161 uses), position cross-checker 213 to 88 (59% fewer, 178 uses), spectrum analyzer 10.3 to 5.2, master
decoder 4.6 to 1.8 in the 15-turn run.

## Performance

Candidate analysis wall clock per game on the Pi 5 (generated reports): about 17 to 23 ms for the optimal solver at 4
digits, 280 ms at 5 digits and 730 ms at 6 digits. The casual model costs 150 to 260 ms per game at 4 digits and 7.6 s
per game at 6 digits (it filters the 151200-code space each turn), which is simulator cost only; the engine's
`suggest_guess` scores at most 300 guesses against at most 2000 sampled candidates.
