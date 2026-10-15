"""FEAT-62 (D43): Vault simulator. Operator tool, not part of the product or the test run. Offline, stdlib only.

usage: simulate_vault.py [--length 4,5,6] [--protocol standard,...] [--turns 9,...] [--tools N,...] [--clues 0..3]
                         [--strategy optimal|careful|exploratory|bold|casual] [--samples N|all] [--seed S] [--output PATH]

Each comma list is crossed into configurations. Without --turns the protocol default budget (DEFAULT_TURNS) is used.
Secrets are sampled reproducibly from the seed (``--samples all`` plays all 5040 four-digit secrets). Every game is
played to a hard horizon of 20 turns so the report can say what budget would be needed, and the completion
probability is counted at the configured budget (plus 11, 13 and 15 turns when those are larger).

Strategies (all see only public constraints): optimal is minimax (weight 0); careful, exploratory and bold use the
blend weights 0.2, 0.5 and 0.9 (exploratory also deviates up to 40 percent of the cap); casual is a random candidate
consistent with ONLY the last two feedbacks (plus the starting clues and tool facts), never repeating a past guess.
Tools: after the first guess the player spends up to ``--tools`` activations, each time on the query (any of the five
kinds, any parameters) with the best expected candidate reduction per turn; a tool costs its turns from the budget.
Clues: ``--clues`` partner clues (make_clues) are known before the first guess. The Open protocol starts with one
digit of the secret. Progress goes to stderr; the Markdown report goes to --output.
"""
import argparse
import math
import random
import statistics
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llmcord_core.vault import (  # noqa: E402
    DEFAULT_TURNS,
    LENGTHS,
    PROTOCOLS,
    TOOLS,
    DigitAt,
    GuessResult,
    OpenDigit,
    VaultError,
    all_secrets,
    evaluate_tool,
    feedback,
    filter_candidates,
    make_clues,
    played,
    open_digit,
    suggest_guess,
    tool_fits,
)

STRATEGIES = {  # name -> (blend weight, unpredictability); casual is handled apart
    "optimal": (0.0, 0), "careful": (0.2, 0), "exploratory": (0.5, 40), "bold": (0.9, 0), "casual": (None, 0),
}
HORIZON = 20
EXTRA_BUDGETS = (11, 13, 15)
TOOL_SAMPLE = 300
DEFAULT_SAMPLES = 200


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def csv_of(parse, name):
    def conv(text):
        try:
            return [parse(p) for p in text.split(",")]
        except (ValueError, KeyError):
            raise argparse.ArgumentTypeError(f"bad {name}: {text!r}")
    return conv


def choice(options):
    def parse(p):
        if p not in options:
            raise ValueError(p)
        return p
    return parse


def ranged(lo, hi):
    def parse(p):
        v = int(p)
        if not lo <= v <= hi:
            raise ValueError(p)
        return v
    return parse


def tool_options(length, history, last_guess):
    digits = "0123456789"
    opts = [("vault_residual_scanner", {"guess": last_guess, "digit": d}) for d in last_guess]
    opts += [("vault_spectrum_analyzer", {"position": p}) for p in range(1, length + 1)]
    opts += [("vault_comparison_circuit", {"a": a, "b": b}) for a in digits for b in digits if a < b]
    opts.append(("vault_position_verifier", {"guess": last_guess}))
    opts += [("vault_master_decoder", {"position": p}) for p in range(1, length + 1)]
    return opts


def pick_tool(cands, length, history, last_guess, turns_left, rng, refused=()):
    """The tool query with the best expected reduction per turn on a sample of the candidates (public data only)."""
    sample = cands if len(cands) <= TOOL_SAMPLE else rng.sample(cands, TOOL_SAMPLE)
    best, best_gain = None, 0.0
    for key, params in tool_options(length, history, last_guess):
        if not tool_fits(key, turns_left) or (key, tuple(params.items())) in refused:
            continue
        groups = {}
        for c in sample:
            try:
                ans = evaluate_tool(c, key, params, history).constraint
            except VaultError:
                ans = None
            groups[ans] = groups.get(ans, 0) + 1
        expected = sum(n * n for n in groups.values()) / len(sample)
        gain = (len(sample) - expected) / TOOLS[key].turn_cost
        if gain > best_gain + 1e-9:
            best, best_gain = (key, params), gain
    return best


def play(secret, cfg, memo, clock):
    length, protocol, budget, strategy, tools, clues, seed = cfg
    weight, unp = STRATEGIES[strategy]
    rng = random.Random(f"{seed}/{cfg}/{secret}")
    start = [OpenDigit(open_digit(secret, rng))] if protocol == "open" else []
    start += make_clues(secret, clues, rng)
    cons, results, disclosed, guessed = list(start), [], [], set()
    t0 = time.perf_counter()
    cands = filter_candidates(cons, length)
    clock[0] += time.perf_counter() - t0
    used, tools_left, after, uses = 0, tools, [], []
    while used < HORIZON:
        if tools_left and results and budget - used >= 1:
            t0 = time.perf_counter()
            hist = results + disclosed
            refused, res = set(), None
            while res is None:  # a refused query is free (nothing new, say): ask for the next best
                pick = pick_tool(cands, length, hist, results[-1].guess, budget - used, rng, refused)
                if not pick:
                    break
                key, params = pick
                try:
                    res = evaluate_tool(secret, key, params, hist)
                except VaultError:
                    refused.add((key, tuple(params.items())))
            if res:
                before = len(cands)
                cons.append(res.constraint)
                if isinstance(res.constraint, DigitAt):
                    disclosed.append(res.constraint)
                cands = filter_candidates([res.constraint], length, base=cands)
                used += TOOLS[key].turn_cost
                tools_left -= 1
                uses.append((key, before, len(cands)))
            clock[0] += time.perf_counter() - t0
        t0 = time.perf_counter()
        if weight is None:
            pool = filter_candidates([c for c in cons if not isinstance(c, GuessResult)] + results[-2:], length)
            pool = [c for c in pool if c not in guessed] or [c for c in cands if c not in guessed] or cands
            guess = rng.choice(pool)
        else:
            key = (strategy, protocol, length, tuple(cons)) if unp == 0 else None
            guess = memo.get(key) if key else None
            if guess is None:
                guess = suggest_guess(cands, protocol, length, weight, unp, f"{seed}/{secret}/{used}")
                if key:
                    memo[key] = guess
        clock[0] += time.perf_counter() - t0
        used += 1
        guessed.add(guess)
        if guess == secret:
            after.append(1)
            return used, after, uses
        res = GuessResult(guess, protocol, feedback(protocol, secret, guess))
        results.append(res)
        new = played(protocol, guess, res.result)
        cons.extend(new)
        t0 = time.perf_counter()
        cands = filter_candidates(new, length, base=cands)
        clock[0] += time.perf_counter() - t0
        after.append(len(cands))
    return None, after, uses


def percentile(values, q):
    """Nearest-rank percentile of ``values`` (None counts as beyond the horizon)."""
    ordered = sorted(HORIZON + 1 if v is None else v for v in values)
    v = ordered[max(0, math.ceil(q * len(ordered)) - 1)]
    return f">{HORIZON}" if v > HORIZON else str(v)


def run_config(cfg, secrets_list, memo):
    clock = [0.0]
    wins, curve, tool_stats = [], {}, {}
    for i, secret in enumerate(secrets_list, 1):
        turns, after, uses = play(secret, cfg, memo, clock)
        wins.append(turns)
        for n, left in enumerate(after, 1):
            curve.setdefault(n, []).append(left)
        for key, before, left in uses:
            tool_stats.setdefault(key, []).append((before, left))
        if i % 50 == 0:
            log(f"  {i}/{len(secrets_list)}")
    return wins, curve, tool_stats, clock[0]


def label(cfg):
    length, protocol, budget, strategy, tools, clues, _ = cfg
    return f"{length} digits, {protocol}, {budget} turns, {strategy}, tools {tools}, clues {clues}"


def section(cfg, outcome, n):
    wins, curve, tool_stats, wall = outcome
    solved = [w for w in wins if w is not None]
    lines = [f"## {label(cfg)}", ""]
    budgets = [cfg[2]] + [b for b in EXTRA_BUDGETS if b > cfg[2]]
    lines += ["| Budget (turns) | Completion probability |", "| --- | --- |"]
    lines += [f"| {b} | {sum(1 for w in solved if w <= b) / n:.1%} |" for b in budgets]
    lines += [""]
    if solved:
        mean = f"{statistics.fmean(solved):.2f}" + ("" if len(solved) == n else f" (solved games only, {n - len(solved)} unsolved)")
        lines.append(f"- Turns to open: mean {mean}, median {percentile(wins, 0.5)}, p90 {percentile(wins, 0.9)}, p95 {percentile(wins, 0.95)}")
    else:
        lines.append(f"- Turns to open: none solved within {HORIZON}")
    lines.append(f"- Failure probability at {cfg[2]} turns: {1 - sum(1 for w in solved if w <= cfg[2]) / n:.1%}")
    lines.append("- Mean candidates left after each guess (games still running): " + ", ".join(
        f"{k}: {statistics.fmean(v):.1f}" for k, v in sorted(curve.items()) if k <= 12))
    if tool_stats:
        lines.append("- Marginal reduction per tool (mean candidates before to after, uses):")
        for key, rows in sorted(tool_stats.items()):
            b, a = statistics.fmean(r[0] for r in rows), statistics.fmean(r[1] for r in rows)
            lines.append(f"  - {key}: {b:.1f} to {a:.1f} ({1 - a / b:.0%} fewer), {len(rows)} uses")
    lines.append(f"- Candidate analysis wall clock: {wall:.1f} s total, {wall / n * 1000:.1f} ms per game")
    return lines + [""]


def summary(rows):
    out = ["| Configuration | Samples | Completion at budget | Mean turns | p95 |", "| --- | --- | --- | --- | --- |"]
    for cfg, n, wins in rows:
        solved = [w for w in wins if w is not None]
        done = sum(1 for w in solved if w <= cfg[2]) / n
        mean = f"{statistics.fmean(solved):.2f}" if solved else "-"
        out.append(f"| {label(cfg)} | {n} | {done:.1%} | {mean} | {percentile(wins, 0.95)} |")
    return out + [""]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Simulate Vault games against the pure engine.")
    ap.add_argument("--length", type=csv_of(ranged(4, 6), "length"), default=[4])
    ap.add_argument("--protocol", type=csv_of(choice(PROTOCOLS), "protocol"), default=["standard"])
    ap.add_argument("--turns", type=csv_of(ranged(1, HORIZON), "turns"), default=None)
    ap.add_argument("--tools", type=csv_of(ranged(0, 5), "tools"), default=[0])
    ap.add_argument("--clues", type=csv_of(ranged(0, 3), "clues"), default=[0])
    ap.add_argument("--strategy", type=csv_of(choice(STRATEGIES), "strategy"), default=["optimal"])
    ap.add_argument("--samples", default=str(DEFAULT_SAMPLES), help="a number, or 'all' (4 digits only)")
    ap.add_argument("--seed", default="vault")
    ap.add_argument("--output", default=str(ROOT / "docs" / "engineering" / "research" / "vault-simulation.md"))
    args = ap.parse_args(argv)
    if args.samples == "all":
        if args.length != [4]:
            ap.error("--samples all needs --length 4")
        count = None
    else:
        try:
            count = int(args.samples)
        except ValueError:
            count = 0
        if count < 1:
            ap.error("--samples must be a positive number or 'all'")
    if any(length not in LENGTHS for length in args.length):
        ap.error("bad length")

    report = [
        "# Vault simulation", "",
        f"Seed `{args.seed}`, samples {args.samples}. Games are played to a horizon of {HORIZON} turns; "
        "completion is counted at the configured budget and at the larger budgets listed.", "",
    ]
    rows, sections, memo = [], [], {}
    for length, protocol, tools, clues, strategy in product(args.length, args.protocol, args.tools, args.clues, args.strategy):
        for budget in args.turns or [DEFAULT_TURNS[protocol]]:
            cfg = (length, protocol, budget, strategy, tools, clues, args.seed)
            space = all_secrets(length)
            picked = list(space) if count is None else random.Random(f"{args.seed}/{length}").sample(space, min(count, len(space)))
            log(f"{label(cfg)}: {len(picked)} games")
            outcome = run_config(cfg, picked, memo)
            rows.append((cfg, len(picked), outcome[0]))
            sections += section(cfg, outcome, len(picked))
    Path(args.output).write_text("\n".join(report + summary(rows) + sections), encoding="utf-8")
    log(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
