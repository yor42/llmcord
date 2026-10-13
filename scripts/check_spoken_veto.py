"""FEAT-49c: operator-run drift check of the decision-model veto against the synthetic blackjack phrases (offline unless --yes).

usage: check_spoken_veto.py --e5 FILE [--phrases CSV] [--tau 0.5] [--yes]

FILE is the output of ``prototypes/decision_eval/cascade.py dump-e5``. Without --yes it only estimates the calls and cost and makes no request.
With --yes (and LLMCORD_SPOKEN_VETO=openrouter-d1 plus OPENROUTER_API_KEY) it sends each non-silent phrase through the production client,
applies the cascade policy and prints the metrics next to the baseline. Exit codes: 0 fine or dry run, 1 drift, 2 refused, bad input or inconclusive (10% or more of the requests failed).
Operator CLI text: English and not translated (docs/engineering/i18n-plan.md). Output never holds the key or a response body.
"""
import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototypes" / "decision_eval"))
import cascade  # noqa: E402

from llmcord_core.spoken import veto_client  # noqa: E402

PHRASES = ROOT / "prototypes" / "spoken_moves" / "phrases.csv"
# Measured per-call price of the Decision-1 reader (conservative): FEAT-47 full run $0.08 / 3,110 calls, all questions per row included.
# Cross-check: FEAT-44 cascade replay, Decision-1 noul tau 0.5, $0.005 per 1,000 messages at 72% asked (~$0.0000069 per call).
PER_CALL_USD = 0.000026
BASELINE = "baseline (Decision-1, tau 0.5, 195 rows): right move 90%, silent on chatter 100%, lost moves 4"
MIN_CHATTER_SILENT = 0.95
MAX_LOST_MOVES = 8
NEED_SWITCH = "set LLMCORD_SPOKEN_VETO=openrouter-d1"
NEED_KEY = "set OPENROUTER_API_KEY to a printable-ASCII key"


def refused_dirs(env):
    dirs = [(ROOT / "data").resolve()]
    if env.get("LLMCORD_DATABASE_PATH"):
        parent = Path(env["LLMCORD_DATABASE_PATH"]).resolve().parent
        if parent != ROOT:
            dirs.append(parent)
    return dirs


def load_items(path):
    with open(path, encoding="utf-8", newline="") as f:
        return [{"id": f"bj{r['id']}", "task": "bj", "text": r["text"], "lang": r["lang"], "label": r["label"],
                 "accept": r["accept"].split(), "legal": r["legal"].split()} for r in csv.DictReader(f)]


async def ask_all(client, asked):
    """({item id: reading or None}, errors); a failed call is None, which the policy treats as e5 standing."""
    readings, errors = {}, 0
    try:
        for it in asked:
            try:
                readings[it["id"]] = {"probs": await client.ask(it["text"], it["legal"])}
            except Exception:
                readings[it["id"]] = None
                errors += 1
    finally:
        await client.close()
    return readings, errors


def main(argv=None, env=None, transport=None) -> int:
    env = os.environ if env is None else env
    p = argparse.ArgumentParser(description="FEAT-49c drift check of the decision-model veto")
    p.add_argument("--e5", required=True)
    p.add_argument("--phrases", default=str(PHRASES))
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--yes", action="store_true", help="really call the provider (costs money)")
    a = p.parse_args(argv)
    phrases = Path(a.phrases).resolve()
    if any(d == phrases or d in phrases.parents for d in refused_dirs(env)):
        print(f"refusing --phrases inside a data directory: {phrases}", file=sys.stderr)
        return 2
    try:
        items = load_items(phrases)
        e5 = json.loads(Path(a.e5).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError) as error:
        print(f"cannot read the input files ({type(error).__name__})", file=sys.stderr)
        return 2
    missing = [it["id"] for it in items if it["id"] not in e5]
    if missing:
        print(f"e5 file lacks {len(missing)} rows, e.g. {missing[:3]}", file=sys.stderr)
        return 2
    asked = [it for it in items if e5[it["id"]]["band"] != "silent"]
    if not a.yes:
        print(f"estimated calls: {len(asked)}")
        print(f"estimated cost: ${cascade.cost_per_1000_messages(len(asked), len(items), PER_CALL_USD):.4f} per 1,000 messages")
        print(f"estimated cost of this run: ${len(asked) * PER_CALL_USD:.4f}")
        print("no request is made without --yes")
        return 0
    if env.get(veto_client.ENV_SWITCH) != veto_client.ENV_VALUE:
        print(NEED_SWITCH, file=sys.stderr)
        return 2
    client = veto_client.from_env(env, transport=transport)
    if client is None:
        print(NEED_KEY, file=sys.stderr)
        return 2
    readings, errors = asyncio.run(ask_all(client, asked))
    base = cascade.finals_of(e5)
    policy = {"cover": "sure+unsure", "tau": a.tau}
    finals = {it["id"]: cascade.apply_policy(e5[it["id"]], readings.get(it["id"]), policy) for it in items}
    table = [{"reader": None, "cover": "-", "tau": None, "result": cascade.evaluate(items, base), "per_call": 0.0, "price": "-", "finals": base},
             {"reader": "or-d1-noul", "cover": "sure+unsure", "tau": a.tau, "result": cascade.evaluate(items, finals), "per_call": PER_CALL_USD,
              "price": "estimated", "finals": finals}]
    print(cascade.render(items, table))
    r = table[1]["result"]
    lost = sum(1 for it in items if it["label"] != "talk" and cascade._right(it, base[it["id"]]["moves"]) and not cascade._right(it, finals[it["id"]]["moves"]))
    print()
    print(f"right move: {cascade.pct(r['move_pass'], r['move_n'])} ({r['move_pass']}/{r['move_n']})")
    print(f"silent on chatter: {cascade.pct(r['talk_silent'], r['talk_n'])} ({r['talk_silent']}/{r['talk_n']})")
    print(f"sure wrong: {r['sure_wrong']}")
    print(f"lost moves: {lost}")
    print(f"errors: {errors}")
    print(BASELINE)
    if errors and errors * 10 >= len(asked):
        print(f"INCONCLUSIVE: {errors} of {len(asked)} requests failed, the result below reflects the local reader for those rows")
        return 2
    drift = (r["talk_n"] and r["talk_silent"] / r["talk_n"] < MIN_CHATTER_SILENT) or lost > MAX_LOST_MOVES
    print(f"DRIFT: silent on chatter under {MIN_CHATTER_SILENT:.0%} or more than {MAX_LOST_MOVES} lost moves" if drift else "OK: no drift against the baseline")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
