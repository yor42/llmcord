"""FEAT-43: run the decision-model evaluation (OpenAI Decisions, Microsoft Decision-1, Haiku, Gemini Flash-Lite) on the spoken-move sets.

Dry run by default: prints per-candidate call counts and an estimated cost, makes no request, needs no key, writes nothing.
--run makes the paid calls (sequentially) and writes results.json and summary.json to --out, then prints the summary.
Keys come from OPENAI_API_KEY / OPENROUTER_API_KEY only and go to the poster only; they are never printed or stored.
Usage: python run_eval.py [--candidates a,b] [--shuffles K] [--limit N] [--out DIR] [--run] [--max-usd X] [--tau T]
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import evalcore as ec  # noqa: E402

TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
MAX_ERRORS = 5
PREDICATE_KINDS = ("predicate", "noul")


def http_post(url, body, key, transport=None):
    """(response_json, wall_seconds); raises with the status code and the first 200 chars of the provider's error body only."""
    import httpx
    t0 = time.perf_counter()
    try:
        with httpx.Client(transport=transport, timeout=30, follow_redirects=False) as client:
            r = client.post(url, json=body, headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError as e:
        raise RuntimeError(f"request failed: {type(e).__name__}") from None
    wall = time.perf_counter() - t0
    if r.status_code >= 300:
        raise RuntimeError(f"HTTP {r.status_code}: {redact(r.text, [key])[:200]}")
    try:
        return r.json(), wall
    except ValueError:
        raise RuntimeError(f"HTTP {r.status_code}: response is not JSON") from None


def scrub(text, keys):
    for k in keys:
        if k:
            text = text.replace(k, "[key]")
    return text


def redact(text, keys):
    """Key values first (on the full text), then anything that looks like a bearer token or an sk- key."""
    text = scrub(text, keys)
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer [redacted]", text)
    return re.sub(r"sk-[A-Za-z0-9_\-]{6,}", "sk-[redacted]", text)


def pct(n, d):
    return f"{n / d:.0%}" if d else "n/a"


def num(v, fmt="{:.2f}"):
    return fmt.format(v) if v is not None else "n/a"


def save(out, meta, records):
    tmp = out / "results.json.tmp"
    tmp.write_text(json.dumps({"meta": meta, "records": records}, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, out / "results.json")


def run_calls(cands, items, a, keys, post, records, flush):
    """Sequential calls, appending to `records` (so an interrupt keeps them). Returns True when the budget stopped the run."""
    spent = 0.0
    for cand in cands:
        n = a.shuffles if cand in ec.SHUFFLED else 1
        errors = 0
        for it in items:
            for k in range(n):
                req = ec.build_request(cand, it, k)
                rec = {"item": it["id"], "task": it["task"], "candidate": cand, "k": k}
                try:
                    resp, wall = post(req["url"], req["json"], keys[cand])
                    rec.update(ec.parse_answer(cand, it, resp, a.tau), wall=wall, error=None)
                    errors = 0
                except Exception as e:  # an error is an invalid answer, never an abort
                    rec.update(answer=None, p=None, probs={}, refusal=False, cost=None, tokens=None, wall=None,
                               error=redact(f"{type(e).__name__}: {e}", keys.values())[:300])
                    errors += 1
                rec["cost_estimated"] = rec["cost"] is None and rec["error"] is None
                if rec["cost_estimated"]:
                    rec["cost"] = ec.est_call_usd(cand, req["json"])
                spent += rec["cost"] or 0.0
                records.append(rec)
                if len(records) % 10 == 0:
                    try:
                        flush()
                    except OSError:
                        print("WARNING: could not write results.json, will retry", file=sys.stderr)
                if spent > a.max_usd:
                    print(f"budget exceeded: ${spent:.4f} > ${a.max_usd:.2f}; stopping", file=sys.stderr)
                    return True
                if errors >= MAX_ERRORS:
                    print(f"{cand}: {MAX_ERRORS} consecutive errors, skipping the rest of this candidate", file=sys.stderr)
                    break
            if errors >= MAX_ERRORS:
                break
    return False


def summarize(cands, items, records, tau):
    """Per candidate and task. Scope is `items`: an item with no record counts as invalid, errored calls are invalid votes."""
    out = {}
    for cand in cands:
        recs = [r for r in records if r["candidate"] == cand]
        kind = ec.CANDIDATES[cand]["kind"]
        per_item = {}
        for r in recs:
            per_item.setdefault(r["item"], []).append(r)
        heads, results, sweep_in = [], [], []
        for it in items:
            rs = per_item.get(it["id"])
            if not rs:
                heads.append((None, None))
                results.append({"answer": None, "p": None, "refusal": False})
                sweep_in.append((None, None))
                continue
            # headline: one answer per item (majority over the K orders for the shuffled candidates)
            h = ec.headline([(r["answer"], r["p"]) for r in rs])
            heads.append(h)
            results.append({"answer": h[0], "p": h[1], "refusal": h[0] is None and any(r["refusal"] for r in rs)})
            pr = rs[0]["probs"]
            if kind in PREDICATE_KINDS:  # sweep from the probabilities, not from the answer cut at tau
                best = max(pr, key=lambda m: pr[m]) if pr else None
                sweep_in.append((best, pr[best]) if best else (None, None))
            else:
                sweep_in.append(h)
        s = ec.score(items, results)
        task_out = {}
        for task in ("bj", "doubt"):
            idx = [i for i, it in enumerate(items) if it["task"] == task]
            if not idx:
                continue
            tits = [items[i] for i in idx]
            tres = [results[i] for i in idx]
            trecs = [r for r in recs if r["task"] == task]
            attempted = [it["id"] for it in tits if it["id"] in per_item]
            good = [r for r in trecs if not r["error"]]
            costs = [r["cost"] for r in trecs if r["cost"] is not None]
            stab = None
            if cand in ec.SHUFFLED:
                stab = ec.order_stability({i: [(r["answer"], r["p"]) for r in per_item[i]] for i in attempted})
            task_out[task] = {
                "score": s if task == "bj" else s["doubt"], "coverage": {"attempted": len(attempted), "total": len(idx)},
                "note": None if recs else "not run (aborted or skipped)",
                "calls": len(trecs), "errored_calls": len(trecs) - len(good),
                "latency": ec.latency_summary([r["wall"] for r in good if r["wall"] is not None]),
                "cost_per_1000_calls": ec.cost_per_1000([r["cost"] for r in good]),
                "cost_per_decision": sum(costs) / len(attempted) if attempted and costs else None,
                "cost_estimated": any(r["cost_estimated"] for r in trecs),
                "stability": stab, **ec.separation(kind, tits, tres),
                "tau_sweep": ec.tau_sweep(tits, [sweep_in[i] for i in idx], TAUS) if any(p is not None for _, p in sweep_in) else None}
        out[cand] = task_out
    return out


def lang_line(by_lang, a, b):
    return " ".join(f"{lang} {pct(v[a], v[b])}" for lang, v in sorted(by_lang.items()))


def render(summary):
    L = []
    for cand, tasks in summary.items():
        for task, t in tasks.items():
            s, lat = t["score"], t["latency"]
            L.append(f"## {cand} / {task} (coverage {t['coverage']['attempted']}/{t['coverage']['total']} items, {t['calls']} calls, {t['errored_calls']} errored)")
            if t["note"]:
                L.append(f"- {t['note']}")
            if task == "bj":
                L.append(f"- right move {pct(s['move_pass'], s['move_n'])} ({s['move_pass']}/{s['move_n']}); silent on chatter {pct(s['talk_silent'], s['talk_n'])} ({s['talk_silent']}/{s['talk_n']})")
                L.append(f"- right move by language: {lang_line(s['by_lang'], 'move_pass', 'move_n')}; silent by language: {lang_line(s['by_lang'], 'talk_silent', 'talk_n')}")
            else:
                L.append(f"- doubt recall {pct(s['recall'], s['doubt_n'])} ({s['recall']}/{s['doubt_n']}); silent on non-doubts {pct(s['silent'], s['talk_n'])} ({s['silent']}/{s['talk_n']})")
                L.append(f"- recall by language: {lang_line(s['by_lang'], 'recall', 'doubt_n')}; silent by language: {lang_line(s['by_lang'], 'silent', 'talk_n')}")
            L.append(f"- invalid {s['invalid']}, refusals {s['refusals']}, errored calls {t['errored_calls']}")
            est = " (partly estimated)" if t["cost_estimated"] else ""
            cost = f"{num(t['cost_per_1000_calls'], '${:.4f}')} per 1,000 calls, {num(t['cost_per_decision'], '${:.6f}')} per decision{est}"
            L.append(f"- latency p50 {num(lat['p50'], '{:.3f}')} s, p95 {num(lat['p95'], '{:.3f}')} s (n={lat['n']}); cost {cost}")
            if t["stability"]:
                L.append(f"- order stability: {num(t['stability']['stable_share'])} of items identical across orders; mean spread of top p {num(t['stability']['mean_p_spread'], '{:.3f}')}")
            L.append(f"- AUROC of top probability (correctness; move-vs-talk for predicate/noul): {num(t['auroc'], '{:.3f}')}")
            if any(b["n"] for b in t["buckets"]):
                L.append("- accuracy by probability bucket (accuracy, n): " + "; ".join(f"{b['lo']}-{b['hi']}: {num(b['accuracy'])}, {b['n']}" for b in t["buckets"]))
            if t["tau_sweep"]:
                L.append("- tau sweep (tau: move right / talk silent): " + "; ".join(f"{x['tau']}: {num(x['move_rate'])} / {num(x['talk_silent_rate'])}" for x in t["tau_sweep"]))
            L.append("")
    return "\n".join(L)


def main(argv=None, env=None, post=None):
    p = argparse.ArgumentParser(description="FEAT-43 decision-model evaluation (dry run unless --run)")
    p.add_argument("--candidates", default=",".join(ec.CANDIDATES))
    p.add_argument("--shuffles", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="items in total, stratified over the combined list (0 = all 311)")
    p.add_argument("--out", default=str(Path.home() / "decision-eval-out"))
    p.add_argument("--run", action="store_true", help="make the paid calls (default: dry run)")
    p.add_argument("--max-usd", type=float, default=0.50)
    p.add_argument("--tau", type=float, default=0.5)
    a = p.parse_args(argv)
    env = os.environ if env is None else env
    post = post or http_post
    cands = [c.strip() for c in a.candidates.split(",") if c.strip()]
    bad = [c for c in cands if c not in ec.CANDIDATES]
    if bad or not cands or a.shuffles < 1:
        print(f"bad arguments: unknown candidates {bad}, valid: {', '.join(ec.CANDIDATES)}" if bad else "bad arguments", file=sys.stderr)
        return 2
    out = Path(a.out).resolve()
    if out == ec.ROOT or ec.ROOT in out.parents:
        print(f"refusing --out inside the repository: {out}", file=sys.stderr)
        return 2
    items = ec.stratified(ec.load_items(), a.limit)
    est = ec.estimate(cands, items, a.shuffles)
    print(f"{len(items)} items, shuffles {a.shuffles}")
    for c in cands:
        print(f"  {c:14s} {est['calls'][c]:5d} calls  ~${est['per_candidate_usd'][c]:.4f}")
    print(f"total {est['total_calls']} calls, estimated ${est['est_usd']:.4f} (prices: see README; budget --max-usd {a.max_usd:.2f})")
    over = est["est_usd"] > a.max_usd
    if over:
        print(f"estimate ${est['est_usd']:.4f} exceeds --max-usd {a.max_usd:.2f}")
    if not a.run:
        print("dry run: no request made. Add --run to call the providers.")
        return 0
    if over:
        print(f"refusing to run: estimate ${est['est_usd']:.4f} exceeds --max-usd {a.max_usd:.2f}; raise the budget or reduce --limit/--shuffles", file=sys.stderr)
        return 2
    missing = sorted({ec.CANDIDATES[c]["key_env"] for c in cands if not env.get(ec.CANDIDATES[c]["key_env"])})
    if missing:
        print(f"missing environment variable(s): {', '.join(missing)}", file=sys.stderr)
        return 2
    keys = {c: env[ec.CANDIDATES[c]["key_env"]] for c in cands}
    out.mkdir(parents=True, exist_ok=True)
    records, state = [], {"aborted": False}

    def meta():
        return {"candidates": cands, "shuffles": a.shuffles, "tau": a.tau, "items": len(items), "aborted": state["aborted"],
                "spent_usd": sum(r["cost"] or 0.0 for r in records)}
    saved = True
    try:
        state["aborted"] = run_calls(cands, items, a, keys, post, records, lambda: save(out, meta(), records))
    finally:
        try:
            save(out, meta(), records)
        except OSError as e:
            saved = False
            print(f"could not write {out / 'results.json'}: {type(e).__name__}", file=sys.stderr)
    for c in cands:
        mine = [r for r in records if r["candidate"] == c]
        if mine and all(r["answer"] is None for r in mine):
            print(f"WARNING: {c}: every answer was invalid (errors, refusals or malformed); check the request or the key", file=sys.stderr)
    try:
        summary = summarize(cands, items, records, a.tau)
        text = render(summary)
        (out / "summary.json").write_text(json.dumps({"meta": meta(), "summary": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        print(f"summary failed ({type(e).__name__}: {redact(str(e), keys.values())[:200]}); results.json is kept in {out}", file=sys.stderr)
        return 1
    print(text)
    if not saved:
        return 1
    print(f"spent ${meta()['spent_usd']:.4f}; results in {out}")
    return 3 if state["aborted"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
