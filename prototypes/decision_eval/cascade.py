"""FEAT-44: e5-first cascade replay. A local e5 reading (band + moves) is optionally re-asked of a decision model; the model's
probabilities then drop or re-order the e5 moves. Nothing here calls a provider: dump-e5 runs the shipped local reader, replay
scores recorded FEAT-43 answers against the e5 bands.

Pure functions (standard library only): apply_policy, reading_of, evaluate, cost_per_1000_messages. Tests: tests/test_decision_cascade.py.
Usage:
  python cascade.py dump-e5 --model-dir DIR --out FILE        (needs onnxruntime, numpy, tokenizers; --out outside the repo)
  python cascade.py replay --e5 FILE --results FILE [--taus 0.5,0.7,0.9] [--out DIR]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import evalcore as ec  # noqa: E402

READERS = ("or-d1-noul", "oai-predicate", "or-haiku")
COVERS = ("unsure", "sure+unsure")


def apply_policy(e5, reader, policy):
    """Final {'band','moves','called'} for one row. policy: {'cover': 'unsure'|'sure+unsure' ('all' is accepted as the same), 'tau'}. A reader of None (no usable answer)
    keeps the e5 reading but still counts as a call. Moves the reader scores below tau are dropped (a move it does not mention
    scores 0), the rest are ordered by its probability (ties keep e5 order); none left means silent."""
    band, moves = e5["band"], list(e5.get("moves", []))
    if band == "silent" or (policy["cover"] == "unsure" and band != "unsure"):
        return {"band": band, "moves": moves, "called": False}
    if reader is None:
        return {"band": band, "moves": moves, "called": True}
    probs = reader["probs"]
    kept = sorted((m for m in moves if probs.get(m, 0.0) >= policy["tau"]), key=lambda m: -probs.get(m, 0.0))
    if not kept:
        return {"band": "silent", "moves": [], "called": True}
    sure = band == "sure" and kept == moves and len(kept) == 1
    return {"band": "sure" if sure else "unsure", "moves": kept, "called": True}


def reading_of(record):
    """The reader's {'probs'} from one results.json record; None for a missing or errored record."""
    if not record or record.get("error"):
        return None
    return {"probs": dict(record.get("probs") or {})}


def _right(item, moves):
    return item["label"] in moves or any(m in moves for m in item["accept"])


def evaluate(items, finals):
    """Counts over the rows; a row with no final counts as silent. talk_silent: no buttons; talk_silent_or_accepted: no buttons or an
    accepted move among them; sure_wrong: a sure band that is not right (on a talk row only an accepted move is right);
    sure_wrong_moves and sure_wrong_talk split it by row kind (sure_wrong_moves is what bands.evaluate counts)."""
    r = {"move_pass": 0, "move_n": 0, "talk_silent": 0, "talk_silent_or_accepted": 0, "talk_n": 0, "sure_wrong": 0, "sure_wrong_moves": 0, "sure_wrong_talk": 0, "by_lang": {}, "calls": 0}
    for it in items:
        f = finals.get(it["id"]) or {"band": "silent", "moves": [], "called": False}
        moves = f["moves"]
        r["calls"] += bool(f["called"])
        accepted = any(m in moves for m in it["accept"])
        if it["label"] == "talk":
            r["talk_n"] += 1
            r["talk_silent"] += not moves
            r["talk_silent_or_accepted"] += not moves or accepted
            ok = accepted
        else:
            ok = _right(it, moves)
            lang = r["by_lang"].setdefault(it["lang"], [0, 0])
            lang[0] += ok
            lang[1] += 1
            r["move_n"] += 1
            r["move_pass"] += ok
        bad = f["band"] == "sure" and not ok
        r["sure_wrong"] += bad
        r["sure_wrong_talk" if it["label"] == "talk" else "sure_wrong_moves"] += bad
    r["calls_per_row"] = r["calls"] / len(items) if items else 0
    return r


def cost_per_1000_messages(calls, rows, per_call_usd):
    return calls / rows * 1000 * per_call_usd if rows else 0.0


# --- dump-e5 -------------------------------------------------------------------------------------------------------------

def inside_repo(path):
    path = Path(path).resolve()
    return path == ec.ROOT or ec.ROOT in path.parents


def dump_e5(items, model_dir):
    """{item_id: {'band','moves','features','source'}} as games_spoken_discord._read does for blackjack: embed the text (the embedder adds
    the 'query: ' prefix), interpret.decide over the row's legal moves with the shipped head and no server examples. The EN/KO word
    matcher (spoken/words.py) is not part of the blackjack path, so every row's source is 'embed'."""
    sys.path.insert(0, str(ec.ROOT))
    from llmcord_core.games_spoken_discord import MAX_CHARS
    from llmcord_core.spoken import blackjack as bj, head as spoken_head, interpret
    from llmcord_core.spoken.onnx_embedder import OnnxEmbedder
    embedder = OnnxEmbedder(model_dir).load()  # verifies the pinned hashes
    head = spoken_head.blackjack_head()
    out = {}
    for it in items:
        text = it["text"].strip()  # as games_spoken_discord.note_message: strip, then ignore empty or over MAX_CHARS
        if not text or len(text) > MAX_CHARS:
            out[it["id"]] = {"band": "silent", "moves": [], "source": "too_long", "features": None}
            continue
        vector = embedder.embed([text])[0]
        d = interpret.decide(vector, [m for m in it["legal"] if m in bj.MOVES], head, ())
        f = d.features
        out[it["id"]] = {"band": d.band, "moves": list(d.moves), "source": "embed",
                         "features": {"p1": f.p1, "margin": f.margin, "talk_gap": f.talk_gap} if f else None}
    return out


def finals_of(e5_rows):
    return {i: {"band": e["band"], "moves": list(e["moves"]), "called": False} for i, e in e5_rows.items()}


def pct(n, d):
    return f"{n / d:.0%}" if d else "n/a"


def cmd_dump(a):
    if inside_repo(a.out):
        print(f"refusing --out inside the repository: {Path(a.out).resolve()}", file=sys.stderr)
        return 2
    items = [it for it in ec.load_items() if it["task"] == "bj"]
    rows = dump_e5(items, a.model_dir)
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    r = evaluate(items, finals_of(rows))
    bands = {b: sum(1 for e in rows.values() if e["band"] == b) for b in ("sure", "unsure", "silent")}
    print(f"{len(rows)} rows -> {a.out}; bands {bands}")
    print(f"e5 alone: right move {pct(r['move_pass'], r['move_n'])} ({r['move_pass']}/{r['move_n']}), silent on chatter "
          f"{pct(r['talk_silent'], r['talk_n'])} ({r['talk_silent']}/{r['talk_n']}), sure_wrong {r['sure_wrong']} "
          f"(moves {r['sure_wrong_moves']}, talk {r['sure_wrong_talk']}), "
          f"by language " + " ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(r["by_lang"].items())))
    return 0


# --- replay --------------------------------------------------------------------------------------------------------------

def reader_inputs(records, reader, items):
    """({item_id: reading or None}, per-call USD, 'measured'|'estimated') from the k=0 bj records of one reader. A reader that gives only
    an answer (chat) is read as probability 1.0 on that move and 0.0 on the other legal moves; an answer of talk or none is no move."""
    mine, dup = {}, 0
    for r in records:
        if r["candidate"] == reader and r["task"] == "bj" and r["k"] == 0:
            dup += r["item"] in mine
            mine[r["item"]] = r
    if dup:
        print(f"warning: {reader} has {dup} duplicate k=0 bj record(s); keeping the last", file=sys.stderr)
    legal = {it["id"]: it["legal"] for it in items}
    out = {}
    for i, rec in mine.items():
        reading = reading_of(rec)
        if reading is not None and not reading["probs"]:
            if rec.get("answer") is None:
                reading = None  # invalid answer: fail open like an error
            else:
                reading = {"probs": {m: 1.0 if m == rec["answer"] else 0.0 for m in legal.get(i, [])}}
        out[i] = reading
    costs = [r["cost"] for r in mine.values() if r.get("cost") is not None and not r.get("error")]
    est = any(r.get("cost_estimated") for r in mine.values())
    if costs:
        per_call = sum(costs) / len(costs)
    else:
        bj_items = [it for it in items if it["task"] == "bj"]
        per_call = sum(ec.est_call_usd(reader, ec.build_request(reader, it, 0)["json"]) for it in bj_items) / len(bj_items) if bj_items else 0.0
    return out, per_call, "estimated" if est or not costs else "measured"


def replay(items, e5_rows, records, taus):
    """[{'reader','cover','tau','result','per_call','price','finals'}], e5 alone first (reader None)."""
    table = [{"reader": None, "cover": "-", "tau": None, "result": evaluate(items, finals_of(e5_rows)), "per_call": 0.0, "price": "-",
              "finals": finals_of(e5_rows)}]
    for reader in READERS:
        readings, per_call, price = reader_inputs(records, reader, items)
        for cover in COVERS:
            for tau in taus:
                finals = {it["id"]: apply_policy(e5_rows[it["id"]], readings.get(it["id"]), {"cover": cover, "tau": tau}) for it in items}
                table.append({"reader": reader, "cover": cover, "tau": tau, "result": evaluate(items, finals), "per_call": per_call,
                              "price": price, "finals": finals})
    return table


def render(items, table):
    n = len(items)
    L = ["| reader | cover | tau | right move | silent on chatter | sure wrong moves/talk | ko move rows right | calls (% of rows) | $ per 1,000 msgs |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for t in table:
        r = t["result"]
        ko = r["by_lang"].get("ko", [0, 0])
        cost = f"{cost_per_1000_messages(r['calls'], n, t['per_call']):.4f}" + ("" if t["price"] == "measured" else " (est.)") if t["reader"] else "0"
        L.append(f"| {t['reader'] or 'e5 alone'} | {t['cover']} | {t['tau'] if t['tau'] is not None else '-'} | {pct(r['move_pass'], r['move_n'])} "
                 f"({r['move_pass']}/{r['move_n']}) | {pct(r['talk_silent'], r['talk_n'])} ({r['talk_silent']}/{r['talk_n']}) | {r['sure_wrong_moves']}/{r['sure_wrong_talk']} | "
                 f"{ko[0]}/{ko[1]} | {r['calls']} ({pct(r['calls'], n)}) | {cost} |")
    L.append("")
    L.append("sure wrong = rows left in the sure band that are not right, split into move rows (what bands.evaluate counts) and chatter rows "
             "(on chatter only an accepted move is right). cover sure+unsure = ask whenever e5 is not silent.")
    L.append("Caveats: the e5 head trains on examples.json/talk_examples.json and phrases.csv is the labelled test set, but the same author wrote "
             "both, so e5 numbers are optimistic; the decision models are zero-shot on the same rows. The tau grid is searched on the same 195 rows, "
             "so no tau here is validated. The cascade can only drop or reorder e5's moves: it never rescues a message e5 called silent and never "
             "raises unsure to sure, so recall can only fall.")
    L.append("Cost = calls per row x 1,000 x $ per call; $ per call is the mean cost recorded for the reader's k=0 blackjack calls "
             "(provider-reported unless marked est., where the harness estimate was used).")
    base = table[0]["finals"]
    by_id = {it["id"]: it for it in items}
    leaked = [i for i, it in by_id.items() if it["label"] == "talk" and base[i]["moves"]]
    L += ["", f"Chatter e5 let through ({len(leaked)}): silenced by (reader cover@tau)"]
    for i in leaked:
        who = [f"{t['reader']} {t['cover']}@{t['tau']}" for t in table[1:] if not t["finals"][i]["moves"]]
        L.append(f"- {i} \"{by_id[i]['text']}\" (e5 {base[i]['band']} {','.join(base[i]['moves'])}): " + (", ".join(who) or "none"))
    L += ["", "Move rows each policy loses (right under e5 alone, wrong after):"]
    for t in table[1:]:
        lost = [i for i, it in by_id.items() if it["label"] != "talk" and _right(it, base[i]["moves"]) and not _right(it, t["finals"][i]["moves"])]
        L.append(f"- {t['reader']} {t['cover']}@{t['tau']}: {len(lost)}" + (" " + " ".join(lost) if lost else ""))
    return "\n".join(L)


def cmd_replay(a):
    try:
        taus = [float(x) for x in a.taus.split(",") if x.strip()]
    except ValueError:
        taus = []
    if not taus:
        print("bad --taus", file=sys.stderr)
        return 2
    items = [it for it in ec.load_items() if it["task"] == "bj"]
    e5_rows = json.loads(Path(a.e5).read_text(encoding="utf-8"))
    records = json.loads(Path(a.results).read_text(encoding="utf-8"))["records"]
    missing = [it["id"] for it in items if it["id"] not in e5_rows]
    if missing:
        print(f"e5 file lacks {len(missing)} rows, e.g. {missing[:3]}", file=sys.stderr)
        return 2
    table = replay(items, e5_rows, records, taus)
    text = render(items, table)
    print(text)
    if a.out:
        out = Path(a.out).resolve()
        if inside_repo(out):
            print(f"refusing --out inside the repository: {out}", file=sys.stderr)
            return 2
        out.mkdir(parents=True, exist_ok=True)
        (out / "cascade.md").write_text(text + "\n", encoding="utf-8")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="FEAT-44 e5-first cascade replay (offline)")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump-e5", help="shipped local reading of the 195 blackjack rows")
    d.add_argument("--model-dir", required=True)
    d.add_argument("--out", required=True)
    r = sub.add_parser("replay", help="score recorded reader answers against the e5 bands")
    r.add_argument("--e5", required=True)
    r.add_argument("--results", required=True)
    r.add_argument("--taus", default="0.5,0.7,0.9")
    r.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Korean phrases on a cp1252 console
    return cmd_dump(a) if a.cmd == "dump-e5" else cmd_replay(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
