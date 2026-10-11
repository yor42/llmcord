"""FEAT-36 spoken-moves prototype harness: fetch two small embedding models, evaluate the cascade, benchmark latency.

Run with the prototype venv: prototypes/spoken_moves/.venv/bin/python prototypes/spoken_moves/run.py {fetch,eval,bench}.
"""
import argparse
import csv
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

import bands  # noqa: E402
import wordrule  # noqa: E402
from llmcord_core.spoken import words  # noqa: E402

CACHE = Path.home() / ".cache" / "llmcord-spoken"
LOCK = HERE / "models.lock.json"
HF = "https://huggingface.co"
MODELS = {
    "e5": {"id": "intfloat/multilingual-e5-small", "prefix": "query: ", "run_file": "model_int8.onnx",
           "files": {"onnx/model.onnx": "model.onnx", "tokenizer.json": "tokenizer.json"}},
    "minilm": {"id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "prefix": "",
               "run_file": "model_qint8_arm64.onnx",
               "files": {"onnx/model_qint8_arm64.onnx": "model_qint8_arm64.onnx", "tokenizer.json": "tokenizer.json"}},
}


def model_dir(name):
    return CACHE / MODELS[name]["id"].replace("/", "__")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def http_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def download(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def selected(arg):
    return list(MODELS) if arg == "both" else [arg]


# fetch

def cmd_fetch(args):
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {"models": {}}
    for name in selected(args.model):
        spec, d = MODELS[name], model_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        entry = lock["models"].get(spec["id"])
        if entry:
            rev = entry["revision"]
        else:
            info = http_json(f"{HF}/api/models/{spec['id']}/revision/main")
            rev, siblings = info["sha"], {s["rfilename"] for s in info["siblings"]}
            missing = [p for p in spec["files"] if p not in siblings]
            if missing:
                sys.exit(f"{spec['id']}@{rev}: not in the repo: {missing}")
            entry = lock["models"][spec["id"]] = {"revision": rev, "files": {}}
        for path, local in spec["files"].items():
            dest, want = d / local, entry["files"].get(path)
            if dest.exists() and (want is None or sha256(dest) == want):
                got = sha256(dest)
            else:
                print(f"{spec['id']}: downloading {path} @ {rev[:8]}", flush=True)
                download(f"{HF}/{spec['id']}/resolve/{rev}/{path}", dest)
                got = sha256(dest)
            if want and got != want:
                dest.unlink()
                sys.exit(f"{spec['id']}/{path}: sha256 {got} does not match models.lock.json ({want})")
            entry["files"][path] = got
        if name == "e5":
            out = d / spec["run_file"]
            if not out.exists():
                from onnxruntime.quantization import QuantType, quantize_dynamic
                print("e5: quantizing to int8 (dynamic, QInt8)", flush=True)
                quantize_dynamic(str(d / "model.onnx"), str(out), weight_type=QuantType.QInt8)
            entry["derived"] = {"model_int8.onnx": sha256(out)}
        print(f"{spec['id']}: ok @ {rev}")
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")


# embedding

class Embedder:
    def __init__(self, name, threads=1):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.np, self.prefix = np, MODELS[name]["prefix"]
        d = model_dir(name)
        if not (d / MODELS[name]["run_file"]).exists():
            sys.exit(f"{name}: model files missing, run `run.py fetch` first")
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(128)
        self.tok.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = threads, 1
        self.session = ort.InferenceSession(str(d / MODELS[name]["run_file"]), opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def embed(self, texts):
        np = self.np
        encs = self.tok.encode_batch([self.prefix + t for t in texts])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feed)[0]
        m = mask[:, :, None].astype(hidden.dtype)
        pooled = (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)
        return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)


def embed_all(emb, texts, batch=16):
    np = emb.np
    return np.concatenate([emb.embed(texts[i:i + batch]) for i in range(0, len(texts), batch)])


# data

def load_data():
    examples = json.loads((HERE / "examples.json").read_text(encoding="utf-8"))
    extra = HERE / "talk_examples.json"
    talk_extra = json.loads(extra.read_text(encoding="utf-8"))["table_talk"] if extra.exists() else []
    wordlist = json.loads((HERE / "wordlist.json").read_text(encoding="utf-8"))
    with open(HERE / "phrases.csv", encoding="utf-8", newline="") as f:
        phrases = list(csv.DictReader(f))
    return examples, talk_extra, wordlist, phrases


def merged_talk(examples, talk_extra, use_extra):
    out = list(examples["table_talk"])
    if use_extra:
        seen = {words.normalize(x) for x in out}
        out += [x for x in talk_extra if words.normalize(x) not in seen and not seen.add(words.normalize(x))]
    return out


def word_reading(text, legal, wordlist, negations):
    rd = words.read(text, wordlist, kinds=legal, negations=negations)
    if rd is None:
        return None, 0
    kinds = [c.kind for c in rd.candidates]
    return ("sure" if rd.sure else "unsure", kinds), wordrule.extra_tokens(text, wordlist, kinds)


# head: multinomial logistic regression (numpy), moves + talk, trained on examples only

def train_head(np, X, y, k, lam, steps=500, lr=0.05):
    """Adam on class-balanced softmax cross-entropy + lam/2 ||W||^2 over standardized features."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    counts = np.bincount(y, minlength=k).astype(float)
    w = (len(y) / (k * np.maximum(counts, 1)))[y]
    w = w / w.sum()
    Y = np.eye(k)[y]
    W, b = np.zeros((X.shape[1], k)), np.zeros(k)
    m = [np.zeros_like(W), np.zeros_like(b)]
    v = [np.zeros_like(W), np.zeros_like(b)]
    for t in range(1, steps + 1):
        P = softmax(np, Z @ W + b)
        G = (P - Y) * w[:, None]
        grads = [Z.T @ G + lam * W, G.sum(0)]
        for i, (p, g) in enumerate(zip((W, b), grads)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            p -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return {"mu": mu, "sd": sd, "W": W, "b": b}


def softmax(np, L):
    L = L - L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def head_logits(np, head, X):
    return ((X - head["mu"]) / head["sd"]) @ head["W"] + head["b"]


def head_proba(np, head, X):
    return softmax(np, head_logits(np, head, X) / head.get("T", 1.0))


def fit_head(np, ex_vecs, talk_vecs, select="logloss"):
    """Pick lam by 5-fold CV on the examples (never phrases.csv), then train on all of them.

    select 'logloss': lam in 1e-3..1 by held-out log loss. 'acc': wider lam range by held-out accuracy (ties: log
    loss), then a temperature fitted on the pooled held-out logits."""
    classes = sorted(ex_vecs) + ["talk"]
    X = np.concatenate([ex_vecs[c] for c in classes[:-1]] + [talk_vecs])
    y = np.array(sum(([i] * len(ex_vecs[c]) for i, c in enumerate(classes[:-1])), []) + [len(classes) - 1] * len(talk_vecs))
    folds = bands.stratified_folds([{"label": int(c)} for c in y], 5, 0)
    lams = (1e-3, 1e-2, 1e-1, 1.0) if select == "logloss" else (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    best = None
    for lam in lams:
        loss, correct, held_logits = 0.0, 0, np.zeros((len(y), len(classes)))
        for held in folds:
            tr = np.array(sorted(set(range(len(y))) - set(held)))
            h = train_head(np, X[tr], y[tr], len(classes), lam)
            held_logits[held] = head_logits(np, h, X[held])
            P = softmax(np, held_logits[held])
            loss += -np.log(P[np.arange(len(held)), y[held]] + 1e-9).mean()
            correct += int((P.argmax(1) == y[held]).sum())
        key = (-loss,) if select == "logloss" else (correct, -loss)
        if best is None or key > best[0]:
            best = (key, lam, held_logits)
    head = train_head(np, X, y, len(classes), best[1])
    if select == "acc":
        grid = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
        nll = [-np.log(softmax(np, best[2] / t)[np.arange(len(y)), y] + 1e-9).mean() for t in grid]
        head["T"] = grid[int(np.argmin(nll))]
    return head, classes, best[1]


class Prepared:
    """Embeddings of examples and phrases for one model and one talk-example choice, plus the trained head."""

    def __init__(self, emb, examples, talk, phrases, wordlist):
        np = emb.np
        self.np, self.phrases, self.wordlist = np, phrases, wordlist
        self.negations = json.loads((ROOT / "llmcord_core/spoken/lexicon/common.json").read_text())["negations"]
        self.ex_vecs = {m: embed_all(emb, xs) for m, xs in examples["examples"].items()}
        self.talk_vecs = embed_all(emb, talk)
        self.vecs = embed_all(emb, [p["text"] for p in phrases])
        self.heads, self.lams = {}, {}
        for mode, select in (("head", "logloss"), ("head2", "acc")):
            self.heads[mode], self.classes, self.lams[mode] = fit_head(np, self.ex_vecs, self.talk_vecs, select)
        self.lam = self.lams["head"]

    def rows(self, mode):
        np = self.np
        probs = head_proba(np, self.heads[mode], self.vecs) if mode != "knn" else None
        rows = []
        for n, (p, v) in enumerate(zip(self.phrases, self.vecs)):
            legal = p["legal"].split()
            if mode != "knn":
                scores, talk = bands.mask_renorm(dict(zip(self.classes, probs[n].tolist())), legal)
            else:
                scores = {m: float(np.max(self.ex_vecs[m] @ v)) for m in legal if m in self.ex_vecs}
                talk = float(np.max(self.talk_vecs @ v))
            word, extra = word_reading(p["text"], legal, self.wordlist, self.negations)
            rows.append({"id": p["id"], "text": p["text"], "lang": p["lang"], "label": p["label"],
                         "accept": tuple(p["accept"].split()), "scores": scores, "talk": talk,
                         "word": word, "extra": extra})
        return rows


# eval

def pct(n, d):
    return f"{100 * n / d:.1f}% ({n}/{d})" if d else "n/a"


def fmt_th(th):
    return ", ".join(f"{k}={v:.3f}" for k, v in th.items())


def breakdown(m):
    lines = ["| group | pass (move rows: right move among buttons; talk rows: silent or accepted button) |", "| --- | --- |"]
    for key in ("by_lang", "by_label"):
        for name, (n, p) in sorted(m[key].items()):
            lines.append(f"| {key[3:]} {name} | {pct(p, n)} |")
    return lines


def failures(m):
    if not m["failures"]:
        return ["_none_"]
    lines = ["| id | text | label | band | buttons | s1 | margin | talk_gap |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r, band, buttons in m["failures"]:
        _, _, s1, mg, gp = bands.features(r["scores"], r["talk"])
        text = r["text"].replace("|", "\\|")
        lines.append(f"| {r['id']} | {text} | {r['label']} | {band} | {', '.join(buttons) or '-'} | {s1:.3f} | {mg:.3f} | {gp:.3f} |")
    return lines


def summary_line(m):
    return (f"move pass {pct(m['move_pass'], m['move_n'])}; talk silent {pct(m['talk_silent'], m['talk_n'])}; "
            f"sure-band wrong moves {m['sure_wrong']} (right {m['sure_right']})")


WORD_OPTIONS = (("no word list", False), ("word N=1", 1))
FLOORS = ((0.95, 0.95), (0.93, 0.93), (0.90, 0.90), (0.95, 0.97), (0.93, 0.95), (0.90, 0.92))  # (target, train floor)
MODES = ("knn", "head", "head2")
CUTOFFS = (0.95, 0.93, 0.91)


def grams(text):
    t = words.normalize(text)
    return {t[i:i + 3] for i in range(max(len(t) - 2, 1))}


def near_pairs(emb, examples, talk_extra, phrases):
    """Every example with its closest phrases.csv row by e5 cosine and by char-3-gram Jaccard."""
    items = [(m, x) for m, xs in examples["examples"].items() for x in xs]
    items += [("talk", x) for x in examples["table_talk"]] + [("talk+", x) for x in talk_extra]
    V, P = embed_all(emb, [x for _, x in items]), embed_all(emb, [p["text"] for p in phrases])
    sims = V @ P.T
    pg = [grams(p["text"]) for p in phrases]
    out = []
    for (kind, text), row in zip(items, sims):
        g = grams(text)
        jac = [len(g & q) / len(g | q) for q in pg]
        ci, ji = int(row.argmax()), max(range(len(pg)), key=jac.__getitem__)
        out.append({"kind": kind, "text": text, "cos": float(row[ci]), "cos_id": phrases[ci]["id"],
                    "cos_text": phrases[ci]["text"], "jac": jac[ji]})
    return out


def dedupe_report(pairs):
    """Markdown lines and the dropped-text set (strictest of cos >= 0.91 and jaccard >= 0.5, whichever drops more)."""
    out = ["| cutoff | examples dropped | move examples | talk examples |", "| --- | --- | --- | --- |"]
    sets = {}
    for label, pred in [(f"cosine >= {c}", (lambda p, c=c: p["cos"] >= c)) for c in CUTOFFS] + [("jaccard >= 0.5", lambda p: p["jac"] >= 0.5)]:
        drop = [p for p in pairs if pred(p)]
        sets[label] = {(p["kind"], p["text"]) for p in drop}
        out.append(f"| {label} | {len(drop)} / {len(pairs)} | {sum(p['kind'] not in ('talk', 'talk+') for p in drop)} | "
                   f"{sum(p['kind'] in ('talk', 'talk+') for p in drop)} |")
    pick = max(("cosine >= 0.91", "jaccard >= 0.5"), key=lambda k: len(sets[k]))
    out += ["", f"Deduped run drops by: {pick} ({len(sets[pick])} examples).", "", "20 closest pairs by cosine:", "",
            "| example (kind) | closest phrases.csv row | cosine | jaccard (to its own closest) |", "| --- | --- | --- | --- |"]
    for p in sorted(pairs, key=lambda p: -p["cos"])[:20]:
        a, b = p["text"].replace("|", "\\|"), p["cos_text"].replace("|", "\\|")
        out.append(f"| {a} ({p['kind']}) | {p['cos_id']}: {b} | {p['cos']:.3f} | {p['jac']:.2f} |")
    mild = sets["cosine >= 0.95"] | sets["jaccard >= 0.5"]
    out += ["", f"A milder second deduped run drops cosine >= 0.95 or jaccard >= 0.5 ({len(mild)} examples)."]
    return out, sets[pick], mild


def drop_examples(examples, talk_extra, dropped):
    ex = {"examples": {m: [x for x in xs if (m, x) not in dropped] for m, xs in examples["examples"].items()},
          "table_talk": [x for x in examples["table_talk"] if ("talk", x) not in dropped]}
    return ex, [x for x in talk_extra if ("talk+", x) not in dropped]


def matrix(emb, examples, talk_extra, wordlist, phrases, title):
    variants = [False] + ([True] if talk_extra else [])
    out = [f"### {title}", "",
           "Cell = CV move pass / CV talk silent, achieved; `*below*` marks CV move pass under the target; (k/5) = training folds "
           "where the fit reached its train floor. Columns 4-6 train at target + 2 points.", "",
           "| talk examples | mode | word rule | " + " | ".join(
               f"{round(t * 100)}%" + (f" (train {round(f * 100)}%)" if f != t else "") for t, f in FLOORS) + " |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    heads = []
    for use_extra in variants:
        prep = Prepared(emb, examples, merged_talk(examples, talk_extra, use_extra), phrases, wordlist)
        heads.append(f"- {'with' if use_extra else 'without'} talk examples ({len(prep.talk_vecs)}): lambda head={prep.lams['head']}, "
                     f"head2={prep.lams['head2']}, head2 T={prep.heads['head2'].get('T')}")
        for mode in MODES:
            rows = prep.rows(mode)
            for wlabel, cascade in WORD_OPTIONS:
                cells = []
                for target, train in FLOORS:
                    cv = bands.cross_validate(rows, cascade, min_move=train)[0]
                    flag = " *below*" if cv["move_rate"] < target - 1e-9 else ""
                    cells.append(f"{100 * cv['move_rate']:.1f} / {100 * cv['talk_rate']:.1f}{flag} ({cv['folds_feasible']}/5)")
                te = f"with ({len(prep.talk_vecs)})" if use_extra else f"without ({len(prep.talk_vecs)})"
                out.append(f"| {te} | {mode} | {wlabel} | " + " | ".join(cells) + " |")
    return out + [""] + heads + [""]


def eval_model(name, examples, talk_extra, wordlist, phrases, dedupe):
    emb = Embedder(name)
    out = [f"## {MODELS[name]['id']} ({name})", "",
           "CV = 5-fold stratified over phrases.csv; thresholds fitted on the training folds; kNN and the heads never see phrases.csv. "
           "knn = nearest example; head = logistic regression (lambda by log loss); head2 = lambda by accuracy over a wider range "
           "plus temperature scaling fitted on the example CV folds.", "",
           "Noise: 1 talk row = 1.85 points of talk silence, 1 move row = 1.04 points of move pass (n = 54 talk, 96 moves); "
           "differences of a few points between cells are not meaningful.", ""]
    out += matrix(emb, examples, talk_extra, wordlist, phrases, "Original examples")
    if dedupe:
        pairs = near_pairs(emb, examples, talk_extra, phrases)
        lines, dropped, mild = dedupe_report(pairs)
        out += ["### Near-duplicates of phrases.csv among the examples", ""] + lines + [""]
        runs = [("Deduped examples (strictest: " + str(len(dropped)) + " dropped)", dropped),
                ("Deduped examples (mild: cosine >= 0.95 or jaccard >= 0.5, " + str(len(mild)) + " dropped)", mild)]
        if dedupe == "talk":
            talk_only = {d for d in mild if d[0] in ("talk", "talk+")}
            runs = [(f"Deduped talk examples only (cosine >= 0.95 or jaccard >= 0.5; {len(talk_only)} talk examples dropped, "
                     "every move example kept)", talk_only)]
        for title, drop in runs:
            ex2, extra2 = drop_examples(examples, talk_extra, drop)
            out += matrix(emb, ex2, extra2, wordlist, phrases, title)
    return out


def cmd_eval(args):
    examples, talk_extra, wordlist, phrases = load_data()
    ex_norm = {words.normalize(x): m for m, xs in examples["examples"].items() for x in xs}
    ex_norm.update({words.normalize(x): "table_talk" for x in examples["table_talk"] + talk_extra})
    overlaps = [(p["id"], p["text"], ex_norm[words.normalize(p["text"])]) for p in phrases
                if words.normalize(p["text"]) in ex_norm]
    out = ["# FEAT-36 eval (second pass)", "", f"{len(phrases)} phrases, "
           f"{sum(len(v) for v in examples['examples'].values())} move examples, {len(examples['table_talk'])} table-talk examples"
           f" + {len(talk_extra)} in talk_examples.json.",
           "Gate: right move among the buttons >= 95% of real moves, silent on >= 85% of table talk, < 100 ms per message.", "",
           "Examples/phrases text overlap: " + ("none" if not overlaps else "")]
    out += [f"- OVERLAP id {i}: {t!r} also an example for {m}" for i, t, m in overlaps]
    out.append("")
    for name in selected(args.model):
        out += eval_model(name, examples, talk_extra, wordlist, phrases, args.dedupe_near)
    text = "\n".join(out) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


# bench

def bench_one(name):
    examples, talk_extra, wordlist, phrases = load_data()
    texts = [p["text"] for p in phrases]
    res = {"model": name, "threads": {}}
    emb0 = Embedder(name)
    prep = Prepared(emb0, examples, merged_talk(examples, talk_extra, True), phrases, wordlist)
    np, head = prep.np, prep.heads["head"]
    del emb0
    for threads in (1, 2, 4):
        t0 = time.perf_counter()
        emb = Embedder(name, threads)
        load = time.perf_counter() - t0
        for t in texts[:5]:
            emb.embed([t])
        lat = []
        for t in texts:
            t0 = time.perf_counter()
            emb.embed([t])
            lat.append((time.perf_counter() - t0) * 1000)
        full = []
        for t in texts:
            t0 = time.perf_counter()
            probs = head_proba(np, head, emb.embed([t]))[0]
            bands.band_of(bands.features(*bands.mask_renorm(dict(zip(prep.classes, probs.tolist())), ["hit", "stand"])),
                          {"t_sure": 0.5, "margin": 0.2, "g_sure": 0.2, "t_floor": 0.3, "g_floor": 0.0})
            full.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        full.sort()
        res["threads"][threads] = {"load_s": load, "p50": statistics.median(lat),
                                   "p95": lat[int(0.95 * (len(lat) - 1))], "max": lat[-1],
                                   "head_p50": statistics.median(full), "head_p95": full[int(0.95 * (len(full) - 1))]}
        del emb
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return res


def cmd_bench(args):
    results = []
    for name in selected(args.model):
        p = subprocess.run([sys.executable, __file__, "bench-one", name], capture_output=True, text=True, check=True)
        results.append(json.loads(p.stdout.strip().splitlines()[-1]))
    out = ["# FEAT-36 bench", "", f"{os.cpu_count()} cpus. Per message: tokenize + infer + pool, one phrase at a time, warm, "
           f"over all phrases (ms). Peak RSS is the process maximum over the three thread settings.", "",
           "With head: the same plus the logistic-regression head, masking and band decision (ms).", "",
           "| model | threads | load s | p50 ms | p95 ms | max ms | with head p50 ms | with head p95 ms | peak RSS MB |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        for th, v in r["threads"].items():
            out.append(f"| {r['model']} | {th} | {v['load_s']:.2f} | {v['p50']:.1f} | {v['p95']:.1f} | {v['max']:.1f} | {v['head_p50']:.1f} | {v['head_p95']:.1f} | {r['peak_rss_mb']:.0f} |")
    text = "\n".join(out) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("fetch", cmd_fetch), ("eval", cmd_eval), ("bench", cmd_bench)):
        p = sub.add_parser(name)
        p.add_argument("--model", choices=["e5", "minilm", "both"], default="e5")
        if name != "fetch":
            p.add_argument("--out")
        if name == "eval":
            p.add_argument("--dedupe-near", nargs="?", const="all", choices=["all", "talk"],
                           help="also report examples close to phrases.csv rows and rerun with them dropped (all: strict and mild; talk: mild, talk examples only)")
        p.set_defaults(fn=fn)
    one = sub.add_parser("bench-one")
    one.add_argument("model", choices=list(MODELS))
    one.set_defaults(fn=lambda a: print(json.dumps(bench_one(a.model))))
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
