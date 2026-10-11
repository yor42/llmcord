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
    wordlist = json.loads((HERE / "wordlist.json").read_text(encoding="utf-8"))
    with open(HERE / "phrases.csv", encoding="utf-8", newline="") as f:
        phrases = list(csv.DictReader(f))
    return examples, wordlist, phrases


def word_reading(text, legal, wordlist, negations):
    rd = words.read(text, wordlist, kinds=legal, negations=negations)
    if rd is None:
        return None
    return ("sure" if rd.sure else "unsure", [c.kind for c in rd.candidates])


def build_rows(emb, examples, wordlist, phrases):
    np = emb.np
    negations = json.loads((ROOT / "llmcord_core/spoken/lexicon/common.json").read_text())["negations"]
    ex = examples["examples"]
    ex_vecs = {m: embed_all(emb, xs) for m, xs in ex.items()}
    talk_vecs = embed_all(emb, examples["table_talk"])
    vecs = embed_all(emb, [p["text"] for p in phrases])
    rows = []
    for p, v in zip(phrases, vecs):
        legal = p["legal"].split()
        rows.append({
            "id": p["id"], "text": p["text"], "lang": p["lang"], "label": p["label"],
            "accept": tuple(p["accept"].split()),
            "scores": {m: float(np.max(ex_vecs[m] @ v)) for m in legal if m in ex_vecs},
            "talk": float(np.max(talk_vecs @ v)),
            "word": word_reading(p["text"], legal, wordlist, negations)})
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


def eval_model(name, examples, wordlist, phrases):
    emb = Embedder(name)
    rows = build_rows(emb, examples, wordlist, phrases)
    out = [f"## {MODELS[name]['id']} ({name})", ""]
    for cascade in (False, True):
        label = "cascade (word list, then embedding)" if cascade else "embedding only"
        th, ins = bands.fit(rows, cascade)
        cv, fold_ths = bands.cross_validate(rows, cascade)
        out += [f"### {label}", "", f"- 5-fold cross-validation: {summary_line(cv)}",
                f"- in-sample fit: {summary_line(ins)}", f"- fitted thresholds (all rows): {fmt_th(th)}"]
        out += [f"- fold {i + 1} thresholds: {fmt_th(t)}" for i, t in enumerate(fold_ths)]
        out += ["", "Cross-validated breakdown:", ""] + breakdown(cv)
        out += ["", "Cross-validated failing rows:", ""] + failures(cv)
        out += ["", "In-sample breakdown:", ""] + breakdown(ins)
        out += ["", "In-sample failing rows:", ""] + failures(ins) + [""]
    return out


def cmd_eval(args):
    examples, wordlist, phrases = load_data()
    ex_norm = {words.normalize(x): m for m, xs in examples["examples"].items() for x in xs}
    ex_norm.update({words.normalize(x): "table_talk" for x in examples["table_talk"]})
    overlaps = [(p["id"], p["text"], ex_norm[words.normalize(p["text"])]) for p in phrases
                if words.normalize(p["text"]) in ex_norm]
    out = ["# FEAT-36 eval", "", f"{len(phrases)} phrases, "
           f"{sum(len(v) for v in examples['examples'].values()) + len(examples['table_talk'])} examples.",
           "Gate: right move among the buttons >= 95% of real moves, silent on >= 85% of table talk, < 100 ms per message.", "",
           "Examples/phrases text overlap: " + ("none" if not overlaps else "")]
    out += [f"- OVERLAP id {i}: {t!r} also an example for {m}" for i, t, m in overlaps]
    out.append("")
    for name in selected(args.model):
        out += eval_model(name, examples, wordlist, phrases)
    text = "\n".join(out) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


# bench

def bench_one(name):
    _, _, phrases = load_data()
    texts = [p["text"] for p in phrases]
    res = {"model": name, "threads": {}}
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
        lat.sort()
        res["threads"][threads] = {"load_s": load, "p50": statistics.median(lat),
                                   "p95": lat[int(0.95 * (len(lat) - 1))], "max": lat[-1]}
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
           "| model | threads | load s | p50 ms | p95 ms | max ms | peak RSS MB |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        for th, v in r["threads"].items():
            out.append(f"| {r['model']} | {th} | {v['load_s']:.2f} | {v['p50']:.1f} | {v['p95']:.1f} | {v['max']:.1f} | {r['peak_rss_mb']:.0f} |")
    text = "\n".join(out) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("fetch", cmd_fetch), ("eval", cmd_eval), ("bench", cmd_bench)):
        p = sub.add_parser(name)
        p.add_argument("--model", choices=["e5", "minilm", "both"], default="both")
        if name != "fetch":
            p.add_argument("--out")
        p.set_defaults(fn=fn)
    one = sub.add_parser("bench-one")
    one.add_argument("model", choices=list(MODELS))
    one.set_defaults(fn=lambda a: print(json.dumps(bench_one(a.model))))
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
