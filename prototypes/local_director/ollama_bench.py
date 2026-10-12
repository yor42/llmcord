"""FEAT-39 round 2: director benchmark against an Ollama server (set OLLAMA_URL; round 2 used the user's Qwen 3.5 4B over the tailnet).

Usage: python ollama_bench.py {tags,ps,director} [--host URL] [--model TAG] [--think {off,on}] [--out DIR]
Raw results go to --out (default: $FEAT39B_OUT or ~/feat39-bench-data/round2), never into the repo.
Other tasks: memory_cases.py, spoken_llm.py (both reuse the client below).
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
import zlib

sys.path.insert(0, os.path.dirname(__file__))
import score  # noqa: E402
from bench import SCHEMA, SYSTEM  # noqa: E402  (bench.py has no import-time side effects beyond creating results/)
from cases import C  # noqa: E402

HOST = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
OUT = os.environ.get('FEAT39B_OUT', os.path.expanduser('~/feat39-bench-data/round2'))


def get(host, path, timeout=10):
    with urllib.request.urlopen(host + path, timeout=timeout) as r:
        return json.load(r)


def chat(host, model, system, user, schema, think=False, num_predict=256, timeout=600):
    """One /api/chat call. Returns text, thinking, wall and Ollama's own timing counters (ns)."""
    body = {'model': model, 'stream': False, 'think': think, **({'format': schema} if schema else {}),
            'options': {'temperature': 0, 'seed': 1, 'num_predict': num_predict},
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}
    req = urllib.request.Request(host + '/api/chat', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    t = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    m = d.get('message', {})
    return {'text': m.get('content', ''), 'thinking': m.get('thinking', ''), 'wall': time.time() - t,
            **{k: d.get(k, 0) for k in ('prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration', 'load_duration', 'total_duration')}}


def resolve_model(host, model):
    if model:
        return model
    names = [m['name'] for m in get(host, '/api/tags')['models']]
    hit = [n for n in names if 'qwen3.5' in n.lower() and '4b' in n.lower()]
    if not hit:
        raise SystemExit(f'no qwen3.5 4b tag among {names}')
    return hit[0]


def lat_summary(rows):
    def med(k, scale=1e6):
        return statistics.median(r[k] for r in rows) / scale
    return {'n': len(rows), 'wall_med_s': statistics.median(r['wall'] for r in rows), 'wall_max_s': max(r['wall'] for r in rows),
            'prompt_tok_med': med('prompt_eval_count', 1), 'prefill_ms_med': med('prompt_eval_duration'),
            'out_tok_med': med('eval_count', 1), 'decode_ms_med': med('eval_duration')}


def stable_payload(c):
    recent = [{'message_id': 1000 + i, 'author_id': 500 + zlib.crc32(a.encode()) % 90, 'author_label': a, 'text': t, 'mentions': []}
              for i, (a, t) in enumerate(c['recent'])][-6:]
    return json.dumps({'cast': [{'id': i, 'name': n} for i, n in c['cast']], 'recent': recent, 'latest': c['latest'],
                       'latest_author': {'author_id': 777, 'author_label': c['author']}, 'ambient': c['ambient'], 'forced': None}, ensure_ascii=False)


def grade_director(rows, step=1):
    """Same metrics as score.py: exact, contains-addressed, ambient silence, by language."""
    out = {'valid': 0, 'groups': {}}
    per = []
    cs = C[::step]
    for i, r in enumerate(rows):
        ans = score.parse(r['text'])
        eff = score.effective(cs[i], ans) if ans is not None else None
        out['valid'] += ans is not None
        per.append({'raw': r['text'], 'eff': eff})
    for grp in ('all', 'en', 'ko', 'mix'):
        idx = [i for i, c in enumerate(cs) if grp == 'all' or c['lang'] == grp]
        addr = [i for i in idx if cs[i]['kind'] != 'amb' and score.contains(cs[i], cs[i]['expected'])]
        amb = [i for i in idx if cs[i]['kind'] == 'amb']
        out['groups'][grp] = {
            'exact': [sum(per[i]['eff'] is not None and score.correct(cs[i], per[i]['eff']) for i in idx), len(idx)],
            'contains_addressed': [sum(per[i]['eff'] is not None and score.contains(cs[i], per[i]['eff']) for i in addr), len(addr)],
            'ambient_silence': [sum(per[i]['eff'] == [] for i in amb), len(amb)]}
    out['per'] = per
    return out


def cmd_director(a):
    model = resolve_model(a.host, a.model)
    think = a.think == 'on'
    chat(a.host, model, SYSTEM, stable_payload(C[0]), SCHEMA, think=False, num_predict=64)  # warm-up (loads the model)
    rows = []
    for c in C[::a.step]:
        r = chat(a.host, model, SYSTEM, stable_payload(c), SCHEMA, think=think, num_predict=2048 if think else 64)
        rows.append(r)
        print('.', end='', flush=True)
    print()
    res = {'model': model, 'think': a.think, 'latency': lat_summary(rows), 'step': a.step, 'grade': grade_director(rows, a.step), 'rows': rows}
    os.makedirs(a.out, exist_ok=True)
    path = f'{a.out}/director_think-{a.think}.json'
    json.dump(res, open(path, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ('model', 'think', 'latency')} | {'groups': res['grade']['groups'], 'valid': res['grade']['valid']}, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['tags', 'ps', 'director'])
    p.add_argument('--host', default=HOST)
    p.add_argument('--model', default='')
    p.add_argument('--think', choices=['off', 'on'], default='off')
    p.add_argument('--step', type=int, default=1, help='run every Nth case (thinking-on is slow, ~100 s per call)')
    p.add_argument('--out', default=OUT)
    a = p.parse_args()
    if a.cmd == 'tags':
        print(json.dumps(get(a.host, '/api/tags'), indent=1))
        try:
            print('version', get(a.host, '/api/version'))
        except Exception as e:
            print('version unavailable', e)
    elif a.cmd == 'ps':
        print(json.dumps(get(a.host, '/api/ps'), indent=1))
    else:
        cmd_director(a)


if __name__ == '__main__':
    main()
