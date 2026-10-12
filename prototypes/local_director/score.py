import json, os, statistics, sys
sys.path.insert(0, os.path.dirname(__file__))
from cases import C
R = os.path.join(os.path.dirname(__file__), 'results')
MODELS = ['qwen2.5-0.5b-instruct-q4_k_m', 'qwen2.5-0.5b-instruct-q4_0', 'qwen2.5-1.5b-instruct-q4_k_m']
def pct(v, p):
    v = sorted(v); k = (len(v) - 1) * p / 100; f = int(k); c = min(f + 1, len(v) - 1); return v[f] + (v[c] - v[f]) * (k - f)
def parse(t):
    try:
        d = json.loads(t)
        if isinstance(d, dict) and isinstance(d.get('speakers'), list): return d['speakers']
    except Exception: pass
    return None
def effective(c, ans):
    ids = {i for i, _ in c['cast']}
    out = list(dict.fromkeys(x for x in ans if type(x) is int and x in ids))
    if not out and not c['ambient']: out = [c['cast'][0][0]]
    return out
def correct(c, eff):
    exp = c['expected']
    if 'any' in exp: return len(eff) >= 1
    return sorted(eff) == sorted(exp)
def contains(c, eff):
    exp = [e for e in c['expected'] if e != 'any']
    return bool(exp) and set(exp) <= set(eff)
def grade(rows, constrained):
    rr = [r for r in rows if r['constrained'] == constrained]
    per = {}
    for r in rr:
        c = C[r['case']]; ans = parse(r['text']); eff = effective(c, ans) if ans is not None else None
        per[r['case']] = dict(raw=r['text'], eff=eff, valid=ans is not None)
    return per
def summarize(model):
    d = json.load(open(f'{R}/{model}.json')); out = {'model': model, 'load_s': d['load_s'], 'peak_rss_mb': d['peak_rss_mb'], 'temp': d['temp'], 'throttled': d['throttled']}
    for mode, name in ((True, 'constrained'), (False, 'unconstrained')):
        rr = [r for r in d['rows'] if r['constrained'] == mode]
        per = grade(d['rows'], mode)
        s = {'wall_med': statistics.median(r['wall'] for r in rr), 'wall_p90': pct([r['wall'] for r in rr], 90),
             'prefill_med_ms': statistics.median(r['prefill_ms'] for r in rr), 'decode_med_ms': statistics.median(r['decode_ms'] for r in rr),
             'prompt_tok_med': statistics.median(r['prompt_n'] for r in rr), 'out_tok_med': statistics.median(r['decode_n'] for r in rr),
             'valid_rate': sum(p['valid'] for p in per.values()) / len(per)}
        for grp in ('all', 'en', 'ko', 'mix'):
            idx = [i for i, c in enumerate(C) if grp == 'all' or c['lang'] == grp]
            ok = [i for i in idx if per[i]['eff'] is not None]
            s[f'exact_{grp}'] = (sum(correct(C[i], per[i]['eff']) for i in ok), len(idx))
            addr = [i for i in idx if contains(C[i], C[i]['expected']) and C[i]['kind'] != 'amb']
            s[f'contains_{grp}'] = (sum(per[i]['eff'] is not None and contains(C[i], per[i]['eff']) for i in addr), len(addr))
            amb = [i for i in idx if C[i]['kind'] == 'amb']
            s[f'silence_{grp}'] = (sum(per[i]['eff'] == [] for i in amb), len(amb))
        s['per'] = per; out[name] = s
    return out
if __name__ == '__main__':
    for m in MODELS:
        if os.path.exists(f'{R}/{m}.json'):
            s = summarize(m)
            print(m, s['load_s'], s['peak_rss_mb'], s['temp'], s['throttled'])
            for n in ('constrained', 'unconstrained'):
                print(' ', n, {k: v for k, v in s[n].items() if k != 'per'})
