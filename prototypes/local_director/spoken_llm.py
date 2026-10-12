"""FEAT-39 round 2: spoken blackjack moves and I Doubt It doubts read by an Ollama model, scored like FEAT-36 / D34.

Blackjack: every row of prototypes/spoken_moves/phrases.csv (the labelled test set; 195 rows, the "673" in the brief counts the
example/talk lists the e5 head trains on, which would be circular here). The model gets the phrase and its legal moves and must answer
one legal move or "talk". Metric as bands.passes: move row right when the answer is the label or an accepted move; talk row silent
when the answer is "talk" (also reported: silent-or-accepted). Doubt: the phrases of tests/test_spoken_doubt.py, answer doubt/talk,
compared with llmcord_core.spoken.doubt.read_doubt ('sure'/'ask' = doubt, None = talk).
Usage: python spoken_llm.py {bj,doubt} [--host URL] [--model TAG] [--think {off,on}] [--limit N] [--out DIR]
"""
import argparse
import ast
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import ollama_bench as ob  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CSV = ROOT / 'prototypes' / 'spoken_moves' / 'phrases.csv'
TESTS = ROOT / 'tests' / 'test_spoken_doubt.py'

MEANING = {'hit': 'take another card', 'stand': 'keep the hand, take no more cards', 'double': 'double the bet and take exactly one more card',
           'surrender': 'give up the hand and lose half the bet', 'insure': 'buy insurance against a dealer blackjack',
           'no_insurance': 'decline insurance', 'even_money': 'take even money on own blackjack against a dealer ace'}
BJ_SYSTEM = ('You read what a blackjack player typed in a Discord table chat. Decide whether the message is the player choosing a move. '
             'Answer with exactly one legal move, or "talk" if the message is chit-chat, a question, a joke or anything that is not a clear instruction to act now. '
             'Messages may be English, Korean or mixed. Never answer a move that is not in the legal list.')
DB_SYSTEM = ('You read a message typed at an I Doubt It card table. Answer "doubt" only if the player is calling another player a liar / accusing them of bluffing now '
             '(English or Korean). Answer "talk" for everything else, including negated doubts ("no doubt", "not lying"), questions, and chit-chat.')


def bj_schema(legal):
    return {'type': 'object', 'properties': {'answer': {'type': 'string', 'enum': [*legal, 'talk']}}, 'required': ['answer'], 'additionalProperties': False}


def rows():
    out = []
    for r in csv.DictReader(open(CSV, encoding='utf-8')):
        out.append({'id': r['id'], 'text': r['text'], 'lang': r['lang'], 'label': r['label'], 'accept': r['accept'].split(), 'legal': r['legal'].split()})
    return out


def stratified(rs, limit):
    if not limit or limit >= len(rs):
        return rs
    step = len(rs) / limit
    return [rs[int(i * step)] for i in range(limit)]


def passes(r, ans):
    if r['label'] == 'talk':
        return ans == 'talk' or ans in r['accept']
    return ans == r['label'] or ans in r['accept']


def answer(text):
    try:
        return json.loads(text).get('answer')
    except Exception:
        return None


def run_bj(a, model):
    think = a.think == 'on'
    rs = stratified(rows(), a.limit)
    ob.chat(a.host, model, 'Say ok.', 'ok', None, num_predict=8)
    res, calls = [], []
    for r in rs:
        usr = json.dumps({'message': r['text'], 'legal_moves': {m: MEANING[m] for m in r['legal']}}, ensure_ascii=False)
        c = ob.chat(a.host, model, BJ_SYSTEM, usr, bj_schema(r['legal']), think=think, num_predict=2048 if think else 48)
        calls.append(c)
        res.append({**r, 'ans': answer(c['text'])})
        print('.', end='', flush=True)
    print()
    mv = [x for x in res if x['label'] != 'talk']
    tk = [x for x in res if x['label'] == 'talk']
    by = {}
    for x in res:
        n, p = by.get(x['lang'], (0, 0))
        by[x['lang']] = (n + 1, p + passes(x, x['ans']))
    out = {'model': model, 'think': a.think, 'n': len(res), 'sampled': bool(a.limit and a.limit < 195), 'latency': ob.lat_summary(calls),
           'move_pass': [sum(passes(x, x['ans']) for x in mv), len(mv)], 'talk_silent': [sum(x['ans'] == 'talk' for x in tk), len(tk)],
           'talk_silent_or_accepted': [sum(passes(x, x['ans']) for x in tk), len(tk)], 'invalid': sum(x['ans'] is None for x in res),
           'by_lang_pass': {k: [v[1], v[0]] for k, v in by.items()},
           'moves_called_on_talk_as': {}, 'failures': [x for x in res if not passes(x, x['ans'])]}
    for x in tk:
        if x['ans'] not in ('talk', None):
            out['moves_called_on_talk_as'][x['ans']] = out['moves_called_on_talk_as'].get(x['ans'], 0) + 1
    return out


def doubt_set():
    """(text, expected) from tests/test_spoken_doubt.py: expected doubt for the sure/ask lists, talk for negated/excluded/table-talk lists."""
    tree = ast.parse(TESTS.read_text(encoding='utf-8'))
    items = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith('test_'):
            continue
        for call in ast.walk(fn):
            if isinstance(call, ast.Call) and getattr(call.func, 'attr', '') == 'check' and call.args:
                want = ast.literal_eval(call.args[0])
                items += [(ast.literal_eval(n), 'talk' if want is None else 'doubt', want) for n in call.args[1:]]
    return items


def run_doubt(a, model):
    from llmcord_core.spoken.doubt import read_doubt
    think = a.think == 'on'
    schema = {'type': 'object', 'properties': {'answer': {'type': 'string', 'enum': ['doubt', 'talk']}}, 'required': ['answer'], 'additionalProperties': False}
    items = [x for x in doubt_set() if x[0] != '']
    ob.chat(a.host, model, 'Say ok.', 'ok', None, num_predict=8)
    res, calls = [], []
    for text, exp, want in items:
        c = ob.chat(a.host, model, DB_SYSTEM, json.dumps({'message': text}, ensure_ascii=False), schema, think=think, num_predict=2048 if think else 32)
        calls.append(c)
        ref = 'talk' if read_doubt(text) is None else 'doubt'
        res.append({'text': text, 'expected': exp, 'want_reader': want, 'ans': answer(c['text']), 'read_doubt': ref})
        print('.', end='', flush=True)
    print()
    pos = [x for x in res if x['expected'] == 'doubt']
    neg = [x for x in res if x['expected'] == 'talk']
    return {'model': model, 'think': a.think, 'n': len(res), 'latency': ob.lat_summary(calls),
            'doubt_recall': [sum(x['ans'] == 'doubt' for x in pos), len(pos)], 'talk_silent': [sum(x['ans'] == 'talk' for x in neg), len(neg)],
            'read_doubt_recall': [sum(x['read_doubt'] == 'doubt' for x in pos), len(pos)], 'read_doubt_talk_silent': [sum(x['read_doubt'] == 'talk' for x in neg), len(neg)],
            'failures': [x for x in res if x['ans'] != x['expected']]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['bj', 'doubt'])
    p.add_argument('--host', default=ob.HOST)
    p.add_argument('--model', default='')
    p.add_argument('--think', choices=['off', 'on'], default='off')
    p.add_argument('--limit', type=int, default=0, help='stratified sample size (0 = all 195)')
    p.add_argument('--out', default=ob.OUT)
    a = p.parse_args()
    model = ob.resolve_model(a.host, a.model)
    res = run_bj(a, model) if a.cmd == 'bj' else run_doubt(a, model)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f'{a.out}/{a.cmd}_think-{a.think}.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'failures'}, ensure_ascii=False, indent=1), f"\nfailures: {len(res['failures'])}")


if __name__ == '__main__':
    main()
