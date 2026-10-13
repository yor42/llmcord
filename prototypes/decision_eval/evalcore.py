"""FEAT-47: pure helpers for the decision-model evaluation (items, requests, answer parsing, scoring, statistics).

No network, no environment reads, no clock; load_items is the only file reader. Standard library only. Wire formats and the
candidates are described in docs/engineering/research/decision-models-eval.md ("Harness design"); the prompt texts are adapted
from prototypes/local_director/spoken_llm.py so the numbers sit next to the Qwen 3.5 4B round.
Used by run_eval.py; tests: tests/test_decision_eval.py.
"""
import ast
import collections
import csv
import json
import math
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "prototypes" / "spoken_moves" / "phrases.csv"
TESTS = ROOT / "tests" / "test_spoken_doubt.py"

MEANING = {"hit": "take another card", "stand": "keep the hand, take no more cards", "double": "double the bet and take exactly one more card",
           "surrender": "give up the hand and lose half the bet", "insure": "buy insurance against a dealer blackjack",
           "no_insurance": "decline insurance", "even_money": "take even money on own blackjack against a dealer ace"}
BJ_SYSTEM = ("You read what a blackjack player typed in a Discord table chat. Decide whether the message is the player choosing a move. "
             "Answer with exactly one legal move, or \"talk\" if the message is chit-chat, a question, a joke or anything that is not a clear instruction to act now. "
             "Messages may be English, Korean or mixed. Never answer a move that is not in the legal list.")
DB_SYSTEM = ("You read a message typed at an I Doubt It card table. Answer \"doubt\" only if the player is calling another player a liar / accusing them of bluffing now "
             "(English or Korean). Answer \"talk\" for everything else, including negated doubts (\"no doubt\", \"not lying\"), questions, and chit-chat.")
OTHER_BJ = "table talk, a question, a joke, or anything that is not a clear instruction to act now"
OTHER_DB = "anything else: table talk, a question, a negated doubt (\"no doubt\", \"not lying\"), a joke or chit-chat"
CHOICE_BJ = ("You read what a blackjack player typed in a Discord table chat (English, Korean or mixed). Pick the one option that best matches the message. "
             "Pick other if the message is not a clear instruction to act now.")
CHOICE_DB = ("You read a message typed at an I Doubt It card table (English or Korean). Pick doubt only if the player is calling another player a liar "
             "or accusing them of bluffing now. Pick other for everything else.")
DOUBT_PREDICATE = "Is the player accusing another player of lying or bluffing right now?"

URL_OAI = "https://api.openai.com/v1/decisions"
URL_D1 = "https://openrouter.ai/api/alpha/decisions"
URL_CHAT = "https://openrouter.ai/api/v1/chat/completions"
D1 = "microsoft/microsoft-decision-1"
# kind: how the question is asked and read. price: USD per 1M tokens (input, output): OpenAI and Decision-1 from the research note,
# Haiku and Flash-Lite OpenRouter list prices, 2026-10-12. out_tokens: assumed output tokens per call (Flash-Lite reasoning is mandatory and billed as output).
CANDIDATES = {
    "oai-choice": {"url": URL_OAI, "key_env": "OPENAI_API_KEY", "model": "gpt-6-luna", "kind": "choice", "price": (0.10, 0.0), "out_tokens": 0},
    "oai-predicate": {"url": URL_OAI, "key_env": "OPENAI_API_KEY", "model": "gpt-6-luna", "kind": "predicate", "price": (0.10, 0.0), "out_tokens": 0},
    "or-d1-choice": {"url": URL_D1, "key_env": "OPENROUTER_API_KEY", "model": D1, "kind": "choice", "price": (0.042, 0.0), "out_tokens": 0},
    "or-d1-noul": {"url": URL_D1, "key_env": "OPENROUTER_API_KEY", "model": D1, "kind": "noul", "price": (0.042, 0.0), "out_tokens": 0},
    "or-haiku": {"url": URL_CHAT, "key_env": "OPENROUTER_API_KEY", "model": "anthropic/claude-haiku-5.5", "kind": "chat", "price": (0.10, 0.50), "out_tokens": 12},
    "or-flashlite": {"url": URL_CHAT, "key_env": "OPENROUTER_API_KEY", "model": "google/gemini-3.5-flash-lite", "kind": "chat", "price": (0.30, 2.50), "out_tokens": 100},
}
SHUFFLED = ("oai-choice", "or-d1-choice")
BUCKETS = (0.0, 0.5, 0.7, 0.9, 0.99, 1.0)


def lang_of(text):
    ko = re.search("[ㄱ-ㆎ가-힣]", text) is not None
    en = re.search("[A-Za-z]", text) is not None
    return "mix" if ko and en else "ko" if ko else "en"


def doubt_rows():
    """(text, expected) from tests/test_spoken_doubt.py as spoken_llm.doubt_set: expected None = not a doubt."""
    tree = ast.parse(TESTS.read_text(encoding="utf-8"))
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("test_"):
            continue
        for call in ast.walk(fn):
            if isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "check" and call.args:
                want = ast.literal_eval(call.args[0])
                out += [(ast.literal_eval(n), want) for n in call.args[1:]]
    return out


def load_items():
    """The 195 blackjack rows of phrases.csv, then the 116 non-empty doubt phrases; see the test module docstring for the keys."""
    items = []
    with open(CSV, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            items.append({"id": f"bj{r['id']}", "task": "bj", "text": r["text"], "lang": r["lang"], "label": r["label"],
                          "accept": r["accept"].split(), "legal": r["legal"].split()})
    for n, (text, want) in enumerate(t for t in doubt_rows() if t[0] != ""):
        items.append({"id": f"db{n:03d}", "task": "doubt", "text": text, "lang": lang_of(text), "label": "talk" if want is None else "doubt",
                      "accept": [], "legal": ["doubt"]})
    return items


def stratified(items, limit):
    if not limit or limit >= len(items):
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def option_order(options, item_id, k, seed=0):
    """Deterministic permutation (string seeds hash with sha512, so PYTHONHASHSEED does not matter)."""
    out = list(options)
    random.Random(f"{seed}:{item_id}:{k}").shuffle(out)
    return out


def options_of(item):
    return item["legal"] + ["other"]


def move_text(move):
    return MEANING.get(move, move)


def option_desc(item, option):
    if option == "other":
        return OTHER_DB if item["task"] == "doubt" else OTHER_BJ
    if item["task"] == "doubt":
        return "The player is accusing another player of lying or bluffing right now"
    return f"The player is choosing to {move_text(option)} right now, as an instruction to act"


def predicate_text(item, move):
    if item["task"] == "doubt":
        return DOUBT_PREDICATE
    return f"Is the player choosing to {move_text(move)} right now, as an instruction to act (not a question, a joke or chit-chat)?"


def context_for(item):
    """One sentence of where the message was typed, given to every candidate (the chat ones also have it in their system prompt)."""
    if item["task"] == "doubt":
        return "This message was typed by a player at an I Doubt It card table, during the doubt window."
    return ("This message was typed in a Discord blackjack table chat by the player whose turn it is, as a possible move for the current hand "
            "(it may be English, Korean or mixed).")


def choice_name(item):
    return "doubt_call" if item["task"] == "doubt" else "player_move"


def build_request(candidate, item, k=0):
    """{'url', 'json', 'key_env'}; never holds a key value."""
    c = CANDIDATES[candidate]
    kind, text = c["kind"], item["text"]
    instr = CHOICE_DB if item["task"] == "doubt" else CHOICE_BJ
    if kind == "chat":
        legal = item["legal"]
        system = DB_SYSTEM if item["task"] == "doubt" else BJ_SYSTEM
        user = {"message": text} if item["task"] == "doubt" else {"message": text, "legal_moves": {m: move_text(m) for m in legal}}
        gemini = "gemini" in c["model"]
        body = {"model": c["model"], "temperature": 0, "max_tokens": 256 if gemini else 64,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {
                    "type": "object", "properties": {"answer": {"type": "string", "enum": [*legal, "talk"]}}, "required": ["answer"], "additionalProperties": False}}}}
        if gemini:  # reasoning is mandatory on Flash-Lite: keep it minimal and out of the response
            body["reasoning"] = {"effort": "minimal", "exclude": True}
        else:  # Haiku 5.5 reasons by default and then has no budget left for the answer
            body["reasoning"] = {"enabled": False}
        return {"url": c["url"], "json": body, "key_env": c["key_env"]}
    openai = c["url"] == URL_OAI
    if kind == "choice":
        order = option_order(options_of(item), item["id"], k)
        if openai:
            q = [{"type": "choice", "name": choice_name(item), "instructions": instr,
                  "choices": [{"value": o, "description": option_desc(item, o)} for o in order]}]
        else:
            q = {choice_name(item): {"type": "choice", "instructions": instr, "criteria": {o: option_desc(item, o) for o in order}}}
    elif openai:
        q = [{"type": "predicate", "name": f"move_{m}", "instructions": predicate_text(item, m)} for m in item["legal"]]
    else:
        q = {f"move_{m}": {"type": "noul", "instructions": predicate_text(item, m)} for m in item["legal"]}
    ctx = context_for(item)
    body = ({"model": c["model"], "input": f"{ctx}\nMessage: \"{text}\"", "questions": q} if openai
            else {"model": c["model"], "state": {"context": ctx, "message": text}, "questions": q})
    return {"url": c["url"], "json": body, "key_env": c["key_env"]}


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def usage_of(resp):
    """(cost, tokens) from a response's usage; None where absent or not a number."""
    u = resp.get("usage") if isinstance(resp, dict) else None
    if not isinstance(u, dict):
        return None, None
    cost = _num(u.get("cost"))
    tokens = _num(u.get("total_tokens"))
    if tokens is None:
        parts = [_num(u.get(n)) for n in ("input_tokens", "prompt_tokens", "output_tokens", "completion_tokens")]
        parts = [x for x in parts if x is not None]
        tokens = sum(parts) if parts else None
    return cost, int(tokens) if tokens is not None else None


def _norm(option):
    return "talk" if option == "other" else option


def _probs_map(raw):
    """{option: p} from either wire shape (list of {value, probability} or a map), with 'other' reported as 'talk'."""
    if isinstance(raw, dict):
        pairs = raw.items()
    elif isinstance(raw, list):
        pairs = [(x.get("value"), x.get("probability")) for x in raw if isinstance(x, dict)]
    else:
        return {}
    return {_norm(o): _num(p) for o, p in pairs if isinstance(o, str) and _num(p) is not None}


def _answer_items(resp):
    """[(name, answer dict)] from answers as a list (OpenAI) or a map (OpenRouter)."""
    a = resp.get("answers") if isinstance(resp, dict) else None
    if isinstance(a, dict):
        return [(n, v) for n, v in a.items() if isinstance(v, dict)]
    if isinstance(a, list):
        return [(v.get("name"), v) for v in a if isinstance(v, dict)]
    return []


def parse_answer(candidate, item, response_json, tau=0.5):
    """{'answer','p','probs','refusal','cost','tokens'}; answer is a legal move, 'talk', 'doubt' or None (invalid or refused)."""
    kind = CANDIDATES[candidate]["kind"]
    cost, tokens = usage_of(response_json)
    out = {"answer": None, "p": None, "probs": {}, "refusal": False, "cost": cost, "tokens": tokens, "raw": None, "finish": None}
    legal = item["legal"]
    try:
        if kind == "chat":
            first = response_json["choices"][0]
            out["finish"] = first.get("finish_reason")
            msg = first["message"]
            if isinstance(msg.get("content"), str):
                out["raw"] = msg["content"][:80]
            if msg.get("refusal"):
                out["refusal"] = True
                return out
            ans = json.loads(msg.get("content") or "").get("answer")
            out["answer"] = ans if isinstance(ans, str) and ans in [*legal, "talk"] else None
            return out
        rows = _answer_items(response_json)
        if any(v.get("type") == "refusal" for _, v in rows):
            out["refusal"] = True
            return out
        if kind == "choice":
            rows = [v for n, v in rows if n == choice_name(item)] or [v for _, v in rows]
            if not rows:
                return out
            v = rows[0]
            choice = v.get("choice")
            out["probs"] = _probs_map(v.get("probabilities"))
            ans = _norm(choice) if isinstance(choice, str) else None
            if ans in [*legal, "talk"]:
                out["answer"], out["p"] = ans, out["probs"].get(ans)
            return out
        key = "probability" if kind == "predicate" else "noul"
        # an answer without a name is matched to the question asked at the same position
        named = [(n if isinstance(n, str) else f"move_{legal[i]}" if n is None and i < len(legal) else None, v) for i, (n, v) in enumerate(rows)]
        probs = {n[5:]: _num(v.get(key)) for n, v in named if isinstance(n, str) and n.startswith("move_") and n[5:] in legal}
        out["probs"] = {m: p for m, p in probs.items() if p is not None}
        if out["probs"]:
            best = max(out["probs"], key=lambda m: out["probs"][m])
            out["p"] = out["probs"][best]
            out["answer"] = best if out["p"] >= tau else "talk"
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        out.update(answer=None, p=None, probs={})
    return out


def passes(item, answer):
    """spoken_llm / bands rule: move row right on the label or an accepted move; talk row right on 'talk' or an accepted move."""
    if answer is None:
        return False
    if item["label"] == "talk":
        return answer == "talk" or answer in item["accept"]
    return answer == item["label"] or answer in item["accept"]


def _tally(items, results, task):
    t = {"move_pass": 0, "move_n": 0, "talk_silent": 0, "talk_n": 0, "invalid": 0, "refusals": 0, "by_lang": {}}
    for it, r in zip(items, results):
        if it["task"] != task:
            continue
        lang = t["by_lang"].setdefault(it["lang"], {"move_pass": 0, "move_n": 0, "talk_silent": 0, "talk_n": 0})
        if it["label"] == "talk":
            for d in (t, lang):
                d["talk_n"] += 1
                d["talk_silent"] += r["answer"] == "talk"
        else:
            for d in (t, lang):
                d["move_n"] += 1
                d["move_pass"] += passes(it, r["answer"])
        if r["refusal"]:
            t["refusals"] += 1
        elif r["answer"] is None:
            t["invalid"] += 1
    return t


def score(items, results):
    """bj counts at top level; the doubt set under 'doubt' (recall = doubts answered 'doubt', silent = non-doubts answered 'talk')."""
    s = _tally(items, results, "bj")
    d = _tally(items, results, "doubt")
    s["doubt"] = {"recall": d["move_pass"], "doubt_n": d["move_n"], "silent": d["talk_silent"], "talk_n": d["talk_n"],
                  "invalid": d["invalid"], "refusals": d["refusals"],
                  "by_lang": {lang: {"recall": v["move_pass"], "doubt_n": v["move_n"], "silent": v["talk_silent"], "talk_n": v["talk_n"]}
                              for lang, v in d["by_lang"].items()}}
    return s


def separation(kind, items, results):
    """{'auroc', 'buckets'} of the top probability. choice/chat: p against passes(); predicate/noul: p (best move) against 'is a move
    row' (a doubt row for doubt items), since a low p on chatter is the right call. Results without p are skipped."""
    pairs = [(r["p"], (it["label"] != "talk") if kind in ("predicate", "noul") else passes(it, r["answer"]))
             for it, r in zip(items, results) if r.get("p") is not None]
    ps, ok = [p for p, _ in pairs], [c for _, c in pairs]
    return {"auroc": auroc(ps, ok), "buckets": calibration_buckets(ps, ok)}


def latency_summary(walls):
    """Nearest-rank p50 and p95."""
    v = sorted(walls)
    if not v:
        return {"p50": None, "p95": None, "n": 0}
    rank = lambda q: v[min(len(v), max(1, math.ceil(round(q * len(v), 9)))) - 1]  # noqa: E731
    return {"p50": rank(0.5), "p95": rank(0.95), "n": len(v)}


def cost_per_1000(costs):
    known = [c for c in costs if c is not None]
    return sum(known) / len(known) * 1000 if known else None


def auroc(scores, labels):
    """Probability that a random positive outscores a random negative (ties count half); None without both classes."""
    pos = [s for s, ok in zip(scores, labels) if ok]
    neg = [s for s, ok in zip(scores, labels) if not ok]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def calibration_buckets(ps, correct):
    """Accuracy by probability bucket: <0.5, 0.5-0.7, 0.7-0.9, 0.9-0.99, >=0.99 (the last includes 1.0)."""
    out = []
    for i, lo in enumerate(BUCKETS[:-1]):
        hi = BUCKETS[i + 1]
        hits = [bool(ok) for p, ok in zip(ps, correct) if p >= lo and (p < hi or hi == 1.0)]
        out.append({"lo": lo, "hi": hi, "n": len(hits), "accuracy": sum(hits) / len(hits) if hits else None})
    return out


def tau_sweep(items, answers_p, taus):
    """Move-right rate over move rows and silent rate over talk rows if an answer with p < tau counts as silent (p None never does)."""
    out = []
    for tau in taus:
        mv = [(it, a) for it, a in zip(items, answers_p) if it["label"] != "talk"]
        tk = [(it, a) for it, a in zip(items, answers_p) if it["label"] == "talk"]
        right = sum(1 for it, (a, p) in mv if a != "talk" and passes(it, a) and (p is None or p >= tau))
        silent = sum(1 for _, (a, p) in tk if a == "talk" or (p is not None and p < tau))
        out.append({"tau": tau, "move_rate": right / len(mv) if mv else None, "talk_silent_rate": silent / len(tk) if tk else None})
    return out


def order_stability(by_item):
    """{item_id: [(answer, p), ...]} over the K orders -> share of items answered the same in every order, mean spread of p."""
    if not by_item:
        return {"stable_share": None, "mean_p_spread": None}
    stable = sum(1 for v in by_item.values() if len({a for a, _ in v}) == 1)
    spreads = [max(ps) - min(ps) for ps in ([p for _, p in v if p is not None] for v in by_item.values()) if ps]
    return {"stable_share": stable / len(by_item), "mean_p_spread": sum(spreads) / len(spreads) if spreads else None}


def headline(answers_p):
    """One (answer, p) for an item from its K orders: majority answer, a tie goes to the order-0 answer, p = mean over the agreeing orders."""
    count = collections.Counter(a for a, _ in answers_p)
    top = max(count.values())
    ans = answers_p[0][0] if count[answers_p[0][0]] == top else next(a for a, _ in answers_p if count[a] == top)
    ps = [p for a, p in answers_p if a == ans and p is not None]
    return ans, sum(ps) / len(ps) if ps else None


def est_call_usd(candidate, body):
    """Rough USD for one call: request size over 3 chars per token, assumed output tokens, assumed prices."""
    c = CANDIDATES[candidate]
    tokens = len(json.dumps(body, ensure_ascii=False)) / 3 + 1
    return (tokens * c["price"][0] + c["out_tokens"] * c["price"][1]) / 1e6


def estimate(candidates, items, shuffles=3):
    calls, usd = {}, {}
    for cand in candidates:
        n = shuffles if cand in SHUFFLED else 1
        calls[cand] = len(items) * n
        usd[cand] = sum(est_call_usd(cand, build_request(cand, it, k)["json"]) for it in items for k in range(n))
    return {"calls": calls, "total_calls": sum(calls.values()), "est_usd": sum(usd.values()), "per_candidate_usd": usd}
