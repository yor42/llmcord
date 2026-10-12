"""FEAT-43: decision-model evaluation harness (prototypes/decision_eval/evalcore.py and run_eval.py). Offline, no keys.

These tests DEFINE the interface of the harness; the module is written to satisfy them. Research only, nothing in
llmcord_core uses it. Interface (all pure except load_items):

  load_items() -> list[dict]  keys id (unique str), task ('bj'|'doubt'), text, lang ('en'|'ko'|'mix'), label ('talk' | move | 'doubt'),
      accept (list), legal (list; ['doubt'] for doubt items). 195 bj rows of prototypes/spoken_moves/phrases.csv, then the 116 non-empty
      phrases of tests/test_spoken_doubt.py (ast, as spoken_llm.doubt_set; expected None -> label 'talk', else 'doubt').
      Doubt language by script: Hangul only 'ko', Hangul + ASCII letters 'mix', else 'en'.
  option_order(options, item_id, k, seed=0) -> list     deterministic permutation, independent of PYTHONHASHSEED.
  build_request(candidate, item, k=0) -> {'url', 'json', 'key_env'}   candidates: oai-choice oai-predicate or-d1-choice or-d1-noul
      or-haiku or-flashlite. Choice options are option_order(legal + ['other'], item['id'], k) (doubt: ['doubt', 'other']).
      OpenAI body: model, input, questions [ {type, name, instructions, choices|...} ]. OpenRouter decision body: model, state,
      questions {name: {type, instructions, criteria}}. Predicate/noul: one question per legal move named 'move_<move>'.
      Chat: OpenRouter chat completions, temperature 0, response_format json_schema whose enum is legal + ['talk'].
      The request never holds a key value.
  parse_answer(candidate, item, response_json, tau=0.5) -> {'answer','p','probs','refusal','cost','tokens'}
      answer is a legal move, 'talk', 'doubt' (doubt items) or None (invalid, refusal, malformed, illegal move). 'other' -> 'talk'.
      probs maps option -> probability with 'other' reported as 'talk' and predicate names without the 'move_' prefix.
      p is the probability of the chosen option (choice) or of the best move (predicate/noul); None when unknown.
      tokens = input + output tokens (total_tokens for chat). cost/tokens None when the response has no usage.
  passes(item, answer) -> bool    spoken_llm / bands rule; None never passes.
  score(items, results) -> dict   bj items only at top level: move_pass, move_n, talk_silent (strictly 'talk'), talk_n,
      by_lang {lang: {move_pass, move_n, talk_silent, talk_n}}, invalid (answer None and not refusal), refusals.
      score['doubt'] = {recall, doubt_n, silent, talk_n, invalid, refusals, by_lang {lang: {recall, doubt_n, silent, talk_n}}}.
  latency_summary(walls) -> {'p50','p95','n'} nearest rank. cost_per_1000(costs) -> float|None. auroc(scores, labels) -> float|None.
  calibration_buckets(ps, correct) -> 5 dicts {lo, hi, n, accuracy}. tau_sweep(items, answers_p, taus): answers_p is a list parallel
      to items of (answer, p); -> [{tau, move_rate, talk_silent_rate}] over move rows / talk rows (fractions).
  order_stability({item_id: [(answer, p), ...]}) -> {'stable_share', 'mean_p_spread'}.
  estimate(candidates, items, shuffles=3) -> {'calls', 'total_calls', 'est_usd', 'per_candidate_usd'}; choice candidates make
      len(items) * shuffles calls, the others len(items).

run_eval.main(argv, env=None, post=None) -> int. Flags: --candidates a,b (default all six), --run, --limit N (N items in total from the
  combined list, stratified), --out DIR, --max-usd X, --shuffles K (default 3). post(url, body, key) -> (response_json, wall_seconds);
  the key is passed to post only, never printed or written. No --run: print the estimate, never call post, need no key. --run with a
  missing key env var for a selected candidate: nonzero before any call. --run makes exactly items x (K for choice, else 1) calls
  (no extra smoke calls), writes JSON under --out. --max-usd: stop once summed usage cost passes it and return nonzero.
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototypes" / "decision_eval"))

import evalcore as ec  # noqa: E402
import run_eval  # noqa: E402

FAKE_OAI = "FAKE-OPENAI-not-a-real-key-0001"
FAKE_OR = "FAKE-OPENROUTER-not-a-real-key-0002"
CANDS = ["oai-choice", "oai-predicate", "or-d1-choice", "or-d1-noul", "or-haiku", "or-flashlite"]
URL_OAI = "https://api.openai.com/v1/decisions"
URL_D1 = "https://openrouter.ai/api/alpha/decisions"
URL_CHAT = "https://openrouter.ai/api/v1/chat/completions"

BJ = {"id": "t1", "task": "bj", "text": "hit me please", "lang": "en", "label": "hit", "accept": [], "legal": ["hit", "stand", "double"]}
BJ_TALK = {"id": "t2", "task": "bj", "text": "this deck hates me", "lang": "en", "label": "talk", "accept": ["hit"], "legal": ["hit", "stand", "double"]}
DB = {"id": "d1", "task": "doubt", "text": "liar!", "lang": "en", "label": "doubt", "accept": [], "legal": ["doubt"]}


def find_key(obj, name):
    """First value stored under `name` anywhere in a nested structure, else None."""
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        for v in obj.values():
            r = find_key(v, name)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, name)
            if r is not None:
                return r
    return None


def qname(cand, item):
    q = ec.build_request(cand, item)["json"]["questions"]
    return q[0]["name"] if isinstance(q, list) else next(iter(q))


def res(answer, **kw):
    return {"answer": answer, "p": None, "probs": {}, "refusal": False, "cost": None, "tokens": None, "wall": 0.1, **kw}


class LoadItemsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = ec.load_items()

    def test_counts_and_language_split_match_fixtures(self):
        bj = [i for i in self.items if i["task"] == "bj"]
        db = [i for i in self.items if i["task"] == "doubt"]
        self.assertEqual((len(bj), len(db), len(self.items)), (195, 116, 311))
        count = lambda rows, f: sum(1 for r in rows if f(r))  # noqa: E731
        self.assertEqual([count(bj, lambda r, x=x: r["lang"] == x) for x in ("en", "ko", "mix")], [71, 101, 23])
        self.assertEqual([count(db, lambda r, x=x: r["lang"] == x) for x in ("en", "ko", "mix")], [51, 61, 4])
        self.assertEqual(count(bj, lambda r: r["label"] == "talk"), 62)
        self.assertEqual((count(db, lambda r: r["label"] == "doubt"), count(db, lambda r: r["label"] == "talk")), (60, 56))

    def test_shape(self):
        self.assertEqual(len({i["id"] for i in self.items}), 311)
        for i in self.items:
            self.assertTrue(i["text"])
            self.assertIsInstance(i["accept"], list)
            self.assertIsInstance(i["legal"], list)
            if i["task"] == "doubt":
                self.assertEqual(i["legal"], ["doubt"])
                self.assertIn(i["label"], ("doubt", "talk"))
            else:
                self.assertTrue(i["legal"])
                self.assertTrue(i["label"] == "talk" or i["label"] in i["legal"])
        first = next(i for i in self.items if i["task"] == "bj")
        self.assertEqual((first["text"], first["lang"], first["label"], first["legal"]), ("hit me pls", "en", "hit", ["hit", "stand", "double"]))


class OptionOrderTests(unittest.TestCase):
    OPTS = ["hit", "stand", "double", "other"]

    def test_permutation_and_deterministic(self):
        a = ec.option_order(self.OPTS, "x", 0)
        self.assertEqual(sorted(a), sorted(self.OPTS))
        self.assertEqual(a, ec.option_order(self.OPTS, "x", 0))
        self.assertEqual(self.OPTS, ["hit", "stand", "double", "other"])  # input not mutated
        self.assertEqual(ec.option_order(["only"], "x", 3), ["only"])

    def test_k_and_seed_change_order(self):
        self.assertGreaterEqual(len({tuple(ec.option_order(self.OPTS, "x", k)) for k in range(10)}), 4)
        self.assertGreaterEqual(len({tuple(ec.option_order(self.OPTS, f"i{n}", 0)) for n in range(10)}), 4)
        self.assertGreaterEqual(len({tuple(ec.option_order(self.OPTS, "x", 0, seed=s)) for s in range(10)}), 4)

    def test_independent_of_hash_randomization(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import evalcore; "
                "print([evalcore.option_order(['hit','stand','double','other'], 'item-7', k) for k in range(5)])")
        outs = []
        for hs in ("1", "2", "random"):
            r = subprocess.run([sys.executable, "-c", code, str(ROOT / "prototypes" / "decision_eval")], capture_output=True, text=True,
                               env={**os.environ, "PYTHONHASHSEED": hs}, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            outs.append(r.stdout)
        self.assertEqual(len(set(outs)), 1)
        self.assertEqual(outs[0].strip(), str([ec.option_order(self.OPTS, "item-7", k) for k in range(5)]))


class BuildRequestTests(unittest.TestCase):
    def test_urls_and_key_env(self):
        want = {"oai-choice": (URL_OAI, "OPENAI_API_KEY"), "oai-predicate": (URL_OAI, "OPENAI_API_KEY"),
                "or-d1-choice": (URL_D1, "OPENROUTER_API_KEY"), "or-d1-noul": (URL_D1, "OPENROUTER_API_KEY"),
                "or-haiku": (URL_CHAT, "OPENROUTER_API_KEY"), "or-flashlite": (URL_CHAT, "OPENROUTER_API_KEY")}
        for c, (url, env) in want.items():
            r = ec.build_request(c, BJ)
            self.assertEqual((r["url"], r["key_env"]), (url, env), c)
            self.assertIn(BJ["text"], json.dumps(r["json"], ensure_ascii=False), c)

    def test_oai_choice(self):
        for k in range(3):
            b = ec.build_request("oai-choice", BJ, k)["json"]
            self.assertEqual(b["model"], "gpt-6-luna")
            self.assertIn(BJ["text"], json.dumps(b["input"], ensure_ascii=False))
            (q,) = b["questions"]
            self.assertEqual(q["type"], "choice")
            self.assertTrue(q["name"] and q["instructions"])
            self.assertEqual([c["value"] for c in q["choices"]], ec.option_order(["hit", "stand", "double", "other"], "t1", k))
            self.assertTrue(all(c["description"] for c in q["choices"]))

    def test_oai_choice_doubt_options(self):
        (q,) = ec.build_request("oai-choice", DB, 1)["json"]["questions"]
        self.assertEqual([c["value"] for c in q["choices"]], ec.option_order(["doubt", "other"], "d1", 1))

    def test_oai_predicate(self):
        b = ec.build_request("oai-predicate", BJ)["json"]
        self.assertEqual(b["model"], "gpt-6-luna")
        self.assertEqual([(q["type"], q["name"]) for q in b["questions"]], [("predicate", f"move_{m}") for m in BJ["legal"]])
        self.assertTrue(all(q["instructions"] for q in b["questions"]))
        (q,) = ec.build_request("oai-predicate", DB)["json"]["questions"]
        self.assertEqual((q["type"], q["name"]), ("predicate", "move_doubt"))

    def test_d1_choice(self):
        for k in range(3):
            b = ec.build_request("or-d1-choice", BJ, k)["json"]
            self.assertEqual(b["model"], "microsoft/microsoft-decision-1")
            self.assertIn(BJ["text"], json.dumps(b["state"], ensure_ascii=False))
            (q,) = b["questions"].values()
            self.assertEqual(q["type"], "choice")
            self.assertTrue(q["instructions"])
            self.assertIsInstance(q["criteria"], dict)
            self.assertEqual(list(q["criteria"]), ec.option_order(["hit", "stand", "double", "other"], "t1", k))
            self.assertTrue(all(q["criteria"].values()))
        (q,) = ec.build_request("or-d1-choice", DB)["json"]["questions"].values()
        self.assertEqual(sorted(q["criteria"]), ["doubt", "other"])

    def test_d1_noul(self):
        b = ec.build_request("or-d1-noul", BJ)["json"]
        self.assertEqual(b["model"], "microsoft/microsoft-decision-1")
        self.assertEqual(list(b["questions"]), [f"move_{m}" for m in BJ["legal"]])
        self.assertTrue(all(q["type"] == "noul" and q["instructions"] for q in b["questions"].values()))
        self.assertEqual(list(ec.build_request("or-d1-noul", DB)["json"]["questions"]), ["move_doubt"])

    def test_chat_candidates(self):
        for c, model in (("or-haiku", "anthropic/claude-haiku-5.5"), ("or-flashlite", "google/gemini-3.5-flash-lite")):
            for item, enum in ((BJ, ["hit", "stand", "double", "talk"]), (DB, ["doubt", "talk"])):
                b = ec.build_request(c, item)["json"]
                self.assertEqual((b["model"], b["temperature"]), (model, 0))
                self.assertEqual(b["response_format"]["type"], "json_schema")
                self.assertEqual(sorted(find_key(b["response_format"], "enum")), sorted(enum))
                self.assertIn(item["text"], json.dumps(b["messages"], ensure_ascii=False))

    def test_request_never_holds_a_key_value(self):
        old = {k: os.environ.get(k) for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY")}
        os.environ["OPENAI_API_KEY"], os.environ["OPENROUTER_API_KEY"] = FAKE_OAI, FAKE_OR
        try:
            for c in CANDS:
                for item in (BJ, DB):
                    blob = json.dumps(ec.build_request(c, item, 2), ensure_ascii=False)
                    self.assertNotIn(FAKE_OAI, blob)
                    self.assertNotIn(FAKE_OR, blob)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class ParseAnswerTests(unittest.TestCase):
    def oai_choice(self, item, choice, probs, name=None):
        return {"answers": [{"type": "choice", "name": name or qname("oai-choice", item), "choice": choice,
                             "probabilities": [{"value": v, "probability": p} for v, p in probs.items()], "confidence": 0.8}]}

    def test_oai_choice(self):
        r = ec.parse_answer("oai-choice", BJ, self.oai_choice(BJ, "hit", {"hit": 0.7, "stand": 0.2, "other": 0.1}))
        self.assertEqual((r["answer"], r["refusal"]), ("hit", False))
        self.assertAlmostEqual(r["p"], 0.7)
        self.assertEqual(r["probs"], {"hit": 0.7, "stand": 0.2, "talk": 0.1})
        self.assertEqual((r["cost"], r["tokens"]), (None, None))

    def test_oai_choice_other_is_talk(self):
        r = ec.parse_answer("oai-choice", BJ, self.oai_choice(BJ, "other", {"hit": 0.1, "other": 0.9}))
        self.assertEqual(r["answer"], "talk")
        self.assertAlmostEqual(r["p"], 0.9)

    def test_oai_choice_illegal_move_is_none(self):
        r = ec.parse_answer("oai-choice", BJ, self.oai_choice(BJ, "surrender", {"surrender": 0.9, "other": 0.1}))
        self.assertIsNone(r["answer"])
        self.assertFalse(r["refusal"])

    def test_oai_choice_doubt(self):
        r = ec.parse_answer("oai-choice", DB, self.oai_choice(DB, "doubt", {"doubt": 0.95, "other": 0.05}))
        self.assertEqual(r["answer"], "doubt")

    def test_oai_refusal(self):
        r = ec.parse_answer("oai-choice", BJ, {"answers": [{"type": "refusal", "name": qname("oai-choice", BJ)}]})
        self.assertEqual((r["answer"], r["refusal"], r["p"]), (None, True, None))

    def predicate(self, hit, stand, double):
        return {"answers": [{"type": "predicate", "name": f"move_{m}", "probability": p}
                            for m, p in (("hit", hit), ("stand", stand), ("double", double))]}

    def test_oai_predicate_best_move_over_tau(self):
        r = ec.parse_answer("oai-predicate", BJ, self.predicate(0.7, 0.2, 0.1))
        self.assertEqual(r["answer"], "hit")
        self.assertAlmostEqual(r["p"], 0.7)
        self.assertEqual(r["probs"], {"hit": 0.7, "stand": 0.2, "double": 0.1})

    def test_oai_predicate_under_tau_is_talk_and_boundary_is_move(self):
        r = ec.parse_answer("oai-predicate", BJ, self.predicate(0.4, 0.2, 0.1))
        self.assertEqual(r["answer"], "talk")
        self.assertAlmostEqual(r["p"], 0.4)
        self.assertEqual(ec.parse_answer("oai-predicate", BJ, self.predicate(0.5, 0.2, 0.1))["answer"], "hit")
        self.assertEqual(ec.parse_answer("oai-predicate", BJ, self.predicate(0.4, 0.2, 0.1), tau=0.3)["answer"], "hit")

    def test_oai_predicate_refusal(self):
        resp = {"answers": [{"type": "refusal", "name": "move_hit"}]}
        r = ec.parse_answer("oai-predicate", BJ, resp)
        self.assertEqual((r["answer"], r["refusal"]), (None, True))

    def test_oai_predicate_doubt(self):
        resp = {"answers": [{"type": "predicate", "name": "move_doubt", "probability": 0.9}]}
        self.assertEqual(ec.parse_answer("oai-predicate", DB, resp)["answer"], "doubt")
        resp = {"answers": [{"type": "predicate", "name": "move_doubt", "probability": 0.1}]}
        self.assertEqual(ec.parse_answer("oai-predicate", DB, resp)["answer"], "talk")

    USAGE = {"cost": 0.0000042, "input_tokens": 100, "output_tokens": 5}

    def test_d1_choice(self):
        resp = {"answers": {qname("or-d1-choice", BJ): {"type": "choice", "choice": "stand", "confidence": 0.9,
                                                       "probabilities": {"hit": 0.1, "stand": 0.8, "other": 0.1}}},
                "id": "x", "model": "microsoft/microsoft-decision-1", "provider": "Azure", "usage": self.USAGE}
        r = ec.parse_answer("or-d1-choice", BJ, resp)
        self.assertEqual(r["answer"], "stand")
        self.assertAlmostEqual(r["p"], 0.8)
        self.assertEqual(r["probs"], {"hit": 0.1, "stand": 0.8, "talk": 0.1})
        self.assertAlmostEqual(r["cost"], 0.0000042)
        self.assertEqual(r["tokens"], 105)

    def test_d1_choice_other_and_illegal(self):
        n = qname("or-d1-choice", BJ)
        mk = lambda c: {"answers": {n: {"type": "choice", "choice": c, "confidence": 0.5, "probabilities": {c: 0.6}}}}  # noqa: E731
        self.assertEqual(ec.parse_answer("or-d1-choice", BJ, mk("other"))["answer"], "talk")
        self.assertIsNone(ec.parse_answer("or-d1-choice", BJ, mk("surrender"))["answer"])

    def test_d1_noul(self):
        resp = {"answers": {"move_hit": {"type": "noul", "noul": 0.2}, "move_stand": {"type": "noul", "noul": 0.9},
                            "move_double": {"type": "noul", "noul": 0.1}}, "usage": self.USAGE}
        r = ec.parse_answer("or-d1-noul", BJ, resp)
        self.assertEqual(r["answer"], "stand")
        self.assertAlmostEqual(r["p"], 0.9)
        self.assertEqual(r["probs"], {"hit": 0.2, "stand": 0.9, "double": 0.1})
        self.assertEqual(r["tokens"], 105)
        low = {"answers": {m: {"type": "noul", "noul": 0.1} for m in ("move_hit", "move_stand", "move_double")}}
        r = ec.parse_answer("or-d1-noul", BJ, low)
        self.assertEqual(r["answer"], "talk")
        self.assertIsNone(r["cost"])

    def chat(self, content):
        return {"choices": [{"message": {"content": content}}], "usage": {"cost": 0.001, "prompt_tokens": 90, "completion_tokens": 5, "total_tokens": 95}}

    def test_chat_enum_answer_and_usage(self):
        for c in ("or-haiku", "or-flashlite"):
            r = ec.parse_answer(c, BJ, self.chat('{"answer": "double"}'))
            self.assertEqual(r["answer"], "double")
            self.assertAlmostEqual(r["cost"], 0.001)
            self.assertEqual((r["tokens"], r["refusal"]), (95, False))
            self.assertEqual(ec.parse_answer(c, BJ, self.chat('{"answer": "talk"}'))["answer"], "talk")
            self.assertEqual(ec.parse_answer(c, DB, self.chat('{"answer": "doubt"}'))["answer"], "doubt")

    def test_chat_malformed_or_illegal_is_none(self):
        for content in ("not json", "{}", '{"answer": "surrender"}', '{"answer": "doubt"}', ""):
            r = ec.parse_answer("or-haiku", BJ, self.chat(content))
            self.assertIsNone(r["answer"], content)
        self.assertIsNone(ec.parse_answer("or-haiku", DB, self.chat('{"answer": "hit"}'))["answer"])
        self.assertIsNone(ec.parse_answer("or-haiku", BJ, {"choices": [{"message": {"content": '{"answer":"hit"}'}}]})["cost"])


    def test_non_numeric_cost_becomes_none(self):
        for bad in ("n/a", "", None, [1]):
            r = ec.parse_answer("or-haiku", BJ, {"choices": [{"message": {"content": '{"answer": "hit"}'}}], "usage": {"cost": bad, "total_tokens": 9}})
            self.assertEqual(r["answer"], "hit")
            self.assertIsNone(r["cost"], bad)
            resp = {"answers": {"move_hit": {"type": "noul", "noul": 0.9}}, "usage": {"cost": bad, "input_tokens": 1, "output_tokens": 1}}
            self.assertIsNone(ec.parse_answer("or-d1-noul", {**BJ, "legal": ["hit"]}, resp)["cost"], bad)

    def test_oai_predicate_without_names_matches_by_position(self):
        resp = {"answers": [{"type": "predicate", "probability": 0.1}, {"type": "predicate", "probability": 0.7}, {"type": "predicate", "probability": 0.2}]}
        r = ec.parse_answer("oai-predicate", BJ, resp)  # request order: hit, stand, double
        self.assertEqual(r["answer"], "stand")
        self.assertEqual(r["probs"], {"hit": 0.1, "stand": 0.7, "double": 0.2})
        r = ec.parse_answer("oai-predicate", DB, {"answers": [{"type": "predicate", "probability": 0.9}]})
        self.assertEqual(r["answer"], "doubt")


class PassesTests(unittest.TestCase):
    def test_rule(self):
        mv = {**BJ, "accept": ["stand"]}
        self.assertTrue(ec.passes(mv, "hit"))
        self.assertTrue(ec.passes(mv, "stand"))
        self.assertFalse(ec.passes(mv, "double"))
        self.assertFalse(ec.passes(mv, "talk"))
        self.assertFalse(ec.passes(mv, None))
        self.assertTrue(ec.passes(BJ_TALK, "talk"))
        self.assertTrue(ec.passes(BJ_TALK, "hit"))
        self.assertFalse(ec.passes(BJ_TALK, "stand"))
        self.assertFalse(ec.passes(BJ_TALK, None))

    def test_doubt(self):
        self.assertTrue(ec.passes(DB, "doubt"))
        self.assertFalse(ec.passes(DB, "talk"))
        nd = {**DB, "label": "talk"}
        self.assertTrue(ec.passes(nd, "talk"))
        self.assertFalse(ec.passes(nd, "doubt"))


class ScoreTests(unittest.TestCase):
    def test_bj_and_doubt_split_by_task(self):
        ko = {**BJ, "id": "a2", "lang": "ko"}
        kt = {**BJ_TALK, "id": "b2", "lang": "ko"}
        d_talk = {**DB, "id": "d2", "label": "talk", "lang": "ko"}
        d_doubt_ko = {**DB, "id": "d3", "lang": "ko"}
        items = [BJ, ko, BJ_TALK, kt, DB, d_talk, d_doubt_ko, {**BJ, "id": "a3"}, {**BJ_TALK, "id": "b3"}]
        results = [res("hit"), res("stand"), res("talk"), res("hit"), res("doubt"), res("talk"), res(None), res(None, refusal=True), res(None)]
        s = ec.score(items, results)
        self.assertEqual((s["move_pass"], s["move_n"], s["talk_silent"], s["talk_n"]), (1, 3, 1, 3))
        self.assertEqual((s["invalid"], s["refusals"]), (1, 1))
        self.assertEqual((s["doubt"]["invalid"], s["doubt"]["refusals"]), (1, 0))
        self.assertEqual(s["by_lang"]["en"], {"move_pass": 1, "move_n": 2, "talk_silent": 1, "talk_n": 2})
        self.assertEqual(s["by_lang"]["ko"], {"move_pass": 0, "move_n": 1, "talk_silent": 0, "talk_n": 1})
        d = s["doubt"]
        self.assertEqual((d["recall"], d["doubt_n"], d["silent"], d["talk_n"]), (1, 2, 1, 1))
        self.assertEqual(d["by_lang"]["ko"], {"recall": 0, "doubt_n": 1, "silent": 1, "talk_n": 1})
        self.assertEqual(d["by_lang"]["en"], {"recall": 1, "doubt_n": 1, "silent": 0, "talk_n": 0})


class StatsTests(unittest.TestCase):
    def test_latency_nearest_rank(self):
        self.assertEqual(ec.latency_summary([float(i) for i in range(10, 0, -1)]), {"p50": 5.0, "p95": 10.0, "n": 10})
        self.assertEqual(ec.latency_summary([float(i) for i in range(1, 21)]), {"p50": 10.0, "p95": 19.0, "n": 20})
        self.assertEqual(ec.latency_summary([0.3]), {"p50": 0.3, "p95": 0.3, "n": 1})
        self.assertEqual(ec.latency_summary([]), {"p50": None, "p95": None, "n": 0})

    def test_cost_per_1000(self):
        self.assertAlmostEqual(ec.cost_per_1000([0.001, None, 0.003]), 2.0)
        self.assertIsNone(ec.cost_per_1000([None, None]))
        self.assertIsNone(ec.cost_per_1000([]))

    def test_auroc(self):
        self.assertEqual(ec.auroc([0.9, 0.8, 0.3, 0.2], [True, True, False, False]), 1.0)
        self.assertEqual(ec.auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]), 0.0)
        self.assertEqual(ec.auroc([0.5, 0.5], [True, False]), 0.5)
        self.assertAlmostEqual(ec.auroc([0.8, 0.4, 0.6, 0.2], [True, True, False, False]), 0.75)
        self.assertAlmostEqual(ec.auroc([0.8, 0.6, 0.6, 0.2], [1, 0, 1, 0]), 0.875)
        self.assertIsNone(ec.auroc([0.8, 0.4], [True, True]))
        self.assertIsNone(ec.auroc([0.8, 0.4], [False, False]))
        self.assertIsNone(ec.auroc([], []))

    def test_calibration_buckets(self):
        ps = [0.1, 0.49, 0.5, 0.69, 0.7, 0.95, 0.99, 1.0, 1.0]
        ok = [False, True, True, True, False, True, True, True, False]
        b = ec.calibration_buckets(ps, ok)
        self.assertEqual([x["lo"] for x in b], [0.0, 0.5, 0.7, 0.9, 0.99])
        self.assertEqual(b[0]["hi"], 0.5)
        self.assertEqual([x["n"] for x in b], [2, 2, 1, 1, 3])
        self.assertEqual([x["accuracy"] for x in b[:4]], [0.5, 1.0, 0.0, 1.0])
        self.assertAlmostEqual(b[4]["accuracy"], 2 / 3)
        empty = ec.calibration_buckets([0.2], [True])
        self.assertEqual([x["n"] for x in empty], [1, 0, 0, 0, 0])
        self.assertEqual([x["accuracy"] for x in empty], [1.0, None, None, None, None])

    def test_tau_sweep(self):
        mv1 = {**BJ, "id": "m1"}
        mv2 = {**BJ, "id": "m2"}
        tk1 = {**BJ_TALK, "id": "k1", "accept": []}
        tk2 = {**BJ_TALK, "id": "k2", "accept": []}
        items = [mv1, mv2, tk1, tk2]
        answers_p = [("hit", 0.9), ("hit", 0.6), ("talk", 0.2), ("hit", 0.7)]
        out = ec.tau_sweep(items, answers_p, [0.0, 0.65, 0.95])
        self.assertEqual([o["tau"] for o in out], [0.0, 0.65, 0.95])
        self.assertEqual([o["move_rate"] for o in out], [1.0, 0.5, 0.0])
        self.assertEqual([o["talk_silent_rate"] for o in out], [0.5, 0.5, 1.0])

    def test_order_stability(self):
        out = ec.order_stability({"a": [("hit", 0.9), ("hit", 0.7), ("hit", 0.8)], "b": [("hit", 0.8), ("stand", 0.4), ("hit", 0.6)]})
        self.assertEqual(out["stable_share"], 0.5)
        self.assertAlmostEqual(out["mean_p_spread"], 0.3)


class EstimateTests(unittest.TestCase):
    def test_call_counts_for_all_items(self):
        e = ec.estimate(CANDS, ec.load_items(), shuffles=3)
        self.assertEqual(e["calls"], {"oai-choice": 933, "oai-predicate": 311, "or-d1-choice": 933, "or-d1-noul": 311, "or-haiku": 311, "or-flashlite": 311})
        self.assertEqual(e["total_calls"], 3110)
        self.assertEqual(set(e["per_candidate_usd"]), set(CANDS))
        self.assertTrue(all(v > 0 for v in e["per_candidate_usd"].values()))
        self.assertAlmostEqual(sum(e["per_candidate_usd"].values()), e["est_usd"])
        self.assertTrue(0 < e["est_usd"] < 0.50)

    def test_shuffles_and_subset(self):
        items = ec.load_items()[:10]
        e = ec.estimate(["oai-choice", "or-haiku"], items, shuffles=2)
        self.assertEqual(e["calls"], {"oai-choice": 20, "or-haiku": 10})
        self.assertEqual(e["total_calls"], 30)
        self.assertGreater(ec.estimate(["or-flashlite"], items)["est_usd"], ec.estimate(["or-d1-noul"], items)["est_usd"])


def fake_post_factory(log, cost=0.0, wall=0.25):
    """Canned response per URL, shaped from the request body. Records (url, body, key) in `log`."""
    def post(url, body, key):
        log.append((url, body, key))
        usage = {"cost": cost, "input_tokens": 50, "output_tokens": 2} if cost else None
        if url == URL_OAI:
            answers = []
            for q in body["questions"]:
                if q["type"] == "choice":
                    first = q["choices"][0]["value"]
                    answers.append({"type": "choice", "name": q["name"], "choice": first, "confidence": 0.5,
                                    "probabilities": [{"value": c["value"], "probability": 0.5 if c["value"] == first else 0.1} for c in q["choices"]]})
                else:
                    answers.append({"type": "predicate", "name": q["name"], "probability": 0.1})
            return {"answers": answers}, wall
        if url == URL_D1:
            answers = {}
            for n, q in body["questions"].items():
                if q["type"] == "choice":
                    answers[n] = {"type": "choice", "choice": "other", "confidence": 0.5, "probabilities": {"other": 0.9}}
                else:
                    answers[n] = {"type": "noul", "noul": 0.1}
            return {"answers": answers, "id": "r", "model": body["model"], "provider": "Fake", "usage": usage or {"cost": 0, "input_tokens": 1, "output_tokens": 1}}, wall
        out = {"choices": [{"message": {"content": '{"answer": "talk"}'}}]}
        if usage:
            out["usage"] = {"cost": cost, "prompt_tokens": 50, "completion_tokens": 2, "total_tokens": 52}
        return out, wall
    return post


class CliTests(unittest.TestCase):
    def run_main(self, argv, env=None, post=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = run_eval.main(argv, env=env, post=post)
        return rc, out.getvalue(), err.getvalue()

    def test_dry_run_prints_estimate_without_calls_or_keys(self):
        log = []
        with tempfile.TemporaryDirectory() as d:
            rc, out, _ = self.run_main(["--out", d], env={}, post=fake_post_factory(log))
            self.assertEqual(os.listdir(d), [])
        self.assertEqual(rc, 0)
        self.assertEqual(log, [])
        self.assertIn("3110", out)
        self.assertIn("933", out)

    def test_run_with_missing_key_fails_before_any_call(self):
        log = []
        with tempfile.TemporaryDirectory() as d:
            for env in ({}, {"OPENAI_API_KEY": FAKE_OAI}):
                rc, out, err = self.run_main(["--run", "--candidates", "or-haiku", "--limit", "3", "--out", d], env=env, post=fake_post_factory(log))
                self.assertNotEqual(rc, 0)
                self.assertNotIn(FAKE_OAI, out + err)
            self.assertEqual(os.listdir(d), [])
        self.assertEqual(log, [])

    def test_run_writes_results_without_key_text(self):
        log = []
        env = {"OPENAI_API_KEY": FAKE_OAI, "OPENROUTER_API_KEY": FAKE_OR}
        with tempfile.TemporaryDirectory() as d:
            rc, out, err = self.run_main(["--run", "--candidates", "or-haiku,oai-choice", "--limit", "3", "--shuffles", "2", "--out", d],
                                         env=env, post=fake_post_factory(log, cost=0.001))
            self.assertEqual(rc, 0, err)
            files = [p for p in Path(d).rglob("*") if p.is_file()]
            self.assertTrue(files)
            texts = [p.read_text(encoding="utf-8") for p in files]
            for t in texts:
                json.loads(t)
        # 3 items: or-haiku once each, oai-choice twice each
        self.assertEqual(sum(1 for u, _, _ in log if u == URL_CHAT), 3)
        self.assertEqual(sum(1 for u, _, _ in log if u == URL_OAI), 6)
        self.assertTrue(all(k == (FAKE_OR if u == URL_CHAT else FAKE_OAI) for u, _, k in log))
        for blob in [out, err, *texts]:
            self.assertNotIn(FAKE_OAI, blob)
            self.assertNotIn(FAKE_OR, blob)
            self.assertNotIn("Authorization", blob)

    def test_max_usd_aborts_once_budget_passed(self):
        log = []
        with tempfile.TemporaryDirectory() as d:
            rc, out, err = self.run_main(["--run", "--candidates", "or-haiku", "--limit", "10", "--max-usd", "0.015", "--out", d],
                                         env={"OPENROUTER_API_KEY": FAKE_OR}, post=fake_post_factory(log, cost=0.01))
        self.assertNotEqual(rc, 0)
        self.assertEqual(len(log), 2)  # 0.01, then 0.02 > 0.015: stop
        self.assertNotIn(FAKE_OR, out + err)


"""Review round 1 additions (FEAT-43). Interfaces pinned here:

  run_eval.http_post(url, body, key, transport=None) -> (json, wall_seconds); transport is an httpx transport (MockTransport). Non-2xx
      raises; str(exc) holds the status code and at most 200 chars of body, with the key scrubbed from the FULL body before truncation
      and Bearer/sk- looking text redacted. Redirects are not followed (302 raises). Sends Authorization: Bearer <key>.
  main() writes <out>/results.json = {"meta": {...}, "records": [record, ...]} after every call (or in a finally), so a crash in
      summarize or a KeyboardInterrupt keeps the paid records. record keys: item, task, candidate, k, answer, p, probs, refusal, cost,
      tokens, wall, error (None or str), cost_estimated (bool; True when the provider gave no usable cost and the estimate was used).
  evalcore.separation(kind, items, results) -> {'auroc', 'buckets'}; kind in 'choice'|'chat'|'predicate'|'noul'. results are parse_answer dicts
      parallel to items. choice/chat: p vs passes(). predicate/noul: p (best move probability) vs "item is a move row / doubt row".
  run_eval.summarize(cands, items, records, tau) -> {candidate: {task: {..., 'cost_per_1000_calls', 'cost_per_decision', 'errored_calls',
      'coverage': {'attempted', 'total'}, 'score': evalcore.score-like dict}}}. A task appears when the candidate has items of it. Items with no
      record count as invalid in score (pass-rate denominators are the full item count). Errored calls are invalid, never silent.
  main(): --run refuses (exit 2, no call) when the dry-run estimate exceeds --max-usd, naming both numbers; a dry run prints a line containing
      "exceeds" and returns 0. All-invalid candidate: stderr line with "WARNING" and the candidate id. Five consecutive call errors skip the rest
      of that candidate (counter resets on a success). --out inside the repository: exit 2, no call, nothing created.
"""
FAKE_BEARER = "Bearer FAKEtoken-ABCDEFGH-12345678"
FAKE_SK = "sk-FAKEFAKEFAKE-proj-0000000000"  # gitleaks:allow (fake)


def chat_resp(content='{"answer": "talk"}', cost=None):
    out = {"choices": [{"message": {"content": content}}]}
    if cost is not None:
        out["usage"] = {"cost": cost, "prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
    return out


class HttpPostTests(unittest.TestCase):
    def post(self, handler, key=FAKE_OR):
        return run_eval.http_post(URL_CHAT, {"a": 1}, key, transport=httpx.MockTransport(handler))

    def test_success_returns_json_and_wall_and_sends_bearer(self):
        seen = []

        def handler(req):
            seen.append((req.headers.get("authorization"), json.loads(req.content)))
            return httpx.Response(200, json={"ok": True})
        data, wall = self.post(handler)
        self.assertEqual(data, {"ok": True})
        self.assertGreaterEqual(wall, 0)
        self.assertEqual(seen, [(f"Bearer {FAKE_OR}", {"a": 1})])

    def test_error_body_scrubs_key_before_truncation(self):
        body = "x" * 180 + FAKE_OR + "y" * 100
        with self.assertRaises(Exception) as cm:
            self.post(lambda req: httpx.Response(401, text=body))
        msg = str(cm.exception)
        self.assertIn("401", msg)
        for i in range(len(FAKE_OR) - 7):
            self.assertNotIn(FAKE_OR[i:i + 8], msg)
        self.assertLessEqual(msg.count("x"), 200)
        self.assertLessEqual(len(msg), 300)

    def test_error_body_redacts_bearer_and_sk_text(self):
        body = f"bad header {FAKE_BEARER} and {FAKE_SK} end"
        with self.assertRaises(Exception) as cm:
            self.post(lambda req: httpx.Response(500, text=body))
        msg = str(cm.exception)
        self.assertIn("500", msg)
        self.assertNotIn("FAKEtoken", msg)
        self.assertNotIn("FAKEFAKEFAKE", msg)

    def test_redirect_is_not_followed(self):
        calls = []

        def handler(req):
            calls.append(str(req.url))
            return httpx.Response(302, headers={"location": "https://evil.invalid/x"})
        with self.assertRaises(Exception) as cm:
            self.post(handler)
        self.assertIn("302", str(cm.exception))
        self.assertEqual(len(calls), 1)


class SeparationTests(unittest.TestCase):
    def r(self, answer, p):
        return res(answer, p=p)

    def test_predicate_uses_move_versus_talk_not_correctness(self):
        mv = lambda i: {**BJ, "id": i}  # noqa: E731
        tk = lambda i: {**BJ_TALK, "id": i, "accept": []}  # noqa: E731
        items = [mv("m1"), mv("m2"), tk("t1"), tk("t2"), tk("t3")]
        results = [self.r("hit", 0.9), self.r("talk", 0.3), self.r("talk", 0.05), self.r("hit", 0.6), self.r("talk", 0.02)]
        for kind in ("predicate", "noul"):
            s = ec.separation(kind, items, results)
            self.assertAlmostEqual(s["auroc"], 5 / 6)  # positives .9,.3 vs negatives .05,.6,.02
            b = s["buckets"]
            self.assertEqual([x["n"] for x in b], [3, 1, 0, 1, 0])
            # "accuracy" is the share of move rows: the correctly-silent chatter rows at p .05/.02 are NOT counted as correct
            self.assertAlmostEqual(b[0]["accuracy"], 1 / 3)
            self.assertEqual((b[1]["accuracy"], b[3]["accuracy"]), (0.0, 1.0))

    def test_predicate_doubt_rows(self):
        items = [DB, {**DB, "id": "d2", "label": "talk"}]
        s = ec.separation("noul", items, [self.r("doubt", 0.8), self.r("talk", 0.1)])
        self.assertEqual(s["auroc"], 1.0)

    def test_choice_and_chat_use_correctness(self):
        a, b = {**BJ, "id": "a"}, {**BJ, "id": "b"}
        c, d = {**BJ_TALK, "id": "c", "accept": []}, {**BJ_TALK, "id": "d", "accept": []}
        results = [self.r("hit", 0.9), self.r("stand", 0.8), self.r("talk", 0.4), self.r("hit", 0.6)]
        for kind in ("choice", "chat"):
            s = ec.separation(kind, [a, b, c, d], results)
            self.assertAlmostEqual(s["auroc"], 0.5)  # correct: .9, .4; wrong: .8, .6
            self.assertEqual([x["n"] for x in s["buckets"]], [1, 1, 1, 1, 0])
            self.assertEqual([x["accuracy"] for x in s["buckets"]], [1.0, 0.0, 0.0, 1.0, None])

    def test_results_without_p_are_skipped(self):
        s = ec.separation("choice", [BJ, BJ_TALK], [self.r(None, None), self.r("talk", 0.9)])
        self.assertIsNone(s["auroc"])
        self.assertEqual(sum(x["n"] for x in s["buckets"]), 1)


def rec(item, cand, k=0, answer="hit", p=None, cost=0.001, error=None, wall=0.1, probs=None):
    return {"item": item["id"], "task": item["task"], "candidate": cand, "k": k, "answer": answer, "p": p, "probs": probs or {},
            "refusal": False, "cost": cost, "tokens": 10, "wall": wall if error is None else None, "error": error, "cost_estimated": False}


def bj_items(n, label="hit", prefix="s"):
    return [{**BJ, "id": f"{prefix}{i}", "label": label} for i in range(n)]


class SummarizeTests(unittest.TestCase):
    def test_cost_per_decision_and_errored_calls(self):
        a, b = bj_items(2)
        recs = [rec(a, "oai-choice", k, p=0.9) for k in range(3)] + [rec(b, "oai-choice", 0, p=0.9), rec(b, "oai-choice", 1, p=0.9),
                                                                      rec(b, "oai-choice", 2, answer=None, cost=None, error="RuntimeError: boom")]
        t = run_eval.summarize(["oai-choice"], [a, b], recs, 0.5)["oai-choice"]["bj"]
        self.assertAlmostEqual(t["cost_per_1000_calls"], 1.0)  # mean of the five good calls x 1000
        self.assertAlmostEqual(t["cost_per_decision"], 0.0025)  # 0.005 total / 2 decisions, not / 6 calls
        self.assertEqual(t["errored_calls"], 1)

    def test_missing_items_count_as_invalid_with_coverage(self):
        items = bj_items(4)
        recs = [rec(items[0], "or-haiku"), rec(items[1], "or-haiku")]
        t = run_eval.summarize(["or-haiku"], items, recs, 0.5)["or-haiku"]["bj"]
        self.assertEqual(t["coverage"], {"attempted": 2, "total": 4})
        self.assertEqual((t["score"]["move_pass"], t["score"]["move_n"], t["score"]["invalid"]), (2, 4, 2))

    def test_errored_and_missing_records_never_silent_and_do_not_raise(self):
        mv, tk = bj_items(1)[0], {**BJ_TALK, "id": "tk", "accept": []}
        tk2 = {**BJ_TALK, "id": "tk2", "accept": []}
        for cand in ("or-haiku", "or-d1-noul", "oai-choice"):
            recs = [rec(mv, cand, answer=None, cost=None, error="RuntimeError: x"), rec(tk, cand, answer=None, cost=None, error="RuntimeError: x")]
            t = run_eval.summarize([cand], [mv, tk, tk2], recs, 0.5)[cand]["bj"]
            self.assertEqual((t["score"]["talk_silent"], t["score"]["talk_n"], t["score"]["move_pass"]), (0, 2, 0), cand)
            self.assertEqual(t["score"]["invalid"], 3, cand)
            self.assertEqual(t["errored_calls"], 2, cand)
            self.assertEqual(t["coverage"], {"attempted": 2, "total": 3}, cand)


class CliRoundOneTests(unittest.TestCase):
    run_main = CliTests.run_main
    ENV = {"OPENAI_API_KEY": FAKE_OAI, "OPENROUTER_API_KEY": FAKE_OR}

    def records(self, d):
        return json.loads((Path(d) / "results.json").read_text(encoding="utf-8"))["records"]

    def test_results_survive_summarize_failure(self):
        log = []
        with tempfile.TemporaryDirectory() as d, mock.patch.object(run_eval, "summarize", side_effect=ValueError("boom")):
            try:
                self.run_main(["--run", "--candidates", "or-haiku", "--limit", "3", "--out", d], env=self.ENV, post=fake_post_factory(log))
            except ValueError:
                pass
            self.assertEqual(len(self.records(d)), 3)
        self.assertEqual(len(log), 3)

    def test_non_numeric_cost_in_response_is_survivable(self):
        def post(url, body, key):
            return chat_resp(cost="n/a"), 0.1
        with tempfile.TemporaryDirectory() as d:
            rc, _, _ = self.run_main(["--run", "--candidates", "or-haiku", "--limit", "3", "--out", d], env=self.ENV, post=post)
            recs = self.records(d)
        self.assertIsInstance(rc, int)
        self.assertEqual(len(recs), 3)
        self.assertTrue(all(isinstance(r["cost"], float) and r["cost"] > 0 and r["cost_estimated"] is True for r in recs))

    def test_keyboard_interrupt_keeps_earlier_records(self):
        log = []
        inner = fake_post_factory(log)

        def post(url, body, key):
            if len(log) == 2:
                raise KeyboardInterrupt
            return inner(url, body, key)
        with tempfile.TemporaryDirectory() as d:
            try:
                self.run_main(["--run", "--candidates", "or-haiku", "--limit", "5", "--out", d], env=self.ENV, post=post)
            except KeyboardInterrupt:
                pass
            self.assertEqual(len(self.records(d)), 2)

    def test_cost_estimated_flag_follows_provider_usage(self):
        with tempfile.TemporaryDirectory() as d:
            self.run_main(["--run", "--candidates", "or-haiku", "--limit", "2", "--out", d], env=self.ENV, post=fake_post_factory([], cost=0.001))
            self.assertEqual([(r["cost"], r["cost_estimated"]) for r in self.records(d)], [(0.001, False)] * 2)
        with tempfile.TemporaryDirectory() as d:
            self.run_main(["--run", "--candidates", "or-haiku", "--limit", "2", "--out", d], env=self.ENV, post=fake_post_factory([], cost=0.0))
            recs = self.records(d)
            self.assertTrue(all(r["cost_estimated"] is True and r["cost"] > 0 for r in recs))

    def test_all_invalid_candidate_warns_on_stderr(self):
        def bad(url, body, key):
            return chat_resp("not json"), 0.1
        with tempfile.TemporaryDirectory() as d:
            rc, out, err = self.run_main(["--run", "--candidates", "or-haiku", "--limit", "3", "--out", d], env=self.ENV, post=bad)
        self.assertEqual(rc, 0)
        self.assertTrue(any("WARNING" in ln and "or-haiku" in ln for ln in err.splitlines()), err)
        with tempfile.TemporaryDirectory() as d:
            _, _, err = self.run_main(["--run", "--candidates", "or-haiku", "--limit", "3", "--out", d], env=self.ENV, post=fake_post_factory([]))
        self.assertNotIn("WARNING", err)

    def test_run_refuses_when_estimate_exceeds_budget(self):
        args = ["--candidates", "oai-choice", "--shuffles", "100", "--max-usd", "0.25"]
        est = ec.estimate(["oai-choice"], ec.load_items(), 100)["est_usd"]
        self.assertGreater(est, 0.25)
        log = []
        with tempfile.TemporaryDirectory() as d:
            rc, out, err = self.run_main(["--run", *args, "--out", d], env=self.ENV, post=fake_post_factory(log))
            self.assertEqual(os.listdir(d), [])
        self.assertEqual((rc, log), (2, []))
        msg = out + err
        self.assertIn("0.25", msg)
        self.assertTrue(any(f"{est:.{n}f}" in msg for n in (2, 3, 4)), msg)

    def test_dry_run_warns_when_estimate_exceeds_budget(self):
        log = []
        with tempfile.TemporaryDirectory() as d:
            rc, out, err = self.run_main(["--candidates", "oai-choice", "--shuffles", "100", "--max-usd", "0.25", "--out", d], env={}, post=fake_post_factory(log))
        self.assertEqual((rc, log), (0, []))
        self.assertIn("exceeds", out + err)
        with tempfile.TemporaryDirectory() as d:
            rc, out, err = self.run_main(["--candidates", "or-haiku", "--limit", "3", "--out", d], env={}, post=fake_post_factory(log))
        self.assertNotIn("exceeds", out + err)

    def test_five_consecutive_errors_skip_the_candidate(self):
        log = []

        def post(url, body, key):
            log.append(1)
            raise RuntimeError("boom")
        with tempfile.TemporaryDirectory() as d:
            rc, out, err = self.run_main(["--run", "--candidates", "or-haiku", "--limit", "8", "--out", d], env=self.ENV, post=post)
            self.assertEqual(len(log), 5)
            self.assertEqual(len(self.records(d)), 5)
            summary = json.loads((Path(d) / "summary.json").read_text(encoding="utf-8"))["summary"]["or-haiku"]
        self.assertEqual(sum(t["coverage"]["attempted"] for t in summary.values()), 5)
        self.assertEqual(sum(t["coverage"]["total"] for t in summary.values()), 8)
        self.assertEqual(sum(t["score"]["invalid"] for t in summary.values() if "move_n" in t["score"]) +
                         sum(t["score"]["invalid"] for t in summary.values() if "move_n" not in t["score"]), 8)
        self.assertNotIn(FAKE_OR, out + err)

    def test_error_counter_resets_after_a_success(self):
        log = []
        inner = fake_post_factory([])

        def post(url, body, key):
            log.append(1)
            if len(log) % 5 != 0:
                raise RuntimeError("boom")
            return inner(url, body, key)
        with tempfile.TemporaryDirectory() as d:
            self.run_main(["--run", "--candidates", "or-haiku", "--limit", "10", "--out", d], env=self.ENV, post=post)
        self.assertEqual(len(log), 10)  # 4 errors, success, 4 errors, success: never 5 in a row

    def test_out_inside_repository_is_refused(self):
        rel = Path("prototypes") / "_decision_eval_test_out"
        cwd = os.getcwd()
        try:
            os.chdir(ROOT)
            for out in (str(rel), str(ROOT / rel), str(ROOT)):
                for extra in ([], ["--run", "--candidates", "or-haiku", "--limit", "2"]):
                    log = []
                    rc, _, _ = self.run_main([*extra, "--out", out], env=self.ENV, post=fake_post_factory(log))
                    self.assertEqual((rc, log), (2, []), (out, extra))
                    self.assertFalse((ROOT / rel).exists())
        finally:
            os.chdir(cwd)
            shutil.rmtree(ROOT / rel, ignore_errors=True)

    def test_out_outside_repository_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            out = str(Path(d) / "sub")
            rc, _, _ = self.run_main(["--run", "--candidates", "or-haiku", "--limit", "1", "--out", out], env=self.ENV, post=fake_post_factory([]))
            self.assertEqual(rc, 0)
            self.assertTrue((Path(out) / "results.json").exists())

class FramingTests(unittest.TestCase):
    """Prompt fairness: decision candidates get the same table-chat context the chat baselines have in their system prompt.
    evalcore.context_for(item) -> str is shared by oai-* (framed `input` string) and or-d1-* (`state` = {"context", "message"})."""
    OAI = ("oai-choice", "oai-predicate")
    D1 = ("or-d1-choice", "or-d1-noul")

    def test_context_for(self):
        bj, db = ec.context_for(BJ), ec.context_for(DB)
        self.assertIsInstance(bj, str)
        for needle in ("blackjack", "Discord", "table", "turn"):
            self.assertIn(needle, bj)
        self.assertIn("I Doubt It", db)
        self.assertNotIn("blackjack", db)
        self.assertEqual(ec.context_for(BJ), ec.context_for({**BJ, "text": "other", "id": "zz"}))

    def test_openai_input_is_framed_string(self):
        for cand in self.OAI:
            for item in (BJ, DB, {**BJ, "text": "카드 한 장 더"}):
                inp = ec.build_request(cand, item)["json"]["input"]
                self.assertIsInstance(inp, str, cand)
                self.assertIn(ec.context_for(item), inp)
                tail = inp.rstrip()[-(len(item["text"]) + 2):]
                self.assertEqual(tail[1:-1], item["text"], cand)
                self.assertEqual(tail[0], tail[-1])
                self.assertIn(tail[0], "\"“”'")
        self.assertIn("blackjack", ec.build_request("oai-choice", BJ)["json"]["input"])
        self.assertIn("I Doubt It", ec.build_request("oai-predicate", DB)["json"]["input"])

    def test_d1_state_is_context_and_message_dict(self):
        for cand in self.D1:
            for item in (BJ, DB):
                self.assertEqual(ec.build_request(cand, item)["json"]["state"], {"context": ec.context_for(item), "message": item["text"]}, cand)

    def test_chat_unchanged(self):
        for cand in ("or-haiku", "or-flashlite"):
            b = ec.build_request(cand, BJ)["json"]
            self.assertNotIn("state", b)
            self.assertNotIn("input", b)
            self.assertIn("messages", b)


class RawExcerptTests(unittest.TestCase):
    def chat(self, content):
        return {"choices": [{"message": {"content": content}}]}

    def test_invalid_chat_answers_keep_first_80_chars(self):
        long_bad = '{"answer": "surrender", "why": "' + "z" * 100 + '"}'
        for content in ("not json " + "q" * 120, long_bad):
            r = ec.parse_answer("or-haiku", BJ, self.chat(content))
            self.assertIsNone(r["answer"])
            self.assertEqual(r["raw"], content[:80])
        self.assertEqual(ec.parse_answer("or-flashlite", BJ, self.chat("oops"))["raw"], "oops")

    def test_raw_is_none_without_chat_content_or_for_other_kinds(self):
        self.assertIsNone(ec.parse_answer("or-haiku", BJ, {"choices": []})["raw"])
        self.assertIsNone(ec.parse_answer("or-haiku", BJ, {})["raw"])
        r = ec.parse_answer("oai-predicate", BJ, {"answers": [{"type": "predicate", "name": "move_hit", "probability": 0.9}]})
        self.assertIsNone(r["raw"])

    def test_results_json_keeps_raw_only_for_invalid_records(self):
        bad = "not json " + "q" * 120
        calls = []

        def post(url, body, key):
            calls.append(1)
            return chat_resp(bad if len(calls) == 1 else '{"answer": "talk"}'), 0.1
        with tempfile.TemporaryDirectory() as d:
            self._run(d, post)
            recs = json.loads((Path(d) / "results.json").read_text(encoding="utf-8"))["records"]
        self.assertEqual(len(recs), 3)
        self.assertIsNone(recs[0]["answer"])
        self.assertEqual(recs[0]["raw"], bad[:80])
        for r in recs[1:]:
            self.assertEqual(r["answer"], "talk")
            self.assertIsNone(r.get("raw"))

    def _run(self, d, post):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_eval.main(["--run", "--candidates", "or-haiku", "--limit", "3", "--out", d], env={"OPENROUTER_API_KEY": FAKE_OR}, post=post)


class ChatReasoningTests(unittest.TestCase):
    """Smoke-run findings: Haiku 5.5 reasons by default and spent all 64 tokens on hidden reasoning; Flash-Lite reasoning is mandatory."""

    def test_haiku_reasoning_disabled_and_small_budget(self):
        for item in (BJ, DB):
            b = ec.build_request("or-haiku", item)["json"]
            self.assertEqual(b["reasoning"], {"enabled": False})
            self.assertEqual(b["max_tokens"], 64)

    def test_flashlite_minimal_hidden_reasoning(self):
        for item in (BJ, DB):
            b = ec.build_request("or-flashlite", item)["json"]
            self.assertEqual(b["reasoning"], {"effort": "minimal", "exclude": True})
            self.assertEqual(b["max_tokens"], 256)

    def test_parse_answer_finish_field(self):
        def resp(content, finish):
            return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}
        self.assertEqual(ec.parse_answer("or-haiku", BJ, resp('{"answer": "hit"}', "stop"))["finish"], "stop")
        r = ec.parse_answer("or-haiku", BJ, resp("", "length"))
        self.assertEqual((r["answer"], r["finish"]), (None, "length"))
        self.assertIsNone(ec.parse_answer("or-flashlite", BJ, {"choices": [{"message": {"content": "{}"}}]})["finish"])
        self.assertIsNone(ec.parse_answer("or-haiku", BJ, {})["finish"])
        self.assertIsNone(ec.parse_answer("oai-predicate", BJ, {"answers": [{"type": "predicate", "name": "move_hit", "probability": 0.9}]})["finish"])

    def test_results_json_keeps_finish_next_to_raw_for_invalid_records_only(self):
        calls = []

        def post(url, body, key):
            calls.append(1)
            if len(calls) == 1:
                return {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}, 0.1
            return {"choices": [{"message": {"content": '{"answer": "talk"}'}, "finish_reason": "stop"}]}, 0.1
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                run_eval.main(["--run", "--candidates", "or-haiku", "--limit", "3", "--out", d], env={"OPENROUTER_API_KEY": FAKE_OR}, post=post)
            recs = json.loads((Path(d) / "results.json").read_text(encoding="utf-8"))["records"]
        self.assertEqual((recs[0]["answer"], recs[0]["finish"], recs[0]["raw"]), (None, "length", ""))
        for r in recs[1:]:
            self.assertEqual(r["answer"], "talk")
            self.assertIsNone(r.get("finish"))


if __name__ == "__main__":
    unittest.main()
