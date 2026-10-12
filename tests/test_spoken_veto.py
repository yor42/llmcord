"""FEAT-49a: the decision-model veto (llmcord_core/spoken/veto.py and veto_client.py). Offline: httpx.MockTransport, fake keys, fake clock.

These tests DEFINE the interface; the modules are written to satisfy them.

veto.py (pure): ORIGIN, PATH, MODEL, TAU, MAX_CHARS, VetoError, build_request(text, legal), parse_response(data, legal),
    apply_policy(band, moves, probs, tau).
veto_client.py: VetoClient(api_key, *, transport, clock, timeout, connect_timeout, max_bytes, fail_limit, cooldown) with
    async ask(text, legal), breaker_open, async close(); from_env(env, **kw).
"""
import json
import logging
import math
import unittest

import httpx

from llmcord_core.spoken import veto
from llmcord_core.spoken.veto import VetoError, apply_policy, build_request, parse_response
from llmcord_core.spoken.veto_client import VetoClient, from_env

KEY = "FAKE-OPENROUTER-not-a-real-key-0002"  # gitleaks:allow
TEXT = "hmm maybe zebra-sandwich one more card"
URL = "https://openrouter.ai/api/alpha/decisions"
CONTEXT = ("This message was typed in a Discord blackjack table chat by the player whose turn it is, as a possible move for the current hand "
           "(it may be English, Korean or mixed).")
HIT_Q = "Is the player choosing to take another card right now, as an instruction to act (not a question, a joke or chit-chat)?"
STAND_Q = "Is the player choosing to keep the hand, take no more cards right now, as an instruction to act (not a question, a joke or chit-chat)?"


def answers(**probs):
    return {"answers": {f"move_{m}": {"type": "noul", "noul": p} for m, p in probs.items()}}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Recorder:
    """MockTransport handler that records requests and replies from a script (a response, an exception, or a callable)."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item(request) if callable(item) else item

    def transport(self):
        return httpx.MockTransport(self)


def ok(**probs):
    return httpx.Response(200, json=answers(**probs))


def client_for(rec, clock=None, **kw):
    return VetoClient(KEY, transport=rec.transport(), clock=clock or Clock(), **kw)


class ConstantsTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(veto.ORIGIN + veto.PATH, URL)
        self.assertEqual(veto.MODEL, "microsoft/microsoft-decision-1")
        self.assertEqual(veto.TAU, 0.5)
        self.assertEqual(veto.MAX_CHARS, 200)

    def test_veto_error_is_exception(self):
        self.assertTrue(issubclass(VetoError, Exception))


class BuildRequestTests(unittest.TestCase):
    def test_body_is_exactly_model_state_questions(self):
        body = build_request(TEXT, ["hit", "stand"])
        self.assertEqual(set(body), {"model", "state", "questions"})
        self.assertEqual(body["model"], veto.MODEL)
        self.assertEqual(body["state"], {"context": CONTEXT, "message": TEXT})
        self.assertEqual(body["questions"], {
            "move_hit": {"type": "noul", "instructions": HIT_Q},
            "move_stand": {"type": "noul", "instructions": STAND_Q},
        })

    def test_question_per_legal_move_only(self):
        body = build_request("double", ["double"])
        self.assertEqual(list(body["questions"]), ["move_double"])
        self.assertIn("double the bet and take exactly one more card", body["questions"]["move_double"]["instructions"])

    def test_body_is_json_serialisable_and_carries_nothing_else(self):
        raw = json.dumps(build_request("안녕 hit", ["hit"]), ensure_ascii=False)
        self.assertIn("안녕 hit", raw)
        self.assertNotIn("guild", raw)
        self.assertNotIn("channel", raw)

    def test_does_not_mutate_legal(self):
        legal = ["hit", "stand"]
        build_request(TEXT, legal)
        self.assertEqual(legal, ["hit", "stand"])

    def test_rejects_bad_input(self):
        for text, legal in (("hit", []), ("", ["hit"]), ("x" * 201, ["hit"])):
            with self.subTest(text=text[:5], legal=legal), self.assertRaises(VetoError):
                build_request(text, legal)

    def test_exactly_max_chars_is_accepted(self):
        self.assertEqual(build_request("x" * 200, ["hit"])["state"]["message"], "x" * 200)


class ParseResponseTests(unittest.TestCase):
    def test_reads_noul_per_legal_move(self):
        self.assertEqual(parse_response(answers(hit=0.9, stand=0.1), ["hit", "stand"]), {"hit": 0.9, "stand": 0.1})

    def test_integer_and_boundary_values(self):
        self.assertEqual(parse_response(answers(hit=0, stand=1), ["hit", "stand"]), {"hit": 0.0, "stand": 1.0})

    def test_extra_answers_ignored(self):
        data = answers(hit=0.7, double=0.9)
        self.assertEqual(parse_response(data, ["hit"]), {"hit": 0.7})

    def test_failures(self):
        good = answers(hit=0.7, stand=0.2)
        cases = {
            "not a dict": [],
            "none": None,
            "no answers": {},
            "answers not a dict": {"answers": [1]},
            "missing legal move": answers(hit=0.7),
            "answer not a dict": {"answers": {"move_hit": 0.5, "move_stand": 0.1}},
            "no noul key": {"answers": {**good["answers"], "move_stand": {"type": "noul"}}},
            "refusal": {"answers": {**good["answers"], "move_stand": {"type": "refusal", "refusal": "no"}}},
            "refusal with noul": {"answers": {**good["answers"], "move_stand": {"type": "refusal", "noul": 0.5}}},
            "string value": answers(hit=0.7, stand="0.2"),
            "none value": answers(hit=0.7, stand=None),
            "bool true": answers(hit=0.7, stand=True),
            "bool false": answers(hit=0.7, stand=False),
            "nan": answers(hit=0.7, stand=math.nan),
            "inf": answers(hit=0.7, stand=math.inf),
            "-inf": answers(hit=0.7, stand=-math.inf),
            "above 1": answers(hit=0.7, stand=1.01),
            "below 0": answers(hit=0.7, stand=-0.01),
        }
        for name, data in cases.items():
            with self.subTest(name), self.assertRaises(VetoError):
                parse_response(data, ["hit", "stand"])

    def test_empty_answer_dict_value_raises(self):
        with self.assertRaises(VetoError):
            parse_response({"answers": {"move_hit": {}}}, ["hit"])


class ApplyPolicyTests(unittest.TestCase):
    def test_silent_passes_through_without_probs(self):
        self.assertEqual(apply_policy("silent", (), {}), ("silent", ()))
        self.assertEqual(apply_policy("silent", ("hit",), {}), ("silent", ("hit",)))

    def test_sure_stays_sure_when_same_single_move_kept(self):
        self.assertEqual(apply_policy("sure", ("hit",), {"hit": 0.9}), ("sure", ("hit",)))

    def test_sure_single_move_dropped_goes_silent(self):
        self.assertEqual(apply_policy("sure", ("hit",), {"hit": 0.1}), ("silent", ()))

    def test_unsure_never_promoted_to_sure(self):
        self.assertEqual(apply_policy("unsure", ("hit",), {"hit": 0.99}), ("unsure", ("hit",)))

    def test_sure_with_two_moves_one_dropped_is_unsure(self):
        self.assertEqual(apply_policy("sure", ("hit", "stand"), {"hit": 0.9, "stand": 0.1}), ("unsure", ("hit",)))

    def test_orders_by_probability_desc(self):
        self.assertEqual(apply_policy("unsure", ("hit", "stand", "double"), {"hit": 0.6, "stand": 0.9, "double": 0.7}),
                         ("unsure", ("stand", "double", "hit")))

    def test_sure_reordered_demotes_to_unsure(self):
        self.assertEqual(apply_policy("sure", ("hit", "stand"), {"hit": 0.6, "stand": 0.9}), ("unsure", ("stand", "hit")))

    def test_ties_keep_e5_order(self):
        self.assertEqual(apply_policy("unsure", ("stand", "hit", "double"), {"hit": 0.8, "stand": 0.8, "double": 0.8}),
                         ("unsure", ("stand", "hit", "double")))

    def test_tau_boundary_is_inclusive(self):
        self.assertEqual(apply_policy("unsure", ("hit",), {"hit": 0.5}), ("unsure", ("hit",)))
        self.assertEqual(apply_policy("unsure", ("hit",), {"hit": 0.4999}), ("silent", ()))

    def test_custom_tau(self):
        self.assertEqual(apply_policy("unsure", ("hit", "stand"), {"hit": 0.7, "stand": 0.9}, tau=0.8), ("unsure", ("stand",)))
        self.assertEqual(apply_policy("unsure", ("hit",), {"hit": 0.7}, tau=0.8), ("silent", ()))

    def test_missing_prob_counts_as_zero(self):
        self.assertEqual(apply_policy("unsure", ("hit", "stand"), {"hit": 0.9}), ("unsure", ("hit",)))

    def test_never_adds_a_move(self):
        self.assertEqual(apply_policy("unsure", ("hit",), {"hit": 0.9, "stand": 0.99}), ("unsure", ("hit",)))

    def test_returns_tuple_and_does_not_mutate(self):
        moves, probs = ["hit", "stand"], {"hit": 0.6, "stand": 0.9}
        band, out = apply_policy("sure", moves, probs)
        self.assertIsInstance(out, tuple)
        self.assertEqual(moves, ["hit", "stand"])
        self.assertEqual(probs, {"hit": 0.6, "stand": 0.9})

    def test_all_dropped_from_unsure_is_silent(self):
        self.assertEqual(apply_policy("unsure", ("hit", "stand"), {"hit": 0.1, "stand": 0.2}), ("silent", ()))


class ClientRequestTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_path_and_exact_request(self):
        rec = Recorder(ok(hit=0.9, stand=0.2))
        client = client_for(rec)
        probs = await client.ask(TEXT, ["hit", "stand"])
        self.assertEqual(probs, {"hit": 0.9, "stand": 0.2})
        self.assertEqual(len(rec.requests), 1)
        req = rec.requests[0]
        self.assertEqual(str(req.url), URL)
        self.assertEqual(req.method, "POST")
        self.assertEqual(req.headers["authorization"], f"Bearer {KEY}")
        self.assertEqual(json.loads(req.content), build_request(TEXT, ["hit", "stand"]))
        self.assertNotIn(KEY, req.content.decode())
        await client.close()

    async def test_key_only_on_the_veto_url(self):
        rec = Recorder(ok(hit=0.9))
        client = client_for(rec)
        await client.ask("hit", ["hit"])
        for req in rec.requests:
            self.assertEqual((req.url.scheme, req.url.host, req.url.path), ("https", "openrouter.ai", "/api/alpha/decisions"))
        await client.close()

    async def test_redirect_is_not_followed_and_key_not_sent_elsewhere(self):
        rec = Recorder(httpx.Response(302, headers={"Location": "https://evil.example/steal"}))
        client = client_for(rec)
        with self.assertRaises(VetoError) as cm:
            await client.ask(TEXT, ["hit"])
        self.assertEqual([r.url.host for r in rec.requests], ["openrouter.ai"])
        self.assertNotIn(KEY, str(cm.exception))
        await client.close()

    async def test_input_validation_sends_nothing(self):
        rec = Recorder(ok(hit=0.9))
        client = client_for(rec)
        for text, legal in (("", ["hit"]), ("x" * 201, ["hit"]), ("hit", [])):
            with self.assertRaises(VetoError):
                await client.ask(text, legal)
        self.assertEqual(rec.requests, [])
        await client.close()

    async def test_failures_raise_veto_error_without_leaking(self):
        body_secret = "RESPONSE-BODY-MARKER-9137"
        cases = {
            "read timeout": httpx.ReadTimeout("slow"),
            "connect error": httpx.ConnectError("refused"),
            "401": httpx.Response(401, text=body_secret),
            "429": httpx.Response(429, text=body_secret),
            "500": httpx.Response(500, text=body_secret),
            "302": httpx.Response(302, headers={"Location": "https://openrouter.ai/elsewhere"}, text=body_secret),
            "invalid json": httpx.Response(200, text="{not json " + body_secret),
            "json not dict": httpx.Response(200, json=[body_secret]),
            "missing answers": httpx.Response(200, json={"note": body_secret}),
            "bad value": httpx.Response(200, json=answers(hit=7)),
            "refusal": httpx.Response(200, json={"answers": {"move_hit": {"type": "refusal", "refusal": body_secret}}}),
            "oversized": httpx.Response(200, text="x" * 500 + body_secret),
        }
        for name, item in cases.items():
            with self.subTest(name):
                rec = Recorder(item)
                client = client_for(rec, max_bytes=300, fail_limit=99)
                with self.assertRaises(VetoError) as cm:
                    await client.ask(TEXT, ["hit"])
                for shown in (str(cm.exception), repr(cm.exception), repr(cm.exception.args)):
                    self.assertNotIn(KEY, shown)
                    self.assertNotIn(TEXT, shown)
                    self.assertNotIn(body_secret, shown)
                await client.close()

    async def test_oversized_body_exactly_at_limit_is_accepted(self):
        payload = json.dumps(answers(hit=0.9)).encode()
        rec = Recorder(httpx.Response(200, content=payload, headers={"content-type": "application/json"}))
        client = client_for(rec, max_bytes=len(payload))
        self.assertEqual(await client.ask("hit", ["hit"]), {"hit": 0.9})
        await client.close()

    async def test_nothing_logged_contains_key_or_message(self):
        records = []

        class Grab(logging.Handler):
            def emit(self, record):
                records.append(record)

        root = logging.getLogger()
        handler, old = Grab(level=logging.DEBUG), root.level
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        try:
            for item in (ok(hit=0.9), httpx.Response(500, text="boom"), httpx.ReadTimeout("slow"), httpx.Response(200, text="nope")):
                client = client_for(Recorder(item), fail_limit=99)
                try:
                    await client.ask(TEXT, ["hit"])
                except VetoError:
                    pass
                await client.close()
        finally:
            root.removeHandler(handler)
            root.setLevel(old)
        for record in records:
            text = " ".join([record.getMessage(), repr(record.args), str(record.exc_info[1]) if record.exc_info else ""])
            self.assertNotIn(KEY, text)
            self.assertNotIn("zebra-sandwich", text)


class BreakerTests(unittest.IsolatedAsyncioTestCase):
    async def test_opens_after_fail_limit_without_sending(self):
        clock = Clock()
        rec = Recorder(httpx.Response(500))
        client = client_for(rec, clock, fail_limit=3, cooldown=60.0)
        for _ in range(3):
            self.assertFalse(client.breaker_open)
            with self.assertRaises(VetoError):
                await client.ask("hit", ["hit"])
        self.assertTrue(client.breaker_open)
        sent = len(rec.requests)
        with self.assertRaises(VetoError) as cm:
            await client.ask("hit", ["hit"])
        self.assertIn("breaker open", str(cm.exception))
        self.assertEqual(len(rec.requests), sent)
        clock.t += 59.0
        with self.assertRaises(VetoError):
            await client.ask("hit", ["hit"])
        self.assertEqual(len(rec.requests), sent)
        await client.close()

    async def test_default_fail_limit_is_five(self):
        rec = Recorder(httpx.Response(500))
        client = client_for(rec)
        for _ in range(4):
            with self.assertRaises(VetoError):
                await client.ask("hit", ["hit"])
        self.assertFalse(client.breaker_open)
        with self.assertRaises(VetoError):
            await client.ask("hit", ["hit"])
        self.assertTrue(client.breaker_open)
        await client.close()

    async def test_half_open_success_closes_and_resets(self):
        clock = Clock()
        rec = Recorder(httpx.Response(500), httpx.Response(500), ok(hit=0.9), httpx.Response(500))
        client = client_for(rec, clock, fail_limit=2, cooldown=60.0)
        for _ in range(2):
            with self.assertRaises(VetoError):
                await client.ask("hit", ["hit"])
        self.assertTrue(client.breaker_open)
        clock.t += 60.0
        self.assertEqual(await client.ask("hit", ["hit"]), {"hit": 0.9})
        self.assertFalse(client.breaker_open)
        # the failure count restarted: one failure does not reopen it
        with self.assertRaises(VetoError):
            await client.ask("hit", ["hit"])
        self.assertFalse(client.breaker_open)
        await client.close()

    async def test_half_open_failure_reopens(self):
        clock = Clock()
        rec = Recorder(httpx.Response(500))
        client = client_for(rec, clock, fail_limit=2, cooldown=60.0)
        for _ in range(2):
            with self.assertRaises(VetoError):
                await client.ask("hit", ["hit"])
        clock.t += 61.0
        with self.assertRaises(VetoError):
            await client.ask("hit", ["hit"])
        self.assertEqual(len(rec.requests), 3)
        self.assertTrue(client.breaker_open)
        with self.assertRaises(VetoError) as cm:
            await client.ask("hit", ["hit"])
        self.assertIn("breaker open", str(cm.exception))
        self.assertEqual(len(rec.requests), 3)
        clock.t += 61.0
        with self.assertRaises(VetoError):
            await client.ask("hit", ["hit"])
        self.assertEqual(len(rec.requests), 4)
        await client.close()

    async def test_success_resets_consecutive_count(self):
        rec = Recorder(httpx.Response(500), httpx.Response(500), ok(hit=0.9), httpx.Response(500), httpx.Response(500), ok(hit=0.9))
        client = client_for(rec, fail_limit=3)
        for item in range(6):
            try:
                await client.ask("hit", ["hit"])
            except VetoError:
                pass
        self.assertFalse(client.breaker_open)
        self.assertEqual(len(rec.requests), 6)
        await client.close()


class CloseTests(unittest.IsolatedAsyncioTestCase):
    async def test_ask_after_close_raises_and_close_is_idempotent(self):
        rec = Recorder(ok(hit=0.9))
        client = client_for(rec)
        await client.close()
        with self.assertRaises(VetoError):
            await client.ask("hit", ["hit"])
        self.assertEqual(rec.requests, [])
        await client.close()


class FromEnvTests(unittest.IsolatedAsyncioTestCase):
    async def test_enabled_returns_client_that_works(self):
        rec = Recorder(ok(hit=0.9))
        client = from_env({"LLMCORD_SPOKEN_VETO": "openrouter-d1", "OPENROUTER_API_KEY": f"  {KEY} "}, transport=rec.transport())
        self.assertIsInstance(client, VetoClient)
        self.assertEqual(await client.ask("hit", ["hit"]), {"hit": 0.9})
        self.assertEqual(rec.requests[0].headers["authorization"], f"Bearer {KEY}")
        await client.close()

    def test_disabled_variants_return_none(self):
        cases = [
            {},
            {"OPENROUTER_API_KEY": KEY},
            {"LLMCORD_SPOKEN_VETO": "openrouter-d1"},
            {"LLMCORD_SPOKEN_VETO": "openrouter-d1", "OPENROUTER_API_KEY": ""},
            {"LLMCORD_SPOKEN_VETO": "openrouter-d1", "OPENROUTER_API_KEY": "   "},
            {"LLMCORD_SPOKEN_VETO": "off", "OPENROUTER_API_KEY": KEY},
            {"LLMCORD_SPOKEN_VETO": "1", "OPENROUTER_API_KEY": KEY},
            {"LLMCORD_SPOKEN_VETO": "OPENROUTER-D1", "OPENROUTER_API_KEY": KEY},
            {"LLMCORD_SPOKEN_VETO": "", "OPENROUTER_API_KEY": KEY},
        ]
        for env in cases:
            with self.subTest(env=sorted(env)):
                self.assertIsNone(from_env(env))

    def test_does_not_read_os_environ_when_env_given(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"LLMCORD_SPOKEN_VETO": "openrouter-d1", "OPENROUTER_API_KEY": KEY}):
            self.assertIsNone(from_env({}))


class FromEnvKeyShapeTests(unittest.IsolatedAsyncioTestCase):
    """FEAT-49a: a key that cannot be a valid header value turns the veto off instead of raising."""

    def test_unusable_keys_return_none_without_raising(self):
        cases = {
            "inner newline": "fake-one\nfake-two",
            "inner carriage return": "fake-one\rfake-two",
            "inner space": "fake one",
            "inner tab": "fake\tone",
            "korean": "fake-키-one",
            "nul": "fake\x00one",
        }
        for name, bad in cases.items():
            with self.subTest(case=name):
                env = {"LLMCORD_SPOKEN_VETO": "openrouter-d1", "OPENROUTER_API_KEY": f" {bad}\n"}
                self.assertIsNone(from_env(env))

    async def test_symbols_in_printable_ascii_key_still_work(self):
        fake_key = "fake-key_0.9~A+b/c="
        rec = Recorder(ok(hit=0.9))
        client = from_env({"LLMCORD_SPOKEN_VETO": "openrouter-d1", "OPENROUTER_API_KEY": f"\t{fake_key}\n"}, transport=rec.transport())
        self.assertIsInstance(client, VetoClient)
        await client.ask("hit", ["hit"])
        self.assertEqual(rec.requests[0].headers["authorization"], f"Bearer {fake_key}")
        await client.close()


if __name__ == "__main__":
    unittest.main()
