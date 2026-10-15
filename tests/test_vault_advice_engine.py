"""Vault NPC advice, pure engine (FEAT-65 phase A, decision D43). Written before the implementation; fails until it exists.

API CONTRACT (new module ``llmcord_core/vault/advice.py``, re-exported from ``llmcord_core.vault`` like the solver)

  ADVICE_TEMPLATE_KEY = "vault_advice_template"   text key of the deterministic fallback line (translated at the Discord edge)

  AdviceFacts   frozen dataclass: guess (str, valid code), remaining (int, candidates left), weight (float 0..1),
                in_candidates (bool, the guess can still be the secret)

  build_advice_facts(constraints, protocol, length, *, risk=50, aggression=50, unpredictability=0, seed="") -> AdviceFacts
      PUBLIC constraints only: there is no secret parameter. remaining = len(filter_candidates(constraints, length)).
      weight = weight_from_style(risk, aggression); guess = suggest_guess(candidates, protocol, length, weight,
      unpredictability, seed). Same arguments, same facts; a different seed may change the guess only when unpredictability > 0.

  check_wording(text, facts) -> bool
      True only for a str that contains facts.guess and whose every maximal digit run is the guess, or str(facts.remaining) when
      that has fewer digits than the code (so "12 left" is fine, a 4-digit remaining count is not an allowed number).
      Any other 4-digit code, a digit inside a longer run, or a lone digit claim ("the first digit is 7") -> False.

  advice_template(facts) -> (ADVICE_TEMPLATE_KEY, {"guess": facts.guess, "remaining": facts.remaining})
      The fallback is data, not prose: no translated text in the engine.
"""
import inspect
import unittest

from llmcord_core.vault import (DigitAt, GuessResult, NotSecret, OpenDigit, all_secrets, feedback, filter_candidates, suggest_guess,
                                valid_code, weight_from_style)
from llmcord_core.vault.advice import ADVICE_TEMPLATE_KEY, AdviceFacts, advice_template, build_advice_facts, check_wording

SECRET = "4829"


def history(*guesses, protocol="standard"):
    out = []
    for g in guesses:
        out += [GuessResult(g, protocol, feedback(protocol, SECRET, g)), NotSecret(g)]
    return out


class FactsTests(unittest.TestCase):
    def test_facts_follow_the_solver_with_the_weight_from_style(self):
        cons = history("1234", "5678")
        facts = build_advice_facts(cons, "standard", 4, risk=80, aggression=60)
        cands = filter_candidates(cons, 4)
        self.assertIsInstance(facts, AdviceFacts)
        self.assertEqual(facts.remaining, len(cands))
        self.assertEqual(facts.weight, weight_from_style(80, 60))
        self.assertEqual(facts.guess, suggest_guess(cands, "standard", 4, weight_from_style(80, 60)))
        self.assertTrue(valid_code(facts.guess, 4))
        self.assertEqual(facts.in_candidates, facts.guess in cands)

    def test_traits_change_the_recommendation_between_cautious_and_bold(self):
        cons = history("1234", protocol="total_match")  # Standard always recommends a candidate; Total Match splits the two
        careful = build_advice_facts(cons, "total_match", 4, risk=0, aggression=0)
        bold = build_advice_facts(cons, "total_match", 4, risk=100, aggression=100)
        self.assertEqual((careful.weight, bold.weight), (0.0, 1.0))
        self.assertTrue(bold.in_candidates)
        self.assertNotEqual(careful.guess, bold.guess)

    def test_deterministic_by_seed_and_seed_matters_only_with_unpredictability(self):
        cons = history("1234")
        a = build_advice_facts(cons, "standard", 4, unpredictability=100, seed="s")
        self.assertEqual(a, build_advice_facts(cons, "standard", 4, unpredictability=100, seed="s"))
        guesses = {build_advice_facts(cons, "standard", 4, unpredictability=100, seed=f"s{i}").guess for i in range(40)}
        self.assertGreater(len(guesses), 1)
        self.assertLessEqual(len(guesses), 3)  # only the top three are ever picked
        steady = {build_advice_facts(cons, "standard", 4, unpredictability=0, seed=f"s{i}").guess for i in range(10)}
        self.assertEqual(len(steady), 1)

    def test_single_candidate_is_recommended_as_is(self):
        cons = [DigitAt(i + 1, d) for i, d in enumerate(SECRET)]
        facts = build_advice_facts(cons, "standard", 4, risk=0, aggression=0)
        self.assertEqual((facts.guess, facts.remaining, facts.in_candidates), (SECRET, 1, True))

    def test_no_secret_parameter_and_facts_depend_on_public_constraints_only(self):
        params = inspect.signature(build_advice_facts).parameters
        self.assertFalse([p for p in params if "secret" in p.lower()])
        # Two worlds with different secrets but identical public constraints give identical facts.
        cons = [OpenDigit("7")]
        self.assertEqual(build_advice_facts(cons, "open", 4, risk=30), build_advice_facts(list(cons), "open", 4, risk=30))

    def test_template_is_data_with_the_guess_and_remaining(self):
        facts = build_advice_facts(history("1234"), "standard", 4)
        key, data = advice_template(facts)
        self.assertEqual(key, ADVICE_TEMPLATE_KEY)
        self.assertEqual(key, "vault_advice_template")
        self.assertEqual(data, {"guess": facts.guess, "remaining": facts.remaining})


class WordingTests(unittest.TestCase):
    def setUp(self):
        self.facts = AdviceFacts(guess="1234", remaining=12, weight=0.5, in_candidates=True)

    def test_accepts_text_with_the_guess_and_the_remaining_count(self):
        for text in ("Try 1234.", "I'd go with 1234 - about 12 left.", "1234!", "12 are left, so 1234 it is"):
            with self.subTest(text=text):
                self.assertTrue(check_wording(text, self.facts))

    def test_rejects_text_without_the_guess(self):
        for text in ("", "Trust your gut.", "Try 12 maybe", None, 1234):
            with self.subTest(text=text):
                self.assertFalse(check_wording(text, self.facts))

    def test_rejects_any_other_four_digit_code(self):
        for text in ("Try 1234, not 4829.", "1234 or 4321", "0000 1234"):
            with self.subTest(text=text):
                self.assertFalse(check_wording(text, self.facts))

    def test_rejects_digit_claims_and_longer_runs(self):
        for text in ("Try 1234, the first digit is 7.", "Try 1234 and 5 is in it", "Try 12345", "Try 91234", "1234 then 99 left"):
            with self.subTest(text=text):
                self.assertFalse(check_wording(text, self.facts))

    def test_a_four_digit_remaining_count_is_not_an_allowed_number(self):
        big = AdviceFacts(guess="1234", remaining=5040, weight=0.5, in_candidates=True)
        self.assertTrue(check_wording("Try 1234.", big))
        self.assertFalse(check_wording("Try 1234, 5040 left.", big))

    def test_wording_check_scans_the_whole_space_consistently(self):
        """Every other code in the space is rejected next to the guess."""
        others = [c for c in all_secrets(4) if c != "1234"][:200]
        self.assertFalse([c for c in others if check_wording(f"Try 1234 not {c}", self.facts)])


if __name__ == "__main__":
    unittest.main()
