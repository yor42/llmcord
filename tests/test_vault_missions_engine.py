"""Vault support missions, pure engine (FEAT-67 phase A, decision D43). Written before the implementation; they fail until it exists.

CONTRACT: new module ``llmcord_core/vault/missions.py`` (pure: no Discord, database, model, clock, i18n or network; randomness only from a
passed ``random.Random`` or a seed string; refusals are ``VaultError`` with a lowercase snake code). It does NOT change ``rules.LENGTHS``,
``DEFAULT_LENGTH``, ``DEFAULT_TURNS`` or ``all_secrets``: the 3-digit game is mission-only, with its own small helpers.

  Constants     MISSION_TYPES = ("forensics", "bulls_cows"); BULLS_COWS_LENGTH = 3; BULLS_COWS_TURNS = 6 (proposal 11.2);
                FORENSICS_LENGTH = 4; FORENSICS_LINES = 4 (records shown, one of them hidden); FORENSICS_ATTEMPTS = 2 (answers allowed).
  Mission B     bulls_cows_secrets() -> all 720 secrets of 3 unique digits, lexicographic ("012" is valid: leading zeros allowed)
                bulls_cows_secret(rng=None) -> one of them (secrets.SystemRandom by default)
                bulls_cows_feedback(secret, guess) -> (strikes, balls), the Standard protocol (same numbers as rules.evaluate)
                check_bulls_cows_guess(guess) -> guess, or VaultError: guess_length (params length=3), guess_not_digits, guess_repeated_digit
  Mission A     ForensicsPuzzle (frozen dataclass): records = tuple of (guess, result) with result a (strikes, balls) tuple, except exactly
                ONE record whose result is None; hidden = index of that record; answer = the true (strikes, balls) of the hidden record.
                generate_forensics(seed) -> ForensicsPuzzle, deterministic from the seed string (own random.Random(seed)). The visible
                records are real Standard feedback of one secret; the answer is PROVEN unique: over every 4-digit secret consistent with the
                visible records, the hidden guess has exactly one result, and it equals ``answer``. The generator retries internally until so.
                forensics_answers(records) -> frozenset of the possible results of the single hidden (None) record over the secrets consistent
                with the visible ones: the solver. One element means the puzzle is solvable; many means ambiguous; none means contradictory.
                valid_forensics_answer(strikes, balls) -> bool: plain ints (bool is not an int), 0 <= strikes, 0 <= balls, strikes + balls <= 4,
                and not (strikes == 3 and balls == 1) (impossible with unique digits).
"""
import unittest
from random import Random

from llmcord_core.vault import (DEFAULT_LENGTH, DEFAULT_TURNS, LENGTHS, GuessResult, VaultError, all_secrets, evaluate, feedback, filter_candidates,
                                random_secret)
from llmcord_core.vault.missions import (BULLS_COWS_LENGTH, BULLS_COWS_TURNS, FORENSICS_ATTEMPTS, FORENSICS_LENGTH, FORENSICS_LINES, MISSION_TYPES,
                                         bulls_cows_feedback, bulls_cows_secret, bulls_cows_secrets, check_bulls_cows_guess, forensics_answers,
                                         generate_forensics, valid_forensics_answer)

SEEDS = [f"seed-{i}" for i in range(40)]


def code_of(call, *args):
    try:
        call(*args)
    except VaultError as error:
        return error.code
    return None


class ConstantsTests(unittest.TestCase):
    def test_contract_constants(self):
        self.assertEqual(MISSION_TYPES, ("forensics", "bulls_cows"))
        self.assertEqual((BULLS_COWS_LENGTH, BULLS_COWS_TURNS, FORENSICS_LENGTH, FORENSICS_LINES), (3, 6, 4, 4))
        self.assertIsInstance(FORENSICS_ATTEMPTS, int)
        self.assertGreaterEqual(FORENSICS_ATTEMPTS, 1)

    def test_the_main_game_is_unchanged(self):
        """Regression (FEAT-67): length 3 is mission-only; the main engine still refuses it and keeps its lengths and budgets."""
        self.assertEqual(LENGTHS, (4, 5, 6))
        self.assertEqual(DEFAULT_LENGTH, 4)
        self.assertEqual(DEFAULT_TURNS, {"standard": 9, "total_match": 9, "position_only": 9, "open": 8})
        self.assertEqual(code_of(all_secrets, 3), "bad_length")
        self.assertEqual(code_of(random_secret, 3), "bad_length")
        self.assertEqual(len(all_secrets(4)), 5040)


class BullsCowsTests(unittest.TestCase):
    def test_secret_space_is_720_unique_digit_triples_with_leading_zeros(self):
        space = bulls_cows_secrets()
        self.assertEqual(len(space), 720)
        self.assertEqual(list(space), sorted(space))
        self.assertTrue(all(len(s) == 3 and len(set(s)) == 3 and s.isdigit() for s in space))
        self.assertEqual(space[0], "012")
        self.assertIn("098", space)

    def test_secret_is_seeded_valid_and_uses_the_whole_space(self):
        self.assertEqual(bulls_cows_secret(Random("a")), bulls_cows_secret(Random("a")))
        picks = {bulls_cows_secret(Random(i)) for i in range(60)}
        self.assertGreater(len(picks), 30)
        self.assertTrue(picks <= set(bulls_cows_secrets()))
        self.assertIn(bulls_cows_secret(), bulls_cows_secrets())

    def test_feedback_examples(self):
        for secret, guess, expected in (("123", "123", (3, 0)), ("123", "132", (1, 2)), ("123", "312", (0, 3)), ("123", "456", (0, 0)),
                                        ("123", "145", (1, 0)), ("123", "451", (0, 1)), ("012", "021", (1, 2)), ("987", "879", (0, 3))):
            with self.subTest(secret=secret, guess=guess):
                self.assertEqual(bulls_cows_feedback(secret, guess), expected)

    def test_feedback_matches_the_standard_rules_on_every_pair(self):
        space = bulls_cows_secrets()
        for secret in space[::37]:
            for guess in space[::11]:
                self.assertEqual(bulls_cows_feedback(secret, guess), evaluate(secret, guess))
                self.assertEqual(bulls_cows_feedback(secret, guess), feedback("standard", secret, guess))

    def test_feedback_is_symmetric_and_a_win_is_three_strikes(self):
        space = bulls_cows_secrets()
        self.assertTrue(all(bulls_cows_feedback(s, s) == (3, 0) for s in space[::13]))
        self.assertEqual(bulls_cows_feedback("123", "145"), bulls_cows_feedback("145", "123"))

    def test_guess_validation_like_the_main_game(self):
        self.assertEqual(check_bulls_cows_guess("012"), "012")
        self.assertEqual(code_of(check_bulls_cows_guess, "12"), "guess_length")
        self.assertEqual(code_of(check_bulls_cows_guess, "1234"), "guess_length")
        self.assertEqual(code_of(check_bulls_cows_guess, ""), "guess_length")
        self.assertEqual(code_of(check_bulls_cows_guess, "12a"), "guess_not_digits")
        self.assertEqual(code_of(check_bulls_cows_guess, "1 2"), "guess_not_digits")
        self.assertEqual(code_of(check_bulls_cows_guess, "112"), "guess_repeated_digit")
        try:
            check_bulls_cows_guess("12")
        except VaultError as error:
            self.assertEqual(error.params, {"length": 3})


def possible_by_brute_force(puzzle):
    """The solver written independently of the module: the vault.constraints filter over the 4-digit space."""
    visible = [GuessResult(g, "standard", r) for g, r in puzzle.records if r is not None]
    hidden_guess = puzzle.records[puzzle.hidden][0]
    candidates = filter_candidates(visible, FORENSICS_LENGTH)
    return candidates, {feedback("standard", c, hidden_guess) for c in candidates}


class ForensicsTests(unittest.TestCase):
    def test_shape(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                p = generate_forensics(seed)
                self.assertEqual(len(p.records), FORENSICS_LINES)
                self.assertEqual([r for _, r in p.records].count(None), 1)
                self.assertEqual(p.records[p.hidden][1], None)
                self.assertTrue(0 <= p.hidden < FORENSICS_LINES)
                guesses = [g for g, _ in p.records]
                self.assertEqual(len(set(guesses)), len(guesses))
                self.assertTrue(all(len(g) == 4 and len(set(g)) == 4 and g.isdigit() for g in guesses))
                self.assertTrue(valid_forensics_answer(*p.answer))

    def test_the_answer_is_proven_unique_by_the_solver(self):
        for seed in SEEDS:
            with self.subTest(seed=seed):
                p = generate_forensics(seed)
                candidates, results = possible_by_brute_force(p)
                self.assertTrue(candidates, "visible records contradict each other")
                self.assertEqual(results, {p.answer})
                self.assertEqual(forensics_answers(p.records), frozenset({p.answer}))

    def test_visible_records_are_real_feedback_of_some_secret(self):
        """At least one 4-digit secret reproduces every visible line AND the hidden one with the stored answer (the puzzle is not made up)."""
        for seed in SEEDS[:15]:
            p = generate_forensics(seed)
            full = [GuessResult(g, "standard", p.answer if r is None else r) for g, r in p.records]
            self.assertTrue(filter_candidates(full, FORENSICS_LENGTH), seed)

    def test_deterministic_by_seed_and_varied_across_seeds(self):
        self.assertEqual(generate_forensics("same"), generate_forensics("same"))
        self.assertEqual(generate_forensics("same").records, generate_forensics("same").records)
        distinct = {generate_forensics(seed).records for seed in SEEDS}
        self.assertGreaterEqual(len(distinct), 30)

    def test_the_puzzle_is_not_trivially_given_away(self):
        """The hidden guess is not also a visible one, and the answer is not simply copied from an identical visible record."""
        for seed in SEEDS:
            p = generate_forensics(seed)
            hidden_guess = p.records[p.hidden][0]
            self.assertEqual([g for g, _ in p.records].count(hidden_guess), 1)

    def test_solver_flags_an_ambiguous_record_set(self):
        records = (("0123", (0, 0)), ("4567", None))
        answers = forensics_answers(records)
        self.assertGreater(len(answers), 1)
        _, truth = possible_by_brute_force(type("P", (), {"records": records, "hidden": 1})())
        self.assertEqual(answers, frozenset(truth))

    def test_solver_flags_a_contradictory_record_set(self):
        self.assertEqual(forensics_answers((("0123", (4, 0)), ("4567", (4, 0)), ("8901", None))), frozenset())

    def test_solver_finds_a_forced_answer(self):
        """If the visible line says the secret is 0123, the hidden guess 0132 must score (2, 2)."""
        self.assertEqual(forensics_answers((("0123", (4, 0)), ("0132", None))), frozenset({(2, 2)}))

    def test_answer_validation(self):
        for good in ((0, 0), (4, 0), (0, 4), (1, 2), (2, 2), (3, 0), (0, 1)):
            self.assertTrue(valid_forensics_answer(*good), good)
        for bad in ((3, 1), (5, 0), (2, 3), (-1, 0), (0, -1), (True, 0), (0, False), (1.0, 0), ("1", 0), (None, 0), (4, 1)):
            self.assertFalse(valid_forensics_answer(*bad), bad)


if __name__ == "__main__":
    unittest.main()
