"""Vault pure engine (llmcord_core/vault/), FEAT-62, decision D43.

These tests define the engine API; they fail today because the package does not exist. The engine is pure (no
Discord, database, model, clock, i18n or network): randomness comes only from a passed ``random.Random`` or a seed
string. It raises ``VaultError(ValueError)`` with a lowercase snake ``code``, never user-facing text.

API seam pinned here (``llmcord_core.vault`` re-exports all of it)
  rules       PROTOCOLS, DEFAULT_LENGTH, LENGTHS, DEFAULT_TURNS, all_secrets, valid_code, random_secret, evaluate,
              check_guess, feedback, is_win, open_digit, VaultError
  constraints GuessResult, DigitPresent, DigitAt, PositionHalf, PairCount, CountEven, CountLow, PositionParity,
              OpenDigit (frozen, ``holds(secret)``), filter_candidates
  clues       make_clues
  tools       TOOLS, ToolSpec, ToolResult, evaluate_tool, tool_fits, is_redundant, cross_validation,
              cross_validation_common
  solver      Suggestion, rank_guesses, weight_from_style, suggest_guess

Parameter shapes the brief leaves open, pinned here (plain dicts, positions 1-based ints, digits one-character strings)
  vault_residual_scanner    {"guess": "1234", "digit": "2"}
  vault_spectrum_analyzer   {"position": 2}
  vault_comparison_circuit  {"a": "2", "b": "8"}
  vault_position_verifier   {"guess": "5124"}
  vault_master_decoder      {"position": 3}
``history`` is a list of past ``GuessResult`` plus previously disclosed ``DigitAt``. ``ToolResult.data`` carries
"position" and "digit" for the position verifier and the master decoder.
"""
import dataclasses
import inspect
import itertools
import random
import unittest

from llmcord_core.vault import (
    DEFAULT_LENGTH,
    DEFAULT_TURNS,
    LENGTHS,
    PROTOCOLS,
    TOOLS,
    CountEven,
    CountLow,
    DigitAt,
    DigitPresent,
    GuessResult,
    NotSecret,
    OpenDigit,
    PairCount,
    PositionHalf,
    PositionParity,
    Suggestion,
    ToolResult,
    ToolSpec,
    VaultError,
    all_secrets,
    check_guess,
    cross_validation,
    cross_validation_common,
    evaluate,
    evaluate_tool,
    feedback,
    filter_candidates,
    is_redundant,
    is_win,
    make_clues,
    open_digit,
    played,
    random_secret,
    rank_guesses,
    suggest_guess,
    tool_fits,
    valid_code,
    weight_from_style,
)

SPEC = "5824"


def gr(secret, guess, protocol="standard"):
    """A GuessResult for ``guess`` against ``secret`` as the engine's own feedback reports it."""
    return GuessResult(guess, protocol, feedback(protocol, secret, guess))


def code_of(test, func, *args, **kwargs):
    """The VaultError code a call raises (fails the test when it raises nothing else)."""
    with test.assertRaises(VaultError) as ctx:
        func(*args, **kwargs)
    return ctx.exception.code


class SecretSpaceTests(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(PROTOCOLS, ("standard", "total_match", "position_only", "open"))
        self.assertEqual(DEFAULT_LENGTH, 4)
        self.assertEqual(LENGTHS, (4, 5, 6))
        self.assertEqual(DEFAULT_TURNS, {"standard": 9, "total_match": 9, "position_only": 9, "open": 8})

    def test_space_sizes_per_length(self):
        self.assertEqual(len(all_secrets()), 5040)
        self.assertEqual(len(all_secrets(4)), 5040)
        self.assertEqual(len(all_secrets(5)), 30240)
        self.assertEqual(len(all_secrets(6)), 151200)

    def test_every_secret_has_unique_digits_of_the_right_length(self):
        for length in LENGTHS:
            space = all_secrets(length)
            with self.subTest(length=length):
                self.assertEqual(len(set(space)), len(space))
                self.assertTrue(all(len(s) == length and len(set(s)) == length and s.isdigit() for s in space))

    def test_space_is_lexicographic_and_keeps_leading_zeros(self):
        space = all_secrets(4)
        self.assertEqual(list(space), sorted(space))
        self.assertEqual(space[0], "0123")
        self.assertEqual(space[-1], "9876")
        self.assertIn("0127", space)
        self.assertEqual(all_secrets(6)[0], "012345")

    def test_space_is_cached_per_length(self):
        for length in LENGTHS:
            self.assertIs(all_secrets(length), all_secrets(length))

    def test_valid_code(self):
        self.assertTrue(valid_code("0127"))
        self.assertTrue(valid_code("5832"))
        self.assertFalse(valid_code("1123"))
        self.assertFalse(valid_code("9999"))
        self.assertFalse(valid_code("123"))
        self.assertFalse(valid_code("12345"))
        self.assertFalse(valid_code("12a4"))
        self.assertFalse(valid_code(""))

    def test_valid_code_follows_the_length(self):
        self.assertTrue(valid_code("12345", 5))
        self.assertFalse(valid_code("1234", 5))
        self.assertTrue(valid_code("012345", 6))
        self.assertFalse(valid_code("112345", 6))

    def test_random_secret_is_valid_and_reproducible_for_a_seeded_rng(self):
        for length in LENGTHS:
            with self.subTest(length=length):
                one = random_secret(length, random.Random(7))
                self.assertTrue(valid_code(one, length))
                self.assertEqual(one, random_secret(length, random.Random(7)))

    def test_random_secret_defaults_to_four_digits_without_an_rng(self):
        self.assertTrue(valid_code(random_secret()))
        self.assertTrue(valid_code(random_secret(5), 5))

    def test_random_secret_can_start_with_zero(self):
        rng = random.Random(1)
        self.assertTrue(any(random_secret(4, rng).startswith("0") for _ in range(200)))


class EvaluateTests(unittest.TestCase):
    def test_spec_example(self):
        # Secret 5824, guess 1234: the 4 is a strike, the 2 a ball.
        self.assertEqual(evaluate(SPEC, "1234"), (1, 1))

    def test_all_strikes_and_no_overlap_and_all_balls(self):
        self.assertEqual(evaluate(SPEC, SPEC), (4, 0))
        self.assertEqual(evaluate(SPEC, "1367"), (0, 0))
        self.assertEqual(evaluate("1234", "4321"), (0, 4))

    def test_leading_zero_secret(self):
        self.assertEqual(evaluate("0127", "7210"), (0, 4))
        self.assertEqual(evaluate("0127", "0172"), (2, 2))

    def test_longer_codes(self):
        self.assertEqual(evaluate("12345", "12354"), (3, 2))
        self.assertEqual(evaluate("012345", "543210"), (0, 6))
        self.assertEqual(evaluate("012345", "012678"), (3, 0))

    def test_strikes_plus_balls_is_the_shared_digit_count(self):
        rng = random.Random(3)
        space = all_secrets(4)
        for _ in range(100):
            a, b = rng.choice(space), rng.choice(space)
            s, bl = evaluate(a, b)
            self.assertEqual(s + bl, len(set(a) & set(b)))

    def test_is_win(self):
        self.assertTrue(is_win(SPEC, SPEC))
        self.assertFalse(is_win(SPEC, "5842"))


class CheckGuessTests(unittest.TestCase):
    def test_valid_guess_passes(self):
        check_guess("0123", 4)
        check_guess("01234", 5)

    def test_wrong_length(self):
        self.assertEqual(code_of(self, check_guess, "123", 4), "guess_length")
        self.assertEqual(code_of(self, check_guess, "12345", 4), "guess_length")
        self.assertEqual(code_of(self, check_guess, "1234", 5), "guess_length")
        self.assertEqual(code_of(self, check_guess, "", 4), "guess_length")

    def test_not_digits(self):
        self.assertEqual(code_of(self, check_guess, "12a4", 4), "guess_not_digits")
        self.assertEqual(code_of(self, check_guess, "12 4", 4), "guess_not_digits")

    def test_repeated_digit(self):
        self.assertEqual(code_of(self, check_guess, "1123", 4), "guess_repeated_digit")
        self.assertEqual(code_of(self, check_guess, "123451", 6), "guess_repeated_digit")

    def test_error_is_a_value_error_with_a_snake_code_and_no_prose(self):
        with self.assertRaises(ValueError) as ctx:
            check_guess("1123", 4)
        self.assertIsInstance(ctx.exception, VaultError)
        self.assertRegex(ctx.exception.code, r"^[a-z][a-z0-9_]*$")


class FeedbackTests(unittest.TestCase):
    def test_standard(self):
        self.assertEqual(feedback("standard", SPEC, "1234"), (1, 1))

    def test_total_match_is_one_value(self):
        self.assertEqual(feedback("total_match", SPEC, "1234"), (2,))
        self.assertEqual(feedback("total_match", SPEC, "4285"), (4,))

    def test_position_only_is_strikes(self):
        self.assertEqual(feedback("position_only", SPEC, "1234"), (1,))
        self.assertEqual(feedback("position_only", SPEC, "4285"), (0,))

    def test_open_behaves_as_standard(self):
        self.assertEqual(feedback("open", SPEC, "1234"), feedback("standard", SPEC, "1234"))

    def test_feedback_is_hashable_for_grouping(self):
        for protocol in PROTOCOLS:
            hash(feedback(protocol, SPEC, "1234"))

    def test_open_digit_is_one_digit_of_the_secret(self):
        rng = random.Random(5)
        for _ in range(30):
            code = random_secret(4, rng)
            d = open_digit(code, rng)
            self.assertEqual(len(d), 1)
            self.assertIn(d, code)

    def test_open_digit_is_reproducible(self):
        self.assertEqual(open_digit(SPEC, random.Random(9)), open_digit(SPEC, random.Random(9)))


# Reference implementations the constraints are checked against (brute force over the whole 4-digit space).
REFERENCE = [
    (DigitPresent("3", True), lambda s: "3" in s),
    (DigitPresent("3", False), lambda s: "3" not in s),
    (DigitAt(1, "5"), lambda s: s[0] == "5"),
    (DigitAt(4, "0"), lambda s: s[3] == "0"),
    (PositionHalf(1, False), lambda s: int(s[0]) <= 4),
    (PositionHalf(2, True), lambda s: int(s[1]) >= 5),
    (PairCount("2", "8", 2), lambda s: "2" in s and "8" in s),
    (PairCount("2", "8", 1), lambda s: ("2" in s) != ("8" in s)),
    (PairCount("2", "8", 0), lambda s: "2" not in s and "8" not in s),
    (CountEven(2), lambda s: sum(int(c) % 2 == 0 for c in s) == 2),
    (CountEven(0), lambda s: all(int(c) % 2 == 1 for c in s)),
    (CountLow(3), lambda s: sum(int(c) <= 4 for c in s) == 3),
    (PositionParity(3, True), lambda s: int(s[2]) % 2 == 1),
    (PositionParity(3, False), lambda s: int(s[2]) % 2 == 0),
    (OpenDigit("7"), lambda s: "7" in s),
]


class ConstraintTests(unittest.TestCase):
    def test_each_constraint_matches_its_reference_over_the_space(self):
        space = all_secrets(4)
        for constraint, ref in REFERENCE:
            with self.subTest(constraint=constraint):
                self.assertEqual([s for s in space if constraint.holds(s)], [s for s in space if ref(s)])

    def test_constraints_are_frozen_and_value_equal(self):
        self.assertEqual(DigitAt(1, "5"), DigitAt(1, "5"))
        self.assertNotEqual(DigitAt(1, "5"), DigitAt(2, "5"))
        for constraint, _ in REFERENCE:
            with self.subTest(constraint=constraint):
                self.assertTrue(dataclasses.is_dataclass(constraint))
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    setattr(constraint, dataclasses.fields(constraint)[0].name, None)
                hash(constraint)

    def test_spec_secret_examples(self):
        self.assertTrue(CountEven(3).holds(SPEC))  # 8, 2, 4
        self.assertTrue(CountLow(2).holds(SPEC))  # 2, 4
        self.assertTrue(DigitPresent("8", True).holds(SPEC))
        self.assertTrue(PositionHalf(2, True).holds(SPEC))
        self.assertTrue(PositionParity(1, True).holds(SPEC))  # 5 is odd
        self.assertTrue(PairCount("2", "8", 2).holds(SPEC))
        self.assertFalse(DigitAt(1, "8").holds(SPEC))

    def test_guess_result_holds_per_protocol(self):
        for protocol in PROTOCOLS:
            with self.subTest(protocol=protocol):
                self.assertTrue(gr(SPEC, "1234", protocol).holds(SPEC))
        self.assertTrue(GuessResult("1234", "standard", (1, 1)).holds(SPEC))
        self.assertFalse(GuessResult("1234", "standard", (2, 0)).holds(SPEC))
        self.assertTrue(GuessResult("1234", "total_match", (2,)).holds(SPEC))
        self.assertFalse(GuessResult("1234", "total_match", (1,)).holds(SPEC))
        self.assertTrue(GuessResult("1234", "position_only", (1,)).holds(SPEC))

    def test_constraints_read_the_length_from_the_secret(self):
        self.assertTrue(CountEven(2).holds("12345"))  # 2, 4
        self.assertTrue(PositionHalf(5, False).holds("56784"))
        self.assertTrue(DigitAt(6, "5").holds("012345"))


class FilterCandidatesTests(unittest.TestCase):
    def game(self, secret, guesses, protocol="standard"):
        return [gr(secret, g, protocol) for g in guesses]

    def test_no_constraints_returns_the_whole_space_in_order(self):
        self.assertEqual(filter_candidates([]), list(all_secrets(4)))

    def test_result_is_lexicographic_and_keeps_the_secret(self):
        rng = random.Random(11)
        for protocol in PROTOCOLS:
            code = random_secret(4, rng)
            guesses = [random_secret(4, rng) for _ in range(4)]
            left = filter_candidates(self.game(code, guesses, protocol))
            with self.subTest(protocol=protocol):
                self.assertEqual(left, sorted(left))
                self.assertIn(code, left)

    def test_a_full_game_narrows_to_the_secret_with_standard_feedback(self):
        code = "0127"
        guesses = ["0123", "4567", "0189", "1027", "0127"]
        left = filter_candidates(self.game(code, guesses))
        self.assertEqual(left, [code])

    def test_incremental_equals_from_scratch(self):
        rng = random.Random(21)
        for protocol in PROTOCOLS:
            code = random_secret(4, rng)
            cons = self.game(code, [random_secret(4, rng) for _ in range(4)], protocol)
            scratch = filter_candidates(cons)
            base = None
            for con in cons:
                base = filter_candidates([con], base=base)
            with self.subTest(protocol=protocol):
                self.assertEqual(base, scratch)
                self.assertEqual(filter_candidates(cons[2:], base=filter_candidates(cons[:2])), scratch)

    def test_base_only_narrows(self):
        base = filter_candidates([DigitAt(1, "5")])
        self.assertEqual(len(base), 504)
        left = filter_candidates([DigitPresent("8", True)], base=base)
        self.assertTrue(set(left) <= set(base))
        self.assertTrue(all(s[0] == "5" and "8" in s for s in left))

    def test_base_is_not_mutated(self):
        base = filter_candidates([DigitAt(1, "5")])
        before = list(base)
        filter_candidates([DigitAt(2, "8")], base=base)
        self.assertEqual(base, before)

    def test_mixed_constraint_types(self):
        cons = [CountEven(3), PositionHalf(2, True), DigitPresent("5", True), PositionParity(1, True)]
        left = filter_candidates(cons)
        self.assertIn(SPEC, left)
        self.assertEqual(left, [s for s in all_secrets(4) if all(c.holds(s) for c in cons)])

    def test_five_and_six_digit_paths(self):
        for length, code in ((5, "01234"), (6, "980123")):
            guess = code[::-1]
            cons = [gr(code, guess)]
            left = filter_candidates(cons, length)
            with self.subTest(length=length):
                self.assertIn(code, left)
                self.assertEqual(left, sorted(left))
                self.assertTrue(all(len(s) == length for s in left))
                self.assertLess(len(left), len(all_secrets(length)))
                half = filter_candidates([DigitAt(1, code[0])], length)
                self.assertEqual(
                    filter_candidates(cons, length, base=half),
                    [s for s in left if s[0] == code[0]],
                )

    def test_signature_has_no_secret(self):
        self.assertNotIn("secret", inspect.signature(filter_candidates).parameters)


class ClueTests(unittest.TestCase):
    ALLOWED = (CountEven, CountLow, DigitPresent, PositionParity, PositionHalf)

    def sample_secrets(self, n=40, length=4, seed=4):
        rng = random.Random(seed)
        return [random_secret(length, rng) for _ in range(n)]

    def test_clues_are_true_for_the_secret_and_use_allowed_templates(self):
        for code in self.sample_secrets():
            clues = make_clues(code, 2, random.Random(1))
            self.assertTrue(1 <= len(clues) <= 2)
            for clue in clues:
                self.assertIsInstance(clue, self.ALLOWED)
                self.assertTrue(clue.holds(code))

    def test_never_an_exact_position_digit(self):
        for code in self.sample_secrets():
            for seed in range(3):
                self.assertFalse(any(isinstance(c, (DigitAt, OpenDigit)) for c in make_clues(code, 3, random.Random(seed))))

    def test_no_duplicate_clue(self):
        for code in self.sample_secrets():
            clues = make_clues(code, 3, random.Random(2))
            self.assertEqual(len(set(clues)), len(clues))

    def test_taken_clues_are_not_repeated(self):
        for code in self.sample_secrets(20):
            first = make_clues(code, 2, random.Random(3))
            more = make_clues(code, 3, random.Random(3), taken=first)
            self.assertFalse(set(more) & set(first))
            for clue in more:
                self.assertTrue(clue.holds(code))

    def test_each_clue_narrows_the_candidates(self):
        for code in self.sample_secrets(20):
            clues = make_clues(code, 3, random.Random(5))
            left = filter_candidates([])
            for clue in clues:
                narrowed = filter_candidates([clue], base=left)
                self.assertLess(len(narrowed), len(left))
                left = narrowed

    def test_all_clues_are_consistent_and_keep_the_secret(self):
        for code in self.sample_secrets(20):
            clues = make_clues(code, 3, random.Random(6))
            self.assertIn(code, filter_candidates(clues))

    def test_deterministic_for_a_seed(self):
        for code in self.sample_secrets(10):
            self.assertEqual(make_clues(code, 2, random.Random(8)), make_clues(code, 2, random.Random(8)))

    def test_a_count_of_zero_gives_no_clues(self):
        self.assertEqual(make_clues(SPEC, 0, random.Random(1)), [])

    def test_five_and_six_digit_clues(self):
        for length in (5, 6):
            for code in self.sample_secrets(8, length):
                clues = make_clues(code, 2, random.Random(2))
                with self.subTest(length=length, code=code):
                    self.assertTrue(clues)
                    self.assertTrue(all(c.holds(code) for c in clues))


class ToolCatalogTests(unittest.TestCase):
    def test_keys_and_costs(self):
        self.assertEqual(
            {k: spec.turn_cost for k, spec in TOOLS.items()},
            {
                "vault_residual_scanner": 1,
                "vault_spectrum_analyzer": 1,
                "vault_comparison_circuit": 1,
                "vault_position_verifier": 1,
                "vault_master_decoder": 2,
            },
        )

    def test_specs_carry_their_key(self):
        for key, spec in TOOLS.items():
            self.assertIsInstance(spec, ToolSpec)
            self.assertEqual(spec.key, key)

    def test_tool_fits(self):
        for key in TOOLS:
            if key == "vault_master_decoder":
                continue
            self.assertTrue(tool_fits(key, 1))
            self.assertFalse(tool_fits(key, 0))
        # The decoder costs two and must leave one turn for a final guess.
        self.assertFalse(tool_fits("vault_master_decoder", 2))
        self.assertTrue(tool_fits("vault_master_decoder", 3))
        self.assertTrue(tool_fits("vault_master_decoder", 9))


class UnknownToolTests(unittest.TestCase):
    def test_tool_fits_rejects_an_unknown_key(self):
        self.assertEqual(code_of(self, tool_fits, "vault_nonsense", 5), "bad_tool")


class ScannerTests(unittest.TestCase):
    KEY = "vault_residual_scanner"

    def test_present_and_absent(self):
        hist = [gr(SPEC, "1234")]
        present = evaluate_tool(SPEC, self.KEY, {"guess": "1234", "digit": "2"}, hist)
        self.assertIsInstance(present, ToolResult)
        self.assertEqual(present.constraint, DigitPresent("2", True))
        absent = evaluate_tool(SPEC, self.KEY, {"guess": "1234", "digit": "1"}, hist)
        self.assertEqual(absent.constraint, DigitPresent("1", False))
        self.assertIsInstance(present.text_key, str)
        self.assertNotEqual(present.text_key, absent.text_key)

    def test_requires_a_past_guess(self):
        self.assertEqual(
            code_of(self, evaluate_tool, SPEC, self.KEY, {"guess": "1234", "digit": "2"}, []),
            "guess_not_in_history",
        )
        hist = [gr(SPEC, "1234")]
        self.assertEqual(
            code_of(self, evaluate_tool, SPEC, self.KEY, {"guess": "9876", "digit": "9"}, hist),
            "guess_not_in_history",
        )

    def test_digit_must_be_in_the_guess(self):
        hist = [gr(SPEC, "1234")]
        self.assertEqual(
            code_of(self, evaluate_tool, SPEC, self.KEY, {"guess": "1234", "digit": "9"}, hist),
            "digit_not_in_guess",
        )

    def test_digit_must_be_a_digit(self):
        hist = [gr(SPEC, "1234")]
        for bad in ("x", "", "12", "-"):
            with self.subTest(digit=bad):
                self.assertEqual(
                    code_of(self, evaluate_tool, SPEC, self.KEY, {"guess": "1234", "digit": bad}, hist),
                    "bad_digit",
                )


class SpectrumTests(unittest.TestCase):
    KEY = "vault_spectrum_analyzer"

    def test_low_and_high(self):
        high = evaluate_tool(SPEC, self.KEY, {"position": 2}, [])
        self.assertEqual(high.constraint, PositionHalf(2, True))  # 8
        low = evaluate_tool(SPEC, self.KEY, {"position": 3}, [])
        self.assertEqual(low.constraint, PositionHalf(3, False))  # 2
        self.assertNotEqual(high.text_key, low.text_key)

    def test_bad_positions(self):
        for position in (0, 5, -1, "a", None):
            with self.subTest(position=position):
                self.assertEqual(code_of(self, evaluate_tool, SPEC, self.KEY, {"position": position}, []), "bad_position")

    def test_position_range_follows_the_length(self):
        self.assertEqual(
            evaluate_tool("012345", self.KEY, {"position": 6}, []).constraint, PositionHalf(6, True)
        )
        self.assertEqual(code_of(self, evaluate_tool, "012345", self.KEY, {"position": 7}, []), "bad_position")


class ComparisonTests(unittest.TestCase):
    KEY = "vault_comparison_circuit"

    def test_counts(self):
        for a, b, count in (("2", "8", 2), ("2", "9", 1), ("0", "9", 0)):
            res = evaluate_tool(SPEC, self.KEY, {"a": a, "b": b}, [])
            with self.subTest(a=a, b=b):
                self.assertEqual(res.constraint.count, count)
                self.assertTrue(res.constraint.holds(SPEC))

    def test_order_does_not_matter(self):
        one = evaluate_tool(SPEC, self.KEY, {"a": "2", "b": "9"}, []).constraint
        two = evaluate_tool(SPEC, self.KEY, {"a": "9", "b": "2"}, []).constraint
        self.assertEqual(sorted(filter_candidates([one])), sorted(filter_candidates([two])))

    def test_constraint_does_not_disclose_positions(self):
        res = evaluate_tool(SPEC, self.KEY, {"a": "2", "b": "8"}, [])
        self.assertIsInstance(res.constraint, PairCount)
        self.assertTrue(res.constraint.holds("8240"))  # same digits, other positions

    def test_digits_must_differ(self):
        self.assertEqual(code_of(self, evaluate_tool, SPEC, self.KEY, {"a": "2", "b": "2"}, []), "digits_equal")

    def test_digits_must_be_digits(self):
        for a, b in (("x", "2"), ("2", "x"), ("12", "3"), ("", "3")):
            with self.subTest(a=a, b=b):
                self.assertEqual(code_of(self, evaluate_tool, SPEC, self.KEY, {"a": a, "b": b}, []), "bad_digit")


class PositionVerifierTests(unittest.TestCase):
    KEY = "vault_position_verifier"

    def test_reveals_a_striking_position(self):
        hist = [gr(SPEC, "5124")]  # strikes at positions 1, 3 and 4
        res = evaluate_tool(SPEC, self.KEY, {"guess": "5124"}, hist)
        self.assertEqual(res.data["position"], 1)
        self.assertEqual(res.data["digit"], "5")
        self.assertEqual(res.constraint, DigitAt(1, "5"))

    def test_picks_the_lowest_undisclosed_striking_position(self):
        guess = "5824"
        hist = [gr(SPEC, guess)]
        first = evaluate_tool(SPEC, self.KEY, {"guess": guess}, hist)
        self.assertEqual((first.data["position"], first.data["digit"]), (1, "5"))
        hist.append(DigitAt(1, "5"))
        second = evaluate_tool(SPEC, self.KEY, {"guess": guess}, hist)
        self.assertEqual((second.data["position"], second.data["digit"]), (2, "8"))
        hist.append(DigitAt(3, "2"))  # a later position disclosed elsewhere does not change the lowest choice
        third = evaluate_tool(SPEC, self.KEY, {"guess": guess}, hist)
        self.assertEqual(third.data["position"], 2)

    def test_choice_is_deterministic(self):
        hist = [gr(SPEC, "5824")]
        results = {evaluate_tool(SPEC, self.KEY, {"guess": "5824"}, hist).data["position"] for _ in range(5)}
        self.assertEqual(results, {1})

    def test_skips_disclosed_positions_until_nothing_is_left(self):
        hist = [gr(SPEC, "5124"), DigitAt(1, "5")]
        res = evaluate_tool(SPEC, self.KEY, {"guess": "5124"}, hist)
        self.assertEqual((res.data["position"], res.data["digit"]), (3, "2"))
        hist.append(DigitAt(3, "2"))
        res = evaluate_tool(SPEC, self.KEY, {"guess": "5124"}, hist)
        self.assertEqual((res.data["position"], res.data["digit"]), (4, "4"))
        hist.append(DigitAt(4, "4"))
        self.assertEqual(code_of(self, evaluate_tool, SPEC, self.KEY, {"guess": "5124"}, hist), "nothing_new")

    def test_zero_strikes_is_a_valid_result(self):
        hist = [gr(SPEC, "1357")]
        res = evaluate_tool(SPEC, self.KEY, {"guess": "1357"}, hist)
        self.assertIsNone(res.data["position"])
        self.assertTrue(res.constraint.holds(SPEC))

    def test_guess_must_be_in_history(self):
        self.assertEqual(code_of(self, evaluate_tool, SPEC, self.KEY, {"guess": "5124"}, []), "guess_not_in_history")
        self.assertEqual(
            code_of(self, evaluate_tool, SPEC, self.KEY, {"guess": "5124"}, [DigitAt(1, "5")]),
            "guess_not_in_history",
        )


class MasterDecoderTests(unittest.TestCase):
    KEY = "vault_master_decoder"

    def test_reveals_the_digit(self):
        res = evaluate_tool(SPEC, self.KEY, {"position": 3}, [])
        self.assertEqual(res.constraint, DigitAt(3, "2"))
        self.assertEqual(res.data["position"], 3)
        self.assertEqual(res.data["digit"], "2")

    def test_leading_zero_digit(self):
        self.assertEqual(evaluate_tool("0127", self.KEY, {"position": 1}, []).data["digit"], "0")

    def test_rejects_an_already_revealed_position(self):
        self.assertEqual(
            code_of(self, evaluate_tool, SPEC, self.KEY, {"position": 3}, [DigitAt(3, "2")]),
            "already_revealed",
        )
        # A different disclosed position does not block it.
        evaluate_tool(SPEC, self.KEY, {"position": 3}, [DigitAt(1, "5")])

    def test_bad_position(self):
        for position in (0, 5):
            with self.subTest(position=position):
                self.assertEqual(code_of(self, evaluate_tool, SPEC, self.KEY, {"position": position}, []), "bad_position")

    def test_turn_rule_is_the_callers_check(self):
        self.assertFalse(tool_fits(self.KEY, 2))
        self.assertTrue(tool_fits(self.KEY, 3))


class ToolPurityTests(unittest.TestCase):
    def test_every_result_constraint_holds_for_the_secret(self):
        rng = random.Random(13)
        for _ in range(30):
            code = random_secret(4, rng)
            guess = random_secret(4, rng)
            hist = [gr(code, guess)]
            cases = (
                ("vault_residual_scanner", {"guess": guess, "digit": guess[0]}),
                ("vault_spectrum_analyzer", {"position": 1 + rng.randrange(4)}),
                ("vault_comparison_circuit", {"a": "0", "b": "9"}),
                ("vault_position_verifier", {"guess": guess}),
                ("vault_master_decoder", {"position": 1 + rng.randrange(4)}),
            )
            for key, params in cases:
                res = evaluate_tool(code, key, params, hist)
                self.assertTrue(res.constraint.holds(code), (key, code))

    def test_inputs_are_not_mutated(self):
        hist = [gr(SPEC, "1234")]
        params = {"guess": "1234", "digit": "2"}
        evaluate_tool(SPEC, "vault_residual_scanner", params, hist)
        self.assertEqual(hist, [gr(SPEC, "1234")])
        self.assertEqual(params, {"guess": "1234", "digit": "2"})

    def test_tools_work_on_longer_codes(self):
        code = "012345"
        hist = [gr(code, "543210")]
        self.assertEqual(
            evaluate_tool(code, "vault_master_decoder", {"position": 6}, []).data["digit"], "5"
        )
        self.assertEqual(
            evaluate_tool(code, "vault_residual_scanner", {"guess": "543210", "digit": "3"}, hist).constraint,
            DigitPresent("3", True),
        )


class RedundancyTests(unittest.TestCase):
    def test_decoder_redundant_when_the_position_is_settled(self):
        same_first = [s for s in all_secrets(4) if s[0] == "5"]
        self.assertTrue(is_redundant("vault_master_decoder", {"position": 1}, same_first))
        self.assertFalse(is_redundant("vault_master_decoder", {"position": 2}, same_first))
        self.assertFalse(is_redundant("vault_master_decoder", {"position": 1}, list(all_secrets(4))))

    def test_spectrum_redundant_when_the_half_is_settled(self):
        low_first = [s for s in all_secrets(4) if int(s[0]) <= 4]
        self.assertTrue(is_redundant("vault_spectrum_analyzer", {"position": 1}, low_first))
        self.assertFalse(is_redundant("vault_spectrum_analyzer", {"position": 2}, low_first))
        self.assertFalse(is_redundant("vault_spectrum_analyzer", {"position": 1}, list(all_secrets(4))))

    def test_comparison_redundant_when_the_count_is_settled(self):
        both = [s for s in all_secrets(4) if "2" in s and "8" in s]
        self.assertTrue(is_redundant("vault_comparison_circuit", {"a": "2", "b": "8"}, both))
        neither = [s for s in all_secrets(4) if "2" not in s and "8" not in s]
        self.assertTrue(is_redundant("vault_comparison_circuit", {"a": "2", "b": "8"}, neither))
        self.assertFalse(is_redundant("vault_comparison_circuit", {"a": "2", "b": "8"}, list(all_secrets(4))))
        self.assertFalse(is_redundant("vault_comparison_circuit", {"a": "2", "b": "8"}, both + neither))

    def test_history_makes_an_already_revealed_position_redundant(self):
        """Every candidate raises (already_revealed), so the query is treated as redundant."""
        full = list(all_secrets(4))
        hist = [DigitAt(1, "5")]
        self.assertTrue(is_redundant("vault_master_decoder", {"position": 1}, full, hist))
        self.assertTrue(is_redundant("vault_master_decoder", {"position": 1}, full, history=hist))
        self.assertFalse(is_redundant("vault_master_decoder", {"position": 2}, full, hist))
        self.assertFalse(is_redundant("vault_master_decoder", {"position": 1}, full))

    def test_a_single_candidate_makes_everything_redundant(self):
        only = [SPEC]
        self.assertTrue(is_redundant("vault_master_decoder", {"position": 2}, only))
        self.assertTrue(is_redundant("vault_spectrum_analyzer", {"position": 4}, only))
        self.assertTrue(is_redundant("vault_comparison_circuit", {"a": "1", "b": "2"}, only))

    def test_signature_has_no_secret(self):
        self.assertNotIn("secret", inspect.signature(is_redundant).parameters)


class CrossValidationTests(unittest.TestCase):
    def test_spec_example(self):
        # Guesses 1234 and 3456 share {3, 4}; only the 4 is in 5824.
        self.assertEqual(cross_validation_common("1234", "3456"), frozenset({"3", "4"}))
        self.assertEqual(cross_validation(SPEC, "1234", "3456"), 1)

    def test_symmetric_and_bounded(self):
        rng = random.Random(17)
        for _ in range(40):
            code, a, b = (random_secret(4, rng) for _ in range(3))
            common = cross_validation_common(a, b)
            self.assertEqual(common, cross_validation_common(b, a))
            self.assertEqual(common, frozenset(set(a) & set(b)))
            n = cross_validation(code, a, b)
            self.assertEqual(n, cross_validation(code, b, a))
            self.assertEqual(n, len(common & set(code)))

    def test_disjoint_guesses_share_nothing(self):
        self.assertEqual(cross_validation_common("0123", "4567"), frozenset())
        self.assertEqual(cross_validation(SPEC, "0123", "4567"), 0)

    def test_common_returns_a_frozenset(self):
        self.assertIsInstance(cross_validation_common("1234", "3456"), frozenset)


class NotSecretTests(unittest.TestCase):
    def test_holds_unless_the_guess_is_the_secret(self):
        self.assertTrue(NotSecret("1234").holds(SPEC))
        self.assertFalse(NotSecret(SPEC).holds(SPEC))
        self.assertEqual([s for s in all_secrets(4) if NotSecret("0123").holds(s)], list(all_secrets(4)[1:]))

    def test_frozen_hashable_and_value_equal(self):
        con = NotSecret("1234")
        self.assertEqual(con, NotSecret("1234"))
        self.assertNotEqual(con, NotSecret("1235"))
        hash(con)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            con.guess = "9999"


class PlayedTests(unittest.TestCase):
    def test_a_wrong_guess_adds_not_secret(self):
        cons = played("standard", "1234", (1, 1))
        self.assertIn(GuessResult("1234", "standard", (1, 1)), cons)
        self.assertIn(NotSecret("1234"), cons)

    def test_a_won_guess_does_not_add_not_secret(self):
        cons = played("standard", SPEC, (4, 0), won=True)
        self.assertEqual(cons, [GuessResult(SPEC, "standard", (4, 0))])
        self.assertFalse(any(isinstance(c, NotSecret) for c in cons))

    def test_won_defaults_to_false(self):
        self.assertIn(NotSecret("1234"), played("total_match", "1234", (4,)))

    def test_total_match_wrong_guess_leaves_the_candidates(self):
        """A total_match result of (L,) is the guess's own feedback; only NotSecret removes the guess itself."""
        code = SPEC
        guess = "4285"  # same digits, every position wrong
        result = feedback("total_match", code, guess)
        self.assertEqual(result, (4,))
        self.assertIn(guess, filter_candidates([GuessResult(guess, "total_match", result)]))
        left = filter_candidates(played("total_match", guess, result))
        self.assertNotIn(guess, left)
        self.assertIn(code, left)

    def test_every_protocol_removes_a_wrong_guess_and_keeps_the_secret(self):
        for protocol in PROTOCOLS:
            guess = "4285"
            result = feedback(protocol, SPEC, guess)
            left = filter_candidates(played(protocol, guess, result))
            with self.subTest(protocol=protocol):
                self.assertNotIn(guess, left)
                self.assertIn(SPEC, left)


class NoRepeatGameTests(unittest.TestCase):
    """Optimal play (weight 0) built on played() never repeats a guess, for the low-information protocols."""

    def play(self, code, protocol, first, max_turns):
        cands = list(all_secrets(4))
        guess, seen = first, []
        for turn in range(1, max_turns + 1):
            self.assertNotIn(guess, seen, (protocol, code, turn))
            seen.append(guess)
            won = is_win(code, guess)
            cands = filter_candidates(played(protocol, guess, feedback(protocol, code, guess), won=won), 4, base=cands)
            self.assertIn(code, cands)
            if won:
                return turn
            self.assertNotIn(guess, cands, (protocol, code, turn))  # the played guess leaves the list
            guess = suggest_guess(cands, protocol, 4, 0, pool_limit=300)
        return None

    def sample(self, n, seed):
        rng = random.Random(seed)
        return [random_secret(4, rng) for _ in range(n)]

    def test_total_match_never_repeats_and_terminates(self):
        first = suggest_guess(list(all_secrets(4)), "total_match", 4, 0, pool_limit=60)
        for code in self.sample(10, 43):
            # Only the digit set is learnable from feedback, then at most 24 orderings; 30 turns is a safe bound.
            turns = self.play(code, "total_match", first, 30)
            with self.subTest(code=code):
                self.assertIsNotNone(turns)

    def test_position_only_never_repeats(self):
        first = suggest_guess(list(all_secrets(4)), "position_only", 4, 0, pool_limit=60)
        for code in self.sample(10, 44):
            self.play(code, "position_only", first, 12)


class RankGuessesTests(unittest.TestCase):
    def test_small_pool_scores_every_candidate_exactly(self):
        cands = list(all_secrets(4)[:40])
        ranked = rank_guesses(cands, "standard", 4)
        self.assertEqual({s.guess for s in ranked}, set(cands))
        self.assertTrue(all(isinstance(s, Suggestion) and s.in_candidates for s in ranked))
        for s in ranked:
            groups = {}
            for c in cands:
                groups.setdefault(feedback("standard", c, s.guess), []).append(c)
            sizes = [len(v) for v in groups.values()]
            self.assertEqual(s.worst_case, max(sizes))
            self.assertAlmostEqual(s.expected_remaining, sum(n * n for n in sizes) / len(cands))

    def test_suggestion_is_frozen(self):
        s = rank_guesses(["0123", "0132"], "standard", 4)[0]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            s.guess = "9999"

    def test_sorted_best_first_with_stable_ties(self):
        cands = list(all_secrets(4)[:60])
        ranked = rank_guesses(cands, "standard", 4)
        keys = [(s.worst_case, s.expected_remaining, not s.in_candidates, s.guess) for s in ranked]
        self.assertEqual(keys, sorted(keys))

    def test_other_protocols_score_with_their_own_feedback(self):
        cands = list(all_secrets(4)[:50])
        for protocol in PROTOCOLS:
            ranked = rank_guesses(cands, protocol, 4)
            with self.subTest(protocol=protocol):
                self.assertEqual(len(ranked), len(cands))
                for s in ranked[:5]:
                    sizes = {}
                    for c in cands:
                        k = feedback(protocol, c, s.guess)
                        sizes[k] = sizes.get(k, 0) + 1
                    self.assertEqual(s.worst_case, max(sizes.values()))

    def test_a_single_candidate(self):
        ranked = rank_guesses(["0123"], "standard", 4)
        self.assertEqual([(s.guess, s.in_candidates, s.worst_case) for s in ranked], [("0123", True, 1)])
        self.assertEqual(ranked[0].expected_remaining, 1)

    def test_large_sets_are_sampled_within_the_limit(self):
        cands = list(all_secrets(4))
        ranked = rank_guesses(cands, "standard", 4, pool_limit=40, rng=random.Random(1))
        self.assertLessEqual(len(ranked), 40)
        self.assertGreater(len(ranked), 0)
        self.assertEqual(len({s.guess for s in ranked}), len(ranked))
        members = set(cands)
        self.assertTrue(all(s.in_candidates == (s.guess in members) for s in ranked))

    def test_sampling_is_reproducible_for_a_seed(self):
        cands = list(all_secrets(4))
        one = rank_guesses(cands, "standard", 4, pool_limit=30, rng=random.Random(5))
        two = rank_guesses(cands, "standard", 4, pool_limit=30, rng=random.Random(5))
        self.assertEqual(one, two)

    def test_sampled_pool_can_include_non_candidates(self):
        cands = [s for s in all_secrets(4) if s.startswith("01")]  # 56 candidates, pool_limit smaller
        ranked = rank_guesses(cands, "standard", 4, pool_limit=30, rng=random.Random(2))
        self.assertLessEqual(len(ranked), 30)
        self.assertTrue(any(s.in_candidates for s in ranked))

    def test_six_digit_candidates_are_ranked_with_a_small_pool(self):
        code = "012345"
        cands = filter_candidates([gr(code, "012678"), gr(code, "345012")], 6)
        self.assertIn(code, cands)
        ranked = rank_guesses(cands, "standard", 6, pool_limit=15, rng=random.Random(3))
        self.assertTrue(0 < len(ranked) <= 15)
        self.assertTrue(all(len(s.guess) == 6 for s in ranked))
        self.assertTrue(all(1 <= s.worst_case <= len(cands) for s in ranked))

    def test_no_secret_parameter(self):
        self.assertNotIn("secret", inspect.signature(rank_guesses).parameters)


class StyleWeightTests(unittest.TestCase):
    def test_weight_is_the_mean_of_risk_and_aggression(self):
        self.assertEqual(weight_from_style(0, 0), 0.0)
        self.assertEqual(weight_from_style(100, 100), 1.0)
        self.assertAlmostEqual(weight_from_style(40, 60), 0.5)
        self.assertAlmostEqual(weight_from_style(100, 0), 0.5)
        self.assertAlmostEqual(weight_from_style(20, 30), 0.25)

    def test_weight_stays_in_the_unit_range(self):
        for risk, aggression in itertools.product((0, 25, 50, 75, 100), repeat=2):
            self.assertTrue(0.0 <= weight_from_style(risk, aggression) <= 1.0)


class SuggestGuessTests(unittest.TestCase):
    CANDS = list(all_secrets(4)[:200])

    def minimax_key(self, s):
        return (s.worst_case, s.expected_remaining, not s.in_candidates, s.guess)

    def test_signature_has_no_secret(self):
        params = inspect.signature(suggest_guess).parameters
        self.assertNotIn("secret", params)
        for name in ("candidates", "protocol", "length", "weight", "unpredictability", "seed", "pool_limit"):
            self.assertIn(name, params)
        self.assertEqual(params["unpredictability"].default, 0)
        self.assertEqual(params["seed"].default, "")
        self.assertEqual(params["pool_limit"].default, 300)

    def test_one_candidate_is_returned(self):
        for weight in (0, 0.5, 1):
            self.assertEqual(suggest_guess(["0123"], "standard", 4, weight), "0123")

    def test_weight_zero_is_minimax(self):
        pick = suggest_guess(self.CANDS, "standard", 4, 0)
        ranked = rank_guesses(self.CANDS, "standard", 4, pool_limit=300)
        best = min(ranked, key=self.minimax_key)
        self.assertEqual(pick, best.guess)

    def test_weight_one_is_a_candidate(self):
        self.assertIn(suggest_guess(self.CANDS, "standard", 4, 1), self.CANDS)
        # Even when a non-candidate would split the candidates better.
        few = ["0123", "0124", "0125"]
        self.assertIn(suggest_guess(few, "standard", 4, 1), few)

    def test_in_between_weights_return_a_legal_guess(self):
        for weight in (0.25, 0.5, 0.75):
            pick = suggest_guess(self.CANDS, "standard", 4, weight)
            self.assertTrue(valid_code(pick, 4))

    def test_deterministic_for_the_same_arguments(self):
        for kwargs in ({}, {"unpredictability": 80, "seed": "t1"}):
            one = suggest_guess(self.CANDS, "standard", 4, 0.3, **kwargs)
            two = suggest_guess(self.CANDS, "standard", 4, 0.3, **kwargs)
            self.assertEqual(one, two)

    def test_deterministic_with_sampling(self):
        cands = list(all_secrets(4))
        one = suggest_guess(cands, "standard", 4, 0.5, 0, "s", 40)
        two = suggest_guess(cands, "standard", 4, 0.5, 0, "s", 40)
        self.assertEqual(one, two)

    def test_zero_unpredictability_ignores_the_seed(self):
        picks = {suggest_guess(self.CANDS, "standard", 4, 0, 0, f"seed{i}") for i in range(10)}
        self.assertEqual(len(picks), 1)

    def test_unpredictability_picks_among_the_top_three(self):
        ranked = rank_guesses(self.CANDS, "standard", 4, pool_limit=300)
        top3 = {s.guess for s in sorted(ranked, key=self.minimax_key)[:3]}
        picks = [suggest_guess(self.CANDS, "standard", 4, 0, 100, f"seed{i}") for i in range(120)]
        self.assertTrue(set(picks) <= top3)

    def test_deviation_is_capped_near_the_playstyle_cap(self):
        cands = list(all_secrets(4)[:30])  # small so 1000 draws stay fast
        best = suggest_guess(cands, "standard", 4, 0, 0)
        picks = [suggest_guess(cands, "standard", 4, 0, 100, f"seed{i}") for i in range(1000)]
        deviated = sum(1 for p in picks if p != best) / len(picks)
        self.assertGreater(deviated, 0)  # unpredictability does something
        # MAX_ERRATIC 0.35; the standard deviation at n=1000 is about 0.015, so 0.05 is over three sigma.
        self.assertLessEqual(deviated, 0.35 + 0.05)

    def test_five_and_six_digit_suggestions(self):
        for length, code in ((5, "01234"), (6, "012345")):
            cands = filter_candidates([gr(code, code[::-1])], length)
            pick = suggest_guess(cands, "standard", length, 0, pool_limit=20)
            with self.subTest(length=length):
                self.assertTrue(valid_code(pick, length))
                self.assertEqual(pick, suggest_guess(cands, "standard", length, 0, pool_limit=20))


class OptimalGameRegressionTests(unittest.TestCase):
    """A full standard 4-digit game with the optimal strategy (weight 0) solves every sampled secret in 9 turns."""

    def play(self, code, first, turns=9):
        cands = list(all_secrets(4))
        guess = first
        for turn in range(1, turns + 1):
            if is_win(code, guess):
                return turn
            cands = filter_candidates([gr(code, guess)], 4, base=cands)
            self.assertIn(code, cands)
            guess = suggest_guess(cands, "standard", 4, 0, pool_limit=300)
        return None if not is_win(code, guess) else turns + 1

    def test_sampled_secrets_are_solved_within_nine_turns(self):
        rng = random.Random(2026)
        sample = [random_secret(4, rng) for _ in range(20)] + ["0123", "9876"]
        first = suggest_guess(list(all_secrets(4)), "standard", 4, 0, pool_limit=60)
        for code in sample:
            turns = self.play(code, first)
            with self.subTest(code=code):
                self.assertIsNotNone(turns)
                self.assertLessEqual(turns, 9)


if __name__ == "__main__":
    unittest.main()
