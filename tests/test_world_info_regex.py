"""World Info regex keys (``/pattern/flags``) evaluated in a shared MiniRacer isolate (REL-03).

Characterizations pin the matching semantics of the process-wide isolate with its cached compiled
RegExp objects. The count tests guard reuse by wrapping ``py_mini_racer.MiniRacer`` (and
``llmcord_core.world_info.MiniRacer`` if the module imports it at top level) with a counting factory
that delegates to the real class.
"""
import itertools
import json
import sys
import time
import unittest
from contextlib import ExitStack, contextmanager
from unittest import mock

import py_mini_racer

from llmcord_core import world_info
from llmcord_core.lore import lore_scopes
from llmcord_core.lorebooks import parse_lorebook
from llmcord_core.store import Store
from llmcord_core.world_info import _key_matches, evaluate

CATASTROPHIC = "/(a+)+$/"
_unique = itertools.count()


def catastrophic_text() -> str:
    """A fresh input each call so no (pattern, flags, text) cache can short-circuit the timeout."""
    return "a" * 30 + "!" + "b" * next(_unique)


@contextmanager
def quiet_unraisable():
    """mini-racer reports a timed-out callback via sys.unraisablehook; keep it out of test output."""
    with mock.patch.object(sys, "unraisablehook", lambda unraisable: None):
        yield


@contextmanager
def count_isolates():
    real = py_mini_racer.MiniRacer
    created = []

    def factory(*args, **kwargs):
        created.append(1)
        return real(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(py_mini_racer, "MiniRacer", factory))
        if hasattr(world_info, "MiniRacer"):
            stack.enter_context(mock.patch.object(world_info, "MiniRacer", factory))
        yield created


class RegexKeySemanticsTests(unittest.TestCase):
    def test_case_insensitive_flag(self):
        """Characterization (REL-03): `/dragon/i` matches regardless of case."""
        self.assertTrue(_key_matches("/dragon/i", "A DRAGON appears", {}))
        self.assertFalse(_key_matches("/dragon/", "A DRAGON appears", {}))

    def test_multiline_flag(self):
        """Characterization (REL-03): `m` makes ^/$ match per line."""
        text = "first\nfoo\nlast"
        self.assertTrue(_key_matches("/^foo$/m", text, {}))
        self.assertFalse(_key_matches("/^foo$/", text, {}))

    def test_global_flag_is_stateless_across_evaluations(self):
        """Characterization (REL-03): a `g` key matches the same text every time (no lastIndex carry-over)."""
        results = [_key_matches("/dragon/g", "a dragon here", {}) for _ in range(3)]
        self.assertEqual(results, [True, True, True])

    def test_sticky_flag_is_stateless_across_evaluations(self):
        """Characterization (REL-03): a `y` key anchored at 0 matches the same text every time."""
        results = [_key_matches("/foo/y", "foo bar", {}) for _ in range(3)]
        self.assertEqual(results, [True, True, True])

    def test_invalid_flags_never_match(self):
        """Characterization (REL-03): flags outside "gimsuyd" make the key a non-match."""
        self.assertFalse(_key_matches("/dragon/x", "dragon", {}))
        self.assertFalse(_key_matches("/dragon/ix", "dragon", {}))

    def test_overlong_pattern_never_matches(self):
        """Characterization (REL-03): patterns longer than 500 chars are rejected."""
        self.assertFalse(_key_matches("/" + "a" * 501 + "/", "a" * 600, {}))
        self.assertTrue(_key_matches("/" + "a" * 500 + "/", "a" * 600, {}))

    def test_malformed_pattern_never_matches(self):
        """Characterization (REL-03): a JS SyntaxError in the pattern is a non-match, not an error."""
        self.assertFalse(_key_matches("/(/", "(", {}))
        self.assertTrue(_key_matches("/\\(/", "(", {}), "isolate still usable after a syntax error")

    def test_injection_shaped_text_is_inert(self):
        """Characterization (REL-03): JS line separators, lone surrogates, quotes, backslashes and
        injection-shaped text are passed as data and cannot overwrite the shared helper."""
        injection = '"); __wiTest = function(){return true}; ("'
        cases = [
            ("/a\u2028b/", "a\u2028b", True),
            ("/a\u2029b/", "a\u2029b", True),
            ("/dragon/", "line\u2028sep\u2029dragon", True),
            ("/dragon/", "line\u2028sep\u2029wyvern", False),
            ("/x/", "lone \ud800 x", True),
            ("/x/", "lone \ud800 y", False),
            ("/dragon/", 'say "dragon" \\ ok', True),
            ("/dragon/", 'say "wyvern" \\ ok', False),
            ("/dragon/", injection, False),
            ("/__wiTest/", injection, True),
        ]
        for key, text, expected in cases:
            with self.subTest(key=key, text=text):
                self.assertIs(_key_matches(key, text, {}), expected)
        self.assertFalse(_key_matches("/zzz/", "abc", {}), "helper not overwritten by injected text")

    def test_pattern_with_quotes_and_backslashes(self):
        """Characterization (REL-03): quotes and backslashes in the pattern reach RegExp intact."""
        key = '/"\\\\/'  # pattern "\\ : a double quote followed by one backslash
        self.assertTrue(_key_matches(key, 'say "\\ ok', {}))
        self.assertFalse(_key_matches(key, 'say " ok', {}))
        self.assertFalse(_key_matches('/"); __wiTest = 1; ("/', "abc", {}))
        self.assertFalse(_key_matches("/zzz/", "abc", {}), "helper not overwritten by injected pattern")

    def test_regex_disabled_treats_key_as_literal(self):
        """Characterization (REL-03): regex_enabled False matches `/x/` as literal text."""
        self.assertTrue(_key_matches("/x/", "see /x/ here", {"regex_enabled": False}))
        self.assertFalse(_key_matches("/x/", "just x", {"regex_enabled": False}))
        self.assertTrue(_key_matches("/x/", "just x", {}), "absent flag still evaluates regex")

    def test_regex_enabled_rejects_plain_key(self):
        """Characterization (REL-03): regex_enabled True with a non-regex key never matches."""
        self.assertFalse(_key_matches("dragon", "dragon", {"regex_enabled": True}))
        self.assertTrue(_key_matches("/dragon/", "dragon", {"regex_enabled": True}))

    def test_catastrophic_pattern_times_out_and_later_keys_still_match(self):
        """Characterization (REL-03): catastrophic backtracking returns False quickly; later keys work."""
        started = time.monotonic()
        with quiet_unraisable():
            result = _key_matches(CATASTROPHIC, catastrophic_text(), {})
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertFalse(result)
        self.assertTrue(_key_matches("/dragon/i", "A DRAGON", {}))
        self.assertFalse(_key_matches("/dragon/i", "a wyvern", {}))


class RegexIsolateReuseTests(unittest.TestCase):
    def test_many_regex_keys_share_one_isolate(self):
        """REL-03 (fixed): twenty distinct regex-key evaluations construct at most one MiniRacer isolate."""
        with count_isolates() as created:
            results = [_key_matches(f"/word{index}/", f"has word{index}", {}) for index in range(20)]
        self.assertEqual(results, [True] * 20)
        self.assertLessEqual(len(created), 1)

    def test_timeout_discards_isolate_and_recreates_once(self):
        """REL-03 (fixed): normal, catastrophic, normal, normal constructs exactly two isolates."""
        with quiet_unraisable():
            # A timeout leaves no live isolate, so counting below starts from a known state.
            self.assertFalse(_key_matches(CATASTROPHIC, catastrophic_text(), {}))
            with count_isolates() as created:
                first = _key_matches("/alpha/", "alpha one", {})
                stuck = _key_matches(CATASTROPHIC, catastrophic_text(), {})
                second = _key_matches("/beta/", "beta two", {})
                third = _key_matches("/gamma/", "gamma three", {})
        self.assertEqual((first, stuck, second, third), (True, False, True, True))
        self.assertEqual(len(created), 2)

    def test_syntax_error_keeps_isolate(self):
        """REL-03 (fixed): a malformed pattern is a plain SyntaxError and does not discard the isolate."""
        self.assertTrue(_key_matches("/warm/", "warm up", {}))
        with count_isolates() as created:
            self.assertFalse(_key_matches("/(/", "(", {}))
            self.assertTrue(_key_matches("/after/", "after error", {}))
        self.assertEqual(len(created), 0)

    def test_missing_mini_racer_is_a_non_match(self):
        """REL-03 (fixed): if py_mini_racer cannot be imported, a regex key is a non-match (as before the
        shared isolate), not an ImportError that aborts the turn; literal keys still match."""
        world_info._discard_isolate()
        with mock.patch.dict(sys.modules, {"py_mini_racer": None}):
            self.assertFalse(_key_matches("/x/", "x", {}))
            self.assertTrue(_key_matches("dragon", "a dragon", {}))


class EvaluateRegexIsolateTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.world = self.store.create_space(1, "World", "world")
        self.store.bind_channel(1, 100, self.world)
        self.alice = self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])
        book = self.store.create_lorebook(1, "Regex", "channel", 100)
        entries = {str(index): {"key": [f"/thing{index}\\b/i"], "content": f"lore {index}"}
                   for index in range(20)}
        self.store.sync_lorebook(1, book, parse_lorebook(json.dumps({"entries": entries}).encode()), {}, 0)

    def tearDown(self):
        self.store.close()

    def run_evaluate(self):
        return evaluate(self.store, 1, lore_scopes(self.store, self.alice, self.world, 100),
                        "THING3 and thing7 appear", 10000,
                        character=self.store.character_by_id(self.alice), response_id=1)

    def test_evaluate_matches_regex_lorebook_keys(self):
        """Characterization (REL-03): evaluate() activates only lorebook entries whose regex key matches."""
        self.assertEqual({item.content for item in self.run_evaluate()}, {"lore 3", "lore 7"})

    def test_evaluate_uses_at_most_one_isolate(self):
        """REL-03 (fixed): one evaluate() over twenty regex-keyed entries constructs at most one isolate."""
        with count_isolates() as created:
            found = self.run_evaluate()
        self.assertEqual({item.content for item in found}, {"lore 3", "lore 7"})
        self.assertLessEqual(len(created), 1)


if __name__ == "__main__":
    unittest.main()
