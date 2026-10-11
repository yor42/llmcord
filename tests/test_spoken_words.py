"""Spoken-move word matcher (llmcord_core/spoken/words.py), D32 / FEAT-37 preliminary seam.

Uses a small local lexicon shaped like the packaged Hold'em one so the matcher is tested game-agnostically.
"""
import dataclasses
import unittest

from llmcord_core.spoken.moves import Reading, SpokenMove
from llmcord_core.spoken.words import read

LEX = {
    "fold": {"en": ["fold", "i fold", "i'm out", "muck"], "ko": ["폴드", "다이", "포기"]},
    "check": {"en": ["check", "knock"], "ko": ["체크"]},
    "call": {"en": ["call", "i call", "i'll call", "call it"], "ko": ["콜", "받을게"]},
    "raise": {"en": ["raise", "bet", "re-raise", "bump it"], "ko": ["레이즈", "베팅"]},
    "all_in": {"en": ["all in", "allin", "shove", "jam"], "ko": ["올인"]},
    "_presets": {
        "min": {"en": ["min", "min raise", "minraise", "minimum"], "ko": ["민레이즈", "최소"]},
        "half_pot": {"en": ["half pot", "half"], "ko": ["하프", "반팟"]},
        "pot": {"en": ["pot", "pot it", "pot size"], "ko": ["팟", "풀팟"]},
    },
    "_preset_alone": {"en": ["half pot", "pot it", "pot size", "min raise", "minraise"],
                      "ko": ["하프", "반팟", "풀팟", "민레이즈"]},
    "_exclude": ["콜라", "콜렉션", "체크카드", "다이어트", "다이아"],
    "_amount_kinds": ["raise"],
}
NEG = {
    "en": ["don't", "dont", "do not", "no", "not", "never", "won't", "wont"],
    "ko": ["안", "말", "하지 마", "하지마", "못"],
}
ALL = ("fold", "check", "call", "raise", "all_in")


def rd(text, kinds=ALL):
    return read(text, LEX, kinds, NEG)


def kinds_of(reading):
    return [c.kind for c in reading.candidates]


class ShapeTests(unittest.TestCase):
    def test_spoken_move_defaults_and_frozen(self):
        m = SpokenMove("call")
        self.assertEqual((m.kind, m.amount, m.mode, m.preset), ("call", None, None, None))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            m.kind = "fold"

    def test_reading_frozen(self):
        r = Reading((SpokenMove("call"),), True)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            r.sure = False

    def test_package_reexports(self):
        from llmcord_core import spoken
        self.assertIs(spoken.SpokenMove, SpokenMove)
        self.assertIs(spoken.Reading, Reading)


class EnglishMatchTests(unittest.TestCase):
    def test_simple_word(self):
        self.assertEqual(rd("call"), Reading((SpokenMove("call"),), True))

    def test_word_boundaries(self):
        self.assertIsNone(rd("open the folder"))
        self.assertIsNone(rd("I will recall that"))
        self.assertIsNone(rd("callous"))

    def test_phrases_and_case_and_whitespace(self):
        self.assertEqual(kinds_of(rd("FINE, I'll   Call")), ["call"])
        self.assertEqual(kinds_of(rd("i'm out")), ["fold"])

    def test_two_entries_of_one_kind_are_one_candidate(self):
        r = rd("I call, call it")
        self.assertEqual(kinds_of(r), ["call"])
        self.assertTrue(r.sure)

    def test_past_tense_and_exact_words(self):
        for text in ("I called him earlier", "he folded", "betting", "betted", "potted plant", "mining", "jammed"):
            with self.subTest(text=text):
                self.assertIsNone(rd(text))
        for text, kind in (("calls", "call"), ("folding", "fold"), ("raising", "raise"), ("bet", "raise")):
            with self.subTest(text=text):
                self.assertEqual(kinds_of(rd(text)), [kind])

    def test_hyphen_and_underscore_are_spaces(self):
        self.assertEqual(kinds_of(rd("all-in")), ["all_in"])
        self.assertEqual(kinds_of(rd("all_in")), ["all_in"])
        self.assertEqual(kinds_of(rd("ALL   IN")), ["all_in"])
        self.assertEqual(kinds_of(rd("allin")), ["all_in"])

    def test_hyphenated_lexicon_entry(self):
        r = rd("re-raise to 400")
        self.assertEqual(r.candidates, (SpokenMove("raise", 400, "to"),))

    def test_unrelated_chat_is_none(self):
        self.assertIsNone(rd("nice hand"))
        self.assertIsNone(rd(""))


class KoreanMatchTests(unittest.TestCase):
    def test_suffixes_allowed(self):
        for text, kind in (("콜", "call"), ("콜요", "call"), ("콜할게", "call"), ("다이할래", "fold"),
                           ("올인 간다", "all_in"), ("그냥 다이", "fold"), ("콜 ㄱㄱ", "call")):
            with self.subTest(text=text):
                self.assertEqual(kinds_of(rd(text)), [kind])

    def test_matches_only_at_token_start(self):
        self.assertIsNone(rd("아이콜"))

    def test_excludes(self):
        self.assertIsNone(rd("콜라 마시고 올게"))
        self.assertIsNone(rd("체크카드 있어?"))
        self.assertIsNone(rd("다이어트 중"))
        self.assertEqual(kinds_of(rd("콜라 말고 콜")), ["call"])

    def test_nfd_input_is_normalized(self):
        import unicodedata
        self.assertEqual(kinds_of(rd(unicodedata.normalize("NFD", "콜할게"))), ["call"])


class SureTests(unittest.TestCase):
    def test_clean_single_kind_is_sure(self):
        self.assertTrue(rd("call").sure)

    def test_english_negation_is_unsure(self):
        for text in ("don't fold", "I won't call", "do not call", "never fold", "not folding", "no call"):
            with self.subTest(text=text):
                r = rd(text)
                self.assertIsNotNone(r)
                self.assertFalse(r.sure)

    def test_negation_needs_word_boundary(self):
        self.assertTrue(rd("call, nothing else").sure)
        self.assertTrue(rd("call, I know").sure)

    def test_korean_negation_is_unsure(self):
        for text in ("콜 안 할래", "콜 못해", "콜하지마", "폴드 말고 콜"):
            with self.subTest(text=text):
                r = rd(text)
                self.assertIsNotNone(r)
                self.assertFalse(r.sure)

    def test_korean_negation_not_inside_ordinary_words(self):
        self.assertTrue(rd("안녕 콜").sure)
        self.assertFalse(rd("콜 안 해").sure)
        self.assertFalse(rd("콜 안해").sure)

    def test_question_mark_is_unsure(self):
        r = rd("should I call?")
        self.assertEqual(kinds_of(r), ["call"])
        self.assertFalse(r.sure)

    def test_two_kinds_two_candidates_unsure_in_order(self):
        r = rd("call or fold")
        self.assertEqual(kinds_of(r), ["call", "fold"])
        self.assertFalse(r.sure)
        self.assertEqual(kinds_of(rd("fold... no wait, call")), ["fold", "call"])

    def test_three_kinds_keep_first_two(self):
        r = rd("check, fold, call")
        self.assertEqual(kinds_of(r), ["check", "fold"])
        self.assertFalse(r.sure)

    def test_mixed_language_order(self):
        self.assertEqual(kinds_of(rd("콜 아니면 fold")), ["call", "fold"])


class KindsFilterTests(unittest.TestCase):
    def test_illegal_kinds_ignored(self):
        self.assertIsNone(rd("fold", kinds=("call", "raise")))
        r = rd("fold or call", kinds=("call", "raise"))
        self.assertEqual(r, Reading((SpokenMove("call"),), True))

    def test_empty_kinds(self):
        self.assertIsNone(rd("call", kinds=()))


class AmountAndPresetTests(unittest.TestCase):
    def test_amount_only_for_amount_kinds(self):
        self.assertEqual(rd("raise to 300 ㅋㅋ").candidates, (SpokenMove("raise", 300, "to"),))
        self.assertEqual(rd("call 300").candidates, (SpokenMove("call"),))

    def test_raise_amount_modes(self):
        self.assertEqual(rd("raise 50 more").candidates, (SpokenMove("raise", 50, "by"),))
        self.assertEqual(rd("300까지 레이즈").candidates, (SpokenMove("raise", 300, "to"),))
        self.assertEqual(rd("raise 200").candidates, (SpokenMove("raise", 200, None),))

    def test_matcher_itself_stays_sure_for_unclear_mode(self):
        self.assertTrue(rd("raise 200").sure)

    def test_presets_with_raise_word(self):
        self.assertEqual(rd("raise half pot").candidates, (SpokenMove("raise", preset="half_pot"),))
        self.assertEqual(rd("raise pot").candidates, (SpokenMove("raise", preset="pot"),))
        self.assertEqual(rd("min raise").candidates, (SpokenMove("raise", preset="min"),))

    def test_preset_alone_phrases_imply_unsure_raise(self):
        for text, preset in (("half pot", "half_pot"), ("하프", "half_pot"), ("pot it", "pot"), ("minraise", "min")):
            with self.subTest(text=text):
                r = rd(text)
                self.assertEqual(r.candidates, (SpokenMove("raise", preset=preset),))
                self.assertFalse(r.sure)

    def test_bare_preset_words_are_not_moves(self):
        for text in ("wait a min", "pot is huge lol", "half the table is asleep", "최소한 이건 아니지", "팟", "pot"):
            with self.subTest(text=text):
                self.assertIsNone(rd(text))

    def test_preset_with_raise_word_is_sure(self):
        for text, preset in (("raise half pot", "half_pot"), ("최소 레이즈", "min"), ("raise min", "min")):
            with self.subTest(text=text):
                self.assertEqual(rd(text), Reading((SpokenMove("raise", preset=preset),), True))

    def test_preset_alone_needs_raise_to_be_legal(self):
        self.assertIsNone(rd("half pot", kinds=("call", "fold")))

    def test_amount_and_preset_both_kept(self):
        r = rd("raise to 300, nice pot")
        self.assertEqual(r.candidates, (SpokenMove("raise", 300, "to", "pot"),))

    def test_presets_match_whole_words_only(self):
        self.assertIsNone(rd("potted plant half pots"))
        self.assertEqual(rd("raise mining").candidates, (SpokenMove("raise"),))

    def test_bare_amount_is_not_a_move(self):
        self.assertIsNone(rd("300"))
        self.assertIsNone(rd("I have 300 chips left"))


if __name__ == "__main__":
    unittest.main()
