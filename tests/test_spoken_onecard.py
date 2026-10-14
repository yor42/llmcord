"""One Card spoken reader (llmcord_core/spoken/onecard.py + lexicon/onecard.json), D39 / FEAT-45 step 1.

Assumed API, mirroring spoken/doubt.py (pure, word list only, no model; only "sure" acts):

    read_onecard(text, seats=(), reply_to=None, phrases=None) -> OneCardReading

    seats     sequence of (seat_index, names) where names is a sequence of display names / aliases
    reply_to  seat index of the message being replied to, or None
    phrases   declaration phrase list (server setting); None means DEFAULT_PHRASES = ("원카드", "one card").
              A given list replaces the default; matching uses the same variants (spacing, hyphen,
              case, NFKC) as the defaults.

    OneCardReading is a frozen dataclass with
        kind         "declaration" | "callout" | "none"
        outcome      "sure" | "ask" | "none"
        target_kind  "name" | "reply" | "none"   (callouts only; "none" otherwise)
        target_seat  matched seat index or None

    A "none" reading is kind "none", outcome "none". Questions read as outcome "ask" with the kind they
    would have had. The reader never guesses: two different seats named (or a name that disagrees with the
    reply) gives a callout with outcome "ask" and no target. The caller decides "implicit" (exactly one
    seat exposed) when target_kind is "none". Only messages of at most 40 characters (after NFKC and
    strip) act. A bare declaration phrase sent as a reply to someone is "ask" (declaring or calling out?).
    DEFAULT_PHRASES and MAX_LEN are module constants.
"""
import json
import unittest
from pathlib import Path

from llmcord_core.spoken.onecard import DEFAULT_PHRASES, MAX_LEN, read_onecard

LEX = Path(__file__).resolve().parent.parent / "llmcord_core" / "spoken" / "lexicon" / "onecard.json"
SEATS = [(0, ("민수", "minsu")), (1, ("지영", "Jiyoung")), (2, ("철수",))]


class OneCardReaderTests(unittest.TestCase):
    def check(self, want, *texts, **kw):
        """want = (kind, outcome, target_kind, target_seat)."""
        for t in texts:
            with self.subTest(t=t):
                r = read_onecard(t, **kw)
                self.assertEqual((r.kind, r.outcome, r.target_kind, r.target_seat), want)

    def silent(self, *texts, **kw):
        self.check(("none", "none", "none", None), *texts, **kw)

    def test_constants(self):
        self.assertEqual(tuple(DEFAULT_PHRASES), ("원카드", "one card"))
        self.assertEqual(MAX_LEN, 40)

    def test_declaration_is_sure(self):
        self.check(("declaration", "sure", "none", None), "원카드", "원카드!", "원카드!!!", "원카드~", "원 카드", "원카드ㅋㅋ", "원카드 ㅇㅇ",
                   "one card", "One Card!", "ONE CARD", "onecard", "one-card", "one   card", "  one card  ", "one card 🔥",
                   "원카드 원카드", "one card one card", "원카드!원카드!", "ㅋㅋ 원카드", "one card 외침", "원카드 called", "원카드.", "원카드…")

    def test_declaration_with_seats_configured_is_unchanged(self):
        self.check(("declaration", "sure", "none", None), "원카드!", "one card", seats=SEATS)

    def test_nfkc_and_fullwidth_forms(self):
        self.check(("declaration", "sure", "none", None), "ＯＮＥ　ＣＡＲＤ", "ｏｎｅ ｃａｒｄ！", "원카드！", "ＯＮＥ－ＣＡＲＤ")

    def test_callout_without_target(self):
        self.check(("callout", "sure", "none", None), "원카드 안 했다", "원카드 안했어", "원카드 안 외쳤어", "원카드 안 했잖아", "원카드 안했네",
                   "원카드 왜 안 해", "원카드 안 불렀다", "didn't say one card", "didnt say one card", "you forgot to say one card",
                   "forgot one card", "he didn't declare", "call out", "callout!", "no one card!", "didn't say one-card")

    def test_callout_by_name(self):
        self.check(("callout", "sure", "name", 0), "민수 원카드 안 했어", "민수가 원카드 안 했다", "민수는 원카드 안했네", "민수 원카드!", "민수가 원카드!",
                   "민수는 원카드!", "민수도 원카드!", "Minsu didn't say one card", "MINSU didn't declare", "minsu one card!", "원카드 안 했다 민수",
                   "민수 didn't declare", "ＭＩＮＳＵ didn't declare", seats=SEATS)
        self.check(("callout", "sure", "name", 1), "지영이 원카드 안 했어", "지영아 원카드!", "jiyoung forgot to say one card", "지영 원카드 안 했다", "지영 원카드!", seats=SEATS)
        self.check(("callout", "sure", "name", 2), "철수가 원카드 안 했어", "철수는 원카드!", "철수 원카드!", seats=SEATS)

    def test_callout_by_name_needs_seats(self):
        self.check(("callout", "sure", "name", 0), "민수가 원카드 안 했어", seats=SEATS)
        self.check(("callout", "sure", "none", None), "민수가 원카드 안 했어", seats=[(5, ("영희",))])

    def test_callout_names_do_not_match_inside_longer_words(self):
        self.check(("callout", "sure", "none", None), "minsuk didn't say one card", seats=SEATS)

    def test_callout_by_reply(self):
        self.check(("callout", "sure", "reply", 1), "원카드 안 했어", "didn't say one card", "call out", "forgot to say one card", seats=SEATS, reply_to=1)
        self.check(("callout", "sure", "name", 1), "지영 원카드 안 했어", seats=SEATS, reply_to=1)

    def test_bare_declaration_as_a_reply_is_ask(self):
        self.check(("declaration", "ask", "none", None), "원카드!", "one card", seats=SEATS, reply_to=0)

    def test_two_names_is_ask_with_no_target(self):
        self.check(("callout", "ask", "none", None), "민수 지영 원카드 안 했어", "minsu and jiyoung didn't say one card", "민수랑 지영이 원카드 안 했어", seats=SEATS)

    def test_name_and_reply_disagree_is_ask(self):
        self.check(("callout", "ask", "none", None), "민수 원카드 안 했어", seats=SEATS, reply_to=1)

    def test_negated_declaration_is_silent(self):
        self.silent("I won't say one card", "not one card yet", "원카드 안 해", "원카드 안 할래", "원카드 안 할 거야", "원카드 아직 아니야", "원카드 아님",
                    "원카드 하지마", "원카드 말아야지", "don't say one card", "no I wont say one card", "not saying one card", "never one card", "원카드 못 해")

    def test_questions_are_ask_never_sure(self):
        self.check(("declaration", "ask", "none", None), "do I say one card now?", "원카드 해야 해?", "원카드?", "원카드？", "one card?", "원카드 지금?", "should I say one card?")
        self.check(("callout", "ask", "none", None), "원카드 안 했어?", "didn't he say one card?")
        self.check(("callout", "ask", "name", 0), "민수 원카드 안 했어?", seats=SEATS)

    def test_quoted_and_reported_speech_is_silent(self):
        self.silent('he said "one card"', '"원카드"', "'one card'", "“원카드”", "「원카드」", "민수가 '원카드' 라고 했어", '민수가 "원카드" 라고 했다',
                    "he said one card", "민수가 원카드라고 했어", "> one card", "> 원카드 안 했어", "`one card`", "`원카드` 입력하면 돼", "```one card```", "```\n원카드\n```",
                    "x said: \"didn't say one card\"", seats=SEATS)

    def test_unquoted_after_quote_marks_closed_is_still_read(self):
        self.check(("declaration", "sure", "none", None), '"nice" 원카드')

    def test_mentions_of_the_game_are_silent(self):
        self.silent("this one card game is fun", "원카드 게임 재밌다", "원카드 규칙 알려줘", "one card left in the deck", "one card at a time", "I like one card",
                    "나 원카드 좋아해", "what is one card", "play one card please", "한 장 남았다", "원 카드게임", "onecard game", "누가 원카드 만들었어")

    def test_table_talk_is_silent(self):
        self.silent("", "   ", "내 차례야", "good game", "nice play", "ㅋㅋㅋ", "who's next", "draw", "카드 좀 섞어줘", "one", "card", "원", "카드", "I have one", "call me later", "민수 차례야", seats=SEATS)

    def test_length_cap(self):
        self.assertEqual(read_onecard("원카드" + "!" * (MAX_LEN - 3)).outcome, "sure")
        self.silent("원카드" + "!" * (MAX_LEN - 2))
        self.silent("one card " + "x" * MAX_LEN, "원카드 " + "ㅋ" * 50, "민수가 원카드 안 했다고 말하고 싶은데 사실은 그냥 길게 이야기하고 싶어서 쓰는 긴 문장입니다", seats=SEATS)

    def test_custom_phrase_list_replaces_default(self):
        phrases = ["라스트카드", "uno"]
        self.check(("declaration", "sure", "none", None), "라스트카드", "라스트 카드!", "UNO", "Uno!", "uno.", phrases=phrases)
        self.silent("원카드", "one card", "unorthodox", "tuno", phrases=phrases)
        self.check(("declaration", "ask", "none", None), "uno?", phrases=phrases)

    def test_custom_phrase_callouts(self):
        self.check(("callout", "sure", "name", 0), "민수 uno!", "민수가 uno 안 했어", seats=SEATS, phrases=["uno"])
        self.check(("callout", "sure", "none", None), "didn't say uno", phrases=["uno"])

    def test_explicit_default_phrases_match_none_argument(self):
        for t in ("원카드", "one card", "원카드 안 했어", "원카드?"):
            with self.subTest(t=t):
                self.assertEqual(read_onecard(t), read_onecard(t, phrases=list(DEFAULT_PHRASES)))

    def test_reading_is_a_frozen_value(self):
        r = read_onecard("원카드")
        self.assertEqual(r, read_onecard("원카드"))
        with self.assertRaises(Exception):
            r.kind = "callout"

    def test_lexicon_shape(self):
        lex = json.loads(LEX.read_text(encoding="utf-8"))
        self.assertTrue({"callout"} <= set(lex))
        self.assertEqual(set(lex["callout"]), {"en", "ko"})


if __name__ == "__main__":
    unittest.main()


class SeatNameAndPhraseTests(unittest.TestCase):
    def test_a_seat_named_like_the_phrase_does_not_turn_a_declaration_into_a_callout(self):
        """FEAT-45: seat names are matched outside the declaration phrase ("One", "Card" are names too)."""
        for name in ("One", "Card", "one card"):
            with self.subTest(name=name):
                for text in ("one card", "One card!", "원카드"):
                    got = read_onecard(text, [(1, (name,))])
                    self.assertEqual((got.kind, got.outcome), ("declaration", "sure"), text)

    def test_a_custom_phrase_equal_to_a_name_is_still_a_declaration(self):
        """FEAT-45: a server phrase "Uno" with a seat named Uno."""
        got = read_onecard("Uno", [(1, ("Uno",))], phrases=["Uno"])
        self.assertEqual((got.kind, got.outcome), ("declaration", "sure"))

    def test_a_name_beside_the_phrase_still_names_the_seat(self):
        got = read_onecard("One forgot one card", [(1, ("One",))])
        self.assertEqual((got.kind, got.outcome, got.target_seat), ("callout", "sure", 1))

    def test_phrases_with_regex_characters_match_literally(self):
        """FEAT-45: a phrase like "a.b" or "(x" cannot crash the reader and does not act as a pattern."""
        for phrase, yes, no in (("a.b", "a.b", "axb"), ("(x", "(x", "x")):
            with self.subTest(phrase=phrase):
                self.assertEqual(read_onecard(yes, phrases=[phrase]).kind, "declaration")
                self.assertEqual(read_onecard(no, phrases=[phrase]).kind, "none")
