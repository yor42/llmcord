"""I Doubt It spoken reader (llmcord_core/spoken/doubt.py + lexicon/doubt.json), D36 / FEAT-38 part A."""
import json
import unittest
from pathlib import Path

from llmcord_core.spoken.doubt import read_doubt

LEX = Path(__file__).resolve().parent.parent / "llmcord_core" / "spoken" / "lexicon" / "doubt.json"


class ReadDoubtTests(unittest.TestCase):
    def check(self, want, *texts):
        for t in texts:
            with self.subTest(t=t):
                self.assertEqual(read_doubt(t), want)

    def test_english_doubt_is_sure(self):
        self.check("sure", "liar!", "I doubt it", "BS", "doubt", "bullshit", "you're lying", "youre lying", "he's lying", "she's lying", "stop lying", "all lies", "pack of lies",
                   "calling bs", "call BS on that", "that's cap", "you're capping", "stop bluffing", "so sus", "liar liar")

    def test_korean_doubt_is_sure(self):
        self.check("sure", "의심", "의심돼", "다우트!", "구라ㅋㅋ", "개구라", "개구라야", "구라치네", "뻥치네", "뻥치시네",
                   "뻥카", "뻥이야", "뻥이지", "뻥이네", "뻥치냐", "뻥쳐", "구라쳐", "구라치냐", "거짓말", "거짓말하네", "거짓말이네", "블러핑", "블러프지", "구라 ㅋㅋ", "거짓말쟁이", "거짓말쟁이야", "거짓말쟁이네", "의심스러운데", "의심스럽네", "의심스러워")

    def test_mixed_is_sure(self):
        self.check("sure", "ㅋㅋ liar", "이거 bs", "doubt ㅋㅋ", "그거 liar 맞지")

    def test_question_is_ask(self):
        self.check("ask", "liar?", "구라?", "구라？", "you're lying?", "I doubt it?", "뻥이야?", "구라치냐?", "정말 거짓말?")

    def test_negated_is_silent(self):
        self.check(None, "no doubt", "I don't doubt it", "I dont doubt it", "not lying", "never lies", "not a liar",
                   "구라 아니야", "뻥 아님", "거짓말 아냐", "구라 아닌데", "거짓말하지마", "거짓말 안 해", "구라 아니야?", "말도 안 돼 구라", "뻥치지마", "구라치지마", "구라 치지마", "구라치지 마")

    def test_excludes_are_silent(self):
        self.check(None, "without a doubt", "Without doubt, nice hand", "doubtless", "I doubt you'll win",
                   "doubt you will", "의심할 여지가 없어", "의심의 여지 없음")

    def test_table_talk_is_silent(self):
        self.check(None, "the answer lies in luck", "lying on the couch", "my hope lies with spades", "lying down for a sec",
                   "어림없어, 내가 이겨", "뻥!", "뻥 터졌네", "the card lies facebook", "nice hand", "my turn?", "cap on the table", "put your cap on", "ㅋㅋㅋ 졌다", "다음 판 가자",
                   "이거 재밌네", "good game", "who's next", "I have a seven", "pass the drinks", "susan is here",
                   "he is doubtful about the rules", "the card lies face down", "bluster", "lie detector",
                   "뻥튀기 먹고 싶다", "카드 좀 섞어줘", "내 차례야", "ㅋㅋㅋㅋ",
                   "구라파 여행 갔다 옴", "오 대박", "한 판 더 하자", "")

    def test_lexicon_shape(self):
        lex = json.loads(LEX.read_text(encoding="utf-8"))
        self.assertEqual(set(lex), {"doubt", "_exclude"})
        self.assertEqual(set(lex["doubt"]), {"en", "ko"})


if __name__ == "__main__":
    unittest.main()
