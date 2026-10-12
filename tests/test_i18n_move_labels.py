"""FEAT-46 move-label alignment (D39): a Korean move button label, typed in chat, is read as that move by the game's reader."""
import json
import unittest
from pathlib import Path

import llmcord_core
from llmcord_core.i18n import Translator
from llmcord_core.spoken import blackjack as spoken_bj
from llmcord_core.spoken import holdem, words
from llmcord_core.spoken.doubt import read_doubt

ko = Translator("ko")
HEAD = Path(llmcord_core.__file__).parent / "spoken" / "heads" / "head-e5-blackjack.json"
HOLDEM_KINDS = ("fold", "check", "call", "raise", "all_in")


def holdem_read(text):
    lexicon, negations = holdem._lexicons()
    return words.read(text, lexicon, HOLDEM_KINDS, negations)


class DoubtLabelTests(unittest.TestCase):
    def test_doubt_buttons_read_as_a_doubt(self):
        """FEAT-46 move-label tests: the table 'Doubt' and the doubt-window 'Doubt it!' labels are read as a doubt."""
        for msgid in ("Doubt", "Doubt it!"):
            with self.subTest(msgid=msgid):
                label = ko.p("button.doubt", msgid)
                self.assertTrue(label)
                self.assertIn(read_doubt(label), ("sure", "ask"), label)

    def test_other_doubt_table_buttons_are_not_read_as_a_doubt(self):
        """FEAT-46 move-label tests: the other I Doubt It buttons stay table talk for the doubt reader."""
        for msgid in ("Play", "Join", "Leave", "Fill seats", "Start", "How to play", "My hand"):
            with self.subTest(msgid=msgid):
                self.assertIsNone(read_doubt(ko.p("button.doubt", msgid)))


class HoldemLabelTests(unittest.TestCase):
    def kind_of(self, msgid, **params):
        label = ko.p("button.holdem", msgid, **params)
        reading = holdem_read(label)
        self.assertIsNotNone(reading, label)
        return reading.candidates[0].kind, label

    def test_fold_check_and_raise_labels_read_as_their_move(self):
        """FEAT-46 move-label tests: Fold, Check and Raise labels are read by the Hold'em word reader as fold, check and raise."""
        for msgid, want in (("Fold", "fold"), ("Check", "check"), ("Raise", "raise")):
            with self.subTest(msgid=msgid):
                kind, label = self.kind_of(msgid)
                self.assertEqual(kind, want, label)

    def test_call_label_with_its_amount_reads_as_call(self):
        """FEAT-46 move-label tests: 'Call {amount}' reads as call."""
        kind, label = self.kind_of("Call {amount:,}", amount=1500)
        self.assertEqual(kind, "call", label)

    def test_raise_to_label_reads_as_raise(self):
        """FEAT-46 move-label tests: 'Raise to {amount}' reads as a raise (the picker button)."""
        kind, label = self.kind_of("Raise to {amount:,}", amount=800)
        self.assertEqual(kind, "raise", label)

    def test_all_in_label_reads_as_all_in(self):
        """FEAT-46 move-label tests: 'All-in {amount}' reads as all in."""
        kind, label = self.kind_of("All-in {amount:,}", amount=2000)
        self.assertEqual(kind, "all_in", label)


class BlackjackLabelTests(unittest.TestCase):
    """Blackjack has no offline word reader: spoken/blackjack.py has only the move set, and reading goes through the ONNX
    embedder plus a trained head (never downloaded in tests). These pin the glossary word of every move button and that
    the head knows every move the buttons offer, so a label rename here is a deliberate change."""

    GLOSSARY = {"hit": "히트", "stand": "스탠드", "double": "더블다운", "surrender": "서렌더", "insure": "인슈어런스",
                "even_money": "이븐 머니", "no_insurance": "인슈어런스 안 함"}

    def label(self, move):
        msgid = spoken_bj.LABELS[move]
        if move == "insure":
            return ko.p("button.blackjack", "Insure ({cost:,})", cost=50)
        return ko.p("button.blackjack", msgid)

    def test_blackjack_buttons_use_the_glossary_words(self):
        """FEAT-46 move-label tests: each blackjack move button says the glossary word (the word players type)."""
        self.assertEqual(set(self.GLOSSARY), set(spoken_bj.MOVES))
        for move, word in self.GLOSSARY.items():
            with self.subTest(move=move):
                self.assertTrue(self.label(move).startswith(word), self.label(move))
        self.assertEqual(self.label("insure"), "인슈어런스 (50)")
        self.assertEqual(ko.p("button.blackjack", spoken_bj.LABELS["no_insurance"]), "인슈어런스 안 함")
        self.assertNotEqual(self.label("insure"), self.label("no_insurance"))

    def test_head_classes_cover_every_button_move(self):
        """FEAT-46 move-label tests: the trained blackjack head has a class for every move button (and talk)."""
        classes = set(json.loads(HEAD.read_text(encoding="utf-8"))["classes"])
        self.assertEqual(classes, set(spoken_bj.MOVES) | {"talk"})

    def test_blackjack_has_no_word_reader_without_the_model(self):
        """FEAT-46 move-label tests: spoken.blackjack carries no word lexicon, so Korean labels cannot be checked offline."""
        self.assertFalse(hasattr(spoken_bj, "read"))
        self.assertFalse((Path(llmcord_core.__file__).parent / "spoken" / "lexicon" / "blackjack.json").exists())


if __name__ == "__main__":
    unittest.main()
