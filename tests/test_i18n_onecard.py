"""MNT-89: One Card surfaces through the audience translator (lobby, board, turn, result, hand, picker, rules, refusals)."""
import itertools
import re
import unittest

import test_onecard_discord as base
from test_onecard_discord import ALICE, G
from test_doubt_discord import labels, selects

from llmcord_core.games import onecard
from llmcord_core.games_onecard_discord import _Picker, _SuitView
from llmcord_core.i18n import Translator

PSEUDO, KO = Translator("pseudo"), Translator("ko")
HANGUL = re.compile("[가-힣]")


class RulesPhraseTests(unittest.TestCase):
    def test_english_text_is_the_joined_phrases(self):
        """MNT-89: rules_text and how_to_play are the English join of their phrases, for every rules combination."""
        for defense, king, last, bankrupt, spade, ante in itertools.product([True, False], [True, False], onecard.LAST_CARD, [True, False], [True, False], [0, 5]):
            rules = dict(defense=defense, king_skips_two=king, last_card=last, bankruptcy=bankrupt, spade_ace_ten=spade, ante=ante)
            self.assertEqual(onecard.rules_text(rules), " · ".join(str(p) for p in onecard.rules_phrases(rules)))
            self.assertEqual(onecard.how_to_play(rules), "\n".join(str(p) for p in onecard.how_to_play_phrases(rules)))

    def test_pseudo_renders_every_line_and_korean_has_no_english_rule_words(self):
        """MNT-89: every rules and how-to-play phrase is translated: pseudo marks each, Korean differs from English."""
        rules = dict(last_card="no_attack_or_joker", bankruptcy=True, ante=5)
        for phrase in [*onecard.rules_phrases(rules), *onecard.how_to_play_phrases(rules)]:
            self.assertTrue(PSEUDO.render(phrase).startswith("[!!") or PSEUDO.render(phrase).count("[!!"), phrase)
            self.assertNotEqual(KO.render(phrase), str(phrase), phrase)
        text = "\n".join(KO.render(p) for p in onecard.how_to_play_phrases(rules))
        self.assertIn("원카드", text)
        self.assertIn("5", text)
        self.assertNotIn("Attacks", text)


class OneCardRenderTests(base.OneCardCase):
    async def test_lobby_board_and_turn_render_in_korean_and_pseudo(self):
        """MNT-89: the lobby, board and turn message take the channel translator; names, mentions and cards stay as they are."""
        await self.playing()
        key, snap, row = (G, self.table()["id"]), self.snap(), self.table()
        top = onecard.card_text(snap["view"]["top"])
        lobby, view = self.onecard.render_lobby(key, snap, row, t=KO)
        self.assertIn("원카드", lobby)
        self.assertIn(f"<@{ALICE}>", lobby)
        self.assertNotIn("Join with", lobby)
        self.assertEqual([c.item.label for c in view.children], [KO.p("button.onecard", x) for x in ("Join", "Leave", "Start", "How to play")])
        board = self.onecard.render_board(key, snap, t=KO)
        self.assertIn(top, board)
        self.assertIn("시계 방향", board)
        self.assertIn("7장", board)
        self.assertNotIn("clockwise", board.replace("시계 방향", ""))
        text, buttons = self.onecard.render_turn(key, snap, t=KO)
        self.assertIn(top, text)
        self.assertIn("다음", text)
        self.assertEqual([c.item.label for c in buttons.children], ["내 패"])
        for rendered in (self.onecard.render_board(key, snap, t=PSEUDO), self.onecard.render_turn(key, snap, t=PSEUDO)[0]):
            self.assertIn("[!!", rendered)
            self.assertNotIn("Top card", rendered)

    async def test_english_is_unchanged_by_default(self):
        """MNT-89: without a server language the table messages stay the English the other tests pin."""
        await self.playing()
        self.assertIn("Top card:", self.board().content)
        self.assertIn("Next:", self.turn().content)
        self.assertEqual(labels(self.turn()), ["My hand"])

    async def test_result_in_korean_names_the_winner_and_pot(self):
        """MNT-89: the settled round's thread result, lobby line and last turn line are Korean."""
        await self.playing()
        await self.play_out()
        tid = self.store.one("SELECT id FROM game_tables ORDER BY id DESC LIMIT 1")["id"]
        snap = self.store.latest_round(G, tid)
        result, lobby, last = self.onecard._result((G, tid), snap, t=KO)
        self.assertIn("팟", result)
        self.assertIn("원카드에서 이겨", lobby)
        self.assertTrue(HANGUL.search(last))
        self.assertNotIn("wins the game", result)

    async def test_hand_text_and_picker_labels_follow_the_invoker(self):
        """MNT-89: the private hand and the picker (placeholder, joker and 7 option labels, Draw button) use the invoker's translator."""
        await self.playing()
        key = (G, self.table()["id"])
        text, picker = self.onecard._hand(key, None, ALICE, t=KO)
        self.assertIn("내 패", text)
        self.assertIn("내 차례예요", text)
        self.assertNotIn("Your turn", text)
        view = _Picker(*picker, t=KO)
        self.assertEqual(selects(view)[0].placeholder, "낼 카드")
        self.assertEqual([i.label for i in view.children if hasattr(i, "label") and not hasattr(i, "options")], [KO("Draw")])
        self.assertEqual(_Picker._label("JKb", KO), "흑백 조커 (JKb)")
        self.assertEqual(_Picker._label("JKc", KO), "컬러 조커 (JKc)")
        self.assertEqual(_Picker._label("7♠", KO), "7♠ (이후 무늬 지정)")
        self.assertEqual(_Picker._label("7♠"), "7♠ (then name a suit)")
        suits = _SuitView(1, 1, 0, "7♠", t=KO)
        self.assertEqual([b.label for b in suits.children], ["♠ 스페이드", "♥ 하트", "♦ 다이아몬드", "♣ 클로버"])

    async def test_refusals_and_private_replies_use_the_invoker_language(self):
        """MNT-89: a refusal sent to one user is rendered by `_say` for that user's translator (coded errors and constants alike)."""
        from llmcord_core.games_onecard_discord import NOT_A_MOVE, ONECARD_OFF, YOU_PLAY, Phrase
        for msg in (NOT_A_MOVE, ONECARD_OFF):
            self.assertTrue(HANGUL.search(KO.render(msg)), msg)
        self.assertEqual(KO.render(Phrase(YOU_PLAY, card="7♠")), "7♠ 카드를 냈어요.")
