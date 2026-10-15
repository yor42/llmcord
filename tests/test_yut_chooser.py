"""FEAT-60 (D42): the pure side of Yut character seats' model chooser (prompt, schema, ask call). Offline; the model call is a fake.
Mirrors ``test_onecard_chooser.py``; the table-level behavior is in ``test_yut_characters.py``.
"""
import dataclasses
import unittest

from test_doubt_chooser import CARD, fake_bot
from test_games_yut_policy import rig

from llmcord_core import game_chooser
from llmcord_core.games import Seat, yut
from llmcord_core.games.playstyle import PlayStyle, describe

G, CH = 1, 100


class YutChooserCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.state = rig([["P1", "P8", None], ["P3", None, None]], ["gae", "do"], turn=0)
        self.view = yut.view(self.state, 0)
        self.legal = yut.choices(self.view, 0)
        self.names = ["Player 1", "Ann"]
        self.character = {"id": 7, "name": "Ann", "card": CARD}

    def test_the_schema_is_the_move_menu_and_a_line(self):
        schema = game_chooser.yut_schema(self.legal)
        self.assertEqual(schema["properties"]["move"]["enum"], list(self.legal))
        self.assertEqual(schema["required"], ["move", "line"])
        self.assertFalse(schema["additionalProperties"])

    def test_the_table_text_names_the_board_the_bank_and_every_legal_move(self):
        text = game_chooser.yut_table_text(self.view, self.names, self.legal)
        self.assertIn("Player 1 (you): piece 1 at P1, piece 2 at P8, piece 3 at the start", text)
        self.assertIn("Ann: piece 1 at P3", text)
        self.assertIn("Banked throws: gae, do", text)
        for move in self.legal:
            self.assertIn(f"- {move}:", text)
        self.assertIn("- move:0:gae: piece 1 at P1 with gae goes to P3, captures 1 piece", text)

    def test_a_non_move_entry_such_as_throw_is_listed_without_a_preview(self):
        text = game_chooser.yut_table_text(yut.view(yut.new("s", [Seat(0, "member", 1, 0), Seat(1, "member", 2, 0)]), 0), self.names, ("throw",))
        self.assertIn("- throw", text)

    def test_the_prompt_explains_the_diagonal_ids(self):
        self.assertIn("a1, a2, b1, b2", game_chooser.YUT_SYSTEM)

    def test_the_names_are_prompt_safe_and_members_are_numbered(self):
        snap = {"seats": [{"kind": "member", "name": None}, {"kind": "character", "name": "A*nn:\n@x"}]}
        self.assertEqual(game_chooser.yut_names(snap), ["Player 1", "A nn x"])

    async def test_one_structured_call_on_the_board(self):
        bot, calls = fake_bot({"move": "move:0:gae", "line": "ha"})
        result = await game_chooser.ask_yut(bot, G, CH, self.character, self.view, self.names, self.legal)
        self.assertEqual(result, {"move": "move:0:gae", "line": "ha"})
        (role, request, name, schema, feature, purpose), = calls
        self.assertEqual((role, name, feature, purpose), ("director", "yut_move", "game", "game"))
        self.assertEqual(schema, game_chooser.yut_schema(self.legal))
        self.assertIn("You are Ann", request.messages[0].text)
        self.assertIn("cheerful gambler named Ann", request.messages[0].text)
        self.assertNotIn("Play style", request.messages[0].text)
        self.assertEqual(request.messages[-1].text, game_chooser.yut_table_text(self.view, self.names, self.legal))

    async def test_a_saved_style_is_told_in_words(self):
        bot, calls = fake_bot({"move": "move:0:gae", "line": ""})
        style = PlayStyle(aggression=95, risk=90)
        await game_chooser.ask_yut(bot, G, CH, self.character, self.view, self.names, self.legal, style)
        self.assertIn(f"Play style: {describe(style)}.", calls[0][1].messages[0].text)

    async def test_a_failing_model_raises_for_the_caller_to_fall_back(self):
        bot, _ = fake_bot(None)

        async def boom(*args):
            raise RuntimeError("down")
        bot.models.structured_compiled = boom
        with self.assertRaises(RuntimeError):
            await game_chooser.ask_yut(bot, G, CH, self.character, self.view, self.names, self.legal)

    def test_the_seed_never_reaches_the_prompt(self):
        state = dataclasses.replace(self.state, seed="SECRET-SEED-VALUE")
        text = game_chooser.yut_table_text(yut.view(state, 0), self.names, self.legal)
        self.assertNotIn("SECRET-SEED-VALUE", text)


if __name__ == "__main__":
    unittest.main()
