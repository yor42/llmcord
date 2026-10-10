"""FEAT-28 part B: blackjack rule options on Discord (insurance, even money and surrender buttons, rules line, how to play, character seats)."""
from unittest import mock

import test_game_character_seats_discord as tcs
import test_games_discord as tgd
from test_games_discord import ALICE, BOB, CARA, CH, G, NOT_YOUR_TURN, STALE

from llmcord_core import game_chooser
from llmcord_core.games import blackjack as bj
from llmcord_core.games_discord import GameButton, GameTables, _outcome

INS = {"insurance": True}
SUR = {"surrender": True}
ALL = {"insurance": True, "surrender": True, "blackjack_pays": "6:5", "ties": "dealer", "stand_on": 18}


class ButtonTests(tgd.GameCase):
    async def deal(self, rules, *ranks, users=(ALICE,), bet=10):
        self.store.set_blackjack_rules(G, rules)
        self.rig(*ranks)
        for user in users:
            await self.bet(user, bet)
        await self.press(ALICE, "Deal now")
        return self.latest()

    def moves(self):
        return [tuple(r) for r in self.store.db.execute("SELECT seat_index,move,actor FROM game_moves ORDER BY id")]

    async def test_insurance_buttons_exist_only_in_the_insurance_phase(self):
        snap = await self.deal(INS, "10", "A", "7", "5")
        self.assertEqual(snap["phase"], "insurance")
        self.assertEqual(self.labels(), ["Insure (5)", "No insurance", "How to play"])
        self.assertIn("Dealer shows an ace. Insurance?", self.games.render(snap))
        click = await self.press(ALICE, "Insure (5)")
        self.assertEqual(self.moves(), [(0, "insure", "member")])
        self.assertEqual(self.balance(ALICE), 85)
        self.assertEqual(self.labels(), ["Hit", "Stand", "Double", "How to play"])
        text = click.response.edited[0]["content"]
        self.assertNotIn("Insurance?", text)
        self.assertIn("(insured 5)", text)

    async def test_only_the_seat_on_turn_can_answer(self):
        snap = await self.deal(INS, "10", "10", "A", "7", "8", "5", users=(ALICE, BOB))
        self.assertEqual(snap["legal"], {0: ("insure", "no_insurance")})
        click = await self.press(BOB, "Insure (5)", snap)
        self.assertEqual(click.followup.sent[0][0], NOT_YOUR_TURN)
        self.assertEqual(self.moves(), [])
        await self.press(ALICE, "No insurance", snap)
        self.assertEqual(self.moves(), [(0, "no_insurance", "member")])
        self.assertEqual(self.latest()["legal"], {1: ("insure", "no_insurance")})

    async def test_the_old_buttons_are_stale_after_an_answer(self):
        snap = await self.deal(INS, "10", "A", "7", "5")
        await self.press(ALICE, "No insurance", snap)
        click = await self.press(ALICE, "Insure (5)", snap)
        self.assertEqual(click.followup.sent[0][0], STALE)
        self.assertEqual(self.balance(ALICE), 100 - 10)

    async def test_even_money_is_offered_to_a_blackjack_seat_only(self):
        snap = await self.deal(INS, "A", "A", "K", "5")
        self.assertEqual(self.labels(), ["Even money", "No insurance", "How to play"])
        click = await self.press(ALICE, "Even money", snap)
        self.assertEqual(self.moves(), [(0, "even_money", "member")])
        self.assertEqual(self.balance(ALICE), 110)
        text = click.response.edited[0]["content"]
        self.assertIn("even money +10", text)
        self.assertEqual(self.latest()["status"], "settled")

    async def test_the_insured_result_is_shown_apart_from_the_hand(self):
        await self.deal(INS, "10", "A", "7", "K")
        click = await self.press(ALICE, "Insure (5)")
        self.assertIn("lose −10, insurance +10", click.response.edited[0]["content"])
        self.assertEqual(self.balance(ALICE), 100 - 10 - 5 + 15)

    async def test_a_missed_insurance_bet_is_shown_as_lost(self):
        await self.deal(INS, "10", "A", "6", "5")
        await self.press(ALICE, "Insure (5)")
        click = await self.press(ALICE, "Stand")
        self.assertIn("lose −10, insurance −5", click.response.edited[0]["content"])

    async def test_a_declined_insurance_timeout_is_not_marked_timed_out(self):
        snap = await self.deal(INS, "10", "A", "6", "5")
        await self.games.on_turn_timer(G, snap["table_id"], snap["round_id"], 0)
        await self.press(ALICE, "Hit")
        click = await self.press(ALICE, "Stand")
        text = click.response.edited[0]["content"]
        self.assertEqual(self.moves()[0], (0, "no_insurance", "timeout"))
        self.assertNotIn("timed out", text)

    async def test_insure_is_hidden_when_the_player_cannot_pay(self):
        self.store.change_balance(G, ALICE, -90, "spend", 1)
        await self.deal(INS, "10", "A", "7", "5", bet=10)
        self.assertEqual(self.balance(ALICE), 0)
        self.assertEqual(self.labels(), ["No insurance", "How to play"])

    async def test_the_even_money_marker_shows_during_play_with_several_seats(self):
        snap = await self.deal(INS, "A", "10", "A", "K", "7", "5", users=(ALICE, BOB))
        self.assertEqual(snap["legal"], {0: ("even_money", "no_insurance")})
        await self.press(ALICE, "Even money", snap)
        click = await self.press(BOB, "No insurance")
        text = click.response.edited[0]["content"]
        self.assertIn("(even money)", text)
        self.assertEqual(self.latest()["status"], "playing")

    async def test_closing_in_the_insurance_phase_repaints_and_refunds_stake_and_insurance(self):
        snap = await self.deal(INS, "10", "10", "A", "7", "8", "5", users=(ALICE, BOB))
        await self.press(ALICE, "Insure (5)", snap)
        self.assertEqual(self.balance(ALICE), 85)
        self.store.close_table(G, snap["table_id"])
        await self.games.sweep()
        self.assertIn("Table closed", self.texts()[-1])
        self.assertEqual((self.balance(ALICE), self.balance(BOB)), (100, 100))

    async def test_cancelling_in_the_play_phase_refunds_stake_and_insurance(self):
        snap = await self.deal(INS, "10", "A", "7", "5")
        await self.press(ALICE, "Insure (5)", snap)
        self.assertEqual(self.balance(ALICE), 85)
        snap = self.latest()
        cancelled = self.store.cancel_round(G, snap["round_id"], "test")
        self.assertEqual(self.balance(ALICE), 100)
        self.assertIn("Round cancelled", self.games.render(cancelled))
        self.assertEqual(self.labels(cancelled), ["Play again (same bet)", "Leave table", "How to play"])

    async def test_the_outcome_words(self):
        seat = {"outcome": "surrender", "payout": 5, "stake": 10, "insurance": 0}
        self.assertEqual(_outcome(seat), "surrendered −5")
        self.assertEqual(_outcome({**seat, "outcome": "even_money", "payout": 20}), "even money +10")
        self.assertEqual(_outcome({"outcome": "push", "payout": 10, "stake": 10, "insurance": 0}), "push")
        self.assertEqual(_outcome({"outcome": "push", "payout": 25, "stake": 10, "insurance": 5}, True), "push, insurance +10")

    async def test_surrender_is_a_button_only_when_the_rule_is_on(self):
        snap = await self.deal({}, "10", "9", "7", "7")
        self.assertEqual(self.labels(), ["Hit", "Stand", "Double", "How to play"])
        self.store.cancel_round(G, snap["round_id"], "x")

    async def test_surrender_press_plays_the_move(self):
        await self.deal(SUR, "10", "9", "7", "7")
        self.assertEqual(self.labels(), ["Hit", "Stand", "Double", "Surrender", "How to play"])
        click = await self.press(ALICE, "Surrender")
        self.assertEqual(self.moves(), [(0, "surrender", "member")])
        self.assertIn("surrendered −5", click.response.edited[0]["content"])
        self.assertEqual(self.balance(ALICE), 95)

    async def test_no_surrender_after_a_hit(self):
        await self.deal(SUR, "10", "9", "3", "7")
        await self.press(ALICE, "Hit")
        self.assertNotIn("Surrender", self.labels())

    async def test_the_turn_timer_in_the_insurance_phase_picks_no_insurance(self):
        snap = await self.deal(INS, "10", "A", "7", "5")
        await self.games.on_turn_timer(G, snap["table_id"], snap["round_id"], 0)
        self.assertEqual(self.moves(), [(0, "no_insurance", "timeout")])

    async def test_the_custom_ids_match_the_template_with_underscores(self):
        await self.deal(INS, "A", "A", "K", "5")
        ids = [c.item.custom_id for c in self.games.buttons(self.latest()).children]
        self.assertTrue(all(GameButton.__discord_ui_compiled_template__.fullmatch(i) for i in ids), ids)
        self.assertIn("even_money", ids[0])


class RulesTextTests(tgd.GameCase):
    async def test_the_rules_line_follows_the_round_not_the_guild(self):
        self.store.set_blackjack_rules(G, ALL)
        self.rig("10", "9", "7", "7")
        await self.bet(ALICE, 10)
        line = "-# " + bj.rules_text(ALL)
        self.assertIn(line, self.games.render(self.latest()))
        self.store.set_blackjack_rules(G, {})
        self.assertIn(line, self.games.render(self.latest()))
        self.assertIn("Blackjack pays 6:5", line)

    async def test_how_to_play_uses_the_round_rules_and_mentions_options_only_when_on(self):
        self.store.set_blackjack_rules(G, ALL)
        await self.bet(ALICE, 10)
        snap = self.latest()
        self.store.set_blackjack_rules(G, {})
        click = tgd.Click(self.bot, BOB)
        await GameButton("help", snap["table_id"], snap["round_id"], 0).callback(click)
        text = click.response.sent[0][0]
        self.assertEqual(text, GameTables.how_to_play(ALL))
        for word in ("Insurance", "Surrender", "6:5", "18"):
            self.assertIn(word, text)
        plain = GameTables.how_to_play()
        self.assertNotIn("Insurance", plain)
        self.assertNotIn("Surrender", plain)

    async def test_a_table_without_rules_uses_the_servers_current_rules(self):
        self.store.set_blackjack_rules(G, INS)
        self.assertEqual(self.games._help_text(G, 999, 999), GameTables.how_to_play(INS))
        with mock.patch.object(self.store, "round_snapshot", return_value={"table_id": 4, "rules": None}):
            self.assertEqual(self.games._help_text(G, 4, 1), GameTables.how_to_play(INS))
        self.assertEqual(self.games._help_text(None, 1, 1), GameTables.how_to_play())

    async def test_seven_insured_seats_fit_a_message(self):
        users = [ALICE, BOB, CARA, 8, 9, 10, 11]
        for user in users:
            self.store.change_balance(G, user, 2000, "start", 1)
        self.store.set_blackjack_rules(G, ALL)
        self.rig(*(("10",) * 7 + ("A",) + ("10",) * 7 + ("5",)))
        for user in users:
            await self.bet(user, 1000)
        await self.press(ALICE, "Deal now")
        for user in users:
            await self.press(user, "Insure (500)")
            self.assertLessEqual(len(self.games.render(self.latest())), 2000)
        self.assertEqual(self.latest()["status"], "playing")
        for user in users:
            await self.press(user, "Surrender")
        snap = self.latest()
        self.assertEqual(snap["status"], "settled")
        text = self.games.render(snap)
        self.assertLessEqual(len(text), 2000)
        self.assertIn("surrendered −500, insurance −500", text)


class CharacterTests(tcs.SeatCase):
    async def start_insurance(self, model=None):
        self.store.set_blackjack_rules(G, {"insurance": True, "surrender": True})
        self.rig("10", "10", "A", "7", "8", "5")
        await self.bet_fav(ALICE, 10)
        await self.games.on_join_timer(G, self.store.open_table_for(G, CH)["id"], self.latest()["round_id"])
        await self.press(ALICE, "No insurance")

    async def test_talk_off_declines_insurance(self):
        self.store.set_game_character_talk(G, False)
        await self.start_insurance()
        self.assertEqual(self.latest()["legal"], {1: ("insure", "no_insurance")})
        await self.fire()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.actors()[-1], (1, "no_insurance", "character"))
        self.assertEqual(self.store.character_balance(G, self.ann), 90)

    async def test_the_chooser_offers_insurance_and_the_model_may_take_it(self):
        self.bot.models.structured_compiled = self.fake_model({"move": "insure", "line": "Just in case."})
        await self.start_insurance()
        await self.fire()
        _, request, _, schema, _ = self.calls[0]
        self.assertEqual(schema["properties"]["move"]["enum"], ["insure", "no_insurance"])
        text = "\n".join(m.text for m in request.messages)
        self.assertIn("insure: put up half your bet", text)
        self.assertIn("no_insurance: decline", text)
        self.assertNotIn("surrender:", text)
        self.assertIn("Dealer shows an ace", text.replace("The dealer", "Dealer"))
        self.assertEqual(self.actors()[-1], (1, "insure", "character"))
        self.assertEqual(self.store.character_balance(G, self.ann), 85)

    async def test_an_illegal_answer_falls_back_to_the_house_move(self):
        self.bot.models.structured_compiled = self.fake_model({"move": "surrender", "line": "hm"})
        await self.start_insurance()
        await self.fire()
        self.assertEqual(self.actors()[-1], (1, "no_insurance", "character"))
        self.assertEqual(self.webhook.sent, [])

    async def test_surrender_is_explained_only_when_legal_and_the_house_never_takes_it(self):
        self.bot.models.structured_compiled = self.fake_model({"move": "stand", "line": ""})
        await self.start_insurance()
        await self.fire()
        await self.press(ALICE, "Stand")
        await self.fire()
        text = "\n".join(m.text for m in self.calls[-1][1].messages)
        self.assertIn("surrender: give up the hand", text)
        self.assertNotIn("insure:", text)
        self.store.set_game_character_talk(G, False)
        self.assertEqual(self.games._house_move(self.latest(), 1, ("hit", "stand", "double", "surrender")), "stand")

    def test_the_prompt_text_lists_help_lines_only_for_legal_moves(self):
        snap = {"view": {"hands": ((0, 9), (1, 2)), "dealer": (0,), "status": ("playing", "playing")}, "seats": [{"kind": "member"}, {"kind": "member"}], "phase": None}
        text = game_chooser.table_text(snap, 0, ("hit", "stand", "double"))
        self.assertNotIn("surrender:", text)
        self.assertNotIn("insure", text)
        self.assertIn("surrender: give up", game_chooser.table_text(snap, 0, ("hit", "stand", "surrender")))
