"""UI-53..56: blackjack table polish. End-of-table summary, dealer pacing, How to play, rules line. Offline, fake Discord."""
import unittest
from unittest import mock

import discord
from helpers import FakeInteraction
from test_game_character_seats_discord import SeatCase
from test_games_discord import ALICE, BOB, CARA, CH, G, GameCase, Click

from llmcord_core import games_discord
from llmcord_core.admin_store import GameError
from llmcord_core.games import blackjack as bj
from llmcord_core.games_discord import GameButton

DEFAULT_RULES = "Dealer stands on all 17s · Blackjack pays 3:2 · Ties push · Double on the first two cards · No splitting · 6 decks"
WIN = ("10", "9", "9", "7")  # one seat: player 19, dealer 16 draws a 2 to 18
MORE = ("10", "10", "9", "10", "9", "10")  # two seats: 20 and 19 against a dealer 19


class RulesTextTests(unittest.TestCase):
    def test_default_rules_line_is_exact(self):
        self.assertEqual(bj.rules_text(), DEFAULT_RULES)
        self.assertEqual(bj.rules_text({}), DEFAULT_RULES)

    def test_rules_dict_changes_the_words(self):
        text = bj.rules_text({"dealer_hits_soft_17": True, "blackjack_pays": "6:5", "double": False, "split": True, "decks": 1})
        self.assertEqual(text, "Dealer hits soft 17 · Blackjack pays 6:5 · Ties push · No doubling · Splitting allowed · 1 deck")

    def test_how_to_play_is_short_and_follows_the_rules(self):
        text = bj.how_to_play()
        self.assertLessEqual(len(text) + len(games_discord.HOW_TO_JOIN) + 1, 900)
        self.assertTrue(text.startswith("**Blackjack in short**"))
        self.assertIn("pays 3:2", text)
        self.assertIn("draws until 17 or more", text)
        other = bj.how_to_play({"blackjack_pays": "6:5", "double": False})
        self.assertIn("pays 6:5", other)
        self.assertNotIn("Double doubles", other)


class RulesLineTests(GameCase):
    async def test_every_table_state_shows_the_rules_above_the_seed_line(self):
        self.rig(*WIN)
        await self.bet(ALICE, 10)
        states = [self.games.render(self.latest())]
        await self.press(ALICE, "Deal now")
        states.append(self.games.render(self.latest()))
        await self.press(ALICE, "Stand")
        states.append(self.games.render(self.latest()))
        for text in states:
            lines = text.split("\n")
            self.assertEqual(lines[-2], "-# " + DEFAULT_RULES)
            self.assertTrue(lines[-1].startswith("-# Seed"))

    async def test_a_cancelled_round_shows_it_too(self):
        await self.bet(ALICE, 10)
        snap = self.store.cancel_round(G, self.latest()["round_id"], "test")
        self.assertIn("-# " + DEFAULT_RULES + "\n-# Seed:", self.games.render(snap))


class HowToPlayTests(GameCase):
    async def test_the_button_is_on_joining_playing_and_settled_views(self):
        self.rig(*WIN)
        await self.bet(ALICE, 10)
        self.assertEqual(self.labels(), ["Deal now", "Leave", "How to play"])
        await self.press(ALICE, "Deal now")
        self.assertEqual(self.labels(), ["Hit", "Stand", "Double", "How to play"])
        await self.press(ALICE, "Stand")
        self.assertEqual(self.labels(), ["Play again (same bet)", "Leave table", "How to play"])
        button = self.games.buttons(self.latest()).children[-1].item
        self.assertEqual(button.style, discord.ButtonStyle.secondary)
        self.assertRegex(button.custom_id, r"^llmcord:bj:help:\d+:\d+:\d+$")
        self.assertTrue(GameButton.__discord_ui_compiled_template__.fullmatch(button.custom_id))

    async def test_a_cancelled_round_has_it_too(self):
        await self.bet(ALICE, 10)
        snap = self.store.cancel_round(G, self.latest()["round_id"], "test")
        self.assertIn("How to play", self.labels(snap))

    async def test_anyone_gets_an_ephemeral_guide_and_nothing_changes(self):
        self.rig(*WIN)
        await self.bet(ALICE, 10)
        before = self.latest()
        click = await self.press(CARA, "How to play")
        text, kwargs = click.response.sent[0]
        self.assertEqual(text, games_discord.GameTables.how_to_play())
        self.assertTrue(kwargs["ephemeral"])
        self.assertEqual(kwargs["allowed_mentions"].to_dict(), discord.AllowedMentions.none().to_dict())
        self.assertLessEqual(len(text), 900)
        self.assertIn("/blackjack bet:<amount>", text)
        self.assertEqual(self.latest(), before)
        self.assertEqual(self.balance(CARA), 100)

    async def test_it_works_on_an_old_message_of_a_closed_table(self):
        await self.bet(ALICE, 10)
        snap = self.latest()
        await self.press(ALICE, "Leave")
        self.assertIsNone(self.store.open_table_for(G, CH))
        click = Click(self.bot, BOB)
        await GameButton("help", snap["table_id"], snap["round_id"], 0).callback(click)
        self.assertEqual(click.replies, [games_discord.GameTables.how_to_play()])
        outside = FakeInteraction(guild_id=None)
        outside.client = self.bot
        await GameButton("help", 1, 1, 0).callback(outside)
        self.assertEqual(outside.replies, [games_discord.GameTables.how_to_play()])


class SummaryTests(GameCase):
    async def play(self, rig=WIN):
        """Alice joins (or plays again) and wins 10 against 18."""
        self.rig(*rig)
        if self.store.open_table_for(G, CH) is None:
            await self.bet(ALICE, 10)
        else:
            await self.press(ALICE, "Play again (same bet)")
        await self.press(ALICE, "Deal now")
        await self.press(ALICE, "Stand")

    async def test_one_round(self):
        await self.play()
        click = await self.press(ALICE, "Leave table")
        self.assertEqual(click.response.edited[0]["content"], f"Table closed.\nRound 1: dealer 18 — <@{ALICE}> +10")
        self.assertIsNone(click.response.edited[0]["view"])

    async def test_seven_rounds_show_the_last_five_with_totals(self):
        for _ in range(7):
            await self.play()
        click = await self.press(ALICE, "Leave table")
        lines = click.response.edited[0]["content"].split("\n")
        self.assertEqual(lines[0], "Table closed.")
        self.assertEqual(lines[1:6], [f"Round {n}: dealer 18 — <@{ALICE}> +10" for n in range(3, 8)])
        self.assertEqual(lines[6:], [f"Net: <@{ALICE}> +50"])

    async def test_win_and_push_and_a_total_per_player(self):
        self.rig(*MORE)  # replays use the shoe of the moment, so both rounds share it
        await self.bet(ALICE, 10)
        await self.bet(BOB, 10)
        for again in (False, True):
            if again:
                await self.press(ALICE, "Play again (same bet)", self.settled)
                await self.press(BOB, "Play again (same bet)", self.settled)
            await self.press(ALICE, "Deal now")
            await self.press(ALICE, "Stand")
            await self.press(BOB, "Stand")
            self.settled = self.latest()
        await self.press(ALICE, "Leave table")
        click = await self.press(BOB, "Leave table")
        self.assertEqual(click.response.edited[0]["content"], "\n".join([
            "Table closed.", f"Round 1: dealer 19 — <@{ALICE}> +10, <@{BOB}> push", f"Round 2: dealer 19 — <@{ALICE}> +10, <@{BOB}> push",
            f"Net: <@{ALICE}> +20, <@{BOB}> even"]))

    async def test_a_cancelled_round_is_skipped(self):
        await self.play()
        await self.press(ALICE, "Play again (same bet)")
        click = await self.press(ALICE, "Leave")  # the only seat leaves the joining round: table closes, round 2 is cancelled
        self.assertEqual(click.response.edited[0]["content"], f"Table closed.\nRound 1: dealer 18 — <@{ALICE}> +10")

    async def test_no_settled_round_is_just_the_closing_text(self):
        await self.bet(ALICE, 10)
        click = await self.press(ALICE, "Leave")
        self.assertEqual(click.response.edited[0]["content"], "Table closed.")

    async def test_the_idle_timer_and_channel_off_paths_carry_it(self):
        await self.play()
        snap = self.latest()
        await self.games.on_idle_timer(G, snap["table_id"], snap["round_id"])
        self.assertEqual(self.texts()[-1], f"Table closed.\nRound 1: dealer 18 — <@{ALICE}> +10")
        self.assertEqual(self.channel.edits[-1][1]["allowed_mentions"].to_dict(), discord.AllowedMentions.none().to_dict())
        await self.play()
        await self.games.set_channel(G, CH, False)
        self.assertEqual(self.texts()[-1], f"Table closed: games were turned off here.\nRound 1: dealer 18 — <@{ALICE}> +10")

    async def test_a_missing_dealer_total_leaves_the_round_line_without_it(self):
        await self.play()
        with mock.patch.object(self.store, "_game_state", side_effect=GameError("damaged")):
            self.assertIsNone(self.store.game_table_summary(G, self.latest()["table_id"], 5)[0]["dealer_total"])
            click = await self.press(ALICE, "Leave table")
        self.assertEqual(click.response.edited[0]["content"], f"Table closed.\nRound 1: <@{ALICE}> +10")

    async def test_the_store_read_is_guild_scoped_and_oldest_first(self):
        await self.play()
        await self.play()
        table_id = self.latest()["table_id"]
        rounds = self.store.game_table_summary(G, table_id, 5)
        self.assertEqual([(r["number"], r["dealer_total"], [(s["ref_id"], s["net"]) for s in r["seats"]]) for r in rounds], [(1, 18, [(ALICE, 10)]), (2, 18, [(ALICE, 10)])])
        self.assertEqual(self.store.game_table_summary(G + 1, table_id, 5), [])
        self.assertEqual(len(self.store.game_table_summary(G, table_id, 1)), 1)
        self.assertEqual(self.store.game_table_summary(G, table_id, 1)[0]["number"], 2)
        self.assertEqual(self.store.game_table_summary(G, table_id, 0), [])

    async def test_the_size_is_read_through_one_function(self):
        for _ in range(3):
            await self.play()
        self.games.summary_rounds = lambda guild_id: 2
        click = await self.press(ALICE, "Leave table")
        self.assertEqual(click.response.edited[0]["content"].count("\nRound "), 2)

    async def test_the_size_follows_the_server_setting_and_defaults_to_five(self):
        """FEAT-27: the summary length is games.summary_rounds of the table's own server."""
        self.assertEqual(self.games.summary_rounds(G), 5)
        self.store.set_game_summary_rounds(G, 2)
        self.assertEqual(self.games.summary_rounds(G), 2)
        self.assertEqual(self.games.summary_rounds(G + 1), 5)

    async def test_seven_seats_and_five_rounds_fit_in_one_message(self):
        users = [ALICE, BOB, CARA, 8, 9, 10, 11]
        for user in users[3:]:
            self.store.change_balance(G, user, 100, "start", 1)
        settled = None
        for _ in range(5):
            self.rig(*(("10",) * 7 + ("9",) + ("10",) * 7 + ("9",)))
            for user in users:
                if settled is None:
                    await self.bet(user, 10)
                else:
                    await self.press(user, "Play again (same bet)", settled)
            await self.press(ALICE, "Deal now")
            for user in users:
                await self.press(user, "Stand")
            settled = self.latest()
            self.assertEqual(settled["status"], "settled")
            self.assertLessEqual(len(self.games.render(settled)), 2000)
        self.rig(*WIN)
        snap = self.latest()
        await self.games.on_idle_timer(G, snap["table_id"], snap["round_id"])
        closing = self.texts()[-1]
        self.assertLessEqual(len(closing), 2000)
        self.assertEqual(closing.count("\nRound "), 5)

    def test_long_names_trim_the_oldest_rounds_first(self):
        seats = [{"kind": "character", "ref_id": i, "name": "N" * 150, "brought_by": None, "net": -5} for i in range(7)]
        rounds = [{"number": n, "dealer_total": 20, "seats": seats} for n in range(1, 6)]
        text = self.games_for_text()._summary_text("Table closed.", rounds)
        self.assertLessEqual(len(text), 2000)
        self.assertIn("Round 5:", text)
        self.assertNotIn("Round 1:", text)
        self.assertEqual(self.games_for_text()._summary_text("Table closed.", []), "Table closed.")

    @staticmethod
    def games_for_text():
        return games_discord.GameTables(mock.Mock())


class SummaryCharacterTests(SeatCase):
    async def test_a_character_seat_shows_by_name_with_its_net(self):
        await self.start()
        await self.fire()  # the model stands on 16 against 17
        self.assertEqual(self.latest()["status"], "settled")
        click = await self.press(ALICE, "Leave table")
        self.assertEqual(click.response.edited[0]["content"], f"Table closed.\nRound 1: dealer 17 — <@{ALICE}> +10, **Ann** −10")


class SummaryCharacterMovesTests(SeatCase):
    async def closing(self, move, *ranks):
        self.bot.models.structured_compiled = self.fake_model({"move": move, "line": "ok"})
        await self.start(*ranks)
        await self.fire()
        self.assertEqual(self.latest()["status"], "settled")
        return (await self.press(ALICE, "Leave table")).response.edited[0]["content"]

    async def test_a_character_push_reads_push(self):
        text = await self.closing("stand", "10", "10", "10", "10", "7", "7")  # Ann 17 against the dealer's 17
        self.assertEqual(text, f"Table closed.\nRound 1: dealer 17 — <@{ALICE}> +10, **Ann** push")

    async def test_a_character_double_counts_the_doubled_stake(self):
        text = await self.closing("double", "10", "10", "10", "10", "6", "7")  # Ann 16 doubles into an 18 and wins 20
        self.assertEqual(text, f"Table closed.\nRound 1: dealer 17 — <@{ALICE}> +10, **Ann** +20")


class CharacterPacingTests(SeatCase):
    async def test_the_table_talk_line_comes_before_the_remaining_dealer_frames(self):
        self.games.DEALER_STEP_SECONDS = 1.5
        seen = []

        async def pause(seconds):
            seen.append(len(self.webhook.sent))
        self.games._pause = pause
        await self.start()
        await self.fire()  # Ann stands on 16 against 17: reveal frame, then the result
        self.assertEqual(self.latest()["status"], "settled")
        self.assertEqual(len(self.webhook.sent), 1)
        self.assertEqual(seen, [1])


class PacingTests(GameCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.games.DEALER_STEP_SECONDS = 1.5
        self.pauses, self.hook = [], None

        async def pause(seconds):
            sender = next(iter(self.games.senders.values()))
            self.pauses.append((seconds, any(lock.locked() for lock in list(self.games.locks.values())), sender.lock.locked()))
            if self.hook:
                await self.hook(len(self.pauses))
        self.games._pause = pause

    async def stand_on_19(self):
        self.rig(*WIN)
        await self.bet(ALICE, 10)
        await self.press(ALICE, "Deal now")
        self.mark = len(self.channel.edits)
        return await self.press(ALICE, "Stand")

    async def test_reveal_then_each_card_then_the_result(self):
        click = await self.stand_on_19()
        first, *later = [click.response.edited[0]["content"], *[kw["content"] for _, kw in self.channel.edits[self.mark:]]]
        self.assertRegex(first, r"Dealer: 9. 7. \(16\)")
        self.assertNotIn("win", first)
        self.assertRegex(later[0], r"Dealer: 9. 7. 2. \(18\)")
        self.assertNotIn("win", later[0])
        self.assertIn("win +10", later[1])
        self.assertEqual(len(later), 2)
        self.assertEqual([p[0] for p in self.pauses], [1.5, 1.5])
        self.assertEqual(self.labels(), ["Play again (same bet)", "Leave table", "How to play"])
        self.assertEqual(self.balance(ALICE), 110)
        for view in (click.response.edited[0]["view"], self.channel.edits[self.mark][1]["view"]):
            self.assertEqual([child.item.label for child in view.children], ["How to play"])  # nothing but the guide until the result
        self.assertEqual([child.item.label for child in self.channel.edits[-1][1]["view"].children], ["Play again (same bet)", "Leave table", "How to play"])

    async def test_busted_players_skip_the_reveal_and_its_pause(self):
        self.rig("10", "9", "6", "7", tail="10")  # the player busts on 26; the dealer neither draws nor decides anything
        await self.bet(ALICE, 10)
        await self.press(ALICE, "Deal now")
        click = await self.press(ALICE, "Hit")
        self.assertEqual(self.pauses, [])
        self.assertIn("bust", click.response.edited[0]["content"])
        self.assertEqual(self.labels(), ["Play again (same bet)", "Leave table", "How to play"])

    async def test_a_standing_player_keeps_the_reveal_even_when_the_dealer_draws_nothing(self):
        self.rig("10", "10", "9", "K")  # 19 against a dealer 20 on two cards
        await self.bet(ALICE, 10)
        await self.press(ALICE, "Deal now")
        click = await self.press(ALICE, "Stand")
        self.assertRegex(click.response.edited[0]["content"], r"Dealer: 10. K. \(20\)")
        self.assertEqual(len(self.pauses), 1)
        self.assertIn("lose", self.texts()[-1])

    async def test_no_lock_is_held_while_waiting(self):
        await self.stand_on_19()
        self.assertEqual([(a, b) for _, a, b in self.pauses], [(False, False)] * 2)

    async def test_money_and_replay_are_settled_before_the_first_frame(self):
        seen = []

        async def hook(n):
            seen.append((self.latest()["status"], self.balance(ALICE)))
        self.hook = hook
        await self.stand_on_19()
        self.assertEqual(seen, [("settled", 110)] * 2)

    async def test_a_newer_paint_is_not_overwritten(self):
        again = []

        async def hook(n):
            if n == 1:
                again.append(await self.press(ALICE, "Play again (same bet)"))
        self.hook = hook
        click = await self.stand_on_19()
        later = [kw["content"] for _, kw in self.channel.edits[self.mark:]]
        self.assertEqual(later, [])  # neither the second card frame nor the result came after the newer paint
        self.assertIn("Join with", again[0].response.edited[0]["content"])
        self.assertNotIn("win +10", " ".join([click.response.edited[0]["content"], *later]))
        self.assertEqual(self.latest()["status"], "joining")

    async def test_a_closing_paint_wins_too(self):
        async def hook(n):
            if n == 1:
                await self.games.set_channel(G, CH, False)
        self.hook = hook
        await self.stand_on_19()
        self.assertEqual(self.texts()[-1], "Table closed: games were turned off here.\nRound 1: dealer 18 — " + f"<@{ALICE}> +10")
        self.assertEqual(sum("Dealer: 9" in t for t in self.texts()[self.mark:]), 0)

    async def test_a_natural_shows_the_reveal_once_then_the_result(self):
        self.rig("9", "A", "9", "K")
        await self.bet(ALICE, 10)
        click = await self.press(ALICE, "Deal now")
        self.assertRegex(click.response.edited[0]["content"], r"Dealer: A. K. \(soft 21\)")
        self.assertNotIn("lose", click.response.edited[0]["content"])
        self.assertEqual(len(self.pauses), 1)
        self.assertIn("lose −10", self.texts()[-1])
        self.assertEqual(self.labels(), ["Play again (same bet)", "Leave table", "How to play"])

    async def test_a_timeout_move_is_paced_too(self):
        self.rig(*WIN)
        await self.bet(ALICE, 10)
        await self.press(ALICE, "Deal now")
        snap = self.latest()
        await self.games.on_turn_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        self.assertEqual(self.latest()["status"], "settled")
        self.assertEqual(len(self.pauses), 2)
        self.assertIn("(timed out)", self.texts()[-1])

    async def test_no_pause_without_a_finished_round(self):
        self.rig("5", "10", "3", "7")
        await self.bet(ALICE, 10)
        await self.press(ALICE, "Deal now")
        await self.press(ALICE, "Hit")
        self.assertEqual(self.pauses, [])
