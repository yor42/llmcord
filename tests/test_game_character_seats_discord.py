"""FEAT-20 part B: character seats in blackjack on Discord (favorites option, rendering, character turns, table talk). Offline: fake model, fake webhook."""
import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import discord
from helpers import invoke
from test_games_discord import ALICE, BOB, CH, G, GameCase, Click

from llmcord_core import game_chooser
from llmcord_core.budget import BudgetExceeded
from llmcord_core.games_discord import GameButton, GameTables, NOT_YOUR_TURN
from llmcord_core.usage import log_attribution

CARD = json.dumps({"description": "A cheerful gambler named {{char}}.", "personality": "Superstitious and chatty."})
LORE = "SECRET-WORLD-LORE the vault code is 4411"
MEMORY = "PERSONAL-FACT likes tea"


class Webhook:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    async def send(self, content=None, **kwargs):
        if self.fail:
            raise discord.HTTPException(SimpleNamespace(status=500, reason="boom"), "boom")
        self.sent.append((content, kwargs))


class SeatCase(GameCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        s = self.store
        self.games.CHARACTER_SECONDS = 3600
        self.world = s.create_space(G, "W", "world")
        s.bind_channel(G, CH, self.world)
        self.ann = s.create_character(G, self.world, "Ann")
        s.db.execute("UPDATE characters SET card=? WHERE id=?", (CARD, self.ann))
        s.db.commit()
        s.add_lore(G, "character", self.ann, LORE, [], constant=True)
        s.add_lore(G, "space", self.world, LORE, [], constant=True)
        s.set_daily_settings(G, 0, 0, 7)
        s.add_favorite(G, ALICE, self.ann)
        s.change_character_balance(G, self.ann, 100, "start", 1)
        self.webhook = Webhook()
        self.bot._webhook = mock.AsyncMock(return_value=self.webhook)
        self.calls = []
        self.model = self.fake_model({"move": "stand", "line": "Feeling lucky."})
        self.bot.models.structured_compiled = self.model

    def fake_model(self, result):
        async def model(role, request, name, schema):
            self.context = (log_attribution()["feature"], log_attribution()["purpose"], self.bot.backend._pinned.get() is not None)
            self.calls.append((role, request, name, schema, any(lock.locked() for lock in list(self.games.locks.values()))))
            if isinstance(result, BaseException):
                raise result
            if callable(result):
                return await result()
            return result
        return model

    async def start(self, *ranks):
        """Alice opens a table with her favorite Ann, then the round is dealt and Alice stands: Ann is on turn."""
        self.rig(*(ranks or ("10", "10", "10", "10", "6", "7")))
        click = await self.bet_fav(ALICE, 10)
        await self.games.on_join_timer(G, self.store.open_table_for(G, CH)["id"], self.latest()["round_id"])
        await self.press(ALICE, "Stand")
        return click

    async def bet_fav(self, user, amount):
        click = Click(self.bot, user)
        await invoke(self.bot, "blackjack", click, bet=amount, favorites=True)
        return click

    async def fire(self):
        snap = self.latest()
        await self.games.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"])

    def actors(self):
        return [(r["seat_index"], r["move"], r["actor"]) for r in self.store.db.execute("SELECT * FROM game_moves ORDER BY id")]


class FavoritesOptionTests(SeatCase):
    async def test_joining_with_favorites_lists_who_sat_and_who_did_not(self):
        bo = self.store.create_character(G, self.world, "Bo")
        self.store.add_favorite(G, ALICE, bo)
        await self.bet(BOB, 10)
        click = await self.bet_fav(ALICE, 10)
        self.assertEqual(click.response.sent[0][0], "You joined the table with 10 coins. Your favorites joined too: Ann (10). Not seated: Bo (not enough coins).")
        self.assertEqual([(s["kind"], s["name"], s["brought_by"]) for s in self.latest()["seats"]],
                         [("member", None, None), ("member", None, None), ("character", "Ann", ALICE)])
        self.assertEqual(self.store.character_balance(G, self.ann), 90)

    async def test_without_the_option_only_the_member_sits(self):
        await self.bet(BOB, 10)
        click = await self.bet(ALICE, 10)
        self.assertEqual(click.response.sent[0][0], "You joined the table with 10 coins.")
        self.assertEqual([s["kind"] for s in self.latest()["seats"]], ["member", "member"])

    async def test_unbound_channel_seats_the_member_alone_and_says_so(self):
        self.store.db.execute("DELETE FROM channels WHERE channel_id=?", (CH,))
        self.store.db.commit()
        await self.bet(BOB, 10)
        click = await self.bet_fav(ALICE, 10)
        self.assertEqual(click.response.sent[0][0], "You joined the table with 10 coins. Characters can't sit at tables in this channel.")
        self.assertEqual([s["kind"] for s in self.latest()["seats"]], ["member", "member"])
        self.assertEqual(self.store.character_balance(G, self.ann), 100)

    async def test_opening_a_table_with_favorites_seats_them_and_notes_in_a_followup(self):
        click = await self.bet_fav(ALICE, 10)
        self.assertEqual([s["kind"] for s in self.latest()["seats"]], ["member", "character"])
        self.assertIn("**Ann** (with <@5>)", click.response.sent[0][0])
        self.assertEqual(click.followup.sent[0][0], "Your favorites joined too: Ann (10).")

    async def test_again_seats_only_the_member(self):
        await self.start()
        await self.fire()
        self.assertEqual(self.latest()["status"], "settled")
        await self.press(ALICE, "Play again (same bet)")
        self.assertEqual([s["kind"] for s in self.latest()["seats"]], ["member"])

    async def test_an_archived_favorite_is_named_as_not_seated(self):
        self.store.set_archived_favorites(G, True)
        self.store.archive_character(G, self.ann, True)
        click = await self.bet_fav(ALICE, 10)
        self.assertEqual(click.followup.sent[0][0], "Not seated: Ann (archived).")


class RenderTests(SeatCase):
    async def test_character_seat_is_bold_with_who_brought_it_and_nobody_is_pinged(self):
        await self.start()
        text = self.games.render(self.latest())
        self.assertIn("2. **Ann** (with <@5>) bet 10 coins", text)
        self.assertIn("On turn: **Ann**.", text)
        self.assertNotIn("<@1>", text.replace("<@5>", ""))
        self.assertTrue(all(kw["allowed_mentions"].users is False for kw in self.channel_kwargs()))

    def channel_kwargs(self):
        return [kw for _, kw in self.channel.edits]

    async def test_name_markup_and_mentions_are_escaped(self):
        self.store.db.execute("UPDATE characters SET name=? WHERE id=?", ("A*b @everyone", self.ann))
        self.store.db.commit()
        await self.start()
        text = self.games.render(self.latest())
        self.assertIn("**A\\*b @​everyone**", text)


class CharacterTurnTests(SeatCase):
    async def test_the_character_timer_is_short_and_armed_for_a_character_seat(self):
        self.assertEqual(GameTables.CHARACTER_SECONDS, 2)
        await self.start()
        key = (G, self.latest()["table_id"])
        self.assertIn("blackjack-character", self.games.timers[key].get_name())

    async def test_talk_off_plays_the_house_rule_without_a_model_call(self):
        self.store.set_game_character_talk(G, False)
        await self.start()
        await self.fire()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.webhook.sent, [])
        self.assertEqual(self.actors()[-2:], [(0, "stand", "member"), (1, "hit", "character")])
        await self.fire()
        self.assertEqual((self.actors()[-1], self.calls, self.latest()["status"]), ((1, "stand", "character"), [], "settled"))

    async def test_talk_on_uses_the_models_move_and_posts_the_line(self):
        await self.start()
        await self.fire()
        self.assertEqual(self.actors()[-1], (1, "stand", "character"))
        self.assertEqual(len(self.calls), 1)
        role, request, name, schema, locked = self.calls[0]
        self.assertEqual((role, name, schema["properties"]["move"]["enum"], locked), ("director", "blackjack_move", ["hit", "stand", "double"], False))
        self.assertEqual(self.context, ("game", "game", True))
        (line, kwargs), = self.webhook.sent
        self.assertEqual(line, "Feeling lucky.")
        self.assertEqual((kwargs["username"], kwargs["allowed_mentions"].everyone, kwargs["allowed_mentions"].users), ("Ann", False, False))
        self.bot._webhook.assert_awaited_once()
        self.assertEqual(self.bot._webhook.await_args.args[1]["name"], "Ann")

    async def test_the_prompt_holds_only_the_card_and_the_table(self):
        await self.start()
        await self.fire()
        text = "\n".join(m.text for m in self.calls[0][1].messages)
        for needle in ("Ann", "cheerful gambler named Ann", "Superstitious", "(16)", "Dealer shows: 10", "Legal moves: hit, stand, double"):
            self.assertIn(needle, text)
        for absent in ("SECRET-WORLD-LORE", "4411", "PERSONAL-FACT", "<@", "Alice", "Tester"):
            self.assertNotIn(absent, text)

    async def test_model_failures_fall_back_to_the_house_rule_with_no_line(self):
        for result in ({"move": "fold", "line": "hmm"}, {"line": "no move"}, RuntimeError("down"), BudgetExceeded(SimpleNamespace())):
            with self.subTest(result=result):
                self.webhook.sent.clear()
                self.bot.models.structured_compiled = self.fake_model(result)
                await self.fresh_round()
                await self.fire()
                self.assertEqual(self.actors()[-1], (1, "hit", "character"))
                self.assertEqual(self.webhook.sent, [])

    async def fresh_round(self):
        """Close the previous table and start another with Ann on turn."""
        table = self.store.open_table_for(G, CH)
        if table:
            self.store.close_table(G, table["id"])
            self.games._forget((G, table["id"]))
        await self.start()

    async def test_a_model_that_takes_too_long_falls_back(self):
        self.games.MODEL_SECONDS = 0.01

        async def slow():
            await asyncio.sleep(5)
        self.bot.models.structured_compiled = self.fake_model(slow)
        await self.start()
        await self.fire()
        self.assertEqual(self.actors()[-1], (1, "hit", "character"))
        self.assertEqual(self.webhook.sent, [])

    async def test_a_move_made_while_the_model_thinks_drops_the_decision(self):
        gate, started = asyncio.Event(), asyncio.Event()

        async def slow():
            started.set()
            await gate.wait()
            return {"move": "stand", "line": "Too late."}
        self.bot.models.structured_compiled = self.fake_model(slow)
        await self.start()
        snap = self.latest()
        task = asyncio.create_task(self.games.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"]))
        await started.wait()
        self.assertFalse(any(lock.locked() for lock in self.games.locks.values()))
        self.store.play(G, snap["round_id"], 1, "hit", "timeout", snap["moves"])
        gate.set()
        await task
        self.assertEqual(self.actors()[-1], (1, "hit", "timeout"))
        self.assertEqual(len(self.actors()), snap["moves"] + 1)
        self.assertEqual(self.webhook.sent, [])

    async def test_a_move_no_longer_legal_falls_back_to_the_house_rule_and_drops_the_line(self):
        await self.start()
        snap = self.latest()
        real = self.store.round_snapshot

        def narrowed(*args):
            out = real(*args)
            if out["legal"] and self.calls:  # after the model answered: only stand is left
                out = {**out, "legal": {s: ("hit",) for s in out["legal"]}}
            return out
        with mock.patch.object(self.store, "round_snapshot", narrowed):
            await self.games.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        self.assertEqual(self.actors()[-1], (1, "hit", "character"))
        self.assertEqual(self.webhook.sent, [])

    async def test_a_refused_move_is_replaced_by_the_house_rule(self):
        from llmcord_core.admin_store import GameError
        await self.start()
        real, calls = self.store.play, []

        def refuse_first(*args):
            calls.append(args[3])
            if len(calls) == 1:
                raise GameError("nope")
            return real(*args)
        with mock.patch.object(self.store, "play", refuse_first):
            await self.fire()
        self.assertEqual((calls, self.actors()[-1], self.webhook.sent, len(self.calls)), (["stand", "hit"], (1, "hit", "character"), [], 1))

    async def test_a_slow_webhook_post_is_given_up(self):
        class Slow(Webhook):
            async def send(self, content=None, **kwargs):
                await asyncio.sleep(5)
        self.bot._webhook = mock.AsyncMock(return_value=Slow())
        self.games.POST_SECONDS = 0.01
        await self.start()
        await self.fire()
        self.assertEqual(self.actors()[-1], (1, "stand", "character"))

    async def test_no_move_buttons_while_a_character_is_on_turn(self):
        await self.start()
        self.assertEqual(self.labels(), [])

    async def test_a_webhook_failure_is_ignored(self):
        self.webhook.fail = True
        await self.start()
        await self.fire()
        self.assertEqual(self.actors()[-1], (1, "stand", "character"))

    async def test_a_member_cannot_press_for_a_character_seat(self):
        await self.start()
        before = self.latest()["moves"]
        snap = self.latest()  # no buttons are shown, so press a leftover one carrying the current move count
        click = Click(self.bot, ALICE)
        await GameButton("hit", snap["table_id"], snap["round_id"], snap["moves"]).callback(click)
        self.assertEqual(click.followup.sent[-1][0], NOT_YOUR_TURN)
        self.assertEqual(self.latest()["moves"], before)

    async def test_a_stale_timer_does_nothing(self):
        await self.start()
        snap = self.latest()
        await self.games.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"] - 1)
        self.assertEqual((self.calls, self.latest()["moves"]), ([], snap["moves"]))

    async def test_table_talk_usage_has_a_label(self):
        from llmcord_core.dashboard import USAGE_FEATURES
        self.assertEqual(USAGE_FEATURES["game"], "Blackjack table talk")

    async def test_line_cleaning(self):
        self.assertEqual(game_chooser.clean_line("  hi\n there  "), "hi there")
        self.assertEqual(len(game_chooser.clean_line("x" * 500)), 200)
        self.assertEqual(game_chooser.clean_line(None), "")


if __name__ == "__main__":
    unittest.main()
