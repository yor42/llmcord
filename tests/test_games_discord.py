"""FEAT-19 part C: blackjack on Discord (commands, buttons, timers, restart recovery). Offline, fake interactions, rigged shoes."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

import discord
from helpers import FakeInteraction, FakeResponse, invoke, make_settings

from llmcord_core.admin_store import ConflictError, GameError
from llmcord_core.discord_bot import SkitBot
from llmcord_core.games import blackjack as bj
from llmcord_core.games_discord import GameButton, NOT_SEATED, NOT_YOUR_TURN, STALE

G, CH = 1, 100
ALICE, BOB, CARA = 5, 6, 7
SEED = "seed-for-the-test-0123456789"


def shoe(*ranks, tail="2"):
    return tuple(bj.RANKS.index(r) for r in ranks) + (bj.RANKS.index(tail),) * 100


class Response(FakeResponse):
    def __init__(self):
        super().__init__()
        self.edited = []

    async def edit_message(self, **kwargs):
        await asyncio.sleep(0)  # a real response yields, so concurrent presses overlap
        self.edited.append(kwargs)
        self._done = True

    async def send_message(self, content=None, **kwargs):
        await asyncio.sleep(0)
        await super().send_message(content, **kwargs)


class Partial:
    def __init__(self, channel, message_id):
        self.channel, self.id = channel, message_id

    async def edit(self, **kwargs):
        if self.id in self.channel.deleted:
            raise discord.NotFound(SimpleNamespace(status=404, reason="Not Found"), "gone")
        self.channel.edits.append((self.id, kwargs))


class Channel:
    def __init__(self, channel_id=CH):
        self.id, self.mention, self.edits, self.deleted = channel_id, f"<#{channel_id}>", [], set()

    def get_partial_message(self, message_id):
        return Partial(self, message_id)


class Click(FakeInteraction):
    def __init__(self, bot, user_id, guild_id=G, channel_id=CH):
        super().__init__(guild_id=guild_id, channel_id=channel_id, user_id=user_id)
        self.channel_id, self.client, self.response = channel_id, bot, Response()


class GameCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bot = SkitBot(make_settings())
        self.store, self.games = self.bot.store, self.bot.games
        self.games.JOIN_SECONDS = self.games.TURN_SECONDS = self.games.IDLE_SECONDS = 3600
        self.channel = Channel()
        self.bot.get_channel = lambda channel_id: self.channel if channel_id == CH else None
        self.shoe = None
        for patcher in (mock.patch.object(bj, "new_shoe", side_effect=lambda seed: self.shoe),
                        mock.patch("llmcord_core.admin_store.new_seed", return_value=SEED)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.store.set_game_channel(G, CH, True)
        for user in (ALICE, BOB, CARA):
            self.store.change_balance(G, user, 100, "start", 1)
        self.addCleanup(self.store.close)
        self.addCleanup(self.games.close)

    def rig(self, *ranks, tail="2"):
        self.shoe = shoe(*ranks, tail=tail)

    async def bet(self, user, amount, channel_id=CH):
        click = Click(self.bot, user, channel_id=channel_id)
        await invoke(self.bot, "blackjack", click, bet=amount)
        return click

    async def press(self, user, label, snap=None, **kw):
        click = Click(self.bot, user, **kw)
        buttons = {child.item.label: child for child in self.games.buttons(snap or self.latest()).children}
        await buttons[label].callback(click)
        return click

    def latest(self):
        return self.store.latest_round(G, self.store.open_table_for(G, CH)["id"])

    def balance(self, user):
        return self.store.balance(G, user)

    def texts(self):
        return [kw["content"] for _, kw in self.channel.edits]

    def labels(self, snap=None):
        view = self.games.buttons(snap or self.latest())
        return [child.item.label for child in view.children] if view else []


class AdminTests(GameCase):
    async def test_channel_on_and_off_defaults_to_this_channel(self):
        self.store.set_game_channel(G, CH, False)
        admin = FakeInteraction(admin=True)
        await invoke(self.bot, "admin games channel", admin, enabled=True)
        self.assertEqual(admin.replies, ["Games are now on in <#100>."])
        self.assertEqual(admin.response.sent[0][1]["ephemeral"], True)
        self.assertEqual(self.store.game_channels(G), {CH})
        admin = FakeInteraction(admin=True)
        await invoke(self.bot, "admin games channel", admin, enabled=False)
        self.assertEqual(admin.replies, ["Games are now off in <#100>."])
        self.assertEqual(self.store.game_channels(G), set())

    async def test_channel_option_targets_another_channel(self):
        admin = FakeInteraction(admin=True)
        await invoke(self.bot, "admin games channel", admin, enabled=True, channel=SimpleNamespace(id=200, mention="<#200>"))
        self.assertEqual(admin.replies, ["Games are now on in <#200>."])
        self.assertEqual(self.store.game_channels(G), {CH, 200})

    async def test_admin_commands_need_administrator(self):
        for path, kwargs in (("admin games channel", {"enabled": False}), ("admin games bets", {"min": 1, "max": 5})):
            member = FakeInteraction(admin=False)
            await invoke(self.bot, path, member, **kwargs)
            self.assertEqual(member.replies, ["Only server administrators can use that command."])
        self.assertEqual(self.store.game_channels(G), {CH})
        self.assertEqual(self.store.game_settings(G), {"min_bet": 1, "max_bet": 1000})

    async def test_turning_games_off_closes_the_open_table_and_refunds(self):
        await self.bet(ALICE, 10)
        await self.bet(BOB, 20)
        self.assertEqual((self.balance(ALICE), self.balance(BOB)), (90, 80))
        message_id = self.latest()["message_id"]
        admin = FakeInteraction(admin=True)
        await invoke(self.bot, "admin games channel", admin, enabled=False)
        self.assertEqual(admin.replies, ["Games are now off in <#100>. The open table was closed and bets were refunded."])
        self.assertEqual((self.balance(ALICE), self.balance(BOB)), (100, 100))
        self.assertIsNone(self.store.open_table_for(G, CH))
        self.assertEqual(self.games.timers, {})
        self.assertEqual(self.channel.edits[-1][0], message_id)
        self.assertEqual(self.channel.edits[-1][1]["content"], "Table closed: games were turned off here.")
        self.assertIsNone(self.channel.edits[-1][1]["view"])

    async def test_turning_off_a_channel_without_a_table_says_nothing_extra(self):
        admin = FakeInteraction(admin=True)
        await invoke(self.bot, "admin games channel", admin, enabled=False)
        self.assertEqual(admin.replies, ["Games are now off in <#100>."])
        self.assertEqual(self.channel.edits, [])

    async def test_bets_saved_and_conflict_and_invalid(self):
        admin = FakeInteraction(admin=True)
        await invoke(self.bot, "admin games bets", admin, min=5, max=500)
        self.assertEqual(admin.replies, ["Bets are now 5 to 500 coins."])
        self.assertEqual(self.store.game_settings(G), {"min_bet": 5, "max_bet": 500})
        admin = FakeInteraction(admin=True)
        with mock.patch.object(self.store, "set_game_settings", side_effect=ConflictError("x")):
            await invoke(self.bot, "admin games bets", admin, min=5, max=500)
        self.assertEqual(admin.replies, ["The game settings were changed elsewhere. Run the command again."])
        admin = FakeInteraction(admin=True)
        await invoke(self.bot, "admin games bets", admin, min=50, max=10)
        self.assertIn("1 <= minimum <= maximum", admin.replies[0])
        self.assertEqual(self.store.game_settings(G), {"min_bet": 5, "max_bet": 500})


class JoinTests(GameCase):
    async def test_refused_outside_game_channels_and_by_bots(self):
        click = await self.bet(ALICE, 10, channel_id=555)
        self.assertEqual(click.replies, ["Games are off in this channel."])
        self.assertTrue(click.response.sent[0][1]["ephemeral"])
        bot_click = Click(self.bot, 99)
        bot_click.user.bot = True
        await invoke(self.bot, "blackjack", bot_click, bet=10)
        self.assertEqual(bot_click.replies, ["Bots cannot play blackjack."])
        self.assertIsNone(self.store.open_table_for(G, CH))
        self.assertEqual(self.balance(ALICE), 100)

    async def test_limits_and_balance_refusals_open_no_table(self):
        self.store.set_game_settings(G, 5, 50)
        for amount, text in ((4, "The bet must be between 5 and 50 coins."), (51, "The bet must be between 5 and 50 coins.")):
            click = await self.bet(ALICE, amount)
            self.assertEqual(click.replies, [text])
        self.store.set_game_settings(G, 5, 500)
        poor = await self.bet(ALICE, 150)
        self.assertEqual(poor.replies, ["You need 150 coins for that bet but have 100."])
        self.assertTrue(poor.response.sent[0][1]["ephemeral"])
        self.assertIsNone(self.store.open_table_for(G, CH))
        self.assertEqual(self.balance(ALICE), 100)
        self.assertEqual(self.store.unfinished_rounds(), [])

    async def test_open_posts_one_public_table_message(self):
        click = await self.bet(ALICE, 10)
        content, kwargs = click.response.sent[0]
        self.assertNotIn("ephemeral", kwargs)
        self.assertEqual(kwargs["allowed_mentions"].to_dict(), discord.AllowedMentions.none().to_dict())
        snap = self.latest()
        self.assertEqual(snap["message_id"], 7001)
        self.assertIn("**Blackjack**", content)
        self.assertIn(f"<@{ALICE}> bet 10 coins", content)
        self.assertRegex(content, r"Join with /blackjack <bet> — dealing <t:\d+:R>\.")
        self.assertIn(f"Seed hash: {snap['seed_hash'][:12]}", content)
        self.assertNotIn(SEED, content)
        self.assertEqual([child.item.label for child in kwargs["view"].children], ["Deal now", "Leave"])
        self.assertEqual(self.balance(ALICE), 90)
        self.assertEqual(set(self.games.timers), {(G, snap["table_id"])})

    async def test_second_member_joins_the_open_table(self):
        await self.bet(ALICE, 10)
        click = await self.bet(BOB, 25)
        self.assertEqual(click.replies, ["You joined the table with 25 coins."])
        self.assertTrue(click.response.sent[0][1]["ephemeral"])
        self.assertEqual([(s["ref_id"], s["stake"]) for s in self.latest()["seats"]], [(ALICE, 10), (BOB, 25)])
        edit = self.channel.edits[-1]
        self.assertEqual(edit[0], 7001)
        self.assertIn(f"<@{BOB}> bet 25 coins", edit[1]["content"])
        self.assertEqual(edit[1]["allowed_mentions"].to_dict(), discord.AllowedMentions.none().to_dict())
        again = await self.bet(BOB, 25)
        self.assertEqual(again.replies, ["You already have a seat in this round."])

    async def test_failed_send_refunds_and_leaves_no_table(self):
        click = Click(self.bot, ALICE)

        real = click.response.send_message
        calls = []

        async def broken(*args, **kwargs):
            if not calls:
                calls.append(1)
                raise RuntimeError("discord down")
            return await real(*args, **kwargs)
        click.response.send_message = broken
        with self.assertLogs(level="ERROR"):
            await invoke(self.bot, "blackjack", click, bet=10)
        self.assertIsNone(self.store.open_table_for(G, CH))
        self.assertEqual(self.balance(ALICE), 100)
        self.assertEqual(self.games.timers, {})

    async def test_join_refused_while_a_round_is_in_play(self):
        self.rig("9", "10", "6", "7")
        await self.bet(ALICE, 10)
        await self.press(ALICE, "Deal now")
        click = await self.bet(BOB, 10)
        self.assertEqual(click.replies, ["A round is being played; join the next one with /blackjack when it ends."])
        self.assertEqual(len(self.latest()["seats"]), 1)
        self.assertEqual(self.balance(BOB), 100)

    async def test_leave_refunds_and_the_last_leaver_closes_the_table(self):
        await self.bet(ALICE, 10)
        await self.bet(BOB, 20)
        click = await self.press(BOB, "Leave")
        self.assertEqual(self.balance(BOB), 100)
        self.assertNotIn(f"<@{BOB}>", click.response.edited[0]["content"])
        click = await self.press(ALICE, "Leave")
        self.assertEqual(self.balance(ALICE), 100)
        self.assertEqual(click.response.edited[0]["content"], "Table closed.")
        self.assertIsNone(click.response.edited[0]["view"])
        self.assertIsNone(self.store.open_table_for(G, CH))
        self.assertEqual(self.games.timers, {})

    async def test_unseated_member_cannot_deal_or_leave(self):
        await self.bet(ALICE, 10)
        for label in ("Deal now", "Leave"):
            click = await self.press(CARA, label)
            self.assertEqual(click.replies, [NOT_SEATED])
        self.assertEqual(self.latest()["status"], "joining")


class DealTests(GameCase):
    async def test_deal_now_by_any_seated_member(self):
        self.rig("9", "10", "10", "6", "8", "7")
        await self.bet(ALICE, 10)
        await self.bet(BOB, 10)
        click = await self.press(BOB, "Deal now")
        snap = self.latest()
        self.assertEqual(snap["status"], "playing")
        content = click.response.edited[0]["content"]
        self.assertIn(f"Dealer: {bj.card_str(bj.RANKS.index('10'))} ??", content)
        self.assertIn(f"On turn: <@{ALICE}>", content)
        self.assertRegex(content, r"decide <t:\d+:R>")
        self.assertEqual(self.labels(snap), ["Hit", "Stand", "Double"])
        self.assertEqual(list(self.games.timers), [(G, snap["table_id"])])

    async def test_deal_by_the_join_timer(self):
        self.rig("9", "10", "6", "7")
        await self.bet(ALICE, 10)
        snap = self.latest()
        await self.games.on_join_timer(G, snap["table_id"], snap["round_id"])
        self.assertEqual(self.latest()["status"], "playing")
        self.assertIn("On turn:", self.texts()[-1])
        await self.games.on_join_timer(G, snap["table_id"], snap["round_id"])  # stale: nothing happens
        self.assertEqual(len(self.texts()), 1)

    async def test_real_timers_deal_and_play_with_the_default_policy(self):
        self.rig("9", "10", "6", "7")
        self.games.JOIN_SECONDS = self.games.TURN_SECONDS = 0.02
        await self.bet(ALICE, 10)
        for _ in range(100):
            await asyncio.sleep(0.02)
            if self.latest()["status"] == "settled":
                break
        snap = self.latest()
        self.assertEqual(snap["status"], "settled")
        rows = self.store.db.execute("SELECT move,actor FROM game_moves").fetchall()
        self.assertEqual([tuple(r) for r in rows], [("hit", "timeout"), ("stand", "timeout")])

    async def test_natural_deals_straight_to_settled(self):
        self.rig("A", "9", "K", "7")
        await self.bet(ALICE, 10)
        click = await self.press(ALICE, "Deal now")
        self.assertEqual(self.latest()["status"], "settled")
        self.assertIn("blackjack +15", click.response.edited[0]["content"])
        self.assertEqual(self.balance(ALICE), 115)


class PlayTests(GameCase):
    async def start(self, *ranks, bets=((ALICE, 10),)):
        self.rig(*ranks)
        for user, amount in bets:
            await self.bet(user, amount)
        await self.press(bets[0][0], "Deal now")

    async def test_only_the_seat_on_turn_may_press(self):
        await self.start("9", "10", "10", "6", "8", "7", bets=((ALICE, 10), (BOB, 10)))
        before = self.latest()["moves"]
        for user, text in ((CARA, NOT_SEATED), (BOB, NOT_YOUR_TURN)):
            click = await self.press(user, "Hit")
            self.assertEqual(click.replies, [text])
            self.assertTrue(click.response.sent[0][1]["ephemeral"])
        self.assertEqual(self.latest()["moves"], before)
        click = await self.press(ALICE, "Stand")
        self.assertEqual(self.latest()["moves"], before + 1)
        self.assertIn(f"On turn: <@{BOB}>", click.response.edited[0]["content"])
        self.assertEqual((await self.press(ALICE, "Hit")).replies, [NOT_YOUR_TURN])

    async def test_stale_custom_ids_are_refused_and_record_nothing(self):
        await self.start("9", "10", "6", "7")
        snap = self.latest()
        before = self.store.db.execute("SELECT COUNT(*) FROM game_moves").fetchone()[0]
        cases = [
            (G, GameButton("hit", snap["table_id"], snap["round_id"], snap["moves"] + 1), ALICE),  # wrong move count
            (G, GameButton("hit", snap["table_id"], snap["round_id"] + 50, 0), ALICE),  # unknown round
            (G, GameButton("hit", snap["table_id"] + 1, snap["round_id"], 0), ALICE),  # table does not match round
            (2, GameButton("hit", snap["table_id"], snap["round_id"], snap["moves"]), ALICE),  # another guild
        ]
        for guild, button, user in cases:
            click = Click(self.bot, user, guild_id=guild)
            await button.callback(click)
            self.assertEqual(click.replies, [STALE], (guild, button.custom_id))
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM game_moves").fetchone()[0], before)
        self.assertEqual(self.balance(ALICE), 90)

    async def test_button_custom_id_round_trip(self):
        button = GameButton("hit", 3, 4, 5, "Hit")
        self.assertEqual(button.item.custom_id, "llmcord:bj:hit:3:4:5")
        match = GameButton.__discord_ui_compiled_template__.fullmatch(button.item.custom_id)
        again = await GameButton.from_custom_id(None, button.item, match)
        self.assertEqual((again.action, again.table_id, again.round_id, again.moves), ("hit", 3, 4, 5))
        self.assertIsNone(GameButton.__discord_ui_compiled_template__.fullmatch("llmcord:bj:hit:x:4:5"))

    async def test_double_is_hidden_when_unaffordable(self):
        self.store.change_balance(G, ALICE, -90, "spend", 1)  # 10 left; bet 10 leaves 0
        await self.start("5", "10", "6", "7")
        self.assertEqual(self.balance(ALICE), 0)
        self.assertEqual(self.labels(), ["Hit", "Stand"])
        self.assertEqual((await self.press(ALICE, "Hit")).response.edited[0]["content"].count("Double"), 0)

    async def test_double_available_and_charges_again(self):
        await self.start("5", "10", "6", "7")
        self.assertEqual(self.labels(), ["Hit", "Stand", "Double"])
        click = await self.press(ALICE, "Double")
        self.assertEqual(self.latest()["status"], "settled")
        self.assertEqual(self.balance(ALICE), 80)  # 13 against the dealer's 17 loses 20
        self.assertIn("lose \u221220", click.response.edited[0]["content"])

    async def test_hit_keeps_the_turn_and_new_buttons_carry_the_new_move_count(self):
        await self.start("5", "10", "3", "7")
        first = self.latest()
        click = await self.press(ALICE, "Hit")
        view = click.response.edited[0]["view"]
        ids = [child.item.custom_id for child in view.children]
        self.assertEqual(ids[0], f"llmcord:bj:hit:{first['table_id']}:{first['round_id']}:1")
        stale = Click(self.bot, ALICE)
        old = GameButton("hit", first["table_id"], first["round_id"], 0)
        await old.callback(stale)
        self.assertEqual(stale.replies, [STALE])

    async def test_turn_timeout_applies_the_default_policy(self):
        await self.start("9", "10", "10", "6", "8", "7", bets=((ALICE, 10), (BOB, 10)))
        snap = self.latest()
        await self.games.on_turn_timer(G, snap["table_id"], snap["round_id"], 0)
        rows = self.store.db.execute("SELECT seat_index,move,actor FROM game_moves").fetchall()
        self.assertEqual([tuple(r) for r in rows], [(0, "hit", "timeout")])  # 15 hits
        self.assertIn(f"1. <@{ALICE}> bet 10 coins", self.texts()[-1])
        self.assertIn("(timed out)", self.texts()[-1])
        snap = self.latest()
        await self.games.on_turn_timer(G, snap["table_id"], snap["round_id"], 0)  # stale move count: nothing
        self.assertEqual(self.latest()["moves"], 1)
        await self.games.on_turn_timer(G, snap["table_id"], snap["round_id"], 1)  # now 17: stands
        self.assertEqual(self.store.db.execute("SELECT move FROM game_moves ORDER BY id").fetchall()[-1][0], "stand")
        self.assertEqual(self.latest()["legal"].keys(), {1})

    async def test_concurrent_presses_apply_once(self):
        await self.start("5", "10", "3", "7")
        click_a, click_b = Click(self.bot, ALICE), Click(self.bot, ALICE)
        snap = self.latest()
        button = lambda: GameButton("hit", snap["table_id"], snap["round_id"], snap["moves"])
        await asyncio.gather(button().callback(click_a), button().callback(click_b))
        self.assertEqual(self.latest()["moves"], 1)
        self.assertEqual(sorted(len(c.response.edited) for c in (click_a, click_b)), [0, 1])
        loser = click_a if not click_a.response.edited else click_b
        self.assertEqual(loser.replies, [STALE])

    async def test_a_press_waits_for_the_table_lock(self):
        await self.start("5", "10", "3", "7")
        snap = self.latest()
        click = Click(self.bot, ALICE)
        async with self.games.lock((G, snap["table_id"])):
            task = asyncio.create_task(GameButton("stand", snap["table_id"], snap["round_id"], 0).callback(click))
            await asyncio.sleep(0.01)
            self.assertFalse(task.done())
        await task
        self.assertEqual(self.latest()["status"], "settled")


class SettleTests(GameCase):
    async def settle_alice_win(self):
        self.rig("10", "9", "9", "7")  # player 19 against 16, the dealer draws a 2 to 18
        await self.bet(ALICE, 10)
        await self.press(ALICE, "Deal now")
        return await self.press(ALICE, "Stand")

    async def test_settled_message_shows_payouts_and_reveals_the_seed_only_then(self):
        click = await self.settle_alice_win()
        content = click.response.edited[0]["content"]
        self.assertIn("win +10", content)
        self.assertRegex(content, r"Dealer: 9. 7. 2. \(18\)")
        self.assertIn(f"Seed: {SEED} (hash {self.latest()['seed_hash'][:12]})", content)
        self.assertNotIn("Seed hash:", content)
        self.assertEqual(self.labels(), ["Play again (same bet)", "Leave table"])
        self.assertEqual(self.balance(ALICE), 110)

    async def test_the_seed_is_absent_from_every_message_before_settlement(self):
        self.rig("10", "9", "9", "7")
        opened = await self.bet(ALICE, 10)
        await self.bet(BOB, 10)
        dealt = await self.press(ALICE, "Deal now")
        seen = [opened.response.sent[0][0], *self.texts(), dealt.response.edited[0]["content"]]
        self.assertTrue(all(SEED not in text for text in seen))
        self.assertTrue(all("Seed hash:" in text for text in seen))
        snapshot = self.latest()
        self.assertIsNone(snapshot["seed"])

    async def test_outcome_words(self):
        self.rig("10", "9", "9", "9", "8", "7")
        await self.bet(ALICE, 10)
        await self.bet(BOB, 10)
        await self.press(ALICE, "Deal now")
        await self.press(ALICE, "Stand")
        click = await self.press(BOB, "Stand")
        content = click.response.edited[0]["content"]
        self.assertIn(f"<@{ALICE}> bet 10 coins", content)
        self.assertIn("win +10", content)
        self.assertIn("lose \u221210", content)

    async def test_push_wording(self):
        self.rig("10", "9", "8", "9")  # 18 against 18
        await self.bet(ALICE, 10)
        await self.press(ALICE, "Deal now")
        click = await self.press(ALICE, "Stand")
        self.assertIn("push", click.response.edited[0]["content"])
        self.assertEqual(self.balance(ALICE), 100)

    async def test_play_again_joins_the_next_round_with_the_same_bet(self):
        await self.settle_alice_win()
        first = self.latest()
        self.rig("9", "10", "6", "7")
        click = await self.press(ALICE, "Play again (same bet)")
        snap = self.latest()
        self.assertEqual((snap["number"], snap["status"]), (first["number"] + 1, "joining"))
        self.assertEqual([(s["ref_id"], s["stake"]) for s in snap["seats"]], [(ALICE, 10)])
        self.assertEqual(self.balance(ALICE), 100)
        self.assertEqual(snap["message_id"], first["message_id"])
        self.assertIn("Seed hash:", click.response.edited[0]["content"])
        self.assertNotIn(SEED, click.response.edited[0]["content"])
        self.assertEqual(self.labels(), ["Deal now", "Leave"])
        other = await self.press(BOB, "Deal now")
        self.assertEqual(other.replies, [NOT_SEATED])

    async def test_play_again_uses_the_original_bet_after_a_double(self):
        self.rig("5", "10", "6", "7")
        await self.bet(ALICE, 10)
        await self.press(ALICE, "Deal now")
        await self.press(ALICE, "Double")
        self.assertEqual(self.latest()["seats"][0]["stake"], 20)
        self.rig("9", "10", "6", "7")
        await self.press(ALICE, "Play again (same bet)")
        self.assertEqual(self.latest()["seats"][0]["stake"], 10)

    async def test_play_again_without_funds_is_refused_and_the_table_stays_usable(self):
        await self.settle_alice_win()
        settled = self.latest()
        self.store.change_balance(G, ALICE, -self.balance(ALICE), "spend", 1)
        click = await self.press(ALICE, "Play again (same bet)", settled)
        self.assertEqual(click.replies, ["You need 10 coins for that bet but have 0."])
        self.assertEqual(self.latest()["status"], "cancelled")
        self.assertEqual(self.store.unfinished_rounds(), [])
        self.store.change_balance(G, ALICE, 50, "back", 1)
        again = await self.press(ALICE, "Play again (same bet)", settled)
        self.assertEqual(self.latest()["status"], "joining")
        self.assertEqual(len(again.response.edited), 1)

    async def test_a_new_member_can_use_the_command_between_rounds(self):
        await self.settle_alice_win()
        self.rig("9", "10", "6", "7")
        click = await self.bet(BOB, 30)
        self.assertEqual(click.replies, ["You joined the table with 30 coins."])
        snap = self.latest()
        self.assertEqual((snap["status"], [s["ref_id"] for s in snap["seats"]]), ("joining", [BOB]))
        self.assertIn("Seed hash:", self.texts()[-1])

    async def test_leave_table_closes_only_when_everyone_has_left(self):
        self.rig("10", "9", "10", "9", "9", "7", tail="2")
        await self.bet(ALICE, 10)
        await self.bet(BOB, 10)
        await self.press(ALICE, "Deal now")
        await self.press(ALICE, "Stand")
        await self.press(BOB, "Stand")
        click = await self.press(ALICE, "Leave table")
        self.assertEqual(click.replies, ["You left the table."])
        self.assertIsNotNone(self.store.open_table_for(G, CH))
        click = await self.press(BOB, "Leave table")
        self.assertEqual(click.response.edited[0]["content"], "Table closed.")
        self.assertIsNone(self.store.open_table_for(G, CH))
        self.assertEqual(self.games.timers, {})

    async def test_idle_timeout_closes_the_table(self):
        await self.settle_alice_win()
        snap = self.latest()
        await self.games.on_idle_timer(G, snap["table_id"], snap["round_id"])
        self.assertIsNone(self.store.open_table_for(G, CH))
        self.assertEqual(self.channel.edits[-1][1]["content"], "Table closed.")
        self.assertIsNone(self.channel.edits[-1][1]["view"])
        self.assertEqual(self.balance(ALICE), 110)

    async def test_idle_timer_does_nothing_once_a_new_round_is_open(self):
        await self.settle_alice_win()
        old = self.latest()
        self.rig("9", "10", "6", "7")
        await self.press(ALICE, "Play again (same bet)")
        await self.games.on_idle_timer(G, old["table_id"], old["round_id"])
        self.assertIsNotNone(self.store.open_table_for(G, CH))
        self.assertEqual(self.latest()["status"], "joining")

    async def test_real_idle_timer_closes_the_table(self):
        self.games.IDLE_SECONDS = 0.02
        await self.settle_alice_win()
        for _ in range(100):
            await asyncio.sleep(0.02)
            if self.store.open_table_for(G, CH) is None:
                break
        self.assertIsNone(self.store.open_table_for(G, CH))


class ReviewFixTests(GameCase):
    async def settle_alice_win(self):
        self.rig("10", "9", "9", "7")
        await self.bet(ALICE, 10)
        await self.press(ALICE, "Deal now")
        return await self.press(ALICE, "Stand")

    async def test_failed_join_between_rounds_leaves_the_idle_timer_able_to_close(self):
        await self.settle_alice_win()
        settled = self.latest()
        self.store.change_balance(G, CARA, -self.balance(CARA), "spend", 1)
        click = await self.bet(CARA, 10)
        self.assertEqual(click.replies, ["You need 10 coins for that bet but have 0."])
        self.assertEqual(self.latest()["status"], "cancelled")
        await self.games.on_idle_timer(G, settled["table_id"], settled["round_id"])
        self.assertIsNone(self.store.open_table_for(G, CH))

    async def test_response_failure_still_refreshes_the_table_message(self):
        self.rig("9", "10", "6", "7")
        await self.bet(ALICE, 10)
        click = Click(self.bot, ALICE)

        async def broken(**kwargs):
            raise discord.HTTPException(SimpleNamespace(status=500, reason="x"), "boom")
        click.response.edit_message = broken
        snap = self.latest()
        with self.assertLogs(level="WARNING"):
            await GameButton("deal", snap["table_id"], snap["round_id"], 0).callback(click)
        self.assertEqual(self.latest()["status"], "playing")
        self.assertIn("On turn:", self.texts()[-1])

    async def test_stale_leave_table_refunds_a_seat_in_the_new_round(self):
        await self.settle_alice_win()
        settled = self.latest()
        self.rig("9", "10", "6", "7")
        await self.press(ALICE, "Play again (same bet)")
        self.assertEqual(self.balance(ALICE), 100)
        click = Click(self.bot, ALICE)
        await GameButton("close", settled["table_id"], settled["round_id"], settled["moves"]).callback(click)
        self.assertEqual(self.balance(ALICE), 110)
        self.assertIsNone(self.store.open_table_for(G, CH))
        self.assertEqual(click.response.edited[-1]["content"], "Table closed.")

    async def test_stale_leave_table_keeps_a_seat_in_a_round_in_play(self):
        await self.settle_alice_win()
        settled = self.latest()
        self.rig("9", "10", "6", "7")
        await self.press(ALICE, "Play again (same bet)")
        await self.press(ALICE, "Deal now")
        click = Click(self.bot, ALICE)
        await GameButton("close", settled["table_id"], settled["round_id"], settled["moves"]).callback(click)
        self.assertIn("A round is being played", click.replies[0])
        self.assertEqual(self.latest()["status"], "playing")
        self.assertIn(ALICE, self.games.table((G, settled["table_id"])).present)

    async def test_set_channel_edits_the_closed_tables_message(self):
        await self.bet(ALICE, 10)
        snap = self.latest()
        self.games.table((G, snap["table_id"]))
        self.assertTrue(await self.games.set_channel(G, CH, False))
        self.assertNotIn((G, snap["table_id"]), self.games.tables)
        self.assertEqual(self.channel.edits[-1][0], snap["message_id"])

    async def test_nothing_is_armed_after_close(self):
        await self.games.close()
        self.games.arm((G, 1), "join", 1)
        self.assertEqual(self.games.timers, {})

    async def test_a_failing_timer_is_retried_once(self):
        self.rig("9", "10", "6", "7")
        self.games.JOIN_SECONDS = self.games.RETRY_SECONDS = 0.01
        real = self.games.on_join_timer
        calls = []

        async def flaky(*args):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("database is locked")
            await real(*args)
        self.games.on_join_timer = flaky
        with self.assertLogs(level="ERROR"):
            await self.bet(ALICE, 10)
            for _ in range(100):
                await asyncio.sleep(0.02)
                if self.latest()["status"] != "joining":
                    break
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.latest()["status"], "playing")


class RecoveryTests(GameCase):
    async def test_restart_refunds_unfinished_rounds_and_closes_every_table(self):
        self.rig("9", "10", "6", "7")
        await self.bet(ALICE, 10)  # a joining round with a live message
        joining = self.latest()
        other = Channel(200)
        self.bot.get_channel = {CH: self.channel, 200: other}.get
        self.store.set_game_channel(G, 200, True)
        playing = self.store.open_table(G, 200, "blackjack", BOB)
        self.store.join_round(G, playing["round_id"], BOB, 30)
        self.store.set_table_message(G, playing["table_id"], 8001)
        self.store.deal(G, playing["round_id"])
        other.deleted.add(8001)  # this message was deleted meanwhile
        self.store.set_game_channel(2, 300, True)  # another server: open table whose last round settled
        self.store.change_balance(2, ALICE, 100, "start", 1)
        done = self.store.open_table(2, 300, "blackjack", ALICE)
        self.store.join_round(2, done["round_id"], ALICE, 10)
        self.store.set_table_message(2, done["table_id"], 9001)
        self.store.deal(2, done["round_id"])
        self.store.play(2, done["round_id"], 0, "stand", "member", 0)
        self.assertEqual(self.store.latest_round(2, done["table_id"])["status"], "settled")
        self.assertEqual((self.balance(ALICE), self.balance(BOB)), (90, 70))
        self.bot.fetch_channel = mock.AsyncMock(side_effect=discord.NotFound(SimpleNamespace(status=404, reason="x"), "gone"))
        await self.games.recover()
        self.assertEqual((self.balance(ALICE), self.balance(BOB)), (100, 100))
        self.assertEqual(self.store.unfinished_rounds(), [])
        for guild, channel in ((G, CH), (G, 200), (2, 300)):
            self.assertIsNone(self.store.open_table_for(guild, channel))
        text = "This table closed when the bot restarted. Bets were refunded."
        self.assertEqual([(i, kw["content"], kw["view"]) for i, kw in self.channel.edits], [(joining["message_id"], text, None)])
        self.assertEqual(other.edits, [])
        self.assertEqual(self.store.latest_round(G, joining["table_id"])["status"], "cancelled")
        self.assertEqual(self.store.latest_round(G, playing["table_id"])["status"], "cancelled")

    async def test_recovery_with_nothing_open_is_quiet(self):
        with self.assertNoLogs(level="WARNING"):
            await self.games.recover()
        self.assertEqual(self.channel.edits, [])

    async def test_recovery_logs_counts_never_seeds(self):
        await self.bet(ALICE, 10)
        with self.assertLogs(level="INFO") as logs:
            await self.games.recover()
        output = "\n".join(logs.output)
        self.assertIn("1 rounds refunded, 1 tables closed", output)
        self.assertNotIn(SEED, output)

    async def test_recover_reaches_unloaded_channels_through_fetch(self):
        await self.bet(ALICE, 10)
        self.bot.get_channel = lambda channel_id: None
        self.bot.fetch_channel = mock.AsyncMock(return_value=self.channel)
        await self.games.recover()
        self.bot.fetch_channel.assert_awaited_once_with(CH)
        self.assertEqual(len(self.channel.edits), 1)


class LifecycleTests(GameCase):
    async def test_close_cancels_every_timer(self):
        await self.bet(ALICE, 10)
        tasks = list(self.games.timers.values())
        self.assertEqual(len(tasks), 1)
        await self.games.close()
        self.assertTrue(all(task.cancelled() for task in tasks))
        self.assertEqual(self.games.timers, {})

    async def test_bot_close_cancels_the_game_timers(self):
        await self.bet(ALICE, 10)
        task = next(iter(self.games.timers.values()))
        with mock.patch.object(SkitBot.__mro__[1], "close", new=mock.AsyncMock()):
            self.bot.models.close = mock.AsyncMock()
            self.store.close = lambda: None
            await self.bot.close()
        self.assertTrue(task.cancelled())

    async def test_one_timer_per_table_and_it_is_replaced_not_stacked(self):
        self.rig("9", "10", "6", "7")
        await self.bet(ALICE, 10)
        first = next(iter(self.games.timers.values()))
        await self.press(ALICE, "Deal now")
        second = next(iter(self.games.timers.values()))
        await asyncio.sleep(0)
        self.assertTrue(first.cancelled())
        self.assertIsNot(first, second)
        self.assertEqual(len(self.games.timers), 1)

    async def test_the_channel_skit_lock_is_never_taken(self):
        self.rig("9", "10", "6", "7")
        self.bot.channel_locks = mock.Mock(side_effect=AssertionError("skit lock"))
        self.bot.channel_locks.lock = mock.Mock(side_effect=AssertionError("skit lock"))
        await self.bet(ALICE, 10)
        await self.press(ALICE, "Deal now")

    async def test_unknown_game_error_text_is_never_a_raw_exception(self):
        with mock.patch.object(self.store, "join_round", side_effect=GameError("The table is full.")):
            click = await self.bet(ALICE, 10)
        self.assertEqual(click.replies, ["The table is full."])
        self.assertIsNone(self.store.open_table_for(G, CH))
        self.assertEqual(self.balance(ALICE), 100)


if __name__ == "__main__":
    unittest.main()
