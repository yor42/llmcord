"""FEAT-58 (D42): Yut on Discord, humans only: lobby, table thread, board image, throws, moves, timers, settlement, recovery, wiring.

Offline: fake channels, threads and interactions from ``helpers``, a fixed seed, the pure engine (``games.yut``) as the oracle, an in-memory
store. Timers are driven by calling their hooks, never by waiting. Modelled on ``test_onecard_discord.py``.
"""
import unittest
from unittest import mock

import discord
from helpers import FakeGameChannel, FakeGameClick, invoke, make_settings, not_found, thread_message
from test_doubt_discord import blob, button, item_of, labels, private, says, selects, views

from llmcord_core import i18n
from llmcord_core.admin_store import GameError
from llmcord_core.games_discord import STALE
from llmcord_core.discord_bot import SkitBot
from llmcord_core.games import registry, yut, yut_render
from llmcord_core.games_yut_discord import YutButton
from llmcord_core.i18n import terms

G, CH = 1, 100
ALICE, BOB, CARA = 5, 6, 7
ANTE = 10
SEED = "yut-ds-1"


def first_throw(seed):
    return yut.throw_result(seed, 0)[0]


def good_seed():
    """A seed whose first throw is a plain 도/개/걸: the thrower then has one banked throw to move."""
    for i in range(500):
        seed = f"yut-ds-{i}"
        if first_throw(seed) in ("do", "gae", "geol"):
            return seed
    raise AssertionError("no seed")


class YutCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bot = SkitBot(make_settings())
        self.store, self.yut = self.bot.store, self.bot.yut
        self.addCleanup(self.store.close)
        self.seed = good_seed()
        self.channel = self.ch = FakeGameChannel(CH)
        self.bot.get_channel = lambda ident: self.channel if ident == CH else self.channel.threads.get(ident)
        self.bot.fetch_channel = mock.AsyncMock(side_effect=self.fetch_channel)
        patcher = mock.patch("llmcord_core.admin_store.new_seed", side_effect=lambda: self.seed)
        patcher.start()
        self.addCleanup(patcher.stop)
        s = self.store
        s.set_game_channel(G, CH, True)
        s.set_daily_settings(G, 0, 0, 7)
        s.set_game_rules(G, "yut", {"ante": ANTE})
        for user in (ALICE, BOB, CARA):
            s.change_balance(G, user, 100, "start", 1)
        s.set_game_character_talk(G, False)
        for name in ("LOBBY_SECONDS", "ARCHIVE_SECONDS", "BOARD_DEBOUNCE_SECONDS"):
            setattr(self.yut, name, 3600)
        self.yut.runtime.retry_seconds = 3600
        self.addAsyncCleanup(self.yut.close)
        self.addAsyncCleanup(self.bot.games.close)
        self.addAsyncCleanup(self.bot.doubt.close)

    async def fetch_channel(self, ident):
        found = self.bot.get_channel(ident)
        if found is None:
            raise not_found("Unknown Channel", 10003)
        return found

    def click(self, user, *, channel=None, message=None):
        return FakeGameClick(self.bot, user, channel=channel or self.ch, message=message)

    async def start(self, user=ALICE, game="yut", favorites=False):
        click = self.click(user)
        await invoke(self.bot, "game start", click, game=game, bet=None, favorites=favorites)
        return click

    async def push(self, user, label, message):
        click = self.click(user, channel=message.host, message=message)
        await button(message, label).callback(click)
        return click

    def table(self):
        return self.store.open_table_for(G, CH)

    def row(self):
        return self.table() or self.store.one("SELECT * FROM game_tables ORDER BY id DESC")

    def snap(self):
        return self.store.latest_round(G, self.table()["id"])

    def lobby(self):
        return self.ch.messages[self.row()["message_id"]]

    def thread(self):
        return self.ch.threads[self.row()["thread_id"]]

    def board(self):
        return self.thread().messages[self.table()["board_message_id"]]

    def turn(self):
        return self.thread().messages[self.table()["turn_message_id"]]

    def log(self):
        return [(r["seat_index"], r["move"], r["actor"]) for r in self.store.db.execute("SELECT * FROM game_moves ORDER BY id")]

    def texts(self):
        return [m.content for m in self.thread().sent if m.content]

    async def playing(self):
        await self.start()
        await self.push(BOB, "Join", self.lobby())
        await self.push(ALICE, "Start", self.lobby())
        self.tid = self.table()["id"]

    async def throw(self, user=ALICE):
        return await self.push(user, "Throw", self.turn())

    async def expire(self):
        snap = self.snap()
        await self.yut.on_turn_timer(G, self.tid, snap["round_id"], snap["moves"])


class LobbyAndStartTests(YutCase):
    async def test_start_opens_a_lobby_thread_and_takes_the_ante(self):
        click = await self.start()
        content, kwargs = click.response.sent[0]
        self.assertFalse(kwargs.get("ephemeral"))
        self.assertIn("Yut", content)
        self.assertEqual(set(labels(self.lobby())), {"Join", "Leave", "Start", "How to play"})
        self.assertEqual(self.snap()["game"], "yut")
        self.assertEqual(self.store.balance(G, ALICE), 100 - ANTE)
        self.assertTrue(self.thread().name.startswith("Yut"))
        self.assertIn(((G, self.table()["id"]), "lobby"), self.yut.runtime.timers)

    async def test_one_seat_cannot_start_and_favorites_get_a_note(self):
        click = await self.start(favorites=True)
        self.assertTrue(any("can't sit at a Yut table" in t for t in says(click)))
        self.assertTrue(item_of(button(self.lobby(), "Start")).disabled)

    async def test_start_posts_the_pinned_board_with_a_png_and_the_text_fallback(self):
        await self.playing()
        board = self.board()
        self.assertTrue(board.pinned)
        self.assertIn("file", board.kwargs)
        self.assertIn(f"<@{ALICE}>: on board - · at start 4 · home 0", board.content)
        self.assertEqual(set(labels(self.turn())), {"Throw", "Forfeit"})
        self.assertEqual(self.snap()["status"], "playing")

    async def test_a_render_failure_still_posts_the_text_board(self):
        with mock.patch.object(yut_render, "render_board_cached", side_effect=RuntimeError("no pillow")):
            await self.playing()
        self.assertNotIn("file", self.board().kwargs)
        self.assertIn(f"<@{BOB}>: on board - · at start 4 · home 0", self.board().content)

    async def test_help_and_rules_are_private(self):
        await self.start()
        click = await self.push(CARA, "How to play", self.lobby())
        self.assertTrue(private(click))
        self.assertIn("Yut in short", blob(click))

    async def test_game_rules_answers_privately(self):
        click = self.click(ALICE)
        await invoke(self.bot, "game rules", click, game="yut")
        self.assertIn("Yut in short", blob(click))

    async def test_a_second_table_in_the_channel_names_the_open_yut_table(self):
        await self.start()
        click = self.click(BOB)
        await invoke(self.bot, "game start", click, game="blackjack", bet=5, favorites=False)
        self.assertIn("A Yut table is open in this channel.", says(click))
        self.assertEqual(self.store.open_table_for(G, CH)["game"], "yut")


class LobbyPressTests(YutCase):
    async def test_leaving_the_lobby_refunds_the_ante(self):
        await self.start()
        await self.push(BOB, "Join", self.lobby())
        self.assertEqual(self.store.balance(G, BOB), 100 - ANTE)
        await self.push(BOB, "Leave", self.lobby())
        self.assertEqual(self.store.balance(G, BOB), 100)
        self.assertEqual(len(self.snap()["seats"]), 1)

    async def test_a_button_from_another_guild_with_the_same_table_id_is_refused(self):
        await self.start()
        snap = self.snap()
        click = FakeGameClick(self.bot, BOB, guild_id=2, channel=self.ch)
        await self.yut.press(click, "join", snap["table_id"], snap["round_id"], 0)
        self.assertEqual(says(click), [STALE])
        self.assertEqual(len(self.snap()["seats"]), 1)
        self.assertEqual(self.store.balance(G, BOB), 100)

    async def test_the_board_is_localized_at_the_edge(self):
        await self.playing()
        text = self.yut.render_board((G, self.tid), self.snap(), t=i18n.Translator("ko"))
        self.assertNotIn("on board", text)
        self.assertIn("<@5>", text)


class PlayTests(YutCase):
    async def test_throw_posts_one_result_line_and_only_the_turn_seat_may_throw(self):
        await self.playing()
        click = await self.throw(BOB)
        self.assertEqual(says(click), ["It's not your turn."])
        self.assertEqual(self.log(), [])
        await self.throw(ALICE)
        self.assertEqual(self.log(), [(0, "throw", "member")])
        name, sticks = yut.throw_result(self.seed, 0)
        line = [m.content for m in self.thread().sent if "threw" in (m.content or "") and m.id != self.turn().id]
        self.assertEqual(len(line), 1)
        self.assertIn(f"({''.join('▮' if s else '▯' for s in sticks)})", line[0])
        self.assertEqual(set(labels(self.turn())), {"Move", "Forfeit"})
        self.assertIn("Banked throws", self.turn().content)

    async def test_a_stale_throw_button_is_refused(self):
        await self.playing()
        old = self.turn().view
        await self.throw(ALICE)
        click = self.click(ALICE, channel=self.thread())
        await self.yut.press(click, "throw", self.tid, self.snap()["round_id"], 0)
        self.assertEqual(self.log(), [(0, "throw", "member")])
        self.assertTrue(private(click))
        self.assertIsNotNone(old)

    async def test_the_move_picker_is_private_and_moves_a_piece(self):
        await self.playing()
        await self.throw(ALICE)
        click = await self.push(ALICE, "Move", self.turn())
        self.assertTrue(private(click))
        picker = views(click)[0]
        (select,) = selects(picker)
        values = [o.value for o in select.options]
        self.assertEqual(len(values), 1)  # four unstarted pieces are one distinct move
        self.assertTrue(values[0].startswith("move:0:"))
        other = self.click(BOB, channel=self.thread())
        await self.yut.play_from_picker(other, self.tid, picker.round_id, picker.moves, values[0])
        self.assertEqual(self.log(), [(0, "throw", "member")])  # Bob is not on turn
        done = self.click(ALICE, channel=self.thread())
        await self.yut.play_from_picker(done, self.tid, picker.round_id, picker.moves, values[0])
        self.assertEqual(self.log()[-1], (0, values[0], "member"))
        self.assertTrue(any("moved piece 1" in t for t in self.texts()))
        self.assertEqual(self.snap()["view"]["turn"], 1)
        self.assertEqual(set(labels(self.turn())), {"Throw", "Forfeit"})
        await self.yut.flush()

    async def test_move_is_refused_for_a_move_not_in_the_list(self):
        await self.playing()
        await self.throw(ALICE)
        snap = self.snap()
        click = self.click(ALICE, channel=self.thread())
        await self.yut.play_from_picker(click, self.tid, snap["round_id"], snap["moves"], "move:0:backdo")
        self.assertEqual(len(self.log()), 1)

    async def test_forfeit_asks_first_then_settles_and_pays_the_other_seat(self):
        await self.playing()
        click = await self.push(BOB, "Forfeit", self.turn())  # not on turn: forfeit is allowed any time
        self.assertTrue(private(click))
        self.assertEqual(self.log(), [])
        confirm = views(click)[0]
        await button_of(confirm).callback(self.click(BOB, channel=self.thread()))
        self.assertEqual(self.log(), [(1, "forfeit", "member")])
        self.assertEqual(self.store.balance(G, ALICE), 100 - ANTE + 2 * ANTE)
        self.assertEqual(self.store.balance(G, BOB), 100 - ANTE)
        self.assertIsNone(self.table())
        self.assertTrue(any("wins the game and takes the pot of 20" in t for t in self.texts()))

    async def test_a_full_game_settles_by_the_engine_and_pays_the_winner(self):
        await self.playing()
        for _ in range(3000):
            snap = self.snap() if self.table() else None
            if snap is None:
                break
            seat = snap["view"]["turn"]
            move = snap["legal"][seat][0]
            user = (ALICE, BOB)[seat]
            if move == "throw":
                await self.throw(user)
            else:
                picker_snap = snap
                await self.yut.play_from_picker(self.click(user, channel=self.thread()), self.tid, snap["round_id"], picker_snap["moves"], move)
        else:
            self.fail("the game did not finish")
        last = self.store.latest_round(G, self.tid)
        self.assertEqual(last["status"], "settled")
        winner = (ALICE, BOB)[last["view"]["winner"]]
        self.assertEqual(self.store.balance(G, winner), 100 - ANTE + 2 * ANTE)


def button_of(view):
    return next(c for c in view.children if isinstance(c, discord.ui.Button))


class TimerTests(YutCase):
    async def miss_until(self, seat):
        for _ in range(30):
            if self.snap()["view"]["turn"] == seat:
                return
            await self.expire()
        self.fail("the turn never reached the seat")

    async def test_a_missed_turn_plays_every_decision_of_the_turn_without_forfeiting(self):
        await self.playing()
        await self.expire()
        self.assertEqual(self.log(), [(0, "throw", "timeout")])
        self.assertTrue(any("a throw was made for them" in t for t in self.texts()))
        await self.expire()  # the move of the same turn
        self.assertEqual([m[2] for m in self.log()], ["timeout", "timeout"])
        self.assertNotIn((0, "forfeit", "timeout"), self.log())
        self.assertEqual(self.snap()["view"]["turn"], 1)

    async def test_an_extra_throw_after_yut_or_mo_is_still_the_same_missed_turn(self):
        self.seed = next(f"yut-x-{i}" for i in range(2000) if first_throw(f"yut-x-{i}") in ("yut", "mo"))
        await self.playing()
        await self.expire()
        self.assertEqual(self.snap()["view"]["phase"], "throw")
        await self.expire()  # the extra throw
        await self.expire()
        self.assertNotIn("forfeit", [m[1] for m in self.log()])
        self.assertEqual(self.snap()["status"], "playing")
        self.assertTrue(all(m[2] == "timeout" for m in self.log()))

    async def test_two_missed_turns_in_a_row_forfeit_and_the_other_seat_takes_the_pot(self):
        await self.playing()
        await self.miss_until(1)  # Alice's first missed turn
        await self.miss_until(0)  # Bob misses his turn too
        await self.expire()  # Alice times out again right after missing her previous turn
        self.assertEqual(self.log()[-1], (0, "forfeit", "timeout"))
        self.assertEqual(self.store.balance(G, BOB), 100 - ANTE + 2 * ANTE)
        self.assertTrue(any("missed two turns in a row" in t for t in self.texts()))

    async def play_turn(self, seat):
        user = (ALICE, BOB)[seat]
        while self.snap()["view"]["turn"] == seat:
            snap = self.snap()
            move = snap["legal"][seat][0]
            if move == "throw":
                await self.throw(user)
            else:
                await self.yut.play_from_picker(self.click(user, channel=self.thread()), self.tid, snap["round_id"], snap["moves"], move)

    async def test_a_turn_without_a_timeout_resets_the_count(self):
        await self.playing()
        await self.miss_until(1)  # Alice misses
        await self.play_turn(1)  # Bob plays by hand
        await self.play_turn(0)  # Alice plays by hand: her count resets
        await self.play_turn(1)
        await self.expire()  # Alice times out again, but her previous turn was not missed
        self.assertNotIn((0, "forfeit", "timeout"), self.log())
        self.assertEqual(self.log()[-1][2], "timeout")
        self.assertFalse(self.yut._missed_previous_turn(G, self.snap()["round_id"], 0))

    async def test_a_refused_auto_move_falls_back_to_forfeit(self):
        await self.playing()
        real = self.store.play
        def refuse(guild_id, round_id, seat, move, actor, expected):
            if move != "forfeit":
                raise GameError("no")
            return real(guild_id, round_id, seat, move, actor, expected)
        with mock.patch.object(self.store, "play", side_effect=refuse):
            await self.expire()
        self.assertEqual(self.log(), [(0, "forfeit", "timeout")])

    async def test_a_timer_and_a_press_for_the_same_move_apply_once(self):
        await self.playing()
        snap = self.snap()
        await self.expire()
        click = self.click(ALICE, channel=self.thread())
        await self.yut.press(click, "throw", self.tid, snap["round_id"], snap["moves"])
        self.assertEqual(self.log(), [(0, "throw", "timeout")])
        self.assertTrue(says(click))

    async def test_a_missed_move_plays_the_first_legal_move(self):
        await self.playing()
        await self.throw(ALICE)
        snap = self.snap()
        first = next(m for m in snap["legal"][0] if m != "forfeit")
        await self.expire()
        self.assertEqual(self.log()[-1], (0, first, "timeout"))
        self.assertTrue(any("a move was made for them" in t for t in self.texts()))

    async def test_a_stale_timer_does_nothing_and_each_move_rearms_the_timer(self):
        await self.playing()
        stale = self.snap()
        await self.throw(ALICE)
        await self.yut.on_turn_timer(G, self.tid, stale["round_id"], stale["moves"])
        self.assertEqual(self.log(), [(0, "throw", "member")])
        self.assertIn(((G, self.tid), "turn"), self.yut.runtime.timers)

    async def test_the_lobby_timer_closes_an_unstarted_table_and_refunds(self):
        await self.start()
        snap = self.snap()
        await self.yut.on_lobby_timer(G, snap["table_id"], snap["round_id"])
        self.assertIsNone(self.table())
        self.assertEqual(self.store.balance(G, ALICE), 100)


class RecoverAndCloseTests(YutCase):
    async def test_recover_refunds_an_unfinished_round_and_closes_the_table(self):
        await self.playing()
        self.yut.tables.clear()
        await self.yut.recover()
        self.assertIsNone(self.table())
        self.assertEqual(self.store.balance(G, ALICE), 100)
        self.assertEqual(self.store.balance(G, BOB), 100)
        self.assertIn("closed when the bot restarted", self.lobby().content)

    async def test_games_off_in_the_channel_closes_the_yut_table(self):
        await self.playing()
        closed, _paint = await self.bot.games.switch_channel(G, CH, False)
        self.assertIsNone(self.table())
        self.assertTrue(closed)
        self.assertEqual(self.store.balance(G, ALICE), 100)

    async def test_a_dashboard_close_is_swept(self):
        await self.playing()
        self.store.close_table(G, self.tid)
        await self.yut.sweep()
        self.assertEqual((self.store.balance(G, ALICE), self.store.balance(G, BOB)), (100, 100))
        self.assertIn("table was closed", self.lobby().content)


class WiringTests(YutCase):
    async def test_registry_entry(self):
        game = registry.get("yut")
        self.assertEqual((game.min_seats, game.max_seats, game.ante, game.cash), (yut.MIN_SEATS, yut.MAX_SEATS, True, False))

    async def test_button_template_and_callback(self):
        self.assertIsNotNone(YutButton.__discord_ui_compiled_template__.fullmatch("llmcord:yut:throw:3:4:5"))
        self.assertIsNone(YutButton.__discord_ui_compiled_template__.fullmatch("llmcord:onecard:throw:3:4:5"))
        with mock.patch.object(self.yut, "press", new=mock.AsyncMock()) as press:
            interaction = self.click(BOB)
            await YutButton("throw", 3, 4, 5).callback(interaction)
        press.assert_awaited_once_with(interaction, "throw", 3, 4, 5)

    async def test_setup_registers_the_button_and_recovers(self):
        self.bot.tree.sync = mock.AsyncMock()
        self.bot.games.recover, self.bot.doubt.recover, self.yut.recover = mock.AsyncMock(), mock.AsyncMock(), mock.AsyncMock()
        await self.bot.setup_hook()
        self.addCleanup(self.bot.cleanup_task.cancel)
        self.assertIn(YutButton, list(self.bot._connection._view_store._dynamic_items.values()))
        self.yut.recover.assert_awaited_once()

    async def test_on_message_calls_the_yut_hooks_and_survives_their_failure(self):
        hook = self.yut.on_thread_message = mock.AsyncMock(side_effect=RuntimeError("boom"))
        self.yut.note_thread_message = mock.AsyncMock(side_effect=RuntimeError("boom"))
        await self.bot.on_message(thread_message(self.ch, ALICE, "hello"))
        hook.assert_awaited_once()

    async def test_thread_messages_are_counted(self):
        await self.playing()
        await self.yut.note_thread_message(thread_message(self.thread(), BOB, "gg"))
        self.assertEqual(self.yut.table((G, self.tid)).talk, 1)

    async def test_closing_the_bot_cancels_the_timers(self):
        await self.start()
        task = self.yut.runtime.timers[((G, self.table()["id"]), "lobby")]
        with mock.patch.object(SkitBot.__mro__[1], "close", new=mock.AsyncMock()):
            self.bot.models.close = mock.AsyncMock()
            self.store.close = lambda: None
            await self.bot.close()
        self.assertTrue(task.cancelled())

    async def test_the_public_table_view_covers_yut_for_spectators(self):
        from llmcord_core.table_view import public_table_view
        await self.playing()
        view = public_table_view(self.bot, G, self.table()["thread_id"], CH)
        self.assertEqual((view["game"], view["seated_character_ids"]), ("yut", []))
        self.assertIn("Yut", view["text"])
        self.assertNotIn("Seed", view["text"])

    async def test_throw_names_in_english_and_korean(self):
        en, ko = i18n.english(), i18n.Translator("ko")
        self.assertEqual([en.render(terms.YUT_THROW[n]) for n in yut.THROWS], ["Do", "Gae", "Geol", "Yut", "Mo", "Back-do"])
        self.assertEqual([ko.render(terms.YUT_THROW[n]) for n in yut.THROWS], ["도", "개", "걸", "윷", "모", "빽도"])


if __name__ == "__main__":
    unittest.main()
