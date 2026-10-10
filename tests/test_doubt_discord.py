"""FEAT-21 part B1: I Doubt It on Discord (`/game`, lobby, table thread, turns, doubt window, timers, settlement, recovery).

TDD-red tests written before the implementation. Offline: fake channels, threads and interactions from ``helpers``, a
fixed seed, and the engine used as an oracle. Timers are driven by calling their hooks, never by waiting.

Assumed API (pinned here, to be created in ``llmcord_core/games_doubt_discord.py`` and wired in ``discord_bot.py``):

  ``bot.doubt = DoubtTables(bot, clock=time.time)``; attributes: ``locks`` (WeakValueDictionary keyed ``(guild_id, table_id)``,
  held only for store work), ``timers`` (dict keyed ``(guild_id, table_id, kind)``, kind in ``lobby window turn character archive``),
  class constants ``LOBBY_SECONDS=300 WINDOW_SECONDS=15 CHARACTER_SECONDS=2 ARCHIVE_SECONDS=600 BOARD_DEBOUNCE_SECONDS=1 BUMP_AFTER=8``.
  The member turn timeout comes from the round's rules (``turn_timeout``).
  Slash commands (a ``game`` group, plus the unchanged ``/blackjack``):
    ``/game start game:<blackjack|doubt> [bet] [favorites]`` (blackjack delegates to ``GameTables.play_command``; doubt calls
    ``DoubtTables.start_command(interaction, favorites)``), ``/game hand`` (``hand_command(interaction)``), ``/game rules game:<..>``.
  Components: ``DoubtButton(action, table_id, round_id, moves, label="", style=...)``, a DynamicItem with custom id
    ``llmcord:doubt:<action>:<table>:<round>:<moves>`` (no seat in it); ``callback`` calls ``interaction.client.doubt.press(
    interaction, action, table_id, round_id, moves)``. Actions: lobby ``join leave fill start help``; turn message ``doubt hand``.
    Button labels: Join, Leave, Fill seats, Start, How to play; Doubt, My hand.
  ``await doubt.play_from_picker(interaction, table_id, round_id, moves, counts)``: ``counts`` maps rank -> how many of it to play
    (the seat comes from ``store.member_seat_index`` for the pressing user; ``moves`` is the move count the picker was built from).
  ``await doubt.note_thread_message(message)`` (called by ``SkitBot.on_message`` for every human message, before the binding check).
  Timer hooks (all stale-safe and lock-disciplined like blackjack's): ``on_lobby_timer(guild_id, table_id, round_id)``,
    ``on_window_timer / on_turn_timer / on_character_timer(guild_id, table_id, round_id, moves)``, ``on_archive_timer(guild_id, table_id)``.
    ``on_character_timer`` lets character seats doubt an open claim first (policy_doubt), then plays for a character seat on turn.
  ``await doubt.flush()`` sends any pending debounced board edit now; ``await doubt.recover()``; ``await doubt.close()``.
  Hand display: the ephemeral hand shows every card as rank+suit (``7♠``), grouped by rank; a picker (discord Select views, <=25 options each)
    appears only when the pressing member is on turn.
  Members are named with mentions (``<@id>``), characters by name; "(with <@id>)" marks a brought favorite, "(house)" a filled seat.
Store follow-up pinned here: ``unfinished_rounds()`` rows carry ``thread_id``.
"""
import asyncio
import functools
import json
import math
import unittest
from unittest import mock

import discord
from helpers import FakeGameChannel, FakeGameClick, forbidden, invoke, make_settings, not_found, thread_message
from test_doubt_store import pick, pure_winner

from llmcord_core.discord_bot import SkitBot
from llmcord_core.games import blackjack as bj
from llmcord_core.games import doubt
from llmcord_core.games.base import Seat
from llmcord_core.games_discord import NOT_SEATED, OFF_HERE, STALE, GameButton

try:  # imported softly so every test fails on its own until the module exists
    from llmcord_core.games_doubt_discord import DoubtButton
except ImportError:  # pragma: no cover - the red state
    DoubtButton = None

G, CH, CH2 = 1, 100, 101
ALICE, BOB, CARA, DAN = 5, 6, 7, 8
MEMBERS = (ALICE, BOB, CARA)
ANTE = 10
NO_MENTIONS = discord.AllowedMentions.none().to_dict()
ANTE_REASON = "I Doubt It ante (table 1, round 1)"
PAYOUT_REASON = "I Doubt It payout (table 1, round 1)"

DOUBT = None  # the DoubtTables under test, so fake Discord calls can tell whether a table lock is held
HTTP = []  # (kind, a table lock was held) for every fake Discord call


def http(kind):
    locks = list(DOUBT.runtime.locks.values()) + list(DOUBT.bot.games.runtime.locks.values()) if DOUBT is not None else []
    HTTP.append((kind, any(lock.locked() for lock in locks)))


@functools.lru_cache(maxsize=None)
def table_seed():
    """A 3-seat deal where seat 0 holds two or more aces and four or more other cards, seat 1 holds an ace and a 2."""
    for i in range(5000):
        seed = f"doubt-b1-{i}"
        hands = doubt.new(seed, [Seat(k, "member", k + 1, 0) for k in range(3)]).hands
        aces = [sum(c[0] == "A" for c in h) for h in hands]
        if aces[0] >= 2 and aces[1] >= 1 and sum(c[0] != "A" for c in hands[0]) >= 4 and any(c[0] == "2" for c in hands[1]):
            return seed
    raise AssertionError("no suitable seed")


def card_text(card):
    return f"{card[0]}{card[1]}"


def counts_of(cards):
    held = {}
    for rank, _ in cards:
        held[rank] = held.get(rank, 0) + 1
    return {rank: held[rank] for rank in doubt.RANKS if rank in held}


def parse_move(move):
    return {rank: int(n) for rank, _, n in (part.rpartition("x") for part in move[5:].split(","))}


def item_of(child):
    return getattr(child, "item", child)


def labels(message):
    return [item_of(child).label for child in message.view.children] if message is not None and message.view else []


def button(message, label):
    for child in message.view.children if message.view else ():
        if item_of(child).label == label:
            return child
    raise AssertionError(f"no {label!r} button among {labels(message)}")


def says(click):
    return [content for content, _ in click.response.sent + click.followup.sent if content]


def private(click):
    entries = click.response.sent + click.followup.sent
    return bool(entries) and all(kw.get("ephemeral") or kw.get("_edit") for _, kw in entries)


def views(click):
    return [kw["view"] for _, kw in click.response.sent + click.followup.sent if kw.get("view") is not None]


def blob(click):
    """Everything a click's private replies show: texts plus the labels and descriptions of every select option and button."""
    parts = list(says(click))
    for view in views(click):
        for child in view.children:
            parts.append(str(getattr(item_of(child), "label", "") or ""))
            for option in getattr(child, "options", ()):
                parts += [option.label, option.description or "", str(option.value)]
    return "\n".join(parts)


def selects(view):
    return [child for child in view.children if isinstance(child, discord.ui.Select)]


class DoubtCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        global DOUBT
        self.bot = SkitBot(make_settings())
        self.store, self.games = self.bot.store, self.bot.games
        self.doubt = DOUBT = getattr(self.bot, "doubt", None)
        HTTP.clear()
        self.seed = table_seed()
        self.channel, self.channel2 = FakeGameChannel(CH), FakeGameChannel(CH2)
        self.ch = self.channel
        for channel in (self.channel, self.channel2):
            channel.on_call = http
        self.bot.get_channel = self.find_channel
        self.bot.fetch_channel = mock.AsyncMock(side_effect=self.fetch_channel)
        patcher = mock.patch("llmcord_core.admin_store.new_seed", side_effect=lambda: self.seed)
        patcher.start()
        self.addCleanup(patcher.stop)
        s = self.store
        for channel_id in (CH, CH2):
            s.set_game_channel(G, channel_id, True)
        s.set_daily_settings(G, 0, 0, 7)
        s.set_game_rules(G, "doubt", {"ante": ANTE})
        self.world = s.create_space(G, "W", "world")
        s.bind_channel(G, CH, self.world)  # CH is bound to a world; CH2 stays unbound (humans-only tables)
        self.ann, self.bo, self.cy = (s.create_character(G, self.world, name) for name in ("Ann", "Bo", "Cy"))
        for character in (self.ann, self.bo, self.cy):
            s.change_character_balance(G, character, 100, "start", 1)
        s.set_cast(CH, None, [self.ann, self.bo, self.cy])
        s.add_favorite(G, ALICE, self.ann)
        for user in (ALICE, BOB, CARA, DAN):
            s.change_balance(G, user, 100, "start", 1)
        self.addCleanup(s.close)
        s.set_game_character_talk(G, False)  # B1 tests pin the talk-off (policy only) behavior
        self.bot.models.structured_compiled = mock.AsyncMock(side_effect=RuntimeError("no model in B1 tests"))  # and nothing reaches a real gateway
        if self.doubt is not None:
            for name in ("LOBBY_SECONDS", "WINDOW_SECONDS", "CHARACTER_SECONDS", "ARCHIVE_SECONDS", "BOARD_DEBOUNCE_SECONDS"):
                setattr(self.doubt, name, 3600)
            self.doubt.runtime.retry_seconds = 3600
            self.addAsyncCleanup(self.doubt.close)
        self.addAsyncCleanup(self.games.close)

    # --- fake Discord ------------------------------------------------------------------------

    def find_channel(self, ident):
        for channel in (self.channel, self.channel2):
            if channel.id == ident:
                return channel
            if ident in channel.threads:
                return channel.threads[ident]
        return None

    async def fetch_channel(self, ident):
        found = self.find_channel(ident)
        if found is None:
            raise not_found("Unknown Channel", 10003)
        return found

    def click(self, user, *, channel=None, message=None):
        return FakeGameClick(self.bot, user, channel=channel or self.ch, message=message, on_call=http)

    # --- driving the bot ---------------------------------------------------------------------

    async def start(self, user=ALICE, *, channel=None, favorites=False, bet=None):
        click = self.click(user, channel=channel or self.ch)
        await invoke(self.bot, "game start", click, game="doubt", bet=bet, favorites=favorites)
        return click

    async def push(self, user, label, message):
        """The user presses a button of the message the way Discord delivers it: a component click in the message's channel."""
        click = self.click(user, channel=message.host, message=message)
        await button(message, label).callback(click)
        return click

    async def lobby3(self):
        await self.start()
        for user in (BOB, CARA):
            await self.push(user, "Join", self.lobby())

    async def playing(self):
        await self.lobby3()
        await self.push(ALICE, "Start", self.lobby())

    async def pick_play(self, user, counts, moves=None):
        snap, click = self.snap(), self.click(user, channel=self.thread())
        await self.doubt.play_from_picker(click, snap["table_id"], snap["round_id"], snap["moves"] if moves is None else moves, counts)
        return click

    async def doubt_press(self, user, moves=None):
        snap, click = self.snap(), self.click(user, channel=self.thread())
        await DoubtButton("doubt", snap["table_id"], snap["round_id"], snap["moves"] if moves is None else moves).callback(click)
        return click

    # --- reading state -----------------------------------------------------------------------

    def table(self, channel_id=None):
        return self.store.open_table_for(G, channel_id or self.ch.id)

    def snap(self):
        return self.store.latest_round(G, self.table()["id"])

    def lobby(self):
        return self.ch.messages[self.table()["message_id"]]

    def thread(self):
        return self.ch.threads[self.table()["thread_id"]]

    def board(self):
        return self.thread().messages[self.table()["board_message_id"]]

    def turn(self):
        return self.thread().messages[self.table()["turn_message_id"]]

    def hand(self, seat):
        snap = self.snap()
        return list(self.store.seat_view(G, snap["round_id"], seat).hand)

    def moves(self):
        return [(r["seat_index"], r["move"], r["actor"]) for r in self.store.db.execute("SELECT * FROM game_moves ORDER BY id")]

    def seats(self, snap=None):
        return [(s["kind"], s["ref_id"], s["brought_by"]) for s in (snap or self.snap())["seats"]]

    def balance(self, user):
        return self.store.balance(G, user)

    def engine_state(self, snap=None):
        snap = snap or self.snap()
        db = self.store.db
        rnd = db.execute("SELECT * FROM game_rounds WHERE id=?", (snap["round_id"],)).fetchone()
        seats = [Seat(r["seat_index"], r["kind"], r["ref_id"], r["stake"]) for r in
                 db.execute("SELECT * FROM game_seats WHERE round_id=? ORDER BY seat_index", (snap["round_id"],))]
        moves = [(r["seat_index"], r["move"]) for r in db.execute("SELECT * FROM game_moves WHERE round_id=? ORDER BY id", (snap["round_id"],))]
        return doubt.replay(rnd["seed"], seats, moves, json.loads(rnd["rules_json"]))

    def store_play(self, seat, move, actor="member"):
        """A move made straight in the store (the bot reads the table fresh on its next step)."""
        snap = self.snap()
        return self.store.play(G, snap["round_id"], seat, move, actor, snap["moves"])

    def any_card(self, seat):
        return {self.hand(seat)[0][0]: 1}

    def thread_texts(self, thread=None):
        return [m.content for m in (thread or self.thread()).sent if not m.deleted and m.content]

    async def play_out(self, limit=2000):
        """Plays the table's scripted game to its end through the bot (closes, doubts and plays as the engine's scripted pick says)."""
        thread, table_id = self.thread(), self.table()["id"]  # the table is closed once the game is over, so read it by id
        for n in range(limit):
            snap = self.store.latest_round(G, table_id)
            state = self.engine_state(snap)
            if state.finished:
                return n
            self.assertEqual(snap["moves"], n)
            seat, move = pick(state, self.seed, n)
            tid, rid = snap["table_id"], snap["round_id"]
            if move == "close":
                await self.doubt.on_window_timer(G, tid, rid, n)
            elif move == "doubt":
                await DoubtButton("doubt", tid, rid, n).callback(self.click(MEMBERS[seat], channel=thread))
            else:
                await self.doubt.play_from_picker(self.click(MEMBERS[seat], channel=thread), tid, rid, n, parse_move(move))
        self.fail("the scripted game did not finish")


class LobbyTests(DoubtCase):
    async def test_start_posts_a_public_lobby_and_seats_the_opener(self):
        click = await self.start()
        content, kwargs = click.response.sent[0]
        self.assertFalse(kwargs.get("ephemeral"))
        self.assertEqual(kwargs["allowed_mentions"].to_dict(), NO_MENTIONS)
        self.assertIn("I Doubt It", content)
        self.assertIn(f"<@{ALICE}>", content)
        self.assertIn("ante", content.lower())
        self.assertIn("pot", content.lower())
        lobby = self.lobby()
        self.assertEqual(set(labels(lobby)), {"Join", "Leave", "Fill seats", "Start", "How to play"})
        self.assertTrue(item_of(button(lobby, "Start")).disabled)
        snap = self.snap()
        self.assertEqual((snap["game"], snap["status"], snap["rules"]["ante"]), ("doubt", "joining", ANTE))
        self.assertEqual(self.seats(snap), [("member", ALICE, None)])
        self.assertEqual(self.balance(ALICE), 100 - ANTE)
        self.assertIn(((G, snap["table_id"]), "lobby"), self.doubt.runtime.timers)

    async def test_a_public_table_thread_is_opened_on_the_lobby_message(self):
        await self.start()
        table = self.table()
        self.assertEqual(list(self.channel.threads), [table["thread_id"]])
        thread = self.thread()
        self.assertTrue(thread.name.startswith("I Doubt It"))
        self.assertIn(f"table {table['id']}", thread.name)
        self.assertEqual(self.store.table_for_thread(G, thread.id)["id"], table["id"])
        self.assertEqual(self.channel.calls.count("create_thread"), 1)

    async def test_thread_creation_failure_closes_the_table_with_refunds_and_names_the_permissions(self):
        self.channel.thread_error = forbidden()
        click = await self.start()
        self.assertIsNone(self.table())
        self.assertEqual(self.balance(ALICE), 100)
        self.assertEqual(self.store.unfinished_rounds(), [])
        warning = [(content, kw) for content, kw in click.response.sent + click.followup.sent if content and "Create Public Threads" in content]
        self.assertEqual(len(warning), 1)
        self.assertIn("Send Messages in Threads", warning[0][0])
        self.assertTrue(warning[0][1].get("ephemeral"))

    async def test_a_bet_for_i_doubt_it_gets_the_ante_note(self):
        click = await self.start(bet=25)
        self.assertIn("I Doubt It uses the server's ante; no bet needed.", says(click))
        self.assertNotEqual(self.balance(ALICE), 75)

    async def test_refused_in_non_game_channels_and_when_the_game_is_off(self):
        click = await self.start(channel=FakeGameChannel(555))
        self.assertEqual(says(click), [OFF_HERE])
        self.store.set_game_enabled(G, "doubt", False)
        click = await self.start()
        self.assertEqual(says(click), ["I Doubt It is off on this server."])
        self.assertTrue(private(click))
        self.assertIsNone(self.table())
        self.assertEqual(self.balance(ALICE), 100)

    async def test_join_and_leave_move_the_ante_and_the_lobby_list(self):
        await self.start()
        lobby = self.lobby()
        click = await self.push(BOB, "Join", lobby)
        self.assertTrue(private(click))
        self.assertEqual(self.seats(), [("member", ALICE, None), ("member", BOB, None)])
        self.assertEqual(self.balance(BOB), 100 - ANTE)
        self.assertIn(f"<@{BOB}>", lobby.content)
        again = await self.push(BOB, "Join", lobby)
        self.assertIn("already", " ".join(says(again)))
        self.assertEqual(self.balance(BOB), 100 - ANTE)
        await self.push(BOB, "Leave", lobby)
        self.assertEqual(self.balance(BOB), 100)
        self.assertEqual(self.seats(), [("member", ALICE, None)])
        self.assertNotIn(f"<@{BOB}>", lobby.content)
        stranger = await self.push(DAN, "Leave", lobby)
        self.assertTrue(says(stranger) and private(stranger))
        self.assertEqual(self.seats(), [("member", ALICE, None)])

    async def test_joining_without_the_ante_is_refused(self):
        await self.start()
        self.store.change_balance(G, DAN, -95, "spend", 1)
        click = await self.push(DAN, "Join", self.lobby())
        self.assertIn("need", " ".join(says(click)))
        self.assertEqual(self.seats(), [("member", ALICE, None)])
        self.assertEqual(self.balance(DAN), 5)

    async def test_start_is_for_the_host_and_needs_three_seats(self):
        await self.start()
        lobby = self.lobby()
        await self.push(BOB, "Join", lobby)
        self.assertTrue(item_of(button(lobby, "Start")).disabled)
        click = await self.push(ALICE, "Start", lobby)
        self.assertIn("3", " ".join(says(click)))
        self.assertEqual(self.snap()["status"], "joining")
        await self.push(CARA, "Join", lobby)
        self.assertFalse(item_of(button(lobby, "Start")).disabled)
        click = await self.push(BOB, "Start", lobby)
        self.assertIn("host", " ".join(says(click)).lower())
        self.assertTrue(private(click))
        self.assertEqual(self.snap()["status"], "joining")
        await self.push(ALICE, "Start", lobby)
        self.assertEqual(self.snap()["status"], "playing")

    async def test_fill_seats_adds_house_characters_from_the_cast_for_the_host_only(self):
        await self.start()
        lobby = self.lobby()
        await self.push(BOB, "Join", lobby)
        click = await self.push(BOB, "Fill seats", lobby)
        self.assertIn("host", " ".join(says(click)).lower())
        self.assertEqual(len(self.snap()["seats"]), 2)
        await self.push(ALICE, "Fill seats", lobby)  # up to three seats in total
        self.assertEqual(self.seats(), [("member", ALICE, None), ("member", BOB, None), ("character", self.ann, None)])
        self.assertEqual(self.store.character_balance(G, self.ann), 100 - ANTE)
        await self.push(ALICE, "Fill seats", lobby)  # already three: one more
        self.assertEqual([s[1] for s in self.seats()[3:]], [self.bo])
        self.assertEqual(lobby.content.count("(house)"), 2)
        for name in ("Ann", "Bo"):
            self.assertIn(name, lobby.content)
        self.assertEqual(self.store.character_balance(G, self.cy), 100)

    async def test_favorites_are_marked_with_their_member_and_fill_skips_those_already_seated(self):
        await self.start(favorites=True)
        self.assertEqual(self.seats(), [("member", ALICE, None), ("character", self.ann, ALICE)])
        lobby = self.lobby()
        self.assertIn(f"(with <@{ALICE}>)", lobby.content)
        await self.push(ALICE, "Fill seats", lobby)
        self.assertEqual(self.seats()[2:], [("character", self.bo, None)])
        self.assertEqual(lobby.content.count("(house)"), 1)

    async def test_an_unbound_channel_is_humans_only(self):
        self.ch = self.channel2
        click = await self.start(favorites=True)
        self.assertEqual(self.seats(), [("member", ALICE, None)])
        self.assertTrue(any("character" in text.lower() for text in says(click)))
        self.assertNotIn("Fill seats", labels(self.lobby()))
        self.assertEqual(self.store.character_balance(G, self.ann), 100)

    async def test_how_to_play_button_answers_privately_for_anyone(self):
        await self.start()
        click = await self.push(DAN, "How to play", self.lobby())
        self.assertTrue(private(click))
        self.assertIn(doubt.how_to_play(self.store.game_rules(G, "doubt")), "\n".join(says(click)))

    async def test_an_idle_lobby_closes_with_refunds(self):
        await self.lobby3()
        snap = self.snap()
        await self.doubt.on_lobby_timer(G, snap["table_id"], snap["round_id"])
        self.assertIsNone(self.table())
        self.assertEqual([self.balance(u) for u in MEMBERS], [100, 100, 100])
        lobby = self.channel.messages[snap["message_id"]]
        self.assertIn("closed", lobby.content.lower())
        self.assertIsNone(lobby.view)

    async def test_a_lobby_timer_after_the_start_does_nothing(self):
        await self.playing()
        snap = self.snap()
        await self.doubt.on_lobby_timer(G, snap["table_id"], snap["round_id"])
        self.assertEqual(self.snap()["status"], "playing")
        self.assertNotIn(((G, snap["table_id"]), "lobby"), self.doubt.runtime.timers)

    async def test_stale_or_foreign_buttons_are_refused(self):
        await self.start()
        snap = self.snap()
        cases = [(G, DoubtButton("join", snap["table_id"], snap["round_id"] + 50, 0)),  # unknown round
                 (G, DoubtButton("join", snap["table_id"] + 1, snap["round_id"], 0)),  # table does not match round
                 (2, DoubtButton("join", snap["table_id"], snap["round_id"], 0))]  # another server
        for guild, item in cases:
            click = FakeGameClick(self.bot, BOB, guild_id=guild, channel=self.channel, on_call=http)
            await item.callback(click)
            self.assertEqual(says(click), [STALE], item.custom_id)
        self.assertEqual(len(self.snap()["seats"]), 1)


class ComponentTests(DoubtCase):
    def test_custom_id_round_trip_and_no_seat_in_it(self):
        item = DoubtButton("hand", 3, 4, 5, "My hand")
        self.assertEqual(item.item.custom_id, "llmcord:doubt:hand:3:4:5")
        template = DoubtButton.__discord_ui_compiled_template__
        match = template.fullmatch(item.item.custom_id)
        self.assertEqual((match["action"], match["table"], match["round"], match["moves"]), ("hand", "3", "4", "5"))
        self.assertIsNone(template.fullmatch("llmcord:doubt:hand:3:4:5:2"))  # a seat index cannot ride along
        self.assertIsNone(template.fullmatch("llmcord:doubt:hand:x:4:5"))
        self.assertIsNone(template.fullmatch("llmcord:bj:hit:3:4:5"))  # blackjack ids stay blackjack's
        self.assertIsNotNone(GameButton.__discord_ui_compiled_template__.fullmatch("llmcord:bj:hit:3:4:5"))
        self.assertIsNone(GameButton.__discord_ui_compiled_template__.fullmatch("llmcord:doubt:hand:3:4:5"))

    async def test_buttons_are_registered_with_the_bot_and_recovery_runs_at_startup(self):
        self.bot.tree.sync = mock.AsyncMock()
        self.games.recover, self.doubt.recover = mock.AsyncMock(), mock.AsyncMock()
        await self.bot.setup_hook()
        self.addCleanup(self.bot.cleanup_task.cancel)
        registered = list(self.bot._connection._view_store._dynamic_items.values())
        self.assertIn(DoubtButton, registered)
        self.assertIn(GameButton, registered)
        self.games.recover.assert_awaited_once()
        self.doubt.recover.assert_awaited_once()

    async def test_closing_the_bot_cancels_the_doubt_timers(self):
        await self.start()
        task = self.doubt.runtime.timers[((G, self.table()["id"]), "lobby")]
        with mock.patch.object(SkitBot.__mro__[1], "close", new=mock.AsyncMock()):
            self.bot.models.close = mock.AsyncMock()
            self.store.close = lambda: None
            await self.bot.close()
        self.assertTrue(task.cancelled())
        self.assertEqual(self.doubt.runtime.timers, {})


class StartAndBoardTests(DoubtCase):
    async def test_start_deals_and_posts_a_pinned_board_and_a_turn_message(self):
        await self.lobby3()
        thread = self.thread()
        self.assertEqual(thread.sent, [])
        await self.push(ALICE, "Start", self.lobby())
        snap = self.snap()
        self.assertEqual((snap["status"], snap["view"]["counts"]), ("playing", (18, 17, 17)))
        table = self.table()
        board, turn = thread.messages[table["board_message_id"]], thread.messages[table["turn_message_id"]]
        self.assertNotEqual(board.id, turn.id)
        self.assertEqual([m.id for m in thread.sent], [board.id, turn.id])
        self.assertTrue(board.pinned)
        self.assertFalse(turn.pinned)
        for user in MEMBERS:
            self.assertIn(f"<@{user}>", board.content)
        self.assertIn("18", board.content)
        self.assertIn("17", board.content)
        self.assertIn("30", board.content)  # the pot
        self.assertIn(f"<@{ALICE}>", turn.content)  # first on turn
        self.assertIn("A", turn.content)  # the rank to claim
        self.assertIn("<t:", turn.content)  # the deadline
        self.assertEqual(labels(turn).count("My hand"), 1)
        self.assertTrue(item_of(button(turn, "Doubt")).disabled)  # no claim yet
        self.assertNotIn(((G, snap["table_id"]), "lobby"), self.doubt.runtime.timers)
        self.assertIn(((G, snap["table_id"]), "turn"), self.doubt.runtime.timers)

    async def test_the_lobby_points_to_the_thread_once_play_starts(self):
        await self.playing()
        self.assertIn("In play", self.lobby().content)

    async def test_a_failed_pin_does_not_stop_the_game(self):
        await self.lobby3()
        self.thread().pin_error = forbidden()
        await self.push(ALICE, "Start", self.lobby())
        self.assertEqual(self.snap()["status"], "playing")
        self.assertFalse(self.board().pinned)
        self.assertTrue(self.turn().content)

    async def test_the_board_is_debounced_and_follows_the_play(self):
        await self.playing()
        board = self.board()
        before = board.content
        await self.pick_play(ALICE, {"A": 1})
        await self.pick_play(BOB, self.any_card(1))
        self.assertEqual(board.edits, [])  # nothing is sent inside the debounce window
        await self.doubt.flush()
        self.assertEqual(len(board.edits), 1)  # two events, one edit
        self.assertNotEqual(board.content, before)
        await self.doubt.flush()
        self.assertEqual(len(board.edits), 1)

    async def test_the_turn_message_is_edited_in_place_with_the_claim(self):
        await self.playing()
        turn, count = self.turn(), len(self.thread().sent)
        self.assertTrue(item_of(button(turn, "Doubt")).disabled)
        await self.pick_play(ALICE, {"A": 2})
        self.assertEqual(self.table()["turn_message_id"], turn.id)
        self.assertEqual(len(self.thread().sent), count)
        self.assertIn(f"<@{ALICE}> claims two", turn.content)
        self.assertIn(f"<@{BOB}>", turn.content)  # next up
        self.assertFalse(item_of(button(turn, "Doubt")).disabled)  # a doubt window is open
        self.assertIn(((G, self.snap()["table_id"]), "window"), self.doubt.runtime.timers)


class HandTests(DoubtCase):
    async def test_my_hand_shows_only_the_pressers_own_cards(self):
        await self.playing()
        everyone = {seat: self.hand(seat) for seat in range(3)}
        for seat, user in enumerate(MEMBERS):
            click = await self.push(user, "My hand", self.turn())
            self.assertTrue(private(click))
            text = blob(click)
            for card in everyone[seat]:
                self.assertIn(card_text(card), text)
            for other in range(3):
                if other != seat:
                    for card in everyone[other]:
                        self.assertNotIn(card_text(card), text)
        self.assertEqual(self.thread_texts(), [self.board().content, self.turn().content])  # nothing public

    async def test_the_picker_appears_only_on_turn_and_fits_discord_limits(self):
        await self.playing()
        waiting = await self.push(BOB, "My hand", self.turn())
        self.assertTrue(says(waiting) and all(not selects(view) for view in views(waiting)))  # the hand, but no picker
        on_turn = await self.push(ALICE, "My hand", self.turn())
        picker = [view for view in views(on_turn) if selects(view)]
        self.assertEqual(len(picker), 1)
        chosen = selects(picker[0])
        buttons = [child for child in picker[0].children if not isinstance(child, discord.ui.Select)]
        self.assertLessEqual(len(chosen) + math.ceil(len(buttons) / 5), 5)
        self.assertTrue(all(0 < len(select.options) <= 25 for select in chosen))
        self.assertEqual(self.snap()["moves"], 0)

    async def test_the_picker_submits_a_legal_play_for_the_pressing_seat(self):
        await self.playing()
        hand = self.hand(0)
        rank = hand[0][0]
        click = await self.pick_play(ALICE, {rank: 1})
        self.assertEqual(self.moves(), [(0, f"play:{rank}x1", "member")])
        self.assertEqual(self.snap()["view"]["counts"], (17, 17, 17))
        self.assertFalse(any("turn" in text.lower() and "not" in text.lower() for text in says(click)))
        two = self.hand(1)[:2]
        await self.pick_play(BOB, counts_of(two))
        self.assertEqual(self.moves()[1], (1, "play:" + ",".join(f"{r}x{n}" for r, n in counts_of(two).items()), "member"))

    async def test_invalid_counts_are_refused_and_change_nothing(self):
        await self.playing()
        hand = self.hand(0)
        held = counts_of(hand)
        rank = next(iter(held))
        bad_plays = [{rank: held[rank] + 1}, {rank: 0}, {}, counts_of(hand[:5]), {"Z": 1}]  # more than held, none, five cards, no such rank
        missing = [r for r in doubt.RANKS if r not in held]
        if missing:
            bad_plays.append({missing[0]: 1})  # a rank the hand does not hold
        for bad in bad_plays:
            click = await self.pick_play(ALICE, bad)
            self.assertTrue(says(click) and private(click), bad)
        self.assertEqual(self.moves(), [])

    async def test_a_member_cannot_play_or_see_for_another_seat(self):
        await self.playing()
        bob_card = self.any_card(1)
        click = await self.pick_play(BOB, bob_card)  # Alice is on turn, not Bob
        self.assertTrue(says(click) and private(click))
        self.assertEqual(self.moves(), [])
        click = await self.pick_play(DAN, {"A": 1})  # not seated at all
        self.assertEqual(says(click), [NOT_SEATED])
        click = await self.push(DAN, "My hand", self.turn())
        self.assertEqual(says(click), [NOT_SEATED])
        for card in self.hand(0) + self.hand(1) + self.hand(2):
            self.assertNotIn(card_text(card), blob(click))
        self.assertEqual(self.moves(), [])

    async def test_a_stale_picker_says_the_table_moved_on(self):
        await self.playing()
        click = await self.pick_play(ALICE, {"A": 1}, moves=3)
        self.assertTrue(any(text.startswith("The table moved on") for text in says(click)))
        self.assertTrue(private(click))
        self.assertEqual(self.moves(), [])

    async def test_game_hand_works_in_the_table_thread(self):
        await self.playing()
        thread = self.thread()
        click = self.click(BOB, channel=thread)
        await invoke(self.bot, "game hand", click)
        self.assertTrue(private(click))
        for card in self.hand(1):
            self.assertIn(card_text(card), blob(click))
        for card in self.hand(0):
            self.assertNotIn(card_text(card), blob(click))
        stranger = self.click(DAN, channel=thread)
        await invoke(self.bot, "game hand", stranger)
        self.assertEqual(says(stranger), [NOT_SEATED])
        elsewhere = self.click(BOB, channel=self.channel)
        await invoke(self.bot, "game hand", elsewhere)
        self.assertTrue(says(elsewhere) and private(elsewhere))

    async def test_game_rules_answers_privately_with_the_guide_and_the_rules(self):
        click = self.click(DAN)
        await invoke(self.bot, "game rules", click, game="doubt")
        self.assertTrue(private(click))
        rules = self.store.game_rules(G, "doubt")
        text = "\n".join(says(click))
        self.assertIn(doubt.how_to_play(rules), text)
        self.assertIn(doubt.rules_text(rules), text)


class DoubtWindowTests(DoubtCase):
    async def test_a_bluff_is_revealed_and_the_claimer_takes_the_pile(self):
        await self.playing()
        bluff = [c for c in self.hand(0) if c[0] != "A"][:2]
        await self.pick_play(ALICE, counts_of(bluff))
        click = await self.doubt_press(BOB)
        self.assertEqual(self.moves()[-1], (1, "doubt", "member"))
        text = next(t for t in self.thread_texts() if "Doubt" in t and "pile" in t)
        self.assertIn(f"<@{ALICE}>", text)
        self.assertIn("takes the pile", text)
        self.assertIn("(2 cards)", text)
        for card in bluff:
            self.assertIn(card[0], text)
        self.assertEqual(self.snap()["view"]["counts"], (18, 17, 17))
        self.assertFalse(says(click) and any("first" in t for t in says(click)))

    async def test_a_true_claim_makes_the_doubter_take_the_pile(self):
        await self.playing()
        await self.pick_play(ALICE, {"A": 2})
        await self.doubt_press(CARA)  # anyone but the claimer may doubt, on turn or not
        self.assertEqual(self.moves()[-1], (2, "doubt", "member"))
        text = next(t for t in self.thread_texts() if "Doubt" in t and "pile" in t)
        self.assertIn(f"<@{CARA}>", text)
        self.assertIn("(2 cards)", text)
        self.assertEqual(self.snap()["view"]["counts"], (16, 17, 19))

    async def test_the_first_doubt_wins_and_the_other_hears_so(self):
        await self.playing()
        await self.pick_play(ALICE, {"A": 2})
        snap = self.snap()
        clicks = {BOB: self.click(BOB, channel=self.thread()), CARA: self.click(CARA, channel=self.thread())}
        await asyncio.gather(*(DoubtButton("doubt", snap["table_id"], snap["round_id"], snap["moves"]).callback(c) for c in clicks.values()))
        doubts = [m for m in self.moves() if m[1] == "doubt"]
        self.assertEqual(len(doubts), 1)
        winner = next(user for user, seat in ((BOB, 1), (CARA, 2)) if seat == doubts[0][0])  # the seat is the presser's own
        loser = CARA if winner == BOB else BOB
        self.assertEqual(says(clicks[loser]), ["Someone doubted first."])
        self.assertTrue(private(clicks[loser]))
        self.assertNotIn("Someone doubted first.", says(clicks[winner]))
        late = self.click(CARA if winner == BOB else BOB, channel=self.thread())
        await DoubtButton("doubt", snap["table_id"], snap["round_id"], snap["moves"]).callback(late)
        self.assertEqual(says(late), ["Someone doubted first."])
        self.assertEqual(len([m for m in self.moves() if m[1] == "doubt"]), 1)

    async def test_refused_doubts_change_nothing(self):
        await self.playing()
        early = await self.doubt_press(BOB)  # no claim yet
        self.assertTrue(says(early) and private(early))
        await self.pick_play(ALICE, {"A": 1})
        own = await self.doubt_press(ALICE)  # your own claim
        self.assertTrue(says(own) and private(own))
        self.assertNotEqual(says(own), ["Someone doubted first."])
        stranger = await self.doubt_press(DAN)
        self.assertEqual(says(stranger), [NOT_SEATED])
        self.assertEqual([m[1] for m in self.moves()], ["play:Ax1"])

    async def test_an_expired_window_closes_the_claim_and_starts_the_next_clock(self):
        await self.playing()
        await self.pick_play(ALICE, {"A": 2})
        snap = self.snap()
        await self.doubt.on_window_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        self.assertEqual(self.moves()[-1], (0, "close", "timeout"))
        self.assertIsNone(self.snap()["view"]["window"])
        self.assertTrue(item_of(button(self.turn(), "Doubt")).disabled)
        self.assertIn(((G, snap["table_id"]), "turn"), self.doubt.runtime.timers)
        before = len(self.moves())
        await self.doubt.on_window_timer(G, snap["table_id"], snap["round_id"], snap["moves"])  # stale
        self.assertEqual(len(self.moves()), before)
        late = await self.doubt_press(BOB, moves=snap["moves"])
        self.assertTrue(says(late) and private(late))
        self.assertEqual(len(self.moves()), before)

    async def test_the_next_players_play_passes_on_the_doubt(self):
        await self.playing()
        await self.pick_play(ALICE, {"A": 1})
        await self.pick_play(BOB, self.any_card(1))
        self.assertEqual([m[1] for m in self.moves()][0], "play:Ax1")
        self.assertNotIn("doubt", [m[1] for m in self.moves()])
        self.assertEqual(self.snap()["view"]["window"]["claimer"], 1)


class TimeoutTests(DoubtCase):
    async def test_a_turn_timeout_plays_the_timeout_move(self):
        await self.playing()
        snap = self.snap()
        await self.doubt.on_turn_timer(G, snap["table_id"], snap["round_id"], 0)
        self.assertEqual(self.moves(), [(0, "play:Ax1", "timeout")])  # seat 0 holds an ace and must claim aces
        await self.doubt.on_turn_timer(G, snap["table_id"], snap["round_id"], 0)  # stale
        self.assertEqual(len(self.moves()), 1)
        self.assertIn(f"<@{ALICE}> claims", self.turn().content)

    async def test_a_second_turn_timeout_forfeits_the_seat(self):
        await self.playing()
        snap = self.snap()
        tid, rid = snap["table_id"], snap["round_id"]
        await self.doubt.on_turn_timer(G, tid, rid, 0)
        self.assertEqual(list(self.snap()["view"]["active"]), [True, True, True])  # one timeout is forgiven
        self.store_play(1, "play:" + f"{self.hand(1)[0][0]}x1")
        self.store_play(2, "play:" + f"{self.hand(2)[0][0]}x1")
        await self.doubt.on_window_timer(G, tid, rid, 3)  # closing a window is not a missed turn
        self.assertEqual(self.moves()[-1], (2, "close", "timeout"))
        await self.doubt.on_turn_timer(G, tid, rid, 4)  # seat 0 again
        self.assertEqual(self.moves()[-1], (0, "forfeit", "timeout"))
        self.assertEqual(list(self.snap()["view"]["active"]), [False, True, True])
        self.assertEqual(self.snap()["status"], "playing")
        self.assertTrue(any("forfeit" in text.lower() and f"<@{ALICE}>" in text for text in self.thread_texts()))

    async def test_a_late_timer_after_the_member_moved_does_nothing(self):
        await self.playing()
        snap = self.snap()
        await self.pick_play(ALICE, {"A": 1})
        await self.doubt.on_turn_timer(G, snap["table_id"], snap["round_id"], 0)
        self.assertEqual(len(self.moves()), 1)


class CharacterSeatTests(DoubtCase):
    async def table_with_ann(self):
        await self.start(favorites=True)  # Alice 0, Ann 1
        await self.push(BOB, "Join", self.lobby())
        await self.push(ALICE, "Start", self.lobby())

    async def test_a_character_doubts_a_claim_that_cannot_be_true(self):
        await self.table_with_ann()
        self.assertEqual(self.seats()[1], ("character", self.ann, ALICE))
        await self.pick_play(ALICE, counts_of(self.hand(0)[:4]))  # four "aces" while Ann holds one herself
        snap = self.snap()
        self.assertIn(((G, snap["table_id"]), "character"), self.doubt.runtime.timers)
        await self.doubt.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        self.assertEqual(self.moves()[-1], (1, "doubt", "character"))
        self.assertTrue(any("Doubt" in text and "pile" in text for text in self.thread_texts()))

    async def test_a_character_on_turn_plays_by_the_fallback_policy(self):
        await self.table_with_ann()
        await self.pick_play(ALICE, {"A": 1})
        snap = self.snap()
        await self.doubt.on_window_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        snap = self.snap()
        self.assertIn(((G, snap["table_id"]), "character"), self.doubt.runtime.timers)  # Ann is next
        held = counts_of(self.hand(1))
        await self.doubt.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        seat, move, actor = self.moves()[-1]
        self.assertEqual((seat, actor), (1, "character"))
        self.assertTrue(move.startswith("play:"))
        self.assertEqual(parse_move(move).get("2"), min(4, held["2"]))  # the policy plays every card of the forced rank
        self.assertLessEqual(sum(parse_move(move).values()), 4)
        await self.doubt.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"])  # stale
        self.assertEqual(len(self.moves()), 3)

    async def test_members_cannot_press_for_a_character_seat(self):
        await self.table_with_ann()
        await self.pick_play(ALICE, {"A": 1})
        snap = self.snap()
        await self.doubt.on_window_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        click = await self.pick_play(BOB, self.any_card(2))  # Ann (seat 1) is on turn
        self.assertTrue(says(click) and private(click))
        self.assertEqual(len(self.moves()), 2)


class BumpTests(DoubtCase):
    async def chatter(self, humans, bots=0, thread=None):
        thread = thread or self.thread()
        for _ in range(humans):
            await self.doubt.note_thread_message(thread_message(thread, BOB))
        for _ in range(bots):
            await self.doubt.note_thread_message(thread_message(thread, 99, bot=True))

    async def test_the_turn_message_is_reposted_once_eight_messages_bury_it(self):
        await self.playing()
        thread, first = self.thread(), self.table()["turn_message_id"]
        await self.chatter(7, bots=8)
        await self.pick_play(ALICE, self.any_card(0))
        self.assertEqual(self.table()["turn_message_id"], first)  # seven people talked: not buried yet
        self.assertFalse(thread.messages[first].deleted)
        await self.chatter(8)
        await self.pick_play(BOB, self.any_card(1))
        second = self.table()["turn_message_id"]
        self.assertNotEqual(second, first)
        self.assertTrue(thread.messages[first].deleted)
        self.assertEqual(thread.sent[-1].id, second)  # at the bottom
        self.assertIn("My hand", labels(thread.messages[second]))
        self.assertIn(f"<@{CARA}>", thread.messages[second].content)
        await self.pick_play(CARA, self.any_card(2))  # nobody talked since the repost
        self.assertEqual(self.table()["turn_message_id"], second)
        self.assertFalse(thread.messages[second].deleted)

    async def test_messages_elsewhere_do_not_count_and_do_not_fail(self):
        await self.playing()
        other = await self.channel.make_thread("elsewhere")
        await self.chatter(10, thread=other)
        await self.doubt.note_thread_message(thread_message(self.channel, BOB))
        first = self.table()["turn_message_id"]
        await self.pick_play(ALICE, self.any_card(0))
        self.assertEqual(self.table()["turn_message_id"], first)

    async def test_the_bot_hands_thread_messages_to_the_table(self):
        """Reaches the table even when the parent channel has no binding (humans-only tables are in unbound channels)."""
        self.ch = self.channel2
        await self.lobby3()
        await self.push(ALICE, "Start", self.lobby())
        thread, first = self.thread(), self.table()["turn_message_id"]
        for _ in range(8):
            await self.bot.on_message(thread_message(thread, BOB))
        await self.pick_play(ALICE, self.any_card(0))
        self.assertNotEqual(self.table()["turn_message_id"], first)


class SettlementTests(DoubtCase):
    async def test_a_full_game_pays_the_pot_and_wraps_up_the_table(self):
        await self.playing()
        table, thread = self.table(), self.thread()
        tid, rid, lobby, board, turn = table["id"], self.snap()["round_id"], self.lobby(), self.board(), self.turn()
        await self.play_out()
        await self.doubt.flush()
        winner = MEMBERS[pure_winner(self.seed, 3)]
        snap = self.store.round_snapshot(G, rid)
        self.assertEqual(snap["status"], "settled")
        self.assertEqual({u: self.balance(u) for u in MEMBERS}, {u: 120 if u == winner else 90 for u in MEMBERS})
        payout = self.store.ledger(G, user_id=winner)[0]
        self.assertEqual((payout["amount"], payout["reason"]), (30, PAYOUT_REASON))
        result = [t for m in thread.sent if m.id not in (board.id, turn.id) and not m.deleted for t in [m.content]
                  if f"<@{winner}>" in t and "30" in t and "pot" in t.lower()]
        self.assertTrue(result, self.thread_texts(thread))
        self.assertNotIn("\n", lobby.content)  # a one-line result
        self.assertIn(f"<@{winner}>", lobby.content)
        self.assertIsNone(lobby.view)
        self.assertTrue(board.edits)
        self.assertIsNone(self.store.open_table_for(G, CH))
        self.assertIn(((G, tid), "archive"), self.doubt.runtime.timers)
        self.assertFalse(thread.archived)
        await self.doubt.on_archive_timer(G, tid)
        self.assertTrue(thread.archived and thread.locked)
        self.assertTrue(any(call.get("archived") and call.get("locked") for call in thread.edit_calls))

    async def test_a_failed_archive_is_not_fatal(self):
        await self.playing()
        tid, thread = self.table()["id"], self.thread()
        await self.play_out()
        thread.thread_edit_error = forbidden()
        await self.doubt.on_archive_timer(G, tid)
        self.assertFalse(thread.archived)

    async def test_a_game_ending_by_forfeits_pays_the_last_seat(self):
        await self.playing()
        tid, rid = self.table()["id"], self.snap()["round_id"]
        self.store_play(1, "forfeit", "timeout")
        self.store_play(2, "forfeit", "timeout")
        snap = self.snap()
        self.assertEqual(snap["status"], "settled")
        await self.doubt.on_turn_timer(G, tid, rid, snap["moves"])  # a late timer on a finished round is harmless
        self.assertEqual([self.balance(u) for u in MEMBERS], [120, 90, 90])

    async def test_the_blackjack_table_summary_survives_a_settled_doubt_round(self):
        """FEAT-21 A3 review follow-up: blackjack's end-of-table summary must not crash on doubt rows."""
        await self.playing()
        tid = self.table()["id"]
        self.store_play(1, "forfeit", "timeout")
        self.store_play(2, "forfeit", "timeout")
        rows = self.store.game_table_summary(G, tid, 5)
        self.assertEqual(len(rows), 1)
        text = self.games._summary_text("Table closed.", rows)
        self.assertTrue(text.startswith("Table closed."))


class NoIOUnderLockTests(DoubtCase):
    def assert_never_under_a_lock(self, expected):
        self.assertTrue(expected <= {kind for kind, _ in HTTP}, sorted({kind for kind, _ in HTTP}))
        self.assertEqual([kind for kind, held in HTTP if held], [])

    async def test_lobby_turns_and_timers_make_no_discord_call_under_a_table_lock(self):
        await self.start(favorites=True)
        lobby = self.lobby()
        await self.push(BOB, "Join", lobby)
        await self.push(BOB, "Leave", lobby)
        await self.push(DAN, "Leave", lobby)  # a refusal
        await self.push(BOB, "Join", lobby)
        await self.push(ALICE, "Fill seats", lobby)
        await self.push(ALICE, "Start", lobby)
        await self.push(DAN, "My hand", self.turn())
        await self.push(ALICE, "My hand", self.turn())
        await self.pick_play(ALICE, counts_of(self.hand(0)[:4]))
        snap = self.snap()
        await self.doubt.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        await self.doubt.flush()
        snap = self.snap()
        await self.doubt.on_window_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        snap = self.snap()
        await self.doubt.on_turn_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        await self.chatter_then_move()
        self.assert_never_under_a_lock({"create_thread", "send", "edit", "pin", "defer", "followup"})

    async def chatter_then_move(self):
        for _ in range(8):
            await self.doubt.note_thread_message(thread_message(self.thread(), BOB))
        state, seats = self.engine_state(), self.snap()["seats"]
        if state.turn is not None and seats[state.turn]["kind"] == "member":  # a play that buries and reposts the turn message
            await self.pick_play(seats[state.turn]["ref_id"], self.any_card(state.turn))

    async def test_a_whole_game_to_its_archive_makes_no_discord_call_under_a_table_lock(self):
        await self.playing()
        tid = self.table()["id"]
        await self.play_out()
        await self.doubt.flush()
        await self.doubt.on_archive_timer(G, tid)
        self.assert_never_under_a_lock({"create_thread", "send", "edit", "pin"})

    async def test_thread_creation_failure_makes_no_call_under_a_lock_either(self):
        self.channel.thread_error = forbidden()
        await self.start()
        self.assert_never_under_a_lock({"create_thread"})


class RecoveryTests(DoubtCase):
    async def two_tables(self):
        await self.playing()  # CH: three members in play, ante paid
        self.ch = self.channel2
        await self.start(DAN)  # CH2: a lobby with one seat
        self.ch = self.channel

    async def test_unfinished_rounds_carry_the_thread(self):
        await self.playing()
        rows = self.store.unfinished_rounds()
        self.assertEqual([r["thread_id"] for r in rows], [self.table()["thread_id"]])

    async def test_restart_refunds_closes_and_leaves_a_note_in_each_thread(self):
        await self.two_tables()
        lobby, thread = self.lobby(), self.thread()
        lobby2, thread2 = self.channel2.messages[self.table(CH2)["message_id"]], self.channel2.threads[self.table(CH2)["thread_id"]]
        await self.doubt.recover()
        self.assertEqual([self.balance(u) for u in (*MEMBERS, DAN)], [100] * 4)
        self.assertEqual(self.store.unfinished_rounds(), [])
        self.assertIsNone(self.table(CH))
        self.assertIsNone(self.table(CH2))
        for message in (lobby, lobby2):
            self.assertIn("restarted", message.content.lower())
            self.assertIsNone(message.view)
        for placed in (thread, thread2):
            self.assertTrue(any("restarted" in text.lower() for text in self.thread_texts(placed)))
        self.assertEqual(self.doubt.runtime.timers, {})

    async def test_recovery_still_refunds_when_nothing_can_be_reached(self):
        await self.two_tables()
        self.bot.get_channel = lambda ident: None
        self.bot.fetch_channel = mock.AsyncMock(side_effect=not_found("Unknown Channel", 10003))
        await self.doubt.recover()
        self.assertEqual([self.balance(u) for u in (*MEMBERS, DAN)], [100] * 4)
        self.assertIsNone(self.table(CH))

    async def test_recovery_with_nothing_open_is_quiet(self):
        with self.assertNoLogs(level="WARNING"):
            await self.doubt.recover()
        self.assertEqual(self.channel.calls, [])

    async def test_each_game_recovers_only_its_own_tables(self):
        await self.playing()
        self.ch = self.channel2
        self.store.set_game_channel(G, CH2, True)
        with mock.patch.object(bj, "new_shoe", return_value=tuple(bj.RANKS.index(r) for r in ("9", "10", "6", "7")) + (1,) * 100):
            await self.games_bet(DAN, 10)
        self.assertEqual(self.balance(DAN), 90)
        doubt_snap = self.store.latest_round(G, self.table(CH)["id"])
        self.ch = self.channel
        await self.games.recover()  # blackjack's recovery
        self.assertIsNone(self.table(CH2))
        self.assertEqual(self.balance(DAN), 100)
        self.assertEqual(self.store.round_snapshot(G, doubt_snap["round_id"])["status"], "playing")  # doubt's table is untouched
        self.assertEqual(self.balance(ALICE), 90)
        self.assertEqual(self.thread_texts(), [self.board().content, self.turn().content])
        self.assertEqual(self.lobby().edits[-1]["content"].count("restarted"), 0)
        await self.doubt.recover()
        self.assertEqual(self.balance(ALICE), 100)
        self.assertIsNone(self.table(CH))

    async def test_doubt_recovery_leaves_a_blackjack_table_alone(self):
        await self.playing()
        self.store.set_game_channel(G, CH2, True)
        await self.games_bet(DAN, 10, channel=self.channel2)
        await self.doubt.recover()
        self.assertEqual(self.table(CH2)["game"], "blackjack")
        self.assertEqual(self.balance(DAN), 90)
        self.assertEqual(self.channel2.messages[self.table(CH2)["message_id"]].edits, [])

    async def games_bet(self, user, bet, channel=None):
        click = self.click(user, channel=channel or self.channel2)
        await invoke(self.bot, "blackjack", click, bet=bet)
        return click


class BlackjackUnchangedTests(DoubtCase):
    def setUp(self):
        shoe = tuple(bj.RANKS.index(r) for r in ("9", "10", "6", "7")) + (bj.RANKS.index("2"),) * 100
        patcher = mock.patch.object(bj, "new_shoe", return_value=shoe)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def check_blackjack_table(self, click):
        table = self.table()
        self.assertEqual(table["game"], "blackjack")
        content, kwargs = click.response.sent[0]
        self.assertIn("**Blackjack**", content)
        self.assertEqual(self.balance(ALICE), 90)
        lobby = self.channel.messages[table["message_id"]]
        self.assertTrue(all(item_of(child).custom_id.startswith("llmcord:bj:") for child in lobby.view.children))
        self.assertIn(((G, table["id"]), "table"), self.games.runtime.timers)
        self.assertEqual(list(self.games.runtime.timers), [((G, table["id"]), "table")])
        self.assertIsNone(table["thread_id"])
        await self.push(ALICE, "Deal now", lobby)
        self.assertEqual(self.store.latest_round(G, table["id"])["status"], "playing")

    async def test_blackjack_command_still_opens_and_plays_a_table(self):
        click = self.click(ALICE)
        await invoke(self.bot, "blackjack", click, bet=10)
        await self.check_blackjack_table(click)

    async def test_game_start_blackjack_delegates_to_it(self):
        click = self.click(ALICE)
        await invoke(self.bot, "game start", click, game="blackjack", bet=10, favorites=False)
        await self.check_blackjack_table(click)

    async def test_blackjack_timers_ignore_a_doubt_table(self):
        await self.playing()
        snap = self.snap()
        key = (G, snap["table_id"])
        self.assertNotIn((key, "table"), self.games.runtime.timers)
        self.assertNotIn(key, self.games.tables)
        await self.games.sweep()
        for hook in (self.games.on_join_timer, self.games.on_idle_timer):
            await hook(G, snap["table_id"], snap["round_id"])
        await self.games.on_turn_timer(G, snap["table_id"], snap["round_id"], 0)
        self.assertEqual(self.snap()["status"], "playing")
        self.assertEqual(self.moves(), [])


class ReviewFixTests(DoubtCase):
    """B1 review fixes: other games' tables, the window and the clocks, hidden-info seeds, forfeits, thread readiness and wording."""

    async def test_blackjack_does_not_join_an_open_doubt_table(self):
        await self.start()
        before = (self.balance(BOB), self.seats())
        for path, kwargs in (("blackjack", {"bet": 10}), ("game start", {"game": "blackjack", "bet": 10, "favorites": False})):
            click = self.click(BOB)
            await invoke(self.bot, path, click, **kwargs)
            self.assertEqual(says(click), ["An I Doubt It table is open in this channel."])
            self.assertTrue(private(click))
        self.assertEqual((self.balance(BOB), self.seats()), before)
        self.assertEqual(self.table()["game"], "doubt")

    async def test_turning_games_off_closes_a_doubt_table_with_its_own_words(self):
        await self.lobby3()
        lobby = self.lobby()
        closed, paint = await self.games.switch_channel(G, CH, False)
        self.assertTrue(closed)
        self.assertIsNone(paint)
        self.assertIn("turned off", lobby.content)
        self.assertIsNone(lobby.view)
        self.assertEqual([self.balance(u) for u in MEMBERS], [100, 100, 100])
        self.assertEqual({k for k in self.doubt.runtime.timers if k[1] != "archive"}, set())

    async def test_while_a_window_is_open_nothing_waits_for_a_play(self):
        await self.table_with_ann()
        await self.pick_play(ALICE, {"A": 1})  # Ann is next, Alice's window is open
        snap = self.snap()
        kinds = {k[1] for k in self.doubt.runtime.timers}
        self.assertIn("window", kinds)
        self.assertIn("character", kinds)  # she may doubt
        await self.doubt.on_character_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        self.assertEqual(len(self.moves()), 1)  # but she does not play before the window closes
        await self.doubt.on_window_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        self.assertIn(((G, snap["table_id"]), "character"), self.doubt.runtime.timers)

    async def table_with_ann(self):
        await self.start(favorites=True)
        await self.push(BOB, "Join", self.lobby())
        await self.push(ALICE, "Start", self.lobby())

    async def test_the_turn_clock_starts_when_the_window_closes(self):
        self.store.set_game_rules(G, "doubt", {"ante": ANTE, "turn_timeout": 15})
        await self.playing()
        snap = self.snap()
        self.assertEqual(snap["rules"]["turn_timeout"], 15)
        await self.pick_play(ALICE, {"A": 1})
        snap = self.snap()
        self.assertNotIn(((G, snap["table_id"]), "turn"), self.doubt.runtime.timers)
        self.assertNotIn("<t:", self.turn().content.split("Next:")[1].splitlines()[0])
        await self.doubt.on_turn_timer(G, snap["table_id"], snap["round_id"], snap["moves"])  # a straggler
        self.assertEqual(len(self.moves()), 1)
        await self.doubt.on_window_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        self.assertIn(((G, snap["table_id"]), "turn"), self.doubt.runtime.timers)
        self.assertIn("<t:", self.turn().content.split("Next:")[1].splitlines()[0])

    async def test_decision_seeds_come_from_the_secret_seed_and_are_deterministic(self):
        await self.playing()
        snap = self.snap()
        public = snap["seed_hash"]
        seed = self.store.policy_seed(G, snap["round_id"], "0/1/doubt")
        self.assertEqual(seed, self.store.policy_seed(G, snap["round_id"], "0/1/doubt"))
        self.assertNotEqual(seed, self.store.policy_seed(G, snap["round_id"], "0/2/doubt"))
        self.assertNotEqual(seed, self.seed)
        self.assertNotEqual(seed, doubt.seed_hash(f"{public}/0/1/doubt"))
        self.assertNotEqual(seed, doubt.seed_hash(f"{self.seed}"))
        self.assertEqual(seed, doubt.seed_hash(f"{self.seed}/0/1/doubt"))

    async def test_the_second_timed_out_play_in_a_row_forfeits_but_a_play_of_your_own_resets_it(self):
        await self.playing()
        tid, rid = self.snap()["table_id"], self.snap()["round_id"]
        await self.doubt.on_turn_timer(G, tid, rid, 0)  # Alice times out
        self.store_play(1, "play:" + f"{self.hand(1)[0][0]}x1")
        self.store_play(2, "play:" + f"{self.hand(2)[0][0]}x1")
        await self.doubt.on_window_timer(G, tid, rid, 3)
        await self.pick_play(ALICE, self.any_card(0))  # she plays herself: the count starts over
        snap = self.snap()
        await self.doubt.on_window_timer(G, tid, rid, snap["moves"])
        self.store_play(1, "play:" + f"{self.hand(1)[0][0]}x1")
        self.store_play(2, "play:" + f"{self.hand(2)[0][0]}x1")
        snap = self.snap()
        await self.doubt.on_window_timer(G, tid, rid, snap["moves"])
        snap = self.snap()
        await self.doubt.on_turn_timer(G, tid, rid, snap["moves"])
        self.assertEqual(self.moves()[-1][2], "timeout")
        self.assertTrue(self.moves()[-1][1].startswith("play:"))  # not a forfeit yet
        self.assertEqual(list(self.snap()["view"]["active"]), [True, True, True])

    async def test_start_waits_for_the_thread(self):
        await self.lobby3()
        self.store.set_table_thread(G, self.table()["id"], None)
        click = await self.push(ALICE, "Start", self.lobby())
        self.assertIn("thread", " ".join(says(click)))
        self.assertEqual(self.snap()["status"], "joining")

    async def test_the_start_button_is_disabled_until_the_thread_exists(self):
        gate = asyncio.Event()
        real = self.channel.make_thread

        async def slow(name, **kwargs):
            await gate.wait()
            return await real(name, **kwargs)

        self.channel.make_thread = slow
        task = asyncio.create_task(self.start())
        for _ in range(20):
            await asyncio.sleep(0)
        self.assertTrue(item_of(button(self.lobby(), "Start")).disabled)
        gate.set()
        await task
        await self.push(BOB, "Join", self.lobby())
        await self.push(CARA, "Join", self.lobby())
        self.assertFalse(item_of(button(self.lobby(), "Start")).disabled)

    async def test_only_missing_permissions_name_the_permissions(self):
        self.channel.thread_error = discord.HTTPException(mock.Mock(status=500, reason="boom"), "boom")
        click = await self.start()
        text = " ".join(says(click))
        self.assertNotIn("Create Public Threads", text)
        self.assertIn("Couldn't open the table thread", text)
        self.assertIn("antes are refunded", text)
        self.assertIsNone(self.table())

    async def test_no_refund_words_when_there_is_no_ante(self):
        self.store.set_game_rules(G, "doubt", {"ante": 0})
        await self.lobby3()
        snap = self.snap()
        await self.doubt.on_lobby_timer(G, snap["table_id"], snap["round_id"])
        self.assertNotIn("refund", self.channel.messages[snap["message_id"]].content.lower())
        self.channel.thread_error = forbidden()
        click = await self.start(DAN)
        self.assertNotIn("refund", " ".join(says(click)).lower())

    async def test_the_final_turn_message_is_edited_not_deleted(self):
        await self.playing()
        turn = self.turn()
        await self.play_out()
        self.assertFalse(turn.deleted)
        self.assertIn("Game over", turn.content)
        self.assertIsNone(turn.view)


class OutsideCloseTests(DoubtCase):
    """MNT-45/46 and the FEAT-21 branch review: tables closed from outside, lock namespaces, the host who left, wording."""

    async def grab(self):
        """(table id, lobby message, thread, turn message or None) of the open table."""
        table = self.table()
        thread = self.thread()
        return table["id"], self.lobby(), thread, thread.messages.get(table["turn_message_id"])

    def assert_cleaned(self, tid, lobby, thread, turn, reason):
        self.assertIn("Table closed", lobby.content)
        self.assertIn(reason, lobby.content)
        self.assertIsNone(lobby.view)
        if turn is not None:
            self.assertEqual(turn.content, "Table closed.")
            self.assertIsNone(turn.view)
        self.assertEqual(len([t for t in self.thread_texts(thread) if t.startswith("Table closed:")]), 1)
        self.assertEqual({k for k in self.doubt.runtime.timers if k[1] != "archive"}, set())
        self.assertIn(((G, tid), "archive"), self.doubt.runtime.timers)

    async def assert_forgotten(self, tid, thread):
        """After the archive timer the table is gone from memory and a later sweep says and archives nothing more."""
        self.doubt.runtime.disarm((G, tid), "archive")
        await self.doubt.on_archive_timer(G, tid)
        self.assertNotIn((G, tid), self.doubt.tables)
        count, edits = len(thread.sent), len(thread.edit_calls)
        await self.doubt.sweep()
        self.assertEqual((len(thread.sent), len(thread.edit_calls)), (count, edits))
        self.assertNotIn(((G, tid), "archive"), self.doubt.runtime.timers)

    async def test_a_dashboard_switch_off_closes_a_table_in_play_through_the_sweeper(self):
        await self.playing()
        tid, lobby, thread, turn = await self.grab()
        HTTP.clear()
        self.store.set_game_enabled(G, "doubt", False)
        await self.doubt.sweep()
        self.assert_cleaned(tid, lobby, thread, turn, "I Doubt It was turned off on this server.")
        self.assertIn("refunded", lobby.content)
        self.assertEqual([self.balance(u) for u in MEMBERS], [100, 100, 100])
        self.assertFalse(any(held for _, held in HTTP))
        count = len(thread.sent)
        await self.doubt.sweep()
        self.assertEqual(len(thread.sent), count)
        await self.assert_forgotten(tid, thread)

    async def test_removing_the_channel_closes_the_lobby_with_its_own_words(self):
        await self.lobby3()
        tid, lobby, thread, turn = await self.grab()
        self.store.set_game_channels(G, {CH2})
        await self.doubt.sweep()
        self.assert_cleaned(tid, lobby, thread, turn, "games were turned off here.")

    async def test_a_generic_close_and_no_refund_words_without_an_ante(self):
        self.store.set_game_rules(G, "doubt", {"ante": 0})
        await self.lobby3()
        tid, lobby, thread, turn = await self.grab()
        self.store.close_table(G, tid)
        await self.doubt.sweep()
        self.assert_cleaned(tid, lobby, thread, turn, "the table was closed.")
        self.assertNotIn("refund", lobby.content.lower())

    async def test_a_press_on_a_table_closed_from_outside_cleans_up_at_once(self):
        await self.lobby3()
        tid, lobby, thread, turn = await self.grab()
        self.store.set_game_enabled(G, "doubt", False)
        click = await self.push(DAN, "Join", lobby)
        self.assertEqual(says(click), [STALE])
        self.assert_cleaned(tid, lobby, thread, turn, "turned off")

    async def test_a_timer_that_finds_the_table_closed_cleans_up_at_once(self):
        await self.playing()
        tid, lobby, thread, turn = await self.grab()
        snap = self.snap()
        self.store.set_game_channels(G, {CH2})
        self.doubt.arm((G, tid), "turn", 0, snap["round_id"], snap["moves"])  # replaces the table's real turn timer with an immediate one
        await asyncio.wait_for(self.doubt.runtime.timers[((G, tid), "turn")], 5)  # a failing hook would otherwise retry only after runtime.retry_seconds (3600 here)
        self.assert_cleaned(tid, lobby, thread, turn, "games were turned off here.")

    async def test_a_game_that_ends_normally_is_not_repainted_by_the_sweeper(self):
        await self.playing()
        tid, lobby, thread, turn = await self.grab()
        await self.play_out()
        text = lobby.content
        await self.doubt.sweep()
        self.assertEqual(lobby.content, text)
        self.assertIn("won I Doubt It", text)
        self.assertEqual([t for t in self.thread_texts(thread) if t.startswith("Table closed:")], [])

    async def test_the_sweeper_is_started_with_the_first_table_and_stopped_with_close(self):
        await self.lobby3()
        tid, lobby, thread, turn = await self.grab()
        sweeper = self.doubt.sweeper
        self.assertIsNotNone(sweeper)
        self.assertFalse(sweeper.done())
        self.store.set_game_enabled(G, "doubt", False)
        await self.doubt.sweep()  # what the loop does every SWEEP_SECONDS
        self.assert_cleaned(tid, lobby, thread, turn, "turned off")
        self.assertFalse(sweeper.done())
        await self.doubt.close()
        self.assertTrue(sweeper.done())
        self.assertTrue(sweeper.cancelled())

    async def test_every_close_path_forgets_the_table(self):
        await self.start()
        key = (G, self.table()["id"])
        self.assertIn(key, self.doubt.tables)
        await self.push(ALICE, "Leave", self.lobby())  # the last player leaves
        await self.doubt.on_archive_timer(*key)
        self.assertNotIn(key, self.doubt.tables)
        await self.start(DAN, channel=self.channel2)
        key2 = (G, self.table(CH2)["id"])
        await self.doubt.recover()
        self.assertNotIn(key2, self.doubt.tables)

    async def test_switching_a_doubt_channel_off_holds_no_blackjack_table_lock(self):
        await self.lobby3()
        tid = self.table()["id"]
        lobby = self.lobby()
        async with self.games.runtime.lock((G, tid)):
            closed, paint = await asyncio.wait_for(self.games.switch_channel(G, CH, False), 2)
        self.assertEqual((closed, paint), (True, None))
        self.assertIn("turned off", lobby.content)
        await self.doubt.on_archive_timer(G, tid)
        self.assertNotIn((G, tid), self.doubt.tables)

    async def test_a_doubt_table_closed_by_a_switch_that_did_not_see_it_is_cleaned_up(self):
        await self.lobby3()
        tid, lobby, thread, turn = await self.grab()
        real, calls = self.store.open_table_for, []

        def blind_once(*args):
            calls.append(args)
            return None if len(calls) == 1 else real(*args)
        with mock.patch.object(self.store, "open_table_for", side_effect=blind_once):
            closed, paint = await self.games.switch_channel(G, CH, False)
        self.assertTrue(closed)
        self.assert_cleaned(tid, lobby, thread, turn, "games were turned off here.")

    async def test_a_step_sealed_before_an_outside_close_cannot_repaint_the_closed_table(self):
        await self.playing()
        tid, lobby, thread, turn = await self.grab()
        key, snap = (G, tid), self.snap()
        async with self.doubt.runtime.lock(key):  # a pick, sealed but not yet delivered
            late = self.doubt._picker_play(key, ALICE, snap["round_id"], snap["moves"], self.any_card(0))
        self.assertTrue(late.ok)
        self.store.set_game_enabled(G, "doubt", False)
        await self.doubt.sweep()
        sent = [m.content for m in thread.sent]
        await self.doubt._publish(key, late)
        self.assertEqual(turn.content, "Table closed.")
        self.assertIsNone(lobby.view)
        self.assertEqual([m.content for m in thread.sent], sent)
        self.assertEqual(len([t for t in self.thread_texts(thread) if t.startswith("Table closed:")]), 1)
        await self.assert_forgotten(tid, thread)

    async def test_a_table_closed_while_its_thread_is_being_made_leaves_no_registered_thread(self):
        real = self.channel.make_thread

        async def close_first(name, **kwargs):
            self.store.set_game_enabled(G, "doubt", False)
            return await real(name, **kwargs)
        with mock.patch.object(self.channel, "make_thread", side_effect=close_first):
            await self.start()
        (thread,) = self.channel.threads.values()
        row = self.store.one("SELECT * FROM game_tables WHERE guild_id=?", (G,))
        self.assertIsNone(row["thread_id"])
        self.assertEqual(self.doubt.threads, {})
        self.assertTrue(thread.archived and thread.locked)
        await self.doubt.sweep()
        self.assertIn("turned off", self.channel.messages[row["message_id"]].content)
        self.assertEqual(thread.sent, [])

    async def test_closing_a_channel_whose_table_closed_already_still_cleans_it_up(self):
        await self.lobby3()
        tid, lobby, thread, turn = await self.grab()
        self.store.close_table(G, tid)
        closed = await self.doubt.close_for_channel(G, CH, tid)
        self.assertIsNone(closed)
        self.assertIn("Table closed", lobby.content)
        self.assertIsNone(lobby.view)

    async def test_when_the_host_left_the_first_seated_member_hosts(self):
        await self.lobby3()
        lobby = self.lobby()
        self.assertIn(f"Host: <@{ALICE}>", lobby.content)
        await self.push(ALICE, "Leave", lobby)
        self.assertIn(f"Host: <@{BOB}>", lobby.content)
        self.assertNotIn(f"Host: <@{ALICE}>", lobby.content)
        self.assertIn("if the host has left", lobby.content)
        await self.push(DAN, "Join", lobby)
        for stranger in (ALICE, CARA + 100):
            click = await self.push(stranger, "Start", lobby)
            self.assertEqual(self.snap()["status"], "joining")
            self.assertTrue(says(click))
        await self.push(CARA, "Start", lobby)
        self.assertEqual(self.snap()["status"], "playing")

    async def test_a_present_host_is_still_the_only_one_who_starts(self):
        await self.lobby3()
        click = await self.push(BOB, "Start", self.lobby())
        self.assertEqual(says(click), ["Only the host can start the game."])
        self.assertEqual(self.snap()["status"], "joining")

    async def test_my_hand_words_before_and_after_the_game(self):
        await self.lobby3()
        thread = self.thread()
        click = self.click(BOB, channel=thread)
        await invoke(self.bot, "game hand", click)
        self.assertEqual(says(click), ["The game hasn't started yet."])
        snap = self.snap()
        await self.doubt.on_lobby_timer(G, snap["table_id"], snap["round_id"])
        click = self.click(BOB, channel=thread)
        await self.doubt._show_hand(click, (G, snap["table_id"]), snap["round_id"])
        self.assertEqual(says(click), ["This game is over."])

    async def test_my_hand_after_a_settled_game_says_the_game_is_over(self):
        await self.playing()
        snap = self.snap()
        await self.play_out()
        click = self.click(BOB, channel=self.ch)
        await self.doubt._show_hand(click, (G, snap["table_id"]), snap["round_id"])
        self.assertEqual(says(click), ["This game is over."])

    async def test_a_join_restarts_the_idle_lobby_clock(self):
        await self.start()
        key = ((G, self.table()["id"]), "lobby")
        first = self.doubt.runtime.timers[key]
        self.doubt.clock = lambda: 5000.0
        await self.push(BOB, "Join", self.lobby())
        self.assertIsNot(self.doubt.runtime.timers[key], first)
        self.assertIn("<t:8600:R>", self.lobby().content)

    async def test_the_forbidden_reply_and_the_restart_note_mention_refunds_only_with_an_ante(self):
        self.channel.thread_error = forbidden()
        click = await self.start()
        text = " ".join(says(click))
        self.assertIn("Send Messages in Threads", text)
        self.assertIn("antes are refunded", text)
        self.channel.thread_error = None
        await self.start(ALICE)
        lobby = self.lobby()
        await self.doubt.recover()
        self.assertIn("antes were refunded", lobby.content)
        self.store.set_game_rules(G, "doubt", {"ante": 0})
        await self.start(DAN, channel=self.channel2)
        lobby2 = self.channel2.messages[self.table(CH2)["message_id"]]
        await self.doubt.recover()
        self.assertIn("restarted", lobby2.content)
        self.assertNotIn("refund", lobby2.content.lower())

    async def test_the_doubt_window_line_reads_the_same_with_and_without_a_stamp(self):
        await self.playing()
        await self.pick_play(ALICE, {"A": 1})
        self.assertRegex(self.turn().content, r"Doubt window open <t:\d+:R>\.")
        snap = self.snap()
        self.doubt.table((G, snap["table_id"])).deadlines.pop("window")
        content, _ = self.doubt.render_turn((G, snap["table_id"]), snap)
        self.assertIn("Doubt window open.", content)


class FallbackMovesTests(DoubtCase):
    async def test_fallback_moves_are_policy_then_timeout_then_forfeit(self):
        """Characterization (MNT-49): _fallback_moves lists the policy play, the timeout play, then forfeit, all for the same seat."""
        await self.playing()
        snap = self.snap()
        seen = self.store.seat_view(G, snap["round_id"], 0)
        moves = self.doubt._fallback_moves(G, snap, 0, seen)
        self.assertEqual([seat for seat, _ in moves], [0, 0, 0])
        self.assertEqual(moves[2][1], "forfeit")
        self.assertTrue(moves[0][1].startswith("play:"))
        seed = self.store.policy_seed(G, snap["round_id"], f"{snap['moves']}/0/fallback")
        self.assertEqual(moves[1][1], doubt.timeout_move_for_view(seen, seed))
        self.assertEqual(moves, self.doubt._fallback_moves(G, snap, 0, seen))  # deterministic


if __name__ == "__main__":
    unittest.main()
