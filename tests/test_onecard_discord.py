"""FEAT-51 step A: One Card on Discord (`/game start game:onecard`, lobby, table thread, hand picker, turns, timers, settlement).

TDD-red tests written before the implementation. Offline: fake channels, threads and interactions from ``helpers``, a fixed seed, the
pure engine (``games.onecard``) used as an oracle, an in-memory store. Timers are driven by calling their hooks, never by waiting.
The module is modelled on ``test_doubt_discord.py`` / ``games_doubt_discord.py``; D40 (docs/engineering/backlog.md) is the spec.

Assumed API (pinned here, to be created in ``llmcord_core/games_onecard_discord.py`` and wired in ``discord_bot.py``):

  ``bot.onecard = OneCardTables(bot, clock=time.time)``, built next to ``bot.doubt``; same attributes as DoubtTables: ``runtime``
  (``TableRuntime``: ``locks``, ``timers`` keyed ``((guild_id, table_id), kind)`` with kind in ``lobby turn archive``, ``retry_seconds``,
  ``sweeper``), ``tables``, ``threads``, ``clock``, class constants ``LOBBY_SECONDS=300 ARCHIVE_SECONDS=600 BOARD_DEBOUNCE_SECONDS=1
  BUMP_AFTER=8`` (there is no doubt window; the character timer and the character seats of step B are covered in ``test_onecard_characters.py``). The turn timeout is the round's rule ``turn_timeout``.
  Wiring: ``SkitBot.setup_hook`` awaits ``onecard.recover()`` and registers ``OneCardButton``; ``SkitBot.close`` awaits ``onecard.close()``;
  ``SkitBot.on_message`` calls ``onecard.note_thread_message(message)``; ``/game start`` offers the choice ``("One Card", "onecard")`` after
  the existing three and starts it with ``onecard.start_command(interaction, favorites)``; ``/game rules game:onecard`` answers privately with the
  game's guide and rules; ``/game hand`` in a One Card thread calls ``onecard.hand_command(interaction)``; ``GameTables.switch_channel`` hands a
  One Card table to ``onecard.close_for_channel(guild_id, channel_id, table_id)``.
  Component: ``OneCardButton(action, table_id, round_id, moves, label="", style=..., disabled=False)``, a DynamicItem with custom id
    ``llmcord:onecard:<action>:<table>:<round>:<moves>`` (no seat in it); its callback calls ``interaction.client.onecard.press(interaction, action,
    table_id, round_id, moves)``. Lobby buttons: Join, Leave, Start, How to play (no "Fill seats": favorites are the only way characters sit, see step B). The turn message has one button, "My hand" (not "Doubt"). The in-play lobby message keeps only "How to play" and says "In play".
  ``start_command(interaction, favorites=False)``: lobby of 2 to 8 seats (``onecard.MIN_SEATS`` / ``MAX_SEATS``) with the round's ante and pot, the
    host, the lobby deadline as ``<t:N:R>``; a table thread named ``One Card · table N``. ``favorites=True`` seats the starter's favorites
    (step B, ``test_onecard_characters.py``).
  ``press(interaction, action, table_id, round_id, moves)``: actions ``join leave start help hand``.
  ``hand_command(interaction)`` and the "My hand" button reply privately with the presser's own cards only. The reply to a seat on turn carries
    the picker (a ``discord.ui.View``): ONE ``Select`` whose option ``value`` is the card text (``onecard.card_text``, e.g. ``9♥``, ``JKb``)
    of every distinct card that has a legal play (a 7 appears once; the suit is chosen in a second private step), and a button labelled "Draw"
    (the label may carry the pending count, e.g. "Draw 2"). No playable card: no Select, just the Draw button. A seat not on turn gets the
    hand without any Select or Draw button.
  ``await onecard.play_from_picker(interaction, table_id, round_id, moves, move)``: ``move`` is the engine move text the picker built
    (``play:9♥``, ``play:7♥=♠``, ``draw``); ``moves`` is the move count the picker was built from. Only ``play:`` and ``draw`` are accepted from a
    picker (``declare`` and ``call_out`` come by chat message in FEAT-52; ``timeout`` and ``forfeit`` are the timers').
  Timer hooks (stale-safe, lock-disciplined): ``on_lobby_timer(guild_id, table_id, round_id)``, ``on_turn_timer(guild_id, table_id, round_id,
    moves)``, ``on_archive_timer(guild_id, table_id)``. A missed turn applies the engine move ``timeout`` (actor ``timeout``: draws and passes) and says
    so in the thread; the second consecutive missed turn of a seat applies the move ``forfeit`` (actor ``timeout``).
    ENGINE SEAM NEEDED: ``games/onecard.py`` has no ``forfeit`` move yet. It must deactivate the seat (like bankruptcy: hand back under the draw
    pile, turn passes on when it was the forfeiter's turn, the last active seat wins) and ``result`` already reports ``outcome='forfeit'``.
  ``await onecard.flush()`` sends the debounced board edit now; ``await onecard.recover()``; ``await onecard.close()``; ``await onecard.sweep()``;
    ``await onecard.close_for_channel(guild_id, channel_id, table_id)``; ``await onecard.note_thread_message(message)``.
  Rendering (public, never a hand): ``render_lobby(key, snap, row) -> (text, view)``, ``render_board(key, snap) -> text``,
    ``render_turn(key, snap) -> (text, view)`` where ``key = (guild_id, table_id)`` and ``snap`` is ``store.round_snapshot``. The board names every
    seat by mention, shows its hand size ("1 card", "7 cards"), the top card (``onecard.card_text``), ``Suit: <suit>``, ``Pending attack: <n>`` only
    while an attack is pending, "clockwise" or "counter-clockwise", whose turn it is, and "exposed" / "declared" marks on a seat's line. The turn
    message says what just happened (``<@id> played 2♦.``, ``<@id> drew a card.``, ``<@id> drew 2 cards.``), then ``Next: <@id>`` with the turn deadline ``<t:N:R>``.
  Internals mirrored from DoubtTables and used by a few tests: ``arm(key, kind, delay, *args)``, ``tables``, ``_show_hand(interaction, key, round_id)``,
    ``_picker_play(key, user_id, round_id, moves, move) -> _Out`` (``.ok``) under ``runtime.lock(key)``, ``_publish(key, out)``.
  Money is the registry's: the ante is pinned in the round's rules, the winner takes the pot, ledger reasons ``One Card ante (table N, round M)``
    and ``One Card payout (table N, round M)``; closing a table refunds the antes.
"""
import asyncio
import copy
import functools
import json
import unittest
from unittest import mock

import discord
from helpers import FakeGameChannel, FakeGameClick, command, forbidden, invoke, make_settings, not_found, thread_message
from test_doubt_discord import blob, button, item_of, labels, private, says, selects, views

from llmcord_core.discord_bot import SkitBot
from llmcord_core.games import onecard
from llmcord_core.games.base import Seat
from llmcord_core.games_discord import NOT_SEATED, NOT_YOUR_TURN, OFF_HERE, STALE, GameButton

try:  # imported softly so every test fails on its own until the module exists
    from llmcord_core.games_onecard_discord import OneCardButton
except ImportError:  # pragma: no cover - the red state
    OneCardButton = None

MISSING = "llmcord_core/games_onecard_discord.py is missing: SkitBot has no .onecard (FEAT-51 step A)"
G, CH, CH2 = 1, 100, 101
ALICE, BOB, CARA, DAN = 5, 6, 7, 8
MEMBERS = (ALICE, BOB, CARA)
ANTE = 10
NO_MENTIONS = discord.AllowedMentions.none().to_dict()
ANTE_REASON = "One Card ante (table 1, round 1)"
PAYOUT_REASON = "One Card payout (table 1, round 1)"

ONECARD = None  # the OneCardTables under test, so fake Discord calls can tell whether a table lock is held
HTTP = []  # (kind, a table lock was held) for every fake Discord call


def http(kind):
    locks = list(ONECARD.runtime.locks.values()) + list(ONECARD.bot.games.runtime.locks.values()) if ONECARD is not None else []
    HTTP.append((kind, any(lock.locked() for lock in locks)))


def seats_of(n):
    return [Seat(k, "member", k + 1, 0) for k in range(n)]


def pick_move(state):
    """A simple deterministic player: the first play (sorted), else draw."""
    seat = state.turn
    moves = sorted(m for m in onecard.legal_moves(state, seat) if m.startswith("play:"))
    return seat, (moves[0] if moves else "draw")


def pure_winner(seed, n_seats=3, limit=2000):
    state = onecard.new(seed, seats_of(n_seats))
    for _ in range(limit):
        if state.finished:
            return state.winner
        state = onecard.apply(state, *pick_move(state))
    raise AssertionError("scripted game did not finish")


@functools.lru_cache(maxsize=None)
def table_seed():
    """A 3-seat deal where seat 0 holds a 2 and a 7 of the top card's suit (both playable first) and seat 1 has no answer to a 2."""
    for i in range(20000):
        seed = f"onecard-d1-{i}"
        state = onecard.new(seed, seats_of(3))
        top = state.discard[-1]
        answers = any(c[0] in ("2", "JK") or (c[0] == "A" and c[1] == top[1]) for c in state.hands[1])
        if ("2", top[1]) in state.hands[0] and ("7", top[1]) in state.hands[0] and not answers:
            return seed
    raise AssertionError("no suitable seed")


def card_move(state, seat, rank, suit=None):
    """The play move for the seat's ``rank`` of the top card's suit (a 7 names ``suit``)."""
    top = state.discard[-1]
    return onecard.play_move((rank, top[1]), suit)


def played_cards(moves):
    return [m[5:].partition("=")[0] for m in moves if m.startswith("play:")]


class OneCardCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        global ONECARD
        self.bot = SkitBot(make_settings())
        self.store, self.games = self.bot.store, self.bot.games
        self.addCleanup(self.store.close)
        self.onecard = ONECARD = getattr(self.bot, "onecard", None)
        self.assertIsNotNone(self.onecard, MISSING)
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
        s.set_game_rules(G, "onecard", {"ante": ANTE})
        self.world = s.create_space(G, "W", "world")
        s.bind_channel(G, CH, self.world)  # CH is bound to a world; CH2 stays unbound
        self.ann = s.create_character(G, self.world, "Ann")
        s.change_character_balance(G, self.ann, 100, "start", 1)
        s.set_cast(CH, None, [self.ann])
        s.add_favorite(G, ALICE, self.ann)
        for user in (ALICE, BOB, CARA, DAN, *range(20, 30)):
            s.change_balance(G, user, 100, "start", 1)
        s.set_game_character_talk(G, False)
        self.bot.models.structured_compiled = mock.AsyncMock(side_effect=RuntimeError("no model in One Card tests"))
        for name in ("LOBBY_SECONDS", "ARCHIVE_SECONDS", "BOARD_DEBOUNCE_SECONDS"):
            setattr(self.onecard, name, 3600)
        self.onecard.runtime.retry_seconds = 3600
        self.addAsyncCleanup(self.onecard.close)
        self.addAsyncCleanup(self.bot.doubt.close)
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

    async def start(self, user=ALICE, *, channel=None, favorites=False, bet=None, game="onecard"):
        click = self.click(user, channel=channel or self.ch)
        await invoke(self.bot, "game start", click, game=game, bet=bet, favorites=favorites)
        return click

    async def push(self, user, label, message):
        """The user presses a button of the message the way Discord delivers it: a component click in the message's channel."""
        click = self.click(user, channel=message.host, message=message)
        await button(message, label).callback(click)
        return click

    async def lobby2(self):
        await self.start()
        await self.push(BOB, "Join", self.lobby())

    async def lobby3(self):
        await self.lobby2()
        await self.push(CARA, "Join", self.lobby())

    async def playing(self):
        await self.lobby3()
        await self.push(ALICE, "Start", self.lobby())

    async def pick_play(self, user, move, moves=None):
        snap, click = self.snap(), self.click(user, channel=self.thread())
        await self.onecard.play_from_picker(click, snap["table_id"], snap["round_id"], snap["moves"] if moves is None else moves, move)
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
        return list(self.store.seat_view(G, self.snap()["round_id"], seat).hand)

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
        return onecard.replay(rnd["seed"], seats, moves, json.loads(rnd["rules_json"]))

    def store_play(self, seat, move, actor="member"):
        """A move made straight in the store (the bot reads the table fresh on its next step)."""
        snap = self.snap()
        return self.store.play(G, snap["round_id"], seat, move, actor, snap["moves"])

    def playable(self, seat):
        """The distinct card texts with a legal play for the seat, by the engine."""
        return {m[5:].partition("=")[0] for m in onecard.legal_moves(self.engine_state(), seat) if m != "draw"}

    def thread_texts(self, thread=None):
        return [m.content for m in (thread or self.thread()).sent if not m.deleted and m.content]

    def everything_public(self):
        return "\n".join([self.lobby().content, self.board().content, self.turn().content, *self.thread_texts()])

    async def play_out(self, limit=2000):
        """Plays the table's scripted game to its end through the bot, one picker play per move."""
        thread, table_id = self.thread(), self.table()["id"]  # the table is closed once the game is over, so read it by id
        for n in range(limit):
            snap = self.store.latest_round(G, table_id)
            state = self.engine_state(snap)
            if state.finished:
                return n
            self.assertEqual(snap["moves"], n)
            seat, move = pick_move(state)
            await self.onecard.play_from_picker(self.click(MEMBERS[seat], channel=thread), table_id, snap["round_id"], n, move)
        self.fail("the scripted game did not finish")


class LobbyTests(OneCardCase):
    async def test_start_posts_a_public_lobby_and_seats_the_opener(self):
        click = await self.start()
        content, kwargs = click.response.sent[0]
        self.assertFalse(kwargs.get("ephemeral"))
        self.assertEqual(kwargs["allowed_mentions"].to_dict(), NO_MENTIONS)
        self.assertIn("One Card", content)
        self.assertIn(f"<@{ALICE}>", content)
        self.assertIn("ante", content.lower())
        self.assertIn("pot", content.lower())
        lobby = self.lobby()
        self.assertEqual(set(labels(lobby)), {"Join", "Leave", "Start", "How to play"})
        self.assertTrue(item_of(button(lobby, "Start")).disabled)
        snap = self.snap()
        self.assertEqual((snap["game"], snap["status"], snap["rules"]["ante"]), ("onecard", "joining", ANTE))
        self.assertEqual(self.seats(snap), [("member", ALICE, None)])
        self.assertEqual(self.balance(ALICE), 100 - ANTE)
        self.assertIn(((G, snap["table_id"]), "lobby"), self.onecard.runtime.timers)
        self.assertEqual(self.store.ledger(G, user_id=ALICE)[0]["reason"], ANTE_REASON)

    async def test_a_public_table_thread_is_opened_on_the_lobby_message(self):
        await self.start()
        table = self.table()
        self.assertEqual(list(self.channel.threads), [table["thread_id"]])
        thread = self.thread()
        self.assertTrue(thread.name.startswith("One Card"))
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
        self.assertIn("refunded", warning[0][0])
        self.assertTrue(warning[0][1].get("ephemeral"))

    async def test_only_missing_permissions_name_the_permissions(self):
        self.channel.thread_error = discord.HTTPException(mock.Mock(status=500, reason="boom"), "boom")
        click = await self.start()
        text = " ".join(says(click))
        self.assertNotIn("Create Public Threads", text)
        self.assertIn("thread", text.lower())
        self.assertIn("refunded", text)
        self.assertIsNone(self.table())
        self.assertEqual(self.balance(ALICE), 100)

    async def test_no_refund_words_when_there_is_no_ante(self):
        self.store.set_game_rules(G, "onecard", {"ante": 0})
        self.channel.thread_error = forbidden()
        click = await self.start()
        self.assertNotIn("refund", " ".join(says(click)).lower())
        self.channel.thread_error = None
        await self.lobby2()
        self.assertIn("No ante", self.lobby().content)
        snap = self.snap()
        await self.onecard.on_lobby_timer(G, snap["table_id"], snap["round_id"])
        self.assertNotIn("refund", self.channel.messages[snap["message_id"]].content.lower())

    async def test_a_bet_for_one_card_gets_the_ante_note(self):
        click = await self.start(bet=25)
        self.assertIn("One Card uses the server's ante; no bet needed.", says(click))
        self.assertIsNone(self.table())
        self.assertEqual(self.balance(ALICE), 100)

    async def test_refused_in_non_game_channels_and_when_the_game_is_off(self):
        click = await self.start(channel=FakeGameChannel(555))
        self.assertEqual(says(click), [OFF_HERE])
        self.store.set_game_enabled(G, "onecard", False)
        click = await self.start()
        self.assertEqual(says(click), ["One Card is off on this server."])
        self.assertTrue(private(click))
        self.assertIsNone(self.table())
        self.assertEqual(self.balance(ALICE), 100)

    async def test_the_lobby_has_no_fill_seats_button_even_with_favorites(self):
        await self.start(favorites=True)
        self.assertNotIn("Fill seats", labels(self.lobby()))

    async def test_another_games_table_blocks_the_channel_both_ways(self):
        await self.start()
        click = await self.start(BOB, game="doubt")
        self.assertTrue(says(click) and private(click))
        self.assertEqual((self.table()["game"], self.balance(BOB)), ("onecard", 100))
        click = self.click(BOB)
        await invoke(self.bot, "blackjack", click, bet=10)
        self.assertTrue(says(click) and private(click))
        self.assertEqual((self.table()["game"], self.balance(BOB), self.seats()), ("onecard", 100, [("member", ALICE, None)]))
        self.store.close_table(G, self.table()["id"])
        self.store.set_game_rules(G, "doubt", {"ante": ANTE})
        await self.start(CARA, game="doubt")
        click = await self.start(BOB)
        self.assertTrue(says(click) and private(click))
        self.assertEqual(self.table()["game"], "doubt")

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

    async def test_leave_mentions_the_refund_only_with_an_ante(self):
        await self.lobby2()
        click = await self.push(BOB, "Leave", self.lobby())
        self.assertIn("refunded", " ".join(says(click)))
        self.store.set_game_rules(G, "onecard", {"ante": 0})
        self.store.close_table(G, self.table()["id"])
        await self.start()
        await self.push(BOB, "Join", self.lobby())
        click = await self.push(BOB, "Leave", self.lobby())
        self.assertEqual(says(click), ["You left the table."])

    async def test_the_last_player_leaving_closes_the_table_with_the_refund(self):
        await self.start()
        key = (G, self.table()["id"])
        lobby = self.lobby()
        await self.push(ALICE, "Leave", lobby)
        self.assertIsNone(self.table())
        self.assertEqual(self.balance(ALICE), 100)
        self.assertIn("closed", lobby.content.lower())
        self.assertIsNone(lobby.view)
        await self.onecard.on_archive_timer(*key)
        self.assertNotIn(key, self.onecard.tables)

    async def test_joining_without_the_ante_is_refused(self):
        await self.start()
        self.store.change_balance(G, DAN, -95, "spend", 1)
        click = await self.push(DAN, "Join", self.lobby())
        self.assertIn("need", " ".join(says(click)))
        self.assertEqual(self.seats(), [("member", ALICE, None)])
        self.assertEqual(self.balance(DAN), 5)

    async def test_at_most_eight_seats(self):
        await self.start()
        for user in range(20, 27):
            await self.push(user, "Join", self.lobby())
        self.assertEqual(len(self.seats()), onecard.MAX_SEATS)
        click = await self.push(27, "Join", self.lobby())
        self.assertTrue(says(click) and private(click))
        self.assertEqual(len(self.seats()), onecard.MAX_SEATS)
        self.assertEqual(self.balance(27), 100)

    async def test_start_is_for_the_host_and_needs_two_seats(self):
        await self.start()
        lobby = self.lobby()
        self.assertTrue(item_of(button(lobby, "Start")).disabled)
        click = await self.push(ALICE, "Start", lobby)
        self.assertIn("2", " ".join(says(click)))
        self.assertEqual(self.snap()["status"], "joining")
        await self.push(BOB, "Join", lobby)
        self.assertFalse(item_of(button(lobby, "Start")).disabled)
        click = await self.push(BOB, "Start", lobby)
        self.assertIn("host", " ".join(says(click)).lower())
        self.assertTrue(private(click))
        self.assertEqual(self.snap()["status"], "joining")
        await self.push(ALICE, "Start", lobby)  # two seats are enough
        self.assertEqual(self.snap()["status"], "playing")

    async def test_when_the_host_left_the_first_seated_member_hosts(self):
        await self.lobby3()
        lobby = self.lobby()
        self.assertIn(f"Host: <@{ALICE}>", lobby.content)
        await self.push(ALICE, "Leave", lobby)
        self.assertIn(f"Host: <@{BOB}>", lobby.content)
        click = await self.push(CARA, "Start", lobby)
        self.assertEqual(self.snap()["status"], "joining")
        self.assertTrue(says(click))
        await self.push(BOB, "Start", lobby)
        self.assertEqual(self.snap()["status"], "playing")

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
        await self.push(BOB, "Join", self.lobby())
        self.assertTrue(item_of(button(self.lobby(), "Start")).disabled)
        gate.set()
        await task
        self.assertFalse(item_of(button(self.lobby(), "Start")).disabled)

    async def test_how_to_play_button_answers_privately_for_anyone(self):
        await self.start()
        click = await self.push(DAN, "How to play", self.lobby())
        self.assertTrue(private(click))
        self.assertIn("One Card", "\n".join(says(click)))

    async def test_an_idle_lobby_closes_with_refunds(self):
        await self.lobby3()
        snap = self.snap()
        await self.onecard.on_lobby_timer(G, snap["table_id"], snap["round_id"])
        self.assertIsNone(self.table())
        self.assertEqual([self.balance(u) for u in MEMBERS], [100, 100, 100])
        lobby = self.channel.messages[snap["message_id"]]
        self.assertIn("closed", lobby.content.lower())
        self.assertIn("refunded", lobby.content)
        self.assertIsNone(lobby.view)

    async def test_a_lobby_timer_after_the_start_does_nothing(self):
        await self.playing()
        snap = self.snap()
        await self.onecard.on_lobby_timer(G, snap["table_id"], snap["round_id"])
        self.assertEqual(self.snap()["status"], "playing")
        self.assertNotIn(((G, snap["table_id"]), "lobby"), self.onecard.runtime.timers)

    async def test_a_join_restarts_the_idle_lobby_clock(self):
        await self.start()
        key = ((G, self.table()["id"]), "lobby")
        first = self.onecard.runtime.timers[key]
        self.onecard.clock = lambda: 5000.0
        await self.push(BOB, "Join", self.lobby())
        self.assertIsNot(self.onecard.runtime.timers[key], first)
        self.assertIn("<t:8600:R>", self.lobby().content)

    async def test_stale_or_foreign_buttons_are_refused(self):
        await self.start()
        snap = self.snap()
        cases = [(G, OneCardButton("join", snap["table_id"], snap["round_id"] + 50, 0)),  # unknown round
                 (G, OneCardButton("join", snap["table_id"] + 1, snap["round_id"], 0)),  # table does not match round
                 (2, OneCardButton("join", snap["table_id"], snap["round_id"], 0))]  # another server
        for guild, item in cases:
            click = FakeGameClick(self.bot, BOB, guild_id=guild, channel=self.channel, on_call=http)
            await item.callback(click)
            self.assertEqual(says(click), [STALE], item.item.custom_id)
        self.assertEqual(len(self.snap()["seats"]), 1)

    async def test_a_doubt_button_cannot_drive_a_one_card_table(self):
        """Buttons carry a table id, so a doubt-prefixed button naming a One Card table must not act on it."""
        await self.start()
        snap = self.snap()
        from llmcord_core.games_doubt_discord import DoubtButton
        click = self.click(BOB)
        await DoubtButton("join", snap["table_id"], snap["round_id"], 0).callback(click)
        self.assertEqual(says(click), [STALE])
        self.assertEqual(len(self.snap()["seats"]), 1)


class ComponentTests(OneCardCase):
    def test_custom_id_round_trip_and_no_seat_in_it(self):
        self.assertIsNotNone(OneCardButton, MISSING)
        item = OneCardButton("hand", 3, 4, 5, "My hand")
        self.assertEqual(item.item.custom_id, "llmcord:onecard:hand:3:4:5")
        template = OneCardButton.__discord_ui_compiled_template__
        match = template.fullmatch(item.item.custom_id)
        self.assertEqual((match["action"], match["table"], match["round"], match["moves"]), ("hand", "3", "4", "5"))
        self.assertIsNone(template.fullmatch("llmcord:onecard:hand:3:4:5:2"))  # a seat index cannot ride along
        self.assertIsNone(template.fullmatch("llmcord:onecard:hand:x:4:5"))
        self.assertIsNone(template.fullmatch("llmcord:doubt:hand:3:4:5"))  # other games' ids stay theirs
        self.assertIsNone(template.fullmatch("llmcord:bj:hit:3:4:5"))
        self.assertIsNone(GameButton.__discord_ui_compiled_template__.fullmatch("llmcord:onecard:hand:3:4:5"))
        from llmcord_core.games_doubt_discord import DoubtButton
        self.assertIsNone(DoubtButton.__discord_ui_compiled_template__.fullmatch("llmcord:onecard:hand:3:4:5"))

    async def test_a_button_calls_the_tables_press_with_its_ids(self):
        with mock.patch.object(self.onecard, "press", new=mock.AsyncMock()) as press:
            interaction = self.click(BOB)
            await OneCardButton("start", 3, 4, 5).callback(interaction)
        press.assert_awaited_once_with(interaction, "start", 3, 4, 5)

    async def test_buttons_are_registered_with_the_bot_and_recovery_runs_at_startup(self):
        self.bot.tree.sync = mock.AsyncMock()
        self.games.recover, self.bot.doubt.recover, self.onecard.recover = mock.AsyncMock(), mock.AsyncMock(), mock.AsyncMock()
        await self.bot.setup_hook()
        self.addCleanup(self.bot.cleanup_task.cancel)
        registered = list(self.bot._connection._view_store._dynamic_items.values())
        self.assertIn(OneCardButton, registered)
        self.assertIn(GameButton, registered)
        self.onecard.recover.assert_awaited_once()

    async def test_closing_the_bot_cancels_the_one_card_timers(self):
        await self.start()
        task = self.onecard.runtime.timers[((G, self.table()["id"]), "lobby")]
        with mock.patch.object(SkitBot.__mro__[1], "close", new=mock.AsyncMock()):
            self.bot.models.close = mock.AsyncMock()
            self.store.close = lambda: None
            await self.bot.close()
        self.assertTrue(task.cancelled())
        self.assertEqual(self.onecard.runtime.timers, {})


class CommandTests(OneCardCase):
    def test_game_start_and_rules_offer_one_card_after_the_existing_games(self):
        for path in ("game start", "game rules"):
            choices = [(c.name, c.value) for c in command(self.bot, path)._params["game"].choices]
            self.assertEqual(choices, [("Blackjack", "blackjack"), ("I Doubt It", "doubt"), ("Texas Hold'em", "holdem"), ("One Card", "onecard"), ("Yut (윷놀이)", "yut")], path)
        self.assertLessEqual(len(self.bot.tree.get_command("game").commands), 25)

    async def test_game_start_one_card_goes_through_the_one_card_lobby(self):
        await self.start()
        self.assertEqual(self.table()["game"], "onecard")
        self.assertIn(f"table {self.table()['id']}", self.lobby().content)
        self.assertTrue(all(item_of(child).custom_id.startswith("llmcord:onecard:") for child in self.lobby().view.children))
        self.assertIn(((G, self.table()["id"]), "lobby"), self.onecard.runtime.timers)
        self.assertEqual({k for k in self.bot.doubt.runtime.timers}, set())

    async def test_game_rules_answers_privately_with_the_guide_and_the_rules(self):
        click = self.click(DAN)
        await invoke(self.bot, "game rules", click, game="onecard")
        self.assertTrue(private(click))
        text = "\n".join(says(click))
        self.assertIn("One Card", text)
        self.assertIn(str(self.store.game_rules(G, "onecard")["turn_timeout"]), text)

    async def test_game_hand_works_in_the_table_thread(self):
        await self.playing()
        thread = self.thread()
        click = self.click(BOB, channel=thread)
        await invoke(self.bot, "game hand", click)
        self.assertTrue(private(click))
        for card in self.hand(1):
            self.assertIn(onecard.card_text(card), blob(click))
        for card in self.hand(0) + self.hand(2):
            self.assertNotIn(onecard.card_text(card), blob(click))
        stranger = self.click(DAN, channel=thread)
        await invoke(self.bot, "game hand", stranger)
        self.assertEqual(says(stranger), [NOT_SEATED])
        elsewhere = self.click(BOB, channel=self.channel)
        await invoke(self.bot, "game hand", elsewhere)
        self.assertTrue(says(elsewhere) and private(elsewhere))

    async def test_my_hand_words_before_and_after_the_game(self):
        await self.lobby3()
        thread = self.thread()
        click = self.click(BOB, channel=thread)
        await invoke(self.bot, "game hand", click)
        self.assertEqual(says(click), ["The game hasn't started yet."])
        snap = self.snap()
        await self.onecard.on_lobby_timer(G, snap["table_id"], snap["round_id"])
        click = self.click(BOB, channel=thread)
        await self.onecard._show_hand(click, (G, snap["table_id"]), snap["round_id"])
        self.assertEqual(says(click), ["This game is over."])


class StartAndBoardTests(OneCardCase):
    async def test_start_deals_and_posts_a_pinned_board_and_a_turn_message(self):
        self.onecard.clock = lambda: 5000.0
        await self.lobby3()
        thread = self.thread()
        self.assertEqual(thread.sent, [])
        await self.push(ALICE, "Start", self.lobby())
        snap = self.snap()
        self.assertEqual((snap["status"], snap["view"]["counts"]), ("playing", (7, 7, 7)))
        table = self.table()
        board, turn = thread.messages[table["board_message_id"]], thread.messages[table["turn_message_id"]]
        self.assertNotEqual(board.id, turn.id)
        self.assertEqual([m.id for m in thread.sent], [board.id, turn.id])
        self.assertTrue(board.pinned)
        self.assertFalse(turn.pinned)
        for user in MEMBERS:
            self.assertIn(f"<@{user}>", board.content)
        self.assertIn("7 cards", board.content)
        self.assertIn("30", board.content)  # the pot
        self.assertIn(onecard.card_text(snap["view"]["top"]), board.content)
        self.assertIn("clockwise", board.content)
        self.assertNotIn("counter-clockwise", board.content)
        self.assertNotIn("Pending attack", board.content)
        self.assertIn(f"<@{ALICE}>", turn.content)  # first on turn
        self.assertIn(onecard.card_text(snap["view"]["top"]), turn.content)
        self.assertIn("<t:5060:R>", turn.content)  # the default turn timeout of 60 s
        self.assertEqual(labels(turn), ["My hand"])
        self.assertNotIn(((G, snap["table_id"]), "lobby"), self.onecard.runtime.timers)
        self.assertIn(((G, snap["table_id"]), "turn"), self.onecard.runtime.timers)

    async def test_the_lobby_points_to_the_thread_once_play_starts(self):
        await self.playing()
        self.assertIn("In play", self.lobby().content)
        self.assertEqual(labels(self.lobby()), ["How to play"])

    async def test_a_failed_pin_does_not_stop_the_game(self):
        await self.lobby3()
        self.thread().pin_error = forbidden()
        await self.push(ALICE, "Start", self.lobby())
        self.assertEqual(self.snap()["status"], "playing")
        self.assertFalse(self.board().pinned)
        self.assertTrue(self.turn().content)

    async def test_a_thread_that_cannot_be_written_closes_the_table_with_refunds(self):
        await self.lobby3()
        lobby = self.lobby()
        self.thread().send_error = forbidden()
        await self.push(ALICE, "Start", lobby)
        self.assertIsNone(self.table())
        self.assertEqual([self.balance(u) for u in MEMBERS], [100, 100, 100])
        self.assertIn("refunded", lobby.content)
        self.assertIsNone(lobby.view)

    async def test_no_card_of_any_hand_is_public_at_the_start(self):
        await self.playing()
        dealt = {seat: [onecard.card_text(c) for c in self.hand(seat)] for seat in range(3)}
        public = self.everything_public()
        for seat, cards in dealt.items():
            for card in cards:
                self.assertNotIn(card, public, f"seat {seat} hand leaked")

    async def test_board_marks_and_attack_suit_direction_are_rendered_from_the_view(self):
        await self.playing()
        key = (G, self.table()["id"])
        snap = copy.deepcopy(self.snap())
        snap["view"].update(counts=(1, 2, 7), declared=(False, True, False), exposed=(0,), pending=3, suit="♣", direction=-1)
        text = self.onecard.render_board(key, snap)
        line = {user: next(row for row in text.splitlines() if f"<@{user}>" in row) for user in MEMBERS}
        self.assertIn("1 card", line[ALICE])
        self.assertNotIn("1 cards", line[ALICE])
        self.assertIn("exposed", line[ALICE].lower())
        self.assertIn("2 cards", line[BOB])
        self.assertIn("declared", line[BOB].lower())
        self.assertNotIn("exposed", line[BOB].lower())
        self.assertIn("7 cards", line[CARA])
        self.assertTrue(all(word not in line[CARA].lower() for word in ("exposed", "declared")))
        self.assertIn("Pending attack: 3", text)
        self.assertIn("Suit: ♣", text)
        self.assertIn("counter-clockwise", text)

    async def test_a_seven_shows_the_named_suit_and_an_attack_shows_the_pending_total(self):
        await self.playing()
        state = self.engine_state()
        await self.pick_play(ALICE, card_move(state, 0, "7", "♣"))
        await self.onecard.flush()
        self.assertIn("Suit: ♣", self.board().content)
        self.assertIn("7", self.board().content)
        self.assertIn("♣", self.turn().content)

    async def test_an_attack_card_shows_pending_on_the_board_until_it_is_drawn(self):
        await self.playing()
        await self.pick_play(ALICE, card_move(self.engine_state(), 0, "2"))
        await self.onecard.flush()
        self.assertIn("Pending attack: 2", self.board().content)
        await self.pick_play(BOB, "draw")  # no answer in hand: takes both
        await self.onecard.flush()
        self.assertNotIn("Pending attack", self.board().content)

    async def test_the_board_is_debounced_and_follows_the_play(self):
        await self.playing()
        board = self.board()
        before = board.content
        await self.pick_play(ALICE, card_move(self.engine_state(), 0, "2"))
        await self.pick_play(BOB, "draw")
        self.assertEqual(board.edits, [])  # nothing is sent inside the debounce window
        await self.onecard.flush()
        self.assertEqual(len(board.edits), 1)  # two events, one edit
        self.assertNotEqual(board.content, before)
        await self.onecard.flush()
        self.assertEqual(len(board.edits), 1)

    async def test_the_turn_message_is_edited_in_place_and_hides_what_was_drawn(self):
        await self.playing()
        turn, count = self.turn(), len(self.thread().sent)
        state = self.engine_state()
        move = card_move(state, 0, "2")
        await self.pick_play(ALICE, move)
        self.assertEqual(self.table()["turn_message_id"], turn.id)
        self.assertEqual(len(self.thread().sent), count)
        self.assertIn(f"<@{ALICE}> played {move[5:]}", turn.content)
        self.assertIn(f"Next: <@{BOB}>", turn.content)
        before = {c for seat in range(3) for c in self.hand(seat)}
        await self.pick_play(BOB, "draw")
        drawn = [c for c in self.hand(1) if c not in before]
        self.assertEqual(len(drawn), 2)  # the pending 2 are drawn
        self.assertIn(f"<@{BOB}> drew 2 cards", turn.content)
        self.assertIn(f"Next: <@{CARA}>", turn.content)
        for card in drawn:
            self.assertNotIn(onecard.card_text(card), self.everything_public())

    async def test_the_turn_deadline_follows_the_rounds_turn_timeout(self):
        self.store.set_game_rules(G, "onecard", {"ante": ANTE, "turn_timeout": 15})
        self.onecard.clock = lambda: 5000.0
        await self.playing()
        self.assertEqual(self.snap()["rules"]["turn_timeout"], 15)
        self.assertIn("<t:5015:R>", self.turn().content)

    async def test_a_round_keeps_the_rules_it_was_opened_with(self):
        await self.start()
        self.store.set_game_rules(G, "onecard", {"ante": 50, "turn_timeout": 30})
        click = await self.push(BOB, "Join", self.lobby())
        self.assertTrue(private(click))
        self.assertEqual(self.balance(BOB), 100 - ANTE)
        self.assertEqual((self.snap()["rules"]["ante"], self.snap()["rules"]["turn_timeout"]), (ANTE, 60))
        await self.push(ALICE, "Start", self.lobby())
        self.assertIn(((G, self.table()["id"]), "turn"), self.onecard.runtime.timers)


class HandTests(OneCardCase):
    async def test_my_hand_shows_only_the_pressers_own_cards(self):
        await self.playing()
        everyone = {seat: self.hand(seat) for seat in range(3)}
        for seat, user in enumerate(MEMBERS):
            click = await self.push(user, "My hand", self.turn())
            self.assertTrue(private(click))
            text = blob(click)
            for card in everyone[seat]:
                self.assertIn(onecard.card_text(card), text)
            for other in range(3):
                if other != seat:
                    for card in everyone[other]:
                        self.assertNotIn(onecard.card_text(card), text)
        self.assertEqual(self.thread_texts(), [self.board().content, self.turn().content])  # nothing public

    async def test_the_picker_offers_exactly_the_legal_cards_and_a_draw_button(self):
        await self.playing()
        on_turn = await self.push(ALICE, "My hand", self.turn())
        (picker,) = [view for view in views(on_turn) if selects(view)]
        (chosen,) = selects(picker)
        expected = self.playable(0)
        self.assertIn(onecard.card_text(("2", self.engine_state().discard[-1][1])), expected)
        self.assertEqual({option.value for option in chosen.options}, expected)
        self.assertEqual(len(chosen.options), len(expected))  # a 7 is one option, its suit is a second step
        self.assertTrue(0 < len(chosen.options) <= 25)
        held = {onecard.card_text(c) for c in self.hand(0)}
        self.assertTrue(expected < held)  # something in the hand is not playable and is not offered
        self.assertIn("Draw", [item_of(child).label.split()[0] for child in picker.children if not isinstance(child, discord.ui.Select)])
        self.assertEqual(self.snap()["moves"], 0)

    async def test_a_seat_not_on_turn_gets_the_hand_without_a_picker(self):
        await self.playing()
        waiting = await self.push(BOB, "My hand", self.turn())
        self.assertTrue(says(waiting))
        for view in views(waiting):
            self.assertEqual(selects(view), [])
            self.assertNotIn("Draw", [str(item_of(child).label).split()[0] for child in view.children])

    async def test_attack_pending_is_shown_only_to_the_seat_on_turn(self):
        await self.playing()
        await self.pick_play(ALICE, card_move(self.engine_state(), 0, "2"))
        bob = await self.push(BOB, "My hand", self.turn())
        self.assertIn("Attack pending", blob(bob))
        alice = await self.push(ALICE, "My hand", self.turn())
        self.assertNotIn("Attack pending", blob(alice))

    async def test_hand_and_help_refuse_a_table_of_another_game(self):
        await self.start()
        self.store.close_table(G, self.table()["id"])
        self.store.set_game_rules(G, "doubt", {"ante": ANTE})
        await self.start(CARA, game="doubt")
        other = self.table()
        self.assertEqual(other["game"], "doubt")
        rnd = self.store.latest_round(G, other["id"])["round_id"]
        self.assertEqual(self.onecard._hand((G, other["id"]), rnd, CARA), (STALE, None))
        self.assertEqual(self.onecard._hand((G, other["id"]), None, CARA), (STALE, None))
        self.assertEqual(self.onecard._help_text(G, other["id"], rnd), STALE)

    async def test_under_an_attack_the_picker_offers_only_the_draw_when_nothing_answers(self):
        await self.playing()
        await self.pick_play(ALICE, card_move(self.engine_state(), 0, "2"))
        click = await self.push(BOB, "My hand", self.turn())
        self.assertEqual(self.playable(1), set())
        (picker,) = views(click)
        self.assertEqual(selects(picker), [])
        draw = [item_of(child) for child in picker.children if str(item_of(child).label).startswith("Draw")]
        self.assertEqual(len(draw), 1)

    async def test_a_named_suit_limits_the_picker_to_that_suit(self):
        await self.playing()
        await self.pick_play(ALICE, card_move(self.engine_state(), 0, "7", "♣"))
        click = await self.push(BOB, "My hand", self.turn())
        expected = self.playable(1)
        self.assertTrue(expected and all(card.endswith("♣") or card.startswith("JK") for card in expected))
        (picker,) = [view for view in views(click) if selects(view)]
        (chosen,) = selects(picker)
        self.assertEqual({option.value for option in chosen.options}, expected)

    async def test_the_picker_plays_a_legal_card_for_the_pressing_seat(self):
        await self.playing()
        move = card_move(self.engine_state(), 0, "2")
        click = await self.pick_play(ALICE, move)
        self.assertEqual(self.moves(), [(0, move, "member")])
        self.assertEqual(self.snap()["view"]["counts"], (6, 7, 7))
        self.assertEqual(self.snap()["view"]["turn"], 1)
        self.assertEqual(self.snap()["view"]["pending"], 2)
        self.assertTrue(says(click) and private(click))
        self.assertIn(((G, self.table()["id"]), "turn"), self.onecard.runtime.timers)

    async def test_a_seven_names_its_suit_in_the_move(self):
        await self.playing()
        move = card_move(self.engine_state(), 0, "7", "♠")
        await self.pick_play(ALICE, move)
        self.assertEqual(self.moves(), [(0, move, "member")])
        self.assertEqual(self.snap()["view"]["suit"], "♠")

    async def test_drawing_takes_a_card_and_passes_the_turn(self):
        await self.playing()
        click = await self.pick_play(ALICE, "draw")
        self.assertEqual(self.moves(), [(0, "draw", "member")])
        self.assertEqual(self.snap()["view"]["counts"], (8, 7, 7))
        self.assertEqual(self.snap()["view"]["turn"], 1)
        self.assertTrue(says(click) and private(click))
        self.assertIn(f"<@{ALICE}> drew a card", self.turn().content)

    async def test_illegal_moves_are_refused_and_change_nothing(self):
        await self.playing()
        state = self.engine_state()
        mine = {onecard.card_text(c): c for c in state.hands[0]}
        unplayable = next(text for text in mine if text not in self.playable(0))
        theirs = next(onecard.card_text(c) for c in state.hands[1] if onecard.card_text(c) not in mine)
        bad = ["", "play:", "play:ZZ", f"play:{unplayable}", f"play:{theirs}", f"play:{unplayable}=♠", card_move(state, 0, "7"),  # a 7 without its suit
               card_move(state, 0, "7", "x"), "declare", "call_out:1", "timeout", "forfeit", "DRAW", 5, None]
        for move in bad:
            click = await self.pick_play(ALICE, move)
            self.assertTrue(says(click) and private(click), move)
        self.assertEqual(self.moves(), [])

    async def test_a_member_cannot_play_or_see_for_another_seat(self):
        await self.playing()
        click = await self.pick_play(BOB, "draw")  # Alice is on turn, not Bob
        self.assertEqual(says(click), [NOT_YOUR_TURN])
        self.assertTrue(private(click))
        self.assertEqual(self.moves(), [])
        click = await self.pick_play(DAN, "draw")  # not seated at all
        self.assertEqual(says(click), [NOT_SEATED])
        click = await self.push(DAN, "My hand", self.turn())
        self.assertEqual(says(click), [NOT_SEATED])
        for seat in range(3):
            for card in self.hand(seat):
                self.assertNotIn(onecard.card_text(card), blob(click))
        self.assertEqual(self.moves(), [])

    async def test_a_stale_picker_says_the_table_moved_on(self):
        await self.playing()
        click = await self.pick_play(ALICE, "draw", moves=3)
        self.assertTrue(any(text.startswith("The table moved on") for text in says(click)))
        self.assertTrue(private(click))
        self.assertEqual(self.moves(), [])
        await self.pick_play(ALICE, "draw")
        late = await self.pick_play(ALICE, "draw", moves=0)  # the picker built before Alice's own draw
        self.assertTrue(any(text.startswith("The table moved on") for text in says(late)))
        self.assertEqual(len(self.moves()), 1)

    async def test_a_double_press_applies_once(self):
        await self.playing()
        snap = self.snap()
        clicks = [self.click(ALICE, channel=self.thread()), self.click(ALICE, channel=self.thread())]
        await asyncio.gather(*(self.onecard.play_from_picker(c, snap["table_id"], snap["round_id"], 0, "draw") for c in clicks))
        self.assertEqual(self.moves(), [(0, "draw", "member")])
        self.assertEqual(sum(1 for c in clicks if any(t.startswith("The table moved on") for t in says(c))), 1)

    async def test_a_picker_for_a_closed_or_foreign_table_is_stale(self):
        await self.playing()
        snap, thread = self.snap(), self.thread()
        foreign = FakeGameClick(self.bot, ALICE, guild_id=2, channel=thread, on_call=http)
        await self.onecard.play_from_picker(foreign, snap["table_id"], snap["round_id"], 0, "draw")
        self.assertEqual(says(foreign), [STALE])
        self.store.close_table(G, snap["table_id"])
        late = self.click(ALICE, channel=thread)  # the closed table can no longer be read through self.snap()
        await self.onecard.play_from_picker(late, snap["table_id"], snap["round_id"], 0, "draw")
        self.assertEqual(says(late), [STALE])
        self.assertEqual(self.moves(), [])


class TimeoutTests(OneCardCase):
    async def test_a_missed_turn_draws_and_passes(self):
        await self.playing()
        snap = self.snap()
        await self.onecard.on_turn_timer(G, snap["table_id"], snap["round_id"], 0)
        self.assertEqual(self.moves(), [(0, "timeout", "timeout")])
        self.assertEqual(self.snap()["view"]["counts"], (8, 7, 7))
        self.assertEqual(self.snap()["view"]["turn"], 1)
        self.assertEqual(list(self.snap()["view"]["active"]), [True, True, True])
        self.assertTrue(any("out of time" in text.lower() and f"<@{ALICE}>" in text for text in self.thread_texts()))
        self.assertIn(f"Next: <@{BOB}>", self.turn().content)
        self.assertIn(((G, snap["table_id"]), "turn"), self.onecard.runtime.timers)  # the next seat's clock
        await self.onecard.on_turn_timer(G, snap["table_id"], snap["round_id"], 0)  # stale
        self.assertEqual(len(self.moves()), 1)

    async def test_a_missed_turn_under_an_attack_draws_the_whole_pile(self):
        await self.playing()
        await self.pick_play(ALICE, card_move(self.engine_state(), 0, "2"))
        snap = self.snap()
        await self.onecard.on_turn_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        self.assertEqual(self.moves()[-1], (1, "timeout", "timeout"))
        self.assertEqual(self.snap()["view"]["counts"], (6, 9, 7))
        self.assertEqual(self.snap()["view"]["pending"], 0)

    async def test_a_second_missed_turn_in_a_row_forfeits_the_seat(self):
        await self.playing()
        tid, rid = self.snap()["table_id"], self.snap()["round_id"]
        await self.onecard.on_turn_timer(G, tid, rid, 0)
        self.assertEqual(list(self.snap()["view"]["active"]), [True, True, True])  # one miss is forgiven
        self.store_play(1, "draw")
        self.store_play(2, "draw")
        await self.onecard.on_turn_timer(G, tid, rid, 3)  # seat 0 again
        self.assertEqual(self.moves()[-1], (0, "forfeit", "timeout"))
        snap = self.snap()
        self.assertEqual(list(snap["view"]["active"]), [False, True, True])
        self.assertEqual((snap["status"], snap["view"]["turn"]), ("playing", 1))
        self.assertTrue(any("forfeit" in text.lower() and f"<@{ALICE}>" in text for text in self.thread_texts()))
        self.assertIn(f"Next: <@{BOB}>", self.turn().content)

    async def test_a_move_of_your_own_resets_the_missed_turns(self):
        await self.playing()
        tid, rid = self.snap()["table_id"], self.snap()["round_id"]
        await self.onecard.on_turn_timer(G, tid, rid, 0)  # Alice misses
        self.store_play(1, "draw")
        self.store_play(2, "draw")
        await self.pick_play(ALICE, "draw")  # she plays herself: the count starts over
        self.store_play(1, "draw")
        self.store_play(2, "draw")
        await self.onecard.on_turn_timer(G, tid, rid, self.snap()["moves"])
        self.assertEqual(self.moves()[-1], (0, "timeout", "timeout"))  # not a forfeit yet
        self.assertEqual(list(self.snap()["view"]["active"]), [True, True, True])

    async def test_a_late_timer_after_the_member_moved_does_nothing(self):
        await self.playing()
        snap = self.snap()
        await self.pick_play(ALICE, "draw")
        await self.onecard.on_turn_timer(G, snap["table_id"], snap["round_id"], 0)
        self.assertEqual(len(self.moves()), 1)

    async def test_a_timer_for_a_lobby_or_a_closed_table_does_nothing(self):
        await self.lobby3()
        snap = self.snap()
        await self.onecard.on_turn_timer(G, snap["table_id"], snap["round_id"], 0)
        self.assertEqual((self.moves(), self.snap()["status"]), ([], "joining"))
        await self.push(ALICE, "Start", self.lobby())
        self.store.close_table(G, snap["table_id"])
        await self.onecard.on_turn_timer(G, snap["table_id"], snap["round_id"], 0)
        self.assertEqual(self.moves(), [])

    async def test_two_seats_forfeiting_in_turn_leave_the_last_one_the_winner_and_pay_the_pot(self):
        await self.playing()
        tid, rid = self.snap()["table_id"], self.snap()["round_id"]
        await self.onecard.on_turn_timer(G, tid, rid, 0)  # Alice's first miss
        self.store_play(1, "draw")
        self.store_play(2, "draw")
        await self.onecard.on_turn_timer(G, tid, rid, 3)  # Alice forfeits; Bob is next
        await self.onecard.on_turn_timer(G, tid, rid, 4)  # Bob's first miss
        self.store_play(2, "draw")
        await self.onecard.on_turn_timer(G, tid, rid, 6)  # Bob's second miss: Cara is alone
        snap = self.store.round_snapshot(G, rid)
        self.assertEqual(snap["status"], "settled")
        self.assertEqual([self.balance(u) for u in MEMBERS], [90, 90, 120])
        self.assertEqual([s["outcome"] for s in snap["seats"]], ["forfeit", "forfeit", "win"])
        self.assertIsNone(self.store.open_table_for(G, CH))
        await self.onecard.on_turn_timer(G, tid, rid, snap["moves"])  # a late timer on a finished round is harmless
        self.assertEqual([self.balance(u) for u in MEMBERS], [90, 90, 120])


class SettlementTests(OneCardCase):
    async def test_a_full_game_pays_the_pot_and_wraps_up_the_table(self):
        await self.playing()
        table, thread = self.table(), self.thread()
        tid, rid, lobby, board, turn = table["id"], self.snap()["round_id"], self.lobby(), self.board(), self.turn()
        await self.play_out()
        await self.onecard.flush()
        winner = MEMBERS[pure_winner(self.seed)]
        snap = self.store.round_snapshot(G, rid)
        self.assertEqual(snap["status"], "settled")
        self.assertEqual(snap["view"]["winner"], MEMBERS.index(winner))
        self.assertEqual({u: self.balance(u) for u in MEMBERS}, {u: 120 if u == winner else 90 for u in MEMBERS})
        payout = self.store.ledger(G, user_id=winner)[0]
        self.assertEqual((payout["amount"], payout["reason"]), (30, PAYOUT_REASON))
        result = [t for m in thread.sent if m.id not in (board.id, turn.id) and not m.deleted for t in [m.content]
                  if f"<@{winner}>" in t and "30" in t and "pot" in t.lower()]
        self.assertTrue(result, self.thread_texts(thread))
        self.assertNotIn("\n", lobby.content)  # a one-line result
        self.assertIn(f"<@{winner}>", lobby.content)
        self.assertIn("won One Card", lobby.content)
        self.assertIsNone(lobby.view)
        self.assertIn("Game over", turn.content)
        self.assertIsNone(turn.view)
        self.assertFalse(turn.deleted)
        self.assertTrue(board.edits)
        self.assertIsNone(self.store.open_table_for(G, CH))
        self.assertIn(((G, tid), "archive"), self.onecard.runtime.timers)
        self.assertEqual({k for k in self.onecard.runtime.timers if k[1] != "archive"}, set())
        self.assertFalse(thread.archived)
        await self.onecard.on_archive_timer(G, tid)
        self.assertTrue(thread.archived and thread.locked)

    async def test_result_text_tells_bankrupt_seats_from_forfeits(self):
        await self.playing()
        tid, rid = self.snap()["table_id"], self.snap()["round_id"]
        await self.onecard.on_turn_timer(G, tid, rid, 0)
        self.store_play(1, "draw")
        self.store_play(2, "draw")
        await self.onecard.on_turn_timer(G, tid, rid, 3)
        await self.onecard.on_turn_timer(G, tid, rid, 4)
        self.store_play(2, "draw")
        await self.onecard.on_turn_timer(G, tid, rid, 6)
        snap = self.store.round_snapshot(G, rid)
        text = self.onecard._result((G, tid), snap)[0]
        self.assertEqual(text.count("(forfeited)"), 2)
        self.assertNotIn("bankrupt", text)
        view = dict(snap["view"], history=[*snap["view"]["history"], {"kind": "bankrupt", "seat": 1}])
        text = self.onecard._result((G, tid), dict(snap, view=view))[0]
        self.assertEqual((text.count("(forfeited)"), text.count("(bankrupt)")), (1, 1))
        self.assertIn(f"<@{BOB}>: -10 (bankrupt)", text.replace("−", "-"))

    async def test_a_game_without_an_ante_settles_with_no_money_moving(self):
        self.store.set_game_rules(G, "onecard", {"ante": 0})
        await self.playing()
        await self.play_out()
        self.assertEqual([self.balance(u) for u in MEMBERS], [100, 100, 100])
        self.assertEqual(self.store.round_snapshot(G, 1)["status"], "settled")

    async def test_a_failed_archive_is_not_fatal(self):
        await self.playing()
        tid, thread = self.table()["id"], self.thread()
        await self.play_out()
        thread.thread_edit_error = forbidden()
        await self.onecard.on_archive_timer(G, tid)
        self.assertFalse(thread.archived)

    async def test_a_game_that_ends_normally_is_not_repainted_by_the_sweeper(self):
        await self.playing()
        lobby, thread = self.lobby(), self.thread()
        await self.play_out()
        text = lobby.content
        await self.onecard.sweep()
        self.assertEqual(lobby.content, text)
        self.assertEqual([t for t in self.thread_texts(thread) if t.startswith("Table closed:")], [])

    async def test_a_two_seat_game_plays_to_a_win(self):
        await self.lobby2()
        await self.push(ALICE, "Start", self.lobby())
        self.assertEqual(self.snap()["view"]["counts"], (7, 7))
        rid = self.snap()["round_id"]
        thread, tid = self.thread(), self.table()["id"]
        for n in range(2000):
            snap = self.store.latest_round(G, tid)
            state = self.engine_state(snap)
            if state.finished:
                break
            seat, move = pick_move(state)
            await self.onecard.play_from_picker(self.click((ALICE, BOB)[seat], channel=thread), tid, rid, n, move)
        else:
            self.fail("the scripted two-seat game did not finish")
        winner = (ALICE, BOB)[pure_winner(self.seed, 2)]
        self.assertEqual({u: self.balance(u) for u in (ALICE, BOB)}, {u: 110 if u == winner else 90 for u in (ALICE, BOB)})


class OutsideCloseTests(OneCardCase):
    """Tables closed from outside (dashboard switch, channel removal, restart) and their refunds."""

    async def grab(self):
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
        self.assertEqual({k for k in self.onecard.runtime.timers if k[1] != "archive"}, set())
        self.assertIn(((G, tid), "archive"), self.onecard.runtime.timers)

    async def test_a_dashboard_switch_off_closes_a_table_in_play_through_the_sweeper(self):
        await self.playing()
        tid, lobby, thread, turn = await self.grab()
        HTTP.clear()
        self.store.set_game_enabled(G, "onecard", False)
        await self.onecard.sweep()
        self.assert_cleaned(tid, lobby, thread, turn, "One Card was turned off on this server.")
        self.assertIn("refunded", lobby.content)
        self.assertEqual([self.balance(u) for u in MEMBERS], [100, 100, 100])
        self.assertFalse(any(held for _, held in HTTP))
        count = len(thread.sent)
        await self.onecard.sweep()
        self.assertEqual(len(thread.sent), count)

    async def test_turning_games_off_in_the_channel_closes_a_one_card_table_with_its_own_words(self):
        await self.lobby3()
        lobby = self.lobby()
        closed, paint = await self.games.switch_channel(G, CH, False)
        self.assertTrue(closed)
        self.assertIsNone(paint)
        self.assertIn("turned off", lobby.content)
        self.assertIsNone(lobby.view)
        self.assertEqual([self.balance(u) for u in MEMBERS], [100, 100, 100])
        self.assertEqual({k for k in self.onecard.runtime.timers if k[1] != "archive"}, set())

    async def test_removing_the_channel_closes_the_lobby_with_its_own_words(self):
        await self.lobby3()
        tid, lobby, thread, turn = await self.grab()
        self.store.set_game_channels(G, {CH2})
        await self.onecard.sweep()
        self.assert_cleaned(tid, lobby, thread, turn, "games were turned off here.")

    async def test_a_generic_close_and_no_refund_words_without_an_ante(self):
        self.store.set_game_rules(G, "onecard", {"ante": 0})
        await self.lobby3()
        tid, lobby, thread, turn = await self.grab()
        self.store.close_table(G, tid)
        await self.onecard.sweep()
        self.assert_cleaned(tid, lobby, thread, turn, "the table was closed.")
        self.assertNotIn("refund", lobby.content.lower())

    async def test_a_press_on_a_table_closed_from_outside_cleans_up_at_once(self):
        await self.lobby3()
        tid, lobby, thread, turn = await self.grab()
        self.store.set_game_enabled(G, "onecard", False)
        click = await self.push(DAN, "Join", lobby)
        self.assertEqual(says(click), [STALE])
        self.assert_cleaned(tid, lobby, thread, turn, "turned off")

    async def test_a_timer_that_finds_the_table_closed_cleans_up_at_once(self):
        await self.playing()
        tid, lobby, thread, turn = await self.grab()
        snap = self.snap()
        self.store.set_game_channels(G, {CH2})
        self.onecard.arm((G, tid), "turn", 0, snap["round_id"], snap["moves"])  # replaces the real turn timer with an immediate one
        await asyncio.wait_for(self.onecard.runtime.timers[((G, tid), "turn")], 5)
        self.assert_cleaned(tid, lobby, thread, turn, "games were turned off here.")

    async def test_closing_a_channel_whose_table_closed_already_still_cleans_it_up(self):
        await self.lobby3()
        tid, lobby, thread, turn = await self.grab()
        self.store.close_table(G, tid)
        closed = await self.onecard.close_for_channel(G, CH, tid)
        self.assertIsNone(closed)
        self.assertIn("Table closed", lobby.content)
        self.assertIsNone(lobby.view)

    async def test_a_step_sealed_before_an_outside_close_cannot_repaint_the_closed_table(self):
        await self.playing()
        tid, lobby, thread, turn = await self.grab()
        key, snap = (G, tid), self.snap()
        async with self.onecard.runtime.lock(key):  # a pick, sealed but not yet delivered
            late = self.onecard._picker_play(key, ALICE, snap["round_id"], snap["moves"], "draw")
        self.assertTrue(late.ok)
        self.store.set_game_enabled(G, "onecard", False)
        await self.onecard.sweep()
        sent = [m.content for m in thread.sent]
        await self.onecard._publish(key, late)
        self.assertEqual(turn.content, "Table closed.")
        self.assertIsNone(lobby.view)
        self.assertEqual([m.content for m in thread.sent], sent)

    async def test_the_sweeper_is_started_with_the_first_table_and_stopped_with_close(self):
        await self.lobby3()
        tid, lobby, thread, turn = await self.grab()
        sweeper = self.onecard.sweeper
        self.assertIsNotNone(sweeper)
        self.assertFalse(sweeper.done())
        await self.onecard.close()
        self.assertTrue(sweeper.cancelled())

    async def test_restart_refunds_closes_and_leaves_a_note_in_each_thread(self):
        await self.playing()
        self.ch = self.channel2
        await self.start(DAN)  # a second lobby with one seat
        self.ch = self.channel
        lobby, thread = self.lobby(), self.thread()
        lobby2, thread2 = self.channel2.messages[self.table(CH2)["message_id"]], self.channel2.threads[self.table(CH2)["thread_id"]]
        await self.onecard.recover()
        self.assertEqual([self.balance(u) for u in (*MEMBERS, DAN)], [100] * 4)
        self.assertEqual(self.store.unfinished_rounds(), [])
        self.assertIsNone(self.table(CH))
        self.assertIsNone(self.table(CH2))
        for message in (lobby, lobby2):
            self.assertIn("restarted", message.content.lower())
            self.assertIn("antes were refunded", message.content)
            self.assertIsNone(message.view)
        for placed in (thread, thread2):
            self.assertTrue(any("restarted" in text.lower() for text in self.thread_texts(placed)))
            self.assertTrue(placed.archived and placed.locked)
        self.assertEqual(self.onecard.runtime.timers, {})

    async def test_recovery_still_refunds_when_nothing_can_be_reached(self):
        await self.playing()
        self.bot.get_channel = lambda ident: None
        self.bot.fetch_channel = mock.AsyncMock(side_effect=not_found("Unknown Channel", 10003))
        await self.onecard.recover()
        self.assertEqual([self.balance(u) for u in MEMBERS], [100, 100, 100])
        self.assertIsNone(self.table(CH))

    async def test_recovery_with_nothing_open_is_quiet(self):
        with self.assertNoLogs(level="WARNING"):
            await self.onecard.recover()
        self.assertEqual(self.channel.calls, [])

    async def test_each_game_recovers_only_its_own_tables(self):
        await self.playing()
        for other in (self.bot.doubt.recover, self.games.recover, self.bot.holdem.recover):
            await other()
        self.assertEqual(self.snap()["status"], "playing")
        self.assertEqual([self.balance(u) for u in MEMBERS], [90, 90, 90])
        self.assertEqual(self.thread_texts(), [self.board().content, self.turn().content])
        self.ch = self.channel2
        self.store.set_game_rules(G, "doubt", {"ante": ANTE})
        await self.start(DAN, game="doubt")
        await self.onecard.recover()
        self.assertEqual([self.balance(u) for u in MEMBERS], [100, 100, 100])
        self.assertIsNone(self.table(CH))
        self.assertEqual(self.table(CH2)["game"], "doubt")  # One Card's recovery leaves the doubt lobby alone
        self.assertEqual(self.balance(DAN), 90)

    async def test_every_close_path_forgets_the_table_after_the_archive_timer(self):
        await self.start()
        key = (G, self.table()["id"])
        self.assertIn(key, self.onecard.tables)
        await self.push(ALICE, "Leave", self.lobby())
        await self.onecard.on_archive_timer(*key)
        self.assertNotIn(key, self.onecard.tables)


class ChatAndLockTests(OneCardCase):
    async def chatter(self, humans, thread=None):
        for _ in range(humans):
            await self.onecard.note_thread_message(thread_message(thread or self.thread(), BOB))

    async def test_the_turn_message_is_reposted_once_eight_messages_bury_it(self):
        await self.playing()
        thread, first = self.thread(), self.table()["turn_message_id"]
        await self.chatter(7)
        await self.pick_play(ALICE, "draw")
        self.assertEqual(self.table()["turn_message_id"], first)  # seven people talked: not buried yet
        self.assertFalse(thread.messages[first].deleted)
        await self.chatter(8)
        await self.pick_play(BOB, "draw")
        second = self.table()["turn_message_id"]
        self.assertNotEqual(second, first)
        self.assertTrue(thread.messages[first].deleted)
        self.assertEqual(thread.sent[-1].id, second)  # at the bottom
        self.assertIn("My hand", labels(thread.messages[second]))
        self.assertIn(f"<@{CARA}>", thread.messages[second].content)

    async def test_messages_elsewhere_do_not_count_and_the_bot_hands_thread_messages_to_the_table(self):
        await self.playing()
        other = await self.channel.make_thread("elsewhere")
        await self.chatter(10, thread=other)
        first = self.table()["turn_message_id"]
        await self.pick_play(ALICE, "draw")
        self.assertEqual(self.table()["turn_message_id"], first)
        for _ in range(8):
            await self.bot.on_message(thread_message(self.thread(), BOB))
        await self.pick_play(BOB, "draw")
        self.assertNotEqual(self.table()["turn_message_id"], first)

    def assert_never_under_a_lock(self, expected):
        self.assertTrue(expected <= {kind for kind, _ in HTTP}, sorted({kind for kind, _ in HTTP}))
        self.assertEqual([kind for kind, held in HTTP if held], [])

    async def test_lobby_turns_and_timers_make_no_discord_call_under_a_table_lock(self):
        await self.start()
        lobby = self.lobby()
        await self.push(BOB, "Join", lobby)
        await self.push(BOB, "Leave", lobby)
        await self.push(DAN, "Leave", lobby)  # a refusal
        await self.push(BOB, "Join", lobby)
        await self.push(ALICE, "Start", lobby)
        await self.push(DAN, "My hand", self.turn())
        await self.push(ALICE, "My hand", self.turn())
        await self.pick_play(ALICE, "draw")
        await self.onecard.flush()
        snap = self.snap()
        await self.onecard.on_turn_timer(G, snap["table_id"], snap["round_id"], snap["moves"])
        self.assert_never_under_a_lock({"create_thread", "send", "edit", "pin", "defer", "followup"})

    async def test_a_whole_game_to_its_archive_makes_no_discord_call_under_a_table_lock(self):
        await self.playing()
        tid = self.table()["id"]
        await self.play_out()
        await self.onecard.flush()
        await self.onecard.on_archive_timer(G, tid)
        self.assert_never_under_a_lock({"create_thread", "send", "edit", "pin"})

    async def test_thread_creation_failure_makes_no_call_under_a_lock_either(self):
        self.channel.thread_error = forbidden()
        await self.start()
        self.assert_never_under_a_lock({"create_thread"})


if __name__ == "__main__":
    unittest.main()
