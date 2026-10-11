"""FEAT-33 part C1a: Texas Hold'em on Discord, tables and seats (`/game start game:holdem`, lobby, Sit / Leave / Fill seats / Assist /
Deal / Close table, buy-in and rebuy, guild isolation, restart recovery). Dealing and hand play are C1b and not covered here.

TDD-red tests written before the implementation (plain failing tests, like ``test_doubt_discord.py``). Offline: fake channels, threads and
interactions from ``helpers``. Timers are driven by calling their hooks, never by waiting.

Assumed API (pinned here, to be created in ``llmcord_core/games_holdem_discord.py`` and wired in ``discord_bot.py``; spec: "Hold'em Discord
layer (FEAT-33 part C)" in docs/engineering/minigames-plan.md):

  ``bot.holdem = HoldemTables(bot, clock=time.time)``; ``runtime`` (a ``TableRuntime``: ``timers`` keyed ``((guild_id, table_id), kind)``),
  class constants ``LOBBY_SECONDS=300 ARCHIVE_SECONDS=600 BOARD_DEBOUNCE_SECONDS=1``; ``recover()``, ``close()``,
  ``close_for_channel(guild_id, channel_id, table_id)``, ``on_lobby_timer(guild_id, table_id, round_id)``.
  ``/game start game:holdem [favorites]``, ``/game rules game:holdem`` and ``/game hand`` (in the table thread) go through the existing ``game`` group.
  ``HoldemButton(action, table_id, round_id, moves, label="", style=..., disabled=False)``: a DynamicItem with custom id
  ``llmcord:holdem:<action>:<table>:<round>:<moves>``; ``callback`` calls ``interaction.client.holdem.press(interaction, action, table_id, round_id, moves)``.
  Lobby labels: Sit, Leave, Fill seats (bound channels only), Assist on / Assist off (only when the server allows Assist), Deal, Close table.
  Sit (and Rebuy) answer a press with ``interaction.response.send_modal(modal)``: a ``discord.ui.Modal`` with one ``TextInput``; the
  member's answer is delivered to ``modal.on_submit(interaction)``. The Sit modal's default is 100 big blinds and it shows the allowed range.
"""
import unittest
from unittest import mock

import discord
from helpers import FakeGameChannel, FakeGameClick, forbidden, invoke, make_settings, not_found

from llmcord_core.discord_bot import SkitBot
from llmcord_core.games import holdem
from llmcord_core.games_discord import NO_CHARACTER_SEATS, OFF_HERE, STALE, GameButton

try:  # imported softly so every test fails on its own until the module exists
    from llmcord_core.games_holdem_discord import HoldemButton
except ImportError:  # pragma: no cover - the red state
    HoldemButton = None

G, G2, CH, CH2, CH3 = 1, 2, 100, 101, 300
ALICE, BOB, CARA, DAN = 5, 6, 7, 8
NO_MENTIONS = discord.AllowedMentions.none().to_dict()
BUY = "Texas Hold'em buy-in (table 1)"
REBUY = "Texas Hold'em rebuy (table 1)"
CASH = "Texas Hold'em cash-out (table 1)"
NO_BET_NOTE = "Texas Hold'em uses the server's blinds; you choose your buy-in when you sit."
LOW, DEFAULT, HIGH = 400, 1000, 2000  # buy-in range and default (100 big blinds) with the default 5/10 blinds
WALLET, CHAR_WALLET = 10000, 5000


def item_of(child):
    return getattr(child, "item", child)


def labels(message):
    return [item_of(child).label for child in message.view.children] if message is not None and message.view else []


def button(message, label):
    for child in message.view.children if message.view else ():
        if item_of(child).label == label:
            return child
    raise AssertionError(f"no {label!r} button among {labels(message)}")


def says(*clicks):
    return [content for click in clicks for content, _ in click.response.sent + click.followup.sent if content]


def private(*clicks):
    entries = [entry for click in clicks for entry in click.response.sent + click.followup.sent]
    return bool(entries) and all(kw.get("ephemeral") or kw.get("_edit") for _, kw in entries)


def text_input(modal):
    inputs = [child for child in modal.children if isinstance(child, discord.ui.TextInput)]
    assert len(inputs) == 1, f"expected one text input, found {len(inputs)}"
    return inputs[0]


class HoldemCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.holdem = getattr(self.bot, "holdem", None)
        self.channel, self.channel2, self.channel3 = FakeGameChannel(CH), FakeGameChannel(CH2), FakeGameChannel(CH3, guild_id=G2)
        self.ch = self.channel
        self.bot.get_channel = self.find_channel
        self.bot.fetch_channel = mock.AsyncMock(side_effect=self.fetch_channel)
        s = self.store
        for guild, channel in ((G, CH), (G, CH2), (G2, CH3)):
            s.set_game_channel(guild, channel, True)
            s.set_daily_settings(guild, 0, 0, 7)
        self.world = s.create_space(G, "W", "world")
        s.bind_channel(G, CH, self.world)  # CH is bound to a world; CH2 stays unbound (humans-only tables)
        self.ann, self.bo, self.cy = (s.create_character(G, self.world, name) for name in ("Ann", "Bo", "Cy"))
        for character in (self.ann, self.bo, self.cy):
            s.change_character_balance(G, character, CHAR_WALLET, "start", 1)
        s.set_cast(CH, None, [self.ann])
        s.add_favorite(G, ALICE, self.ann)
        for user in (ALICE, BOB, CARA, DAN):
            s.change_balance(G, user, WALLET, "start", 1)
            s.change_balance(G2, user, WALLET, "start", 1)
        self.addCleanup(s.close)
        self.bot.models.structured_compiled = mock.AsyncMock(side_effect=RuntimeError("no model in C1a tests"))
        if self.holdem is not None:
            for name in ("LOBBY_SECONDS", "ARCHIVE_SECONDS", "BOARD_DEBOUNCE_SECONDS"):
                setattr(self.holdem, name, 3600)
            self.holdem.runtime.retry_seconds = 3600
            self.addAsyncCleanup(self.holdem.close)
        self.addAsyncCleanup(self.bot.games.close)
        self.addAsyncCleanup(self.bot.doubt.close)

    # --- fake Discord ------------------------------------------------------------------------

    def find_channel(self, ident):
        for channel in (self.channel, self.channel2, self.channel3):
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

    def click(self, user, *, channel=None, message=None, guild=G):
        return FakeGameClick(self.bot, user, guild_id=guild, channel=channel or self.ch, message=message)

    # --- driving the bot ---------------------------------------------------------------------

    async def start(self, user=ALICE, *, channel=None, favorites=False, bet=None, guild=G):
        click = self.click(user, channel=channel or self.ch, guild=guild)
        await invoke(self.bot, "game start", click, game="holdem", bet=bet, favorites=favorites)
        return click

    async def push(self, user, label, message=None, *, guild=G):
        """The user presses a button of the message the way Discord delivers it: a component click in the message's channel."""
        message = message or self.lobby()
        click = self.click(user, channel=message.host, message=message, guild=guild)
        await button(message, label).callback(click)
        return click

    async def ask(self, user, label, text, message=None):
        """Presses a button that opens a modal, types ``text`` into it and submits. Returns (press click, submit click or None)."""
        message = message or self.lobby()
        press = await self.push(user, label, message)
        if not press.response.modals:
            return press, None
        modal = press.response.modals[-1]
        text_input(modal)._value = text
        submit = self.click(user, channel=message.host, message=message)
        await modal.on_submit(submit)
        return press, submit

    async def sit(self, user, amount=DEFAULT):
        press, submit = await self.ask(user, "Sit", str(amount))
        return [click for click in (press, submit) if click is not None]

    # --- reading state -----------------------------------------------------------------------

    def table(self, channel_id=None):
        return self.store.open_table_for(G, channel_id or self.ch.id)

    def snap(self, table_id=None):
        return self.store.latest_round(G, table_id or self.table()["id"])

    def lobby(self):
        return self.ch.messages[self.table()["message_id"]]

    def thread(self):
        return self.ch.threads[self.table()["thread_id"]]

    def seats(self, snap=None):
        return [(s["kind"], s["ref_id"], s["brought_by"], s["stack"]) for s in (snap or self.snap())["seats"]]

    def balance(self, user, guild=G):
        return self.store.balance(guild, user)

    def reasons(self, guild=G):
        return [r["reason"] for r in self.store.db.execute("SELECT reason FROM currency_ledger WHERE guild_id=? AND source='game' ORDER BY id", (guild,))]

    def name(self):
        return self.store.currency_name(G)


class StartTests(HoldemCase):
    async def test_start_posts_a_public_lobby_with_the_controls(self):
        """C1a: /game start game:holdem opens a joining round and posts one public lobby message with the rules and the lobby buttons."""
        click = await self.start()
        content, kwargs = click.response.sent[0]
        self.assertFalse(kwargs.get("ephemeral"))
        self.assertEqual(kwargs["allowed_mentions"].to_dict(), NO_MENTIONS)
        self.assertIn("Texas Hold'em", content)
        self.assertIn(f"<@{ALICE}>", content)  # the host
        self.assertIn("5/10", content)  # the blinds
        lobby = self.lobby()
        self.assertLessEqual({"Sit", "Leave", "Fill seats", "Deal", "Close table"}, set(labels(lobby)))
        self.assertNotIn("Assist on", labels(lobby))  # the server does not allow Assist
        self.assertNotIn("Assist: on", lobby.content)
        self.assertTrue(item_of(button(lobby, "Deal")).disabled)  # nobody (fewer than two) is seated
        snap = self.snap()
        self.assertEqual((snap["game"], snap["status"]), ("holdem", "joining"))
        self.assertEqual(self.seats(snap), [])
        self.assertIn(((G, snap["table_id"]), "lobby"), self.holdem.runtime.timers)
        self.assertEqual(self.balance(ALICE), WALLET)  # nothing is bought in until the host sits

    async def test_a_public_table_thread_is_opened_on_the_lobby_message(self):
        """C1a: the table thread is created on the lobby message and registered for the table."""
        await self.start()
        table = self.table()
        self.assertEqual(list(self.channel.threads), [table["thread_id"]])
        thread = self.thread()
        self.assertTrue(thread.name.startswith("Texas Hold'em"))
        self.assertIn(f"table {table['id']}", thread.name)
        self.assertEqual(self.store.table_for_thread(G, thread.id)["id"], table["id"])
        self.assertEqual(self.channel.calls.count("create_thread"), 1)

    async def test_thread_creation_failure_closes_the_table_refunds_and_names_the_permissions(self):
        """C1a: a thread that cannot be made closes the table (favorites buy-ins cashed out) and the reply names the missing permissions."""
        self.channel.thread_error = forbidden()
        click = await self.start(favorites=True)
        self.assertIsNone(self.table())
        self.assertEqual(self.balance(ALICE), WALLET)
        self.assertEqual(self.store.character_balance(G, self.ann), CHAR_WALLET)
        self.assertEqual(self.store.open_cash_tables(), [])
        warning = [(content, kw) for content, kw in click.response.sent + click.followup.sent if content and "Create Public Threads" in content]
        self.assertEqual(len(warning), 1)
        self.assertIn("Send Messages in Threads", warning[0][0])
        self.assertTrue(warning[0][1].get("ephemeral"))

    async def test_a_bet_gets_the_blinds_note_and_opens_nothing(self):
        """C1a: a bet is not a Hold'em option; the reply says the buy-in is chosen when sitting."""
        click = await self.start(bet=25)
        self.assertIn(NO_BET_NOTE, says(click))
        self.assertTrue(private(click))
        self.assertIsNone(self.table())
        self.assertEqual(self.balance(ALICE), WALLET)

    async def test_refused_in_non_game_channels_and_when_the_game_is_off(self):
        """C1a: error wording: OFF_HERE outside game channels, the store's "off on this server" line when the game is off."""
        click = await self.start(channel=FakeGameChannel(555))
        self.assertEqual(says(click), [OFF_HERE])
        self.store.set_game_enabled(G, "holdem", False)
        click = await self.start()
        self.assertEqual(says(click), ["Texas Hold'em is off on this server."])
        self.assertTrue(private(click))
        self.assertIsNone(self.table())

    async def test_a_second_table_in_the_channel_is_refused(self):
        """C1a: one table per channel; the second /game start gets the store's wording privately and changes nothing."""
        await self.start()
        before = self.table()
        click = await self.start(BOB)
        self.assertEqual(says(click), ["There is already a table open in this channel."])
        self.assertTrue(private(click))
        self.assertEqual(self.table(), before)

    async def test_an_unbound_channel_is_humans_only(self):
        """C1a: in an unbound channel favorites get the note, only the member sits, and there is no Fill seats button."""
        self.ch = self.channel2
        click = await self.start(favorites=True)
        self.assertIn(NO_CHARACTER_SEATS, says(click))
        self.assertEqual([seat[:2] for seat in self.seats()], [("member", ALICE)])
        self.assertNotIn("Fill seats", labels(self.lobby()))
        self.assertEqual(self.store.character_balance(G, self.ann), CHAR_WALLET)

    async def test_favorites_seat_the_host_and_the_favorites_with_the_default_buy_in(self):
        """C1a (design assumption): `favorites` seats the host at the default 100 big blinds and brings favorites at min(that, their wallet, max)."""
        await self.start(favorites=True)
        self.assertEqual(self.seats(), [("member", ALICE, None, DEFAULT), ("character", self.ann, ALICE, DEFAULT)])
        self.assertEqual(self.balance(ALICE), WALLET - DEFAULT)
        self.assertEqual(self.store.character_balance(G, self.ann), CHAR_WALLET - DEFAULT)
        content = self.lobby().content
        self.assertIn(f"(with <@{ALICE}>)", content)
        self.assertEqual(self.reasons().count(BUY), 2)

    async def test_a_brought_favorite_below_the_minimum_buy_in_is_skipped(self):
        """C1a: a favorite whose wallet is under the minimum buy-in stays standing; the host still sits and is told."""
        self.store.change_character_balance(G, self.ann, LOW - 1 - CHAR_WALLET, "spend", 1)
        click = await self.start(favorites=True)
        self.assertEqual(self.seats(), [("member", ALICE, None, DEFAULT)])
        self.assertTrue(any("Ann" in text for text in says(click)))

    async def test_a_poor_favorite_sits_with_what_the_wallet_allows(self):
        """C1a: a favorite with 700 chips buys in for 700 (min of the host's buy-in, the wallet and the max)."""
        self.store.change_character_balance(G, self.ann, 700 - CHAR_WALLET, "spend", 1)
        await self.start(favorites=True)
        self.assertEqual(self.seats()[1], ("character", self.ann, ALICE, 700))

    async def test_a_bot_cannot_open_a_table(self):
        """C1a: bots never take part; nothing opens."""
        click = self.click(ALICE)
        click.user.bot = True
        await invoke(self.bot, "game start", click, game="holdem", bet=None, favorites=False)
        self.assertIsNone(self.table())
        self.assertTrue(says(click))


class RulesAndHandCommandTests(HoldemCase):
    async def test_rules_command_shows_the_servers_blinds_privately(self):
        """C1a: /game rules game:holdem answers with the how-to-play text and the server's rules line."""
        self.store.set_game_rules(G, "holdem", {"small_blind": 10})
        click = self.click(BOB)
        await invoke(self.bot, "game rules", click, game="holdem")
        rules = self.store.game_rules(G, "holdem")
        text = "\n".join(says(click))
        self.assertIn(holdem.how_to_play(rules), text)
        self.assertIn(holdem.rules_text(rules), text)
        self.assertIn("10/20", text)
        self.assertTrue(private(click))

    async def test_hand_command_outside_a_table_thread_points_to_the_thread(self):
        """C1a: /game hand outside any table thread gets the same pointer as for I Doubt It."""
        click = self.click(ALICE)
        await invoke(self.bot, "game hand", click)
        self.assertEqual(says(click), ["Use /game hand in your table's thread."])

    async def test_hand_command_in_a_holdem_thread_before_the_deal_is_private_and_not_a_doubt_hand(self):
        """C1a: in a Hold'em lobby thread there is no hand yet; the reply is private and says so (no I Doubt It text)."""
        await self.start()
        await self.sit(ALICE)
        click = self.click(ALICE, channel=self.thread())
        await invoke(self.bot, "game hand", click)
        self.assertTrue(private(click))
        text = " ".join(says(click)).lower()
        self.assertIn("started", text)
        self.assertNotIn("doubt", text)


class SitTests(HoldemCase):
    async def test_sit_opens_a_modal_with_the_default_and_the_range(self):
        """C1a: Sit answers with a buy-in modal (default 100 big blinds) that shows the allowed range; no money moves yet."""
        await self.start()
        press = await self.push(BOB, "Sit")
        self.assertEqual(len(press.response.modals), 1)
        modal = press.response.modals[0]
        field = text_input(modal)
        self.assertEqual(str(field.default), str(DEFAULT))
        shown = " ".join([str(modal.title), str(field.label), str(field.placeholder or ""), str(field.default or "")]).replace(",", "")
        self.assertIn(str(LOW), shown)
        self.assertIn(str(HIGH), shown)
        self.assertEqual(self.seats(), [])
        self.assertEqual(self.balance(BOB), WALLET)

    async def test_submitting_a_buy_in_seats_the_member_with_that_stack(self):
        """C1a: a valid buy-in debits the wallet (ledger reason), seats the member with stack = stake, and updates the lobby."""
        await self.start()
        before = self.lobby().content
        clicks = await self.sit(BOB, 1500)
        self.assertTrue(private(*clicks))
        self.assertEqual(self.seats(), [("member", BOB, None, 1500)])
        self.assertEqual(self.snap()["seats"][0]["stake"], 1500)
        self.assertEqual(self.balance(BOB), WALLET - 1500)
        self.assertEqual(self.reasons(), [BUY])
        lobby = self.lobby()
        self.assertNotEqual(lobby.content, before)
        self.assertIn(f"<@{BOB}>", lobby.content)
        self.assertIn("1,500", lobby.content)

    async def test_sitting_restarts_the_idle_clock(self):
        """C1a: a seat taken in the lobby re-arms the lobby timer (a new task replaces the old one)."""
        await self.start()
        slot = ((G, self.table()["id"]), "lobby")
        old = self.holdem.runtime.timers[slot]
        await self.sit(BOB)
        self.assertIsNot(self.holdem.runtime.timers[slot], old)

    async def test_deal_unlocks_with_two_seats(self):
        """C1a: Deal is disabled until two seats are taken (the minimum for Hold'em)."""
        await self.start()
        await self.sit(ALICE)
        self.assertTrue(item_of(button(self.lobby(), "Deal")).disabled)
        await self.sit(BOB)
        self.assertFalse(item_of(button(self.lobby(), "Deal")).disabled)

    async def test_invalid_buy_ins_are_refused_and_change_nothing(self):
        """C1a: error wording: out-of-range amounts get the store's range line; non-numbers get a whole-number line; nothing is seated or charged."""
        await self.start()
        name = self.name()
        for text in (str(LOW - 1), str(HIGH + 1), "0", "-500"):
            clicks = await self.sit(BOB, text)
            self.assertEqual(says(*clicks), [f"The buy-in must be between {LOW:,} and {HIGH:,} {name}."], text)
            self.assertTrue(private(*clicks))
        for text in ("abc", "10.5", "", "1e3"):
            clicks = await self.sit(BOB, text)
            reply = " ".join(says(*clicks)).lower()
            self.assertIn("whole number", reply, repr(text))
            self.assertTrue(private(*clicks))
        self.assertEqual(self.seats(), [])
        self.assertEqual(self.balance(BOB), WALLET)
        self.assertEqual(self.reasons(), [])

    async def test_the_range_edges_are_accepted(self):
        """C1a: the minimum and the maximum buy-in both seat."""
        await self.start()
        await self.sit(BOB, LOW)
        await self.sit(CARA, HIGH)
        self.assertEqual([seat[3] for seat in self.seats()], [LOW, HIGH])

    async def test_a_buy_in_the_wallet_cannot_cover_is_refused(self):
        """C1a: error wording: the store's "You need X for that buy-in but have Y" line."""
        await self.start()
        self.store.change_balance(G, DAN, 300 - WALLET, "spend", 1)
        clicks = await self.sit(DAN, 500)
        self.assertEqual(says(*clicks), [f"You need 500 {self.name()} for that buy-in but have 300."])
        self.assertEqual(self.seats(), [])
        self.assertEqual(self.balance(DAN), 300)

    async def test_sitting_twice_is_refused_without_a_second_charge(self):
        """C1a: one seat per member."""
        await self.start()
        await self.sit(BOB, 500)
        clicks = await self.sit(BOB, 600)
        self.assertIn("already", " ".join(says(*clicks)))
        self.assertEqual(self.seats(), [("member", BOB, None, 500)])
        self.assertEqual(self.balance(BOB), WALLET - 500)

    async def test_a_full_table_refuses_the_tenth_seat(self):
        """C1a: nine seats at most; the tenth member is told the table is full."""
        await self.start()
        round_id = self.snap()["round_id"]
        for user in range(50, 59):
            self.store.change_balance(G, user, WALLET, "start", 1)
            self.store.join_round(G, round_id, user, LOW)
        clicks = await self.sit(BOB)
        self.assertIn("The table is full.", says(*clicks))
        self.assertEqual(len(self.snap()["seats"]), 9)

    async def test_a_modal_for_a_table_that_closed_meanwhile_is_stale(self):
        """C1a: submitting a Sit modal after the table closed gets STALE and charges nothing."""
        await self.start()
        press = await self.push(BOB, "Sit")
        self.store.close_table(G, self.table()["id"])
        modal = press.response.modals[0]
        text_input(modal)._value = str(DEFAULT)
        submit = self.click(BOB)
        await modal.on_submit(submit)
        self.assertEqual(says(submit), [STALE])
        self.assertEqual(self.balance(BOB), WALLET)

    async def test_sitting_after_the_buttons_were_dealt_over_is_refused(self):
        """C1a: a Sit press on a lobby message whose round has started gets the store's "already started" line, not a seat."""
        await self.start()
        await self.sit(ALICE)
        await self.sit(BOB)
        lobby = self.lobby()
        self.store.deal(G, self.snap()["round_id"])
        clicks = [c for c in await self.ask(CARA, "Sit", str(DEFAULT), lobby) if c is not None]
        self.assertTrue(says(*clicks))
        self.assertTrue(private(*clicks))
        self.assertEqual(len(self.snap()["seats"]), 2)
        self.assertEqual(self.balance(CARA), WALLET)


class LeaveTests(HoldemCase):
    async def test_leave_cashes_the_stack_out_to_the_wallet(self):
        """C1a: Leave pays the whole stack back (ledger reason), removes the seat, updates the lobby, and records the cash-out."""
        await self.start()
        await self.sit(ALICE)
        await self.sit(BOB, 1500)
        click = await self.push(BOB, "Leave")
        self.assertTrue(private(click))
        self.assertEqual(self.balance(BOB), WALLET)
        self.assertEqual([seat[:2] for seat in self.seats()], [("member", ALICE)])
        self.assertIn(CASH, self.reasons())
        self.assertNotIn(f"<@{BOB}>", self.lobby().content)
        rows = {(r["kind"], r["ref_id"]): r for r in self.store.table_cash_summary(G, self.table()["id"])["cashouts"]}
        self.assertEqual((rows[("member", BOB)]["bought_in"], rows[("member", BOB)]["cashed_out"]), (1500, 1500))

    async def test_leaving_without_a_seat_is_refused(self):
        """C1a: error wording: the store's "You do not have a seat" line, privately."""
        await self.start()
        await self.sit(ALICE)
        click = await self.push(DAN, "Leave")
        self.assertEqual(says(click), ["You do not have a seat in this round."])
        self.assertTrue(private(click))
        self.assertEqual(len(self.seats()), 1)

    async def test_leaving_takes_the_brought_characters_along_and_cashes_them_out(self):
        """C1a: when a member leaves, the favorites they brought leave too and go back to their own wallets."""
        await self.start(favorites=True)
        await self.sit(BOB)
        await self.push(ALICE, "Leave")
        self.assertEqual([seat[:2] for seat in self.seats()], [("member", BOB)])
        self.assertEqual(self.balance(ALICE), WALLET)
        self.assertEqual(self.store.character_balance(G, self.ann), CHAR_WALLET)
        self.assertEqual(self.table()["id"], self.snap()["table_id"])  # the table stays open for the remaining player


class RebuyTests(HoldemCase):
    async def test_rebuy_adds_to_the_stack_and_the_total_bought_in(self):
        """C1a (design assumption: a Rebuy button with an amount modal in the lobby): stack and stake both grow, ledger reason, lobby updated."""
        await self.start()
        await self.sit(BOB, 1500)
        press, submit = await self.ask(BOB, "Rebuy", "300")
        self.assertTrue(private(press, submit))
        snap = self.snap()
        self.assertEqual((snap["seats"][0]["stack"], snap["seats"][0]["stake"]), (1800, 1800))
        self.assertEqual(self.balance(BOB), WALLET - 1800)
        self.assertEqual(self.reasons(), [BUY, REBUY])
        self.assertIn("1,800", self.lobby().content)

    async def test_rebuy_beyond_the_maximum_stack_is_refused(self):
        """C1a: error wording: the store's "A stack can hold at most" line with what can still be added."""
        await self.start()
        await self.sit(BOB, 1500)
        press, submit = await self.ask(BOB, "Rebuy", "600")
        self.assertEqual(says(press, submit), [f"A stack can hold at most {HIGH:,} {self.name()}; you can add 500."])
        self.assertEqual(self.snap()["seats"][0]["stack"], 1500)
        self.assertEqual(self.balance(BOB), WALLET - 1500)

    async def test_rebuy_amounts_must_be_whole_numbers_above_zero(self):
        """C1a: zero, negative and non-numeric amounts are refused privately and change nothing."""
        await self.start()
        await self.sit(BOB, 1500)
        for text in ("0", "-5", "abc", "2.5"):
            press, submit = await self.ask(BOB, "Rebuy", text)
            self.assertTrue(says(press, submit), repr(text))
            self.assertTrue(private(press, submit))
        self.assertEqual(self.snap()["seats"][0]["stack"], 1500)
        self.assertEqual(self.reasons(), [BUY])

    async def test_rebuy_needs_a_seat_and_the_money(self):
        """C1a: someone without a seat cannot rebuy; a short wallet gets the store's line."""
        await self.start()
        press, submit = await self.ask(DAN, "Rebuy", "100")
        self.assertIn("You do not have a seat in this round.", says(press, *([submit] if submit else [])))
        await self.sit(BOB, 500)
        self.store.change_balance(G, BOB, 50 - (WALLET - 500), "spend", 1)
        press, submit = await self.ask(BOB, "Rebuy", "100")
        self.assertEqual(says(press, submit), [f"You need 100 {self.name()} for that rebuy but have 50."])
        self.assertEqual(self.snap()["seats"][0]["stack"], 500)


class FillSeatsTests(HoldemCase):
    async def test_fill_seats_is_for_the_host_only(self):
        """C1a: permission wording: another player pressing Fill seats is told only the host can, and nothing is seated."""
        await self.start()
        await self.sit(ALICE)
        await self.sit(BOB)
        click = await self.push(BOB, "Fill seats")
        self.assertIn("host", " ".join(says(click)).lower())
        self.assertTrue(private(click))
        self.assertEqual(len(self.seats()), 2)

    async def test_host_fills_a_house_seat_from_the_cast_at_100_big_blinds(self):
        """C1a: the host's Fill seats seats a cast character with no brought_by ("house"), buying in 100 big blinds from its own wallet."""
        await self.start()
        await self.sit(ALICE)
        click = await self.push(ALICE, "Fill seats")
        self.assertTrue(private(click))
        self.assertEqual(self.seats()[1], ("character", self.ann, None, DEFAULT))
        self.assertEqual(self.store.character_balance(G, self.ann), CHAR_WALLET - DEFAULT)
        self.assertEqual(self.balance(ALICE), WALLET - DEFAULT)  # the host's wallet is untouched by the house seat
        content = self.lobby().content
        self.assertIn("Ann", content)
        self.assertIn("(house)", content)

    async def test_a_house_character_below_the_minimum_buy_in_is_skipped(self):
        """C1a: a cast character under the minimum buy-in is not seated; the host is told privately and nothing is charged."""
        self.store.change_character_balance(G, self.ann, LOW - 1 - CHAR_WALLET, "spend", 1)
        await self.start()
        await self.sit(ALICE)
        click = await self.push(ALICE, "Fill seats")
        self.assertTrue(says(click) and private(click))
        self.assertEqual(len(self.seats()), 1)
        self.assertEqual(self.store.character_balance(G, self.ann), LOW - 1)

    async def test_a_character_already_seated_is_not_seated_twice(self):
        """C1a: a favorite brought by the host is not also added as a house seat."""
        await self.start(favorites=True)
        click = await self.push(ALICE, "Fill seats")
        self.assertTrue(says(click))
        self.assertEqual([seat[:2] for seat in self.seats()], [("member", ALICE), ("character", self.ann)])
        self.assertEqual(self.store.character_balance(G, self.ann), CHAR_WALLET - DEFAULT)

    async def test_fill_seats_in_an_unbound_channel_is_refused(self):
        """C1a: a Fill seats press on a table whose channel has no world says characters can't sit here."""
        await self.start()
        lobby = self.lobby()
        self.store.execute("DELETE FROM channels WHERE channel_id=?", (CH,))  # the channel loses its world
        click = await self.push(ALICE, "Fill seats", lobby)
        self.assertEqual(says(click), [NO_CHARACTER_SEATS])
        self.assertEqual(self.seats(), [])


class AssistTests(HoldemCase):
    async def test_the_button_appears_only_when_the_server_allows_assist(self):
        """C1a: Assist on shows in the lobby only if the Games setting allows it."""
        self.store.set_game_rules(G, "holdem", {"assist": True})
        await self.start()
        self.assertIn("Assist on", labels(self.lobby()))

    async def test_the_host_turns_assist_on_and_off_and_the_lobby_shows_it(self):
        """C1a: the host's press pins Assist in the round rules; the lobby says "Assist: on" and the button flips."""
        self.store.set_game_rules(G, "holdem", {"assist": True})
        await self.start()
        await self.push(ALICE, "Assist on")
        self.assertTrue(self.snap()["rules"]["assist"])
        lobby = self.lobby()
        self.assertIn("Assist: on", lobby.content)
        self.assertIn("Assist off", labels(lobby))
        await self.push(ALICE, "Assist off")
        self.assertFalse(self.snap()["rules"]["assist"])
        self.assertNotIn("Assist: on", self.lobby().content)

    async def test_only_the_host_can_change_assist(self):
        """C1a: permission wording: a non-host press changes nothing and is answered privately."""
        self.store.set_game_rules(G, "holdem", {"assist": True})
        await self.start()
        await self.sit(ALICE)
        click = await self.push(BOB, "Assist on")
        self.assertTrue(says(click) and private(click))
        self.assertFalse(self.snap()["rules"]["assist"])

    async def test_a_stale_assist_button_is_refused_when_the_server_turned_it_off(self):
        """C1a: error wording: the store's "Assist is not allowed on this server." reaches the host privately."""
        self.store.set_game_rules(G, "holdem", {"assist": True})
        await self.start()
        lobby = self.lobby()
        self.store.set_game_rules(G, "holdem", {"assist": False})
        click = await self.push(ALICE, "Assist on", lobby)
        self.assertEqual(says(click), ["Assist is not allowed on this server."])
        self.assertFalse(self.snap()["rules"]["assist"])


class DealAndCloseTests(HoldemCase):
    """Only the lobby-side checks of Deal (who, how many); dealing itself is C1b."""

    async def test_deal_is_for_the_host_and_needs_two_seats(self):
        """C1a: permission and count wording; neither refusal starts a hand."""
        await self.start()
        await self.sit(ALICE)
        click = await self.push(ALICE, "Deal")
        self.assertTrue(says(click) and private(click))
        self.assertEqual(self.snap()["status"], "joining")
        await self.sit(BOB)
        click = await self.push(BOB, "Deal")
        self.assertIn("host", " ".join(says(click)).lower())
        self.assertTrue(private(click))
        self.assertEqual(self.snap()["status"], "joining")

    async def test_close_table_cashes_every_stack_out_and_closes(self):
        """C1a: the host's Close table pays every seat (members and characters) back, closes the table and removes the lobby buttons."""
        await self.start(favorites=True)
        await self.sit(BOB, 1500)
        table_id = self.table()["id"]
        lobby = self.lobby()
        await self.push(ALICE, "Close table")
        self.assertIsNone(self.table())
        self.assertEqual([self.balance(u) for u in (ALICE, BOB)], [WALLET, WALLET])
        self.assertEqual(self.store.character_balance(G, self.ann), CHAR_WALLET)
        self.assertEqual(self.reasons().count(CASH), 3)
        self.assertIn("closed", lobby.content.lower())
        self.assertIsNone(lobby.view)
        self.assertNotIn(((G, table_id), "lobby"), self.holdem.runtime.timers)

    async def test_only_the_host_can_close_the_table(self):
        """C1a: permission wording: a seated non-host press changes nothing."""
        await self.start()
        await self.sit(ALICE)
        await self.sit(BOB)
        click = await self.push(BOB, "Close table")
        self.assertIn("host", " ".join(says(click)).lower())
        self.assertTrue(private(click))
        self.assertIsNotNone(self.table())
        self.assertEqual(self.balance(BOB), WALLET - DEFAULT)

    async def test_an_idle_lobby_closes_and_cashes_out(self):
        """C1a: after the lobby wait, the table closes, every stack is cashed out, and the lobby message says so without buttons."""
        await self.start(favorites=True)
        await self.sit(BOB, 700)
        snap = self.snap()
        lobby = self.lobby()
        await self.holdem.on_lobby_timer(G, snap["table_id"], snap["round_id"])
        self.assertIsNone(self.table())
        self.assertEqual([self.balance(u) for u in (ALICE, BOB)], [WALLET, WALLET])
        self.assertEqual(self.store.character_balance(G, self.ann), CHAR_WALLET)
        self.assertIn("closed", lobby.content.lower())
        self.assertIsNone(lobby.view)

    async def test_a_lobby_timer_for_a_table_closed_meanwhile_does_nothing(self):
        """C1a: a stale timer neither errors nor touches wallets."""
        await self.start()
        await self.sit(BOB)
        snap = self.snap()
        await self.push(ALICE, "Close table")
        paid = self.reasons().count(CASH)
        await self.holdem.on_lobby_timer(G, snap["table_id"], snap["round_id"])
        self.assertEqual(self.reasons().count(CASH), paid)
        self.assertEqual(self.balance(BOB), WALLET)

    async def test_turning_games_off_in_the_channel_closes_and_cashes_out(self):
        """C1a: close_for_channel (the admin channel switch) closes the table, cashes out, and tells the lobby."""
        await self.start()
        await self.sit(BOB, 900)
        table_id, lobby = self.table()["id"], self.lobby()
        closed = await self.holdem.close_for_channel(G, CH, table_id)
        self.assertEqual(closed, table_id)
        self.assertIsNone(self.table())
        self.assertEqual(self.balance(BOB), WALLET)
        self.assertIn("closed", lobby.content.lower())
        self.assertIsNone(lobby.view)


class ComponentTests(HoldemCase):
    def test_custom_id_round_trip_and_the_other_games_ids_are_not_matched(self):
        """C1a: ids are llmcord:holdem:<action>:<table>:<round>:<moves> and never collide with blackjack or I Doubt It."""
        item = HoldemButton("sit", 3, 4, 0, "Sit")
        self.assertEqual(item.item.custom_id, "llmcord:holdem:sit:3:4:0")
        template = HoldemButton.__discord_ui_compiled_template__
        match = template.fullmatch(item.item.custom_id)
        self.assertEqual((match["action"], match["table"], match["round"], match["moves"]), ("sit", "3", "4", "0"))
        self.assertIsNone(template.fullmatch("llmcord:holdem:sit:x:4:0"))
        self.assertIsNone(template.fullmatch("llmcord:doubt:join:3:4:0"))
        self.assertIsNone(template.fullmatch("llmcord:bj:hit:3:4:5"))
        self.assertIsNone(GameButton.__discord_ui_compiled_template__.fullmatch("llmcord:holdem:sit:3:4:0"))

    async def test_buttons_are_registered_and_recovery_runs_at_startup(self):
        """C1a: setup_hook registers the dynamic button and awaits holdem.recover() once."""
        self.bot.tree.sync = mock.AsyncMock()
        self.bot.games.recover, self.bot.doubt.recover, self.holdem.recover = mock.AsyncMock(), mock.AsyncMock(), mock.AsyncMock()
        await self.bot.setup_hook()
        self.addCleanup(self.bot.cleanup_task.cancel)
        self.assertIn(HoldemButton, list(self.bot._connection._view_store._dynamic_items.values()))
        self.holdem.recover.assert_awaited_once()

    async def test_closing_the_bot_cancels_the_holdem_timers(self):
        """C1a: shutdown cancels the lobby timer and leaves none behind."""
        await self.start()
        task = self.holdem.runtime.timers[((G, self.table()["id"]), "lobby")]
        with mock.patch.object(SkitBot.__mro__[1], "close", new=mock.AsyncMock()):
            self.bot.models.close = mock.AsyncMock()
            self.store.close = lambda: None
            await self.bot.close()
        self.assertTrue(task.cancelled())
        self.assertEqual(self.holdem.runtime.timers, {})

    async def test_presses_outside_a_server_or_by_bots_are_stale(self):
        """C1a: a press with no guild, or from a bot, gets STALE and changes nothing."""
        await self.start()
        lobby = self.lobby()
        dm = FakeGameClick(self.bot, BOB, guild_id=None, channel=self.channel, message=lobby)
        await button(lobby, "Sit").callback(dm)
        self.assertEqual(says(dm), [STALE])
        bot_click = self.click(BOB, message=lobby)
        bot_click.user.bot = True
        await button(lobby, "Sit").callback(bot_click)
        self.assertEqual(says(bot_click), [STALE])
        self.assertEqual(self.seats(), [])


class GuildIsolationTests(HoldemCase):
    async def test_stale_or_foreign_buttons_are_refused(self):
        """C1a: unknown round, mismatched table and another server's press on this server's table all get STALE and seat nobody."""
        await self.start()
        snap = self.snap()
        cases = [(G, HoldemButton("sit", snap["table_id"], snap["round_id"] + 50, 0)),
                 (G, HoldemButton("sit", snap["table_id"] + 1, snap["round_id"], 0)),
                 (G2, HoldemButton("sit", snap["table_id"], snap["round_id"], 0)),
                 (G2, HoldemButton("leave", snap["table_id"], snap["round_id"], 0)),
                 (G2, HoldemButton("close", snap["table_id"], snap["round_id"], 0))]
        for guild, item in cases:
            click = FakeGameClick(self.bot, BOB, guild_id=guild, channel=self.channel)
            await item.callback(click)
            self.assertEqual(says(click), [STALE], item.custom_id)
            self.assertEqual(click.response.modals, [])
        self.assertEqual(self.seats(), [])
        self.assertIsNotNone(self.table())

    async def test_a_modal_opened_for_one_server_cannot_be_submitted_from_another(self):
        """C1a: a Sit modal answered through a different server's interaction seats nobody and charges nobody."""
        await self.start()
        press = await self.push(BOB, "Sit")
        modal = press.response.modals[0]
        text_input(modal)._value = str(DEFAULT)
        foreign = FakeGameClick(self.bot, BOB, guild_id=G2, channel=self.channel3)
        await modal.on_submit(foreign)
        self.assertEqual(self.seats(), [])
        self.assertEqual(self.balance(BOB), WALLET)
        self.assertEqual(self.balance(BOB, G2), WALLET)

    async def test_two_servers_run_separate_tables_and_wallets(self):
        """C1a: a table in each server; seats, wallets and ledgers stay apart, and closing one leaves the other open."""
        await self.start()
        await self.sit(BOB, 500)
        click = await self.start(ALICE, channel=self.channel3, guild=G2)
        self.assertTrue(click.response.sent)
        other = self.store.open_table_for(G2, CH3)
        self.assertIsNotNone(other)
        message = self.channel3.messages[other["message_id"]]
        press = await self.push(CARA, "Sit", message, guild=G2)
        text_input(press.response.modals[0])._value = "800"
        await press.response.modals[0].on_submit(self.click(CARA, channel=self.channel3, message=message, guild=G2))
        self.assertEqual(self.balance(BOB), WALLET - 500)
        self.assertEqual(self.balance(BOB, G2), WALLET)
        self.assertEqual(self.balance(CARA, G2), WALLET - 800)
        self.assertEqual(self.balance(CARA), WALLET)
        self.assertEqual([seat[:2] for seat in self.seats()], [("member", BOB)])
        closed = await self.holdem.close_for_channel(G, CH, self.table()["id"])
        self.assertIsNotNone(closed)
        self.assertIsNotNone(self.store.open_table_for(G2, CH3))
        self.assertEqual(self.balance(CARA, G2), WALLET - 800)

    async def test_games_must_be_turned_on_in_the_servers_own_channel(self):
        """C1a: a channel that is a game channel only in another server does not let Hold'em start."""
        self.store.set_game_channel(G2, CH3, False)
        click = await self.start(channel=self.channel3, guild=G2)
        self.assertEqual(says(click), [OFF_HERE])
        self.assertIsNone(self.store.open_table_for(G2, CH3))


class RecoveryTests(HoldemCase):
    async def test_a_restart_closes_a_lobby_table_and_cashes_out(self):
        """C1a: recover() finds the table through open_cash_tables(), closes it, pays every stack back, edits the lobby, posts one line, archives the thread."""
        await self.start(favorites=True)
        await self.sit(BOB, 1500)
        lobby, thread, table_id = self.lobby(), self.thread(), self.table()["id"]
        self.assertEqual([(t["guild_id"], t["table_id"]) for t in self.store.open_cash_tables()], [(G, table_id)])
        await self.holdem.recover()
        self.assertIsNone(self.table())
        self.assertEqual(self.store.open_cash_tables(), [])
        self.assertEqual([self.balance(u) for u in (ALICE, BOB)], [WALLET, WALLET])
        self.assertEqual(self.store.character_balance(G, self.ann), CHAR_WALLET)
        self.assertIn("restart", lobby.content.lower())
        self.assertIsNone(lobby.view)
        lines = [m.content for m in thread.sent if m.content]
        self.assertEqual(len(lines), 1)
        self.assertIn("restart", lines[0].lower())
        self.assertTrue(thread.archived or ((G, table_id), "archive") in self.holdem.runtime.timers)

    async def test_a_restart_voids_an_unfinished_hand_and_returns_the_starting_stacks(self):
        """C1a: an unfinished hand (blinds posted) is voided: everyone is cashed out at the stack they sat down with."""
        await self.start()
        await self.sit(ALICE, 1000)
        await self.sit(BOB, 1200)
        self.store.deal(G, self.snap()["round_id"])
        self.assertEqual(self.store.open_cash_tables()[0]["status_of_newest_round"], "playing")
        await self.holdem.recover()
        self.assertIsNone(self.table())
        self.assertEqual([self.balance(u) for u in (ALICE, BOB)], [WALLET, WALLET])

    async def test_a_restart_handles_every_server_and_only_hold_em(self):
        """C1a: tables of all servers are closed; an I Doubt It table in the same store is left for its own recovery."""
        await self.start()
        await self.sit(BOB, 600)
        await self.start(ALICE, channel=self.channel3, guild=G2)
        other = self.store.open_table_for(G2, CH3)
        message = self.channel3.messages[other["message_id"]]
        press = await self.push(CARA, "Sit", message, guild=G2)
        text_input(press.response.modals[0])._value = "800"
        await press.response.modals[0].on_submit(self.click(CARA, channel=self.channel3, message=message, guild=G2))
        self.store.open_table(G, CH2, "doubt", DAN)
        await self.holdem.recover()
        self.assertEqual(self.store.open_cash_tables(), [])
        self.assertEqual((self.balance(BOB), self.balance(CARA, G2)), (WALLET, WALLET))
        self.assertIsNotNone(self.store.open_table_for(G, CH2))
        self.assertIn("restart", message.content.lower())

    async def test_recovery_still_closes_and_pays_when_discord_cannot_be_reached(self):
        """C1a: the money side does not depend on the messages: failed lobby edits do not stop the close."""
        await self.start()
        await self.sit(BOB, 600)
        self.channel.edit_error = not_found("Unknown Message", 10008)
        await self.holdem.recover()
        self.assertIsNone(self.table())
        self.assertEqual(self.balance(BOB), WALLET)


if __name__ == "__main__":
    unittest.main()
