"""FEAT-33 part C1b: Texas Hold'em on Discord, the hand (Deal, turn message, Fold / Check / Call / Raise, My cards and Assist, action timer,
streets, showdown, next hand, busts and table end, restart mid-hand). Tables, seats and the lobby are C1a (``test_holdem_discord_lobby.py``).
Character decisions by a model and table talk are C2 / part D and not covered; a character seat only has to finish a hand through its
fallback policy.

TDD-red tests written before the implementation (plain failing tests, like ``test_holdem_discord_lobby.py``). Offline: the fakes from ``helpers``
and the lobby test's ``HoldemCase``; hole cards are read through the store (``seat_view``) and results come from the engine used as an
oracle (``holdem.new`` / ``apply`` / ``result``). Timers are driven by calling their hooks, never by waiting.

Assumed API (pinned here; spec: "Hold'em Discord layer (FEAT-33 part C)" in docs/engineering/minigames-plan.md), on top of C1a's:

  Deal (lobby button, host) deals in the store, posts a pinned board message (``table["board_message_id"]``) and the single turn message
  (``table["turn_message_id"]``) in the thread. Cards are written rank+suit (``7♠``, ``10♥``), as in I Doubt It.
  Turn message: street, board cards, pot, ``<@user>`` of the seat on turn (a character by name) with a ``<t:..>`` deadline for members,
  stacks, the last action, a short seed hash (first 12 characters of ``seed_hash``). Buttons: ``Fold``, ``Check`` or ``Call <amount>``, ``Raise``,
  ``My cards``. Every button is a ``HoldemButton`` whose ``moves`` is the number of moves made when the message was painted.
  Raise: ``Raise`` answers privately with a view of option buttons, each label containing its total bet (``raise:<n>``) for every raise in the
  engine's menu (min, half pot, pot, all-in; all-in is labelled "All-in"), plus an ``Exact amount`` button that opens a modal with one TextInput
  (the total bet). A bad amount gets a private reply and no move. The pinned board is edited after ``await holdem.flush()``.
  Timers (hooks stale-safe): ``on_turn_timer(guild, table, round, moves)`` for a member's action timeout (check if free, else fold; actor
  ``timeout``), ``on_character_timer(guild, table, round, moves)`` plays a character seat on turn (actor ``character``),
  ``on_next_timer(guild, table, lobby_round)`` ends the pause between hands. ``runtime.timers`` kinds: ``turn``, ``character``, ``next``,
  ``lobby``, ``archive``. Class constants ``NEXT_HAND_SECONDS`` and ``CHARACTER_SECONDS``.
  After a hand: a result post (board, shown hands with names, pots and winners, stack changes, the revealed seed) and the turn message becomes
  "Next hand in ..." with ``Sit``, ``Leave``, ``Rebuy``, ``Deal now`` (aimed at the next lobby round). When fewer than two seats remain after
  busts the table is closed at once and the end-of-table summary (``table_cash_summary``) is posted in the thread.
  When everyone else folded, the result message carries a ``Show`` button the winner can press once.
"""
import asyncio
import functools
import re
import unittest
from unittest import mock

import test_holdem_discord_lobby as lobby_tests
from helpers import invoke
from test_holdem_discord_lobby import ALICE, BOB, CARA, DAN, G, G2, WALLET, item_of, labels, private, says, text_input

from llmcord_core.games import holdem, seed_hash
from llmcord_core.games.base import Seat
from llmcord_core.games_discord import NOT_SEATED, NOT_YOUR_TURN, STALE

try:  # imported softly so every test fails on its own until the module exists
    from llmcord_core.games_holdem_discord import HoldemButton
except ImportError:  # pragma: no cover - the red state
    HoldemButton = None

CUSTOM_ID = re.compile(r"llmcord:holdem:(?P<action>[a-z_]+):(?P<table>\d+):(?P<round>\d+):(?P<moves>\d+)")


def card(c):
    return f"{c[0]}{c[1]}"


def numbers(text):
    return [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", text or "")]


def find(message, prefix):
    for child in message.view.children if message.view else ():
        if (item_of(child).label or "").startswith(prefix):
            return child
    raise AssertionError(f"no button starting {prefix!r} among {labels(message)}")


def shove(menu, snap=None, seat=None):
    raises = [m for m in menu if m.startswith("raise:")]
    return raises[-1] if raises else "call" if "call" in menu else "check"


def call_down(menu, snap=None, seat=None):
    return "check" if "check" in menu else "call"


def run_hand(seed, stacks, chooser, button_seat=None):
    """The engine oracle: the same deal the store makes for these stacks (members in sit order, default rules), played with ``chooser(menu)``."""
    seats = [Seat(i, "member", 100 + i, s) for i, s in enumerate(stacks)]
    state = holdem.new(seed, seats, holdem.normalize_rules(None), int(seed_hash(seed)[:8], 16) % len(seats) if button_seat is None else button_seat)
    while not state.finished:
        menu = holdem.legal_moves_for_view(holdem.view(state, state.turn))
        state = holdem.apply(state, state.turn, chooser(menu))
    return state, holdem.result(state)


@functools.lru_cache(maxsize=None)
def find_seed(kind):
    """A seed where everybody shoving and calling gives the outcome ``kind`` wants (no ties, so the expectations are exact)."""
    stacks = {"heads_up": (1000, 1000), "bust": (400, 700, 1000)}[kind]
    for i in range(4000):
        seed = f"c1b-{kind}-{i}"
        _, res = run_hand(seed, stacks, shove)
        ends = [s.end for s in res.seats]
        if kind == "heads_up" and {s.outcome for s in res.seats} == {"win", "lose"}:
            return seed
        if kind == "bust" and ends[0] == 0 and all(s.outcome in ("win", "lose") for s in res.seats) and len(res.pots) == 2 and len(res.shown) == 3:
            return seed
    raise AssertionError("no suitable seed")


class HandCase(lobby_tests.HoldemCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.seed, self.seeds = "c1b-default", []
        for target in ("llmcord_core.admin_store.new_seed", "llmcord_core.holdem_store.new_seed"):
            patcher = mock.patch(target, side_effect=lambda: self.seeds.pop(0) if self.seeds else self.seed)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ("NEXT_HAND_SECONDS", "CHARACTER_SECONDS"):
            setattr(self.holdem, name, 3600)

    # --- driving ------------------------------------------------------------------------------

    async def deal_table(self, sits=((ALICE, 1000), (BOB, 1000)), favorites=False):
        """Opens a table in CH (host ALICE), seats ``sits`` (the host too unless ``favorites`` seated her), and presses Deal."""
        await self.start(favorites=favorites)
        for user, amount in sits:
            await self.sit(user, amount)
        click = await self.push(ALICE, "Deal")
        self.tid, self.th = self.table()["id"], self.thread()
        self.rid = self.snap()["round_id"]
        return click

    def now(self):
        return self.store.round_snapshot(G, self.rid)

    def user_of(self, snap, seat):
        row = snap["seats"][seat]
        return row["ref_id"] if row["kind"] == "member" else None

    def turn_msg(self):
        return self.th.messages[self.store.one("SELECT turn_message_id FROM game_tables WHERE id=?", (self.tid,))[0]]

    def board_msg(self):
        return self.th.messages[self.store.one("SELECT board_message_id FROM game_tables WHERE id=?", (self.tid,))[0]]

    def text(self):
        return "\n".join(m.content for m in self.th.messages.values() if not m.deleted and m.content)

    def holes(self):
        return [self.store.seat_view(G, self.rid, i).hole for i in range(len(self.now()["seats"]))]

    def moves_log(self, rid=None):
        rows = self.store.db.execute("SELECT seat_index,move,actor FROM game_moves WHERE round_id=? ORDER BY id", (rid or self.rid,))
        return [(r["seat_index"], r["move"], r["actor"]) for r in rows]

    def menu(self, snap=None):
        snap = snap or self.now()
        return holdem.legal_moves_for_view(self.store.seat_view(G, self.rid, snap["view"]["turn"]))

    def actor(self):
        snap = self.now()
        return snap["view"]["turn"], self.user_of(snap, snap["view"]["turn"])

    def stranger(self, snap=None):
        """A seated member who is not on turn."""
        snap = snap or self.now()
        return next(s["ref_id"] for s in snap["seats"] if s["kind"] == "member" and s["index"] != snap["view"]["turn"])

    async def press_turn(self, user, prefix, *, message=None, guild=G, channel=None):
        message = message or self.turn_msg()
        click = self.click(user, channel=channel or message.host, message=message, guild=guild)
        await find(message, prefix).callback(click)
        return click

    async def open_picker(self, user, message=None):
        click = await self.press_turn(user, "Raise", message=message)
        entries = click.response.sent + click.followup.sent
        view = entries[-1][1].get("view") if entries else None
        return click, view

    @staticmethod
    def option(view, amount):
        hits = [c for c in view.children if "exact" not in (item_of(c).label or "").lower() and amount in numbers(item_of(c).label)]
        assert len(hits) == 1, f"expected one option for {amount}, found {[item_of(c).label for c in view.children]}"
        return hits[0]

    async def raise_to(self, user, amount):
        click, view = await self.open_picker(user)
        pick = self.click(user, channel=self.th)
        await self.option(view, amount).callback(pick)
        return click, pick

    async def do(self, user, move):
        if move.startswith("raise:"):
            return await self.raise_to(user, int(move[6:]))
        return await self.press_turn(user, {"fold": "Fold", "check": "Check", "call": "Call"}[move])

    async def play_hand(self, chooser, limit=300):
        """Plays the current hand to its end through the buttons and timer hooks. ``chooser(menu, snap, seat)``: a move, or ``timeout``."""
        for _ in range(limit):
            snap = self.now()
            if snap["status"] != "playing":
                return snap
            seat = snap["view"]["turn"]
            row = snap["seats"][seat]
            move = chooser(self.menu(snap), snap, seat)
            if row["kind"] == "character":
                await self.holdem.on_character_timer(G, self.tid, self.rid, snap["moves"])
            elif move == "timeout":
                await self.holdem.on_turn_timer(G, self.tid, self.rid, snap["moves"])
            else:
                await self.do(row["ref_id"], move)
            self.assertGreater(self.now()["moves"], snap["moves"], f"the {move!r} of seat {seat} did not advance the hand")
        self.fail("the hand did not finish")

    async def next_hand(self):
        lobby = self.store.latest_round(G, self.tid)
        self.assertEqual(lobby["status"], "joining")
        await self.holdem.on_next_timer(G, self.tid, lobby["round_id"])
        self.rid = self.store.latest_round(G, self.tid)["round_id"]
        return self.now()

    def message_with(self, label):
        for message in self.th.messages.values():
            if not message.deleted and label in labels(message):
                return message
        raise AssertionError(f"no thread message has a {label!r} button")


# --- Deal ---------------------------------------------------------------------------------------

class DealTests(HandCase):
    async def test_deal_posts_the_pinned_board_and_one_turn_message_with_the_buttons(self):
        """C1b: Deal starts the hand in the store, pins a board message in the thread and posts one turn message with Fold, Check/Call, Raise, My cards."""
        await self.deal_table()
        snap = self.now()
        self.assertEqual(snap["status"], "playing")
        self.assertEqual(snap["view"]["bets"], (5, 10) if snap["rules"]["button"] == 0 else (10, 5))  # heads-up the button posts the small blind
        board, turn = self.board_msg(), self.turn_msg()
        self.assertTrue(board.pinned)
        self.assertIn("5/10", board.content)
        names = labels(turn)
        self.assertTrue({"Fold", "Raise", "My cards"} <= set(names), names)
        self.assertTrue(any(n.startswith(("Check", "Call")) for n in names), names)
        self.assertEqual(sum(1 for m in self.th.messages.values() if "My cards" in labels(m)), 1)

    async def test_the_turn_message_shows_street_pot_actor_stacks_and_deadline(self):
        """C1b: street, pot (blinds 15), the seat on turn (mention and deadline), and both stacks after the blinds."""
        await self.deal_table()
        snap = self.now()
        seat, user = self.actor()
        text = self.turn_msg().content
        self.assertIn("preflop", text.lower().replace("-", ""))
        self.assertIn("15", text)
        self.assertIn(f"<@{user}>", text)
        self.assertIn("<t:", text)  # the action deadline
        for stack in snap["view"]["stacks"]:
            self.assertIn(f"{stack:,}", text)

    async def test_nothing_public_shows_a_hole_card_before_showdown(self):
        """C1b: neither the turn message nor the board message nor any other thread post names a hole card during the hand."""
        await self.deal_table()
        text = self.text() + self.board_msg().content
        for hole in self.holes():
            for c in hole:
                self.assertNotIn(card(c), text)

    async def test_the_deal_shows_the_short_seed_hash(self):
        """C1b (design assumption): like I Doubt It the hand shows the first 12 characters of the seed hash; the seed itself stays secret."""
        await self.deal_table()
        snap = self.now()
        self.assertIn(snap["seed_hash"][:12], self.text() + self.board_msg().content)
        self.assertNotIn(self.seed, self.text() + self.board_msg().content)

    async def test_turn_buttons_carry_table_round_and_move_count_and_no_seat(self):
        """C1b: every turn button is a HoldemButton whose custom id has this table, round and the number of moves made so far; the count follows each move."""
        await self.deal_table()
        for child in self.turn_msg().view.children:
            match = CUSTOM_ID.fullmatch(item_of(child).custom_id)
            self.assertIsNotNone(match)
            self.assertEqual((int(match["table"]), int(match["round"]), int(match["moves"])), (self.tid, self.rid, 0))
        seat, user = self.actor()
        await self.do(user, "call")
        for child in self.turn_msg().view.children:
            self.assertEqual(int(CUSTOM_ID.fullmatch(item_of(child).custom_id)["moves"]), 1)

    async def test_deal_arms_the_member_turn_timer_and_drops_the_lobby_timer(self):
        """C1b: a member on turn gets an action timer; the lobby idle timer is gone while a hand is played."""
        await self.deal_table()
        timers = self.holdem.runtime.timers
        self.assertIn(((G, self.tid), "turn"), timers)
        self.assertNotIn(((G, self.tid), "lobby"), timers)

    async def test_a_character_on_turn_gets_the_short_pause_timer_not_an_action_timer(self):
        """C1b: a character seat on turn is played after a short pause (character timer); there is no member deadline for it."""
        await self.start(favorites=True)
        await self.sit(BOB)
        await self.push(ALICE, "Deal")
        self.tid, self.th, self.rid = self.table()["id"], self.thread(), self.snap()["round_id"]
        for _ in range(10):
            snap = self.now()
            if snap["seats"][snap["view"]["turn"]]["kind"] == "character":
                break
            await self.do(self.actor()[1], call_down(self.menu(snap)))
        else:
            self.fail("a character never came on turn")
        timers = self.holdem.runtime.timers
        self.assertIn(((G, self.tid), "character"), timers)
        self.assertNotIn(((G, self.tid), "turn"), timers)

    async def test_a_double_press_on_deal_deals_one_hand(self):
        """C1b: two Deal presses racing under the table lock make one hand and one turn message; the other press is answered privately."""
        await self.start()
        await self.sit(ALICE)
        await self.sit(BOB)
        first, second = await asyncio.gather(self.push(ALICE, "Deal"), self.push(ALICE, "Deal"))
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM game_rounds").fetchone()[0], 1)
        self.assertEqual(self.snap()["status"], "playing")
        self.th = self.thread()
        self.assertEqual(sum(1 for m in self.th.messages.values() if "My cards" in labels(m)), 1)
        self.assertEqual(sum(1 for c in (first, second) if says(c)), 1)

    async def test_no_discord_call_is_made_while_a_table_lock_is_held(self):
        """C1b: the table lock covers store work only; sends, edits and pins happen after it is released (deal and one move)."""
        held = []
        self.channel.on_call = lambda kind: held.append((kind, any(lock.locked() for lock in self.holdem.runtime.locks.values())))
        await self.deal_table()
        await self.do(self.actor()[1], "call")
        self.assertTrue(held)
        self.assertEqual([kind for kind, locked in held if locked], [])

    async def test_the_hand_moves_no_money_in_the_ledger(self):
        """C1b: stacks change inside the table; wallets and ledger move only at buy-in, rebuy and cash-out."""
        await self.deal_table()
        before, wallets = self.reasons(), [self.balance(ALICE), self.balance(BOB)]
        await self.play_hand(call_down)
        self.assertEqual(self.now()["status"], "settled")
        self.assertEqual(self.reasons(), before)
        self.assertEqual([self.balance(ALICE), self.balance(BOB)], wallets)


# --- the buttons --------------------------------------------------------------------------------

class ActionTests(HandCase):
    async def test_call_then_check_deals_the_flop_and_repaints_the_turn_message(self):
        """C1b: the small blind completes, the big blind checks: the turn message now shows the flop's three cards and a fresh move count."""
        await self.deal_table()
        for move in ("call", "check"):
            self.assertIn(move, self.menu())
            await self.do(self.actor()[1], move)
        snap = self.now()
        self.assertEqual((snap["view"]["street"], len(snap["view"]["board"])), ("flop", 3))
        text = self.turn_msg().content
        self.assertIn("flop", text.lower())
        for c in snap["view"]["board"]:
            self.assertIn(card(c), text)
        self.assertIn("20", text)  # the pot
        self.assertEqual(int(CUSTOM_ID.fullmatch(item_of(find(self.turn_msg(), "Fold")).custom_id)["moves"]), 2)

    async def test_the_pinned_board_follows_the_streets_after_the_debounce(self):
        """C1b (design assumption): the pinned board is repainted (debounced, ``flush()``) with the community cards and the pot."""
        await self.deal_table()
        for move in ("call", "check"):
            await self.do(self.actor()[1], move)
        await self.holdem.flush()
        text = self.board_msg().content
        for c in self.now()["view"]["board"]:
            self.assertIn(card(c), text)
        self.assertIn("20", text)

    async def test_the_button_label_is_check_when_free_and_call_with_the_amount_otherwise(self):
        """C1b: facing the big blind the small blind sees ``Call 5``; after the call the big blind sees ``Check``."""
        await self.deal_table()
        label = item_of(find(self.turn_msg(), "Call")).label
        self.assertIn(5, numbers(label))
        await self.do(self.actor()[1], "call")
        self.assertTrue(any(n.startswith("Check") for n in labels(self.turn_msg())))

    async def test_the_last_action_is_named_in_the_turn_message(self):
        """C1b: after a call the message names who called."""
        await self.deal_table()
        user = self.actor()[1]
        await self.do(user, "call")
        text = self.turn_msg().content
        self.assertIn(f"<@{user}>", text)
        self.assertIn("call", text.lower())

    async def test_fold_ends_a_heads_up_hand_and_the_other_seat_takes_the_blinds(self):
        """C1b: the small blind folds: the big blind wins 15 (net +5), stacks 1005 / 995, hand settled with outcomes win / fold."""
        await self.deal_table()
        seat, user = self.actor()
        await self.do(user, "fold")
        snap = self.now()
        self.assertEqual(snap["status"], "settled")
        self.assertEqual(self.moves_log(), [(seat, "fold", "member")])
        self.assertEqual(sorted((s["outcome"], s["end"]) for s in snap["seats"]), [("fold", 995), ("win", 1005)])

    async def test_only_the_seat_on_turn_can_act(self):
        """C1b: another seated member gets NOT_YOUR_TURN, someone without a seat NOT_SEATED; no move is made."""
        await self.deal_table(sits=((ALICE, 1000), (BOB, 1000)))
        other = self.stranger()
        for prefix in ("Fold", "Raise"):
            click = await self.press_turn(other, prefix)
            self.assertEqual(says(click), [NOT_YOUR_TURN])
            self.assertTrue(private(click))
        click = await self.press_turn(other, "Call")
        self.assertEqual(says(click), [NOT_YOUR_TURN])
        click = await self.press_turn(DAN, "Fold")
        self.assertEqual(says(click), [NOT_SEATED])
        self.assertEqual(self.moves_log(), [])

    async def test_a_button_of_an_earlier_state_is_stale_and_changes_nothing(self):
        """C1b: pressing a button the message showed before the last move answers STALE privately; the hand is untouched."""
        await self.deal_table()
        old = find(self.turn_msg(), "Fold")
        seat, user = self.actor()
        await self.do(user, "call")
        before = self.now()["moves"]
        click = self.click(self.actor()[1], channel=self.th, message=self.turn_msg())
        await old.callback(click)
        self.assertEqual(says(click), [STALE])
        self.assertTrue(private(click))
        self.assertEqual(self.now()["moves"], before)

    async def test_pressing_the_same_button_twice_applies_once(self):
        """C1b: a double press (same move count) applies the move once; the second answer is STALE."""
        await self.deal_table()
        seat, user = self.actor()
        message = self.turn_msg()
        call = find(message, "Call")
        first, second = self.click(user, channel=self.th, message=message), self.click(user, channel=self.th, message=message)
        await call.callback(first)
        await call.callback(second)
        self.assertEqual(self.moves_log(), [(seat, "call", "member")])
        self.assertEqual(says(second), [STALE])

    async def test_two_presses_racing_for_the_same_turn_apply_one(self):
        """C1b: Fold and Call pressed at the same moment: the lock lets one through, the other gets STALE, exactly one move is stored."""
        await self.deal_table()
        seat, user = self.actor()
        message = self.turn_msg()
        a, b = self.click(user, channel=self.th, message=message), self.click(user, channel=self.th, message=message)
        await asyncio.gather(find(message, "Fold").callback(a), find(message, "Call").callback(b))
        self.assertEqual(len(self.moves_log()), 1)
        self.assertEqual(sorted(says(a) + says(b)), [STALE])

    async def test_a_member_cannot_press_for_a_character_seat(self):
        """C1b: while a character is on turn the human buttons do nothing (NOT_YOUR_TURN) and no move is made."""
        await self.start(favorites=True)
        await self.sit(BOB)
        await self.push(ALICE, "Deal")
        self.tid, self.th, self.rid = self.table()["id"], self.thread(), self.snap()["round_id"]
        for _ in range(10):
            snap = self.now()
            if snap["seats"][snap["view"]["turn"]]["kind"] == "character":
                break
            await self.do(self.actor()[1], call_down(self.menu(snap)))
        before = self.now()["moves"]
        click = await self.press_turn(ALICE, "Fold")
        self.assertEqual(says(click), [NOT_YOUR_TURN])
        self.assertEqual(self.now()["moves"], before)


class RaiseTests(HandCase):
    async def test_raise_opens_a_private_picker_with_the_menu_amounts_and_exact_amount(self):
        """C1b: Raise answers only the presser, with an option per raise in the engine's menu (min, half pot, pot, all-in) and an Exact amount button."""
        await self.deal_table()
        seat, user = self.actor()
        raises = [int(m[6:]) for m in self.menu() if m.startswith("raise:")]
        self.assertGreaterEqual(len(raises), 2)
        click, view = await self.open_picker(user)
        self.assertTrue(private(click))
        self.assertIsNotNone(view)
        names = [item_of(c).label for c in view.children]
        self.assertTrue(any("exact" in n.lower() for n in names), names)
        for amount in raises:
            self.option(view, amount)
        self.assertEqual(len(view.children), len(raises) + 1)
        allin = [c for c in view.children if "all" in (item_of(c).label or "").lower()]
        self.assertEqual([numbers(item_of(c).label) for c in allin], [[raises[-1]]])
        self.assertEqual(self.moves_log(), [])

    async def test_the_minimum_raise_option_makes_the_move_and_passes_the_turn(self):
        """C1b: picking the min raise stores ``raise:20`` (a total bet), the turn passes and the turn message names the raise."""
        await self.deal_table()
        seat, user = self.actor()
        low = min(int(m[6:]) for m in self.menu() if m.startswith("raise:"))
        self.assertEqual(low, 20)
        await self.raise_to(user, low)
        self.assertEqual(self.moves_log(), [(seat, "raise:20", "member")])
        snap = self.now()
        self.assertNotEqual(snap["view"]["turn"], seat)
        self.assertEqual(snap["view"]["bets"][seat], 20)
        self.assertIn("20", self.turn_msg().content)

    async def test_the_all_in_option_puts_the_whole_stack_in(self):
        """C1b: the all-in option stores the largest raise and the seat's stack is zero."""
        await self.deal_table()
        seat, user = self.actor()
        top = [int(m[6:]) for m in self.menu() if m.startswith("raise:")][-1]
        await self.raise_to(user, top)
        snap = self.now()
        self.assertEqual(snap["view"]["stacks"][seat], 0)
        self.assertTrue(snap["view"]["all_in"][seat])
        self.assertEqual(self.moves_log()[0][1], f"raise:{top}")

    async def test_only_the_seat_on_turn_gets_a_picker(self):
        """C1b: another member pressing Raise gets NOT_YOUR_TURN and no picker."""
        await self.deal_table()
        click, view = await self.open_picker(self.stranger())
        self.assertEqual(says(click), [NOT_YOUR_TURN])
        self.assertTrue(view is None or not getattr(view, "children", None))

    async def test_a_picker_opened_before_the_table_moved_on_is_stale(self):
        """C1b: an option picked after the turn passed on (the hand moved) answers STALE and stores no raise."""
        await self.deal_table()
        seat, user = self.actor()
        _, view = await self.open_picker(user)
        snap = self.now()
        self.store.play(G, self.rid, seat, "call", "member", snap["moves"])
        pick = self.click(user, channel=self.th)
        await self.option(view, 20).callback(pick)
        self.assertIn(STALE, says(pick))
        self.assertEqual(self.moves_log(), [(seat, "call", "member")])

    async def test_exact_amount_accepts_a_legal_total_bet(self):
        """C1b: the Exact amount modal takes the total bet; 35 is stored as ``raise:35``."""
        await self.deal_table()
        seat, user = self.actor()
        _, view = await self.open_picker(user)
        press = self.click(user, channel=self.th)
        await find_in(view, "Exact").callback(press)
        modal = press.response.modals[-1]
        text_input(modal)._value = "35"
        await modal.on_submit(self.click(user, channel=self.th))
        self.assertEqual(self.moves_log(), [(seat, "raise:35", "member")])
        self.assertEqual(self.now()["view"]["bets"][seat], 35)

    async def test_exact_amount_refuses_illegal_amounts_privately_and_keeps_the_turn(self):
        """C1b: below the minimum, above the stack, zero, and words each get a private reply and no move."""
        await self.deal_table()
        seat, user = self.actor()
        _, view = await self.open_picker(user)
        for text in ("11", "5000", "0", "abc", "-20", ""):
            press = self.click(user, channel=self.th)
            await find_in(view, "Exact").callback(press)
            modal = press.response.modals[-1]
            text_input(modal)._value = text
            submit = self.click(user, channel=self.th)
            await modal.on_submit(submit)
            self.assertTrue(says(submit) and private(submit), text)
        self.assertEqual(self.moves_log(), [])
        self.assertEqual(self.actor()[0], seat)

    async def test_an_exact_amount_modal_cannot_be_submitted_from_another_server(self):
        """C1b: the modal is bound to the server it was opened for; another server's submit is STALE and moves nothing."""
        await self.deal_table()
        seat, user = self.actor()
        _, view = await self.open_picker(user)
        press = self.click(user, channel=self.th)
        await find_in(view, "Exact").callback(press)
        modal = press.response.modals[-1]
        text_input(modal)._value = "35"
        foreign = self.click(user, channel=self.th, guild=G2)
        await modal.on_submit(foreign)
        self.assertEqual(says(foreign), [STALE])
        self.assertEqual(self.moves_log(), [])


def find_in(view, prefix):
    for child in view.children:
        if (item_of(child).label or "").startswith(prefix):
            return child
    raise AssertionError(f"no {prefix!r} among {[item_of(c).label for c in view.children]}")


# --- My cards, Assist, /game hand ---------------------------------------------------------------

class CardsTests(HandCase):
    async def test_my_cards_shows_only_the_pressers_hole_cards_privately(self):
        """C1b: My cards answers privately with exactly that seat's two cards; the other seat's cards are not in it."""
        await self.deal_table()
        holes = self.holes()
        for seat, user in ((0, ALICE), (1, BOB)):
            click = await self.press_turn(user, "My cards")
            text = " ".join(says(click))
            self.assertTrue(private(click))
            for c in holes[seat]:
                self.assertIn(card(c), text)
            for c in holes[1 - seat]:
                self.assertNotIn(card(c), text)

    async def test_my_cards_needs_a_seat(self):
        """C1b: someone who is not at the table gets NOT_SEATED."""
        await self.deal_table()
        click = await self.press_turn(DAN, "My cards")
        self.assertEqual(says(click), [NOT_SEATED])

    async def test_my_cards_works_when_it_is_not_your_turn(self):
        """C1b: looking at your cards never needs the turn."""
        await self.deal_table()
        other = self.stranger()
        click = await self.press_turn(other, "My cards")
        seat = self.now()["seats"]
        hole = self.holes()[next(s["index"] for s in seat if s["ref_id"] == other)]
        self.assertIn(card(hole[0]), " ".join(says(click)))

    async def test_without_assist_my_cards_shows_no_odds(self):
        """C1b: Assist is off by default; the reply is the cards alone (no percentages)."""
        await self.deal_table()
        text = " ".join(says(await self.press_turn(ALICE, "My cards")))
        self.assertNotIn("%", text)

    async def test_with_assist_on_my_cards_adds_the_current_hand_and_the_odds(self):
        """C1b: when the server allows Assist and the host turned it on, My cards adds the made hand and chances; the table shows ``Assist: on``."""
        self.store.set_game_rules(G, "holdem", {"assist": True})
        await self.start()
        await self.push(ALICE, "Assist on")
        await self.sit(ALICE)
        await self.sit(BOB)
        await self.push(ALICE, "Deal")
        self.tid, self.th, self.rid = self.table()["id"], self.thread(), self.snap()["round_id"]
        self.assertIn("Assist: on", self.board_msg().content)
        hole = self.holes()[0]
        text = " ".join(says(await self.press_turn(ALICE, "My cards")))
        self.assertIn(holdem.assist(hole, (), 1, "any-seed").current, text)
        self.assertIn("%", text)

    async def test_assist_numbers_do_not_change_between_presses(self):
        """C1b: the odds are cached per hand, seat and street: the same text twice, even for a different press."""
        self.store.set_game_rules(G, "holdem", {"assist": True})
        await self.start()
        await self.push(ALICE, "Assist on")
        await self.sit(ALICE)
        await self.sit(BOB)
        await self.push(ALICE, "Deal")
        self.tid, self.th, self.rid = self.table()["id"], self.thread(), self.snap()["round_id"]
        first = says(await self.press_turn(ALICE, "My cards"))
        second = says(await self.press_turn(ALICE, "My cards"))
        self.assertEqual(first, second)

    async def test_assist_never_appears_when_the_host_did_not_turn_it_on(self):
        """C1b: the server allows Assist but the table did not switch it on: no odds."""
        self.store.set_game_rules(G, "holdem", {"assist": True})
        await self.deal_table()
        self.assertNotIn("Assist: on", self.board_msg().content)
        self.assertNotIn("%", " ".join(says(await self.press_turn(ALICE, "My cards"))))

    async def test_game_hand_in_the_thread_shows_the_private_cards(self):
        """C1b: /game hand in the table thread answers privately with the member's two cards; a stranger gets NOT_SEATED."""
        await self.deal_table()
        hole = self.holes()[0]
        click = self.click(ALICE, channel=self.th)
        await invoke(self.bot, "game hand", click)
        self.assertTrue(private(click))
        for c in hole:
            self.assertIn(card(c), " ".join(says(click)))
        stranger = self.click(DAN, channel=self.th)
        await invoke(self.bot, "game hand", stranger)
        self.assertEqual(says(stranger), [NOT_SEATED])


# --- timers -------------------------------------------------------------------------------------

class TimerTests(HandCase):
    async def test_a_timeout_when_facing_a_bet_folds(self):
        """C1b: the action timer folds a member who owes chips (actor ``timeout``) and the hand goes on."""
        await self.deal_table()
        seat, _ = self.actor()
        await self.holdem.on_turn_timer(G, self.tid, self.rid, 0)
        self.assertEqual(self.moves_log(), [(seat, "fold", "timeout")])
        self.assertEqual(self.now()["status"], "settled")

    async def test_a_timeout_when_checking_is_free_checks(self):
        """C1b: the big blind with nothing to call is checked, not folded."""
        await self.deal_table()
        await self.do(self.actor()[1], "call")
        seat, _ = self.actor()
        await self.holdem.on_turn_timer(G, self.tid, self.rid, 1)
        self.assertEqual(self.moves_log()[-1], (seat, "check", "timeout"))
        self.assertEqual(self.now()["view"]["street"], "flop")

    async def test_a_turn_timer_for_an_earlier_move_count_does_nothing(self):
        """C1b: a timer armed before the last move fires late: no move is made."""
        await self.deal_table()
        await self.do(self.actor()[1], "call")
        await self.holdem.on_turn_timer(G, self.tid, self.rid, 0)
        self.assertEqual(len(self.moves_log()), 1)
        self.assertEqual(self.now()["status"], "playing")

    async def test_a_turn_timer_for_a_settled_hand_does_nothing(self):
        """C1b: the hand ended before the timer fired: no extra move, no error."""
        await self.deal_table()
        await self.do(self.actor()[1], "fold")
        await self.holdem.on_turn_timer(G, self.tid, self.rid, 1)
        self.assertEqual(len(self.moves_log()), 1)

    async def test_timers_for_a_closed_table_do_nothing(self):
        """C1b: the table was closed (host, dashboard, restart) while a timer was pending: neither the action nor the character timer errors or moves."""
        await self.deal_table()
        self.store.close_table(G, self.tid)
        await self.holdem.on_turn_timer(G, self.tid, self.rid, 0)
        await self.holdem.on_character_timer(G, self.tid, self.rid, 0)
        self.assertEqual(self.moves_log(), [])

    async def test_each_move_rearms_the_timer_for_the_next_seat(self):
        """C1b: a move replaces the action timer (a new task) so the next member gets the full time."""
        await self.deal_table()
        old = self.holdem.runtime.timers[((G, self.tid), "turn")]
        await self.do(self.actor()[1], "call")
        self.assertIsNot(self.holdem.runtime.timers[((G, self.tid), "turn")], old)

    async def test_a_finished_hand_leaves_no_action_timer(self):
        """C1b: after the hand settles there is no turn timer; the next-hand timer is armed instead."""
        await self.deal_table()
        await self.do(self.actor()[1], "fold")
        timers = self.holdem.runtime.timers
        self.assertNotIn(((G, self.tid), "turn"), timers)
        self.assertIn(((G, self.tid), "next"), timers)

    def timeout_x(self, x):
        def choose(menu, snap, seat):
            return "timeout" if snap["seats"][seat]["ref_id"] == x else call_down(menu)
        return choose

    async def test_two_timeouts_in_a_row_cash_the_seat_out_after_the_hand(self):
        """C1b: a member who times out in two hands in a row is removed after the second hand and cashed out; the others play on."""
        await self.deal_table(sits=((ALICE, 1000), (BOB, 1000), (CARA, 1000)))
        await self.play_hand(self.timeout_x(CARA))
        self.assertIn(CARA, [s["ref_id"] for s in self.store.latest_round(G, self.tid)["seats"]])  # once is forgiven
        await self.next_hand()
        last = await self.play_hand(self.timeout_x(CARA))
        end = next(s["end"] for s in last["seats"] if s["ref_id"] == CARA)
        lobby = self.store.latest_round(G, self.tid)
        self.assertEqual(lobby["status"], "joining")
        self.assertNotIn(CARA, [s["ref_id"] for s in lobby["seats"]])
        self.assertEqual(self.balance(CARA), WALLET - 1000 + end)

    async def test_a_move_in_between_resets_the_timeout_count(self):
        """C1b: a timeout, then a hand where the member acts, then another timeout: still seated (the count is of consecutive hands)."""
        await self.deal_table(sits=((ALICE, 1000), (BOB, 1000), (CARA, 1000)))
        await self.play_hand(self.timeout_x(CARA))
        await self.next_hand()
        await self.play_hand(lambda menu, snap, seat: call_down(menu))
        await self.next_hand()
        await self.play_hand(self.timeout_x(CARA))
        self.assertIn(CARA, [s["ref_id"] for s in self.store.latest_round(G, self.tid)["seats"]])


# --- character seats ----------------------------------------------------------------------------

class CharacterSeatTests(HandCase):
    async def setup_with_ann(self):
        await self.start(favorites=True)
        await self.sit(BOB)
        await self.push(ALICE, "Deal")
        self.tid, self.th, self.rid = self.table()["id"], self.thread(), self.snap()["round_id"]

    async def test_a_character_seat_plays_a_legal_move_when_its_timer_fires(self):
        """C1b: on_character_timer plays the character on turn with the engine policy: a legal move, actor ``character``; a stale timer does nothing."""
        await self.setup_with_ann()
        for _ in range(10):
            snap = self.now()
            if snap["seats"][snap["view"]["turn"]]["kind"] == "character":
                break
            await self.do(self.actor()[1], call_down(self.menu(snap)))
        snap = self.now()
        seat, menu = snap["view"]["turn"], self.menu(snap)
        await self.holdem.on_character_timer(G, self.tid, self.rid, snap["moves"] - 1)  # stale
        self.assertEqual(self.now()["moves"], snap["moves"])
        await self.holdem.on_character_timer(G, self.tid, self.rid, snap["moves"])
        made = self.moves_log()[-1]
        self.assertEqual((made[0], made[2]), (seat, "character"))
        self.assertIn(made[1], menu)

    async def test_a_hand_with_a_character_finishes_and_pays_out_consistently(self):
        """C1b: the member plays by the buttons, the character by its timer; the hand settles, the stacks add up, no seat is lost."""
        await self.setup_with_ann()
        snap = await self.play_hand(lambda menu, snap, seat: call_down(menu))
        self.assertEqual(snap["status"], "settled")
        self.assertEqual(sum(s["end"] for s in snap["seats"]), 3000)
        self.assertIn("character", {a for _, _, a in self.moves_log()})


# --- showdown and result ------------------------------------------------------------------------

class ShowdownTests(HandCase):
    async def test_a_showdown_post_reveals_the_shown_hands_the_board_pot_winner_and_stack_changes(self):
        """C1b: an all-in call-down heads-up: the result post has the five board cards, both hole cards with the hand names, the pot, the winner and the +/- stack change, and the seed is revealed."""
        self.seed = find_seed("heads_up")
        await self.deal_table()
        holes = self.holes()
        await self.play_hand(shove)
        snap = self.now()
        _, expected = run_hand(self.seed, (1000, 1000), shove, button_seat=snap["rules"]["button"])
        text = self.text()
        for c in expected.board:
            self.assertIn(card(c), text)
        for hole in holes:
            for c in hole:
                self.assertIn(card(c), text)
        self.assertIn(holdem.hand_name(expected.pots[0].hand), text)
        self.assertIn(f"{expected.pots[0].amount:,}", text)
        winner = expected.pots[0].winners[0]
        self.assertIn(f"<@{self.user_of(snap, winner)}>", text)
        self.assertIn("1,000", text)  # the stack change
        self.assertIn(self.seed, text)  # revealed after the hand

    async def test_stacks_after_the_showdown_match_the_engine(self):
        """C1b: the settled stacks equal the oracle's end stacks; the loser is busted."""
        self.seed = find_seed("heads_up")
        await self.deal_table()
        await self.play_hand(shove)
        snap = self.now()
        _, expected = run_hand(self.seed, (1000, 1000), shove, button_seat=snap["rules"]["button"])
        self.assertEqual([s["end"] for s in snap["seats"]], [s.end for s in expected.seats])
        self.assertEqual([s["busted"] for s in snap["seats"]], [s.end == 0 for s in expected.seats])

    async def test_a_folded_seat_never_has_its_cards_shown(self):
        """C1b: three seats, the first to act folds, the other two check it down: the post shows the two live hands and not the folded cards."""
        await self.deal_table(sits=((ALICE, 1000), (BOB, 1000), (CARA, 1000)))
        holes = self.holes()
        snap = await self.play_hand(lambda menu, snap, seat: "fold" if snap["moves"] == 0 else call_down(menu))
        folded = next(s["index"] for s in snap["seats"] if s["outcome"] == "fold")
        text = self.text()
        for seat, hole in enumerate(holes):
            for c in hole:
                if seat == folded:
                    self.assertNotIn(card(c), text)
                else:
                    self.assertIn(card(c), text)

    async def test_when_everyone_folds_nothing_is_shown_and_the_winner_may_press_show_once(self):
        """C1b: no cards in the result post; the winner's Show button posts the two cards once (the second press and a loser's press are refused privately)."""
        await self.deal_table()
        holes = self.holes()
        seat, user = self.actor()
        await self.do(user, "fold")
        winner = 1 - seat
        wuser = self.user_of(self.now(), winner)
        for hole in holes:
            for c in hole:
                self.assertNotIn(card(c), self.text())
        message = self.message_with("Show")
        loser = self.click(user, channel=self.th, message=message)
        await find(message, "Show").callback(loser)
        self.assertTrue(private(loser))
        for c in holes[winner]:
            self.assertNotIn(card(c), self.text())
        first = self.click(wuser, channel=self.th, message=message)
        await find(message, "Show").callback(first)
        text = self.text()
        for c in holes[winner]:
            self.assertEqual(text.count(card(c)), 1)
        second = self.click(wuser, channel=self.th, message=message)
        await find(message, "Show").callback(second)
        self.assertTrue(private(second))
        for c in holes[winner]:
            self.assertEqual(self.text().count(card(c)), 1)

    async def test_side_pots_are_listed_with_their_winners(self):
        """C1b (design assumption: the wording ``side``): stacks 400 / 700 / 1000 all in: both pots (1,200 main, 600 side) and the winner mentions are in the result."""
        self.seed = find_seed("bust")
        await self.deal_table(sits=((ALICE, 400), (BOB, 700), (CARA, 1000)))
        await self.play_hand(shove)
        snap = self.now()
        _, expected = run_hand(self.seed, (400, 700, 1000), shove, button_seat=snap["rules"]["button"])
        text = self.text()
        self.assertIn("side", text.lower())
        for pot in expected.pots:
            self.assertIn(f"{pot.amount:,}", text)
            for w in pot.winners:
                self.assertIn(f"<@{self.user_of(snap, w)}>", text)

    async def test_wallets_stay_untouched_by_a_showdown_while_the_table_goes_on(self):
        """C1b: after a three-way all-in the winners' wallets are what they were (the chips are in their stacks); only the busted seat was cashed out, for nothing."""
        self.seed = find_seed("bust")
        await self.deal_table(sits=((ALICE, 400), (BOB, 700), (CARA, 1000)))
        await self.play_hand(shove)
        self.assertEqual([self.balance(u) for u in (ALICE, BOB, CARA)], [WALLET - 400, WALLET - 700, WALLET - 1000])
        self.assertIsNotNone(self.table())


# --- after the hand -----------------------------------------------------------------------------

class NextHandTests(HandCase):
    async def finish_one(self, sits=((ALICE, 1000), (BOB, 1000), (CARA, 1000))):
        await self.deal_table(sits=sits)
        return await self.play_hand(lambda menu, snap, seat: "fold" if snap["moves"] < 2 and "fold" in menu else call_down(menu))

    async def test_the_turn_message_becomes_the_next_hand_pause_with_the_between_hand_buttons(self):
        """C1b: after the result, the turn message offers Sit, Leave, Rebuy and Deal now for the next lobby round, whose seats carry the end stacks."""
        snap = await self.finish_one()
        turn = self.turn_msg()
        self.assertIn("next hand", turn.content.lower())
        self.assertEqual(set(labels(turn)), {"Sit", "Leave", "Rebuy", "Deal now"})
        lobby = self.store.latest_round(G, self.tid)
        self.assertEqual((lobby["status"], lobby["number"]), ("joining", snap["number"] + 1))
        self.assertEqual([s["stack"] for s in lobby["seats"]], [s["end"] for s in snap["seats"]])
        self.assertEqual(int(CUSTOM_ID.fullmatch(item_of(find(turn, "Leave")).custom_id)["round"]), lobby["round_id"])
        self.assertIn(((G, self.tid), "next"), self.holdem.runtime.timers)

    async def test_the_pause_ends_with_the_next_hand_and_the_button_moves(self):
        """C1b: when the pause ends the next hand is dealt (same thread, fresh buttons) with the dealer button one seat on."""
        first = await self.finish_one()
        second = await self.next_hand()
        self.assertEqual(second["status"], "playing")
        self.assertEqual(second["number"], first["number"] + 1)
        self.assertEqual(second["rules"]["button"], (first["rules"]["button"] + 1) % 3)
        self.assertIn("Fold", labels(self.turn_msg()))
        self.assertEqual(int(CUSTOM_ID.fullmatch(item_of(find(self.turn_msg(), "Fold")).custom_id)["round"]), second["round_id"])

    async def test_deal_now_deals_at_once_and_the_old_pause_timer_does_nothing(self):
        """C1b: the host's Deal now starts the next hand without waiting; the pause timer that fires afterwards for the same lobby deals nothing more."""
        await self.finish_one()
        lobby = self.store.latest_round(G, self.tid)
        await self.press_turn(ALICE, "Deal now")
        self.assertEqual(self.store.round_snapshot(G, lobby["round_id"])["status"], "playing")
        await self.holdem.on_next_timer(G, self.tid, lobby["round_id"])
        self.assertEqual(self.store.db.execute("SELECT COUNT(*) FROM game_rounds WHERE table_id=?", (self.tid,)).fetchone()[0], lobby["number"])

    async def test_a_member_can_sit_between_hands_and_is_dealt_in(self):
        """C1b: a new member sits through the pause's Sit button (modal) and plays the next hand."""
        await self.finish_one(sits=((ALICE, 1000), (BOB, 1000)))
        self.assertEqual(self.store.latest_round(G, self.tid)["status"], "joining")
        press, submit = await self.ask(CARA, "Sit", "800", message=self.turn_msg())
        self.assertEqual(self.balance(CARA), WALLET - 800)
        second = await self.next_hand()
        self.assertIn(CARA, [s["ref_id"] for s in second["seats"]])

    async def test_a_member_can_rebuy_between_hands(self):
        """C1b: Rebuy on the pause message tops the stack up (wallet down, stake up) and the next hand starts with it."""
        first = await self.finish_one()
        stack = next(s["end"] for s in first["seats"] if s["ref_id"] == BOB)
        press, submit = await self.ask(BOB, "Rebuy", "100", message=self.turn_msg())
        self.assertEqual(self.balance(BOB), WALLET - 1000 - 100)
        second = await self.next_hand()
        self.assertEqual(next(s["stack"] for s in second["seats"] if s["ref_id"] == BOB), stack + 100)

    async def test_a_member_can_leave_between_hands_and_is_cashed_out(self):
        """C1b: Leave on the pause message pays the end stack to the wallet and the next hand has one seat fewer."""
        first = await self.finish_one()
        end = next(s["end"] for s in first["seats"] if s["ref_id"] == CARA)
        await self.press_turn(CARA, "Leave")
        self.assertEqual(self.balance(CARA), WALLET - 1000 + end)
        second = await self.next_hand()
        self.assertNotIn(CARA, [s["ref_id"] for s in second["seats"]])

    async def test_a_busted_member_is_dropped_from_the_next_hand_and_may_sit_again(self):
        """C1b: three seats, one loses everything: the next lobby has the other two with their end stacks and the busted member can sit again."""
        self.seed = find_seed("bust")
        await self.deal_table(sits=((ALICE, 400), (BOB, 700), (CARA, 1000)))
        await self.play_hand(shove)
        lobby = self.store.latest_round(G, self.tid)
        self.assertEqual([s["ref_id"] for s in lobby["seats"]], [BOB, CARA])
        self.assertEqual(self.balance(ALICE), WALLET - 400)  # the bust returned nothing
        self.assertEqual(self.store.table_cash_summary(G, self.tid)["cashouts"][0]["cashed_out"], 0)
        press, submit = await self.ask(ALICE, "Sit", "500", message=self.turn_msg())
        self.assertEqual(self.balance(ALICE), WALLET - 400 - 500)

    async def test_a_heads_up_bust_ends_the_table_with_the_summary(self):
        """C1b: one of two seats busts: the table closes at once, the winner is cashed out, the thread gets the end-of-table summary and archives later."""
        self.seed = find_seed("heads_up")
        await self.deal_table()
        thread, tid = self.th, self.tid
        before = len(self.reasons())
        snap = await self.play_hand(shove)
        self.assertIsNone(self.table())
        self.assertEqual(self.store.open_cash_tables(), [])
        winner = next(s["ref_id"] for s in snap["seats"] if s["end"] > 0)
        loser = next(s["ref_id"] for s in snap["seats"] if s["end"] == 0)
        self.assertEqual((self.balance(winner), self.balance(loser)), (WALLET + 1000, WALLET - 1000))
        summary = self.store.table_cash_summary(G, tid)
        text = self.text()
        for row in summary["cashouts"]:
            self.assertIn(f"<@{row['ref_id']}>", text)
            self.assertIn(f"{row['cashed_out']:,}", text)
        self.assertIn(((G, tid), "archive"), self.holdem.runtime.timers)
        await self.holdem.on_archive_timer(G, tid)
        self.assertTrue(thread.archived)
        self.assertEqual(len(self.reasons()) - before, 1)  # the winner's cash-out; the busted seat is owed nothing, so no row

    async def test_fewer_than_two_seats_after_the_pause_means_no_deal_and_the_table_waits_then_closes_idle(self):
        """C1b (design assumption): members leaving during the pause leave one seat: the pause end deals nothing, the table stays open, and the idle timer closes it and cashes out."""
        await self.finish_one(sits=((ALICE, 1000), (BOB, 1000)))
        await self.press_turn(BOB, "Leave")
        lobby = await self.next_hand_noop()
        self.assertEqual(lobby["status"], "joining")
        self.assertIsNotNone(self.table())
        await self.holdem.on_lobby_timer(G, self.tid, lobby["round_id"])
        self.assertIsNone(self.table())
        self.assertEqual(self.balance(ALICE) + self.balance(BOB), 2 * WALLET)

    async def next_hand_noop(self):
        lobby = self.store.latest_round(G, self.tid)
        await self.holdem.on_next_timer(G, self.tid, lobby["round_id"])
        return self.store.latest_round(G, self.tid)

    async def test_a_pause_timer_for_a_closed_table_does_nothing(self):
        """C1b: a stale pause timer after the table closed neither errors nor deals."""
        await self.finish_one()
        lobby = self.store.latest_round(G, self.tid)
        self.store.close_table(G, self.tid)
        await self.holdem.on_next_timer(G, self.tid, lobby["round_id"])
        self.assertIsNone(self.table())


# --- guild isolation ----------------------------------------------------------------------------

class GuildTests(HandCase):
    async def test_a_turn_button_pressed_from_another_server_is_stale(self):
        """C1b: the same buttons pressed with a click from another server (even by a member with a seat there) answer STALE and change nothing."""
        await self.deal_table()
        await self.start(ALICE, channel=self.channel3, guild=G2)
        seat, user = self.actor()
        for prefix in ("Fold", "Raise", "My cards"):
            foreign = await self.press_turn(user, prefix, guild=G2)
            self.assertEqual(says(foreign), [STALE])
        self.assertEqual(self.moves_log(), [])

    async def test_two_servers_play_separate_hands(self):
        """C1b: a hand dealt in one server does not touch the other server's table or wallets."""
        await self.deal_table()
        await self.start(ALICE, channel=self.channel3, guild=G2)
        other = self.store.open_table_for(G2, lobby_tests.CH3)
        message = self.channel3.messages[other["message_id"]]
        for user in (ALICE, BOB):
            press = await self.push(user, "Sit", message, guild=G2)
            text_input(press.response.modals[0])._value = "1000"
            await press.response.modals[0].on_submit(self.click(user, channel=self.channel3, message=message, guild=G2))
        await self.do(self.actor()[1], "fold")
        self.assertEqual(self.store.latest_round(G2, other["id"])["status"], "joining")
        self.assertEqual(self.balance(ALICE, G2), WALLET - 1000)


# --- restart ------------------------------------------------------------------------------------

class RecoveryTests(HandCase):
    async def test_a_restart_in_the_middle_of_a_hand_voids_it_and_cashes_out_the_starting_stacks(self):
        """C1b: with moves already made, recover() voids the hand (no replay, no result), pays both starting stacks back, posts one line and disarms every timer."""
        await self.deal_table()
        await self.do(self.actor()[1], "call")
        await self.holdem.recover()
        self.assertIsNone(self.table())
        self.assertEqual([self.balance(ALICE), self.balance(BOB)], [WALLET, WALLET])
        self.assertEqual(self.store.round_snapshot(G, self.rid)["status"], "cancelled")
        self.assertEqual([k for k in self.holdem.runtime.timers if k[0] == (G, self.tid) and k[1] != "archive"], [])
        self.assertIn("restart", self.text().lower())

    async def test_turning_games_off_in_the_channel_during_a_hand_voids_it_and_cashes_out(self):
        """C1b: closing the table from outside mid-hand: the hand is voided, stacks are cashed out at their starting values, timers are gone."""
        await self.deal_table()
        await self.do(self.actor()[1], "call")
        await self.holdem.close_for_channel(G, lobby_tests.CH, self.tid)
        self.assertIsNone(self.table())
        self.assertEqual([self.balance(ALICE), self.balance(BOB)], [WALLET, WALLET])
        self.assertEqual([k for k in self.holdem.runtime.timers if k[0] == (G, self.tid) and k[1] != "archive"], [])
        self.assertEqual(self.store.round_snapshot(G, self.rid)["status"], "cancelled")

    async def test_buttons_of_a_voided_hand_are_stale(self):
        """C1b: after a restart the old turn buttons answer STALE (the table is closed)."""
        await self.deal_table()
        message = self.turn_msg()
        fold = find(message, "Fold")
        await self.holdem.recover()
        click = self.click(ALICE, channel=self.th, message=message)
        await fold.callback(click)
        self.assertEqual(says(click), [STALE])


if __name__ == "__main__":
    unittest.main()
