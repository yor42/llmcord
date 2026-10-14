"""FEAT-48 / D40 (step 1): one public table view for the three games ("cast watches games").

Written before the implementation. Assumed seams (proposed, pinned here; the implementer creates them):

  ``llmcord_core.table_view.public_table_view(bot, guild_id, channel_id, parent_channel_id=None)`` -> ``None`` or a dict with exactly the keys
    ``game`` ('blackjack' | 'doubt' | 'holdem'), ``table_id``, ``thread_id`` (``None`` for blackjack), ``status`` (the newest round's status),
    ``seated_character_ids`` (seat order), ``text`` and ``moment`` (``str | None``). Resolution: ``channel_id`` or ``parent_channel_id`` being
    the thread of a Doubt/Hold'em table (``table_talk_context``), else the open table of ``channel_id`` (then of ``parent_channel_id``)
    via ``store.open_table_for``. Guild-scoped: another server's table never resolves.
  ``DoubtTables.public_block`` / ``HoldemTables.public_block`` / ``GameTables.public_block`` (``(guild_id, key)``) back ``text``: the table
    block without the seated viewer's extras. Every seat in third person ("Player N" for members, sanitised character names), bets,
    stacks, turn, recent moves; NEVER a hole card, a face-down card, a hand, a seed, a seed hash or Assist odds. Doubt: face-down counts
    only. Hold'em: the board only. Blackjack: the dealer's up card, a "hole card face down" line, the visible player cards.
  ``GameTables.table_talk_context(guild_id, channel_id)`` (blackjack has no thread, so the key is the table's channel): the Doubt/Hold'em shape
    (``table_id``, ``round_id``, ``status``, ``seated_character_ids``).
  The seated ``table_block`` of Doubt and Hold'em stays byte-identical (characterization tests below pass today).
"""
import re

import test_doubt_table_talk as dt
import test_game_character_seats_discord as bjseats
import test_holdem_table_talk as ht
from helpers import FakeInteraction, invoke

G, G2, CH, CH2 = dt.G, 2, dt.CH, 101
KEYS = {"game", "table_id", "thread_id", "status", "seated_character_ids", "text", "moment"}
MESSY = "A*n_n<@123>"  # sanitised like the table talk names: no mention, no markdown
CLEAN = "Ann123"


def table_view():
    from llmcord_core import table_view as module  # late import so every test reports its own failure
    return module


def lookup(bot, guild, channel, parent=None):
    return table_view().public_table_view(bot, guild, channel, parent)


def rename(store, character_id, name):
    store.db.execute("UPDATE characters SET name=? WHERE id=?", (name, character_id))
    store.db.commit()


class DoubtView(dt.TalkCase):
    def view(self, channel=None, parent=None, guild=G):
        return lookup(self.bot, guild, (channel or self.thread()).id if not isinstance(channel, int) else channel, parent)

    def assert_public(self, text):
        for seat in range(len(self.snap()["seats"])):
            for card in self.cards_of(seat):
                self.assertNotIn(card, text, f"{card} (face down) leaked")
        for marker in ("Your hand", "(you)", "<@", self.seed, self.snap()["seed_hash"][:12]):
            self.assertNotIn(marker, text)

    async def test_shape_and_resolution_from_thread_parent_and_channel(self):
        await self.table3()
        thread, snap = self.thread(), self.snap()
        found = self.view(thread)
        self.assertEqual(set(found), KEYS)
        self.assertEqual((found["game"], found["table_id"], found["thread_id"], found["status"], found["seated_character_ids"]),
                         ("doubt", snap["table_id"], thread.id, "playing", [self.ann]))
        self.assertIsInstance(found["text"], str)
        self.assertEqual(self.view(thread, CH), found)
        self.assertEqual(self.view(CH), found)  # a normal channel with the table open resolves to the same table
        plain = await self.ch.make_thread("plain chat")
        self.assertEqual(self.view(plain, CH), found)  # a plain thread under the channel sees the channel's table

    async def test_no_card_of_any_seat_is_in_the_public_text(self):
        await self.table5()
        text = self.view()["text"]
        self.assert_public(text)
        view = self.snap()["view"]
        for count in set(view["counts"]):
            self.assertRegex(text, rf"\b{count} cards?\b")
        self.assertRegex(text, rf"\b{view['pile']} cards?\b")
        for name in ("Ann", "Bo", "Cy"):
            self.assertIn(name, text)
        self.assertIn("I Doubt It", text)

    async def test_stays_hidden_after_a_face_down_play(self):
        await self.table3()
        await self.pick_play(dt.ALICE, self.any_card(0))
        found = self.view()
        self.assert_public(found["text"])
        self.assertRegex(found["text"], rf"\b{self.snap()['view']['pile']} cards?\b")

    async def test_joining_lobby_resolves_with_status_joining(self):
        await self.start(favorites=True)
        found = self.view()
        self.assertEqual((found["status"], found["seated_character_ids"]), ("joining", [self.ann]))

    async def test_guild_isolation(self):
        await self.table3()
        thread = self.thread()
        self.assertIsNone(self.view(thread, CH, guild=G2))
        self.assertIsNone(self.view(CH, guild=G2))
        self.assertIsNone(lookup(self.bot, G2, thread.id))

    async def test_none_without_an_open_table(self):
        self.assertIsNone(self.view(CH))
        self.assertIsNone(self.view(424242, CH))
        await self.table3()
        plain = await self.channel2.make_thread("elsewhere")
        self.assertIsNone(self.view(plain, CH2))
        self.assertIsNone(self.view(CH2))

    async def test_none_once_the_table_thread_is_archived(self):
        await self.table3()
        thread, tid = self.thread(), self.table()["id"]
        await self.finish_by_forfeits()
        await self.doubt.on_archive_timer(G, tid)
        self.assertIsNone(self.view(thread.id))
        self.assertIsNone(self.view(CH))

    async def test_character_names_are_sanitised(self):
        await self.table3()
        rename(self.store, self.ann, MESSY)
        text = self.view()["text"]
        self.assertIn(CLEAN, text)
        for bad in ("<@", "*", "_", MESSY):
            self.assertNotIn(bad, text)

    async def test_seated_block_is_byte_identical(self):
        """Characterization (FEAT-48): the seated viewer's table_block does not change when the public block is added."""
        await self.table3()
        before = self.doubt.table_block(G, self.thread().id, self.ann)
        self.assertEqual(before, (
            "You are sitting at a table of I Doubt It (a card game of bluffing: players claim the rank to play, add cards face down, and anyone may doubt).\n"
            "Rank to claim now: A. Pile: 0 cards.\n- Player 1: 18 cards (on turn)\n- Ann (you): 17 cards\n- Player 2: 17 cards\n"
            "Keep your hand secret unless you choose to bluff about it.\n"
            "Your hand: A♠, A♣, 2♠, 2♦, 2♣, 5♥, 6♦, 7♦, 8♠, 8♦, 9♦, 10♠, 10♦, J♣, Q♥, K♠, K♣"))
        await self.pick_play(dt.ALICE, self.any_card(0))
        self.assertEqual(self.doubt.table_block(G, self.thread().id, self.ann), (
            "You are sitting at a table of I Doubt It (a card game of bluffing: players claim the rank to play, add cards face down, and anyone may doubt).\n"
            "Rank to claim now: 2. Pile: 1 card.\n- Player 1: 17 cards\n- Ann (you): 17 cards (on turn)\n- Player 2: 17 cards\n"
            "Open claim: Player 1 claimed one A; it can still be doubted.\nRecent play: Player 1 claimed one A.\n"
            "Keep your hand secret unless you choose to bluff about it.\n"
            "Your hand: A♠, A♣, 2♠, 2♦, 2♣, 5♥, 6♦, 7♦, 8♠, 8♦, 9♦, 10♠, 10♦, J♣, Q♥, K♠, K♣"))


class HoldemView(ht.TalkCase):
    def view(self, channel=None, parent=None, guild=G):
        return lookup(self.bot, guild, (channel or self.th).id if not isinstance(channel, int) else channel, parent)

    def assert_public(self, text):
        for hole in self.dealt:
            for token in hole:
                self.assertNotIn(token, text, f"{token} (hole card) leaked")
        for seed in self.seeds_in_play:
            self.assertNotIn(seed, text)
            self.assertNotIn(ht.seed_hash(seed)[:12], text)
        for marker in ("Your hole", "(you)", "<@", "odds", "equity"):
            self.assertNotIn(marker, text)

    async def test_shape_and_resolution_from_thread_parent_and_channel(self):
        await self.table3()
        found = self.view()
        self.assertEqual(set(found), KEYS)
        self.assertEqual((found["game"], found["table_id"], found["thread_id"], found["status"], found["seated_character_ids"]),
                         ("holdem", self.table()["id"], self.th.id, "playing", [self.ann]))
        self.assertEqual(self.view(self.th, CH), found)
        self.assertEqual(self.view(CH), found)
        plain = await self.ch.make_thread("plain chat")
        self.assertEqual(self.view(plain, CH), found)

    async def test_preflop_text_has_stacks_bets_and_no_hole_card(self):
        await self.table5()
        text = self.view()["text"]
        self.assert_public(text)
        self.assertIn("Texas Hold'em", text)
        self.assertIn("preflop", text)
        self.assertRegex(text, r"stack 1,000")
        self.assertIn("Ann", text)
        self.assertIn("Player 1", text)

    async def test_flop_shows_the_board_but_never_a_hole_card(self):
        await self.table3()
        await self.to_street("flop")
        text = self.view()["text"]
        self.assert_public(text)
        self.assertIn("4♦ K♣ A♣", text)
        self.assertIn("Recent play", text)

    async def test_joining_lobby_resolves(self):
        await self.lobby_only()
        found = self.view()
        self.assertEqual((found["status"], found["seated_character_ids"]), ("joining", [self.ann]))

    async def test_guild_isolation(self):
        await self.table3()
        self.assertIsNone(self.view(self.th, CH, guild=G2))
        self.assertIsNone(self.view(CH, guild=G2))
        self.assertIsNone(lookup(self.bot, G2, self.th.id))

    async def test_none_without_an_open_table(self):
        self.assertIsNone(self.view(CH))
        self.assertIsNone(self.view(424242, CH))
        await self.table3()
        plain = await self.channel2.make_thread("elsewhere")
        self.assertIsNone(self.view(plain, CH2))

    async def test_character_names_are_sanitised(self):
        await self.table3()
        rename(self.store, self.ann, MESSY)
        text = self.view()["text"]
        self.assertIn(CLEAN, text)
        for bad in ("<@", "*", "_", MESSY):
            self.assertNotIn(bad, text)

    async def test_seated_block_is_byte_identical(self):
        """Characterization (FEAT-48): the seated viewer's table_block does not change when the public block is added."""
        await self.table3()
        self.assertEqual(self.block(self.ann), (
            "You are sitting at a table of Texas Hold'em (no-limit poker: each seat holds two secret hole cards, five shared cards come on the board, the best five-card hand wins the pot).\n"
            "Hand 1, preflop. Board: (no cards yet). Pot: 15.\n- Player 1: stack 990 (bet 10)\n- Ann (you): stack 1,000 (dealer, on turn)\n- Player 2: stack 995 (bet 5)\n"
            "Your hole cards (keep them secret unless you choose to bluff about them): 9♦ 9♣"))
        await self.to_street("flop")
        self.assertEqual(self.block(self.ann), (
            "You are sitting at a table of Texas Hold'em (no-limit poker: each seat holds two secret hole cards, five shared cards come on the board, the best five-card hand wins the pot).\n"
            "Hand 1, flop. Board: 4♦ K♣ A♣. Pot: 30.\n- Player 1: stack 990\n- Ann (you): stack 990 (dealer)\n- Player 2: stack 990 (on turn)\n"
            "Recent play: Ann called; Player 2 called; Player 1 checked.\n"
            "Your hole cards (keep them secret unless you choose to bluff about them): 9♦ 9♣"))


HOLE = "7♠"  # the dealer's hole card in the rigged shoe below


class BlackjackView(bjseats.SeatCase):
    """Alice (member) and her favorite Ann; dealer up card 5, hole card 7 (shoe order: seat 0, seat 1, dealer up, then the second round)."""

    async def deal(self):
        await self.start("10", "9", "5", "8", "6", "7")

    def view(self, channel=CH, parent=None, guild=G):
        return lookup(self.bot, guild, channel, parent)

    async def test_shape_and_resolution_from_channel_and_parent_thread(self):
        await self.deal()
        found = self.view()
        self.assertEqual(set(found), KEYS)
        self.assertEqual((found["game"], found["table_id"], found["thread_id"], found["status"], found["seated_character_ids"]),
                         ("blackjack", self.latest()["table_id"], None, "playing", [self.ann]))
        self.assertEqual(self.view(9999, CH), found)  # a thread under the table's channel sees the channel's table

    async def test_only_the_dealer_upcard_is_shown_and_the_hole_card_is_face_down(self):
        await self.deal()
        text = self.view()["text"]
        self.assertIn("5♠", text)  # the up card
        self.assertNotIn(HOLE, text)
        self.assertRegex(text, r"(?i)hole card.*face down|face down.*hole card")
        for visible in ("10♠", "8♠", "9♠", "6♠"):  # the players' cards are public
            self.assertIn(visible, text)
        self.assertIn("Ann", text)
        self.assertIn("Blackjack", text)

    async def test_no_mention_markup_seed_or_hash(self):
        await self.deal()
        text = self.view()["text"]
        snap = self.latest()
        for bad in ("<@", "**", "Seed", "seed", snap["seed_hash"][:12], "seed-for-the-test"):
            self.assertNotIn(bad, text)
        self.assertIn("Player 1", text)  # the member in third person

    async def test_joining_round_has_no_cards(self):
        self.rig("10", "9", "5", "8", "6", "7")
        await self.bet(bjseats.ALICE, 10)
        found = self.view()
        self.assertEqual((found["game"], found["status"], found["seated_character_ids"]), ("blackjack", "joining", []))
        self.assertNotIn("<@", found["text"])

    async def test_table_talk_context_has_the_shared_shape(self):
        await self.deal()
        snap = self.latest()
        self.assertEqual(self.games.table_talk_context(G, CH), {
            "table_id": snap["table_id"], "round_id": snap["round_id"], "status": "playing", "seated_character_ids": [self.ann]})
        self.assertIsNone(self.games.table_talk_context(G2, CH))
        self.assertIsNone(self.games.table_talk_context(G, CH2))

    async def test_guild_isolation(self):
        await self.deal()
        self.assertIsNone(self.view(CH, guild=G2))
        self.assertIsNone(self.view(9999, CH, guild=G2))

    async def test_none_without_an_open_table(self):
        self.assertIsNone(self.view())
        self.assertIsNone(self.view(CH2))
        self.assertIsNone(self.view(9999, CH2))

    async def test_none_once_the_table_is_closed(self):
        await self.deal()
        admin = FakeInteraction(admin=True)
        await invoke(self.bot, "admin games channel", admin, enabled=False)
        self.assertIsNone(self.view())

    async def test_character_names_are_sanitised(self):
        await self.deal()
        rename(self.store, self.ann, MESSY)
        text = self.view()["text"]
        self.assertIn(CLEAN, text)
        for bad in ("<@", "*", "_", MESSY):
            self.assertNotIn(bad, text)

    async def test_rendered_table_message_is_unchanged(self):
        """Characterization (FEAT-48): the table message the channel sees keeps its mentions and up card (the public view is separate)."""
        await self.deal()
        text = self.games.render(self.latest())
        self.assertIn("<@5>", text)
        self.assertRegex(text, re.compile(r"Dealer: 5♠ \?\?"))
