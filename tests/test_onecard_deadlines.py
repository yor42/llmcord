"""FEAT-45 step 2: a declaration or call-out by chat does not change whose turn it is, so it must not extend the on-turn seat's deadline."""
from unittest import mock

from test_onecard_calls import CallCase, ALICE, BOB, G


class DeadlineTests(CallCase):
    async def start_exposed(self):
        self.onecard.CALLOUT_SECONDS = 3600
        await self.duo(["9♥", "4♥"])
        await self.pick_play(ALICE, "play:9♥")  # Alice is exposed; Bob is on turn
        self.key = (G, self.tid)
        self.deadline = self.onecard.table(self.key).deadlines["turn"]

    async def test_repeated_call_outs_keep_the_original_deadline_and_the_remaining_time(self):
        """FEAT-45: a hit and then false call-outs, each later in time, leave the stored deadline alone and re-arm for what is left."""
        await self.start_exposed()
        armed = []
        real = self.onecard.arm
        self.onecard.arm = lambda key, kind, delay, *args: (armed.append((kind, delay)), real(key, kind, delay, *args))[1]
        await self.tick(10)
        await self.say(BOB, "Alice forgot one card")  # a hit
        await self.tick(10)
        await self.say(ALICE, "Bob forgot one card")  # false: Alice draws
        await self.tick(10)
        await self.say(ALICE, "Bob forgot one card")
        self.assertEqual(self.onecard.table(self.key).deadlines["turn"], self.deadline)
        turns = [delay for kind, delay in armed if kind == "turn"]
        self.assertEqual(len(turns), 3)
        self.assertEqual([round(d) for d in turns], [50, 40, 30])
        self.assertEqual(self.log().count((0, "call_out:1", "member")), 2)

    async def test_the_turn_still_times_out_on_the_original_deadline(self):
        """FEAT-45: after the chat moves the timer armed for the newest move count fires and Bob draws on time (stale counts do nothing)."""
        await self.start_exposed()
        stale = self.snap()
        await self.tick(10)
        await self.say(ALICE, "Bob forgot one card")
        await self.tick(60)
        await self.onecard.on_turn_timer(G, self.tid, stale["round_id"], stale["moves"])
        self.assertNotIn((1, "timeout", "timeout"), self.log())
        snap = self.snap()
        await self.onecard.on_turn_timer(G, self.tid, snap["round_id"], snap["moves"])
        self.assertEqual(self.log()[-1], (1, "timeout", "timeout"))

    async def test_a_play_still_restarts_the_clock(self):
        """FEAT-45: only declarations and call-outs keep the deadline; the next real move gets a fresh one."""
        await self.start_exposed()
        await self.tick(30)
        await self.pick_play(BOB, "draw")
        self.assertGreater(self.onecard.table(self.key).deadlines["turn"], self.deadline)

    async def test_the_turn_message_is_not_moved_by_a_call_out(self):
        """FEAT-45: a call-out leaves the turn unchanged, so the move is not 'advanced' (the turn message is edited in place)."""
        await self.start_exposed()
        outs = []
        real = self.onecard._publish
        async def spy(key, out):
            outs.append(out)
            await real(key, out)
        with mock.patch.object(self.onecard, "_publish", spy):
            await self.say(BOB, "Alice forgot one card")
        self.assertEqual([o.advanced for o in outs], [False])
