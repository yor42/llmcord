"""FEAT-49b2: the decision-model veto wired into the spoken blackjack move check (docs/engineering/research/spoken-veto-design.md).

Offline: a fake veto client, the stub e5 runtime and fake Discord objects of test_games_spoken_discord.py. These tests are written ahead
of the implementation. The seams they require are:

* ``SpokenMoves.veto``: the client or ``None``. Anything with ``async ask(text, legal)`` -> ``{move: probability}`` (raising
  ``VetoError`` on failure), a ``breaker_open`` property and ``async close()``. ``SpokenMoves(bot, env=None)`` builds it with
  ``veto_client.from_env(env if env is not None else os.environ)``; tests assign the attribute directly.
* ``SpokenMoves.veto_timeout``: seconds the whole ``ask`` may take (``asyncio.wait_for``); on expiry the e5 buttons stand.
* ``SpokenMoves.limiter``: a ``llmcord_core.spoken.veto_limits.VetoLimiter``. ``VetoLimiter(per_minute=30, per_day=5000, clock=time.monotonic,
  today=<callable returning the current day, any hashable>)`` with attributes ``per_minute``/``per_day`` and ``allow(guild_id) -> bool``
  (consumes one call from the guild's token bucket and from the process-wide daily count; False means "use e5 and do not call").
* ``ask`` is called once as ``ask(text, legal)`` where ``text`` is the stripped message e5 embedded and ``legal`` is the list of legal
  move names that ``interpret.decide`` was given; the result goes through ``veto.apply_policy(band, e5 moves, probs)``.
* ``SpokenMoves.close()`` awaits ``veto.close()`` once and drops the client.
"""
import asyncio
import logging
import unittest
from unittest import mock

from test_games_spoken_discord import ALICE, BOB, G, Decision, Said, SpokenCase, gsd
from test_games_spoken_doubt import SpokenDoubtCase

from llmcord_core.spoken import veto_client
from llmcord_core.spoken.veto import VetoError

SENTINEL = "zq-sentinel-phrase-7731"
FAKE_KEY = "FAKE-veto-key-not-a-real-key"
ON_ENV = {"LLMCORD_SPOKEN_VETO": "openrouter-d1", "OPENROUTER_API_KEY": FAKE_KEY}


class FakeVeto:
    def __init__(self):
        self.calls, self.probs, self.error, self.gate, self.breaker_open = [], {}, None, None, False
        self.entered, self.closes = asyncio.Event(), 0

    async def ask(self, text, legal):
        self.calls.append((text, list(legal)))
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return {m: self.probs.get(m, 0.0) for m in legal}

    async def close(self):
        self.closes += 1


class VetoCase(SpokenCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.fake = self.spoken.veto = FakeVeto()
        self.store.set_spoken_veto_enabled(G, True)
        self.decision = Decision("unsure", ("hit", "stand"))
        self.fake.probs = {"hit": 0.9, "stand": 0.9}

    def assert_e5_buttons(self, msg):
        self.assertEqual(len(msg.posts), 1)
        self.assertEqual(self.labels(msg), ["Hit", "Stand", "Just table talk"])
        self.assertEqual(msg.posts[0].content, "Did you mean **Hit** or **Stand**?")

    def assert_called_once_for(self, text):
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(self.fake.calls[0][0], text)
        self.assertEqual(sorted(self.fake.calls[0][1]), self.legal_seen[-1])


class OffTests(VetoCase):
    async def test_no_client_is_identical_to_today(self):
        """FEAT-49b2: with no client the move check is exactly e5's."""
        await self.deal()
        self.spoken.veto = None
        self.assert_e5_buttons(await self.say(ALICE))

    async def test_guild_switch_off_never_calls(self):
        """FEAT-49b2: the server switch defaults off and gates the call."""
        await self.deal()
        self.store.set_spoken_veto_enabled(G, False)
        self.assert_e5_buttons(await self.say(ALICE))
        self.assertEqual(self.fake.calls, [])

    async def test_member_opt_out_never_calls(self):
        """FEAT-49b2: a member who opted out of the veto is not sent anywhere."""
        await self.deal()
        self.store.set_spoken_veto_member_on(G, ALICE, False)
        self.assert_e5_buttons(await self.say(ALICE))
        self.assertEqual(self.fake.calls, [])


class NoCallTests(VetoCase):
    async def test_e5_silent_never_calls(self):
        """FEAT-49b2: only a non-silent e5 reading leaves the Pi."""
        await self.deal()
        self.decision = Decision("silent", ())
        msg = await self.say(ALICE)
        self.assertEqual((msg.posts, self.fake.calls), ([], []))

    async def test_not_the_members_turn_never_calls(self):
        """FEAT-49b2: only the member on turn is read."""
        await self.deal()
        msg = await self.say(BOB)
        self.assertEqual((msg.posts, self.fake.calls, self.runtime.texts), ([], [], []))

    async def test_long_message_never_calls(self):
        """FEAT-49b2: more than 200 characters is chat and is not sent."""
        await self.deal()
        msg = await self.say(ALICE, "x" * (gsd.MAX_CHARS + 1))
        self.assertEqual((msg.posts, self.fake.calls), ([], []))

    async def test_member_spoken_moves_off_never_calls(self):
        """FEAT-49b2: spoken moves off for the member."""
        await self.deal()
        self.store.set_spoken_member_on(G, ALICE, False)
        msg = await self.say(ALICE)
        self.assertEqual((msg.posts, self.fake.calls), ([], []))

    async def test_guild_spoken_moves_off_never_calls(self):
        """FEAT-49b2: spoken moves off for the server."""
        await self.deal()
        self.store.set_spoken_moves_enabled(G, False)
        msg = await self.say(ALICE)
        self.assertEqual((msg.posts, self.fake.calls), ([], []))

    async def test_open_breaker_skips_the_call_and_keeps_e5(self):
        """FEAT-49b2: an open breaker means no call and e5's buttons."""
        await self.deal()
        self.fake.breaker_open = True
        self.assert_e5_buttons(await self.say(ALICE))
        self.assertEqual(self.fake.calls, [])


class DoubtPathTests(SpokenDoubtCase):
    async def test_i_doubt_it_checks_never_call(self):
        """FEAT-49b2: the doubt reading is not part of the veto."""
        fake = self.bot.spoken.veto = FakeVeto()
        self.store.set_spoken_veto_enabled(G, True)
        seat = await self.window()
        chat = await self.said(self.doubter(seat))
        self.assertEqual(len(chat.posts), 1)  # the doubt check was offered, so the path really ran
        self.assertEqual(fake.calls, [])


class AskTests(VetoCase):
    async def test_ask_gets_only_the_stripped_text_and_legal_moves(self):
        """FEAT-49b2: one call with (text, legal) and nothing identifying."""
        await self.deal()
        await self.say(ALICE, f"  {SENTINEL}  ")
        self.assert_called_once_for(SENTINEL)
        self.assertEqual(self.runtime.texts, [SENTINEL])
        flat = repr(self.fake.calls)
        for private in (str(ALICE), str(BOB), str(G), str(self.key()[1]), "Said"):
            self.assertNotIn(private, flat.replace(SENTINEL, ""))

    async def test_all_kept_is_e5_unchanged(self):
        """FEAT-49b2: a veto that agrees changes nothing."""
        await self.deal()
        self.assert_e5_buttons(await self.say(ALICE))
        self.assert_called_once_for("hit me")
        self.assertEqual(self.spoken.pending[self.key()].band, "unsure")

    async def test_dropped_move_means_fewer_buttons(self):
        """FEAT-49b2: a move scored below tau is not offered."""
        await self.deal()
        self.fake.probs = {"hit": 0.9, "stand": 0.1}
        msg = await self.say(ALICE)
        self.assertEqual(self.labels(msg), ["Hit", "Just table talk"])
        self.assertEqual(msg.posts[0].content, "Did you mean **Hit**?")
        self.assertEqual(self.spoken.pending[self.key()].offered, ("hit",))

    async def test_all_dropped_posts_nothing_like_silent(self):
        """FEAT-49b2: nothing kept is treated as e5-silent: no check."""
        await self.deal()
        self.fake.probs = {"hit": 0.2, "stand": 0.1}
        msg = await self.say(ALICE)
        self.assertEqual((msg.posts, self.spoken.pending), ([], {}))
        self.assertEqual(len(self.fake.calls), 1)

    async def test_reorder_is_respected(self):
        """FEAT-49b2: the higher-scored move comes first."""
        await self.deal()
        self.fake.probs = {"hit": 0.6, "stand": 0.8}
        msg = await self.say(ALICE)
        self.assertEqual(self.labels(msg), ["Stand", "Hit", "Just table talk"])
        self.assertEqual(msg.posts[0].content, "Did you mean **Stand** or **Hit**?")
        self.assertEqual(self.spoken.pending[self.key()].offered, ("stand", "hit"))

    async def test_sure_stays_sure_when_its_single_move_is_kept(self):
        """FEAT-49b2: sure survives only for the same single move."""
        await self.deal()
        self.decision = Decision("sure", ("hit",))
        self.fake.probs = {"hit": 0.9}
        msg = await self.say(ALICE)
        self.assertEqual(self.labels(msg), ["Hit", "Just table talk"])
        self.assert_called_once_for("hit me")
        self.assertEqual(self.spoken.pending[self.key()].band, "sure")

    async def test_sure_dropped_is_silent(self):
        """FEAT-49b2: a vetoed sure reading offers nothing."""
        await self.deal()
        self.decision = Decision("sure", ("hit",))
        self.fake.probs = {"hit": 0.1}
        msg = await self.say(ALICE)
        self.assertEqual((msg.posts, self.spoken.pending), ([], {}))

    async def test_unsure_is_never_promoted_to_sure(self):
        """FEAT-49b2: dropping one of two moves leaves an unsure check."""
        await self.deal()
        self.fake.probs = {"hit": 0.99, "stand": 0.0}
        await self.say(ALICE)
        self.assertEqual(len(self.fake.calls), 1)
        self.assertEqual(self.spoken.pending[self.key()].band, "unsure")


class FailOpenTests(VetoCase):
    async def _fails_open(self, error):
        await self.deal()
        self.fake.error = error
        self.assert_e5_buttons(await self.say(ALICE))
        self.assertEqual(len(self.fake.calls), 1)

    async def test_veto_error_keeps_e5(self):
        """FEAT-49b2: a VetoError leaves e5's buttons."""
        await self._fails_open(VetoError("status 500"))

    async def test_asyncio_timeout_keeps_e5(self):
        """FEAT-49b2: a timeout from the client leaves e5's buttons."""
        await self._fails_open(asyncio.TimeoutError())

    async def test_unexpected_exception_keeps_e5(self):
        """FEAT-49b2: even an unexpected error leaves e5's buttons and escapes nowhere."""
        await self._fails_open(RuntimeError("boom"))

    async def test_slow_client_hits_the_configured_timeout(self):
        """FEAT-49b2: veto_timeout bounds the whole ask."""
        await self.deal()
        self.spoken.veto_timeout = 0.05
        self.fake.gate = asyncio.Event()  # never set
        self.assert_e5_buttons(await self.say(ALICE))
        self.assertEqual(len(self.fake.calls), 1)


class StaleTests(VetoCase):
    async def test_table_moved_on_during_the_await_posts_nothing(self):
        """FEAT-49b2: the staleness check is repeated after the veto call."""
        await self.deal()
        self.fake.gate = asyncio.Event()
        msg = Said(ALICE, "hit me")
        self.spoken.note_message(msg)
        await asyncio.wait_for(self.fake.entered.wait(), 1)
        await self.press(ALICE, "Hit")
        self.fake.gate.set()
        await self.settle()
        self.assertEqual((msg.posts, self.spoken.pending), ([], {}))

    async def test_newer_phrase_during_the_await_drops_the_older_reading(self):
        """FEAT-49b2: a superseded reading posts nothing after the veto answers."""
        await self.deal()
        self.fake.gate = asyncio.Event()
        old = Said(ALICE, "hit")
        self.spoken.note_message(old)
        await asyncio.wait_for(self.fake.entered.wait(), 1)
        new = Said(ALICE, "stand")
        self.spoken.note_message(new)
        self.fake.gate.set()
        await self.settle()
        self.assertEqual((old.posts, len(new.posts)), ([], 1))


class LimiterUnitTests(unittest.TestCase):
    def make(self, **kw):
        from llmcord_core.spoken.veto_limits import VetoLimiter
        self.now, self.day = [0.0], [1]
        return VetoLimiter(clock=lambda: self.now[0], today=lambda: self.day[0], **kw)

    def test_defaults(self):
        """FEAT-49b2: 30 calls a minute per guild, 5,000 a day per process."""
        from llmcord_core.spoken.veto_limits import VetoLimiter
        limiter = VetoLimiter()
        self.assertEqual((limiter.per_minute, limiter.per_day), (30, 5000))

    def test_guild_bucket_runs_out_then_refills(self):
        """FEAT-49b2: the per-guild token bucket."""
        limiter = self.make(per_minute=2, per_day=100)
        self.assertEqual([limiter.allow(1) for _ in range(3)], [True, True, False])
        self.now[0] = 30.0  # half a minute refills one of two
        self.assertEqual([limiter.allow(1) for _ in range(2)], [True, False])

    def test_buckets_are_per_guild(self):
        """FEAT-49b2: one busy server does not block another."""
        limiter = self.make(per_minute=1, per_day=100)
        self.assertTrue(limiter.allow(1))
        self.assertFalse(limiter.allow(1))
        self.assertTrue(limiter.allow(2))

    def test_daily_cap_is_per_process_and_resets_on_a_new_day(self):
        """FEAT-49b2: the daily cap spans guilds and resets with the day."""
        limiter = self.make(per_minute=100, per_day=3)
        self.assertEqual([limiter.allow(g) for g in (1, 2, 3, 1)], [True, True, True, False])
        self.day[0] = 2
        self.assertTrue(limiter.allow(1))


class LimitWiringTests(VetoCase):
    def limit(self, **kw):
        from llmcord_core.spoken.veto_limits import VetoLimiter
        self.spoken.limiter = VetoLimiter(clock=lambda: 0.0, today=lambda: 1, **kw)

    async def test_per_minute_limit_falls_back_to_e5_without_calling(self):
        """FEAT-49b2: beyond the bucket the client is not called and e5's buttons stand."""
        await self.deal()
        self.limit(per_minute=1, per_day=100)
        self.fake.probs = {"hit": 0.9, "stand": 0.1}
        first = await self.say(ALICE, "one")
        self.assertEqual(self.labels(first), ["Hit", "Just table talk"])  # vetoed
        second = await self.say(ALICE, "two")
        self.assert_e5_buttons(second)  # limited: e5 untouched
        self.assertEqual([c[0] for c in self.fake.calls], ["one"])

    async def test_daily_cap_falls_back_to_e5_without_calling(self):
        """FEAT-49b2: the daily cap behaves the same way."""
        await self.deal()
        self.limit(per_minute=100, per_day=1)
        self.fake.probs = {"hit": 0.9, "stand": 0.1}
        await self.say(ALICE, "one")
        self.assert_e5_buttons(await self.say(ALICE, "two"))
        self.assertEqual(len(self.fake.calls), 1)


class PrivacyTests(VetoCase):
    def capture(self):
        records = []

        class Grab(logging.Handler):
            def emit(self, record):
                records.append(logging.Formatter().format(record))

        root, handler = logging.getLogger(), Grab(logging.DEBUG)
        old = root.level
        root.setLevel(logging.DEBUG)
        root.addHandler(handler)
        self.addCleanup(root.setLevel, old)
        self.addCleanup(root.removeHandler, handler)
        return records

    async def test_text_never_reaches_a_log_record(self):
        """FEAT-49b2: message text is not logged on success, error, timeout, unexpected-error or limit paths."""
        records = self.capture()
        await self.deal()
        self.fake.probs = {"hit": 0.9, "stand": 0.1}
        await self.say(ALICE, f"success {SENTINEL}")
        self.fake.probs = {"hit": 0.1, "stand": 0.1}
        await self.say(ALICE, f"dropped {SENTINEL}")
        for error in (VetoError("status 500"), asyncio.TimeoutError(), RuntimeError("boom")):
            self.fake.error = error
            await self.say(ALICE, f"error {SENTINEL}")
        self.fake.error = None
        from llmcord_core.spoken.veto_limits import VetoLimiter
        self.spoken.limiter = VetoLimiter(per_minute=1, per_day=1, clock=lambda: 0.0, today=lambda: 1)
        await self.say(ALICE, f"first {SENTINEL}")
        await self.say(ALICE, f"limited {SENTINEL}")
        self.assertTrue(self.fake.calls)
        self.assertNotIn(SENTINEL, "\n".join(records))


class LifecycleTests(SpokenCase):
    async def test_close_closes_the_client_once(self):
        """FEAT-49b2: close() awaits the client's close exactly once, even when called again."""
        fake = self.spoken.veto = FakeVeto()
        await self.spoken.close()
        await self.spoken.close()
        self.assertEqual(fake.closes, 1)

    async def test_no_client_without_the_env_switch(self):
        """FEAT-49b2: the operator switch decides; nothing is created otherwise."""
        for env in ({}, {"OPENROUTER_API_KEY": FAKE_KEY}, {"LLMCORD_SPOKEN_VETO": "off", "OPENROUTER_API_KEY": FAKE_KEY},
                    {"LLMCORD_SPOKEN_VETO": "openrouter-d1"}):
            self.assertIsNone(gsd.SpokenMoves(self.bot, env=env).veto, env)

    async def test_client_made_from_the_env_and_closed_with_the_handler(self):
        """FEAT-49b2: LLMCORD_SPOKEN_VETO=openrouter-d1 plus a key builds the real client; close() closes it."""
        spoken = gsd.SpokenMoves(self.bot, env=dict(ON_ENV))
        self.assertIsInstance(spoken.veto, veto_client.VetoClient)
        with mock.patch.object(veto_client.VetoClient, "close", new_callable=mock.AsyncMock) as close:
            await spoken.close()
        close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
