"""Memory extraction + scene summary after delivery, and the per-channel channel lock (REL-02 part 2).

Seams: ``SkitBot.run_scene`` driven end to end through the real ``_webhook`` path with ``FakeTextChannel``,
under ``bot.channel_locks[channel]`` exactly like ``on_message`` / ``/summon`` do. Background memory work is
reached through ``bot.memory_tasks`` (channel id -> latest task; see ``helpers.drain_memory_tasks``), the
module constants ``llmcord_core.discord_bot.MEMORY_WAIT_SECONDS`` (how long the next turn waits for the
previous turn's memory task) and ``llmcord_core.discord_bot.MEMORY_CLOSE_SECONDS`` (how long ``close()`` waits
before cancelling), both patched with ``create=True`` so the known-defect tests fail on behavior, not import.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from discord.ext import commands

from helpers import MemoryModels, FakeTextChannel, drain_memory_tasks, make_settings
from llmcord_core.discord_bot import SkitBot
from llmcord_core.engine import SceneContext

CHANNEL = 100


class GatedModels(MemoryModels):
    """MemoryModels whose summary calls (``text``) are numbered 0, 1, ... in call order.

    Calls whose index is in ``blocked`` wait on ``gate``; call 0 additionally sleeps ``first_delay`` seconds.
    ``events`` (shared with the test via ``order``) records ``("start", i)``, ``("end", i)`` and
    ``("cancelled", i)``. Each summary returns ``"Summary #i"``. ``prompts`` holds the joined system + message
    text of every dialogue stream request.
    """

    def __init__(self, speakers, blocked=(), first_delay=0.0, order=None):
        super().__init__()
        self.chosen = speakers
        self.blocked, self.first_delay = set(blocked), first_delay
        self.gate, self.started = asyncio.Event(), asyncio.Event()
        self.events = order if order is not None else []
        self.prompts = []

    async def text(self, role, system, messages, max_tokens=None):
        index = len(self.summaries)
        self.events.append(("start", index))
        self.started.set()
        try:
            if index == 0 and self.first_delay:
                await asyncio.sleep(self.first_delay)
            if index in self.blocked:
                await self.gate.wait()
        except asyncio.CancelledError:
            self.events.append(("cancelled", index))
            raise
        await super().text(role, system, messages, max_tokens)
        self.events.append(("end", index))
        return f"Summary #{index}"

    async def stream_text(self, role, system, messages):
        self.prompts.append(self._joined(system, messages))
        async for piece in super().stream_text(role, system, messages):
            yield piece


class MemoryBackgroundCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.world = self.store.create_space(1, "World", "world")
        self.store.bind_channel(1, CHANNEL, self.world)
        self.alice = self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])
        self.store.set_cast(CHANNEL, None, [self.alice])
        self.channel = FakeTextChannel(CHANNEL)

    async def asyncTearDown(self):
        for task in getattr(self.bot, "memory_tasks", {}).values():
            if isinstance(task, asyncio.Future) and not task.done():
                task.cancel()
        await asyncio.sleep(0)

    def tearDown(self):
        self.store.close()

    def use(self, models):
        self.bot.engine.models = self.bot.models = models
        return models

    async def turn(self, ident, text="Hi", parent=None, within=1.0):
        """One turn the way on_message runs it: hold the channel lock around run_scene (bounded)."""
        scene = SceneContext(1, CHANNEL, None, self.world, 9, ident, text, parent, [], [])
        async with self.bot.channel_locks.setdefault(CHANNEL, asyncio.Lock()):
            await asyncio.wait_for(self.bot.run_scene(scene, self.channel), within)

    @property
    def replies(self):
        return [post for hook in self.channel.hooks for post in hook.posts]

    def last_reply_id(self):
        return self.replies[-1].id


class MemoryCharacterizationTests(MemoryBackgroundCase):
    async def test_turn_runs_extraction_and_summary_on_last_reply(self):
        """Characterization (REL-02): after a successful turn, memory extraction and the scene summary both run
        once for that turn and the summary is saved on the last character reply node."""
        models = self.use(GatedModels([self.alice]))
        await self.turn(1001)
        await drain_memory_tasks(self.bot)
        self.assertEqual(self.channel.errors, [])
        self.assertEqual(len(models.extractions), 1)
        self.assertEqual(len(models.summaries), 1)
        self.assertIn("Hello.", models.extractions[0]["text"])
        self.assertEqual(self.store.summary(self.last_reply_id()), "Summary #0")

    async def test_summary_model_failure_keeps_delivered_reply(self):
        """Characterization (REL-02): a summary model failure is logged; the delivered reply stays posted and
        saved, nothing is reported in the channel, and no summary is saved."""
        self.use(MemoryModels(summary_failures=1)).chosen = [self.alice]
        with self.assertLogs(level="ERROR") as logs:
            await self.turn(1001)
            await drain_memory_tasks(self.bot)
        self.assertIn("summary provider down", "\n".join(logs.output))
        reply = self.replies[-1]
        self.assertFalse(reply.deleted)
        self.assertIn("Hello.", reply.content)
        self.assertIsNotNone(self.store.node(reply.id))
        self.assertIsNone(self.store.summary(reply.id))
        self.assertEqual(self.channel.errors, [])

    async def test_next_turn_prompt_sees_previous_summary(self):
        """Characterization (REL-02 ordering guarantee): when turn 1's summary finishes within
        MEMORY_WAIT_SECONDS (here it takes 50 ms), turn 2's dialogue prompt already carries turn 1's summary."""
        models = self.use(GatedModels([self.alice], first_delay=0.05))
        await self.turn(1001, "First words")
        await self.turn(1002, "Second words", parent=self.last_reply_id())
        await drain_memory_tasks(self.bot)
        self.assertEqual(self.channel.errors, [])
        self.assertEqual(len(models.prompts), 2)
        self.assertIn("Summary #0", models.prompts[1])

    async def test_memory_usage_is_attributed_to_the_scene_guild(self):
        """Characterization (REL-02): memory-model usage (extraction + summary) recorded through the real
        ModelGateway usage sink lands in model_usage with the scene's guild_id: one row per provider call (rows
        with no guild are dropped, so losing the attribution in a background task would lose rows)."""
        alice = self.alice

        async def create(**kwargs):
            usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
            if kwargs.get("stream"):
                async def stream():
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                        content="<emotion>neutral</emotion>\nHello."))], usage=None)
                    yield SimpleNamespace(choices=[], usage=usage)
                return stream()
            schema = kwargs.get("response_format", {}).get("json_schema", {}).get("schema", {})
            if "speakers" in schema.get("properties", {}):
                text = '{"speakers":[' + str(alice) + ']}'
            elif schema:
                text = '{"shared_facts":[],"personal_facts":[],"encounter_facts":[]}'
            else:
                text = "Gateway summary"
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text),
                                                            finish_reason="stop")], usage=usage)

        profiles = self.bot.settings.profiles
        profiles["test"] = replace(profiles["test"], structured_outputs=True)
        client = AsyncMock(side_effect=create)
        self.bot.models.clients["test"] = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=client)))
        with self.assertNoLogs(level="ERROR"):
            await self.turn(1001)
            await drain_memory_tasks(self.bot)
        self.assertEqual(self.channel.errors, [])
        self.assertEqual(self.store.summary(self.last_reply_id()), "Gateway summary")
        rows = [tuple(row) for row in self.store.all("SELECT guild_id, role FROM model_usage")]
        self.assertEqual(len(rows), client.call_count, "a model call's usage was not attributed to a guild")
        self.assertEqual({guild for guild, _ in rows}, {1})
        self.assertGreaterEqual(sum(role == "memory" for _, role in rows), 2)  # extraction + summary


class MemoryBackgroundDefectTests(MemoryBackgroundCase):
    async def test_lock_released_before_summary_completes(self):
        """REL-02 (fixed): run_scene returns once the reply is delivered (channel lock released) while the summary is
        still running; releasing the blocked summary then saves it on the reply node."""
        models = self.use(GatedModels([self.alice], blocked={0}))
        await self.turn(1001)
        self.assertFalse(self.bot.channel_locks[CHANNEL].locked())
        self.assertIn("Hello.", self.replies[-1].content)
        await asyncio.wait_for(models.started.wait(), 1.0)
        self.assertEqual(models.events, [("start", 0)])
        self.assertIsNone(self.store.summary(self.last_reply_id()))
        models.gate.set()
        await drain_memory_tasks(self.bot)
        self.assertEqual(self.store.summary(self.last_reply_id()), "Summary #0")

    async def test_next_turn_waits_only_bounded_time(self):
        """REL-02 (fixed): with turn 1's summary hung, turn 2 waits at most MEMORY_WAIT_SECONDS, logs a WARNING,
        and is delivered within ~1 s; releasing the hang afterwards still saves turn 1's summary."""
        models = self.use(GatedModels([self.alice], blocked={0}))
        with patch("llmcord_core.discord_bot.MEMORY_WAIT_SECONDS", 0.2, create=True):
            await self.turn(1001)
            first_reply = self.last_reply_id()
            started = time.monotonic()
            with self.assertLogs(level="WARNING") as logs:
                await self.turn(1002, parent=first_reply)
            self.assertLess(time.monotonic() - started, 1.0)
        self.assertTrue(any(record.levelno == logging.WARNING for record in logs.records), logs.output)
        self.assertEqual(len(self.replies), 2)
        self.assertEqual(self.channel.errors, [])
        self.assertIsNone(self.store.summary(first_reply))
        models.gate.set()
        await drain_memory_tasks(self.bot)
        self.assertEqual(self.store.summary(first_reply), "Summary #0")

    async def test_memory_tasks_are_chained_per_channel(self):
        """REL-02 (fixed): memory tasks for one channel run in turn order: even when turn 2 stopped waiting for turn 1's
        hung summary, turn 2's summary starts only after turn 1's finishes."""
        models = self.use(GatedModels([self.alice], blocked={0}))
        with patch("llmcord_core.discord_bot.MEMORY_WAIT_SECONDS", 0.01, create=True), \
                self.assertLogs(level="WARNING"):
            await self.turn(1001)
            await self.turn(1002, parent=self.last_reply_id())
        await asyncio.sleep(0.05)
        self.assertEqual(models.events, [("start", 0)], "turn 2's summary started before turn 1's finished")
        models.gate.set()
        await drain_memory_tasks(self.bot)
        self.assertEqual(models.events, [("start", 0), ("end", 0), ("start", 1), ("end", 1)])

    async def test_background_failure_logged_not_posted(self):
        """REL-02 (fixed): an exception escaping the background summary is logged, is not posted to the channel
        ("Character response failed during scene summary"), and never surfaces as "Task exception was never
        retrieved"."""
        self.use(GatedModels([self.alice]))
        loop = asyncio.get_running_loop()
        unhandled = []
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
        try:
            with patch.object(self.bot.engine, "summarize_scene", AsyncMock(side_effect=RuntimeError("summary boom"))), \
                    self.assertLogs(level="ERROR") as logs:
                await self.turn(1001)
                await drain_memory_tasks(self.bot)
            if hasattr(self.bot, "memory_tasks"):
                self.bot.memory_tasks.clear()  # all finished; drop the last references so asyncio can report
            gc.collect()
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous)
        self.assertIn("summary boom", "\n".join(logs.output))
        self.assertIn("Hello.", self.replies[-1].content)
        self.assertEqual(self.channel.errors, [])
        self.assertEqual([c for c in unhandled if "never retrieved" in str(c.get("message", ""))], [])

    async def test_close_cancels_hung_memory_task_before_store_close(self):
        """REL-02 (fixed): close() with a hung memory task returns within a bounded time (MEMORY_CLOSE_SECONDS), cancels
        the task before closing the store, and nothing is written afterwards."""
        order = []
        models = self.use(GatedModels([self.alice], blocked={0}, order=order))
        turn = asyncio.create_task(self.turn(1001, within=5.0))
        try:
            await asyncio.wait_for(models.started.wait(), 1.0)
            await asyncio.sleep(0)
            reply_id = self.last_reply_id()
            with patch("llmcord_core.discord_bot.MEMORY_CLOSE_SECONDS", 0.1, create=True), \
                    patch.object(self.store, "close", lambda: order.append("store.close")), \
                    patch.object(commands.Bot, "close", AsyncMock()):
                await asyncio.wait_for(self.bot.close(), 1.0)
            self.assertIn(("cancelled", 0), order)
            self.assertLess(order.index(("cancelled", 0)), order.index("store.close"))
            models.gate.set()
            await asyncio.sleep(0.05)
            self.assertIsNone(self.store.summary(reply_id), "memory task wrote after close()")
        finally:
            models.gate.set()
            if not turn.done():
                turn.cancel()
            await asyncio.gather(turn, return_exceptions=True)

    async def test_close_cancels_queued_and_hung_memory_tasks_in_cascade(self):
        """REL-02 (fixed): with turn 1's memory task hung in its summary and turn 2's memory task queued behind it
        (turn 2 stopped waiting after MEMORY_WAIT_SECONDS), close() only tracks turn 2's task, but cancelling it
        cascades to turn 1's: both tasks are finished before the store closes, and releasing the hang afterwards
        writes no summary for either turn."""
        order = []
        models = self.use(GatedModels([self.alice], blocked={0}, order=order))
        with patch("llmcord_core.discord_bot.MEMORY_WAIT_SECONDS", 0.01, create=True), \
                self.assertLogs(level="WARNING"):
            await self.turn(1001)
            first_reply = self.last_reply_id()
            first_task = self.bot.memory_tasks[CHANNEL]
            await self.turn(1002, parent=first_reply)
        second_reply = self.last_reply_id()
        second_task = self.bot.memory_tasks[CHANNEL]
        self.assertIsNot(first_task, second_task)
        await asyncio.sleep(0.02)
        self.assertEqual(order, [("start", 0)], "turn 2's memory work was not queued behind turn 1's")
        self.assertFalse(first_task.done() or second_task.done())

        def close_store():
            order.append(("store.close", first_task.done(), second_task.done()))

        try:
            with patch("llmcord_core.discord_bot.MEMORY_CLOSE_SECONDS", 0.05, create=True), \
                    patch.object(self.store, "close", close_store), \
                    patch.object(commands.Bot, "close", AsyncMock()):
                await asyncio.wait_for(self.bot.close(), 1.0)
            self.assertIn(("store.close", True, True), order, "a memory task was still running at store.close")
            self.assertIn(("cancelled", 0), order)
            self.assertLess(order.index(("cancelled", 0)), order.index(("store.close", True, True)))
            self.assertTrue(first_task.cancelled())
            self.assertTrue(second_task.cancelled())
            models.gate.set()
            await asyncio.sleep(0.05)
            self.assertNotIn(("end", 0), order)
            self.assertEqual(len(models.summaries), 0, "a summary model call completed after close()")
            self.assertIsNone(self.store.summary(first_reply), "turn 1's memory task wrote after close()")
            self.assertIsNone(self.store.summary(second_reply), "turn 2's memory task wrote after close()")
        finally:
            models.gate.set()
            await asyncio.gather(first_task, second_task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
