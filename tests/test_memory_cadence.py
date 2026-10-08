"""BUG-03 (decisions D6 + D4): scene summary window, summary/extraction cadence and memory token budget.

Seams: the public ``Engine.summarize_scene`` / ``Engine.extract_memories`` with an in-memory ``Store`` and the
``MemoryModels`` fake (records each summary/extraction payload and the summary ``max_tokens``); limits come from
``settings.limits`` (``make_settings(**limits)``) and, for config validation, the real ``load_settings``.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from helpers import MemoryModels, make_settings
from llmcord_core.engine import Engine, SceneContext
from llmcord_core.store import Store
from helpers import settings_for

USER = 111
OTHER_USER = 222


def line(ident: int) -> str:
    return f"line-{ident}-x"


class MemoryCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Store()
        self.world = self.store.create_space(1, "World", "world")
        self.store.bind_channel(1, 100, self.world)
        self.courier = self.store.add_character(1, self.world, "Courier", {"name": "Courier"}, None, [])
        self.guard = self.store.add_character(1, self.world, "Guard", {"name": "Guard"}, None, [])

    def tearDown(self):
        self.store.close()

    def engine(self, models, **limits) -> Engine:
        return Engine(self.store, models, make_settings(**limits))

    def chain(self, count, start=1000, parent=None) -> int:
        """Record ``count`` user nodes in one branch; returns the newest node id."""
        for ident in range(start, start + count):
            self.store.record_node(ident, 1, 100, parent, USER, None, line(ident), created_at=1_700_000_000.0)
            parent = ident
        return parent

    def scene(self, ident, text="Hello", parent=None, recent=None, user_id=USER) -> SceneContext:
        return SceneContext(1, 100, None, self.world, user_id, ident, text, parent, recent or [], [], user_label="Sam")

    def characters(self, *ids):
        return [self.store.character_by_id(ident) for ident in ids]


class SummaryCharacterizationTests(MemoryCase):
    async def test_short_branch_summarized_every_turn_with_prior_summary(self):
        """Characterization (BUG-03): by default a branch with <= 30 new nodes is summarized on every call;
        the payload carries the previous summary and every new node, and the result is saved on last_message_id."""
        models = MemoryModels(summary="Second summary")
        engine = self.engine(models)
        first = self.chain(5)
        self.store.save_summary(first, "First summary")
        last = self.chain(30, start=1005, parent=first)
        await engine.summarize_scene(last)
        self.assertEqual(len(models.summaries), 1)
        payload = models.summaries[0]["text"]
        self.assertIn("First summary", payload)
        self.assertIn(line(1005), payload)
        self.assertIn(line(last), payload)
        self.assertNotIn(line(1000), payload)  # covered by the previous summary
        self.assertEqual(self.store.summary(last), "Second summary")

        self.store.record_node(5000, 1, 100, last, USER, None, "one more")
        await engine.summarize_scene(5000)
        self.assertEqual(len(models.summaries), 2)
        self.assertIn("Second summary", models.summaries[1]["text"])
        self.assertEqual(self.store.summary(5000), "Second summary")

    async def test_default_summary_max_tokens_is_550(self):
        """Characterization (BUG-03/D4): with default limits the summary call gets max_tokens=550."""
        models = MemoryModels()
        await self.engine(models).summarize_scene(self.chain(3))
        self.assertEqual(models.summaries[0]["max_tokens"], 550)

    async def test_model_failure_logs_and_saves_nothing(self):
        """Characterization (BUG-03): a failing summary call is logged at ERROR and no summary is saved."""
        models = MemoryModels(summary_failures=1)
        last = self.chain(4)
        with self.assertLogs(level="ERROR") as logs:
            await self.engine(models).summarize_scene(last)
        self.assertIn("Scene summarization failed", "\n".join(logs.output))
        self.assertIsNone(self.store.summary(last))

    async def test_transcript_failure_is_logged_not_raised(self):
        """BUG-03/D4 (fixed): an exception while building the summary transcript (Engine.history_messages) is caught
        inside summarize_scene, logged at ERROR as "Scene summarization failed", and nothing is saved."""
        models = MemoryModels()
        last = self.chain(4)
        with patch.object(Engine, "history_messages", side_effect=RuntimeError("transcript boom")):
            with self.assertLogs(level="ERROR") as logs:
                await self.engine(models).summarize_scene(last)
        self.assertIn("Scene summarization failed", "\n".join(logs.output))
        self.assertEqual(models.summaries, [])
        self.assertIsNone(self.store.summary(last))

    async def test_default_cadence_summarizes_after_one_new_node(self):
        """Characterization (BUG-03/D4): with default summary_every_messages a single new node triggers a call."""
        models = MemoryModels()
        first = self.chain(3)
        self.store.save_summary(first, "Old")
        last = self.chain(1, start=1003, parent=first)
        await self.engine(models).summarize_scene(last)
        self.assertEqual(len(models.summaries), 1)
        self.assertEqual(self.store.summary(last), "Fresh summary")


class SummaryWindowTests(MemoryCase):
    async def test_long_branch_summarizes_latest_window(self):
        """BUG-03 (fixed): a branch with 40 unsummarized nodes is summarized from its most recent window (D6), sized by
        memory_input_tokens: the newest node is in the payload, the oldest is not, and the summary is saved."""
        models = MemoryModels()
        last = self.chain(40)
        await self.engine(models, memory_input_tokens=60).summarize_scene(last)
        self.assertEqual(len(models.summaries), 1)
        payload = models.summaries[0]["text"]
        self.assertIn(line(last), payload)
        self.assertNotIn(line(1000), payload)
        self.assertEqual(self.store.summary(last), "Fresh summary")

    async def test_long_window_keeps_previous_summary(self):
        """BUG-03 (fixed): past the cap the windowed payload is still prefixed by the previous saved summary."""
        models = MemoryModels()
        first = self.chain(3)
        self.store.save_summary(first, "Prior summary text")
        last = self.chain(40, start=1003, parent=first)
        await self.engine(models, memory_input_tokens=60).summarize_scene(last)
        self.assertEqual(len(models.summaries), 1)
        payload = models.summaries[0]["text"]
        self.assertIn("Prior summary text", payload)
        self.assertIn(line(last), payload)
        self.assertNotIn(line(1003), payload)

    async def test_failed_long_summary_does_not_stall(self):
        """BUG-03 (fixed): after a failed summary at 31+ unsummarized nodes, the next turn still summarizes."""
        models = MemoryModels(summary_failures=1)
        engine = self.engine(models)
        last = self.chain(31)
        with self.assertLogs(level="ERROR"):
            await engine.summarize_scene(last)
        self.assertIsNone(self.store.summary(last))
        self.store.record_node(5000, 1, 100, last, USER, None, "one more")
        await engine.summarize_scene(5000)
        self.assertEqual(self.store.summary(5000), "Fresh summary")

    async def test_memory_output_tokens_is_summary_max_tokens(self):
        """BUG-03/D4 (fixed): limits.memory_output_tokens is passed as the summary call's max_tokens."""
        models = MemoryModels()
        await self.engine(models, memory_output_tokens=321).summarize_scene(self.chain(3))
        self.assertEqual(models.summaries[0]["max_tokens"], 321)

    async def test_summary_every_messages_gates_calls(self):
        """BUG-03/D4 (fixed): with summary_every_messages=3, two new nodes since the last summary make no model call;
        the third new node makes exactly one."""
        models = MemoryModels()
        engine = self.engine(models, summary_every_messages=3)
        first = self.chain(5)
        self.store.save_summary(first, "Old")
        second = self.chain(2, start=1005, parent=first)
        await engine.summarize_scene(second)
        self.assertEqual(models.summaries, [])
        self.assertIsNone(self.store.summary(second))
        third = self.chain(1, start=1007, parent=second)
        await engine.summarize_scene(third)
        self.assertEqual(len(models.summaries), 1)
        self.assertEqual(self.store.summary(third), "Fresh summary")


class ExtractionCharacterizationTests(MemoryCase):
    async def test_extraction_runs_once_per_speaker_each_turn(self):
        """Characterization (BUG-03/D4): by default every turn runs one extraction call per speaker that spoke,
        and none when no character spoke."""
        models = MemoryModels()
        engine = self.engine(models)
        self.chain(1)
        speakers = self.characters(self.courier, self.guard)
        await engine.extract_memories(self.scene(1000), speakers, [("Courier", "Hi"), ("Guard", "Halt")], 1000)
        self.assertEqual(len(models.extractions), 2)
        self.assertIn("Character: Courier", models.extractions[0]["text"])
        self.assertIn("Character: Guard", models.extractions[1]["text"])
        await engine.extract_memories(self.scene(1000), speakers, [], 1000)
        self.assertEqual(len(models.extractions), 2)
        self.store.record_node(1001, 1, 100, 1000, None, self.courier, "Hi")
        self.chain(1, start=1002, parent=1001)
        await engine.extract_memories(self.scene(1002, parent=1001), speakers[:1], [("Courier", "Again")], 1000)
        self.assertEqual(len(models.extractions), 3)

    async def test_personal_fact_needs_consent_and_goes_to_current_user_only(self):
        """Characterization (BUG-03, consent invariant): personal facts are stored only for scene.user_id, only
        after opt-in, sourced to the current user message; other users in the recent chat get nothing, and
        opt-out deletes them."""
        models = MemoryModels(extraction_result={"shared_facts": [], "personal_facts": ["Sam likes green tea"],
                                                 "encounter_facts": []})
        engine = self.engine(models)
        self.chain(1)
        recent = [{"message_id": 900, "author_id": OTHER_USER, "author_label": "Lee", "text": "I like coffee"}]
        speaker = self.characters(self.courier)
        await engine.extract_memories(self.scene(1000, recent=recent), speaker, [("Courier", "Noted")], 1000)
        self.assertEqual(self.store.personal(1, USER), [])

        self.store.set_consent(1, USER, True)
        self.store.set_consent(1, OTHER_USER, True)
        await engine.extract_memories(self.scene(1000, recent=recent), speaker, [("Courier", "Noted")], 1000)
        rows = self.store.personal(1, USER)
        self.assertEqual([(r["content"], r["character_id"], r["source_message_id"]) for r in rows],
                         [("Sam likes green tea", self.courier, 1000)])
        self.assertEqual(self.store.personal(1, OTHER_USER), [])

        self.store.set_consent(1, USER, False)
        self.assertEqual(self.store.personal(1, USER), [])

    async def test_short_extraction_text_is_sent_whole(self):
        """Characterization (BUG-03/D4): with default limits a short turn's full text reaches extraction."""
        models = MemoryModels()
        self.chain(1)
        scene = self.scene(1000, text="ALPHA-START hello")
        await self.engine(models).extract_memories(scene, self.characters(self.courier), [("Courier", "OMEGA-END")], 1000)
        self.assertIn("ALPHA-START", models.extractions[0]["text"])
        self.assertIn("OMEGA-END", models.extractions[0]["text"])

    async def test_extraction_text_failure_is_logged_not_raised(self):
        """BUG-03/D4 (fixed): an exception while building the extraction text (the user line, via engine.user_line) is
        caught inside extract_memories and logged at ERROR as "Memory extraction failed"."""
        models = MemoryModels()
        self.chain(1)
        with patch("llmcord_core.engine.user_line", side_effect=RuntimeError("text boom")):
            with self.assertLogs(level="ERROR") as logs:
                await self.engine(models).extract_memories(self.scene(1000), self.characters(self.courier),
                                                           [("Courier", "Hi")], 1000)
        self.assertIn("Memory extraction failed", "\n".join(logs.output))
        self.assertEqual(models.extractions, [])


class ExtractionCadenceTests(MemoryCase):
    async def test_extraction_every_turns_runs_on_even_user_turns(self):
        """BUG-03/D4 (fixed): with extraction_every_turns=2 extraction runs only when the count of user-authored nodes on
        the branch (ancestors of scene.user_message_id, inclusive) is even."""
        models = MemoryModels()
        engine = self.engine(models, extraction_every_turns=2)
        speaker = self.characters(self.courier)
        counts = []
        parent = None
        for user_node in (1000, 1002, 1004):
            self.store.record_node(user_node, 1, 100, parent, USER, None, f"turn {user_node}")
            await engine.extract_memories(self.scene(user_node, parent=parent), speaker, [("Courier", "Reply")], 1000)
            counts.append(len(models.extractions))
            self.store.record_node(user_node + 1, 1, 100, user_node, None, self.courier, "Reply")
            parent = user_node + 1
        self.assertEqual(counts, [0, 1, 1])

    async def test_extraction_cadence_past_ancestor_cap(self):
        """BUG-03/D4 (fixed): extraction_every_turns counts every user-authored node on the branch, not just those inside
        the 200-node ancestors() window. On a 266-node alternating user/character branch with extraction_every_turns=3,
        the last six user turns (counts 128..133) extract exactly on counts 129 and 132."""
        models = MemoryModels()
        engine = self.engine(models, extraction_every_turns=3)
        speaker = self.characters(self.courier)
        parent, ran = None, []
        for turn in range(1, 134):
            user_node = 10_000 + 2 * turn
            self.store.record_node(user_node, 1, 100, parent, USER, None, f"turn {turn}", created_at=1_700_000_000.0 + turn)
            if turn >= 128:
                before = len(models.extractions)
                await engine.extract_memories(self.scene(user_node, parent=parent), speaker, [("Courier", "Reply")], 10_002)
                if len(models.extractions) > before:
                    ran.append(turn)
            self.store.record_node(user_node + 1, 1, 100, user_node, None, self.courier, "Reply",
                                   created_at=1_700_000_000.5 + turn)
            parent = user_node + 1
        self.assertEqual(len(self.store.ancestors(parent, limit=10_000)), 266)
        self.assertEqual(ran, [129, 132])

    def test_user_ancestor_count_is_guild_scoped(self):
        """BUG-03/D4 (guild isolation): Store.count_user_ancestors counts only nodes of the given guild_id."""
        self.store.record_node(2000, 2, 200, None, USER, None, "other guild")
        self.store.record_node(2001, 1, 100, 2000, USER, None, "child")
        self.store.record_node(2002, 1, 100, 2001, None, self.courier, "reply")
        self.store.record_node(2003, 1, 100, 2002, USER, None, "again")
        self.store.execute("UPDATE nodes SET parent_id=2000 WHERE message_id=2001")  # forced cross-guild link
        self.assertEqual(self.store.count_user_ancestors(1, 2003), 2)
        self.assertEqual(self.store.count_user_ancestors(2, 2003), 0)

    async def test_extraction_text_capped_keeping_end(self):
        """BUG-03/D4 (fixed): extraction input over memory_input_tokens is truncated from the start, keeping the end."""
        models = MemoryModels()
        self.chain(1)
        scene = self.scene(1000, text="ALPHA-START " + "filler " * 100)
        await self.engine(models, memory_input_tokens=40).extract_memories(
            scene, self.characters(self.courier), [("Courier", "middle"), ("Courier", "OMEGA-END")], 1000)
        self.assertEqual(len(models.extractions), 1)
        text = models.extractions[0]["text"]
        self.assertIn("OMEGA-END", text)
        self.assertNotIn("ALPHA-START", text)
        self.assertNotIn("filler " * 60, text)

    async def test_extraction_truncation_keeps_user_line(self):
        """BUG-03/D4 (fixed): when extraction input exceeds memory_input_tokens the user line is always kept and only the
        character lines are trimmed from the start, so the payload has the user's message and the end of the last
        character line but not the start of the first."""
        models = MemoryModels()
        self.chain(1)
        scene = self.scene(1000, text="USER-MARK hi")
        lines = [("Courier", "CHAR-START " + "pad " * 60), ("Courier", "tail words OMEGA-END")]
        await self.engine(models, memory_input_tokens=40).extract_memories(scene, self.characters(self.courier), lines, 1000)
        self.assertEqual(len(models.extractions), 1)
        text = models.extractions[0]["text"]
        self.assertIn("USER-MARK", text)
        self.assertIn("OMEGA-END", text)
        self.assertNotIn("CHAR-START", text)


class MemoryLimitConfigTests(unittest.TestCase):
    KEYS = {"memory_input_tokens": 6000, "memory_output_tokens": 550,
            "summary_every_messages": 1, "extraction_every_turns": 1}

    def test_memory_limit_defaults(self):
        """BUG-03/D4 (fixed): load_settings fills the four memory limits with defaults that keep today's behavior."""
        limits = settings_for().limits
        self.assertEqual({key: limits.get(key) for key in self.KEYS}, self.KEYS)

    def test_memory_limits_read_from_yaml(self):
        """BUG-03/D4 (fixed): limits.memory_* / *_every_* are read from config YAML."""
        wanted = {"memory_input_tokens": 3000, "memory_output_tokens": 400,
                  "summary_every_messages": 4, "extraction_every_turns": 3}
        extra = "limits:\n" + "".join(f"  {key}: {value}\n" for key, value in wanted.items())
        limits = settings_for(extra=extra).limits
        self.assertEqual({key: limits.get(key) for key in wanted}, wanted)

    def test_memory_limits_must_be_positive(self):
        """BUG-03/D4 (fixed): zero or negative memory limits raise ValueError at load, like the other limits."""
        for key in self.KEYS:
            for value in (0, -1):
                with self.assertRaises(ValueError, msg=f"{key}={value}"):
                    settings_for(extra=f"limits:\n  {key}: {value}\n")

    def test_cadence_limits_accept_100(self):
        """BUG-03/D4: summary_every_messages and extraction_every_turns of 100 (the ancestors window) load fine."""
        for key in ("summary_every_messages", "extraction_every_turns"):
            self.assertEqual(settings_for(extra=f"limits:\n  {key}: 100\n").limits[key], 100)

    def test_cadence_limits_over_100_rejected(self):
        """BUG-03/D4 (fixed): summary_every_messages / extraction_every_turns above 100 could never trigger inside the
        ancestors window, so load_settings rejects 101 with ValueError."""
        for key in ("summary_every_messages", "extraction_every_turns"):
            with self.assertRaises(ValueError, msg=f"{key}=101"):
                settings_for(extra=f"limits:\n  {key}: 101\n")


if __name__ == "__main__":
    unittest.main()
