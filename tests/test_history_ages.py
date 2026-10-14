"""MNT-85: message ages in history.

Written before the implementation. Assumed seams (proposed, pinned here):

  ``llmcord_core.prompts.age_label(now, then) -> str``: ``''`` under 90 s (or a negative age); whole minutes (floor) from 90 s to 4 min 59 s
  (``[1 min ago]``..``[4 min ago]``); then minutes floored to a multiple of 5 up to 59 min (``[5 min ago]``..``[55 min ago]``); whole hours
  up to 47 h (``[1 h ago]``..``[47 h ago]``); then whole days (``[2 d ago]``).
  ``Engine.history_messages(nodes, target_character_id, now=None)``: with ``now``, every user-role line (human nodes, observed ``context_json``
  parts by snowflake time, other characters' lines) is prefixed with ``label + ' '``; the target character's own lines never are.
  Without ``now`` the output is unchanged. ``_history`` builds summary text without ages. ``now`` is fixed once per turn on the scene
  (wall clock, or the replied-to node's ``created_at`` for a rewind), like ``happenings_now`` (MNT-84).
"""
import unittest
from unittest import mock

from helpers import CoreFakeModels, core_settings

from llmcord_core.engine import Engine, SceneContext
from llmcord_core.identity import user_line
from llmcord_core.store import Store

G, CH = 1, 100
NOW = 1_800_000_000.0
DISCORD_EPOCH_MS = 1420070400000


def snowflake(unix_seconds):
    return (int(unix_seconds * 1000) - DISCORD_EPOCH_MS) << 22


def age_label():
    from llmcord_core import prompts  # late import so the other tests report their own failure
    return prompts.age_label


class AgeLabelTests(unittest.TestCase):
    def check(self, seconds, expected):
        self.assertEqual(age_label()(NOW, NOW - seconds), expected, f"age {seconds}s")

    def test_under_ninety_seconds_is_empty(self):
        for seconds in (0, 1, 30, 89, 89.9):
            self.check(seconds, '')

    def test_negative_age_or_clock_skew_is_empty(self):
        for seconds in (-1, -3600):
            self.check(seconds, '')

    def test_minutes_floor_under_five(self):
        for seconds, expected in ((90, '[1 min ago]'), (119, '[1 min ago]'), (120, '[2 min ago]'), (180, '[3 min ago]'), (299, '[4 min ago]')):
            self.check(seconds, expected)

    def test_minutes_floor_to_multiples_of_five_up_to_fifty_nine(self):
        for seconds, expected in ((300, '[5 min ago]'), (599, '[5 min ago]'), (600, '[10 min ago]'), (1799, '[25 min ago]'),
                                  (3299, '[50 min ago]'), (3300, '[55 min ago]'), (3599, '[55 min ago]')):
            self.check(seconds, expected)

    def test_whole_hours_up_to_forty_seven(self):
        for seconds, expected in ((3600, '[1 h ago]'), (7199, '[1 h ago]'), (7200, '[2 h ago]'), (86400, '[24 h ago]'), (47 * 3600 + 3599, '[47 h ago]')):
            self.check(seconds, expected)

    def test_whole_days_from_forty_eight_hours(self):
        for seconds, expected in ((48 * 3600, '[2 d ago]'), (86400 * 3 - 1, '[2 d ago]'), (86400 * 3, '[3 d ago]'), (86400 * 40, '[40 d ago]')):
            self.check(seconds, expected)


class HistoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)
        self.world = self.store.create_space(G, "Harbor", "world")
        self.store.bind_channel(G, CH, self.world)
        self.ann = self.store.add_character(G, self.world, "Ann", {"name": "Ann"}, None, [])
        self.bea = self.store.add_character(G, self.world, "Bea", {"name": "Bea"}, None, [])
        self.store.set_cast(CH, None, [self.ann, self.bea])
        self.models = CoreFakeModels()
        self.engine = Engine(self.store, self.models, core_settings())

    def human(self, mid, text, at, parent=None, context=None):
        self.store.record_node(mid, G, CH, parent, 7, None, text, context, created_at=at, author_label="Mina")
        return mid

    def char(self, mid, cid, text, at, parent):
        self.store.record_node(mid, G, CH, parent, None, cid, text, None, created_at=at)
        return mid

    def chain(self):
        """Human 1 (10 min old) -> Ann (5 min old) -> Bea (3 min old) -> human 4 (now)."""
        self.human(1, "one", NOW - 600)
        self.char(2, self.ann, "from ann", NOW - 300, 1)
        self.char(3, self.bea, "from bea", NOW - 180, 2)
        self.human(4, "four", NOW - 10, 3)
        return self.store.ancestors(4)

    def texts(self, nodes, target, **kwargs):
        return [message.text for message in self.engine.history_messages(nodes, target, **kwargs)[0]]

    def test_now_none_is_unchanged(self):
        nodes = self.chain()
        expected = [user_line(7, "Mina", "one"), "Ann: from ann", "Bea: from bea", user_line(7, "Mina", "four")]
        self.assertEqual(self.texts(nodes, self.ann), expected)
        self.assertEqual(self.texts(nodes, self.ann, now=None), expected)

    def test_human_and_other_character_lines_are_tagged_but_the_targets_own_are_not(self):
        nodes = self.chain()
        history, ids = self.engine.history_messages(nodes, self.ann, now=NOW)
        self.assertEqual([m.text for m in history], [
            '[10 min ago] ' + user_line(7, "Mina", "one"),
            "Ann: from ann",
            '[3 min ago] Bea: from bea',
            user_line(7, "Mina", "four"),  # 10 s old: no label
        ])
        self.assertEqual([m.role for m in history], ['user', 'assistant', 'user', 'user'])
        self.assertEqual(ids, [1, 2, 3, 4])

    def test_the_other_character_sees_the_mirror_image(self):
        texts = self.texts(self.chain(), self.bea, now=NOW)
        self.assertEqual(texts[1], '[5 min ago] Ann: from ann')
        self.assertEqual(texts[2], "Bea: from bea")

    def test_observed_context_parts_are_aged_by_their_snowflake(self):
        parts = [
            {'message_id': snowflake(NOW - 7200), 'author_id': 8, 'author_label': 'Bo', 'text': 'old chatter'},
            {'message_id': snowflake(NOW - 20), 'author_id': 8, 'author_label': 'Bo', 'text': 'fresh chatter'},
        ]
        self.human(1, "hello", NOW - 60, context=parts)
        texts = self.texts(self.store.ancestors(1), self.ann, now=NOW)
        self.assertEqual(texts, ['[2 h ago] ' + user_line(8, "Bo", "old chatter"), user_line(8, "Bo", "fresh chatter"), user_line(7, "Mina", "hello")])

    def test_context_part_duplicating_a_node_is_still_listed_once(self):
        self.human(5, "dup", NOW - 3600)
        parts = [{'message_id': 5, 'author_id': 7, 'author_label': 'Mina', 'text': 'dup'}]
        self.human(6, "next", NOW - 100, parent=5, context=parts)
        texts = self.texts(self.store.ancestors(6), self.ann, now=NOW)
        self.assertEqual(texts, ['[1 h ago] ' + user_line(7, "Mina", "dup"), '[1 min ago] ' + user_line(7, "Mina", "next")])

    async def test_summary_payload_and_saved_summary_carry_no_ages(self):
        previous = None
        for number in range(1, 31):
            previous = self.human(number, f"line {number}", NOW - 4 * 3600 + number * 60, parent=previous)
        sent = []
        original = self.models.text

        async def text(role, system, messages, max_tokens=None):
            sent.append(system + "\n" + "\n".join(getattr(m, 'text', str(m)) for m in messages))
            return await original(role, system, messages, max_tokens)

        self.models.text = text
        scene = SceneContext(G, CH, None, self.world, 7, 30, "line 30", None, [], [], user_label="Mina")
        with mock.patch('llmcord_core.engine.time.time', return_value=NOW):
            summary, history, _, _ = await self.engine._history(scene, self.ann)
        self.assertTrue(sent, "summarisation did not run")
        for payload in sent:
            self.assertIn("line 1", payload)
            self.assertNotIn(" ago]", payload)
        self.assertNotIn(" ago]", summary)
        self.assertNotIn(" ago]", self.store.summary(30) or '')
        self.assertEqual(len(history), 12)
        self.assertTrue(all(m.text.startswith('[3 h ago] ') for m in history[:-1]), [m.text for m in history])

    async def prepare(self, scene, cid):
        request, _ = await self.engine.prepare_dialogue(scene, self.store.character_by_id(cid), [])
        return "\n".join(m.text for m in request.messages)

    async def test_two_speakers_share_one_now_even_when_the_wall_clock_moves(self):
        self.human(1, "ancient", NOW - 7200)
        self.human(2, "hi all", NOW - 5, parent=1)
        scene = SceneContext(G, CH, None, self.world, 7, 2, "hi all", None, [], [], user_label="Mina")
        with mock.patch('llmcord_core.engine.time.time', return_value=NOW):
            first = await self.prepare(scene, self.ann)
        with mock.patch('llmcord_core.engine.time.time', return_value=NOW + 3 * 3600):
            second = await self.prepare(scene, self.bea)
        for text in (first, second):
            self.assertIn('[2 h ago] ' + user_line(7, "Mina", "ancient"), text)
            self.assertNotIn('[5 h ago]', text)
        # A fresh turn (fresh scene) later in the day uses the later clock.
        fresh = SceneContext(G, CH, None, self.world, 7, 2, "hi all", None, [], [], user_label="Mina")
        with mock.patch('llmcord_core.engine.time.time', return_value=NOW + 3 * 3600):
            later = await self.prepare(fresh, self.ann)
        self.assertIn('[5 h ago] ' + user_line(7, "Mina", "ancient"), later)

    async def test_own_lines_stay_untagged_in_the_built_prompt(self):
        self.human(1, "question", NOW - 3600)
        self.char(2, self.ann, "OWN-ANSWER", NOW - 3000, 1)
        self.human(3, "follow up", NOW - 2, parent=2)
        scene = SceneContext(G, CH, None, self.world, 7, 3, "follow up", None, [], [], user_label="Mina")
        with mock.patch('llmcord_core.engine.time.time', return_value=NOW):
            text = await self.prepare(scene, self.ann)
        self.assertIn("Ann: OWN-ANSWER", text)
        self.assertNotIn("] Ann: OWN-ANSWER", text)
        self.assertIn('[1 h ago] ' + user_line(7, "Mina", "question"), text)

    async def test_rewind_measures_ages_from_the_replied_to_node(self):
        """The branch point is the clock for a rewind, so a reply to an old message does not see ages from today's clock."""
        self.human(1, "first", NOW - 7200)
        self.char(2, self.ann, "Bea-less reply", NOW - 3600, 1)
        self.human(3, "rewind here", NOW - 1800, parent=2)
        self.human(10, "reply to node 2", NOW, parent=2)
        scene = SceneContext(G, CH, None, self.world, 7, 10, "reply to node 2", 2, [], [], user_label="Mina", rewind=True)
        with mock.patch('llmcord_core.engine.time.time', return_value=NOW + 86400):
            text = await self.prepare(scene, self.bea)
        # Parent node 2 was created 1 h after node 1: relative to it, "first" is [1 h ago], not [26 h ago].
        self.assertIn('[1 h ago] ' + user_line(7, "Mina", "first"), text)
        self.assertNotIn('[26 h ago]', text)
        self.assertNotIn("rewind here", text)
