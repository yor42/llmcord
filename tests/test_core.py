import asyncio
import base64
import json
import struct
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image, PngImagePlugin

from llmcord_core.cards import _embedded_png_card, parse_card
from llmcord_core.config import ModelProfile, Settings
from llmcord_core.engine import Engine, SceneContext
from llmcord_core.lore import lore_scopes, retrieve_lore
from llmcord_core.store import Store


class FakeModels:
    def __init__(self, speakers=None):
        self.chosen = speakers

    async def structured(self, role, system, messages, schema_name, schema):
        if schema_name == "choose_speakers":
            return {"speakers": self.chosen or []}
        return {"shared_facts": [], "personal_facts": [], "encounter_facts": []}

    async def text(self, role, system, messages, max_tokens=None):
        return "Earlier scene summary"


def settings():
    profile = ModelProfile("compatible", "test", 16000, True, base_url="http://localhost/v1")
    return Settings("test", None, ":memory:", 90, {"test": profile}, "test", "test", "test",
        {"max_input_tokens": 12000, "max_output_tokens": 700, "max_images": 3,
         "max_attachment_bytes": 8388608, "max_speakers": 3, "recent_messages": 12,
         "recent_window_seconds": 600, "ambient_cooldown_seconds": 120})


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.a = self.store.create_space(1, "World A", "world")
        self.b = self.store.create_space(1, "World B", "world")
        self.hub = self.store.create_space(1, "Hub", "hub")
        self.store.link_world(1, self.hub, self.a)
        self.store.link_world(1, self.hub, self.b)
        self.store.bind_channel(1, 100, self.a)
        self.store.bind_channel(1, 200, self.b)
        self.store.bind_channel(1, 300, self.hub)
        self.alice = self.store.add_character(1, self.a, "Alice", {"name": "Alice", "description": "World A resident"}, None, [])
        self.bob = self.store.add_character(1, self.b, "Bob", {"name": "Bob", "description": "World B resident"}, None, [])

    def tearDown(self):
        self.store.close()

    def test_hub_guest_sees_own_world_and_hub_but_not_other_world(self):
        self.store.add_lore(1, "space", self.a, "A's moon is red", ["moon"])
        self.store.add_lore(1, "space", self.b, "B's moon is blue", ["moon"])
        self.store.add_lore(1, "space", self.hub, "The hub is between worlds", ["hub"])
        self.store.add_lore(1, "channel", 300, "The cafe has tea", ["cafe"])
        scopes = lore_scopes(self.store, self.alice, self.hub, 300)
        found = retrieve_lore(self.store, 1, scopes, "moon hub cafe")
        text = "\n".join(item.content for item in found)
        self.assertIn("red", text)
        self.assertIn("between worlds", text)
        self.assertIn("tea", text)
        self.assertNotIn("blue", text)
        self.assertEqual({row["name"] for row in self.store.eligible_characters(1, self.a)}, {"Alice"})
        self.assertEqual({row["name"] for row in self.store.eligible_characters(1, self.hub)}, {"Alice", "Bob"})

    def test_lore_constants_order_budget_and_local_promotion(self):
        self.store.add_lore(1, "channel", 100, "always here", [], True, 50)
        self.store.add_lore(1, "channel", 100, "first", ["test"], False, 1)
        self.store.add_lore(1, "channel", 100, "second", ["test"], False, 2)
        found = retrieve_lore(self.store, 1, [("channel", 100)], "test", budget_tokens=100)
        self.assertEqual([item.content for item in found], ["always here", "first", "second"])
        self.assertEqual(len(retrieve_lore(self.store, 1, [("channel", 100)], "test", budget_tokens=1)), 0)
        source = self.store.add_lore(1, "channel", 100, "The bell rings twice", ["bell"])
        promoted = self.store.promote_lore(1, source, "space", self.a)
        self.assertEqual(self.store.lore_row(1, promoted)["promoted_from"], source)
        self.assertFalse(any(row["content"] == "The bell rings twice" for row in self.store.list_lore(1, "space", self.b)))

    def test_candidates_need_two_distinct_scenes_and_stay_local(self):
        self.store.add_candidate(1, "channel", 100, "The bell rings twice", 1000, 1001)
        self.store.add_candidate(1, "channel", 100, "The bell rings twice", 1000, 1002)
        self.assertFalse(self.store.list_lore(1, "channel", 100))
        self.store.add_candidate(1, "channel", 100, "The bell rings twice", 2000, 2001)
        self.assertEqual(len(self.store.list_lore(1, "channel", 100)), 1)
        self.assertFalse(self.store.list_lore(1, "space", self.a))
        self.assertIn("The bell rings twice", [item.content for item in
            retrieve_lore(self.store, 1, [("channel", 100)], "unrelated conversation")])

    def test_branch_rewind_excludes_later_siblings(self):
        self.store.record_node(1, 1, 100, None, 9, None, "begin")
        self.store.record_node(2, 1, 100, 1, None, self.alice, "first reply")
        self.store.record_node(3, 1, 100, 2, 9, None, "later event")
        self.store.record_node(4, 1, 100, 3, None, self.alice, "later reply")
        self.store.record_node(5, 1, 100, 2, 9, None, "alternate event")
        self.assertEqual([row["message_id"] for row in self.store.ancestors(5)], [1, 2, 5])
        self.assertEqual(self.store.node(5)["root_id"], 1)
        self.assertEqual(self.store.node(4)["root_id"], 1)

    def test_thread_cast_and_sqlite_restart_recovery(self):
        self.store.set_cast(100, None, [self.alice])
        self.store.set_cast(101, 100, [self.alice])
        self.assertEqual(self.store.get_cast(101, 100), [self.alice])
        self.assertEqual(self.store.get_cast(100), [self.alice])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            path = Path(temporary) / "bot.sqlite3"
            first = Store(path)
            world = first.create_space(3, "Persistent", "world")
            first.bind_channel(3, 400, world)
            first.record_node(500, 3, 400, None, 9, None, "saved line")
            first.save_webhook_id(400, 7, 9876)
            first.close()
            second = Store(path)
            try:
                self.assertEqual(second.binding(400)["space_id"], world)
                self.assertEqual(second.node(500)["content"], "saved line")
                self.assertEqual(second.webhook_id(400, 7), 9876)
            finally:
                second.close()

    def test_prompt_rewind_excludes_later_memory_and_other_world(self):
        self.store.add_lore(1, "space", self.a, "A has a red moon", constant=True)
        self.store.add_lore(1, "space", self.b, "B's secret moon is blue", constant=True)
        self.store.add_lore(1, "space", self.hub, "The hub is a cafe", constant=True)
        self.store.record_node(1000, 1, 300, None, 9, None, "hello", created_at=10)
        self.store.record_node(1001, 1, 300, 1000, None, self.alice, "welcome", created_at=20)
        self.store.record_node(1002, 1, 300, 1001, 9, None, "later scene", created_at=30)
        self.store.add_lore(1, "channel", 300, "Later hub event", constant=True, source_id=1002)
        self.store.set_consent(1, 9, True)
        self.store.add_personal(1, 9, self.alice, "User likes later event", 1002)
        self.store.add_encounter(1, self.alice, self.hub, "Saw later event", 1002)
        scene = SceneContext(1, 300, None, self.hub, 9, 1003, "rewind", 1001, [], [])
        engine = Engine(self.store, FakeModels(), settings())
        engine.record_user(scene)
        system, messages, sources = asyncio.run(engine.prompt_for(scene,
            self.store.character_by_id(self.alice), []))
        self.assertIn("red moon", system)
        self.assertIn("hub is a cafe", system)
        self.assertNotIn("blue", system)
        self.assertNotIn("Later hub event", system)
        self.assertNotIn("User likes later event", system)
        self.assertNotIn("Saw later event", system)
        self.assertEqual(sources["messages"], [1000, 1001, 1003])
        self.assertFalse(any("later scene" in message.text for message in messages))

    def test_personal_consent_and_character_space_memory(self):
        self.store.add_personal(1, 9, self.alice, "likes tea", 1)
        self.assertEqual(self.store.personal(1, 9), [])
        self.store.set_consent(1, 9, True)
        self.store.add_personal(1, 9, self.alice, "likes tea", 1)
        self.assertEqual(len(self.store.personal(1, 9, self.alice)), 1)
        self.assertEqual(self.store.personal(1, 9, self.bob), [])
        self.store.add_encounter(1, self.alice, self.a, "met in A", 2)
        self.store.add_encounter(1, self.alice, self.hub, "met in hub", 3)
        self.assertEqual(len(self.store.encounters(1, self.alice, self.a)), 1)
        self.store.set_consent(1, 9, False)
        self.assertEqual(self.store.personal(1, 9), [])

    def test_history_retention_keeps_durable_lore(self):
        self.store.record_node(7000, 1, 100, None, 9, None, "old raw chat", created_at=100)
        self.store.save_summary(7000, "old summary")
        self.store.save_trace(7000, {"messages": [7000]})
        lore_id = self.store.add_lore(1, "channel", 100, "Durable bell", constant=True, source_id=7000)
        removed = self.store.expire_history(90, now=100 + 91 * 86400)
        self.assertEqual(removed, 1)
        self.assertIsNone(self.store.node(7000))
        self.assertIsNone(self.store.summary(7000))
        self.assertIsNone(self.store.trace(7000))
        self.assertEqual(self.store.lore_row(1, lore_id)["content"], "Durable bell")

    def test_director_forced_guest_and_ambient_gate(self):
        self.store.set_cast(300, None, [self.alice])
        engine = Engine(self.store, FakeModels([self.bob, self.alice]), settings())
        scene = SceneContext(1, 300, None, self.hub, 9, 123, "Bob, join this", None, [], [], forced_character_id=self.bob)
        chosen = asyncio.run(engine.speakers(scene))
        self.assertEqual([row["name"] for row in chosen], ["Bob", "Alice"])
        ambient = SceneContext(1, 300, None, self.hub, 9, 124, "random weather", None, [], [], ambient=True)
        self.assertEqual(asyncio.run(engine.speakers(ambient)), [])

    def test_card_json_and_png_chunk_precedence(self):
        card = {"spec": "chara_card_v3", "data": {"name": "Alice", "character_book": {"entries": [
            {"content": "Always", "constant": True, "keys": []}
        ]}}}
        parsed = parse_card("alice.json", json.dumps(card).encode())
        self.assertEqual(parsed.name, "Alice")
        self.assertTrue(parsed.entries[0]["constant"])
        old = {"spec": "chara_card_v2", "data": {"name": "Old"}}
        def chunk(key, value):
            body = key.encode() + b"\0" + base64.b64encode(json.dumps(value).encode())
            return struct.pack(">I", len(body)) + b"tEXt" + body + b"\0\0\0\0"
        png = b"\x89PNG\r\n\x1a\n" + chunk("chara", old) + chunk("ccv3", card)
        self.assertEqual(_embedded_png_card(png)["data"]["name"], "Alice")
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("ccv3", base64.b64encode(json.dumps(card).encode()).decode())
        buffer = BytesIO()
        Image.new("RGB", (24, 24), "red").save(buffer, format="PNG", pnginfo=metadata)
        imported = parse_card("alice.png", buffer.getvalue())
        self.assertEqual(imported.name, "Alice")
        self.assertTrue(imported.avatar.startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
