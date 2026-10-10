"""FEAT-23 part B: /favorites replies, director options for favorites (lean / step in), ambient gate and threshold."""
import asyncio
import json
import re
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from helpers import FakeInteraction, FakeModels, autocomplete, core_settings, invoke, make_settings

from llmcord_core.discord_bot import SkitBot
from llmcord_core.engine import Engine, SceneContext
from llmcord_core.store import Store


class CaptureModels(FakeModels):
    def __init__(self, speakers=None):
        super().__init__(speakers)
        self.payloads = []

    async def structured(self, role, system, messages, schema_name, schema):
        if schema_name == 'choose_speakers':
            text = system + '\n'.join(getattr(m, 'text', str(m)) for m in messages)
            match = re.search(r'\{"cast".*\}', text)
            self.payloads.append(json.loads(match.group(0)))
        return await super().structured(role, system, messages, schema_name, schema)


class CommandBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.world = self.store.create_space(1, 'Harbor', 'world')
        self.other = self.store.create_space(1, 'Elsewhere', 'world')
        self.store.bind_channel(1, 100, self.world)
        self.alice = self.store.add_character(1, self.world, 'Alice', {'name': 'Alice'}, None, [])
        self.bob = self.store.add_character(1, self.world, 'Bob', {'name': 'Bob'}, None, [])
        self.zed = self.store.add_character(1, self.other, 'Zed', {'name': 'Zed'}, None, [])

    def tearDown(self):
        self.store.close()

    async def run_cmd(self, path, *args, channel_id=100, **kwargs):
        interaction = FakeInteraction(channel_id=channel_id)
        await invoke(self.bot, path, interaction, *args, **kwargs)
        return interaction


class FavoritesCommandTests(CommandBase):
    async def test_add_remove_list_mode_clear_texts(self):
        self.assertEqual((await self.run_cmd('favorites list')).replies, ['You have no favorites yet. Add one with /favorites add.'])
        self.assertEqual((await self.run_cmd('favorites add', 'alice')).replies, ['Added Alice to your favorites.'])
        await self.run_cmd('favorites add', 'Zed')
        listing = await self.run_cmd('favorites list')
        self.assertEqual(listing.replies, ['1. Alice\n2. Zed (not available here)\nMode: lean — your favorites in the cast are more likely to answer you.'])
        self.assertTrue(listing.response.sent[0][1]['ephemeral'])
        self.assertEqual(listing.response.sent[0][1]['allowed_mentions'].everyone, False)
        self.assertEqual((await self.run_cmd('favorites mode', 'step_in')).replies, ['Favorites mode: step in.'])
        self.assertIn('Mode: step in — your favorites can also answer you when they are not in the cast.', (await self.run_cmd('favorites list')).replies[0])
        self.assertEqual((await self.run_cmd('favorites mode', 'lean')).replies, ['Favorites mode: lean.'])
        self.assertEqual((await self.run_cmd('favorites remove', 'ALICE')).replies, ['Removed Alice from your favorites.'])
        self.assertEqual((await self.run_cmd('favorites remove', 'Alice')).replies, ['Alice is not one of your favorites.'])
        self.assertEqual((await self.run_cmd('favorites clear')).replies, ['Cleared 1 favorite.'])
        await self.run_cmd('favorites add', 'Alice')
        await self.run_cmd('favorites add', 'Bob')
        self.assertEqual((await self.run_cmd('favorites clear')).replies, ['Cleared 2 favorites.'])

    async def test_unbound_channel_has_no_marks(self):
        await self.run_cmd('favorites add', 'Zed')
        self.assertNotIn('not available', (await self.run_cmd('favorites list', channel_id=999)).replies[0])

    async def test_refusals_show_store_message(self):
        await self.run_cmd('favorites add', 'Alice')
        self.assertEqual((await self.run_cmd('favorites add', 'Alice')).replies, ['That character is already one of your favorites.'])
        self.assertEqual((await self.run_cmd('favorites add', 'Nobody')).replies, ['No character named Nobody.'])
        self.store.set_cast_limits(1, 5, 1)
        self.assertEqual((await self.run_cmd('favorites add', 'Bob')).replies, ['You can have at most 1 favorites. Remove one first.'])

    async def test_autocomplete_scopes(self):
        self.store.archive_character(1, self.bob, True)
        await self.run_cmd('favorites add', 'Alice')
        add = await autocomplete(self.bot, 'favorites add', 'character', FakeInteraction(), '')
        self.assertEqual({c.value for c in add}, {'Alice', 'Zed'})
        remove = await autocomplete(self.bot, 'favorites remove', 'character', FakeInteraction(), '')
        self.assertEqual([c.value for c in remove], ['Alice'])
        other_member = await autocomplete(self.bot, 'favorites remove', 'character', FakeInteraction(user_id=10), '')
        self.assertEqual(other_member, [])


class ArchivedFavoritesCommandTests(CommandBase):
    async def test_add_list_autocomplete_follow_the_switch(self):
        self.store.archive_character(1, self.bob, True)
        self.assertEqual((await self.run_cmd('favorites add', 'Bob')).replies, ['No character named Bob.'])
        self.store.set_archived_favorites(1, True)
        add = await autocomplete(self.bot, 'favorites add', 'character', FakeInteraction(), '')
        self.assertEqual({c.value for c in add}, {'Alice', 'Bob', 'Zed'})
        self.assertEqual((await self.run_cmd('favorites add', 'Bob')).replies, ['Added Bob to your favorites.'])
        await self.run_cmd('favorites add', 'Alice')
        self.assertEqual((await self.run_cmd('favorites list')).replies[0].splitlines()[:2], ['1. Bob (archived)', '2. Alice'])
        await self.run_cmd('favorites add', 'Zed')
        self.store.archive_character(1, self.zed, True)
        self.assertIn('3. Zed (archived) (not available here)', (await self.run_cmd('favorites list')).replies[0])
        self.store.set_archived_favorites(1, False)
        self.assertEqual((await self.run_cmd('favorites list')).replies[0].splitlines()[0], '1. Alice')
        add = await autocomplete(self.bot, 'favorites add', 'character', FakeInteraction(), '')
        self.assertEqual({c.value for c in add}, {'Alice'})

    async def test_summon_of_an_archived_favorite_is_refused(self):
        self.store.archive_character(1, self.bob, True)
        self.store.set_archived_favorites(1, True)
        await self.run_cmd('favorites add', 'Bob')
        interaction = await self.run_cmd('summon', 'Bob', 'hello')
        self.assertEqual(interaction.replies, ['No character named Bob available here.'])


class DirectorBase(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.a = self.store.create_space(1, 'A', 'world')
        self.b = self.store.create_space(1, 'B', 'world')
        self.hub = self.store.create_space(1, 'Hub', 'hub')
        self.store.link_world(1, self.hub, self.a)
        self.store.bind_channel(1, 100, self.a)
        self.alice = self.store.add_character(1, self.a, 'Alice', {'name': 'Alice'}, None, [])
        self.cara = self.store.add_character(1, self.a, 'Cara', {'name': 'Cara'}, None, [])
        self.dan = self.store.add_character(1, self.a, 'Dan', {'name': 'Dan'}, None, [])
        self.bob = self.store.add_character(1, self.b, 'Bob', {'name': 'Bob'}, None, [])
        self.store.set_cast(100, None, [self.alice, self.cara])

    def tearDown(self):
        self.store.close()

    def scene(self, text='hello', ambient=False, user=9, **kw):
        return SceneContext(1, 100, None, self.a, user, 500, text, None, [], [], ambient=ambient, user_label='Sam', **kw)

    def run_speakers(self, models, scene):
        return asyncio.run(Engine(self.store, models, core_settings()).speakers(scene))


class FavoritesDirectorTests(DirectorBase):
    def test_no_favorites_payload_unchanged(self):
        models = CaptureModels([self.alice])
        self.run_speakers(models, self.scene())
        payload = models.payloads[0]
        self.assertEqual(payload['cast'], [{'id': self.alice, 'name': 'Alice'}, {'id': self.cara, 'name': 'Cara'}])
        self.assertNotIn('favorites_hint', payload)

    def test_lean_marks_cast_favorites_only(self):
        self.store.add_favorite(1, 9, self.cara)
        self.store.add_favorite(1, 9, self.dan)
        models = CaptureModels([self.dan, self.cara])
        chosen = self.run_speakers(models, self.scene())
        payload = models.payloads[0]
        self.assertEqual(payload['cast'], [{'id': self.alice, 'name': 'Alice'}, {'id': self.cara, 'name': 'Cara', 'favorite': True}])
        self.assertIn('prefers', payload['favorites_hint'])
        self.assertEqual([row['name'] for row in chosen], ['Cara'])

    def test_step_in_adds_eligible_non_cast_favorites_for_this_turn_only(self):
        self.store.add_favorite(1, 9, self.dan)
        self.store.add_favorite(1, 9, self.bob)
        self.store.set_favorites_mode(1, 9, 'step_in')
        models = CaptureModels([self.dan, self.bob])
        chosen = self.run_speakers(models, self.scene())
        self.assertEqual([o['id'] for o in models.payloads[0]['cast']], [self.alice, self.cara, self.dan])
        self.assertTrue(models.payloads[0]['cast'][2]['favorite'])
        self.assertEqual([row['name'] for row in chosen], ['Dan'])
        self.assertEqual(self.store.get_cast(100), [self.alice, self.cara])

    def test_step_in_never_reaches_other_guild_or_unlinked_world(self):
        other = self.store.create_space(2, 'Z', 'world')
        stranger = self.store.add_character(2, other, 'Stranger', {'name': 'Stranger'}, None, [])
        self.store.add_favorite(1, 9, self.bob)
        with self.assertRaises(ValueError):
            self.store.add_favorite(1, 9, stranger)
        self.store.set_favorites_mode(1, 9, 'step_in')
        models = CaptureModels([self.bob, stranger])
        chosen = self.run_speakers(models, self.scene())
        self.assertEqual([o['name'] for o in models.payloads[0]['cast']], ['Alice', 'Cara'])
        self.assertEqual([row['name'] for row in chosen], ['Alice'])

    def test_hub_step_in_reaches_linked_world_character(self):
        self.store.bind_channel(1, 300, self.hub)
        self.store.set_cast(300, None, [self.alice])
        self.store.add_favorite(1, 9, self.dan)
        self.store.set_favorites_mode(1, 9, 'step_in')
        models = CaptureModels([self.dan])
        scene = SceneContext(1, 300, None, self.hub, 9, 500, 'hi', None, [], [])
        chosen = self.run_speakers(models, scene)
        self.assertEqual([row['name'] for row in chosen], ['Dan'])

    def test_other_authors_favorites_do_not_count(self):
        self.store.add_favorite(1, 10, self.dan)
        self.store.set_favorites_mode(1, 10, 'step_in')
        models = CaptureModels([self.alice])
        self.run_speakers(models, self.scene(user=9))
        self.assertEqual([o['name'] for o in models.payloads[0]['cast']], ['Alice', 'Cara'])

    def test_ambient_gate_passes_only_with_favorite_option(self):
        models = CaptureModels([self.alice])
        self.assertEqual(self.run_speakers(models, self.scene('random weather', ambient=True)), [])
        self.assertEqual(models.payloads, [])
        self.store.add_favorite(1, 9, self.dan)
        self.assertEqual(self.run_speakers(models, self.scene('random weather', ambient=True)), [])
        self.store.add_favorite(1, 9, self.alice)
        chosen = self.run_speakers(models, self.scene('random weather', ambient=True))
        self.assertEqual([row['name'] for row in chosen], ['Alice'])

    def test_ambient_gate_step_in_favorite_outside_cast(self):
        self.store.add_favorite(1, 9, self.dan)
        self.store.set_favorites_mode(1, 9, 'step_in')
        models = CaptureModels([self.dan])
        chosen = self.run_speakers(models, self.scene('random weather', ambient=True))
        self.assertEqual([row['name'] for row in chosen], ['Dan'])

    def test_forced_and_max_speakers_kept(self):
        for ident in (self.cara, self.dan):
            self.store.add_favorite(1, 9, ident)
        self.store.set_favorites_mode(1, 9, 'step_in')
        models = CaptureModels([self.dan, self.cara, self.alice])
        scene = self.scene(forced_character_id=self.alice)
        chosen = self.run_speakers(models, scene)
        self.assertEqual([row['name'] for row in chosen], ['Alice', 'Dan', 'Cara'])

    def test_empty_cast_lean_is_silent_and_step_in_uses_favorites(self):
        self.store.set_cast(100, None, [])
        self.store.add_favorite(1, 9, self.dan)
        models = CaptureModels([self.dan])
        self.assertEqual(self.run_speakers(models, self.scene()), [])
        self.assertEqual(models.payloads, [])
        self.store.set_favorites_mode(1, 9, 'step_in')
        chosen = self.run_speakers(models, self.scene())
        self.assertEqual([row['name'] for row in chosen], ['Dan'])
        self.assertEqual([o['name'] for o in models.payloads[0]['cast']], ['Dan'])
        ambient = self.run_speakers(models, self.scene('random weather', ambient=True))
        self.assertEqual([row['name'] for row in ambient], ['Dan'])
        self.assertEqual(self.run_speakers(models, self.scene(user=10)), [])

    def test_prepare_dialogue_group_includes_stepped_in_speaker(self):
        from dataclasses import replace
        engine = Engine(self.store, FakeModels(), core_settings())
        scene = replace(self.scene(), speaker_ids=(self.dan,), preset=engine.store.active_preset(1))
        captured = {}
        real = engine.compile
        def compile_spy(scene_, purpose, values, *a, **k):
            if purpose == 'dialogue':
                captured.update(values)
            return real(scene_, purpose, values, *a, **k)
        engine.compile = compile_spy
        asyncio.run(engine.prepare_dialogue(scene, self.store.character_by_id(self.alice), []))
        self.assertEqual(captured['group'], 'Alice, Cara, Dan')

    def test_favorite_available(self):
        engine = Engine(self.store, None, core_settings())
        check = lambda user=9: engine.favorite_available(1, user, self.a, 100, None)
        self.assertFalse(check())
        self.store.add_favorite(1, 9, self.dan)
        self.assertFalse(check())
        self.store.set_favorites_mode(1, 9, 'step_in')
        self.assertTrue(check())
        self.assertFalse(check(user=10))
        self.store.set_favorites_mode(1, 9, 'lean')
        self.store.add_favorite(1, 9, self.alice)
        self.assertTrue(check())
        self.assertFalse(engine.favorite_available(1, 0, self.a, 100, None))


class AmbientThresholdTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        world = self.store.create_space(1, 'W', 'world')
        self.store.bind_channel(1, 100, world)
        self.store.set_ambient(100, True)
        self.alice = self.store.add_character(1, world, 'Alice', {'name': 'Alice'}, None, [])
        self.store.set_cast(100, None, [self.alice])
        self.scenes = []

        async def run_scene(scene, channel):
            self.scenes.append(scene)
        self.bot.run_scene = run_scene

        async def no_recent(*_a, **_k):
            return []
        import llmcord_core.discord_bot as module
        self._orig = module.recent_human_lines
        module.recent_human_lines = no_recent
        self.module = module

    def tearDown(self):
        self.module.recent_human_lines = self._orig
        self.store.close()

    async def post(self, ident):
        class Chan:
            id = 100
        message = SimpleNamespace(id=ident, guild=SimpleNamespace(id=1), channel=Chan(), author=SimpleNamespace(id=9, display_name='Sam', bot=False),
            content='hey', mentions=[], webhook_id=None, reference=None, attachments=[], created_at=datetime.now(timezone.utc), clean_content='hey')
        await self.bot.on_message(message)

    async def test_threshold_two_without_favorite(self):
        await self.post(1)
        self.assertEqual(self.store.ambient_state(100)[0], 1)
        self.assertEqual(self.scenes, [])

    async def test_threshold_two_with_lean_favorite_outside_cast(self):
        world = self.store.channel(100)['space_id']
        bob = self.store.add_character(1, world, 'Bob', {'name': 'Bob'}, None, [])
        self.store.add_favorite(1, 9, bob)
        await self.post(1)
        self.assertEqual(self.store.ambient_state(100)[0], 1)
        self.store.set_favorites_mode(1, 9, 'step_in')
        self.store.mark_ambient_response(100, now=1.0)
        await self.post(2)
        count, last = self.store.ambient_state(100)
        self.assertEqual(count, 0)
        self.assertGreater(last, 1.0)

    async def test_favorite_check_skipped_during_cooldown(self):
        self.store.add_favorite(1, 9, self.alice)
        self.store.mark_ambient_response(100)
        calls = []
        self.bot.engine.favorite_available = lambda *a: calls.append(a) or True
        await self.post(1)
        self.assertEqual(calls, [])

    async def test_threshold_one_with_favorite(self):
        self.store.add_favorite(1, 9, self.alice)
        await self.post(1)
        self.assertEqual(self.store.ambient_state(100)[0], 0)
        self.assertGreater(self.store.ambient_state(100)[1], 0)


class ArchivedDirectorTests(DirectorBase):
    def setUp(self):
        super().setUp()
        self.eve = self.store.add_character(1, self.a, 'Eve', {'name': 'Eve'}, None, [])
        self.store.set_archived_favorites(1, True)
        self.store.add_favorite(1, 9, self.eve)
        self.store.archive_character(1, self.eve, True)

    def names(self, models):
        return [o['name'] for o in models.payloads[-1]['cast']]

    def test_lean_and_step_in_both_offer_it_to_its_owner_only(self):
        for mode in ('lean', 'step_in'):
            self.store.set_favorites_mode(1, 9, mode)
            models = CaptureModels([self.eve])
            chosen = self.run_speakers(models, self.scene())
            self.assertEqual(self.names(models), ['Alice', 'Cara', 'Eve'])
            self.assertTrue(models.payloads[-1]['cast'][2]['favorite'])
            self.assertEqual([row['name'] for row in chosen], ['Eve'])
        models = CaptureModels([self.eve])
        chosen = self.run_speakers(models, self.scene(user=10))
        self.assertEqual(self.names(models), ['Alice', 'Cara'])
        self.assertEqual([row['name'] for row in chosen], ['Alice'])
        self.assertEqual(self.store.get_cast(100), [self.alice, self.cara])

    def test_switch_off_hides_it(self):
        self.store.set_archived_favorites(1, False)
        models = CaptureModels([self.eve])
        self.run_speakers(models, self.scene())
        self.assertEqual(self.names(models), ['Alice', 'Cara'])

    def test_unlinked_world_and_other_guild_never(self):
        far = self.store.add_character(1, self.b, 'Far', {'name': 'Far'}, None, [])
        self.store.add_favorite(1, 9, far)
        self.store.archive_character(1, far, True)
        other = self.store.create_space(2, 'Z', 'world')
        stranger = self.store.add_character(2, other, 'Stranger', {'name': 'Stranger'}, None, [])
        self.store.archive_character(2, stranger, True)
        self.store.set_archived_favorites(2, True)
        with self.assertRaises(ValueError):
            self.store.add_favorite(1, 9, stranger)
        models = CaptureModels([far, stranger, self.eve])
        self.run_speakers(models, self.scene())
        self.assertEqual(self.names(models), ['Alice', 'Cara', 'Eve'])

    def test_hub_reaches_linked_world_archived_favorite(self):
        self.store.bind_channel(1, 300, self.hub)
        models = CaptureModels([self.eve])
        scene = SceneContext(1, 300, None, self.hub, 9, 500, 'hi', None, [], [])
        self.assertEqual([r['name'] for r in self.run_speakers(models, scene)], ['Eve'])

    def test_empty_cast_lean_still_uses_it_and_ambient_gate_passes(self):
        self.store.set_cast(100, None, [])
        models = CaptureModels([self.eve])
        self.assertEqual([r['name'] for r in self.run_speakers(models, self.scene())], ['Eve'])
        self.assertEqual([r['name'] for r in self.run_speakers(models, self.scene('random weather', ambient=True))], ['Eve'])
        self.assertEqual(self.run_speakers(models, self.scene(user=10)), [])

    def test_forced_archived_character_is_refused_and_not_in_a_cast(self):
        with self.assertRaisesRegex(ValueError, 'That character cannot join this world or hub.'):
            self.run_speakers(CaptureModels([self.eve]), self.scene(forced_character_id=self.eve))
        with self.assertRaises(ValueError):
            self.store.set_cast(100, None, [self.eve])

    def test_extras_count_within_max_favorites(self):
        self.store.set_cast_limits(1, 5, 1)
        self.store.set_favorites_mode(1, 9, 'step_in')
        models = CaptureModels([self.eve])
        self.store.remove_favorite(1, 9, self.eve)
        self.store.set_cast_limits(1, 5, 5)
        self.store.add_favorite(1, 9, self.dan)
        self.store.add_favorite(1, 9, self.eve)
        self.store.set_cast_limits(1, 5, 1)
        self.run_speakers(models, self.scene())
        self.assertEqual(self.names(models), ['Alice', 'Cara', 'Dan'])

    def test_prepare_dialogue_group_includes_archived_speaker(self):
        from dataclasses import replace
        engine = Engine(self.store, FakeModels(), core_settings())
        captured = {}
        real = engine.compile
        def compile_spy(scene_, purpose, values, *a, **k):
            if purpose == 'dialogue':
                captured.update(values)
            return real(scene_, purpose, values, *a, **k)
        engine.compile = compile_spy
        scene = replace(self.scene(), speaker_ids=(self.eve,), preset=engine.store.active_preset(1))
        asyncio.run(engine.prepare_dialogue(scene, self.store.character_by_id(self.alice), []))
        self.assertEqual(captured['group'], 'Alice, Cara, Eve')
        captured.clear()
        asyncio.run(engine.prepare_dialogue(replace(scene, user_id=10), self.store.character_by_id(self.alice), []))
        self.assertEqual(captured['group'], 'Alice, Cara')

    def test_favorite_available_counts_it_for_its_member_in_both_modes(self):
        engine = Engine(self.store, None, core_settings())
        for mode in ('lean', 'step_in'):
            self.store.set_favorites_mode(1, 9, mode)
            self.assertTrue(engine.favorite_available(1, 9, self.a, 100, None))
            self.assertFalse(engine.favorite_available(1, 10, self.a, 100, None))
        self.store.set_archived_favorites(1, False)
        self.store.set_favorites_mode(1, 9, 'lean')
        self.assertFalse(engine.favorite_available(1, 9, self.a, 100, None))


if __name__ == '__main__':
    unittest.main()
