"""FEAT-24: per-hub tone (in character | off duty): store, dialogue prompt line, /admin space tone, /space list."""
import unittest

from helpers import CoreFakeModels as FakeModels, FakeInteraction, autocomplete, core_settings, invoke, make_settings

from llmcord_core.admin_store import ConflictError
from llmcord_core.discord_bot import SkitBot
from llmcord_core.engine import Engine, SceneContext
from llmcord_core.store import Store

LINE = ("This hub is an off-duty lounge shared with characters from other worlds. Keep your personality and voice, "
        "but you are off duty here: chat casually, and don't push your world's plot, quests or conflicts.")
REFUSAL = "Only hubs have a tone. Choose a hub."


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.hub = self.store.create_space(1, 'Lobby', 'hub')
        self.world = self.store.create_space(1, 'Harbor', 'world')

    def tearDown(self):
        self.store.close()

    def test_default_is_in_character(self):
        self.assertEqual(self.store.hub_tone(1, self.hub), 'in_character')
        self.assertEqual(self.store.space_by_id(self.hub)['hub_tone'], 'in_character')

    def test_set_and_get(self):
        self.assertEqual(self.store.set_hub_tone(1, self.hub, 'off_duty', 'in_character'), 'off_duty')
        self.assertEqual(self.store.hub_tone(1, self.hub), 'off_duty')
        self.assertEqual(self.store.list_spaces(1)[0]['hub_tone'], 'off_duty')
        self.store.set_hub_tone(1, self.hub, 'in_character')
        self.assertEqual(self.store.hub_tone(1, self.hub), 'in_character')

    def test_world_or_missing_refused(self):
        for ident in (self.world, 9999):
            with self.subTest(ident=ident), self.assertRaisesRegex(ValueError, REFUSAL):
                self.store.set_hub_tone(1, ident, 'off_duty')
        self.assertEqual(self.store.hub_tone(1, self.world), 'in_character')

    def test_other_guild_refused(self):
        with self.assertRaisesRegex(ValueError, REFUSAL):
            self.store.set_hub_tone(2, self.hub, 'off_duty')
        self.assertEqual(self.store.hub_tone(1, self.hub), 'in_character')
        self.store.set_hub_tone(1, self.hub, 'off_duty')
        self.assertEqual(self.store.hub_tone(2, self.hub), 'in_character')

    def test_bad_tone_refused(self):
        for tone in ('loud', '', None):
            with self.subTest(tone=tone), self.assertRaisesRegex(ValueError, 'The tone must be in character or off duty.'):
                self.store.set_hub_tone(1, self.hub, tone)

    def test_stale_expected_conflicts(self):
        self.store.set_hub_tone(1, self.hub, 'off_duty')
        with self.assertRaisesRegex(ConflictError, 'The hub tone was changed elsewhere. Reload the page and try again.'):
            self.store.set_hub_tone(1, self.hub, 'in_character', 'in_character')
        self.assertEqual(self.store.hub_tone(1, self.hub), 'off_duty')


class PromptTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = Store()
        self.hub = self.store.create_space(1, 'Lobby', 'hub')
        self.world = self.store.create_space(1, 'Harbor', 'world')
        self.store.link_world(1, self.hub, self.world)
        self.store.bind_channel(1, 100, self.world)
        self.store.bind_channel(1, 200, self.hub)
        self.character = self.store.add_character(1, self.world, 'Courier', {'name': 'Courier'}, None, [])
        self.store.set_cast(100, None, [self.character])
        self.store.set_cast(200, None, [self.character])
        self.engine = Engine(self.store, FakeModels(), core_settings())

    def tearDown(self):
        self.store.close()

    async def text(self, channel, space):
        scene = SceneContext(1, channel, None, space, 111, 5000 + channel, 'Hello', None, [], [], user_label='Sam')
        self.engine.record_user(scene)
        request, _ = await self.engine.prepare_dialogue(scene, self.store.character_by_id(self.character), [])
        return '\n'.join(m.text for m in request.messages)

    async def test_off_duty_hub_adds_line_after_location(self):
        before = await self.text(200, self.hub)
        self.assertNotIn(LINE, before)
        self.store.set_hub_tone(1, self.hub, 'off_duty')
        after = await self.text(200, self.hub)
        self.assertIn('You are in hub Lobby. ' + LINE, after)
        self.assertEqual(after.replace(' ' + LINE, ''), before)

    async def test_in_character_hub_and_world_prompts_unchanged(self):
        world_before = await self.text(100, self.world)
        self.store.set_hub_tone(1, self.hub, 'off_duty')
        self.assertEqual(await self.text(100, self.world), world_before)
        self.assertIn('You are in world Harbor.', world_before)
        self.assertNotIn('off-duty', world_before)
        self.store.set_hub_tone(1, self.hub, 'in_character')
        self.assertNotIn('off-duty', await self.text(200, self.hub))


class CommandTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.hub = self.store.create_space(1, 'Lobby', 'hub')
        self.world = self.store.create_space(1, 'Harbor', 'world')
        self.foreign = self.store.create_space(2, 'Elsewhere', 'hub')

    def tearDown(self):
        self.store.close()

    async def run_cmd(self, path, *args):
        interaction = FakeInteraction(admin=True)
        await invoke(self.bot, path, interaction, *args)
        return interaction

    async def test_tone_replies_and_persists(self):
        done = await self.run_cmd('admin space tone', 'Lobby', 'off_duty')
        self.assertEqual(done.replies, ['Hub Lobby tone: off duty.'])
        self.assertTrue(done.response.sent[0][1]['ephemeral'])
        self.assertEqual(done.response.sent[0][1]['allowed_mentions'].everyone, False)
        self.assertEqual(self.store.hub_tone(1, self.hub), 'off_duty')
        self.assertEqual((await self.run_cmd('admin space tone', 'lobby', 'in_character')).replies, ['Hub Lobby tone: in character.'])
        self.assertEqual(self.store.hub_tone(1, self.hub), 'in_character')

    async def test_world_and_other_guild_hub_refused(self):
        world = await self.run_cmd('admin space tone', 'Harbor', 'off_duty')
        self.assertEqual(world.replies, ['Harbor is a world, not a hub.'])
        foreign = await self.run_cmd('admin space tone', 'Elsewhere', 'off_duty')
        self.assertEqual(len(foreign.replies), 1)
        self.assertNotIn('tone:', foreign.replies[0])
        self.assertEqual(self.store.hub_tone(1, self.world), 'in_character')
        self.assertEqual(self.store.hub_tone(2, self.foreign), 'in_character')

    async def test_autocomplete_lists_hubs_only(self):
        choices = await autocomplete(self.bot, 'admin space tone', 'hub', FakeInteraction(admin=True), '')
        self.assertEqual([c.name for c in choices], ['Lobby'])

    async def test_space_list_marks_off_duty_hubs(self):
        self.assertEqual((await self.run_cmd('space list')).replies,
                         [f'#{self.hub} hub: Lobby\n#{self.world} world: Harbor'])
        self.store.set_hub_tone(1, self.hub, 'off_duty')
        self.assertEqual((await self.run_cmd('space list')).replies,
                         [f'#{self.hub} hub: Lobby (off duty)\n#{self.world} world: Harbor'])
