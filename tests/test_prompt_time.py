from dataclasses import replace
from datetime import datetime, timezone
import unittest

import httpx

from llmcord_core import prompts
from llmcord_core.engine import Engine, SceneContext
from llmcord_core.prompts import block, compatibility, default_bundle
from llmcord_core.store import Store
from llmcord_core.web import create_app
from helpers import CoreFakeModels as FakeModels, core_settings as settings

EPOCH = 1420070400000


def snowflake(*args):
    ms = int(datetime(*args, tzinfo=timezone.utc).timestamp() * 1000)
    return (ms - EPOCH) << 22


class TimeValuesTests(unittest.TestCase):
    def test_all_keys_and_formats_in_member_zone(self):
        """FEAT-02: time_values returns English, locale-free values in the requested zone."""
        values = prompts.time_values(snowflake(2026, 10, 8, 8, 39), 'Asia/Seoul')
        self.assertEqual(values, {
            'time': '5:39 PM', 'date': 'October 8, 2026', 'weekday': 'Thursday',
            'isotime': '17:39', 'isodate': '2026-10-08',
            'local_time': 'Thursday, October 8, 2026, 5:39 PM (Asia/Seoul)'})

    def test_local_date_can_differ_from_utc(self):
        """FEAT-02: 23:30Z on Oct 8 is already Friday Oct 9 in Seoul."""
        values = prompts.time_values(snowflake(2026, 10, 8, 23, 30), 'Asia/Seoul')
        self.assertEqual((values['weekday'], values['date'], values['isodate'], values['time']),
                         ('Friday', 'October 9, 2026', '2026-10-09', '8:30 AM'))
        self.assertEqual(values['date'], 'October 9, 2026')
        self.assertEqual(values['isodate'], '2026-10-09')
        self.assertEqual(values['isotime'], '08:30')

    def test_midnight_and_noon(self):
        """FEAT-02: the 12-hour clock shows 12:00 AM at midnight and 12:00 PM at noon, without a leading zero."""
        self.assertEqual(prompts.time_values(snowflake(2026, 10, 8, 0, 0), 'UTC')['time'], '12:00 AM')
        self.assertEqual(prompts.time_values(snowflake(2026, 10, 8, 12, 0), 'UTC')['time'], '12:00 PM')
        self.assertEqual(prompts.time_values(snowflake(2026, 10, 8, 0, 0), 'UTC')['isotime'], '00:00')
        self.assertEqual(prompts.time_values(snowflake(2026, 10, 8, 9, 5), 'UTC')['time'], '9:05 AM')

    def test_empty_or_none_zone_is_utc(self):
        """FEAT-02: an empty or missing zone name falls back to UTC and says so."""
        for zone in ('', None):
            with self.subTest(zone=zone):
                values = prompts.time_values(snowflake(2026, 10, 8, 17, 39), zone)
                self.assertEqual(values['local_time'], 'Thursday, October 8, 2026, 5:39 PM (UTC)')

    def test_same_message_id_is_stable(self):
        """FEAT-02: the instant comes from the snowflake, so repeated calls agree."""
        ident = snowflake(2026, 10, 8, 8, 39)
        self.assertEqual(prompts.time_values(ident, 'Asia/Seoul'), prompts.time_values(ident, 'Asia/Seoul'))


    def test_dst_spring_forward_new_york(self):
        """FEAT-02: 06:59Z on 2026-03-08 is 1:59 AM EST; one minute later the clock jumps to 3:00 AM EDT."""
        self.assertEqual(prompts.time_values(snowflake(2026, 3, 8, 6, 59), 'America/New_York')['time'], '1:59 AM')
        self.assertEqual(prompts.time_values(snowflake(2026, 3, 8, 7, 0), 'America/New_York')['time'], '3:00 AM')


class PresetTests(unittest.TestCase):
    def test_macros_registered_and_unknown_still_reported(self):
        """FEAT-02: the time macros are supported; unknown macros are still flagged."""
        for name in ('time', 'date', 'weekday', 'isotime', 'isodate', 'local_time'):
            self.assertIn(name, prompts.MACROS)
        bundle = default_bundle()
        bundle['purposes']['dialogue'].append(block('extra', content='{{time}} {{date}} {{weekday}} {{isotime}} {{isodate}} {{local_time}}'))
        self.assertEqual([p for p in compatibility(bundle, {}) if 'macros' in p], [])
        bundle['purposes']['dialogue'].append(block('bad', content='{{foo}}'))
        problems = [p for p in compatibility(bundle, {}) if 'macros' in p]
        self.assertEqual(len(problems), 1)
        self.assertIn('foo', problems[0])

    def test_default_dialogue_has_time_block_after_location(self):
        """FEAT-02: the default dialogue preset carries a time block directly after location."""
        dialogue = default_bundle()['purposes']['dialogue']
        ids = [b['id'] for b in dialogue]
        self.assertEqual(ids[ids.index('location') + 1], 'time')
        item = dialogue[ids.index('time')]
        self.assertEqual(item['content'], 'Local time for {{user}}: {{local_time}}')
        self.assertEqual(item['source'], 'text')


class EngineTimeTests(unittest.IsolatedAsyncioTestCase):
    MESSAGE = snowflake(2026, 10, 8, 8, 39)

    def setUp(self):
        self.store = Store()
        self.world = self.store.create_space(1, 'World', 'world')
        self.store.bind_channel(1, 100, self.world)
        self.character = self.store.add_character(1, self.world, 'Courier', {'name': 'Courier'}, None, [])
        self.store.set_cast(100, None, [self.character])
        self.engine = Engine(self.store, FakeModels(), settings())

    def tearDown(self):
        self.store.close()

    async def system_text(self, preset=None, ident=None):
        scene = SceneContext(1, 100, None, self.world, 111, ident or self.MESSAGE, 'Hello', None, [], [], user_label='Sam')
        if preset:
            scene = replace(scene, preset={'id': 99, 'revision': 1, 'bundle': preset})
        self.engine.record_user(scene)
        request, _ = await self.engine.prepare_dialogue(scene, self.store.character_by_id(self.character), [])
        return '\n'.join(m.text for m in request.messages)

    async def test_member_zone(self):
        """FEAT-02: the member's own timezone is used for the local time line."""
        self.store.set_guild_timezone(1, 'America/New_York')
        self.store.set_user_timezone(1, 111, 'Asia/Seoul')
        self.assertIn('Local time for Sam: Thursday, October 8, 2026, 5:39 PM (Asia/Seoul)', await self.system_text())

    async def test_server_zone(self):
        """FEAT-02: without a member zone the server default is used."""
        self.store.set_guild_timezone(1, 'America/New_York')
        self.assertIn('Local time for Sam: Thursday, October 8, 2026, 4:39 AM (America/New_York)', await self.system_text())

    async def test_utc_default(self):
        """FEAT-02: with no zone configured the prompt uses UTC."""
        self.assertIn('Local time for Sam: Thursday, October 8, 2026, 8:39 AM (UTC)', await self.system_text())

    async def test_recompiling_same_message_is_identical(self):
        """FEAT-02: retry/rewind of the same user message yields identical time text."""
        self.store.set_user_timezone(1, 111, 'Asia/Seoul')
        first = await self.system_text()
        second = await self.system_text()
        line = lambda text: [l for l in text.splitlines() if l.startswith('Local time for')]
        self.assertEqual(len(line(first)), 1)
        self.assertEqual(line(first), line(second))

    async def test_preset_macros_are_substituted(self):
        """FEAT-02: a preset text block can use the individual time macros."""
        bundle = default_bundle()
        bundle['purposes']['dialogue'].append(block('stamp', content='STAMP {{time}}|{{date}}|{{weekday}}|{{isotime}}|{{isodate}}'))
        self.store.set_user_timezone(1, 111, 'Asia/Seoul')
        text = await self.system_text(bundle)
        self.assertIn('STAMP 5:39 PM|October 8, 2026|Thursday|17:39|2026-10-08', text)

    async def test_invalid_stored_member_zone_falls_back_to_server_zone(self):
        """FEAT-02: a corrupt stored member zone is ignored; the server zone is used and nothing raises."""
        self.store.set_guild_timezone(1, 'America/New_York')
        with self.store.write_admin():
            self.store.db.execute('INSERT INTO user_timezones(guild_id,user_id,timezone,updated_at) VALUES(?,?,?,?)',
                                  (1, 111, 'Not/AZone', 0.0))
        self.assertIn('Local time for Sam: Thursday, October 8, 2026, 4:39 AM (America/New_York)', await self.system_text())

    async def test_user_label_macro_text_stays_literal(self):
        """FEAT-02: macros expand in one pass, so a user label of '{{local_time}}' is not expanded again."""
        scene = SceneContext(1, 100, None, self.world, 111, self.MESSAGE, 'Hello', None, [], [], user_label='{{local_time}}')
        self.engine.record_user(scene)
        request, _ = await self.engine.prepare_dialogue(scene, self.store.character_by_id(self.character), [])
        text = '\n'.join(m.text for m in request.messages)
        self.assertIn('Local time for {{local_time}}: Thursday, October 8, 2026, 8:39 AM (UTC)', text)


class PreviewTimeTests(unittest.IsolatedAsyncioTestCase):
    async def preview_line(self, zone=None):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[]))) as http:
            app = create_app(':memory:', 'https://pi.test', 'c', 's', 'b', http, enable_dashboard=False)
            try:
                if zone:
                    app.state.store.set_guild_timezone(1, zone)
                request = app.state.admin.preview_prompt(1, default_bundle(), 'dialogue', None, 'Sample', 'Earlier sample')
            finally:
                app.state.store.close()
        prefix = 'Local time for Sample user: '
        lines = [l for m in request.messages for l in m.text.splitlines() if l.startswith(prefix)]
        self.assertEqual(len(lines), 1)
        return lines[0][len(prefix):]

    async def test_preview_time_block_defaults_to_utc(self):
        """FEAT-02: the admin preview renders a non-empty time value labelled UTC without a server zone."""
        value = await self.preview_line()
        self.assertTrue(value.endswith('(UTC)'), value)

    async def test_preview_time_block_uses_server_zone(self):
        """FEAT-02: the admin preview labels the time with the server zone when one is set."""
        value = await self.preview_line('Asia/Seoul')
        self.assertTrue(value.endswith('(Asia/Seoul)'), value)
