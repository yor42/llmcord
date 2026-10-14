"""FEAT-52 step 2: ``SkitBot.on_message`` hands a thread message to ``onecard.on_thread_message`` (a failure there never stops the message), and
``SkitBot.run_table_event(..., speaker_hint=<character id>)`` makes that seated character the speaker of the event turn."""
from unittest import mock

import test_doubt_table_talk as talk
from helpers import drain_memory_tasks


class WiringTests(talk.TalkCase):
    async def test_on_message_passes_a_thread_message_to_the_one_card_reader(self):
        """FEAT-52: the hook is called with the message; if it raises, the rest of on_message still runs (the doubt counter is next to it)."""
        await self.table3()
        hook = self.bot.onecard.on_thread_message = mock.AsyncMock(side_effect=RuntimeError("boom"))
        await self.say(talk.ALICE, "hello", mention=False)
        hook.assert_awaited_once()
        self.assertEqual(hook.await_args.args[0].content, "hello")


class SpeakerHintTests(talk.TalkCase):
    async def test_a_hinted_seated_character_speaks_whatever_the_director_picks(self):
        """FEAT-52: Ann is the hinted speaker; the director would have picked Dee (not seated), and only one character speaks."""
        await self.table3()
        self.models.pick = [self.dee]
        await self.bot.run_table_event(self.thread(), "Alice has one card left.", speaker_hint=self.ann)
        await drain_memory_tasks(self.bot)
        self.assertEqual(self.spoke(), [self.ann])

    async def test_a_hint_for_someone_not_seated_is_ignored(self):
        """FEAT-52: a hint outside the table's characters changes nothing (the usual restriction holds)."""
        await self.table3()
        self.models.pick = [self.ann]
        await self.bot.run_table_event(self.thread(), "Alice has one card left.", speaker_hint=self.dee)
        await drain_memory_tasks(self.bot)
        self.assertEqual(self.spoke(), [self.ann])
