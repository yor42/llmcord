"""Error mapping for turns and slash commands (UX-07, SEC-05 error part; decision D3 in docs/engineering/roadmap.md).

D3: the public channel gets a generic message with a short reference id; the person who ran a command gets the
stage and (sanitized) provider detail as an ephemeral reply when there is an interaction; the detail is logged with
the same reference id. Command outcomes that used to be ambiguous ("removed if it belonged to you", "deleted if
present") become definite.

Seams: ``SkitBot.run_scene`` with ``FakeTextChannel`` (mention-triggered turns have no interaction), the ``/summon``
callback via ``helpers.invoke`` with a ``FakeTextChannel`` as the interaction channel, and slash commands via
``helpers.invoke`` (runs the tree's real error handler). The reference id format is not pinned: tests only require
that the token found in the user-facing text (``helpers.reference_ids``) also appears in the logged ERROR line.
"""
import logging
import re
import sqlite3
import unittest
from types import SimpleNamespace

import anthropic
import discord
import httpx
import openai
from discord import app_commands

from helpers import (FakeInteraction, FakeModels, FakeSentMessage, FakeTextChannel, command, invoke, make_settings,
                     not_found, reference_ids)

from llmcord_core.discord_bot import SkitBot
from llmcord_core.engine import SceneContext

MODEL = "secret-model-x1"
PROVIDER_TEXT = "quota exceeded for org-4471"
CREDENTIAL = "sk-live-0042"


class ProviderQuotaError(Exception):
    """A provider failure whose class name, message and model name must not reach the public channel."""


def provider_error():
    return ProviderQuotaError(f"{PROVIDER_TEXT} on {MODEL} (Bearer {CREDENTIAL}) see https://api.example/v1/usage")


def timeout_errors():
    request = httpx.Request("POST", "http://localhost/v1/chat/completions")
    return [httpx.ReadTimeout("The read operation timed out", request=request),
            httpx.PoolTimeout("pool timed out"),
            openai.APITimeoutError(request=request),
            anthropic.APITimeoutError(request=request)]


class FailingModels(FakeModels):
    """Director picks the given speakers; dialogue streaming raises ``error`` (stage 'dialogue generation')."""

    def __init__(self, speakers, error):
        super().__init__(speakers)
        self.error = error

    async def stream_text(self, role, system, messages):
        raise self.error
        yield


def error_lines(logs) -> list[str]:
    return [record.getMessage() for record in logs.records if record.levelno >= logging.ERROR]


class TurnFailureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings(model=MODEL))
        self.store = self.bot.store
        self.world = self.store.create_space(1, "World", "world")
        self.store.bind_channel(1, 100, self.world)
        self.alice = self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])
        self.store.set_cast(100, None, [self.alice])
        self.channel = FakeTextChannel(100)

    def tearDown(self):
        self.store.close()

    def use_models(self, models):
        self.bot.engine.models = self.bot.models = models

    async def mention_turn(self, error):
        """A mention-style turn (no interaction) that fails during dialogue generation; returns ERROR log lines."""
        self.use_models(FailingModels([self.alice], error))
        with self.assertLogs(level="ERROR") as logs:
            await self.bot.run_scene(SceneContext(1, 100, None, self.world, 9, 1000, "Hi", None, [], []), self.channel)
        return error_lines(logs)

    async def summon(self, error=None):
        """Run ``/summon Alice`` whose turn fails with ``error`` (or succeeds when None); returns (interaction, log lines)."""
        self.use_models(FailingModels([self.alice], error) if error else FakeModels([self.alice]))
        interaction = FakeInteraction(channel=self.channel)
        if error is None:
            await invoke(self.bot, "summon", interaction, "Alice", "Wave hello")
            return interaction, []
        with self.assertLogs(level="ERROR") as logs:
            await invoke(self.bot, "summon", interaction, "Alice", "Wave hello")
        return interaction, error_lines(logs)

    def assert_generic(self, public):
        for leaked in ("ProviderQuotaError", PROVIDER_TEXT, MODEL, CREDENTIAL, "api.example"):
            self.assertNotIn(leaked, public)

    def ref_shared_with_log(self, text, log_lines):
        """The reference token in ``text`` that also appears in a logged ERROR line; returns (token, that line)."""
        tokens = reference_ids(text)
        self.assertTrue(tokens, f"no reference id in {text!r}")
        for token in tokens:
            for line in log_lines:
                if token in line:
                    return token, line
        self.fail(f"reference ids {tokens} from {text!r} not found in ERROR log {log_lines!r}")

    async def test_turn_failure_public_message_is_generic(self):
        """UX-07/SEC-05 (D3): a failed turn posts one public message without the error type, provider text, model
        name or credentials (previously it posted 'Character response failed during <stage>: <error_detail>')."""
        await self.mention_turn(provider_error())
        self.assertEqual(len(self.channel.errors), 1, self.channel.errors)
        self.assert_generic(self.channel.errors[0])

    async def test_turn_failure_reference_id_in_public_message_and_log(self):
        """UX-07/SEC-05 (D3): the public failure message carries a reference id, and the logged ERROR line with the same
        id still has the stage and the error detail."""
        log_lines = await self.mention_turn(provider_error())
        self.assertEqual(len(self.channel.errors), 1, self.channel.errors)
        _, line = self.ref_shared_with_log(self.channel.errors[0], log_lines)
        self.assertIn("dialogue generation", line)
        self.assertIn(PROVIDER_TEXT, line)

    async def test_turn_failure_fallback_send_is_generic_with_reference(self):
        """UX-07/SEC-05 (D3): when editing the progress message fails, the failure is posted with channel.send; that
        message is generic too and carries the logged reference id."""
        original_send = self.channel.send

        class StuckStatus(FakeSentMessage):
            async def edit(self, *, content, **kwargs):
                if not content.startswith("⏳"):
                    raise discord.HTTPException(SimpleNamespace(status=500, reason="Server Error"), "edit failed")
                await super().edit(content=content, **kwargs)

        async def send(content=None, **kwargs):
            message = await original_send(content, **kwargs)
            if content and content.startswith("⏳"):
                stuck = StuckStatus(message.id, content)
                self.channel.sent[-1] = stuck
                return stuck
            return message

        self.channel.send = send
        log_lines = await self.mention_turn(provider_error())
        self.assertEqual(len(self.channel.errors), 1, self.channel.errors)
        failure = next(m for m in self.channel.sent if not m.deleted and m.content == self.channel.errors[0])
        self.assertNotIsInstance(failure, StuckStatus, "the failure should be a new channel.send message")
        public = failure.content
        self.assert_generic(public)
        self.ref_shared_with_log(public, log_lines)

    async def test_mention_turn_failure_posts_only_one_public_message(self):
        """Characterization (UX-07): a mention-triggered turn has no interaction; its failure produces exactly one
        visible non-status message in the channel and an ERROR log line with the stage."""
        log_lines = await self.mention_turn(provider_error())
        visible = [m.content for m in self.channel.sent if not m.deleted]
        self.assertEqual(visible, self.channel.errors)
        self.assertEqual(len(visible), 1)
        self.assertTrue(any("dialogue generation" in line for line in log_lines), log_lines)

    async def test_summon_failure_sends_ephemeral_detail_with_reference(self):
        """UX-07/SEC-05 (D3): a failed /summon turn posts the generic public message and sends the invoker an
        ephemeral followup with the stage, the sanitized detail and the same reference id as the public message
        and the log."""
        interaction, log_lines = await self.summon(provider_error())
        self.assertEqual(len(self.channel.errors), 1, self.channel.errors)
        public = self.channel.errors[0]
        self.assert_generic(public)
        token, _ = self.ref_shared_with_log(public, log_lines)
        private = [(content, kwargs) for content, kwargs in interaction.followup.sent if content and token in content]
        self.assertEqual(len(private), 1, interaction.followup.sent)
        content, kwargs = private[0]
        self.assertIs(kwargs.get("ephemeral"), True)
        self.assertIn("dialogue generation", content)
        self.assertIn(PROVIDER_TEXT, content)
        self.assertNotIn(CREDENTIAL, content)
        self.assertNotIn("api.example", content)

    async def test_summon_timeout_detail_is_friendly(self):
        """UX-07 (D3, R3 follow-up): a provider timeout (httpx.TimeoutException, openai/anthropic APITimeoutError) gives
        the /summon invoker the friendly detail "The model provider did not respond in time", not the raw class or
        SDK text."""
        for error in timeout_errors():
            with self.subTest(type(error).__name__):
                self.channel = FakeTextChannel(100)
                self.bot.webhooks.clear()
                self.bot.webhook_defaults.clear()
                interaction, log_lines = await self.summon(error)
                self.assertEqual(len(self.channel.errors), 1, self.channel.errors)
                token, _ = self.ref_shared_with_log(self.channel.errors[0], log_lines)
                private = [content for content, kwargs in interaction.followup.sent
                           if content and token in content and kwargs.get("ephemeral")]
                self.assertEqual(len(private), 1, interaction.followup.sent)
                self.assertIn("did not respond in time", private[0])
                for raw in ("Timeout", "timed out"):
                    self.assertNotIn(raw, private[0])
                self.assertNotIn("Timeout", self.channel.errors[0])

    async def test_successful_summon_sends_no_ephemeral_error(self):
        """Regression (UX-07): a /summon turn that succeeds sends only the public invitation, no followup and no
        failure message."""
        interaction, _ = await self.summon()
        self.assertEqual(interaction.followup.sent, [])
        self.assertEqual(len(interaction.response.sent), 1)
        self.assertEqual(self.channel.errors, [])
        self.assertTrue(self.channel.hooks and self.channel.hooks[0].posts)

    # --- R4 step 1 reviewer follow-ups (UX-07/SEC-05, D3) ---------------------------------------------------------

    async def summon_internal_failure(self):
        """Run a /summon whose turn fails outside the provider stages (models succeed); returns (private ephemeral
        followups carrying the shared ref, the matching ERROR log line)."""
        interaction, log_lines = await self.summon_raw()
        self.assertEqual(len(self.channel.errors), 1, self.channel.errors)
        token, line = self.ref_shared_with_log(self.channel.errors[0], log_lines)
        private = [(content, kwargs) for content, kwargs in interaction.followup.sent if content and token in content]
        self.assertEqual(len(private), 1, interaction.followup.sent)
        self.assertIs(private[0][1].get("ephemeral"), True)
        return private[0][0], line

    async def summon_raw(self):
        self.use_models(FakeModels([self.alice]))
        interaction = FakeInteraction(channel=self.channel)
        with self.assertLogs(level="ERROR") as logs:
            await invoke(self.bot, "summon", interaction, "Alice", "Wave hello")
        return interaction, error_lines(logs)

    def fail_record_node(self, *, character_reply):
        """Make ``Store.record_node`` raise a UNIQUE IntegrityError for the user input (``saving scene input``) or
        for the character reply (``saving character reply``)."""
        original = self.store.record_node

        def record_node(message_id, guild_id, channel_id, parent_id, author_id, character_id, *args, **kwargs):
            if (character_id is not None) == character_reply:
                raise sqlite3.IntegrityError("UNIQUE constraint failed: nodes.message_id")
            return original(message_id, guild_id, channel_id, parent_id, author_id, character_id, *args, **kwargs)

        self.store.record_node = record_node

    def assert_no_internal_detail(self, private):
        for leaked in ("UNIQUE", "constraint", "nodes", "message_id", "IntegrityError", "sqlite", "KeyError",
                       "'neutral'"):
            self.assertNotIn(leaked, private)

    async def test_summon_storage_failure_on_input_hides_detail(self):
        """UX-07/SEC-05 (D3): a /summon turn that fails while saving the scene input (sqlite UNIQUE IntegrityError)
        gives the invoker an ephemeral message with the ref and a generic text, not the raw sqlite text, table or
        column names or exception class; the ERROR log line with that ref keeps the full detail. Only provider stages
        add ``user_detail(error)`` to the private followup."""
        self.fail_record_node(character_reply=False)
        private, line = await self.summon_internal_failure()
        self.assertIn("UNIQUE constraint failed: nodes.message_id", line)
        self.assertIn("saving scene input", line)
        self.assert_no_internal_detail(private)

    async def test_summon_storage_failure_on_reply_hides_detail(self):
        """UX-07/SEC-05 (D3): same as the scene-input case, for a UNIQUE IntegrityError while saving the character
        reply (stage 'saving character reply'); the raw sqlite text stays in the log only."""
        self.fail_record_node(character_reply=True)
        private, line = await self.summon_internal_failure()
        self.assertIn("UNIQUE constraint failed: nodes.message_id", line)
        self.assertIn("saving character reply", line)
        self.assert_no_internal_detail(private)

    async def test_summon_internal_key_error_hides_detail(self):
        """UX-07/SEC-05 (D3): a KeyError('neutral') in a non-provider stage (no usable avatar slots, stage 'webhook
        setup') gives the invoker the ref and a generic text, without the exception class or key; the ERROR log keeps
        the detail."""
        self.store.usable_avatars = lambda _guild_id, _character_id: []
        private, line = await self.summon_internal_failure()
        self.assertIn("neutral", line)
        self.assertIn("KeyError", line)
        self.assert_no_internal_detail(private)

    async def test_avatar_lookup_failure_is_logged_with_stage(self):
        """ARCH-01: a failure while resolving the emotion avatar mid-stream is logged with stage 'avatar lookup' and
        the invoker sees the generic internal-error text."""
        async def resolve_avatar(_slot):
            raise RuntimeError("avatar boom")

        self.bot.resolve_avatar = resolve_avatar
        private, line = await self.summon_internal_failure()
        self.assertIn("avatar lookup", line)
        self.assertIn("avatar lookup", private)
        self.assertNotIn("avatar boom", private)

    async def assert_stage_failure(self, stage):
        private, line = await self.summon_internal_failure()
        self.assertIn(f"failed during {stage}", private)
        self.assertIn(stage, line)
        self.assertIn("stage boom", line)

    async def test_speaker_selection_failure_names_stage(self):
        """MNT-09: a failure while choosing speakers is reported and logged as 'speaker selection'."""
        async def speakers(_scene):
            raise RuntimeError("stage boom")

        self.bot.engine.speakers = speakers
        await self.assert_stage_failure("speaker selection")

    async def test_image_description_failure_names_stage(self):
        """MNT-09: a failure while describing images is reported and logged as 'image description'."""
        async def describe_images(_scene):
            raise RuntimeError("stage boom")

        self.bot.engine.describe_images = describe_images
        await self.assert_stage_failure("image description")

    async def test_prompt_preparation_failure_names_stage(self):
        """MNT-09: a failure while building a character's prompt is reported and logged as 'preparing character prompt'."""
        async def prepare_dialogue(*_args):
            raise RuntimeError("stage boom")

        self.bot.engine.prepare_dialogue = prepare_dialogue
        await self.assert_stage_failure("preparing character prompt")

    async def test_webhook_setup_failure_names_stage(self):
        """MNT-09: a failure creating the character webhook is reported and logged as 'webhook setup'."""
        async def webhook(*_args):
            raise RuntimeError("stage boom")

        self.bot._webhook = webhook
        await self.assert_stage_failure("webhook setup")

    def direct_turn(self, interaction):
        """A scene run directly through ``run_scene`` with an interaction (so an escaping exception is visible to
        the test instead of being swallowed by ``invoke``)."""
        self.use_models(FailingModels([self.alice], provider_error()))
        scene = SceneContext(1, 100, None, self.world, 9, 1000, "Hi", None, [], [], forced_character_id=self.alice)
        return self.bot.run_scene(scene, self.channel, interaction)

    async def test_public_failure_post_error_still_notifies_privately(self):
        """UX-07 (D3): when both the progress edit and the fallback ``channel.send`` of the public failure raise
        ``discord.HTTPException``, ``run_scene`` does not raise (a WARNING carries the ref), the invoker still gets
        the ephemeral followup with the logged ref, and the progress message is still cleaned up."""
        original_send = self.channel.send

        def http_error():
            return discord.HTTPException(SimpleNamespace(status=500, reason="Server Error"), "send failed")

        class StuckStatus(FakeSentMessage):
            async def edit(self, *, content, **kwargs):
                if not content.startswith("⏳"):
                    raise http_error()
                await super().edit(content=content, **kwargs)

        statuses = []

        async def send(content=None, **kwargs):
            if not (content or "").startswith("⏳"):
                raise http_error()
            message = await original_send(content, **kwargs)
            stuck = StuckStatus(message.id, content)
            self.channel.sent[-1] = stuck
            statuses.append(stuck)
            return stuck

        self.channel.send = send
        interaction = FakeInteraction(channel=self.channel)
        with self.assertLogs(level="ERROR") as logs:
            await self.direct_turn(interaction)
        self.assertEqual(len(statuses), 1, statuses)
        self.assertTrue(statuses[0].deleted, "the progress message should still be cleaned up")
        private = [(content, kwargs) for content, kwargs in interaction.followup.sent if reference_ids(content)]
        self.assertEqual(len(private), 1, interaction.followup.sent)
        self.assertIs(private[0][1].get("ephemeral"), True)
        self.ref_shared_with_log(private[0][0], error_lines(logs))

    async def test_followup_failure_is_logged_and_public_message_stays(self):
        """Regression (UX-07, D3): when ``interaction.followup.send`` raises ``discord.NotFound`` (expired
        interaction), the public failure is still posted, no progress message is left, a WARNING carries the same
        ref, and ``run_scene`` returns normally."""
        interaction = FakeInteraction(channel=self.channel)

        async def gone(content=None, **kwargs):
            raise not_found("Unknown Webhook")

        interaction.followup.send = gone
        with self.assertLogs(level="WARNING") as logs:
            await self.direct_turn(interaction)
        self.assertEqual(len(self.channel.errors), 1, self.channel.errors)
        self.assertEqual([m.content for m in self.channel.sent if not m.deleted and m.content.startswith("⏳")], [])
        token, _ = self.ref_shared_with_log(self.channel.errors[0], error_lines(logs))
        warnings = [record.getMessage() for record in logs.records if record.levelno == logging.WARNING]
        self.assertTrue(any(token in line for line in warnings), (token, warnings))


class CommandErrorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.world = self.store.create_space(1, "Harbor", "world")
        self.store.bind_channel(1, 100, self.world)

    def tearDown(self):
        self.store.close()

    async def test_duplicate_space_says_already_exists(self):
        """UX-07 (fixed): a UNIQUE conflict (duplicate /admin space create) becomes a plain ephemeral "already exists"
        message with no sqlite text, table or column names; the raw 'UNIQUE constraint failed: ...' never reaches the
        user."""
        interaction = FakeInteraction(admin=True)
        await invoke(self.bot, "admin space create", interaction, "world", "Harbor")
        self.assertEqual(len(interaction.replies), 1, interaction.replies)
        reply = interaction.replies[0]
        self.assertIn("already exists", reply.lower())
        for leaked in ("UNIQUE", "constraint", "spaces.", "guild_id", "IntegrityError"):
            self.assertNotIn(leaked, reply)
        self.assertIs(interaction.response.sent[0][1].get("ephemeral"), True)

    async def test_value_error_message_reaches_user_ephemerally(self):
        """Characterization (UX-07): ValueError messages come from our own code and are still shown as written,
        ephemerally."""
        interaction = FakeInteraction(channel_id=999)
        await invoke(self.bot, "cast show", interaction)
        self.assertEqual(interaction.replies, ["This channel is not bound to a world or hub. Ask an administrator to bind it with /admin space bind."])
        self.assertIs(interaction.response.sent[0][1].get("ephemeral"), True)

    async def test_missing_permissions_message_unchanged(self):
        """Characterization (UX-07): MissingPermissions keeps its clear ephemeral message."""
        interaction = FakeInteraction(admin=False)
        await invoke(self.bot, "admin space create", interaction, "world", "Elsewhere")
        self.assertEqual(interaction.replies, ["Only server administrators can use that command."])
        self.assertIs(interaction.response.sent[0][1].get("ephemeral"), True)

    async def test_unexpected_error_is_generic_with_logged_reference(self):
        """UX-07/SEC-05: any other exception becomes a generic ephemeral reply with a reference id, without the
        exception type or text; the ERROR log line has the same id and the detail (previously: 'Command failed: KeyError: ...')."""
        interaction = FakeInteraction(admin=True)

        def broken(_guild_id):
            raise KeyError("internal-column-secret")

        self.store.list_spaces = broken
        with self.assertLogs(level="ERROR") as logs:
            await invoke(self.bot, "space list", interaction)
        self.assertEqual(len(interaction.replies), 1, interaction.replies)
        reply = interaction.replies[0]
        self.assertNotIn("KeyError", reply)
        self.assertNotIn("internal-column-secret", reply)
        self.assertIs(interaction.response.sent[0][1].get("ephemeral"), True)
        tokens = reference_ids(reply)
        self.assertTrue(tokens, f"no reference id in {reply!r}")
        lines = [line for line in error_lines(logs) if any(token in line for token in tokens)]
        self.assertTrue(lines, f"reference ids {tokens} not in ERROR log {error_lines(logs)!r}")
        self.assertIn("internal-column-secret", lines[0])

    async def test_unexpected_error_after_response_uses_ephemeral_followup(self):
        """UX-07/SEC-05: when the interaction was already answered, the generic reference message goes out as an
        ephemeral followup (no exception text)."""
        interaction = FakeInteraction(admin=True)
        await interaction.response.send_message("working")
        error = sqlite3.OperationalError("database is locked: table spaces")
        with self.assertLogs(level="ERROR") as logs:
            await self.bot.tree.on_error(interaction, app_commands.CommandInvokeError(command(self.bot, "space list"), error))
        self.assertEqual(len(interaction.followup.sent), 1, interaction.followup.sent)
        content, kwargs = interaction.followup.sent[0]
        self.assertIs(kwargs.get("ephemeral"), True)
        self.assertNotIn("database is locked", content)
        self.assertNotIn("OperationalError", content)
        tokens = reference_ids(content)
        self.assertTrue(tokens and any(t in line for t in tokens for line in error_lines(logs)), (content, logs.output))

    async def test_non_unique_integrity_error_is_generic_with_logged_reference(self):
        """Regression (UX-07/SEC-05, D3): only UNIQUE conflicts map to "already exists"; another IntegrityError
        (FOREIGN KEY) gets the generic ephemeral reference message without sqlite text, and the ERROR log line with
        the same ref has the detail."""
        interaction = FakeInteraction(admin=True)

        def broken(_guild_id):
            raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")

        self.store.list_spaces = broken
        with self.assertLogs(level="ERROR") as logs:
            await invoke(self.bot, "space list", interaction)
        self.assertEqual(len(interaction.replies), 1, interaction.replies)
        reply = interaction.replies[0]
        self.assertNotIn("already exists", reply.lower())
        for leaked in ("FOREIGN", "constraint", "IntegrityError"):
            self.assertNotIn(leaked, reply)
        self.assertIs(interaction.response.sent[0][1].get("ephemeral"), True)
        tokens = reference_ids(reply)
        self.assertTrue(tokens, f"no reference id in {reply!r}")
        lines = [line for line in error_lines(logs) if any(token in line for token in tokens)]
        self.assertTrue(lines, f"reference ids {tokens} not in ERROR log {error_lines(logs)!r}")
        self.assertIn("FOREIGN KEY constraint failed", lines[0])


NOT_FOUND = re.compile(r"(?i)\b(no|not)\b")
AMBIGUOUS = re.compile(r"(?i)\bif\b")


class DefiniteOutcomeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.world = self.store.create_space(1, "Harbor", "world")
        self.store.bind_channel(1, 100, self.world)
        self.store.bind_channel(1, 101, self.world)
        self.alice = self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])
        self.other_world = self.store.create_space(2, "Elsewhere", "world")
        self.store.bind_channel(2, 200, self.other_world)
        self.carol = self.store.add_character(2, self.other_world, "Carol", {"name": "Carol"}, None, [])

    def tearDown(self):
        self.store.close()

    def remember(self, guild_id, user_id, character_id, content):
        self.store.set_consent(guild_id, user_id, True)
        if not self.store.node(1000 + guild_id):
            self.store.record_node(1000 + guild_id, guild_id, 100 * guild_id, None, 9, None, "src")
        self.store.add_personal(guild_id, user_id, character_id, content, 1000 + guild_id)
        return next(row["id"] for row in self.store.personal(guild_id, user_id) if row["content"] == content)

    async def forget(self, memory_id, user_id=9, guild_id=1):
        interaction = FakeInteraction(guild_id=guild_id, channel_id=100 if guild_id == 1 else 200, user_id=user_id)
        await invoke(self.bot, "memory forget", interaction, memory_id)
        self.assertEqual(len(interaction.replies), 1, interaction.replies)
        self.assertIs(interaction.response.sent[0][1].get("ephemeral"), True)
        return interaction.replies[0]

    async def delete_lore(self, lore_id, guild_id=1, channel_id=100):
        interaction = FakeInteraction(guild_id=guild_id, channel_id=channel_id, admin=True)
        await invoke(self.bot, "admin lore delete", interaction, lore_id)
        self.assertEqual(len(interaction.replies), 1, interaction.replies)
        self.assertIs(interaction.response.sent[0][1].get("ephemeral"), True)
        return interaction.replies[0]

    def assert_removed_reply(self, reply, ident):
        self.assertIn(f"#{ident}", reply)
        self.assertRegex(reply.lower(), r"removed|deleted")
        self.assertNotRegex(reply, NOT_FOUND)
        self.assertNotRegex(reply, AMBIGUOUS)

    def assert_not_found_reply(self, reply, ident):
        self.assertIn(f"#{ident}", reply)
        self.assertRegex(reply, NOT_FOUND)
        self.assertNotRegex(reply, AMBIGUOUS)

    async def test_memory_forget_reports_removed(self):
        """UX-07: /memory forget on your own memory says it was removed (e.g. "Memory #N removed.") and deletes it
        (previously: "Memory removed if it belonged to you.")."""
        mine = self.remember(1, 9, self.alice, "Likes tea")
        self.assert_removed_reply(await self.forget(mine), mine)
        self.assertEqual(self.store.personal(1, 9), [])

    async def test_memory_forget_reports_not_found(self):
        """UX-07: /memory forget says nothing was found (e.g. "No memory #N of yours was found.") for an unknown id,
        another user's memory, or the caller's own memory id from another guild; those rows stay."""
        theirs = self.remember(1, 77, self.alice, "Hates rain")
        elsewhere = self.remember(2, 9, self.carol, "Plays chess")
        for ident, label in ((9999, "unknown id"), (theirs, "other user's memory"), (elsewhere, "other guild's memory")):
            with self.subTest(label):
                self.assert_not_found_reply(await self.forget(ident), ident)
        self.assertEqual([row["content"] for row in self.store.personal(1, 77)], ["Hates rain"])
        self.assertEqual([row["content"] for row in self.store.personal(2, 9)], ["Plays chess"])

    async def test_memory_forget_never_deletes_across_users_or_guilds(self):
        """Regression (UX-07, consent/guild isolation): /memory forget only deletes the caller's row in the caller's
        guild."""
        mine = self.remember(1, 9, self.alice, "Likes tea")
        theirs = self.remember(1, 77, self.alice, "Hates rain")
        elsewhere = self.remember(2, 9, self.carol, "Plays chess")
        await self.forget(theirs)
        await self.forget(elsewhere)
        await self.forget(mine, user_id=77)
        self.assertEqual(len(self.store.personal(1, 9)), 1)
        self.assertEqual(len(self.store.personal(1, 77)), 1)
        self.assertEqual(len(self.store.personal(2, 9)), 1)

    async def test_lore_delete_reports_deleted(self):
        """UX-07: /admin lore delete on an entry it can delete says so (e.g. "Deleted lore #N.") and removes it."""
        ident = self.store.add_lore(1, "channel", 100, "The tide is high")
        self.assert_removed_reply(await self.delete_lore(ident), ident)
        self.assertIsNone(self.store.lore_row(1, ident))

    async def test_lore_delete_reports_not_found(self):
        """UX-07: /admin lore delete says nothing was found for an unknown id or another guild's lore id, and the other
        guild's entry stays."""
        foreign = self.store.add_lore(2, "channel", 200, "Snow falls upward")
        for ident, label in ((9999, "unknown id"), (foreign, "other guild's lore")):
            with self.subTest(label):
                self.assert_not_found_reply(await self.delete_lore(ident), ident)
        self.assertIsNotNone(self.store.lore_row(2, foreign))

    async def test_lore_delete_never_deletes_across_guilds(self):
        """Regression (UX-07, guild isolation): /admin lore delete refuses another guild's lore id."""
        foreign = self.store.add_lore(2, "channel", 200, "Snow falls upward")
        await self.delete_lore(foreign)
        self.assertIsNotNone(self.store.lore_row(2, foreign))

    async def test_lore_delete_scope_is_the_guild(self):
        """Characterization (UX-07): /admin lore delete is scoped by guild only (``Store.delete_lore(guild_id, id)``), not
        by the invoking channel or space: an admin can delete another channel's lore in the same guild by id. Step 1
        keeps this scope; only the reply wording changes."""
        other_channel = self.store.add_lore(1, "channel", 101, "The bell rings twice")
        await self.delete_lore(other_channel, channel_id=100)
        self.assertIsNone(self.store.lore_row(1, other_channel))


if __name__ == "__main__":
    unittest.main()
