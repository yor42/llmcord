"""Webhook reuse and avatar-asset checks during delivery (REL-04).

These drive ``SkitBot.run_scene`` through the real ``_webhook`` path with ``FakeTextChannel`` (a
``discord.TextChannel`` subclass) and count Discord calls: ``listings`` = ``parent_channel.webhooks()``,
``creates`` = ``create_webhook``.
"""
import logging
import unittest
from types import SimpleNamespace

import discord

from llmcord_core.discord_bot import SkitBot
from llmcord_core.engine import SceneContext
from helpers import FakeModels, FakeTextChannel, ShiftedClock, make_settings, not_found


class LongModels(FakeModels):
    async def stream_text(self, role, system, messages):
        yield "<emotion>neutral</emotion>\n" + "word " * 600


class WebhookCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.world = self.store.create_space(1, "World", "world")
        self.store.bind_channel(1, 100, self.world)
        self.alice = self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])
        self.store.set_cast(100, None, [self.alice])
        self.bot.engine.models = self.bot.models = FakeModels([self.alice])
        self.channel = FakeTextChannel(100)
        self.turns = 0
        logging.disable(logging.ERROR)  # expected scene failures; failed_turn() re-enables logging to assert on it

    def tearDown(self):
        logging.disable(logging.NOTSET)
        self.store.close()

    async def turn(self):
        self.turns += 1
        await self.bot.run_scene(SceneContext(1, 100, None, self.world, 9, 1000 + self.turns, "Hi", None, [], []),
                                 self.channel)

    async def failed_turn(self) -> str:
        """Run a turn expected to fail; return its logged ERROR text. Stage and detail are pinned in the log, which
        carries them before and after D3 (public turn failures become generic; tests/test_error_mapping.py)."""
        logging.disable(logging.NOTSET)
        try:
            with self.assertLogs(level="ERROR") as logs:
                await self.turn()
        finally:
            logging.disable(logging.ERROR)
        return "\n".join(record.getMessage() for record in logs.records if record.levelname == "ERROR")

    def rename(self, name):
        self.store.execute("UPDATE characters SET name=? WHERE id=?", (name, self.alice))

    def assert_delivered_by(self, hook, start=0, prior_errors=0):
        self.assertEqual(len(self.channel.errors), prior_errors, self.channel.errors)
        posts = hook.posts[start:]
        self.assertTrue(posts, "no reply was delivered through the expected webhook")
        self.assertIn("Hello.", posts[0].content)

    # --- Characterization: must keep working ---------------------------------------------------

    async def test_first_turn_creates_webhook_and_saves_its_id(self):
        """Characterization (REL-04): with no stored id, the first turn creates one webhook (no listing),
        delivers through it with the character's name, and stores its id."""
        await self.turn()
        self.assertEqual((self.channel.listings, self.channel.creates), (0, 1))
        hook = self.channel.hooks[0]
        self.assert_delivered_by(hook)
        self.assertEqual(hook.name, "Alice")
        self.assertEqual(hook.options[0]["username"], "Alice")
        self.assertEqual(self.store.webhook_id(100, self.alice), hook.id)

    async def test_stored_webhook_is_reused_after_restart_and_identity_set_once(self):
        """Characterization (REL-04): a stored id that Discord still lists is reused (no create); a fresh
        process edits it once to set the name/avatar, and does not edit again while the identity is unchanged."""
        hook = await self.channel.create_webhook(name="Old name")
        self.channel.creates = 0
        self.store.save_webhook_id(100, self.alice, hook.id)
        await self.turn()
        await self.turn()
        self.assertEqual(self.channel.creates, 0)
        self.assertEqual(hook.edits, [{"name": "Alice", "avatar": None}])
        self.assertEqual(len(hook.posts), 2)
        self.assertEqual(self.channel.errors, [])

    async def test_identity_change_edits_webhook(self):
        """Characterization (REL-04): renaming a character makes the next turn edit the existing webhook's
        default name (no new webhook) and deliver under the new name."""
        await self.turn()
        hook = self.channel.hooks[0]
        self.rename("Alicia")
        await self.turn()
        self.assertEqual(self.channel.creates, 1)
        self.assertEqual(hook.edits, [{"name": "Alicia", "avatar": None}])
        self.assert_delivered_by(hook, start=1)
        self.assertEqual(hook.options[-1]["username"], "Alicia")

    async def test_forbidden_create_raises_manage_webhooks_error(self):
        """Characterization (REL-04): Forbidden from create_webhook becomes the "Manage Webhooks" ValueError,
        which the turn reports as a webhook-setup failure."""
        self.channel.create_error = discord.Forbidden(SimpleNamespace(status=403, reason="Forbidden"), "Missing Permissions")
        character = self.store.character_by_id(self.alice)
        with self.assertRaisesRegex(ValueError, "Manage Webhooks"):
            await self.bot._webhook(self.channel, character)
        logged = await self.failed_turn()
        self.assertEqual(len(self.channel.errors), 1)
        self.assertIn("webhook setup", logged)
        self.assertIn("Manage Webhooks", logged)
        self.assertIsNone(self.store.webhook_id(100, self.alice))

    async def test_webhook_deleted_between_turns_is_replaced(self):
        """Characterization (REL-04): if the webhook was deleted on Discord before a turn, that turn still
        delivers through a replacement webhook and stores the new id."""
        await self.turn()
        old = self.channel.hooks[0]
        self.channel.kill(old)
        await self.turn()
        new = self.channel.hooks[0]
        self.assertIsNot(new, old)
        self.assert_delivered_by(new)
        self.assertEqual(self.store.webhook_id(100, self.alice), new.id)

    async def test_not_found_on_continuation_chunk_fails_turn_and_next_turn_replaces_webhook(self):
        """Characterization (REL-04): NotFound on a later chunk fails the turn as webhook delivery (no retry),
        and the next turn does not touch the dead webhook again: it looks up, creates a new one and delivers."""
        self.bot.engine.models = self.bot.models = LongModels([self.alice])
        await self.turn()
        old = self.channel.hooks[0]
        self.assertGreater(len(old.posts), 1)
        old.fail_next_chunk = True
        logged = await self.failed_turn()
        self.assertEqual(len(self.channel.errors), 1)
        self.assertIn("webhook delivery", logged)
        self.assertIn("404", logged)
        attempts, listings = old.send_attempts, self.channel.listings
        self.bot.engine.models = self.bot.models = FakeModels([self.alice])
        await self.turn()
        self.assertEqual(old.send_attempts, attempts, "the dead webhook was reused after a chunk NotFound")
        self.assertGreaterEqual(self.channel.listings, listings + 1)
        new = self.channel.hooks[0]
        self.assert_delivered_by(new, prior_errors=1)
        self.assertEqual(self.store.webhook_id(100, self.alice), new.id)

    async def test_webhook_that_keeps_returning_not_found_fails_turn(self):
        """Characterization (REL-04): when every webhook is gone by the time it is used, the turn fails as a
        webhook-delivery error after one retry (no unbounded loop)."""
        self.channel.dead_on_create = True
        logged = await self.failed_turn()
        self.assertEqual(len(self.channel.errors), 1)
        self.assertIn("webhook delivery", logged)
        self.assertIn("Unknown Webhook", logged)
        self.assertEqual(self.channel.creates, 2)

    # --- REL-04 (fixed): cached webhooks -------------------------------------------------------

    async def test_later_turn_reuses_cached_webhook_without_listing(self):
        """REL-04 (fixed): after a webhook is found or created, later turns for the same channel+character reuse it
        without calling parent_channel.webhooks() again."""
        await self.turn()
        listings = self.channel.listings
        await self.turn()
        await self.turn()
        self.assertEqual(self.channel.listings - listings, 0)
        self.assertEqual(self.channel.creates, 1)
        self.assertEqual(len(self.channel.hooks[0].posts), 3)

    async def test_cached_webhook_is_renamed_without_listing(self):
        """REL-04 (fixed): a cached webhook is still edited when the character identity changes, without a listing."""
        await self.turn()
        hook = self.channel.hooks[0]
        self.rename("Alicia")
        await self.turn()
        self.assertEqual(hook.edits, [{"name": "Alicia", "avatar": None}])
        self.assertEqual(self.channel.listings, 0)
        self.assert_delivered_by(hook, start=1)

    async def test_placeholder_not_found_recreates_webhook_and_retries(self):
        """REL-04 (fixed): if the resolved webhook was deleted on Discord, the placeholder send's NotFound drops it,
        a replacement is looked up/created once, the send is retried, and the new id is stored."""
        await self.turn()
        old = self.channel.hooks[0]
        self.channel.doom(old)
        await self.turn()
        self.assertEqual(self.channel.errors, [])
        self.assertTrue(old.dead)
        new = self.channel.hooks[0]
        self.assertIsNot(new, old)
        self.assert_delivered_by(new)
        self.assertEqual(self.store.webhook_id(100, self.alice), new.id)

    async def test_placeholder_not_found_retry_delivers_every_chunk_through_new_webhook(self):
        """Characterization (ARCH-01): when the placeholder send hits NotFound on a multi-chunk reply, the retry
        replaces the webhook and the placeholder edit and every extra chunk go through the new one; the old
        webhook gets no posts and is forgotten."""
        self.bot.engine.models = self.bot.models = LongModels([self.alice])
        await self.turn()
        old = self.channel.hooks[0]
        old_posts = len(old.posts)
        self.assertGreater(old_posts, 1)
        self.channel.doom(old)
        await self.turn()
        self.assertEqual(self.channel.errors, [])
        self.assertTrue(old.dead)
        self.assertEqual(len(old.posts), old_posts)
        self.assertNotIn(old, self.channel.hooks)
        new = self.channel.hooks[0]
        self.assertIsNot(new, old)
        self.assertEqual(self.channel.creates, 2)
        self.assertEqual(len(new.posts), old_posts)
        self.assertGreater(len(new.posts), 1)
        self.assertEqual(self.store.webhook_id(100, self.alice), new.id)

    async def test_edit_not_found_recreates_webhook(self):
        """REL-04 (fixed): NotFound from webhook.edit (identity change on a deleted webhook) drops it and proceeds
        to lookup/create, so the turn still delivers under the new name."""
        await self.turn()
        old = self.channel.hooks[0]
        self.rename("Alicia")
        self.channel.doom(old)
        await self.turn()
        self.assertEqual(self.channel.errors, [])
        new = self.channel.hooks[0]
        self.assertIsNot(new, old)
        self.assert_delivered_by(new)
        self.assertEqual(new.options[-1]["username"], "Alicia")
        self.assertEqual(self.store.webhook_id(100, self.alice), new.id)


    # --- MNT-02: only a gone webhook is forgotten ----------------------------------------------

    @staticmethod
    def unauthorized():
        return discord.HTTPException(SimpleNamespace(status=401, reason="Unauthorized"),
                                     {"message": "Invalid Webhook Token", "code": 50027})

    def fail_sends(self, hook, error):
        """Make the hook's next send raise ``error`` once (a placeholder send)."""
        original, state = hook.send, {"left": 1}

        async def send(content, **kwargs):
            if state["left"]:
                state["left"] -= 1
                raise error
            return await original(content, **kwargs)
        hook.send = send

    def fail_placeholder_edits(self, hook, error):
        original = hook.send

        async def send(content, **kwargs):
            message = await original(content, **kwargs)

            async def edit(**_kwargs):
                raise error
            message.edit = edit
            return message
        hook.send = send
        return original

    async def test_unauthorized_placeholder_send_drops_cache_and_retries(self):
        """MNT-02: an HTTP 401 (invalid webhook token) on the placeholder send forgets the cached webhook, looks
        one up again and retries the send."""
        await self.turn()
        hook = self.channel.hooks[0]
        self.fail_sends(hook, self.unauthorized())
        await self.turn()
        self.assertEqual(self.channel.errors, [])
        self.assertEqual(self.channel.listings, 1)
        self.assert_delivered_by(self.channel.hooks[0], start=1)

    async def test_unknown_message_on_placeholder_send_keeps_webhook(self):
        """MNT-02: a NotFound that is not Unknown Webhook (10008) on the placeholder send fails the turn without
        retrying and without dropping the cached webhook."""
        await self.turn()
        hook = self.channel.hooks[0]
        self.fail_sends(hook, not_found("Unknown Message", 10008))
        await self.failed_turn()
        self.assertEqual(len(self.channel.errors), 1)
        self.assertEqual(self.channel.listings, 0)
        await self.turn()
        self.assertEqual(self.channel.listings, 0)
        self.assertEqual(self.channel.creates, 1)

    async def test_unknown_message_on_placeholder_edit_keeps_webhook(self):
        """MNT-02: someone deleting the placeholder mid-stream (NotFound 10008 on edit) fails the turn but the
        cached webhook is kept."""
        await self.turn()
        hook = self.channel.hooks[0]
        original = self.fail_placeholder_edits(hook, not_found("Unknown Message", 10008))
        await self.failed_turn()
        self.assertEqual(len(self.channel.errors), 1)
        hook.send = original
        await self.turn()
        self.assertEqual(self.channel.listings, 0)
        self.assertEqual(self.channel.creates, 1)

    async def test_unknown_webhook_on_placeholder_edit_drops_cache(self):
        """MNT-02: NotFound 10015 on the placeholder edit fails the turn and forgets the cached webhook, so the
        next turn looks one up again."""
        await self.turn()
        hook = self.channel.hooks[0]
        original = self.fail_placeholder_edits(hook, not_found())
        await self.failed_turn()
        hook.send = original
        await self.turn()
        self.assertEqual(self.channel.listings, 1)

    async def test_unauthorized_on_placeholder_edit_drops_cache(self):
        """MNT-02: HTTP 401 on the placeholder edit forgets the cached webhook."""
        await self.turn()
        hook = self.channel.hooks[0]
        original = self.fail_placeholder_edits(hook, self.unauthorized())
        await self.failed_turn()
        hook.send = original
        await self.turn()
        self.assertEqual(self.channel.listings, 1)


    def fail_chunks(self, hook, error):
        original = hook.send

        async def send(content, **kwargs):
            if content != "…":
                raise error
            return await original(content, **kwargs)
        hook.send = send

    async def test_unknown_message_on_continuation_chunk_keeps_webhook(self):
        """MNT-02: a non-10015 NotFound on a later chunk fails the turn (re-raised) but keeps the cached webhook."""
        self.bot.engine.models = self.bot.models = LongModels([self.alice])
        await self.turn()
        self.fail_chunks(self.channel.hooks[0], not_found("Unknown Message", 10008))
        await self.failed_turn()
        self.assertEqual(len(self.channel.errors), 1)
        self.assertEqual(len(self.bot.webhooks), 1)

    async def test_unauthorized_on_continuation_chunk_drops_webhook(self):
        """MNT-02: HTTP 401 on a later chunk forgets the cached webhook and re-raises."""
        self.bot.engine.models = self.bot.models = LongModels([self.alice])
        await self.turn()
        self.fail_chunks(self.channel.hooks[0], self.unauthorized())
        await self.failed_turn()
        self.assertEqual(len(self.channel.errors), 1)
        self.assertEqual(len(self.bot.webhooks), 0)

    async def test_webhook_identity_reraises_other_not_found(self):
        """MNT-02: webhook.edit failing with a NotFound that is not Unknown Webhook propagates and keeps the cache."""
        await self.turn()
        hook = self.channel.hooks[0]

        async def edit(**kwargs):
            raise not_found("Unknown Message", 10008)
        hook.edit = edit
        self.rename("Alicia")
        await self.failed_turn()
        self.assertEqual(len(self.bot.webhooks), 1)

    async def test_other_not_found_on_placeholder_send_is_not_retried(self):
        """MNT-02: NotFound 10003 (Unknown Channel) on the placeholder send is not retried."""
        await self.turn()
        hook = self.channel.hooks[0]
        self.fail_sends(hook, not_found("Unknown Channel", 10003))
        await self.failed_turn()
        self.assertEqual(self.channel.listings, 0)
        self.assertEqual(self.channel.creates, 1)
        self.assertEqual(len(self.bot.webhooks), 1)


class AvatarAssetCheckTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.channel = FakeTextChannel(100)
        self.bot.get_channel = lambda _ident: self.channel
        self.asset = self.store.execute(
            "INSERT INTO avatar_assets(guild_id,character_id,slot_key,image_hash,channel_id,message_id,url,created_at)"
            " VALUES(?,?,?,?,?,?,?,?)", (1, 1, "happy", "hash", 100, 701, "https://cdn.example/701.png", 0))
        self.selected = {"slot_key": "happy", "asset_id": self.asset, "url": "https://cdn.example/701.png"}

    def tearDown(self):
        self.store.close()

    async def test_asset_check_is_reused_shortly_after(self):
        """Characterization (REL-04): a verified asset is not re-fetched on the next use moments later."""
        self.assertEqual(await self.bot.resolve_avatar(self.selected), self.selected)
        self.assertEqual(await self.bot.resolve_avatar(self.selected), self.selected)
        self.assertEqual(self.channel.fetches, 1)

    async def test_missing_asset_falls_back_to_static(self):
        """Characterization (REL-04): an asset whose message is gone resolves to the static avatar."""
        self.channel.missing.add(701)
        resolved = await self.bot.resolve_avatar(self.selected)
        self.assertIsNone(resolved["url"])
        self.assertIsNone(resolved["asset_id"])

    async def test_avatar_asset_check_expires_after_ttl(self):
        """REL-04 (fixed): asset checks expire (TTL 600 s, monotonic clock); an asset message deleted after
        it was verified falls back to the static avatar once the TTL has passed."""
        self.assertEqual(await self.bot.resolve_avatar(self.selected), self.selected)
        self.channel.missing.add(701)
        with ShiftedClock() as clock:
            clock.advance(86400)
            resolved = await self.bot.resolve_avatar(self.selected)
        self.assertEqual(self.channel.fetches, 2)
        self.assertIsNone(resolved["url"])
        self.assertIsNone(resolved["asset_id"])


if __name__ == "__main__":
    unittest.main()
