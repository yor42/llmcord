"""FEAT-22 part C: /shop, /inventory and /use replies, buttons, autocomplete and isolation."""
import unittest
from types import SimpleNamespace
from unittest import mock

import discord

from helpers import FakeInteraction, FakeModels, autocomplete, invoke, make_settings

from llmcord_core.discord_bot import SkitBot
from llmcord_core.items_discord import PALETTE, color_for

G, G2, U, U2 = 1, 2, 9, 10
BALL, COOKIE, DICE = '8ball', 'fortune_cookie', 'dice'
NO_ITEMS = 'You are not holding any items. Visit /shop to buy some.'


class FakeButtonInteraction(FakeInteraction):
    """A component interaction: adds the message-edit response."""

    def __init__(self, user_id=U, guild_id=G, **kwargs):
        super().__init__(guild_id=guild_id, user_id=user_id, **kwargs)
        self.edited, self.posts, self.originals = [], [], []

        async def edit_message(**kw):
            self.edited.append(kw)
            self.response._done = True

        async def send(**kw):
            self.posts.append(kw)

        async def edit_original_response(**kw):
            self.originals.append(kw)
        self.response.edit_message = edit_message
        self.channel.send = send
        self.edit_original_response = edit_original_response


def sent_embed(interaction, index=0):
    return interaction.response.sent[index][1]['embed']


class ItemsBase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        for key in (BALL, COOKIE, DICE):
            self.store.save_shop_item(G, key, expected_revision=0, for_sale=True)
        self.store.set_currency_name(G, 'gold')
        self.store.change_balance(G, U, 1000, 'start', 1)

    def tearDown(self):
        self.store.close()

    async def run_cmd(self, path, *args, user_id=U, guild_id=G, **kwargs):
        interaction = FakeInteraction(guild_id=guild_id, user_id=user_id)
        await invoke(self.bot, path, interaction, *args, **kwargs)
        return interaction

    async def open_shop(self, user_id=U):
        shop = await self.run_cmd('shop', user_id=user_id)
        return shop, shop.response.sent[0][1]['view']

    @staticmethod
    def button(view, text):
        return next(b for b in view.children if text in b.label)

    @staticmethod
    def capture_edits(origin):
        origin.edits = []

        async def edit_original_response(**kw):
            origin.edits.append(kw)
        origin.edit_original_response = edit_original_response


class ShopTests(ItemsBase):
    async def test_listing_content_and_buttons(self):
        shop = await self.run_cmd('shop')
        reply = shop.response.sent[0][1]
        self.assertTrue(reply['ephemeral'])
        self.assertEqual(reply['allowed_mentions'].everyone, False)
        embed = reply['embed']
        self.assertEqual(embed.description, 'Your balance: 1,000 gold')
        names = [f.name for f in embed.fields]
        self.assertIn('🎱 Magic 8-ball · 150 gold', names)
        self.assertNotIn('Tarot', ' '.join(names))  # not for sale
        field = next(f for f in embed.fields if 'Magic 8-ball' in f.name)
        self.assertEqual(field.value, 'Ask a question and get an answer.\nreusable · You hold none')
        labels = [b.label for b in reply['view'].children]
        self.assertIn('Buy 🎱 Magic 8-ball · 150', labels)
        self.assertEqual(len(labels), 3)
        self.assertFalse(any(b.disabled for b in reply['view'].children))

    async def test_empty_shop(self):
        for key in (BALL, COOKIE, DICE):
            self.store.save_shop_item(G, key, expected_revision=1, for_sale=False)
        shop = await self.run_cmd('shop')
        self.assertEqual(sent_embed(shop).description, 'Nothing is for sale on this server right now.\nYour balance: 1,000 gold')
        self.assertEqual(len(shop.response.sent[0][1]['view'].children), 0)

    async def test_disabled_states(self):
        self.store.change_balance(G, U, -900, 'spend', 1)  # 100 gold left; the 8-ball costs 150
        self.store.buy_item(G, U, DICE)
        self.store.set_item_hold_cap(G, 1)
        self.store.buy_item(G, U, COOKIE)
        _, view = await self.open_shop()
        broke = self.button(view, 'Need')
        self.assertEqual((broke.label, broke.disabled), ('Need 125 gold more', True))
        self.assertTrue(self.button(view, 'Owned').disabled)
        full = self.button(view, 'At the limit')
        self.assertEqual((full.label, full.disabled), ('At the limit (1)', True))

    async def test_buy_button_updates_message(self):
        _, view = await self.open_shop()
        click = FakeButtonInteraction()
        await self.button(view, 'Fortune cookie').callback(click)
        edit = click.edited[0]
        self.assertEqual(edit['content'], 'Bought 🥠 Fortune cookie for 25 gold.')
        self.assertEqual(edit['embed'].description, 'Your balance: 975 gold')
        self.assertEqual(self.store.item_qty(G, U, COOKIE), 1)
        self.assertEqual(self.store.balance(G, U), 975)
        cookie = next(f for f in edit['embed'].fields if 'Fortune cookie' in f.name)
        self.assertTrue(cookie.value.endswith('consumable · You hold 1'))

    async def test_buy_reusable_then_owned(self):
        _, view = await self.open_shop()
        click = FakeButtonInteraction()
        await self.button(view, 'Magic 8-ball').callback(click)
        self.assertTrue(self.button(click.edited[0]['view'], 'Owned').disabled)

    async def test_typed_errors_are_friendly(self):
        _, view = await self.open_shop()
        self.store.change_balance(G, U, -900, 'spend', 1)
        click = FakeButtonInteraction()
        await self.button(view, 'Magic 8-ball').callback(click)
        self.assertEqual(click.edited[0]['content'], 'Magic 8-ball costs 150 gold; you have 100.')
        self.store.change_balance(G, U, 1000, 'more', 1)
        self.store.buy_item(G, U, BALL)
        click = FakeButtonInteraction()
        await self.button(view, 'Magic 8-ball').callback(click)
        self.assertEqual(click.edited[0]['content'], 'You already own Magic 8-ball.')
        self.store.set_item_hold_cap(G, 1)
        self.store.buy_item(G, U, COOKIE)
        click = FakeButtonInteraction()
        await self.button(view, 'Fortune cookie').callback(click)
        self.assertEqual(click.edited[0]['content'], 'You can hold at most 1 of Fortune cookie.')
        self.store.save_shop_item(G, DICE, expected_revision=1, for_sale=False)
        click = FakeButtonInteraction()
        await self.button(view, 'Dice').callback(click)
        self.assertEqual(click.edited[0]['content'], 'Dice is not for sale right now.')

    async def test_other_member_cannot_press(self):
        _, view = await self.open_shop()
        stranger = FakeButtonInteraction(user_id=U2)
        self.assertFalse(await view.interaction_check(stranger))
        self.assertTrue(stranger.response.sent[0][1]['ephemeral'])
        self.assertTrue(await view.interaction_check(FakeButtonInteraction()))
        self.assertFalse(await view.interaction_check(FakeButtonInteraction(guild_id=G2)))

    async def test_timeout_disables_buttons(self):
        origin, view = await self.open_shop()
        self.capture_edits(origin)
        await view.on_timeout()
        self.assertTrue(all(b.disabled for b in view.children))
        self.assertIs(origin.edits[0]['view'], view)

    async def test_other_guild_shop_is_independent(self):
        shop = await self.run_cmd('shop', guild_id=G2)
        self.assertEqual(sent_embed(shop).description, 'Nothing is for sale on this server right now.\nYour balance: 0 coins')

    async def test_outside_guild(self):
        self.assertEqual((await self.run_cmd('shop', guild_id=0)).replies, ['Use this command in a server channel.'])


class InventoryTests(ItemsBase):
    async def test_empty_points_to_shop(self):
        inv = await self.run_cmd('inventory')
        self.assertEqual(inv.replies, [NO_ITEMS])
        self.assertTrue(inv.response.sent[0][1]['ephemeral'])

    async def test_listing_and_use_button(self):
        self.store.buy_item(G, U, BALL)
        self.store.buy_item(G, U, COOKIE)
        self.store.buy_item(G, U, COOKIE)
        inv = await self.run_cmd('inventory')
        reply = inv.response.sent[0][1]
        fields = {f.name: f.value for f in reply['embed'].fields}
        self.assertEqual(fields['🎱 Magic 8-ball × 1'], 'reusable · ready')
        self.assertEqual(fields['🥠 Fortune cookie × 2'], 'consumable')
        click = FakeButtonInteraction()
        await self.button(reply['view'], 'Fortune cookie').callback(click)
        public = click.posts[0]['embed']
        self.assertEqual(public.title, '🥠 Fortune cookie')
        self.assertIn('1 left', public.footer.text)
        self.assertEqual(click.posts[0]['allowed_mentions'].everyone, False)
        self.assertEqual(click.response.sent, [])  # acknowledged with a defer, nothing sent through the ephemeral response
        self.assertIn('🥠 Fortune cookie × 1', [f.name for f in click.originals[0]['embed'].fields])

    async def test_button_send_failure_is_private(self):
        self.store.buy_item(G, U, COOKIE)
        inv = await self.run_cmd('inventory')
        click = FakeButtonInteraction()

        async def boom(**kw):
            raise discord.Forbidden(SimpleNamespace(status=403, reason='x'), 'no')
        click.channel.send = boom
        with self.assertLogs(level='WARNING'):
            await self.button(inv.response.sent[0][1]['view'], 'Fortune cookie').callback(click)
        self.assertEqual(click.replies, ["Your Fortune cookie was used, but the result could not be posted. It is saved in the use log an admin can see."])
        self.assertTrue(click.followup.sent[0][1]['ephemeral'])
        self.assertEqual(self.store.item_qty(G, U, COOKIE), 0)

    async def test_button_without_channel_is_private(self):
        self.store.buy_item(G, U, COOKIE)
        inv = await self.run_cmd('inventory')
        click = FakeButtonInteraction()
        click.channel = None
        with self.assertLogs(level='WARNING'):
            await self.button(inv.response.sent[0][1]['view'], 'Fortune cookie').callback(click)
        self.assertIn('could not be posted', click.replies[0])

    async def test_button_refresh_failure_is_private(self):
        self.store.buy_item(G, U, COOKIE)
        inv = await self.run_cmd('inventory')
        click = FakeButtonInteraction()

        async def boom(**kw):
            raise RuntimeError('edit failed')
        click.edit_original_response = boom
        with self.assertLogs(level='WARNING'):
            await self.button(inv.response.sent[0][1]['view'], 'Fortune cookie').callback(click)
        self.assertEqual(len(click.posts), 1)
        self.assertEqual(click.replies, ['Something went wrong with Fortune cookie. Check /inventory and try again.'])

    async def test_button_refusal_is_private(self):
        self.store.buy_item(G, U, BALL)
        inv = await self.run_cmd('inventory')
        view = inv.response.sent[0][1]['view']
        await self.button(view, 'Magic 8-ball').callback(FakeButtonInteraction())
        click = FakeButtonInteraction()
        await self.button(view, 'Magic 8-ball').callback(click)
        self.assertRegex(click.replies[0], r'^Magic 8-ball is recharging\. Try again in \d+ s\.$')
        self.assertTrue(click.followup.sent[0][1]['ephemeral'])
        self.assertEqual(click.posts, [])

    async def test_cooldown_state_and_use_off(self):
        self.store.buy_item(G, U, BALL)
        self.store.use_item(G, U, 100, BALL)
        inv = await self.run_cmd('inventory')
        self.assertIn('recharging, ready <t:', inv.response.sent[0][1]['embed'].fields[0].value)
        self.store.save_shop_item(G, BALL, expected_revision=1, for_sale=True, allow_use=False)
        inv = await self.run_cmd('inventory')
        self.assertIn('use is off right now', inv.response.sent[0][1]['embed'].fields[0].value)
        self.assertEqual(len(inv.response.sent[0][1]['view'].children), 0)

    async def test_inventory_is_guild_scoped(self):
        self.store.buy_item(G, U, BALL)
        self.assertEqual((await self.run_cmd('inventory', guild_id=G2)).replies, [NO_ITEMS])


class UseTests(ItemsBase):
    async def test_public_embed_fields(self):
        self.store.buy_item(G, U, COOKIE)
        use = await self.run_cmd('use', 'Fortune cookie')
        reply = use.response.sent[0][1]
        self.assertNotIn('ephemeral', reply)
        self.assertEqual((reply['allowed_mentions'].everyone, reply['allowed_mentions'].users, reply['allowed_mentions'].roles), (False, False, False))
        embed = reply['embed']
        self.assertEqual(embed.title, '🥠 Fortune cookie')
        self.assertTrue(embed.description)
        self.assertTrue(embed.footer.text.startswith('Toy shop'))
        self.assertTrue(embed.footer.text.endswith('0 left'))
        self.assertEqual(self.store.item_qty(G, U, COOKIE), 0)
        self.assertEqual(len(self.store.item_use_log(G)), 1)

    async def test_reusable_footer_and_color(self):
        self.store.buy_item(G, U, DICE)
        embed = sent_embed(await self.run_cmd('use', 'dice'))
        self.assertNotIn('left', embed.footer.text)
        self.assertIn(embed.color.value, PALETTE)
        self.assertEqual(embed.color.value, color_for(DICE))

    async def test_not_held_cooldown_and_disabled(self):
        miss = await self.run_cmd('use', 'Magic 8-ball')
        self.assertEqual(miss.replies, ['You do not have Magic 8-ball. Buy one with /shop.'])
        self.assertTrue(miss.response.sent[0][1]['ephemeral'])
        self.store.buy_item(G, U, BALL)
        await self.run_cmd('use', 'Magic 8-ball')
        wait = await self.run_cmd('use', 'Magic 8-ball')
        self.assertRegex(wait.replies[0], r'^Magic 8-ball is recharging\. Try again in \d+ s\.$')
        self.assertTrue(wait.response.sent[0][1]['ephemeral'])
        self.store.save_shop_item(G, BALL, expected_revision=1, for_sale=True, allow_use=False)
        off = await self.run_cmd('use', 'Magic 8-ball')
        self.assertEqual(off.replies, ['Magic 8-ball cannot be used on this server right now.'])
        self.assertTrue(off.response.sent[0][1]['ephemeral'])

    async def test_cross_guild_isolation(self):
        self.store.buy_item(G, U, BALL)
        other = await self.run_cmd('use', 'Magic 8-ball', guild_id=G2)
        self.assertEqual(other.replies, ['You do not have Magic 8-ball. Buy one with /shop.'])
        self.assertEqual(self.store.item_use_log(G2), [])

    async def test_outside_guild(self):
        self.assertEqual((await self.run_cmd('use', 'dice', guild_id=0)).replies, ['Use this command in a server channel.'])

    async def test_send_failure_is_logged_and_told_privately(self):
        self.store.buy_item(G, U, COOKIE)
        interaction = FakeInteraction()

        async def failing(content=None, **kw):
            if 'embed' in kw:
                raise discord.HTTPException(SimpleNamespace(status=500, reason='x'), 'boom')
            interaction.response.sent.append((content, kw))
        interaction.response.send_message = failing
        with self.assertLogs(level='WARNING') as logs:
            await invoke(self.bot, 'use', interaction, 'Fortune cookie')
        self.assertTrue(any('Could not post' in line for line in logs.output))
        self.assertEqual(interaction.replies, ["Your Fortune cookie was used, but the result could not be posted. It is saved in the use log an admin can see."])
        self.assertTrue(interaction.response.sent[0][1]['ephemeral'])
        self.assertEqual(self.store.item_qty(G, U, COOKIE), 0)
        self.assertEqual(len(self.store.item_use_log(G)), 1)

    def post_rows(self):
        return [row['use_id'] for row in self.store.all('SELECT use_id FROM item_use_posts')]

    async def test_slash_use_marks_the_use_posted_after_the_public_send(self):
        """MNT-84: a use whose public post succeeded gets a post row (so the cast may see it)."""
        self.store.buy_item(G, U, COOKIE)
        await self.run_cmd('use', 'Fortune cookie')
        use_id = self.store.item_use_log(G)[0]['id']
        self.assertEqual(self.post_rows(), [use_id])

    async def test_slash_use_with_failed_public_send_leaves_no_post_row(self):
        """MNT-84: a failed public send leaves the use unposted, so the cast never sees a result nobody read."""
        self.store.buy_item(G, U, COOKIE)
        interaction = FakeInteraction()

        async def failing(content=None, **kw):
            if 'embed' in kw:
                raise discord.HTTPException(SimpleNamespace(status=500, reason='x'), 'boom')
            interaction.response.sent.append((content, kw))
        interaction.response.send_message = failing
        with self.assertLogs(level='WARNING'):
            await invoke(self.bot, 'use', interaction, 'Fortune cookie')
        self.assertEqual(len(self.store.item_use_log(G)), 1)
        self.assertEqual(self.post_rows(), [])

    async def test_refused_use_creates_no_use_and_no_post_row(self):
        """MNT-84: a refusal (not held) writes neither a use nor a post row."""
        await self.run_cmd('use', 'dice')
        self.assertEqual(self.store.item_use_log(G), [])
        self.assertEqual(self.post_rows(), [])

    async def test_button_use_marks_posted_after_the_channel_message(self):
        """MNT-84: the menu-button (via_channel) path also records the post."""
        self.store.buy_item(G, U, COOKIE)
        inv = await self.run_cmd('inventory')
        click = FakeButtonInteraction()
        await self.button(inv.response.sent[0][1]['view'], 'Fortune cookie').callback(click)
        self.assertEqual(len(click.posts), 1)
        self.assertEqual(self.post_rows(), [self.store.item_use_log(G)[0]['id']])

    async def test_button_use_with_failed_channel_send_leaves_no_post_row(self):
        """MNT-84: the via_channel path marks nothing when the channel send raises."""
        self.store.buy_item(G, U, COOKIE)
        inv = await self.run_cmd('inventory')
        click = FakeButtonInteraction()

        async def boom(**kw):
            raise discord.Forbidden(SimpleNamespace(status=403, reason='x'), 'no')
        click.channel.send = boom
        with self.assertLogs(level='WARNING'):
            await self.button(inv.response.sent[0][1]['view'], 'Fortune cookie').callback(click)
        self.assertEqual(len(self.store.item_use_log(G)), 1)
        self.assertEqual(self.post_rows(), [])

    async def test_autocomplete_held_only_and_guild_scoped(self):
        self.store.buy_item(G, U, BALL)
        self.store.buy_item(G, U, DICE)
        self.store.change_balance(G2, U, 100, 'x', 1)
        self.store.save_shop_item(G2, COOKIE, expected_revision=0, for_sale=True)
        self.store.buy_item(G2, U, COOKIE)
        choices = await autocomplete(self.bot, 'use', 'item', FakeInteraction(), '')
        self.assertEqual([c.value for c in choices], ['Magic 8-ball', 'Dice'])
        self.assertEqual([c.value for c in await autocomplete(self.bot, 'use', 'item', FakeInteraction(), 'DIC')], ['Dice'])
        self.assertEqual([c.value for c in await autocomplete(self.bot, 'use', 'item', FakeInteraction(guild_id=G2), '')], ['Fortune cookie'])
        self.assertEqual(await autocomplete(self.bot, 'use', 'item', FakeInteraction(user_id=U2), ''), [])
        self.assertEqual(await autocomplete(self.bot, 'use', 'item', FakeInteraction(guild_id=0), ''), [])

    async def test_never_calls_the_model(self):
        self.store.buy_item(G, U, DICE)
        with mock.patch.object(FakeModels, 'text', side_effect=AssertionError) as text:
            await self.run_cmd('use', 'dice')
        text.assert_not_called()


class LimitsTests(ItemsBase):
    def crowd(self, count, **changes):
        """`count` extra catalog items cloned from the dice recipe, each with a very long name and emoji."""
        from dataclasses import replace

        from llmcord_core.items import catalog
        base = catalog._ITEMS[DICE]
        extra = {f'toy{n:02d}': replace(base, key=f'toy{n:02d}', name=f'Toy {n:02d} ' + 'x' * 30, emoji='🧸' * 8, default_price=10 + n, **changes)
                 for n in range(count)}
        patcher = mock.patch.dict(catalog._ITEMS, extra)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in extra:
            self.store.save_shop_item(G, key, expected_revision=0, for_sale=True)
        return list(extra)

    async def test_shop_with_30_items_caps_fields_and_buttons(self):
        self.crowd(30)
        shop = await self.run_cmd('shop')
        embed, view = sent_embed(shop), shop.response.sent[0][1]['view']
        self.assertEqual(len(embed.fields), 25)
        self.assertEqual(len(view.children), 25)
        self.assertLess(len(embed), 6000)
        self.assertRegex(embed.description, r'\+8 more not shown$')  # 33 for sale, 25 shown
        for button in view.children:
            self.assertLessEqual(len(button.label), 80)
            self.assertLessEqual(len(button.custom_id), 100)

    async def test_inventory_with_30_items_caps_fields_and_buttons(self):
        keys = self.crowd(30)
        self.store.change_balance(G, U, 100000, 'rich', 1)
        for key in keys:
            self.store.buy_item(G, U, key)
        inv = await self.run_cmd('inventory')
        embed, view = sent_embed(inv), inv.response.sent[0][1]['view']
        self.assertEqual(len(embed.fields), 25)
        self.assertLess(len(embed), 6000)
        self.assertIn('more not shown', embed.description)
        self.assertEqual(len(view.children), 25)
        self.assertTrue(all(len(b.label) <= 80 for b in view.children))

    async def test_price_in_label_and_changed_price_refuses(self):
        _, view = await self.open_shop()
        buy = self.button(view, 'Magic 8-ball')
        self.assertEqual(buy.label, 'Buy 🎱 Magic 8-ball · 150')
        self.store.save_shop_item(G, BALL, expected_revision=1, for_sale=True, price=500)
        click = FakeButtonInteraction()
        await buy.callback(click)
        self.assertEqual(click.edited[0]['content'], 'The price changed to 500. Open /shop again.')
        self.assertEqual(self.store.balance(G, U), 1000)
        self.assertEqual(self.store.item_qty(G, U, BALL), 0)
        self.assertEqual(self.button(click.edited[0]['view'], 'Magic 8-ball').label, 'Buy 🎱 Magic 8-ball · 500')

    async def test_admin_text_renders_without_pinging(self):
        text = '@everyone <@123> **bold** [x](http://example.invalid)'
        self.store.save_shop_item(G, BALL, expected_revision=1, for_sale=True, replace_entries={'en': [text]})
        self.store.buy_item(G, U, BALL)
        inv = await self.run_cmd('inventory')
        click = FakeButtonInteraction()
        await self.button(inv.response.sent[0][1]['view'], 'Magic 8-ball').callback(click)
        self.assertIn('@everyone <@123> **bold**', click.posts[0]['embed'].description)
        mentions = click.posts[0]['allowed_mentions']
        self.assertEqual((mentions.everyone, mentions.users, mentions.roles), (False, False, False))
        self.store.change_balance(G, U2, 500, 'start', 1)
        self.store.buy_item(G, U2, BALL)
        slash = await self.run_cmd('use', 'Magic 8-ball', user_id=U2)
        self.assertIn('@everyone', sent_embed(slash).description)
        self.assertEqual(slash.response.sent[0][1]['allowed_mentions'].everyone, False)
        self.assertEqual(slash.response.sent[0][1]['allowed_mentions'].users, False)

    async def test_buy_menu_edit_failure_after_a_successful_buy(self):
        _, view = await self.open_shop()
        click = FakeButtonInteraction()

        async def boom(**kw):
            raise RuntimeError('edit failed')
        click.response.edit_message = boom
        with self.assertLogs(level='WARNING'):
            await self.button(view, 'Fortune cookie').callback(click)
        self.assertEqual(self.store.item_qty(G, U, COOKIE), 1)
        self.assertEqual(click.replies, ['Bought 🥠 Fortune cookie for 25 gold. The menu could not be refreshed. Open /shop again.'])
        self.assertTrue(click.response.sent[0][1]['ephemeral'])


if __name__ == '__main__':
    unittest.main()
