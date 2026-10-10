"""Bounded per-channel caches and lock pools on SkitBot (MNT-02)."""
import asyncio
import unittest

from llmcord_core.discord_bot import CACHE_LIMIT, LockPool, LruDict, SkitBot
from helpers import make_settings


class LruDictTests(unittest.TestCase):
    def test_oldest_entry_is_evicted_past_the_cap(self):
        """MNT-02: an LruDict never grows past its cap and drops the least recently used key first."""
        cache = LruDict(3)
        for key in "abc":
            cache[key] = key
        self.assertEqual(cache.get("a"), "a")  # a becomes most recent
        cache["d"] = "d"
        self.assertEqual(sorted(cache), ["a", "c", "d"])
        self.assertNotIn("b", cache)

    def test_dict_operations_still_work(self):
        """MNT-02: get/setitem/pop/contains keep today's dict semantics."""
        cache = LruDict(2)
        cache["a"] = 1
        self.assertEqual((cache["a"], cache.get("x"), "a" in cache), (1, None, True))
        self.assertEqual(cache.pop("a", None), 1)
        self.assertIsNone(cache.pop("a", None))

    def test_default_cap_is_module_constant(self):
        """MNT-02: the bot's caches use CACHE_LIMIT."""
        self.assertEqual(LruDict().cap, CACHE_LIMIT)


class BotWiringTests(unittest.TestCase):
    def test_bot_uses_bounded_containers(self):
        """MNT-02: SkitBot's per-channel caches are LruDict and its lock maps are LockPool."""
        bot = SkitBot(make_settings())
        self.addCleanup(bot.store.close)
        for name in ("webhooks", "webhook_defaults", "checked_avatar_assets"):
            self.assertIsInstance(getattr(bot, name), LruDict, name)
        for name in ("channel_locks", "webhook_locks"):
            self.assertIsInstance(getattr(bot, name), LockPool, name)


class LockPoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_key_same_lock(self):
        """MNT-02: lock(key) returns the same Lock object every time while the key is cached."""
        pool = LockPool(4)
        self.assertIs(pool.lock(1), pool.lock(1))

    async def test_idle_locks_are_dropped_past_the_cap(self):
        """MNT-02: past the cap the pool drops idle locks, oldest first."""
        pool = LockPool(3)
        for key in range(6):
            pool.lock(key)
        self.assertEqual(len(pool), 3)
        self.assertEqual(sorted(pool), [3, 4, 5])

    async def test_held_lock_is_never_evicted(self):
        """MNT-02: a held lock survives eviction pressure and the same key keeps returning it, so mutual
        exclusion holds."""
        pool = LockPool(2)
        held = pool.lock("held")
        async with held:
            for key in range(10):
                pool.lock(key)
            self.assertIs(pool.lock("held"), held)
            self.assertTrue(pool.lock("held").locked())

    async def test_lock_with_waiters_is_never_evicted(self):
        """MNT-02: a lock that is idle only because its release is waking a waiter is not evicted."""
        pool = LockPool(2)
        lock = pool.lock("busy")
        order = []

        async def waiter():
            async with pool.lock("busy"):
                order.append("waiter")

        await lock.acquire()
        task = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        lock.release()  # waiter woken but not yet running: locked() is False
        for key in range(10):
            pool.lock(key)
        self.assertIs(pool.lock("busy"), lock)
        await task
        self.assertEqual(order, ["waiter"])


if __name__ == "__main__":
    unittest.main()
