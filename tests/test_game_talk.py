"""MNT-59: one failed-attempt counter per (guild, server day), shared by every table game."""
import unittest
from types import SimpleNamespace

from llmcord_core.game_talk import TalkBudget
from llmcord_core.games_discord import GameTables
from llmcord_core.games_doubt_discord import DoubtTables
from llmcord_core.games_holdem_discord import HoldemTables


class FakeStore:
    def __init__(self, limit=3, calls=0):
        self.limit, self.calls, self.days = limit, calls, {}

    def _server_day(self, guild_id, now):
        return (None, self.days.get(guild_id, "d1"))

    def game_talk_calls_today(self, guild_id):
        return self.calls

    def game_talk_daily_limit(self, guild_id):
        return self.limit


class TalkBudgetTests(unittest.TestCase):
    def test_failures_count_against_limit_per_guild(self):
        budget = TalkBudget(FakeStore(limit=3, calls=1))
        self.assertEqual(budget.remaining(1), 2)
        budget.note_failure(1)
        self.assertEqual((budget.remaining(1), budget.remaining(2)), (1, 2))
        budget.note_failure(1)
        budget.note_failure(1)
        self.assertEqual(budget.remaining(1), 0)

    def test_reading_one_guild_keeps_the_others_counts(self):
        budget = TalkBudget(FakeStore(limit=3))
        budget.note_failure(1)
        self.assertEqual(budget.remaining(2), 3)
        self.assertEqual(budget.remaining(1), 2)

    def test_new_server_day_drops_only_that_guilds_old_counts(self):
        store = FakeStore(limit=2)
        budget = TalkBudget(store)
        budget.note_failure(1)
        budget.note_failure(2)
        store.days[2] = "d2"
        self.assertEqual(budget.remaining(2), 2)
        self.assertEqual(budget.remaining(1), 1)
        self.assertEqual(budget.failed, {(1, "d1"): 1})

    def test_games_on_one_bot_share_the_counter(self):
        store = FakeStore(limit=3)
        bot = SimpleNamespace(store=store, talk_budget=TalkBudget(store))
        games = [GameTables(bot), DoubtTables(bot), HoldemTables(bot)]
        games[0].talk.note_failure(1)
        games[1]._note_failure(1)
        self.assertEqual([g.talk_failed for g in games], [bot.talk_budget.failed] * 3)
        self.assertEqual(bot.talk_budget.remaining(1), 1)
        self.assertEqual(sum(g.talk_failed.get(k, 0) for g in games[2:] for k in g.talk_failed), 2)


if __name__ == "__main__":
    unittest.main()
