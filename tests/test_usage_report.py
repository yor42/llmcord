import unittest
from datetime import datetime, timezone

from llmcord_core.dashboard import USAGE_RANGES, USAGE_UNKNOWN, usage_feature_rows, usage_range_keys, usage_since
from llmcord_core.store import Store
from llmcord_core.usage import ModelUsage

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc).timestamp()


def add(store, guild, created_at, model='m', inputs=10, outputs=5, cost=0.01, channel=None, feature='', profile='p'):
    store.record_model_usage(ModelUsage(guild, profile, model, 'dialogue', inputs, outputs, 0, 0, cost, 'x', created_at, channel, feature))


class UsageReportTests(unittest.TestCase):
    def test_days_follow_the_server_timezone(self):
        """FEAT-12: 23:30 UTC on the 8th is already the 9th in Seoul, so the daily bucket moves with the server timezone."""
        store = Store()
        at = datetime(2026, 10, 8, 23, 30, tzinfo=timezone.utc).timestamp()
        add(store, 1, at)
        self.assertEqual(list(store.usage_report(1, NOW - 5 * 86400, NOW)['daily']['m']), ['2026-10-08'])
        store.set_guild_timezone(1, 'Asia/Seoul')
        report = store.usage_report(1, NOW - 5 * 86400, NOW)
        self.assertEqual((report['zone'], report['daily']['m']), ('Asia/Seoul', {'2026-10-09': [10, 5]}))

    def test_half_hour_zone_and_invalid_zone(self):
        """FEAT-12: a +05:30 zone buckets by its own midnight; an invalid stored zone falls back to UTC."""
        store = Store()
        add(store, 1, datetime(2026, 10, 8, 18, 45, tzinfo=timezone.utc).timestamp())
        store.set_guild_timezone(1, 'Asia/Kolkata')
        self.assertEqual(list(store.usage_report(1, NOW - 5 * 86400, NOW)['daily']['m']), ['2026-10-09'])
        store.db.execute("UPDATE guild_settings SET timezone='Not/AZone' WHERE guild_id=1"); store.db.commit()
        report = store.usage_report(1, NOW - 5 * 86400, NOW)
        self.assertEqual((report['zone'], list(report['daily']['m'])), ('UTC', ['2026-10-08']))

    def test_since_is_clamped_to_retention(self):
        """FEAT-12: asking for more than the retention window returns only rows inside it."""
        store = Store()
        store.set_turn_log(1, False, 7)
        add(store, 1, NOW - 10 * 86400, inputs=1)
        add(store, 1, NOW - 3 * 86400, inputs=2)
        report = store.usage_report(1, NOW - 30 * 86400, NOW)
        self.assertEqual(report['totals']['input_tokens'], 2)
        self.assertEqual(report['since'], NOW - 7 * 86400)

    def test_guild_isolation(self):
        """FEAT-12: another server's calls never appear in the totals, channels or daily buckets."""
        store = Store()
        add(store, 1, NOW - 100, model='mine', channel=5, feature='reply')
        add(store, 2, NOW - 100, model='theirs', inputs=999, channel=6, feature='ambient')
        report = store.usage_report(1, NOW - 86400, NOW)
        self.assertEqual(list(report['daily']), ['mine'])
        self.assertEqual([r['channel_id'] for r in report['by_channel']], [5])
        self.assertEqual([r['feature'] for r in report['by_feature']], ['reply'])
        self.assertEqual(report['totals']['input_tokens'], 10)

    def test_unknown_channel_and_feature_and_counts(self):
        """FEAT-12: old rows (NULL channel, empty feature) form one unknown bucket; unpriced and unreported calls are counted."""
        store = Store()
        add(store, 1, NOW - 100, channel=5, feature='reply', cost=0.5)
        add(store, 1, NOW - 90, channel=None, feature='', cost=None)
        add(store, 1, NOW - 80, channel=None, feature='', inputs=None, outputs=None)
        report = store.usage_report(1, NOW - 86400, NOW)
        channels = {r['channel_id']: r for r in report['by_channel']}
        self.assertEqual((channels[None]['requests'], channels[None]['unpriced'], channels[None]['unreported']), (2, 1, 1))
        self.assertEqual(channels[5]['cost_usd'], 0.5)
        self.assertEqual({r['feature']: r['requests'] for r in report['by_feature']}, {'reply': 1, None: 2})
        self.assertEqual((report['totals']['requests'], report['totals']['unpriced'], report['totals']['unreported']), (3, 1, 1))
        self.assertEqual(report['by_model'][0]['profile'], 'p')

    def test_empty_report(self):
        """FEAT-12: no calls gives zero totals and no buckets."""
        report = Store().usage_report(1, NOW - 86400, NOW)
        self.assertEqual((report['totals']['requests'], report['days'], report['by_model']), (0, [], []))


if __name__ == '__main__':
    unittest.main()


class MonitoringHelperTests(unittest.TestCase):
    def test_range_keys_follow_retention(self):
        """FEAT-12: period options never reach past retention; 'month' only for 30 days or the current-month setting."""
        self.assertEqual(usage_range_keys(7), ['day', '7'])
        self.assertEqual(usage_range_keys(14), ['day', '7', '14'])
        self.assertEqual(usage_range_keys(30), list(USAGE_RANGES))
        self.assertEqual(usage_range_keys(0), ['day', '7', 'month'])

    def test_month_starts_at_local_midnight(self):
        """FEAT-12: 20:00 UTC on Oct 31 is already Nov 1 in Seoul, so 'This month' starts Nov 1 00:00 KST."""
        now = datetime(2026, 10, 31, 20, 0, tzinfo=timezone.utc).timestamp()
        self.assertEqual(usage_since('Asia/Seoul', 'month', now), datetime(2026, 10, 31, 15, 0, tzinfo=timezone.utc).timestamp())
        self.assertEqual(usage_since('UTC', 'month', now), datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp())

    def test_feature_labels_merge_unknown_names(self):
        """FEAT-12: empty features are 'before this update'; other unrecognised names merge into one 'Unknown feature' row."""
        def row(feature, requests, cost, unpriced=0, unreported=0):
            return {'feature': feature, 'requests': requests, 'input_tokens': requests * 10, 'output_tokens': requests, 'cost_usd': cost, 'unpriced': unpriced, 'unreported': unreported}
        rows = usage_feature_rows([row('reply', 1, 0.5), row(None, 2, 0.0, 1, 1), row('old_thing', 3, 0.25, 1), row('other', 2, None, 0, 1)])
        by = {r['feature']: r for r in rows}
        self.assertEqual(list(by), ['Replies', USAGE_UNKNOWN, 'Unknown feature'])
        unknown = by['Unknown feature']
        self.assertEqual((unknown['requests'], unknown['input_tokens'], unknown['output_tokens'], unknown['cost_usd'], unknown['unpriced'], unknown['unreported']), (5, 50, 5, 0.25, 3, 1))
        self.assertEqual(by[USAGE_UNKNOWN]['requests'], 2)
