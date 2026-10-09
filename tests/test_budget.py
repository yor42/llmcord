import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from llmcord_core import budget
from llmcord_core.admin_store import ConflictError
from llmcord_core.store import Store


def ts(y, m, d, h=0, mi=0, s=0):
    return datetime(y, m, d, h, mi, s, tzinfo=timezone.utc).timestamp()


class PeriodTests(unittest.TestCase):
    def test_period_start_and_next_reset_across_boundaries(self):
        cases = [
            (ts(2026, 1, 10), 15, date(2025, 12, 15), date(2026, 1, 15)),
            (ts(2026, 3, 20), 15, date(2026, 3, 15), date(2026, 4, 15)),
            (ts(2026, 1, 10), 1, date(2026, 1, 1), date(2026, 2, 1)),
            (ts(2026, 12, 31), 28, date(2026, 12, 28), date(2027, 1, 28)),
            (ts(2026, 3, 1), 28, date(2026, 2, 28), date(2026, 3, 28)),
            (ts(2026, 1, 1), 28, date(2025, 12, 28), date(2026, 1, 28)),
        ]
        for now, day, start, nxt in cases:
            with self.subTest(now=now, day=day):
                self.assertEqual(budget.period_start(now, day), start)
                self.assertEqual(budget.next_reset(now, day), nxt)

    def test_reset_instant_begins_new_period_and_one_second_before_is_old(self):
        self.assertEqual(budget.period_start(ts(2026, 5, 15), 15), date(2026, 5, 15))
        self.assertEqual(budget.period_start(ts(2026, 5, 14, 23, 59, 59), 15), date(2026, 4, 15))

    def test_period_key_is_iso(self):
        self.assertEqual(budget.period_key(date(2026, 5, 15)), '2026-05-15')


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / 'b.db')
        self.store = Store(self.path)

    def spend(self, day, cost, unpriced=0):
        with self.store.db:
            self.store.db.execute('INSERT INTO spend_days(day,cost_usd,unpriced_calls) VALUES(?,?,?)', (day, cost, unpriced))


class StateTests(StoreCase):
    def test_state_sums_only_days_in_period_and_counts_unpriced(self):
        for day, cost, un in (('2026-04-14', 100, 9), ('2026-04-15', 1.5, 1), ('2026-05-14', 2, 2), ('2026-05-15', 50, 7)):
            self.spend(day, cost, un)
        self.store.save_budget(None, None, 15, False, 0)
        st = budget.state(self.store, ts(2026, 5, 1))
        self.assertEqual((st.spent_usd, st.unpriced_calls), (3.5, 3))
        self.assertEqual((st.period, st.resets_on, st.revision), ('2026-04-15', date(2026, 5, 15), 1))

    def test_caps_off_never_reached_and_zero_cap_reached_immediately(self):
        st = budget.state(self.store, ts(2026, 5, 1))
        self.assertFalse(st.soft_reached or st.hard_reached)
        self.store.save_budget(0, 0, 1, True, 0)
        st = budget.state(self.store, ts(2026, 5, 1))
        self.assertTrue(st.soft_reached and st.hard_reached and st.channel_notice)

    def test_soft_and_hard_thresholds(self):
        self.spend('2026-05-02', 5)
        self.store.save_budget(5, 10, 1, False, 0)
        st = budget.state(self.store, ts(2026, 5, 3))
        self.assertTrue(st.soft_reached)
        self.assertFalse(st.hard_reached)


class SaveBudgetTests(StoreCase):
    def test_happy_path_bumps_revision(self):
        self.assertEqual(self.store.budget_settings()['revision'], 0)
        self.store.save_budget(1, 2.5, 10, True, 0)
        s = self.store.budget_settings()
        self.assertEqual((s['soft_cap_usd'], s['hard_cap_usd'], s['reset_day'], s['channel_notice'], s['revision']), (1.0, 2.5, 10, 1, 1))

    def test_stale_revision_conflicts_without_writing(self):
        self.store.save_budget(1, 2, 10, False, 0)
        with self.assertRaises(ConflictError):
            self.store.save_budget(5, 6, 3, True, 0)
        s = self.store.budget_settings()
        self.assertEqual((s['soft_cap_usd'], s['reset_day'], s['revision']), (1.0, 10, 1))

    def test_validation_rejects_bad_values(self):
        bad = [(3, 2, 1, False), (-1, None, 1, False), (None, -0.5, 1, False),
               (float('nan'), None, 1, False), (None, float('inf'), 1, False), (True, None, 1, False),
               (None, None, 0, False), (None, None, 29, False), (None, None, True, False), (None, None, 1, 1)]
        for soft, hard, day, notice in bad:
            with self.subTest(soft=soft, hard=hard, day=day, notice=notice):
                with self.assertRaises(ValueError):
                    self.store.save_budget(soft, hard, day, notice, 0)
        self.assertEqual(self.store.budget_settings()['revision'], 0)


class NoticeTests(StoreCase):
    def test_once_per_period_kind_target(self):
        self.assertTrue(self.store.claim_notice('2026-05-01', 'soft', 0, 100.0))
        self.assertFalse(self.store.claim_notice('2026-05-01', 'soft', 0, 200.0))
        self.assertTrue(self.store.claim_notice('2026-05-01', 'soft', 7, 100.0))
        self.assertTrue(self.store.claim_notice('2026-05-01', 'hard', 0, 100.0))
        self.assertTrue(self.store.claim_notice('2026-06-01', 'soft', 0, 100.0))

    def test_interval_form(self):
        self.assertTrue(self.store.claim_notice('p', 'k', 1, 1000.0, 60))
        self.assertFalse(self.store.claim_notice('p', 'k', 1, 1059.0, 60))
        self.assertTrue(self.store.claim_notice('p', 'k', 1, 1060.0, 60))
        self.assertFalse(self.store.claim_notice('p', 'k', 1, 1100.0, 60))

    def test_two_stores_on_one_file_have_one_winner(self):
        other = Store(self.path)
        results = [self.store.claim_notice('p', 'k', 1, 5.0), other.claim_notice('p', 'k', 1, 5.0)]
        self.assertEqual(results, [True, False])
        results = [self.store.claim_notice('q', 'k', 1, 5.0, 60), other.claim_notice('q', 'k', 1, 6.0, 60)]
        self.assertEqual(results, [True, False])

    def test_expire_history_prunes_old_notices(self):
        now = 1_800_000_000.0
        self.store.claim_notice('old', 'k', 1, now - 401 * 86400)
        self.store.claim_notice('new', 'k', 1, now - 399 * 86400)
        self.store.expire_history(30, now)
        left = [r['period'] for r in self.store.all('SELECT period FROM budget_notices')]
        self.assertEqual(left, ['new'])


if __name__ == '__main__':
    unittest.main()
