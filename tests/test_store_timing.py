"""REL-01 (measure only): slow-sqlite timing logs and counters on Store.

The planned design times every ``execute``/``executemany``/``executescript``/``commit``
on ``Store.db``, logs one WARNING on the ``llmcord_core.store`` logger for calls taking
at least ``store.SLOW_QUERY_SECONDS`` (read at call time), and exposes counters via
``Store.timing_stats()``. Bound parameters must never reach the log (they can hold
personal facts). The characterization tests pin that nothing else about Store changes.
"""
import logging
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from llmcord_core.store import Store

LOGGER = "llmcord_core.store"
SECRET = "SECRET-personal-fact-7f3a9c"


def threshold(value):
    # create=True so the known-defect tests fail on the missing log/counters,
    # not on patching an attribute that does not exist yet.
    return patch("llmcord_core.store.SLOW_QUERY_SECONDS", value, create=True)


class StoreTimingTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.world = self.store.create_space(1, "World", "world")
        self.alice = self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])
        self.store.set_consent(1, 9, True)

    def tearDown(self):
        self.store.close()

    def test_slow_call_logs_duration_and_sql_but_not_params(self):
        """REL-01 (fixed): with every call slow, a personal-memory write logs a WARNING with ms and the SQL prefix, never the bound values."""
        with threshold(0), self.assertLogs(LOGGER, logging.WARNING) as logs:
            self.store.record_node(555, 1, 100, None, 9, None, "src")
            self.store.add_personal(1, 9, self.alice, SECRET, 555)
        self.assertEqual(self.store.personal(1, 9)[0]["content"], SECRET)
        output = "\n".join(logs.output)
        self.assertIn("ms", output)
        self.assertIn("INSERT OR IGNORE INTO personal_memories", output)
        self.assertNotIn(SECRET, output)
        self.assertNotIn("555", output)

    def test_slow_call_log_collapses_whitespace_and_truncates_sql(self):
        """REL-01 (fixed): the logged SQL is whitespace-collapsed and at most 80 characters; raw execute args stay out of the log."""
        sql = "SELECT   ?   AS\n\n   secret_value,\t" + ", ".join(f"{n} AS c{n}" for n in range(60))
        with threshold(0), self.assertLogs(LOGGER, logging.WARNING) as logs:
            self.store.one(sql, (SECRET,))
        output = "\n".join(logs.output)
        collapsed = " ".join(sql.split())
        self.assertIn(collapsed[:80], output)
        self.assertNotIn(collapsed[:81], output)
        self.assertNotIn(SECRET, output)

    def test_fast_query_logs_nothing_at_default_threshold(self):
        """REL-01 (guard): a trivial query under the default threshold produces no WARNING (passes today and must stay so)."""
        with self.assertNoLogs(LOGGER, logging.WARNING):
            self.assertEqual(self.store.one("SELECT 1 AS one")["one"], 1)
            self.store.has_consent(1, 9)

    def test_timing_stats_counts_calls(self):
        """REL-01 (fixed): Store.timing_stats() reports calls, total/max seconds and slow calls."""
        before = self.store.timing_stats()
        self.assertEqual(set(before), {"calls", "total_seconds", "max_seconds", "slow_calls"})
        self.store.has_consent(1, 9)
        self.store.personal(1, 9)
        self.store.record_node(556, 1, 100, None, 9, None, "src")
        self.store.add_personal(1, 9, self.alice, "fact", 556)
        after = self.store.timing_stats()
        self.assertGreaterEqual(after["calls"], before["calls"] + 3)
        self.assertGreater(after["total_seconds"], 0)
        self.assertGreaterEqual(after["max_seconds"], 0)
        self.assertLessEqual(after["max_seconds"], after["total_seconds"])
        with threshold(0), self.assertLogs(LOGGER, logging.WARNING):
            self.store.one("SELECT 1")
        self.assertGreater(self.store.timing_stats()["slow_calls"], after["slow_calls"])

    def test_genuinely_slow_query_detected_at_default_threshold(self):
        """REL-01 (fixed): a query that really takes ~60 ms logs exactly one WARNING at the default 50 ms threshold."""
        self.store.db.create_function("slow_fn", 0, lambda: time.sleep(0.06) or 1)
        with self.assertLogs(LOGGER, logging.WARNING) as logs:
            self.assertEqual(self.store.one("SELECT slow_fn() AS v")["v"], 1)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("slow_fn", logs.output[0])

    def test_with_block_commit_and_rollback_are_timed(self):
        """REL-01 (fixed): the implicit COMMIT/ROLLBACK of `with store.db:` (used by store.execute) is timed and logged."""
        with threshold(0), self.assertLogs(LOGGER, logging.WARNING) as logs:
            self.store.execute("INSERT INTO consent VALUES(?,?,?)", (1, 10, 1))
        self.assertTrue(any("COMMIT" in line for line in logs.output), logs.output)

        error = RuntimeError("boom")
        with threshold(0), self.assertLogs(LOGGER, logging.WARNING) as logs:
            with self.assertRaises(RuntimeError) as raised:
                with self.store.db:
                    self.store.db.execute("INSERT INTO consent VALUES(?,?,?)", (1, 11, 1))
                    raise error
        self.assertIs(raised.exception, error)
        self.assertIsNone(self.store.one("SELECT * FROM consent WHERE guild_id=1 AND user_id=11"))
        self.assertTrue(any("ROLLBACK" in line for line in logs.output), logs.output)

    def test_failing_execute_propagates_and_is_counted(self):
        """Characterization (REL-01): a failing db.execute raises the original OperationalError unchanged and still counts as a call."""
        before = self.store.timing_stats()["calls"]
        with self.assertRaises(sqlite3.OperationalError) as raised:
            self.store.db.execute("SELEC nonsense FROM nowhere")
        self.assertIs(type(raised.exception), sqlite3.OperationalError)
        self.assertIn("syntax error", str(raised.exception))
        self.assertEqual(self.store.timing_stats()["calls"], before + 1)


class StoreBehaviorUnchangedTests(unittest.TestCase):
    def test_transaction_rollback_on_exception(self):
        """Characterization (REL-01): `with store.db:` still rolls back on exception."""
        store = Store()
        try:
            world = store.create_space(1, "World", "world")
            with self.assertRaises(RuntimeError):
                with store.db:
                    store.db.execute("UPDATE spaces SET name='Changed' WHERE id=?", (world,))
                    raise RuntimeError("boom")
            self.assertEqual(store.space_by_id(world)["name"], "World")
        finally:
            store.close()

    def test_rows_are_sqlite_row_and_lastrowid_returned(self):
        """Characterization (REL-01): in-memory Store yields sqlite3.Row and execute() returns lastrowid."""
        store = Store()
        try:
            self.assertIsInstance(store.db, sqlite3.Connection)
            self.assertIs(store.db.row_factory, sqlite3.Row)
            ident = store.execute("INSERT INTO consent VALUES(?,?,?)", (1, 2, 1))
            self.assertIsInstance(ident, int)
            row = store.one("SELECT * FROM consent WHERE guild_id=1")
            self.assertIsInstance(row, sqlite3.Row)
            self.assertEqual(row["user_id"], 2)
            self.assertEqual(store.one("PRAGMA foreign_keys")[0], 1)
        finally:
            store.close()

    def test_temp_file_store_migrates_and_reopens(self):
        """Characterization (REL-01): a file-backed Store creates schema v3, persists data, and reopens without error."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "skit.sqlite3"
            store = Store(path)
            world = store.create_space(1, "World", "world")
            self.assertEqual(store.one("PRAGMA user_version")[0], 9)
            store.close()
            reopened = Store(path)
            try:
                self.assertEqual(reopened.space_by_id(world)["name"], "World")
                self.assertEqual(reopened.one("PRAGMA user_version")[0], 9)
                self.assertEqual(reopened.one("PRAGMA journal_mode")[0], "wal")
            finally:
                reopened.close()
            self.assertEqual(list(Path(tmp, "sub").glob("*.pre-*")), [])


if __name__ == "__main__":
    unittest.main()
