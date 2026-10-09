import os
import time
import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from llmcord_core.errors import error_detail, mask_facts, redact
from llmcord_core.store import Store


def usage(created_at, guild_id=1):
    from types import SimpleNamespace
    return SimpleNamespace(guild_id=guild_id, profile="p", model="m", role="dialogue", input_tokens=1, output_tokens=1,
                           cached_tokens=0, reasoning_tokens=0, cost_usd=0.1, cost_basis="x", created_at=created_at,
                           channel_id=None, feature="")


class TurnLogTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)

    def on(self, guild=1, days=14):
        self.store.set_turn_log(guild, True, days)

    def count(self, table="turn_log", guild=None):
        sql = f"SELECT COUNT(*) FROM {table}" + (" WHERE guild_id=%d" % guild if guild else "")
        return self.store.one(sql)[0]

    def test_settings_defaults_and_round_trip(self):
        """FEAT-05: no row means off and 14 days; settings round-trip per guild."""
        self.assertEqual(self.store.turn_log_settings(1), {"enabled": False, "days": 14})
        self.store.set_turn_log(1, True, 30)
        self.assertEqual(self.store.turn_log_settings(1), {"enabled": True, "days": 30})
        self.assertEqual(self.store.turn_log_settings(2), {"enabled": False, "days": 14})
        self.store.set_turn_log(1, False, 0)
        self.assertEqual(self.store.turn_log_settings(1), {"enabled": False, "days": 0})

    def test_settings_validation(self):
        """FEAT-05: bad switch or day values are rejected with plain messages."""
        for enabled, days in ((1, 7), ("yes", 7), (True, 5), (True, True), (True, "7"), (True, 7.0), (True, -1)):
            with self.assertRaises(ValueError):
                self.store.set_turn_log(1, enabled, days)

    def test_off_writes_nothing(self):
        """FEAT-05: with the switch off, or no guild, nothing is stored."""
        self.assertIsNone(self.store.add_turn_log(1, request_text="x"))
        self.assertIsNone(self.store.add_turn_log(None, request_text="x"))
        self.store.set_turn_log(1, True, 7)
        self.store.set_turn_log(1, False, 7)
        self.assertIsNone(self.store.add_turn_log(1, request_text="x"))
        self.assertEqual(self.count(), 0)

    def test_on_writes_row(self):
        """FEAT-05: with the switch on a row is written and its id returned."""
        self.on()
        rid = self.store.add_turn_log(1, channel_id=5, message_id=6, stage="reply", profile="p", model="m", request_text="hi",
                                      response_text="yo", input_tokens=3, output_tokens=4, now=time.time() - 5)
        row = self.store.turn_log_entry(1, rid)
        self.assertEqual((row["channel_id"], row["message_id"], row["stage"], row["status"], row["request_text"], row["response_text"],
                          row["input_tokens"], row["output_tokens"]), (5, 6, "reply", "ok", "hi", "yo", 3, 4))

    def test_status_validation(self):
        """FEAT-05: status must be ok or error."""
        self.on()
        with self.assertRaises(ValueError):
            self.store.add_turn_log(1, status="weird")

    def test_masking_before_cap(self):
        """FEAT-05: a fact straddling the cap boundary is hidden entirely, not left half-visible."""
        self.on()
        fact = "SECRETFACTSTRING"
        text = "a" * (16000 - 5) + fact + "tail"
        rid = self.store.add_turn_log(1, response_text=text, masked_facts=[fact])
        out = self.store.turn_log_entry(1, rid)["response_text"]
        self.assertNotIn("SECRET", out)
        self.assertNotIn("SECRETFACT", out)
        self.assertNotIn("ECRET", out)

    def test_mask_longest_first(self):
        """FEAT-05: overlapping facts are masked longest first."""
        self.assertEqual(mask_facts("likes red apples", ["red", "red apples", ""]), "likes [personal fact hidden]")

    def test_redaction(self):
        """FEAT-05: env secrets, extra values and Bearer tokens are redacted from every text field."""
        self.on()
        with patch.dict(os.environ, {"FOO_API_KEY": "envsecret123", "SHORT_KEY": "abc", "OTHER": "notsecret"}):
            rid = self.store.add_turn_log(1, request_text="k=envsecret123 extra-val Bearer abcdefghijklmnopqrstuvwx.yz notsecret",
                                          error_detail="envsecret123", secret_values=["extra-val"], status="error")
        row = self.store.turn_log_entry(1, rid)
        self.assertEqual(row["request_text"], "k=[redacted] [redacted] [credential omitted] notsecret")
        self.assertEqual(row["error_detail"], "[redacted]")
        self.assertEqual(redact("x", [""]), "x")

    def test_prose_survives_redaction(self):
        """FEAT-05: roleplay text such as "the bot waves" or "Bot Fred" is not mangled; header forms are redacted."""
        self.on()
        rid = self.store.add_turn_log(1, response_text="the bot waves; Bot Fred nods. Authorization: Bearer x", request_text="abc")
        row = self.store.turn_log_entry(1, rid)
        self.assertEqual(row["response_text"], "the bot waves; Bot Fred nods. [credential omitted]")
        self.assertEqual(row["request_text"], "abc")

    def test_short_env_values_ignored(self):
        """FEAT-05: env secrets under 8 characters are not replaced, but caller values always are."""
        with patch.dict(os.environ, {"X_KEY": "abc"}):
            self.assertEqual(redact("abc tiny"), "abc tiny")
            self.assertEqual(redact("abc tiny", ["tiny"]), "abc [redacted]")

    def test_mask_facts_normalised(self):
        """FEAT-05: facts are stripped, matched case-insensitively and literally; blank facts are ignored."""
        self.assertEqual(mask_facts("Loves Cats.", ["  loves cats "]), "[personal fact hidden].")
        self.assertEqual(mask_facts("a b", ["   ", "", None]), "a b")
        self.assertEqual(mask_facts("cost is $5 (approx.)? yes", ["$5 (approx.)?"]), "cost is [personal fact hidden] yes")
        self.assertEqual(mask_facts("axb", ["a.b"]), "axb")

    def test_reads_hide_rows_past_cutoff(self):
        """FEAT-05 (D19): rows older than the retention cutoff are hidden from reads before cleanup runs."""
        self.on(1, 7)
        now = time.time()
        old = self.store.add_turn_log(1, now=now - 8 * 86400)
        new = self.store.add_turn_log(1, now=now - 6 * 86400)
        self.assertEqual([r["id"] for r in self.store.turn_log_page(1)], [new])
        self.assertIsNone(self.store.turn_log_entry(1, old))
        self.assertIsNotNone(self.store.turn_log_entry(1, new))

    def test_monitoring_cutoff_and_bad_timezone(self):
        """FEAT-05 (D19): the cutoff helper handles windows, full month, and falls back to UTC for a bad timezone."""
        self.assertEqual(self.store.monitoring_cutoff(9, 100 * 86400), 86 * 86400)
        self.store.set_turn_log(1, True, 0)
        with self.store.db:
            self.store.db.execute("UPDATE guild_settings SET timezone='Not/AZone' WHERE guild_id=1")
        now = datetime(2026, 10, 31, 12, tzinfo=ZoneInfo("UTC")).timestamp()
        self.assertEqual(self.store.monitoring_cutoff(1, now), datetime(2026, 10, 1, tzinfo=ZoneInfo("UTC")).timestamp())

    def test_caps_with_marker(self):
        """FEAT-05: long fields are cut with a count marker; short ones are untouched."""
        self.on()
        rid = self.store.add_turn_log(1, request_text="r" * 48010, response_text="s" * 16001, error_detail="e" * 8002,
                                      stage="t" * 205, profile="p" * 200, model="m" * 201, reference_id="f" * 300)
        row = self.store.turn_log_entry(1, rid)
        self.assertEqual(row["request_text"], "r" * 48000 + "\n[… 10 characters cut]")
        self.assertEqual(row["response_text"], "s" * 16000 + "\n[… 1 characters cut]")
        self.assertEqual(row["error_detail"], "e" * 8000 + "\n[… 2 characters cut]")
        self.assertEqual(row["stage"], "t" * 200 + "\n[… 5 characters cut]")
        self.assertEqual(row["profile"], "p" * 200)
        self.assertTrue(row["model"].endswith("[… 1 characters cut]"))
        self.assertTrue(row["reference_id"].endswith("[… 100 characters cut]"))

    def test_page_filters_and_order(self):
        """FEAT-05: pages are newest first, filterable, paginated, clamped, and omit request/response text."""
        self.on()
        ids = [self.store.add_turn_log(1, channel_id=10 + i % 2, status="error" if i % 3 == 0 else "ok", reference_id=f"r{i}",
                                       request_text="REQ", response_text="resp%d" % i, error_detail="boom%d" % i) for i in range(6)]
        page = self.store.turn_log_page(1)
        self.assertEqual([r["id"] for r in page], ids[::-1])
        self.assertNotIn("request_text", page[0])
        self.assertNotIn("response_text", page[0])
        self.assertEqual(page[0]["preview"], "resp5")
        self.assertEqual(self.store.turn_log_page(1, errors_only=True)[0]["preview"], "boom3")
        self.assertEqual([r["id"] for r in self.store.turn_log_page(1, channel_id=10)], [ids[4], ids[2], ids[0]])
        self.assertEqual([r["id"] for r in self.store.turn_log_page(1, errors_only=True)], [ids[3], ids[0]])
        self.assertEqual([r["id"] for r in self.store.turn_log_page(1, reference_id="r2")], [ids[2]])
        self.assertEqual([r["id"] for r in self.store.turn_log_page(1, before_id=ids[3], limit=2)], [ids[2], ids[1]])
        self.assertEqual(len(self.store.turn_log_page(1, limit=0)), 1)
        self.assertEqual(len(self.store.turn_log_page(1, limit=9999)), 6)
        self.assertEqual(self.store.turn_log_page(2), [])

    def test_preview_is_short(self):
        """FEAT-05: previews hold at most 200 characters."""
        self.on()
        self.store.add_turn_log(1, response_text="x" * 500)
        self.assertEqual(len(self.store.turn_log_page(1)[0]["preview"]), 200)

    def test_entry_isolated_by_guild(self):
        """FEAT-05: another server's entry id reads as missing."""
        self.on(1)
        rid = self.store.add_turn_log(1, request_text="x")
        self.assertIsNotNone(self.store.turn_log_entry(1, rid))
        self.assertIsNone(self.store.turn_log_entry(2, rid))

    def seed(self, guild, times):
        for t in times:
            with self.store.db:
                self.store.db.execute("INSERT INTO turn_log(guild_id,created_at) VALUES(?,?)", (guild, t))
            self.store.record_model_usage(usage(t, guild))

    def test_expire_day_windows(self):
        """FEAT-05 (D19): 7/14/30 day windows delete older rows from both tables; other guilds keep theirs."""
        now = 100 * 86400
        for days in (7, 14, 30):
            with self.store.db:
                self.store.db.execute("DELETE FROM turn_log")
                self.store.db.execute("DELETE FROM model_usage")
            self.store.set_turn_log(1, True, days)
            self.seed(1, [now - days * 86400 - 1, now - days * 86400 + 1])
            self.seed(2, [now - 40 * 86400])
            self.store.expire_monitoring(now)
            self.assertEqual(self.count(guild=1), 1)
            self.assertEqual(self.count("model_usage", guild=1), 1)
            self.assertEqual(self.count(guild=2), 0)

    def test_expire_default_14_without_settings(self):
        """FEAT-05 (D19): guilds with rows but no settings row use 14 days."""
        now = 100 * 86400
        self.seed(3, [now - 15 * 86400, now - 13 * 86400])
        self.store.expire_monitoring(now)
        self.assertEqual(self.count(guild=3), 1)
        self.assertEqual(self.count("model_usage", guild=3), 1)

    def test_expire_full_month_in_timezone(self):
        """FEAT-05 (D19): full month starts at the 1st in the server timezone, not UTC."""
        seoul = ZoneInfo("Asia/Seoul")
        now = datetime(2026, 11, 1, 0, 30, tzinfo=seoul).timestamp()
        self.store.set_guild_timezone(1, "Asia/Seoul")
        self.store.set_turn_log(1, True, 0)
        before = datetime(2026, 10, 31, 23, 0, tzinfo=seoul).timestamp()
        after = datetime(2026, 11, 1, 0, 10, tzinfo=seoul).timestamp()
        self.seed(1, [before, after])
        self.store.set_turn_log(2, True, 0)
        self.seed(2, [before, after])  # UTC guild: now is still Oct 31 UTC, so both survive
        self.store.expire_monitoring(now)
        self.assertEqual(self.count(guild=1), 1)
        self.assertEqual(self.count("model_usage", guild=1), 1)
        self.assertEqual(self.count(guild=2), 2)

    def test_expire_history_keeps_model_usage(self):
        """FEAT-05 (D19): history retention no longer deletes model_usage."""
        self.store.record_model_usage(usage(10.0))
        self.store.expire_history(1, now=100 * 86400)
        self.assertEqual(self.count("model_usage"), 1)

    def test_error_detail_unchanged(self):
        """FEAT-05: error_detail still redacts credentials and URLs."""
        with patch.dict(os.environ, {"X_TOKEN": "tok12345"}):
            out = error_detail(RuntimeError("tok12345 at https://a.b/c Bearer zzz"))
        self.assertEqual(out, "RuntimeError: [redacted] at [URL omitted] [credential omitted]")


if __name__ == "__main__":
    unittest.main()
