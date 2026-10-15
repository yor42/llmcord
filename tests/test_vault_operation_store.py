"""Vault operation (co-op) in the store, FEAT-66 phase A, schema v26, decision D43. Written before the implementation; they fail until it exists.

API CONTRACT (new methods on ``VaultStore`` in ``llmcord_core/vault_store.py``; same conventions as tests/test_vault_store.py: synchronous,
``guild_id`` first, every query guild-scoped, ``now`` Unix seconds, writes in ``write_admin()``, refusals are ``VaultStoreError`` with
``err.code.code`` and nothing written on a refusal)

  Settings (own table ``vault_operation_settings``, one row per guild; ``vault_settings()`` is NOT changed)
    vault_operation_settings(guild_id) -> {"enabled": False, "channel_id": None, "week_start": 0, "stage_protocols": ["standard","standard","open"],
        "contribution_reward": 0, "team_reward": 0, "min_guesses": 1, "revision": 0}
    set_vault_operation_settings(guild_id, *, enabled=None, channel_id=None, week_start=None, stage_protocols=None, contribution_reward=None,
        team_reward=None, min_guesses=None, expected_revision=None) -> new dict      (None leaves a field alone; revision +1; stale -> ConflictError)
        week_start 0..6 (Monday=0, bool is not an int); stage_protocols a list of exactly 3 vault.PROTOCOLS; rewards 0..MAX_CURRENCY_CHANGE;
        min_guesses 1..30. Bad input raises UserError and writes nothing. The operation also needs ``vault_settings(guild_id)["enabled"]``.

  Lifecycle
    open_vault_operation(guild_id, user_id, channel_id, *, npc_ids=None, now=None, seed=None, parent_id=None) -> view
        Join the current week's operation of the configured channel, creating it first if none exists (started by the first join).
        Joining costs no turn. Refusals, in order: vault.disabled (Vault off), vault.operation_disabled, vault.operation_channel (channel is not
        the configured one), vault.operation_npcs (fewer than two eligible cast characters, or npc_ids not exactly two distinct ids that are
        eligible and in the channel cast; only checked when creating). The week: week_key = ISO date of the latest local ``week_start`` day
        midnight at or before ``now`` in the SERVER timezone; starts_at that midnight, expires_at seven local days later (calendar days,
        so a DST week is 167 or 169 hours). Stored at creation and never recomputed. One row per (guild, channel, week_key).
        An operation whose expires_at <= now is marked expired (with its active stage) and a new one is created for the new week.
        npc_ids None -> the first two eligible characters of ``get_cast(channel_id, parent_id)`` in cast order. Persisted.
    vault_operation(guild_id, user_id, operation_id, *, now=None) -> view     participants only: vault.not_found (other guild / unknown id),
        vault.operation_not_participant (member never joined). Lazily expires.
    current_vault_operation_id(guild_id, channel_id, *, now=None) -> int | None     the active operation of the channel, else None.
    Every action below takes ``operation_id`` after ``user_id`` and applies the same two refusals first, then vault.not_active
    (operation solved, failed or expired).

  Stages: three, sequential; protocol per settings at creation; stage 1 starts active, the others are "locked". A stage starts with its
    secret (random.Random(seed)-derived, never exposed while active), turn_budget = vault_settings(guild)["turns"][protocol] at that moment,
    the open digit for protocol "open", and two clues (vault.make_clues) held one each by the two NPCs. The budget is SHARED: guesses, tool
    turn costs and cross-validation all spend it. Exhausting it without a win: first time -> automatic retry (same stage number, retry=1,
    NEW secret, fresh budget, new clues, previous attempt's guesses no longer in the stage view); second time -> stage and operation "failed".
    A win marks the stage "solved" and starts the next; the third win marks the operation "solved".
    submit_operation_guess(guild_id, user_id, operation_id, guess, *, action_key=None, now=None) -> result
        {"number" (1-based within the attempt), "guess", "feedback", "won", "replayed", "event", "operation": view}
        event: None | "stage_solved" | "stage_retry" | "operation_failed" | "operation_solved".
        Invalid guesses use the personal codes (vault.guess_length / guess_not_digits / guess_repeated_digit); a guess already tried in the
        attempt by ANY member -> vault.guess_duplicate. Refusals are free. A replayed action_key (per operation) returns the first result,
        replayed=True, spends nothing, even after the stage moved on.
    reveal_operation_clue(guild_id, user_id, operation_id, position, *, now=None) -> clue entry    free, idempotent, shared by the team;
        vault.clue_unknown for a bad 1-based position.

  View (dict): id, guild_id, channel_id, week_key, starts_at, expires_at, status ("active"|"solved"|"failed"|"expired"), created (bool, only
    on the creating call), joined (bool, only on the call that added this member), npcs [{"character_id","name"}], participants
    [{"user_id","joined_at","advisor": {"character_id","name"}|None,"contributions": int,"reward_contribution": R|None,"reward_team": R|None}]
    with R = {"amount","ledger_id","reversed"}, stage (the active stage entry or None), stages (always 3 entries: {"number","protocol","length",
    "status": "locked"|"active"|"solved"|"failed"|"expired","retry": 0|1,"turn_budget","turns_used","turns_left","open_digit","guesses":
    [{"number","guess","feedback","won","user_id"}],"tools":[{"tool","params","text_key","data","turn_cost","user_id"}],"tools_used",
    "cross":[{"numbers":(a,b),"common":[digit,...],"count","user_id"}],"clues":[{"position","character_id","name","revealed","revealed_at"} +
    "constraint" only when revealed or the stage ended],"advice":[...],"constraints":[public constraints],"secret": None until the stage ended}),
    "secret" is not a key of the operation; a locked stage has no guesses, clues or secret.
  Tables (v26): vault_operation_settings, vault_operations, vault_operation_participants, vault_operation_stages (columns include
    operation_id, number, retry, protocol, length, turn_budget, secret, open_digit, status; one row per started stage holding the CURRENT
    attempt), vault_operation_actions, vault_operation_clues, vault_operation_claims (guild_id, operation_id, user_id, type
    "contribution"|"team", amount, ledger_id, created_at; UNIQUE(operation_id, user_id, type)). Details of the migration: tests/test_vault_migration_v26.py.

  Advisors and advice: set_operation_advisor(guild_id, user_id, operation_id, character_id | None, *, now=None) -> view   one favorite per member
    (None clears; replaces the previous); the character must be one of the member's eligible favorites on the channel (archived ones only
    while archived_favorites is on): otherwise vault.partner_unavailable. A favorite never holds a clue.
    record_operation_advice(guild_id, user_id, operation_id, character_id, *, text=None, usage=None, action_key=None, now=None) -> advice entry
        character_id must be one of the two NPCs or the member's own advisor (else vault.advice_not_partner). Same accounting as personal advice
        (advice_cap per stage attempt shared by the team -> vault.advice_limit; guild talk limit -> vault.advice_talk_limit), spends no turn, no item,
        and is not a contribution. Appears in the stage view "advice".

  Tools and cross-validation (both are contributions and spend the shared budget):
    use_operation_tool(guild_id, user_id, operation_id, tool_key, params, *, action_key=None, now=None) -> result    like use_vault_tool
        (same codes and order; result["operation"] instead of result["session"]); consumes ONE unit of the CALLER's own inventory inside the same
        transaction; vault.tool_limit (limit 2) counts the stage attempt's tools across ALL members.
    operation_cross_validate(guild_id, user_id, operation_id, number_a, number_b, *, action_key=None, now=None) -> result
        {"count","common","numbers","turn_cost": 1,"replayed","event","operation"}. numbers are guess numbers of the current attempt, two different
        ints (bool rejected) written by two DIFFERENT members. count = vault.cross_validation(secret, guess_a, guess_b); the public constraint
        "the secret has exactly ``count`` of the shared digits" joins the stage constraints. Refusals, in order: vault.cross_unknown (not two
        existing different numbers), vault.cross_same_member, vault.cross_used (one per stage attempt), vault.cross_redundant (no shared digit or
        every remaining candidate gives the same count), vault.tool_no_turns (needs 1 turn). No item is needed.

  Rewards (ledger source "vault", positive, actor = the member; paid in the transaction of the action that qualifies; no timer):
    contribution_reward: once per member per operation when the member's contributions (valid guesses, tool uses, cross-validations; advice,
    joining, reading and rejected actions do not count) reach min_guesses. team_reward: to every member who has qualified, when the operation
    becomes "solved". Amount 0 pays nothing and writes no claim. Reversal is the ordinary reverse_entry (flagged ``reversed``, never repaid).
"""
import itertools
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from llmcord_core.admin_store import ConflictError
from llmcord_core.i18n import UserError
from llmcord_core.store import Store
from llmcord_core.vault import GuessResult, all_secrets, cross_validation, feedback, filter_candidates
from llmcord_core.vault_store import VaultStoreError

from test_vault_store import ALICE, BOB, CH, G, G2, NOW, VaultCase, code_of

CAROL = 7
SPECTRUM = "vault_spectrum_analyzer"
MONDAY = datetime(2026, 10, 12, tzinfo=timezone.utc).timestamp()   # NOW is Thursday 2026-10-15 12:00 UTC
WEEK = 7 * 86400


class OpCase(VaultCase):
    def setUp(self):
        super().setUp()
        self.store.set_vault_operation_settings(G, enabled=True, channel_id=CH)
        self.store.set_vault_settings(G, turns={"standard": 3, "open": 3})

    def join(self, user=ALICE, now=NOW, guild=G, channel=CH, **kw):
        return self.store.open_vault_operation(guild, user, channel, now=now, seed=kw.pop("seed", "op-1"), **kw)

    def op(self, user=ALICE, now=NOW, guild=G, oid=None):
        return self.store.vault_operation(guild, user, oid or self.oid, now=now)

    def start(self, **kw):
        self.view0 = self.join(**kw)
        self.oid = self.view0["id"]
        return self.view0

    def secret(self, number=None, oid=None):
        oid = oid or self.oid
        number = number or self.op()["stage"]["number"]
        return self.store.one("SELECT secret FROM vault_operation_stages WHERE operation_id=? AND number=?", (oid, number))["secret"]

    def g(self, code, user=ALICE, now=NOW, **kw):
        return self.store.submit_operation_guess(G, user, self.oid, code, now=now, **kw)

    def wrongs(self, n=1, user=ALICE):
        stage = self.op(user)["stage"]
        secret = self.secret()
        tried = {x["guess"] for x in stage["guesses"]}
        return [c for c in all_secrets(4) if c != secret and c not in tried][:n]

    def burn(self, n, user=ALICE):
        last = None
        for code in self.wrongs(n, user):
            last = self.g(code, user)
        return last

    def solve(self, user=ALICE, now=NOW):
        return self.g(self.secret(), user, now=now)

    def solve_all(self, user=ALICE):
        return [self.solve(user) for _ in range(3)]

    def give(self, key, qty=1, user=ALICE, guild=G):
        self.store.db.execute("INSERT INTO inventory(guild_id,user_id,item_key,qty) VALUES(?,?,?,?) ON CONFLICT(guild_id,user_id,item_key) DO UPDATE SET qty=excluded.qty",
                              (guild, user, key, qty))
        self.store.db.commit()

    def tool(self, params, user=ALICE, key=SPECTRUM, **kw):
        return self.store.use_operation_tool(G, user, self.oid, key, params, now=kw.pop("now", NOW), **kw)

    def refused(self, code, call, *args, **kw):
        with self.assertRaises(VaultStoreError) as ctx:
            call(*args, **kw)
        self.assertEqual(code_of(ctx.exception), code)
        return ctx.exception

    def ledger_total(self, user):
        return self.store.balance(G, user)


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)

    def test_defaults_are_off_unbound_monday_and_no_reward(self):
        self.assertEqual(self.store.vault_operation_settings(G), {"enabled": False, "channel_id": None, "week_start": 0, "stage_protocols": ["standard", "standard", "open"],
                                                                  "contribution_reward": 0, "team_reward": 0, "min_guesses": 1, "revision": 0})

    def test_save_reads_back_bumps_revision_keeps_other_fields_and_is_per_guild(self):
        first = self.store.set_vault_operation_settings(G, enabled=True, channel_id=CH, week_start=3)
        second = self.store.set_vault_operation_settings(G, team_reward=40)
        self.assertEqual((first["revision"], second["revision"]), (1, 2))
        self.assertEqual((second["enabled"], second["channel_id"], second["week_start"], second["team_reward"]), (True, CH, 3, 40))
        self.assertEqual(self.store.vault_operation_settings(G2), self.store.vault_operation_settings(99))
        self.assertFalse(self.store.vault_operation_settings(G2)["enabled"])
        with self.assertRaises(ConflictError):
            self.store.set_vault_operation_settings(G, week_start=1, expected_revision=1)
        self.assertEqual(self.store.vault_operation_settings(G)["week_start"], 3)

    def test_bad_values_are_refused_and_write_nothing(self):
        before = self.store.vault_operation_settings(G)
        for bad in ({"enabled": 1}, {"week_start": 7}, {"week_start": True}, {"stage_protocols": ["standard"]}, {"stage_protocols": ["standard", "standard", "x"]},
                    {"contribution_reward": -1}, {"team_reward": True}, {"min_guesses": 0}, {"min_guesses": 31}, {"channel_id": "a"}):
            with self.subTest(bad=bad), self.assertRaises(UserError):
                self.store.set_vault_operation_settings(G, **bad)
        self.assertEqual(self.store.vault_operation_settings(G), before)


class LifecycleTests(OpCase):
    def test_first_join_creates_the_week_with_three_stages_and_two_persisted_npcs(self):
        """Regression (FEAT-66): created on first join, only stage 1 is open, the NPCs are the first two cast members."""
        view = self.start()
        self.assertEqual((view["created"], view["joined"], view["status"], view["channel_id"], view["guild_id"]), (True, True, "active", CH, G))
        self.assertEqual([n["character_id"] for n in view["npcs"]], [self.ann, self.bea])
        self.assertEqual([n["name"] for n in view["npcs"]], ["Ann", "Bea"])
        self.assertEqual([s["status"] for s in view["stages"]], ["active", "locked", "locked"])
        self.assertEqual([s["protocol"] for s in view["stages"]], ["standard", "standard", "open"])
        self.assertEqual(view["stage"]["number"], 1)
        self.assertEqual((view["stage"]["turn_budget"], view["stage"]["turns_used"], view["stage"]["turns_left"], view["stage"]["retry"]), (3, 0, 3, 0))
        self.assertEqual([p["user_id"] for p in view["participants"]], [ALICE])
        again = self.join(user=BOB)
        self.assertEqual([n["character_id"] for n in again["npcs"]], [self.ann, self.bea])
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_operations")[0], 1)

    def test_explicit_npcs_are_kept_and_must_be_two_eligible_cast_members(self):
        for bad in ([self.ann], [self.ann, self.ann], [self.ann, self.far], [self.ann, self.cy], [self.ann, True], "ab"):
            with self.subTest(npc_ids=bad):
                self.refused("vault.operation_npcs", self.join, npc_ids=bad)
        self.store.set_cast(CH, None, [self.ann, self.bea, self.cy])
        view = self.join(npc_ids=[self.cy, self.ann])
        self.assertEqual([n["character_id"] for n in view["npcs"]], [self.cy, self.ann])

    def test_one_eligible_cast_member_is_not_enough(self):
        self.store.set_cast(CH, None, [self.ann])
        self.refused("vault.operation_npcs", self.join)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_operations")[0], 0)

    def test_off_by_default_and_the_channel_is_the_configured_one(self):
        fresh = Store()
        self.addCleanup(fresh.close)
        w = fresh.create_space(G, "W", "world")
        a, b = fresh.create_character(G, w, "A"), fresh.create_character(G, w, "B")
        fresh.bind_channel(G, CH, w)
        fresh.set_cast(CH, None, [a, b])
        self.assertEqual(code_of(self.assertRaises_(fresh.open_vault_operation, G, ALICE, CH, now=NOW)), "vault.disabled")
        fresh.set_vault_settings(G, enabled=True)
        self.assertEqual(code_of(self.assertRaises_(fresh.open_vault_operation, G, ALICE, CH, now=NOW)), "vault.operation_disabled")
        fresh.set_vault_operation_settings(G, enabled=True)
        self.assertEqual(code_of(self.assertRaises_(fresh.open_vault_operation, G, ALICE, CH, now=NOW)), "vault.operation_channel")
        fresh.set_vault_operation_settings(G, channel_id=CH + 1)
        self.assertEqual(code_of(self.assertRaises_(fresh.open_vault_operation, G, ALICE, CH, now=NOW)), "vault.operation_channel")
        self.assertEqual(fresh.one("SELECT COUNT(*) FROM vault_operations")[0], 0)

    def assertRaises_(self, call, *args, **kw):
        with self.assertRaises(VaultStoreError) as ctx:
            call(*args, **kw)
        return ctx.exception

    def test_join_adds_a_participant_once_costs_no_turn_and_is_not_a_contribution(self):
        self.start()
        joined = self.join(user=BOB)
        self.assertEqual((joined["created"], joined["joined"], joined["id"]), (False, True, self.oid))
        self.assertEqual([p["user_id"] for p in joined["participants"]], [ALICE, BOB])
        again = self.join(user=BOB)
        self.assertEqual((again["joined"], len(again["participants"])), (False, 2))
        self.assertEqual((again["stage"]["turns_used"], [p["contributions"] for p in again["participants"]]), (0, [0, 0]))

    def test_only_participants_can_view_or_act(self):
        self.start()
        self.refused("vault.operation_not_participant", self.store.vault_operation, G, BOB, self.oid, now=NOW)
        for call, args in ((self.store.submit_operation_guess, ("1234",)), (self.store.reveal_operation_clue, (1,)),
                           (self.store.use_operation_tool, (SPECTRUM, {"position": 1})), (self.store.operation_cross_validate, (1, 2)),
                           (self.store.record_operation_advice, (self.ann,)), (self.store.set_operation_advisor, (self.ann,))):
            with self.subTest(call=call.__name__):
                self.refused("vault.operation_not_participant", call, G, BOB, self.oid, *args, now=NOW)
        self.assertEqual(self.op()["stage"]["turns_used"], 0)

    def test_current_operation_id_follows_the_channel_and_week(self):
        self.assertIsNone(self.store.current_vault_operation_id(G, CH, now=NOW))
        self.start()
        self.assertEqual(self.store.current_vault_operation_id(G, CH, now=NOW), self.oid)
        self.assertIsNone(self.store.current_vault_operation_id(G, CH + 1, now=NOW))
        self.assertIsNone(self.store.current_vault_operation_id(G2, CH, now=NOW))
        self.assertIsNone(self.store.current_vault_operation_id(G, CH, now=MONDAY + WEEK))


class WeekTests(OpCase):
    def test_week_runs_from_the_server_week_start_day_and_is_stored(self):
        view = self.start()
        self.assertEqual((view["week_key"], view["starts_at"], view["expires_at"]), ("2026-10-12", MONDAY, MONDAY + WEEK))

    def test_week_start_setting_moves_the_boundary(self):
        self.store.set_vault_operation_settings(G, week_start=3)
        view = self.start()
        self.assertEqual((view["week_key"], view["starts_at"]), ("2026-10-15", MONDAY + 3 * 86400))
        self.store.set_vault_operation_settings(G, week_start=4)
        other = self.join(now=NOW + 86400)  # Friday: the same week-start Thursday's operation is still running
        self.assertEqual(other["id"], self.oid)

    def test_week_uses_the_server_timezone_and_a_later_change_moves_nothing(self):
        self.store.set_guild_timezone(G, "Asia/Seoul")
        sunday_night_utc = MONDAY + WEEK - 8 * 3600     # Sunday 16:00 UTC is already Monday 01:00 in Seoul
        view = self.start(now=sunday_night_utc)
        self.assertEqual(view["week_key"], "2026-10-19")
        self.assertEqual(view["starts_at"], MONDAY + WEEK - 9 * 3600)
        self.store.set_guild_timezone(G, "America/Los_Angeles")
        later = self.op(now=sunday_night_utc + 60)
        self.assertEqual((later["week_key"], later["starts_at"], later["expires_at"]), (view["week_key"], view["starts_at"], view["expires_at"]))
        self.assertEqual(self.join(user=BOB, now=sunday_night_utc + 60)["id"], self.oid)

    def test_one_operation_per_guild_channel_week(self):
        self.start()
        for user in (BOB, CAROL, ALICE):
            self.assertEqual(self.join(user=user, now=NOW + 3600)["id"], self.oid)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_operations WHERE guild_id=? AND channel_id=?", (G, CH))[0], 1)

    def test_week_row_is_unique_in_the_database(self):
        self.start()
        row = self.store.one("SELECT * FROM vault_operations WHERE id=?", (self.oid,))
        columns = [c for c in row.keys() if c not in ("id",)]
        values = {c: row[c] for c in columns}
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.db.execute(f"INSERT INTO vault_operations({','.join(columns)}) VALUES({','.join('?' * len(columns))})", tuple(values.values()))

    def test_after_expiry_the_old_operation_ends_and_a_new_week_starts(self):
        self.start()
        self.burn(1)
        later = MONDAY + WEEK + 60
        new = self.join(user=BOB, now=later)
        self.assertEqual((new["created"], new["week_key"], new["status"]), (True, "2026-10-19", "active"))
        self.assertNotEqual(new["id"], self.oid)
        old = self.store.one("SELECT status FROM vault_operations WHERE id=?", (self.oid,))
        self.assertEqual(old["status"], "expired")
        self.assertEqual(self.store.one("SELECT status FROM vault_operation_stages WHERE operation_id=? AND number=1", (self.oid,))["status"], "expired")
        self.refused("vault.not_active", self.g, "0123", now=later)  # Alice never joined the new week; the old one is over
        self.assertEqual(self.store.vault_operation(G, ALICE, self.oid, now=later)["status"], "expired")

    def test_joining_a_finished_operation_is_refused_and_records_nobody(self):
        """FEAT-66: a solved, failed or expired operation takes no new participant."""
        self.start()
        self.solve_all()
        self.refused("vault.not_active", self.join, user=BOB)
        self.assertEqual([p["user_id"] for p in self.op()["participants"]], [ALICE])
        self.store.db.execute("UPDATE vault_operations SET status='failed' WHERE id=?", (self.oid,))
        self.store.db.commit()
        self.refused("vault.not_active", self.join, user=BOB)
        self.store.db.execute("UPDATE vault_operations SET status='expired' WHERE id=?", (self.oid,))
        self.store.db.commit()
        self.refused("vault.not_active", self.join, user=BOB)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_operation_participants WHERE operation_id=?", (self.oid,))[0], 1)

    def test_actions_at_expiry_are_refused_and_the_stage_secret_is_shown(self):
        self.start()
        secret = self.secret()
        self.refused("vault.not_active", self.g, self.wrongs()[0], now=MONDAY + WEEK)
        ended = self.op(now=MONDAY + WEEK)
        self.assertEqual((ended["status"], ended["stages"][0]["status"], ended["stages"][0]["secret"]), ("expired", "expired", secret))
        self.assertEqual(self.store.expire_vault_sessions(G, now=MONDAY + WEEK), 0)


class StageTests(OpCase):
    def setUp(self):
        super().setUp()
        self.start()
        self.join(user=BOB)
        self.join(user=CAROL)

    def test_each_stage_has_its_own_secret_made_when_it_starts(self):
        first = self.secret(1)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_operation_stages WHERE operation_id=?", (self.oid,))[0], 1)
        self.solve()
        second = self.secret(2)
        self.solve()
        third = self.secret(3)
        self.assertEqual(len({first, second, third}), 3)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_operation_stages WHERE operation_id=?", (self.oid,))[0], 3)

    def test_solving_opens_the_next_stage_and_the_third_solves_the_operation(self):
        events, views = [], []
        for _ in range(3):
            result = self.solve()
            events.append(result["event"])
            views.append(result["operation"])
        self.assertEqual(events, ["stage_solved", "stage_solved", "operation_solved"])
        self.assertEqual([s["status"] for s in views[0]["stages"]], ["solved", "active", "locked"])
        self.assertEqual([s["status"] for s in views[1]["stages"]], ["solved", "solved", "active"])
        self.assertEqual(views[2]["status"], "solved")
        self.assertIsNone(views[2]["stage"])
        self.assertEqual([s["status"] for s in views[2]["stages"]], ["solved"] * 3)
        self.assertEqual(views[0]["stage"]["number"], 2)
        self.refused("vault.not_active", self.g, "0123")

    def test_open_stage_shows_one_digit_and_stage_budget_comes_from_settings_at_stage_start(self):
        self.solve()
        self.solve()
        stage3 = self.op()["stage"]
        self.assertEqual((stage3["protocol"], stage3["turn_budget"]), ("open", 3))
        self.assertIn(stage3["open_digit"], self.secret(3))
        self.assertIsNone(self.op()["stages"][0]["open_digit"])

    def test_custom_stage_protocols_apply_to_a_new_operation(self):
        self.store.set_vault_operation_settings(G, stage_protocols=["open", "standard", "standard"])
        view = self.join(now=MONDAY + WEEK + 60, user=ALICE)
        self.assertEqual([s["protocol"] for s in view["stages"]], ["open", "standard", "standard"])
        self.assertIn(view["stage"]["open_digit"], self.store.one("SELECT secret FROM vault_operation_stages WHERE operation_id=?", (view["id"],))["secret"])

    def test_a_late_budget_change_affects_only_stages_not_started(self):
        self.store.set_vault_settings(G, turns={"standard": 5})
        self.assertEqual(self.op()["stage"]["turn_budget"], 3)
        self.solve()
        self.assertEqual(self.op()["stage"]["turn_budget"], 5)

    def test_secret_stays_the_same_after_reopening_the_database(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "op.sqlite3"
            store = Store(path)
            w = store.create_space(G, "W", "world")
            a, b = store.create_character(G, w, "A"), store.create_character(G, w, "B")
            store.bind_channel(G, CH, w)
            store.set_cast(CH, None, [a, b])
            store.set_vault_settings(G, enabled=True)
            store.set_vault_operation_settings(G, enabled=True, channel_id=CH)
            oid = store.open_vault_operation(G, ALICE, CH, now=NOW, seed="persist")["id"]
            before = store.one("SELECT secret FROM vault_operation_stages WHERE operation_id=?", (oid,))["secret"]
            clues = [c["position"] for c in store.vault_operation(G, ALICE, oid, now=NOW)["stage"]["clues"]]
            store.close()
            store = Store(path)
            try:
                self.assertEqual(store.one("SELECT secret FROM vault_operation_stages WHERE operation_id=?", (oid,))["secret"], before)
                again = store.open_vault_operation(G, BOB, CH, now=NOW + 5, seed="other")
                self.assertEqual((again["id"], again["created"]), (oid, False))
                self.assertEqual([c["position"] for c in again["stage"]["clues"]], clues)
            finally:
                store.close()


class SharedTurnTests(OpCase):
    def setUp(self):
        super().setUp()
        self.start()
        self.join(user=BOB)
        self.join(user=CAROL)

    def test_turns_are_spent_by_everyone_from_one_budget(self):
        a, b, c = self.wrongs(3)
        self.assertEqual(self.g(a, ALICE)["operation"]["stage"]["turns_left"], 2)
        self.assertEqual(self.g(b, BOB)["operation"]["stage"]["turns_left"], 1)
        self.assertEqual(self.op(CAROL)["stage"]["turns_used"], 2)
        self.assertEqual([(x["number"], x["user_id"]) for x in self.op()["stage"]["guesses"]], [(1, ALICE), (2, BOB)])

    def test_guess_numbers_run_across_members_and_the_result_reports_them(self):
        a, b = self.wrongs(2)
        first, second = self.g(a, ALICE), self.g(b, BOB)
        self.assertEqual((first["number"], second["number"], second["won"], second["replayed"], second["event"]), (1, 2, False, False, None))
        self.assertEqual(first["feedback"], feedback("standard", self.secret(), a))

    def test_the_last_shared_turn_triggers_the_retry_whoever_spends_it(self):
        a, b, c = self.wrongs(3)
        self.g(a, ALICE)
        self.g(b, BOB)
        result = self.g(c, CAROL)
        self.assertEqual(result["event"], "stage_retry")
        stage = result["operation"]["stage"]
        self.assertEqual((stage["number"], stage["retry"], stage["turns_used"], stage["turns_left"], stage["guesses"], result["operation"]["status"]), (1, 1, 0, 3, [], "active"))

    def test_a_duplicate_across_members_is_free_and_changes_nothing(self):
        a = self.wrongs()[0]
        self.g(a, ALICE)
        err = self.refused("vault.guess_duplicate", self.g, a, BOB)
        self.assertNotIn(self.secret(), repr(err.params))
        self.assertEqual(self.op()["stage"]["turns_used"], 1)

    def test_invalid_guesses_are_free(self):
        for bad, code in (("123", "vault.guess_length"), ("12a4", "vault.guess_not_digits"), ("1123", "vault.guess_repeated_digit")):
            self.refused(code, self.g, bad, BOB)
        self.assertEqual(self.op()["stage"]["turns_used"], 0)

    def test_retry_gets_a_new_secret_clues_and_budget_and_keeps_the_retry_budget_separate(self):
        old = self.secret()
        old_clues = self.op()["stage"]["clues"]
        self.burn(3)
        stage = self.op()["stage"]
        new = self.secret()
        self.assertNotEqual(new, old)
        self.assertEqual((stage["retry"], stage["turn_budget"], stage["turns_left"]), (1, 3, 3))
        self.assertEqual(len(stage["clues"]), 2)
        self.assertTrue(all(not c["revealed"] and "constraint" not in c for c in stage["clues"]))
        self.assertEqual(len(old_clues), 2)
        self.assertEqual(self.g(self.wrongs()[0], BOB)["number"], 1)
        self.assertEqual(self.g(old, BOB)["number"], 2)      # the old secret is only a wrong guess now

    def test_a_retry_can_be_won_and_the_run_continues(self):
        self.burn(3)
        result = self.solve(BOB)
        self.assertEqual((result["event"], result["operation"]["stage"]["number"]), ("stage_solved", 2))
        self.assertEqual(result["operation"]["stages"][0]["status"], "solved")

    def test_a_second_failure_of_a_stage_fails_the_operation_and_shows_the_secrets(self):
        self.burn(3)
        second = self.secret()
        result = self.burn(3, BOB)
        self.assertEqual(result["event"], "operation_failed")
        view = result["operation"]
        self.assertEqual((view["status"], view["stages"][0]["status"], view["stages"][0]["secret"], view["stage"]), ("failed", "failed", second, None))
        self.refused("vault.not_active", self.g, "0123")
        self.refused("vault.not_active", self.g, "0123", user=BOB)

    def test_one_retry_per_stage_but_each_stage_has_its_own(self):
        self.solve()
        self.burn(3)
        self.assertEqual(self.op()["stage"]["retry"], 1)
        self.solve()
        self.burn(3)
        stage = self.op()["stage"]
        self.assertEqual((stage["number"], stage["retry"], self.op()["status"]), (3, 1, "active"))

    def test_active_views_never_hold_the_secret(self):
        secret = self.secret()
        self.burn(1)
        view = self.op(BOB)
        text = repr(view)
        self.assertNotIn(secret, text)
        self.assertEqual([s["secret"] for s in view["stages"]], [None, None, None])
        self.assertIsNone(view["stage"]["secret"])
        for result in (self.store.reveal_operation_clue(G, BOB, self.oid, 1, now=NOW), self.g(self.wrongs()[0], CAROL)):
            self.assertNotIn(secret, repr(result.get("operation", result)) if isinstance(result, dict) else repr(result))
        err = self.refused("vault.guess_length", self.g, "12", BOB)
        self.assertNotIn(secret, repr(err.params) + str(err.code))

    def test_the_solved_stage_shows_its_secret_and_all_its_clues(self):
        secret = self.secret()
        result = self.solve(BOB)
        stage1 = result["operation"]["stages"][0]
        self.assertEqual((stage1["status"], stage1["secret"]), ("solved", secret))
        self.assertTrue(all("constraint" in c for c in stage1["clues"]))
        self.assertIsNone(result["operation"]["stages"][1]["secret"])

    def test_idempotent_guess_by_action_key(self):
        a, b = self.wrongs(2)
        first = self.g(a, ALICE, action_key="i-1")
        again = self.g(a, ALICE, action_key="i-1")
        self.assertEqual((first["replayed"], again["replayed"], again["number"], again["feedback"]), (False, True, 1, first["feedback"]))
        self.assertEqual(self.op()["stage"]["turns_used"], 1)
        self.g(b, BOB)
        self.assertTrue(self.g(a, BOB, action_key="i-1")["replayed"])      # keys are per operation, whoever repeats it
        self.assertEqual(self.op()["stage"]["turns_used"], 2)

    def test_replay_of_the_winning_guess_after_the_stage_moved_on_changes_nothing(self):
        won = self.g(self.secret(), ALICE, action_key="win")
        self.assertEqual(won["event"], "stage_solved")
        again = self.g(self.secret(1), ALICE, action_key="win")
        self.assertTrue(again["replayed"])
        self.assertEqual((self.op()["stage"]["number"], self.op()["stage"]["turns_used"]), (2, 0))

    def test_guess_that_wins_on_the_last_turn_does_not_fail(self):
        self.burn(2)
        result = self.solve(CAROL)
        self.assertEqual((result["won"], result["event"]), (True, "stage_solved"))
        self.assertEqual(self.op()["stages"][0]["turns_used"], 3)


class ClueTests(OpCase):
    def setUp(self):
        super().setUp()
        self.start()
        self.join(user=BOB)

    def test_the_two_npcs_hold_one_persisted_clue_each_and_it_is_hidden_until_revealed(self):
        clues = self.op()["stage"]["clues"]
        self.assertEqual([(c["position"], c["character_id"], c["revealed"]) for c in clues], [(1, self.ann, False), (2, self.bea, False)])
        self.assertTrue(all("constraint" not in c for c in clues))
        rows = self.store.all("SELECT * FROM vault_operation_clues WHERE operation_id=? ORDER BY position", (self.oid,))
        self.assertEqual([r["character_id"] for r in rows], [self.ann, self.bea])

    def test_a_late_join_adds_no_clues(self):
        self.join(user=CAROL)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_operation_clues WHERE operation_id=?", (self.oid,))[0], 2)

    def test_reveal_is_free_shared_by_the_team_and_idempotent(self):
        first = self.store.reveal_operation_clue(G, ALICE, self.oid, 1, now=NOW)
        later = self.store.reveal_operation_clue(G, BOB, self.oid, 1, now=NOW + 60)
        self.assertEqual((first["revealed"], later["revealed_at"], later["character_id"]), (True, NOW, self.ann))
        self.assertTrue(first["constraint"].holds(self.secret()))
        stage = self.op(BOB)["stage"]
        self.assertEqual((stage["turns_used"], [c["revealed"] for c in stage["clues"]]), (0, [True, False]))
        self.assertEqual(self.op(BOB)["participants"][0]["contributions"], 0)

    def test_bad_positions_are_refused(self):
        for position in (0, 3, True, "1", None):
            with self.subTest(position=position):
                self.refused("vault.clue_unknown", self.store.reveal_operation_clue, G, ALICE, self.oid, position, now=NOW)


class AdvisorTests(OpCase):
    def setUp(self):
        super().setUp()
        s = self.store
        s.add_favorite(G, ALICE, self.cy)
        s.add_favorite(G, ALICE, self.ann)
        s.add_favorite(G, ALICE, self.far)
        self.start()
        self.join(user=BOB)
        s.add_favorite(G, BOB, self.cy)

    def advisor(self, user, cid, **kw):
        return self.store.set_operation_advisor(G, user, self.oid, cid, now=NOW, **kw)

    def test_one_favorite_per_member_is_an_advisor_and_replaces_the_previous(self):
        view = self.advisor(ALICE, self.cy)
        self.assertEqual(view["participants"][0]["advisor"], {"character_id": self.cy, "name": "Cy"})
        self.assertIsNone(view["participants"][1]["advisor"])
        view = self.advisor(ALICE, self.ann)
        self.assertEqual(view["participants"][0]["advisor"]["character_id"], self.ann)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_operation_participants WHERE operation_id=? AND user_id=? AND advisor_id IS NOT NULL", (self.oid, ALICE))[0], 1)
        self.assertIsNone(self.advisor(ALICE, None)["participants"][0]["advisor"])

    def test_only_the_members_own_eligible_favorites_qualify(self):
        self.refused("vault.partner_unavailable", self.advisor, ALICE, self.bea)    # cast, not a favorite
        self.refused("vault.partner_unavailable", self.advisor, ALICE, self.far)    # favorite of a world the channel cannot see
        self.refused("vault.partner_unavailable", self.advisor, BOB, self.ann)      # someone else's favorite
        self.refused("vault.partner_unavailable", self.advisor, ALICE, True)
        self.assertIsNone(self.op()["participants"][0]["advisor"])

    def test_archived_favorites_only_while_the_server_allows_them(self):
        self.store.db.execute("UPDATE characters SET archived=1 WHERE id=?", (self.cy,))
        self.store.db.commit()
        self.refused("vault.partner_unavailable", self.advisor, ALICE, self.cy)
        self.store.set_archived_favorites(G, True)
        self.assertEqual(self.advisor(ALICE, self.cy)["participants"][0]["advisor"]["character_id"], self.cy)

    def test_a_favorite_never_holds_a_clue_and_two_members_may_bring_the_same_one(self):
        self.advisor(ALICE, self.cy)
        self.advisor(BOB, self.cy)
        self.assertEqual({c["character_id"] for c in self.op()["stage"]["clues"]}, {self.ann, self.bea})
        self.assertEqual([p["advisor"]["character_id"] for p in self.op()["participants"]], [self.cy, self.cy])

    def test_advice_from_an_npc_or_my_advisor_costs_no_turn_and_is_not_a_contribution(self):
        self.advisor(ALICE, self.cy)
        npc = self.store.record_operation_advice(G, ALICE, self.oid, self.ann, now=NOW)
        mine = self.store.record_operation_advice(G, ALICE, self.oid, self.cy, now=NOW)
        self.assertEqual((npc["character_id"], mine["character_id"], npc["source"]), (self.ann, self.cy, "template"))
        view = self.op(BOB)
        self.assertEqual((view["stage"]["turns_used"], len(view["stage"]["advice"]), view["participants"][0]["contributions"]), (0, 2, 0))
        self.assertNotIn(self.secret(), repr(view["stage"]["advice"]))

    def test_advice_from_anyone_else_is_refused_free(self):
        self.advisor(ALICE, self.cy)
        talk = self.store.game_talk_calls_today(G, NOW)
        for user, cid in ((BOB, self.cy), (ALICE, self.far), (ALICE, True), (ALICE, 9999)):   # Alice's advisor is not Bob's
            with self.subTest(user=user, cid=cid):
                self.refused("vault.advice_not_partner", self.store.record_operation_advice, G, user, self.oid, cid, now=NOW)
        self.assertEqual(self.store.game_talk_calls_today(G, NOW), talk)

    def test_advice_cap_is_per_stage_and_shared_and_talk_limit_applies(self):
        self.store.set_vault_settings(G, advice_cap=2)
        self.store.record_operation_advice(G, ALICE, self.oid, self.ann, now=NOW)
        self.store.record_operation_advice(G, BOB, self.oid, self.bea, now=NOW)
        err = self.refused("vault.advice_limit", self.store.record_operation_advice, G, ALICE, self.oid, self.ann, now=NOW)
        self.assertEqual(err.params["limit"], 2)
        self.solve()
        self.store.record_operation_advice(G, ALICE, self.oid, self.ann, now=NOW)
        self.store.set_game_talk_daily_limit(G, self.store.game_talk_calls_today(G, NOW))
        self.refused("vault.advice_talk_limit", self.store.record_operation_advice, G, BOB, self.oid, self.bea, now=NOW)

    def test_advice_action_key_replays_without_spending(self):
        first = self.store.record_operation_advice(G, ALICE, self.oid, self.ann, action_key="adv", now=NOW)
        talk = self.store.game_talk_calls_today(G, NOW)
        again = self.store.record_operation_advice(G, ALICE, self.oid, self.ann, action_key="adv", now=NOW)
        self.assertEqual((again["replayed"], again["id"], self.store.game_talk_calls_today(G, NOW)), (True, first["id"], talk))


class ToolTests(OpCase):
    def setUp(self):
        super().setUp()
        self.store.set_vault_settings(G, turns={"standard": 9, "open": 9})
        self.start()
        self.join(user=BOB)
        self.join(user=CAROL)

    def test_a_tool_spends_the_callers_own_item_and_a_shared_turn(self):
        self.give(SPECTRUM, 2, ALICE)
        self.give(SPECTRUM, 1, BOB)
        result = self.tool({"position": 1}, ALICE)
        self.assertEqual((result["qty"], result["turn_cost"], result["replayed"]), (1, 1, False))
        self.assertEqual(self.store.item_qty(G, BOB, SPECTRUM), 1)
        stage = result["operation"]["stage"]
        self.assertEqual((stage["turns_used"], stage["tools_used"], stage["tools"][0]["user_id"]), (1, 1, ALICE))
        self.assertTrue(result["constraint"].holds(self.secret()))
        self.assertEqual(self.op(BOB)["stage"]["turns_left"], 8)
        self.assertEqual(len(self.store.item_use_log(G)), 1)

    def test_a_teammates_inventory_is_never_used(self):
        self.give(SPECTRUM, 3, BOB)
        before = (self.store.item_qty(G, BOB, SPECTRUM), len(self.store.item_use_log(G)), self.op()["stage"]["turns_used"])
        self.refused("vault.tool_not_held", self.tool, {"position": 1}, ALICE)
        self.assertEqual((self.store.item_qty(G, BOB, SPECTRUM), len(self.store.item_use_log(G)), self.op()["stage"]["turns_used"]), before)

    def test_at_most_two_tools_per_stage_across_the_team(self):
        for user in (ALICE, BOB, CAROL):
            self.give(SPECTRUM, 2, user)
        self.tool({"position": 1}, ALICE)
        self.tool({"position": 2}, BOB)
        err = self.refused("vault.tool_limit", self.tool, {"position": 3}, CAROL)
        self.assertEqual(err.params["limit"], 2)
        self.assertEqual((self.store.item_qty(G, CAROL, SPECTRUM), self.op()["stage"]["tools_used"], self.op()["stage"]["turns_used"]), (2, 2, 2))
        self.refused("vault.tool_limit", self.tool, {"position": 3}, ALICE)

    def test_the_cap_resets_for_the_next_stage_and_the_retry(self):
        self.give(SPECTRUM, 6, ALICE)
        self.tool({"position": 1})
        self.tool({"position": 2})
        self.solve(BOB)
        self.assertEqual(self.op()["stage"]["tools_used"], 0)
        self.tool({"position": 1})
        self.tool({"position": 2})
        self.assertEqual((self.op()["stage"]["tools_used"], self.store.item_qty(G, ALICE, SPECTRUM)), (2, 2))

    def test_refusals_are_free_and_keep_the_item(self):
        self.give(SPECTRUM, 1, ALICE)
        before = lambda: (self.store.item_qty(G, ALICE, SPECTRUM), self.op()["stage"]["turns_used"], len(self.store.item_use_log(G)))  # noqa: E731
        start = before()
        self.refused("vault.tool_unknown", self.tool, {}, ALICE, key="nope")
        self.refused("vault.tool_bad_params", self.tool, {"position": 9}, ALICE)
        self.assertEqual(before(), start)

    def test_not_enough_shared_turns_refuses_the_master_decoder_without_consuming(self):
        self.store.set_vault_settings(G, turns={"standard": 2})
        self.solve()  # next stage (standard) is made with the 2-turn budget
        self.give("vault_master_decoder", 1, ALICE)
        self.burn(1, BOB)
        self.refused("vault.tool_no_turns", self.tool, {"a": "1", "b": "2"}, ALICE, key="vault_master_decoder")
        self.assertEqual(self.store.item_qty(G, ALICE, "vault_master_decoder"), 1)

    def test_a_tool_spending_the_last_turn_triggers_the_retry(self):
        self.store.set_vault_settings(G, turns={"standard": 3})
        self.solve()
        self.give(SPECTRUM, 1, BOB)
        self.burn(2, ALICE)
        result = self.tool({"position": 1}, BOB)
        self.assertEqual(result["event"], "stage_retry")
        self.assertEqual(self.op()["stage"]["retry"], 1)
        self.assertEqual(self.store.item_qty(G, BOB, SPECTRUM), 0)

    def test_tool_action_key_replays_without_a_second_item(self):
        self.give(SPECTRUM, 2, ALICE)
        first = self.tool({"position": 1}, ALICE, action_key="t-1")
        again = self.tool({"position": 1}, ALICE, action_key="t-1")
        self.assertEqual((again["replayed"], again["qty"], again["text_key"]), (True, 1, first["text_key"]))
        self.assertEqual((self.op()["stage"]["tools_used"], len(self.store.item_use_log(G))), (1, 1))

    def test_tool_in_a_finished_operation_is_refused(self):
        self.solve_all()
        self.give(SPECTRUM, 1, ALICE)
        self.refused("vault.not_active", self.tool, {"position": 1}, ALICE)
        self.assertEqual(self.store.item_qty(G, ALICE, SPECTRUM), 1)


class CrossValidationTests(OpCase):
    def setUp(self):
        super().setUp()
        self.store.set_vault_settings(G, turns={"standard": 9, "open": 9})
        self.start()
        self.join(user=BOB)
        self.join(user=CAROL)

    def pair(self):
        """Two wrong guesses sharing digits whose count is still undecided by the public record."""
        secret = self.secret()
        wrong = [c for c in all_secrets(4) if c != secret][:60]
        for a, b in itertools.permutations(wrong, 2):
            common = set(a) & set(b)
            if not common:
                continue
            known = [GuessResult(x, "standard", feedback("standard", secret, x)) for x in (a, b)]
            counts = {len(common & set(c)) for c in filter_candidates(known, 4)}
            if len(counts) > 1:
                return a, b
        self.fail("no undecided pair for this seed")

    def two_guesses(self):
        a, b = self.pair()
        self.g(a, ALICE)
        self.g(b, BOB)
        return a, b

    def cross(self, x=1, y=2, user=CAROL, **kw):
        return self.store.operation_cross_validate(G, user, self.oid, x, y, now=NOW, **kw)

    def test_the_count_of_shared_digits_in_the_secret_is_revealed_and_becomes_public(self):
        a, b = self.two_guesses()
        result = self.cross()
        self.assertEqual(result["count"], cross_validation(self.secret(), a, b))
        self.assertEqual((result["common"], result["turn_cost"], result["numbers"], result["replayed"]), (sorted(set(a) & set(b)), 1, (1, 2), False))
        stage = result["operation"]["stage"]
        self.assertEqual(stage["turns_used"], 3)
        self.assertEqual(stage["cross"], [{"numbers": (1, 2), "common": sorted(set(a) & set(b)), "count": result["count"], "user_id": CAROL}])
        self.assertEqual(stage["tools_used"], 0)
        common = set(a) & set(b)
        remaining = filter_candidates(stage["constraints"], 4)
        self.assertIn(self.secret(), remaining)
        self.assertTrue(all(len(common & set(c)) == result["count"] for c in remaining))

    def test_it_needs_no_item_and_counts_as_a_contribution(self):
        self.two_guesses()
        self.cross()
        self.assertEqual({p["user_id"]: p["contributions"] for p in self.op()["participants"]}, {ALICE: 1, BOB: 1, CAROL: 1})
        self.assertEqual(len(self.store.item_use_log(G)), 0)

    def test_two_guesses_by_one_member_do_not_count(self):
        a, b = self.pair()
        self.g(a, ALICE)
        self.g(b, ALICE)
        self.refused("vault.cross_same_member", self.cross, 1, 2, CAROL)
        self.assertEqual(self.op()["stage"]["turns_used"], 2)

    def test_unknown_or_malformed_guess_numbers_are_refused(self):
        self.two_guesses()
        for x, y in ((1, 1), (1, 3), (0, 2), (True, 2), ("1", 2), (None, 2)):
            with self.subTest(numbers=(x, y)):
                self.refused("vault.cross_unknown", self.cross, x, y)
        self.assertEqual(self.op()["stage"]["turns_used"], 2)

    def test_once_per_stage_attempt(self):
        a, b = self.two_guesses()
        secret, common = self.secret(), set(a) & set(b)
        known = [GuessResult(x, "standard", feedback("standard", secret, x)) for x in (a, b)]

        def undecided(x):   # the (1, 2) count must stay open after the third guess, or the first cross is rightly redundant
            counts = {len(common & set(k)) for k in filter_candidates(known + [GuessResult(x, "standard", feedback("standard", secret, x))], 4)}
            return len(counts) > 1
        c = next(x for x in self.wrongs(300, BOB) if set(x) & set(a) and x not in (a, b) and undecided(x))
        self.g(c, CAROL)
        self.cross(1, 2)
        self.refused("vault.cross_used", self.cross, 1, 3, BOB)
        self.assertEqual(self.op()["stage"]["turns_used"], 4)

    def test_guesses_with_no_shared_digit_teach_nothing(self):
        secret = self.secret()
        wrong = [c for c in all_secrets(4) if c != secret]
        a = wrong[0]
        b = next(c for c in wrong if not set(c) & set(a))
        self.g(a, ALICE)
        self.g(b, BOB)
        self.refused("vault.cross_redundant", self.cross)
        self.assertEqual(self.op()["stage"]["turns_used"], 2)

    def test_needs_a_turn_left(self):
        self.store.set_vault_settings(G, turns={"standard": 3})
        self.solve()
        self.store.set_vault_settings(G, turns={"standard": 3, "open": 9})
        a, b = self.pair()
        self.g(a, ALICE)
        self.g(b, BOB)
        third = next(c for c in self.wrongs(30) if c not in (a, b))
        self.g(third, CAROL)
        self.assertEqual(self.op()["stage"]["retry"], 1)   # the budget ran out: a new attempt has no guesses to compare
        self.refused("vault.cross_unknown", self.cross)

    def test_action_key_replays_without_a_second_turn(self):
        self.two_guesses()
        first = self.cross(action_key="cv-1")
        again = self.cross(action_key="cv-1")
        self.assertEqual((again["replayed"], again["count"], first["replayed"]), (True, first["count"], False))
        self.assertEqual(self.op()["stage"]["turns_used"], 3)

    def test_cross_validation_comes_from_a_participant_only(self):
        self.two_guesses()
        outsider = 999
        self.refused("vault.operation_not_participant", self.store.operation_cross_validate, G, outsider, self.oid, 1, 2, now=NOW)


class RewardTests(OpCase):
    def setUp(self):
        super().setUp()
        self.store.set_vault_operation_settings(G, contribution_reward=20, team_reward=50)
        self.store.set_vault_settings(G, turns={"standard": 9, "open": 9})
        self.start()
        self.join(user=BOB)
        self.join(user=CAROL)

    def vault_ledger(self, user=None):
        rows = [dict(r) for r in self.store.db.execute("SELECT * FROM currency_ledger WHERE guild_id=? AND source='vault' ORDER BY id", (G,))]
        return [r for r in rows if user is None or r["user_id"] == user]

    def participant(self, user):
        return next(p for p in self.op()["participants"] if p["user_id"] == user)

    def test_the_first_valid_contribution_pays_the_contribution_reward_once_at_that_moment(self):
        self.assertEqual(self.vault_ledger(), [])
        result = self.g(self.wrongs()[0], ALICE)
        mine = result["operation"]["participants"][0]
        self.assertEqual((mine["reward_contribution"]["amount"], mine["reward_contribution"]["reversed"], mine["reward_team"]), (20, False, None))
        self.assertEqual(self.store.balance(G, ALICE), 20)
        self.g(self.wrongs()[0], ALICE)
        self.assertEqual(self.store.balance(G, ALICE), 20)
        self.assertEqual(len(self.vault_ledger(ALICE)), 1)
        entry = self.vault_ledger(ALICE)[0]
        self.assertEqual((entry["amount"], entry["source"]), (20, "vault"))
        self.assertEqual(mine["reward_contribution"]["ledger_id"], entry["id"])

    def test_joining_reading_clues_and_advice_pay_nothing(self):
        self.store.reveal_operation_clue(G, BOB, self.oid, 1, now=NOW)
        self.store.record_operation_advice(G, BOB, self.oid, self.ann, now=NOW)
        self.assertEqual(self.vault_ledger(), [])
        self.assertEqual(self.participant(BOB)["reward_contribution"], None)

    def test_rejected_actions_pay_nothing(self):
        for bad in ("12", "12a4", "1123"):
            with self.assertRaises(VaultStoreError):
                self.g(bad, BOB)
        a = self.wrongs()[0]
        self.g(a, ALICE)
        with self.assertRaises(VaultStoreError):
            self.g(a, BOB)
        self.assertEqual(self.store.balance(G, BOB), 0)
        self.assertEqual([r["user_id"] for r in self.vault_ledger()], [ALICE])

    def test_min_guesses_sets_the_threshold(self):
        self.store.set_vault_operation_settings(G, min_guesses=2)
        a, b, c = self.wrongs(3)
        self.g(a, ALICE)
        self.assertEqual(self.store.balance(G, ALICE), 0)
        self.g(b, ALICE)
        self.assertEqual(self.store.balance(G, ALICE), 20)
        self.g(c, ALICE)
        self.assertEqual(self.store.balance(G, ALICE), 20)

    def test_a_tool_use_is_a_contribution(self):
        self.give(SPECTRUM, 1, BOB)
        self.tool({"position": 1}, BOB)
        self.assertEqual(self.store.balance(G, BOB), 20)

    def test_a_replayed_action_pays_nothing_more(self):
        a = self.wrongs()[0]
        self.g(a, ALICE, action_key="k")
        self.g(a, ALICE, action_key="k")
        self.assertEqual(self.store.balance(G, ALICE), 20)

    def test_team_reward_goes_to_every_qualified_member_when_the_operation_is_solved(self):
        self.g(self.wrongs()[0], ALICE)
        self.solve(BOB)
        self.assertEqual([r["amount"] for r in self.vault_ledger()], [20, 20])
        self.solve(BOB)
        self.assertEqual(len(self.vault_ledger()), 2)
        done = self.solve(ALICE)
        self.assertEqual(done["event"], "operation_solved")
        self.assertEqual((self.store.balance(G, ALICE), self.store.balance(G, BOB), self.store.balance(G, CAROL)), (70, 70, 0))
        view = done["operation"]
        self.assertEqual([bool(p["reward_team"]) for p in view["participants"]], [True, True, False])
        self.assertEqual(view["participants"][0]["reward_team"]["amount"], 50)
        self.assertEqual(len(self.vault_ledger()), 4)
        self.refused("vault.not_active", self.g, "0123", user=CAROL)

    def test_a_failed_or_expired_operation_pays_no_team_reward(self):
        self.store.set_vault_settings(G, turns={"standard": 2})
        self.solve()                      # stage 2 is made with the 2-turn budget
        self.burn(2, BOB)
        self.burn(2, BOB)
        self.assertEqual(self.op()["status"], "failed")
        self.assertEqual({r["amount"] for r in self.vault_ledger()}, {20})
        self.assertEqual({r["user_id"] for r in self.vault_ledger()}, {ALICE, BOB})
        self.assertEqual(len(self.vault_ledger()), 2)

    def test_zero_amounts_pay_nothing_and_write_no_claim(self):
        self.store.set_vault_operation_settings(G, contribution_reward=0, team_reward=0)
        self.g(self.wrongs()[0], ALICE)
        self.solve_all()
        self.assertEqual((self.vault_ledger(), self.store.one("SELECT COUNT(*) FROM vault_operation_claims")[0]), ([], 0))

    def test_rewards_in_force_when_earned_are_the_ones_paid(self):
        self.g(self.wrongs()[0], ALICE)
        self.store.set_vault_operation_settings(G, contribution_reward=500)
        self.g(self.wrongs()[0], BOB)
        self.assertEqual((self.store.balance(G, ALICE), self.store.balance(G, BOB)), (20, 500))

    def test_claims_are_unique_per_operation_member_and_type(self):
        self.g(self.wrongs()[0], ALICE)
        row = self.store.one("SELECT * FROM vault_operation_claims WHERE operation_id=? AND user_id=?", (self.oid, ALICE))
        self.assertEqual((row["type"], row["amount"], row["guild_id"]), ("contribution", 20, G))
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.db.execute("INSERT INTO vault_operation_claims(guild_id,operation_id,user_id,type,amount,ledger_id,created_at) VALUES(?,?,?,?,?,?,?)",
                                  (G, self.oid, ALICE, "contribution", 20, row["ledger_id"], NOW))
        self.store.db.rollback()
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_operation_claims")[0], 1)

    def test_reversal_is_flagged_takes_the_money_back_and_never_repays(self):
        first = self.g(self.wrongs()[0], ALICE)
        ledger_id = first["operation"]["participants"][0]["reward_contribution"]["ledger_id"]
        self.store.reverse_entry(G, ledger_id, 1, "undo")
        mine = self.participant(ALICE)
        self.assertEqual((mine["reward_contribution"]["amount"], mine["reward_contribution"]["reversed"]), (20, True))
        self.assertEqual(self.store.balance(G, ALICE), 0)
        self.g(self.wrongs()[0], ALICE)
        self.g(self.wrongs()[0], ALICE)
        self.assertEqual(self.store.balance(G, ALICE), 0)
        self.assertFalse(self.participant(BOB)["reward_contribution"])

    def test_team_reward_reversal_leaves_the_contribution_reward_alone(self):
        self.g(self.wrongs()[0], ALICE)
        for _ in range(3):
            done = self.solve(ALICE)
        team = done["operation"]["participants"][0]["reward_team"]
        self.store.reverse_entry(G, team["ledger_id"], 1, "undo")
        mine = self.participant(ALICE)
        self.assertEqual((mine["reward_team"]["reversed"], mine["reward_contribution"]["reversed"], self.store.balance(G, ALICE)), (True, False, 20))


class IsolationTests(OpCase):
    def test_another_guild_cannot_see_or_act_on_an_operation(self):
        self.start()
        self.join(user=BOB)
        for call, args in ((self.store.vault_operation, ()), (self.store.submit_operation_guess, ("0123",)), (self.store.reveal_operation_clue, (1,)),
                           (self.store.operation_cross_validate, (1, 2)), (self.store.set_operation_advisor, (self.ann,)),
                           (self.store.record_operation_advice, (self.ann,)), (self.store.use_operation_tool, (SPECTRUM, {"position": 1}))):
            with self.subTest(call=call.__name__):
                self.refused("vault.not_found", call, G2, ALICE, self.oid, *args, now=NOW)
        self.assertEqual(self.op()["stage"]["turns_used"], 0)

    def test_guilds_get_their_own_operations_settings_and_rewards(self):
        self.start()
        self.assertEqual(self.store.vault_operation_settings(G2), self.store.vault_operation_settings(12345))
        self.refused("vault.disabled", self.join, guild=G2)
        self.store.set_vault_settings(G2, enabled=True)
        self.refused("vault.operation_disabled", self.join, guild=G2)
        self.g(self.wrongs()[0], ALICE)
        self.assertEqual(self.ledger(G2), [])
        self.assertIsNone(self.store.current_vault_operation_id(G2, CH, now=NOW))

    def test_operation_id_of_another_guild_is_not_found_even_for_a_same_user_participant_there(self):
        self.start()
        s = self.store
        w = s.create_space(G2, "W2", "world")
        a, b = s.create_character(G2, w, "A2"), s.create_character(G2, w, "B2")
        s.bind_channel(G2, CH + 5, w)
        s.set_cast(CH + 5, None, [a, b])
        s.set_vault_settings(G2, enabled=True)
        s.set_vault_operation_settings(G2, enabled=True, channel_id=CH + 5)
        other = s.open_vault_operation(G2, ALICE, CH + 5, now=NOW, seed="g2")
        self.assertNotEqual(other["id"], self.oid)
        self.refused("vault.not_found", s.vault_operation, G2, ALICE, self.oid, now=NOW)
        self.refused("vault.not_found", s.vault_operation, G, ALICE, other["id"], now=NOW)


if __name__ == "__main__":
    unittest.main()
