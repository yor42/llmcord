"""Vault support missions in the store, FEAT-67 phase A, schema v27, decision D43. Written before the implementation; they fail until it exists.

API CONTRACT (new methods on ``VaultOperationStore`` in ``llmcord_core/vault_operation_store.py`` or a mixin next to it; same conventions as
tests/test_vault_operation_store.py: synchronous, ``guild_id`` first, every query guild-scoped, ``now`` Unix seconds, writes in ``write_admin()``,
refusals are ``VaultStoreError`` with ``err.code.code`` and nothing written on a refusal). The pure rules are in tests/test_vault_missions_engine.py.

  Setting      vault_operation_settings(guild)["support_reward"] (default 0) and set_vault_operation_settings(..., support_reward=None):
               int 0..MAX_CURRENCY_CHANGE (bool is not an int), UserError(VAULT_SETTINGS_REWARD) otherwise, revision +1 like the other fields.
               Stored as a column of vault_operation_settings added by the v27 migration (see tests/test_vault_migration_v27.py).
  Tables (v27) vault_support_missions(id, guild_id, operation_id -> vault_operations, user_id, mission_type "forensics"|"bulls_cows", puzzle_json,
               answer_json, status "active"|"solved"|"failed", attempts_used, created_at, finished_at; UNIQUE(operation_id, user_id)) and
               vault_support_attempts(id, guild_id, mission_id -> vault_support_missions, number, answer_json, feedback_json, correct,
               action_key, created_at). answer_json of a bulls_cows row is the JSON string "123"; of a forensics row [strikes, balls].
               The PUZZLE AND ANSWER ARE PERSISTED AT CREATION and never regenerated. The reward uses the existing vault_operation_claims
               with type "support" (UNIQUE(operation_id, user_id, type) already makes it one per member; the column has no CHECK to widen).
  start_support_mission(guild_id, user_id, operation_id, mission_type, *, now=None, seed=None) -> mission view
        One mission per member per operation (proposal 11.4). Creates it (created=True) or returns the persisted one (created=False,
        ``seed`` ignored). ``seed`` is a test seam: the seed string alone fixes the puzzle (generate_forensics(seed) / Random(seed) for B).
        Refusals in order: vault.not_found (other guild / unknown id), vault.operation_not_participant, vault.not_active (operation solved,
        failed or expired: lazily expired), vault.mission_type_unknown (not one of MISSION_TYPES, including non-str),
        vault.mission_exists (the member already has a mission of ANOTHER type; their one mission is not replaced).
  support_mission(guild_id, user_id, operation_id, *, now=None) -> view | None     the member's own mission, None if never started.
        Same not_found / not_participant refusals; readable after the operation ended.
  submit_support_answer(guild_id, user_id, operation_id, answer, *, action_key=None, now=None) -> result
        {"correct": bool, "feedback": (strikes, balls) | None (bulls_cows only, None for forensics), "number": attempt number,
         "event": None | "solved" | "failed", "replayed": bool, "mission": view}
        bulls_cows: ``answer`` a guess string validated like a main guess (vault.guess_length / guess_not_digits / guess_repeated_digit,
        length 3); an already tried guess -> vault.guess_duplicate. forensics: ``answer`` a (strikes, balls) pair (tuple or list of ints)
        valid per valid_forensics_answer, otherwise vault.mission_answer_invalid; an already tried pair -> vault.guess_duplicate.
        Invalid and duplicate answers are FREE (no attempt used). Refusals order: not_found, operation_not_participant, vault.not_active,
        vault.mission_none (no mission yet), vault.mission_finished (solved or failed), then validation. A correct answer solves
        (event "solved"); using the last attempt without it fails the mission (event "failed", final: no retry, no new puzzle).
        A replayed action_key (per operation) returns the first result with replayed=True, spends and pays nothing, even after the
        mission finished or the operation ended.
  Mission view {"id","operation_id","guild_id","user_id","type","status","created","turn_budget" (BULLS_COWS_TURNS / FORENSICS_ATTEMPTS),
        "attempts_used","attempts_left","puzzle": bulls_cows -> {"length": 3}; forensics -> {"records": [{"guess","result": (s,b)|None}], "hidden": int},
        "history": [{"number","answer","feedback","correct"}] (bulls_cows answer = the guess string; forensics = (s,b)),
        "answer": None while active; once solved or failed the answer ("123" / (s,b)), "reward": R|None with R = {"amount","ledger_id","reversed"}}
  Operation view gains participants[i]["reward_support"] (R | None, like reward_contribution), and a SOLVED mission adds exactly one to
        participants[i]["contributions"] (an attempt, a failed mission or opening one adds nothing).
  Rewards      solved + settings support_reward > 0: one ledger credit (source "vault", positive, actor the member) and one claim row type
        "support", in the transaction of the solving answer, unique per (operation, member), amount in force then. A solved mission also counts as
        one contribution: the existing contribution_reward / min_guesses qualification and the team reward see it. It spends no shared stage
        turn and changes nothing a stage view shows (and never exposes any stage secret).
"""
import json
import sqlite3
import unittest

from llmcord_core.i18n import UserError
from llmcord_core.vault.missions import BULLS_COWS_TURNS, FORENSICS_ATTEMPTS, bulls_cows_feedback, bulls_cows_secrets

from test_vault_operation_store import BOB, CAROL, G, G2, NOW, OpCase
from test_vault_store import ALICE

FORENSICS, BC = "forensics", "bulls_cows"
LATER = NOW + 600


BAD_TYPES = (UserError, TypeError)


class MissionCase(OpCase):
    def setUp(self):
        super().setUp()
        self.store.set_vault_settings(G, turns={"standard": 9, "open": 9})
        self.start()
        self.join(user=BOB)
        self.join(user=CAROL)

    def begin(self, kind=BC, user=ALICE, seed="m1", **kw):
        return self.store.start_support_mission(G, user, self.oid, kind, now=kw.pop("now", NOW), seed=seed, **kw)

    def answer(self, value, user=ALICE, **kw):
        return self.store.submit_support_answer(G, user, self.oid, value, now=kw.pop("now", NOW), **kw)

    def row(self, user=ALICE):
        return self.store.one("SELECT * FROM vault_support_missions WHERE operation_id=? AND user_id=?", (self.oid, user))

    def truth(self, user=ALICE):
        return json.loads(self.row(user)["answer_json"])

    def bc_wrongs(self, n, user=ALICE):
        truth = self.truth(user)
        return [c for c in bulls_cows_secrets() if c != truth][:n]

    def fx_wrongs(self, n, user=ALICE):
        truth = tuple(self.truth(user))
        pairs = [(s, b) for s in range(5) for b in range(5) if s + b <= 4 and (s, b) != (3, 1) and (s, b) != truth]
        return pairs[:n]

    def solve_bc(self, user=ALICE, **kw):
        return self.answer(self.truth(user), user, **kw)

    def vault_ledger(self, user=None):
        rows = [dict(r) for r in self.store.db.execute("SELECT * FROM currency_ledger WHERE guild_id=? AND source='vault' ORDER BY id", (G,))]
        return [r for r in rows if user is None or r["user_id"] == user]

    def participant(self, user):
        return next(p for p in self.op()["participants"] if p["user_id"] == user)


class SettingTests(OpCase):
    def test_support_reward_defaults_to_zero_and_is_stored_with_a_revision(self):
        before = self.store.vault_operation_settings(G)
        self.assertEqual(before["support_reward"], 0)
        after = self.store.set_vault_operation_settings(G, support_reward=15)
        self.assertEqual((after["support_reward"], after["revision"]), (15, before["revision"] + 1))
        self.assertEqual({k: v for k, v in after.items() if k not in ("support_reward", "revision")},
                         {k: v for k, v in before.items() if k not in ("support_reward", "revision")})
        self.assertEqual(self.store.vault_operation_settings(G2)["support_reward"], 0)

    def test_other_settings_changes_keep_the_support_reward(self):
        self.store.set_vault_operation_settings(G, support_reward=15)
        self.assertEqual(self.store.set_vault_operation_settings(G, team_reward=5)["support_reward"], 15)

    def test_bad_support_reward_is_rejected_and_writes_nothing(self):
        before = self.store.vault_operation_settings(G)
        for bad in (-1, True, 1.5, "10", [1], self.store.MAX_CURRENCY_CHANGE + 1):
            with self.subTest(bad=bad), self.assertRaises(UserError):
                self.store.set_vault_operation_settings(G, support_reward=bad)
        self.assertEqual(self.store.vault_operation_settings(G), before)


class StartTests(MissionCase):
    def test_bulls_cows_starts_active_with_a_persisted_secret_and_hidden_answer(self):
        view = self.begin(BC)
        self.assertEqual((view["type"], view["status"], view["created"], view["guild_id"], view["operation_id"], view["user_id"]), (BC, "active", True, G, self.oid, ALICE))
        self.assertEqual((view["turn_budget"], view["attempts_used"], view["attempts_left"], view["history"]), (BULLS_COWS_TURNS, 0, BULLS_COWS_TURNS, []))
        self.assertEqual(view["puzzle"], {"length": 3})
        self.assertIsNone(view["answer"])
        self.assertIsNone(view["reward"])
        row = self.row()
        self.assertEqual((row["mission_type"], row["status"], row["attempts_used"], row["guild_id"]), (BC, "active", 0, G))
        self.assertIn(self.truth(), bulls_cows_secrets())

    def test_forensics_starts_active_with_one_hidden_result_and_hidden_answer(self):
        view = self.begin(FORENSICS)
        self.assertEqual((view["type"], view["status"], view["turn_budget"]), (FORENSICS, "active", FORENSICS_ATTEMPTS))
        records = view["puzzle"]["records"]
        self.assertEqual([r["result"] for r in records].count(None), 1)
        self.assertIsNone(records[view["puzzle"]["hidden"]]["result"])
        self.assertIsNone(view["answer"])
        self.assertEqual(self.row()["mission_type"], FORENSICS)

    def test_the_stored_forensics_answer_is_the_one_the_solver_proves(self):
        from llmcord_core.vault.missions import forensics_answers
        view = self.begin(FORENSICS)
        records = tuple((r["guess"], r["result"]) for r in view["puzzle"]["records"])
        self.assertEqual(forensics_answers(records), frozenset({tuple(self.truth())}))

    def test_the_answer_never_appears_in_the_active_view_or_a_refusal(self):
        view = self.begin(BC)
        truth = self.truth()
        self.assertNotIn(f'"{truth}"', json.dumps(view, default=str))
        err = self.refused("vault.guess_length", self.answer, truth[:2])
        self.assertNotIn(truth, repr(err) + repr(getattr(err, "params", "")))
        read = self.store.support_mission(G, ALICE, self.oid, now=NOW)
        self.assertNotIn(f'"{truth}"', json.dumps(read, default=str))
        self.assertIsNone(read["answer"])

    def test_seed_fixes_the_puzzle(self):
        a = self.begin(BC, ALICE, seed="same")
        b = self.begin(BC, BOB, seed="same")
        self.assertEqual(self.truth(ALICE), self.truth(BOB))
        self.assertEqual(a["puzzle"], b["puzzle"])
        c = self.begin(FORENSICS, CAROL, seed="same")
        d = self.store.start_support_mission(G, ALICE, self.oid, BC, now=NOW)   # reopen: seed absent, same persisted mission
        self.assertEqual(self.truth(ALICE), self.truth(BOB))
        self.assertFalse(d["created"])
        self.assertEqual(c["type"], FORENSICS)

    def test_reopening_returns_the_persisted_puzzle_and_never_regenerates(self):
        first = self.begin(BC, seed="one")
        before = dict(self.row())
        for seed in ("two", None):
            again = self.begin(BC, seed=seed)
            self.assertFalse(again["created"])
            self.assertEqual(again["puzzle"], first["puzzle"])
            self.assertEqual(self.row()["answer_json"], before["answer_json"])
        self.assertEqual(dict(self.row()), before)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_support_missions")[0], 1)

    def test_reopening_forensics_returns_the_same_records(self):
        first = self.begin(FORENSICS, seed="f1")
        again = self.begin(FORENSICS, seed="f2")
        self.assertEqual((again["puzzle"], again["created"]), (first["puzzle"], False))

    def test_reopening_keeps_the_puzzle_after_attempts_and_after_failure(self):
        first = self.begin(BC)
        truth = self.truth()
        for code in self.bc_wrongs(BULLS_COWS_TURNS):
            self.answer(code)
        again = self.begin(BC, seed="new-seed")
        self.assertEqual((again["status"], again["created"], again["puzzle"], self.truth()), ("failed", False, first["puzzle"], truth))

    def test_one_mission_per_member_per_operation_a_second_type_is_refused(self):
        self.begin(BC)
        self.refused("vault.mission_exists", self.begin, FORENSICS)
        self.assertEqual(self.row()["mission_type"], BC)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_support_missions")[0], 1)
        self.refused("vault.mission_exists", self.begin, FORENSICS, ALICE, "x")

    def test_the_schema_allows_one_mission_row_per_member_and_operation(self):
        self.begin(BC)
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.db.execute("INSERT INTO vault_support_missions(guild_id,operation_id,user_id,mission_type,puzzle_json,answer_json,status,attempts_used,created_at) "
                                  "SELECT guild_id,operation_id,user_id,mission_type,puzzle_json,answer_json,status,attempts_used,created_at FROM vault_support_missions")
        self.store.db.rollback()
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_support_missions")[0], 1)

    def test_members_get_their_own_missions(self):
        self.begin(BC, ALICE, "a")
        self.begin(FORENSICS, BOB, "b")
        self.assertEqual((self.row(ALICE)["mission_type"], self.row(BOB)["mission_type"]), (BC, FORENSICS))
        self.assertIsNone(self.store.support_mission(G, CAROL, self.oid, now=NOW))

    def test_unknown_type_is_refused_and_writes_nothing(self):
        for bad in ("equipment", "C", "", None, 1, ["bulls_cows"], True):
            with self.subTest(bad=bad):
                self.refused("vault.mission_type_unknown", self.begin, bad)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_support_missions")[0], 0)

    def test_only_participants_of_an_active_operation(self):
        self.refused("vault.operation_not_participant", self.store.start_support_mission, G, 999, self.oid, BC, now=NOW)
        self.refused("vault.not_found", self.store.start_support_mission, G, ALICE, self.oid + 99, BC, now=NOW)
        self.refused("vault.not_found", self.store.start_support_mission, G2, ALICE, self.oid, BC, now=NOW)
        self.refused("vault.operation_not_participant", self.store.support_mission, G, 999, self.oid, now=NOW)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_support_missions")[0], 0)

    def test_a_solved_failed_or_expired_operation_starts_no_mission(self):
        self.store.db.execute("UPDATE vault_operations SET status='solved' WHERE id=?", (self.oid,))
        self.store.db.commit()
        self.refused("vault.not_active", self.begin)
        self.store.db.execute("UPDATE vault_operations SET status='failed' WHERE id=?", (self.oid,))
        self.store.db.commit()
        self.refused("vault.not_active", self.begin)
        self.store.db.execute("UPDATE vault_operations SET status='active' WHERE id=?", (self.oid,))
        self.store.db.commit()
        expires = self.store.one("SELECT expires_at FROM vault_operations WHERE id=?", (self.oid,))["expires_at"]
        self.refused("vault.not_active", self.begin, now=expires + 1)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_support_missions")[0], 0)


class BullsCowsAnswerTests(MissionCase):
    def setUp(self):
        super().setUp()
        self.begin(BC)

    def test_a_wrong_guess_returns_feedback_uses_an_attempt_and_stays_hidden(self):
        guess = self.bc_wrongs(1)[0]
        result = self.answer(guess)
        self.assertEqual((result["correct"], result["event"], result["replayed"], result["number"]), (False, None, False, 1))
        self.assertEqual(result["feedback"], bulls_cows_feedback(self.truth(), guess))
        view = result["mission"]
        self.assertEqual((view["attempts_used"], view["attempts_left"], view["status"], view["answer"]), (1, BULLS_COWS_TURNS - 1, "active", None))
        self.assertEqual(view["history"], [{"number": 1, "answer": guess, "feedback": result["feedback"], "correct": False}])
        self.assertEqual(self.row()["attempts_used"], 1)

    def test_the_correct_guess_solves_and_reveals_the_answer(self):
        truth = self.truth()
        result = self.solve_bc()
        self.assertEqual((result["correct"], result["event"], result["feedback"]), (True, "solved", (3, 0)))
        view = result["mission"]
        self.assertEqual((view["status"], view["answer"], view["attempts_used"]), ("solved", truth, 1))
        self.assertEqual(self.row()["status"], "solved")
        self.assertEqual(self.store.support_mission(G, ALICE, self.oid, now=NOW)["answer"], truth)

    def test_the_last_attempt_may_still_win(self):
        for code in self.bc_wrongs(BULLS_COWS_TURNS - 1):
            self.answer(code)
        result = self.solve_bc()
        self.assertEqual((result["event"], result["mission"]["status"], result["mission"]["attempts_used"]), ("solved", "solved", BULLS_COWS_TURNS))

    def test_exhausting_the_attempts_fails_the_mission_and_reveals_the_answer(self):
        truth = self.truth()
        events = [self.answer(code)["event"] for code in self.bc_wrongs(BULLS_COWS_TURNS)]
        self.assertEqual(events, [None] * (BULLS_COWS_TURNS - 1) + ["failed"])
        view = self.store.support_mission(G, ALICE, self.oid, now=NOW)
        self.assertEqual((view["status"], view["answer"], view["attempts_left"]), ("failed", truth, 0))

    def test_a_failed_mission_stays_failed(self):
        truth = self.truth()
        for code in self.bc_wrongs(BULLS_COWS_TURNS):
            self.answer(code)
        self.refused("vault.mission_finished", self.answer, truth)
        self.assertEqual(self.row()["status"], "failed")
        self.assertEqual(self.begin(BC, seed="retry")["status"], "failed")
        self.refused("vault.mission_exists", self.begin, FORENSICS)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_support_missions")[0], 1)
        self.assertEqual(self.vault_ledger(), [])

    def test_a_solved_mission_takes_no_more_answers(self):
        self.solve_bc()
        self.refused("vault.mission_finished", self.answer, self.bc_wrongs(1)[0])
        self.assertEqual(self.row()["attempts_used"], 1)

    def test_invalid_input_is_rejected_free(self):
        for bad, code in (("12", "vault.guess_length"), ("1234", "vault.guess_length"), ("", "vault.guess_length"), ("12a", "vault.guess_not_digits"),
                          ("1 2", "vault.guess_not_digits"), ("112", "vault.guess_repeated_digit")):
            with self.subTest(bad=bad):
                self.refused(code, self.answer, bad)
        for bad in (None, 123, ["1", "2", "3"], (1, 2, 3)):
            with self.subTest(bad=bad), self.assertRaises(BAD_TYPES):
                self.answer(bad)
        view = self.store.support_mission(G, ALICE, self.oid, now=NOW)
        self.assertEqual((view["attempts_used"], view["history"]), (0, []))
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_support_attempts")[0], 0)

    def test_a_repeated_guess_is_rejected_free(self):
        guess = self.bc_wrongs(1)[0]
        self.answer(guess)
        self.refused("vault.guess_duplicate", self.answer, guess)
        self.assertEqual(self.store.support_mission(G, ALICE, self.oid, now=NOW)["attempts_used"], 1)

    def test_answering_before_starting_is_refused(self):
        self.refused("vault.mission_none", self.answer, "012", BOB)
        self.refused("vault.operation_not_participant", self.answer, "012", 999)
        self.refused("vault.not_found", self.store.submit_support_answer, G2, ALICE, self.oid, "012", now=NOW)

    def test_nobody_can_answer_for_another_member(self):
        """Each member answers only their own mission: Bob has none, so Alice's secret does nothing for him."""
        self.refused("vault.mission_none", self.answer, self.truth(), BOB)
        self.assertEqual(self.row()["status"], "active")

    def test_answers_stop_when_the_operation_ends(self):
        self.store.db.execute("UPDATE vault_operations SET status='solved' WHERE id=?", (self.oid,))
        self.store.db.commit()
        self.refused("vault.not_active", self.answer, self.bc_wrongs(1)[0])
        self.assertEqual(self.row()["attempts_used"], 0)
        expires = self.store.one("SELECT expires_at FROM vault_operations WHERE id=?", (self.oid,))["expires_at"]
        self.store.db.execute("UPDATE vault_operations SET status='active' WHERE id=?", (self.oid,))
        self.store.db.commit()
        self.refused("vault.not_active", self.answer, self.bc_wrongs(1)[0], now=expires + 1)

    def test_action_key_replays_without_a_second_attempt(self):
        guess = self.bc_wrongs(1)[0]
        first = self.answer(guess, action_key="k1")
        again = self.answer(guess, action_key="k1")
        self.assertEqual((again["replayed"], first["replayed"], again["feedback"], again["number"]), (True, False, first["feedback"], 1))
        self.assertEqual(self.row()["attempts_used"], 1)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_support_attempts")[0], 1)

    def test_a_replayed_key_works_after_the_mission_finished_and_other_keys_are_refused(self):
        solved = self.solve_bc(action_key="win")
        again = self.answer("ignored", action_key="win")
        self.assertEqual((again["replayed"], again["correct"], again["event"] == solved["event"]), (True, True, True))
        self.refused("vault.mission_finished", self.answer, "012", action_key="other")


class ForensicsAnswerTests(MissionCase):
    def setUp(self):
        super().setUp()
        self.begin(FORENSICS, seed="f-seed")

    def test_the_correct_pair_solves_and_reveals_the_answer(self):
        truth = tuple(self.truth())
        result = self.answer(truth)
        self.assertEqual((result["correct"], result["event"], result["feedback"]), (True, "solved", None))
        self.assertEqual((result["mission"]["status"], tuple(result["mission"]["answer"])), ("solved", truth))

    def test_a_list_pair_is_accepted_too(self):
        self.assertTrue(self.answer(list(self.truth()))["correct"])

    def test_wrong_pairs_use_the_attempts_then_fail_and_reveal(self):
        truth = tuple(self.truth())
        wrong = self.fx_wrongs(FORENSICS_ATTEMPTS)
        events = [self.answer(w)["event"] for w in wrong]
        self.assertEqual(events, [None] * (FORENSICS_ATTEMPTS - 1) + ["failed"])
        view = self.store.support_mission(G, ALICE, self.oid, now=NOW)
        self.assertEqual((view["status"], tuple(view["answer"])), ("failed", truth))
        self.refused("vault.mission_finished", self.answer, truth)

    def test_a_wrong_pair_is_not_a_win_and_shows_no_feedback(self):
        result = self.answer(self.fx_wrongs(1)[0])
        self.assertEqual((result["correct"], result["feedback"]), (False, None))
        if FORENSICS_ATTEMPTS > 1:
            self.assertIsNone(result["mission"]["answer"])

    def test_invalid_pairs_are_rejected_free(self):
        for bad in ((3, 1), (5, 0), (2, 3), (-1, 0), (True, 0), (1.0, 0), ("1", "0"), (1,), (1, 2, 3), None, "10", 10, {"s": 1, "b": 0}):
            with self.subTest(bad=bad):
                self.refused("vault.mission_answer_invalid", self.answer, bad)
        self.assertEqual(self.store.support_mission(G, ALICE, self.oid, now=NOW)["attempts_used"], 0)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_support_attempts")[0], 0)

    def test_a_repeated_wrong_pair_is_rejected_free(self):
        pair = self.fx_wrongs(1)[0]
        self.answer(pair)
        self.refused("vault.guess_duplicate", self.answer, list(pair))
        self.assertEqual(self.store.support_mission(G, ALICE, self.oid, now=NOW)["attempts_used"], 1)

    def test_failed_forensics_stays_failed_and_pays_nothing(self):
        self.store.set_vault_operation_settings(G, support_reward=15)
        for pair in self.fx_wrongs(FORENSICS_ATTEMPTS):
            self.answer(pair)
        self.assertEqual(self.begin(FORENSICS, seed="again")["status"], "failed")
        self.assertEqual((self.vault_ledger(), self.store.balance(G, ALICE)), ([], 0))


class NoSharedTurnsTests(MissionCase):
    def test_a_whole_mission_changes_nothing_the_stage_shows(self):
        before = self.op()["stage"]
        self.begin(BC)
        for code in self.bc_wrongs(BULLS_COWS_TURNS):
            self.answer(code)
        self.begin(FORENSICS, BOB, "x")
        self.answer(self.truth(BOB), BOB)
        after = self.op()["stage"]
        self.assertEqual(after, before)
        self.assertEqual((after["turns_used"], after["secret"], after["guesses"]), (0, None, []))
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_operation_actions")[0], 0)

    def test_missions_do_not_disturb_main_guesses_or_the_stage_secret(self):
        secret = self.secret()
        self.begin(BC)
        self.solve_bc()
        self.assertEqual(self.secret(), secret)
        self.g(self.wrongs()[0], BOB)
        self.assertEqual(self.op()["stage"]["turns_used"], 1)

    def test_no_view_or_result_carries_the_stage_secret(self):
        secret = self.secret()
        view = self.begin(FORENSICS)
        text = json.dumps([view, self.answer(self.fx_wrongs(1)[0])], default=str)
        self.assertNotIn(f'"{secret}"', text)
        self.assertIsNone(self.op()["stage"]["secret"])


class RewardTests(MissionCase):
    def setUp(self):
        super().setUp()
        self.store.set_vault_operation_settings(G, support_reward=15)
        self.begin(BC)

    def test_solving_pays_the_support_reward_once_through_the_ledger(self):
        result = self.solve_bc()
        reward = result["mission"]["reward"]
        self.assertEqual((reward["amount"], reward["reversed"]), (15, False))
        self.assertEqual(self.store.balance(G, ALICE), 15)
        entry = self.vault_ledger(ALICE)
        self.assertEqual([(e["amount"], e["source"], e["id"]) for e in entry], [(15, "vault", reward["ledger_id"])])
        claim = self.store.one("SELECT * FROM vault_operation_claims WHERE operation_id=? AND user_id=?", (self.oid, ALICE))
        self.assertEqual((claim["type"], claim["amount"], claim["ledger_id"], claim["guild_id"]), ("support", 15, reward["ledger_id"], G))
        mine = self.participant(ALICE)
        self.assertEqual(mine["reward_support"]["amount"], 15)
        self.assertEqual(self.participant(BOB)["reward_support"], None)

    def test_nothing_is_paid_before_solving_or_for_attempts_or_a_failed_mission(self):
        self.begin(BC, BOB, "b")
        for code in self.bc_wrongs(BULLS_COWS_TURNS - 1):
            self.answer(code)
        self.assertEqual(self.vault_ledger(), [])
        self.answer(self.bc_wrongs(BULLS_COWS_TURNS)[-1])
        self.assertEqual((self.vault_ledger(), self.store.one("SELECT COUNT(*) FROM vault_operation_claims")[0]), ([], 0))

    def test_never_paid_twice_by_replay_reopening_or_more_answers(self):
        self.solve_bc(action_key="win")
        self.solve_bc(action_key="win")
        self.begin(BC, seed="again")
        self.refused("vault.mission_finished", self.solve_bc)
        self.assertEqual(self.store.balance(G, ALICE), 15)
        self.assertEqual(len(self.vault_ledger()), 1)

    def test_the_claim_is_unique_per_operation_member_and_type_in_the_database(self):
        self.solve_bc()
        row = self.store.one("SELECT * FROM vault_operation_claims WHERE user_id=?", (ALICE,))
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.db.execute("INSERT INTO vault_operation_claims(guild_id,operation_id,user_id,type,amount,ledger_id,created_at) VALUES(?,?,?,?,?,?,?)",
                                  (G, self.oid, ALICE, "support", 15, row["ledger_id"], NOW))
        self.store.db.rollback()

    def test_one_reward_per_member_each_member_paid_for_their_own_mission(self):
        self.begin(FORENSICS, BOB, "b")
        self.solve_bc()
        self.assertEqual(self.store.balance(G, BOB), 0)
        self.answer(self.truth(BOB), BOB)
        self.assertEqual((self.store.balance(G, ALICE), self.store.balance(G, BOB), self.store.balance(G, CAROL)), (15, 15, 0))

    def test_zero_reward_pays_nothing_and_writes_no_claim(self):
        self.store.set_vault_operation_settings(G, support_reward=0)
        result = self.solve_bc()
        self.assertEqual((result["mission"]["status"], result["mission"]["reward"]), ("solved", None))
        self.assertEqual((self.vault_ledger(), self.store.one("SELECT COUNT(*) FROM vault_operation_claims")[0]), ([], 0))

    def test_the_reward_in_force_when_solved_is_the_one_paid(self):
        self.store.set_vault_operation_settings(G, support_reward=40)
        self.solve_bc()
        self.assertEqual(self.store.balance(G, ALICE), 40)

    def test_reversal_is_flagged_takes_the_money_back_and_never_repays(self):
        reward = self.solve_bc()["mission"]["reward"]
        self.store.reverse_entry(G, reward["ledger_id"], 1, "undo")
        self.assertEqual(self.store.balance(G, ALICE), 0)
        mine = self.participant(ALICE)["reward_support"]
        self.assertEqual((mine["amount"], mine["reversed"]), (15, True))
        self.begin(BC, seed="again")
        self.assertEqual(self.store.balance(G, ALICE), 0)
        self.assertEqual(self.store.support_mission(G, ALICE, self.oid, now=NOW)["reward"]["reversed"], True)

    def test_a_refused_answer_pays_nothing(self):
        self.refused("vault.guess_length", self.answer, "1")
        self.assertEqual(self.vault_ledger(), [])

    def test_a_replayed_key_still_reports_the_reward_without_paying_again(self):
        first = self.solve_bc(action_key="w")
        again = self.answer("000", action_key="w")
        self.assertEqual(again["mission"]["reward"], first["mission"]["reward"])
        self.assertEqual(self.store.balance(G, ALICE), 15)


class ContributionTests(MissionCase):
    def setUp(self):
        super().setUp()
        self.store.set_vault_operation_settings(G, contribution_reward=20, team_reward=50, support_reward=15)
        self.begin(BC)

    def test_a_solved_mission_is_exactly_one_contribution(self):
        self.assertEqual(self.participant(ALICE)["contributions"], 0)
        for code in self.bc_wrongs(2):
            self.answer(code)
        self.assertEqual(self.participant(ALICE)["contributions"], 0)
        self.solve_bc()
        self.assertEqual(self.participant(ALICE)["contributions"], 1)
        self.assertEqual(self.participant(BOB)["contributions"], 0)

    def test_opening_a_failed_mission_or_reopening_is_not_a_contribution(self):
        for code in self.bc_wrongs(BULLS_COWS_TURNS):
            self.answer(code)
        self.begin(BC)
        self.assertEqual(self.participant(ALICE)["contributions"], 0)
        self.assertEqual(self.vault_ledger(), [])

    def test_a_solved_mission_pays_only_the_support_reward_never_the_contribution_reward(self):
        """FEAT-67 (proposal 12.2): the mission pays support_reward only; the main contribution reward still needs a real guess, tool or cross."""
        self.solve_bc()
        self.assertEqual(self.store.balance(G, ALICE), 15)
        self.assertEqual([e["amount"] for e in self.vault_ledger(ALICE)], [15])
        self.assertIsNone(self.participant(ALICE)["reward_contribution"])
        self.g(self.wrongs()[0], ALICE)
        self.assertEqual(self.store.balance(G, ALICE), 35)

    def test_with_min_guesses_two_the_mission_does_not_count_toward_the_contribution_reward(self):
        """FEAT-67: one real guess is still one short of min_guesses=2 for the contribution reward, whatever missions were solved."""
        self.store.set_vault_operation_settings(G, min_guesses=2)
        self.solve_bc()
        self.assertEqual(self.store.balance(G, ALICE), 15)
        self.assertIsNone(self.participant(ALICE)["reward_contribution"])
        self.g(self.wrongs()[0], ALICE)
        self.assertEqual(self.store.balance(G, ALICE), 15)

    def test_a_main_contribution_first_then_the_mission_pays_support_once_more_only(self):
        self.g(self.wrongs()[0], ALICE)
        self.assertEqual(self.store.balance(G, ALICE), 20)
        self.solve_bc()
        self.assertEqual(self.store.balance(G, ALICE), 35)
        self.assertEqual(self.participant(ALICE)["contributions"], 2)

    def test_a_member_with_only_a_mission_shares_the_team_reward(self):
        self.solve_bc(user=ALICE)
        self.g(self.wrongs()[0], BOB)
        for _ in range(3):
            self.solve(BOB)
        self.assertEqual(self.op()["status"], "solved")
        self.assertEqual((self.store.balance(G, ALICE), self.store.balance(G, BOB), self.store.balance(G, CAROL)), (15 + 50, 70, 0))

    def test_a_failed_mission_does_not_qualify_for_the_team_reward(self):
        for code in self.bc_wrongs(BULLS_COWS_TURNS):
            self.answer(code)
        self.g(self.wrongs()[0], BOB)
        for _ in range(3):
            self.solve(BOB)
        self.assertEqual(self.store.balance(G, ALICE), 0)

    def test_a_member_may_play_a_mission_after_the_main_stages_are_all_solved_only_while_active(self):
        """Characterization (FEAT-67): once the operation is solved it is no longer active, so a mission can no longer be started or answered
        (proposal 11.5 would keep it open until expiry; the brief says off unless the operation is active)."""
        self.solve_all(BOB)
        self.assertEqual(self.op()["status"], "solved")
        self.refused("vault.not_active", self.begin, BC, CAROL)
        self.refused("vault.not_active", self.answer, self.bc_wrongs(1)[0])
        self.assertEqual(self.store.support_mission(G, ALICE, self.oid, now=NOW)["status"], "active")


class IsolationTests(MissionCase):
    def test_another_guild_cannot_start_read_or_answer(self):
        self.begin(BC)
        self.refused("vault.not_found", self.store.start_support_mission, G2, ALICE, self.oid, BC, now=NOW)
        self.refused("vault.not_found", self.store.support_mission, G2, ALICE, self.oid, now=NOW)
        self.refused("vault.not_found", self.store.submit_support_answer, G2, ALICE, self.oid, "012", now=NOW)
        self.assertEqual(self.row()["attempts_used"], 0)

    def test_rows_and_rewards_are_scoped_to_their_guild(self):
        self.store.set_vault_operation_settings(G, support_reward=15)
        self.begin(BC)
        self.solve_bc()
        self.assertEqual(self.store.one("SELECT guild_id FROM vault_support_missions")["guild_id"], G)
        self.assertEqual(self.store.one("SELECT guild_id FROM vault_support_attempts")["guild_id"], G)
        self.assertEqual(self.ledger(G2), [])
        self.assertEqual(self.store.balance(G2, ALICE), 0)
        self.assertEqual(self.store.vault_operation_settings(G2)["support_reward"], 0)

    def test_the_tables_carry_guild_and_operation_ids(self):
        for table, needs in (("vault_support_missions", {"guild_id", "operation_id", "user_id", "mission_type", "puzzle_json", "answer_json", "status", "attempts_used"}),
                             ("vault_support_attempts", {"guild_id", "mission_id", "number", "answer_json", "action_key", "created_at"})):
            columns = {r[1] for r in self.store.all(f"PRAGMA table_info({table})")}
            self.assertTrue(needs <= columns, table)


if __name__ == "__main__":
    unittest.main()
