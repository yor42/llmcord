"""Vault tools in the store (FEAT-64 phase A, decision D43). Written before the implementation; they fail until it exists.

API CONTRACT (new method on ``VaultStore`` in ``llmcord_core/vault_store.py``; the items side is in tests/test_vault_items.py)

  use_vault_tool(guild_id, user_id, session_id, tool_key, params, *, action_key=None, now=None) -> result
      tool_key is one of ``vault.TOOLS``; params is the engine's plain dict (1-based int positions, one-character string digits).
      One ``write_admin()`` transaction does everything or nothing: checks, the engine query, ONE inventory unit consumed through
      ``consume_item`` (the items_store primitive, called inside this transaction), the tool action row, and the budget/state
      transition. Any refusal below raises ``VaultStoreError`` before anything is written: no item consumed, no turn spent,
      no use-log row, and it does not count towards the two-per-stage cap.
      result = {"tool": key, "params": params, "text_key": engine ToolResult.text_key, "data": engine ToolResult.data,
                "turn_cost": vault.TOOLS[key].turn_cost, "constraint": engine ToolResult.constraint, "qty": units left in the
                member's inventory, "replayed": bool, "session": view (after the action)}
      Replay: the same action_key in the same session returns the first result with replayed=True, consumes and spends nothing, even
      when the inventory is now empty or the vault has ended (same rule as submit_vault_guess).

  Refusal codes (``err.code.code``); the check ORDER is the order of this list (earlier wins when several apply):
      vault.not_found      session is not the member's own in this guild
      vault.not_active     solved, failed or expired
      vault.tool_unknown   tool_key not in vault.TOOLS (params["tool"] carries the key)
      vault.tool_limit     two tools already used in this stage (params["limit"] == 2)
      vault.tool_no_turns  not vault.tool_fits(key, turns_left): the Master Decoder needs turns_left >= 3 (it costs 2)
      vault.tool_not_held  the member's own inventory in this guild holds none of the item
      vault.tool_bad_params  the engine refused the params on the real secret (bad_position, bad_digit, digits_equal,
                           guess_not_in_history, digit_not_in_guess, already_revealed: all depend on params/history only);
                           params["reason"] is the engine VaultError code. Judged BEFORE redundancy, so a disclosed position is bad_params.
                           Free refusals never depend on the secret: the verifier's nothing_new (every strike disclosed) is answered
                           as a normal paid result when the public candidates still differ.
      vault.tool_redundant every remaining candidate (vault.filter_candidates on the PUBLIC constraints) gives the same answer
                           (vault.is_redundant), so the query would teach nothing

  view additions: "tools": [{"tool", "params", "text_key", "data", "turn_cost"}] oldest first; "tools_used": len(tools).
      "constraints" gains each tool's constraint. "guesses" stays guesses only. turns_used = guesses + sum of tool turn costs;
      turns_left = budget - turns_used. The same turns_used is what vault_history()["turns_used"] reports.
  Budget: a tool that leaves turns_left == 0 without a win marks the vault "failed" (secret revealed in that view).
  Tool constraints are public: they are never hidden, and the secret itself is not exposed by an active view or result.
"""
import itertools
import sqlite3
import unittest

from llmcord_core.vault import TOOLS, GuessResult, all_secrets, evaluate_tool, feedback, filter_candidates
from llmcord_core.vault_store import VaultStoreError

from test_vault_store import ALICE, BOB, CH, DAY, G, G2, NOW, VaultCase

SPECTRUM, DECODER, SCANNER, COMPARE, VERIFIER = ("vault_spectrum_analyzer", "vault_master_decoder", "vault_residual_scanner",
                                                 "vault_comparison_circuit", "vault_position_verifier")


class ToolCase(VaultCase):
    def setUp(self):
        super().setUp()
        self.view = self.open()
        self.secret = self.secret_of(self.view)

    def give(self, key, qty=1, user=ALICE, guild=G):
        self.store.db.execute("INSERT INTO inventory(guild_id,user_id,item_key,qty) VALUES(?,?,?,?) ON CONFLICT(guild_id,user_id,item_key) DO UPDATE SET qty=excluded.qty",
                              (guild, user, key, qty))
        self.store.db.commit()

    def use(self, key, params, *, user=ALICE, guild=G, view=None, now=NOW, **kw):
        return self.store.use_vault_tool(guild, user, (view or self.view)["id"], key, params, now=now, **kw)

    def session(self, user=ALICE, guild=G, view=None):
        return self.store.vault_session(guild, user, (view or self.view)["id"], now=NOW)

    def qty(self, key, user=ALICE, guild=G):
        return self.store.item_qty(guild, user, key)

    def state(self, key):
        """What a rejected call must leave untouched."""
        s = self.session()
        return (self.qty(key), s["turns_used"], s["tools_used"], len(s["guesses"]), s["status"], len(self.store.item_use_log(G)))

    def refused(self, code, key, params, **kw):
        before = self.state(key)
        with self.assertRaises(VaultStoreError) as ctx:
            self.use(key, params, **kw)
        self.assertEqual(ctx.exception.code.code, code)
        self.assertEqual(self.state(key), before, "a refusal must be free")
        return ctx.exception

    def swapped_guess(self):
        """A valid guess sharing positions 1 and 2 with the secret but not equal to it."""
        s = self.secret
        return s[0] + s[1] + s[3] + s[2]

    def absent_digit(self):
        return next(d for d in "0123456789" if d not in self.secret)


class UseToolTests(ToolCase):
    def test_spectrum_analyzer_consumes_one_unit_spends_one_turn_and_logs_the_use(self):
        """Regression (FEAT-64): the happy path in one transaction: item, turn, action, constraint and use log."""
        self.give(SPECTRUM, 2)
        result = self.use(SPECTRUM, {"position": 1})
        high = self.secret[0] >= "5"
        self.assertEqual((result["tool"], result["params"], result["turn_cost"], result["replayed"]), (SPECTRUM, {"position": 1}, 1, False))
        self.assertEqual(result["text_key"], "vault_spectrum_high" if high else "vault_spectrum_low")
        self.assertEqual(result["data"], {"position": 1, "high": high})
        self.assertEqual(result["qty"], 1)
        self.assertEqual(self.qty(SPECTRUM), 1)
        view = result["session"]
        self.assertEqual((view["turns_used"], view["turns_left"], view["tools_used"], view["status"]), (1, self.view["turn_budget"] - 1, 1, "active"))
        self.assertEqual(view["guesses"], [])
        self.assertEqual(view["tools"], [{"tool": SPECTRUM, "params": {"position": 1}, "text_key": result["text_key"], "data": result["data"], "turn_cost": 1}])
        log = self.store.item_use_log(G)
        self.assertEqual(len(log), 1)
        self.assertEqual((log[0]["item_key"], log[0]["user_id"], log[0]["channel_id"], log[0]["used_at"]), (SPECTRUM, ALICE, CH, NOW))
        self.assertEqual(log[0]["result"]["tool"], SPECTRUM)
        self.assertEqual(self.session(), view)

    def test_each_tool_returns_the_engine_fact_and_adds_its_constraint(self):
        """Regression (FEAT-64): all five tools, each result equals the engine's and joins the public constraints."""
        for n, key in enumerate(TOOLS):
            user = 100 + n
            view = self.open(user=user, seed=f"tool-{n}")
            secret = self.secret_of(view)
            swapped = secret[0] + secret[1] + secret[3] + secret[2]
            absent = next(d for d in "0123456789" if d not in secret)
            probe = secret[0] + "".join(d for d in "0123456789" if d not in secret)[:2] + secret[3]  # two digits the secret lacks
            params = {"vault_residual_scanner": {"guess": probe, "digit": secret[0]}, "vault_spectrum_analyzer": {"position": 3},
                      "vault_comparison_circuit": {"a": secret[0], "b": absent}, "vault_position_verifier": {"guess": swapped},
                      "vault_master_decoder": {"position": 3}}[key]
            asked = {"vault_residual_scanner": probe, "vault_position_verifier": swapped}.get(key)
            if asked:
                self.store.submit_vault_guess(G, user, view["id"], asked, now=NOW)
            self.give(key, user=user)
            history = [GuessResult(asked, "standard", feedback("standard", secret, asked))] if asked else []
            expected = evaluate_tool(secret, key, params, history)
            result = self.store.use_vault_tool(G, user, view["id"], key, params, now=NOW)
            with self.subTest(key=key):
                self.assertEqual((result["text_key"], result["data"], result["turn_cost"], result["qty"]), (expected.text_key, expected.data, TOOLS[key].turn_cost, 0))
                self.assertEqual(result["constraint"], expected.constraint)
                self.assertIn(expected.constraint, result["session"]["constraints"])
                self.assertEqual(result["session"]["turns_used"], (1 if asked else 0) + TOOLS[key].turn_cost)
                self.assertEqual(result["session"]["tools_used"], 1)

    def test_two_tools_per_stage(self):
        """Regression (FEAT-64): the third activation is refused and keeps its item; refused calls do not count towards the two."""
        self.give(SPECTRUM, 3)
        self.refused("vault.tool_bad_params", SPECTRUM, {"position": 9})  # does not count
        self.use(SPECTRUM, {"position": 1})
        self.use(SPECTRUM, {"position": 2})
        err = self.refused("vault.tool_limit", SPECTRUM, {"position": 3})
        self.assertEqual(err.params["limit"], 2)
        self.assertEqual(self.qty(SPECTRUM), 1)

    def test_scanner_and_comparison_text_keys(self):
        s = self.secret
        probe = s[0] + "".join(d for d in "0123456789" if d not in s)[:2] + s[3]
        self.guess(self.view, probe)
        self.give(SCANNER)
        scan = self.use(SCANNER, {"guess": probe, "digit": s[0]})
        self.assertEqual((scan["text_key"], scan["data"]), ("vault_scanner_present", {"guess": probe, "digit": s[0], "present": True}))
        self.give(COMPARE)
        absent = self.absent_digit()
        cmp = self.use(COMPARE, {"a": s[0], "b": absent})
        self.assertEqual((cmp["text_key"], cmp["data"]), ("vault_comparison_1", {"a": s[0], "b": absent, "count": 1}))
        self.assertEqual(cmp["session"]["turns_used"], 3)

    def test_verifier_and_decoder_disclose_a_digit_and_the_decoder_costs_two(self):
        swapped = self.swapped_guess()
        self.guess(self.view, swapped)
        self.give(VERIFIER)
        found = self.use(VERIFIER, {"guess": swapped})
        self.assertEqual((found["text_key"], found["data"]["position"], found["data"]["digit"]), ("vault_verifier_position", 1, self.secret[0]))
        self.give(DECODER)
        decoded = self.use(DECODER, {"position": 3})
        self.assertEqual((decoded["text_key"], decoded["data"], decoded["turn_cost"]), ("vault_decoder_digit", {"position": 3, "digit": self.secret[2]}, 2))
        self.assertEqual(decoded["session"]["turns_used"], 1 + 1 + 2)
        self.assertEqual(decoded["session"]["turns_left"], self.view["turn_budget"] - 4)

    def test_tool_constraints_narrow_the_public_candidates_without_the_secret(self):
        """Regression (FEAT-64): the constraint is public, keeps the secret possible, and the active view never carries it."""
        self.give(DECODER)
        result = self.use(DECODER, {"position": 2})
        cands = filter_candidates(result["session"]["constraints"])
        self.assertIn(self.secret, cands)
        self.assertTrue(all(c[1] == self.secret[1] for c in cands))
        self.assertIsNone(result["session"]["secret"])
        self.assertNotIn(self.secret, repr(result))

    def test_tool_turns_count_against_guesses_too(self):
        """Regression (FEAT-64): one cost-1 tool leaves budget-1 guesses before the vault fails."""
        self.give(SPECTRUM)
        self.use(SPECTRUM, {"position": 1})
        wrong = self.wrong(self.view, self.view["turn_budget"])
        for code in wrong[:-2]:
            self.assertEqual(self.guess(self.view, code)["session"]["status"], "active")
        last = self.guess(self.view, wrong[-2])
        self.assertEqual((last["session"]["status"], last["session"]["turns_left"]), ("failed", 0))
        self.assertIsNotNone(last["session"]["secret"])
        self.assertEqual(self.store.vault_history(G, ALICE, now=NOW)[0]["turns_used"], self.view["turn_budget"])

    def test_history_turns_used_includes_tool_costs(self):
        self.give(DECODER)
        self.use(DECODER, {"position": 1})
        self.assertEqual(self.store.vault_history(G, ALICE, now=NOW)[0]["turns_used"], 2)


class RejectionsAreFreeTests(ToolCase):
    def test_unknown_tool(self):
        self.give(SPECTRUM)
        before = self.state(SPECTRUM)
        with self.assertRaises(VaultStoreError) as ctx:
            self.use("vault_nope", {})
        self.assertEqual(ctx.exception.code.code, "vault.tool_unknown")
        self.assertEqual(self.state(SPECTRUM), before)

    def test_not_held(self):
        self.refused("vault.tool_not_held", SPECTRUM, {"position": 1})

    def test_invalid_params_are_free_with_the_engine_reason(self):
        s, swapped = self.secret, self.swapped_guess()
        self.guess(self.view, swapped)
        outsider = next(g for g in all_secrets(4) if g != s and g != swapped and not (set(g) & {s[0]}))
        cases = [(SPECTRUM, {"position": 0}, "bad_position"), (SPECTRUM, {"position": 5}, "bad_position"), (SPECTRUM, {"position": True}, "bad_position"),
                 (SPECTRUM, {"position": "1"}, "bad_position"), (COMPARE, {"a": "3", "b": "3"}, "digits_equal"), (COMPARE, {"a": "x", "b": "3"}, "bad_digit"),
                 (SCANNER, {"guess": outsider, "digit": outsider[0]}, "guess_not_in_history"), (VERIFIER, {"guess": outsider}, "guess_not_in_history")]
        absent_in_guess = [d for d in "0123456789" if d not in swapped]
        cases.append((SCANNER, {"guess": swapped, "digit": absent_in_guess[0]}, "digit_not_in_guess"))
        for key, params, reason in cases:
            self.give(key)
            with self.subTest(key=key, params=params):
                err = self.refused("vault.tool_bad_params", key, params)
                self.assertEqual(err.params["reason"], reason)

    def test_already_revealed_position_is_bad_params_not_redundant(self):
        """Regression (FEAT-64): the order bad_params -> redundant; a disclosed position asked again is refused free."""
        self.give(DECODER, 2)
        self.use(DECODER, {"position": 1})
        err = self.refused("vault.tool_bad_params", DECODER, {"position": 1})
        self.assertEqual(err.params["reason"], "already_revealed")

    def test_verifier_nothing_new_is_publicly_redundant_and_free(self):
        """Regression (FEAT-64): Standard shows the strike count, so a verifier on an already-disclosed lone strike is redundant on public data."""
        s = self.secret
        one_strike = next(g for g in all_secrets(4) if g[0] == s[0] and all(g[i] != s[i] for i in (1, 2, 3)) and len(set(g)) == 4)
        self.guess(self.view, one_strike)
        self.give(DECODER)
        self.use(DECODER, {"position": 1})
        self.give(VERIFIER)
        self.refused("vault.tool_redundant", VERIFIER, {"guess": one_strike})

    def test_total_match_verifier_nothing_new_is_paid_not_a_free_leak(self):
        """Regression (FEAT-64): under Total Match a free nothing_new refusal would reveal the hidden strike count, so it is a paid answer."""
        self.store.set_vault_settings(G, protocols=["standard", "total_match"])
        view = self.open(user=BOB, protocol="total_match", seed="tm")
        s = self.secret_of(view)
        one_strike = s[0] + s[2] + s[3] + s[1]  # shares every digit, only position 1 is a strike
        self.store.submit_vault_guess(G, BOB, view["id"], one_strike, now=NOW)
        self.give(DECODER, user=BOB)
        self.give(VERIFIER, user=BOB)
        self.use(DECODER, {"position": 1}, user=BOB, view=view)
        result = self.use(VERIFIER, {"guess": one_strike}, user=BOB, view=view)
        self.assertEqual((result["text_key"], result["data"]["position"], result["turn_cost"]), ("vault_verifier_position", 1, 1))
        self.assertEqual(self.qty(VERIFIER, user=BOB), 0)

    def test_consume_item_outside_a_transaction_is_a_programming_error(self):
        self.give(SPECTRUM)
        before = self.state(SPECTRUM)
        with self.assertRaises(RuntimeError):
            self.store.consume_item(G, ALICE, CH, SPECTRUM, now=NOW)
        self.assertEqual(self.state(SPECTRUM), before)

    def test_redundant_query_is_free_and_does_not_count_towards_the_cap(self):
        """Regression (FEAT-64): every remaining candidate answers the same, judged on public constraints only."""
        self.give(SPECTRUM, 3)
        self.use(SPECTRUM, {"position": 1})
        self.refused("vault.tool_redundant", SPECTRUM, {"position": 1})
        self.assertEqual(self.qty(SPECTRUM), 2)
        self.use(SPECTRUM, {"position": 2})  # still allowed: the redundant call was not an activation
        self.assertEqual(self.session()["tools_used"], 2)

    def test_redundancy_after_a_decoder_disclosure(self):
        self.give(DECODER)
        self.give(SPECTRUM)
        self.use(DECODER, {"position": 2})
        self.refused("vault.tool_redundant", SPECTRUM, {"position": 2})

    def test_master_decoder_needs_three_turns_left(self):
        """Regression (FEAT-64): tool_fits rule at the store: turns_left 2 refuses the decoder, 3 allows it."""
        self.give(DECODER, 2)
        budget = self.view["turn_budget"]
        for code in self.wrong(self.view, budget - 3):
            self.guess(self.view, code)
        self.assertEqual(self.session()["turns_left"], 3)
        fits = self.use(DECODER, {"position": 1})
        self.assertEqual(fits["session"]["turns_left"], 1)

    def test_master_decoder_refused_with_two_turns_left(self):
        self.give(DECODER)
        for code in self.wrong(self.view, self.view["turn_budget"] - 2):
            self.guess(self.view, code)
        self.assertEqual(self.session()["turns_left"], 2)
        self.refused("vault.tool_no_turns", DECODER, {"position": 1})

    def test_inactive_and_foreign_sessions(self):
        self.give(SPECTRUM, 3)
        self.give(SPECTRUM, 3, user=BOB)
        with self.assertRaises(VaultStoreError) as ctx:
            self.store.use_vault_tool(G, BOB, self.view["id"], SPECTRUM, {"position": 1}, now=NOW)
        self.assertEqual(ctx.exception.code.code, "vault.not_found")
        self.assertEqual(self.qty(SPECTRUM, user=BOB), 3)
        self.fail_vault()
        self.refused("vault.not_active", SPECTRUM, {"position": 1})

    def fail_vault(self):
        for code in self.wrong(self.view, self.view["turn_budget"]):
            self.guess(self.view, code)

    def test_expired_vault_is_not_active_and_keeps_the_item(self):
        self.give(SPECTRUM)
        with self.assertRaises(VaultStoreError) as ctx:
            self.use(SPECTRUM, {"position": 1}, now=NOW + 2 * DAY)
        self.assertEqual(ctx.exception.code.code, "vault.not_active")
        self.assertEqual(self.qty(SPECTRUM), 1)
        self.assertEqual(self.store.item_use_log(G), [])


class BudgetAndReplayTests(ToolCase):
    def test_tool_that_spends_the_last_turn_fails_the_vault(self):
        """Regression (FEAT-64): budget exhaustion by a tool marks the vault failed and reveals the secret."""
        self.give(SPECTRUM, 2)
        absent = [d for d in "0123456789" if d not in self.secret][:4]  # guesses that teach nothing about the high/low halves, so the tool is not redundant
        for code in ["".join(p) for p in itertools.permutations(absent)][:self.view["turn_budget"] - 1]:
            self.guess(self.view, code)
        result = self.use(SPECTRUM, {"position": 1})
        self.assertEqual((result["session"]["status"], result["session"]["turns_left"], result["session"]["secret"]), ("failed", 0, self.secret))
        self.assertEqual(self.qty(SPECTRUM), 1)
        self.refused("vault.not_active", SPECTRUM, {"position": 2})

    def test_replay_returns_the_first_result_and_consumes_nothing(self):
        self.give(SPECTRUM, 2)
        first = self.use(SPECTRUM, {"position": 1}, action_key="int-1")
        again = self.use(SPECTRUM, {"position": 1}, action_key="int-1")
        other = self.use(SPECTRUM, {"position": 2}, action_key="int-1")
        for r in (again, other):
            self.assertTrue(r["replayed"])
            self.assertEqual((r["tool"], r["params"], r["text_key"], r["data"]), (first["tool"], first["params"], first["text_key"], first["data"]))
        self.assertFalse(first["replayed"])
        self.assertEqual(self.qty(SPECTRUM), 1)
        self.assertEqual(self.session()["turns_used"], 1)
        self.assertEqual(len(self.store.item_use_log(G)), 1)

    def test_replay_works_after_the_inventory_is_empty_and_the_vault_ended(self):
        self.give(SPECTRUM)
        first = self.use(SPECTRUM, {"position": 1}, action_key="int-2")
        self.assertEqual(self.qty(SPECTRUM), 0)
        for code in self.wrong(self.view, self.view["turn_budget"] - 1):
            self.guess(self.view, code)
        self.assertEqual(self.session()["status"], "failed")
        again = self.use(SPECTRUM, {"position": 1}, action_key="int-2")
        self.assertTrue(again["replayed"])
        self.assertEqual(again["text_key"], first["text_key"])
        self.assertEqual(len(self.store.item_use_log(G)), 1)

    def test_inventory_runs_out(self):
        self.give(SPECTRUM)
        self.use(SPECTRUM, {"position": 1})
        self.assertEqual(self.qty(SPECTRUM), 0)
        self.assertNotIn(SPECTRUM, [i["key"] for i in self.store.inventory(G, ALICE)])
        self.refused("vault.tool_not_held", SPECTRUM, {"position": 2})


class AtomicityTests(ToolCase):
    def trigger(self, table, when):
        self.store.db.execute(f"CREATE TRIGGER boom BEFORE INSERT ON {table} WHEN {when} BEGIN SELECT RAISE(ABORT, 'boom'); END")
        self.store.db.commit()

    def test_failed_action_row_loses_no_item_and_no_log(self):
        self.give(SPECTRUM)
        self.trigger("vault_tool_uses", "NEW.action_key='boom'")
        with self.assertRaises(sqlite3.Error):
            self.use(SPECTRUM, {"position": 1}, action_key="boom")
        self.assertEqual((self.qty(SPECTRUM), self.store.item_use_log(G), self.session()["turns_used"]), (1, [], 0))

    def test_failed_use_log_row_leaves_no_action_and_no_spent_item(self):
        self.give(SPECTRUM)
        self.trigger("item_uses", "1")
        with self.assertRaises(sqlite3.Error):
            self.use(SPECTRUM, {"position": 1})
        self.assertEqual((self.qty(SPECTRUM), self.session()["tools_used"], self.session()["turns_used"]), (1, 0, 0))


class IsolationTests(ToolCase):
    def test_only_the_members_own_inventory_in_this_guild_counts(self):
        self.give(SPECTRUM, 5, user=BOB)
        self.give(SPECTRUM, 5, guild=G2)
        self.refused("vault.tool_not_held", SPECTRUM, {"position": 1})
        self.assertEqual((self.qty(SPECTRUM, user=BOB), self.qty(SPECTRUM, guild=G2)), (5, 5))

    def test_other_guild_cannot_use_the_session_and_its_inventory_is_untouched(self):
        self.give(SPECTRUM, 2)
        self.give(SPECTRUM, 2, guild=G2)
        with self.assertRaises(VaultStoreError) as ctx:
            self.store.use_vault_tool(G2, ALICE, self.view["id"], SPECTRUM, {"position": 1}, now=NOW)
        self.assertEqual(ctx.exception.code.code, "vault.not_found")
        self.assertEqual((self.qty(SPECTRUM), self.qty(SPECTRUM, guild=G2)), (2, 2))

    def test_use_consumes_only_the_acting_members_unit(self):
        self.give(SPECTRUM, 2)
        self.give(SPECTRUM, 2, user=BOB)
        self.use(SPECTRUM, {"position": 1})
        self.assertEqual((self.qty(SPECTRUM), self.qty(SPECTRUM, user=BOB)), (1, 2))


if __name__ == "__main__":
    unittest.main()


class AllowUseTests(ToolCase):
    def test_a_tool_switched_off_in_the_shop_is_refused_for_free(self):
        """Regression (final review): the shop's Allow use switch governs Vault tools too; the refusal spends nothing."""
        self.give(SPECTRUM)
        self.store.save_shop_item(G, SPECTRUM, expected_revision=0, for_sale=True, allow_use=False)
        self.refused("vault.tool_disabled", SPECTRUM, {"position": 1})
        self.store.save_shop_item(G, SPECTRUM, expected_revision=1, for_sale=True, allow_use=True)
        self.assertEqual(self.use(SPECTRUM, {"position": 1})["qty"], 0)
