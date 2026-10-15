"""Vault NPC advice in the store (FEAT-65 phase A, decision D43). Written before the implementation; fails until it exists.

The pure engine contract is in tests/test_vault_advice_engine.py (``llmcord_core/vault/advice.py``). New API on ``VaultStore``
(``llmcord_core/vault_store.py``); all synchronous, guild_id first, ``now`` where time matters, writes in ``write_admin()``.

  Settings (columns of ``vault_settings``, added to the unreleased v25 CREATE plus the additive migrate pattern)
    vault_settings(guild_id) gains  "advice_cap": 6   NPC advice units one member's vault may spend (default 6, whole number 0..50,
                                                      0 = no advice)
                                    "clue_voice": "template"   or "llm" (admin option for the Reveal clue wording; store only here)
    set_vault_settings(..., advice_cap=None, clue_voice=None) validate like the other fields (bool is not an int, unknown voice
        refused with UserError) and bump the revision once per call.

  record_vault_advice(guild_id, user_id, session_id, character_id, *, text=None, action_key=None, now=None) -> advice
      Delivers and stores ONE piece of advice from one chosen partner. The store computes the facts itself from the session's public
      "constraints" (vault.build_advice_facts, traits from store.resolved_play_style(guild_id, character_id): risk, aggression,
      unpredictability; seed "vault-advice/<session_id>/<character_id>/<turns_used>"). The Discord layer words them: ``text`` is the
      model's wording or None. The store re-checks text with vault.check_wording; text that is None, not a str or fails the check is
      replaced by the deterministic template (fail-soft, no error). Never raises because of the text.
      advice = {"id", "character_id", "name", "guess", "remaining", "source": "llm" | "template", "text": str | None,
                "text_key": ADVICE_TEMPLATE_KEY when source == "template" else None,
                "data": {"guess", "remaining"}, "created_at", "replayed": bool}
      Spending, ONE transaction, both or neither: 1 unit of the member's session advice cap (units already spent in this session are
      counted) AND 1 shared table-talk call, visible as +1 on ``store.game_talk_calls_today(guild_id, now)`` and against
      ``game_talk_daily_limit``. Turns, items and the ledger are untouched. A refusal spends neither and stores nothing.
      Replay: the same action_key in the same session returns the first advice with replayed=True and spends nothing (even when
      the cap or the talk limit is exhausted now).
      Refusal codes (``err.code.code``), in check order:
        vault.not_found           session is not the member's own in this guild
        vault.not_active          solved, failed or expired
        vault.advice_not_partner  character_id is not one of the session's chosen partners (also non-int / bool ids)
        vault.advice_limit        the session's advice_cap is used up (params["limit"] == cap)
        vault.advice_talk_limit   the guild's game_talk_daily_limit is used up for the server day (params["limit"])
      The Vault switch is not checked: an open vault keeps working when the feature is turned off (as guesses do).

  record_vault_meeting(guild_id, user_id, session_id, *, texts=None, action_key=None, now=None) -> [advice, ...]
      Strategy meeting: asks every chosen partner (partners_json order, at most 2) in one transaction, one cap unit and one talk
      call per partner asked. ``texts`` maps character_id -> text (missing = template). All-or-nothing: if the cap or the talk
      limit cannot cover every partner nothing is spent and nothing stored. Same action_key replays the stored batch
      (each advice replayed=True). A vault without partners -> vault.advice_not_partner.
      Codes and check order as above.

  view["advice"]: every stored advice of the session, oldest first, same dicts (replayed False); vault_session and
      open_personal_vault return it, so reopening shows what was said. Advice is only for the owner (vault_session is owner-only).

The secret never appears in stored advice, in the model-facing facts, or in any error text or params.
"""
import unittest

from llmcord_core.games.playstyle import PlayStyle
from llmcord_core.i18n import UserError
from llmcord_core.vault import ADVICE_TEMPLATE_KEY, build_advice_facts, weight_from_style
from llmcord_core.vault_store import VaultStoreError

from test_vault_store import ALICE, BOB, G, G2, NOW, VaultCase, code_of


class AdviceCase(VaultCase):
    def setUp(self):
        super().setUp()
        self.view = self.open(partner_ids=[self.ann, self.bea])
        self.secret = self.secret_of(self.view)

    def ask(self, cid=-1, *, user=ALICE, guild=G, view=None, now=NOW, **kw):
        return self.store.record_vault_advice(guild, user, (view or self.view)["id"], self.ann if cid == -1 else cid, now=now, **kw)

    def meet(self, *, user=ALICE, guild=G, view=None, now=NOW, **kw):
        return self.store.record_vault_meeting(guild, user, (view or self.view)["id"], now=now, **kw)

    def talk(self):
        return self.store.game_talk_calls_today(G, NOW)

    def advice_count(self):
        return len(self.store.vault_session(G, ALICE, self.view["id"], now=NOW)["advice"])

    def state(self):
        s = self.store.vault_session(G, ALICE, self.view["id"], now=NOW)
        return (self.talk(), len(s["advice"]), s["turns_used"], len(s["guesses"]))

    def refused(self, code, call, *args, **kw):
        before = self.state()
        with self.assertRaises(VaultStoreError) as ctx:
            call(*args, **kw)
        self.assertEqual(code_of(ctx.exception), code)
        self.assertEqual(self.state(), before)
        self.assertNotIn(self.secret, repr(ctx.exception.params) + str(ctx.exception.code))
        return ctx.exception

    def usage_row(self, n=1):
        for _ in range(n):
            self.store.db.execute("INSERT INTO model_usage(guild_id,profile,model,role,cached_tokens,reasoning_tokens,cost_basis,created_at,feature) "
                                  "VALUES(?,?,?,?,0,0,'none',?,'game')", (G, "p", "m", "dialogue", NOW))
        self.store.db.commit()


class SettingsTests(unittest.TestCase):
    def setUp(self):
        from llmcord_core.store import Store
        self.store = Store()
        self.addCleanup(self.store.close)

    def test_defaults_are_a_small_cap_and_the_template_voice(self):
        settings = self.store.vault_settings(G)
        self.assertEqual((settings["advice_cap"], settings["clue_voice"]), (6, "template"))

    def test_save_reads_back_bumps_revision_once_and_keeps_other_fields(self):
        self.store.set_vault_settings(G, enabled=True, reward=5)
        saved = self.store.set_vault_settings(G, advice_cap=3, clue_voice="llm")
        self.assertEqual((saved["advice_cap"], saved["clue_voice"], saved["revision"], saved["enabled"], saved["reward"]), (3, "llm", 2, True, 5))
        self.assertEqual(self.store.vault_settings(G), saved)
        back = self.store.set_vault_settings(G, clue_voice="template")
        self.assertEqual((back["advice_cap"], back["clue_voice"]), (3, "template"))
        self.assertEqual(self.store.set_vault_settings(G, advice_cap=0)["advice_cap"], 0)
        self.assertEqual(self.store.set_vault_settings(G, advice_cap=50)["advice_cap"], 50)

    def test_bad_values_are_refused_and_write_nothing(self):
        for case in ({"advice_cap": -1}, {"advice_cap": 51}, {"advice_cap": True}, {"advice_cap": "3"}, {"advice_cap": 2.5},
                     {"clue_voice": "gpt"}, {"clue_voice": ""}, {"clue_voice": None, "advice_cap": "x"}, {"clue_voice": True}):
            with self.subTest(case=case):
                with self.assertRaises(UserError):
                    self.store.set_vault_settings(G, **case)
        self.assertEqual(self.store.vault_settings(G)["revision"], 0)

    def test_settings_are_per_guild(self):
        self.store.set_vault_settings(G, advice_cap=1, clue_voice="llm")
        other = self.store.vault_settings(G2)
        self.assertEqual((other["advice_cap"], other["clue_voice"]), (6, "template"))


class DeliveryTests(AdviceCase):
    def test_template_advice_has_the_engine_guess_and_no_secret_in_its_text(self):
        advice = self.ask()
        view = self.store.vault_session(G, ALICE, self.view["id"], now=NOW)
        facts = build_advice_facts(view["constraints"], view["protocol"], view["length"], risk=self.style(self.ann).risk,
                                   aggression=self.style(self.ann).aggression, unpredictability=self.style(self.ann).unpredictability,
                                   seed=f"vault-advice/{view['id']}/{self.ann}/{view['turns_used']}")
        self.assertEqual((advice["guess"], advice["remaining"]), (facts.guess, facts.remaining))
        self.assertEqual((advice["source"], advice["text"], advice["text_key"]), ("template", None, ADVICE_TEMPLATE_KEY))
        self.assertEqual(advice["data"], {"guess": facts.guess, "remaining": facts.remaining})
        self.assertEqual((advice["character_id"], advice["name"], advice["replayed"]), (self.ann, "Ann", False))

    def style(self, cid):
        return self.store.resolved_play_style(G, cid)

    def test_partner_traits_steer_the_recommendation(self):
        for cid, traits in ((self.ann, dict(risk=100, aggression=100)), (self.bea, dict(risk=0, aggression=0))):
            self.store.set_character_play_style(G, cid, PlayStyle(unpredictability=0, **traits), self.store.owner_revision(G, "character", cid))
        self.fail_to_one_guess()
        bold, careful = self.ask(self.ann), self.ask(self.bea)
        s = self.store.vault_session(G, ALICE, self.view["id"], now=NOW)
        self.assertEqual(bold["guess"], build_advice_facts(s["constraints"], s["protocol"], s["length"], risk=100, aggression=100).guess)
        self.assertEqual(careful["guess"], build_advice_facts(s["constraints"], s["protocol"], s["length"], risk=0, aggression=0).guess)
        self.assertEqual(weight_from_style(100, 100), 1.0)

    def fail_to_one_guess(self):
        self.guess(self.view, self.wrong(self.view)[0])

    def test_model_wording_with_the_guess_is_stored_as_llm(self):
        guess = self.ask(action_key="probe")["guess"]
        advice = self.ask(text=f"Hmm, I'd try {guess}. Trust me.", action_key="voiced")
        self.assertEqual((advice["source"], advice["text"], advice["text_key"]), ("llm", f"Hmm, I'd try {guess}. Trust me.", None))
        self.assertEqual(advice["guess"], guess)

    def test_bad_wording_falls_back_to_the_template_without_an_error(self):
        self.store.set_vault_settings(G, advice_cap=20)
        guess = self.ask(action_key="probe")["guess"]
        other = next(c for c in ("1234", "9876") if c != guess)
        for n, text in enumerate((None, "", f"Try {other}.", f"{guess} or {other}", "No digits here", 1234, f"Try {guess}, digit 7 is in it")):
            with self.subTest(text=text):
                advice = self.ask(text=text, action_key=f"k{n}")
                self.assertEqual((advice["source"], advice["text"], advice["text_key"]), ("template", None, ADVICE_TEMPLATE_KEY))

    def test_the_secret_never_reaches_stored_advice(self):
        advice = self.ask()
        self.assertNotEqual(advice["guess"], self.secret)  # fixed seed: the opening recommendation is not the secret
        leaked = self.ask(text=f"Try {advice['guess']} (really {self.secret})", action_key="leak")
        self.assertEqual(leaked["source"], "template")
        view = self.store.vault_session(G, ALICE, self.view["id"], now=NOW)
        self.assertNotIn(self.secret, repr(view["advice"]))
        self.assertIsNone(view["secret"])

    def test_advice_costs_no_turns_items_or_ledger(self):
        before_ledger = len(self.store.all("SELECT 1 FROM currency_ledger WHERE guild_id=?", (G,)))
        before_log = len(self.store.item_use_log(G))
        self.ask()
        self.meet(action_key="m")
        s = self.store.vault_session(G, ALICE, self.view["id"], now=NOW)
        self.assertEqual((s["turns_used"], s["turns_left"], s["status"]), (0, s["turn_budget"], "active"))
        self.assertEqual(len(self.store.all("SELECT 1 FROM currency_ledger WHERE guild_id=?", (G,))), before_ledger)
        self.assertEqual(len(self.store.item_use_log(G)), before_log)

    def test_same_state_same_advice_and_reopen_shows_it(self):
        a, b = self.ask(action_key="a"), self.ask(action_key="b")
        self.assertEqual((a["guess"], a["remaining"]), (b["guess"], b["remaining"]))
        for view in (self.store.vault_session(G, ALICE, self.view["id"], now=NOW), self.open()):
            shown = view["advice"]
            self.assertEqual([x["id"] for x in shown], [a["id"], b["id"]])
            self.assertEqual([x["guess"] for x in shown], [a["guess"], b["guess"]])
            self.assertTrue(all(x["replayed"] is False for x in shown))

    def test_a_vault_with_no_advice_has_an_empty_list(self):
        self.assertEqual(self.store.vault_session(G, ALICE, self.view["id"], now=NOW)["advice"], [])


class AccountingTests(AdviceCase):
    def test_each_advice_spends_one_cap_unit_and_one_talk_call(self):
        self.store.set_vault_settings(G, advice_cap=3)
        self.assertEqual(self.talk(), 0)
        for n in range(3):
            self.ask(action_key=f"k{n}")
            self.assertEqual((self.talk(), self.advice_count()), (n + 1, n + 1))
        self.refused("vault.advice_limit", self.ask, action_key="k3")

    def test_limit_refusal_reports_the_cap_and_spends_neither(self):
        self.store.set_vault_settings(G, advice_cap=1)
        self.ask()
        err = self.refused("vault.advice_limit", self.ask, action_key="again")
        self.assertEqual(err.params["limit"], 1)

    def test_cap_zero_means_no_advice(self):
        self.store.set_vault_settings(G, advice_cap=0)
        self.refused("vault.advice_limit", self.ask)

    def test_talk_limit_is_shared_with_table_talk_and_refusal_spends_no_cap(self):
        self.store.set_game_talk_daily_limit(G, 2)
        self.usage_row(1)  # a blackjack character already spent one call today
        self.ask(action_key="a")
        self.assertEqual(self.talk(), 2)
        err = self.refused("vault.advice_talk_limit", self.ask, action_key="b")
        self.assertEqual(err.params["limit"], 2)
        self.store.set_game_talk_daily_limit(G, 3)
        self.ask(action_key="c")  # the refused attempt consumed no cap unit
        self.assertEqual((self.talk(), self.advice_count()), (3, 2))

    def test_cap_is_checked_before_the_talk_limit(self):
        self.store.set_vault_settings(G, advice_cap=1)
        self.ask()
        self.store.set_game_talk_daily_limit(G, 1)
        self.refused("vault.advice_limit", self.ask, action_key="x")

    def test_the_cap_counts_per_vault(self):
        self.store.set_vault_settings(G, advice_cap=1)
        bob = self.open(user=BOB, partner_ids=[self.ann])
        self.ask()
        self.store.record_vault_advice(G, BOB, bob["id"], self.ann, now=NOW)  # Bob has his own allowance
        self.assertEqual(self.talk(), 2)

    def test_other_guilds_spending_does_not_count(self):
        self.store.set_game_talk_daily_limit(G, 1)
        self.store.set_vault_settings(G2, enabled=True)
        self.ask()
        self.assertEqual(self.store.game_talk_calls_today(G2, NOW), 0)

    def test_replay_returns_the_first_advice_and_spends_nothing_even_at_the_limits(self):
        self.store.set_vault_settings(G, advice_cap=1)
        self.store.set_game_talk_daily_limit(G, 1)
        first = self.ask(action_key="once")
        before = self.state()
        again = self.ask(action_key="once")
        self.assertEqual((again["id"], again["guess"], again["replayed"]), (first["id"], first["guess"], True))
        self.assertEqual(self.state(), before)

    def test_replay_works_after_the_vault_ended(self):
        first = self.ask(action_key="once")
        self.fail(self.view)
        again = self.ask(action_key="once")
        self.assertEqual((again["id"], again["replayed"]), (first["id"], True))

    def test_a_failed_transaction_leaves_both_counters_alone(self):
        self.store.set_vault_settings(G, advice_cap=1)
        self.ask()
        self.refused("vault.advice_limit", self.ask, text="whatever", action_key="no")
        self.assertEqual(self.talk(), 1)


class RefusalTests(AdviceCase):
    def test_only_chosen_partners(self):
        for cid in (self.cy, self.far, 99999, True, "x", None):
            with self.subTest(cid=cid):
                self.refused("vault.advice_not_partner", self.ask, cid)

    def test_a_vault_without_partners_has_nobody_to_ask(self):
        solo = self.open(user=BOB)
        with self.assertRaises(VaultStoreError) as ctx:
            self.store.record_vault_advice(G, BOB, solo["id"], self.ann, now=NOW)
        self.assertEqual(code_of(ctx.exception), "vault.advice_not_partner")
        with self.assertRaises(VaultStoreError) as ctx:
            self.store.record_vault_meeting(G, BOB, solo["id"], now=NOW)
        self.assertEqual(code_of(ctx.exception), "vault.advice_not_partner")

    def test_owner_only_and_guild_isolation(self):
        self.refused("vault.not_found", self.ask, user=BOB)
        self.refused("vault.not_found", self.ask, guild=G2)
        self.refused("vault.not_found", self.meet, user=BOB)
        self.refused("vault.not_found", self.meet, guild=G2)
        with self.assertRaises(VaultStoreError):
            self.store.vault_session(G, BOB, self.view["id"], now=NOW)
        self.assertEqual(self.store.all("SELECT 1 FROM model_usage"), [])

    def test_ended_vaults_give_no_advice(self):
        self.fail(self.view)
        self.refused("vault.not_active", self.ask)
        self.refused("vault.not_active", self.meet)

    def test_expired_vault_gives_no_advice(self):
        with self.assertRaises(VaultStoreError) as ctx:
            self.store.record_vault_advice(G, ALICE, self.view["id"], self.ann, now=self.view["expires_at"] + 1)
        self.assertEqual(code_of(ctx.exception), "vault.not_active")

    def test_advice_still_works_when_the_feature_is_switched_off(self):
        self.store.set_vault_settings(G, enabled=False)
        self.assertEqual(self.ask()["character_id"], self.ann)


class MeetingTests(AdviceCase):
    def test_asks_every_partner_and_spends_one_unit_each(self):
        out = self.meet(texts={self.ann: None})
        self.assertEqual([a["character_id"] for a in out], [self.ann, self.bea])
        self.assertEqual((self.talk(), self.advice_count()), (2, 2))
        self.assertTrue(all(a["source"] == "template" for a in out))

    def test_one_partner_costs_one_unit(self):
        solo = self.open(user=BOB, partner_ids=[self.bea])
        out = self.store.record_vault_meeting(G, BOB, solo["id"], now=NOW)
        self.assertEqual(([a["character_id"] for a in out], self.talk()), ([self.bea], 1))

    def test_texts_are_checked_per_partner(self):
        probe = self.ask(self.ann, action_key="probe")["guess"]
        out = self.meet(texts={self.ann: f"Go {probe}!", self.bea: "ten is my favourite 4829"})
        self.assertEqual((out[0]["guess"], [a["source"] for a in out]), (probe, ["llm", "template"]))

    def test_cap_that_cannot_cover_everyone_spends_nothing(self):
        self.store.set_vault_settings(G, advice_cap=1)
        self.refused("vault.advice_limit", self.meet)

    def test_cap_with_one_left_is_not_enough_for_two(self):
        self.store.set_vault_settings(G, advice_cap=2)
        self.ask()
        self.refused("vault.advice_limit", self.meet)
        self.assertEqual(self.advice_count(), 1)

    def test_talk_limit_that_cannot_cover_everyone_spends_nothing(self):
        self.store.set_game_talk_daily_limit(G, 1)
        self.refused("vault.advice_talk_limit", self.meet)

    def test_meeting_replays_by_action_key(self):
        first = self.meet(action_key="m")
        before = self.state()
        again = self.meet(action_key="m")
        self.assertEqual([a["id"] for a in again], [a["id"] for a in first])
        self.assertTrue(all(a["replayed"] for a in again))
        self.assertEqual(self.state(), before)


if __name__ == "__main__":
    unittest.main()
