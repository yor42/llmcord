"""Vault partner advice on Discord (llmcord_core/vault_discord.py), FEAT-65 phase B, decision D43. Written before the implementation.

The store side (``record_vault_advice`` / ``record_vault_meeting`` and the ``advice_cap`` / ``clue_voice`` settings, phase A,
tests/test_vault_advice_store.py) and the pure engine (``llmcord_core/vault/advice.py``) are the source of truth: they pick the
recommended guess from PUBLIC constraints, check the wording, spend the per-vault cap and the shared talk allowance in one transaction
and write the one ``model_usage`` row. This layer adds the buttons, asks the model to WORD the facts, and turns results and refusals
into words. Everything is private to the member (``invoker_t``), like the rest of ``/vault personal``. Fake clicks, an in-memory
store and a fake model only; the clock is ``bot.vault.clock``.

CONTRACT (extends the contracts in tests/test_vault_discord.py and tests/test_vault_discord_tools.py)

  Buttons (``render_buttons``; the caller gives it what it needs, for example ``view["partners"]``)
    * While the vault is ACTIVE and has chosen partners, one button per chosen partner that still exists, in partner order, labelled
      ``"Ask {partner name}"`` (a ``VaultButton``, custom id ``llmcord:vault:ask_<character id>:<session id>``; the id carries the
      character and the session, never the member or any code). With TWO partners a ``"Strategy meeting"`` button follows
      (custom id ``llmcord:vault:meeting:<session id>``). With one partner there is no meeting button (it would equal the Ask button).
      Without partners, or once the vault has ended, none of these exist. They sit after Guess, Reveal clue and Tools. The button
      template accepts digits in the action, so ``VaultButton.template`` matches these ids and ``VaultButton.from_custom_id`` rebuilds
      each button (same label, same custom id) and its callback works after a restart.
  Pressing (``VaultPlay.press(interaction, "ask_<character id>" | "meeting", session_id)``)
    * Owner-only: another member gets the usual "This vault belongs to someone else..." text, ephemeral, no edit, no model call, nothing spent.
    * Success: the model is asked ONCE per asked partner (a meeting: once per partner) through ``bot.models.structured_compiled(role,
      request, name, schema)`` (the table-talk seam of game_chooser: ``request.messages`` hold the system and user text, the answer is a
      dict whose string field(s) carry the wording, whatever the schema calls them). The prompt holds only public facts: the partner's
      name and card basics (game_chooser.persona), the public constraints of the vault, the recommended guess and the number of
      candidates left. It NEVER holds the secret (the schema neither). The stored advice is then recorded with the store
      (``text=`` the wording, or None) and the member sees: ONE ``response.edit_message`` refreshing the status message (it now lists
      the advice) and an ephemeral followup with the advice: the model's wording when it passed the store's facts check
      (``source == "llm"``), else the deterministic template line (it names the partner and the recommended guess). A meeting sends
      the advice of both partners (one followup each or one together). No turn, tool or ledger entry is spent.
    * Fail-soft: a model error, a timeout (``TimeoutError``), an unconfigured model (``ModelConfigError``) or wording that fails the
      check (no recommended guess, another 4-digit code, the secret) delivers the template line silently: no error text, the advice
      is stored as ``template`` and still spends exactly one cap unit and one talk call.
    * Usage: the call runs under a usage capture WITHOUT a guild (the gateway then adds no row of its own) and the captured record goes to
      the store, which inserts the one real ``feature='game'`` row of a delivered advice (a placeholder row when no model answered), so n
      advices leave exactly n rows, whatever the model returned.
    * Refusals are the store's codes through ``t.error`` (English texts in the tests below), sent with ``response.send_message(...,
      ephemeral=True)``, with no edit, no model call and nothing spent: ``vault.advice_limit`` ("This vault allows at most {limit}
      pieces of partner advice, and they are used up."), ``vault.advice_talk_limit`` ("This server has used up its {limit} character
      talk calls for today. Try again tomorrow."), ``vault.advice_not_partner`` ("Only a partner you chose for this vault can give
      advice.") and, for a vault that ended since the buttons were drawn, ``vault.not_active`` ("This vault has already ended.") which,
      like a guess, also refreshes the status with an edit.
    * Idempotent by interaction id: the store ``action_key`` is ``str(interaction.id)`` (a meeting uses the same id for all partners). A
      replayed id shows the FIRST stored advice and spends nothing more.
  Status (``render_main``): once advice exists the status lists it (each entry shows the partner name and the advice text, the model's
    wording or the template line with the guess), so reopening ``/vault personal`` shows it again. Character lines are never translated.
  Reveal clue voice (guild setting ``clue_voice``, store only): ``"template"`` (default) makes NO model call. ``"llm"`` words the
    clue being revealed through the model (same seam; the prompt holds the partner and that clue's public text only, never the secret
    or a hidden clue); the wording is used when it keeps the clue's numbers and adds no other digit run, and is sent with the reveal
    (an ephemeral followup). A model failure or wording that fails the check keeps the template clue silently. The status keeps the
    template "Clue from {name}: {text}" line (nothing about the wording is stored). Revealing stays free and spends no advice cap.
  Korean: glossary 파트너, 작전, 금고. Button labels and refusals are hand-translated, the template advice line is Korean; the model's
    wording is shown as the model wrote it.

Open decisions (the implementer decides, then these tests may be adjusted in the same change): the exact label text and button order,
one followup vs several for a meeting, the id shape of the buttons, and whether the layer pre-checks the cap/limits before the model call
(these tests require that a refused press makes no model call).
"""
import asyncio
import json
import re
import unittest
from types import SimpleNamespace

import discord

from llmcord_core.errors import ModelConfigError
from llmcord_core.usage import ModelUsage, _capture
from llmcord_core.vault import build_advice_facts
from llmcord_core.vault_discord import VaultButton, clue_text
from llmcord_core.vault_store import CLUE_KINDS

from test_vault_discord import ALICE, BOB, G, HANGUL, NOT_YOURS, NOW, VaultCase, item_of

LIMIT_TEXT = "This vault allows at most {limit} pieces of partner advice, and they are used up."
TALK_TEXT = "This server has used up its {limit} character talk calls for today. Try again tomorrow."
NOT_PARTNER = "Only a partner you chose for this vault can give advice."
ENDED = "This vault has already ended."
DIGITS = re.compile(r"\d+")


class AdviceModels:
    """Fake ``bot.models``: records every structured call and answers with the wording the test chose.

    ``answer`` is None (obey: word the recommended guess of whichever partner's prompt it is), a string, a callable ``prompt -> str``
    or an exception to raise. Like the real gateway it adds a usage record to an active capture and writes a ``model_usage`` row only when the capture
    has a guild, which is how a layer that records usage itself shows up as a second row.
    """

    def __init__(self, case):
        self.case, self.calls, self.answer = case, [], None

    async def structured_compiled(self, role, request, name, schema):
        prompt = "\n".join(m.text for m in request.messages)
        self.calls.append({"role": role, "name": name, "prompt": prompt, "schema": schema})
        await asyncio.sleep(0)
        answer = self.answer(prompt) if callable(self.answer) else self.answer
        if isinstance(answer, BaseException):
            raise answer
        if answer is None:
            found = [g for g in self.case.expected.values() if g in prompt]
            answer = f"Psst, try {found[0]} and thank me later." if found else "Trust your gut."
        captured = _capture.get()
        if captured:  # like the real gateway: the record joins the capture, and the sink writes a row only for a capture that has a guild
            record = ModelUsage(captured[0], "p", "m", role, 10, 5, 0, 0, None, "none", NOW, captured[2], captured[3])
            captured[1].append(record)
            if captured[0] is not None:
                self.case.store.record_model_usage(record)
        return {key: answer for key, spec in schema.get("properties", {}).items() if spec.get("type") == "string"}


class AdviceUiCase(VaultCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.models = self.bot.models = AdviceModels(self)
        self.expected = {}

    async def start(self, *, partners=("Ann", "Bea"), locale=None):
        """A fresh vault with the chosen partners; the secret is pinned to a code the advice does not recommend."""
        first, second = (list(partners) + [None, None])[:2]
        click = await self.personal(locale=locale, partner=first, second_partner=second)
        sid = self.row()["id"]
        ids = {"Ann": self.ann, "Bea": self.bea}
        self.expected = {ids[n]: self.guess_for(ids[n]) for n in partners}
        secret = next(c for c in ("1357", "2468", "9876") if c not in self.expected.values())
        self.store.db.execute("UPDATE vault_sessions SET secret=? WHERE id=?", (secret, sid))
        self.store.db.commit()
        self.sid = sid
        return click

    def guess_for(self, cid, user=ALICE):
        """The recommendation the engine makes now (the store's seed), for judging prompts and stored advice."""
        view = self.store.vault_session(G, user, self.row(user)["id"], now=self.now)
        style = self.store.resolved_play_style(G, cid)
        seed = f"vault-advice/{view['id']}/{cid}/{view['turns_used']}"
        return build_advice_facts(view["constraints"], view["protocol"], view["length"], risk=style.risk, aggression=style.aggression,
                                  unpredictability=style.unpredictability, seed=seed).guess

    async def act(self, action, *, user=ALICE, ident=None, locale=None, sid=None):
        click = self.click(user, ident=ident, locale=locale)
        await self.vault.press(click, action, sid or self.sid)
        return click

    def button_of(self, click, label):
        for child in self.buttons(self.view_of(click)):
            if item_of(child).label == label:
                return child
        raise AssertionError(f"no {label!r} button among {self.labels(click)}")

    def advice_rows(self):
        return self.store.all("SELECT * FROM vault_advice WHERE session_id=? ORDER BY id", (self.sid,))

    def usage(self):
        return self.store.one("SELECT COUNT(*) FROM model_usage")[0]

    def spent(self):
        """Everything advice may spend or change: advice rows, usage rows, guesses, tools, the talk count."""
        return (len(self.advice_rows()), self.usage(), len(self.actions()), self.store.game_talk_calls_today(G, NOW))

    def edits(self, click):
        return [e.get("content") for e in click.response.edited]

    def refused(self, click, text):
        """One ephemeral response with `text`, no edit, nothing sent as a followup."""
        self.assertEqual([c for c, _ in click.response.sent], [text])
        self.assertTrue(self.ephemeral(click))
        self.assertEqual((click.response.edited, click.followup.sent), ([], []))

    def seen(self, click):
        return "\n".join([self.everything(click), *(c or "" for c in self.edits(click))])


class ButtonTests(AdviceUiCase):
    async def test_one_ask_button_per_partner_and_a_meeting_button_for_two(self):
        """FEAT-65: with two partners the status offers Ask Ann, Ask Bea and Strategy meeting after Guess and Reveal clue; ids carry
        the character and the session only."""
        start = await self.start()
        labels = self.labels(start)
        self.assertEqual(labels[:2], ["Guess", "Reveal clue"])
        self.assertEqual([x for x in labels if x not in ("Guess", "Reveal clue")], ["Ask Ann", "Ask Bea", "Strategy meeting"])
        ids = {item_of(c).label: item_of(c).custom_id for c in self.buttons(self.view_of(start))}
        self.assertEqual(ids["Ask Ann"], f"llmcord:vault:ask_{self.ann}:{self.sid}")
        self.assertEqual(ids["Ask Bea"], f"llmcord:vault:ask_{self.bea}:{self.sid}")
        self.assertEqual(ids["Strategy meeting"], f"llmcord:vault:meeting:{self.sid}")
        self.assert_no_secret(start)

    async def test_one_partner_has_an_ask_button_and_no_meeting(self):
        """FEAT-65 (choice): a meeting needs two partners; with one partner only that partner's Ask button exists."""
        start = await self.start(partners=("Ann",))
        self.assertIn("Ask Ann", self.labels(start))
        self.assertNotIn("Ask Bea", self.labels(start))
        self.assertNotIn("Strategy meeting", self.labels(start))

    async def test_no_partner_no_advice_buttons(self):
        """FEAT-65: a vault without partners has nobody to ask."""
        start = await self.start(partners=())
        self.assertEqual(self.labels(start), ["Guess"])

    async def test_buttons_are_gone_once_the_vault_ended(self):
        """FEAT-65: advice is offered only while the vault is active: a solved vault shows no Ask or meeting button."""
        start = await self.start()
        won = await self.submit(start, self.secret())
        self.assertIn("Status: Solved", self.status(won))
        self.assertEqual(self.labels(won), [])
        self.assertEqual(self.labels(await self.personal()), [])

    async def test_every_advice_button_rebuilds_from_its_custom_id_and_works(self):
        """FEAT-65: after a restart the dynamic item is rebuilt from the custom id alone (label, style, id) and its callback gives advice."""
        start = await self.start()
        for button in [c for c in self.buttons(self.view_of(start)) if item_of(c).label.startswith(("Ask", "Strategy"))]:
            with self.subTest(label=item_of(button).label):
                shown = item_of(button)
                match = VaultButton.__discord_ui_compiled_template__.fullmatch(shown.custom_id)
                self.assertIsNotNone(match, shown.custom_id)
                rebuilt = await VaultButton.from_custom_id(self.click(), shown, match)
                self.assertEqual((item_of(rebuilt).custom_id, item_of(rebuilt).label), (shown.custom_id, shown.label))
                before = len(self.advice_rows())
                click = self.click()
                await rebuilt.callback(click)
                self.assertGreater(len(self.advice_rows()), before)
                self.assertTrue(self.log(click))


class AskTests(AdviceUiCase):
    async def test_valid_wording_is_used_and_listed_on_reopen(self):
        """FEAT-65: the model words the recommended guess; its line is the followup, is stored as llm, costs no turn, and the status lists it."""
        start = await self.start()
        guess = self.expected[self.ann]
        click = await self.press(start, "Ask Ann")
        self.assertEqual(len(self.models.calls), 1)
        call = self.models.calls[0]
        wording = f"Psst, try {guess} and thank me later."
        self.assertIn(wording, self.log(click))
        (row,) = self.advice_rows()
        self.assertEqual((row["source"], row["text"], row["guess"], row["character_id"]), ("llm", wording, guess, self.ann))
        self.assertEqual((len(click.response.edited), len(self.actions())), (1, 0))
        self.assertTrue(self.ephemeral(click))
        self.assertIn(guess, self.edits(click)[-1])
        self.assertIn("Turns left: 9 of 9", self.edits(click)[-1])
        self.assertEqual(self.models.calls, [call])
        again = await self.personal()
        self.assertIn(wording, self.status(again))
        self.assert_no_secret(click, again)

    async def test_the_prompt_holds_public_facts_and_the_recommended_guess_never_the_secret(self):
        """FEAT-65: the captured prompt names the partner, quotes the card basics, holds the recommended guess, and neither the prompt
        nor the schema contains the secret, its digits as a code, or a hidden clue."""
        self.store.db.execute("UPDATE characters SET card=? WHERE id=?", (json.dumps({"description": "A gruff retired locksmith", "personality": "Dry, patient"}), self.ann))
        self.store.db.commit()
        start = await self.start()
        secret = self.secret()
        hidden = [clue_text(CLUE_KINDS[c["kind"]](**json.loads(c["data_json"]))) for c in self.store.all("SELECT * FROM vault_clues WHERE session_id=?", (self.sid,))]
        await self.press(start, "Ask Ann")
        self.assertEqual(len(self.models.calls), 1)
        call = self.models.calls[0]
        prompt = call["prompt"]
        self.assertIn("Ann", prompt)
        self.assertIn("gruff retired locksmith", prompt)
        self.assertIn(self.expected[self.ann], prompt)
        self.assertNotIn(secret, prompt)
        self.assertNotIn(secret, json.dumps(call["schema"]))
        for text in hidden:
            self.assertNotIn(text, prompt)  # hidden clues are facts about the secret

    async def test_the_prompt_after_guesses_and_a_revealed_clue_still_has_no_secret(self):
        """FEAT-65: past guesses and revealed clues are public to the member and may be in the prompt; the secret never is."""
        start = await self.start()
        secret = self.secret()
        await self.submit(start, self.wrong(1)[0])
        await self.press(await self.personal(), "Reveal clue")
        self.expected = {self.ann: self.guess_for(self.ann)}
        fresh = await self.personal()
        await self.press(fresh, "Ask Ann")
        self.assertEqual(len(self.models.calls), 1)
        call = self.models.calls[0]
        self.assertNotIn(secret, call["prompt"])
        self.assertIn(self.expected[self.ann], call["prompt"])

    async def test_wording_that_fails_the_check_falls_back_silently(self):
        """FEAT-65: wording without the guess, with another code, or with the secret is replaced by the template line, with no error shown."""
        start = await self.start()
        self.store.set_vault_settings(G, advice_cap=20)
        guess, secret = self.expected[self.ann], self.secret()
        other = next(c for c in ("4821", "6039") if c != guess)
        for n, wording in enumerate(("No digits here", f"Try {other}.", f"Try {guess} or {other}", f"It is {secret}", f"Try {guess}, digit 7 is in it", "")):
            with self.subTest(wording=wording):
                self.models.answer = wording
                click = await self.press(start, "Ask Ann", ident=1000 + n)
                text = self.log(click)
                self.assertIn(guess, text)
                self.assertNotIn(other, text)
                self.assertNotIn(secret, text)
                self.assertNotIn("Something went wrong", text)
                self.assertEqual(self.advice_rows()[-1]["source"], "template")
                self.assertEqual(len(click.response.edited), 1)

    async def test_model_errors_timeouts_and_missing_models_fall_back_silently(self):
        """FEAT-65: whatever goes wrong with the model, the member gets the template advice, no error, and it spends one cap unit once."""
        start = await self.start()
        self.store.set_vault_settings(G, advice_cap=20)
        guess = self.expected[self.ann]
        failures = (RuntimeError("boom"), TimeoutError(), ModelConfigError("no model is configured"), ValueError("bad"))
        for n, failure in enumerate(failures):
            with self.subTest(error=type(failure).__name__):
                self.models.answer = failure
                before = len(self.advice_rows())
                click = await self.press(start, "Ask Ann", ident=2000 + n)
                text = self.everything(click)
                self.assertIn(guess, text)
                for word in ("boom", "no model", "Something went wrong", "rror"):
                    self.assertNotIn(word, text)
                self.assertEqual(len(self.advice_rows()), before + 1)
                self.assertEqual(self.advice_rows()[-1]["source"], "template")
        self.assertEqual(self.usage(), len(failures))

    async def test_the_model_path_writes_no_usage_row_of_its_own(self):
        """FEAT-65: n delivered advices leave exactly n model_usage rows (the store's), not n plus the model call's own."""
        start = await self.start()
        await self.press(start, "Ask Ann", ident=3001)
        self.assertEqual((len(self.advice_rows()), self.usage()), (1, 1))
        await self.press(start, "Ask Bea", ident=3002)
        self.assertEqual((len(self.advice_rows()), self.usage()), (2, 2))
        self.assertEqual({r["source"] for r in self.advice_rows()}, {"llm"})

    async def test_the_stored_usage_row_is_the_real_call_not_a_placeholder(self):
        """FEAT-65: a model-worded advice leaves one usage row carrying the call's own profile and tokens; a template advice leaves the placeholder."""
        start = await self.start()
        await self.press(start, "Ask Ann", ident=1)
        self.models.answer = RuntimeError("down")
        await self.press(start, "Ask Bea", ident=2)
        rows = self.store.all("SELECT profile,input_tokens,feature FROM model_usage ORDER BY id")
        self.assertEqual([tuple(r) for r in rows], [("p", 10, "game"), ("vault", None, "game")])

    async def test_advice_spends_one_cap_unit_and_one_talk_call_only(self):
        """FEAT-65: advice costs no turn, no tool and no ledger entry; the cap and the shared talk count each go up by one."""
        start = await self.start()
        inventory = self.store.all("SELECT * FROM inventory")
        ledger = self.store.one("SELECT COUNT(*) FROM ledger")[0] if self.store.one("SELECT name FROM sqlite_master WHERE name='ledger'") else 0
        await self.press(start, "Ask Ann")
        view = self.store.vault_session(G, ALICE, self.sid, now=NOW)
        self.assertEqual((view["turns_left"], len(view["advice"]), self.store.game_talk_calls_today(G, NOW)), (9, 1, 1))
        self.assertEqual([tuple(r) for r in self.store.all("SELECT * FROM inventory")], [tuple(r) for r in inventory])
        if self.store.one("SELECT name FROM sqlite_master WHERE name='ledger'"):
            self.assertEqual(self.store.one("SELECT COUNT(*) FROM ledger")[0], ledger)

    async def test_a_replayed_interaction_shows_the_first_advice_and_spends_nothing(self):
        """FEAT-65: pressing with the same interaction id again re-shows the stored advice, not a new one, and adds no row or usage."""
        start = await self.start()
        guess = self.expected[self.ann]
        first = await self.press(start, "Ask Ann", ident=4242)
        self.models.answer = f"Different words: {guess}!"
        again = await self.press(start, "Ask Ann", ident=4242)
        self.assertIn(f"Psst, try {guess} and thank me later.", self.log(first))
        self.assertIn(f"Psst, try {guess} and thank me later.", self.log(again))
        self.assertNotIn("Different words", self.seen(again))
        self.assertEqual((len(self.advice_rows()), self.usage()), (1, 1))

    async def test_advice_for_a_different_press_is_a_new_one(self):
        """FEAT-65: a second press with a new interaction id is a second advice (a second cap unit)."""
        start = await self.start()
        await self.press(start, "Ask Ann", ident=1)
        await self.press(start, "Ask Ann", ident=2)
        self.assertEqual((len(self.advice_rows()), self.usage()), (2, 2))


class RefusalTests(AdviceUiCase):
    async def test_the_vault_cap_refuses_with_nothing_spent_and_no_model_call(self):
        """FEAT-65: once the per-vault advice cap is used up the press says so, ephemerally, and neither spends nor asks the model."""
        start = await self.start()
        self.store.set_vault_settings(G, advice_cap=1)
        await self.press(start, "Ask Ann", ident=1)
        before, calls = self.spent(), len(self.models.calls)
        click = await self.press(start, "Ask Bea", ident=2)
        self.refused(click, LIMIT_TEXT.format(limit=1))
        self.assertEqual((self.spent(), len(self.models.calls)), (before, calls))

    async def test_a_zero_cap_means_no_advice(self):
        """FEAT-65: advice_cap 0 switches advice off for the server: the press is refused and nothing is asked or written."""
        start = await self.start()
        self.store.set_vault_settings(G, advice_cap=0)
        click = await self.press(start, "Ask Ann")
        self.refused(click, LIMIT_TEXT.format(limit=0))
        self.assertEqual((self.spent(), self.models.calls), ((0, 0, 0, 0), []))

    async def test_the_shared_talk_limit_refuses_and_spends_nothing(self):
        """FEAT-65: when the server's daily character talk allowance is gone the press is refused with its own text; no cap unit is spent."""
        start = await self.start()
        self.store.set_game_talk_daily_limit(G, 1)
        self.store.db.execute("INSERT INTO model_usage(guild_id,profile,model,role,cached_tokens,reasoning_tokens,cost_basis,created_at,feature) "
                              "VALUES(?,?,?,?,0,0,'none',?,'game')", (G, "p", "m", "dialogue", NOW))
        self.store.db.commit()
        before = self.spent()
        click = await self.press(start, "Ask Ann")
        self.refused(click, TALK_TEXT.format(limit=1))
        self.assertEqual((self.spent(), self.models.calls), (before, []))

    async def test_a_character_that_is_not_a_partner_is_refused(self):
        """FEAT-65: a forged or stale ask for a character the member did not choose (Cy) is refused; nothing is asked or spent."""
        await self.start()
        click = await self.act(f"ask_{self.cy}")
        self.refused(click, NOT_PARTNER)
        self.assertEqual((self.spent(), self.models.calls), ((0, 0, 0, 0), []))

    async def test_an_ended_vault_refuses_and_refreshes_the_status(self):
        """FEAT-65: pressing a stale Ask button after the vault ended shows the ended status (with an edit) and the store's text; no model call."""
        start = await self.start()
        button = self.button_of(start, "Ask Ann")
        await self.submit(start, self.secret())  # solved: the vault ended
        before = self.spent()
        click = self.click()
        await button.callback(click)
        self.assertIn(ENDED, self.log(click) + self.everything(click))
        self.assertTrue(self.ephemeral(click))
        self.assertIn("Status: Solved", self.edits(click)[-1])
        self.assertEqual((self.spent(), self.models.calls), (before, []))

    async def test_another_member_cannot_ask_through_my_buttons(self):
        """FEAT-65: owner-only: Bob pressing Alice's Ask button gets the usual not-yours text; nothing is asked, spent or edited."""
        start = await self.start()
        click = await self.press(start, "Ask Ann", user=BOB)
        self.refused(click, NOT_YOURS)
        self.assertEqual((self.spent(), self.models.calls), ((0, 0, 0, 0), []))

    async def test_another_server_cannot_ask_through_my_buttons(self):
        """FEAT-65: guild isolation: the same button pressed in another server finds no such vault; nothing happens."""
        start = await self.start()
        button = self.button_of(start, "Ask Ann")
        click = self.click(guild_id=2)
        await button.callback(click)
        self.assertEqual((self.spent(), self.models.calls), ((0, 0, 0, 0), []))
        self.assertEqual(click.response.edited, [])
        self.assertNotIn(self.expected[self.ann], self.everything(click))

    async def test_a_refusal_never_shows_the_secret(self):
        """FEAT-65: refusal texts carry the limit or nothing, never the code."""
        start = await self.start()
        self.store.set_vault_settings(G, advice_cap=0)
        click = await self.press(start, "Ask Ann")
        self.assert_no_secret(click)


class MeetingTests(AdviceUiCase):
    async def test_a_meeting_asks_both_partners_and_spends_one_unit_each(self):
        """FEAT-65: Strategy meeting asks the model once per partner (each prompt names its partner), stores two advices, shows both
        lines, edits the status once, and spends two cap units, two talk calls and two usage rows."""
        start = await self.start()
        click = await self.press(start, "Strategy meeting")
        self.assertEqual(len(self.models.calls), 2)
        prompts = [c["prompt"] for c in self.models.calls]
        self.assertTrue(any("Ann" in p for p in prompts) and any("Bea" in p for p in prompts))
        rows = self.advice_rows()
        self.assertEqual({r["character_id"] for r in rows}, {self.ann, self.bea})
        for row in rows:
            self.assertEqual((row["source"], row["guess"]), ("llm", self.expected[row["character_id"]]))
            self.assertIn(row["text"], self.log(click))
        self.assertEqual((len(click.response.edited), self.usage(), self.store.game_talk_calls_today(G, NOW), len(self.actions())), (1, 2, 2, 0))
        self.assertTrue(self.ephemeral(click))
        again = await self.personal()
        for row in rows:
            self.assertIn(row["text"], self.status(again))
        self.assert_no_secret(click, again)

    async def test_each_partners_wording_is_checked_on_its_own(self):
        """FEAT-65: one partner's bad wording falls back to the template; the other's good wording is kept."""
        start = await self.start()
        ann_guess = self.expected[self.ann]

        def answer(prompt):
            return "I have no idea." if "Bea" in prompt else f"Go with {ann_guess}."
        self.models.answer = answer
        await self.press(start, "Strategy meeting")
        sources = {r["character_id"]: r["source"] for r in self.advice_rows()}
        self.assertEqual(sources, {self.ann: "llm", self.bea: "template"})

    async def test_a_meeting_with_model_trouble_still_delivers_template_advice(self):
        """FEAT-65: a failing model never blocks the meeting: both partners advise by template, no error text."""
        start = await self.start()
        self.models.answer = RuntimeError("down")
        click = await self.press(start, "Strategy meeting")
        self.assertEqual([r["source"] for r in self.advice_rows()], ["template", "template"])
        text = self.everything(click)
        for guess in self.expected.values():
            self.assertIn(guess, text)
        self.assertNotIn("down", text)

    async def test_a_cap_that_cannot_cover_both_spends_nothing(self):
        """FEAT-65: with one cap unit left the meeting (which needs two) is refused whole: nothing stored, asked or counted."""
        start = await self.start()
        self.store.set_vault_settings(G, advice_cap=1)
        click = await self.press(start, "Strategy meeting")
        self.refused(click, LIMIT_TEXT.format(limit=1))
        self.assertEqual((self.spent(), self.models.calls), ((0, 0, 0, 0), []))

    async def test_a_talk_limit_that_cannot_cover_both_spends_nothing(self):
        """FEAT-65: the shared allowance is checked for both partners together."""
        start = await self.start()
        self.store.set_game_talk_daily_limit(G, 1)
        click = await self.press(start, "Strategy meeting")
        self.refused(click, TALK_TEXT.format(limit=1))
        self.assertEqual((self.spent(), self.models.calls), ((0, 0, 0, 0), []))

    async def test_a_replayed_meeting_spends_nothing_more(self):
        """FEAT-65: the same interaction id again shows the same two advices and adds no row or usage."""
        start = await self.start()
        await self.press(start, "Strategy meeting", ident=777)
        before = self.spent()
        again = await self.press(start, "Strategy meeting", ident=777)
        self.assertEqual(self.spent(), before)
        for row in self.advice_rows():
            self.assertIn(row["text"], self.log(again))

    async def test_only_the_owner_can_call_a_meeting(self):
        """FEAT-65: owner-only, like Ask."""
        start = await self.start()
        click = await self.press(start, "Strategy meeting", user=BOB)
        self.refused(click, NOT_YOURS)
        self.assertEqual(self.spent(), (0, 0, 0, 0))


class ClueVoiceTests(AdviceUiCase):
    async def revealing(self, start):
        """Reveal the next clue; returns (click, template text of the clue that was revealed, holder name)."""
        row = self.store.all("SELECT * FROM vault_clues WHERE session_id=? AND revealed_at IS NULL ORDER BY position", (self.sid,))[0]
        template = clue_text(CLUE_KINDS[row["kind"]](**json.loads(row["data_json"])))
        click = await self.press(start, "Reveal clue")
        return click, template, row

    async def test_the_default_voice_is_the_template_and_asks_no_model(self):
        """FEAT-65: clue_voice 'template' (the default) reveals the clue with the fixed text and makes no model call."""
        start = await self.start()
        click, template, _ = await self.revealing(start)
        self.assertIn(template, self.edits(click)[-1])
        self.assertEqual(self.models.calls, [])
        self.assertEqual(self.usage(), 0)

    async def test_the_llm_voice_words_the_clue_through_the_model(self):
        """FEAT-65: with clue_voice 'llm' the model is asked once with the clue's public text (no secret, no hidden clue) and its wording
        reaches the member; revealing spends no advice cap and no turn."""
        self.store.set_vault_settings(G, clue_voice="llm")
        start = await self.start()
        secret = self.secret()
        hidden = [clue_text(CLUE_KINDS[c["kind"]](**json.loads(c["data_json"]))) for c in self.store.all("SELECT * FROM vault_clues WHERE session_id=? ORDER BY position", (self.sid,))]
        row = self.store.all("SELECT * FROM vault_clues WHERE session_id=? ORDER BY position", (self.sid,))[0]
        template = hidden[0]
        wording = f"Listen close: {template}"
        self.models.answer = wording
        click = await self.press(start, "Reveal clue")
        self.assertEqual(len(self.models.calls), 1)
        call = self.models.calls[0]
        self.assertIn(template, call["prompt"])
        self.assertNotIn(secret, call["prompt"])
        if hidden[1] != template:
            self.assertNotIn(hidden[1], call["prompt"])
        self.assertIn(wording, self.seen(click))
        self.assertIsNotNone(self.store.one("SELECT 1 FROM vault_clues WHERE id=? AND revealed_at IS NOT NULL", (row["id"],)))
        self.assertEqual((len(self.advice_rows()), len(self.actions())), (0, 0))
        self.assertEqual(self.store.vault_session(G, ALICE, self.sid, now=NOW)["turns_left"], 9)
        self.assert_no_secret(click)

    async def test_a_failed_defer_still_shows_the_status_after_the_committed_reveal(self):
        """Regression (final review): with the llm voice the reveal is committed before the defer; if Discord refuses the defer the press
        must still edit the normal status (template path), make no model call, and not reveal a second clue."""
        self.store.set_vault_settings(G, clue_voice="llm")
        await self.start()
        click = self.click()

        async def refuse(**kwargs):
            raise discord.HTTPException(SimpleNamespace(status=500, reason="x"), "x")
        click.response.defer = refuse
        await self.bot.vault.press(click, "clue", self.sid)
        self.assertEqual(self.models.calls, [])
        self.assertEqual(len(click.response.edited), 1)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_clues WHERE session_id=? AND revealed_at IS NOT NULL", (self.sid,))[0], 1)

    async def test_bad_clue_wording_keeps_the_template_clue_silently(self):
        """FEAT-65: wording that drops the clue's numbers or adds another digit run is not shown; the template clue is."""
        self.store.set_vault_settings(G, clue_voice="llm")
        start = await self.start()
        row = self.store.all("SELECT * FROM vault_clues WHERE session_id=? ORDER BY position", (self.sid,))[0]
        template = clue_text(CLUE_KINDS[row["kind"]](**json.loads(row["data_json"])))
        numbers = DIGITS.findall(template)
        bad = ("Nothing to see here" if not numbers else "I will not say numbers", f"{template} The code is 4821.")
        for n, wording in enumerate(bad):
            with self.subTest(wording=wording):
                self.store.db.execute("UPDATE vault_clues SET revealed_at=NULL WHERE session_id=?", (self.sid,))
                self.store.db.commit()
                self.models.answer = wording
                click = await self.press(start, "Reveal clue", ident=5000 + n)
                self.assertNotIn(wording, self.seen(click))
                self.assertIn(template, self.edits(click)[-1])
                self.assertNotIn("Something went wrong", self.seen(click))

    async def test_a_model_failure_keeps_the_template_clue_silently(self):
        """FEAT-65: an error, a timeout or a missing model never blocks Reveal clue."""
        self.store.set_vault_settings(G, clue_voice="llm")
        start = await self.start()
        for n, failure in enumerate((RuntimeError("boom"), TimeoutError(), ModelConfigError("no model is configured"))):
            with self.subTest(error=type(failure).__name__):
                self.store.db.execute("UPDATE vault_clues SET revealed_at=NULL WHERE session_id=?", (self.sid,))
                self.store.db.commit()
                row = self.store.all("SELECT * FROM vault_clues WHERE session_id=? ORDER BY position", (self.sid,))[0]
                template = clue_text(CLUE_KINDS[row["kind"]](**json.loads(row["data_json"])))
                self.models.answer = failure
                click = await self.press(start, "Reveal clue", ident=6000 + n)
                self.assertIn(template, self.edits(click)[-1])
                for word in ("boom", "no model", "Something went wrong", "rror"):
                    self.assertNotIn(word, self.seen(click))

    async def test_the_llm_voice_counts_as_a_game_talk_call_and_respects_the_allowance(self):
        """FEAT-65: a model-worded clue writes the real feature='game' usage row (so it counts toward the daily talk limit); with the
        allowance used up the clue is revealed by template with no model call."""
        self.store.set_vault_settings(G, clue_voice="llm")
        start = await self.start()
        await self.press(start, "Reveal clue")
        self.assertEqual((len(self.models.calls), self.store.game_talk_calls_today(G, NOW), len(self.advice_rows())), (1, 1, 0))
        self.store.set_game_talk_daily_limit(G, 1)
        before = len(self.models.calls)
        click = await self.press(await self.personal(), "Reveal clue")
        self.assertEqual(len(self.models.calls), before)
        self.assertNotIn("Something went wrong", self.seen(click))
        self.assertEqual(self.store.game_talk_calls_today(G, NOW), 1)

    async def test_the_template_voice_after_switching_back_asks_no_model(self):
        """FEAT-65: the setting is read at each press: switching back to 'template' stops model calls at once."""
        self.store.set_vault_settings(G, clue_voice="llm")
        start = await self.start()
        self.store.set_vault_settings(G, clue_voice="template")
        await self.press(start, "Reveal clue")
        self.assertEqual(self.models.calls, [])

    async def test_an_llm_clue_voice_does_not_touch_advice_accounting(self):
        """FEAT-65: Reveal clue with the model voice spends no advice cap (the store counts advice only)."""
        self.store.set_vault_settings(G, clue_voice="llm", advice_cap=1)
        start = await self.start()
        self.models.answer = RuntimeError("down")
        await self.press(start, "Reveal clue")
        click = await self.press(start, "Ask Ann")
        self.assertEqual(len(self.advice_rows()), 1)
        self.assertIn(self.advice_rows()[0]["guess"], self.log(click))


class KoreanTests(AdviceUiCase):
    async def test_korean_buttons_and_template_advice(self):
        """FEAT-65: in Korean every advice button label has hangul (the meeting one uses 작전), the template advice line is Korean and
        names the guess; the model's own wording is shown as the model wrote it."""
        start = await self.start(locale="ko")
        shown = [item_of(c).label for c in self.buttons(self.view_of(start))]
        asks = [x for x in shown if "Ann" in x or "Bea" in x]
        self.assertEqual(len(asks), 2)
        self.assertTrue(all(HANGUL.search(x) for x in shown), shown)
        self.assertTrue(any("작전" in x for x in shown), shown)
        self.models.answer = RuntimeError("down")
        click = await self.press(start, asks[0], locale="ko")
        text = self.log(click)
        self.assertRegex(text, HANGUL)
        self.assertIn(self.expected[self.ann], text)
        self.assertTrue(self.ephemeral(click))

    async def test_korean_refusals_are_translated_and_a_model_line_is_not(self):
        """FEAT-65: a Korean member gets the cap refusal in Korean; the model's English wording is kept as written."""
        start = await self.start(locale="ko")
        self.store.set_vault_settings(G, advice_cap=1)
        asks = {n: next((x for x in self.labels(start) if n in x), None) for n in ("Ann", "Bea")}
        self.assertTrue(all(asks.values()), self.labels(start))
        first = await self.press(start, asks["Ann"], locale="ko", ident=1)
        guess = self.expected[self.ann]
        self.assertIn(f"Psst, try {guess} and thank me later.", self.log(first))
        refused = await self.press(start, asks["Bea"], locale="ko", ident=2)
        self.assertRegex(self.everything(refused), HANGUL)
        self.assertNotIn(LIMIT_TEXT.format(limit=1), self.everything(refused))
        self.assertEqual(refused.response.edited, [])


class EnglishIsPlainTests(AdviceUiCase):
    async def test_english_advice_has_no_hangul(self):
        """FEAT-65: tests run in English; the template advice and labels carry no Korean."""
        start = await self.start()
        self.models.answer = RuntimeError("down")
        click = await self.press(start, "Ask Ann")
        self.assertIsNone(HANGUL.search(self.seen(click) + " ".join(self.labels(start))))


if __name__ == "__main__":
    unittest.main()
