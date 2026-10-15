"""Vault support missions on Discord, FEAT-67 phase B, decision D43. Written before the implementation; they fail until it exists.

The store (``vault_support_store.py``: ``start_support_mission`` / ``support_mission`` / ``submit_support_answer``, phase A) is the only source of
truth and raises ``VaultStoreError`` codes. This layer, in ``llmcord_core/vault_operation_discord.py`` (or a module it imports), only takes input and
builds words. Everything a member sees is an ephemeral reply in the invoker's language (``invoker_t``); nothing is posted in the channel. Fakes,
clicks and the in-memory store come from ``test_vault_operation_discord.OpCase`` (``helpers`` only); time is ``bot.vault_operation.clock``.

CONTRACT

  Buttons (all ``VaultOperationButton``, so ``llmcord:vault:op_<action>:<operation_id>``; one class, one template, nothing new to register in
  ``setup_hook``; none matches ``VaultButton.template``; none carries a member, a code or an answer):
      status button   label "Support mission", action ``mission``, i.e. ``llmcord:vault:op_mission:<operation_id>``, shown while the operation is
                      active and placed right after "Cross-check". Its press goes through ``VaultOperationPlay.press(interaction, "mission", oid)``.
      picker buttons  "Record forensics" (``mission_forensics``) and "Three-digit Bulls and Cows" (``mission_bulls_cows``).
      answer button   "Answer" (``mission_answer``), only on a mission that is still active.
  Press ``mission`` (participant check first: a non-participant gets "Join the operation first.", a click from another guild "That vault was not
      found.", an ended or expired operation "This vault has already ended."; nothing is created in any of these cases):
      * no mission yet -> an ephemeral message "Choose a support mission." whose view holds exactly the two picker buttons.
      * a mission exists -> an ephemeral mission message instead (no picker buttons), see below.
  Picker button press -> ``store.start_support_mission(guild, user, operation, type, now=clock())`` (the layer may run it through
      ``asyncio.to_thread``: the tests only look at the outcome and accept a defer plus followup or a plain response), then the mission message.
      Pressing the picker button of the OTHER type once a mission exists is refused with the store's text ("You already took a different support
      mission in this operation."), the persisted mission is untouched.
  Mission message (one string, lines in this order, ``?`` conditional):
      "Support mission: {name}"                  name = "Record forensics" | "Three-digit Bulls and Cows"
      forensics: "Recorded guesses of one secret code:" then one line per record "{n}. {guess} — {feedback}" (feedback as ``feedback_text`` in
                 the standard protocol, e.g. "1 strike, 2 balls"); the hidden record reads "{n}. {guess} — ?"; then
                 "What are the strikes and balls of line {n}?"
      bulls_cows: "Guess a code of 3 different digits." then one line per answer so far "{n}. {guess} — {feedback}"
      "Attempts left: {left} of {budget}"        (forensics budget 2, Bulls and Cows 6)
      ended only: "Support mission complete!" or "Support mission failed.", then the answer line ``"The code was {code}."`` (Bulls and Cows) or
                 ``"The hidden result was {feedback}."`` (forensics). The answer line exists ONLY for a solved or failed mission.
      view: the "Answer" button while the mission is active, no view otherwise.
  Answer modals (``OperationMissionModal(operation_id, mission_type, *, t=None)``; ``.inputs`` keyed by name, like ``OperationToolModal``):
      bulls_cows: one input ``code``.    forensics: two inputs ``strikes`` and ``balls``.
      Submit calls ``VaultOperationPlay.submit_mission(interaction, operation_id, mission_type, values)`` with the typed texts; store
      ``action_key=str(interaction.id)``. Non-numeric forensics input is refused with the store's invalid-answer text ("Answer with the strikes
      and balls ..."), like an out-of-range pair; a malformed code with the same text as a malformed guess. Invalid and repeated answers cost
      nothing (no attempt row) and leave the mission active.
  Answer reply (ephemeral):
      wrong, still active: "Answer {n}: {guess} — {feedback}" (Bulls and Cows) or "Answer {n}: {feedback}" (forensics), then
                 "Attempts left: {left} of {budget}". Never the secret or the hidden result.
      solved:    "Support mission complete!" + the answer line + EITHER "Reward: {amount}" (a reward was paid) OR "No reward was paid for this mission."
      failed:    "Support mission failed." + the answer line (the answer is revealed because the mission is terminal).
  Idempotent: the same interaction id twice records one attempt, pays once and answers with the same feedback.
  Refusals are ``t.error(code)`` in the invoker's language; free; no model call.
  Korean glossary: 작전, 단계, 금고 (the Korean strings are hand written; this file only smokes that they are Korean).
"""
import json
from llmcord_core import i18n
from llmcord_core.vault import bulls_cows_feedback, bulls_cows_secrets
from llmcord_core.vault_discord import VaultButton
from llmcord_core.vault_store import VaultStoreError

import test_vault_operation_discord as ops
from test_vault_discord import ALICE, BOB, G, HANGUL, fb_en, item_of

from llmcord_core.vault_operation_discord import VaultOperationButton

try:
    from llmcord_core.vault_operation_discord import OperationMissionModal
except ImportError:  # pragma: no cover - the red state
    OperationMissionModal = None

PICK = "Choose a support mission."
FORENSICS, BULLS = "Record forensics", "Three-digit Bulls and Cows"
SOLVED, FAILED = "Support mission complete!", "Support mission failed."
NO_REWARD = "No reward was paid for this mission."
EXISTS = "You already took a different support mission in this operation."
ENDED = "This vault has already ended."
INVALID_FORENSICS = ("Answer with the strikes and balls of the hidden record, for example 1 and 2. Together they cannot be more than 4, "
                     "and 3 strikes with 1 ball is impossible.")
NOT_FOUND = "That vault was not found."
MISSING = "OperationMissionModal is missing (FEAT-67 phase B)"


class MissionCase(ops.OpCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.start(BOB)  # Alice and Bob participate; the stage secret is fixed

    # --- reading replies --------------------------------------------------------------------------

    @staticmethod
    def views(click):
        return [kw["view"] for _, kw in click.response.sent + click.followup.sent if kw.get("view") is not None]

    def labels_of(self, click):
        return [item_of(c).label for v in self.views(click) for c in v.children]

    def ids_of(self, click):
        return [item_of(c).custom_id for v in self.views(click) for c in v.children]

    async def tap(self, source, label, user=ALICE, **kw):
        for view in self.views(source):
            for child in view.children:
                if item_of(child).label == label:
                    click = self.click(user, **kw)
                    await child.callback(click)
                    return click
        raise AssertionError(f"no {label!r} among {self.labels_of(source)}")

    # --- driving ----------------------------------------------------------------------------------

    async def menu(self, user=ALICE, **kw):
        return await self.action("mission", user, **kw)

    async def begin(self, kind=BULLS, user=ALICE, **kw):
        """Mission button -> picker -> the picker button of `kind`. Returns (picker click, mission click)."""
        picker = await self.menu(user, **kw)
        return picker, await self.tap(picker, kind, user, **kw)

    async def answer(self, source, values, user=ALICE, ident=None, **kw):
        """Answer button of `source` -> modal -> the member types `values` (a dict, or a string for the one code input) -> submit."""
        button = next(c for v in self.views(source) for c in v.children if "op_mission_answer:" in item_of(c).custom_id)  # by id: the label is translated
        opened = self.click(user, **kw)
        await button.callback(opened)
        if not opened.response.modals:  # refused at the button instead of at the submit: the reply is the refusal
            return opened
        modal = opened.response.modals[0]
        values = {"code": values} if isinstance(values, str) else values
        for name, text in values.items():
            modal.inputs[name]._value = text
        submit = self.click(user, ident=ident, **kw)
        await modal.on_submit(submit)
        return submit

    # --- oracle -----------------------------------------------------------------------------------

    def mission(self, user=ALICE):
        return self.store.one("SELECT * FROM vault_support_missions WHERE operation_id=? AND user_id=?", (self.oid, user))

    def truth(self, user=ALICE):
        return json.loads(self.mission(user)["answer_json"])

    def puzzle(self, user=ALICE):
        return json.loads(self.mission(user)["puzzle_json"])

    def used(self, user=ALICE):
        return self.store.one("SELECT COUNT(*) FROM vault_support_attempts WHERE mission_id=?", (self.mission(user)["id"],))[0]

    def missions(self):
        return self.store.one("SELECT COUNT(*) FROM vault_support_missions")[0]

    def wrong_code(self, n=1, user=ALICE):
        truth = self.truth(user)
        return [c for c in bulls_cows_secrets() if c != truth][:n]

    def wrong_pair(self, user=ALICE):
        truth = tuple(self.truth(user))
        return next((s, b) for s in range(4) for b in range(4 - s + 1) if (s, b) != truth and (s, b) != (3, 1))

    def money(self, user=ALICE):
        return self.store.balance(G, user)

    async def solve_bulls(self, user=ALICE, **kw):
        _, shown = await self.begin(BULLS, user, **kw)
        return await self.answer(shown, self.truth(user), user, **kw)

    async def solve_forensics(self, user=ALICE, **kw):
        _, shown = await self.begin(FORENSICS, user, **kw)
        s, b = self.truth(user)
        return await self.answer(shown, {"strikes": str(s), "balls": str(b)}, user, **kw)


class ButtonTests(MissionCase):
    async def test_the_support_mission_button_is_on_the_status_with_the_operation_custom_id(self):
        """FEAT-67: the status message has a "Support mission" button right after Cross-check, id llmcord:vault:op_mission:<operation id>."""
        labels = self.labels_now()
        self.assertEqual(labels[:5], ["Guess", "Reveal clue", "Tools", "Cross-check", "Support mission"])
        child = next(c for c in self.message().view.children if item_of(c).label == "Support mission")
        self.assertIsInstance(child, VaultOperationButton)
        self.assertEqual(item_of(child).custom_id, f"llmcord:vault:op_mission:{self.oid}")

    async def test_the_mission_ids_survive_a_restart_and_collide_with_nothing(self):
        """FEAT-67: op_mission, op_mission_forensics, op_mission_bulls_cows and op_mission_answer are all VaultOperationButton ids that rebuild
        with the exact action; the personal VaultButton template matches none of them; ask_<id> and the other op_ actions are unaffected."""
        template = VaultOperationButton.__discord_ui_compiled_template__
        for action in ("mission", "mission_forensics", "mission_bulls_cows", "mission_answer"):
            button = VaultOperationButton(action, 12, "x")
            match = template.fullmatch(button.item.custom_id)
            self.assertEqual((match["action"], match["operation"]), (action, "12"))
            rebuilt = await VaultOperationButton.from_custom_id(None, button.item, match)
            self.assertEqual((rebuilt.action, rebuilt.operation_id, rebuilt.item.custom_id), (action, 12, f"llmcord:vault:op_{action}:12"))
            self.assertIsNone(VaultButton.__discord_ui_compiled_template__.fullmatch(button.item.custom_id))
        self.assertEqual(template.fullmatch("llmcord:vault:op_ask_3:12")["action"], "ask_3")

    async def test_every_mission_button_in_a_reply_is_an_operation_button_without_secrets(self):
        """FEAT-67: picker and Answer buttons use operation ids (operation id only: no member, no code, no answer)."""
        picker, shown = await self.begin(BULLS)
        for click in (picker, shown):
            for custom_id in self.ids_of(click):
                self.assertRegex(custom_id, rf"^llmcord:vault:op_[a-z0-9_]+:{self.oid}$")
        self.assertNotIn(self.truth(), "".join(self.ids_of(picker) + self.ids_of(shown)))

    async def test_no_button_when_the_operation_has_ended(self):
        """FEAT-67: once the operation is not active the status has no view, so there is no Support mission button."""
        self.assertIn("Support mission", self.labels_now())
        self.now += 8 * 86400
        await self.action("refresh")
        await self.play.flush()
        self.assertNotIn("Support mission", self.labels_now())


class PickerTests(MissionCase):
    async def test_the_status_button_offers_both_missions_and_creates_nothing(self):
        """FEAT-67: pressing Support mission without a mission shows an ephemeral picker with exactly the two missions; nothing is stored."""
        click = await self.press("Support mission")
        self.assertTrue(self.gone(click))
        self.assertEqual(self.private(click), PICK)
        self.assertEqual(self.labels_of(click), [FORENSICS, BULLS])
        self.assertEqual(self.missions(), 0)
        self.assertEqual(self.channel.sent[1:], [])

    async def test_choosing_bulls_and_cows_creates_the_persisted_mission_through_the_store(self):
        """FEAT-67: the Bulls and Cows button stores one mission (the store's view is the same afterwards) and shows its message with Answer."""
        _, shown = await self.begin(BULLS)
        self.assertTrue(self.gone(shown))
        self.assertEqual(self.missions(), 1)
        self.assertEqual(self.mission()["mission_type"], "bulls_cows")
        text = self.private(shown)
        self.assertEqual(text.splitlines()[0], f"Support mission: {BULLS}")
        self.assertIn("Guess a code of 3 different digits.", text)
        self.assertIn("Attempts left: 6 of 6", text)
        self.assertEqual(self.labels_of(shown), ["Answer"])
        self.assertEqual(self.store.support_mission(G, ALICE, self.oid, now=self.now)["status"], "active")

    async def test_choosing_forensics_shows_the_records_with_one_hidden_line_and_no_answer(self):
        """FEAT-67: forensics lists every recorded guess with its result, the hidden one as '?', asks for that line and never shows its result."""
        _, shown = await self.begin(FORENSICS)
        puzzle, text = self.puzzle(), self.private(shown)
        self.assertEqual(self.mission()["mission_type"], "forensics")
        self.assertEqual(text.splitlines()[0], f"Support mission: {FORENSICS}")
        self.assertIn("Recorded guesses of one secret code:", text)
        for i, (guess, result) in enumerate(puzzle["records"]):
            if result is None:
                self.assertEqual(puzzle["hidden"], i)
                self.assertIn(f"{i + 1}. {guess} — ?", text)
                self.assertNotIn(f"{i + 1}. {guess} — {fb_en(tuple(self.truth()))}", text)
                self.assertIn(f"What are the strikes and balls of line {i + 1}?", text)
            else:
                self.assertIn(f"{i + 1}. {guess} — {fb_en(tuple(result))}", text)
        self.assertIn("Attempts left: 2 of 2", text)
        self.assertEqual(self.labels_of(shown), ["Answer"])

    async def test_an_existing_mission_is_shown_instead_of_the_picker(self):
        """FEAT-67: with a mission the Support mission button shows it (history, attempts left, Answer) and never the picker; same puzzle."""
        _, shown = await self.begin(BULLS)
        code = self.wrong_code()[0]
        await self.answer(shown, code)
        before = (self.mission()["puzzle_json"], self.mission()["answer_json"])
        again = await self.press("Support mission")
        text = self.private(again)
        self.assertNotIn(PICK, text)
        self.assertEqual(self.labels_of(again), ["Answer"])
        self.assertIn(f"Support mission: {BULLS}", text)
        self.assertIn(f"1. {code} — {fb_en(bulls_cows_feedback(self.truth(), code))}", text)
        self.assertIn("Attempts left: 5 of 6", text)
        self.assertNotIn(self.truth(), text)
        self.assertEqual((self.mission()["puzzle_json"], self.mission()["answer_json"], self.missions()), (*before, 1))

    async def test_a_stale_picker_for_the_other_mission_is_refused_with_the_exists_text(self):
        """FEAT-67: pressing the other type's old picker button says the mission_exists wording, keeps the first mission and is free."""
        picker, _ = await self.begin(FORENSICS)
        other = await self.tap(picker, BULLS)
        self.assertEqual(self.private(other), EXISTS)
        self.assertEqual((self.missions(), self.mission()["mission_type"]), (1, "forensics"))

    async def test_the_same_picker_button_twice_shows_the_same_mission(self):
        """FEAT-67: a double click on a picker button never regenerates: same stored answer, one row."""
        picker, _ = await self.begin(BULLS)
        before = self.mission()["answer_json"]
        await self.tap(picker, BULLS)
        self.assertEqual((self.missions(), self.mission()["answer_json"]), (1, before))

    async def test_each_member_gets_their_own_mission(self):
        """FEAT-67: two members have independent missions of the same operation; Bob never sees Alice's guesses."""
        _, mine = await self.begin(BULLS, ALICE)
        code = self.wrong_code()[0]
        await self.answer(mine, code)
        _, theirs = await self.begin(BULLS, BOB)
        self.assertEqual(self.missions(), 2)
        self.assertIn("Attempts left: 6 of 6", self.private(theirs))
        self.assertNotIn(f"1. {code}", self.private(theirs))


class RefusalTests(MissionCase):
    async def test_a_non_participant_is_told_to_join_first_and_nothing_is_created(self):
        """FEAT-67: participant-only: the button, a picker button and a stale answer modal all say "Join the operation first." for a stranger."""
        _, shown = await self.begin(BULLS, ALICE)
        carol = ops.CAROL
        click = await self.action("mission", carol)
        self.assertEqual(self.private(click), ops.JOIN_FIRST)
        self.assertEqual(self.views(click), [])
        picker = await self.menu(BOB)
        stray = await self.tap(picker, FORENSICS, carol)
        self.assertEqual(self.private(stray), ops.JOIN_FIRST)
        self.assertIsNotNone(OperationMissionModal, MISSING)
        modal = OperationMissionModal(self.oid, "bulls_cows")
        modal.inputs["code"]._value = self.wrong_code()[0]
        submit = self.click(carol)
        await modal.on_submit(submit)
        self.assertEqual(self.private(submit), ops.JOIN_FIRST)
        self.assertEqual((self.missions(), self.used()), (1, 0))
        self.assertTrue(self.gone(click) and self.gone(stray) and self.gone(submit))

    async def test_a_click_from_another_server_is_not_found(self):
        """FEAT-67: guild scoping: an operation id pressed from another guild answers "That vault was not found." and creates nothing."""
        click = self.click(ALICE, guild_id=G + 1)
        await self.play.press(click, "mission", self.oid)
        self.assertEqual(self.private(click), NOT_FOUND)
        self.assertEqual(self.missions(), 0)

    async def test_an_ended_operation_refuses_every_mission_step_in_words(self):
        """FEAT-67: after expiry the button, a picker button and an answer are refused with "This vault has already ended."; nothing changes."""
        picker = await self.menu()
        _, shown = await self.begin(BULLS, BOB)
        code, oid = self.wrong_code(user=BOB)[0], self.oid
        self.now += 8 * 86400
        total = lambda: (self.missions(), self.store.one("SELECT COUNT(*) FROM vault_support_attempts")[0])  # noqa: E731 - the operation id changes with the clock
        before = total()
        pressed = self.click(ALICE)
        await self.play.press(pressed, "mission", oid)
        chosen = await self.tap(picker, FORENSICS)
        late = await self.answer(shown, code, BOB)
        for click in (pressed, chosen, late):
            self.assertEqual(self.private(click), ENDED)
            self.assertTrue(self.gone(click))
        self.assertEqual(total(), before)

    async def test_refusals_use_the_invoker_language(self):
        """FEAT-67: a Korean invoker gets the refusal in Korean, an English one in English (the channel language does not matter)."""
        self.store.set_server_language(G, "en")
        ko = await self.action("mission", ops.CAROL, locale="ko")
        self.assertRegex(self.private(ko), HANGUL)
        self.assertNotEqual(self.private(ko), ops.JOIN_FIRST)
        picker, _ = await self.begin(FORENSICS, ALICE)
        self.store.set_server_language(G, "ko")
        wrong = await self.tap(picker, BULLS, ALICE, locale="en-US")
        self.assertEqual(self.private(wrong), EXISTS)
        korean = await self.tap(picker, BULLS, ALICE, locale="ko")
        self.assertRegex(self.private(korean), HANGUL)


class BullsAndCowsTests(MissionCase):
    async def test_the_modal_has_one_code_input(self):
        """FEAT-67: the Answer button opens OperationMissionModal(bulls_cows) with a single required 'code' input."""
        _, shown = await self.begin(BULLS)
        opened = await self.tap(shown, "Answer")
        modal = opened.response.modals[0]
        self.assertIsInstance(modal, OperationMissionModal, MISSING)
        self.assertEqual(list(modal.inputs), ["code"])
        self.assertTrue(modal.inputs["code"].required)

    async def test_a_wrong_code_shows_strikes_balls_and_attempts_left_but_not_the_secret(self):
        """FEAT-67: feedback is "Answer 1: <code> — <strikes>, <balls>" plus the attempts left; the secret code is not in the reply."""
        _, shown = await self.begin(BULLS)
        code = self.wrong_code()[0]
        click = await self.answer(shown, code)
        text = self.private(click)
        self.assertIn(f"Answer 1: {code} — {fb_en(bulls_cows_feedback(self.truth(), code))}", text)
        self.assertIn("Attempts left: 5 of 6", text)
        self.assertNotIn(self.truth(), text)
        self.assertNotIn("The code was", text)
        self.assertEqual((self.used(), self.mission()["status"]), (1, "active"))

    async def test_invalid_codes_are_free_and_explained_in_words(self):
        """FEAT-67: a short code, a repeated digit and non-digits are refused like invalid guesses (the store's texts): no attempt is used."""
        _, shown = await self.begin(BULLS)
        for text in ("12", "112", "abc", "1234"):
            click = await self.answer(shown, text)
            with self.assertRaises(VaultStoreError) as raised:
                self.store.submit_support_answer(G, ALICE, self.oid, text, now=self.now)
            self.assertEqual(self.private(click), i18n.english().error(raised.exception), text)
            self.assertNotIn("Attempts left", self.private(click))
        self.assertEqual((self.used(), self.mission()["status"]), (0, "active"))

    async def test_a_repeated_code_is_free_and_explained(self):
        """FEAT-67: sending the same code twice costs one attempt only and explains the repeat."""
        _, shown = await self.begin(BULLS)
        code = self.wrong_code()[0]
        await self.answer(shown, code)
        again = await self.answer(shown, code)
        self.assertEqual(self.used(), 1)
        self.assertIn("already", self.private(again).lower())
        self.assertNotIn("Attempts left", self.private(again))

    async def test_solving_pays_the_reward_once_and_says_so(self):
        """FEAT-67: with support_reward 7 the solve reveals the code, shows "Reward: 7", credits the member once and closes the mission."""
        self.store.set_vault_operation_settings(G, support_reward=7)
        before = self.money()
        click = await self.solve_bulls()
        text = self.private(click)
        self.assertIn(SOLVED, text)
        self.assertIn(f"The code was {self.truth()}.", text)
        self.assertIn("Reward: 7", text)
        self.assertNotIn(NO_REWARD, text)
        self.assertEqual(self.money() - before, 7)
        self.assertEqual(self.mission()["status"], "solved")
        self.assertEqual(self.views(click), [])
        reopened = await self.press("Support mission")
        self.assertIn(SOLVED, self.private(reopened))
        self.assertEqual(self.views(reopened), [])
        self.assertEqual(self.money() - before, 7)

    async def test_solving_with_no_reward_says_nothing_was_paid(self):
        """FEAT-67: a solve that pays nothing (support_reward 0, the default) tells the member so and shows no Reward line."""
        before = self.money()
        text = self.private(await self.solve_bulls())
        self.assertIn(SOLVED, text)
        self.assertIn(NO_REWARD, text)
        self.assertNotIn("Reward:", text)
        self.assertEqual(self.money(), before)

    async def test_a_reward_refused_by_the_balance_cap_says_so_instead_of_nothing_was_paid(self):
        """MNT-91: a support reward the balance cap refuses is reported as such, not as "No reward was paid"."""
        self.store.set_vault_operation_settings(G, support_reward=7)
        self.store.db.execute("INSERT INTO currency_balances(guild_id,user_id,balance,updated_at) VALUES(?,?,?,?)", (G, ALICE, self.store.MAX_CURRENCY_BALANCE - 3, 0))
        self.store.db.commit()
        text = self.private(await self.solve_bulls())
        self.assertIn(SOLVED, text)
        self.assertIn("Your reward could not be paid because your balance is at its limit.", text)
        self.assertNotIn(NO_REWARD, text)
        self.assertNotIn("Reward: 7", text)

    async def test_running_out_of_attempts_fails_for_good_and_reveals_the_code(self):
        """FEAT-67: six wrong codes fail the mission; only the last reply reveals the code; reopening shows it failed with no Answer button."""
        _, shown = await self.begin(BULLS)
        replies = [self.private(await self.answer(shown, code)) for code in self.wrong_code(6)]
        for early in replies[:5]:
            self.assertNotIn(self.truth(), early)
            self.assertNotIn(FAILED, early)
        self.assertIn(FAILED, replies[5])
        self.assertIn(f"The code was {self.truth()}.", replies[5])
        self.assertNotIn("Reward:", replies[5])
        self.assertEqual(self.mission()["status"], "failed")
        reopened = await self.press("Support mission")
        self.assertIn(f"The code was {self.truth()}.", self.private(reopened))
        self.assertEqual(self.labels_of(reopened), [])
        late = await self.answer(shown, self.truth())
        self.assertEqual((self.used(), self.mission()["status"]), (6, "failed"))
        self.assertIn("over", self.private(late))

    async def test_the_secret_is_in_no_active_message(self):
        """FEAT-67: while the mission is active neither the picker, the mission message, the reopened message, an answer reply nor the shared
        status (even after a flush) holds the secret code."""
        picker, shown = await self.begin(BULLS)
        answered = await self.answer(shown, self.wrong_code()[0])
        reopened = await self.press("Support mission")
        await self.play.flush()
        secret = self.truth()
        for click in (picker, shown, answered, reopened):
            self.assertNotIn(secret, self.private(click) + "".join(self.ids_of(click)))
        self.assertNotIn(secret, self.text())
        self.assertNotIn(secret, "".join(self.announcements()))

    async def test_the_same_interaction_twice_counts_one_attempt_and_pays_once(self):
        """FEAT-67: a double-delivered modal submit (same interaction id) records one attempt, answers the same feedback and pays one reward."""
        self.store.set_vault_operation_settings(G, support_reward=4)
        _, shown = await self.begin(BULLS)
        code = self.wrong_code()[0]
        first = await self.answer(shown, code, ident=555)
        second = await self.answer(shown, code, ident=555)
        self.assertEqual(self.used(), 1)
        self.assertIn(fb_en(bulls_cows_feedback(self.truth(), code)), self.private(second))
        self.assertEqual(self.private(first).splitlines()[:1], self.private(second).splitlines()[:1])
        before = self.money()
        win = await self.answer(shown, self.truth(), ident=556)
        again = await self.answer(shown, self.truth(), ident=556)
        self.assertEqual(self.money() - before, 4)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_operation_claims WHERE type='support'")[0], 1)
        self.assertIn("Reward: 4", self.private(win))
        self.assertIn(SOLVED, self.private(again))

    async def test_a_mission_spends_no_shared_turn_and_does_not_edit_the_channel(self):
        """FEAT-67: missions never touch the stage: turns left, guesses and the channel messages stay as they were."""
        stage = self.op()["stage"]
        _, shown = await self.begin(BULLS)
        await self.answer(shown, self.wrong_code()[0])
        await self.play.flush()
        after = self.op()["stage"]
        self.assertEqual((after["turns_left"], after["guesses"]), (stage["turns_left"], stage["guesses"]))
        self.assertEqual(self.announcements(), [])
        self.assertEqual(self.secret(), ops.SECRET)


class ForensicsTests(MissionCase):
    async def test_the_modal_has_strikes_and_balls_inputs(self):
        """FEAT-67: the forensics Answer modal has two required inputs, 'strikes' and 'balls'."""
        _, shown = await self.begin(FORENSICS)
        modal = (await self.tap(shown, "Answer")).response.modals[0]
        self.assertIsInstance(modal, OperationMissionModal, MISSING)
        self.assertEqual(list(modal.inputs), ["strikes", "balls"])
        self.assertTrue(all(i.required for i in modal.inputs.values()))

    async def test_a_wrong_pair_costs_one_attempt_and_does_not_reveal_the_hidden_result(self):
        """FEAT-67: a wrong (but legal) pair answers "Answer 1: <feedback>" and "Attempts left: 1 of 2", without the true result or the answer line."""
        _, shown = await self.begin(FORENSICS)
        s, b = self.wrong_pair()
        click = await self.answer(shown, {"strikes": str(s), "balls": str(b)})
        text = self.private(click)
        self.assertIn(f"Answer 1: {fb_en((s, b))}", text)
        self.assertIn("Attempts left: 1 of 2", text)
        self.assertNotIn(fb_en(tuple(self.truth())), text)
        self.assertNotIn("The hidden result", text)
        self.assertEqual(self.used(), 1)

    async def test_invalid_pairs_are_free_and_explained(self):
        """FEAT-67: text, negatives, a sum over 4, and 3 strikes with 1 ball are refused with the invalid-answer text and cost nothing."""
        _, shown = await self.begin(FORENSICS)
        for strikes, balls in (("x", "1"), ("1", ""), ("-1", "2"), ("3", "3"), ("3", "1"), ("1.5", "0"), ("5", "0")):
            click = await self.answer(shown, {"strikes": strikes, "balls": balls})
            self.assertEqual(self.private(click), INVALID_FORENSICS, (strikes, balls))
        self.assertEqual((self.used(), self.mission()["status"]), (0, "active"))

    async def test_solving_reveals_the_result_and_pays(self):
        """FEAT-67: the right pair solves it: "The hidden result was <feedback>." plus "Reward: 3" and one credit."""
        self.store.set_vault_operation_settings(G, support_reward=3)
        before = self.money()
        text = self.private(await self.solve_forensics())
        self.assertIn(SOLVED, text)
        self.assertIn(f"The hidden result was {fb_en(tuple(self.truth()))}.", text)
        self.assertIn("Reward: 3", text)
        self.assertEqual(self.money() - before, 3)

    async def test_two_wrong_pairs_fail_and_reveal_only_then(self):
        """FEAT-67: the second wrong answer fails the mission and reveals the hidden result; the first reply does not."""
        _, shown = await self.begin(FORENSICS)
        truth = tuple(self.truth())
        wrongs = [(s, b) for s in range(5) for b in range(5 - s) if (s, b) != truth and (s, b) != (3, 1)][:2]
        first = self.private(await self.answer(shown, {"strikes": str(wrongs[0][0]), "balls": str(wrongs[0][1])}))
        second = self.private(await self.answer(shown, {"strikes": str(wrongs[1][0]), "balls": str(wrongs[1][1])}))
        self.assertNotIn("The hidden result", first)
        self.assertIn(FAILED, second)
        self.assertIn(f"The hidden result was {fb_en(truth)}.", second)
        self.assertEqual(self.mission()["status"], "failed")
        reopened = await self.press("Support mission")
        self.assertEqual(self.labels_of(reopened), [])
        self.assertIn(f"The hidden result was {fb_en(truth)}.", self.private(reopened))

    async def test_the_same_interaction_twice_counts_one_attempt(self):
        """FEAT-67: re-delivery of the same modal submit costs one of the two attempts only."""
        _, shown = await self.begin(FORENSICS)
        s, b = self.wrong_pair()
        values = {"strikes": str(s), "balls": str(b)}
        await self.answer(shown, values, ident=777)
        again = await self.answer(shown, values, ident=777)
        self.assertEqual((self.used(), self.mission()["status"]), (1, "active"))
        self.assertIn("Attempts left: 1 of 2", self.private(again))


class KoreanTests(MissionCase):
    async def test_a_korean_member_gets_korean_picker_mission_and_results(self):
        """FEAT-67: with a Korean invoker the picker, buttons, mission message, wrong answer, win and reward line are Korean (hand written
        strings; 작전 and 금고 are the glossary words, the secret still never shows before the end)."""
        self.store.set_vault_operation_settings(G, support_reward=1234)
        picker = await self.menu(locale="ko")
        for text in [self.private(picker), *self.labels_of(picker)]:
            self.assertRegex(text, HANGUL)
        shown = await self.tap(picker, self.labels_of(picker)[1], locale="ko")
        self.assertEqual(self.mission()["mission_type"], "bulls_cows")
        for text in [self.private(shown), *self.labels_of(shown)]:
            self.assertRegex(text, HANGUL)
        wrong = await self.answer(shown, self.wrong_code()[0], locale="ko")
        self.assertRegex(self.private(wrong), HANGUL)
        self.assertNotIn(self.truth(), self.private(wrong))
        win = await self.answer(shown, self.truth(), locale="ko")
        text = self.private(win)
        self.assertRegex(text, HANGUL)
        self.assertNotIn(SOLVED, text)
        self.assertNotIn("Reward:", text)
        self.assertIn(self.truth(), text)
        self.assertIn("1234", text)

    async def test_the_status_button_label_is_korean_on_a_korean_server(self):
        """FEAT-67: on a Korean server the status has a Korean mission button label (not the English one)."""
        self.store.set_server_language(G, "ko")
        await self.action("refresh", locale="ko")
        await self.play.flush()
        labels = self.labels_now()
        self.assertNotIn("Support mission", labels)
        self.assertEqual(len(labels), 11)
        self.assertTrue(HANGUL.search(labels[4]), labels)

    async def test_korean_forensics_invalid_answer_is_a_korean_message(self):
        """FEAT-67: invalid forensics input is refused in Korean for a Korean invoker, with no attempt used."""
        picker = await self.menu(locale="ko")
        shown = await self.tap(picker, self.labels_of(picker)[0], locale="ko")
        click = await self.answer(shown, {"strikes": "9", "balls": "9"}, locale="ko")
        self.assertRegex(self.private(click), HANGUL)
        self.assertNotEqual(self.private(click), INVALID_FORENSICS)
        self.assertEqual(self.used(), 0)

