"""Vault tools on Discord (llmcord_core/vault_discord.py), FEAT-64 phase B, decision D43. Written before the implementation.

The store side (``VaultStore.use_vault_tool``, phase A, tests/test_vault_tools_store.py) is the source of truth; this layer adds the
button, the picker, the modal, the words and the refusals. Everything is private to the member (invoker_t), like the rest of
``/vault personal``. Fake clicks and an in-memory store only; the clock is ``bot.vault.clock``.

CONTRACT (extends the module contract in tests/test_vault_discord.py)

  Status message
    * ``render_buttons(view, *, t=None, tools_held=0)``: while the vault is active and ``tools_held`` > 0 a "Tools" button is added after
      "Guess" and "Reveal clue" (a ``VaultButton`` with action ``tools``). Not shown otherwise (no tool held, or the vault has ended).
      ``tools_held`` is the number of tool units the member holds (``store.item_qty`` over ``vault.TOOLS``), computed by the caller.
      ``render_main(view, *, t=None, tools_held=0)`` adds, after the four fixed first lines, ``"Tools used: {n} of 2"`` only when
      ``view["tools_used"] > 0`` or ``tools_held > 0`` (a member with no tools and none used sees no tool line at all), and, once a tool
      was used, ``"Tool results:"`` followed by one ``"{Tool name}: {fact}"`` line per use, oldest first (the facts are public).
  Picker (the same status message, edited in place, one ``response.edit_message``)
    * ``press(interaction, "tools", session_id)`` edits the status message: same text, view = one button per HELD tool in
      ``vault.TOOLS`` order plus "Back". Label ``"{Tool name} ×{qty} ({cost} turn)"`` / ``"({cost} turns)"``, e.g. ``"Spectrum Analyzer ×2
      (1 turn)"``, ``"Master Decoder ×1 (2 turns)"``. Tools that are not held are not listed. The message text is the status text;
      the picker adds no separate message. Custom ids: ``llmcord:vault:use_<tool key>:<session id>`` (a ``VaultButton`` action
      ``use_vault_spectrum_analyzer`` ...), "Back" is action ``back`` (edits the status message back to the normal buttons).
    * ``press(interaction, "use_<tool key>", session_id)`` opens a ``VaultToolModal(session_id, tool_key, *, t=None)`` through
      ``response.send_modal`` (nothing is posted or edited). ``modal.inputs`` is a dict of ``TextInput`` keyed by engine param name:
      scanner ``guess`` + ``digit``; spectrum ``position``; comparison ``a`` + ``b``; verifier ``guess``; decoder ``position``.
      ``guess`` is the NUMBER of one of the member's past guesses (the "1" of "1. 1530 — ..."); a 4-digit code that is one of them is
      accepted too. Anything else is passed through unchanged so the store refuses it (``guess_not_in_history``). A ``position`` that is
      not a whole number is passed through as text (store: ``bad_position``). An unknown ``use_`` key is answered ``vault.tool_unknown``.
      ``on_submit`` calls ``interaction.client.vault.submit_tool(interaction, session_id, tool_key, values)``.
    * Every one of these presses is owner-only: another member gets "This vault belongs to someone else. Run /vault personal to open
      your own." (ephemeral, nothing edited, no modal), as for Guess.
  Activation
    * ``await submit_tool(interaction, session_id, tool_key, values: dict[str, str])`` calls ``store.use_vault_tool(..., action_key=
      str(interaction.id), now=clock())``. On success: ONE ``response.edit_message`` of the status (new turns, "Tools used", tool results,
      buttons recomputed from the inventory left) and ONE ephemeral followup ``"{Tool name}: {fact}"`` (+ ``"Out of turns. The vault stays
      closed today."`` when the tool spent the last turn). A replayed interaction id edits and logs the same way and consumes nothing.
    * Facts (English): scanner "The code contains the digit {digit}." / "The code does not contain the digit {digit}."; spectrum
      "Digit {position} is high (5 to 9)." / "Digit {position} is low (0 to 4)."; comparison "Of the digits {a} and {b}, none is in the
      code." / "..., exactly one is in the code." / "..., both are in the code."; verifier "In guess {guess}, digit {position} is {digit}."
      / "Guess {guess} has no digit in the right position."; decoder "Digit {position} is {digit}."
    * Refusals (store codes) are sent with ``response.send_message(..., ephemeral=True)``, NO edit, nothing consumed: ``t.error(error)``
      (the English code text). ``vault.tool_bad_params`` is that text followed by a space and a reason sentence: bad_position "Choose a
      position from 1 to 4." / bad_digit "Choose a single digit from 0 to 9." / digits_equal "Choose two different digits." /
      guess_not_in_history "Choose one of your own guesses." / digit_not_in_guess "That digit is not in that guess." / nothing_new
      "That guess has no new strike to show." / already_revealed "That position is already revealed." ``vault.not_active`` refreshes the
      status with an edit (like a guess) and then sends the error.
  Korean: glossary 금고 도구; tool names are the recipe names (잔류 스캐너, 스펙트럼 분석기, 비교 회로, 위치 검증기, 마스터 디코더);
      the picker/status strings are hand-translated; the picker's lead text is the English "Choose a Vault tool." and rides on the
      "Back" status text only through the buttons, so the Korean smoke asserts the labels and replies.
"""
import re
import unittest

import discord

from llmcord_core.vault_discord import VaultButton

from test_vault_discord import G, ALICE, BOB, DAY, NOT_YOURS, VaultCase, item_of

SPECTRUM, DECODER, SCANNER, COMPARE, VERIFIER = ("vault_spectrum_analyzer", "vault_master_decoder", "vault_residual_scanner",
                                                 "vault_comparison_circuit", "vault_position_verifier")
NAMES = {SPECTRUM: "Spectrum Analyzer", DECODER: "Master Decoder", SCANNER: "Residual Scanner", COMPARE: "Comparison Circuit", VERIFIER: "Position Verifier"}
KO_NAMES = {SPECTRUM: "스펙트럼 분석기", DECODER: "마스터 디코더", SCANNER: "잔류 스캐너", COMPARE: "비교 회로", VERIFIER: "위치 검증기"}
HANGUL = re.compile("[가-힣]")
SECRET = "1357"
BAD = "That tool cannot be used that way. Nothing was spent."
REASONS = {"bad_position": "Choose a position from 1 to 4.", "bad_digit": "Choose a single digit from 0 to 9.", "digits_equal": "Choose two different digits.",
           "guess_not_in_history": "Choose one of your own guesses.", "digit_not_in_guess": "That digit is not in that guess.",
           "nothing_new": "That guess has no new strike to show.", "already_revealed": "That position is already revealed."}


class ToolUiCase(VaultCase):
    def give(self, key, qty=1, user=ALICE):
        self.store.db.execute("INSERT INTO inventory(guild_id,user_id,item_key,qty) VALUES(?,?,?,?) ON CONFLICT(guild_id,user_id,item_key) DO UPDATE SET qty=excluded.qty",
                              (G, user, key, qty))
        self.store.db.commit()

    def qty(self, key, user=ALICE):
        return self.store.item_qty(G, user, key)

    def fix_secret(self, code=SECRET, user=ALICE):
        self.store.db.execute("UPDATE vault_sessions SET secret=? WHERE guild_id=? AND user_id=?", (code, G, user))
        self.store.db.commit()

    async def started(self, *tools, locale=None, **kw):
        """A fresh vault with the secret pinned to SECRET and the given tools in the inventory (one each, or (key, qty))."""
        first = await self.personal(locale=locale, **kw)
        self.fix_secret()
        for tool in tools:
            key, qty = tool if isinstance(tool, tuple) else (tool, 1)
            self.give(key, qty)
        return await self.personal(locale=locale) if tools else first

    def play(self, *codes, user=ALICE):
        sid = self.row(user)["id"]
        for code in codes:
            self.store.submit_vault_guess(G, user, sid, code, now=self.now)

    async def use(self, source, tool, values, *, user=ALICE, ident=None, **kw):
        """Tools button -> picker -> tool button -> modal -> submit. Returns (submit click, picker click, modal)."""
        picker = await self.press(source, "Tools", user, **kw) if not self.picker_open(source) else source
        pick = await self.press(picker, self.tool_label(picker, tool), user, **kw)
        modal = pick.response.modals[0]
        for name, value in values.items():
            modal.inputs[name]._value = value
        submit = self.click(user, ident=ident, **kw)
        await modal.on_submit(submit)
        return submit, picker, modal

    def picker_open(self, source):
        return "Back" in self.labels(source) if hasattr(source, "response") else False

    def tool_label(self, picker, tool):
        for child in self.buttons(self.view_of(picker)):
            label = item_of(child).label
            if label.startswith(NAMES[tool]) or label.startswith(KO_NAMES[tool]):
                return label
        raise AssertionError(f"{tool} is not in the picker: {self.labels(picker)}")

    def untouched(self, click):
        self.assertEqual(click.response.edited, [])
        self.assertEqual(click.response.modals, [])


class ToolButtonTests(ToolUiCase):
    async def test_no_tools_means_no_tools_button_and_no_tool_line(self):
        """FEAT-64: a member holding no tool sees the old status: Guess only, no "Tools" button, no "Tools used" line."""
        start = await self.personal()
        self.assertEqual(self.labels(start), ["Guess"])
        self.assertNotIn("Tools used", self.status(start))
        self.assertNotIn("Tool results", self.status(start))

    async def test_holding_a_tool_adds_the_button_and_the_remaining_count(self):
        """FEAT-64: with one tool in the inventory the status shows Guess and Tools, and "Tools used: 0 of 2" (max two per vault)."""
        shown = await self.started(SPECTRUM)
        self.assertEqual(self.labels(shown), ["Guess", "Tools"])
        self.assertIn("Tools used: 0 of 2", self.status(shown).splitlines())
        self.assertNotIn("Tool results:", self.status(shown))
        partners = await self.personal(BOB, partner=None)
        self.assertEqual(self.labels(partners), ["Guess"])  # another member's inventory does not count

    async def test_the_button_follows_the_clue_button(self):
        """FEAT-64: with a partner and a tool the order is Guess, Reveal clue, Tools."""
        await self.personal(partner="Ann")
        self.give(SPECTRUM)
        again = await self.personal()
        self.assertEqual(self.labels(again), ["Guess", "Reveal clue", "Tools", "Ask Ann"])

    async def test_an_ended_vault_has_no_tools_button(self):
        """FEAT-64: once the vault is solved or failed there are no buttons, tools or not."""
        start = await self.started(SPECTRUM)
        win = await self.submit(start, SECRET)
        self.assertEqual(self.labels(win), [])
        self.assertEqual(self.qty(SPECTRUM), 1)

    async def test_the_picker_lists_only_held_tools_with_count_and_cost(self):
        """FEAT-64: the Tools button edits the status message once into one button per held tool (label: name, count, turn cost) and Back."""
        start = await self.started((SPECTRUM, 2), DECODER)
        picker = await self.press(start, "Tools")
        self.assertEqual(len(picker.response.edited), 1)
        self.assertEqual(self.labels(picker), ["Spectrum Analyzer ×2 (1 turn)", "Master Decoder ×1 (2 turns)", "Back"])
        self.assertEqual(picker.response.edited[0]["content"], self.status(start))
        self.assertTrue(self.ephemeral(picker))
        self.assertEqual((self.channel.sent, picker.response.modals), ([], []))

    async def test_the_picker_hides_tools_the_member_does_not_hold(self):
        """FEAT-64: each tool appears only when held; a used-up tool disappears."""
        start = await self.started(SCANNER, COMPARE)
        picker = await self.press(start, "Tools")
        self.assertEqual(self.labels(picker), ["Residual Scanner ×1 (1 turn)", "Comparison Circuit ×1 (1 turn)", "Back"])

    async def test_back_returns_to_the_normal_buttons(self):
        """FEAT-64: Back edits the status message back to Guess and Tools; nothing is spent."""
        start = await self.started(SPECTRUM)
        picker = await self.press(start, "Tools")
        back = await self.press(picker, "Back")
        self.assertEqual(self.labels(back), ["Guess", "Tools"])
        self.assertEqual(len(back.response.edited), 1)
        self.assertEqual((self.qty(SPECTRUM), self.store.one("SELECT COUNT(*) FROM vault_tool_uses")[0]), (1, 0))

    async def test_a_tool_button_opens_a_modal_with_the_right_inputs(self):
        """FEAT-64: each tool opens a modal (nothing posted or edited) whose inputs are named after the engine params."""
        expected = {SCANNER: {"guess", "digit"}, SPECTRUM: {"position"}, COMPARE: {"a", "b"}, VERIFIER: {"guess"}, DECODER: {"position"}}
        start = await self.started(*expected)
        for tool, names in expected.items():
            with self.subTest(tool=tool):
                picker = await self.press(start, "Tools")
                opened = await self.press(picker, self.tool_label(picker, tool))
                self.assertEqual(len(opened.response.modals), 1)
                modal = opened.response.modals[0]
                self.assertEqual(set(modal.inputs), names)
                self.assertTrue(all(isinstance(i, discord.ui.TextInput) and i.required for i in modal.inputs.values()))
                self.assertEqual((opened.response.sent, opened.response.edited), ([], []))


class ToolResultTests(ToolUiCase):
    async def check(self, tool, values, fact, *, guesses=(), cost=1, **start_kw):
        start = await self.started(tool, **start_kw)
        self.play(*guesses)
        done, _, _ = await self.use(start, tool, values)
        self.assertEqual(len(done.response.edited), 1)
        self.assertIn(f"{NAMES[tool]}: {fact}", self.log(done).splitlines())
        self.assertEqual(len(done.followup.sent), 1)
        self.assertTrue(self.ephemeral(done))
        text = self.status(done)
        self.assertIn(f"Turns left: {9 - len(guesses) - cost} of 9", text)
        self.assertIn("Tools used: 1 of 2", text.splitlines())
        self.assertIn("Tool results:", text.splitlines())
        self.assertIn(f"{NAMES[tool]}: {fact}", text.splitlines())
        self.assertEqual(self.qty(tool), 0)
        self.assertEqual(self.channel.sent, [])
        self.assert_no_secret(done)
        return done

    async def test_scanner_present(self):
        """FEAT-64: scanner on guess 1 ("1530"), digit 5: the code contains it."""
        await self.check(SCANNER, {"guess": "1", "digit": "5"}, "The code contains the digit 5.", guesses=("1530",))

    async def test_scanner_absent(self):
        """FEAT-64: scanner, digit 0 of guess 1: the code does not contain it."""
        await self.check(SCANNER, {"guess": "1", "digit": "0"}, "The code does not contain the digit 0.", guesses=("1530",))

    async def test_scanner_accepts_the_guess_code_as_well_as_its_number(self):
        """FEAT-64: the guess input also takes the 4-digit code of a past guess."""
        await self.check(SCANNER, {"guess": "1530", "digit": "5"}, "The code contains the digit 5.", guesses=("1530",))

    async def test_spectrum_high_and_low(self):
        """FEAT-64: spectrum on position 3 (digit 5) is high; position 1 (digit 1) is low."""
        await self.check(SPECTRUM, {"position": "3"}, "Digit 3 is high (5 to 9).")
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_tool_uses")[0], 1)

    async def test_spectrum_low(self):
        await self.check(SPECTRUM, {"position": "1"}, "Digit 1 is low (0 to 4).")

    async def test_comparison_zero_one_and_two(self):
        """FEAT-64: comparison reads none / exactly one / both, per how many of the two digits are in 1357."""
        for a, b, fact in (("0", "2", "none is in the code."), ("1", "9", "exactly one is in the code."), ("1", "3", "both are in the code.")):
            with self.subTest(a=a, b=b):
                self.now += DAY  # a new local day is a new vault
                await self.check(COMPARE, {"a": a, "b": b}, f"Of the digits {a} and {b}, {fact}")

    async def test_verifier_with_a_strike(self):
        """FEAT-64: verifier on guess 1 ("1358") discloses the first right position and digit."""
        await self.check(VERIFIER, {"guess": "1"}, "In guess 1358, digit 1 is 1.", guesses=("1358",))

    async def test_verifier_without_a_strike(self):
        """FEAT-64: with Total Match feedback (matches only) a guess whose digits are all misplaced yields the "no digit" fact."""
        self.store.set_vault_settings(G, protocols=["standard", "total_match"])
        await self.check(VERIFIER, {"guess": "1"}, "Guess 5713 has no digit in the right position.", guesses=("5713",), protocol="total_match")

    async def test_decoder_costs_two_turns(self):
        """FEAT-64: decoder on position 3 reveals the digit and spends two turns."""
        await self.check(DECODER, {"position": "3"}, "Digit 3 is 5.", cost=2)

    async def test_the_status_keeps_the_results_and_the_new_buttons(self):
        """FEAT-64: buttons are recomputed from what is left: a second unit keeps Tools, the last unit removes it."""
        start = await self.started((SPECTRUM, 2))
        first, picker, _ = await self.use(start, SPECTRUM, {"position": "1"})
        self.assertEqual(self.labels(first), ["Guess", "Tools"])
        self.assertEqual(self.qty(SPECTRUM), 1)
        second, _, _ = await self.use(first, SPECTRUM, {"position": "2"})
        self.assertEqual(self.labels(second), ["Guess"])
        self.assertIn("Tools used: 2 of 2", self.status(second).splitlines())
        self.assertEqual(self.status(second).count("Spectrum Analyzer: "), 2)

    async def test_the_action_key_is_the_interaction_id(self):
        """FEAT-64: the store records str(interaction.id) as the action key."""
        start = await self.started(SPECTRUM)
        await self.use(start, SPECTRUM, {"position": "1"}, ident=5150)
        self.assertEqual(self.store.one("SELECT action_key FROM vault_tool_uses")["action_key"], "5150")

    async def test_a_double_submitted_interaction_spends_once(self):
        """FEAT-64: the same interaction id twice: one use, one unit, one turn; the replay shows the same status and result."""
        start = await self.started(SPECTRUM)
        first, picker, modal = await self.use(start, SPECTRUM, {"position": "1"}, ident=777)
        replay = self.click(ident=777)
        await modal.on_submit(replay)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_tool_uses")[0], 1)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM item_uses")[0], 1)
        self.assertEqual(self.status(first), self.status(replay))
        self.assertIn("Turns left: 8 of 9", self.status(replay))
        self.assertEqual(self.log(first), self.log(replay))

    async def test_the_secret_is_never_in_an_active_state(self):
        """FEAT-64: the full code appears only once the vault ended, never in the status, picker or logs of an active vault with tools."""
        start = await self.started((DECODER, 1), (SPECTRUM, 1))
        await self.press(start, "Tools")
        done, _, _ = await self.use(start, DECODER, {"position": "1"})
        self.assertNotIn(SECRET, self.status(done) + self.log(done))
        done2, _, _ = await self.use(done, SPECTRUM, {"position": "2"})
        self.assert_no_secret(done, done2)
        self.assertNotIn("Secret:", self.status(done2))


class ToolRefusalTests(ToolUiCase):
    async def refuse(self, tool, values, message, *, start=None, guesses=(), source=None, **kw):
        start = start or await self.started(tool)
        self.play(*guesses)
        before = (self.qty(tool), self.store.one("SELECT COUNT(*) FROM vault_tool_uses")[0], self.store.one("SELECT COUNT(*) FROM item_uses")[0],
                  self.store.one("SELECT COUNT(*) FROM vault_actions")[0])
        done, _, _ = await self.use(source or start, tool, values, **kw)
        self.assertIn(message, self.everything(done))
        self.assertEqual(done.response.edited, [])
        self.assertTrue(self.ephemeral(done))
        self.assertEqual(before, (self.qty(tool), self.store.one("SELECT COUNT(*) FROM vault_tool_uses")[0], self.store.one("SELECT COUNT(*) FROM item_uses")[0],
                                  self.store.one("SELECT COUNT(*) FROM vault_actions")[0]))
        return done

    async def test_bad_params_say_why_and_cost_nothing(self):
        """FEAT-64: every engine reason has its own sentence after the base text; nothing is consumed or edited."""
        cases = [
            (SPECTRUM, {"position": "9"}, "bad_position", ()),
            (SPECTRUM, {"position": "abc"}, "bad_position", ()),
            (COMPARE, {"a": "ab", "b": "3"}, "bad_digit", ()),
            (COMPARE, {"a": "3", "b": "3"}, "digits_equal", ()),
            (SCANNER, {"guess": "7", "digit": "5"}, "guess_not_in_history", ("1530",)),
            (VERIFIER, {"guess": "2"}, "guess_not_in_history", ("1530",)),
            (SCANNER, {"guess": "1", "digit": "9"}, "digit_not_in_guess", ("1530",)),
        ]
        for tool, values, reason, guesses in cases:
            with self.subTest(tool=tool, values=values):
                self.now += DAY  # a new local day is a new vault
                done = await self.refuse(tool, values, BAD, guesses=guesses)
                self.assertIn(REASONS[reason], self.everything(done))

    async def test_already_revealed_and_nothing_new(self):
        """FEAT-64: decoding a position twice is "already revealed"; a verifier whose only strike was disclosed is "nothing new"."""
        start = await self.started((DECODER, 2))
        self.play("1946")
        first, _, _ = await self.use(start, DECODER, {"position": "1"})
        again = await self.refuse(DECODER, {"position": "1"}, BAD, start=first, source=first)
        self.assertIn(REASONS["already_revealed"], self.everything(again))
        self.give(VERIFIER)
        await self.refuse(VERIFIER, {"guess": "1"}, "That would tell you nothing new. Nothing was spent.", start=first, source=await self.personal())

    async def test_redundant_is_free_and_explained(self):
        """FEAT-64: after the decoder disclosed position 1, a spectrum on position 1 would teach nothing: refused, nothing spent."""
        start = await self.started(DECODER, SPECTRUM)
        first, _, _ = await self.use(start, DECODER, {"position": "1"})
        done = await self.refuse(SPECTRUM, {"position": "1"}, "That would tell you nothing new. Nothing was spent.", start=first, source=first)
        self.assertEqual(self.qty(SPECTRUM), 1)
        self.assertNotIn("Spectrum Analyzer:", self.status(first))
        self.assertTrue(done.response.sent)

    async def test_at_most_two_tools_per_vault(self):
        """FEAT-64: the third activation is refused with the limit text; the item stays."""
        start = await self.started((SPECTRUM, 3))
        first, _, _ = await self.use(start, SPECTRUM, {"position": "1"})
        second, _, _ = await self.use(first, SPECTRUM, {"position": "2"})
        self.assertEqual(self.qty(SPECTRUM), 1)
        await self.refuse(SPECTRUM, {"position": "3"}, "You can use at most 2 tools in one vault.", start=second, source=second)
        self.assertEqual(self.qty(SPECTRUM), 1)

    async def test_the_decoder_needs_three_turns(self):
        """FEAT-64: with two turns left the master decoder is refused ("not enough turns"), free."""
        await self.started(DECODER)
        self.play(*[c for c in ("0246", "0268", "0289", "0246"[::-1], "2460", "4602", "6024")][:7])
        shown = await self.personal()
        await self.refuse(DECODER, {"position": "1"}, "You do not have enough turns left for that tool.", start=shown, source=shown)

    async def test_a_tool_that_is_no_longer_held_is_refused(self):
        """FEAT-64: the modal was opened, then the tool left the inventory: the submit says so (and points to /shop); nothing happens."""
        start = await self.started(SPECTRUM)
        picker = await self.press(start, "Tools")
        pick = await self.press(picker, self.tool_label(picker, SPECTRUM))
        modal = pick.response.modals[0]
        modal.inputs["position"]._value = "1"
        self.store.db.execute("DELETE FROM inventory")
        self.store.db.commit()
        submit = self.click()
        await modal.on_submit(submit)
        self.assertIn("You do not have that tool. Buy one with /shop.", self.everything(submit))
        self.assertEqual(submit.response.edited, [])
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_tool_uses")[0], 0)

    async def test_an_unknown_tool_is_refused(self):
        """FEAT-64: a tool key that is not a Vault tool (stale or forged custom id) gets the unknown text, no modal, no edit."""
        await self.started(SPECTRUM)
        sid = self.row()["id"]
        click = self.click()
        await self.bot.vault.press(click, "use_vault_bogus", sid)
        self.assertEqual(self.everything(click), "That is not a Vault tool.")
        self.untouched(click)
        direct = self.click()
        await self.bot.vault.submit_tool(direct, sid, "vault_bogus", {})
        self.assertEqual(self.everything(direct), "That is not a Vault tool.")
        self.assertEqual(direct.response.edited, [])
        self.assertEqual(self.qty(SPECTRUM), 1)

    async def test_the_last_turn_spent_on_a_tool_fails_the_vault(self):
        """FEAT-64: a tool that uses the last turn ends the vault as failed: out-of-turns log, secret and no buttons in the status."""
        await self.started(SPECTRUM)
        self.play("0246", "0268", "0289", "2460", "4602", "6024", "2046", "4026")
        shown = await self.personal()
        done, _, _ = await self.use(shown, SPECTRUM, {"position": "3"})
        self.assertIn("Out of turns. The vault stays closed today.", self.log(done))
        self.assertIn("Spectrum Analyzer: Digit 3 is high (5 to 9).", self.log(done))
        self.assertIn("Status: Failed", self.status(done))
        self.assertIn(f"Secret: {SECRET}", self.status(done).splitlines())
        self.assertEqual(self.labels(done), [])
        self.assertEqual(self.qty(SPECTRUM), 0)

    async def test_an_ended_vault_refreshes_the_status(self):
        """FEAT-64: a tool submit after the day ended shows the Expired status and the ended text; the item is kept."""
        start = await self.started(SPECTRUM)
        picker = await self.press(start, "Tools")
        pick = await self.press(picker, self.tool_label(picker, SPECTRUM))
        modal = pick.response.modals[0]
        modal.inputs["position"]._value = "1"
        self.now += DAY
        submit = self.click()
        await modal.on_submit(submit)
        self.assertIn("This vault has already ended.", self.everything(submit))
        self.assertIn("Status: Expired", self.status(submit))
        self.assertEqual(self.qty(SPECTRUM), 1)


class ToolOwnerTests(ToolUiCase):
    async def test_only_the_owner_opens_the_picker_or_a_modal_or_submits(self):
        """FEAT-64: another member cannot open the picker, a tool modal, or submit a modal built for my vault; nothing is spent."""
        start = await self.started(SPECTRUM)
        self.give(SPECTRUM, 1, user=BOB)
        refused = await self.press(start, "Tools", user=BOB)
        self.assertEqual(self.everything(refused), NOT_YOURS)
        self.untouched(refused)
        picker = await self.press(start, "Tools")
        stolen = await self.press(picker, self.tool_label(picker, SPECTRUM), user=BOB)
        self.assertEqual(self.everything(stolen), NOT_YOURS)
        self.untouched(stolen)
        back = await self.press(picker, "Back", user=BOB)
        self.assertEqual(self.everything(back), NOT_YOURS)
        mine = await self.press(picker, self.tool_label(picker, SPECTRUM))
        modal = mine.response.modals[0]
        modal.inputs["position"]._value = "1"
        sneaky = self.click(BOB)
        await modal.on_submit(sneaky)
        self.assertEqual(self.everything(sneaky), NOT_YOURS)
        self.assertEqual((self.qty(SPECTRUM), self.qty(SPECTRUM, BOB), self.store.one("SELECT COUNT(*) FROM vault_tool_uses")[0]), (1, 1, 0))


class ToolStatusTests(ToolUiCase):
    async def test_help_text_counts_tools_only_when_relevant(self):
        """FEAT-64: the "Tools used: n of 2" line appears for a member holding tools or having used one, and for nobody else."""
        plain = await self.personal(BOB)
        self.assertNotIn("Tools used", self.status(plain))
        start = await self.started(SPECTRUM)
        self.assertIn("Tools used: 0 of 2", self.status(start))
        done, _, _ = await self.use(start, SPECTRUM, {"position": "1"})
        self.assertEqual(self.qty(SPECTRUM), 0)
        self.assertIn("Tools used: 1 of 2", self.status(done))
        resumed = await self.personal()
        self.assertIn("Tools used: 1 of 2", self.status(resumed))

    async def test_history_counts_tool_turns(self):
        """FEAT-64: /vault history reports guesses plus tool turn costs as turns used."""
        start = await self.started(DECODER)
        await self.use(start, DECODER, {"position": "1"})
        text = self.everything(await self.history())
        self.assertIn("In progress · 2/9 turns", text)


class ToolCustomIdTests(ToolUiCase):
    async def test_ids_round_trip_and_carry_no_secret(self):
        """FEAT-64: the Tools, picker and Back buttons rebuild from their custom ids alone and still work."""
        start = await self.started(SPECTRUM)
        sid = self.row()["id"]
        picker = await self.press(start, "Tools")
        for source in (start, picker):
            for child in self.buttons(self.view_of(source)):
                item = item_of(child)
                with self.subTest(label=item.label):
                    self.assertLessEqual(len(item.custom_id), 100)
                    self.assertNotIn(SECRET, item.custom_id)
                    match = VaultButton.__discord_ui_compiled_template__.fullmatch(item.custom_id)
                    self.assertIsNotNone(match, item.custom_id)
                    rebuilt = await VaultButton.from_custom_id(self.click(), item, match)
                    self.assertEqual((rebuilt.action, rebuilt.session_id), (match["action"], sid))
        ids = [item_of(c).custom_id for c in self.buttons(self.view_of(picker))]
        self.assertIn(f"llmcord:vault:use_{SPECTRUM}:{sid}", ids)
        tool = next(c for c in self.buttons(self.view_of(picker)) if item_of(c).custom_id == f"llmcord:vault:use_{SPECTRUM}:{sid}")
        click = self.click()
        rebuilt = await VaultButton.from_custom_id(click, item_of(tool), VaultButton.__discord_ui_compiled_template__.fullmatch(item_of(tool).custom_id))
        await rebuilt.callback(click)
        self.assertEqual(len(click.response.modals), 1)


class ToolKoreanTests(ToolUiCase):
    async def test_the_korean_flow_uses_glossary_terms_and_recipe_names(self):
        """FEAT-64: a Korean member sees Korean buttons with the recipe name (스펙트럼 분석기), and a Korean result and refusal; no English words."""
        ko = discord.Locale.korean
        start = await self.started(SPECTRUM, locale=ko)
        self.assertTrue(all(HANGUL.search(item_of(c).label) for c in self.buttons(self.view_of(start))), self.labels(start))
        self.assertNotIn("Tools used", self.status(start))
        picker = await self.press(start, self.labels(start)[-1], locale=ko)
        labels = self.labels(picker)
        self.assertIn(KO_NAMES[SPECTRUM], labels[0])
        self.assertTrue(all(HANGUL.search(label) for label in labels), labels)
        pick = await self.press(picker, labels[0], locale=ko)
        modal = pick.response.modals[0]
        self.assertTrue(HANGUL.search(modal.title))
        modal.inputs["position"]._value = "3"
        done = self.click(locale=ko)
        await modal.on_submit(done)
        self.assertIn(KO_NAMES[SPECTRUM], self.log(done))
        self.assertIsNone(re.search("Digit|Spectrum|Tools", self.log(done) + self.status(done)))
        self.assertIn(KO_NAMES[SPECTRUM], self.status(done))
        bad = self.click(locale=ko)
        modal.inputs["position"]._value = "9"
        self.give(SPECTRUM)
        await modal.on_submit(bad)
        self.assertTrue(HANGUL.search(self.everything(bad)))
        self.assertIsNone(re.search("[A-Za-z]{4}", self.everything(bad)), self.everything(bad))

    async def test_the_korean_picker_uses_the_glossary_term_for_vault_tool(self):
        """FEAT-64: the Korean picker/status vocabulary says 금고 도구 (glossary row 49); the Tools button label is Korean."""
        ko = discord.Locale.korean
        start = await self.started(SPECTRUM, locale=ko)
        button = next(c for c in self.buttons(self.view_of(start)) if item_of(c).custom_id.split(":")[2] == "tools")
        self.assertIn("도구", item_of(button).label)
        self.assertIn("금고 도구", self.status(start))

    async def test_english_has_no_korean(self):
        """FEAT-64: nothing Korean leaks into the English tool flow."""
        start = await self.started(SPECTRUM)
        picker = await self.press(start, "Tools")
        self.assertIsNone(HANGUL.search(self.status(start) + "".join(self.labels(picker))))


if __name__ == "__main__":
    unittest.main()
