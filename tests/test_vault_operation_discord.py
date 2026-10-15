"""Vault operation (co-op) on Discord (llmcord_core/vault_operation_discord.py), FEAT-66 phase B, decision D43. Written before the implementation.

The store (``vault_operation_store.py``, phase A, contract in tests/test_vault_operation_store.py) is the only source of truth. This layer
renders one shared status message in the operation channel, takes input, asks the model to WORD public facts and turns store codes into
words. Fake channel, clicks and in-memory store from ``helpers`` only; time is ``bot.vault_operation.clock`` and the debounce timer is the
injectable ``bot.vault_operation.sleep``, never the wall clock.

CONTRACT (module ``llmcord_core/vault_operation_discord.py``)

  ``register_vault_operation_commands(bot, ctx)``   called from ``discord_bot.register_commands`` right after ``register_vault_commands``.
      Builds ``bot.vault_operation = VaultOperationPlay(bot)`` and adds ``operation`` to the EXISTING ``/vault`` group (qualified name
      ``vault operation``, no options, guild-only through ``ctx.require_guild``, no admin check). ``setup_hook`` registers
      ``VaultOperationButton`` with the other dynamic items.
  ``VaultOperationPlay(bot, clock=time.time, sleep=asyncio.sleep, debounce_seconds=3.0)``
      * ``clock`` is the only time source (store ``now=``). ``sleep`` is what the debounce timer awaits (tests replace it by a gate).
      * ``await open_command(interaction)``     the ``/vault operation`` callback (below).
      * ``await press(interaction, action, operation_id)``      every button.
      * ``await submit_guess(interaction, operation_id, text)``, ``await submit_tool(interaction, operation_id, tool_key, values)``,
        ``await submit_cross(interaction, operation_id, first, second)`` (the modals' submits; texts are what the member typed),
        ``await select_advisor(interaction, operation_id, character_id | None)``     (the advisor select's callback).
      * ``await flush()``     sends every pending status edit now (cancelling the timers). ``await close()`` cancels the timers.
      * The status message id of an operation lives in memory only (no schema change); it is posted by the first successful open.
  ``VaultOperationButton(action, operation_id, label="", style=..., disabled=False)``   a ``discord.ui.DynamicItem[Button]``, custom id
      ``llmcord:vault:op_<action>:<operation_id>`` (template ``llmcord:vault:op_(?P<action>[a-z0-9_]+):(?P<operation>\\d+)``). It carries no
      member and no secret. ``VaultButton.template`` of the personal vault must NOT match these ids (it would swallow them). Its callback
      calls ``interaction.client.vault_operation.press``.
  ``OperationGuessModal(operation_id, length=4, *, t=None)``  one required ``TextInput`` at ``.guess``.
  ``OperationCrossModal(operation_id, *, t=None)``     two required inputs ``.first`` and ``.second`` (guess NUMBERS of this attempt).
  ``OperationToolModal(operation_id, tool_key, *, t=None)``    ``.inputs`` keyed by engine param name, like ``VaultToolModal``; a ``guess`` input
      is the NUMBER of a guess of the current attempt (any member's), mapped to its code; anything else is passed through for the store to refuse.

  ``/vault operation``  (the invoker's reply is ephemeral, ``invoker_t``; the channel message is ``channel_t(guild)`` resolved at EACH render)
      Refusals, in order, all ephemeral, nothing created, no channel message, store codes through ``t.error``: Vault off, operation off,
      a THREAD (even a thread whose id is the configured channel) or any channel other than the configured one -> ``vault.operation_channel``,
      fewer than two eligible cast characters -> ``vault.operation_npcs``.
      Success: ``store.open_vault_operation(guild, user, channel, now=clock(), parent_id=...)`` (NO ``seed``); the FIRST success of an operation
      posts the status message with ``channel.send(content, view=..., allowed_mentions=none)``; later joins post nothing (the status is
      marked for a debounced edit). The invoker's ephemeral reply: ``"You joined the operation."`` (new participant) or
      ``"You are already in the operation."`` (idempotent, no change). ``bot.get_channel(channel_id)`` finds the channel for later edits.
  Status message (English ids, one per line; ? conditional):
      "Weekly operation ({week})"                    week_key of the operation
      "Status: {status}"                             In progress / Solved / Failed / Expired
      "Stage {n} of 3: {protocol}"                   current stage (the last one once the operation has ended)
      "Attempt {n} of 2"                             retry + 1
      "Shared turns left: {left} of {budget}"        of the current attempt
      "Open digit: {digit}"                          ? protocol open
      "Team tools used: {n} of 2"
      "Characters: {names}"                          the two NPCs, ", "
      "Participants: {mentions}"                     ``<@id>`` mentions, joined order, ", " (no member lookup: the guesser name is a mention)
      "Advisors: {mention}: {name}; ..."             ? one entry per member who chose one
      "Clues revealed: {n} of 2" and "Clue from {name}: {text}" per revealed clue (clue texts as ``vault_discord.clue_text``)
      "Tool results:" then "{Tool name}: {fact}" per tool use (facts as ``vault_discord.tool_fact``)
      "Cross-check of guesses {a} and {b}: the code has exactly {count} of the digits {digits}."   ? digits joined by ", "
      "Guesses:" then "{n}. {mention}: {guess} — {feedback}"     every member's guess of the current attempt
      "Secret of stage {n}: {code}"                  ? only for stages that ended (never the active attempt)
      Buttons while active, in order (labels; ids ``op_<action>``): Guess (guess), Reveal clue (clue, only while a clue is hidden),
      Tools (tools), Cross-check (cross), "Ask {npc name}" per NPC (ask_<character id>), "Ask my advisor" (ask_advisor), Advisor (advisor),
      Join (join), Refresh (refresh). ``view`` is None once the operation is not active.
  Presses (every one answers the presser privately, ``invoker_t``; participant check first: a non-participant gets
      ``vault.operation_not_participant`` ("Join the operation first.") except on ``join``; a click from another guild gets ``vault.not_found``)
      guess / cross   ``send_modal`` (nothing else). tools   an ephemeral message with one button per HELD tool (``op_use_<tool key>``,
      label ``"{Tool} ×{qty} ({n} turn)"``) and ``Back``; nothing held -> ``vault.tool_not_held`` text. ``op_use_<key>`` opens the tool modal.
      clue   reveals the lowest hidden clue (free, shared), followup ``"Clue from {name}: {text}"``. ask_<id>/ask_advisor   one model call
      (the personal FEAT-65 flow: ``bot.models.structured_compiled``; fail-soft template; real usage through the store; the cap/talk checks run
      BEFORE the model; ``action_key=str(interaction.id)``), followup = ``vault_discord.advice_line``; no turn is spent. advisor   an ephemeral
      message whose view holds a ``discord.ui.Select`` (options = the member's eligible favorites, values ``str(character_id)``).
      join   joins (like the command). refresh   edits the status message NOW (no timer).
  Action results: the guesser gets an ephemeral reply (the modal's response) ``"Guess {n}: {guess} — {feedback}"``; the status message is only
      MARKED dirty (debounced: at most one edit per ``debounce_seconds``, coalescing every change; ``flush`` forces it). Tools reply
      ``"{Tool name}: {fact}"``; cross-check replies with the status line above. The store ``action_key`` is ``str(interaction.id)``.
  Events (immediate, a NEW channel message in ``channel_t``; the status is also marked dirty):
      stage_solved  "Stage {stage} cleared! Stage {next} is open."          stage_retry   "Stage {stage} ran out of turns. The team gets one more
      try with a new code."    operation_failed  "The operation failed. The vault stays closed this week."   operation_solved  "The operation is solved!
      All three stages are cleared."      The invoker's own ephemeral reply is in ``invoker_t`` (a different language than the channel's is allowed).
  Korean glossary: 작전, 단계, 금고, 파트너, 단서, 교차 검증. Character lines and names are never translated.
"""
import asyncio
import re
import unittest

import discord
from helpers import FakeGameChannel, invoke, leaf_commands

from llmcord_core.vault import all_secrets, cross_validation, feedback, filter_candidates, GuessResult
from llmcord_core.vault_discord import VaultButton, clue_text

from test_vault_discord import ALICE, BOB, CH, G, HANGUL, NOW, VaultCase, fb_en, item_of
from test_vault_discord_advice import AdviceModels

try:
    from llmcord_core.vault_operation_discord import OperationCrossModal, OperationGuessModal, OperationToolModal, VaultOperationButton
except ImportError:  # pragma: no cover - the red state
    OperationCrossModal = OperationGuessModal = OperationToolModal = VaultOperationButton = None

CAROL = 7
SECRET = "1357"
SPECTRUM, VERIFIER = "vault_spectrum_analyzer", "vault_position_verifier"
MISSING = "llmcord_core/vault_operation_discord.py is missing: SkitBot has no .vault_operation (FEAT-66 phase B)"
JOIN_FIRST = "Join the operation first."
WRONG_CHANNEL = "The operation runs in a different channel. Join it there."
OP_OFF = "The weekly operation is not turned on for this server."
NEED_NPCS = "The operation needs two eligible characters in this channel's cast to hold the clues."
NO_TOOL = "You do not have that tool. Buy one with /shop."
TOOL_LIMIT = "You can use at most 2 tools in one vault."
NOT_PARTNER = "Only a partner you chose for this vault can give advice."
LIMIT_TEXT = "This vault allows at most {limit} pieces of partner advice, and they are used up."
STAGE_SOLVED = "Stage {stage} cleared! Stage {next} is open."
STAGE_RETRY = "Stage {stage} ran out of turns. The team gets one more try with a new code."
OP_FAILED = "The operation failed. The vault stays closed this week."
OP_SOLVED = "The operation is solved! All three stages are cleared."
A, B, C = f"<@{ALICE}>", f"<@{BOB}>", f"<@{CAROL}>"


class OpCase(VaultCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.play = getattr(self.bot, "vault_operation", None)
        self.assertIsNotNone(self.play, MISSING)
        self.play.clock = lambda: self.now
        self.gate = asyncio.Event()

        async def held(seconds):  # the debounce timer waits here until a test opens the gate
            await self.gate.wait()

        self.play.sleep = held
        self.addAsyncCleanup(self.play.close)
        self.bot.get_channel = lambda ident: self.channel if ident == CH else None
        self.models = self.bot.models = AdviceModels(self)
        self.expected = {}
        self.models.answer = self.obey
        s = self.store
        s.add_favorite(G, ALICE, self.cy)
        s.set_vault_operation_settings(G, enabled=True, channel_id=CH)
        s.set_vault_settings(G, turns={"standard": 6, "open": 6})

    @staticmethod
    def obey(prompt):
        found = re.search(r"try (\d{4})", prompt)
        return f"Psst, try {found[1]} and thank me later." if found else "Trust your gut."

    # --- driving ----------------------------------------------------------------------------------------

    async def join(self, user=ALICE, *, locale=None, channel=None, guild_id=G):
        click = self.click(user, locale=locale, guild_id=guild_id)
        if channel is not None:
            click.channel, click.channel_id = channel, channel.id
        await invoke(self.bot, "vault operation", click)
        return click

    async def start(self, *users, fix=True, **kw):
        click = await self.join(ALICE, **kw)
        for user in users:
            await self.join(user, **kw)
        if fix:
            self.fix_secret()
        return click

    @property
    def oid(self):
        return self.store.current_vault_operation_id(G, CH, now=self.now)

    def op(self, user=ALICE, now=None):
        return self.store.vault_operation(G, user, self.oid, now=self.now if now is None else now)

    def fix_secret(self, code=SECRET):
        self.store.db.execute("UPDATE vault_operation_stages SET secret=? WHERE operation_id=? AND status='active'", (code, self.oid))
        self.store.db.commit()

    def secret(self):
        return self.store.one("SELECT secret FROM vault_operation_stages WHERE operation_id=? AND status='active'", (self.oid,))["secret"]

    def wrongs(self, n=1):
        secret = self.secret()
        tried = {x["guess"] for x in self.op()["stage"]["guesses"]}
        return [c for c in all_secrets(4) if c != secret and c not in tried][:n]

    def small(self, turns=3):
        self.store.set_vault_settings(G, turns={"standard": turns, "open": turns})

    def message(self):
        return self.channel.sent[0]

    def text(self):
        return self.message().content

    def edits(self):
        return len(self.message().edits)

    def labels_now(self):
        view = self.message().view
        return [item_of(c).label for c in (view.children if view is not None else [])]

    def announcements(self):
        return [m.content for m in self.channel.sent[1:]]

    def private(self, click):
        return "\n".join(c for c, _ in click.response.sent + click.followup.sent if c)

    def gone(self, click):
        return bool(click.response.sent + click.followup.sent) and all(kw.get("ephemeral") or kw.get("_edit") for _, kw in click.response.sent + click.followup.sent)

    def free_guess(self):
        return self.wrongs(1)[0]

    async def guess(self, code=None, user=ALICE, ident=None, locale=None):
        click = self.click(user, ident=ident, locale=locale)
        await self.play.submit_guess(click, self.oid, code or self.free_guess())
        return click

    async def press(self, label, user=ALICE, **kw):
        for child in self.message().view.children:
            if item_of(child).label == label:
                click = self.click(user, **kw)
                await child.callback(click)
                return click
        raise AssertionError(f"no {label!r} among {self.labels_now()}")

    async def action(self, action, user=ALICE, **kw):
        click = self.click(user, **kw)
        await self.play.press(click, action, self.oid)
        return click

    async def modal_guess(self, code, user=ALICE, ident=None, **kw):
        opened = await self.press("Guess", user, **kw)
        modal = opened.response.modals[0]
        modal.guess._value = code
        submit = self.click(user, ident=ident, **kw)
        await modal.on_submit(submit)
        return submit

    def give(self, key, qty=1, user=ALICE):
        self.store.db.execute("INSERT INTO inventory(guild_id,user_id,item_key,qty) VALUES(?,?,?,?) ON CONFLICT(guild_id,user_id,item_key) DO UPDATE SET qty=excluded.qty",
                              (G, user, key, qty))
        self.store.db.commit()

    def qty(self, key, user=ALICE):
        return self.store.item_qty(G, user, key)

    def actions(self, kind=None):
        sql = "SELECT * FROM vault_operation_actions WHERE operation_id=?" + (" AND kind=?" if kind else "") + " ORDER BY id"
        return self.store.all(sql, (self.oid, kind) if kind else (self.oid,))

    def usage(self):
        return self.store.one("SELECT COUNT(*) FROM model_usage")[0]

    def spent(self):
        """Everything an action may change: action rows, usage rows, inventory, the talk count."""
        return (len(self.actions()), self.usage(), [tuple(r) for r in self.store.all("SELECT * FROM inventory ORDER BY user_id,item_key")],
                self.store.game_talk_calls_today(G, NOW))

    async def settle(self):
        self.gate.set()
        for _ in range(12):
            await asyncio.sleep(0)

    def pair(self):
        secret = self.secret()
        wrong = [c for c in all_secrets(4) if c != secret][:60]
        import itertools
        for a, b in itertools.permutations(wrong, 2):
            common = set(a) & set(b)
            if not common:
                continue
            known = [GuessResult(x, "standard", feedback("standard", secret, x)) for x in (a, b)]
            if len({len(common & set(c)) for c in filter_candidates(known, 4)}) > 1:
                return a, b
        self.fail("no undecided pair")

    async def finish_stage(self, user=ALICE):
        self.fix_secret()
        return await self.guess(SECRET, user)

    async def burn(self, n, user=ALICE):
        for code in self.wrongs(n):
            await self.guess(code, user)


class RegistrationTests(OpCase):
    async def test_the_command_is_registered_under_vault_and_the_buttons_survive_a_restart(self):
        """FEAT-66: /vault operation sits next to /vault personal; op buttons carry the operation id only and rebuild from their custom id."""
        self.assertIn("vault operation", leaf_commands(self.bot))
        self.assertIn("vault personal", leaf_commands(self.bot))
        self.assertIsNotNone(VaultOperationButton, MISSING)
        button = VaultOperationButton("guess", 12, "Guess", discord.ButtonStyle.primary)
        self.assertEqual(button.item.custom_id, "llmcord:vault:op_guess:12")
        match = VaultOperationButton.__discord_ui_compiled_template__.fullmatch(button.item.custom_id)
        self.assertEqual((match["action"], match["operation"]), ("guess", "12"))
        rebuilt = await VaultOperationButton.from_custom_id(None, button.item, match)
        self.assertEqual((rebuilt.action, rebuilt.operation_id, rebuilt.item.custom_id, rebuilt.item.label), ("guess", 12, button.item.custom_id, "Guess"))

    async def test_the_personal_button_template_does_not_swallow_operation_ids(self):
        """FEAT-66: llmcord:vault:op_guess:5 belongs to the operation button; VaultButton must not match it (and keeps matching its own ids)."""
        template = VaultButton.__discord_ui_compiled_template__
        self.assertIsNone(template.fullmatch("llmcord:vault:op_guess:5"))
        self.assertIsNone(template.fullmatch("llmcord:vault:op_ask_12:5"))
        self.assertIsNotNone(template.fullmatch("llmcord:vault:guess:5"))
        self.assertIsNotNone(template.fullmatch("llmcord:vault:ask_12:5"))

    async def test_dm_is_refused(self):
        """FEAT-66: outside a server the command asks for a server channel and nothing is created."""
        click = self.click(guild_id=None)
        click.guild = None
        await invoke(self.bot, "vault operation", click)
        self.assertIn("server", self.private(click).lower())
        self.assertEqual((self.channel.sent, self.store.one("SELECT COUNT(*) FROM vault_operations")[0]), ([], 0))


class OpenTests(OpCase):
    async def test_the_first_join_opens_the_operation_and_posts_the_status_in_the_channel(self):
        """FEAT-66: the first /vault operation creates the week's operation, joins the member, answers privately and posts ONE public status."""
        click = await self.join()
        self.assertEqual(self.private(click), "You joined the operation.")
        self.assertTrue(self.gone(click))
        self.assertEqual(len(self.channel.sent), 1)
        lines = self.text().splitlines()
        self.assertEqual(lines[:5], ["Weekly operation (2026-10-12)", "Status: In progress", "Stage 1 of 3: Standard", "Attempt 1 of 2", "Shared turns left: 6 of 6"])
        self.assertIn("Team tools used: 0 of 2", lines)
        self.assertIn("Characters: Ann, Bea", lines)
        self.assertIn(f"Participants: {A}", lines)
        self.assertNotIn("Guesses:", self.text())
        self.assertEqual(self.labels_now(), ["Guess", "Reveal clue", "Tools", "Cross-check", "Support mission", "Ask Ann", "Ask Bea", "Ask my advisor", "Advisor", "Join", "Refresh"])
        self.assertFalse(self.channel.sent[0].kwargs.get("allowed_mentions", discord.AllowedMentions.none()).users)

    async def test_the_buttons_use_the_operation_custom_ids(self):
        """FEAT-66: every status button is a VaultOperationButton with llmcord:vault:op_<action>:<operation id>."""
        await self.start()
        ids = [item_of(c).custom_id for c in self.message().view.children]
        oid = self.oid
        self.assertEqual(ids[:4], [f"llmcord:vault:op_{a}:{oid}" for a in ("guess", "clue", "tools", "cross")])
        self.assertIn(f"llmcord:vault:op_ask_{self.ann}:{oid}", ids)
        self.assertIn(f"llmcord:vault:op_ask_advisor:{oid}", ids)
        self.assertTrue(all(isinstance(c, VaultOperationButton) for c in self.message().view.children))

    async def test_a_second_member_joins_without_a_new_message_and_the_status_follows_debounced(self):
        """FEAT-66: joining costs no turn and posts no second status; the participant list is edited in place (debounced)."""
        await self.start()
        bob = await self.join(BOB)
        self.assertEqual(self.private(bob), "You joined the operation.")
        self.assertEqual(len(self.channel.sent), 1)
        self.assertEqual(self.edits(), 0)
        await self.play.flush()
        self.assertIn(f"Participants: {A}, {B}", self.text())
        self.assertIn("Shared turns left: 6 of 6", self.text())

    async def test_joining_twice_is_idempotent(self):
        """FEAT-66: running the command again says so, adds no participant and posts no message."""
        await self.start()
        again = await self.join()
        self.assertEqual(self.private(again), "You are already in the operation.")
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_operation_participants")[0], 1)
        self.assertEqual(len(self.channel.sent), 1)

    async def test_the_join_button_adds_a_non_participant(self):
        """FEAT-66: a non-participant pressing Join becomes a participant; the status lists them after the debounce."""
        await self.start()
        click = await self.press("Join", BOB)
        self.assertEqual(self.private(click), "You joined the operation.")
        self.assertEqual(len(self.op(BOB)["participants"]), 2)
        await self.play.flush()
        self.assertIn(B, self.text())

    async def test_refusals_are_private_and_create_nothing(self):
        """FEAT-66: vault off, operation off, other channel and too few cast characters are refused with the store's text, ephemerally."""
        cases = []
        self.store.set_vault_settings(G, enabled=False)
        cases.append((await self.join(), "Vault is not turned on for this server."))
        self.store.set_vault_settings(G, enabled=True)
        self.store.set_vault_operation_settings(G, enabled=False)
        cases.append((await self.join(), OP_OFF))
        self.store.set_vault_operation_settings(G, enabled=True)
        cases.append((await self.join(channel=FakeGameChannel(200)), WRONG_CHANNEL))
        self.store.set_cast(CH, None, [self.ann])
        cases.append((await self.join(), NEED_NPCS))
        for click, text in cases:
            with self.subTest(text=text):
                self.assertEqual(self.private(click), text)
                self.assertTrue(self.gone(click))
        self.assertEqual((self.channel.sent, self.store.one("SELECT COUNT(*) FROM vault_operations")[0]), ([], 0))

    async def test_a_thread_is_refused_even_when_it_is_the_configured_channel(self):
        """FEAT-66: the operation channel must be a plain channel; a thread (here configured as THE channel) is refused and nothing is created."""
        thread = await self.channel.make_thread("t")
        self.store.set_vault_operation_settings(G, channel_id=thread.id)
        click = await self.join(channel=thread)
        self.assertEqual(self.private(click), WRONG_CHANNEL)
        self.assertEqual((thread.sent, self.channel.sent, self.store.one("SELECT COUNT(*) FROM vault_operations")[0]), ([], [], 0))

    async def test_a_click_from_another_server_is_not_found(self):
        """FEAT-66: guild isolation: a press from another guild on this operation id is answered 'not found' and changes nothing."""
        await self.start()
        before = self.spent()
        click = self.click(ALICE, guild_id=2)
        await self.play.press(click, "refresh", self.oid)
        self.assertEqual(self.private(click), "That vault was not found.")
        self.assertEqual(self.spent(), before)


class ParticipantTests(OpCase):
    async def test_a_non_participant_is_told_to_join_for_every_action(self):
        """FEAT-66: Guess, Reveal clue, Tools, Cross-check, Ask, Advisor and Refresh need participation: 'Join the operation first.', nothing opens or changes."""
        await self.start()
        before = self.spent()
        for label in ("Guess", "Reveal clue", "Tools", "Cross-check", "Ask Ann", "Ask my advisor", "Advisor", "Refresh"):
            with self.subTest(label=label):
                click = await self.press(label, CAROL)
                self.assertEqual(self.private(click), JOIN_FIRST)
                self.assertEqual(click.response.modals, [])
        for call in (self.play.submit_guess(self.click(CAROL), self.oid, "0123"), self.play.submit_cross(self.click(CAROL), self.oid, "1", "2"),
                     self.play.submit_tool(self.click(CAROL), self.oid, SPECTRUM, {"position": "1"}), self.play.select_advisor(self.click(CAROL), self.oid, self.cy)):
            await call
        await self.play.flush()
        self.assertEqual((self.spent(), self.edits()), (before, 0))
        self.assertEqual(len(self.op()["participants"]), 1)


class GuessTests(OpCase):
    async def test_guess_modal_replies_privately_and_edits_the_shared_status_after_the_debounce(self):
        """FEAT-66: Guess -> modal -> the guesser gets the feedback privately; the shared message shows everyone's guess with the guesser's mention."""
        await self.start(BOB)
        await self.play.flush()
        base = self.edits()
        opened = await self.press("Guess")
        self.assertIsInstance(opened.response.modals[0], OperationGuessModal)
        code = self.wrongs(1)[0]
        modal = opened.response.modals[0]
        modal.guess._value = code
        submit = self.click(ALICE)
        await modal.on_submit(submit)
        self.assertEqual(self.private(submit), f"Guess 1: {code} — {fb_en(feedback('standard', SECRET, code))}")
        self.assertTrue(self.gone(submit))
        self.assertEqual(self.edits(), base)  # debounced
        await self.play.flush()
        self.assertEqual(self.edits(), base + 1)
        self.assertIn("Shared turns left: 5 of 6", self.text())
        self.assertIn(f"1. {A}: {code} — {fb_en(feedback('standard', SECRET, code))}", self.text())
        other = (await self.guess(self.wrongs(1)[0], BOB))
        await self.play.flush()
        self.assertIn(f"2. {B}: ", self.text())
        self.assertIn("Guess 2: ", self.private(other))
        self.assertIn("Shared turns left: 4 of 6", self.text())

    async def test_edits_are_debounced_and_coalesced(self):
        """FEAT-66: a burst of changes produces ONE edit (with all of them) when the timer fires, and none before; a quiet period leaves no edit."""
        await self.start(BOB)
        await self.play.flush()
        base = self.edits()
        await self.guess(None, ALICE)
        await self.guess(None, BOB)
        await self.guess(None, ALICE)
        self.assertEqual(self.edits(), base)
        await self.settle()
        self.assertEqual(self.edits(), base + 1)
        self.assertIn("Shared turns left: 3 of 6", self.text())
        self.assertEqual(self.text().count(" — "), 3)
        await self.settle()
        self.assertEqual(self.edits(), base + 1)

    async def test_flush_sends_one_edit_and_nothing_when_nothing_changed(self):
        """FEAT-66: flush with a pending change edits once; a second flush with nothing pending does not touch Discord."""
        await self.start()
        await self.guess()
        await self.play.flush()
        count = self.edits()
        await self.play.flush()
        self.assertEqual(self.edits(), count)

    async def test_the_edit_resolves_the_channel_language_when_it_is_sent(self):
        """FEAT-66: channel_t is resolved at each render: a server language set after the action but before the edit is used."""
        await self.start()
        await self.guess()
        self.store.set_server_language(G, "ko")
        await self.play.flush()
        self.assertRegex(self.text(), HANGUL)
        self.assertIn("작전", self.text())

    async def test_refresh_edits_immediately(self):
        """FEAT-66: the Refresh button edits the shared message now, without waiting for the timer."""
        await self.start()
        await self.guess()
        self.assertEqual(self.edits(), 0)
        await self.press("Refresh")
        self.assertEqual(self.edits(), 1)
        self.assertIn("Shared turns left: 5 of 6", self.text())
        self.assertEqual(len(self.channel.sent), 1)

    async def test_refusals_are_translated_free_and_leave_the_status_alone(self):
        """FEAT-66: invalid and duplicate guesses are refused privately with the store's text, spend no turn and cause no edit."""
        await self.start(BOB)
        await self.play.flush()
        count = self.edits()
        first = self.wrongs(1)[0]
        await self.guess(first)
        await self.play.flush()
        count += 1
        before = self.spent()
        for code, text in (("12", "A guess must be 4 digits."), ("12a4", "A guess can only use the digits 0 to 9."), ("1123", "Every digit in a guess must be different."),
                           (first, "You already tried that guess.")):
            with self.subTest(code=code):
                click = await self.guess(code, BOB)
                self.assertEqual(self.private(click), text)
        await self.play.flush()
        self.assertEqual((self.spent(), self.edits()), (before, count))

    async def test_korean_refusal_is_translated_for_a_korean_invoker(self):
        """FEAT-66: the invoker's locale decides the refusal language, whatever the channel's language is."""
        await self.start()
        click = await self.guess("12", locale="ko")
        self.assertRegex(self.private(click), HANGUL)
        self.assertNotIn("A guess must be", self.private(click))

    async def test_double_submit_of_one_interaction_spends_one_turn(self):
        """FEAT-66: the store action key is the interaction id: a replayed modal submit shows the same feedback and spends nothing more."""
        await self.start()
        code = self.free_guess()
        one = await self.guess(code, ident=777)
        two = await self.guess(code, ident=777)
        self.assertEqual(self.private(one), self.private(two))
        self.assertEqual(len(self.actions("guess")), 1)
        self.assertEqual(self.op()["stage"]["turns_left"], 5)

    async def test_modal_length_follows_the_stage(self):
        """FEAT-66: the guess modal is bound to the operation and asks for the stage's code length."""
        await self.start()
        opened = await self.press("Guess")
        modal = opened.response.modals[0]
        self.assertEqual(modal.operation_id, self.oid)
        self.assertTrue(modal.guess.required)


class EventTests(OpCase):
    async def test_stage_solved_is_announced_publicly_and_opens_the_next_stage(self):
        """FEAT-66: a winning guess posts the stage_solved announcement as a NEW channel message; the status shows stage 2 and the solved stage's secret."""
        await self.start()
        click = await self.guess(SECRET)
        self.assertEqual(self.announcements(), [STAGE_SOLVED.format(stage=1, next=2)])
        self.assertIn("Guess 1: 1357", self.private(click))
        self.assertNotIn(STAGE_SOLVED.format(stage=1, next=2), self.private(click))
        await self.play.flush()
        self.assertIn("Stage 2 of 3: Standard", self.text())
        self.assertIn("Secret of stage 1: 1357", self.text())
        self.assertIn("Attempt 1 of 2", self.text())
        self.assertNotIn(self.secret(), self.text())
        self.assertIn("Guess", self.labels_now())

    async def test_stage_retry_is_announced_and_the_attempt_resets(self):
        """FEAT-66: spending the shared budget posts stage_retry; the status shows attempt 2, a fresh budget and no old guesses or secret."""
        self.small()
        await self.start(BOB)
        old = self.secret()
        await self.burn(2)
        await self.guess(self.wrongs(1)[0], BOB)
        self.assertEqual(self.announcements(), [STAGE_RETRY.format(stage=1)])
        await self.play.flush()
        text = self.text()
        self.assertIn("Stage 1 of 3: Standard", text)
        self.assertIn("Attempt 2 of 2", text)
        self.assertIn("Shared turns left: 3 of 3", text)
        self.assertNotIn("Guesses:", text)
        if old != self.secret():
            self.assertNotIn(old, text)
        self.assertNotIn(self.secret(), text)

    async def test_a_second_failure_fails_the_operation_and_reveals_the_secret(self):
        """FEAT-66: the second exhausted budget posts operation_failed, ends the status (no buttons) and reveals only the ended stage's secret."""
        self.small()
        await self.start()
        await self.burn(3)
        self.fix_secret("2468")
        await self.burn(3)
        self.assertEqual(self.announcements(), [STAGE_RETRY.format(stage=1), OP_FAILED])
        await self.play.flush()
        self.assertIn("Status: Failed", self.text())
        self.assertIn("Secret of stage 1: 2468", self.text())
        self.assertIsNone(self.message().view)

    async def test_the_last_stage_solved_announces_the_operation_solved(self):
        """FEAT-66: three wins announce stage_solved twice and operation_solved once; the status is Solved without buttons and shows every secret."""
        await self.start()
        await self.finish_stage()
        await self.finish_stage()
        await self.finish_stage()
        self.assertEqual(self.announcements(), [STAGE_SOLVED.format(stage=1, next=2), STAGE_SOLVED.format(stage=2, next=3), OP_SOLVED])
        await self.play.flush()
        self.assertIn("Status: Solved", self.text())
        self.assertEqual(self.text().count("Secret of stage"), 3)
        self.assertIsNone(self.message().view)

    async def test_public_announcements_use_the_channel_language_and_private_replies_the_invoker_language(self):
        """FEAT-66: Korean server + English invoker -> a Korean announcement and an English reply; English server + Korean invoker -> the reverse."""
        self.store.set_server_language(G, "ko")
        await self.start()
        click = await self.guess(SECRET, locale="en-US")
        self.assertRegex(self.announcements()[0], HANGUL)
        self.assertNotEqual(self.announcements()[0], STAGE_SOLVED.format(stage=1, next=2))
        self.assertNotRegex(self.private(click), HANGUL)
        self.store.set_server_language(G, "en")
        self.fix_secret()
        click = await self.guess(SECRET, locale="ko")
        self.assertEqual(self.announcements()[1], STAGE_SOLVED.format(stage=2, next=3))
        self.assertRegex(self.private(click), HANGUL)

    async def test_announcements_never_hold_the_active_secret(self):
        """FEAT-66: a retry announcement and the replies around it carry neither the old nor the new active secret."""
        self.small()
        await self.start()
        old = self.secret()
        await self.burn(3)
        new = self.secret()
        for text in self.announcements():
            self.assertNotIn(new, text)
            self.assertNotIn(old, text)


class ClueTests(OpCase):
    async def test_reveal_clue_is_free_shared_and_named_after_its_holder(self):
        """FEAT-66: Reveal clue shows the lowest hidden clue to the presser and, after the debounce, to the whole channel, with its NPC; no turn is spent."""
        await self.start(BOB)
        self.assertNotIn("Clue from", self.text())
        click = await self.press("Reveal clue")
        entry = next(c for c in self.op()["stage"]["clues"] if c["revealed"])
        self.assertEqual(entry["position"], 1)
        line = f"Clue from {entry['name']}: {clue_text(entry['constraint'])}"
        self.assertEqual(self.private(click), line)
        self.assertEqual(self.op()["stage"]["turns_left"], 6)
        await self.play.flush()
        self.assertIn(line, self.text())
        self.assertIn("Clues revealed: 1 of 2", self.text())
        again = await self.press("Reveal clue", BOB)
        self.assertEqual(self.op()["stage"]["turns_left"], 6)
        await self.play.flush()
        self.assertIn("Clues revealed: 2 of 2", self.text())
        self.assertNotIn("Reveal clue", self.labels_now())
        self.assertIn("Clue from", self.private(again))

    async def test_clues_stay_hidden_in_the_status_until_revealed(self):
        """FEAT-66: an unrevealed clue's text appears nowhere in the status."""
        await self.start()
        hidden = [clue_text(c["constraint"]) for c in self.store.vault_operation(G, ALICE, self.oid, now=NOW)["stage"]["clues"] if "constraint" in c]
        self.assertEqual(hidden, [])
        self.assertNotIn("Clue from", self.text())


class ToolTests(OpCase):
    async def test_the_picker_lists_only_the_members_own_tools_privately(self):
        """FEAT-66: Tools answers privately with the member's held tools and Back; a member with none gets the store's not-held text."""
        await self.start(BOB)
        self.give(SPECTRUM, 2)
        click = await self.press("Tools")
        self.assertTrue(self.gone(click))
        view = click.response.sent[0][1]["view"]
        self.assertEqual([item_of(c).label for c in view.children], ["Spectrum Analyzer ×2 (1 turn)", "Back"])
        self.assertEqual(item_of(view.children[0]).custom_id, f"llmcord:vault:op_use_{SPECTRUM}:{self.oid}")
        none = await self.press("Tools", BOB)
        self.assertEqual(self.private(none), NO_TOOL)

    async def test_using_a_tool_spends_the_callers_unit_and_the_shared_budget(self):
        """FEAT-66: picker -> modal -> the caller's unit is consumed, one shared turn is spent, the fact goes to the caller and, debounced, to the channel."""
        await self.start(BOB)
        self.give(SPECTRUM, 2)
        self.give(SPECTRUM, 1, BOB)
        picker = await self.press("Tools")
        opened = self.click()
        await picker.response.sent[0][1]["view"].children[0].callback(opened)
        modal = opened.response.modals[0]
        self.assertIsInstance(modal, OperationToolModal)
        self.assertEqual(set(modal.inputs), {"position"})
        modal.inputs["position"]._value = "2"
        submit = self.click()
        await modal.on_submit(submit)
        self.assertEqual(self.private(submit), "Spectrum Analyzer: Digit 2 is low (0 to 4).")
        self.assertEqual((self.qty(SPECTRUM), self.qty(SPECTRUM, BOB)), (1, 1))
        self.assertEqual(self.op()["stage"]["turns_left"], 5)
        await self.play.flush()
        self.assertIn("Team tools used: 1 of 2", self.text())
        self.assertIn("Tool results:\nSpectrum Analyzer: Digit 2 is low (0 to 4).", self.text())

    async def test_guess_inputs_take_the_number_of_any_guess_of_the_attempt(self):
        """FEAT-66: a verifier's guess input is the NUMBER of a guess of this attempt (a teammate's too); the fact names the code."""
        await self.start(BOB)
        self.give(VERIFIER)
        await self.guess("1250", BOB)
        click = self.click()
        await self.play.submit_tool(click, self.oid, VERIFIER, {"guess": "1"})
        self.assertRegex(self.private(click), r"^Position Verifier: In guess 1250, digit \d is \d\.$")
        self.assertEqual(self.qty(VERIFIER), 0)

    async def test_refusals_spend_nothing_and_translate_the_reason(self):
        """FEAT-66: not held, bad parameters and the team limit of 2 are refused privately with no item, turn or edit spent."""
        await self.start(BOB)
        before = self.spent()
        click = self.click(BOB)
        await self.play.submit_tool(click, self.oid, SPECTRUM, {"position": "1"})
        self.assertEqual(self.private(click), NO_TOOL)
        self.give(SPECTRUM, 5)
        bad = self.click()
        await self.play.submit_tool(bad, self.oid, SPECTRUM, {"position": "9"})
        self.assertEqual(self.private(bad), "That tool cannot be used that way. Nothing was spent. Choose a position from 1 to 4.")
        self.assertEqual(self.spent()[:2], before[:2])
        for n in (1, 2):
            await self.play.submit_tool(self.click(), self.oid, SPECTRUM, {"position": str(n)})
        limit = self.click(BOB)
        self.give(SPECTRUM, 1, BOB)
        await self.play.submit_tool(limit, self.oid, SPECTRUM, {"position": "3"})
        self.assertEqual(self.private(limit), TOOL_LIMIT)
        self.assertEqual(self.qty(SPECTRUM, BOB), 1)
        self.assertEqual(self.qty(SPECTRUM), 3)

    async def test_a_replayed_tool_interaction_consumes_once(self):
        """FEAT-66: the same interaction id twice uses one unit and one turn."""
        await self.start()
        self.give(SPECTRUM, 3)
        for _ in range(2):
            click = self.click(ident=4242)
            await self.play.submit_tool(click, self.oid, SPECTRUM, {"position": "1"})
        self.assertEqual((self.qty(SPECTRUM), self.op()["stage"]["turns_left"]), (2, 5))


class CrossTests(OpCase):
    async def test_cross_check_shares_a_public_constraint_and_costs_a_turn(self):
        """FEAT-66: two members' guesses -> the third member's cross-check modal -> a status line with the exact count; one shared turn."""
        await self.start(BOB, CAROL)
        a, b = self.pair()
        await self.guess(a, ALICE)
        await self.guess(b, BOB)
        opened = await self.press("Cross-check", CAROL)
        modal = opened.response.modals[0]
        self.assertIsInstance(modal, OperationCrossModal)
        modal.first._value, modal.second._value = "1", "2"
        submit = self.click(CAROL)
        await modal.on_submit(submit)
        common = sorted(set(a) & set(b))
        count = cross_validation(SECRET, a, b)
        line = f"Cross-check of guesses 1 and 2: the code has exactly {count} of the digits {', '.join(common)}."
        self.assertIn(line, self.private(submit))
        self.assertEqual(self.op()["stage"]["turns_left"], 3)
        await self.play.flush()
        self.assertIn(line, self.text())

    async def test_refusals_translate_and_spend_nothing(self):
        """FEAT-66: same-member guesses, unknown numbers and a second cross-check are refused privately and free."""
        await self.start(BOB)
        a, b = self.pair()
        await self.guess(a, ALICE)
        await self.guess(self.wrongs(1)[0], ALICE)
        before = self.spent()
        same = self.click()
        await self.play.submit_cross(same, self.oid, "1", "2")
        self.assertEqual(self.private(same), "Cross-validation needs guesses from two different members.")
        junk = self.click()
        await self.play.submit_cross(junk, self.oid, "x", "y")
        self.assertEqual(self.private(junk), "Pick two different guesses from this attempt.")
        self.assertEqual(self.spent(), before)


class AdvisorTests(OpCase):
    async def test_the_advisor_select_offers_the_members_favorites_only(self):
        """FEAT-66: Advisor answers privately with a Select of the member's eligible favorites; choosing one stores it and the status names it."""
        await self.start()
        click = await self.press("Advisor")
        self.assertTrue(self.gone(click))
        select = next(c for c in click.response.sent[0][1]["view"].children if isinstance(item_of(c), discord.ui.Select))
        values = {o.value: o.label for o in item_of(select).options}
        self.assertEqual(values, {str(self.ann): "Ann", str(self.cy): "Cy"})
        chosen = self.click()
        await self.play.select_advisor(chosen, self.oid, self.cy)
        self.assertEqual(self.op()["participants"][0]["advisor"]["name"], "Cy")
        self.assertIn("Cy", self.private(chosen))
        await self.play.flush()
        self.assertIn(f"Advisors: {A}: Cy", self.text())

    async def test_an_unavailable_character_is_refused(self):
        """FEAT-66: choosing a character that is not the member's favorite here is refused with the store's text; nothing is stored."""
        await self.start()
        click = self.click()
        await self.play.select_advisor(click, self.oid, self.far)
        self.assertEqual(self.private(click), "One of the chosen partners is not available to you here.")
        self.assertIsNone(self.op()["participants"][0]["advisor"])

    async def test_ask_my_advisor_needs_one_and_then_asks_it(self):
        """FEAT-66: without an advisor the press is refused with no model call; with one the advisor is asked once, in character."""
        await self.start()
        refused = await self.press("Ask my advisor")
        self.assertEqual(self.private(refused), NOT_PARTNER)
        self.assertEqual(self.models.calls, [])
        await self.play.select_advisor(self.click(), self.oid, self.cy)
        asked = await self.press("Ask my advisor")
        self.assertEqual(len(self.models.calls), 1)
        self.assertIn("Cy", self.models.calls[0]["prompt"])
        self.assertIn("Psst, try", self.private(asked))


class AdviceTests(OpCase):
    async def test_asking_an_npc_words_the_public_facts_and_spends_no_turn(self):
        """FEAT-66: Ask Ann: one model call (prompt names Ann, never the secret), the model's line is the reply, no turn or item is spent, one usage row."""
        await self.start()
        before = self.op()["stage"]["turns_left"]
        click = await self.press("Ask Ann", ident=31)
        self.assertEqual(len(self.models.calls), 1)
        self.assertIn("Ann", self.models.calls[0]["prompt"])
        self.assertNotIn(SECRET, self.models.calls[0]["prompt"])
        self.assertIn("Ann: Psst, try", self.private(click))
        self.assertTrue(self.gone(click))
        self.assertEqual((self.op()["stage"]["turns_left"], self.usage(), [a["source"] for a in self.actions("advice")]), (before, 1, ["llm"]))

    async def test_prompts_never_carry_the_secret_or_hidden_clues(self):
        """FEAT-66: across several asks the prompts and schemas hold no trace of the active secret."""
        await self.start()
        for n, label in enumerate(("Ask Ann", "Ask Bea")):
            await self.press(label, ident=40 + n)
        for call in self.models.calls:
            self.assertNotIn(SECRET, call["prompt"] + str(call["schema"]))
            self.assertNotIn("secret", str(call["schema"]).lower())

    async def test_a_model_failure_is_silent_and_still_counts_once(self):
        """FEAT-66: errors, timeouts and bad wording give the template line (it names the guess), no error text, one cap unit and one usage row."""
        await self.start()
        for n, failure in enumerate((RuntimeError("boom"), TimeoutError(), ValueError("bad"))):
            with self.subTest(error=type(failure).__name__):
                self.models.answer = failure
                before = len(self.actions("advice"))
                click = await self.press("Ask Ann", ident=50 + n)
                row = self.actions("advice")[-1]
                self.assertEqual((len(self.actions("advice")), row["source"]), (before + 1, "template"))
                self.assertIn(row["guess"], self.private(click))
                for word in ("boom", "Something went wrong", "rror"):
                    self.assertNotIn(word, self.private(click))
        self.assertEqual(self.usage(), 3)

    async def test_usage_is_recorded_once_per_advice_and_replays_ask_nothing(self):
        """FEAT-66: n delivered advices leave n usage rows; a replayed interaction id shows the first advice and asks the model nothing."""
        await self.start()
        first = await self.press("Ask Ann", ident=61)
        again = await self.action(f"ask_{self.ann}", ident=61)
        self.assertEqual(len(self.models.calls), 1)
        self.assertEqual((len(self.actions("advice")), self.usage()), (1, 1))
        self.assertEqual(self.private(first), self.private(again))
        await self.press("Ask Bea", ident=62)
        self.assertEqual((len(self.actions("advice")), self.usage()), (2, 2))

    async def test_the_team_cap_refuses_before_the_model_is_asked(self):
        """FEAT-66: the shared advice cap is checked ahead of the model: refused press, no call, nothing spent."""
        await self.start(BOB)
        self.store.set_vault_settings(G, advice_cap=1)
        await self.press("Ask Ann", ident=71)
        calls, before = len(self.models.calls), self.spent()
        refused = await self.press("Ask Bea", BOB, ident=72)
        self.assertEqual(self.private(refused), LIMIT_TEXT.format(limit=1))
        self.assertEqual((len(self.models.calls), self.spent()), (calls, before))

    async def test_a_non_participant_asks_nothing(self):
        """FEAT-66: Ask NPC by a non-participant is refused before any model call."""
        await self.start()
        click = await self.press("Ask Ann", CAROL)
        self.assertEqual((self.private(click), self.models.calls), (JOIN_FIRST, []))

    async def test_a_korean_member_gets_a_korean_refusal_and_keeps_the_model_line_verbatim(self):
        """FEAT-66: refusals follow the invoker; the model's wording is never translated."""
        await self.start()
        self.store.set_vault_settings(G, advice_cap=1)
        first = await self.press("Ask Ann", ident=81, locale="ko")
        self.assertIn("Psst, try", self.private(first))
        refused = await self.press("Ask Bea", ident=82, locale="ko")
        self.assertRegex(self.private(refused), HANGUL)
        self.assertNotIn("pieces of partner advice", self.private(refused))


class SecretTests(OpCase):
    async def test_no_active_message_ever_holds_the_secret(self):
        """FEAT-66: through guesses, clues, a tool, a cross-check and advice, neither the status, the buttons nor any private reply shows the active secret."""
        await self.start(BOB)
        self.give(SPECTRUM)
        a, b = self.pair()
        clicks = [await self.guess(a, ALICE), await self.guess(b, BOB), await self.press("Reveal clue"), await self.press("Ask Ann"), await self.press("Tools")]
        tool = self.click()
        await self.play.submit_tool(tool, self.oid, SPECTRUM, {"position": "1"})
        cross = self.click(BOB)
        await self.play.submit_cross(cross, self.oid, "1", "2")
        await self.play.flush()
        shown = [self.text()] + [self.private(c) for c in clicks + [tool, cross]]
        shown += [str(item_of(c).custom_id) + str(item_of(c).label) for c in self.message().view.children]
        shown += [call["prompt"] for call in self.models.calls]
        self.assertNotIn(SECRET, "\n".join(shown))
        self.assertNotIn("Secret of stage", self.text())
        self.assertNotIn("Secret:", self.text())


class LengthTests(OpCase):
    async def test_a_long_status_keeps_the_secrets_and_the_latest_guess_within_the_limit(self):
        """FEAT-66: ~30 participants and many guesses: the text stays within 2000 characters, whole lines only, and keeps the header, every secret and the last guess."""
        self.store.set_vault_settings(G, turns={"standard": 30, "open": 30})
        await self.start()
        crowd = [10 ** 17 + n for n in range(30)]
        for user in crowd:
            await self.join(user)
        await self.finish_stage()
        await self.finish_stage()
        await self.burn(29)
        last = await self.finish_stage()
        self.assertIn("Guess 30: 1357", self.private(last))
        await self.play.flush()
        text = self.text()
        self.assertLessEqual(len(text), 2000)
        self.assertIn("Status: Solved", text)
        self.assertEqual(text.count("Secret of stage"), 3)
        self.assertIn("Participants: 31", text)
        self.assertIn("30. <@5>: 1357", text)
        self.assertTrue(all(line.count("<@") == line.count(">") for line in text.splitlines()))


class KoreanTests(OpCase):
    async def test_korean_server_and_member_see_the_glossary_terms(self):
        """FEAT-66: a Korean server's status and a Korean member's replies use 작전, 단계, 단서 and 교차 검증; no English UI words remain in the labels."""
        self.store.set_server_language(G, "ko")
        join = await self.start(locale="ko")
        self.assertRegex(self.private(join), HANGUL)
        self.assertNotEqual(self.private(join), "You joined the operation.")
        self.assertIn("작전", self.text())
        self.assertIn("단계", self.text())
        labels = self.labels_now()
        self.assertTrue(all(HANGUL.search(x) for x in labels if x not in ("Ask Ann", "Ask Bea") and "Ann" not in x and "Bea" not in x), labels)
        self.assertTrue(any("교차 검증" in x for x in labels), labels)
        await self.press(next(x for x in labels if "단서" in x), locale="ko")
        await self.play.flush()
        self.assertIn("단서", self.text())
        self.assertNotIn("Weekly operation", self.text())
        self.assertNotIn("Clue from", self.text())

    async def test_english_output_has_no_korean(self):
        """FEAT-66: without a Korean locale nothing Korean leaks into the status, labels or replies."""
        join = await self.start()
        await self.guess()
        await self.play.flush()
        self.assertIsNone(HANGUL.search(self.text() + " ".join(self.labels_now()) + self.private(join)))


if __name__ == "__main__":
    unittest.main()
