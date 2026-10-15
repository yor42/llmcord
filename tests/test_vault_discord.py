"""Vault Discord layer (llmcord_core/vault_discord.py), FEAT-63 phase B, decision D43.

Written before the implementation: these tests define the Discord surface and fail today because the module does not exist
(``SkitBot`` has no ``.vault``, the command tree has no ``/vault``). The store (``vault_store.py``, phase A) is the only source of
truth; this layer renders it, takes input, and turns ``VaultStoreError`` codes into words. Fake channels, clicks and an in-memory
store from ``helpers`` only; time is ``bot.vault.clock``, never the wall clock.

CONTRACT (module ``llmcord_core/vault_discord.py``, registered from ``discord_bot.register_commands`` next to the other game layers)

  ``register_vault_commands(bot, ctx)``     builds ``bot.vault = VaultPlay(bot)`` and adds the ``/vault`` group (description "Experimental:
      open your personal vault by guessing its code"), guild-only through ``ctx.require_guild``, no admin check:
        ``/vault personal [protocol] [partner] [second_partner] [guess]``  options pinned in test_command_tree.py. ``protocol`` choices
            Standard / Total Match / Position-Only / Open; ``partner`` and ``second_partner`` are names with autocomplete over
            ``store.vault_partner_choices`` (the member's favorites and the channel cast, nothing else); a name that is not in that list is
            refused (nothing is created) with the ordinary name-not-found text. protocol and partners only matter when the day's vault is
            created; an existing vault is simply shown again. ``guess`` plays a guess in the same call.
        ``/vault history``    ephemeral list of the member's recent vaults, newest first.
  ``VaultPlay(bot, clock=time.time)``       ``clock`` is the only time source (store ``now=``). Methods (tests call the commands, buttons and
      modal, plus ``press`` and ``submit_guess`` below):
        ``await personal_command(interaction, protocol=None, partner=None, second_partner=None, guess=None)``
        ``await history_command(interaction)``
        ``await press(interaction, action, session_id)``     actions ``guess`` (opens the modal) and ``clue`` (reveals the lowest-numbered
            hidden clue, free). Owner check first: another member's press is answered ephemerally "This vault belongs to someone else. Run
            /vault personal to open your own." and changes nothing.
        ``await submit_guess(interaction, session_id, text)``   the modal's submit and the ``guess=`` argument; the store's ``action_key`` is
            ``str(interaction.id)``, so a double-submitted interaction does not spend two turns.
        ``render_main(view, *, t) -> str`` and ``render_buttons(view, *, t) -> discord.ui.View | None`` (None when the vault is not active).
  ``VaultButton(action, session_id, label="", style=..., disabled=False)``   a ``discord.ui.DynamicItem[Button]`` with custom id
      ``llmcord:vault:<action>:<session_id>``; it carries no member id and no secret, so it survives restarts (``setup_hook`` adds it with the
      other dynamic items); its callback calls ``interaction.client.vault.press``. Labels (English): "Guess", "Reveal clue". "Reveal clue" is
      present only while the vault is active and has a hidden clue; "Guess" only while it is active.
  ``VaultGuessModal(session_id, *, t=None)``   one required ``TextInput`` at ``.guess``; ``on_submit(interaction)`` calls ``submit_guess``.

  Every message is ephemeral (invoker_t(interaction)); the channel never sees a Vault message. The status message is the response of the
  command, and every later action EDITS it in place with ``response.edit_message`` (a modal opened from a button edits the message the button
  sits on); the per-action line (the "log") is an ephemeral followup. The edit is the interaction's own response, so there is no timer and
  no debounce to test: one edit per action, never more.

  Main view (English message ids, one string per line, in this order; lines marked ? are conditional):
      "Personal vault ({date})"
      "Protocol: {protocol}"                protocol label: Standard, Total Match, Position-Only, Open
      "Status: {status}"                    In progress, Solved, Failed, Expired
      "Turns left: {left} of {budget}"
      "Open digit: {digit}"                 ? only for the Open protocol
      "Partners: {names}"                   ? names joined by ", "; a character that no longer exists reads "a deleted character"
      "Clues revealed: {n} of {total}"      ? only with partners
      "Clue from {name}: {text}"            ? one per revealed clue; all clues once the vault has ended
      "Guesses:" then "{n}. {guess} — {feedback}"     ? one per guess. Feedback: Standard/Open "{s} strike(s), {b} ball(s)" (singular at
                                                      1), Total Match "{n} match(es)", Position-Only "{s} strike(s)"
      "Secret: {code}"                      ? only once the vault is not active, never earlier
  Clue texts: CountEven "Exactly {n} digits are even." ("Exactly 1 digit is even."), CountLow "Exactly {n} digits are low (0 to 4)."
      ("Exactly 1 digit is low (0 to 4)."), DigitPresent "The code contains the digit {d}." / "The code does not contain the digit {d}.",
      PositionParity "Digit {p} is odd." / "Digit {p} is even.", PositionHalf "Digit {p} is high (5 to 9)." / "Digit {p} is low (0 to 4)."
  Logs (followup): "Guess {n}: {guess} — {feedback}" (+ "You opened the vault!" on a win, "Out of turns. The vault stays closed today." on the
      last miss, "Reward: {amount} {currency}" when paid, "Your reward could not be paid because your balance is at its limit." when the store
      says ``reward_refused``). Refusals are ``t.error(error)`` of the store error (invalid and duplicate guesses, not found, off), sent with
      ``response.send_message(..., ephemeral=True)`` and with NO status edit and no turn spent. The one exception is ``vault.not_active``
      (expired or finished vault): the status is refreshed with ``edit_message`` (so the member sees Expired/Failed and the secret) and the
      error text follows as a followup.
  History lines: "{date} · {protocol} · {status} · {used}/{budget} turns", plus " · Secret: {code}" for ended vaults only, plus
      " · Reward: {amount} {currency}" when paid and " (reversed)" after it once the ledger entry was reversed. Empty: "You have no vaults
      yet. Start one with /vault personal."
  Korean uses the glossary: 개인 금고, 방식, 파트너, 스트라이크, 볼.
"""
import re
import unittest
from datetime import datetime, timezone

import discord
from helpers import FakeGameChannel, FakeGameClick, FakeInteraction, autocomplete, invoke, leaf_commands, make_settings

from llmcord_core.discord_bot import SkitBot
from llmcord_core.vault import all_secrets, feedback

try:  # soft import so every test fails on its own until the module exists
    from llmcord_core.vault_discord import VaultButton
except ImportError:  # pragma: no cover - the red state
    VaultButton = None

MISSING = "llmcord_core/vault_discord.py is missing: SkitBot has no .vault (FEAT-63 phase B)"
G, CH = 1, 100
ALICE, BOB = 5, 6
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc).timestamp()
DAY = 86400
HANGUL = re.compile("[가-힣]")
NOT_YOURS = "This vault belongs to someone else. Run /vault personal to open your own."


def item_of(child):
    return getattr(child, "item", child)


def fb_en(result):
    s, b = result
    return f"{s} {'strike' if s == 1 else 'strikes'}, {b} {'ball' if b == 1 else 'balls'}"


class VaultCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bot = SkitBot(make_settings())
        self.store = self.bot.store
        self.addCleanup(self.store.close)
        self.vault = getattr(self.bot, "vault", None)
        self.assertIsNotNone(self.vault, MISSING)
        self.now, self._ids = NOW, 0
        self.vault.clock = lambda: self.now
        self.channel = FakeGameChannel(CH)
        s = self.store
        self.world = s.create_space(G, "W", "world")
        self.world2 = s.create_space(G, "Elsewhere", "world")
        self.ann, self.bea, self.cy = (s.create_character(G, self.world, n) for n in ("Ann", "Bea", "Cy"))
        self.far = s.create_character(G, self.world2, "Far")
        s.bind_channel(G, CH, self.world)
        s.set_cast(CH, None, [self.ann, self.bea])
        s.add_favorite(G, ALICE, self.ann)
        s.set_vault_settings(G, enabled=True)

    # --- driving the bot ------------------------------------------------------------------------

    def click(self, user=ALICE, *, ident=None, locale=None, guild_id=G):
        click = FakeGameClick(self.bot, user, guild_id=guild_id, channel=self.channel)
        self._ids += 1
        click.id = ident if ident is not None else 9000 + self._ids
        if locale is not None:
            click.locale = locale
        return click

    async def personal(self, user=ALICE, *, locale=None, **kwargs):
        click = self.click(user, locale=locale)
        await invoke(self.bot, "vault personal", click, **{"protocol": None, "partner": None, "second_partner": None, "guess": None, **kwargs})
        return click

    async def history(self, user=ALICE, **kw):
        click = self.click(user, **kw)
        await invoke(self.bot, "vault history", click)
        return click

    @staticmethod
    def buttons(view):
        return [child for child in view.children] if view is not None else []

    def view_of(self, click):
        """The buttons of the click's status message: the last edit's view, else the first response's."""
        if click.response.edited:
            return click.response.edited[-1].get("view")
        return next((kw["view"] for _, kw in click.response.sent if kw.get("view") is not None), None)

    def labels(self, click):
        return [item_of(c).label for c in self.buttons(self.view_of(click))]

    def status(self, click):
        """The status text: the last edit of the message, else the command's response."""
        if click.response.edited:
            return click.response.edited[-1]["content"]
        return click.response.sent[0][0]

    def log(self, click):
        return "\n".join(c for c, _ in click.followup.sent if c)

    def everything(self, click):
        return "\n".join(c for c, _ in click.response.sent + click.followup.sent if c)

    def ephemeral(self, click):
        entries = click.response.sent + click.followup.sent
        return bool(entries) and all(kw.get("ephemeral") or kw.get("_edit") for _, kw in entries)

    async def press(self, source, label, user=ALICE, **kw):
        """The member presses a button of `source`'s status message."""
        view = self.view_of(source) if isinstance(source, FakeGameClick) else source
        for child in self.buttons(view):
            if item_of(child).label == label:
                click = self.click(user, **kw)
                await child.callback(click)
                return click
        raise AssertionError(f"no {label!r} button among {[item_of(c).label for c in self.buttons(view)]}")

    async def submit(self, source, code, user=ALICE, ident=None, label="Guess", **kw):
        """Guess button -> modal -> the member types `code` and submits. Returns the submit click."""
        opened = await self.press(source, label, user, **kw)
        modal = opened.response.modals[0]
        modal.guess._value = code
        submit = self.click(user, ident=ident, **kw)
        await modal.on_submit(submit)
        return submit

    # --- database oracle --------------------------------------------------------------------------

    def row(self, user=ALICE):
        return self.store.one("SELECT * FROM vault_sessions WHERE guild_id=? AND user_id=? ORDER BY id DESC", (G, user))

    def secret(self, user=ALICE):
        return self.row(user)["secret"]

    def wrong(self, n=1, user=ALICE):
        secret = self.secret(user)
        return [c for c in all_secrets(4) if c != secret][:n]

    def actions(self, user=ALICE):
        return self.store.all("SELECT * FROM vault_actions WHERE session_id=? ORDER BY number", (self.row(user)["id"],))

    def count(self, table="vault_sessions"):
        return self.store.one(f"SELECT COUNT(*) FROM {table}")[0]

    def assert_no_secret(self, *clicks, user=ALICE):
        secret = self.secret(user)
        for click in clicks:
            parts = [self.everything(click)] + [e.get("content") or "" for e in click.response.edited]
            parts += [str(item_of(c).label) + str(getattr(item_of(c), "custom_id", "")) for c in self.buttons(self.view_of(click))]
            self.assertNotIn(secret, "\n".join(parts))


class StartTests(VaultCase):
    async def test_the_command_group_is_registered(self):
        """FEAT-63: /vault personal and /vault history exist; the layer lives on the bot as bot.vault."""
        self.assertLessEqual({"vault personal", "vault history"}, set(leaf_commands(self.bot)))
        self.assertIsNotNone(VaultButton, MISSING)

    async def test_off_server_is_refused_privately_and_nothing_is_created(self):
        """FEAT-63: with the server switch off /vault personal says so, ephemerally; no row is written."""
        self.store.set_vault_settings(G, enabled=False)
        click = await self.personal()
        self.assertEqual(click.response.sent[0][0], "Vault is not turned on for this server.")
        self.assertTrue(self.ephemeral(click))
        self.assertEqual((self.count(), self.channel.sent), (0, []))

    async def test_dm_is_refused(self):
        """FEAT-63: outside a server the command asks for a server channel."""
        click = self.click(guild_id=None)
        click.guild = None
        await invoke(self.bot, "vault personal", click, protocol=None, partner=None, second_partner=None, guess=None)
        self.assertIn("server", self.everything(click).lower())
        self.assertEqual(self.count(), 0)

    async def test_start_shows_the_private_main_view(self):
        """FEAT-63: the first call creates today's vault and shows protocol, status, turn budget and a Guess button, only to the invoker."""
        click = await self.personal()
        text = self.status(click)
        lines = text.splitlines()
        self.assertEqual(lines[:4], ["Personal vault (2026-10-15)", "Protocol: Standard", "Status: In progress", "Turns left: 9 of 9"])
        self.assertNotIn("Partners:", text)
        self.assertNotIn("Open digit:", text)
        self.assertEqual(self.labels(click), ["Guess"])
        self.assertTrue(self.ephemeral(click))
        self.assertEqual(self.channel.sent, [])
        self.assertEqual((self.row()["channel_id"], self.row()["user_id"], self.count()), (CH, ALICE, 1))
        self.assert_no_secret(click)

    async def test_the_open_protocol_shows_its_opening_digit(self):
        """FEAT-63: Open adds a line with the stored opening digit and the Open budget; the secret itself stays hidden."""
        click = await self.personal(protocol="open")
        digit = self.row()["open_digit"]
        self.assertIn("Protocol: Open", self.status(click))
        self.assertIn(f"Open digit: {digit}", self.status(click).splitlines())
        self.assertIn("Turns left: 8 of 8", self.status(click))
        self.assert_no_secret(click)

    async def test_a_protocol_the_server_does_not_allow_is_refused_until_enabled(self):
        """FEAT-63: Total Match is admin opt-in: refused with the store's words, then accepted and labelled once allowed."""
        refused = await self.personal(protocol="total_match")
        self.assertEqual(refused.response.sent[0][0], "This server does not allow that Vault protocol.")
        self.assertEqual(self.count(), 0)
        self.store.set_vault_settings(G, protocols=["standard", "total_match"])
        allowed = await self.personal(protocol="total_match")
        self.assertIn("Protocol: Total Match", self.status(allowed))

    async def test_calling_again_resumes_the_same_vault(self):
        """FEAT-63: the second call the same day shows the same vault with its progress; protocol and partners are ignored."""
        first = await self.personal()
        await self.submit(first, self.wrong()[0])
        again = await self.personal(protocol="open", partner="Ann")
        text = self.status(again)
        self.assertEqual(self.count(), 1)
        self.assertIn("Protocol: Standard", text)
        self.assertIn("Turns left: 8 of 9", text)
        self.assertIn(f"1. {self.wrong()[0]} — ", text)
        self.assertNotIn("Partners:", text)

    async def test_partner_arguments_are_ignored_when_resuming_but_refused_on_creation(self):
        """FEAT-63: partners are chosen at creation only: resuming with an unknown or unreachable partner name shows the existing vault
        (no refusal, nothing changed); a first-time call with such a name is refused and creates nothing."""
        refused = await self.personal(partner="Nobody")
        self.assertIn("Nobody", self.everything(refused))
        self.assertEqual(self.count(), 0)
        await self.personal()
        for kwargs in ({"partner": "Nobody"}, {"partner": "Far", "second_partner": "Cy"}):
            with self.subTest(kwargs=kwargs):
                again = await self.personal(**kwargs)
                self.assertIn("Personal vault (2026-10-15)", self.status(again))
                self.assertNotIn("Partners:", self.status(again))
                self.assertEqual((self.count(), self.store.one("SELECT partners_json FROM vault_sessions")[0]), (1, "[]"))

    async def test_nobody_else_sees_my_vault(self):
        """FEAT-63: another member gets their own vault; the two never share state."""
        mine = await self.personal()
        await self.submit(mine, self.wrong()[0])
        theirs = await self.personal(BOB)
        self.assertIn("Turns left: 9 of 9", self.status(theirs))
        self.assertEqual(self.count(), 2)


class PartnerTests(VaultCase):
    async def test_partners_from_favorites_and_cast_are_shown_with_hidden_clues(self):
        """FEAT-63: a favorite (Ann) and a cast member (Bea) can both help; the view names them, counts the clues and reveals none yet."""
        click = await self.personal(partner="Ann", second_partner="Bea")
        text = self.status(click)
        self.assertIn("Partners: Ann, Bea", text.splitlines())
        self.assertIn("Clues revealed: 0 of 2", text.splitlines())
        self.assertNotIn("Clue from", text)
        self.assertEqual(self.labels(click), ["Guess", "Reveal clue"])
        holders = [r["character_id"] for r in self.store.all("SELECT character_id FROM vault_clues ORDER BY position")]
        self.assertEqual(holders, [self.ann, self.bea])
        self.assert_no_secret(click)

    async def test_one_partner_holds_both_clues(self):
        """FEAT-63: with one partner both initial clues are that partner's."""
        click = await self.personal(partner="Ann")
        self.assertIn("Clues revealed: 0 of 2", self.status(click))
        self.assertEqual({r["character_id"] for r in self.store.all("SELECT character_id FROM vault_clues")}, {self.ann})

    async def test_no_partner_means_no_clue_button_and_no_clue_line(self):
        """FEAT-63: without a partner there are no clues to reveal."""
        click = await self.personal()
        self.assertNotIn("Clues revealed", self.status(click))
        self.assertNotIn("Reveal clue", self.labels(click))

    async def test_a_character_the_member_cannot_reach_is_refused(self):
        """FEAT-63: a name outside the favorites and the channel cast (another world's, or a world character who is neither) is refused and creates nothing."""
        for name in ("Far", "Cy", "Nobody"):
            with self.subTest(name=name):
                click = await self.personal(partner=name)
                self.assertIn(name, self.everything(click))
                self.assertTrue(self.ephemeral(click))
                self.assertEqual(self.count(), 0)
        second = await self.personal(partner="Ann", second_partner="Cy")
        self.assertIn("Cy", self.everything(second))
        self.assertEqual(self.count(), 0)

    async def test_archived_favorites_count_only_when_the_server_allows_them(self):
        """FEAT-63: an archived favorite outside the cast is refused until the server's archived-favorites setting is on."""
        self.store.add_favorite(G, ALICE, self.cy)
        self.store.archive_character(G, self.cy, True)
        refused = await self.personal(partner="Cy")
        self.assertEqual(self.count(), 0)
        self.assertIn("Cy", self.everything(refused))
        self.store.set_archived_favorites(G, True)
        allowed = await self.personal(partner="Cy")
        self.assertIn("Partners: Cy", self.status(allowed).splitlines())

    async def test_partner_autocomplete_offers_only_reachable_characters(self):
        """FEAT-63: the partner options suggest the member's favorites and the channel cast, never other worlds' or unrelated characters."""
        def fresh():
            return FakeInteraction(guild_id=G, channel_id=CH, user_id=ALICE)
        for option in ("partner", "second_partner"):
            with self.subTest(option=option):
                values = {c.value for c in await autocomplete(self.bot, "vault personal", option, fresh(), "")}
                self.assertEqual(values, {"Ann", "Bea"})
        narrowed = await autocomplete(self.bot, "vault personal", "partner", fresh(), "be")
        self.assertEqual([c.value for c in narrowed], ["Bea"])


class ClueTests(VaultCase):
    async def start(self):
        return await self.personal(partner="Ann")

    async def test_reveal_clue_shows_the_next_hidden_clue_for_free(self):
        """FEAT-63: the button reveals clue 1, then clue 2, edits the one status message in place, costs no turn, and then goes away."""
        start = await self.start()
        first = await self.press(start, "Reveal clue")
        self.assertEqual(len(first.response.edited), 1)
        self.assertEqual(first.response.sent[0][1].get("_edit"), True)
        text = self.status(first)
        self.assertIn("Clues revealed: 1 of 2", text)
        self.assertEqual(text.count("Clue from Ann:"), 1)
        self.assertIn("Turns left: 9 of 9", text)
        self.assertEqual(self.labels(first), ["Guess", "Reveal clue"])
        second = await self.press(first, "Reveal clue")
        self.assertEqual(self.status(second).count("Clue from Ann:"), 2)
        self.assertIn("Clues revealed: 2 of 2", self.status(second))
        self.assertEqual(self.labels(second), ["Guess"])
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_clues WHERE revealed_at IS NOT NULL")[0], 2)
        self.assertEqual(self.channel.sent, [])
        self.assert_no_secret(first, second)

    async def test_each_clue_kind_has_an_english_template_naming_the_partner(self):
        """FEAT-63: CountEven, CountLow, DigitPresent, PositionParity and PositionHalf each read as their template, with the holder's name."""
        cases = [
            ("CountEven", '{"count": 2}', "Exactly 2 digits are even."),
            ("CountEven", '{"count": 1}', "Exactly 1 digit is even."),
            ("CountLow", '{"count": 3}', "Exactly 3 digits are low (0 to 4)."),
            ("CountLow", '{"count": 1}', "Exactly 1 digit is low (0 to 4)."),
            ("DigitPresent", '{"digit": "7", "present": true}', "The code contains the digit 7."),
            ("DigitPresent", '{"digit": "3", "present": false}', "The code does not contain the digit 3."),
            ("PositionParity", '{"position": 2, "odd": true}', "Digit 2 is odd."),
            ("PositionParity", '{"position": 4, "odd": false}', "Digit 4 is even."),
            ("PositionHalf", '{"position": 3, "high": true}', "Digit 3 is high (5 to 9)."),
            ("PositionHalf", '{"position": 1, "high": false}', "Digit 1 is low (0 to 4)."),
        ]
        start = await self.start()
        for kind, data, expected in cases:
            with self.subTest(kind=kind, data=data):
                self.store.db.execute("UPDATE vault_clues SET kind=?,data_json=?,revealed_at=NULL WHERE position=0", (kind, data))
                self.store.db.commit()
                click = await self.press(start, "Reveal clue")
                self.assertIn(f"Clue from Ann: {expected}", self.status(click).splitlines())

    async def test_a_deleted_partner_reads_as_a_deleted_character(self):
        """FEAT-63: a partner whose character was deleted is named by the translated fallback, in the partner line and on the clue."""
        await self.start()
        self.store.delete_character(G, self.ann, self.store.owner_revision(G, "character", self.ann))
        shown = await self.personal()
        self.assertIn("Partners: a deleted character", shown.response.sent[0][0].splitlines())
        click = await self.press(shown, "Reveal clue")
        self.assertIn("Clue from a deleted character: ", self.status(click))
        self.assertNotIn("None", self.status(click))

    async def test_only_the_owner_can_reveal_a_clue(self):
        """FEAT-63: another member pressing Reveal clue gets the ephemeral refusal and no clue is revealed."""
        start = await self.start()
        other = await self.press(start, "Reveal clue", user=BOB)
        self.assertEqual(self.everything(other), NOT_YOURS)
        self.assertTrue(self.ephemeral(other))
        self.assertEqual(other.response.edited, [])
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_clues WHERE revealed_at IS NOT NULL")[0], 0)

    async def test_a_clue_press_after_the_vault_ended_explains_and_refreshes(self):
        """FEAT-63: pressing a stale Reveal clue on a finished vault shows the ended status and the explanation, privately."""
        start = await self.start()
        self.now += DAY
        click = await self.press(start, "Reveal clue")
        self.assertIn("This vault has already ended.", self.everything(click))
        self.assertIn("Status: Expired", self.status(click))
        self.assertTrue(self.ephemeral(click))


class GuessTests(VaultCase):
    async def test_the_guess_button_opens_a_modal_with_one_input(self):
        """FEAT-63: Guess opens a modal (the response is the modal, nothing is posted) with a single text input."""
        start = await self.personal()
        opened = await self.press(start, "Guess")
        self.assertEqual(len(opened.response.modals), 1)
        self.assertIsInstance(opened.response.modals[0].guess, discord.ui.TextInput)
        self.assertEqual((opened.response.sent, opened.response.edited, self.channel.sent), ([], [], []))

    async def test_a_valid_wrong_guess_edits_the_status_and_logs_privately(self):
        """FEAT-63: one valid guess spends one turn, edits the status once with the guess and its feedback, sends one ephemeral log line,
        records the interaction id as the action key, and never shows the secret."""
        start = await self.personal()
        guess = self.wrong()[0]
        result = feedback("standard", self.secret(), guess)
        done = await self.submit(start, guess, ident=777)
        self.assertEqual(len(done.response.edited), 1)
        self.assertIn("Turns left: 8 of 9", self.status(done))
        self.assertIn("Guesses:", self.status(done).splitlines())
        self.assertIn(f"1. {guess} — {fb_en(result)}", self.status(done).splitlines())
        self.assertIn(f"Guess 1: {guess} — {fb_en(result)}", self.log(done))
        self.assertTrue(self.ephemeral(done))
        self.assertEqual([(a["guess"], a["action_key"]) for a in self.actions()], [(guess, "777")])
        self.assertEqual(self.channel.sent, [])
        self.assert_no_secret(done)

    async def test_invalid_guesses_are_free_and_explained(self):
        """FEAT-63: a short, non-digit or repeated-digit guess is answered privately with the reason, edits nothing, and spends no turn."""
        start = await self.personal()
        for text, reason in (("12", "A guess must be 4 digits."), ("12a4", "A guess can only use the digits 0 to 9."),
                             ("1123", "Every digit in a guess must be different."), ("", "A guess must be 4 digits.")):
            with self.subTest(text=text):
                done = await self.submit(start, text)
                self.assertEqual(self.everything(done), reason)
                self.assertTrue(self.ephemeral(done))
                self.assertEqual(done.response.edited, [])
        self.assertEqual(self.actions(), [])
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_actions")[0], 0)

    async def test_a_repeated_guess_is_free_and_explained(self):
        """FEAT-63: guessing the same code twice (a new interaction) is refused with the duplicate text and spends nothing more."""
        start = await self.personal()
        guess = self.wrong()[0]
        await self.submit(start, guess)
        again = await self.submit(start, guess)
        self.assertEqual(self.everything(again), "You already tried that guess.")
        self.assertEqual(again.response.edited, [])
        self.assertEqual(len(self.actions()), 1)

    async def test_a_double_submitted_interaction_spends_one_turn(self):
        """FEAT-63: the same interaction id delivered twice records one action and shows the same result, never a second guess."""
        start = await self.personal()
        guess = self.wrong()[0]
        first = await self.submit(start, guess, ident=4242)
        second = await self.submit(start, guess, ident=4242)
        self.assertEqual(len(self.actions()), 1)
        self.assertIn("Turns left: 8 of 9", self.status(second))
        self.assertNotIn("Guess 2", self.everything(second))
        self.assertIn("Guess 1:", self.everything(second))
        self.assertEqual(self.status(first), self.status(second))

    async def test_the_guess_argument_plays_a_guess_in_the_same_call(self):
        """FEAT-63: /vault personal guess=... on an existing vault shows the updated status and logs the guess; a bad one is explained and free."""
        await self.personal()
        guess = self.wrong()[0]
        played = await self.personal(guess=guess)
        self.assertIn("Turns left: 8 of 9", self.status(played))
        self.assertIn(f"Guess 1: {guess}", self.everything(played))
        bad = await self.personal(guess="99")
        self.assertIn("A guess must be 4 digits.", self.everything(bad))
        self.assertEqual(len(self.actions()), 1)

    async def test_only_the_owner_can_guess(self):
        """FEAT-63: another member cannot open the modal from my button, nor submit my modal; no guess is recorded."""
        start = await self.personal()
        refused = await self.press(start, "Guess", user=BOB)
        self.assertEqual(self.everything(refused), NOT_YOURS)
        self.assertEqual(refused.response.modals, [])
        opened = await self.press(start, "Guess")
        modal = opened.response.modals[0]
        modal.guess._value = self.wrong()[0]
        sneaky = self.click(BOB)
        await modal.on_submit(sneaky)
        self.assertEqual(self.everything(sneaky), NOT_YOURS)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM vault_actions")[0], 0)

    async def test_winning_reveals_the_secret_and_closes_the_buttons(self):
        """FEAT-63: the right code marks Solved, says so, shows the secret for the first time and removes the buttons. Earlier calls never showed it."""
        start = await self.personal()
        miss = await self.submit(start, self.wrong()[0])
        self.assert_no_secret(start, miss)
        secret = self.secret()
        win = await self.submit(start, secret)
        self.assertIn("You opened the vault!", self.log(win))
        self.assertIn("Status: Solved", self.status(win))
        self.assertIn(f"Secret: {secret}", self.status(win).splitlines())
        self.assertEqual(self.labels(win), [])
        self.assertNotIn("Reward", self.everything(win))

    async def test_a_win_pays_the_reward_once_and_says_so(self):
        """FEAT-63: with a reward configured the win log names it, the member is paid exactly once even if the interaction repeats."""
        self.store.set_vault_settings(G, reward=50)
        start = await self.personal()
        win = await self.submit(start, self.secret(), ident=31)
        await self.submit(start, self.secret(), ident=31)
        self.assertIn("Reward: 50 coins", self.log(win))
        self.assertEqual(self.store.balance(G, ALICE), 50)
        self.assertEqual(self.store.one("SELECT COUNT(*) FROM currency_ledger WHERE source='vault'")[0], 1)

    async def test_a_refused_reward_is_announced_on_the_winning_call(self):
        """FEAT-63 (phase-B follow-up): when the balance cap refuses the reward the member is told on the winning call; the win still stands."""
        cap = self.store.MAX_CURRENCY_BALANCE
        self.store.db.execute("INSERT INTO currency_balances(guild_id,user_id,balance,updated_at) VALUES(?,?,?,?)", (G, ALICE, cap - 10, NOW))
        self.store.db.commit()
        self.store.set_vault_settings(G, reward=50)
        start = await self.personal()
        win = await self.submit(start, self.secret())
        self.assertIn("Your reward could not be paid because your balance is at its limit.", self.log(win))
        self.assertNotIn("Reward: 50", self.everything(win))
        self.assertIn("Status: Solved", self.status(win))
        self.assertEqual(self.store.balance(G, ALICE), cap - 10)

    async def test_running_out_of_turns_fails_the_vault_and_it_stays_failed(self):
        """FEAT-63: the last miss says Out of turns, reveals the secret and every clue, removes the buttons, and calling again the same day
        shows the failed vault (no fresh secret)."""
        start = await self.personal(partner="Ann")
        guesses = self.wrong(9)
        results = []
        for guess in guesses:
            results.append(await self.submit(start, guess))
        for early in results[:-1]:
            self.assert_no_secret(early)
            self.assertNotIn("Out of turns", self.everything(early))
        secret, last = self.secret(), results[-1]
        self.assertIn("Out of turns. The vault stays closed today.", self.log(last))
        self.assertIn("Status: Failed", self.status(last))
        self.assertIn(f"Secret: {secret}", self.status(last).splitlines())
        self.assertEqual(self.status(last).count("Clue from Ann:"), 2)
        self.assertEqual(self.labels(last), [])
        again = await self.personal()
        self.assertIn("Status: Failed", self.status(again))
        self.assertEqual((self.count(), self.secret()), (1, secret))
        self.assertEqual(self.labels(again), [])

    async def test_a_guess_after_the_day_ended_explains_and_shows_the_expired_vault(self):
        """FEAT-63: past the server midnight a guess is refused with the ended text, the status turns to Expired and reveals the secret."""
        start = await self.personal()
        self.now += DAY
        done = await self.submit(start, self.wrong()[0])
        self.assertIn("This vault has already ended.", self.everything(done))
        self.assertIn("Status: Expired", self.status(done))
        self.assertIn(f"Secret: {self.secret()}", self.status(done).splitlines())
        self.assertEqual(len(self.actions()), 0)
        self.assertTrue(self.ephemeral(done))

    async def test_buttons_survive_a_restart(self):
        """FEAT-63: the custom id holds only the action and the session; rebuilt from it the button still works and leaks nothing."""
        start = await self.personal(partner="Ann")
        for child in self.buttons(self.view_of(start)):
            item = item_of(child)
            session_id = self.row()["id"]
            self.assertLessEqual(len(item.custom_id), 100)
            self.assertNotIn(self.secret(), item.custom_id)
            match = VaultButton.__discord_ui_compiled_template__.fullmatch(item.custom_id)
            self.assertIsNotNone(match, item.custom_id)
            rebuilt = await VaultButton.from_custom_id(self.click(), item, match)
            self.assertEqual((rebuilt.action, rebuilt.session_id, rebuilt.item.label), (match["action"], session_id, item.label))
        rebuilt_clue = next(c for c in self.buttons(self.view_of(start)) if item_of(c).label == "Reveal clue")
        clicked = self.click()
        await (await VaultButton.from_custom_id(clicked, item_of(rebuilt_clue), VaultButton.__discord_ui_compiled_template__.fullmatch(item_of(rebuilt_clue).custom_id))).callback(clicked)
        self.assertIn("Clue from Ann:", self.status(clicked))


class HistoryTests(VaultCase):
    async def test_empty_history_says_how_to_start(self):
        """FEAT-63: no vaults yet -> a private pointer to /vault personal."""
        click = await self.history()
        self.assertEqual(self.everything(click), "You have no vaults yet. Start one with /vault personal.")
        self.assertTrue(self.ephemeral(click))

    async def test_secrets_only_for_ended_vaults_newest_first(self):
        """FEAT-63: yesterday's failed vault shows its secret, today's active one does not; newest first; ephemeral."""
        start = await self.personal()
        for guess in self.wrong(9):
            await self.submit(start, guess)
        old_secret = self.secret()
        self.now += DAY
        await self.personal()
        new_secret = self.secret()
        click = await self.history()
        text = self.everything(click)
        lines = {line.split(" · ")[0]: line for line in text.splitlines() if " · " in line}
        self.assertLess(text.index("2026-10-16"), text.index("2026-10-15"))
        self.assertIn("Standard · Failed · 9/9 turns", lines["2026-10-15"])
        self.assertIn(f"Secret: {old_secret}", lines["2026-10-15"])
        self.assertIn("Standard · In progress · 0/9 turns", lines["2026-10-16"])
        self.assertNotIn("Secret", lines["2026-10-16"])
        self.assertNotIn(new_secret, text)
        self.assertTrue(self.ephemeral(click))
        self.assertEqual(self.channel.sent, [])

    async def test_history_is_the_members_own(self):
        """FEAT-63: another member's history does not list my vault or its secret."""
        await self.personal()
        mine = self.secret()
        click = await self.history(BOB)
        self.assertEqual(self.everything(click), "You have no vaults yet. Start one with /vault personal.")
        self.assertNotIn(mine, self.everything(click))

    async def test_a_reversed_reward_is_marked(self):
        """FEAT-63: a paid reward shows its amount; once an admin reverses the ledger entry the line says (reversed)."""
        self.store.set_vault_settings(G, reward=50)
        start = await self.personal()
        await self.submit(start, self.secret())
        paid = self.everything(await self.history())
        self.assertIn("Solved", paid)
        self.assertIn("Reward: 50 coins", paid)
        self.assertNotIn("reversed", paid)
        entry = self.store.one("SELECT id FROM currency_ledger WHERE source='vault'")["id"]
        self.store.reverse_entry(G, entry, 1, "test")
        reversed_ = self.everything(await self.history())
        self.assertIn("Reward: 50 coins (reversed)", reversed_)


class KoreanTests(VaultCase):
    async def test_the_main_view_and_a_guess_log_render_in_korean_with_glossary_terms(self):
        """FEAT-63: a Korean invoker gets the main view with the glossary terms and Korean buttons; English words are gone; the guess log
        names strikes and balls in Korean. Digits, names and dates stay as they are."""
        ko = discord.Locale.korean
        start = await self.personal(locale=ko, partner="Ann")
        text = self.status(start)
        for term in ("개인 금고", "방식", "파트너"):
            self.assertIn(term, text)
        for english in ("Personal vault", "Protocol", "Partners", "Turns left", "Status"):
            self.assertNotIn(english, text)
        self.assertIn("Ann", text)
        self.assertIn("2026-10-15", text)
        self.assertTrue(all(HANGUL.search(item_of(c).label) for c in self.buttons(self.view_of(start))), self.labels(start))
        done = await self.submit(start, self.wrong()[0], locale=ko, label=item_of(self.buttons(self.view_of(start))[0]).label)
        self.assertIn("스트라이크", self.log(done))
        self.assertIn("볼", self.log(done))
        self.assertIn(self.wrong()[0], self.log(done))
        self.assertNotIn("Guess", self.log(done))

    async def test_english_output_has_no_korean(self):
        """FEAT-63: without a Korean locale nothing Korean leaks into the English view."""
        start = await self.personal(partner="Ann")
        self.assertIsNone(HANGUL.search(self.status(start)))


if __name__ == "__main__":
    unittest.main()
