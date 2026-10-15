"""Vault personal store (llmcord_core/vault_store.py), FEAT-63 phase A, schema v25, decision D43.

Written before the implementation: these tests define the store API and fail today because the module and tables do
not exist. The engine (``llmcord_core.vault``, FEAT-62) stays pure; this store adds persistence, the day rule, the
guild settings and the reward. Phase B (Discord) uses only this API.

API CONTRACT (mixin ``VaultStore`` in ``llmcord_core/vault_store.py``, listed in ``class Store(...)`` like HoldemStore)
Every method is synchronous, takes ``guild_id`` first, scopes every query by it, and takes ``now`` (Unix seconds, default
``time.time()``) wherever time matters. Writes use ``write_admin()``. Refusals raise ``VaultStoreError`` (a ``UserError``
subclass defined in ``vault_store.py``); ``err.code.code`` is the dotted code in the tables below, ``err.params`` the values.
Revision conflicts raise ``admin_store.ConflictError``.

  Settings (own table ``vault_settings``; one row per guild, defaults when absent)
    vault_settings(guild_id) -> {"enabled": False, "protocols": ["standard", "open"], "turns": dict(DEFAULT_TURNS),
                                 "reward": 0, "revision": 0, "advice_cap": 6, "clue_voice": "template"}
    set_vault_settings(guild_id, *, enabled=None, protocols=None, turns=None, reward=None, expected_revision=None) -> new dict
        None leaves a field alone; revision +1 per successful call; expected_revision mismatch -> ConflictError, nothing written.
        protocols: non-empty subset of vault.PROTOCOLS (stored in PROTOCOLS order). turns: {protocol: int 1..30}, merged over the
        current map. reward: int 0..MAX_CURRENCY_CHANGE paid on a win (0 = no reward, like daily_amount 0). bool is not an int.
        Bad input raises UserError (any code) and writes nothing.

  Sessions (``vault_sessions`` + ``vault_stages`` + ``vault_actions`` + ``vault_clues`` + ``vault_reward_claims``)
    open_personal_vault(guild_id, user_id, channel_id, *, protocol="standard", partner_ids=(), now=None, seed=None) -> view
        Start or resume the member's vault for the server day. If the member has a vault whose expires_at > now (any status, so a
        failed or solved vault is returned again, never reset), return it with created=False and ignore protocol/partner_ids/seed.
        Otherwise the day must be enabled (vault.disabled), protocol allowed (vault.protocol_not_allowed), partners accessible
        (vault.partner_unavailable) and at most 2 after removing duplicates (vault.partners_too_many); then create: secret from
        ``random.Random(seed)`` (seed None -> secrets.SystemRandom), local_date and expires_at (next local midnight) from the server
        timezone at creation and never recomputed, turn_budget from the guild settings at creation, open digit for protocol "open",
        clues from vault.make_clues(secret, 2, rng) persisted now: 1 partner holds both, 2 partners one each, no partner no clues.
        Clues start UNREVEALED (column ``vault_clues.revealed_at``, nullable). Optional ``parent_id`` (the thread's parent channel,
        as in vault_partner_choices) is used for the partner check. If the creation date already belongs to an earlier (expired)
        vault of the member, which can happen after a timezone change, that existing row is returned.
        Opening an existing vault never needs the feature on; only creating one does.
    reveal_vault_clue(guild_id, user_id, session_id, position, *, now=None) -> clue entry (as in view["clues"], revealed=True)
        position is the clue's 1-based number. First call stores revealed_at=now; later calls return the same entry and keep the
        first revealed_at. Codes: vault.not_found (other member/guild/session), vault.not_active (solved, failed or expired
        session), vault.clue_unknown (position not an int 1..number of clues; bool/str rejected; a vault without partners has none).
        It costs no turn.
        Expired sessions found on the way are marked expired first. Column ``vault_sessions.secret`` holds the code.
    vault_session(guild_id, user_id, session_id, *, now=None) -> view     other member / other guild -> vault.not_found
    expire_vault_sessions(guild_id, now=None) -> int     marks active sessions with expires_at <= now expired; returns the count
        (every read and write above also does this lazily for the session it touches).
    vault_partner_choices(guild_id, user_id, channel_id, parent_id=None) -> [{"character_id", "name", "source": "favorite"|"cast"}]
        The member's eligible favorites (store.eligible_favorites on the channel's space; archived ones only while
        archived_favorites is on) plus the channel cast (store.get_cast) members that are eligible_characters; a character
        in both appears once with source "favorite". Unbound channel -> [].
    submit_vault_guess(guild_id, user_id, session_id, guess, *, action_key=None, now=None) -> result
        result = {"number", "guess", "feedback", "won", "replayed", "session": view}. Invalid guess -> vault.guess_length /
        vault.guess_not_digits / vault.guess_repeated_digit; repeated guess -> vault.guess_duplicate; inactive session
        (solved, failed, expired) -> vault.not_active. All of these cost nothing and write nothing. A valid guess is one
        vault_actions row (unique per session+action_key when a key is given): same key again returns the first result with
        replayed=True and changes nothing. A win marks solved, exhausting the budget marks failed.
    vault_history(guild_id, user_id, *, limit=10, now=None) -> [summary]     newest first, the member's own only

  view (a dict): id, guild_id, user_id, channel_id, local_date (ISO str), protocol, length, status ("active"|"solved"|
    "failed"|"expired"), turn_budget, turns_used, turns_left, created_at, expires_at, created (bool, True only on the call that
    created it), open_digit (str or None), partners [{"character_id", "name"}] (name None for a deleted
    character), clues [{"position", "character_id", "name", "revealed", "revealed_at"}] for EVERY clue, plus "constraint" (a vault
    dataclass such as CountEven) only when revealed (the key is absent otherwise), guesses [{"number", "guess", "feedback", "won"}] oldest first,
    constraints (the public constraints for the solver: OpenDigit, REVEALED clue constraints only, and vault.played(...) per guess),
    secret (None until status is not "active"), reward ({"amount", "ledger_id"} or None), reward_reversed (bool, True once the
    reward's ledger entry was reversed through reverse_entry; reward keeps the amount paid), reward_refused (True on the winning
    call's result["session"] and result when the balance cap refused the reward, else False/absent).
  summary: id, local_date, protocol, status, turns_used, turn_budget, secret (None unless terminal), reward (amount int or 0), reward_reversed (bool).

  Reward: on a win with settings reward > 0, in the same transaction: ledger row (source "vault", positive amount, actor = the
    member) and one vault_reward_claims row unique per (session, user, type); never paid twice, never on failed or expired.
    Reversal is the ordinary store.reverse_entry; it does not re-pay. A balance cap that refuses the reward leaves the win
    solved with no ledger row and no claim row (reward_refused).
  Settings storage: vault_settings.turns_json holds only the guild's OVERRIDES of vault.DEFAULT_TURNS; vault_settings() merges them
    over the current defaults, so a guild that never set turns follows later changes of the defaults.
  Open vaults keep playing: switching the feature off, or removing a protocol from the allowed set, does not stop a vault already
    open (guesses still work and the win still pays); only creating a new vault is refused.

  Error codes: vault.disabled, vault.protocol_not_allowed, vault.partner_unavailable, vault.partners_too_many, vault.not_found,
    vault.not_active, vault.guess_length, vault.guess_not_digits, vault.guess_repeated_digit, vault.guess_duplicate, vault.clue_unknown.
  Partner ids must be non-bool ints, otherwise vault.partner_unavailable and nothing is stored.

The secret never appears in a view while active, in any repr, or in any error text or params.
"""
import sqlite3
import tempfile
import json
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

from llmcord_core.admin_store import ConflictError
from llmcord_core.i18n import UserError
from llmcord_core.store import Store
from llmcord_core.vault import DEFAULT_TURNS, CountEven, CountLow, DigitPresent, PositionHalf, PositionParity, all_secrets, feedback, filter_candidates
from llmcord_core.vault_store import VaultStoreError

G, G2, CH = 1, 2, 100
ALICE, BOB = 5, 6
NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc).timestamp()
MIDNIGHT = datetime(2026, 10, 16, 0, 0, tzinfo=timezone.utc).timestamp()
DAY = 86400
CLUE_TYPES = (CountEven, CountLow, DigitPresent, PositionHalf, PositionParity)


def code_of(err):
    return err.code.code


class VaultCase(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)
        s = self.store
        self.w = s.create_space(G, "W", "world")
        self.w2 = s.create_space(G, "Elsewhere", "world")
        self.ann, self.bea, self.cy = (s.create_character(G, self.w, n) for n in ("Ann", "Bea", "Cy"))
        self.far = s.create_character(G, self.w2, "Far")
        s.bind_channel(G, CH, self.w)
        s.set_cast(CH, None, [self.ann, self.bea])
        s.set_vault_settings(G, enabled=True)

    def open(self, user=ALICE, now=NOW, seed="seed-1", guild=G, channel=CH, **kw):
        return self.store.open_personal_vault(guild, user, channel, now=now, seed=seed, **kw)

    def secret_of(self, view):
        return self.store.one("SELECT secret FROM vault_sessions WHERE id=?", (view["id"],))["secret"]

    def wrong(self, view, n=1, skip=()):
        secret = self.secret_of(view)
        return [c for c in all_secrets(4) if c != secret and c not in skip][:n]

    def guess(self, view, code, user=ALICE, now=NOW, **kw):
        return self.store.submit_vault_guess(G, user, view["id"], code, now=now, **kw)

    def fail(self, view, now=NOW):
        for code in self.wrong(view, view["turn_budget"]):
            result = self.guess(view, code, now=now)
        return result

    def ledger(self, guild=G):
        return [dict(r) for r in self.store.db.execute("SELECT * FROM currency_ledger WHERE guild_id=? AND source='vault' ORDER BY id", (guild,))]


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)

    def test_defaults_are_off_standard_and_open_with_engine_budgets_and_no_reward(self):
        settings = self.store.vault_settings(G)
        self.assertEqual(settings, {"enabled": False, "protocols": ["standard", "open"], "turns": dict(DEFAULT_TURNS), "reward": 0, "revision": 0, "advice_cap": 6, "clue_voice": "template"})

    def test_save_reads_back_bumps_revision_and_leaves_other_fields(self):
        first = self.store.set_vault_settings(G, enabled=True, reward=25)
        self.assertEqual((first["enabled"], first["reward"], first["revision"], first["protocols"]), (True, 25, 1, ["standard", "open"]))
        second = self.store.set_vault_settings(G, protocols=["open", "total_match", "standard"], turns={"total_match": 12}, expected_revision=1)
        self.assertEqual(second["protocols"], ["standard", "total_match", "open"])
        self.assertEqual(second["turns"], {**DEFAULT_TURNS, "total_match": 12})
        self.assertEqual((second["enabled"], second["reward"], second["revision"]), (True, 25, 2))
        self.assertEqual(self.store.vault_settings(G), second)

    def test_stale_revision_conflicts_and_writes_nothing(self):
        self.store.set_vault_settings(G, enabled=True)
        for stale in (0, 5):
            with self.assertRaises(ConflictError):
                self.store.set_vault_settings(G, reward=9, expected_revision=stale)
        self.assertEqual((self.store.vault_settings(G)["reward"], self.store.vault_settings(G)["revision"]), (0, 1))

    def test_bad_input_is_refused_and_writes_nothing(self):
        bad = [{"protocols": []}, {"protocols": ["nope"]}, {"protocols": "standard"}, {"turns": {"standard": 0}}, {"turns": {"standard": True}},
               {"turns": {"standard": 99}}, {"turns": {"nope": 5}}, {"reward": -1}, {"reward": True}, {"reward": 10 ** 9}, {"enabled": "yes"}]
        for case in bad:
            with self.subTest(case=case):
                with self.assertRaises(UserError):
                    self.store.set_vault_settings(G, **case)
        self.assertEqual(self.store.vault_settings(G)["revision"], 0)

    def test_only_overrides_are_stored_so_defaults_keep_applying(self):
        """Saving the switch alone must not freeze today's DEFAULT_TURNS into the guild."""
        self.store.set_vault_settings(G, enabled=True)
        self.assertEqual(json.loads(self.store.one("SELECT turns_json FROM vault_settings WHERE guild_id=?", (G,))["turns_json"] or "{}"), {})
        with mock.patch.dict(DEFAULT_TURNS, {"standard": 11}):
            self.assertEqual(self.store.vault_settings(G)["turns"], {**DEFAULT_TURNS, "standard": 11})
            self.store.set_vault_settings(G, turns={"open": 5})
            stored = json.loads(self.store.one("SELECT turns_json FROM vault_settings WHERE guild_id=?", (G,))["turns_json"])
            self.assertEqual(stored, {"open": 5})
            self.assertEqual(self.store.vault_settings(G)["turns"], {**DEFAULT_TURNS, "standard": 11, "open": 5})

    def test_settings_are_per_guild(self):
        self.store.set_vault_settings(G, enabled=True, reward=7)
        self.assertEqual(self.store.vault_settings(G2), {"enabled": False, "protocols": ["standard", "open"], "turns": dict(DEFAULT_TURNS), "reward": 0, "revision": 0, "advice_cap": 6, "clue_voice": "template"})


class SessionTests(VaultCase):
    def test_off_by_default_refuses_to_start_and_stores_nothing(self):
        with self.assertRaises(VaultStoreError) as caught:
            self.open(guild=G2)
        self.assertEqual(code_of(caught.exception), "vault.disabled")
        self.assertEqual(self.store.all("SELECT 1 FROM vault_sessions"), [])

    def test_start_creates_an_active_personal_vault(self):
        view = self.open()
        self.assertTrue(view["created"])
        self.assertEqual((view["guild_id"], view["user_id"], view["channel_id"], view["protocol"], view["length"]), (G, ALICE, CH, "standard", 4))
        self.assertEqual((view["status"], view["turn_budget"], view["turns_used"], view["turns_left"]), ("active", 9, 0, 9))
        self.assertEqual((view["local_date"], view["expires_at"], view["created_at"]), ("2026-10-15", MIDNIGHT, NOW))
        self.assertEqual((view["guesses"], view["partners"], view["clues"], view["reward"], view["open_digit"]), ([], [], [], None, None))
        secret = self.secret_of(view)
        self.assertEqual((len(secret), len(set(secret)), secret.isdigit()), (4, 4, True))

    def test_resume_returns_the_same_vault_without_regenerating(self):
        first = self.open(partner_ids=[self.ann])
        again = self.open(seed="a-different-seed", protocol="open", partner_ids=[self.bea], now=NOW + 3600)
        self.assertFalse(again["created"])
        self.assertEqual({k: first[k] for k in ("id", "protocol", "turn_budget", "expires_at", "partners", "clues")},
                         {k: again[k] for k in ("id", "protocol", "turn_budget", "expires_at", "partners", "clues")})
        self.assertEqual(self.secret_of(first), self.secret_of(again))
        self.assertEqual(len(self.store.all("SELECT 1 FROM vault_sessions")), 1)

    def test_progress_survives_a_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vault.sqlite3"
            store = Store(path)
            w = store.create_space(G, "W", "world")
            ann = store.create_character(G, w, "Ann")
            store.bind_channel(G, CH, w)
            store.set_cast(CH, None, [ann])
            store.set_vault_settings(G, enabled=True)
            view = store.open_personal_vault(G, ALICE, CH, partner_ids=[ann], now=NOW, seed="persist")
            secret = store.one("SELECT secret FROM vault_sessions")["secret"]
            wrong = [c for c in all_secrets(4) if c != secret][0]
            store.submit_vault_guess(G, ALICE, view["id"], wrong, now=NOW)
            store.close()
            store = Store(path)
            try:
                back = store.vault_session(G, ALICE, view["id"], now=NOW + 60)
                self.assertEqual((back["turns_used"], [g["guess"] for g in back["guesses"]], back["clues"]), (1, [wrong], view["clues"]))
                self.assertEqual(store.one("SELECT secret FROM vault_sessions")["secret"], secret)
            finally:
                store.close()

    def test_one_vault_per_member_and_guild(self):
        a, b = self.open(user=ALICE), self.open(user=BOB, seed="seed-2")
        self.assertNotEqual(a["id"], b["id"])
        other = self.store.create_space(G2, "W2", "world")
        self.store.bind_channel(G2, CH + 1, other)
        self.store.set_vault_settings(G2, enabled=True)
        c = self.store.open_personal_vault(G2, ALICE, CH + 1, now=NOW, seed="seed-1")
        self.assertNotEqual(a["id"], c["id"])
        self.assertEqual(c["guild_id"], G2)

    def test_same_seed_gives_the_same_secret_and_clues(self):
        a = self.open(partner_ids=[self.ann, self.bea], seed="fixed")
        b = self.open(user=BOB, partner_ids=[self.ann, self.bea], seed="fixed")
        self.assertEqual(self.secret_of(a), self.secret_of(b))
        for position in (1, 2):
            self.store.reveal_vault_clue(G, ALICE, a["id"], position, now=NOW)
            self.store.reveal_vault_clue(G, BOB, b["id"], position, now=NOW)
        a, b = (self.store.vault_session(G, u, v["id"], now=NOW) for u, v in ((ALICE, a), (BOB, b)))
        self.assertEqual([c["constraint"] for c in a["clues"]], [c["constraint"] for c in b["clues"]])
        other = self.open(user=7, seed="not-fixed")
        self.assertNotEqual(self.secret_of(a), self.secret_of(other))

    def test_timezone_change_keeps_the_stored_date_and_expiry(self):
        """The date and expiry are decided at creation; a later server timezone change moves neither."""
        self.store.set_guild_timezone(G, "Asia/Seoul")
        late = NOW + 8 * 3600  # 20:00 UTC: already 16 October in Seoul, still 15 October in New York
        first = self.open(now=late)
        self.assertEqual(first["local_date"], "2026-10-16")
        seoul_midnight = datetime(2026, 10, 17, 0, 0, tzinfo=ZoneInfo("Asia/Seoul")).timestamp()
        self.assertEqual(first["expires_at"], seoul_midnight)
        self.store.set_guild_timezone(G, "America/New_York")
        again = self.open(now=late + 60)
        self.assertEqual((again["id"], again["local_date"], again["expires_at"], again["created"]), (first["id"], "2026-10-16", seoul_midnight, False))
        self.assertEqual(self.guess(first, self.wrong(first)[0], now=late + 60)["session"]["status"], "active")

    def test_expiry_marks_old_vault_expired_and_starts_a_new_one(self):
        first = self.open(partner_ids=[self.ann])
        second = self.open(now=MIDNIGHT, seed="seed-2")
        self.assertTrue(second["created"])
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(second["local_date"], "2026-10-16")
        old = self.store.vault_session(G, ALICE, first["id"], now=MIDNIGHT)
        self.assertEqual((old["status"], old["secret"]), ("expired", self.secret_of(first)))

    def test_expire_sweep_only_touches_active_sessions_of_the_guild(self):
        active, solved = self.open(), self.open(user=BOB, seed="b")
        self.guess(solved, self.secret_of(solved), user=BOB)
        self.assertEqual(self.store.expire_vault_sessions(G, now=MIDNIGHT - 1), 0)
        self.assertEqual(self.store.expire_vault_sessions(G2, now=MIDNIGHT + 1), 0)
        self.assertEqual(self.store.expire_vault_sessions(G, now=MIDNIGHT), 1)
        self.assertEqual(self.store.expire_vault_sessions(G, now=MIDNIGHT), 0)
        self.assertEqual(self.store.vault_session(G, ALICE, active["id"], now=MIDNIGHT)["status"], "expired")
        self.assertEqual(self.store.vault_session(G, BOB, solved["id"], now=MIDNIGHT)["status"], "solved")

    def test_a_failed_vault_cannot_be_reset_the_same_day(self):
        view = self.open()
        self.fail(view)
        again = self.open(seed="new-secret-please", now=NOW + 600)
        self.assertEqual((again["id"], again["status"], again["created"]), (view["id"], "failed", False))
        self.assertEqual(self.secret_of(again), self.secret_of(view))
        self.assertEqual(len(self.store.all("SELECT 1 FROM vault_sessions")), 1)

    def test_a_solved_vault_is_not_replaced_the_same_day(self):
        view = self.open()
        self.guess(view, self.secret_of(view))
        again = self.open(seed="again", now=NOW + 600)
        self.assertEqual((again["id"], again["status"]), (view["id"], "solved"))

    def test_a_failed_vault_is_replaced_by_a_new_one_next_day(self):
        view = self.open()
        self.fail(view)
        nxt = self.open(now=MIDNIGHT + 5, seed="tomorrow")
        self.assertTrue(nxt["created"])
        self.assertEqual((nxt["local_date"], nxt["status"]), ("2026-10-16", "active"))

    def test_switching_the_feature_off_keeps_an_open_vault_playable_and_paying(self):
        self.store.set_vault_settings(G, reward=20)
        view = self.open()
        self.store.set_vault_settings(G, enabled=False)
        self.assertEqual(self.guess(view, self.wrong(view)[0])["session"]["status"], "active")
        self.assertEqual(self.open(now=NOW + 5)["id"], view["id"])
        won = self.guess(view, self.secret_of(view))["session"]
        self.assertEqual((won["status"], won["reward"]["amount"], self.store.balance(G, ALICE)), ("solved", 20, 20))
        with self.assertRaises(VaultStoreError) as caught:
            self.open(user=BOB, seed="b")
        self.assertEqual(code_of(caught.exception), "vault.disabled")

    def test_a_protocol_removed_after_start_stays_playable_for_the_open_vault(self):
        view = self.open(protocol="open")
        self.store.set_vault_settings(G, protocols=["standard"])
        self.assertEqual(self.guess(view, self.secret_of(view))["session"]["status"], "solved")
        with self.assertRaises(VaultStoreError) as caught:
            self.open(user=BOB, protocol="open", seed="b")
        self.assertEqual(code_of(caught.exception), "vault.protocol_not_allowed")

    def test_timezone_moved_back_onto_a_used_date_returns_the_existing_vault(self):
        """The new day computes a date that an earlier, expired vault already holds: no duplicate row, that row comes back."""
        self.store.set_guild_timezone(G, "Asia/Seoul")
        late = NOW + 8 * 3600  # 16 October in Seoul
        first = self.open(now=late)
        self.assertEqual(first["local_date"], "2026-10-16")
        after = datetime(2026, 10, 17, 0, 0, tzinfo=ZoneInfo("Asia/Seoul")).timestamp() + 60  # first has expired; Seoul says 17 October
        self.store.set_guild_timezone(G, "America/Los_Angeles")  # 16 October 08:01 there: the date the expired vault holds
        again = self.open(now=after, seed="clash")
        self.assertEqual((again["id"], again["local_date"], again["status"]), (first["id"], "2026-10-16", "expired"))
        self.assertEqual(len(self.store.all("SELECT 1 FROM vault_sessions")), 1)

    def test_session_is_readable_only_by_its_owner_in_its_guild(self):
        view = self.open()
        for guild, user in ((G, BOB), (G2, ALICE)):
            with self.assertRaises(VaultStoreError) as caught:
                self.store.vault_session(guild, user, view["id"], now=NOW)
            self.assertEqual(code_of(caught.exception), "vault.not_found")
        with self.assertRaises(VaultStoreError):
            self.store.vault_session(G, ALICE, view["id"] + 99, now=NOW)

    def test_new_tables_exist(self):
        names = {r["name"] for r in self.store.all("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertGreaterEqual(names, {"vault_settings", "vault_sessions", "vault_stages", "vault_actions", "vault_clues", "vault_reward_claims"})


class ProtocolTests(VaultCase):
    def test_standard_and_open_are_allowed_by_default(self):
        self.assertEqual(self.open()["protocol"], "standard")
        view = self.open(user=BOB, protocol="open", seed="o")
        self.assertEqual((view["protocol"], view["turn_budget"]), ("open", DEFAULT_TURNS["open"]))

    def test_open_shows_one_digit_of_the_secret_standard_none(self):
        view = self.open(protocol="open")
        self.assertIn(view["open_digit"], self.secret_of(view))
        self.assertEqual(len(view["open_digit"]), 1)
        self.assertIsNone(self.open(user=BOB, seed="s")["open_digit"])

    def test_total_match_and_position_only_need_admin_opt_in(self):
        for protocol in ("total_match", "position_only", "nonsense"):
            with self.subTest(protocol=protocol):
                with self.assertRaises(VaultStoreError) as caught:
                    self.open(protocol=protocol)
                self.assertEqual(code_of(caught.exception), "vault.protocol_not_allowed")
        self.assertEqual(self.store.all("SELECT 1 FROM vault_sessions"), [])
        self.store.set_vault_settings(G, protocols=["standard", "open", "total_match", "position_only"])
        tm = self.open(protocol="total_match")
        po = self.open(user=BOB, protocol="position_only", seed="p")
        secret = self.secret_of(tm)
        code = self.wrong(tm)[0]
        self.assertEqual(self.guess(tm, code)["feedback"], feedback("total_match", secret, code))
        self.assertEqual(len(self.guess(po, self.wrong(po)[0], user=BOB)["feedback"]), 1)

    def test_a_protocol_the_admin_removed_cannot_start(self):
        self.store.set_vault_settings(G, protocols=["standard"])
        with self.assertRaises(VaultStoreError):
            self.open(protocol="open")

    def test_budget_comes_from_guild_settings_and_is_fixed_at_creation(self):
        self.store.set_vault_settings(G, turns={"standard": 3})
        view = self.open()
        self.assertEqual(view["turn_budget"], 3)
        self.store.set_vault_settings(G, turns={"standard": 5})
        self.assertEqual(self.store.vault_session(G, ALICE, view["id"], now=NOW)["turn_budget"], 3)
        self.assertEqual(self.open(user=BOB, seed="b")["turn_budget"], 5)

    def test_total_match_budget_can_be_raised_by_the_admin(self):
        self.store.set_vault_settings(G, protocols=["standard", "total_match"], turns={"total_match": 15})
        self.assertEqual(self.open(protocol="total_match")["turn_budget"], 15)


class PartnerTests(VaultCase):
    def setUp(self):
        super().setUp()
        s = self.store
        s.add_favorite(G, ALICE, self.cy)      # in the world, not in the cast
        s.add_favorite(G, ALICE, self.far)     # in a world the channel cannot see
        s.add_favorite(G, ALICE, self.ann)     # in both
        self.stranger = s.create_character(G, self.w, "Dee")  # eligible in the world, neither favorite nor cast
        s.create_space(G2, "Other", "world")
        self.foreign = s.create_character(G2, s.one("SELECT id FROM spaces WHERE guild_id=?", (G2,))["id"], "Zed")

    def choices(self, user=ALICE, channel=CH):
        return {c["character_id"]: c for c in self.store.vault_partner_choices(G, user, channel)}

    def test_choices_are_eligible_favorites_plus_the_channel_cast(self):
        got = self.choices()
        self.assertEqual(set(got), {self.ann, self.bea, self.cy})
        self.assertEqual((got[self.ann]["source"], got[self.cy]["source"], got[self.bea]["source"]), ("favorite", "favorite", "cast"))
        self.assertEqual(got[self.bea]["name"], "Bea")
        self.assertEqual(set(self.choices(user=BOB)), {self.ann, self.bea})

    def test_unbound_channel_has_no_choices(self):
        self.assertEqual(self.store.vault_partner_choices(G, ALICE, 999), [])

    def test_no_partner_means_no_clues(self):
        view = self.open()
        self.assertEqual((view["partners"], view["clues"]), ([], []))

    def test_one_partner_holds_both_clues(self):
        view = self.open(partner_ids=[self.bea])
        self.assertEqual(view["partners"], [{"character_id": self.bea, "name": "Bea"}])
        self.assertEqual([(c["character_id"], c["name"]) for c in view["clues"]], [(self.bea, "Bea")] * 2)

    def test_two_partners_hold_one_clue_each(self):
        view = self.open(partner_ids=[self.ann, self.cy])
        self.assertEqual([p["character_id"] for p in view["partners"]], [self.ann, self.cy])
        self.assertEqual([c["character_id"] for c in view["clues"]], [self.ann, self.cy])

    def test_duplicate_partner_ids_collapse(self):
        view = self.open(partner_ids=[self.ann, self.ann])
        self.assertEqual(len(view["partners"]), 1)
        self.assertEqual(len(view["clues"]), 2)

    def test_more_than_two_partners_is_refused(self):
        with self.assertRaises(VaultStoreError) as caught:
            self.open(partner_ids=[self.ann, self.bea, self.cy])
        self.assertEqual(code_of(caught.exception), "vault.partners_too_many")
        self.assertEqual(self.store.all("SELECT 1 FROM vault_sessions"), [])

    def test_inaccessible_partners_are_refused_and_nothing_is_stored(self):
        for bad in (self.stranger, self.far, self.foreign, 99999):
            with self.subTest(character=bad):
                with self.assertRaises(VaultStoreError) as caught:
                    self.open(partner_ids=[self.ann, bad])
                self.assertEqual(code_of(caught.exception), "vault.partner_unavailable")
        self.assertEqual(self.store.all("SELECT 1 FROM vault_sessions"), [])
        self.assertEqual(self.store.all("SELECT 1 FROM vault_clues"), [])

    def test_another_members_favorite_is_not_mine(self):
        self.store.add_favorite(G, BOB, self.stranger)
        with self.assertRaises(VaultStoreError):
            self.open(partner_ids=[self.stranger])

    def test_archived_favorite_only_while_the_server_allows_it(self):
        self.store.archive_character(G, self.cy, True)
        self.assertNotIn(self.cy, self.choices())
        with self.assertRaises(VaultStoreError):
            self.open(partner_ids=[self.cy])
        self.store.set_archived_favorites(G, True)
        self.assertIn(self.cy, self.choices())
        self.assertEqual(self.open(partner_ids=[self.cy])["partners"][0]["character_id"], self.cy)

    def test_archived_cast_member_is_not_offered(self):
        self.store.archive_character(G, self.bea, True)
        self.assertNotIn(self.bea, self.choices())
        with self.assertRaises(VaultStoreError):
            self.open(user=BOB, partner_ids=[self.bea])

    def test_clues_start_hidden_with_no_constraint_data(self):
        view = self.open(partner_ids=[self.ann, self.bea])
        self.assertEqual([(c["position"], c["character_id"], c["revealed"]) for c in view["clues"]], [(1, self.ann, False), (2, self.bea, False)])
        for clue in view["clues"]:
            self.assertNotIn("constraint", clue)
        self.assertFalse(any(isinstance(c, CLUE_TYPES) for c in view["constraints"]))
        self.assertEqual(self.store.vault_session(G, ALICE, view["id"], now=NOW)["clues"], view["clues"])

    def test_revealed_clues_are_true_facts_and_stay_the_same_on_reopen(self):
        view = self.open(partner_ids=[self.ann, self.bea])
        entry = self.store.reveal_vault_clue(G, ALICE, view["id"], 2, now=NOW + 5)
        self.assertEqual((entry["position"], entry["character_id"], entry["revealed"], entry["revealed_at"]), (2, self.bea, True, NOW + 5))
        self.assertIsInstance(entry["constraint"], CLUE_TYPES)
        self.assertTrue(entry["constraint"].holds(self.secret_of(view)))
        after = self.store.vault_session(G, ALICE, view["id"], now=NOW + 6)
        self.assertEqual([c["revealed"] for c in after["clues"]], [False, True])
        self.assertNotIn("constraint", after["clues"][0])
        self.assertIn(entry["constraint"], after["constraints"])
        again = self.open(seed="other", now=NOW + 7, partner_ids=[self.cy])
        self.assertEqual(again["clues"], after["clues"])

    def test_reveal_is_idempotent_and_keeps_the_first_time(self):
        view = self.open(partner_ids=[self.ann])
        first = self.store.reveal_vault_clue(G, ALICE, view["id"], 1, now=NOW + 1)
        second = self.store.reveal_vault_clue(G, ALICE, view["id"], 1, now=NOW + 99)
        self.assertEqual(first, second)
        self.assertEqual(second["revealed_at"], NOW + 1)
        self.assertEqual(self.store.vault_session(G, ALICE, view["id"], now=NOW)["turns_used"], 0)
        self.assertEqual(len(self.store.all("SELECT 1 FROM vault_clues WHERE revealed_at IS NOT NULL")), 1)

    def test_revealing_narrows_the_public_constraints_without_losing_the_secret(self):
        view = self.open(partner_ids=[self.ann])
        before = filter_candidates(self.store.vault_session(G, ALICE, view["id"], now=NOW)["constraints"], 4)
        for position in (1, 2):
            self.store.reveal_vault_clue(G, ALICE, view["id"], position, now=NOW)
        after = filter_candidates(self.store.vault_session(G, ALICE, view["id"], now=NOW)["constraints"], 4)
        self.assertIn(self.secret_of(view), after)
        self.assertLess(len(after), len(before))

    def test_unrevealed_clue_data_is_not_in_any_view(self):
        view = self.open(partner_ids=[self.ann, self.bea])
        result = self.guess(view, self.wrong(view)[0])
        for out in (view, result, self.store.vault_session(G, ALICE, view["id"], now=NOW)):
            for clue in out["clues"] if "clues" in out else out["session"]["clues"]:
                self.assertFalse(clue["revealed"])
                self.assertNotIn("constraint", clue)
        stored = [dict(r) for r in self.store.all("SELECT kind FROM vault_clues")]
        for row in stored:
            self.assertNotIn(row["kind"], repr(view["clues"]))

    def test_reveal_refusals(self):
        view = self.open(partner_ids=[self.ann])
        for bad in (0, 3, -1, True, "1", 1.0, None):
            with self.subTest(position=bad):
                with self.assertRaises(VaultStoreError) as caught:
                    self.store.reveal_vault_clue(G, ALICE, view["id"], bad, now=NOW)
                self.assertEqual(code_of(caught.exception), "vault.clue_unknown")
        for guild, user in ((G, BOB), (G2, ALICE)):
            with self.assertRaises(VaultStoreError) as caught:
                self.store.reveal_vault_clue(guild, user, view["id"], 1, now=NOW)
            self.assertEqual(code_of(caught.exception), "vault.not_found")
        solo = self.open(user=BOB, seed="b")
        with self.assertRaises(VaultStoreError) as caught:
            self.store.reveal_vault_clue(G, BOB, solo["id"], 1, now=NOW)
        self.assertEqual(code_of(caught.exception), "vault.clue_unknown")
        self.assertEqual(self.store.all("SELECT 1 FROM vault_clues WHERE revealed_at IS NOT NULL"), [])

    def test_reveal_needs_an_active_vault(self):
        won, lost, late = (self.open(user=u, seed=str(u), partner_ids=[self.ann]) for u in (30, 31, 32))
        self.guess(won, self.secret_of(won), user=30)
        for code in self.wrong(lost, 9):
            self.guess(lost, code, user=31)
        for view, user, now in ((won, 30, NOW), (lost, 31, NOW), (late, 32, MIDNIGHT)):
            with self.assertRaises(VaultStoreError) as caught:
                self.store.reveal_vault_clue(G, user, view["id"], 1, now=now)
            self.assertEqual(code_of(caught.exception), "vault.not_active")
        self.assertEqual(self.store.all("SELECT 1 FROM vault_clues WHERE revealed_at IS NOT NULL"), [])

    def test_partner_ids_must_be_plain_ints(self):
        for bad in (True, "1", 1.5, None):
            with self.subTest(partner=bad):
                with self.assertRaises(UserError) as caught:
                    self.open(partner_ids=[bad])
                self.assertEqual(code_of(caught.exception), "vault.partner_unavailable")
        self.assertEqual(self.store.all("SELECT 1 FROM vault_sessions"), [])

    def test_deleted_character_shows_no_name(self):
        view = self.open(partner_ids=[self.ann])
        self.store.delete_character(G, self.ann, self.store.owner_revision(G, "character", self.ann))
        got = self.store.vault_session(G, ALICE, view["id"], now=NOW)
        self.assertEqual(got["partners"], [{"character_id": self.ann, "name": None}])
        self.assertEqual([(c["character_id"], c["name"]) for c in got["clues"]], [(self.ann, None)] * 2)
        revealed = self.store.reveal_vault_clue(G, ALICE, view["id"], 1, now=NOW)
        self.assertIsNone(revealed["name"])
        self.assertNotIn("Deleted", repr(got))

    def test_choices_in_a_thread_use_the_parent_channels_cast(self):
        thread = 555
        got = {c["character_id"] for c in self.store.vault_partner_choices(G, BOB, thread, parent_id=CH)}
        self.assertEqual(got, {self.ann, self.bea})
        self.assertEqual(self.store.vault_partner_choices(G, BOB, thread), [])
        view = self.store.open_personal_vault(G, BOB, thread, parent_id=CH, partner_ids=[self.bea], now=NOW, seed="t")
        self.assertEqual(view["partners"], [{"character_id": self.bea, "name": "Bea"}])
        self.store.set_cast(thread, CH, [self.stranger])  # a thread's own cast replaces the parent's
        got = {c["character_id"] for c in self.store.vault_partner_choices(G, 77, thread, parent_id=CH)}
        self.assertEqual(got, {self.stranger})


class GuessTests(VaultCase):
    def test_invalid_guesses_are_free_and_coded(self):
        view = self.open()
        for bad, code in (("123", "vault.guess_length"), ("12345", "vault.guess_length"), ("12a4", "vault.guess_not_digits"),
                          ("1123", "vault.guess_repeated_digit"), ("", "vault.guess_length")):
            with self.subTest(guess=bad):
                with self.assertRaises(VaultStoreError) as caught:
                    self.guess(view, bad)
                self.assertEqual(code_of(caught.exception), code)
        again = self.store.vault_session(G, ALICE, view["id"], now=NOW)
        self.assertEqual((again["turns_used"], again["guesses"]), (0, []))
        self.assertEqual(self.store.all("SELECT 1 FROM vault_actions"), [])

    def test_repeated_guess_is_free(self):
        view = self.open()
        code = self.wrong(view)[0]
        self.guess(view, code)
        with self.assertRaises(VaultStoreError) as caught:
            self.guess(view, code)
        self.assertEqual(code_of(caught.exception), "vault.guess_duplicate")
        again = self.store.vault_session(G, ALICE, view["id"], now=NOW)
        self.assertEqual((again["turns_used"], len(again["guesses"])), (1, 1))

    def test_valid_guess_spends_a_turn_and_records_the_feedback(self):
        view = self.open()
        secret = self.secret_of(view)
        code = self.wrong(view)[0]
        result = self.guess(view, code)
        self.assertEqual((result["number"], result["guess"], result["won"], result["replayed"]), (1, code, False, False))
        self.assertEqual(result["feedback"], feedback("standard", secret, code))
        session = result["session"]
        self.assertEqual((session["turns_used"], session["turns_left"], session["status"]), (1, 8, "active"))
        self.assertEqual(session["guesses"], [{"number": 1, "guess": code, "feedback": result["feedback"], "won": False}])
        self.assertEqual(self.store.vault_session(G, ALICE, view["id"], now=NOW)["guesses"], session["guesses"])

    def test_guesses_keep_their_order_and_narrow_the_public_constraints(self):
        for protocol in ("standard", "total_match"):
            with self.subTest(protocol=protocol):
                self.store.set_vault_settings(G, protocols=["standard", "total_match"])
                view = self.open(user=10 + len(protocol), protocol=protocol, partner_ids=[self.ann], seed=protocol)
                secret = self.secret_of(view)
                tried = self.wrong(view, 3)
                for code in tried:
                    view = self.guess(view, code, user=view["user_id"])["session"]
                self.assertEqual([g["guess"] for g in view["guesses"]], tried)
                left = filter_candidates(view["constraints"], 4)
                self.assertIn(secret, left)
                for code in tried:
                    self.assertNotIn(code, left)

    def test_open_digit_is_a_public_constraint(self):
        view = self.open(protocol="open")
        left = filter_candidates(view["constraints"], 4)
        self.assertIn(self.secret_of(view), left)
        self.assertTrue(all(view["open_digit"] in c for c in left))

    def test_winning_solves_and_reveals_the_secret(self):
        view = self.open()
        secret = self.secret_of(view)
        self.guess(view, self.wrong(view)[0])
        result = self.guess(view, secret)
        self.assertTrue(result["won"])
        self.assertEqual(result["feedback"], (4, 0))
        done = result["session"]
        self.assertEqual((done["status"], done["secret"], done["turns_used"]), ("solved", secret, 2))
        self.assertTrue(done["guesses"][-1]["won"])

    def test_winning_on_the_last_turn_still_wins(self):
        self.store.set_vault_settings(G, turns={"standard": 2})
        view = self.open()
        self.guess(view, self.wrong(view)[0])
        self.assertEqual(self.guess(view, self.secret_of(view))["session"]["status"], "solved")

    def test_running_out_of_turns_fails_and_reveals_the_secret(self):
        view = self.open()
        result = self.fail(view)
        failed = result["session"]
        self.assertEqual((failed["status"], failed["turns_used"], failed["turns_left"], failed["secret"]), ("failed", 9, 0, self.secret_of(view)))
        self.assertFalse(result["won"])

    def test_finished_vault_takes_no_more_guesses(self):
        for outcome in ("won", "lost"):
            view = self.open(user=20 + len(outcome), seed=outcome)
            if outcome == "won":
                self.guess(view, self.secret_of(view), user=view["user_id"])
            else:
                for code in self.wrong(view, 9):
                    self.guess(view, code, user=view["user_id"])
            before = self.store.all("SELECT * FROM vault_actions WHERE session_id=?", (view["id"],))
            with self.assertRaises(VaultStoreError) as caught:
                self.guess(view, self.wrong(view, 1, skip=[g["guess"] for g in self.store.vault_session(G, view["user_id"], view["id"], now=NOW)["guesses"]])[0], user=view["user_id"])
            self.assertEqual(code_of(caught.exception), "vault.not_active")
            self.assertEqual(len(self.store.all("SELECT * FROM vault_actions WHERE session_id=?", (view["id"],))), len(before))

    def test_guess_after_the_day_ended_expires_the_vault_and_spends_nothing(self):
        view = self.open()
        with self.assertRaises(VaultStoreError) as caught:
            self.guess(view, self.wrong(view)[0], now=MIDNIGHT)
        self.assertEqual(code_of(caught.exception), "vault.not_active")
        done = self.store.vault_session(G, ALICE, view["id"], now=MIDNIGHT)
        self.assertEqual((done["status"], done["turns_used"], done["secret"]), ("expired", 0, self.secret_of(view)))

    def test_double_submission_with_the_same_action_key_spends_once(self):
        view = self.open()
        code = self.wrong(view)[0]
        first = self.guess(view, code, action_key="interaction-1")
        second = self.guess(view, code, action_key="interaction-1")
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])
        self.assertEqual({k: second[k] for k in ("number", "guess", "feedback", "won")}, {k: first[k] for k in ("number", "guess", "feedback", "won")})
        self.assertEqual(self.store.vault_session(G, ALICE, view["id"], now=NOW)["turns_used"], 1)
        self.assertEqual(len(self.store.all("SELECT 1 FROM vault_actions")), 1)

    def test_the_same_key_on_a_different_vault_is_independent(self):
        a, b = self.open(), self.open(user=BOB, seed="b")
        self.guess(a, self.wrong(a)[0], action_key="k")
        result = self.guess(b, self.wrong(b)[0], user=BOB, action_key="k")
        self.assertFalse(result["replayed"])
        self.assertEqual(len(self.store.all("SELECT 1 FROM vault_actions")), 2)

    def test_replayed_winning_guess_pays_and_solves_once(self):
        self.store.set_vault_settings(G, reward=40)
        view = self.open()
        win = self.guess(view, self.secret_of(view), action_key="w")
        again = self.guess(view, self.secret_of(view), action_key="w")
        self.assertTrue(win["won"] and again["won"] and again["replayed"])
        self.assertEqual(self.store.balance(G, ALICE), 40)
        self.assertEqual(len(self.ledger()), 1)

    def test_only_the_owner_can_guess_and_nothing_changes(self):
        view = self.open()
        for guild, user in ((G, BOB), (G2, ALICE)):
            with self.assertRaises(VaultStoreError) as caught:
                self.store.submit_vault_guess(guild, user, view["id"], self.wrong(view)[0], now=NOW)
            self.assertEqual(code_of(caught.exception), "vault.not_found")
        self.assertEqual(self.store.all("SELECT 1 FROM vault_actions"), [])
        self.assertEqual(self.store.vault_session(G, ALICE, view["id"], now=NOW)["turns_used"], 0)


class SecretHiddenTests(VaultCase):
    def test_secret_is_hidden_while_active_everywhere_a_caller_can_see(self):
        view = self.open(protocol="open", partner_ids=[self.ann, self.bea])
        secret = self.secret_of(view)
        result = self.guess(view, self.wrong(view)[0])
        outputs = [view, result, result["session"], self.store.vault_session(G, ALICE, view["id"], now=NOW),
                   self.store.vault_history(G, ALICE, now=NOW), self.store.vault_partner_choices(G, ALICE, CH), self.store.vault_settings(G)]
        for out in outputs:
            self.assertNotIn(secret, repr(out))
        self.assertIsNone(view["secret"])
        self.assertIsNone(result["session"]["secret"])
        self.assertIsNone(self.store.vault_history(G, ALICE, now=NOW)[0]["secret"])

    def test_errors_never_carry_the_secret(self):
        view = self.open()
        secret = self.secret_of(view)
        errors = []
        for bad in ("1", "abcd", "1123", self.wrong(view)[0]):
            try:
                self.guess(view, bad)
                self.guess(view, bad)
            except VaultStoreError as error:
                errors.append(error)
        with self.assertRaises(VaultStoreError) as caught:
            self.store.vault_session(G, BOB, view["id"], now=NOW)
        errors.append(caught.exception)
        self.assertTrue(errors)
        for error in errors:
            self.assertNotIn(secret, f"{error} {error.params!r} {error!r}")

    def test_secret_appears_once_terminal(self):
        view = self.open()
        self.guess(view, self.secret_of(view))
        self.assertEqual(self.store.vault_history(G, ALICE, now=NOW)[0]["secret"], self.secret_of(view))


class RewardTests(VaultCase):
    def setUp(self):
        super().setUp()
        self.store.set_vault_settings(G, reward=50)

    def test_a_win_pays_through_the_ledger_exactly_once(self):
        view = self.open()
        done = self.guess(view, self.secret_of(view))["session"]
        self.assertEqual(self.store.balance(G, ALICE), 50)
        rows = self.ledger()
        self.assertEqual([(r["user_id"], r["amount"], r["source"], r["actor_id"]) for r in rows], [(ALICE, 50, "vault", ALICE)])
        self.assertEqual(done["reward"], {"amount": 50, "ledger_id": rows[0]["id"]})
        self.assertEqual(len(self.store.all("SELECT 1 FROM vault_reward_claims")), 1)
        self.assertEqual(self.store.vault_session(G, ALICE, view["id"], now=NOW)["reward"], done["reward"])
        with self.assertRaises(VaultStoreError):
            self.guess(view, self.secret_of(view))
        self.open(now=NOW + 5)
        self.assertEqual((self.store.balance(G, ALICE), len(self.ledger())), (50, 1))

    def test_each_win_on_its_own_day_pays_its_own_reward(self):
        first = self.open()
        self.guess(first, self.secret_of(first))
        second = self.open(now=MIDNIGHT + 1, seed="d2")
        self.guess(second, self.secret_of(second), now=MIDNIGHT + 1)
        self.assertEqual((self.store.balance(G, ALICE), len(self.ledger()), len(self.store.all("SELECT 1 FROM vault_reward_claims"))), (100, 2, 2))

    def test_failure_and_expiry_pay_nothing(self):
        failed = self.open()
        self.fail(failed)
        expired = self.open(user=BOB, seed="b")
        self.store.expire_vault_sessions(G, now=MIDNIGHT)
        self.assertEqual((self.store.balance(G, ALICE), self.store.balance(G, BOB), self.ledger()), (0, 0, []))
        self.assertEqual(self.store.vault_session(G, ALICE, failed["id"], now=NOW)["reward"], None)
        self.assertEqual(self.store.vault_session(G, BOB, expired["id"], now=MIDNIGHT)["reward"], None)

    def test_reward_zero_by_default_writes_no_ledger_row(self):
        self.store.set_vault_settings(G, reward=0)
        view = self.open()
        done = self.guess(view, self.secret_of(view))["session"]
        self.assertEqual((done["status"], done["reward"], self.ledger(), self.store.balance(G, ALICE)), ("solved", None, [], 0))
        self.assertEqual(self.store.all("SELECT 1 FROM vault_reward_claims"), [])

    def test_reward_is_per_guild(self):
        self.assertEqual(self.store.vault_settings(G2)["reward"], 0)
        view = self.open()
        self.guess(view, self.secret_of(view))
        self.assertEqual((self.store.balance(G2, ALICE), self.ledger(G2)), (0, []))

    def test_the_reward_amount_is_the_one_in_force_at_the_win(self):
        view = self.open()
        self.store.set_vault_settings(G, reward=75)
        self.assertEqual(self.guess(view, self.secret_of(view))["session"]["reward"]["amount"], 75)

    def test_a_reward_refused_by_the_balance_cap_leaves_the_win_unpaid(self):
        cap = self.store.MAX_CURRENCY_BALANCE
        self.store.db.execute("INSERT INTO currency_balances(guild_id,user_id,balance,updated_at) VALUES(?,?,?,?)", (G, ALICE, cap - 10, NOW))
        self.store.db.commit()
        view = self.open()
        result = self.guess(view, self.secret_of(view))
        done = result["session"]
        self.assertEqual((done["status"], done["reward"], result["won"]), ("solved", None, True))
        self.assertTrue(done["reward_refused"])
        self.assertTrue(result["reward_refused"])
        self.assertEqual((self.store.balance(G, ALICE), self.ledger(), self.store.all("SELECT 1 FROM vault_reward_claims")), (cap - 10, [], []))
        self.assertFalse(self.store.vault_history(G, ALICE, now=NOW)[0]["reward"])

    def test_a_paid_win_is_not_marked_refused(self):
        view = self.open()
        result = self.guess(view, self.secret_of(view))
        self.assertFalse(result["session"].get("reward_refused"))
        self.assertFalse(result.get("reward_refused"))

    def test_reversal_is_flagged_in_the_view_and_history(self):
        view = self.open()
        done = self.guess(view, self.secret_of(view))["session"]
        self.assertFalse(done["reward_reversed"])
        self.assertFalse(self.store.vault_history(G, ALICE, now=NOW)[0]["reward_reversed"])
        self.store.reverse_entry(G, done["reward"]["ledger_id"], 1, "undo")
        after = self.store.vault_session(G, ALICE, view["id"], now=NOW)
        self.assertEqual((after["reward"]["amount"], after["reward_reversed"]), (50, True))
        summary = self.store.vault_history(G, ALICE, now=NOW)[0]
        self.assertEqual((summary["reward"], summary["reward_reversed"]), (50, True))
        self.assertEqual(self.store.balance(G, ALICE), 0)
        self.assertFalse(self.open(user=BOB, seed="b")["reward_reversed"])

    def test_reversal_takes_it_back_and_does_not_repay(self):
        view = self.open()
        done = self.guess(view, self.secret_of(view))["session"]
        self.store.reverse_entry(G, done["reward"]["ledger_id"], 1, "undo")
        self.assertEqual(self.store.balance(G, ALICE), 0)
        self.open(now=NOW + 10)
        with self.assertRaises(VaultStoreError):
            self.guess(view, self.secret_of(view), now=NOW + 10)
        self.assertEqual(self.store.balance(G, ALICE), 0)
        self.assertEqual(len(self.store.all("SELECT 1 FROM currency_ledger WHERE guild_id=?", (G,))), 2)


class HistoryTests(VaultCase):
    def test_lists_own_sessions_newest_first_with_secrets_only_when_terminal(self):
        d1 = self.open()
        self.guess(d1, self.secret_of(d1))
        d2 = self.open(now=MIDNIGHT + 1, seed="d2")
        self.guess(d2, self.wrong(d2)[0], now=MIDNIGHT + 1)
        d3 = self.open(now=MIDNIGHT + DAY + 1, seed="d3")
        rows = self.store.vault_history(G, ALICE, now=MIDNIGHT + DAY + 1)
        self.assertEqual([r["id"] for r in rows], [d3["id"], d2["id"], d1["id"]])
        self.assertEqual([r["status"] for r in rows], ["active", "expired", "solved"])
        self.assertEqual([r["secret"] for r in rows], [None, self.secret_of(d2), self.secret_of(d1)])
        self.assertEqual([r["local_date"] for r in rows], ["2026-10-17", "2026-10-16", "2026-10-15"])
        self.assertEqual((rows[1]["turns_used"], rows[1]["turn_budget"], rows[1]["protocol"]), (1, 9, "standard"))

    def test_limit_applies_and_only_own_vaults_in_this_guild_show(self):
        days = [self.open(now=NOW + n * DAY, seed=f"s{n}") for n in range(4)]
        self.open(user=BOB, seed="bob")
        self.assertEqual([r["id"] for r in self.store.vault_history(G, ALICE, limit=2, now=NOW + 4 * DAY)], [days[3]["id"], days[2]["id"]])
        self.assertEqual(len(self.store.vault_history(G, BOB, now=NOW)), 1)
        self.assertEqual(self.store.vault_history(G2, ALICE, now=NOW), [])
        self.assertEqual(self.store.vault_history(G, 999, now=NOW), [])

    def test_summary_shows_the_reward_paid(self):
        self.store.set_vault_settings(G, reward=30)
        view = self.open()
        self.guess(view, self.secret_of(view))
        self.assertEqual(self.store.vault_history(G, ALICE, now=NOW)[0]["reward"], 30)


class MigrationTests(unittest.TestCase):
    VAULT_TABLES = ("vault_reward_claims", "vault_actions", "vault_clues", "vault_stages", "vault_sessions", "vault_settings")

    def v24_file(self, folder):
        path = Path(folder) / "v24.sqlite3"
        store = Store(path)
        w = store.create_space(G, "W", "world")
        store.create_character(G, w, "Ann")
        store.change_balance(G, ALICE, 120, "start", 1)
        store.close()
        with closing(sqlite3.connect(path)) as db, db:
            for table in self.VAULT_TABLES:
                db.execute(f"DROP TABLE IF EXISTS {table}")
            db.execute("PRAGMA user_version=24")
        return path

    def tables(self, path):
        with closing(sqlite3.connect(path)) as db:
            return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def test_v24_file_upgrades_to_25_with_a_backup_and_keeps_its_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.v24_file(folder)
            self.assertFalse(set(self.VAULT_TABLES) & self.tables(path))
            store = Store(path)
            try:
                self.assertEqual(store.one("PRAGMA user_version")[0], 27)
                self.assertEqual(store.balance(G, ALICE), 120)
                self.assertEqual(store.vault_settings(G)["enabled"], False)
                self.assertEqual(store.all("SELECT 1 FROM vault_sessions"), [])
            finally:
                store.close()
            self.assertTrue(set(self.VAULT_TABLES) <= self.tables(path))
            backups = list(Path(folder).glob("v24.sqlite3.pre-v25-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            with closing(sqlite3.connect(backups[0])) as old:
                self.assertEqual(old.execute("PRAGMA user_version").fetchone()[0], 24)
            self.assertFalse(set(self.VAULT_TABLES) & self.tables(backups[0]))

    def test_upgrade_is_idempotent_and_makes_one_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.v24_file(folder)
            Store(path).close()
            Store(path).close()
            self.assertEqual(len(list(Path(folder).glob("*.pre-v25-*.sqlite3"))), 1)

    def test_fresh_store_is_v25_and_a_vault_works_after_upgrade(self):
        self.assertEqual(Store().one("PRAGMA user_version")[0], 27)
        with tempfile.TemporaryDirectory() as folder:
            store = Store(self.v24_file(folder))
            try:
                w = store.one("SELECT id FROM spaces")["id"]
                store.bind_channel(G, CH, w)
                store.set_vault_settings(G, enabled=True)
                view = store.open_personal_vault(G, ALICE, CH, now=NOW, seed="x")
                self.assertEqual(view["status"], "active")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
