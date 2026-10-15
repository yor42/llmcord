"""FEAT-68 (D43): the Vault section of the dashboard Games tab and the Shop tab's vault tool kind. Written before the implementation; they fail until it exists.

API CONTRACT, pure helpers (no NiceGUI import needed) in ``llmcord_core/games_ui.py`` unless noted; the NiceGUI ``vault_section(ctx)`` is a thin
shell over them (Experimental label, default off, savebar tracks per sub-section) and is reached from ``games_panel`` after the game sections.

  vault_form(store, gid) -> dict   the personal-vault form values, exactly these keys, all from ``store.vault_settings(gid)``:
      enabled (bool), protocols (list, PROTOCOLS order), turns ({protocol: int} for all four protocols, defaults filled in), reward (int),
      advice_cap (int), clue_voice ('template' | 'llm'), revision (int)
  operation_form(store, gid) -> dict   exactly the keys of ``store.vault_operation_settings(gid)``: enabled, channel_id (int | None), week_start (0=Monday..6),
      stage_protocols (3 protocols), contribution_reward, team_reward, min_guesses, support_reward, revision
  save_vault(store, gid, form, base) -> dict   ``base`` is the dict ``vault_form`` returned when the page was built; ``form`` is the user's values (same
      keys; ``revision`` is ignored). Only keys whose value differs from ``base`` are sent to ``store.set_vault_settings(gid, ..., expected_revision=
      base['revision'])``; for ``turns`` only the protocols that differ. Whole floats from number inputs (3.0) are sent as ints, anything else as-is
      so the store's message is shown. Nothing changed -> no store call, returns ``store.vault_settings(gid)``. Returns the store's new settings.
      A stale revision raises ``ConflictError`` (the page shows its existing conflict UX); a bad value raises the store's ``UserError`` (the page
      shows ``t.error(err)`` as a toast) and nothing at all is written, not even the other changed fields.
  save_operation(store, gid, form, base) -> dict   the same for ``store.set_vault_operation_settings`` / ``operation_form``
  vault_channel_options(channel_names, thread_names, current=None) -> {channel_id: label}   the operation channel picker: ``channel_names`` is
      ``ctx.channel_names`` ({id: '#name'}); any id that is also in ``thread_names`` is dropped (threads cannot be the operation channel); a
      ``current`` id that is not listed (deleted channel, or a thread saved earlier) is added as ``str(id)`` so the stored value stays visible.
  week_start_options(t) -> {0: 'Monday', ..., 6: 'Sunday'}   translated day names for the week start picker
  vault_protocol_options(t) -> {protocol: label}   the four PROTOCOLS with the same labels as the Discord command (Standard, Total Match, Position-Only, Open)

``llmcord_core/shop_ui.py``:
  ITEM_KINDS['vault_tool'] = NP_('item kind', 'Vault tool'); Korean '금고 도구'. ``kind_line`` therefore reads 'Vault tool · Consumable'.
  kind_description(recipe, t) -> str | None   a plain sentence under the item name; None for a kind without one (random_message). For vault_tool:
      'Used inside your vault with /vault personal, not with /use. It costs one turn.'  (Korean translation required)
      (open decision: the exact English text is the implementer's; the tests only require a non-empty string that differs per locale.)

Every new msgid the Vault helpers and section pass to ``t`` / ``t.p`` / ``t.n`` must have a Korean catalog entry (checked by scanning the source).
"""
import ast
import inspect
import unittest
from pathlib import Path

from llmcord_core import games_ui, items, shop_ui, vault_discord
from llmcord_core.admin_store import ConflictError
from llmcord_core.i18n import Translator, UserError, codes
from llmcord_core.i18n.catalog import load_catalog
from llmcord_core.store import Store
from llmcord_core.vault import PROTOCOLS

G, OTHER = 1, 2
EN, KO = Translator('en'), Translator('ko')


def helper(name):
    return getattr(games_ui, name)


class Case(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.addCleanup(self.store.close)


class FormValues(Case):
    def test_vault_form_defaults_for_an_untouched_guild(self):
        """FEAT-68: defaults: off, Standard and Open allowed, default turns for all four protocols, no reward, template clue voice."""
        form = helper('vault_form')(self.store, G)
        self.assertEqual(set(form), {'enabled', 'protocols', 'turns', 'reward', 'advice_cap', 'clue_voice', 'revision'})
        self.assertEqual((form['enabled'], form['protocols'], form['reward'], form['clue_voice'], form['revision']), (False, ['standard', 'open'], 0, 'template', 0))
        self.assertEqual(set(form['turns']), set(PROTOCOLS))
        self.assertEqual(form['advice_cap'], self.store.vault_settings(G)['advice_cap'])

    def test_vault_form_mirrors_the_stored_settings(self):
        """FEAT-68: the form is built from the store, including a per-protocol turn override and the revision to save against."""
        self.store.set_vault_settings(G, enabled=True, protocols=['standard', 'total_match'], turns={'open': 5}, reward=40, advice_cap=3, clue_voice='llm')
        stored = self.store.vault_settings(G)
        form = helper('vault_form')(self.store, G)
        self.assertEqual(form, {k: stored[k] for k in form})
        self.assertEqual((form['turns']['open'], form['protocols'], form['revision']), (5, ['standard', 'total_match'], 1))

    def test_operation_form_mirrors_the_stored_settings(self):
        """FEAT-68: the operation form carries every operation setting and the revision."""
        self.assertEqual(helper('operation_form')(self.store, G), self.store.vault_operation_settings(G))
        self.store.set_vault_operation_settings(G, enabled=True, channel_id=77, week_start=3, stage_protocols=['open', 'standard', 'position_only'],
                                                contribution_reward=2, team_reward=9, min_guesses=4, support_reward=1)
        form = helper('operation_form')(self.store, G)
        self.assertEqual(form, self.store.vault_operation_settings(G))
        self.assertEqual((form['channel_id'], form['week_start'], form['stage_protocols'], form['min_guesses'], form['revision']), (77, 3, ['open', 'standard', 'position_only'], 4, 1))

    def test_forms_are_per_guild(self):
        """FEAT-68 guild isolation: another guild's settings never show in this guild's forms or change when this guild saves."""
        self.store.set_vault_settings(OTHER, enabled=True, reward=99)
        self.store.set_vault_operation_settings(OTHER, enabled=True, channel_id=5, team_reward=50)
        self.assertEqual(helper('vault_form')(self.store, G)['reward'], 0)
        self.assertEqual(helper('operation_form')(self.store, G)['channel_id'], None)
        base = helper('vault_form')(self.store, G)
        helper('save_vault')(self.store, G, {**base, 'enabled': True, 'reward': 7}, base)
        op = helper('operation_form')(self.store, G)
        helper('save_operation')(self.store, G, {**op, 'team_reward': 8}, op)
        self.assertEqual((self.store.vault_settings(OTHER)['reward'], self.store.vault_settings(OTHER)['revision']), (99, 1))
        self.assertEqual((self.store.vault_operation_settings(OTHER)['team_reward'], self.store.vault_operation_settings(OTHER)['revision']), (50, 1))


class SpyStore:
    """Wraps a real Store and records the keyword arguments of each settings write."""

    def __init__(self, store):
        self.store, self.calls = store, []

    def __getattr__(self, name):
        target = getattr(self.store, name)
        if name in ('set_vault_settings', 'set_vault_operation_settings'):
            def recorded(*args, **kw):
                self.calls.append((name, args, kw))
                return target(*args, **kw)
            return recorded
        return target


class SaveVault(Case):
    def save(self, store=None, **changes):
        store = store or self.store
        base = helper('vault_form')(store, G)
        return helper('save_vault')(store, G, {**base, **changes}, base)

    def test_saves_only_the_changed_settings_with_the_expected_revision(self):
        """FEAT-68: changing the reward alone sends reward and the revision, nothing else; the rest of the row is untouched."""
        spy = SpyStore(self.store)
        result = self.save(spy, reward=25)
        self.assertEqual(spy.calls, [('set_vault_settings', (G,), {'reward': 25, 'expected_revision': 0})])
        self.assertEqual((result['reward'], result['revision'], result['enabled']), (25, 1, False))
        self.assertEqual(self.store.vault_settings(G), result)

    def test_several_changes_go_in_one_write(self):
        """FEAT-68: enable, protocols, advice cap and clue voice change together in one revision step."""
        spy = SpyStore(self.store)
        result = self.save(spy, enabled=True, protocols=['standard', 'position_only'], advice_cap=2, clue_voice='llm')
        self.assertEqual(len(spy.calls), 1)
        self.assertEqual(spy.calls[0][2], {'enabled': True, 'protocols': ['standard', 'position_only'], 'advice_cap': 2, 'clue_voice': 'llm', 'expected_revision': 0})
        self.assertEqual((result['enabled'], result['protocols'], result['advice_cap'], result['clue_voice'], result['revision']), (True, ['standard', 'position_only'], 2, 'llm', 1))

    def test_only_the_changed_protocol_turns_are_sent(self):
        """FEAT-68: editing one turn budget sends that protocol only, so the other protocols keep following the built-in defaults."""
        spy = SpyStore(self.store)
        base = helper('vault_form')(self.store, G)
        helper('save_vault')(spy, G, {**base, 'turns': {**base['turns'], 'open': 6}}, base)
        self.assertEqual(spy.calls[0][2], {'turns': {'open': 6}, 'expected_revision': 0})
        self.assertEqual(self.store.vault_settings(G)['turns']['open'], 6)
        self.assertEqual(self.store.vault_settings(G)['turns']['standard'], base['turns']['standard'])

    def test_nothing_changed_writes_nothing(self):
        """FEAT-68: saving an untouched form makes no store call and does not bump the revision."""
        spy = SpyStore(self.store)
        result = self.save(spy)
        self.assertEqual(spy.calls, [])
        self.assertEqual(result['revision'], 0)
        self.assertEqual(self.store.vault_settings(G)['revision'], 0)

    def test_whole_floats_from_number_inputs_are_saved_as_ints(self):
        """FEAT-68: NiceGUI numbers arrive as floats; 3.0 is sent as 3 (the store refuses floats), and the stored value is an int."""
        result = self.save(reward=30.0, advice_cap=4.0)
        self.assertEqual((result['reward'], result['advice_cap']), (30, 4))
        self.assertIsInstance(result['reward'], int)

    def test_stale_revision_conflicts_and_writes_nothing(self):
        """FEAT-68: another admin saved first; the second save raises ConflictError and the first admin's values stay."""
        base = helper('vault_form')(self.store, G)
        self.store.set_vault_settings(G, reward=5)
        with self.assertRaises(ConflictError) as raised:
            helper('save_vault')(self.store, G, {**base, 'reward': 9, 'enabled': True}, base)
        self.assertEqual(raised.exception.code, codes.VAULT_SETTINGS_CHANGED)
        stored = self.store.vault_settings(G)
        self.assertEqual((stored['reward'], stored['enabled'], stored['revision']), (5, False, 1))

    def test_validation_error_writes_nothing_not_even_the_valid_changes(self):
        """FEAT-68: a bad value raises the store's UserError with its code; the valid change in the same save is not written either."""
        base = helper('vault_form')(self.store, G)
        untouched = self.store.vault_settings(G)
        cases = [({'reward': -1}, codes.VAULT_SETTINGS_REWARD), ({'reward': 2.5}, codes.VAULT_SETTINGS_REWARD),
                 ({'protocols': []}, codes.VAULT_SETTINGS_PROTOCOLS), ({'advice_cap': 51}, codes.VAULT_SETTINGS_ADVICE_CAP),
                 ({'turns': {**base['turns'], 'open': 0}}, codes.VAULT_SETTINGS_TURNS), ({'clue_voice': 'dice'}, codes.VAULT_SETTINGS_CLUE_VOICE)]
        for bad, code in cases:
            with self.subTest(bad=bad):
                with self.assertRaises(UserError) as raised:
                    helper('save_vault')(self.store, G, {**base, 'enabled': True, **bad}, base)
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(self.store.vault_settings(G), untouched)

    def test_validation_message_is_translatable_for_the_toast(self):
        """FEAT-68: the toast text is t.error(err); Korean differs from English for the Vault refusals."""
        base = helper('vault_form')(self.store, G)
        with self.assertRaises(UserError) as raised:
            helper('save_vault')(self.store, G, {**base, 'advice_cap': 99}, base)
        self.assertIn('50', EN.error(raised.exception))
        self.assertNotEqual(KO.error(raised.exception), EN.error(raised.exception))

    def test_saves_are_per_guild(self):
        """FEAT-68 guild isolation: saving guild 1 leaves guild 2 at its defaults, and a guild-2 revision cannot satisfy guild 1's save."""
        self.save(enabled=True)
        self.assertEqual(self.store.vault_settings(OTHER)['revision'], 0)
        self.assertFalse(self.store.vault_settings(OTHER)['enabled'])


class SaveOperation(Case):
    def save(self, store=None, **changes):
        store = store or self.store
        base = helper('operation_form')(store, G)
        return helper('save_operation')(store, G, {**base, **changes}, base)

    def test_saves_only_the_changed_settings_with_the_expected_revision(self):
        """FEAT-68: enabling and choosing the channel sends exactly those two fields and the revision."""
        spy = SpyStore(self.store)
        result = self.save(spy, enabled=True, channel_id=555)
        self.assertEqual(spy.calls, [('set_vault_operation_settings', (G,), {'enabled': True, 'channel_id': 555, 'expected_revision': 0})])
        self.assertEqual((result['enabled'], result['channel_id'], result['revision']), (True, 555, 1))

    def test_every_operation_field_round_trips(self):
        """FEAT-68: week start, three stage protocols, the three rewards and the minimum contributions all save and read back."""
        result = self.save(week_start=6, stage_protocols=['open', 'total_match', 'standard'], contribution_reward=3.0, team_reward=40, support_reward=2, min_guesses=5.0)
        self.assertEqual((result['week_start'], result['stage_protocols'], result['contribution_reward'], result['team_reward'], result['support_reward'], result['min_guesses']),
                         (6, ['open', 'total_match', 'standard'], 3, 40, 2, 5))
        self.assertEqual(helper('operation_form')(self.store, G), {**self.store.vault_operation_settings(G)})

    def test_nothing_changed_writes_nothing(self):
        """FEAT-68: an untouched operation form makes no write."""
        spy = SpyStore(self.store)
        self.save(spy)
        self.assertEqual(spy.calls, [])
        self.assertEqual(self.store.vault_operation_settings(G)['revision'], 0)

    def test_stale_revision_conflicts_and_writes_nothing(self):
        """FEAT-68: a concurrent save makes ours raise ConflictError; the other admin's values stay."""
        base = helper('operation_form')(self.store, G)
        self.store.set_vault_operation_settings(G, team_reward=11)
        with self.assertRaises(ConflictError):
            helper('save_operation')(self.store, G, {**base, 'team_reward': 12, 'enabled': True}, base)
        stored = self.store.vault_operation_settings(G)
        self.assertEqual((stored['team_reward'], stored['enabled'], stored['revision']), (11, False, 1))

    def test_validation_error_writes_nothing(self):
        """FEAT-68: bad week start, stage list, reward or minimum raises the store's UserError and leaves the row (and revision) alone."""
        base = helper('operation_form')(self.store, G)
        cases = [({'week_start': 7}, codes.VAULT_OPERATION_SETTINGS_WEEK_START), ({'stage_protocols': ['open']}, codes.VAULT_OPERATION_SETTINGS_PROTOCOLS),
                 ({'team_reward': -5}, codes.VAULT_SETTINGS_REWARD), ({'min_guesses': 0}, codes.VAULT_OPERATION_SETTINGS_MIN_GUESSES),
                 ({'min_guesses': 1.5}, codes.VAULT_OPERATION_SETTINGS_MIN_GUESSES)]
        for bad, code in cases:
            with self.subTest(bad=bad):
                with self.assertRaises(UserError) as raised:
                    helper('save_operation')(self.store, G, {**base, 'enabled': True, **bad}, base)
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(self.store.vault_operation_settings(G), base)

    def test_saves_are_per_guild(self):
        """FEAT-68 guild isolation: guild 2's operation settings are not touched by guild 1's save."""
        self.save(enabled=True, channel_id=9)
        self.assertEqual(self.store.vault_operation_settings(OTHER)['channel_id'], None)
        self.assertEqual(self.store.vault_operation_settings(OTHER)['revision'], 0)


class Options(unittest.TestCase):
    def test_channel_options_exclude_threads(self):
        """FEAT-68: the operation channel picker lists text channels only; an id that is a thread is dropped."""
        options = helper('vault_channel_options')({10: '#general', 11: '#games', 12: '#thread-in-games'}, {12: '#thread-in-games'})
        self.assertEqual(options, {10: '#general', 11: '#games'})

    def test_stored_channel_stays_visible_when_it_is_not_listed(self):
        """FEAT-68: a saved channel that no longer exists (or was a thread) shows as its id rather than disappearing."""
        fn = helper('vault_channel_options')
        self.assertEqual(fn({10: '#general'}, {}, 99), {10: '#general', 99: '99'})
        self.assertEqual(fn({10: '#general'}, {12: '#t'}, 12), {10: '#general', 12: '12'})
        self.assertEqual(fn({10: '#general'}, {}, 10), {10: '#general'})
        self.assertEqual(fn({}, {}, None), {})

    def test_week_start_options_name_the_days(self):
        """FEAT-68: 0 is Monday through 6 Sunday, matching the store; Korean day names."""
        days = helper('week_start_options')(EN)
        self.assertEqual(list(days), [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual((days[0], days[6]), ('Monday', 'Sunday'))
        korean = helper('week_start_options')(KO)
        self.assertEqual((korean[0], korean[6]), ('월요일', '일요일'))

    def test_protocol_options_match_the_discord_labels(self):
        """FEAT-68: the four protocols carry the same words as /vault, in both languages."""
        for t in (EN, KO):
            options = helper('vault_protocol_options')(t)
            self.assertEqual(list(options), list(PROTOCOLS))
            self.assertEqual(options, {p: t.render(vault_discord.PROTOCOLS[p]) for p in PROTOCOLS})
        self.assertEqual(helper('vault_protocol_options')(EN)['position_only'], 'Position-Only')


class WebProcessImports(unittest.TestCase):
    def test_protocol_labels_do_not_import_the_discord_vault_module(self):
        """MNT-91: the dashboard reads the protocol labels from a neutral module, so the web process never loads vault_discord."""
        import subprocess
        import sys
        code = ("import sys; from llmcord_core import games_ui; from llmcord_core.i18n import english; "
                "labels = games_ui.vault_protocol_options(english()); "
                "assert labels['position_only'] == 'Position-Only', labels; assert 'llmcord_core.vault_discord' not in sys.modules")
        done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_discord_and_dashboard_share_the_same_labels(self):
        from llmcord_core import vault_discord
        from llmcord_core.i18n.terms import VAULT_PROTOCOLS
        self.assertIs(vault_discord.PROTOCOLS, VAULT_PROTOCOLS)


class Strings(unittest.TestCase):
    NAMES = ('vault_section', 'vault_form', 'operation_form', 'save_vault', 'save_operation', 'vault_channel_options', 'week_start_options',
             'vault_protocol_options')

    def msgids(self):
        tree = ast.parse(inspect.getsource(games_ui))
        found = set()
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in self.NAMES):
            for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
                f = call.func
                text = lambda i: call.args[i].value if len(call.args) > i and isinstance(call.args[i], ast.Constant) and isinstance(call.args[i].value, str) else None
                if isinstance(f, ast.Name) and f.id == 't':
                    found.add((None, text(0)))
                elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == 't':
                    if f.attr == 'p':
                        found.add((text(0), text(1)))
                    elif f.attr == 'n':
                        found.update({(None, text(0)), (None, text(1))})
        return {m for m in found if m[1]}

    def test_every_new_string_has_a_korean_entry(self):
        """FEAT-68: each literal the Vault section and helpers pass to t() has a hand-written Korean translation (not the English fallback)."""
        self.assertIn('vault_section', {n.name for n in ast.walk(ast.parse(inspect.getsource(games_ui))) if isinstance(n, ast.FunctionDef)})
        msgids = self.msgids()
        self.assertGreater(len(msgids), 15, 'the Vault section should have a title, labels and explanations')
        ko = load_catalog('ko')
        missing = sorted(m for m in msgids if ko.get(*m) in (None, '') or ko.get(*m) == m[1] and len(m[1]) > 3 and m[1] not in ('Standard', 'Open'))
        self.assertEqual(missing, [])

    def test_section_title_and_experimental_label_are_translated_with_the_agreed_words(self):
        """FEAT-68: the section reads 금고 in Korean and carries the Experimental label like the other experimental switches."""
        msgids = {m[1] for m in self.msgids()}
        self.assertTrue(any('Vault' in m for m in msgids))
        self.assertTrue(any('Experimental' in m for m in msgids))
        self.assertTrue(any(KO(m).count('금고') for m in msgids if 'Vault' in m))
        self.assertTrue(any('Operation' in m or 'operation' in m for m in msgids))
        self.assertTrue(any(KO(m).count('작전') for m in msgids if 'peration' in m))


class ShopKind(unittest.TestCase):
    def tool(self):
        return items.get_item('vault_master_decoder')

    def test_item_kinds_has_a_vault_tool_entry_with_korean_label(self):
        """FEAT-68: the Shop tab no longer shows the English fallback for vault tools."""
        self.assertIn('vault_tool', shop_ui.ITEM_KINDS)
        self.assertEqual(EN.render(shop_ui.ITEM_KINDS['vault_tool']), 'Vault tool')
        self.assertEqual(KO.render(shop_ui.ITEM_KINDS['vault_tool']), '금고 도구')

    def test_every_catalog_kind_has_an_entry(self):
        """FEAT-68: every kind the built-in catalog uses has a label, so no kind falls back to derived English."""
        self.assertEqual({r.kind for r in items.all_items()} - set(shop_ui.ITEM_KINDS), set())

    def test_kind_line_reads_in_both_languages(self):
        """FEAT-68: item card sub-line for a vault tool."""
        self.assertEqual(shop_ui.kind_line(self.tool(), EN), 'Vault tool · Consumable')
        line = shop_ui.kind_line(self.tool(), KO)
        self.assertIn('금고 도구', line)
        self.assertIn('소모품', line)

    def test_vault_tools_have_a_description_other_kinds_do_not(self):
        """FEAT-68: vault tools explain that they are used in the vault, not with /use; toys have no extra description. Open decision: exact wording."""
        describe = getattr(shop_ui, 'kind_description')
        english, korean = describe(self.tool(), EN), describe(self.tool(), KO)
        self.assertTrue(english and english.strip())
        self.assertIn('/vault', english)
        self.assertTrue(korean and korean != english)
        self.assertIsNone(describe(items.get_item('dice'), EN))

    def test_every_vault_tool_is_described_the_same_way(self):
        """FEAT-68: all five tools get the description (the Shop catalog lists them all)."""
        tools = [r for r in items.all_items() if r.kind == 'vault_tool']
        self.assertEqual(len(tools), 5)
        for recipe in tools:
            self.assertTrue(getattr(shop_ui, 'kind_description')(recipe, EN))


if __name__ == '__main__':
    unittest.main()
