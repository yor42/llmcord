"""FEAT-14: macros expand in card fields and lore text (one pass, unknown stays literal).
"""
import unittest

import httpx

from llmcord_core.lore import LoreMatch
from llmcord_core.models import TurnMessage
from llmcord_core.prompts import block, compile_prompt, default_bundle
from llmcord_core.web import create_app

CARD_SOURCES = ('description', 'personality', 'scenario', 'opening', 'examples', 'card_instructions', 'card_post_history')
LORE_SOURCES = ('lore_before_char', 'lore_after_char', 'lore_before_examples', 'lore_after_examples')
TURN = {'char': 'Ada', 'user': 'Bo'}


def render(source, value, values=None, content='', lore_injections=()):
    bundle = default_bundle()
    bundle['purposes']['dialogue'] = [block('b', source, content=content), block('history', 'history', role='user')]
    merged = {**TURN, **(values or {}), source: value}
    request = compile_prompt(bundle, 'dialogue', merged, [TurnMessage('user', 'INPUT')], 'compatible', 4000, lore_injections=lore_injections)
    return [m.text for m in request.messages]


class MacroExpansionTests(unittest.TestCase):
    def test_card_sources_expand_macros(self):
        """FEAT-14: known macros in card field values expand with the turn values."""
        for source in CARD_SOURCES:
            with self.subTest(source=source):
                self.assertIn('Ada smiles at Bo', render(source, '{{char}} smiles at {{user}}'))

    def test_lore_sources_expand_macros(self):
        """FEAT-14: known macros in lore position values expand with the turn values."""
        for source in LORE_SOURCES:
            with self.subTest(source=source):
                self.assertIn('Ada smiles at Bo', render(source, '{{char}} smiles at {{user}}'))

    def test_in_chat_lore_expands_macros(self):
        """FEAT-14: in-chat lore injection content expands macros inside the World Info line."""
        item = LoreMatch(1, '{{char}} meets {{user}}', 'channel', 100, 'constant', 'key', 'in_chat', 1, 'system', rule={'order': 0})
        bundle = default_bundle()
        bundle['purposes']['dialogue'] = [block('history', 'history', role='user'), block('lore', 'lore_in_chat')]
        request = compile_prompt(bundle, 'dialogue', dict(TURN), [TurnMessage('user', 'OLD'), TurnMessage('user', 'NOW')], 'compatible', 4000, lore_injections=[item])
        self.assertIn('World Info [key]: Ada meets Bo', [m.text for m in request.messages])

    def test_unknown_macros_stay_literal_in_card_and_lore(self):
        """FEAT-14 (guard): unknown {{...}} in card and lore values is never blanked."""
        value = 'x {{random:a,b}} {{bogus}} y'
        for source in CARD_SOURCES + LORE_SOURCES:
            with self.subTest(source=source):
                self.assertIn(value, render(source, value))

    def test_unbalanced_braces_do_not_swallow_later_macro(self):
        """FEAT-14: an unbalanced {{ in one lore entry cannot pair with a later }} and hide a real macro."""
        self.assertIn('[a] see {{ here\n[b] Ada waits }}', render('lore_before_char', '[a] see {{ here\n[b] {{char}} waits }}'))

    def test_unclosed_braces_leave_field_intact(self):
        """FEAT-14: an unclosed {{ leaves the field as written."""
        self.assertIn('open {{ never closed', render('description', 'open {{ never closed'))

    def test_time_macro_expands_in_card_field(self):
        """FEAT-14: time macros in a card field expand to the provided turn value."""
        self.assertIn('It is 17:39', render('description', 'It is {{isotime}}', {'isotime': '17:39'}))

    def test_known_and_unknown_mix(self):
        """FEAT-14: a known macro expands while an unknown one beside it stays literal."""
        self.assertIn('Ada {{bogus}}', render('description', '{{char}} {{bogus}}'))

    def test_expansion_is_one_pass(self):
        """FEAT-14: a card value naming another macro gets that macro's raw value, which is not expanded again."""
        self.assertIn('{{char}}', render('description', '{{scenario}}', {'scenario': '{{char}}'}))

    def test_macro_values_are_not_reexpanded(self):
        """FEAT-14 (guard): a turn value containing a macro (user label '{{user}}') stays literal."""
        self.assertIn('hi {{user}}', render('description', 'hi {{user}}', {'user': '{{user}}'}))

    def test_template_expands_and_inserts_expanded_source(self):
        """FEAT-14: with a {0} template, the template's macros and the source's macros both expand."""
        self.assertIn('Ada says: Ada smiles at Bo', render('description', '{{char}} smiles at {{user}}', content='{{char}} says: {0}'))

    def test_template_still_wraps_source(self):
        """FEAT-14 (guard): a {0} template wraps the source and expands its own macros."""
        self.assertIn('Ada says: plain', render('description', 'plain', content='{{char}} says: {0}'))

    def test_non_card_sources_stay_verbatim(self):
        """FEAT-14 (guard): payload, personal, encounters, summary, preceding, recent and location never expand macros."""
        for source in ('payload', 'personal', 'encounters', 'summary', 'preceding', 'recent', 'location'):
            with self.subTest(source=source):
                self.assertIn('{{char}}', render(source, '{{char}}'))

    def test_history_messages_stay_verbatim(self):
        """FEAT-14 (guard): a '{{char}}' in a history message reaches the model literally."""
        bundle = default_bundle()
        bundle['purposes']['dialogue'] = [block('history', 'history', role='user')]
        request = compile_prompt(bundle, 'dialogue', dict(TURN), [TurnMessage('user', 'say {{char}}')], 'compatible', 4000)
        self.assertEqual([m.text for m in request.messages], ['say {{char}}'])


class PreviewMacroTests(unittest.IsolatedAsyncioTestCase):
    async def test_preview_expands_card_macros(self):
        """FEAT-14: the dashboard prompt preview shows expanded card macros."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[]))) as http:
            app = create_app(':memory:', 'https://pi.test', 'c', 's', 'b', http, enable_dashboard=False)
            try:
                store = app.state.store
                world = store.create_space(1, 'W', 'world')
                char = store.add_character(1, world, 'Alice', {'name': 'Alice', 'description': '{{char}} waves at {{user}}'}, None, [])
                request = app.state.admin.preview_prompt(1, default_bundle(), 'dialogue', char, 'Sample', 'Earlier sample')
            finally:
                app.state.store.close()
        self.assertIn('Alice waves at Sample user', '\n'.join(m.text for m in request.messages))


if __name__ == '__main__':
    unittest.main()
