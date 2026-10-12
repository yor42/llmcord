"""MNT-68 (1c): Phrase, Translator.render, Msg/Phrase params, and coded exceptions that are not UserErrors."""
from __future__ import annotations

import pickle
import unittest

from llmcord_core import i18n
from llmcord_core.games import holdem
from llmcord_core.games.base import IllegalMove
from llmcord_core.i18n import E_, N_, NP_, CodedError, Msg, Phrase, Translator, UserError

KINGS = NP_('poker rank plural', 'Kings')
PAIR = NP_('poker hand', 'Pair of {rank}')
CODE = E_('test.phrase_game', '{game} is off for {who}.')
GAME = NP_('game name', 'Blackjack')


class PhraseTests(unittest.TestCase):
    def test_equality_hash_and_repr(self):
        a, b = Phrase(PAIR, rank=KINGS), Phrase(PAIR, rank=KINGS)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, Phrase(PAIR, rank=NP_('poker rank plural', 'Aces')))
        self.assertNotEqual(a, Phrase(N_('Pair of {rank}'), rank=KINGS))  # the context is part of the identity
        self.assertNotEqual(a, 'Pair of Kings')
        self.assertEqual(repr(a), "Phrase(NP_('poker hand', 'Pair of {rank}'), rank='Kings')")
        self.assertEqual(repr(Phrase(N_('Hello'))), "Phrase('Hello')")

    def test_immutable_and_picklable(self):
        phrase = Phrase(PAIR, rank=KINGS)
        with self.assertRaises(AttributeError):
            phrase.msg = 'x'
        with self.assertRaises(AttributeError):
            del phrase.params
        phrase.params['rank'] = 'changed'  # a copy
        self.assertEqual(str(phrase), 'Pair of Kings')
        clone = pickle.loads(pickle.dumps(phrase))
        self.assertEqual(clone, phrase)
        self.assertEqual(clone.msg.context, 'poker hand')

    def test_str_is_english(self):
        phrase = Phrase(PAIR, rank=KINGS)
        self.assertEqual(str(phrase), 'Pair of Kings')
        self.assertEqual(str(phrase), i18n.english().render(phrase))


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.t = Translator(i18n.PSEUDO_LOCALE)
        self.en = i18n.english()

    def test_render_phrase_msg_and_plain(self):
        self.assertEqual(self.en.render(Phrase(PAIR, rank=KINGS)), 'Pair of Kings')
        self.assertEqual(self.en.render(KINGS), 'Kings')
        for value in ('Luna', 3, 1500, '10♠', None):
            self.assertEqual(self.en.render(value), str(value))
        self.assertEqual(self.t.render('Luna'), 'Luna')  # plain text is never translated
        self.assertEqual(self.t.render(3), '3')

    def test_pseudo_wraps_msg_and_phrase(self):
        self.assertEqual(self.t.render(KINGS), i18n.pseudo('Kings'))
        out = self.t.render(Phrase(PAIR, rank=KINGS))
        self.assertTrue(out.startswith('[!! '))
        self.assertIn(i18n.pseudo('Kings'), out)
        self.assertIn(i18n.pseudo('Pair of ')[4:8], out)

    def test_nested_params_translate_but_plain_strings_do_not(self):
        inner = Phrase(NP_('x', 'inner {n}'), n=5)
        out = self.t('Outer {a} {b} {c} {d}', a=KINGS, b=inner, c='plain', d=7)
        self.assertIn(i18n.pseudo('Kings'), out)
        self.assertIn(i18n.pseudo('inner {n}').replace('{n}', '5'), out)
        self.assertIn(' plain ', out)
        self.assertIn(' 7 ', out)

    def test_render_nested_phrase_in_phrase(self):
        outer = Phrase(N_('{one} and {two}'), one=Phrase(PAIR, rank=KINGS), two=KINGS)
        out = self.t.render(outer)
        self.assertIn(i18n.pseudo('Kings'), out)
        self.assertIn(i18n.pseudo('Pair of {rank}').replace('{rank}', i18n.pseudo('Kings')), out)
        self.assertEqual(str(outer), 'Pair of Kings and Kings')

    def test_msg_param_in_english_is_unchanged(self):
        self.assertEqual(self.en('{x} wins', x=KINGS), 'Kings wins')
        self.assertEqual(self.en.n('{count} card', '{count} cards', 2, count=2), '2 cards')


class CodedErrorTests(unittest.TestCase):
    def test_user_error_msg_param(self):
        err = UserError(CODE, game=GAME, who='Luna')
        self.assertEqual(str(err), 'Blackjack is off for Luna.')
        self.assertIsInstance(err, ValueError)
        self.assertEqual(i18n.english().error(err), 'Blackjack is off for Luna.')
        out = Translator(i18n.PSEUDO_LOCALE).error(err)
        self.assertIn(i18n.pseudo('Blackjack'), out)
        self.assertIn('Luna', out)
        self.assertEqual(pickle.loads(pickle.dumps(err)).params, {'game': GAME, 'who': 'Luna'})

    def test_user_error_phrase_param(self):
        err = UserError(CODE, game=Phrase(PAIR, rank=KINGS), who='x')
        self.assertEqual(str(err), 'Pair of Kings is off for x.')

    def test_illegal_move_is_coded_but_not_a_value_error(self):
        err = IllegalMove(CODE, game=GAME, who='Luna')
        self.assertIsInstance(err, Exception)
        self.assertNotIsInstance(err, ValueError)
        self.assertNotIsInstance(err, UserError)
        self.assertIsInstance(err, CodedError)
        self.assertEqual(str(err), 'Blackjack is off for Luna.')
        self.assertEqual(i18n.english().error(err), 'Blackjack is off for Luna.')
        out = Translator(i18n.PSEUDO_LOCALE).error(err)
        self.assertIn(i18n.pseudo('Blackjack'), out)
        self.assertEqual(pickle.loads(pickle.dumps(err)).code, CODE)

    def test_illegal_move_legacy_text(self):
        err = IllegalMove('the round is over')
        self.assertEqual(str(err), 'the round is over')
        self.assertEqual(err.args, ('the round is over',))
        self.assertIsNone(err.code)
        self.assertEqual(i18n.english().error(err), 'the round is over')
        with self.assertRaises(TypeError):
            IllegalMove('text', extra=1)
        with self.assertRaises(IllegalMove):
            raise IllegalMove('x')

    def test_msg_stays_a_str_with_context(self):
        self.assertIsInstance(KINGS, Msg)
        self.assertEqual(KINGS, 'Kings')
        self.assertEqual(KINGS.context, 'poker rank plural')


class EngineTwinTests(unittest.TestCase):
    def test_hand_phrase_matches_hand_name(self):
        rank = (2, 13, 12, 5)
        self.assertEqual(str(holdem.hand_phrase(rank)), holdem.hand_name(rank))
        self.assertEqual(holdem.hand_name(rank), 'Two pair, Kings and Queens')
        self.assertEqual(holdem.hand_phrase((1, 6, 9, 8, 7)), Phrase(NP_('poker hand', 'Pair of {rank}'), rank=holdem._PLURAL[6]))
        self.assertEqual(holdem.hand_name((1, 6, 9, 8, 7)), 'Pair of Sixes')

    def test_pseudo_translates_the_rank_inside_the_hand(self):
        out = Translator(i18n.PSEUDO_LOCALE).render(holdem.hand_phrase((1, 13, 9, 8, 7)))
        self.assertIn(i18n.pseudo('Kings'), out)
        self.assertIn(i18n.pseudo('Pair of {rank}').split('{rank}')[0][4:], out)


if __name__ == '__main__':
    unittest.main()
