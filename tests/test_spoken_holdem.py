"""Hold'em adapter for spoken moves (llmcord_core/spoken/holdem.py + packaged lexicon), D32 / FEAT-37 seam.

Spots are real engine states (heads-up, stacks 1000 unless noted, blinds 5/10):
  open     seat 0 (SB) to act preflop: level 10, to call 5, min raise to 20.
  unbet    seat 1 (BB) after seat 0 calls: check or raise, level 10, top 1000.
  facing   seat 1 after seat 0 raises to 60: level 60, min raise to 110, pot 70, top 1000.
  flop     seat 1 on the flop after call/check: level 0, min raise to 10, pot 20.
"""
import json
import random
import unittest
from pathlib import Path

from llmcord_core.games import Seat, holdem
from llmcord_core.spoken import holdem as sh
from llmcord_core.spoken.moves import Reading, SpokenMove

LEX_DIR = Path(__file__).resolve().parent.parent / "llmcord_core" / "spoken" / "lexicon"
KINDS = ("fold", "check", "call", "raise", "all_in")


def mk(stacks=(1000, 1000), seed="s"):
    return holdem.new(seed, [Seat(i, "member", 100 + i, s) for i, s in enumerate(stacks)], None, 0)


def act(state, *moves):
    for m in moves:
        state = holdem.apply(state, state.turn, m)
    return state


def vw(state):
    return holdem.view(state, state.turn)


def accepted(state, move):
    try:
        holdem.apply(state, state.turn, move)
    except Exception:  # noqa: BLE001 - any rejection counts
        return False
    return True


class SpotMixin:
    def setUp(self):
        self.open = mk()
        self.unbet = act(mk(), "call")
        self.facing = act(mk(), "raise:60")
        self.flop = act(mk(), "call", "check")


class KindsForViewTests(SpotMixin, unittest.TestCase):
    def kinds(self, state):
        return set(sh.kinds_for_view(vw(state)))

    def test_open_spot(self):
        self.assertEqual(self.kinds(self.open), {"fold", "call", "raise", "all_in"})

    def test_unbet_spot_call_means_check(self):
        self.assertEqual(self.kinds(self.unbet), {"check", "call", "raise", "all_in"})

    def test_facing_spot_has_no_check(self):
        self.assertEqual(self.kinds(self.facing), {"fold", "call", "raise", "all_in"})

    def test_result_is_a_tuple(self):
        self.assertIsInstance(sh.kinds_for_view(vw(self.facing)), tuple)

    def test_not_viewers_turn_is_empty(self):
        other = holdem.view(self.facing, 0)
        self.assertEqual(sh.kinds_for_view(other), ())
        self.assertIsNone(sh.read_holdem("raise", other))

    def test_finished_hand_is_empty(self):
        done = act(self.facing, "fold")
        self.assertEqual(sh.kinds_for_view(holdem.view(done, 0)), ())

    def test_cannot_raise_spot(self):
        s = mk((1000, 8))  # BB is all in for 8; SB can only fold or call
        self.assertEqual(self.kinds(s), {"fold", "call", "all_in"})

    def test_short_stack_that_cannot_reach_a_raise(self):
        s = act(mk((1000, 30)), "raise:60")  # seat 1 has 30 total, level is 60
        self.assertEqual(self.kinds(s), {"fold", "call", "all_in"})

    def test_short_raise_still_counts_as_raise(self):
        s = act(mk((1000, 40)), "raise:30")  # legal: fold, call, raise:40 (all in)
        self.assertEqual(self.kinds(s), {"fold", "call", "raise", "all_in"})


class ReadHoldemTests(SpotMixin, unittest.TestCase):
    def test_call_when_unbet_reads_as_call(self):
        r = sh.read_holdem("call", vw(self.unbet))
        self.assertEqual(r, Reading((SpokenMove("call"),), True))

    def test_check_word_when_unbet(self):
        r = sh.read_holdem("check", vw(self.unbet))
        self.assertEqual(r, Reading((SpokenMove("check"),), True))

    def test_illegal_kind_word_is_ignored(self):
        self.assertIsNone(sh.read_holdem("check", vw(self.facing)))
        self.assertIsNone(sh.read_holdem("fold", vw(self.unbet)))

    def test_raise_to_is_sure(self):
        r = sh.read_holdem("raise to 300 ㅋㅋ", vw(self.facing))
        self.assertEqual(r, Reading((SpokenMove("raise", 300, "to"),), True))

    def test_raise_by_is_sure(self):
        r = sh.read_holdem("raise 50 more", vw(self.facing))
        self.assertEqual(r, Reading((SpokenMove("raise", 50, "by"),), True))

    def test_raise_with_unclear_amount_is_unsure(self):
        r = sh.read_holdem("raise 200", vw(self.facing))
        self.assertEqual(r.candidates, (SpokenMove("raise", 200, None),))
        self.assertFalse(r.sure)

    def test_raise_without_amount_is_sure(self):
        r = sh.read_holdem("raise", vw(self.facing))
        self.assertEqual(r, Reading((SpokenMove("raise"),), True))

    def test_preset_alone(self):
        r = sh.read_holdem("half pot", vw(self.facing))
        self.assertEqual(r, Reading((SpokenMove("raise", preset="half_pot"),), False))
        self.assertIsNone(sh.read_holdem("wait a min", vw(self.facing)))

    def test_all_in_words(self):
        for text in ("all in", "all-in", "올인 간다", "shove"):
            with self.subTest(text=text):
                r = sh.read_holdem(text, vw(self.facing))
                self.assertEqual(r, Reading((SpokenMove("all_in"),), True))

    def test_korean_call_and_fold(self):
        self.assertEqual(sh.read_holdem("콜 ㄱㄱ", vw(self.facing)).candidates, (SpokenMove("call"),))
        self.assertEqual(sh.read_holdem("그냥 다이", vw(self.facing)).candidates, (SpokenMove("fold"),))

    def test_two_kinds_unsure(self):
        r = sh.read_holdem("fold or call?", vw(self.facing))
        self.assertEqual([c.kind for c in r.candidates], ["fold", "call"])
        self.assertFalse(r.sure)

    def test_negation_unsure(self):
        r = sh.read_holdem("I won't fold", vw(self.facing))
        self.assertEqual([c.kind for c in r.candidates], ["fold"])
        self.assertFalse(r.sure)

    def test_amount_with_stray_preset_word_is_unsure(self):
        r = sh.read_holdem("raise to 300, nice pot", vw(self.facing))
        self.assertEqual(r.candidates, (SpokenMove("raise", 300, "to", "pot"),))
        self.assertFalse(r.sure)

    def test_more_negations_are_unsure(self):
        for text in ("I can't fold", "cannot call", "shouldn't fold", "wouldnt call", "nope, fold"):
            with self.subTest(text=text):
                self.assertFalse(sh.read_holdem(text, vw(self.facing)).sure)

    def test_chat_and_bare_numbers_are_none(self):
        v = vw(self.facing)
        for text in ("nice hand", "ㅋㅋㅋ 미쳤다", "300", "콜라 마시고 올게", "체크카드 어디갔지"):
            with self.subTest(text=text):
                self.assertIsNone(sh.read_holdem(text, v))


class ResolveTests(SpotMixin, unittest.TestCase):
    def res(self, state, move):
        return sh.resolve(move, vw(state))

    def test_resolution_shape(self):
        r = self.res(self.facing, SpokenMove("call"))
        self.assertEqual((r.moves, r.needs_picker), (("call",), False))

    def test_fold_and_check(self):
        self.assertEqual(self.res(self.facing, SpokenMove("fold")).moves, ("fold",))
        self.assertEqual(self.res(self.unbet, SpokenMove("check")).moves, ("check",))

    def test_check_when_facing_a_bet_is_nothing(self):
        r = self.res(self.facing, SpokenMove("check"))
        self.assertEqual((r.moves, r.needs_picker), ((), False))

    def test_fold_when_unbet_is_nothing(self):
        self.assertEqual(self.res(self.unbet, SpokenMove("fold")).moves, ())

    def test_call_when_unbet_becomes_check(self):
        self.assertEqual(self.res(self.unbet, SpokenMove("call")).moves, ("check",))
        self.assertEqual(self.res(self.flop, SpokenMove("call")).moves, ("check",))

    def test_call_when_facing(self):
        self.assertEqual(self.res(self.facing, SpokenMove("call")).moves, ("call",))

    def test_all_in_is_top_raise(self):
        self.assertEqual(self.res(self.facing, SpokenMove("all_in")).moves, ("raise:1000",))
        self.assertEqual(self.res(self.unbet, SpokenMove("all_in")).moves, ("raise:1000",))
        self.assertEqual(self.res(self.open, SpokenMove("all_in")).moves, ("raise:1000",))
        self.assertEqual(self.res(self.flop, SpokenMove("all_in")).moves, ("raise:990",))

    def test_all_in_top_includes_current_bet(self):
        s = act(mk((500, 1000)), "raise:100")  # seat 1 holds 990 behind plus 10 posted
        self.assertEqual(self.res(s, SpokenMove("all_in")).moves, ("raise:1000",))
        s2 = mk((500, 1000))  # seat 0: bet 5 + stack 495
        self.assertEqual(self.res(s2, SpokenMove("all_in")).moves, ("raise:500",))

    def test_all_in_falls_back_to_call_when_cannot_raise(self):
        for s in (mk((1000, 8)), act(mk((1000, 30)), "raise:60")):
            r = self.res(s, SpokenMove("all_in"))
            self.assertEqual((r.moves, r.needs_picker), (("call",), False))

    def test_all_in_short_raise(self):
        s = act(mk((1000, 40)), "raise:30")
        self.assertEqual(self.res(s, SpokenMove("all_in")).moves, ("raise:40",))

    def test_presets_unbet(self):
        self.assertEqual(self.res(self.unbet, SpokenMove("raise", preset="min")).moves, ("raise:20",))
        self.assertEqual(self.res(self.unbet, SpokenMove("raise", preset="half_pot")).moves, ("raise:20",))
        self.assertEqual(self.res(self.unbet, SpokenMove("raise", preset="pot")).moves, ("raise:30",))

    def test_presets_facing(self):
        # level 60, pot 70, to call 50 -> full = 120
        self.assertEqual(self.res(self.facing, SpokenMove("raise", preset="min")).moves, ("raise:110",))
        self.assertEqual(self.res(self.facing, SpokenMove("raise", preset="half_pot")).moves, ("raise:120",))
        self.assertEqual(self.res(self.facing, SpokenMove("raise", preset="pot")).moves, ("raise:180",))

    def test_presets_postflop(self):
        self.assertEqual(self.res(self.flop, SpokenMove("raise", preset="min")).moves, ("raise:10",))
        self.assertEqual(self.res(self.flop, SpokenMove("raise", preset="half_pot")).moves, ("raise:10",))
        self.assertEqual(self.res(self.flop, SpokenMove("raise", preset="pot")).moves, ("raise:20",))

    def test_amount_to(self):
        r = self.res(self.facing, SpokenMove("raise", 300, "to"))
        self.assertEqual((r.moves, r.needs_picker), (("raise:300",), False))

    def test_amount_by_adds_to_level(self):
        self.assertEqual(self.res(self.facing, SpokenMove("raise", 50, "by")).moves, ("raise:110",))
        self.assertEqual(self.res(self.unbet, SpokenMove("raise", 40, "by")).moves, ("raise:50",))

    def test_amount_unclear_gives_both_targets_to_first(self):
        r = self.res(self.facing, SpokenMove("raise", 200, None))
        self.assertEqual((r.moves, r.needs_picker), (("raise:200", "raise:260"), False))

    def test_amount_unclear_drops_the_illegal_target(self):
        # to 50 is below the minimum (110); by 50 -> 110 is fine
        self.assertEqual(self.res(self.facing, SpokenMove("raise", 50, None)).moves, ("raise:110",))
        # to 1000 is the top; by 1000 -> 1060 is over it
        self.assertEqual(self.res(self.facing, SpokenMove("raise", 1000, None)).moves, ("raise:1000",))

    def test_unclear_targets_deduplicate_postflop(self):
        r = self.res(self.flop, SpokenMove("raise", 100, None))
        self.assertEqual(r.moves, ("raise:100",))

    def test_target_above_top_is_dropped_not_clamped(self):
        r = self.res(self.facing, SpokenMove("raise", 5000, "to"))
        self.assertEqual((r.moves, r.needs_picker), ((), True))

    def test_target_below_minimum_is_dropped(self):
        r = self.res(self.facing, SpokenMove("raise", 50, "to"))
        self.assertEqual((r.moves, r.needs_picker), ((), True))

    def test_short_all_in_target_is_legal_below_minimum(self):
        s = act(mk((1000, 40)), "raise:30")  # min raise to 50, top 40
        self.assertEqual(self.res(s, SpokenMove("raise", 40, "to")).moves, ("raise:40",))
        self.assertEqual(self.res(s, SpokenMove("raise", preset="min")).moves, ("raise:40",))
        r = self.res(s, SpokenMove("raise", 35, "to"))
        self.assertEqual((r.moves, r.needs_picker), ((), True))

    def test_explicit_amount_wins_over_preset(self):
        r = self.res(self.facing, SpokenMove("raise", 300, "to", "pot"))
        self.assertEqual(r.moves, ("raise:300",))
        self.assertEqual(self.res(self.facing, SpokenMove("raise", 300, "to", "min")).moves, ("raise:300",))

    def test_raise_without_amount_needs_picker(self):
        r = self.res(self.facing, SpokenMove("raise"))
        self.assertEqual((r.moves, r.needs_picker), ((), True))

    def test_raise_when_no_raise_is_legal(self):
        for s in (mk((1000, 8)), act(mk((1000, 30)), "raise:60")):
            for move in (SpokenMove("raise"), SpokenMove("raise", 300, "to"), SpokenMove("raise", preset="pot")):
                with self.subTest(state=s.stacks, move=move):
                    r = self.res(s, move)
                    self.assertEqual((r.moves, r.needs_picker), ((), False))

    def test_not_viewers_turn_resolves_to_nothing(self):
        r = sh.resolve(SpokenMove("call"), holdem.view(self.facing, 0))
        self.assertEqual((r.moves, r.needs_picker), ((), False))


class PackagedLexiconTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lex = json.loads((LEX_DIR / "holdem.json").read_text(encoding="utf-8"))
        cls.common = json.loads((LEX_DIR / "common.json").read_text(encoding="utf-8"))

    def examples(self):
        for key in ("examples", "_examples"):
            if key in self.lex:
                return self.lex[key]
        self.fail("holdem.json needs an 'examples' (or '_examples') object keyed by kind")

    def table_talk(self):
        for key in ("table_talk", "_table_talk"):
            if key in self.lex:
                return self.lex[key]
        examples = self.examples()
        if "table_talk" in examples:
            return examples["table_talk"]
        self.fail("holdem.json needs a 'table_talk' list")

    def test_common_negations(self):
        for lang in ("en", "ko"):
            self.assertTrue(self.common["negations"][lang] if "negations" in self.common else self.common[lang])

    def test_every_kind_has_en_and_ko_words(self):
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.assertTrue(self.lex[kind]["en"])
                self.assertTrue(self.lex[kind]["ko"])

    def test_required_words_present(self):
        self.assertIn("muck", self.lex["fold"]["en"])
        self.assertIn("올인", self.lex["all_in"]["ko"])
        self.assertIn("콜라", self.lex["_exclude"])
        self.assertEqual(set(self.lex["_presets"]), {"min", "half_pot", "pot"})
        self.assertIn("raise", self.lex["_amount_kinds"])

    def test_at_least_eight_examples_per_kind(self):
        ex = self.examples()
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.assertGreaterEqual(len(ex[kind]), 8)
                self.assertTrue(all(isinstance(p, str) and p.strip() for p in ex[kind]))

    def test_examples_cover_english_korean_and_mixed(self):
        ex = self.examples()
        for kind in KINDS:
            phrases = ex[kind]
            has_ko = [any("가" <= ch <= "힣" or "ㄱ" <= ch <= "ㅎ" for ch in p) for p in phrases]
            has_en = [any(ch.isascii() and ch.isalpha() for ch in p) for p in phrases]
            with self.subTest(kind=kind):
                self.assertTrue(any(has_ko))
                self.assertTrue(any(has_en))

    def test_table_talk_negatives(self):
        talk = self.table_talk()
        self.assertGreaterEqual(len(talk), 15)
        self.assertIn("nice hand", talk)


class WholeFlowPropertyTests(unittest.TestCase):
    PHRASES = ("call", "check", "fold", "raise", "raise 200", "raise to 60", "raise to 300", "raise to 5000",
               "raise 50 more", "raise 7", "all in", "올인", "pot", "half pot", "min raise", "하프", "레이즈 3백",
               "raise to 40", "raise to 1,000", "콜", "bet 15")

    def test_every_resolved_move_is_accepted_by_the_engine(self):
        checked = 0
        for seed in ("a", "b", "c", "d", "e"):
            for stacks in ((1000, 1000), (60, 400, 25), (300, 20, 90, 1000)):
                rng = random.Random(f"{seed}{stacks}")
                state = mk(stacks, seed)
                while not state.finished:
                    v = vw(state)
                    for text in self.PHRASES:
                        reading = sh.read_holdem(text, v)
                        if reading is None:
                            continue
                        for cand in reading.candidates:
                            res = sh.resolve(cand, v)
                            for move in res.moves:
                                checked += 1
                                self.assertTrue(accepted(state, move), (seed, stacks, state.street, text, move))
                    legal = holdem.legal_moves(state, state.turn)
                    state = holdem.apply(state, state.turn, rng.choice(legal))
        self.assertGreater(checked, 200)


if __name__ == "__main__":
    unittest.main()
