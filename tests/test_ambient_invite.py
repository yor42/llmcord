"""FEAT-40 (D37): ambient silence is decided by an EN/KO invitation rule; the director only runs when it fires."""
import asyncio
import time
import unittest

from helpers import FakeModels, core_settings
from llmcord_core import ambient
from llmcord_core.engine import Engine, SceneContext
from llmcord_core.store import Store

NAMES = ["Luna", "Al", "하늘", "Mary Jane"]


def sig(latest, recent=(), names=NAMES, **kw):
    return ambient.invite(latest, list(recent), names, **kw).signals


class NameTests(unittest.TestCase):
    def test_english_names_are_word_bounded_and_case_insensitive(self):
        self.assertEqual(sig("hey luna"), ("name",))
        self.assertEqual(sig("Luna's coat"), ("name",))
        self.assertEqual(sig("ok, Luna, ok"), ("name",))
        self.assertEqual(sig("Luna는 어때"), ("name",))
        self.assertEqual(sig("is that really so"), ())
        self.assertEqual(sig("Alan came"), ())
        self.assertEqual(sig("hi Al"), ("name",))
        self.assertEqual(sig("Lunatic"), ())

    def test_korean_names_take_particles_and_vocatives(self):
        for text in ("하늘", "하늘아", "하늘이 왔다", "하늘은 어때", "하늘 생각은", "얘 하늘!"):
            self.assertEqual(sig(text), ("name",), text)
        for text in ("하늘색 좋다", "푸른하늘", "하늘나라"):
            self.assertEqual(sig(text), (), text)

    def test_korean_names_take_extended_particles(self):
        for text in ("하늘의 책", "하늘한테 말해", "하늘에게 줘", "하늘씨", "하늘님", "하늘과 함께", "하늘랑 놀자", "하늘에서 봐"):
            self.assertEqual(sig(text), ("name",), text)

    def test_non_ascii_latin_names_are_bounded(self):
        self.assertEqual(sig("hi Zoë", names=["Zoë"]), ("name",))
        self.assertEqual(sig("Zoëlla came", names=["Zoë"]), ())
        self.assertEqual(sig("xZoë", names=["Zoë"]), ())

    def test_multi_word_names_skip_titles(self):
        self.assertEqual(sig("the weather is nice", names=["The Narrator"]), ())
        self.assertEqual(sig("narrator, hello", names=["The Narrator"]), ("name",))
        self.assertEqual(sig("sir who", names=["Sir Al Bo"]), ())
        self.assertEqual(sig("hi Bob", names=["Lady Bob Smith"]), ("name",))
        self.assertEqual(sig("the narrator", names=["The Narrator"]), ("name",))

    def test_multi_word_names_match_full_or_first_word(self):
        self.assertEqual(ambient.invite("hi Mary", [], NAMES).named, (3,))
        self.assertEqual(ambient.invite("hi mary jane", [], NAMES).named, (3,))
        self.assertEqual(ambient.invite("hi Jane", [], NAMES).named, ())
        self.assertEqual(ambient.invite("hi Jo", [], ["Jo Bo"]).named, ())
        self.assertEqual(ambient.invite("안녕 루나야", [], ["루나 공주"]).named, (0,))

    def test_only_latest_and_last_two_recent_lines_count(self):
        self.assertEqual(sig("yeah", ["Luna?", "ok", "sure"]), ())
        self.assertEqual(sig("yeah", ["a", "Luna?", "sure"]), ("name",))
        self.assertEqual(ambient.invite("yeah", ["x", "Al is here", "ok"], NAMES).named, (1,))


class PhraseTests(unittest.TestCase):
    def test_english_and_korean_phrases(self):
        for text in ("What do you think?", "any thoughts", "join us!", "your turn", "who wants tea",
                     "어떻게 생각해?", "너는?", "넌 어때", "누구 없어?", "같이 하자"):
            self.assertEqual(sig(text), ("phrase",), text)

    def test_weak_phrases_need_a_question(self):
        for text in ("anyone up for cards?", "anybody home?", "누가 이겼어?", "다들 어때?", "이 의견 어때?", "너희는 뭐 해?"):
            self.assertEqual(sig(text), ("phrase",), text)
        for text in ("누가 뭐 했대", "다들 잘 자", "anyone up for cards", "너희는 좋겠다", "의견 있음"):
            self.assertEqual(sig(text), (), text)

    def test_other_phrase_wins_when_an_exclude_covers_a_different_one(self):
        self.assertEqual(sig("my thoughts are with you, what do you think?"), ("phrase",))
        self.assertEqual(sig("anyone who wants to play?"), ("phrase",))
        self.assertEqual(sig("if anyone wants to join us"), ("phrase",))
        self.assertEqual(sig("anyone's guess?"), ())

    def test_urls_are_not_questions(self):
        self.assertEqual(sig("https://x.com/a?b=1 anyone", favorite=True), ())

    def test_korean_question_endings_are_specific(self):
        for text in ("어디 갈래", "뭐 하니", "숙제 했니", "누구세요 나요"):
            self.assertEqual(sig(text, favorite=True), ("favorite",), text)
        for text in ("우리 어머니", "아니", "그러니까"):
            self.assertEqual(sig(text, favorite=True), (), text)

    def test_plain_chat_and_excludes_are_silent(self):
        for text in ("random weather", "my thoughts and prayers", "I don't tell anyone's secrets", "누가 봐도 그렇지", "anyonexyz", "너는 바보"):
            self.assertEqual(sig(text), (), text)

    def test_fullwidth_question_mark(self):
        self.assertEqual(sig("너는？"), ("phrase",))


class FollowUpAndFavoriteTests(unittest.TestCase):
    def test_follow_up_needs_question_or_reaction(self):
        for text in ("really?", "so what now?", "뭐 먹지？", "그래서 어디 갔냐", "lol", "wow", "ㅋㅋㅋ", "헐", "대박", "진짜?", "ㄹㅇ"):
            self.assertEqual(sig(text, follow_up=True), ("follow_up",), text)
            if text != "진짜?":
                self.assertEqual(sig(text), (), text)
        for text in ("I went home", "그래서 그랬다", "그러니까", "this is a long sentence with lol in it somewhere really"):
            self.assertEqual(sig(text, follow_up=True), (), text)

    def test_reactions_are_whole_message(self):
        for text in ("lol", "wow really?", "ㅋㅋㅋ 대박", "헐 진짜?", "no way!", "와!", "와아", "wow 😮"):
            self.assertEqual(sig(text, follow_up=True), ("follow_up",), text)
        for text in ("I really need sleep", "that is not true", "same thing happened to me", "와이파이 되나", "lol that was great fun"):
            self.assertEqual(sig(text, follow_up=True), (), text)

    def test_question_alone_is_silent_without_follow_up_or_favorite(self):
        self.assertEqual(sig("is it raining?"), ())

    def test_favorite_loosens_to_a_question_only(self):
        self.assertEqual(sig("is it raining?", favorite=True), ("favorite",))
        self.assertEqual(sig("it is raining", favorite=True), ())
        self.assertEqual(sig("lol", favorite=True), ())

    def test_invited_wrapper(self):
        self.assertTrue(ambient.invited("hi Luna", [], NAMES))
        self.assertFalse(ambient.invited("hi", [], NAMES))


class EngineGateTests(unittest.TestCase):
    def setUp(self):
        self.store = Store()
        self.world = self.store.create_space(1, "World", "world")
        self.store.bind_channel(1, 100, self.world)
        self.alice = self.store.add_character(1, self.world, "Alice", {"name": "Alice"}, None, [])
        self.bob = self.store.add_character(1, self.world, "Bob", {"name": "Bob"}, None, [])
        self.store.set_cast(100, None, [self.alice, self.bob])

    def tearDown(self):
        self.store.close()

    def speakers(self, models, text, recent=(), **kw):
        scene = SceneContext(1, 100, None, self.world, 9, 5000, text, None, list(recent), [], ambient=True, **kw)
        return asyncio.run(Engine(self.store, models, core_settings()).speakers(scene))

    def test_rule_silent_makes_no_director_call(self):
        models = FakeModels([self.alice])
        self.assertEqual(self.speakers(models, "random weather"), [])
        self.assertEqual(models.calls, [])

    def test_invited_with_empty_director_falls_back_to_named_character(self):
        models = FakeModels([])
        self.assertEqual([r["name"] for r in self.speakers(models, "bob, what's up")], ["Bob"])
        self.assertEqual(len(models.calls), 1)

    def test_invited_with_empty_director_falls_back_to_first_option(self):
        models = FakeModels([])
        self.assertEqual([r["name"] for r in self.speakers(models, "anyone here?")], ["Alice"])

    def test_invited_with_empty_director_falls_back_to_favorite(self):
        self.store.add_favorite(1, 9, self.bob)
        models = FakeModels([])
        self.assertEqual([r["name"] for r in self.speakers(models, "is it late?")], ["Bob"])
        self.assertEqual(self.speakers(FakeModels([]), "it is late"), [])

    def test_director_choice_is_kept_and_payload_is_not_ambient(self):
        models = FakeModels([self.bob])
        self.assertEqual([r["name"] for r in self.speakers(models, "Alice, what do you think?")], ["Bob"])

    def test_event_turn_unchanged(self):
        models = FakeModels([])
        self.assertEqual(self.speakers(models, "random weather", event=True), [])
        self.assertEqual(len(models.calls), 1)

    def test_follow_up_after_recent_character_reply(self):
        self.store.record_node(900, 1, 100, None, None, self.alice, "Hello", [])
        recent = [{"message_id": 800, "text": "a"}, {"message_id": 950, "text": "b"}]
        self.assertEqual([r["name"] for r in self.speakers(FakeModels([]), "really?", recent)], ["Alice"])
        recent = [{"message_id": 950, "text": "b"}, {"message_id": 960, "text": "c"}]
        self.assertEqual(self.speakers(FakeModels([]), "really?", recent), [])

    def test_missing_message_id_on_second_to_last_recent_is_not_a_follow_up(self):
        self.store.record_node(900, 1, 100, None, None, self.alice, "Hello", [])
        recent = [{"text": "a"}, {"text": "b"}]
        self.assertEqual(self.speakers(FakeModels([self.alice]), "really?", recent), [])

    def test_stale_character_reply_is_not_a_follow_up(self):
        self.store.record_node(900, 1, 100, None, None, self.alice, "Hello", [], created_at=time.time() - 3600)
        self.assertEqual(self.speakers(FakeModels([self.alice]), "really?"), [])


if __name__ == "__main__":
    unittest.main()
