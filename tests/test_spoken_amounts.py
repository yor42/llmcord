"""Spoken-move amount parser (llmcord_core/spoken/amounts.py), D32 / FEAT-37 preliminary seam."""
import unittest

from llmcord_core.spoken.amounts import parse_amount


class ParseAmountTests(unittest.TestCase):
    def test_plain_digits_have_no_mode(self):
        self.assertEqual(parse_amount("raise 200"), (200, None))
        self.assertEqual(parse_amount("200"), (200, None))

    def test_thousands_commas(self):
        self.assertEqual(parse_amount("raise 1,500"), (1500, None))
        self.assertEqual(parse_amount("1,000,000"), (1000000, None))

    def test_k_suffix(self):
        self.assertEqual(parse_amount("2k"), (2000, None))
        self.assertEqual(parse_amount("raise 2K"), (2000, None))
        self.assertEqual(parse_amount("1.5k"), (1500, None))

    def test_korean_units(self):
        self.assertEqual(parse_amount("2만"), (20000, None))
        self.assertEqual(parse_amount("1만5천"), (15000, None))
        self.assertEqual(parse_amount("5천"), (5000, None))
        self.assertEqual(parse_amount("3백"), (300, None))
        self.assertEqual(parse_amount("레이즈 3백"), (300, None))

    def test_mode_to(self):
        self.assertEqual(parse_amount("raise to 300"), (300, "to"))
        self.assertEqual(parse_amount("total 300"), (300, "to"))
        self.assertEqual(parse_amount("300까지 레이즈"), (300, "to"))
        self.assertEqual(parse_amount("총 300"), (300, "to"))
        self.assertEqual(parse_amount("raise to 1,500"), (1500, "to"))
        self.assertEqual(parse_amount("raise to 2k"), (2000, "to"))

    def test_mode_by(self):
        self.assertEqual(parse_amount("raise 50 more"), (50, "by"))
        self.assertEqual(parse_amount("50 더"), (50, "by"))
        self.assertEqual(parse_amount("raise by 50"), (50, "by"))
        self.assertEqual(parse_amount("extra 50"), (50, "by"))
        self.assertEqual(parse_amount("추가 50"), (50, "by"))

    def test_korean_unit_with_mode(self):
        self.assertEqual(parse_amount("1만5천까지"), (15000, "to"))
        self.assertEqual(parse_amount("5천 더"), (5000, "by"))

    def test_korean_units_with_space(self):
        self.assertEqual(parse_amount("1만 5천"), (15000, None))

    def test_plain_fraction_is_not_an_amount(self):
        self.assertIsNone(parse_amount("2.5"))
        self.assertEqual(parse_amount("2.5 then 300"), (300, None))
        self.assertEqual(parse_amount("1.15k"), (1150, None))
        self.assertEqual(parse_amount("1.5만"), (15000, None))

    def test_malformed_comma_run_is_rejected(self):
        self.assertIsNone(parse_amount("1,2345"))
        self.assertEqual(parse_amount("1,234"), (1234, None))

    def test_mode_by_before_number(self):
        self.assertEqual(parse_amount("더 300"), (300, "by"))
        self.assertEqual(parse_amount("more 300"), (300, "by"))

    def test_deo_must_be_a_word(self):
        self.assertEqual(parse_amount("300 더럽다"), (300, None))
        self.assertEqual(parse_amount("300 더 해"), (300, "by"))
        self.assertEqual(parse_amount("300더"), (300, "by"))

    def test_first_amount_wins(self):
        self.assertEqual(parse_amount("raise 200 or 300"), (200, None))

    def test_no_amount(self):
        self.assertIsNone(parse_amount("no number here"))
        self.assertIsNone(parse_amount(""))

    def test_zero_is_not_an_amount(self):
        self.assertIsNone(parse_amount("0"))
        self.assertIsNone(parse_amount("raise 0"))

    def test_amount_inside_sentence(self):
        self.assertEqual(parse_amount("ok fine, raise to 300 lol"), (300, "to"))


if __name__ == "__main__":
    unittest.main()
