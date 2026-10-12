"""MNT-78: an item error's `code` is the plain ErrorCode; the old stable string lives in `kind`."""
import pickle
import unittest

from llmcord_core.i18n import ErrorCode, UserError, codes
from llmcord_core.items_store import (
    AlreadyOwned, HoldCapReached, InsufficientFunds, ItemError, NotForSale, NotHeld, OnCooldown, PriceChanged, UseDisabled,
)
from llmcord_core.store import Store

G, U = 1, 10

CASES = [
    ('not_for_sale', codes.ITEM_NOT_FOR_SALE, lambda: NotForSale(codes.ITEM_NOT_FOR_SALE)),
    ('price_changed', codes.ITEM_PRICE_CHANGED, lambda: PriceChanged(codes.ITEM_PRICE_CHANGED, 7, price=7)),
    ('already_owned', codes.ITEM_ALREADY_OWNED, lambda: AlreadyOwned(codes.ITEM_ALREADY_OWNED, name='Dice')),
    ('hold_cap', codes.ITEM_HOLD_CAP, lambda: HoldCapReached(codes.ITEM_HOLD_CAP, 3, cap=3, name='Dice')),
    ('insufficient', codes.ITEM_INSUFFICIENT, lambda: InsufficientFunds(codes.ITEM_INSUFFICIENT, 0, 5, name='Dice', price=5, currency='coins', balance=0)),
    ('not_held', codes.ITEM_NOT_HELD, lambda: NotHeld(codes.ITEM_NOT_HELD)),
    ('use_disabled', codes.ITEM_USE_DISABLED, lambda: UseDisabled(codes.ITEM_USE_DISABLED, name='Dice')),
    ('on_cooldown', codes.ITEM_ON_COOLDOWN, lambda: OnCooldown(codes.ITEM_ON_COOLDOWN, 3, name='Dice', remaining=3)),
]


class ItemErrorKindTests(unittest.TestCase):
    def test_code_is_the_plain_error_code_for_every_kind(self):
        """MNT-78: error.code is exactly an ErrorCode (no subclass), the same one that was raised."""
        for kind, code, make in CASES:
            with self.subTest(kind):
                error = make()
                self.assertIs(type(error.code), ErrorCode)
                self.assertEqual(error.code, code)
                self.assertIsInstance(error, UserError)

    def test_code_is_hashable_as_the_error_code(self):
        """MNT-78: a code works as a set member or dict key the same as the ErrorCode constant."""
        for kind, code, make in CASES:
            with self.subTest(kind):
                error = make()
                self.assertEqual(hash(error.code), hash(code))
                self.assertEqual({error.code: 1}[code], 1)
                self.assertIn(code, {error.code})

    def test_code_no_longer_equals_the_old_string(self):
        """MNT-78: the string comparison trap is gone; only `kind` holds the string."""
        for kind, code, make in CASES:
            with self.subTest(kind):
                error = make()
                self.assertNotEqual(error.code, kind)
                self.assertFalse(error.code == kind)

    def test_kind_holds_the_old_string_for_every_kind(self):
        """MNT-78: error.kind is the stable string callers used to compare error.code with."""
        for kind, code, make in CASES:
            with self.subTest(kind):
                self.assertEqual(make().kind, kind)

    def test_kind_is_set_on_the_class_for_a_bare_item_error(self):
        """MNT-78: the base class keeps its own kind string."""
        self.assertEqual(ItemError(codes.ITEM_UNKNOWN).kind, 'item_error')

    def test_kinds_are_distinct(self):
        """MNT-78: the eight kinds are eight different strings."""
        self.assertEqual(len({make().kind for _, _, make in CASES}), len(CASES))

    def test_class_docstring_documents_kind(self):
        """MNT-78: the ItemError docstring names the new attribute."""
        self.assertIn('kind', ItemError.__doc__)

    def test_pickle_round_trip_keeps_code_and_kind(self):
        """MNT-78: a pickled item error keeps its plain code, its kind and its extra fields."""
        for kind, code, make in CASES:
            with self.subTest(kind):
                copy = pickle.loads(pickle.dumps(make()))
                self.assertIs(type(copy.code), ErrorCode)
                self.assertEqual((copy.code, copy.kind), (code, kind))
        copy = pickle.loads(pickle.dumps(make_insufficient()))
        self.assertEqual((copy.balance, copy.price), (0, 5))


def make_insufficient():
    return CASES[4][2]()


class ItemErrorKindFromStoreTests(unittest.TestCase):
    def test_a_refused_purchase_carries_code_and_kind(self):
        """MNT-78: a real refusal from buy_item has the ErrorCode as code and the old string as kind."""
        store = Store()
        self.addCleanup(store.close)
        store.save_shop_item(G, 'tarot', expected_revision=0, for_sale=True)
        with self.assertRaises(InsufficientFunds) as caught:
            store.buy_item(G, U, 'tarot')
        self.assertEqual(caught.exception.code, codes.ITEM_INSUFFICIENT)
        self.assertIs(type(caught.exception.code), ErrorCode)
        self.assertEqual(caught.exception.kind, 'insufficient')
        with self.assertRaises(NotForSale) as caught:
            store.buy_item(G, U, 'nope')
        self.assertEqual((caught.exception.code, caught.exception.kind), (codes.ITEM_NOT_FOR_SALE, 'not_for_sale'))


if __name__ == '__main__':
    unittest.main()
