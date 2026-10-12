"""Chips amount parser for chat text: '200', '1,500', '2k', '1.5k', Korean units (2만, 1만5천, 1만 5천, 3백).

Digits take thousands commas only in proper groups ('1,500'); a malformed run like '1,2345' is rejected, not split.
After a Korean unit the next number joins the amount only when it has a unit too ('1만 5천'; '5천 2 more' is 5000).
A fraction counts only with k or a Korean unit ('1.5k', '1.5만'); plain '2.5' is skipped.
"""
import re
from decimal import Decimal

_NUM = re.compile(r"(?<!\d)(?<!\d,)(?:\d{1,3}(?:,\d{3})+(?!\d)(?:\.\d+)?|\d+(?:\.\d+)?(?!,\d))")
_NEXT = re.compile(r"\s*(\d+(?:\.\d+)?)")
_K = re.compile(r"k(?![a-z])", re.I)
_UNITS = {"백": 100, "천": 1000, "만": 10000}
_TO_BEFORE = re.compile(r"(?:\bto|\btotal|총)\s*$")
_TO_AFTER = re.compile(r"\s*까지")
_BY_BEFORE = re.compile(r"(?:\bby|\bextra|\bmore|(?<!\S)더|추가)\s*$")
_BY_AFTER = re.compile(r"\s*(?:more\b|extra\b|더(?![가-힣])|추가)")


def _value(s):
    return Decimal(s.replace(",", ""))


def _read(text, m):
    """(total or None, end) for the number at m; None when it is a fraction with no k or unit."""
    val, end = _value(m.group()), m.end()
    k = _K.match(text, end)
    if k:
        return val * 1000, k.end()
    total, units = Decimal(0), False
    while end < len(text) and text[end] in _UNITS:
        total += val * _UNITS[text[end]]
        units, end = True, end + 1
        nxt = _NEXT.match(text, end)
        if not nxt or nxt.end() >= len(text) or text[nxt.end()] not in _UNITS:
            return total, end
        val, end = _value(nxt.group(1)), nxt.end()
    if not units and val != val.to_integral_value():
        return None, end
    return (total + val if units else val), end


def _mode(text, start, end):
    before, after = text[:start], text[end:]
    if _TO_BEFORE.search(before) or _TO_AFTER.match(after):
        return "to"
    if _BY_BEFORE.search(before) or _BY_AFTER.match(after):
        return "by"
    return None


def parse_amount(text):
    """The first amount in text and its mode ('to' | 'by' | None), or None when there is no amount of at least 1."""
    text = text.casefold()
    for m in _NUM.finditer(text):
        total, end = _read(text, m)
        if total is None:
            continue
        amount = int(total)
        return (amount, _mode(text, m.start(), end)) if amount >= 1 else None
    return None
