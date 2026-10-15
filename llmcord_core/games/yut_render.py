"""Yut (윷놀이) board image (FEAT-57, D42).

The only Pillow code for Yut. Pure: takes the dict from ``yut.board_view`` and returns PNG bytes; no Discord, DB or clock. Shapes only
(no text), so no font file is needed. The empty board is drawn once (lazily) and copied per render; finished images are cached by
piece positions only.
"""
import io
import threading
from collections import OrderedDict

from PIL import Image, ImageDraw

SIZE = (480, 480)
_MARGIN = 60
_SIDE = SIZE[0] - 2 * _MARGIN
CACHE_MAX = 64

_CORNERS = {"P0": (_SIDE, _SIDE), "P5": (_SIDE, 0), "P10": (0, 0), "P15": (0, _SIDE)}


def _lerp(a, b, k, n):
    return (_MARGIN + round(a[0] + (b[0] - a[0]) * k / n), _MARGIN + round(a[1] + (b[1] - a[1]) * k / n))


def _build_points():
    pts = {}
    names = ("P0", "P5", "P10", "P15")
    for i, start in enumerate(names):
        end = names[(i + 1) % 4]
        for k in range(5):
            pts[f"P{i * 5 + k}"] = _lerp(_CORNERS[start], _CORNERS[end], k, 5)
    for chain, (a, b) in (
        (("a1", "a2", "b1", "b2"), ("P5", "P15")),
        (("c1", "c2", "d1", "d2"), ("P10", "P0")),
    ):
        order = (chain[0], chain[1], "C", chain[2], chain[3])
        for j, name in enumerate(order, start=1):
            pts[name] = _lerp(_CORNERS[a], _CORNERS[b], j, 6)
    for name in ("P0", "P5", "P10", "P15"):
        pts[name] = (_MARGIN + _CORNERS[name][0], _MARGIN + _CORNERS[name][1])
    return pts


POINTS = _build_points()
_BIG = {"P0", "P5", "P10", "P15", "C"}
_SEAT_COLORS = ((214, 56, 56), (46, 108, 214), (46, 160, 82), (230, 172, 28))
_SEAT_OFFSETS = ((-8, -8), (8, -8), (-8, 8), (8, 8))
_STACK_STEP = (0, -5)
_PIECE_R = 9

_base = None
_cache = OrderedDict()
_lock = threading.Lock()


def _board_base():
    global _base
    with _lock:
        if _base is None:
            _base = _draw_base()
        return _base


def _draw_base():
    img = Image.new("RGB", SIZE, (240, 222, 178))
    d = ImageDraw.Draw(img)
    names = ("P0", "P5", "P10", "P15")
    for i, n in enumerate(names):
        d.line([POINTS[n], POINTS[names[(i + 1) % 4]]], fill=(90, 62, 36), width=4)
    d.line([POINTS["P5"], POINTS["P15"]], fill=(90, 62, 36), width=4)
    d.line([POINTS["P10"], POINTS["P0"]], fill=(90, 62, 36), width=4)
    for name, (x, y) in POINTS.items():
        r = 14 if name in _BIG else 8
        d.ellipse([x - r, y - r, x + r, y + r], fill=(250, 244, 226), outline=(90, 62, 36), width=3)
    return img


def _key(view):
    return tuple(tuple(row) for row in view["pieces"])


def piece_positions(view):
    """[(seat, piece index, (x, y))] for pieces on the board, offset by seat and by stack position so none share a coordinate."""
    out = []
    for s, row in enumerate(view["pieces"]):
        seen = {}
        for i, p in enumerate(row):
            if p in (None, "done"):
                continue
            k = seen[p] = seen.get(p, -1) + 1
            x, y = POINTS[p]
            ox, oy = _SEAT_OFFSETS[s % 4]
            out.append((s, i, (x + ox + k * _STACK_STEP[0], y + oy + k * _STACK_STEP[1])))
    return out


def render_board(view):
    img = _board_base().copy()
    d = ImageDraw.Draw(img)
    for s, _i, (x, y) in piece_positions(view):
        d.ellipse([x - _PIECE_R, y - _PIECE_R, x + _PIECE_R, y + _PIECE_R], fill=_SEAT_COLORS[s % 4], outline=(30, 20, 12), width=2)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def render_board_cached(view):
    key = _key(view)
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    data = render_board(view)
    with _lock:
        _cache[key] = data
        _cache.move_to_end(key)
        while len(_cache) > CACHE_MAX:
            _cache.popitem(last=False)
    return data


def clear_cache():
    with _lock:
        _cache.clear()


def text_fallback(view):
    """English plain text, one line per seat: points held, pieces still at the start, pieces done."""
    lines = []
    for s, row in enumerate(view["pieces"]):
        on = [p for p in row if p not in (None, "done")]
        board = ", ".join(on) if on else "-"
        lines.append(f"Seat {s + 1}: on board {board}; at start {sum(p is None for p in row)}; done {sum(p == 'done' for p in row)}")
    return "\n".join(lines)
