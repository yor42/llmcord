"""FEAT-20: a character's blackjack decision as one structured model call.

The prompt holds only the character's name and own card text (description and personality), the visible table and
the legal moves: no lore, scene history, memory or member identities, so nothing private can reach the model.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

from .games import blackjack, doubt, onecard, playstyle, yut
from .models import TurnMessage
from .usage import capture_usage, log_purpose, log_scope

ROLE = "director"  # strict structured outputs and the cheaper model; the dialogue role is tuned for long replies
MAX_LINE = 200
MAX_CARD_TEXT = 1500
SYSTEM = ("You are {name}, playing blackjack at a table in a roleplay server. Stay in character.\n"
          "{persona}\n"
          "Choose your move from the legal moves. You may also say one short line in character, at most "
          f"{MAX_LINE} characters, or leave it empty. Do not mention anyone with @. Return only JSON.")


def schema(legal) -> dict:
    return {"type": "object", "properties": {"move": {"type": "string", "enum": list(legal)}, "line": {"type": "string"}},
            "required": ["move", "line"], "additionalProperties": False}


def persona(character) -> str:
    try:
        card = json.loads(character["card"])
    except (TypeError, ValueError):
        card = {}
    parts = []
    for key in ("description", "personality"):
        text = card.get(key) if isinstance(card, dict) else None
        if isinstance(text, str) and text.strip():
            text = text.replace("{{char}}", character["name"]).replace("{{Char}}", character["name"]).replace("{{user}}", "the player").replace("{{User}}", "the player")
            parts.append(f"{key.capitalize()}: {text.strip()[:MAX_CARD_TEXT]}")
    return "\n".join(parts)


def with_style(text, style) -> str:
    """The persona plus a short in-words play style for a character that has a saved one (None: the text is unchanged)."""
    return text if style is None else f"{text}\nPlay style: {playstyle.describe(style)}."


MOVE_HELP = {"insure": "insure: put up half your bet that the dealer has blackjack (pays 2 to 1)",
             "even_money": "even_money: take your bet back plus the same again now, instead of risking a dealer blackjack",
             "no_insurance": "no_insurance: decline",
             "surrender": "surrender: give up the hand and get half your bet back"}


def table_text(snap, seat, legal) -> str:
    view = snap["view"]
    hands, dealer = view["hands"], view["dealer"]
    lines = [f"Your hand: {blackjack.hand_str(hands[seat])} ({blackjack.total_str(hands[seat])})",
             f"Dealer shows: {blackjack.hand_str(dealer)}"]
    others = []
    for i, other in enumerate(snap["seats"]):
        if i != seat:
            who = other["name"] if other["kind"] == "character" else "A player"
            word = {"stand": " stands", "bust": " bust", "blackjack": " blackjack", "surrender": " surrendered"}.get(view["status"][i], "")
            others.append(f"{who}: {blackjack.hand_str(hands[i])} ({blackjack.total_str(hands[i])}){word}")
    if others:
        lines.append("Other seats: " + "; ".join(others))
    if snap.get("phase") == "insurance":
        lines.append("The dealer shows an ace. Insurance?")
    lines.append("Legal moves: " + ", ".join(legal))
    lines.extend(MOVE_HELP[m] for m in legal if m in MOVE_HELP)
    return "\n".join(lines)


def clean_line(value) -> str:
    return " ".join(value.split())[:MAX_LINE] if isinstance(value, str) else ""


async def ask(bot, guild_id, channel_id, character, snap, seat, legal, style=None) -> dict:
    """One model call on a pinned backend snapshot, recorded as usage feature 'game'. Raises on any failure (the caller falls back)."""
    legal = tuple(legal)
    request = SimpleNamespace(messages=[TurnMessage("system", SYSTEM.format(name=character["name"], persona=with_style(persona(character), style))),
                                        TurnMessage("user", table_text(snap, seat, legal))])
    with bot.backend.pin(), capture_usage(guild_id, channel_id, "game"), log_scope(None), log_purpose("game"):
        return await bot.models.structured_compiled(ROLE, request, "blackjack_move", schema(legal))


# --- I Doubt It (FEAT-21 B2): a seat's play or doubt decision; the prompt holds only that seat's own view ---

DOUBT_SYSTEM = ("You are {name}, playing I Doubt It at a table in a roleplay server. Stay in character.\n"
                "{persona}\n"
                "Rules: on your turn you play 1 to 4 cards face down and claim they are all the rank to claim. You must "
                "claim that rank but may bluff. Anyone may doubt the latest claim: if it was false the claimer takes the pile, "
                "otherwise the doubter does. The first to empty their hand wins. Return only JSON.")
RECENT = 5


def doubt_play_schema(view) -> dict:
    held = doubt.held_counts(view.hand)
    return {"type": "object", "properties": {rank: {"type": "integer", "enum": list(range(min(n, doubt.MAX_PLAY) + 1))} for rank, n in held.items()},
            "required": list(held), "additionalProperties": False}


def doubt_doubt_schema() -> dict:
    return {"type": "object", "properties": {"doubt": {"type": "boolean"}}, "required": ["doubt"], "additionalProperties": False}


def doubt_table_text(view, names, purpose) -> str:
    lines = ["Your hand: " + ", ".join(f"{rank} x{n}" for rank, n in doubt.held_counts(view.hand).items()), f"Rank to claim: {view.rank}"]
    for i, count in enumerate(view.counts):
        if i != view.seat:
            lines.append(f"{names[i]}: {doubt.cards_text(count)}" + ("" if view.active[i] else " (out of the game)"))
    lines.append(f"Pile: {doubt.cards_text(view.pile)}")
    for item in view.history[-RECENT:]:
        if item.kind == "play":
            lines.append(f"{names[item.seat]} claimed {item.count} {item.rank}")
        else:
            lines.append(f"{names[item.doubter]} doubted {names[item.claimer]}'s claim of {item.rank}: the cards were "
                         + ", ".join(f"{rank}{suit}" for rank, suit in item.cards) + (" (a bluff)" if item.bluff else " (true)"))
    w = view.window
    if w is not None:
        lines.append(f"The doubt window is open on {names[w.claimer]}'s claim.")
    if purpose == "doubt" and w is not None:
        lines.append(f"Claim to judge: {names[w.claimer]} claims {w.count} {w.rank}")
        lines.append("From your hand this claim " + ("cannot be true." if doubt.impossible(view) else "could be true."))
        lines.append("Decide whether to doubt it.")
    else:
        lines.append("Decide how many cards of each rank you hold to play.")
    return "\n".join(lines)


async def _ask_doubt(bot, guild_id, channel_id, character, view, names, purpose, name, schema, style=None):
    request = SimpleNamespace(messages=[TurnMessage("system", DOUBT_SYSTEM.format(name=character["name"], persona=with_style(persona(character), style))),
                                        TurnMessage("user", doubt_table_text(view, names, purpose))])
    with bot.backend.pin(), capture_usage(guild_id, channel_id, "game"), log_scope(None), log_purpose("game"):
        return await bot.models.structured_compiled(ROLE, request, name, schema)


async def ask_doubt_play(bot, guild_id, channel_id, character, view, names, style=None) -> dict:
    """One call on the seat's own view; returns the raw answer (the caller validates it). Raises on any failure."""
    return await _ask_doubt(bot, guild_id, channel_id, character, view, names, "play", "doubt_play", doubt_play_schema(view), style)


async def ask_doubt_doubt(bot, guild_id, channel_id, character, view, names, style=None) -> bool:
    result = await _ask_doubt(bot, guild_id, channel_id, character, view, names, "doubt", "doubt_decision", doubt_doubt_schema(), style)
    answer = result.get("doubt") if isinstance(result, dict) else None
    if not isinstance(answer, bool):
        raise ValueError("doubt answer is not a boolean")
    return answer


# --- Texas Hold'em (FEAT-33 C2): a seat's move from its own view; no seed, other hole cards, member identities or odds ---

HOLDEM_SYSTEM = ("You are {name}, playing Texas Hold'em (no-limit cash game) at a table in a roleplay server. Stay in character.\n"
                 "{persona}\n"
                 "Choose your move from the legal moves: fold, check, call, or raise:<n> (n is your total bet this round). "
                 f"You may also say one short line in character, at most {MAX_LINE} characters, or leave it empty. Do not mention anyone with @. Return only JSON.")
HOLDEM_RECENT = 6


def holdem_schema(legal) -> dict:
    return schema(legal)


def clean_name(value, sep="") -> str:
    """The one name sanitiser: one line, no markup, mention or colon characters (each replaced by ``sep``), at most 40 characters."""
    return " ".join(re.sub(r"[*_~`|>@<\\:]", sep, str(value)).split())[:40]


def _seat_names(snap) -> list:
    """Prompt-safe names by seat: characters by name (one line, no markup or colons), members as Player 1, 2, ... (never member names or ids)."""
    names, players = [], 0
    for seat in snap["seats"]:
        if seat["kind"] == "character":
            name = clean_name(seat["name"], " ")
        else:
            players += 1
            name = f"Player {players}"
        names.append(name or f"Seat {len(names) + 1}")
    return names


def holdem_names(snap) -> list:
    return _seat_names(snap)


def onecard_names(snap) -> list:
    """One Card prompt names: the same shape, but its own entry point (not Hold'em's)."""
    return _seat_names(snap)


def _cards(cards) -> str:
    return " ".join(f"{c[0]}{c[1]}" for c in cards)


def holdem_table_text(view, names, legal) -> str:
    me = view.seat
    lines = [f"Your cards: {_cards(view.hole)}", f"Street: {view.street}. Board: {_cards(view.board) or 'none yet'}", f"Pot: {view.pot}"]
    for i, name in enumerate(names):
        flags = (" (you)" if i == me else "") + (" (dealer)" if view.button == i else "") + (" folded" if view.folded[i] else " all-in" if view.all_in[i] else "")
        lines.append(f"Seat {i + 1}, {name}{flags}: stack {view.stacks[i]}, bet this round {view.bets[i]}")
    if view.to_call:
        lines.append(f"To call: {view.to_call}")
    for seat, move in view.history[-HOLDEM_RECENT:]:
        lines.append(f"Earlier: {names[seat]} " + (f"raised to {move[6:]}" if move.startswith("raise:") else {"fold": "folded", "check": "checked", "call": "called"}.get(move, move)))
    lines.append("Legal moves: " + ", ".join(legal))
    return "\n".join(lines)


def holdem_move(result, legal):
    """The answer's move when it is a dict whose ``move`` is a string equal to a menu id, else None (``line`` is never read as a move)."""
    move = result.get("move") if isinstance(result, dict) else None
    return move if isinstance(move, str) and move in tuple(legal) else None


async def ask_holdem(bot, guild_id, channel_id, character, view, names, legal, style=None) -> dict:
    """One call on the seat's own view; returns the raw answer (the caller validates it). Raises on any failure."""
    legal = tuple(legal)
    request = SimpleNamespace(messages=[TurnMessage("system", HOLDEM_SYSTEM.format(name=character["name"], persona=with_style(persona(character), style))),
                                        TurnMessage("user", holdem_table_text(view, names, legal))])
    with bot.backend.pin(), capture_usage(guild_id, channel_id, "game"), log_scope(None), log_purpose("game"):
        return await bot.models.structured_compiled(ROLE, request, "holdem_move", holdem_schema(legal))


# --- One Card (FEAT-51 B): a seat's move from its own view; no seed, other hands or draw pile cards, member identities ---

ONECARD_SYSTEM = ("You are {name}, playing One Card at a table in a roleplay server. Stay in character.\n"
                  "{persona}\n"
                  "Rules: be the first to play your last card. Play a card that matches the top card's suit or rank (or the named suit), or draw one. "
                  "Jokers play on anything. Attacks (a 2, an ace, a joker) make the next player draw unless they defend with another attack. "
                  "A jack skips a player, a queen reverses the direction, a 7 lets you name the next suit (play:<7>=<suit>). "
                  "Choose your move from the legal moves. "
                  f"You may also say one short line in character, at most {MAX_LINE} characters, or leave it empty. Do not mention anyone with @. Return only JSON.")


def onecard_schema(legal) -> dict:
    return schema(legal)


def _n_cards(n) -> str:
    return f"{n} card" + ("" if n == 1 else "s")


def onecard_table_text(view, names, legal) -> str:
    lines = ["Your hand: " + ", ".join(onecard.card_text(card) for card in view.hand), f"Top card: {onecard.card_text(view.top)}"]
    if view.suit:
        lines.append(f"Suit: {view.suit}")
    lines.append("Direction: " + ("clockwise" if view.direction > 0 else "counter-clockwise"))
    if view.pending:
        lines.append(f"Pending attack: {view.pending}")
    for i, count in enumerate(view.counts):
        if i != view.seat:
            lines.append(f"{names[i]}: {_n_cards(count)}" + ("" if view.active[i] else " (out of the game)"))
    lines.append(f"Draw pile: {_n_cards(view.draw)}")
    for item in view.history[-HOLDEM_RECENT:]:
        if item.kind == "play":
            lines.append(f"{names[item.seat]} played {onecard.card_text(item.card)}" + (f" and named {item.suit}" if item.suit else ""))
        elif item.kind == "draw":
            lines.append(f"{names[item.seat]} drew " + ("a card" if item.count == 1 else _n_cards(item.count)))
        elif item.kind in ("forfeit", "bankrupt"):
            lines.append(f"{names[item.seat]} " + ("left the game" if item.kind == "forfeit" else "went bankrupt and is out"))
    lines.append("Legal moves: " + ", ".join(legal))
    return "\n".join(lines)


async def ask_onecard(bot, guild_id, channel_id, character, view, names, legal, style=None) -> dict:
    """One call on the seat's own view; returns the raw answer (the caller validates it). Raises on any failure."""
    legal = tuple(legal)
    request = SimpleNamespace(messages=[TurnMessage("system", ONECARD_SYSTEM.format(name=character["name"], persona=with_style(persona(character), style))),
                                        TurnMessage("user", onecard_table_text(view, names, legal))])
    with bot.backend.pin(), capture_usage(guild_id, channel_id, "game"), log_scope(None), log_purpose("game"):
        return await bot.models.structured_compiled(ROLE, request, "onecard_move", onecard_schema(legal))


# --- Yut (FEAT-60): a seat's move among the legal ones; everything is public, so the prompt holds the whole board (no seed, no member identities) ---

YUT_SYSTEM = ("You are {name}, playing Yut (a Korean board game) at a table in a roleplay server. Stay in character.\n"
              "{persona}\n"
              "Rules: on your turn you throw four sticks and move your pieces along the board by the banked throws (Do 1 step, Gae 2, Geol 3, Yut 4, Mo 5; "
              "Back-do 1 step back). A Yut or a Mo, or a capture, gives another throw. Landing on an opposing piece sends it back to the start. "
              "Landing on your own piece joins them into a stack that moves together. A piece on a corner may take a shortcut across the board. "
              "The ring points are P1 to P19, P0 is the finish: passing it brings a piece home. "
              "The shortcuts: a1, a2, b1, b2 lie on the diagonal from P5 to P15 through the center C; c1, c2, d1, d2 lie on the diagonal from P10 to P0 through C. "
              "The first to bring every piece home wins. "
              "Choose your move from the legal moves. "
              f"You may also say one short line in character, at most {MAX_LINE} characters, or leave it empty. Do not mention anyone with @. Return only JSON.")


def yut_schema(legal) -> dict:
    return schema(legal)


def yut_names(snap) -> list:
    """Yut prompt names: the same shape as the other tables, with its own entry point."""
    return _seat_names(snap)


def _yut_where(point) -> str:
    return "at the start" if point is None else "home" if point == "done" else f"at {point}"


def yut_table_text(view, names, legal) -> str:
    me = view.seat
    lines = []
    for i, name in enumerate(names):
        pieces = ", ".join(f"piece {n + 1} {_yut_where(p)}" for n, p in enumerate(view.pieces[i]))
        lines.append(f"{name}{' (you)' if i == me else ''}{'' if view.active[i] else ' (out of the game)'}: {pieces}")
    if view.bank:
        lines.append("Banked throws: " + ", ".join(view.bank))
    for item in view.history[-HOLDEM_RECENT:]:
        who = names[item.seat]
        if item.kind == "throw":
            lines.append(f"Earlier: {who} threw {item.name}")
        elif item.kind == "move":
            lines.append(f"Earlier: {who} moved piece {'+'.join(str(p + 1) for p in item.pieces)} with {item.throw}"
                         + (f" and captured {names[item.captured[0][0]]}" if item.captured else "") + (" and brought it home" if item.end == "done" else ""))
        elif item.kind == "forfeit":
            lines.append(f"Earlier: {who} left the game")
    lines.append("Legal moves:")
    for move in legal:
        if not move.startswith("move:"):
            lines.append(f"- {move}")
            continue
        info = yut.preview(view, me, move)
        note = f"piece {info['piece'] + 1} {_yut_where(info['start'])} with {info['throw']} goes " + ("home" if info["home"] else "back to the start" if info["end"] is None else f"to {info['end']}")
        if info["captured"]:
            note += f", captures {info['captured']} piece" + ("" if info["captured"] == 1 else "s")
        lines.append(f"- {move}: {note}")
    return "\n".join(lines)


async def ask_yut(bot, guild_id, channel_id, character, view, names, legal, style=None) -> dict:
    """One call on the board (all of it public); returns the raw answer (the caller validates it). Raises on any failure."""
    legal = tuple(legal)
    request = SimpleNamespace(messages=[TurnMessage("system", YUT_SYSTEM.format(name=character["name"], persona=with_style(persona(character), style))),
                                        TurnMessage("user", yut_table_text(view, names, legal))])
    with bot.backend.pin(), capture_usage(guild_id, channel_id, "game"), log_scope(None), log_purpose("game"):
        return await bot.models.structured_compiled(ROLE, request, "yut_move", yut_schema(legal))
