"""FEAT-20: a character's blackjack decision as one structured model call.

The prompt holds only the character's name and own card text (description and personality), the visible table and
the legal moves: no lore, scene history, memory or member identities, so nothing private can reach the model.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

from .games import blackjack, doubt, playstyle
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


def holdem_names(snap) -> list:
    """Prompt-safe names by seat: characters by name (one line, no markup or colons), members as Player 1, 2, ... (never member names or ids)."""
    names, players = [], 0
    for seat in snap["seats"]:
        if seat["kind"] == "character":
            name = " ".join(re.sub(r"[*_~`|>@<\\:]", " ", str(seat["name"])).split())[:40]
        else:
            players += 1
            name = f"Player {players}"
        names.append(name or f"Seat {len(names) + 1}")
    return names


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
