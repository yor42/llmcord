"""FEAT-20: a character's blackjack decision as one structured model call.

The prompt holds only the character's name and own card text (description and personality), the visible table and
the legal moves: no lore, scene history, memory or member identities, so nothing private can reach the model.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from .games import blackjack
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


async def ask(bot, guild_id, channel_id, character, snap, seat, legal) -> dict:
    """One model call on a pinned backend snapshot, recorded as usage feature 'game'. Raises on any failure (the caller falls back)."""
    legal = tuple(legal)
    request = SimpleNamespace(messages=[TurnMessage("system", SYSTEM.format(name=character["name"], persona=persona(character))),
                                        TurnMessage("user", table_text(snap, seat, legal))])
    with bot.backend.pin(), capture_usage(guild_id, channel_id, "game"), log_scope(None), log_purpose("game"):
        return await bot.models.structured_compiled(ROLE, request, "blackjack_move", schema(legal))
