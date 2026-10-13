"""FEAT-49a: pure pieces of the decision-model veto (wire format, answer reading, policy). Standard library only; no network.

A decision model (microsoft/microsoft-decision-1 on OpenRouter) is asked one yes/no question per move the local e5 reading
offered and returns a probability for each. apply_policy drops moves it scores below TAU and never adds one. The HTTP side is
veto_client.py. Wire format and wording come from the FEAT-47/44 evaluation (docs/engineering/research/decision-models-eval.md).
"""
import math

ORIGIN = "https://openrouter.ai"
PATH = "/api/alpha/decisions"
MODEL = "microsoft/microsoft-decision-1"
TAU = 0.5
MAX_CHARS = 200

# Internal diagnostics for VetoError; never shown to a user.
BAD_LENGTH = "bad message length"
NO_MOVES = "no legal moves"
NO_ANSWERS = "no answers"
BAD_ANSWER = "unusable answer"
BAD_PROBABILITY = "unusable probability"

MEANING = {"hit": "take another card", "stand": "keep the hand, take no more cards", "double": "double the bet and take exactly one more card",
           "surrender": "give up the hand and lose half the bet", "insure": "buy insurance against a dealer blackjack",
           "no_insurance": "decline insurance", "even_money": "take even money on own blackjack against a dealer ace"}
CONTEXT = ("This message was typed in a Discord blackjack table chat by the player whose turn it is, as a possible move for the current hand "
           "(it may be English, Korean or mixed).")


class VetoError(Exception):
    """Any failure to get a usable reading; messages never carry the key, the message text or the response body."""


def predicate_text(move):
    return f"Is the player choosing to {MEANING.get(move, move)} right now, as an instruction to act (not a question, a joke or chit-chat)?"


def build_request(text, legal):
    """The JSON body for one message and the legal moves; raises VetoError for empty or oversized text or no moves."""
    if not isinstance(text, str) or not text or len(text) > MAX_CHARS:
        raise VetoError(BAD_LENGTH)
    if not legal or not all(isinstance(m, str) and m for m in legal):
        raise VetoError(NO_MOVES)
    return {"model": MODEL, "state": {"context": CONTEXT, "message": text},
            "questions": {f"move_{m}": {"type": "noul", "instructions": predicate_text(m)} for m in legal}}


def parse_response(data, legal):
    """{move: probability in [0, 1]} for every legal move; raises VetoError on anything else, including a refusal."""
    answers = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(answers, dict):
        raise VetoError(NO_ANSWERS)
    out = {}
    for move in legal:
        answer = answers.get(f"move_{move}")
        if not isinstance(answer, dict) or answer.get("type") == "refusal" or "noul" not in answer:
            raise VetoError(BAD_ANSWER)
        p = answer["noul"]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise VetoError(BAD_PROBABILITY)
        out[move] = float(p)
    return out


def apply_policy(band, moves, probs, tau=TAU):
    """(band, moves) after the veto. Moves scored below tau (missing counts 0) are dropped and the rest ordered by probability
    (ties keep e5 order); none left is silent. Sure survives only when its single move is kept; nothing is promoted to sure."""
    moves = list(moves)
    if band == "silent":
        return band, tuple(moves)
    kept = sorted((m for m in moves if probs.get(m, 0.0) >= tau), key=lambda m: -probs.get(m, 0.0))
    if not kept:
        return "silent", ()
    sure = band == "sure" and kept == moves and len(kept) == 1
    return ("sure" if sure else "unsure"), tuple(kept)
