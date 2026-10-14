"""FEAT-48 / D40: one public table view for the three games (what a spectator may know: never a hand, a face-down card, a seed or odds)."""
from __future__ import annotations


def _tables(bot):
    return {"doubt": bot.doubt, "holdem": bot.holdem, "blackjack": bot.games}


def _view(game, tables, guild_id, key, thread_id, context):
    text = tables[game].public_block(guild_id, key)
    if text is None:
        return None
    return {"game": game, "table_id": context["table_id"], "thread_id": thread_id, "status": context["status"],
            "seated_character_ids": list(context["seated_character_ids"]), "text": text, "moment": None}


def public_table_view(bot, guild_id, channel_id, parent_channel_id=None) -> dict | None:
    """The public view of the table that owns this thread, else of the open table of the channel (or of its parent); None without one."""
    tables = _tables(bot)
    for thread_id in (channel_id, parent_channel_id):
        for game in ("doubt", "holdem"):
            context = tables[game].table_talk_context(guild_id, thread_id) if thread_id is not None else None
            if context is not None:
                return _view(game, tables, guild_id, (guild_id, context["table_id"]), thread_id, context)
    for where in (channel_id, parent_channel_id):
        row = bot.store.open_table_for(guild_id, where) if where is not None else None
        if row is None or row["game"] not in tables:
            continue
        game = row["game"]
        context = tables[game].table_talk_context(guild_id, row["thread_id"] if game != "blackjack" else where)
        if context is not None:
            return _view(game, tables, guild_id, (guild_id, row["id"]), row["thread_id"] if game != "blackjack" else None, context)
    return None
