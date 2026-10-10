"""Blackjack on Discord (FEAT-19 part C): commands, buttons, timers and restart recovery.

Rules, money and persistence live in ``games`` and the store (part A and B). This module only turns
presses into store calls and store snapshots into messages. Every table runs under its own lock (never
the channel skit lock); timers re-check the stored state under that lock, so a stale timer does nothing.
"""
from __future__ import annotations

import asyncio
import logging
import time
import weakref
from dataclasses import dataclass, field
from types import SimpleNamespace

import discord
from discord import app_commands

from .admin_store import ConflictError, GameError
from .games import DefaultPolicy, blackjack

STALE = "That table has moved on — use the latest buttons."
OFF_HERE = "Games are off in this channel."
BUSY = "A round is being played; join the next one with /blackjack when it ends."
NOT_SEATED = "You're not seated at this table."
NOT_YOUR_TURN = "It's not your turn."
MAX_BET = 100_000
PREFIX = "llmcord:bj"
NO_MENTIONS = discord.AllowedMentions.none()


@dataclass
class _Table:
    bets: dict = field(default_factory=dict)  # member id -> the bet they joined with
    present: set = field(default_factory=set)  # members still at the table between rounds
    deadline: float | None = None
    join_round: int | None = None  # round whose join timer is armed
    timeouts: set = field(default_factory=set)  # (round id, seat index)


class GameButton(discord.ui.DynamicItem[discord.ui.Button], template=PREFIX + r":(?P<action>[a-z]+):(?P<table>\d+):(?P<round>\d+):(?P<moves>\d+)"):
    """One button; its custom_id carries the table, the round and the move count the message showed."""

    def __init__(self, action: str, table_id: int, round_id: int, moves: int, label: str = "", style=discord.ButtonStyle.secondary):
        super().__init__(discord.ui.Button(label=label, style=style, custom_id=f"{PREFIX}:{action}:{table_id}:{round_id}:{moves}"))
        self.action, self.table_id, self.round_id, self.moves = action, table_id, round_id, moves

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["action"], int(match["table"]), int(match["round"]), int(match["moves"]), item.label or "", item.style)

    async def callback(self, interaction: discord.Interaction):
        await interaction.client.games.press(interaction, self.action, self.table_id, self.round_id, self.moves)


def _sign(net: int) -> str:
    return f"+{net:,}" if net > 0 else f"−{-net:,}" if net < 0 else ""


def _outcome(seat: dict) -> str:
    outcome, net = seat["outcome"], (seat["payout"] or 0) - seat["stake"]
    if outcome == "push":
        return "push"
    if outcome == "refund":
        return "refunded"
    return f"{outcome} {_sign(net)}".strip()


class GameTables:
    JOIN_SECONDS = 30
    TURN_SECONDS = 60
    IDLE_SECONDS = 120
    RETRY_SECONDS = 5

    def __init__(self, bot, clock=time.time):
        self.bot, self.clock = bot, clock
        self.locks: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self.timers: dict[tuple[int, int], asyncio.Task] = {}
        self.tables: dict[tuple[int, int], _Table] = {}
        self.closed = False

    @property
    def store(self):
        return self.bot.store

    def lock(self, key) -> asyncio.Lock:
        lock = self.locks.get(key)
        if lock is None:
            lock = self.locks[key] = asyncio.Lock()
        return lock

    def table(self, key) -> _Table:
        return self.tables.setdefault(key, _Table())

    # --- timers -----------------------------------------------------------------------------

    def arm(self, key, kind: str, round_id: int, moves: int = 0) -> None:
        if self.closed:
            return
        delay = {"join": self.JOIN_SECONDS, "turn": self.TURN_SECONDS, "idle": self.IDLE_SECONDS}[kind]
        self.disarm(key)
        self.table(key).deadline = self.clock() + delay
        task = self.timers[key] = asyncio.create_task(self._timer(key, kind, round_id, moves, delay), name=f"blackjack-{kind}-{key[1]}")
        task.add_done_callback(lambda done: self.timers.get(key) is done and self.timers.pop(key, None))

    def disarm(self, key) -> None:
        old = self.timers.pop(key, None)
        if old is not None and old is not asyncio.current_task():
            old.cancel()

    async def _timer(self, key, kind, round_id, moves, delay) -> None:
        await asyncio.sleep(delay)
        for attempt in (1, 2):  # one retry, so held bets do not wait for a restart after a transient failure
            try:
                if kind == "join":
                    await self.on_join_timer(*key, round_id)
                elif kind == "turn":
                    await self.on_turn_timer(*key, round_id, moves)
                else:
                    await self.on_idle_timer(*key, round_id)
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("Blackjack %s timer failed (table %s, attempt %d)", kind, key[1], attempt)
                if attempt == 1:
                    await asyncio.sleep(self.RETRY_SECONDS)

    async def close(self) -> None:
        self.closed = True
        tasks = list(self.timers.values())
        self.timers.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _after(self, key, snap) -> None:
        """Arms the timer the new state needs. Call under the table lock."""
        table, status = self.table(key), snap["status"]
        if status == "joining":
            if snap["seats"] and table.join_round != snap["round_id"]:
                table.join_round = snap["round_id"]
                self.arm(key, "join", snap["round_id"])
        elif status == "playing":
            self.arm(key, "turn", snap["round_id"], snap["moves"])
        elif status == "settled":
            table.present = {seat["ref_id"] for seat in snap["seats"]}
            self.arm(key, "idle", snap["round_id"])
        else:
            self.disarm(key)

    def _forget(self, key) -> None:
        self.disarm(key)
        self.tables.pop(key, None)

    # --- rendering --------------------------------------------------------------------------

    def render(self, snap) -> str:
        key = (snap["guild_id"], snap["table_id"])
        table, store, guild_id = self.table(key), self.store, snap["guild_id"]
        name, limits = store.currency_name(guild_id), store.game_settings(guild_id)
        status, view, deadline = snap["status"], snap["view"], table.deadline
        stamp = f"<t:{int(deadline)}:R>" if deadline else ""
        lines = ["**Blackjack**"]
        for seat in snap["seats"]:
            i = seat["index"]
            line = f"{i + 1}. <@{seat['ref_id']}> bet {(view['stakes'][i] if view else seat['stake']):,} {name}"
            if view:
                cards = view["hands"][i]
                line += f" — {blackjack.hand_str(cards)} ({blackjack.total_str(cards)})"
                if status == "playing":
                    word = {"stand": "stands", "bust": "bust", "blackjack": "blackjack"}.get(view["status"][i])
                    line += f" {word}" if word else ""
                    line += " (timed out)" if (snap["round_id"], i) in table.timeouts else ""
            if status == "settled":
                line += f" — {_outcome(seat)}"
                line += " (timed out)" if (snap["round_id"], i) in table.timeouts else ""
            lines.append(line)
        if view:
            dealer = view["dealer"]
            if view["dealer_hidden"]:
                lines.append(f"Dealer: {blackjack.hand_str(dealer)} ??")
            else:
                text = f"Dealer: {blackjack.hand_str(dealer)} ({blackjack.total_str(dealer)})"
                lines.append(text + (" bust" if blackjack.hand_total(dealer)[0] > 21 else ""))
        if status == "joining":
            lines.append(f"Join with /blackjack <bet> — dealing {stamp}." if stamp else "Join with /blackjack <bet>.")
            lines.append(f"Bets: {limits['min_bet']:,} to {limits['max_bet']:,} {name}.")
        elif status == "playing" and view and view["turn"] is not None:
            lines.append(f"On turn: <@{snap['seats'][view['turn']]['ref_id']}>" + (f" — decide {stamp}." if stamp else "."))
        elif status == "settled":
            lines.append(f"Play again or leave. The table closes {stamp} if nobody plays again." if stamp else "Play again or leave.")
        if status == "settled" and snap["seed"]:
            lines.append(f"Seed: {snap['seed']} (hash {snap['seed_hash'][:12]})")
        else:
            lines.append(f"Seed hash: {snap['seed_hash'][:12]}")
        return "\n".join(lines)

    def buttons(self, snap) -> discord.ui.View | None:
        status, table_id, round_id, moves = snap["status"], snap["table_id"], snap["round_id"], snap["moves"]
        spec = []
        if status == "joining":
            spec = [("deal", "Deal now", discord.ButtonStyle.primary), ("leave", "Leave", discord.ButtonStyle.secondary)]
        elif status == "playing":
            legal = next(iter(snap["legal"].values()), ())
            spec = [(m, m.capitalize(), discord.ButtonStyle.primary if m == "hit" else discord.ButtonStyle.secondary) for m in ("hit", "stand", "double") if m in legal]
        elif status == "settled":
            spec = [("again", "Play again (same bet)", discord.ButtonStyle.success), ("close", "Leave table", discord.ButtonStyle.secondary)]
        if not spec:
            return None
        view = discord.ui.View(timeout=None)
        for action, label, style in spec:
            view.add_item(GameButton(action, table_id, round_id, moves, label, style))
        return view

    async def _partial(self, channel_id, message_id):
        channel = self.bot.get_channel(channel_id) or await self.bot.fetch_channel(channel_id)
        return channel.get_partial_message(message_id)

    async def edit(self, channel_id, message_id, content, view=None) -> None:
        """Best-effort edit of the table message; a deleted message or a Discord error is logged, not raised."""
        if message_id is None:
            return
        try:
            partial = await self._partial(channel_id, message_id)
            await partial.edit(content=content, view=view, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning("Could not edit blackjack message %s in channel %s: %s", message_id, channel_id, type(error).__name__)
        except Exception:
            logging.exception("Could not edit blackjack message %s in channel %s", message_id, channel_id)

    async def refresh(self, snap) -> None:
        await self.edit(snap["channel_id"], snap["message_id"], self.render(snap), self.buttons(snap))

    async def _respond(self, interaction, snap) -> None:
        """Answer a button press by editing the message the button is on; if Discord refuses, edit it by channel so it is never left stale."""
        try:
            await interaction.response.edit_message(content=self.render(snap), view=self.buttons(snap), allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning("Blackjack press response failed (%s); refreshing the table message", type(error).__name__)
            await self.refresh(snap)

    async def _respond_closed(self, interaction, snap) -> None:
        try:
            await interaction.response.edit_message(content="Table closed.", view=None, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning("Blackjack press response failed (%s); editing the table message", type(error).__name__)
            await self.edit(snap["channel_id"], snap["message_id"], "Table closed.", None)

    async def _say(self, interaction, text: str) -> None:
        try:
            await interaction.response.send_message(text, ephemeral=True, allowed_mentions=NO_MENTIONS)
        except discord.HTTPException as error:
            logging.warning("Blackjack reply failed: %s", type(error).__name__)

    # --- table life -------------------------------------------------------------------------

    def _close_in_store(self, key) -> None:
        self.store.close_table(*key)
        self._forget(key)

    async def set_channel(self, guild_id, channel_id, enabled: bool) -> bool:
        """Turns games on or off in a channel. Returns True when an open table was closed (and refunded)."""
        table = None if enabled else self.store.open_table_for(guild_id, channel_id)
        if table is None:
            self.store.set_game_channel(guild_id, channel_id, enabled)
            return False
        async with self.lock((guild_id, table["id"])):
            current = self.store.open_table_for(guild_id, channel_id) or table
            closed = self.store.set_game_channel(guild_id, channel_id, False)["closed_table"]
            if closed is not None:
                self._forget((guild_id, closed))
                message_id = current["message_id"] if current["id"] == closed else table["message_id"]
                await self.edit(channel_id, message_id, "Table closed: games were turned off here.", None)
        return closed is not None

    # --- /blackjack -------------------------------------------------------------------------

    async def play_command(self, interaction: discord.Interaction, bet: int) -> None:
        guild_id = interaction.guild_id
        channel_id = getattr(interaction, "channel_id", None) or interaction.channel.id
        if interaction.user.bot:
            await self._say(interaction, "Bots cannot play blackjack.")
            return
        if channel_id not in self.store.game_channels(guild_id):
            await self._say(interaction, OFF_HERE)
            return
        opened = None
        table = self.store.open_table_for(guild_id, channel_id)
        if table is None:
            try:
                opened = self.store.open_table(guild_id, channel_id, "blackjack", interaction.user.id)
            except GameError as error:
                await self._say(interaction, str(error))
                return
            table = {"id": opened["table_id"]}
        key = (guild_id, table["id"])
        async with self.lock(key):
            try:
                snap = self.store.latest_round(*key)
                if snap["status"] == "playing":
                    await self._say(interaction, BUSY)
                    return
                fresh = snap["status"] != "joining"
                if fresh:
                    snap = self.store.next_round(*key)
                try:
                    snap = self.store.join_round(guild_id, snap["round_id"], interaction.user.id, bet)
                except GameError:
                    if fresh and opened is None:
                        self.store.cancel_round(guild_id, snap["round_id"], "nobody joined")
                    raise
            except GameError as error:
                if opened is not None:
                    self._close_in_store(key)
                await self._say(interaction, str(error))
                return
            self.table(key).bets[interaction.user.id] = bet
            self.table(key).present.add(interaction.user.id)
            self._after(key, snap)
            name = self.store.currency_name(guild_id)
            if opened is not None:
                await self._open_message(interaction, key, snap)
            else:
                await self._say(interaction, f"You joined the table with {bet:,} {name}.")
                await self.refresh(snap)

    async def _open_message(self, interaction, key, snap) -> None:
        try:
            await interaction.response.send_message(self.render(snap), view=self.buttons(snap), allowed_mentions=NO_MENTIONS)
            message = await interaction.original_response()
            self.store.set_table_message(*key, message.id)
        except Exception:
            self._close_in_store(key)  # refunds; no table without a message
            raise

    # --- buttons ----------------------------------------------------------------------------

    async def press(self, interaction: discord.Interaction, action: str, table_id: int, round_id: int, moves: int) -> None:
        guild_id = interaction.guild_id
        if guild_id is None or interaction.user.bot:
            await self._say(interaction, STALE)
            return
        key = (guild_id, table_id)
        async with self.lock(key):
            try:
                snap = self.store.round_snapshot(guild_id, round_id)
            except GameError:
                await self._say(interaction, STALE)
                return
            if snap["table_id"] != table_id:
                await self._say(interaction, STALE)
                return
            seat = next((s["index"] for s in snap["seats"] if s["kind"] == "member" and s["ref_id"] == interaction.user.id), None)
            if seat is None:
                await self._say(interaction, NOT_SEATED)
                return
            try:
                if action in ("hit", "stand", "double"):
                    await self._move(interaction, key, snap, seat, action, moves)
                elif action == "deal":
                    await self._deal(interaction, key, snap)
                elif action == "leave":
                    await self._leave(interaction, key, snap)
                elif action == "again":
                    await self._again(interaction, key, interaction.user.id)
                elif action == "close":
                    await self._leave_table(interaction, key, snap)
                else:
                    await self._say(interaction, STALE)
            except ConflictError:
                await self._say(interaction, STALE)
            except GameError as error:
                await self._say(interaction, str(error))

    async def _move(self, interaction, key, snap, seat, move, moves) -> None:
        if snap["status"] != "playing" or snap["moves"] != moves:
            await self._say(interaction, STALE)
            return
        if seat not in snap["legal"]:
            await self._say(interaction, NOT_YOUR_TURN)
            return
        snap = self.store.play(key[0], snap["round_id"], seat, move, "member", moves)
        self._after(key, snap)
        await self._respond(interaction, snap)

    async def _deal(self, interaction, key, snap) -> None:
        snap = self.store.deal(key[0], snap["round_id"])
        self._after(key, snap)
        await self._respond(interaction, snap)

    async def _leave(self, interaction, key, snap) -> None:
        snap = self.store.leave_round(key[0], snap["round_id"], interaction.user.id)
        self.table(key).bets.pop(interaction.user.id, None)
        if not snap["seats"]:
            self._close_in_store(key)
            await self._respond_closed(interaction, snap)
            return
        await self._respond(interaction, snap)

    async def _again(self, interaction, key, user_id) -> None:
        guild_id, table = key[0], self.table(key)
        snap = self.store.latest_round(*key)
        if snap["status"] == "playing":
            await self._say(interaction, BUSY)
            return
        bet = table.bets.get(user_id)
        if bet is None:
            await self._say(interaction, STALE)
            return
        fresh = snap["status"] != "joining"
        if fresh:
            snap = self.store.next_round(*key)
        try:
            snap = self.store.join_round(guild_id, snap["round_id"], user_id, bet)
        except GameError:
            if fresh:
                self.store.cancel_round(guild_id, snap["round_id"], "nobody joined")
            raise
        table.present.add(user_id)
        self._after(key, snap)
        await self._respond(interaction, snap)

    async def _leave_table(self, interaction, key, snap) -> None:
        user_id, table = interaction.user.id, self.table(key)
        latest = self.store.latest_round(*key)
        seated = any(seat["kind"] == "member" and seat["ref_id"] == user_id for seat in latest["seats"])
        if latest["status"] == "playing" and seated:
            await self._say(interaction, BUSY)
            return
        if latest["status"] == "joining" and seated:
            latest = self.store.leave_round(key[0], latest["round_id"], user_id)  # refunds the bet
            if not latest["seats"]:
                self._close_in_store(key)
                await self._respond_closed(interaction, latest)
                return
            await self.refresh(latest)
        table.present.discard(user_id)
        table.bets.pop(user_id, None)
        if table.present or latest["status"] in ("joining", "playing"):
            await self._say(interaction, "You left the table.")
            return
        self._close_in_store(key)
        await self._respond_closed(interaction, latest)

    # --- timers' work -----------------------------------------------------------------------

    async def on_join_timer(self, guild_id, table_id, round_id) -> None:
        key = (guild_id, table_id)
        async with self.lock(key):
            snap = self.store.round_snapshot(guild_id, round_id)
            if snap["status"] != "joining" or not snap["seats"]:
                return
            snap = self.store.deal(guild_id, round_id)
            self._after(key, snap)
            await self.refresh(snap)

    async def on_turn_timer(self, guild_id, table_id, round_id, moves) -> None:
        key = (guild_id, table_id)
        async with self.lock(key):
            snap = self.store.round_snapshot(guild_id, round_id)
            if snap["status"] != "playing" or snap["moves"] != moves or not snap["legal"]:
                return
            seat, legal = next(iter(snap["legal"].items()))
            move = DefaultPolicy.pick(SimpleNamespace(total=blackjack.hand_total(snap["view"]["hands"][seat])[0]), legal)
            snap = self.store.play(guild_id, round_id, seat, move, "timeout", moves)
            self.table(key).timeouts.add((round_id, seat))
            self._after(key, snap)
            await self.refresh(snap)

    async def on_idle_timer(self, guild_id, table_id, round_id) -> None:
        key = (guild_id, table_id)
        async with self.lock(key):
            snap = self.store.latest_round(guild_id, table_id)
            open_table = self.store.open_table_for(guild_id, snap["channel_id"])
            if open_table is None or open_table["id"] != table_id or snap["status"] in ("joining", "playing"):
                return
            self._close_in_store(key)
            await self.edit(snap["channel_id"], snap["message_id"], "Table closed.", None)

    # --- restart ----------------------------------------------------------------------------

    async def recover(self) -> None:
        """Refunds and closes every table left open by a previous run, then tells the channel (best effort)."""
        store, refunded, closed, failed = self.store, 0, 0, 0
        rounds = store.unfinished_rounds()
        tables = {(r["guild_id"], r["table_id"]): (r["channel_id"], r["message_id"]) for r in rounds}
        for row in store.all("SELECT id,guild_id,channel_id,message_id FROM game_tables WHERE status='open'"):
            tables.setdefault((row["guild_id"], row["id"]), (row["channel_id"], row["message_id"]))
        for r in rounds:
            try:
                store.cancel_round(r["guild_id"], r["round_id"], "bot restarted")
                refunded += 1
            except Exception:
                failed += 1
                logging.exception("Could not refund blackjack round %s on restart", r["round_id"])
        for (guild_id, table_id), (channel_id, message_id) in tables.items():
            try:
                store.close_table(guild_id, table_id)
                closed += 1
            except Exception:
                failed += 1
                logging.exception("Could not close blackjack table %s on restart", table_id)
                continue
            await self.edit(channel_id, message_id, "This table closed when the bot restarted. Bets were refunded.", None)
        if refunded or closed or failed:
            logging.info("Blackjack recovery: %d rounds refunded, %d tables closed, %d failures", refunded, closed, failed)


def register_game_commands(bot, ctx: SimpleNamespace, admin_games: app_commands.Group) -> None:
    require_guild = ctx.require_guild

    async def reply(interaction: discord.Interaction, message: str) -> None:
        await interaction.response.send_message(message, ephemeral=True, allowed_mentions=NO_MENTIONS)

    @bot.tree.command(name="blackjack", description="Join or open a blackjack table in this channel")
    @app_commands.describe(bet="How much to bet")
    async def blackjack_command(interaction: discord.Interaction, bet: app_commands.Range[int, 1, MAX_BET]):
        require_guild(interaction)
        await bot.games.play_command(interaction, bet)

    @admin_games.command(name="channel", description="Turn games on or off in a channel")
    @app_commands.describe(enabled="On or off", channel="The channel (default: this one)")
    @app_commands.checks.has_permissions(administrator=True)
    async def games_channel(interaction: discord.Interaction, enabled: bool, channel: discord.TextChannel | None = None):
        require_guild(interaction)
        channel_id = channel.id if channel else getattr(interaction, "channel_id", None) or interaction.channel.id
        where = channel.mention if channel else f"<#{channel_id}>"
        closed = await bot.games.set_channel(interaction.guild_id, channel_id, enabled)
        text = f"Games are now {'on' if enabled else 'off'} in {where}."
        await reply(interaction, text + (" The open table was closed and bets were refunded." if closed else ""))

    @admin_games.command(name="bets", description="Set the smallest and largest bet")
    @app_commands.describe(min="Smallest bet", max="Largest bet")
    @app_commands.checks.has_permissions(administrator=True)
    async def games_bets(interaction: discord.Interaction, min: app_commands.Range[int, 1, MAX_BET], max: app_commands.Range[int, 1, MAX_BET]):
        require_guild(interaction)
        try:
            saved = bot.store.set_game_settings(interaction.guild_id, min, max, expected=bot.store.game_settings(interaction.guild_id))
        except ConflictError:
            await reply(interaction, "The game settings were changed elsewhere. Run the command again.")
            return
        name = bot.store.currency_name(interaction.guild_id)
        await reply(interaction, f"Bets are now {saved['min_bet']:,} to {saved['max_bet']:,} {name}.")
