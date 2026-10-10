"""Characterization (ARCH-01): the full slash-command tree is pinned.

Guards the `register_commands` split: every command's qualified name,
description, parameters (name, type, required, description, autocomplete,
choices), check count, default permissions and guild_only flag, plus every
group, must stay identical.
"""

import unittest

from discord import app_commands
from helpers import make_settings

from llmcord_core.discord_bot import SkitBot

# command -> (description, params, check count, default_permissions, guild_only)
# param -> (name, type, required, description, has_autocomplete, ((choice name, value), ...))
EXPECTED_COMMANDS = {
    "admin ambient off": (
        "Disable ambient participation in this channel",
        (),
        1,
        "None",
        False,
    ),
    "admin ambient on": (
        "Enable ambient participation in this channel",
        (),
        1,
        "None",
        False,
    ),
    "admin cast default": (
        "Set the channel's default cast",
        (("characters", "string", True, "…", True, ()),),
        1,
        "None",
        False,
    ),
    "admin character import": (
        "Import a V2/V3 JSON or PNG character card",
        (
            ("world", "string", True, "…", True, ()),
            ("attachment", "attachment", True, "…", False, ()),
        ),
        1,
        "None",
        False,
    ),
    "admin currency daily": (
        "Set the daily check-in bonus (amount 0 turns it off)",
        (
            ("amount", "integer", True, "Currency paid for each check-in (0 turns check-ins off)", False, ()),
            ("streak_bonus", "integer", False, "Extra per consecutive day (default: keep the current value)", False, ()),
            ("streak_days", "integer", False, "How many streak days earn the bonus, up to 365 (default: keep the current value)", False, ()),
        ),
        1,
        "None",
        False,
    ),
    "admin games bets": (
        "Set the smallest and largest bet",
        (
            ("min", "integer", True, "Smallest bet", False, ()),
            ("max", "integer", True, "Largest bet", False, ()),
        ),
        1,
        "None",
        False,
    ),
    "admin games channel": (
        "Turn games on or off in a channel",
        (
            ("enabled", "boolean", True, "On or off", False, ()),
            ("channel", "channel", False, "The channel (default: this one)", False, ()),
        ),
        1,
        "None",
        False,
    ),
    "admin currency grant": (
        "Give a member currency",
        (
            ("member", "user", True, "Who receives it", False, ()),
            ("amount", "integer", True, "How much to give", False, ()),
            ("reason", "string", True, "Why (kept in the ledger)", False, ()),
        ),
        1,
        "None",
        False,
    ),
    "admin currency name": (
        "Set what this server calls its currency",
        (("name", "string", True, "For example coins or gold (1 to 32 characters)", False, ()),),
        1,
        "None",
        False,
    ),
    "admin currency revoke": (
        "Take currency from a member",
        (
            ("member", "user", True, "Who loses it", False, ()),
            ("amount", "integer", True, "How much to take", False, ()),
            ("reason", "string", True, "Why (kept in the ledger)", False, ()),
        ),
        1,
        "None",
        False,
    ),
    "admin lore add": (
        "Add lore to this channel (or thread) or to its world or hub",
        (
            ("content", "string", True, "…", False, ()),
            ("keys", "string", False, "…", False, ()),
            (
                "scope",
                "string",
                False,
                "…",
                False,
                (("channel", "channel"), ("space", "space")),
            ),
        ),
        1,
        "None",
        False,
    ),
    "admin lore delete": (
        "Delete a lore entry",
        (("lore_id", "integer", True, "…", False, ()),),
        1,
        "None",
        False,
    ),
    "admin lore edit": (
        "Correct the text of a lore entry",
        (
            ("lore_id", "integer", True, "…", False, ()),
            ("content", "string", True, "…", False, ()),
        ),
        1,
        "None",
        False,
    ),
    "admin lore pin": (
        "Pin a lore entry",
        (("lore_id", "integer", True, "…", False, ()),),
        1,
        "None",
        False,
    ),
    "admin lore promote": (
        "Copy lore into a channel or world/hub",
        (
            ("lore_id", "integer", True, "…", False, ()),
            (
                "destination",
                "string",
                True,
                "…",
                False,
                (("channel", "channel"), ("space", "space")),
            ),
            ("space", "string", False, "…", True, ()),
        ),
        1,
        "None",
        False,
    ),
    "admin scene delete": (
        "Delete a stored message and everything after it in its branch",
        (
            (
                "message_id",
                "string",
                True,
                "Any stored message in this channel's scene",
                False,
                (),
            ),
        ),
        1,
        "None",
        False,
    ),
    "admin space bind": (
        "Bind a text channel to a world or hub",
        (
            ("channel", "channel", True, "…", False, ()),
            ("space", "string", True, "…", True, ()),
        ),
        1,
        "None",
        False,
    ),
    "admin space create": (
        "Create a world or hub",
        (
            ("kind", "string", True, "…", False, (("world", "world"), ("hub", "hub"))),
            ("name", "string", True, "…", False, ()),
        ),
        1,
        "None",
        False,
    ),
    "admin space link_world": (
        "Link a world to a hub so its characters can appear there",
        (
            ("hub", "string", True, "…", True, ()),
            ("world", "string", True, "…", True, ()),
        ),
        1,
        "None",
        False,
    ),
    "admin space unlink_world": (
        "Unlink a world from a hub and drop its characters from that "
        "hub's channel casts",
        (
            ("hub", "string", True, "…", True, ()),
            ("world", "string", True, "…", True, ()),
        ),
        1,
        "None",
        False,
    ),
    "daily": ("Claim your daily check-in bonus", (), 0, "None", False),
    "blackjack": (
        "Join or open a blackjack table in this channel",
        (("bet", "integer", True, "How much to bet", False, ()),),
        0,
        "None",
        False,
    ),
    "balance": (
        "Show your balance, or another member's",
        (("member", "user", False, "Whose balance to show (default: yours)", False, ()),),
        0,
        "None",
        False,
    ),
    "ambient status": ("Show this channel's ambient setting", (), 0, "None", False),
    "cast add": (
        "Add an eligible character to the active cast",
        (("character", "string", True, "…", True, ()),),
        0,
        "None",
        False,
    ),
    "cast remove": (
        "Remove a character from the active cast",
        (("character", "string", True, "…", True, ()),),
        0,
        "None",
        False,
    ),
    "cast set": (
        "Set the active cast with comma-separated names",
        (("characters", "string", True, "…", True, ()),),
        0,
        "None",
        False,
    ),
    "cast show": ("Show this channel or thread's active cast", (), 0, "None", False),
    "catchup": (
        "Privately summarize what you missed in this channel",
        (
            ("focus", "string", False, "What to emphasize, such as what concerns you", False, ()),
            ("hours", "integer", False, "Look back this many hours instead", False, ()),
        ),
        0,
        "None",
        False,
    ),
    "character info": (
        "Show a character's home world",
        (("character", "string", True, "…", True, ()),),
        0,
        "None",
        False,
    ),
    "character list": ("List characters available here", (), 0, "None", False),
    "context": (
        "Inspect what informed the last character line",
        (("message_id", "string", False, "…", False, ()),),
        1,
        "<Permissions value=8>",
        False,
    ),
    "favorites add": (
        "Add a character to your favorites",
        (("character", "string", True, "The character to add", True, ()),),
        0,
        "None",
        False,
    ),
    "favorites clear": ("Remove all your favorites", (), 0, "None", False),
    "favorites list": ("Show your favorites and your favorites mode", (), 0, "None", False),
    "favorites mode": (
        "Choose how far your favorites may go",
        (
            (
                "mode",
                "string",
                True,
                "Lean: favorites in the cast answer more. Step in: favorites outside the cast can answer too",
                False,
                (("Lean", "lean"), ("Step in", "step_in")),
            ),
        ),
        0,
        "None",
        False,
    ),
    "favorites remove": (
        "Remove a character from your favorites",
        (("character", "string", True, "The favorite to remove", True, ()),),
        0,
        "None",
        False,
    ),
    "lore list": ("Show the lore available in this location", (), 0, "None", False),
    "memory forget": (
        "Remove one of your personal memories",
        (("memory_id", "integer", True, "…", False, ()),),
        0,
        "None",
        False,
    ),
    "memory list": ("List what characters remember about you", (), 0, "None", False),
    "memory opt_in": (
        "Allow characters to remember facts you explicitly state",
        (),
        0,
        "None",
        False,
    ),
    "memory opt_out": (
        "Disable and erase your personal memories",
        (),
        0,
        "None",
        False,
    ),
    "scene reset": (
        "Start a fresh scene from your next invitation",
        (),
        0,
        "None",
        False,
    ),
    "space list": ("List the server's worlds and hubs", (), 0, "None", False),
    "summon": (
        "Invite an eligible character for one turn",
        (
            ("character", "string", True, "…", True, ()),
            ("prompt", "string", True, "…", False, ()),
        ),
        0,
        "None",
        False,
    ),
    "time clear": ("Remove your timezone in this server", (), 0, "None", False),
    "time set": (
        "Choose your timezone in this server",
        (("zone", "string", True, "IANA timezone name, such as Asia/Seoul", True, ()),),
        0,
        "None",
        False,
    ),
    "time show": ("Show the timezone characters use for you", (), 0, "None", False),
}

# group -> (description, default_permissions, guild_only)
EXPECTED_GROUPS = {
    "admin": ("Server administrator commands", "<Permissions value=8>", False),
    "admin ambient": ("Turn ambient participation on or off", "None", False),
    "admin cast": ("Set a channel's default cast", "None", False),
    "admin character": ("Import characters", "None", False),
    "admin currency": ("Give, take and name the server currency", "None", False),
    "admin games": ("Turn games on in channels and set bet limits", "None", False),
    "admin lore": ("Add and manage lore", "None", False),
    "admin scene": ("Delete stored scenes", "None", False),
    "admin space": ("Manage worlds, hubs and channel bindings", "None", False),
    "ambient": ("Show this channel's ambient setting", "None", False),
    "cast": ("Manage this channel or thread's active cast", "None", False),
    "character": ("Inspect characters available here", "None", False),
    "favorites": ("Keep favorite characters who are more likely to answer you", "None", False),
    "lore": ("Show the lore available here", "None", False),
    "memory": ("Control your personal character memories", "None", False),
    "scene": ("Start a fresh scene", "None", False),
    "space": ("List the server's worlds and hubs", "None", False),
    "time": ("Set the timezone characters use for you", "None", False),
}


def snapshot_commands(bot):
    commands, groups = {}, {}

    def walk(node):
        if isinstance(node, app_commands.Group):
            groups[node.qualified_name] = (
                node.description,
                str(node.default_permissions),
                node.guild_only,
            )
            for child in node.commands:
                walk(child)
            return
        params = tuple(
            (
                p.name,
                p.type.name,
                p.required,
                p.description,
                p.autocomplete,
                tuple((c.name, c.value) for c in p.choices),
            )
            for p in node.parameters
        )
        commands[node.qualified_name] = (
            node.description,
            params,
            len(node.checks),
            str(node.default_permissions),
            node.guild_only,
        )

    for node in bot.tree.get_commands():
        walk(node)
    return commands, groups


class CommandTreeTests(unittest.TestCase):
    def setUp(self):
        self.commands, self.groups = snapshot_commands(SkitBot(make_settings()))

    def test_command_names_are_stable(self):
        """Characterization (ARCH-01): the set of qualified command names."""
        self.assertEqual(sorted(self.commands), sorted(EXPECTED_COMMANDS))

    def test_group_tree_is_stable(self):
        """Characterization (ARCH-01): groups, descriptions and permissions."""
        self.assertEqual(self.groups, EXPECTED_GROUPS)

    def test_command_details_are_stable(self):
        """Characterization (ARCH-01): descriptions, parameters, checks, flags."""
        for name, expected in EXPECTED_COMMANDS.items():
            with self.subTest(command=name):
                self.assertEqual(self.commands[name], expected)


if __name__ == "__main__":
    unittest.main()
