"""Shared user-facing error codes (errors domain): NAME = E_(dotted.code, English message). Filled from Phase 1 (MNT-66)."""
# Each batch appends only to its own section. NAME = E_('area.name', 'English message with {param}.')
from . import E_  # noqa: F401  (codes are added by the Phase 1 batches)

# --- MNT-66 (1a): admin_store.py, store.py



# --- MNT-67 (1b): other shared modules



# --- MNT-68 (1c): game engines



# --- MNT-69 (1d): command metadata



# --- MNT-70 (1e): discord_bot.py, favorites_discord.py, items_discord.py, usage.py
ATTACHMENT_TOO_BIG = E_('discord.attachment_too_big', 'An attachment exceeds the configured size limit')
TOO_MANY_IMAGES = E_('discord.too_many_images', 'Too many image attachments')
MODEL_NO_IMAGES = E_('discord.model_no_images', 'The configured dialogue model cannot read images')
ATTACHMENT_TYPE = E_('discord.attachment_type', 'Only text and image attachments are supported')
USE_IN_SERVER = E_('discord.use_in_server', 'Use this command in a server channel.')
NOT_BOUND = E_('discord.not_bound', 'This channel is not bound to a world or hub. Ask an administrator to bind it with /admin space bind.')
CARD_TOO_BIG = E_('discord.card_too_big', 'Card exceeds 8 MiB. Upload a smaller file.')
CAST_NAMES_NEEDED = E_('discord.cast_names_needed', 'Give one or more character names, separated by commas.')
LORE_NOT_FOUND = E_('discord.lore_not_found', 'Lore entry not found. Check the number with /lore list.')
DESTINATION_NOT_FOUND = E_('discord.destination_not_found', 'Destination world or hub not found. Check the name with /space list.')
NUMERIC_MESSAGE_ID = E_('discord.numeric_message_id', 'Give a numeric message ID.')
SCENE_MESSAGE_NOT_FOUND = E_('discord.scene_message_not_found', 'Scene message not found in this channel. Give the ID of a message in this scene.')
NO_SAVED_CONTEXT = E_('discord.no_saved_context', 'No saved context for that character line. Leave the ID empty to use the latest one.')



# --- MNT-71 (1f): game surfaces



# --- MNT-72 (1g): dashboard.py part 1



# --- MNT-73 (1h): dashboard.py part 2



# --- MNT-74 (1i): other dashboard modules

