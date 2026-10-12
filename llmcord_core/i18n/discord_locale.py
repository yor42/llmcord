"""Discord edge: slash-command translations and locale choice (FEAT-43, D39)."""
from __future__ import annotations

import logging

from discord import app_commands

from .catalog import available_locales, load_catalog
from .negotiate import normalize, negotiate

log = logging.getLogger('llmcord_core.i18n')

_NAMES = (
    app_commands.TranslationContextLocation.command_name,
    app_commands.TranslationContextLocation.group_name,
    app_commands.TranslationContextLocation.parameter_name,
)
_LIMITED = (
    app_commands.TranslationContextLocation.command_description,
    app_commands.TranslationContextLocation.group_description,
    app_commands.TranslationContextLocation.parameter_description,
    app_commands.TranslationContextLocation.choice_name,
)


def C_(msgid, /) -> app_commands.locale_str:
    return app_commands.locale_str(msgid)


class LlmcordTranslator(app_commands.Translator):
    def __init__(self, root=None):
        super().__init__()
        self.root = root

    async def translate(self, string, locale, context):
        if context.location in _NAMES:
            return None
        loc = negotiate([locale], available_locales(self.root))
        if loc in ('en', 'pseudo'):
            return None
        text = load_catalog(loc, self.root).get(string.extras.get('context'), string.message)
        if text is None:
            return None
        if context.location in _LIMITED and len(text) > 100:
            log.warning('translation too long for %s: locale=%s msgid=%r', context.location.name, loc, string.message)
            return None
        return text


def invoker_locale(interaction, available=None) -> str:
    return negotiate([interaction.locale], available or available_locales())


def invoker_t(interaction):
    """Translator for a message only the invoking user sees (ephemeral replies, modals, autocomplete).

    English until FEAT-44, which returns Translator(invoker_locale(interaction)).
    """
    from . import english
    return english()


def channel_t(guild):
    """Translator for a message the channel sees.

    Resolve it at each render; never capture it in a timer or closure.
    English until FEAT-44, which uses the server bot language.
    """
    from . import english
    return english()


def channel_locale(server_language, guild, available=None) -> str:
    available = available or available_locales()
    chosen = normalize(server_language)
    if chosen in available:
        return chosen
    if guild is not None and 'COMMUNITY' in (getattr(guild, 'features', None) or ()):
        return negotiate([guild.preferred_locale], available)
    return 'en'
