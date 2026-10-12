"""Name resolution and autocomplete suggestions for slash command options (UX-04).

Callers pass rows already scoped to one guild (and space, where relevant); nothing here queries the store.
"""
from __future__ import annotations

from discord import app_commands

from .i18n import UserError, codes, terms

CHOICE_LIMIT = 25
SINGULAR = {'character': terms.NOUN_CHARACTER, 'world': terms.NOUN_WORLD, 'hub': terms.NOUN_HUB, 'world or hub': terms.NOUN_WORLD_OR_HUB}
PLURALS = {'character': terms.NOUN_CHARACTERS, 'world': terms.NOUN_WORLDS, 'hub': terms.NOUN_HUBS, 'world or hub': terms.NOUN_WORLDS_OR_HUBS}
NOT_FOUND = {'': codes.NAME_NOT_FOUND, ' in this server': codes.NAME_NOT_FOUND_IN_SERVER, ' available here': codes.NAME_NOT_FOUND_AVAILABLE_HERE,
             ' available here or in the cast': codes.NAME_NOT_FOUND_HERE_OR_CAST}
VALUE_LIMIT = 100


class NameNotFound(UserError):
    pass


def resolve(rows, text: str, noun: str, where: str = ''):
    """An exact name wins; otherwise one casefold match; several casefold matches or none raise ValueError."""
    wanted = text.strip()
    noun = SINGULAR.get(noun, noun)
    if not wanted:
        raise UserError(codes.NAME_GIVE, noun=noun)
    exact = [row for row in rows if row['name'] == wanted]
    if len(exact) == 1:
        return exact[0]
    folded = [row for row in rows if row['name'].strip().casefold() == wanted.casefold()]
    if len(folded) == 1:
        return folded[0]
    if folded:
        names = ', '.join(row['name'] for row in folded[:10])
        raise UserError(codes.NAME_AMBIGUOUS, plural=PLURALS.get(noun, noun + 's'), wanted=wanted, names=names)
    if where in NOT_FOUND:
        raise NameNotFound(NOT_FOUND[where], noun=noun, wanted=wanted)
    raise NameNotFound(codes.NAME_NOT_FOUND_WHERE, noun=noun, wanted=wanted, where=where)


def resolve_space(rows, text: str, kind: str | None = None):
    """Resolve among spaces of ``kind`` (any kind when None); a name that only matches the other kind says so."""
    if kind is None:
        return resolve(rows, text, 'world or hub')
    try:
        return resolve([row for row in rows if row['kind'] == kind], text, kind)
    except NameNotFound as missing:
        try:
            other = resolve([row for row in rows if row['kind'] != kind], text, 'world or hub')
        except ValueError:
            raise missing from None
        raise UserError(codes.NAME_WRONG_KIND, name=other['name'], actual=SINGULAR.get(other['kind'], other['kind']), kind=SINGULAR.get(kind, kind)) from None


def suggest(names, current: str, many: bool = False) -> list[app_commands.Choice[str]]:
    """Case-insensitive substring suggestions; with ``many`` the last comma-separated fragment is completed."""
    chosen, fragment = [], current
    if many:
        *chosen, fragment = current.split(',')
        chosen = [part.strip() for part in chosen if part.strip()]
    taken = {name.casefold() for name in chosen}
    needle = fragment.strip().casefold()
    choices = []
    for name in names:
        if needle not in name.casefold() or name.casefold() in taken:
            continue
        value = ', '.join([*chosen, name])
        if len(value) <= VALUE_LIMIT:
            choices.append(app_commands.Choice(name=value, value=value))
        if len(choices) >= CHOICE_LIMIT:
            break
    return choices
