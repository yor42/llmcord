"""Name resolution and autocomplete suggestions for slash command options (UX-04).

Callers pass rows already scoped to one guild (and space, where relevant); nothing here queries the store.
"""
from __future__ import annotations

from discord import app_commands

CHOICE_LIMIT = 25
PLURALS = {'world or hub': 'worlds or hubs'}
VALUE_LIMIT = 100


class NameNotFound(ValueError):
    pass


def resolve(rows, text: str, noun: str, where: str = ''):
    """An exact name wins; otherwise one casefold match; several casefold matches or none raise ValueError."""
    wanted = text.strip()
    if not wanted:
        raise ValueError(f'Give a {noun} name.')
    exact = [row for row in rows if row['name'] == wanted]
    if len(exact) == 1:
        return exact[0]
    folded = [row for row in rows if row['name'].strip().casefold() == wanted.casefold()]
    if len(folded) == 1:
        return folded[0]
    if folded:
        names = ', '.join(row['name'] for row in folded[:10])
        raise ValueError(f'Several {PLURALS.get(noun, noun + "s")} match {wanted}: {names}. Use the exact name.')
    raise NameNotFound(f'No {noun} named {wanted}{where}.')


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
        raise ValueError(f"{other['name']} is a {other['kind']}, not a {kind}.") from None


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
