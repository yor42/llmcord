"""The item kinds: one frozen entry per kind, looked up by key. Unknown kinds and audience mismatches are rejected, never ignored."""
from dataclasses import dataclass
from typing import Callable

from . import random_message
from .recipe import DEFAULT_LOCALE, Recipe, RecipeError, parse_common


@dataclass(frozen=True)
class Kind:
    key: str
    audience: str  # 'none' (self) in v1; later 'member', 'any', 'npc'
    validate: Callable  # (Recipe) -> None; raises RecipeError
    roll: Callable  # (recipe, seed, locale, extra_entries, replace_entries, params) -> Result
    validate_entries: Callable  # (recipe, extra_entries, replace_entries) -> None; raises RecipeError
    param_keys: tuple = ()


_KINDS = {
    'random_message': Kind('random_message', 'none', random_message.validate, random_message.roll, random_message.validate_entries, random_message.PARAM_KEYS),
}


def get_kind(key) -> Kind:
    try:
        return _KINDS[key]
    except (KeyError, TypeError):
        raise RecipeError(f'unknown kind {key!r}') from None


def validate_recipe(data) -> Recipe:
    recipe = parse_common(data)
    kind = get_kind(recipe.kind)
    if recipe.audience != kind.audience:
        raise RecipeError(f'audience {recipe.audience!r} does not match kind {kind.key!r} ({kind.audience!r})')
    kind.validate(recipe)
    return recipe


def roll(recipe: Recipe, seed, locale=DEFAULT_LOCALE, extra_entries=None, replace_entries=None, params=None):
    return get_kind(recipe.kind).roll(recipe, seed, locale, extra_entries, replace_entries, params)


def validate_entries(recipe: Recipe, extra_entries=None, replace_entries=None):
    return get_kind(recipe.kind).validate_entries(recipe, extra_entries, replace_entries)
