"""The vault_tool kind: a consumable used inside a personal vault (FEAT-64, D43). It has no roll; /use refuses it and the Vault store spends it."""
from ..i18n import codes
from .recipe import RecipeError

PARAM_KEYS = ()


def validate(recipe):
    if recipe.params or recipe.tables or recipe.template:
        raise RecipeError(codes.RECIPE_UNKNOWN_PARAMS, params='params, tables, template')


def validate_entries(recipe, extra_entries=None, replace_entries=None):
    return {}, {}


def roll(recipe, seed, locale, extra_entries=None, replace_entries=None, params=None):
    raise RecipeError(codes.RECIPE_UNKNOWN_KIND, kind=repr(recipe.kind))
