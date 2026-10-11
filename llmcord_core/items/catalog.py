"""The built-in items, loaded once from recipes/*.json (one file per item). Prices are suggestions; servers decide their own."""
import json
from pathlib import Path

from .registry import validate_recipe

_DIR = Path(__file__).parent / 'recipes'


def _load():
    items = {}
    for path in sorted(_DIR.glob('*.json')):
        try:
            recipe = validate_recipe(json.loads(path.read_text(encoding='utf-8')))
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(f'bad built-in recipe {path.name}: {exc}') from exc
        if recipe.key != path.stem or recipe.key in items:
            raise ValueError(f'catalog file {path.name} does not match its key {recipe.key!r}')
        items[recipe.key] = recipe
    return items


_ITEMS = _load()


def all_items():
    return list(_ITEMS.values())


def get_item(key):
    return _ITEMS[key]
