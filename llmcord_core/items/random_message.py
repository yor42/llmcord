"""The random_message kind: pick a row from each table with a seeded generator and fill a template.

A table is locale-keyed ({"en": [...]}) holding plain strings or records (dicts of named strings), or a numeric range
({"range": [lo, hi]}, where hi may be the parameter name "sides"). A template ({"en": {"text", "fields", "footer_note"}})
uses {name} placeholders only: a table's name, a record's field names, "orientation"/"meaning" for two-sided records
(upright/reversed fields, flipped by the reversed_chance param), and the parameter names. Formatting is done here,
never with str.format.

A stored use result is the record of what happened; replay it from the stored result, not by re-rolling, because editing
the admin entries changes row counts and so what a seed picks.
"""
import random
import string
from dataclasses import dataclass, replace

from ..i18n import codes
from .recipe import (
    DEFAULT_LOCALE, MAX_ENTRIES, MAX_TEXT, Recipe, RecipeError, check_entries, check_int, check_locale_keyed, check_text, lookup,
)

PARAM_KEYS = ('sides', 'reversed_chance')
MAX_FIELDS = 8
MAX_RANGE = 1000
_IDENT = set(string.ascii_lowercase + '_')


@dataclass(frozen=True)
class Result:
    text: str
    fields: tuple = ()  # ((name, value), ...)
    footer_note: str = ''

    def to_json(self):
        return {'text': self.text, 'fields': [[n, v] for n, v in self.fields], 'footer_note': self.footer_note}


def _ident(name, where):
    if not isinstance(name, str) or not 1 <= len(name) <= 16 or not set(name) <= _IDENT:
        raise RecipeError(codes.RECIPE_NAME_FORM, where=where)
    return name


def _is_range(table):
    return isinstance(table, dict) and 'range' in table


def _is_records(rows):
    return bool(rows) and isinstance(rows[0], dict)


def _check_rows(rows, where):
    check_entries(rows, where)
    if _is_records(rows):
        names = None
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or not 1 <= len(row) <= MAX_FIELDS:
                raise RecipeError(codes.RECIPE_RECORD_FIELDS, where=where, index=i, limit=MAX_FIELDS)
            if names is None:
                names = set(row)
            elif set(row) != names:
                raise RecipeError(codes.RECIPE_RECORD_DIFFER, where=where, index=i)
            for k, v in row.items():
                check_text(v, f'{where}[{i}].{_ident(k, where)}')
    else:
        for i, row in enumerate(rows):
            check_text(row, f'{where}[{i}]')


def _validate_params(params):
    unknown = set(params) - set(PARAM_KEYS)
    if unknown:
        raise RecipeError(codes.RECIPE_UNKNOWN_PARAMS, params=', '.join(sorted(map(str, unknown))))
    if 'sides' in params:
        check_int(params['sides'], 'params.sides', 2, MAX_RANGE)
    if 'reversed_chance' in params:
        v = params['reversed_chance']
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
            raise RecipeError(codes.RECIPE_REVERSED_CHANCE)


def _range_bounds(table, params, where):
    spec = table['range']
    if set(table) != {'range'} or not isinstance(spec, list) or len(spec) != 2:
        raise RecipeError(codes.RECIPE_RANGE_FORM, where=where)
    lo, hi = spec
    check_int(lo, f'{where}.range[0]', 0, MAX_RANGE)
    if hi == 'sides':
        if 'sides' not in params:
            raise RecipeError(codes.RECIPE_RANGE_SIDES, where=where)
        hi = params['sides']
    check_int(hi, f'{where}.range[1]', max(lo, 1), MAX_RANGE)
    return lo, hi


def placeholders(recipe: Recipe):
    names = set(recipe.params)
    for name, table in recipe.tables.items():
        rows = None if _is_range(table) else table[DEFAULT_LOCALE]
        two_sided = bool(rows) and _is_records(rows) and {'upright', 'reversed'} <= set(rows[0])
        fields = set(rows[0]) | ({'orientation', 'meaning'} if two_sided else set()) if rows and _is_records(rows) else {name}
        if fields & names:
            raise RecipeError(codes.RECIPE_PLACEHOLDER_CLASH, name=name)
        names |= fields
    return names


def _parse_template(text, allowed, where):
    parts = []
    try:
        for literal, name, spec, conv in string.Formatter().parse(text):
            if name is not None and (name not in allowed or spec or conv):
                raise RecipeError(codes.RECIPE_PLACEHOLDER_DENIED, where=where, name=name)
            parts.append((literal, name))
    except ValueError as exc:
        raise RecipeError(codes.RECIPE_TEMPLATE_SYNTAX, where=where, detail=str(exc)) from None
    return parts


def _fill(text, values):
    return ''.join(lit + (values[name] if name is not None else '') for lit, name in _parse_template(text, values, 'template'))


def _check_template(raw, allowed):
    if isinstance(raw, str):
        raw = {'text': raw}
    if not isinstance(raw, dict) or not set(raw) <= {'text', 'fields', 'footer_note'} or 'text' not in raw:
        raise RecipeError(codes.RECIPE_TEMPLATE_SHAPE)
    _parse_template(check_text(raw['text'], 'template.text', 1000), allowed, 'template.text')
    if raw.get('footer_note'):
        _parse_template(check_text(raw['footer_note'], 'template.footer_note'), allowed, 'template.footer_note')
    pairs = raw.get('fields', [])
    if not isinstance(pairs, list) or len(pairs) > 6:
        raise RecipeError(codes.RECIPE_TEMPLATE_FIELDS)
    for i, pair in enumerate(pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            raise RecipeError(codes.RECIPE_TEMPLATE_PAIR, index=i)
        check_text(pair[0], f'template.fields[{i}][0]', 64)
        _parse_template(check_text(pair[1], f'template.fields[{i}][1]'), allowed, f'template.fields[{i}][1]')


def validate(recipe: Recipe):
    _validate_params(recipe.params)
    if not recipe.tables:
        raise RecipeError(codes.RECIPE_TABLES_REQUIRED)
    for name, table in recipe.tables.items():
        where = f'tables.{_ident(name, "tables")}'
        if _is_range(table):
            _range_bounds(table, recipe.params, where)
            continue
        check_locale_keyed(table, where)
        shapes = set()
        for loc, rows in table.items():
            _check_rows(rows, f'{where}.{loc}')
            shapes.add(tuple(sorted(rows[0])) if _is_records(rows) else None)
        if len(shapes) != 1:
            raise RecipeError(codes.RECIPE_LOCALES_MIX, where=where)
    allowed = placeholders(recipe)
    check_locale_keyed(recipe.template, 'template')
    for raw in recipe.template.values():
        _check_template(raw, allowed)


def _entries(extra, where, allow_empty=True):
    if not extra:
        return {}
    check_locale_keyed(extra, where)
    for loc, rows in extra.items():
        if not isinstance(rows, list) or len(rows) > MAX_ENTRIES or not (rows or allow_empty):
            raise RecipeError(codes.RECIPE_ENTRIES_UP_TO if allow_empty else codes.RECIPE_ENTRIES_ONE_TO, where=where, loc=loc, limit=MAX_ENTRIES)
        for i, row in enumerate(rows):
            check_text(row, f'{where}.{loc}[{i}]', MAX_TEXT)
    return extra


def _plain_rows(rows, locale, extra, replace):
    base = lookup(replace, locale) if replace else None
    out = list(base or rows) + list((lookup(extra, locale) or []) if extra else [])
    if len(out) > MAX_ENTRIES:
        raise RecipeError(codes.RECIPE_TOO_MANY_ENTRIES, limit=MAX_ENTRIES)
    return out


def _first_plain(recipe):
    return next((n for n, t in recipe.tables.items() if not _is_range(t) and not _is_records(t[DEFAULT_LOCALE])), None)


def validate_entries(recipe: Recipe, extra_entries=None, replace_entries=None):
    """Raise RecipeError for anything roll() would reject in the admin entries (shape, length, empty replace list, 500 cap).

    Extra and replace entries apply to the first plain-string table; a recipe with none (tarot, dice) ignores them.
    """
    extra = _entries(extra_entries, 'extra_entries')
    swap = _entries(replace_entries, 'replace_entries', allow_empty=False)
    name = _first_plain(recipe)
    if name is not None:
        for loc in {DEFAULT_LOCALE, *extra, *swap}:
            _plain_rows(lookup(recipe.tables[name], loc), loc, extra, swap)
    return extra, swap


def roll(recipe: Recipe, seed, locale=DEFAULT_LOCALE, extra_entries=None, replace_entries=None, params=None) -> Result:
    """Same recipe, seed, locale and entries give the same Result. `params` overrides recipe params (validated)."""
    if params:
        recipe = replace(recipe, params={**recipe.params, **params})
        validate(recipe)
    extra, swap = validate_entries(recipe, extra_entries, replace_entries)
    first_plain = _first_plain(recipe)
    rng = random.Random(seed)
    values = {k: str(v) for k, v in recipe.params.items()}
    for name, table in recipe.tables.items():
        if _is_range(table):
            lo, hi = _range_bounds(table, recipe.params, name)
            values[name] = str(rng.randint(lo, hi))
            continue
        rows = lookup(table, locale)
        if name == first_plain:
            rows = _plain_rows(rows, locale, extra, swap)
        row = rows[rng.randrange(len(rows))]
        if isinstance(row, dict):
            values.update(row)
            if {'upright', 'reversed'} <= set(row):
                flipped = rng.random() < recipe.params.get('reversed_chance', 0)
                values['orientation'] = 'Reversed' if flipped else 'Upright'
                values['meaning'] = row['reversed' if flipped else 'upright']
        else:
            values[name] = row
    raw = lookup(recipe.template, locale)
    if isinstance(raw, str):
        raw = {'text': raw}
    return Result(
        text=_fill(raw['text'], values),
        fields=tuple((n, _fill(v, values)) for n, v in raw.get('fields', [])),
        footer_note=_fill(raw['footer_note'], values) if raw.get('footer_note') else '',
    )
