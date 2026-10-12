"""An item's definition as data. A recipe never carries code; its kind (see registry.py) validates the kind-specific parts."""
import json
import re
from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType

from ..i18n import UserError, codes
from ..i18n.negotiate import candidates

KEY_RE = re.compile(r'^[a-z0-9_]{1,32}$')
LOCALE_RE = re.compile(r'^[a-z]{2,3}(-[A-Za-z0-9]{2,8})?$')
MAX_NAME = 40
MAX_EMOJI = 16
MAX_ENTRIES = 500
MAX_TEXT = 300
MAX_BYTES = 64_000  # of the UTF-8 JSON encoding
MAX_PRICE = 1_000_000
MAX_COOLDOWN = 7 * 86400
DEFAULT_LOCALE = 'en'
AUDIENCES = ('none', 'member', 'any', 'npc')
_FIELDS = {'key', 'kind', 'name', 'emoji', 'consumable', 'default_price', 'default_cooldown_seconds', 'params', 'tables', 'template', 'audience'}


class RecipeError(UserError):
    """A recipe (or admin-added entries) failed validation; the message says what and where."""


@dataclass(frozen=True)
class Recipe:
    key: str
    kind: str
    name: str
    emoji: str
    consumable: bool
    default_price: int
    default_cooldown_seconds: int
    params: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    template: dict = field(default_factory=dict)
    audience: str = 'none'
    names: Mapping = None  # locale -> name; {'en': name} when the recipe gave a plain string

    def __post_init__(self):
        object.__setattr__(self, 'names', MappingProxyType(dict(self.names) if self.names else {DEFAULT_LOCALE: self.name}))

    @property
    def reusable(self):
        return not self.consumable

    @property
    def hold_cap(self):
        """1 for a reusable item; None for a consumable (a server setting decides)."""
        return None if self.consumable else 1


def lookup(by_locale, locale):
    """The value stored for `locale` (tag normalized, then its base language), else the 'en' one."""
    for tag in candidates(locale):
        for key in (tag, tag.replace('_', '-')):
            if key in by_locale:
                return by_locale[key]
    return by_locale.get(DEFAULT_LOCALE)


def display_name(recipe, locale):
    """The recipe's name for `locale`; the English name when there is none."""
    return lookup(recipe.names, locale)


def check_text(value, where, limit=MAX_TEXT):
    if not isinstance(value, str) or not value.strip():
        raise RecipeError(codes.RECIPE_TEXT_EMPTY, where=where)
    if len(value) > limit:
        raise RecipeError(codes.RECIPE_TEXT_LONG, where=where, limit=limit)
    return value


def check_int(value, where, lo, hi):
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise RecipeError(codes.RECIPE_INT_RANGE, where=where, lo=lo, hi=hi)
    return value


def check_locale_keyed(value, where):
    """{"en": X, ...} with an 'en' entry; returns it unchanged."""
    if not isinstance(value, dict) or DEFAULT_LOCALE not in value:
        raise RecipeError(codes.RECIPE_LOCALE_OBJECT, where=where, locale=DEFAULT_LOCALE)
    for loc in value:
        if not isinstance(loc, str) or not LOCALE_RE.match(loc):
            raise RecipeError(codes.RECIPE_LOCALE_KEY, where=where, key=repr(loc))
    return value


def check_entries(value, where):
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_ENTRIES:
        raise RecipeError(codes.RECIPE_ENTRY_COUNT, where=where, limit=MAX_ENTRIES)
    return value


def parse_common(data):
    """Kind-independent checks. Returns a Recipe; the registry then applies the kind's own validation."""
    if not isinstance(data, dict):
        raise RecipeError(codes.RECIPE_NOT_OBJECT)
    try:
        size = len(json.dumps(data, ensure_ascii=False).encode('utf-8'))
    except (TypeError, ValueError, RecursionError) as exc:
        raise RecipeError(codes.RECIPE_NOT_PLAIN, detail=str(exc)) from None
    if size > MAX_BYTES:
        raise RecipeError(codes.RECIPE_TOO_LARGE, limit=MAX_BYTES)
    unknown = set(data) - _FIELDS
    if unknown:
        raise RecipeError(codes.RECIPE_UNKNOWN_FIELDS, fields=', '.join(sorted(map(str, unknown))))
    missing = {'key', 'kind', 'name', 'consumable', 'default_price'} - set(data)
    if missing:
        raise RecipeError(codes.RECIPE_MISSING_FIELDS, fields=', '.join(sorted(missing)))
    key = data['key']
    if not isinstance(key, str) or not KEY_RE.match(key):
        raise RecipeError(codes.RECIPE_KEY_FORM)
    kind = data['kind']
    if not isinstance(kind, str):
        raise RecipeError(codes.RECIPE_KIND_TEXT)
    consumable = data['consumable']
    if not isinstance(consumable, bool):
        raise RecipeError(codes.RECIPE_CONSUMABLE_BOOL)
    cooldown = check_int(data.get('default_cooldown_seconds', 0), 'default_cooldown_seconds', 0, MAX_COOLDOWN)
    if consumable:
        cooldown = 0
    emoji = data.get('emoji', '')
    if not isinstance(emoji, str) or len(emoji) > MAX_EMOJI:
        raise RecipeError(codes.RECIPE_EMOJI, limit=MAX_EMOJI)
    audience = data.get('audience', 'none')
    if audience not in AUDIENCES:
        raise RecipeError(codes.RECIPE_AUDIENCE_ONE_OF, audiences=', '.join(AUDIENCES))
    for name in ('params', 'tables', 'template'):
        if not isinstance(data.get(name, {}), dict):
            raise RecipeError(codes.RECIPE_SECTION_OBJECT, name=name)
    raw_name = data['name']
    if isinstance(raw_name, dict):
        names = {loc: check_text(value, f'name.{loc}', MAX_NAME) for loc, value in check_locale_keyed(raw_name, 'name').items()}
    else:
        names = {DEFAULT_LOCALE: check_text(raw_name, 'name', MAX_NAME)}
    return Recipe(key=key, kind=kind, name=names[DEFAULT_LOCALE], names=names, emoji=emoji, consumable=consumable,
                  default_price=check_int(data['default_price'], 'default_price', 0, MAX_PRICE), default_cooldown_seconds=cooldown,
                  params=dict(data.get('params', {})), tables=dict(data.get('tables', {})), template=dict(data.get('template', {})),
                  audience=audience)
