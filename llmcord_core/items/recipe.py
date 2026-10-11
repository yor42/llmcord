"""An item's definition as data. A recipe never carries code; its kind (see registry.py) validates the kind-specific parts."""
import json
import re
from dataclasses import dataclass, field

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


class RecipeError(ValueError):
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

    @property
    def reusable(self):
        return not self.consumable

    @property
    def hold_cap(self):
        """1 for a reusable item; None for a consumable (a server setting decides)."""
        return None if self.consumable else 1


def lookup(by_locale, locale):
    """The value stored for `locale`, else the 'en' one."""
    return by_locale.get(locale, by_locale.get(DEFAULT_LOCALE))


def check_text(value, where, limit=MAX_TEXT):
    if not isinstance(value, str) or not value.strip():
        raise RecipeError(f'{where}: expected non-empty text')
    if len(value) > limit:
        raise RecipeError(f'{where}: longer than {limit} characters')
    return value


def check_int(value, where, lo, hi):
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise RecipeError(f'{where}: expected a whole number from {lo} to {hi}')
    return value


def check_locale_keyed(value, where):
    """{"en": X, ...} with an 'en' entry; returns it unchanged."""
    if not isinstance(value, dict) or DEFAULT_LOCALE not in value:
        raise RecipeError(f'{where}: expected a locale-keyed object with an "{DEFAULT_LOCALE}" entry')
    for loc in value:
        if not isinstance(loc, str) or not LOCALE_RE.match(loc):
            raise RecipeError(f'{where}: bad locale key {loc!r}')
    return value


def check_entries(value, where):
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_ENTRIES:
        raise RecipeError(f'{where}: expected a list of 1 to {MAX_ENTRIES} entries')
    return value


def parse_common(data):
    """Kind-independent checks. Returns a Recipe; the registry then applies the kind's own validation."""
    if not isinstance(data, dict):
        raise RecipeError('recipe must be an object')
    try:
        size = len(json.dumps(data, ensure_ascii=False).encode('utf-8'))
    except (TypeError, ValueError, RecursionError) as exc:
        raise RecipeError(f'recipe is not plain data: {exc}') from None
    if size > MAX_BYTES:
        raise RecipeError(f'recipe is larger than {MAX_BYTES} bytes')
    unknown = set(data) - _FIELDS
    if unknown:
        raise RecipeError(f'unknown recipe field(s): {", ".join(sorted(map(str, unknown)))}')
    missing = {'key', 'kind', 'name', 'consumable', 'default_price'} - set(data)
    if missing:
        raise RecipeError(f'missing recipe field(s): {", ".join(sorted(missing))}')
    key = data['key']
    if not isinstance(key, str) or not KEY_RE.match(key):
        raise RecipeError('key: use 1-32 characters from a-z, 0-9 and _')
    kind = data['kind']
    if not isinstance(kind, str):
        raise RecipeError('kind: expected text')
    consumable = data['consumable']
    if not isinstance(consumable, bool):
        raise RecipeError('consumable: expected true or false')
    cooldown = check_int(data.get('default_cooldown_seconds', 0), 'default_cooldown_seconds', 0, MAX_COOLDOWN)
    if consumable:
        cooldown = 0
    emoji = data.get('emoji', '')
    if not isinstance(emoji, str) or len(emoji) > MAX_EMOJI:
        raise RecipeError(f'emoji: expected text up to {MAX_EMOJI} characters')
    audience = data.get('audience', 'none')
    if audience not in AUDIENCES:
        raise RecipeError(f'audience: must be one of {", ".join(AUDIENCES)}')
    for name in ('params', 'tables', 'template'):
        if not isinstance(data.get(name, {}), dict):
            raise RecipeError(f'{name}: expected an object')
    return Recipe(key=key, kind=kind, name=check_text(data['name'], 'name', MAX_NAME), emoji=emoji, consumable=consumable,
                  default_price=check_int(data['default_price'], 'default_price', 0, MAX_PRICE), default_cooldown_seconds=cooldown,
                  params=dict(data.get('params', {})), tables=dict(data.get('tables', {})), template=dict(data.get('template', {})),
                  audience=audience)
