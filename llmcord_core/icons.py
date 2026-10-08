"""Lucide icons for the dashboard (UI-30): vendored SVGs in ``icons/lucide`` drawn as CSS masks.

Each icon is a ``<span class="ll-icon ll-i-NAME" aria-hidden="true">`` filled with ``currentColor`` and
clipped by a ``data:`` SVG mask, so it follows the button text colour and needs no network, font or script.
``icon_css()`` is added once by ``dashboard.apply_theme``. Add an icon by vendoring the SVG (see docs/engineering/ui-guide.md).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

ICON_DIR = Path(__file__).with_name('icons') / 'lucide'
BUTTON_SIZE = '1.715em'  # same as Quasar's button icons


@lru_cache(maxsize=None)
def available() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in ICON_DIR.glob('*.svg')))


@lru_cache(maxsize=None)
def _data_url(name: str) -> str:
    if not re.fullmatch(r'[a-z0-9-]+', name or '') or name not in available():
        raise KeyError(f'Unknown icon {name!r}; vendored: {", ".join(available())}')
    svg = re.sub(r'<!--.*?-->', '', (ICON_DIR / f'{name}.svg').read_text(encoding='utf-8'), flags=re.S).strip()
    svg = re.sub(r'\s+', ' ', svg)
    return 'data:image/svg+xml,' + quote(svg, safe="/:=' ")


@lru_cache(maxsize=None)
def icon_css() -> str:
    rules = ['.ll-icon { display: inline-block; flex: none; width: %s; height: %s; background-color: currentColor; order: -1;'
             ' margin-right: 8px; vertical-align: middle; -webkit-mask: var(--ll-mask) center / contain no-repeat; mask: var(--ll-mask) center / contain no-repeat; }'
             % (BUTTON_SIZE, BUTTON_SIZE),
             '.ll-icon-solo { order: 0; margin-right: 0; }']
    rules += [f'.ll-i-{name} {{ --ll-mask: url("{_data_url(name)}"); }}' for name in available()]
    return '\n'.join(rules)


def lucide(name: str, size: str | None = None):
    """Create an aria-hidden icon in the current NiceGUI context (put it inside a button, or use it alone as a handle).

    Raises KeyError for a name that is not vendored. ``size`` is a CSS length; the default matches Quasar button icons.
    """
    from nicegui import ui
    _data_url(name)  # validate
    element = ui.element('span').classes(f'll-icon ll-i-{name}').props('aria-hidden="true"')
    if size:
        element.style(f'width: {size}; height: {size}')
    return element


def lucide_button(text: str, name: str, **kwargs):
    """``ui.button(text, **kwargs)`` with a leading Lucide icon (inherits the button colour)."""
    from nicegui import ui
    button = ui.button(text, **kwargs)
    with button:
        lucide(name)
    return button
