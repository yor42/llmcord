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
SMALL_BUTTON_SIZE = '1.3em'  # icons in size=xs/sm buttons


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
             '.ll-icon-solo { order: 0; margin-right: 0; }',
             # Quasar puts size=xs/sm on the button as an inline font-size (8px/10px); the default icon is sized for
             # normal buttons and looks large there, so shrink only icons without an explicit size.
             '.q-btn[style*="font-size: 8px"] .ll-icon:not([style]), .q-btn[style*="font-size: 10px"] .ll-icon:not([style])'
             ' { width: %s; height: %s; margin-right: 6px; }' % (SMALL_BUTTON_SIZE, SMALL_BUTTON_SIZE),
             # Quasar q-item menu entries match the account menu's ll-menu-item padding.
             '.ll-menu .q-item { padding: 10px 16px; min-height: 0; }',
             # Windows high contrast: a mask filled with currentColor can be forced to a background colour; keep it text-coloured.
             '@media (forced-colors: active) { .ll-icon { background-color: CanvasText; forced-color-adjust: none; } }']
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


def more_menu(name: str):
    """Title-row ⋯ menu (D21): an icon-only flat round button and its ``ll-menu``; use as ``with more_menu(name):`` and add items.

    The button is named ``More actions for <name>`` and does not propagate its click, so it can sit in an expansion header.
    """
    from nicegui import ui
    button = ui.button().props('flat round dense text-color=white').classes('ll-more')
    button.props['aria-label'] = f'More actions for {name}'  # not string-parsed: names are user data
    stop = '(event) => event.stopPropagation()'
    button.on('click', js_handler=stop).on('keyup', js_handler=stop)
    with button:
        lucide('ellipsis', '1.4em').classes('ll-icon-solo')
        tooltip = ui.tooltip('More actions')
        menu = ui.menu().classes('ll-menu')
    menu.on('before-show', js_handler=f'() => getElement({tooltip.id}).hide()')  # the tooltip would stay over the open menu
    return menu
