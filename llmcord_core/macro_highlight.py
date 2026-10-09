"""One page-level helper that highlights {{macros}} in opted-in textareas (class ``ll-macro``)."""
from nicegui.element import Element

from .prompts import MACROS


class MacroHighlight(Element, component='macro_highlight.js'):
    def __init__(self):
        super().__init__()
        self._props['known'] = sorted(MACROS)
