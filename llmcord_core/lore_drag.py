"""Lore transfers use stable entry snapshots instead of moving NiceGUI slots."""
from nicegui.element import Element
from nicegui.elements.sortable import sortable
from pathlib import Path


class LoreDrag(Element, component='lore_drag.js', esm={'nicegui-sortable': str(Path(sortable.__file__).parent / 'dist')}):
    def __init__(self, root_id, on_move):
        super().__init__()
        self._props.update(rootId=root_id, listIds=[])
        self.on('move', on_move)

    def lists(self, containers):
        self._props['listIds'] = [container.html_id for container in containers]
        self.update()

    def busy(self, value):
        self.run_method('setBusy', value)
