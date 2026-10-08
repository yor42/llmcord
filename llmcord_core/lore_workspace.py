"""Lore lists, stable drag commands, and atomic selection actions."""
from __future__ import annotations

import asyncio

from nicegui import ui

from .lore_drag import LoreDrag
from .lorebooks import normalize_entry


def snapshot(entry):
    return {key: entry[key] for key in ('entry_key', 'revision', 'owner_kind', 'owner_id')}


def render_lore_workspace(ctx, entry_editor, on_import=None, on_entry_import=None):
    owners = ctx.snapshot.owners
    options = {f"{owner['kind']}:{owner['id']}": owner['label'] for owner in owners}
    ui.label('Lore workspace').classes('text-xl font-bold')
    if on_import:
        ui.button('Import lorebook', icon='upload_file', on_click=on_import)
        ui.label('Import JSON directly into either owner below, or manage reusable named lorebooks in Imports.').classes('text-slate-400')
    ui.label('Drag entries onto either navy drop area, or select entries to move or delete together. Higher insertion orders appear later in each prompt position.')
    if not owners:
        ui.label('Create a world or import a character first.')
        return
    _LoreWorkspace(ctx, options, entry_editor, on_entry_import).build()


class _LoreWorkspace:
    def __init__(self, ctx, options, entry_editor, on_entry_import):
        self.ctx, self.options, self.entry_editor, self.on_entry_import = ctx, options, entry_editor, on_entry_import
        self.selected, self.pages, self.checkboxes, self.bulk_buttons = {}, {}, {}, []
        self.operation_lock = asyncio.Lock()

        @ui.refreshable
        def render_board():
            self._render_board()
        self.render_board = render_board

    def build(self):
        ctx, options = self.ctx, self.options
        self.root = ui.column().classes('w-full lore-workspace')
        with self.root:
            with ui.row().classes('w-full items-end'):
                initial_owner = getattr(ctx, 'lore_owner', None)
                self.left = ui.select(options, value=initial_owner if initial_owner in options else next(iter(options)), label='Left owner', with_input=True).classes('flex-1')
                self.right = ui.select(options, value=list(options)[-1], label='Right owner', with_input=True).classes('flex-1')
                self.query = ui.input('Search content and keywords').props('debounce=300').classes('flex-1')
            toolbar = ui.row().classes('w-full items-center')
            self.editor = ui.column().classes('w-full')
            board = ui.row().classes('w-full items-stretch flex-nowrap overflow-auto')
        self._build_toolbar(toolbar)
        self.drag = LoreDrag(self.root.html_id, self.moved)
        with board:
            self.render_board()
        for control in (self.left, self.right, self.query):
            control.on_value_change(lambda _: self.refresh())

    async def refresh(self):
        async with self.operation_lock:
            if await self.ctx.run(lambda: True):
                self.render_board.refresh()

    def _build_toolbar(self, toolbar):
        selected, bulk_buttons = self.selected, self.bulk_buttons
        with toolbar:
            self.counter = ui.label('0 selected')
            bulk_buttons.append(ui.button('Move selected left', icon='arrow_back',
                                          on_click=lambda: self.move(list(selected.values()), self.left)))
            bulk_buttons.append(ui.button('Move selected right', icon='arrow_forward',
                                          on_click=lambda: self.move(list(selected.values()), self.right)))
            bulk_buttons.append(ui.button('Delete selected', icon='delete', color='negative',
                                          on_click=lambda: self.confirm_delete(list(selected.values()))))
            ui.button('Clear selection', on_click=lambda: self.select(list(selected.values()), False))
            self.update_selection()

    async def mutate(self, entries, action, target=None):
        ctx, left, right, query, pages, selected, drag = self.ctx, self.left, self.right, self.query, self.pages, self.selected, self.drag
        async with self.operation_lock:
            drag.busy(True)
            try:
                def operation():
                    if action == 'lore.delete':
                        return ctx.store.delete_entries(ctx.guild_id, entries)
                    return ctx.store.transfer_entries(ctx.guild_id, entries, *target)
                result = await ctx.run(operation, action, {'keys': [entry['entry_key'] for entry in entries], 'destination': target})
                if result is not None:
                    if target:
                        destination = ctx.store.admin_entries(ctx.guild_id, *target)
                        if query.value:
                            destination = [entry for entry in destination if query.value.casefold() in (entry['content'] + ' ' + ' '.join(entry['rule']['keys'])).casefold()]
                        moved_keys = {entry['entry_key'] for entry in entries}
                        index = next((index for index, entry in enumerate(destination) if entry['entry_key'] in moved_keys), 0)
                        for side, control in (('left', left), ('right', right)):
                            if control.value == f'{target[0]}:{target[1]}':
                                pages[(side, control.value)] = index // 50 + 1
                    ui.notify(f"{'Deleted' if action == 'lore.delete' else 'Moved'} {len(entries)} lore {'entry' if len(entries) == 1 else 'entries'}", type='positive')
                return result
            finally:
                # Reload authoritative lists even when validation rejects a move.
                selected.clear()
                self.update_selection()
                self.render_board.refresh()
                drag.busy(False)

    async def move(self, entries, control):
        kind, ident = control.value.split(':', 1)
        target = (kind, int(ident))
        entries = [entry for entry in entries if (entry['owner_kind'], entry['owner_id']) != target]
        if not entries:
            ui.notify('Selected entries are already in that list.')
            return None
        return await self.mutate(entries, 'lore.move', target)

    def confirm_delete(self, entries):
        ctx = self.ctx
        if not entries:
            return
        count = len(entries)
        with ui.dialog() as dialog, ui.card():
            ui.label(f"Delete {count} lore {'entry' if count == 1 else 'entries'}?").classes('text-xl')
            for entry in entries[:5]:
                row = ctx.store.entry_by_key(ctx.guild_id, entry['entry_key'])
                if row:
                    owner = self.options.get(f"{row['owner_kind']}:{row['owner_id']}", row['owner_kind'])
                    ui.label(f"{owner}: {row['content'][:100] or '(empty entry)'}")
            if count > 5:
                ui.label(f'And {count - 5} more selected entries.')
            ui.label('This also keeps your deletion choice for future reimports.')
            with ui.row():
                ui.button('Cancel', on_click=dialog.close)
                async def apply():
                    dialog.close()
                    await self.mutate(entries, 'lore.delete')
                ui.button(f"Delete {count} {'entry' if count == 1 else 'entries'}", on_click=apply, color='negative')
        dialog.open()

    def update_selection(self):
        selected = self.selected
        self.counter.set_text(f'{len(selected)} selected')
        for button in self.bulk_buttons:
            button.set_enabled(bool(selected))
        for key, controls in self.checkboxes.items():
            for checkbox in controls:
                if checkbox.value != (key in selected):
                    checkbox.set_value(key in selected)

    def select(self, entries, value):
        for entry in entries:
            if value:
                self.selected[entry['entry_key']] = snapshot(entry)
            else:
                self.selected.pop(entry['entry_key'], None)
        self.update_selection()

    async def moved(self, event):
        try:
            args = event.args
            entry = {'entry_key': args['entry_key'], 'revision': int(args['revision']),
                     'owner_kind': args['owner_kind'], 'owner_id': int(args['owner_id'])}
            await self.mutate([entry], 'lore.move', (args['target_kind'], int(args['target_id'])))
        except (ValueError, KeyError, TypeError):
            ui.notify('Move could not be confirmed. Refresh and try again.', type='negative')
            self.render_board.refresh()
            self.drag.busy(False)

    def _render_board(self):
        self.checkboxes.clear()
        containers = []
        for side, control, opposite in (('left', self.left, self.right), ('right', self.right, self.left)):
            self._render_panel(side, control, opposite, containers)
        self.drag.lists(containers)

    def _render_panel(self, side, control, opposite, containers):
        ctx, pages, query, options = self.ctx, self.pages, self.query, self.options
        render_board = self.render_board
        kind, ident = control.value.split(':', 1)
        ident = int(ident)
        with ui.column().classes(f'flex-1 min-w-80 lore-panel lore-panel-{side}'):
            ui.label(options[control.value]).classes('text-lg font-bold owner-heading')
            page_key = (side, control.value)
            page_number = max(1, pages.get(page_key, 1))
            visible, total = ctx.store.admin_entries_page(ctx.guild_id, kind, ident, query.value, 50, (page_number - 1) * 50)
            last_page = max(1, (total + 49) // 50)
            if page_number > last_page:
                page_number = last_page
                visible, total = ctx.store.admin_entries_page(ctx.guild_id, kind, ident, query.value, 50, (page_number - 1) * 50)
            ui.label(f'{total} entries · page {page_number}').classes('owner-heading text-slate-400')
            if total > 50:
                pagination = ui.select(list(range(1, last_page + 1)), value=page_number, label='Page').classes('owner-heading')
                async def page_changed(event, page_key=page_key):
                    async with self.operation_lock:
                        if await ctx.run(lambda: True):
                            pages[page_key] = event.value
                            render_board.refresh()
                pagination.on_value_change(page_changed)
            with ui.row().classes('owner-heading'):
                ui.button('Select page', on_click=lambda visible=visible: self.select(visible, True))
                ui.button('Clear page', on_click=lambda visible=visible: self.select(visible, False))
            # Only cards are children of the Sortable container. Headings,
            # pagination and action controls stay outside its drop area.
            with ui.column().classes(f'w-full min-h-48 bg-slate-900 rounded-lg p-4 lore-drop-zone lore-drop-{side}') as container:
                container._props.update({'data-owner-kind': kind, 'data-owner-id': str(ident)})
                containers.append(container)
                for entry in visible:
                    self._render_entry(entry, side, control, opposite)
            self._render_panel_actions(kind, ident)

    def _render_entry(self, entry, side, control, opposite):
        ctx = self.ctx
        with ui.card().classes('w-full lore-entry') as tile:
            tile._props.update({'data-entry-key': entry['entry_key'], 'data-revision': str(entry['revision'])})
            with ui.row().classes('items-center'):
                checkbox = ui.checkbox('Select entry', value=entry['entry_key'] in self.selected,
                    on_change=lambda event, entry=entry: self.select([entry], event.value)).props('dense')
                self.checkboxes.setdefault(entry['entry_key'], []).append(checkbox)
                ui.icon('drag_indicator').classes('drag-handle cursor-grab')
                ui.label(' · '.join(entry['rule']['keys']) or 'No keywords').classes('font-bold')
            ui.label(entry['content'][:220])
            ui.label(f"Priority {entry['rule']['order']} · {'Enabled' if entry['rule']['enabled'] else 'Disabled'}")
            async def edit(entry=entry):
                fresh = await ctx.run(lambda: ctx.store.entry_by_key(ctx.guild_id, entry['entry_key']))
                if fresh:
                    self.editor.clear()
                    with self.editor:
                        self.entry_editor(ctx, fresh, self.options, self.render_board.refresh)
            with ui.row().classes('items-center flex-wrap gap-2'):
                ui.button('Edit / transfer', on_click=edit)
                ui.button('Move right' if side == 'left' else 'Move left',
                    icon='arrow_forward' if side == 'left' else 'arrow_back',
                    on_click=lambda entry=entry, opposite=opposite: self.move([snapshot(entry)], opposite)).set_enabled(control.value != opposite.value)
                ui.button('Delete', icon='delete', color='negative',
                          on_click=lambda entry=entry: self.confirm_delete([snapshot(entry)]))

    def _render_panel_actions(self, kind, ident):
        ctx, render_board = self.ctx, self.render_board
        async def add(kind=kind, ident=ident):
            if await ctx.run(lambda: True):
                self.editor.clear()
                with self.editor:
                    self.entry_editor(ctx, {'owner_kind': kind, 'owner_id': ident, 'content': '', 'rule': normalize_entry('new', {}).rule, 'pinned': False}, self.options, render_board.refresh)
        with ui.row().classes('owner-heading'):
            ui.button('New entry', on_click=add)
            def download(kind=kind, ident=ident):
                import json
                ui.download.content(json.dumps(ctx.store.export_lore(ctx.guild_id, kind, ident), ensure_ascii=False, indent=2), 'lorebook.json')
                return True
            ctx.button('Export owner', download)
            if self.on_entry_import:
                def imported(refs):
                    self.selected.clear()
                    self.update_selection()
                    render_board.refresh()
                ui.button('Import JSON entries', icon='upload_file', on_click=lambda kind=kind, ident=ident: self.on_entry_import(kind, ident, imported))
