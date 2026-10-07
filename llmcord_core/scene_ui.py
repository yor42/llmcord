"""World/channel guidelines, owner deletion, and additive lore imports."""
from __future__ import annotations

from nicegui import ui

from .lorebooks import parse_lorebook


def guideline_editor(ctx, kind, owner_id, label):
    current = ctx.store.guidelines(ctx.guild_id, kind, owner_id)
    control = ui.textarea(label, value=current['content']).classes('w-full')
    control.props('placeholder="Setting, participants\' fictional roles, tone, and interaction conventions"')
    ctx.button('Save ' + label.lower(),
               lambda: ctx.store.save_guidelines(ctx.guild_id, kind, owner_id, control.value or '', current['revision']),
               'guidelines.edit', {'kind': kind, 'id': owner_id}, then=lambda _: ui.navigate.reload())


async def delete_space_dialog(ctx, space):
    impact = await ctx.run(lambda: ctx.store.space_delete_impact(ctx.guild_id, space['id']))
    if impact is None:
        return
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-lg'):
        ui.label(f"Delete {space['kind']} {space['name']}?").classes('text-xl font-bold')
        ui.label(f"This removes its guidelines, {impact['entries']} owned lore entries, encounters, and hub/lorebook links. Lorebooks and past messages remain.")
        if impact['characters']:
            ui.label('Move or delete these home characters first: ' + ', '.join(impact['characters']))
        if impact['channels']:
            ui.label(f"Rebind its {len(impact['channels'])} Discord channel(s) to another world or hub first.")
        with ui.row():
            ui.button('Cancel', on_click=dialog.close)
            ctx.button('Delete ' + space['kind'],
                       lambda: ctx.store.delete_space(ctx.guild_id, space['id'], impact['revision']),
                       'space.delete', {'id': space['id']}, then=lambda _: ui.navigate.reload(), color='negative').set_enabled(not impact['characters'] and not impact['channels'])
    dialog.open()


async def delete_book_dialog(ctx, book):
    def preview():
        ctx.store.validate_owner(ctx.guild_id, 'book', book['id'])
        return len(ctx.store.admin_entries(ctx.guild_id, 'book', book['id'])), ctx.store.owner_revision(ctx.guild_id, 'book', book['id'])
    result = await ctx.run(preview)
    if result is None:
        return
    count, revision = result
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-lg'):
        ui.label(f"Delete lorebook {book['name']}?").classes('text-xl font-bold')
        ui.label(f'This deletes the book, its {count} remaining entries, and its world/hub links. Entries moved to another owner remain.')
        with ui.row():
            ui.button('Cancel', on_click=dialog.close)
            ctx.button('Delete lorebook', lambda: ctx.store.delete_lorebook(ctx.guild_id, book['id'], revision),
                       'book.delete', {'id': book['id']}, then=lambda _: ui.navigate.to(f'/guild/{ctx.guild_id}?tab=imports'), color='negative')
    dialog.open()


def direct_import_dialog(ctx, kind, owner_id, on_saved=None):
    owner = next((owner for owner in ctx.service.owners(ctx.guild_id) if (owner['kind'], owner['id']) == (kind, owner_id)), None)
    if owner is None:
        raise ValueError('Choose an owner in this server')
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-4xl'):
        ui.label('Import entries into ' + owner['label']).classes('text-xl font-bold')
        ui.label('Adds entries directly to this owner. Existing lore stays in place; identical entries are skipped. No new book is created.')
        area = ui.column().classes('w-full')
        async def uploaded(event):
            imported = parse_lorebook(await event.file.read())
            ctx.store.validate_owner(ctx.guild_id, kind, owner_id)
            revision = ctx.store.owner_revision(ctx.guild_id, kind, owner_id)
            changes = ctx.store.preview_entry_import(ctx.guild_id, kind, owner_id, imported)
            area.clear()
            with area:
                additions = sum(change['status'] == 'add' for change in changes)
                format_name = 'RisuAI' if imported.source_format == 'risu' else 'SillyTavern'
                ui.label(f'{format_name}: {additions} new entries, {len(changes) - additions} already present').classes('font-bold')
                ui.table(columns=[{'name': key, 'label': label, 'field': key, 'align': 'left'} for key, label in
                                  (('uid', 'Source ID'), ('status', 'Action'), ('content', 'Content'), ('warnings', 'Compatibility notes'))],
                         rows=[{'uid': change['uid'], 'status': change['status'], 'content': change['after'], 'warnings': '; '.join(change['warnings'])} for change in changes],
                         row_key='uid', pagination=10).classes('w-full')
                def saved(refs):
                    dialog.close()
                    ui.notify(f'Imported {len(refs)} entries', type='positive')
                    if on_saved:
                        on_saved(refs)
                    else:
                        ui.navigate.to(f'/guild/{ctx.guild_id}?tab=lore&owner={kind}:{owner_id}')
                ctx.button('Apply entry import', lambda: ctx.store.import_lore_entries(ctx.guild_id, kind, owner_id, imported, revision),
                           'lore.import', {'kind': kind, 'id': owner_id}, then=saved)
        ctx.upload(uploaded, 'Upload JSON entries')
        ui.button('Cancel', on_click=dialog.close)
    dialog.open()
