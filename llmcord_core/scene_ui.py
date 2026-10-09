"""World/channel guidelines, owner deletion, and additive lore imports."""
from __future__ import annotations

import inspect

from nicegui import ui

from .lorebooks import parse_lorebook


def guideline_editor(ctx, kind, owner_id, label, button=True):
    """Guidelines textarea; with button=False no Save button is added and (control, current) is returned for a save-bar editor."""
    current = ctx.store.guidelines(ctx.guild_id, kind, owner_id)
    control = ui.textarea(label, value=current['content']).classes('w-full')
    control.props('placeholder="Setting, participants\' fictional roles, tone, and interaction conventions"')
    if not button:
        return control, current
    ctx.button('Save ' + label.lower(),
               lambda: ctx.store.save_guidelines(ctx.guild_id, kind, owner_id, control.value or '', current['revision']),
               'guidelines.edit', {'kind': kind, 'id': owner_id}, then=lambda _: ctx.refresh())


def confirm_dialog(ctx, title, lines, button_label, operation, action, detail=None, then=None, enabled=True):
    """One shape for destructive confirmations: bold title, explanation lines, Cancel and a red confirm button."""
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-lg'):
        ui.label(title).classes('text-xl font-bold')
        for line in lines:
            ui.label(line).classes('ll-muted')
        async def done(result):
            dialog.close()
            if then:
                followup = then(result)
                if inspect.isawaitable(followup):
                    await followup
        with ui.element('div').classes('ll-form-row justify-end'):
            ui.button('Cancel', on_click=dialog.close).props('flat')
            ctx.button(button_label, operation, action, detail, then=done, color='negative').set_enabled(enabled)
    dialog.open()


async def delete_space_dialog(ctx, space):
    impact = await ctx.run(lambda: ctx.store.space_delete_impact(ctx.guild_id, space['id']))
    if impact is None:
        return
    lines = [f"This removes its guidelines, {impact['entries']} owned lore entries, encounters, and hub/lorebook links. Lorebooks and past messages remain."]
    if impact['characters']:
        lines.append('Move or delete these home characters first: ' + ', '.join(impact['characters']))
    if impact['channels']:
        lines.append(f"Rebind its {len(impact['channels'])} Discord channel(s) to another world or hub first.")
    confirm_dialog(ctx, f"Delete {space['kind']} {space['name']}?", lines, 'Delete ' + space['kind'],
                   lambda: ctx.store.delete_space(ctx.guild_id, space['id'], impact['revision']),
                   'space.delete', {'id': space['id']}, then=lambda _: ctx.refresh(),
                   enabled=not impact['characters'] and not impact['channels'])


async def delete_book_dialog(ctx, book):
    def preview():
        ctx.store.validate_owner(ctx.guild_id, 'book', book['id'])
        return len(ctx.store.admin_entries(ctx.guild_id, 'book', book['id'])), ctx.store.owner_revision(ctx.guild_id, 'book', book['id'])
    result = await ctx.run(preview)
    if result is None:
        return
    count, revision = result
    confirm_dialog(ctx, f"Delete lorebook {book['name']}?",
                   [f'This deletes the lorebook, its {count} remaining entries, and its world/hub links. Entries moved to another owner remain.'],
                   'Delete lorebook', lambda: ctx.store.delete_lorebook(ctx.guild_id, book['id'], revision),
                   'book.delete', {'id': book['id']}, then=lambda _: ctx.refresh('imports'))


def direct_import_dialog(ctx, kind, owner_id, on_saved=None):
    owner = next((owner for owner in ctx.service.owners(ctx.guild_id, ctx.channel_names) if (owner['kind'], owner['id']) == (kind, owner_id)), None)
    if owner is None:
        raise ValueError('Choose an owner in this server')
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-4xl'):
        ui.label('Import entries into ' + owner['label']).classes('text-xl font-bold')
        ui.label('Adds entries directly to this owner. Existing lore stays in place; identical entries are skipped. No new lorebook is created.').classes('ll-muted')
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
                ui.label(f'{format_name}: {additions} new entries, {len(changes) - additions} already present').classes('ll-subtitle')
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
                        return ctx.refresh('lore', owner=f'{kind}:{owner_id}')
                with ui.element('div').classes('ll-form-row justify-end'):
                    ctx.button('Apply entry import', lambda: ctx.store.import_lore_entries(ctx.guild_id, kind, owner_id, imported, revision),
                               'lore.import', {'kind': kind, 'id': owner_id}, then=saved)
        with ui.element('div').classes('ll-stack'):
            ctx.upload(uploaded, 'Upload JSON entries')
        with ui.element('div').classes('ll-form-row justify-end'):
            ui.button('Cancel', on_click=dialog.close).props('flat')
    dialog.open()
