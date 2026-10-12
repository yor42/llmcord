"""World/channel guidelines, owner deletion, and additive lore imports."""
from __future__ import annotations

import inspect

from nicegui import ui

from .i18n import UserError, codes, web_locale
from .lorebooks import parse_lorebook


def guideline_editor(ctx, kind, owner_id, label, button=True):
    """Guidelines textarea; with button=False no Save button is added and (control, current) is returned for a save-bar editor."""
    t = web_locale.current()
    current = ctx.store.guidelines(ctx.guild_id, kind, owner_id)
    control = ui.textarea(label, value=current['content']).classes('w-full')
    control.props['placeholder'] = t("Setting, participants' fictional roles, tone, and interaction conventions")
    if not button:
        return control, current
    # i18n: {label} is the guidelines field label in lower case, such as "world guidelines"
    ctx.button(t('Save {label}', label=label.lower()),
               lambda: ctx.store.save_guidelines(ctx.guild_id, kind, owner_id, control.value or '', current['revision']),
               'guidelines.edit', {'kind': kind, 'id': owner_id}, then=lambda _: ctx.refresh())


def confirm_dialog(ctx, title, lines, button_label, operation, action, detail=None, then=None, enabled=True, conflict=None):
    """One shape for destructive confirmations: bold title, explanation lines, Cancel and a red confirm button."""
    t = web_locale.current()
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
        async def stale():
            dialog.close()
            followup = conflict()
            if inspect.isawaitable(followup):
                await followup
        with ui.element('div').classes('ll-form-row justify-end'):
            ui.button(t.p('button', 'Cancel'), on_click=dialog.close).props('flat')
            ctx.button(button_label, operation, action, detail, then=done, color='negative', **({'conflict': stale} if conflict else {})).set_enabled(enabled)
    dialog.on('hide', dialog.delete)
    dialog.open()


async def delete_space_dialog(ctx, space):
    t = web_locale.current()
    impact = await ctx.run(lambda: ctx.store.space_delete_impact(ctx.guild_id, space['id']))
    if impact is None:
        return
    lines = [t('This removes its guidelines, {entries} owned lore entries, encounters, and hub/lorebook links. Lorebooks and past messages remain.', entries=impact['entries'])]
    if impact['characters']:
        lines.append(t('Move or delete these home characters first: {names}', names=', '.join(impact['characters'])))
    if impact['channels']:
        lines.append(t('Rebind its {count} Discord channel(s) to another world or hub first.', count=len(impact['channels'])))
    world = space['kind'] == 'world'
    confirm_dialog(ctx, t('Delete world {name}?', name=space['name']) if world else t('Delete hub {name}?', name=space['name']), lines, t('Delete world') if world else t('Delete hub'),
                   lambda: ctx.store.delete_space(ctx.guild_id, space['id'], impact['revision']),
                   'space.delete', {'id': space['id']}, then=lambda _: ctx.refresh(),
                   enabled=not impact['characters'] and not impact['channels'])


async def delete_book_dialog(ctx, book):
    t = web_locale.current()
    def preview():
        ctx.store.validate_owner(ctx.guild_id, 'book', book['id'])
        return len(ctx.store.admin_entries(ctx.guild_id, 'book', book['id'])), ctx.store.owner_revision(ctx.guild_id, 'book', book['id'])
    result = await ctx.run(preview)
    if result is None:
        return
    count, revision = result
    confirm_dialog(ctx, t('Delete lorebook {name}?', name=book['name']),
                   [t('This deletes the lorebook, its {count} remaining entries, and its world/hub links. Entries moved to another owner remain.', count=count)],
                   t('Delete lorebook'), lambda: ctx.store.delete_lorebook(ctx.guild_id, book['id'], revision),
                   'book.delete', {'id': book['id']}, then=lambda _: ctx.refresh('imports'))


def direct_import_dialog(ctx, kind, owner_id, on_saved=None):
    t = web_locale.current()
    owner = next((owner for owner in ctx.service.owners(ctx.guild_id, ctx.channel_names, ctx.thread_names) if (owner['kind'], owner['id']) == (kind, owner_id)), None)
    if owner is None:
        raise UserError(codes.SCENE_OWNER_CHOOSE)
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-4xl'):
        ui.label(t('Import entries into {owner}', owner=owner['label'])).classes('text-xl font-bold')
        ui.label(t('Adds entries directly to this owner. Existing lore stays in place; identical entries are skipped. No new lorebook is created.')).classes('ll-muted')
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
                ui.label(t('{format}: {additions} new entries, {present} already present', format=format_name, additions=additions, present=len(changes) - additions)).classes('ll-subtitle')
                statuses = {'add': t.p('status', 'add'), 'duplicate': t.p('status', 'duplicate')}
                ui.table(columns=[{'name': key, 'label': label, 'field': key, 'align': 'left'} for key, label in
                                  (('uid', t.p('column', 'Source ID')), ('status', t.p('column', 'Action')), ('content', t.p('column', 'Content')), ('warnings', t('Compatibility notes')))],
                         rows=[{'uid': change['uid'], 'status': statuses.get(change['status'], change['status']), 'content': change['after'], 'warnings': '; '.join(change['warnings'])} for change in changes],
                         row_key='uid', pagination=10).classes('w-full')
                def saved(refs):
                    dialog.close()
                    ui.notify(t('Imported {count} entries', count=len(refs)), type='positive')
                    if on_saved:
                        on_saved(refs)
                    else:
                        return ctx.refresh('lore', owner=f'{kind}:{owner_id}')
                with ui.element('div').classes('ll-form-row justify-end'):
                    ctx.button(t('Apply entry import'), lambda: ctx.store.import_lore_entries(ctx.guild_id, kind, owner_id, imported, revision),
                               'lore.import', {'kind': kind, 'id': owner_id}, then=saved)
        with ui.element('div').classes('ll-stack'):
            ctx.upload(uploaded, t('Upload JSON entries'))
        with ui.element('div').classes('ll-form-row justify-end'):
            ui.button(t.p('button', 'Cancel'), on_click=dialog.close).props('flat')
    dialog.on('hide', dialog.delete)
    dialog.open()
