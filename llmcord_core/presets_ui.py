"""Dashboard Prompts tab: the presets workspace (list, editor, import, export, preview)."""
from __future__ import annotations

import copy
import json

from .i18n import N_, NP_, UserError, codes, web_locale
from .icons import lucide, more_menu
from .macro_highlight import MacroHighlight
from .prompts import PURPOSES, SOURCES, block, compatibility, default_bundle, ST_MARKERS, export_preset, parse_preset, purpose_blocks
from .scene_ui import confirm_dialog


PURPOSE_LABELS = {'dialogue': NP_('model role', 'Dialogue'), 'director': NP_('model role', 'Director'), 'extraction': N_('Extraction'),
                  'summary': N_('Summary'), 'images': N_('Images')}
ADAPT_LABELS = {'top_system': N_('moved to top-level system instructions'), 'user': N_('converted to user instructions')}


def _preset_snapshot(state, name):
    """Text snapshot of the editable preset state (name, bundle, raw-JSON textareas as typed) for the save bar's dirty check.

    Viewing another Purpose fills in its default blocks, so every purpose is normalised first; raw JSON is compared as text and never parsed."""
    from .dashboard import pretty
    def plain(value):
        if isinstance(value, float) and value == int(value):
            return int(value)
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, list):
            return [plain(item) for item in value]
        return value
    bundle = copy.deepcopy(state['bundle'])
    purposes = bundle.get('purposes', {})
    for purpose in PURPOSES:
        if purposes.get(purpose) is None:
            purposes[purpose] = copy.deepcopy(default_bundle()['purposes'][purpose])
    if not bundle.get('source'):
        bundle.pop('source', None)
    raw = {}
    for blocks in purposes.values():
        for b in blocks:
            control = state['raw_controls'].get(b['id'])
            text = control[1].value if control and control[0] is not None else pretty(b.get('raw'))
            raw[b['id']] = text
            b.pop('raw', None)
    return json.dumps({'name': name.value or '', 'bundle': plain(bundle), 'raw': raw}, sort_keys=True, default=str)


def presets_panel(ctx):
    from nicegui import ui
    from .dashboard import section
    t = web_locale.current()
    store, gid = ctx.store, ctx.guild_id
    presets = store.list_presets(gid)
    IMPORTED = -1

    def library_options(rows, preview=False):
        return {0: t('Built-in default'), **{r['id']: t('{name} · draft {revision}', name=r['name'], revision=r['revision']) for r in rows}, **({IMPORTED: t('Imported (not saved)')} if preview else {})}

    def subtitle_text(rows):
        current = store.active_preset(gid)
        return t('Active preset: {label} · revision {revision}', label=library_options(rows).get(current['id'], str(current['id'])), revision=current['revision'])
    choices = library_options(presets)
    active = store.active_preset(gid)
    selected = next((r for r in presets if r['id'] == active['id']), None)
    state = {'id': active['id'], 'revision': selected['revision'] if selected else 0, 'bundle': store.preset_bundle(gid, active['id']), 'raw_controls': {},
             'controls': [], 'after_render': None, 'saved': None, 'preview': False, 'label': selected['name'] if selected else t('Built-in default')}
    editor = None
    baseline_label = {'text': state['label']}

    def collect():
        for b, control in state['raw_controls'].values():
            b['raw'] = json.loads(control.value or '{}')
        return copy.deepcopy(state['bundle'])

    def preset_menu():
        with more_menu('preset'):
            ui.menu_item(t('Save as new preset'), on_click=lambda: save_new()).props('role=menuitem')
            ui.menu_item(t('Activate saved revision'), on_click=lambda: activate_saved()).props('role=menuitem')
            ui.separator()
            ui.menu_item(t('Delete preset'), on_click=lambda: ask_delete()).classes('text-negative').props('role=menuitem')

    with section(t('Preset library'), header=preset_menu):
        subtitle = ui.label(subtitle_text(presets)).classes('ll-subtitle')
        select = ui.select(choices, value=active['id'], label=t('Preset library')).classes('w-full')
        name = ui.input(t('Preset name'), value=selected['name'] if selected else 'Default copy').classes('w-full')

    def rebase(label):
        """Take the saved baseline (after a render) that Reset and the dirty check compare against."""
        state['label'] = baseline_label['text'] = label
        state['saved'] = {'snapshot': _preset_snapshot(state, name), 'bundle': copy.deepcopy(state['bundle']), 'name': name.value, 'id': state['id'], 'revision': state['revision']}
        if editor is not None:
            editor.name = label

    def dirty():
        return _preset_snapshot(state, name) != state['saved']['snapshot']

    def tracked():
        return {control: None for control in [name, *state['controls']]}

    def after_render():
        if editor is not None:
            ctx.savebar.retrack(editor, tracked())
            if not state.get('quiet'):
                ctx.savebar.check(editor)

    def show_library(rows=None):
        """Point the library select at the loaded preset, or at 'Imported (not saved)' while an import preview is unsaved."""
        select.set_options(library_options(rows if rows is not None else store.list_presets(gid), state['preview']), value=IMPORTED if state['preview'] else state['id'])

    def reset():
        baseline = state['saved']
        was_preview = state['preview']
        state.update(id=baseline['id'], revision=baseline['revision'], bundle=copy.deepcopy(baseline['bundle']), label=baseline_label['text'], preview=False)
        editor.name = baseline_label['text']
        if was_preview:
            show_library()  # drops the 'Imported (not saved)' option
        else:
            select.value = baseline['id']
        name.value = baseline['name']
        render_editor.refresh()

    def adopt(label):
        """Show a freshly loaded, saved or imported preset as the new unmodified baseline."""
        state['quiet'] = True
        try:
            render_editor.refresh()
        finally:
            state['quiet'] = False
        rebase(label)
        ctx.savebar.check(editor)

    async def load():
        shown = IMPORTED if state['preview'] else state['id']
        if select.value == shown:
            return
        if ctx.savebar.refuse():
            select.value = shown
            return
        bundle = await ctx.run(lambda: store.preset_bundle(gid, select.value))
        if bundle is None:
            select.value = shown
            return
        chosen = next((r for r in store.list_presets(gid) if r['id'] == select.value), None)
        state.update(id=select.value, revision=chosen['revision'] if chosen else 0, bundle=bundle, preview=False)
        show_library()
        state.get('open_blocks', set()).clear()
        name.value = chosen['name'] if chosen else 'Default copy'
        adopt(chosen['name'] if chosen else t('Built-in default'))

    select.on_value_change(lambda _: load())

    def save():
        return store.save_preset(gid, name.value or '', collect(), state['id'] or None, state['revision'] if state['id'] else None)

    def saved(result):
        state['id'], state['revision'] = result
        state['preview'] = False
        fresh = store.list_presets(gid)
        show_library(fresh)
        subtitle.set_text(subtitle_text(fresh))
        rebase(name.value or state['label'])
        ctx.savebar.check(editor)
        ui.notify(t('Draft saved. Activate it when ready.'), type='positive')

    with section(t('Prompt blocks')):
        purpose = ui.select({p: t.render(PURPOSE_LABELS[p]) for p in PURPOSES}, value='dialogue', label=t.p('field', 'Purpose'))
        MacroHighlight()
        render_editor = _preset_editor(ctx, state, purpose, collect)
        render_editor()
        rebase(state['label'])
        editor = ctx.savebar.track(state['label'], tracked(), save, 'preset.save', then=saved, dirty=dirty, reset=reset, save_label=t.p('button', 'Save draft'))
        state['after_render'] = after_render
        async def change_purpose():
            if await ctx.run(lambda: True):
                collect()
                render_editor.refresh()
        purpose.on_value_change(lambda _: change_purpose())

    async def save_new():
        if ctx.savebar.active is not None and ctx.savebar.active is not editor:
            ctx.savebar.refuse()
            return

        def duplicate():
            taken = {r['name'] for r in store.list_presets(gid)}
            base = (name.value or '').strip()
            if base in taken:  # an unchanged name would hit the unique name; propose the next free "<name> copy"
                base = base[:60]
                proposal = f'{base} copy'
                n = 2
                while proposal in taken:
                    proposal, n = f'{base} copy {n}', n + 1
                name.value = proposal
            return store.save_preset(gid, name.value or '', collect())
        ok, result = await ctx._attempt(duplicate, 'preset.create')
        if ok:
            ctx.savebar.settle()
            saved(result)

    async def activate_saved():
        def activate():
            store.activate_preset(gid, state['id'], state['revision'], ctx.service.providers)
            return True
        if await ctx.run(activate, 'preset.activate'):
            await ctx.refresh()

    def ask_delete():
        if ctx.savebar.refuse():
            return
        def delete():
            store.delete_preset(gid, state['id'])
            return True
        if not state['id']:
            ui.notify(t('The built-in default cannot be deleted'), type='negative', timeout=8000)
            return
        current = next((r for r in store.list_presets(gid) if r['id'] == state['id']), None)
        confirm_dialog(ctx, t('Delete preset {name}?', name=current['name'] if current else state['id']),
                       [t('This permanently deletes the preset. The active preset cannot be deleted; activate another one first.')],
                       t('Delete preset'), delete, 'preset.delete', then=lambda _: ctx.refresh())

    def imported(bundle):
        editor.name = state['label'] = t('Imported preset')
        state.update(id=0, revision=0, bundle=bundle, preview=True)
        state.get('open_blocks', set()).clear()
        show_library()
        name.value = 'Imported preset'
        render_editor.refresh()
        ctx.savebar.check(editor)  # an import preview is unsaved: the baseline stays the previous preset
        ui.notify(t('Import preview loaded. Resolve compatibility items, save a draft, then activate.'))

    _preset_import(ctx, state, imported)
    _preset_export(ctx, collect)
    _preset_preview(ctx, purpose, collect)


def _read_button(ctx, text, operation):
    """A button that runs a read-only operation through the guard but stays available while the save bar is active."""
    from nicegui import ui
    async def clicked():
        await ctx.run(operation)
    return ui.button(text, on_click=clicked)


def _preset_editor(ctx, state, purpose, collect):
    from nicegui import ui
    from .dashboard import pretty, _span, _prompt_options, PROMPT_ROLE_LABELS, PROMPT_PLACEMENT_LABELS, PROMPT_PLACEMENT_SHORT, PROMPT_SOURCE_LABELS
    t = web_locale.current()

    @ui.refreshable
    def render_editor():
        state['raw_controls'] = {}
        state['controls'] = controls = []
        def tracked(control):
            controls.append(control)
            return control
        blocks = state['bundle']['purposes'].setdefault(purpose.value, copy.deepcopy(default_bundle()['purposes'][purpose.value]))
        with ui.column().classes('w-full gap-4') as ordered:
            for position, b in enumerate(blocks):
                with ui.card().classes('w-full prompt-block ll-stack ll-result') as card:
                    card.prompt_id = b['id']
                    key = (purpose.value, b['id'])  # block ids repeat across purposes
                    opened = key in state.setdefault('open_blocks', set())
                    if opened:
                        card.classes('ll-block-open')
                    with ui.element('div').classes('ll-block-head'):
                        lucide('grip-vertical', '1.25em').classes('prompt-handle cursor-grab self-center ll-icon-solo')
                        toggle = ui.element('button').classes('ll-block-toggle prompt-toggle')
                        toggle.props['type'] = 'button'
                        toggle.props['aria-expanded'] = 'true' if opened else 'false'
                        with toggle:
                            # phrasing content only (a <button> cannot hold divs); trailing spaces keep the accessible name readable
                            _span().classes('ll-block-name').bind_text_from(b, 'name', backward=lambda v: (v or t('Untitled block')) + ' ')
                            with ui.element('span').classes('ll-block-meta'):
                                _span().bind_text_from(b, 'role', backward=lambda v: t.render(PROMPT_ROLE_LABELS.get(v, v)))
                                _span(' \u00b7 ')
                                _span().bind_text_from(b, 'placement', backward=lambda v: f'{t.render(PROMPT_PLACEMENT_SHORT.get(v, v))} ')
                            _span(t.p('status', 'Disabled')).classes('ll-pill ll-pill-off').bind_visibility_from(b, 'enabled', backward=lambda v: not v)
                            lucide('chevron-down', '1.25em').classes('ll-icon-solo ll-block-caret')
                        async def remove(b=b, key=key):
                            if await ctx.run(lambda: True):
                                collect()
                                blocks.remove(b)
                                state['open_blocks'].discard(key)
                                render_editor.refresh()
                        async def move(delta, b=b):
                            # same path as a drag: collect the fields, reorder the list, re-render (the save bar sees the new order)
                            if await ctx.run(lambda: True):
                                collect()
                                at = blocks.index(b)
                                if 0 <= at + delta < len(blocks):
                                    blocks.insert(at + delta, blocks.pop(at))
                                render_editor.refresh()
                        with more_menu(b['name'] or t('Untitled block')) as menu:
                            more_button = menu.parent_slot.parent
                            up = ui.menu_item(t('Move up'), on_click=lambda b=b: move(-1, b)).props('role=menuitem')
                            down = ui.menu_item(t('Move down'), on_click=lambda b=b: move(1, b)).props('role=menuitem')
                            if position == 0:
                                up.props('disable')
                            if position == len(blocks) - 1:
                                down.props('disable')
                            ui.separator()
                            ui.menu_item(t('Remove block'), on_click=remove).classes('text-negative').props('role=menuitem')
                    body = ui.element('div').classes('ll-stack w-full')
                    if not opened:
                        body.classes('ll-collapsed')
                    def flip(card=card, toggle=toggle, body=body, bid=key):
                        open_ids = state['open_blocks']
                        now_open = bid not in open_ids
                        (open_ids.add if now_open else open_ids.discard)(bid)
                        card.classes(add='ll-block-open') if now_open else card.classes(remove='ll-block-open')
                        body.classes(remove='ll-collapsed') if now_open else body.classes(add='ll-collapsed')
                        toggle.props['aria-expanded'] = 'true' if now_open else 'false'
                        toggle.update()
                    toggle.on('click', flip)
                    with body:
                        with ui.element('div').classes('ll-form-row'):
                            name_input = tracked(ui.input(t('Block name'))).bind_value(b, 'name')
                            def rename(event, button=more_button):
                                button.props['aria-label'] = t('More actions for {name}', name=event.value or t('Untitled block'))
                                button.update()
                            name_input.on_value_change(rename)
                            tracked(ui.switch(t.p('field', 'Enabled'))).bind_value(b, 'enabled')
                        sources = _prompt_options({k: t.render(PROMPT_SOURCE_LABELS.get(k, k)) for k in sorted(SOURCES, key=lambda k: t.render(PROMPT_SOURCE_LABELS.get(k, k)))}, b['source'], t=t)
                        tracked(ui.select(sources, label=t('Context source'))).bind_value(b, 'source').classes('w-full')
                        tracked(ui.textarea(t('Prompt text / template'))).bind_value(b, 'content').classes('w-full ll-macro').props('rows=6')
                        with ui.element('div').classes('ll-form-row'):
                            tracked(ui.select(_prompt_options(PROMPT_ROLE_LABELS, b['role'], t=t), label=t.p('field', 'Role'))).bind_value(b, 'role')
                            tracked(ui.select(_prompt_options(PROMPT_PLACEMENT_LABELS, b['placement'], t=t), label=t.p('field', 'Placement'))).bind_value(b, 'placement')
                            tracked(ui.number(t.p('field', 'Depth'), precision=0)).bind_value(b, 'depth', backward=lambda v: int(v or 0))
                            tracked(ui.number(t('Injection order'), precision=0)).bind_value(b, 'order', backward=lambda v: int(v or 0))
                            tracked(ui.number(t('Trimming priority'), precision=0)).bind_value(b, 'priority', backward=lambda v: int(v or 0))
                        tracked(ui.select({'': t('Exact placement'), 'top_system': t('Move to top-level system instructions'), 'user': t('Convert late system block to user instructions')}, label=t('System instruction placement'))).bind_value(b, 'adaptation').classes('w-full')
                        with ui.expansion(t('Preserved import fields and compatibility remapping')).classes('w-full ll-subpanel'), ui.element('div').classes('ll-stack'):
                            raw = tracked(ui.textarea(t('Original prompt fields (JSON)'), value=pretty(b['raw']))).classes('w-full').props('rows=6')
                            state['raw_controls'][b['id']] = (b, raw)
                            ui.label(t('Block ID: {id}', id=b['id'])).classes('ll-muted')
                            ui.label(t('To remap triggers/extensions: clear injection_trigger and set extension to false.')).classes('ll-muted')
            async def reordered(event):
                if await ctx.run(lambda: True):
                    collect()
                    by_id = {b['id']: b for b in blocks}
                    blocks[:] = [by_id[child.prompt_id] for child in ordered if hasattr(child, 'prompt_id')]
                    render_editor.refresh()
            ordered.make_sortable(handle='.prompt-handle', on_end=reordered, options={'draggable': '.prompt-block'})
        async def add():
            if await ctx.run(lambda: True):
                collect()
                import uuid
                fresh = block('custom-' + uuid.uuid4().hex[:8])
                blocks.append(fresh)
                state.setdefault('open_blocks', set()).add((purpose.value, fresh['id']))
                render_editor.refresh()
        with ui.element('div').classes('ll-form-row'):
            ui.button(t('Add prompt block'), on_click=add).props('outline')
        for problem in compatibility(state['bundle'], ctx.service.providers):
            ui.label(problem).classes('text-amber-300')
        source = state['bundle'].setdefault('source', {})
        if source.get('assistant_prefill'):
            tracked(ui.checkbox(t('Disable imported assistant prefill'))).bind_value(source, 'prefill_disabled')
        if state['after_render']:
            state['after_render']()

    return render_editor


def _preset_import(ctx, state, imported):
    from nicegui import ui
    from .dashboard import section
    t = web_locale.current()
    with section(t('Import preset')):
        order_area = ui.column().classes('w-full')
        async def uploaded(event):
            data = await event.file.read()
            bundle, profiles = parse_preset(data)
            if profiles:
                order_area.clear()
                with order_area, ui.element('div').classes('ll-form-row'):
                    order = ui.select({p['index']: t('SillyTavern order {label}', label=p['label']) for p in profiles}, label=t('Choose an order profile'))
                    def choose():
                        if order.value is None:
                            raise UserError(codes.PRESETS_UI_ORDER_REQUIRED)
                        return parse_preset(data, order.value)[0]
                    ctx.button(t('Preview selected profile'), choose, then=imported)
            else:
                imported(bundle)
        with ui.element('div').classes('ll-stack'):
            ctx.upload(uploaded, t('Upload native or SillyTavern Chat Completion JSON'))


def _preset_export(ctx, collect):
    from nicegui import ui
    from .dashboard import pretty, section
    t = web_locale.current()
    with section(t('Export preset')):
        def native_export():
            ui.download.content(pretty(export_preset(collect())), 'llmcord-preset.json', 'application/json')
            return True
        with ui.element('div').classes('ll-form-row'):
            _read_button(ctx, t('Export full native bundle'), native_export).props('outline')
        def block_options():
            try:
                portable = set(ST_MARKERS.values())  # the check export_preset uses
                def option(b):
                    label = f"{b['name'] or t('Untitled block')} ({b['id']})"
                    return label if b['source'] == 'text' or b['source'] in portable or b['source'].startswith('unsupported:') else t('{label} \u2014 not portable', label=label)
                return {b['id']: option(b) for b in purpose_blocks(collect(), 'dialogue')}
            except ValueError:
                return {}
        omit = ui.select(block_options(), multiple=True, value=[], label=t('Leave out blocks')).props('use-chips').classes('w-full')
        omit.tooltip(t('SillyTavern cannot represent some blocks; choose them here to export without them.'))
        def refresh_options():
            omit.set_options(block_options(), value=[v for v in omit.value if v in block_options()])
        omit.on('popup-show', refresh_options)
        def st_export():
            ui.download.content(pretty(export_preset(collect(), sillytavern=True, omit=list(omit.value))), 'sillytavern-preset.json', 'application/json')
            return True
        with ui.element('div').classes('ll-form-row'):
            _read_button(ctx, t('Export SillyTavern dialogue preset'), st_export).props('outline')


def _preset_preview(ctx, purpose, collect):
    from nicegui import ui
    from .dashboard import section, PROMPT_ROLE_LABELS
    t = web_locale.current()
    with section(t('Assembled request preview')):
        chars = {r['id']: r['name'] for r in ctx.snapshot.characters}
        with ui.element('div').classes('ll-form-row'):
            character = ui.select(chars, label=t('Sample character'))
            channel = ui.select({row['channel_id']: ctx.channel_names.get(row['channel_id'], str(row['channel_id'])) for row in ctx.snapshot.channels}, label=t('Sample channel (optional)'))
        sample = ui.textarea(t('Sample input'), value=t.p('sample input', 'Hello!')).classes('w-full')
        history = ui.textarea(t('Sample history (one message per line)')).classes('w-full')
        diagnostics = ui.column().classes('w-full')
        def preview():
            bundle = collect()
            request = ctx.service.preview_prompt(ctx.guild_id, bundle, purpose.value, character.value, sample.value or '', history.value or '', channel.value, t=t)
            diagnostics.clear()
            with diagnostics:
                ui.label(t('Estimated input tokens: {count}', count=request.estimated_tokens))
                names = {b['id']: b.get('name') or b['id'] for b in purpose_blocks(bundle, purpose.value)}
                def omitted_text(entry):
                    ident, sep, rest = entry.partition(':')
                    if sep and rest.isdigit():
                        return t('{name} (history message {number})', name=names.get(ident, ident), number=int(rest) + 1)
                    return names.get(entry, entry)
                def adapted_text(entry):
                    ident, sep, rest = entry.partition(': ')
                    return f'{names.get(ident, ident)}: {t.render(ADAPT_LABELS.get(rest, rest))}' if sep else entry
                ui.label(t('Omitted: {items}', items=', '.join(omitted_text(e) for e in request.omitted) or t.p('value', 'none')))
                ui.label(t('Adaptations: {items}', items=', '.join(adapted_text(e) for e in request.adaptations) or t.p('value', 'none')))
                for message in request.messages:
                    with ui.expansion(t.render(PROMPT_ROLE_LABELS.get(message.role, message.role))).classes('w-full'):
                        ui.label(message.text).classes('whitespace-pre-wrap')
            return True
        with ui.element('div').classes('ll-form-row'):
            _read_button(ctx, t('Preview without a model call'), preview)
