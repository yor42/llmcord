"""Sticky unsaved-changes bar (D21): one per page, at most one dirty editor, writes go through LiveContext._attempt."""
from __future__ import annotations

import inspect
import types

_LEAVE_WARNING = 'window.onbeforeunload = (e) => { e.preventDefault(); e.returnValue = ""; return ""; };'


def _same(a, b):
    return ('' if a is None else a) == ('' if b is None else b)


class SaveBar:
    def __init__(self, ctx):
        from nicegui import ui
        self.ctx, self.editors, self.active, self.saving = ctx, [], None, False
        self.bar = ui.element('div').classes('ll-savebar').props('role=region aria-label="Unsaved changes"')
        with self.bar:
            self.text = ui.label('').classes('ll-savebar-text').props('aria-live=polite')
            with ui.element('div').classes('ll-savebar-actions'):
                ui.button('Reset', on_click=self.reset).props('flat no-caps').classes('ll-savebar-reset')
                self.save_button = ui.button('Save changes', on_click=self.save).props('no-caps')
        self.bar.set_visibility(False)

    def track(self, name, controls, save, action=None, detail=None, then=None, success=None, on_reset=None):
        """Register an editor: controls maps each tracked control to its saved value; save runs guarded/audited on Save changes."""
        editor = types.SimpleNamespace(name=name, controls=list(controls.items()), save=save, action=action,
                                       detail=detail, then=then, success=success, on_reset=on_reset)
        self.editors.append(editor)
        for control, _ in editor.controls:
            control.on_value_change(lambda _event, editor=editor: self.check(editor))
            if self.active:
                control.disable()
        return editor

    def differs(self, editor):
        return any(not _same(control.value, saved) for control, saved in editor.controls)

    def check(self, editor):
        if self.active is not None and self.active is not editor:
            return
        if self.differs(editor):
            if self.active is None:
                self.active = editor
                self.text.set_text(f'{editor.name} has unsaved changes.')
                self.bar.set_visibility(True)
                for other in self.editors:
                    if other is not editor:
                        for control, _ in other.controls:
                            control.disable()
                self.ctx.selector.client.run_javascript(_LEAVE_WARNING)
        elif self.active is editor:
            self.settle()

    def settle(self):
        """Hide the bar and re-enable every tracked field."""
        was_active, self.active = self.active, None
        self.bar.set_visibility(False)
        for editor in self.editors:
            for control, _ in editor.controls:
                control.enable()
        if was_active is not None:
            self.ctx.selector.client.run_javascript('window.onbeforeunload = null;')

    def clear(self):
        """Drop every registration (the editors are being rebuilt)."""
        self.settle()
        self.editors.clear()

    def refuse(self):
        """True (after a toast and alert flash) when another write must wait for the dirty editor."""
        if self.active is None:
            return False
        from nicegui import ui
        ui.notify(f'Save or reset your changes to {self.active.name} first.', type='negative', timeout=8000)
        self.ctx.selector.client.run_javascript(
            f'const b = document.getElementById({self.bar.html_id!r}); if (b) {{ b.classList.add("ll-savebar-alert"); setTimeout(() => b.classList.remove("ll-savebar-alert"), 1000); }}')
        return True

    def reset(self):
        editor = self.active
        if editor is None:
            return
        for control, saved in editor.controls:
            control.set_value(saved)
        if editor.on_reset:
            editor.on_reset()
        self.settle()

    async def save(self):
        from nicegui import ui
        editor = self.active
        if editor is None or self.saving:
            return
        self.saving = True
        self.save_button.disable()
        try:
            ok, result = await self.ctx._attempt(editor.save, editor.action, editor.detail)
        finally:
            self.saving = False
            self.save_button.enable()
        if not ok:
            return
        if editor.success:
            ui.notify(editor.success, type='positive')
        self.settle()
        if editor.then:
            followup = editor.then(result)
            if inspect.isawaitable(followup):
                await followup
