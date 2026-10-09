"""Sticky unsaved-changes bar (D21): one per page, at most one dirty editor, writes go through LiveContext._attempt."""
from __future__ import annotations

import inspect
import logging
import types

logger = logging.getLogger(__name__)

_SAVE_LABEL = 'Save changes'
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
                self.save_button = ui.button(_SAVE_LABEL, on_click=self.save).props('no-caps')
        self.bar.set_visibility(False)

    def track(self, name, controls, save=None, action=None, detail=None, then=None, success=None, on_reset=None,
              dirty=None, reset=None, save_label=_SAVE_LABEL, parts=None):
        """Register an editor: controls maps each tracked control to its saved value; save runs guarded/audited on Save.

        dirty: optional callable replacing the per-control comparison (an exception counts as dirty).
        reset: optional callable used by Reset instead of restoring control values (on_reset still runs after).
        save_label: the bar's Save button text while this editor is active.
        Writes made through ctx._attempt (not ctx.run/button) are never refused, so an editor may run its own
        write (e.g. Save as new) that way and then call settle().

        parts: instead of save/action/detail, a list of dicts {controls, operation, action, detail=None} (a subset of
        the editor's controls mapped to their saved values, plus the operation, audit action and detail passed to
        ctx._attempt). The tracked controls are the union of the parts' controls plus any extra `controls`. Save runs
        only the parts whose controls differ, in order, each its own guarded/audited write; a part that succeeds
        becomes the new saved baseline, a failure stops the rest and leaves the bar active for the unsaved parts.
        When all succeed: one success toast, settle, then then(results), results being the list of per-part
        results in run order. save and parts are mutually exclusive, and extra non-part controls are rejected with
        parts (ValueError). retrack() raises NotImplementedError for a parts editor. Operations should read control
        values at call time: the baseline is rebased to the values captured just before each part ran. If a control
        was edited during Save the bar stays active (then still runs). On a partial failure then/success are not
        called, even for parts that already succeeded."""
        if parts and save is not None:
            raise ValueError('save and parts are mutually exclusive')
        if parts and controls:
            raise ValueError('controls must be given through parts')
        parts = [types.SimpleNamespace(controls=dict(part['controls']), operation=part['operation'],
                                       action=part.get('action'), detail=part.get('detail')) for part in parts or []]
        controls = {**controls}
        for part in parts:
            controls.update(part.controls)
        editor = types.SimpleNamespace(name=name, controls=[], save=save, action=action, detail=detail, then=then,
                                       success=success, on_reset=on_reset, dirty=dirty, reset=reset, save_label=save_label,
                                       parts=[])
        self.editors.append(editor)
        self.retrack(editor, controls)
        editor.parts = parts
        return editor

    def retrack(self, editor, controls):
        """Replace an editor's tracked controls (after a re-render); the old ones stop reporting. Call check(editor) after. The old controls must already be gone from the page: they stop being tracked, so settle() will not re-enable them."""
        if editor.parts:
            raise NotImplementedError('retrack is not supported for editors with parts')
        previous = [tracked for tracked, _ in editor.controls]
        editor.controls = list(controls.items())
        for control, _ in editor.controls:
            if not any(control is old for old in previous):
                control.on_value_change(lambda _event, editor=editor, control=control: self._changed(editor, control))
            if self.active is not None and self.active is not editor:
                control.disable()

    def _changed(self, editor, control):
        if any(control is tracked for tracked, _ in editor.controls):
            self.check(editor)

    def differs(self, editor):
        if editor.dirty:
            try:
                return bool(editor.dirty())
            except Exception:
                logger.exception('Save bar dirty check failed for %s', editor.name)
                return True
        return any(not _same(control.value, saved) for control, saved in editor.controls)

    def check(self, editor):
        if self.active is not None and self.active is not editor:
            return
        if self.differs(editor):
            if self.active is None:
                self.active = editor
                self.text.set_text(f'{editor.name} has unsaved changes.')
                self.save_button.set_text(editor.save_label)
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
        self.save_button.set_text(_SAVE_LABEL)
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
        if editor.reset:
            editor.reset()
        else:
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
            if editor.parts:
                ok, result = await self._save_parts(editor)
            else:
                ok, result = await self.ctx._attempt(editor.save, editor.action, editor.detail)
        finally:
            self.saving = False
            self.save_button.enable()
        if not ok:
            return
        if editor.success:
            ui.notify(editor.success, type='positive')
        if not editor.parts or not self.differs(editor):
            self.settle()
        if editor.then:
            followup = editor.then(result)
            if inspect.isawaitable(followup):
                await followup

    async def _save_parts(self, editor):
        saved = {id(control): value for control, value in editor.controls}
        results = []
        for part in editor.parts:
            sent = {control: control.value for control in part.controls}
            if all(_same(value, saved[id(control)]) for control, value in sent.items()):
                continue
            ok, result = await self.ctx._attempt(part.operation, part.action, part.detail)
            if not ok:
                return False, None
            results.append(result)
            editor.controls = [(control, sent[control] if control in sent else value) for control, value in editor.controls]
            saved.update({id(control): value for control, value in sent.items()})
        return True, results
