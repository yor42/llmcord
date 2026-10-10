"""Offline tests for SaveBar's dirty/reset/retrack/label logic (stub controls; the browser suite covers the UI)."""
import types
import unittest
from unittest import mock

from llmcord_core.admin_store import ConflictError
from llmcord_core.dashboard import AttemptFailed
from llmcord_core.savebar import SaveBar


class Control:
    def __init__(self, value=''):
        self.value, self.disabled, self.handlers = value, False, []

    def on_value_change(self, handler):
        self.handlers.append(handler)

    def set_value(self, value):
        self.value = value
        for handler in self.handlers:
            handler(None)

    def disable(self):
        self.disabled = True

    def enable(self):
        self.disabled = False


class Widget:
    def __init__(self):
        self.visible, self.label = False, 'Save changes'

    def set_visibility(self, visible):
        self.visible = visible

    def set_text(self, text):
        self.label = text

    def run_javascript(self, _script):
        pass

    def disable(self):
        pass

    def enable(self):
        pass


def make():
    bar = object.__new__(SaveBar)
    widget = Widget()
    client = types.SimpleNamespace(run_javascript=lambda _script: None)
    bar.ctx = types.SimpleNamespace(selector=types.SimpleNamespace(client=client))
    bar.editors, bar.active, bar.saving = [], None, False
    bar.bar, bar.text, bar.save_button, bar.reset_button = widget, Widget(), Widget(), Widget()
    return bar


class SaveBarLogicTests(unittest.TestCase):
    def test_control_comparison_unchanged(self):
        bar, field = make(), Control('a')
        editor = bar.track('X', {field: 'a'}, save=None)
        field.set_value('b')
        self.assertIs(bar.active, editor)
        self.assertTrue(bar.bar.visible)
        field.set_value('a')
        self.assertIsNone(bar.active)
        self.assertFalse(bar.bar.visible)

    def test_dirty_callable_replaces_comparison_and_errors_count_as_dirty(self):
        bar, state = make(), {'dirty': False}
        editor = bar.track('P', {}, save=None, dirty=lambda: state['dirty'])
        self.assertFalse(bar.differs(editor))
        state['dirty'] = True
        bar.check(editor)
        self.assertIs(bar.active, editor)
        editor.dirty = lambda: 1 / 0
        self.assertTrue(bar.differs(editor))

    def test_reset_callable_replaces_restore_and_on_reset_runs_after(self):
        bar, field, calls = make(), Control('a'), []
        bar.track('P', {field: 'a'}, save=None, reset=lambda: calls.append('reset'), on_reset=lambda: calls.append('on'))
        field.set_value('b')
        bar.reset()
        self.assertEqual(calls, ['reset', 'on'])
        self.assertEqual(field.value, 'b')
        self.assertIsNone(bar.active)

    def test_retrack_drops_old_controls_and_disables_new_when_other_active(self):
        bar, old, new, other = make(), Control('a'), Control('a'), Control('x')
        editor = bar.track('P', {old: 'a'}, save=None)
        second = bar.track('Q', {other: 'x'}, save=None)
        bar.retrack(editor, {new: 'a'})
        old.set_value('zzz')
        self.assertIsNone(bar.active)
        new.set_value('b')
        self.assertIs(bar.active, editor)
        self.assertTrue(other.disabled)
        late = Control('x')
        bar.retrack(second, {late: 'x'})
        self.assertTrue(late.disabled)

    def test_retrack_keeps_one_handler_on_kept_controls(self):
        bar, kept, extra = make(), Control('a'), Control('a')
        editor = bar.track('P', {kept: 'a'}, save=None)
        bar.retrack(editor, {kept: 'a', extra: 'a'})
        bar.retrack(editor, {kept: 'a', extra: 'a'})
        self.assertEqual(len(kept.handlers), 1)
        self.assertEqual(len(extra.handlers), 1)
        checks = []
        bar.check = checks.append
        kept.set_value('b')
        self.assertEqual(checks, [editor])

    def test_save_label_follows_active_editor(self):
        bar, field = make(), Control('a')
        bar.track('P', {field: 'a'}, save=None, save_label='Save draft')
        field.set_value('b')
        self.assertEqual(bar.save_button.label, 'Save draft')
        bar.settle()
        self.assertEqual(bar.save_button.label, 'Save changes')


class SaveBarPartsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bar, self.calls, self.fail = make(), [], set()
        self.notes = []

        async def attempt(operation, action, detail):
            self.calls.append((operation, action, detail))
            return (operation not in self.fail), f'r-{operation}'
        self.bar.ctx._attempt = attempt
        self.fields = [Control('a'), Control('b'), Control('c')]
        self.editor = self.bar.track('Server', {}, parts=[
            {'controls': {field: field.value}, 'operation': name, 'action': f'act-{name}', 'detail': f'd-{name}'}
            for field, name in zip(self.fields, 'xyz')], then=self.notes.append, success=None)

    async def test_only_changed_parts_run_in_order(self):
        self.fields[2].set_value('c2')
        self.fields[0].set_value('a2')
        await self.bar.save()
        self.assertEqual(self.calls, [('x', 'act-x', 'd-x'), ('z', 'act-z', 'd-z')])
        self.assertIsNone(self.bar.active)
        self.assertEqual(self.notes, [['r-x', 'r-z']])

    async def test_failure_stops_later_parts_and_reset_restores_only_unsaved(self):
        for field in self.fields:
            field.set_value(field.value + '2')
        self.fail = {'y'}
        with mock.patch('nicegui.ui.notify'):
            await self.bar.save()
        self.assertEqual([call[0] for call in self.calls], ['x', 'y'])
        self.assertIs(self.bar.active, self.editor)
        self.assertEqual(self.notes, [])
        self.assertFalse(self.bar.saving)
        self.bar.reset()
        self.assertEqual([field.value for field in self.fields], ['a2', 'b', 'c'])
        self.assertIsNone(self.bar.active)

    async def test_retry_after_failure_reruns_only_remaining(self):
        self.fields[0].set_value('a2')
        self.fields[1].set_value('b2')
        self.fail = {'y'}
        await self.bar.save()
        self.fail, self.calls[:] = set(), []
        await self.bar.save()
        self.assertEqual([call[0] for call in self.calls], ['y'])
        self.assertIsNone(self.bar.active)
        self.assertEqual(self.notes, [['r-y']])

    async def test_edit_during_later_part_keeps_bar_active(self):
        self.fields[0].set_value('a2')
        self.fields[1].set_value('b2')
        inner = self.bar.ctx._attempt

        async def attempt(operation, action, detail):
            if operation == 'y':
                self.fields[0].value = 'a3'
            return await inner(operation, action, detail)
        self.bar.ctx._attempt = attempt
        await self.bar.save()
        self.assertIs(self.bar.active, self.editor)
        self.assertEqual(self.notes, [['r-x', 'r-y']])

    async def test_failure_toast_says_later_changes_are_pending(self):
        """UI-42: a failed part tells the user the later changed parts were not saved; no extra toast when none are pending."""
        self.fields[0].set_value('a2')
        self.fields[1].set_value('b2')
        self.fail = {'x'}
        with mock.patch('nicegui.ui.notify') as notify:
            await self.bar.save()
        notify.assert_called_once_with('Your other changes were not saved. They are still pending.', type='warning', timeout=8000)
        self.bar.reset()
        self.fields[0].set_value('a2')
        with mock.patch('nicegui.ui.notify') as notify:
            await self.bar.save()
        notify.assert_not_called()

    async def test_success_toast_on_full_success(self):
        self.editor.success = 'Saved'
        self.fields[0].set_value('a2')
        with mock.patch('nicegui.ui.notify') as notify:
            await self.bar.save()
        notify.assert_called_once_with('Saved', type='positive')

    async def test_unexpected_exception_propagates_and_releases_bar(self):
        async def boom(operation, action, detail):
            raise RuntimeError('x')
        self.bar.ctx._attempt = boom
        disabled = []
        self.bar.save_button.disable = lambda: disabled.append(1)
        self.bar.save_button.enable = lambda: disabled.append(0)
        self.fields[0].set_value('a2')
        with self.assertRaises(RuntimeError):
            await self.bar.save()
        self.assertFalse(self.bar.saving)
        self.assertEqual(disabled, [1, 0])
        self.assertIs(self.bar.active, self.editor)

    def test_extra_controls_rejected_with_parts(self):
        with self.assertRaises(ValueError):
            make().track('X', {Control(): ''}, parts=[{'controls': {}, 'operation': None}])

    def test_retrack_unsupported_with_parts(self):
        with self.assertRaises(NotImplementedError):
            self.bar.retrack(self.editor, {})

    def test_save_and_parts_are_exclusive(self):
        with self.assertRaises(ValueError):
            make().track('X', {}, save=lambda: None, parts=[{'controls': {}, 'operation': None}])


class SaveBarStaleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bar, self.field, self.calls = make(), Control('a'), []
        self.outcome = (True, 'done')

        async def attempt(operation, action, detail):
            return self.outcome
        self.bar.ctx._attempt = attempt
        self.bar.ctx.selector.value = 'tab'

        async def refresh(tab):
            self.calls.append(('refresh', tab))
        self.bar.ctx.refresh = refresh
        self.editor = self.bar.track('Alice', {self.field: 'a'}, save=lambda: None)
        self.field.set_value('b')

    async def test_conflict_marks_stale_and_plain_error_does_not(self):
        self.outcome = AttemptFailed((False, None), ValueError('bad'))
        await self.bar.save()
        self.assertFalse(self.editor.stale)
        self.outcome = AttemptFailed((False, None), ConflictError('changed'))
        await self.bar.save()
        self.assertTrue(self.editor.stale)
        self.assertEqual(self.bar.reset_button.label, 'Reload')
        self.assertEqual(self.bar.text.label, 'Alice was changed somewhere else.')

    async def test_stale_blocks_settle_in_check(self):
        self.editor.stale = True
        self.field.set_value('a')
        self.assertIs(self.bar.active, self.editor)

    async def test_reload_clears_then_uses_given_reload_or_default_refresh(self):
        self.editor.reload = None
        await self.bar.reload()
        self.assertEqual(self.calls, [('refresh', 'tab')])
        self.assertEqual((self.bar.editors, self.bar.active), ([], None))
        other = Control('b')
        editor = self.bar.track('Bob', {other: 'b'}, save=lambda: None, reload=lambda: self._reloaded())
        other.set_value('c')
        self.assertIs(self.bar.active, editor)
        await self.bar.reload()
        self.assertEqual(self.calls[-1], 'custom')
        self.assertEqual(self.bar.editors, [])

    async def _reloaded(self):
        self.calls.append('custom')

    async def test_success_clears_stale(self):
        self.outcome = AttemptFailed((False, None), ConflictError('changed'))
        await self.bar.save()
        self.outcome = (True, 'done')
        await self.bar.save()
        self.assertFalse(self.editor.stale)
        self.assertIsNone(self.bar.active)
        self.assertEqual(self.bar.reset_button.label, 'Reset')

    async def test_success_edited_during_save_clears_stale_and_restores_labels(self):
        bar, a, b = make(), Control('a'), Control('b')
        editor = bar.track('Server', {}, parts=[{'controls': {a: 'a'}, 'operation': 'x'}, {'controls': {b: 'b'}, 'operation': 'y'}])
        a.set_value('a2')
        editor.stale = True
        bar._mark_stale(editor)

        async def attempt(operation, action, detail):
            a.value = 'a3'
            return True, 'r'
        bar.ctx._attempt = attempt
        await bar.save()
        self.assertIs(bar.active, editor)
        self.assertFalse(editor.stale)
        self.assertEqual(bar.reset_button.label, 'Reset')
        self.assertEqual(bar.text.label, 'Server has unsaved changes.')

    async def test_reload_and_reset_ignored_while_saving(self):
        self.editor.stale = True
        self.bar.saving = True
        await self.bar.reload()
        self.bar.reset()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.field.value, 'b')
        self.assertIs(self.bar.active, self.editor)
        self.assertEqual(self.bar.editors, [self.editor])


if __name__ == '__main__':
    unittest.main()
