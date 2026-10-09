"""Offline tests for SaveBar's dirty/reset/retrack/label logic (stub controls; the browser suite covers the UI)."""
import types
import unittest

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


def make():
    bar = object.__new__(SaveBar)
    widget = Widget()
    client = types.SimpleNamespace(run_javascript=lambda _script: None)
    bar.ctx = types.SimpleNamespace(selector=types.SimpleNamespace(client=client))
    bar.editors, bar.active, bar.saving = [], None, False
    bar.bar, bar.text, bar.save_button = widget, Widget(), Widget()
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

    def test_save_label_follows_active_editor(self):
        bar, field = make(), Control('a')
        bar.track('P', {field: 'a'}, save=None, save_label='Save draft')
        field.set_value('b')
        self.assertEqual(bar.save_button.label, 'Save draft')
        bar.settle()
        self.assertEqual(bar.save_button.label, 'Save changes')


if __name__ == '__main__':
    unittest.main()
