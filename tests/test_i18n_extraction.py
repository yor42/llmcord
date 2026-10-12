"""FEAT-46 unextracted messages: every literal marker call in llmcord_core reaches the .pot templates."""
import ast
import unittest
from pathlib import Path

from babel.messages.pofile import read_po

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "llmcord_core"
TEMPLATES = PACKAGE / "locales" / "templates"

# Marker -> (index of context arg or None, index of first msgid arg, number of msgid args).
MODULE_MARKERS = {"N_": (None, 0, 1), "C_": (None, 0, 1), "NP_": (0, 1, 1), "E_": (None, 1, 1)}
METHOD_MARKERS = {"n": (None, 0, 2), "p": (0, 1, 1), "np": (0, 1, 2)}
HINT = ("a nested marker call (t(...), t.render(...), t.p(...)) in the arguments hides the outer "
        "message from Babel; compute the inner values into locals first")


def _literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _marker_spec(call):
    func = call.func
    if isinstance(func, ast.Name):
        if func.id == "t":
            return (None, 0, 1)
        return MODULE_MARKERS.get(func.id)
    if isinstance(func, ast.Attribute):  # any receiver (t, ct, ...): Babel matches the method name only
        return METHOD_MARKERS.get(func.attr)
    return None


def scan_calls():
    """Yield (file:line, context, msgid, is_context_call) for each marker call with literal strings."""
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            spec = _marker_spec(node)
            if spec is None:
                continue
            ctx_i, first, count = spec
            args = node.args
            if len(args) < first + count:
                continue
            context = None
            if ctx_i is not None:
                context = _literal(args[ctx_i])
                if context is None:
                    continue
            ids = [_literal(a) for a in args[first:first + count]]
            if any(i is None or not i for i in ids):
                continue
            yield f"{rel}:{node.lineno}", context, ids[0], ctx_i is not None


def template_entries():
    """Return {(context, msgid)} across all templates."""
    found = set()
    for pot in sorted(TEMPLATES.glob("*.pot")):
        with open(pot, "rb") as fh:
            catalog = read_po(fh)
        for message in catalog:
            singular = message.id[0] if isinstance(message.id, tuple) else message.id
            if singular:
                found.add((message.context, singular))
    return found


class TemplateExtractionTests(unittest.TestCase):
    def test_every_literal_marker_call_is_in_templates(self):
        """FEAT-46 unextracted messages: each (context, msgid) of a literal marker call is in a template."""
        present = template_entries()
        calls = list(scan_calls())
        self.assertGreater(len(calls), 500)
        missing = [f"{where}  {context!r}  {msgid!r}" for where, context, msgid, _ in calls
                   if (context, msgid) not in present]
        self.assertEqual(missing, [], "messages missing from templates; " + HINT + ":\n" + "\n".join(missing))

    def test_no_template_msgid_equals_a_context_string(self):
        """FEAT-46 unextracted messages: a t.p/NP_ context never appears as a context-less msgid."""
        contexts = {context for _, context, _, is_ctx in scan_calls() if is_ctx}
        plain = {msgid for context, msgid in template_entries() if context is None}
        bogus = sorted(contexts & plain)
        self.assertEqual(bogus, [], "contexts extracted as plain msgids; " + HINT)


if __name__ == "__main__":
    unittest.main()
