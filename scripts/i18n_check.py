"""Catalog, marker and string-ratchet checks for the i18n tooling (see scripts/i18n.py)."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import string

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = 'llmcord_core'
LOCALES = f'{PACKAGE}/locales'
RATCHET = 'scripts/i18n_ratchet.json'

# 0-based positions of the msgid arguments of each marker call.
MARKERS = {'t': (0,), 'n': (0, 1), 'p': (1,), 'np': (1, 2), 'N_': (0,), 'NP_': (1,), 'C_': (0,), 'E_': (1,)}
ALLOWED = {'bot': {'t', 'n', 'p', 'np', 'N_', 'NP_'}, 'web': {'t', 'n', 'p', 'np', 'N_', 'NP_'},
           'common': {'N_', 'NP_'}, 'errors': {'E_'}}
MARKER_DOMAINS = tuple(ALLOWED)
LIMITS = (('button', 80), ('select', 100), ('modal', 45))

UI_CALLS = frozenset((
    'label button notify tab input select textarea number switch checkbox tooltip menu_item expansion link markdown '
    'upload radio toggle section lucide_button confirm_dialog more_menu input_chips chip badge send_message send '
    'edit_message edit_original_response reply Embed add_field set_footer set_author Button SelectOption TextInput '
    'Modal Group command describe Choice').split())
UI_KEYWORDS = frozenset(('text label placeholder hint title message success content description name value caption '
                         'tooltip footer prefix suffix header').split())
RAISE_TYPES = frozenset(('ValueError RuntimeError HTTPException PermissionError LookupError KeyError TypeError '
                         'ConflictError RecipeError ModelConfigError AppCommandError CheckFailure Exception').split())
IDENTIFIER_LIKE = re.compile(r'^[a-z0-9_.:/#-]+$')
TEXTY = re.compile(r'[A-Za-z가-힣ㄱ-ㅣ]')


def _tool():
    from scripts import i18n as tool
    return tool


def _read(path: Path):
    from babel.messages.pofile import read_po
    with open(path, 'rb') as f:
        return read_po(f, abort_invalid=True)


def _ids(message) -> tuple[str, ...]:
    return tuple(message.id) if isinstance(message.id, (tuple, list)) else (message.id,)


def _strings(message) -> tuple[str, ...]:
    return tuple(message.string) if isinstance(message.string, (tuple, list)) else (message.string,)


def _key(message) -> tuple:
    return (message.context or None, _ids(message)[0])


def _messages(catalog):
    return [m for m in catalog if m.id]


def _short(text: str) -> str:
    return repr(text if len(text) <= 50 else text[:47] + '...')


def _specs(template: str) -> list[str]:
    try:
        return [spec for _, name, spec, _ in string.Formatter().parse(template) if name is not None and spec]
    except ValueError:
        return []


def _check_text(where: str, msgid: str, text: str, is_id: bool, errors: list[str]) -> bool:
    """Syntax and josa checks for one string; returns whether its fields could be read."""
    from llmcord_core.i18n.format import MessageFormatError, fields, is_josa_spec
    for spec in _specs(text):
        if '/' in spec and not is_josa_spec(spec):
            errors.append(f'[josa] {where}: format spec {spec!r} is not a josa spec in {_short(msgid)}')
        elif is_id and is_josa_spec(spec):
            errors.append(f'[josa] {where}: josa spec {spec!r} in an English msgid {_short(msgid)}')
    try:
        fields(text)
    except MessageFormatError as error:
        errors.append(f'[field-syntax] {where}: {error} in {_short(text)}')
        return False
    return True


def _limits(where: str, message, strings, domain: str, errors: list[str]) -> None:
    caps = [cap for prefix, cap in LIMITS if (message.context or '').startswith(prefix)]
    if domain == 'commands':
        caps.append(100)
    if not caps:
        return
    for text in strings:
        if text and len(text) > min(caps):
            errors.append(f'[limit] {where}: {len(text)} characters (max {min(caps)}) in {_short(text)}')


def _check_catalogs(root: Path, tool, errors: list[str], warnings: list[str]) -> None:
    from babel.messages.plurals import get_plural
    from llmcord_core.i18n.format import MessageFormatError, fields
    locales = root / LOCALES
    template_keys: dict[str, set] = {}
    for domain in tool.DOMAINS:
        path = locales / 'templates' / f'{domain}.pot'
        keys: set = set()
        if path.exists():
            try:
                catalog = _read(path)
            except Exception as error:
                errors.append(f'[po-parse] {path.relative_to(root).as_posix()}: {error}')
                template_keys[domain] = keys
                continue
            for message in _messages(catalog):
                keys.add(_key(message))
                where = f'templates/{domain}'
                for text in _ids(message):
                    _check_text(where, message.id[0] if isinstance(message.id, tuple) else message.id, text, True, errors)
                _limits(where, message, _ids(message), domain, errors)
        template_keys[domain] = keys
    if not locales.is_dir():
        return
    for locale_dir in sorted(p for p in locales.iterdir() if p.is_dir() and p.name != 'templates'):
        locale = locale_dir.name
        seen: dict[tuple, tuple[str, tuple]] = {}
        for domain in tool.DOMAINS:
            path = locale_dir / 'LC_MESSAGES' / f'{domain}.po'
            if not path.exists():
                continue
            where = f'{locale}/{domain}'
            try:
                catalog = _read(path)
            except Exception as error:
                errors.append(f'[po-parse] {path.relative_to(root).as_posix()}: {error}')
                continue
            try:
                expected = get_plural(locale).num_plurals
            except Exception:
                expected = None
            if expected is None or catalog.num_plurals != expected:
                errors.append(f'[plural-forms] {where}: nplurals {catalog.num_plurals}, expected {expected}')
            language = catalog.mime_headers and dict(catalog.mime_headers).get('Language')
            if language != locale:
                errors.append(f'[plural-forms] {where}: Language header {language!r} differs from the directory')
            if catalog.obsolete:
                errors.append(f'[obsolete] {where}: {len(catalog.obsolete)} obsolete entries')
            messages = _messages(catalog)
            if {_key(m) for m in messages} != template_keys[domain]:
                errors.append(f'[stale-catalog] {where}: keys differ from the template; run python scripts/i18n.py update')
            fuzzy = sum(1 for m in messages if m.fuzzy)
            if fuzzy:
                warnings.append(f'[fuzzy] {where}: {fuzzy} fuzzy entries')
            for message in messages:
                strings = _strings(message)
                if message.fuzzy or not any(strings):
                    continue
                ids = _ids(message)
                msgid = ids[0]
                _limits(where, message, strings, domain, errors)
                id_fields: set[str] | None = set()
                try:
                    for text in ids:
                        id_fields |= fields(text)
                except MessageFormatError:
                    id_fields = None
                ok = True
                for text in strings:
                    if text:
                        ok = _check_text(where, msgid, text, False, errors) and ok
                if not ok or id_fields is None:
                    continue
                if len(ids) == 1:
                    if fields(strings[0]) != id_fields:
                        errors.append(f'[placeholder] {where}: fields differ from msgid {_short(msgid)}')
                else:
                    extra = [t for t in strings if t and not fields(t) <= id_fields | {'count'}]
                    if extra:
                        errors.append(f'[placeholder] {where}: unknown field in a plural form of {_short(msgid)}')
                if all(strings):
                    key = _key(message)
                    if key in seen and seen[key][1] != strings:
                        errors.append(f'[conflict] {locale}: {_short(msgid)} translated differently in '
                                      f'{seen[key][0]} and {domain}')
                    seen.setdefault(key, (domain, strings))


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None


def _is_literal(node) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _non_literal(node) -> bool:
    return (isinstance(node, ast.JoinedStr)
            or (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)))
            or (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'format'))


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in (root / PACKAGE).rglob('*.py') if '__pycache__' not in p.parts and 'locales' not in p.parts)


def _parse(path: Path):
    try:
        return ast.parse(path.read_text(encoding='utf-8'))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None


def _check_markers(root: Path, tool, errors: list[str]) -> None:
    mapped: dict[str, set[str]] = {}
    for domain in MARKER_DOMAINS:
        for path in tool.domain_files(root, domain):
            mapped.setdefault(path.relative_to(root).as_posix(), set()).add(domain)
    for rel, domains in sorted(mapped.items()):
        if len(domains) > 1:
            errors.append(f'[unmapped] {rel}: matched by several domains ({", ".join(sorted(domains))})')
    for path in _python_files(root):
        rel = path.relative_to(root).as_posix()
        tree = _parse(path)
        if tree is None:
            continue
        domains = mapped.get(rel, set())
        allowed = set().union(*(ALLOWED[d] for d in domains)) if domains else set()
        in_i18n = rel.startswith(f'{PACKAGE}/i18n/')
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in MARKERS and not in_i18n:
                errors.append(f'[name-collision] {rel}:{node.lineno}: {node.name} is a reserved marker name')
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name not in MARKERS:
                continue
            for position in MARKERS[name]:
                if position >= len(node.args):
                    continue
                arg = node.args[position]
                if _non_literal(arg):
                    errors.append(f'[non-literal] {rel}:{node.lineno}: {name}() msgid must be a plain string literal')
                elif _is_literal(arg) and name != 'C_' and name not in allowed and not in_i18n:
                    errors.append(f'[unmapped] {rel}:{node.lineno}: {name}() is not allowed in this file\'s domain')


def _bare(node) -> int:
    """Number of bare text literals in an argument (str constants, f-strings, branches, + operands)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value
    elif isinstance(node, ast.JoinedStr):
        text = ''.join(v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
    elif isinstance(node, ast.IfExp):
        return _bare(node.body) + _bare(node.orelse)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _bare(node.left) + _bare(node.right)
    else:
        return 0
    return int(bool(TEXTY.search(text)) and not IDENTIFIER_LIKE.match(text))


def _ui_count(tree) -> int:
    total = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) not in UI_CALLS:
            continue
        total += sum(_bare(a) for a in node.args)
        every = _call_name(node) == 'describe'
        total += sum(_bare(k.value) for k in node.keywords if k.arg and (every or k.arg in UI_KEYWORDS))
    return total


def _raise_count(tree) -> int:
    total = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        call = node.exc
        if _call_name(call) not in RAISE_TYPES:
            continue
        arg = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == 'detail'), None)
        total += bool(arg is not None and _bare(arg))
    return total


def ratchet_counts(root: Path | None = None) -> dict[str, dict[str, int]]:
    root = Path(root or ROOT)
    tool = _tool()
    ui_files = {p for d in ('bot', 'web') for p in tool.domain_files(root, d)}
    counts: dict[str, dict[str, int]] = {}
    for path in _python_files(root):
        rel = path.relative_to(root).as_posix()
        tree = _parse(path)
        if tree is None:
            continue
        entry = {}
        if path in ui_files and (n := _ui_count(tree)):
            entry['ui'] = n
        if not rel.startswith(f'{PACKAGE}/i18n/') and (n := _raise_count(tree)):
            entry['raise'] = n
        if entry:
            counts[rel] = entry
    return counts


def write_ratchet(root: Path | None = None) -> Path:
    root = Path(root or ROOT)
    path = root / RATCHET
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'version': 1, 'files': ratchet_counts(root)}, sort_keys=True, indent=2) + '\n',
                    encoding='utf-8')
    return path


def check_ratchet(root: Path | None = None) -> list[str]:
    root = Path(root or ROOT)
    path = root / RATCHET
    try:
        baseline = json.loads(path.read_text(encoding='utf-8')).get('files', {})
    except (OSError, ValueError, AttributeError):
        return [f'[ratchet] {RATCHET} is missing or unreadable; run python scripts/i18n.py ratchet --update']
    current = ratchet_counts(root)
    errors = []
    for rel in sorted(set(baseline) | set(current)):
        for kind in ('ui', 'raise'):
            was, now = baseline.get(rel, {}).get(kind, 0), current.get(rel, {}).get(kind, 0)
            if now > was:
                errors.append(f'[ratchet] {rel}: {kind} literals {was} -> {now}; wrap new strings in t(...) or raise UserError')
            elif now < was:
                errors.append(f'[ratchet] {rel}: {kind} literals {was} -> {now}; '
                              'run python scripts/i18n.py ratchet --update to lower the baseline')
    return errors


def check(root: Path | None = None) -> tuple[list[str], list[str]]:
    root = Path(root or ROOT)
    tool = _tool()
    errors: list[str] = []
    warnings: list[str] = []
    for domain in tool.DOMAINS:
        expected = tool.render_template(tool.extract_domain(root, domain))
        path = root / LOCALES / 'templates' / f'{domain}.pot'
        actual = path.read_text(encoding='utf-8') if path.exists() else ''
        if actual != expected:
            errors.append(f'[stale-template] templates/{domain}.pot is out of date; run python scripts/i18n.py extract')
    _check_catalogs(root, tool, errors, warnings)
    _check_markers(root, tool, errors)
    errors += check_ratchet(root)
    return errors, warnings
