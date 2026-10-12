"""Extract, merge and inspect gettext catalogs (FEAT-43, D39).

    python scripts/i18n.py extract | update | init LANG | check | stats | coverage [--min PCT] [--locale LANG ...] | ratchet [--update [--allow-increase]]
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmcord_core.i18n.negotiate import normalize  # noqa: E402

DOMAINS = ('bot', 'commands', 'web', 'errors', 'common')
PACKAGE = 'llmcord_core'
LOCALES = 'llmcord_core/locales'
KEYWORDS = {
    'bot': {'t': None, 'n': (1, 2), 'p': ((1, 'c'), 2), 'np': ((1, 'c'), 2, 3),
            'N_': None, 'NP_': ((1, 'c'), 2)},
    'common': {'N_': None, 'NP_': ((1, 'c'), 2)},
    'errors': {'E_': (2,)},
    'commands': {'C_': None},
}
KEYWORDS['web'] = KEYWORDS['bot']
DOMAIN_FILES = {
    'bot': ['llmcord_core/discord_bot.py', 'llmcord_core/*_discord.py',
            'llmcord_core/games_spoken_doubt.py', 'llmcord_core/usage.py', 'llmcord_core/catchup.py'],
    'web': ['llmcord_core/dashboard.py', 'llmcord_core/*_ui.py',
            'llmcord_core/lore_workspace.py', 'llmcord_core/lore_drag.py',
            'llmcord_core/macro_highlight.py', 'llmcord_core/savebar.py',
            'llmcord_core/icons.py', 'llmcord_core/web.py', 'llmcord_core/auth.py',
            'llmcord_core/admin.py'],
    'common': ['llmcord_core/i18n/terms.py'],
    'errors': ['llmcord_core/i18n/codes.py'],
    'commands': ['llmcord_core/**/*.py'],
}
COMMENT_TAGS = ('i18n:',)
_RESERVED = {'templates', 'pseudo', 'en'}


def domain_files(root, domain) -> list[Path]:
    root = Path(root)
    locales = root / LOCALES
    found = set()
    for pattern in DOMAIN_FILES[domain]:
        found.update(p for p in root.glob(pattern) if p.is_file() and locales not in p.parents)
    return sorted(found)


def extract_domain(root, domain):
    from babel.messages.catalog import Catalog
    from babel.messages.extract import extract_from_file
    root = Path(root)
    catalog = Catalog()
    for path in domain_files(root, domain):
        rel = path.relative_to(root).as_posix()
        for _line, message, comments, context in extract_from_file(
                'python', path, keywords=KEYWORDS[domain], comment_tags=COMMENT_TAGS):
            parts = message if isinstance(message, tuple) else (message,)
            if not parts or any(not isinstance(p, str) for p in parts) or not parts[0]:
                continue
            catalog.add(message, locations=[(rel, 0)], auto_comments=comments, context=context)
    return catalog


def render_template(catalog) -> str:
    from babel.messages.pofile import write_po
    out = io.BytesIO()
    write_po(out, catalog, width=0, omit_header=True, include_lineno=False)
    return out.getvalue().decode('utf-8')


def _write(path: Path, text: str) -> bool:
    if path.exists() and path.read_text(encoding='utf-8') == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')
    return True


def write_templates(root) -> list[Path]:
    root = Path(root)
    changed = []
    for domain in DOMAINS:
        path = root / LOCALES / 'templates' / f'{domain}.pot'
        if _write(path, render_template(extract_domain(root, domain))):
            changed.append(path)
    return changed


def _read(path: Path, locale=None):
    from babel.messages.pofile import read_po
    with path.open('rb') as handle:
        return read_po(handle, locale=locale, abort_invalid=True)


def _plural_forms(lang: str) -> str:
    from babel.core import Locale
    from babel.messages.plurals import get_plural
    plural = get_plural(Locale.parse(lang))
    return f'nplurals={plural.num_plurals}; plural={plural.plural_expr};'


def _render_po(catalog, lang: str) -> str:
    header = ('msgid ""\nmsgstr ""\n'
              f'"Language: {lang}\\n"\n'
              '"MIME-Version: 1.0\\n"\n'
              '"Content-Type: text/plain; charset=UTF-8\\n"\n'
              '"Content-Transfer-Encoding: 8bit\\n"\n'
              f'"Plural-Forms: {_plural_forms(lang)}\\n"\n')
    body = render_template_po(catalog)
    return header + ('\n' + body if body else '')


def render_template_po(catalog) -> str:
    from babel.messages.pofile import write_po
    out = io.BytesIO()
    write_po(out, catalog, width=0, omit_header=True, include_lineno=False,
             ignore_obsolete=True)
    return out.getvalue().decode('utf-8')


def _sync(root: Path, lang: str, domain: str) -> bool:
    from babel.messages.catalog import Catalog
    template = _read(root / LOCALES / 'templates' / f'{domain}.pot')
    path = root / LOCALES / lang / 'LC_MESSAGES' / f'{domain}.po'
    catalog = _read(path, lang) if path.exists() else Catalog(locale=lang)
    catalog.update(template)
    for message in catalog:  # fuzzy on an untranslated entry is noise; keep it where a translation was carried over
        strings = message.string if isinstance(message.string, (tuple, list)) else (message.string,)
        if message.id and not any(strings):
            message.flags.discard('fuzzy')
    catalog.obsolete.clear()
    return _write(path, _render_po(catalog, lang))


def _locale_dirs(root: Path) -> list[str]:
    base = root / LOCALES
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir()
                  if p.is_dir() and p.name not in _RESERVED and normalize(p.name) == p.name)


def update_catalogs(root) -> list[Path]:
    root = Path(root)
    changed = []
    for lang in _locale_dirs(root):
        for domain in DOMAINS:
            if _sync(root, lang, domain):
                changed.append(root / LOCALES / lang / 'LC_MESSAGES' / f'{domain}.po')
    return changed


def init_locale(root, lang) -> list[Path]:
    from babel.core import Locale
    root = Path(root)
    if lang in _RESERVED or normalize(lang) != lang:
        raise ValueError(f'cannot create locale {lang!r}: use a normalized tag such as ko or zh_TW')
    try:
        Locale.parse(lang)
    except Exception as exc:
        raise ValueError(f'Babel does not know locale {lang!r}') from exc
    if (root / LOCALES / lang).exists():
        raise ValueError(f'locale {lang!r} already exists')
    for domain in DOMAINS:
        if not (root / LOCALES / 'templates' / f'{domain}.pot').is_file():
            raise ValueError('templates missing: run python scripts/i18n.py extract')
    for domain in DOMAINS:
        _sync(root, lang, domain)
    return [root / LOCALES / lang / 'LC_MESSAGES' / f'{d}.po' for d in DOMAINS]


def stats(root) -> list[dict]:
    root = Path(root)
    rows = []
    for lang in _locale_dirs(root):
        for domain in DOMAINS:
            path = root / LOCALES / lang / 'LC_MESSAGES' / f'{domain}.po'
            if not path.exists():
                continue
            total = translated = fuzzy = 0
            for message in _read(path, lang):
                if not message.id:
                    continue
                total += 1
                strings = message.string if isinstance(message.string, (list, tuple)) else [message.string]
                if message.fuzzy:
                    fuzzy += 1
                elif all(strings):
                    translated += 1
            rows.append({'locale': lang, 'domain': domain, 'total': total,
                         'translated': translated, 'fuzzy': fuzzy,
                         'untranslated': total - translated - fuzzy})
    return rows


def _pct(translated: int, total: int) -> str:
    return f'{100.0 * translated / total if total else 100.0:.1f}%'


def coverage_report(root, locales=None, minimum=None) -> tuple[list[str], list[str]]:
    """Report lines and the locale/domain pairs below `minimum` percent."""
    rows = stats(root)
    if locales:
        rows = [r for r in rows if r['locale'] in locales]
    lines, low = [], []
    for lang in sorted({r['locale'] for r in rows}):
        mine = [r for r in rows if r['locale'] == lang]
        total = sum(r['total'] for r in mine)
        done = sum(r['translated'] for r in mine)
        lines.append(f'{lang}: {done}/{total} {_pct(done, total)} fuzzy={sum(r["fuzzy"] for r in mine)} '
                     f'untranslated={sum(r["untranslated"] for r in mine)}')
        for r in mine:
            lines.append(f'  {lang}/{r["domain"]}: {r["translated"]}/{r["total"]} {_pct(r["translated"], r["total"])} '
                         f'fuzzy={r["fuzzy"]} untranslated={r["untranslated"]}')
            if minimum is not None and r['total'] and 100.0 * r['translated'] / r['total'] < minimum:
                low.append(f'{lang}/{r["domain"]}: {_pct(r["translated"], r["total"])} is below {minimum:g}%')
    return lines, low


def main(argv=None, root=None) -> int:
    root = Path(root) if root else ROOT
    parser = argparse.ArgumentParser(prog='i18n.py', description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('extract', 'update', 'check', 'stats'):
        sub.add_parser(name)
    cov = sub.add_parser('coverage')
    cov.add_argument('--min', type=float, default=None)
    cov.add_argument('--locale', nargs='+', action='extend', default=None)
    sub.add_parser('init').add_argument('lang')
    ratchet = sub.add_parser('ratchet')
    ratchet.add_argument('--update', action='store_true')
    ratchet.add_argument('--allow-increase', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'extract':
        changed = write_templates(root)
        print(f'templates: {len(changed)} changed')
    elif args.command == 'update':
        changed = update_catalogs(root)
        print(f'catalogs: {len(changed)} changed')
    elif args.command == 'init':
        try:
            made = init_locale(root, args.lang)
        except ValueError as exc:
            print(f'error: {exc}', file=sys.stderr)
            return 1
        print(f'created {len(made)} catalogs for {args.lang}')
    elif args.command == 'stats':
        for row in stats(root):
            print('{locale:8} {domain:9} total={total} translated={translated} '
                  'fuzzy={fuzzy} untranslated={untranslated}'.format(**row))
    elif args.command == 'coverage':
        lines, low = coverage_report(root, args.locale, args.min)
        print(*lines, sep='\n')
        if low:
            print(*(f'below minimum: {item}' for item in low), sep='\n')
            return 1
    elif args.command == 'ratchet' and args.update:
        from scripts import i18n_check
        raised = [] if args.allow_increase else i18n_check.ratchet_increases(root)
        if raised:
            print(*raised, sep='\n')
            print('refusing to raise the baseline; wrap the new strings, or pass --allow-increase', file=sys.stderr)
            return 1
        print(f'wrote {i18n_check.write_ratchet(root)}')
    else:
        from scripts import i18n_check
        errors, warnings = i18n_check.check(root)
        if args.command == 'ratchet':
            errors = [e for e in errors if e.startswith('[ratchet]')]
            warnings = []
        for line in (*errors, *warnings):
            print(line)
        if errors:
            return 1
        print(f'i18n {args.command}: OK ({len(warnings)} warnings)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
