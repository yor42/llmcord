"""Build the GitHub wiki pages from docs/ and check that every link and section survives.

  python scripts/build_wiki.py --check          validate only
  python scripts/build_wiki.py --out DIR        write the pages into an empty or new DIR

docs/*.md are split at "## " headings into the pages named in PAGES (page names are
live wiki URLs: keep them stable), links are rewritten to wiki pages or GitHub URLs,
and docs/wiki/*.md (home page, sidebar, footer, hand-written pages) are copied as is.

Only inline links "[text](target)" are rewritten and validated. Links with titles
("[t](x.md "title")"), reference-style links and URLs containing ")" are left untouched.
"""
from __future__ import annotations

import argparse
import posixpath
import re
import sys
from pathlib import Path

REPO_URL = 'https://github.com/yor42/llmcord'
BRANCH = 'main'

# page -> [(doc file, section)]. "__intro__" is the text before the first "## ";
# "__all__" is the whole file (intro and every section).
PAGES = {
    'Installation': [('getting-started.md', '__intro__'), ('getting-started.md', 'Requirements'),
                     ('getting-started.md', 'Install and configure')],
    'First-Server-Setup': [('getting-started.md', 'First server setup')],
    'Configuration': [('getting-started.md', 'Configuration reference')],
    'Docker-on-Raspberry-Pi': [('deployment-raspberry-pi.md', s) for s in (
        '__intro__', 'Before you start', 'Configure the repository', 'Set up Discord OAuth',
        'Start and check services', 'Backups and upgrades')],
    'Native-Pi-Hosting': [('live-testing-pi.md', '__all__')],
    'Troubleshooting': [('deployment-raspberry-pi.md', 'Common problems')],
    'Releases-and-CI': [('github-ci.md', '__all__')],
    'Worlds-Hubs-and-Casts': [('server-guide.md', 'Worlds, hubs and casts')],
    'Scenes-and-Ambient-Mode': [('server-guide.md', 'Starting and continuing scenes'),
                                ('server-guide.md', 'Conversation participants')],
    'Discord-Commands': [('server-guide.md', 'Discord commands')],
    'Favorites': [('server-guide.md', 'Favorites')],
    'Language': [('server-guide.md', 'Language')],
    'Lore-and-Memory': [('lore-and-memory.md', '__all__')],
    'Items-and-Shop': [('server-guide.md', 'Items and the shop')],
    'Blackjack': [('server-guide.md', 'Blackjack')],
    'I-Doubt-It': [('server-guide.md', 'I Doubt It')],
    'Texas-Holdem': [('server-guide.md', "Texas Hold'em")],
    'One-Card': [('server-guide.md', 'One Card')],
    'Dashboard-Overview': [('admin-console.md', '__intro__'), ('server-guide.md', 'Admin dashboard'),
                           ('admin-console.md', 'Server setup and characters'), ('admin-console.md', 'Language')],
    'Dashboard-Lore-and-Imports': [('admin-console.md', 'Lore workspace'), ('admin-console.md', 'Imports')],
    'Prompt-Presets': [('admin-console.md', 'Server prompt presets')],
    'Emotion-Avatars': [('admin-console.md', 'Emotion avatars')],
    'Monitoring-and-Costs': [('admin-console.md', 'Monitoring'),
                             ('admin-console.md', 'Reply footer and cost estimates')],
    'Dashboard-Currency-and-Games': [('admin-console.md', 'Currency'), ('admin-console.md', 'Games')],
    'Bot-Settings': [('admin-console.md', 'Bot settings and spending caps'),
                     ('admin-console.md', 'Model profiles (Backend tab)')],
    'Architecture': [('architecture.md', '__all__'), ('admin-console.md', 'Deployment and security')],
    'Testing-and-Verification': [('verification.md', '__all__')],
    'Glossary': [('glossary.md', '__all__')],
}

NOT_MAPPED = {'README.md'}
LINK = re.compile(r'\]\(([^)\s]+)\)')
SCHEME = re.compile(r'^[a-z][a-z0-9+.-]*:', re.I)
HEADING = re.compile(r'^(#{1,6}) +(.+?)(?:\s+#+)? *$')
FENCE = re.compile(r'^\s*(`{3,}|~{3,})')


class WikiError(Exception):
    def __init__(self, errors):
        super().__init__('\n'.join(errors))
        self.errors = errors


def slug(heading: str) -> str:
    h = re.sub(r'[^\w\- ]', '', heading.strip().lower())
    return h.replace(' ', '-')


def outside_fences(text: str):
    """Yield (line, in_fence) for each line; fence marker lines count as in a fence."""
    marker = None
    for line in text.splitlines():
        m = FENCE.match(line)
        if marker is None:
            if m:
                marker = m.group(1)
            yield line, marker is not None
        else:
            if m and m.group(1)[0] == marker[0] and len(m.group(1)) >= len(marker) \
                    and not line.strip().strip(marker[0]):
                marker = None
            yield line, True


def split(text: str):
    """Return (intro without the H1, [(h2, body)]); fenced lines never start a section."""
    intro, sections, cur = [], [], None
    for line, fenced in outside_fences(text):
        m = None if fenced else HEADING.match(line)
        if m and len(m.group(1)) == 2:
            cur = (m.group(2).strip(), [])
            sections.append(cur)
        elif cur is not None:
            cur[1].append(line)
        elif fenced or not line.startswith('# '):
            intro.append(line)
    return '\n'.join(intro).strip(), [(h, '\n'.join(b).strip()) for h, b in sections]


def headings(text: str):
    return [m.group(2) for line, fenced in outside_fences(text)
            if not fenced and (m := HEADING.match(line))]


def page_slugs(text: str) -> set[str]:
    seen, out = {}, set()
    for h in headings(text):
        s = slug(h)
        n = seen.get(s, 0)
        seen[s] = n + 1
        out.add(s if n == 0 else f'{s}-{n}')
    return out


def sub_link(text: str, fix) -> str:
    return '\n'.join(line if fenced else LINK.sub(fix, line) for line, fenced in outside_fences(text))


def build(root: Path, pages: dict = PAGES, repo_url: str = REPO_URL) -> dict[str, str]:
    """Return {page name: markdown} for the repo at root; raise WikiError listing every problem."""
    root = Path(root)
    docs = root / 'docs'
    errors: list[str] = []
    repo_url = repo_url.rstrip('/')

    on_disk = {p.name for p in docs.glob('*.md')}
    mapped = {name for parts in pages.values() for name, _ in parts}
    for name in sorted(on_disk - mapped - NOT_MAPPED):
        errors.append(f'docs/{name} is not mapped to any wiki page (add it to PAGES in scripts/build_wiki.py)')
    parsed = {}
    for name in sorted(mapped & on_disk):
        parsed[name] = split((docs / name).read_text(encoding='utf-8'))
    for name in sorted(mapped - on_disk):
        errors.append(f'docs/{name} is mapped in PAGES but does not exist')

    used = {(n, s) for parts in pages.values() for n, s in parts}
    for name, (_, sections) in parsed.items():
        if (name, '__all__') in used:
            continue
        for h, _ in sections:
            if (name, h) not in used:
                errors.append(f'docs/{name}: section "{h}" is not used by any wiki page')

    # Pass 1: page bodies (unrewritten), and where each doc section/heading lands.
    anchors: dict[tuple[str, str], str] = {}
    claimed: dict[str, tuple[str, str]] = {}

    def register(name, s, target, strict=True):
        old = anchors.get((name, s))
        if old is not None and old != target:
            if strict:
                errors.append(f'docs/{name}: heading anchor "{s}" is ambiguous ({old} and {target})')
            return
        anchors[(name, s)] = target
        if '#' in target and claimed.setdefault(target, (name, s)) != (name, s):
            errors.append(f'wiki page {target.split("#")[0]}: anchor "{s}" comes from two different docs')

    filepage: dict[str, str] = {}
    raw: dict[str, list[tuple[str, str]]] = {}
    for page, parts in pages.items():
        single = len(parts) == 1 and parts[0][1] not in ('__intro__', '__all__')
        chunks = raw[page] = []
        for name, sec in parts:
            if name not in parsed:
                continue
            intro, sections = parsed[name]
            filepage.setdefault(name, page)
            if sec == '__intro__':
                filepage[name] = page
            if sec in ('__intro__', '__all__') and intro:
                chunks.append((name, intro))
            picked = sections if sec == '__all__' else [s for s in sections if s[0] == sec]
            if sec not in ('__intro__', '__all__') and not picked:
                errors.append(f'docs/{name}: section "{sec}" (page {page}) no longer exists')
            for h, body in picked:
                if single:
                    register(name, slug(h), page)
                    chunks.append((name, re.sub(r'(?m)^### ', '## ', body)))
                else:
                    register(name, slug(h), f'{page}#{slug(h)}')
                    chunks.append((name, f'## {h}\n\n{body}'))
                for sub in headings(body):
                    register(name, slug(sub), f'{page}#{slug(sub)}')

    for name, (intro, _) in parsed.items():
        if intro and (name, '__intro__') not in used and (name, '__all__') not in used:
            errors.append(f'docs/{name}: the text above the first "## " is not used by any wiki page')

    def rewrite(src: str, text: str) -> str:
        def fix(m):
            target = m.group(1)
            if SCHEME.match(target):
                return m.group(0)
            path, _, anchor = target.partition('#')
            doc = None if path else src
            if path:
                full = posixpath.normpath(posixpath.join('docs', path))
                if full.startswith('..'):
                    errors.append(f'docs/{src}: link {target} leaves the repository')
                    return m.group(0)
                if full.startswith('docs/') and full.count('/') == 1 and full[5:] in parsed:
                    doc = full[5:]
                else:
                    if not (root / full).exists():
                        errors.append(f'docs/{src}: link {target} points at {full}, which does not exist')
                        return m.group(0)
                    if full == 'README.md' and not anchor:
                        return f']({repo_url}#readme)'
                    kind = 'tree' if (root / full).is_dir() else 'blob'
                    return f']({repo_url}/{kind}/{BRANCH}/{full}' + (f'#{anchor}' if anchor else '') + ')'
            if anchor:
                dest = anchors.get((doc, anchor))
                if dest is None:
                    errors.append(f'docs/{src}: link {target} has an anchor that is not on any wiki page')
                    return m.group(0)
            else:
                dest = filepage.get(doc)
                if dest is None:
                    errors.append(f'docs/{src}: link {target} points at a doc with no wiki page')
                    return m.group(0)
            return f']({dest})'
        return sub_link(text, fix)

    out: dict[str, str] = {}
    for page, chunks in raw.items():
        out[page] = '\n\n'.join(rewrite(n, c) for n, c in chunks).rstrip() + '\n'
    for p in sorted((docs / 'wiki').glob('*.md')):
        if p.stem in out:
            errors.append(f'docs/wiki/{p.name} collides with the generated page {p.stem}')
        else:
            out[p.stem] = p.read_text(encoding='utf-8')

    known = {name: page_slugs(text) for name, text in out.items()}
    for page, text in out.items():
        def check(m, page=page):
            target = m.group(1)
            if SCHEME.match(target):
                return m.group(0)
            dest, _, anchor = target.partition('#')
            dest = dest or page
            if dest not in known:
                errors.append(f'wiki page {page}: link {target} points at a page that does not exist')
            elif anchor and anchor not in known[dest]:
                errors.append(f'wiki page {page}: link {target} points at a heading that does not exist')
            return m.group(0)
        sub_link(text, check)
    if errors:
        raise WikiError(list(dict.fromkeys(errors)))
    return out


def write(pages: dict[str, str], out: Path) -> None:
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise WikiError([f'{out} exists and is not an empty directory'])
    try:
        out.mkdir(parents=True, exist_ok=True)
        for name, text in pages.items():
            (out / f'{name}.md').write_text(text, encoding='utf-8')
    except OSError as e:
        raise WikiError([f'cannot write {out}: {e}']) from e


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='build in memory and validate only')
    mode.add_argument('--out', type=Path, help='write the pages into this empty or new directory')
    ap.add_argument('--repo-url', default=REPO_URL, help='base URL for links to repository files')
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    try:
        pages = build(root, repo_url=args.repo_url)
        if args.out:
            write(pages, args.out)
    except WikiError as e:
        print('wiki build failed:', file=sys.stderr)
        for err in e.errors:
            print(f'  - {err}', file=sys.stderr)
        return 1
    print(f'{len(pages)} wiki pages ' + (f'written to {args.out}' if args.out else 'checked'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
