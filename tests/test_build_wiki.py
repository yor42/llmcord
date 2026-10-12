import contextlib
import io
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from scripts import build_wiki

REPO = Path(__file__).resolve().parent.parent
URL = 'https://example.test/r'

GUIDE = """# Guide

Intro text.

## Alpha

Alpha body, see [beta](#beta) and [the other doc](other.md) and [more](other.md#deep-dive).

### Alpha detail

Detail.

## Beta

Beta body.
"""
OTHER = "# Other\n\n## Deep dive\n\nSee [guide alpha](guide.md#alpha-detail).\n"
PAGES = {
    'Guide-Alpha': [('guide.md', 'Alpha')],
    'Guide-Intro': [('guide.md', '__intro__')],
    'Guide-Beta': [('guide.md', 'Beta')],
    'Other': [('other.md', '__all__')],
}


def make(files, pages=PAGES, **kw):
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    base = {'docs/guide.md': GUIDE, 'docs/other.md': OTHER, 'docs/README.md': '# Docs\n'}
    base.update(files)
    for rel, text in base.items():
        if text is None:
            continue
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding='utf-8')
    return tmp, root


class BuildWikiTests(unittest.TestCase):
    def build(self, files=None, pages=PAGES):
        tmp, root = make(files or {})
        self.addCleanup(tmp.cleanup)
        return build_wiki.build(root, pages, URL)

    def fails(self, files=None, pages=PAGES):
        with self.assertRaises(build_wiki.WikiError) as cm:
            self.build(files, pages)
        return '\n'.join(cm.exception.errors)

    def test_links_rewritten_to_pages_and_anchors(self):
        out = self.build()
        self.assertIn('(Other)', out['Guide-Alpha'])
        self.assertIn('(Other#deep-dive)', out['Guide-Alpha'])
        self.assertIn('[beta](Guide-Beta)', out['Guide-Alpha'])
        self.assertIn('(Guide-Alpha#alpha-detail)', out['Other'])

    def test_single_section_page_drops_heading_and_promotes_subheadings(self):
        page = self.build()['Guide-Alpha']
        self.assertNotIn('## Alpha\n', page)
        self.assertIn('## Alpha detail', page)
        self.assertNotIn('### Alpha detail', page)

    def test_multi_section_page_keeps_headings(self):
        pages = {'Guide': [('guide.md', '__intro__'), ('guide.md', 'Alpha'), ('guide.md', 'Beta')],
                 'Other': [('other.md', '__all__')]}
        out = self.build(pages=pages)
        self.assertIn('## Alpha\n', out['Guide'])
        self.assertIn('### Alpha detail', out['Guide'])
        self.assertIn('(Guide#beta)', out['Guide'])

    def test_repo_file_and_dir_links_become_github_urls(self):
        tmp, root = make({'docs/guide.md': GUIDE + '\nSee [src](../src/mod.py), [dir](engineering/) and [top](../README.md).\n',
                          'src/mod.py': 'x = 1\n', 'docs/engineering/a.md': 'a\n', 'README.md': 'r\n'})
        self.addCleanup(tmp.cleanup)
        text = build_wiki.build(root, PAGES, URL)['Guide-Beta']
        self.assertIn(f'({URL}/blob/main/src/mod.py)', text)
        self.assertIn(f'({URL}/tree/main/docs/engineering)', text)
        self.assertIn(f'({URL}#readme)', text)

    def test_external_links_untouched(self):
        text = self.build({'docs/guide.md': GUIDE + '\n[x](https://example.com/a#b)\n'})['Guide-Beta']
        self.assertIn('(https://example.com/a#b)', text)

    def test_unknown_anchor_fails(self):
        self.assertIn('anchor', self.fails({'docs/guide.md': GUIDE + '\n[x](#nope)\n'}))

    def test_unmapped_h2_fails(self):
        msg = self.fails({'docs/guide.md': GUIDE + '\n## Gamma\n\nNew.\n'})
        self.assertIn('"Gamma"', msg)

    def test_unmapped_doc_file_fails(self):
        msg = self.fails({'docs/new.md': '# New\n'})
        self.assertIn('docs/new.md', msg)

    def test_missing_repo_path_fails(self):
        msg = self.fails({'docs/guide.md': GUIDE + '\n[x](../nowhere.py)\n'})
        self.assertIn('nowhere.py', msg)

    def test_missing_mapped_section_fails(self):
        pages = dict(PAGES, Gone=[('guide.md', 'Removed')])
        self.assertIn('no longer exists', self.fails(pages=pages))

    def test_fenced_h2_is_not_a_section(self):
        files = {'docs/guide.md': GUIDE + '\n```\n## Not a heading\n```\n'}
        out = self.build(files)
        self.assertIn('## Not a heading', out['Guide-Beta'])

    def test_wiki_pages_copied_and_link_checked(self):
        out = self.build({'docs/wiki/Home.md': 'Home [a](Other) [b](Other#deep-dive)\n'})
        self.assertEqual(out['Home'], 'Home [a](Other) [b](Other#deep-dive)\n')
        self.assertIn('does not exist', self.fails({'docs/wiki/Home.md': '[x](Missing-Page)\n'}))
        self.assertIn('heading', self.fails({'docs/wiki/Home.md': '[x](Other#zzz)\n'}))

    def test_wiki_page_name_collision_fails(self):
        self.assertIn('collides', self.fails({'docs/wiki/Other.md': 'x\n'}))

    def test_write_refuses_non_empty_dir(self):
        out = self.build()
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'stale.md').write_text('x')
            with self.assertRaises(build_wiki.WikiError):
                build_wiki.write(out, Path(d))
            self.assertEqual([p.name for p in Path(d).iterdir()], ['stale.md'])

    def test_write_creates_new_dir(self):
        out = self.build()
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'new'
            build_wiki.write(out, target)
            self.assertEqual({p.stem for p in target.glob('*.md')}, set(out))

    def test_slug_table(self):
        for heading, want in [("Texas Hold'em", 'texas-holdem'),
                              ('Model profiles (Backend tab)', 'model-profiles-backend-tab'),
                              ('서버 설정', '서버-설정'), ('C#', 'c')]:
            self.assertEqual(build_wiki.slug(heading), want)

    def test_duplicate_heading_slug_gets_suffix(self):
        self.assertEqual(build_wiki.page_slugs('## Foo\n\n## Foo\n'), {'foo', 'foo-1'})

    def test_closing_hashes_stripped_only_after_whitespace(self):
        self.assertEqual(build_wiki.headings('## C#\n## Foo ##\n'), ['C#', 'Foo'])
        self.assertEqual(build_wiki.split('## C#\nx\n')[1], [('C#', 'x')])

    def test_tilde_and_indented_fences_hide_headings(self):
        text = '## Real\n\n~~~\n## Fake\n~~~\n\n  ```\n## Fake2\n  ```\n\n````\n```\n## Fake3\n````\n'
        self.assertEqual([h for h, _ in build_wiki.split(text)[1]], ['Real'])
        self.assertEqual(build_wiki.headings(text), ['Real'])

    def test_unmapped_intro_fails(self):
        pages = {k: v for k, v in PAGES.items() if k != 'Guide-Intro'}
        self.assertIn('above the first', self.fails(pages=pages))

    def test_ambiguous_anchor_fails(self):
        guide = GUIDE + '\n### Alpha detail\n\nAgain.\n'
        self.assertIn('ambiguous', self.fails({'docs/guide.md': guide}))

    def test_main_fails_on_bad_tree(self):
        err = io.StringIO()
        with mock.patch.object(build_wiki, 'build', side_effect=build_wiki.WikiError(['boom'])), \
                contextlib.redirect_stderr(err):
            self.assertEqual(build_wiki.main(['--check']), 1)
        self.assertIn('wiki build failed:', err.getvalue())

    def test_unwritable_out_reported_cleanly(self):
        err = io.StringIO()
        with mock.patch.object(Path, 'write_text', side_effect=PermissionError('denied')), \
                tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = build_wiki.main(['--out', str(Path(d) / 'o')])
        self.assertEqual(rc, 1)
        self.assertIn('wiki build failed:', err.getvalue())
        self.assertIn('denied', err.getvalue())


class RepoDocsTests(unittest.TestCase):
    def test_repo_docs_build_cleanly(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = build_wiki.main(['--check'])
        self.assertEqual(rc, 0, err.getvalue())


if __name__ == '__main__':
    unittest.main()
