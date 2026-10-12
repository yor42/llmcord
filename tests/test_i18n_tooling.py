import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from babel.messages.pofile import read_po

from scripts import i18n as tool
from scripts import i18n_check

SOURCES = {
    'llmcord_core/x_discord.py': '''\
from llmcord_core.i18n import N_, NP_


GREETING = N_("Hello there")
CTX = NP_("button.save", "Save")


async def run(t, name, n):
    t("Plain message")
    t("Welcome {name}", name=name)
    # i18n: shown after a vote
    t.p("verb", "Open")
    t.n("{count} card", "{count} cards", n)
    t.np("noun", "{count} file", "{count} files", n)
    t(f"Skipped {name}")
''',
    'llmcord_core/y_ui.py': '''\
def page(t):
    t("Web only")
''',
    'llmcord_core/slash.py': '''\
from llmcord_core.i18n.discord_locale import C_


DESC = C_("Roll the dice")
''',
    'llmcord_core/i18n/terms.py': '''\
from llmcord_core.i18n import NP_

POT = NP_("food", "Pot")
''',
    'llmcord_core/i18n/codes.py': '''\
from llmcord_core.i18n import E_

NOPE = E_("x.nope", "Nope, not allowed")
''',
}


def make_tree(root: Path, sources=SOURCES) -> Path:
    for rel, text in sources.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    return root


def run_main(argv, root):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = tool.main(argv, root=root)
    return code, out.getvalue(), err.getvalue()


def po_path(root, lang, domain):
    return Path(root) / 'llmcord_core/locales' / lang / 'LC_MESSAGES' / f'{domain}.po'


def pot_path(root, domain):
    return Path(root) / 'llmcord_core/locales/templates' / f'{domain}.pot'


def keys(catalog):
    return {(m.context, m.id if isinstance(m.id, str) else m.id[0]) for m in catalog if m.id}


class TempTree(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = make_tree(Path(tmp.name))


class ExtractTests(TempTree):
    """FEAT-43: extraction output per domain."""

    def extract(self, domain):
        return {(m.context, m.id if isinstance(m.id, str) else m.id[0]): m
                for m in tool.extract_domain(self.root, domain) if m.id}

    def test_bot_domain_messages_contexts_and_plurals(self):
        found = self.extract('bot')
        self.assertEqual(set(found), {
            (None, 'Hello there'), ('button.save', 'Save'), (None, 'Plain message'),
            (None, 'Welcome {name}'), ('verb', 'Open'), (None, '{count} card'),
            ('noun', '{count} file')})
        self.assertEqual(found[(None, '{count} card')].id, ('{count} card', '{count} cards'))
        self.assertEqual(found[('noun', '{count} file')].id, ('{count} file', '{count} files'))

    def test_translator_comment(self):
        message = self.extract('bot')[('verb', 'Open')]
        self.assertEqual(message.auto_comments, ['i18n: shown after a vote'])

    def test_fstring_skipped(self):
        for domain in ('bot', 'web'):
            self.assertFalse(any('Skipped' in k[1] for k in self.extract(domain)))

    def test_files_map_to_domains(self):
        self.assertIn((None, 'Web only'), self.extract('web'))
        self.assertNotIn((None, 'Web only'), self.extract('bot'))
        self.assertEqual(set(self.extract('commands')),
                         {(None, 'Roll the dice')})
        self.assertEqual(set(self.extract('common')), {('food', 'Pot')})
        self.assertEqual(set(self.extract('errors')), {(None, 'Nope, not allowed')})

    def test_locations_are_relative_without_line_numbers(self):
        text = tool.render_template(tool.extract_domain(self.root, 'bot'))
        self.assertIn('#: llmcord_core/x_discord.py\n', text)
        self.assertNotIn(str(self.root), text)
        self.assertNotIn('x_discord.py:', text)

    def test_template_has_msgctxt_and_plural_entries(self):
        text = tool.render_template(tool.extract_domain(self.root, 'bot'))
        self.assertIn('msgctxt "noun"\nmsgid "{count} file"\nmsgid_plural "{count} files"\n', text)
        self.assertIn('#. i18n: shown after a vote\n', text)

    def test_extraction_is_deterministic(self):
        first = [tool.render_template(tool.extract_domain(self.root, d)) for d in tool.DOMAINS]
        second = [tool.render_template(tool.extract_domain(self.root, d)) for d in tool.DOMAINS]
        self.assertEqual(first, second)

    def test_write_templates_creates_files_then_is_idempotent(self):
        changed = tool.write_templates(self.root)
        self.assertEqual({p.name for p in changed}, {f'{d}.pot' for d in tool.DOMAINS})
        self.assertEqual(tool.write_templates(self.root), [])

    def test_extract_command_writes_templates(self):
        code, out, _ = run_main(['extract'], self.root)
        self.assertEqual(code, 0)
        self.assertIn('5 changed', out)
        self.assertTrue(pot_path(self.root, 'bot').is_file())


class InitTests(TempTree):
    """FEAT-43: init creates a locale from the templates."""

    def setUp(self):
        super().setUp()
        tool.write_templates(self.root)

    def test_init_ko_creates_every_domain(self):
        made = tool.init_locale(self.root, 'ko')
        self.assertEqual(len(made), len(tool.DOMAINS))
        for domain in tool.DOMAINS:
            path = po_path(self.root, 'ko', domain)
            self.assertTrue(path.is_file(), domain)
            with path.open('rb') as handle:
                catalog = read_po(handle, locale='ko')
            self.assertEqual(catalog.num_plurals, 1)
            self.assertIn('nplurals=1; plural=0', catalog.plural_forms)
        with po_path(self.root, 'ko', 'bot').open('rb') as handle:
            self.assertIn((None, 'Plain message'), keys(read_po(handle, locale='ko')))

    def test_refuses_reserved_invalid_and_existing(self):
        for lang in ('en', 'pseudo', 'templates', 'not a tag', 'KO', 'ko-KR', ''):
            with self.assertRaises(ValueError, msg=lang):
                tool.init_locale(self.root, lang)
        self.assertFalse((self.root / 'llmcord_core/locales/pseudo').exists())
        tool.init_locale(self.root, 'ko')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            tool.init_locale(self.root, 'ko')

    def test_main_init_returns_zero_then_one(self):
        self.assertEqual(run_main(['init', 'ko'], self.root)[0], 0)
        code, _, err = run_main(['init', 'ko'], self.root)
        self.assertEqual(code, 1)
        self.assertIn('already exists', err)
        self.assertEqual(run_main(['init', 'en'], self.root)[0], 1)


class UpdateTests(TempTree):
    """FEAT-43: update merges templates into catalogs."""

    def setUp(self):
        super().setUp()
        tool.write_templates(self.root)
        tool.init_locale(self.root, 'ko')

    def load(self, domain='bot'):
        with po_path(self.root, 'ko', domain).open('rb') as handle:
            return read_po(handle, locale='ko')

    def translate(self, msgid, text, domain='bot'):
        path = po_path(self.root, 'ko', domain)
        source = path.read_text(encoding='utf-8')
        marker = f'msgid "{msgid}"\nmsgstr ""'
        self.assertIn(marker, source)
        path.write_text(source.replace(marker, f'msgid "{msgid}"\nmsgstr "{text}"'),
                        encoding='utf-8')

    def edit_source(self, old, new):
        path = self.root / 'llmcord_core/x_discord.py'
        path.write_text(path.read_text(encoding='utf-8').replace(old, new), encoding='utf-8')
        tool.write_templates(self.root)

    def test_new_msgid_is_merged(self):
        self.edit_source('t("Plain message")', 't("Plain message")\n    t("Brand new")')
        self.assertNotIn((None, 'Brand new'), keys(self.load()))
        self.assertTrue(tool.update_catalogs(self.root))
        self.assertIn((None, 'Brand new'), keys(self.load()))

    def test_changed_msgid_becomes_fuzzy_and_old_is_dropped(self):
        self.translate('Plain message', '평범한 메시지')
        self.edit_source('"Plain message"', '"Plain messages"')
        tool.update_catalogs(self.root)
        catalog = self.load()
        message = catalog.get('Plain messages')
        self.assertTrue(message.fuzzy)
        self.assertEqual(message.string, '평범한 메시지')
        self.assertNotIn((None, 'Plain message'), keys(catalog))
        self.assertEqual(len(catalog.obsolete), 0)
        self.assertNotIn('#~', po_path(self.root, 'ko', 'bot').read_text(encoding='utf-8'))

    def test_removed_msgid_is_dropped(self):
        self.translate('Plain message', '평범한 메시지')
        self.edit_source('    t("Plain message")\n', '')
        tool.update_catalogs(self.root)
        self.assertNotIn('Plain message', po_path(self.root, 'ko', 'bot').read_text(encoding='utf-8'))
        self.assertNotIn('#~', po_path(self.root, 'ko', 'bot').read_text(encoding='utf-8'))

    def test_translation_survives_update(self):
        self.translate('Plain message', '평범한 메시지')
        tool.update_catalogs(self.root)
        self.assertEqual(self.load().get('Plain message').string, '평범한 메시지')

    def test_update_is_idempotent(self):
        self.translate('Plain message', '평범한 메시지')
        self.edit_source('"Plain message"', '"Plain messages"')
        tool.update_catalogs(self.root)
        before = {p: p.read_text(encoding='utf-8')
                  for p in (po_path(self.root, 'ko', d) for d in tool.DOMAINS)}
        self.assertEqual(tool.update_catalogs(self.root), [])
        after = {p: p.read_text(encoding='utf-8') for p in before}
        self.assertEqual(before, after)

    def test_update_creates_missing_domain_file(self):
        po_path(self.root, 'ko', 'web').unlink()
        changed = tool.update_catalogs(self.root)
        self.assertEqual([p.name for p in changed], ['web.po'])
        self.assertTrue(po_path(self.root, 'ko', 'web').is_file())

    def test_main_update_returns_zero(self):
        code, out, _ = run_main(['update'], self.root)
        self.assertEqual(code, 0)
        self.assertIn('0 changed', out)


class StatsTests(TempTree):
    """FEAT-43: stats counts translated, fuzzy and untranslated entries."""

    def test_counts(self):
        tool.write_templates(self.root)
        tool.init_locale(self.root, 'ko')
        path = po_path(self.root, 'ko', 'bot')
        text = path.read_text(encoding='utf-8')
        text = text.replace('msgid "Plain message"\nmsgstr ""',
                            'msgid "Plain message"\nmsgstr "평범"')
        text = text.replace('msgid "Hello there"\nmsgstr ""',
                            'msgid "Hello there"\nmsgstr "안녕"')
        text = text.replace('msgid "Hello there"', '#, fuzzy\nmsgid "Hello there"')
        path.write_text(text, encoding='utf-8')
        rows = {(r['locale'], r['domain']): r for r in tool.stats(self.root)}
        self.assertEqual(len(rows), len(tool.DOMAINS))
        bot = rows[('ko', 'bot')]
        self.assertEqual(bot['total'], 7)
        self.assertEqual((bot['translated'], bot['fuzzy'], bot['untranslated']), (1, 1, 5))
        self.assertEqual(rows[('ko', 'common')]['untranslated'], 1)

    def test_main_stats_prints_rows(self):
        tool.write_templates(self.root)
        tool.init_locale(self.root, 'ko')
        code, out, _ = run_main(['stats'], self.root)
        self.assertEqual(code, 0)
        self.assertIn('total=7', out)

    def test_no_locales_gives_no_rows(self):
        self.assertEqual(tool.stats(self.root), [])


class MainTests(unittest.TestCase):
    """FEAT-43: main exit codes."""

    def test_bad_subcommand_exits_nonzero(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            tool.main(['bogus'])
        self.assertNotEqual(ctx.exception.code, 0)

    def test_real_repository_is_consistent(self):
        """FEAT-43: deliberately mirrors the verify.sh/CI i18n stage, ratchet included.

        When UI literal counts change, refresh the baseline with `python scripts/i18n.py ratchet --update`.
        """
        code, out, _ = run_main(['check'], None)
        self.assertEqual(code, 0, out)


class CheckCliTests(TempTree):
    """FEAT-43: exit codes and output of the check and ratchet subcommands on a temp tree."""

    UI = 'async def show(i):\n    await i.response.send_message("Hello there")\n'

    def setUp(self):
        super().setUp()
        path = self.root / 'llmcord_core/x_discord.py'
        path.write_text(path.read_text(encoding='utf-8').replace('    t(f"Skipped {name}")\n', ''), encoding='utf-8')
        tool.write_templates(self.root)
        tool.init_locale(self.root, 'ko')
        i18n_check.write_ratchet(self.root)

    def add_source(self, rel, text):
        (self.root / rel).write_text(text, encoding='utf-8')

    def test_clean_tree(self):
        code, out, _ = run_main(['check'], self.root)
        self.assertEqual(code, 0, out)
        self.assertEqual(run_main(['ratchet'], self.root)[0], 0)

    def test_stale_template_fails_check_but_not_ratchet(self):
        self.add_source('llmcord_core/y_ui.py', 'def page(t):\n    t("Web only")\n    t("Brand new")\n')
        code, out, _ = run_main(['check'], self.root)
        self.assertEqual(code, 1)
        self.assertTrue(any(line.startswith('[stale-template]') for line in out.splitlines()), out)
        code, out, _ = run_main(['ratchet'], self.root)
        self.assertEqual(code, 0, out)

    def test_ratchet_increase_fails_both(self):
        self.add_source('llmcord_core/z_discord.py', self.UI)
        for argv in (['check'], ['ratchet']):
            code, out, _ = run_main(argv, self.root)
            self.assertEqual(code, 1, argv)
            self.assertTrue(any(line.startswith('[ratchet]') for line in out.splitlines()), out)

    def test_missing_baseline_fails_ratchet(self):
        (self.root / 'scripts/i18n_ratchet.json').unlink()
        code, out, _ = run_main(['ratchet'], self.root)
        self.assertEqual(code, 1)
        self.assertIn('is missing', out)

    def test_ratchet_update_refuses_to_raise_baseline(self):
        self.add_source('llmcord_core/z_discord.py', self.UI)
        path = self.root / 'scripts/i18n_ratchet.json'
        before = path.read_text(encoding='utf-8')
        code, out, err = run_main(['ratchet', '--update'], self.root)
        self.assertEqual(code, 1)
        self.assertIn('z_discord.py', out)
        self.assertIn('refusing', err)
        self.assertEqual(path.read_text(encoding='utf-8'), before)
        self.assertEqual(run_main(['ratchet'], self.root)[0], 1)

    def test_ratchet_update_allow_increase_writes_baseline(self):
        self.add_source('llmcord_core/z_discord.py', self.UI)
        code, out, _ = run_main(['ratchet', '--update', '--allow-increase'], self.root)
        self.assertEqual(code, 0, out)
        self.assertIn('z_discord.py', (self.root / 'scripts/i18n_ratchet.json').read_text(encoding='utf-8'))
        self.assertEqual(run_main(['ratchet'], self.root)[0], 0)

    def test_ratchet_update_accepts_a_decrease(self):
        self.add_source('llmcord_core/z_discord.py', self.UI)
        i18n_check.write_ratchet(self.root)
        self.add_source('llmcord_core/z_discord.py', 'async def show(i):\n    pass\n')
        self.assertEqual(run_main(['ratchet'], self.root)[0], 1)
        code, out, _ = run_main(['ratchet', '--update'], self.root)
        self.assertEqual(code, 0, out)
        self.assertEqual(run_main(['ratchet'], self.root)[0], 0)

    def test_ratchet_update_creates_missing_baseline(self):
        (self.root / 'scripts/i18n_ratchet.json').unlink()
        code, out, _ = run_main(['ratchet', '--update'], self.root)
        self.assertEqual(code, 0, out)
        self.assertTrue((self.root / 'scripts/i18n_ratchet.json').is_file())

    def corrupt_baseline(self, text='<<<<<<< HEAD\n{\n=======\n{}\n>>>>>>> branch\n'):
        path = self.root / 'scripts/i18n_ratchet.json'
        path.write_text(text, encoding='utf-8')
        return path

    def test_ratchet_update_refuses_corrupt_baseline(self):
        """FEAT-43: an unreadable baseline is not treated as missing, so --update cannot silently raise it."""
        for text in ('{', '<<<<<<< HEAD\n{}\n=======\n{}\n>>>>>>> branch\n'):
            with self.subTest(text=text):
                path = self.corrupt_baseline(text)
                self.add_source('llmcord_core/z_discord.py', self.UI)
                code, out, err = run_main(['ratchet', '--update'], self.root)
                self.assertEqual(code, 1, (out, err))
                self.assertIn('unreadable', out + err)
                self.assertIn('ratchet.json', out + err)
                self.assertEqual(path.read_text(encoding='utf-8'), text)

    def test_ratchet_update_allow_increase_repairs_corrupt_baseline(self):
        self.corrupt_baseline('{')
        self.add_source('llmcord_core/z_discord.py', self.UI)
        code, out, err = run_main(['ratchet', '--update', '--allow-increase'], self.root)
        self.assertEqual(code, 0, (out, err))
        path = self.root / 'scripts/i18n_ratchet.json'
        self.assertIn('z_discord.py', path.read_text(encoding='utf-8'))
        json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(run_main(['ratchet'], self.root)[0], 0)

    def test_check_corrupt_baseline_says_unreadable_without_plain_update_advice(self):
        self.corrupt_baseline('{')
        code, out, _ = run_main(['check'], self.root)
        self.assertEqual(code, 1, out)
        lines = [line for line in out.splitlines() if line.startswith('[ratchet]')]
        self.assertTrue(lines, out)
        self.assertIn('unreadable', lines[0])
        self.assertNotIn('run python scripts/i18n.py ratchet --update', lines[0])

    def test_check_missing_baseline_says_missing_and_recommends_update(self):
        (self.root / 'scripts/i18n_ratchet.json').unlink()
        code, out, _ = run_main(['check'], self.root)
        self.assertEqual(code, 1, out)
        lines = [line for line in out.splitlines() if line.startswith('[ratchet]')]
        self.assertTrue(lines, out)
        self.assertIn('missing', lines[0])
        self.assertIn('ratchet --update', lines[0])


if __name__ == '__main__':
    unittest.main()
