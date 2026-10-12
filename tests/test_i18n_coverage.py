"""MNT-75 automation (D39): `i18n.py coverage`, the completeness gate in `check`, and locales/incomplete.txt."""
import contextlib
import io
import unittest

from scripts import i18n as tool
from scripts import i18n_check
from test_i18n_check import CARD, HELLO, SLASH, TreeCase, entry, plural_entry

INCOMPLETE = 'llmcord_core/locales/incomplete.txt'


def header(lang):
    return ('msgid ""\nmsgstr ""\n' f'"Language: {lang}\\n"\n"Content-Type: text/plain; charset=UTF-8\\n"\n'
            '"Plural-Forms: nplurals=1; plural=0;\\n"\n\n')


class CoverageCase(TreeCase):
    def gate(self):
        self.write(INCOMPLETE, '# gate on\n')

    def complete(self, locale='ko'):
        self.set_po('bot', HELLO + CARD, header(locale), locale)
        self.set_po('commands', SLASH, header(locale), locale)

    def partial(self, locale='ko'):
        """bot 1/2 translated, commands 1/1, other domains empty: 2/3 in all."""
        self.set_po('bot', HELLO + plural_entry('{count} card', '{count} cards', ['']), header(locale), locale)
        self.set_po('commands', SLASH, header(locale), locale)

    def coverage(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = tool.main(['coverage', *args], root=self.root)
        return code, out.getvalue()

    def cli_check(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = tool.main(['check'], root=self.root)
        return code, out.getvalue()

    def coverage_errors(self):
        return [e for e in self.run_check()[0] if e.startswith('[coverage]')]


class CoverageReportTests(CoverageCase):
    def test_complete_locale_reports_full_and_exits_zero(self):
        """MNT-75 automation: a complete locale prints 100.0% per locale and domain and exits 0 even with --min 100."""
        self.build()
        self.complete()
        code, text = self.coverage('--min', '100')
        self.assertEqual(code, 0, text)
        self.assertIn('100.0%', text)
        self.assertIn('3/3', text)
        self.assertIn('ko', text)

    def test_partial_locale_prints_counts_percent_fuzzy_untranslated(self):
        """MNT-75 automation: the locale line sums all domains and the domain line shows translated/total, one-decimal percent, fuzzy and untranslated."""
        self.build()
        self.partial()
        code, text = self.coverage()
        self.assertEqual(code, 0, text)
        lines = text.splitlines()
        self.assertTrue(any('2/3' in line and '66.7%' in line for line in lines), text)
        bot = [line for line in lines if 'bot' in line and '1/2' in line]
        self.assertEqual(len(bot), 1, text)
        self.assertIn('50.0%', bot[0])
        self.assertRegex(bot[0], r'fuzzy\D*0')
        self.assertRegex(bot[0], r'untranslated\D*1')

    def test_fuzzy_is_counted_apart_from_translated(self):
        """MNT-75 automation: a fuzzy entry is not translated; it shows in the fuzzy column, not the untranslated one."""
        self.build()
        self.set_po('bot', entry('Hello {name}', '{name:이/가} 안녕', flags='fuzzy') + CARD, header('ko'))
        self.set_po('commands', SLASH, header('ko'))
        code, text = self.coverage()
        self.assertEqual(code, 0, text)
        bot = [line for line in text.splitlines() if 'bot' in line]
        self.assertEqual(len(bot), 1, text)
        self.assertIn('1/2', bot[0])
        self.assertRegex(bot[0], r'fuzzy\D*1')
        self.assertRegex(bot[0], r'untranslated\D*0')

    def test_without_min_always_exits_zero(self):
        """MNT-75 automation: an untranslated locale still exits 0 when --min is absent."""
        self.build()
        code, text = self.coverage()
        self.assertEqual(code, 0, text)
        self.assertIn('0.0%', text)

    def test_min_above_coverage_exits_one_and_names_locale_and_domain(self):
        """MNT-75 automation: --min makes a locale/domain below the minimum exit 1 and be named in the output."""
        self.build()
        self.partial()
        code, text = self.coverage('--min', '80')
        self.assertEqual(code, 1, text)
        self.assertRegex(text, r'ko[/ :]+bot')

    def test_min_equal_to_coverage_passes(self):
        """MNT-75 automation: coverage exactly at --min is not below it."""
        self.build()
        self.set_po('bot', HELLO + CARD, header('ko'))
        self.set_po('commands', SLASH, header('ko'))
        self.assertEqual(self.coverage('--min', '100')[0], 0)

    def test_locale_filter_limits_output(self):
        """MNT-75 automation: --locale limits the report (and --min) to the named locales."""
        self.build()
        tool.init_locale(self.root, 'ja')
        self.partial('ko')
        self.complete('ja')
        code, text = self.coverage('--locale', 'ja', '--min', '100')
        self.assertEqual(code, 0, text)
        self.assertIn('ja', text)
        self.assertNotRegex(text, r'\bko\b')
        code, text = self.coverage('--locale', 'ko', '--min', '100')
        self.assertEqual(code, 1, text)
        self.assertNotRegex(text, r'\bja\b')

    def test_default_covers_every_locale_except_templates_and_pseudo(self):
        """MNT-75 automation: with no --locale every locale folder is listed; templates and pseudo never are."""
        self.build()
        tool.init_locale(self.root, 'ja')
        code, text = self.coverage()
        self.assertEqual(code, 0, text)
        self.assertRegex(text, r'\bko\b')
        self.assertRegex(text, r'\bja\b')
        self.assertNotIn('templates', text)
        self.assertNotIn('pseudo', text)

    def test_stats_still_returns_rows(self):
        """MNT-75 automation: stats() keeps its row shape for existing callers."""
        self.build()
        self.partial()
        rows = {(r['locale'], r['domain']): r for r in tool.stats(self.root)}
        self.assertEqual(rows[('ko', 'bot')], {'locale': 'ko', 'domain': 'bot', 'total': 2, 'translated': 1,
                                                'fuzzy': 0, 'untranslated': 1})


class CompletenessGateTests(CoverageCase):
    def test_incomplete_locale_is_a_coverage_error(self):
        """MNT-75 automation: a locale below 100% translated gets a [coverage] error naming locale/domain and counts."""
        self.build()
        self.gate()
        self.partial()
        errors = self.coverage_errors()
        self.assertIn('[coverage] ko/bot: 1/2 translated', errors)

    def test_untranslated_locale_errors_in_every_populated_domain(self):
        """MNT-75 automation: an untouched locale fails each domain that has messages."""
        self.build()
        self.gate()
        errors = self.coverage_errors()
        self.assertIn('[coverage] ko/bot: 0/2 translated', errors)
        self.assertIn('[coverage] ko/commands: 0/1 translated', errors)

    def test_fully_translated_locale_passes(self):
        """MNT-75 automation: a fully translated temp locale has no errors and no warnings."""
        self.build()
        self.gate()
        self.complete()
        self.assertEqual(self.run_check(), ([], []))

    def test_fuzzy_entry_is_an_error_not_a_warning(self):
        """MNT-75 automation: a fuzzy entry is an error in check when the completeness gate is on (it stays a warning without incomplete.txt)."""
        self.build()
        self.gate()
        self.set_po('bot', entry('Hello {name}', '{name:이/가} 안녕', flags='fuzzy') + CARD, header('ko'))
        self.set_po('commands', SLASH, header('ko'))
        errors, _ = self.run_check()
        self.assertTag('fuzzy', errors)

    def test_cli_check_fails_on_incomplete_locale(self):
        """MNT-75 automation: `i18n.py check` exits 1 and prints the [coverage] line for an incomplete locale."""
        self.build()
        self.gate()
        self.partial()
        code, text = self.cli_check()
        self.assertEqual(code, 1, text)
        self.assertIn('[coverage] ko/bot: 1/2 translated', text)

    def test_cli_check_passes_for_complete_locale(self):
        """MNT-75 automation: `i18n.py check` exits 0 for a complete locale."""
        self.build()
        self.gate()
        self.complete()
        code, text = self.cli_check()
        self.assertEqual(code, 0, text)


class IncompleteAllowlistTests(CoverageCase):
    def test_listed_locale_is_exempt_from_coverage_error(self):
        """MNT-75 automation: a locale listed in incomplete.txt may be below 100% without a [coverage] error."""
        self.build()
        self.partial()
        self.write(INCOMPLETE, 'ko\n')
        self.assertEqual(self.coverage_errors(), [])
        self.assertEqual(self.cli_check()[0], 0)

    def test_comments_and_blank_lines_are_ignored(self):
        """MNT-75 automation: incomplete.txt skips # comment lines and blank lines, and trims whitespace."""
        self.build()
        self.partial()
        self.write(INCOMPLETE, '# a pack being translated\n\n   \n# ko is not listed here\n  ko  \n\n')
        self.assertEqual(self.coverage_errors(), [])

    def test_comment_alone_does_not_exempt(self):
        """MNT-75 automation: a locale named only inside a comment line stays subject to the gate."""
        self.build()
        self.gate()
        self.partial()
        self.write(INCOMPLETE, '# ko\n\n')
        self.assertTrue(self.coverage_errors())

    def test_only_listed_locales_are_exempt(self):
        """MNT-75 automation: listing ja does not exempt an incomplete ko."""
        self.build()
        self.gate()
        tool.init_locale(self.root, 'ja')
        self.partial()
        self.complete('ja')
        self.write(INCOMPLETE, 'ja\n')
        errors = self.coverage_errors()
        self.assertTrue(errors and all(e.startswith('[coverage] ko/') for e in errors), errors)

    def test_missing_file_means_gate_off(self):
        """MNT-75 automation: without incomplete.txt the completeness gate is off (temp trees and packs not yet gated)."""
        self.build()
        self.partial()
        self.assertFalse((self.root / INCOMPLETE).exists())
        self.assertEqual(self.coverage_errors(), [])

    def test_repo_allowlist_file_documents_itself_and_lists_only_real_locales(self):
        """MNT-75 automation: the repo's incomplete.txt starts with a # header and every listed locale has a folder."""
        path = i18n_check.ROOT / INCOMPLETE
        self.assertTrue(path.is_file())
        lines = path.read_text(encoding='utf-8').splitlines()
        self.assertTrue(lines and lines[0].startswith('#'), lines[:1])
        listed = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        for locale in listed:
            self.assertTrue((i18n_check.ROOT / 'llmcord_core/locales' / locale).is_dir(), locale)


if __name__ == '__main__':
    unittest.main()
