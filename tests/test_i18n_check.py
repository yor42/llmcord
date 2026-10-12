import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import i18n as tool
from scripts import i18n_check

BASE = {
    'llmcord_core/x_discord.py': (
        'def run(t, name):\n'
        '    t("Hello {name}", name=name)\n'
        '    t.n("{count} card", "{count} cards", 2)\n'),
    'llmcord_core/slash.py': 'DESC = C_("Roll the dice")\n',
}
HEADER = ('msgid ""\nmsgstr ""\n"Language: ko\\n"\n"Content-Type: text/plain; charset=UTF-8\\n"\n'
          '"Plural-Forms: nplurals=1; plural=0;\\n"\n\n')


def entry(msgid, msgstr, ctx=None, flags=''):
    return ((f'#, {flags}\n' if flags else '') + (f'msgctxt "{ctx}"\n' if ctx else '')
            + f'msgid "{msgid}"\nmsgstr "{msgstr}"\n\n')


def plural_entry(msgid, plural, forms):
    body = ''.join(f'msgstr[{i}] "{form}"\n' for i, form in enumerate(forms))
    return f'msgid "{msgid}"\nmsgid_plural "{plural}"\n{body}\n'


HELLO = entry('Hello {name}', '{name:이/가} 안녕')
CARD = plural_entry('{count} card', '{count} cards', ['카드 {count}장'])
SLASH = entry('Roll the dice', '주사위 굴리기')


class TreeCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def write(self, rel, text, header=None):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')

    def build(self, files=None, locale='ko'):
        for rel, text in (BASE if files is None else files).items():
            self.write(rel, text)
        tool.write_templates(self.root)
        if locale:
            tool.init_locale(self.root, locale)
        i18n_check.write_ratchet(self.root)

    def set_po(self, domain, body, header=HEADER, locale='ko'):
        self.write(f'llmcord_core/locales/{locale}/LC_MESSAGES/{domain}.po', header + body)

    def run_check(self):
        return i18n_check.check(self.root)

    def assertTag(self, tag, lines):
        self.assertTrue(any(line.startswith(f'[{tag}]') for line in lines), (tag, lines))

    def assertNoTag(self, tag, lines):
        self.assertFalse(any(line.startswith(f'[{tag}]') for line in lines), (tag, lines))

    def errors_with(self, tag):
        errors, _ = self.run_check()
        self.assertTag(tag, errors)
        return [e for e in errors if e.startswith(f'[{tag}]')]


class CatalogCheckTests(TreeCase):
    """FEAT-43: template and catalog consistency checks on temporary trees."""

    def test_clean_tree_has_no_errors(self):
        self.build()
        self.assertEqual(self.run_check(), ([], []))

    def test_clean_tree_with_valid_translations(self):
        self.build()
        self.set_po('bot', HELLO + CARD)
        self.set_po('commands', SLASH)
        self.assertEqual(self.run_check(), ([], []))

    def test_stale_template(self):
        self.build()
        self.write('llmcord_core/x_discord.py', BASE['llmcord_core/x_discord.py'] + '    t("New text")\n')
        self.errors_with('stale-template')

    def test_stale_template_when_template_missing(self):
        self.build()
        (self.root / 'llmcord_core/locales/templates/bot.pot').unlink()
        self.errors_with('stale-template')

    def test_stale_catalog(self):
        self.build()
        self.set_po('bot', HELLO)
        errors, _ = self.run_check()
        self.assertTag('stale-catalog', errors)
        self.assertNoTag('stale-template', errors)

    def test_po_parse_error_in_catalog(self):
        self.build()
        self.set_po('bot', 'msgid "Hello {name}"\n#, python-format\nmsgstr "%(a)s %(b)s"\n\n'
                    'msgid "x"\nmsgstr "%(a)s"\n#, python-format\n')
        self.errors_with('po-parse')

    def test_po_parse_error_in_template(self):
        self.build()
        self.write('llmcord_core/locales/templates/bot.pot',
                   'msgid "Hello {name}"\n#, python-format\nmsgstr "x"\n\n')
        lines = self.errors_with('po-parse')
        self.assertTrue(any('bot.pot' in line for line in lines), lines)

    def test_plural_forms_nplurals_mismatch(self):
        self.build()
        header = HEADER.replace('nplurals=1; plural=0;', 'nplurals=2; plural=(n != 1);')
        self.set_po('bot', HELLO + plural_entry('{count} card', '{count} cards', ['a', 'b']), header=header)
        self.errors_with('plural-forms')

    def test_plural_forms_language_header_mismatch(self):
        self.build()
        self.set_po('bot', HELLO + CARD, header=HEADER.replace('Language: ko', 'Language: ja'))
        lines = self.errors_with('plural-forms')
        self.assertTrue(any('Language' in line for line in lines), lines)

    def test_obsolete_entries(self):
        self.build()
        self.set_po('bot', HELLO + CARD + '#~ msgid "Old"\n#~ msgstr "옛날"\n')
        errors, _ = self.run_check()
        self.assertTag('obsolete', errors)
        self.assertNoTag('stale-catalog', errors)

    def test_field_syntax_in_msgstr(self):
        self.build()
        self.set_po('bot', entry('Hello {name}', '안녕 {}') + CARD)
        self.errors_with('field-syntax')

    def test_field_syntax_in_msgid(self):
        self.build({'llmcord_core/x_discord.py': 'def run(t):\n    t("Hello {0}")\n'})
        self.errors_with('field-syntax')

    def test_placeholder_mismatch(self):
        self.build()
        self.set_po('bot', entry('Hello {name}', '안녕 {other}') + CARD)
        self.errors_with('placeholder')

    def test_placeholder_missing_field(self):
        self.build()
        self.set_po('bot', entry('Hello {name}', '안녕') + CARD)
        self.errors_with('placeholder')

    def test_placeholder_plural_form_with_unknown_field(self):
        self.build()
        self.set_po('bot', HELLO + plural_entry('{count} card', '{count} cards', ['{count} {oops}']))
        self.errors_with('placeholder')

    def test_plural_form_may_omit_count(self):
        self.build()
        self.set_po('bot', HELLO + plural_entry('{count} card', '{count} cards', ['카드']))
        self.assertEqual(self.run_check(), ([], []))

    def test_josa_non_josa_slash_spec_in_msgstr(self):
        self.build()
        self.set_po('bot', entry('Hello {name}', '{name:x/y} 안녕') + CARD)
        self.errors_with('josa')

    def test_josa_spec_in_english_msgid(self):
        self.build({'llmcord_core/x_discord.py': 'def run(t, n):\n    t("{name:이/가} won", name=n)\n'})
        self.errors_with('josa')

    def test_limit_commands_msgid(self):
        self.build({'llmcord_core/slash.py': f'DESC = C_("{"a" * 101}")\n'})
        self.errors_with('limit')

    def test_limit_commands_exactly_100_ok(self):
        self.build({'llmcord_core/slash.py': f'DESC = C_("{"a" * 100}")\n'})
        self.assertEqual(self.run_check(), ([], []))

    def test_limit_commands_msgstr(self):
        self.build()
        self.set_po('commands', entry('Roll the dice', '가' * 101))
        self.errors_with('limit')

    def test_limit_button_context_msgid(self):
        self.build({'llmcord_core/x_discord.py': f'X = NP_("button.save", "{"a" * 81}")\n'})
        self.errors_with('limit')

    def test_limit_button_context_msgstr(self):
        self.build({'llmcord_core/x_discord.py': 'X = NP_("button.save", "Save")\n'})
        self.set_po('bot', entry('Save', '가' * 81, ctx='button.save'))
        self.errors_with('limit')

    def test_limit_button_80_and_modal_45_boundaries_ok(self):
        self.build({'llmcord_core/x_discord.py': f'X = NP_("button.a", "{"a" * 80}")\nY = NP_("modal.b", "{"b" * 45}")\n'})
        self.assertEqual(self.run_check(), ([], []))

    def test_limit_modal_46(self):
        self.build({'llmcord_core/x_discord.py': f'Y = NP_("modal.b", "{"b" * 46}")\n'})
        self.errors_with('limit')

    def test_conflict_between_domains(self):
        self.build({'llmcord_core/x_discord.py': 'def run(t):\n    t("Save")\n    C_("Save")\n'})
        self.set_po('bot', entry('Save', '저장'))
        self.set_po('commands', entry('Save', '보관'))
        self.errors_with('conflict')

    def test_same_translation_in_two_domains_is_not_a_conflict(self):
        self.build({'llmcord_core/x_discord.py': 'def run(t):\n    t("Save")\n    C_("Save")\n'})
        self.set_po('bot', entry('Save', '저장'))
        self.set_po('commands', entry('Save', '저장'))
        self.assertEqual(self.run_check(), ([], []))

    def test_fuzzy_warning_is_not_an_error(self):
        self.build()
        self.set_po('bot', entry('Hello {name}', 'wrong {other}', flags='fuzzy') + CARD)
        errors, warnings = self.run_check()
        self.assertEqual(errors, [])
        self.assertTag('fuzzy', warnings)


class MarkerCheckTests(TreeCase):
    """FEAT-43: marker-call checks (unmapped, non-literal, name-collision)."""

    def test_unmapped_error_marker_outside_codes(self):
        self.build({'llmcord_core/foo.py': 'X = E_("a.b", "Message")\n'})
        self.errors_with('unmapped')

    def test_unmapped_error_marker_in_bot_file(self):
        self.build({'llmcord_core/x_discord.py': 'X = E_("a.b", "Message")\n'})
        self.errors_with('unmapped')

    def test_unmapped_t_in_unmapped_file(self):
        self.build({'llmcord_core/foo.py': 'def run(t):\n    t("Message")\n'})
        self.errors_with('unmapped')

    def test_error_marker_in_codes_is_allowed(self):
        self.build({'llmcord_core/i18n/codes.py': 'X = E_("a.b", "Message")\n'})
        self.assertEqual(self.run_check(), ([], []))

    def test_command_marker_is_allowed_anywhere(self):
        self.build({'llmcord_core/foo.py': 'X = C_("Message")\n'})
        self.assertEqual(self.run_check(), ([], []))

    def test_unmapped_file_in_two_domains(self):
        patched = {'web': tool.DOMAIN_FILES['web'] + ['llmcord_core/usage.py']}
        with mock.patch.dict(tool.DOMAIN_FILES, patched):
            self.build({'llmcord_core/usage.py': 'def run(t):\n    t("Message")\n'})
            lines = self.errors_with('unmapped')
        self.assertTrue(any('usage.py' in line and 'several' in line for line in lines), lines)

    def test_non_literal_f_string_in_plural(self):
        self.build({'llmcord_core/x_discord.py': 'def run(t, n, x):\n    t.n(f"{x} a", "b", n)\n'})
        self.errors_with('non-literal')

    def test_non_literal_concatenation(self):
        self.build({'llmcord_core/x_discord.py': 'def run(t, b):\n    t("a" + b)\n'})
        self.errors_with('non-literal')

    def test_non_literal_format_call(self):
        self.build({'llmcord_core/x_discord.py': 'def run(t, x):\n    t("{x}".format(x=x))\n'})
        self.errors_with('non-literal')

    def test_non_literal_percent_expression(self):
        self.build({'llmcord_core/x_discord.py': 'def run(t, x):\n    t("a %s" % x)\n'})
        self.errors_with('non-literal')

    def test_constant_argument_is_allowed(self):
        self.build({'llmcord_core/x_discord.py': 'MSG = N_("Const text")\n\n\ndef run(t, kind):\n    t(MSG)\n    t(kind)\n'})
        self.assertEqual(self.run_check(), ([], []))

    def test_name_collision_outside_i18n_only(self):
        self.build({
            'llmcord_core/foo.py': 'def t():\n    pass\n\n\nclass A:\n    async def n(self):\n        pass\n',
            'llmcord_core/i18n/__init__.py': 'def N_(msgid):\n    return msgid\n',
        })
        lines = self.errors_with('name-collision')
        self.assertEqual(len(lines), 2, lines)
        self.assertTrue(all('foo.py' in line for line in lines), lines)


class RatchetTests(TreeCase):
    """FEAT-43: string ratchet comparison against scripts/i18n_ratchet.json."""

    SOURCE = 'async def run(interaction):\n    await interaction.response.send_message("Hello")\n'

    def test_unchanged_counts_pass(self):
        self.build({'llmcord_core/x_discord.py': self.SOURCE})
        self.assertEqual(self.run_check(), ([], []))

    def test_higher_count_fails(self):
        self.build({'llmcord_core/x_discord.py': self.SOURCE})
        self.write('llmcord_core/x_discord.py', self.SOURCE + '    await interaction.followup.send("More")\n')
        lines = self.errors_with('ratchet')
        self.assertTrue(any('wrap new strings in t(...)' in line for line in lines), lines)

    def test_higher_raise_count_fails(self):
        self.build({'llmcord_core/foo.py': 'x = 1\n'})
        self.write('llmcord_core/foo.py', 'def f():\n    raise ValueError("Bad input")\n')
        lines = self.errors_with('ratchet')
        self.assertTrue(any('raise UserError' in line for line in lines), lines)

    def test_lower_count_fails(self):
        self.build({'llmcord_core/x_discord.py': self.SOURCE})
        self.write('llmcord_core/x_discord.py', 'async def run(interaction):\n    pass\n')
        lines = self.errors_with('ratchet')
        self.assertTrue(any('ratchet --update' in line for line in lines), lines)

    def test_update_clears_lower_count(self):
        self.build({'llmcord_core/x_discord.py': self.SOURCE})
        self.write('llmcord_core/x_discord.py', 'async def run(interaction):\n    pass\n')
        i18n_check.write_ratchet(self.root)
        self.assertEqual(self.run_check(), ([], []))


class RatchetCountTests(TreeCase):
    """FEAT-43: ratchet_counts and write_ratchet on temporary trees."""

    def counts(self, files):
        for rel, text in files.items():
            self.write(rel, text)
        return i18n_check.ratchet_counts(self.root)

    def test_ui_literals_in_bot_file(self):
        source = (
            'async def run(interaction, t, name, x):\n'
            '    await interaction.response.send_message("Hello there")\n'
            '    await interaction.response.send_message(content="Second text")\n'
            '    await interaction.response.send_message("ok_value")\n'
            '    await interaction.response.send_message("!!! 123")\n'
            '    await interaction.followup.send(t("Wrapped text"))\n'
            '    await interaction.followup.send("Yes" if name else "No way")\n'
            '    await interaction.followup.send("Hi " + name + " bye")\n'
            '    await interaction.followup.send(f"Hi {name}")\n'
            '    app_commands.describe(amount="How many", other="Which one")\n'
            '    Embed(title="Scores", colour="red")\n'
            '    unrelated("Not counted text")\n'
            '    Button(label="가나다", custom_id="Some Id")\n')
        # 1 + 1 + 0 + 0 + 0 + 2 + 2 + 1 + 2 + 1 + 0 + 1
        self.assertEqual(self.counts({'llmcord_core/x_discord.py': source}), {'llmcord_core/x_discord.py': {'ui': 11}})

    def test_ui_literals_in_web_file(self):
        source = (
            'def page(ui):\n'
            '    ui.label("Heading")\n'
            '    ui.button("Go", on_click=f)\n'
            '    ui.input(label="Name", placeholder="Type here")\n'
            '    ui.notify("saved_ok")\n')
        self.assertEqual(self.counts({'llmcord_core/y_ui.py': source}), {'llmcord_core/y_ui.py': {'ui': 4}})

    def test_ui_ignored_outside_bot_and_web(self):
        source = 'async def run(i):\n    await i.send_message("Hello there")\n'
        self.assertEqual(self.counts({'llmcord_core/other.py': source}), {})

    def test_literals_inside_markers_are_not_counted(self):
        source = 'async def run(i, t):\n    await i.send_message(t("Hello there"))\n    i.send_message(N_("Hi"))\n'
        self.assertEqual(self.counts({'llmcord_core/x_discord.py': source}), {})

    def test_raise_literals(self):
        source = (
            'def f(x, t):\n'
            '    raise ValueError("Bad thing")\n'
            '    raise HTTPException(status_code=400, detail="Nope sorry")\n'
            '    raise RuntimeError(f"Failed {x}")\n'
            '    raise ValueError("bad_code")\n'
            '    raise ValueError(t("Wrapped"))\n'
            '    raise OSError("Not a counted type")\n'
            '    raise ValueError\n'
            '    raise errors.ConflictError("Changed elsewhere")\n'
            '    raise ValueError(x)\n')
        self.assertEqual(self.counts({'llmcord_core/foo.py': source}), {'llmcord_core/foo.py': {'raise': 4}})

    def test_raise_ignored_in_i18n_package(self):
        source = 'def f():\n    raise ValueError("Bad thing")\n'
        self.assertEqual(self.counts({'llmcord_core/i18n/format.py': source}), {})

    def test_raise_and_ui_combine_in_one_file(self):
        source = 'def f(i):\n    i.reply("Hello there")\n    raise ValueError("Bad thing")\n'
        self.assertEqual(self.counts({'llmcord_core/x_discord.py': source}),
                         {'llmcord_core/x_discord.py': {'ui': 1, 'raise': 1}})

    def test_write_ratchet_is_sorted_indented_with_trailing_newline(self):
        self.write('llmcord_core/z_ui.py', 'def f(ui):\n    ui.label("Zed")\n')
        self.write('llmcord_core/a_ui.py', 'def f(ui):\n    ui.label("Ay")\n    raise ValueError("Bad")\n')
        path = i18n_check.write_ratchet(self.root)
        self.assertEqual(path, self.root / 'scripts/i18n_ratchet.json')
        expected = {'version': 1, 'files': {'llmcord_core/a_ui.py': {'raise': 1, 'ui': 1},
                                            'llmcord_core/z_ui.py': {'ui': 1}}}
        text = path.read_text(encoding='utf-8')
        self.assertEqual(text, json.dumps(expected, sort_keys=True, indent=2) + '\n')
        self.assertTrue(text.startswith('{\n  "files": {'))
        self.assertTrue(text.endswith('}\n'))


if __name__ == '__main__':
    unittest.main()
