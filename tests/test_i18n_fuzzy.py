from pathlib import Path
import tempfile
import unittest

from babel.messages.pofile import read_po

from scripts import i18n as tool

HEADER = ('msgid ""\nmsgstr ""\n'
          '"Language: ko\\n"\n'
          '"MIME-Version: 1.0\\n"\n'
          '"Content-Type: text/plain; charset=UTF-8\\n"\n'
          '"Content-Transfer-Encoding: 8bit\\n"\n'
          '"Plural-Forms: nplurals=1; plural=0;\\n"\n')


def sync(root: Path, template_body: str, po_body: str):
    base = root / 'llmcord_core/locales'
    (base / 'templates').mkdir(parents=True)
    (base / 'ko/LC_MESSAGES').mkdir(parents=True)
    (base / 'templates/bot.pot').write_text(
        'msgid ""\nmsgstr ""\n"Content-Type: text/plain; charset=UTF-8\\n"\n\n' + template_body,
        encoding='utf-8')
    po = base / 'ko/LC_MESSAGES/bot.po'
    po.write_text(HEADER + '\n' + po_body, encoding='utf-8')
    tool._sync(root, 'ko', 'bot')
    with po.open('rb') as handle:
        return read_po(handle, locale='ko')


class FuzzyFlagTests(unittest.TestCase):
    def test_empty_translation_is_not_left_fuzzy(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = sync(Path(tmp),
                           'msgid "Did you mean {first} or {second}?"\nmsgstr ""\n',
                           'msgid "Did you mean {moves}?"\nmsgstr ""\n')
        message = catalog.get('Did you mean {first} or {second}?')
        self.assertIsNotNone(message)
        self.assertNotIn('fuzzy', message.flags)

    def test_carried_over_translation_stays_fuzzy(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = sync(Path(tmp),
                           'msgid "Did you mean {first} or {second}?"\nmsgstr ""\n',
                           'msgid "Did you mean {moves}?"\nmsgstr "old translation"\n')
        message = catalog.get('Did you mean {first} or {second}?')
        self.assertIsNotNone(message)
        self.assertIn('fuzzy', message.flags)
        self.assertEqual(message.string, 'old translation')

    def test_empty_plural_is_not_left_fuzzy(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog = sync(Path(tmp),
                           'msgid "{count} cards left"\nmsgid_plural "{count} cards left!"\n'
                           'msgstr[0] ""\n',
                           'msgid "{count} cards"\nmsgid_plural "{count} cards in all"\n'
                           'msgstr[0] ""\n')
        message = catalog.get(('{count} cards left', '{count} cards left!'))
        self.assertIsNotNone(message)
        self.assertNotIn('fuzzy', message.flags)


if __name__ == '__main__':
    unittest.main()
