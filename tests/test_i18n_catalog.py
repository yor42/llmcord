"""FEAT-43: catalog discovery and loading for llmcord_core.i18n (fixture catalogs in a temp dir)."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from babel.messages.catalog import Catalog as PoCatalog
from babel.messages.catalog import Message
from babel.messages.pofile import write_po

from llmcord_core.i18n import catalog as catalog_module
from llmcord_core.i18n import available_locales, clear_cache, load_catalog

LOGGER = 'llmcord_core.i18n'
ENV_KEYS = ('LLMCORD_LOCALES', 'LLMCORD_I18N_PSEUDO')


def write_domain(root: Path, locale: str, domain: str, entries=(), obsolete=(), locale_arg=None) -> Path:
    po = PoCatalog(locale=locale_arg or locale)
    for args, kwargs in entries:
        po.add(*args, **kwargs)
    for msgid, msgstr in obsolete:
        po.obsolete[msgid] = Message(msgid, msgstr)
    path = root / locale / 'LC_MESSAGES' / f'{domain}.po'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        write_po(fh, po)
    return path


def make_root(tmp: str) -> Path:
    root = Path(tmp)
    write_domain(root, 'ko', 'bot', [
        (('Hello',), {'string': '안녕'}),
        (('Save',), {'string': '저장', 'context': 'button'}),
        (('Save',), {'string': '구하기'}),
        ((('One file', '{count} files'),), {'string': ('파일 {count}개',)}),
        (('Fuzzy one',), {'string': '퍼지', 'flags': ['fuzzy']}),
        (('Empty one',), {'string': ''}),
    ], obsolete=[('Old one', '옛날')])
    write_domain(root, 'ko', 'common', [(('Only common',), {'string': '공통'})])
    return root


class EnvCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = make_root(self._tmp.name)
        patcher = mock.patch.dict('os.environ')
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in ENV_KEYS:
            import os
            os.environ.pop(key, None)
        clear_cache()
        self.addCleanup(clear_cache)


class AvailableLocalesTests(EnvCase):
    """FEAT-43: available_locales lists en first, then valid catalog directories."""

    def test_en_first_and_catalog_locales_sorted(self):
        write_domain(self.root, 'ja', 'web')
        self.assertEqual(available_locales(self.root), ('en', 'ja', 'ko'))

    def test_invalid_and_non_catalog_directories_ignored(self):
        (self.root / 'templates').mkdir()
        (self.root / 'pseudo' / 'LC_MESSAGES').mkdir(parents=True)
        (self.root / 'en' / 'LC_MESSAGES').mkdir(parents=True)
        (self.root / 'empty').mkdir()
        (self.root / 'fr' / 'LC_MESSAGES').mkdir(parents=True)  # no .po
        write_domain(self.root, 'zh-TW', 'bot', locale_arg='zh_TW')  # not normalized
        write_domain(self.root, 'not a locale', 'bot', locale_arg='de')
        (self.root / 'stray.txt').write_text('x')
        self.assertEqual(available_locales(self.root), ('en', 'ko'))

    def test_filter_keeps_en_and_listed(self):
        write_domain(self.root, 'ja', 'web')
        with mock.patch.dict('os.environ', {'LLMCORD_LOCALES': ' ja , xx'}):
            self.assertEqual(available_locales(self.root), ('en', 'ja'))

    def test_filter_empty_means_all(self):
        with mock.patch.dict('os.environ', {'LLMCORD_LOCALES': ' '}):
            self.assertEqual(available_locales(self.root), ('en', 'ko'))

    def test_pseudo_appended_when_enabled(self):
        with mock.patch.dict('os.environ', {'LLMCORD_I18N_PSEUDO': '1'}):
            self.assertEqual(available_locales(self.root), ('en', 'ko', 'pseudo'))
        with mock.patch.dict('os.environ', {'LLMCORD_I18N_PSEUDO': '0'}):
            self.assertEqual(available_locales(self.root), ('en', 'ko'))

    def test_pseudo_survives_locale_filter(self):
        with mock.patch.dict('os.environ', {'LLMCORD_LOCALES': 'ko', 'LLMCORD_I18N_PSEUDO': '1'}):
            self.assertEqual(available_locales(self.root), ('en', 'ko', 'pseudo'))

    def test_missing_root_gives_only_en(self):
        self.assertEqual(available_locales(self.root / 'nope'), ('en',))


class LoadCatalogTests(EnvCase):
    """FEAT-43: load_catalog reads usable translations and tolerates bad files."""

    def test_translated_entries(self):
        cat = load_catalog('ko', self.root)
        self.assertEqual(cat.locale, 'ko')
        self.assertEqual(cat.get(None, 'Hello'), '안녕')
        self.assertEqual(cat.get('button', 'Save'), '저장')
        self.assertEqual(cat.get(None, 'Save'), '구하기')
        self.assertEqual(cat.get(None, 'Only common'), '공통')

    def test_skips_fuzzy_empty_header_and_obsolete(self):
        cat = load_catalog('ko', self.root)
        for msgid in ('Fuzzy one', 'Empty one', 'Old one', ''):
            self.assertIsNone(cat.get(None, msgid), msgid)
        # Hello, Save x2, plural, Only common
        self.assertEqual(len(cat), 5)

    def test_plural_rule_from_header(self):
        cat = load_catalog('ko', self.root)
        self.assertEqual(cat.num_plurals, 1)
        self.assertEqual(cat.get_plural(None, 'One file', 1), '파일 {count}개')
        self.assertEqual(cat.get_plural(None, 'One file', 5), '파일 {count}개')
        self.assertIsNone(cat.get_plural(None, 'Missing', 5))

    def test_plural_entry_with_empty_form_skipped(self):
        write_domain(self.root, 'de', 'bot', [((('One', 'Many'), ), {'string': ('Eins', '')})])
        self.assertIsNone(load_catalog('de', self.root).get_plural(None, 'One', 1))

    def test_english_pseudo_and_unknown_locales_are_empty_english_rule(self):
        for locale in ('en', 'pseudo', 'fr'):
            cat = load_catalog(locale, self.root)
            self.assertEqual(len(cat), 0, locale)
            self.assertEqual(cat.num_plurals, 2)
            self.assertIsNone(cat.get(None, 'Hello'))
            self.assertIsNone(cat.get_plural(None, 'Hello', 2))

    def test_conflicting_domains_keep_first_and_warn(self):
        write_domain(self.root, 'ko', 'web', [(('Shared',), {'string': 'A'})])
        write_domain(self.root, 'ko', 'common', [(('Shared',), {'string': 'B'}), (('Only common',), {'string': '공통'})])
        with self.assertLogs(LOGGER, 'WARNING') as logs:
            cat = load_catalog('ko', self.root)
        self.assertEqual(cat.get(None, 'Shared'), 'A')
        self.assertEqual(cat.get(None, 'Only common'), '공통')
        text = '\n'.join(logs.output)
        self.assertIn('Shared', text)
        self.assertEqual(len(logs.records), 1)

    def test_same_translation_in_two_domains_does_not_warn(self):
        write_domain(self.root, 'ja', 'bot', [(('Same',), {'string': 'X'})])
        write_domain(self.root, 'ja', 'web', [(('Same',), {'string': 'X'})])
        with self.assertNoLogs(LOGGER, 'WARNING'):
            self.assertEqual(load_catalog('ja', self.root).get(None, 'Same'), 'X')

    def test_unparsable_file_logged_and_skipped(self):
        bad = self.root / 'ko' / 'LC_MESSAGES' / 'web.po'
        bad.write_text('msgid "a"\nmsgstr "b"\nmsgid "unterminated\n"broken"\n', encoding='utf-8')
        with self.assertLogs(LOGGER, 'ERROR') as logs:
            cat = load_catalog('ko', self.root)
        self.assertTrue(any('web.po' in line for line in logs.output))
        self.assertEqual(cat.get(None, 'Hello'), '안녕')

    def test_cached_until_clear_cache(self):
        first = load_catalog('ko', self.root)
        self.assertIs(load_catalog('ko', self.root), first)
        clear_cache()
        second = load_catalog('ko', self.root)
        self.assertIsNot(second, first)
        self.assertEqual(second.get(None, 'Hello'), '안녕')

    def test_untrusted_locale_names_load_nothing(self):
        """FEAT-43: path-like or non-normalized tags never reach the filesystem or the cache."""
        outside = Path(self._tmp.name).parent / 'outside-llmcord-i18n'
        self.addCleanup(shutil.rmtree, outside, True)
        write_domain(outside, 'ko', 'bot', [(('Hello',), {'string': '밖'})])
        write_domain(self.root, 'ko-KR', 'bot', [(('Hello',), {'string': '지역'})], locale_arg='ko')
        hostile = ('../outside-llmcord-i18n/ko', str(outside / 'ko'), 'ko-KR', 'KO', '', 'a/b')
        cached = getattr(catalog_module, '_load_catalog', load_catalog)
        before = cached.cache_info().currsize
        with mock.patch('babel.messages.pofile.read_po') as read_po:
            for locale in hostile:
                with self.subTest(locale=locale):
                    cat = load_catalog(locale, self.root)
                    self.assertEqual(len(cat), 0)
                    self.assertIsNone(cat.get(None, 'Hello'))
        read_po.assert_not_called()
        self.assertEqual(cached.cache_info().currsize, before)
        self.assertEqual(load_catalog('ko', self.root).get(None, 'Hello'), '안녕')

    def test_well_formed_tag_without_directory_loads_nothing_and_is_not_cached(self):
        """FEAT-43: a valid tag with no catalog directory neither reads files nor grows the cache."""
        cached = getattr(catalog_module, '_load_catalog', load_catalog)
        before = cached.cache_info().currsize
        with mock.patch('babel.messages.pofile.read_po') as read_po:
            for locale in ('zz', 'qq_419'):
                with self.subTest(locale=locale):
                    cat = load_catalog(locale, self.root)
                    self.assertEqual(len(cat), 0)
                    self.assertIsNone(cat.get(None, 'Hello'))
        read_po.assert_not_called()
        self.assertEqual(cached.cache_info().currsize, before)

    def test_real_locale_directory_still_loads_and_is_cached_once(self):
        cached = getattr(catalog_module, '_load_catalog', load_catalog)
        before = cached.cache_info().currsize
        first = load_catalog('ko', self.root)
        self.assertEqual(first.get(None, 'Hello'), '안녕')
        self.assertIs(load_catalog('ko', self.root), first)
        self.assertEqual(cached.cache_info().currsize, before + 1)

    def test_cache_is_per_root(self):
        with tempfile.TemporaryDirectory() as other:
            self.assertIsNot(load_catalog('ko', Path(other)), load_catalog('ko', self.root))
