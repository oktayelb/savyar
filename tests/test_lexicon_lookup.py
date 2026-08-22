import tempfile
import unittest
from pathlib import Path

import util.decomposer as sfx
import util.word_methods as wrd
from util.suffix import DISABLED_SUFFIX_NAMES, is_enabled


class LexiconKeyTest(unittest.TestCase):
    def test_turkish_casing_rules_are_used(self):
        # The dotted/dotless I pair is why str.lower() cannot be used here.
        self.assertEqual(wrd.lexicon_key("İSTANBUL"), "istanbul")
        self.assertEqual(wrd.lexicon_key("IŞIK"), "ışık")
        self.assertEqual(wrd.lexicon_key("Ankara"), "ankara")

    def test_already_folded_words_are_unchanged(self):
        self.assertEqual(wrd.lexicon_key("kitap"), "kitap")


class CaseInsensitiveLookupTest(unittest.TestCase):
    def test_lowercased_proper_nouns_resolve(self):
        # words.txt stores these capitalised while every word reaching the
        # decomposer has been tr_lower()'d by the input pipeline.
        for root in ("ankara", "türkiye", "istanbul"):
            self.assertTrue(wrd.can_be_noun(root), root)

    def test_capitalised_input_resolves_too(self):
        for root in ("Ankara", "İstanbul", "TÜRKİYE"):
            self.assertTrue(wrd.can_be_noun(root), root)

    def test_verbs_fold_as_well(self):
        self.assertTrue(wrd.can_be_verb("gel"))
        self.assertTrue(wrd.can_be_verb("GEL"))

    def test_unsuffixables_fold_as_well(self):
        sample = next(iter(wrd.UNSUFFIXABLE_SET))
        self.assertTrue(wrd.is_unsuffixable(sample))
        self.assertTrue(wrd.is_unsuffixable(sample.upper()))

    def test_soft_l_entries_still_resolve(self):
        soft_l_entry = next((w for w in wrd.WORDS_SET if w.endswith("ł")), None)
        if soft_l_entry is None:
            self.skipTest("no soft-l entries in words.txt")
        self.assertTrue(wrd.can_be_noun(soft_l_entry[:-1] + "l"))

    def test_unknown_words_still_fail(self):
        self.assertFalse(wrd.exists("zzzyok"))
        self.assertFalse(wrd.can_be_noun(""))


class LexiconStateTest(unittest.TestCase):
    def test_entries_keep_their_original_case(self):
        # data_manager.delete() rewrites words.txt from these sets, so folding
        # must never reach them.
        self.assertIn("Ankara", wrd.WORDS_SET)
        self.assertIn("Ankara", wrd.get_all_words())

    def test_delete_word_is_case_insensitive_and_reindexes(self):
        self.assertTrue(wrd.can_be_noun("ankara"))
        try:
            self.assertTrue(wrd.delete_word("ANKARA"))
            self.assertNotIn("Ankara", wrd.WORDS_SET)
            self.assertFalse(wrd.can_be_noun("ankara"))
        finally:
            wrd.WORDS_SET.add("Ankara")
            wrd._reindex_dictionary()
        self.assertTrue(wrd.can_be_noun("ankara"))

    def test_delete_word_reports_nothing_removed(self):
        self.assertFalse(wrd.delete_word("zzzyok"))


class DerivedLexiconTest(unittest.TestCase):
    def test_noun_set_is_the_union_of_its_files(self):
        self.assertEqual(
            wrd.WORDS_SET,
            wrd._FILE_ENTRIES[wrd.DATA_FILE] | wrd._FILE_ENTRIES[wrd.DERIVED_DATA_FILE],
        )

    def test_verb_set_is_the_union_of_its_files(self):
        self.assertEqual(
            wrd.VERB_SET,
            wrd._FILE_ENTRIES[wrd.VERB_DATA_FILE] | wrd._FILE_ENTRIES[wrd.DERIVED_VERB_DATA_FILE],
        )

    def test_derived_entries_resolve_like_core_ones(self):
        derived = wrd._FILE_ENTRIES[wrd.DERIVED_VERB_DATA_FILE]
        if not derived:
            self.skipTest("verbs_derived.txt is empty")
        self.assertTrue(wrd.can_be_verb(next(iter(derived))))

    def test_missing_derived_file_is_simply_empty(self):
        self.assertEqual(wrd._read_entries(Path("data") / "no_such_lexicon.txt"), set())

    def test_delete_only_touches_the_file_that_owns_the_word(self):
        # Provenance has to survive a delete, or a rewrite would collapse the
        # core and derived lexicons into one file.
        derived = wrd._FILE_ENTRIES[wrd.DERIVED_DATA_FILE]
        if not derived:
            self.skipTest("nouns_derived.txt is empty")
        victim = next(iter(derived))
        core_size = len(wrd._FILE_ENTRIES[wrd.DATA_FILE])
        try:
            self.assertTrue(wrd.delete_word(victim))
            self.assertNotIn(victim, wrd._FILE_ENTRIES[wrd.DERIVED_DATA_FILE])
            self.assertEqual(len(wrd._FILE_ENTRIES[wrd.DATA_FILE]), core_size)
        finally:
            wrd._FILE_ENTRIES[wrd.DERIVED_DATA_FILE].add(victim)
            wrd._rebuild_sets()

    def test_save_dictionary_writes_each_file_from_its_own_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            noun_path = Path(tmp) / "core.txt"
            derived_path = Path(tmp) / "derived.txt"
            original = wrd._FILE_ENTRIES
            wrd._FILE_ENTRIES = {noun_path: {"kitap", "Ankara"}, derived_path: {"bilgi"}}
            try:
                self.assertTrue(wrd.save_dictionary())
            finally:
                wrd._FILE_ENTRIES = original
            self.assertEqual(noun_path.read_text(encoding="utf-8").split(), ["Ankara", "kitap"])
            self.assertEqual(derived_path.read_text(encoding="utf-8").split(), ["bilgi"])


class DisabledSuffixTest(unittest.TestCase):
    def reachable_suffix_names(self):
        return {
            suffix.name
            for targets in sfx.SUFFIX_TRANSITIONS.values()
            for suffixes in targets.values()
            for suffix in suffixes
        }

    def test_disabled_suffixes_are_unreachable(self):
        self.assertFalse(self.reachable_suffix_names() & DISABLED_SUFFIX_NAMES)

    def test_vocabulary_still_contains_them(self):
        # They are dropped from the transition tables, not from ALL_SUFFIXES,
        # so token ids stay stable across the change.
        self.assertTrue({s.name for s in sfx.ALL_SUFFIXES} >= DISABLED_SUFFIX_NAMES)

    def test_is_enabled_agrees_with_the_tables(self):
        for suffix in sfx.ALL_SUFFIXES:
            self.assertEqual(is_enabled(suffix), suffix.name in self.reachable_suffix_names(), suffix.name)

    def test_lexicalised_stem_no_longer_competes_with_a_derivation(self):
        # "hükümet" was read as hüküm+abstractifier_iyat, a suffix no gold
        # annotation uses; only the lexical entry should survive.
        self.assertEqual([root for root, _pos, _chain, _final in sfx.decompose("hükümet")], ["hükümet"])


class DecompositionReachTest(unittest.TestCase):
    def test_proper_noun_root_is_reachable_from_lowercased_text(self):
        roots = {root for root, _pos, _chain, _final in sfx.decompose("türkiyenin")}
        self.assertIn("türkiye", roots)


if __name__ == "__main__":
    unittest.main()
