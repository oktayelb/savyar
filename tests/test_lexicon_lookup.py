import tempfile
import unittest
from pathlib import Path

import util.decomposer as sfx
import util.word_methods as wrd
from util.suffix import DISABLED_SUFFIX_NAMES, is_enabled


def readings(word):
    return {(root, tuple(s.name for s in chain)) for root, _pos, chain, _final in sfx.decompose(word)}


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
        for word in sorted(wrd.UNSUFFIXABLE_SET):
            self.assertTrue(wrd.is_unsuffixable(word), word)
        self.assertTrue(wrd.is_unsuffixable("YA"))
        self.assertTrue(wrd.is_unsuffixable("ÇÜŞ"))

    def test_ascii_uppercase_i_is_not_a_turkish_uppercase_i(self):
        # "ki".upper() is "KI" in Python, whose Turkish lowercase is "kı" - a
        # different word. Folding it back to "ki" would be the bug, not this.
        self.assertTrue(wrd.is_unsuffixable("Kİ"))
        self.assertFalse(wrd.is_unsuffixable("KI"))

    def test_soft_l_entries_still_resolve(self):
        soft_l_entry = next((w for w in wrd.WORDS_SET if w.endswith("ł")), None)
        if soft_l_entry is None:
            self.skipTest("no soft-l entries in words.txt")
        self.assertTrue(wrd.can_be_noun(soft_l_entry[:-1] + "l"))

    def test_a_soft_l_inside_an_entry_resolves(self):
        for entry, surface in (("kałp", "kalp"), ("mahałłe", "mahalle"), ("gołf", "golf")):
            self.assertIn(entry, wrd.WORDS_SET)
            self.assertTrue(wrd.can_be_noun(surface), surface)

    def test_a_soft_l_after_the_last_vowel_fronts_the_suffix(self):
        self.assertIn(("kalp", ("accusative_i",)), readings("kalbi"))
        self.assertNotIn(("kalp", ("accusative_i",)), readings("kalbı"))
        self.assertIn(("golf", ("accusative_i",)), readings("golfü"))
        self.assertIn(("alkol", ("accusative_i",)), readings("alkolü"))

    def test_a_soft_l_before_the_last_vowel_does_not(self):
        self.assertIn(("imalat", ("accusative_i",)), readings("imalatı"))
        self.assertNotIn(("imalat", ("accusative_i",)), readings("imalati"))

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


class LexiconFilesTest(unittest.TestCase):
    def test_noun_set_is_its_file(self):
        self.assertEqual(wrd.WORDS_SET, wrd._FILE_ENTRIES[wrd.DATA_FILE])

    def test_verb_set_is_its_file(self):
        self.assertEqual(wrd.VERB_SET, wrd._FILE_ENTRIES[wrd.VERB_DATA_FILE])

    def test_missing_file_is_simply_empty(self):
        self.assertEqual(wrd._read_entries(Path("data") / "no_such_lexicon.txt"), set())

    def test_delete_only_touches_the_file_that_owns_the_word(self):
        nouns = wrd._FILE_ENTRIES[wrd.DATA_FILE]
        noun_keys = {wrd.lexicon_key(n) for n in nouns}
        victim = next(v for v in sorted(wrd._FILE_ENTRIES[wrd.VERB_DATA_FILE]) if wrd.lexicon_key(v) not in noun_keys)
        noun_count = len(nouns)
        try:
            self.assertTrue(wrd.delete_word(victim))
            self.assertNotIn(victim, wrd._FILE_ENTRIES[wrd.VERB_DATA_FILE])
            self.assertEqual(len(wrd._FILE_ENTRIES[wrd.DATA_FILE]), noun_count)
        finally:
            wrd._FILE_ENTRIES[wrd.VERB_DATA_FILE].add(victim)
            wrd._rebuild_sets()

    def test_save_dictionary_writes_each_file_from_its_own_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            noun_path = Path(tmp) / "nouns.txt"
            verb_path = Path(tmp) / "verbs.txt"
            original = wrd._FILE_ENTRIES
            wrd._FILE_ENTRIES = {noun_path: {"kitap", "Ankara"}, verb_path: {"gel"}}
            try:
                self.assertTrue(wrd.save_dictionary())
            finally:
                wrd._FILE_ENTRIES = original
            self.assertEqual(noun_path.read_text(encoding="utf-8").split(), ["Ankara", "kitap"])
            self.assertEqual(verb_path.read_text(encoding="utf-8").split(), ["gel"])


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
