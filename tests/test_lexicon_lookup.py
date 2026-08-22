import unittest

import util.decomposer as sfx
import util.word_methods as wrd


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


class DecompositionReachTest(unittest.TestCase):
    def test_proper_noun_root_is_reachable_from_lowercased_text(self):
        roots = {root for root, _pos, _chain, _final in sfx.decompose("türkiyenin")}
        self.assertIn("türkiye", roots)


if __name__ == "__main__":
    unittest.main()
