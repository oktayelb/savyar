import unittest

import util.decomposer as sfx
import util.word_methods as wrd
from data.treebank_adapter_commons import (
    DECOMPOSED_LEMMAS,
    bare_root_entry,
    build_treebank_forced_entry,
    expand_decomposed_lemma,
)


def is_root(word, pos):
    return wrd.can_be_noun(word) if pos == "noun" else wrd.can_be_verb(word)


class DecomposedLemmaTableTest(unittest.TestCase):
    def test_clearly_suffixed_words_are_not_roots(self):
        for word, pos in (("kullan", "verb"), ("çalış", "verb"), ("bulun", "verb"), ("yaşan", "verb"),
                          ("olan", "noun"), ("olarak", "noun"), ("durum", "noun"), ("yemek", "noun")):
            self.assertFalse(is_root(word, pos), word)
            self.assertIn(pos, DECOMPOSED_LEMMAS[word], word)

    def test_words_that_only_look_suffixed_stay_roots(self):
        for word in ("gece", "kara", "yer", "dolar", "ekmek", "alan", "deniz", "oyun"):
            self.assertTrue(wrd.can_be_noun(word), word)
            self.assertNotIn(word, DECOMPOSED_LEMMAS, word)

    def test_every_split_is_one_the_decomposer_builds(self):
        rows = [
            (lemma, pos, base, names)
            for lemma, splits in DECOMPOSED_LEMMAS.items()
            for pos, (base, names) in splits.items()
        ]
        for lemma, pos, base, names in rows[::25]:
            self.assertFalse(is_root(lemma, pos), lemma)
            produced = {
                (root, tuple(s.name for s in chain))
                for root, _p, chain, final_pos in sfx.decompose(lemma)
                if final_pos == pos
            }
            self.assertIn((base, tuple(names)), produced, lemma)


class LemmaExpansionTest(unittest.TestCase):
    def gold(self, surface, lemma, names):
        entry = build_treebank_forced_entry(surface, lemma, names)
        return entry["root"], [s["name"] for s in entry["suffixes"]]

    def test_a_removed_lemma_is_expanded_into_its_base(self):
        self.assertEqual(
            self.gold("kullanıyor", "kullan", ["continuous_iyor"]),
            ("kul", ["applicative_le", "reflexive_in", "continuous_iyor"]),
        )
        self.assertEqual(self.gold("durumu", "durum", ["possessive_3sg"]), ("dur", ["nounifier_im", "possessive_3sg"]))

    def test_the_part_of_speech_picks_the_split(self):
        self.assertEqual(self.gold("oluşur", "oluş", ["factative_ir"]), ("ol", ["reflexive_is", "factative_ir"]))

    def test_a_verb_tagged_lemma_takes_the_verb_split(self):
        self.assertEqual(expand_decomposed_lemma("alın", ["conjugation_1sg"]), ("alın", ["conjugation_1sg"]))
        self.assertEqual(
            expand_decomposed_lemma("alın", ["conjugation_1sg"], verb_lemma=True),
            ("al", ["passive_il", "conjugation_1sg"]),
        )

    def test_a_bare_removed_lemma_keeps_its_suffixes(self):
        entry = bare_root_entry("durum")
        self.assertEqual((entry["root"], [s["name"] for s in entry["suffixes"]]), ("dur", ["nounifier_im"]))
        self.assertEqual(bare_root_entry("elma")["suffixes"], [])

    def test_kept_lemmas_are_untouched(self):
        self.assertEqual(self.gold("evler", "ev", ["plural_ler"]), ("ev", ["plural_ler"]))


def readings(word):
    return {(root, tuple(s.name for s in chain)) for root, _pos, chain, _final in sfx.decompose(word)}


class SuffixRuleTest(unittest.TestCase):
    def test_polysyllabic_l_stems_take_the_causative_t(self):
        self.assertIn(("yüksel", ("active_it", "pasttense_di")), readings("yükseltti"))
        self.assertIn(("azal", ("active_it", "infinitive_mek")), readings("azaltmak"))

    def test_monosyllabic_l_stems_do_not(self):
        self.assertNotIn(("öl", ("active_it", "pasttense_di")), readings("öltdü"))
        self.assertIn(("bil", ("active_dir", "pasttense_di")), readings("bildirdi"))

    def test_imperatives_attach_to_derived_verb_stems(self):
        self.assertIn(("kul", ("applicative_le", "reflexive_in", "conjugation_2pl")), readings("kullanın"))
        self.assertIn(("kul", ("applicative_le", "reflexive_in", "conjugation_3sg")), readings("kullansın"))

    def test_the_plural_takes_relative_ce(self):
        self.assertIn(("bin", ("plural_ler", "relative_ce")), readings("binlerce"))
        self.assertIn(("yıl", ("plural_ler", "relative_ce")), readings("yıllarca"))

    def test_the_diminutive_softens(self):
        self.assertIn(("söz", ("diminutive_cik", "possessive_3sg")), readings("sözcüğü"))


class LexiconHygieneTest(unittest.TestCase):
    def test_infinitives_are_not_noun_entries(self):
        for word in ("okumak", "gitmek", "akdetmek", "yükseltmek"):
            self.assertFalse(wrd.can_be_noun(word), word)

    def test_nouns_that_look_like_infinitives_stay(self):
        for word in ("ekmek", "tokmak", "başparmak"):
            self.assertTrue(wrd.can_be_noun(word), word)

    def test_inflected_surfaces_are_not_entries(self):
        for word in ("oğlum", "ablan", "ekonomide", "yaptık", "bombanının"):
            self.assertFalse(wrd.can_be_noun(word) or wrd.can_be_verb(word), word)

    def test_core_verbs_are_entries(self):
        for word in ("sil", "kır", "aş", "bat", "in", "yalvar", "çürü", "kabar", "der"):
            self.assertTrue(wrd.can_be_verb(word), word)

    def test_dictionary_nouns_that_look_inflected_stay(self):
        for word in ("yastık", "tarife", "dana", "vana", "testi", "yazın", "öğrenci", "diken"):
            self.assertTrue(wrd.can_be_noun(word), word)

    def test_nouns_are_not_verb_entries(self):
        for word in ("adam", "tel", "tepki", "çalışan", "buruk", "altıpar", "yaş", "soru", "genel", "kayıt"):
            self.assertFalse(wrd.can_be_verb(word), word)
        self.assertTrue(wrd.can_be_noun("adam"))


class DerivedLexiconGuardTest(unittest.TestCase):
    def test_decomposed_lemmas_are_not_added_back(self):
        import tools.build_derived_lexicons as builder
        self.assertIn("kullan", builder.DECOMPOSED_LEMMAS)


if __name__ == "__main__":
    unittest.main()
