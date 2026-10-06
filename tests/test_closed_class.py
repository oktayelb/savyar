import unittest

import app.nlp_pipeline as nlp
import util.decomposer as sfx
import util.word_methods as wrd
from data.treebank_adapter_commons import build_closed_class_entry
from ml.ml_ranking_model import SPECIAL_ROOT_NOUN, SPECIAL_ROOT_VERB, SUFFIX_OFFSET, Trainer, build_sentence_sequence
from util.words.closed_class import CLOSED_CLASS_LEXEMES, lexeme_of


def readings(word):
    return {(root, pos, tuple(s.name for s in chain)) for root, pos, chain, _ in sfx.decompose(word)}


def root_token_id(root, pos):
    return nlp.root_token(root, pos)[0]


class LexemeTokenTest(unittest.TestCase):
    def test_every_lexeme_has_its_own_token_after_the_suffixes(self):
        ids = [root_token_id(lexeme, "noun") for lexeme in CLOSED_CLASS_LEXEMES]
        self.assertEqual(len(set(ids)), len(CLOSED_CLASS_LEXEMES))
        self.assertTrue(all(i >= SUFFIX_OFFSET + len(sfx.ALL_SUFFIXES) for i in ids))

    def test_open_class_roots_stay_interchangeable(self):
        self.assertEqual(root_token_id("elma", "noun"), SPECIAL_ROOT_NOUN)
        self.assertEqual(root_token_id("armut", "noun"), SPECIAL_ROOT_NOUN)
        self.assertEqual(root_token_id("gel", "verb"), SPECIAL_ROOT_VERB)

    def test_pronouns_are_not_interchangeable(self):
        self.assertNotEqual(root_token_id("ben", "noun"), root_token_id("biz", "noun"))

    def test_spellings_share_their_lexeme_token(self):
        self.assertEqual(root_token_id("mı", "noun"), root_token_id("mi", "noun"))
        self.assertEqual(root_token_id("da", "noun"), root_token_id("de", "noun"))

    def test_a_verb_homograph_is_an_open_class_verb(self):
        self.assertEqual(root_token_id("de", "verb"), SPECIAL_ROOT_VERB)
        self.assertEqual(root_token_id("var", "verb"), SPECIAL_ROOT_VERB)

    def test_the_lexeme_sits_in_the_root_slot_before_the_suffixes(self):
        dative = sfx.SUFFIX_BY_NAME["dative_e"]
        bana = nlp.encode_suffix_chain([dative], "ben", "noun")
        elmaya = nlp.encode_suffix_chain([dative], "elma", "noun")
        self.assertEqual(bana[1:], elmaya[1:])
        self.assertNotEqual(bana[0], elmaya[0])

    def test_suffix_metrics_ignore_lexeme_tokens(self):
        bana = nlp.encode_suffix_chain([sfx.SUFFIX_BY_NAME["dative_e"]], "ben", "noun")
        morph = Trainer._morph_tokens_from_sequence(build_sentence_sequence([bana]))
        self.assertEqual(morph, [bana[1][0]])


class ClosedClassReadingTest(unittest.TestCase):
    def test_every_lexeme_is_a_noun_root(self):
        for lexeme in CLOSED_CLASS_LEXEMES:
            self.assertTrue(wrd.can_be_noun(lexeme), lexeme)

    def test_suppletive_datives(self):
        self.assertIn(("ben", "noun", ("dative_e",)), readings("bana"))
        self.assertIn(("sen", "noun", ("dative_e",)), readings("sana"))
        self.assertIn(("ben", "noun", ("dative_e", "if_se")), readings("banaysa"))

    def test_demonstratives_take_the_pronominal_n(self):
        self.assertIn(("o", "noun", ("accusative_i",)), readings("onu"))
        self.assertIn(("o", "noun", ("plural_ler",)), readings("onlar"))
        self.assertIn(("bu", "noun", ("plural_ler", "ablative_den")), readings("bunlardan"))
        self.assertIn(("şu", "noun", ("noun_compound",)), readings("şunun"))

    def test_demonstratives_take_no_possessive(self):
        for word in ("onu", "onun", "bunu"):
            for root, _pos, chain in readings(word):
                if root in ("o", "bu"):
                    self.assertFalse(any(name.startswith("possessive") for name in chain), (word, chain))

    def test_first_person_genitive_is_im(self):
        self.assertIn(("ben", "noun", ("noun_compound",)), readings("benim"))
        self.assertIn(("biz", "noun", ("noun_compound",)), readings("bizim"))
        self.assertIn(("ben", "noun", ("noun_compound", "confactuous_le")), readings("benimle"))

    def test_irregular_possessive_stems(self):
        self.assertIn(("hep", "noun", ("possessive_3sg",)), readings("hepsi"))
        self.assertIn(("hep", "noun", ("possessive_3sg", "accusative_i")), readings("hepsini"))
        self.assertIn(("bir", "noun", ("possessive_3sg", "accusative_i")), readings("birisini"))
        self.assertIn(("bir", "noun", ("possessive_3pl", "dative_e")), readings("birilerine"))

    def test_uninflected_words_take_no_suffix_as_nouns(self):
        self.assertIn(("de", "noun", ()), readings("de"))
        self.assertFalse(any(root == "de" and pos == "noun" for root, pos, _ in readings("deniz")))
        self.assertIn(("de", "verb", ("pasttense_di",)), readings("dedi"))

    def test_inflected_forms_are_not_roots(self):
        for word in ("bana", "onlar", "bunlara", "kendisine", "misiniz", "hepimiz", "nereden"):
            self.assertNotIn((word, "noun", ()), readings(word), word)
            self.assertTrue(any(lexeme_of(root) for root, _pos, _chain in readings(word)), word)


class NoDuplicateReadingsTest(unittest.TestCase):
    WORDS = (
        "bana", "onu", "onun", "onlar", "benim", "hepsini", "kendine", "biri", "birini",
        "misiniz", "değilim", "sonra", "ile", "ve", "yaz", "kitaplarımız", "hesaplarımız",
        "geldi", "erbakanın", "chp",
    )

    def assert_unique(self, analyses):
        signatures = [(root, pos, tuple(s.name for s in chain)) for root, pos, chain, _ in analyses]
        self.assertEqual(len(signatures), len(set(signatures)), signatures)

    def test_decompose_never_repeats_a_reading(self):
        for word in self.WORDS:
            self.assert_unique(sfx.decompose_with_fallback(word))

    def test_root_guided_decomposition_keeps_both_parts_of_speech(self):
        analyses = sfx.decompose_with_root("yaz", "yaz")
        self.assert_unique(analyses)
        self.assertEqual({pos for _r, pos, _c, _f in analyses}, {"noun", "verb"})


class IrregularStemDisplayTest(unittest.TestCase):
    def test_reconstruction_starts_from_the_irregular_stem(self):
        decomp = next(d for d in sfx.decompose("bana") if d[0] == "ben")
        view = nlp.reconstruct_morphology("bana", decomp)
        self.assertEqual(view["formation_str"], "ben → bana")
        self.assertEqual(view["suffixes_str"], "a")
        self.assertEqual(nlp.build_suffix_log_info("bana", decomp)[0]["form"], "a")
        self.assertEqual(nlp.format_detailed_decomp("bana", decomp), "ben+dative_e_a")


class ClosedClassGoldTest(unittest.TestCase):
    def gold(self, surface, lemma, names):
        entry = build_closed_class_entry(surface, lemma, names)
        return entry["root"], [s["name"] for s in entry["suffixes"]]

    def test_gold_is_rooted_in_the_lexeme(self):
        self.assertEqual(self.gold("bana", "ben", ["dative_e"]), ("ben", ["dative_e"]))
        self.assertEqual(self.gold("onlar", "o", ["plural_ler"]), ("o", ["plural_ler"]))
        self.assertEqual(self.gold("mısınız", "mı", ["conjugation_2pl"]), ("mı", ["conjugation_2pl"]))

    def test_features_choose_between_readings(self):
        self.assertEqual(self.gold("benim", "ben", ["noun_compound"]), ("ben", ["noun_compound"]))

    def test_inherent_features_do_not_invent_suffixes(self):
        self.assertEqual(self.gold("biz", "biz", ["plural_ler"]), ("biz", []))

    def test_the_longest_lexeme_wins(self):
        self.assertEqual(
            self.gold("birbirlerine", "birbiri", ["plural_ler", "possessive_3pl", "dative_e"]),
            ("birbir", ["possessive_3pl", "dative_e"]),
        )

    def test_gold_uses_only_trainable_suffixes(self):
        self.assertIsNone(build_closed_class_entry("ve/veya", "ve/veya", []))

    def test_inflected_lemmas_resolve_to_their_lexeme(self):
        self.assertEqual(self.gold("hepsini", "hepsi", ["possessive_3pl", "accusative_i"]), ("hep", ["possessive_3sg", "accusative_i"]))
        self.assertEqual(self.gold("çoğu", "çoğu", ["possessive_3sg"]), ("çok", ["possessive_3sg"]))


class DerivedLexiconGuardTest(unittest.TestCase):
    def test_inflected_closed_class_forms_are_refused(self):
        import tools.build_derived_lexicons as builder
        for root in ("bunlara", "kendisine", "misiniz", "birbirlerine"):
            self.assertTrue(builder.is_inflected_closed_class_form(root), root)
        for root in ("bilgi", "birlik", "deniz"):
            self.assertFalse(builder.is_inflected_closed_class_form(root), root)


if __name__ == "__main__":
    unittest.main()
