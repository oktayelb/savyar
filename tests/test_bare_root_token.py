import unittest

import app.nlp_pipeline as nlp
import util.decomposer as sfx
from ml.ml_ranking_model import (
    SPECIAL_BOS,
    SPECIAL_EOS,
    SPECIAL_MASK,
    SPECIAL_PAD,
    SPECIAL_ROOT_NOUN,
    SPECIAL_ROOT_VERB,
    SPECIAL_WORD_SEP,
    SUFFIX_OFFSET,
    Trainer,
    _chain_tokens,
    build_sentence_sequence,
)

NOUN = nlp.root_token("elma", "noun")
VERB = nlp.root_token("gel", "verb")
SUFFIXED = [NOUN, (SUFFIX_OFFSET + 3, 4, 2)]


class RootTokenTest(unittest.TestCase):
    def test_tokens_are_distinct_from_the_other_specials(self):
        specials = (SPECIAL_PAD, SPECIAL_WORD_SEP, SPECIAL_BOS, SPECIAL_MASK, SPECIAL_EOS)
        self.assertNotIn(SPECIAL_ROOT_NOUN, specials)
        self.assertNotIn(SPECIAL_ROOT_VERB, specials)
        self.assertNotEqual(SPECIAL_ROOT_NOUN, SPECIAL_ROOT_VERB)
        self.assertLess(max(SPECIAL_ROOT_NOUN, SPECIAL_ROOT_VERB), SUFFIX_OFFSET)

    def test_root_ids_can_never_collide_with_a_real_suffix(self):
        ids = [nlp._SUFFIX_TO_ID[s.name] for s in sfx.ALL_SUFFIXES]
        self.assertNotIn(SPECIAL_ROOT_NOUN, ids)
        self.assertNotIn(SPECIAL_ROOT_VERB, ids)
        self.assertTrue(all(i >= SUFFIX_OFFSET for i in ids))

    def test_every_word_opens_with_its_root_token(self):
        dative = {s.name: s for s in sfx.ALL_SUFFIXES}["dative_e"]
        self.assertEqual(nlp.encode_suffix_chain([], "elma", "noun"), [NOUN])
        self.assertEqual(nlp.encode_suffix_chain([], "gel", "verb"), [VERB])
        self.assertEqual(nlp.encode_suffix_chain([dative], "elma", "noun")[0], NOUN)

    def test_suffixes_follow_the_root(self):
        encoded = nlp.encode_suffix_names([{"name": "dative_e", "makes": "NOUN"}], "elma", "noun")
        self.assertEqual(encoded[0], NOUN)
        self.assertEqual(encoded[1][2], 2)

    def test_a_bare_word_emits_its_root_token_and_nothing_else(self):
        s, g, p = _chain_tokens([[NOUN]])
        self.assertEqual(s, [SPECIAL_ROOT_NOUN, SPECIAL_WORD_SEP])
        self.assertEqual(len(s), len(g))
        self.assertEqual(len(s), len(p))

    def test_one_root_token_per_word(self):
        s, _, _ = _chain_tokens([[NOUN], SUFFIXED, [VERB]])
        self.assertEqual(s.count(SPECIAL_ROOT_NOUN) + s.count(SPECIAL_ROOT_VERB), 3)

    def test_noun_and_verb_readings_of_a_bare_word_differ(self):
        analysis = nlp.analyze_word("yaz")
        chains = [tuple(chain) for chain in analysis["encoded_chains"]]
        self.assertIn((NOUN,), chains)
        self.assertIn((VERB,), chains)

    def test_suffix_metrics_ignore_root_tokens(self):
        seq = build_sentence_sequence([[NOUN], SUFFIXED, [VERB]])
        morph = Trainer._morph_tokens_from_sequence(seq)
        self.assertNotIn(SPECIAL_ROOT_NOUN, morph)
        self.assertNotIn(SPECIAL_ROOT_VERB, morph)
        self.assertEqual(morph, [SUFFIX_OFFSET + 3])


class FallbackRootPosTest(unittest.TestCase):
    def names(self, *names):
        return [{"name": name} for name in names]

    def test_a_verb_only_first_suffix_means_a_verb_root(self):
        self.assertEqual(nlp.root_pos_from_suffix_names(self.names("pasttense_di", "conjugation_1sg")), "verb")
        self.assertEqual(nlp.root_pos_from_suffix_names(self.names("infinitive_me")), "verb")

    def test_anything_else_defaults_to_a_noun_root(self):
        self.assertEqual(nlp.root_pos_from_suffix_names(self.names("dative_e")), "noun")
        self.assertEqual(nlp.root_pos_from_suffix_names(self.names("conjugation_1sg")), "noun")
        self.assertEqual(nlp.root_pos_from_suffix_names([]), "noun")


class NoHardcodedPriorTest(unittest.TestCase):
    def test_the_config_knob_is_gone(self):
        from ml.config import config
        self.assertFalse(hasattr(config, "bare_root_prior_logprob"))

    def test_scoring_no_longer_adds_a_constant(self):
        import inspect
        for method in (Trainer.score_candidates, Trainer.score_sentence_chains):
            self.assertNotIn("prior", inspect.getsource(method))


if __name__ == "__main__":
    unittest.main()


class GoldChainsIncludeBareRootsTest(unittest.TestCase):
    """The token is only learnable if correct bare roots reach the gold."""

    @classmethod
    def setUpClass(cls):
        from app.engine import WorkflowEngine
        cls.parts = WorkflowEngine._candidate_parts_from_word_entries

    def word(self, surface, suffixes):
        return {"word": surface, "root": surface, "suffixes": suffixes, "final_pos": "noun"}

    def test_a_suffixless_word_is_kept(self):
        parts = self.parts([self.word("kitap", [])])
        self.assertIsNotNone(parts, "a bare-root word must not be dropped from the sequence")
        gold_chains, _cands, _idx, word_count = parts
        self.assertEqual(word_count, 1)
        self.assertEqual(gold_chains[0], [NOUN])

    def test_gold_root_pos_comes_from_the_matched_decomposition(self):
        entry = {
            "word": "yazdım",
            "root": "yaz",
            "suffixes": [
                {"name": "pasttense_di", "makes": "NOUN"},
                {"name": "conjugation_1sg", "makes": "NOUN"},
            ],
            "final_pos": "noun",
        }
        gold_chains, _cands, _idx, _count = self.parts([entry])
        self.assertEqual(gold_chains[0][0], VERB)

    def test_a_sentence_keeps_all_of_its_words(self):
        entries = [
            self.word("kitap", []),
            self.word("kitaplar", [{"name": "plural_ler", "makes": "NOUN"}]),
            self.word("ev", []),
        ]
        _gold, _cands, _idx, word_count = self.parts(entries)
        self.assertEqual(word_count, 3, "dropping bare roots used to leave holes mid-sentence")

    def test_a_chain_the_tables_cannot_encode_still_rejects_the_sentence(self):
        # _entries_to_sequences catches this and counts the entry as skipped.
        with self.assertRaises(ValueError):
            self.parts([self.word("x", [{"name": "no_such_suffix", "makes": "NOUN"}])])


class RootGuidedDecompositionTest(unittest.TestCase):
    """Annotated data names the lemma; the lexicon need not know it."""

    def test_a_root_the_lexicon_lacks_still_yields_chains(self):
        import util.word_methods as wrd
        self.assertFalse(wrd.exists("erbakan"))
        chains = {tuple(s.name for s in c) for _r, _p, c, _f in sfx.decompose_with_root("erbakanın", "erbakan")}
        self.assertIn(("noun_compound",), chains)
        self.assertGreater(len(chains), 1, "the word should arrive with alternatives, not one answer")

    def test_every_analysis_uses_the_given_root(self):
        for _r, _p, _c, _f in sfx.decompose_with_root("kullandığı", "kullan"):
            self.assertEqual(_r, "kullan")

    def test_chains_must_still_span_the_surface(self):
        # Nothing is invented: a chain that cannot reach the end is not offered.
        self.assertEqual(sfx.decompose_with_root("erbakanın", "zzz"), [])

    def test_a_softened_stem_is_spliced_onto_the_surface(self):
        chains = {tuple(s.name for s in c) for _r, _p, c, _f in sfx.decompose_with_root("kitabı", "kitap")}
        self.assertTrue(chains, "kitap + accusative should be reachable from the lemma")

    def test_empty_inputs_are_safe(self):
        self.assertEqual(sfx.decompose_with_root("", "kitap"), [])
        self.assertEqual(sfx.decompose_with_root("kitabı", ""), [])


class PassiveAllomorphTest(unittest.TestCase):
    """-Il after a consonant, -n after a vowel, -In after l."""

    def setUp(self):
        self.passive = {s.name: s for s in sfx.ALL_SUFFIXES}["passive_il"]

    def forms(self, stem):
        return self.passive.form(stem, current_chain=[])

    def test_vowel_final_stems_take_n(self):
        for stem in ("bekle", "oku", "ye", "dışla"):
            self.assertIn("n", self.forms(stem), stem)

    def test_l_final_stems_take_in(self):
        self.assertIn("ın", self.forms("al"))
        self.assertIn("un", self.forms("bul"))

    def test_consonant_stems_still_take_il(self):
        self.assertIn("ıl", self.forms("yap"))
        self.assertIn("ül", self.forms("gör"))

    def test_the_passive_reading_beats_nothing_being_available(self):
        # "beklenen" used to be analysable only as bekle + reflexive_in.
        chains = {tuple(s.name for s in c) for _r, _p, c, _f in sfx.decompose_with_root("beklenen", "bekle")}
        self.assertIn(("passive_il", "factative_en"), chains)
