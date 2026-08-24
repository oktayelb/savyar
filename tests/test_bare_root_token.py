import unittest

import app.nlp_pipeline as nlp
import util.decomposer as sfx
from ml.ml_ranking_model import (
    SPECIAL_BOS,
    SPECIAL_EOS,
    SPECIAL_MASK,
    SPECIAL_NO_SUFFIX,
    SPECIAL_PAD,
    SPECIAL_WORD_SEP,
    SUFFIX_OFFSET,
    Trainer,
    _chain_tokens,
    build_sentence_sequence,
)

SUFFIXED = [(SUFFIX_OFFSET + 3, 4, 1)]


class NoSuffixTokenTest(unittest.TestCase):
    def test_token_is_distinct_from_the_other_specials(self):
        specials = (SPECIAL_PAD, SPECIAL_WORD_SEP, SPECIAL_BOS, SPECIAL_MASK, SPECIAL_EOS)
        self.assertNotIn(SPECIAL_NO_SUFFIX, specials)
        self.assertLess(SPECIAL_NO_SUFFIX, SUFFIX_OFFSET)

    def test_no_suffix_id_can_never_collide_with_a_real_suffix(self):
        ids = [nlp._SUFFIX_TO_ID[s.name] for s in sfx.ALL_SUFFIXES]
        self.assertNotIn(SPECIAL_NO_SUFFIX, ids)
        self.assertTrue(all(i >= SUFFIX_OFFSET for i in ids))

    def test_a_bare_word_emits_exactly_one_no_suffix_token(self):
        s, g, p = _chain_tokens([[]])
        self.assertEqual(s, [SPECIAL_NO_SUFFIX, SPECIAL_WORD_SEP])
        self.assertEqual(len(s), len(g))
        self.assertEqual(len(s), len(p))

    def test_one_token_per_bare_word(self):
        s, _, _ = _chain_tokens([[], SUFFIXED, []])
        self.assertEqual(s.count(SPECIAL_NO_SUFFIX), 2)

    def test_bare_and_suffixed_candidates_differ_in_content(self):
        # The point of the token: the two readings of one word are no longer
        # distinguished only by the suffixed one being longer.
        bare = build_sentence_sequence([[]])[0]
        suffixed = build_sentence_sequence([SUFFIXED])[0]
        self.assertEqual(len(bare), len(suffixed))
        self.assertNotEqual(bare, suffixed)

    def test_it_occupies_the_first_suffix_slot(self):
        _, _, p = _chain_tokens([[]])
        self.assertEqual(p[0], 1)

    def test_suffix_metrics_ignore_it(self):
        # Keeps suffix precision/recall/F1 comparable with earlier runs.
        seq = build_sentence_sequence([[], SUFFIXED])
        self.assertNotIn(SPECIAL_NO_SUFFIX, Trainer._morph_tokens_from_sequence(seq))


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
        self.assertEqual(gold_chains[0], [])

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
