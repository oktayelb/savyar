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
