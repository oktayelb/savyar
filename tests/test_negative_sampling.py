import unittest

from app.engine import WorkflowEngine
from ml.config import config


def chain(*token_ids):
    """A minimal encoded chain: (token_id, group_id, position_in_word)."""
    return [(tid, 1, i + 1) for i, tid in enumerate(token_ids)]


def sentence(word_count, candidates_per_word, first_token=10):
    """Gold chains plus candidate lists where every word is equally ambiguous."""
    gold_chains = []
    candidate_lists = []
    gold_indices = []
    for word_idx in range(word_count):
        base = first_token + word_idx * 100
        candidates = [chain(base + c) for c in range(candidates_per_word)]
        gold_chains.append(candidates[0])
        candidate_lists.append(candidates)
        gold_indices.append(0)
    return gold_chains, candidate_lists, gold_indices


def perturbed_word_indices(gold_chains, negatives):
    """Which word each negative replaced."""
    indices = []
    for neg in negatives:
        differing = [i for i, c in enumerate(neg) if c is not gold_chains[i]]
        assert len(differing) == 1, "a single-substitution negative must differ in exactly one word"
        indices.append(differing[0])
    return indices


class NegativeSamplingTest(unittest.TestCase):
    def setUp(self):
        self.negatives_for = WorkflowEngine._single_substitution_negatives

    # --- fairness across word positions -----------------------------------

    def test_every_ambiguous_word_gets_a_negative_when_budget_allows(self):
        gold, cands, gold_idx = sentence(word_count=12, candidates_per_word=6)
        negatives = self.negatives_for(gold, cands, gold_idx, limit=12)

        self.assertEqual(len(negatives), 12)
        self.assertEqual(set(perturbed_word_indices(gold, negatives)), set(range(12)))

    def test_late_words_are_reached_before_early_words_repeat(self):
        # Word 0 alone could supply 19 negatives; depth-first would stop there.
        gold, cands, gold_idx = sentence(word_count=10, candidates_per_word=20)
        negatives = self.negatives_for(gold, cands, gold_idx, limit=10)

        touched = perturbed_word_indices(gold, negatives)
        self.assertEqual(len(set(touched)), 10, "one pass must cover every word before repeating")

    def test_budget_smaller_than_sentence_still_spreads(self):
        gold, cands, gold_idx = sentence(word_count=8, candidates_per_word=9)
        negatives = self.negatives_for(gold, cands, gold_idx, limit=4)

        touched = perturbed_word_indices(gold, negatives)
        self.assertEqual(len(negatives), 4)
        self.assertEqual(len(set(touched)), 4, "a short budget must still hit four distinct words")

    def test_rotation_offset_moves_the_starting_word_between_sentences(self):
        # With a budget below the word count, a fixed start would always pick
        # word 0. Across many distinct sentences the start must move around.
        starts = set()
        for variant in range(40):
            gold, cands, gold_idx = sentence(
                word_count=10, candidates_per_word=4, first_token=10 + variant,
            )
            negatives = self.negatives_for(gold, cands, gold_idx, limit=3)
            starts.add(perturbed_word_indices(gold, negatives)[0])
        self.assertGreater(len(starts), 1, "round-robin must not always start at word 0")

    def test_first_negative_is_not_always_the_first_candidate(self):
        chosen = set()
        for variant in range(40):
            gold, cands, gold_idx = sentence(
                word_count=1, candidates_per_word=4, first_token=10 + variant,
            )
            negative = self.negatives_for(gold, cands, gold_idx, limit=1)[0]
            chosen.add(cands[0].index(negative[0]))
        self.assertEqual(chosen, {1, 2, 3}, "every wrong reading must get its turn as the negative")

    def test_rotation_offset_is_deterministic(self):
        gold, cands, gold_idx = sentence(word_count=9, candidates_per_word=5)
        first = perturbed_word_indices(gold, self.negatives_for(gold, cands, gold_idx, limit=4))
        second = perturbed_word_indices(gold, self.negatives_for(gold, cands, gold_idx, limit=4))
        self.assertEqual(first, second, "preprocessing cache validity depends on this being stable")

    # --- exhaustion and edge cases ----------------------------------------

    def test_exhausted_words_are_skipped_and_the_budget_is_still_spent(self):
        # Word 0 has one alternative, word 1 has five. Budget of 5 must be filled
        # rather than stopping when word 0 runs dry.
        gold = [chain(10), chain(20)]
        cands = [[chain(10), chain(11)], [chain(20)] + [chain(20 + c) for c in range(1, 6)]]
        negatives = self.negatives_for(gold, cands, [0, 0], limit=5)

        self.assertEqual(len(negatives), 5)
        touched = perturbed_word_indices(gold, negatives)
        self.assertEqual(touched.count(0), 1)
        self.assertEqual(touched.count(1), 4)

    def test_unambiguous_words_are_never_selected(self):
        gold = [chain(10), chain(20), chain(30)]
        cands = [[chain(10)], [chain(20), chain(21), chain(22)], [chain(30)]]
        negatives = self.negatives_for(gold, cands, [0, 0, 0], limit=8)

        self.assertEqual(set(perturbed_word_indices(gold, negatives)), {1})
        self.assertEqual(len(negatives), 2, "only the two non-gold candidates exist")

    def test_no_ambiguity_yields_no_negatives(self):
        gold, cands, gold_idx = sentence(word_count=5, candidates_per_word=1)
        self.assertEqual(self.negatives_for(gold, cands, gold_idx, limit=5), [])

    def test_zero_budget_yields_no_negatives(self):
        gold, cands, gold_idx = sentence(word_count=5, candidates_per_word=4)
        self.assertEqual(self.negatives_for(gold, cands, gold_idx, limit=0), [])

    # --- deduplication -----------------------------------------------------

    def test_candidate_matching_the_gold_signature_is_not_used_as_a_negative(self):
        # Two roots, identical suffix chain: the encoded sequences are identical,
        # so this cannot be ranked below the gold.
        gold = [chain(10), chain(20)]
        cands = [[chain(10), chain(10)], [chain(20), chain(21)]]
        negatives = self.negatives_for(gold, cands, [0, 0], limit=5)

        self.assertEqual(len(negatives), 1)
        self.assertEqual(perturbed_word_indices(gold, negatives), [1])

    def test_duplicate_negatives_are_emitted_once(self):
        gold = [chain(10), chain(20)]
        cands = [[chain(10), chain(11), chain(11)], [chain(20)]]
        negatives = self.negatives_for(gold, cands, [0, 0], limit=5)

        self.assertEqual(len(negatives), 1)

    # --- budget ------------------------------------------------------------

    def test_budget_scales_with_the_number_of_ambiguous_words(self):
        _, few, _ = sentence(word_count=2, candidates_per_word=4)
        _, many, _ = sentence(word_count=12, candidates_per_word=4)

        self.assertEqual(
            WorkflowEngine._negative_budget(few),
            config.max_negative_candidates,
            "short sentences keep the configured floor",
        )
        self.assertGreater(WorkflowEngine._negative_budget(many), config.max_negative_candidates)

    def test_budget_respects_the_ceiling(self):
        _, huge, _ = sentence(word_count=200, candidates_per_word=4)
        self.assertEqual(
            WorkflowEngine._negative_budget(huge),
            max(config.max_negative_candidates, config.max_negative_candidates_cap),
        )

    def test_budget_ignores_unambiguous_words(self):
        gold, cands, gold_idx = sentence(word_count=30, candidates_per_word=1)
        cands[0] = [chain(1), chain(2)]
        self.assertEqual(WorkflowEngine._negative_budget(cands), config.max_negative_candidates)


if __name__ == "__main__":
    unittest.main()
