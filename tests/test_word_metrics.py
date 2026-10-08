import unittest

from ml.ml_ranking_model import Trainer, SUFFIX_OFFSET
from util.decomposer import ALL_SUFFIXES


def token_id(suffix_name):
    for idx, suffix in enumerate(ALL_SUFFIXES):
        if suffix.name == suffix_name:
            return SUFFIX_OFFSET + idx
    raise AssertionError(f"Unknown suffix: {suffix_name}")


class MultisetSuffixMatchTest(unittest.TestCase):
    def test_a_shared_suffix_after_a_length_change_is_still_a_match(self):
        buckets = {}
        Trainer._update_suffix_metric_buckets(
            buckets,
            [token_id("adjectifier_dik"), token_id("possessive_3pl"), token_id("accusative_i")],
            [token_id("adjectifier_dik"), token_id("plural_ler"), token_id("possessive_3sg"), token_id("accusative_i")],
        )

        self.assertEqual(buckets["accusative_i"]["tp"], 1)
        self.assertEqual(buckets["accusative_i"]["fp"], 0)
        self.assertEqual(buckets["accusative_i"]["fn"], 0)
        self.assertEqual(buckets["possessive_3pl"]["fn"], 1)
        self.assertEqual(buckets["plural_ler"]["fp"], 1)
        self.assertEqual(buckets["possessive_3sg"]["fp"], 1)


def encoded(*suffix_names):
    return [(5, 0, 1)] + [(token_id(name), 0, idx + 2) for idx, name in enumerate(suffix_names)]


def ambiguous_result(candidates, scores, gold_idx):
    return {
        "status": "ambiguous",
        "candidates": candidates,
        "scores": scores,
        "gold_idx": gold_idx,
        "pred_idx": max(range(len(scores)), key=lambda i: scores[i]),
    }


class AmbiguousWordMetricsTest(unittest.TestCase):
    def test_metrics_come_from_each_word_against_all_its_candidates(self):
        from app.engine import WorkflowEngine

        results = [
            ambiguous_result([encoded("plural_ler"), encoded("conjugation_3pl")], [0.9, 0.1], 0),
            ambiguous_result(
                [encoded("possessive_3pl", "accusative_i"), encoded("plural_ler", "possessive_3sg", "accusative_i"), encoded("plural_ler")],
                [0.2, 0.5, 0.1],
                0,
            ),
            {"status": "single", "candidates": [encoded()], "scores": [], "gold_idx": 0, "pred_idx": 0},
        ]
        metrics = WorkflowEngine._ambiguous_word_metrics(results)

        self.assertEqual(metrics["words"], 2)
        self.assertAlmostEqual(metrics["word_acc"], 0.5)
        self.assertAlmostEqual(metrics["top2_acc"], 1.0)
        self.assertAlmostEqual(metrics["mean_candidates"], 2.5)
        self.assertAlmostEqual(metrics["margin"], (0.8 + (0.2 - 0.5)) / 2)
        self.assertEqual(metrics["suffix_metrics"]["accusative_i"]["tp"], 1)
        self.assertEqual(metrics["suffix_metrics"]["possessive_3pl"]["fn"], 1)
        self.assertEqual(metrics["suffix_metrics"]["possessive_3sg"]["fp"], 1)



if __name__ == "__main__":
    unittest.main()
