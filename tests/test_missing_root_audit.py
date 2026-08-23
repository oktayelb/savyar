import importlib.util
import unittest
from pathlib import Path

import util.word_methods as wrd

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "audit_missing_roots", REPO_ROOT / "tools" / "audit_missing_roots.py"
)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def example(word, suffixes, final_pos="noun"):
    return {"word": word, "suffixes": list(suffixes), "final_pos": final_pos}


class LexiconCandidateTest(unittest.TestCase):
    def test_plain_turkish_roots_are_candidates(self):
        for root in ("kullan", "türkiye", "hâl", "güney-doğu"):
            self.assertTrue(audit.is_lexicon_candidate(root), root)

    def test_foreign_letters_are_candidates(self):
        # Proper nouns in the treebanks use letters outside the Turkish alphabet.
        for root in ("washington", "new", "x", "á"):
            self.assertTrue(audit.is_lexicon_candidate(root), root)

    def test_annotation_defects_are_rejected(self):
        for root in ("*unknown*", "dr.", "1996da", "hayat(ı)", "20.00de", ""):
            self.assertFalse(audit.is_lexicon_candidate(root), root)


class ResolvabilityTest(unittest.TestCase):
    def test_dictionary_roots_resolve(self):
        self.assertTrue(audit.is_resolvable("gel"))     # verbs.txt
        self.assertTrue(audit.is_resolvable("kitap"))   # words.txt

    def test_closed_class_and_unsuffixable_resolve(self):
        self.assertTrue(audit.is_resolvable("ve"))
        self.assertTrue(audit.is_resolvable(next(iter(wrd.UNSUFFIXABLE_SET))))

    def test_derived_lexicon_roots_resolve(self):
        # nouns_derived.txt / verbs_derived.txt load into the same sets.
        self.assertTrue(audit.is_resolvable("kullan"))

    def test_absent_roots_do_not_resolve(self):
        self.assertFalse(audit.is_resolvable("zzzyok"))


class CaseMismatchTest(unittest.TestCase):
    def setUp(self):
        self.index = audit.build_case_folded_index()

    def test_lowercased_proper_noun_finds_its_capitalised_entry(self):
        # words.txt stores proper nouns capitalised, the pipeline tr_lower()s
        # everything, so these are not missing entries at all.
        self.assertEqual(audit.lexicon_case_variants("ankara", self.index), ["Ankara"])

    def test_genuinely_absent_root_has_no_variant(self):
        self.assertEqual(audit.lexicon_case_variants("kullan", self.index), [])

    def test_entry_does_not_match_itself(self):
        self.assertEqual(audit.lexicon_case_variants("kitap", self.index), [])


class PosProbeTest(unittest.TestCase):
    def test_verb_chain_is_only_reachable_from_a_verb_start(self):
        self.assertTrue(audit.chain_is_reachable("kullan", "kullanıyor", ["continuous_iyor"], "verb"))
        self.assertFalse(audit.chain_is_reachable("kullan", "kullanıyor", ["continuous_iyor"], "noun"))

    def test_noun_chain_is_only_reachable_from_a_noun_start(self):
        self.assertTrue(audit.chain_is_reachable("hükümet", "hükümetler", ["plural_ler"], "noun"))
        self.assertFalse(audit.chain_is_reachable("hükümet", "hükümetler", ["plural_ler"], "verb"))

    def test_probe_routes_a_verb_root_to_verbs_txt(self):
        pos, evidence = audit.probe_root_pos("kullan", [example("kullanıyor", ["continuous_iyor"], "verb")])
        self.assertEqual((pos, evidence), (["verb"], "chain"))
        self.assertEqual(audit.suggested_file(pos), audit.VERB_FILE)

    def test_probe_routes_a_noun_root_to_words_txt(self):
        pos, evidence = audit.probe_root_pos("hükümet", [example("hükümetler", ["plural_ler"])])
        self.assertEqual((pos, evidence), (["noun"], "chain"))
        self.assertEqual(audit.suggested_file(pos), audit.NOUN_FILE)

    def test_bare_root_is_attributed_by_its_final_pos(self):
        # No suffix means no morphological evidence; the annotation decides.
        pos, evidence = audit.probe_root_pos("hükümet", [example("hükümet", [])])
        self.assertEqual((pos, evidence), (["noun"], "final_pos"))

    def test_spanning_tier_places_a_root_whose_gold_chain_is_lossy(self):
        # The gold chain of "tartışmalardan" omits the -ma nominaliser, so no
        # start POS replays it - but a verb start still carries the root across
        # the surface, which is enough to file it.
        pos, evidence = audit.probe_root_pos(
            "tartış", [example("tartışmalardan", ["plural_ler", "ablative_den"])]
        )
        self.assertEqual((pos, evidence), (["verb"], "spanning"))
        self.assertEqual(audit.suggested_file(pos), audit.VERB_FILE)

    def test_root_nothing_can_place_yields_no_target_file(self):
        pos, evidence = audit.probe_root_pos("zzz", [example("zzzqqq", ["plural_ler"])])
        self.assertEqual((pos, evidence), ([], "unreplayable"))
        self.assertIsNone(audit.suggested_file(pos))


if __name__ == "__main__":
    unittest.main()
