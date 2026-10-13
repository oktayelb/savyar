import tempfile
import unittest
from pathlib import Path

import data.trmor2006_treebank.treebank_adapter as trmor2006
from app.nlp_pipeline import analyze_word, match_decompositions


SENTENCE = """<S>\t<S>+BSTag
gelsin\tgel+Verb+Pos+Imp+A3sg
gelmesin\tgel+Verb+Neg+Imp+A3sg
gelsinler\tgel+Verb+Pos+Imp+A3pl
ikiye\tiki+Num+Card^DB+Noun+Zero+A3sg+Pnon+Dat
ikişer\tiki+Num+Card^DB+Adj+Dist
ikinci\tikinci+Num+Ord
ikincilik\tikinci+Num+Ord^DB+Noun+Ness+A3sg+Pnon+Nom
20\t20+Num+Card
</S>\t</S>+ESTag
"""


def adapt(text):
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "mini.conllu"
        target = Path(tmp) / "out.jsonl"
        source.write_text(text, encoding="utf-8")
        trmor2006.adapt_treebank(str(source), str(target))
        import json
        return json.loads(target.read_text(encoding="utf-8").splitlines()[0])["words"]


class ImperativeAndNumeralTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.words = {w["word"]: w for w in adapt(SENTENCE)}

    def names(self, word):
        return [s["name"] for s in self.words[word]["suffixes"]]

    def test_third_person_imperatives_are_marked(self):
        self.assertEqual(self.names("gelsin"), ["conjugation_3sg"])
        self.assertEqual(self.names("gelmesin"), ["negative_me", "conjugation_3sg"])
        self.assertEqual(self.names("gelsinler"), ["conjugation_3pl"])

    def test_spelled_out_numerals_keep_their_case(self):
        self.assertEqual(self.words["ikiye"]["root"], "iki")
        self.assertEqual(self.names("ikiye"), ["dative_e"])

    def test_an_ordinal_lemma_is_not_marked_twice(self):
        self.assertEqual(self.names("ikinci"), ["ordinal_inci"])
        self.assertEqual(self.names("ikincilik"), ["ordinal_inci", "suitative_lik"])

    def test_every_gold_is_reachable(self):
        for word, entry in self.words.items():
            self.assertTrue(match_decompositions([entry], analyze_word(word)["decomps"]), word)


if __name__ == "__main__":
    unittest.main()
