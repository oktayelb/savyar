import unittest

import util.decomposer as sfx


def readings(word):
    return {(root, tuple(s.name for s in chain)) for root, _pos, chain, _ in sfx.decompose(word)}


def roots(word):
    return {root for root, _chain in readings(word)}


class CausativeTest(unittest.TestCase):
    def test_listed_roots_take_the_ir_causative(self):
        self.assertIn(("geç", ("active_ir", "pasttense_di")), readings("geçirdi"))
        self.assertIn(("vazgeç", ("active_ir", "pasttense_di")), readings("vazgeçirdi"))

    def test_other_roots_do_not_take_the_ir_causative(self):
        for word in ("bilirdi", "otururdu", "okurdu", "sanırım"):
            chains = {chain for _root, chain in readings(word)}
            self.assertFalse(any("active_ir" in chain for chain in chains), word)

    def test_listed_roots_take_the_ar_causative(self):
        self.assertIn(("çık", ("active_er", "pasttense_di")), readings("çıkardı"))
        self.assertIn(("git", ("active_er",)), readings("gider"))

    def test_other_roots_do_not_take_the_ar_causative(self):
        self.assertEqual(readings("takardı"), {("tak", ("factative_ir", "pasttense_noundi"))})

    def test_causative_never_follows_another_suffix(self):
        chains = {chain for _root, chain in readings("insanların")}
        self.assertFalse(any("active_ir" in chain or "active_er" in chain for chain in chains))


class AoristTest(unittest.TestCase):
    def test_monosyllables_outside_the_list_take_ar(self):
        self.assertIn(("bit", ("factative_ir",)), readings("biter"))
        self.assertNotIn(("bit", ("factative_ir", "pasttense_noundi")), readings("bitirdi"))

    def test_listed_monosyllables_take_ir(self):
        self.assertIn(("gel", ("factative_ir",)), readings("gelir"))
        self.assertEqual(readings("geler"), set())

    def test_polysyllabic_stems_keep_both_vowels(self):
        self.assertIn(("otur", ("factative_ir",)), readings("oturur"))
        self.assertIn(("vazgeç", ("factative_ir",)), readings("vazgeçer"))


class VowelDropTest(unittest.TestCase):
    def test_listed_roots_drop_their_vowel(self):
        self.assertIn(("ağız", ("possessive_3sg",)), readings("ağzı"))
        self.assertIn(("kayıp", ("possessive_3sg",)), readings("kaybı"))
        self.assertIn(("ayır", ("passive_il", "pasttense_di")), readings("ayrıldı"))

    def test_other_roots_keep_their_vowel(self):
        self.assertNotIn("bulut", roots("buldun"))
        self.assertNotIn("yapıt", roots("yaptın"))
        self.assertNotIn("karış", roots("karşı"))

    def test_a_restored_root_needs_a_suffix(self):
        self.assertNotIn("misina", roots("misin"))

    def test_listed_roots_elide_their_final_vowel(self):
        self.assertIn(("nere", ("locative_de",)), readings("nerde"))
        self.assertIn(("ora", ("ablative_den",)), readings("ordan"))


class RootRestrictionTest(unittest.TestCase):
    def test_degil_takes_only_predicate_suffixes(self):
        self.assertEqual(
            {chain for root, chain in readings("değilim") if root == "değil"},
            {("conjugation_1sg",)},
        )
        self.assertIn(("değil", ("pasttense_noundi",)), readings("değildi"))
        self.assertIn(("değil", ("when_ken",)), readings("değilken"))

    def test_only_pronouns_take_the_comitative_after_the_genitive(self):
        self.assertIn(("o", ("noun_compound", "confactuous_le")), readings("onunla"))
        self.assertIn(("ben", ("noun_compound", "confactuous_le")), readings("benimle"))
        self.assertNotIn(("on", ("noun_compound", "confactuous_le")), readings("onunla"))
        self.assertNotIn(
            ("saç", ("plural_ler", "noun_compound", "confactuous_le")),
            readings("saçlarınla"),
        )


if __name__ == "__main__":
    unittest.main()
