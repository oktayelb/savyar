import unittest

import util.decomposer as sfx


def readings(word):
    return {(root, tuple(s.name for s in chain)) for root, _pos, chain, _ in sfx.decompose(word)}


def roots(word):
    return {root for root, _chain in readings(word)}


def readings_with_pos(word):
    return {(root, pos, tuple(s.name for s in chain)) for root, pos, chain, _ in sfx.decompose(word)}


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

    def test_only_pronouns_take_a_first_or_second_person_ending_after_the_genitive(self):
        self.assertIn(("ben", ("noun_compound", "conjugation_2sg")), readings("benimsin"))
        self.assertIn(("sen", ("noun_compound", "conjugation_1sg")), readings("seninim"))
        self.assertNotIn(("kitap", ("noun_compound", "conjugation_1pl")), readings("kitabınız"))
        self.assertNotIn(("düş", ("noun_compound", "if_se", "conjugation_1sg")), readings("düşünsem"))

    def test_any_genitive_takes_a_third_person_predicate(self):
        self.assertIn(("biz", ("noun_compound", "nounaorist_dir")), readings("bizimdir"))
        self.assertIn(("firma", ("plural_ler", "noun_compound", "if_se")), readings("firmalarınsa"))

    def test_pronouns_do_not_take_a_verbifier(self):
        self.assertEqual(roots("olan"), {"ol"})
        self.assertNotIn("o", roots("olarak"))
        self.assertNotIn(("ne", ("applicative_le", "factative_ir")), readings("neler"))

    def test_personal_and_demonstrative_pronouns_take_no_possessive(self):
        self.assertEqual(readings("bizi"), {("biz", ("accusative_i",))})
        self.assertEqual(readings("beni"), {("ben", ("accusative_i",))})
        self.assertEqual(readings("senin"), {("sen", ("noun_compound",))})
        self.assertNotIn(("o", ("plural_ler", "possessive_3sg")), readings("onları"))
        self.assertNotIn(("bu", ("plural_ler", "possessive_3sg")), readings("bunları"))
        self.assertNotIn(("siz", ("possessive_3pl",)), readings("sizleri"))

    def test_other_pronouns_and_nouns_built_on_one_take_a_possessive(self):
        self.assertIn(("bura", ("possessive_3sg",)), readings("burası"))
        self.assertIn(("kendi", ("possessive_3sg",)), readings("kendisi"))
        self.assertIn(("ben", ("suitative_lik", "possessive_3sg")), readings("benliği"))

    def test_nouns_built_on_a_pronoun_still_do(self):
        self.assertIn(
            ("kim", ("suitative_lik", "privative_siz", "applicative_le", "reflexive_is", "infinitive_me")),
            readings("kimliksizleşme"),
        )


class PersonEndingTest(unittest.TestCase):
    def test_imperative_endings_need_a_verb_stem(self):
        self.assertIn(("yaz", "verb", ("conjugation_2pl",)), readings_with_pos("yazın"))
        self.assertNotIn(("yaz", "noun", ("conjugation_2pl",)), readings_with_pos("yazın"))
        self.assertNotIn(("ol", "noun", ("conjugation_3sg",)), readings_with_pos("olsun"))
        self.assertIn(("baş", ("applicative_le", "conjugation_2pl")), readings("başlayın"))
        self.assertIn(("gel", ("negative_me", "conjugation_3sg")), readings("gelmesin"))
        self.assertIn(("ol", ("conjugation_3pl",)), readings("olsunlar"))

    def test_imperative_endings_do_not_follow_a_tense(self):
        self.assertEqual(
            readings("geliyorsun"),
            {("gel", ("continuous_iyor", "conjugation_2sg"))},
        )
        self.assertIn(("iste", ("continuous_iyor", "conjugation_2pl")), readings("istiyorsunuz"))

    def test_second_person_singular_is_never_in(self):
        for word in ("yazın", "bakın", "evin"):
            chains = {chain for _root, chain in readings(word)}
            self.assertFalse(any("conjugation_2sg" in chain for chain in chains), word)
        self.assertIn(("gel", ("pasttense_di", "conjugation_2sg")), readings("geldin"))


if __name__ == "__main__":
    unittest.main()
