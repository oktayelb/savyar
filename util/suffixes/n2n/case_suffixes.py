import util.word_methods as wrd
from util.suffix import Suffix, Type, SuffixGroup
## COMPLETE
class CaseSuffix(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.NOUN,
                makes=Type.NOUN,
                has_major_harmony=True, 
                has_minor_harmony=None,  # Set to None to detect if the user passed a value
                needs_y_buffer=False, 
                group=SuffixGroup.CASE, 
                is_unique=False):
        
        # Dynamic default assignment for minor harmony
        if has_minor_harmony is None:
            # If the suffix contains any narrow vowel, it defaults to having minor harmony
            if any(vowel in suffix for vowel in ['ı', 'i', 'u', 'ü']): # only i is enough bc of the standart narrow front vowel converntion
                has_minor_harmony = True
            else:
                has_minor_harmony = False
        super().__init__(
            name=name,
            suffix=suffix,
            comes_to=Type.NOUN,
            makes=Type.NOUN,
            form_function=None, # Force the use of the overridden _default_form
            has_major_harmony=has_major_harmony,
            has_minor_harmony=has_minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )

    @staticmethod
    def _harmonized_base(word, suffix_obj):
        base = suffix_obj.suffix
        base = Suffix._apply_major_harmony(word, base, suffix_obj.has_major_harmony)
        base = Suffix._apply_minor_harmony(word, base, suffix_obj.has_minor_harmony)
        return Suffix._apply_consonant_hardening(word, base)

    @staticmethod
    def _looks_like_possessive_3_surface(word):
        if not word:
            return False

        candidates = []
        if len(word) > 1 and word[-1] in ["ı", "i", "u", "ü"]:
            candidates.append(word[:-1])
        if len(word) > 2 and word[-2] == "s" and word[-1] in ["ı", "i", "u", "ü"]:
            candidates.append(word[:-2])

        for stem in candidates:
            if wrd.can_be_noun(stem):
                return True
            if stem.endswith(("lar", "ler")) and wrd.can_be_noun(stem[:-3]):
                return True

        return False

    @staticmethod
    def _default_form(word, suffix_obj):
        """
        Direct case forms for bare nominal stems.

        Vowel-initial cases cannot attach raw to vowel-final stems: başka+a and
        başka+na are not direct dative forms; başka+ya is. The pronominal n is
        emitted only when the current accumulated surface already looks like a
        third-person possessive form: evi-n-e, kapısı-n-dan, evleri-n-e.
        """
        base = CaseSuffix._harmonized_base(word, suffix_obj)
        has_possessive_n = CaseSuffix._looks_like_possessive_3_surface(word)

        if word and base and word[-1] in wrd.VOWELS and base[0] in wrd.VOWELS:
            candidates = []
            if suffix_obj.needs_y_buffer:
                candidates.append("y" + base)
            else:
                candidates.append("n" + base)

            if has_possessive_n:
                n_form = "n" + base
                if n_form not in candidates:
                    candidates.append(n_form)

            return candidates

        if word and base and word[-1] in wrd.VOWELS and has_possessive_n:
            return [base, "n" + base]

        return [base]


noun_compound = CaseSuffix("noun_compound"  , "in") # köy ağzında needs_y_buffer doğru
accusative_i  = CaseSuffix("accusative_i"   , "i", needs_y_buffer=True)
ablative_den  = CaseSuffix("ablative_den"   , "den")
##sorunlu
dative_e      = CaseSuffix("dative_e"       , "e", needs_y_buffer=True)

locative_de   = CaseSuffix("locative_de"    , "de")


CASESUFFIX = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]
