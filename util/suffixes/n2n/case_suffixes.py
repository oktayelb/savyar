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
    def _default_form(word, suffix_obj, current_chain=None):
        """
        Direct case forms for bare nominal stems.

        Vowel-initial cases cannot attach raw to vowel-final stems: başka+a and
        başka+na are not direct dative forms; başka+ya is. The pronominal n is
        emitted after suffixes that create pronominal case contexts:
        evi-n-e, kapısı-n-dan, evleri-n-e, evdeki-n-den.
        """
        base = CaseSuffix._harmonized_base(word, suffix_obj)
        last_suffix = current_chain[-1] if current_chain else None
        has_pronominal_n = bool(
            last_suffix
            and last_suffix.name in {"posessive_3sg", "posessive_3pl", "marking_ki"}
        )

        if has_pronominal_n:
            if word and base and word[-1] in wrd.VOWELS:
                return ["n" + base]
            return [base]

        if word and base and word[-1] in wrd.VOWELS and base[0] in wrd.VOWELS:
            candidates = []
            if suffix_obj.needs_y_buffer:
                candidates.append("y" + base)
            else:
                candidates.append("n" + base)

            return candidates

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
