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
    
    def _default_form(word, suffix_obj):
        """
        Overridden default form handler specifically for Case Suffixes.
        Consolidates 'n' and 'y' buffer logic for Turkish nominal inflection.
        
        """
        
        base = suffix_obj.suffix
        candidates = []
        # Apply standard harmonies using the parent class's static methods
        base = Suffix._apply_major_harmony(word, base, suffix_obj.has_major_harmony)
        base = Suffix._apply_minor_harmony(word, base, suffix_obj.has_minor_harmony)
        base = Suffix._apply_consonant_hardening(word, base)

        candidates.append(base)  # Always include the base form (e.g. kapı+dan, kalma+da)

        # Buffer consonants only apply when the stem ends in a vowel.
        # 'n' covers post-possessive chains (evi-n-den); 'y' covers vowel+vowel (kapı-y-ı).
        if word and word[-1] in ["a","e","ı","i","o","ö","u","ü"]:
            candidates.append('n' + base)

            if suffix_obj.needs_y_buffer:
                candidates.append('y' + base)

        return candidates


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