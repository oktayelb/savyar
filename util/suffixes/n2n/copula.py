from util.suffix import Suffix, Type, SuffixGroup

VOWELS = ["a","e","ı","i","o","ö","u","ü"]

## COMPLETE
class Copula(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.NOUN,
                makes=Type.NOUN,
                has_major_harmony=True, 
                has_minor_harmony=None,  # Set to None to detect if the user passed a value
                needs_y_buffer=True, 
                group=SuffixGroup.PREDICATIVE, 
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
            comes_to=comes_to,
            makes=makes,
            form_function=None, # Force the use of the overridden _default_form
            has_major_harmony=has_major_harmony,
            has_minor_harmony=has_minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )

    @staticmethod
    
    def _default_form(word, suffix_obj, current_chain=None):
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
        
        candidates.append(base)  # Always include the base form

        if word and word[-1] in VOWELS:  # If the last character is a vowel, we need to consider buffer consonants
            if suffix_obj.needs_y_buffer:
                candidates.remove(base)
                candidates.append('y' + base)

        return candidates





nounaorist_dir   = Copula("nounaorist_dir"   , "dir",needs_y_buffer=False)
pasttense_noundi = Copula("pasttense_noundi"     , "di")
if_se            = Copula("if_se"            , "se")
copula_mis       = Copula("copula_mis"       , "miş")


COPULA = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]
