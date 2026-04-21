from util.suffix import Suffix, Type,  SuffixGroup

class CompoundVerb(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.VERB,
                makes=Type.VERB,
                has_major_harmony=True, 
                has_minor_harmony=False, 
                needs_y_buffer=True, 
                group=SuffixGroup.VERB_COMPOUND, 
                is_unique=False,
                form_function=None):

        super().__init__(
            name=name,
            suffix=suffix,
            comes_to=comes_to,
            makes=makes,
            form_function=form_function, 
            has_major_harmony=has_major_harmony,
            has_minor_harmony=has_minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )
    @staticmethod
    
    def _default_form(word, suffix_obj):


        ebase = suffix_obj.suffix[0]
        candidates = []
        # Apply standard harmonies using the parent class's static methods
        ebase = Suffix._apply_major_harmony(word, ebase, suffix_obj.has_major_harmony)
        ebase = Suffix._apply_minor_harmony(word, ebase, suffix_obj.has_minor_harmony)
        
        
        if  (suffix_obj.suffix != "iyor") and  word[-1] in ["a","e","ı","i","o","ö","u","ü"]:  
            ebase = "y" + ebase  # Default buffer consonant is 'y'

        
        return [ebase + suffix_obj.suffix[1:]]  # Append the rest of the suffix after the first character
 
# ============================================================================
# VERB TO VERB SUFFIXES (v2v) - Hepsi VERB_DERIVATIONAL (Grup 10)
# ============================================================================

### Buna napacaz...
continuous_iyor       = CompoundVerb("continuous_iyor" , "iyor", has_minor_harmony=True, group=SuffixGroup.PREDICATIVE)

possibiliative_ebil   = CompoundVerb("possibilitative_ebil", "ebil")
almostative_eyazmak   = CompoundVerb("almostative_eyazmak", "eyaz")
continuative_edurmak  = CompoundVerb("continuative_edurmak", "edur")
remainmative_kalmak   = CompoundVerb("remainmative_ekalmak", "ekal")
persistive_egelmek    = CompoundVerb("persistive_egelmek", "egel")
suddenative_ivermek   = CompoundVerb("suddenative_ivermek", "iver")

#iyoru buraya bir şekilde koysak mı?


VERB_COMPOUNDS = [
    value for name, value in globals().items()
    if isinstance(value, Suffix) and name != "Suffix"
]