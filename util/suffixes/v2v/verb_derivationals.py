from util.suffix import Suffix, Type, SuffixGroup
from util.word_methods import VOWELS
# ============================================================================
# FORM FUNCTIONS
# ============================================================================

class VerbDerivationalSuffix(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.VERB,
                makes=Type.VERB,
                has_major_harmony=True, 
                has_minor_harmony=None,  # Set to None to detect if the user passed a value
                needs_y_buffer=False,
                form_function=None, 
                group=SuffixGroup.V2V_DERIVATIONAL, 
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
            form_function=form_function, # Force the use of the overridden _default_form
            has_major_harmony=has_major_harmony,
            has_minor_harmony=has_minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )

"""
def form_for_passive_il(word, suffix_obj):

    result_list = []
    
    # il form with harmony
    il_base = 'il'
    il_base = Suffix._apply_major_harmony(word, il_base, suffix_obj.has_major_harmony)
    il_base = Suffix._apply_minor_harmony(word, il_base, suffix_obj.has_minor_harmony)
    result_list.append(il_base)

    if word.endswith("l") :
        in_base = 'in'
        in_base = Suffix._apply_major_harmony(word, in_base, suffix_obj.has_major_harmony)
        in_base = Suffix._apply_minor_harmony(word, in_base, suffix_obj.has_minor_harmony)
        result_list.append(in_base)


    return result_list
"""
def form_for_active_it(word, suffix_obj):
    """
    Form function for active_it suffix (Active Sıfat-Fiil)
    - Default forms: it, dir
    - Note: Does not soften (t and d are soft/continuant).
    """
    result_list = []
    
    # it form with harmony
    it_base = "it"
    it_base = Suffix._apply_major_harmony(word, it_base, suffix_obj.has_major_harmony)
    it_base = Suffix._apply_minor_harmony(word, it_base, suffix_obj.has_minor_harmony)
    result_list.append(it_base)

    if word[-1] in (["r"] + VOWELS):
        result_list.append("t")

    return result_list

reflexive_ik    = VerbDerivationalSuffix("reflexive_ik"    , "ik" )
reflexive_is    = VerbDerivationalSuffix("reflexive_is"    , "iş" )
active_it       = VerbDerivationalSuffix("active_it"       , "it", form_function=form_for_active_it )
active_dir      = VerbDerivationalSuffix("active_dir"      , "dir")
##ikisinin ayrı olması iyi değil, belki tekleştirilebilir.
active_ir       = VerbDerivationalSuffix("active_ir"       , "ir" )
active_er       = VerbDerivationalSuffix("active_er"       , "er" )

passive_il      = VerbDerivationalSuffix("passive_il"      , "il" )
reflexive_in    = VerbDerivationalSuffix("reflexive_in"    , "in" )
randomative_ele = VerbDerivationalSuffix("randomative_ele" , "ele")

VERB_DERIVATIONALS = [
    value for name, value in globals().items()
    if isinstance(value, Suffix) and name != "Suffix"
]