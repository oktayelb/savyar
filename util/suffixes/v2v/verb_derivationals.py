from util.suffix import Suffix, Type, SuffixGroup
from util.word_methods import VOWELS
from util.words.root_classes import CAUSATIVE_IR_ROOTS, CAUSATIVE_AR_ROOTS
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

def form_for_passive_il(word, suffix_obj, current_chain=None):
    """The Turkish passive has three allomorphs, not one.

    -Il after a consonant (yap-ıl, gör-ül), -n after a vowel (bekle-n, oku-n,
    ye-n), and -In after l, where -Il would double the consonant (al-ın,
    bul-un). This function existed but was commented out, so passive_il fell
    back to the default form and only ever produced -Il. Every vowel-final and
    l-final verb therefore had no passive at all, and the only reading left for
    "beklenen" was reflexive_in, which does produce -n and is a different
    morpheme - 12,084 words across the corpus.
    """
    result_list = []

    def harmonised(base):
        base = Suffix._apply_major_harmony(word, base, suffix_obj.has_major_harmony)
        return Suffix._apply_minor_harmony(word, base, suffix_obj.has_minor_harmony)

    if word and word[-1] in VOWELS:
        result_list.append("n")
        return result_list

    result_list.append(harmonised("il"))
    if word.endswith("l"):
        result_list.append(harmonised("in"))

    return result_list
def form_for_active_it(word, suffix_obj, current_chain=None):
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

    ends_in_polysyllabic_l = word[-1] == "l" and sum(1 for ch in word if ch in VOWELS) >= 2
    if word[-1] in (["r"] + VOWELS) or ends_in_polysyllabic_l:
        result_list.append("t")

    return result_list


def form_for_active_ir(word, suffix_obj, current_chain=None):
    if current_chain or word not in CAUSATIVE_IR_ROOTS:
        return []
    return Suffix._default_form(word, suffix_obj, current_chain=current_chain)


def form_for_active_er(word, suffix_obj, current_chain=None):
    if current_chain or word not in CAUSATIVE_AR_ROOTS:
        return []
    return Suffix._default_form(word, suffix_obj, current_chain=current_chain)

#reflexive_ik    = VerbDerivationalSuffix("reflexive_ik"    , "ik" )
reflexive_is    = VerbDerivationalSuffix("reflexive_is"    , "iş" )
active_it       = VerbDerivationalSuffix("active_it"       , "it", form_function=form_for_active_it )
active_dir      = VerbDerivationalSuffix("active_dir"      , "dir")
##ikisinin ayrı olması iyi değil, belki tekleştirilebilir.
active_ir       = VerbDerivationalSuffix("active_ir"       , "ir", form_function=form_for_active_ir )
active_er       = VerbDerivationalSuffix("active_er"       , "er", form_function=form_for_active_er )

passive_il      = VerbDerivationalSuffix("passive_il"      , "il", form_function=form_for_passive_il )
reflexive_in    = VerbDerivationalSuffix("reflexive_in"    , "in" )
randomative_ele = VerbDerivationalSuffix("randomative_ele" , "ele")

VERB_DERIVATIONALS = [
    value for name, value in globals().items()
    if isinstance(value, Suffix) and name != "Suffix"
]
