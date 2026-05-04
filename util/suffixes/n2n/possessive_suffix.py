from util.suffix import Suffix, Type, SuffixGroup


## COMPLETE
class PossessiveSuffix(Suffix):
    def __init__(self, name, suffix, 
                 comes_to=Type.NOUN,
                 makes=Type.NOUN,
                 has_major_harmony=True, 
                 has_minor_harmony=True,  # Set to None to detect if the user passed a value
                 needs_y_buffer=False, 
                 form_function=None,
                 group=SuffixGroup.POSSESSIVE, 
                 is_unique=False):
        

        super().__init__(
            name=name,
            suffix=suffix,
            comes_to=Type.NOUN,
            makes=Type.NOUN,
            form_function=form_function, # Force the use of the overridden _default_form
            has_major_harmony=has_major_harmony,
            has_minor_harmony=has_minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )

def form_for_possessive_3sg(word, suffix_obj, current_chain=None):
    base = "i"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.has_major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.has_minor_harmony)
    
    if word[-1] in ["a","e","ı","i","o","ö","u","ü"]:
        base = "s" + base

    return [base]


possessive_1sg = PossessiveSuffix("possessive_1sg", "im")
possessive_2sg = PossessiveSuffix("possessive_2sg", "in")
possessive_3sg = PossessiveSuffix("possessive_3sg", "i", form_function=form_for_possessive_3sg)
possessive_1pl = PossessiveSuffix("possessive_1pl", "imiz")
possessive_2pl = PossessiveSuffix("possessive_2pl", "iniz")
possessive_3pl = PossessiveSuffix("possessive_3pl", "leri",has_minor_harmony= False)

POSSESSIVE_SUFFIX = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]
