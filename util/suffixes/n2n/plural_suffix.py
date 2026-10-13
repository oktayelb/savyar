from util.suffix import Suffix, Type, SuffixGroup
from util.words.closed_class import N_STEM_PRONOUNS

## COMPLETE
class Plural(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.NOUN,
                makes=Type.NOUN,
                 has_major_harmony= True, 
                 has_minor_harmony=None, 
                 needs_y_buffer=False, 
                 form_function=None,
                 group=SuffixGroup.PLURAL, 
                 is_unique=True):
        
        if has_minor_harmony is None:
            # If the suffix contains any narrow vowel, it defaults to having minor harmony
            if any(vowel in suffix for vowel in ['ı', 'i', 'u', 'ü']): # only i is enough bc of the standart narrow front vowel converntion
                has_minor_harmony = True
            else:
                has_minor_harmony = False

        if needs_y_buffer is None:
            if suffix[0] in ['a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü']:
                needs_y_buffer = True
            else:
                needs_y_buffer = False
        super().__init__(
            name=name,
            suffix=suffix,
            comes_to=comes_to,
            makes=makes,
            form_function=form_function ,
            has_major_harmony=has_major_harmony,
            has_minor_harmony=has_minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )

    @staticmethod
    def _default_form(word, suffix_obj, current_chain=None):
        forms = Suffix._default_form(word, suffix_obj, current_chain=current_chain)
        if not current_chain and word in N_STEM_PRONOUNS:
            return ["n" + form for form in forms]
        return forms






plural_ler  = Plural("plural_ler", "ler")

PLURALS = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]