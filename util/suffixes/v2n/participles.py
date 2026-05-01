from util.suffix import Suffix, Type,  SuffixGroup
import util.word_methods as wrd



class Participle(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.VERB,
                makes=Type.NOUN,
                has_major_harmony=True, 
                has_minor_harmony=None,  # Set to None to detect if the user passed a value
                needs_y_buffer=None,
                form_function=None, 
                group=SuffixGroup.V2N_DERIVATIONAL, 
                is_unique=False):
        
        # Dynamic default assignment for minor harmony
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
            form_function=form_function, # Force the use of the overridden _default_form
            has_major_harmony=has_major_harmony,
            has_minor_harmony=has_minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )


def form_for_factative_ir(word, suffix_obj, current_chain=None):
    """
    Form function for factative_ir suffix (Geniş Zaman Sıfat-Fiil)
    - Default forms: er, ir, z (negative)
    - Note: Does not soften (r and z are soft/continuant).
    """
    result_list = []
    
    # Geniş zamanın olumsuzu (maz/mez) kökü için 'z'
    if len(word) > 2 and word[-2:] in ["ma","me"]:
        z_base = "z"
        result_list.append(z_base)

    # ir form with harmony
    ir_base = 'ir'
    ir_base = Suffix._apply_major_harmony(word, ir_base, suffix_obj.has_major_harmony)
    result_list.append(ir_base)
    
    # er form with harmony (Gider, Yapar)
    er_base = 'er'
    er_base = Suffix._apply_major_harmony(word, er_base, suffix_obj.has_major_harmony)
    result_list.append(er_base)
    
    # Vowel drop variant for vowel-ending words (Oku-r)
    if word and word[-1] in ["a","e","ı","i","o","ö","u","ü"]:
        r_form = 'r'
        if r_form not in result_list:
            result_list.append(r_form)
    
    return result_list

# gen gan ile birleşmesi söz konusu
factative_en      = Participle("factative_en", "en")
pastfactative_miş = Participle("pastfactative_miş", "miş")
adjectifier_dik   = Participle("adjectifier_dik", "dik" )
nounifier_ecek    = Participle("nounifier_ecek", "ecek")
factative_ir      = Participle("factative_ir", "ir", needs_y_buffer= False, form_function= form_for_factative_ir)
willing_esi       = Participle("willing_esi", "esi", has_minor_harmony=False) # Ölesiye 
## suffix mez silinid , me+ z olarak analiz edilecek

PARICIPLES = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]
