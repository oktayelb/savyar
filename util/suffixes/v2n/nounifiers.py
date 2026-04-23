from util.suffix import Suffix, Type,  SuffixGroup

VOWELS = ["a","e","ı","i","o","ö","u","ü"]


class Nounifier(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.VERB,
                makes=Type.NOUN,
                has_major_harmony=True, 
                has_minor_harmony=None,  # Set to None to detect if the user passed a value
                needs_y_buffer=None, 
                group=SuffixGroup.V2N_DERIVATIONAL_NOUNIFIER, 
                is_unique=False,
                form_function=None):
        
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
            form_function=form_function,
            has_major_harmony=has_major_harmony,
            has_minor_harmony=has_minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )


def form_for_nounifier_inti(word, suffix_obj):
    result_list = []
    base = suffix_obj.suffix
    base = Suffix._apply_major_harmony(word, base, suffix_obj.has_major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.has_minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)
    if word and word[-1] in VOWELS:
        reduced = base[1:]
        result_list.append(reduced)
    else:
        result_list.append(base)

    if word:
        if len(word) >= 2:
            last_two = word[-2:]
            if last_two in ['in', 'ın', 'un', 'ün']:
                ti_form = 'ti'
                ti_form = Suffix._apply_major_harmony(word, ti_form, suffix_obj.has_major_harmony)
                ti_form = Suffix._apply_consonant_hardening(word, ti_form)
                if ti_form not in result_list:
                    result_list.append(ti_form)
    
    return result_list

def form_for_toolifier_geç(word, suffix_obj):
    """
    Form for toolifier_geç (Süz-geç -> Süz-gec-i)
    Includes softening (ç -> c).
    """
    result_list = []
    
    # Base: geç
    gec_base = suffix_obj.suffix    
    gec_base = Suffix._apply_major_harmony(word, gec_base, suffix_obj.has_major_harmony)
    gec_base = Suffix._apply_minor_harmony(word, gec_base, suffix_obj.has_minor_harmony)
    gec_base = Suffix._apply_consonant_hardening(word, gec_base)
    result_list.append(gec_base)
    result_list.append(gec_base[1:] ) # eç form 
    # Softening: geç -> gec 
    soft_gec = Suffix._apply_softening(gec_base)  
    result_list.append(soft_gec)
    result_list.append(soft_gec[1:])   # ec form
    

    return result_list


toolative_ek        = Nounifier("toolative_ek", "ek")
##k lı biçimleri de eklenmeli , birleşebilir belki
##TODO form for form for constofactattive koy, agan eğen biçimleri için
constofactative_gen = Nounifier("constofactative_gen", "gen")
constofactative_gin = Nounifier("constofactative_gin", "gin")
perfectative_ik     = Nounifier("perfectative_ik", "ik")
nounifier_i         = Nounifier("nounifier_i" , "i",needs_y_buffer=False)   
#belki birleşebilir
nounifier_gi        = Nounifier("nounifier_gi", "gi")
nounifier_ge        = Nounifier("nounifier_ge", "ge")
nounifier_im        = Nounifier("nounifier_im", "im",needs_y_buffer=False)
nounifier_in        = Nounifier("nounifier_in", "in",needs_y_buffer=False)
nounifier_it        = Nounifier("nounifier_it", "it")
nounifier_inç       = Nounifier("nounifier_inç","inç")
## belki bu ek sadece ti olabilir
nounifier_inti      = Nounifier("nounifier_inti", "inti", form_function= form_for_nounifier_inti)
#birleştirilebilir
toolifier_geç       = Nounifier("toolifier_geç", "geç",  form_function= form_for_toolifier_geç)
subjectifier_giç    = Nounifier("subjectifier_giç", "giç")
#birleştirilebilir
nounifier_anak      = Nounifier("nounifier_anak", "anak")
nounifier_amak      = Nounifier("nounifier_amak", "amak")
subjectifier_men    = Nounifier("subjectifier_men", "men") 


#nounifier_ce   = Nounifier("nounifier_ce", "ce", Type.VERB, Type.NOUN, major_harmony=HasMajorHarmony.Yes, minor_harmony=HasMinorHarmony.No, )


NOUNIFIERS = [
    value for name, value in globals().items()
    if isinstance(value, Suffix) and name != "Suffix"
]