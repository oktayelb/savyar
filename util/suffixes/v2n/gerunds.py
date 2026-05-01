from util.suffix import Suffix, Type,  SuffixGroup
##yeni suffix hiyerarkı gerek
class Gerund(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.VERB,
                makes=Type.NOUN,
                has_major_harmony=True, 
                has_minor_harmony=None,  # Set to None to detect if the user passed a value
                needs_y_buffer=None, 
                group=SuffixGroup.DERIVATIONAL_LOCKING_VERB, 
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
            form_function=None, # Force the use of the overridden _default_form
            has_major_harmony=has_major_harmony,
            has_minor_harmony=has_minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )

#gidereknten kullanımını kabul etmeyeceğiz.
adverbial_erek    = Gerund("adverbial_erek", "erek")
adverbial_ince    = Gerund("adverbial_ince", "ince" )
adverbial_ip      = Gerund("adverbial_ip", "ip")
adverbial_e       = Gerund("adverbial_e", "e")
adverbial_dikçe   = Gerund("adverbial_dikçe", "dikçe")
since_eli         = Gerund("since_eli", "eli", has_minor_harmony=False)
#nondoing_meden    = Gerund("nondoing_meden", "meden", has_minor_harmony=False)

##esiye eklenmeli mi?
GERUNDS = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]   

