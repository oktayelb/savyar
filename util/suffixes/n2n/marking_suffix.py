from util.suffix import Suffix, Type, SuffixGroup

##ek çalışmıyor
class MarkingKi(Suffix):
    def __init__(self, name, suffix, 
                comes_to=Type.NOUN,
                makes=Type.NOUN,
                has_major_harmony=False, 
                has_minor_harmony=None, 
                needs_y_buffer=False, 
                group=SuffixGroup.MARKING_KI, 
                is_unique=True):
        
        super().__init__(
            name=name,
            suffix=suffix,
            comes_to=comes_to,
            makes=makes,
            form_function=None ,
            has_major_harmony=has_major_harmony,
            has_minor_harmony=has_minor_harmony,
            needs_y_buffer=needs_y_buffer,
            group=group,
            is_unique=is_unique
        )
    @staticmethod
    
    def _default_form(word, suffix_obj, current_chain=None):
        last_suffix = current_chain[-1] if current_chain else None
        if last_suffix and last_suffix.name in {"locative_de", "noun_compound"}:
            return ["ki"]
        return []

###form fonksiyonu mu dizsek?


#### evdeki    TRUE
#### evdenki   FALSE ilginç
#### eviki     FALSE
#### evinki    TRUE
#### eveki     FALSE
## bu yılkiler TRUE
####

marking_ki = MarkingKi("marking_ki", "ki")

MARKINGS = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]
