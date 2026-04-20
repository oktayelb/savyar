from enum import Enum, IntEnum
import util.word_methods as wrd

# Eklerin hiyerarşisi.
# Kural: Bir ek, kendinden daha BÜYÜK numaralı bir gruptan sonra GELEMEZ.
class SuffixGroup(IntEnum):                                                                                         
     # fiilden fiil yapan ekler; -iş -il -in -tir...
    V2V_DERIVATIONAL = 25

    # fiili olumsuz yapan ekler; -me -eme
    VERB_NEGATING = 35
    
    # birleşik fiil ekleri, -ebil -eyaz -edur...
    VERB_COMPOUND = 40         
    
    # İsimden Fiile yapım ekleri; -le  -e -se...
    N2V_DERIVATIONAL = 50      
    
    # İsimden isim yapım ekleri -lık -lı -cı...
    N2N_DERIVATIONAL = 50   

    # Fiilden isim yapan ekler; -iş -me -ma -ış...
    V2N_DERIVATIONAL = 50      
    
    # Zarf fiiller
    DERIVATIONAL_LOCKING_VERB = 55 
    
    # Çoğul eki  -ler
    PLURAL = 60                
    
    # İyelik Ekleri; -im -in -imiz
    POSSESSIVE = 150           
    
    # Hal Ekleri -e -de -i -den -nin
    CASE = 200                       
    
    # İşaret eki -ki
    MARKING_KI = 225                 
    
    # Birliktelik eki -le
    WITH_LE = 230                    
    
    # isimden Zarf yapan ekler; leyin, (in), cesine, ken
    DERIVATIONAL_LOCKING_NOUN = 240  
    
    # Ek-fiil -dir -idi -imiş -ise
    PREDICATIVE = 250                
    
    # Fiil şahıs çekimleri -im -sin -ler
    CONJUGATION = 300                

class Type(Enum):
    NOUN = "noun"
    VERB = "verb"
    BOTH = "both"


class Suffix:
    def __init__(self, name, suffix, comes_to, makes, 
                 form_function=None, has_major_harmony=None, has_minor_harmony=None, needs_y_buffer=False,
                 group=None, is_unique=False):
        
        self.name = name
        self.suffix = str(suffix)
        self.comes_to = comes_to
        self.makes = makes
        self.has_major_harmony = has_major_harmony
        self.has_minor_harmony = has_minor_harmony
        self.needs_y_buffer = needs_y_buffer
        self.form_function = form_function if form_function else self._default_form
        
        # Hiyerarşi ve Tekrarlama Kontrolü
        self.group = group
        self.is_unique = is_unique
    
    def form(self, word):
        return self.form_function(word, self)
    
    @staticmethod
    def _default_form(word, suffix_obj):
        # 1. Baz formu al
        base = suffix_obj.suffix    
        
        # 2. Uyumları uygula
        base = Suffix._apply_major_harmony(word, base, suffix_obj.has_major_harmony)
        base = Suffix._apply_minor_harmony(word, base, suffix_obj.has_minor_harmony)
        base = Suffix._apply_consonant_hardening(word, base)
        
        candidates = [] # Start empty!
    
        # 4. Çarpışma Kontrolü (Collision Check)
        vowel_collision = Suffix._vowel_collision(word, base)

        if vowel_collision:

            if suffix_obj.needs_y_buffer:
                candidates.append('y' + base)        
            elif len(base) > 1:
                candidates.append(base[1:]) 


        else:        
            candidates.append(base)
        final_results = []
        for cand in candidates:
            final_results.append(cand) 
            
            softened = Suffix._apply_softening(cand)
            if softened != cand:
                final_results.append(softened) 
        
        return final_results
    
    @staticmethod
    def _apply_major_harmony(word, result, has_major_harmony):
        if has_major_harmony != True:
            return result
        
        if wrd.major_harmony(word) == wrd.MajorHarmony.BACK:
            result = result.replace("e", "a")
            result = result.replace("i", "ı")
            result = result.replace("ü", "u")
            result = result.replace("ö", "o")
        
        return result
    
    @staticmethod
    def _apply_minor_harmony(word, result, has_minor_harmony):
        if has_minor_harmony != True:
            return result
        
        word_harmony = wrd.minor_harmony(word)
        
        if word_harmony == wrd.MinorHarmony.BACK_ROUND:
            result = result.replace("ı", "u")
        elif word_harmony == wrd.MinorHarmony.FRONT_ROUND:
            result = result.replace("i", "ü")
        
        return result
    
    @staticmethod
    def _apply_consonant_hardening(word, result):
        """
        Ünsüz Sertleşmesi (Benzeşmesi):
        Fıstıkçı Şahap ile biten kelimeye 'c, d, g' ile başlayan ek gelirse 'ç, t, k' olur.
        """
        if not word or not result:
            return result
        
        if word[-1] not in wrd.HARD_CONSONANTS:
            return result
        
        first_char = result[0]
        # Yumuşak ünsüz -> Sert ünsüz haritası
        hardening_map = {'g': 'k', 'd': 't', 'c': 'ç', 'ğ': 'k'}
        
        if first_char in hardening_map:
             return hardening_map[first_char] + result[1:]
             
        return result

    @staticmethod
    def _apply_softening(form):
        """
        Ünsüz Yumuşaması (Suffix Softening):
        Ekin kendisi ünlü ile başlayan başka bir ek aldığında sonundaki harf değişebilir.
        Bu metot, ekin son harfini kontrol eder ve yumuşamış halini döndürür.
        
        Örnek: 'ecek' -> 'eceğ', 'dik' -> 'diğ', 'amaç' -> 'amac'
        """
        if not form:
            return form
        
        last_char = form[-1]
        
        # Sık görülen: k -> ğ (Gelecek-im -> Geleceğim)
        if last_char == 'k':
            return form[:-1] + 'ğ'
        
        # Diğer yumuşamalar (Suffixlerde daha nadir ama mümkün)
        elif last_char == 'ç':
            return form[:-1] + 'c'

        
        # Eğer yumuşama yoksa orijinali döndür
        return form

    @staticmethod
    def _vowel_collision(word, suffix):
        return ( 
                word[-1] in wrd.VOWELS and 
                suffix[0] in wrd.VOWELS)