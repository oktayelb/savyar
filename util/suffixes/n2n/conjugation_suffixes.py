from util.suffix import Suffix, Type, SuffixGroup
import util.word_methods as wrd

VOWELS = ["a","e","ı","i","o","ö","u","ü"]

PAST_OR_CONDITIONAL_SUFFIXES = {"pasttense_di", "if_se", "wish_suffix"}
NEGATIVE_SUFFIXES = {"negative_me", "negative_able"}


def _last_suffix_name(current_chain):
    if not current_chain:
        return None
    return current_chain[-1].name


def _after_past_or_conditional(current_chain):
    return _last_suffix_name(current_chain) in PAST_OR_CONDITIONAL_SUFFIXES


def _after_negative(current_chain):
    return _last_suffix_name(current_chain) in NEGATIVE_SUFFIXES


def _after_adverbial_e(current_chain):
    return _last_suffix_name(current_chain) == "adverbial_e"


def form_for_conjugation_1sg(word, suffix_obj, current_chain=None):
    """
    1. Tekil Şahıs (-m, -im)
    - Gel-di-m (Sadece m)
    - Doktor-um (im)
    - Baba-y-ım (y + im)
    - sev-me-m (me + m)
    """
    return_list = []
    
    # 1. Durum: Geçmiş zaman (-di), Şart (-se), Olumsuz emir (-me) sonrası sadece 'm'
    if _after_past_or_conditional(current_chain) or _after_negative(current_chain):
        return_list.append("m")
        return return_list 
    #TODO eyim geleyim gideyim ekle
    # 2. Durum: Standart -im hali (Geleceğ-im, Doktor-um, Arkadaşlar-ım)
    base = "im"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.has_major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.has_minor_harmony)
    
    # Kaynaştırma harfi (Hasta-y-ım)
    if word and word[-1] in VOWELS:
        return_list.append('y' + base)
    else:
        return_list.append(base)


    return return_list


def form_for_conjugation_2sg(word, suffix_obj, current_chain=None):
    """
    2. Tekil Şahıs (-n, -sin)
    - Gel-di-n (Sadece n)
    - Doktor-sun (sin)
    - Arkadaşlarımızdan-sın (sin)
    """
    return_list = []

    # 1. Durum: Geçmiş zaman (-di) veya Şart (-se) sonrası sadece 'n'
    if _after_past_or_conditional(current_chain):
        return_list.append("n")
        return return_list
        
    ## TODO if is verb  add ""
    
    # 2. Durum: Standart -sin hali (Predicative / Geniş Zaman / Şimdiki Zaman)
    sin_base = "sin"
    sin_base = Suffix._apply_major_harmony(word, sin_base, suffix_obj.has_major_harmony)
    sin_base = Suffix._apply_minor_harmony(word, sin_base, suffix_obj.has_minor_harmony)
    return_list.append(sin_base)

    in_base = "in"
    in_base = Suffix._apply_major_harmony(word, in_base, suffix_obj.has_major_harmony)
    in_base = Suffix._apply_minor_harmony(word, in_base, suffix_obj.has_minor_harmony)
    if word and word[-1] in VOWELS:
        return_list.append('y' + in_base)
    else:
        return_list.append(in_base)
    return return_list 


def form_for_conjugation_3sg(word, suffix_obj, current_chain=None):
    """
    3. Tekil Şahıs (Genelde boştur, emir kipinde -sin)
    """
    # 1. Durum: Emir kipi (Gel-sin, Yap-sın, Gel-me-sin)
    # Sadece fiil kökenli veya olumsuzluk ekinden sonra mantıklıdır.
    # "O doktor-sun" denmez, "O doktor" denir. O yüzden her yere -sin eklemiyoruz.
    if _after_negative(current_chain):
         sin_base = "sin"
         sin_base = Suffix._apply_major_harmony(word, sin_base, suffix_obj.has_major_harmony)
         sin_base = Suffix._apply_minor_harmony(word, sin_base, suffix_obj.has_minor_harmony)
         return [sin_base]

    # Eğer kelime saf fiil kökü ise (Gel-sin)
    if wrd.can_be_verb(word):
         sin_base = "sin"
         sin_base = Suffix._apply_major_harmony(word, sin_base, suffix_obj.has_major_harmony)
         sin_base = Suffix._apply_minor_harmony(word, sin_base, suffix_obj.has_minor_harmony)
         # Bunu listeye ekleyelim ama return etmeyelim, belki aşağıdan boş string de döner.
         # Ancak 3. tekil şahıs genelde null suffix'tir.
         return [sin_base, ""]

    # Standart durum: 3. Tekil Şahıs eki YOKTUR. (O doktor, O geldi)
    return [""]


def form_for_conjugation_1pl(word, suffix_obj, current_chain=None):
    """
    1. Çoğul Şahıs (-k, -iz)
    - Gel-di-k (k)
    - Doktor-uz (iz)
    - Hasta-y-ız (y + iz)
    """
    return_list = []
    
    # 1. Durum: Geçmiş zaman (-di) ve Şart (-se) sonrası 'k'
    if _after_past_or_conditional(current_chain):
        k_base = "k" 
        return_list.append(k_base)
        return return_list
        
    # 2. Durum: Standart -iz hali
    iz_base = "iz"
    iz_base = Suffix._apply_major_harmony(word, iz_base, suffix_obj.has_major_harmony)
    iz_base = Suffix._apply_minor_harmony(word, iz_base, suffix_obj.has_minor_harmony)


    # 3. Durum eğer zarf eki -e üstüne gelirse -lim/-lım olması gerekiyor.
    if _after_adverbial_e(current_chain):
        lim_base = "lim"
        lim_base = Suffix._apply_major_harmony(word, lim_base, suffix_obj.has_major_harmony)
        return_list.append(lim_base)


    if word and word[-1] in VOWELS:
        return_list.append('y' + iz_base)
    else:
        return_list.append(iz_base)

    return return_list 
 

def form_for_conjugation_2pl(word, suffix_obj, current_chain=None):
    """
    2. Çoğul Şahıs (-niz, -siniz)
    - Gel-di-niz (niz)
    - Doktor-sunuz (siniz)
    - Arkadaşlarımızdan-sınız (siniz)
    """
    return_list = []
    
    # 1. Durum: Geçmiş zaman (-di) ve Şart (-se) sonrası '-niz'
    if _after_past_or_conditional(current_chain):
        niz_base = "niz"
        niz_base = Suffix._apply_major_harmony(word, niz_base, suffix_obj.has_major_harmony)
        niz_base = Suffix._apply_minor_harmony(word, niz_base, suffix_obj.has_minor_harmony)
        return_list.append(niz_base)
        return return_list
        
    # 2. Durum: Standart -siniz (Predicative)
    siniz_base = "siniz"
    siniz_base = Suffix._apply_major_harmony(word, siniz_base, suffix_obj.has_major_harmony)
    siniz_base = Suffix._apply_minor_harmony(word, siniz_base, suffix_obj.has_minor_harmony)
    return_list.append(siniz_base)
    
    # 3. Durum: Emir kipi (Gel-in, Gel-iniz) - Sadece fiilse
    if wrd.can_be_verb(word) or _after_negative(current_chain):
         in_base = "in"
         in_base = Suffix._apply_major_harmony(word, in_base, suffix_obj.has_major_harmony)
         in_base = Suffix._apply_minor_harmony(word, in_base, suffix_obj.has_minor_harmony)
         
         if word[-1] in VOWELS:
             return_list.append('y' + in_base)
         else:
             return_list.append(in_base)

    return return_list 


def form_for_conjugation_3pl(word, suffix_obj, current_chain=None):
    result_list  = []
    # Standart: -ler (Gel-ir-ler, Ev-ler)
    base = "ler"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.has_major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.has_minor_harmony)
    result_list.append(base)
    
    # Emir kipi 3. çoğul: Gel-sin-ler
    if wrd.can_be_verb(word) or _after_negative(current_chain):
        base_2 = "sinler"
        base_2 = Suffix._apply_major_harmony(word, base_2, suffix_obj.has_major_harmony)
        base_2 = Suffix._apply_minor_harmony(word, base_2, suffix_obj.has_minor_harmony)
        result_list.append(base_2)

    return result_list


conjugation_1sg = Suffix("conjugation_1sg", "im", Type.BOTH, Type.NOUN, form_function= form_for_conjugation_1sg, has_major_harmony=True, has_minor_harmony=True, is_unique=True, group=SuffixGroup.CONJUGATION)
conjugation_2sg = Suffix("conjugation_2sg","sin", Type.BOTH, Type.NOUN, form_function= form_for_conjugation_2sg, has_major_harmony=True, has_minor_harmony=True, is_unique=True, group=SuffixGroup.CONJUGATION)
conjugation_3sg = Suffix("conjugation_3sg", "", Type.BOTH, Type.NOUN, form_function=form_for_conjugation_3sg , has_major_harmony=True, has_minor_harmony=True, is_unique=True, group=SuffixGroup.CONJUGATION)

conjugation_1pl = Suffix("conjugation_1pl", "iz", Type.BOTH, Type.NOUN, form_function= form_for_conjugation_1pl, has_major_harmony=True, has_minor_harmony=True, is_unique=True, group=SuffixGroup.CONJUGATION)
conjugation_2pl = Suffix("conjugation_2pl", "siniz", Type.BOTH, Type.NOUN, form_function=form_for_conjugation_2pl , has_major_harmony=True, has_minor_harmony=True, is_unique=True, group=SuffixGroup.CONJUGATION)
conjugation_3pl = Suffix("conjugation_3pl", "ler", Type.BOTH, Type.NOUN, form_function=form_for_conjugation_3pl , has_major_harmony=True, has_minor_harmony=True, is_unique=True, group=SuffixGroup.CONJUGATION)

CONJUGATIONS = [
    value for name, value in globals().items() 
    if isinstance(value, Suffix) and name != "Suffix"
]
