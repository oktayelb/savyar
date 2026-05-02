from util.suffix import Suffix, Type, SuffixGroup


VOWELS = ['a', 'e', 'ı', 'i', 'o', 'ö', 'u', 'ü']


def form_for_approximative_si(word, suffix_obj, current_chain=None):
    base1 = "imsi"

    base1 = Suffix._apply_major_harmony(word, base1, suffix_obj.has_major_harmony)
    base1 = Suffix._apply_minor_harmony(word, base1, suffix_obj.has_minor_harmony)

    base2 = "si"
    base2 = Suffix._apply_major_harmony(word, base2, suffix_obj.has_major_harmony)
    base2 = Suffix._apply_minor_harmony(word, base2, suffix_obj.has_minor_harmony)

    if word[-1] in VOWELS:
        return [base1[1:]]

    return [base1, base2]

def form_for_abstractifier_iyat(word, suffix_obj, current_chain=None):
    result_list = ["iye","iyet","iyat","at","et"]
    return result_list

def form_for_counting_er(word, suffix_obj, current_chain=None):

    base = "er"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.has_major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.has_minor_harmony)


    if (word and word[-1] in VOWELS):
        base = "ş" + base
    
    return [base]

def form_for_diminutive_cik(word, suffix_obj, current_chain=None):  
    base = "cik"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.has_major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.has_minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base) 

    base2 = "cek"
    base2 = Suffix._apply_major_harmony(word, base2, suffix_obj.has_major_harmony)
    base2 = Suffix._apply_minor_harmony(word, base2, suffix_obj.has_minor_harmony)
    base2 = Suffix._apply_consonant_hardening(word, base2)
    
## cağız ayırabilirim
    base3 = "cağız"
    base3 = Suffix._apply_major_harmony(word, base3, suffix_obj.has_major_harmony)
    base3 = Suffix._apply_consonant_hardening(word, base3)


    return [base,base2,base3]


actor_ci            = Suffix("actor_ci", "ci", Type.NOUN, Type.NOUN, has_major_harmony=True, has_minor_harmony=True, group=SuffixGroup.N2N_DERIVATIONAL)
privative_siz       = Suffix("privative_siz", "siz", Type.NOUN, Type.NOUN, has_major_harmony=True, has_minor_harmony=True, group=SuffixGroup.N2N_DERIVATIONAL)
compositive_li     = Suffix("compositive_li", "li", Type.NOUN, Type.NOUN, has_major_harmony=True, has_minor_harmony=True, group=SuffixGroup.N2N_DERIVATIONAL)
suitative_lik       = Suffix("suitative_lik", "lik", Type.NOUN, Type.NOUN, has_major_harmony=True, has_minor_harmony=True, group=SuffixGroup.N2N_DERIVATIONAL)


counting_er          = Suffix("counting_er", "er", Type.NOUN, Type.NOUN, form_function= form_for_counting_er,  has_major_harmony=True, has_minor_harmony=False, group=SuffixGroup.N2N_DERIVATIONAL)
cooperative_deş      = Suffix("cooperative_deş", "deş", Type.NOUN, Type.NOUN, has_major_harmony=True, has_minor_harmony=False, group=SuffixGroup.N2N_DERIVATIONAL)
relative_ce          = Suffix("relative_ce", "ce", Type.NOUN, Type.NOUN, has_major_harmony=True, has_minor_harmony=False, group=SuffixGroup.N2N_DERIVATIONAL)
relative_sel         = Suffix("relative_sel", "sel", Type.NOUN, Type.NOUN, has_major_harmony=True, has_minor_harmony=False, group=SuffixGroup.N2N_DERIVATIONAL)
diminutive_cik       = Suffix("diminutive_cik", "cik", Type.NOUN, Type.NOUN,  form_function= form_for_diminutive_cik,has_major_harmony=True, has_minor_harmony=True, group=SuffixGroup.N2N_DERIVATIONAL)

philicative_cil      = Suffix("philicative_cil", "cil", Type.NOUN, Type.NOUN, has_major_harmony=True, has_minor_harmony=True, group=SuffixGroup.N2N_DERIVATIONAL)
abstractifier_iyat   = Suffix("abstractifier_iyat", "iyat", Type.NOUN, Type.NOUN,form_function= form_for_abstractifier_iyat, has_major_harmony=False, has_minor_harmony=False, group=SuffixGroup.N2N_DERIVATIONAL)
ideologicative_izm   = Suffix("ideologicative_izm", "izm", Type.NOUN, Type.NOUN, has_major_harmony=False, has_minor_harmony=False, group=SuffixGroup.N2N_DERIVATIONAL)
scientist_olog       = Suffix("scientist_olog", "olog", Type.NOUN, Type.NOUN, has_major_harmony=False, has_minor_harmony=False, group=SuffixGroup.N2N_DERIVATIONAL)
familative_gil       = Suffix("familative_gil", "gil", Type.NOUN, Type.NOUN, has_major_harmony=False, has_minor_harmony=False, group=SuffixGroup.N2N_DERIVATIONAL)
approximative_si     = Suffix("approximative_si", "si", Type.NOUN, Type.NOUN, form_function= form_for_approximative_si, has_major_harmony=True, has_minor_harmony=True, group=SuffixGroup.N2N_DERIVATIONAL)
approximative_imtrak = Suffix("approximative_imtrak", "imtrak", Type.NOUN, Type.NOUN, has_major_harmony=False, has_minor_harmony=False, group=SuffixGroup.N2N_DERIVATIONAL)


#adverbial in+ ci olarak analiz edilip silinebilir
ordinal_inci         = Suffix("ordinal_inci", "inci", Type.NOUN, Type.NOUN, has_major_harmony=True, has_minor_harmony=True, group=SuffixGroup.N2N_DERIVATIONAL)


DERIVATIONALS = [
value for name, value in globals().items() 
if isinstance(value, Suffix) and name != "Suffix"
]
