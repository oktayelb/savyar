from util.suffix import Suffix, Type, SuffixGroup

VOWELS = ["a", "e", "ı", "i", "o", "ö", "u", "ü"]


def form_for_confactuous_le(word, suffix_obj):
    base = "le"
    base = Suffix._apply_major_harmony(word, base, suffix_obj.has_major_harmony)
    base = Suffix._apply_minor_harmony(word, base, suffix_obj.has_minor_harmony)
    base = Suffix._apply_consonant_hardening(word, base)

    if word[-1] in VOWELS:
        base = "y" + base

    return [base]


confactous_le = Suffix(
    "confactuous_le", "le", Type.NOUN, Type.NOUN,
    form_function=form_for_confactuous_le,
    has_major_harmony=True, has_minor_harmony=True,
    group=SuffixGroup.WITH_LE,
    is_unique=True,
)


WITH_LE = [
    value for name, value in globals().items()
    if isinstance(value, Suffix) and name != "Suffix"
]
