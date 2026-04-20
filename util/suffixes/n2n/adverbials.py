from util.suffix import Suffix, Type, SuffixGroup

VOWELS = ["a", "e", "ı", "i", "o", "ö", "u", "ü"]


def form_for_when_ken(word, suffix_obj):
    base = "ken"
    if word and word[-1] in VOWELS:
        base = "y" + base
    return [base]


temporative_leyin = Suffix(
    "temporative_leyin", "leyin", Type.NOUN, Type.NOUN,
    has_major_harmony=False, has_minor_harmony=False,
    group=SuffixGroup.DERIVATIONAL_LOCKING_NOUN,
)
"""
adverbial_in = Suffix(
    "adverbial_in", "in", Type.NOUN, Type.NOUN,
    has_major_harmony=True, has_minor_harmony=True,
    group=SuffixGroup.DERIVATIONAL_LOCKING_NOUN,
)
"""
adverbial_cesine = Suffix(
    "adverbial_cesine", "cesine", Type.NOUN, Type.NOUN,
    has_major_harmony=True, has_minor_harmony=False,
    group=SuffixGroup.DERIVATIONAL_LOCKING_NOUN,
)

# le'den sonra ken gelebiliyor
when_ken = Suffix(
    "when_ken", "ken", Type.NOUN, Type.NOUN,
    form_function=form_for_when_ken,
    has_major_harmony=False, has_minor_harmony=False,
    group=SuffixGroup.DERIVATIONAL_LOCKING_NOUN,
)


ADVERBIALS = [
    value for name, value in globals().items()
    if isinstance(value, Suffix) and name != "Suffix"
]
