from typing import Dict, List, Optional, Tuple


PERSONAL_PRONOUNS = ["ben", "sen", "o", "biz", "siz"]
REFLEXIVE_PRONOUNS = ["kendi", "birbir"]
DEMONSTRATIVES = ["bu", "şu", "bura", "ora"]
INTERROGATIVES = ["ne", "kim", "nere", "hangi", "kaç", "nasıl", "niye", "niçin"]
QUANTIFIERS = [
    "bir", "her", "hep", "bazı", "tüm", "bütün", "hiç", "hiçbir", "birkaç",
    "birçok", "kimi", "kimse", "herkes", "diğer", "başka",
]
DEGREE_WORDS = ["çok", "daha", "en", "pek", "az", "biraz", "fazla", "oldukça"]
CONJUNCTIONS = [
    "ve", "veya", "ya", "ama", "fakat", "ancak", "çünkü", "ki", "hem", "eğer",
    "ise", "yani", "oysa", "hatta", "üstelik", "diye",
]
PARTICLES = ["de", "mi", "bile", "dahi"]
POSTPOSITIONS = [
    "için", "gibi", "ile", "kadar", "göre", "sonra", "önce", "karşı", "doğru",
    "rağmen", "karşın", "beri", "dek", "değin", "üzere", "dair", "ilişkin",
    "yönelik", "ait", "boyunca", "itibaren", "dolayı", "birlikte", "beraber",
]
PREDICATES = ["değil", "var", "yok"]
ADVERBS = [
    "şimdi", "artık", "hala", "henüz", "hemen", "yine", "zaten", "belki",
    "sadece", "yalnızca", "yalnız",
]
INTERJECTIONS = ["evet", "hayır"]

CLOSED_CLASS_LEXEMES: List[str] = (
    PERSONAL_PRONOUNS
    + REFLEXIVE_PRONOUNS
    + DEMONSTRATIVES
    + INTERROGATIVES
    + QUANTIFIERS
    + DEGREE_WORDS
    + CONJUNCTIONS
    + PARTICLES
    + POSTPOSITIONS
    + PREDICATES
    + ADVERBS
    + INTERJECTIONS
)

LEXEME_SPELLINGS: Dict[str, str] = {
    "da": "de",
    "mı": "mi",
    "mu": "mi",
    "mü": "mi",
    "hâlâ": "hala",
    "gene": "yine",
}

UNINFLECTED_LEXEMES = set(CONJUNCTIONS) | {"de", "bile", "ile", "evet"}

N_STEM_PRONOUNS = {"o", "bu", "şu"}

GENITIVE_IM_STEMS = {"ben", "biz"}

IRREGULAR_STEMS: Dict[str, Tuple[str, str, str]] = {
    "bana": ("ben", "dative_e", "a"),
    "sana": ("sen", "dative_e", "a"),
    "hepsi": ("hep", "possessive_3sg", "si"),
    "birisi": ("bir", "possessive_3sg", "isi"),
    "birileri": ("bir", "possessive_3pl", "ileri"),
    "hiçbirisi": ("hiçbir", "possessive_3sg", "isi"),
}

_LEXEME_SET = set(CLOSED_CLASS_LEXEMES)


def lexeme_of(root: str) -> Optional[str]:
    lexeme = LEXEME_SPELLINGS.get(root, root)
    if lexeme in _LEXEME_SET:
        return lexeme
    return None


def irregular_stem(word: str, root: str, first_suffix_name: Optional[str]) -> Optional[str]:
    for stem, (lexeme, suffix_name, _form) in IRREGULAR_STEMS.items():
        if lexeme == root and suffix_name == first_suffix_name and word.startswith(stem):
            return stem
    return None
