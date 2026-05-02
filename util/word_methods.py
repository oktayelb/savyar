from enum import Enum
from pathlib import Path
import random
from typing import List, Tuple, Optional

_TR_LOWER_TABLE = str.maketrans("İI", "iı")

def tr_lower(s: str) -> str:
    """Lowercase a Turkish string correctly: İ→i, I→ı."""
    return s.translate(_TR_LOWER_TABLE).lower()

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "words.txt"
VERB_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "verbs.txt"


## Vowel Classes
BACK_FLAT   = ['a','ı']
BACK_ROUND  = ['o','u']

FRONT_FLAT  = ['e','i']
FRONT_ROUND = ['ö','ü']

BACK_VOWELS  = BACK_FLAT  + BACK_ROUND
FRONT_VOWELS = FRONT_FLAT + FRONT_ROUND 

VOWELS = BACK_VOWELS + FRONT_VOWELS

HARD_CONSONANTS = ['f','s','t','k','ç','ş','h','p']  # fıstıkçı şahap

# --- Enums ---
class MajorHarmony(Enum):
    BACK = "back"
    FRONT = "front"

class MinorHarmony(Enum):
    BACK_ROUND = 0
    BACK_WIDE  = 1
    FRONT_ROUND = 2
    FRONT_WIDE  = 3

# --- Centralized Dictionary State ---
WORDS_SET: set = set()
VERB_SET: set = set()

def _load_dictionary():
    global WORDS_SET, VERB_SET
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            WORDS_SET = {line.strip() for line in f if line.strip()}
        with open(VERB_DATA_FILE, "r", encoding="utf-8") as f:
            VERB_SET = {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        print(f"Warning: {DATA_FILE} or {VERB_DATA_FILE} not found")
        WORDS_SET = set()
        VERB_SET = set()

# Initialize on module load
_load_dictionary()

def delete_word(word: str) -> bool:
    """Removes a word from the in-memory dictionary state."""
    word = tr_lower(word.strip())
    removed = False

    if word in WORDS_SET:
        WORDS_SET.discard(word)
        removed = True

    if word in VERB_SET:
        VERB_SET.discard(word)
        removed = True

    if removed:
        _DERIVED_CACHE.clear()
        return True

    return False

def get_all_words() -> List[str]:
    """Returns the current list of dictionary words."""
    return sorted(WORDS_SET)

def get_all_verbs() -> List[str]:
    """Returns the current list of dictionary verb roots."""
    return sorted(VERB_SET)

def infinitive_form(root: str) -> Optional[str]:
    """Build the -mAk infinitive for a verb root using the actual suffix object."""
    root = tr_lower(root.strip())
    if not root:
        return None

    from util.suffixes.v2n.infinitives import infinitive_mek

    forms = infinitive_mek.form(root)
    if not forms:
        return None
    return root + forms[0]

def get_random_word() -> Optional[str]:
    """Returns a random word from the dictionary."""
    return random.choice(list(WORDS_SET)) if WORDS_SET else None

def exists(word: str) -> bool:
    return can_be_noun(word) or can_be_verb(word)

def get_closed_class_categories(word: str) -> List[str]:
    """Returns list of closed-class categories for a word, or empty list if open-class."""
    from util.words.closed_class import CLOSED_CLASS_LOOKUP
    entries = CLOSED_CLASS_LOOKUP.get(word, [])
    return list({e.category for e in entries})

def is_non_ben_pronoun_surface(word: str) -> bool:
    from util.words.closed_class import NON_BEN_PRONOUN_SURFACES
    return word in NON_BEN_PRONOUN_SURFACES

def can_be_noun(word: str) -> bool:
    if not word:
        return False

    if word in WORDS_SET:
        return True

    if word.endswith("l"):
        soft_l = word[:-1] + "ł"
        if soft_l in WORDS_SET:
            return True

    return False

def can_be_verb(word: str) -> bool:
    """Checks if a root is a verb by looking it up in VERB_SET."""
    return word in VERB_SET

# --- Harmony functions ---
def major_harmony(word: str) -> MajorHarmony | None:
    """Determines major vowel harmony based on last vowel"""
    if word.endswith("l"):              
        soft_l = word[:-1] + "ł"
        if can_be_noun(soft_l):
            return MajorHarmony.FRONT
    for ch in reversed(word):
        if ch in VOWELS:
            return MajorHarmony.BACK if ch in BACK_VOWELS else MajorHarmony.FRONT
    return None  # no vowels

def minor_harmony(word: str) -> MinorHarmony | None:
    """Determines minor vowel harmony based on last vowel"""
    for ch in reversed(word):
        if ch not in VOWELS:
            continue
        if ch in ['o', 'u',"ö","ü"]:  
            return MinorHarmony.BACK_ROUND if major_harmony(word) == MajorHarmony.BACK else MinorHarmony.FRONT_ROUND
        if ch in ['a', 'ı','e', 'i']:
             return MinorHarmony.BACK_WIDE if major_harmony(word) == MajorHarmony.BACK else MinorHarmony.FRONT_WIDE
    return None

# --- Morphological utilities ---

def ends_with_consonant(word: str) -> bool:
    """Check if word ends with a consonant"""
    return word and word[-1] not in VOWELS

# --- Derived-word detection ---
# Common Turkish derivational suffix patterns that create new dictionary entries.
# If a word ends with one of these AND stripping it yields a valid root,
# the word is likely derived (not a true root).
_DERIVATIONAL_ENDINGS_NOUN = [
    # Noun-from-verb
    ('ıcı', 'verb'), ('ici', 'verb'), ('ucu', 'verb'), ('ücü', 'verb'),
    ('gıcı', 'verb'), ('gici', 'verb'), ('gucu', 'verb'), ('gücü', 'verb'),
    ('ma', 'verb'), ('me', 'verb'),
    ('ış', 'verb'), ('iş', 'verb'), ('uş', 'verb'), ('üş', 'verb'),
    # Noun-from-noun
    ('lık', 'noun'), ('lik', 'noun'), ('luk', 'noun'), ('lük', 'noun'),
    ('cı', 'noun'), ('ci', 'noun'), ('cu', 'noun'), ('cü', 'noun'),
    ('çı', 'noun'), ('çi', 'noun'), ('çu', 'noun'), ('çü', 'noun'),
    ('sız', 'noun'), ('siz', 'noun'), ('suz', 'noun'), ('süz', 'noun'),
    ('lı', 'noun'), ('li', 'noun'), ('lu', 'noun'), ('lü', 'noun'),
]

_DERIVATIONAL_ENDINGS_VERB = [
    # Verb-from-verb (reciprocal, reflexive, causative, passive)
    ('ışmak', 'verb'), ('işmek', 'verb'), ('uşmak', 'verb'), ('üşmek', 'verb'),
    ('ınmak', 'verb'), ('inmek', 'verb'), ('unmak', 'verb'), ('ünmek', 'verb'),
    ('dırmak', 'verb'), ('dirmek', 'verb'), ('tırmak', 'verb'), ('tirmek', 'verb'),
    ('durmak', 'verb'), ('dürmek', 'verb'), ('turmak', 'verb'), ('türmek', 'verb'),
    ('ılmak', 'verb'), ('ilmek', 'verb'), ('ulmak', 'verb'), ('ülmek', 'verb'),
    # Verb-from-noun
    ('lamak', 'noun'), ('lemek', 'noun'),
]

# Cache of words confirmed as derived
_DERIVED_CACHE: dict = {}

def is_derived_word(word: str) -> bool:
    """
    Returns True if `word` is likely a derived form (not a true root).
    Checks if stripping a common derivational suffix yields a valid shorter root.
    """
    if word in _DERIVED_CACHE:
        return _DERIVED_CACHE[word]

    result = False

    # Check noun roots that might be derived
    if word in WORDS_SET:
        for ending, source_pos in _DERIVATIONAL_ENDINGS_NOUN:
            if word.endswith(ending) and len(word) > len(ending) + 1:
                stem = word[:-len(ending)]
                if source_pos == 'verb':
                    if can_be_verb(stem):
                        result = True
                        break
                else:
                    if can_be_noun(stem):
                        result = True
                        break

    # Check verb infinitives that might be derived
    if not result and word in WORDS_SET and (word.endswith('mak') or word.endswith('mek')):
        verb_root = word[:-3]
        for ending, source_pos in _DERIVATIONAL_ENDINGS_VERB:
            suffix_part = ending[:-3]  # strip mak/mek from the ending
            if verb_root.endswith(suffix_part) and len(verb_root) > len(suffix_part) + 1:
                stem = verb_root[:-len(suffix_part)]
                if source_pos == 'verb':
                    if can_be_verb(stem):
                        result = True
                        break
                else:
                    if can_be_noun(stem):
                        result = True
                        break

    _DERIVED_CACHE[word] = result
    return result


def get_root_candidates(surface_root: str) -> List[str]:
    """Analyzes the text segment and returns Surface Forms that are Dictionary Lemmas."""
    candidates = [] 

    def check_and_add_softened(form_to_check):
        if not form_to_check: return

        last_char = form_to_check[-1]
        candidate = form_to_check

        if last_char   == 'b':  candidate = form_to_check[:-1] + 'p'
        elif last_char == 'c':  candidate = form_to_check[:-1] + 'ç'
        elif last_char == 'd':  candidate = form_to_check[:-1] + 't'
        elif last_char == 'ğ':  candidate = form_to_check[:-1] + 'k'
        elif last_char == 'g':  candidate = form_to_check[:-1] + 'k'

        if (can_be_noun(candidate) or can_be_verb(candidate)) and candidate not in candidates:
            candidates.append(candidate)

    check_and_add_softened(surface_root)

    if 2 < len(surface_root) < 10 and ends_with_consonant(surface_root):
        prefix = surface_root[:-1]
        suffix_char = surface_root[-1]
        
        for vowel in ['ı', 'i', 'u', 'ü']:
            restored = prefix + vowel + suffix_char 
            if can_be_noun(restored) or can_be_verb(restored):
                candidates.append(restored)
            check_and_add_softened(restored)

    if not can_be_noun(surface_root) and len(surface_root) > 1:
        for terminal_vowel in ['a', 'e']:
            restored = surface_root + terminal_vowel
            if can_be_noun(restored) or can_be_verb(restored):
                 candidates.append(restored)

    # Consonant gemination reversal: hiss→his, hakk→hak, redd→ret
    # Common in Arabic/Persian loanwords where the final consonant doubles
    # before vowel-initial suffixes (hak→hakkı, his→hissi, ret→reddi)
    if (
        len(surface_root) >= 3
        and surface_root[-1] == surface_root[-2]
        and surface_root[-1] not in VOWELS
    ):
        degeminated = surface_root[:-1]
        if (can_be_noun(degeminated) or can_be_verb(degeminated)) and degeminated not in candidates:
            candidates.append(degeminated)
        check_and_add_softened(degeminated)

    return candidates
