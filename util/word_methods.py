from enum import Enum
from pathlib import Path
import random
from typing import List, Tuple, Optional

from util.words.closed_class import (
    CLOSED_CLASS_LEXEMES,
    LEXEME_SPELLINGS,
    UNINFLECTED_LEXEMES,
    lexeme_of,
)
from util.words.root_classes import VOWEL_DROPPING_ROOTS

_TR_LOWER_TABLE = str.maketrans("İI", "iı")

def tr_lower(s: str) -> str:
    """Lowercase a Turkish string correctly: İ→i, I→ı."""
    return s.translate(_TR_LOWER_TABLE).lower()

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATA_FILE = _DATA_DIR / "words.txt"
VERB_DATA_FILE = _DATA_DIR / "verbs.txt"
UNSUFFIXABLE_FILE = _DATA_DIR / "ekistemez.txt"

NOUN_FILES = (DATA_FILE,)
VERB_FILES = (VERB_DATA_FILE,)


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
# The *_SET sets hold the entries exactly as the data files write them, because
# they are what gets written back on delete. Lookups never touch them directly:
# they go through the folded key indexes below.
WORDS_SET: set = set()
VERB_SET: set = set()
UNSUFFIXABLE_SET: set = set()

# Which file each entry came from, so a delete rewrites only the file that
# owns the word.
_FILE_ENTRIES: dict = {}

_NOUN_KEYS: set = set()
_VERB_KEYS: set = set()
_UNSUFFIXABLE_KEYS: set = set()
_PALATAL_L_KEYS: set = set()


def lexicon_key(word: str) -> str:
    """The form a word is looked up under.

    Every word that reaches the decomposer has been tr_lower()'d by the input
    pipeline, while words.txt keeps proper nouns capitalised (Ankara, Türkiye,
    İstanbul). Folding both sides through this one function is what lets
    "ankara" find "Ankara"; it is also the single place to add any further
    difference the lexicon should not care about.
    """
    return tr_lower(word).replace("ł", "l")


def has_palatal_l(entry: str) -> bool:
    folded = tr_lower(entry)
    last_vowel = max((i for i, ch in enumerate(folded) if ch in VOWELS), default=-1)
    return "ł" in folded[last_vowel + 1:]


def _reindex_dictionary():
    """Rebuild the folded lookup indexes from the loaded entries."""
    global _NOUN_KEYS, _VERB_KEYS, _UNSUFFIXABLE_KEYS, _PALATAL_L_KEYS
    noun_keys = {lexicon_key(word) for word in WORDS_SET}
    noun_keys.update(CLOSED_CLASS_LEXEMES)
    noun_keys.update(LEXEME_SPELLINGS)
    _NOUN_KEYS = noun_keys
    _VERB_KEYS = {lexicon_key(word) for word in VERB_SET}
    _UNSUFFIXABLE_KEYS = {lexicon_key(word) for word in UNSUFFIXABLE_SET}
    _PALATAL_L_KEYS = {lexicon_key(word) for word in WORDS_SET | VERB_SET if has_palatal_l(word)}


def _read_entries(path) -> set:
    """One entry per line; a file that is not there yet is simply empty."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        return set()


def _rebuild_sets():
    """Recombine the per-file entries into the two lookup sets."""
    global WORDS_SET, VERB_SET
    WORDS_SET = set().union(*(_FILE_ENTRIES[path] for path in NOUN_FILES))
    VERB_SET = set().union(*(_FILE_ENTRIES[path] for path in VERB_FILES))
    _reindex_dictionary()


def _load_dictionary():
    global UNSUFFIXABLE_SET, _FILE_ENTRIES
    _FILE_ENTRIES = {path: _read_entries(path) for path in NOUN_FILES + VERB_FILES}
    UNSUFFIXABLE_SET = _read_entries(UNSUFFIXABLE_FILE)
    if not _FILE_ENTRIES[DATA_FILE] or not _FILE_ENTRIES[VERB_DATA_FILE]:
        print(f"Warning: {DATA_FILE} or {VERB_DATA_FILE} not found")
    _rebuild_sets()


def save_dictionary() -> bool:
    """Write every lexicon file back from the entries it owns."""
    try:
        for path, entries in _FILE_ENTRIES.items():
            if not entries and not Path(path).exists():
                continue
            with open(path, "w", encoding="utf-8") as f:
                for entry in sorted(entries):
                    f.write(entry + "\n")
        return True
    except OSError:
        return False

# Initialize on module load
_load_dictionary()


def _in_index(word: str, index: set) -> bool:
    """Membership under lexicon_key(), without folding what is already folded.

    This is the hottest lookup in the project - decompose() runs it on every
    prefix of every word - so the folded form is only built when the word
    actually carries case the index cannot have: the pipeline hands over
    sanitized words and the decomposer slices its roots out of them, so in
    practice the first test decides.
    """
    if word in index:
        return True
    return (not word.islower() or "ł" in word) and lexicon_key(word) in index


def delete_word(word: str) -> bool:
    """Removes a word from the in-memory dictionary state."""
    key = lexicon_key(word.strip())
    if not key:
        return False

    removed = False
    for entries in _FILE_ENTRIES.values():
        for entry in [e for e in entries if lexicon_key(e) == key]:
            entries.discard(entry)
            removed = True

    if removed:
        _rebuild_sets()
    return removed

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


def is_unsuffixable(word: str) -> bool:
    """Interjections and particles that never take a suffix (ha, çüş, ki)."""
    return _in_index(word, _UNSUFFIXABLE_KEYS)


def is_uninflected_noun(word: str) -> bool:
    return lexeme_of(word) in UNINFLECTED_LEXEMES

def can_be_noun(word: str) -> bool:
    if not word:
        return False
    return _in_index(word, _NOUN_KEYS)

def can_be_verb(word: str) -> bool:
    """Checks if a root is a verb by looking it up in the verb index."""
    return _in_index(word, _VERB_KEYS)

# --- Harmony functions ---
def major_harmony(word: str) -> MajorHarmony | None:
    """Determines major vowel harmony based on last vowel"""
    if _in_index(word, _PALATAL_L_KEYS):
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

def progressive_narrow_vowel(stem: str) -> str | None:
    """Return the high vowel used when final a/e narrows before -yor."""
    if not stem or stem[-1] not in ['a', 'e']:
        return None

    for ch in reversed(stem[:-1]):
        if ch in BACK_ROUND:
            return 'u'
        if ch in FRONT_ROUND:
            return 'ü'
        if ch in BACK_FLAT:
            return 'ı'
        if ch in FRONT_FLAT:
            return 'i'

    return 'ı' if stem[-1] == 'a' else 'i'

# --- Morphological utilities ---

def ends_with_consonant(word: str) -> bool:
    """Check if word ends with a consonant"""
    return word and word[-1] not in VOWELS


SOFTENED_TO_HARD = {'b': 'p', 'c': 'ç', 'd': 't', 'ğ': 'k', 'g': 'k'}


def unsoftened(form: str) -> str:
    if form and form[-1] in SOFTENED_TO_HARD:
        return form[:-1] + SOFTENED_TO_HARD[form[-1]]
    return form


def get_root_candidates(surface_root: str) -> List[str]:
    """Analyzes the text segment and returns Surface Forms that are Dictionary Lemmas."""
    candidates = [] 

    def check_and_add_softened(form_to_check):
        if not form_to_check: return

        candidate = unsoftened(form_to_check)

        if (can_be_noun(candidate) or can_be_verb(candidate)) and candidate not in candidates:
            candidates.append(candidate)

    check_and_add_softened(surface_root)

    if 2 < len(surface_root) < 10 and ends_with_consonant(surface_root):
        prefix = surface_root[:-1]
        suffix_char = surface_root[-1]
        
        for vowel in ['ı', 'i', 'u', 'ü']:
            restored = prefix + vowel + suffix_char
            for lemma in (restored, unsoftened(restored)):
                if lemma in VOWEL_DROPPING_ROOTS and exists(lemma) and lemma not in candidates:
                    candidates.append(lemma)

    if not can_be_noun(surface_root) and len(surface_root) > 1:
        for terminal_vowel in ['a', 'e']:
            restored = surface_root + terminal_vowel
            if can_be_noun(restored) or can_be_verb(restored):
                 candidates.append(restored)
    elif len(surface_root) == 1:
        for terminal_vowel in ['a', 'e']:
            restored = surface_root + terminal_vowel
            if can_be_verb(restored):
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
