"""
Suffix Index — optional acceleration layer for the decomposer.

Builds lookup structures at import time so find_suffix_chain can skip
irrelevant suffixes quickly. The decomposer can use this or ignore it;
all public functions are pure helpers with no side effects on the
suffix definitions themselves.

Usage in decomposer.py:
    from util.suffix_index import SuffixIndex
    _INDEX = SuffixIndex()          # build once at module level
    _INDEX.get_candidates(...)      # inside find_suffix_chain

To disable: simply stop calling _INDEX methods. The decomposer's
correctness never depends on this module.
"""

from typing import Dict, List, Tuple, Optional
from util.suffix import Suffix, SuffixGroup
import util.word_methods as wrd


# 8 vowel classes + "no vowel" + hard/soft ending = key space
def _vowel_key(word: str) -> Tuple[Optional[str], bool]:
    """Return (last_vowel_or_None, ends_with_hard_consonant) for a word."""
    last_vowel = None
    for ch in reversed(word):
        if ch in wrd.VOWELS:
            last_vowel = ch
            break
    hard_end = bool(word) and word[-1] in wrd.HARD_CONSONANTS
    return (last_vowel, hard_end)


def _ends_with_vowel(word: str) -> bool:
    return bool(word) and word[-1] in wrd.VOWELS


class SuffixIndex:
    """
    Pre-computed index for fast suffix candidate lookup.

    Indexes suffixes by (start_pos, target_pos, first_char_of_form).
    Since suffix forms depend on vowel harmony and consonant hardening,
    the index pre-computes forms for every (suffix, vowel_class) pair
    and stores them grouped by first character.
    """

    # Representative stems: one per (last_vowel, hard_ending, vowel_ending) combo.
    # We only need the last vowel + ending characteristics for harmony.
    _VOWEL_REPS = {
        'a': 'bak',  'ı': 'kır',  'o': 'kol',  'u': 'kul',
        'e': 'gel',  'i': 'bil',  'ö': 'göl',  'ü': 'gül',
    }
    # Hard-ending variants
    _HARD_REPS = {
        'a': 'at',  'ı': 'sıt', 'o': 'ot',  'u': 'kut',
        'e': 'et',  'i': 'it',  'ö': 'ört', 'ü': 'üt',
    }
    # Vowel-ending variants
    _VOWEL_END_REPS = {
        'a': 'ba',  'ı': 'sı',  'o': 'ko',  'u': 'su',
        'e': 'be',  'i': 'bi',  'ö': 'kö',  'ü': 'sü',
    }
    # Morpheme-tail reps: stems ending in -ma / -me built on a real verb root.
    # Needed to expose context-dependent forms (e.g. factative_ir emits the
    # -z negative-aorist form only when the preceding morpheme is ma/me and
    # stripping it leaves a valid verb root). Without these, the index never
    # learns about such forms and the decomposer silently misses chains.
    # Use synthetic rep_keys ("ma_tail", "me_tail") so these don't collide
    # with the regular (last_vowel, hard_end, vowel_end) keys in _form_cache.
    # Each tail triggers context-dependent forms inside suffix.form functions
    # (e.g. conjugation_2sg emits 'n' only after -di/-dı/.../-se/-sa;
    # factative_ir emits 'z' only after -ma/-me with a valid verb before).
    # We cover all tails that any form_function in util/suffixes/**/*.py
    # peeks at via word[-2:] checks.
    _MORPH_TAIL_REPS = {
        'ma_tail': 'bakma',
        'me_tail': 'gelme',
        'di_tail': 'geldi',
        'dı_tail': 'bildı',   # synthetic: tail carrier only
        'du_tail': 'oldu',
        'dü_tail': 'gördü',
        'ti_tail': 'gitti',
        'tı_tail': 'baktı',
        'tu_tail': 'tuttu',
        'tü_tail': 'öptü',
        'se_tail': 'gelse',
        'sa_tail': 'baksa',
    }

    def __init__(self, suffix_transitions: dict = None):
        """
        Build index from SUFFIX_TRANSITIONS dict.
        If None, imports the global one from decomposer (lazy to avoid circular).
        """
        if suffix_transitions is None:
            from util.decomposer import SUFFIX_TRANSITIONS
            suffix_transitions = SUFFIX_TRANSITIONS

        # _dispatch[start_pos][target_pos][first_char] = list of (suffix_obj, form_str)
        # We store pre-computed forms per representative stem class.
        # At query time, we pick the right representative and look up.
        self._dispatch: Dict[str, Dict[str, Dict[str, List[Tuple[Suffix, str]]]]] = {}

        # _form_cache[(suffix.name, last_vowel, hard_end, vowel_end)] = [form_strings]
        self._form_cache: Dict[Tuple, List[str]] = {}

        self._transitions = suffix_transitions
        self._build()

    def _build(self):
        """Pre-compute suffix forms for all vowel classes and index by first char."""
        all_reps = self._all_representative_stems()

        for start_pos, targets in self._transitions.items():
            self._dispatch[start_pos] = {}
            for target_pos, suffixes in targets.items():
                char_map: Dict[str, List[Tuple[Suffix, str]]] = {}
                for suffix_obj in suffixes:
                    for rep_key, rep_stem in all_reps:
                        forms = self._cached_form(suffix_obj, rep_stem, rep_key)
                        for form_str in forms:
                            if not form_str:  # empty string (3sg)
                                ch = ''
                            else:
                                ch = form_str[0]
                            char_map.setdefault(ch, [])
                            # avoid duplicate (suffix, form) pairs
                            pair = (suffix_obj, form_str)
                            if pair not in char_map[ch]:
                                char_map[ch].append(pair)
                self._dispatch[start_pos][target_pos] = char_map

    def _all_representative_stems(self) -> List[Tuple[Tuple, str]]:
        """Return list of ((last_vowel, hard_end, vowel_end), stem_string)."""
        reps = []
        for v, stem in self._VOWEL_REPS.items():
            reps.append(((v, False, False), stem))
        for v, stem in self._HARD_REPS.items():
            reps.append(((v, True, False), stem))
        for v, stem in self._VOWEL_END_REPS.items():
            reps.append(((v, False, True), stem))
        for key, stem in self._MORPH_TAIL_REPS.items():
            reps.append(((key, False, stem.endswith(('a','e','ı','i','o','ö','u','ü'))), stem))
        return reps

    def _cached_form(self, suffix_obj: Suffix, stem: str, rep_key: Tuple) -> List[str]:
        cache_key = (suffix_obj.name, rep_key)
        if cache_key in self._form_cache:
            return self._form_cache[cache_key]
        try:
            forms = suffix_obj.form(stem)
        except Exception:
            forms = []
        self._form_cache[cache_key] = forms
        return forms

    def _classify_word(self, word: str) -> Tuple:
        """Classify a word into a representative key."""
        last_vowel = None
        for ch in reversed(word):
            if ch in wrd.VOWELS:
                last_vowel = ch
                break
        hard_end = bool(word) and word[-1] in wrd.HARD_CONSONANTS
        vowel_end = _ends_with_vowel(word)
        return (last_vowel, hard_end, vowel_end)

    def get_candidates(self, start_pos: str, rest: str, root: str
                       ) -> List[Tuple[str, Suffix, str]]:
        """
        Return [(target_pos, suffix_obj, form_str), ...] that could match `rest`.

        Filters by first character of `rest` to eliminate impossible suffixes.
        The caller still must do startswith and hierarchy checks.
        This is a HINT — it may include false positives but never false negatives
        for the given vowel class. The caller always recomputes the exact form.
        """
        if start_pos not in self._dispatch:
            return []

        first_char = rest[0] if rest else ''
        results = []
        for target_pos, char_map in self._dispatch[start_pos].items():
            # Exact first-char match
            if first_char in char_map:
                for suffix_obj, form_str in char_map[first_char]:
                    results.append((target_pos, suffix_obj, form_str))
            # Always include empty-string forms (3sg etc.)
            if first_char != '' and '' in char_map:
                for suffix_obj, form_str in char_map['']:
                    results.append((target_pos, suffix_obj, form_str))
        return results

    def form_for(self, suffix_obj: Suffix, root: str) -> List[str]:
        """
        Cached suffix form computation. Uses representative stem class
        to avoid redundant harmony calculations across similar roots.

        NOTE: This is an approximation cache. For suffixes with custom form
        functions that inspect more than just the last vowel + ending type,
        the result may differ from suffix_obj.form(root). The decomposer
        should fall back to suffix_obj.form(root) for correctness when needed.

        For the common case (harmony-only forms), this is exact.
        """
        rep_key = self._classify_word(root)
        return self._cached_form(suffix_obj, root, rep_key)
