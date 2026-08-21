from typing import List, Tuple, Set, Optional
import functools

# ============================================================================
# IMPORTS
# ============================================================================
from util.suffixes.v2n.nounifiers import Nounifier
from util.suffixes.v2v_suffixes import VERB2VERB
from util.suffixes.n2v_suffixes import NOUN2VERB
from util.suffixes.n2n_suffixes import NOUN2NOUN
from util.suffixes.v2n_suffixes import VERB2NOUN

import util.word_methods as wrd
from util.suffix import Type, Suffix, SuffixGroup

ALL_SUFFIXES = NOUN2NOUN + NOUN2VERB + VERB2NOUN + VERB2VERB
IYOR_VARIATIONS = ('iyor', 'ıyor', 'uyor', 'üyor')

# ============================================================================
# SUFFIX TYPES
# ============================================================================
SUFFIX_TRANSITIONS = {
    'noun': {
        # Standard N->N + (V->N suffixes that also accept Nouns)
        'noun': NOUN2NOUN + [s for s in VERB2NOUN if s.comes_to == Type.BOTH],
        
        # Standard N->V + (V->V suffixes that also accept Nouns)
        'verb': NOUN2VERB + [s for s in VERB2VERB if s.comes_to == Type.BOTH]
    },
    'verb': {
        # Standard V->N + (N->N suffixes that also accept Verbs)
        'noun': VERB2NOUN + [s for s in NOUN2NOUN if s.comes_to == Type.BOTH],
        
        # Standard V->V + (N->V suffixes that also accept Verbs)
        'verb': VERB2VERB + [s for s in NOUN2VERB if s.comes_to == Type.BOTH]
    }
}

# ============================================================================
# SUFFIX HIERARCHY RULES
# ============================================================================
def is_valid_transition(last_suffix: Suffix, next_suffix: Suffix) -> bool:
    """
    Implements the State Machine logic for Turkish Agglutination.
    """
    last_g = last_suffix.group
    next_g = next_suffix.group


    ## main waterfall rule.
   

    # --- RULE 1: The Post-Case Loop (-ki Exception) ---
    if last_g == SuffixGroup.MARKING_KI and next_g <= SuffixGroup.MARKING_KI:
        return True
    
    ## fiillerle alakalı kısım burası
    ## şelale akışına istisna olarak  yapım eklerinden sonra fiil ekleri gelebilir 
    if last_g == SuffixGroup.N2N_DERIVATIONAL and next_g <= SuffixGroup.N2N_DERIVATIONAL:
        return True   
    
    ## ebilmekten gibi eklerden sonra  fiil ekleri gelebilir.  gidebilmeyen gitmeyebilmek
    if last_g == SuffixGroup.VERB_COMPOUND and next_g <= SuffixGroup.VERB_COMPOUND:
        return True

    if last_g == SuffixGroup.VERB_NEGATING and type(next_suffix) == Nounifier:
        return False   

    # --- RULE 4: Self-Looping Constraints ---
    if next_g == last_g:
        if last_g in [SuffixGroup.N2N_DERIVATIONAL, SuffixGroup.V2V_DERIVATIONAL, SuffixGroup.PREDICATIVE]:
            return True
        return False
    
    # zarf fiilden sonra bir şey gelmez. erekten örneği gözardı ediliyor.
    # Sadece  ol-a -lım  ol-a-yım ol-a-sın örnekleri için conjugation alabilir kabul edilecek
    if last_g is SuffixGroup.VERB_TO_ADVERB and next_g != SuffixGroup.CONJUGATION: 
        return False
    
    if last_g == SuffixGroup.CONJUGATION and next_g == SuffixGroup.PREDICATIVE:
        if last_suffix.name == "conjugation_3pl":
            return True
    ## şelale hallediyor diye silindi
    # isim tamlamasından sonra yalnızca ki gelebilir
    # if last_g == SuffixGroup.CASE and not next_g >= SuffixGroup.MARKING_KI:
    #    return False
    
    if next_g < last_g:
        return False

    return True

pekistirme_suffix = Suffix("pekistirme", "pekistirme", Type.NOUN, Type.NOUN, is_unique=True)

## TODO hatalı
def get_pekistirme_analyses(word: str) -> List[Tuple]:


    analyses = []
    
    # Fast exit
    if len(word) < 4: 
        return analyses

    # 1. Detect first wowel 
    first_vowel_index = -1
    for i in range(len(word)):
        if word[i] in wrd.VOWELS:
            first_vowel_index = i
            break
            
    if first_vowel_index == -1 or (first_vowel_index + 1) >= len(word):
        return analyses


    # 2. Strategy: Try to find a valid (Prefix, Root) pair
    detected_root = None
    detected_prefix = None
    
    # Case A: Standard (mas-mavi) -> Split at index + 2
    idx_std = first_vowel_index + 2
    # Case B: Extended (gü-p-e-gündüz) -> Split at index + 3 (if connecting vowel exists)
    idx_ext = first_vowel_index + 3

    # Helper to scan for root
    def find_root_in_rest(prefix_len):
        potential_rest = word[prefix_len:]
        # The rest must start with the duplicated part (e.g. 'mas-mavi' -> 'mavi' starts with 'ma')
        if potential_rest.startswith(word[:first_vowel_index + 1]):
            # Greedy search: match longest possible valid dictionary word
            for k in range(len(potential_rest), 1, -1):
                candidate = potential_rest[:k]
                if wrd.can_be_noun(candidate) or wrd.can_be_verb(candidate):
                    return word[:prefix_len], candidate
        return None, None

    # Try Case A
    detected_prefix, detected_root = find_root_in_rest(idx_std)

    # Try Case B if A failed, checking for connecting vowel (a/e)
    if not detected_root and idx_ext < len(word):
        if word[first_vowel_index + 2] in ['a', 'e']:
             detected_prefix, detected_root = find_root_in_rest(idx_ext)

    if not detected_root:
        return analyses

    # 3. Build Analysis Chain
    # Scenario 1: Pure Pekiştirme (e.g., "masmavi")
    if len(detected_prefix + detected_root) == len(word):
        analyses.append((detected_root, "noun", [pekistirme_suffix], "noun"))
    
    # Scenario 2: Pekiştirme with suffixes (e.g., "masmaviyim")
    else:
        # Extract the logical part for suffix analysis: "mas" + "maviyim"
        virtual_word_part = word[len(detected_prefix):] # e.g., "maviyim"
        
        # Recursive call: Find suffixes attached to the root "mavi" within "maviyim"
        chains = find_suffix_chain(virtual_word_part, "noun", detected_root)
        
        for chain, final_pos in chains:
            full_chain = [pekistirme_suffix] + chain
            analyses.append((detected_root, "noun", full_chain, final_pos))

    return analyses

def _suffix_iter(start_pos, root, current_chain):
    """Try every suffix and compute forms with the current suffix chain."""
    for target_pos, candidate_suffixes in SUFFIX_TRANSITIONS[start_pos].items():
        for suffix_obj in candidate_suffixes:
            yield target_pos, suffix_obj, suffix_obj.form(root, current_chain=current_chain)


def find_suffix_chain(word: str, start_pos: str, root: str,
                      current_chain: List = None, visited: Set = None,
                      shared_cache: dict = None) -> List:
    """
    Recursive suffix chain finder with optional shared cross-root cache.
    Cache key includes the current surface and suffix-chain signature because
    suffix form functions can depend on both.
    """

    if current_chain is None: current_chain = []
    if visited is None: visited = set()
    if shared_cache is None: shared_cache = {}

    # --- Per-call memoization (prevents revisiting within one call tree) ---
    chain_signature = tuple(s.name for s in current_chain)
    state_key = (root, start_pos, chain_signature)
    if state_key in visited: return []
    visited.add(state_key)

    root_len = len(root)
    rest = word[root_len:]

    # Base Case
    if not rest:
        return [([], start_pos)]

    if start_pos not in SUFFIX_TRANSITIONS:
        return []

    # --- Shared cross-root cache ---
    # Key includes the suffix chain because form functions can depend on it.
    cache_key = (root, rest, start_pos, chain_signature)

    if cache_key in shared_cache:
        return shared_cache[cache_key]

    results = []

    _iter = _suffix_iter(start_pos, root, current_chain)

    for target_pos, suffix_obj, suffix_forms in _iter:

            # --- HIERARCHY VALIDATION ---
            if current_chain:
                last_suffix = current_chain[-1]
                if not is_valid_transition(last_suffix, suffix_obj):
                    continue

            # --- UNIQUENESS CHECK ---
            if suffix_obj.is_unique:
                if any(s.name == suffix_obj.name for s in current_chain):
                    continue

            # --- FORM MATCHING ---
            for suffix_form in suffix_forms:
                sf_len = len(suffix_form)
                if sf_len > len(rest):
                    continue

                # --- MATCH TYPE 1: Standard ---
                if rest.startswith(suffix_form):
                    subchains = find_suffix_chain(
                        word, target_pos,
                        word[:root_len + sf_len],
                        current_chain + [suffix_obj],
                        visited,
                        shared_cache
                    )
                    for chain, final_pos in subchains:
                        results.append(([suffix_obj] + chain, final_pos))

                # --- MATCH TYPE 2: Vowel Narrowing ---
                elif sf_len > 0 and suffix_form[-1] in ('a', 'e'):
                    narrowed_form = suffix_form[:-1]
                    if rest.startswith(narrowed_form):
                        rest_after = rest[len(narrowed_form):]
                        if any(rest_after.startswith(v) for v in IYOR_VARIATIONS):
                            if suffix_obj.makes == Type.VERB and narrowed_form:
                                next_word = word
                                next_root = word[:root_len + len(narrowed_form)]
                            else:
                                next_word = root + suffix_form + rest_after
                                next_root = root + suffix_form
                            subchains = find_suffix_chain(
                                next_word,
                                target_pos,
                                next_root,
                                current_chain + [suffix_obj],
                                visited,
                                shared_cache
                            )
                            for chain, final_pos in subchains:
                                results.append(([suffix_obj] + chain, final_pos))

    # Store in shared cache before returning
    # IMPORTANT: only cache if there are no unique suffixes in current chain,
    # because uniqueness constraints make results chain-dependent.
    has_unique_in_chain = any(s.is_unique for s in current_chain)
    if not has_unique_in_chain:
        shared_cache[cache_key] = results

    return results


def decompose_with_cc(word: str) -> List[Tuple]:
    """
    Like decompose(), but also includes closed-class word analyses.

    For each closed-class interpretation of `word` (pronoun, conjunction, etc.)
    a tuple is appended:
        (surface_form, "cc_<category>", [ClosedClassMarker(cc_obj)], "cc_<category>")

    This allows the ML model to see closed-class tokens in the sentence sequence,
    and allows the workflow to handle words (e.g. "ve", "ile") that the regular
    decomposer cannot match because they are not in words.txt as open-class roots.

    Regular suffix-chain decompositions are always included first.
    """
    from util.words.closed_class import CLOSED_CLASS_LOOKUP, ClosedClassMarker

    analyses = list(decompose(word))

    cc_entries = CLOSED_CLASS_LOOKUP.get(word, [])
    seen_categories: set = set()
    for cc_obj in cc_entries:
        cat_key = (cc_obj.category, cc_obj.word)
        if cat_key in seen_categories:
            continue
        seen_categories.add(cat_key)
        pos_tag = f"cc_{cc_obj.category}"
        analyses.append((word, pos_tag, [ClosedClassMarker(cc_obj, surface_form=word)], pos_tag))

    return analyses


def append_analysis(word, pos, root, analyses_list, shared_cache: dict = None):
    possible_chains = find_suffix_chain(word, pos, root, shared_cache=shared_cache)
    for chain, final_pos in possible_chains:
        analyses_list.append((root, pos, chain, final_pos))


def append_progressive_vowel_drop_candidates(word: str, surface_root: str, analyses_list, shared_cache: dict = None):
    rest = word[len(surface_root):]
    if not any(rest.startswith(v) for v in IYOR_VARIATIONS):
        return

    for terminal_vowel in ['a', 'e']:
        lemma_root = surface_root + terminal_vowel
        if lemma_root in wrd.UNSUFFIXABLE_SET:
            continue
        if wrd.can_be_verb(lemma_root):
            append_analysis(lemma_root + rest, "verb", lemma_root, analyses_list, shared_cache)


@functools.lru_cache(maxsize=100000)
def decompose(word: str,  force: Optional[bool] = False) -> List[Tuple]:
    """
    Finds all possible root-suffix decompositions for a word.\n
    Uses a shared cache across all root iterations to avoid recomputing
    suffix chains for the same remaining text + POS + last_group context.\n
    The lru_cache rapidly short-circuits re-evaluations across entire files.
    """
    if (not force) and wrd.is_non_ben_pronoun_surface(word):
        return []

    # Shared across all append_analysis calls in this decompose invocation
    shared_cache = {}

    analyses = get_pekistirme_analyses(word)

    for i in range(1, len(word) + 1):
        root = word[:i]

        if root in wrd.UNSUFFIXABLE_SET:
            continue


        if force or wrd.can_be_noun(root) :
            append_analysis(word, "noun", root, analyses, shared_cache)

        if force or wrd.can_be_verb(root):
            append_analysis(word, "verb", root, analyses, shared_cache)

        if (not force ) and (not wrd.exists(root)):
            root_pairs = wrd.get_root_candidates(word[:i])
            for lemma_root in root_pairs:
                virtual_word = lemma_root + word[i:]

                if wrd.can_be_noun(lemma_root):
                    append_analysis(virtual_word, "noun", lemma_root, analyses, shared_cache)

                if wrd.can_be_verb(lemma_root):
                    append_analysis(virtual_word, "verb", lemma_root, analyses, shared_cache)

        if (not force) and wrd.exists(root):
            append_progressive_vowel_drop_candidates(word, root, analyses, shared_cache)

    return analyses
