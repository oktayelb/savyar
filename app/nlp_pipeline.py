"""Unified NLP Pipeline: Input Sanitation + Morphology Adaptation + Analyzer.

Handles the complete flow from raw text input down to step-by-step
morphology reconstructions and encoded view models for the downstream ML models.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import util.decomposer as sfx
from util.word_methods import tr_lower
from util.words.closed_class import (
    CLOSED_CLASS_LEXEMES,
    IRREGULAR_STEMS,
    irregular_stem,
    lexeme_of,
)
from ml.ml_ranking_model import (
    SUFFIX_OFFSET,
    GROUP_TO_ID,
    SPECIAL_FEATURE_ID,
    SPECIAL_ROOT_NOUN,
    SPECIAL_ROOT_VERB,
)

_APOSTROPHE_RE = re.compile(r"['’‘]")
_PUNCT_RE = re.compile(r"[^\w\s]|_")


def sanitize_word(raw: str) -> str:
    """Canonical form of a single-word input."""
    return tr_lower(_APOSTROPHE_RE.sub("", raw.strip()))


def sanitize_sentence(raw: str) -> List[str]:
    """Split a sentence into sanitized words (punct stripped, tr_lower'd)."""
    s = _APOSTROPHE_RE.sub("", raw)
    s = _PUNCT_RE.sub(" ", s)
    return tr_lower(s).split()


# --- Module-level ID lookup caches ---
def _build_caches():
    suffix_to_id = {s.name: idx + SUFFIX_OFFSET for idx, s in enumerate(sfx.ALL_SUFFIXES)}
    suffix_by_name = {s.name: s for s in sfx.ALL_SUFFIXES}
    lexeme_offset = SUFFIX_OFFSET + len(sfx.ALL_SUFFIXES)
    lexeme_to_id = {lexeme: lexeme_offset + idx for idx, lexeme in enumerate(CLOSED_CLASS_LEXEMES)}
    return suffix_to_id, suffix_by_name, lexeme_to_id

_SUFFIX_TO_ID, _SUFFIX_BY_NAME, _LEXEME_TO_ID = _build_caches()


def _suffix_names_attaching_to(start_pos: str) -> set:
    return {
        suffix.name
        for suffixes in sfx.SUFFIX_TRANSITIONS[start_pos].values()
        for suffix in suffixes
    }

_VERB_ONLY_SUFFIX_NAMES = _suffix_names_attaching_to("verb") - _suffix_names_attaching_to("noun")


def _expand_legacy_suffix_dicts(suffix_dicts: List[Dict]) -> List[Dict]:
    expanded = []
    for sd in suffix_dicts:
        if sd.get('name') == 'nondoing_meden':
            expanded.append({'name': 'infinitive_me', 'makes': 'NOUN'})
            expanded.append({'name': 'ablative_den', 'makes': 'NOUN'})
            continue
        expanded.append(sd)
    return expanded


def _normalize_entry_suffix_names(suffix_dicts: List[Dict]) -> List[str]:
    return [sd['name'] for sd in _expand_legacy_suffix_dicts(suffix_dicts)]


def match_decompositions(entries: List[Dict], decompositions: List[Tuple]) -> List[int]:
    """Matches logged decomposition entries against dynamically generated decompositions."""
    indices = []
    for entry in entries:
        entry_root = entry['root']
        entry_suffixes = _normalize_entry_suffix_names(entry.get('suffixes', []))
        for idx, (root, _, chain, _) in enumerate(decompositions):
            if root != entry_root:
                continue
            chain_suffixes = [s.name for s in chain] if chain else []
            if chain_suffixes == entry_suffixes and idx not in indices:
                indices.append(idx)
                break
    return indices


def build_suffix_log_info(word: str, decomposition: Tuple) -> List[Dict[str, Any]]:
    """Build log-friendly suffix metadata for a selected decomposition."""
    root, _pos, chain, _final_pos = decomposition

    if not chain:
        return []

    word_lower = tr_lower(word)
    current = root
    cursor = len(root)
    accepted_chain = []
    suffix_info: List[Dict[str, Any]] = []
    start_idx = 0

    stem = irregular_stem(word_lower, root, chain[0].name)
    if stem:
        first_suffix = chain[0]
        suffix_info.append({
            'name': first_suffix.name,
            'form': IRREGULAR_STEMS[stem][2],
            'makes': first_suffix.makes.name if first_suffix.makes else None,
        })
        current = stem
        cursor = len(stem)
        accepted_chain.append(first_suffix)
        start_idx = 1
    elif not word_lower.startswith(root):
        first_suffix = chain[0]
        possible_forms = first_suffix.form(root, current_chain=[])
        for offset in range(3):
            test_cursor = len(root) - offset
            if test_cursor <= 0:
                break
            rest_of_word = word_lower[test_cursor:]
            if any(form and rest_of_word.startswith(form) for form in possible_forms):
                cursor = test_cursor
                break

    for idx in range(start_idx, len(chain)):
        suffix = chain[idx]
        forms = suffix.form(current, current_chain=accepted_chain)
        rest = word_lower[cursor:]
        used_form = ""

        for form in forms:
            if form and rest.startswith(form):
                used_form = form
                break

        if not used_form:
            has_iyor_ahead = any("iyor" in chain[k].name for k in range(idx + 1, len(chain)))
            if has_iyor_ahead:
                for form in forms:
                    if form and form[-1] in ['a', 'e']:
                        shortened = form[:-1]
                        if shortened and rest.startswith(shortened):
                            rest_after = rest[len(shortened):]
                            if any(rest_after.startswith(v) for v in sfx.IYOR_VARIATIONS):
                                used_form = shortened if suffix.makes.name == "VERB" else form
                                break

        if not used_form:
            used_form = forms[0] if forms else ""

        replaces_final_vowel = (
            suffix.name == "continuous_iyor"
            and current.endswith(("a", "e"))
            and cursor == len(current) - 1
        )
        suffix_info.append({
            'name': suffix.name,
            'form': used_form,
            'makes': suffix.makes.name if suffix.makes else None,
        })
        if replaces_final_vowel:
            current = current[:-1] + used_form
        else:
            current += used_form
        cursor += len(used_form)
        accepted_chain.append(suffix)

    return suffix_info


def root_token(root: str, root_pos: str) -> Tuple[int, int, int]:
    lexeme = lexeme_of(root) if root_pos == "noun" else None
    if lexeme is not None:
        token_id = _LEXEME_TO_ID[lexeme]
    elif root_pos == "verb":
        token_id = SPECIAL_ROOT_VERB
    else:
        token_id = SPECIAL_ROOT_NOUN
    return (token_id, SPECIAL_FEATURE_ID, 1)


def root_pos_from_suffix_names(suffix_dicts: List[Dict]) -> str:
    suffix_dicts = _expand_legacy_suffix_dicts(suffix_dicts)
    if suffix_dicts and suffix_dicts[0]['name'] in _VERB_ONLY_SUFFIX_NAMES:
        return "verb"
    return "noun"


def encode_suffix_names(suffix_dicts: List[Dict], root: str, root_pos: str) -> List[Tuple[int, int, int]]:
    """Encode suffix chain directly from JSONL suffix dicts (name/makes strings)."""
    encoded = [root_token(root, root_pos)]
    suffix_dicts = _expand_legacy_suffix_dicts(suffix_dicts)
    for idx, sd in enumerate(suffix_dicts):
        name = sd['name']
        if name not in _SUFFIX_TO_ID:
            raise ValueError(f"Unknown suffix name in training data: {name!r}")
        token_id = _SUFFIX_TO_ID[name]
        suffix_obj = _SUFFIX_BY_NAME.get(name)
        group_id = GROUP_TO_ID.get(getattr(suffix_obj, 'group', None), SPECIAL_FEATURE_ID)
        encoded.append((
            token_id, group_id, idx + 2,
        ))
    return encoded


def encode_suffix_chain(suffix_chain: List, root: str, root_pos: str) -> List[Tuple[int, int, int]]:
    """Encodes a root and its suffix chain into ML token feature tuples."""
    encoded = [root_token(root, root_pos)]
    for idx, s in enumerate(suffix_chain):
        token_id = _SUFFIX_TO_ID.get(s.name, SUFFIX_OFFSET)
        encoded.append((
            token_id,
            GROUP_TO_ID.get(getattr(s, 'group', None), SPECIAL_FEATURE_ID),
            idx + 2,
        ))
    return encoded


def reconstruct_morphology(word: str, decomposition: Tuple) -> Dict[str, Any]:
    """Reconstructs the step-by-step morphology string from a root and suffix chain."""
    root, pos, chain, final_pos = decomposition

    if not chain:
        verb_marker = "-" if pos == "verb" else ""
        return {
            'root_str':      f"{root} ({pos})",
            'final_pos':     final_pos,
            'has_chain':     False,
            'formation_str': f"{root}{verb_marker} (no suffixes)",
        }
    
    current_stem = root
    suffix_forms = []
    suffix_names = []
    formation    = [root + ("-" if pos == "verb" else "")]
    
    cursor    = len(root)
    start_idx = 0
    
    stem = irregular_stem(word, root, chain[0].name)
    if stem:
        suffix_forms.append(IRREGULAR_STEMS[stem][2])
        suffix_names.append(chain[0].name)
        current_stem = stem
        formation.append(stem)
        cursor = len(stem)
        start_idx = 1

    if chain[0].name == "pekistirme":
        root_idx = word.find(root)
        if root_idx > 0:
            prefix_str = word[:root_idx]
            suffix_forms.append(prefix_str)
            suffix_names.append(chain[0].name)
            current_stem = prefix_str + root
            formation.append(current_stem)
            cursor    = root_idx + len(root)
            start_idx = 1

    if start_idx == 0:
        if not word.startswith(root) and chain:
            first_suffix = chain[0]
            possible_forms = first_suffix.form(root, current_chain=[])
            match_found = False
            for offset in range(3):
                test_cursor = len(root) - offset
                if test_cursor <= 0:
                    break
                rest_of_word = word[test_cursor:]
                for form in possible_forms:
                    if rest_of_word.startswith(form):
                        cursor     = test_cursor
                        match_found = True
                        break
                if match_found:
                    break

    for i in range(start_idx, len(chain)):
        suffix_obj     = chain[i]
        possible_forms = suffix_obj.form(current_stem, current_chain=chain[:i])
        found_form     = None 
        
        for form in possible_forms:
            if word.startswith(form, cursor):
                found_form = form
                break
        
        if found_form is None:
            has_iyor_ahead = any("iyor" in chain[k].name for k in range(i + 1, len(chain)))
            if has_iyor_ahead:
                for form in possible_forms:
                    if form and form[-1] in ['a', 'e']:
                        shortened = form[:-1]
                        if word.startswith(shortened, cursor):
                            found_form = shortened
                            break

        if found_form is None:
            for form in possible_forms:
                if len(form) > 0 and word.startswith(form, cursor - 1):
                    found_form = form
                    cursor -= 1
                    break
        
        if found_form is None:
            if possible_forms:
                suffix_forms.append(possible_forms[0] + "?")
                suffix_names.append(suffix_obj.name)
                current_stem += possible_forms[0]
                cursor       += len(possible_forms[0])
            continue
        
        replaces_final_vowel = (
            suffix_obj.name == "continuous_iyor"
            and current_stem.endswith(("a", "e"))
            and cursor == len(current_stem) - 1
        )

        suffix_forms.append(found_form if found_form else "(ø)")
        suffix_names.append(suffix_obj.name)
        if replaces_final_vowel:
            current_stem = current_stem[:-1] + found_form
        else:
            current_stem += found_form
        cursor       += len(found_form)
        
        verb_marker = "-" if suffix_obj.makes.name == "Verb" else ""
        formation.append(current_stem + verb_marker)
        
    return {
        'root_str':      f"{root} ({pos})",
        'final_pos':     final_pos,
        'has_chain':     True,
        'suffixes_str':  ' + '.join(suffix_forms),
        'names_str':     ' + '.join(suffix_names),
        'formation_str': ' → '.join(formation),
    }


def format_detailed_decomp(word: str, decomp: Tuple) -> str:
    """Formats decomposition to include both suffix name and specific surface form."""
    root, pos, chain, final_pos = decomp
    
    if not chain:
        return root

    parts = [root]
    current = root
    accepted_chain = []
    remaining_chain = chain
    stem = irregular_stem(word, root, chain[0].name)
    if stem:
        parts.append(f"{chain[0].name}_{IRREGULAR_STEMS[stem][2]}")
        current = stem
        accepted_chain.append(chain[0])
        remaining_chain = chain[1:]

    for suffix in remaining_chain:
        forms = suffix.form(current, current_chain=accepted_chain)
        
        # Use getattr as a safety net to prevent AttributeError
        used_form = forms[0] if forms else getattr(suffix, 'suffix', '')
        
        if used_form:
            parts.append(f"{suffix.name}_{used_form}")
        else:
            parts.append(suffix.name)
            
        current += used_form
        accepted_chain.append(suffix)
        
    return "+".join(parts)


def analyze_word(word: str) -> Dict[str, Any]:
    """Decompose one sanitized word and bundle everything downstream needs."""
    decomps = sfx.decompose_with_fallback(word)

    encoded_chains: List[List] = []
    vms: List[Dict[str, Any]] = []
    typing_strings: List[str] = []

    for decomp in decomps:
        root, root_pos, chain, _ = decomp
        encoded_chains.append(encode_suffix_chain(chain, root, root_pos))
        vm = reconstruct_morphology(word, decomp)
        vms.append(vm)
        if vm.get('has_chain'):
            typing_strings.append(f"{root} {vm['suffixes_str'].replace(' + ', ' ')}")
        else:
            typing_strings.append(root)

    return {
        'word': word,
        'decomps': decomps,
        'encoded_chains': encoded_chains,
        'vms': vms,
        'typing_strings': typing_strings,
    }


def analyze_word_with_root(word: str, root: str) -> Dict[str, Any]:
    """analyze_word() for a word whose lemma is already known.

    Used where annotated data supplies the root, so that a word the lexicon
    cannot reach still arrives with the alternatives it has to be ranked
    against instead of its gold answer alone.
    """
    decomps = sfx.decompose_with_root(word, root)
    return {
        'word': word,
        'decomps': decomps,
        'encoded_chains': [encode_suffix_chain(chain, root, root_pos) for root, root_pos, chain, _f in decomps],
    }


def analyze_words(words: List[str]) -> List[Dict[str, Any]]:
    return [analyze_word(w) for w in words]


def score_and_sort(analysis: Dict[str, Any], trainer) -> Optional[List[float]]:
    """Rank an analysis's candidates by ML score (highest first)."""
    if len(analysis['decomps']) <= 1:
        return None

    try:
        _, scores = trainer.predict(analysis['encoded_chains'])
    except Exception:
        return None

    for i, vm in enumerate(analysis['vms']):
        vm['score'] = scores[i]

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    def _reorder(lst):
        return [lst[i] for i in order]

    analysis['decomps'] = _reorder(analysis['decomps'])
    analysis['encoded_chains'] = _reorder(analysis['encoded_chains'])
    analysis['vms'] = _reorder(analysis['vms'])
    analysis['typing_strings'] = _reorder(analysis['typing_strings'])

    return [scores[i] for i in order]
