"""Unified NLP Pipeline: Input Sanitation + Morphology Adaptation + Analyzer.

Handles the complete flow from raw text input down to step-by-step
morphology reconstructions and encoded view models for the downstream ML models.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import util.decomposer as sfx
from util.word_methods import tr_lower
from util.suffix import  Type
from util.words.closed_class import ClosedClassMarker, CLOSED_CLASS_TOKEN_SPECS
from ml.ml_ranking_model import (
    SUFFIX_OFFSET,
    CATEGORY_CLOSED_CLASS,
    GROUP_TO_ID,
    TYPE_TO_ID,
    SPECIAL_FEATURE_ID,
    WORD_FINAL_NO,
    WORD_FINAL_YES,
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
    cc_offset = SUFFIX_OFFSET + len(sfx.ALL_SUFFIXES)
    cc_surface_to_id = {
        (category, surface): cc_offset + idx
        for idx, (category, surface) in enumerate(CLOSED_CLASS_TOKEN_SPECS)
    }
    cc_name_to_default_id = {}
    for idx, (category, _surface) in enumerate(CLOSED_CLASS_TOKEN_SPECS):
        name = f"cc_{category}"
        if name not in cc_name_to_default_id:
            cc_name_to_default_id[name] = cc_offset + idx
    return suffix_to_id, suffix_by_name, cc_surface_to_id, cc_name_to_default_id, cc_offset

_SUFFIX_TO_ID, _SUFFIX_BY_NAME, _CC_SURFACE_TO_ID, _CC_NAME_TO_DEFAULT_ID, _CC_OFFSET = _build_caches()


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


def encode_suffix_names(suffix_dicts: List[Dict]) -> List[Tuple[int, int, int, int, int, int, int]]:
    """Encode suffix chain directly from JSONL suffix dicts (name/makes strings)."""
    category_to_id = {'NOUN': 0, 'VERB': 1, 'noun': 0, 'verb': 1, 'Noun': 0, 'Verb': 1}
    type_name_to_enum = {
        'NOUN': Type.NOUN, 'noun': Type.NOUN, 'Noun': Type.NOUN,
        'VERB': Type.VERB, 'verb': Type.VERB, 'Verb': Type.VERB,
        'BOTH': Type.BOTH, 'both': Type.BOTH, 'Both': Type.BOTH,
    }
    encoded = []
    suffix_dicts = _expand_legacy_suffix_dicts(suffix_dicts)
    last_idx = len(suffix_dicts) - 1
    for idx, sd in enumerate(suffix_dicts):
        name = sd['name']
        makes = sd.get('makes', 'NOUN')
        if name.startswith('cc_'):
            category = name[3:]
            surface = sd.get('cc_surface') or sd.get('root') or ""
            token_id = _CC_SURFACE_TO_ID.get(
                (category, surface),
                _CC_NAME_TO_DEFAULT_ID.get(name, _CC_OFFSET),
            )
            encoded.append((
                token_id, CATEGORY_CLOSED_CLASS, SPECIAL_FEATURE_ID,
                SPECIAL_FEATURE_ID, SPECIAL_FEATURE_ID, idx + 1,
                WORD_FINAL_YES if idx == last_idx else WORD_FINAL_NO,
            ))
        else:
            token_id = _SUFFIX_TO_ID.get(name, SUFFIX_OFFSET)
            cat_id = category_to_id.get(makes, 0)
            suffix_obj = _SUFFIX_BY_NAME.get(name)
            group_id = GROUP_TO_ID.get(getattr(suffix_obj, 'group', None), SPECIAL_FEATURE_ID)
            comes_to_id = TYPE_TO_ID.get(getattr(suffix_obj, 'comes_to', None), SPECIAL_FEATURE_ID)
            makes_id = TYPE_TO_ID.get(type_name_to_enum.get(makes), SPECIAL_FEATURE_ID)
            encoded.append((
                token_id, cat_id, group_id, comes_to_id, makes_id, idx + 1,
                WORD_FINAL_YES if idx == last_idx else WORD_FINAL_NO,
            ))
    return encoded


def encode_suffix_chain(suffix_chain: List) -> List[Tuple[int, int, int, int, int, int, int]]:
    """Encodes a suffix chain into (token_id, category_id) pairs for the ML model."""
    if not suffix_chain:
        return []
    encoded = []
    last_idx = len(suffix_chain) - 1
    for idx, s in enumerate(suffix_chain):
        if isinstance(s, ClosedClassMarker):
            token_id = _CC_SURFACE_TO_ID.get(
                (s.cc_word.category, getattr(s, 'surface_form', s.cc_word.word)),
                _CC_NAME_TO_DEFAULT_ID.get(s.name, _CC_OFFSET),
            )
            encoded.append((
                token_id, CATEGORY_CLOSED_CLASS, SPECIAL_FEATURE_ID,
                SPECIAL_FEATURE_ID, SPECIAL_FEATURE_ID, idx + 1,
                WORD_FINAL_YES if idx == last_idx else WORD_FINAL_NO,
            ))
        else:
            token_id = _SUFFIX_TO_ID.get(s.name, SUFFIX_OFFSET)
            cat_id   = 1 if s.makes.name == 'Verb' else 0
            encoded.append((
                token_id,
                cat_id,
                GROUP_TO_ID.get(getattr(s, 'group', None), SPECIAL_FEATURE_ID),
                TYPE_TO_ID.get(getattr(s, 'comes_to', None), SPECIAL_FEATURE_ID),
                TYPE_TO_ID.get(getattr(s, 'makes', None), SPECIAL_FEATURE_ID),
                idx + 1,
                WORD_FINAL_YES if idx == last_idx else WORD_FINAL_NO,
            ))
    return encoded


def reconstruct_morphology(word: str, decomposition: Tuple) -> Dict[str, Any]:
    """Reconstructs the step-by-step morphology string from a root and suffix chain."""
    root, pos, chain, final_pos = decomposition

    if chain and isinstance(chain[0], ClosedClassMarker):
        cc = chain[0].cc_word
        return {
            'root_str':      f"{root} ({cc.category})",
            'final_pos':     final_pos,
            'has_chain':     False,
            'formation_str': f"{root} [{cc.category}]",
        }

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
    
    if chain and chain[0].name == "pekistirme":
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
        
        suffix_forms.append(found_form if found_form else "(ø)")
        suffix_names.append(suffix_obj.name)
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


def format_detailed_decomp(decomp: Tuple) -> str:
    """Formats decomposition to include both suffix name and specific surface form."""
    root, pos, chain, final_pos = decomp
    
    # Fast path for closed-class words since they don't have standard suffixes
    if chain and isinstance(chain[0], ClosedClassMarker):
        cc = chain[0].cc_word
        return f"{root}_{cc.category}"
        
    if not chain:
        return root
        
    parts = [root]
    current = root
    accepted_chain = []
    for suffix in chain:
        # Fallback check in case a CC marker is mixed into a standard chain
        if isinstance(suffix, ClosedClassMarker):
            parts.append(suffix.name)
            continue
            
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


def analyze_word(word: str, *, include_closed_class: bool = True) -> Dict[str, Any]:
    """Decompose one sanitized word and bundle everything downstream needs."""
    decomps = sfx.decompose_with_cc(word) if include_closed_class else sfx.decompose(word)

    encoded_chains: List[List] = []
    vms: List[Dict[str, Any]] = []
    typing_strings: List[str] = []

    for decomp in decomps:
        root, _, chain, _ = decomp
        encoded_chains.append(encode_suffix_chain(chain))
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


def analyze_words(words: List[str], *, include_closed_class: bool = True) -> List[Dict[str, Any]]:
    return [analyze_word(w, include_closed_class=include_closed_class) for w in words]


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