"""Per-word analysis: decomposition + encoding + view model.

Stage 2 of the flow:
    sanitized words -> analyzer -> one analysis dict per word -> ...

Each analysis dict is index-aligned across its list fields (decomps,
encoded_chains, vms, typing_strings). Downstream code picks by index and
trusts alignment.

Both the single-word path and the sentence-word loop funnel through
analyze_word, so decomposition, encoding, view-model reconstruction, and
typing-string assembly live in exactly one place.
"""
from typing import Any, Dict, List, Optional

import util.decomposer as sfx
import app.morphology_adapter as morph


def analyze_word(word: str, *, include_closed_class: bool = True) -> Dict[str, Any]:
    """Decompose one sanitized word and bundle everything downstream needs.

    Returns a dict with index-aligned lists:
      word: the sanitized word as handed in
      decomps: (root, pos, chain, final_pos) tuples from the decomposer
      encoded_chains: per-decomp token/category id pairs for the ML model
      vms: display-ready view models from morphology_adapter
      typing_strings: per-decomp flat strings for sentence-level matching
    """
    decomps = sfx.decompose_with_cc(word) if include_closed_class else sfx.decompose(word)

    encoded_chains: List[List] = []
    vms: List[Dict[str, Any]] = []
    typing_strings: List[str] = []

    for decomp in decomps:
        root, _, chain, _ = decomp
        encoded_chains.append(morph.encode_suffix_chain(chain))
        vm = morph.reconstruct_morphology(word, decomp)
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
    """Rank an analysis's candidates by ML score (highest first).

    Sets vm['score'] on each view model, then reorders every aligned list
    (decomps/encoded_chains/vms/typing_strings) in place so index i always
    refers to the same candidate across fields.

    Returns the sorted score list, or None when scoring was skipped (single
    candidate) or the predictor raised.
    """
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
