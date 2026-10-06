# Decomposer: Rule-Based Morphological Analysis

## Overview

The decomposer (`util/decomposer.py`) is the rule-based engine that generates **all valid morphological decompositions** of a Turkish word. Given an input like `"evlerinden"`, it produces every possible root + suffix chain interpretation — e.g. `ev + ler + in + den`, `evler + in + den`, etc. These candidates are then ranked by the ML model.

## Core Algorithm

### 1. Root Enumeration

`decompose(word)` iterates over every prefix of the input word as a potential root:

```
word = "geldiler"
roots tried: g, ge, gel, geld, geldi, geldir, geldile, geldiler
```

For each prefix, it checks:
- **`can_be_noun(root)`** — Is this root in the dictionary as a noun?
- **`can_be_verb(root)`** — Is this root in the dictionary as a verb stem?
- **`is_derived_word(root)`** — If the root is itself a derived form (e.g. `alıcı = al + ıcı`), it is **skipped** as a root candidate. This forces deeper decomposition through the true root.

If a prefix is not a dictionary word, **phonological restoration** is attempted via `get_root_candidates()`. Turkish has consonant mutations at morpheme boundaries (e.g. `kitab-` → `kitap` in isolation). This step generates lemma candidates for mutated surface forms and tries those as roots.

The function is wrapped with `@functools.lru_cache` to avoid recomputing decompositions for the same word within a session.

### 2. Suffix Chain Search — `find_suffix_chain()`

This is the heart of the decomposer: a **recursive depth-first search** that builds suffix chains from a root outward.

**Parameters:**
- `word` — the full surface form
- `start_pos` — current part of speech (`"noun"` or `"verb"`)
- `root` — the stem consumed so far (grows as suffixes are matched)
- `current_chain` — the suffix objects accumulated so far
- `shared_cache` — cross-root memoization dictionary

**At each step:**
1. Compute `rest = word[len(root):]` — the unconsumed remainder.
2. If `rest` is empty → base case: return the current chain as a valid decomposition.
3. Look up applicable suffixes from `SUFFIX_TRANSITIONS[current_pos]`, which maps `(current_pos) → {target_pos: [suffix_list]}`.
4. For each candidate suffix:
   - Check **hierarchy validity** via `is_valid_transition()`.
   - Check **uniqueness constraints** (some suffixes can only appear once per chain).
   - Generate **surface forms** via `suffix.form(root)` — this applies vowel harmony, consonant hardening, and buffer letter rules.
   - If a form matches the start of `rest`, recurse with the extended root and chain.

### 3. Suffix Transition Table

The `SUFFIX_TRANSITIONS` dict encodes which suffix lists apply based on the current POS:

```
SUFFIX_TRANSITIONS = {
    'noun': {
        'noun': NOUN2NOUN + [V2N suffixes with comes_to=BOTH],
        'verb': NOUN2VERB + [V2V suffixes with comes_to=BOTH],
    },
    'verb': {
        'noun': VERB2NOUN + [N2N suffixes with comes_to=BOTH],
        'verb': VERB2VERB + [N2V suffixes with comes_to=BOTH],
    }
}
```

Each suffix carries a `comes_to` (what POS it attaches to) and `makes` (what POS the resulting stem becomes) property, enabling POS tracking through the chain.

### 4. Suffix Hierarchy — The Waterfall Model

`is_valid_transition(last_suffix, next_suffix)` enforces ordering constraints via `SuffixGroup` enum values. The core rule is a **waterfall**: a suffix with a higher group number cannot be followed by one with a lower group number.

```
next_group < last_group → INVALID (waterfall violated)
```

**Exceptions to the waterfall:**
- **`-ki` marker**: After the `-ki` suffix, the chain resets — case suffixes can follow again (e.g. `evdekilerden`).
- **N2N derivational suffixes**: Can loop (derivation → derivation), e.g. `göz + lük + çü`.
- **Verb compound suffixes**: Can loop for constructions like `gidebilmeyen`.
- **Self-looping**: Only allowed for derivational and predicative groups; blocked for inflectional groups.

### 5. Vowel Narrowing (iyor contraction)

A special matching rule handles the `-iyor` progressive suffix. When a suffix ending in `-a/-e` is followed by `-iyor`, the final vowel drops:

```
gel + e + iyor → geliyor  (not *geleiyor)
```

The code detects this by checking if a suffix form minus its final vowel matches the remainder, and if `iyor` follows.

### 6. Pekistirme (Intensifier Reduplication)

`get_pekistirme_analyses()` handles Turkish intensifier reduplication patterns:

```
masmavi = mas + mavi  (intensified "blue")
bembeyaz = bem + beyaz
güpegündüz = güpe + gündüz
```

The algorithm:
1. Finds the first vowel in the word.
2. Tries splitting at `vowel_index + 2` (standard: `mas-mavi`) and `vowel_index + 3` (extended with connecting vowel: `güpe-gündüz`).
3. Verifies the remainder starts with the same consonant-vowel prefix as the word.
4. If suffixes follow the reduplicated root (e.g. `masmaviydim`), it recursively calls `find_suffix_chain()` on the remainder.

### 7. Closed-Class Words

Closed-class lexemes (`util/words/closed_class.py`) are ordinary noun roots, so `bana` is `("ben", "noun", [dative_e], "noun")`. Their irregular forms come from the suffix rules (pronominal `n` after `o`/`bu`/`şu`, genitive `-im` after `ben`/`biz`) and from `IRREGULAR_STEMS` (e.g. `bana`, `hepsi`, `birisi`). Conjunctions and clitics take no suffix as nouns. `decompose()` returns each `(root, pos, chain)` once.

`decompose_with_fallback()` is what callers use: it returns `decompose()`'s analyses, or hypothesised roots when there are none.

### 8. Caching Strategy

Two levels of caching prevent redundant computation:

- **`@lru_cache` on `decompose()`**: Full-word level. If the same word is queried twice (common in batch processing), the result is returned instantly.
- **`shared_cache` within `find_suffix_chain()`**: Keyed on `(remaining_text, current_pos, last_suffix_group)`. Since hierarchy validation only depends on the last suffix's group, different root paths that reach the same remainder with the same POS and group produce identical subtrees — these are computed once and reused.

Uniqueness constraints (`is_unique` suffixes) bypass the shared cache because their validity depends on the full chain history, not just the last group.

## Output Format

Each decomposition is a 4-tuple:

```python
(root: str, start_pos: str, suffix_chain: List[Suffix], final_pos: str)
```

- `root` — the dictionary lemma
- `start_pos` — POS of the root (`"noun"` or `"verb"`)
- `suffix_chain` — ordered list of `Suffix` objects
- `final_pos` — POS after applying all suffixes
