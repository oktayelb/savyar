# util/ — Rule-Based Morphological Engine

This folder is the core of SAVYAR's generate-then-rank pipeline. It contains the rule-based decomposer that enumerates every legal root + suffix chain for a Turkish word, the suffix definitions that encode Turkish morphology, and the dictionary and phonological utilities.

```
util/
├── decomposer.py        # Main entry point: decompose() and find_suffix_chain()
├── suffix.py            # Suffix base class, SuffixGroup hierarchy, vowel harmony
├── word_methods.py      # Dictionary, harmony functions, root candidate generation
├── suffixes/            # All suffix definitions, organized by POS transition
│   ├── n2n/             # Noun → Noun (case, possessive, plural, derivational...)
│   ├── n2v/             # Noun → Verb (verbifiers: -le, -e, -se...)
│   ├── v2n/             # Verb → Noun (participles, gerunds, infinitives, tense...)
│   └── v2v/             # Verb → Verb (passive, causative, negation, compounds...)
└── words/               # Closed-class word definitions
    ├── words.py         # Base Word class
    └── closed_class.py  # Pronouns, conjunctions, postpositions, adverbs, etc.
```

---

## Suffixes

### suffix.py — The Suffix Class

Every Turkish suffix is an instance of `Suffix` (or a subclass). A suffix carries:

| Field | Purpose |
|-------|---------|
| `name` | Unique identifier, used as ML token key (e.g. `"accusative_i"`, `"plural_ler"`) |
| `suffix` | Base form string before harmony (e.g. `"ler"`, `"in"`, `"ecek"`) |
| `comes_to` | What POS this suffix attaches to: `Type.NOUN`, `Type.VERB`, or `Type.BOTH` |
| `makes` | What POS the word becomes after this suffix: `Type.NOUN` or `Type.VERB` |
| `has_major_harmony` | 2-way vowel harmony (e→a, i→ı, ü→u, ö→o for back vowels) |
| `has_minor_harmony` | 4-way vowel harmony (additionally ı→u for back-round, i→ü for front-round) |
| `needs_y_buffer` | Whether a `y` buffer consonant is inserted on vowel collision |
| `group` | `SuffixGroup` enum value — controls ordering in the waterfall hierarchy |
| `is_unique` | If True, this suffix can appear at most once per chain |

**Form generation** — `suffix.form(word, current_chain=None)` returns a list of possible surface forms for the suffix when attached to `word`. During decomposition, `current_chain` is the suffix chain already accepted before the candidate suffix, so suffix files can make context-sensitive form decisions from previous suffixes. The default logic:

1. Apply major harmony (2-way: front/back)
2. Apply minor harmony (4-way: rounded/unrounded)
3. Apply consonant hardening (g→k, d→t, c→ç after hard consonants)
4. Handle vowel collision (drop first vowel, or insert y-buffer)
5. Generate softened variants (k→ğ, ç→c for when the next suffix starts with a vowel)

Subclasses (e.g. `CaseSuffix`, `PossessiveSuffix`, `Copula`) override form generation for special buffer logic (n-buffer for case after possessive, s-buffer for 3sg possessive, etc.).

### SuffixGroup — The Waterfall Hierarchy

Turkish agglutination follows a strict ordering. The `SuffixGroup` enum encodes this as an integer hierarchy — a suffix can only follow one with an **equal or smaller** group number (with a few exceptions):

```
V2V_DERIVATIONAL  =  25   # Voice: passive, causative, reflexive, reciprocal
VERB_NEGATING     =  35   # -me, -eme (negation)
VERB_COMPOUND     =  40   # -ebil, -eyaz, -edur... (compound verbs)
N2V_DERIVATIONAL  =  50   # Noun→Verb: -le, -e, -se...
N2N_DERIVATIONAL  =  50   # Noun→Noun: -lık, -lı, -cı, -siz...
V2N_DERIVATIONAL  =  50   # Verb→Noun: -ecek, -miş, -en, -ir, -me, -iş...
PLURAL            =  60   # -ler
POSSESSIVE        = 150   # -im, -in, -i, -imiz, -iniz, -leri
CASE              = 200   # -e, -de, -den, -i, -in
MARKING_KI        = 225   # -ki
WITH_LE           = 230   # -le (comitative)
DERIVATIONAL_LOCKING = 240 # Gerunds: -ip, -erek, -ince... (block further inflection)
PREDICATIVE       = 250   # Copula/tense: -dir, -di, -miş, -se, -iyor
CONJUGATION       = 300   # Person: -im, -sin, -iz, -siniz, -ler
```

The waterfall rule: `next_suffix.group >= last_suffix.group` (with exceptions for derivational reset, -ki reset, and compound verb reset — see `is_valid_transition()` in decomposer.py).

### Suffix Subfolders

**`n2n/`** — Noun-to-Noun suffixes:
- `case_suffixes.py` — Genitive (-in), Accusative (-i), Dative (-e), Locative (-de), Ablative (-den)
- `possessive_suffix.py` — 1sg (-im), 2sg (-in), 3sg (-i/-si), 1pl (-imiz), 2pl (-iniz), 3pl (-leri)
- `plural_suffix.py` — Plural (-ler)
- `derivationals.py` — -cı, -lık, -lı, -siz, -ce, -cik, -sel, ordinal -inci, etc.
- `conjugation_suffixes.py` — Person agreement: -im, -sin, -iz, -siniz, -ler (attaches to nouns, which includes nounified verbs)
- `copula.py` — Noun predicative: -dir, -di, -se, -miş
- `marking_suffix.py` — -ki (relativizer)
- `intensifier.py` — (placeholder for pekistirme, handled in decomposer.py)

**`n2v/`** — Noun-to-Verb suffixes:
- `verbifiers.py` — -le (applicative), -e (verbifier), -se (absentative), -de (onomatopoeic), -ik

**`v2n/`** — Verb-to-Noun suffixes (this is where tenses live, since this grammar treats tense markers as nounifiers):
- `participles.py` — -en (aorist), -miş (evidential), -dik (factive), -ecek (future), -ir (habitual), -esi (desiderative)
- `infinitives.py` — -me, -mek (infinitive), -iş (verbal noun)
- `gerunds.py` — -erek, -ince, -ip, -e (adverbial), -dikçe, -eli (since), -meden
- `nounifiers.py` — -ik (perfectative), -gen/-gin, -ek, -gi, -im, -in, -it, -inti, -geç, -giç, -anak, -men, etc.
- `predicatives.py` — -iyor (continuous), -di (past tense), -se (conditional) — these produce nouns that then take conjugations

**`v2v/`** — Verb-to-Verb suffixes:
- `verb_derivationals.py` — Passive (-il), Causative (-it, -dir, -ir, -er), Reflexive (-in, -ik), Reciprocal (-iş), Randomative (-ele)
- `verb_negative.py` — -me (negation), -eme (inability)
- `verb_compounds.py` — -ebil (ability), -eyaz (approximative), -edur (continuative), -ekal (persistive), -egel (habitual), -iver (suddenative)

---

## Closed-Class Words (`words/`)

Closed-class words are function words from fixed categories: pronouns, conjunctions, postpositions, adverbs, determiners, interjections, particles. They are enumerated in `closed_class.py` (186 words across 7 categories).

Key structures:
- `ClosedClassWord` — base class with `category`, `can_take_suffixes` fields
- `CLOSED_CLASS_LOOKUP` — dict mapping surface form → list of `ClosedClassWord` objects (handles ambiguity: "o" → pronoun + determiner)
- `ClosedClassMarker` — wrapper used in suffix chains to represent a closed-class analysis

The decomposer integrates closed-class words via `decompose_with_cc()`, which appends closed-class analyses alongside regular suffix-chain decompositions.

---

## word_methods.py — Dictionary & Phonology

This module manages the word dictionary and provides all phonological utilities.

### Dictionary
- Words loaded from `data/words.txt` into `WORDS_SET` (set) and `WORDS_LIST` (list) at import time
- `can_be_noun(word)` — checks if word is in dictionary (also handles soft-ł encoding)
- `can_be_verb(word)` — checks if the infinitive form (word + mak/mek) is in dictionary
- `exists(word)` — can be noun OR verb

### Vowel Harmony
- `major_harmony(word)` → `MajorHarmony.BACK` or `.FRONT` (based on last vowel)
- `minor_harmony(word)` → `MinorHarmony.BACK_ROUND`, `.BACK_WIDE`, `.FRONT_ROUND`, or `.FRONT_WIDE`

### Root Candidate Generation
`get_root_candidates(surface_root)` handles Turkish root mutations that occur when suffixes attach. Given a surface root that isn't in the dictionary, it tries:

1. **Consonant devoicing reversal** — b→p, c→ç, d→t, ğ→k, g→k (e.g. surface "kitab" → dictionary "kitap")
2. **Vowel drop restoration** — inserts ı/i/u/ü before final consonant (e.g. "oğl" → "oğul")
3. **Terminal vowel restoration** — appends a/e (e.g. "ney" → "neye")
4. **Consonant gemination reversal** — if last two chars are identical, try single (e.g. "hiss" → "his", "hakk" → "hak", "redd" → "ret")

### Derived Word Detection
`is_derived_word(word)` returns True if a dictionary word is a derived form (e.g. "güzellik" = güzel + lik). The decomposer skips these as root candidates, forcing decomposition through the true root.

---

## decomposer.py — The Decomposition Engine

This is the heart of the rule-based layer. It takes a Turkish word and returns **every** valid root + suffix chain decomposition.

### Core API

```python
import util.decomposer as sfx

# Basic decomposition — returns list of (root, pos, chain, final_pos) tuples
analyses = sfx.decompose("evlerden")
# [('ev', 'noun', [plural_ler, ablative_den], 'noun'), ...]

# With closed-class analyses appended
analyses = sfx.decompose_with_cc("ile")
# [...regular analyses..., ('ile', 'cc_conjunction', [ClosedClassMarker(...)], 'cc_conjunction')]
```

Each result tuple:
- `root` — the dictionary lemma (e.g. `"ev"`)
- `pos` — starting POS of the root (`"noun"` or `"verb"`)
- `chain` — list of `Suffix` objects in attachment order
- `final_pos` — POS after the last suffix (`"noun"`, `"verb"`, or `"cc_*"`)

### How decompose() Works

```
decompose("geleceğimizi")
│
├─ 1. Try pekistirme (reduplication: "masmavi" → mas+mavi)
│
├─ 2. For each possible root length (1..N):
│   ├─ root = word[:i]
│   ├─ Skip if is_derived_word(root)
│   ├─ If can_be_noun(root): find_suffix_chain(word, "noun", root)
│   ├─ If can_be_verb(root): find_suffix_chain(word, "verb", root)
│   └─ If not exists(root):
│       └─ For each candidate in get_root_candidates(root):
│           ├─ Build virtual_word = lemma_root + remaining_text
│           └─ find_suffix_chain(virtual_word, pos, lemma_root)
│
└─ 3. Return all analyses (LRU-cached per word)
```

### find_suffix_chain() — The DFS Engine

Recursive depth-first search over the suffix transition graph:

```
find_suffix_chain(word, start_pos, root, current_chain, visited, shared_cache)
│
├─ Base case: no remaining text → return [([], start_pos)]
│
├─ Check shared_cache for (current_surface, remaining_text, pos, current_chain_signature)
│
├─ For each candidate suffix allowed by the current POS:
│   ├─ Validate hierarchy: is_valid_transition(last_suffix, next_suffix)
│   ├─ Check uniqueness: skip if is_unique and already in chain
│   ├─ Generate forms: suffix.form(current_root, current_chain=current_chain)
│   └─ For each form:
│       ├─ MATCH TYPE 1: rest.startswith(form) → recurse
│       └─ MATCH TYPE 2: Vowel narrowing before -iyor → recurse
│
└─ Cache results (if no unique suffixes in chain)
```

### SUFFIX_TRANSITIONS — The POS Transition Map

```python
SUFFIX_TRANSITIONS = {
    'noun': {
        'noun': NOUN2NOUN + [s for s in VERB2NOUN if s.comes_to == Type.BOTH],
        'verb': NOUN2VERB + [s for s in VERB2VERB if s.comes_to == Type.BOTH]
    },
    'verb': {
        'noun': VERB2NOUN + [s for s in NOUN2NOUN if s.comes_to == Type.BOTH],
        'verb': VERB2VERB + [s for s in NOUN2VERB if s.comes_to == Type.BOTH]
    }
}
```

At each DFS step, the current POS determines which suffixes are candidates. Suffixes with `comes_to=Type.BOTH` appear in both noun and verb contexts (e.g. conjugations attach to both).

### is_valid_transition() — Hierarchy State Machine

Beyond the basic waterfall rule (`next_group >= last_group`), there are three exceptions:

1. **-ki reset** — After MARKING_KI, the hierarchy resets back to ≤ MARKING_KI. This allows "evdekinin" (ev+de+ki+nin): after -ki, case/possessive/plural can attach again.

2. **Derivational reset** — After N2N_DERIVATIONAL, hierarchy resets to ≤ N2N_DERIVATIONAL. This allows stacking derivational suffixes: "güzellikçi" (güzel+lik+çi).

3. **Compound reset** — After VERB_COMPOUND, hierarchy resets to ≤ VERB_COMPOUND. This allows negation after ability: "gidemeyecek" (git+eme+yecek) and ability after negation paths.

Self-looping is allowed only for: N2N_DERIVATIONAL, V2V_DERIVATIONAL, PREDICATIVE.

### Caching Strategy

Three layers of caching prevent redundant computation:

1. **`visited` set** (per call tree) — prevents revisiting the same `(root_len, pos, chain_signature)` state within one DFS traversal.

2. **`shared_cache` dict** (per decompose() call) — keyed on `(remaining_text, pos, last_group)`. When two different roots lead to the same remaining text with the same POS and hierarchy state, results are shared. This is the main optimization. Only caches when the chain contains no `is_unique` suffixes (uniqueness is chain-dependent).

3. **`@lru_cache`** (across calls) — `decompose()` itself is LRU-cached by word string. The cache is cleared when the dictionary changes (word deletion).

### Vowel Narrowing (Match Type 2)

Turkish drops the final vowel of some suffixes before -iyor:
- "bekle+me+iyor" → "beklemiyor" (the 'e' of -me is dropped before -iyor)

The decomposer handles this: if a suffix form ends in 'a' or 'e', it tries a shortened form (dropping that vowel) and checks if -iyor follows immediately.

---

## Usage Guide

### Basic Decomposition

```python
import util.decomposer as sfx

# Decompose a word
for root, pos, chain, final_pos in sfx.decompose("geleceğimizi"):
    suffix_names = [s.name for s in chain]
    print(f"  {root} ({pos}) + {suffix_names} → {final_pos}")
```

Output:
```
  gel (verb) + [nounifier_ecek, possessive_1sg, accusative_i] → noun
  gel (verb) + [nounifier_ecek, possessive_1pl, accusative_i] → noun
  ...
```

### With Closed-Class Words

```python
# Includes closed-class analyses (pronouns, conjunctions, etc.)
for root, pos, chain, final_pos in sfx.decompose_with_cc("ile"):
    if pos.startswith("cc_"):
        print(f"  {root} → {pos}")
    else:
        print(f"  {root} ({pos}) + {[s.name for s in chain]}")
```

### Inspecting a Suffix Chain

```python
word = "evlerinden"
for root, pos, chain, final_pos in sfx.decompose(word):
    current = root
    accepted_chain = []
    for suffix in chain:
        forms = suffix.form(current, current_chain=accepted_chain)
        # Find which form was actually used
        rest = word[len(current):]
        used = next((f for f in forms if rest.startswith(f)), forms[0])
        print(f"  {current} + {suffix.name}({used}) → ", end="")
        current += used
        accepted_chain.append(suffix)
    print(current)
```

### Cache Management

```python
# decompose() is LRU-cached (100k entries)
sfx.decompose("ev")       # computed
sfx.decompose("ev")       # cache hit

# Clear after dictionary changes
sfx.decompose.cache_clear()

```
