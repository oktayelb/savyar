"""
Google Turkish Treebank (UD) → Savyar Adapter
==============================================
Translates the Google Turkish Universal Dependencies treebank (web.conllu,
wiki.conllu) into the sentence_valid_decompositions.jsonl format consumed by
Savyar's training pipeline.

Key differences from the METU-Sabancı treebank:
  - Features use Key=Value syntax (e.g. Case=Loc, Derivation=Make).
  - Words are split across multiple tokens linked by the `ig` (inflection
    group) deprel — every morpheme layer has its own row with its own
    features. The ROOT row has a lemma; intermediate rows have lemma="_".
  - Surface forms are explicit per morpheme segment. The full word surface
    is the concatenation of all tokens in an ig-chain (they always carry
    SpaceAfter=No).
  - Nominal predicates (xpos=NOMP) carry BOTH noun features (Case, Possessive,
    A-PersonNumber) AND verb features (Copula, V-PersonNumber) on one row.

Strategy: same as METU — DECOMPOSER-VALIDATED MATCHING
  1. Parse the .conllu files into sentences.
  2. Merge each ig-chain (or single-row token) into a "word" with:
        - surface = concatenation of chain rows
        - lemma = root-row lemma
        - feature_layers = list of (upos, xpos, features_dict) per layer
  3. Map feature_layers → ordered list of Savyar suffix names.
  4. Run decompose(surface) and pick the candidate whose chain matches
     the expected suffixes (normalizing known ambiguities).
  5. Emit a JSONL word-entry per word, plus a per-sentence entry.

Files produced (alongside this adapter):
  - treebank_adapted.jsonl           matched + treebank-forced entries
  - treebank_adapted_unmatched.jsonl diagnostic log for mismatches
  - treebank_adaptation_stats.json   run statistics
  - unmapped_features.json           every feature value we did NOT map,
                                     with frequency + examples — the user
                                     fills these in over time to grow the
                                     DERIVATION_MAP / TAM_MAP / etc. tables

Whenever a feature isn't confidently mappable it is recorded in
unmapped_features.json instead of silently being given a wrong mapping.
"""

import json
import sys
import os
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from util.decomposer import decompose, ALL_SUFFIXES, enable_index

enable_index()
from util.suffix import Type
from util.words.closed_class import CLOSED_CLASS_LOOKUP
from util.word_methods import tr_lower
import util.word_methods as wrd
from data.treebank_vnoun import (
    AMBIGUOUS_VNOUN,
    has_unexpected_nounifier_is,
    resolve_ambiguous_vnoun_suffixes,
)

SUFFIX_BY_NAME = {s.name: s for s in ALL_SUFFIXES}

# Quote-like characters to strip from surface/lemma before processing.
_QUOTE_CHARS = "\"'`‘’“”„«»‹›"

def _strip_quotes(s):
    if not s:
        return s
    for q in _QUOTE_CHARS:
        s = s.replace(q, "")
    return s


# =============================================================================
# FEATURE MAPPING TABLES
# =============================================================================
# Keep the mapping tables grouped by feature key (the LHS of the Key=Value
# pair in .conllu features). Values that we map to None or leave out mean
# "zero morpheme" (no Savyar suffix emitted for this feature). Values that
# are deliberately UNMAPPED live in UNMAPPED_* below and are routed to the
# unmapped_features.json report instead of being silently coerced.

# ── Derivation = single-suffix mappings ──
# Each entry maps a UD Derivation value to the Savyar suffix name that best
# realises the same morpheme. Confirmed by surface-form sampling from the
# treebank and comparison to the METU adapter's feature table.
DERIVATION_MAP = {
    # verb-to-verb (voice / ability / compound)
    "Make":      "aplicative_le",        # -le/-la (noun → verb)
    "Cau":       "active_dir",           # -dir/-t (causative)
    "Pass":      "passive_il",           # -il/-in/-n
    "Rcp":       "reflexive_is",         # -iş (reciprocal)
    "Rfx":       "reflexive_in",         # -in/-n (reflexive)
    "Able":      "possibilitative_ebil", # -ebil/-abil
    "Haste":     "suddenative_ivermek",  # -iver
    "Ever":      "persistive_egelmek",   # -egel/-agel
    # participles / nominalizations
    "PresPart":  "factative_en",         # -en/-an (present participle)
    "PastPart":  "adjectifier_dik",      # -dik/-dığ
    "FutPart":   "nounifier_ecek",       # -ecek/-acak
    "PerPart":   "pastfactative_miş",    # -miş
    "AorPart":   "factative_ir",         # -ir/-er (aorist participle)
    "PresNom":   "factative_en",
    "PastNom":   "adjectifier_dik",
    "FutNom":    "nounifier_ecek",
    "PerNom":    "pastfactative_miş",
    "AorNom":    "factative_ir",
    "Inf":       "infinitive_mek",       # -mek/-mak
    "Nonf":      AMBIGUOUS_VNOUN,        # surface decides between -me/-ma and -iş/-ış/-uş/-üş
    # adverbial / gerund
    "Ger":       "adverbial_erek",       # -erek/-arak
    "After":     "adverbial_ip",         # -ip/-ıp
    "While":     "when_ken",             # -ken
    "When":      "adverbial_ince",       # -ince/-ınca
    "As":        "adverbial_dikçe",      # -dikçe/-dıkça (as-long-as)
    "Since":     "since_eli",            # -eli
    # N2N derivational
    "With":      "composessive_li",      # -li/-lı/-lu/-lü
    "Wout":      "privative_siz",        # -siz/-sız
    "Ness":      "suitative_lik",        # -lik/-lık
    "Rel":       "marking_ki",           # -ki
    "Agt":       "actor_ci",             # -ci/-cı
    "Like":      "adverbial_cesine",     # -cesine
    "Ly":        "relative_ce",          # -ce/-ca (manner)
    "Lang":      "relative_ce",          # -ce (language)
    "Act":       "relative_ce",          # -ce (manner, güzelce)
    "Rtd":       "relative_sel",         # -sel
    "Dim":       "dimunitive_cik",       # -cik/-cık
    "Fam":       "familative_gil",       # -giller
    "Sim":       "approximative_si",     # -si
    "Aff":       "philicative_cil",      # -cil/-cül
    "Doct":      "ideologicative_izm",   # -izm
    # ── User-directed routings (semantic differences intentionally ignored) ──
    # Inh (-ıcı habitual doer) → if_se per directive.
    "Inh":       "actor_ci",
    # From (-li from-origin) → composessive_li (shares surface -li/-lı).
    "From":      "composessive_li",
    # Everything else previously unmapped routes to suitative_lik (-lık):
    # For (-lık for), Foll (-ist), By (-ce by-means), Of (-lerce/-larca),
    # Snd (sound-related), Coll (-ce collective), Inter (inter/between),
    # ProNom (-esiye rare nominalization).
    "For":       "suitative_lik",
    "Foll":      "suitative_lik",
    "By":        "suitative_lik",
    "Of":        "suitative_lik",
    "Snd":       "suitative_lik",
    "Coll":      "suitative_lik",
    "Inter":     "suitative_lik",
    "ProNom":    "suitative_lik",
    # `Derivation=True` is a treebank tagging artifact — it appears on
    # apostrophe-separated proper-noun case suffix rows like Afyon'da,
    # Sistem'i, etc. There's no derivation there, just a case marker on a
    # proper noun, so we map it to "no suffix" to keep those words mappable.
    "True":      None,
}

# ── Derivation = multi-suffix expansions ──
# Some UD Derivation values correspond to a FUSED pair of Savyar suffixes.
DERIVATION_MULTI = {
    "Bcm": ["aplicative_le", "reflexive_is"],  # -leş (become)
    "Acq": ["aplicative_le", "reflexive_in"],  # -lan (acquire)
}

# ── Derivation values the user must resolve manually ──
# These are RECORDED (with surface examples) in unmapped_features.json so the
# user can promote them into DERIVATION_MAP over time. Each note captures our
# current hypothesis so you don't have to rediscover it.
UNMAPPED_DERIVATIONS = {
    # (No currently-unmapped derivations — all have been routed into
    # DERIVATION_MAP per user directive. Future novel values discovered
    # during parsing will still land here via the catch-all path.)
}

# ── TenseAspectMood ──
TAM_MAP = {
    "Past":   "pasttense_di",            # -di/-dı
    "Aor":    "factative_ir",            # -ir/-er/-r
    "Fut":    "nounifier_ecek",          # -ecek/-acak
    "Nar":    "pastfactative_miş",       # -miş
    "Prog1":  "continuous_iyor",         # -iyor
    "Cond":   "if_se",                   # -se/-sa (conditional)
    "Opt":    "adverbial_e",             # -e/-a (optative)
    "Desr":   "wish_suffix",             # -se/-sa (desiderative)
    # "Imp":   handled specially — zero in A2sg, else keep person marker
    # "Nec":   handled via TAM_MULTI
    # "Prog2": handled via TAM_MULTI
}

# TAM values that expand into a pair of suffixes.
TAM_MULTI = {
    "Nec":   ["infinitive_me", "composessive_li"],  # -meli/-malı
    "Prog2": ["infinitive_mek", "locative_de"],     # -mekte
}

# ── Copula ──
# Note PresCop is the zero present copula (skipped entirely).
COPULA_MAP = {
    "PresCop":  None,
    "PastCop":  "pasttense_di",          # -ydi/-ydı
    "NarCop":   "copula_mis",            # -ymiş
    "EvCop":    "copula_mis",            # evidential copula (surface-identical to NarCop)
    "CndCop":   "if_se",                 # -yse/-ysa
    "GenCop":   "nounaorist_dir",        # -dir/-dır
}

# ── Case ──
CASE_MAP = {
    "Bare":  None,
    "Nom":   None,
    "Loc":   "locative_de",
    "Dat":   "dative_e",
    "Acc":   "accusative_i",
    "Gen":   "noun_compound",            # genitive = Savyar's noun_compound
    "Abl":   "ablative_den",
    "Ins":   "confactuous_le",           # instrumental -le/-la
}

# ── Possessive ──
POSS_MAP = {
    "Pnon":  None,
    "P1sg":  "posessive_1sg",
    "P2sg":  "posessive_2sg",
    "P3sg":  "posessive_3sg",
    "P1pl":  "posessive_1pl",
    "P2pl":  "posessive_2pl",
    "P3pl":  "posessive_3pl",
}

# ── PersonNumber (noun side — A-values) ──
# A3sg is the zero default; A3pl marks plural on a noun (plural_ler).
# A1sg/A2sg/A1pl/A2pl on nouns are rare and usually appear on pronouns —
# pronouns are handled via the closed-class path, so we skip them here.
A_PERSON_MAP = {
    "A3sg":  None,
    "A3pl":  "plural_ler",
    "A1sg":  None,
    "A2sg":  None,
    "A1pl":  None,
    "A2pl":  None,
}

# ── PersonNumber (verb side — V-values) ──
V_PERSON_MAP = {
    "V3sg":  None,                       # zero 3rd person singular
    "V1sg":  "conjugation_1sg",
    "V2sg":  "conjugation_2sg",
    "V3pl":  "conjugation_3pl",
    "V1pl":  "conjugation_1pl",
    "V2pl":  "conjugation_2pl",
}

# ── Feature keys whose values never carry morphology (always skipped). ──
SKIP_FEATURE_KEYS = {
    "Proper",           # capitalisation flag
    "Apostrophe",       # apostrophe flag (we strip apostrophes on surface)
    "Temporal",         # lexical class marker (not a suffix)
    "ConjunctionType",  # CC sub-typing (handled via closed-class path)
    "DeterminerType",   # DET sub-typing (handled via closed-class path)
    "ComplementType",   # ADP sub-typing (handled via closed-class path)
    "NumberType",       # handled specially for NumberType=Ord below
    "Polarity",         # handled specially (Neg logic)
    "Contrast",         # handled specially below (emits if_se)
}

# ── UPOS → Savyar closed-class category ──
UPOS_TO_CC_CATEGORY = {
    "CONJ":   "conjunction",
    "ADP":    "postposition",
    "ADV":    "adverb",
    "DET":    "determiner",
    "INTJ":   "interjection",
    "PRT":    "particle",
    "PRON":   "pronoun",
}

# UPOS categories we treat as bare-root (no suffix learning).
SKIP_UPOS = {"NUM", "PUNCT", "X", "ONOM", "AFFIX", "SYM"}

# Able+Neg fusion: when both appear on the SAME feature layer we emit the
# fused negative_able (which replaces possibilitative_ebil + negative_me).
# When they appear on SEPARATE layers they stay as the two individual suffixes.

# Known suffix ambiguities between the treebank's analysis and Savyar's
# decomposer. Same shape as the METU adapter uses.
SUFFIX_ALTERNATIVES = {
    "active_dir":        ["active_it", "active_ir", "active_er"],
    "passive_il":        ["reflexive_in"],
    "reflexive_in":      ["passive_il"],
    "adverbial_erek":    ["adverbial_ip"],
    "adverbial_ip":      ["adverbial_erek"],
    "copula_mis":        ["pastfactative_miş"],
    "pastfactative_miş": ["copula_mis"],
    "composessive_li":   ["relative_sel"],
    "relative_sel":      ["composessive_li"],
    "actor_ci":          ["factative_ir"],
}

# Suffix-chain equivalences (from METU).
EQUIVALENT_SEQUENCES = [
    (["aplicative_le", "factative_ir"], ["plural_ler"]),
]


# =============================================================================
# CONLL-U PARSER
# =============================================================================

def parse_conllu(filepath):
    """Parse a .conllu file → list of sentences (each = list of token dicts)."""
    sentences = []
    current = []
    with open(filepath, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            # Skip multi-word tokens and empty-node rows (ids containing "-" or ".")
            if "-" in parts[0] or "." in parts[0]:
                continue

            feats = {}
            feats_multi = []  # preserve duplicate keys (NOMP has PersonNumber twice)
            if parts[5] and parts[5] != "_":
                for item in parts[5].split("|"):
                    if "=" not in item:
                        continue
                    k, v = item.split("=", 1)
                    feats_multi.append((k, v))
                    # Last value wins for dict access, but feats_multi preserves all.
                    feats[k] = v

            token = {
                "id":          parts[0],
                "surface":     parts[1],
                "lemma":       parts[2],
                "upos":        parts[3],
                "xpos":        parts[4],
                "features":    feats,
                "features_multi": feats_multi,
                "head":        parts[6],
                "deprel":      parts[7],
                "misc":        parts[9] if len(parts) >= 10 else "_",
            }
            current.append(token)
    if current:
        sentences.append(current)
    return sentences


def merge_ig_chains(sentence_tokens):
    """Merge `ig`-linked tokens into single word entries.

    Returns a list of "words". Each word carries:
        surface         — concatenation of ig-chain token surfaces
        lemma           — lemma of the first chain token
        feature_layers  — list of per-layer dicts: {upos, xpos, features, features_multi, surface}
        is_chain        — True if the word came from a multi-token chain
    """
    merged = []
    i = 0
    n = len(sentence_tokens)
    while i < n:
        tok = sentence_tokens[i]

        # Skip punctuation entirely
        if tok["upos"] == "PUNCT":
            i += 1
            continue

        # Greedily consume tokens while the CURRENT last-in-chain has deprel=="ig".
        chain = [tok]
        while chain[-1]["deprel"] == "ig" and (i + 1) < n:
            i += 1
            chain.append(sentence_tokens[i])

        # Build merged surface
        surface = "".join(t["surface"] for t in chain if t["surface"] != "_")

        # Lemma: first non-"_" lemma in the chain (always chain[0] in practice).
        lemma = None
        for t in chain:
            if t["lemma"] and t["lemma"] != "_":
                lemma = t["lemma"]
                break
        if lemma is None:
            # Fallback: use the first token's surface as a pseudo-lemma.
            lemma = chain[0]["surface"]

        feature_layers = [
            {
                "upos":            t["upos"],
                "xpos":            t["xpos"],
                "features":        t["features"],
                "features_multi":  t["features_multi"],
                "surface":         t["surface"],
            }
            for t in chain
        ]

        merged.append({
            "surface":        surface,
            "lemma":          lemma,
            "feature_layers": feature_layers,
            "is_chain":       len(chain) > 1,
            "head_upos":      chain[-1]["upos"],
            "head_xpos":      chain[-1]["xpos"],
        })
        i += 1
    return merged


# =============================================================================
# FEATURE → SUFFIX MAPPING
# =============================================================================

def _layer_is_verb_context(layer):
    return layer["upos"] == "VERB" and layer["xpos"] != "NOMP"


def _layer_is_nomp(layer):
    return layer["xpos"] == "NOMP"


def features_to_suffix_names(word, unmapped_sink):
    """Convert a merged word's feature_layers into the ordered list of Savyar
    suffix names expected for the surface form.

    Mutates `unmapped_sink` (a dict) when a feature value cannot be mapped.
    Returns (suffix_names, unmapped_feats_on_this_word, has_unmappable).
    """
    suffix_names = []
    unmapped_on_word = []
    has_unmappable = False

    # We intentionally process the feature keys in a canonical Turkish
    # morpheme order regardless of the order they appeared in the .conllu
    # line: Derivation → Polarity → TAM → A-plural → Possessive → Case
    #        → Copula → V-person.
    #
    # NB: In practice a single layer carries at most one of each key (except
    # NOMP which has PersonNumber twice — once A*, once V*).

    for layer in word["feature_layers"]:
        feats = layer["features"]
        feats_multi = layer["features_multi"]
        xpos = layer["xpos"]
        upos = layer["upos"]

        # Collect every PersonNumber value (can appear twice on NOMP).
        pn_values = [v for k, v in feats_multi if k == "PersonNumber"]
        a_person = next((v for v in pn_values if v.startswith("A")), None)
        v_person = next((v for v in pn_values if v.startswith("V")), None)

        is_nomp = _layer_is_nomp(layer)
        is_verb = _layer_is_verb_context(layer)
        is_imp = feats.get("TenseAspectMood") == "Imp"

        # 1) Derivation
        if "Derivation" in feats:
            dval = feats["Derivation"]
            if dval in DERIVATION_MULTI:
                suffix_names.extend(DERIVATION_MULTI[dval])
            elif dval in DERIVATION_MAP:
                mapped = DERIVATION_MAP[dval]
                # Able+Neg on same layer → negative_able (fused)
                if dval == "Able" and feats.get("Polarity") == "Neg":
                    suffix_names.append("negative_able")
                elif mapped is not None:
                    suffix_names.append(mapped)
            elif dval in UNMAPPED_DERIVATIONS:
                has_unmappable = True
                unmapped_on_word.append(f"Derivation={dval}")
                _record_unmapped(unmapped_sink, "Derivation", dval, word)
            else:
                has_unmappable = True
                unmapped_on_word.append(f"Derivation={dval}")
                _record_unmapped(unmapped_sink, "Derivation", dval, word)

        # 2) Polarity=Neg (only if NOT already absorbed by Able on this layer)
        if feats.get("Polarity") == "Neg":
            if not (feats.get("Derivation") == "Able"):
                suffix_names.append("negative_me")

        # 3) TenseAspectMood
        tam = feats.get("TenseAspectMood")
        if tam:
            if tam == "Imp":
                # Imperative: A2sg / V2sg is a zero; handled by the person
                # marker below (which maps to None for singular 2nd person
                # outside A_PERSON_MAP coverage — see handling below).
                pass
            elif tam in TAM_MULTI:
                suffix_names.extend(TAM_MULTI[tam])
            elif tam in TAM_MAP:
                suffix_names.append(TAM_MAP[tam])
            else:
                has_unmappable = True
                unmapped_on_word.append(f"TenseAspectMood={tam}")
                _record_unmapped(unmapped_sink, "TenseAspectMood", tam, word)

        # 4) A-PersonNumber (noun number / plural)
        if a_person:
            if is_verb or is_nomp:
                # On a verb layer, A3pl behaves as conjugation_3pl.
                # (Other A-values don't co-occur with a verb head in practice.)
                if a_person == "A3pl" and not v_person:
                    suffix_names.append("conjugation_3pl")
                elif a_person in A_PERSON_MAP and A_PERSON_MAP[a_person] is not None:
                    # Shouldn't generally happen; fall through harmlessly.
                    suffix_names.append(A_PERSON_MAP[a_person])
                elif A_PERSON_MAP.get(a_person) is None:
                    pass  # zero
                else:
                    has_unmappable = True
                    unmapped_on_word.append(f"PersonNumber={a_person}")
                    _record_unmapped(unmapped_sink, "PersonNumber", a_person, word)
            else:
                # Noun/adj/adv layer
                mapped = A_PERSON_MAP.get(a_person, "__MISSING__")
                if mapped is None:
                    pass  # zero
                elif mapped == "__MISSING__":
                    has_unmappable = True
                    unmapped_on_word.append(f"PersonNumber={a_person}")
                    _record_unmapped(unmapped_sink, "PersonNumber", a_person, word)
                else:
                    suffix_names.append(mapped)

        # 5) Possessive
        poss = feats.get("Possessive")
        if poss:
            mapped = POSS_MAP.get(poss, "__MISSING__")
            if mapped is None:
                pass
            elif mapped == "__MISSING__":
                has_unmappable = True
                unmapped_on_word.append(f"Possessive={poss}")
                _record_unmapped(unmapped_sink, "Possessive", poss, word)
            else:
                suffix_names.append(mapped)

        # 6) Case
        case = feats.get("Case")
        if case:
            mapped = CASE_MAP.get(case, "__MISSING__")
            if mapped is None:
                pass
            elif mapped == "__MISSING__":
                has_unmappable = True
                unmapped_on_word.append(f"Case={case}")
                _record_unmapped(unmapped_sink, "Case", case, word)
            else:
                suffix_names.append(mapped)

        # 7) Copula (appears on NOMP + on standalone verbs as the carrier
        # of person/number). PresCop is zero and skipped.
        cop = feats.get("Copula")
        if cop:
            mapped = COPULA_MAP.get(cop, "__MISSING__")
            if mapped is None:
                pass
            elif mapped == "__MISSING__":
                has_unmappable = True
                unmapped_on_word.append(f"Copula={cop}")
                _record_unmapped(unmapped_sink, "Copula", cop, word)
            else:
                suffix_names.append(mapped)

        # 8) V-PersonNumber (verb conjugation)
        if v_person:
            # Imperative 2sg/2pl: 2sg is zero (bare root), 2pl usually -in/-yın.
            # We skip 2sg conj on imperatives entirely.
            if is_imp and v_person == "V2sg":
                pass
            else:
                mapped = V_PERSON_MAP.get(v_person, "__MISSING__")
                if mapped is None:
                    pass
                elif mapped == "__MISSING__":
                    has_unmappable = True
                    unmapped_on_word.append(f"PersonNumber={v_person}")
                    _record_unmapped(unmapped_sink, "PersonNumber", v_person, word)
                else:
                    suffix_names.append(mapped)

        # 9) NumberType=Ord → ordinal_inci (NUM tokens with written '.')
        if feats.get("NumberType") == "Ord":
            suffix_names.append("ordinal_inci")

        # 10) Contrast=True → the -se/-sa contrastive copula suffix (if_se).
        # Examples: Bazense=bazen+se, tanığıysa=tanık+ı+y+sa,
        # girmektense=gir+mek+ten+se, bilgilerse=bilgi+ler+se.
        if feats.get("Contrast") == "True":
            suffix_names.append("if_se")

        # 11) Catch any feature keys we haven't explicitly handled.
        for k, v in feats_multi:
            if k in SKIP_FEATURE_KEYS:
                continue
            if k in {
                "Derivation", "TenseAspectMood", "Case", "Possessive",
                "PersonNumber", "Copula", "NumberType",
            }:
                continue
            # Anything else is genuinely unrecognised.
            has_unmappable = True
            unmapped_on_word.append(f"{k}={v}")
            _record_unmapped(unmapped_sink, k, v, word)

    suffix_names = resolve_ambiguous_vnoun_suffixes(
        word["surface"],
        word["lemma"],
        suffix_names,
        SUFFIX_BY_NAME,
    )
    return suffix_names, unmapped_on_word, has_unmappable


def _record_unmapped(sink, feat_key, feat_val, word):
    """Record an unmapped feature occurrence with a surface example."""
    slot = sink.setdefault(feat_key, {}).setdefault(feat_val, {
        "count": 0,
        "examples": [],
        "note": UNMAPPED_DERIVATIONS.get(feat_val, "") if feat_key == "Derivation" else "",
    })
    slot["count"] += 1
    if len(slot["examples"]) < 8:
        ex = f"{word['surface']}({word['lemma']})"
        if ex not in slot["examples"]:
            slot["examples"].append(ex)


# =============================================================================
# DECOMPOSER MATCHING  (duplicated from the METU adapter so this file stays
# self-contained — each dataset folder can evolve independently)
# =============================================================================

def _try_add_verb_lemma_to_dict(lemma, treebank_says_verb=False):
    import util.word_methods as wrd
    lemma_lower = tr_lower(lemma)
    if wrd.can_be_verb(lemma_lower):
        return False
    for inf in (lemma_lower + "mek", lemma_lower + "mak"):
        if inf in wrd.WORDS_SET:
            wrd.WORDS_SET.add(lemma_lower)
            decompose.cache_clear()
            return True
    if treebank_says_verb and lemma_lower:
        from util.word_methods import MajorHarmony, major_harmony
        harmony = major_harmony(lemma_lower)
        inf = lemma_lower + ("mak" if harmony == MajorHarmony.BACK else "mek")
        wrd.WORDS_SET.add(inf)
        decompose.cache_clear()
        return True
    return False


def match_against_decomposer(surface, lemma, expected_suffixes, force=False,
                             treebank_says_verb=False):
    """Run decompose(surface) and find a candidate whose chain matches the
    expected suffix sequence (with known-ambiguity normalisations)."""
    try:
        candidates = decompose(tr_lower(surface), force=force)
    except Exception:
        return None

    if not candidates:
        if _try_add_verb_lemma_to_dict(lemma, treebank_says_verb=treebank_says_verb):
            try:
                candidates = decompose(tr_lower(surface), force=force)
            except Exception:
                return None
    if not candidates:
        return None

    lemma_lower = tr_lower(lemma)

    def normalize_ler_poss(names):
        """plural_ler+posessive_3sg ↔ posessive_3pl."""
        result = []
        i = 0
        while i < len(names):
            if (i + 1 < len(names)
                    and names[i] == "plural_ler"
                    and names[i + 1] in ("posessive_3sg", "posessive_3pl")):
                result.append("_PLURAL_P3_")
                i += 2
            elif names[i] == "posessive_3pl":
                result.append("_PLURAL_P3_")
                i += 1
            else:
                result.append(names[i])
                i += 1
        return result

    def normalize_plural_conj(names):
        """plural_ler ↔ conjugation_3pl (surface -ler/-lar)."""
        return ["_PLURAL_OR_3PL_" if n in ("plural_ler", "conjugation_3pl") else n for n in names]

    def apply_equiv(names):
        result = list(names)
        for decomp_seq, tb_equiv in EQUIVALENT_SEQUENCES:
            k = len(decomp_seq)
            i = 0
            out = []
            while i < len(result):
                if result[i:i + k] == decomp_seq:
                    out.extend(tb_equiv)
                    i += k
                else:
                    out.append(result[i])
                    i += 1
            result = out
        return result

    def normalize_full(names):
        return normalize_plural_conj(normalize_ler_poss(apply_equiv(names)))

    def expand_alternatives(expected):
        results = [expected]
        for name, alts in SUFFIX_ALTERNATIVES.items():
            if name in expected:
                for alt in alts:
                    results.append([alt if n == name else n for n in expected])
        # Negative aorist zeroing: -ma-m / -ama-m drops factative_ir.
        for i in range(len(expected) - 1):
            if expected[i] in ("negative_me", "negative_able") and expected[i + 1] == "factative_ir":
                results.append(expected[:i + 1] + expected[i + 2:])
        return results

    expected_filtered = [n for n in expected_suffixes if n != "conjugation_3sg"]

    def get_chain_names(chain):
        return [s.name for s in chain if s.name != "conjugation_3sg"]

    if not expected_filtered:
        for root, start_pos, chain, final_pos in candidates:
            if root == lemma_lower and not chain:
                return (root, start_pos, chain, final_pos)
        for root, start_pos, chain, final_pos in candidates:
            if not chain:
                return (root, start_pos, chain, final_pos)
        return None

    all_expected_variants = expand_alternatives(expected_filtered)

    def tail_matches(chain_names, expected):
        k = len(expected)
        if len(chain_names) < k:
            return False
        return chain_names[-k:] == expected

    def tail_matches_normalized(chain_names, expected):
        cn = normalize_full(chain_names)
        en = normalize_full(expected)
        k = len(en)
        if len(cn) < k:
            return False
        return cn[-k:] == en

    # ── Tier-based matching ──
    # Treebanks are ground truth: if their suffix chain "somehow exists"
    # within a decomposer candidate, accept that candidate and write the
    # decomposition using Savyar's suffix nomenclature.
    # Tiers (higher wins):
    #   5 exact/normalized-exact  4 tail either direction
    #   3 contains contiguous     2 ordered subsequence  1 multiset equal
    def _ends_with(a, b):
        # Reject empty-vs-nonempty: an empty chain never "ends with" a
        # non-empty expected and vice versa. Both-empty is handled by the
        # tier-5 equality check above, so we need not allow it here.
        if not a or not b:
            return False
        return len(b) <= len(a) and a[-len(b):] == b

    def _contains_contig(a, b):
        if not a or not b:
            return False
        if len(b) > len(a):
            return False
        for i in range(len(a) - len(b) + 1):
            if a[i:i+len(b)] == b:
                return True
        return False

    def _is_subseq(a, b):
        if not a or not b:
            return False
        j = 0
        for x in a:
            if j < len(b) and x == b[j]:
                j += 1
        return j == len(b)

    def _match_tier(cn, en):
        cn_n, en_n = normalize_full(cn), normalize_full(en)
        if cn == en or cn_n == en_n:
            return 5
        if _ends_with(cn, en) or _ends_with(cn_n, en_n):
            return 4
        if _ends_with(en, cn) or _ends_with(en_n, cn_n):
            return 4
        if _contains_contig(cn, en) or _contains_contig(cn_n, en_n):
            return 3
        if _contains_contig(en, cn) or _contains_contig(en_n, cn_n):
            return 3
        if _is_subseq(cn, en) or _is_subseq(cn_n, en_n):
            return 2
        if _is_subseq(en, cn) or _is_subseq(en_n, cn_n):
            return 2
        if sorted(cn) == sorted(en) or sorted(cn_n) == sorted(en_n):
            return 1
        return -1

    best = None
    best_score = (False, -1, float("-inf"), float("-inf"))
    for root, start_pos, chain, final_pos in candidates:
        chain_names = get_chain_names(chain)
        if has_unexpected_nounifier_is(root, lemma_lower, chain_names, expected_filtered):
            continue
        is_lemma = (root == lemma_lower)
        for exp_variant in all_expected_variants:
            tier = _match_tier(chain_names, exp_variant)
            if tier < 0:
                continue
            length_penalty = -abs(len(chain_names) - len(exp_variant))
            score = (is_lemma, tier, length_penalty, -len(chain_names))
            if score > best_score:
                best = (root, start_pos, chain, final_pos)
                best_score = score

    return best


def diagnose_mismatch(surface, lemma, expected_suffixes, force=False):
    try:
        candidates = decompose(tr_lower(surface), force=force)
    except Exception as e:
        return {
            "reason": "decompose_error",
            "detail": f"decompose() raised: {e}",
            "expected": expected_suffixes,
            "closest": None,
            "diff": None,
        }

    lemma_lower = tr_lower(lemma)

    if not candidates:
        import util.word_methods as wrd
        lemma_known = wrd.can_be_noun(lemma_lower) or wrd.can_be_verb(lemma_lower)
        if lemma_known:
            return {
                "reason": "chain_build_failed",
                "detail": (
                    f"lemma '{lemma_lower}' IS in dictionary, but decompose() "
                    f"could not build any suffix chain for '{surface}'. "
                    f"Expected: {expected_suffixes}."
                ),
                "expected": expected_suffixes,
                "closest": None,
                "diff": None,
            }
        return {
            "reason": "root_not_in_dict",
            "detail": (
                f"lemma '{lemma_lower}' not in words.txt (and infinitive "
                f"{lemma_lower}mek/{lemma_lower}mak also absent)."
            ),
            "expected": expected_suffixes,
            "closest": None,
            "diff": None,
        }

    candidates_with_lemma = [c for c in candidates if c[0] == lemma_lower]
    if not candidates_with_lemma:
        other_roots = sorted({r for r, _, _, _ in candidates})[:4]
        return {
            "reason": "root_not_found",
            "detail": f"lemma '{lemma_lower}' not among decomposer roots. Roots: {other_roots}",
            "expected": expected_suffixes,
            "closest": {
                "root": candidates[0][0],
                "suffixes": [s.name for s in candidates[0][2]],
            },
            "diff": None,
        }

    import difflib

    def edit_distance(a, b):
        return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()

    def suffix_diff(chain_names, expected):
        matcher = difflib.SequenceMatcher(None, chain_names, expected)
        ops = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            elif tag == "replace":
                ops.append(f"replace {chain_names[i1:i2]} → {expected[j1:j2]}")
            elif tag == "delete":
                ops.append(f"extra {chain_names[i1:i2]} not in expected")
            elif tag == "insert":
                ops.append(f"missing {expected[j1:j2]}")
        return ops

    best = min(
        candidates_with_lemma,
        key=lambda c: edit_distance([s.name for s in c[2]], expected_suffixes),
    )
    best_names = [s.name for s in best[2]]
    diff_ops = suffix_diff(best_names, expected_suffixes)

    if not best_names and expected_suffixes:
        reason = "root_bare_expected_suffixes"
        detail = f"decomposer found bare root '{lemma_lower}', expected {expected_suffixes}"
    elif best_names and not expected_suffixes:
        reason = "root_has_extra_suffixes"
        detail = f"decomposer found suffixes {best_names}, expected bare root"
    else:
        reason = "suffix_sequence_mismatch"
        detail = (
            f"root '{lemma_lower}' found; decomposer={best_names} vs "
            f"expected={expected_suffixes}. Diff: {'; '.join(diff_ops)}"
        )

    all_root_chains = [[s.name for s in ch] for _, _, ch, _ in candidates_with_lemma][:5]
    return {
        "reason": reason,
        "detail": detail,
        "expected": expected_suffixes,
        "closest": {"root": lemma_lower, "suffixes": best_names},
        "all_root_candidates": all_root_chains,
        "diff": diff_ops,
    }


# =============================================================================
# WORD-ENTRY BUILDERS
# =============================================================================

def build_word_entry(surface, decomposition):
    root, start_pos, chain, final_pos = decomposition
    morphology_parts = [root]
    suffixes = []
    current_stem = root
    surface_lower = tr_lower(surface)
    for s in chain:
        forms = s.form(current_stem)
        form_used = ""
        rest = surface_lower[len(current_stem):]
        for f in forms:
            if f and rest.startswith(f):
                form_used = f
                break
        if not form_used:
            for f in forms:
                if f:
                    form_used = f
                    break
        if not form_used:
            form_used = s.suffix
        morphology_parts.append(form_used)
        suffixes.append({
            "name": s.name,
            "form": form_used,
            "makes": "VERB" if str(s.makes).upper().endswith("VERB") else "NOUN",
        })
        current_stem = current_stem + form_used
    return {
        "word": surface,
        "morphology_string": " ".join(morphology_parts),
        "root": root,
        "suffixes": suffixes,
        "final_pos": final_pos,
    }


def build_treebank_forced_entry(surface, lemma, expected_suffix_names):
    surface_lower = tr_lower(surface)
    root = tr_lower(lemma)

    suffixes = []
    current_stem = root
    for sname in expected_suffix_names:
        sobj = SUFFIX_BY_NAME.get(sname)
        if sobj:
            makes_str = "VERB" if sobj.makes == Type.VERB else "NOUN"
            try:
                forms = sobj.form(current_stem)
                form_str = ""
                rest = surface_lower[len(current_stem):]
                for form in forms:
                    if form and rest.startswith(form):
                        form_str = form
                        break
                if not form_str:
                    form_str = forms[0] if forms else sobj.suffix
            except Exception:
                form_str = sobj.suffix
            suffixes.append({"name": sname, "form": form_str, "makes": makes_str})
        else:
            suffixes.append({"name": sname, "form": "", "makes": "NOUN"})
        current_stem = current_stem + (suffixes[-1]["form"] or "")

    morphology_parts = [root] + [s["form"] for s in suffixes if s["form"]]
    return {
        "word": surface_lower,
        "morphology_string": " ".join(morphology_parts),
        "root": root,
        "suffixes": suffixes,
        "final_pos": "verb" if suffixes and suffixes[-1]["makes"] == "VERB" else "noun",
    }


def _build_cc_entry(surface_lower, cc_category):
    cc_entries = CLOSED_CLASS_LOOKUP.get(surface_lower, [])
    if not cc_entries:
        return None
    matched_cc = next((c for c in cc_entries if c.category == cc_category), cc_entries[0])
    suffix_name = f"cc_{matched_cc.category}"
    return {
        "word": surface_lower,
        "morphology_string": surface_lower,
        "root": surface_lower,
        "suffixes": [{"name": suffix_name, "form": "", "makes": "", "cc_surface": surface_lower}],
        "final_pos": suffix_name,
    }


# =============================================================================
# MAIN ADAPTER
# =============================================================================

def adapt_treebank(conllu_paths, output_path, stats_path=None,
                   unmatched_path=None, unmapped_path=None,
                   sentence_diagnostics_path=None):
    """Run the adapter over one or more .conllu files."""
    if isinstance(conllu_paths, (str, os.PathLike)):
        conllu_paths = [conllu_paths]

    # Parse every input file and combine.
    all_sentences = []
    for path in conllu_paths:
        print(f"Parsing: {path}")
        sents = parse_conllu(path)
        print(f"  -> {len(sents)} sentences")
        all_sentences.extend(sents)

    total_words = 0
    matched_words = 0
    forced_words = 0
    unmappable_words = 0
    no_suffix_words = 0

    matched_sentences = 0
    partial_sentences = 0
    failed_sentences = 0

    output_entries = []
    unmatched_log = []
    unmapped_features = {}   # {feature_key: {value: {count, examples, note}}}
    sentence_diagnostics = []

    for sent_idx, sentence_tokens in enumerate(all_sentences):
        if sent_idx % 500 == 0:
            print(f"  Processing sentence {sent_idx}/{len(all_sentences)}...")

        words = merge_ig_chains(sentence_tokens)
        if not words:
            continue

        original_parts = [w["surface"] for w in words]
        original_sentence = " ".join(original_parts)

        word_entries = []
        sentence_all_matched = True
        sentence_has_any = False
        sentence_unmappable = []
        bare_root_words = []
        skipped_words = []
        trainable_words_in_sentence = 0

        for word in words:
            total_words += 1

            # Strip apostrophes and other quote-like chars.
            surface = _strip_quotes(word["surface"])
            surface_lower = tr_lower(surface)
            lemma = _strip_quotes(word["lemma"])

            head_upos = word["head_upos"]
            head_xpos = word["head_xpos"]

            # Skip UPOS categories we don't morphologise.
            if head_upos in SKIP_UPOS:
                skipped_words.append(surface_lower)
                bare_root_words.append(surface_lower)
                word_entries.append({
                    "word": surface_lower,
                    "morphology_string": surface_lower,
                    "root": surface_lower,
                    "suffixes": [],
                    "final_pos": "noun",
                })
                no_suffix_words += 1
                continue

            # Closed-class path (single-token only — ig-chained words are
            # never closed-class). Pronouns stay closed-class even when they
            # are inflected; we do not want words.txt-style noun analyses for
            # pronoun paradigms.
            if not word["is_chain"]:
                cc_category = UPOS_TO_CC_CATEGORY.get(head_upos)
                if cc_category:
                    entry = _build_cc_entry(surface_lower, cc_category)
                    if entry:
                        word_entries.append(entry)
                        matched_words += 1
                        sentence_has_any = True
                        trainable_words_in_sentence += 1
                    else:
                        bare_root_words.append(surface_lower)
                        word_entries.append({
                            "word": surface_lower,
                            "morphology_string": surface_lower,
                            "root": surface_lower,
                            "suffixes": [],
                            "final_pos": "noun",
                        })
                        no_suffix_words += 1
                    continue

            # Map features → expected suffix names
            expected_suffixes, unmapped_feats, has_unmappable = \
                features_to_suffix_names(word, unmapped_features)

            if has_unmappable:
                unmappable_words += 1
                sentence_all_matched = False
                sentence_unmappable.append({
                    "surface": surface_lower,
                    "lemma": lemma,
                    "feature_layers": [
                        {"upos": l["upos"], "xpos": l["xpos"], "features": l["features"]}
                        for l in word["feature_layers"]
                    ],
                    "unmapped": list(unmapped_feats),
                })
                unmatched_log.append({
                    "surface": surface_lower,
                    "lemma": lemma,
                    "feature_layers": [
                        {"upos": l["upos"], "xpos": l["xpos"], "features": l["features"]}
                        for l in word["feature_layers"]
                    ],
                    "reason": "unmappable_features",
                    "detail": f"unmapped: {unmapped_feats}",
                })
                word_entries.append({
                    "word": surface_lower,
                    "morphology_string": surface_lower,
                    "root": surface_lower,
                    "suffixes": [],
                    "final_pos": "noun",
                })
                bare_root_words.append(surface_lower)
                continue

            if not expected_suffixes:
                no_suffix_words += 1
                bare_root_words.append(surface_lower)
                word_entries.append({
                    "word": surface_lower,
                    "morphology_string": surface_lower,
                    "root": surface_lower,
                    "suffixes": [],
                    "final_pos": "noun",
                })
                continue

            entry = build_treebank_forced_entry(surface_lower, lemma, expected_suffixes)
            word_entries.append(entry)
            matched_words += 1
            sentence_has_any = True
            trainable_words_in_sentence += 1

        # Emit the sentence entry
        if word_entries:
            decomposed_parts = [we["morphology_string"] for we in word_entries]
            output_entries.append({
                "type": "sentence",
                "original_sentence": original_sentence,
                "decomposed_sentence": " ".join(decomposed_parts),
                "words": word_entries,
            })
            if sentence_all_matched and sentence_has_any:
                matched_sentences += 1
            elif sentence_has_any:
                partial_sentences += 1
                sentence_diagnostics.append({
                    "sentence_index": sent_idx,
                    "original_sentence": original_sentence,
                    "diagnostic_type": "partially_trainable_sentence",
                    "why": "At least one token was trainable, but one or more tokens had unmappable features or had to remain bare roots.",
                    "how_to_fix": "Inspect the unmappable token list first. If it is empty, this sentence is only partially trainable because some tokens are bare roots or skipped POS.",
                    "trainable_word_count": trainable_words_in_sentence,
                    "bare_root_words": bare_root_words,
                    "skipped_words": skipped_words,
                    "unmappable_tokens": sentence_unmappable,
                })
            else:
                failed_sentences += 1
                diagnostic_type = "non_trainable_sentence"
                why = "No token in the sentence produced a trainable suffix sequence."
                how_to_fix = (
                    "Usually not an adapter bug. These are often suffixless fragments, titles, numeric snippets, or unmappable tokens."
                )
                if sentence_unmappable:
                    diagnostic_type = "non_trainable_due_to_unmappable_tokens"
                    why = "No token was trainable and at least one token has unmappable treebank features."
                    how_to_fix = "Add the missing treebank→Savyar mapping for the listed unmappable tokens."
                sentence_diagnostics.append({
                    "sentence_index": sent_idx,
                    "original_sentence": original_sentence,
                    "diagnostic_type": diagnostic_type,
                    "why": why,
                    "how_to_fix": how_to_fix,
                    "trainable_word_count": trainable_words_in_sentence,
                    "bare_root_words": bare_root_words,
                    "skipped_words": skipped_words,
                    "unmappable_tokens": sentence_unmappable,
                })

    # ── Write outputs ──
    print(f"\nWriting {len(output_entries)} sentences to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in output_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if unmatched_path is None:
        unmatched_path = output_path.replace(".jsonl", "_unmatched.jsonl")
    with open(unmatched_path, "w", encoding="utf-8") as f:
        for entry in unmatched_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if sentence_diagnostics_path is None:
        sentence_diagnostics_path = output_path.replace(".jsonl", "_sentence_diagnostics.jsonl")
    with open(sentence_diagnostics_path, "w", encoding="utf-8") as f:
        for entry in sentence_diagnostics:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Sort unmapped features by count (descending) for easier triage
    unmapped_sorted = {}
    for fkey in sorted(unmapped_features.keys()):
        by_val = unmapped_features[fkey]
        unmapped_sorted[fkey] = dict(sorted(
            by_val.items(), key=lambda kv: -kv[1]["count"]
        ))

    if unmapped_path is None:
        unmapped_path = os.path.join(os.path.dirname(output_path), "unmapped_features.json")
    with open(unmapped_path, "w", encoding="utf-8") as f:
        json.dump({
            "_header": (
                "Each feature value here was not mapped to a Savyar suffix. "
                "Fill in the mapping by editing the corresponding *_MAP dict "
                "at the top of treebank_adapter.py. Entries with a `note` are "
                "ones we had a hypothesis about but intentionally left for "
                "you to resolve."
            ),
            "unmapped": unmapped_sorted,
        }, f, indent=2, ensure_ascii=False)

    trainable_words = matched_words + forced_words
    stats = {
        "input_files":                      [str(p) for p in conllu_paths],
        "total_sentences":                  len(all_sentences),
        "total_words":                      total_words,
        "translated_words (treebank-authoritative)": matched_words,
        "compat_words (legacy-forced)":         forced_words,
        "trainable_words (total)":              trainable_words,
        "unmappable_words":                     unmappable_words,
        "no_suffix_words":                      no_suffix_words,
        "trainable_rate":
            f"{trainable_words / max(total_words - no_suffix_words, 1) * 100:.1f}%",
        "fully_trainable_sentences":     matched_sentences,
        "partially_trainable_sentences": partial_sentences,
        "non_trainable_sentences":       failed_sentences,
        "sentence_diagnostics_count":    len(sentence_diagnostics),
        "unmapped_feature_value_count":  sum(len(v) for v in unmapped_features.values()),
    }

    print("\n=== ADAPTATION STATS ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if stats_path:
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    # ── Console diagnostics (mirror the METU adapter's shape) ──
    if unmatched_log:
        decomp_mismatches = [e for e in unmatched_log if e.get("reason") not in (None, "", "unmappable_features")]
        unmappable_entries = [e for e in unmatched_log if e.get("reason") == "unmappable_features"]

        print(f"\n=== UNMAPPABLE WORDS ({len(unmappable_entries)}) ===")
        feat_counts = Counter()
        for e in unmappable_entries:
            detail = e.get("detail", "")
            for f in detail.replace("unmapped: ", "").strip("[]").split(","):
                s = f.strip().strip("'\"")
                if s:
                    feat_counts[s] += 1
        for feat, n in feat_counts.most_common(30):
            print(f"  {n:4d}x  {feat}")

        print(f"\n=== DECOMPOSER MISMATCH BREAKDOWN ({len(decomp_mismatches)}) ===")
        reason_counts = Counter(e["reason"] for e in decomp_mismatches)
        for reason, count in reason_counts.most_common():
            print(f"  {count:4d}x  {reason}")

    print(f"\nUnmapped feature VALUES recorded in: {unmapped_path}")
    return stats


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    inputs = [
        os.path.join(base_dir, "web.conllu"),
        os.path.join(base_dir, "wiki.conllu"),
    ]
    adapt_treebank(
        inputs,
        output_path=os.path.join(base_dir, "treebank_adapted.jsonl"),
        stats_path=os.path.join(base_dir, "treebank_adaptation_stats.json"),
        unmatched_path=os.path.join(base_dir, "treebank_adapted_unmatched.jsonl"),
        unmapped_path=os.path.join(base_dir, "unmapped_features.json"),
        sentence_diagnostics_path=os.path.join(base_dir, "treebank_adapted_sentence_diagnostics.jsonl"),
    )
