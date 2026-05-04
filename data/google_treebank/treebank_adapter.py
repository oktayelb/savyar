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

from util.decomposer import decompose, ALL_SUFFIXES
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
    "Make":      "applicative_le",        # -le/-la (noun → verb)
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
    "With":      "compositive_li",      # -li/-lı/-lu/-lü
    "Wout":      "privative_siz",        # -siz/-sız
    "Ness":      "suitative_lik",        # -lik/-lık
    "Rel":       "marking_ki",           # -ki
    "Agt":       "actor_ci",             # -ci/-cı
    "Like":      "adverbial_cesine",     # -cesine
    "Ly":        "relative_ce",          # -ce/-ca (manner)
    "Lang":      "relative_ce",          # -ce (language)
    "Act":       "relative_ce",          # -ce (manner, güzelce)
    "Rtd":       "relative_sel",         # -sel
    "Dim":       "diminutive_cik",       # -cik/-cık
    "Fam":       "familative_gil",       # -giller
    "Sim":       "approximative_si",     # -si
    "Aff":       "philicative_cil",      # -cil/-cül
    "Doct":      "ideologicative_izm",   # -izm
    # ── User-directed routings (semantic differences intentionally ignored) ──
    # Inh (-ıcı habitual doer) → if_se per directive.
    "Inh":       "actor_ci",
    # From (-li from-origin) → compositive_li (shares surface -li/-lı).
    "From":      "compositive_li",
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
    "Bcm": ["applicative_le", "reflexive_is"],  # -leş (become)
    "Acq": ["applicative_le", "reflexive_in"],  # -lan (acquire)
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
    "Nec":   ["infinitive_me", "compositive_li"],  # -meli/-malı
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
    "P1sg":  "possessive_1sg",
    "P2sg":  "possessive_2sg",
    "P3sg":  "possessive_3sg",
    "P1pl":  "possessive_1pl",
    "P2pl":  "possessive_2pl",
    "P3pl":  "possessive_3pl",
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
    "compositive_li":   ["relative_sel"],
    "relative_sel":      ["compositive_li"],
    "actor_ci":          ["factative_ir"],
}

# Suffix-chain equivalences (from METU).
EQUIVALENT_SEQUENCES = [
    (["applicative_le", "factative_ir"], ["plural_ler"]),
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


# =============================================================================
# WORD-ENTRY BUILDERS
# =============================================================================

def build_treebank_forced_entry(surface, lemma, expected_suffix_names):
    surface_lower = tr_lower(surface)
    root = tr_lower(lemma)

    suffixes = []
    current_stem = root
    accepted_chain = []
    for sname in expected_suffix_names:
        sobj = SUFFIX_BY_NAME.get(sname)
        if sobj:
            makes_str = "VERB" if sobj.makes == Type.VERB else "NOUN"
            try:
                forms = sobj.form(current_stem, current_chain=accepted_chain)
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
            accepted_chain.append(sobj)
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
