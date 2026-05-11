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

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.treebank_adapter_commons import (
    AMBIGUOUS_VNOUN,
    SUFFIX_BY_NAME,
    adapt_normalized_treebank,
    make_layer,
    make_word,
    parse_conllu,
    record_unmapped as _record_unmapped,
    resolve_ambiguous_vnoun_suffixes,
)


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
# TREEBANK ROW MERGER
# =============================================================================

def parse_google_conllu(filepath):
    return parse_conllu(filepath, keep_feature_order=True)


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
            make_layer(
                t["upos"], t["xpos"], t["features"],
                lemma=t["lemma"], surface=t["surface"],
                features_multi=t["features_multi"],
            )
            for t in chain
        ]

        merged.append(make_word(
            surface,
            lemma,
            feature_layers,
            is_multiword=len(chain) > 1,
            is_chain=len(chain) > 1,
            head_upos=chain[-1]["upos"],
            head_xpos=chain[-1]["xpos"],
        ))
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


# =============================================================================
# TREEBANK-SPECIFIC PIPELINE HOOKS
# =============================================================================

def should_skip_word(word):
    return word["head_upos"] in SKIP_UPOS


def closed_class_category(word):
    if word["is_multiword"]:
        return None
    return UPOS_TO_CC_CATEGORY.get(word["head_upos"])


def adapt_treebank(conllu_paths, output_path, stats_path=None,
                   unmatched_path=None, unmapped_path=None,
                   sentence_diagnostics_path=None):
    return adapt_normalized_treebank(
        conllu_paths,
        output_path,
        parse_sentences=parse_google_conllu,
        words_from_sentence=merge_ig_chains,
        translate_word=features_to_suffix_names,
        should_skip_word=should_skip_word,
        closed_class_category=closed_class_category,
        stats_path=stats_path,
        unmatched_path=unmatched_path,
        unmapped_path=unmapped_path,
        sentence_diagnostics_path=sentence_diagnostics_path,
        include_input_files_in_stats=True,
        include_unmapped_feature_value_count=True,
        unmapped_header=(
            "Each feature value here was not mapped to a Savyar suffix. "
            "Fill in the mapping by editing the corresponding *_MAP dict "
            "at the top of treebank_adapter.py. Entries with a `note` are "
            "ones we had a hypothesis about but intentionally left for "
            "you to resolve."
        ),
    )


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
