"""
BOUN Turkish Treebank (UD) → Savyar Adapter
===========================================
Translates the BOUN Turkish Universal Dependencies treebank (tr_boun-ud-*.conllu)
into the sentence_valid_decompositions.jsonl format consumed by Savyar's
training pipeline.

Key differences from the Google treebank adapter:
  - BOUN uses STANDARD UD features (Case, Number, Number[psor], Person,
    Person[psor], Tense, Aspect, Evident, Mood, Voice, VerbForm, ...) rather
    than Google's custom PersonNumber=A*/V* + TenseAspectMood encoding.
  - Morphology is collapsed onto ONE row per token: the whole inflection sits
    on a single feature bundle, not in ig-chain splits. Exception: multi-word
    tokens (MWTs) encoded as `i-j` span rows that stitch together a main token
    and an AUX (copula / question particle) — e.g. `yılındayız` = `yılında`
    (NOUN) + `yız` (AUX, lemma=i, copula).
  - Voice=Cau/Pass/Rfl/Rcp is treated as INFLECTIONAL on the verb row rather
    than Derivation=Cau/Pass/Rfx on its own layer.
  - VerbForm=Part/Conv/Vnoun plus the Tense value together determine which
    participle / converb / verbal-noun suffix was used.

Strategy: same as the METU & Google adapters — DECOMPOSER-VALIDATED MATCHING
  1. Parse the .conllu files into sentences.
  2. Merge MWT spans (`i-j` header rows) into single "words" with one
     feature-layer per sub-row. Non-MWT tokens become single-layer words.
  3. Map each layer's features → ordered list of Savyar suffix names.
  4. Run decompose(surface) and pick the candidate whose chain matches
     the expected suffix sequence (normalizing known ambiguities).
  5. Emit a JSONL word-entry per word, plus a per-sentence entry.

Files produced (alongside this adapter):
  - treebank_adapted.jsonl           matched + treebank-forced entries
  - treebank_adapted_unmatched.jsonl diagnostic log for mismatches
  - treebank_adaptation_stats.json   run statistics
  - unmapped_features.json           every feature value / combination we did
                                     NOT map, with frequency + examples — the
                                     user fills these in over time.
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

# ── Case ──
CASE_MAP = {
    "Nom":  None,                       # zero nominative
    "Loc":  "locative_de",
    "Dat":  "dative_e",
    "Acc":  "accusative_i",
    "Gen":  "noun_compound",            # genitive = Savyar's noun_compound
    "Abl":  "ablative_den",
    "Ins":  "confactuous_le",           # instrumental -le/-la
    "Equ":  "relative_ce",              # equative -ce (rare, same surface as manner)
}

# ── Possessive (Person[psor] + Number[psor] → suffix) ──
POSS_MAP = {
    ("1", "Sing"): "possessive_1sg",
    ("2", "Sing"): "possessive_2sg",
    ("3", "Sing"): "possessive_3sg",
    ("1", "Plur"): "possessive_1pl",
    ("2", "Plur"): "possessive_2pl",
    ("3", "Plur"): "possessive_3pl",
}

# ── V-person (verb conjugation on VERB/AUX) ──
V_PERSON_MAP = {
    ("1", "Sing"): "conjugation_1sg",
    ("2", "Sing"): "conjugation_2sg",
    ("3", "Sing"): None,                # zero 3sg
    ("1", "Plur"): "conjugation_1pl",
    ("2", "Plur"): "conjugation_2pl",
    ("3", "Plur"): "conjugation_3pl",
}

# ── Voice = derivational / inflectional voice on VERB ──
VOICE_MAP = {
    "Cau": "active_dir",
    "Pass": "passive_il",
    "Rfl": "reflexive_in",
    "Rcp": "reflexive_is",
}

# ── Mood values that map to a single suffix ──
MOOD_MAP = {
    "Ind":   None,                       # indicative — no suffix
    "Abil":  "possibilitative_ebil",     # -ebil/-abil
    "Cnd":   "if_se",                    # -se/-sa (conditional)
    "Gen":   "nounaorist_dir",           # -dir/-dır (generalizing copula)
    "Des":   "wish_suffix",              # desiderative
    "Opt":   "adverbial_e",              # -e/-a optative
    "Rapid": "suddenative_ivermek",      # -iver
    "Dur":   "persistive_egelmek",       # -egel
    # Imp handled specially (imperative 2sg/2pl)
    # Nec handled specially (-meli = infinitive_me + compositive_li)
    # Iter: no direct Savyar equivalent — recorded as unmapped
}

# ── Mood expansions (multi-suffix) ──
MOOD_MULTI = {
    "Nec": ["infinitive_me", "compositive_li"],   # -meli/-malı
}

# ── AUX "i" (i-copula) feature pattern → ordered suffix list (sans person). ──
# Person/Number come from the AUX row's own Number+Person and are appended
# AFTER these suffixes. Present indicative copula is zero, so we only emit
# person marker for Pres+Ind (and Perf-aspect).
def copula_suffixes_from_feats(feats):
    mood = feats.get("Mood")
    tense = feats.get("Tense")
    evident = feats.get("Evident")
    polarity = feats.get("Polarity", "Pos")
    out = []
    if mood == "Cnd":
        out.append("if_se")
    elif mood == "Gen":                     # -dır generalizing copula
        out.append("nounaorist_dir")
    elif mood == "Ind":
        if tense == "Past":
            if evident == "Nfh":
                out.append("copula_mis")     # -(y)miş
            else:
                out.append("pasttense_di")   # -(y)di
        # Pres+Ind = zero copula (no suffix emitted — just person marker)
    if polarity == "Neg":
        # Extremely rare on copula; record as unmapped in a note.
        out.append("__UNMAPPED_COPULA_NEG__")
    return out


# ── VERB-side TAM combinations. Returns list of suffixes for this combo. ──
def tam_suffixes_from_feats(feats):
    """Compute the TENSE/ASPECT/EVIDENTIAL suffix sequence for a VERB layer.

    The rules below were derived by sampling ~100 verb rows from BOUN. Unknown
    combinations return ``None`` so the caller records them as unmapped.
    """
    aspect  = feats.get("Aspect")
    evident = feats.get("Evident")
    tense   = feats.get("Tense")

    # Pluperfect / past-on-past (-miş + -ti)
    if tense in ("Pqb", "Pqp"):
        if evident == "Nfh":
            return ["pastfactative_miş", "copula_mis"]
        return ["pastfactative_miş", "pasttense_di"]

    # Future — always -ecek
    if tense in ("Fut", "Future"):
        return ["nounifier_ecek"]

    # Progressive — -iyor, optionally + -du
    if aspect == "Prog":
        if tense == "Past":
            return ["continuous_iyor", "pasttense_di"]
        return ["continuous_iyor"]

    # Habitual / aorist — -ir, optionally + -di
    if aspect == "Hab" or tense == "Aor":
        if tense == "Past":
            return ["factative_ir", "pasttense_di"]
        return ["factative_ir"]

    # Hearsay / non-first-hand past — always -miş (regardless of Aspect).
    if evident == "Nfh" and tense == "Past":
        return ["pastfactative_miş"]

    # Perfect
    if aspect == "Perf":
        if evident == "Nfh":
            return ["pastfactative_miş"]
        if tense == "Past":
            return ["pasttense_di"]
        if tense == "Pres":
            return []  # zero "perfect present" = just person marker
        return []

    # Imperfective past — treat as simple past
    if aspect == "Imp" and tense == "Past":
        return ["pasttense_di"]

    # If there's no aspect/tense at all, nothing to emit.
    if not aspect and not tense and not evident:
        return []

    return None  # unmapped combo


# ── VerbForm (nominalization) → which participle/converb/vnoun suffix ──
def verbform_suffix_from_feats(feats):
    """Map VerbForm + Tense → participle/converb/Vnoun suffix list.

    Returns ``None`` if the combination is unmapped.
    """
    vf = feats.get("VerbForm")
    if not vf:
        return []
    tense = feats.get("Tense")
    aspect = feats.get("Aspect")
    polarity = feats.get("Polarity", "Pos")

    if vf == "Part":
        if tense == "Past":
            return ["adjectifier_dik"]
        if tense in ("Fut", "Future"):
            return ["nounifier_ecek"]
        if tense == "Pres":
            return ["factative_en"]
        if tense == "Aor":
            return ["factative_ir"]
        # Part with no tense → default to -en if polarity=Pos
        if polarity == "Pos":
            return ["factative_en"]
        return None

    if vf == "Conv":
        # Converb surface is most commonly -erek/-arak; -ip/-ıp also common.
        # Without additional disambiguation cues, default to -erek.
        return ["adverbial_erek"]

    if vf == "Vnoun":
        # Verbal noun: resolve -me / -mek / -iş from the full surface later.
        return [AMBIGUOUS_VNOUN]

    return None


# ── UPOS → Savyar closed-class category ──
UPOS_TO_CC_CATEGORY = {
    "CCONJ":  "conjunction",
    "SCONJ":  "conjunction",
    "ADP":    "postposition",
    "ADV":    "adverb",
    "DET":    "determiner",
    "INTJ":   "interjection",
    "PRON":   "pronoun",
}

# UPOS categories we treat as bare-root (no suffix learning).
SKIP_UPOS = {"NUM", "PUNCT", "X", "SYM"}

# Feature keys we never turn into morphology (handled specially or irrelevant).
SKIP_FEATURE_KEYS = {
    "PronType",     # sub-type of pronoun (handled via closed-class path)
    "NumType",      # NumType=Ord handled specially below
    "Abbr",         # Abbreviation flag (not a suffix)
    "Echo",         # Reduplication marker (not a single suffix)
    "Reflex",       # Reflex=Yes on kendi-* (handled via closed-class/stem)
    "Polite",       # politeness flag (rare)
    "Polarity",     # handled specially (Neg logic)
    "Voice",        # handled specially
    "VerbForm",     # handled specially
    "Tense",        # consumed by TAM / VerbForm logic
    "Aspect",       # consumed by TAM logic
    "Evident",      # consumed by TAM logic
    "Mood",         # handled specially
    "Person",       # handled specially
    "Number",       # handled specially (plural / person-number)
    "Person[psor]", # handled specially (possessive)
    "Number[psor]", # handled specially (possessive)
    "Case",         # handled specially
}

# Known surface-level ambiguities between BOUN and Savyar's decomposer.
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
    # BOUN treats `-ti` after `-miş` as the same pasttense_di, but Savyar's
    # decomposer models it as pasttense_noundi (the nominal past-tense variant).
    "pasttense_di":      ["pasttense_noundi"],
}

# Suffix-chain equivalences (copied from METU).
EQUIVALENT_SEQUENCES = [
    (["applicative_le", "factative_ir"], ["plural_ler"]),
]


# =============================================================================
# TREEBANK ROW MERGER
# =============================================================================

def parse_boun_conllu(filepath):
    return parse_conllu(filepath, preserve_mwt=True)


def merge_mwt_words(sentence_tokens):
    """Collapse MWT spans into single word entries, preserving each sub-row as
    a feature layer. Non-MWT tokens become single-layer words."""
    # Index rows by their integer id (sub-rows only).
    sub_rows = {t["id_int"]: t for t in sentence_tokens if t.get("id_int") is not None}
    # MWT headers keep their order; sub-rows not inside any MWT stand alone.
    covered = set()
    words = []
    i = 0
    while i < len(sentence_tokens):
        tok = sentence_tokens[i]
        if tok.get("mwt_range"):
            a, b = tok["mwt_range"]
            layers = []
            for j in range(a, b + 1):
                r = sub_rows.get(j)
                if r is None:
                    continue
                covered.add(j)
                layers.append(make_layer(
                    r["upos"], r["xpos"], r["features"],
                    lemma=r["lemma"], surface=r["surface"],
                ))
            if layers:
                # Lemma = first non-"_" lemma from sub-rows (typically the
                # content word, not the AUX copula).
                content_lemma = next(
                    (l["lemma"] for l in layers if l["lemma"] and l["lemma"] != "_"),
                    layers[0]["lemma"],
                )
                words.append(make_word(
                    tok["surface"],
                    content_lemma,
                    layers,
                    is_multiword=True,
                    is_mwt=True,
                    head_upos=layers[0]["upos"],
                    head_xpos=layers[0]["xpos"],
                ))
            i += 1
            continue

        # Standalone sub-row — only emit if it wasn't already covered by an MWT.
        if tok.get("id_int") is not None and tok["id_int"] not in covered:
            if tok["upos"] == "PUNCT":
                i += 1
                continue
            words.append(make_word(
                tok["surface"],
                tok["lemma"],
                [make_layer(tok["upos"], tok["xpos"], tok["features"], lemma=tok["lemma"], surface=tok["surface"])],
                is_multiword=False,
                is_mwt=False,
                head_upos=tok["upos"],
                head_xpos=tok["xpos"],
            ))
        i += 1
    return words


# =============================================================================
# FEATURE → SUFFIX MAPPING
# =============================================================================

def features_to_suffix_names(word, unmapped_sink):
    """Map a merged word's feature_layers into the expected Savyar suffix chain.

    Returns (suffix_names, unmapped_feats_on_word, has_unmappable)."""
    suffix_names = []
    unmapped_on_word = []
    has_unmappable = False

    for layer_idx, layer in enumerate(word["feature_layers"]):
        feats = layer["features"]
        upos = layer["upos"]
        xpos = layer["xpos"]
        lemma = layer["lemma"]

        person = feats.get("Person")
        number = feats.get("Number")
        psor_person = feats.get("Person[psor]")
        psor_number = feats.get("Number[psor]")
        case = feats.get("Case")
        polarity = feats.get("Polarity")
        voice = feats.get("Voice")
        vform = feats.get("VerbForm")
        mood = feats.get("Mood")
        tense = feats.get("Tense")
        aspect = feats.get("Aspect")

        # Detect a "verb-verb" row: UPOS=VERB and the features describe verb
        # inflection (TAM/VerbForm/Mood/Voice/Polarity). Otherwise, a UPOS=VERB
        # row with only case/person features is a nominal predicate (NOMP-like).
        is_verb_layer = upos == "VERB" and any(
            feats.get(k) for k in ("Tense", "Aspect", "Evident", "VerbForm", "Mood", "Voice", "Polarity")
        )

        is_aux_copula = (upos == "AUX" and lemma in ("i", "YDİ", "YDU", "DU", "TU", "TİR"))
        is_aux_question = (upos == "AUX" and xpos == "Ques")
        is_aux_dur = (upos == "AUX" and lemma in ("dur", "tur", "dür", "tür", "dır", "tır"))
        is_closed_class_layer = upos in UPOS_TO_CC_CATEGORY

        # --------------------------------------------------------------
        # AUX / copula layer
        # --------------------------------------------------------------
        if is_aux_copula:
            cop_suffixes = copula_suffixes_from_feats(feats)
            for s in cop_suffixes:
                if s.startswith("__UNMAPPED"):
                    has_unmappable = True
                    unmapped_on_word.append(s)
                    _record_unmapped(unmapped_sink, "CopulaCombo",
                                     f"{feats.get('Mood')}|{feats.get('Polarity')}", word)
                else:
                    suffix_names.append(s)
            # Append verb-side person marker.
            if person and number:
                pm = V_PERSON_MAP.get((person, number), "__MISSING__")
                if pm is None:
                    pass
                elif pm == "__MISSING__":
                    has_unmappable = True
                    unmapped_on_word.append(f"PersonNumber={person}/{number}")
                    _record_unmapped(unmapped_sink, "PersonNumber",
                                     f"{person}/{number}", word)
                else:
                    suffix_names.append(pm)
            continue

        if is_aux_question:
            # Question particle mı/mi/mu/mü → closed-class "particle".
            suffix_names.append("cc_particle")
            continue

        if is_aux_dur:
            # Lexicalised auxiliaries: dur/tur with nounaorist_dir semantics +
            # optional person marker.
            suffix_names.append("nounaorist_dir")
            if person and number:
                pm = V_PERSON_MAP.get((person, number))
                if pm:
                    suffix_names.append(pm)
            continue

        # --------------------------------------------------------------
        # Closed-class layer (PRON/ADP/DET/INTJ/CCONJ/ADV/SCONJ)
        # Pronouns may carry case/possessive; handle them like nouns for
        # inflection and still route to the closed-class entry at word level.
        # Other CC categories with no inflection are emitted as bare CC.
        # --------------------------------------------------------------
        if is_closed_class_layer and upos != "PRON":
            # bare CC — no per-layer suffixes emitted here; the word-level
            # common pipeline routes to the closed-class entry builder.
            continue

        # --------------------------------------------------------------
        # VERB layer (with verb features) → canonical Turkish morpheme order:
        #   Voice → Polarity(Neg) → VerbForm(Part/Conv/Vnoun) → TAM → Mood → V-person
        # --------------------------------------------------------------
        if is_verb_layer:
            # 1) Voice
            if voice:
                vmap = VOICE_MAP.get(voice)
                if vmap:
                    suffix_names.append(vmap)
                else:
                    has_unmappable = True
                    unmapped_on_word.append(f"Voice={voice}")
                    _record_unmapped(unmapped_sink, "Voice", voice, word)

            # 2) Polarity=Neg (unless absorbed by Mood=Abil fusion → negative_able)
            if polarity == "Neg":
                if mood == "Abil":
                    suffix_names.append("negative_able")
                else:
                    suffix_names.append("negative_me")

            # 3) VerbForm (participle / converb / verbal noun)
            if vform:
                vf_suffs = verbform_suffix_from_feats(feats)
                if vf_suffs is None:
                    has_unmappable = True
                    combo = f"{vform}|Tense={tense}|Aspect={aspect}|Polarity={polarity}"
                    unmapped_on_word.append(f"VerbForm={combo}")
                    _record_unmapped(unmapped_sink, "VerbForm", combo, word)
                else:
                    suffix_names.extend(vf_suffs)
            else:
                # 4) TAM (tense/aspect/evidential) — only when NOT a VerbForm row.
                tam = tam_suffixes_from_feats(feats)
                if tam is None:
                    has_unmappable = True
                    combo = f"Tense={tense}|Aspect={aspect}|Evident={feats.get('Evident')}"
                    unmapped_on_word.append(f"TAMCombo={combo}")
                    _record_unmapped(unmapped_sink, "TAMCombo", combo, word)
                else:
                    suffix_names.extend(tam)

            # 5) Mood (apart from Ind / Abil fusion already consumed above)
            if mood and mood not in ("Ind", "Abil"):
                if mood == "Imp":
                    # Imperative: 2sg zero, 2pl handled below as conjugation_2pl
                    pass
                elif mood in MOOD_MULTI:
                    suffix_names.extend(MOOD_MULTI[mood])
                elif mood in MOOD_MAP:
                    mapped = MOOD_MAP[mood]
                    if mapped is not None:
                        suffix_names.append(mapped)
                else:
                    has_unmappable = True
                    unmapped_on_word.append(f"Mood={mood}")
                    _record_unmapped(unmapped_sink, "Mood", mood, word)
            elif mood == "Abil" and polarity != "Neg":
                suffix_names.append("possibilitative_ebil")

            # 6) Case appearing on a verb layer = the verb is nominalised
            # (usually via VerbForm=Part or Vnoun). Apply it after verb
            # morphology but before person marking.
            if case and case != "Nom":
                cm = CASE_MAP.get(case, "__MISSING__")
                if cm is None:
                    pass
                elif cm == "__MISSING__":
                    has_unmappable = True
                    unmapped_on_word.append(f"Case={case}")
                    _record_unmapped(unmapped_sink, "Case", case, word)
                else:
                    # Possessive on nominalised verb (e.g. ol+duğ+u+n+u)
                    if psor_person and psor_number:
                        pm = POSS_MAP.get((psor_person, psor_number))
                        if pm:
                            suffix_names.append(pm)
                    suffix_names.append(cm)
            else:
                # Possessive on verb row without case — still mark it.
                if psor_person and psor_number:
                    pm = POSS_MAP.get((psor_person, psor_number))
                    if pm:
                        suffix_names.append(pm)

            # 7) Verb-side person marker
            if person and number and not vform:
                # VerbForm nominalisations don't take verb-side person; they
                # take possessive (handled above). Plain finite verbs do.
                pm = V_PERSON_MAP.get((person, number), "__MISSING__")
                if pm is None:
                    pass
                elif pm == "__MISSING__":
                    has_unmappable = True
                    unmapped_on_word.append(f"PersonNumber={person}/{number}")
                    _record_unmapped(unmapped_sink, "PersonNumber",
                                     f"{person}/{number}", word)
                else:
                    suffix_names.append(pm)
            continue

        # --------------------------------------------------------------
        # NOUN / ADJ / PROPN / NOMP-like VERB layer (noun inflection)
        # --------------------------------------------------------------
        # 1) Number=Plur on a noun → plural_ler
        if number == "Plur" and upos not in ("VERB",):
            suffix_names.append("plural_ler")

        # 2) Possessive
        if psor_person and psor_number:
            pm = POSS_MAP.get((psor_person, psor_number))
            if pm:
                suffix_names.append(pm)

        # 3) Case
        if case:
            cm = CASE_MAP.get(case, "__MISSING__")
            if cm is None:
                pass
            elif cm == "__MISSING__":
                has_unmappable = True
                unmapped_on_word.append(f"Case={case}")
                _record_unmapped(unmapped_sink, "Case", case, word)
            else:
                suffix_names.append(cm)

        # 4) NumType=Ord → ordinal_inci
        if feats.get("NumType") == "Ord":
            suffix_names.append("ordinal_inci")

        # 5) NOMP-like: UPOS=VERB with only noun features + Person marker = copula'd nominal
        if upos == "VERB" and not is_verb_layer and person and number:
            pm = V_PERSON_MAP.get((person, number))
            if pm:
                suffix_names.append(pm)

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
        parse_sentences=parse_boun_conllu,
        words_from_sentence=merge_mwt_words,
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
            "Fill in the mapping by editing the corresponding *_MAP / combo "
            "function at the top of treebank_adapter.py."
        ),
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    inputs = [
        os.path.join(base_dir, "tr_boun-ud-train.conllu"),
        os.path.join(base_dir, "tr_boun-ud-dev.conllu"),
        os.path.join(base_dir, "tr_boun-ud-test.conllu"),
    ]
    adapt_treebank(
        inputs,
        output_path=os.path.join(base_dir, "treebank_adapted.jsonl"),
        stats_path=os.path.join(base_dir, "treebank_adaptation_stats.json"),
        unmatched_path=os.path.join(base_dir, "treebank_adapted_unmatched.jsonl"),
        unmapped_path=os.path.join(base_dir, "unmapped_features.json"),
        sentence_diagnostics_path=os.path.join(base_dir, "treebank_adapted_sentence_diagnostics.jsonl"),
    )
