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

import json
import sys
import os
from collections import Counter

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

# Quote-like characters to strip from surface/lemma. BOUN leaks leading `"`
# into surface forms for quoted-dialog tokens (e.g. `"Müjdeler`).
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
# CONLL-U PARSER
# =============================================================================

def parse_conllu(filepath):
    """Parse a .conllu file → list of sentences. Each sentence is a list of
    token rows (every row is a dict). MWT header rows (id like `i-j`) are
    preserved with key ``mwt_range`` so we can stitch them downstream."""
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

            # Empty nodes: skip
            if "." in parts[0]:
                continue

            token = {
                "id":       parts[0],
                "surface":  parts[1],
                "lemma":    parts[2],
                "upos":     parts[3],
                "xpos":     parts[4],
                "features": _parse_feats(parts[5]),
                "head":     parts[6],
                "deprel":   parts[7],
                "misc":     parts[9] if len(parts) >= 10 else "_",
            }

            if "-" in parts[0]:
                a, b = parts[0].split("-")
                token["mwt_range"] = (int(a), int(b))
            else:
                token["mwt_range"] = None
                token["id_int"] = int(parts[0])

            current.append(token)
    if current:
        sentences.append(current)
    return sentences


def _parse_feats(field):
    if not field or field == "_":
        return {}
    out = {}
    for item in field.split("|"):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


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
                layers.append({
                    "upos":     r["upos"],
                    "xpos":     r["xpos"],
                    "features": r["features"],
                    "lemma":    r["lemma"],
                    "surface":  r["surface"],
                })
            if layers:
                # Lemma = first non-"_" lemma from sub-rows (typically the
                # content word, not the AUX copula).
                content_lemma = next(
                    (l["lemma"] for l in layers if l["lemma"] and l["lemma"] != "_"),
                    layers[0]["lemma"],
                )
                words.append({
                    "surface":        tok["surface"],
                    "lemma":          content_lemma,
                    "feature_layers": layers,
                    "is_mwt":         True,
                    "head_upos":      layers[0]["upos"],
                    "head_xpos":      layers[0]["xpos"],
                })
            i += 1
            continue

        # Standalone sub-row — only emit if it wasn't already covered by an MWT.
        if tok.get("id_int") is not None and tok["id_int"] not in covered:
            if tok["upos"] == "PUNCT":
                i += 1
                continue
            words.append({
                "surface": tok["surface"],
                "lemma":   tok["lemma"],
                "feature_layers": [{
                    "upos":     tok["upos"],
                    "xpos":     tok["xpos"],
                    "features": tok["features"],
                    "lemma":    tok["lemma"],
                    "surface":  tok["surface"],
                }],
                "is_mwt":    False,
                "head_upos": tok["upos"],
                "head_xpos": tok["xpos"],
            })
        i += 1
    return words


# =============================================================================
# FEATURE → SUFFIX MAPPING
# =============================================================================

def _record_unmapped(sink, feat_key, feat_val, word):
    slot = sink.setdefault(feat_key, {}).setdefault(feat_val, {
        "count": 0,
        "examples": [],
    })
    slot["count"] += 1
    if len(slot["examples"]) < 8:
        ex = f"{word['surface']}({word['lemma']})"
        if ex not in slot["examples"]:
            slot["examples"].append(ex)


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
            # builder routes to _build_cc_entry.
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
# DECOMPOSER MATCHING  (mirrors the METU / Google adapters)
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
    if isinstance(conllu_paths, (str, os.PathLike)):
        conllu_paths = [conllu_paths]

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
    unmapped_features = {}
    sentence_diagnostics = []

    for sent_idx, sentence_tokens in enumerate(all_sentences):
        if sent_idx % 500 == 0:
            print(f"  Processing sentence {sent_idx}/{len(all_sentences)}...")

        words = merge_mwt_words(sentence_tokens)
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

            surface = _strip_quotes(word["surface"])
            surface_lower = tr_lower(surface)
            lemma = _strip_quotes(word["lemma"])

            head_upos = word["head_upos"]

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

            # Closed-class words route to CC entries directly. Pronouns stay
            # closed-class even when they are inflected; we do not want
            # words.txt-style noun analyses for pronoun paradigms.
            if not word["is_mwt"] and head_upos in UPOS_TO_CC_CATEGORY:
                cc_category = UPOS_TO_CC_CATEGORY[head_upos]
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
                "Fill in the mapping by editing the corresponding *_MAP / combo "
                "function at the top of treebank_adapter.py."
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
