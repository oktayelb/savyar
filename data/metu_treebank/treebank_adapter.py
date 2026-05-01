"""
Treebank-to-Savyar Adapter
===========================
Translates METUSABANCI CoNLL treebank into sentence_valid_decompositions.jsonl format.

Strategy: DECOMPOSER-VALIDATED MATCHING
  1. Parse treebank → sentences with (word, lemma, features) per token
  2. Map treebank features → expected ordered list of Savyar suffix names
  3. Run decompose(word) → get all candidate decompositions
  4. Find the candidate whose root matches the lemma AND suffix names match
  5. Emit as JSONL training data

This gives us correct surface forms from the decomposer (no guessing morpheme boundaries).

"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from util.decomposer import decompose, ALL_SUFFIXES, enable_index

enable_index()
from util.suffix import  Type
from util.words.closed_class import CLOSED_CLASS_LOOKUP
from util.word_methods import tr_lower
import util.word_methods as wrd
from data.treebank_vnoun import (
    AMBIGUOUS_VNOUN,
    has_unexpected_nounifier_is,
    resolve_ambiguous_vnoun_suffixes,
)

# Name → suffix object lookup for building treebank-forced entries
SUFFIX_BY_NAME = {s.name: s for s in ALL_SUFFIXES}

# Quote-like characters to strip from surface/lemma before processing.
# Some treebanks leak these into tokens (e.g. BOUN prefixes `"Müjdeler`).
_QUOTE_CHARS = "\"'`‘’“”„«»‹›"

def _strip_quotes(s):
    if not s:
        return s
    for q in _QUOTE_CHARS:
        s = s.replace(q, "")
    return s

# =============================================================================
# TREEBANK FEATURE → SAVYAR SUFFIX NAME MAPPING
# =============================================================================
# The treebank features are listed in the order they typically appear.
# We map each feature to the Savyar suffix name it corresponds to.
#
# NAMING NOTES for Savyar:
#   locative_de  = locative case (-de/-da, "at/in")
#   ablative_den = ablative case (-den/-dan, "from")
#   noun_compound = genitive case (-in/-ın/-un/-ün/-nın/-nin)

# ── Zero morphemes: skip these ──
ZERO_FEATURES = {
    "A3sg",   # 3rd person singular agreement (zero suffix — NOT learned)
    "Pnon",   # No possession (absence of suffix)
    "Nom",    # Nominative case (zero suffix)
    "Pos",    # Positive polarity (absence of negation)
    "Prop",   # Proper noun marker (not a suffix)
    "Imp",    # Imperative mood (no tense nounifier — verb stays raw, only person/neg are real suffixes)
    "Demons", # Demonstrative base ("bu", "şu") — bare root, no suffix to learn
}

# ── V2V derivational features (voice) ──
V2V_FEATURES = {
    "Pass":   "passive_il",
    "Caus":   "active_dir",      # ambiguous: could be active_it/ir/er — best guess
    "Recip":  "reflexive_is",
    "Reflex": "reflexive_in",
}

# ── V2V compound features ──
V2V_COMPOUND_FEATURES = {
    "Able":      "possibilitative_ebil",
    "Hastily":   "suddenative_ivermek",
    "Stay":      "remainmative_ekalmak",
}

# ── Negation features ──
NEGATION_FEATURES = {
    "Neg": "negative_me",
    # "Neg" after "Able" (Able|Neg) is handled specially → negative_able
}

# ── V2N tense/aspect features (these are NOUNIFIERS in Savyar's grammar) ──
V2N_TENSE_FEATURES = {
    "Past":   "pasttense_di",       # -di/-dı/-tı/-du (predicative, V2N)
    "Narr":   "pastfactative_miş",  # -miş (V2N participle / evidential)
    "Prog1":  "continuous_iyor",    # -iyor (V2N predicative)
    "Aor":    "factative_ir",       # -ir/-er/-r (V2N participle / aorist)
    "Fut":    "nounifier_ecek",     # -ecek/-acak (V2N participle / future)
}

# ── V2N gerund/adverbial features ──
V2N_GERUND_FEATURES = {
    "ByDoingSo":            "adverbial_erek",    # -erek/-arak
    "AfterDoingSo":         "adverbial_ip",      # -ip/-ıp/-up/-üp (sequential)
    "When":                 "adverbial_ince",    # -ince/-ınca
    "While":                "when_ken",          # -ken (while/when)
    "AsLongAs":             "adverbial_dikçe",   # -dikçe/-dıkça
    "SinceDoingSo":         "adverbial_dikçe",   # approximate
    "WithoutHavingDoneSo":  ["infinitive_me", "ablative_den"],  # -meden/-madan
    "InBetween":            "adverbial_ip",      # -ip (in-between actions)
}

# ── V2N participle features (from XPOS column) ──
PARTICIPLE_XPOS = {
    "APresPart":  "factative_en",       # present participle as adj: -en/-an
    "APastPart":  "adjectifier_dik",    # past participle as adj: -dik/-dığ
    "AFutPart":   "nounifier_ecek",     # future participle as adj: -ecek/-acak
    "NPastPart":  "adjectifier_dik",    # past participle as noun: -dik/-dığ
    "NFutPart":   "nounifier_ecek",     # future participle as noun: -ecek/-acak
    "PresPart":   "factative_en",       # present participle
}

# ── V2N infinitive features (from XPOS) ──
INFINITIVE_XPOS = {
    "NInf": None,  # Could be infinitive_me, infinitive_mek, or nounifier_iş — resolved from surface
    "Inf2": "infinitive_me",
    "Inf3": "nounifier_iş",
}

# ── N2N case features ──
N2N_CASE_FEATURES = {
    "Dat":  "dative_e",         # -e/-a/-ye/-ya
    "Acc":  "accusative_i",     # -i/-ı/-u/-ü/-yi/-yı/-yu/-yü/-ni/-nı/-nu/-nü
    "Loc":  "locative_de",      # -de/-da/-te/-ta
    "Abl":  "ablative_den",     # -den/-dan/-ten/-tan
    "Gen":  "noun_compound",    # -in/-ın/-un/-ün/-nin/-nın/-nun/-nün
    "Ins":  "confactuous_le",   # -le/-la/-yle/-yla (instrumental)
    "Equ":  "relative_ce",      # -ce/-ca/-çe/-ça (equative ≈ relative_ce)
}

# ── N2N possessive features ──
N2N_POSSESSIVE_FEATURES = {
    "P1sg":  "posessive_1sg",
    "P2sg":  "posessive_2sg",
    "P3sg":  "posessive_3sg",
    "P1pl":  "posessive_1pl",
    "P2pl":  "posessive_2pl",
    "P3pl":  "posessive_3pl",
}

# ── N2N derivational features ──
N2N_DERIVATIONAL_FEATURES = {
    "Ness":    "suitative_lik",    # -lik/-lık/-luk/-lük
    "With":    "composessive_li",  # -li/-lı/-lu/-lü
    "Without": "privative_siz",    # -siz/-sız/-suz/-süz
    "Agt":     "actor_ci",         # -ci/-cı/-cu/-cü/-çi/-çı/-çu/-çü
    "Rel":     "marking_ki",       # -ki
    "Ly":      "relative_ce",      # -ce/-ca
    "FitFor":  "suitative_lik",    # -lik (approximate)
    "Related": "composessive_li",  # -li or -sel (approximate)
}

# ── Agreement/conjugation features ──
CONJUGATION_FEATURES = {
    "A1sg":  "conjugation_1sg",
    "A2sg":  "conjugation_2sg",
    # "A3sg" is zero — skipped
    "A1pl":  "conjugation_1pl",
    "A2pl":  "conjugation_2pl",
    "A3pl":  "conjugation_3pl",
}

# ── Copula features (noun predicates: Past/Narr on nouns) ──
COPULA_FEATURES = {
    "Past":  "pasttense_di",     # copula past: -ydı/-ydi
    "Narr":  "copula_mis",       # copula evidential: -ymış/-ymiş
    "Cop":   "nounaorist_dir",   # copula aorist: -dir/-dır/-tir/-tır
    "Cond":  "if_se",            # copula conditional: -se/-sa/-yse/-ysa
    "Pres":  None,               # present copula is zero (skip)
}

# ── Neces: -malı/-meli = infinitive_me + composessive_li ──
# başlamalı = başla + me + lı (must start)
NECES_SUFFIXES = ["infinitive_me", "composessive_li"]

# ── Cond: -se/-sa = if_se (copula in copula.py) ──
# gelse = gel + se (if he/she comes)
COND_SUFFIX = "if_se"

# ── Desr: desiderative -se/-sa on a verb = wish_suffix (V2N predicative) ──
# versem = ver + se(wish_suffix) + m(conjugation_1sg)
# arasan = ara + sa(wish_suffix) + n(conjugation_2sg)
DESR_SUFFIX = "wish_suffix"

# ── Acquire: -lan verbification = aplicative_le + reflexive_in ──
# heyecanlan = heyecan + la(aplicative_le) + n(reflexive_in)
ACQUIRE_SUFFIXES = ["aplicative_le", "reflexive_in"]

# ── Become: -leş mutual verbification = aplicative_le + reflexive_is ──
# demokratikleş = demokratik + le(aplicative_le) + ş(reflexive_is)
BECOME_SUFFIXES = ["aplicative_le", "reflexive_is"]

# ── As: -ce = relative_ce (equative/as-if) ──
# güzelce = güzel + ce
AS_SUFFIX = "relative_ce"

# ── AsIf: -cesine = adverbial_cesine ──
# delicesine = deli + cesine
ASIF_SUFFIX = "adverbial_cesine"

# ── JustLike: -ce = relative_ce ──
# çocukça = çocuk + ca
JUSTLIKE_SUFFIX = "relative_ce"

# ── Ord: -inci = ordinal_inci ──
# birinci = bir + inci, ikinci = iki + nci
ORD_SUFFIX = "ordinal_inci"

# ── Since: -eli = since_eli (gerund) + nounaorist_dir (copula) ──
# geleli = gel + eli; geleli(dir) = gel + eli + dir
SINCE_SUFFIXES = ["since_eli", "nounaorist_dir"]

# ── NotState: değil = negative_me + factative_ir + suitative_lik ──
NOTSTATE_SUFFIXES = ["negative_me", "factative_ir", "suitative_lik"]

# ── Prog2: -mekte = infinitive_mek + locative_de ──
# etmektedir = et + mek(infinitive_mek) + te(locative_de) + dir(nounaorist_dir)
PROG2_SUFFIXES = ["infinitive_mek", "locative_de"]

# ── Sequence equivalences for matching ──
# Each entry: (decomposer_sequence, treebank_equivalent)
# When a decomposer chain contains the LHS sequence, it is treated as the RHS
# for the purpose of matching against treebank expected suffixes.
EQUIVALENT_SEQUENCES = [
    (["aplicative_le", "factative_ir"], ["plural_ler"]),
]

OPTATIVE_SUFFIXES = ["adverbial_e"]

# ── Features we cannot map yet (not implemented in Savyar) ──
UNMAPPABLE_FEATURES = {
    "Dist",     # distributive
    "Time",     # zaman (temporal)
    "Demons",   # demonstrative base
}

# ── Treebank UPOS/XPOS → Savyar closed-class category ──
UPOS_TO_CC_CATEGORY = {
    "Conj":   "conjunction",
    "Postp":  "postposition",
    "Adv":    "adverb",
    "Interj": "interjection",
    "Det":    "determiner",
}
# Pron XPOS subtypes all map to "pronoun"
PRON_XPOS = {"PersP", "DemonsP", "QuesP", "ReflexP", "Pron"}

# ── Postposition case features ──
POSTP_CASE_FEATURES = {
    "PCNom":  None,
    "PCAcc":  "accusative_i",
    "PCDat":  "dative_e",
    "PCAbl":  "ablative_den",
    "PCGen":  "noun_compound",
    "PCIns":  "confactuous_le",
}


# =============================================================================
# TREEBANK PARSER
# =============================================================================

def parse_treebank(filepath):
    """Parse CoNLL file into list of sentences.
    Each sentence = list of token dicts.
    Multi-row DERIV tokens are merged into single words."""
    sentences = []
    current_sentence = []
    current_tokens = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if current_tokens:
                    sentence = merge_deriv_tokens(current_tokens)
                    if sentence:
                        sentences.append(sentence)
                    current_tokens = []
                continue

            parts = line.split("\t")
            if len(parts) < 8:
                continue

            token = {
                "id":       parts[0],
                "surface":  parts[1],
                "lemma":    parts[2],
                "upos":     parts[3],
                "xpos":     parts[4],
                "features": parts[5] if parts[5] != "_" else "",
                "head":     parts[6],
                "deprel":   parts[7],
            }
            current_tokens.append(token)

    if current_tokens:
        sentence = merge_deriv_tokens(current_tokens)
        if sentence:
            sentences.append(sentence)

    return sentences


def merge_deriv_tokens(tokens):
    """Merge multi-row DERIV chains into single word entries.

    In the treebank, a derived word like 'yapamazlar' is split as:
      row 6: _ | yap | Verb | Verb | _      | 7 | DERIV
      row 7: yapamazlar | _ | Verb | Verb | Able|Neg|Aor|A3pl | 8 | SENTENCE

    We merge these into a single entry with:
      surface = 'yapamazlar'
      lemma = 'yap'
      feature_chain = [('Verb', 'Verb', ''), ('Verb', 'Verb', 'Able|Neg|Aor|A3pl')]
    """
    merged = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # Skip punctuation
        if tok["upos"] == "Punc":
            i += 1
            continue

        # Check if this starts a DERIV chain
        if tok["deprel"] == "DERIV":
            chain_tokens = [tok]
            # Find the head token (the one this derives into)
            head_id = tok["head"]
            j = i + 1
            while j < len(tokens):
                next_tok = tokens[j]
                if next_tok["id"] == head_id:
                    # This could itself be a DERIV, or the final surface token
                    chain_tokens.append(next_tok)
                    if next_tok["deprel"] == "DERIV":
                        head_id = next_tok["head"]
                        j += 1
                        continue
                    else:
                        break
                j += 1

            # Build merged entry
            # Surface form comes from the last token with a real surface
            surface = None
            for ct in reversed(chain_tokens):
                if ct["surface"] != "_":
                    surface = ct["surface"]
                    break

            # Lemma comes from the first token with a real lemma
            lemma = None
            for ct in chain_tokens:
                if ct["lemma"] != "_":
                    lemma = ct["lemma"]
                    break

            if surface and lemma:
                # Build feature chain: list of (upos, xpos, features) for each step
                feature_chain = []
                for ct in chain_tokens:
                    feature_chain.append({
                        "upos": ct["upos"],
                        "xpos": ct["xpos"],
                        "features": ct["features"],
                    })

                merged.append({
                    "surface": surface,
                    "lemma": lemma,
                    "feature_chain": feature_chain,
                    "is_deriv_chain": True,
                })

                # Skip all tokens in this chain
                # Mark head tokens so we don't double-process
                chain_ids = {ct["id"] for ct in chain_tokens}
                i += 1
                while i < len(tokens) and tokens[i]["id"] in chain_ids:
                    i += 1
                continue
            else:
                # Fallback: treat as normal token
                pass

        # Normal (non-DERIV) token
        # Skip if this token was already consumed as part of a DERIV chain above
        merged.append({
            "surface": tok["surface"] if tok["surface"] != "_" else None,
            "lemma": tok["lemma"],
            "feature_chain": [{
                "upos": tok["upos"],
                "xpos": tok["xpos"],
                "features": tok["features"],
            }],
            "is_deriv_chain": False,
        })
        i += 1

    # Filter out entries without surface forms
    return [m for m in merged if m["surface"]]


# =============================================================================
# FEATURE → SUFFIX MAPPING
# =============================================================================

def features_to_suffix_names(token):
    """Convert treebank feature chain to expected Savyar suffix name sequence.

    Returns (suffix_names: list[str], unmapped: list[str], has_unmappable: bool)
    """
    suffix_names = []
    unmapped = []
    has_unmappable = False

    for step in token["feature_chain"]:
        upos = step["upos"]
        xpos = step["xpos"]
        feat_str = step["features"]
        feats = feat_str.split("|") if feat_str else []

        # Track what POS context we're in (noun vs verb) for disambiguation
        is_verb_context = upos == "Verb"
        is_noun_context = upos in ("Noun", "Adj", "Adv", "Pron", "Det")
        is_zero_verb = xpos == "Zero"  # copula "zero" derivation (noun→verb)
        is_pronoun = upos == "Pron" or xpos in ("PersP", "DemonsP", "ReflexP", "QuesP")

        # ── Handle XPOS-based participles/infinitives first ──
        if xpos in PARTICIPLE_XPOS:
            suffix_names.append(PARTICIPLE_XPOS[xpos])

        if xpos in INFINITIVE_XPOS:
            if INFINITIVE_XPOS[xpos]:
                suffix_names.append(INFINITIVE_XPOS[xpos])
            elif xpos == "NInf":
                # NInf is ambiguous: resolve it later from the actual surface.
                suffix_names.append(AMBIGUOUS_VNOUN)

        # ── Process each feature ──
        able_seen = False
        imp_seen = "Imp" in feats  # Imperative 2sg is zero (bare root)
        for feat in feats:
            if feat in ZERO_FEATURES:
                continue

            # In imperative mood, A2sg is zero — no conjugation suffix
            if imp_seen and feat == "A2sg":
                continue

            if feat == "Able":
                able_seen = True
                suffix_names.append(V2V_COMPOUND_FEATURES.get("Able", "possibilitative_ebil"))
                continue

            if feat == "Neg":
                if able_seen:
                    # Able|Neg → the -eme form (negative_able replaces possibilitative_ebil)
                    if suffix_names and suffix_names[-1] == "possibilitative_ebil":
                        suffix_names[-1] = "negative_able"
                    else:
                        suffix_names.append("negative_able")
                else:
                    suffix_names.append("negative_me")
                continue

            # ── V2V voice features ──
            if feat in V2V_FEATURES:
                suffix_names.append(V2V_FEATURES[feat])
                continue

            # ── V2V compound features (other than Able) ──
            if feat in V2V_COMPOUND_FEATURES:
                suffix_names.append(V2V_COMPOUND_FEATURES[feat])
                continue

            # ── Tense/aspect: context-dependent ──
            if feat in V2N_TENSE_FEATURES:
                if is_zero_verb or (is_verb_context and not is_noun_context):
                    # After a noun with Zero copula, tense is copula
                    if is_zero_verb and feat in COPULA_FEATURES:
                        mapped = COPULA_FEATURES[feat]
                        if mapped:
                            suffix_names.append(mapped)
                    else:
                        suffix_names.append(V2N_TENSE_FEATURES[feat])
                elif is_noun_context and feat in COPULA_FEATURES:
                    mapped = COPULA_FEATURES[feat]
                    if mapped:
                        suffix_names.append(mapped)
                else:
                    suffix_names.append(V2N_TENSE_FEATURES[feat])
                continue

            # ── Copula-only features ──
            if feat == "Cop":
                mapped = COPULA_FEATURES.get(feat)
                if mapped:
                    suffix_names.append(mapped)
                continue

            if feat == "Pres":
                # Present copula is usually zero
                continue

            # ── Gerunds/adverbials ──
            if feat in V2N_GERUND_FEATURES:
                mapped = V2N_GERUND_FEATURES[feat]
                if isinstance(mapped, list):
                    suffix_names.extend(mapped)
                else:
                    suffix_names.append(mapped)
                continue

            # ── Plural (A3pl on nouns = plural_ler) ──
            if feat == "A3pl":
                if is_noun_context or is_pronoun:
                    suffix_names.append("plural_ler")
                elif is_verb_context:
                    suffix_names.append("conjugation_3pl")
                continue

            # ── Possessive ──
            if feat in N2N_POSSESSIVE_FEATURES:
                suffix_names.append(N2N_POSSESSIVE_FEATURES[feat])
                continue

            # ── Case ──
            if feat in N2N_CASE_FEATURES:
                suffix_names.append(N2N_CASE_FEATURES[feat])
                continue

            # ── N2N derivational ──
            if feat in N2N_DERIVATIONAL_FEATURES:
                suffix_names.append(N2N_DERIVATIONAL_FEATURES[feat])
                continue

            # ── Conjugation/agreement ──
            # Skip person agreement on pronouns — "ben" is inherently 1sg,
            # A1sg on a pronoun is NOT a conjugation suffix
            if feat in CONJUGATION_FEATURES:
                if not is_pronoun:
                    suffix_names.append(CONJUGATION_FEATURES[feat])
                continue

            # ── Postposition case ──
            if feat in POSTP_CASE_FEATURES:
                mapped = POSTP_CASE_FEATURES[feat]
                if mapped:
                    suffix_names.append(mapped)
                continue

            # ── Neces: -malı/-meli = infinitive_me + composessive_li ──
            # başlamalı = başla + me + lı → V2N (infinitive) then N2N (composessive)
            if feat == "Neces":
                suffix_names.extend(NECES_SUFFIXES)
                continue

            # ── Cond: -se/-sa = if_se (copula) ──
            # gelse = gel+se, evdeyse = evde+yse
            if feat == "Cond":
                suffix_names.append(COND_SUFFIX)
                continue

            # ── Desr: desiderative -se/-sa on verb = wish_suffix (V2N predicative) ──
            # versem = ver + se + m, differs from Cond in that it expresses a wish
            if feat == "Desr":
                suffix_names.append(DESR_SUFFIX)
                continue

            # ── Acquire: -lan = aplicative_le + reflexive_in ──
            # heyecanlan = heyecan + la + n
            if feat == "Acquire":
                suffix_names.extend(ACQUIRE_SUFFIXES)
                continue

            if feat == "Opt":
                suffix_names.extend(OPTATIVE_SUFFIXES)
                continue

            # ── Become: -leş = aplicative_le + reflexive_is ──
            # demokratikleş = demokratik + le + ş
            if feat == "Become":
                suffix_names.extend(BECOME_SUFFIXES)
                continue

            # ── As: -ce = relative_ce ──
            if feat == "As":
                suffix_names.append(AS_SUFFIX)
                continue

            # ── AsIf: -cesine = adverbial_cesine ──
            if feat == "AsIf":
                suffix_names.append(ASIF_SUFFIX)
                continue

            # ── Prog2: -mekte = infinitive_mek + locative_de ──
            # etmektedir = et + mek + te + dir
            if feat == "Prog2":
                suffix_names.extend(PROG2_SUFFIXES)
                continue

            # ── JustLike: -ce = relative_ce ──
            if feat == "JustLike":
                suffix_names.append(JUSTLIKE_SUFFIX)
                continue

            # ── Ord: -inci = ordinal_inci ──
            if feat in ("Ord", "ord"):
                suffix_names.append(ORD_SUFFIX)
                continue

            # ── Since: -eli = since_eli + nounaorist_dir ──
            if feat in ("Since", "since"):
                suffix_names.extend(SINCE_SUFFIXES)
                continue

            # ── NotState: değil = negative_me + factative_ir + suitative_lik ──
            if feat == "NotState":
                suffix_names.extend(NOTSTATE_SUFFIXES)
                continue

            # ── Unmappable ──
            if feat in UNMAPPABLE_FEATURES:
                has_unmappable = True
                unmapped.append(feat)
                continue

            # Unknown feature
            if feat not in {"A3e"}:  # rare/malformed
                unmapped.append(feat)

    suffix_names = resolve_ambiguous_vnoun_suffixes(
        token["surface"],
        token["lemma"],
        suffix_names,
        SUFFIX_BY_NAME,
    )
    return suffix_names, unmapped, has_unmappable


# =============================================================================
# MISMATCH DIAGNOSTICS
# =============================================================================

def diagnose_mismatch(surface, lemma, expected_suffixes, force=False):
    """Classify exactly why the decomposer failed to match.

    Returns a dict with:
        reason       — short machine-readable category
        detail       — human-readable explanation
        expected     — the suffix sequence we wanted
        closest      — best candidate the decomposer found (root + suffixes)
        diff         — how the closest candidate differs from expected

    `force` must mirror what match_against_decomposer used so the diagnosis
    reflects the matcher's actual view (not a stale unforced re-run).
    """
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

    # ── Case 1: no decompositions at all ──
    if not candidates:
        # Don't lie about the cause. Actually check whether the lemma is
        # in the dictionary before blaming dictionary coverage.
        import util.word_methods as wrd
        lemma_known = (
            wrd.can_be_noun(lemma_lower) or wrd.can_be_verb(lemma_lower)
        )
        if lemma_known:
            reason = "chain_build_failed"
            detail = (
                f"lemma '{lemma_lower}' IS in dictionary, but decompose() "
                f"could not build any suffix chain for '{surface}'. "
                f"Expected: {expected_suffixes}. "
                f"Likely cause: a suffix form or hierarchy rule rejects this chain."
            )
        else:
            reason = "root_not_in_dict"
            detail = (
                f"lemma '{lemma_lower}' not in words.txt (and infinitive "
                f"{lemma_lower}mek/{lemma_lower}mak also absent). "
                f"decompose() returned zero candidates for '{surface}'."
            )
        return {
            "reason": reason,
            "detail": detail,
            "expected": expected_suffixes,
            "closest": None,
            "diff": None,
        }

    # ── Case 2: root present or absent ──
    candidates_with_lemma = [(r, pos, ch, fp) for r, pos, ch, fp in candidates if r == lemma_lower]
    candidates_without_lemma = [(r, pos, ch, fp) for r, pos, ch, fp in candidates if r != lemma_lower]

    if not candidates_with_lemma:
        # Decomposer found the word but under a different root
        other_roots = sorted({r for r, _, _, _ in candidates})[:4]
        return {
            "reason": "root_not_found",
            "detail": (
                f"lemma '{lemma_lower}' not among decomposer roots for '{surface}'. "
                f"Decomposer roots: {other_roots}"
            ),
            "expected": expected_suffixes,
            "closest": {
                "root": candidates[0][0],
                "suffixes": [s.name for s in candidates[0][2]],
            },
            "diff": None,
        }

    # ── Case 3: root found, but suffix sequences don't match ──
    # Find the closest candidate by edit distance on suffix name lists
    def suffix_diff(chain_names, expected):
        """Return list of (op, name) describing the difference."""
        # Use simple difflib-style comparison
        import difflib
        matcher = difflib.SequenceMatcher(None, chain_names, expected)
        ops = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue
            elif tag == 'replace':
                ops.append(f"replace {chain_names[i1:i2]} → {expected[j1:j2]}")
            elif tag == 'delete':
                ops.append(f"extra {chain_names[i1:i2]} not in expected")
            elif tag == 'insert':
                ops.append(f"missing {expected[j1:j2]}")
        return ops

    # Score each candidate with root==lemma by how close it is to expected
    def edit_distance(a, b):
        # Simple length-weighted diff count
        import difflib
        return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()

    best = min(
        candidates_with_lemma,
        key=lambda c: edit_distance([s.name for s in c[2]], expected_suffixes)
    )
    best_names = [s.name for s in best[2]]
    diff_ops = suffix_diff(best_names, expected_suffixes)

    # Characterise the gap
    if not best_names and expected_suffixes:
        detail = (
            f"decomposer found bare root '{lemma_lower}' (no suffixes), "
            f"but expected {expected_suffixes}"
        )
        reason = "root_bare_expected_suffixes"
    elif best_names and not expected_suffixes:
        detail = (
            f"decomposer found suffixes {best_names} on '{lemma_lower}', "
            f"but expected bare root"
        )
        reason = "root_has_extra_suffixes"
    else:
        detail = (
            f"root '{lemma_lower}' found, suffix mismatch: "
            f"decomposer={best_names}, expected={expected_suffixes}. "
            f"Diff: {'; '.join(diff_ops)}"
        )
        reason = "suffix_sequence_mismatch"

    # Collect all candidates with the right root for reference
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
# DECOMPOSER MATCHING
# =============================================================================

def _try_add_verb_lemma_to_dict(lemma: str, treebank_says_verb: bool = False) -> bool:
    """Make the decomposer able to find `lemma` as a verb root.

    Two paths:
      (1) lemma+mek/mak is already in the dictionary → the bare lemma is a
          legitimate verb root that was simply absent as a standalone entry.
      (2) treebank_says_verb=True: the treebank asserts this is a verb, but
          neither the lemma nor its infinitive is in words.txt. We trust
          the treebank and inject the right infinitive so can_be_verb works.

    Returns True if anything was added.
    """
    import util.word_methods as wrd
    lemma_lower = tr_lower(lemma)
    if wrd.can_be_verb(lemma_lower):
        return False  # decomposer can already find it

    # Path (1): infinitive already known, bare lemma just missing
    for inf in (lemma_lower + "mek", lemma_lower + "mak"):
        if inf in wrd.WORDS_SET:
            wrd.WORDS_SET.add(lemma_lower)
            decompose.cache_clear()
            return True

    # Path (2): treebank ground truth — inject the matching infinitive
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
    """Run decomposer on surface form and find a candidate whose suffix chain
    CONTAINS the treebank's expected suffixes.

    Philosophy: the treebank is ground truth for WHICH suffixes a word has.
    The decomposer may find a deeper root (going past the treebank's lemma),
    which prepends extra derivational suffixes.  We accept any candidate whose
    chain ENDS WITH the expected suffixes (with known ambiguity normalizations).
    We REJECT candidates that substitute different suffixes where the treebank
    dictates specific ones.

    Returns the matched decomposition tuple or None.
    """
    try:
        candidates = decompose(tr_lower(surface), force=force)
    except Exception:
        return None

    # If decomposer found nothing, check whether lemma+mek/mak exists in the
    # dictionary — if so, the bare lemma is a valid verb root that was simply
    # absent as a standalone entry.  Add it and retry once.
    if not candidates:
        if _try_add_verb_lemma_to_dict(lemma, treebank_says_verb=treebank_says_verb):
            try:
                candidates = decompose(tr_lower(surface), force=force)
            except Exception:
                return None

    if not candidates:
        return None

    lemma_lower = tr_lower(lemma)

    # ── Normalization helpers for known Turkish ambiguities ──

    def normalize_ler_poss(names):
        """plural_ler+posessive_3sg ↔ posessive_3pl (surface-identical -ları/-leri)."""
        result = []
        i = 0
        while i < len(names):
            if (i + 1 < len(names)
                and names[i] == "plural_ler"
                and names[i+1] in ("posessive_3sg", "posessive_3pl")):
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
        """plural_ler ↔ conjugation_3pl (surface-identical -ler/-lar)."""
        return ["_PLURAL_OR_3PL_" if n in ("plural_ler", "conjugation_3pl") else n for n in names]

    def apply_equiv(names):
        """Replace known equivalent subsequences (e.g. aplicative_le+factative_ir ↔ plural_ler)."""
        result = list(names)
        for decomp_seq, tb_equiv in EQUIVALENT_SEQUENCES:
            n = len(decomp_seq)
            i = 0
            out = []
            while i < len(result):
                if result[i:i+n] == decomp_seq:
                    out.extend(tb_equiv)
                    i += n
                else:
                    out.append(result[i])
                    i += 1
            result = out
        return result

    def normalize_full(names):
        """Apply all normalizations.
        Order matters: ler_poss must run before plural_conj so that
        plural_ler+posessive_3pl collapses before plural_ler is renamed."""
        return normalize_plural_conj(normalize_ler_poss(apply_equiv(names)))

    # Known suffix ambiguities: treebank may say X, decomposer may produce Y
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
        # -ti after -miş is surface-identical to pasttense_noundi (Savyar's
        # nominal past-tense variant). The treebank labels both as Past.
        "pasttense_di":      ["pasttense_noundi"],
        "pasttense_noundi":  ["pasttense_di"],
    }

    def expand_alternatives(expected):
        """Generate all plausible alternative expected sequences from known ambiguities."""
        results = [expected]
        for name, alts in SUFFIX_ALTERNATIVES.items():
            if name in expected:
                for alt in alts:
                    results.append([alt if n == name else n for n in expected])
        # Negative aorist: factative_ir after negative_me/negative_able may be
        # zero in some persons (anlamam = anla+ma+m, yapamam = yap+ama+m).
        for i in range(len(expected) - 1):
            if expected[i] in ("negative_me", "negative_able") and expected[i+1] == "factative_ir":
                results.append(expected[:i+1] + expected[i+2:])
        return results

    # ── Filter out conjugation_3sg (always zero) from both sides ──
    expected_filtered = [n for n in expected_suffixes if n != "conjugation_3sg"]

    def get_chain_names(chain):
        return [s.name for s in chain if s.name != "conjugation_3sg"]

    # ── Bare root fallback: if no suffixes expected, accept bare root ──
    if not expected_filtered:
        # Prefer lemma-matching root
        for root, start_pos, chain, final_pos in candidates:
            if root == lemma_lower and not chain:
                return (root, start_pos, chain, final_pos)
        for root, start_pos, chain, final_pos in candidates:
            if not chain:
                return (root, start_pos, chain, final_pos)
        return None

    # ── Generate all alternative expected sequences ──
    all_expected_variants = expand_alternatives(expected_filtered)

    # ── Matching: find candidates whose chain ends with expected suffixes ──
    # The decomposer may go deeper than the treebank lemma, prepending
    # derivational suffixes.  So we check: does the chain END WITH
    # the expected suffixes?  Extra prefix suffixes are the deeper root path.

    def tail_matches(chain_names, expected):
        """Check if chain_names ends with expected, or equals it exactly."""
        n = len(expected)
        if len(chain_names) < n:
            return False
        return chain_names[-n:] == expected

    def tail_matches_normalized(chain_names, expected):
        """Tail match with all normalizations applied."""
        cn = normalize_full(chain_names)
        en = normalize_full(expected)
        n = len(en)
        if len(cn) < n:
            return False
        return cn[-n:] == en

    # ── Tier-based matching ──
    # Treebanks are ground truth: if their suffix chain "somehow exists"
    # within a decomposer candidate, we accept that candidate and write the
    # decomposition using Savyar's suffix nomenclature.
    # Tiers (higher wins):
    #   5 = exact / normalized-exact
    #   4 = tail match either direction (chain ends with expected, OR
    #       expected ends with chain — covers deeper-root case where the
    #       decomposer baked initial expected suffixes into its root)
    #   3 = contains contiguous (either direction)
    #   2 = ordered subsequence (either direction)
    #   1 = multiset equal (same bag of suffixes, any order)
    # Score tuple: (is_lemma, tier, -abs(len_chain - len_expected), -len_chain).
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


def build_word_entry(surface, decomposition):
    """Build a word entry dict matching JSONL format."""
    root, start_pos, chain, final_pos = decomposition

    morphology_parts = [root]
    suffixes = []
    current_stem = root
    surface_lower = tr_lower(surface)
    for s in chain:
        # Compute form using the stem accumulated so far (vowel harmony depends on last vowel)
        forms = s.form(current_stem)
        # Find which form actually appears in the surface string
        form_used = ""
        rest = surface_lower[len(current_stem):]
        for f in forms:
            if f and rest.startswith(f):
                form_used = f
                break
        # Fallback: first non-empty form, then raw suffix
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
    """Build a word entry directly from treebank info, bypassing decomposer.

    The treebank is ground truth. If the decomposer doesn't produce a matching
    candidate, we still trust the treebank's analysis and build the entry from
    the suffix names it tells us.

    Uses SUFFIX_BY_NAME to look up real suffix objects for form/makes info.
    Falls back to raw name strings if a suffix isn't found in our inventory.
    """
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
            suffixes.append({
                "name": sname,
                "form": form_str,
                "makes": makes_str,
            })
        else:
            suffixes.append({
                "name": sname,
                "form": "",
                "makes": "NOUN",
            })
        current_stem = current_stem + (suffixes[-1]["form"] or "")

    morphology_parts = [root] + [s["form"] for s in suffixes if s["form"]]

    return {
        "word": surface_lower,
        "morphology_string": " ".join(morphology_parts),
        "root": root,
        "suffixes": suffixes,
        "final_pos": "verb" if suffixes and suffixes[-1]["makes"] == "VERB" else "noun",
    }


# =============================================================================
# CLOSED-CLASS WORD ENTRY BUILDER
# =============================================================================

def _build_cc_entry(surface_lower, cc_category):
    """Build a word entry for a closed-class word.

    Looks up surface_lower in CLOSED_CLASS_LOOKUP, finds a match for
    cc_category, and returns a JSONL word entry with the cc_XXX suffix name
    so that match_decompositions + encode_suffix_chain can handle it.

    Returns None if the word is not in CLOSED_CLASS_LOOKUP.
    """
    cc_entries = CLOSED_CLASS_LOOKUP.get(surface_lower, [])
    if not cc_entries:
        return None

    # Find a CC object matching the expected category
    matched_cc = None
    for cc_obj in cc_entries:
        if cc_obj.category == cc_category:
            matched_cc = cc_obj
            break
    # Fallback: use any CC entry for this surface if category not found
    if matched_cc is None:
        matched_cc = cc_entries[0]

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

def adapt_treebank(treebank_path, output_path, stats_path=None, sentence_diagnostics_path=None):
    """Main entry point: convert treebank to JSONL training data."""

    print(f"Parsing treebank: {treebank_path}")
    sentences = parse_treebank(treebank_path)
    print(f"Found {len(sentences)} sentences")

    total_words = 0
    matched_words = 0
    forced_words = 0
    unmatched_words = 0
    unmappable_words = 0  # words with features Savyar doesn't have
    no_suffix_words = 0   # bare roots (no suffixes to learn)
    skipped_pos = 0       # skipped POS categories

    matched_sentences = 0
    partial_sentences = 0
    failed_sentences = 0

    output_entries = []
    unmatched_log = []
    sentence_diagnostics = []

    skip_upos = {"Num", "Ques"}   # truly non-morphological; CC words handled below

    for sent_idx, sentence in enumerate(sentences):
        if sent_idx % 500 == 0:
            print(f"  Processing sentence {sent_idx}/{len(sentences)}...")

        # Build original sentence text
        original_parts = [tok["surface"] for tok in sentence]
        original_sentence = " ".join(original_parts)

        word_entries = []
        sentence_all_matched = True
        sentence_has_any = False
        sentence_unmappable = []
        bare_root_words = []
        skipped_words = []
        trainable_words_in_sentence = 0

        for tok in sentence:
            surface = tok["surface"]
            lemma = tok["lemma"]
            total_words += 1

            # Strip apostrophes and other quote-like chars (e.g. leading "
            # from quoted-dialog tokens). Turkish orthography separates
            # proper-noun roots with ' but it is not part of the morphology.
            surface = _strip_quotes(surface)
            lemma = _strip_quotes(lemma)

            # Lowercase surface for decomposer compatibility
            surface_lower = tr_lower(surface)

            # ── Numeric / question words: skip as bare root ──
            first_step = tok["feature_chain"][0]
            first_upos = first_step["upos"]
            first_xpos = first_step["xpos"]

            if first_upos in skip_upos:
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

            # ── Closed-class words: Conj, Postp, Adv, Det, Interj, Pron ──
            cc_category = UPOS_TO_CC_CATEGORY.get(first_upos)
            if first_upos == "Pron" and first_xpos in PRON_XPOS:
                cc_category = "pronoun"

            if cc_category:
                entry = _build_cc_entry(surface_lower, cc_category)
                if entry:
                    word_entries.append(entry)
                    matched_words += 1
                    sentence_has_any = True
                    trainable_words_in_sentence += 1
                else:
                    # CC word not in CLOSED_CLASS_LOOKUP — store as bare root
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

            # Map features to expected suffix names
            expected_suffixes, unmapped_feats, has_unmappable = features_to_suffix_names(tok)

            if has_unmappable:
                unmappable_words += 1
                sentence_all_matched = False
                sentence_unmappable.append({
                    "surface": surface_lower,
                    "lemma": lemma,
                    "features": [s["features"] for s in tok["feature_chain"]],
                    "unmapped": list(unmapped_feats),
                })
                unmatched_log.append({
                    "surface": surface_lower,
                    "lemma": lemma,
                    "features": [s["features"] for s in tok["feature_chain"]],
                    "reason": f"unmappable features: {unmapped_feats}",
                })
                # Store with surface as root so _preload_replay_buffer can match it
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
                # Bare root — no suffixes to learn
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

        # Build sentence entry
        if word_entries:
            decomposed_parts = []
            for we in word_entries:
                decomposed_parts.append(we["morphology_string"])

            entry = {
                "type": "sentence",
                "original_sentence": original_sentence,
                "decomposed_sentence": " ".join(decomposed_parts),
                "words": word_entries,
            }
            output_entries.append(entry)

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

    # Write output
    print(f"\nWriting {len(output_entries)} sentences to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in output_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Write unmatched log
    unmatched_path = output_path.replace(".jsonl", "_unmatched.jsonl")
    with open(unmatched_path, "w", encoding="utf-8") as f:
        for entry in unmatched_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if sentence_diagnostics_path is None:
        sentence_diagnostics_path = output_path.replace(".jsonl", "_sentence_diagnostics.jsonl")
    with open(sentence_diagnostics_path, "w", encoding="utf-8") as f:
        for entry in sentence_diagnostics:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Stats
    trainable_words = matched_words + forced_words
    stats = {
        "total_sentences": len(sentences),
        "total_words": total_words,
        "translated_words (treebank-authoritative)": matched_words,
        "compat_words (legacy-forced)": forced_words,
        "trainable_words (total)": trainable_words,
        "unmappable_words": unmappable_words,
        "no_suffix_words": no_suffix_words,
        "trainable_rate": f"{trainable_words / max(total_words - no_suffix_words, 1) * 100:.1f}%",
        "fully_trainable_sentences": matched_sentences,
        "partially_trainable_sentences": partial_sentences,
        "non_trainable_sentences": failed_sentences,
        "sentence_diagnostics_count": len(sentence_diagnostics),
    }

    print("\n=== ADAPTATION STATS ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if stats_path:
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    # ── Diagnostic report ──
    if unmatched_log:
        from collections import Counter

        decomp_mismatches = [e for e in unmatched_log if e.get("reason") not in (None, "") and not e["reason"].startswith("unmappable")]
        unmappable_entries = [e for e in unmatched_log if str(e.get("reason", "")).startswith("unmappable")]

        print(f"\n=== UNMAPPABLE FEATURES ({len(unmappable_entries)} words) ===")
        feat_counts = Counter()
        for e in unmappable_entries:
            for f in e.get("reason", "").replace("unmappable features: ", "").strip("[]'").split("', '"):
                feat_counts[f.strip("[]' ")] += 1
        for feat, n in feat_counts.most_common():
            print(f"  {n:4d}x  {feat}")

        print(f"\n=== DECOMPOSER MISMATCH BREAKDOWN ({len(decomp_mismatches)} words) ===")
        reason_counts = Counter(e["reason"] for e in decomp_mismatches)
        for reason, count in reason_counts.most_common():
            print(f"  {count:4d}x  {reason}")

        # ── chain_build_failed: lemma IS in dict, decomposer still got 0 ──
        chain_failed = [e for e in decomp_mismatches if e["reason"] == "chain_build_failed"]
        if chain_failed:
            print(f"\n  CHAIN_BUILD_FAILED — lemma IS in dictionary, decompose() returned 0 "
                  f"({len(chain_failed)} words): suffix-form or hierarchy issue")
            seen = set()
            for e in chain_failed:
                key = (e["surface"], tuple(e["expected"]))
                if key in seen: continue
                seen.add(key)
                try:
                    print(f"    {e['surface']:22s} lemma={e['lemma']:12s} expected: {e['expected']}")
                except UnicodeEncodeError:
                    pass
                if len(seen) >= 12: break

        # ── root_not_in_dict: lemma genuinely absent from words.txt ──
        root_missing = [e for e in decomp_mismatches if e["reason"] == "root_not_in_dict"]
        if root_missing:
            print(f"\n  ROOT_NOT_IN_DICT — lemma genuinely missing from words.txt "
                  f"({len(root_missing)} words):")
            seen = set()
            for e in root_missing:
                key = (e["surface"], tuple(e["expected"]))
                if key in seen: continue
                seen.add(key)
                try:
                    print(f"    {e['surface']:22s} lemma={e['lemma']:12s} expected: {e['expected']}")
                except UnicodeEncodeError:
                    pass
                if len(seen) >= 12: break

        # ── root_not_found: decomposer uses a different root ──
        wrong_root = [e for e in decomp_mismatches if e["reason"] == "root_not_found"]
        if wrong_root:
            print(f"\n  ROOT_NOT_FOUND — lemma not among decomposer roots ({len(wrong_root)} words):")
            seen = set()
            for e in wrong_root:
                key = e["surface"]
                if key in seen: continue
                seen.add(key)
                closest = e.get("closest") or {}
                try:
                    print(f"    {e['surface']:22s} lemma={e['lemma']:12s}  decomposer_root={closest.get('root','?'):12s}  decomposer_suffixes={closest.get('suffixes','?')}")
                except UnicodeEncodeError:
                    pass
                if len(seen) >= 12: break

        # ── suffix_sequence_mismatch: root found but wrong suffixes ──
        suffix_mismatch = [e for e in decomp_mismatches if e["reason"] == "suffix_sequence_mismatch"]
        if suffix_mismatch:
            print(f"\n  SUFFIX_SEQUENCE_MISMATCH — root found, wrong suffixes ({len(suffix_mismatch)} words):")
            seen = set()
            for e in suffix_mismatch:
                key = e["surface"]
                if key in seen: continue
                seen.add(key)
                closest = e.get("closest") or {}
                diff = e.get("diff") or []
                try:
                    print(f"    {e['surface']:22s}  expected={e['expected']}")
                    print(f"    {'':22s}  closest ={closest.get('suffixes','?')}")
                    if diff:
                        print(f"    {'':22s}  diff    : {' | '.join(diff)}")
                except UnicodeEncodeError:
                    pass
                if len(seen) >= 10: break

        # ── root_bare_expected_suffixes: decomposer gives bare root ──
        bare_root = [e for e in decomp_mismatches if e["reason"] == "root_bare_expected_suffixes"]
        if bare_root:
            print(f"\n  ROOT_BARE_EXPECTED_SUFFIXES — decomposer strips all suffixes ({len(bare_root)} words):")
            seen = set()
            for e in bare_root:
                key = e["surface"]
                if key in seen: continue
                seen.add(key)
                try:
                    print(f"    {e['surface']:22s}  expected: {e['expected']}")
                except UnicodeEncodeError:
                    pass
                if len(seen) >= 12: break

    return stats


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    treebank_path = os.path.join(base_dir, "METUSABANCI_treebank_v-1.conll")
    output_path = os.path.join(base_dir, "treebank_adapted.jsonl")
    stats_path = os.path.join(base_dir, "treebank_adaptation_stats.json")
    sentence_diagnostics_path = os.path.join(base_dir, "treebank_adapted_sentence_diagnostics.jsonl")

    adapt_treebank(treebank_path, output_path, stats_path, sentence_diagnostics_path)
