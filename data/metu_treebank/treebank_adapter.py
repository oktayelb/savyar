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

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.treebank_adapter_commons import (
    AMBIGUOUS_VNOUN,
    SUFFIX_BY_NAME,
    adapt_normalized_treebank,
    make_layer,
    make_word,
    resolve_ambiguous_vnoun_suffixes,
)

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
    "P1sg":  "possessive_1sg",
    "P2sg":  "possessive_2sg",
    "P3sg":  "possessive_3sg",
    "P1pl":  "possessive_1pl",
    "P2pl":  "possessive_2pl",
    "P3pl":  "possessive_3pl",
}

# ── N2N derivational features ──
N2N_DERIVATIONAL_FEATURES = {
    "Ness":    "suitative_lik",    # -lik/-lık/-luk/-lük
    "With":    "compositive_li",  # -li/-lı/-lu/-lü
    "Without": "privative_siz",    # -siz/-sız/-suz/-süz
    "Agt":     "actor_ci",         # -ci/-cı/-cu/-cü/-çi/-çı/-çu/-çü
    "Rel":     "marking_ki",       # -ki
    "Ly":      "relative_ce",      # -ce/-ca
    "FitFor":  "suitative_lik",    # -lik (approximate)
    "Related": "compositive_li",  # -li or -sel (approximate)
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

# ── Neces: -malı/-meli = infinitive_me + compositive_li ──
# başlamalı = başla + me + lı (must start)
NECES_SUFFIXES = ["infinitive_me", "compositive_li"]

# ── Cond: -se/-sa = if_se (copula in copula.py) ──
# gelse = gel + se (if he/she comes)
COND_SUFFIX = "if_se"

# ── Desr: desiderative -se/-sa on a verb = wish_suffix (V2N predicative) ──
# versem = ver + se(wish_suffix) + m(conjugation_1sg)
# arasan = ara + sa(wish_suffix) + n(conjugation_2sg)
DESR_SUFFIX = "wish_suffix"

# ── Acquire: -lan verbification = applicative_le + reflexive_in ──
# heyecanlan = heyecan + la(applicative_le) + n(reflexive_in)
ACQUIRE_SUFFIXES = ["applicative_le", "reflexive_in"]

# ── Become: -leş mutual verbification = applicative_le + reflexive_is ──
# demokratikleş = demokratik + le(applicative_le) + ş(reflexive_is)
BECOME_SUFFIXES = ["applicative_le", "reflexive_is"]

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
    (["applicative_le", "factative_ir"], ["plural_ler"]),
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

    We merge these into a normalized word with:
      surface = 'yapamazlar'
      lemma = 'yap'
      feature_layers = [('Verb', 'Verb', ''), ('Verb', 'Verb', 'Able|Neg|Aor|A3pl')]
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
                # Build normalized feature layers for each derivational step.
                feature_layers = []
                for ct in chain_tokens:
                    feature_layers.append(make_layer(
                        ct["upos"], ct["xpos"], ct["features"],
                        lemma=ct["lemma"], surface=ct["surface"],
                    ))

                merged.append(make_word(
                    surface,
                    lemma,
                    feature_layers,
                    is_multiword=True,
                    is_deriv_chain=True,
                ))

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
        merged.append(make_word(
            tok["surface"] if tok["surface"] != "_" else None,
            tok["lemma"],
            [make_layer(tok["upos"], tok["xpos"], tok["features"], lemma=tok["lemma"], surface=tok["surface"])],
            is_multiword=False,
            is_deriv_chain=False,
        ))
        i += 1

    # Filter out entries without surface forms
    return [m for m in merged if m["surface"]]


# =============================================================================
# FEATURE → SUFFIX MAPPING
# =============================================================================

def features_to_suffix_names(word, _unmapped_sink=None):
    """Convert treebank feature chain to expected Savyar suffix name sequence.

    Returns (suffix_names: list[str], unmapped: list[str], has_unmappable: bool)
    """
    suffix_names = []
    unmapped = []
    has_unmappable = False

    for step in word["feature_layers"]:
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

            # ── Neces: -malı/-meli = infinitive_me + compositive_li ──
            # başlamalı = başla + me + lı → V2N (infinitive) then N2N (compositive)
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

            # ── Acquire: -lan = applicative_le + reflexive_in ──
            # heyecanlan = heyecan + la + n
            if feat == "Acquire":
                suffix_names.extend(ACQUIRE_SUFFIXES)
                continue

            if feat == "Opt":
                suffix_names.extend(OPTATIVE_SUFFIXES)
                continue

            # ── Become: -leş = applicative_le + reflexive_is ──
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
        word["surface"],
        word["lemma"],
        suffix_names,
        SUFFIX_BY_NAME,
    )
    return suffix_names, unmapped, has_unmappable


# =============================================================================
# TREEBANK-SPECIFIC PIPELINE HOOKS
# =============================================================================

def should_skip_word(word):
    first_step = word["feature_layers"][0]
    return first_step["upos"] in {"Num", "Ques"}


def closed_class_category(word):
    first_step = word["feature_layers"][0]
    first_upos = first_step["upos"]
    first_xpos = first_step["xpos"]
    if first_upos == "Pron" and first_xpos in PRON_XPOS:
        return "pronoun"
    return UPOS_TO_CC_CATEGORY.get(first_upos)


def metu_unmappable_context(word):
    return [s["features"] for s in word["feature_layers"]]


def adapt_treebank(treebank_path, output_path, stats_path=None, sentence_diagnostics_path=None):
    """Main entry point: convert treebank to JSONL training data."""
    return adapt_normalized_treebank(
        treebank_path,
        output_path,
        parse_sentences=parse_treebank,
        translate_word=features_to_suffix_names,
        should_skip_word=should_skip_word,
        closed_class_category=closed_class_category,
        stats_path=stats_path,
        sentence_diagnostics_path=sentence_diagnostics_path,
        word_context=metu_unmappable_context,
        unmappable_context_key="features",
        unmappable_reason="unmappable features",
        unmappable_detail=False,
        parse_message="Parsing treebank: {path}",
        parsed_message="Found {count} sentences",
        summary_unmappable_label="UNMAPPABLE FEATURES",
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    treebank_path = os.path.join(base_dir, "METUSABANCI_treebank_v-1.conll")
    output_path = os.path.join(base_dir, "treebank_adapted.jsonl")
    stats_path = os.path.join(base_dir, "treebank_adaptation_stats.json")
    sentence_diagnostics_path = os.path.join(base_dir, "treebank_adapted_sentence_diagnostics.jsonl")

    adapt_treebank(treebank_path, output_path, stats_path, sentence_diagnostics_path)
