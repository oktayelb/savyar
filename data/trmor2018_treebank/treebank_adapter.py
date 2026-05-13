"""
TRMor2018 Treebank -> Savyar Adapter
====================================

TRMor2018 is not CoNLL-U despite the local file extension. It is an XML-ish
corpus where each sentence contains token rows in this shape:

    surface<TAB>gold_analysis<TAB>alternative_analysis...

The adapter keeps the TRMor2018-specific job here:
  - read sentence blocks and select the first/gold analysis,
  - parse TRmorph analysis strings into normalized feature layers,
  - translate TRmorph tags into Savyar suffix names.

The common treebank pipeline handles formatting, JSONL writing, stats,
diagnostics, closed-class entries, and treebank-forced Savyar word entries.
"""

"""
@article{DBLP:journals/corr/abs-1805-07946,
  author    = {Erenay Dayanik and Ekin Aky{\"{u}}rek and Deniz Yuret},
  title     = {MorphNet: {A} sequence-to-sequence model that combines morphological analysis and disambiguation},
  journal   = {CoRR},
  volume    = {abs/1805.07946},
  year      = {2018},
  url       = {http://arxiv.org/abs/1805.07946},
  archivePrefix = {arXiv},
  eprint    = {1805.07946},
  timestamp = {Mon, 13 Aug 2018 16:47:09 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1805-07946},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.treebank_adapter_commons import (
    AMBIGUOUS_VNOUN,
    SUFFIX_BY_NAME,
    adapt_normalized_treebank,
    make_layer,
    make_word,
    record_unmapped,
    resolve_ambiguous_vnoun_suffixes,
)


# =============================================================================
# TRMORPH TAG -> SAVYAR SUFFIX NAME MAPPING
# =============================================================================

ZERO_FEATURES = {
    "A3sg",
    "Pnon",
    "Nom",
    "Pos",
    "Prop",
    "Imp",
    "Pres",
    "Card",
    "Real",
    "Ratio",
    "Noun",
    "Zero",
}

V2V_FEATURES = {
    "Pass": "passive_il",
    "Caus": "active_dir",
    "Recip": "reflexive_is",
    "Reflex": "reflexive_in",
}

V2V_COMPOUND_FEATURES = {
    "Able": "possibilitative_ebil",
    "Hastily": "suddenative_ivermek",
    "Stay": "remainmative_ekalmak",
    "EverSince": "persistive_egelmek",
}

V2N_TENSE_FEATURES = {
    "Past": "pasttense_di",
    "Narr": "pastfactative_miş",
    "Prog1": "continuous_iyor",
    "Aor": "factative_ir",
    "Fut": "nounifier_ecek",
}

V2N_GERUND_FEATURES = {
    "ByDoingSo": "adverbial_erek",
    "AfterDoingSo": "adverbial_ip",
    "When": "adverbial_ince",
    "While": "when_ken",
    "AsLongAs": "adverbial_dikçe",
    "SinceDoingSo": "since_eli",
    "WithoutHavingDoneSo": ["infinitive_me", "ablative_den"],
    "WithoutBeingAbleToHaveDoneSo": ["negative_able", "infinitive_me", "ablative_den"],
    "InBetween": "adverbial_ip",
}

PARTICIPLE_FEATURES = {
    "PresPart": "factative_en",
    "PastPart": "adjectifier_dik",
    "FutPart": "nounifier_ecek",
    "AorPart": "factative_ir",
    "NarrPart": "pastfactative_miş",
}

INFINITIVE_FEATURES = {
    "Inf": AMBIGUOUS_VNOUN,
    "Inf1": "infinitive_mek",
    "Inf2": "infinitive_me",
    "Inf3": "nounifier_iş",
}

N2N_CASE_FEATURES = {
    "Dat": "dative_e",
    "Acc": "accusative_i",
    "Loc": "locative_de",
    "Abl": "ablative_den",
    "Gen": "noun_compound",
    "Ins": "confactuous_le",
    "Equ": "relative_ce",
}

N2N_POSSESSIVE_FEATURES = {
    "P1sg": "possessive_1sg",
    "P2sg": "possessive_2sg",
    "P3sg": "possessive_3sg",
    "P1pl": "possessive_1pl",
    "P2pl": "possessive_2pl",
    "P3pl": "possessive_3pl",
}

N2N_DERIVATIONAL_FEATURES = {
    "Ness": "suitative_lik",
    "With": "compositive_li",
    "Without": "privative_siz",
    "Agt": "actor_ci",
    "Rel": "marking_ki",
    "Ly": "relative_ce",
    "Related": "relative_sel",
    "Dim": "diminutive_cik",
    "JustLike": "approximative_si",
    "As": "relative_ce",
    "AsIf": "adverbial_cesine",
    "Distrib": "counting_er",
    "Dist": "counting_er",
    "FeelLike": "willing_esi",
    "ActOf": "infinitive_me",
    "Adamantly": ["willing_esi", "dative_e"],
}

CONJUGATION_FEATURES = {
    "A1sg": "conjugation_1sg",
    "A2sg": "conjugation_2sg",
    "A1pl": "conjugation_1pl",
    "A2pl": "conjugation_2pl",
    "A3pl": "conjugation_3pl",
}

COPULA_FEATURES = {
    "Past": "pasttense_di",
    "Narr": "copula_mis",
    "Cop": "nounaorist_dir",
    "Cond": "if_se",
    "Pres": None,
}

POSTP_CASE_FEATURES = {
    "PCNom": None,
    "PCAcc": "accusative_i",
    "PCDat": "dative_e",
    "PCAbl": "ablative_den",
    "PCGen": "noun_compound",
    "PCIns": "confactuous_le",
}

NECES_SUFFIXES = ["infinitive_me", "compositive_li"]
ACQUIRE_SUFFIXES = ["applicative_le", "reflexive_in"]
BECOME_SUFFIXES = ["applicative_le", "reflexive_is"]
PROG2_SUFFIXES = ["infinitive_mek", "locative_de"]
SINCE_SUFFIXES = ["since_eli", "nounaorist_dir"]
NOTSTATE_SUFFIXES = ["negative_me", "factative_ir", "suitative_lik"]

UPOS_TO_CC_CATEGORY = {
    "Conj": "conjunction",
    "Postp": "postposition",
    "Adverb": "adverb",
    "Det": "determiner",
    "Interj": "interjection",
    "Pron": "pronoun",
}

SKIP_UPOS = {"Num", "Ques", "?", "Dup"}
POS_TAGS = {
    "Noun", "Verb", "Adj", "Adverb", "Det", "Conj", "Pron", "Postp",
    "Num", "Ques", "Interj", "Punct", "Dup", "?",
}


# =============================================================================
# TRMOR2018 PARSER
# =============================================================================

def parse_analysis(analysis):
    layers = []
    lemma = None

    for raw_layer in analysis.split("^DB"):
        parts = [p for p in raw_layer.split("+") if p]
        if not parts:
            continue

        if raw_layer.startswith("+") or parts[0] in POS_TAGS:
            layer_lemma = None
            upos = parts[0]
            features = parts[1:]
        else:
            layer_lemma = parts[0]
            upos = parts[1] if len(parts) > 1 else "?"
            features = parts[2:]
            if lemma is None:
                lemma = layer_lemma

        layers.append(make_layer(
            upos,
            upos,
            features,
            lemma=layer_lemma,
        ))

    return lemma, layers


def parse_trmor2018(filepath):
    sentences = []
    current = []
    in_sentence = False

    with open(filepath, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("<S "):
                current = []
                in_sentence = True
                continue
            if line.startswith("</S>"):
                sentences.append(current)
                current = []
                in_sentence = False
                continue
            if not in_sentence or not line or line.startswith("<"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            surface = parts[0]
            analysis = parts[1]
            lemma, layers = parse_analysis(analysis)
            if not layers or layers[0]["upos"] == "Punct":
                continue

            current.append(make_word(
                surface,
                lemma or surface,
                layers,
                is_multiword=len(layers) > 1,
                analysis=analysis,
                head_upos=layers[-1]["upos"],
                head_xpos=layers[-1]["xpos"],
            ))

    return sentences


# =============================================================================
# FEATURE -> SUFFIX MAPPING
# =============================================================================

def _append_mapped(out, mapped):
    if mapped is None:
        return
    if isinstance(mapped, list):
        out.extend(mapped)
    else:
        out.append(mapped)


def features_to_suffix_names(word, unmapped_sink):
    suffix_names = []
    unmapped_on_word = []
    has_unmappable = False

    for layer in word["feature_layers"]:
        upos = layer["upos"]
        xpos = layer["xpos"]
        feats = layer["features"]

        is_verb_context = upos == "Verb"
        is_noun_context = upos in ("Noun", "Adj", "Adverb", "Pron", "Det", "Num")
        is_zero_verb = xpos == "Verb" and "Zero" in feats
        is_pronoun = upos == "Pron"

        able_seen = False
        imp_seen = "Imp" in feats

        for feat in feats:
            if feat in ZERO_FEATURES:
                continue

            if imp_seen and feat == "A2sg":
                continue

            if feat == "Able":
                able_seen = True
                suffix_names.append("possibilitative_ebil")
                continue

            if feat == "Neg":
                if able_seen:
                    if suffix_names and suffix_names[-1] == "possibilitative_ebil":
                        suffix_names[-1] = "negative_able"
                    else:
                        suffix_names.append("negative_able")
                else:
                    suffix_names.append("negative_me")
                continue

            if feat in V2V_FEATURES:
                suffix_names.append(V2V_FEATURES[feat])
                continue

            if feat in V2V_COMPOUND_FEATURES:
                suffix_names.append(V2V_COMPOUND_FEATURES[feat])
                continue

            if feat in PARTICIPLE_FEATURES:
                suffix_names.append(PARTICIPLE_FEATURES[feat])
                continue

            if feat in INFINITIVE_FEATURES:
                suffix_names.append(INFINITIVE_FEATURES[feat])
                continue

            if feat in V2N_TENSE_FEATURES:
                if is_zero_verb and feat in COPULA_FEATURES:
                    _append_mapped(suffix_names, COPULA_FEATURES[feat])
                elif is_noun_context and feat in COPULA_FEATURES:
                    _append_mapped(suffix_names, COPULA_FEATURES[feat])
                else:
                    suffix_names.append(V2N_TENSE_FEATURES[feat])
                continue

            if feat == "Cop":
                _append_mapped(suffix_names, COPULA_FEATURES.get(feat))
                continue

            if feat in V2N_GERUND_FEATURES:
                _append_mapped(suffix_names, V2N_GERUND_FEATURES[feat])
                continue

            if feat == "A3pl":
                if is_noun_context or is_pronoun:
                    suffix_names.append("plural_ler")
                elif is_verb_context:
                    suffix_names.append("conjugation_3pl")
                continue

            if feat in N2N_POSSESSIVE_FEATURES:
                suffix_names.append(N2N_POSSESSIVE_FEATURES[feat])
                continue

            if feat in N2N_CASE_FEATURES:
                suffix_names.append(N2N_CASE_FEATURES[feat])
                continue

            if feat in N2N_DERIVATIONAL_FEATURES:
                _append_mapped(suffix_names, N2N_DERIVATIONAL_FEATURES[feat])
                continue

            if feat in CONJUGATION_FEATURES:
                if not is_pronoun:
                    suffix_names.append(CONJUGATION_FEATURES[feat])
                continue

            if feat in POSTP_CASE_FEATURES:
                _append_mapped(suffix_names, POSTP_CASE_FEATURES[feat])
                continue

            if feat == "Neces":
                suffix_names.extend(NECES_SUFFIXES)
                continue

            if feat == "Cond":
                suffix_names.append("if_se")
                continue

            if feat == "Desr":
                suffix_names.append("wish_suffix")
                continue

            if feat == "Opt":
                suffix_names.append("adverbial_e")
                continue

            if feat == "Acquire":
                suffix_names.extend(ACQUIRE_SUFFIXES)
                continue

            if feat == "Become":
                suffix_names.extend(BECOME_SUFFIXES)
                continue

            if feat == "Prog2":
                suffix_names.extend(PROG2_SUFFIXES)
                continue

            if feat == "Since":
                suffix_names.extend(SINCE_SUFFIXES)
                continue

            if feat == "NotState":
                suffix_names.extend(NOTSTATE_SUFFIXES)
                continue

            if feat == "Ord":
                suffix_names.append("ordinal_inci")
                continue

            has_unmappable = True
            unmapped_on_word.append(feat)
            record_unmapped(unmapped_sink, "TRMorFeature", feat, word)

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
    return word["feature_layers"][0]["upos"] in SKIP_UPOS


def closed_class_category(word):
    first_upos = word["feature_layers"][0]["upos"]
    return UPOS_TO_CC_CATEGORY.get(first_upos)


def trmor_context(word):
    return [
        {"upos": layer["upos"], "xpos": layer["xpos"], "features": layer["features"]}
        for layer in word["feature_layers"]
    ]


def adapt_treebank(input_path, output_path, stats_path=None,
                   unmatched_path=None, unmapped_path=None,
                   sentence_diagnostics_path=None):
    return adapt_normalized_treebank(
        input_path,
        output_path,
        parse_sentences=parse_trmor2018,
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
            "Each TRMor2018 feature value here was not mapped to a Savyar "
            "suffix. Add treebank-specific mappings in data/trmor2018_treebank/"
            "treebank_adapter.py."
        ),
        word_context=trmor_context,
        parse_message="Parsing TRMor2018 treebank: {path}",
        parsed_message="Found {count} sentences",
    )


def adapt_gold_test_set(input_path, output_path):
    """Adapt TRMor2018 gold data into the test JSONL only.

    The test set deliberately has no stats/unmatched/diagnostic sidecar
    files; it is just the adapted Savyar JSONL consumed by the training code.
    """
    return adapt_normalized_treebank(
        input_path,
        output_path,
        parse_sentences=parse_trmor2018,
        translate_word=features_to_suffix_names,
        should_skip_word=should_skip_word,
        closed_class_category=closed_class_category,
        include_input_files_in_stats=True,
        include_unmapped_feature_value_count=True,
        unmapped_header=None,
        word_context=trmor_context,
        parse_message="Parsing TRMor2018 gold test treebank: {path}",
        parsed_message="Found {count} test sentences",
        write_unmatched_log=False,
        write_sentence_diagnostics=False,
        write_unmapped_report=False,
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "trmor2018.conllu")
    gold_input_path = os.path.join(base_dir, "trmor2018_gold.conllu")
    gold_output_path = os.path.join(base_dir, "trmor2018_gold_adapted.jsonl")

    adapt_treebank(
        input_path,
        output_path=os.path.join(base_dir, "treebank_adapted.jsonl"),
        stats_path=os.path.join(base_dir, "treebank_adaptation_stats.json"),
        unmatched_path=os.path.join(base_dir, "treebank_adapted_unmatched.jsonl"),
        unmapped_path=os.path.join(base_dir, "unmapped_features.json"),
        sentence_diagnostics_path=os.path.join(base_dir, "treebank_adapted_sentence_diagnostics.jsonl"),
    )

    if os.path.exists(gold_input_path):
        adapt_gold_test_set(gold_input_path, gold_output_path)
