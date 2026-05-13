"""
TRMor2006 Treebank -> Savyar Adapter
===================================

TRMor2006 uses the same TRmorph analysis strings as TRMor2018, but the outer
sentence tags differ:

    <S>    <S>+BSTag
    token  gold_analysis  alternative_analysis...
    </S>   </S>+ESTag

The morphology mapping is reused from TRMor2018. This adapter keeps only the
TRMor2006-specific file reader here.
"""
"""
@InProceedings{yuret-ture:2006:HLT-NAACL06-Main,
  author    = {Yuret, Deniz  and  Ture, Ferhan},
  title     = {Learning Morphological Disambiguation Rules for Turkish},
  booktitle = {Proceedings of the Human Language Technology Conference of the NAACL, Main Conference},
  month     = {June},
  year      = {2006},
  address   = {New York City, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {328--334},
  url       = {http://www.aclweb.org/anthology/N/N06/N06-1042}
}
"""
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from data.treebank_adapter_commons import adapt_normalized_treebank, make_word
from data.trmor2018_treebank.treebank_adapter import (
    closed_class_category,
    features_to_suffix_names,
    parse_analysis,
    should_skip_word as should_skip_trmor2018_word,
    trmor_context,
)


def _is_sentence_start(line):
    first_field = line.split("\t", 1)[0]
    return first_field == "<S>" or first_field.startswith("<S ")


def _is_sentence_end(line):
    first_field = line.split("\t", 1)[0]
    return first_field == "</S>"


def parse_trmor2006(filepath):
    sentences = []
    current = []
    in_sentence = False

    with open(filepath, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if _is_sentence_start(line):
                if current:
                    sentences.append(current)
                current = []
                in_sentence = True
                continue
            if _is_sentence_end(line):
                if in_sentence:
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
            if not layers or layers[0]["upos"] in {"Punct", "Punc"}:
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

    if in_sentence and current:
        sentences.append(current)

    return sentences


def should_skip_word(word):
    return (
        word["feature_layers"][0]["upos"] == "Punc"
        or should_skip_trmor2018_word(word)
    )


def adapt_treebank(input_path, output_path, stats_path=None,
                   unmatched_path=None, unmapped_path=None,
                   sentence_diagnostics_path=None):
    return adapt_normalized_treebank(
        input_path,
        output_path,
        parse_sentences=parse_trmor2006,
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
            "Each TRMor2006 feature value here was not mapped to a Savyar "
            "suffix. Add treebank-specific mappings in data/trmor2006_treebank/"
            "treebank_adapter.py."
        ),
        word_context=trmor_context,
        parse_message="Parsing TRMor2006 treebank: {path}",
        parsed_message="Found {count} sentences",
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "trmor2006.conllu")

    adapt_treebank(
        input_path,
        output_path=os.path.join(base_dir, "treebank_adapted.jsonl"),
        stats_path=os.path.join(base_dir, "treebank_adaptation_stats.json"),
        unmatched_path=os.path.join(base_dir, "treebank_adapted_unmatched.jsonl"),
        unmapped_path=os.path.join(base_dir, "unmapped_features.json"),
        sentence_diagnostics_path=os.path.join(base_dir, "treebank_adapted_sentence_diagnostics.jsonl"),
    )
