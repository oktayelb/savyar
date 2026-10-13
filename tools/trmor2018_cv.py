#!/usr/bin/env python3
import argparse
import csv
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

import app.nlp_pipeline as nlp
import data.trmor2018_treebank.treebank_adapter as trmor
import util.decomposer as sfx
from app.engine import WorkflowEngine
from ml.candidate_model import CandidateDisambiguator, CandidateTrainer
from ml.config import config
from ml.ml_ranking_model import SentenceDisambiguator, Trainer, build_sentence_sequence
from util.words.closed_class import CLOSED_CLASS_LEXEMES

RAW_PATH = REPO_ROOT / "data" / "trmor2018_treebank" / "trmor2018.conllu"
SEEDS = [1, 2, 3, 4, 5]
SUMMARY_METRICS = [
    "acc_all",
    "acc_ambiguous",
    "acc_unambiguous",
    "acc_reachable",
    "acc_sentence",
    "acc_oov",
    "acc_gold_context",
    "acc_punct_counted",
    "coverage",
    "ambiguity_rate",
    "candidates_per_word",
]


def count_raw_tokens(raw_path):
    sentences = []
    in_sentence = False
    tokens = 0
    punct = 0
    with open(raw_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("<S "):
                in_sentence = True
                tokens = 0
                punct = 0
                continue
            if line.startswith("</S>"):
                sentences.append({"tokens": tokens, "punct": punct})
                in_sentence = False
                continue
            if not in_sentence or not line or line.startswith("<"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            tokens += 1
            _lemma, layers = trmor.parse_analysis(parts[1])
            if not layers or layers[0]["upos"] == "Punct":
                punct += 1
    return sentences


def adapt_corpus(adapted_dir):
    adapted_path = adapted_dir / "trmor2018_adapted.jsonl"
    if not adapted_path.exists():
        adapted_dir.mkdir(parents=True, exist_ok=True)
        trmor.adapt_treebank(
            str(RAW_PATH),
            output_path=str(adapted_path),
            stats_path=str(adapted_dir / "adaptation_stats.json"),
            unmatched_path=str(adapted_dir / "unmatched.jsonl"),
            unmapped_path=str(adapted_dir / "unmapped_features.json"),
            sentence_diagnostics_path=str(adapted_dir / "sentence_diagnostics.jsonl"),
        )
    with open(adapted_path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_masked(word):
    if trmor.should_skip_word(word):
        return True
    _expected, _unmapped, has_unmappable = trmor.features_to_suffix_names(word, {})
    return has_unmappable


def load_sentences(out_dir):
    adapted = adapt_corpus(out_dir / "adapted")
    parsed = trmor.parse_trmor2018(str(RAW_PATH))
    raw_counts = count_raw_tokens(RAW_PATH)
    if len(raw_counts) != len(parsed):
        raise RuntimeError(f"raw sentence count {len(raw_counts)} != parsed {len(parsed)}")

    records = []
    next_adapted = 0
    for index, words in enumerate(parsed):
        record = {
            "index": index,
            "entry": None,
            "masked": [],
            "tokens": raw_counts[index]["tokens"],
            "punct": raw_counts[index]["punct"],
        }
        if words:
            text = " ".join(word["surface"] for word in words)
            if next_adapted < len(adapted) and adapted[next_adapted]["original_sentence"] == text:
                entry = adapted[next_adapted]
                next_adapted += 1
                if len(entry["words"]) != len(words):
                    raise RuntimeError(f"sentence {index}: {len(entry['words'])} adapted words != {len(words)} parsed")
                record["entry"] = entry
                record["masked"] = [is_masked(word) for word in words]
        records.append(record)
    if next_adapted != len(adapted):
        raise RuntimeError(f"aligned {next_adapted} of {len(adapted)} adapted sentences")
    return records


def split_indices(count, seed):
    order = list(range(count))
    random.Random(seed).shuffle(order)
    train_end = int(count * 0.8)
    dev_end = int(count * 0.9)
    return order[:train_end], order[train_end:dev_end], order[dev_end:]


def entries_for(records, indices):
    return [records[i]["entry"] for i in indices if records[i]["entry"] is not None]


def build_trainer(engine, train_path):
    model = SentenceDisambiguator(
        suffix_vocab_size=len(sfx.ALL_SUFFIXES),
        closed_class_vocab_size=len(CLOSED_CLASS_LEXEMES),
        device=engine.device,
    )
    return Trainer(model=model, path=str(train_path), device=engine.device)


def train_split(engine, train_entries, dev_entries, seed, seed_dir):
    random.seed(seed)
    torch.manual_seed(seed)
    train_path = seed_dir / "training_checkpoint.pt"
    if train_path.exists():
        train_path.unlink()
    trainer = build_trainer(engine, train_path)
    engine.trainer = trainer

    started = time.monotonic()
    train_seqs, train_words, train_skipped = engine._entries_to_sequences(
        train_entries, log_progress=True, label=f"Seed {seed} train preprocessing"
    )
    dev_seqs, dev_words, dev_skipped = engine._entries_to_sequences(
        dev_entries, log_progress=True, label=f"Seed {seed} dev preprocessing"
    )
    preprocessing_seconds = time.monotonic() - started

    history = []
    best = {"rank_acc": -1.0, "epoch": 0, "state": None}
    plain_validate = trainer.validate

    def validate_and_keep_best(val_seqs, batch_size=64):
        stats = plain_validate(val_seqs, batch_size=batch_size)
        epoch = len(history) + 1
        history.append({"epoch": epoch, "dev_rank_loss": stats["loss"], "dev_rank_acc": stats["rank_acc"]})
        if stats["rank_acc"] > best["rank_acc"]:
            best["rank_acc"] = stats["rank_acc"]
            best["epoch"] = epoch
            best["state"] = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
        return stats

    def train_sequences_for_epoch(epoch):
        sequences, _words, _skipped = engine._entries_to_sequences(
            train_entries, log_progress=True, label=f"Seed {seed} epoch {epoch + 1} negatives", epoch=epoch
        )
        return sequences

    resample = train_sequences_for_epoch if config.resample_negatives else None
    trainer.validate = validate_and_keep_best
    started = time.monotonic()
    trainer.train_bulk(train_seqs, validation_seqs=dev_seqs, resample=resample)
    training_seconds = time.monotonic() - started
    trainer.validate = plain_validate

    trainer.model.load_state_dict(best["state"])
    torch.save(trainer.model.state_dict(), seed_dir / "best_model.pt")
    if train_path.exists():
        train_path.unlink()

    info = {
        "train_candidate_sets": len(train_seqs),
        "train_words": train_words,
        "train_skipped": train_skipped,
        "dev_candidate_sets": len(dev_seqs),
        "dev_words": dev_words,
        "dev_skipped": dev_skipped,
        "epochs": config.bulk_epochs,
        "dropout": config.dropout,
        "weight_decay": config.weight_decay,
        "sum_token_scores": config.sum_token_scores,
        "resample_negatives": config.resample_negatives,
        "best_epoch": best["epoch"],
        "best_dev_rank_acc": best["rank_acc"],
        "history": history,
        "preprocessing_seconds": preprocessing_seconds,
        "training_seconds": training_seconds,
    }
    return trainer, info


def candidate_sentences(engine, entries):
    sentences = []
    skipped = 0
    for entry in entries:
        try:
            parts = engine._candidate_parts_from_word_entries(entry["words"])
        except ValueError:
            parts = None
        if parts is None:
            skipped += 1
            continue
        _gold_chains, candidate_lists, gold_indices, _word_count = parts
        sentences.append({
            "candidates": candidate_lists,
            "gold": gold_indices,
            "final_token": nlp.final_punctuation_token(entry.get("final_punctuation")),
        })
    return sentences, skipped


def build_candidate_trainer(engine):
    model = CandidateDisambiguator(
        suffix_vocab_size=len(sfx.ALL_SUFFIXES),
        closed_class_vocab_size=len(CLOSED_CLASS_LEXEMES),
        device=engine.device,
    )
    return CandidateTrainer(model, device=engine.device)


def train_candidate_split(engine, train_entries, dev_entries, seed, seed_dir):
    random.seed(seed)
    torch.manual_seed(seed)
    started = time.monotonic()
    train_sentences, train_skipped = candidate_sentences(engine, train_entries)
    dev_sentences, dev_skipped = candidate_sentences(engine, dev_entries)
    preprocessing_seconds = time.monotonic() - started

    trainer = build_candidate_trainer(engine)
    started = time.monotonic()
    history = trainer.train(train_sentences, dev_sentences, epochs=config.bulk_epochs)
    training_seconds = time.monotonic() - started
    torch.save(trainer.model.state_dict(), seed_dir / "best_model.pt")

    best = max(history, key=lambda row: row["dev_ambiguous_accuracy"])
    info = {
        "model": "candidate",
        "train_sentences": len(train_sentences),
        "train_skipped": train_skipped,
        "dev_sentences": len(dev_sentences),
        "dev_skipped": dev_skipped,
        "epochs": config.bulk_epochs,
        "dropout": config.dropout,
        "weight_decay": config.weight_decay,
        "best_epoch": best["epoch"],
        "best_dev_ambiguous_accuracy": best["dev_ambiguous_accuracy"],
        "history": history,
        "preprocessing_seconds": preprocessing_seconds,
        "training_seconds": training_seconds,
    }
    return trainer, info


def load_trained_candidates(engine, seed_dir):
    trainer = build_candidate_trainer(engine)
    trainer.model.load_state_dict(torch.load(seed_dir / "best_model.pt", map_location=engine.device))
    return trainer


def load_trained(engine, seed_dir):
    trainer = build_trainer(engine, seed_dir / "unused_checkpoint.pt")
    trainer.model.load_state_dict(torch.load(seed_dir / "best_model.pt", map_location=engine.device))
    engine.trainer = trainer
    return trainer


def fallback_chain(word):
    return [nlp.root_token(word, "noun")]


def beam_decode(trainer, candidate_lists, beam_width, final_token=None):
    max_len = int(config.max_sequence_length)
    beams = [[]]
    last = len(candidate_lists) - 1
    for position, candidates in enumerate(candidate_lists):
        expanded = [beam + [k] for beam in beams for k in range(len(candidates))]
        if len(expanded) <= beam_width and position < last:
            beams = expanded
            continue
        if len(expanded) == 1:
            beams = expanded
            continue
        sequences = []
        for picks in expanded:
            chains = [candidate_lists[i][k] for i, k in enumerate(picks)]
            sequence = build_sentence_sequence(chains, final_token)
            if len(sequence[0]) > max_len:
                return None
            sequences.append(sequence)
        scores = trainer.score_flat_sequences(sequences)
        ranked = sorted(range(len(expanded)), key=lambda i: scores[i], reverse=True)
        beams = [expanded[i] for i in ranked[:beam_width]]
    return beams[0]


def evaluate_split(engine, trainer, test_records, train_surfaces, beam_width, candidate_model=False):
    word_rows = []
    sentence_total = 0
    sentence_correct = 0
    scoring_failures = 0
    punct_tokens = 0
    masked_words = 0

    test_entries = [record["entry"] for record in test_records if record["entry"] is not None]
    if candidate_model:
        gold_context_iter = None
    else:
        gold_context_iter = iter(engine._score_test_words(test_entries))

    for record in test_records:
        punct_tokens += record["punct"]
        entry = record["entry"]
        if entry is None:
            continue
        words = entry["words"]
        analyses = [nlp.analyze_word(word["word"], word.get("surface")) for word in words]
        candidate_lists = [a["encoded_chains"] or [fallback_chain(a["word"])] for a in analyses]
        final_token = nlp.final_punctuation_token(entry.get("final_punctuation"))
        if candidate_model:
            picks = trainer.predict([{"candidates": candidate_lists, "final_token": final_token}])[0]
        else:
            picks = beam_decode(trainer, candidate_lists, beam_width, final_token)
        if picks is None:
            scoring_failures += 1

        all_correct = True
        evaluated_in_sentence = 0
        for position, (word, analysis) in enumerate(zip(words, analyses)):
            gold_context = None if gold_context_iter is None else next(gold_context_iter)
            if record["masked"][position]:
                masked_words += 1
                continue
            evaluated_in_sentence += 1
            gold_set = set(nlp.match_decompositions([word], analysis["decomps"])) if analysis["decomps"] else set()
            pick = None if picks is None else picks[position]
            correct = pick is not None and pick in gold_set
            chains = analysis["encoded_chains"]
            gold_signatures = {tuple(tuple(t) for t in chains[i]) for i in gold_set}
            indistinguishable = any(
                tuple(tuple(t) for t in chain) in gold_signatures
                for i, chain in enumerate(chains)
                if i not in gold_set
            )
            if gold_context is None:
                gold_context_correct = correct
            else:
                gold_context_correct = (
                    gold_context["status"] in ("single", "ambiguous")
                    and gold_context["pred_idx"] in gold_set
                )
            word_rows.append({
                "sentence": record["index"],
                "word": word["word"],
                "candidates": len(analysis["decomps"]),
                "reachable": bool(gold_set),
                "correct": correct,
                "oov": word["word"] not in train_surfaces,
                "indistinguishable": indistinguishable,
                "gold_context_correct": gold_context_correct,
                "gold": sorted(gold_set),
                "pick": pick,
            })
            all_correct = all_correct and correct
        if evaluated_in_sentence:
            sentence_total += 1
            sentence_correct += all_correct
    return word_rows, {
        "sentences": sentence_total,
        "sentences_correct": sentence_correct,
        "scoring_failures": scoring_failures,
        "punct_tokens": punct_tokens,
        "masked_words": masked_words,
    }


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def summarise(word_rows, counts):
    total = len(word_rows)
    correct = sum(row["correct"] for row in word_rows)
    ambiguous = [row for row in word_rows if row["candidates"] > 1]
    unambiguous = [row for row in word_rows if row["candidates"] <= 1]
    reachable = [row for row in word_rows if row["reachable"]]
    oov = [row for row in word_rows if row["oov"]]
    by_candidates = Counter()
    by_candidates_correct = Counter()
    for row in word_rows:
        bucket = min(row["candidates"], 6)
        by_candidates[bucket] += 1
        by_candidates_correct[bucket] += row["correct"]
    return {
        "words": total,
        "correct": correct,
        "acc_all": ratio(correct, total),
        "ambiguous_words": len(ambiguous),
        "acc_ambiguous": ratio(sum(r["correct"] for r in ambiguous), len(ambiguous)),
        "unambiguous_words": len(unambiguous),
        "acc_unambiguous": ratio(sum(r["correct"] for r in unambiguous), len(unambiguous)),
        "reachable_words": len(reachable),
        "acc_reachable": ratio(sum(r["correct"] for r in reachable), len(reachable)),
        "acc_sentence": ratio(counts["sentences_correct"], counts["sentences"]),
        "oov_words": len(oov),
        "acc_oov": ratio(sum(r["correct"] for r in oov), len(oov)),
        "acc_gold_context": ratio(sum(r["gold_context_correct"] for r in word_rows), total),
        "acc_punct_counted": ratio(correct + counts["punct_tokens"], total + counts["punct_tokens"]),
        "coverage": ratio(len(reachable), total),
        "ambiguity_rate": ratio(len(ambiguous), total),
        "candidates_per_word": ratio(sum(r["candidates"] for r in word_rows), total),
        "indistinguishable_words": sum(r["indistinguishable"] for r in word_rows),
        "indistinguishable_errors": sum(r["indistinguishable"] and not r["correct"] for r in word_rows),
        "by_candidates": {
            str(bucket): {"words": by_candidates[bucket], "correct": by_candidates_correct[bucket]}
            for bucket in sorted(by_candidates)
        },
        **counts,
    }


def run_seed(engine, records, seed, out_dir, beam_width, candidate_model=False):
    seed_dir = out_dir / f"seed{seed}"
    result_path = seed_dir / "result.json"
    if result_path.exists():
        print(f"Seed {seed}: result exists, skipping", flush=True)
        with open(result_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    seed_dir.mkdir(parents=True, exist_ok=True)

    train_idx, dev_idx, test_idx = split_indices(len(records), seed)
    train_entries = entries_for(records, train_idx)
    dev_entries = entries_for(records, dev_idx)
    test_records = [records[i] for i in test_idx]
    print(
        f"Seed {seed}: {len(train_idx)}/{len(dev_idx)}/{len(test_idx)} sentences, "
        f"{len(train_entries)}/{len(dev_entries)}/{sum(r['entry'] is not None for r in test_records)} adapted",
        flush=True,
    )

    if (seed_dir / "best_model.pt").exists():
        if candidate_model:
            trainer = load_trained_candidates(engine, seed_dir)
        else:
            trainer = load_trained(engine, seed_dir)
        training_info = {"note": "loaded existing best_model.pt"}
    elif candidate_model:
        trainer, training_info = train_candidate_split(engine, train_entries, dev_entries, seed, seed_dir)
    else:
        trainer, training_info = train_split(engine, train_entries, dev_entries, seed, seed_dir)

    train_surfaces = {word["word"] for entry in train_entries for word in entry["words"]}
    started = time.monotonic()
    word_rows, counts = evaluate_split(engine, trainer, test_records, train_surfaces, beam_width, candidate_model)
    result = summarise(word_rows, counts)
    result["seed"] = seed
    result["evaluation_seconds"] = time.monotonic() - started
    result["training"] = training_info
    result["split_sentences"] = {"train": len(train_idx), "dev": len(dev_idx), "test": len(test_idx)}

    with open(seed_dir / "test_words.jsonl", "w", encoding="utf-8") as handle:
        for row in word_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(
        f"Seed {seed}: all {result['acc_all']:.4f} ambiguous {result['acc_ambiguous']:.4f} "
        f"reachable {result['acc_reachable']:.4f} sentence {result['acc_sentence']:.4f} "
        f"gold-context {result['acc_gold_context']:.4f}",
        flush=True,
    )

    del trainer
    engine.trainer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_summary(results, out_dir):
    summary = {}
    for metric in SUMMARY_METRICS:
        values = [result[metric] for result in results]
        summary[metric] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "per_seed": values,
        }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std"] + [f"seed{r['seed']}" for r in results])
        for metric in SUMMARY_METRICS:
            row = summary[metric]
            writer.writerow([metric, f"{row['mean']:.4f}", f"{row['std']:.4f}"] + [f"{v:.4f}" for v in row["per_seed"]])
    print("\n=== TrMor2018, mean ± std over seeds ===", flush=True)
    for metric in SUMMARY_METRICS:
        row = summary[metric]
        scale = 1 if metric == "candidates_per_word" else 100
        print(f"{metric:22s} {row['mean'] * scale:6.2f} ± {row['std'] * scale:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO_ROOT / "experiments" / "trmor2018_cv"))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--beam", type=int, default=50)
    parser.add_argument("--no-punctuation", action="store_true")
    parser.add_argument("--no-apostrophe", action="store_true")
    parser.add_argument("--mean-pooling", action="store_true")
    parser.add_argument("--fixed-negatives", action="store_true")
    parser.add_argument("--candidate-model", action="store_true")
    parser.add_argument("--epochs", type=int, default=config.bulk_epochs)
    args = parser.parse_args()
    config.use_final_punctuation = not args.no_punctuation
    config.use_apostrophe_boundary = not args.no_apostrophe
    config.sum_token_scores = not args.mean_pooling
    config.resample_negatives = not args.fixed_negatives
    config.bulk_epochs = args.epochs

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_sentences(out_dir)
    print(
        f"TrMor2018: {len(records)} sentences, {sum(r['entry'] is not None for r in records)} adapted, "
        f"{sum(r['tokens'] for r in records)} tokens, {sum(r['punct'] for r in records)} punctuation",
        flush=True,
    )

    engine = WorkflowEngine()
    engine.trainer = None
    engine.model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = [run_seed(engine, records, seed, out_dir, args.beam, args.candidate_model) for seed in args.seeds]
    write_summary(results, out_dir)


if __name__ == "__main__":
    main()
