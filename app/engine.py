"""Orchestrator and ML Logic. 
Combines the Workflow Engine, Sequence Matching, and K-Fold Cross Validation.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import math
import random
import tempfile
import shutil
import time
import torch
from typing import List, Optional, Tuple, Dict, Any, Callable, Sequence

import util.decomposer as sfx
import util.word_methods as wrd
from util.word_methods import tr_lower
from app.data_manager import DataManager
import app.nlp_pipeline as nlp
from ml.ml_ranking_model import SentenceDisambiguator, Trainer, build_sentence_sequence, resolve_torch_device
from ml.config import config
from util.words.closed_class import CLOSED_CLASS_TOKEN_SPECS

STATIC_PREPROCESS_CACHE_VERSION = 4

# --------------------------------------------------------------------------- #
# K-Fold Cross Validation Logic
# --------------------------------------------------------------------------- #
_T_CRIT_95: Dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,  5: 2.571,
    6:  2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042,
}

def _t_crit_95(df: int) -> float:
    if df <= 0: return float("nan")
    if df in _T_CRIT_95: return _T_CRIT_95[df]
    if df > 30: return 1.96
    for k in sorted(_T_CRIT_95):
        if k >= df: return _T_CRIT_95[k]
    return 1.96

FoldRunner = Callable[[List[Any], List[Any], int], Dict[str, float]]

def k_fold_split(n: int, k: int, seed: int = 42) -> List[List[int]]:
    if k <= 0: raise ValueError("k must be >= 1")
    if n < k: raise ValueError(f"Cannot split {n} items into {k} folds (n < k).")
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    folds: List[List[int]] = [[] for _ in range(k)]
    for i, idx in enumerate(indices): folds[i % k].append(idx)
    return folds

def run_k_fold_cv(dataset: Sequence[Any], k: int, fold_runner: FoldRunner, seed: int = 42, verbose: bool = True) -> Dict[str, Any]:
    n = len(dataset)
    folds = k_fold_split(n, k, seed=seed)
    per_fold: List[Dict[str, float]] = []
    
    for fi, val_indices in enumerate(folds):
        val_set = set(val_indices)
        train_items = [dataset[i] for i in range(n) if i not in val_set]
        val_items = [dataset[i] for i in val_indices]
        if verbose:
            print(f"\n=== Fold {fi + 1}/{k}:  train={len(train_items)}  val={len(val_items)} ===")
        stats = fold_runner(train_items, val_items, fi)
        per_fold.append(stats)
        if verbose:
            cells = [f"{m}={v:.4f}" for m, v in stats.items() if isinstance(v, (int, float))]
            print(f"   Fold {fi + 1} metrics: " + " | ".join(cells))
            
    summary = _aggregate(per_fold, k)
    if verbose: _print_summary(summary, k)
    return {"folds": per_fold, "summary": summary, "k": k, "n": n}

def _aggregate(per_fold: List[Dict[str, float]], k: int) -> Dict[str, Dict[str, float]]:
    if not per_fold: return {}
    names = sorted({m for d in per_fold for m, v in d.items() if isinstance(v, (int, float))})
    t = _t_crit_95(max(k - 1, 1))
    out: Dict[str, Dict[str, float]] = {}
    for name in names:
        values = [d[name] for d in per_fold if name in d and isinstance(d[name], (int, float))]
        if not values: continue
        mean = sum(values) / len(values)
        if len(values) > 1:
            var  = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std  = math.sqrt(var)
            half = t * std / math.sqrt(len(values))
        else:
            std, half = 0.0, 0.0
        out[name] = {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half, "half_width": half, "n": float(len(values))}
    return out

def _print_summary(summary: Dict[str, Dict[str, float]], k: int) -> None:
    bar = "=" * 78
    print("\n" + bar)
    print(f"  {k}-FOLD CV SUMMARY   (95% CI via t-distribution, df={k - 1})")
    print(bar)
    if not summary:
        print("  (no numeric metrics were returned by fold_runner)")
        print(bar)
        return
    name_w = max(len(n) for n in summary)
    for name, s in summary.items():
        print(f"  {name:<{name_w}}  mean={s['mean']:+.4f}  ± {s['half_width']:.4f}   "
              f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]   std={s['std']:.4f}  n={int(s['n'])}")
    print(bar)


# --------------------------------------------------------------------------- #
# Sequence Matcher Logic
# --------------------------------------------------------------------------- #
def find_matching_combinations(word_data: List[Dict], target_str: str, trainer) -> Tuple[List[Dict], str, int]:
    matches = []
    furthest_match_text = ""
    furthest_word_idx = 0
    clean_target = " ".join(target_str.replace("(ø)", "").split())
    target_tokens = clean_target.split()
    
    def is_valid_prefix(current_tokens: List[str], t_tokens: List[str]) -> bool:
        if not t_tokens or not current_tokens: return True
        min_len = min(len(current_tokens), len(t_tokens))
        for i in range(min_len - 1):
            if current_tokens[i] != t_tokens[i]: return False
        idx = min_len - 1
        if len(t_tokens) > len(current_tokens):
            if current_tokens[idx] != t_tokens[idx]: return False
        else:
            if not current_tokens[idx].startswith(t_tokens[idx]): return False
        return True

    def dfs(word_idx, current_indices, current_text_parts):
        nonlocal furthest_match_text, furthest_word_idx
        full_text = " ".join(current_text_parts).strip()
        clean_full = " ".join(full_text.replace("(ø)", "").split())
        current_tokens = clean_full.split()
        
        if word_idx > furthest_word_idx:
            furthest_word_idx = word_idx
            furthest_match_text = full_text
        elif word_idx == furthest_word_idx and len(current_tokens) > len(" ".join(furthest_match_text.replace("(ø)", "").split()).split()):
            furthest_match_text = full_text

        if word_idx == len(word_data):
            if len(target_tokens) > len(current_tokens): return
            if is_valid_prefix(current_tokens, target_tokens):
                matches.append((current_indices, full_text, current_text_parts))
            return
            
        for d_idx, t_str in enumerate(word_data[word_idx]['typing_strings']):
            next_parts = current_text_parts + [t_str]
            next_text = " ".join(next_parts).strip()
            clean_next = " ".join(next_text.replace("(ø)", "").split())
            next_tokens = clean_next.split()
            if is_valid_prefix(next_tokens, target_tokens):
                dfs(word_idx + 1, current_indices + [d_idx], next_parts)

    dfs(0, [], [])
    
    scored_matches = []
    trainer.model.eval()
    with torch.no_grad():
        for indices, full_text, parts in matches:
            sentence_chains = [word_data[w_idx]['encoded_chains'][d_idx] for w_idx, d_idx in enumerate(indices)]
            total_score = trainer.score_sentence_chains(sentence_chains)
            scored_matches.append({'score': total_score, 'combo_indices': indices, 'text': full_text, 'parts': parts})
            
    scored_matches.sort(key=lambda x: x['score'], reverse=True)
    return scored_matches, furthest_match_text, furthest_word_idx

def get_top_sentence_predictions(word_data: List[Dict], trainer, top_k: int = 10, beam_width: int = 50) -> List[Dict]:
    beams = [{'score': 0.0, 'combo_indices': [], 'parts': []}]
    trainer.model.eval()
    with torch.no_grad():
        for w_idx, wd in enumerate(word_data):
            new_beams = []
            for beam in beams:
                for d_idx in range(len(wd['decomps'])):
                    new_indices = beam['combo_indices'] + [d_idx]
                    new_parts = beam['parts'] + [wd['typing_strings'][d_idx]]
                    sentence_chains = [word_data[i]['encoded_chains'][idx] for i, idx in enumerate(new_indices)]
                    score = trainer.score_sentence_chains(sentence_chains)
                    new_beams.append({'score': score, 'combo_indices': new_indices, 'parts': new_parts, 'text': " ".join(new_parts).strip()})
            new_beams.sort(key=lambda x: x['score'], reverse=True)
            beams = new_beams[:beam_width]
    return beams[:top_k]


# --------------------------------------------------------------------------- #
# Workflow Engine
# --------------------------------------------------------------------------- #
class WorkflowEngine:
    def __init__(self):
        self.data_manager = DataManager()
        self.device = resolve_torch_device()
        self.model = SentenceDisambiguator(
            suffix_vocab_size=len(sfx.ALL_SUFFIXES),
            closed_class_vocab_size=len(CLOSED_CLASS_TOKEN_SPECS),
            device=self.device,
        )
        self.trainer = Trainer(model=self.model, device=self.device)
        self.training_count = self.data_manager.load_training_count()
        self.decomp_cache = {}

    def get_decompositions(self, word: str) -> List[Tuple]:
        word = word.replace("'", "")
        if word not in self.decomp_cache:
            self.decomp_cache[word] = sfx.decompose_with_cc(word)
        return self.decomp_cache[word]

    def save(self):
        self.trainer.save_checkpoint()
        self.data_manager.save_training_count(self.training_count)

    def _save_final_suffix_metrics(self) -> None:
        report: Dict[str, Any] = {}
        if self.trainer.last_train_stats:
            report["training"] = self.trainer.last_train_stats
        validation = self.trainer.last_validation_report or self.trainer.last_validation_stats
        if validation:
            report["validation"] = validation
        if report:
            self.data_manager.save_final_suffix_metrics(report)

    def analyze_word(self, word: str) -> Optional[Dict[str, Any]]:
        analysis = nlp.analyze_word(word, include_closed_class=True)
        if not analysis['decomps']:
            return None
        if self.training_count > 0:
            nlp.score_and_sort(analysis, self.trainer)
        return analysis

    def analyze_sentence_with_failures(self, words: List[str]) -> Tuple[Optional[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        if not words:
            return None, []
        analyses = nlp.analyze_words(words, include_closed_class=True)
        failures = [
            {'index': i + 1, 'word': a['word']}
            for i, a in enumerate(analyses)
            if not a['decomps']
        ]
        if failures:
            return None, failures
        return analyses, []

    def analyze_sentence(self, words: List[str]) -> Optional[List[Dict[str, Any]]]:
        analyses, _failures = self.analyze_sentence_with_failures(words)
        return analyses

    def commit_word(self, analysis: Dict[str, Any], selected_indices: List[int]) -> Tuple[float, List[str]]:
        word = analysis['word']
        word_lower = tr_lower(word)
        correct_decomps = [analysis['decomps'][i] for i in selected_indices]
        correct_encoded = [analysis['encoded_chains'][i] for i in selected_indices]

        log_entries: List[Dict[str, Any]] = []
        for decomp in correct_decomps:
            root, _, _, final_pos = decomp
            suffix_info = nlp.build_suffix_log_info(word, decomp)
            log_entries.append({'word': word, 'root': root, 'suffixes': suffix_info, 'final_pos': final_pos})
        self.data_manager.log_decompositions(log_entries)

        deleted_messages: List[str] = []
        for decomp in correct_decomps:
            root = tr_lower(decomp[0])
            if root == word_lower: continue
            if self.data_manager.delete(word_lower):
                deleted_messages.append(f"Deleted '{word}' (root '{root}' exists)")
                sfx.decompose.cache_clear()
                self.decomp_cache.pop(word_lower, None)
            infinitive_form = wrd.infinitive_form(root)
            if infinitive_form and self.data_manager.delete(infinitive_form):
                deleted_messages.append(f"Deleted infinitive '{infinitive_form}'")
                sfx.decompose.cache_clear()
                self.decomp_cache.pop(infinitive_form, None)

        loss = 0.0
        correct_signatures = {tuple(tok[0] for tok in encoded) for encoded in correct_encoded}
        for encoded in correct_encoded:
            negatives = [
                [candidate]
                for candidate in analysis['encoded_chains']
                if tuple(tok[0] for tok in candidate) not in correct_signatures
            ][:config.max_negative_candidates]
            loss = self.trainer.train_sentence([encoded], negative_word_chains=negatives)

        self.training_count += 1
        if self.training_count % self.trainer.checkpoint_frequency == 0:
            self.save()
        return loss, deleted_messages

    def evaluate_sentence_target(self, word_data: List[Dict], target_str: str) -> Tuple[List[Dict], str, int]:
        return find_matching_combinations(word_data, target_str, self.trainer)

    def commit_sentence_training(self, sentence: str, words: List[str], word_data: List[Dict], correct_combo: List[int]) -> float:
        confirmed_chains = []
        log_entries = []

        for w_idx, correct_d_idx in enumerate(correct_combo):
            wd = word_data[w_idx]
            word = wd['word']
            decomps = wd['decomps']
            typing_str = wd['typing_strings'][correct_d_idx]
            confirmed_chain = wd['encoded_chains'][correct_d_idx]
            confirmed_chains.append(confirmed_chain)
            root, _, _, final_pos = decomps[correct_d_idx]
            suffix_info = nlp.build_suffix_log_info(word, decomps[correct_d_idx])
            log_entries.append({
                'word': word, 'morphology_string': typing_str,
                'root': root, 'suffixes': suffix_info, 'final_pos': final_pos,
            })

        self.data_manager.log_sentence_decompositions(log_entries, sentence)
        candidate_lists = [wd['encoded_chains'] for wd in word_data]
        negatives = self._single_substitution_negatives(confirmed_chains, candidate_lists, correct_combo)
        loss = self.trainer.train_sentence(confirmed_chains, negative_word_chains=negatives)

        self.training_count += len(confirmed_chains)
        if self.training_count % self.trainer.checkpoint_frequency == 0:
            self.save()
        return loss

    def evaluate_word(self, word: str) -> Optional[Dict]:
        analysis = nlp.analyze_word(word, include_closed_class=True)
        if not analysis['decomps']: return None
        scores = nlp.score_and_sort(analysis, self.trainer)
        if scores is None and len(analysis['decomps']) > 1: return None
        return analysis['vms'][0]

    def prepare_sentence_training(self, sentence: str) -> Optional[List[Dict]]:
        return self.analyze_sentence(sentence.strip().split())

    def _single_substitution_negatives(
        self, gold_chains: List[List], candidate_lists: List[List[List]], gold_indices: List[int], limit: Optional[int] = None,
    ) -> List[List[List]]:
        if limit is None: limit = config.max_negative_candidates
        negatives: List[List[List]] = []
        seen = set()
        for word_idx, candidates in enumerate(candidate_lists):
            gold_idx = gold_indices[word_idx]
            for cand_idx, candidate in enumerate(candidates):
                if cand_idx == gold_idx: continue
                neg = list(gold_chains)
                neg[word_idx] = candidate
                signature = tuple(tuple(tok[0] for tok in chain) for chain in neg)
                if signature in seen: continue
                seen.add(signature)
                negatives.append(neg)
                if len(negatives) >= limit: return negatives
        return negatives

    def _candidate_parts_from_word_entries(self, word_entries: List[Dict]) -> Optional[Tuple[List, List, List, int]]:
        gold_chains = []
        candidate_lists = []
        gold_indices = []
        for word_entry in word_entries:
            sfx_dicts = word_entry.get('suffixes', [])
            if not sfx_dicts: continue
            encoded_gold = nlp.encode_suffix_names(sfx_dicts)
            if not encoded_gold: continue
            try:
                word_analysis = nlp.analyze_word(word_entry['word'], include_closed_class=True)
                matched = nlp.match_decompositions([word_entry], word_analysis['decomps'])
            except Exception:
                matched = []
                word_analysis = None
            if matched and word_analysis is not None:
                gold_idx = matched[0]
                gold_chain = word_analysis['encoded_chains'][gold_idx]
                candidates = word_analysis['encoded_chains']
            else:
                gold_idx = 0
                gold_chain = encoded_gold
                candidates = [encoded_gold]
            gold_chains.append(gold_chain)
            candidate_lists.append(candidates)
            gold_indices.append(gold_idx)
        if not gold_chains: return None
        return gold_chains, candidate_lists, gold_indices, len(gold_chains)

    def _select_dynamic_negatives(self, scored_negatives: List[Tuple[float, Any]], rng: random.Random) -> List[Any]:
        if not scored_negatives: return []
        ranked = [item for _, item in sorted(scored_negatives, key=lambda x: x[0], reverse=True)]
        max_neg = max(0, int(config.max_negative_candidates))
        hard_count = min(int(config.hard_negative_count), max_neg, len(ranked))
        selected = ranked[:hard_count]
        selected_ids = {id(item) for item in selected}

        remaining_slots = max_neg - len(selected)
        easy_count = min(int(config.easy_negative_count), remaining_slots, max(0, len(ranked) - len(selected)))
        if easy_count:
            easy_pool = [item for item in reversed(ranked) if id(item) not in selected_ids]
            easy = easy_pool[:easy_count]
            selected.extend(easy)
            selected_ids.update(id(item) for item in easy)
            remaining_slots = max_neg - len(selected)

        medium_count = min(int(config.medium_negative_count), remaining_slots)
        if medium_count:
            medium_pool = [item for item in ranked[hard_count:] if id(item) not in selected_ids]
            medium = rng.sample(medium_pool, medium_count) if len(medium_pool) > medium_count else medium_pool
            selected.extend(medium)
            selected_ids.update(id(item) for item in medium)

        if len(selected) < max_neg:
            for item in ranked:
                if id(item) in selected_ids: continue
                selected.append(item)
                selected_ids.add(id(item))
                if len(selected) >= max_neg: break
        return selected

    def _dynamic_candidate_set_from_word_entries(self, word_entries: List[Dict], rng: random.Random) -> Optional[Tuple[List, int]]:
        parts = self._candidate_parts_from_word_entries(word_entries)
        if parts is None: return None
        gold_chains, candidate_lists, gold_indices, word_count = parts
        gold_seq = build_sentence_sequence(gold_chains)
        if len(gold_seq[0]) > int(config.max_sequence_length): return None
        negatives = self._single_substitution_negatives(gold_chains, candidate_lists, gold_indices, limit=config.dynamic_negative_pool_size)
        if not negatives: return None
        negative_seqs = [
            seq for seq in (build_sentence_sequence(neg) for neg in negatives)
            if len(seq[0]) <= int(config.max_sequence_length)
        ]
        if not negative_seqs: return None
        scores = self.trainer.score_flat_sequences(negative_seqs)
        selected = self._select_dynamic_negatives(list(zip(scores, negative_seqs)), rng)
        if not selected: return None
        return [gold_seq] + selected, word_count

    def _candidate_set_from_word_entries(self, word_entries: List[Dict]) -> Optional[Tuple[List, int]]:
        parts = self._candidate_parts_from_word_entries(word_entries)
        if parts is None: return None
        gold_chains, candidate_lists, gold_indices, word_count = parts
        gold_seq = build_sentence_sequence(gold_chains)
        negatives = self._single_substitution_negatives(gold_chains, candidate_lists, gold_indices)
        candidate_set = [gold_seq] + [build_sentence_sequence(neg) for neg in negatives]
        return candidate_set, word_count

    def _candidate_set_fits_model(self, candidate_set: List[Any]) -> bool:
        max_len = int(config.max_sequence_length)
        return all(len(seq[0]) <= max_len for seq in candidate_set)

    @staticmethod
    def _log_relearn_progress(
        label: str,
        processed: int,
        total: int,
        built: int,
        words: int,
        skipped: int,
        started_at: float,
    ) -> None:
        elapsed = max(time.monotonic() - started_at, 1e-6)
        rate = processed / elapsed
        print(
            f"   {label}: {processed}/{total} entries | "
            f"candidate_sets={built} words={words} skipped={skipped} "
            f"rate={rate:.1f}/s elapsed={elapsed:.1f}s",
            flush=True,
        )

    @staticmethod
    def _json_digest(payload: Any) -> str:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def _entries_digest(cls, entries: List[Dict]) -> str:
        digest = hashlib.sha256()
        for entry in entries:
            digest.update(
                json.dumps(
                    entry,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\n")
        return digest.hexdigest()

    def _static_sequence_cache_metadata(
        self,
        scope: str,
        entries: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "cache_version": STATIC_PREPROCESS_CACHE_VERSION,
            "scope": scope,
            "sources": self.data_manager.get_preprocess_source_signature(include_code=False),
            "suffix_inventory": [suffix.name for suffix in sfx.ALL_SUFFIXES],
            "closed_class_inventory": [list(spec) for spec in CLOSED_CLASS_TOKEN_SPECS],
            "config": {
                "max_negative_candidates": int(config.max_negative_candidates),
                "max_sequence_length": int(config.max_sequence_length),
            },
        }
        if entries is not None:
            metadata["entries"] = {
                "count": len(entries),
                "digest": self._entries_digest(entries),
            }
        metadata["cache_key"] = self._json_digest(metadata)
        return metadata

    def _load_static_sequence_cache(
        self,
        scope: str,
        *,
        entries: Optional[List[Dict]] = None,
        label: str = "Preprocessing",
    ) -> Optional[Tuple[List[List[Any]], int, int]]:
        metadata = self._static_sequence_cache_metadata(scope, entries)
        cached = self.data_manager.load_preprocessed_sequences_cache(metadata["cache_key"], metadata)
        if cached is None:
            return None
        all_seqs, total_words, skipped = cached
        cache_path = (
            self.data_manager.last_preprocessed_sequences_cache_path
            or self.data_manager.preprocessed_sequences_cache_path(metadata["cache_key"])
        )
        print(
            f"   Loaded cached {label}: {len(all_seqs)} candidate sets, "
            f"{total_words} words, {skipped} skipped ({cache_path})",
            flush=True,
        )
        return cached

    def _save_static_sequence_cache(
        self,
        scope: str,
        entries: Optional[List[Dict]],
        all_seqs: List[List[Any]],
        total_words: int,
        skipped: int,
        *,
        label: str = "Preprocessing",
    ) -> None:
        metadata = self._static_sequence_cache_metadata(scope, entries)
        if self.data_manager.save_preprocessed_sequences_cache(metadata, all_seqs, total_words, skipped):
            cache_path = self.data_manager.preprocessed_sequences_cache_path(metadata["cache_key"])
            print(f"   Saved cached {label} to {cache_path}", flush=True)

    def _load_or_build_static_sequences(
        self,
        entries: List[Dict],
        *,
        scope: str,
        log_progress: bool = False,
        label: str = "Preprocessing",
    ) -> Tuple[List[List[Any]], int, int]:
        cached = self._load_static_sequence_cache(scope, entries=entries, label=label)
        if cached is not None:
            return cached
        all_seqs, total_words, skipped = self._entries_to_sequences(
            entries,
            log_progress=log_progress,
            label=label,
        )
        self._save_static_sequence_cache(scope, entries, all_seqs, total_words, skipped, label=label)
        return all_seqs, total_words, skipped

    def _load_training_entries_with_progress(self) -> List[Dict]:
        entries = []
        load_started_at = time.monotonic()
        load_interval = max(1, int(config.relearn_preprocess_log_interval))
        for idx, entry in enumerate(self.data_manager.iter_valid_decomps(), start=1):
            entries.append(entry)
            if idx == 1 or idx % load_interval == 0:
                elapsed = max(time.monotonic() - load_started_at, 1e-6)
                print(f"   Loading training entries: {idx} entries ({idx / elapsed:.1f}/s)", flush=True)
        print(
            f"   Loaded {len(entries)} training entries | "
            f"device={self.device} | {self.trainer.cuda_memory_report()}",
            flush=True,
        )
        return entries

    def _entries_to_sequences(
        self,
        entries: List[Dict],
        *,
        log_progress: bool = False,
        label: str = "Preprocessing",
    ) -> Tuple[List[List[Any]], int, int]:
        all_seqs = []
        skipped = 0
        total_words = 0
        started_at = time.monotonic()
        total_entries = len(entries)
        interval = max(1, int(config.relearn_preprocess_log_interval))
        if log_progress:
            print(f"   {label}: preparing candidate sets for {total_entries} entries...", flush=True)
        for idx, entry in enumerate(entries, start=1):
            try:
                if entry.get('type') == 'sentence': result = self._candidate_set_from_word_entries(entry.get('words', []))
                else: result = self._candidate_set_from_word_entries([entry])
                if result is None:
                    skipped += 1
                    if log_progress and (idx == 1 or idx == total_entries or idx % interval == 0):
                        self._log_relearn_progress(label, idx, total_entries, len(all_seqs), total_words, skipped, started_at)
                    continue
                candidate_set, word_count = result
                if len(candidate_set) >= 2:
                    if self._candidate_set_fits_model(candidate_set):
                        all_seqs.append(candidate_set)
                        total_words += word_count
                    else:
                        skipped += 1
                else: skipped += 1
            except Exception: skipped += 1
            if log_progress and (idx == 1 or idx == total_entries or idx % interval == 0):
                self._log_relearn_progress(label, idx, total_entries, len(all_seqs), total_words, skipped, started_at)
        return all_seqs, total_words, skipped

    def _entries_to_dynamic_sequences(
        self,
        entries: List[Dict],
        rng: random.Random,
        *,
        log_progress: bool = False,
        label: str = "Dynamic preprocessing",
    ) -> Tuple[List[List[Any]], int, int]:
        all_seqs = []
        skipped = 0
        total_words = 0
        started_at = time.monotonic()
        total_entries = len(entries)
        interval = max(1, int(config.relearn_preprocess_log_interval))
        if log_progress:
            print(f"   {label}: preparing candidate sets for {total_entries} entries...", flush=True)
        for idx, entry in enumerate(entries, start=1):
            try:
                if entry.get('type') == 'sentence': result = self._dynamic_candidate_set_from_word_entries(entry.get('words', []), rng)
                else: result = self._dynamic_candidate_set_from_word_entries([entry], rng)
                if result is None:
                    skipped += 1
                    if log_progress and (idx == 1 or idx == total_entries or idx % interval == 0):
                        self._log_relearn_progress(label, idx, total_entries, len(all_seqs), total_words, skipped, started_at)
                    continue
                candidate_set, word_count = result
                if len(candidate_set) >= 2:
                    if self._candidate_set_fits_model(candidate_set):
                        all_seqs.append(candidate_set)
                        total_words += word_count
                    else:
                        skipped += 1
                else: skipped += 1
            except Exception: skipped += 1
            if log_progress and (idx == 1 or idx == total_entries or idx % interval == 0):
                self._log_relearn_progress(label, idx, total_entries, len(all_seqs), total_words, skipped, started_at)
        return all_seqs, total_words, skipped

    def _split_train_validation_sequences(self, all_seqs: List[Any]) -> Tuple[List[Any], List[Any]]:
        if len(all_seqs) < 10 or config.validation_split <= 0.0: return all_seqs, []
        data = list(all_seqs)
        random.Random(config.validation_seed).shuffle(data)
        val_count = max(1, int(round(len(data) * config.validation_split)))
        if val_count >= len(data): val_count = len(data) - 1
        if val_count <= 0: return all_seqs, []
        val_seqs = data[:val_count]
        train_seqs = data[val_count:]
        print(f"   Validation split created from training data: {len(train_seqs)} train / {len(val_seqs)} val")
        return train_seqs, val_seqs

    def relearn_all(self) -> Tuple[int, int]:
        started_at = time.monotonic()
        print("\n=== Relearn started ===", flush=True)
        cached = self._load_static_sequence_cache(
            "training-all",
            label="Relearn preprocessing",
        )
        if cached is None:
            entries = self._load_training_entries_with_progress()
            all_seqs, total_words, skipped = self._entries_to_sequences(
                entries,
                log_progress=True,
                label="Relearn preprocessing",
            )
            self._save_static_sequence_cache(
                "training-all",
                None,
                all_seqs,
                total_words,
                skipped,
                label="Relearn preprocessing",
            )
        else:
            all_seqs, total_words, skipped = cached
        print(
            f"   Relearn preprocessing complete: {len(all_seqs)} candidate sets, "
            f"{total_words} words, {skipped} skipped in {time.monotonic() - started_at:.1f}s",
            flush=True,
        )
        train_seqs, val_seqs = self._split_train_validation_sequences(all_seqs)
        if train_seqs:
            print(
                f"   Bulk training on {len(train_seqs)} train sets and {len(val_seqs)} validation sets "
                f"({total_words} words)...",
                flush=True,
            )
            self.trainer.train_bulk(train_seqs, validation_seqs=val_seqs)
            self._save_final_suffix_metrics()
        self.training_count += total_words
        self.save()
        print(f"=== Relearn finished in {time.monotonic() - started_at:.1f}s ===", flush=True)
        return total_words, skipped

    def train_curriculum(self, generations: Optional[int] = None, warmup_epochs: Optional[int] = None, mining_epochs: Optional[int] = None) -> Dict[str, Any]:
        if generations is None: generations = config.curriculum_generations
        if warmup_epochs is None: warmup_epochs = config.curriculum_warmup_epochs
        if mining_epochs is None: mining_epochs = config.curriculum_mining_epochs
        entries = self.data_manager.get_valid_decomps()
        if not entries: return {'trained_words': 0, 'skipped': 0, 'generations': 0}
        train_entries = list(entries)
        val_seqs = []
        if len(entries) >= 10 and config.validation_split > 0.0:
            shuffled = list(entries)
            random.Random(config.validation_seed).shuffle(shuffled)
            val_count = max(1, int(round(len(shuffled) * config.validation_split)))
            if val_count >= len(shuffled): val_count = len(shuffled) - 1
            val_entries = shuffled[:val_count]
            train_entries = shuffled[val_count:]
            val_seqs, _, _ = self._load_or_build_static_sequences(
                val_entries,
                scope="curriculum-validation",
                label="Curriculum validation preprocessing",
            )
        total_trained_words = 0
        total_skipped = 0
        if warmup_epochs > 0:
            warmup_seqs, warmup_words, skipped = self._load_or_build_static_sequences(
                train_entries,
                scope="curriculum-warmup",
                label="Curriculum warm-up preprocessing",
            )
            total_skipped += skipped
            if warmup_seqs:
                print(f"   Curriculum warm-up: {len(warmup_seqs)} static candidate sets ({warmup_words} words), {warmup_epochs} epochs")
                self.trainer.train_bulk(warmup_seqs, epochs=warmup_epochs, validation_seqs=val_seqs)
                total_trained_words += warmup_words
                self.training_count += warmup_words
                self.save()
        completed_generations = 0
        for generation in range(1, generations + 1):
            rng = random.Random(config.validation_seed + generation + self.trainer.global_step)
            mined_seqs, mined_words, skipped = self._entries_to_dynamic_sequences(train_entries, rng)
            total_skipped += skipped
            if not mined_seqs: continue
            print(f"   Curriculum generation {generation}/{generations}: mined {len(mined_seqs)} candidate sets ({mined_words} words), {mining_epochs} epochs")
            self.trainer.train_bulk(mined_seqs, epochs=mining_epochs, validation_seqs=val_seqs)
            total_trained_words += mined_words
            self.training_count += mined_words
            self.save()
            completed_generations += 1
        self._save_final_suffix_metrics()
        return {'trained_words': total_trained_words, 'skipped': total_skipped, 'generations': completed_generations}

    def run_kfold_cv(self, k: int = 10, seed: int = 42) -> Optional[Dict[str, Any]]:
        cached = self._load_static_sequence_cache(
            "training-all",
            label="K-fold preprocessing",
        )
        if cached is None:
            entries = self.data_manager.get_valid_decomps()
            all_seqs, total_words, skipped = self._entries_to_sequences(
                entries,
                label="K-fold preprocessing",
            )
            self._save_static_sequence_cache(
                "training-all",
                None,
                all_seqs,
                total_words,
                skipped,
                label="K-fold preprocessing",
            )
        else:
            all_seqs, total_words, skipped = cached
        if len(all_seqs) < k: return None
        print(f"   Running {k}-fold CV on {len(all_seqs)} sequences ({total_words} words, {skipped} skipped).")
        tmp_dir = tempfile.mkdtemp(prefix="savyar_kfold_")
        def fold_runner(train_seqs, val_seqs, fold_idx: int):
            fold_path = os.path.join(tmp_dir, f"fold_{fold_idx}.pt")
            model = SentenceDisambiguator(
                suffix_vocab_size=len(sfx.ALL_SUFFIXES),
                closed_class_vocab_size=len(CLOSED_CLASS_TOKEN_SPECS),
                device=self.device,
            )
            trainer = Trainer(model=model, path=fold_path, device=self.device)
            trainer.train_bulk(list(train_seqs), validation_seqs=None)
            stats = trainer.validate(list(val_seqs))
            del trainer, model
            try:
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except Exception: pass
            return {name: float(val) for name, val in stats.items() if isinstance(val, (int, float)) and name != "n_batches"}
        try: result = run_k_fold_cv(all_seqs, k=k, fold_runner=fold_runner, seed=seed)
        finally:
            try: shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception: pass
        return result

    @staticmethod
    def _encoded_chain_suffix_names(chain: List) -> List[Optional[str]]:
        return [Trainer._suffix_name_for_token_id(tok[0]) for tok in chain]

    @staticmethod
    def _gold_entry_display(word_entry: Dict[str, Any]) -> str:
        suffixes = word_entry.get("suffixes", [])
        if not suffixes:
            return str(word_entry.get("root") or word_entry.get("word") or "")
        suffix_names = "+".join(sd.get("name", "?") for sd in suffixes)
        return f"{word_entry.get('root', word_entry.get('word', ''))}+{suffix_names}"

    def _candidate_diagnostics_from_word_entries(
        self,
        word_entries: List[Dict],
        entry: Dict[str, Any],
    ) -> Optional[Tuple[List[Any], Dict[str, Any]]]:
        gold_chains = []
        candidate_lists = []
        gold_indices = []
        candidate_displays = []

        for word_entry in word_entries:
            sfx_dicts = word_entry.get("suffixes", [])
            if not sfx_dicts:
                continue
            encoded_gold = nlp.encode_suffix_names(sfx_dicts)
            if not encoded_gold:
                continue

            try:
                word_analysis = nlp.analyze_word(word_entry["word"], include_closed_class=True)
                matched = nlp.match_decompositions([word_entry], word_analysis["decomps"])
            except Exception:
                word_analysis = None
                matched = []

            if matched and word_analysis is not None:
                gold_idx = matched[0]
                candidates = word_analysis["encoded_chains"]
                displays = [
                    nlp.format_detailed_decomp(decomp)
                    for decomp in word_analysis["decomps"]
                ]
                gold_chain = candidates[gold_idx]
            else:
                gold_idx = 0
                gold_chain = encoded_gold
                candidates = [encoded_gold]
                displays = [self._gold_entry_display(word_entry)]

            gold_chains.append(gold_chain)
            candidate_lists.append(candidates)
            gold_indices.append(gold_idx)
            candidate_displays.append(displays)

        if not gold_chains:
            return None

        candidate_set = [build_sentence_sequence(gold_chains)]
        combos = [list(gold_indices)]
        seen = {tuple(tuple(tok[0] for tok in chain) for chain in gold_chains)}
        max_candidate_set_size = 1 + max(0, int(config.max_negative_candidates))

        for word_idx, candidates in enumerate(candidate_lists):
            gold_idx = gold_indices[word_idx]
            for cand_idx, candidate in enumerate(candidates):
                if cand_idx == gold_idx:
                    continue
                neg_chains = list(gold_chains)
                neg_chains[word_idx] = candidate
                signature = tuple(tuple(tok[0] for tok in chain) for chain in neg_chains)
                if signature in seen:
                    continue
                seen.add(signature)
                candidate_set.append(build_sentence_sequence(neg_chains))
                combo = list(gold_indices)
                combo[word_idx] = cand_idx
                combos.append(combo)
                if len(candidate_set) >= max_candidate_set_size:
                    break
            if len(candidate_set) >= max_candidate_set_size:
                break

        if len(candidate_set) < 2:
            return None

        return candidate_set, {
            "entry": entry,
            "word_entries": word_entries,
            "gold_chains": gold_chains,
            "candidate_lists": candidate_lists,
            "gold_indices": gold_indices,
            "candidate_displays": candidate_displays,
            "combos": combos,
        }

    def _collect_test_detail(
        self,
        entries: List[Dict],
        suffix_metrics: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        worst_suffixes = [
            (name, metrics)
            for name, metrics in suffix_metrics.items()
            if (int(metrics.get("fp", 0)) + int(metrics.get("fn", 0))) > 0
        ]
        worst_suffixes.sort(
            key=lambda item: (
                float(item[1].get("f1", 0.0)),
                -int(item[1].get("fp", 0)) - int(item[1].get("fn", 0)),
                item[0],
            )
        )
        worst_names = [name for name, _ in worst_suffixes[:20]]
        examples_by_suffix: Dict[str, List[Dict[str, Any]]] = {
            name: [] for name in worst_names
        }

        diagnostic_sets: List[Tuple[List[Any], Dict[str, Any]]] = []
        skipped = 0
        for entry in entries:
            word_entries = entry.get("words", []) if entry.get("type") == "sentence" else [entry]
            try:
                result = self._candidate_diagnostics_from_word_entries(word_entries, entry)
                if result is None:
                    skipped += 1
                    continue
                candidate_set, meta = result
                if self._candidate_set_fits_model(candidate_set):
                    diagnostic_sets.append((candidate_set, meta))
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        batch_size = 64
        for start in range(0, len(diagnostic_sets), batch_size):
            batch = diagnostic_sets[start:start + batch_size]
            flat = [seq for candidate_set, _meta in batch for seq in candidate_set]
            sizes = [len(candidate_set) for candidate_set, _meta in batch]
            scores = self.trainer.score_flat_sequences(flat)
            offset = 0
            for (candidate_set, meta), size in zip(batch, sizes):
                group = scores[offset:offset + size]
                offset += size
                if not group:
                    continue
                best_idx = max(range(len(group)), key=lambda i: group[i])
                if best_idx == 0:
                    continue

                pred_combo = meta["combos"][best_idx]
                gold_indices = meta["gold_indices"]
                for word_idx, (gold_idx, pred_idx) in enumerate(zip(gold_indices, pred_combo)):
                    if gold_idx == pred_idx:
                        continue

                    gold_chain = meta["candidate_lists"][word_idx][gold_idx]
                    pred_chain = meta["candidate_lists"][word_idx][pred_idx]
                    gold_names = self._encoded_chain_suffix_names(gold_chain)
                    pred_names = self._encoded_chain_suffix_names(pred_chain)
                    max_len = max(len(gold_names), len(pred_names))

                    word_entry = meta["word_entries"][word_idx]
                    example_base = {
                        "sentence": meta["entry"].get("original_sentence") or word_entry.get("word", ""),
                        "word": word_entry.get("word", ""),
                        "gold": meta["candidate_displays"][word_idx][gold_idx],
                        "predicted": meta["candidate_displays"][word_idx][pred_idx],
                        "gold_score": group[0],
                        "pred_score": group[best_idx],
                    }

                    for pos in range(max_len):
                        gold_name = gold_names[pos] if pos < len(gold_names) else None
                        pred_name = pred_names[pos] if pos < len(pred_names) else None
                        if gold_name == pred_name:
                            continue

                        if gold_name in examples_by_suffix and len(examples_by_suffix[gold_name]) < 10:
                            examples_by_suffix[gold_name].append({
                                **example_base,
                                "failure": "missed",
                                "expected": gold_name,
                                "got": pred_name or "(none)",
                            })
                        if pred_name in examples_by_suffix and len(examples_by_suffix[pred_name]) < 10:
                            examples_by_suffix[pred_name].append({
                                **example_base,
                                "failure": "false_positive",
                                "expected": gold_name or "(none)",
                                "got": pred_name,
                            })

        return {
            "worst_suffixes": [
                {"name": name, **metrics}
                for name, metrics in worst_suffixes[:20]
            ],
            "examples": examples_by_suffix,
            "diagnostic_sequences": len(diagnostic_sets),
            "diagnostic_skipped": skipped,
        }

    def _evaluate_overall_test_tokens(self, entries: List[Dict]) -> Dict[str, Any]:
        total = 0
        correct = 0
        ambiguous_total = 0
        ambiguous_correct = 0
        single_total = 0
        single_correct = 0
        root_only_total = 0
        no_candidate = 0
        unmatched_gold = 0
        scoring_errors = 0
        scoring_items: List[Tuple[List[Any], Dict[str, Any]]] = []

        for entry in entries:
            word_entries = entry.get("words", []) if entry.get("type") == "sentence" else [entry]
            matched_infos: List[Dict[str, Any]] = []

            for word_entry in word_entries:
                total += 1
                if not word_entry.get("suffixes", []):
                    root_only_total += 1
                    single_total += 1
                    single_correct += 1
                    correct += 1
                    continue

                try:
                    analysis = nlp.analyze_word(word_entry["word"], include_closed_class=True)
                except Exception:
                    analysis = None

                if not analysis or not analysis.get("decomps"):
                    no_candidate += 1
                    continue

                matched = nlp.match_decompositions([word_entry], analysis["decomps"])
                if not matched:
                    unmatched_gold += 1
                    single_total += 1
                    single_correct += 1
                    correct += 1
                    continue

                gold_idx = matched[0]
                if len(analysis["decomps"]) <= 1:
                    single_total += 1
                else:
                    ambiguous_total += 1

                matched_infos.append({
                    "encoded_chains": analysis["encoded_chains"],
                    "gold_idx": gold_idx,
                    "is_ambiguous": len(analysis["decomps"]) > 1,
                })

            if not any(info["is_ambiguous"] for info in matched_infos):
                correct += len(matched_infos)
                single_correct += len(matched_infos)
                continue

            gold_indices = [info["gold_idx"] for info in matched_infos]
            gold_chains = [
                info["encoded_chains"][info["gold_idx"]]
                for info in matched_infos
            ]
            candidate_lists = [info["encoded_chains"] for info in matched_infos]
            candidate_set = [build_sentence_sequence(gold_chains)]
            combos = [list(gold_indices)]
            seen = {tuple(tuple(tok[0] for tok in chain) for chain in gold_chains)}
            max_candidate_set_size = 1 + max(0, int(config.max_negative_candidates))

            for word_idx, candidates in enumerate(candidate_lists):
                gold_idx = gold_indices[word_idx]
                for cand_idx, candidate in enumerate(candidates):
                    if cand_idx == gold_idx:
                        continue
                    neg_chains = list(gold_chains)
                    neg_chains[word_idx] = candidate
                    signature = tuple(tuple(tok[0] for tok in chain) for chain in neg_chains)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    candidate_set.append(build_sentence_sequence(neg_chains))
                    combo = list(gold_indices)
                    combo[word_idx] = cand_idx
                    combos.append(combo)
                    if len(candidate_set) >= max_candidate_set_size:
                        break
                if len(candidate_set) >= max_candidate_set_size:
                    break

            if len(candidate_set) < 2 or not self._candidate_set_fits_model(candidate_set):
                scoring_errors += len(matched_infos)
                continue

            scoring_items.append((candidate_set, {
                "infos": matched_infos,
                "gold_indices": gold_indices,
                "combos": combos,
            }))

        batch_size = 64
        for start in range(0, len(scoring_items), batch_size):
            batch = scoring_items[start:start + batch_size]
            flat = [seq for candidate_set, _meta in batch for seq in candidate_set]
            sizes = [len(candidate_set) for candidate_set, _meta in batch]
            try:
                scores = self.trainer.score_flat_sequences(flat)
            except Exception:
                scoring_errors += sum(len(meta["infos"]) for _candidate_set, meta in batch)
                continue

            offset = 0
            for (_candidate_set, meta), size in zip(batch, sizes):
                group = scores[offset:offset + size]
                offset += size
                if not group:
                    scoring_errors += len(meta["infos"])
                    continue
                best_idx = max(range(len(group)), key=lambda i: group[i])
                pred_combo = meta["combos"][best_idx]
                for info, gold_idx, pred_idx in zip(meta["infos"], meta["gold_indices"], pred_combo):
                    if pred_idx == gold_idx:
                        correct += 1
                        if info["is_ambiguous"]:
                            ambiguous_correct += 1
                        else:
                            single_correct += 1

        return {
            "token_acc": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "ambiguous_token_acc": ambiguous_correct / ambiguous_total if ambiguous_total else 0.0,
            "ambiguous_correct": ambiguous_correct,
            "ambiguous_total": ambiguous_total,
            "single_token_acc": single_correct / single_total if single_total else 0.0,
            "single_correct": single_correct,
            "single_total": single_total,
            "single_ratio": single_total / total if total else 0.0,
            "root_only_total": root_only_total,
            "no_candidate": no_candidate,
            "unmatched_gold": unmatched_gold,
            "scoring_errors": scoring_errors,
        }

    def test_model(self, detailed: bool = False) -> Dict[str, Any]:
        entries = self.data_manager.get_test_entries()
        if not entries:
            return {
                'entries': 0,
                'sequences': 0,
                'words': 0,
                'skipped': 0,
                'metrics': None,
                'overall_metrics': None,
            }
        test_seqs, total_words, skipped = self._entries_to_sequences(entries)
        metrics = self.trainer.validate(test_seqs) if test_seqs else None
        report = {
            'entries': len(entries),
            'sequences': len(test_seqs),
            'words': total_words,
            'skipped': skipped,
            'metrics': metrics,
            'overall_metrics': self._evaluate_overall_test_tokens(entries),
        }
        if detailed and metrics:
            report["detail"] = self._collect_test_detail(
                entries,
                metrics.get("suffix_metrics", {}),
            )
        return report

    def sample_text(self, filename: str) -> bool:
        text = self.data_manager.get_text_tokenized(filename)
        if not text: return False
        unique_words = list(set(text))
        cache = {}
        for word in unique_words:
            decomps = self.get_decompositions(word)
            if not decomps: cache[word] = word
            elif len(decomps) == 1: cache[word] = nlp.format_detailed_decomp(decomps[0])
            else:
                suffix_chains = [chain for _, _, chain, _ in decomps]
                encoded_chains = [nlp.encode_suffix_chain(chain) for chain in suffix_chains]
                best_idx = 0
                if self.training_count > 0:
                    try: best_idx, _ = self.trainer.predict(encoded_chains)
                    except Exception: best_idx = 0
                if best_idx >= len(decomps): best_idx = 0
                cache[word] = nlp.format_detailed_decomp(decomps[best_idx])
        final_output = [cache.get(word, word) for word in text]
        return self.data_manager.write_decomposed_text('\n'.join(final_output))

    def sample_sentences(self) -> bool:
        raw_text = self.data_manager.get_raw_sentences_text()
        if not raw_text: return False
        output_lines = []
        for line in raw_text.split('\n'):
            if not line.strip():
                output_lines.append("")
                continue
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', line) if s.strip()]
            line_output = []
            for sentence in sentences:
                clean_sentence = re.sub(r"['’‘]", "", sentence)
                clean_sentence = tr_lower(re.sub(r'[^\w\s]|_', ' ', clean_sentence))
                word_data = self.prepare_sentence_training(clean_sentence)
                if not word_data:
                    line_output.append(sentence)
                    continue
                top_predictions = get_top_sentence_predictions(word_data, self.trainer, top_k=1)
                if top_predictions:
                    best_combo = top_predictions[0]['combo_indices']
                    decomposed_words = []
                    for w_idx, cand_idx in enumerate(best_combo):
                        decomp = word_data[w_idx]['decomps'][cand_idx]
                        decomposed_words.append(nlp.format_detailed_decomp(decomp))
                    line_output.append(" ".join(decomposed_words) + ".")
                else: line_output.append(sentence)
            output_lines.append("  ".join(line_output))
        return self.data_manager.write_decomposed_sentences("\n".join(output_lines))

    def get_stats(self) -> Dict:
        stats = {'total': self.training_count, 'recent_avg': 0.0, 'latest': 0.0, 'best_val': self.trainer.best_val_loss}
        if self.trainer.train_history:
            recent = self.trainer.train_history[-20:]
            stats['recent_avg'] = sum(recent)/len(recent)
            stats['latest'] = self.trainer.train_history[-1]
        return stats
