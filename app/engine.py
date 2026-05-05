"""Orchestrator and ML Logic. 
Combines the Workflow Engine, Sequence Matching, and K-Fold Cross Validation.
"""
from __future__ import annotations
import os
import re
import math
import random
import tempfile
import shutil
import torch
from typing import List, Optional, Tuple, Dict, Any, Callable, Sequence

import util.decomposer as sfx
import util.word_methods as wrd
from util.word_methods import tr_lower
from app.data_manager import DataManager
import app.nlp_pipeline as nlp
from ml.ml_ranking_model import SentenceDisambiguator, Trainer, build_sentence_sequence
from ml.config import config
from util.words.closed_class import CLOSED_CLASS_TOKEN_SPECS

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
        self.model = SentenceDisambiguator(
            suffix_vocab_size=len(sfx.ALL_SUFFIXES),
            closed_class_vocab_size=len(CLOSED_CLASS_TOKEN_SPECS),
        )
        self.trainer = Trainer(model=self.model)
        self.training_count = self.data_manager.load_training_count()
        self.decomp_cache = {}

        if not self.trainer.replay_buffer:
            self._preload_replay_buffer()

    def _preload_replay_buffer(self) -> None:
        entries = self.data_manager.get_valid_decomps()
        loaded = 0
        for entry in entries:
            try:
                if entry.get('type') == 'sentence':
                    chains = []
                    for word_entry in entry.get('words', []):
                        decomps = self.get_decompositions(word_entry['word'])
                        matched = nlp.match_decompositions([word_entry], decomps)
                        if matched:
                            chain = [chain for _, _, chain, _ in decomps][matched[0]]
                            chains.append(nlp.encode_suffix_chain(chain))
                        else:
                            sfx_dicts = word_entry.get('suffixes', [])
                            if sfx_dicts:
                                chains.append(nlp.encode_suffix_names(sfx_dicts))
                    if chains:
                        sids, cids, gids, comes_to_ids, makes_ids, word_pos_ids, word_final = build_sentence_sequence(chains)
                        if len(sids) >= 2:
                            self.trainer._add_to_replay(sids, cids, gids, comes_to_ids, makes_ids, word_pos_ids, word_final)
                            loaded += 1
                else:
                    decomps = self.get_decompositions(entry['word'])
                    matched = nlp.match_decompositions([entry], decomps)
                    if matched:
                        chain = [c for _, _, c, _ in decomps][matched[0]]
                        encoded = nlp.encode_suffix_chain(chain)
                    else:
                        sfx_dicts = entry.get('suffixes', [])
                        encoded = nlp.encode_suffix_names(sfx_dicts) if sfx_dicts else []
                    if encoded:
                        sids, cids, gids, comes_to_ids, makes_ids, word_pos_ids, word_final = build_sentence_sequence([encoded])
                        if len(sids) >= 2:
                            self.trainer._add_to_replay(sids, cids, gids, comes_to_ids, makes_ids, word_pos_ids, word_final)
                            loaded += 1
            except Exception:
                continue
        if loaded:
            random.shuffle(self.trainer.replay_buffer)
            print(f"Replay buffer pre-loaded with {loaded} past examples.")

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

    def analyze_sentence(self, words: List[str]) -> Optional[List[Dict[str, Any]]]:
        if not words:
            return None
        analyses = nlp.analyze_words(words, include_closed_class=True)
        if any(not a['decomps'] for a in analyses):
            return None
        return analyses

    def commit_word(self, analysis: Dict[str, Any], selected_indices: List[int]) -> Tuple[float, List[str]]:
        from util.words.closed_class import ClosedClassMarker as _CCMarker

        word = analysis['word']
        word_lower = tr_lower(word)
        correct_decomps = [analysis['decomps'][i] for i in selected_indices]
        correct_encoded = [analysis['encoded_chains'][i] for i in selected_indices]

        log_entries: List[Dict[str, Any]] = []
        for decomp in correct_decomps:
            root, pos, chain, final_pos = decomp
            suffix_info: List[Dict[str, Any]] = []
            if chain and not isinstance(chain[0], _CCMarker):
                current = root
                accepted_chain = []
                for suffix in chain:
                    forms = suffix.form(current, current_chain=accepted_chain)
                    rest = word_lower[len(current):]
                    used_form = ""
                    for f in forms:
                        if f and rest.startswith(f):
                            used_form = f
                            break
                    if not used_form:
                        used_form = forms[0] if forms else ""
                    suffix_info.append({
                        'name': suffix.name,
                        'form': used_form,
                        'makes': suffix.makes.name if suffix.makes else None,
                    })
                    current += used_form
                    accepted_chain.append(suffix)
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
        from util.words.closed_class import ClosedClassMarker as _CCMarker

        confirmed_chains = []
        log_entries = []

        for w_idx, correct_d_idx in enumerate(correct_combo):
            wd = word_data[w_idx]
            word = wd['word']
            decomps = wd['decomps']
            typing_str = wd['typing_strings'][correct_d_idx]
            confirmed_chain = wd['encoded_chains'][correct_d_idx]
            confirmed_chains.append(confirmed_chain)
            root, pos, chain, final_pos = decomps[correct_d_idx]
            suffix_info = []
            word_lower = tr_lower(word)
            if chain and not isinstance(chain[0], _CCMarker):
                current = root
                accepted_chain = []
                for suffix in chain:
                    forms = suffix.form(current, current_chain=accepted_chain)
                    rest = word_lower[len(current):]
                    used_form = ""
                    for f in forms:
                        if f and rest.startswith(f):
                            used_form = f
                            break
                    if not used_form:
                        used_form = forms[0] if forms else ""
                    suffix_info.append({'name': suffix.name, 'form': used_form, 'makes': suffix.makes.name if suffix.makes else None})
                    current += used_form
                    accepted_chain.append(suffix)
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
        negatives = self._single_substitution_negatives(gold_chains, candidate_lists, gold_indices, limit=config.dynamic_negative_pool_size)
        if not negatives: return None
        negative_seqs = [build_sentence_sequence(neg) for neg in negatives]
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

    def _entries_to_sequences(self, entries: List[Dict]) -> Tuple[List[List[Any]], int, int]:
        all_seqs = []
        skipped = 0
        total_words = 0
        for entry in entries:
            try:
                if entry.get('type') == 'sentence': result = self._candidate_set_from_word_entries(entry.get('words', []))
                else: result = self._candidate_set_from_word_entries([entry])
                if result is None:
                    skipped += 1
                    continue
                candidate_set, word_count = result
                if len(candidate_set) >= 2:
                    all_seqs.append(candidate_set)
                    total_words += word_count
                else: skipped += 1
            except Exception: skipped += 1
        return all_seqs, total_words, skipped

    def _entries_to_dynamic_sequences(self, entries: List[Dict], rng: random.Random) -> Tuple[List[List[Any]], int, int]:
        all_seqs = []
        skipped = 0
        total_words = 0
        for entry in entries:
            try:
                if entry.get('type') == 'sentence': result = self._dynamic_candidate_set_from_word_entries(entry.get('words', []), rng)
                else: result = self._dynamic_candidate_set_from_word_entries([entry], rng)
                if result is None:
                    skipped += 1
                    continue
                candidate_set, word_count = result
                if len(candidate_set) >= 2:
                    all_seqs.append(candidate_set)
                    total_words += word_count
                else: skipped += 1
            except Exception: skipped += 1
        return all_seqs, total_words, skipped

    def _load_validation_sequences(self) -> List[Any]:
        entries = self.data_manager.get_validation_entries()
        if not entries: return []
        val_seqs, val_words, _ = self._entries_to_sequences(entries)
        if val_seqs: print(f"   Validation set loaded: {len(val_seqs)} sequences ({val_words} words)")
        return val_seqs

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
        entries = self.data_manager.get_valid_decomps()
        all_seqs, total_words, skipped = self._entries_to_sequences(entries)
        val_seqs = self._load_validation_sequences()
        train_seqs = all_seqs
        if not val_seqs: train_seqs, val_seqs = self._split_train_validation_sequences(all_seqs)
        if train_seqs:
            print(f"   Bulk training on {len(train_seqs)} sequences ({total_words} words)...")
            self.trainer.train_bulk(train_seqs, validation_seqs=val_seqs)
            self._save_final_suffix_metrics()
        self.training_count += total_words
        self.save()
        return total_words, skipped

    def train_curriculum(self, generations: Optional[int] = None, warmup_epochs: Optional[int] = None, mining_epochs: Optional[int] = None) -> Dict[str, Any]:
        if generations is None: generations = config.curriculum_generations
        if warmup_epochs is None: warmup_epochs = config.curriculum_warmup_epochs
        if mining_epochs is None: mining_epochs = config.curriculum_mining_epochs
        entries = self.data_manager.get_valid_decomps()
        if not entries: return {'trained_words': 0, 'skipped': 0, 'generations': 0}
        val_seqs = self._load_validation_sequences()
        train_entries = list(entries)
        if not val_seqs and len(entries) >= 10 and config.validation_split > 0.0:
            shuffled = list(entries)
            random.Random(config.validation_seed).shuffle(shuffled)
            val_count = max(1, int(round(len(shuffled) * config.validation_split)))
            if val_count >= len(shuffled): val_count = len(shuffled) - 1
            val_entries = shuffled[:val_count]
            train_entries = shuffled[val_count:]
            val_seqs, _, _ = self._entries_to_sequences(val_entries)
        total_trained_words = 0
        total_skipped = 0
        if warmup_epochs > 0:
            warmup_seqs, warmup_words, skipped = self._entries_to_sequences(train_entries)
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
        entries = self.data_manager.get_valid_decomps()
        all_seqs, total_words, skipped = self._entries_to_sequences(entries)
        if len(all_seqs) < k: return None
        print(f"   Running {k}-fold CV on {len(all_seqs)} sequences ({total_words} words, {skipped} skipped).")
        tmp_dir = tempfile.mkdtemp(prefix="savyar_kfold_")
        def fold_runner(train_seqs, val_seqs, fold_idx: int):
            fold_path = os.path.join(tmp_dir, f"fold_{fold_idx}.pt")
            model = SentenceDisambiguator(suffix_vocab_size=len(sfx.ALL_SUFFIXES), closed_class_vocab_size=len(CLOSED_CLASS_TOKEN_SPECS))
            trainer = Trainer(model=model, path=fold_path)
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