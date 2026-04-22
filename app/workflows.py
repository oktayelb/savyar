import re
from typing import List, Optional, Tuple, Dict, Any

import util.decomposer as sfx
import util.word_methods as wrd
from util.word_methods import tr_lower
from app.data_manager import DataManager
import app.morphology_adapter as morph
from app.sequence_matcher import find_matching_combinations, get_top_sentence_predictions
from ml.ml_ranking_model import SentenceDisambiguator, Trainer
from util.words.closed_class import ALL_CLOSED_CLASS_WORDS

class WorkflowEngine:
    def __init__(self):
        sfx.enable_index()
        self.data_manager = DataManager()
        self.model = SentenceDisambiguator(
            suffix_vocab_size=len(sfx.ALL_SUFFIXES),
            closed_class_vocab_size=len(ALL_CLOSED_CLASS_WORDS),
        )
        self.trainer = Trainer(model=self.model)
        self.training_count = self.data_manager.load_training_count()
        self.decomp_cache = {}

        # If the checkpoint carried no replay buffer (fresh start or old checkpoint),
        # rebuild it from logged confirmed decompositions so the model immediately
        # has past data to replay during the first training call.
        if not self.trainer.replay_buffer:
            self._preload_replay_buffer()

    def _preload_replay_buffer(self) -> None:
        """Reconstruct the replay buffer from the confirmed-decompositions JSONL log."""
        from ml.ml_ranking_model import build_sentence_sequence
        entries = self.data_manager.get_valid_decomps()
        loaded = 0
        for entry in entries:
            try:
                if entry.get('type') == 'sentence':
                    chains = []
                    for word_entry in entry.get('words', []):
                        decomps = self.get_decompositions(word_entry['word'])
                        matched = morph.match_decompositions([word_entry], decomps)
                        if matched:
                            chain = [chain for _, _, chain, _ in decomps][matched[0]]
                            chains.append(morph.encode_suffix_chain(chain))
                        else:
                            # Fallback: encode directly from suffix names (treebank-forced entries)
                            sfx_dicts = word_entry.get('suffixes', [])
                            if sfx_dicts:
                                chains.append(morph.encode_suffix_names(sfx_dicts))
                    if chains:
                        sids, cids = build_sentence_sequence(chains)
                        if len(sids) >= 2:
                            self.trainer._add_to_replay(sids, cids)
                            loaded += 1
                else:
                    decomps = self.get_decompositions(entry['word'])
                    matched = morph.match_decompositions([entry], decomps)
                    if matched:
                        chain = [c for _, _, c, _ in decomps][matched[0]]
                        encoded = morph.encode_suffix_chain(chain)
                    else:
                        # Fallback: encode directly from suffix names
                        sfx_dicts = entry.get('suffixes', [])
                        encoded = morph.encode_suffix_names(sfx_dicts) if sfx_dicts else []
                    if encoded:
                        sids, cids = build_sentence_sequence([encoded])
                        if len(sids) >= 2:
                            self.trainer._add_to_replay(sids, cids)
                            loaded += 1
            except Exception:
                continue
        if loaded:
            import random
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

    def prepare_word_training(self, word: str) -> Optional[Dict[str, Any]]:
        decompositions = self.get_decompositions(word)
        if not decompositions:
            return None
        
        suffix_chains = [chain for _, _, chain, _ in decompositions]
        if len(suffix_chains) == 1:
            return {'single_decomposition': True}

        encoded_chains = [morph.encode_suffix_chain(chain) for chain in suffix_chains]

        scores = None
        if self.training_count > 0:
            try:
                _, scores = self.trainer.predict(encoded_chains)
            except Exception:
                pass
        
        scored_decomps = [
            (scores[i] if scores else 0.0, decomp)
            for i, decomp in enumerate(decompositions)
        ]
        if scores:
            scored_decomps.sort(key=lambda x: x[0], reverse=True)

        view_models = []
        sorted_decomps = []
        
        for score, decomp in scored_decomps:
            vm = morph.reconstruct_morphology(word, decomp)
            vm['score'] = score if scores else None
            view_models.append(vm)
            sorted_decomps.append(decomp)

        return {
            'single_decomposition': False,
            'view_models': view_models,
            'sorted_decomps': sorted_decomps,
            'encoded_chains': encoded_chains,
            'has_scores': bool(scores),
            'original_decompositions': decompositions
        }

    def commit_word_training(self, word: str, correct_decomps: List[Tuple], encoded_chains: List[List[Tuple[int, int]]], original_indices: List[int]) -> Tuple[float, List[str]]:
        log_entries = []
        deleted_messages = []
        
        from util.words.closed_class import ClosedClassMarker as _CCMarker
        word_lower = tr_lower(word)
        for decomp in correct_decomps:
            root, pos, chain, final_pos = decomp
            suffix_info = []
            if chain and not isinstance(chain[0], _CCMarker):
                current = root
                for suffix in chain:
                    forms = suffix.form(current)
                    # Match against the actual surface string
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
            log_entries.append({
                'word': word,
                'root': root,
                'suffixes': suffix_info,
                'final_pos': final_pos,
            })
        self.data_manager.log_decompositions(log_entries)

        for decomp in correct_decomps:
            root = tr_lower(decomp[0])
            if root == word_lower:
                continue
            if self.data_manager.delete(word_lower):
                deleted_messages.append(f"Deleted '{word}' (root '{root}' exists)")
                sfx.decompose.cache_clear()
                self.decomp_cache.pop(word_lower, None)
            infinitive_form = wrd.infinitive(word_lower)
            if self.data_manager.delete(infinitive_form):
                deleted_messages.append(f"Deleted infinitive '{infinitive_form}'")
                sfx.decompose.cache_clear()
                self.decomp_cache.pop(infinitive_form, None)

        loss = 0.0
        for idx in original_indices:
            confirmed_chain = encoded_chains[idx]
            loss = self.trainer.train_sentence([confirmed_chain])

        self.training_count += 1
        if self.training_count % self.trainer.checkpoint_frequency == 0:
            self.save()
        return loss, deleted_messages

    def get_decompositions_with_cc(self, word: str) -> List[Tuple]:
        """Like get_decompositions but also includes closed-class word analyses."""
        word = word.replace("'", "")
        return sfx.decompose_with_cc(word)

    def prepare_sentence_training(self, sentence: str) -> Optional[List[Dict]]:
        words = sentence.strip().split()
        if not words:
            return None

        word_data = []
        for word in words:
            decomps = self.get_decompositions_with_cc(word)
            if not decomps:
                return None
            word_data.append({'word': word, 'decomps': decomps})

        for wd in word_data:
            suffix_chains = [chain for _, _, chain, _ in wd['decomps']]
            wd['encoded_chains'] = [morph.encode_suffix_chain(chain) for chain in suffix_chains]
            wd['vms'] = []
            wd['typing_strings'] = []
            
            for decomp in wd['decomps']:
                vm = morph.reconstruct_morphology(wd['word'], decomp)
                wd['vms'].append(vm)
                root = decomp[0]
                if vm['has_chain']:
                    suffixes = vm['suffixes_str'].replace(" + ", " ")
                    wd['typing_strings'].append(f"{root} {suffixes}")
                else:
                    wd['typing_strings'].append(root)
        return word_data

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
            
            root, pos, chain, final_pos = decomps[correct_d_idx]
            suffix_info = []
            from util.words.closed_class import ClosedClassMarker as _CCMarker
            word_lower = tr_lower(word)
            if chain and not isinstance(chain[0], _CCMarker):
                current = root
                for suffix in chain:
                    forms = suffix.form(current)
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
            log_entries.append({
                'word': word,
                'morphology_string': typing_str,
                'root': root,
                'suffixes': suffix_info,
                'final_pos': final_pos,
            })

        self.data_manager.log_sentence_decompositions(log_entries, sentence)
        loss = self.trainer.train_sentence(confirmed_chains)

        self.training_count += len(confirmed_chains)
        if self.training_count % self.trainer.checkpoint_frequency == 0:
            self.save()
        return loss

    def _entries_to_sequences(self, entries: List[Dict]) -> Tuple[List[Tuple[List[int], List[int]]], int, int]:
        """Convert logged/treebank-adapted entries into flat (suffix_ids, category_ids) sequences.

        Returns (sequences, total_words, skipped).
        """
        from ml.ml_ranking_model import build_sentence_sequence

        all_seqs: List[Tuple[List[int], List[int]]] = []
        skipped = 0
        total_words = 0

        for entry in entries:
            try:
                if entry.get('type') == 'sentence':
                    chains = []
                    for word_entry in entry.get('words', []):
                        sfx_dicts = word_entry.get('suffixes', [])
                        if sfx_dicts:
                            chains.append(morph.encode_suffix_names(sfx_dicts))
                    if chains:
                        sids, cids = build_sentence_sequence(chains)
                        if len(sids) >= 2:
                            all_seqs.append((sids, cids))
                            total_words += len(chains)
                else:
                    sfx_dicts = entry.get('suffixes', [])
                    if sfx_dicts:
                        encoded = morph.encode_suffix_names(sfx_dicts)
                        if encoded:
                            sids, cids = build_sentence_sequence([encoded])
                            if len(sids) >= 2:
                                all_seqs.append((sids, cids))
                                total_words += 1
                    else:
                        skipped += 1
            except Exception:
                skipped += 1

        return all_seqs, total_words, skipped

    def _load_validation_sequences(self) -> List[Tuple[List[int], List[int]]]:
        """Load and encode the held-out validation set as training sequences."""
        entries = self.data_manager.get_validation_entries()
        if not entries:
            return []
        val_seqs, val_words, _ = self._entries_to_sequences(entries)
        if val_seqs:
            print(f"   Validation set loaded: {len(val_seqs)} sequences ({val_words} words)")
        return val_seqs

    def relearn_all(self) -> Tuple[int, int]:
        entries = self.data_manager.get_valid_decomps()
        all_seqs, total_words, skipped = self._entries_to_sequences(entries)

        val_seqs = self._load_validation_sequences()

        if all_seqs:
            print(f"   Bulk training on {len(all_seqs)} sequences ({total_words} words)...")
            self.trainer.train_bulk(all_seqs, validation_seqs=val_seqs)

        self.training_count += total_words
        self.save()
        return total_words, skipped

    def evaluate_word(self, word: str) -> Optional[Dict]:
        decompositions = self.get_decompositions(word)
        if not decompositions:
            return None
        
        suffix_chains = [chain for _, _, chain, _ in decompositions]
        encoded_chains = [morph.encode_suffix_chain(chain) for chain in suffix_chains]

        try:
            best_idx, scores = self.trainer.predict(encoded_chains)
            best_decomp = decompositions[best_idx]
            vm = morph.reconstruct_morphology(word, best_decomp)
            vm['score'] = scores[best_idx]
            return vm
        except Exception:
            return None

    def sample_text(self, filename: str) -> bool:
        text = self.data_manager.get_text_tokenized(filename)
        if not text:
            return False
            
        unique_words = list(set(text))
        cache = {}
        
        for word in unique_words:
            decomps = self.get_decompositions(word)
            if not decomps:
                cache[word] = word
            elif len(decomps) == 1:
                cache[word] = morph.format_detailed_decomp(decomps[0])
            else:
                suffix_chains = [chain for _, _, chain, _ in decomps]
                encoded_chains = [morph.encode_suffix_chain(chain) for chain in suffix_chains]
                
                best_idx = 0
                if self.training_count > 0:
                    try:
                        best_idx, _ = self.trainer.predict(encoded_chains)
                    except Exception:
                        best_idx = 0

                if best_idx >= len(decomps):
                    best_idx = 0
                        
                cache[word] = morph.format_detailed_decomp(decomps[best_idx])

        final_output = [cache.get(word, word) for word in text]
        output_text = '\n'.join(final_output)
        return self.data_manager.write_decomposed_text(output_text)
        
    def sample_sentences(self) -> bool:
        raw_text = self.data_manager.get_raw_sentences_text()
        if not raw_text:
            return False
            
        output_lines = []
        lines = raw_text.split('\n')
        
        for line in lines:
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
                        decomposed_words.append(morph.format_detailed_decomp(decomp))
                    
                    line_output.append(" ".join(decomposed_words) + ".")
                else:
                    line_output.append(sentence)
                    
            output_lines.append("  ".join(line_output))
            
        final_output = "\n".join(output_lines)
        return self.data_manager.write_decomposed_sentences(final_output)

    def get_stats(self) -> Dict:
        stats = {
            'total': self.training_count,
            'recent_avg': 0.0,
            'latest': 0.0,
            'best_val': self.trainer.best_val_loss
        }
        if self.trainer.train_history:
            recent = self.trainer.train_history[-20:]
            stats['recent_avg'] = sum(recent)/len(recent)
            stats['latest'] = self.trainer.train_history[-1]
        return stats