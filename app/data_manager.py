import os
import json
import re
from typing import List, Optional, Dict

from app.file_paths import FilePaths
import util.word_methods as wrd
from util.word_methods import tr_lower

class DataManager:
    def __init__(self):
        self.paths = FilePaths()

    def load_training_count(self) -> int:
        try:
            if os.path.exists(self.paths.training_count_path):
                with open(self.paths.training_count_path, "r") as f:
                    return int(f.read().strip())
        except Exception:
            pass
        return 0

    def save_training_count(self, count: int):
        try:
            with open(self.paths.training_count_path, "w") as f:
                f.write(str(count))
        except Exception:
            pass

    def save_final_suffix_metrics(self, metrics: Dict) -> bool:
        try:
            training = metrics.get("training", {})
            validation = metrics.get("validation", metrics)
            payload = {
                "training": {
                    "rank_accuracy": float(training.get("rank_acc", 0.0)),
                    "top2_accuracy": float(training.get("top2_acc", 0.0)),
                    "top3_accuracy": float(training.get("top3_acc", 0.0)),
                    "loss": float(training.get("loss", 0.0)),
                    "margin": float(training.get("margin", 0.0)),
                    "n_batches": int(training.get("n_batches", 0)),
                    "total_sets": int(training.get("total", 0)),
                },
                "validation": {
                    "suffix_accuracy": float(validation.get("suff_acc", 0.0)),
                    "suffix_precision": float(validation.get("suff_precision", 0.0)),
                    "suffix_recall": float(validation.get("suff_recall", 0.0)),
                    "suffix_f1": float(validation.get("suff_f1", 0.0)),
                    "rank_accuracy": float(validation.get("rank_acc", 0.0)),
                    "top2_accuracy": float(validation.get("top2_acc", 0.0)),
                    "top3_accuracy": float(validation.get("top3_acc", 0.0)),
                    "validation_loss": float(validation.get("loss", 0.0)),
                    "margin": float(validation.get("margin", 0.0)),
                    "n_batches": int(validation.get("n_batches", 0)),
                },
                "suffixes": validation.get("suffix_metrics", {}),
                "groups": validation.get("suffix_group_metrics", {}),
            }
            with open(self.paths.final_suffix_metrics_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.write("\n")
            return True
        except Exception:
            return False

    def random_word(self) -> Optional[str]:
        return wrd.get_random_word()

    def get_text_tokenized(self, filename: str = None) -> List[str]:
        text_path = filename if filename and os.path.exists(filename) else self.paths.sample_text_path
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            content = re.sub(r"['’‘]", "", content)
            content = re.sub(r'[^\w\s]|_', ' ', content)
            
            words = [tr_lower(word) for word in content.split()]
            return words
        except Exception:
            return []
            
    def get_raw_sentences_text(self) -> str:
        text_path = getattr(self.paths, 'sample_sentence_path', 'sample/sample_sentence.txt')
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def get_valid_decomps(self) -> List[Dict]:
        entries = []
        paths_to_load = [
            self.paths.valid_decompositions_path,
            self.paths.treebank_adapted_path,
            self.paths.google_treebank_adapted_path,
            self.paths.boun_treebank_adapted_path,
        ]
        for path in paths_to_load:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                entries.append(json.loads(line))
                            except Exception:
                                continue
            except FileNotFoundError:
                continue
        return entries

    def get_validation_entries(self) -> List[Dict]:
        """Load the validation-set entries (same schema as get_valid_decomps).

        If the adapted JSONL is missing but the raw .connlu exists, run the
        Google-treebank adapter once to materialise it — this keeps the
        validation pipeline idempotent without requiring a manual step.
        """
        adapted_path = self.paths.validation_adapted_path
        conllu_path  = self.paths.validation_conllu_path

        if not os.path.exists(adapted_path) and os.path.exists(conllu_path):
            try:
                from data.google_treebank.treebank_adapter import adapt_treebank
                print(f"Adapting validation set: {conllu_path} -> {adapted_path}")
                adapt_treebank(
                    conllu_path,
                    output_path=adapted_path,
                    stats_path=None,
                    unmatched_path=adapted_path.replace('.jsonl', '_unmatched.jsonl'),
                    unmapped_path=os.path.join(
                        os.path.dirname(adapted_path), 'unmapped_features.json'
                    ),
                )
            except Exception as e:
                print(f"Could not adapt validation conllu: {e}")
                return []

        entries = []
        try:
            with open(adapted_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            continue
        except FileNotFoundError:
            return []
        return entries

    def log_decompositions(self, log_entries: List[Dict]) -> bool:
        try:
            with open(self.paths.valid_decompositions_path, 'a', encoding='utf-8') as f:
                for entry in log_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            return True
        except Exception:
            return False

    def write_decomposed_text(self, text: str) -> bool:
        output_path = self.paths.sample_decomposed_path
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            return True
        except Exception:
            return False
            
    def write_decomposed_sentences(self, text: str) -> bool:
        output_path = getattr(self.paths, 'sample_sentence_decomposed_path', 'sample/sample_sentence_decomposed.txt')
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            return True
        except Exception:
            return False
    
    def delete(self, word: str) -> bool:
        try:
            if wrd.delete_word(word):
                with open(self.paths.words_path, "w", encoding="utf-8") as f:
                    for w in wrd.get_all_words():
                        f.write(w + "\n")
                with open(self.paths.verbs_path, "w", encoding="utf-8") as f:
                    for v in wrd.get_all_verbs():
                        f.write(v + "\n")
                return True
            return False
        except Exception:
            return False

    def log_sentence_decompositions(self, log_entries: List[Dict], original_sentence: str) -> bool:
        try:
            decomposed_str = " ".join([e.get('morphology_string', e['word']) for e in log_entries])
            sentence_entry = {
                'type': 'sentence',
                'original_sentence': original_sentence,
                'decomposed_sentence': decomposed_str,
                'words': log_entries
            }
            with open(self.paths.valid_decompositions_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sentence_entry, ensure_ascii=False) + '\n')
            return True
        except Exception:
            return False