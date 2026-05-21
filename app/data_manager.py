import os
import json
import re
from pathlib import Path
from typing import Iterator, List, Optional, Dict

from app.file_paths import FilePaths
import util.word_methods as wrd
from util.word_methods import tr_lower

class DataManager:
    def __init__(self):
        self.paths = FilePaths()

    @staticmethod
    def _numbered_jsonl_key(path: Path) -> int:
        match = re.search(r"_(\d+)\.jsonl$", path.name)
        return int(match.group(1)) if match else 0

    @classmethod
    def _jsonl_shards_for(cls, base_path: Path) -> List[Path]:
        parent = base_path.parent
        stem = base_path.stem
        numbered_re = re.compile(rf"^{re.escape(stem)}_\d+\.jsonl$")
        numbered = [
            path for path in parent.glob(f"{stem}_*.jsonl")
            if numbered_re.fullmatch(path.name)
        ]
        if numbered:
            return sorted(numbered, key=lambda path: (cls._numbered_jsonl_key(path), path.name))
        return [base_path] if base_path.exists() else []

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

    def get_treebank_adapted_paths(self) -> List[str]:
        data_dir = Path(self.paths.data_dir)
        if not data_dir.exists():
            return []

        test_path = Path(self.paths.test_adapted_path)
        try:
            test_path = test_path.resolve()
        except OSError:
            pass

        treebank_dirs = set()
        adapted_name_re = re.compile(r"^treebank_adapted(?:_\d+)?\.jsonl$")
        for path in data_dir.rglob("treebank_adapted*.jsonl"):
            if adapted_name_re.fullmatch(path.name):
                treebank_dirs.add(path.parent)

        treebank_paths = []
        for parent in sorted(treebank_dirs):
            for path in self._jsonl_shards_for(parent / "treebank_adapted.jsonl"):
                try:
                    if path.resolve() == test_path:
                        continue
                except OSError:
                    pass
                treebank_paths.append(str(path))
        return treebank_paths

    def get_valid_decomps(self) -> List[Dict]:
        return list(self.iter_valid_decomps())

    def iter_valid_decomps(self) -> Iterator[Dict]:
        paths_to_load = [
            self.paths.valid_decompositions_path,
            *self.get_treebank_adapted_paths(),
        ]
        for path in paths_to_load:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                yield json.loads(line)
                            except Exception:
                                continue
            except FileNotFoundError:
                continue

    def get_test_entries(self) -> List[Dict]:
        """Load the adapted TRMor2018 gold test JSONL."""
        entries = []
        for path in self._jsonl_shards_for(Path(self.paths.test_adapted_path)):
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
