import os
import json
import pickle
import re
from pathlib import Path
from typing import Any, Iterator, List, Optional, Dict, Tuple

from app.file_paths import FilePaths
import util.word_methods as wrd
from util.word_methods import tr_lower

class DataManager:
    def __init__(self):
        self.paths = FilePaths()
        self.last_preprocessed_sequences_cache_path: Optional[Path] = None

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
                    "word_accuracy": float(validation.get("word_acc", 0.0)),
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

    @staticmethod
    def _path_signature(path: Path) -> Dict[str, Any]:
        normalized = str(path)
        try:
            stat = path.stat()
        except FileNotFoundError:
            return {"path": normalized, "exists": False}
        return {
            "path": normalized,
            "exists": True,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def get_preprocess_source_signature(self, *, include_code: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        entry_paths = [
            Path(self.paths.valid_decompositions_path),
            *[Path(path) for path in self.get_treebank_adapted_paths()],
        ]
        dependency_paths = [
            Path(self.paths.words_path),
            Path(self.paths.verbs_path),
            Path(self.paths.unsuffixable_words_path),
        ]
        code_paths = sorted({
            Path("app/nlp_pipeline.py"),
            Path("ml/ml_ranking_model.py"),
            Path("util/decomposer.py"),
            Path("util/suffix.py"),
            Path("util/word_methods.py"),
            *Path("util/words").rglob("*.py"),
            *Path("util/suffixes").rglob("*.py"),
        })
        signature = {
            "entries": [self._path_signature(path) for path in entry_paths],
            "dependencies": [self._path_signature(path) for path in dependency_paths],
        }
        if include_code:
            signature["code"] = [self._path_signature(path) for path in code_paths]
        return signature

    def preprocessed_sequences_cache_path(self, cache_key: str) -> Path:
        return Path(self.paths.preprocessed_sequences_cache_dir) / f"{cache_key}.pkl"

    def load_preprocessed_sequences_cache(
        self,
        cache_key: str,
        expected_metadata: Dict[str, Any],
    ) -> Optional[Tuple[List[List[Any]], int, int]]:
        path = self.preprocessed_sequences_cache_path(cache_key)
        self.last_preprocessed_sequences_cache_path = None
        loaded = self._load_preprocessed_sequences_cache_path(path, expected_metadata, exact=True)
        if loaded is not None:
            self.last_preprocessed_sequences_cache_path = path
            return loaded

        cache_dir = Path(self.paths.preprocessed_sequences_cache_dir)
        try:
            candidates = sorted(
                cache_dir.glob("*.pkl"),
                key=lambda candidate: candidate.stat().st_mtime_ns,
                reverse=True,
            )
        except OSError:
            return None

        for candidate in candidates:
            if candidate == path:
                continue
            loaded = self._load_preprocessed_sequences_cache_path(candidate, expected_metadata, exact=False)
            if loaded is not None:
                self.last_preprocessed_sequences_cache_path = candidate
                return loaded
        return None

    @staticmethod
    def _metadata_compatible(actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        for key in (
            "cache_version",
            "scope",
            "suffix_inventory",
            "root_inventory",
            "closed_class_inventory",
            "config",
            "entries",
        ):
            if actual.get(key) != expected.get(key):
                return False

        actual_sources = actual.get("sources", {})
        expected_sources = expected.get("sources", {})
        for key in ("entries", "dependencies"):
            if actual_sources.get(key) != expected_sources.get(key):
                return False
        return True

    def _load_preprocessed_sequences_cache_path(
        self,
        path: Path,
        expected_metadata: Dict[str, Any],
        *,
        exact: bool,
    ) -> Optional[Tuple[List[List[Any]], int, int]]:
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except FileNotFoundError:
            return None
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return None
        if exact:
            if metadata != expected_metadata:
                return None
        elif not self._metadata_compatible(metadata, expected_metadata):
            return None

        all_seqs = payload.get("all_seqs")
        total_words = payload.get("total_words")
        skipped = payload.get("skipped")
        if not isinstance(all_seqs, list) or not isinstance(total_words, int) or not isinstance(skipped, int):
            return None
        return all_seqs, total_words, skipped

    def save_preprocessed_sequences_cache(
        self,
        metadata: Dict[str, Any],
        all_seqs: List[List[Any]],
        total_words: int,
        skipped: int,
    ) -> bool:
        cache_key = str(metadata.get("cache_key", ""))
        if not cache_key:
            return False
        path = self.preprocessed_sequences_cache_path(cache_key)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "metadata": metadata,
            "all_seqs": all_seqs,
            "total_words": total_words,
            "skipped": skipped,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
            return True
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            return False

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
