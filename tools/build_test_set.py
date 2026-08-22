#!/usr/bin/env python3
"""Carve a held-out test set out of one treebank.

The project has had no test set since the TRMor2018 gold file was removed for
overlapping the training treebanks, so `test` reports zeros and every number
quoted about the model comes from the validation slice relearn_all() carves
out of its own training data.

This moves whole sentences out of a single treebank shard into
data/test_adapted.jsonl, which DataManager.get_treebank_adapted_paths()
already excludes from training. Moving rather than copying is what makes the
split real: a sentence lives in exactly one file afterwards.

METU is the default source. It is one shard from one adapter, so the format is
uniform, and it is the best behaved of the five: 74% of its sentences are fully
parseable against 39-57% elsewhere, and 80% of its gold analyses are reachable.
A test set on a treebank whose annotations the decomposer cannot represent
measures the annotation, not the model.

Selection is by blake2b of the sentence text, not by position, so membership
survives the corpus being re-adapted, re-ordered or re-deduplicated. Sentences
whose text also appears in another treebank are skipped: those would leak.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.data_manager import DataManager
from data.treebank_adapter_commons import sentence_text

DEFAULT_SOURCE = "data/metu_treebank/treebank_adapted_001.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Treebank shard to carve the test set out of.")
    parser.add_argument("--count", type=int, default=1000, help="Sentences to hold out.")
    parser.add_argument("--min-words", type=int, default=3, help="Skip sentences shorter than this.")
    parser.add_argument("--dry-run", action="store_true", help="Report the split, write nothing.")
    return parser.parse_args()


def selection_key(entry: Dict[str, Any]) -> str:
    """Stable per-sentence key: independent of file order and of neighbours."""
    return hashlib.blake2b(sentence_text(entry).encode("utf-8"), digest_size=16).hexdigest()


def read_entries(path: Path) -> List[Dict[str, Any]]:
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def texts_outside(source: Path) -> Set[str]:
    """Every sentence text the rest of the training corpus carries."""
    data_manager = DataManager()
    others = set()
    for path_str in [data_manager.paths.valid_decompositions_path, *data_manager.get_treebank_adapted_paths()]:
        path = Path(path_str)
        if path.resolve() == source.resolve() or not path.exists():
            continue
        for entry in read_entries(path):
            others.add(sentence_text(entry))
    return others


def choose(entries: List[Dict[str, Any]], count: int, min_words: int, elsewhere: Set[str]) -> Tuple[List[Dict], Dict[str, int]]:
    stats = {"too_short": 0, "duplicate_elsewhere": 0, "duplicate_within": 0, "eligible": 0}
    seen: Set[str] = set()
    eligible = []
    for entry in entries:
        text = sentence_text(entry)
        if len(entry.get("words", [])) < min_words:
            stats["too_short"] += 1
            continue
        if text in elsewhere:
            # Present in another treebank too, so holding it out here would
            # still leave it in training.
            stats["duplicate_elsewhere"] += 1
            continue
        if text in seen:
            stats["duplicate_within"] += 1
            continue
        seen.add(text)
        eligible.append(entry)
    stats["eligible"] = len(eligible)
    eligible.sort(key=selection_key)
    return eligible[:count], stats


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        print(f"{source} not found", file=sys.stderr)
        return 1

    entries = read_entries(source)
    elsewhere = texts_outside(source)
    held_out, stats = choose(entries, args.count, args.min_words, elsewhere)
    if len(held_out) < args.count:
        print(f"only {len(held_out)} sentences are eligible, wanted {args.count}", file=sys.stderr)

    chosen_keys = {selection_key(entry) for entry in held_out}
    remaining = [entry for entry in entries if selection_key(entry) not in chosen_keys]

    test_path = Path(DataManager().paths.test_adapted_path)
    held_words = sum(len(entry.get("words", [])) for entry in held_out)
    kept_words = sum(len(entry.get("words", [])) for entry in remaining)

    print(f"source                 : {source}  ({len(entries)} sentences)")
    print(f"  too short (<{args.min_words} words) : {stats['too_short']}")
    print(f"  also in another shard : {stats['duplicate_elsewhere']}")
    print(f"  repeated within shard : {stats['duplicate_within']}")
    print(f"  eligible              : {stats['eligible']}")
    print(f"held out               : {len(held_out)} sentences, {held_words} words -> {test_path}")
    print(f"left for training      : {len(remaining)} sentences, {kept_words} words")

    if args.dry_run:
        print("\ndry run, nothing written")
        return 0

    with test_path.open("w", encoding="utf-8") as handle:
        for entry in held_out:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    tmp = source.with_suffix(source.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for entry in remaining:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    tmp.replace(source)
    print(f"\nwrote {test_path} and rewrote {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
