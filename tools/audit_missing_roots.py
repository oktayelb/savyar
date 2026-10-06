#!/usr/bin/env python3
"""One-off audit: every gold root that Savyar's lexicon cannot resolve.

The decomposer can only reach a gold analysis if the gold root is reachable as
a lemma, i.e. it is in data/words.txt, data/verbs.txt, data/ekistemez.txt or the
closed-class inventory. Every root that is not is a hard ceiling on candidate
coverage: no amount of ranking can recover a word whose gold analysis was never
generated.

The audit walks the same corpus the trainer reads (sentence_valid_decompositions
plus every treebank_adapted shard), collects the roots that fail that lookup and
writes them to data/nonexistant.txt, one root per line, most frequent first.
A companion JSON report records, per root, how often it occurs, where it comes
from, example gold words and which lexicon file it belongs in.

The target file is decided by replaying the gold suffix chain through
find_suffix_chain() from a noun start and from a verb start: whichever start
reproduces the chain is the part of speech the root has to be entered as. A root
that only ever appears bare is attributed by its gold final_pos instead.

Roots flagged "case_mismatch" already exist in the lexicon under a different
case (words.txt stores proper nouns capitalised while the pipeline tr_lower()s
everything). Those are not missing entries and must not be appended; they need
the lexicon loader to fold case instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.data_manager import DataManager
import app.nlp_pipeline as nlp
import util.decomposer as sfx
import util.word_methods as wrd
from util.decomposer import find_suffix_chain
from util.word_methods import tr_lower
from util.words.closed_class import lexeme_of

DEFAULT_OUTPUT = REPO_ROOT / "data" / "nonexistant.txt"
DEFAULT_REPORT = REPO_ROOT / "data" / "nonexistant_report.json"

NOUN_FILE = "data/words.txt"
VERB_FILE = "data/verbs.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Plain root list to write.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Detailed JSON report to write.")
    parser.add_argument("--min-count", type=int, default=1, help="Only report roots seen at least this often.")
    parser.add_argument("--examples", type=int, default=3, help="Gold examples kept per root.")
    parser.add_argument("--include-test", action="store_true", help="Also scan FilePaths.test_adapted_path.")
    parser.add_argument("--no-pos-probe", action="store_true", help="Skip the noun/verb replay (faster, no target file).")
    parser.add_argument("--no-decompose-probe", action="store_true", help="Skip running the decomposer on each root and example.")
    parser.add_argument("--top", type=int, default=25, help="Rows printed in the stdout summary.")
    return parser.parse_args()


def corpus_paths(data_manager: DataManager, include_test: bool) -> List[str]:
    paths = [data_manager.paths.valid_decompositions_path, *data_manager.get_treebank_adapted_paths()]
    if include_test:
        paths.extend(str(path) for path in data_manager._jsonl_shards_for(Path(data_manager.paths.test_adapted_path)))
    return [path for path in paths if Path(path).exists()]


def iter_word_entries(paths: Sequence[str]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield (source_path, word_entry) for every annotated word in the corpus."""
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                words = entry.get("words", []) if entry.get("type") == "sentence" else [entry]
                for word_entry in words:
                    if isinstance(word_entry, dict):
                        yield path, word_entry


def is_lexicon_candidate(root: str) -> bool:
    """Could this string plausibly be a lexicon entry at all?

    Letters (including the "ł" soft-l marker words.txt uses and the accented
    letters of foreign names) and the hyphen of compound proper nouns are
    allowed. Everything else - "*unknown*", "dr.", "1996da", "hayat(ı)" - is an
    annotation defect rather than a missing word.
    """
    return bool(root) and all(char.isalpha() or char == "-" for char in root)


def is_resolvable(root: str) -> bool:
    """Can the decomposer reach this root as a lemma today?"""
    return bool(
        wrd.exists(root)
        or wrd.is_unsuffixable(root)
        or lexeme_of(root) is not None
    )


def build_case_folded_index() -> Dict[str, List[str]]:
    """tr_lower(entry) -> lexicon entries, so case-only misses are cheap to spot."""
    index: Dict[str, List[str]] = defaultdict(list)
    for entry in (wrd.WORDS_SET | wrd.VERB_SET):
        index[tr_lower(entry)].append(entry)
    return index


def lexicon_case_variants(root: str, folded_index: Dict[str, List[str]]) -> List[str]:
    """Lexicon entries that differ from the root only by case."""
    return sorted(entry for entry in folded_index.get(tr_lower(root), []) if entry != root)


def chain_is_reachable(root: str, word: str, suffix_names: Sequence[str], start_pos: str) -> bool:
    """Would find_suffix_chain() rebuild this gold chain if the root existed?"""
    # The gold root is a lemma, so the surface may differ from it (hâl/hal,
    # kitab/kitap). Splice the lemma onto the surface remainder the way
    # decompose() does for its own root candidates.
    surface = word if word.startswith(root) else root + word[len(root):]
    try:
        chains = find_suffix_chain(surface, start_pos, root)
    except Exception:
        return False
    target = tuple(suffix_names)
    return any(tuple(suffix.name for suffix in chain) == target for chain, _final_pos in chains)


def surface_is_spannable(root: str, word: str, start_pos: str) -> bool:
    """Can any suffix chain at all carry this root across the surface?"""
    surface = word if word.startswith(root) else root + word[len(root):]
    try:
        return bool(find_suffix_chain(surface, start_pos, root))
    except Exception:
        return False


def probe_root_pos(root: str, examples: Sequence[Dict[str, Any]]) -> Tuple[List[str], str]:
    """Which part(s) of speech this root has to be entered as, and on what evidence.

    Three tiers, strongest first:
      "chain"    - a start POS replays the gold suffix chain exactly.
      "spanning" - no start replays the gold chain (the annotation drops a
                   suffix, or the suffix table lacks an allomorph) but one of
                   them can still carry the root across the surface. Weaker,
                   and the only evidence available for verbs like "bekle",
                   whose gold "bekle l en" omits nothing the tables can rebuild.
      "final_pos"- the root only ever appears bare, so the annotation decides.
    """
    votes: set = set()
    bare_votes: set = set()
    for example in examples:
        suffix_names = example["suffixes"]
        if not suffix_names:
            # A bare root carries no morphological evidence; its final_pos is
            # the root's own part of speech.
            final_pos = example.get("final_pos")
            if final_pos in ("noun", "verb"):
                bare_votes.add(final_pos)
            continue
        for start_pos in ("noun", "verb"):
            if chain_is_reachable(root, example["word"], suffix_names, start_pos):
                votes.add(start_pos)
    if votes:
        return sorted(votes), "chain"
    if bare_votes:
        return sorted(bare_votes), "final_pos"

    for example in examples:
        if not example["suffixes"]:
            continue
        for start_pos in ("noun", "verb"):
            if surface_is_spannable(root, example["word"], start_pos):
                votes.add(start_pos)
    if votes:
        return sorted(votes), "spanning"
    return [], "unreplayable"


def probe_decomposition(root: str, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """What savyar actually does with this root and with the words built on it.

    A root missing from the lexicon is not necessarily a word savyar cannot
    analyse: most of the frequent ones are derived stems it reaches through a
    shorter lemma ("kullan" as kul+la+n, "çalış" as çal+ış). Typing such a root
    in returns a decomposition, just not the treebank's segmentation - which is
    a different problem from a word that produces nothing at all, and needs a
    different judgement call about whether the lemma belongs in the lexicon.
    """
    root_decomps = sfx.decompose_with_fallback(root)
    outcomes: Counter = Counter()
    for example in examples:
        word = example.get("word") or ""
        if not word:
            continue
        decomps = sfx.decompose_with_fallback(word)
        gold = [{"root": root, "suffixes": [{"name": name} for name in example["suffixes"]]}]
        if nlp.match_decompositions(gold, decomps):
            outcomes["gold_produced"] += 1
        elif not decomps:
            outcomes["no_candidates"] += 1
        elif len(decomps) == 1:
            # One candidate and it is wrong: training drops the word for want
            # of a gold to rank against, and evaluation scores the set correct
            # because there was nothing else to pick.
            outcomes["single_wrong_candidate"] += 1
        else:
            outcomes["candidates_without_gold"] += 1
    return {
        "root_analysable": bool(root_decomps),
        "root_analysis": nlp.format_detailed_decomp(root_decomps[0]) if root_decomps else None,
        "word_outcome": outcomes.most_common(1)[0][0] if outcomes else "no_examples",
        "word_outcomes": dict(outcomes),
    }


def suggested_file(root_pos: Sequence[str]) -> Optional[str]:
    if root_pos == ["verb"]:
        return VERB_FILE
    if root_pos == ["noun"]:
        return NOUN_FILE
    if root_pos:
        return f"{NOUN_FILE} + {VERB_FILE}"
    return None


def collect(paths: Sequence[str], max_examples: int) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    counts: Counter = Counter()
    sources: Dict[str, Counter] = defaultdict(Counter)
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_examples: Dict[str, set] = defaultdict(set)
    stats = Counter()

    for path, word_entry in iter_word_entries(paths):
        stats["gold_words"] += 1
        root = word_entry.get("root")
        if not root:
            stats["no_root"] += 1
            continue
        if not any(char.isalpha() for char in root):
            # Numbers and punctuation are not lexicon entries.
            stats["non_alphabetic_root"] += 1
            continue
        if is_resolvable(root):
            stats["resolvable"] += 1
            continue

        stats["unresolvable"] += 1
        counts[root] += 1
        sources[root][path] += 1
        suffix_names = nlp._normalize_entry_suffix_names(word_entry.get("suffixes", []) or [])
        signature = (word_entry.get("word"), tuple(suffix_names))
        if len(examples[root]) < max_examples and signature not in seen_examples[root]:
            seen_examples[root].add(signature)
            examples[root].append({
                "word": word_entry.get("word"),
                "suffixes": suffix_names,
                "final_pos": word_entry.get("final_pos"),
                "morphology_string": word_entry.get("morphology_string"),
            })

    records = {
        root: {
            "root": root,
            "count": count,
            "sources": dict(sources[root]),
            "examples": examples[root],
        }
        for root, count in counts.items()
    }
    return records, dict(stats)


def classify(records: Dict[str, Dict[str, Any]], pos_probe: bool, decompose_probe: bool) -> None:
    folded_index = build_case_folded_index()
    for record in records.values():
        root = record["root"]
        variants = lexicon_case_variants(root, folded_index)
        record["lexicon_case_variants"] = variants
        if not is_lexicon_candidate(root):
            record["status"] = "malformed"
        elif variants:
            record["status"] = "case_mismatch"
        else:
            record["status"] = "absent"

        if record["status"] == "malformed":
            record["root_pos"] = []
            record["probe"] = "skipped"
            record["evidence"] = "skipped"
            record["suggested_file"] = None
            record["root_analysable"] = None
            record["word_outcome"] = "skipped"
            continue

        if decompose_probe:
            record.update(probe_decomposition(root, record["examples"]))
        else:
            record["root_analysable"] = None
            record["word_outcome"] = "skipped"

        if not pos_probe:
            record["root_pos"] = []
            record["probe"] = "skipped"
            record["evidence"] = "skipped"
            record["suggested_file"] = None
            continue

        record["root_pos"], record["evidence"] = probe_root_pos(root, record["examples"])
        record["suggested_file"] = suggested_file(record["root_pos"])
        # Nothing places the root: the annotation is unreplayable and no chain
        # spans the surface either, so adding the root alone would not recover
        # the word.
        record["probe"] = "unreplayable" if not record["root_pos"] else "+".join(record["root_pos"])


def write_outputs(
    ordered: List[Dict[str, Any]],
    stats: Dict[str, int],
    output_path: Path,
    report_path: Path,
    paths: Sequence[str],
    min_count: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    listed = [record for record in ordered if record["status"] != "malformed"]
    with output_path.open("w", encoding="utf-8") as handle:
        for record in listed:
            handle.write(record["root"] + "\n")

    summary = {
        "gold_words": stats.get("gold_words", 0),
        "resolvable_roots": stats.get("resolvable", 0),
        "unresolvable_roots": stats.get("unresolvable", 0),
        "non_alphabetic_roots": stats.get("non_alphabetic_root", 0),
        "words_without_root": stats.get("no_root", 0),
        "distinct_unresolvable": len(ordered),
        "written_to_root_list": len(listed),
        "reported_occurrences": sum(record["count"] for record in ordered),
        "by_status": dict(Counter(record["status"] for record in ordered)),
        "by_probe": dict(Counter(record.get("probe", "skipped") for record in ordered)),
        "by_evidence": dict(Counter(record.get("evidence", "skipped") for record in ordered)),
        "by_root_analysable": dict(Counter(str(record.get("root_analysable")) for record in ordered)),
        "by_word_outcome": dict(Counter(record.get("word_outcome", "skipped") for record in ordered)),
        "by_suggested_file": dict(Counter(str(record["suggested_file"]) for record in ordered)),
        "min_count": min_count,
        "sources_scanned": list(paths),
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "roots": ordered}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def print_summary(ordered: List[Dict[str, Any]], stats: Dict[str, int], top: int) -> None:
    gold_words = stats.get("gold_words", 0) or 1
    unresolvable = stats.get("unresolvable", 0)
    print(f"gold words scanned      : {stats.get('gold_words', 0)}")
    print(f"roots the lexicon knows : {stats.get('resolvable', 0)} ({stats.get('resolvable', 0) / gold_words:.1%})")
    print(f"roots it does not       : {unresolvable} ({unresolvable / gold_words:.1%})")
    print(f"non-alphabetic roots    : {stats.get('non_alphabetic_root', 0)} (numbers, punctuation; not counted)")
    print(f"distinct missing roots  : {len(ordered)}")

    by_status = Counter(record["status"] for record in ordered)
    case_occurrences = sum(record["count"] for record in ordered if record["status"] == "case_mismatch")
    malformed_occurrences = sum(record["count"] for record in ordered if record["status"] == "malformed")
    print(f"  absent from lexicon   : {by_status.get('absent', 0)}")
    print(f"  case mismatch only    : {by_status.get('case_mismatch', 0)} roots / {case_occurrences} occurrences")
    print(f"  malformed annotations : {by_status.get('malformed', 0)} roots / {malformed_occurrences} occurrences (not listed)")

    print()
    print("where each root belongs:")
    by_file = Counter(
        str(record["suggested_file"]) if record["suggested_file"] else record.get("probe", "skipped")
        for record in ordered
    )
    for target, count in by_file.most_common():
        print(f"  {target:<28} {count}")

    absent = [record for record in ordered if record["status"] == "absent"]
    if absent and absent[0].get("root_analysable") is not None:
        absent_occurrences = sum(record["count"] for record in absent) or 1
        derived = sum(record["count"] for record in absent if record["root_analysable"])
        print()
        print("what savyar does with the root itself (by occurrence):")
        print(f"  reaches it through a shorter lemma  {derived:>8} ({derived / absent_occurrences:.1%})")
        print(f"  cannot analyse it at all            {absent_occurrences - derived:>8} ({1 - derived / absent_occurrences:.1%})")

        outcome_occurrences: Counter = Counter()
        for record in absent:
            outcomes = record.get("word_outcomes") or {}
            total = sum(outcomes.values())
            for outcome, hits in outcomes.items():
                outcome_occurrences[outcome] += record["count"] * hits / total
        print()
        print("what happens to the words built on it (by occurrence):")
        for outcome, weight in outcome_occurrences.most_common():
            print(f"  {outcome:<34} {weight:>8.0f} ({weight / absent_occurrences:.1%})")

    print()
    print(f"top {top} by occurrence:")
    for record in ordered[:top]:
        target = record["suggested_file"] or record.get("probe", "skipped")
        analysis = record.get("root_analysis") or "no analysis"
        print(f"  {record['count']:>6}  {record['root']:<14} {target:<24} savyar reads it as: {analysis}")


def main() -> int:
    args = parse_args()
    data_manager = DataManager()
    paths = corpus_paths(data_manager, args.include_test)
    if not paths:
        print("No corpus files found.", file=sys.stderr)
        return 1

    records, stats = collect(paths, args.examples)
    classify(records, pos_probe=not args.no_pos_probe, decompose_probe=not args.no_decompose_probe)
    ordered = sorted(
        (record for record in records.values() if record["count"] >= args.min_count),
        key=lambda record: (-record["count"], record["root"]),
    )

    output_path = Path(args.output)
    report_path = Path(args.report)
    write_outputs(ordered, stats, output_path, report_path, paths, args.min_count)
    print_summary(ordered, stats, args.top)
    listed = sum(1 for record in ordered if record["status"] != "malformed")
    print()
    print(f"wrote {output_path} ({listed} roots)")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
