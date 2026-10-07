#!/usr/bin/env python3
"""Write the corpus-derived lemmas into data/nouns_derived.txt and verbs_derived.txt.

Reads the report produced by tools/audit_missing_roots.py and takes the roots
the treebanks annotate as lemmas which savyar can already decompose through a
shorter lemma - "bilgi" as bil+gi, "kullan" as kul+la+n, "başla" as baş+la.
Those words are not unanalysable; savyar simply segments past the treebank's
lemma, so the gold analysis is never among the candidates and the ranker is
handed either nothing or a single wrong option.

Adding the lemma does not remove savyar's own reading: it puts the gold reading
beside it so the ranker has a real choice. Each root goes to the file its part
of speech was probed as (see probe_root_pos in the audit).

Merging, never replacing: the roots are unioned into whatever the files already
hold, so re-running after the lexicon has absorbed them is a no-op instead of
truncating them back to empty.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import util.decomposer as sfx
import util.word_methods as wrd
from util.suffix import SuffixGroup
from util.words.closed_class import lexeme_of
from data.treebank_adapter_commons import DECOMPOSED_LEMMAS

DEFAULT_REPORT = REPO_ROOT / "data" / "nonexistant_report.json"

NOUN_FILE = "data/words.txt"
VERB_FILE = "data/verbs.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Audit report to read.")
    parser.add_argument("--min-count", type=int, default=1, help="Only add roots seen at least this often.")
    parser.add_argument(
        "--include-unanalysable",
        action="store_true",
        help="Also add roots savyar cannot decompose at all (off by default: those are unverified).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Rewrite the files from this selection instead of merging into them. "
             "Needed to apply a stricter --min-count; drops anything the selection omits.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written, write nothing.")
    return parser.parse_args()


INFLECTIONAL_GROUPS = {
    SuffixGroup.PLURAL,
    SuffixGroup.POSSESSIVE,
    SuffixGroup.CASE,
    SuffixGroup.MARKING_KI,
    SuffixGroup.WITH_LE,
    SuffixGroup.PREDICATIVE,
    SuffixGroup.CONJUGATION,
}


def is_inflected_closed_class_form(root: str) -> bool:
    key = wrd.lexicon_key(root)
    for analysis_root, pos, chain, _final_pos in sfx.decompose(key):
        if analysis_root == key or pos != "noun" or not lexeme_of(analysis_root):
            continue
        if all(suffix.group in INFLECTIONAL_GROUPS for suffix in chain):
            return True
    return False


def selected_roots(report_path: Path, min_count: int, include_unanalysable: bool) -> List[Dict]:
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    roots = []
    for record in report.get("roots", []):
        if record.get("status") != "absent":
            continue
        if record.get("count", 0) < min_count:
            continue
        if not include_unanalysable and not record.get("root_analysable"):
            continue
        if not record.get("suggested_file"):
            # Nothing places the root: neither the gold chain nor any chain at
            # all carries it across its own surface forms.
            continue
        if is_inflected_closed_class_form(record["root"]):
            continue
        if wrd.lexicon_key(record["root"]) in DECOMPOSED_LEMMAS:
            continue
        roots.append(record)
    return roots


def route(roots: List[Dict]) -> Dict[str, Set[str]]:
    additions: Dict[str, Set[str]] = {"noun": set(), "verb": set()}
    for record in roots:
        target = record["suggested_file"]
        if NOUN_FILE in target:
            additions["noun"].add(record["root"])
        if VERB_FILE in target:
            additions["verb"].add(record["root"])
    return additions


def merge_into(path: Path, roots: Set[str], dry_run: bool, replace: bool = False) -> Dict[str, int]:
    existing = set() if replace else wrd._read_entries(path)
    # A root already carried by the core lexicon under any casing does not
    # belong here a second time.
    folded_existing = {wrd.lexicon_key(entry) for entry in existing}
    new = {root for root in roots if wrd.lexicon_key(root) not in folded_existing}

    merged = sorted(existing | new)
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for entry in merged:
                handle.write(entry + "\n")
    return {"existing": len(existing), "added": len(new), "total": len(merged)}


def main() -> int:
    args = parse_args()
    report_path = Path(args.report)
    if not report_path.exists():
        print(f"{report_path} not found - run tools/audit_missing_roots.py first.", file=sys.stderr)
        return 1

    roots = selected_roots(report_path, args.min_count, args.include_unanalysable)
    additions = route(roots)
    occurrences = sum(record["count"] for record in roots)
    evidence = Counter(record.get("evidence", "?") for record in roots)

    print(f"{len(roots)} roots selected, {occurrences} word occurrences")
    print(f"  part of speech decided by: {', '.join(f'{k} {v}' for k, v in evidence.most_common())}")

    targets = {
        "noun": Path(wrd.DERIVED_DATA_FILE),
        "verb": Path(wrd.DERIVED_VERB_DATA_FILE),
    }
    for kind, path in targets.items():
        stats = merge_into(path, additions[kind], args.dry_run, args.replace)
        verb = "would write" if args.dry_run else "wrote"
        rel = path.relative_to(REPO_ROOT)
        print(f"  {verb} {rel}: {stats['added']} new, {stats['existing']} already there, {stats['total']} total")

    if not args.dry_run:
        wrd._load_dictionary()
        print(f"\nlexicon now: {len(wrd.WORDS_SET)} nouns, {len(wrd.VERB_SET)} verbs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
