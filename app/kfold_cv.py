"""Generic K-fold cross-validation runner.

Intentionally free of any ML / torch imports. Callers supply the
train-and-evaluate work via a `fold_runner` callable; this module only
handles splitting, bookkeeping, and confidence-interval reporting.

Usage:
    from app.kfold_cv import run_k_fold_cv

    def fold_runner(train_items, val_items, fold_idx):
        # train a fresh model, return dict of metric_name -> float
        ...

    result = run_k_fold_cv(dataset=all_seqs, k=10, fold_runner=fold_runner)
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Sequence


# 95% two-sided critical t-values by degrees of freedom.
# Covers realistic k-fold ranges; falls back to 1.96 (normal approx) for df > 30.
_T_CRIT_95: Dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,  5: 2.571,
    6:  2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042,
}


def _t_crit_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T_CRIT_95:
        return _T_CRIT_95[df]
    if df > 30:
        return 1.96
    # Conservative fallback: nearest higher tabulated df.
    for k in sorted(_T_CRIT_95):
        if k >= df:
            return _T_CRIT_95[k]
    return 1.96


# (train_items, val_items, fold_idx) -> {metric_name: float, ...}
FoldRunner = Callable[[List[Any], List[Any], int], Dict[str, float]]


def k_fold_split(n: int, k: int, seed: int = 42) -> List[List[int]]:
    """Partition range(n) into k disjoint index lists.

    Shuffles indices once (seeded) then round-robins into k buckets so folds
    differ in size by at most 1.
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    if n < k:
        raise ValueError(f"Cannot split {n} items into {k} folds (n < k).")
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    folds: List[List[int]] = [[] for _ in range(k)]
    for i, idx in enumerate(indices):
        folds[i % k].append(idx)
    return folds


def run_k_fold_cv(
    dataset: Sequence[Any],
    k: int,
    fold_runner: FoldRunner,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run K-fold CV and summarise per-metric 95% confidence intervals.

    Args:
        dataset: opaque sequence of items, passed through to `fold_runner`.
        k: number of folds.
        fold_runner: caller-supplied callable that trains on the train slice
            and evaluates on the val slice, returning a dict of floats.
        seed: RNG seed for the split (fixed for reproducibility).
        verbose: print per-fold results and the final summary.

    Returns:
        {
          'folds':   [per-fold metric dicts...],
          'summary': {metric_name: {'mean', 'std', 'ci_low', 'ci_high',
                                    'half_width', 'n'}},
          'k': k,
          'n': len(dataset),
        }
    """
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
    if verbose:
        _print_summary(summary, k)

    return {"folds": per_fold, "summary": summary, "k": k, "n": n}


def _aggregate(per_fold: List[Dict[str, float]], k: int) -> Dict[str, Dict[str, float]]:
    """Per-metric mean, sample stdev, and 95% CI using the t-distribution."""
    if not per_fold:
        return {}

    names = sorted({
        m for d in per_fold for m, v in d.items() if isinstance(v, (int, float))
    })
    t = _t_crit_95(max(k - 1, 1))

    out: Dict[str, Dict[str, float]] = {}
    for name in names:
        values = [
            d[name] for d in per_fold
            if name in d and isinstance(d[name], (int, float))
        ]
        if not values:
            continue
        mean = sum(values) / len(values)
        if len(values) > 1:
            var  = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std  = math.sqrt(var)
            half = t * std / math.sqrt(len(values))
        else:
            std, half = 0.0, 0.0
        out[name] = {
            "mean":       mean,
            "std":        std,
            "ci_low":     mean - half,
            "ci_high":    mean + half,
            "half_width": half,
            "n":          float(len(values)),
        }
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
        print(
            f"  {name:<{name_w}}  mean={s['mean']:+.4f}  ± {s['half_width']:.4f}   "
            f"[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]   "
            f"std={s['std']:.4f}  n={int(s['n'])}"
        )
    print(bar)
