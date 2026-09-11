"""Combinatorial Purged Cross-Validation folds. Purge = drop training points
within `embargo_days` of any test fold so overlapping labels cannot bridge train and test."""
from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd

def make_folds(index: pd.DatetimeIndex, n_folds: int = 10) -> list[pd.DatetimeIndex]:
    boundaries = np.array_split(np.arange(len(index)), n_folds)
    return [index[b] for b in boundaries]

def cpcv_combos(n_folds: int = 10, k: int = 2) -> list[tuple[int, ...]]:
    return list(combinations(range(n_folds), k))

def purged_train_index(all_index: pd.DatetimeIndex,
                       test_folds: list[pd.DatetimeIndex],
                       embargo_days: int) -> pd.DatetimeIndex:
    banned = set()
    for tf in test_folds:
        banned.update(tf)
        lo = tf.min() - pd.Timedelta(days=embargo_days)
        hi = tf.max() + pd.Timedelta(days=embargo_days)
        for d in all_index:
            if lo <= d <= hi:
                banned.add(d)
    keep = [d for d in all_index if d not in banned]
    return pd.DatetimeIndex(keep)
