"""Stratified k-fold splitting over a pooled metadata frame.

Pure and testable: given a metadata frame and k, return fold index arrays.
Used by scripts/6_evaluate/kfold_cnn.py to cross-validate the fused CNN over
the pooled official train+val rows (the official test split is never touched
here — it stays held out, scored once, by the caller).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DEFAULT_STRATIFY_COL = "shd_moderate_or_greater_flag"
DEFAULT_SEED = 42


def make_folds(
    pool_meta: pd.DataFrame,
    k: int = 5,
    stratify_col: str = DEFAULT_STRATIFY_COL,
    seed: int = DEFAULT_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return `k` (train_idx, heldout_idx) pairs of positional indices into
    `pool_meta` (0..len(pool_meta)-1), stratified on `stratify_col`.

    Each row of `pool_meta` appears in exactly one fold's heldout_idx across
    the returned list, and the fold assignment is deterministic for a fixed
    `seed`. `pool_meta` must be a 0..N-1 reset-index frame (see
    src.dataset_kfold.build_pool_metadata), since the returned arrays are
    positional, not label-based.
    """
    if stratify_col not in pool_meta.columns:
        raise KeyError(f"stratify_col '{stratify_col}' not in pool_meta columns")

    y = pool_meta[stratify_col].to_numpy()
    if pd.isna(y).any():
        raise ValueError(
            f"stratify_col '{stratify_col}' contains NaN; cannot stratify on it"
        )

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    x_dummy = np.zeros(len(pool_meta))
    return [(train_idx, heldout_idx) for train_idx, heldout_idx in skf.split(x_dummy, y)]
