"""Cross-file dataset indexing for k-fold CV over the pooled train+val rows.

The official EchoNext splits ship as two separate memory-mapped waveform
files (train, val). To run stratified k-fold CV over their union without
copying either 16GB/1GB array into RAM, `ConcatSplitDataset` holds both
underlying `ECGDataset`s (each backed by its own mmap) and maps a global pool
index to (file, row) lazily, per `__getitem__` call.

Demographic encoding must be refit per fold on the in-fold training rows only
(refitting on the whole pool would leak heldout-fold/test rows' statistics
into the encoder). `fit_fold_demographic_encoder` / `transform_demo` do that
refit without touching `src/dataset.py` or `src/preprocessing.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.constants import CATEGORICAL_DEMOGRAPHIC_FEATURES, NUMERIC_DEMOGRAPHIC_FEATURES
from src.dataset import ECGDataset, SplitData
from src.preprocessing import build_demographic_preprocessor

DEMO_COLUMNS = NUMERIC_DEMOGRAPHIC_FEATURES + CATEGORICAL_DEMOGRAPHIC_FEATURES


def build_pool_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Concatenate the official `train` + `val` metadata rows, train first,
    each in its own file's row order (the same convention `src.dataset.
    load_split` relies on and checks via its row-count assertion).

    For the returned frame `pool`, row i with i < n_train corresponds to row
    i of the train waveform array; row i with i >= n_train corresponds to row
    (i - n_train) of the val waveform array. Callers that need `n_train` can
    recompute it as `(metadata["split"] == "train").sum()`.
    """
    train_meta = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_meta = metadata[metadata["split"] == "val"].reset_index(drop=True)
    return pd.concat([train_meta, val_meta], ignore_index=True)


class ConcatSplitDataset(Dataset):
    """Logical concatenation of two `SplitData` splits (e.g. official train +
    val), indexed as one dataset without copying either's waveform array.

    Global index i < len(first) routes to `first`; i >= len(first) routes to
    `second` at local row (i - len(first)). Meant to be wrapped in
    `torch.utils.data.Subset` with fold-specific index arrays from
    `src.crossval.make_folds`.
    """

    def __init__(self, first: SplitData, second: SplitData):
        if first.label_names != second.label_names:
            raise ValueError("first and second SplitData must share label_names")
        self._first = ECGDataset(first)
        self._second = ECGDataset(second)
        self.first_len = len(self._first)
        self.total_len = self.first_len + len(self._second)

    def __len__(self) -> int:
        return self.total_len

    def file_and_row(self, idx: int) -> tuple[str, int]:
        """Global pool index -> ("first" | "second", local row index)."""
        if idx < 0 or idx >= self.total_len:
            raise IndexError(f"index {idx} out of range for dataset of length {self.total_len}")
        if idx < self.first_len:
            return "first", idx
        return "second", idx - self.first_len

    def __getitem__(self, idx: int):
        which, row = self.file_and_row(idx)
        return self._first[row] if which == "first" else self._second[row]


def fit_fold_demographic_encoder(pool_meta: pd.DataFrame, train_idx: np.ndarray):
    """Fit a fresh demographic ColumnTransformer on the in-fold training rows
    only. Fitting on the whole pool (or on pool rows outside train_idx) would
    leak heldout-fold / downstream-test statistics into imputation medians,
    the StandardScaler and the OneHotEncoder's category set.
    """
    encoder = build_demographic_preprocessor()
    encoder.fit(pool_meta.iloc[train_idx].reindex(columns=DEMO_COLUMNS))
    return encoder


def transform_demo(encoder, meta: pd.DataFrame) -> np.ndarray:
    """Transform any metadata frame's demographic columns with a
    (fold-specific) fitted encoder."""
    return encoder.transform(meta.reindex(columns=DEMO_COLUMNS)).astype(np.float32)
