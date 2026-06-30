"""Dataset loading and preprocessing for the ECG CNN pipelines.

Waveform layout in the .npy files: (N, 1, 2500, 12) at 250 Hz, 10 seconds, 12 leads.
This module reshapes them to (N, 12, 2500) for Conv1d (leads as channels).

Demographic features (age, sex, race, location) are encoded with a sklearn
ColumnTransformer that is fit on the training split and reused for val/test.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset

from src.constants import (
    DEMOGRAPHIC_CATEGORICAL_COLS,
    DEMOGRAPHIC_NUMERIC_COLS,
    TARGET_LABELS,
)


@dataclass
class SplitData:
    waveforms: np.ndarray       # (N, 12, 2500) float32
    labels: np.ndarray          # (N, n_labels) float32, may contain NaN
    demo_features: np.ndarray   # (N, n_demo) float32, no NaN; shape (N, 0) if unused
    label_names: list[str]


class ECGDataset(Dataset):
    """PyTorch dataset wrapping waveforms, optional demographic features, and labels.

    Each item is a 4-tuple:
      (waveform, demo, label, valid_mask)
    where valid_mask is True for labels that are not NaN (those contribute to the loss).
    If demographics were not included, demo has shape (0,).
    """

    def __init__(self, data: SplitData):
        self.waveforms = data.waveforms.astype(np.float32)
        self.demo = data.demo_features.astype(np.float32)
        raw_labels = data.labels.astype(np.float32)
        self.valid_mask = ~np.isnan(raw_labels)
        self.labels = np.where(self.valid_mask, raw_labels, 0.0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int):
        wave = torch.from_numpy(self.waveforms[idx])
        demo = torch.from_numpy(self.demo[idx])
        label = torch.from_numpy(self.labels[idx])
        mask = torch.from_numpy(self.valid_mask[idx])
        return wave, demo, label, mask


def build_demo_encoder() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, DEMOGRAPHIC_NUMERIC_COLS),
        ("cat", categorical_pipe, DEMOGRAPHIC_CATEGORICAL_COLS),
    ])


def _extract_labels(metadata: pd.DataFrame, label_names: list[str]) -> np.ndarray:
    available = [c for c in label_names if c in metadata.columns]
    labels = metadata[available].values.astype(np.float32)
    return labels


def _load_waveforms(dataset_dir: Path, split: str) -> np.ndarray:
    path = dataset_dir / f"EchoNext_{split}_waveforms.npy"
    if not path.exists():
        raise FileNotFoundError(f"Waveform file not found: {path}")
    raw = np.load(path)  # (N, 1, 2500, 12)
    # reshape to (N, 12, 2500): move leads to channel dim, drop the size-1 dim
    reshaped = raw[:, 0, :, :].transpose(0, 2, 1)  # (N, 12, 2500)
    return reshaped.astype(np.float32)


def load_split(
    split: str,
    metadata: pd.DataFrame,
    dataset_dir: Path,
    label_names: list[str] = TARGET_LABELS,
    demo_encoder: ColumnTransformer | None = None,
    fit_encoder: bool = False,
) -> tuple[SplitData, ColumnTransformer | None]:
    """Load waveforms, labels, and optionally demographic features for one split.

    Args:
        split: "train", "val", or "test".
        metadata: full metadata DataFrame (all splits); filtered by split column here.
        dataset_dir: path to the EchoNext dataset directory.
        label_names: which target columns to extract.
        demo_encoder: fitted sklearn ColumnTransformer, or None to skip demographics.
        fit_encoder: if True, fit demo_encoder on this split's data (use only for train).

    Returns:
        (SplitData, fitted_encoder_or_None)
    """
    split_meta = metadata[metadata["split"] == split].reset_index(drop=True)

    waveforms = _load_waveforms(dataset_dir, split)
    labels = _extract_labels(split_meta, label_names)

    if len(split_meta) != len(waveforms):
        raise ValueError(
            f"Row count mismatch for split '{split}': "
            f"metadata has {len(split_meta)} rows, waveforms have {len(waveforms)} rows."
        )

    if demo_encoder is not None:
        # reindex ensures all expected columns are present; missing ones become NaN
        # and are handled by the imputers inside the ColumnTransformer
        expected_cols = DEMOGRAPHIC_NUMERIC_COLS + DEMOGRAPHIC_CATEGORICAL_COLS
        demo_df = split_meta.reindex(columns=expected_cols)

        if fit_encoder:
            demo_features = demo_encoder.fit_transform(demo_df).astype(np.float32)
        else:
            demo_features = demo_encoder.transform(demo_df).astype(np.float32)
    else:
        demo_features = np.empty((len(waveforms), 0), dtype=np.float32)

    print(
        f"  {split}: {len(waveforms):,} samples, "
        f"waveforms {waveforms.shape[1:]}, "
        f"demo features {demo_features.shape[1]}, "
        f"labels {labels.shape[1]}"
    )

    data = SplitData(
        waveforms=waveforms,
        labels=labels,
        demo_features=demo_features,
        label_names=label_names,
    )
    return data, demo_encoder
