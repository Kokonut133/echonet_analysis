from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from torch.utils.data import Dataset

from src.constants import (
    CATEGORICAL_DEMOGRAPHIC_FEATURES,
    NUMERIC_DEMOGRAPHIC_FEATURES,
    TARGET_LABELS,
)
from src.preprocessing import build_demographic_preprocessor


@dataclass
class SplitData:
    waveforms: np.ndarray       # (N, 1, 2500, 12) mmap or ndarray; reshaped lazily in ECGDataset
    labels: np.ndarray          # (N, n_labels) float32, may contain NaN
    demo_features: np.ndarray   # (N, n_demo) float32; shape (N, 0) if unused
    label_names: list[str]


class ECGDataset(Dataset):
    def __init__(self, data: SplitData):
        self.waveforms = data.waveforms
        self._waveforms_are_raw = (
            data.waveforms.ndim == 4 and data.waveforms.shape[1] == 1
        )
        self.demo = np.asarray(data.demo_features, dtype=np.float32)
        raw_labels = np.asarray(data.labels, dtype=np.float32)
        self.valid_mask = ~np.isnan(raw_labels)
        self.labels = np.where(self.valid_mask, raw_labels, 0.0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.waveforms)

    def _read_waveform(self, idx: int) -> np.ndarray:
        if self._waveforms_are_raw:
            sample = self.waveforms[idx, 0, :, :]  # (2500, 12)
            return np.ascontiguousarray(sample.T, dtype=np.float32)  # (12, 2500)
        return np.asarray(self.waveforms[idx], dtype=np.float32)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self._read_waveform(idx)),
            torch.from_numpy(self.demo[idx]),
            torch.from_numpy(self.labels[idx]),
            torch.from_numpy(self.valid_mask[idx]),
        )


build_demo_encoder = build_demographic_preprocessor


def _extract_labels(df: pd.DataFrame, label_names: list[str]) -> np.ndarray:
    available = [c for c in label_names if c in df.columns]
    return df[available].values.astype(np.float32)


def _load_waveforms(dataset_dir: Path, split: str, mmap: bool = True) -> np.ndarray:
    path = dataset_dir / f"EchoNext_{split}_waveforms.npy"
    if not path.exists():
        raise FileNotFoundError(f"Waveform file not found: {path}")
    size_gb = path.stat().st_size / 1e9
    mode = "r" if mmap else None
    print(
        f"  Opening {path.name} ({size_gb:.1f} GB, "
        f"{'memory-mapped' if mmap else 'in-memory'})...",
        end=" ",
        flush=True,
    )
    waveforms = np.load(path, mmap_mode=mode)
    print("done")
    return waveforms


def load_split(
    split: str,
    metadata: pd.DataFrame,
    dataset_dir: Path,
    label_names: list[str] = TARGET_LABELS,
    demo_encoder: ColumnTransformer | None = None,
    fit_encoder: bool = False,
    mmap_waveforms: bool = True,
) -> tuple[SplitData, ColumnTransformer | None]:
    split_meta = metadata[metadata["split"] == split].reset_index(drop=True)

    waveforms = _load_waveforms(dataset_dir, split, mmap=mmap_waveforms)
    labels = _extract_labels(split_meta, label_names)

    if len(split_meta) != len(waveforms):
        raise ValueError(
            f"Row count mismatch for split '{split}': "
            f"metadata has {len(split_meta)} rows, waveforms have {len(waveforms)} rows."
        )

    if demo_encoder is not None:
        expected_cols = NUMERIC_DEMOGRAPHIC_FEATURES + CATEGORICAL_DEMOGRAPHIC_FEATURES
        demo_df = split_meta.reindex(columns=expected_cols)
        if fit_encoder:
            print("  Fitting demographic encoder...", end=" ", flush=True)
            demo_features = demo_encoder.fit_transform(demo_df).astype(np.float32)
            print("done")
        else:
            print("  Transforming demographic features...", end=" ", flush=True)
            demo_features = demo_encoder.transform(demo_df).astype(np.float32)
            print("done")
    else:
        demo_features = np.empty((len(waveforms), 0), dtype=np.float32)

    waveform_shape = "(12, 2500) per sample (lazy)" if mmap_waveforms else str(waveforms.shape[1:])
    print(
        f"  {split}: {len(waveforms):,} samples | "
        f"waveforms {waveform_shape} | "
        f"demo features {demo_features.shape[1]} | "
        f"labels {labels.shape[1]}"
    )

    return SplitData(
        waveforms=waveforms,
        labels=labels,
        demo_features=demo_features,
        label_names=label_names,
    ), demo_encoder
