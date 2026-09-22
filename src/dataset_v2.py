"""Dataset wrapper for CNN v2: applies a waveform transform (see
src/augmentation.py) on top of the existing ECGDataset. Subclasses rather
than edits src/dataset.py, which other in-flight work depends on.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from src.dataset import ECGDataset, SplitData


class AugmentedECGDataset(ECGDataset):
    """ECGDataset that applies `transform(waveform, rng=...)` to each
    (n_leads, n_samples) waveform before it is turned into a tensor.

    `transform` should follow the calling convention used in
    src/augmentation.py: `transform(x: np.ndarray, rng: np.random.Generator
    | None) -> np.ndarray`. Pass `src.augmentation.train_augment` for
    training (normalisation + random augmentation) or
    `src.augmentation.eval_transform` for validation/test (normalisation
    only, no randomness).

    `rng`, if given, is a single Generator reused for every `__getitem__`
    call — deterministic and convenient for tests, but not safe to share
    across multiple DataLoader worker processes (they'd fork the same
    state). Leave `rng=None` (the default) for real training: each call
    then draws a fresh OS-entropy-seeded Generator, which is not correlated
    across forked workers.
    """

    def __init__(
        self,
        data: SplitData,
        transform: Optional[Callable] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(data)
        self.transform = transform
        self.rng = rng

    def __getitem__(self, idx: int):
        waveform = self._read_waveform(idx)  # (n_leads, n_samples) numpy float32
        if self.transform is not None:
            rng = self.rng if self.rng is not None else np.random.default_rng()
            waveform = self.transform(waveform, rng=rng)
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)

        return (
            torch.from_numpy(waveform),
            torch.from_numpy(self.demo[idx]),
            torch.from_numpy(self.labels[idx]),
            torch.from_numpy(self.valid_mask[idx]),
        )
