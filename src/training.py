"""Training loop with early stopping for the ECG CNN pipelines.

The Trainer handles:
  - class-imbalance via per-label pos_weight in BCEWithLogitsLoss
  - masked loss (NaN labels are excluded per sample per label)
  - learning-rate reduction on plateau (val mean-AUROC)
  - early stopping with checkpoint saving for the best val mean-AUROC
  - per-epoch logging returned as a DataFrame
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.metrics import evaluate_loader, mean_auroc
from src.model import ECGConvNet


@dataclass
class TrainConfig:
    n_epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 5
    device: str = "auto"
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    num_workers: int = 4


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class Trainer:
    """Trains an ECGConvNet with masked BCE loss, LR scheduling, and early stopping.

    Args:
        model: the ECGConvNet to train.
        config: hyperparameter and runtime settings.
        pos_weights: (n_labels,) tensor of positive class weights for BCE loss.
            If None, all classes are weighted equally.
        label_names: label column names used for metrics reporting.
        checkpoint_name: filename stem for saving the best checkpoint.
    """

    def __init__(
        self,
        model: ECGConvNet,
        config: TrainConfig,
        pos_weights: torch.Tensor | None = None,
        label_names: list[str] | None = None,
        checkpoint_name: str = "best_model",
    ):
        self.config = config
        self.label_names = label_names or []
        self.device = resolve_device(config.device)

        self.model = model.to(self.device)

        pw = pos_weights.to(self.device) if pos_weights is not None else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=config.lr_reduce_factor,
            patience=config.lr_reduce_patience,
        )

        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = config.checkpoint_dir / f"{checkpoint_name}.pt"

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> pd.DataFrame:
        """Train for up to config.n_epochs and return the epoch log as a DataFrame."""
        best_auroc = -1.0
        epochs_without_improvement = 0
        log: list[dict] = []

        print(f"Training on {self.device} | {len(train_loader.dataset):,} train, "
              f"{len(val_loader.dataset):,} val samples")

        for epoch in range(1, self.config.n_epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_metrics = evaluate_loader(self.model, val_loader, self.device, self.label_names)
            val_auroc = mean_auroc(val_metrics)

            self.scheduler.step(val_auroc)
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:3d}/{self.config.n_epochs} | "
                f"loss={train_loss:.4f} | val mean-AUROC={val_auroc:.4f} | lr={lr:.2e}"
            )

            log.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_mean_auroc": round(val_auroc, 6),
                "lr": lr,
            })

            if val_auroc > best_auroc:
                best_auroc = val_auroc
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    print(f"Early stopping at epoch {epoch} (best val AUROC={best_auroc:.4f})")
                    break

        print(f"\nLoading best checkpoint (val mean-AUROC={best_auroc:.4f})")
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))

        return pd.DataFrame(log)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_elements = 0

        for waveforms, demo, labels, valid_mask in loader:
            waveforms = waveforms.to(self.device)
            demo = demo.to(self.device)
            labels = labels.to(self.device)
            valid_mask = valid_mask.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(waveforms, demo)

            loss_per_element = self.criterion(logits, labels)
            mask_float = valid_mask.float()
            loss = (loss_per_element * mask_float).sum() / mask_float.sum().clamp(min=1)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            n_elements = int(mask_float.sum().item())
            total_loss += loss.item() * n_elements
            total_elements += n_elements

        return total_loss / max(total_elements, 1)


def compute_pos_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute per-label positive weights for BCEWithLogitsLoss from a label matrix.

    Args:
        labels: (N, n_labels) float array possibly containing NaN.

    Returns:
        (n_labels,) float32 tensor with weight = (1 - p) / p per label.
    """
    weights = []
    for i in range(labels.shape[1]):
        col = labels[:, i]
        valid = col[~np.isnan(col)]
        p = float(valid.mean()) if len(valid) > 0 else 0.5
        p = np.clip(p, 1e-6, 1 - 1e-6)
        weights.append((1.0 - p) / p)
    return torch.tensor(weights, dtype=torch.float32)
