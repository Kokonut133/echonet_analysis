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

from src.constants import load_config
from src.metrics import evaluate_loader, mean_auroc
from src.models import ECGConvNet

_cnn = load_config()["training"]["cnn"]


@dataclass
class TrainConfig:
    n_epochs: int           = _cnn["n_epochs"]
    batch_size: int         = _cnn["batch_size"]
    learning_rate: float    = _cnn["learning_rate"]
    weight_decay: float     = _cnn["weight_decay"]
    patience: int           = _cnn["patience"]
    lr_reduce_factor: float = _cnn["lr_reduce_factor"]
    lr_reduce_patience: int = _cnn["lr_reduce_patience"]
    device: str             = "auto"
    checkpoint_dir: Path    = field(default_factory=lambda: Path("checkpoints"))
    num_workers: int        = _cnn["num_workers"]


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class Trainer:

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
        best_val_auroc = -1.0
        patience_counter = 0
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

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch} (best val AUROC={best_val_auroc:.4f})")
                    break

        print(f"\nLoading best checkpoint (val mean-AUROC={best_val_auroc:.4f})")
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))

        return pd.DataFrame(log)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_masked_loss = 0.0
        n_valid_label_elements = 0

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
            total_masked_loss += loss.item() * n_elements
            n_valid_label_elements += n_elements

        return total_masked_loss / max(n_valid_label_elements, 1)


def compute_pos_weights(labels: np.ndarray) -> torch.Tensor:
    """(N, n_labels) float array with NaN → (n_labels,) tensor with weight = (1-p)/p per label."""
    weights = []
    for i in range(labels.shape[1]):
        col = labels[:, i]
        valid = col[~np.isnan(col)]
        p = float(valid.mean()) if len(valid) > 0 else 0.5
        p = np.clip(p, 1e-6, 1 - 1e-6)
        weights.append((1.0 - p) / p)
    return torch.tensor(weights, dtype=torch.float32)
