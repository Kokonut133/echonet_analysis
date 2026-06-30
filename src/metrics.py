from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from src.models import ECGConvNet


def compute_per_label_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(label_names):
        col_true = y_true[:, i]
        col_prob = y_prob[:, i]

        valid = ~np.isnan(col_true)
        col_true = col_true[valid]
        col_prob = col_prob[valid]

        n_total = int(valid.sum())
        n_positive = int(col_true.sum())

        if n_total < 10 or len(np.unique(col_true)) < 2:
            auroc = auprc = bal_acc = float("nan")
        else:
            col_pred = (col_prob >= 0.5).astype(int)
            auroc = float(roc_auc_score(col_true, col_prob))
            auprc = float(average_precision_score(col_true, col_prob))
            bal_acc = float(balanced_accuracy_score(col_true, col_pred))

        rows.append({
            "label": name,
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "balanced_acc": round(bal_acc, 4),
            "n_positive": n_positive,
            "n_total": n_total,
            "prevalence": round(n_positive / n_total, 4) if n_total > 0 else float("nan"),
        })

    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_loader(
    model: ECGConvNet,
    loader: DataLoader,
    device: torch.device,
    label_names: list[str],
) -> pd.DataFrame:
    model.eval()

    all_probs: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    all_mask: list[np.ndarray] = []

    for waveforms, demo, labels, valid_mask in loader:
        waveforms = waveforms.to(device)
        demo = demo.to(device)
        probs = torch.sigmoid(model(waveforms, demo)).cpu().numpy()
        all_probs.append(probs)
        all_true.append(labels.numpy())
        all_mask.append(valid_mask.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_true, axis=0).astype(float)
    y_mask = np.concatenate(all_mask, axis=0)

    y_true[~y_mask] = float("nan")

    return compute_per_label_metrics(y_true, y_prob, label_names)


def mean_auroc(metrics_df: pd.DataFrame) -> float:
    valid = metrics_df["auroc"].dropna()
    return float(valid.mean()) if len(valid) > 0 else 0.0
