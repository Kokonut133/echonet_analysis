from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


@dataclass
class TargetResult:
    target: str
    model: str
    feature_set: str
    auroc: float
    auprc: float
    balanced_acc: float
    prevalence: float
    n_positive: int
    n_total: int


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    has_both_classes = len(np.unique(y_true)) > 1
    return {
        "auroc":        roc_auc_score(y_true, y_prob) if has_both_classes else float("nan"),
        "auprc":        average_precision_score(y_true, y_prob) if has_both_classes else float("nan"),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred) if has_both_classes else float("nan"),
    }


def results_to_dataframe(results: list[TargetResult]) -> pd.DataFrame:
    return pd.DataFrame([dataclasses.asdict(r) for r in results])


def print_best_per_target(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        print("No results to display.")
        return

    best = (
        results_df.sort_values("auroc", ascending=False)
        .groupby(["target", "feature_set"])
        .first()
        .reset_index()
        [["target", "feature_set", "model", "auroc", "auprc", "balanced_acc", "prevalence"]]
    )
    print("\n=== Best model per target × feature set (AUROC) ===")
    print(best.to_string(index=False))


def save_results(results_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Saved results → {output_path}")
