"""Optuna hyperparameter search over the classical (non-CNN) models.

Reviewer objection this answers: the project reports every classical model at
fixed, untuned hyperparameters straight out of data/project_config.json, which
makes "the interpretable feature-based model comes within ~0.02 AUROC of the
CNN" a weak claim — an untuned baseline is a weak baseline. This script tunes
HistGradientBoostingClassifier and LogisticRegression with Optuna TPE on the
`combined` feature set (tabular ECG metadata + hand-crafted waveform
features), fit on the official train split and selected on the official val
split. The test split is never touched here.

CPU-only: sets OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS before
importing numpy/sklearn, and src.tuning additionally wraps every .fit() call
in threadpoolctl.threadpool_limits — total parallelism stays <= 3 regardless
of how many trials/jobs are requested (Optuna itself is run with n_jobs=1;
trial-level parallelism is not used).

Usage:
    .venv/bin/python scripts/11_tuning/tune_classical.py
    .venv/bin/python scripts/11_tuning/tune_classical.py --targets shd_moderate_or_greater_flag --n-trials 10 --timeout 300
"""
from __future__ import annotations

import os

# Must happen before numpy/sklearn import: bounds BLAS/OpenMP thread pools at
# process start. threadpoolctl.threadpool_limits (in src.tuning) is a second,
# runtime-adjustable layer of the same limit.
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from src.constants import DATASET_SUBDIR, METADATA_FILENAME, load_config
from src.plotting import TARGET_SHORT_NAMES, apply_style
from src.tuning import (
    MODEL_NAMES,
    run_study,
    safe_param_importances,
    trial_value_for_reporting,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / DATASET_SUBDIR
FEATURES_DIR = PROJECT_ROOT / "data" / "extracted_features"
REPORTS_DIR = PROJECT_ROOT / "reports"

HEADLINE_TARGETS: list[str] = [
    "shd_moderate_or_greater_flag",
    "lvef_lte_45_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "aortic_stenosis_moderate_or_greater_flag",
]

DEFAULT_N_TRIALS = 40
DEFAULT_TIMEOUT_S = 25 * 60  # 25 minutes per (model, target) study

# The tuned HistGradientBoostingClassifier's natural untuned comparison point
# is the project's existing (slow) GradientBoostingClassifier — that's the
# whole reason for swapping it in (see module docstring / tuning_notes.md);
# LogisticRegression baselines against itself by name.
BASELINE_MODEL_NAME = {
    "HistGradientBoostingClassifier": "GradientBoosting",
    "LogisticRegression": "LogisticRegression",
}

_SEED = load_config()["training"]["sklearn"]["random_seed"]


def load_combined_split(split: str, metadata: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """tabular (7 cols) + waveform (180 cols) features, concatenated, aligned to metadata rows
    for one official split — same construction as scripts/4_classical_ml/compare_ecg_feature_sets.py.
    Never loads a waveform .npy (raw signal); only the cached feature matrices."""
    split_meta = metadata[metadata["split"] == split].reset_index(drop=True)

    tabular = np.load(DATASET_DIR / f"EchoNext_{split}_tabular_features.npy").astype(np.float32)
    waveform = np.load(FEATURES_DIR / f"ecg_waveform_features_{split}.npy").astype(np.float32)

    if not (len(split_meta) == len(tabular) == len(waveform)):
        raise ValueError(
            f"Row count mismatch in '{split}': metadata={len(split_meta)}, "
            f"tabular={len(tabular)}, waveform={len(waveform)}"
        )

    combined = np.concatenate([tabular, waveform], axis=1)
    return combined, split_meta


def target_xy(target: str, X: np.ndarray, labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    valid = labels[target].notna().values
    y = labels[target].values[valid].astype(int)
    return X[valid], y


def load_baseline_auroc(target: str, model_name: str) -> float | None:
    """Untuned val AUROC for (target, model, feature_set=combined) from the
    existing results file this task must not modify."""
    baseline_path = REPORTS_DIR / "ecg_feature_model_results.csv"
    df = pd.read_csv(baseline_path)
    baseline_model = BASELINE_MODEL_NAME[model_name]
    row = df[(df["target"] == target) & (df["model"] == baseline_model) & (df["feature_set"] == "combined")]
    if row.empty:
        return None
    return float(row["auroc"].iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=HEADLINE_TARGETS, help="targets to tune (default: 4 headline targets)")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES, help="models to tune")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS, help="max trials per (model, target) study")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="max seconds per (model, target) study")
    parser.add_argument("--seed", type=int, default=_SEED)
    args = parser.parse_args()

    print(f"Targets: {args.targets}")
    print(f"Models:  {args.models}")
    print(f"Budget:  <= {args.n_trials} trials or <= {args.timeout:.0f}s per study")

    metadata = pd.read_csv(DATASET_DIR / METADATA_FILENAME)
    print(f"Loaded metadata: {len(metadata):,} rows")

    print("Loading train/val combined feature matrices (tabular + waveform, cached — no raw waveform .npy touched) ...")
    X_train_all, train_meta = load_combined_split("train", metadata)
    X_val_all, val_meta = load_combined_split("val", metadata)
    print(f"  train: {X_train_all.shape}, val: {X_val_all.shape}")
    print("  test split is not loaded anywhere in this script.")

    all_trial_rows: list[dict] = []
    best_rows: list[dict] = []
    winner_by_target: dict[str, dict] = {}  # target -> {"model", "study"} for the plot

    run_started = time.time()

    for t_idx, target in enumerate(args.targets, start=1):
        if target not in metadata.columns:
            print(f"[{target}] not in metadata columns, skipping")
            continue

        X_train, y_train = target_xy(target, X_train_all, train_meta)
        X_val, y_val = target_xy(target, X_val_all, val_meta)
        prevalence = (y_train.sum() + y_val.sum()) / (len(y_train) + len(y_val))
        print(f"\n[{t_idx}/{len(args.targets)}] {target}  (train n={len(y_train):,}, val n={len(y_val):,}, prevalence={prevalence:.3f})")

        target_best_auroc = -1.0
        for model_name in args.models:
            print(f"  -- {model_name} --")
            t0 = time.time()
            study = run_study(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_trials=args.n_trials,
                timeout=args.timeout,
                seed=args.seed,
                study_name=f"{target}__{model_name}",
            )
            elapsed = time.time() - t0
            n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
            n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
            print(f"     {len(study.trials)} trials ({n_complete} complete, {n_pruned} pruned) in {elapsed:.1f}s — best val AUROC {study.best_value:.4f}")

            for trial in study.trials:
                all_trial_rows.append({
                    "target": target,
                    "model": model_name,
                    "trial": trial.number,
                    "params_json": json.dumps(trial.params),
                    "val_auroc": trial_value_for_reporting(trial),
                })

            baseline_auroc = load_baseline_auroc(target, model_name)
            best_rows.append({
                "target": target,
                "model": model_name,
                "best_params_json": json.dumps(study.best_params),
                "val_auroc": round(study.best_value, 4),
                "baseline_val_auroc": baseline_auroc,
                "delta": round(study.best_value - baseline_auroc, 4) if baseline_auroc is not None else None,
            })

            if study.best_value > target_best_auroc:
                target_best_auroc = study.best_value
                winner_by_target[target] = {"model": model_name, "study": study}

    total_elapsed = time.time() - run_started
    print(f"\nTotal tuning wall-clock time: {total_elapsed / 60:.1f} min")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    trials_df = pd.DataFrame(all_trial_rows)
    trials_df.to_csv(REPORTS_DIR / "tuning_results.csv", index=False)
    print(f"Saved {REPORTS_DIR / 'tuning_results.csv'} ({len(trials_df)} rows)")

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(REPORTS_DIR / "tuning_best.csv", index=False)
    print(f"Saved {REPORTS_DIR / 'tuning_best.csv'} ({len(best_df)} rows)")

    plot_importance_figure(winner_by_target, args.targets)


def plot_importance_figure(winner_by_target: dict[str, dict], targets: list[str]) -> None:
    """Per-target hyperparameter importance (winning model) + optimisation-history
    panel, in one figure, using the project's shared matplotlib style."""
    apply_style()

    targets = [t for t in targets if t in winner_by_target]
    if not targets:
        print("No winning studies to plot — skipping tuning_importance.png")
        return

    n = len(targets)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 8.5))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, target in enumerate(targets):
        info = winner_by_target[target]
        study = info["study"]
        model_name = info["model"]
        label = TARGET_SHORT_NAMES.get(target, target)

        # -- top row: hyperparameter importance --
        ax_imp = axes[0, col]
        importances = safe_param_importances(study)
        if importances:
            items = sorted(importances.items(), key=lambda kv: kv[1])
            names = [k for k, _ in items]
            values = [v for _, v in items]
            ax_imp.barh(names, values, color="#2CA6A4")
            ax_imp.set_xlabel("Optuna importance")
        else:
            ax_imp.text(0.5, 0.5, "importance unavailable\n(too few trials)", ha="center", va="center", transform=ax_imp.transAxes)
        ax_imp.set_title(f"{label}\nwinner: {model_name}", fontsize=11)

        # -- bottom row: optimisation history --
        ax_hist = axes[1, col]
        numbers = [t.number for t in study.trials]
        values = [trial_value_for_reporting(t) for t in study.trials]
        pruned = [t.state == optuna.trial.TrialState.PRUNED for t in study.trials]
        complete_x = [x for x, p in zip(numbers, pruned) if not p]
        complete_y = [y for y, p in zip(values, pruned) if not p]
        pruned_x = [x for x, p in zip(numbers, pruned) if p]
        pruned_y = [y for y, p in zip(values, pruned) if p]
        best_so_far = np.maximum.accumulate(values) if values else []

        ax_hist.scatter(complete_x, complete_y, s=14, color="#60A5FA", label="complete", zorder=3)
        if pruned_x:
            ax_hist.scatter(pruned_x, pruned_y, s=14, color="#F4A340", marker="x", label="pruned", zorder=3)
        ax_hist.plot(numbers, best_so_far, color="#111827", linewidth=1.3, label="best so far", zorder=2)
        ax_hist.set_xlabel("trial")
        if col == 0:
            ax_hist.set_ylabel("val AUROC")
            ax_hist.legend(loc="lower right", fontsize=8)

    fig.suptitle("Optuna tuning: hyperparameter importance and optimisation history (winning model per target)", fontsize=13, y=1.02)
    fig.tight_layout()

    out_path = PROJECT_ROOT / "figures" / "results" / "tuning_importance.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
