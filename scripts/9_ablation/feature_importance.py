"""Classical (GradientBoosting) feature importance on the 'combined' feature set.

For each headline target, fits the GradientBoosting pipeline from src.classifiers
on tabular(7) + waveform(180) = 187 features (train split, exactly as built in
scripts/4_classical_ml/compare_ecg_feature_sets.py) and computes permutation
importance on the val split.

in:  data/<dataset>/EchoNext_{split}_tabular_features.npy
     data/extracted_features/ecg_waveform_features_{split}.npy
     data/extracted_features/ecg_waveform_feature_names.json
out: reports/feature_importance.csv   (target, feature, group_lead, group_type, importance_mean, importance_std)
     figures/ablation/feature_importance.png

Usage:
  .venv/bin/python scripts/9_ablation/feature_importance.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.ablation import assign_feature_groups, group_color
from src.classifiers import gradient_boosting_pipeline
from src.constants import DATASET_SUBDIR, LEAD_NAMES, METADATA_FILENAME, load_config

# Column order of EchoNext_{split}_tabular_features.npy (see dataset README).
TABULAR_NAMES = [
    "sex", "ventricular_rate", "atrial_rate", "pr_interval",
    "qrs_duration", "qt_corrected", "age_at_ecg",
]

HEADLINE_TARGETS = [
    "shd_moderate_or_greater_flag",
    "lvef_lte_45_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "aortic_stenosis_moderate_or_greater_flag",
]
HEADLINE_LABELS = {
    "shd_moderate_or_greater_flag": "SHD (any)",
    "lvef_lte_45_flag": "LVEF ≤ 45%",
    "rv_systolic_dysfunction_moderate_or_greater_flag": "RV dysfunction",
    "aortic_stenosis_moderate_or_greater_flag": "Aortic stenosis",
}

_seed = load_config()["training"]["sklearn"]["random_seed"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 200,
})


@dataclass(frozen=True)
class Paths:
    dataset_dir: Path
    features_dir: Path
    metadata_path: Path
    results_path: Path
    figures_dir: Path


@dataclass(frozen=True)
class Config:
    paths: Paths
    targets: list[str]
    random_seed: int = _seed
    train_split: str = "train"
    eval_split: str = "val"
    n_repeats: int = 5
    n_jobs: int = 3


def load_combined_features(
    split: str, metadata: pd.DataFrame, paths: Paths
) -> tuple[np.ndarray, pd.DataFrame]:
    split_meta = metadata[metadata["split"] == split].reset_index(drop=True)
    tabular = np.load(paths.dataset_dir / f"EchoNext_{split}_tabular_features.npy").astype(np.float32)
    waveform = np.load(paths.features_dir / f"ecg_waveform_features_{split}.npy").astype(np.float32)

    if not (len(split_meta) == len(tabular) == len(waveform)):
        raise ValueError(
            f"Row count mismatch in '{split}': metadata={len(split_meta)}, "
            f"tabular={len(tabular)}, waveform={len(waveform)}"
        )

    combined = np.concatenate([tabular, waveform], axis=1)
    return combined, split_meta


def fit_and_score_target(
    target: str,
    X_train: np.ndarray,
    y_train_full: pd.Series,
    X_val: np.ndarray,
    y_val_full: pd.Series,
    feature_names: list[str],
    config: Config,
) -> pd.DataFrame | None:
    valid_train = y_train_full.notna().values
    valid_val = y_val_full.notna().values
    if valid_train.sum() < 100 or valid_val.sum() < 20:
        print(f"  [{target}] skipping — insufficient labelled rows")
        return None

    y_train = y_train_full[valid_train].astype(int).values
    y_val = y_val_full[valid_val].astype(int).values
    X_tr = X_train[valid_train]
    X_v = X_val[valid_val]

    pipeline = gradient_boosting_pipeline(config.random_seed)
    pipeline.fit(X_tr, y_train)

    result = permutation_importance(
        pipeline, X_v, y_val,
        n_repeats=config.n_repeats,
        scoring="roc_auc",
        n_jobs=config.n_jobs,
        random_state=config.random_seed,
    )

    groups = assign_feature_groups(feature_names, LEAD_NAMES, TABULAR_NAMES)
    df = groups.copy()
    df["target"] = target
    df["importance_mean"] = result.importances_mean
    df["importance_std"] = result.importances_std
    return df[["target", "feature", "group_lead", "group_type", "importance_mean", "importance_std"]]


def run_feature_importance(config: Config) -> pd.DataFrame:
    metadata = pd.read_csv(config.paths.metadata_path)
    print(f"Loaded metadata: {len(metadata):,} rows")

    with open(config.paths.features_dir / "ecg_waveform_feature_names.json") as f:
        waveform_names = json.load(f)
    feature_names = TABULAR_NAMES + waveform_names
    assert len(feature_names) == 187, f"expected 187 combined features, got {len(feature_names)}"

    print("Loading train/val combined feature matrices...")
    X_train, train_meta = load_combined_features(config.train_split, metadata, config.paths)
    X_val, val_meta = load_combined_features(config.eval_split, metadata, config.paths)
    print(f"  train: {X_train.shape} | val: {X_val.shape}")

    all_results = []
    for i, target in enumerate(config.targets, start=1):
        print(f"\n[{i}/{len(config.targets)}] {target}: fitting GradientBoosting + permutation importance...")
        df = fit_and_score_target(
            target, X_train, train_meta[target], X_val, val_meta[target], feature_names, config
        )
        if df is not None:
            top5 = df.sort_values("importance_mean", ascending=False).head(5)
            print(top5[["feature", "importance_mean", "importance_std"]].to_string(index=False))
            all_results.append(df)

    results = pd.concat(all_results, ignore_index=True)

    config.paths.results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.paths.results_path, index=False)
    print(f"\nSaved {len(results)} rows -> {config.paths.results_path}")

    config.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    make_feature_importance_figure(results, config.targets, config.paths.figures_dir)

    return results


def make_feature_importance_figure(results: pd.DataFrame, targets: list[str], figures_dir: Path) -> None:
    fig, axes = plt.subplots(2, len(targets), figsize=(4.6 * len(targets), 9.5))

    for col, target in enumerate(targets):
        sub = results[results["target"] == target]

        # --- top row: importance summed by lead (12 leads + tabular group) ---
        by_lead = sub.groupby("group_lead")["importance_mean"].sum().sort_values()
        ax = axes[0, col]
        colors = [group_color(g) for g in by_lead.index]
        ax.barh(by_lead.index, by_lead.values, color=colors)
        ax.set_title(HEADLINE_LABELS[target], fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel("By lead", fontsize=11, fontweight="bold")
        ax.set_xlabel("Σ permutation importance\n(Δ AUROC)")
        ax.tick_params(axis="y", labelsize=8.5)
        xmax = by_lead.values.max()
        ax.set_xlim(0, xmax * 1.2 if xmax > 0 else 1)

        # --- bottom row: top-10 feature types (summed across leads; tabular kept as-is) ---
        type_totals = sub.groupby("group_type")["importance_mean"].sum().sort_values(ascending=False).head(10)
        type_totals = type_totals.sort_values()
        # colour: tabular feature-types are grey; waveform feature-types are a neutral teal
        tabular_set = set(TABULAR_NAMES)
        colors2 = ["#7F7F7F" if t in tabular_set else "#55A868" for t in type_totals.index]
        ax2 = axes[1, col]
        ax2.barh(type_totals.index, type_totals.values, color=colors2)
        if col == 0:
            ax2.set_ylabel("Top-10 feature types", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Σ permutation importance\n(Δ AUROC)")
        ax2.tick_params(axis="y", labelsize=8.5)
        xmax2 = type_totals.values.max()
        ax2.set_xlim(0, xmax2 * 1.2 if xmax2 > 0 else 1)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(color="#4C72B0", label="Limb lead"),
            Patch(color="#DD8452", label="Chest lead"),
            Patch(color="#7F7F7F", label="Tabular (metadata / feature-type)"),
            Patch(color="#55A868", label="Waveform feature type"),
        ],
        loc="lower center", ncol=4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "GradientBoosting permutation importance on combined (tabular + waveform) features, val split",
        fontsize=13, y=1.01,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_path = figures_dir / "feature_importance.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "data" / DATASET_SUBDIR
    config = Config(
        paths=Paths(
            dataset_dir=dataset_dir,
            features_dir=project_root / "data" / "extracted_features",
            metadata_path=dataset_dir / METADATA_FILENAME,
            results_path=project_root / "reports" / "feature_importance.csv",
            figures_dir=project_root / "figures" / "ablation",
        ),
        targets=HEADLINE_TARGETS,
    )
    run_feature_importance(config)


if __name__ == "__main__":
    main()
