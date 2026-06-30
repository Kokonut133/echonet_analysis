"""Trains classifiers on three feature sets and compares AUROC across all SHD targets.

Feature sets:
  tabular_only   — precomputed tabular ECG metadata (age, rates, intervals)
  waveform_only  — lead-wise features from extract_waveform_features.py
  combined       — tabular + waveform concatenated

Run extract_waveform_features.py first to generate the waveform feature cache.
Uses the official train/val split from the metadata 'split' column.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.classifiers import standard_classifier_suite
from src.constants import DATASET_SUBDIR, METADATA_FILENAME, TARGET_LABELS, load_config
from src.evaluation import TargetResult, compute_binary_metrics, print_best_per_target, results_to_dataframe, save_results

_sk = load_config()["training"]["sklearn"]


@dataclass(frozen=True)
class Paths:
    dataset_dir: Path
    features_dir: Path
    metadata_path: Path
    output_path: Path


@dataclass
class TabularSplitData:
    tabular: np.ndarray
    waveform_features: np.ndarray
    labels: pd.DataFrame

    @property
    def combined(self) -> np.ndarray:
        return np.concatenate([self.tabular, self.waveform_features], axis=1)


@dataclass
class Config:
    paths: Paths
    targets: list[str] = field(default_factory=lambda: TARGET_LABELS)
    random_seed: int = _sk["random_seed"]
    train_split: str = "train"
    eval_split: str = "val"


def load_split_data(split: str, metadata: pd.DataFrame, paths: Paths) -> TabularSplitData:
    split_meta = metadata[metadata["split"] == split].reset_index(drop=True)

    tabular_path = paths.dataset_dir / f"EchoNext_{split}_tabular_features.npy"
    waveform_path = paths.features_dir / f"ecg_waveform_features_{split}.npy"

    if not tabular_path.exists():
        raise FileNotFoundError(f"Tabular features not found: {tabular_path}")
    if not waveform_path.exists():
        raise FileNotFoundError(
            f"Waveform features not found: {waveform_path}. Run extract_waveform_features.py first."
        )

    tabular = np.load(tabular_path).astype(np.float32)
    waveform = np.load(waveform_path).astype(np.float32)

    if len(split_meta) != len(tabular):
        raise ValueError(
            f"Row count mismatch in '{split}': metadata={len(split_meta)}, tabular={len(tabular)}"
        )
    if len(tabular) != len(waveform):
        raise ValueError(
            f"Row count mismatch in '{split}': tabular={len(tabular)}, waveform={len(waveform)}"
        )

    print(
        f"  {split}: {len(tabular):,} samples — "
        f"{tabular.shape[1]} tabular + {waveform.shape[1]} waveform features"
    )
    return TabularSplitData(tabular=tabular, waveform_features=waveform, labels=split_meta)


def evaluate_target_across_feature_sets(
    target: str,
    train: TabularSplitData,
    val: TabularSplitData,
    feature_sets: dict[str, tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> list[TargetResult]:
    valid_train = train.labels[target].notna()
    valid_val = val.labels[target].notna()

    if valid_train.sum() < 100 or valid_val.sum() < 20:
        print(f"  [{target}] skipping — insufficient labelled rows")
        return []

    y_train = train.labels[target][valid_train].values.astype(int)
    y_val = val.labels[target][valid_val].values.astype(int)
    prevalence = (y_train.sum() + y_val.sum()) / (len(y_train) + len(y_val))

    results: list[TargetResult] = []
    classifiers = standard_classifier_suite(seed)

    for fs_name, (X_tr_full, X_val_full) in feature_sets.items():
        X_tr = X_tr_full[valid_train.values]
        X_val = X_val_full[valid_val.values]

        for model_name, pipeline in classifiers.items():
            pipeline.fit(X_tr, y_train)
            y_prob = pipeline.predict_proba(X_val)[:, 1]
            metrics = compute_binary_metrics(y_val, y_prob)

            results.append(TargetResult(
                target=target,
                model=model_name,
                feature_set=fs_name,
                auroc=round(metrics["auroc"], 4),
                auprc=round(metrics["auprc"], 4),
                balanced_acc=round(metrics["balanced_acc"], 4),
                prevalence=round(prevalence, 4),
                n_positive=int(y_train.sum() + y_val.sum()),
                n_total=len(y_train) + len(y_val),
            ))
            print(
                f"    {fs_name:<22} {model_name:<22} "
                f"AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}"
            )

    return results


def run_ecg_feature_comparison(config: Config) -> pd.DataFrame:
    metadata = pd.read_csv(config.paths.metadata_path)
    print(f"Loaded metadata: {len(metadata):,} rows")

    print(f"\nLoading splits ...")
    train = load_split_data(config.train_split, metadata, config.paths)
    val = load_split_data(config.eval_split, metadata, config.paths)

    feature_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "tabular_only": (train.tabular, val.tabular),
        "waveform_only": (train.waveform_features, val.waveform_features),
        "combined": (train.combined, val.combined),
    }

    available_targets = [t for t in config.targets if t in metadata.columns]
    all_results: list[TargetResult] = []

    for i, target in enumerate(available_targets, start=1):
        print(f"\n[{i}/{len(available_targets)}] {target}")
        all_results.extend(
            evaluate_target_across_feature_sets(target, train, val, feature_sets, config.random_seed)
        )

    results_df = results_to_dataframe(all_results)
    print_best_per_target(results_df)
    save_results(results_df, config.paths.output_path)

    return results_df


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "data" / DATASET_SUBDIR

    config = Config(
        paths=Paths(
            dataset_dir=dataset_dir,
            features_dir=project_root / "data" / "extracted_features",
            metadata_path=dataset_dir / METADATA_FILENAME,
            output_path=project_root / "reports" / "ecg_feature_model_results.csv",
        ),
    )

    run_ecg_feature_comparison(config)


if __name__ == "__main__":
    main()
