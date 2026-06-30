"""Demographic-only baseline: predicts all SHD targets from age, sex, race, care setting.

Uses a random stratified 80/20 split across the full metadata (not the official
train/val/test split) so results are comparable to prior literature that does not
use the official split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src.classifiers import standard_classifier_suite
from src.constants import DEMOGRAPHIC_FEATURES, DATASET_SUBDIR, METADATA_FILENAME, TARGET_LABELS, load_config
from src.evaluation import TargetResult, compute_binary_metrics, print_best_per_target, results_to_dataframe, save_results
from src.preprocessing import build_demographic_preprocessor

_sk = load_config()["training"]["sklearn"]


@dataclass
class Config:
    metadata_path: Path
    output_path: Path
    targets: list[str] = field(default_factory=lambda: TARGET_LABELS)
    random_seed: int = _sk["random_seed"]
    test_size: float = _sk["test_size"]


def stratified_train_test_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y))
    return X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]


def evaluate_target_across_models(
    target: str,
    metadata: pd.DataFrame,
    config: Config,
) -> list[TargetResult]:
    subset = metadata[DEMOGRAPHIC_FEATURES + [target]].dropna(subset=[target])

    if len(subset) < 100:
        print(f"  [{target}] skipping — only {len(subset)} rows")
        return []

    X = subset[DEMOGRAPHIC_FEATURES]
    y = subset[target].values.astype(int)
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, config.test_size, config.random_seed)

    preprocessor = build_demographic_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    prevalence = (y_train.sum() + y_test.sum()) / (len(y_train) + len(y_test))
    n_positive = int(y_train.sum() + y_test.sum())
    n_total = len(y_train) + len(y_test)

    results: list[TargetResult] = []
    for model_name, pipeline in standard_classifier_suite(config.random_seed).items():
        pipeline.fit(X_train_enc, y_train)
        y_prob = pipeline.predict_proba(X_test_enc)[:, 1]
        metrics = compute_binary_metrics(y_test, y_prob)

        results.append(TargetResult(
            target=target,
            model=model_name,
            feature_set="demographics",
            auroc=round(metrics["auroc"], 4),
            auprc=round(metrics["auprc"], 4),
            balanced_acc=round(metrics["balanced_acc"], 4),
            prevalence=round(prevalence, 4),
            n_positive=n_positive,
            n_total=n_total,
        ))
        print(f"    {model_name:<22} AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}")

    return results


def run_demographic_baseline(config: Config) -> pd.DataFrame:
    metadata = pd.read_csv(config.metadata_path)
    print(f"Loaded metadata: {len(metadata):,} rows")

    available_targets = [t for t in config.targets if t in metadata.columns]
    all_results: list[TargetResult] = []

    for i, target in enumerate(available_targets, start=1):
        print(f"\n[{i}/{len(available_targets)}] {target}")
        all_results.extend(evaluate_target_across_models(target, metadata, config))

    results_df = results_to_dataframe(all_results)
    print_best_per_target(results_df)
    save_results(results_df, config.output_path)

    return results_df


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / DATASET_SUBDIR

    config = Config(
        metadata_path=data_dir / METADATA_FILENAME,
        output_path=project_root / "reports" / "demographic_baseline_results.csv",
    )

    run_demographic_baseline(config)


if __name__ == "__main__":
    main()
