"""Demographic-only baseline predictor for all structural heart disease labels.

Trains Logistic Regression, Random Forest, and Gradient Boosting on demographic
features (age, sex, race/ethnicity, care setting) and evaluates each model
against all flag targets using AUROC, AUPRC, and balanced accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEMOGRAPHIC_FEATURES = ["age_at_ecg", "sex", "race_ethnicity", "location_setting"]

TARGET_LABELS = [
    "shd_moderate_or_greater_flag",
    "lvef_lte_45_flag",
    "lvwt_gte_13_flag",
    "aortic_stenosis_moderate_or_greater_flag",
    "aortic_regurgitation_moderate_or_greater_flag",
    "mitral_regurgitation_moderate_or_greater_flag",
    "tricuspid_regurgitation_moderate_or_greater_flag",
    "pulmonary_regurgitation_moderate_or_greater_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "pericardial_effusion_moderate_large_flag",
    "pasp_gte_45_flag",
    "tr_max_gte_32_flag",
]

NUMERIC_FEATURES = ["age_at_ecg"]
CATEGORICAL_FEATURES = ["sex", "race_ethnicity", "location_setting"]


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass
class TargetResult:
    target: str
    model: str
    auroc: float
    auprc: float
    balanced_accuracy: float
    n_positive: int
    n_total: int
    prevalence: float


@dataclass
class RunConfig:
    metadata_path: Path
    output_dir: Path
    test_size: float = 0.2
    random_seed: int = 42
    stratify_on: str = "shd_moderate_or_greater_flag"
    features: list[str] = field(default_factory=lambda: DEMOGRAPHIC_FEATURES)
    targets: list[str] = field(default_factory=lambda: TARGET_LABELS)


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns from {path.name}")
    return df


def select_usable_rows(df: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    required = features + [target]
    available = [col for col in required if col in df.columns]
    missing_cols = set(required) - set(available)

    if missing_cols:
        print(f"  [{target}] Missing columns: {missing_cols}")

    subset = df[available].dropna(subset=[target])
    return subset


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def build_models(random_seed: int) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()

    logistic = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_seed)),
    ])
    random_forest = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_seed, n_jobs=-1)),
    ])
    gradient_boosting = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=random_seed)),
    ])

    return {
        "LogisticRegression": logistic,
        "RandomForest": random_forest,
        "GradientBoosting": gradient_boosting,
    }


def split_data(df: pd.DataFrame, features: list[str], target: str, config: RunConfig) -> SplitData:
    available_features = [f for f in features if f in df.columns]
    X = df[available_features]
    y = df[target].values.astype(int)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_seed)
    train_idx, test_idx = next(splitter.split(X, y))

    return SplitData(
        X_train=X.iloc[train_idx],
        X_test=X.iloc[test_idx],
        y_train=y[train_idx],
        y_test=y[test_idx],
    )


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)

    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    return {"auroc": auroc, "auprc": auprc, "balanced_accuracy": bal_acc}


def train_and_evaluate_target(
    target: str,
    df: pd.DataFrame,
    models: dict[str, Pipeline],
    config: RunConfig,
) -> list[TargetResult]:
    subset = select_usable_rows(df, config.features, target)

    if len(subset) < 100:
        print(f"  [{target}] Skipping — only {len(subset)} rows available after dropping NaN targets.")
        return []

    split = split_data(subset, config.features, target, config)
    n_positive = int(split.y_train.sum() + split.y_test.sum())
    n_total = len(split.y_train) + len(split.y_test)
    prevalence = n_positive / n_total

    results: list[TargetResult] = []

    for model_name, pipeline in models.items():
        pipeline.fit(split.X_train, split.y_train)
        y_prob = pipeline.predict_proba(split.X_test)[:, 1]
        metrics = evaluate_predictions(split.y_test, y_prob)

        results.append(TargetResult(
            target=target,
            model=model_name,
            auroc=metrics["auroc"],
            auprc=metrics["auprc"],
            balanced_accuracy=metrics["balanced_accuracy"],
            n_positive=n_positive,
            n_total=n_total,
            prevalence=prevalence,
        ))

    return results


def results_to_dataframe(results: list[TargetResult]) -> pd.DataFrame:
    rows = [
        {
            "target": r.target,
            "model": r.model,
            "auroc": round(r.auroc, 4),
            "auprc": round(r.auprc, 4),
            "balanced_acc": round(r.balanced_accuracy, 4),
            "prevalence": round(r.prevalence, 4),
            "n_positive": r.n_positive,
            "n_total": r.n_total,
        }
        for r in results
    ]
    return pd.DataFrame(rows)


def print_results_summary(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        print("No results to display.")
        return

    best_per_target = (
        results_df.sort_values("auroc", ascending=False)
        .groupby("target")
        .first()
        .reset_index()
        [["target", "model", "auroc", "auprc", "balanced_acc", "prevalence"]]
    )

    print("\n=== Best model per target (by AUROC) ===")
    print(best_per_target.to_string(index=False))

    print("\n=== Full results ===")
    print(results_df.to_string(index=False))


def save_results(results_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "demographic_baseline_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")


def run(config: RunConfig) -> pd.DataFrame:
    df = load_metadata(config.metadata_path)

    available_targets = [t for t in config.targets if t in df.columns]
    missing_targets = set(config.targets) - set(available_targets)

    if missing_targets:
        print(f"Warning: targets not found in metadata and will be skipped: {missing_targets}")

    models = build_models(config.random_seed)
    all_results: list[TargetResult] = []

    for i, target in enumerate(available_targets, start=1):
        print(f"\n[{i}/{len(available_targets)}] Target: {target}")
        target_results = train_and_evaluate_target(target, df, models, config)
        all_results.extend(target_results)

    results_df = results_to_dataframe(all_results)
    print_results_summary(results_df)
    save_results(results_df, config.output_dir)

    return results_df


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / (
        "echonext-a-dataset-for-detecting-echocardiogram-confirmed-structural-heart-disease-from-ecgs-1.1.0"
    )

    config = RunConfig(
        metadata_path=data_dir / "echonext_metadata_100k.csv",
        output_dir=project_root / "reports",
    )

    run(config)


if __name__ == "__main__":
    main()
