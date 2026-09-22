"""Final held-out evaluation on the official `test` split (5,442 records).

Five tiers on the information ladder, all using the same train/val/test
protocol (fit on train, refit-selection on val where needed, scored once on test):

  demographics        age, sex, race, care setting
  tabular_ecg         + heart rate, PR/QRS/QTc intervals
  waveform_features   hand-crafted lead-wise waveform features only
  combined            tabular_ecg + waveform_features
  cnn_raw_waveform    1D CNN (ECGConvNet) on the raw 12-lead signal

For the three classical ECG feature-set tiers, the "best" model per
(target, feature_set) is read from reports/ecg_feature_model_results.csv
(highest val AUROC), then refit on the full train split. The demographics
tier has no such precomputed comparison on this split protocol, so its best
model is selected the same way: fit all three classifiers on train, compare
on val, keep the winner. The CNN loads a pretrained checkpoint and only runs
inference.

Outputs:
  reports/final_results.csv
  reports/final_results_summary.md
  reports/predictions/{tier}_test.npz   (y_true, y_prob, label_names, record_index)

Caching: if reports/predictions/{tier}_test.npz already exists, that tier's
predictions are reused (metrics + bootstrap CIs are still recomputed) instead
of refitting/re-running inference. Pass --refit to force everything to rerun.
Evaluating a second CNN checkpoint (different --tag) reuses the four cached
classical/demographics tiers and only runs inference for the new checkpoint;
if the default checkpoint's cached predictions also exist, its rows are kept
in final_results.csv alongside the new checkpoint's for side-by-side comparison.

Usage:
  python scripts/6_evaluate/evaluate_test_set.py [--checkpoint PATH] [--tag NAME] [--refit]
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.bootstrap import DEFAULT_N_RESAMPLES, DEFAULT_SEED, bootstrap_auroc_auprc
from src.classifiers import standard_classifier_suite
from src.constants import (
    DATASET_SUBDIR,
    DEMOGRAPHIC_FEATURES,
    METADATA_FILENAME,
    N_LEADS,
    TARGET_LABELS,
    load_config,
)
from src.dataset import ECGDataset, load_split
from src.evaluation import compute_binary_metrics
from src.models import ECGConvNet
from src.plotting import TARGET_SHORT_NAMES, TIER_ORDER
from src.preprocessing import build_demographic_preprocessor

_sk = load_config()["training"]["sklearn"]
_SEED = _sk["random_seed"]
_FEATURE_SET_BY_TIER = {
    "tabular_ecg": "tabular_only",
    "waveform_features": "waveform_only",
    "combined": "combined",
}
_CNN_INFERENCE_BATCH_SIZE = 64
_SKLEARN_MAX_JOBS = 4  # shared 8-core box; leave headroom for other agents
_DEFAULT_CNN_TAG = "cnn_waveforms"
_DEFAULT_CNN_TIER = "cnn_raw_waveform"
# standard_classifier_suite() keys -> sklearn class name, used to resolve the "model"
# column for cached classical-tier predictions without refitting.
_MODEL_CLASS_NAMES = {
    "LogisticRegression": "LogisticRegression",
    "RandomForest": "RandomForestClassifier",
    "GradientBoosting": "GradientBoostingClassifier",
}


def cnn_tier_name(tag: str) -> str:
    """cnn_waveforms -> cnn_raw_waveform; cnn_waveforms_v2 -> cnn_raw_waveform_v2; else suffixed."""
    if tag == _DEFAULT_CNN_TAG:
        return _DEFAULT_CNN_TIER
    if tag.startswith(_DEFAULT_CNN_TAG):
        return _DEFAULT_CNN_TIER + tag[len(_DEFAULT_CNN_TAG):]
    return f"{_DEFAULT_CNN_TIER}_{tag}"


@dataclass(frozen=True)
class Paths:
    dataset_dir: Path
    features_dir: Path
    metadata_path: Path
    feature_results_path: Path
    checkpoint_path: Path
    reports_dir: Path
    predictions_dir: Path


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
    cnn_tag: str = _DEFAULT_CNN_TAG
    n_bootstrap: int = DEFAULT_N_RESAMPLES
    bootstrap_seed: int = DEFAULT_SEED
    random_seed: int = _SEED
    refit: bool = False  # if True, ignore any cached reports/predictions/*.npz and rerun everything


def load_tabular_split(split: str, metadata: pd.DataFrame, paths: Paths) -> TabularSplitData:
    split_meta = metadata[metadata["split"] == split].reset_index(drop=True)
    tabular = np.load(paths.dataset_dir / f"EchoNext_{split}_tabular_features.npy").astype(np.float32)
    waveform = np.load(paths.features_dir / f"ecg_waveform_features_{split}.npy").astype(np.float32)

    if len(split_meta) != len(tabular):
        raise ValueError(f"Row mismatch in '{split}': metadata={len(split_meta)}, tabular={len(tabular)}")
    if len(tabular) != len(waveform):
        raise ValueError(f"Row mismatch in '{split}': tabular={len(tabular)}, waveform={len(waveform)}")

    return TabularSplitData(tabular=tabular, waveform_features=waveform, labels=split_meta)


def best_model_name(feature_results: pd.DataFrame, target: str, feature_set: str) -> str | None:
    subset = feature_results[
        (feature_results["target"] == target) & (feature_results["feature_set"] == feature_set)
    ]
    if subset.empty:
        return None
    return subset.sort_values("auroc", ascending=False).iloc[0]["model"]


def build_target_row(
    target: str,
    tier: str,
    model_name: str,
    y_true_col: np.ndarray,
    y_prob_col: np.ndarray,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict | None:
    valid = ~np.isnan(y_true_col)
    n_total = int(valid.sum())
    if n_total == 0:
        return None

    y_true = y_true_col[valid].astype(int)
    y_prob = y_prob_col[valid]
    if len(np.unique(y_true)) < 2:
        return None

    metrics = compute_binary_metrics(y_true, y_prob)
    ci = bootstrap_auroc_auprc(y_true, y_prob, n_resamples=n_bootstrap, seed=bootstrap_seed)
    n_positive = int(y_true.sum())

    return {
        "target": target,
        "tier": tier,
        "model": model_name,
        "split": "test",
        "auroc": round(metrics["auroc"], 4),
        "auroc_ci_low": round(ci["auroc_ci_low"], 4),
        "auroc_ci_high": round(ci["auroc_ci_high"], 4),
        "auprc": round(metrics["auprc"], 4),
        "auprc_ci_low": round(ci["auprc_ci_low"], 4),
        "auprc_ci_high": round(ci["auprc_ci_high"], 4),
        "balanced_acc": round(metrics["balanced_acc"], 4),
        "prevalence": round(n_positive / n_total, 4),
        "n_positive": n_positive,
        "n_total": n_total,
    }


def save_predictions(
    filename_stem: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
    predictions_dir: Path,
    model_names: list[str] | None = None,
) -> None:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    extra = {"model_names": np.array(model_names, dtype=str)} if model_names is not None else {}
    np.savez(
        predictions_dir / f"{filename_stem}_test.npz",
        y_true=y_true.astype(np.float32),
        y_prob=y_prob.astype(np.float32),
        label_names=np.array(label_names, dtype=str),
        record_index=np.arange(y_true.shape[0]),
        **extra,
    )
    print(f"  Saved predictions -> {predictions_dir / f'{filename_stem}_test.npz'}")


def load_cached_predictions(tier: str, predictions_dir: Path) -> dict | None:
    """Returns cached {y_true, y_prob, label_names, model_names} for `tier`, or None if no
    cache file exists. `model_names` is None when the cached file predates that field."""
    path = predictions_dir / f"{tier}_test.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "y_true": data["y_true"],
        "y_prob": data["y_prob"],
        "label_names": [str(n) for n in data["label_names"]],
        "model_names": [str(n) for n in data["model_names"]] if "model_names" in data else None,
    }


# --- demographics tier: fit-on-train / select-on-val, same protocol as the others ---

def prepare_demographic_features(
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    train_meta = metadata[metadata["split"] == "train"].reset_index(drop=True)
    val_meta = metadata[metadata["split"] == "val"].reset_index(drop=True)
    test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)

    preprocessor = build_demographic_preprocessor()
    X_train = preprocessor.fit_transform(train_meta[DEMOGRAPHIC_FEATURES]).astype(np.float32)
    X_val = preprocessor.transform(val_meta[DEMOGRAPHIC_FEATURES]).astype(np.float32)
    X_test = preprocessor.transform(test_meta[DEMOGRAPHIC_FEATURES]).astype(np.float32)

    return train_meta, val_meta, test_meta, X_train, X_val, X_test


def select_best_demographic_pipeline(
    target: str,
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
    X_train: np.ndarray,
    X_val: np.ndarray,
    seed: int,
):
    y_train_full = train_meta[target].values.astype(np.float32)
    y_val_full = val_meta[target].values.astype(np.float32)
    train_valid = ~np.isnan(y_train_full)
    val_valid = ~np.isnan(y_val_full)

    if train_valid.sum() < 100 or val_valid.sum() < 20:
        return None, None

    best_name, best_pipeline, best_auroc = None, None, -1.0
    for model_name, pipeline in standard_classifier_suite(seed).items():
        pipeline.fit(X_train[train_valid], y_train_full[train_valid].astype(int))
        y_prob = pipeline.predict_proba(X_val[val_valid])[:, 1]
        metrics = compute_binary_metrics(y_val_full[val_valid].astype(int), y_prob)
        if not np.isnan(metrics["auroc"]) and metrics["auroc"] > best_auroc:
            best_name, best_pipeline, best_auroc = model_name, pipeline, metrics["auroc"]

    return best_name, best_pipeline


def _rows_from_cache(
    tier: str,
    cached: dict,
    targets: list[str],
    model_name_for: Callable[[int, str], str],
    n_bootstrap: int,
    bootstrap_seed: int,
) -> list[dict]:
    rows: list[dict] = []
    for target in targets:
        if target not in cached["label_names"]:
            continue
        i = cached["label_names"].index(target)
        model_name = model_name_for(i, target)
        row = build_target_row(
            target, tier, model_name, cached["y_true"][:, i], cached["y_prob"][:, i],
            n_bootstrap, bootstrap_seed,
        )
        if row is not None:
            rows.append(row)
            print(f"    {target:<55} {model_name:<28} AUROC={row['auroc']:.4f}  [cached]")
    return rows


def run_demographics_tier(
    metadata: pd.DataFrame, config: Config
) -> tuple[list[dict], np.ndarray, np.ndarray, list[str] | None, bool]:
    print("\n=== Tier: demographics ===")

    if not config.refit:
        cached = load_cached_predictions("demographics", config.paths.predictions_dir)
        if cached is not None and cached["model_names"] is not None:
            print("  Using cached predictions -> demographics_test.npz")
            rows = _rows_from_cache(
                "demographics", cached, config.targets,
                lambda i, t: cached["model_names"][i],
                config.n_bootstrap, config.bootstrap_seed,
            )
            return rows, cached["y_true"], cached["y_prob"], cached["model_names"], True
        if cached is not None:
            print("  Cache found but has no stored model names (pre-dates caching) — refitting once.")

    train_meta, val_meta, test_meta, X_train, X_val, X_test = prepare_demographic_features(metadata)

    n_test = len(test_meta)
    y_true = np.full((n_test, len(config.targets)), np.nan, dtype=np.float32)
    y_prob = np.zeros((n_test, len(config.targets)), dtype=np.float32)
    model_names: list[str] = ["" for _ in config.targets]
    rows: list[dict] = []

    for i, target in enumerate(config.targets):
        if target not in metadata.columns:
            continue
        model_name, pipeline = select_best_demographic_pipeline(
            target, train_meta, val_meta, X_train, X_val, config.random_seed
        )
        if pipeline is None:
            continue

        y_true[:, i] = test_meta[target].values.astype(np.float32)
        y_prob[:, i] = pipeline.predict_proba(X_test)[:, 1]
        clf_name = pipeline.named_steps["clf"].__class__.__name__
        model_names[i] = clf_name

        row = build_target_row(
            target, "demographics", clf_name, y_true[:, i], y_prob[:, i],
            config.n_bootstrap, config.bootstrap_seed,
        )
        if row is not None:
            rows.append(row)
            print(f"    {target:<55} {clf_name:<20} AUROC={row['auroc']:.4f}")

    return rows, y_true, y_prob, model_names, False


# --- classical ECG feature-set tiers: best model per (target, feature_set) from val ---

def resolve_classical_model_name(feature_results: pd.DataFrame, target: str, feature_set: str) -> str:
    friendly = best_model_name(feature_results, target, feature_set)
    if friendly is None:
        return "unknown"
    return _MODEL_CLASS_NAMES.get(friendly, friendly)


def run_classical_tier(
    tier: str,
    feature_set: str,
    train_data: TabularSplitData | None,
    test_data: TabularSplitData | None,
    feature_results: pd.DataFrame,
    config: Config,
) -> tuple[list[dict], np.ndarray, np.ndarray, bool]:
    print(f"\n=== Tier: {tier} ===")

    if not config.refit:
        cached = load_cached_predictions(tier, config.paths.predictions_dir)
        if cached is not None:
            print(f"  Using cached predictions -> {tier}_test.npz")
            rows = _rows_from_cache(
                tier, cached, config.targets,
                lambda i, t: resolve_classical_model_name(feature_results, t, feature_set),
                config.n_bootstrap, config.bootstrap_seed,
            )
            return rows, cached["y_true"], cached["y_prob"], True

    feature_arrays = {
        "tabular_only": (train_data.tabular, test_data.tabular),
        "waveform_only": (train_data.waveform_features, test_data.waveform_features),
        "combined": (train_data.combined, test_data.combined),
    }
    X_train, X_test = feature_arrays[feature_set]

    n_test = len(test_data.labels)
    y_true = np.full((n_test, len(config.targets)), np.nan, dtype=np.float32)
    y_prob = np.zeros((n_test, len(config.targets)), dtype=np.float32)
    rows: list[dict] = []

    for i, target in enumerate(config.targets):
        if target not in train_data.labels.columns:
            continue
        model_name = best_model_name(feature_results, target, feature_set)
        if model_name is None:
            print(f"  [{target}] skipping — no entry in {config.paths.feature_results_path.name}")
            continue

        y_train_full = train_data.labels[target].values.astype(np.float32)
        train_valid = ~np.isnan(y_train_full)
        if train_valid.sum() < 100:
            continue

        pipeline = standard_classifier_suite(config.random_seed)[model_name]
        pipeline.fit(X_train[train_valid], y_train_full[train_valid].astype(int))

        y_true[:, i] = test_data.labels[target].values.astype(np.float32)
        y_prob[:, i] = pipeline.predict_proba(X_test)[:, 1]
        clf_name = pipeline.named_steps["clf"].__class__.__name__

        row = build_target_row(
            target, tier, clf_name, y_true[:, i], y_prob[:, i],
            config.n_bootstrap, config.bootstrap_seed,
        )
        if row is not None:
            rows.append(row)
            print(f"    {target:<55} {clf_name:<20} AUROC={row['auroc']:.4f}")

    return rows, y_true, y_prob, False


# --- CNN tier: pretrained checkpoint, inference only ---

@torch.no_grad()
def cnn_predict(model: ECGConvNet, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_true, all_mask = [], [], []
    for waveforms, demo, labels, valid_mask in tqdm(loader, desc="  cnn inference", unit="batch"):
        waveforms = waveforms.to(device)
        demo = demo.to(device)
        probs = torch.sigmoid(model(waveforms, demo)).cpu().numpy()
        all_probs.append(probs)
        all_true.append(labels.numpy())
        all_mask.append(valid_mask.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_true, axis=0).astype(np.float32)
    y_mask = np.concatenate(all_mask, axis=0)
    y_true[~y_mask] = np.nan

    return y_true, y_prob


def run_cnn_tier(
    metadata: pd.DataFrame, config: Config
) -> tuple[list[dict], np.ndarray, np.ndarray, list[str], str, bool]:
    """The checkpoint's output layer is fixed at architecture time (n_labels=len(TARGET_LABELS)),
    so the CNN always predicts the full label set regardless of `config.targets`; results are then
    filtered down to `config.targets` for reporting."""
    tier = cnn_tier_name(config.cnn_tag)
    print(f"\n=== Tier: {tier} ===")

    if not config.refit:
        cached = load_cached_predictions(tier, config.paths.predictions_dir)
        if cached is not None:
            print(f"  Using cached predictions -> {tier}_test.npz")
            rows = _rows_from_cache(
                tier, cached, config.targets, lambda i, t: "ECGConvNet",
                config.n_bootstrap, config.bootstrap_seed,
            )
            return rows, cached["y_true"], cached["y_prob"], cached["label_names"], tier, True

    if not config.paths.checkpoint_path.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {config.paths.checkpoint_path}")

    cnn_label_names = TARGET_LABELS
    test_data, _ = load_split("test", metadata, config.paths.dataset_dir, label_names=cnn_label_names)
    test_ds = ECGDataset(test_data)
    loader = DataLoader(test_ds, batch_size=_CNN_INFERENCE_BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGConvNet(n_leads=N_LEADS, n_labels=len(cnn_label_names), n_demo_features=0)
    state_dict = torch.load(config.paths.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    print(f"  Loaded {config.paths.checkpoint_path} (tag={config.cnn_tag}) on {device}")
    y_true, y_prob = cnn_predict(model, loader, device)

    rows: list[dict] = []
    for target in config.targets:
        if target not in cnn_label_names:
            continue
        i = cnn_label_names.index(target)
        row = build_target_row(
            target, tier, "ECGConvNet", y_true[:, i], y_prob[:, i],
            config.n_bootstrap, config.bootstrap_seed,
        )
        if row is not None:
            rows.append(row)
            print(f"    {target:<55} {'ECGConvNet':<20} AUROC={row['auroc']:.4f}")

    return rows, y_true, y_prob, cnn_label_names, tier, False


def include_default_cnn_if_missing(config: Config, all_rows: list[dict], current_tier: str) -> None:
    """When evaluating a non-default CNN checkpoint, also surface the default (v1) CNN's cached
    results in final_results.csv (if available) so multiple checkpoints appear side by side."""
    if current_tier == _DEFAULT_CNN_TIER:
        return
    cached = load_cached_predictions(_DEFAULT_CNN_TIER, config.paths.predictions_dir)
    if cached is None:
        return
    print(f"\n  Also including cached {_DEFAULT_CNN_TIER}_test.npz for comparison")
    rows = _rows_from_cache(
        _DEFAULT_CNN_TIER, cached, config.targets, lambda i, t: "ECGConvNet",
        config.n_bootstrap, config.bootstrap_seed,
    )
    all_rows.extend(rows)


def write_summary_markdown(results_df: pd.DataFrame, targets: list[str], output_path: Path) -> None:
    present = set(results_df["tier"].unique())
    tiers_present = [t for t in TIER_ORDER if t in present]
    tiers_present += [t for t in sorted(present) if t not in tiers_present]
    header = "| target | " + " | ".join(tiers_present) + " |"
    divider = "|---" * (len(tiers_present) + 1) + "|"
    lines = [header, divider]

    for target in targets:
        if target not in results_df["target"].unique():
            continue
        cells = [TARGET_SHORT_NAMES.get(target, target)]
        for tier in tiers_present:
            match = results_df[(results_df["target"] == target) & (results_df["tier"] == tier)]
            if match.empty:
                cells.append("—")
            else:
                r = match.iloc[0]
                cells.append(f"{r['auroc']:.3f} [{r['auroc_ci_low']:.3f}–{r['auroc_ci_high']:.3f}]")
        lines.append("| " + " | ".join(cells) + " |")

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved summary -> {output_path}")


def run_evaluation(config: Config) -> pd.DataFrame:
    t0 = time.time()
    metadata = pd.read_csv(config.paths.metadata_path)
    print(f"Loaded metadata: {len(metadata):,} rows")
    feature_results = pd.read_csv(config.paths.feature_results_path)

    all_rows: list[dict] = []

    with joblib.parallel_backend("loky", n_jobs=_SKLEARN_MAX_JOBS):
        demo_rows, demo_true, demo_prob, demo_model_names, demo_cached = run_demographics_tier(metadata, config)
        all_rows.extend(demo_rows)
        if not demo_cached:
            save_predictions(
                "demographics", demo_true, demo_prob, config.targets,
                config.paths.predictions_dir, model_names=demo_model_names,
            )

        classical_tiers = list(_FEATURE_SET_BY_TIER.items())
        need_split_data = config.refit or any(
            not (config.paths.predictions_dir / f"{tier}_test.npz").exists()
            for tier, _ in classical_tiers
        )
        if need_split_data:
            print("\nLoading tabular + waveform-feature splits ...")
            train_tab = load_tabular_split("train", metadata, config.paths)
            test_tab = load_tabular_split("test", metadata, config.paths)
        else:
            print("\nAll classical ECG tiers cached — skipping tabular/waveform split load.")
            train_tab = test_tab = None

        for tier, feature_set in classical_tiers:
            rows, y_true, y_prob, used_cache = run_classical_tier(
                tier, feature_set, train_tab, test_tab, feature_results, config
            )
            all_rows.extend(rows)
            if not used_cache:
                save_predictions(tier, y_true, y_prob, config.targets, config.paths.predictions_dir)
        del train_tab, test_tab

    cnn_rows, cnn_true, cnn_prob, cnn_label_names, cnn_tier, cnn_cached = run_cnn_tier(metadata, config)
    all_rows.extend(cnn_rows)
    if not cnn_cached:
        save_predictions(cnn_tier, cnn_true, cnn_prob, cnn_label_names, config.paths.predictions_dir)
    include_default_cnn_if_missing(config, all_rows, cnn_tier)

    results_df = pd.DataFrame(all_rows)
    target_rank = {t: i for i, t in enumerate(config.targets)}
    tier_rank = {t: i for i, t in enumerate(TIER_ORDER)}
    results_df["_t"] = results_df["target"].map(target_rank)
    results_df["_s"] = results_df["tier"].map(lambda t: tier_rank.get(t, len(TIER_ORDER)))
    results_df = results_df.sort_values(["_t", "_s"]).drop(columns=["_t", "_s"]).reset_index(drop=True)

    config.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.paths.reports_dir / "final_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results -> {output_path}")

    write_summary_markdown(results_df, config.targets, config.paths.reports_dir / "final_results_summary.md")

    print(f"\nDone in {time.time() - t0:.1f}s")
    return results_df


def build_run_config(checkpoint: Path | None, tag: str, refit: bool = False) -> Config:
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "data" / DATASET_SUBDIR
    checkpoint_path = checkpoint or (project_root / "checkpoints" / "cnn_waveforms.pt")

    paths = Paths(
        dataset_dir=dataset_dir,
        features_dir=project_root / "data" / "extracted_features",
        metadata_path=dataset_dir / METADATA_FILENAME,
        feature_results_path=project_root / "reports" / "ecg_feature_model_results.csv",
        checkpoint_path=checkpoint_path,
        reports_dir=project_root / "reports",
        predictions_dir=project_root / "reports" / "predictions",
    )
    return Config(paths=paths, cnn_tag=tag, refit=refit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to CNN checkpoint (.pt)")
    parser.add_argument("--tag", type=str, default="cnn_waveforms", help="Label for this CNN checkpoint")
    parser.add_argument(
        "--refit", action="store_true",
        help="Ignore cached reports/predictions/*.npz and refit/rerun every tier from scratch",
    )
    args = parser.parse_args()

    config = build_run_config(args.checkpoint, args.tag, refit=args.refit)
    run_evaluation(config)


if __name__ == "__main__":
    main()
