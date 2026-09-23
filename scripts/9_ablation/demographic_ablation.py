"""What does the ECG+demographics CNN actually lean on?

The `cnn_combined` checkpoint fuses 13 encoded demographic features (age, sex,
race/ethnicity, care setting) into the classification head alongside the 256-dim
waveform embedding. This script zeroes those inputs at inference — the whole
block, then one group at a time — and measures the AUROC each group is worth.

Zeroing is the neutral value for both encodings: a one-hot block of zeros
carries no category, and the age column is standardised so zero is the training
mean. The model is NOT retrained, so this is a sensitivity analysis in the same
sense as scripts/9_ablation/lead_ablation.py.

Inputs:  checkpoints/cnn_combined.pt
         checkpoints/cnn_combined_demo_encoder.joblib
         reports/cnn_combined_demo_features.json
Outputs: reports/demographic_ablation_results.csv
         figures/ablation/demographic_ablation.png
         figures/ablation/race_ethnicity_reliance.png  (only if race/ethnicity matters)

Usage:
  python scripts/9_ablation/demographic_ablation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ablation import (
    DEMOGRAPHIC_GROUP_COLORS,
    DEMOGRAPHIC_GROUPS,
    demographic_group_indices,
    zero_demographic_columns,
)
from src.bootstrap import DEFAULT_SEED
from src.constants import (
    CATEGORICAL_DEMOGRAPHIC_FEATURES,
    DATASET_SUBDIR,
    METADATA_FILENAME,
    N_LEADS,
    NUMERIC_DEMOGRAPHIC_FEATURES,
    TARGET_LABELS,
)
from src.dataset import ECGDataset, load_split
from src.models import ECGConvNet
from src.plotting import TARGET_SHORT_NAMES, apply_style

BATCH_SIZE = 64
N_BOOTSTRAP = 1000
# A group counts as "material" if zeroing it costs at least this much mean AUROC
# AND the per-target loss clears bootstrap noise on the headline target.
MATERIALITY_THRESHOLD = 0.005
HEADLINE_TARGET = "shd_moderate_or_greater_flag"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = PROJECT_ROOT / "figures" / "ablation"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "cnn_combined.pt"
ENCODER = PROJECT_ROOT / "checkpoints" / "cnn_combined_demo_encoder.joblib"
FEATURE_NAMES = REPORTS_DIR / "cnn_combined_demo_features.json"


def load_test_inputs() -> tuple[DataLoader, np.ndarray, torch.Tensor, pd.DataFrame]:
    metadata = pd.read_csv(PROJECT_ROOT / "data" / DATASET_SUBDIR / METADATA_FILENAME)
    encoder = joblib.load(ENCODER)
    split_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)
    columns = NUMERIC_DEMOGRAPHIC_FEATURES + CATEGORICAL_DEMOGRAPHIC_FEATURES
    demo = encoder.transform(split_meta.reindex(columns=columns)).astype(np.float32)

    test_data, _ = load_split(
        "test", metadata, PROJECT_ROOT / "data" / DATASET_SUBDIR, label_names=TARGET_LABELS
    )
    if len(demo) != len(test_data.waveforms):
        raise ValueError(f"demo rows {len(demo)} != waveform rows {len(test_data.waveforms)}")
    test_data.demo_features = demo

    dataset = ECGDataset(test_data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    # ECGDataset stores missing labels as 0 plus a validity mask; restore NaN so
    # the metric helpers can skip unlabelled rows per target.
    y_true = np.where(dataset.valid_mask, dataset.labels, np.nan).astype(np.float32)
    return loader, y_true, torch.from_numpy(demo), split_meta


@torch.no_grad()
def predict_all_ablations(
    model: ECGConvNet,
    loader: DataLoader,
    device: torch.device,
    ablations: dict[str, list[int]],
) -> dict[str, np.ndarray]:
    """One pass over the waveforms, producing probabilities for every ablation.

    Reading the memory-mapped waveform file is by far the slowest part, so each
    batch is scored once per ablation while it is in memory rather than looping
    over the file for each variant.
    """
    model.eval()
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in ablations}
    for waveforms, demo, _labels, _mask in tqdm(loader, desc="  ablation sweep", unit="batch"):
        waveforms = waveforms.to(device)
        demo = demo.to(device)
        for name, indices in ablations.items():
            ablated = zero_demographic_columns(demo, indices)
            probs = torch.sigmoid(model(waveforms, ablated)).cpu().numpy()
            chunks[name].append(probs)
    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


def auroc_or_nan(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    valid = ~np.isnan(y_true)
    y, p = y_true[valid], y_prob[valid]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y.astype(int), p))


def bootstrap_delta_ci(
    y_true: np.ndarray,
    y_prob_base: np.ndarray,
    y_prob_ablated: np.ndarray,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    """95% CI for the paired AUROC difference (ablated − baseline).

    Both models are scored on the same resampled records, so this is a paired
    comparison and the interval is much tighter than differencing two
    independent CIs.
    """
    valid = ~np.isnan(y_true)
    y, base, abl = y_true[valid].astype(int), y_prob_base[valid], y_prob_ablated[valid]
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        deltas.append(roc_auc_score(ys, abl[idx]) - roc_auc_score(ys, base[idx]))
    if not deltas:
        return float("nan"), float("nan")
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def build_results(
    probs: dict[str, np.ndarray], y_true: np.ndarray, targets: list[str]
) -> pd.DataFrame:
    rows: list[dict] = []
    base = probs["baseline"]
    for i, target in enumerate(targets):
        base_auroc = auroc_or_nan(y_true[:, i], base[:, i])
        for name, prob in probs.items():
            auroc = auroc_or_nan(y_true[:, i], prob[:, i])
            delta = auroc - base_auroc
            if name == "baseline":
                lo = hi = 0.0
            else:
                lo, hi = bootstrap_delta_ci(y_true[:, i], base[:, i], prob[:, i])
            rows.append({
                "ablation": name,
                "target": target,
                "auroc": round(auroc, 4),
                "delta_auroc": round(delta, 4),
                "delta_ci_low": round(lo, 4),
                "delta_ci_high": round(hi, 4),
            })
    return pd.DataFrame(rows)


def figure_demographic_ablation(results: pd.DataFrame, output_path: Path) -> None:
    apply_style()
    ablations = [a for a in results["ablation"].unique() if a != "baseline"]
    mean_delta = (
        results[results["ablation"] != "baseline"]
        .groupby("ablation")["delta_auroc"]
        .mean()
        .reindex(ablations)
        .sort_values()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})

    ax = axes[0]
    colors = [DEMOGRAPHIC_GROUP_COLORS.get(a, "#9CA3AF") for a in mean_delta.index]
    labels = [a.replace("_", " ").replace("age at ecg", "age") for a in mean_delta.index]
    ax.barh(range(len(mean_delta)), mean_delta.values, color=colors)
    ax.set_yticks(range(len(mean_delta)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color="#4B5563", linewidth=0.9)
    span = max(abs(mean_delta.min()), 1e-4)
    ax.set_xlim(mean_delta.min() - 0.42 * span, max(0.0, mean_delta.max()) + 0.12 * span)
    for y, v in enumerate(mean_delta.values):
        ax.text(v - 0.03 * span, y, f"{v:+.4f}", va="center", ha="right", fontsize=8.5, color="#333333")
    ax.set_xlabel("Mean ΔAUROC when zeroed\n(across all 12 targets)")
    ax.set_title("What each demographic input is worth", fontweight="bold", loc="left")

    ax2 = axes[1]
    heat_targets = list(TARGET_LABELS)
    heat = (
        results[results["ablation"] != "baseline"]
        .pivot(index="ablation", columns="target", values="delta_auroc")
        .reindex(index=mean_delta.index, columns=heat_targets)
    )
    # Age costs 0.145 AUROC on aortic stenosis — an order of magnitude more than
    # anything else. Scaling to that outlier would render the other 11 columns
    # blank, so the colour scale is clipped to the 85th percentile and the
    # saturated cells keep their printed value.
    # Only 2 of 60 cells exceed 0.03 (both age-on-aortic-stenosis); the 95th
    # percentile keeps a usable gradient over the other 58 while letting those
    # two saturate. Saturated cells keep their printed value.
    finite = np.abs(heat.values[~np.isnan(heat.values)])
    limit = float(np.percentile(finite, 95)) if finite.size else 1e-4
    limit = max(limit, 1e-4)
    im = ax2.imshow(heat.values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax2.set_xticks(range(len(heat_targets)))
    ax2.set_xticklabels(
        [TARGET_SHORT_NAMES.get(t, t) for t in heat_targets], rotation=38, ha="right", fontsize=8
    )
    ax2.set_yticks(range(len(heat.index)))
    ax2.set_yticklabels(labels, fontsize=9)
    for r in range(heat.shape[0]):
        for c in range(heat.shape[1]):
            v = heat.values[r, c]
            if np.isnan(v) or abs(v) < 0.005:
                continue
            ax2.text(
                c, r, f"{v:+.03f}".replace("+0.", "+.").replace("-0.", "−."),
                ha="center", va="center", fontsize=6.8,
                color="white" if abs(v) > 0.7 * limit else "#1F2937",
            )
    ax2.set_title("ΔAUROC per target  (cells below 0.005 left blank)", fontweight="bold", loc="left")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.015, extend="both")
    cbar.set_label("ΔAUROC (scale clipped at p95)", fontsize=9)

    fig.suptitle(
        "Zeroing demographic inputs of the ECG+demographics CNN (model not retrained)",
        fontsize=12.5, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def figure_race_reliance(
    results: pd.DataFrame, split_meta: pd.DataFrame, probs: dict[str, np.ndarray],
    y_true: np.ndarray, output_path: Path,
) -> None:
    """Only drawn when race/ethnicity turns out to be material: per-target loss
    from removing it, and whether that loss is concentrated in one group."""
    apply_style()
    race = results[results["ablation"] == "race_ethnicity"].set_index("target")
    heat_targets = [t for t in TARGET_LABELS if t in race.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

    ax = axes[0]
    vals = [race.loc[t, "delta_auroc"] for t in heat_targets]
    err_lo = [race.loc[t, "delta_auroc"] - race.loc[t, "delta_ci_low"] for t in heat_targets]
    err_hi = [race.loc[t, "delta_ci_high"] - race.loc[t, "delta_auroc"] for t in heat_targets]
    order = np.argsort(vals)
    ax.barh(
        range(len(order)), [vals[i] for i in order],
        xerr=[[err_lo[i] for i in order], [err_hi[i] for i in order]],
        color=DEMOGRAPHIC_GROUP_COLORS["race_ethnicity"], error_kw={"lw": 0.9, "ecolor": "#4B5563"},
    )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([TARGET_SHORT_NAMES.get(heat_targets[i], heat_targets[i]) for i in order], fontsize=9)
    ax.axvline(0, color="#4B5563", linewidth=0.9)
    ax.set_xlabel("ΔAUROC when race/ethnicity is zeroed\n(95% paired bootstrap CI)")
    ax.set_title("Per-target reliance on race/ethnicity", fontweight="bold", loc="left")

    ax2 = axes[1]
    i = TARGET_LABELS.index(HEADLINE_TARGET)
    groups = sorted(split_meta["race_ethnicity"].dropna().unique())
    base_v, abl_v, names = [], [], []
    for g in groups:
        mask = (split_meta["race_ethnicity"] == g).values & ~np.isnan(y_true[:, i])
        if mask.sum() < 30 or len(np.unique(y_true[mask, i])) < 2:
            continue
        names.append(f"{g}\n(n={int(mask.sum())})")
        base_v.append(auroc_or_nan(y_true[mask, i], probs["baseline"][mask, i]))
        abl_v.append(auroc_or_nan(y_true[mask, i], probs["race_ethnicity"][mask, i]))
    x = np.arange(len(names))
    ax2.bar(x - 0.19, base_v, 0.38, label="with race/ethnicity", color=DEMOGRAPHIC_GROUP_COLORS["all_demographics"])
    ax2.bar(x + 0.19, abl_v, 0.38, label="zeroed", color="#9CA3AF")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=8.5)
    ax2.set_ylim(0.5, 1.0)
    ax2.set_ylabel("AUROC")
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.set_title(
        f"{TARGET_SHORT_NAMES.get(HEADLINE_TARGET, HEADLINE_TARGET)}, by patient group",
        fontweight="bold", loc="left",
    )

    fig.suptitle("Does the model's use of race/ethnicity help or harm specific groups?", fontsize=12.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"{CHECKPOINT} not found — run scripts/5_deep_learning/cnn_waveforms_with_demographics.py first"
        )
    feature_names = json.loads(FEATURE_NAMES.read_text())
    group_indices = demographic_group_indices(feature_names)
    print(f"Demographic groups: { {k: len(v) for k, v in group_indices.items()} }")

    loader, y_true, demo, split_meta = load_test_inputs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGConvNet(
        n_leads=N_LEADS, n_labels=len(TARGET_LABELS), n_demo_features=demo.shape[1]
    )
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.to(device)
    print(f"Loaded {CHECKPOINT.name} on {device}")

    ablations: dict[str, list[int]] = {"baseline": []}
    ablations["all_demographics"] = list(range(demo.shape[1]))
    for group in DEMOGRAPHIC_GROUPS:
        ablations[group] = group_indices[group]

    probs = predict_all_ablations(model, loader, device, ablations)
    results = build_results(probs, y_true, list(TARGET_LABELS))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / "demographic_ablation_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(results)} rows)")

    summary = (
        results[results["ablation"] != "baseline"]
        .groupby("ablation")["delta_auroc"]
        .mean()
        .sort_values()
    )
    print("\nMean ΔAUROC when zeroed:")
    print(summary.to_string())

    figure_demographic_ablation(results, FIG_DIR / "demographic_ablation.png")

    race_mean = float(summary.get("race_ethnicity", 0.0))
    headline = results[
        (results["ablation"] == "race_ethnicity") & (results["target"] == HEADLINE_TARGET)
    ]
    headline_significant = bool(
        not headline.empty and headline.iloc[0]["delta_ci_high"] < 0.0
    )
    race_material = abs(race_mean) >= MATERIALITY_THRESHOLD and headline_significant
    print(
        f"\nrace/ethnicity: mean ΔAUROC={race_mean:+.4f}, "
        f"headline CI excludes 0: {headline_significant} -> "
        f"{'material' if race_material else 'not material'}"
    )
    if race_material:
        figure_race_reliance(results, split_meta, probs, y_true, FIG_DIR / "race_ethnicity_reliance.png")
    else:
        print("Skipping race_ethnicity_reliance.png — reliance is below the materiality threshold.")

    (REPORTS_DIR / "demographic_ablation_verdict.json").write_text(
        json.dumps(
            {
                "race_material": race_material,
                "race_mean_delta_auroc": round(race_mean, 4),
                "headline_ci_excludes_zero": headline_significant,
                "mean_delta_by_group": {k: round(v, 4) for k, v in summary.items()},
                "materiality_threshold": MATERIALITY_THRESHOLD,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
