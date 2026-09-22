"""CNN lead-ablation sensitivity analysis.

The trained ECGConvNet (checkpoints/cnn_waveforms.pt) was fit on all 12 leads.
This script does NOT retrain it — it zeroes lead channels at inference time and
re-scores the val split, which is a sensitivity analysis ("how much does the
model lean on this lead's signal"), not a "trained on fewer leads" result.

in:  checkpoints/cnn_waveforms.pt
     data/<dataset>/EchoNext_val_waveforms.npy   (memory-mapped, val split only)
     data/<dataset>/<metadata>.csv
out: reports/lead_ablation_results.csv           (ablation, target, auroc, delta_auroc)
     figures/ablation/lead_ablation.png
     figures/ablation/reduced_lead_sets.png

Usage:
  .venv/bin/python scripts/9_ablation/lead_ablation.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm

from src.ablation import (
    CHEST_COLOR,
    CHEST_LEADS,
    LIMB_COLOR,
    LIMB_LEADS,
    keep_indices_to_zero,
    lead_category,
    zero_leads,
)
from src.constants import DATASET_SUBDIR, LEAD_NAMES, METADATA_FILENAME, N_LEADS, TARGET_LABELS
from src.metrics import compute_per_label_metrics
from src.models import ECGConvNet

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
FOUR_LEAD_SET = ["I", "II", "V1", "V5"]

REDUCED_SET_COLORS = {
    "baseline": "#555555",
    "limb_only": LIMB_COLOR,
    "chest_only": CHEST_COLOR,
    "lead_II_only": "#8172B2",
    "four_lead_I_II_V1_V5": "#55A868",
}
REDUCED_SET_LABELS = {
    "baseline": "Full 12-lead",
    "limb_only": "Limb only (6)",
    "chest_only": "Chest only (6)",
    "lead_II_only": "Lead II only",
    "four_lead_I_II_V1_V5": "4-lead (I, II, V1, V5)",
}

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
    metadata_path: Path
    checkpoint_path: Path
    results_path: Path
    figures_dir: Path


@dataclass(frozen=True)
class Config:
    paths: Paths
    batch_size: int = 64
    split: str = "val"


def build_ablation_configs() -> dict[str, list[int]]:
    """ablation name -> list of lead indices to zero out (channels dim=1)."""
    name_to_idx = {name: i for i, name in enumerate(LEAD_NAMES)}
    configs: dict[str, list[int]] = {"baseline": []}

    for lead in LEAD_NAMES:
        configs[f"drop_{lead}"] = [name_to_idx[lead]]

    chest_idx = [name_to_idx[l] for l in CHEST_LEADS]
    limb_idx = [name_to_idx[l] for l in LIMB_LEADS]
    configs["limb_only"] = chest_idx  # zero chest leads, keep limb leads
    configs["chest_only"] = limb_idx  # zero limb leads, keep chest leads
    configs["lead_II_only"] = keep_indices_to_zero(N_LEADS, [name_to_idx["II"]])
    configs["four_lead_I_II_V1_V5"] = keep_indices_to_zero(
        N_LEADS, [name_to_idx[l] for l in FOUR_LEAD_SET]
    )
    return configs


@torch.no_grad()
def run_ablations(
    model: ECGConvNet,
    waveforms: np.ndarray,
    ablation_configs: dict[str, list[int]],
    device: torch.device,
    batch_size: int,
    n_labels: int,
) -> dict[str, np.ndarray]:
    """One pass over `waveforms` (N,1,2500,12 mmap); every ablated variant is
    produced from the same batch read (no re-reading the file per ablation)."""
    model.eval()
    n = waveforms.shape[0]
    probs = {name: np.empty((n, n_labels), dtype=np.float32) for name in ablation_configs}

    n_batches = (n + batch_size - 1) // batch_size
    for b in tqdm(range(n_batches), desc="  ablation batches", unit="batch"):
        start, end = b * batch_size, min((b + 1) * batch_size, n)
        batch = np.asarray(waveforms[start:end, 0, :, :], dtype=np.float32)  # (b, 2500, 12)
        x = torch.from_numpy(batch).permute(0, 2, 1).contiguous().to(device)  # (b, 12, 2500)

        for name, drop_idx in ablation_configs.items():
            x_ablated = zero_leads(x, drop_idx) if drop_idx else x
            logits = model(x_ablated)
            probs[name][start:end] = torch.sigmoid(logits).cpu().numpy()

    return probs


def evaluate_ablations(
    probs: dict[str, np.ndarray], y_true: np.ndarray, label_names: list[str]
) -> pd.DataFrame:
    baseline_auroc = (
        compute_per_label_metrics(y_true, probs["baseline"], label_names)
        .set_index("label")["auroc"]
    )

    rows = []
    for name, p in probs.items():
        metrics = compute_per_label_metrics(y_true, p, label_names).set_index("label")["auroc"]
        for target in label_names:
            auroc = metrics[target]
            delta = 0.0 if name == "baseline" else round(auroc - baseline_auroc[target], 4)
            rows.append({"ablation": name, "target": target, "auroc": auroc, "delta_auroc": delta})
    return pd.DataFrame(rows)


def make_lead_ablation_figure(results: pd.DataFrame, figures_dir: Path) -> None:
    drop_rows = results[results["ablation"].str.startswith("drop_")].copy()
    drop_rows["lead"] = drop_rows["ablation"].str.removeprefix("drop_")

    mean_delta = (
        drop_rows.groupby("lead")["delta_auroc"].mean().reindex(LEAD_NAMES).sort_values()
    )
    colors = [CHEST_COLOR if lead_category(l) == "chest" else LIMB_COLOR for l in mean_delta.index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), width_ratios=[1.1, 1])

    ax = axes[0]
    ax.barh(mean_delta.index, mean_delta.values, color=colors, edgecolor="none")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Mean ΔAUROC vs. full 12-lead baseline\n(across all 12 targets)")
    ax.set_title("Impact of zeroing one lead", loc="left", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)

    vmin, vmax = float(mean_delta.min()), float(mean_delta.max())
    span = vmax - vmin if vmax > vmin else abs(vmin) or 1.0
    ax.set_xlim(vmin - 0.28 * span, max(0.0, vmax) + 0.16 * span)
    offset = 0.035 * span
    for y, v in enumerate(mean_delta.values):
        ax.text(
            v - offset if v < 0 else v + offset, y, f"{v:+.3f}",
            va="center", ha="right" if v < 0 else "left", fontsize=8, color="#333333",
        )
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color=LIMB_COLOR, label="Limb lead"), Patch(color=CHEST_COLOR, label="Chest lead")],
        loc="upper left", frameon=False, fontsize=9,
    )

    ax2 = axes[1]
    lead_order = list(mean_delta.index)
    heat = (
        drop_rows[drop_rows["target"].isin(HEADLINE_TARGETS)]
        .pivot(index="lead", columns="target", values="delta_auroc")
        .reindex(index=lead_order, columns=HEADLINE_TARGETS)
    )
    vmax = float(np.abs(heat.values).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax2.imshow(heat.values, aspect="auto", cmap="RdBu_r", norm=norm)
    ax2.set_xticks(range(len(HEADLINE_TARGETS)))
    ax2.set_xticklabels([HEADLINE_LABELS[t] for t in HEADLINE_TARGETS], rotation=30, ha="right", fontsize=9)
    ax2.set_yticks(range(len(lead_order)))
    ax2.set_yticklabels(lead_order, fontsize=10)
    ax2.set_title("ΔAUROC by lead × headline target", loc="left", fontsize=12, fontweight="bold")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.values[i, j]
            ax2.text(
                j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(val) > vmax * 0.6 else "#222222",
            )
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("ΔAUROC")

    fig.suptitle(
        "CNN lead-ablation sensitivity (leave-one-lead-out, zeroed at inference; model not retrained)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out_path = figures_dir / "lead_ablation.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def make_reduced_lead_sets_figure(results: pd.DataFrame, figures_dir: Path) -> None:
    configs = ["baseline", "limb_only", "chest_only", "lead_II_only", "four_lead_I_II_V1_V5"]
    subset = results[
        results["ablation"].isin(configs) & results["target"].isin(HEADLINE_TARGETS)
    ]
    pivot = subset.pivot(index="target", columns="ablation", values="auroc").reindex(
        index=HEADLINE_TARGETS, columns=configs
    )

    n_targets = len(HEADLINE_TARGETS)
    n_configs = len(configs)
    x = np.arange(n_targets)
    width = 0.8 / n_configs

    y_min = min(0.45, float(pivot.values.min()) - 0.05)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, cfg in enumerate(configs):
        offset = (i - (n_configs - 1) / 2) * width
        bars = ax.bar(
            x + offset, pivot[cfg].values, width=width,
            color=REDUCED_SET_COLORS[cfg], label=REDUCED_SET_LABELS[cfg],
        )
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([HEADLINE_LABELS[t] for t in HEADLINE_TARGETS], fontsize=10)
    ax.set_ylabel("AUROC (val)")
    ax.set_ylim(y_min, 1.0)
    ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_title(
        "Reduced lead sets: AUROC on headline targets\n"
        "(missing leads zeroed at inference; model trained on all 12 leads)",
        loc="left", fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=1)
    fig.tight_layout()
    out_path = figures_dir / "reduced_lead_sets.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def run_lead_ablation(config: Config) -> pd.DataFrame:
    metadata = pd.read_csv(config.paths.metadata_path)
    split_meta = metadata[metadata["split"] == config.split].reset_index(drop=True)
    y_true = split_meta[TARGET_LABELS].values.astype(float)

    waveform_path = config.paths.dataset_dir / f"EchoNext_{config.split}_waveforms.npy"
    waveforms = np.load(waveform_path, mmap_mode="r")
    if len(waveforms) != len(split_meta):
        raise ValueError(
            f"Row count mismatch: metadata={len(split_meta)}, waveforms={len(waveforms)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGConvNet(n_leads=N_LEADS, n_labels=len(TARGET_LABELS), n_demo_features=0)
    state_dict = torch.load(config.paths.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    ablation_configs = build_ablation_configs()
    print(
        f"Running {len(ablation_configs)} ablations over {len(waveforms):,} "
        f"'{config.split}' records on {device}..."
    )
    probs = run_ablations(
        model, waveforms, ablation_configs, device, config.batch_size, len(TARGET_LABELS)
    )

    results = evaluate_ablations(probs, y_true, TARGET_LABELS)

    config.paths.results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.paths.results_path, index=False)
    print(f"Saved {len(results)} rows -> {config.paths.results_path}")

    config.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    make_lead_ablation_figure(results, config.paths.figures_dir)
    make_reduced_lead_sets_figure(results, config.paths.figures_dir)

    return results


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "data" / DATASET_SUBDIR
    config = Config(
        paths=Paths(
            dataset_dir=dataset_dir,
            metadata_path=dataset_dir / METADATA_FILENAME,
            checkpoint_path=project_root / "checkpoints" / "cnn_waveforms.pt",
            results_path=project_root / "reports" / "lead_ablation_results.csv",
            figures_dir=project_root / "figures" / "ablation",
        ),
    )
    run_lead_ablation(config)


if __name__ == "__main__":
    main()
