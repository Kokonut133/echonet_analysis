"""Interpretability figures for the waveform CNN: saliency-overlaid ECG traces,
per-lead saliency share, and Grad-CAM aligned to the average beat.

Runs entirely on the `test` split (never train/val — see project ground rules),
using SmoothGrad input-gradient saliency and Grad-CAM on `model.stage4`.

Outputs:
  figures/interpretability/saliency_examples.png
  figures/interpretability/lead_importance.png
  figures/interpretability/gradcam_beat_average.png
  reports/lead_importance.csv

Usage:
  .venv/bin/python scripts/8_interpretability/saliency_examples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from src.constants import DATASET_SUBDIR, LEAD_NAMES, METADATA_FILENAME, TARGET_LABELS
from src.dataset import ECGDataset, load_split
from src.interpretability import grad_cam_1d, input_gradient_saliency, lead_importance
from src.plotting import TARGET_SHORT_NAMES
from src.models import ECGConvNet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / DATASET_SUBDIR
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "cnn_waveforms.pt"
FIG_DIR = PROJECT_ROOT / "figures" / "interpretability"
REPORTS_DIR = PROJECT_ROOT / "reports"

SAMPLE_RATE_HZ = 250
LEAD_II_IDX = LEAD_NAMES.index("II")

HEADLINE_TARGETS = [
    "shd_moderate_or_greater_flag",
    "lvef_lte_45_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "aortic_stenosis_moderate_or_greater_flag",
]
PANEL_A_TARGETS = [
    "lvef_lte_45_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
]
N_BULK_RECORDS = 300
BATCH_SIZE = 32
RNG_SEED = 42

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})
CMAP = plt.get_cmap("magma")
BEAT_COLOR = CMAP(0.55)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device: torch.device) -> ECGConvNet:
    model = ECGConvNet(n_leads=12, n_labels=len(TARGET_LABELS), n_demo_features=0)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_test_dataset() -> tuple[ECGDataset, pd.DataFrame]:
    metadata = pd.read_csv(DATASET_DIR / METADATA_FILENAME)
    test_data, _ = load_split("test", metadata, DATASET_DIR)
    test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)
    return ECGDataset(test_data), test_meta


@torch.no_grad()
def predict_all(model: ECGConvNet, dataset: ECGDataset, device: torch.device) -> np.ndarray:
    """Forward pass over the whole test set in small batches -> (N, n_labels) probs."""
    n = len(dataset)
    probs = np.zeros((n, len(TARGET_LABELS)), dtype=np.float32)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = torch.stack([
            torch.from_numpy(dataset._read_waveform(i)) for i in range(start, end)
        ]).to(device)
        logits = model(batch)
        probs[start:end] = torch.sigmoid(logits).cpu().numpy()
    return probs


def pick_beat_window(
    lead_ii: np.ndarray, sr: int = SAMPLE_RATE_HZ, target_s: float = 3.2
) -> tuple[int, int]:
    """Pick a window (target_s seconds, clipped to [2.5, 4.0]) that contains >=2 beats."""
    target_s = float(np.clip(target_s, 2.5, 4.0))
    target_len = int(round(target_s * sr))
    n = len(lead_ii)

    peaks, _ = find_peaks(
        np.abs(lead_ii), distance=int(0.3 * sr), height=np.abs(lead_ii).std() * 0.5
    )
    if len(peaks) >= 3:
        idx = min(1, len(peaks) - 2)
        center = (peaks[idx] + peaks[idx + 1]) // 2
    elif len(peaks) >= 1:
        center = peaks[len(peaks) // 2]
    else:
        center = n // 2

    start = max(0, center - target_len // 2)
    end = min(n, start + target_len)
    start = max(0, end - target_len)
    return start, end


SALIENCY_SMOOTH_SIGMA_SAMPLES = 5  # ~20ms sigma (~46ms FWHM) at 250Hz
BAND_ALPHA_MAX = 0.8
BAND_CLIP_FRAC = 0.2  # bottom 20% of the normalized range is fully transparent


def plot_lead_stack(
    ax, waveform_window: np.ndarray, saliency_window: np.ndarray, t_axis: np.ndarray
) -> plt.cm.ScalarMappable:
    """12-lead stack on a single axis: a translucent saliency heat band behind each
    lead's row, with the trace itself drawn as a thin dark-grey line on top."""
    n_leads = waveform_window.shape[0]
    # Scale row spacing to this record's own amplitude so tall QRS spikes never
    # clip into the row above (raw amplitude varies a lot record-to-record).
    offset_step = max(3.0, 1.3 * float(np.abs(waveform_window).max()))
    band_half_height = offset_step * 0.85 / 2

    # Smooth in time so attention reads as contiguous blobs, not per-sample speckle.
    smoothed = gaussian_filter1d(
        saliency_window, sigma=SALIENCY_SMOOTH_SIGMA_SAMPLES, axis=1, mode="nearest"
    )

    vmax = np.percentile(smoothed, 99)
    if vmax <= 0:
        vmax = smoothed.max() if smoothed.max() > 0 else 1e-6
    norm_val = np.clip(smoothed / vmax, 0.0, 1.0)  # (12, W) in [0, 1]

    # Clip the bottom BAND_CLIP_FRAC to fully transparent so baseline stays clean;
    # remap the rest to [0, BAND_ALPHA_MAX].
    alpha = np.clip((norm_val - BAND_CLIP_FRAC) / (1.0 - BAND_CLIP_FRAC), 0.0, 1.0)
    alpha = alpha * BAND_ALPHA_MAX

    for i, lead in enumerate(LEAD_NAMES):
        y_off = (n_leads - 1 - i) * offset_step

        rgba = CMAP(norm_val[i][None, :])  # (1, W, 4)
        rgba[..., 3] = alpha[i][None, :]
        ax.imshow(
            rgba,
            extent=(t_axis[0], t_axis[-1], y_off - band_half_height, y_off + band_half_height),
            origin="lower", aspect="auto", interpolation="bilinear", zorder=1,
        )

        y = waveform_window[i] + y_off
        ax.plot(t_axis, y, color="#2b2b2b", linewidth=1.1, zorder=2)

        ax.text(
            t_axis[0] - 0.025 * (t_axis[-1] - t_axis[0]), y_off, lead,
            ha="right", va="center", fontsize=11, color="#222222", fontweight="bold", zorder=3,
        )

    ax.set_xlim(t_axis[0], t_axis[-1])
    ax.set_ylim(-offset_step * 0.75, n_leads * offset_step - offset_step * 0.15)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    return plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=0, vmax=1))


def make_saliency_examples_figure(model, dataset, test_meta, probs, device) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 14))
    sm = None

    for col, target in enumerate(PANEL_A_TARGETS):
        label_idx = TARGET_LABELS.index(target)
        y_true = test_meta[target].to_numpy(dtype=float)
        y_prob = probs[:, label_idx]

        valid = ~np.isnan(y_true)
        positive = valid & (y_true == 1)
        pos_idx = np.where(positive)[0]
        best_idx = int(pos_idx[np.argmax(y_prob[pos_idx])])

        ax = axes[col]
        waveform = dataset._read_waveform(best_idx)  # (12, 2500) raw amplitude
        wf_tensor = torch.from_numpy(waveform)
        saliency = input_gradient_saliency(
            model, wf_tensor, label_idx, n_smooth=20, noise_frac=0.1, device=device
        )  # (12, 2500)

        start, end = pick_beat_window(waveform[LEAD_II_IDX])
        t0 = start / SAMPLE_RATE_HZ
        t_axis = np.arange(start, end) / SAMPLE_RATE_HZ - t0
        wf_window = waveform[:, start:end]
        sal_window = saliency[:, start:end]

        sm = plot_lead_stack(ax, wf_window, sal_window, t_axis)
        pretty_target = TARGET_SHORT_NAMES.get(target, target.replace("_", " ").replace(" flag", ""))
        ax.set_title(
            f"{pretty_target}\npredicted p={y_prob[best_idx]:.2f} | true positive",
            fontsize=13,
        )

    fig.suptitle(
        "CNN saliency (SmoothGrad): where does the model look for each beat?",
        fontsize=17, y=1.0,
    )
    fig.text(
        0.5, 0.955,
        "Brighter = larger influence on the prediction. The model concentrates on "
        "the QRS complex, most strongly in V1–V2 and lead I.",
        ha="center", fontsize=12, color="#444444",
    )
    fig.tight_layout(rect=(0, 0, 0.93, 0.93))
    cbar_ax = fig.add_axes((0.945, 0.15, 0.014, 0.68))
    fig.colorbar(
        sm, cax=cbar_ax,
        label="Saliency (SmoothGrad, smoothed)\nnormalized per example (0-99th pct.)",
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "saliency_examples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def sample_indices_for_target(test_meta: pd.DataFrame, target: str, n: int) -> np.ndarray:
    valid = test_meta[target].notna().to_numpy()
    valid_idx = np.where(valid)[0]
    rng = np.random.default_rng(RNG_SEED)
    n = min(n, len(valid_idx))
    return rng.choice(valid_idx, size=n, replace=False)


def compute_lead_importance_and_gradcam(
    model, dataset, test_meta, device
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """For each headline target: mean per-lead saliency share, and Grad-CAM windows
    aligned to R-peaks (in lead II) for the beat-average figure."""
    li_rows = []
    gradcam_store: dict[str, dict[str, np.ndarray]] = {}
    half_win = int(0.4 * SAMPLE_RATE_HZ)  # 100 samples = 400 ms

    for target in HEADLINE_TARGETS:
        label_idx = TARGET_LABELS.index(target)
        indices = sample_indices_for_target(test_meta, target, N_BULK_RECORDS)

        lead_shares: list[np.ndarray] = []
        cam_windows: list[np.ndarray] = []
        beat_windows: list[np.ndarray] = []

        for start in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]
            waveforms_np = np.stack([dataset._read_waveform(int(i)) for i in batch_idx])
            wf_tensor = torch.from_numpy(waveforms_np)

            saliency_batch = input_gradient_saliency(
                model, wf_tensor, label_idx, n_smooth=20, noise_frac=0.1, device=device
            )  # (B, 12, 2500)
            for i in range(saliency_batch.shape[0]):
                lead_shares.append(lead_importance(saliency_batch[i]))

            cam_batch = grad_cam_1d(model, wf_tensor, label_idx)  # (B, 2500)

            for i, rec_idx in enumerate(batch_idx):
                lead_ii = waveforms_np[i, LEAD_II_IDX]
                peaks, _ = find_peaks(
                    np.abs(lead_ii), distance=int(0.3 * SAMPLE_RATE_HZ),
                    height=np.abs(lead_ii).std() * 0.5,
                )
                for p in peaks:
                    if p - half_win < 0 or p + half_win >= len(lead_ii):
                        continue
                    cam_windows.append(cam_batch[i, p - half_win:p + half_win + 1])
                    beat_windows.append(lead_ii[p - half_win:p + half_win + 1])

        mean_share = np.mean(np.stack(lead_shares), axis=0)
        for lead, share in zip(LEAD_NAMES, mean_share):
            li_rows.append({"target": target, "lead": lead, "saliency_share": float(share)})

        gradcam_store[target] = {
            "mean_cam": np.mean(np.stack(cam_windows), axis=0),
            "mean_beat": np.mean(np.stack(beat_windows), axis=0),
            "n_beats": len(cam_windows),
        }

    li_df = pd.DataFrame(li_rows)
    return li_df, gradcam_store


def make_lead_importance_figure(li_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), sharex=True)
    for ax, target in zip(axes, HEADLINE_TARGETS):
        sub = li_df[li_df["target"] == target].set_index("lead").reindex(LEAD_NAMES)
        colors = CMAP(0.25 + 0.65 * (sub["saliency_share"] / sub["saliency_share"].max()))
        ax.barh(LEAD_NAMES, sub["saliency_share"], color=colors)
        ax.invert_yaxis()
        pretty_target = TARGET_SHORT_NAMES.get(target, target.replace("_", " ").replace(" flag", ""))
        ax.set_title(pretty_target, fontsize=10.5)
        ax.set_xlabel("Mean saliency share")
        ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)

    fig.suptitle(
        f"Which leads drive each prediction? (mean over {N_BULK_RECORDS} test records/target)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "lead_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_gradcam_beat_figure(gradcam_store: dict[str, dict[str, np.ndarray]]) -> None:
    half_win = int(0.4 * SAMPLE_RATE_HZ)
    t_ms = np.linspace(-400, 400, 2 * half_win + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for ax, target in zip(axes.flat, HEADLINE_TARGETS):
        data = gradcam_store[target]
        mean_cam = data["mean_cam"]
        mean_beat = data["mean_beat"]

        ax.fill_between(t_ms, mean_cam, color=BEAT_COLOR, alpha=0.35, zorder=1)
        ax.plot(t_ms, mean_cam, color=BEAT_COLOR, linewidth=2.0, zorder=2,
                 label="Grad-CAM importance")
        ax.axvline(0, color="#999999", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_ylabel("Grad-CAM importance (norm.)", color=BEAT_COLOR)
        ax.tick_params(axis="y", labelcolor=BEAT_COLOR)

        ax2 = ax.twinx()
        ax2.plot(t_ms, mean_beat, color="#444444", linewidth=1.0, zorder=3,
                  label="Mean lead II morphology")
        ax2.set_ylabel("Lead II amplitude (mV)", color="#444444")
        ax2.tick_params(axis="y", labelcolor="#444444")
        ax2.spines["top"].set_visible(False)

        pretty_target = TARGET_SHORT_NAMES.get(target, target.replace("_", " ").replace(" flag", ""))
        ax.set_title(f"{pretty_target}  (n={data['n_beats']} beats)", fontsize=10.5)
        ax.set_xlabel("Time relative to R-peak (ms)")

    fig.suptitle(
        "Grad-CAM importance aligned to the R-peak, vs. mean beat morphology (lead II)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "gradcam_beat_average.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(only_examples: bool = False) -> None:
    device = resolve_device()
    print(f"Device: {device}")

    model = load_model(device)
    dataset, test_meta = load_test_dataset()
    print(f"Test set: {len(dataset):,} records")

    print("Running full-test-set inference for confident true positives...")
    probs = predict_all(model, dataset, device)

    print("Building saliency_examples.png ...")
    make_saliency_examples_figure(model, dataset, test_meta, probs, device)

    if only_examples:
        print(f"--only-examples: skipped lead-importance/Grad-CAM recompute. Figures in {FIG_DIR}")
        return

    print("Computing lead importance + Grad-CAM beat windows for headline targets "
          f"({N_BULK_RECORDS} records each)...")
    li_df, gradcam_store = compute_lead_importance_and_gradcam(model, dataset, test_meta, device)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    li_df.to_csv(REPORTS_DIR / "lead_importance.csv", index=False)
    print(f"Saved {REPORTS_DIR / 'lead_importance.csv'}")

    print("Building lead_importance.png ...")
    make_lead_importance_figure(li_df)

    print("Building gradcam_beat_average.png ...")
    make_gradcam_beat_figure(gradcam_store)

    print(f"Done. Figures in {FIG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only-examples", action="store_true",
        help="Regenerate saliency_examples.png only; skip the lead-importance/Grad-CAM "
             "recompute so reports/lead_importance.csv and the other two figures are left untouched.",
    )
    args = parser.parse_args()
    main(only_examples=args.only_examples)
