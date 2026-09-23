"""Builds the headline results figures from reports/final_results.csv and
reports/predictions/*.npz (written by scripts/6_evaluate/evaluate_test_set.py).

Outputs (figures/results/):
  hero_information_ladder.png     test AUROC per target x information tier
  roc_pr_headline_targets.png     ROC + PR curves, 4 headline targets, all tiers
  operating_points.png            specificity/PPV/NPV at fixed sensitivities (CNN)
  subgroup_auroc.png              AUROC by demographic subgroup (CNN vs combined)
  ecg_example_positive_vs_negative.png   one true-positive, one true-negative 12-lead ECG

Also writes reports/subgroup_results.csv and figures/results/README.md.

Usage:
  python scripts/7_figures/make_results_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from src.bootstrap import bootstrap_ci
from src.constants import DATASET_SUBDIR, LEAD_NAMES, METADATA_FILENAME, SAMPLE_RATE_HZ, TARGET_LABELS
from src.plotting import TARGET_SHORT_NAMES, TIER_COLORS, TIER_LABELS, TIER_ORDER, apply_style

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
FIGURES_DIR = PROJECT_ROOT / "figures" / "results"
DATASET_DIR = PROJECT_ROOT / "data" / DATASET_SUBDIR

HEADLINE_TARGETS = [
    "shd_moderate_or_greater_flag",
    "lvef_lte_45_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "aortic_stenosis_moderate_or_greater_flag",
]
SUBGROUP_TARGETS = ["shd_moderate_or_greater_flag", "lvef_lte_45_flag"]
SUBGROUP_TIERS = ["combined", "cnn_raw_waveform", "cnn_ecg_and_demographics"]
SENSITIVITY_LEVELS = [0.80, 0.90, 0.95]

LEAD_LAYOUT = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
RHYTHM_LEAD = "II"
EXAMPLE_TARGET = "lvef_lte_45_flag"

AGE_BAND_EDGES = [(-np.inf, 50, "<50"), (50, 65, "50-65"), (65, 80, "65-80"), (80, np.inf, "≥80")]
AGE_BAND_ORDER = [label for _, _, label in AGE_BAND_EDGES]


# --- data loading ------------------------------------------------------------

def load_predictions(tier: str) -> dict | None:
    path = PREDICTIONS_DIR / f"{tier}_test.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "y_true": data["y_true"],
        "y_prob": data["y_prob"],
        "label_names": [str(n) for n in data["label_names"]],
        "record_index": data["record_index"],
    }


def target_column(preds: dict, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-length (y_true, y_prob, valid_mask) for one target."""
    i = preds["label_names"].index(target)
    y_true = preds["y_true"][:, i]
    y_prob = preds["y_prob"][:, i]
    return y_true, y_prob, ~np.isnan(y_true)


def age_band(age: float) -> str:
    for lo, hi, label in AGE_BAND_EDGES:
        if lo <= age < hi:
            return label
    return "unknown"


# --- figure 1: hero information ladder ---------------------------------------

def make_hero_information_ladder(results: pd.DataFrame) -> None:
    apply_style()
    targets = [t for t in TARGET_LABELS if t in results["target"].unique()]
    tiers = [t for t in TIER_ORDER if t in results["tier"].unique()]
    n_targets = len(targets)

    fig, ax = plt.subplots(figsize=(10, 1.0 + 0.68 * n_targets))
    offsets = np.linspace(-0.28, 0.28, len(tiers))

    for gi, target in enumerate(targets):
        y_base = n_targets - gi
        for oi, tier in enumerate(tiers):
            match = results[(results["target"] == target) & (results["tier"] == tier)]
            if match.empty:
                continue
            r = match.iloc[0]
            y = y_base + offsets[oi]
            xerr = [[max(r["auroc"] - r["auroc_ci_low"], 0)], [max(r["auroc_ci_high"] - r["auroc"], 0)]]
            ax.errorbar(
                r["auroc"], y, xerr=xerr, fmt="o",
                color=TIER_COLORS[tier], ecolor=TIER_COLORS[tier],
                elinewidth=1.5, capsize=2.5, markersize=6.5,
                markeredgecolor="white", markeredgewidth=0.7, zorder=3,
            )

    ax.axvline(0.5, color="#9CA3AF", linewidth=1, linestyle="--", zorder=1)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.3, n_targets + 0.9)
    ax.set_yticks([n_targets - i for i in range(n_targets)])
    ax.set_yticklabels([TARGET_SHORT_NAMES.get(t, t) for t in targets])
    ax.set_xlabel("Test-set AUROC  (0.5 = coin flip, 1.0 = perfect)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#E5E7EB")

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TIER_COLORS[t],
               markeredgecolor="white", markeredgewidth=0.6, markersize=9, label=TIER_LABELS[t])
        for t in tiers
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=len(tiers),
        bbox_to_anchor=(0.5, -0.005), frameon=False, title=None,
    )

    fig.suptitle(
        "How much does the ECG add on top of who the patient is?",
        fontsize=16, fontweight="bold", x=0.02, ha="left", y=0.985,
    )
    fig.text(
        0.02, 0.955,
        "Test-set AUROC for each structural heart disease target, moving from demographics alone, through\n"
        "hand-engineered ECG features, up to a deep model reading the raw 12-lead signal — with and without\n"
        "demographics fused back in. Dots = point estimate, bars = 95% CI.",
        fontsize=10.5, color="#4B5563", ha="left", va="top",
    )

    fig.tight_layout(rect=(0, 0.045, 1, 0.90))
    fig.savefig(FIGURES_DIR / "hero_information_ladder.png")
    plt.close(fig)


# --- figure 2: ROC + PR grid for 4 headline targets ---------------------------

def make_roc_pr_grid(predictions: dict[str, dict]) -> None:
    apply_style()
    tiers = [t for t in TIER_ORDER if t in predictions]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.2))

    for ci, target in enumerate(HEADLINE_TARGETS):
        ax_roc, ax_pr = axes[0, ci], axes[1, ci]
        prevalence = None

        for tier in tiers:
            y_true_full, y_prob_full, valid = target_column(predictions[tier], target)
            y_true, y_prob = y_true_full[valid].astype(int), y_prob_full[valid]
            if prevalence is None:
                prevalence = y_true.mean()

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ax_roc.plot(fpr, tpr, color=TIER_COLORS[tier], linewidth=1.9, label=TIER_LABELS[tier])

            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ax_pr.plot(rec, prec, color=TIER_COLORS[tier], linewidth=1.9)

        ax_roc.plot([0, 1], [0, 1], color="#D1D5DB", linewidth=1, linestyle="--", zorder=0)
        ax_roc.set_title(TARGET_SHORT_NAMES.get(target, target), fontsize=11.5, fontweight="bold", pad=8)
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1.02)
        ax_roc.set_xlabel("False positive rate")

        ax_pr.axhline(prevalence, color="#D1D5DB", linewidth=1.1, linestyle="--", zorder=0)
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1.02)
        ax_pr.set_xlabel("Recall")

        if ci == 0:
            ax_roc.set_ylabel("True positive rate")
            ax_pr.set_ylabel("Precision")

    handles = [Line2D([0], [0], color=TIER_COLORS[t], lw=2.2, label=TIER_LABELS[t]) for t in tiers]
    handles.append(Line2D([0], [0], color="#D1D5DB", lw=1.2, linestyle="--", label="Chance / prevalence"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle(
        "ROC and precision-recall curves for the four headline targets",
        fontsize=15, fontweight="bold", y=1.0,
    )
    fig.text(0.5, 0.955, "Top: ROC (higher/left is better). Bottom: precision-recall (dashed line = prevalence baseline).",
              ha="center", fontsize=10, color="#4B5563")
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    fig.savefig(FIGURES_DIR / "roc_pr_headline_targets.png")
    plt.close(fig)


# --- figure 3: operating points (CNN tier) ------------------------------------

def operating_point_metrics(y_true: np.ndarray, y_prob: np.ndarray, sensitivity: float) -> dict[str, float]:
    pos_probs = y_prob[y_true == 1]
    threshold = np.quantile(pos_probs, 1 - sensitivity)
    pred = y_prob >= threshold

    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))

    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "ppv": tp / (tp + fp) if (tp + fp) else float("nan"),
        "npv": tn / (tn + fn) if (tn + fn) else float("nan"),
    }


def make_operating_points(predictions: dict[str, dict]) -> None:
    apply_style()
    cnn = predictions.get("cnn_raw_waveform")
    if cnn is None:
        return

    metric_colors = {"specificity": "#2563EB", "ppv": "#F59E0B", "npv": "#10B981"}
    metric_labels = {"specificity": "Specificity", "ppv": "PPV (precision)", "npv": "NPV"}
    x = np.arange(len(SENSITIVITY_LEVELS))
    bar_width = 0.24

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)

    for ax, target in zip(axes, HEADLINE_TARGETS):
        y_true_full, y_prob_full, valid = target_column(cnn, target)
        y_true, y_prob = y_true_full[valid].astype(int), y_prob_full[valid]

        for mi, metric in enumerate(["specificity", "ppv", "npv"]):
            values = [operating_point_metrics(y_true, y_prob, s)[metric] for s in SENSITIVITY_LEVELS]
            ax.bar(x + (mi - 1) * bar_width, values, width=bar_width, color=metric_colors[metric])

        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(s * 100)}%" for s in SENSITIVITY_LEVELS])
        ax.set_ylim(0, 1.05)
        ax.set_title(TARGET_SHORT_NAMES.get(target, target), fontsize=11, fontweight="bold")
        ax.set_xlabel("Sensitivity target")
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Rate")

    handles = [plt.Rectangle((0, 0), 1, 1, color=metric_colors[m]) for m in metric_colors]
    fig.legend(handles=handles, labels=list(metric_labels.values()), loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.04), frameon=False)

    shd_true_full, shd_prob_full, shd_valid = target_column(cnn, "shd_moderate_or_greater_flag")
    op90 = operating_point_metrics(
        shd_true_full[shd_valid].astype(int), shd_prob_full[shd_valid], 0.90
    )
    flagged_pct = (1 - op90["specificity"]) * 100

    fig.suptitle("What do we trade off to catch more true cases?", fontsize=15, fontweight="bold", y=1.03)
    fig.text(
        0.5, 0.965,
        f"CNN model on raw ECG. Example: to catch 90% of true SHD cases, "
        f"about {flagged_pct:.0f}% of people without SHD would still be flagged for follow-up.",
        ha="center", fontsize=10.5, color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    fig.savefig(FIGURES_DIR / "operating_points.png")
    plt.close(fig)


# --- figure 4: subgroup AUROC -------------------------------------------------

def compute_subgroup_results(predictions: dict[str, dict], test_meta: pd.DataFrame) -> pd.DataFrame:
    subgroup_series = {
        "sex": test_meta["sex"],
        "age_band": test_meta["age_at_ecg"].apply(age_band),
        "race_ethnicity": test_meta["race_ethnicity"],
        "location_setting": test_meta["location_setting"],
    }

    rows: list[dict] = []
    for target in SUBGROUP_TARGETS:
        for tier in SUBGROUP_TIERS:
            preds = predictions.get(tier)
            if preds is None:
                continue
            y_true_full, y_prob_full, valid = target_column(preds, target)

            for subgroup_type, series in subgroup_series.items():
                for value in series.unique():
                    mask = valid & (series.values == value)
                    n = int(mask.sum())
                    if n < 30:
                        continue
                    y_true, y_prob = y_true_full[mask].astype(int), y_prob_full[mask]
                    if len(np.unique(y_true)) < 2:
                        continue
                    auroc = roc_auc_score(y_true, y_prob)
                    lo, hi = bootstrap_ci(y_true, y_prob, roc_auc_score, n_resamples=1000, seed=42)
                    rows.append({
                        "target": target,
                        "tier": tier,
                        "subgroup_type": subgroup_type,
                        "subgroup_value": value,
                        "auroc": round(auroc, 4),
                        "auroc_ci_low": round(lo, 4),
                        "auroc_ci_high": round(hi, 4),
                        "n": n,
                        "n_positive": int(y_true.sum()),
                    })
    return pd.DataFrame(rows)


def make_subgroup_figure(subgroup_df: pd.DataFrame) -> None:
    if subgroup_df.empty:
        return
    apply_style()
    subgroup_types = ["sex", "age_band", "race_ethnicity", "location_setting"]
    subgroup_titles = {
        "sex": "Sex", "age_band": "Age",
        "race_ethnicity": "Race / ethnicity", "location_setting": "Care setting",
    }

    fig, axes = plt.subplots(len(SUBGROUP_TARGETS), len(subgroup_types), figsize=(15, 6.4))

    for ti, target in enumerate(SUBGROUP_TARGETS):
        for si, subgroup_type in enumerate(subgroup_types):
            ax = axes[ti, si]
            sub = subgroup_df[(subgroup_df["target"] == target) & (subgroup_df["subgroup_type"] == subgroup_type)]

            values = sub["subgroup_value"].unique().tolist()
            if subgroup_type == "age_band":
                values = [v for v in AGE_BAND_ORDER if v in values]
            else:
                values = sorted(values)

            x = np.arange(len(values))
            width = 0.36

            for oi, tier in enumerate(SUBGROUP_TIERS):
                means, err_lo, err_hi = [], [], []
                for v in values:
                    row = sub[(sub["tier"] == tier) & (sub["subgroup_value"] == v)]
                    if row.empty:
                        means.append(0.0); err_lo.append(0.0); err_hi.append(0.0)
                    else:
                        r = row.iloc[0]
                        means.append(r["auroc"])
                        err_lo.append(max(r["auroc"] - r["auroc_ci_low"], 0))
                        err_hi.append(max(r["auroc_ci_high"] - r["auroc"], 0))
                offset = (oi - 0.5) * width
                ax.bar(
                    x + offset, means, width=width, color=TIER_COLORS[tier],
                    yerr=[err_lo, err_hi], capsize=2.5, error_kw={"linewidth": 1},
                )

            ax.set_xticks(x)
            rotate = subgroup_type == "race_ethnicity"
            ax.set_xticklabels(values, rotation=22 if rotate else 0, ha="right" if rotate else "center", fontsize=9.5)
            ax.set_ylim(0.4, 1.0)
            ax.axhline(0.5, color="#D1D5DB", linewidth=0.9, linestyle="--", zorder=0)
            if si == 0:
                ax.set_ylabel(f"{TARGET_SHORT_NAMES.get(target, target)}\nAUROC", fontsize=10)
            if ti == 0:
                ax.set_title(subgroup_titles[subgroup_type], fontsize=11.5, fontweight="bold")

    handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_COLORS[t]) for t in SUBGROUP_TIERS]
    fig.legend(handles=handles, labels=[TIER_LABELS[t] for t in SUBGROUP_TIERS],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.suptitle("Does performance hold up across patient subgroups?", fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, 0.965, "Test-set AUROC with 95% bootstrap CI; groups with fewer than 30 labelled records are omitted.",
              ha="center", fontsize=10, color="#4B5563")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(FIGURES_DIR / "subgroup_auroc.png")
    plt.close(fig)


# --- figure 5: example 12-lead ECGs -------------------------------------------

def _draw_ecg_paper_grid(ax, t_max: float, y_lim: float = 1.2) -> None:
    ax.set_facecolor("#FFF8F8")
    for x in np.arange(0, t_max + 1e-6, 0.2):
        ax.axvline(x, color="#F4C6D0", linewidth=0.4, alpha=0.7, zorder=0)
    for y in np.arange(-y_lim, y_lim + 1e-6, 0.4):
        ax.axhline(y, color="#F4C6D0", linewidth=0.4, alpha=0.7, zorder=0)


def pick_example_records(predictions: dict[str, dict]) -> tuple[int, int, float, float]:
    cnn = predictions["cnn_raw_waveform"]
    y_true, y_prob, valid = target_column(cnn, EXAMPLE_TARGET)

    pos_idx = np.where(valid & (y_true == 1))[0]
    neg_idx = np.where(valid & (y_true == 0))[0]

    tp_idx = int(pos_idx[np.argmax(y_prob[pos_idx])])
    tn_idx = int(neg_idx[np.argmin(y_prob[neg_idx])])

    return tp_idx, tn_idx, float(y_prob[tp_idx]), float(y_prob[tn_idx])


def make_ecg_example_figure(predictions: dict[str, dict]) -> None:
    if "cnn_raw_waveform" not in predictions:
        return
    apply_style()

    waveform_path = DATASET_DIR / "EchoNext_test_waveforms.npy"
    if not waveform_path.exists():
        return
    waveforms = np.load(waveform_path, mmap_mode="r")  # (N, 1, 2500, 12)

    tp_idx, tn_idx, tp_prob, tn_prob = pick_example_records(predictions)
    lead_index = {name: i for i, name in enumerate(LEAD_NAMES)}
    window_samples = int(2.5 * SAMPLE_RATE_HZ)

    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(4, 8, height_ratios=[1, 1, 1, 0.85], hspace=0.6, wspace=0.3)

    cases = [
        (f"True positive — LVEF ≤ 45% (CNN risk {tp_prob:.2f})", tp_idx, 0),
        (f"True negative — LVEF > 45% (CNN risk {tn_prob:.2f})", tn_idx, 4),
    ]

    for case_title, record_idx, col_start in cases:
        record = np.array(waveforms[record_idx, 0, :, :], dtype=np.float32)  # (2500, 12)
        scale = np.max(np.abs(record))
        record_norm = record / scale if scale > 0 else record

        for row_i, lead_row in enumerate(LEAD_LAYOUT):
            for col_j, lead_name in enumerate(lead_row):
                ax = fig.add_subplot(gs[row_i, col_start + col_j])
                signal = record_norm[:window_samples, lead_index[lead_name]]
                t = np.arange(len(signal)) / SAMPLE_RATE_HZ
                _draw_ecg_paper_grid(ax, t[-1])
                ax.plot(t, signal, color="#111827", linewidth=0.9, zorder=2)
                ax.set_title(lead_name, fontsize=9, fontweight="bold", pad=2, loc="left", color="#374151")
                ax.set_xlim(0, t[-1])
                ax.set_ylim(-1.2, 1.2)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        ax_rhythm = fig.add_subplot(gs[3, col_start:col_start + 4])
        rhythm_signal = record_norm[:, lead_index[RHYTHM_LEAD]]
        t_full = np.arange(len(rhythm_signal)) / SAMPLE_RATE_HZ
        _draw_ecg_paper_grid(ax_rhythm, t_full[-1])
        ax_rhythm.plot(t_full, rhythm_signal, color="#111827", linewidth=0.9, zorder=2)
        ax_rhythm.set_xlim(0, t_full[-1])
        ax_rhythm.set_ylim(-1.2, 1.2)
        ax_rhythm.set_yticks([])
        ax_rhythm.set_xlabel("seconds", fontsize=9)
        ax_rhythm.set_title("Lead II — rhythm strip (10s)", fontsize=9, fontweight="bold", loc="left", pad=2, color="#374151")
        for spine in ax_rhythm.spines.values():
            spine.set_visible(False)

        x_center = 0.28 if col_start == 0 else 0.76
        fig.text(x_center, 0.965, case_title, fontsize=12, fontweight="bold", ha="center", color="#111827")

    fig.suptitle("What a true positive and a true negative look like on the raw ECG", fontsize=15, fontweight="bold", y=1.04)
    fig.savefig(FIGURES_DIR / "ecg_example_positive_vs_negative.png", bbox_inches="tight")
    plt.close(fig)


# --- README + entrypoint ------------------------------------------------------

_FIGURE_DESCRIPTIONS = [
    ("hero_information_ladder.png",
     "Headline figure: test-set AUROC per target across the five information tiers "
     "(demographics -> ECG metadata -> waveform features -> combined -> CNN on raw ECG), with 95% CIs."),
    ("roc_pr_headline_targets.png",
     "ROC (top) and precision-recall (bottom) curves for the four headline targets, all tiers overlaid; "
     "dashed line in the PR panels marks the prevalence baseline."),
    ("operating_points.png",
     "For the CNN tier on the four headline targets: specificity, PPV and NPV at fixed sensitivity "
     "thresholds of 80/90/95%, i.e. the trade-off from choosing a more sensitive screening cutoff."),
    ("subgroup_auroc.png",
     "AUROC with 95% CI by sex, age band, race/ethnicity and care setting for the combined and CNN tiers, "
     "on the SHD and LVEF <=45% targets; underlying numbers in reports/subgroup_results.csv."),
    ("ecg_example_positive_vs_negative.png",
     "One true-positive and one true-negative 12-lead ECG (LVEF <=45% target) in the standard clinical "
     "3x4 layout plus a lead-II rhythm strip, with the CNN's predicted risk for each."),
]


def write_figures_readme() -> None:
    lines = ["# Results figures", ""]
    for filename, description in _FIGURE_DESCRIPTIONS:
        lines.append(f"- **{filename}** — {description}")
    (FIGURES_DIR / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(REPORTS_DIR / "final_results.csv")
    predictions = {tier: load_predictions(tier) for tier in TIER_ORDER}
    predictions = {t: p for t, p in predictions.items() if p is not None}

    print("Building hero_information_ladder.png ...")
    make_hero_information_ladder(results)

    print("Building roc_pr_headline_targets.png ...")
    make_roc_pr_grid(predictions)

    print("Building operating_points.png ...")
    make_operating_points(predictions)

    print("Building subgroup_auroc.png ...")
    metadata = pd.read_csv(DATASET_DIR / METADATA_FILENAME)
    test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)
    subgroup_df = compute_subgroup_results(predictions, test_meta)
    subgroup_df.to_csv(REPORTS_DIR / "subgroup_results.csv", index=False)
    make_subgroup_figure(subgroup_df)

    print("Building ecg_example_positive_vs_negative.png ...")
    make_ecg_example_figure(predictions)

    write_figures_readme()
    print(f"\nDone. Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
