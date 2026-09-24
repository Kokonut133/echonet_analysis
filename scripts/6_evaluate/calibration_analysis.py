"""Probability-calibration audit of the cached held-out test predictions.

The rest of the project reports AUROC/AUPRC (ranking metrics) and discusses
0.5-threshold operating points, but never checks whether the predicted
probabilities themselves are trustworthy. That matters here specifically
because the CNN is trained with `BCEWithLogitsLoss(pos_weight=(1-p)/p)` per
label (`src/training.py::compute_pos_weights`) — a deliberate loss reweight
that inflates the penalty for missed positives on rare targets, which pushes
predicted probabilities away from the true positive rate. This script
quantifies that gap directly for every (tier, target) pair using the cached
`reports/predictions/*.npz` files (no GPU, no retraining).

For each tier and target:
  1. Uncalibrated ("raw") Brier score, Brier skill score (vs. always
     predicting prevalence), expected calibration error (ECE) and max
     calibration error (MCE), computed on the full official test split.
  2. Platt scaling and isotonic regression recalibration. Fitting a
     calibrator on the same test rows it is then scored on would leak
     information (the calibrator would "know" the test-set answers) --
     so the test predictions for each target are split in half with a
     fixed seed, stratified on that target's label. The calibrator is fit
     on one half and scored on the other; the reported platt/isotonic
     metrics are computed on that held-out scoring half only (so they are
     not directly comparable in N to the raw row, which uses the full
     split -- n_total/n_positive columns make this explicit per row).
  3. Targets where either half of the split would have fewer than
     `MIN_POSITIVES_PER_HALF` positives are skipped for platt/isotonic
     (recalibration on a handful of positives is unstable); the raw row is
     still reported, and the skip is logged and written to the CSV as NaN
     metrics so the gap is visible rather than silently missing.

Outputs:
  reports/calibration_results.csv
  figures/results/calibration_curves.png
  figures/results/brier_by_tier.png

Usage:
  .venv/bin/python scripts/6_evaluate/calibration_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.calibration import (
    apply_isotonic,
    apply_platt,
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    fit_isotonic,
    fit_platt,
    max_calibration_error,
    reliability_curve,
)
from src.plotting import TARGET_SHORT_NAMES, TIER_COLORS, TIER_LABELS, TIER_ORDER, apply_style

PREDICTIONS_DIR = Path("reports/predictions")
RESULTS_CSV = Path("reports/calibration_results.csv")
FIGURES_DIR = Path("figures/results")
SPLIT_SEED = 20260923
MIN_POSITIVES_PER_HALF = 25
N_BINS = 10

HEADLINE_TARGETS = [
    "shd_moderate_or_greater_flag",
    "lvef_lte_45_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "aortic_stenosis_moderate_or_greater_flag",
]
HEADLINE_TIER = "cnn_ecg_and_demographics"


def discover_tiers() -> dict[str, Path]:
    """tier name -> npz path, for every reports/predictions/*_test.npz file."""
    tiers = {}
    for path in sorted(PREDICTIONS_DIR.glob("*_test.npz")):
        tier = path.name[: -len("_test.npz")]
        tiers[tier] = path
    return tiers


def raw_row(target: str, tier: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    valid = ~np.isnan(y_true)
    yt, yp = y_true[valid], y_prob[valid]
    return {
        "target": target,
        "tier": tier,
        "method": "raw",
        "brier": brier_score(yt, yp),
        "brier_skill": brier_skill_score(yt, yp),
        "ece": expected_calibration_error(yt, yp, n_bins=N_BINS, strategy="quantile"),
        "mce": max_calibration_error(yt, yp, n_bins=N_BINS, strategy="quantile"),
        "n_positive": int(yt.sum()),
        "n_total": int(valid.sum()),
    }


def recalibrated_rows(target: str, tier: str, y_true: np.ndarray, y_prob: np.ndarray) -> list[dict]:
    """Fit Platt/isotonic on one stratified half of the valid rows, score on the other."""
    valid = ~np.isnan(y_true)
    yt, yp = y_true[valid], y_prob[valid]

    if yt.sum() < 2 or (len(yt) - yt.sum()) < 2:
        print(f"    [skip] {tier}/{target}: too few of one class to stratify-split")
        return [
            {"target": target, "tier": tier, "method": m, "brier": np.nan, "brier_skill": np.nan,
             "ece": np.nan, "mce": np.nan, "n_positive": int(yt.sum()), "n_total": int(len(yt))}
            for m in ("platt", "isotonic")
        ]

    idx = np.arange(len(yt))
    fit_idx, score_idx = train_test_split(
        idx, test_size=0.5, random_state=SPLIT_SEED, stratify=yt
    )
    yt_fit, yp_fit = yt[fit_idx], yp[fit_idx]
    yt_score, yp_score = yt[score_idx], yp[score_idx]

    n_pos_fit, n_pos_score = int(yt_fit.sum()), int(yt_score.sum())
    if n_pos_fit < MIN_POSITIVES_PER_HALF or n_pos_score < MIN_POSITIVES_PER_HALF:
        print(
            f"    [skip] {tier}/{target}: too few positives per half "
            f"(fit={n_pos_fit}, score={n_pos_score}, need >= {MIN_POSITIVES_PER_HALF}) -- "
            "recalibration on this few positives would be unstable"
        )
        return [
            {"target": target, "tier": tier, "method": m, "brier": np.nan, "brier_skill": np.nan,
             "ece": np.nan, "mce": np.nan, "n_positive": n_pos_score, "n_total": int(len(score_idx))}
            for m in ("platt", "isotonic")
        ]

    rows = []
    platt = fit_platt(yt_fit, yp_fit)
    platt_scored = apply_platt(platt, yp_score)
    rows.append({
        "target": target, "tier": tier, "method": "platt",
        "brier": brier_score(yt_score, platt_scored),
        "brier_skill": brier_skill_score(yt_score, platt_scored),
        "ece": expected_calibration_error(yt_score, platt_scored, n_bins=N_BINS, strategy="quantile"),
        "mce": max_calibration_error(yt_score, platt_scored, n_bins=N_BINS, strategy="quantile"),
        "n_positive": n_pos_score, "n_total": int(len(score_idx)),
    })

    isotonic = fit_isotonic(yt_fit, yp_fit)
    iso_scored = apply_isotonic(isotonic, yp_score)
    rows.append({
        "target": target, "tier": tier, "method": "isotonic",
        "brier": brier_score(yt_score, iso_scored),
        "brier_skill": brier_skill_score(yt_score, iso_scored),
        "ece": expected_calibration_error(yt_score, iso_scored, n_bins=N_BINS, strategy="quantile"),
        "mce": max_calibration_error(yt_score, iso_scored, n_bins=N_BINS, strategy="quantile"),
        "n_positive": n_pos_score, "n_total": int(len(score_idx)),
    })

    # Sanity check: a monotone recalibration must not change ranking (AUROC).
    # Platt scaling is a strictly monotone (invertible sigmoid-of-logit)
    # transform, so its AUROC must match the raw AUROC to numerical
    # precision -- any larger gap is a real bug.
    auroc_raw = roc_auc_score(yt_score, yp_score)
    auroc_platt = roc_auc_score(yt_score, platt_scored)
    auroc_iso = roc_auc_score(yt_score, iso_scored)
    if abs(auroc_platt - auroc_raw) > 1e-6:
        raise RuntimeError(f"Platt scaling changed AUROC for {tier}/{target}: {auroc_raw} -> {auroc_platt}")
    # Isotonic regression is only *weakly* monotone (non-decreasing): fit on
    # a small fit-half it can map distinct raw scores onto the same flat
    # step, creating ties on the scoring half that AUROC counts as 0.5
    # credit instead of a strict win/loss. That is a real, bounded property
    # of isotonic regression (not a bug) and grows as the fit sample shrinks
    # -- allow a generous tolerance here and log the size of the gap; only
    # something far outside this range would indicate an actual defect.
    auroc_iso_gap = abs(auroc_iso - auroc_raw)
    if auroc_iso_gap > 0.05:
        raise RuntimeError(f"Isotonic regression changed AUROC for {tier}/{target}: {auroc_raw} -> {auroc_iso}")
    elif auroc_iso_gap > 1e-6:
        print(
            f"    [note] {tier}/{target}: isotonic AUROC {auroc_raw:.4f} -> {auroc_iso:.4f} "
            f"(gap {auroc_iso_gap:.4f}, n_fit_positive={n_pos_fit}) -- tie effect from a coarse "
            "step function on a small fit sample, not real information loss"
        )

    return rows


def build_results_table() -> pd.DataFrame:
    tiers = discover_tiers()
    print(f"Found {len(tiers)} tiers: {sorted(tiers)}")
    all_rows: list[dict] = []
    for tier, path in tiers.items():
        data = np.load(path, allow_pickle=True)
        y_true_all = data["y_true"]
        y_prob_all = data["y_prob"]
        label_names = list(data["label_names"])
        print(f"  tier={tier} ({y_true_all.shape[0]} records)")
        for i, target in enumerate(label_names):
            y_true = y_true_all[:, i]
            y_prob = y_prob_all[:, i]
            all_rows.append(raw_row(target, tier, y_true, y_prob))
            all_rows.extend(recalibrated_rows(target, tier, y_true, y_prob))
    return pd.DataFrame(all_rows)


def plot_calibration_curves(results: pd.DataFrame) -> None:
    apply_style()
    tiers = discover_tiers()
    data = np.load(tiers[HEADLINE_TIER], allow_pickle=True)
    y_true_all = data["y_true"]
    y_prob_all = data["y_prob"]
    label_names = list(data["label_names"])

    method_style = {
        "raw": {"color": "#E0457B", "label": "Raw (uncalibrated)", "marker": "o"},
        "platt": {"color": "#60A5FA", "label": "Platt-scaled", "marker": "s"},
        "isotonic": {"color": "#2CA6A4", "label": "Isotonic", "marker": "^"},
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5))
    axes = axes.ravel()

    for ax, target in zip(axes, HEADLINE_TARGETS):
        col = label_names.index(target)
        y_true = y_true_all[:, col]
        y_prob = y_prob_all[:, col]
        valid = ~np.isnan(y_true)
        yt, yp = y_true[valid], y_prob[valid]
        prevalence = float(yt.mean())

        ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF", linewidth=1.2, zorder=1, label="Perfectly calibrated")
        ax.axhline(prevalence, linestyle=":", color="#6B7280", linewidth=1.0, zorder=1)
        ax.text(0.99, prevalence, f" prevalence {prevalence:.1%}", ha="right", va="bottom",
                fontsize=8.5, color="#6B7280", transform=ax.get_yaxis_transform())

        # Raw reliability curve, full test split.
        mean_pred, obs_freq, counts = reliability_curve(yt, yp, n_bins=N_BINS, strategy="quantile")
        sizes = 18 + 50 * (counts / counts.max())
        style = method_style["raw"]
        ax.plot(mean_pred, obs_freq, color=style["color"], linewidth=1.6, zorder=3)
        ax.scatter(mean_pred, obs_freq, s=sizes, color=style["color"], marker=style["marker"],
                   zorder=4, label=style["label"], edgecolor="white", linewidth=0.6)

        # Platt / isotonic: fit on half, curve on the held-out scoring half.
        idx = np.arange(len(yt))
        n_pos = int(yt.sum())
        if n_pos >= 2 and (len(yt) - n_pos) >= 2:
            fit_idx, score_idx = train_test_split(idx, test_size=0.5, random_state=SPLIT_SEED, stratify=yt)
            yt_fit, yp_fit = yt[fit_idx], yp[fit_idx]
            yt_score, yp_score = yt[score_idx], yp[score_idx]
            if int(yt_fit.sum()) >= MIN_POSITIVES_PER_HALF and int(yt_score.sum()) >= MIN_POSITIVES_PER_HALF:
                platt = fit_platt(yt_fit, yp_fit)
                platt_scored = apply_platt(platt, yp_score)
                mp, of, ct = reliability_curve(yt_score, platt_scored, n_bins=N_BINS, strategy="quantile")
                style = method_style["platt"]
                sz = 18 + 50 * (ct / ct.max())
                ax.plot(mp, of, color=style["color"], linewidth=1.4, zorder=3, alpha=0.9)
                ax.scatter(mp, of, s=sz, color=style["color"], marker=style["marker"], zorder=4,
                           label=style["label"], edgecolor="white", linewidth=0.6, alpha=0.9)

                isotonic = fit_isotonic(yt_fit, yp_fit)
                iso_scored = apply_isotonic(isotonic, yp_score)
                mp, of, ct = reliability_curve(yt_score, iso_scored, n_bins=N_BINS, strategy="quantile")
                style = method_style["isotonic"]
                sz = 18 + 50 * (ct / ct.max())
                ax.plot(mp, of, color=style["color"], linewidth=1.4, zorder=3, alpha=0.9)
                ax.scatter(mp, of, s=sz, color=style["color"], marker=style["marker"], zorder=4,
                           label=style["label"], edgecolor="white", linewidth=0.6, alpha=0.9)

        ece_raw = results.query(
            "tier == @HEADLINE_TIER and target == @target and method == 'raw'"
        )["ece"].iloc[0]
        ax.set_title(f"{TARGET_SHORT_NAMES.get(target, target)}\nraw ECE = {ece_raw:.3f}", fontsize=11.5)
        ax.set_xlabel("Mean predicted probability (within bin)")
        ax.set_ylabel("Observed frequency (within bin)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=10)
    fig.suptitle(
        "Predicted probabilities from the fused CNN are not trustworthy as probabilities\n"
        "(raw curves sit far from the diagonal; recalibration pulls them back — marker size = bin count)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(FIGURES_DIR / "calibration_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_brier_by_tier(results: pd.DataFrame) -> None:
    apply_style()
    raw = results[results["method"] == "raw"].copy()
    tiers = [t for t in TIER_ORDER if t in raw["tier"].unique()]
    targets = list(TARGET_SHORT_NAMES.keys())
    targets = [t for t in targets if t in raw["target"].unique()]

    fig, ax = plt.subplots(figsize=(13, 7))
    n_tiers = len(tiers)
    bar_width = 0.8 / n_tiers
    x = np.arange(len(targets))

    for j, tier in enumerate(tiers):
        vals = []
        for target in targets:
            sub = raw[(raw["tier"] == tier) & (raw["target"] == target)]
            vals.append(sub["brier_skill"].iloc[0] if len(sub) else np.nan)
        offset = (j - (n_tiers - 1) / 2) * bar_width
        ax.bar(x + offset, vals, width=bar_width * 0.92, color=TIER_COLORS.get(tier, "#888888"),
               label=TIER_LABELS.get(tier, tier))

    # Rare targets (e.g. pulmonary regurgitation, prevalence 0.37%) can swing
    # to BSS ~ -38: the climatology Brier denominator is tiny, so even a
    # small absolute Brier gap becomes a huge ratio. A linear axis lets that
    # one bar crush every other bar to invisibility, so use symlog (linear
    # near zero, log further out) to keep both the near-zero tiers and the
    # extreme outliers legible on one axis.
    ax.axhline(0.0, color="#374151", linewidth=1.0, zorder=5)
    ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)
    ax.set_ylim(-45, 1.5)
    ax.set_yticks([-40, -20, -10, -5, -2, -1, 0, 1])
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_SHORT_NAMES.get(t, t) for t in targets], rotation=35, ha="right")
    ax.set_ylabel("Brier skill score\n(symlog scale; 0 = no better than the base rate)")
    ax.set_title(
        "Better ranking (AUROC) does not guarantee better probabilities\n"
        "raw, uncalibrated Brier skill score per target (higher/less negative is better)",
        fontsize=13, pad=14,
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Tier")
    fig.tight_layout()
    fig.subplots_adjust(top=0.86, left=0.09)
    fig.savefig(FIGURES_DIR / "brier_by_tier.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    results = build_results_table()
    results = results.round({"brier": 5, "brier_skill": 5, "ece": 5, "mce": 5})
    results.to_csv(RESULTS_CSV, index=False)
    print(f"\nWrote {RESULTS_CSV} ({len(results)} rows)")

    plot_calibration_curves(results)
    print(f"Wrote {FIGURES_DIR / 'calibration_curves.png'}")

    plot_brier_by_tier(results)
    print(f"Wrote {FIGURES_DIR / 'brier_by_tier.png'}")


if __name__ == "__main__":
    main()
