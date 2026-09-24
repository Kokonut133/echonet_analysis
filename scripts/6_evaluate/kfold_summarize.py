"""Summarize reports/kfold_results.csv into reports/kfold_summary.md and
figures/results/kfold_variance.png.

Reads the per-fold CNN cross-validation results written by
scripts/6_evaluate/kfold_cnn.py alongside the single-run headline numbers
already in reports/final_results.csv, and answers the question this whole
exercise exists for: is the fused-vs-waveform-only AUROC gap (fused CNN with
demographics vs. CNN on raw waveform only) bigger or smaller than the
fold-to-fold standard deviation on the official test split?

Usage:
  python scripts/6_evaluate/kfold_summarize.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting import TARGET_SHORT_NAMES, apply_style

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KFOLD_RESULTS_PATH = PROJECT_ROOT / "reports" / "kfold_results.csv"
FINAL_RESULTS_PATH = PROJECT_ROOT / "reports" / "final_results.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "kfold_summary.md"
FIGURE_PATH = PROJECT_ROOT / "figures" / "results" / "kfold_variance.png"

FUSED_TIER = "cnn_ecg_and_demographics"
WAVEFORM_ONLY_TIER = "cnn_raw_waveform"

HEADLINE_TARGETS = [
    "shd_moderate_or_greater_flag",
    "lvef_lte_45_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "aortic_stenosis_moderate_or_greater_flag",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    kfold = pd.read_csv(KFOLD_RESULTS_PATH)
    final = pd.read_csv(FINAL_RESULTS_PATH)
    return kfold, final


def per_target_fold_stats(kfold: pd.DataFrame) -> pd.DataFrame:
    """Per-target mean/std of test-split AUROC across the individual
    (non-ensemble) folds."""
    test_rows = kfold[(kfold["split"] == "test") & (kfold["fold"] != "ensemble")].copy()
    test_rows["fold"] = test_rows["fold"].astype(str)
    stats = (
        test_rows.groupby("target")["auroc"]
        .agg(fold_mean="mean", fold_std="std", n_folds="count")
        .reset_index()
    )
    return stats


def ensemble_row(kfold: pd.DataFrame, target: str) -> float | None:
    match = kfold[
        (kfold["fold"] == "ensemble") & (kfold["split"] == "test") & (kfold["target"] == target)
    ]
    if match.empty:
        return None
    return float(match.iloc[0]["auroc"])


def single_run_auroc(final: pd.DataFrame, target: str, tier: str) -> float | None:
    match = final[(final["target"] == target) & (final["tier"] == tier)]
    if match.empty:
        return None
    return float(match.iloc[0]["auroc"])


def write_summary(kfold: pd.DataFrame, final: pd.DataFrame) -> None:
    fold_stats = per_target_fold_stats(kfold).set_index("target")
    all_targets = [t for t in final["target"].unique() if t in fold_stats.index]
    # Order: headline targets first (in their canonical order), then the rest.
    ordered_targets = [t for t in HEADLINE_TARGETS if t in all_targets]
    ordered_targets += [t for t in all_targets if t not in ordered_targets]

    lines = [
        "# 5-fold CV variance estimate: fused CNN vs. waveform-only CNN",
        "",
        "Every confidence interval elsewhere in this project comes from "
        "bootstrap-resampling the (fixed, single-run) test set — none of them "
        "capture training/seed variance. This report fills that gap: the "
        "fused architecture (`ECGConvNet` with waveform + demographics) is "
        "retrained 5 times on 5 stratified folds of the pooled official "
        "train+val rows (77,101 records, stratified on "
        "`shd_moderate_or_greater_flag`), early-stopped on held-out-fold mean "
        "AUROC, and each fold's model is scored on the untouched official "
        "test split. The demographic encoder is refit per fold on the "
        "in-fold training rows only.",
        "",
        "## Per-target: fold-to-fold mean ± std vs. the single-run number",
        "",
        "| target | single-run fused AUROC | 5-fold mean ± std (test) | "
        "n folds | 5-fold ensemble | single-run waveform-only AUROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for target in ordered_targets:
        short = TARGET_SHORT_NAMES.get(target, target)
        single_fused = single_run_auroc(final, target, FUSED_TIER)
        single_wave = single_run_auroc(final, target, WAVEFORM_ONLY_TIER)
        row = fold_stats.loc[target]
        fold_mean, fold_std, n_folds = row["fold_mean"], row["fold_std"], int(row["n_folds"])
        ens = ensemble_row(kfold, target)

        def fmt(v):
            return f"{v:.4f}" if v is not None and not pd.isna(v) else "—"

        lines.append(
            f"| {short} | {fmt(single_fused)} | "
            f"{fmt(fold_mean)} ± {fmt(fold_std)} | {n_folds} | "
            f"{fmt(ens)} | {fmt(single_wave)} |"
        )

    lines += ["", "## The deliverable: does the fused-vs-waveform-only gap survive fold noise?", ""]

    for target in HEADLINE_TARGETS:
        if target not in fold_stats.index:
            continue
        short = TARGET_SHORT_NAMES.get(target, target)
        single_fused = single_run_auroc(final, target, FUSED_TIER)
        single_wave = single_run_auroc(final, target, WAVEFORM_ONLY_TIER)
        fold_std = fold_stats.loc[target, "fold_mean"], fold_stats.loc[target, "fold_std"]
        fold_mean, fold_std = fold_stats.loc[target, "fold_mean"], fold_stats.loc[target, "fold_std"]
        ens = ensemble_row(kfold, target)

        if single_fused is None or single_wave is None or pd.isna(fold_std):
            lines.append(f"- **{short}**: insufficient data to compare.")
            continue

        gap = single_fused - single_wave
        verdict = "LARGER than" if abs(gap) > fold_std else "SMALLER than (i.e. within noise of)"
        lines.append(
            f"- **{short}**: single-run gap (fused {single_fused:.4f} − "
            f"waveform-only {single_wave:.4f}) = {gap:+.4f}. Fold-to-fold std "
            f"of the fused model's test AUROC = {fold_std:.4f} "
            f"(fold mean {fold_mean:.4f}). The gap is **{verdict}** one "
            f"fold-to-fold standard deviation"
            + (f"; the 5-fold ensemble reaches {ens:.4f}." if ens is not None else ".")
        )

    lines += [
        "",
        "## Notes",
        "",
        "- `split` in `reports/kfold_results.csv` is `heldout_fold` (in-CV "
        "held-out fold) or `test` (official test split, scored per fold "
        "model). `fold` is `0`-`4` for individual folds or `ensemble` for "
        "the mean of the 5 models' predicted probabilities on the test "
        "split.",
        "- The official test split was never used for fold selection, "
        "early stopping, or the demographic encoder fit — only for this "
        "final per-fold scoring.",
        "",
    ]

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"Saved {SUMMARY_PATH}")


def make_figure(kfold: pd.DataFrame, final: pd.DataFrame) -> None:
    apply_style()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    targets = [t for t in HEADLINE_TARGETS if t in kfold["target"].unique()]
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 5), sharey=False)
    if len(targets) == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        fold_rows = kfold[
            (kfold["target"] == target) & (kfold["split"] == "test") & (kfold["fold"] != "ensemble")
        ].copy()
        fold_rows["fold"] = fold_rows["fold"].astype(int)
        fold_rows = fold_rows.sort_values("fold")
        aurocs = fold_rows["auroc"].to_numpy()

        x_dots = np.full(len(aurocs), 1.0)
        ax.scatter(x_dots, aurocs, s=55, color="#7C3AED", zorder=3, label="fold (test)")

        mean, std = aurocs.mean(), aurocs.std(ddof=1) if len(aurocs) > 1 else 0.0
        ax.errorbar(
            [1.15], [mean], yerr=[std], fmt="o", color="#111827", capsize=5, elinewidth=1.6,
            markersize=7, zorder=4, label="mean ± std",
        )

        single_fused = single_run_auroc(final, target, FUSED_TIER)
        if single_fused is not None:
            ax.axhline(single_fused, color="#E0457B", linestyle="--", linewidth=1.6, zorder=2,
                       label="single run (fused)")

        single_wave = single_run_auroc(final, target, WAVEFORM_ONLY_TIER)
        if single_wave is not None:
            ax.axhline(single_wave, color="#60A5FA", linestyle=":", linewidth=1.6, zorder=2,
                       label="single run (waveform only)")

        ens = ensemble_row(kfold, target)
        if ens is not None:
            ax.scatter([1.3], [ens], marker="D", s=60, color="#2CA6A4", zorder=4, label="5-fold ensemble")

        ax.set_xlim(0.8, 1.5)
        ax.set_xticks([])
        ax.set_title(TARGET_SHORT_NAMES.get(target, target), fontsize=11)
        ax.set_ylabel("Test AUROC" if ax is axes[0] else "")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=9)
    fig.suptitle("Fold-to-fold variance of the fused CNN on the official test split", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGURE_PATH}")


def main() -> None:
    kfold, final = load_data()
    write_summary(kfold, final)
    make_figure(kfold, final)


if __name__ == "__main__":
    main()
