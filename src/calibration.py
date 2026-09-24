"""Probability-calibration diagnostics and recalibration helpers.

The CNN and sklearn tiers in this project are trained/selected to maximize
ranking metrics (AUROC/AUPRC) under heavy class-imbalance correction
(`BCEWithLogitsLoss(pos_weight=(1-p)/p)` for the CNN — see
`src/training.py::compute_pos_weights`). That objective deliberately inflates
the loss on positives for rare labels, which pushes predicted probabilities
away from the true positive rate. Good ranking does not imply that
`P(y=1 | score=s) ~= s`; this module exists to measure that gap directly and,
where useful, fix it post hoc with monotone recalibration (Platt / isotonic).

All functions are pure (no I/O, no plotting) and operate on 1D arrays for a
single target at a time. Every function drops NaN labels before computing
anything, since `y_true` in this project's cached predictions uses NaN for
records where a given target was not adjudicated.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _drop_nan(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to 1D float arrays and drop entries with NaN labels."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(f"y_true and y_prob must have the same length, got {y_true.shape[0]} vs {y_prob.shape[0]}")
    valid = ~np.isnan(y_true)
    return y_true[valid], y_prob[valid]


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probability and binary outcome.

    Lower is better; 0 is perfect. NaN labels are dropped first.
    """
    y_true, y_prob = _drop_nan(y_true, y_prob)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier skill relative to always predicting the sample prevalence.

    BSS = 1 - Brier(model) / Brier(climatology), where climatology always
    predicts the observed positive rate. BSS > 0 means the model's
    probabilities are more informative than the base rate alone; BSS <= 0
    means they are no better (or worse) than just reporting the prevalence,
    which can happen even for a model with excellent AUROC if its outputs
    are badly miscalibrated. NaN labels are dropped first.
    """
    y_true, y_prob = _drop_nan(y_true, y_prob)
    if y_true.size == 0:
        return float("nan")
    prevalence = float(np.mean(y_true))
    brier_model = float(np.mean((y_prob - y_true) ** 2))
    brier_climatology = float(np.mean((prevalence - y_true) ** 2))
    if brier_climatology < _EPS:
        return float("nan")
    return 1.0 - brier_model / brier_climatology


def _bin_edges(y_prob: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "quantile":
        # Quantile (equal-count) bins. With prevalences from 0.8% to 52% in this
        # project, equal-WIDTH bins put almost every record (positive or
        # negative) into the same one or two low-probability bins for the rare
        # targets, leaving the rest of the bins nearly empty (n~0) and their
        # calibration error undefined/unstable. Equal-count bins guarantee
        # every bin has enough records to estimate an observed frequency.
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(y_prob, quantiles))
        if edges.size < 2:
            edges = np.array([y_prob.min(), y_prob.max() + _EPS])
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges
    raise ValueError(f"unknown strategy: {strategy!r}")


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (mean predicted probability, observed positive frequency, count).

    Bins with zero records are omitted from the output. Default `strategy`
    is "quantile" — see `_bin_edges` for why (equal-width bins are mostly
    empty at the prevalences seen in this project).
    """
    y_true, y_prob = _drop_nan(y_true, y_prob)
    if y_true.size == 0:
        return np.array([]), np.array([]), np.array([])
    edges = _bin_edges(y_prob, n_bins, strategy)
    bin_idx = np.digitize(y_prob, edges[1:-1], right=False)

    mean_pred, obs_freq, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred.append(float(y_prob[mask].mean()))
        obs_freq.append(float(y_true[mask].mean()))
        counts.append(count)
    return np.array(mean_pred), np.array(obs_freq), np.array(counts)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> float:
    """Count-weighted mean absolute gap between predicted and observed frequency.

    ECE = sum_b (n_b / N) * |mean_pred_b - obs_freq_b|. Default `strategy` is
    "quantile" (equal-count bins): with target prevalences from 0.8% to 52%
    in this project, equal-width bins leave most bins nearly empty for rare
    targets (almost every score falls in the lowest bin or two), making the
    per-bin observed frequency noisy or undefined. Equal-count bins keep
    every bin populated so the estimate is stable.
    """
    y_true, y_prob = _drop_nan(y_true, y_prob)
    if y_true.size == 0:
        return float("nan")
    mean_pred, obs_freq, counts = reliability_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    if counts.size == 0:
        return float("nan")
    weights = counts / counts.sum()
    return float(np.sum(weights * np.abs(mean_pred - obs_freq)))


def max_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> float:
    """Worst-case (max over bins) absolute gap between predicted and observed frequency."""
    y_true, y_prob = _drop_nan(y_true, y_prob)
    if y_true.size == 0:
        return float("nan")
    mean_pred, obs_freq, counts = reliability_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    if counts.size == 0:
        return float("nan")
    return float(np.max(np.abs(mean_pred - obs_freq)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def fit_platt(y_true: np.ndarray, y_prob: np.ndarray) -> LogisticRegression:
    """Fit Platt scaling: 1D logistic regression of y_true on logit(y_prob).

    Returns the fitted `LogisticRegression`; use `apply_platt` to score new
    probabilities with it. NaN labels are dropped before fitting.
    """
    y_true, y_prob = _drop_nan(y_true, y_prob)
    x = _logit(y_prob).reshape(-1, 1)
    model = LogisticRegression(C=1e10, solver="lbfgs")
    model.fit(x, y_true)
    return model


def apply_platt(model: LogisticRegression, y_prob: np.ndarray) -> np.ndarray:
    """Score raw probabilities with a Platt-scaling model fit by `fit_platt`."""
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    x = _logit(y_prob).reshape(-1, 1)
    return model.predict_proba(x)[:, 1]


def fit_isotonic(y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression (monotone, non-parametric) from y_prob to y_true.

    NaN labels are dropped before fitting.
    """
    y_true, y_prob = _drop_nan(y_true, y_prob)
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(y_prob, y_true)
    return model


def apply_isotonic(model: IsotonicRegression, y_prob: np.ndarray) -> np.ndarray:
    """Score raw probabilities with an isotonic model fit by `fit_isotonic`."""
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    return model.predict(y_prob)
